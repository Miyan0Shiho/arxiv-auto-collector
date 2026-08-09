# RING: Retrieval-Internalized Generation for Continual Large-Scale Knowledge Injection

**Authors**: Shicheng Xu, Liang Pang, Liyi Chen, Zihao Wei, Jingcheng Deng, Yan Gao, Yi Wu, Yao Hu, Huawei Shen, Xueqi Cheng

**Published**: 2026-08-03 03:00:43

**PDF URL**: [https://arxiv.org/pdf/2608.01630v1](https://arxiv.org/pdf/2608.01630v1)

## Abstract
Retrieval-augmented generation (RAG) improves factuality but adds latency and engineering overhead at serving time. We propose RING (Retrieval-Internalized Generation), a holistic paradigm spanning both architecture and training that injects large-scale external knowledge into a \textit{Mixture-of-Memory Experts} and learns parametric search over this internal memory via reinforcement learning, removing the external retriever entirely. Training proceeds in three stages: continued pre-training injects new corpora into a Knowledge Expert via our novel \textit{Dual Causal Attention}; supervised fine-tuning teaches a ``search-then-answer'' pattern; and reinforcement learning with hierarchical rewards optimizes the routing-and-search policy over the parametric memory. Unlike prior parametric injection methods that pair internal memory with a fixed or rule-based retriever, RING {learns} its retrieval policy directly from task signals. We further frame RING theoretically as a search-free approximation to the classical RAG objective. To evaluate large-scale injection of genuinely {new} knowledge without test-time leakage, we further construct News-2025, a benchmark built from news strictly post-dating the base LLM's pretraining cutoff. RING matches or surpasses both search-based RAG and parametric injection baselines in accuracy and efficiency.

## Full Text


<!-- PDF content starts -->

RING:Retrieval-InternalizedGeneration for Continual Large-Scale
Knowledge Injection
Shicheng Xu1,2Liang Pang1*Liyi Chen3Zihao Wei1,2Jingcheng Deng1,2
Yan Gao3Yi Wu3Yao Hu3Huawei Shen1Xueqi Cheng1
1State Key Laboratory of AI Safety, Institute of Computing Technology, CAS
2University of Chinese Academy of Sciences3Xiaohongshu Inc.
xushicheng21s@ict.ac.cn pangliang@ict.ac.cn
Abstract
Retrieval-augmented generation (RAG) im-
proves factuality but adds latency and engi-
neering overhead at serving time. We propose
RING(Retrieval-Internalized Generation), a
holistic paradigm spanning both architecture
and training that injects large-scale external
knowledge into aMixture-of-Memory Experts
and learnsparametric searchover this internal
memory via reinforcement learning, removing
the external retriever entirely. Training pro-
ceeds in three stages: continued pre-training in-
jects new corpora into a Knowledge Expert via
our novelDual Causal Attention; supervised
fine-tuning teaches a “search-then-answer” pat-
tern; and reinforcement learning with hierarchi-
cal rewards optimizes the routing-and-search
policy over the parametric memory. Unlike
prior parametric injection methods that pair
internal memory with a fixed or rule-based
retriever, RING learns its retrieval policy di-
rectly from task signals. We further frame
RING theoretically as a search-free approx-
imation to the classical RAG objective. To
evaluate large-scale injection of genuinely new
knowledge without test-time leakage, we fur-
ther constructNews-2025, a benchmark built
from news strictly post-dating the base LLM’s
pretraining cutoff. RING matches or surpasses
both search-based RAG and parametric injec-
tion baselines in accuracy and efficiency.
1 Introduction
Large language models (LLMs) store factual
knowledge in their parameters (Hu et al., 2024) but
struggle to assimilate new knowledge reliably, lead-
ing to hallucinations and outdated outputs (Wang
et al., 2023). Retrieval-Augmented Generation
(RAG) addresses this by bridging LLMs with ex-
ternal knowledge sources (Lewis et al., 2020; Xu
et al., 2024a): relevant documents are retrieved and
*Corresponding authors.appended to the LLM’s context, grounding gener-
ation in external evidence (Gao et al., 2023b; Wei
et al., 2026; Deng et al., 2025a).
Despite RAG’s success, its reliance on external
retrievers introduces significant inference latency
and engineering complexity, and even generative
retrieval (Li et al., 2025) still depends on an ex-
ternal corpus for content mapping. Research has
thus pivoted toward purely parametric injection,
but existing approaches face severe bottlenecks:
Knowledge Editing (Zhang et al., 2024) and Con-
tinual Learning (Wang et al., 2024a) struggle with
unstructured updates (Deng et al., 2025b) and catas-
trophic forgetting (Nguyen et al., 2019); Modular
Injection methods (KBLAM (Wang et al., 2025),
SR-KI (Yu et al., 2026)) require memory linear in
corpus size; and Parametric RAG (Su et al., 2025)
still depends on an external search engine to iden-
tify relevant documents at inference. A scalable,
fully parametric solution that matches RAG’s per-
formance remains elusive.
To address this gap, we propose to internalize
external search within the LLM parameters and
call thisholistic paradigm spanning both architec-
ture and trainingRetrieval-Internalized Gener-
ation (RING), in direct contrast to RAG: where
RAG augments the LLM with an external retriever,
RING internalizes the retriever inside the LLM as
a learned policy via the joint co-design of architec-
ture and training. The central thesis of RING is that
the bottleneck of parametric knowledge injection
is not how to store knowledge in weights, but how
to search those weights at inference time. Prior
parametric methods all pair internal memory with a
fixed or rule-based retrieval mechanism—attention
similarity (KBLAM, SR-KI), kNN imitation (MLP
Memory), hierarchical pruning (AtlasKV), or unsu-
pervised routing (LAG)—and consequently under-
perform RAG at scale. RING is the first to learn the
parametric retrieval policy end-to-end via reinforce-
ment learning: hierarchical rewards over routing
1
arXiv:2608.01630v1  [cs.CL]  3 Aug 2026

Question ResponseIndexText 
DatabaseEmbedding
Model Embedding
ModelText Database Index
LLM
Text 
DatabaseQuestionLLM
Know BasicLLM
Know Basic Response QuestionLLM
Know BasicResponse(a) Conventional RAG
Online Service Offline Indexing
(b) Our RING
Stage1: Memorization Stage 2, 3: SFT & RL for Search from ParametersOffline Injection Online ServiceVery Heavy and Long Pipeline!
Do Not Need Search!
Figure 1: Overview of RING versus Conventional RAG. Unlike traditional methods that depend on external
embedding models and vector databases (top), RING (bottom) unifies indexing and retrieval within the LLM
parameters via a Mixture-of-Experts architecture. This design shifts retrieval computation to the training stage,
achieving high-efficiency search-free inference for RAG.
accuracy, search relevance, and answer correctness
train the model to jointly select which expert holds
the relevant knowledge and which sub-region in-
side that expert to attend to. This transforms para-
metric retrieval from a static lookup into a trainable
behavior, eliminating the external index entirely
while yielding cost profiles that scale with model
FLOPs rather than index size.
Architecture.RING augments a dense LLM
with a sparseMixture-of-Memory Experts: aBasic
Expertpreserves the original LLM weights to miti-
gate catastrophic forgetting; aKnowledge Expert
parametrically memorizes and indexes the new cor-
pus; and a learnableRouterselects between them at
each token, integrating newly injected knowledge
with the LLM’s base reasoning.
Three-Stage Training.RING operationalizes
this architecture with a CPT–SFT–RL pipeline. (i)
Continued Pre-training injects the new corpus into
the Knowledge Expert via our novel Dual Causal
Attention (DCA), which models new passages bidi-
rectionally while preserving the LLM’s causal gen-
eration (Zhang et al., 2025b). (ii) Supervised Fine-
Tuning teaches asearch-then-answerresponse pat-
tern within a single generative pass, with no ex-
ternal retriever. (iii) Reinforcement Learning is
the core of our pipeline and the locus of RING’s
main innovation. Drawing on evidence that RL
improves an LLM’s ability to sample useful trajec-
tories from its own parameter space (Chen et al.,
2026), we cast parametric retrieval as a policy op-
timization problem and use hierarchical rewards
to simultaneously train (a) the Router to select the
right expert, (b) the Knowledge Expert to surface
the relevant memory fragment, and (c) the genera-
tor to answer grounded in that fragment. The resultis a fully learned parametric retriever—unlike all
prior parametric injection methods whose retrieval
mechanism is hand-designed and frozen.
Theoretical Grounding.We further formulate
RING as a discrete latent variable model and prove
that the Knowledge Expert acts as a differentiable
Key–Value memory index, while our CPT–SFT–
RL pipeline performs variational inference that
maximizes the Evidence Lower Bound, theoreti-
cally guaranteeing RING’s capacity to approximate
the classical RAG objective.
News-2025 Benchmark.To rigorously evaluate
large-scale injection of genuinely new knowledge
without test-time leakage, we construct News-2025,
a bilingual benchmark of ∼170k Chinese / English
news documents (341M tokens) strictly post-dating
the base LLM’s pretraining cutoff, so that correct
answers cannot come from pre-existing knowledge
or associative reasoning. Our contributions are:
•We propose Retrieval-Internalized Generation
(RING), a holistic paradigm spanning both
architecture and training that eliminates exter-
nal search by internalizing both indexing and
retrieval into LLM parameters.
•We construct News-2025, a bilingual post-
cutoff benchmark for contamination-free eval-
uation of knowledge injection at scale.
•Empirically, RING outperforms parametric
injection baselines on News-2025, matches
strong RAG pipelines in accuracy, and runs
3×–19×faster than RAG at inference.
2 Related Work
Retrieval-Augmented GenerationRAG com-
bines a parametric LLM with an external retriever
to ground generation in retrieved text (Lewis et al.,
2

2020; Izacard and Grave, 2021), but introduces la-
tency and engineering overhead at serving time.
Generative retrieval (Tay et al., 2022; Li et al.,
2025; Bevilacqua et al., 2022) replaces vector
search with a generative process predicting doc-
ument IDs, yet still depends on an external corpus
to resolve IDs into content. RING instead parame-
terizes retrieval entirely within the LLM, removing
all runtime dependence on external search.
Parameterized Knowledge InjectionParame-
terized knowledge injection internalizes external
information into model weights. Continual learn-
ing (Shi et al., 2025) suffers from catastrophic for-
getting on unstructured updates; knowledge editing
(e.g., MEMIT (Meng et al., 2022)) is limited to
atomic facts; modular injection (KBLAM (Wang
et al., 2025), SR-KI (Yu et al., 2026)) has memory
linear in corpus size; Parametric RAG (Su et al.,
2025) parameterizes documents via adapters but
still queries an external retriever at inference; Temp-
LoRA (Wang et al., 2024b) produces ephemeral
adapters for long context, not permanent memo-
rization. RING unifies memory, retrieval, and gen-
eration inside the LLM, achieving fully parametric,
search-free augmentation at scale.
3 RING
3.1 Model Architecture
OverviewAs shown in Figure 2, RING replaces
a dense decoder with a sparse MoE layer stack
composed of abasic expert Ebaseand aknowledge
expert Eknw. This is initialized by duplicating the
MLP weights; thebasic expertretains the original
weights, while theknowledge expertis initialized
by the original MLP and used for subsequent new
knowledge memorization. ARouter rϕproduces
a sparse gate gt∈0,1 at each layer and position,
deciding which expert computes the token’s trans-
formation. Ebasepreserves general abilities and
minimizes forgetting, while Eknwparameterizes
the external corpus and acts as an implicit index.
Sparse RoutingAt layer ℓand position twith
hidden state h(ℓ)
t, theRouterfirst produces expert
probabilitiesp(ℓ)
tand makes a top-1 selectiong(ℓ)
t:
p(ℓ)
t= softmax 
W(ℓ)
rh(ℓ)
t
, g(ℓ)
t= arg max
k∈Kp(ℓ)
t,k.
Let the expert set be K={base,knw} , and denote
byE(ℓ)
k(·)the transformation of expert kat layer ℓ.
With hard (top -1) routing (we omit LayerNorm and
Self 
AttentionMLPDecoder Layer
Self AttentionBasic ExpertDecoder Layer ( MoE)
Router
(Sparse)Convert
(a) Dense Model (b) RING: Sparse Model SiLU×
Memory
UpMemory
 Down
Search
GateKnowledge 
ExpertFigure 2: Model Architecture in our RING.
residual connections to simplify the expression):
˜h(ℓ+1)
t =X
k∈K1h
g(ℓ)
t=ki
E(ℓ)
k
h(ℓ)
t
.(1)
Parameter SeparationWe further separate the
parameters ofknowledge expert Eknwinto memory
Em
knwand search Es
knw. This ensures that subse-
quent training of search capabilities will not affect
the original memory and allows for incremental
addition of new memory data. As shown in Fig-
ure 2, Em
knwconsists of memory up and down pro-
jection layers and Es
knwis a gate projection fol-
lowed by a Sigmoid Linear Unit (SiLU) function.
This design is inspired by existing observations
that up and down projection layers in Transformers
are important key-value memories responsible for
organizing the massive amounts of knowledge in
LLMs (Geva et al., 2021). In our design,Memory
Downis used to store knowledge, whileMemory
Upis a parametric index of the stored knowledge.
Search Gateworks in conjunction with SiLU to
reweight indexes for different queries in order to
acquire corresponding knowledge in text genera-
tion. This parameter separation avoids interference
between different training stages in RING.
3.2 Continued Pre-training for Memorization
Data PreprocessGiven a large-scale unstruc-
tured knowledge base Dthat requires LLMs to
memorize (just like the corpus for retrieval in con-
ventional RAG). We perform data preprocessing to
construct training data for CPT in RING. During
this process, we segment each unstructured long
document into multiple non-overlapping fragments,
each consisting of three consecutive sentences. We
combine the document title and the fragment con-
tent into a JSON-formatted text as:
u={"title": "title of the document",
"content": "content of the fragment"}
3

Token Sequence (left to right)
Token Sequence (left to right)Token Sequence (left to right)Token Sequence (left to right) Token Sequence (left to right)
Token Sequence (left to right)
Standard Causal Attention1
2  Dual Causal Attention1
4  Dual Causal AttentionFigure 3: Dual Causal Attention. The white areas are
visible tokens, the black areas are masked tokens, and
the shaded areas represent invalid tokens.
which is the unit for memorization and search in
the CPT, SFT, and RL stages. To ensure the se-
mantic integrity of the long documents, we also
add the full text of the document to the training
set in CPT. However, this full-text data is only for
memory assistance and will not be generated dur-
ing the downstream search process. After the above
process on the entire corpus, we obtain the CPT
training setD CPT.
Dual Causal AttentionCPT injects new knowl-
edge into the Knowledge Expert. While bidirec-
tional attention is known to be more effective for
memorization (Zhang et al., 2025b), the LLM’s
autoregressive causal mask conflicts with it. We
proposeDual Causal Attention (DCA): each CPT
example is run with three attention masks M(α),
α∈ {1,1
2,1
4}, that partition the sequence into a
target block Bα(causally masked internally) and
a complement Cα(visible as backward evidence);
the resulting per-mask NLL losses are combined as
LDCA=P
αλαL(α)(Figure 3). α= 1 recovers
standard causal pre-training; α∈ {1
2,1
4}introduce
bidirectional supervisionwithoutleaking future tar-
gets inside Bα. At inference, only M(1)is used,
so generation remains strictly autoregressive. For-
mal mask definitions, the per-block NLL, and the
combined loss are given in Appendix A.5.
TrainingCPT freezes the entire model except the
Knowledge Expert’sMemory DownandMemory
Upprojections, and forces the Router to dispatch
every token to the Knowledge Expert. Optimiz-
ingLDCA onDCPT concentrates injected knowl-
edge into the storing (Memory Down) and indexing
(Memory Up) projections while preserving the base
model’s capabilities.
3.3 SFT for Search-then-Answer
Search-then-Answer ParadigmIn the SFT
stage, we introduce a novelSearch-then-Answerparadigm designed to integrate knowledge retrieval
seamlessly within the LLM’s generation process.
When the LLM receives a query, it first identi-
fies whether new, externally injected knowledge
is required to answer the question. If so, it per-
forms a parametric retrieval from the knowledge
expert Eknwto generate the top-1 relevant frag-
ment. This fragment is enclosed within special
tokens <retrieval> and</retrieval> . The re-
trieved fragment is typically short (with about 200
tokens) to avoid excessive additional computational
overhead during generation. Once the relevant frag-
ment is identified, the LLM transitions to the next
phase where it generates the final answer within
the<answer> and</answer> tags. This design
follows the natural generative flow.
Instruction Fine-tuning Data and LossTo
bootstrap theSearch-then-Answerbehavior, we
useQwen3-235B to synthesize {(qi, ai, ri)}triples
from documents in DCPT, where riis a reference
span justifying ai, and align each rito a memory
unitu⋆
i∈ D CPT via exact string match (triples
without a match are discarded). Each accepted
triple is formatted as a search-then-answer target
string with u⋆
iin the <retrieval> block and ai
in the <answer> block. We mix in no-retrieval in-
structions from a generic corpus to teach “retrieve
only when needed.” The model is trained with stan-
dard cross-entropy on the target string. During SFT
we freeze the CPTMemory Down/Upand only up-
date the Router and Search Gate, ensuring retrieval
behavior is learned without overwriting memorized
content. Full data recipe, target template, and loss
formulation are in Appendix B.
3.4 Reinforcement Learning for Search from
the Parametric Knowledge
RL stage in our RING aims to enable the LLM
to learn a generalizable ability to retrieve informa-
tion from its parametric knowledge base. Drawing
on existing research that RL enhances the LLM’s
ability to sample more effectively from its param-
eter space (Chen et al., 2026; Deng et al., 2026),
we find this process closely mirrors traditional re-
trieval, where the goal is to identify the most rel-
evant document to answer a query from a library.
Hence, we believe RL holds great potential in en-
abling the LLM to perform generative retrieval di-
rectly from its parametric knowledge base. After
the SFT phase, which provides the LLM with a ba-
sicSearch-then-Answerparadigm, we apply RL to
4

teach the LLM how to search for the most relevant
memory units from the parametric knowledge base.
The RL algorithm we use is GSPO (Zheng et al.,
2025). The reward structure is designed to guide
the LLM’s generative retrieval, rank and answer
process in three ways:
1. Format Reward.( Rformat ) When facing
instruction that is relevant to the external knowl-
edge, the LLM must adhere to theSearch-then-
Answerformat. The response should first in-
clude the retrieval block enclosed within the tokens
<retrieval> and</retrieval> , followed by the
answer within <answer> and</answer> . If the
format is correct, the reward is 1; otherwise, it is 0.
2. Search Reward.( Rsearch ) This reward
aims to guide LLM to find the most match-
ing units from the external corpus DCPT mem-
orized in the CPT stage, and output the mem-
ory units in <retrieval> ...</retrieval> , we
introduce a dense reward mechanism to achieve
this. This mechanism extracts the title and
content of the memory unit within <retrieval>
and </retrieval> of the LLM’s generated se-
quences, and then calculates a series of rewards
to guide the LLM’s learning of both retrieval and
re-ranking in its generation. The design principle
of these rewards is to encourage LLM to generate
units that are highly consistent with correct mem-
ory units. (1) Learning to Retrieve Reward ( R1
dense):
This reward measures the normalized longest com-
mon substring (LCS) length between the retrieved
memory unit’s title and the title of the correct
memory unit. A higher LCS indicates better re-
trieval performance, and a higher reward is given.
(2) Learning to Rerank Reward ( R2
dense): This re-
ward measures the normalized LCS between the
retrieved memory unit’s content and the content
of the correct memory unit. This reward encour-
ages the model to select more relevant content con-
ditioned on the already fuzzy-matched titles. (3)
Exact Match Reward ( R3
dense): This reward checks
if the reference passage (ref) is exactly present in
the generated content . If the reference is present,
the reward is 1; otherwise, the reward is 0. This re-
ward helps the model to retrieve exact top-1 ground
truth. The combined search reward is:
Rsearch =λ aR1
dense+λ bR2
dense+λ cR3
dense,(2)
λa,λb, andλ care hyperparameters in Section C.4.
We use exact substring matching rather than se-
mantic matching because we observe with seman-
tic matching, the model might learn to generatesemantically similar, but incorrect, responses that
still satisfy the reward conditions. Literal matching
is straightforward and prevents such issues.
3. Answer Reward.( Ranswer ) The LLM’s an-
swer, enclosed in the <answer> block, is com-
pared to the correct reference answer. We use
Qwen3-30B-A3B to judge whether the LLM’s gen-
erated answer matches the correct answer. If the
answer is correct, the reward is 1; if it is incorrect,
the reward is 0.
The total reward is the sum of the rewards:
Rtotal=λ formatRformat+λanswerRanswer+λsearchRsearch,
where λformat ,λanswer , andλsearch are hyperparame-
ters in Section C.4. By RL, LLM learns to retrieve
and generate from the parametric knowledge base.
4 Theoretical Analysis
We formulate RING as a principled discrete latent
variable model that approximates the classical RAG
objective. This theoretically justifies our MoE ar-
chitecture and the training pipeline. Detailed proofs
are provided in Appendix A.
Latent Evidence and RING Approximation
Letxbe the query, ythe answer, and U={u z}N
z=1
the universe of injected memory units. Conven-
tional RAG marginalizes over a retrieved latent
variablez:
p(y|x) =NX
z=1p(z|x,U)p θ(y|x, u z).(3)
Conventional RAG relies on a non-parametric p(z|
x,U) via external vector search. RING parameter-
izes this retrieval fully internally as:
pΘ(y, z|x) =p ϕ(z|x)|{z}
Router & Gate·pθgen(y|x, u z; Θmem)| {z }
Knowledge Expert,
where Θmemstores U. We prove in Theorem A.4
that RING’s hypothesis space contains explicit top-
KRAG, implying that parametric retrieval can
theoretically match external search expressivity
(Corollary A.5).
Knowledge Expert as Key-Value MemoryWe
justify our specific Knowledge Expert architecture
by showing it is structurally isomorphic to a dif-
ferentiable Key-Value retrieval mechanism. As
derived in Proposition A.2, the computation of the
5

Knowledge ExpertE knw(h)is:
Eknw(h) =X
iSiLU(⟨g i, h⟩)⟨k i, h⟩| {z }
Attention Scoreα i(h)vi,(4)
where ki(rows of Wup) act as Keys, vi(cols of
Wdown) as Values, and gias a query-dependent
Search Gate. Lemma A.3 suggests that with
learned gating, this mechanism approaches Max-
imum Inner Product Search, allowing precise re-
trieval from millions of parametric slots.
Training as Variational InferenceWe aim to
maximize the marginal log-likelihood of generating
the correct answer, logp Θ(y|x) . Since marginal-
izing over all latent memory units zis intractable,
we maximize the Evidence Lower Bound (ELBO).
Introducing a variational posterior q(z|x, y) over
the evidence (derivation in Eq. (16)):
logp Θ(y|x)≥E q(z|x,y)
logpΘ(y, z|x)
q(z|x, y)
=E q(z|x,y) [logp θgen(y|x, u z)]| {z }
Generation (Reconstruction)
−KL(q(z|x, y)||p ϕ(z|x))| {z }
Retrieval (Alignment).
This decomposition provides a unified theoretical
view of our three-stage pipeline:
CPT (Evidence Maximization): The generation
term logp(y|x, u z)fundamentally relies on the
model’s ability to comprehend and reconstruct the
memory unit uz. By minimizing the DCA loss
(Prop. A.7), CPT injects UintoΘmem, ensuring
that conditioned on a retrieved z, the likelihood of
generating relevant tokens is maximized. We prove
in Prop. A.7 that DCA provides a tighter bound on
conditional entropy than standard causal attention,
theoretically guaranteeing more efficient storage.
SFT (Posterior Alignment): In SFT, we are given
the ground truth evidence u⋆. This is equivalent to
setting the variational posterior to a delta function
q(z|x, y) =δ(z=z⋆). The ELBO maximization
then simplifies to standard supervised learning:
max 
logp θgen(y|x, u⋆)| {z }
Answer Generation+ logp ϕ(z⋆|x)|{z}
Retrieval Supervision
.
(5)
This forces the router policy pϕto align with the
"ideal" posterior defined by the dataset.
RL (Generalizing the Policy): When ground
truth evidence is unavailable or multiple valid snip-
pets exist, the fixed posterior δ(z=z⋆)is subopti-
mal. RL optimizes the policy pϕ(z|x) to maximizethe expected reward. As shown in Lemma A.8, our
dense search rewards (e.g., LCS) serve as a proxy
for the unobserved latent posterior, guiding the re-
trieval policy to maximize the ELBO’s expectation
term even without explicit supervision.
5 Experiments
5.1 Datasets
News-2025is our constructed bilingual benchmark
of 170k Chinese news documents (341.93M to-
kens) from June–July 2025, translated to English
with DeepSeek-R1-0528 . Since the base LLM
(Qwen3) predates this period, correct answers are
hard to be obtained from pre-existing parametric
knowledge or associative reasoning, isolating each
method’s true injection-and-utilization capability.
We construct News-2025 to probe the two capa-
bilities RING targets: the injection of large-scale
new knowledge and the utilization of this injected
parametric knowledge. The corpus is partitioned
into subsets for CPT, SFT, and RL (Appendix B).
The CPT set includes the whole documents col-
lection, which serves as the external knowledge
baseDCPT that needs to be injected. The SFT set
is designed to teach the model the “Search-then-
Answer” paradigm. The RL set is used for the train-
ing of “search from parametric knowledge”. As for
the test set, we sample a held-out document set and
useGPT-5 to construct 10k QA pairs whose source
documents donotappear in SFT or RL, jointly
probing the injection-and-utilization capability.
5.2 Baselines
We group baselines into three categories (Ta-
ble 1; full descriptions in Appendix C.3):(i) Con-
ventional RAG (Top-1)with SOTA embedders
BGE-M3 (Chen et al., 2024), OpenAI-Emb-3L,
Gemini-Embedding (Lee et al., 2025), and Qwen3-
Embedding (Zhang et al., 2025a);(ii) Stronger
RAG Pipelines(all using Qwen3-8B-Emb): Top- k
retrieval with k∈ {3,5,10} , cross-encoder rerank-
ing on top-10 to top-3 (BGE-Reranker-v2-m3 (Li
et al., 2023) and Qwen3-Reranker-4B (Zhang et al.,
2025a)), HyDE (Gao et al., 2023a), and base-LLM
Query Rewrite;(iii) Parametric Knowledge In-
jection: Full Tuning, LoRA Tuning, modular meth-
ods KBLAM (Wang et al., 2025) and SR-KI (Yu
et al., 2026), and three recent retriever-imitating
methods closest to RING in spirit—LAG (Flesh-
man and Van Durme, 2025) (per-document LoRA +
data-free routing), MLP Memory (Wei et al., 2025)
6

Table 1: Performance on News-2025 (English & Chinese), reported as Accuracy (%) and Time To First Token (TTFT,
seconds). “Top- k” denotes the number of documents prepended to the LLM. “+Rerank” denotes retrieve-top-10
with the best embedding followed by a cross-encoder reranker selecting top-3.Boldmarks the global best in each
column; underline marks the best within the KNOWLEDGEINJECTIONWITHOUTEXTERNALSEARCHgroup
(where RING belongs). When the same entry is best both globally and within the group, only bold is used.
Method SearchBased on Qwen3-8B Based on Qwen3-14B
English Chinese English Chinese
Acc (↑) TTFT (↓) Acc (↑) TTFT (↓) Acc (↑) TTFT (↓) Acc (↑) TTFT (↓)
Original LLM None 6.75 0.30 7.15 0.35 7.26 0.57 7.73 0.48
KNOWLEDGEINJECTIONWITHEXTERNALSEARCH(RAG, TOP-1)
BGE-M3 28.57 1.32 28.94 1.41 26.54 2.20 27.88 2.32
Gemini-Emb-Exp 32.76 3.46 34.12 3.52 30.72 4.25 32.17 4.39
RAG OpenAI-Emb-3L 29.99 3.95 30.97 3.88 27.96 4.80 29.35 4.54
Qwen3-0.6B-Emb 30.84 1.29 32.17 1.45 30.41 2.21 31.84 2.19
Qwen3-8B-Emb 33.83 1.81 35.28 1.73 31.80 2.60 33.26 2.55
STRONGERRAG PIPELINES
RAG (Top-3) Qwen3-8B-Emb 36.42 2.85 37.91 2.78 34.05 3.55 35.62 3.42
RAG (Top-5) Qwen3-8B-Emb 37.18 4.05 38.65 3.92 35.21 4.85 36.84 4.68
RAG (Top-10) Qwen3-8B-Emb 36.85 6.45 38.34 6.22 35.94 7.62 37.61 7.35
+Rerank (BGE-v2-m3) Qwen3-8B-Emb 38.73 3.45 40.21 3.32 37.42 4.18 39.18 4.05
+Rerank (Qwen3-Rerank) Qwen3-8B-Emb39.514.0541.053.88 38.26 4.8539.824.68
+HyDE Qwen3-8B-Emb 36.94 5.15 38.42 4.92 34.85 6.75 36.51 6.48
+Query Rewrite Qwen3-8B-Emb 35.62 3.85 37.18 3.68 33.74 4.72 35.42 4.55
KNOWLEDGEINJECTIONWITHOUTEXTERNALSEARCH
Lora-Tuning None 21.370.3422.120.3222.70 0.65 26.37 0.60
Full-Tuning None 26.92 0.39 27.06 0.34 28.780.6031.830.58
KBLAM None 29.13 2.25 28.32 2.15 33.21 3.65 33.48 2.98
SR-KI None 31.85 2.03 29.22 2.09 36.65 3.20 33.50 2.85
LAG None 24.51 0.78 23.86 0.82 28.34 1.18 27.62 1.12
MLP Memory None 28.45 0.55 27.34 0.52 32.18 0.88 30.95 0.85
AtlasKV None 32.46 1.85 30.18 1.72 37.62 2.65 34.85 2.48
RING (ours) None 35.08 0.40 32.36 0.49 41.19 0.78 37.92 0.75
(kNN-distribution imitation), and AtlasKV (Huang
et al., 2025) (compressed KV-cache injection). Un-
like all parametric baselines whose retrieval mech-
anism is hand-designed and frozen, RING (i) keeps
a separate Basic Expert to prevent forgetting, (ii)
injects via bidirectional DCA, and (iii) learns re-
trieval end-to-end via RL with dense rewards.
5.3 Main Results
The specific experimental settings can be found in
Section C. The main results are in Table 1.
vs. Parametric Knowledge Injection.RING dom-
inates every parametric injection baseline on both
languages and both backbones. With Qwen3-8B,
RING reaches 35.08%/32.36% (EN/ZH), exceed-
ing the strongest prior method SR-KI by 3.23 and
3.14 points. Among the three recently proposed
retriever-imitating baselines, AtlasKV is the closest
competitor since KV-level injection retains finer-
grained semantics than logit-level interpolation;
however, its KG2KV pipeline loses information
when news passages cannot be cleanly decom-posed into ⟨h, r, t⟩ triples, and hierarchical KV
pruning still incurs non-trivial attention overhead
(TTFT 1.85s vs. 0.40s). MLP Memory is fast
(TTFT 0.55s, comparable to Full-Tuning) but its
probability-interpolation interface cannot resolve
fine-grained factual queries with the same precision
as KV injection. LAG is the weakest of the three:
data-free per-token adapter routing was designed
for libraries on the order of 103adapters, and false
selections compound as the library scales to our
170k-document corpus. The advantage widens on
Qwen3-14B, where RING exceeds every paramet-
ric baseline by at least 3.5 points.
vs. External-Search RAG.RING occupies the
accuracy–latency Pareto frontier. On the 8B back-
bone, the strongest pipeline (Qwen3-Reranker on
top-10, output top-3) surpasses RING by 4.43
EN/8.69 ZH points but at 10× TTFT (4.05s
vs. 0.40s); on 14B the trade-off flips—RING
(41.19/37.92) beats this same pipeline on EN and
trails by under 2 points on ZH while remaining 6×
faster. Lighter Top- kretrieval plateaus near 37% on
7

20 30 40 50 60 70 80 90 100
Injected Data Size (k documents)101520253035Accuracy on Original Knowledge (%)
RING (Ours)
KBLAM
Full Fine-Tuning
LoRA Fine-Tuning
(a) Sequential Injection
20 30 40 50 60 70 80 90 100
Injected Data Size (k documents)101520253035Accuracy on Original Knowledge (%)
RING (Ours)
KBLAM
Full Fine-Tuning
LoRA Fine-Tuning
 (b) Mixed Training
Figure 4: Comparison of knowledge retention strate-
gies.(a)Sequential injection means the model is trained
strictly on the sequence of new knowledge.(b)Mixed
training means introduces a replay buffer of original
data during training to mitigate forgetting.
8B with diminishing returns ( k=10 pushes TTFT
to 6.45s); HyDE and Query Rewrite add LLM for-
wards without statistically meaningful gains over
Top-3, since the base LLM cannot expand queries
for post-cutoff news.
Latency.RING’s TTFT matches Full-Tuning—
the only other paradigm without inference-time
computation beyond a forward pass—and is the
lowest among all evaluated methods: 3–19× faster
than every RAG variant and 3–6×faster than KV-
injection baselines, confirming that RING internal-
izes retrieval at minimal serving cost.
5.4 Analysis
Incremental Updating of New KnowledgeWe
evaluate robustness to catastrophic forgetting by
injecting 20k to 100k documents and comparing
against baselines (Figure 4) under two settings.(1)
Sequential Injection:training only on new data
causes severe forgetting for conventional methods:
Full Fine-Tuningdrops from ∼27% to ∼12% at
100k, withLoRAandKBLAMalso degrading no-
ticeably. In contrast, RING remains stable, keeping
accuracy above 30%, indicating effective isolation
of new memories via parameter separation.(2)
Mixed Training:adding replay flattens the base-
lines’ curves, but RING still achieves the best over-
all retention, staying >33% whileKBLAMand
Full Fine-Tuningplateau around∼25% to∼18%.
Ablation StudyWe ablate key components of
RING in Table 2. (1) Architecture: Sparse MoE
improves 30.59% →32.71%, and adding our gat-
ing further boosts to 35.08%. (2) Attention: DCA
outperforms standard causal attention (35.08% vs.
33.49%). (3) Training: RL is essential—SFT only
reaches 28.41%, while SFT+RL achieves 35.08%.
(4) Rewards: search reward beats answer-only re-
ward (35.08% vs. 32.17%), showing the benefit ofTable 2: Ablation study of RING. Performance is re-
ported as the accuracy (%) on English dataset.
Configuration Accuracy (↑)
1. Impact of Model Architecture
Dense (Standard) 30.59
Sparse (MoE) 32.71
Sparse + Gate (Ours)35.08
2. Impact of Attention Mechanism
Uni-directional Attention (Standard) 33.49
Dual Causal Attention (Ours)35.08
3. Impact of Training Stages
SFT Only 28.41
SFT + RL35.08
4. Impact of Reward Components in RL
RL w/R format +R answer (Answer Reward) 32.17
RL w/R format +R answer +R search (Search Reward)35.08
0 2 4 6 8 10 12
PPL (Perplexity)0.000.050.100.150.200.250.30QA AccuracySFT
RL 800 step
RL 1600 step
RL 2400 step
RL 3200 step
(a) PPL-Accuracy mapping
through RL training steps.
400 800 1200 1600 2000 2400 2800 3200 3600 4000
Training Steps0.10.20.30.40.5ScoreAccuracy Performance
Raw Data
Accuracy (Trend)(b) (Q&A accuracy) varies by
the RL training steps.
Figure 5: (a) We plot the linear relationship between
Perplexity ( x) and Q&A Accuracy ( y) for SFT and four
RL checkpoints. (b) We show the training dynamics of
Q&A accuracy in RL training.
fine-grained retrieval supervision.
Effect of RLTo investigate how RL facilitates
the utilization of parametric knowledge, we ana-
lyze the relationship between memorization quality
(PPL) and QA accuracy. Figure 5a shows two key
findings. (1) Better utilization under good mem-
orization: in the low-PPL region, RL markedly
improves accuracy (SFT ∼20%→RL>32% ),
indicating that memorization alone is not sufficient
and RL helps the model reliably activate and use
stored knowledge. (2) Stronger grounding on para-
metric memory: as RL proceeds, the PPL–accuracy
curve becomes much steeper—accuracy becomes
tightly coupled with memorization quality. When
PPL is high, the RL model degrades sharply, sug-
gesting reduced reliance on pre-trained priors and
fewer “lucky guesses.” RL bridges memorization
and effective retrieval-based usage of parametric
knowledge.
6 Conclusion
We introduce RING, a fully parametric, search-
free RAG framework that unifies indexing, re-
trieval, and generation inside a LLM. RING injects
8

large-scale new knowledge into a dedicated Knowl-
edge Expert via continued pre-training with Dual
Causal Attention, and learns tosearch-then-answer
through supervised fine-tuning and reinforcement
learning. We further provide a latent-variable view
connecting RING to the classical RAG objective,
showing it as a search-free approximation with a
principled inference interpretation. Experiments
show that RING matches or exceeds search-based
RAG baselines while reducing inference latency.
Limitations
First, RING targets settings where a large knowl-
edge corpus is relatively stable and can be inter-
nalized before deployment. Unlike conventional
RAG systems, which can update an external index
by adding or deleting documents, RING requires
additional training to refresh the parametric mem-
ory. This makes it less suitable for highly dynamic
knowledge sources unless efficient incremental up-
date strategies are developed.
Second, although RING generates an explicit
<retrieval> block, this generated evidence is not
identical to returning a verbatim document from an
external database. The model may compress, para-
phrase, or occasionally distort memorized content,
which can make auditing and provenance tracking
harder than in search-based RAG. Applications that
require strict citation fidelity or legal traceability
may still benefit from hybrid designs that combine
parametric retrieval with external verification.
Third, our experiments focus on knowledge-
intensive question answering over a constructed
news corpus. While this setting stresses large-scale
memorization and evidence selection, it does not
cover all retrieval scenarios, such as multi-hop web
search, rapidly changing facts, multilingual knowl-
edge bases, or long-form generation with many
cited sources. Extending RING to these settings is
an important direction for future work.
Finally, the proposed training pipeline introduces
non-trivial upfront cost. Continued pre-training
with Dual Causal Attention, supervised search-
then-answer learning, and RL with dense retrieval
rewards require carefully constructed data and sub-
stantial computation. The inference benefits of
search-free generation should therefore be weighed
against the cost of building and maintaining the
parametric memory.Ethical Considerations
This paper proposes RING, a search-free retrieval-
augmented generation framework that internalizes
indexing and retrieval into model parameters to
reduce latency and deployment complexity. The
primary positive impact is enabling more efficient
and accessible knowledge-augmented systems, es-
pecially in settings where maintaining external re-
trieval infrastructure is costly or impractical.
Potential risks are similar to those of other LLM-
based systems. Because retrieval is performed para-
metrically, errors in memorization or retrieval may
lead to confident but incorrect outputs, and harm-
ful or biased content present in the injected cor-
pus could be reinforced or surfaced during gener-
ation. In addition, internalizing knowledge may
make it harder to audit, trace, or remove specific
information compared to external retrieval, raising
concerns around data governance and privacy if
sensitive data were injected.
To mitigate these risks, RING should be used
with careful dataset curation and filtering, privacy-
preserving data handling, and evaluation for bias
and safety. Where high-stakes decisions are in-
volved, we recommend adding safeguards such
as refusal policies, post-hoc verification, and
provenance-aware logging or optional external val-
idation. We hope this work encourages further
research on controllable, auditable, and safe para-
metric knowledge integration.
References
Michele Bevilacqua, Giuseppe Ottaviano, Patrick Lewis,
Scott Yih, Sebastian Riedel, and Fabio Petroni. 2022.
Autoregressive search engines: Generating substrings
as document identifiers.Advances in Neural Infor-
mation Processing Systems, 35:31668–31683.
Jianlv Chen, Shitao Xiao, Peitian Zhang, Kun Luo, Defu
Lian, and Zheng Liu. 2024. Bge m3-embedding:
Multi-lingual, multi-functionality, multi-granularity
text embeddings through self-knowledge distillation.
arXiv preprint arXiv:2402.03216, 4(5).
Zhiqi Chen, Rui Lu, Andrew Zhao, Zhaokai Wang, Yang
Yue, Shiji Song, and Gao Huang. 2026. Does rein-
forcement learning really incentivize reasoning ca-
pacity in llms beyond the base model?Advances in
Neural Information Processing Systems, 38:57654–
57689.
Jingcheng Deng, Liang Pang, Zihao Wei, Shicheng Xu,
Zenghao Duan, Kun Xu, Yang Song, Huawei Shen,
and Xueqi Cheng. 2025a. Latent reasoning in llms
as a vocabulary-space superposition.
9

Jingcheng Deng, Zihao Wei, Liang Pang, Hanxing Ding,
Huawei Shen, and Xueqi Cheng. 2025b. Everything
is editable: Extend knowledge editing to unstructured
data in large language models. InInternational Con-
ference on Learning Representations, volume 2025,
pages 628–652.
Jingcheng Deng, Zihao Wei, Liang Pang, Junhong
Wu, Shicheng Xu, Zenghao Duan, and Huawei
Shen. 2026. Latent-grpo: Group relative policy
optimization for latent reasoning.arXiv preprint
arXiv:2604.27998.
Zenghao Duan, Yuxin Tian, Zhiyi Yin, Liang Pang,
Jingcheng Deng, Zihao Wei, Shicheng Xu, Yuyao
Ge, and Xueqi Cheng. 2026. Skillattack: Automated
red teaming of agent skills through attack path refine-
ment.arXiv preprint arXiv:2604.04989.
William Fleshman and Benjamin Van Durme.
2025. Lora-augmented generation (lag) for
knowledge-intensive language tasks.arXiv preprint
arXiv:2507.05346.
Luyu Gao, Xueguang Ma, Jimmy Lin, and Jamie Callan.
2023a. Precise zero-shot dense retrieval without rel-
evance labels. InProceedings of the 61st Annual
Meeting of the Association for Computational Lin-
guistics (Volume 1: Long Papers), pages 1762–1777.
Yunfan Gao, Yun Xiong, Xinyu Gao, Kangxiang Jia,
Jinliu Pan, Yuxi Bi, Yixin Dai, Jiawei Sun, Haofen
Wang, Haofen Wang, and 1 others. 2023b. Retrieval-
augmented generation for large language models: A
survey.arXiv preprint arXiv:2312.10997, 2(1):32.
Mor Geva, Roei Schuster, Jonathan Berant, and Omer
Levy. 2021. Transformer feed-forward layers are
key-value memories. InProceedings of the 2021
Conference on Empirical Methods in Natural Lan-
guage Processing, pages 5484–5495.
Xuming Hu, Junzhe Chen, Xiaochuan Li, Yufei Guo, Li-
jie Wen, Philip Yu, and Zhijiang Guo. 2024. Towards
understanding factual knowledge of large language
models. InInternational Conference on Learning
Representations, volume 2024, pages 28680–28715.
Haoyu Huang, Hong Ting Tsang, Jiaxin Bai, Xi Peng,
Gong Zhang, and Yangqiu Song. 2025. At-
laskv: Augmenting llms with billion-scale knowl-
edge graphs in 20gb vram.arXiv preprint
arXiv:2510.17934.
Gautier Izacard and Edouard Grave. 2021. Leveraging
passage retrieval with generative models for open do-
main question answering. InProceedings of the 16th
conference of the european chapter of the association
for computational linguistics: main volume, pages
874–880.
Jinhyuk Lee, Feiyang Chen, Sahil Dua, Daniel
Cer, Madhuri Shanbhogue, Iftekhar Naim, Gus-
tavo Hernández Ábrego, Zhe Li, Kaifeng Chen, Hen-
rique Schechter Vera, and 1 others. 2025. Gemini
embedding: Generalizable embeddings from gemini.
arXiv preprint arXiv:2503.07891.Patrick Lewis, Ethan Perez, Aleksandra Piktus, Fabio
Petroni, Vladimir Karpukhin, Naman Goyal, Hein-
rich Küttler, Mike Lewis, Wen-tau Yih, Tim Rock-
täschel, and 1 others. 2020. Retrieval-augmented gen-
eration for knowledge-intensive nlp tasks.Advances
in neural information processing systems, 33:9459–
9474.
Chaofan Li, Zheng Liu, Shitao Xiao, and Yingxia Shao.
2023. Making large language models a better founda-
tion for dense retrieval.Preprint, arXiv:2312.15503.
Xiaoxi Li, Jiajie Jin, Yujia Zhou, Yuyao Zhang, Peitian
Zhang, Yutao Zhu, and Zhicheng Dou. 2025. From
matching to generation: A survey on generative infor-
mation retrieval.ACM Transactions on Information
Systems, 43(3):1–62.
Kevin Meng, Arnab Sen Sharma, Alex Andonian,
Yonatan Belinkov, and David Bau. 2022. Mass-
editing memory in a transformer.arXiv preprint
arXiv:2210.07229.
Cuong V Nguyen, Alessandro Achille, Michael Lam,
Tal Hassner, Vijay Mahadevan, and Stefano Soatto.
2019. Toward understanding catastrophic for-
getting in continual learning.arXiv preprint
arXiv:1908.01091.
Haizhou Shi, Zihao Xu, Hengyi Wang, Weiyi Qin,
Wenyuan Wang, Yibin Wang, Zifeng Wang, Sayna
Ebrahimi, and Hao Wang. 2025. Continual learning
of large language models: A comprehensive survey.
ACM Computing Surveys, 58(5):1–42.
Weihang Su, Yichen Tang, Qingyao Ai, Junxi Yan,
Changyue Wang, Hongning Wang, Ziyi Ye, Yujia
Zhou, and Yiqun Liu. 2025. Parametric retrieval
augmented generation. InProceedings of the 48th
International ACM SIGIR Conference on Research
and Development in Information Retrieval, pages
1240–1250.
Yi Tay, Vinh Tran, Mostafa Dehghani, Jianmo Ni, Dara
Bahri, Harsh Mehta, Zhen Qin, Kai Hui, Zhe Zhao,
Jai Gupta, and 1 others. 2022. Transformer memory
as a differentiable search index.Advances in neural
information processing systems, 35:21831–21843.
Cunxiang Wang, Xiaoze Liu, Yuanhao Yue, Xiangru
Tang, Tianhang Zhang, Cheng Jiayang, Yunzhi Yao,
Wenyang Gao, Xuming Hu, Zehan Qi, and 1 others.
2023. Survey on factuality in large language models:
Knowledge, retrieval and domain-specificity.arXiv
preprint arXiv:2310.07521.
Liyuan Wang, Xingxing Zhang, Hang Su, and Jun Zhu.
2024a. A comprehensive survey of continual learn-
ing: Theory, method and application.IEEE transac-
tions on pattern analysis and machine intelligence,
46(8):5362–5383.
Xi Wang, Taketomo Isazawa, Liana Mikaelyan, and
James Hensman. 2025. Kblam: Knowledge base
augmented language model. InInternational Con-
ference on Learning Representations, volume 2025,
pages 51629–51658.
10

Yan Wang, Dongyang Ma, and Deng Cai. 2024b. With
greater text comes greater necessity: Inference-time
training helps long text generation.arXiv preprint
arXiv:2401.11504.
Rubin Wei, Jiaqi Cao, Jiarui Wang, Jushi Kai, Qipeng
Guo, Bowen Zhou, and Zhouhan Lin. 2025. Mlp
memory: A retriever-pretrained memory for large
language models.arXiv preprint arXiv:2508.01832.
Zihao Wei, Wenjie Shi, Liang Pang, Jingcheng Deng,
Shicheng Xu, Shasha Guo, Zenghao Duan, Jia-
hao Liu, Jingang Wang, Huawei Shen, and 1 oth-
ers. 2026. Dynamic rollout editing for reducing
overthinking in rl-trained reasoning models.arXiv
preprint arXiv:2606.17890.
Shicheng Xu, Liang Pang, Huawei Shen, and Xueqi
Cheng. 2025. A theory for token-level harmonization
in retrieval-augmented generation. InInternational
Conference on Learning Representations, volume
2025, pages 36616–36642.
Shicheng Xu, Liang Pang, Huawei Shen, Xueqi Cheng,
and Tat-Seng Chua. 2024a. Search-in-the-chain: In-
teractively enhancing large language models with
search for knowledge-intensive tasks. InProceedings
of the ACM Web Conference 2024, pages 1362–1373.
Shicheng Xu, Liang Pang, Mo Yu, Fandong Meng,
Huawei Shen, Xueqi Cheng, and Jie Zhou. 2024b.
Unsupervised information refinement training of
large language models for retrieval-augmented gener-
ation. InProceedings of the 62nd annual meeting of
the association for computational linguistics (volume
1: Long papers), pages 133–145.
Bohan Yu, Wei Huang, and Kang Liu. 2026. Sr-ki:
Scalable and real-time knowledge integration into
llms via supervised attention. InProceedings of
the AAAI Conference on Artificial Intelligence, vol-
ume 40, pages 34486–34494.
Ningyu Zhang, Yunzhi Yao, Bozhong Tian, Peng Wang,
Shumin Deng, Mengru Wang, Zekun Xi, Shengyu
Mao, Jintian Zhang, Yuansheng Ni, and 1 others.
2024. A comprehensive study of knowledge edit-
ing for large language models.arXiv preprint
arXiv:2401.01286.
Yanzhao Zhang, Mingxin Li, Dingkun Long, Xin Zhang,
Huan Lin, Baosong Yang, Pengjun Xie, An Yang,
Dayiheng Liu, Junyang Lin, and 1 others. 2025a.
Qwen3 embedding: Advancing text embedding and
reranking through foundation models.arXiv preprint
arXiv:2506.05176.
Yuwei Zhang, Wenhao Yu, Shangbin Feng, Yifan
Zhu, Letian Peng, Jayanth Srinivasa, Gaowen Liu,
and Jingbo Shang. 2025b. Bidirectional lms are
better knowledge memorizers? a benchmark for
real-world knowledge injection.arXiv preprint
arXiv:2505.12306.
Chujie Zheng, Shixuan Liu, Mingze Li, Xiong-Hui
Chen, Bowen Yu, Chang Gao, Kai Dang, YuqiongLiu, Rui Men, An Yang, and 1 others. 2025.
Group sequence policy optimization.arXiv preprint
arXiv:2507.18071.
A Theoretical Framework
This section provides a principled view ofRING
as asearch-freeapproximation to the classical
Retrieval-Augmented Generation (RAG) objective.
Our key message is: (i) classical RAG can be writ-
ten as marginalizing a discrete latentevidencevari-
able over a corpus; (ii) RING parameterizes this la-
tent retrieval distribution entirely within the model
(router + knowledge expert), after injecting the cor-
pus into Θmem via CPT; and (iii) our CPT–SFT–RL
pipeline can be interpreted as optimizing a varia-
tional lower bound and an evidence-selection pol-
icy.
A.1 Problem Formulation: Discrete Latent
Evidence for RAG
Let the external knowledge corpus beD, and let
U={u z}N
z=1
denote the set ofmemory unitsderived from D
after preprocessing (Section 3.2): each uzis the
JSON-style fragment {title, content} (possi-
bly truncated) used consistently in CPT/SFT/RL.
Given a query xand target answer y, we model
knowledge-intensive generation with a discrete la-
tent evidence indexz∈ {1, . . . , N}:
p(y|x,U) =NX
z=1p(z|x,U)p θgen(y|x, u z).
(6)
Here p(z|x,U) is the retrieval distribution, and
pθgen(y|x, u z)is the generator conditioned on
evidence uz(Xu et al., 2025, 2024b; Duan et al.,
2026).
Explicit (search-based) RAG as truncated
marginalizationIn conventional RAG, p(z|
x,U) is implemented by an external retriever (e.g.,
embedding similarity + ANN search) with score
sη(x, u z):
pη(z|x,U) =exp(s η(x, u z))PN
j=1exp(s η(x, u j)).
Since Nis large, systems typically retrieve a top- K
setT K(x)and renormalize:
pRAG(y|x,U)≈X
z∈TK(x)˜pη(z|x,U)p θgen(y|x, u z),
(7)
where˜p η(z) =p η(z)/P
z′∈TK(x)pη(z′).
11

A.2 RING Objective: Search-Free Parametric
Retrieval
RING eliminates external access toUat inference
by(i)injecting Uinto the knowledge expert pa-
rameters during CPT, and(ii)learning a parametric
retrieval policy during SFT/RL.
Definition A.1(RING Latent Evidence Objec-
tive).Let Θ = ( ¯Θ,Θ mem, ϕ), where ¯Θde-
notes all frozen backbone parameters (including
the base expert and attention blocks), Θmem =
{W(ℓ)
knw,down, W(ℓ)
knw,up}L
ℓ=1are the trainable memory
projections in the knowledge expert (Section 3.2),
andϕare the parameters of the router rϕ(and the
Search Gate in Es
knw). We define the parametric
joint model
pΘ(y, z|x) =p ϕ(z|x)p θgen(y|x, u z; Θmem,¯Θ),
(8)
and the marginal
pΘ(y|x) =NX
z=1pϕ(z|x)p θgen(y|x, u z; Θmem,¯Θ).
(9)
At inference, RING does not query Uexternally;
the dependence on Uis realized only through the
injected parametersΘ mem.
Connection to <retrieval> generationIn the
Search-then-Answerparadigm (Section 3.3), RING
explicitly generates a retrieval block that is in-
tended to reproduce (one of) the memorized units
uz. Thus, we can interpret pϕ(z|x) as the induced
probability that the model generates unit uzinside
<retrieval> ...</retrieval> given the question
x.
Top-1 (MAP) evidence approximation and an
error boundIn practice, RING often aims to
retrieve a single best unit (top-1) before answering.
Define a score
az= logp ϕ(z|x)+logp θgen(y|x, u z; Θmem,¯Θ),
and let z⋆= arg max zaz. Then log-sum-exp
yields a standard approximation bound:
0≤logNX
z=1eaz−az⋆
= log
1 +X
z̸=z⋆eaz−az⋆
≤log 
1 + (N−1)e−∆
,(10)where ∆ =a z⋆−max z̸=z⋆azis the score gap.
Therefore, when the evidence posterior is peaked
(large ∆), the top-1 evidence approximation is
provably close to the full marginal in Eq. (9).
A.3 Architecture View: Knowledge Expert as
Parametric Key–Value Retrieval
We now connect RING’s knowledge expert to a
differentiable retrieval mechanism.
Single-layer viewConsider the knowledge ex-
pert MLP at layer ℓapplied to a hidden state
h∈Rd. With the Memory Up/Down and Search
Gate separation (Section 3.1), a convenient abstrac-
tion is the gated-MLP form (e.g., SiLU-gating):
E(ℓ)
knw(h) =W(ℓ)
knw,down
SiLU 
W(ℓ)
knw,gateh
⊙ 
W(ℓ)
knw,uph
,(11)
where W(ℓ)
knw,downandW(ℓ)
knw,upbelong to Θmem
(trainable in CPT), whileW(ℓ)
knw,gatecorresponds to
the Search Gate parameters in Es
knw(trainable in
SFT/RL).
Proposition A.2(Gated-MLP as Key–Value Mem-
ory Readout).Let dffbe the intermediate width of
the MLP , and denote by k(ℓ)
ithei-th row of W(ℓ)
knw,up,
byg(ℓ)
ithei-th row of W(ℓ)
knw,gate, and by v(ℓ)
ithei-th
column of W(ℓ)
knw,down. Then Eq. (11) can be written
as
E(ℓ)
knw(h) =dffX
i=1α(ℓ)
i(h)v(ℓ)
i,
α(ℓ)
i(h) = SiLU 
⟨g(ℓ)
i, h⟩
· ⟨k(ℓ)
i, h⟩.(12)
Hence each hidden representation hacts as aquery,
the rows of Wknw,up (modulated by the Search Gate)
act askeys, and the columns of Wknw,down act as
values.
Proof. Expand the matrix products in Eq. (11)
coordinate-wise: Wknw,down handWknw,gate hpro-
ducedffscalar activations; the elementwise product
yields coefficients αi(h), and the final multiplica-
tion by Wknw,down sums columns weighted by these
coefficients.
Lemma A.3(Approximate MIPS via Low-Temper-
ature Softmax).Let si(h)be any scalar scores (e.g.,
si(h) =α i(h)or an affine transform thereof). De-
fineπβ(i|h) = softmax i(βsi(h)) and the read-
outRβ(h) =P
iπβ(i|h)v i. If the maximizer
12

i⋆= arg max isi(h)is unique, then πβ(i⋆|h)→
1andR β(h)→v i⋆asβ→ ∞.
Proof. This is the standard low-temperature limit
of softmax: probability mass concentrates on the
unique argmax asβ→ ∞.
Two-level sparsity: router vs. memory-slot se-
lectionRING has two complementary sparsity
mechanisms: (i) the router rϕselects between ex-
perts EbaseandEknwviag(ℓ)
t(Section 3.1), con-
trollingwhento consult injected knowledge; (ii)
within Eknw, the gated-MLP readout (Eq. (12)) acts
as a differentiable retrieval over a large number of
implicit memory slots, controllingwhatknowledge
is activated. This provides a parameter-space ana-
logue of similarity search.
A.4 Why RING Can Match or Exceed
Explicit RAG
We formalize the comparison as a hypothesis-class
inclusion argument.
Theorem A.4(Expressivity Dominance on the
Query Domain).Assume that after CPT, the in-
jected parameters Θmem store the evidence units
{uz}N
z=1 with sufficient fidelity, and that the
router/Search Gate parameters ϕhave enough ca-
pacity to represent the same retrieval distribution
as an external retriever on the query domain of
interest; i.e., for all relevantx,
pϕ(z|x)≈˜p η(z|x,U)forz∈ T K(x),
up to arbitrarily small error. Then for any explicit
top-KRAG model in Eq. (7), there exists a RING
instance in Eq. (9)that matches its conditional dis-
tribution p(y|x) arbitrarily well on that domain
(in total variation / KL, depending on the approxi-
mation metric).
Proof sketch. Construct pϕ(z|x) to approximate
the external truncated-retrieval distribution ˜pη(z|
x,U) onTK(x), and set the conditional genera-
torpθgen(y|x, u z)to match the same evidence-
conditioned generator. Since the evidence units
are stored in Θmem, RING can reproduce the same
mixture in Eq. (7) without external search.
Corollary A.5(Best Achievable NLL is Not Worse
(Under Capacity Assumptions)).Let R(f) =
E(x,y)∼P⋆[−logp f(y|x)] be population NLL. Un-
der the assumptions of Theorem A.4, the optimalachievable risk of RING is no worse than that of
explicit top-KRAG on the same domain:
inf
ΘR 
pΘ
≤inf
η,θgenR 
pRAG
.
RemarkA.6 (When can RING be strictly better?).
Even with identical backbone generators, explicit
RAG is constrained by external retrieval form (e.g.,
embedding similarity), ANN approximation, top-
Ktruncation, and finite prompt/context budget.
RING jointly optimizes routing and generation and
can learn task-aligned evidence selection beyond
embedding similarity, potentially achieving lower
risk on distributions where similarity search is a
suboptimal proxy for selecting evidence.
A.5 Dual Causal Attention: Formal Mask and
Loss Definitions
We give the full formal specification of the Dual
Causal Attention (DCA) objective used in CPT
(Section 3.2).
Let a token sequence be x1:T= (x 1, . . . , x T)
and consider a decoder layer with queries, keys,
and values Q, K, V∈RT×d. For a binary mask
M∈ {0,1}T×T, the masked attention is
Attn(Q, K, V;M) = softmaxQK⊤
√
d+ logM
V,
withlog 0 :=−∞,log 1 := 0.
Forα∈ {1,1
2,1
4}, letKα=⌊αT⌋ and define
thetarget block Bα={1, . . . , K α}and its com-
plement Cα={K α+ 1, . . . , T} . The masks
M(α)∈ {0,1}T×Tspecify which source positions
jare visible when predicting position i. Forα= 1
the standard causal mask is M(1)
i,j=1[j≤i] ; for
α∈ {1
2,1
4}the dual causal mask is
M(α)
i,j=(
1, i∈B α, j≤iorj∈C α,
1[j≤i], i∈C α.
(13)
Thus, when predicting tokens inside Bα, the model
preserves causality within Bα(no look-ahead to
future targets in the block) while attending to all of
Cαas backward evidence; positions in Cαretain
the standard causal constraint.
For each α, the negative log-likelihood over the
corresponding target block is
L(α)=1
|Bα|X
t∈Bα−logp θ
xtx1:T;M(α)
,
(14)
13

where pθ(·|x 1:T;M(α))is obtained by running the
network on the full sequence with mask M(α)and
reading logits only at positions t∈B α. The three
losses are combined as
LDCA=λ1L(1)+λ2L(1/2)+λ3L(1/4).(15)
In practice, the same training example is passed
three times (once per mask), gradients are accumu-
lated according to Eq. (14) and mixed via Eq. (15);
at inference, onlyM(1)is used.
A.6 Training as Variational Inference:
Linking CPT–SFT–RL to an ELBO
We derive an evidence lower bound (ELBO) for the
discrete latent model in Eq. (8).
ELBO with a variational posteriorIntroduce a
variational posteriorq(z|x, y). Then
logp Θ(y|x)
= logNX
z=1q(z|x, y)pϕ(z|x)p θgen(y|x, u z)
q(z|x, y)
≥E q
logp θgen(y|x, u z)
| {z }
reconstruction
−KL 
q(z|x, y)∥p ϕ(z|x)
| {z }
posterior–policy alignment.(16)
At test time, RING uses the policy pϕ(z|x) (or its
top-1 approximation), while q(z|x, y) is only a
training-time analysis tool.
Step 1 (CPT): fitting the parametric mem-
oryΘmem During CPT, we force routing to
the knowledge expert and update only Θmem=
{W(ℓ)
knw,up, W(ℓ)
knw,down}, optimizing the DCA objec-
tive on DCPT (Section 3.2). This stage increases
the capacity of pθgen(· |x, u z)to represent and
reproduce evidence units uzthrough the knowl-
edge expert, preparing the model for subsequent
evidence selection.
A principled view of DCADCA provides addi-
tional right-context evidence for predicting tokens
in a target block while retaining autoregressive gen-
eration at inference. The key theoretical property
we rely on is monotonicity of optimal conditional
cross-entropy with respect to conditioning sets.
Proposition A.7(DCA Does Not Worsen the Opti-
mal Conditional Cross-Entropy).Consider predict-
ing a target block conditioned on its complement.Any training objective that allows tokens in the tar-
get block to condition on asupersetof information
(as DCA does for the target block) has an optimal
achievable negative log-likelihood no larger than
that of standard causal conditioning, because the
model can always learn to ignore additional con-
text. Consequently, DCA can provide a tighter (not
looser) upper bound on the conditional entropy of
the target block given its complement, improving
the information available for storing bidirectional
dependencies intoΘ mem during CPT.
Step 2 (SFT): supervised posterior collapse via
<retrieval> In SFT, each instruction sample
provides a gold evidence unit u⋆=uz⋆obtained
by exact string match (Section 3.3). Setting q(z|
x, y) =δ(z=z⋆)in Eq. (16) yields the supervised
objective
max logp ϕ(z⋆|x) + logp θgen(y|x, u z⋆),
which corresponds exactly to teacher-forcing on the
<retrieval> block (learning evidence selection)
and the <answer> block (learning answer synthe-
sis given evidence). Freezing Θmem during SFT
prevents overwriting the memorized corpus and
focuses learning onr ϕand the Search Gate.
Step 3 (RL): learning a generalizable evidence-
selection policyAfter SFT, we apply RLVR to
further align the evidence-selection behavior with
task success. Because different implementations
may define “rewards” or “penalties” with oppo-
site signs, we present the RL objective in a sign-
agnostic form.
Letℓfmt, ℓans, ℓdense ∈[0,1] denotepenal-
ties(lower is better) for (i) format compliance,
(ii) verified answer correctness, and (iii) evi-
dence match (e.g., overlap/LCS-based signals on
title/content ), respectively. Define the total
penalty
ℓtotal=λfmtℓfmt+λansℓans+λdenseℓdense,
and maximize thenegativepenalty (equivalently,
minimize expected penalty):
max
ϕEz∼pϕ(·|x)
−ℓtotal(z, x, y)
⇐⇒min
ϕEz∼pϕ(·|x)
ℓtotal(z, x, y)
.
This formulation is compatible with either conven-
tion (reward-as-1-for-correct or penalty-as-1-for-
incorrect).
14

Lemma A.8(Dense Penalties Improve Credit As-
signment for Retrieval).Suppose ℓansis sparse
(many actions zreceive identical penalties be-
cause answer correctness is hard to achieve), while
ℓdense(z, z⋆)varies smoothly with the match qual-
ity between the sampled evidence zand the gold
evidence z⋆. Then optimizing E[ℓdense]provides
informative gradients for the policy pϕ(z|x) even
when ℓansprovides little signal, accelerating the
concentration of pϕ(· |x) around high-quality evi-
dence choices.
RemarkA.9 (Relation to the ELBO).When the
dense signal correlates with evidence correctness,
minimizing expected penalty can be viewed as a
practical surrogate for reducing the mismatch be-
tween the policy pϕ(z|x) and the ideal posterior
over evidence. This complements the supervised
alignment in SFT and improves generalization of
evidence selection beyond exact-match supervi-
sion.
B Datasets
Memorization Set (CPT):We collect the raw
news corpus and segment it into fine-grained units,
resulting in approximately 1.6 million fragments.
This set serves as the external knowledge base
DCPT that needs to be injected.
Instruction Set (SFT):Based on the CPT cor-
pus, we generate 317k instruction-following sam-
ples using the pipeline described in Section 3.3.
These samples are designed to teach the model
the “Search-then-Answer” paradigm. The detailed
construction recipe is given below.
Alignment Set (RL):We construct a further
200k samples specifically for the RL stage.
B.1 Instruction Fine-tuning Data
Construction Details
We expand the main-text description of the SFT
data pipeline.
LetDbe the external corpus to be injected and
DCPT the preprocessed set of memory units used
during CPT (each unit uis the JSON-style {title,
content} fragment defined in Section 3.2). We
subsample documents S ⊆ D with rate ρ∈(0,1] ,
a tunable hyperparameter trading off coverage
against training cost.
Triple SynthesisWe employ
qwen3-235b-a22b-instruct-2507 to pro-
duce nsupervision triples {(qi, ai, ri)}n
i=1per
sampled document d∈ S , where qiis aninstruction-style question, aiis a concise answer
grounded in d, and riis an explicitreference span
from dthat justifies ai. We control nto balance
coverage and cost.
Reference–Memory AlignmentLet U={u j}
be the set of memory units in DCPT. We align
each reference rito its corresponding memory unit
u∈ D CPT via exact string match: the pair is ac-
cepted only if there exists a u⋆
i∈ U that matches
ri; otherwise the triple is discarded. The accepted
u⋆
iis truncated to at most 256tokens to bound gen-
eration cost.
Target TemplateEach retained triple is
converted into asearch-then-answertarget
string that the model must autoregressively
produce given the question. Concretely,
Instruction :xin=q i;Response :ytg=
I can retrieve this passage to answer
the question <retrieval>u⋆
i</retrieval>
<answer>a i</answer> . The <retrieval> block
supervises generative retrieval, and the <answer>
block supervises final answering.
Retrieve-only-when-neededTo prevent the
model from always emitting <retrieval> , we
mix in no-retrieval instructions from a generic in-
struction corpus G(math, coding, reasoning), for
which the response does not follow the Search-
then-Answer template. The final SFT set is
I={(q, u⋆, a)} ∪ {(q,∅, a)} , stratified into
train/dev/test by document ID to prevent leakage.
During training we interleave Iwith a general in-
struction set to stabilize early learning and improve
instruction following.
LossGiven input xinand target ytg, the parame-
tersθare optimized with standard cross-entropy:
LSFT(θ) =E (q,u⋆,a)∼Ih
−|ytg|X
t=1logp θ 
ytg
t|xin, ytg
<ti
,
where the presence of the <retrieval> block di-
rectly supervises the decision to retrieve.
C Experimental Setting
C.1 Inference Pipelines
To ensure a rigorous and fair comparison between
the parametric and external retrieval paradigms,
we strictly utilize the identical knowledge corpus
for all methods. For conventional search-based
RAG baselines, the inference pipeline follows a
15

standard retrieve-then-generate workflow: given a
query, the system searches the external vector index
to retrieve the relevant documents. This retrieved
context is then provided to the LLM to guide an-
swer generation. In contrast, our RING operates
in a search-free manner. The model is presented
solely with the query, and the retrieval process is
internalized. The model directly generates the an-
swer by leveraging the knowledge encoded within
its parameters (specifically the Knowledge Expert),
completely eliminating the interaction with exter-
nal search engines during inference.
C.2 Evaluation Metrics
We assess performance based on factual accuracy.
The ground truth answers in our benchmark are pre-
processed to be concise and unambiguous. To scale
the evaluation, we employ a model-based judge
using DeepSeek-R1-0528 . The judge compares
the generated response against the ground truth and
assigns a binary score (1 for a match, 0 otherwise).
We report the final Accuracy (%) averaged across
the test set.
C.3 Baseline Details
We provide the full descriptions of all baseline cat-
egories compared in Section 5.3.
Conventional Search-based RAG (Top-1)This
category represents the standard retrieve-and-read
RAG. They rely on an external search engine to
fetch relevant context, which is then concatenated
to the input of the frozen base LLM. We employ a
variety of state-of-the-art (SOTA) embedding mod-
els to perform retrieval, ensuring a strong upper
bound: BGE-M3 (Chen et al., 2024), OpenAI-
text-embedding-3-large, Gemini-Embedding (Lee
et al., 2025), and Qwen3-Embedding (Zhang et al.,
2025a).
Stronger RAG PipelinesTo address the con-
cern that a top-1 retrieve-and-read setup under-
estimates modern RAG, we additionally evaluate:
(i) Top- kretrievalwith k∈ {3,5,10} using the
strongest embedding (Qwen3-8B-Emb);(ii) re-
trieval + reranking, where we retrieve top-10
candidates and rerank them with a cross-encoder
(BGE-Reranker-v2-m3 and Qwen3-Reranker-4B)
to select the final top-3;(iii) HyDE(Gao et al.,
2023a), which generates a hypothetical answer to
rewrite the query before retrieval; and(iv) Query
Rewrite, in which we prompt the base LLM to
paraphrase the question prior to retrieval.Parametric Knowledge Injection and Memoriza-
tionThese baselines aim to inject the external
corpus Ddirectly into the model parameters.(1)
Full Tuning: standard full-parameter CPT on the
raw corpus DCPTfollowed by instruction tuning
on large-scale Q&A pairs constructed from D.(2)
LoRA Tuning: same training set as Full Tuning but
applied via LoRA.(3) Modular Injection: meth-
ods that mount knowledge into the LLM’s com-
putational flow, including KBLAM (Wang et al.,
2025) and SR-KI (Yu et al., 2026).(4) Retriever-
Imitating Parametric Memory: three recent
methods closest to RING in spirit—LAG (Flesh-
man and Van Durme, 2025) attaches per-document
LoRA adapters and selects them at inference via
data-free routing; MLP Memory (Wei et al., 2025)
pretrains a dedicated MLP to imitate a kNN re-
triever’s distribution; AtlasKV (Huang et al., 2025)
stores billion-scale knowledge as compressed key–
value caches augmented into the LLM. A signif-
icant drawback of (3) and partially of (4) is that
GPU memory usage grows with the size of the
injected corpus. Compared with these baselines,
RING (i) keeps a separate Basic Expert to prevent
forgetting, (ii) injects the corpus via bidirectional
DCA rather than fitting a retriever’s logits or stor-
ing raw KV slabs, and (iii) learnswhenandwhat
to retrieve through RL with dense rewards rather
than via a fixed similarity-based mechanism.
C.4 Implementation Details
We utilize Qwen3-8B and Qwen3-14B as our back-
bone foundation model. Our training pipeline con-
sists of three sequential stages. We use the AdamW
optimizer with a cosine learning rate scheduler for
all stages. In CPT, we train the model on the con-
structed News-2025 corpus. During this stage, we
freeze all parameters except for the Memory Down
and Memory Up projections of the Knowledge Ex-
pert. We employ the proposed Dual Causal Atten-
tion with mask ratios α∈ {1,1/2,1/4} and loss
weights λ1= 1.0 ,λ2= 0.3 , and λ3= 0.1 . The
max sequence length is set to 4096, the learning
rate is 2e-5 and epoch is 8. In SFT, we utilize
the Search-then-Answer instruction set mixed with
general instructions at a ratio of 1:1. Only Router
and Search Gate are trainable. Retrieved fragments
are truncated to 256 tokens. The max sequence
length is set to 4096, the learning rate is 2e-5 and
epoch is 4 with early-exit. In RL, we use ms-swift
as the framework. λa,λb, and λcare 0.8, 0.8, 1.0
respectively. λformat ,λanswer , and λsearch are 0.1,
16

1.0, 1.0 respectively. All models are implemented
using PyTorch and trained on 32 ×NVIDIA H800
GPUs. We utilize DeepSpeed Zero-3 for memory
optimization. The learning rate is 1e-6.
D AI Use Statement
We used AI-based writing assistance only for lan-
guage polishing, including grammar correction,
wording refinement, and improving readability. AI
tools were not used for formulating the research
ideas, designing the method, conducting experi-
ments, analyzing results, or generating substantive
scientific content. All technical claims, experimen-
tal results, and conclusions were produced and ver-
ified by the authors.
17