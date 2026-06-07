# RCEM: Embedder Equipped with Query Rewriting Skill for Robust Conversational Search in Distributional Shift

**Authors**: Kilho Son, Paul Hsu, Cha Zhang, Dinei Florencio

**Published**: 2026-06-01 05:04:17

**PDF URL**: [https://arxiv.org/pdf/2606.01697v1](https://arxiv.org/pdf/2606.01697v1)

## Abstract
Conversational search has become increasingly important in retrieval-augmented generation (RAG) systems, where users interact with AI assistants through multi-turn conversations containing context-dependent queries. We propose RCEM, a conversational dense retrieval model that distills the query reformulation capability of LLMs into the embedding model, enabling context-aware retrieval without explicit query rewriting during inference. Unlike prior conversational dense retrieval approaches that learn direct conversation-to-document matching, RCEM aligns conversational-query embeddings with rewritten-query embeddings, improving robustness under distributional shift. RCEM does not require conversational query-to-document relevance mappings for training, which are often expensive and difficult to obtain with high quality. Extensive experiments on QReCC, TopiOCQA, and TREC CAsT demonstrate that RCEM consistently outperforms strong conversational retrieval baselines, achieving particularly large gains under distributional shift, including up to 20% improvement in Recall@10. RCEM further extends the base embedding model with conversational query rewriting capability while preserving its original retrieval functionality, allowing both standalone and conversational queries to be encoded by a single model and searched against existing document indexes without rebuilding the retrieval database.

## Full Text


<!-- PDF content starts -->

RCEM: Embedder Equipped with Query Rewriting Skill for Robust
Conversational Search in Distributional Shift
Kilho Son1, Paul Hsu, Cha Zhang, Dinei Florencio
15010 NE 36th Street, Redmond, WA 98052, US
{kilhoson, paulhsu, chazhang, dinei}@microsoft.com
Abstract
Conversational search has become increasingly
important in retrieval-augmented generation
(RAG) systems, where users interact with AI
assistants through multi-turn conversations con-
taining context-dependent queries. We propose
RCEM, a conversational dense retrieval model
that distills the query reformulation capabil-
ity of LLMs into the embedding model, en-
abling context-aware retrieval without explicit
query rewriting during inference. Unlike prior
conversational dense retrieval approaches that
learn direct conversation-to-document match-
ing, RCEM aligns conversational-query em-
beddings with rewritten-query embeddings, im-
proving robustness under distributional shift.
RCEM does not require conversational query-
to-document relevance mappings for training,
which are often expensive and difficult to ob-
tain with high quality. Extensive experiments
on QReCC, TopiOCQA, and TREC CAsT
demonstrate that RCEM consistently outper-
forms strong conversational retrieval baselines,
achieving particularly large gains under distri-
butional shift, including up to 20% improve-
ment in Recall@10. RCEM further extends
the base embedding model with conversational
query rewriting capability while preserving its
original retrieval functionality, allowing both
standalone and conversational queries to be en-
coded by a single model and searched against
existing document indexes without rebuilding
the retrieval database.
1 Introduction
Conversational search has become a fundamental
component of modern retrieval-augmented gener-
ation (RAG) systems, where users interact with
AI assistants through multi-turn dialogue rather
than isolated queries. In these interactions, user
queries are often context-dependent and omit in-
formation mentioned in previous turns. Resolving
1Corresponding author
[ST]
q0: What is a physician’s assistant?
a0: Physician assistant are medical …
q1: What are the educational requirements ..
a1: Complete your bachelor degree …
[ST]
𝑞2: What does it cost?
Rewrite to
What does it cost to become a physician's assistant ?
Comput e embedding of
What does it cost to become a physician's assistant?RCEM 
Dense (Vector) Searchq0: What is a physician’s assistant?
a0: Physician assistant are medical …
q1: What are the educational requirements ..
a1: Complete your bachelor degree …
𝑞2: What does it cost?
Rewrite to
What does it cost to become a physician's assistant?
Compute embedding of
What does it cost to become a physician's assistant?LLM
Embedder
Dense (Vector) Search
(a) Prior methods (b) Proposed RCEMFigure 1:Overview: The proposed a RCEM that pro-
duces embedding of a rewritten query directly from a
conversational context. With one-step inference (trig-
gered with Special Token, [ST]), RCEM enables an
embedder to implicitly resolve the conversation into
a clear standalone query and computes its embedding,
eliminating the need for a separate rewriting stage. Im-
portantly, RCEM equips an embedder with an additional
query-rewriting capability while preserving its original
standalone query functionality and maintaining the same
embedding space. As a result, an RCEM remains com-
patible with existing retrieval systems: both standalone
queries and conversational queries can be encoded by
the same model and searched against an existing docu-
ment index without rebuilding the database.
such contextual dependencies is therefore essential
for retrieving relevant documents and generating
accurate responses.
A common solution is to rewrite conversational
queries into standalone queries using large lan-
guage models (LLMs) before retrieval. Recent ad-
vances in conversational query reformulation and
conversational dense retrieval have substantially
improved retrieval effectiveness in multi-turn set-
tings (Mao et al., 2023b; Mo et al., 2024b; Lai et al.,
2025a). However, existing approaches still face im-
portant limitations. LLM-based rewriting pipelines
require expensive generation during inference, in-
troducing substantial latency and computational
cost. On the other hand, conversational dense re-
trieval methods typically learn direct conversation-arXiv:2606.01697v1  [cs.CL]  1 Jun 2026

to-document matching using collection-specific rel-
evance supervision, which may reduce robustness
under distributional shift.
In this work, we propose RCEM, an embedding
model that directly encodes conversational context
into an embedding corresponding to the rewritten
standalone query. Instead of explicitly generat-
ing rewritten text with separate step before embed-
ding, RCEM distills the query rewriting capability
of LLMs into the embedding model by aligning
conversational-query embeddings with rewritten-
query embeddings, while preserving compatibil-
ity with existing document embedding spaces
which enables to use existing retrieval indexes.
Unlike prior conversational dense retrieval ap-
proaches (Lin et al., 2021; Yang et al., 2025b),
RCEM does not rely on conversation-document
relevance mappings during training. By learn-
ing alignment toward rewritten-query embeddings
rather than document relevance patterns, RCEM
improves generalization and robustness across re-
trieval distributions. Acquiring high-quality con-
versational query-to-document relevance mappings
is often expensive and challenging, limiting scal-
able training. In contrast, RCEM does not require
such annotations, making it easier to scale training
and improve generalization.
Extensive experiments demonstrate that RCEM
consistently outperforms strong conversational re-
trieval baselines, with particularly strong gains un-
der challenging distribution-shift settings.
Our contributions are summarized as follows:
•We propose RCEM, a conversational embed-
ding model that distills LLM-based query
rewriting capability into the embedding space,
enabling conversational retrieval without ex-
plicit query rewriting step before embedding.
•Unlike existing conversational dense retrieval
methods that optimize direct conversation-to-
document matching, RCEM learns alignment
between conversational inputs and rewritten-
query embeddings, improving robustness un-
der distributional shift. Extensive experiments
show that RCEM achieves state-of-the-art con-
versational retrieval performance.
•RCEM preserves compatibility with exist-
ing retrieval infrastructures, including pre-
computed document embeddings and retrieval
indexes, without requiring document-side re-
encoding.2 Related Works
Prior conversational search research can be broadly
categorized into two streams: conversational query
reformulation (CQR) and conversational dense re-
trieval (CDR). CQR methods rewrite conversa-
tional utterances into standalone queries before
retrieval while CDR methods directly encode con-
versational context into dense representations for
retrieval.
Conversational Query Reformulation.Con-
versational query reformulation methods trans-
form conversational queries into standalone search
queries that can be directly consumed by sparse
retrievers or dense retrievers after embedding. Mo
et al. (2023) propose ConvGQR, a genera-
tive reformulation framework for conversational
search. Mao et al. (2023a) introduce search-
oriented conversational query editing to better pre-
serve retrieval-oriented intent during reformulation.
With the rise of large language models, Mao et al.
(2023b) propose a prompting-based conversational
search framework that leverages LLMs for contex-
tual query understanding. Ye et al. (2023) further
enhance reformulation quality using LLM-aided
informative query rewriting. Mo et al. (2024a) pro-
pose CHIQ, which improves reformulation through
contextual history enhancement. Lai et al. (2025b)
introduce AdaCQR, which aligns sparse and dense
retrieval signals to improve query reformulation
effectiveness. (Lai et al., 2025a) introduces
a test-time adaptation framework that leverages
prompting-based candidate generation and reward-
guided selection to improve conversational query
rewriting quality. CQR approaches are more modu-
lar and retriever-agnostic, enabling easy integra-
tion with existing sparse or dense retrieval sys-
tems. They also naturally benefit from advances in
large language models and prompting techniques.
However, this approach incurs substantial compu-
tational cost and inference latency due to repeated
LLM calls, making it difficult to deploy for all user
requests at scale.
Conversational Dense Retrieval.Conversa-
tional dense retrieval methods aim to model depen-
dency across dialogue turns within the retriever it-
self. Lin et al. (2021) propose contextualized query
embeddings that incorporate conversation history
into dense retrieval representations for conversa-
tional search. Mo et al. (2024b) introduce history-
aware conversational dense retrieval, which explic-

itly models historical interactions for improved con-
text understanding. Mao et al. (2024) presents Cha-
tRetriever, which adapts large language models
for generalized conversational dense retrieval with
stronger robustness across search settings. Cheng
et al. (2024) further study the interpretability of
conversational dense retrieval through rewriting-
enhanced inversion of session embeddings. (Yang
et al., 2025b) proposes a contextual retrieval frame-
work that learns robust conversational representa-
tions by modeling dialogue context directly within
the retriever. These methods benefit from end-to-
end optimization and deep integration of conver-
sational context within retrieval representations.
However, existing conversational dense retrieval
methods often require specialized retriever train-
ing based on contrastive learning between con-
versational queries and ground-truth documents.
Acquiring high-quality conversation-to-document
relevance mappings is expensive and challenging,
which limits scalable training. Moreover, training
with limited conversational relevance data can eas-
ily lead to overfitting. Contrastive objectives may
also alter the original embedding space, reducing
compatibility with existing retrieval systems. In
contrast, RCEM does not require conversation-to-
document relevance supervision for training and is
explicitly designed to preserve the original embed-
ding space, improving generalization, scalability,
and system compatibility.
3 RCEM
We propose RCEM, a conversational embed-
ding model that directly maps conversational inputs
into the embedding space of rewritten standalone
queries. Unlike conventional conversational search
pipelines that rely on explicit LLM-based query
rewriting during inference, RCEM distills rewrit-
ing capability into an embedding model, enabling
low-latency conversational retrieval while preserv-
ing compatibility with existing retrieval indexes.
See an overview of the RCEM and comparison
with the prior work (Lin et al., 2021) in Figure 2.
Setting.Let qidenote the current query at dia-
logue turn i, and let q<i, a<idenote the previous
conversational history. Dgt
iis a groundtruth docu-
ment of the conversational query qi;q<i, a<i.Di
andqirepresent a general document and standalone
query respectively. Existing conversational searchsystems commonly use an LLM-based rewriter
Ri=LLM(q i;q<i, a<i),
where Riis a standalone rewritten query suitable
for retrieval. To distinguish conversational retrieval
from standard retrieval, we prepend a special token
[ST]to the conversational input:
ci= [ST, q i;q<ia<i].
We have two embedding functions Fθ(x)and
G(x), where,
Fθ(x) = MLP SELU(LoRA(G(x))).
G(x) is original embedder and Fθ(x)is built on
top of the frozen G(x) with LoRA adapters and a
lightweight two-layer perceptron head with SELU
activation. The backbone parameters are initialized
fromG(x) , while only the LoRA adapters and the
MLP head are updated during training.
Structure Learning.RCEM updates parameters,
θin embedderF θby minimizing loss below:
L(θ) =L point(θ) +λL pair(θ),
Lpoint(θ) =X
j∈Ω∥zj,1−zj,2∥
Lpair(θ) =X
j∈ΩX
k∈Ω\j(∥zj,1−zk,1∥ − ∥z j,2−zk,2∥),
where λcontrols the contribution of the pair-
wise structure term. We set λas1.unless oth-
erwise stated. zj,1isj-th embedding from Fθ(x)
where xis{ci, qi, Di} ∈Ω train andzj,2is cor-
responding j-th embedding from G(x) where x
is{Ri, qi, Di} ∈Ω train. By mapping conver-
sational query with special token toward rewrit-
ten query by LLM Fθ(ci)→G(R i), RCEM dis-
tills conversational query rewriting behavior into
the embedding space without explicitly generat-
ing rewritten text. Instead of generating Riduring
inference, RCEM directly encodes the conversa-
tional input into the embedding representation of
the rewritten query. Unlike prior methods (Lin
et al., 2021; Yang et al., 2025b), RCEM does not
require ground-truth document mappings Dgt
ifor
conversational queries (qi;q<ia<i). This removes
the dependence on expensive and difficult-to-obtain
conversational relevance annotations, making the
training pipeline more scalable and improving ro-
bustness under distributional shift. While distilling

Embedding space, Changed
𝐹 𝑞𝑖;𝑞<𝑖𝑎<𝑖
𝐷𝑖𝑔𝑡𝐹(𝑞𝑖;𝑞<𝑖𝑎<𝑖)
𝐺
𝐺(𝐷𝑖𝑔𝑡)
𝑞𝑖;𝑞<𝑖 𝑎<𝑖 𝐸𝑚𝑏𝑞𝑖;𝑞<𝑖 𝑎<𝑖: Conversational search on existing index
𝑞𝑖 𝐸𝑚𝑏𝑞𝑖: Normal search on existing index 
with less performance contrastive
𝐹Embedding space, Changed
𝐹θ 𝑞𝑖;𝑞<𝑖𝑎<𝑖
𝐷𝑖𝑔𝑡𝐹(𝑞𝑖;𝑞<𝑖𝑎<𝑖)
𝐺
𝐺(𝐷𝑖𝑔𝑡)
𝑞𝑖;𝑞<𝑖 𝑎<𝑖 𝐸𝑚𝑏𝑞𝑖;𝑞<𝑖 𝑎<𝑖: Conversational search on existing indexcontrastive
𝐹θ
(a) Lin et al.<Train Time>
<Test Time>
𝐸𝑚𝑏 𝑞𝑖: Normal search on existing indexEmbedding space, Unchanged
𝐹θ [𝑆𝑇,𝑞𝑖;𝑞<𝑖𝑎<𝑖],𝑞𝑖,𝐷𝑖
𝑅𝑖=𝐿𝐿𝑀 (𝑞𝑖;𝑞<𝑖𝑎<𝑖),𝑞𝑖,𝐷𝑖 𝐺(𝑅𝑖)𝐹([𝑆𝑇,𝑞𝑖;𝑞<𝑖𝑎<𝑖])
𝐺
𝐹(𝐷𝑖)
𝐺(𝐷𝑖)<Train Time>
<Test Time>𝐹(𝑞𝑖)
𝐺(𝑞𝑖)𝑆𝐿 𝑆𝐿 𝑆𝐿
[𝑆𝑇,𝑞𝑖;𝑞<𝑖𝑎<𝑖] 𝐸𝑚𝑏 𝑅𝑖: Conversational search on existing index
𝐹θ𝑞𝑖
(b) Proposed RCEM
𝐺 
𝐹θ 
[𝑆𝑇,𝑞𝑖;𝑞<𝑖𝑎<𝑖] 
𝑞𝑖,𝐷𝑖 
𝐷𝑖𝑔𝑡 
𝑅𝑖 : Original embedder frozen
: Embedder trained from the original embedder, 𝐺 by our proposed RCEM method
: Current 𝑖−𝑡ℎ query, 𝑞𝑖 and prior conversations, 𝑞<𝑖𝑎<𝑖 with Special Token, ST prepended.
: General any queries and documents
: Groundtruth passages mapping to the conversation until 𝑖, 𝑞𝑖;𝑞<𝑖𝑎<𝑖
: Query rewriting 𝐿𝐿𝑀 (𝑞𝑖;𝑞<𝑖𝑎<𝑖) resultsFigure 2:RCEM overview and comparison with prior work: RCEM first precomputes embeddings using the
original frozon embedder G. Specifically, it embeds the LLM-rewritten query , generated from the current query
and conversation history using LLM Ri=LLM(q i;q<ia<i), as well as general queries qiand documents Di.
The a embedding model, denoted as Fθ, is then trained to map the conversational input with a special token,
Fθ([ST, q i;q<ia<i]), to the embedding of the rewritten query embedded by the frozen embedder G(R i)using the
proposed Structure Learning (SL) loss. During training, RCEM also aligns general queries and documents with their
precomputed embeddings, i.e., Fθ(qi)→G(q i)andFθ(Di)→G(D i), to preserve the original embedding space.
This design enables a single model to support both conversational search, triggered by the special token, and standard
search over the existing index. It also serves as a regularizer, making the model more robust to distributional shift.
In contrast, (Lin et al., 2021) trains the model to map the conversational query embedding Fθ(qi;q<ia<i)directly
to the ground-truth document G(Dgt
i), which can alter the embedding space. Unlike prior approaches, RCEM does
not require conversational-query-to-ground-truth-passage mappings, which are often difficult to obtain accurately.
This makes RCEM more scalable, as it reduces the need for expensive and high-quality relevance annotations
query rewriting skill into the embedder, RCEM also
attempts to preserve embedding space by mapping
Fθ(qi)→G(q i)andF θ(Di)→G(D i).
This design offers several advantages. First,
RCEM eliminates expensive LLM-based rewrit-
ing during inference, substantially reducing la-
tency and computational cost. Second, RCEM
does not require conversational-query-to-document
relevance supervision, making training more scal-
able. Finally, because RCEM preserves the origi-
nal embedding space, it remains compatible with
existing document embeddings and prebuilt re-
trieval indexes without requiring document-side
re-encoding.4 Experiments
Experiment setting.For our experiments, we
use widely adopted conversational search bench-
marks: QReCC (Anantha et al., 2021), Top-
iOCQA (Adlakha et al., 2022), and TREC
CAsT (Dalton et al., 2020, 2021). QReCC and
TopiOCQA provide both training and test splits,
while the TREC CAsT datasets are used only for
evaluation under a zero-shot setting. QReCC con-
tains approximately 14K conversations and 80K
question–answer pairs, with retrieval performed
over a large collection of about 54M passages. Top-
iOCQA consists of 3.9K topic-switching conversa-
tions and 51K question–answer pairs, paired with
a passage corpus of roughly 25M passages. In

Figure 3:In domain performance comparison: We
train model using QReCC train and evaluate the perfor-
mance with test data in the same domain with the same
setups for fair comparison. Our proposed method out-
performs Lin et al. (2021) by 3.0% in Recall@10, 2.7%
in MRR, and 2.5% in NDCG@3. Note that QReCC
has an index size of approximately 54M passages, and
Qwen3-0.6B is used as the base embedder.
addition, we evaluate on TREC CAsT datasets,
which are designed to assess complex conversa-
tional search scenarios and are commonly used
for zero-shot evaluation. These datasets allow us
to examine the robustness of RCEM across both
in-domain and distributionally shifted conversa-
tional retrieval settings. For consistent evalua-
tion, we followed conventions of prior works (Lai
et al., 2025a; Yang et al., 2025b; Lin et al., 2021).
We report NDCG@3, MRR and Recall@10 using
pytrec_eval(Van Gysel and de Rijke, 2018).
Conversational search methods can be broadly
grouped into two categories. The first follows a
two-step pipeline: an LLM rewrites the conversa-
tional query, and an embedder then retrieves docu-
ments. Existing work improves either the rewriting
model during training or the rewritten query qual-
ity at test time. Although effective, these methods
are costly and slow because they require an ad-
ditional LLM-based rewriting step. The second
category directly embeds the conversation in a sin-
gle inference step, enabling efficient dense retrieval.
However, these methods typically do not generate
explicit query text, which limits their applicability
to sparse retrieval without additional modifications.
Our work belongs to this direct-embedding line.
The main prior studies in this direction are Lin
et al. (2021) and Yang et al. (2025b); in our ex-
periments, we primarily compare against Lin et al.
(2021).
Unless otherwise stated, we use Qwen3-0.6B(Yang et al., 2025a) as the base embedder. For
RCEM training, we added LoRA and a two-layer
perceptron with SELU activation (Klambauer et al.,
2017) is added on top of the base model. All mod-
els are trained with a batch size of 256 on 8 H100
GPUs. RCEM uses GPT-4.1 as the LLM to gener-
ate the training data with a simple query-rewriting
prompt. Unlike prior methods (Lin et al., 2021;
Yang et al., 2025b), RCEM does not require map-
pings between conversational queries and ground-
truth documents, which are often difficult to obtain
with high quality. Instead, it only requires LLM-
generated rewritten queries from conversational in-
puts. This provides convenient and low-cost train-
ing labels, enabling RCEM to scale more easily to
large-scale training.
In domain experiments.We conduct in-domain
experiments using the QReCC dataset for both
training and testing. For a fair comparison, we keep
most settings identical, including the hardware
setup, baseline model (Qwen3 0.6B), adaptation
architecture, and number of training epochs. The
only difference is the training objective. Notably,
RCEM does not require conversation-to-ground-
truth passage mappings, which are difficult to ob-
tain accurately, making the training process more
scalable. In contrast, Lin et al. (2021) relies on such
mappings because it uses contrastive loss, limiting
scalability. As shown in Figure3, our method out-
performs Lin et al. (2021) by 3.0% in Recall@10,
2.7% in MRR, and 2.5% in NDCG@3. We observe
that contrastive training is highly sensitive: even
a small increase in the number of epochs can sub-
stantially degrade performance. In contrast, our
proposed method remains more stable with various
epochs.
Distributional shift experiments.In the distri-
butional shift experiments,Lin et al. (2021) exhibits
a pronounced overfitting issue, whereas RCEM
maintains robust performance even under domain
shift. Figure4 reports the results when models are
trained on one dataset and evaluated on another un-
der the same experimental setup for a fair compari-
son. RCEM consistently outperforms the baseline
across all dataset combinations and evaluation met-
rics. In particular, when Lin et al. (2021) is trained
on QReCC, its performance is substantially lower
than that of RCEM, with gaps often ranging from
10% to 20%.
This performance degradation is likely due to

Figure 4:Domain shift performance comparison: We train the models on one dataset and evaluate them on
another under the same experimental setup to ensure a fair comparison. The proposed RCEM method maintains
strong performance under distributional shift, whereas the performance of Lin et al. (2021) drops noticeably. In
particular, when trained on QReCC and tested on TREC CAsT 2020, RCEM achieves approximately 20% higher
Recall@10 than Lin et al. (2021). Across all dataset combinations and evaluation metrics, RCEM consistently
outperforms Lin et al. (2021).
the contrastive objective used by Lin et al. (2021),
which directly maps conversational queries to pas-
sages. This task is more complex and can cause
the model to overfit to the training dataset. In con-
trast, RCEM maps conversational inputs to their
rewritten queries, which provides a simpler learn-
ing target while preserving the original embedding
space. Preserving the embedding space serves as
an effective regularizer, helping prevent overfitting
and improving robustness to distributional shift.
Preserving Original Embedding Space.We
evaluate whether RCEM preserves the original em-
bedding space. Specifically, we encode rewritten
queries using RCEM and perform dense retrieval
on an index built from embeddings generated by
the original embedder. We then compare retrieval
performance against the original embedder and the
prior conversational dense retrieval method (Lin
et al., 2021) on the QReCC dataset. The results
show that RCEM preserves retrieval performance
on the original embedding index, whereas Lin
et al. (2021) degrades retrieval effectiveness due to
changes in the embedding space.
Figure 5:Preserving Original Embedding Space:
General query is embedded by RCEM and run retrieval
on index built from the original embedding. RCEM
preserves retrieval performance similar to the original
embedder.
5 Conclusion
We presented RCEM, a conversational dense re-
trieval model that maps conversational inputs
into the embedding space of rewritten standalone
queries without explicit LLM-based rewriting dur-
ing inference. Unlike prior methods that learn di-
rect conversation-to-document matching, RCEM
aligns conversational embeddings with rewritten-
query embeddings while preserving the original
embedding space. As a result, RCEM does not

require expensive conversation-to-document rele-
vance supervision and remains compatible with ex-
isting retrieval indexes and document embeddings.
Extensive experiments demonstrate that RCEM
achieves strong retrieval performance and im-
proved robustness under distributional shift while
substantially reducing inference latency and com-
putational cost.
Limitations
The main bottleneck of the proposed method is the
LLM’s query rewriting capability. Although an
embedder trained with RCEM can sometimes out-
perform the two-step baseline (LLM-based query
rewriting followed by embedding), likely because
it learns a stronger latent representation than the
explicit rewritten text, its performance still depends
on the quality of query rewriting. Therefore, im-
proving the LLM’s rewriting ability during training
or test time, by leveraging prior rewriting meth-
ods, can further enhance the performance of the
RCEM-trained embedder.
References
Vaibhav Adlakha, Shehzaad Dhuliawala, Kaheer Sule-
man, Harm de Vries, and Siva Reddy. 2022. Top-
iOCQA: Open-domain conversational question an-
swering with topic switching.Transactions of the
Association for Computational Linguistics, 10:468–
483.
Raviteja Anantha, Svitlana Vakulenko, Zhucheng Tu,
Shayne Longpre, Stephen Pulman, and Srinivas
Chappidi. 2021. Open-domain question answering
goes conversational via question rewriting. InPro-
ceedings of the 2021 Conference of the North Amer-
ican Chapter of the Association for Computational
Linguistics: Human Language Technologies, pages
520–534, Online. Association for Computational Lin-
guistics.
Yiruo Cheng, Kelong Mao, and Zhicheng Dou. 2024. In-
terpreting conversational dense retrieval by rewriting-
enhanced inversion of session embedding. InPro-
ceedings of the 62nd Annual Meeting of the Associa-
tion for Computational Linguistics (Volume 1: Long
Papers), pages 2879–2893, Bangkok, Thailand. As-
sociation for Computational Linguistics.
Jeffrey Dalton, Chenyan Xiong, and Jamie Callan.
2020. Cast 2019: The conversational assistance track
overview. InIn Proceedings of TREC.
Jeffrey Dalton, Chenyan Xiong, and Jamie Callan.
2021. Cast 2020: The conversational assistance track
overview. InIn Proceedings of TREC.Günter Klambauer, Thomas Unterthiner, Andreas Mayr,
and Sepp Hochreiter. 2017. Self-normalizing neural
networks.Advances in neural information process-
ing systems, 30.
Yilong Lai, Jialong Wu, Zhenglin Wang, and Deyu
Zhou. 2025a. AdaRewriter: Unleashing the power of
prompting-based conversational query reformulation
via test-time adaptation. InProceedings of the 2025
Conference on Empirical Methods in Natural Lan-
guage Processing, pages 3889–3905, Suzhou, China.
Association for Computational Linguistics.
Yilong Lai, Jialong Wu, Congzhi Zhang, Haowen Sun,
and Deyu Zhou. 2025b. Adacqr: Enhancing query
reformulation for conversational search via sparse
and dense retrieval alignment. InProceedings of
the 31st International Conference on Computational
Linguistics, pages 7698–7720, Abu Dhabi, UAE. As-
sociation for Computational Linguistics.
Sheng-Chieh Lin, Jheng-Hong Yang, and Jimmy Lin.
2021. Contextualized query embeddings for conver-
sational search. InProceedings of the 2021 Confer-
ence on Empirical Methods in Natural Language Pro-
cessing, pages 1004–1015, Online and Punta Cana,
Dominican Republic. Association for Computational
Linguistics.
Kelong Mao, Chenlong Deng, Haonan Chen, Fengran
Mo, Zheng Liu, Tetsuya Sakai, and Zhicheng Dou.
2024. Chatretriever: Adapting large language mod-
els for generalized and robust conversational dense
retrieval. InProceedings of the 2024 Conference on
Empirical Methods in Natural Language Processing,
pages 1227–1240, Miami, Florida, USA. Association
for Computational Linguistics.
Kelong Mao, Zhicheng Dou, Bang Liu, Hongjin Qian,
Fengran Mo, Xiangli Wu, Xiaohua Cheng, and Zhao
Cao. 2023a. Search-oriented conversational query
editing. InFindings of the Association for Compu-
tational Linguistics: ACL 2023, pages 4160–4172,
Toronto, Canada. Association for Computational Lin-
guistics.
Kelong Mao, Zhicheng Dou, Fengran Mo, Jiewen Hou,
Haonan Chen, and Hongjin Qian. 2023b. Large lan-
guage models know your contextual search intent:
A prompting framework for conversational search.
InFindings of the Association for Computational
Linguistics: EMNLP 2023, pages 1211–1225. Asso-
ciation for Computational Linguistics.
Fengran Mo, Abbas Ghaddar, Kelong Mao, Mehdi Reza-
gholizadeh, Boxing Chen, Qun Liu, and Jian-Yun Nie.
2024a. Chiq: Contextual history enhancement for
improving query rewriting in conversational search.
InProceedings of the 2024 Conference on Empiri-
cal Methods in Natural Language Processing, pages
2253–2268, Miami, Florida, USA. Association for
Computational Linguistics.
Fengran Mo, Kelong Mao, Yutao Zhu, Yihong Wu,
Kaiyu Huang, and Jian-Yun Nie. 2023. Convgqr:

Generative query reformulation for conversational
search. InProceedings of the 61st Annual Meeting of
the Association for Computational Linguistics (Vol-
ume 1: Long Papers), pages 4998–5012, Toronto,
Canada. Association for Computational Linguistics.
Fengran Mo, Chen Qu, Kelong Mao, Tianyu Zhu, Zhan
Su, Kaiyu Huang, and Jian-Yun Nie. 2024b. History-
aware conversational dense retrieval. InFindings of
the Association for Computational Linguistics: ACL
2024, pages 13366–13378, Bangkok, Thailand. As-
sociation for Computational Linguistics.
Christophe Van Gysel and Maarten de Rijke. 2018.
Pytrec_eval: An extremely fast python interface to
trec_eval. InThe 41st International ACM SIGIR Con-
ference on Research & Development in Information
Retrieval, pages 873–876.
An Yang, Anfeng Li, Baosong Yang, Beichen Zhang,
Binyuan Hui, Bo Zheng, Bowen Yu, Chang Gao,
Chengen Huang, Chenxu Lv, et al. 2025a. Qwen3
technical report.arXiv preprint arXiv:2505.09388.
Seunghan Yang, Juntae Lee, Jihwan Bang, Kyuhong
Shim, Minsoo Kim, and Simyung Chang. 2025b.
Learning contextual retrieval for robust conversa-
tional search. InProceedings of the 2025 Conference
on Empirical Methods in Natural Language Process-
ing, pages 11991–12003, Suzhou, China. Association
for Computational Linguistics.
Fanghua Ye, Meng Fang, Shenghui Li, and Emine Yil-
maz. 2023. Enhancing conversational search: Large
language model-aided informative query rewriting.
InFindings of the Association for Computational Lin-
guistics: EMNLP 2023, pages 5985–6006, Singapore.
Association for Computational Linguistics.
A Prompts
We use a simple prompt to convert each conversa-
tional query into a rewritten query for generating
the training dataset, as shown below:
Given the recent conversation:
{conversation_history}
Rewrite the following ambiguous query
into a self-contained and more clear query.
This is the ambiguous query:
{query}
RCEM could potentially achieve even better
performance by incorporating improved prompts
or advanced query rewriting methods from prior
work (Lai et al., 2025a).