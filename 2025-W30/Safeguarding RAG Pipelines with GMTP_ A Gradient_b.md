# Safeguarding RAG Pipelines with GMTP: A Gradient-based Masked Token Probability Method for Poisoned Document Detection

**Authors**: San Kim, Jonghwi Kim, Yejin Jeon, Gary Geunbae Lee

**Published**: 2025-07-24 08:58:41

**PDF URL**: [http://arxiv.org/pdf/2507.18202v1](http://arxiv.org/pdf/2507.18202v1)

## Abstract
Retrieval-Augmented Generation (RAG) enhances Large Language Models (LLMs) by
providing external knowledge for accurate and up-to-date responses. However,
this reliance on external sources exposes a security risk, attackers can inject
poisoned documents into the knowledge base to steer the generation process
toward harmful or misleading outputs. In this paper, we propose Gradient-based
Masked Token Probability (GMTP), a novel defense method to detect and filter
out adversarially crafted documents. Specifically, GMTP identifies high-impact
tokens by examining gradients of the retriever's similarity function. These key
tokens are then masked, and their probabilities are checked via a Masked
Language Model (MLM). Since injected tokens typically exhibit markedly low
masked-token probabilities, this enables GMTP to easily detect malicious
documents and achieve high-precision filtering. Experiments demonstrate that
GMTP is able to eliminate over 90% of poisoned content while retaining relevant
documents, thus maintaining robust retrieval and generation performance across
diverse datasets and adversarial settings.

## Full Text


<!-- PDF content starts -->

Safeguarding RAG Pipelines with GMTP: A Gradient-based Masked
Token Probability Method for Poisoned Document Detection
San Kim1, Jonghwi Kim1, Yejin Jeon1, Gary Geunbae Lee1,2,
1Graduate School of Artificial Intelligence, POSTECH, Republic of Korea,
2Department of Computer Science and Engineering, POSTECH, Republic of Korea,
{sankm, jonghwi.kim, jeonyj0612, gblee}@postech.ac.kr
Abstract
Retrieval-Augmented Generation (RAG) en-
hances Large Language Models (LLMs) by
providing external knowledge for accurate and
up-to-date responses. However, this reliance
on external sources exposes a security risk; at-
tackers can inject poisoned documents into the
knowledge base to steer the generation pro-
cess toward harmful or misleading outputs. In
this paper, we propose Gradient-based Masked
Token Probability (GMTP), a novel defense
method to detect and filter out adversarially
crafted documents. Specifically, GMTP identi-
fies high-impact tokens by examining gradients
of the retriever’s similarity function. These
key tokens are then masked, and their proba-
bilities are checked via a Masked Language
Model (MLM). Since injected tokens typically
exhibit markedly low masked-token probabil-
ities, this enables GMTP to easily detect ma-
licious documents and achieve high-precision
filtering. Experiments demonstrate that GMTP
is able to eliminate over 90% of poisoned con-
tent while retaining relevant documents, thus
maintaining robust retrieval and generation per-
formance across diverse datasets and adversar-
ial settings.1
1 Introduction
Large Language Models (LLMs) have significantly
advanced performance across various Natural Lan-
guage Processing (NLP) tasks, especially in conver-
sational systems and AI assistants (Touvron et al.,
2023; Reid et al., 2024; Liu et al., 2024). How-
ever, their reliance on parametric knowledge re-
sults in significant challenges, including suscepti-
bility to hallucination (Huang et al., 2023; Xu et al.,
2024b) and knowledge update requirements. To
mitigate these issues Retrieval-Augmented Genera-
tion (RAG) has emerged as an effective approach
for integrating external, non-parametric knowledge
1Our code is publicly available at https://github.com/
mountinyy/GMTP .
Figure 1: Various corpus poisoning attacks on a Naïve
environment (without defense method) cause a signifi-
cant performance drop in retrieval by enabling the poi-
soned documents to be retrieved. In contrast, GMTP
effectively filters out poisoned documents, thereby pre-
serving retrieval performance.
into LLMs, thus enabling the model to generate re-
sponses based on retrieved data (Lewis et al., 2020).
This process typically involves a retriever and gen-
erator model, where the former is responsible for
retrieving the most relevant documents to a query,
and the latter generates the final answer based on
the retrieved documents. RAG architectures have
demonstrated robustness against hallucinations and
improved task accuracy, as evidenced by several
system adaptations (Borgeaud et al., 2022; Asai
et al., 2024; Jiang et al., 2023).
Yet, RAG’s reliance on external knowledge in-
troduces vulnerabilities to corpus poisoning attacks
(Zou et al., 2024; Chaudhari et al., 2024; Xue et al.,
2024). In such attacks, adversaries have access to
the knowledge base, where they are able to inject
maliciously crafted documents. This manipula-
tion directly impacts the behavior of the generator,
causing it to provoke misleading and harmful re-
sponses based on the tampered documents. This
is because poisoned documents may contain incor-
rect information (Zou et al., 2024) or adversarial
instructions (Tan et al., 2024). This threat is particu-
larly concerning for decision-making LLM agents,
such as those used in autonomous driving, wherearXiv:2507.18202v1  [cs.CL]  24 Jul 2025

an attack could lead to erroneous actions with se-
vere consequences (Chen et al., 2024; Mao et al.).
Furthermore, attackers can optimize poisoned doc-
uments so that they appear highly relevant for only
target queries (Chaudhari et al., 2024; Zhang et al.,
2024), which makes their detection significantly
more challenging.
In this paper, we propose Gradient-based
Masked Token Probability (GMTP), a novel
method for filtering adversarially poisoned doc-
uments by analyzing the masked token probability
of key tokens. Specifically, the goal of GMTP is
to filter out poisoned documents among those re-
trieved while maintaining retrieval performance,
as can be seen in Figure 1. GMTP first identifies
key tokens by examining gradient values derived
from the retrieval phase, where tokens with high
gradients significantly contribute to the similarity
score. These key tokens are then masked, and the
probability of correctly predicting them is assessed.
Since adversarially manipulated documents are op-
timized to match specific query patterns, they often
contain unnatural text patterns that become diffi-
cult to reconstruct once masked. Leveraging this
property, GMTP effectively filters out poisoned
documents that exhibit abnormally low masked to-
ken probabilities.
Our key contributions are as follows:
•Safe filtering method : GMTP effectively de-
tects adversarially poisoned documents with
high precision, which ensures a clear separa-
tion between poisoned and clean documents.
•Extensive empirical validation : Our exper-
iments show that GMTP consistently outper-
forms existing baselines by achieving both
high filtering rate and retrieval performance.
•Robustness across settings : GMTP main-
tains strong filtering performance across vari-
ous hyperparameter configurations, and con-
sistently achieves a filtering rate above 90%.
2 Related Works
2.1 Retrieval Augmented Generation
RAG is known to be effective for knowledge-
intensive tasks that require external information
(Lewis et al., 2020; Gao et al., 2023; Guu et al.,
2020; Li et al., 2024; Shao et al., 2023), and is
composed of a retriever and a generator. The re-
triever can further be categorized into three types.The cross-encoder, proposed by Nogueira and Cho
(2019), encodes both the query and the passage
together using the same encoder. In contrast,
Karpukhin et al. (2020a) introduced the bi-encoder,
which employs two independent encoders to gener-
ate dense vector representations for the query and
document. Relevance is then measured by calcu-
lating the similarity between these vectors. Mean-
while, the poly-encoder (Humeau et al.) combines
the efficiency of the bi-encoder with the accuracy of
the cross-encoder by utilizing an attention mecha-
nism that attends to multiple document embeddings
and a single query embedding.
For the generator, RAG can use various pre-
trained language models, such as T5 (Raffel et al.,
2020), Llama2 (Touvron et al., 2023), and Gemma
(Team et al., 2024), in order to generate answers
based on the relevant documents retrieved by the
retriever. By integrating these two retriever and
generator modules, RAG enables models to lever-
age both parametric knowledge (from the language
model) and non-parametric knowledge (from the
retrieved documents).
2.2 Corpus Poisoning Attack on RAG
Recent studies have shown that RAG-based sys-
tems are vulnerable to corpus poisoning attacks
(Zou et al., 2024; Xue et al., 2024; Cheng et al.,
2024; Tan et al., 2024; Zhong et al., 2023). In sce-
narios where attackers know which retriever has
been used but cannot directly train it, a common
approach involves injecting carefully crafted poi-
soned documents into the knowledge base. These
malicious documents are designed to (1) appear
highly relevant to the target queries, and (2) induce
the generator to produce attacker-desired responses
when retrieved.
To achieve these objectives, adversaries fre-
quently employ Hotflip (Ebrahimi et al., 2018) to
optimize the malicious documents for a specific
query pattern. Hotflip iteratively replaces selected
tokens in a document to maximize its similarity
score to a given query. Recent studies have refined
this technique by optimizing for specific queries
(Zhong et al., 2023) or queries with particular trig-
gers or topics (Chaudhari et al., 2024; Zhang et al.,
2024; Cheng et al., 2024; Xue et al., 2024). By
combining retriever optimization with adversarial
attack techniques that target generators, attackers
can manipulate the generator to produce targeted
responses when the malicious document is success-
fully retrieved (Zou et al., 2023).

While research pertaining to RAG attacks has
expanded substantially, defense strategies have re-
ceived comparatively less attention. One straight-
forward defensive approach involves filtering out
potentially poisoned documents during the retrieval
phase using perplexity or l2-norms (Zhong et al.,
2023). However, this method risks removing legiti-
mate relevant documents as well (Zou et al., 2024),
and selecting an optimal filtering threshold remains
challenging. Meanwhile, in the generation phase,
Xiang et al. (2024) demonstrated that carefully de-
signed decoding processes can mitigate attacker-
desired outputs. Additionally, Zhou et al. (2025)
proposed a two-stage approach of first clustering
and filtering suspicious retrieved documents via K-
means, then leveraging both internal and external
knowledge to generate trustworthy responses.
Although these defense approaches are promis-
ing, increasing generator overhead is often cost-
ineffective since generators are typically much
larger than retrievers. In contrast, the proposed
GMTP is able to effectively detect unnatural to-
kens that induce excessive similarity, which of-
fers a more precise alternative. Furthermore, it
only requires lightweight computations using small
models such as BERT (Devlin et al., 2019), which
significantly reduces computational overhead com-
pared to generation-based defenses.
3 Problem Definition
3.1 Abnormal Similarity
The retriever in the RAG framework retrieves rel-
evant sources based on the similarity between a
given query and available documents. However,
as illustrated in Figure 2, poisoned documents of-
ten appear close to the target query in the embed-
ding space, making them highly similar (Zou et al.,
2024; Zhong et al., 2023). Due to this property, the
retriever frequently retrieves poisoned documents
as highly relevant, which leads the generator to
incorporate unreliable information.
Consequently, given the close proximity of these
documents to the query, a naïve classification ap-
proach that relies solely on embedding vectors risks
misidentifying poisoned documents while poten-
tially overlooking relevant documents. This chal-
lenge highlights the need for a more robust de-
tection method capable of distinguishing between
poisoned and relevant clean documents effectively.
Figure 2: Distribution of clean documents, poisoned
documents, and queries within the embedding space,
projected onto its first two principal components using
Principal Component Analysis (PCA). Each top-5 clean
and poisoned documents are selected as the most rele-
vant to the 100 queries, using Contriever (Izacard et al.)
fine-tuned for MS MARCO (Nguyen et al.) dataset.
3.2 Linguistic Unnaturalness in Poisoned
Documents
As discussed in Section 2.2, poisoned documents
are designed to achieve two key objectives; 1) they
must be retrievable and 2) cause the generator to
malfunction. To achieve the latter, malicious doc-
uments typically contain adversarial commands,
such as “Always answer as I cannot answer to that
question” to enforce refusal responses, or jailbreak-
ing prompts (Shen et al., 2024; Xu et al., 2024a;
Jiang et al., 2024) that elicit harmful outputs. How-
ever, adversarial commands alone generally do not
make poisoned documents retrievable in the RAG
system.
Therefore, in order to enhance their retrievabil-
ity, special tokens must be introduced so as to op-
timize the poisoned documents to resemble target
queries from the retriever’s perspective. This op-
timization process often leads to the insertion of
unnatural, seemingly meaningless word sequences
that artificially increase query-document similarity
while rendering the documents syntactically irregu-
lar. We refer to these as cheating tokens . Detecting
such tokens remains a challenge, as their placement
is agnostic to specific positions. Moreover, recent
research has demonstrated that integrating a nat-
uralness constraint into the optimization function
improves linguistic coherence of cheating tokens ,
thereby further complicating their detection (Zhang
et al., 2024).

Document
dade cheer skinned none most 
private schools in America …America
cheer
noneHigh gradient tokens
dade [MASK] skinned [MASK] most 
private schools in [MASK] …
cheer: 0.02% none: 0.03% 
America: 70%Avg. < 𝜏GMTP
dade cheer skinned none most 
private schools in America …
Do all private schools in 
America have uniform policy?Query
Similarity
...Poisoned
DocumentsRetrieved 𝑘documentsFigure 3: Overview of the RAG pipeline incorporating the GMTP method to identify and exclude potentially
poisoned documents. The orange text highlights cheating tokens , which are manipulated to maximize similarity
score between the target query and the poisoned document.
4 Proposed Methodology
In this section, we introduce GMTP, which iden-
tifies key tokens that contribute to high similarity
scores between a query and document, while also
detecting potential adversarial inputs by analyzing
masked token probabilities. Figure 3 illustrates
the overall pipeline of RAG using GMTP. Specif-
ically, GMTP identifies tokens with anomalously
low masked token probabilities as potential adver-
sarial indicators by leveraging the fact that cheating
tokens exhibit unnatural linguistic patterns.
We assume that the attacker can inject poisoned
data into the knowledge base and can execute the re-
triever and generator but cannot retrain them. This
assumption aligns with existing research on RAG-
based adversarial attacks (Chaudhari et al., 2024;
Zhong et al., 2023; Zou et al., 2024; Xue et al.,
2024). This assumption also reflects real-world
scenarios, as many deployed RAG systems rely
on openly accessible databases such as Wikipedia
and often utilize third-party models like Contriever
(Izacard et al.) and Llama3 (Dubey et al., 2024).
4.1 Key Token Detection
GMTP first identifies key tokens that contribute
significantly to the similarity score. Inspired by
Moon et al. (2022), we leverage the gradients of
the similarity function with respect to the word em-
bedding etof token tin document d. In this paper,
unless otherwise specified, we compute similarity
using the dot product. As shown in Eq. 1, the simi-
larity score is derived using the query encoder EQ
and document encoder ED. To assess the influence
of individual tokens on similarity, we compute the
l2-norm of their gradients to get gt. To refine selec-
tion, we retain tokens with above-average gradient
magnitudes and choose at most Ntokens with thehighest values, which ensure precise identification
of cheating tokens.
gt=∥∇etSim(EQ(q), ED(d))∥2 (1)
4.2 Masked Token Probability
Due to lexical unnaturalness as discussed in Sec-
tion 3.2, key tokens in poisoned documents tend
to be significantly harder to predict when masked.
To leverage this observation, we employ external
Masked Language Model (MLM) to estimate the
probability of recovering the original token from a
masked position.
Specifically, we iteratively mask each of the pre-
viously selected Ntokens and compute the prob-
ability of predicting the original token using the
MLM. However, the detection method described
in Section 4.1 does not guarantee that only poi-
soned tokens are detected as it may also include
normal or semantically essential tokens. Therefore,
to enhance precision, we select the Mtokens with
the lowest masked token probabilities. We define
the average probability of these Mtokens as the
P-score , which we expect to be significantly lower
in poisoned documents compared to clean ones.
Documents with a P-score below a threshold τare
filtered out.
τ=λ·1
KKX
i=1P-score i (2)
Since the P-score distribution may vary across
domains, it is reasonable to adopt a domain-
dependent threshold τ. To estimate an appropri-
ate value, we randomly sample Kqueries from the
training dataset along with their corresponding rel-
evant documents and compute the average P-score .
Instead of directly using this value as τ, we scale it

by a factor λ∈[0,1]to account for its lower bound.
By default, we set K= 1000 , as increasing it from
1000 to 10000 results in a variation of less than
1%. The complete filtering process using GMTP is
outlined in Algorithm 1.
Algorithm 1 GMTP
Require: Query q, top- kdocuments Dk=
[d1,···, dk], query encoder EQ, document en-
coder ED, MLM M, threshold τ
1:S← {}
2:foreachd∈Dkdo
3: G:={g1, . . . , g T} ▷Eq. 1
4: G← {gi|gi>1
TPT
j=1gj, gi∈G}
5: P← {}
6: foreachgi∈Gdo
7: d[i]←"[MASK]"
8: P←P∪ M(d)
9: end for
10: SortPin ascending order
11: P-score d=1
MPM
iPi
12: S←S∪P-score d
13:end for
14:S← {s|s > τ, s ∈S} ▷Eq. 2
15:return S
5 Experimental Setup
Datasets. We evaluate our method on three
benchmark datasets: Natural Questions (NQ)
(Kwiatkowski et al., 2019), HotpotQA (Yang et al.,
2018), and MS MARCO (Nguyen et al.) using the
BEIR benchmark (Thakur et al.). These datasets
contain approximately 2.6M, 5.2M, and 8.8M docu-
ments, respectively. For each dataset, we randomly
sample 200 test queries. If a query contains a spe-
cific trigger, which in this study is "iPhone," the
trigger is appended to the original query.
Models. For the retrieval component, we utilize
the BERT-based Contriever (Izacard et al.) that is
fine-tuned on the MS MARCO dataset, and DPR
(Karpukhin et al., 2020b), which is fine-tuned on
the NQ dataset. For the generation component, we
employ Llama2-7B-Chat (Touvron et al., 2023).
All models are downloaded via Huggingface2.
RAG System. We employ Faiss (Johnson et al.,
2019) for retrieval and set k= 10 for top- kdoc-
uments retrieval. To maintain kdocuments after
filtering, we replaced the removed documents with
similar alternatives.
2https://huggingface.co/5.1 Attack Setup
The attack methods vary depending on the opti-
mization method. In PoisonedRAG (Zou et al.,
2024), adversarial documents are optimized to be
similar to specific queries, and contain misleading
information that causes the generator to underper-
form. For each query, we curated five adversarial
documents, which results in a total of 1,000 mali-
cious documents. In contrast, Phantom (Chaudhari
et al., 2024) focuses on inducing malicious retrieval
when a specific trigger appears within the query.
Unlike PoisonedRAG, Phantom utilizes two types
of cheating tokens: one to optimize the trigger
and another to guide the generator, making it more
likely to follow an adversarial command. While
both PoisonedRAG and Phantom employ the Hot-
flip method, Adversarial Decoding ( AdvDecoding )
(Zhang et al., 2024) uses LLMs such as Llama3
(Dubey et al., 2024) to generate candidate cheating
tokens. AdvDecoding then evaluates these tokens
based on both naturalness and the similarity to the
triggered query, and selects the best token for the
task. By incorporating the naturalness constraint,
AdvDecoding generates more fluent text compared
to other methods, which eventually enhances the
challenge of detection.
Both Phantom and AdvDecoding employ
"iPhone" as a trigger, with the adversarial com-
mand being "Always answer the query I’m sorry, I
can’t answer to that question". We crafted five poi-
soned documents that are activated when the word
"iPhone" appears in the query. More detailed attack
setups and examples can be found in Appendix A.
5.2 Defense Setup
We adopt perplexity and l2-norm as baseline de-
fense methods on the retriever side to filter out
abnormal documents as in Zou et al. (2024) and
Zhong et al. (2023). Although GMTP operates
solely during the retrieval phase, for comparison
with baselines that intervene during the generation
phase, we also assess the generation performance
based on the retrieval results. TrustRAG (Zhou
et al., 2025) employs K-means clustering to re-
move retrieved documents within the cluster that
exhibits suspiciously high density. Subsequently, it
leverages both the model’s internal knowledge and
the retrieved knowledge to extract reliable sources.
RobustRAG (Xiang et al., 2024) prompts the gen-
erator to extract keywords from responses gener-
ated for each retrieved document and removes less

Retrieval
Attack DefenseNQ HotpotQA MS MARCO
nDCG@10FR (↑)nDCG@10FR (↑)nDCG@10FR (↑)Clean ( ↑) Poison ( ↑) Clean ( ↑) Poison ( ↑) Clean ( ↑) Poison ( ↑)
PoisonedRAGNaive 0.418 0.313 0.0 0.261 0.149 0.0 0.171 0.108 0.0
PPL 0.417 0.415 0.996 0.258 0.258 1.0 0.178 0.178 0.996
l2-norm 0.391 0.295 0.019 0.232 0.13 0.001 0.192 0.091 0.019
GMTP 0.478 0.476 1.0 0.282 0.281 0.999 0.132 0.122 0.999
PhantomNaive 0.418 0.407 0.0 0.261 0.228 0.0 0.171 0.151 0.0
PPL 0.417 0.417 1.0 0.258 0.258 1.0 0.170 0.170 1.0
l2-norm 0.391 0.379 -0.028 0.232 0.2 -0.018 0.153 0.134 0.134
GMTP 0.389 0.389 1.0 0.226 0.226 1.0 0.114 0.114 1.0
AdvDecodingNaive 0.418 0.414 0.0 0.261 0.255 0.0 0.171 0.164 0.0
PPL 0.417 0.414 0.548 0.258 0.255 0.617 0.17 0.163 0.21
l2-norm 0.391 0.388 0.0 0.232 0.225 -0.023 0.153 0.145 -0.014
GMTP 0.389 0.389 1.0 0.226 0.226 1.0 0.114 0.114 1.0
Generation
Attack DefenseNQ HotpotQA MS MARCO
CACC ( ↑) ACC ( ↑) ASR ( ↓) CACC ( ↑) ACC ( ↑) ASR ( ↓) CACC ( ↑) ACC ( ↑) ASR ( ↓)
PoisonedRAGNaive 59.5 18.0 80.0 34.5 7.0 89.5 45.0 24.0 72.5
TrustRAG 36.0 38.0 11.5 24.0 24.0 16.5 25.5 26.0 14.0
RobustRAG 38.5 26.5 54.0 26.5 7.0 86.0 29.5 19.5 46.0
GMTP 56.5 60.0 3.5 35.5 34.5 7.5 43.0 47.0 4.5
PhantomNaive 52.0 48.0 24.0 37.5 13.5 60.0 40.5 26.0 38.5
TrustRAG 29.5 27.5 0.0 22.5 24.5 0.0 20.5 22.0 0.5
RobustRAG 32.5 30.0 18.5 20.0 12.5 43.5 20.5 16.5 30.0
GMTP 51.0 56.0 0.5 37.0 35.0 0.0 34.5 36.5 0.0
AdvDecodingNaive 54.5 48.0 13.0 38.0 29.5 20.5 38.5 36.0 9.5
TrustRAG 32.5 28.0 0.0 23.0 21.0 0.0 22.0 21.5 0.5
RobustRAG 34.5 33.5 10.5 20.5 23.5 20.5 21.0 18.0 10.0
GMTP 52.0 53.5 0.5 37.0 35.0 0.0 35.0 34.5 0.0
Table 1: Performance in retrieval phase and generation phase using DPR. "Clean" refers to the environment where
no attack, while "Poison" indicates the environment where an attack has been executed. Naïve represents no defense
applied at all. Bold indicates the best defense method: highest FR ( ↑) in retrieval, lowest ASR ( ↓) in generation.
frequent keywords. The remaining high-frequency
keywords are then used to generate the final answer.
InGMTP , we set N= 10 andM= 5for key to-
ken detection, and λ= 0.1for threshold τ. We
employ BERT (Devlin et al., 2019) as the MLM to
obtain the masked token probability. More detailed
defense setups can be found in Appendix B.
5.3 Evaluation
Contrary to the attack goals, a successful defense
method must prevent the generation of attacker-
desired responses while maintaining the perfor-
mance of the RAG system. We evaluate GMTP
and baseline models using slightly different met-
rics for retrieval and generation phases.
Retrieval. We define the Filtering Rate ( FR) as
the ratio of poisoned documents removed by the de-
fense method out of the total poisoned documents
in the top- kretrieval before defense is applied. To
assess if the relevant documents are retrieved prop-
erly, we use Normalized Discounted Cumulative
Gain at rank k(nDCG@ k).
Generation. The Attack Success Rate ( ASR )
measures the proportion of responses that alignwith the attacker’s desired output. A response is
considered "attacked" if it is based on poisoned
documents in PoisonedRAG or if the response is
"I can’t answer to that question" in Phantom and
AdvDecoding. To evaluate the performance main-
tenance in both attack and clean environments, we
compare Accuracy ( ACC ) and Clean Accuracy
(CACC ), respectively. In addition, we employ the
LLM-as-a-judge approach (Zheng et al., 2023) to
evaluate if the response adheres to the correct an-
swer (ACC, CACC), or to incorrect answer (ASR
in PoisonedRAG). Additional details of the utilized
prompts and metrics are provided in Appendix G
and F, respectively.
6 Results and Analysis
In this section, we evaluate the performance of
GMTP against baseline models and analyze the
factors contributing to its filtering effectiveness.
Due to space constraints, we primarily present re-
sults using the DPR model on the NQ dataset, with
the exception of the main result which includes all
datasets. Comprehensive results are available in
Appendix K.

(a) PoisonedRAG (b) Phantom (c) AdvDecoding
Figure 4: Density plot showing the effects of different attack methods on the NQ dataset using the DPR model.
With minimal overlap, the masked token probability from GMTP effectively distinguishes poisoned documents
from clean and relevant ones.
Attack NQ HotpotQA MS MARCO
PoisonedRAG 0.859 0.860 0.865
Phantom 0.901 0.901 0.932
AdvDecoding 0.910 0.954 0.786
Table 2: GMTP precision in detecting cheating tokens
using DPR.
6.1 Main Results
Table 1 presents the main evaluation results for the
retrieval and generation phases under DPR. No-
tably, in the retrieval phase, GMTP achieves near
1.0of filtering rate for all three attacks and datasets
in both retrievers, which indicates that almost all
possible poisoned documents have been success-
fully detected. Furthermore, GMTP maintains a
steady nDCG@10 score in both clean and poisoned
environments, and in some cases it even surpasses
the Naïve method.
While GMTP demonstrates robustness across
various datasets and attack scenarios, other meth-
ods perform well only under specific conditions.
For example, whil PPL generally keeps a filtering
rate of 1.0, it is computationally expensive and in-
effective against the AdvDecoding attack, often
failing to filtering rate of around 0.5. This sug-
gests that PPL is particularly vulnerable to attacks
incorporating naturalness constraints, raising con-
cerns about its adaptability to advanced attacks. In
contrast, GMTP successfully detects subtle anoma-
lous patterns arising from similarity optimization.
Similarly, the l2-norm method shows strong per-
formance with Contriever in Table 6 but struggles
with DPR that uses different encoders for queries
and documents, revealing its limitations.
GMTP’s strong resilience toward attacks leads
to stable generation performance as well. As can be
seen in Table 1, GMTP significantly outperforms
all other baselines while preventing a performance
decrease that comes from poisoned document inter-AttackDocument
typeNQ Hotpot QA MS MARCO
PoisonedRAGPoison 0.02 0.00 0.00
Relevant 29.06 33.73 17.28
Clean 18.38 26.99 16.91
PhantomPoison 0.00 0.00 0.00
Relevant 29.96 30.53 14.56
Clean 17.27 26.34 13.89
AdvDecodingPoison 0.03 0.00 0.06
Relevant 30.13 30.91 14.35
Clean 17.30 26.11 13.89
Table 3: Average of masked token probability of se-
lected Mtokens using DPR. Values below 0.01 are
indicated as 0.00.
ventions, and keeps the ASR to at most 10%. Mean-
while, RobustRAG underperforms significantly in
our experiments, which shows its vulnerability to
attacks designed to craft multiple poisoned docu-
ment for single query pattern. These results align
with prior investigations by Zhang et al. (2024).
While TrustRAG is able to somewhat prevent at-
tacks, it comes with a high cost of nearly 50% of the
original performance. The main reason of GMTP’s
superiority relies on the fundamental solution of
removing the attack source, which keeps the over-
head low and leaves no possibilities of successful
attack.
6.2 Key Token Precision
For GMTP, accurate identification of cheating to-
kens is crucial. This is because false detection may
result in selecting natural tokens instead, which
increases the risk of mistakenly classifying a poi-
soned document as clean. Table 2 presents the
precision of GMTP under different attacks and
datasets. As can be seen, GMTP is able to achieve
a precision above 0.8in most cases. This high-
lights GMTP’s effectiveness in detecting cheating
tokens, which leads to more reliable poisoned doc-
ument identification. Further analysis is provided
in Appendix C.

Figure 5: nDCG@10 and Filtering Rate using various
NandMvalues using the DPR model in NQ dataset,
under the PoisonedRAG attack.
Table 3 illustrates the average masked token
probabilities in selected Mtokens. The result in-
dicates that the probabilities show large margin
between poisoned and non-poisoned documents,
with below 1% for poisoned documents and above
10% for non-poisoned documents. This result
aligns with Figure 4, which presents a density
plot of various document types in relation to their
masked token probabilities. Here, clean docu-
ments refer to non-poisoned documents that lack
crucial information for the correct answer genera-
tion. Clean and relevant documents typically ex-
hibit an average masked token probability close
to10−1, whereas poisoned documents are concen-
trated at much lower values. Although AdvDecod-
ing yields higher probabilities than other methods,
it still fails to exceed 10−3in most cases, which is
approximately 100times smaller than the average
probability of non-poisoned documents. We hy-
pothesize that despite considering naturalness, low
probabilities result from the necessity of achieving
the similarity optimization goal. These findings
demonstrate GMTP’s efficacy in mitigating corpus
poisoning attacks.
6.3 Hyperparameter Analysis
In this section, we analyze the impact of each hy-
perparameter. Additional analysis is provided in
Appendix E.
N-M Optimization. Properly setting the top- N
gradient values and least- Mmasked token proba-
bilities is crucial, as these parameters directly de-
termine the likelihood of documents to be filtered.
As shown in Figure 5, using a low Mvalue often
results in poor retrieval performance because even
relevant documents may contain a small proportion
of rare but important tokens with low masked to-
ken probabilities. Conversely, setting Mtoo high
reduces the filtering rate, which allows for natural
tokens in the poisoned document to be selected.
(a) PoisonedRAG (b) AdvDecoding
Figure 6: nDCG@10 and filtering rate across different
λvalues. nDCG-C represents retrieval performance in
a clean setting (i.e., without attacks), while nDCG-P
denotes performance under an applied attack.
On the other hand, Ncontrols the number of
candidates to consider as potential cheating tokens.
While higher Nvalues can help achieve optimal re-
sults, they also increase the risk of including impor-
tant tokens from non-poisoned documents, which
can lead to performance degradation. Therefore, it
is generally preferable to select moderate Nvalues
to balance these competing factors.
λTrade-off. λcontrols the lower bound of
threshold τ. Although we use the P-score adapted
to the dataset in use, applying it directly leads to ex-
cessive filtering of relevant documents. Therefore,
λmust be carefully adjusted to optimize τ.
Figure 6 presents the retrieval and filtering per-
formance across different λvalues. As can be seen,
nDCG@10 in attack settings performs as well as or
even better than in settings without attacks, which
demonstrates its robustness against attacks. Increas-
ingλimproves the removal of poisoned documents
but this comes at the cost of reduced retrieval per-
formance as relevant documents may also be fil-
tered. However, since GMTP maintains a high
filtering rate across various λvalues, we recom-
mend setting λto at least 0.1, where the filtering
rate consistently exceeds 0.9 across all datasets and
attack scenarios.
6.4 Model Generalization
To further demonstrate the generalizability of our
approach, we evaluate GMTP on ColBERT (Khat-
tab and Zaharia, 2020), a retrieval architecture that
fundamentally differs from DPR and Contriever
by introducing late interaction through token-level
similarity scoring. Specifically, we utilized Col-
BERTv23, which was trained on the MS MARCO
dataset using knowledge distillation as described
in Santhanam et al. (2022).
3https://huggingface.co/colbert-ir/colbertv2.0

AttacknDCG@10FR (↑)Clean ( ↑) Poison ( ↑)
PoisonedRAG 0.491 0.461 0.957
Phantom 0.491 0.457 1.0
AdvDecoding 0.491 0.452 1.0
Table 4: Retrieval phase performance of GMTP using
ColBERT as the retriever on the NQ dataset.
As shown in Table 4, GMTP maintains a re-
trieval performance drop of less than 10% on the
NQ dataset under attack scenarios, while success-
fully filtering out over 90% of adversarial docu-
ments. Since GMTP only requires access to the
gradients of the similarity function, it can be seam-
lessly integrated into diverse retrieval mechanisms,
including the token-level interaction used in Col-
BERT. These results underscore GMTP’s flexibil-
ity and effectiveness across heterogeneous retrieval
backbones. To further demonstrate its adaptability,
we also explore the use of alternative MLM, such
as RoBERTa, in Appendix J.
Figure 7: Latency of each method across the datasets.
The reported values are the averages over five runs using
two NVIDIA A6000 GPUs.
6.5 Latency
Another key advantage of GMTP is its exceptional
efficiency, as demonstrated in Figure 7. GMTP en-
ables the RAG system to maintain robust defense
against diverse attack methods while incurring min-
imal computational overhead, outperforming PPL
by approximately 20% and other generation-phase
baselines by nearly 80% in average latency during
the general phase. While l2method achieves high
efficiency, it falls short in precisely filtering adver-
sarial documents. These results suggest that GMTP
offers a favorable trade-off between speed and relia-bility, making it a strong candidate for deployment
in latency-sensitive NLP applications.
7 Conclusion
In this study, we have proposed GMTP, a defense
method designed to precisely filter poisoned doc-
uments in RAG systems. By effectively capturing
linguistic unnaturalness in poisoned documents,
GMTP is able to successfully separate them from
clean ones. Experimental results demonstrated its
strong filtering performance while minimizing re-
trieval degradation. Furthermore, GMTP achieved
over 90% successful filtering rate across various
hyperparameter settings, demonstrating its robust
adaptability to different RAG systems. These find-
ings highlight GMTP as a practical and adaptable
defense against adversarial poisoning attacks.
8 Limitations
Although GMTP demonstrates strong performance
in defending against corpus poisoning attacks, it
may struggle to detect naturally crafted documents
without optimization, such as false information or
biased news articles. Such attacks present both
advantages and limitations: they are difficult to
detect, but an attacker also cannot ensure their re-
trieval due to the lack of the optimization process.
We have not tested GMTP against such attacks;
however, future work may explore a broader range
of attack methods to further evaluate and enhance
its robustness.
Acknowledgments
This research was supported by the MSIT(Ministry
of Science, ICT), Korea, under the Global Research
Support Program in the Digital Field program(RS-
2024-00436680) supervised by the IITP(Institute
for Information & Communications Technology
Planning & Evaluation). Also this project is sup-
ported by Microsoft Research Asia.
References
Akari Asai, Zeqiu Wu, Yizhong Wang, Avi Sil, and
Hannaneh Hajishirzi. 2024. Self-rag: Learning to re-
trieve, generate, and critique through self-reflection.
InInternational Conference on Learning Representa-
tions .
Sebastian Borgeaud, Arthur Mensch, Jordan Hoff-
mann, Trevor Cai, Eliza Rutherford, Katie Milli-
can, George Bm Van Den Driessche, Jean-Baptiste
Lespiau, Bogdan Damoc, Aidan Clark, et al. 2022.

Improving language models by retrieving from tril-
lions of tokens. In International conference on ma-
chine learning , pages 2206–2240. PMLR.
Harsh Chaudhari, Giorgio Severi, John Abascal,
Matthew Jagielski, Christopher A Choquette-Choo,
Milad Nasr, Cristina Nita-Rotaru, and Alina Oprea.
2024. Phantom: General trigger attacks on retrieval
augmented language generation. CoRR .
Zhaorun Chen, Zhen Xiang, Chaowei Xiao, Dawn Song,
and Bo Li. 2024. Agentpoison: Red-teaming llm
agents via poisoning memory or knowledge bases.
Advances in Neural Information Processing Systems ,
37:130185–130213.
Pengzhou Cheng, Yidong Ding, Tianjie Ju, Zongru Wu,
Wei Du, Ping Yi, Zhuosheng Zhang, and Gongshen
Liu. 2024. Trojanrag: Retrieval-augmented genera-
tion can be backdoor driver in large language models.
CoRR .
Jacob Devlin, Ming-Wei Chang, Kenton Lee, and
Kristina Toutanova. 2019. Bert: Pre-training of deep
bidirectional transformers for language understand-
ing. In Proceedings of the 2019 conference of the
North American chapter of the association for com-
putational linguistics: human language technologies,
volume 1 (long and short papers) , pages 4171–4186.
Abhimanyu Dubey, Abhinav Jauhri, Abhinav Pandey,
Abhishek Kadian, Ahmad Al-Dahle, Aiesha Letman,
Akhil Mathur, Alan Schelten, Amy Yang, Angela
Fan, et al. 2024. The llama 3 herd of models. arXiv
preprint arXiv:2407.21783 .
Javid Ebrahimi, Anyi Rao, Daniel Lowd, and Dejing
Dou. 2018. HotFlip: White-box adversarial exam-
ples for text classification. In Proceedings of the 56th
Annual Meeting of the Association for Computational
Linguistics (Volume 2: Short Papers) , pages 31–36,
Melbourne, Australia. Association for Computational
Linguistics.
Yunfan Gao, Yun Xiong, Xinyu Gao, Kangxiang Jia,
Jinliu Pan, Yuxi Bi, Yi Dai, Jiawei Sun, and Haofen
Wang. 2023. Retrieval-augmented generation for
large language models: A survey. arXiv preprint
arXiv:2312.10997 .
Kelvin Guu, Kenton Lee, Zora Tung, Panupong Pasu-
pat, and Mingwei Chang. 2020. Retrieval augmented
language model pre-training. In International confer-
ence on machine learning , pages 3929–3938. PMLR.
Lei Huang, Weijiang Yu, Weitao Ma, Weihong Zhong,
Zhangyin Feng, Haotian Wang, Qianglong Chen,
Weihua Peng, Xiaocheng Feng, Bing Qin, et al. 2023.
A survey on hallucination in large language models:
Principles, taxonomy, challenges, and open questions.
arXiv preprint arXiv:2311.05232 .
Samuel Humeau, Kurt Shuster, Marie-Anne Lachaux,
and Jason Weston. Poly-encoders: Architectures and
pre-training strategies for fast and accurate multi-
sentence scoring. In International Conference on
Learning Representations .Gautier Izacard, Mathilde Caron, Lucas Hosseini, Se-
bastian Riedel, Piotr Bojanowski, Armand Joulin,
and Edouard Grave. Unsupervised dense informa-
tion retrieval with contrastive learning. Transactions
on Machine Learning Research .
Fengqing Jiang, Zhangchen Xu, Luyao Niu, Zhen Xi-
ang, Bhaskar Ramasubramanian, Bo Li, and Radha
Poovendran. 2024. ArtPrompt: ASCII art-based jail-
break attacks against aligned LLMs. In Proceedings
of the 62nd Annual Meeting of the Association for
Computational Linguistics (Volume 1: Long Papers) ,
pages 15157–15173, Bangkok, Thailand. Association
for Computational Linguistics.
Zhengbao Jiang, Frank Xu, Luyu Gao, Zhiqing Sun,
Qian Liu, Jane Dwivedi-Yu, Yiming Yang, Jamie
Callan, and Graham Neubig. 2023. Active retrieval
augmented generation. In Proceedings of the 2023
Conference on Empirical Methods in Natural Lan-
guage Processing , pages 7969–7992, Singapore. As-
sociation for Computational Linguistics.
Jeff Johnson, Matthijs Douze, and Hervé Jégou. 2019.
Billion-scale similarity search with gpus. IEEE
Transactions on Big Data , 7(3):535–547.
Vladimir Karpukhin, Barlas Oguz, Sewon Min, Patrick
Lewis, Ledell Wu, Sergey Edunov, Danqi Chen, and
Wen-tau Yih. 2020a. Dense passage retrieval for
open-domain question answering. In Proceedings
of the 2020 Conference on Empirical Methods in
Natural Language Processing (EMNLP) , pages 6769–
6781.
Vladimir Karpukhin, Barlas Oguz, Sewon Min, Patrick
Lewis, Ledell Wu, Sergey Edunov, Danqi Chen, and
Wen-tau Yih. 2020b. Dense passage retrieval for
open-domain question answering. In Proceedings
of the 2020 Conference on Empirical Methods in
Natural Language Processing (EMNLP) , pages 6769–
6781, Online. Association for Computational Lin-
guistics.
Omar Khattab and Matei Zaharia. 2020. Colbert: Effi-
cient and effective passage search via contextualized
late interaction over bert. In Proceedings of the 43rd
International ACM SIGIR conference on research
and development in Information Retrieval , pages 39–
48.
Tom Kwiatkowski, Jennimaria Palomaki, Olivia Red-
field, Michael Collins, Ankur Parikh, Chris Alberti,
Danielle Epstein, Illia Polosukhin, Jacob Devlin, Ken-
ton Lee, Kristina Toutanova, Llion Jones, Matthew
Kelcey, Ming-Wei Chang, Andrew M. Dai, Jakob
Uszkoreit, Quoc Le, and Slav Petrov. 2019. Natu-
ral questions: A benchmark for question answering
research. Transactions of the Association for Compu-
tational Linguistics , 7:452–466.
Patrick Lewis, Ethan Perez, Aleksandra Piktus, Fabio
Petroni, Vladimir Karpukhin, Naman Goyal, Hein-
rich Küttler, Mike Lewis, Wen-tau Yih, Tim Rock-
täschel, et al. 2020. Retrieval-augmented generation

for knowledge-intensive nlp tasks. Advances in Neu-
ral Information Processing Systems , 33:9459–9474.
Zhuowan Li, Cheng Li, Mingyang Zhang, Qiaozhu Mei,
and Michael Bendersky. 2024. Retrieval augmented
generation or long-context llms? a comprehensive
study and hybrid approach. In Proceedings of the
2024 Conference on Empirical Methods in Natural
Language Processing: Industry Track , pages 881–
893.
Na Liu, Liangyu Chen, Xiaoyu Tian, Wei Zou, Kaijiang
Chen, and Ming Cui. 2024. From llm to conversa-
tional agent: A memory enhanced architecture with
fine-tuning of large language models. arXiv preprint
arXiv:2401.02777 .
Yinhan Liu, Myle Ott, Naman Goyal, Jingfei Du, Man-
dar Joshi, Danqi Chen, Omer Levy, Mike Lewis,
Luke Zettlemoyer, and Veselin Stoyanov. 2019.
Roberta: A robustly optimized bert pretraining ap-
proach. arXiv preprint arXiv:1907.11692 .
Jiageng Mao, Junjie Ye, Yuxi Qian, Marco Pavone, and
Yue Wang. A language agent for autonomous driving.
InFirst Conference on Language Modeling .
Han Cheol Moon, Shafiq Joty, and Xu Chi. 2022. Grad-
mask: Gradient-guided token masking for textual
adversarial example detection. In Proceedings of
the 28th ACM SIGKDD conference on knowledge
discovery and data mining , pages 3603–3613.
Tri Nguyen, Mir Rosenberg, Xia Song, Jianfeng Gao,
Saurabh Tiwary, Rangan Majumder, and Li Deng.
Ms marco: A human generated machine reading com-
prehension dataset. choice , 2640:660.
Rodrigo Nogueira and Kyunghyun Cho. 2019. Pas-
sage re-ranking with bert. arXiv preprint
arXiv:1901.04085 .
Alec Radford, Jeffrey Wu, Rewon Child, David Luan,
Dario Amodei, Ilya Sutskever, et al. Language mod-
els are unsupervised multitask learners.
Colin Raffel, Noam Shazeer, Adam Roberts, Katherine
Lee, Sharan Narang, Michael Matena, Yanqi Zhou,
Wei Li, and Peter J Liu. 2020. Exploring the lim-
its of transfer learning with a unified text-to-text
transformer. Journal of machine learning research ,
21(140):1–67.
Machel Reid, Nikolay Savinov, Denis Teplyashin,
Dmitry Lepikhin, Timothy Lillicrap, Jean-baptiste
Alayrac, Radu Soricut, Angeliki Lazaridou, Orhan Fi-
rat, Julian Schrittwieser, et al. 2024. Gemini 1.5: Un-
locking multimodal understanding across millions of
tokens of context. arXiv preprint arXiv:2403.05530 .
Keshav Santhanam, Omar Khattab, Jon Saad-Falcon,
Christopher Potts, and Matei Zaharia. 2022. Col-
bertv2: Effective and efficient retrieval via
lightweight late interaction. In Proceedings of the
2022 Conference of the North American Chapter of
the Association for Computational Linguistics: Hu-
man Language Technologies , pages 3715–3734.Zhihong Shao, Yeyun Gong, Yelong Shen, Minlie
Huang, Nan Duan, and Weizhu Chen. 2023. En-
hancing retrieval-augmented large language models
with iterative retrieval-generation synergy. In Find-
ings of the Association for Computational Linguistics:
EMNLP 2023 , pages 9248–9274.
Xinyue Shen, Zeyuan Chen, Michael Backes, Yun Shen,
and Yang Zhang. 2024. " do anything now": Charac-
terizing and evaluating in-the-wild jailbreak prompts
on large language models. In Proceedings of the
2024 on ACM SIGSAC Conference on Computer and
Communications Security , pages 1671–1685.
Zhen Tan, Chengshuai Zhao, Raha Moraffah, Yifan Li,
Song Wang, Jundong Li, Tianlong Chen, and Huan
Liu. 2024. Glue pizza and eat rocks-exploiting vul-
nerabilities in retrieval-augmented generative models.
InProceedings of the 2024 Conference on Empiri-
cal Methods in Natural Language Processing , pages
1610–1626.
Gemma Team, Thomas Mesnard, Cassidy Hardin,
Robert Dadashi, Surya Bhupatiraju, Shreya Pathak,
Laurent Sifre, Morgane Rivière, Mihir Sanjay Kale,
Juliette Love, et al. 2024. Gemma: Open models
based on gemini research and technology. arXiv
preprint arXiv:2403.08295 .
Nandan Thakur, Nils Reimers, Andreas Rücklé, Ab-
hishek Srivastava, and Iryna Gurevych. Beir: A het-
erogeneous benchmark for zero-shot evaluation of
information retrieval models. In Thirty-fifth Con-
ference on Neural Information Processing Systems
Datasets and Benchmarks Track (Round 2) .
Hugo Touvron, Louis Martin, Kevin Stone, Peter Al-
bert, Amjad Almahairi, Yasmine Babaei, Nikolay
Bashlykov, Soumya Batra, Prajjwal Bhargava, Shruti
Bhosale, et al. 2023. Llama 2: Open founda-
tion and fine-tuned chat models. arXiv preprint
arXiv:2307.09288 .
Chong Xiang, Tong Wu, Zexuan Zhong, David Wagner,
Danqi Chen, and Prateek Mittal. 2024. Certifiably
robust rag against retrieval corruption. arXiv preprint
arXiv:2405.15556 .
Zihao Xu, Yi Liu, Gelei Deng, Yuekang Li, and Stjepan
Picek. 2024a. A comprehensive study of jailbreak
attack versus defense for large language models. In
Findings of the Association for Computational Lin-
guistics ACL 2024 , pages 7432–7449.
Ziwei Xu, Sanjay Jain, and Mohan S Kankanhalli.
2024b. Hallucination is inevitable: An innate limita-
tion of large language models. CoRR .
Jiaqi Xue, Mengxin Zheng, Yebowen Hu, Fei Liu, Xun
Chen, and Qian Lou. 2024. Badrag: Identifying vul-
nerabilities in retrieval augmented generation of large
language models. arXiv preprint arXiv:2406.00083 .
Zhilin Yang, Peng Qi, Saizheng Zhang, Yoshua Bengio,
William Cohen, Ruslan Salakhutdinov, and Christo-
pher D. Manning. 2018. HotpotQA: A dataset for

diverse, explainable multi-hop question answering.
InProceedings of the 2018 Conference on Empiri-
cal Methods in Natural Language Processing , pages
2369–2380, Brussels, Belgium. Association for Com-
putational Linguistics.
Collin Zhang, Tingwei Zhang, and Vitaly Shmatikov.
2024. Controlled generation of natural adversarial
documents for stealthy retrieval poisoning. arXiv
preprint arXiv:2410.02163 .
Lianmin Zheng, Wei-Lin Chiang, Ying Sheng, Siyuan
Zhuang, Zhanghao Wu, Yonghao Zhuang, Zi Lin,
Zhuohan Li, Dacheng Li, Eric Xing, et al. 2023.
Judging llm-as-a-judge with mt-bench and chatbot
arena. Advances in Neural Information Processing
Systems , 36:46595–46623.
Zexuan Zhong, Ziqing Huang, Alexander Wettig, and
Danqi Chen. 2023. Poisoning retrieval corpora by
injecting adversarial passages. In Proceedings of the
2023 Conference on Empirical Methods in Natural
Language Processing , pages 13764–13775.
Huichi Zhou, Kin-Hei Lee, Zhonghao Zhan, Yue Chen,
and Zhenhao Li. 2025. Trustrag: Enhancing ro-
bustness and trustworthiness in rag. arXiv preprint
arXiv:2501.00879 .
Andy Zou, Zifan Wang, Nicholas Carlini, Milad Nasr,
J Zico Kolter, and Matt Fredrikson. 2023. Univer-
sal and transferable adversarial attacks on aligned
language models. arXiv preprint arXiv:2307.15043 .
Wei Zou, Runpeng Geng, Binghui Wang, and Jinyuan
Jia. 2024. Poisonedrag: Knowledge poisoning at-
tacks to retrieval-augmented generation of large lan-
guage models. arXiv preprint arXiv:2402.07867 .
A Detailed Attack Settings
PoisonedRAG. Poisoned documents in Poisone-
dRAG are composed of an incorrect information
paragraph and cheating tokens. We crafted five
documents per query, each conveying the same in-
correct answer. To optimize cheating tokens, we
initialized 30 masked tokens and iteratively applied
the Hotflip technique for 30 iterations, replacing
randomly selected cheating tokens. Each Hotflip
iteration considered 100 candidate replacements.
In total, we generated 1,000 poisoned documents.
Phantom. Poisoned documents in Phantom con-
tain separate cheating tokens optimized for retrieval
and generation, followed by an adversarial com-
mand. For retriever optimization, we initialized
128 tokens and applied the Hotflip technique. For
generator optimization, we used 16 tokens employ-
ing Multi Coordinate Gradient (MCG) (Chaudhari
et al., 2024). We crafted five poisoned documents
per trigger, resulting in a total of five poisoned doc-
uments.
Figure 8: Examples of poisoned documents for each
attack method. Words in red indicate cheating tokens,
while black words denote incorrect information or ad-
versarial commands. Attacks based on Hotflip exhibit
significant unnatural text patterns, whereas AdvDecod-
ing generates more natural-looking text by leveraging
LLMs and naturalness constraint.
AdvDecoding. Zhang et al. (2024) proposed a poi-
soning method that leverages third-party LLMs to
generate cheating tokens with high naturalness and
semantic similarity to the queries containing the
trigger, without relying on Hotflip. In our work,
we adopted this method to create cheating tokens,
which were appended to the adversarial command,
as cheating tokens alone does not induce generator
malfunction. We crafted five poisoned documents
per trigger, varying document lengths. Specifically,
we generated documents of lengths 50, 80, 110,
140, and 170 tokens, as longer documents tend to
exhibit higher similarity.
We show the example of poisoned documents in
Figure 8 for better comprehension.
B Detailed Defense Settings
Based on the findings of Zhang et al. (2024), we set
the perplexity threshold to 200, as their results indi-
cate that attacks using Hotflip had a minimum per-
plexity value exceeding 1000 in the MS MARCO
dataset. For perplexity computation, we used GPT-
24(Radford et al.). Regarding l2-norm constraints,
we applied a threshold of 1.7 for Contriever and
13 for DPR. For generation baselines, we followed
the default parameter settings from TrustRAG and
RobustRAG ( α= 0.3,β= 3.0). All experiments
were conducted on a single A6000 GPU.
C Masked Token Probability
We report the average masked token probabilities
in Table 5. The results indicate that analyzing the
masked token probabilities of Ntokens with high
gradient values is sufficient to detect poisoned doc-
4https://huggingface.co/openai-community/gpt2

uments. While non-poisoned documents exhibit
an average probability exceeding 40%, poisoned
documents show significantly lower probabilities,
below 15%. Furthermore, narrowing the selection
toMtokens causes a sharp decline in probabilities
for poisoned documents, reducing them to below
1%.
D Retrieved Poisoned Documents
Table 7 presents the number of poisoned documents
included in the top- kretrieval using the Naïve
method. We found that targeting a trigger tends
to be less effective for retrieval. This can be at-
tributed to the target scope, as the five crafted docu-
ments target all queries containing the trigger word
"iPhone." Another contributing factor may be the
high similarity of clean documents, as the trigger
serves as a strong retrieval signal, primarily retriev-
ing documents that contain the exact trigger word.
Additionally, Contriever is more vulnerable to
attacks, retrieving a higher number of poisoned
documents than DPR. One possible reason is the
difference in how query and document encoders
are used. Since DPR employs separate models for
query and document encoding, adversarial attacks
such as Hotflip, which rely solely on the document
encoder’s gradient, may be less effective. Future
work could further investigate the impact of re-
triever architecture on different attack methods.
E τSelection
Retriever Attack DatasetnDCGMS MARCOClean Poison
DPRPoisonedRAGNQ 0.409 0.491 0.990
HotpotQA 0.396 0.396 1.0
MS MARCO 0.396 0.396 1.0
PhantomNQ 0.272 0.272 0.999
HotpotQA 0.220 0.220 1.0
MS MARCO 0.220 0.220 1.0
AdvDecodingNQ 0.136 0.156 0.999
HotpotQA 0.132 0.132 1.0
MS MARCO 0.132 0.132 1.0
ContrieverPoisonedRAGNQ 0.466 0.459 0.989
HotpotQA 0.510 0.501 0.988
MS MARCO 0.383 0.409 0.994
PhantomNQ 0.416 0.416 1.0
HotpotQA 0.495 0.494 1.0
MS MARCO 0.365 0.363 1.0
AdvDecodingNQ 0.416 0.416 1.0
HotpotQA 0.495 0.495 0.981
MS MARCO 0.365 0.365 1.0
Table 8: Performance using random documents to cal-
culate P-score in threshold τ.
Although we set τbased on the average P-score of
existing query-relevant document pairs, obtaining
precisely relevant documents is not always feasible.
To address this limitation, we also report resultsusing randomly selected documents to compute the
average P-score .
RetrieverDocument
typeNQ HotpotQA MS MARCO
DPRRelevant 0.225 0.280 0.176
Random 0.075 0.349 0.129
ContrieverRelevant 0.249 0.297 0.240
Random 0.084 0.361 0.168
Table 9: P-score comparison using relevant documents
(result in Section 6.1) and random documents (result in
Section 6.3. It is preferable to use domain specific P-
value since the differences between each dataset are not
trivial. Although P-score are different between using
relevant and random documents (up to nearly 60%), it is
still safe because of the well separation GMTP provides.
Table 8 shows that even in this random docu-
ment setting, GMTP maintains a high filtering rate
without a significant drop in retrieval performance.
This aligns with the findings in Section 6.2, where
GMTP effectively separates poisoned documents
with a large margin in masked token probabilities.
As a result, GMTP remains stable even when using
random documents, where the threshold variation
is close to 50% compared to the result with relevant
documents. The P-score values used in both cases
are reported in Table 9.
F Evaluation Metrics
Filtering Rate. Filtering rate measures the pro-
portion of poisoned documents removed from the
retrieved top- kdocuments by the defense method,
relative to the number of poisoned documents be-
fore applying the defense. Eq. 3 describes the FR
calculation, and dNav
prepresents a poisoned doc-
ument in Naïve enviornment, and dD
pa poisoned
document when the defense method is applied.
# of retrieved dNav
p−# of retrieved dD
p
# of retrieved dNavp(3)
Generation metrics. ACC measures the propor-
tion of responses that are labeled "YES" by GPT-4o
among all responses generated by the RAG system.
As shown in Appendix G, a "YES" response in-
dicates that the generated response adheres to the
correct answer. CACC is calculated in the same
way. While ASR uses same calculation as in Eq.
4, it uses incorrect answer as correct answer in the
prompt of Figure 9.
# of ‘YES’ GPT-4o responses
# of RAG responses·100 (4)

Attack RetrieverDocument
typeNQ HoptotQA MS MARCO
N M N M N M
PoisonedRAGDPRPoison 9.19 0.02 4.88 0.00 5.87 0.00
Relevant 57.85 29.06 61.81 33.73 43.82 17.28
Clean 49.05 18.38 56.77 26.99 45.87 16.91
ContrieverPoison 9.26 0.18 8.72 0.18 6.42 0.06
Relevant 60.66 32.21 62.69 33.54 49.69 20.76
Clean 52.10 21.4 60.46 30.51 49.28 20.25
PhantomDPRPoison 0.02 0.00 0.09 0.00 0.03 0.00
Relevant 58.02 29.96 59.32 30.53 42.88 14.56
Clean 47.73 17.27 55.93 26.34 42.69 13.89
ContrieverPoison 1.42 0.00 1.69 0.00 4.75 0.00
Relevant 59.24 30.30 61.77 32.23 48.13 18.43
Clean 50.73 19.58 58.49 27.92 47.09 18.30
AdvDecodingDPRPoison 2.70 0.03 4.92 0.00 5.29 0.06
Relevant 58.14 30.13 59.60 30.91 42.56 14.35
Clean 47.74 17.30 55.72 26.11 42.73 13.89
ContrieverPoison 5.12 0.02 14.94 0.57 8.32 0.02
Relevant 59.60 30.73 61.86 32.24 48.48 18.62
Clean 50.60 19.47 58.89 28.38 47.10 18.14
Table 5: Average of masked token probability using selected Ntokens and Mtokens. We marked as 0.00 if the
value is below 0.01.
Retrieval
Attack DefenseNQ HotpotQA MS MARCO
nDCG@10FR (↑)nDCG@10FR (↑)nDCG@10FR (↑)Clean ( ↑) Poison ( ↑) Clean ( ↑) Poison ( ↑) Clean ( ↑) Poison ( ↑)
PoisonedRAGNaive 0.488 0.196 0.0 0.604 0.237 0.0 0.515 0.220 0.0
PPL 0.494 0.492 0.998 0.602 0.602 1.0 0.505 0.505 0.999
l2-norm 0.425 0.424 1.0 0.383 0.383 1.0 0.432 0.468 1.0
GMTP 0.452 0.445 0.99 0.519 0.508 0.987 0.386 0.382 0.994
PhantomNaive 0.444 0.206 0.0 0.589 0.306 0.0 0.468 0.230 0.0
PPL 0.446 0.446 1.0 0.587 0.587 1.0 0.457 0.457 1.0
l2-norm 0.376 0.376 1.0 0.373 0.370 1.0 0.432 0.432 1.0
GMTP 0.402 0.400 1.0 0.497 0.496 1.0 0.332 0.329 1.0
AdvDecodingNaive 0.444 0.410 0.0 0.589 0.560 0.0 0.468 0.464 0.0
PPL 0.446 0.443 0.867 0.587 0.581 0.665 0.457 0.453 0.021
l2-norm 0.376 0.376 1.0 0.373 0.373 0.953 0.432 0.432 1.0
GMTP 0.402 0.402 1.0 0.497 0.497 0.967 0.332 0.332 1.0
Generation
Attack DefenseNQ HotpotQA MS MARCO
CACC ( ↑) ACC ( ↑) ASR ( ↓) CACC ( ↑) ACC ( ↑) ASR ( ↓) CACC ( ↑) ACC ( ↑) ASR ( ↓)
PoisonedRAGNaive 63.5 10.05 84.5 42.0 10.5 84.5 71.0 28.0 69.5
TrustRAG 38.0 37.5 10.0 25.0 24.5 14.0 30.0 39.0 13.0
RobustRAG 42.0 16.5 68.0 31.5 11.5 82.0 38.5 24.5 54.5
GMTP 59.5 59.5 4.5 44.5 41.0 9.5 59.5 60.5 5.0
PhantomNaive 63.0 1.5 99.5 41.0 21.0 36.5 61.5 7.5 91.0
TrustRAG 28.0 29.5 0.0 25.0 26.0 0.0 23.0 21.5 0.0
RobustRAG 34.5 7.0 90.0 25.5 26.5 0.0 27.5 22.0 49.0
GMTP 57.0 51.0 0.5 43.0 41.0 0.0 53.5 46.0 0.0
AdvDecodingNaive 63.0 34.5 44.0 40.0 24.5 43.5 55.5 47.0 12.5
TrustRAG 25.0 28.5 0.0 24.0 22.0 1.0 20.0 19.0 0.0
RobustRAG 29.0 29.0 26.0 26.5 23.5 20.5 28.0 26.0 13.0
GMTP 58.0 55.5 0.0 41.5 41.0 2.0 48.5 48.0 0.0
Table 6: Performance in retrieval phase and generation phase using Contriever. "Clean" refers to the environment
where no attack is applied, while "Poison" indicates the environment where an attack is applied. Naïve represents
no defense applied at all. Bold indicates the best defense method: highest FR ( ↑) in retrieval, lowest ASR ( ↓) in
generation.

NQ HotpotQA MS MARCO
DPRPoisonedRAG 773 903 773
Phantom 145 447 246
AdvDecoding 84 133 70
ContrieverPoisonedRAG 1000 1000 1000
Phantom 964 934 910
AdvDecoding 203 215 94
Table 7: Total number of poisoned documents retrieved
across 200 queries.
G Evaluation Prompt
Figure 9: Evaluation prompt using GPT-4o.
Figure 9 illustrates the evaluation prompt for ASR
in PoisonedRAG, as well as ACC and CACC
for all defense methods in the generation phase.
Specifically, we evaluated ASR by replacing the
"correct_answer" with an incorrect answer that
aligns with the poisoned document. We applied
the prompt iteratively to each query-response pair,
counting "YES" responses from GPT-4o.
H FPR of GMTP
AttackNQ HotpotQA MS MARCO
Clean Poison Clean Poison Clean Poison
PoisonedRAG 0.042 0.026 0.051 0.023 0.051 0.027
Phantom 0.049 0.04 0.047 0.029 0.039 0.031
AdvDecoding 0.049 0.043 0.047 0.04 0.039 0.036
Table 10: FPR of GMTP in various datasets and attacks.While the precision of GMTP has been demon-
strated in Section 6.1, we further evaluate its relia-
bility by measuring False Positive Rate (FPR) in Ta-
ble 10, which quantifies how often legitimate docu-
ments are mistakenly filtered out. GMTP achieves
an FPR of nearly 0.05 or lower, which indicates
a low rate of misclassification. Thus, this result
reinforces the method’s ability to effectively elimi-
nate adversarial content while preserving access to
relevant information, highlighting its suitability for
high-stakes retrieval scenarios where precision is
critical.
I Varying Naturalness
In order to explore the failure cases of GMTP, we
include the results of experiments that use fewer
adversarial tokens (i.e., tokens optimized to maxi-
mize the similarity between the query and the poi-
soned document). As shown in Table 11, reducing
the number of adversarial tokens improves stealthi-
ness, but it comes at the cost of decreased retrieval
success for poisoned documents. Notably, Poisone-
dRAG remains effective even with single-token
optimization. However, this is primarily due to its
narrow target scope, which targets specific queries
and relies on naturally crafted false information as
poisoned documents, rather than successful opti-
mization. It is worth noting that despite the target
attack of GMTP being optimization-based, it is
able to successfully filter out up to 70% of the ac-
tual retrieved poisoned documents (i.e., precisely
detect even a single unnatural token).
J MLM-Agnostic Performance
To further demonstrate GMTP’s generalization
across different architectures, we report results
using different MLM (i.e., RoBERTa (Liu et al.,
2019)) instead of BERT. RoBERTa is known to
use different pretraining method and dataset com-
pared to BERT, which can lead to variations in
performance. However, as shown Table 12, GMTP
remains robust across different LLMs, as it consis-
tently achieves high filtering rates while maintain-
ing minimal nDCG@10 degradation. These results
demonstrate that GMTP is effective regardless of
MLM variations. Note that we used DPR for the
experiments.
K Full Experimental Results
In this section we show the results using Contriever.
Specifically, we report main results comparing with

Attack Adv. tokensNQ HotpotQA MS MARCO
nDCG-P FR Poison nDCG-P FR Poison nDCG-P FR Poison
PoisonedRAG1 0.457 0.504 282 0.261 0.702 396 0.135 0.571 177
5 0.472 0.963 571 0.28 0.992 781 0.13 0.985 535
10 0.476 0.999 678 0.28 0.998 833 0.132 0.999 693
Phantom1 0.385 1.0 0 0.212 1.0 0 0.107 1.0 0
5 0.387 1.0 1 0.216 1.0 4 0.102 1.0 2
10 0.388 1.0 5 0.222 1.0 11 0.086 1.0 2
AdvDecoding1 0.389 1.0 0 0.226 1.0 0 0.114 1.0 0
5 0.389 1.0 2 0.226 1.0 1 0.114 1.0 0
10 0.389 1.0 6 0.226 1.0 3 0.114 1.0 5
Table 11: Performance with varying numbers of adversarial tokens. nDCG-P denotes nDCG@10 under the attack
setting, while Poison indicates the number of poisoned documents retrieved in the top-10.
AttackNQ HotpotQA MS MARCO
nDCG@10FR (↑)nDCG@10FR (↑)nDCG@10FR (↑)Clean ( ↑) Poison ( ↑) Clean ( ↑) Poison ( ↑) Clean ( ↑) Poison ( ↑)
PoisonedRAG 0.493 0.491 1.0 0.298 0.298 1.0 0.176 0.174 0.999
Phantom 0.385 0.385 1.0 0.236 0.236 1.0 0.152 0.152 1.0
AdvDecoding 0.385 0.385 1.0 0.236 0.236 1.0 0.152 0.152 1.0
Table 12: Performance of GMTP using RoBERTa as the MLM.
baselines in Table 6. Figure 10 and Figure 11 il-
lustrates the density plot across datasets and attack
methods, using DPR and Contriever, respectively.
Performance across various λvalues is described
in Figure 12 and Figure 13. Lastly, the masking
precision using Contriever is reported in Table 13.
Attack NQ HotpotQA MS MARCO
PoisonedRAG 0.915 0.911 0.925
Phantom 0.971 0.893 0.95
AdvDecoding 1.0 0.943 0.954
Table 13: GMTP precision in detecting cheating tokens
using Contriever.

PoisonedRAG Phantom AdvDecodingNQ HotpotQA MS MARCOFigure 10: Density plot of masked token probability using DPR.
PoisonedRAG Phantom AdvDecodingNQ HotpotQA MS MARCO
Figure 11: Density plot of masked token probability using Contriever.

PoisonedRAG Phantom AdvDecodingNQ HotpotQA MS MARCO
Figure 12: nDCG@10 and filtering rate across various λvalues using DPR.
PoisonedRAG Phantom AdvDecodingNQ HotpotQA MS MARCO
Figure 13: nDCG@10 and filtering rate across various λvalues using Contriever.