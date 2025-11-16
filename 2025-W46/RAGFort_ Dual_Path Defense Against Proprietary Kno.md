# RAGFort: Dual-Path Defense Against Proprietary Knowledge Base Extraction in Retrieval-Augmented Generation

**Authors**: Qinfeng Li, Miao Pan, Ke Xiong, Ge Su, Zhiqiang Shen, Yan Liu, Bing Sun, Hao Peng, Xuhong Zhang

**Published**: 2025-11-13 09:34:46

**PDF URL**: [https://arxiv.org/pdf/2511.10128v1](https://arxiv.org/pdf/2511.10128v1)

## Abstract
Retrieval-Augmented Generation (RAG) systems deployed over proprietary knowledge bases face growing threats from reconstruction attacks that aggregate model responses to replicate knowledge bases. Such attacks exploit both intra-class and inter-class paths, progressively extracting fine-grained knowledge within topics and diffusing it across semantically related ones, thereby enabling comprehensive extraction of the original knowledge base. However, existing defenses target only one path, leaving the other unprotected. We conduct a systematic exploration to assess the impact of protecting each path independently and find that joint protection is essential for effective defense. Based on this, we propose RAGFort, a structure-aware dual-module defense combining "contrastive reindexing" for inter-class isolation and "constrained cascade generation" for intra-class protection. Experiments across security, performance, and robustness confirm that RAGFort significantly reduces reconstruction success while preserving answer quality, offering comprehensive defense against knowledge base extraction attacks.

## Full Text


<!-- PDF content starts -->

RAGFort: Dual-Path Defense Against Proprietary Knowledge Base Extraction
in Retrieval-Augmented Generation
Qinfeng Li1*, Miao Pan1*, Ke Xiong1*, Ge Su1,
Zhiqiang Shen2, Yan Liu2, SUN Bing3, Hao Peng4, Xuhong Zhang1,5†
1Zhejiang University
2Ant Group
3Universal Identification Technology (Hangzhou) Co., Ltd.
4Zhejiang Normal University
5Ningbo Global Innovation Center, Zhejiang University
Abstract
Retrieval-Augmented Generation (RAG) systems deployed
over proprietary knowledge bases face growing threats from
reconstruction attacks that aggregate model responses to
replicate knowledge bases. Such attacks exploit both intra-
class and inter-class paths—progressively extracting fine-
grained knowledge within topics and diffusing it across se-
mantically related ones, thereby enabling comprehensive ex-
traction of the original knowledge base. However, existing de-
fenses target only one path, leaving the other unprotected. We
conduct a systematic exploration to assess the impact of pro-
tecting each path independently and find that joint protection
is essential for effective defense. Based on this, we propose
RAGFort, a structure-aware dual-module defense combin-
ingcontrastive reindexingfor inter-class isolation andcon-
strained cascade generationfor intra-class protection. Ex-
periments across security, performance, and robustness con-
firm that RAGFort significantly reduces reconstruction suc-
cess while preserving answer quality, offering comprehensive
defense against knowledge base extraction attacks.
Code— https://github.com/happywinder/RAGFort
1 Introduction
Large Language Models (LLMs) have demonstrated re-
markable capabilities across a wide range of natural lan-
guage processing tasks, particularly in open-domain ques-
tion answering and knowledge reasoning (Brown et al. 2020;
OpenAI 2023). To enhance factual accuracy and controlla-
bility, Retrieval-Augmented Generation (RAG) has emerged
as a widely adopted paradigm (Lewis et al. 2020; Izac-
ard and Grave 2021), enabling models to retrieve exter-
nal knowledge to support answer generation. Beyond public
knowledge like Wikipedia, RAG systems are increasingly
deployed in high-value domains, such as healthcare (Singhal
et al. 2022), and finance (Wu et al. 2023), where the knowl-
edge base is proprietary, domain-specific, and constitutes a
core intellectual asset for commercial applications.
*These authors contributed equally.
†Corresponding tozhangxuhong@zju.edu.cn
Copyright © 2026, Association for the Advancement of Artificial
Intelligence (www.aaai.org). All rights reserved.However, the privatized nature of these knowledge bases
exposes a new vulnerability (Ni et al. 2025). Specifically,
recent studies reveal that even by interacting with the RAG
system in a black-box manner, attackers can extract a proxy
knowledge base by aggregating responses from the RAG
system (Zeng et al. 2024a; Cohen, Bitton, and Nassi 2024).
In particular, attackers may utilize the extracted knowl-
edge base to construct functionally equivalent RAG sys-
tems (Jiang et al. 2024), thereby effectively replicating and
misusing the original RAG system’s capabilities. Such repli-
cation enables competitors to offer highly similar services,
leading to direct financial losses and substantially eroding
the competitive edge of the original provider. We refer to
this threat as theknowledge base extraction attack.
Specifically, recent studies show that attackers can ex-
tract information from a knowledge base by collecting and
combining responses generated by the RAG system. For ex-
ample, document extraction attacks have shown that attack-
ers can craft queries that specifically target certain infor-
mation. By prompting the model with these carefully de-
signed queries, attackers can induce the RAG system to gen-
erate responses that contain or reveal underlying knowledge
base content (Zeng et al. 2024a). More alarmingly, some
emerging methods pose a more systemic extraction of the
entire knowledge base rather than focusing on only iso-
lated pieces of content. To achieve this, these attacks not
only iteratively refine queries to extract detailed information
within a specific targeted topic (referred to asintra-class
extraction), but also gradually expand their query scope
to cover other semantically related topics. Specifically, at-
tackers leverage previously extracted knowledge to recur-
sively generate queries targeting new semantically related
topics. By iteratively expanding the extraction boundary, the
attack gradually propagates through the knowledge base,
eventually achieving nearly complete coverage (referred to
asinter-class extraction). For example, RAG-Thief (Jiang
et al. 2024) employs an automated agent that starts with
an initial adversarial query, stores the extracted chunks in
memory, and then analyzes these chunks to generate more
targeted queries for uncovering missing or previously un-arXiv:2511.10128v1  [cs.AI]  13 Nov 2025

known information. By iteratively updating its memory, the
agent can both fill in missing details within the same topic
(i.e., intra-class extraction) and gradually expand extraction
to neighboring topics that were previously unknown or un-
queried (i.e., inter-class extraction), ultimately reconstruct-
ing nearly the entire knowledge base.
However, existing defense strategies are still at a prelim-
inary stage, typically focusing on individual attack paths
rather than providing comprehensive protection. For exam-
ple, intra-class defenses, such as paraphrasing or summa-
rizing retrieved content, reduce leakage within individual
chunks but fail to prevent cross-topic aggregation (Zeng
et al. 2024a,b). Conversely, inter-class defenses restrict re-
trieval to closely related content, limiting topic drift but al-
lowing in-depth extraction within a single class (Zeng et al.
2024a), such as limiting retrieval to the few most relevant
chunks. Worse still, existing intra-class and inter-class de-
fense schemes are difficult to combine effectively. For ex-
ample,Set Distance Threshold(Zeng et al. 2024a), as an
inter-class protection, relies on strictly limiting the retrieval
distance so that only highly relevant chunks are returned,
which essentially forces the RAG system tofocus on only a
few chunksand prevents extraction from expanding to other
topics. In contrast, to achieve intra-class protection,Sum-
marization with Relevant Query(Zeng et al. 2024a) empha-
sizes generating diverse or aggregated content across mul-
tiple chunks within the same topic (i.e., through abstrac-
tive summarization) to prevent attackers from reconstructing
the precise information of any single chunk, and thus inher-
ently requires the RAG systemnot to focus on only a few
chunks. As a result, this fundamental conflict poses signifi-
cant challenges for combining intra-class and inter-class de-
fenses in practice. Nevertheless, no prior work has systemat-
ically compared the two defense paths or explored integrated
strategies that jointly secure both protection paths.
To bridge this gap, we first pose a key question:Between
intra-class and inter-class protections, which is more crit-
ical for safeguarding proprietary knowledge base in RAG
systems, and can integrated protection provide stronger se-
curity than either path alone?To explore this, we design a
targeted experiment that masks portions of knowledge along
each attack path, and assess the resulting impact on an-
swer quality. Our findings reveal a key insight: protecting
only one path offers limited resistance, whereas joint protec-
tion along both paths reduces the effectiveness of the proxy
knowledge base. These results indicate that intra-class ro-
bustness and inter-class isolation are complementary, and
that effective defense requires their coordination.
Based on this insight, we propose a structure-aware dual-
module defense framework, RAGFort, as shown in Figure 2
(see also Appendix A for an illustrative example). For inter-
class protection, we introduce a structure-aware reindex-
ing strategy that reorganizes the retriever’s dense index to
enhance semantic separability between topic classes. This
transformation aims to reinforce topic boundaries, making
it harder for attackers to expand queries from one class
to another, thereby preventing inter-class extraction attacks.
Specifically, the method begins with unsupervised HDB-
SCAN clustering to identify latent topic structures, followedby supervised contrastive learning to train an embedding en-
coder that preserves inter-category margins.
For intra-class protection, we propose a cascaded gener-
ation framework that combines a lightweight draft model
with a robust reference model. In this framework, the draft
model first proposes candidate tokens, while the reference
model selectively filters or replaces high-risk tokens. Specif-
ically, when the draft model exhibits excessive uncertainty
or its output contains sensitive content, the reference model
intervenes and generates safer tokens. Essentially, we have
designed a near-optimal and efficient rejection strategy that
fuses the output distributions of the draft model and the ref-
erence model. The new distribution significantly reduces the
probability of sensitive content being output and improves
the model’s inference performance.
We validate the RAGFort through comprehensive experi-
ments on security, system performance, and ablation studies.
For security, we evaluate RAGFort under advanced black-
box extraction attacks and show that it reduces the pro-
portion of knowledge base entries recovered by attackers
by more than half compared to prior defenses. For system
performance, RAGFort consistently preserves high answer
quality while also maintaining generation efficiency, thus
ensuring practical usability. Through extensive ablation, we
show that each module contributes to knowledge protection,
and their combination delivers strong, complementary ef-
fects. Overall, our contributions are summarized as follows:
• To the best of our knowledge, this is the first systematic
study that defends against knowledge base extraction in
RAG systems. This overlooked vulnerability poses sig-
nificant threats in real-world deployments and leads to
significant commercial risks for RAG providers.
• We identify and formalize two critical paths to protect
the RAG knowledge base: inter-class isolation and intra-
class protection. We design a targeted experiment to vali-
date the role of each path and demonstrate that their com-
bined defense is essential for protecting knowledge base
extraction attacks.
• We propose a structure-aware dual-module defense that
jointly addresses inter-class and intra-class threats. It
combines contrastive reindexing to enhance topic-level
isolation and a cascade framework to suppress sensitive
content generation under adversarial prompts.
• We conduct a comprehensive empirical evaluation of
RAGFort. Our evaluation demonstrates that RAGFort
significantly enhances knowledge base security while
preserving RAG system performance. Ablation studies
confirm that each module is effective on its own and fur-
ther enhances overall protection when combined.
2 Preliminaries
Knowledge Base Extraction Attack.RAG systems gen-
erate answers by first retrieving relevant chunks from an ex-
ternal knowledge base via a retriever module and then us-
ing a language model to compose the final response. While
RAG systems enhance factual generation by retrieving ex-
ternal knowledge, they also expose proprietary content to

0% 10% 20% 30% 40% 50% 60%
(a) HealthCareMagic58606264666870Answer ACC
0% 10% 20% 30% 40% 50% 60%
(b) Enron Email848790939699
0% 10% 20% 30% 40% 50% 60%
(c) Math QA657075808590
Proportion of Protected KnowledgeWithout Protection
Inter-Class Protection
Intra-Class Protection
Joint ProtectionFigure 1: Dual-Path Protection Diagram. This diagram illustrates the structure-aware dual-module defense framework (RAG-
Fort) for protecting data across vertical and horizontal axes.
extraction attacks. Specifically, recent studies show that ad-
versaries can reconstruct knowledge bases by collecting
and combining RAG responses to carefully crafted or para-
phrased queries. By systematically piecing together related
fragments returned by the system, attackers are able to re-
cover large portions of the original content (Ni et al. 2025;
Zeng et al. 2024a; Jiang et al. 2024). Emerging methods
further streamline this process by using adaptive querying
agents that automatically generate and refine queries to ex-
plore both within individual topics (intra-class) and across
different topics (inter-class) (Jiang et al. 2024; Cohen, Bit-
ton, and Nassi 2024). This two-path strategy significantly
improves the fidelity and coverage of the extraction.
Existing Defense.Existing defense strategies remain at an
early stage, limited to a single path, with few truly system-
atic approaches available.Intra-classmethods paraphrase or
summarize individual chunks to obscure fine-grained con-
tent, or replace them altogether with synthetic data (Zeng
et al. 2024a,b), whileinter-classdefenses restrict retrieval
to highly relevant chunks via distance thresholds, implicitly
masking broader topical structures (Zeng et al. 2024a). How-
ever, current approaches address only one direction of attack
in isolation. No unified framework exists to jointly defend
across both axes, leaving RAG systems vulnerable to coor-
dinated extraction strategies.
Cascade Generation.For intra-class protection strate-
gies, we adopt a cascade generation mechanism, namely a
two-stage generation strategy (Narasimhan et al. 2024), to
control the degree of knowledge exposure during the gener-
ation process. In the cascade framework, a lightweight draft
model first proposes a set of candidate tokens, which are
then selectively validated or rejected by a more powerful
verifier model. If the verifier model chooses to reject, it gen-
erates a higher-quality token. This approach achieves faster
generation speeds by enabling parallel evaluation while
maintaining output quality. Unlike previous approaches that
emphasized the efficiency of inference and quality cascad-
ing, we primarily modified the rejection rules in the cascad-
ing system to reduce the probability of retrieving sensitive
content from the knowledge base during generation.
Threat Model.In this paper, we consider two parties:
the defender and the attacker. Unlike privacy attacks that
target individual records, we consider attackers whose ob-
jective is to extract the entire underlying knowledge base
of the RAG system. The attacker is restricted to a black-
box setting and can only access the RAG system via APIqueries, which aligns with real-world deployments of com-
mercial and enterprise RAG services. Defenders are system
providers with full access to the RAG pipeline (including re-
trievers, databases, and language models). The defense ob-
jective is to protect proprietary knowledge bases from recon-
struction while ensuring usability for legitimate users.
3 Comparison of Different Defense Paths
To investigate which protection, intra-class or inter-class,
is more critical for safeguarding proprietary knowledge
in RAG systems, and whether integrated protection offers
stronger security than either path alone, we design an exper-
iment that evaluates the impact of different protection paths
by selectively masking portions of the knowledge base and
analyzing the residual utility of protected knowledge bases.
3.1 Experimental Design
We begin with a complete knowledge baseK fullconstructed
from domain-specific datasets. From this base, we derive
three partially leaked variants by systematically removing
knowledge portions considered well-protected, which are
thus excluded from the proxy knowledge bases. The remain-
ing parts constitute the knowledge bases that the attacker has
successfully reconstructed through model stealing.
•Without Protection: No entries are removed or masked;
the attacker obtains the full original knowledge base
without any restrictions.
•Inter-class Protection: Removes detailed entries within
each category (e.g., specific conditions under ”cardiol-
ogy”), simulating intra-class protection.
•Intra-class Protection: Removes entire topic categories
(e.g., ”dermatology” or ”algebra”), simulating inter-class
protection.
•Joint Protection: Applies both inter-class and intra-class
masking to simulate two-path coordinated defense.
Each variant is used to build an independent RAG sys-
tem, and we evaluate its ability to correctly answer ques-
tions.Higher Answer ACCindicates that the proxy knowl-
edge base still retains substantial utility, suggesting that the
corresponding protection strategy isless effective. The de-
tailed experimental setup is described in Section 5.1.
3.2 Results and Analysis
Figure 1 presents the accuracy of attacker-constructed RAG
systems under each protection path across three representa-

tive domains. The results reveal a clear trend: either inter-
class or intra-class protection alone is insufficient to signif-
icantly limit knowledge base leakage. By contrast, the joint
strategy substantially reduces the utility of the stolen knowl-
edge base. Specifically, the performance of the proxy knowl-
edge base declines as the proportion of protected knowl-
edge increases, and this decline is much more pronounced
with joint protection than with either single-path strategy.
This difference grows even further as more data is protected.
These findings empirically confirm our central hypothesis:
inter-class and intra-class protection are complementary,
and only by combining both protection paths can protec-
tion achieve comprehensive and effective defense.
4 Our Design: RAGFort
In this section, we introduce RAGFort, a structure-aware
dual-path defense framework to counter knowledge base re-
construction attacks in RAG systems.
4.1 Overview
RAGFort is achieved by integrating a contrastive reindex-
ing mechanism with a constrained generation cascade, each
targeting a different path of the knowledge base, as shown
in Figure 2. An additional example is provided in Ap-
pendix A. Specifically, forinter-class protection, we restruc-
ture the retriever’s index to enforce stronger semantic sepa-
ration across topic categories. This process begins with un-
supervised clustering using HDBSCAN to reveal latent topic
structures. Then, we train a supervised contrastive encoder
that preserves inter-cluster margins, enabling better class
discrimination in the embedding space. As a result, it be-
comes significantly harder for attackers to retrieve content
from unrelated topics.
Forintra-class protection, we develop a cascade genera-
tion mechanism that improves both generation quality and
defense robustness. This design combines a fast draft model
with a stronger verifier model. During inference, the draft
model proposes candidate tokens, which are selectively val-
idated by the reference model based on a rejection rule.
To mitigate the generation of sensitive content in cascaded
systems, we propose a computationally efficient rejection
rule and provide theoretical guarantees that its performance
closely approximates that of the optimal rule.
4.2 Inter-Class Protection
Our inter-class protection method proceeds in three stages:
unsupervised clustering, supervised contrastive learning,
and index replacement.
Pseudo-Labeling via HDBSCAN Clustering.We be-
gin by applying unsupervised clustering to identify latent
topic categories within the knowledge base. First, each
chunkk i∈ K=k 1, . . . , k Nis embedded using a pre-
trained encoder (e.g., Sentence-BERT) to obtain dense rep-
resentationsE(k i). Next, we use Hierarchical Density-
Based Spatial Clustering of Applications with Noise (HDB-
SCAN) (Campello, Moulavi, and Sander 2013) to parti-
tion the chunks intoCclustersC 1, ...,C C, thereby assign-
ing a pseudo-labely i∈1, ..., Cto eachk i. Notably, HDB-SCAN is chosen due to its capability to automatically infer
the number of clusters, handle outliers, and adapt to high-
dimensional semantic spaces.
Learning Class-Separable Embeddings via SupCon
Loss.Using the pseudo-labels from clustering, we train a
new encoder with supervised contrastive learning (SupCon)
to produce class-discriminative embeddings. Given a mini-
batch of encoded chunks{z 1, . . . , z B}with corresponding
cluster labels{y 1, . . . , y B}, the SupCon loss is defined as:
Lsup=BX
i=1−1
|P(i)|X
p∈P(i)logexp(sim(z i, zp)/τ)P
a∈A(i)exp(sim(z i, za)/τ),
(1)
whereP(i)is the set of samples with the same label asi
(excludingi),A(i)includes all samples in the batch except
i, and sim(·,·)denotes cosine similarity.τis the tempera-
ture scaling factor. This contrastive objective strengthens the
separation between different semantic categories, improving
the structural integrity of indexed representations.
Reindexing with Tokenizer-Aligned Contrastive Embed-
dings.After contrastive training, we obtain a structure-
aware encoderf sup(·)that outputs enhanced chunk embed-
dingsz′
i=f sup(ki). We then construct a new dense index for
the retriever using these updated representations. Notably,
these embeddings are generated based on a custom tokenizer
aligned with the SupCon training process.
To preserve the generative performance of the original
RAG system, we do not modify the query encoder or the de-
coder input pipeline. Instead, during inference, the retriever
operates on the SupCon-based index, while the generator
still consumes the original (non-contrastive) chunk content
encoded using the original tokenizer.
4.3 Intra-Class Protection via Cascade
To enhance intra-class protection in RAG systems, we in-
troduce a cascaded generation framework combining a fast
draft model with a robust reference model. Letq(z|x <t, z)
denote the probability of generating chunkzunder the orig-
inal draft modelq. The objective of cascade is to construct
a new generation distributionπ= (1−r(x <t, z))·q t+
r(x<t, z)·p t, wherer(x <t, z)is a rejection rule that can
only take the values 0 or 1. If we want to improve inference
efficiency and generation quality and reduce the sampling
probability ofπ(z|x <t, z), we need to design an optimal re-
jection ruler. The optimal rejection rulershould minimize
the loss of the large model supervision under the cascade dis-
tributionπ. To control the cost of fallback top, we minimize
the expected loss under a constraint on the rejection budget
B. Meanwhile, to reduce the probability of chunkzbeing
generated, we introduce the thresholdC, which requires that
the probability of draft model generating sensitive content be
significantly lower than that of reference model:
min
rEy∼P(·|x <t,z)[(1−r(x <t, z))·ℓ(y, q t)
+r(x <t, z)·ℓ(y, p t)]
s.t.r(x <t, z)·D TV(pt, qt)≤B,
(1−r(x <t, z))·qt(z)
pt(z)≤C.(2)

Structure-aware 
Encoder
Embedding Space
Before Inter-Class Defend After Inter-Class Defend
HDBSCAN
ClusteringSupCon
Training
Theme Clustering
After Intra-Class Defend
Draft 
Model Distribution of Model  
tokens
Verifier 
ModelGenerate 
Next Token
Rejection 
Rule
LLMNext Circle
Chunk
Distribution Shift
ChunkStructure-aware
Embedding Space
Before Intra-Class Defend
TOriginal RAG
Embedding 
Encoder
Embedding 
Encoder
Leakage 
Output User Query
LLMRelevant 
ChunksAfter Protection
Secured 
Output User Query
Speculative 
CascadeStructure-aware 
Encoder
Relevant 
Chunks
Probability Drop
Figure 2: An overview of RAGFort. For inter-class protection, we introduce a structure-aware encoder trained to cluster and
separate topics in the embedding space, making it harder for attackers to retrieve content across categories. For intra-class
protection, candidate tokens proposed by a draft model are rigorously screened by a verifier model through a tailored rejection
rule, which blocks sensitive or risky outputs.
Analysis of Rejection Rule.To ease optimization, we
transform it into an unconstrained problem using Lagrangian
relaxation:
L(r;x <t, z)
=Ey∼P(·|x <t,z)[r(x<t, z)·(ℓ(y, p t) +α·D TV(pt, qt))
+(1−r(x <t, z))·(ℓ(y, q t) +η·qt(z)
pt(z))],(3)
whereαandηare Lagrangian multipliers. The optimalrfor
minimizing (3) is given by:
Lemma 1LetP t(y) =P(y|x <t, z)is real distribution.
Then the optimal rejection rule is:
r∗(x<t, z)
=

1ifE y∼Pt[ℓ(y, q t)]>E y∼Pt[ℓ(y, p t)]
+α·D TV(pt, qt)−η·qt(z)
pt(z),
0otherwise.(4)
Intuitively, the cascade system will only choose to generate a
verifier model when the expected loss of the verifier model is
significantly too high. SinceP tis unknown, we approximate
the expected loss using confidence scores:
ˆr(x<t, z) = 1if and only if
max
yqt(y)<max
ypt(y)−α·D TV(pt, qt) +η·qt(z)
pt(z).(5)
This rule triggers fallback whenp tis sufficiently more con-
fident thanq t. Next, we analyze the regret (performance gap)
between the simple rejection ruleˆrand the optimal rejection
ruler.
Lemma 2LetP t(y) =P(y|x <t, z)is real distribution.
Then for fixedx <tandz, we have
L(ˆr;x <t, z)−min
rL(r;x <t, z)
≤max
y∈Y|Pt(y)−q t(y)|+ max
y∈Y|Pt(y)−p t(y)|.(6)This result shows that the regret is small when bothq tand
ptare close to the ground-truth distributionP t.
The algorithm 1 describes a cascading generation process
in which the draft model first proposes a block ofγtokens
and calculates the probabilityq t. For each token position,
the verifier model concurrently computes the probabilityp t
and the rejection ruler, and then usesrto generateπ. Sub-
sequently, the earliest rejection positionj∗is sought. If a
rejection occurs, a residual distribution is constructed to op-
timize token generation atj∗; otherwise, the draft output
is retained, and the verifier model is used to generate the
starting token for the next block. Importantly, we utilize the
Bernoulli distribution during the rejection process. This en-
sures that the entire generation process starts fromqbut ulti-
mately generates tokens whose distribution does not deviate
from the verifier modelp(Yan, Agarwal, and Venkataraman
2024; Zhou et al. 2023).
5 Experiments
5.1 Evaluation Settings
RAG Architecture and Implementation Details.We im-
plement a standard RAG pipeline using the LangChain
framework (Chase 2022). Each data sample (either a
QA pair or a document) is treated as a distinct chunk
in the knowledge base. At inference time, the built-
in LangChain retriever, which operates on dense vector
similarity, retrieves the top 5 relevant chunks for each
query. The retrieved chunks are then concatenated with
the question to form the input to the generator, follow-
ing the prompt format: “Context: [...] Question:
[...] Answer:”. For the generator, we evaluate three
large language models: Qwen-14B (Jiang et al. 2023),
a high-capacity open-source model; DeepSeek-R1-Distill-
Llama-8B (abbreviated asDeepSeek-R1-8B) (Guo et al.
2025), a lightweight distilled variant of the LLaMA se-
ries optimized for chat-based generation; and Gemma-3-
27B (Team et al. 2025), a strong 27B-parameter model fea-
turing enhanced reasoning and factual consistency through
large-scale multi-domain instruction tuning. All RAG sys-

Datasets ModelsWithout protection RAGFort (Ours) Re-ranking Summarization
Worm-attack RAG-Thief Worm-attack RAG-Thief Worm-attack RAG-Thief Worm-attack RAG-Thief
HealthCareMagicQwen-14B 17.60±1.5857.16±2.138.84±0.9127.96±1.3216.28±1.1756.44±2.0815.44±1.2047.24±1.81
DeepSeek-R1-8B 13.92±1.2951.64±1.928.56±0.8326.72±1.1812.64±0.9449.28±1.7612.88±0.9142.52±1.54
Gemma-3-27B 18.36±1.3359.88±1.719.76±0.8829.44±1.1118.24±0.9657.36±1.8416.12±0.9747.28±1.62
Enron EmailQwen-14B 31.60±1.7352.60±1.9514.30±1.0120.60±0.8928.65±1.5844.35±1.4925.35±1.4346.75±1.66
DeepSeek-R1-8B 28.75±1.4547.85±1.8810.15±0.7517.65±0.6923.45±1.1243.75±1.3724.75±1.1640.65±1.31
Gemma-3-27B 32.35±1.4154.15±1.9914.40±0.7124.55±0.7628.85±1.2846.50±1.5227.60±1.1347.30±1.43
Math QAQwen-14B 47.84±2.2470.31±2.8729.06±1.0238.13±1.4139.49±1.9074.69±2.9935.31±1.7868.44±2.67
DeepSeek-R1-8B 43.13±2.1167.81±2.5428.44±1.3937.81±1.8740.00±1.9758.75±2.0141.88±1.9155.63±1.88
Gemma-3-27B 48.75±2.2669.06±2.6229.69±0.9838.44±1.3843.13±1.7863.44±1.9543.44±1.5164.69±1.57
Relative Mean CRR 1×0.51×0.91×0.87×
Table 1: Security assessment of RAGFort in preventing knowledge base extraction attack. We report the Chunk Recovery Rate
(CRR, %) of RAG systems under two black-box attack strategies:Worm-AttackandRAG-Thief. Each column block corresponds
to a specific defense method—Without protection,RAGFort (Ours),Re-ranking, andSummarization—while each sub-column
reports CRR under a specific attack. Lower CRR↓indicates stronger protection. The last row shows the average attack CRR as
a multiple of the baseline without protection (×).
Algorithm 1: CASCADES
Input:Draft modelq, Reference modelp, Rejection ruler,
Chunkz, Prefixx <t, Block sizeγ
Output:x t, . . . , x t+j∗
1:forj= 0toγ−1do
2:Computeq t+j(· |x <t+j, z)andq t(z)
3:end for
4:Computep t(z)
5:forj= 1toγdo
6:Computep t+j(· |x <t+j, z)andr t+j(x<t+j, z)
7:π t+j(·)←(1−r(x <t+j, z))·q+r(x <t+j, z)·p
8:end for
9:j∗=γ
10:forj= 0toγ−1do
11:ifr t+j(x<t+j, z) = 0then
12:continue
13:else
14:a j∼Bernoulli
minn
1,pt+j(xt+j)
qt+j(xt+j)o
15:ifa j= 0thenj∗=jand break
16:end if
17:end if
18:end for
19:ifj∗< γthen
20:p cas(·)←norm(max{0, π t+j∗(·)−q t+j∗(·)})
21:else
22:p cas(·)←π t+γ
23:end if
24:Samplex t+j∗∼p ref(·)
tems share the same retriever, prompting format, and de-
coding settings to ensure a fair comparison. To implement
cascade generation in RAGFort, we adopt a two-stage setup
with a draft and a verifier model. For Qwen-14B, we use
Qwen-7B as the draft model. For Gemma-3-27B, we use
Gemma-3-4B as the draft model. For DeepSeek-R1-8B, the
same model serves both roles, simulating scenarios where
no smaller model is available. We apply different decoding
temperatures for the draft and verifier. This setup shows that
our defense remains effective even without auxiliary models.Dataset and Evaluation Protocol.We evaluate on
three datasets spanning different domains: (1)Health-
CareMagic(Pampari et al. 2018): A medical question-
answering dataset containing symptom descriptions, diag-
noses, and treatment suggestions across various special-
ties. (2)Enron Email(Klimt and Yang 2004): A collection
of corporate email communications covering diverse top-
ics, organizational roles, and business contexts. (3)Math
QA(Amini et al. 2019): A dataset consisting of math word
problems and multiple-choice solutions that require numer-
ical reasoning and problem-solving skills.
Each RAG system is evaluated using QA pairs con-
structed from the corresponding dataset. We use GPT-4-
turbo (gpt-4-0125-preview) (OpenAI 2024) as an automatic
evaluator to determine whether each generated answer is se-
mantically consistent and factually aligned with the ground
truth. A response is considered correct if it meets these crite-
ria. The overall performance is measured by the percentage
of correct responses, reported asAnswer Accuracy (ACC).
Attack Strategies.We evaluate defense robustness under
the two most widely recognized attack strategies: (1)Worm-
Attack(Cohen, Bitton, and Nassi 2024): Utilizes adversar-
ial, self-replicating prompts to trigger cascading extraction
and propagation across RAG-based applications, simulat-
ing a computer worm in the GenAI ecosystem. Each ex-
periment runs 1,000 attack rounds with 10 collision vec-
tor search iterations per round. (2)RAG-Thief(Jiang et al.
2024): An agent-based attack that maintains memory of
extracted chunks and iteratively generates new adversarial
queries through self-improvement, enabling scalable extrac-
tion from the knowledge base. Each round issues 10 queries,
initialized with 300, 250, and 50 GPT-generated prompts for
HealthCareMagic, EnronMail, and MathQA, respectively
(final chunk sets: 2,500, 2,000, and 320).
Baseline Defenses.Our proposed method is compared to
two recently proposed defense baselines: (1)Re-ranking
Protection(Zeng et al. 2024a): an inter-class protection that
enforces semantic similarity thresholds during retrieval to
filter loosely related documents and limit inter-class expo-
sure. (2)Summarization Protection(Zeng et al. 2024b):
Substitutes retrieved passages with abstract summaries to

Datasets ModelsWithout protection RAGFort (Ours) RAGFort InterOnly RAGFort IntraOnly
Worm-attack RAG-Thief Worm-attack RAG-Thief Worm-attack RAG-Thief Worm-attack RAG-Thief
HealthCareMagicQwen-14B 17.60±1.5857.16±2.138.84±0.9127.96±1.3214.52±0.6241.16±1.7113.62±0.8351.28±2.57
DeepSeek-R1-8B 13.92±1.2951.64±1.928.56±0.8326.72±1.189.04±0.4539.56±2.2211.51±0.6842.36±2.67
Gemma-3-27B 18.36±1.3359.88±1.719.76±0.8829.44±1.1115.28±0.5146.32±1.3514.72±0.7351.56±2.31
Enron EmailQwen-14B 31.60±1.7352.60±1.9514.30±1.0120.60±0.8919.75±0.9033.25±1.2526.10±1.7536.70±1.40
DeepSeek-R1-8B 28.75±1.4547.85±1.8810.15±0.7517.65±0.6924.15±0.9531.40±1.1522.55±1.6527.85±2.60
Gemma-3-27B 32.35±1.4154.15±1.9914.40±0.7124.55±0.7624.85±1.1234.85±1.2227.35±1.3045.05±1.95
Math QAQwen-14B 47.84±2.2470.31±2.8729.06±1.0238.13±1.4130.94±0.9154.61±2.1943.75±1.5165.63±2.58
DeepSeek-R1-8B 43.13±2.1167.81±2.5428.44±1.3937.81±1.8741.25±1.6152.19±1.6842.19±2.2353.78±1.74
Gemma-3-27B 48.75±2.2669.06±2.6229.69±0.9838.44±1.3837.81±1.4354.06±2.0244.06±1.7862.19±2.26
Relative Mean CRR 1×0.51×0.75×0.83×
Table 2: Ablation study of RAGFort defense modules against knowledge base extraction attacks. We report the Chunk Re-
covery Rate (CRR, %) for the fullRAGFortand its two path variants under different attack strategies, where a lower CRR
indicates stronger protection.RAGFort InterOnly includes only the inter-class protection module (contrastive reindexing), while
RAGFort IntraOnly includes only the intra-class protection module (cascade).
Datasets ModelsBefore protection After protection
ACC FLOPs (T) ACC FLOPs (T)
HealthCareMagicQwen-14B 68.16±1.4538.03±0.2766.57±1.2219.40±0.35
DeepSeek-R1-8B 61.36±0.6120.37±0.2361.12±0.4420.83±0.24
Gemma-3-27B 69.64±0.7584.83±0.5868.88±0.3634.33±0.21
Enron EmailQwen-14B 99.10±0.11106.14±0.3098.85±0.2556.84±0.40
DeepSeek-R1-8B 97.50±0.1556.90±0.1597.60±0.1757.43±0.48
Gemma-3-27B 99.25±0.21236.47±0.8699.15±0.3891.40±0.33
Math QAQwen-14B 85.31±1.6347.09±0.3582.81±1.5119.30±0.37
DeepSeek-R1-8B 78.44±1.4925.22±0.1577.19±2.4725.37±0.36
Gemma-3-27B 87.50±0.41105.21±0.2285.94±1.2631.95±0.31
Table 3: System performance before and after applying
RAGFort protection. We report the Answer Accuracy (ACC,
%) and computational cost (FLOPs (T)) of RAG systems
with and without RAGFort.
obscure fine-grained content and reduce intra-class leakage.
Security Metrics.Following prior work (Jiang et al.
2024), we introduce theChunk Recovery Rate (CRR),
which jointly captures both lexical and semantic similarity
between generated outputs and original knowledge chunks.
A chunk is counted as recovered only if it satisfies both
of the following criteria: (1)ROUGE-L(Lexical Similar-
ity): The ROUGE-L score between the generated output and
the original chunk exceeds a predefined threshold of 0.5.
(2)Semantic Similarity: The cosine similarity between the
embedding vectors of the generated and original text, com-
puted using a pre-trained model such as Sentence-BERT or
BERTScore, exceeds a semantic threshold of 0.85.
5.2 Security
To evaluate defense effectiveness, we simulate knowledge
extraction attacks and assess how well each method pre-
vents unauthorized replication. The attacker interacts with
the RAG system to reconstruct a proxy knowledge base, and
we measure defense performance by the number of recov-
ered chunks. Lower values indicate stronger protection.
As shown in Table 1, RAGFort achieves the lowest Chunk
Recovery Rate (CRR) across both Worm-Attack and RAG-
Thief, demonstrating significantly stronger protection than
existing defenses. Specifically, RAGFort reduces the Rela-
tive Mean CRR to just 0.51× of the original. By compari-
son, re-ranking and summarization achieve only 0.91× and
0.87×, respectively, indicating substantially weaker protec-tion. These results validate the effectiveness of our dual-path
design: contrastive reindexing restricts inter-class exposure,
while cascade suppresses intra-class leakage, jointly block-
ing both attack paths.
5.3 System Performance
To assess the impact on system performance of RAGFort,
we compare QA accuracy and FLOPs before and after pro-
tection (as shown in Table 3). Results show that RAGFort
almost does not affect system performance: for all mod-
els and datasets, the ACC drops by less than 2 percent-
age points (e.g., Qwen-14B on HealthCareMagic: 68.16→
66.57; DeepSeek-R1-8B: 61.36→61.12), and FLOPs re-
main nearly unchanged or even decrease.
We believe the small accuracy drop mainly arises from the
gap between the draft and original models, not the defense
itself. Specifically, Qwen-14B, which uses a smaller draft
model, sees a slightly greater drop than DeepSeek-R1-8B,
which uses the same model for both stages. Thus, we rec-
ommend stronger draft models when accuracy is a priority.
Regarding efficiency, in the Qwen-14B group, the decrease
in FLOPs results from replacing large-model inference with
small-model inference, while in DeepSeek-R1-8B, which
uses the same model as the draft model, FLOPs are nearly
unchanged. This indicates that even when no smaller draft
model is available, RAGFort introduces negligible overhead.
5.4 Ablation Study
To assess the individual contributions of each defense mod-
ule, we perform an ablation study by isolating the inter-class
and intra-class protection components. Similar to the secu-
rity evaluation, we apply attacks on RAG systems equipped
with only one defense module and measure the resulting
chunk recovery rate and QA accuracy. This allows us to
quantify the security gains provided by each component and
validate their complementary roles. As shown in Table 2,
neither RAGFort InterOnly nor RAGFort IntraOnly alone provides
sufficient defense. For instance, under RAG-Thief, the full
RAGFort reduces CRR from 57.16% to 27.96% (Health-
CareMagic, Qwen-14B), whereas the single-module vari-
ants still expose over 40% of content. Averaged across all
settings, the full method lowers the relative mean CRR to
0.51×, significantly outperforming RAGFort InterOnly (0.75×)

and RAGFort IntraOnly (0.83×). These results demonstrate that
inter-class and intra-class protection are complementary,
only their joint deployment can provide robust and compre-
hensive defense against knowledge extraction attacks.
6 Limitation and Discussion
Experimental Scope and Systematic Evaluation.Due to
page and resource constraints, we report only the primary
experiments, including the dual-path defense comparison,
security evaluation, system performance, and ablation study.
Further studies can assess the robustness and practicality of
RAGFort from more perspectives, such as parameter sensi-
tivity and practical deployment efficiency.
7 Conclusion
In this paper, we address the problem of protecting pro-
prietary knowledge bases in retrieval-augmented generation
systems. Through systematic experiments, we demonstrate
that effective protection requires jointly securing both intra-
class and inter-class paths. Based on these insights, we pro-
pose RAGFort, a dual-path defense combining contrastive
reindexing for inter-class isolation and cascade generation
for intra-class robustness. Experimental results show that
RAGFort significantly reduces knowledge base reconstruc-
tion while maintaining high answer quality and efficiency.
To the best of our knowledge, RAGFort is the first systematic
framework addressing dual-path protection in RAG systems,
offering a practical solution for secure RAG deployments.Acknowledgements
This work was supported by the Key Project of the Na-
tional Natural Science Foundation of China under Grant no.
62536007, the Zhejiang Province Science Foundation under
Grant no. LD24F020002 and the Zhejiang Province’s 2025
“Leading Goose + X” Science and Technology Plan under
Grant no. 2025C02034.
References
Amini, A.; Gabriel, S.; Lin, P.; Koncel-Kedziorski, R.; and
Hajishirzi, H. 2019. MathQA: Towards interpretable math
word problem solving with operation-based formalisms. In
NAACL.
Brown, T. B.; Mann, B.; Ryder, N.; et al. 2020. Language
models are few-shot learners.NeurIPS.
Campello, R. J.; Moulavi, D.; and Sander, J. 2013. Density-
based clustering based on hierarchical density estimates. In
Pacific-Asia Conference on Knowledge Discovery and Data
Mining, 160–172. Springer.
Chase, H. 2022. LangChain. https://github.com/hwchase17/
langchain. Accessed: 2025-06-23.
Cohen, S.; Bitton, R.; and Nassi, B. 2024. Unleashing
worms and extracting data: Escalating the outcome of at-
tacks against rag-based inference in scale and severity using
jailbreaking.arXiv preprint arXiv:2409.08045.
Guo, D.; Yang, D.; Zhang, H.; Song, J.; Zhang, R.; Xu, R.;
Zhu, Q.; Ma, S.; Wang, P.; Bi, X.; et al. 2025. Deepseek-r1:
Incentivizing reasoning capability in llms via reinforcement
learning.arXiv preprint arXiv:2501.12948.
Izacard, G.; and Grave, E. 2021. Towards faithful retrieval-
augmented generation: A benchmark, evaluation and model.
InNeurIPS.
Jiang, C.; Pan, X.; Hong, G.; Bao, C.; and Yang, M. 2024.
Rag-thief: Scalable extraction of private data from retrieval-
augmented generation applications with agent-based at-
tacks.arXiv preprint arXiv:2411.14110.
Jiang, Y .; Hua, W.; Qin, Y .; and et al. 2023. Qwen: Scaling
Up Language Models with Decoupled Parameter and Data
Growth.
Klimt, B.; and Yang, Y . 2004. The Enron corpus: A new
dataset for email classification research. InEuropean Con-
ference on Machine Learning, 217–226. Springer.
Lewis, P.; Perez, E.; Piktus, A.; et al. 2020. Retrieval-
augmented generation for knowledge-intensive NLP tasks.
InNeurIPS.
Narasimhan, H.; Jitkrittum, W.; Rawat, A. S.; Kim, S.;
Gupta, N.; Menon, A. K.; and Kumar, S. 2024. Faster
Cascades via Speculative Decoding.arXiv preprint
arXiv:2405.19261.
Ni, B.; Liu, Z.; Wang, L.; Lei, Y .; Zhao, Y .; Cheng,
X.; Zeng, Q.; Dong, L.; Xia, Y .; Kenthapadi, K.; et al.
2025. Towards Trustworthy Retrieval Augmented Genera-
tion for Large Language Models: A Survey.arXiv preprint
arXiv:2502.06872.
OpenAI. 2023. GPT-4 Technical Report. https://openai.com/
research/gpt-4.

OpenAI. 2024. GPT-4 Turbo Technical Report. https:
//openai.com/gpt-4-turbo. Accessed: July 2025.
Pampari, A.; Raghavan, P.; Liang, J.; and Yu, R. J. M. 2018.
emrQA: A large corpus for question answering on electronic
medical records. InEMNLP.
Singhal, K.; Azizi, S.; Tu, T.; et al. 2022. Large lan-
guage models encode clinical knowledge.arXiv preprint
arXiv:2212.13138. Demonstrates retrieval-augmented
prompting on medical QA tasks.
Team, G.; Kamath, A.; Ferret, J.; Pathak, S.; Vieillard, N.;
Merhej, R.; Perrin, S.; Matejovicova, T.; Ram ´e, A.; Rivi `ere,
M.; et al. 2025. Gemma 3 technical report.arXiv preprint
arXiv:2503.19786.
Wu, S.; et al. 2023. BloombergGPT: A large language model
for finance.arXiv preprint arXiv:2303.17564. Includes
experiments combining BloombergGPT with retrieval-
augmented prompting.
Yan, M.; Agarwal, S.; and Venkataraman, S. 2024. Decoding
speculative decoding.arXiv preprint arXiv:2402.01528.
Zeng, S.; Zhang, J.; He, P.; Liu, Y .; Xing, Y .; Xu, H.; Ren,
J.; Chang, Y .; Wang, S.; Yin, D.; et al. 2024a. The Good and
The Bad: Exploring Privacy Issues in Retrieval-Augmented
Generation (RAG). InFindings of the Association for Com-
putational Linguistics ACL 2024, 4505–4524.
Zeng, S.; Zhang, J.; He, P.; Ren, J.; Zheng, T.; Lu, H.; Xu,
H.; Liu, H.; Xing, Y .; and Tang, J. 2024b. Mitigating the pri-
vacy issues in retrieval-augmented generation (rag) via pure
synthetic data.arXiv preprint arXiv:2406.14773.
Zhou, Y .; Lyu, K.; Rawat, A. S.; Menon, A. K.; Ros-
tamizadeh, A.; Kumar, S.; Kagy, J.-F.; and Agarwal, R.
2023. Distillspec: Improving speculative decoding via
knowledge distillation.arXiv preprint arXiv:2310.08461.

A Illustrative Case of the RAGFort Process
USER Query: “What are the main 
effects of beta blockers in treating 
cardiovascular conditions, and are 
there any other considerations?”“Beta blockers are commonly 
prescribed for heart arrhythmias and 
hypertension. They reduce the heart’s 
workload and oxygen demand.”
“Beta blockers may cause dizziness, 
fatigue, and in rare cases, short-term 
memory loss, particularly in older 
adults.”
“Propranolol, a non-selective beta 
blocker, crosses the blood-brain 
barrier and is associated with vivid 
dreams or depression in sensitive 
individuals.”Output: “Beta blockers, such as 
propranolol, are effective in 
treating cardiac conditions like 
arrhythmias. However, due to 
their ability to cross the blood-
brain barrier, they may lead to 
neurological side effects 
including fatigue, dizziness, or 
short-term memory loss, 
especially in older adults.”
“Beta blockers are commonly 
prescribed for heart arrhythmias and 
hypertension. They reduce the heart’s 
workload and oxygen demand.”
“Patients with ischemic heart disease 
benefit from beta blockers through 
reduced angina and myocardial 
oxygen consumption.”
“Long-term beta blocker therapy 
improves survival in patients with 
systolic heart failure by reducing 
arrhythmic events.”Cardiology
PharmacologyNeurology
Output: “Beta blockers are 
frequently prescribed to treat 
heart arrhythmias, hypertension, 
and systolic heart failure. They 
reduce the heart’s workload and 
lower myocardial oxygen 
demand, which helps alleviate 
angina in patients with ischemic 
heart disease. Over the long 
term, this therapy has been 
shown to improve survival by 
reducing arrhythmic events.”
Output: “Beta blockers are 
frequently prescribed to treat 
irregular heartbeat, high blood 
pressure, and weakened heart 
muscle performance. By 
lessening cardiac strain and 
enhancing efficiency, they help 
manage chest discomfort in 
certain patients. Clinical use 
over time may be associated 
with improved outcomes in 
individuals at risk of electrical 
rhythm instability.”Cardiology
Cardiology
CardiologyStructure-aware 
Encoder Embedding
Encoder Original LLM
Speculative 
Cascade
Cross-domain 
leakage !
Direct chunk 
leakage !
Protected 
generation
Figure 3: A representative example illustrating how RAGFort jointly applies inter-class and intra-class protection during infer-
ence.

B Proofs of Lemmas
B.1 Proof of Lemma 1
Proof.Expanding the deferral risk in Equation (3), we have:
Lspec(r;x <t, z)
=r(x <t, z)·(E xt∼P(·|x <t,z)[ℓ(xt, pt)] +α·D TV(pt, qt)
−η·qt(z)
pt(z)−Ext∼P(·|x <t,z)[ℓ(xt, qt)])
+Ext∼P(·|x <t,z)[ℓ(xt, qt)] +qt(z)
pt(z).
(7)
This objective is minimized by a deferral ruler:Yt−1→
{0,1}that, for each prefixx <t, chooses the value ofr(x <t)
minimizing the total loss. Therefore, the optimal decision
r∗(x<t) = 1if and only if the term within the parentheses
is negative:
Ext∼P(·|x <t,z)[ℓ(xt, pt)] +α·D TV(pt, qt)
−η·qt(z)
pt(z)−Ext∼P(·|x <t,z)[ℓ(xt, qt)]<0,(8)
andr∗(x<t) = 0otherwise. Re-arranging the terms com-
pletes the proof.
B.2 Proof of Lemma 2
For a fixed prefixx <tandz, we can write the deferral risk
in Equation (3) as:
Lspec(r;x <t, z)
=r(x <t, z)·(E xt∼P(·|x <t,z)[ℓ(xt, pt)] +α·D TV(pt, qt)
−η·qt(z)
pt(z)−Ext∼P(·|x <t,z)[ℓ(xt, qt)]) +Const,
(9)
whereConstis a constant independent of the deferral rule.
Letr∗:Yt−1→ {0,1}denote the optimal deferral rule.
Then for any prefixx <tandz:
Lspec(ˆrOPT;x<t, z)−L spec(r∗;x<t, z)
=(ˆr OPT(x<t, z)−r∗(x<t, z))·(E xt∼P[ℓ(xt, pt)]
+α·D TV(pt, qt)−η·qt(z)
pt(z)−Ext∼P[ℓ(xt, qt)]).(10)
Adding and subtractingmax yqt(y)−max ypt(y)in the
parentheses:
Lspec(ˆrOPT;x<t, z)−L spec(r∗;x<t, z)
=(ˆr OPT(x<t, z)−r∗(x<t, z))·(max
yqt(y) +α·D TV(pt, qt)
−η·qt(z)
pt(z)−max
ypt(y))
+(ˆr OPT(x<t, z)−r∗(x<t, z))·(E xt∼P[ℓ(xt, pt)]−
Ext∼P[ℓ(xt, qt)]−max
yqt(y) + max
ypt(y)).
(11)Using|ˆr OPT(x<t, z)−r∗(x<t, z)| ≤1, we upper-bound
each term:
Term 1= max
yqt(y) +α·D TV(pt, qt)−η·qt(z)
pt(z)−max
ypt(y),
Term 2=Ext∼P[ℓ(xt, pt)]−1 + max
ypt(y),
Term 3= 1−max
yqt(y)−E xt∼P[ℓ(xt, qt)].
(12)
Hence, we have
Lspec(ˆrOPT;x<t, z)−L spec(r∗;x<t, z)
≤Term 1+Term 2+Term 3.(13)
We now bound each term. For Term 1, we consider two
cases: (i) Ifmax yqt(y) +α·D TV(pt, qt)−η·qt(z)
pt(z)−
max ypt(y)≤0, thenˆr OPT(x<t, z) = 1, and
Term 1≤max
ypt(y)+α·D TV(pt, qt)−η·πt(z)
qt(z)−max
yqt(y)≤0.
(14)
(ii) Otherwise,ˆr OPT(x<t, z) = 0, and Term 1= 0regard-
less ofr∗(x<t, z). Therefore, Term 1≤0. Next, for Term 2,
usingℓ=ℓ 0−1, we have:
Term 2≤ |Term 2|
=|Ext∼P[ℓ(xt, pt)]−1 + max
ypt(y)|
=|X
xt∈YP(xt|x<t, z)·I{x t̸= arg max
ypt(y)} −1 + max
ypt(y)|
=|max
ypt(y)−X
xt∈YP(xt|x<t, z)·I{x t= arg max
ypt(y)}|
=|pt(y∗)−P(y∗|x<t, z)| ≤max
y|pt(y)−P(y|x <t, z)|,
(15)
wherey∗= arg max ypt(y).
Similarly, for Term 3:
Term 3≤max
y|qt(y)−P(y|x <t, z)|. (16)
Substituting the bounds for Term 1–Term 3into Equa-
tion (13) completes the proof.