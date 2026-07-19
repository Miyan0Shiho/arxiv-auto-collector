# Is External Database Protection Static in Retrieval-Augmented Generation? Rethinking Privacy Preservation under Dynamic Queries

**Authors**: Gang Zhang, Mingyu Tian, Xukun Luan, Yuanchi Ma, Jinyan Liu

**Published**: 2026-07-16 10:28:23

**PDF URL**: [https://arxiv.org/pdf/2607.14811v1](https://arxiv.org/pdf/2607.14811v1)

## Abstract
Retrieval-augmented generation (RAG) enhances large language models via external document retrieval, but retrieved contexts may leak sensitive information. Current privacy protection methods typically rely on a document-level static risk assumption, treating all retrieved documents as having the same privacy leakage risk. However, this assumption overlooks a fundamental characteristic of RAG: the privacy risk of a document is highly dependent on the user's query, making privacy leakage inherently query-driven and dynamic. To address this challenge, we propose a Prompt-Aware Dynamic Hierarchical Differential Privacy framework (PA-HDP) for privacy-preserving RAG. PA-HDP first performs a prompt-aware risk hierarchy to dynamically assess privacy risks under different queries. It then applies adaptive sensitive entity replacement and exponential mechanism-based text selection to provide differentiated privacy protection while preserving semantic utility. By protecting only the content that is truly sensitive under a given query, PA-HDP minimizes unnecessary modifications to the retrieval corpus. Extensive experiments on benchmark datasets demonstrate that PA-HDP significantly reduces privacy leakage while maintaining high retrieval quality, achieving a better privacy-utility trade-off than prior methods.

## Full Text


<!-- PDF content starts -->

JOURNAL OF L ATEX CLASS FILES, VOL. 14, NO. 8, AUGUST 2021 1
Is External Database Protection Static in
Retrieval-Augmented Generation? Rethinking
Privacy Preservation under Dynamic Queries
Gang Zhang1, Mingyu Tian1, Xukun Luan1, Yuanchi Ma2, Jinyan Liu1
1School of Computer Science and Technology, Beijing Institute of Technology , Beijing, China
2Department of Computer Science and Technology, Tsinghua University, Beijing, China
Abstract—Retrieval-augmented generation (RAG) enhances
large language models via external document retrieval, but re-
trieved contexts may leak sensitive information. Current privacy
protection methods typically rely on a document-level static
risk assumption, treating all retrieved documents as having the
same privacy leakage risk. However, this assumption overlooks
a fundamental characteristic of RAG: the privacy risk of a
document is highly dependent on the user’s query, making
privacy leakage inherently query-driven and dynamic. To address
this challenge, we propose a Prompt-Aware Dynamic Hierarchical
Differential Privacy framework (PA-HDP) for privacy-preserving
RAG. PA-HDP first performs a prompt-aware risk hierarchy
to dynamically assess privacy risks under different queries. It
then applies adaptive sensitive entity replacement and exponential
mechanism-based text selection to provide differentiated privacy
protection while preserving semantic utility. By protecting only
the content that is truly sensitive under a given query, PA-
HDP minimizes unnecessary modifications to the retrieval corpus.
Extensive experiments on benchmark datasets demonstrate that
PA-HDP significantly reduces privacy leakage while maintaining
high retrieval quality, achieving a better privacy–utility trade-off
than prior methods.
Index Terms—Retrieval-augmented generation; Differential
privacy; Privacy protection; Large language models.
I. INTRODUCTION
The emergence of Retrieval-Augmented Generation (RAG)
technology has significantly mitigated the core challenges
prevalent in large language models (LLMs) [1], such as factual
inaccuracies and content hallucinations [2]. By deeply integrat-
ing external knowledge bases with LLMs, RAG first retrieves
the most relevant document segments from the knowledge
repository based on the user’s query prior to response gen-
eration. These retrieved results are then fed into the model as
contextual information, thereby guiding the model to produce
more accurate and reliable outputs. Currently, RAG technol-
ogy has been widely adopted in high-stakes domains where
information accuracy is paramount, including healthcare [3],
finance [4], and government services [5].
However, when the external knowledge base of a RAG
system contains a large amount of sensitive information,
such as patient medical records and personally identifiable
information (PII), the retrieved text segments often contain
highly sensitive content [6]. This sensitive information may
be leaked by large language models through various means,
including direct verbatim reproduction, indirect inference, andgradual reconstruction, leading to uncontrollable privacy risks.
Multiple studies [7]–[9] have pointed out that RAG systems
face severe privacy and security challenges when the retrieval
process involves private data. Therefore, enhancing the privacy
protection capabilities of RAG systems and preventing the
leakage of retrieved data are crucial for preventing unautho-
rized access and misuse of data [10].
Differential Privacy (DP) [11] offers a mathematically rigor-
ous solution to this challenge. By injecting carefully calibrated
noise into the data, DP guarantees that an adversary cannot
determine with high confidence whether a specific individual’s
record is present in the dataset, even if they have access to
all other records. Several existing works have leveraged DP
to protect the privacy of retrieved contexts in RAG systems.
Koga et al. [12] proposed selective privacy budget allocation
to sensitive tokens to support accurate long-form generation
under constrained budgets. Grislain [10] further demonstrated
the feasibility of applying DP to token-level generation in
RAG. Beyond token-level protection, some works have also
explored document-level privacy mechanisms [13]. Mori et al.
[14] proposed a differentially private synthetic text generation
approach that replaces sensitive documents with DP-compliant
synthetic alternatives, mitigating leakage at the data source.
The aforementioned studies have provided valuable insights
into privacy protection for retrieved contexts in RAG systems.
However, these methods share a fundamental limitation: they
are all based on the document-level static privacy risk assump-
tion, which presumes that all retrieved documents carry an
equal risk of privacy leakage. This overlooks the critical fact
that the privacy leakage risk of the same document is highly
dependent on the specific content of the user’s query (prompt).
For instance, for the same internal company document, a query
asking to ”summarize the company’s cultural philosophy”
poses almost no privacy risk, while a query requesting to
”list the names and contact information of all department
employees” will lead to severe privacy leakage.This implies
that the privacy protection process in RAG systems should
essentially be a query-driven dynamic process rather than
the document-level static process currently widely adopted.
To address the inherent limitations of the aforementioned
static privacy protection methods, this paper proposes a
Prompt-Aware Dynamic Hierarchical Differential Privacy (PA-
HDP) protection method. In particular, this method introduces
arXiv:2607.14811v1  [cs.CR]  16 Jul 2026

JOURNAL OF L ATEX CLASS FILES, VOL. 14, NO. 8, AUGUST 2021 2
a prompt-driven fine-grained privacy risk assessment mecha-
nism into the retrieval phase of RAG systems. First, it performs
sentence-level sensitive semantic recognition on the retrieved
contexts based on the user’s query (prompt), and divides the
text segments into different privacy risk levels according to
the type of sensitive information, the severity of leakage harm,
and the relevance to the query intent. Then, for text content at
different risk levels, we design an adaptive candidate selection
strategy based on the exponential mechanism, where different
privacy budgets are assigned according to the risk level.
Specifically, a diverse candidate set is first generated through
semantically equivalent replacement of sensitive entities. The
exponential mechanism then selects a protected text from the
candidate set that satisfies differential privacy constraints while
preserving the highest possible semantic fidelity. This method
can achieve an optimal trade-off between privacy security and
model generation utility without compromising the semantic
coherence and information integrity of the original context.
Our experimental results show that using our algorithm can
achieve comparable performance with using original data
while substantially reducing the associated privacy risks.
Our contributions are summarized as follows :
1.An important yet underexplored research question
has been raised.We point out the prevalent flaw of
document-level static risk assumption in RAG privacy
protection and reveal the query-dependent dynamic pri-
vacy leakage problem.
2.A Prompt-Aware Dynamic Hierarchical Differential
Privacy framework (PA-HDP) is proposed.We pro-
pose a Prompt-Aware Dynamic Hierarchical Differential
Privacy protection framework. The framework integrates
risk stratification, adaptive sensitive entity replacement,
and exponential mechanism-based text selection to pro-
vide differentiated privacy protection while preserving
semantic utility.
3.The experiments verified the effectiveness of the pro-
posed method.Extensive experimental results demon-
strate that, under the premise of strictly satisfying the
mathematical constraints of differential privacy, the PA-
HDP method can significantly reduce the semantic loss of
generated content and achieves a better trade-off between
privacy and model utility.
II. RELATED WORK
Retrieval-Augmented Generation (RAG) improves the fac-
tuality of large language models but introduces new privacy
risks, as sensitive information contained in retrieved docu-
ments may be exposed through generated responses. Ensuring
strong privacy guarantees is therefore critical, with differential
privacy emerging as a principled and widely adopted frame-
work for mitigating information leakage. Existing privacy-
preserving RAG approaches can be broadly categorized into
three directions: (1) token-level generation protection based on
differential privacy, (2) protection based on external databases,
and (3) other privacy protection methods.A. Privacy protection for Token-level Generation
These methods directly apply differential privacy mecha-
nisms during the LLM decoding phase, injecting noise into
token probability distributions or logits to limit the leakage
of retrieved information. Early work [12] explored the funda-
mental challenges of differentially private RAG, namely how
to generate accurate long text under limited privacy budgets,
and proposed a strategy that allocates budgets only to sensitive
tokens. InvisibleInk [15] interprets next-token sampling as an
exponential mechanism over logits, reducing privacy costs by
isolating sensitive information in public versus private logits,
and samples from a superset of top-k private tokens to improve
utility, achieving an 8-fold reduction in computational cost
under the same privacy level. Privacy-Aware Decoding (PAD)
[16] is a lightweight inference-time defense that adaptively
adds calibrated Gaussian noise to token logits, combining
confidence filtering with efficient sensitivity estimation, pro-
tecting only high-risk tokens in a model-agnostic manner with
very low computational overhead. A complementary work
[10] further validates the feasibility of differential privacy
based token generation in private RAG scenarios. PEARL [17]
proposes an adaptive differential privacy decoding framework
based on confidence gaps and entropy regulation, dynamically
allocating privacy budgets at both token and sentence lev-
els, focusing protection on personally identifiable information
fragments.
B. Privacy protection for External Databases
The core objective of these methods is to directly protect
the retrieved original documents so that LLMs or attackers
cannot access sensitive content. V AGUE-Gate [13] proposes
a plug-and-play LDP protection mechanism that performs
budget-constrained token filtering and random rewriting per-
turbation on retrieved text, hiding the content of original
documents while satisfying𝜀-LDP. LPRAG [18] applies LDP
perturbation only to private entities (words, numbers, or
phrases) within the text, identifying entity types and allo-
cating adaptive privacy budgets, using different perturbation
mechanisms for each type. Differentially private synthetic
text [14] avoids leakage at the source by generating DP-
compliant synthetic documents to replace sensitive original
data. DP-KSA [19] adopts keyword semantic compression and
a propose-test-release mechanism to extract high-frequency
keywords from retrieved documents in a differentially private
manner, augmenting only the compressed keywords into the
prompt. Text-DP [20] proposes a textual differential privacy
paradigm for context-aware inference, achieving sensitive
information anonymization through differential embedding
hashing. RemoteRAG [21] formalizes the privacy protection
problem for cloud-based RAG services, introducing an(𝑛,𝜀)-
DistanceDP mechanism to perturb queries and document em-
beddings, narrowing the retrieval scope based on perturbed
embedding vectors. ppRAG [22] combines distance-preserving
symmetric encryption (CAPRISE) with the DistanceDP mech-
anism to enable similarity computation over encrypted embed-
dings in untrusted cloud environments, ensuring that plaintext
documents are never exposed.

JOURNAL OF L ATEX CLASS FILES, VOL. 14, NO. 8, AUGUST 2021 3
Another class of methods does not rely on differential
privacy, protecting original documents through synthetic data,
knowledge distillation, embedding shifting, or model unlearn-
ing. Pure synthetic data methods [7] replace the retrieval
database entirely with artificially generated text that contains
no sensitive information. Parametric RAG [23] adopts knowl-
edge distillation to convert each document into a parameterized
LoRA module, requiring no access to original documents
during inference, with document content masked by special
tokens. PRESS [24] fine-tunes the model via embedding
space shifting, enabling RAG inference without needing to
access original sensitive documents. Additionally, [25] studies
methods to erase private knowledge from multiple documents,
fine-tuning the model to actively forget sensitive information
from retrieved documents.
C. Privacy protection for Training stage
Beyond the direct protections targeting the generation pro-
cess and retrieved documents, researchers have also worked
on training-stage privacy, multi-query privacy accounting,
and system security evaluation. Regarding user-level pri-
vacy protection during the training phase of RAG systems,
Charles et al. [26] designs two scalable DP-SGD variants
(ELS example-level sampling and ULS user-level sampling)
and derives a tight user-level privacy accounting framework,
adaptively selecting user group sizes between privacy and
computational resources via data-driven heuristics; Boenisch
et al. [27] further proposes a personalized differential pri-
vacy method that assigns different sampling rates or clipping
norms to different users to achieve individualized protection
strength. In multi-turn interaction scenarios, continuous pri-
vacy budget consumption is a key challenge for practical
deployment. Private-RAG (MURAG) [28] shifts privacy ac-
counting from the query level to the document level through
document access frequency-based individual privacy filtering
and adaptive threshold release, effectively controlling privacy
accumulation in multi-turn interactions. Beyond Per-Query
Privacy [29] systematically studies privacy budget consump-
tion in online interactions, designing both fixed-threshold and
adaptive-threshold private retrieval algorithms, making privacy
consumption related to document retrieval frequency rather
than total query count, achieving more efficient document se-
lection and multi-turn QA generation under𝜀-DP guarantees.
III. PRELIMINARY
Differential privacy is a standard paradigm for protecting
privacy of individuals. It requires that changing one entry can
create only a small change of the output distributions.
Definition III.1.(Differential Privacy (DP) [30]). A ran-
domized mechanismM:X𝑛→Rsatisfies(𝜖,𝛿)-differential
privacy if for any two adjacent inputsD,D′and for any
subset of outputs𝑆⊆R, it holds that
Pr[M(D)∈𝑆]≤𝑒𝜖𝑃𝑟[M(D′)∈𝑆]+𝛿 .Lemma III.1(Laplace Mechanism [31]).Given a function
𝑓:X𝑛→Rwith sensitivityΔ𝑓=max D,D′adjacent𝑓(D)−
𝑓(D′), the Laplace mechanism outputs
M(D)=𝑓(D)+𝜂,
where𝜂∼Lap
Δ𝑓
𝜖
. ThenMsatisfies𝜖-differential privacy.
Lemma III.2.(Exponential Mechanism [32]) For a dataset
D, let𝑢(D,𝑟)be a utility function with respect to output
𝑟∈ R, and let△𝑢be the sensitivity of𝑢. The exponential
mechanismM(D,𝑢,R)selects an element𝑟∈Rwith proba-
bility proportional to
exp𝜖𝑢(D,𝑟)
2△𝑢
,
and this mechanism satisfies𝜖-differential privacy.
To adapt differential privacy to textual data, we introduce
Semantic Metric Differential Privacy (Definition III.2).
Definition III.2.[(𝜖,𝛿)-Semantic Metric Differential Privacy]
LetXdenote the text space, where each text𝑥∈ Xis
represented as a token sequence𝑥=(𝑤 1,...,𝑤𝑛). Define
an edit-based adjacency relation𝑥∼𝑥′such that𝑥′can be
obtained from𝑥by a single token- or phrase-level insertion,
deletion, or substitution operation, i.e.,𝑑 edit(𝑥,𝑥′)=1.Let
𝜙:X→R𝑑be a semantic embedding function, and define
the semantic metric
𝑑𝜙(𝑥,𝑥′)=∥𝜙(𝑥)−𝜙(𝑥′)∥2.
A randomized mechanism𝑀:X→Yis said to satisfy
(𝜖,𝛿)-Semantic Metric Differential Privacy if the following
conditions hold: For any adjacent texts𝑥∼𝑥′and any
measurable set𝑆⊆Y,
Pr[𝑀(𝑥)∈𝑆]≤𝑒𝜖𝑑𝜙(𝑥,𝑥′)Pr[𝑀(𝑥′)∈𝑆]+𝛿.
Based on this definition, we further propose the Semantic
Laplace Mechanism and the Semantic Exponential Mecha-
nism, which serve as the core privacy-preserving components
of our method.
IV. METHOD
This paper proposes a Prompt-Aware Dynamic Hierarchi-
cal Differential Privacy (PA-HDP) protection method. This
method decomposes the entire privacy protection pipeline into
three tightly coupled stages: the normalization and definition
of privacy units, prompt-aware risk assessment, hierarchical
DP protection.
Specifically, the first stage performs standardized data pre-
processing and defines the sentence as the minimum gran-
ularity unit for privacy protection. The second stage per-
forms fine-grained privacy risk assessment and hierarchical
classification on these retrieved segments by analyzing their
semantic correlation with user queries. In the final stage, the
framework first generates semantically equivalent candidate
texts by applying differentiated sensitive entity replacement
strategies. It then performs hierarchical differential privacy
protection by selecting the optimal sanitized text from the

JOURNAL OF L ATEX CLASS FILES, VOL. 14, NO. 8, AUGUST 2021 4
Algorithm 1PA-HDP
Input:Prompt𝑞, retrieval document𝐷= {𝑑1,...,𝑑𝑛},
retrieval number𝐾, size of candidate set𝑚, privacy budget
𝜖, and parameters𝜆,𝛼∈[0,1].
1://Retrieval and text preprocessing//
2:Encode document via embedding model to obtain dense
vectors{𝐸(𝑑 1),...,𝐸(𝑑 𝑛)}
3:Build vector index database for efficient similarity search
4:Encode user query to obtain𝐸(𝑞)
5:foreach document chunk𝑑 𝑖in vector indexdo
6:Compute cosine similarity for (𝐸(𝑑 𝑖),𝐸(𝑞))
7:end for
8:Select top-𝐾most relevant contexts {𝑠1,...,𝑠𝐾}
9:Divide text segment𝑠 𝑖,𝑗for𝑠𝑖.
10://prompt-aware risk assessment//
11:foreach text segment𝑠 𝑖,𝑗do
12:Encode text segment to obtain𝐸(𝑠 𝑖,𝑗)
13:Compute risk-aware semantic similarity𝑆 𝑟𝑒𝑙
14:Obtain sensitive field set𝐸 𝑖,𝑗={𝑒1
𝑖,𝑗,...,𝑒𝑚
𝑖,𝑗}
15:Compute field sensitivity𝑆 𝑠𝑒𝑛
16:Compute risk score𝑆 𝑡𝑜𝑡𝑎𝑙(𝑞,𝑠𝑖,𝑗)by Eq.1
17:Obtain noisy risk score
𝑆∗
𝑡𝑜𝑡𝑎𝑙(𝑞,𝑠𝑖,𝑗)=𝑆𝑡𝑜𝑡𝑎𝑙(𝑞,𝑠𝑖,𝑗)+𝐿𝑎𝑝(2
𝜖)
18:Divide privacy risk levels according to𝑆∗
𝑡𝑜𝑡𝑎𝑙(𝑞,𝑠𝑖,𝑗)
19:end for
20://Hierarchical DP protection//
21:foreach text segment𝑠 𝑖,𝑗do
22:if𝑆∗
𝑡𝑜𝑡𝑎𝑙(𝑞,𝑠𝑖,𝑗)≥𝜏 1then
23:Generate candidate texts𝐶=n
𝑠1
𝑖,𝑗,...,𝑠𝑚
𝑖,𝑗o
by
replacing entities
24:Use the exponential mechanism to privately output
the optimal𝑠∗
𝑖,𝑗∈𝐶with𝜖 𝐻/𝜖𝑀
25:end if
26:end for
27:Replace𝑠 𝑖,𝑗with𝑠∗
𝑖,𝑗to get
𝑠∗
1,...,𝑠∗
𝐾	
28:returnPrivate contexts
𝑠∗
1,...,𝑠∗
𝐾	
candidate set through the exponential mechanism, while rigor-
ously satisfying the formal constraints of differential privacy.
The complete pseudo-code is shown in Algorithm 1.
Furthermore, Fig. 1 provides an example illustrating the
pipeline of our privacy protection algorithm.
A. Retrieval and Text Preprocessing
In Retrieval-Augmented Generation (RAG), the retrieval
stage determines the external knowledge accessible to the
large language model (LLM) and serves as the primary entry
point for privacy leakage. Given an external knowledge base
𝐷={𝑑 1,...,𝑑𝑛}, each document is first chunked, embedded
using an encoder𝐸(·), and indexed as dense vectors. For a
user query𝑞, the system computes its embedding𝐸(𝑞)and
retrieves the top-𝐾text segments{𝑠 1,...,𝑠𝐾}based on cosine
similarity. The fixed top-𝐾strategy limits context length,reduces semantic redundancy, and confines privacy protection
to query-relevant content.
To improve processing quality, the retrieved text is further
preprocessed through sentence segmentation and text normal-
ization, including punctuation-based boundary detection and
the removal of redundant symbols, spaces, and line breaks.
These operations provide standardized inputs for subsequent
sensitive entity identification and differential privacy protec-
tion.
After preprocessing, each sentence𝑠 𝑖,𝑗is defined as aBasic
Privacy Unit (PU), where𝑠 𝑖,𝑗denotes the𝑗-th sentence in the
𝑖-th retrieved context. A privacy unit is the minimum process-
ing unit for sensitive entity recognition, sensitivity assessment,
candidate generation, and differential privacy perturbation.
Each PU preserves semantic completeness while maintaining
an independent privacy level and budget.
The independent design of privacy units enables fine-
grained privacy protection and naturally satisfies differential
privacy composition. It allows stronger protection for high-risk
content while preserving the utility of low-risk text, thereby
achieving a better privacy–utility trade-off.
B. Prompt-aware risk assessment
After completing retrieval and text preprocessing, to achieve
fine-grained differentiated privacy protection, it is necessary
to perform accurate localization, quantitative assessment, and
hierarchical classification of sensitive information in the re-
trieved segments. In RAG scenarios, only sensitive information
relevant to the current user query constitutes genuine privacy
leakage risks. Even if query-irrelevant sensitive content exists
in the retrieval results, it is difficult for Large Language
Models (LLMs) to attend to and output it, thus requiring no
high-intensity protection.
To realize prompt-aware dynamic risk assessment, this
paper proposes a two-dimensional risk quantification model.
It separately calculates the risk-aware semantic relevance𝑆 𝑟𝑒𝑙
between the retrieved segment and the user query𝑞, and
the field sensitivity𝑆 𝑠𝑒𝑛of the segment itself. The final
comprehensive risk score𝑆 𝑡𝑜𝑡𝑎𝑙 is obtained through weighted
fusion, based on which the text is divided into different privacy
risk levels.
1) Risk-aware semantic similarity:We design a risk-aware
gating mechanism that restricts semantic relevance computa-
tion to cases where the user query explicitly involves poten-
tially sensitive entities. Specifically, we first assess whether
the query contains potentially sensitive entities (e.g., person
names, locations, or organizations). This assessment deter-
mines whether the current query is likely to trigger privacy
leakage in the retrieved documents.
Let the user query be𝑞, and the j-th segment of the retrieved
context𝑠𝑖be𝑠𝑖,𝑗. We first apply an entity recognition module
E(·)to extract entities from the query:
E(𝑞)={𝑒 1,𝑒2,...,𝑒𝑚}.
We then define a sensitive-entity indicator function:

JOURNAL OF L ATEX CLASS FILES, VOL. 14, NO. 8, AUGUST 2021 5
Fig. 1. Pipeline of PA-HDP. a) Prompt-aware risk assessment stage. b) Hierarchical differential privacy protection stage. c) Full retrieval-augmented generation
pipeline.
𝐼(𝑞)=(
1,ifE(𝑞)∩E 𝑠𝑒𝑛𝑠≠∅
0,otherwise
whereE𝑠𝑒𝑛𝑠 denotes a predefined set of sensitive entity types.
When𝐼(𝑞)=1, we compute semantic similarity between
the query and each segment using the same embedding model
𝐸(·)employed in the retrieval stage:
v𝑞=𝐸(𝑞),v 𝑠𝑖, 𝑗=𝐸(𝑠𝑖,𝑗),
𝑆𝑟𝑒𝑙(𝑞,𝑠𝑖,𝑗)=v𝑞·v𝑠𝑖, 𝑗
∥v𝑞∥·∥v𝑠𝑖, 𝑗∥.
When𝐼(𝑞)=0, we conservatively assign𝑆 𝑟𝑒𝑙(𝑞,𝑠𝑖,𝑗)=0,
where·denotes the vector dot product and∥·∥denotes the
L2 norm. Thus,𝑆 𝑟𝑒𝑙(𝑞,𝑠𝑖,𝑗)∈[0,1].
Intuitively, when such entities are present, a higher semantic
similarity between the query and a retrieved segment indicates
a higher probability that the segment may be leveraged by
the generation model, thereby increasing the risk of privacy
leakage. Conversely, when no sensitive entities are detected
in the query, we assume negligible entity-driven leakage risk
from direct semantic matching and set the relevance score to
zero.
2) Field sensitivity:Field sensitivity measures the inherent
leakage harm degree of sensitive information contained in a
retrieved text segment. This paper adopts a hybrid strategy
combining Named Entity Recognition (NER) and rule-based
extraction to comprehensively extract sensitive fields from
retrieved text segments, forming a sensitive field setE 𝑖,𝑗=
{𝑒1
𝑖,𝑗,...,𝑒𝑚
𝑖,𝑗}, where𝑒𝑘
𝑖,𝑗denotes the k-th sensitive entity inthe𝑠𝑖,𝑗text segment. The NER module is implemented based
on a pre-trained semantic model, which can automatically
recognize and locate unstructured sensitive entities such as
personal names, locations, ages, and disease diagnoses in the
text. These entities usually have the ability to directly or
indirectly identify user identities and are the primary sources
of privacy leakage. On this basis, the rule-based extraction
module complements the recognition of highly structured
sensitive information such as phone numbers, email addresses,
ID card numbers, and passwords through regular expressions,
keyword libraries, and sensitive pattern matching. Such infor-
mation cannot be fully recognized solely by semantic features
but has fixed formats and combination patterns. This hybrid
strategy ensures that all content that may pose privacy risks is
included in the assessment scope, avoiding protection failure
caused by omissions. Subsequently, we assign predefined risk
weights𝑤 𝑘∈ [0,1]to different types of sensitive entities
according to their leakage harm degrees. For example, high
risk weights (𝑤 𝑘∈ [0.7,1.0]) are assigned to entities such
as ID card numbers, phone numbers, personal names, and
disease diagnoses, while low risk weights (𝑤 𝑘∈ [0.1,0.3])
are assigned to entities such as locations, dates, and common
nouns. The field sensitivity of the j-th segment is defined as
the maximum value of the risk weights of all sensitive entities
in the text segment:
𝑆𝑠𝑒𝑛(𝑠𝑖,𝑗)=max
1≤𝑘≤𝑚𝑤𝑘.
Using the maximum value instead of the average value
as the field sensitivity ensures that a text segment will be
classified as high risk as long as it contains one high-risk

JOURNAL OF L ATEX CLASS FILES, VOL. 14, NO. 8, AUGUST 2021 6
sensitive entity, thereby avoiding the underestimation of high-
risk content caused by the averaging effect.
3) Risk assessment:The semantic relevance and field sen-
sitivity are weighted and fused to obtain the comprehensive
risk score of the j-th segment:
𝑆𝑡𝑜𝑡𝑎𝑙(𝑞,𝑠𝑖,𝑗)=𝜆·𝑆𝑟𝑒𝑙(𝑞,𝑠𝑖,𝑗)+(1−𝜆)·𝑆 𝑠𝑒𝑛(𝑠𝑖,𝑗).(1)
Where𝜆∈[0,1]is the balance coefficient used to adjust the
relative importance of semantic relevance and field sensitivity
in risk assessment. In our experiments, the optimal value𝜆=
0.6is determined through grid search, indicating that semantic
relevance has a slightly greater impact on the final risk than
the inherent sensitivity of the field itself.
Furthermore, we introduce the semantic Laplace mechanism
to achieve rigorous privacy protection, and yield the noisy
risk scores𝑆∗
𝑡𝑜𝑡𝑎𝑙(𝑞,𝑠𝑖,𝑗). (The details are presented in the
appendix.)
Finally, text segments are divided into three privacy risk
levels according to their comprehensive risk scores:
RiskLevel(𝑠 𝑖,𝑗)= 
Low Risk, 𝑆∗
𝑡𝑜𝑡𝑎𝑙(𝑞,𝑠𝑖,𝑗)<𝜏 1
Medium Risk, 𝜏 1≤𝑆∗
𝑡𝑜𝑡𝑎𝑙(𝑞,𝑠𝑖,𝑗)<𝜏 2
High Risk, 𝑆∗
𝑡𝑜𝑡𝑎𝑙(𝑞,𝑠𝑖,𝑗)≥𝜏 2
where𝜏 1and𝜏 2denote the risk-level partition thresholds.
In our experiments, we set𝜏 1=0.3and𝜏 2=0.7based
on empirical observations. Text segments (sentences) with
different risk levels will adopt differential privacy protection
strategies with different budgets to achieve the differentiated
privacy protection objective of ”low risk, low protection; high
risk, high protection”.
C. Hierarchical DP protection
To achieve provable and fine-grained privacy protection, we
propose the semantic metric differential privacy as the refined
privacy definition, and adopt the exponential mechanism as
the core privacy algorithm. This section sequentially performs
candidate text generation, utility function design, semantic
exponential mechanism construction and hierarchical DP pro-
tection, and finally outputs secure and highly usable sanitized
text under the premise of strictly satisfying differential privacy
constraints.
1) Candidate text generation:For each retrieved text seg-
ment𝑠𝑖,𝑗, we construct a set of semantically equivalent and
structurally consistent candidate texts, providing a secure
candidate space for the exponential mechanism to select the
optimal output under differential privacy constraints. The
candidate text construction adopts a core strategy centered on
sensitive entity replacement, generating multiple semantically
equivalent valid candidate texts while strictly preserving the
overall semantic structure and non-sensitive content of the
original sentence.
Specifically, we pre-construct a large-scale sanitized entity
corpus𝑅as the secure material source for sensitive infor-
mation replacement. We employ the hybrid entity extraction
strategy proposed in Section 3.2.2 (combining Named EntityRecognition (NER) and rule-based extraction) to extract all en-
tities from the external knowledge base and perform category
annotation.
For each retrieved text segment, the system first detects
all sensitive entities, including their positions and categories.
Next, it randomly selects entities of the same category from the
sanitized entity corpus𝑅to perform isomorphic replacement.
Ultimately, the system produces𝑚semantically identical
candidate segments, denoted as𝐶=n
𝑠1
𝑖,𝑗,...,𝑠𝑚
𝑖,𝑗o
. We set
𝑚=50in our experiments.
2) Utility function design:To privately select the most ap-
propriate candidate text from a candidate set while preserving
semantic fidelity and reducing privacy leakage risk, we design
a composite utility function for the exponential mechanism.
The utility function jointly considers two objectives: semantic
preservation and privacy safety. Specifically, given a retrieved
segment𝑠 𝑖,𝑗∈Xand a candidate text𝑠𝑣
𝑖,𝑗∈𝐶, the utility
function is defined as:
𝑢(𝑠𝑖,𝑗,𝑠𝑣
𝑖,𝑗)=−𝛼·𝑑 sem(𝑠𝑖,𝑗,𝑠𝑣
𝑖,𝑗)−(1−𝛼)·𝑟(𝑠 𝑖,𝑗,𝑠𝑣
𝑖,𝑗)
where𝑑 sem(𝑠𝑖,𝑗,𝑠𝑣
𝑖,𝑗)denotes the semantic distance between
the original segment and the candidate text,𝑟(𝑠 𝑖,𝑗,𝑠𝑣
𝑖,𝑗)repre-
sents the privacy leakage risk of the candidate text,𝛼 >0is
a trade-off parameter balancing semantic quality and privacy
protection, and𝐷 >0is a normalization constant used to
stabilize the utility scale and control sensitivity.
To measure semantic preservation, both the original seg-
ment and the candidate text are encoded into dense semantic
embeddings using a pretrained text embedding model𝜙.
The semantic distance is then computed using the Euclidean
distance in the embedding space:
𝑑sem(𝑠𝑖,𝑗,𝑠𝑣
𝑖,𝑗)=∥𝜙(𝑠𝑖,𝑗)−𝜙(𝑠𝑣
𝑖,𝑗)∥2.
A smaller semantic distance indicates that the candidate text
preserves the semantic meaning of the original text more
effectively. Therefore, the utility function assigns higher utility
values to candidates with smaller embedding distances.
To evaluate privacy security, we first extract all sensitive en-
tities present in each candidate text and form a set, denoted as
𝐸(𝑠𝑣
𝑖,𝑗). Each element in𝐸(𝑠𝑣
𝑖,𝑗)denotes a potentially sensitive
entity or phrase, such as names, addresses, phone numbers,
or medical identifiers. If no sensitive span is detected, the
privacy risk𝑟(𝑠 𝑖,𝑗,𝑠𝑣
𝑖,𝑗)=0. Otherwise, each sensitive entity
is embedded into the same semantic space, and its semantic
similarity to the original segment is measured using cosine
similarity. The overall privacy leakage risk of the candidate
text is defined as the maximum similarity among all remaining
sensitive spans:
𝑟(𝑠𝑖,𝑗,𝑠𝑣
𝑖,𝑗)=max
𝑒∈𝐸(𝑠𝑣
𝑖, 𝑗)cos(𝑠𝑖,𝑗,𝑒).
This worst-case formulation emphasizes the most privacy-
sensitive residual entity, where even a single highly correlated
sensitive span may lead to successful information recovery. In
addition, the parameter𝛼controls the relative importance of
privacy protection versus semantic preservation: larger values
favor candidates with lower privacy leakage risk, while smaller
values place greater emphasis on semantic fidelity. In our

JOURNAL OF L ATEX CLASS FILES, VOL. 14, NO. 8, AUGUST 2021 7
experiments, we set𝛼=0.5, which was found to provide the
best trade-off between semantic fidelity and privacy protection.
It is worth noting that entity-level privacy risk assessment
is also performed on the retrieved progress. However, the
two stages serve different purposes: the former evaluates the
privacy leakage risk introduced by the prompt itself, whereas
the latter measures the residual privacy risk of candidate
texts with respect to the original segment. Additionally, to
properly bound the semantic distance in the embedding space,
all embedding vectors are normalized to unit norm, i.e.,
||𝜙(𝑠)|| 2≤1. This normalization constrains the embedding
space to a bounded domain, ensuring numerical stability and
facilitating the derivation of the utility sensitivity bound under
the semantic metric.
3) Semantic exponential mechanism construction:In this
section, we first propose the Semantic Metric Differential
Privacy, providing a theoretical basis for privacy protection
methods.
Based on the above utility function, the exponential mech-
anism selects a candidate text according to the following
probability distribution:
Lemma IV .1(Semantic Exponential Mechanism).LetCbe a
finite candidate text set, and let
𝑢(𝑠𝑖,𝑗,𝑠𝑣
𝑖,𝑗)=−𝛼·𝑑 sem(𝑠𝑖,𝑗,𝑠𝑣
𝑖,𝑗)−(1−𝛼)·𝑟(𝑠 𝑖,𝑗,𝑠𝑣
𝑖,𝑗),
where𝛼≥0. The Semantic Exponential Mechanism selects an
output𝑠∗
𝑖,𝑗∈Caccording to the probability distribution
Pr[𝑀(𝑠𝑖,𝑗)=𝑠∗
𝑖,𝑗]=exp𝜖𝑢(𝑠 𝑖, 𝑗,𝑠∗
𝑖, 𝑗)
2
Í
𝑠𝑣
𝑖, 𝑗∈Cexp𝜖𝑢(𝑠 𝑖, 𝑗,𝑠𝑣
𝑖, 𝑗)
2.
Then𝑀satisfies(𝜖,0)-Semantic Metric Differential Pri-
vacy.
The proof details are presented in the appendix.
4) Hierarchical DP protection:After determining the can-
didate text set and privacy mechanism, the reasonable alloca-
tion of the privacy budget is the key to ensuring the balance
between the protection effect of differential privacy and the
utility of text generation. To provide differentiated privacy
protection for texts with varying sensitivity levels, we first
classify all texts into three privacy risk categories: high-risk,
medium-risk, and low-risk. High-risk texts typically contain
highly sensitive information, while medium-risk texts contain
partially sensitive or identifiable content. Low-risk texts are
not considered in the subsequent privacy budget allocation.
After risk classification, differential privacy budgets are
allocated separately to high-risk and medium-risk texts. Let
the total privacy budget be𝜖, the number of high-risk texts be
𝑁ℎ, and the number of medium-risk texts be𝑁 𝑚. Since high-
risk texts require stronger privacy protection, they are assigned
smaller privacy budgets than medium-risk texts.
We adopt a risk-aware budget allocation strategy defined as:
𝜀𝐻=𝜀
2(𝑁𝐻+𝜆𝑁𝑀), 𝜀𝑀=𝛾𝜀𝐻where𝜀𝐻and𝜀𝑀denote the privacy budgets for high-risk
and medium-risk texts, respectively, and𝛾 >1controls the
protection gap between different risk levels. In our experi-
ments, we set𝛾=2, meaning that medium-risk texts receive
twice the privacy budget of high-risk texts. This strategy
enables stronger protection for highly sensitive texts while
preserving better semantic utility for medium-risk texts.
V. EXPERIMENT
This section verifies the effectiveness and privacy protection
performance of the proposed PA-HDP method through a series
of experiments. We first describe the experimental setup in
Section A. Subsequently, Sections B and C evaluate the utility
and privacy outcomes of the proposed method. Furthermore,
we conduct an ablation study in Section D to investigate how
different privacy budgets and numbers of retrieved documents
affect the performance of our method. Finally, we perform
a comprehensive hyperparameter comparison in Section F to
identify the optimal hyperparameter configuration.
A. Experimental details
In this section, we introduce the evaluation datasets, baseline
algorithms, and experimental setup. The overall experimental
evaluation followed the experimental setup in work [7].
1) Datasets:We evaluate the effectiveness of our proposed
methods under two privacy-relevant experimental scenarios.
The first scenario targets real-world medical dialogue mon-
itoring, where we adopt the HealthcareMagic-101 dataset
(comprising 200,000 doctor-patient medical dialogues) as the
retrieval corpus.
In the second scenario, following Huang et al. [33], we
consider a setting where private information is mixed with
public data. Specifically, we construct the Wiki-PII dataset by
embedding personally identifiable information (PII) fragments
extracted from the private Enron Mail dataset into each sample
of the public WikiText-103 dataset. We then benchmark our
methods on four standard open-domain question answering
(ODQA) benchmarks: Natural Questions (NQ) [34], TriviaQA
(TQA) [35], Web Questions (WQ) [36], and CuratedTrec (CT)
[37]. Full details on dataset construction and characteristics are
provided in Appendix A.9.
2) Baselines:To evaluate the effectiveness of our method,
we compare against four baselines. The first isParaphras-
ing[7], which uses an LLM to extract the most relevant
parts of the original context without performing any semantic
rewriting. The second isZeroGen[38], a zero-shot learning
framework that first generates a synthetic dataset from scratch
using a large pre-trained language model guided by task-
specific prompts, then trains a tiny task model (e.g., LSTM)
on the synthesized data for efficient inference. The third is
AttrPrompt[39], a data synthesis approach that generates
training data using diversely attributed prompts (specifying
attributes such as length, style, or topic), which yields more
diverse and less biased synthetic datasets compared to simple
class-conditional prompts. The fourth isSAGE[7], a two-
stage synthetic data generation paradigm that mitigates the
privacy issues by pure synthetic data. In this method,Stage-1

JOURNAL OF L ATEX CLASS FILES, VOL. 14, NO. 8, AUGUST 2021 8
refers to the outputs generated from attribute-based generation,
whileStage-2represents the outputs of the complete SAGE
pipeline. The fifth isLPRAG[18], a privacy-preserving
RAG framework with formal privacy guarantees based on
local differential privacy (LDP). Detailed descriptions of these
baselines are provided in Appendix A.3.
In addition, we report several variants of our framework to
analyze the contribution of each component.0-shotdenotes
generation without retrieval augmentation.Originuses the
original dataset directly as the retrieval corpus.
3) Experimental Setup:For both utility and privacy evalua-
tions, we employ two representative generation backbones: the
closed-sourceGPT-3.5-Turboand the open-sourceLlama3-8B-
Chat (L8C). Using both proprietary and open-source models
allows us to assess the generality of our approach across
different model families and deployment settings. We include
L8C because its performance on the target tasks relies heavily
on retrieval augmentation, making it suitable for evaluating
the utility gains provided by RAG. In addition, both models
have undergone safety alignment, enabling us to investigate
privacy risks in standard RAG pipelines and evaluate the
effectiveness of our proposed protection mechanism under
realistic deployment conditions.
We adoptbge-large-en-v1.5as the embedding model. Un-
less otherwise specified, similarity is computed using the
Cosine similarity between normalized embeddings, GPT-3.5-
Turbo is adopted as the generation model, only the top
retrieved document (𝑘=1) and a privacy budget of𝜖=5
are provided for each query.
B. Utility evaluation
To evaluate the utility of synthetic data as retrieval corpora,
we assess the quality of generated answers by comparing them
against the corresponding ground-truth answers. We report two
widely used text generation metrics,ROUGE-LandBLEU,
as the primary evaluation measures.
ROUGE-L evaluates the longest common subsequence be-
tween the generated answer and the reference answer, captur-
ing their similarity in terms of content coverage and sentence-
level structure. BLEU measures the overlap of n-grams be-
tween the generated and reference answers, reflecting the
lexical accuracy and fluency of the generated text. Higher
ROUGE-L and BLEU scores indicate better agreement with
the ground truth and, consequently, higher utility of the
synthetic retrieval data.
1) Results on Medical Dialog:In this section, we split the
data into two parts:99%of the data is used as the retrieval
data, and the remaining1%is used as the test data. As shown
in Table I, our method consistently achieves the best perfor-
mance across both generation backbones. Compared with all
baselines, it substantially improves both BLEU and ROUGE-
L scores. In particular, under GPT-3.5, our method improves
BLEU from 0.1594 (Stage-2) to 0.3275 and ROUGE-L from
0.1288 to 0.3445. Similar gains are observed for Llama3-8B-
Chat. These results demonstrate that our approach is able to
preserve the utility of retrieval data while maintaining high
answer quality.TABLE I
UTILITY RESULTS ONHEALTHCAREMAGIC DATASET
Method GPT-3.5 Llama3-8b-Chat
BLEU ROUGE-L BLEU ROUGE-L
0-shot 0.1143 0.1045 0.081 0.0765
Origin 0.1193 0.1078 0.0846 0.0789
Paraphrase 0.1481 0.1303 0.105 0.0952
ZeroGen 0.1199 0.1050 0.085 0.0769
LPRAG 0.3186 0.33100.16340.1596
AttrPrompt 0.1114 0.0915 0.079 0.0670
Stage-1 0.1578 0.1306 0.114 0.0956
Stage-2 0.1594 0.1288 0.1192 0.1160
Ours0.3275 0.34450.15120.1726
The superior performance can be attributed to the hierarchi-
cal protection strategy adopted in our framework. Instead of
rewriting or regenerating entire documents, our method first
identifies sensitive content and applies protection only where
necessary. Most non-sensitive information is preserved in its
original form, retaining the semantic integrity and knowledge
content of the source documents. By minimizing unnecessary
modifications and restricting transformations to a small subset
of sensitive spans, the protected retrieval corpus remains
highly faithful to the original data, leading to significantly bet-
ter performance than methods that rely on extensive rewriting
or synthetic generation.
2) Results on ODQA:Table II presents the ODQA results
on the Wiki-PII dataset. Overall, our method achieves the
best or near-best performance across the four benchmark
datasets (NQ, TQA, WQ, and CT) under both GPT-3.5-Turbo
and Llama3-8B-Chat. Compared with existing synthetic data
generation approaches, including ZeroGen, Stage-2 and Attr-
Prompt, the proposed method consistently yields higher BLEU
and ROUGE-L scores. Notably, the performance advantage
is observed for both closed-source and open-source models,
indicating that the effectiveness of our approach is not tied to
a specific generation backbone. These results demonstrate that
the proposed framework can preserve the utility of retrieval
data while providing privacy protection, enabling downstream
QA systems to maintain high answer quality.
These results highlight the advantage of our fine-grained,
risk-aware protection strategy. Existing synthetic generation
methods often modify or regenerate large portions of the
retrieval corpus, which can introduce semantic distortions
and weaken the connection between retrieved documents and
downstream questions. In contrast, our method selectively pro-
tects only the sensitive content while preserving the majority
of the original document structure, factual knowledge, and
contextual information. By minimizing unnecessary perturba-
tions to non-sensitive content, the resulting retrieval corpus
remains highly faithful to the original data distribution and
therefore retains strong retrieval utility. Overall, the results
demonstrate that effective privacy protection and high retrieval
utility are not necessarily conflicting objectives, and that
carefully controlled modifications can preserve downstream
ODQA performance while substantially reducing privacy risks.

JOURNAL OF L ATEX CLASS FILES, VOL. 14, NO. 8, AUGUST 2021 9
TABLE II
UTILITY RESULTS ONWIKI-PIIDATASET
GPT-3.5 NQ TQA WQ CT
Method BLEU↑ROUGE-L↑BLEU↑ROUGE-L↑BLEU↑ROUGE-L↑BLEU↑ROUGE-L↑
0-shot 0.0239 0.0528 0.0371 0.0776 0.0252 0.0622 0.0368 0.0680
Origin 0.0598 0.1223 0.0660 0.1345 0.0517 0.11790.0742 0.1464
Paraphrase 0.0508 0.1044 0.0558 0.1241 0.0331 0.0813 0.0563 0.1142
ZeroGen 0.0113 0.0245 0.0251 0.0494 0.0366 0.0874 0.0484 0.0929
AttrPrompt 0.0203 0.0415 0.0264 0.0534 0.0211 0.0478 0.0260 0.0503
Stage-1 0.0435 0.0998 0.0550 0.1231 0.0465 0.1205 0.0509 0.1097
Stage-2 0.0588 0.1250 0.0576 0.1221 0.0609 0.1296 0.0538 0.1210
Ours0.0674 0.1437 0.0688 0.1476 0.0666 0.14430.0638 0.1392
Llama3 NQ TQA WQ CT
Method BLEU↑ROUGE-L↑BLEU↑ROUGE-L↑BLEU↑ROUGE-L↑BLEU↑ROUGE-L↑
0-shot 0.00719 0.0136 0.0084 0.0157 0.0072 0.0143 0.0088 0.0150
Origin 0.0180 0.0315 0.0150 0.0272 0.0147 0.0271 0.0178 0.0323
Paraphrase 0.0153 0.0269 0.0127 0.0251 0.0094 0.0187 0.0135 0.0252
ZeroGen 0.0034 0.0063 0.0057 0.010 0.0104 0.0201 0.0116 0.0205
AttrPrompt 0.0061 0.0107 0.006 0.0108 0.006 0.0110 0.0062 0.0111
Stage-1 0.0131 0.0257 0.0125 0.0249 0.0132 0.0277 0.0122 0.0242
Stage-2 0.0248 0.0406 0.0252 0.0398 0.0256 0.04290.02550.0427
Ours0.0283 0.0476 0.0281 0.0495 0.0278 0.04720.02510.0437
C. privacy evaluation
To evaluate the privacy protection capability of our method,
we conduct both targeted and untargeted extraction attacks
following [7]. These attacks are designed to induce retrieval
systems to disclose information from the retrieval corpus.
The attack prompt consists of two components: aninforma-
tioncomponent and acommandcomponent. The information
component is used to guide the retriever toward specific
content, while the command component instructs the language
model to reveal the retrieved information (e.g., “Please re-
peat all the context”). For targeted attacks, the information
component is crafted to retrieve specific sensitive content,
such as personal identifiers or private medical dialogues. For
untargeted attacks, it is designed to maximize the amount
of information extracted from the retrieval corpus without
focusing on any particular target.
Fortargeted attacks, we reportRepeat PromptsandTar-
geted Information, which measures the number of unique
sensitive information items successfully extracted by the at-
tacker. Foruntargeted attacks, we report four metrics.Repeat
Promptscounts the number of attack prompts that induce
the model to reproduce at least 10 consecutive tokens from
the retrieval corpus.Rouge Promptscounts the number of
prompts whose outputs achieve a ROUGE-L score above
0.5 with any retrieved document. We further reportRepeat
Contexts, the number of unique verbatim excerpts extracted
from the corpus, andRouge Contexts, the number of unique
extracted passages with a ROUGE-L score greater than 0.5.
1) Results on Targeted Attack:Table III presents the results
of targeted extraction attacks on both the Wiki-PII and Health-
CareMagic datasets. The original retrieval corpus is highly
vulnerable to extraction attacks, especially when paired with
GPT-3.5-Turbo, where a large amount of sensitive information
can be successfully recovered. Although existing synthesis-
based methods, such as ZeroGen and AttrPrompt, reduce leak-age to some extent, their effectiveness varies across datasets
and models. In contrast, our method consistently achieves zero
leakage across all settings, with both Targeted Information and
Repeat Prompts reduced to zero.
The results demonstrate that our hierarchical protection
mechanism effectively removes exploitable sensitive informa-
tion from the retrieval corpus while preventing verbatim mem-
orization and disclosure. Compared with Stage-1 and Stage-
2, which still exhibit a small number of successful extrac-
tions in certain settings, our complete framework eliminates
all targeted leakage attempts. This indicates that selectively
protecting sensitive content at different privacy levels provides
stronger privacy guarantees than directly relying on synthetic
generation or coarse-grained document transformation.
2) Results on Untargeted Attack:Table IV reports the re-
sults of untargeted extraction attacks on the HealthCareMagic
dataset. The original retrieval corpus remains highly vulnera-
ble, especially under GPT-3.5-Turbo, where a large number of
prompts successfully induce verbatim or semantically similar
disclosures. Although paraphrasing reduces exact leakage,
substantial privacy risks still remain. In contrast, our method
achieves zero leakage across all prompt-level and context-
level metrics under both generation backbones. Compared
with Stage-1, which still allows a small number of successful
extractions, our complete framework effectively prevents both
exact memorization and semantic reconstruction of sensitive
content, demonstrating strong robustness against untargeted
extraction attacks.
D. Ablation Study
In this section, we conduct two ablation studies to further
analyze the effectiveness of our method. Specifically, we
investigate the impact of different privacy budgets and the
effect of varying the number of retrieved documents. These
experiments aim to provide a more fine-grained understanding

JOURNAL OF L ATEX CLASS FILES, VOL. 14, NO. 8, AUGUST 2021 10
TABLE III
TARGETED ATTACK RESULTS ONWIKI-PIIANDHEALTHCAREMAGIC DATASET(250PROMPTS)
Target-wiki-llama-3-8b Target-wiki-gpt-3.5 Target-chat-llama-3-8b Target-chat-gpt-3.5
Method Target info↓Repeat prompts↓ Target info↓Repeat prompts↓ Target info↓Repeat prompts↓ Target info↓Repeat prompts↓
origin 25 12 167 64 7 23 75 132
para 9 1 28 9 17 26 42 81
ZeroGen 4 5 5 2 0 3 1 6
AttrPrompt 0 0 0 0 0 0 0 0
Stage-1 1 4 3 19 3 11 12 36
Stage-2 0 0 0 7 0 0 0 0
Ours 0 0 0 0 0 0 0 0
TABLE IV
UNTARGETED ATTACK RESULTS ONHEALTHCAREMAGIC DATASET(250PROMPTS)
Untarget-chat-llama Untarget-chat-gpt3.5
Method Repeat prompt↓ROUGE prompt↓ Repeat context↓ROUGE context↓ Repeat prompt↓ROUGE prompt↓ Repeat context↓ROUGE context↓
origin 19 17 16 13 61 67 49 67
para 23 13 22 11 45 63 33 50
ZeroGen 0 0 0 0 0 0 0 0
AttrPrompt 0 0 0 0 0 0 0 0
Stage-1 1 2 1 2 1 0 1 0
Stage-2 0 0 0 0 0 0 0 0
Ours 0 0 0 0 0 0 0 0
Fig. 2. Utility Results under Different Privacy Budgets
of how key design choices affect both privacy protection and
utility performance.
1) Impact of Privacy Budgets:Fig.2 illustrates the impact
of the privacy budget𝜖on utility across four ODQA bench-
marks (NQ, TQA, WQ, and CT). Overall, both BLEU-1 and
ROUGE-L scores consistently increase as𝜖grows, indicating
a clear trade-off between privacy strength and answear quality.
When𝜖is small (stronger privacy), performance is relatively
lower due to more aggressive perturbation. As𝜖increases, the
model is allowed to retain more accurate information, leading
to steady improvements in both metrics. The trend is consistent
across all datasets, demonstrating that higher privacy budgets
enable better preservation of retrieval utility while maintaining
stable gains across benchmarks.
Table V reports targeted attack results under different pri-
vacy budgets𝜖on both the HealthCareMagic and Wiki-PII
datasets using GPT-3.5 as the generation backbone.Table V
reports targeted attack results under different privacy budgets
𝜖on both the HealthCareMagic and Wiki-PII datasets using
GPT-3.5 as the generation backbone. Across all settings, we
observe zero successful extraction, including both Targeted
Information and Repeat Prompts, under all privacy budgets
from𝜖=0.1to𝜖=10. This indicates that the proposed
method is robust to varying levels of privacy strength, andTABLE V
TARGETEDATTACKRESULTS UNDER DIFFERENT PRIVACY BUDGETS
Chat Wiki
𝜖 Target info↓Repeat prompts↓ Target info↓Repeat prompts↓
0.1 0 0 0 0
1 0 0 0 0
5 0 0 0 0
10 0 0 0 0
even under weaker privacy constraints (larger𝜖), no sensitive
information can be recovered via targeted attacks. These
results demonstrate that the hierarchical protection mechanism
effectively eliminates direct leakage of sensitive content while
maintaining consistent behavior across different privacy bud-
gets.
2) Impact of the retrieved number of documents:Table VI
reports the impact of varying the number of retrieved docu-
ments𝑘on utility across HealthCareMagic and four ODQA
benchmarks. Overall, the results remain stable when increasing
𝑘from 1 to 5, with only minor fluctuations in BLEU and
ROUGE scores. On HealthCareMagic, performance is consis-
tently high and even slightly improves at𝑘=3, indicating that

JOURNAL OF L ATEX CLASS FILES, VOL. 14, NO. 8, AUGUST 2021 11
TABLE VI
UTILITYRESULTS ONNUMBER OFRETRIEVEDDOCS
HealthCareMagic NQ TQA WQ CT
𝑘BLEU ROUGE BLEU ROUGE-L BLEU ROUGE-L BLEU ROUGE-L BLEU ROUGE-L
10.3275 0.3445 0.0674 0.1437 0.0688 0.1476 0.0666 0.1443 0.0638 0.1392
30.3300 0.3456 0.0666 0.1432 0.0683 0.1453 0.0693 0.1455 0.0648 0.1388
50.3246 0.3430 0.0667 0.1419 0.0670 0.1427 0.0696 0.1459 0.0623 0.1371
Fig. 3. Utility Results under Different Generation Models
a moderate increase in retrieved context can provide additional
useful information. For ODQA datasets (NQ, TQA, WQ,
and CT), we observe similarly small variations, suggesting
that increasing the number of retrieved documents does not
introduce noticeable degradation in generation quality.
These results demonstrate that our pipeline maintains robust
utility across different retrieval settings. In particular, expand-
ing the retrieval scope does not harm performance, indicating
that the generated data is able to support multi-document
retrieval without overfitting to a specific retrieval size. This
further confirms the stability and generalization capability of
our method under varying retrieval configurations.
3) Impact of Model Choice:We evaluate the utility perfor-
mance of Mistral-7B-Instruct-v0.2(M7I-v0.2), L8C and GPT-
3.5 on ODQA and HealthCareMagic benchmarks, where
BLEU-1 and ROUGE-L are adopted to quantify the similarity
between generated content and ground-truth references. No-
tably, the ODQA results plotted in the figure correspond to the
averaged values across its four sub-datasets. On both datasets,
M7I-v0.2 and L8C obtain highly comparable but relatively low
metric values. GPT-3.5 achieves obvious performance advan-
tages over the two open-source models, and the performance
gap is more prominent on the challenging ODQA task than
the HealthCareMagic task.
The close results of M7I-v0.2 and L8C reveal that
lightweight open-source LLMs have similar constrained ca-
pacity for our stage-1 data generation, which produces low-
fidelity synthetic text. In contrast, GPT-3.5 provides steady
and superior utility across two different task scenarios. This
outcome confirms GPT-3.5 acts as a reliable backbone for our
two-stage privacy-preserving generation framework.
E. Hyperparameter Choice
Furthermore, we investigate the impact of different hy-
perparameter combinations on the utility of the proposed
method. Specifically,𝛾balances the relative importance ofsemantic relevance and field sensitivity in risk assessment,
while𝛼controls the trade-off between privacy protection and
semantic preservation during candidate selection. As shown in
Table VII, the proposed method exhibits stable performance
across different parameter settings, indicating that it is not
overly sensitive to hyperparameter choices. Among all config-
urations,𝛾=0.6and𝛼=0.5consistently achieve the best
overall results. This suggests that assigning slightly greater
importance to semantic relevance than field sensitivity leads
to more accurate risk assessment, while a balanced weighting
between privacy protection and semantic preservation provides
the most effective trade-off between privacy and utility. There-
fore, we adopt𝛾=0.6and𝛼=0.5as the default setting in
all experiments.
F . Efficiency Analysis
To evaluate the computational efficiency of the proposed
method, particularly the time overhead introduced by the
privacy protection procedure, we compare its runtime with five
representative privacy-preserving approaches. The experiments
are conducted with the number of retrieved documents set
to K=1, which eliminates the impact of retrieval-scale vari-
ations and enables a fair comparison of the privacy protection
overhead. As shown in Table VIII, all methods have the
same retrieval time, and the runtime differences mainly come
from the privacy protection process. Our method achieves a
relatively low privacy protection overhead and is only slower
than PARA. Although PARA has lower runtime, it relies
on online LLM API calls for privacy processing, while our
method can operate offline without LLM invocation, making
it more suitable for practical deployment.
VI. CONCLUSION
In this paper, we revisit privacy protection in retrieval-
augmented generation from a new perspective and identify a
fundamental limitation of existing approaches: the document-
level static privacy risk assumption. We show that privacy
leakage in RAG is inherently query-dependent, and the same
document may pose substantially different privacy risks under
different user prompts. Motivated by this observation, we pro-
pose PA-HDP, a Prompt-Aware Dynamic Hierarchical Differ-
ential Privacy protection framework that dynamically assesses
privacy risks and applies differentiated protection through
risk stratification, adaptive sensitive entity replacement, and
exponential mechanism-based text selection.

JOURNAL OF L ATEX CLASS FILES, VOL. 14, NO. 8, AUGUST 2021 12
TABLE VII
UTILITYRESULTS ONCHOICE OFHYPERPARAMETER
HealthCareMagic NQ TQA WQ CT
𝛾, 𝛼 BLEU ROUGE BLEU ROUGE-L BLEU ROUGE-L BLEU ROUGE-L BLEU ROUGE-L
0.6,0.3 0.3182 0.3351 0.0605 0.1344 0.0619 0.1382 0.0597 0.1350 0.0571 0.1299
0.6,0.7 0.3153 0.3337 0.0598 0.1326 0.0601 0.1333 0.0627 0.1366 0.0556 0.1278
0.4,0.5 0.31900.3420 0.0612 0.1351 0.0624 0.1390 0.0603 0.1355 0.0576 0.1305
0.5,0.5 0.3211 0.3383 0.0650 0.1411 0.06910.1408 0.0622 0.1374 0.06240.1401
0.6,0.5 0.32750.3445 0.0674 0.1437 0.06880.1476 0.0666 0.1443 0.06380.1392
TABLE VIII
RUNTIMECOMPARISON(SECONDS)
Retrieval Protection Total
para 6.29 1628.37 1634.66
ZeroGen 6.29 8792.48 8798.77
AttrPrompt 6.29 3681.30 3687.59
Stage-2 6.29 3259.11 3265.40
LPRAG 6.29 3977.83 3984.12
Ours 6.29 2712.26 2718.55
Extensive experiments on medical dialogue and open-
domain question answering benchmarks demonstrate that PA-
HDP effectively mitigates both targeted and untargeted privacy
attacks while maintaining high retrieval utility. Moreover, un-
der rigorous differential privacy guarantees, PA-HDP achieves
a favorable trade-off between privacy protection and model
performance. We hope this work highlights the importance of
query-aware privacy protection and provides a foundation for
the secure deployment of RAG systems on sensitive data.
1) limitation:PA-HDP assumes a fixed privacy budget
that is predetermined before deployment. However, in real-
world RAG systems, the number of user queries is typically
unknown in advance, and privacy budgets may be gradually
exhausted as queries accumulate. Future work will investigate
adaptive privacy budget allocation and accounting mechanisms
to support long-term deployments while maintaining rigorous
privacy guarantees.
REFERENCES
[1] Y . Gao, Y . Xiong, X. Gao, K. Jia, J. Pan, Y . Bi, Y . Dai, J. Sun,
M. Wang, and H. Wang, “Retrieval-augmented generation for large
language models: A survey,” arXiv preprint arXiv:2312.10997, 2023.
[2] L. Huang, W. Yu, W. Ma, W. Zhong, Z. Feng, H. Wang, Q. Chen,
W. Peng, X. Feng, B. Qin etal., “A survey on hallucination in large
language models: Principles, taxonomy, challenges, and open questions,”
ACM Transactions onInformation Systems, vol. 43, no. 2, pp. 1–55,
2025.
[3] L. M. Amugongo, P. Mascheroni, S. Brooks, S. Doering, and J. Seidel,
“Retrieval augmented generation for large language models in health-
care: A systematic review,” PLOS Digital Health, vol. 4, no. 6, p.
e0000877, 2025.
[4] Y . Zhao, P. Singh, H. Bhathena, B. Ramos, A. Joshi, S. Gadiyaram,
and S. Sharma, “Optimizing llm based retrieval augmented genera-
tion pipelines in the financial domain,” in Proceedings ofthe2024
Conference oftheNorth American Chapter oftheAssociation for
Computational Linguistics: Human Language Technologies (V olume 6:
Industry Track), 2024, pp. 279–294.
[5] M. Hindi, L. Mohammed, O. Maaz, and A. Alwarafy, “Enhancing the
precision and interpretability of retrieval-augmented generation (rag) in
legal technology: A survey,” IEEE Access, 2025.[6] S. Zeng, J. Zhang, P. He, Y . Liu, Y . Xing, H. Xu, J. Ren, Y . Chang,
S. Wang, D. Yin etal., “The good and the bad: Exploring privacy issues
in retrieval-augmented generation (rag),” in Findings oftheAssociation
forComputational Linguistics: ACL 2024, 2024, pp. 4505–4524.
[7] S. Zeng, J. Zhang, P. He, J. Ren, T. Zheng, H. Lu, H. Xu, H. Liu,
Y . Xing, and J. Tang, “Mitigating the privacy issues in retrieval-
augmented generation (rag) via pure synthetic data,” in Proceedings
ofthe2025 Conference onEmpirical Methods inNatural Language
Processing, 2025, pp. 24 538–24 569.
[8] S. Guan, H. C. Kwok, N. F. Law, G. Stiglic, H. Qin, and V . Hui,
“Privacy challenges and solutions in retrieval-augmented generation-
enhanced llms for healthcare chatbots: A review of applications, risks,
and future directions,” arXiv preprint arXiv:2511.11347, 2025.
[9] Y . Mu, H. Hu, F. Li, Q. Yuan, J. Wu, Z. Liu, P. Liu, M. Wang,
H. Zhou, and Y . Liu, “Towards secure retrieval-augmented generation:
A comprehensive review of threats, defenses and benchmarks,” arXiv
preprint arXiv:2603.21654, 2026.
[10] N. Grislain, “Rag with differential privacy,” in 2025 IEEE Conference
onArtificial Intelligence (CAI). IEEE, 2025, pp. 847–852.
[11] C. Dwork, “Differential privacy,” in Encyclopedia ofCryptography,
Security andPrivacy. Springer, 2025, pp. 649–652.
[12] T. Koga, R. Wu, Z. Zhang, and K. Chaudhuri, “Privacy-preserving
retrieval-augmented generation with differential privacy,” 2025. [Online].
Available: https://arxiv.org/abs/2412.04697
[13] A. Hemmat, M. Moqadas, A. Mamanpoosh, A. Rismanchian, and
A. Fatemi, “Vague-gate: Plug-and-play local-privacy shield for retrieval-
augmented generation,” in Proceedings ofthe14th International Joint
Conference onNatural Language Processing and the4thConference
ofthe Asia-Pacific Chapter ofthe Association forComputational
Linguistics, 2025, pp. 3715–3730.
[14] J. Mori, K. Kakizaki, T. Miyagawa, and J. Sakuma, “Differentially
private synthetic text generation for retrieval-augmented generation
(rag),” arXiv preprint arXiv:2510.06719, 2025.
[15] V . Vinod, K. Pillutla, and A. Guha Thakurta, “Invisibleink: High-utility
and low-cost text generation with differential privacy,” Advances in
Neural Information Processing Systems, vol. 38, pp. 67 254–67 298,
2026.
[16] H. Wang, X. Xu, B. Huang, and K. Shu, “Privacy-aware decoding: Mit-
igating privacy leakage of large language models in retrieval-augmented
generation,” arXiv preprint arXiv:2508.03098, 2025.
[17] S. Joo, H. Koh, and K. Jung, “Pearl: Differentially
private and entropy-aware regulated language generation,”
openreview.net/pdf?id=qIUR54yyro, 2026.
[18] L. He, P. Tang, Y . Zhang, P. Zhou, and S. Su, “Mitigating privacy risks
in retrieval-augmented generation via locally private entity perturbation,”
Information Processing &Management, vol. 62, no. 4, p. 104150, 2025.
[19] T. Tang, J. Flemings, Y . Wang, and M. Annavaram, “Differentially pri-
vate retrieval-augmented generation,” arXiv preprint arXiv:2602.14374,
2026.
[20] J. Yu, J. Zhou, Y . Ding, L. Zhang, Y . Guo, and H. Sato, “Textual differ-
ential privacy for context-aware reasoning with large language model,”
in2024 IEEE 48th Annual Computers, Software, and Applications
Conference (COMPSAC). IEEE, 2024, pp. 988–997.
[21] Y . Cheng, L. Zhang, J. Wang, M. Yuan, and Y . Yao, “Remoterag: A
privacy-preserving llm cloud rag service,” in Findings oftheAssociation
forComputational Linguistics: ACL 2025, 2025, pp. 3820–3837.
[22] H. Ye, J. Guo, Z. Liu, and K.-Y . Lam, “Efficient privacy-preserving
retrieval augmented generation with distance-preserving encryption,” in
2025 3rdInternational Conference onFoundation andLarge Language
Models (FLLM). IEEE, 2025, pp. 668–676.

JOURNAL OF L ATEX CLASS FILES, VOL. 14, NO. 8, AUGUST 2021 13
[23] J. Chen, H. Zhang, L. Pang, Y . Tong, H. Zhou, Y . Zhan,
W. Lin, and Z. Zheng, “Privacy-preserving reasoning with knowledge-
distilled parametric retrieval augmented generation,” arXiv preprint
arXiv:2509.01088, 2025.
[24] J. He, C. Liu, G. Hou, W. Jiang, and J. Li, “Press: Defending pri-
vacy in retrieval-augmented generation via embedding space shifting,”
inICASSP 2025-2025 IEEE International Conference onAcoustics,
Speech andSignal Processing (ICASSP). IEEE, 2025, pp. 1–5.
[25] Y . Wang, H. Zhang, L. Pang, Y . Tong, B. Guo, H. Zheng,
and Z. Zheng, “Learning to erase private knowledge from multi-
documents for retrieval-augmented large language models,” arXiv
preprint arXiv:2504.09910, 2025.
[26] Z. Charles, A. Ganesh, R. McKenna, H. B. McMahan, N. Mitchell,
K. Pillutla, and K. Rush, “Learning with user-level differential privacy
under fixed compute budgets,” in 2025 IEEE Conference onSecure and
Trustworthy Machine Learning (SaTML). IEEE, 2025, pp. 901–920.
[27] F. Boenisch, C. M ¨uhl, A. Dziedzic, R. Rinberg, and N. Papernot, “Have
it your way: Individualized privacy assignment for dp-sgd,” Advances
inNeural Information Processing Systems, vol. 36, pp. 19 073–19 103,
2023.
[28] R. Wu, E. Wang, Z. Zhang, and Y .-X. Wang, “Private-rag: Answering
multiple queries with llms while keeping your data private,” arXiv
preprint arXiv:2511.07637, 2025.
[29] R. Wu, E. Wang, and Y .-X. Wang, “Beyond per-question privacy: Multi-
query differential privacy for rag systems,” in NeurIPS 2025 Workshop:
Reliable MLfrom Unreliable Data, 2025.
[30] C. Dwork, “Differential privacy,” in International colloquium on
automata, languages, andprogramming. Springer, 2006, pp. 1–12.
[31] C. Dwork, F. McSherry, K. Nissim, and A. Smith, “Calibrating noise
to sensitivity in private data analysis,” in Theory ofCryptography
Conference. Springer, 2006, pp. 265–284.
[32] F. McSherry and K. Talwar, “Mechanism design via differential privacy,”
in48th Annual IEEE Symposium onFoundations ofComputer Science
(FOCS’07). IEEE, 2007, pp. 94–103.
[33] Y . Huang, S. Gupta, Z. Zhong, K. Li, and D. Chen, “Privacy implica-
tions of retrieval-based language models,” in Proceedings ofthe2023
Conference onEmpirical Methods inNatural Language Processing,
2023, pp. 14 887–14 902.
[34] T. Kwiatkowski, J. Palomaki, O. Redfield, M. Collins, A. Parikh,
C. Alberti, D. Epstein, I. Polosukhin, J. Devlin, K. Lee etal., “Natural
questions: a benchmark for question answering research,” Transactions
oftheAssociation forComputational Linguistics, vol. 7, pp. 453–466,
2019.
[35] M. Joshi, E. Choi, D. S. Weld, and L. Zettlemoyer, “Triviaqa: A large
scale distantly supervised challenge dataset for reading comprehension,”
inProceedings ofthe55th Annual Meeting oftheAssociation for
Computational Linguistics (V olume 1:Long Papers), 2017, pp. 1601–
1611.
[36] J. Berant, A. Chou, R. Frostig, and P. Liang, “Semantic parsing on
freebase from question-answer pairs,” in Proceedings ofthe2013
conference onempirical methods innatural language processing, 2013,
pp. 1533–1544.
[37] P. Baudi ˇs and J. ˇSediv `y, “Modeling of the question answering task in
the yodaqa system,” in International Conference ofthecross-language
evaluation Forum forEuropean languages. Springer, 2015, pp. 222–
228.
[38] J. Ye, J. Gao, Q. Li, H. Xu, J. Feng, Z. Wu, T. Yu, and L. Kong, “Zero-
gen: Efficient zero-shot learning via dataset generation,” in Proceedings
ofthe2022 Conference onEmpirical Methods inNatural Language
Processing, 2022, pp. 11 653–11 669.
[39] Y . Yu, Y . Zhuang, J. Zhang, Y . Meng, A. J. Ratner, R. Krishna,
J. Shen, and C. Zhang, “Large language model as attributed training data
generator: A tale of diversity and bias,” Advances inneural information
processing systems, vol. 36, pp. 55 734–55 784, 2023.
[40] N. Carlini, F. Tramer, E. Wallace, M. Jagielski, A. Herbert-V oss, K. Lee,
A. Roberts, T. Brown, D. Song, U. Erlingsson etal., “Extracting training
data from large language models,” in 30th USENIX security symposium
(USENIX Security 21), 2021, pp. 2633–2650.
APPENDIX
A. Missing experimental results
Table IX reports the utility results on all datasets under
different privacy budgets. Overall, the proposed method ex-
hibits a clear privacy–utility trade-off: as the privacy budget𝜖increases, the utility metrics generally improve across all
datasets. This trend is expected, since a larger privacy bud-
get introduces less perturbation during the privacy-preserving
generation process, allowing the generated responses to better
preserve the semantic information of the retrieved content.
More importantly, the performance degradation under strin-
gent privacy budgets remains relatively small. Even with a very
strong privacy guarantee (𝜖=0.1), the BLEU and ROUGE
scores are only slightly lower than those obtained with much
larger privacy budgets. For example, on the HealthCareMagic
dataset, the ROUGE score increases from 0.3387 at𝜖=0.1
to only 0.3501 at𝜖=10, while similar trends are observed
on the remaining datasets. This demonstrates that our method
effectively preserves generation quality while providing rig-
orous differential privacy guarantees, making it suitable for
privacy-sensitive RAG applications.
The relatively stable performance across a wide range of
privacy budgets indicates that the proposed prompt-aware
mechanism can accurately identify privacy-critical content,
enabling the privacy budget to be utilized efficiently rather
than perturbing all retrieved information uniformly.
B. Differential Privacy Guarantee of PA-HDP
We prove that PA-HDP satisfies differential privacy by
decomposing the algorithm into three components: (i) Laplace
perturbation for risk scoring, (ii) post-processing-based hier-
archical filtering, and (iii) exponential mechanism for private
candidate selection. The overall privacy guarantee follows
from sequential composition and post-processing invariance.
a) Step 1: Laplace Mechanism for Risk Score.:For each
segment𝑠 𝑖,𝑗, the risk score is computed as𝑆 𝑡𝑜𝑡𝑎𝑙(𝑞,𝑠𝑖,𝑗).
Since the risk score is normalized to the interval[0,1],
changing one neighboring segment under the semantic metric
alters the score by at most one. Therefore, the semantic
sensitivity of the risk score is bounded byΔ𝑆=1.
To privately release the risk score, PA-HDP allocates half
of the total privacy budget, i.e.,𝜖/2, and adds Laplace noise:
𝑆∗
𝑡𝑜𝑡𝑎𝑙(𝑞,𝑠𝑖,𝑗)=𝑆𝑡𝑜𝑡𝑎𝑙(𝑞,𝑠𝑖,𝑗)+Lap2
𝜖
.
Lemma A.1.The above mechanism satisfies(𝜖/2,0)-
Semantic Metric Differential Privacy.
Proof.Since the semantic sensitivity of the released risk score
isΔ𝑆=1, adding Laplace noise with scale
Δ𝑆
𝜖/2=2
𝜖
guarantees(𝜖/2,0)-Semantic Metric Differential Privacy ac-
cording to the Laplace mechanism under the semantic met-
ric.□
b) Step 2: Hierarchical Filtering (Post-processing).:The
algorithm applies thresholding:
𝑆∗
𝑡𝑜𝑡𝑎𝑙(𝑞,𝑠𝑖,𝑗)≥𝜏 1.
Since this operation is a deterministic function of the DP
output𝑆∗
𝑡𝑜𝑡𝑎𝑙, by the post-processing theorem of differential
privacy, it does not incur additional privacy loss.

JOURNAL OF L ATEX CLASS FILES, VOL. 14, NO. 8, AUGUST 2021 14
TABLE IX
UTILITYRESULTS UNDERDIFFERENTPRIVACYBUDGETS
HealthCareMagic NQ TQA WQ CT
𝜖 BLEU ROUGE BLEU ROUGE-L BLEU ROUGE-L BLEU ROUGE-L BLEU ROUGE-L
0.1 0.3231 0.3387 0.0666 0.1429 0.0638 0.1383 0.0592 0.1381 0.0623 0.1369
1 0.3266 0.3445 0.0672 0.1435 0.0673 0.1444 0.0629 0.1429 0.0638 0.1387
5 0.3275 0.3445 0.0674 0.1437 0.0688 0.1476 0.0666 0.1443 0.0638 0.1392
10 0.3337 0.3501 0.0676 0.1452 0.0676 0.1435 0.0685 0.1437 0.0650 0.1409
c) Step 3: Exponential Mechanism for Candidate Selec-
tion.:
LemmaIV .1[Semantic Exponential Mechanism] LetCbe
a finite candidate text set, and let
𝑢(𝑠𝑖,𝑗,𝑠𝑣
𝑖,𝑗)=−𝛼·𝑑 sem(𝑠𝑖,𝑗,𝑠𝑣
𝑖,𝑗)−(1−𝛼)·𝑟(𝑠 𝑖,𝑗,𝑠𝑣
𝑖,𝑗),
where𝛼≥0. The Semantic Exponential Mechanism selects
an output𝑠∗
𝑖,𝑗∈Caccording to the probability distribution
Pr[𝑀(𝑠𝑖,𝑗)=𝑠∗
𝑖,𝑗]=exp𝜖𝑢(𝑠 𝑖, 𝑗,𝑠∗
𝑖, 𝑗)
2
Í
𝑠𝑣
𝑖, 𝑗∈Cexp𝜖𝑢(𝑠 𝑖, 𝑗,𝑠𝑣
𝑖, 𝑗)
2.
Then𝑀satisfies(𝜖,0)-Semantic Metric Differential Privacy.
Proof.To prove that the mechanism satisfies Semantic Metric
Differential Privacy, it suffices to show that the utility function
is Lipschitz continuous with respect to the semantic metric𝑑 𝜙,
namely,
|𝑢(𝑠𝑖,𝑗,𝑠𝑣
𝑖,𝑗)−𝑢(𝑠′
𝑖,𝑗,𝑠𝑣
𝑖,𝑗)|≤Δ𝑢𝑑𝜙(𝑠𝑖,𝑗,𝑠′
𝑖,𝑗).
For the proposed utility function
𝑢(𝑠𝑖,𝑗,𝑠𝑣
𝑖,𝑗)=−𝛼·𝑑 sem(𝑠𝑖,𝑗,𝑠𝑣
𝑖,𝑗)−(1−𝛼)·𝑟(𝑠 𝑖,𝑗,𝑠𝑣
𝑖,𝑗),
we first analyze the semantic preservation term:
Δ𝑢(sem)=𝑑sem(𝑠𝑖,𝑗,𝑠𝑣
𝑖,𝑗)−𝑑 sem(𝑠′
𝑖,𝑗,𝑠𝑣
𝑖,𝑗)
=∥𝜙(𝑠𝑖,𝑗)−𝜙(𝑠𝑣
𝑖,𝑗)∥2−∥𝜙(𝑠′
𝑖,𝑗)−𝜙(𝑠𝑣
𝑖,𝑗)∥2
≤∥𝜙(𝑠𝑖,𝑗)−𝜙(𝑠′
𝑖,𝑗)∥2
=𝑑𝜙(𝑠𝑖,𝑗,𝑠′
𝑖,𝑗).
Thus, the semantic preservation term is1-Lipschitz contin-
uous with respect to𝑑 𝜙. Since this term is weighted by𝛼, its
contribution to the overall sensitivity is bounded by𝛼.
Next, we analyze the privacy risk term:
𝑟(𝑠𝑖,𝑗,𝑠𝑣
𝑖,𝑗)=max
𝑒∈𝐸(𝑠𝑣
𝑖, 𝑗)cos(𝜙(𝑠𝑖,𝑗),𝜙(𝑒)),
where𝐸(𝑠𝑣
𝑖,𝑗)denotes the set of sensitive entities extracted
from the candidate text𝑠𝑣
𝑖,𝑗. Assuming all embedding vectorsare normalized, the cosine similarity is equivalent to the inner
product. For two adjacent texts𝑠 𝑖,𝑗∼𝑠′
𝑖,𝑗, we have
Δ𝑢(risk)=|𝑟(𝑠𝑖,𝑗,𝑠𝑣
𝑖,𝑗)−𝑟(𝑠′
𝑖,𝑗,𝑠𝑣
𝑖,𝑗)|
=max
𝑒∈𝐸(𝑠𝑣
𝑖, 𝑗)cos(𝜙(𝑠𝑖,𝑗),𝜙(𝑒))−max
𝑒∈𝐸(𝑠𝑣
𝑖, 𝑗)cos(𝜙(𝑠′
𝑖,𝑗),𝜙(𝑒))
≤max
𝑒∈𝐸(𝑠𝑣
𝑖, 𝑗)|(𝜙(𝑠𝑖,𝑗)−𝜙(𝑠′
𝑖,𝑗))⊤𝜙(𝑒)|
≤∥𝜙(𝑠𝑖,𝑗)−𝜙(𝑠′
𝑖,𝑗)∥2·∥𝜙(𝑒)∥ 2
≤𝑑𝜙(𝑠𝑖,𝑗,𝑠′
𝑖,𝑗).
Therefore, the privacy risk term is also1-Lipschitz continu-
ous with respect to the semantic metric𝑑 𝜙. Since this term is
weighted by(1−𝛼), its contribution to the overall sensitivity
is bounded by(1−𝛼).
Combining both terms yields
|𝑢(𝑠𝑖,𝑗,𝑠𝑣
𝑖,𝑗)−𝑢(𝑠′
𝑖,𝑗,𝑠𝑣
𝑖,𝑗)|≤(𝛼+(1−𝛼))𝑑 𝜙(𝑠𝑖,𝑗,𝑠′
𝑖,𝑗).
Hence, the global sensitivity of the utility function under
the semantic metric is bounded by
Δ𝑢≤𝛼+(1−𝛼)=1.
Substituting the sensitivity bound into the exponential mech-
anism gives
Pr[𝑀(𝑠𝑖,𝑗)=𝑠∗
𝑖,𝑗]∝exp©­­­­­
«−𝜖
2
𝛼·𝑑 sem(𝑠𝑖,𝑗,𝑠∗
𝑖,𝑗)
+(1−𝛼)·𝑟(𝑠 𝑖,𝑗,𝑠∗
𝑖,𝑗)ª®®®®®
¬
which matches the formulation stated in the lemma. There-
fore, the mechanism satisfies(𝜖,0)-Semantic Metric Differen-
tial Privacy.
□
d) Step 4: Composition Over Segments.:Since the pri-
vacy budget𝜖/2is evenly allocated to the high-risk and
medium-risk segments, by the sequential composition theo-
rem, these two mechanisms together satisfy(𝜖/2,0)-Semantic
Metric Differential Privacy.
e) Theorem (Privacy Guarantee of PA-HDP).:PA-HDP
satisfies(𝜖,0)-Semantic Metric Differential Privacy.
Proof.The mechanism consists of: (i) Semantic Laplace
mechanism satisfying(𝜖,0)-Semantic Metric Differential Pri-
vacy, (ii) post-processing operations preserving DP, (iii) Se-
mantic exponential mechanism satisfying(𝜖,0)-Semantic Met-
ric Differential Privacy.

JOURNAL OF L ATEX CLASS FILES, VOL. 14, NO. 8, AUGUST 2021 15
By sequential composition and post-processing invariance,
the overall mechanism satisfies(𝜖,0)-Semantic Metric Differ-
ential Privacy.□
C. Details of Field Sensitivity Weights
To instantiate the field sensitivity defined in Section X, we
assign a predefined leakage risk weight𝑤 𝑘∈[0,1]to each
sensitive entity category. The weights are manually designed
according to the potential privacy harm caused by disclosing
each entity type, following the principle that entities enabling
direct identity disclosure receive substantially higher scores
than contextual or auxiliary information.
Specifically, we divide sensitive entities into two broad
categories.
a) High-risk entities (𝑤 𝑘∈ [0.5,1.0]):These entities
can directly identify an individual or expose highly sensitive
credentials (Table X ).
TABLE X
HIGH-RISK ENTITIES
Entity Type Weight Description
ID Number 0.95 Unique personal identifier with se-
vere privacy leakage risk
Phone Number 0.90 Direct contact identifier
Person Name 0.90 Primary identity attribute
Email Address 0.75 Online identity linkage
Password 0.80 Authentication credential
MED-HIGH 0.85 Diagnosis of high-risk diseases
b) Low-risk entities (𝑤 𝑘∈ [0.1,0.5]):These entities
mainly provide contextual or auxiliary information (Table XI
).
TABLE XI
LOW-RISK ENTITIES
Entity Type Weight Description
Location 0.30 Geographical contextual informa-
tion
Organization 0.40 Affiliation or institutional context
Vital Signs 0.20 Basic physiological measurements
Amount 0.50 Financial context without identity
disclosure
Date 0.20 Temporal information
MED-MED 0.10 Common disorder
We emphasize that the specific numerical values are em-
pirically defined and can be adapted to different application
scenarios without modifying the overall framework.
D. Implementation Details of Baseline Approaches
We adopt the baseline methods and implementation proto-
cols from the work of Zeng et al. [7]. The detailed implemen-
tation settings are described as follows.
1) Paraphrase:This approach employs large language
models to extract relevant and essential components from the
retrieved passages. Insignificant portions can be discarded,
while selected sentences are rephrased to improve clarity or
relevance. The instruction template we use for paraphrasing is
provided in Table XII.TABLE XII
PROMPT OF PARAPHRASE
Prompt
Given the following context, extract the useful or important part of the Context.
Remember,*DO NOT*edit the extracted parts of the context.
>Context:
>>>
{input_context}
>>>
Extracted relevant parts:
2) ZeroGen:The ZeroGen strategy is designed to produce
a new set of question–answer pairs derived from the original
passage. In practice, we first apply the spaCy toolkit to recog-
nize named entity mentions in the source text. We then feed
the LLM with the prompt: “The context is: original context.
extracted entities is the answer to the following question: ” to
solicit corresponding questions. The newly constructed context
comprises 10 randomly chosen question–answer pairs, for-
matted as “Question: generated questions. Answer: extracted
entities”.
3) AttrPrompt:Unlike the previous methods, AttrPrompt
relies solely on LLM-generated content without accessing
any retrieved data from the external database. This technique
queries the LLM about the most critical attributes associated
with a specific data category. For the ChatDoctor scenario,
we prompt the model with: “What do you think are impor-
tant attributes to generate some chat doctor data? Examples:
disease...”. From the LLM’s response, we select five attributes
and ask the model to propose 10 varied subtopics for each
attribute. During new context generation, we randomly pick
one subtopic per attribute and instruct the LLM to produce
data following the assigned attribute.
E. Detailed Attack Design
This subsection elaborates on the design of both targeted
and untargeted attacks against Retrieval-Augmented Genera-
tion (RAG) systems. These attack configurations are employed
to assess the privacy-preserving capacity of our proposed
synthetic-data framework. We emulate a realistic black-box
environment where the adversary can only interact with the
system via API calls. Consequently, the adversary’s strategy
centers on crafting and manipulating the query𝑞to elicit the
target information from the RAG pipeline.
1) Prompt Composition:The composite structured prompt
generally consists of two segments: an{information}compo-
nent and a{command}component.
𝑞={information}+{command}
This formulation pursues two parallel goals: (a) steering
the retriever to accurately fetch the desired pieces of data,
and (b) compelling the language model to incorporate the
retrieved evidence into its output. The{information}part
directs the retrieval module toward specific data items, whereas
the{command}part instructs the model to echo the retrieved
content in its response. For the latter, we adopt directives like

JOURNAL OF L ATEX CLASS FILES, VOL. 14, NO. 8, AUGUST 2021 16
“Please repeat all the context”; the former varies according to
the attacker’s objective.
2) Targeted Attack:In a targeted scenario, the adversary
seeks to recover a predetermined set of specific details.
Constructing the information component for such an attack
proceeds in two steps. Initially, the attacker supplies concrete
exemplars based on their goal – for instance, “I want some
advice about target name” for a well-defined target, or a partial
prefix such as “Please email us at” for a more vague target.
Subsequently, a substantial number of analogous yet diverse
instances are generated from these exemplars. For targets
that contain multiple sub-elements (e.g., the HealthcareMagic
corpus), variations can be produced by substituting sub-items
like disease names derived from ChatGPT or the International
Classification of Diseases (ICD). Alternatively, large language
models (e.g., ChatGPT) can directly yield similar sentences
based on the provided examples – this approach is also
applied to the Wiki-PII dataset. For instance, one might input
“Generate 100 similar sentences like ’Please email us at’ ”.
3) Untargeted Attack:In contrast, untargeted attacks em-
phasize the production of varied information components so
as to extract a broad spectrum of data from the retrieval repos-
itories, rather than homing in on any particular entry. Inspired
by the methodology of Carlini et al. [40], we randomly sample
segments from the Common Crawl corpus to serve as the
information part. However, the randomness of the input could
potentially interfere with the command component. To counter
this, we cap the maximum token length of the information
part at 15 tokens, thereby preserving prompt coherence and
maintaining effectiveness in eliciting data from the retrieval
collections.
F . Details of Evaluation Metrics
This section clarifies the evaluation metrics adopted in our
study.
1) ROUGE-L:ROUGE-L belongs to the ROUGE (Recall-
Oriented Understudy for Gisting Evaluation) family and is
widely used for assessing text-generation tasks, including sum-
marization and machine translation. It measures the overlap
between a generated output and a reference text via the
Longest Common Subsequence (LCS).
•Longest Common Subsequence (LCS):ROUGE-L
identifies the longest word sequence that appears in both
the generated and reference texts while preserving the
original order, without requiring contiguity.
•Recall, Precision, and F-measure:
–Recall is defined as the ratio of the LCS length to the
reference length (𝑛): Recall=LCS(𝑋,𝑌)/𝑛. It reflects
the fraction of the reference sequence captured by the
generator.
–Precision is the ratio of the LCS length to the generated
length (𝑚): Precision=LCS(𝑋,𝑌)/𝑚. It indicates
how much of the generated content aligns with the
reference.–The F-measure, which balances precision and recall via
their harmonic mean, is computed as:
𝐹lcs=(1+𝛽2)·𝑅 lcs·𝑃lcs
𝑅lcs+𝛽2·𝑃lcs
where𝛽controls the trade-off between precision and
recall (typically set to1.0). In our experimental results,
we report the F-measure as the ROUGE-L score.
Let𝐶denote the candidate translation, and𝑅represent the
set of reference translations.
2) BLEU-1:BLEU-1 evaluates translation quality based on
unigram precision.
•Unigram precision:
𝑃1=Í
𝑤min Count𝐶(𝑤),maxCount 𝑅(𝑤)
Í
𝑤Count𝐶(𝑤)
where Count 𝐶(𝑤)is the frequency of word𝑤in the can-
didate, andmaxCount 𝑅(𝑤)is the maximum frequency
of𝑤across any single reference translation.
•Brevity penalty (BP):
BP=min
1,exp
1−𝑟
𝑐
where𝑐is the candidate length and𝑟is the reference
length that is closest to𝑐.
•Final BLEU-1 score:
BLEU-1=BP×𝑃 1
The score ranges from 0 to 1, with 1 indicating perfect
unigram matching against the reference.
3) Additional Metrics:Beyond the above, we incorporate
several new metrics to further corroborate our approach:
•Exact Match (EM):This metric checks whether the
ground-truth answer appears verbatim within the LLM’s
generated response.
•LLM-based Correctness Judgment:We employ Ragas,
a widely adopted automatic evaluation pipeline for RAG
(with over 5.9k stars on GitHub), to assess the factual
correctness of the generated answers.
G. Construction of Wiki-PII Dataset
To assess the privacy-preserving performance of our pro-
posed approach against targeted extraction attempts, we adopt
the dataset construction methodology introduced in [7] to
create the Wiki-PII dataset. This corpus is specifically de-
signed to contain a substantial volume of personally identi-
fiable information (PII), thereby providing a rigorous testbed
for evaluating privacy-protection strategies. The construction
procedure comprises three main phases.
1)PII extraction:We first harvested real PII instances
from the Enron Mail corpus. Specifically, we employed
theurlextractlibrary to extract web URLs and
applied regular expressions to capture phone numbers and
personal email addresses.
2)Text chunking:In the second phase, we partitioned the
Wikipedia text collection using the recursive character

JOURNAL OF L ATEX CLASS FILES, VOL. 14, NO. 8, AUGUST 2021 17
text splitter provided by LangChain, with a chunk size of
1500 characters.
3)PII insertion:Finally, for each resulting text chunk, we
randomly appended the extracted PII items to the end of
every sentence within that chunk.
H. Dataset Preparation Details
In our experiments, we adopt the configuration established
in prior RAG privacy research [33] to simulate a realistic
scenario where a large-scale public repository inadvertently
contains certain private information — for instance, personal
names or phone numbers that appear within broadly accessible
web content. In practice, virtually any sentence in a text corpus
may carry privacy-sensitive entities. Consequently, embedding
PII into the dataset serves as a plausible and effective strategy
for modeling such real-world privacy risks.
I. Examples of Protected Contexts
Table XIII presents representative examples of the protected
query–response pairs in the retrieval corpus. Compared with
the original records, the protected versions effectively obfus-
cate privacy-sensitive information while preserving the se-
mantic content required for downstream retrieval. Specifically,
sensitive entities, including medication names, disease de-
scriptions, temporal information, numerical values, and demo-
graphic attributes, are replaced with plausible alternatives or
generalized expressions, making it substantially more difficult
to recover the original records or infer personal information.
Despite these perturbations, the protected query–response
pairs largely retain their original semantics. In the first ex-
ample, although several medications and personal details are
replaced, the protected record still clearly describes a fertility-
related consultation and its corresponding medical advice.
Similarly, in the second example, detailed medical history
and treatment information are concealed, while the protected
response continues to capture the clinical recommendation
regarding pregnancy risk. These examples demonstrate that our
protection mechanism effectively removes privacy-sensitive
content while preserving the essential medical knowledge
required for retrieval.
We also observe that stronger perturbations may occa-
sionally introduce minor semantic deviations or less natural
expressions, as illustrated in the third example. Nevertheless,
the overall medical intent and question–answer correspon-
dence remain largely intact, ensuring that the protected corpus
continues to provide useful evidence for retrieval-augmented
generation. These qualitative examples further demonstrate
that our method achieves a favorable balance between privacy
protection and retrieval utility.
J. Discussions when Adapting PA-HDP to Domain-Specific
Applications
Although PA-HDP is presented in the context of medical
and QA RAG, the overall framework is domain-independent
and can be readily adapted to other application scenarios.
The primary modification lies in the sensitive entity extractionmodule, as different domains involve different categories of
privacy-sensitive information.
Specifically, our framework relies on the extracted sensitive
entities to estimate field sensitivity and subsequently allocate
privacy budgets. Therefore, when applying PA-HDP to a new
domain, the Named Entity Recognition (NER) model and
rule-based extraction patterns should be customized according
to the domain-specific privacy taxonomy. For example, in
the financial domain, sensitive entities may include bank
account numbers, credit card numbers, transaction records,
and customer identifiers, while in the legal domain, case
numbers, client identities, and confidential agreements may
require protection. Existing domain-specific NER models can
be directly integrated into our framework, or custom entity
recognizers can be trained when annotated data are available.
In addition, the predefined sensitivity weights assigned to
different entity types should be adjusted to reflect their privacy
risks in the target domain. Since the subsequent privacy budget
allocation only depends on the extracted entity categories
and their corresponding sensitivity scores, no modification
to the remaining components of PA-HDP is required. This
modular design enables PA-HDP to be easily extended to
diverse privacy-sensitive RAG applications while preserving
its query-aware privacy protection mechanism.

JOURNAL OF L ATEX CLASS FILES, VOL. 14, NO. 8, AUGUST 2021 18
TABLE XIII
EXAMPLES OFPROTECTEDCONTEXTS
Original context Protected context
input: yes, can you tell me if lisinopril hctz , omeprazole ,
and simvastain effect my husbands sperm or fertility ? we
are trying to get pregnant and have been for 9yrs and we
have been to dr after dr test after test ... and no answers.
so im doing some findings and wanting to know about his
meds. thanks
output: Hi, Thanks for writing to Chat Doctor. Let me
explain you one by one. Surprisingly there are fewer studies
or researches on this topic. Omeprazole is not know to cause
any adverse effect on sperm count or fertility. Lisinopril
is not having direct effect on sperm count, but it effects
fertility by following side effects—Decreased interest in
sexual intercourse-inability to have or keep an erection-
lack or loss of strength-loss in sexual ability, desire, Chat
Doctor. But these are rare side effects. Simvastin rarely
cause erectile dysfunction. It is also shown in some studies
that it reduce testosterone levels which is required for sperm
production. This may be the cause of reduced sperm count.
You can discuss with your doctor to switch to alternative.input: yes, can you tell me if Eurax hctz , breakage of acid
bottle , and smoke effect my husbands sperm or fertility.
we are trying to get pregnant and have been for 6.8yrs and
we have been to dr after dr test after test. and no answers.
so im doing some findings and wanting to know about his
meds. thanks.
output: hi, thanks for writing to chat doctor. let me explain
you one by one. surprisingly there are fewer studies or
researches on this topic. palmetto is not know to cause
any adverse effect on sperm count or fertility. lisinopril
is not having direct effect on sperm count, but it effects
fertility by following side effects—decreased interest in
sexual intercourse-inability to have or keep an erection-lack
or loss of strength-loss in sexual ability, desire, chat doctor.
but these are rare side effects. simvastin rarely cause erectile
dysfunction. it is also shown in some studies that it reduce
testosterone levels which is required for sperm production.
this may be the cause of reduced sperm count. you can
discuss with your doctor to switch to alternative.
input: My wife has pulmonary fibrosis 2yrs now doing
fair,cough present but functional.this past feb. she had her
ascend colon removed secondary to colon cancer stage
1c(2).no chemo was needed....a month ago she turns up
pregnant. My question is how dangerous is it for her to
have this child.she is 42,not in great shape from 2 yrs of of
meds ,prednisone(ended in April).
output: Hi and welcome to Chat Doctor, thank you for your
query. Considering all this, her pregnancy may be risky, and
you should consult gynecologist before you decide to have
baby. Her condition is serious, and it may have bad effect on
possible pregnancy and baby. So I wouldst suggest going on
without more detailed consultation with gynecologist. Wish
you good health. Regards.input: My wife has pulmonary WAS EXPERIENCING
Sunday now doing fair,irritable present but functional. this
past feb. she had her ascend colon removed secondary to
cancer 1c(2). no chemo was needed. a week she turns up
pregnant. my question is how dangerous is it for her to have
this child. she is 42,not in great shape from 2 yrs of of meds
,prednisone(ended in april).
output: Hi and welcome to chat doctor, thank you for your
query. considering all this, her pregnancy may be risky, and
you should consult gynecologist before you decide to have
baby. her condition is serious, and it may have neurotoxic on
possible pregnancy and baby. so i wouldst suggest going on
without more detailed consultation with gynecologist. wish
you good health. regards.
input: Hi my daughter of 4.5 years old is suffering from
cough and fever since yesterday morning. Can I give her
zifi 100 mg ready syrup . Last year doctor had advised her
same antibiotic with crocin and for fever and cough. What
should be the dosage for her.
output: Hello, I am sorry your daughter is ill. With the
symptoms you listed-cough and fever- it is likely that she
has a viral respiratory illness. There are more viruses than
bacteria that cause respiratory infections. How high is her
fever? Is her appetite normal? Does she have her usual level
of activity? The severity of a viral respiratory illness is
usually less that for a bacterial respiratory illness. WiFi is a
brand name for the antibiotic called Definite. While this is
a good antibiotic to treat various bacterial infections, it does
not treat viral infections. In fact, no antibiotic does. Plus,
WiFi, like all Chat Doctor. Some of its side effects are:-
headache-loss of appetite-bloody or watery stool-dizziness-
itchy rash-swelling of face, lips, tongue So for the symptoms
you provided, this medicine most probably will not be
effective, and it could cause side effects. Based on the
information given, I feel it is not a good idea to give her
this medicine. Hope I have answered your query. Let me
know if I can assist you further.input: hi my daughter of 4. 22nd April is suffering from
blisters of 0 and 3 days around 3 minutes. Can i give her
zifi 94 mg ready syrup. 9 weeks doctor had advised her
same antibiotic with breakage of acid bottle for bad nappy
rash and cough. what should be the dosage for her.
output: hello, i am sorry your daughter is ill. with a brain
tumor you listed-cough and fever- it is likely that she has a
viral allergic reaction. there are more viruses than bacteria
that cause ascitis. how high is her cough. is her appetite
normal. does she have her usual level of activity. the severity
of a viral respiratory illness is usually less that for a bacterial
respiratory illness. wifi is a brand name for the antibiotic
called definite. while this is a good antibiotic to treat various
bacterial infections, it does not treat bad back pain. in fact,
no antibiotic does. plus, wifi, like all chat doctor. some of
its side effects stomach pains of abd painf face, lips, tongue
so for this pain medication you provided, this medicine
most probably will not be effective, and it could cause side
effects. based on the information given, i feel it is not a
good idea to give her this medicine. hope i have answered
your query. let me know if i can assist you further.