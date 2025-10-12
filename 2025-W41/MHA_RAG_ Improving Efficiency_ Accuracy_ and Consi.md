# MHA-RAG: Improving Efficiency, Accuracy, and Consistency by Encoding Exemplars as Soft Prompts

**Authors**: Abhinav Jain, Xinyu Yao, Thomas Reps, Christopher Jermaine

**Published**: 2025-10-06 20:41:43

**PDF URL**: [http://arxiv.org/pdf/2510.05363v1](http://arxiv.org/pdf/2510.05363v1)

## Abstract
Adapting Foundation Models to new domains with limited training data is
challenging and computationally expensive. While prior work has demonstrated
the effectiveness of using domain-specific exemplars as in-context
demonstrations, we investigate whether representing exemplars purely as text is
the most efficient, effective, and stable approach. We explore an alternative:
representing exemplars as soft prompts with an exemplar order invariant model
architecture. To this end, we introduce Multi-Head Attention
Retrieval-Augmented Generation (MHA-RAG), a framework with the number of
attention heads serving as a simple hyperparameter to control soft
prompt-generation across different tasks. Across multiple question-answering
benchmarks and model scales, MHA-RAG achieves a 20-point performance gain over
standard RAG, while cutting inference costs by a factor of 10X
GFLOPs-delivering both higher accuracy and greater efficiency, invariant to
exemplar order.

## Full Text


<!-- PDF content starts -->

Preprint
MHA-RAG: IMPROVINGEFFICIENCY, ACCURACY,
ANDCONSISTENCY BYENCODINGEXEMPLARS AS
SOFTPROMPTS
Abhinav Jain∗,1,Xinyu Yao∗,1,Thomas Reps2,Christopher Jermaine†,1
1Rice University2University of Wisconsin–Madison
aj70@rice.edu, xy38@rice.edu, reps@cs.wisc.edu, cmj4@rice.edu
ABSTRACT
Adapting Foundation Models to new domains with limited training data is chal-
lenging and computationally expensive. While prior work has demonstrated the
effectiveness of using domain-specific exemplars as in-context demonstrations,
we investigate whether representing exemplars purely as text is the most efficient,
effective, and stable approach. We explore an alternative: representing exem-
plars as soft prompts with an exemplar order invariant model architecture. To
this end, we introduce Multi-Head Attention Retrieval-Augmented Generation
(MHA-RAG), a framework with the number of attention heads serving as a simple
hyperparameter to control soft prompt-generation across different tasks. Across
multiple question-answering benchmarks and model scales, MHA-RAG achieves a
20-point performance gain over standard RAG, while cutting inference costs by a
factor of 10× GFLOPs—delivering both higher accuracy and greater efficiency,
invariant to exemplar order.
1 INTRODUCTION
With the rapid scaling of model parameters, tuning Foundation Models for domain adaptation has
become increasingly challenging due to both computational and data constraints. As an alternative,
In-Context Learning (ICL) has emerged as a training-free adaptation strategy (Xie et al., 2021; Min
et al., 2022). Rather than updating model parameters through gradient descent, ICL conditions the
model at inference time by providing a small set of task-specific exemplars directly in the input
prompt. Pioneering work such as GPT-3 (Brown et al., 2020) demonstrated that LLMs exhibit
strong in-context learning capabilities when presented with exemplars in natural-language form.
Subsequent studies in Wei et al. (2022) show that adding chain-of-thought exemplars further enhances
performance by encouraging LLMs to generate intermediate reasoning steps. Remarkably, in certain
settings, ICL has even outperformed fine-tuning-based methods (Mosbach et al., 2023; Pingua et al.,
2025; Mallen et al., 2022).
Despite its promise, ICL faces three fundamental limitations. First, representing exemplars in
textual format leads toincreasing inference cost, because attention complexity quadratically grows
with context length (Vaswani et al., 2017). Second, RAG has been shown to giveunsatisfiable
performancein domains with out-of-distribution data (Yu et al., 2024; Gupta et al., 2024). Third,
ICL is highly sensitive to exemplar order (examplar-order variance): prior studies demonstrate
that reordering exemplars can cause substantial performance fluctuations (Pezeshkpour & Hruschka,
2023; Zheng et al., 2023), and we observe similar phenomenon in our own experiments (see Table 2).
To mitigate these issues, soft-prompt-based tuning is a viable choice. Instead of representing
exemplars as long sequences of text, soft prompts encode exemplar information as a fixed set of
trainable continuous vectors, which are prepended to the input. Because these vectors are much
shorter than raw text, inference cost scales with the number of soft tokens rather than the full text
*Equal contribution.
†Corresponding author.
1arXiv:2510.05363v1  [cs.AI]  6 Oct 2025

Preprint
length, greatly reducing quadratic overhead. While methods have been proposed for compressing text
exemplars into soft prompts (Mu et al., 2023; Chevalier et al., 2023; Cheng et al., 2024; Rau et al.,
2024; Li et al., 2024), these approaches often incur a noticeable performance degradation compared
to text-based ICL. An alternative line of research investigates whether soft prompting can stand as an
effective parameter-efficient fine-tuning method for domain adaptation (Lester et al., 2021; Liu et al.,
2024), (Liu et al., 2021). Approaches such as ATTEMPT (Asai et al., 2022), Instance-Dependent
Prompt Generation (Wu et al., 2022) and LoPA (Jain et al., 2024a) demonstrate that adapting language
models with soft prompts is a cost-saving mechanism that can enhance their performance without
modifying model weights.
Therefore, in this paper, we ask the question: Can we learn a soft-prompt representation over
in-context exemplars that simultaneously leverages 1) the cost efficiency of soft prompts, 2) the
adaptability of in-context learning with retrieved exemplars, and 3) the architecture of neural networks
that provide an exemplar-order invariant framework?
To address this question, we propose the Multi-Head Attention Retrieval-Augmented Generation
framework (MHA-RAG). MHA-RAG learns to represent in-context exemplars as compact soft
prompts and employs a multi-head scaled dot-product-based attention (Vaswani et al., 2017) to
capture rich interactions between the query and each exemplar. Moreover, MHA-RAG introduces
the number of heads as a tunable hyperparameter, enabling flexible control over the length of the
generated soft prompts while simultaneously enforcing order-invariant aggregation across exemplar
representations. Our experiments with a wide range of language models on domains like Chemistry
and Medical QA demonstrate that this design provides (i) substantial gains in inference efficiency
by reducing quadratic token overhead, (ii) performance improvements over text-based retrieval and
prompt-tuning baselines, and (iii) stable and consistent results with the property of order-invariance
embodied in the design.
In this work, we make the following contributions:
•We address the three fundamental limitations in RAG, which include high inference cost,
unsatisfactory performance, and exemplar order variance within domain adaptions.
•We propose the MHA-RAG framework, which learns compact soft-prompt representations
over in-context examples to reduce inference cost, while achieving high performance.
•We propose an exemplar order invariant multi-head-attention architecture to control soft-
prompt generation in MHA-RAG, where the number of heads is a tunable hyperparameter.
•Our experiments show on average a 20-point performance gain over standard RAG across
benchmarks and models, while cutting inference cost by10×GFLOPs.
2 RELATEDWORK
2.1 PROMPT-COMPRESSION METHODS
Prompt-compression methods can be mainly divided into two categories: soft-prompt methods and
hard-prompt methods. Soft-prompt methods compress long contexts into dense representations to
reduce token count. GIST (Mu et al., 2023) learns “gist tokens” via modified attention masks, letting
generation condition only on a compact set of vectors. AutoCompressor (Chevalier et al., 2023)
and ICAE (Ge et al., 2023) similarly encode long inputs into summary vectors or memory slots
that can be reused without re-encoding. More recent works like PISCO (Louis et al., 2025) and
Embedding-based Memory Compressors (Dai et al., 2025) train memory tokens through distillation
or pretraining, while CMT (Li et al., 2025) encodes documents into dense vectors via cross-attention.
These methods achieve efficiency by replacing raw text with compact learned embeddings.
Hard-prompt methods remove tokens or text that contains less information. LLMLingua (Jiang et al.,
2023) iteratively prunes tokens with low perplexity, while RECOMP (Xu et al., 2023) selects or
summarizes sentences before prepending them to the input. Hierarchical methods, such as From
Reading to Compressing (R2C) (Choi et al., 2024), compress at the token, chunk, and sentence levels,
and training-free frameworks, like Perception Compressor (Tang et al., 2024), dynamically allocate
compression ratios across different parts of the input. These techniques are often model-agnostic, but
must balance compression rate with semantic fidelity.
2

Preprint
Figure 1: Comparison of domain-adaptation methods: (a) RAG uses retrieved exemplars directly as
context, while (b) xRAG, (c) xRAG-K, and (d) MHA-RAG derive soft-prompt representations from
exemplars. The figure illustrates how each method forms its representation (single vector in xRAG,
Kvectors in xRAG-K, multi-head representations in MHA-RAG) and highlights their trainable
components. Among the four methods, only MHA-RAG—due to its use of scaled dot-product
attention—is invariant to the order of exemplars.
In retrieval settings, compression enables handling many documents efficiently. COMPACT (Yoon
et al., 2024) sequentially compresses document segments, while EXIT (Hwang et al., 2024) filters
retrieved sentences to keep only relevant ones. xRAG (Cheng et al., 2024) projects a whole document
into a single token embedding, and COCOM (Rau et al., 2024) pre-computes compact embeddings for
offline reuse. Beyond token inputs, DyPRAG (Tan et al., 2025) injects retrieved knowledge as LoRA
weight updates, and AttentionRAG (Fang et al., 2025) uses model attention to prune retrievals in a
training-free manner. Together, these methods reduce redundancy in RAG pipelines while controlling
cost. Our MHA-RAG approach employs a multi-head attention-based soft prompt over retrieved
in-context exemplars, enabling rich interactions between the query and exemplars, while ensuring that
exemplar aggregation is order-invariant. Moreover, we can use the number of heads in the multi-head
attention mechanism as a tunable hyperparameter to control soft-prompt generation across various
tasks and models.
2.2 RETRIEVAL METHOD VERSUS FINE-TUNING
Retrieval-Augmented Generation (RAG) augments Foundation Models with external documents, pro-
viding grounded and up-to-date responses that surpass static fine-tuning methods in many knowledge-
intensive tasks (Lewis et al., 2020). The proposed RAG model achieved state-of-the-art accuracy on
open-domain QA benchmarks by retrieving different documents per query, outperforming purely
fine-tuned parametric baselines. Similarly, REALM (Guu et al., 2020) and RETRO (Borgeaud et al.,
2022) demonstrated that retrieval integration can enable smaller models to rival or surpass much
larger fine-tuned ones.
Moreover, RAG leverages retrieved examples as in-context demonstrations, enabling Foundation
Models to adapt to new domains without retraining. This strategy has been shown to improve
performance in low-data and specialised settings, with unsupervised domain-adaptation approaches
demonstrating significant gains in tasks like sentiment analysis and named entity recognition, etc.
(Guu et al., 2020; Lewis et al., 2020; Borgeaud et al., 2022; Jain et al., 2024b). More recent
3

Preprint
benchmarks highlight that retrieval augmentation offers a lightweight, cost-efficient alternative to
fine-tuning by dynamically expanding the model’s accessible knowledge (Behrouz et al., 2024).
Another fundamental difference between RAG and fine-tuned models lies in test-time adaptability.
RAG performs a fresh retrieval for each user query, dynamically tailoring the context to the question at
hand. Recent work on dynamic test-time compute for LLMs explores strategies that allocate inference
resources adaptively based on query complexity. Snell et al. (2024) formalizes compute-optimal
strategies—either using verifier-based search or adaptive response distribution, and outperforms
much larger models under a fixed FLOPs budget. MHA-RAG leverages the benefits of RAG for
domain adaptation—achieving superior performance in domain-specific tasks with less inference cost
compared to RAG.
3 METHODOLOGY
The objective of domain adaptation is to adapt a language model fθto a new task with data D=
{(xi, yi)}N
i=1such thaty=f θ(x).
3.1 PRELIMINARIES
Formally, domain adaptation involves updating the parameters of the model θto maximize the
likelihood of the response y, i.e., max θE(x,y)∼D [logp θ(y|x)] . In practice, Foundation Models are
often adapted using parameter-efficient fine-tuning (PEFT) methods (He et al., 2021), where only
a subset of parameters θ′⊂θ is updated. This approach reduces training costs while preserving
performance.
Domain adaptation with in-context exemplars.In domains where parameter updates are not
feasible, either due to limited training data or training being computationally expensive, adaptation
can be achieved by providing a set of a few domain-specific samples {(xk, yk)}K
k=1⊂D as in-
context exemplars. In this setting, the model prediction takes the form y=f θ({(x k, yk)}K
k=1, x).
This adaptation can be further enhanced by leveraging a relevance or similarity function Φ(x, x′).
Specifically, for each input x, the top- Kmost relevant samples in Dare retrieved as in-context
exemplars—i.e.,(x k, yk) = arg max(x′,y′)∈D\{(x,y),(x 1,y1),...,(x k−1,yk−1)}Φ(x, x′), for1≤k≤K.
Representing in-context exemplars.While approaches based on RAG typically represent in-context
exemplars in text form, we instead consider soft prompts as an alternative representation. To this
end, we define an encoding function g(DK|x) =Z that embeds DK, the top- Kin-domain samples
retrieved for a given x, into a soft prompt Z∈Rd×m, where mdenotes the virtual prompt length and
dis the hidden dimension of the language model. Similar to prompt tuning, the learned soft prompt
is prepended to the word embeddings ofx, thereby enabling domain adaptation without updatingθ.
3.2 MULTI-HEADEDATTENTION-BASEDRAG
In this paper, we propose a multi-headed attention-based encoding function for RAG (MHA-RAG),
where each head projects the top- Kin-context exemplars into the embedding space of the language
model as a single soft-prompt vector z∈ Rd. In particular, performing attention with Hheads
results in the soft prompt ZMHA= [z(1), . . .z(i). . . ,z(H)]of length m=H , where the output from
theithhead is
z(i)=ATTENTION(qi,Ki,Vi)
= softmax(qiKi
√
d)Vi,
withKi= [ki
1, . . .ki
k. . . ,ki
K]andVi= [vi
1, . . .vi
k. . . ,vi
K],
where qi=WQ
iExis the attention query corresponding to a given x, andki
k=WK
iExk⊕ykand
vi
k=WV
iExk⊕ykare the keys and values corresponding to(x k, yk)∈D K, respectively.
Ex∈ Rd′represents the dense representation of xobtained by a sentence-embedding model with
hidden dimension d′;⊕denotes the concatenation operator; and WQ
i∈ Rd×d′,WK
i∈ Rd×d′,
4

Preprint
BenchmarksRAG xRAG xRAG-K MHA-RAG
Qwen3-0.6B
BACE 76.01 60.88↓-15.1366.48↓-9.5375.08↓-0.93
BBBP 66.62 68.16↑+1.5476.57↑+9.9587.82↑+21.20
ClinTox 53.30 42.54↓-10.7666.05↑+12.7597.12↑+43.82
PubMedQA 58.09 72.49↑+14.469.73↑+11.6466.52↑+8.43
Qwen3-4B
BACE 75.87 59.07↓-16.855.46↓-20.4176.27↑+0.4
BBBP 69.33 64.44↓-4.8981.40↑+12.0782.49↑+13.16
ClinTox 44.24 40.19↓-4.0557.07↑+12.8396.32↑+52.08
PubMedQA 79.26 71.16↓-8.1070.36↓-8.976.59↓-2.67
Llama3.2-3B-Instruct
BACE 49.89 54.85↓-4.9664.04↑+14.1567.62↑+17.73
BBBP 46.15 72.87↑+26.7284.09↑+37.9487.61↑+41.46
ClinTox 37.90 62.88↑+24.9886.89↑+48.9994.44↑+56.54
PubMedQA 71.38 75.08↑+3.775.71↑+4.3376.34↑+4.96
∆avg
RAG -0.36 8.99 19.66
Table 1: Baseline comparison with RAG across benchmarks ( K= 5 ). Performance is reported
as effective accuracy—i.e, the geometric mean of the True-Positive and True-Negative rates (↑:
improvement relative to RAG,↓: drop relative to RAG). Random guessing yields an effective
accuracy of 50. Hyperparameters are tuned via sweeps: MHA-RAG ( H∈ {1,2,4,8} ). Results are
averaged over3random seeds. Underlined indicates the best effective accuracy.
andWV
i∈ Rd×d′represent the attention query, key, and value weights associated with head i,
respectively. Overall, we optimize the following objective:
max
φE(x,y)∼D,D K=Φ(x,·)
logp(y|g φ(DK|x), x)
,
such thatg φ(DK|x) =Z MHA, whereφrepresents the parameters of the encoding functiong.
Context compression and lower inference cost.Prior work (Cheng et al., 2024) interprets the
encoding function g(·)as a context-compression module achieving a compression ratio of|DK|
m. It
reduces the tokenized length of the context DKfrom|DK|tom, thereby lowering overall inference
cost during deployment. RAG without any compression yields a compression ratio of 1. In contrast,
MHA-RAG achieves a higher and configurable compression ratio of|DK|
H(m=H ). Configurability
comes from tailoring the soft-prompt size to the domain by varying the number of heads to adjust its
representational capacity. In comparison, xRAG (Cheng et al., 2024), while achieving the highest
possible compression (with m= 1 ), has a fixed ratio of |DK|as it collapses the context into a single
vector (see Figure 1 for reference).
Order Invariance.We argue that the soft prompts generated by MHA-RAG are invariant to the order
of retrieved exemplars. This property follows from the use of scaled dot-product attention, which
depends only on the set of input tokens rather than their order (Vaswani et al., 2017). Specifically,
the inputs are exemplar embeddings[E x1⊕y1, . . . , E xK⊕yK], and permuting them does not alter the
aggregated representation produced by each head. As a result, the generated soft prompt remains
identical regardless of exemplar ordering. A proof of this property is provided in Appendix A.1.
4 ADAPTING TOQUESTION-ANSWERINGDOMAINS
4.1 EXPERIMENTALSETUP
Benchmarks.In this work, we focus on domains where language models exhibit weak zero-shot
performance, but can benefit substantially from in-context exemplars or documents. Such exemplars
5

Preprint
provide cues in the form of analogies to similar samples, domain-specific formats (e.g., SMILES
strings for molecules), specialized vocabulary and facts, or templates for step-by-step reasoning.
We therefore benchmark across two groups of tasks. The first group involves limited-data molecular-
property-prediction tasks: (a)BACE[train/test split: 1413/100]—binary classification of whether a
molecule inhibits BACE1; (b)BBBP[train/test split: 1950/100]—prediction of blood–brain barrier
penetration (Yes/No); (c)ClinTox[train/test split: 1384/100]—classification of molecules as clinically
trial toxic vs. non-toxic (Guo et al., 2023). The second group consists of a medium-scale task: (d)
PubMedQA[train/test split: 30250/890]—biomedical question answering (Jin et al., 2019).
Figure 2: Comparison of inference compute in FLOPS
with respect toKPerformance Metrics.For yes/no clas-
sification tasks, we report the geomet-
ric mean of the True-Positive and True-
Negative rates, because it equally values
both classes and avoids the biases of accu-
racy or F1 under skewed label distributions.
We call this quantity the “Effective Accu-
racy.” To quantify overall improvement
relative to RAG, we compute ∆avg
RAG=
nqQn
i=1(100 +omethod
i −oRAG
i)−100
where oiis the score of a method on task
i, and nis the total number of benchmark
tasks across language models.
Foundation Models ( θ).We evaluate two
families of models:Llama-3.2-3B-Instruct
and theQwen3series, specificallyQwen3-
0.6Band4B(Yang et al., 2025).
Sentence-Embedding Models ( Ex).To encode in-context exemplars, we employ domain-specific
embedding models or encoder-only language models. For molecular tasks (BACE1,BBBP,ClinTox),
we useChemBERTa-2-10M-MTR(Ahmad et al., 2022). For biomedical QA (PubMedQA), we use
Qwen3-Embedding-0.6B(Zhang et al., 2025).
Retrieval Functions ( Φ).We assume that a domain-specific retrieval function is provided. For
molecular datasets, we retrieve the top- Krelevant exemplars using Tanimoto Similarity (Tanimoto,
1958; Guo et al., 2023), which computes scaffold-level similarity between SMILES representations.
ForPubMedQA, we retrieve the top- Kdocuments most relevant to a question from the training
corpus. Each document dockis embedded into a dense vector EdockusingQwen3-Embedding-8B
(Zhang et al., 2025), and retrieval is performed via cosine similarity.
Baselines.We compare against domain-adaptation baselines that do not fine-tune the foundation
model. Instead, they leverage retrieved in-context exemplars from the training database. These
include (a)RAG, which presents exemplars directly as text to the model, and (b)xRAGand (c)
xRAG-K, which construct soft prompts from the exemplars. InxRAG, all exemplars are encoded
into a single vector—i.e., Z=MLP(E x⊕x 1⊕y1...xK⊕yK), where Z∈ Rd×1.xRAG-K, on the other
hand, encodes each exemplar separately; i.e., Z=MLP(E x⊕x 1⊕y1)⊕. . .⊕MLP(E x⊕xK⊕yK)
where Z∈ Rd×K. Both methods employ a single-layer MLP to project the soft prompts from the
embedding space of the sentence encoder into the base model’s input space. Refer Figure 1.
Training Details.Models are tuned for 10 epochs on the limited-data molecular benchmarks (BACE1,
BBBP,ClinTox) and 1 epoch on the larger-scale benchmarks (PubMedQA). Learning rates are selected
from{1e−5,3e−5,5e−4} . The execution platform used 8× V100 GPUs (32GB VRAM each).
The embedding models are fine-tuned with LoRA (rank 64). The comparison of overall trainable
parameters are provided in Appendix A.5.
We report results on additional benchmarks and experiments in Appendix A.2 and A.3.
6

Preprint
(a)K= 1,Qwen3-4B
 (b)K= 5,Qwen3-4B
Figure 3: Varying number of heads in MHA-RAG. Performance averaged across 3 seeded runs.
4.2 BASELINECOMPARISON
Performance.As shown by the underlining in Table 1, MHA-RAG achieves the greatest improvement
in effective accuracy over RAG for almost all configurations tested, with an average gain of 19.66 ,
computed as the geometric mean of improvements across tasks. This improvement stems from its
flexible representational capacity: by varying the number of attention heads—each with its own set
of weights—the model can specialize in capturing different types of dependencies across exemplars.
In contrast, xRAG performs extreme compression by collapsing all retrieved samples into a single
vector, which can possibly lead to loss of information. Its extension, xRAG-K improves over RAG
by encoding each exemplar separately, but all exemplars are processed with identical weights. In
contrast, MHA-RAG assigns distinct weights to each head and computes representations through
attention over all exemplars, allowing head-specific representations.
Inference Cost.Figure 2 further shows that across K= 1. . .10 , RAG incurs substantially higher
inference cost (in FLOPS) than MHA-RAG, due to its longer context length and the quadratic scaling
of attention with respect to K. With a single head, MHA-RAG matches the cost of xRAG, and for
head counts< K, it is more computationally efficient than xRAG-K.
BenchmarksRAG xRAG xRAG-K Ours
BACE 3.18 2.80 1.120.0
BBBP 2.25 3.02 1.270.0
ClinTox 8.52 0.12 4.910.0
PubMedQA 4.30 0.61 2.000.0
Table 2: Order-(in)variance analysis forQwen3-4Bwith
K= 5 : standard deviation in performance when ex-
emplar order is randomized across 5 seeded shuffles.
(Lower numbers are better; zero means order-invariant.)
Boldshows the lowest standard deviation achieved.Order (In)variance.From Table 2, we ob-
serve that all baselines—RAG, xRAG, and
xRAG-K—exhibit non-zero variance when
the order of in-context exemplars or docu-
ments is shuffled. For RAG, this sensitivity
is expected because exemplars are concate-
nated as text, and order directly influences
the model’s input. In xRAG, order depen-
dence arises from the positional encodings
of the sentence-embedding encoder model:
when all exemplars are jointly encoded into
a single vector, reordering alters the result-
ing embedding. In xRAG-K, although each
exemplar is encoded independently, the po-
sition of each resulting vector within the
soft prompt is determined by exemplar order, leading to downstream variance. In contrast, our
proposed MHA-RAG is order-invariant by design, as formally established in Appendix A.1.
Findings.These results demonstrate that by spending an upfront cost for training, MHA-RAG
achieves both higher effective accuracy than baselines and lower inference cost compared to RAG.
7

Preprint
BenchmarksRAG MHA-RAG
K=1 K=5 K=10 K=1 K=5 K=10
Qwen3-4B
BACE 58.50 75.87 75.87 59.8976.27 52.42
BBBP 67.50 69.33 68.01 72.1282.49 80.43
ClinTox 30.94 44.24 54.48 63.1996.32 89.44
PubMedQA 52.68 79.2680.34 55.33 76.59 66.07
Llama3.2-3B-Instruct
BACE 51.71 49.89 36.36 52.3767.62 54.73
BBBP 35.81 46.15 52.08 71.7887.61 86.80
ClinTox 38.59 37.90 47.27 64.9594.44 89.33
PubMedQA 54.59 71.38 73.53 57.9176.34 72.50
Table 3: Effective accuracy of MHA-RAG vs. RAG under varying K. Results are averaged over
3 random seeds.Boldindicates the best performance. Underlined indicates the value of Kthat
achieves the best effective accuracy for a given method.
4.3 CONTEXTSATURATION WITHSOFTPROMPTS
In this section, we examine how effective accuracy varies with the number of retrieved exemplars
(K), focusing on context saturation (Vladika & Matthes, 2025), the point at which adding more
context introduces noise and causes effective accuracy to plateau or decline.
As shown in Table 3, increasing Kgenerally improves effective accuracy for both RAG ( K= 1→
5→10 ) and MHA-RAG ( K= 1→5 ), reflecting the benefit of richer context. A key finding is that
MHA-RAG at K=5 generally outperforms RAG at both K=5 and K=10, indicating that it is more
effective at extracting and representing information that is sufficient to answer the questions (Joren
et al., 2024). Moreover, it achieves this performance with fewer FLOPs than RAG (See Figure 2).
We further observe that MHA-RAG reaches context saturation earlier than RAG: effective accuracy
peaks around K=5, whereas RAG continues to improve up to K=10. Beyond these points, additional
exemplars degrade effective accuracy, likely due to the inclusion of less relevant or noisy samples,
consistent with findings of Liu et al. (2023); Hsieh et al. (2024); Zhao (2023).
Findings.MHA-RAG enables more effective domain adaptation with fewer exemplars over RAG.
4.4 EFFECTIVEACCURACY AS AFUNCTION OFATTENTIONHEADS
Having examined effective-accuracy variation with K, we now study the impact of the number of
heads. From Figures 3a and 3b, we observe that when K= 1 , increasing the number of heads
does not improve effective accuracy, likely due to the limited context—only a single exemplar or
document—offering little room for multiple heads to learn diverse representations. In contrast,
atK= 5 , increasing the number of heads generally leads to higher effective accuracy across
benchmarks, suggesting that the benefit of multiple heads emerges only when sufficient context is
available. Additional plots are provided in Appendix A.4.
Findings.MHA-RAG can effectively exploit additional context with an increasing number of heads.
4.5 COMPARISON WITH FINE-TUNING BASELINES
Because MHA-RAG requires some upfront training, we compare it against other fine-tuning methods.
Specifically, we include LoRA (Hu et al., 2022) as a standard baseline for domain adaptation and
evaluate against other soft-prompt fine-tuning methods—Prompt Tuning (PT) (Lester et al., 2021),
Instance-Dependent Prompt Generation (IDPG) (Wu et al., 2022). As shown in Table 4, MHA-RAG
achieves the highest average improvement of 47.45 in effective-accuracy over LoRA, with gains
computed as the geometric mean across tasks. We next explain the performance gaps method by
method, focusing on one type of task at a time.
8

Preprint
BenchmarksLoRA Off-the-Shelf PT IDPG MHA-RAG
Qwen3-0.6B
BACE 0.0 36.67↑+36.670.0 53.75↑+53.7575.08↑+75.08
BBBP 0.0 22.21↑+22.210.0 86.52↑+86.5287.82↑+87.82
ClinTox 0.0 0.0 0.0 94.87↑+94.8797.12↑+97.12
PubMedQA 73.94 13.96↓-59.9838.99↓-34.9545.50↓-28.4466.52↓-7.42
Qwen3-4B
BACE 12.32 0.0↓-12.320.0↓-12.3262.54↑50.2276.27↑-7.42
BBBP 19.61 10.88↓-8.7311.10↓-8.5187.14↑67.5382.49↑63.95
ClinTox 0.0 0.0 0.0 94.36↑+94.3696.32↑+96.32
PubMedQA 78.31 53.65↓-24.6681.04↑+2.7383.65↑+5.3476.59↓-1.72
Llama3.2-3B-Instruct
BACE 0.0 0.0 13.61↑+13.6135.28↑+35.2867.62↑+67.62
BBBP 54.77 0.0↓-54.7736.89↓-17.8887.14↑+32.3787.61↑+32.84
ClinTox 31.62 0.0↓-31.6261.88↑+30.2694.87↑+63.2594.44↑+62.82
PubMedQA 82.15 56.73↓-25.4264.68↓-17.4789.29↑+7.1476.34↓-5.81
∆avg
LoRA -17.96 -5.04 41.52 47.45
Table 4: Baseline comparison with other PEFT methods. Performance is reported as effective
accuracy—i.e, the geometric mean of the True-Positive and True-Negative rates (↑: improvement
relative to LoRA,↓: drop relative to LoRA). Random guessing yields an effective accuracy of 50.
Hyperparameters are tuned via sweeps: Prompt Tuning (virtual tokens ∈ {1,5,10} ), LoRA (rank
∈ {16,32,64} ), and MHA-RAG ( H∈ {1,2,4,8} ).Underlined indicates the best effective accuracy.
Molecular-property-prediction tasks (BACE, BBBP, and ClinTox).As shown in Table 4, LoRA
often overfits on limited-data BACE, BBBP, and ClinTox benchmarks, collapsing to predictions
dominated by a single class. Further, on these tasks MHA-RAG outperforms both PT and IDPG,
achieving ∆avg
PT= 69.95 and∆avg
IDPG= 7.01 . This gap can be explained by how soft prompts are
constructed: PT learns a fixed, question-independent prompt shared across the entire task, while
IDPG conditions its generated prompt on the input question, but ignores related training exemplars.
MHA-RAG instead conditions the soft prompt on retrieved exemplars that are most relevant to the
current query, allowing it to further leverage task-specific information. These results highlight the
benefit of grounding adaptation in limited data settings with retrieved exemplars, rather than relying
solely on parametric tuning.
Question Answering in PubMedQA.Unlike molecular-property-prediction tasks, answering ques-
tions in PubMedQA requires access to supporting documents. To ensure comparability, fine-tuning
baselines are provided with golden documents as additional context sufficient for answering each
question (Joren et al., 2024), presented directly to the foundation model during training and inference.
In contrast, MHA-RAG encodes these documents into soft-prompt vectors via its multi-headed
attention, rather than appending them directly to the model’s input. We also include a comparison
with Off-the-Shelf model, where the foundation model is queried without access to these documents,
serving as a measure of how much parametric knowledge alone supports question answering. MHA-
RAG improves over Off-the-Shelf by an average effective-accuracy gain of 30.91 , indicating that its
soft prompts effectively capture and represent knowledge from golden documents. Lastly, relative to
LoRA, MHA-RAG shows an average effective-accuracy drop of−5.01, computed as the geometric
mean of drops in PubMedQA. This is expected given that it compresses documents into at most eight
soft-prompt tokens. However, inference with MHA-RAG in PubMedQA is computationally cheaper
than LoRA, PT and IDPG, because the latter fine-tuning baselines incur inference costs at least as
high as RAG (see Figure 2) when documents are pre-pended to the model input.
Findings.In domains with limited data, MHA-RAG is an effective method for adapting foundation
models; it also efficiently compresses supporting documents into compact soft-prompt tokens.
9

Preprint
5 CONCLUSION
In this paper, we propose MHA-RAG, an attention-based method inspired by RAG that utilize
in-context exemplars to enhance performance. Our approach combines soft-prompt-based domain
adaptation with an order-invariant architecture to reduce inference cost and stabilize performance.
We also provide the number of heads in the architecture as a tunable parameter to control soft-prompt
generation across different experimental setups. With a more efficient, effective, and consistent way
of representing in-context exemplars, this work aims to position MHA-RAG as an alternative to RAG
for adapting language models to domains. In future, we want to study how well MHA-RAG scales to
tasks requiring inference with longer documents and how robust it is to ‘lost-in-the-middle’ problem
(Liu et al., 2023), a common issue in RAG that appears with scaling..
REFERENCES
Walid Ahmad, Elana Simon, Seyone Chithrananda, Gabriel Grand, and Bharath Ramsundar.
Chemberta-2: Towards chemical foundation models.arXiv preprint arXiv:2209.01712, 2022.
Akari Asai, Mohammadreza Salehi, Matthew E Peters, and Hannaneh Hajishirzi. Attempt:
Parameter-efficient multi-task tuning via attentional mixtures of soft prompts.arXiv preprint
arXiv:2205.11961, 2022.
Ali Behrouz, Peilin Zhong, and Vahab Mirrokni. Titans: Learning to memorize at test time.arXiv
preprint arXiv:2501.00663, 2024.
Sebastian Borgeaud, Arthur Mensch, Jordan Hoffmann, Trevor Cai, Eliza Rutherford, Katie Millican,
George Bm Van Den Driessche, Jean-Baptiste Lespiau, Bogdan Damoc, Aidan Clark, et al.
Improving language models by retrieving from trillions of tokens. InInternational conference on
machine learning, pp. 2206–2240. PMLR, 2022.
Tom Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared D Kaplan, Prafulla Dhariwal,
Arvind Neelakantan, Pranav Shyam, Girish Sastry, Amanda Askell, et al. Language models are
few-shot learners.Advances in neural information processing systems, 33:1877–1901, 2020.
Xin Cheng, Xun Wang, Xingxing Zhang, Tao Ge, Si-Qing Chen, Furu Wei, Huishuai Zhang, and
Dongyan Zhao. xrag: Extreme context compression for retrieval-augmented generation with one
token.Advances in Neural Information Processing Systems, 37:109487–109516, 2024.
Alexis Chevalier, Alexander Wettig, Anirudh Ajith, and Danqi Chen. Adapting language models to
compress contexts.arXiv preprint arXiv:2305.14788, 2023.
Eunseong Choi, Sunkyung Lee, Minjin Choi, June Park, and Jongwuk Lee. From reading to
compressing: Exploring the multi-document reader for prompt compression.arXiv preprint
arXiv:2410.04139, 2024.
Yuhong Dai, Jianxun Lian, Yitian Huang, Wei Zhang, Mingyang Zhou, Mingqi Wu, Xing Xie,
and Hao Liao. Pretraining context compressor for large language models with embedding-based
memory. InProceedings of the 63rd Annual Meeting of the Association for Computational
Linguistics (Volume 1: Long Papers), pp. 28715–28732, 2025.
Yixiong Fang, Tianran Sun, Yuling Shi, and Xiaodong Gu. Attentionrag: Attention-guided context
pruning in retrieval-augmented generation.arXiv preprint arXiv:2503.10720, 2025.
Tao Ge, Jing Hu, Lei Wang, Xun Wang, Si-Qing Chen, and Furu Wei. In-context autoencoder for
context compression in a large language model.arXiv preprint arXiv:2307.06945, 2023.
Taicheng Guo, Bozhao Nan, Zhenwen Liang, Zhichun Guo, Nitesh Chawla, Olaf Wiest, Xiangliang
Zhang, et al. What can large language models do in chemistry? a comprehensive benchmark on
eight tasks.Advances in Neural Information Processing Systems, 36:59662–59688, 2023.
Shailja Gupta, Rajesh Ranjan, and Surya Narayan Singh. A comprehensive survey of retrieval-
augmented generation (rag): Evolution, current landscape and future directions.arXiv preprint
arXiv:2410.12837, 2024.
10

Preprint
Kelvin Guu, Kenton Lee, Zora Tung, Panupong Pasupat, and Mingwei Chang. Retrieval augmented
language model pre-training. InInternational conference on machine learning, pp. 3929–3938.
PMLR, 2020.
Junxian He, Chunting Zhou, Xuezhe Ma, Taylor Berg-Kirkpatrick, and Graham Neubig. Towards a
unified view of parameter-efficient transfer learning.arXiv preprint arXiv:2110.04366, 2021.
Cheng-Yu Hsieh, Yung-Sung Chuang, Chun-Liang Li, Zifeng Wang, Long T Le, Abhishek Kumar,
James Glass, Alexander Ratner, Chen-Yu Lee, Ranjay Krishna, et al. Found in the middle: Calibrat-
ing positional attention bias improves long context utilization.arXiv preprint arXiv:2406.16008,
2024.
Edward J Hu, Yelong Shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang,
Weizhu Chen, et al. Lora: Low-rank adaptation of large language models.ICLR, 1(2):3, 2022.
Taeho Hwang, Sukmin Cho, Soyeong Jeong, Hoyun Song, SeungYoon Han, and Jong C Park.
Exit: Context-aware extractive compression for enhancing retrieval-augmented generation.arXiv
preprint arXiv:2412.12559, 2024.
Abhinav Jain, Swarat Chaudhuri, Thomas Reps, and Chris Jermaine. Prompt tuning strikes back:
Customizing foundation models with low-rank prompt adaptation.Advances in Neural Information
Processing Systems, 37:47297–47316, 2024a.
Abhinav Jain, Chris Jermaine, and Vaibhav Unhelkar. Rag-modulo: Solving sequential tasks using
experience, critics, and language models.arXiv preprint arXiv:2409.12294, 2024b.
Huiqiang Jiang, Qianhui Wu, Chin-Yew Lin, Yuqing Yang, and Lili Qiu. Llmlingua: Compressing
prompts for accelerated inference of large language models.arXiv preprint arXiv:2310.05736,
2023.
Qiao Jin, Bhuwan Dhingra, Zhengping Liu, William W Cohen, and Xinghua Lu. Pubmedqa: A
dataset for biomedical research question answering.arXiv preprint arXiv:1909.06146, 2019.
Hailey Joren, Jianyi Zhang, Chun-Sung Ferng, Da-Cheng Juan, Ankur Taly, and Cyrus Rashtchian.
Sufficient context: A new lens on retrieval augmented generation systems.arXiv preprint
arXiv:2411.06037, 2024.
Brian Lester, Rami Al-Rfou, and Noah Constant. The power of scale for parameter-efficient prompt
tuning.arXiv preprint arXiv:2104.08691, 2021.
Patrick Lewis, Ethan Perez, Aleksandra Piktus, Fabio Petroni, Vladimir Karpukhin, Naman Goyal,
Heinrich K ¨uttler, Mike Lewis, Wen-tau Yih, Tim Rockt ¨aschel, et al. Retrieval-augmented genera-
tion for knowledge-intensive nlp tasks.Advances in neural information processing systems, 33:
9459–9474, 2020.
Dongfang Li, Zetian Sun, Xinshuo Hu, Baotian Hu, and Min Zhang. Cmt: A memory compression
method for continual knowledge learning of large language models. InProceedings of the AAAI
Conference on Artificial Intelligence, volume 39, pp. 24413–24421, 2025.
Zongqian Li, Yixuan Su, and Nigel Collier. 500xcompressor: Generalized prompt compression for
large language models.arXiv preprint arXiv:2408.03094, 2024.
Nelson F Liu, Kevin Lin, John Hewitt, Ashwin Paranjape, Michele Bevilacqua, Fabio Petroni,
and Percy Liang. Lost in the middle: How language models use long contexts.arXiv preprint
arXiv:2307.03172, 2023.
Xiao Liu, Kaixuan Ji, Yicheng Fu, Weng Lam Tam, Zhengxiao Du, Zhilin Yang, and Jie Tang.
P-tuning v2: Prompt tuning can be comparable to fine-tuning universally across scales and tasks.
arXiv preprint arXiv:2110.07602, 2021.
Xiao Liu, Yanan Zheng, Zhengxiao Du, Ming Ding, Yujie Qian, Zhilin Yang, and Jie Tang. Gpt
understands, too.AI Open, 5:208–215, 2024.
11

Preprint
Maxime Louis, Herv ´e D´ejean, and St ´ephane Clinchant. Pisco: Pretty simple compression for
retrieval-augmented generation.arXiv preprint arXiv:2501.16075, 2025.
Alex Mallen, Akari Asai, Victor Zhong, Rajarshi Das, Daniel Khashabi, and Hannaneh Hajishirzi.
When not to trust language models: Investigating effectiveness of parametric and non-parametric
memories.arXiv preprint arXiv:2212.10511, 2022.
Sewon Min, Xinxi Lyu, Ari Holtzman, Mikel Artetxe, Mike Lewis, Hannaneh Hajishirzi, and Luke
Zettlemoyer. Rethinking the role of demonstrations: What makes in-context learning work?arXiv
preprint arXiv:2202.12837, 2022.
Marius Mosbach, Tiago Pimentel, Shauli Ravfogel, Dietrich Klakow, and Yanai Elazar. Few-shot fine-
tuning vs. in-context learning: A fair comparison and evaluation.arXiv preprint arXiv:2305.16938,
2023.
Jesse Mu, Xiang Li, and Noah Goodman. Learning to compress prompts with gist tokens.Advances
in Neural Information Processing Systems, 36:19327–19352, 2023.
Pouya Pezeshkpour and Estevam Hruschka. Large language models sensitivity to the order of options
in multiple-choice questions.arXiv preprint arXiv:2308.11483, 2023.
Bhagyajit Pingua, Adyakanta Sahoo, Meenakshi Kandpal, Deepak Murmu, Jyotirmayee Rautaray,
Rabindra Kumar Barik, and Manob Jyoti Saikia. Medical llms: Fine-tuning vs. retrieval-augmented
generation.Bioengineering, 12(7):687, 2025.
David Rau, Shuai Wang, Herv ´e D´ejean, and St ´ephane Clinchant. Context embeddings for efficient
answer generation in rag.arXiv preprint arXiv:2407.09252, 2024.
Jiayi Sheng, Luna Lyu, Jikai Jin, Tony Xia, Alex Gu, James Zou, and Pan Lu. Solving inequality
proofs with large language models.arXiv preprint arXiv:2506.07927, 2025.
Charlie Snell, Jaehoon Lee, Kelvin Xu, and Aviral Kumar. Scaling llm test-time compute optimally
can be more effective than scaling model parameters.arXiv preprint arXiv:2408.03314, 2024.
Yuqiao Tan, Shizhu He, Huanxuan Liao, Jun Zhao, and Kang Liu. Dynamic parametric retrieval
augmented generation for test-time knowledge enhancement.arXiv preprint arXiv:2503.23895,
2025.
Jiwei Tang, Jin Xu, Tingwei Lu, Zhicheng Zhang, Yiming Zhao, Lin Hai, and Hai-Tao Zheng.
Perception compressor: A training-free prompt compression framework in long context scenarios.
arXiv preprint arXiv:2409.19272, 2024.
Taffee T Tanimoto. An elementary mathematical theory of classification and prediction. 1958.
Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez, Łukasz
Kaiser, and Illia Polosukhin. Attention is all you need.Advances in neural information processing
systems, 30, 2017.
Juraj Vladika and Florian Matthes. On the influence of context size and model choice in retrieval-
augmented generation systems.arXiv preprint arXiv:2502.14759, 2025.
Jason Wei, Xuezhi Wang, Dale Schuurmans, Maarten Bosma, Fei Xia, Ed Chi, Quoc V Le, Denny
Zhou, et al. Chain-of-thought prompting elicits reasoning in large language models.Advances in
neural information processing systems, 35:24824–24837, 2022.
Zhuofeng Wu, Sinong Wang, Jiatao Gu, Rui Hou, Yuxiao Dong, VG Vydiswaran, and Hao Ma. Idpg:
An instance-dependent prompt generation method.arXiv preprint arXiv:2204.04497, 2022.
Sang Michael Xie, Aditi Raghunathan, Percy Liang, and Tengyu Ma. An explanation of in-context
learning as implicit bayesian inference.arXiv preprint arXiv:2111.02080, 2021.
Fangyuan Xu, Weijia Shi, and Eunsol Choi. Recomp: Improving retrieval-augmented lms with
compression and selective augmentation.arXiv preprint arXiv:2310.04408, 2023.
12

Preprint
An Yang, Anfeng Li, Baosong Yang, Beichen Zhang, Binyuan Hui, Bo Zheng, Bowen Yu, Chang
Gao, Chengen Huang, Chenxu Lv, et al. Qwen3 technical report.arXiv preprint arXiv:2505.09388,
2025.
Chanwoong Yoon, Taewhoo Lee, Hyeon Hwang, Minbyul Jeong, and Jaewoo Kang. Compact: Com-
pressing retrieved documents actively for question answering.arXiv preprint arXiv:2407.09014,
2024.
Hao Yu, Aoran Gan, Kai Zhang, Shiwei Tong, Qi Liu, and Zhaofeng Liu. Evaluation of retrieval-
augmented generation: A survey. InCCF Conference on Big Data, pp. 102–120. Springer, 2024.
Yanzhao Zhang, Mingxin Li, Dingkun Long, Xin Zhang, Huan Lin, Baosong Yang, Pengjun Xie,
An Yang, Dayiheng Liu, Junyang Lin, et al. Qwen3 embedding: Advancing text embedding and
reranking through foundation models.arXiv preprint arXiv:2506.05176, 2025.
Jiachen Zhao. In-context exemplars as clues to retrieving from large associative memory.arXiv
preprint arXiv:2311.03498, 2023.
Chujie Zheng, Hao Zhou, Fandong Meng, Jie Zhou, and Minlie Huang. Large language models are
not robust multiple choice selectors.arXiv preprint arXiv:2309.03882, 2023.
13

Preprint
A APPENDIX
A.1 MHA-RAG: IN-CONTEXTEXEMPLAR-ORDERINVARIANCE
In this section, we argue that the soft prompt generated by MHA-RAG is invariant to the order of
in-context exemplars/documents.
Claim 1(Exemplar-Order Invariance).Let qbe a query vector for a given xand for k= 1. . . K , let
kk∈ Rdandvk∈ Rdbe keys and value vectors derived from each retrieved example. Define the
attention weights as αk:=eskPK
j=1esjwhere score, sk:=q·kk√
d. The attention output for each head
is given by z=PK
k=1αkvk. Then zis invariant under any permutation of the retrieved examples:
permuting the indexing of the pairs{(k k,vk)}K
k=1does not changez.
Proof. Let πbe a permutation of 1. . . K and consider the permuted sequence
(kπ(1),vπ(1)), . . . ,(k π(K),vπ(K)). For the permuted sequence, the scores and weights are
s′
k=q·k π(k)√
d=sπ(k) α′
k=es′
k
PK
j=1es′
j=esπ(k)
PK
j=1esπ(j)
Because {sπ(j):j= 1, . . . , K} is a reordering of {sj:j= 1, . . . , K} , we havePK
j=1esπ(j)=PK
j=1esj. Therefore,α′
k=απ(k).
Now, the head’s resulting output under the permuted attention is z′=PK
k=1α′
kvπ(k) =PK
k=1απ(k)vπ(k). Without loss of generality, reindex the sum by setting j=π(k) . Because
πis a bijection,k7→jpermutes the index set{1, . . . , K}, resulting inz′=PK
j=1αjvj=z.
Thuszis unchanged by the permutation π. The result holds for any permutation, so the MHA’s
per-head output is order invariant.
14

Preprint
A.2 BENCHMARKING ONMATHREASONING
In this section, we investigate whether MHA-RAG can improve performance on mathematical
reasoning tasks. Specifically, we evaluate onInequality Math[train/dev split: 1252/100], an Olympiad-
level benchmark that requires proving bounds preserving inequalities and establishing relations
between algebraic expressions (Sheng et al., 2025). For retrieval, we compute dense representations
of question–exemplar pairs usingQwen3-Embedding-8B(Zhang et al., 2025) and rank candidates by
cosine similarity.
We report results primarily onLlama3.2-3B-Instruct, since models in the Qwen3 family already
achieve high off-the-shelf accuracy (e.g.,Qwen3-4Bat 65%), where adding retrieved exemplars
led to performance drops. This observation is consistent with Sheng et al. (2025), who found that
only certain model families benefit from in-domain exemplars. A plausible explanation is that
high-performing models may have already been exposed to mathematical reasoning tasks during
pre-training, reducing the marginal utility of retrieval.
IneqMathOff-The-Shelf RAG xRAG xRAG-K MHA-RAG
Llama3.2-3B-Instruct
Bound Accuracy 10.0 10.0 4.0 8.0 14.0
Relation Accuracy 22.00 24.0 24.0 24.0 22.0
Final Accuracy 16.00 17.0 14.0 16.018.0
Table 5: Baseline comparison onIneqMathrequiring step-by-step reasoning (with K= 2 ). Accuracy
is measured as an exact match of derived numerical values with ground-truth. Hperparameters are
tuned via sweeps: MHA-RAG ( H∈ {1,2,4} ).Boldindicates the best performance. The model is
trained for 1 epoch with a learning rate of3e−4.
From Table 5, we observe that MHA-RAG yields a slight improvement over both off-the-shelf and
RAG, whereas training with xRAG and xRAG-K results in a performance drop. A possible explana-
tion is that theInequality Mathdataset contains multiple problems that rely on the same theorems
or follow similar reasoning steps. If retrieval fails to surface such structurally related exemplars,
the model cannot fully benefit from exemplar conditioning. Future work could therefore focus on
improving retrieval quality—e.g., by incorporating reasoning-aware similarity metrics—which may
allow MHA-RAG to better exploit shared problem structure and yield stronger overall gains.
15

Preprint
A.3 DOESADDINGEXEMPLARSBACKHELP?
BenchmarksK=5 K=10
c=0 c=1 c=5 c=0 c=1 c=5
Qwen3-4B
BACE 76.27 75.27 79.03 52.42 76.2779.19
BBBP 82.49 68.5688.14 80.43 70.01 87.5
ClinTox 96.32 76.63 94.55 89.44 70.7299.04
PubMedQA 76.59 70.8783.02 66.07 72.88 82.92
Llama3.2-3B-Instruct
BACE 66.1679.96 78.82 54.73 74.23 77.75
BBBP 87.61 88.9989.74 86.80 85.44 86.94
ClinTox 94.44 88.9798.40 89.33 94.49 88.97
PubMedQA 76.34 76.56 83.17 72.50 76.4583.37
Table 6: Effect of re-inserting top- cexemplars in textual form (up to c= 5 , given a fixed inference
budget) into the in-context prompt while still using top- Kfor soft-prompt computation in MHA-RAG.
TheK= 5 ,c= 0 column corresponds to the MHA-RAG column of Table 1.Boldindicates the best
effective accuracy. Underlined indicates the value of cthat achieves the best effective accuracy for a
givenK.
To investigate whether performance can be further improved, we augment MHA-RAG’s soft prompts
with a small number of exemplars directly included in the model’s context at inference. Specifically,
we add the top- cretrieved exemplars alongside the soft prompts and study the effect across different
Kvalues.
As shown in Table 6, effective accuracy improves consistently with increasing c, with the best results
achieved at c= 5 for both K= 5 andK= 10 . The FLOPs analysis in Figure 4 reveals that increasing
Kfrom 5 to 10 incurs only a minor cost, while increasing cfrom 0 to 5 leads to a logarithmic rise in
FLOPs. These results suggest that ccan serve as a tunable knob for balancing accuracy gains against
inference cost, allowing practitioners to adapt MHA-RAG to different computational budgets.
Figure 4: Total FLOPs for inference with encoder ChemBERTa-10M-MTR and foundation model
Qwen3-4B using an increasing number of exemplars cin the context, given K= 5 andK= 10 as
input to create a soft prompt.
A.4 MHA-RAG: ABLATION ON NUMBER OF HEADS
16

Preprint
(a)K= 1,Llama3.2-3B-Instruct
 (b)K= 5,Llama3.2-3B-Instruct
Figure 5: Varying number of heads in MHA-RAG. Performance averaged across 3 seeded runs.
A.5 TRAINABLEPARAMETERS
Training Methods Qwen3-0.6B Qwen3-4B Llama3.2-3B-Instruct
Retrieval-Based Baselines
xRAG 1.59M 2.19M 2.38M
xRAG-K1.59M 2.19M 2.38M
MHA-RAG(m= 1)1.38M 4.16M 4.75M
MHA-RAG(m= 2)3.57M 7.11M 8.3M
MHA-RAG(m= 4)5.93M 13.03M 15.39M
PEFT Baselines
PT(m= 10)10.24K 25.6K 30.72K
IDPG(m= 1)1.59M 2.19M 2.38M
LoRA(r= 64)40.37M 132.12M 97.26M
Table 7: Number of trainable parameters when a given foundation model is updated with a spe-
cific baseline approach. For retrieval-based baselines, we report overall trainable parameters with
ChemBERTa-2-10M-MTRas the Encoder.
17