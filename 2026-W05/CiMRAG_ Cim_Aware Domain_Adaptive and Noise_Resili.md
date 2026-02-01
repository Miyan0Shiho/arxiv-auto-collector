# CiMRAG: Cim-Aware Domain-Adaptive and Noise-Resilient Retrieval-Augmented Generation for Edge-Based LLMs

**Authors**: Shih-Hsuan Chiu, Ming-Syan Chen

**Published**: 2026-01-27 20:30:59

**PDF URL**: [https://arxiv.org/pdf/2601.20041v1](https://arxiv.org/pdf/2601.20041v1)

## Abstract
Personalized virtual assistants powered by large language models (LLMs) on edge devices are attracting growing attention, with Retrieval-Augmented Generation (RAG) emerging as a key method for personalization by retrieving relevant profile data and generating tailored responses. However, deploying RAG on edge devices faces efficiency hurdles due to the rapid growth of profile data, such as user-LLM interactions and recent updates. While Computing-in-Memory (CiM) architectures mitigate this bottleneck by eliminating data movement between memory and processing units via in-situ operations, they are susceptible to environmental noise that can degrade retrieval precision. This poses a critical issue in dynamic, multi-domain edge-based scenarios (e.g., travel, medicine, and law) where both accuracy and adaptability are paramount. To address these challenges, we propose Task-Oriented Noise-resilient Embedding Learning (TONEL), a framework that improves noise robustness and domain adaptability for RAG in noisy edge environments. TONEL employs a noise-aware projection model to learn task-specific embeddings compatible with CiM hardware constraints, enabling accurate retrieval under noisy conditions. Extensive experiments conducted on personalization benchmarks demonstrate the effectiveness and practicality of our methods relative to strong baselines, especially in task-specific noisy scenarios.

## Full Text


<!-- PDF content starts -->

CIMRAG: CIM-AWARE DOMAIN-ADAPTIVE AND NOISE-RESILIENT
RETRIEVAL-AUGMENTED GENERATION FOR EDGE-BASED LLMS
Shih-Hsuan Chiu and Ming-Syan Chen
National Taiwan University, Taipei, Taiwan
shchiu@arbor.ee.ntu.edu.tw, mschen@ntu.edu.tw
ABSTRACT
Personalized virtual assistants powered by large language mod-
els (LLMs) on edge devices are attracting growing attention, with
Retrieval-Augmented Generation (RAG) emerging as a key method
for personalization by retrieving relevant profile data and generating
tailored responses. However, deploying RAG on edge devices faces
efficiency hurdles due to the rapid growth of profile data, such as
user-LLM interactions and recent updates. While Computing-in-
Memory (CiM) architectures mitigate this bottleneck by eliminating
data movement between memory and processing units via in-situ op-
erations, they are susceptible to environmental noise that can degrade
retrieval precision. This poses a critical issue in dynamic, multi-
domain edge-based scenarios (e.g., travel, medicine, and law) where
both accuracy and adaptability are paramount. To address these
challenges, we proposeT ask-O riented N oise-resilient E mbedding
Learning(TONEL), a framework that improves noise robustness
and domain adaptability for RAG in noisy edge environments.
TONEL employs a noise-aware projection model to learn task-
specific embeddings compatible with CiM hardware constraints,
enabling accurate retrieval under noisy conditions. Extensive ex-
periments conducted on personalization benchmarks demonstrate
the effectiveness and practicality of our methods relative to strong
baselines, especially in task-specific noisy scenarios.
Index Terms—Retrieval-Augmented Generation, Edge LLMs,
Computing-in-Memory, Noise-Resilience, Domain Adaptability
1. INTRODUCTION
Large language models (LLMs) have become essential across many
applications due to their strong reasoning capabilities [1,2]. To meet
the growing demands for personalized interactions, deploying these
LLMs on edge devices (what we term ”edge LLMs”) has gained
significant traction [3–5]. For effective personalization, edge LLMs
must adapt to profile data such as user-LLM interaction or recent
updates. However, as with cloud-based LLMs, edge LLMs primarily
rely on fine-tuning model parameters, which is impractical for edge
devices due to their limited computing power and memory [6, 7].
Retrieval-Augmented Generation (RAG) has emerged as the de
facto alternative, enabling personalization of edge LLMs without
fine-tuning [8, 9]. RAG retrieves the most semantically relevant
documents to the input query viamaximum inner product search
(MIPS)[9] and feeds them, together with the query, into an LLM
to generate personalized responses. Of note, all documents from
profile data are first converted into numerical embedding vectors by
an embedding model and stored as a matrix within an RAG system
used for MIPS, as illustrated in Figure 1.
Despite its efficiency on edge devices, RAG faces two key chal-
lenges for real-time interactions. First, as user data (e.g., conver-
sation history with LLMs) grows beyond RAM, reliance on slower
storage (e.g., HDDs or SSDs) increases data transfer latency [11].
Fig. 1. A schematic depiction of RAG for edge LLMs with CiM
devices; the workflow is inspired by [10]. CiM employs MIPS to
retrieve top-ranked documents, which are combined with the user
query to enable the LLM to generate personalized responses. This
paper focuses on enhancing thetext embedding modelto produce
task-aware, noise-resilient CiM-friendly embeddings for MIPSin
domain-specific edge environments.
Second, the efficiency of MIPS-based retrieval degrades with larger
datasets, making RAG impractical for extensive user data [9]. For-
tunately, a seminal work by Qin et al. [10] leveraged computing-in-
memory (CiM) architectures [12] to accelerate RAG by speeding up
matrix-vector multiplication, the core operation in MIPS. As shown
in Figure 1, CiM performs computations directly in memory, reduc-
ing data movement (i.e., between memory and processing units) and
boosting efficiency [13, 14].
Nevertheless, CiM arrays often rely on non-volatile memory
(NVM), which is susceptible to environmental noise (e.g., temper-
ature, humidity, and aging), potentially corrupting stored document
embeddings and impairing retrieval (MIPS) performance [15]. To
address this, Qin [10] proposes a noise-aware contrastive learning
method that trains an embedding model to generate noise-resilient
embeddings by pulling similar data closer and pushing dissimilar
data apart, showing promising results. However, in practice, edge
LLMs are deployed across diverse domains such as travel, medicine,
and law. In such environments, environmental noise does not merely
introduce random errors to the embeddings stored in CiM but signif-
icantly degrades the retrieval precision required for domain-specific
tasks. Consequently, optimizing MIPS retrieval under noise in
such dynamic, multi-domain scenarios remains a critical yet un-
derexplored challenge. Moreover, real-world users often engage in
multi-turn interactions with edge LLMs, leading to rapidly growing
and topic-diverse profile documents, an issue not addressed in Qin’s
method [10].
To address the above challenges, we identify two key issues: (i)
MIPS robustness under noise: Environmental noise (e.g., CiM varia-
tions) can distort matrix-vector similarity scores, necessitating a textarXiv:2601.20041v1  [cs.LG]  27 Jan 2026

Fig. 2. The overview of the proposedTONELframework. It focuses on optimizing theprojection modelto producetask-aware, noise-resilient
CiM-friendly vectors(in gold) suitable for MIPS in domain-specific edge environments.
embedding model that produces task-aware, noise-resilient embed-
dings. (ii)Label-free task adaptation: In real-world scenarios, users
interact with edge LLMs across diverse domains, producing a large
volume of unlabeled documents. As manual labeling is impractical,
the embedding model must adapt effectively in a label-free manner.
To this end, we proposeTONEL(T ask-O riented N oise-resilient
Embedding L earning), a label-free framework that enables noise-
robust and domain-adaptive MIPS retrieval on CiM hardware.
TONEL incorporates two core components:(1) To ensure MIPS
robustness under noise, we develop anoise-aware task-oriented
optimization strategy (NATO)that trains a projection model to gen-
erate task-specific embeddings compatible with CiM hardware. It
uses pseudo topic labels under noisy conditions to produce noise-
resilient representations.(2) To enable label-free task adaptation,
we present apseudo-label generation mechanism (PGM)that guides
the text embedding model to learn domain-adaptive, noise-resilient
representations without manual annotation by generating latent task
labels for NATO. Overall, TONEL is designed toenhance text em-
bedding models to generate task-specific, noise-resilient embeddings
tailored for MIPS on CiM hardware.
2. PROBLEM DEFINITION
2.1. RAG on CiM devices with MIPS-based retrieval
An RAG system personalizes edge-based LLMs by retrieving rele-
vant context from user profile data (viewed as a collection of docu-
ments). It comprises a text embedding model, a retriever, and a gen-
erator. Assume a user hasNdocuments (e.g., conversation history),
each documentD iis converted into ad-dimensional vectorv(D i)
via embedding modelEmb(.). These embeddings are assembled
into a matrixDand stored in a CiM array, as shown in Figure 1.
v(D i) = Emb(D i)∈Rd,fori= 1, . . . , N(1)
D= [v(D 1),v(D 2), . . . ,v(D N)]∈Rd×N(2)
During execution, the user query is also encoded into an em-
beddingv(Q)using the same model. The retriever then performs
maximum inner product search (MIPS)[9] by computing similarity
scores via matrix-vector multiplication with stored document em-
beddings:
s=D⊤v(Q)∈RN(3)Finally, the RAG system concatenates the top-scoring (Top-K)
retrieved documents with the query to form a prompt, which is then
fed into the generator, typically an LLM, to produce a contextual
response ˆYfor the user:
Prompt= [D i]i∈TopK(s) ∥Q(4)
ˆY= LLM(Prompt)(5)
Notably, document embeddings would be stored in a CiM ar-
ray (Figure 1, middle), which is vulnerable to environmental noise,
potentially degrading retrieval (MIPS) accuracy. Thus, this work fo-
cuses onenhancing the text embedding model to produce embed-
dings that are robust to noise and adaptable across domains for
MIPS in edge-based environments.
3. PROPOSED METHOD
We proposeTONEL(T ask-O riented N oise-resilient E mbedding
Learning), a framework designed to enhance text embedding models
that can generate task-aware, noise-resilient CiM-friendly embed-
dings for MIPS. As depicted in Figure 2, TONEL comprises two key
components: (1) a noise-aware task-oriented optimization strategy
(NATO), and (2) a pseudo-label generation mechanism (PGM).
3.1. Noise-aware Task-oriented Optimization Strategy
To tackle the challenge of MIPS robustness under noise, we develop
anoise-aware task-oriented optimization strategy (NATO)that trains
a projection model to produce task-aware, noise-resilient embed-
dings while adhering to CiM hardware constraints. As shown in
Figure 2, the training process first encodes allNdocuments using
a pretrained encoderEnc(.)(inherent to the LLM) to obtain em-
beddings with their original dimensionality and bit precision (e.g.,
384-dimensional vectors in 32-bit floating-point format):
e(Di) = Enc(D i)∈R384
FP32 (6)
Since the document embeddings would be stored in a CiM ar-
chitecture that is implemented as a ”crossbar array” with typically
fixed dimensions (e.g., 64×64) and bit precision (e.g., 8-bit inte-
gers) [10, 16], we employ a projection modelProj(.)to map them
to 64 dimensions and apply simulated INT8 quantization via fake

Fig. 3. A schematic depiction of PGM for generating pseudo task
labels used in TONEL.
quantization and rounding [17], generating CiM-friendly vectors that
conform to crossbar array constraints:
e(Di)′= Proj(e(D i))∈R64
FP32 (7)
e(Di)′
q= clampe(Di)′
s
,−2b−1,2b−1−1
(8)
s=|max(e(D i)′)|
2b−1−1(9)
˜e(Di) =s·e(D i)′
q (10)
whereb= 8(i.e., quantizing original vectors to 8-bit integers),⌊·⌉
denotes the round-to-nearest (RTN) operator,clamp(x, l, h)clamps
xto the range[l, h], and the reconstruction vector ˜e(Di)is obtained
using the uniform scaling factors.
To account for hardware-induced perturbations, noise variations
measured from real CiM devices are injected into the embeddings.
The resulting noisy embeddings are fed into a task predictorPred(.),
which outputs a prediction score for each task. For model training,
we employ the Cross-Entropy (CE) loss [18] and adapt it into the
CiM-aware Cross-Entropy (CiMCE) lossto incorporate the above
factors, serving as our objective function. Given a set of documents
D={D i}N
i=1={D 1, D2. . . D N}, each assigned to a class among
Cclasses, the CiMCE loss is formulated as:
LCiMCE =−1
NNX
i=1CX
c=1ˆyi,clogP 
c|Pred( ˜e(Di) +η)
(11)
Note thatη∼ N(0, σ v)refers to the CiM variation noise,
modeled as Gaussian with zero mean and a standard deviationσ v
specific to each value [19, 20]. Theˆy i,crepresents a pseudo task
label, indicating whether documentD ibelongs to classc, generated
by the PGM module (Section 3.2). The model parameters are jointly
optimized with task-specific information and noise perturbations,
ensuring the embeddings remain discriminative under varying noise
conditions. At inference time, the projection model produces task-
aware, noise-resilient CiM-friendly embeddings (64 dimensions
with 8-bit integers) for queries and documents, facilitating more
accurate MIPS-based retrieval for domain-specific tasks.
3.2. Pseudo-Label Generation Mechanism
In real-world edge-based scenarios, such as travel or medical tasks,
user-LLM interactions often span multiple topics and contexts over
time, with rapidly growing profile data forming implicit task-level
information. Incorporating these task-level cues into the embed-
dings of both documents and queries can significantly improve re-
trieval (i.e., MIPS) performance. To flesh out this notion, we present
aPseudo-Label Generation Mechanism (PGM)for TONEL (cf. Fig-
ure 2) that supports its training process by alleviating the need forTable 1. Device variations from different real CiM devices [10].
NameDevice Variationsσ v
L0 L1 L2 L3
RRAM 1(Device-1) 0.0100 0.0100 0.0100 0.0100
FeFET 2(Device-2) 0.0067 0.0135 0.0135 0.0067
FeFET 3(Device-3) 0.0049 0.0146 0.0146 0.0049
RRAM 4(Device-4) 0.0038 0.0151 0.0151 0.0038
manually labeling the rapidly growing profile data with task-level in-
formation. More specifically, as depicted in Figure 3, PGM assigns
each document to one of the predefinedKgroups by clustering their
embeddings from the pretrained encoderEnc(.)using an unsuper-
vised clustering algorithm (e.g., theK-means algorithm [21]). Each
documentD iis thus mapped to a specific group in a latent space,
serving as its “pseudo” classˆy i. This PGM module enables TONEL
to train the projection model with task-specific cues by automatically
generating latent topics as pseudo task labels, thereby removing the
need for manual annotation.
4. EMPIRICAL EXPERIMENTS
4.1. Experimental Setup
Benchmark Datasets.We evaluate TONEL on two personalization
datasets from the LaMP benchmark: Movie Tagging (Movie) and
Product Rating (Rating) [22]. In both datasets, each user has pro-
file data consisting of textual history and corresponding labels. The
Movie and Rating tasks are formulated as 15-class and 5-class clas-
sification problems, respectively.
Comparative Methods.We compare TONEL with two strong base-
lines (i.e., PCA and RoCR). PCA [23] is a classical, widely used
dimensionality-reduction method, while RoCR [10] is a seminal ap-
proach that enables RAG on CiM architectures via noise-aware rep-
resentation learning for documents.
Evaluation Metrics.Our primary metric is top-1 accuracy (Acc@1)
of MIPS: the fraction of queries whose top-1 results from noisy
CiM-friendly embeddings match the Oracle results from original
full-precision embeddings without noise. We also present two met-
rics, Precision@5 (Prec@5), the ratio of oracle-relevant documents
in the top-5, and nDCG@5, a position-aware score with respect to
the oracle ranking. For downstream evaluation, we prepend the top-5
retrieved documents to the query and apply two representative edge-
friendly LLMs (e.g., Gemma-2B [24] and Llama-3.2-3B [25]) as
generators, reporting classification Accuracy (Acc) and F1 score.
Noise Settings.Based on the device variations observed in real CiM
devices shown in Table 1, the noise injection for document embed-
dings during testing follows [10, 20]:
emb(D i)d×p
σ= 
e′·L0+e′·L1+e′·L2+e′·L3
·σ(12)
wheree′= emb(D i)d×pis the embedding generated by the projec-
tion model, withdandprespectively denoting the reduced dimen-
sionality and bit precision (e.g.,d= 64,p= 8), andσrepresenting
the standard deviation of the injected Gaussian noise.
4.2. Experimental Results
In the first set of experiments, we compare TONEL against two
strong baselines, PCA [23] and RoCR [10], with Oracle and Random
results listed for reference in Table 2. Evaluations are conducted on
two datasets (i.e., Movie and Rating) [22]. Notably, TONEL sup-
ports two training modes: the first leverages pseudo task labels from
the PGM module (cf. Section 3.2), referred to as TONEL (w/ PL),

Table 2. The MIPS (top-1 accuracy) results obtained by TONEL
in comparison to that of baselines on the Movie and Rating datasets
under noise variations from CiM Device-2 with different proportions
of noisy documents.
Method Noise (%) Movie Rating
Oracle - 1 1
Random - 0.00068 0.00083
PCA [23]Clean 0.3478 0.0432
50% 0.3078 0.0383
100% 0.2138 0.0346
RoCR [10]Clean 0.3826 0.0517
50% 0.3537 0.0485
100% 0.3295 0.0453
TONEL (w/ PL)Clean 0.4313 0.0638
50% 0.4067 0.0566
100% 0.3883 0.0584
TONEL (w/ TL)Clean 0.7667 0.2387
50% 0.7298 0.2336
100% 0.7034 0.2452
and the second assumes access to ground-truth task labels, denoted
as TONEL (w/ TL). To better assess our methods, we evaluate each
approach under different proportions of noisy documents (Clean,
50%, and 100%). Two key observations can be drawn from Table
2, which presents MIPS top-1 accuracy under noise variations from
Device-2 (cf. Table 1). First, our methods, TONEL (w/ PL) and
TONEL (w/ TL), consistently outperform the baselines across both
datasets and noise levels. Second, under 100% noise (i.e., all docu-
ment embeddings perturbed), TONEL (w/ PL) and TONEL (w/ TL)
outperform RoCR by 6.2% and 37.4%, respectively, on the Movie
dataset. These results indeed demonstrate the efficacy of TONEL to
produce task-aware, noise-resilient embeddings tailored for MIPS.
In the second set of experiments, we evaluate our methods on
the Movie dataset where all documents are perturbed (100%) un-
der noise variations induced by different real CiM devices (cf. Ta-
ble 1). We exhibit three evaluation metrics (Acc@1, Precision@5,
and nDCG@5) to comprehensively assess the robustness and per-
formance of TONEL methods in comparison to the baselines. The
corresponding results are presented in Table 3. Inspection of Table
3 reveals three noteworthy points. First, TONEL (w/ PL) consis-
tently outperforms PCA and RoCR on all CiM devices and metrics,
demonstrating its robustness to CiM-induced noise without requir-
ing manual task labels. Second, TONEL (w/ PL) achieves average
relative gains of 12.6% in Precision@5 and 10.0% in nDCG@5 over
RoCR, indicating more accurate top-5 retrieval and ranking under
noise conditions. Third, TONEL (w/ TL) delivers substantial im-
provements over all baselines and approaches Oracle performance,
reflecting the full potential of TONEL with task supervision.
To further validate the practicality and feasibility of TONEL, we
evaluate its downstream classification accuracy on the Movie Tag-
ging dataset, a 15-category task (i.e., given a query of movie de-
scription, the LLM predicts a topic tag such as comedy and action).
We select two representative edge-friendly medium-size LLMs (i.e.,
Gemma-2B [24] and Llama-3.2-3B [25]) as generators to compare
the performance of TONEL with Baseline and RoCR. Gemma-2B
is one of the earliest open models by Google, with 4.95GB model
weights. Llama-3.2-3B is a new SOTA open model by Meta, with
6.85GB model weights. The Baseline predicts a topic tag from the
query alone, while RoCR and TONEL incorporate additional context
by retrieving relevant profile documents. The corresponding results
shown in Table 4 yield three notable observations. First, augment-
ing the query with longer context (e.g., top-5 retrieved documentsTable 3. The Acc@1, Precision@5, nDCG@5 results obtained by
TONEL in comparison to that of baselines on the Movie dataset un-
der noise variations from different CiM devices.
Method Metric Device-1 Device-2 Device-3 Device-4
Oracle - 1 1 1 1
PCA [23]Acc@1 0.2701 0.2138 0.1770 0.2092
Prec@5 0.4839 0.4161 0.3621 0.4218
nDCG@5 0.3833 0.3175 0.2748 0.3220
RoCR [10]Acc@1 0.3531 0.3295 0.2713 0.3195
Prec@5 0.6011 0.5736 0.5011 0.5218
nDCG@5 0.5040 0.4718 0.3923 0.4291
TONEL (w/ PL)Acc@1 0.3713 0.3883 0.3241 0.3368
Prec@5 0.6471 0.6149 0.5977 0.6069
nDCG@5 0.5154 0.4979 0.4731 0.4781
TONEL (w/ TL)Acc@1 0.7701 0.7034 0.6885 0.6897
Prec@5 0.8920 0.8701 0.8529 0.8678
nDCG@5 0.8340 0.7920 0.7795 0.7856
Table 4. The classification Accuracy and F1 results obtained by
TONEL in comparison to that of baselines on the Movie dataset un-
der noise variations from Device-2, using two edge-friendly LLMs
(i.e., Gemma-2B and Llama-3.2-3B) as generators.
Method Gemma-2B Llama-3.2-3B
Acc (↑) F1 (↑) Acc (↑) F1 (↑)
Baseline 0.1460 0.0933 0.1084 0.0498
RoCR [10] 0.3412 0.3107 0.3258 0.2893
TONEL (w/ PL) 0.4104 0.3802 0.3974 0.3438
TONEL (w/ TL) 0.5116 0.4780 0.5010 0.4847
in this evaluation) via RAG improves LLM prediction accuracy and
personalization compared to the Baseline, without modifying model
parameters. Second, TONEL (w/ PL) outperforms RoCR, achieving
relative improvements of 20.3% in accuracy and 22.4% in F1 score
on Gemma-2B, confirming its effectiveness in a label-free setting.
Third, TONEL (w/ TL) further improves over TONEL (w/ PL) by
26.1% in accuracy and 41.0% in F1 score on Llama-3.2-3B, reflect-
ing its full potential when ground-truth task labels are available.
5. CONCLUSION
This work presents TONEL, a novel label-free framework that gener-
ates task-specific, noise-resilient embeddings compatible with CiM
hardware for MIPS-based retrieval in RAG systems, enabling better
personalization of LLMs in domain-specific edge environments.
A series of empirical evaluations on personalization benchmarks
demonstrates its practical utility over baselines and a representative
RAG approach on CiM devices. Future directions include extending
TONEL to better adapt to dynamic user profiles by developing ef-
ficient clustering algorithms co-designed with specialized hardware
architectures, tailored for real-world edge-based LLM applications.
6. ACKNOWLEDGMENT
This research is supported in part by the National Science and
Technology Council (NSTC), Taiwan, under Grant Number NSTC
114-2223-E-002-009, by the Ministry of Education, Taiwan, through
the Higher Education Sprout Project—The Featured Area Research
Center Program, and by the NTU–Delta Electronics Innovation Re-
search Funding Project. Any findings and implications in the paper
do not necessarily reflect those of the sponsors.

7. REFERENCES
[1] Shervin Minaee, Tomas Mikolov, Narjes Nikzad, Meysam
Chenaghlu, Richard Socher, Xavier Amatriain, and Jianfeng
Gao. Large language models: A survey.arXiv preprint
arXiv:2402.06196, 2024.
[2] Zhiheng Xi, Wenxiang Chen, Xin Guo, Wei He, Yiwen Ding,
Boyang Hong, Ming Zhang, Junzhe Wang, Senjie Jin, Enyu
Zhou, et al. The rise and potential of large language model
based agents: A survey.Science China Information Sciences,
68(2):121101, 2025.
[3] Guanqiao Qu, Qiyuan Chen, Wei Wei, Zheng Lin, Xianhao
Chen, and Kaibin Huang. Mobile edge intelligence for large
language models: A contemporary survey.IEEE Communica-
tions Surveys & Tutorials, 2025.
[4] Yue Zheng, Yuhao Chen, Bin Qian, Xiufang Shi, Yuanchao
Shu, and Jiming Chen. A review on edge large language mod-
els: Design, execution, and applications.ACM Computing Sur-
veys, 57(8):1–35, 2025.
[5] Fali Wang, Zhiwei Zhang, Xianren Zhang, Zongyu Wu,
Tzuhao Mo, Qiuhao Lu, Wanjing Wang, Rui Li, Junjie Xu,
Xianfeng Tang, et al. A comprehensive survey of small lan-
guage models in the era of large language models: Techniques,
enhancements, applications, collaboration with llms, and trust-
worthiness.arXiv preprint arXiv:2411.03350, 2024.
[6] Ruiyang Qin, Dancheng Liu, Chenhui Xu, Zheyu Yan, Zhaox-
uan Tan, Zhenge Jia, Amir Nassereldine, Jiajie Li, Meng Jiang,
Ahmed Abbasi, et al. Empirical guidelines for deploying llms
onto resource-constrained edge devices.ACM Transactions on
Design Automation of Electronic Systems, 2024.
[7] Jiajun Xu, Zhiyuan Li, Wei Chen, Qun Wang, Xin Gao, Qi Cai,
and Ziyuan Ling. On-device language models: A comprehen-
sive review.arXiv preprint arXiv:2409.00088, 2024.
[8] Wenqi Fan, Yujuan Ding, Liangbo Ning, Shijie Wang,
Hengyun Li, Dawei Yin, Tat-Seng Chua, and Qing Li. A sur-
vey on rag meeting llms: Towards retrieval-augmented large
language models. InProceedings of the 30th ACM SIGKDD
conference on knowledge discovery and data mining, pages
6491–6501, 2024.
[9] Patrick Lewis, Ethan Perez, Aleksandra Piktus, Fabio Petroni,
Vladimir Karpukhin, Naman Goyal, Heinrich K ¨uttler, Mike
Lewis, Wen-tau Yih, Tim Rockt ¨aschel, et al. Retrieval-
augmented generation for knowledge-intensive nlp tasks.Ad-
vances in neural information processing systems, 33:9459–
9474, 2020.
[10] Ruiyang Qin, Zheyu Yan, Dewen Zeng, Zhenge Jia, Dancheng
Liu, Jianbo Liu, Ahmed Abbasi, Zhi Zheng, Ningyuan Cao,
Kai Ni, et al. Robust implementation of retrieval-augmented
generation on edge-based computing-in-memory architectures.
InProceedings of the 43rd IEEE/ACM International Confer-
ence on Computer-Aided Design, pages 1–9, 2024.
[11] Yangwook Kang, Yang-suk Kee, Ethan L Miller, and Chanik
Park. Enabling cost-effective data processing with smart ssd.
In2013 IEEE 29th symposium on mass storage systems and
technologies (MSST), pages 1–12. IEEE, 2013.
[12] Ali BanaGozar, Kanishkan Vadivel, Sander Stuijk, Henk Cor-
poraal, Stephan Wong, Muath Abu Lebdeh, Jintao Yu, and Said
Hamdioui. Cim-sim: computation in memory simuiator. In
Proceedings of the 22nd International Workshop on Software
and Compilers for Embedded Systems, pages 1–4, 2019.[13] Vivienne Sze, Yu-Hsin Chen, Tien-Ju Yang, and Joel S Emer.
Efficient processing of deep neural networks: A tutorial and
survey.Proceedings of the IEEE, 105(12):2295–2329, 2017.
[14] Xiaochen Peng, Shanshi Huang, Yandong Luo, Xiaoyu Sun,
and Shimeng Yu. Dnn+ neurosim: An end-to-end benchmark-
ing framework for compute-in-memory accelerators with ver-
satile device technologies. In2019 IEEE international electron
devices meeting (IEDM), pages 32–5. IEEE, 2019.
[15] Zheyu Yan, Xiaobo Sharon Hu, and Yiyu Shi. Compute-in-
memory-based neural network accelerators for safety-critical
systems: Worst-case scenarios and protections.IEEE Trans-
actions on Computer-Aided Design of Integrated Circuits and
Systems, 43(8):2452–2464, 2024.
[16] Weiwen Jiang, Qiuwen Lou, Zheyu Yan, Lei Yang, Jing-
tong Hu, Xiaobo Sharon Hu, and Yiyu Shi. Device-circuit-
architecture co-exploration for computing-in-memory neural
accelerators.IEEE Transactions on Computers, 70(4):595–
605, 2020.
[17] Benoit Jacob, Skirmantas Kligys, Bo Chen, Menglong Zhu,
Matthew Tang, Andrew Howard, Hartwig Adam, and Dmitry
Kalenichenko. Quantization and training of neural networks
for efficient integer-arithmetic-only inference. InProceedings
of the IEEE conference on computer vision and pattern recog-
nition, pages 2704–2713, 2018.
[18] Anqi Mao, Mehryar Mohri, and Yutao Zhong. Cross-entropy
loss functions: Theoretical analysis and applications. InIn-
ternational conference on Machine learning, pages 23803–
23828. pmlr, 2023.
[19] Zheyu Yan, Xiaobo Sharon Hu, and Yiyu Shi. Swim: Selective
write-verify for computing-in-memory neural accelerators. In
Proceedings of the 59th ACM/IEEE Design Automation Con-
ference, pages 277–282, 2022.
[20] Zheyu Yan, Da-Cheng Juan, Xiaobo Sharon Hu, and Yiyu Shi.
Uncertainty modeling of emerging device based computing-in-
memory neural accelerators with application to neural architec-
ture search. InProceedings of the 26th Asia and South Pacific
Design Automation Conference, pages 859–864, 2021.
[21] James B McQueen. Some methods of classification and analy-
sis of multivariate observations. InProc. of 5th Berkeley Sym-
posium on Math. Stat. and Prob., pages 281–297, 1967.
[22] Alireza Salemi, Sheshera Mysore, Michael Bendersky, and
Hamed Zamani. Lamp: When large language models meet
personalization.arXiv preprint arXiv:2304.11406, 2023.
[23] Christopher M Bishop and Nasser M Nasrabadi.Pattern recog-
nition and machine learning, volume 4. Springer, 2006.
[24] Gemma Team, Thomas Mesnard, Cassidy Hardin, Robert
Dadashi, Surya Bhupatiraju, Shreya Pathak, Laurent Sifre,
Morgane Rivi `ere, Mihir Sanjay Kale, Juliette Love, et al.
Gemma: Open models based on gemini research and technol-
ogy.arXiv preprint arXiv:2403.08295, 2024.
[25] Abhimanyu Dubey, Abhinav Jauhri, Abhinav Pandey, Ab-
hishek Kadian, Ahmad Al-Dahle, Aiesha Letman, Akhil
Mathur, Alan Schelten, Amy Yang, Angela Fan, et al. The
llama 3 herd of models.arXiv e-prints, pages arXiv–2407,
2024.