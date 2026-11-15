# Edge-Native Multimodal RAG: Hardware-Aware Quantization and Architectural Optimization for Consumer GPUs

**Authors**: Zahraa Selim, Menna Hamed, Wesam Ahmed, Sohyla Said, Sara Basheer, Rami Zewail

**Published**: 2026-11-11 02:26:07

**PDF URL**: [https://arxiv.org/pdf/2611.14924v1](https://arxiv.org/pdf/2611.14924v1)

## Abstract
Deployment of retrieval-augmented generation (RAG) for Edge-AI is constrained by memory and bandwidth limitations. A standard 7B parameter model consumes 14 GB Video Random Access Memory (VRAM) in half-precision (FP16), precluding co-location of vector indices and Key-Value (KV) caches. This paper presents an engineering integration and empirical evaluation of a multimodal RAG framework optimized for consumer-grade Turing hardware (e.g., NVIDIA Tesla T4). Our system-level analysis identifies a critical trade-off: while generative pre-trained transformer quantization (GPTQ) offers superior throughput (51.6 tok/s), it exhibits significant context instability at 2048+ tokens on Turing architectures. Conversely, NormalFloat 4-bit (NF4) maintains robust long-context performance with viable latency (81 ms/tok). By coupling NF4 with a decoupled Florence-2 vision integration, we achieve visual reasoning with only 430 MB VRAM overhead. Furthermore, empirical observations of prompt-induced search space pruning show that structured prompting reduces latency by 47% without altering computational intensity (1.88 GFLOPs/token). The integrated architecture maintains an 8.37 GB footprint, establishing a viable RAG pipeline for 12-16 GB GPUs. Finally, structured few-shot prompting effectively compensates for quantization-induced reasoning loss, restoring mathematical accuracy from 35.0% to 92.0% across 200 trials.

## Full Text


<!-- PDF content starts -->

Edge-Native Multimodal RAG: Hardware-Aware
Quantization and Architectural Optimization for
Consumer GPUs
Zahraa Selim, Menna Hamed, Wesam Ahmed, Sohyla Said, Sara Basheer, and Rami Zewail
Computer Science and Engineering Department
Egypt-Japan University of Science and Technology (E-JUST)
Alexandria, Egypt
{zahraa.selim, menna.hamed, wesam.elraghy, sohyla.abdelwahed, sara.basheer, rami.zewail}@ejust.edu.eg
Abstract—Deployment of retrieval-augmented generation
(RAG) for Edge-AI is constrained by memory and bandwidth
limitations. A standard 7B parameter model consumes 14 GB
Video Random Access Memory (VRAM) in half-precision (FP16),
precluding co-location of vector indices and Key-Value (KV)
caches. This paper presents an engineering integration and
empirical evaluation of a multimodal RAG framework optimized
for consumer-grade Turing hardware (e.g., NVIDIA Tesla T4).
Our system-level analysis identifies a critical trade-off: while
generative pre-trained transformer quantization (GPTQ) offers
superior throughput (51.6tok/s), it exhibits significant context
instability at 2048+ tokens on Turing architectures. Conversely,
NormalFloat 4-bit (NF4) maintains robust long-context perfor-
mance with viable latency (81ms/tok). By coupling NF4 with
a decoupled Florence-2 vision integration, we achieve visual
reasoning with only 430 MB VRAM overhead. Furthermore,
empirical observations of prompt-induced search space pruning
show that structured prompting reduces latency by 47% without
altering computational intensity (1.88 GFLOPs/token). The inte-
grated architecture maintains an 8.37 GB footprint, establishing a
viable RAG pipeline for 12–16 GB GPUs. Finally, structured few-
shot prompting effectively compensates for quantization-induced
reasoning loss, restoring mathematical accuracy from 35.0% to
92.0% across 200 trials.
Index Terms—Edge AI, Retrieval-Augmented Generation,
Post-Training Quantization, Multimodal Learning, Prompt En-
gineering.
I. INTRODUCTION
The proliferation of open-weight large language models
(LLMs) has catalyzed a paradigm shift in the deployment of
artificial intelligence. Compact architectures, such as Mistral-
7B [1], demonstrate that advanced capabilities are no longer
the exclusive preserve of hyperscale cloud infrastructure.
However, migrating these capabilities to consumer-grade hard-
ware—typified by the NVIDIA T4 or RTX 3060—exposes
critical resource bottlenecks.
A standard seven-billion parameter model utilizing FP16
precision imposes a static memory footprint of approximately
© 2026 IEEE. Personal use of this material is permitted. Permission from
IEEE must be obtained for all other uses, in any current or future media,
including reprinting/republishing this material for advertising or promotional
purposes, creating new collective works, for resale or redistribution to servers
or lists, or reuse of any copyrighted component of this work in other works.14 GB. This baseline consumption virtually exhausts the mem-
ory budget of a 16-GB device, leaving negligible headroom for
inference. This resource contention is exacerbated in retrieval-
augmented generation (RAG) pipelines [2], where the system
must simultaneously accommodate embedding models (≈300
MB), vector indices (≈200 MB), and context-dependent Key-
Value (KV) caches. A naive implementation inevitably exceeds
the physical VRAM limits of ubiquitous 12–16-GB GPUs,
often demanding an aggregate footprint of 17–19 GB [3].
A. Motivation and Research Gaps
This study is predicated on addressing three persistent de-
ployment lacunae through systems engineering and empirical
validation:
•Architectural Agnosticism in Quantization:Tech-
niques optimized for specific kernels often incur un-
characterized stability penalties when deployed on older
hardware, such as NVIDIA Turing GPUs.
•Monolithic VRAM Bottlenecks:End-to-end multimodal
LLMs drastically exceed consumer VRAM budgets, pre-
cluding their use in local, disconnected RAG environ-
ments.
•Silent Generative Failures:Prevalent evaluation relies
heavily on perplexity, obfuscating how aggressive bit-
width reduction specifically degrades multi-step reason-
ing and faithfulness in retrieval tasks.
B. Contributions
To bridge these gaps, we propose a tripartite system integra-
tion framework. Our primary contributions correlate directly
to the identified gaps:
1)Hardware-Aware Quantization Benchmarking (Ad-
dressing Gap 1):We isolate a critical stability bottleneck
on the NVIDIA Turing architecture. While GPTQ pro-
vides the fastest inference (19.36ms/tok), it exhibits high
variance and context collapse. NF4 serves as the optimal
middle ground, maintaining viable edge performance (81
ms/tok) with superior stability.
arXiv:2611.14924v1  [cs.AI]  11 Nov 2026

2)Decoupled Multimodal Architecture (Addressing Gap
2):We integrate a decoupled visual pipeline using
Florence-2 Base, achieving visual QA capabilities com-
parable to larger models with only a negligible 430 MB
VRAM overhead.
3)Reasoning Restoration and Prompt Efficiency (Ad-
dressing Gap 3):We demonstrate that while quantiza-
tion degrades procedural reasoning, structured few-shot
prompting compensates for this loss, recovering mathe-
matical accuracy from 35.0% to 92.0% across 200 trials.
Concurrently, we document a prompt-induced efficiency
gain, reducing generation latency by 47% via search-
space pruning.
II. RELATEDWORK
Deploying LLMs in resource-constrained edge environ-
ments necessitates a rigorous reduction in computational
complexity [4]. Techniques such as Quantization,Prunning,
and Knowledge Distillation have emerged as a highly viable
strategy for immediate memory relief without necessitating
extensive architectural retraining.
A. Quantization Strategies
Quantization fundamentally alters the arithmetic intensity of
inference by lowering the precision of parameters. Weight-only
strategies compress static model parameters while retaining
high-precision activations. A prominent example is GPTQ [5],
which minimizes reconstruction error by exploiting second-
order Hessian information. Mixed-precision decomposition
techniques have emerged as a solution to address precision
loss, such as LLM.int8() [6] and SpQR [7].
Standard weight-only methods often falter when confronted
with activation outliers. Activation-aware strategies postulate
that preserving the fidelity of these outliers is paramount,
as seen in AWQ [8]. Complementary approaches, such as
SmoothQuant [9], employ mathematical transformations to
smooth the activation landscape. In RAG workflows, the linear
growth of the KV cache constitutes a formidable memory bot-
tleneck, prompting specialized algorithms such as KVQuant
[10] and KIVI [11], though integrating these natively with
consumer hardware APIs remains complex.
B. Pruning and Knowledge Distillation
Pruning methodologies allow for the excision of redundant
weights. Unstructured pruning induces sparsity at the scalar
level (e.g., SparseGPT [12]), while structured pruning removes
coherent architectural blocks (e.g., LLM-Pruner [13]). Knowl-
edge distillation functions to compress the semantic density
of a ”teacher” model. Techniques such as Fine-tune-CoT [14]
and Chain of Preference Optimization [15] are designed to
explicitly optimize the student’s ability to emulate complex
reasoning paths.
C. Resource-Constrained Edge Learning
Beyond LLM-specific compression, the broader paradigm of
resource-constrained learning has seen significant explorationacross various edge applications. Building on our previous
work, the synergy of sparsity and quantization has been
extensively validated for enabling lightweight edge deploy-
ment across various domains [16], [17]. Furthermore, our
prior studies have demonstrated the considerable potential of
few-shot learning in reducing computational dependencies for
Edge-AI in both medical [18] and industrial contexts [19].
The current work adapts these foundational principles from
our group—specifically few-shot prompting and post-training
quantization—to system-level RAG integration.
III. SYSTEMARCHITECTURE ANDMETHODOLOGY
To rigorously evaluate the interplay between hardware con-
straints and generation fidelity, this study adopted a bifurcated
empirical methodology, decoupling the arithmetic impact of
quantization on latency from its downstream semantic conse-
quences.
A. Quantization Benchmarking Study
In the initial phase, we employed the foundationalMistral-
7B-v0.1model [1]. This base model was intentionally se-
lected for the benchmarking phase to evaluate the raw weight
distribution entropy without the bias of instruction-tuning
interventions. Benchmarking was executed exclusively on an
NVIDIA Tesla T4 (16 GB GDDR6)to emulate a constrained
edge environment.
4-bit Quantization Strategies
Distribution-
BasedHessian/
Error-BasedActivation-
Aware
NF4
(NormalFloat)GPTQ
(Inverse Hessian)A WQ
(Salience)
Fig. 1. Classification of evaluated quantization techniques.
We evaluated three PTQ paradigms (Fig. 1):
•NF4:Employs nested quantile quantization optimized for
zero-mean normal distributions.
•GPTQ:A layer-wise reconstruction algorithm (g= 128,
λ= 0.01) calibrated on WikiText-2 [20].
•A WQ:An activation-aware approach protecting the top
1% of weights based on activation magnitude.
B. Multimodal RAG Architecture
Leveraging the stability findings from Phase 1, we ar-
chitected the final system to prioritize stability over raw
throughput. The production pipeline employs theMistral-7B-
Instruct-v0.2checkpoint to natively support Sliding Window
Attention (SWA) for 32k context handling and to ensure strict
adherence to retrieval constraints. The system orchestrates a
modular pipeline operating strictly within a 16 GB envelope
(Fig. 2).

Source Docs
(PDF)Semantic
Chunking
(512 tokens)Embedding
Model
(MiniLM-L6)Vector
Index
(ChromaDB)
Retrieval
(Top-K)Query
EmbeddingUser
QueryQuantized LLM
(Mistral-7B)Generated
AnswerSearch
ContextPrompt TextPhase 1: Indexing
Phase 2: Inference
Fig. 2. The Modular RAG Pipeline Architecture. Indexing (Phase 1) is decoupled from inference (Phase 2).
1) Adaptive Ingestion and Dual-Path Retrieval:Documents
undergo heuristic classification to route parsing tasks between
fast text extraction (PyMuPDF) and structure-preserving Op-
tical Character Recognition (OCR) (Marker/Surya). The
integrity of the ingested text is verified via an empirical
coherence metric,Q extract :
Qextract =Cvalid
Ctotal×Scount
Sexp(1)
This serves as a heuristic baseline to trigger OCR retries if the
score falls belowτ= 0.5. Content is chunked dynamically
(256–768 tokens) to prevent breaking semantic structures like
tables.
Retrieval utilizes a dual-path mechanism combining dense
vectors (BGE-small-en-v1.5[21]) and sparse lexical represen-
tations (BM25 [22]). The disparate scores are fused using
Reciprocal Rank Fusion (RRF) [23]:
RRF(d) =X
r∈{dense,sparse}1
60 +rank r(d)(2)
2) Decoupled Multimodal Integration:To endow the sys-
tem with visual reasoning without the prohibitive memory
demands of monolithic VLMs, we implemented a decoupled
topology utilizingMicrosoft Florence-2 Base(0.23B param-
eters) [24] (Fig. 3).
Document Image Florence-2 Base
Multimodal Fusion OCR OutputQuantized LLM
(Mistral NF4)Text Tokens
Fig. 3. Decoupled Vision-Language Architecture.
Images are processed via chained prompts (<OD>,
<MORE_DETAILED_CAPTION>) to extract spatial ground-
ing and narrative semantics. The serialized outputs are injected
into a standardized textual template, effectively ”translating”
visual features into text. This incurs a negligible VRAM
overhead of approximately430 MB.3) Structured Prompting and Fidelity Validation:To coun-
teract reasoning degradation inherent in 4-bit models, we
deploy structured few-shot prompting, enforcing intent-driven
output topologies. The system employs a post-generation
heuristic filter (Q valid), assessing candidates against an em-
pirical threshold ofτ= 0.6:
Qvalid=X
d∈Dwd·Sd (3)
While these dimensions (w d) were heuristically selected based
on empirical observations in our target domain (e.g., Ground-
ednessw= 0.5), they provide a functional baseline structural
filter.
IV. EXPERIMENTS ANDRESULTS
A. Quantization Benchmarking Analysis
Latency and throughput were evaluated on a Tesla T4.
As shown in Table I, GPTQ provides the highest theoretical
throughput (51.65tok/s) and the best Time to First Token
(TTFT). However, empirical evaluation revealed severe insta-
bility and F1-score collapse at context lengths exceeding 2048
tokens on this architecture. Consequently, NF4, while slower
(12.29tok/s), was selected for the RAG pipeline to ensure
deterministic system reliability.
TABLE I
INFERENCEPERFORMANCE ANDHARDWARECOMPATIBILITY(TESLAT4)
Model Latency (ms/tok) Throughput (tok/s) TTFT (ms)
FP16 Baseline63.45±0.20 15.76 71.19
NF4 (Ours) 81.37±0.95 12.29 178.09
AWQ102.96±1.13 9.71 114.31
GPTQ19.36±0.14 51.65 43.02
B. System-Level Memory Feasibility
The breakdown of the aggregate memory footprint (Table
II) confirms the feasibility of edge deployment. The complete
multimodal pipeline incurs a peak VRAM load of 8.37 GB
on the Tesla T4. This footprint theoretically accommodates
the tighter 12-GB envelope of consumer GPUs like the RTX
3060.

TABLE II
COMPLETESYSTEMMEMORYBUDGET(TESLAT4)
Component Memory (GB) Share (%)
Quantized LLM (Mistral-7B) 3.70 44.2%
CUDA Context & Kernels 2.84 33.9%
Vision Agent (Florence-2) 0.43 5.1%
KV Cache (4k Context) 0.40 4.8%
Embedding Model (BGE) 0.30 3.6%
Vector Index (FAISS) 0.20 2.4%
System Overhead 0.50 6.0%
Total Footprint 8.37 100.0%
C. Retrieval Fidelity and Answer Quality
The RRF hybrid retrieval engine achieved a Recall@5 of
0.850 on SQUADv2 [25] (Table III). Generative fidelity data
(Table IV) reveal that the NF4 quantized Instruct model incurs
only a slight degradation compared to the FP16 baseline (F1
0.157vs.0.164).
TABLE III
RETRIEVALQUALITYMETRICS(SQUADV2)
Metric Score Implication
Recall@5 (Hybrid) 0.850 RRF Fusion of Dense + Sparse
Recall@5 (Dense) 0.700 Semantic Embedding (BGE)
Recall@5 (Sparse) 0.650 Keyword Matching (BM25)
TABLE IV
RAG ANSWERQUALITY(GENERATIVEFIDELITY)
Model F1 Score Faithfulness
FP16 Baseline 0.164 0.618
NF4 (Ours) 0.157 0.583
AWQ 0.168 0.604
D. Reasoning Restoration and Prompt Efficiency
We investigated the impact of structured prompting on
reasoning accuracy and computational cost (Table V). Com-
putational intensity remained invariant at 1.88 GFLOPs/tok,
yet mean total generation time dropped by 47% (from 20.06
s to 10.64 s). We attribute this to prompt-induced search-
space pruning; forcing explicit structural constraints eliminates
verbose hallucination loops. Crucially, this structured few-
shot formatting successfully compensates for the reasoning
loss observed in zero-shot NF4 models, restoring accuracy on
empirical mathematical tasks from 35.0% to 92.0% over 200
trials.
E. Multimodal Integration and Comparative Efficacy
To validate the decoupled system, we evaluated a custom
dataset of 50 tri-modal QA dyads. As shown in Table VI,
the integration of Florence-2 Base yielded a 192.7% accuracy
uplift relative to the text-only baseline. While native mono-
lithic VLMs offer higher theoretical ceilings, their massive
footprint excludes them from this 8GB hardware budget. Thedecoupled approach achieves functional parity (0.249 vs 0.280
score) with the 1.86B-parameter Moondream2 model [26]
while consuming approximately 50% less VRAM.
V. DISCUSSION, LIMITATIONS,ANDFUTUREWORK
A. Architectural Bottlenecks
A granular decomposition of latency exposes a critical
trade-off. While GPTQ offers superior throughput, empirical
benchmarking reveals a critical stability failure at extended
context lengths on the Turing architecture. We explicitly
prioritize NF4 to guarantee inferential robustness, trading peak
throughput for system reliability. Furthermore, we demon-
strate that structural constraints effectively compensate for the
quantization-induced loss of parametric confidence without
requiring weight-adjustment methods.
B. Limitations
While this study establishes a functional engineering base-
line, several limitations warrant future exploration. First, the
root-cause of GPTQ context collapse (e.g., quantization noise
accumulation versus ExLlamaV2 kernel bugs) requires low-
level profiling outside the scope of this system integration.
Second, to maintain simplicity on older consumer APIs, the
KV cache remained in uncompressed FP16; integrating ex-
plicit KV cache quantization (e.g., KIVI) or newer formats
like Q4 KM remains an open avenue for further memory
reduction. Third, mathematical evaluation relied on a domain-
specific dataset rather than standard benchmarks like GSM8K
to target distinct reasoning failures. Finally, while static esti-
mations suggest viability on the RTX 3060, comprehensive
end-to-end testing across Ada and Ampere architectures is
required to validate generalizability.
VI. CONCLUSION
This study provides an empirical evaluation of retrieval-
augmented generation on memory-constrained edge hardware.
By articulating a systems-engineering framework spanning
NF4 arithmetic quantization, decoupled visual integration,
and prompt-driven search-space pruning, we establish that
functional RAG is attainable on ubiquitous 12–16 GB GPUs
within a sustainable8.37 GB operational envelope. We
identify a severe context degradation with GPTQ on Turing
architectures, validating NF4 as a robust alternative. Most
significantly, we demonstrate that ”reasoning degradation” is
reversible via structured few-shot prompting, restoring ac-
curacy across 200 mathematical trials while simultaneously
accelerating generation latency by 47%. As a future direction,
the proposed framework opens the door for exploring the
potential of lightweight LLMs in heavy industries such as
the oil and gas sector, where an edge-native RAG pipeline
can be integrated with autonomous downhole systems to turn
complex, unstructured sensor data into actionable insights with
minimal cloud reliance [28] .

TABLE V
IMPACT OFSTRUCTUREDPROMPTING ONEFFICIENCY& REASONING
Metric Domain Metric Name Math Baseline (Zero-Shot) Structured (Few-Shot) Delta / Improvement
Reasoning PerformanceMath Accuracy (N= 50) 35.0% (Mean) 92.0% (Mean) +57.0 pp
Reasoning Consistency±1.0% (18 vs 17) 0.0% (46 vs 46) -1.0 pp
Reasoning Faithfulness 0.216 0.915 +323%
System EfficiencyPerplexity (WikiText-2) 13.02 6.67 -48.8%
Computational Cost 1.88 GFLOPs/tok 1.88 GFLOPs/tok 0.0%
Avg. Generation Time 20.06 s 10.64 s -47.0%
TABLE VI
COMPARATIVEEVALUATION OFVISIONENCODERS(T4 GPU)
Model Architecture Parameters Peak VRAM Vision-Only Score Overall RAG Score Relative Gain Deployment Status
No Vision (Baseline) – 2.47 GB 0.090 0.235 – Feasible (Text-Only)
Florence-2 Base[24]0.23B 2.90 GB 0.265 0.249 +192.7% Selected (Ours)
Florence-2 Large [24] 0.77B 3.92 GB 0.256 0.263 +183.1% Feasible
Moondream2 [26] 1.86B 6.07 GB 0.256 0.280 +183.7% Constrained
ViT-GPT2 [27] 0.16B 2.94 GB 0.251 0.210 +177.6% Feasible
ACKNOWLEDGMENT
The author(s) acknowledge the use of Paperpal for grammar
correction and stylistic refinement during the preparation of
this manuscript. The author(s) have reviewed and edited the
content and assume full responsibility for the final publication.
REFERENCES
[1] A. Q. Jiang, A. Sablayrolles, A. Mensch, C. Bamford, C. Devendra,
G. Lampleet al., “Mistral 7b,”arXiv preprint arXiv:2310.06825, 2023.
[2] P. Lewis, E. Perez, A. Piktus, F. Petroni, V . Karpukhin, N. Goyalet al.,
“Retrieval-augmented generation for knowledge-intensive nlp tasks,” in
Adv. Neural Inf. Process. Syst. (NeurIPS), 2020.
[3] S. Kimet al., “Squeezellm: Dense-and-sparse quantization,” inProc.
Int. Conf. Mach. Learn. (ICML), 2024.
[4] X. Zhuet al., “A survey on model compression for large language
models,” inProc. Int. Jt. Conf. Artif. Intell. (IJCAI), 2023.
[5] E. Frantar, S. Ashkboos, A. Torralba, and D. Alistarh, “Gptq: Accurate
post-training quantization for generative pre-trained transformers,” in
Proc. Int. Conf. Learn. Represent. (ICLR), 2023.
[6] T. Dettmers, M. Lewis, Y . Belkada, and L. Zettlemoyer, “Llm.int8():
8-bit matrix multiplication for transformers at scale,” inAdv. Neural Inf.
Process. Syst. (NeurIPS), 2022.
[7] T. Dettmers, R. Svirschevski, V . Egiazarian, D. Kuznedelev, E. Frantar,
S. Ashkbooset al., “Spqr: A sparse-quantized representation for near-
lossless llm weight compression,” inProc. Int. Conf. Learn. Represent.
(ICLR), 2024.
[8] J. Lin, J. Tang, H. Tang, S. Yang, X. Dang, and S. Han, “Awq:
Activation-aware weight quantization for llm compression,” inProc.
Mach. Learn. Syst. (MLSys), 2024.
[9] G. Xiao, J. Lin, M. Seznec, H. Wu, J. Demouth, and S. Han,
“Smoothquant: Accurate and efficient post-training quantization for large
language models,” inProc. Int. Conf. Mach. Learn. (ICML), 2023.
[10] S. Leeet al., “Kvquant: Towards 10 million context length llm infer-
ence,”arXiv preprint arXiv:2401.18079, 2024.
[11] Z. Liuet al., “Kivi: A tuning-free asymmetric 2bit quantization for kv
cache,”arXiv preprint arXiv:2402.02750, 2024.
[12] E. Frantar and D. Alistarh, “Sparsegpt: Massive language models can be
accurately pruned in one-shot,” inProc. Int. Conf. Mach. Learn. (ICML),
2023.
[13] X. Ma, G. Fang, and X. Wang, “Llm-pruner: On the structural pruning
of large language models,” inAdv. Neural Inf. Process. Syst. (NeurIPS),
2023.
[14] E. Loboet al., “On the impact of fine-tuning on chain-of-thought
reasoning,”arXiv preprint arXiv:2411.15382, 2024.[15] X. Zhanget al., “Chain of preference optimization: improving chain-of-
thought reasoning in llms,” inAdv. Neural Inf. Process. Syst. (NeurIPS),
2024.
[16] D. Aboutahoun, R. Zewail, K. K. Kimura, and M. Soliman, “Lightweight
histological tumor classification using a joint sparsity-quantization aware
training framework,”IEEE Access, 2023.
[17] J. Herman, R. Zewail, T. Ogawa, and S. ElSagheer, “A lightweight
transfer learning-based model for building classification in aerial im-
agery,” in15th IEEE International Conference on Computer Research
and Development, 2023.
[18] D. Aboutahoun, R. Zewail, and M. Soliman, “On potentials of few-
shot learning for ai-enabled internet of medical things,” in2022 IEEE
Globecom Workshops (GC Wkshps), 2022.
[19] R. Zewail and B. Mokhtar, “Resource-aware contrastive scattering meta-
learning for efficient few-shot acoustic anomaly detection,”Advanced
Intelligent Systems, vol. 8, no. 9, p. 2501454, 2026.
[20] S. Merity, C. Xiong, J. Bradbury, and R. Socher, “Pointer sentinel
mixture models,”arXiv preprint arXiv:1609.07843, 2016.
[21] S. Xiaoet al., “C-pack: Packaged resources to advance general chinese
embedding,”arXiv preprint arXiv:2309.07597, 2023.
[22] S. Robertson and H. Zaragoza, “The probabilistic relevance framework:
Bm25 and beyond,”Foundations and Trends® in Information Retrieval,
vol. 3, no. 4, pp. 333–389, 2009.
[23] G. V . Cormack, C. L. Clarke, and S. Buettcher, “Reciprocal rank fusion
outperforms condorcet and individual rank learning methods,” inProc.
32nd Int. ACM SIGIR Conf. Res. Develop. Inf. Retr. (SIGIR), Boston,
MA, USA, Jul 2009, pp. 758–759.
[24] B. Xiao, H. Wu, W. Xu, X. Dai, H. Hu, Y . Xieet al., “Florence-2:
Advancing a unified representation for a variety of vision tasks,”arXiv
preprint arXiv:2311.06242, 2024.
[25] P. Rajpurkar, R. Jia, and P. Liang, “Know what you don’t know:
Unanswerable questions for squad,” inProc. 56th Annu. Meeting Assoc.
Comput. Linguist. (ACL), Melbourne, Australia, 2018, pp. 784–789.
[26] K. Vet al., “Moondream2: A tiny vision language model,” 2024, gitHub
Repository. [Online]. Available: https://github.com/vikhyat/moondream
[27] NLP Connect, “Vit-gpt2 image captioning,” 2024, hugging Face
Model Hub. [Online]. Available: https://huggingface.co/nlpconnect/
vit-gpt2-image-captioning
[28] S. T. Al Amin, K. Wu, A. Mathur, and C. Koritala, “Llm-in-the-loop: A
framework for enhancing drilling data analytics,” inSPE/IADC Interna-
tional Drilling Conference and Exhibition, 2025, dOI: 10.2118/223805-
MS.