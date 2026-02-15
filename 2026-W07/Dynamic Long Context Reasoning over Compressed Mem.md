# Dynamic Long Context Reasoning over Compressed Memory via End-to-End Reinforcement Learning

**Authors**: Zhuoen Chen, Dongfang Li, Meishan Zhang, Baotian Hu, Min Zhang

**Published**: 2026-02-09 08:33:11

**PDF URL**: [https://arxiv.org/pdf/2602.08382v1](https://arxiv.org/pdf/2602.08382v1)

## Abstract
Large Language Models (LLMs) face significant challenges in long-context processing, including quadratic computational costs, information forgetting, and the context fragmentation inherent in retrieval-augmented generation (RAG). We propose a cognitively inspired framework for efficient long-context inference based on chunk-wise compression and selective memory recall, rather than processing all raw tokens. The framework segments long inputs into chunks and encodes each chunk into compressed memory representations using a learned compressor. A gating module dynamically selects relevant memory blocks, which are then iteratively processed by a reasoning module with an evolving working memory to solve downstream tasks. The compressor and reasoner are jointly optimized via end-to-end reinforcement learning, while the gating module is trained separately as a classifier. Experimental results show that the proposed method achieves competitive accuracy on multi-hop reasoning benchmarks such as RULER-HQA, extrapolates context length from 7K to 1.75M tokens, and offers a favorable accuracy-efficiency trade-off compared to strong long-context baselines. In particular, it achieves up to a 2 times reduction in peak GPU memory usage and a 6 times inference speedup over MemAgent.

## Full Text


<!-- PDF content starts -->

Lychee
Dynamic Long Context Reasoning over Compressed Memory
via End-to-End Reinforcement Learning
Zhuoen Chen, Dongfang Li, Meishan Zhang, Baotian Hu, Min Zhang
Research Institute of Computing and Intelligence
Harbin Institute of Technology, Shenzhen
Abstract
Large Language Models (LLMs) face severe challenges in long-context processing, including
quadratic computational costs, information forgetting, and the context fragmentation inher-
ent in Retrieval-Augmented Generation (RAG). We introduce LycheeMemory, a cognitively
inspired framework that enables efficient long-context inference via chunk-wise compression
and selective memory recall, rather than processing all raw tokens. LycheeMemory segments
the input into chunks and encodes each into compressed KV-cache-style representations
using a Compressor . AGate then dynamically selects relevant memory blocks, which a
Reasoner iteratively processes with an evolving working memory to solve downstream
tasks. The Compressor andReasoner are jointly optimized via end-to-end reinforce-
ment learning, while the Gate is trained separately as a classifier. Experimental results
demonstrate that LycheeMemory achieves competitive accuracy (up to 82% in ablation
variants) on multi-hop reasoning benchmarks (e.g., RULER-HQA), successfully extrapolates
context length from 7K to 1.75M, and provides a favorable accuracy–efficiency trade-off
against strong long-context baselines. Notably, compared to MemAgent, LycheeMemory
achieves an average 2 ×reduction in peak GPU memory usage and a 6 ×speedup during
inference.
14k 56k 112k 256k020406080100Accuracy (sub-EM %)81.25 81.25 80.47 79.69 79.69
77.3479.69
72.66
60.94
57.03
50.00
37.50
(a) Performance Comparison8k16k 32k 64k 128k103104Inference Time (s)
(b) Inference EfficiencyLycheeMemory MemAgent Full Context (Qwen2.5-7B-Instruct)
Figure 1:LycheeMemory achieved the best performance and latency.Left:Relative performance comparison
of various methods on the Qwen2.5-7B model across different LongBench datasets.Right:Inference time
comparison across different context lengths of 128 samples.arXiv:2602.08382v1  [cs.CL]  9 Feb 2026

CONTENTS
Contents
1 Introduction 4
2 Related Work 4
3 Methodology 5
3.1 Overview . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 5
3.2 Compressed Memory Construction . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 6
3.2.1 KV-cache Style Compression via Memory Tokens . . . . . . . . . . . . . . . . . . 6
3.2.2 Pre-optimization of the Compressor . . . . . . . . . . . . . . . . . . . . . . . . . . 6
3.3 Dynamic Recall and Reasoning Workflow . . . . . . . . . . . . . . . . . . . . . . . . . . . 7
3.3.1 LoRA Gate . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 7
3.4 End-to-End RL Optimization . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 8
3.4.1 Joint Policy Formulation . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 8
3.5 Complexity and Efficiency Analysis . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 8
4 Experiments 9
4.1 Experimental Setup . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 9
4.2 Main Results . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 9
4.3 Inference Efficiency Analysis . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 9
4.4 Zero-shot Generalization . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 11
4.5 Ablation Study . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 11
4.5.1 Different Compression Ratios . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 11
4.5.2 Ablation on Gate . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 11
4.5.3 Analysis of Staged Optimization Strategies . . . . . . . . . . . . . . . . . . . . . . 11
5 Conclusion 12
A Implementation Details 16
A.1 Stage 1: Compressor Pre-training . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 16
A.2 Stage 2: Joint Reinforcement Learning Optimization . . . . . . . . . . . . . . . . . . . . . 16
A.3 Stage 3: Gate Module Training . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 17
A.4 Evaluation and Baselines . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 17
A.5 Storage and Computation Trade-off . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 18
B Training Convergence Analysis 18
C Failure Mode Analysis 19
C.1 Unidirectional Dependency Mismatch (35%) . . . . . . . . . . . . . . . . . . . . . . . . . 19
C.2 Premature Inference Anchoring (21%) . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 20
C.3 Compression-Induced Hallucination (17%) . . . . . . . . . . . . . . . . . . . . . . . . . . 20
C.4 Other Error Types (27%) . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 20
2

CONTENTS
D Computational Complexity 21
D.1 FLOPs Formulation . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 21
D.2 Quantitative Comparison . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 22
E Out-of-Distribution (OOD) Generalization Analysis 23
F Ablation on Working Memory Capacity 23
G Dynamic Evolution of Working Memory 24
H Case Analysis 24
H.1 Hallucination via Feature Collapse . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 24
H.2 Associative Reasoning and Self-Correction . . . . . . . . . . . . . . . . . . . . . . . . . . 25
I Comparison with RAG 26
3

1. Introduction
1 Introduction
Despite the remarkable capabilities demonstrated by Large Language Models (LLMs), efficiently processing
long contexts remains a critical challenge Liu et al. (2025); Comanici et al. (2025); Wan et al. (2025). To
address this bottleneck, current methodologies primarily diverge into three paradigms, each facing inherent
trade-offs between efficiency and capability. Sparse and linear attention mechanisms (Beltagy et al., 2020; Xiao
et al., 2023; Katharopoulos et al., 2020) reduce computational complexity but often suffer from performance
degradation on extremely long sequences. Retrieval-Augmented Generation (RAG) (Lewis et al., 2020;
Karpukhin et al., 2020; Izacard & Grave, 2021) mitigates length constraints while facing severe context
fragmentation. By treating text chunks as independent entities, it disrupts the logical dependencies essential for
multi-hop reasoning and struggles to capture implicit semantic connections. Conversely, recurrent architectures
like RecurrentGPT (Zhou et al., 2023) and MemAgent Yu et al. (2025) rely on sequential state updates, resulting
in slow serial inference speeds that significantly hinder scalability.
To overcome these limitations, we draw inspiration from the mechanisms of human memory and propose Ly-
cheeMemory. By mimicking the division of labor between compressed memory bank (i.e.,long-term memory)
and dynamic working memory Atkinson & Shiffrin (1968), our framework splits the input text into chunks and
compresses them into efficient, high-fidelity compressed KV-cache representations. This builds a compressed
memory bank that preserves semantic information while reducing computational costs. During inference, we
use a dynamic recall and reasoning workflow driven by a Gate and a Reasoner . It starts with an empty
working memory, explicitly instantiated as a fixed-length, token-level context window. This design preserves
a discrete action space, thereby enabling the memory update process to be optimized via Reinforcement
Learning (RL). Subsequently, LycheeMemory sequentially traverses the compressed memory bank: for each
chunk, the Gate evaluates whether the chunk contributes to the current reasoning state, given the current
working memory and the user query. If deemed relevant, the Reasoner utilizes the chunk to update current
working memory; otherwise, the chunk is skipped. Through this selective update and iterative refinement, the
Reasoner facilitates multi-step reasoning across multiple memory chunks, avoiding the blind processing of
the entire input sequence characteristic of traditional recurrent architectures.
A core challenge is ensuring that the compressed memory can be effectively used by the Reasoner for
precise inference. We adopt a joint policy optimization strategy: we train the Compressor andReasoner
end-to-end with RL, and train the Gate separately as a classifier. We evaluate LycheeMemory on RULER-
HQA Yang et al. (2018); Hsieh et al. (2024), 2WikiMultihopQA Ho et al. (2020), and StreamingQA Liska
et al. (2022). Experimental results show that LycheeMemory maintains competitive accuracy on multi-hop
reasoning, extrapolates context length from 7K to 1.75M, and improves the accuracy–efficiency trade-off.
Compared to MemAgent Yu et al. (2025), LycheeMemory reduces peak GPU memory usage by 2 ×and speeds
up inference by 6×.
The main contributions of this work are summarized as follows:
•We propose LycheeMemory, a framework comprising a Compressor , aGate , and a Reasoner ,
which transforms long-context processing from direct modeling of raw tokens into efficient iterative
reasoning over a compressed memory bank.
•We introduce a joint policy optimization strategy that trains the Compressor andReasoner
end-to-end via RL, enabling the compressed memory to be directly optimized for downstream tasks.
•Experimental results show that LycheeMemory scales the context size to 1.75M tokens and improves
inference efficiency while maintaining competitive accuracy.
2 Related Work
Explicit Memory Methods.Explicit memory methods externalize context as human-readable text or sym-
bols. Standard RAG retrieves static chunks via semantic similarity but often suffers from context fragmentation
and limited precision in multi-hop reasoning Gutiérrez et al. (2025); Weller et al. (2025); Merola & Singh
(2025). Agentic memory systems mitigate this by actively managing external memory, such as MemGPT’s
OS-inspired hierarchy Packer et al. (2023) and Mem0’s lifecycle-based memory updates Chhikara et al. (2025).
More recent RL-based approaches (e.g., MemAgent Yu et al. (2025), Mem1 Zhou et al. (2025)) learn to
manage a bounded memory by selectively overwriting or integrating observations during streaming. Despite
their interpretability, these methods operate on raw tokens and incur substantial computational overhead. In
contrast, our approach leverages compressed memory with selective retrieval, achieving lower peak memory
usage and inference latency.
4

3. Methodology
StaticMemoryKV 1 KV 2 KV n
chunk 1 chunk 2 chunk n
CompressorLLM
KV 1 KV 2 KV nLLM
gate reasonerKV iWorking
Memory
Unrelated
Next StepRelated
KV iWorking
MemoryUpdated
Memoryquery
Step i
Memory Compression Dynamic Recallstep 1 step 2 step n answer
Figure 2:Overview of the LycheeMemory framework. The left panel illustrates compressed memory
construction, where a long document is segmented and compressed into compact KV-cache representations
by the compressor. The right panel depicts the dynamic recall and reasoning workflow, in which the gate
selectively activates relevant memory blocks and the reasoner iteratively updates the working memory to
produce the final answer.
Implicit Memory Methods.Implicit memory methods optimize internal representations via activation
compression or parametric updates. To alleviate the quadratic cost of self-attention, cache compression
approaches exploit attention sparsity, retaining only salient tokens (e.g., H2O Zhang et al. (2023), SnapKV Li
et al. (2024)). Beyond static pruning, dynamic methods retrieve relevant cache blocks on demand Xiao et al.
(2024); Gao et al. (2025). Parametric alternatives, such as DyPRAG Tan et al. (2025), encode documents into
latent LoRA adapters Hu et al. (2022) and route queries to specialized weights. While effective in reducing
memory footprint, aggressive compression often degrades long-tail reasoning Zhang et al. (2025), and purely
latent approaches Hao et al. (2024); Eyuboglu et al. (2025) lack structured retrieval needed for large-scale
multi-document streams. In contrast, our method couples selective retrieval with iterative working memory
updates via aGateandReasoner, enabling robust multi-hop reasoning over million-token contexts.
Overall, LycheeMemory bridges explicit and implicit memory: it stores documents as compressed KV-cache
representations, while performing state-dependent retrieval and reasoning through a plaintext working memory.
This retains the scalability benefits of compression and yields an interpretable trace over selected evidence
chunks.
3 Methodology
3.1 Overview
We address long-context modeling where a model takes ultra-long documents D(length N) and a user query
Qto generate an answer A. Due to the prohibitive length of D, processing the entire sequence directly
is computationally infeasible. To address this, we proposeLycheeMemory, a dual-system framework for
long-context processing. As illustrated in Figure 2, the architecture comprises three core roles:
•CompressorΦ comp: Composed of the base model Φaugmented with a compression LoRA module
Ψcomp, responsible for encoding raw text into KV-cache-style memory.
•GateΦ gate: Composed of the base model Φaugmented with a gating LoRA module Ψgate, acting as
a relevance filter.
5

3.2 Compressed Memory Construction
•ReasonerΦ reason: Composed of the base model Φaugmented with a reasoning LoRA module
Ψreason , responsible for complex reasoning based on recalled memories.
LetDbe segmented into Ksequential chunks (size sz, i.e., K=N/sz ) asD={C 1, C2, . . . , C K}. In our
experiments, we setsz= 4096. The processing workflow of LycheeMemory involves two main phases:
Memory Compression:In this phase (detailed in §3.2), each text chunk Ckis processed by the
CompressorΦ comp and encoded into a compact latent representation θk. This representation is subse-
quently stored in the compressed memory bankΘ, i.e.,Θ ={θ 1, . . . , θ K}.
Dynamic Recall and Reasoning:Distinct from the latent representations used for storage, the model
maintains a working memory mduring the dynamic recall and reasoning phase. mexists as plaintext tokens
within the model’s context and is iteratively updated as the model scans the compressed memory bank to
maximize reasoning capability. When receiving a user query Q(detailed in §3.3), the model scans memory
blocks with index i= 1, . . . , K , activating the GateΦ gateandReasonerΦ reason. The process starts with
an initial empty working memory m0. At scan step i, theGate evaluates the compressed memory block θi
in conjunction with the current working memory mtand query Q. If deemed relevant, the Reasoner is
invoked to update the working memory state: mt+1= Φ reason(mt, θi, Q), and we increment the update index
t←t+ 1 ; otherwise, we skip this block and keep tunchanged. Finally, the model synthesizes the answer A
based onm TandQ, whereT≤K.
3.2 Compressed Memory Construction
The construction of the compressed memory bank Θis central to the LycheeMemory framework. We present
a KV-cache compression style method that achieves an optimal balance between information density and
computational efficiency.
3.2.1 KV-cache Style Compression via Memory Tokens
Similar to previous works Chevalier et al. (2023); Deng et al. (2025), we define the compression as a mapping
from text to a latent representation, Ci→θi. We utilize base model Φaugmented by a LoRA module Ψcomp
as the Compressor , eliminating the need for an external encoder. For any text chunk Ci= [xi
1, . . . , xi
w]of
length w, we first determine a compression ratio αi. We then define a set of zi=w/α itrainable memory
tokens Vi={⟨v⟩i
1, . . . ,⟨v⟩i
zi}. Next, we interleave ViwithCiby inserting a memory token after every αi
original tokens, forming interleaved sequenceC′
i:
C′
i= Interleave(C i, Vi) = [xi
1, . . . , xi
αi,⟨v⟩i
1, . . . , xi
w,⟨v⟩i
zi]
This sequence C′
iis passed through Φcompfor a single forward pass. During this process, the model is trained
to embed the semantic information of the preceding αitokens into the hidden state of the subsequent memory
token⟨v⟩i
j. Finally, the set of hidden states corresponding to all memory tokens constitutes the compact
KV-cache style representationθ istored in the compressed memory bankΘ:
θi={h(⟨v⟩i
1), . . . , h(⟨v⟩i
zi)},
whereh(·) = HiddenState(Φ comp(C′
i)).
3.2.2 Pre-optimization of the Compressor
Before end-to-end RL, to ensure that θiretains the core semantic information of Cidespite high compression,
we jointly optimize the LoRA module Ψcompwhile keeping the base model Φfrozen using data augmentation
and diverse tasks. Note that in the encoding phase ( C′
i→θ i), the base model Φcombined with Ψcomp
generates the compressed representation θi. Conversely, in all subsequent decoding tasks based on θi, we
utilize only the frozen base model Φwithout Ψcompfor generation. This design ensures that the gradient flows
only throughΨ comp, effectively decoupling the compression capability from the general generation ability of
the base model. Given a compressed representation θi, the model Φis trained to perform three distinct tasks.
LetP Φ(Y|context)be the probability generatingYgiven the context:
Text Reconstruction.The model must regenerate the original textC iusing onlyθ ias context.
Lrecon=−logP Φ(Ci|θi)
6

3.3 Dynamic Recall and Reasoning Workflow
QA Generation.We pre-generate synthetic question-answer pairs (Qj, Aj)forCi. The model generates Aj
givenθ iandQ j. The lossL qais computed only over the answerA j.
Lqa=−E (Qj,Aj)∼Ci[logP Φ(Aj|θi, Qj)]
Creative Generation.The model performs high-level semantic tasks based on θi, such as generating a
summaryS i. We use the model output based on the original text,Φ(C i), as the ground-truth labelY creative .
Lcreat=−logP Φ(Ycreat|θi)
whereY creat= Φ(C i, Ptask)
The total lossL compis a weighted sum of the above losses, minimized by updatingΨ comp:
min
ΨcompLcomp=ECi∼D[w1Lrecon+w 2Lqa+w 3Lcreat]
We train separate projection matrices for the memory tokens vmem, functionally isolating them from regular
token representations to learn a dedicated compression subspace.
3.3 Dynamic Recall and Reasoning Workflow
After constructing the compressed memory bank Θ, the core of LycheeMemory lies in efficiently retrieving
and reasoning over these compressed representations. In contrast to methods like MemAgent (Yu et al., 2025),
which employ linear scanning with forced updates for every chunk, we introduce a relevance threshold τ. As
the system traverses the compressed memory bank, the Gate scores each compressed memory block, and
only blocks exceeding this threshold trigger theReasonerto update the working memory.
3.3.1 LoRA Gate
To avoid the overhead of unnecessary memory updates, we require a filter to discard static blocks irrelevant to
the user query Q. An intuitive solution would be an embedding model calculating cosine similarity between
chunks and Q. While such lightweight retrieval can be reasonably strong on recall, it only captures static
semantic similarity and lacks state-dependent retrieval conditioned on the evolving working memory m(See
§4). This limitation becomes salient in multi-hop settings, where later-hop evidence may only become relevant
after intermediate entities are added into m. A further critical limitation is that external retrievers cannot
leverage the working memory m, which often contains key secondary clues (e.g., intermediate entities) derived
from the query and previously processed memory chunks. Motivated by this, we implement the GateΦ gateby
adding a LoRA adapterΨ gateto the base modelΦ.
Architecture and Inference.Given a user query Q, the current working memory mt, and a candidate
memory block θi∈Θ (represented by its memory tokens), we concatenate them and extract the hidden state of
the final token, hlast. This state is projected by a trainable linear head Wgatefollowed by a sigmoid activation
to produce a relevance probability:
P=σ(W gate·hlast(Φ(Q,m t, θi; Ψgate)))
The memory block is used to update the working memory only ifP > τ.
Training Objective.Due to the gradient discontinuity caused by discrete recall decisions, we treat gating
training as a separate binary classification task beyond RL in §3.4. We align text chunks with downstream
tasks (e.g., QA pairs) to construct training data. A memory block θiis labeled positive ( y∗
i= 1) if it contains
evidence required to answer Q, and negative ( y∗
i= 0) otherwise. We optimize the gate parameters (LoRA
Ψgateand HeadW gate) using Binary Cross-Entropy (BCE) loss:
Lgate=−1
NNX
i=1h
y∗
ilogP i+ (1−y∗
i) log(1−P i)i
where Piis the predicted probability. This lightweight design ensures the model identifies memory chunks
relevant to the query and current reasoning state in the latent space.
7

3.4 End-to-End RL Optimization
3.4 End-to-End RL Optimization
To empower LycheeMemory with the capability of complex reasoning over compressed memories, we propose
an enhanced reinforcement learning framework. Unlike prior approaches that optimize components in isolation,
we formulate the entire lifecycle from memory compression to reasoning as a unified joint policy optimization
problem. This allows the gradient from the final reasoning outcome to backpropagate through the recall
workflow and update the Compressor , ensuring the Θ(i.e.,long-term memory) is optimized specifically for
downstream inference.
3.4.1 Joint Policy Formulation
We define the joint policy πϑparameterized by ϑ, which encompasses both the Compressor parameters
(Ψcomp) and the Reasoner parameters ( Ψreason ). For a given input document Dand query Q, the generation
of an answerAinvolves a hierarchical trajectory:
πϑ(A,M,Θ|D, Q) =KY
k=1πcomp(θk|Ck)
| {z }
Memory Construction·TY
t=1πreason(mt|mt−1,Θ, Q)
| {z }
Dynamic Recall and Reasoning
where Θ ={θ k}represents the compressed memory bank, and M={m t}T
t=0represents the sequence of
working memory updates. Our goal is to maximize the expected reward of the final answer Aby optimizing ϑ.
The Unified Objective Function.We formulate the unified objective to jointly optimize compression and
reasoning:
J(ϑ) =EQ∼D,{O i}G
i=1∼πϑold
1
GGX
i=11
niniX
j=1 
LCLIP
i,j(ϑ)−βD KL(πϑ∥πref)

whereLCLIP
i,j(ϑ) = min
ρi,j(ϑ)ˆAi,j,clip(ρ i,j(ϑ),1−ϵ,1 +ϵ) ˆAi,j
Here, ρi,j(ϑ)represents the sequence-level importance sampling weight defined by GSPO Zheng et al. (2025).
Gdenotes the group size (number of sampled trajectories per prompt), and nidenotes the number of tokens in
thei-th trajectory. By maximizing J(ϑ) , the model learns to compress context into Θsuch that the reasoning
policy maximizes the likelihood of high-advantage trajectories.
3.5 Complexity and Efficiency Analysis
In this section, we analyze the computational efficiency of LycheeMemory compared to existing long-context
methods.
Memory Construction.This phase incurs O(N) complexity, but it is a one-time, fully parallelizable
pre-processing cost.
Gate.While approaches like MemAgent (Yu et al., 2025) achieve O(N/sz) linear complexity via streaming,
they require performing full token generation (i.e., memory updates) for every text chunk. In contrast,
although the Gate in LycheeMemory must also traverse all Kcompressed memory blocks to determine
relevance, maintaining an O(N/sz) complexity, the computational cost per block is drastically reduced. The
Gate requires only a single forward pass for scalar classification, rather than the computationally expensive
autoregressive generation used in standard streaming methods.
Dynamic Recall and Reasoning.The heavy computational load of the Reasoner is decoupled from the
document lengthNand depends only on the number of retrieved blocksT:
Oinference ≈O(N/sz×C gate+T×C reason)
where Cgate≪C reason . Since the Gate efficiently filters out irrelevant information ( T≪N/sz ), LycheeMem-
ory achieves a significantly lower constant factor in its linear scaling compared to methods that reason over
every chunk.
8

4. Experiments
Model 7K 14K 28K 56K 112K 224K 448K 896K 1.75M
QwenLong-L1-32B Wan et al. (2025) 72.66 75.00 72.66 60.94 31.25 17.19 13.28 11.72 OOM
Qwen2.5-Instruct-14B-1M Yang et al. (2025a) 60.16 60.94 50.00 57.03 50.00 37.50 8.59 0.00 OOM
Qwen2.5-Instruct-7B-1M Yang et al. (2025a) 61.72 56.25 53.91 55.47 51.56 33.59 12.50 0.00 OOM
DS-Distill-Qwen-32B Guo et al. (2025) 70.31 66.41 65.62 46.88 23.44 13.28 7.81 7.03 OOM
DS-Distill-Qwen-14B Guo et al. (2025) 64.06 64.84 57.03 40.62 14.84 8.59 3.12 6.25 OOM
DS-Distill-Qwen-7B Guo et al. (2025) 30.47 12.50 3.12 0.00 0.00 0.78 0.00 0.00 OOM
RAG + Qwen2.5-7B-Instruct 67.19 66.41 66.41 67.19 64.84 64.06 62.5 61.72 62.38
Search-R1 Jin et al. (2025) 72.66 71.88 67.71 73.96 66.67 62.5 64.58 67.71 67.19
RL-MemAgent-7B Yu et al. (2025)82.0379.69 78.91 77.34 79.69 72.66 74.2276.5675.78
LycheeMemory-7B(ours) 77.34 1.0× 76.56 1.2× 75.00 1.6× 76.56 2.5× 75.78 3.5× 73.44 5.9× 74.22 9.7× 72.66 17.7× 71.09 28.2×
LycheeMemory-7B w/o Gate(ours) 80.4781.25 82.03 81.25 80.47 79.69 75.0075.7878.12
Table 1:Comparison of main experimental results under different context lengths. All values are normalized
sub-EM accuracy (%). Blue indicates the inference speedup of LycheeMemory relative to its without Gate
ablation.
4 Experiments
In this section, we evaluate LycheeMemory on long-context QA, analyze inference efficiency and zero-shot
generalization, and validate core design choices through ablations.
4.1 Experimental Setup
Model ConfigurationWe use Qwen2.5-Instruct Yang et al. (2025b) as the base model and train
LycheeMemory-3B/LycheeMemory-7B initialized from Qwen2.5-3B/7B-Instruct.
Dataset ConstructionFollowing MemAgent Yu et al. (2025), we synthesize long-document training data
from RULER-HQA Yang et al. (2018); Hsieh et al. (2024) by mixing query-relevant articles with distractors
(avg. 20K tokens). We evaluated contexts from 7K to 1.75M tokens for length extrapolation and reported
zero-shot results on 2WikiMultihopQA Ho et al. (2020), StreamingQA Liska et al. (2022).
BaselinesWe compare with Search-R1 Jin et al. (2025), MemAgent Yu et al. (2025), DeepSeek-R1-Distill-
Qwen Guo et al. (2025), Qwen-2.5-Instruct-1M Yang et al. (2025a), and QwenLong-L1 Wan et al. (2025),
using official configurations. Additional details are in Appendix A and Appendix A.4.
4.2 Main Results
We first evaluate LycheeMemory on the synthesized HotpotQA dataset as context length grows. Table 1 shows
the comparison with baselines.
Performance at ScaleWe compare models from 7K to 896K context lengths. For memory-based models
(Search-R1, MemAgent, and LycheeMemory), we further evaluate extrapolation at an ultra-long 1.75M tokens
to inspect generalization beyond standard training ranges. As shown in Table 1, several baselines fail even
within their nominal windows. Reasoning models (e.g., DS-Distill-Qwen series) degrade rapidly as context
length increases. In contrast, MemAgent and LycheeMemory show strong length extrapolation, with only mild
performance drop as input length increases, validating the effectiveness of the chunked memory mechanism.
Comparison with MemAgentCompared to MemAgent, our LycheeMemory-7B w/o Gate ablation achieves
higher accuracy across most evaluated context lengths, while LycheeMemory with Gate trades a small accuracy
drop for substantially improved inference efficiency (see §4.3). This indicates that compressed memory with
RL-trained reasoning is competitive in accuracy, and the Gate provides an effective accuracy–efficiency
trade-off in ultra-long contexts.
4.3 Inference Efficiency Analysis
A key advantage of LycheeMemory is computational efficiency. We measure end-to-end inference time on 2 ×
A100 (80GB) for 128 samples from 8K to 128K tokens (generation length 1024, largest non-OOM batch).
The reported time includes compression and I/O. Figure 3 shows three regimes:
9

4.3 Inference Efficiency Analysis
8 16 32 64 128
Context Length (ktokens)0200040006000800010000120001400016000Inference Time (s)Inference Efficiency: Time vs. Context Length (128 Samples)
LycheeMemory
LycheeMemory w/o Gate
MemAgent
FullContext
Figure 3:Inference latency as context length increases. LycheeMemory exhibits a nearly flat latency curve, in
contrast to the quadratic and linear increases observed in the full-context and MemAgent baselines respectively.
Quadratic ExplosionThe Qwen2.5-7B baseline exhibits the expected O(N2)latency growth. At 64K it is
markedly slower than memory-based methods and at 128K it further fails due to OOM.
Linear GrowthMemAgent and our ablation LycheeMemory without Gate (linear scan over all compressed
memory blocks) show linear O(N) complexity. However, LycheeMemory without Gate remains faster than
MemAgent because our compressed memory is a highly compressed KV-cache ( α≫1 ), so the effective
sequence length processed by the reasoning workflow is much shorter than the text stream of MemAgent.
Near-Constant InferenceWith the Gate module, LycheeMemory shows striking efficiency. As context
grows from 8K to 128K, inference time rises only slightly. Compression and Gate overhead grows linearly
(with a tiny coefficient), while the costly reasoning (with memory update) steps run on only a few retrieved
blocks. In terms of results, at 128K we achieve a 6 ×speedup over MemAgent and a 3.5 ×speedup over the
w/o Gate baseline; meanwhile, Table 1 shows that the accuracy drop on the closest reported bucket (112K) is
only 6%. Additional analyses are in Appendix A.5 and Appendix D.
Method 2WikiMultihopQA StreamingQA
14K 28K 56K F1 sub-EM
Qwen2.5-Instruct-7B 57.0 42.2 37.5 30.5 23.4
RAG 68.8 64.1 59.4 84.367.2
MemAgent 74.273.471.1 77.9 60.2
LycheeMemory 75.070.373.4 80.873.4
Table 2:Zero-shot comparison results of 2WikiMulti-
hopQA and StreamingQA.Method 56K 112K 224K
Text-embedding-3-large 94.3 82.1 80.9
Gate (Query Only) 88.2 76.4 74.8
Gate (Query + Memory) 98.5 86.3 84.1
Table 3:Recall of gold supporting chunks on multi-
hop QA samples across context lengths. All methods
retrieve the top 8 chunks under an identical retrieval
budget.
10

4.4 Zero-shot Generalization
4.4 Zero-shot Generalization
We evaluate LycheeMemory zero-shot on 2WikiMultihopQA and StreamingQA. Table 2 shows strong perfor-
mance on unseen multi-document reasoning tasks. Due to space limitations, additional OOD evaluations of
LongBench benchmark Bai et al. (2024) on Appendix E.
4.5 Ablation Study
To analyze the contribution of each component, we conduct a series of ablation studies using the
LycheeMemory-3B model.
4.5.1 Different Compression Ratios
We study the effect of compression ratios ( α∈ {2,4,8,16} ) on reasoning accuracy over context lengths from
2K to 128K tokens (Figure 4). Results reveal a clear trade-off between memory efficiency and information
retention. Both 2×and4×compression maintain near-lossless performance, preserving >80% accuracy
even at 128K, with a negligible gap ( <1% ) between them, indicating that 4×compression is sufficient to
capture semantic density without redundancy. In contrast, 16× compression degrades sharply (71.5% at 2K
to 42.0% at 128K), while 8×provides a compromise but exhibits mild attrition ( <10% ) at extreme lengths.
Accordingly, we adopt α= 4 as the default, halving the memory footprint of α= 2 with no statistically
significant loss in reasoning performance.
4.5.2 Ablation on Gate
Experimental Setup.We evaluate different retrieval strategies under increasing context lengths by seg-
menting the input into non-overlapping 4096-token chunks. For the embedding baseline, we further split
each 4096-token chunk into 1024-token micro-chunks, score each micro-chunk with the query, and use the
maximum score as the chunk score.
Results and Analysis.As shown in Table 3, all methods perform well at shorter contexts (56K). However,
baselines show a clear performance drop as context length increases. Static embedding-based retrieval and
query-only Gate decline at 112K, with the strongest baseline dropping to 82.1%. In contrast, our Gate
conditioned on both the query and the evolving working memory maintains a high recall of 86.3% at 112K
and 84.1% at 224K, consistently surpassing other retrieval strategies.
This trend reflects the state-dependent nature of multi-hop reasoning: static retrievers model P(Chunk|Q)
and overemphasize early-hop evidence, whereas LycheeMemory conditions retrieval on the evolving memory
state, modelingP(Chunk|Q,m t), which enables adaptive evidence discovery across reasoning steps.
4.5.3 Analysis of Staged Optimization Strategies
Table 4 analyzes the impact of each training stage. Memory Compression (Stage-2) achieves performance
comparable to Naive Chunking (Stage-1) with reduced token usage, indicating that compression alone requires
2k 4k 8k 16k 32k 64k 128k
Context Length (Tokens)2030405060708090100QA Accuracy (%)
Ratio 2x Ratio 4x (Ours) Ratio 8x Ratio 16x
Figure 4:QA Accuracy across varying context
lengths under different compression ratios. The 4×
ratio (Ours) achieves the optimal balance, matching
the stability of 2×while significantly outperforming
aggressive compression (16×).Models / StagesEvaluation Metrics (sub-EM)
HotpotQA 2Wiki Avg.
Qwen2.5-3B-Instruct
⊢Stage-1: Naive Chunking 38.28 35.16 36.72
⊢Stage-2: Memory Compression 39.84 36.72 38.28
Stage-3: w/ SFT 60.16 58.59 59.38
Stage-3: w/ RL 68.75 64.84 66.80
Stage-3: w/ End-to-End RL70.31 67.19 68.75
Table 4:Ablation study of the staged optimization
process. The base model is Qwen2.5-3B-Instruct with
a fixed context length of 16k tokens.
11

5. Conclusion
further alignment. Stage-3 SFT yields a notable improvement (+21.10 sub-EM) by learning basic interaction
patterns, but is surpassed by RL Optimization, which better supports multi-hop navigation and error correction.
The best performance (68.75 Avg. sub-EM) is obtained with End-to-End RL, where joint optimization enables
gradients to reach the compressor, encouraging reasoning-aware representations and validating the need
for unified perception–reasoning training. We further provide a training convergence analysis for the joint
optimization stage in Appendix B.
5 Conclusion
We introduce LycheeMemory, a cognitively inspired framework that enables efficient long-context reasoning
by mimicking the human memory’s division into long-term storage and dynamic working memory. Our
method integrates a Compressor , aGate , and a Reasoner : we jointly optimize the Compressor
andReasoner through end-to-end reinforcement learning, and train the Gate separately as a classifier.
Experimental results demonstrate that the LycheeMemory w/o Gate ablation can reach up to 82% normalized
sub-EM accuracy on multi-hop benchmarks and scales context length to 1.75M tokens, while the full model
provides a favorable accuracy–efficiency trade-off. Compared to MemAgent, LycheeMemory provides a 2 ×
reduction in peak GPU memory and a 6 ×inference speedup. Overall, LycheeMemory offers an efficient
solution for ultra-long context modeling.
12

REFERENCES
References
Richard C Atkinson and Richard M Shiffrin. Human memory: A proposed system and its control processes.
InPsychology oflearning andmotivation, volume 2, pp. 89–195. Elsevier, 1968.
Yushi Bai, Xin Lv, Jiajie Zhang, Hongchang Lyu, Jiankai Tang, Zhidian Huang, Zhengxiao Du, Xiao Liu,
Aohan Zeng, Lei Hou, et al. Longbench: A bilingual, multitask benchmark for long context understanding.
InProceedings ofthe62nd annual meeting oftheassociation forcomputational linguistics (volume 1:
Long papers), pp. 3119–3137, 2024.
Iz Beltagy, Matthew E Peters, and Arman Cohan. Longformer: The long-document transformer. arXiv preprint
arXiv:2004.05150, 2020.
Alexis Chevalier, Alexander Wettig, Anirudh Ajith, and Danqi Chen. Adapting language models to compress
contexts. In The2023 Conference onEmpirical Methods inNatural Language Processing , 2023. URL
https://openreview.net/forum?id=kp1U6wBPXq.
Prateek Chhikara, Dev Khant, Saket Aryan, Taranjeet Singh, and Deshraj Yadav. Mem0: Building production-
ready ai agents with scalable long-term memory, 2025. URL https://arxiv.org/abs/2504.
19413.
Gheorghe Comanici, Eric Bieber, Mike Schaekermann, Ice Pasupat, Noveen Sachdeva, Inderjit Dhillon, Marcel
Blistein, Ori Ram, Dan Zhang, Evan Rosen, et al. Gemini 2.5: Pushing the frontier with advanced reasoning,
multimodality, long context, and next generation agentic capabilities. arXiv preprint arXiv:2507.06261 ,
2025.
Chenlong Deng, Zhisong Zhang, Kelong Mao, Shuaiyi Li, Tianqing Fang, Hongming Zhang, Haitao Mi,
Dong Yu, and Zhicheng Dou. Unigist: Towards general and hardware-aligned sequence-level long context
compression. In TheThirty-ninth Annual Conference onNeural Information Processing Systems , 2025.
URLhttps://openreview.net/forum?id=1C4mXyh31p.
Sabri Eyuboglu, Ryan Ehrlich, Simran Arora, Neel Guha, Dylan Zinsley, Emily Liu, Will Tennien, Atri
Rudra, James Zou, Azalia Mirhoseini, et al. Cartridges: Lightweight and general-purpose long context
representations via self-study. arXiv preprint arXiv:2506.06266, 2025.
Chaochen Gao, Xing W, Qi Fu, and Songlin Hu. Quest: Query-centric data synthesis approach for long-context
scaling of large language model. In TheThirteenth International Conference onLearning Representations ,
2025. URLhttps://openreview.net/forum?id=sAYnDWaGd5.
Daya Guo, Dejian Yang, Haowei Zhang, Junxiao Song, Ruoyu Zhang, Runxin Xu, Qihao Zhu, Shirong Ma,
Peiyi Wang, Xiao Bi, et al. Deepseek-r1: Incentivizing reasoning capability in llms via reinforcement
learning. arXiv preprint arXiv:2501.12948, 2025.
Bernal Jiménez Gutiérrez, Yiheng Shu, Weijian Qi, Sizhe Zhou, and Yu Su. From RAG to memory: Non-
parametric continual learning for large language models. In Forty-second International Conference on
Machine Learning, 2025. URLhttps://openreview.net/forum?id=LWH8yn4HS2.
Shibo Hao, Sainbayar Sukhbaatar, DiJia Su, Xian Li, Zhiting Hu, Jason Weston, and Yuandong Tian. Training
large language models to reason in a continuous latent space. CoRR , abs/2412.06769, 2024. doi: 10.48550/
ARXIV .2412.06769. URLhttps://doi.org/10.48550/arXiv.2412.06769.
Xanh Ho, Anh-Khoa Duong Nguyen, Saku Sugawara, and Akiko Aizawa. Constructing a multi-hop qa dataset
for comprehensive evaluation of reasoning steps. arXiv preprint arXiv:2011.01060, 2020.
Cheng-Ping Hsieh, Simeng Sun, Samuel Kriman, Shantanu Acharya, Dima Rekesh, Fei Jia, and Boris Ginsburg.
Ruler: What’s the real context size of your long-context language models? In First Conference onLanguage
Modeling, 2024.
Edward J Hu, yelong shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, and Weizhu
Chen. LoRA: Low-rank adaptation of large language models. In International Conference onLearning
Representations, 2022. URLhttps://openreview.net/forum?id=nZeVKeeFYf9.
Gautier Izacard and Edouard Grave. Leveraging passage retrieval with generative models for open domain
question answering. In Proceedings ofthe16th conference oftheeuropean chapter oftheassociation for
computational linguistics: main volume, pp. 874–880, 2021.
13

REFERENCES
Bowen Jin, Hansi Zeng, Zhenrui Yue, Jinsung Yoon, Sercan Arik, Dong Wang, Hamed Zamani, and Jiawei
Han. Search-r1: Training llms to reason and leverage search engines with reinforcement learning. arXiv
preprint arXiv:2503.09516, 2025.
Vladimir Karpukhin, Barlas Oguz, Sewon Min, Patrick SH Lewis, Ledell Wu, Sergey Edunov, Danqi Chen, and
Wen-tau Yih. Dense passage retrieval for open-domain question answering. In EMNLP (1), pp. 6769–6781,
2020.
Angelos Katharopoulos, Apoorv Vyas, Nikolaos Pappas, and François Fleuret. Transformers are rnns: Fast
autoregressive transformers with linear attention. In International conference onmachine learning , pp.
5156–5165. PMLR, 2020.
Patrick Lewis, Ethan Perez, Aleksandra Piktus, Fabio Petroni, Vladimir Karpukhin, Naman Goyal, Heinrich
Küttler, Mike Lewis, Wen-tau Yih, Tim Rocktäschel, et al. Retrieval-augmented generation for knowledge-
intensive nlp tasks. Advances inneural information processing systems, 33:9459–9474, 2020.
Yuhong Li, Yingbing Huang, Bowen Yang, Bharat Venkitesh, Acyr Locatelli, Hanchen Ye, Tianle Cai,
Patrick Lewis, and Deming Chen. SnapKV: LLM knows what you are looking for before generation. In
The Thirty-eighth Annual Conference onNeural Information Processing Systems , 2024. URL https:
//openreview.net/forum?id=poE54GOq2l.
Adam Liska, Tomas Kocisky, Elena Gribovskaya, Tayfun Terzi, Eren Sezener, Devang Agrawal, Cyprien
De Masson D’Autume, Tim Scholtes, Manzil Zaheer, Susannah Young, et al. Streamingqa: A benchmark
for adaptation to new knowledge over time in question answering models. In International Conference on
Machine Learning, pp. 13604–13622. PMLR, 2022.
Jiaheng Liu, Dawei Zhu, Zhiqi Bai, Yancheng He, Huanxuan Liao, Haoran Que, Zekun Wang, Chenchen
Zhang, Ge Zhang, Jiebin Zhang, et al. A comprehensive survey on long context language modeling. arXiv
preprint arXiv:2503.17407, 2025.
Carlo Merola and Jaspinder Singh. Reconstructing context: Evaluating advanced chunking strategies for
retrieval-augmented generation, 2025. URLhttps://arxiv.org/abs/2504.19754.
Charles Packer, Vivian Fang, Shishir G. Patil, Kevin Lin, Sarah Wooders, and Joseph E. Gonzalez. Memgpt:
Towards llms as operating systems. CoRR , abs/2310.08560, 2023. URL https://doi.org/10.
48550/arXiv.2310.08560.
Yuqiao Tan, Shizhu He, Huanxuan Liao, Jun Zhao, and Kang Liu. Dynamic parametric retrieval augmented
generation for test-time knowledge enhancement, 2025. URL https://arxiv.org/abs/2503.
23895.
Fanqi Wan, Weizhou Shen, Shengyi Liao, Yingcheng Shi, Chenliang Li, Ziyi Yang, Ji Zhang, Fei Huang, Jin-
gren Zhou, and Ming Yan. Qwenlong-l1: Towards long-context large reasoning models with reinforcement
learning. arXiv preprint arXiv:2505.17667, 2025.
Maurice Weber, Dan Fu, Quentin Anthony, Yonatan Oren, Shane Adams, Anton Alexandrov, Xiaozhong Lyu,
Huu Nguyen, Xiaozhe Yao, Virginia Adams, et al. Redpajama: an open dataset for training large language
models. Advances inneural information processing systems, 37:116462–116492, 2024.
Orion Weller, Michael Boratko, Iftekhar Naim, and Jinhyuk Lee. On the theoretical limitations of embedding-
based retrieval, 2025. URLhttps://arxiv.org/abs/2508.21038.
Chaojun Xiao, Pengle Zhang, Xu Han, Guangxuan Xiao, Yankai Lin, Zhengyan Zhang, Zhiyuan Liu, and
Maosong Sun. InfLLM: Training-free long-context extrapolation for LLMs with an efficient context
memory. In TheThirty-eighth Annual Conference onNeural Information Processing Systems , 2024. URL
https://openreview.net/forum?id=bTHFrqhASY.
Guangxuan Xiao, Yuandong Tian, Beidi Chen, Song Han, and Mike Lewis. Efficient streaming language
models with attention sinks. arXiv preprint arXiv:2309.17453, 2023.
An Yang, Bowen Yu, Chengyuan Li, Dayiheng Liu, Fei Huang, Haoyan Huang, Jiandong Jiang, Jianhong
Tu, Jianwei Zhang, Jingren Zhou, et al. Qwen2. 5-1m technical report. arXiv preprint arXiv:2501.15383 ,
2025a.
14

REFERENCES
Qwen: An Yang, Baosong Yang, Beichen Zhang, Binyuan Hui, Bo Zheng, Bowen Yu, Chengyuan Li, Dayiheng
Liu, Fei Huang, Haoran Wei, Huan Lin, Jian Yang, Jianhong Tu, Jianwei Zhang, Jianxin Yang, Jiaxi Yang,
Jingren Zhou, Junyang Lin, Kai Dang, Keming Lu, Keqin Bao, Kexin Yang, Le Yu, Mei Li, Mingfeng Xue,
Pei Zhang, Qin Zhu, Rui Men, Runji Lin, Tianhao Li, Tianyi Tang, Tingyu Xia, Xingzhang Ren, Xuancheng
Ren, Yang Fan, Yang Su, Yichang Zhang, Yu Wan, Yuqiong Liu, Zeyu Cui, Zhenru Zhang, and Zihan Qiu.
Qwen2.5 technical report, 2025b. URLhttps://arxiv.org/abs/2412.15115.
Zhilin Yang, Peng Qi, Saizheng Zhang, Yoshua Bengio, William Cohen, Ruslan Salakhutdinov, and Christo-
pher D Manning. Hotpotqa: A dataset for diverse, explainable multi-hop question answering. In Proceedings
ofthe2018 conference onempirical methods innatural language processing, pp. 2369–2380, 2018.
Hongli Yu, Tinghong Chen, Jiangtao Feng, Jiangjie Chen, Weinan Dai, Qiying Yu, Ya-Qin Zhang, Wei-Ying
Ma, Jingjing Liu, Mingxuan Wang, et al. Memagent: Reshaping long-context llm with multi-conv rl-based
memory agent. arXiv preprint arXiv:2507.02259, 2025.
Peitian Zhang, Zheng Liu, Shitao Xiao, Ninglu Shao, Qiwei Ye, and Zhicheng Dou. Long context compression
with activation beacon. In The Thirteenth International Conference onLearning Representations , 2025.
URLhttps://openreview.net/forum?id=1eQT9OzfNQ.
Zhenyu Zhang, Ying Sheng, Tianyi Zhou, Tianlong Chen, Lianmin Zheng, Ruisi Cai, Zhao Song, Yuandong
Tian, Christopher Re, Clark Barrett, Zhangyang Wang, and Beidi Chen. H2o: Heavy-hitter oracle for
efficient generative inference of large language models. In Thirty-seventh Conference onNeural Information
Processing Systems, 2023. URLhttps://openreview.net/forum?id=RkRrPp7GKO.
Chujie Zheng, Shixuan Liu, Mingze Li, Xiong-Hui Chen, Bowen Yu, Chang Gao, Kai Dang, Yuqiong Liu, Rui
Men, An Yang, et al. Group sequence policy optimization. arXiv preprint arXiv:2507.18071, 2025.
Wangchunshu Zhou, Yuchen Eleanor Jiang, Peng Cui, Tiannan Wang, Zhenxin Xiao, Yifan Hou, Ryan
Cotterell, and Mrinmaya Sachan. Recurrentgpt: Interactive generation of (arbitrarily) long text. arXiv
preprint arXiv:2305.13304, 2023.
Zijian Zhou, Ao Qu, Zhaoxuan Wu, Sunghwan Kim, Alok Prakash, Daniela Rus, Jinhua Zhao, Bryan
Kian Hsiang Low, and Paul Pu Liang. Mem1: Learning to synergize memory and reasoning for efficient
long-horizon agents, 2025. URLhttps://arxiv.org/abs/2506.15841.
15

A. Implementation Details
Algorithm 1Joint Policy Optimization for LycheeMemory (GSPO)
Require: Joint policy πϑ={π comp, πreason}, reference model πref(frozen), dataset D, group size G, clipping
ϵ, KL coefficientβ
Ensure:Optimized parametersϑ
1:whilenot convergeddo
2:Sample document–query pair(D, Q)∼ D
3:forg= 1toGdo▷Group sampling for the same(D, Q)
4:Sample memoryΘ g∼π comp(· |D)
5:Sample answerA g∼π reason(· |Θ g, Q)
6:Define trajectoryy g= (Θ g, Ag)
7:Compute rewardˆr g=R(Q, A g)
8:Compute KL penaltyd g=D KL(πϑ(yg)∥π ref(yg))
9:r g←ˆrg−βd g
10:end for
11:{ ˆAg}G
g=1←GROUPNORM({r g}G
g=1)
12:forg= 1toGdo
13:ρ g←πϑ(yg)
πϑold(yg)
14:J g←min
ρgˆAg,clip(ρ g,1−ϵ,1 +ϵ) ˆAg
15:end for
16:ϑ←ϑ+η∇ ϑ1
GPG
g=1Jg
17:end while
A Implementation Details
This appendix provides the technical specifications necessary for reproducing LycheeMemory. We detail the
three-stage training pipeline: (1) Pre-training of the Compressor withsynthetic supervision(QA pairs
generated via self-annotation), (2) Joint Reinforcement Learning of theCompressorandReasoner, and
(3)supervisedtraining of theGateas a binary classifier.
All models are initialized from the Qwen2.5-3B-Instruct and Qwen2.5-7B-Instruct base models. We utilize 2
×NVIDIA A100 (80GB) GPUs for training.
A.1 Stage 1: Compressor Pre-training
The objective of the Compressor is to encode textual information into the latent space of memory tokens.
We employ a random compression ratio α∈ {2,4,8,16} , meaning one memory token is inserted for every α
text tokens.
Data Construction.We sample up to 1B tokens from the RedPajama Weber et al. (2024), and train on
approximately 160M effective tokens. For each document, we create splits of sizes 2048, 4096, and 8192,
denoted as Ci. We use self-annotation to generate synthetic Question-Answer pairs(i.e., synthetic supervision),
serving as training targets for reconstruction and comprehension tasks.
Configuration.We train a LoRA adapter ( Ψcomp) for the Compressor with a rank of r= 64 and a
LoRA alpha of αlora= 128 . The larger rank is selected to ensure sufficient representation capacity for the
compression task. We use the AdamW optimizer with an initial learning rate of 5e−5 and a cosine annealing
schedule, with the maximum learning rate set to 1e−4 . The batch size is set to 8, and training proceeds for
5,000 steps.
A.2 Stage 2: Joint Reinforcement Learning Optimization
We employ the GSPO algorithm for training. The full procedure is summarized in Algorithm 1. The chunk
size is set to 4096, the rollout batch size to 128, the group size Gto 12, and the update batch size to 16. The
KL divergence coefficient βis set to 1e−3 . We use the AdamW optimizer. Since LoRA is the optimization
target, we set the learning rate to 3e−5 with a linear warmup scheduler over 10 steps. In our runs, the joint
16

A.3 Stage 3: Gate Module Training
optimization typically converges within 150 optimizer update steps and takes about three days of wall-clock
time on 2×A100 (80GB).
Reward Configuration.During training, we employ a strict rule-based reward validator to prevent reward
hacking. We extract tokens within the <answer></answer> tags of the final output. If the extracted
answer matches the ground truth exactly, every update step in the trajectory receives a reward of 1.0; otherwise,
the reward is 0.0. We adopt this stricter validator during RL to avoid exploiting normalization artifacts that are
acceptable for evaluation.
Dataset Construction.We follow the dataset construction methodology of MemAgent. Each training
sample consists of 130 documents from HotpotQA, with a total token length of approximately 20K. We
thoroughly cleaned the dataset by filtering out questions where Qwen2.5-7B-Instruct could achieve a 100%
Best-Of-2 score without any context (zero-shot). We selected the top 32,768 processed samples as our training
set. Similarly, we synthesized 192 samples from the HotpotQA validation set. For extrapolation testing, we
used the same pipeline to synthesize test sets with varying context lengths, where the number of Wikipedia
entries ranges from 50 to 3,200, corresponding to context lengths from approximately 7K to 1.75 million
tokens.
A.3 Stage 3: Gate Module Training
TheGate is trained separately as a binary classifier to determine whether a memory block has retrieval and
reasoning value given the current query and working memory.
Label Assignment.Training data is derived from the rollout process in the RL stage. For multi-hop questions,
chunk updates (i.e., memory block updates) containing supporting facts are labeled asPositive( y= 1 ). Chunk
updates containing no supporting facts are labeled asNegative(y= 0).
Objective.We minimize the Binary Cross-Entropy (BCE) loss. To mitigate the class imbalance problem
(where irrelevant paragraphs far outnumber relevant ones), we apply a positive class weight of pos_weight=
3.0.
Configuration.The Gate LoRA adapter ( Ψgate) uses a smaller rank of r= 16 . We train for 3 epochs with a
learning rate of5e−5. During inference, the gating thresholdτis empirically set to 0.5.
A.4 Evaluation and Baselines
Evaluation Metrics.During evaluation, we report normalized sub-EM (Exact Match). We normalize both
the model answer and the ground truth (e.g., removing definite articles, ignoring case differences) and compute
a sub-EM score. This means if an answer contains all elements of the standard answer, it is considered correct.
When an answer consists of multiple parts, the score corresponds to the proportion of correct parts provided.
Long-Context Benchmarks.We evaluate our model on three long-context QA benchmarks, including
RULER-HQA Yang et al. (2018); Hsieh et al. (2024), 2WikiMultihopQA Ho et al. (2020), and Stream-
ingQA Liska et al. (2022). Below we describe the benchmark construction and our implementation details.
•RULER-HQA:A synthetic long-context HotpotQA benchmark derived from the RULER frame-
work. Similar to HotpotQA, each query has two supporting documents (gold evidence). We
construct long contexts by mixing the gold evidence with irrelevant distractor documents
(sourced from other samples). We build test sets with varying total context lengths ( N∈
{7k,14k,28k, ...,448k,896k,1.75M}), with randomized evidence positions.
•2WikiMultihopQA:A multi-hop QA dataset built from Wikipedia. We follow the same long-
context construction and evaluation pipeline as RULER-HQA: we take the query-relevant evidence
documents from 2WikiMultihopQA and mix them with distractor documents to reach the target
context length (14K/28K/56K in our experiments). We use the same chunking setting ( sz= 4096 ),
memory compression, dynamic recall, and normalized sub-EM evaluation.
•StreamingQA:A streaming QA benchmark designed for evaluation under continuously growing cor-
pora. For our long-context setting, we concatenate the documents of all questions into a single global
document of approximately 800k tokens, and evaluate each question by running LycheeMemory over
17

A.5 Storage and Computation Trade-off
this shared 800k context. We use the same chunking setting ( sz= 4096 ) and normalized sub-EM
evaluation.
Baselines.We compare LycheeMemory against three categories of strong baselines:
•RAG Agent:We implement a standard Retrieval-Augmented Generation agent using OpenAI’s
text-embedding-3-large as the retriever. The document is segmented into semantic chunks
(Wikipedia entries). For each query, the agent retrieves the top-8 most relevant chunks and feeds
them into the base model for generation.
•Search-R1:We use a Search-R1 agent trained on Qwen2.5-7B. Similar to the RAG agent, it uses
OpenAI’s text-embedding-3-large as the retriever and retrieves the top-8 most relevant
chunks. The agent then runs an iterative ReAct loop: generating a search query, retrieving context,
reasoning over the results, and deciding whether to search again or answer.
•MemAgent:We utilize the official implementation of MemAgent (Yu et al., 2025). MemAgent
processes long documents in fixed-size segments (set to 5k tokens by default in the official imple-
mentation). It maintains a global memory panel and employs a learnable policy to decide whether
to read, write, or overwrite information at each step. We align the prompt settings and base model
(Qwen2.5-7B-Instruct) with LycheeMemory to ensure a fair comparison of the memory mechanisms.
A.5 Storage and Computation Trade-off
A critical challenge in long-context processing is the management of GPU memory (VRAM). Even with our
efficient compression mechanism, maintaining a compressed KV-cache for extremely long sequences can
impose a prohibitive storage overhead.
Storage Analysis.Consider a scenario with a context length of N= 1.75M tokens using the Qwen2.5-3B
model (Hidden size d= 2304 , Layers l= 36 ). With a compression ratio of α= 4 , the system generates
approximately 437.5k latent memory tokens. Since Qwen2.5-3B utilizes Grouped Query Attention (GQA)
with 2 KV heads and 16 Query heads, the storage requirement for the KV-cache (in bfloat16 precision) is:
MKV≈2×l×d head×n kv×N
α×2bytes≈18.1GB
While this fits within the memory of high-end GPUs like the A100 (80GB), it still consumes a significant
portion of VRAM, limiting the space available for activations and larger batch sizes. Furthermore, our goal is
to enable efficient reasoning on consumer-grade hardware and to support scaling to even longer contexts (e.g.,
10M tokens), where static storage becomes prohibitive.
Optimization Strategy.To address this, we identify two potential strategies:
•Offloading:Temporarily offloading the compressed KV-cache to CPU RAM or NVMe SSDs and
swapping them back to GPU only during the retrieval phase.
•Just-in-Time (JIT) Compression:Storing only the raw text and using the Compressor to regen-
erate the latent representations on the fly when needed.
By default, LycheeMemory uses offline pre-compression and stores the compressed KV-cache for inference.
For contexts beyond 1M tokens, we optionally enable Just-in-Time (JIT) compression as an engineering
strategy to reduce storage overhead. While re-computing embeddings incurs a computational cost, it can be
more efficient than the I/O bottleneck of memory swapping. The compression process is a single parallel
forward pass, whereas the reasoning process is autoregressive. For a chunk of size 4096, compression
requires only 1 forward step. In contrast, generating a reasoning chain often requires hundreds of serial
steps. Therefore, the amortized computational overhead of on-the-fly compression is minimal compared to the
benefits of reduced memory footprint and improved scalability.
B Training Convergence Analysis
We address the potential concern regarding the stability of jointly optimizing the Compressor and Reasoner,
given the sparsity of reward signals in sequence generation tasks. Figure 5 presents the raw, unsmoothed
training reward curves over 50 checkpoints, comparing our Joint Optimization strategy against a baseline with
a Frozen Compressor.
18

C. Failure Mode Analysis
0 20 40 60 80 100 120 140
Training Compute0.450.500.550.600.650.700.750.800.85Training Reward
End-to-End
Frozen Compressor
Figure 5:Training reward curves (raw data). The blue line (Frozen Compressor) converges quickly but hits a
performance plateau. The red line (End-to-End) exhibits higher variance initially due to the exploration of the
compression policy but achieving higher rewards.
Observation.End-to-End RL curve (red) shows higher volatility in the early stages (Checkpoints 0-15)
compared to the Frozen Compressor (blue). This is expected, as the gradient updates must propagate through
the reasoning steps back to the compression module, causing shifts in the memory representation Θ. The
Frozen Compressor rapidly converges to a local optimum ( ≈0.67 ) but fails to improve further, as the reasoner
is limited by a static, suboptimal memory bank. In contrast, our method steadily climbs after the initial
adaptation phase, reaching a higher reward (≈0.7).
Conclusion.The empirical results demonstrate that despite the inherent variance in RL, the joint policy
successfully converges. The fluctuating but upward trend confirms that the Compressor is actively learning
to retain task-critical features that maximize the Reasoner’s success rate, validating the effectiveness of our
end-to-end optimization framework.
C Failure Mode Analysis
To gain deeper insights into the limitations of LycheeMemory, we conducted a manual error analysis on
128 randomly sampled incorrect instances from the HotpotQA and 2WikiMultihopQA validation sets. We
categorized the primary causes of failure into three dominant modes:Compression-Induced Hallucination,
Unidirectional Dependency Mismatch, andPremature Inference Anchoring.
C.1 Unidirectional Dependency Mismatch (35%)
The most significant source of error (approx. 35%) stems from the inherent limitation of the single-pass,
streaming architecture. In multi-hop reasoning, the relevance of an early piece of evidence often depends on
information that appears later in the document.
Mechanism.When the model encounters a critical clue (e.g., at Step 3), it may not be semantically similar
to the current query or working memory, causing the Gate to filter it out. Later (e.g., at Step 5), the model
discovers the bridge entity that makes the previous clue relevant. However, since the static memory has already
been processed and the model cannot backtrack, this information is permanently lost.
19

C.2 Premature Inference Anchoring (21%)
Case Study 1: The Late-Binding Problem
Query:What represents the nationality of the director of the film "The Blue Kite"?
• Step 3 (Context Chunk): "...Tian Zhuangzhuang was born in Beijing, China, and began his career..."
• Model Action: [Gate: Ignore]→The working memory contains no link to "Tian Zhuangzhuang" yet.
• Step 8 (Context Chunk): "...’The Blue Kite’ is a 1993 drama film directed by Tian Zhuangzhuang..."
• Model Action: [Gate: Retrieve]→Update Working Memory: "Director is Tian Zhuangzhuang."
•Reasoning Failure: The model now knows the director, but the information about his nationality (China) was in
Step 3, which was discarded. The model answers "Unknown" or hallucinates based on the name.
C.2 Premature Inference Anchoring (21%)
Approximately 21% of errors occur when the model aggressively acts on partial evidence, forming a correct-
looking but ultimately wrong conclusion early in the process. This creates a confirmation bias in the Working
Memory.
Mechanism.The Reasoner generates an intermediate answer based on a partial match (e.g., a shared
name). This incorrect entry in the Working Memory then dominates the attention mechanism, causing the
model to either ignore subsequent contradictory evidence or misinterpret it to fit the existing hypothesis.
Case Study 2: Premature Anchoring
Query:Which band’s lead singer also released the solo album "Euphoria"?
• Step 2 (Context Chunk): "...Enrique Iglesias released an album titled ’Euphoria’ in 2010..."
•Model Action: Update Working Memory: "Candidate: Enrique Iglesias (Solo Artist)." →Wrong Path. The
question asks for a band’s lead singer.
•Step 6 (Context Chunk): "...Def Leppard’s lead singer Joe Elliott released a projected titled..." (Irrelevant text
follows).
•Step 9 (Context Chunk): "...The band ’Morningwood’ features lead singer Chantal Claret..." (Target info
appears later).
•Reasoning Failure: The working memory is already anchored on Enrique Iglesias. The model stops actively
searching for "bands" or tries to justify why Enrique fits the description, ignoring the correct entity appearing
later.
C.3 Compression-Induced Hallucination (17%)
As analyzed in Appendix H.1, about 17% of errors are due toFeature Collapsewithin theCompressor.
Mechanism.High compression ratios can cause distinct entities with similar semantic embeddings (e.g.,
brothers, movies in the same franchise, dates close in time) to merge in the latent space. The Reasoner
retrieves a blurred representation, leading to attribute swapping.
Case Study 3: Attribute Swapping
Query:Who was born earlier, William Johnson or Wilson Johnson?
•Compressed Memory: Encodes "William... born 1856" and "Wilson... born 1860" into adjacent latent vectors.
• Retrieval: The Reasoner retrieves the block containing both.
•Reasoning Failure: Due to vector smoothing, the specific binding of dates to names is lost. The model outputs:
"Wilson was born in 1856,"effectively swapping the birth years.
C.4 Other Error Types (27%)
The remaining errors include:
20

D. Computational Complexity
•Context Overflow: The accumulation of too many potentially relevant chunks fills the working
memory context limit, flushing out early correct evidence.
•Instruction Misalignment: The model correctly retrieves evidence but fails to align the final answer
format with user instructions (e.g., answering Yes instead of a specific entity name).
D Computational Complexity
To rigorously evaluate the efficiency of LycheeMemory, we model the theoretical FLOPs required to process a
document of lengthNand generate an answer. We compare three distinct paradigms:
Full-Context.Processes the entire sequence simultaneously.
MemAgent.Processes the sequence in chunks, performing an autoregressive memory update forevery
chunk.
LycheeMemory (Ours):Compresses the sequence first, then employs a sparse, gated retrieval mechanism.
D.1 FLOPs Formulation
LetNbe the total document length, szbe the chunk size, LQbe the query length, and LAbe the output
generation length. The document is segmented into K=⌈N/sz⌉ chunks. We denote the model’s hidden
dimension as dand depth as l. The complexity of generating a sequence of length Lgengiven a prompt of
lengthL pmtis dominated by attention, scaling asO(l·d·(L pmt+Lgen)2).
Full-Context.The computational complexity is dominated by the global self-attention mechanism over the
entire sequence.
CFull≈ O 
l·d·(N+L Q+LA)2
≈ O(N2) (whenN≫L Q, LA)
The prefill stage processes N+L Qtokens, followed by decoding LAtokens. As Ngrows to millions, the
quadratic term makes this prohibitive.
MemAgent.MemAgent adopts a linear scanning approach but incurs a heavy constant factor due to forced
memory updates. It performs generation foreverychunk. Let the input per step be Lin=LQ+Mem size+sz ,
and output beL out=Mem update . The total FLOPs sums over allKsteps:
CMemAgent ≈K× O 
l·d·(L in+L out)2
=N
sz×C gen≈ O(N)
Although asymptotically linear, the constant Cgeninvolves a full generation process (KV-cache read + autore-
gressive write) for every chunk, leading to a steep increase in computational cost.
LycheeMemory (Ours).Our method decouples processing into efficient compression and sparse reasoning.
We utilize a compression ratio ofα= 4.
•Phase 1: Compression. The model processes chunks in parallel to encode KV-cache. Since there is
no autoregressive decoding, the cost is proportional to the input tokens.
Ccomp≈ O(l·d·N)
•Phase 2: Gate. The Gate scores all Kblocks. Due to compression, the effective sequence length is
N/α. The Gate requires only a single forward pass per block.
Cgate≈ O
l·d·N
α
21

D.2 Quantitative Comparison
10000 20000 30000 40000 50000 60000
Context Length (tokens)108109FLOPs (log scale)Full-Context
MemAgent
Ours (Adaptive Recall)
Figure 6:FLOPs Scaling Analysis (Log Scale). We compare the estimated computational cost across context
lengths from 8K to 64K tokens using a logarithmic FLOPs axis. For LycheeMemory, the effective recall ratio
is assumed to decrease linearly from 100% to 40% as context length increases, reflecting increasingly selective
memory access under long contexts. FLOPs are analytically estimated under simplified assumptions and are
intended to illustrate relative scaling trends rather than exact runtime measurements.
• Phase 3: Reasoning. TheReasoneris activated only for the top-Trelevant chunks (T≪K).
Creason≈T× O 
l·d·(L in+L out)2
=T×C gen
The total cost isC Ours=C comp+C gate+C reason . Comparing the dominant terms:
•MemAgent:N
sz×C gen(Dense Generation)
•LycheeMemory:O(N) +T×C gen(Sparse Generation)
Since T≪N/sz (e.g., retrieving only 10% of chunks), LycheeMemory significantly reduces the number of
expensive generation calls, resulting in a much flatter scaling curve.
D.2 Quantitative Comparison
Figure 6 illustrates the FLOPs scaling behavior under increasing context lengths. The Full-Context baseline
exhibits the expected O(N2)complexity due to dense self-attention, which appears as a straight line with a
steep slope under the logarithmic FLOPs axis, indicating rapidly growing computational cost. In contrast,
both MemAgent and LycheeMemory achieve linear scaling with respect to context length. Despite sharing
the same asymptotic complexity, LycheeMemory consistently incurs lower FLOPs than MemAgent across
all evaluated settings. This improvement is primarily attributed to theCompress-then-Reasonparadigm:
first, the input sequence is compressed by 4×, substantially reducing the number of tokens processed by the
gating mechanism; second, unlike MemAgent which performs mandatory write operations for every chunk,
LycheeMemory employs adaptive gating to activate the computationally heavy Reasoner only for a small
fraction of chunks, resulting in a significantly smaller constant factor in practice.
22

E. Out-of-Distribution (OOD) Generalization Analysis
ModelGovReport MultiNews
R-1 R-2 R-L Avg. R-1 R-2 R-L Avg.
Qwen2.5-7B-Instruct 30.9111.68 15.20 19.26 46.64 12.01 28.33 28.99
MemAgent-7B 30.28 12.3715.3719.34 48.4914.4130.91 31.27
LycheeMemory-7B(ours) 30.1213.0815.0719.42 47.4315.2830.33 31.01
Table 5:Zero-shot performance on LongBench summarization tasks. Despite being trained for QA, Ly-
cheeMemory achieves competitive performance compared to MemAgent, demonstrating strong generalization
capabilities.
WM Capacity RULER-HQA Context Length Avg.
(LWM) 7K 14K 28K 56K Score
1024 82.03 79.69 78.91 77.34 79.49
2048 82.03 78.91 78.91 77.34 79.30
3072 81.25 78.91 77.34 75.78 78.32
4096 80.47 77.34 76.56 75.00 77.34
Table 6:Ablation results on Working Memory Capacity. The standard capacity of 1024 achieves the best
performance, indicating that larger buffers do not necessarily improve reasoning.
E Out-of-Distribution (OOD) Generalization Analysis
While LycheeMemory is primarily optimized for multi-hop QA, we also test whether its compressed memory
bank Θtransfers to other long-context formats. We conduct OOD experiments on long-document sum-
marization tasks from LongBench (Bai et al., 2024), which differ from the QA-based training setup. We
consider GovReport (summarizing lengthy government reports) and MultiNews (summarizing multiple news
documents). We report ROUGE-1 (R-1), ROUGE-2 (R-2), and ROUGE-L (R-L) scores, and compare our
method against the base model Qwen2.5-7B-Instruct as well as MemAgent-7B.
Results and Discussion.As shown in Table 5, LycheeMemory exhibits strong zero-shot generalization.
These results suggest that the compressed memory bank Θconstructed by LycheeMemory captures transferable
semantic features that support both targeted extraction (QA) and global aggregation (summarization).
F Ablation on Working Memory Capacity
In the dynamic recall and reasoning phase, the capacity of the Working Memory ( m) is a critical hyperparameter.
Given that we segment documents into chunks of size Lchunk= 4096 , the working memory capacity determines
the maximum buffer size available for the active context before synthesis. We investigate the impact of varying
the working memory capacity limit ( LWM∈ {1024,2048,3072,4096} ) on the RULER-HQA benchmark
across increasing context lengths ranging from 7K to 56K tokens.
Results and Analysis.To ensure fairness and isolate the effect of capacity, we excluded the Gate mechanism
in this ablation, as it was specifically optimized for 4096-token chunks. As shown in Table 6, we observe
that performance does not improve with increased working memory capacity; in fact, LWM= 1024 yields
the highest average accuracy of 79.49%. The results suggest that expanding the working memory budget
beyond the necessary chunk size tends to introduce excessive noise tokens. This accumulated noise distracts
the model’s attention mechanism during the final answer synthesis, leading to a degradation in reasoning
precision rather than an improvement.
23

G. Dynamic Evolution of Working Memory
0 5 10 15 20 25 30
Reasoning Step (Turn)0100200300400500600Working Memory Length
Avg. Working Memory Length
Overall Mean (380)
Figure 7:Evolution of Working Memory length. The length grows initially as evidence is collected but
stabilizes at a peak of ≈500 tokens around Turn 23. This demonstrates that LycheeMemory effectively
manages its context budget without unbounded growth.
G Dynamic Evolution of Working Memory
To verify the efficiency of our reasoning mechanism, we tracked the token usage of the Working Memory state
macross 32 reasoning steps. Figure 7 presents the average length evolution on 128 samples from the 128K
context validation set.
Analysis.As shown in Figure 7, the working memory length exhibits a clear saturation behavior rather
than monotonically increasing with reasoning steps. While the memory expands during the early stages to
accumulate relevant evidence, it gradually stabilizes at approximately 503 tokens—well below the predefined
capacity limit of 1024. This indicates that the model does not passively append retrieved information, but
instead learns to actively regulate its memory state by retaining only task-critical content. Such learned
memory management effectively prevents unbounded context growth and ensures stable, efficient reasoning
over long interaction horizons.
H Case Analysis
H.1 Hallucination via Feature Collapse
To understand the failure mode of high compression, we conduct a qualitative analysis using a constructed
synthetic narrative,Chronicle of the Four Johnson Brothers, which is dense with confounding entities (similar
names and dates). We compare the generated responses of the4×(Ours) and16×models.
Discussion.As shown in Table 7, while the 16× model correctly retrieves high-level entities (names), it fails
atattribute binding . It incorrectly assigns the birth year “1860” to both wives. This suggests that aggressive
compression causes feature collapse in the latent space, where distinct numerical tokens (1860 vs. 1865) closer
in proximity are averaged into a single representation. In multi-step reasoning, these small hallucinations
accumulate, leading to the rapid performance decay observed in the 16× curve (Figure 4). The 2×,4×, and
8×models correctly identified the dates in this test, confirming that moderate compression preserves the
fidelity required to distinguish fine-grained details.
24

H.2 Associative Reasoning and Self-Correction
Query: Distinguish between the wives of William Henry Johnson and Wilson Harold Johnson.
Context Facts:
1. William’s wife: Elizabeth Ann Smith, born1860.
2. Wilson’s wife: Elizabeth Marie Smith, born1865.
Model Response (α= 16, High Compression):
"William’s wife was Elizabeth Ann... Wilson’s wife was Elizabeth Marie... Both women were born in 1860, but their
names are distinct."
(Error: Hallucinated Wilson’s wife’s birth year by merging it with William’s wife’s.)
Model Response (α= 4):
"Elizabeth Ann was born in1860, while Elizabeth Marie was born in1865. They are cousins."
Table 7:Case study on fine-grained information retrieval. The 16× model suffers from attribute merging
(hallucination), while4×retains precision.
H.2 Associative Reasoning and Self-Correction
Question: The city that contains the Yunnan Provincial Museum is also known by what nickname?
Chunk k
... The Yunnan Provincial Museum is located in Kunming, the capital city of Yunnan Province in southwest China.
The museum features exhibitions on Yunnan’s history, ethnic groups, and archaeology. ...
Working memory k:
- The Yunnan Provincial Museum is located in Kunming City, Yunnan.
- Yangzonghai Lake is located near Kunming, Yunnan, in southern China.
Chunk k+1
... Dian Lake is a large freshwater lake on the Yunnan–Guizhou Plateau near Kunming. Dian Lake is sometimes
described as the “Sparkling Pearl Embedded in a Highland”. ...
Working memory k+1:
- The Yunnan Provincial Museum is located in Kunming City, Yunnan.
- Dian Lake is a large lake located on the Yunnan-Guizhou Plateau close to Kunming, Yunnan, its nickname is
“Sparkling Pearl Embedded in a Highland”.
- Yangzonghai Lake is located near Kunming, Yunnan, in southern China.
The city that contains the Yunnan Provincial Museum might be “Sparkling Pearl Embedded in a Highland”.
Chunk k+2
... Kunming has a mild climate throughout the year and is widely known as the “Spring City”. The city serves as a
major cultural and transportation hub in Yunnan. ...
Working memory k+2:
- The Yunnan Provincial Museum is located in Kunming City, Yunnan.
- Dian Lake is a large lake located on the Yunnan-Guizhou Plateau close to Kunming, Yunnan, its nickname is
“Sparkling Pearl Embedded in a Highland”.
- Kunming is also known as the “Spring City” due to its weather.
- Yangzonghai Lake is located near Kunming, Yunnan, in southern China.
The new content provides Kunming’s exact nickname. Therefore, the city that contains the Yunnan Provincial
Museum is known as the “Spring City”.
As illustrated in the case study above, the model demonstrates robust associative reasoning andself-correction
capabilities. Initially, the agent identifies “Kunming” as the pivotal entity by associating the Yunnan Provincial
Museum with its location (Chunk k). It then forms a tentative hypothesis after processing memory block k+ 1 ,
incorrectly inferring that the city’s nickname might be “Sparkling Pearl” based on the proximal descriptions of
Dian Lake. However, the reasoning remains flexible; upon retrieving memory block k+ 2 which explicitly
describes Kunming as the "Spring City," the agent successfully detects the conflict. It differentiates the
25

I. Comparison with RAG
distraction (the lake’s nickname) from the city’s actual alias and rectifies its working memory, effectively
overriding the previous tentative inference with the verified fact.
I Comparison with RAG
RAG and LycheeMemory address different bottlenecks in long-context reasoning. RAG enables fast retrieval
with high recall via approximate nearest neighbor search, making it well-suited for large external corpora.
Our goal is not to replace RAG, but to provide an alternative long-context processing paradigm based on
compressed memory and state-dependent retrieval: RAG typically ranks chunks by query-only similarity
and may miss late-hop evidence that becomes relevant only after intermediate entities are discovered, while
LycheeMemory conditions theGateon both the query and the evolving working memorym(Table 1). The
two approaches are complementary: documents retrieved by RAG can be treated as additional context streams,
compressed intoΘ, and then reasoned over by the same dynamic recall and reasoning workflow.
26