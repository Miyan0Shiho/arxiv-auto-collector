# LLavaCode: Compressed Code Representations for Retrieval-Augmented Code Generation

**Authors**: Daria Cherniuk, Nikita Sukhorukov, Nikita Sushko, Daniil Gusak, Danil Sivtsov, Elena Tutubalina, Evgeny Frolov

**Published**: 2025-10-22 14:49:21

**PDF URL**: [http://arxiv.org/pdf/2510.19644v1](http://arxiv.org/pdf/2510.19644v1)

## Abstract
Retrieval-augmented generation has emerged as one of the most effective
approaches for code completion, particularly when context from a surrounding
repository is essential. However, incorporating context significantly extends
sequence length, leading to slower inference - a critical limitation for
interactive settings such as IDEs. In this work, we introduce LlavaCode, a
framework that compresses code into compact, semantically rich representations
interpretable by code LLM, enhancing generation quality while reducing the
retrieved context to only a few compressed single-token vectors. Using a small
projector module we can significantly increase the EM and ES metrics of coding
model with negligible latency increase. Our experiments demonstrate that
compressed context enables 20-38% reduction in Time-to-First-Token (TTFT) on
line completion tasks compared to full-RAG pipelines.

## Full Text


<!-- PDF content starts -->

LLAVACODE: COMPRESSEDCODEREPRESENTATIONS
FORRETRIEVAL-AUGMENTEDCODEGENERATION
Daria Cherniuk
Personalization Technologies
Moscow, Russia
kamikazizen@gmail.comNikita Sukhorukov
Personalization Technologies,
AIC
Moscow, Russia
niksukhorukov7@gmail.comNikita Sushko
Independent Researcher
Moscow, Russia
gorakievskaya@gmail.com
Daniil Gusak
Personalization Technologies,
AIC
Moscow, Russia
gusak.di18@physics.msu.ruDanil Sivtsov
Computational Intelligence Group
Moscow, Russia
sivtsovdt@gmail.comElena Tutubalina
Domain-specific NLP Group
Moscow, Russia
tutubalinaev@gmail.com
Evgeny Frolov
Personalization Technologies,
HSE University
Moscow, Russia
evfro@live.ru
October 23, 2025
ABSTRACT
Retrieval-augmented generation has emerged as one of the most effective approaches for code com-
pletion, particularly when context from a surrounding repository is essential. However, incorporating
context significantly extends sequence length, leading to slower inference—a critical limitation for
interactive settings such as IDEs. In this work, we introduce LlavaCode, a framework that compresses
code into compact, semantically rich representations interpretable by code LLM, enhancing genera-
tion quality while reducing the retrieved context to only a few compressed single-token vectors. Using
a small projector module we can significantly increase the EM and ES metrics of coding model with
negligible latency increase. Our experiments demonstrate that compressed context enables 20-38%
reduction in Time-to-First-Token (TTFT) on line completion tasks compared to full-RAG pipelines.
1 Introduction
Recently, more and more IDEs started to feature code completion as one of the central tools. Code editors such as
Windsurf1and Cursor2started integrating large language models (LLMs) to provide single- and multiline prediction,
which substantially improve developer productivity, but they also impose strict latency requirements: even small delays
in time-to-first-token (TTFT) break the interactive coding experience and using this feature becomes frustrating.
Additionally, retrieval-augmented generation (RAG) [ 1] has been widely adopted to improve both QA and completion
quality, since it allows models to incorporate external context such as documentation, relevant snippets of code or
function declarations into the prompt (Figure 1a). However, the additional tokens from retrieval significantly increase
1Windsurf homepage
2Cursor homepagearXiv:2510.19644v1  [cs.CL]  22 Oct 2025

APREPRINT- OCTOBER23, 2025
prompt processing time and, consequently, TTFT, making vanilla RAG less practical for latency-critical settings like
code completion.
A promising solution is context compression via embedding projection. Originally introduced in multimodal models
such as Flamingo [ 2] and LLaV A [ 3], these methods use a separate visual encoder and a lightweight projection module
to map input image embeddings into a small set of tokens for the language model. Subsequent works, such as xRAG [ 4],
extended this idea to textual retrieval, showing that compressed representations can match vanilla RAG performance
while reducing inference cost.
Despite this progress, no prior work has applied embedding projection to the code completion task, where the
latency–quality trade-off is especially severe. Furthermore, existing training objectives (e.g., cross-entropy) are poorly
aligned with developer-relevant code generation quality metrics such as Exact Match (EM) and Edit Similarity (ES),
limiting the effectiveness of current approaches. Additionally, we can incorporate other code modalities, such as
Abstract Syntax Trees (AST), into the retrieved embeddings to enrich the representations with syntactic information.
In this work, we address both challenges. We propose a LLaV A-like projection mechanism for incorporating retrieved
embeddings into model input. This projection mechanism is trained without unfreezing of LLM, combined with
reinforcement learning that directly optimizes EM and ES. This enables efficient context integration for code completion
without sacrificing prediction quality.
Our contributions are the following:
•To the best of our knowledge, our approach is the first to apply LLaV A-like embedding projection to code
completion taskswithoutembedder or LLM finetuning, resulting in significantly higher EM and ES scores
with negligible latency increase compared to base model, while maintaining 20-38% better latency compared
to full RAG.
•We’ve experimented with incorporating additional code modalities such as ASTs to investigate whether
alternative representations of code can improve representation quality.
• We’ve shown the limits of conventional LLM training with cross-entropy loss and proposed using Reinforce-
ment Learning to directly train the projector for target EM and ES metrics, without unfreezing the retrieved
context embedder or code-generating LLM.
All the code and weights for projector modules will be available under permissive license. Code will be published in
LlavaCode github repository3.
2 Related Work
2.1 Coding LLMs
StarCoder [ 5] introduced a family of code generation models, including larger LLMs optimized for code-centric dialogue
and smaller ones tailored for code completion. Trained on the permissively licensed The Stack dataset [ 6], these models
achieved strong performance, surpassing most prior approaches on both code completion and instruction-following
benchmarks. The Qwen-2.5-Coder series [ 7] represented another significant advancement in code-focused LLMs.
Trained on a proprietary mixture of data, the models were released in sizes ranging from 0.5B to 32B parameters and
were designed to support text completion, code chat, and fill-in-the-middle tasks.
In our work, we use these models as reader LLM baselines for RAG and evaluate our proposed system with Qwen-2.5-
Coder-1.5B as backbone LLMs.
2.2 Context compression methods
Despite decoder-only transformer optimizations such as KV-Caching [ 8] and more efficient attention implementations
like GQA [ 9], time per-token inference latency still scales linearly with context size. Since Retrieval Augmented
Generation [ 1] retrieves information from the knowledge base and puts it into the context of language models, this
increases the context size that needs to be processed and subsequently increases end-to-end latency.
In the paper xRAG [ 4] the authors propose an approach, which is similar to multi-modal language models training: they
push the embedding vector of the retrieved text from textual encoder through a lightweight projector layer to align it
with the reader model. The resulting architecture is trained in a two-stage manner. In the first stage, both the encoder
3https://github.com/KamikaziZen/LlavaCode
2

APREPRINT- OCTOBER23, 2025
and LLM are frozen, while the projection layer is trained with cross-entropy loss on paraphrases of the same document.
During the second stage, the projector is trained on a mix of tasks such as reading comprehension, open-domain QA
and summarization, adding self-distillation from RAG teacher via KL term alongside with usual negative log-likelihood
loss. Models trained in such way perform competitively with vanilla RAG systems, while being much more efficient
and having lower TTFT due to the reduction in prompt length.
Our approach is conceptually similar to xRAG method. By using a LLaV A-like projection from the encoder to the code
completion model, we compress the retrieved context and maintain good generation quality, while lowering the TTFT.
However, due to the specificity of our domain, we applied additional techniques to increase code-specific metrics and
quality of predictions. Furthermore, we train only the projector with both the encoder and reader LLM frozen in a
single stage manner.
2.3 Embedding models for code
Code-search embeddings are commonly obtained by converting a decoder-only language model into embedding model
by training them to produce last token embeddings for code search via contrastive learning. One such model is Qwen3-
Embedding-0.6B [ 10], which was converted from Qwen3-0.6B [ 11] model. Initialized from a powerful pretrained
decoder-only model, Qwen3-Embedding-0.6B shows competitive scores on MTEB [ 12] benchmarks among similarly
sized embedding models.
Additionally, some of the encoder models were trained not only on pure text and code data, but also on structured
graphs, retrieved from code, such as Data Flow Graphs (DFG) and Abstract Syntax Trees (AST). Examples of such
models are GraphCodeBERT [ 13] and UniXcoder [ 14] models, which joined both code, text and graph data to improve
representation quality for code-understanding and retrieval tasks.
We have evaluated representative models as encoders in our architecture to investigate how different modalities of code
effect the projection quality.
2.4 Reinforcement Learning in Language Modeling
Training language models solely for next-token prediction optimizes perplexity but not other objectives such as lack of
toxicity, aligning with human preferences, or – specifically for our task – Exact Match (EM) and Edit Similarity (ES)
scores.
In Self-Critical Sequence Training (SCST) paper [ 15] a variation of REINFORCE [ 16] with a baseline is applied to
train an image captioning model. SCST uses the reward of the sequence produced by the current model under the
test-time inference algorithm as the baseline, yielding an unbiased, lower-variance REINFORCE estimator.
In our work, we utilize the same REINFORCE-like approach as in SCST, but without baseline term. We directly
optimizeES+EMmetric, which leads to performance increase.
3 Methodology
3.1 Model architecture
To decrease the amount of tokens in the context of RAG reader model, we need to somehow compress the retrieved
information. In case of LlavaCode, we compress retrieved texts using an off-the-shelf text embedding models and then
use a small LLaV A-like projector to make it align better with the embeddings of the reader model.
To compress the retrieved context, we use text embedding model, which transforms a chunk of code into a single
embedding vector. This single vector is being passed through a projection layer, which converts this embedding into a
shape that is compatible with LLM embeddings. In our experiments, we take top-10 retrieved chunks per completion
and compress them into 10 embeddings, which are concatenated with the LLM embedding of the prompt (Figure 1b).
This leads to negligible latency increase (see Section 5 for more latency measurements), since we can directly retrieve
precomputed text projections from the RAG database, without the need to inference the encoder model and the projector
module.
For our main experiments, we use Qwen-2.5-Coder-1.5b as code-completion model with Qwen-3-Embedding-0.6B as
embedder. The projector follows the same architecture as projector of LLaV A [ 3]: an MLP, with GeLU [ 17] activation
function and a LayerNorm [ 18]. We selected two MLP architectures: a 2-layer and a 3-layer MLP. For more details on
projector architecture see Table 2 and Appendix B.
3

APREPRINT- OCTOBER23, 2025
R A G 
( c o s i n e / j a c c a r d / b m 2 5 )L L MC F C  T O K E NP R O M P T  T O K E N
(a) Vanilla RAG
L L M E N C O D E R 
( A S T ,  C F G ,  D F G ,  N C S )P R O J E C T O RR A G 
( c o s i n e / j a c c a r d / b m 2 5 )C F C  E M B E D D I N GP R O M P T  E M B E D D I N G (b) LlavaCode
Figure 1: Comparison between Vanilla RAG 1a and LlavaCode 1b architectures. Instead of retrieving text passages and
putting them into the context of the reader language model, LlavaCode uses a pretrained encoder to compress the text
representations and projects them into continuous tokens, thus, reducing the prompt processing time.
3.2 Cross-Entropy Issue
In previous works, both in textual and multimodal compression, training was carried out in two stages. On the first
stage, the projection layer is pretrained on a simple task to enable better alignment of the compressed embeddings with
a large language model. On this stage the model is freezed and only projection layer is unfreezed. On the second stage,
either the model and the projector or only the projector are trained on downstream tasks. In contrast, our approach uses
a single-stage training, omitting the pretraining stage. We’ve experimented with pretraining the projector, but this did
not yield any improvements. More information is available in Appendix D.
Most prior work on the related task—training a projection from encoder outputs into the embedding space of an LLM
— has relied on instruction-tuned models and QA datasets, and trained primarily with cross-entropy loss [ 19,3,20].
There are, however, notable exceptions. For example, xRAG incorporated KL divergence loss [ 4], reporting that it had
a greater impact on downstream performance than NLL loss. Another deviation from pure cross-entropy training is
Flamingo [2], which employed the two-term contrastive loss introduced in [21].
Cross-entropy (negative log-likelihood) is the standard objective for training autoregressive LLMs: it measures how
well the model’s predicted next-token distribution matches the target distribution. The formula for cross-entropy loss is
the following:
LCE(θ) =−1
TTX
t=1logp θ(yt|y1, . . . , y t−1).
In our experiments, we have found that relying solely on cross-entropy loss was insufficient, since it does not directly
correlate with EM and ES metrics. Exact Match measures the percentage of predictions that match the reference output
exactly, character for character. It is a strict metric that gives credit only for completely correct generations. Edit
Similarity measures the similarity between the prediction and the reference based on the minimum number of edits
(insertions, deletions, substitutions) needed to transform one into the other. It provides a softer evaluation by rewarding
partial correctness. These are sequence-length metrics, whereas cross-entropy is token-level, maximizing the likelihood
of the next token prediction given ground truth. Therefore, it is to be expected that optimizing only for cross-entropy led
to suboptimal results on key target metrics — EM and ES — even though it produced lower cross-entropy loss value
compared to the baseline model (see Table 1). These results motivated us to explore methods for directly optimizing
sequence-based metrics, including approaches from reinforcement learning.
3.3 REINFORCE
As noted in [ 15], deep generative models for text are typically trained to maximize the likelihood of the next ground-truth
word conditioned on the previous ground-truth word via backpropagation. This training paradigm is commonly referred
to as Teacher Forcing [ 22]. However, it introduces a discrepancy between training and inference: at test time, the model
generates each word conditioned on its own previous predictions rather than the ground-truth sequence. This exposure
4

APREPRINT- OCTOBER23, 2025
(a) Encoder output
 (b) Collapsed projector output
 (c) Projector output
Figure 2: Pairwise cosine distances between vector outputs. While the encoder representations remain well-separated
(a), the projected vectors may collapse, becoming nearly indistinguishable (b). Introducing the Cosine Alignment Loss
3 helps preserve the distinctions among the projections, preventing excessive overlap.
bias [ 23] can lead to the accumulation of errors during generation, as the model has never been exposed to its own
predictions during training.
Our target metrics — Exact Match (EM) and Edit Similarity (ES) — are inherently affected by teacher-forcing bias,
as they evaluate predictions at the sequence level. Previous studies have shown that both exposure bias and the non-
differentiability of sequence-based evaluation metrics can be mitigated using techniques from Reinforcement Learning
(RL) [ 24]. In particular, [ 23] and [ 15] apply the REINFORCE algorithm [ 16] to directly optimize non-differentiable,
sequence-level metrics.
Assume we are training an LLM decoder model with parametersθ. REINFORCE is based on the observation that the
expected gradient of a non-differentiable reward function can be computed as follows:
∇θLR(θ) =−E y∼p θ
r(y)∇ θlogp θ(y)
,(1)
wherey= (y 1, . . . , y T)is a sequence of generated tokens,y t∼pθ(yt|y1, . . . , y t−1).
In practice, the expected gradient can be approximated using a single Monte-Carlo sample from pθ. Using the sum of
our target metrics as a reward function brings us to the final expression for our REINFORCE loss component:
LR(θ) =−(EM(y) +ES(y))TX
t=1logp θ(yt|y1, . . . , y t−1),(2)
whereES(y)andEM(y)are the EM and ES metrics computed from a model rollout with greedy approach.
3.4 Cosine Alignment Loss
While training the projection from encoder representations into the LLM embedding space in our initial experiments,
we observed that the projection MLP often collapsed to an almost one-dimensional subspace: the angles between
projected vectors converged to nearly zero across most pairs (see Figure 2b), while the encoder itself is expressive,
producing embeddings with pairwise cosine similarities broadly distributed in the range[0.0,1.0](see Figure 2a).
This behavior is undesirable, since we aim to preserve the distinctions between retrieved text chunks. To address this
collapse and retain the relative differences among encoder embeddings after projection, we introduce a specialized
Cosine Alignment Loss:
LA(θ) =1√
2∥SC(yenc)−S C(yproj)∥F,(3)
where SCdenotes the cosine similarity matrix between vectors of the output batch, and yencandyprojrepresent the
encoder and projection output batches, respectively. This loss enforces preservation of pairwise cosine similarities within
5

APREPRINT- OCTOBER23, 2025
Figure 3: Relationship between KL-divergence loss and performance metrics (Exact Match (EM) and Edit Similarity
(ES)) on Qwen2.5-Coder-1.5B. The figure shows that reduction in KL-loss is accompanied by decreases in averaged
Exact Match (EM) and Edit Similarity (ES) metrics.
a batch by minimizing the mean squared error (MSE) between the similarity matrices. The factor1√
2compensates for
the symmetry of the cosine similarity matrix.
The loss formulation in 3 helps preserve relative differences between retrieved contexts. Figure 2c shows the resulting
cosine distance matrix for 100 random samples, demonstrating that the projections remain mostly well-separated and
their cosine distance matrix has the same structure as the original embeddings’ matrix. In contrast to that, collapsed
projector outputs form a single indistinguishable representation, as seen in Figure 2b.
3.5 Final Loss Function
3.6 KL-loss
In contrast to the findings reported by [ 4], we observe that the KL-divergence loss between models trained with
compressed versus uncompressed retrieved context does not improve prediction quality. In contrast, our experiments
show that minimizing the KL loss consistently results in substantial declines in both Edit Similarity and Exact Match
metrics (Figure 3), despite the presence of additional loss components described above.
This outcome directly conflicts with the results presented in the xRAG paper, where the KL-loss term was assigned a
weight of 2.0 relative to the NLL-loss weight of 1.0. Their choice was based on ablation studies with instruction-tuned
models on QA datasets, where they attributed performance gains primarily to the KL-loss component, arguing that it
improved the model’s resilience rate—defined as the proportion of cases in which responses remained correct both
before and after retrieval augmentation.
We optimize our model using the following composite loss function:
L(θ) =α CELCE(θ) +α RLR(θ) +α ALA(θ),(4)
where the coefficients αCE,αR, and αAare weighting factors. These weights are selected through hyperparameter
tuning using the Optuna framework [ 25]4. These and other training hyperparameters are listed in Appendix B. The loss
dynamic and its correspondence to target metrics EM and ES can be seen in Figure 4.
4 Experiments
4.1 Dataset
We trained our models on the Python subset of The Stack dataset [ 6]. To ensure dataset quality, we organized files by
repository and applied the following filtering steps: we excluded repositories with fewer than 50 stars, fewer than 5
files, or files containing fewer than 3 import statements. After filtering, the dataset contained approximately 150k code
completion samples, each paired with at least ten relevant cross-file context snippets. Relevant examples were identified
using the Jaccard text similarity metric applied to code chunks drawn from the surrounding repository (excluding the
current file used for code completion).
4https://optuna.org
6

APREPRINT- OCTOBER23, 2025
Figure 4: Relationship between the three loss components (Cross-Entropy, REINFORCE, and Cosine Alignment) and
the evaluation metrics Exact Match (EM) and Edit Similarity (ES).
Model Prompt size CE Loss↓EM↑ES↑
Qwen2.5Coder-1.5B 2000 0.97 45.97 66.57
LlavaCode Qwen2.5Coder-1.5B2010 1.02 47.66 68.74
Ablation Studies
Qwen2.5Coder-1.5B w/ CFC 2512 0.9950.87 69.43
LlavaCode Qwen2.5Coder-1.5B (only CE) 20100.8038.57 63.6
LlavaCode Qwen2.5Coder-1.5B (only REINFORCE) 2010 5.18 40.61 63.91
Table 1: Comparison of the LlavaCode approach with the baselines, along with ablation studies on different loss
components and RAG retrieval. Models with cross-file context are denoted by “w/ CFC”. Metrics are reported on
evaluation subset of our dataset (≈4.3k samples).
The code completion task takes fill-in-the-middle (FIM) format, where the left and right contexts are provided and the
missing middle segment must be generated by the LLM. Each target segment consists of ntlines ( 1≤n t≤9), with nt
sampled from a Poisson distribution. Code was segmented into chunks of 10×n tlines with an overlap of 5×n tlines.
We tried to enhance RAG with more sophisticated code search techniques, such as utilizing cosine distance between
text embeddings from various models, but Jaccard showed the best results. For more information, see Appendix E.
During training, we use all available length of the target, while evaluation is performed specifically on single line
completions. For evaluation, the dataset was split at the repository level to ensure that samples from a given repository
appeared exclusively in either the training or validation set. Additionally, we remove all leading and trailing whitespace
to ensure that ES metric is not artificially inflated.
4.2 Training
We train 2- and 3-layer MLP projection modules that map sentence encoder outputs (e.g., UniXCoder or
Qwen3Embedding) into the dimension of code LLM embeddings. For each sample, the top ten cross-file contexts are
projected into vector representations and concatenated with the code completion prompt embeddings before being
passed to the LLM. To ensure a fair comparison, when evaluating against the same LLM with non-compressed text
context, we keep the base prompt (without retrieved context) identical. In our setup, each retrieved context is truncated
to 512 tokens, while the code completion prompt budget (without retrieved context) is 2k tokens. As a result, the input
sequence length in our approach is 502 tokens shorter than in conventional RAG. Detailed discussion of the effect on
latency is discussed in Section 5.
During training, only the projection weights are updated, while both the encoder and the LLM remain frozen. Optimiza-
tion is performed using the joint loss described in Section 3.5, which combines all three loss components. Cross-Entropy
is only computed over the sequence after the <|fim_middle|> special token. For REINFORCE loss, we generate
50 tokens using greedy decoding and evaluate EM and ES metrics on the obtained sequence. A full list of training
hyperparameters, including the coefficients for each loss component, is provided in Appendix B.
Our main results are reported in Table 1. We compare our approach with retrieved context compression with a base
model without any additional context. Despite negligible 5 latency impact introduced by the additional 10 tokens,
our approach surpasses the no-CFC baseline on EM and ES metrics by a sizable margin, which makes our approach
preferable in latency-limited environments, such as IDE code completion. As an ablation, we also compare our model
7

APREPRINT- OCTOBER23, 2025
with base modelwithcross-file context, which introduces noticeable latency impact in the range of 20-38%, but also
increases EM and ES metrics. Detailed latency measurements are presented in Section 5 and in Tables 6, 5.
We conduct ablation studies with two alternative loss formulations, comparing Cross-Entropy-only and REINFORCE-
only objectives. As discussed in Section 3.2, relying exclusively on the cross-entropy objective degrades performance
on EM and ES metrics. Conversely, optimizing solely with the REINFORCE loss leads to uncontrolled entropy
growth and fails to surpass the w/o CFC baseline (Table 1), primarily due to the noise introduced by concatenated new
vectors. In contrast, only a carefully balanced combination of the three loss components (Section 3.5) enables consistent
improvements in the target metrics (Figure 4).
The ablation in Table 2 studies the effect of encoder choice and projection depth. We evaluate two encoders and
code modalities: UniXCoder with AST representations of retrieved code, and the Qwen-3-Embedding-0.6B model
with retrieved code. Qwen-3-Embedding-0.6B used as the retrieved-context compressor outperforms UniXCoder. A
three-layer MLP projection further improves both EM and ES but increases the number of trainable parameters by
roughly 4×.
Encoder Modality Projection # Trainable Parameters EM ES
UniXCoder AST 2-layer MLP 3.5M 46.69 67.65
UniXCoder AST 3-layer MLP 16.5M 46.94 68.26
Qwen3Embedding Code 2-layer MLP 3.9M 47.01 68.15
Qwen3Embedding Code 3-layer MLP 17.3M47.66 68.74
Table 2: Comparison of different encoders and projection heads with their trainable parameters and performance metrics.
Differences in the number of trainable parameters emerge from the encoder output dimension and the number of MLP
layers. All configurations were trained for≈6,600 training steps (3 epochs).
5 Speedup Estimation
Two deployment patterns dominate today’s LLM serving landscape. First,prefill–decode mixing, uses single engine
which interleaves chunks from prompt prefill with decoding passes across requests. For instance, one of the inference
engines, which utilizes this approach, is vLLM framework [ 26]. Second,disaggregated prefill-decode, when prefill
and decode run on separate GPU pools or nodes (possibly on different clusters) with independent resource plans. An
example of an engine that uses this approach is DistServe [27].
Colocating prefill and decode is utilization-friendly and achieves high throughput on single machines via memory-
efficient KV management and continuous batching. However, prefill and decode contend for distinct resources and
interfere with each other, which makes it hard to independently control TTFT (time to first token) and TPOT (time per
output token) under enterprise’s Service Level Agreement (SLA). As a result, systems are often over-provisioned with
hardware to satisfy both metrics. [28, 29]
Separating the phases decouples resource allocation and parallelism strategies, eliminating prefill–decode interference
and enabling direct tuning of TTFT (prefill stage) and TPOT (decode stage). Operationally, it simplifies capacity
planning and horizontal scaling because each fleet can scale along its own bottleneck. User will operate over IDE in
interactive manner, so TTFT of code completion LLM is the main metric to which the experience is sensitive, since, as
soon as tokens start generating, user can start reviewing code suggestions.
For disaggregated serving (transformers) and colocated prefill–decode (vllm) the results are shown in Table 5. For
performance measurements, we report scaling metrics for both inference patterns. For benchmarking, we implement
separate prefill and decode workers using the transformers runtime [ 30]. More detailed results, including TPOT metric,
are listed in Appendix C.
Reducing prompt length primarily improves TTFT; in colocated engines it often yields limited gains on decode-side
TPOT, which remains dominated by iterative decode dynamics and batching. Under disaggregation, the effect becomes
more predictable: shorter contexts directly reduce prefill latency and lower the number of GPUs requirements to handle
the same load while leaving decode behavior isolated, allowing clearer SLA tuning for each phase.
6 Conclusions and Future Work
In conclusion, we propose a novel pipeline for retrieval augmented code generation using LLaV A-like projection of
retrieved code chunks into LLM embeddings, which significantly increases the quality of text completions, while
8

APREPRINT- OCTOBER23, 2025
Sequence compression Model TTFT transformers TTFT vllm
2500→2010↓20%Qwen2.5-Coder-1.5B198.2→156.6↓21% 74.7→68.2↓9%
Qwen2.5-Coder-7B668.6→541.1↓19% 198.3→166.5↓16%
Qwen2.5-Coder-14B822.8→661.3↓20% 349.8→291.7↓17%
2000→1510↓24%Qwen2.5-Coder-1.5B157.4→113.4↓28% 65.3→58.4↓11%
Qwen2.5-Coder-7B540.0→406.8↓25% 179.0→134.0↓25%
Qwen2.5-Coder-14B662.2→496.3↓25% 291.5→232.9↓20%
1500→1010↓33%Qwen2.5-Coder-1.5B112.2→69.7↓38% 58.9→50.3↓15%
Qwen2.5-Coder-7B406.4→282.2↓31% 138.0→112.4↓19%
Qwen2.5-Coder-14B495.6→339.6↓31% 238.2→174.6↓27%
Table 3: For disaggregated inference deployment (measured with transformers library) context compression directly
leads to almost same decrease of TTFT. This way, response for user’s query start generating and showing to user much
earlier. For prefill-decode mixing, as described in Section 5, speedup is lower than context compression, due to decode
workload dominating on latency. Measured on NVIDIA A100.
introducing negligible effect on latency. Compared to full RAG, our approach results in 20-38% better prompt processing
speed and latency metrics, which is critical for code completion applications, while maintaining slightly worse, but
comparable generation quality.
To the best of our knowledge, our work is the first among the LLaV A-like approaches to apply compression to code
generation models, explore the addition of semantically rich code modalities, utilize base models instead of instruction-
tuned models, and apply reinforcement learning to train the projection for downstream code-completion tasks. Using
REINFORCE algorithm, we directly optimize ES and EM metrics, which are particularly connected to positive user
experience in interactive code completion environments. Furthermore, we achieve this by training only a lightweight
projection module, without modifying the embedding model and code generation LLM. Moreover, we introduced
cosine-alignment loss to overcome projector representation collapse.
Future work could investigate alternative variants of REINFORCE to improve alignment with EM/ES training, such
as SCST with bias, PPO, or GRPO. Another promising direction is scaling the proposed architecture to larger reader
and encoder models, which remains unexplored. In addition, as new encoders for graph modalities are developed, our
approach could be re-evaluated using these improved architectures. Finally, our current experiments are limited to the
Python subset of The Stack dataset; extending the evaluation to other widely used languages such as Java, C#, and
beyond would provide a broader assessment of the method’s generality.
References
[1]Patrick Lewis, Ethan Perez, Aleksandra Piktus, Fabio Petroni, Vladimir Karpukhin, Naman Goyal, Heinrich
Küttler, Mike Lewis, Wen tau Yih, Tim Rocktäschel, Sebastian Riedel, and Douwe Kiela. Retrieval-augmented
generation for knowledge-intensive nlp tasks, 2021.
[2]Jean-Baptiste Alayrac, Jeff Donahue, Pauline Luc, Antoine Miech, Iain Barr, Yana Hasson, Karel Lenc, Arthur
Mensch, Katie Millican, Malcolm Reynolds, Roman Ring, Eliza Rutherford, Serkan Cabi, Tengda Han, Zhitao
Gong, Sina Samangooei, Marianne Monteiro, Jacob Menick, Sebastian Borgeaud, Andrew Brock, Aida Ne-
matzadeh, Sahand Sharifzadeh, Mikolaj Binkowski, Ricardo Barreira, Oriol Vinyals, Andrew Zisserman, and
Karen Simonyan. Flamingo: a visual language model for few-shot learning, 2022.
[3] Haotian Liu, Chunyuan Li, Qingyang Wu, and Yong Jae Lee. Visual instruction tuning, 2023.
[4]Xin Cheng, Xun Wang, Xingxing Zhang, Tao Ge, Si-Qing Chen, Furu Wei, Huishuai Zhang, and Dongyan Zhao.
xrag: Extreme context compression for retrieval-augmented generation with one token, 2024.
[5]Raymond Li, Loubna Ben Allal, Yangtian Zi, Niklas Muennighoff, Denis Kocetkov, Chenghao Mou, Marc
Marone, Christopher Akiki, Jia Li, Jenny Chim, Qian Liu, Evgenii Zheltonozhskii, Terry Yue Zhuo, Thomas
Wang, Olivier Dehaene, Mishig Davaadorj, Joel Lamy-Poirier, João Monteiro, Oleh Shliazhko, Nicolas Gontier,
Nicholas Meade, Armel Zebaze, Ming-Ho Yee, Logesh Kumar Umapathi, Jian Zhu, Benjamin Lipkin, Muhtasham
Oblokulov, Zhiruo Wang, Rudra Murthy, Jason Stillerman, Siva Sankalp Patel, Dmitry Abulkhanov, Marco Zocca,
Manan Dey, Zhihan Zhang, Nour Fahmy, Urvashi Bhattacharyya, Wenhao Yu, Swayam Singh, Sasha Luccioni,
Paulo Villegas, Maxim Kunakov, Fedor Zhdanov, Manuel Romero, Tony Lee, Nadav Timor, Jennifer Ding, Claire
Schlesinger, Hailey Schoelkopf, Jan Ebert, Tri Dao, Mayank Mishra, Alex Gu, Jennifer Robinson, Carolyn Jane
Anderson, Brendan Dolan-Gavitt, Danish Contractor, Siva Reddy, Daniel Fried, Dzmitry Bahdanau, Yacine Jernite,
9

APREPRINT- OCTOBER23, 2025
Carlos Muñoz Ferrandis, Sean Hughes, Thomas Wolf, Arjun Guha, Leandro von Werra, and Harm de Vries.
Starcoder: may the source be with you!, 2023.
[6]Denis Kocetkov, Raymond Li, Loubna Ben Allal, Jia Li, Chenghao Mou, Carlos Muñoz Ferrandis, Yacine Jernite,
Margaret Mitchell, Sean Hughes, Thomas Wolf, Dzmitry Bahdanau, Leandro von Werra, and Harm de Vries. The
stack: 3 tb of permissively licensed source code, 2022.
[7]Binyuan Hui, Jian Yang, Zeyu Cui, Jiaxi Yang, Dayiheng Liu, Lei Zhang, Tianyu Liu, Jiajun Zhang, Bowen
Yu, Keming Lu, Kai Dang, Yang Fan, Yichang Zhang, An Yang, Rui Men, Fei Huang, Bo Zheng, Yibo Miao,
Shanghaoran Quan, Yunlong Feng, Xingzhang Ren, Xuancheng Ren, Jingren Zhou, and Junyang Lin. Qwen2.5-
coder technical report, 2024.
[8]Reiner Pope, Sholto Douglas, Aakanksha Chowdhery, Jacob Devlin, James Bradbury, Anselm Levskaya, Jonathan
Heek, Kefan Xiao, Shivani Agrawal, and Jeff Dean. Efficiently scaling transformer inference, 2022.
[9]Joshua Ainslie, James Lee-Thorp, Michiel de Jong, Yury Zemlyanskiy, Federico Lebrón, and Sumit Sanghai. Gqa:
Training generalized multi-query transformer models from multi-head checkpoints, 2023.
[10] Yanzhao Zhang, Mingxin Li, Dingkun Long, Xin Zhang, Huan Lin, Baosong Yang, Pengjun Xie, An Yang,
Dayiheng Liu, Junyang Lin, Fei Huang, and Jingren Zhou. Qwen3 embedding: Advancing text embedding and
reranking through foundation models, 2025.
[11] An Yang, Anfeng Li, Baosong Yang, Beichen Zhang, Binyuan Hui, Bo Zheng, Bowen Yu, Chang Gao, Chengen
Huang, Chenxu Lv, Chujie Zheng, Dayiheng Liu, Fan Zhou, Fei Huang, Feng Hu, Hao Ge, Haoran Wei, Huan
Lin, Jialong Tang, Jian Yang, Jianhong Tu, Jianwei Zhang, Jianxin Yang, Jiaxi Yang, Jing Zhou, Jingren Zhou,
Junyang Lin, Kai Dang, Keqin Bao, Kexin Yang, Le Yu, Lianghao Deng, Mei Li, Mingfeng Xue, Mingze Li, Pei
Zhang, Peng Wang, Qin Zhu, Rui Men, Ruize Gao, Shixuan Liu, Shuang Luo, Tianhao Li, Tianyi Tang, Wenbiao
Yin, Xingzhang Ren, Xinyu Wang, Xinyu Zhang, Xuancheng Ren, Yang Fan, Yang Su, Yichang Zhang, Yinger
Zhang, Yu Wan, Yuqiong Liu, Zekun Wang, Zeyu Cui, Zhenru Zhang, Zhipeng Zhou, and Zihan Qiu. Qwen3
technical report, 2025.
[12] Niklas Muennighoff, Nouamane Tazi, Loïc Magne, and Nils Reimers. Mteb: Massive text embedding benchmark,
2023.
[13] Daya Guo, Shuo Ren, Shuai Lu, Zhangyin Feng, Duyu Tang, Shujie Liu, Long Zhou, Nan Duan, Alexey
Svyatkovskiy, Shengyu Fu, Michele Tufano, Shao Kun Deng, Colin Clement, Dawn Drain, Neel Sundaresan, Jian
Yin, Daxin Jiang, and Ming Zhou. Graphcodebert: Pre-training code representations with data flow, 2021.
[14] Daya Guo, Shuai Lu, Nan Duan, Yanlin Wang, Ming Zhou, and Jian Yin. Unixcoder: Unified cross-modal
pre-training for code representation, 2022.
[15] Steven J. Rennie, Etienne Marcheret, Youssef Mroueh, Jarret Ross, and Vaibhava Goel. Self-critical sequence
training for image captioning, 2017.
[16] Ronald J. Williams. Simple statistical gradient-following algorithms for connectionist reinforcement learning.
Mach. Learn., 8(3–4):229–256, May 1992.
[17] Dan Hendrycks and Kevin Gimpel. Gaussian error linear units (gelus), 2023.
[18] Jimmy Lei Ba, Jamie Ryan Kiros, and Geoffrey E. Hinton. Layer normalization, 2016.
[19] Andrew Jaegle, Felix Gimeno, Andy Brock, Oriol Vinyals, Andrew Zisserman, and Joao Carreira. Perceiver:
General perception with iterative attention. In Marina Meila and Tong Zhang, editors,Proceedings of the 38th
International Conference on Machine Learning, volume 139 ofProceedings of Machine Learning Research, pages
4651–4664. PMLR, 18–24 Jul 2021.
[20] Tatiana Zemskova and Dmitry Yudin. 3dgraphllm: Combining semantic graphs and large language models for 3d
scene understanding, 2025.
[21] Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry,
Amanda Askell, Pamela Mishkin, Jack Clark, Gretchen Krueger, and Ilya Sutskever. Learning transferable visual
models from natural language supervision. In Marina Meila and Tong Zhang, editors,Proceedings of the 38th
International Conference on Machine Learning, volume 139 ofProceedings of Machine Learning Research, pages
8748–8763. PMLR, 18–24 Jul 2021.
[22] Samy Bengio, Oriol Vinyals, Navdeep Jaitly, and Noam Shazeer. Scheduled sampling for sequence prediction
with recurrent neural networks. InProceedings of the 29th International Conference on Neural Information
Processing Systems - Volume 1, NIPS’15, page 1171–1179, Cambridge, MA, USA, 2015. MIT Press.
[23] Marc’Aurelio Ranzato, Sumit Chopra, Michael Auli, and Wojciech Zaremba. Sequence level training with
recurrent neural networks.CoRR, abs/1511.06732, 2015.
10

APREPRINT- OCTOBER23, 2025
[24] Richard S. Sutton and Andrew G. Barto. Reinforsement learning: An introduction, adaptive computation and
machine learning series. 1998.
[25] Takuya Akiba, Shotaro Sano, Toshihiko Yanase, Takeru Ohta, and Masanori Koyama. Optuna: A next-generation
hyperparameter optimization framework, 2019.
[26] Woosuk Kwon, Zhuohan Li, Siyuan Zhuang, Ying Sheng, Lianmin Zheng, Cody Hao Yu, Joseph Gonzalez, Hao
Zhang, and Ion Stoica. Efficient memory management for large language model serving with pagedattention. In
Proceedings of the 29th symposium on operating systems principles, pages 611–626, 2023.
[27] Yinmin Zhong, Shengyu Liu, Junda Chen, Jianbo Hu, Yibo Zhu, Xuanzhe Liu, Xin Jin, and Hao Zhang.
{DistServe }: Disaggregating prefill and decoding for goodput-optimized large language model serving. In18th
USENIX Symposium on Operating Systems Design and Implementation (OSDI 24), pages 193–210, 2024.
[28] Amey Agrawal, Nitin Kedia, Ashish Panwar, Jayashree Mohan, Nipun Kwatra, Bhargav Gulavani, Alexey
Tumanov, and Ramachandran Ramjee. Taming {Throughput-Latency }tradeoff in {LLM}inference with {Sarathi-
Serve}. In18th USENIX Symposium on Operating Systems Design and Implementation (OSDI 24), pages 117–134,
2024.
[29] Zhibin Wang, Shipeng Li, Yuhang Zhou, Xue Li, Rong Gu, Nguyen Cam-Tu, Chen Tian, and Sheng Zhong.
Revisiting slo and goodput metrics in llm serving.arXiv preprint arXiv:2410.14257, 2024.
[30] Thomas Wolf, Lysandre Debut, Victor Sanh, Julien Chaumond, Clement Delangue, Anthony Moi, Pierric Cistac,
Tim Rault, Remi Louf, Morgan Funtowicz, et al. Transformers: State-of-the-art natural language processing. In
Proceedings of the 2020 conference on empirical methods in natural language processing: system demonstrations,
pages 38–45, 2020.
[31] Yuri Kuratov, Mikhail Arkhipov, Aydar Bulatov, and Mikhail Burtsev. Cramming 1568 tokens into a single vector
and back again: Exploring the limits of embedding space capacity, 2025.
A LLM usage statement
We used ChatGPT-5 and ChatGPT-4o to correct grammatical and stylistic errors, condense text, perform translations
and rephrase content.
B Training Parameters
For our primary evaluations, we used the Qwen2.5Coder-1.5B model, with the Qwen-3-Embedding-0.6B model serving
as the encoder. A three-layer MLP was employed as the projector, mapping from the encoder dimension to twice the
embedding size of the LLM, and finally down to the LLM’s embedding size. A GELU activation and a LayerNorm
were applied between the first and second layers, and again between the second and final layer.
Hyperparameter Value
optimizer AdamW
alpha Cosine Alignment 0.1
alpha Cross-Entropy 0.9
alpha REINFORCE 0.1
learning rate 1e-3
lr scheduler type cosine
warmup ratio 0.03
weight dacay 0.0
epochs 3
effective batch size 66
train samples 150k
Table 4: Hyperparameters for projection training
C Detailed Latency and Load measurements
This section expands on the results presented in Section 5, including TTOP measurements as shown in Tables 5 and 6,
as well as latency reduction measurements for prefill-only regime (1-token generation), reported in Tables 7 and 8.
11

APREPRINT- OCTOBER23, 2025
Sequence compression Model TTFT TPOT
2500→2010↓20%Qwen2.5-Coder-1.5B198.2→156.6↓21% 23.6→23.2↓2%
Qwen2.5-Coder-7B668.6→541.1↓19% 27.2→25.4↓7%
Qwen2.5-Coder-14B822.8→661.3↓20% 58.5→52.8↓10%
2000→1510↓24%Qwen2.5-Coder-1.5B157.4→113.4↓28% 23.5→23.1↓2%
Qwen2.5-Coder-7B540.0→406.8↓25% 25.1→24.1↓4%
Qwen2.5-Coder-14B662.2→496.3↓25% 52.7→47.0↓11%
1500→1010↓33%Qwen2.5-Coder-1.5B112.2→69.7↓38% 23.2→23.5↑1%
Qwen2.5-Coder-7B406.4→282.2↓31% 24.1→24.2
Qwen2.5-Coder-14B495.6→339.6↓31% 46.8→41.1↓12%
Table 5: For disaggregated inference deployment (measured with transformers library) context compression directly
leads to almost same decrease of TTFT. This way, response for user’s query start generating and showing to user much
earlier. Measured on a single NVIDIA A100.
Sequence compression Model TTFT TPOT
2500→2010↓20%Qwen2.5-Coder-1.5B74.7→68.2↓9% 5.3→5.3
Qwen2.5-Coder-7B198.3→166.5↓16% 11.7→11.7
Qwen2.5-Coder-14B349.8→291.7↓17% 22.3→21.8↓2%
2000→1510↓24%Qwen2.5-Coder-1.5B65.3→58.4↓11% 5.3→5.6↑5%
Qwen2.5-Coder-7B179.0→134.0↓25% 11.7→11.6↓1%
Qwen2.5-Coder-14B291.5→232.9↓20% 21.8→21.8
1500→1010↓33%Qwen2.5-Coder-1.5B58.9→50.3↓15% 5.4→6.3↑16%
Qwen2.5-Coder-7B138.0→112.4↓19% 11.6→11.5↓1%
Qwen2.5-Coder-14B238.2→174.6↓27% 21.7→21.4↓1%
Table 6: For prefill-decode mixing, context compression leads to more efficiency. But, as described in Section 5,
speedup is lower than for context compression, due to decode workload dominating on latency. Measured on NVIDIA
A100.
Sequence compression Model TTFT
2500→2010↓20%Qwen2.5-Coder-1.5B198.2→159.3↓20%
2500→2010↓20%Qwen2.5-Coder-7B668.1→539.1↓19%
2500→2010↓20%Qwen2.5-Coder-14B820.9→661.1↓19%
2000→1510↓24%Qwen2.5-Coder-1.5B159.9→121.2↓24%
2000→1510↓24%Qwen2.5-Coder-7B539.1→406.3↓25%
2000→1510↓24%Qwen2.5-Coder-14B660.6→495.8↓25%
1500→1010↓33%Qwen2.5-Coder-1.5B120.7→75.5↓37%
1500→1010↓33%Qwen2.5-Coder-7B405.4→281.0↓31%
1500→1010↓33%Qwen2.5-Coder-14B494.9→339.6↓31%
Table 7: Latency reduction in prefill-only regime (generation of 1 token). Transformers library.
Sequence compression Model TTFT
2500→2010↓20%Qwen2.5-Coder-7B197.4→165.9↓16%
2500→2010↓20%Qwen2.5-Coder-14B351.5→291.2↓17%
2500→2010↓20%Qwen2.5-Coder-1.5B79.1→67.0↓15%
2000→1510↓24%Qwen2.5-Coder-7B164.4→135.8↓17%
2000→1510↓24%Qwen2.5-Coder-14B290.0→240.9↓17%
2000→1510↓24%Qwen2.5-Coder-1.5B65.9→56.4↓14%
1500→1010↓33%Qwen2.5-Coder-7B136.5→104.6↓23%
1500→1010↓33%Qwen2.5-Coder-14B240.2→191.4↓20%
1500→1010↓33%Qwen2.5-Coder-1.5B56.4→48.7↓14%
Table 8: Latency reduction in prefill-only regime (generation of 1 token). vLLM framework.
12

APREPRINT- OCTOBER23, 2025
D On pretraining of the projection module
Whereas most prior work adopts two-stage training, we use a single-stage pipeline based on a composite loss function,
discussed in Section 3.5. For completeness, we also evaluated a conventional two-stage pretrain–finetune pipeline for
projection training.
In prior work, pretraining often relies on parallel datasets, such as paraphrase pairs in xRAG or image–caption pairs in
LLaV A. Inspired by xRAG, we experimented with a similar pretraining approach, attempting to reconstruct retrieved
context chunks from projected vectors by optimizing the entropy loss. This approach did not yield improvements in the
second stage of training, likely due to the entropy issues discussed in Section 3.2.
[31] demonstrate that up to 1,568 tokens can be compressed into a single continuous "memory" token by treating the
token as a trainable parameter and optimizing it via backpropagation with a cross-entropy reconstruction loss. Because
these continuous tokens reconstruct to reference texts, we treat them as ground truth for training our projection layer.
Concretely, we encode text with our encoder, project the resulting embeddings into a single token, and optimize a
mixture of Mean Squared Error (MSE) and cosine-similarity (CS) losses between the projected embedding and the
trained ground-truth compressed token.
However, the space spanned by the memory tokens proved to be highly non-smooth. For instance, identical text inputs
could be compressed into vectors that are widely separated, and introducing even small perturbations to a learned
memory token often results in reconstruction of completely different text. This leads to poor generalization for an
MLP module attempting to map into this space. Consequently, learning a projection into such a space requires extreme
overparameterization, effectively amounting to memorizing the entire dataset. As a result, we could only overfit on a
small subset of memory tokens and were unable to learn a meaningful translation into the memory token space.
We leave the more sophisticated pretraining of the projection module for code compression to future work.
E On different retrieval techniques
We evaluated multiple retrieval metrics for selecting the top-10 most relevant code chunks. Specifically, we compared
sparse retrievers such as BM25 and Jaccard with dense retrievers based on cosine similarity over embeddings from
UniXCoder and Jina v2. Each retriever-augmented model was benchmarked against a baseline model without any
additional retrieved context. The comparison was conducted on a subset of 1,600 code completion tasks from our
dataset as described in Section 4.1. The results show that Jaccard and UniXCoder achieved the best performance. Given
its lower latency, we adopt Jaccard as the primary retrieval method in Section 4.1.
Method EM ES
No CFC 50.50 73.12
BM25 55.56 76.5
Jaccard 56.1976.68
UniXCoder 56.00 76.84
Jina v2 54.31 75.56
Table 9: Comparison of different retrieval strategies.
13