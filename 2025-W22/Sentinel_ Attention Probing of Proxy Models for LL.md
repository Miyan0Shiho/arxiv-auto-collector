# Sentinel: Attention Probing of Proxy Models for LLM Context Compression with an Understanding Perspective

**Authors**: Yong Zhang, Yanwen Huang, Ning Cheng, Yang Guo, Yun Zhu, Yanmeng Wang, Shaojun Wang, Jing Xiao

**Published**: 2025-05-29 09:24:12

**PDF URL**: [http://arxiv.org/pdf/2505.23277v1](http://arxiv.org/pdf/2505.23277v1)

## Abstract
Retrieval-augmented generation (RAG) enhances large language models (LLMs)
with external context, but retrieved passages are often lengthy, noisy, or
exceed input limits. Existing compression methods typically require supervised
training of dedicated compression models, increasing cost and reducing
portability. We propose Sentinel, a lightweight sentence-level compression
framework that reframes context filtering as an attention-based understanding
task. Rather than training a compression model, Sentinel probes decoder
attention from an off-the-shelf 0.5B proxy LLM using a lightweight classifier
to identify sentence relevance. Empirically, we find that query-context
relevance estimation is consistent across model scales, with 0.5B proxies
closely matching the behaviors of larger models. On the LongBench benchmark,
Sentinel achieves up to 5$\times$ compression while matching the QA performance
of 7B-scale compression systems. Our results suggest that probing native
attention signals enables fast, effective, and question-aware context
compression. Code available at: https://github.com/yzhangchuck/Sentinel.

## Full Text


<!-- PDF content starts -->

arXiv:2505.23277v1  [cs.CL]  29 May 2025Sentinel: Attention Probing of Proxy Models for LLM Context
Compression with an Understanding Perspective
Yong Zhang1, Yanwen Huang1,2, Ning Cheng1,*,
Yang Guo1,Yun Zhu1,Yanmeng Wang1,Shaojun Wang1,Jing Xiao1,
1Ping An Technology (Shenzhen) Co., Ltd., China
2University of Electronic Science and Technology of China
zhangyong203@pingan.com.cn
Abstract
Retrieval-augmented generation (RAG) en-
hances large language models (LLMs) with ex-
ternal context, but retrieved passages are often
lengthy, noisy, or exceed input limits. Existing
compression methods typically require super-
vised training of dedicated compression mod-
els, increasing cost and reducing portability.
We propose Sentinel , a lightweight sentence-
level compression framework that reframes
context filtering as an attention-based under-
standing task . Rather than training a compres-
sion model, Sentinel probes decoder attention
from an off-the-shelf 0.5B proxy LLM using a
lightweight classifier to identify sentence rele-
vance. Empirically, we find that query-context
relevance estimation is consistent across model
scales , with 0.5B proxies closely matching the
behaviors of larger models. On the LongBench
benchmark, Sentinel achieves up to 5 ×com-
pression while matching the QA performance
of 7B-scale compression systems. Our results
suggest that probing native attention signals
enables fast, effective, and question-aware con-
text compression.1
1 Introduction
Large language models (LLMs) have achieved im-
pressive performance across open-domain ques-
tion answering, reasoning, and dialogue tasks
(Brown et al., 2020; OpenAI, 2024). To scale
their capabilities to knowledge-intensive applica-
tions, Retrieval-Augmented Generation (RAG) has
emerged as a powerful paradigm that augments
model inputs with retrieved evidence from exter-
nal corpora (Lewis et al., 2020; Guu et al., 2020;
Shi et al., 2024). However, long retrieved contexts
are often noisy, redundant, or exceed model input
limits, making context compression essential for
both efficiency and effectiveness (Liu et al., 2024;
Yoran et al., 2024).
*Corresponding author
1Our code is available at https://github.com/
yzhangchuck/Sentinel .This challenge has motivated a shift from coarse
document-level reranking (Karpukhin et al., 2020)
to fine-grained context compression, broadly cate-
gorized into token-level and sentence-level selec-
tion strategies. Token-based methods (e.g., LLM-
Lingua 1/2 (Jiang et al., 2023; Pan et al., 2024),
QGC (Cao et al., 2024)) estimate token impor-
tance via perplexity or query-aware signals but of-
ten fragment discourse coherence. Sentence-level
approaches (e.g., RECOMP, EXIT (Xu et al., 2024;
Hwang et al., 2024)) preserve syntactic structure
by selecting full sentences, yet typically require
generator feedback or task-specific tuning. Despite
progress, existing methods remain tightly coupled
to supervision or generation signals—highlighting
the need for a lightweight alternative.
Among these challenges, a growing body of
work suggests that LLMs inherently reveal internal
semantic and relevance signals during inference.
Specifically, decoder attention patterns have been
shown to capture factual attribution and grounding
behaviors (Wu et al., 2024; Huang et al., 2025),
while final hidden states under specialized prompts
can serve as effective compressed semantic em-
beddings, as demonstrated in PromptEOL (Jiang
et al., 2024b). These findings point to an emerging
consensus: LLMs naturally aggregate context un-
derstanding into their final inference steps , provid-
ing an opportunity for lightweight signal extraction
without explicit generation.
Building on these findings, we hypothesize that
query-context understanding tends to be sta-
ble across model scales , even as generation ca-
pabilities differ. Smaller models may exhibit at-
tention patterns that align closely with those of
larger LLMs, despite limited generation capacity.
This suggests a promising direction: instead of re-
lying on full-scale generation or reranking, we can
decode relevance signals directly from native atten-
tion behaviors in compact models. We adopt this
perspective to design a lightweight and efficient

ProbingClassifierTop-K Filtered Sentences
LLMQuery + Retrieved ContextSentence-Level Attention Feature ConstructionAttention Matrix
🤖Proxy LM
🔍Figure 1: Sentinel Framework Overview. Given a query–context pair, an off-the-shelf proxy model provides
final-token decoder attention. A probing classifier interprets sentence-level attention features to select relevant
context, enabling lightweight and model-agnostic compression for downstream LLMs.
approach to context compression.
However, existing methods often use attention
in shallow or heuristic ways. Many rely on raw
thresholding over decoder attention, which tends to
be noisy, brittle, and poorly aligned with sentence-
level semantics (Wang et al., 2024; Fang et al.,
2025). These approaches fall short of providing a
lightweight and interpretable compression mech-
anism that can robustly leverage native model be-
havior.
In this work, we propose Sentinel , a lightweight
and model-agnostic framework for sentence-level
context compression based on attention probing.
Given a query–context pair, Sentinel uses an off-
the-shelf 0.5B-scale proxy LLM to extract multi-
layer decoder attention directed toward input sen-
tences. Instead of training a dedicated compression
model, we treat compression as a probing task: ag-
gregating attention features across heads and layers,
and using a simple logistic regression classifier to
predict sentence relevance. This design eliminates
the need to train a task-specific compression model
enabling efficient, interpretable, and plug-and-play
compression.
Empirically, Sentinel achieves up to 5 ×input
compression on LongBench (Bai et al., 2024),
while matching the QA performance of 7B-scale
compression systems using only a 0.5B proxy
model. It generalizes effectively across QA tasks,
languages, and LLM backbones, without requiring
generator supervision or prompt-specific tuning.
Our contributions are as follows:
•We reframe context compression as an
attention-based understanding task , and im-
plement it via a probing classifier—without
training a dedicated compression model.
•We propose Sentinel , a lightweight sentence-
level compression framework that probes na-
tive attention signals from an off-the-shelf
0.5B proxy model to identify query-relevant
context for compression.•We empirically observe that query-context
relevance estimation remains stable across
model scales , enabling compact proxies to ap-
proximate large-model compression behavior.
•On LongBench, Sentinel achieves up to 5 ×in-
put compression while matching the QA per-
formance of 7B-scale compression systems
across both English and Chinese tasks.
2 Methodology
We propose Sentinel , a lightweight framework that
probes native attention behaviors in small proxy
models to identify query-relevant sentences. Rather
than training a compression model end-to-end, we
decode attention-based signals already embedded
in the model’s inference dynamics. Our pipeline
consists of three stages: attention feature extrac-
tion, probing classifier training, and integration
with downstream LLMs.
2.1 Task Formulation
Given a query qand a retrieved context C=
{s1, s2, . . . , s n}composed of sentences, our goal
is to select a subset C′⊆Cthat retains essential
information for answering q.
We frame this as a probing task: for each sen-
tence si, we train a binary classifier to predict a
relevance label yi∈ {0,1}based solely on atten-
tion signals from a compact proxy model. At in-
ference time, we use the predicted probability as a
soft relevance score for sentence selection.
2.2 Attention-Based Utility Estimation
The core of Sentinel is a feature extractor that lever-
ages decoder attention to estimate sentence utility.
Given the input sequence:
[q]⊕[s1]⊕[s2]⊕ ··· ⊕ [sn]
we feed it into a small decoder-only language
model with instruction-following capability. To
encourage semantic compression at the final posi-
tion, we apply a prompt template that requests a

one-word answer following the context and query,
similar in spirit to PromptEOL (Jiang et al., 2024b).
We extract the attention tensor A∈RL×H×Tfrom
the final decoder token, capturing attention scores
across layers, heads, and input tokens.
Why attention reflects utility. Decoder attention
has been shown to reflect alignment and attribution
signals (Huang et al., 2025). In particular, the at-
tention received by the final token often encodes
which input segments are most relevant for gen-
eration. From an information-theoretic perspec-
tive, Barbero et al. (2024) show that decoder-only
models compress prior context into the final token
representation — over-squashing .
Feature representation. For each sentence si,
we compute a set of attention-based features de-
rived from the attention received by its tokens from
the final decoder token. Specifically, we extract the
attention weights from each layer and head directed
toward context tokens, and normalize them by the
total attention mass over the context span. This nor-
malization removes the influence of non-context
elements such as the query or prompt.
We then average the normalized attention
weights over the tokens in si, independently for
each attention head and layer, yielding a feature
vector vi∈RLH, where Lis the number of de-
coder layers and His the number of attention heads.
Each element of vireflects the relative contribution
ofsias measured by a specific attention channel.
2.3 Probing Classifier for Sentence Relevance
To decode relevance from attention features, we
train a lightweight logistic regression (LR) probe.
The probe computes:
ˆyi=σ(w⊤vi+b),
where σis the sigmoid function. The model outputs
a probability score for each sentence, which is used
directly for sentence ranking.
2.4 Weak Supervision and Robust Probing
To effectively probe the model’s internal relevance
behavior, we train the classifier using weak super-
vision from QA datasets, combined with retrieval
filtering and robustness enhancements.
2.5 Probing Data Construction
We construct the initial training data from widely
used QA datasets, covering both single-hop andmulti-hop question answering scenarios. Specif-
ically, we use examples from SQuAD, Natural
Questions (single-hop), and HotpotQA (multi-hop),
where answer spans are annotated in retrieved con-
texts. For each QA example, sentences containing
the gold answer span are labeled as positive, while
all other sentences are labeled as negative. This
weak supervision allows us to build large-scale
training data without requiring manual annotation
of sentence relevance, and ensures that the classi-
fier is exposed to both simple factual questions and
complex multi-hop reasoning patterns.
2.5.1 Selecting Context-Reliant Samples
To purify supervision, we retain only QA examples
that require retrieved context for correct answering.
Specifically, We retain only QA examples where
the model fails to answer correctly without the re-
trieved context but succeeds when the context is
provided. This filtering conceptually echoes prior
work that probes model behavior via intervention-
based output changes (Meng et al., 2022). This cri-
terion ensures that the positive sentences truly con-
tribute critical information needed for answering,
and reduces contamination from internal model
memorization or hallucinated knowledge. By fil-
tering for retrieval-dependency, we focus training
on cases where relevance decoding from context is
essential, aligning with the goal of context under-
standing rather than internal recall.
2.5.2 Robustness via Sentence Shuffling
To mitigate positional biases (Liu et al., 2024), es-
pecially common in multi-document retrieval set-
tings, we apply sentence shuffling during training
by randomly permuting sentence order within each
passage. This simple perturbation encourages the
classifier to rely on semantic relevance rather than
fixed positions, improving generalization to real-
world RAG inputs with noisy or varied structure.
2.6 Inference via Attention Probing
At inference time, given a query–context pair
(q, C), we run the proxy model with a fixed QA-
style prompt, extract final-token decoder attention,
and compute sentence-level attention features. The
probing classifier assigns a relevance score to each
sentence, and the top-ranked subset is selected to
fit the length budget. This compressed context is
passed to the downstream LLM for generation.
Since Sentinel operates solely on proxy model
attention, it is model-agnostic and can be inte-

grated into any RAG pipeline without modifying
or fine-tuning the target LLM.
3 Experiments
Datasets We evaluate our method on the English
subset of LongBench (Bai et al., 2024), which cov-
ers six task categories: Single-Document QA ,Multi-
Document QA ,Summarization ,Few-shot Reason-
ing,Synthetic Retrieval , and Code Completion .
To ensure comparability with prior work (e.g.,
GPT-3.5-Turbo baselines), we include all original
tasks in comparison tables. However, for our Qwen-
based experiments, we exclude LCC andPassage-
Count as their input formats and task goals conflict
with context compression (see Appendix A). In our
Qwen-based tables, we annotate the Synth. and
Code columns with an asterisk (*) to indicate mod-
ified category composition.
Probing Data We construct our training set from
3,000 QA examples sampled from NewsQA (50%),
SQuAD (20%), and HotpotQA (30%), covering
both single-hop and multi-hop reasoning tasks. For
each QA example, we extract one positive sentence
supporting the gold answer span, and one negative
sentence from the same passage, resulting in 6,000
sentence-level training instances.
In NewsQA, 30.1% of examples contain 0–500
tokens (tokenized with Qwen-2.5) and 69.9% have
500–1,000 tokens. In SQuAD, 99.3% fall within
0–500 tokens. For HotpotQA, we restrict all exam-
ples to 0–500 tokens by limiting unrelated content.
Each context is segmented into sentences using
spaCy’s sentencizer . To extract features, we ap-
ply a QA-style prompt to each example and collect
decoder attention from the final token. The prompt
format is:
Given the following information: {context}
Answer the following question based on the
given information with one or few words:
{question}
Answer:
Decoder attention weights from the final token to
each sentence are aggregated to form fixed-length
feature vectors for classification.
Context-Reliant Sample Selection To improve
label quality, we retain only examples where the
context is necessary for correct answering. For
NewsQA and SQuAD, we keep examples where
the memory-based answer is incorrect and thecontext-based answer is correct, both judged by
EM. For HotpotQA, we retain only samples with
memory F1 ≤0.2and context F1 ≥0.5.
Probing Classifier Training We train a logis-
tic regression (LR) model on attention-derived
features, using 5-fold cross-validation with bal-
anced accuracy as the scoring metric. We per-
form grid search over regularization strengths
C∈ {0.01,0.1,1.0,10.0,100.0}, and use the lib-
linear solver with ℓ2regularization, class-balanced
weighting, and a maximum of 2,000 iterations. The
best model is selected based on AUC on the valida-
tion set.
Compression Strategy We implement a length-
controlled compression strategy based on the tar-
get LLM’s tokenizer. For a given context x=
s1, s2, . . . , s n, our classifier scores each sentence,
and we select a subset that meets one of the follow-
ing constraints:
•Token-length budget: Retain top-ranked sen-
tences until their total token count (measured
by the target model’s tokenizer) reaches a
fixed budget B(e.g., 2000 tokens).
•Token-ratio constraint: Retain top-ranked
sentences whose cumulative token count does
not exceed a fraction τ(e.g., 0.1 to 0.5) of the
original context’s tokenized length.
In both cases, selected sentences are concate-
nated in their original order to form the compressed
input.
Proxy Model Setup In our main experiments, we
adopt Sentinel with Qwen-2.5-0.5B-Instruct as
the default proxy model for extracting attention-
based features and performing sentence relevance
classification. Unless otherwise specified, all re-
ported results use a chunk size of 1024 tokens,
where the retrieved context is segmented into non-
overlapping token blocks of up to 1024 tokens be-
fore being passed to the proxy model.
Evaluation Models Following the Long-
Bench and LLMLingua setup, we use ChatGPT
(gpt-3.5-turbo) as the primary model for QA
evaluation. To assess the generality of our method,
we also experiment with Qwen-2.5-7B-Instruct
in our main results. All evaluations follow the
LongBench prompt and decoding setup (Bai et al.,
2024), as detailed in Appendix J.

MethodsLongBench (GPT-3.5-Turbo, 2000-token constraint) Compression Stats
SingleDoc MultiDoc Summ. FewShot Synth. Code A VG Tokens 1/ τ
Selective-Context (LLaMA-2-7B-Chat) 16.2 34.8 24.4 15.7 8.4 49.2 24.8 1,925 5x
LLMLingua (LLaMA-2-7B-Chat) 22.4 32.1 24.5 61.2 10.4 56.8 34.6 1,950 5x
LLMLingua-2 29.8 33.1 25.3 66.4 21.3 58.9 39.1 1,954 5x
LongLLMLingua (LLaMA-2-7B-Chat) 39.0 42.2 27.4 69.3 53.8 56.6 48.0 1,809 6x
CPC (Mistral-7B-Instruct-v0.2) 42.6 48.6 23.7 69.4 52.8 60.0 49.5 1,844 5x
Sentinel (Qwen-2.5-0.5B-Instruct) 40.1 47.4 25.8 69.9 46.3 58.0 47.89 1,885 5x
Sentinel (Qwen-2.5-1.5B-Instruct) 40.6 48.1 26.0 69.1 49.0 57.6 48.4 1,883 5x
Original Prompt 39.7 38.7 26.5 67.0 37.8 54.2 44.0 10,295 -
Table 1: Performance on LongBench using GPT-3.5-Turbo as the inference model. Best results are in bold ,
second-best are underlined .
MethodsLongBench-En (Qwen-2.5-7B-Instruct, 2000-token constraint) LongBench-Zh (Qwen-2.5-7B-Instruct, 2000-token constraint)
SingleDoc MultiDoc Summ. FewShot *Synth. *Code En-A VG SingleDoc MultiDoc Summ. FewShot Synth. Zh-A VG Overall A VG
Empty Context 10.74 19.97 13.52 40.1 2.5 56.0 23.81 15.48 12.25 10.06 18.5 4.38 12.13 19.78
Random 26.54 28.03 23.83 62.5 18.25 62.12 36.88 42.37 17.64 13.93 19.25 35.0 25.64 36.33
Raw Attention (0.5B) 33.11 38.39 24.22 63.27 77.67 62.14 49.8 56.24 20.99 13.61 44.50 72.76 41.62 48.47
Sentinel (0.5B) 37.11 44.98 25.02 64.88 85.54 63.04 53.43 59.18 24.15 13.19 43.00 74.68 42.84 51.23
Sentinel (1.5B) 38.47 44.95 25.03 65.32 90.50 62.16 54.40 58.88 25.52 13.54 42.17 83.57 44.74 52.48
Sentinel (3B) 39.06 45.44 25.21 66.00 86.46 62.93 54.18 59.10 25.83 13.86 43.00 77.03 43.76 52.04
Original Prompt 37.88 40.31 25.32 69.21 98.58 66.74 56.34 60.60 20.67 15.03 43.25 86.93 45.30 54.05
Table 2: Joint performance on LongBench-En and LongBench-Zh using Qwen-2.5-7B-Instruct. Best results are in
bold , second-best are underlined .
Baselines We compare Sentinel against a range
of context compression baselines, including token-
level methods (LLMLingua-1/2 (Jiang et al., 2023;
Pan et al., 2024)), sentence-level approaches
(Selective-Context (Li et al., 2023), CPC (Liskavets
et al., 2024), LongLLMLingua (Jiang et al.,
2024a)), and attention-based heuristics such as Raw
Attention (Wang et al., 2024; Fang et al., 2025).
We also include non-learning baselines including
Random Selection and Empty Context. Full de-
scriptions of these baselines are provided in Ap-
pendix B.
Metrics We follow the LongBench evaluation
protocol and adopt task-specific metrics for each
task category: QA-F1 for Single-Document QA,
Multi-Document QA, and Few-shot Reasoning;
ROUGE-L for Summarization; classification ac-
curacy for Synthetic Retrieval. All metrics are
computed using the official evaluation scripts.
3.1 Results on LongBench
We evaluate Sentinel across two settings: (1) En-
glish tasks using GPT-3.5-Turbo as the inference
model, and (2) both English and Chinese tasks
using Qwen-2.5-7B-Instruct. All results are re-
ported under a 2,000-token input constraint with
chunk size 1024 unless otherwise noted. Although
the probing classifier is trained on external QA
datasets, it generalizes effectively to downstream
LongBench tasks.English Evaluation with GPT-3.5-Turbo Ta-
ble 1 shows that Sentinel, with a 0.5B proxy, per-
forms strongly across all English LongBench cate-
gories, outperforming task-agnostic baselines like
LLMLingua-1/2 and matching 7B-scale systems
like CPC and LongLLMLingua. This suggests that
small proxies suffice for effective sentence selec-
tion. On Chinese tasks, Sentinel also outperforms
LLMLingua under tight budgets when evaluated
with GPT-3.5-Turbo (Appendix E).
English Evaluation with Qwen-2.5-7B In Ta-
ble 2, we report results on the English subset of
LongBench using Qwen-2.5-7B-Instruct. Sentinel
consistently outperforms Raw Attention, Random,
and Empty Context baselines across all task types.
Notably, on Single-Doc and Multi-Doc QA, Sen-
tinel even surpasses the Original Prompt despite
using a 5 ×shorter input, confirming its ability to
distill high-relevance content.
However, on Summarization, FewShot, and
Code tasks, Sentinel underperforms the Original
Prompt. This is likely due to the loss of global struc-
ture: summarization benefits from full-passage con-
text, few-shot tasks rely on intact example format-
ting, and code inputs are sensitive to line breaks,
which our sentence segmentation does not preserve.
Chinese Evaluation with Qwen-2.5-7B On the
Chinese subset of LongBench, Sentinel demon-
strates competitive performance, generally outper-
forming Raw Attention across tasks. It achieves

strong results on Single-Doc and notably surpasses
the Original Prompt on Multi-Doc QA. While its
performance on FewShot and Summarization re-
mains lower than the Original, Sentinel performs
comparably on the Synth task, suggesting robust-
ness to long-context scenarios despite minor for-
matting disruptions.
Overall, these results demonstrate that Sentinel
generalizes well to multilingual settings and re-
mains effective in compressing high-relevance con-
tent across languages.
Learning vs. Raw Attention Sentinel (0.5B)
outperforms Raw Attention (0.5B) on most tasks,
especially Single-Doc, Multi-Doc QA, and Syn-
thetic tasks. Improvements are consistent in En-
glish and smaller but generally positive in Chinese,
supporting the value of learning explicit relevance
over using raw attention scores.
3.2 Proxy Model Size Robustness
Our framework is based on the hypothesis that
query-context relevance, as reflected in attention,
is relatively stable across proxy model scales. To
evaluate this, we compare three Qwen-2.5 models
(0.5B, 1.5B, 3B) under the same training setup,
evaluated with 1024-token chunks and a 2,000-
token input budget. As shown in Table 2, perfor-
mance remains relatively stable across scales, with
average F1 varying by less than 2 points. The
0.5B model already achieves competitive results at
a much lower computational cost, supporting the
efficiency of using smaller proxy models.
To further assess alignment across models, we
compute pairwise sentence-level overlap in se-
lected sentences. Sentence-level overlap increases
with budget, ranging from 0.63–0.70 at 1000 tokens
to 0.74–0.78 at 3000 tokens, suggesting consistent
relevance estimation across proxy model scales.
Full results are provided in Appendix F.
These findings confirm that attention-based rele-
vance estimation is stable across model scales. A
0.5B proxy can closely approximate the sentence
selection behavior of larger models, enabling accu-
rate and efficient compression.
3.3 Ablation
We ablate three aspects of our method: attention
feature design, chunk size, and sentence selec-
tion strategy. Unless specified, all experiments
useQwen-2.5-7B-Instruct for generation and
are evaluated on LongBench.Feature HotpotQA SQuAD NewsQA Overall AUC Overall A VG
All Layers 0.9228 0.9987 0.9838 0.9700 51.23
Selected 0.9171 0.9943 0.9832 0.9662 50.20
Last Layer 0.8606 0.9538 0.9588 0.9121 49.04
Table 3: AUC comparison of different attention feature
extraction strategies and their effectiveness on Qwen-
2.5-7B-Instruct compressed evaluation.
3.3.1 Attention Feature Ablations
We evaluate three attention-based feature construc-
tion strategies on Qwen-2.5-0.5B-Instruct.
1.All Layers : Aggregate attention scores from
all decoder layers.
2.Last Layer : Use only the attention scores
from the final decoder layer.
3.Selected : Use mRMR (Ding and Peng, 2005)
to select a compact set of attention heads, lim-
ited to no more than those in one decoder layer.
See Appendix C for details.
As shown in Table 3, the All Layers strategy
achieves the highest AUC and downstream perfor-
mance, while the Selected variant offers a strong
trade-off between compactness and accuracy.
3.3.2 Chunk Size Variants
We evaluate Sentinel under varying chunk sizes
(512, 1024, 2048, 4096), all constrained to a 2000-
token budget during inference. Larger chunks allow
attention to span broader contiguous context, while
reducing the number of input segments. As shown
in Figure 2 (top), Sentinel consistently outperforms
the Raw Attention baseline across all chunk sizes.
Although the proxy was trained on sequences
shorter than 1024 tokens, performance continues
to improve with larger chunk sizes, reaching the
best result at 4096 tokens. This improvement may
stem from the model’s ability to attend over a wider
context window during inference, which facilitates
more effective context compression. Full task-level
results are provided in Appendix D.
3.3.3 Compression Ratio Variants
We further evaluate robustness under compression
constraints τ∈ {0.1,0.2,0.3,0.4,0.5}, where
lower τdenotes more aggressive context pruning.
As shown in Figure 2 (bottom), Raw attention
degrades sharply under tighter constraints, espe-
cially when τ <0.3. In contrast, Sentinel main-
tains stable performance across all compression
levels, with only mild degradation even at τ= 0.1.

Figure 2: Ablation results on Qwen-2.5-7B-Instruct
with 0.5B proxy. Top: Chunk size ablation under a
2000-token constraint. Bottom: Compression ratio
ablation at chunk size 1024.
These results highlight Sentinel’s robustness in low-
resource scenarios and its ability to extract semanti-
cally rich signals under strong token budgets. Full
task-level results are available in Appendix D.
3.3.4 Latency and Inference Efficiency
We evaluate end-to-end inference latency across
different Sentinel configurations, focusing on the
effects of chunk size and attention feature design.
Table 4 reports average and median latency per sam-
ple on the English LongBench dataset, measured
on a single A100 GPU.2
With a chunk size of 1024 and All Layers atten-
tion features, Sentinel achieves 1.13 ×speedup over
LLMLingua-2 while reaching 51.23 F1. Increasing
the chunk size to 2048 yields both higher accuracy
(51.68 F1) and slightly faster inference (0.65s avg),
demonstrating the benefit of longer-span attention
at inference time. Interestingly, using even larger
chunks (4096) leads to the best overall accuracy
(51.87 F1), but latency returns to 0.78s—on par
with LLMLingua-2. This is due to increased GPU
memory pressure from computing longer-range at-
tention, which becomes the primary bottleneck at
this scale.
2We monkey-patch the model to extract only the final-
token attention used by our method, replacing other activations
with None to reduce overhead.To further improve runtime, we evaluate SEN-
TINEL (SELECTED ), which uses compact mRMR-
selected features. At chunk size 1024, this variant
reduces latency to 0.60s (1.30 ×) with only minor
performance degradation (50.20 F1), offering an
efficient alternative for low-latency scenarios.
4 Analysis and Discussion
4.1 Probing Feature Distributions across
Datasets and Scales
We visualize probing features via t-SNE (Ap-
pendix G). Positive/negative samples are well-
separated in SQuAD and NewsQA, but overlap
in HotpotQA—highlighting the challenge of multi-
hop compression. The feature distributions remain
consistent across proxy scales, reinforcing the sta-
bility of attention-based relevance. We also vary
the number of probing examples (Appendix ??) and
observe stable performance across training sizes.
4.2 Attention Behavior and Sentence Selection
To better understand how Sentinel decodes
sentence-level relevance, we visualize attention be-
havior on an example from 2WikiMultihopQA , a
multi-hop QA task in LongBench. The input is pro-
cessed using the Qwen-2.5-0.5B-Instruct proxy
model with a chunk size of 1024.
We compare three strategies for visualizing sen-
tence importance: (1) Token-Level Attention ,
which directly displays the attention weights from
the final decoder token to each input token; (2) Av-
eraged Sentence Attention (corresponding to our
Raw Attention baseline), which computes sentence-
level scores by averaging attention weights over
each sentence’s tokens; and (3) Sentinel Probing
Prediction , which uses a trained classifier to output
sentence relevance probabilities based on attention-
derived features.
As shown in Appendix I, raw token-level atten-
tion exhibits a strong attention sink effect (Bon-
darenko et al., 2021; Son et al., 2024), with a large
portion of weights concentrated on the final token
in the input. As a result, sentence-level averaging
(i.e., the Raw Attention baseline) is heavily skewed
and often fails to reflect true semantic relevance.
While attention distributions do encode useful sig-
nals, they are noisy and unstructured. In contrast,
Sentinel’s probing-based classifier avoids attention
sink and more effectively decodes relevance pat-
terns embedded in the model’s behavior.
Compression as Understanding Unlike prior
methods that require training a dedicated com-

Method Chunk Size Proxy Model Avg. Time (s) ↓Med. Time (s) ↓Speedup vs. LLMLingua-2 ↑ Overall-A VG
LLMLingua-2 (trained) 512 XLM-RoBERTa-Large (561M) 0.78 0.70 1.00 × 32.65
Raw Attention 512 Qwen-2.5-0.5B-Instruct (494M) 1.01 0.84 0.77 × 45.70
Raw Attention 1024 Qwen-2.5-0.5B-Instruct (494M) 0.65 0.54 1.20 × 48.47
Sentinel (ours) 512 Qwen-2.5-0.5B-Instruct (494M) 1.02 0.84 0.76 × 49.51
Sentinel (ours) selected 512 Qwen-2.5-0.5B-Instruct (494M) 0.85 0.70 1.09 × 46.30
Sentinel (ours) 1024 Qwen-2.5-0.5B-Instruct (494M) 0.69 0.57 1.13 × 51.23
Sentinel (ours) selected 1024 Qwen-2.5-0.5B-Instruct (494M) 0.60 0.49 1.30 × 50.20
Sentinel (ours) 2048 Qwen-2.5-0.5B-Instruct (494M) 0.65 0.51 1.20 × 51.68
Sentinel (ours) 4096 Qwen-2.5-0.5B-Instruct (494M) 0.78 0.64 1.00 × 51.87
Table 4: Inference latency per QA sample on the full LongBench dataset (lower is better). LLMLingua-2 is trained
for token-level compression and limited to 512-token chunks. Sentinel uses a smaller, untrained decoder-only proxy
and supports larger chunk sizes for improved efficiency. Speedup is relative to LLMLingua-2 (chunk size 512).
Overall-A VG denotes average accuracy across all chunk settings per method.
pression model with large-scale labeled data, we
frame compression as a pure understanding prob-
lem: identifying which parts of the context are rel-
evant to the question. Crucially, we do not train a
compression model directly—instead, we probe the
attention behaviors of a small proxy LLM to extract
relevance signals already embedded in its internal
computation. This design leverages the model’s na-
tive understanding capabilities, rather than its gen-
eration capacity. Our experiments show that such
relevance behaviors are stable across model scales:
even a 0.5B proxy yields similar sentence-level
decisions as larger models. Although the probing
classifier is trained on a small set of external QA
data (3,000 examples), it generalizes effectively to
the diverse downstream tasks in LongBench. This
enables a lightweight and interpretable framework
for query-aware compression, with no need for
task-specific tuning or large-scale supervision.
5 Related Work
Token-Level Compression Token-level methods
aim to prune irrelevant or redundant content at fine
granularity. LLMLingua 1/2 (Jiang et al., 2023;
Pan et al., 2024) estimates token importance via
self-information and token selection distillation us-
ing small LMs. QGC (Cao et al., 2024) performs
query-guided compression by aligning token rep-
resentations with the query through pooling and
embedding refinement. While these approaches
achieve high compression ratios, they often frag-
ment discourse coherence.
Sentence-Level Compression Sentence-level
compression preserves semantic units by selecting
full sentences rather than tokens. Extractive meth-
ods such as RECOMP (Xu et al., 2024) and EXIT
(Hwang et al., 2024) formulate compression as a
binary classification task, supervised by generator
feedback or contrastive signals. CPC (Liskavets
et al., 2024), in contrast, learns a sentence encoderto rank sentences by query relevance, optimizing a
retrieval-style objective. Structure-enhanced meth-
ods like Refiner (Li et al., 2024) and FineFilter
(Zhang et al., 2025) further reorganize or rerank
selected content to support multi-hop reasoning
and long-context understanding. While these ap-
proaches are effective, they often require large-
scale supervision or task-specific tuning, which
limits their adaptability across tasks and models.
Attention-Based Compression Recent work ex-
plores leveraging decoder attention as a native rel-
evance signal. QUITO (Wang et al., 2024) intro-
duces a trigger token to guide attention-based to-
ken scoring, while ATTENTIONRAG (Fang et al.,
2025) reformulates QA as masked prediction and
uses attention to prune low-utility sentences. How-
ever, these methods often rely on prompt engineer-
ing or direct thresholding of raw attention, limiting
generality and robustness.
Our Approach In contrast to prior work, we pro-
pose Sentinel , a lightweight and model-agnostic
framework that probes native attention signals
from small proxy models to predict sentence
relevance. Rather than training a compression
model or relying on raw scoring mechanisms, Sen-
tinel treats compression as an understanding prob-
lem—decoding which parts of the input are inter-
nally attended to during question answering. By
empirically validating the stability of attention-
based relevance across model scales, our approach
enables efficient and interpretable compression
without generation supervision or task-specific
training.
6 Conclusion
We present Sentinel , a lightweight and inter-
pretable sentence-level compression framework
that probes multi-layer attention from an off-the-
shelf 0.5B proxy model to decode sentence rele-

vance—without supervision or task-specific tuning.
On LongBench, Sentinel achieves up to 5 ×com-
pression while matching the QA performance of
7B-scale systems across English and Chinese tasks.
These results show that attention probing offers an
efficient alternative to supervised compression, and
that small models can effectively support context
understanding in large language systems.

Limitations
Format Sensitivity in FewShot and Code Tasks.
Our approach uses generic sentence segmentation
and does not account for task-specific structural
constraints. As a result, performance on FewShot
and Code tasks may degrade due to formatting
disruption—such as broken input–output pairs in
FewShot prompts or misaligned line boundaries in
code. These tasks are inherently sensitive to layout
and structure, which current compression does not
explicitly preserve.
Proxy Model Scope. All probing experiments
use Qwen-2.5-Instruct models as the proxy. While
Qwen offers strong alignment and open availability,
we have not tested cross-architecture generalization
to other families such as LLaMA, Mistral, or GPT-
based models. Future work could explore whether
attention patterns are equally probe-able across di-
verse architectures.
Limited Evaluation Backbones. Our evaluation
is conducted on two decoder LLMs—GPT-3.5-
Turbo and Qwen-2.5-7B-Instruct. While these
provide good coverage of open and proprietary
systems, additional testing on other models (e.g.,
LLaMA, Mistral, Claude) would better estab-
lish generality across architectures and instruction
styles.
References
Yushi Bai, Xin Lv, Jiajie Zhang, Hongchang Lyu,
Jiankai Tang, Zhidian Huang, Zhengxiao Du, Xiao
Liu, Aohan Zeng, Lei Hou, et al. 2024. Longbench:
A bilingual, multitask benchmark for long context
understanding. In Proceedings of the 62nd Annual
Meeting of the Association for Computational Lin-
guistics (Volume 1: Long Papers) , pages 3119–3137.
Federico Barbero, Andrea Banino, Steven Kapturowski,
Dharshan Kumaran, João Madeira Araújo, Oleksandr
Vitvitskyi, Razvan Pascanu, and Petar Veli ˇckovi ´c.
2024. Transformers need glasses! information over-
squashing in language tasks. Advances in Neural
Information Processing Systems , 37:98111–98142.
Yelysei Bondarenko, Markus Nagel, and Tijmen
Blankevoort. 2021. Understanding and overcoming
the challenges of efficient transformer quantization.
InProceedings of the 2021 Conference on Empiri-
cal Methods in Natural Language Processing , pages
7947–7969. Association for Computational Linguis-
tics.
Tom Brown, Benjamin Mann, Nick Ryder, Melanie
Subbiah, Jared D Kaplan, Prafulla Dhariwal, ArvindNeelakantan, Pranav Shyam, Girish Sastry, Amanda
Askell, et al. 2020. Language models are few-shot
learners. Advances in neural information processing
systems , 33:1877–1901.
Zhiwei Cao, Qian Cao, Yu Lu, Ningxin Peng, Luyang
Huang, Shanbo Cheng, and Jinsong Su. 2024. Retain-
ing key information under high compression ratios:
Query-guided compressor for LLMs. In Proceedings
of the 62nd Annual Meeting of the Association for
Computational Linguistics (Volume 1: Long Papers) ,
pages 12685–12695.
Chris Ding and Hanchuan Peng. 2005. Minimum re-
dundancy feature selection from microarray gene ex-
pression data. Journal of bioinformatics and compu-
tational biology , 3(02):185–205.
Yixiong Fang, Tianran Sun, Yuling Shi, and Xiaodong
Gu. 2025. Attentionrag: Attention-guided context
pruning in retrieval-augmented generation. arXiv
preprint arXiv:2503.10720 .
Kelvin Guu, Kenton Lee, Zora Tung, Panupong Pasu-
pat, and Mingwei Chang. 2020. Retrieval augmented
language model pre-training. In International confer-
ence on machine learning , pages 3929–3938. PMLR.
Yanwen Huang, Yong Zhang, Ning Cheng, Zhitao Li,
Shaojun Wang, and Jing Xiao. 2025. Dynamic
attention-guided context decoding for mitigating con-
text faithfulness hallucinations in large language mod-
els.arXiv preprint arXiv:2501.01059 .
Taeho Hwang, Sukmin Cho, Soyeong Jeong, Hoyun
Song, SeungYoon Han, and Jong C Park. 2024. Exit:
Context-aware extractive compression for enhanc-
ing retrieval-augmented generation. arXiv preprint
arXiv:2412.12559 .
Huiqiang Jiang, Qianhui Wu, Chin-Yew Lin, Yuqing
Yang, and Lili Qiu. 2023. LLMLingua: Compressing
prompts for accelerated inference of large language
models. In Proceedings of the 2023 Conference on
Empirical Methods in Natural Language Processing ,
pages 13358–13376.
Huiqiang Jiang, Qianhui Wu, Xufang Luo, Dong-
sheng Li, Chin-Yew Lin, Yuqing Yang, and Lili Qiu.
2024a. LongLLMLingua: Accelerating and enhanc-
ing LLMs in long context scenarios via prompt com-
pression. In Proceedings of the 62nd Annual Meeting
of the Association for Computational Linguistics (Vol-
ume 1: Long Papers) , pages 1658–1677.
Ting Jiang, Shaohan Huang, Zhongzhi Luan, Deqing
Wang, and Fuzhen Zhuang. 2024b. Scaling sentence
embeddings with large language models. In Find-
ings of the Association for Computational Linguis-
tics: EMNLP 2024 , pages 3182–3196.
Vladimir Karpukhin, Barlas Oguz, Sewon Min, Patrick
Lewis, Ledell Wu, Sergey Edunov, Danqi Chen, and
Wen-tau Yih. 2020. Dense passage retrieval for open-
domain question answering. In Proceedings of the
2020 Conference on Empirical Methods in Natural
Language Processing (EMNLP) , pages 6769–6781.

Patrick Lewis, Ethan Perez, Aleksandra Piktus, Fabio
Petroni, Vladimir Karpukhin, Naman Goyal, Hein-
rich Küttler, Mike Lewis, Wen-tau Yih, Tim Rock-
täschel, et al. 2020. Retrieval-augmented generation
for knowledge-intensive nlp tasks. Advances in neu-
ral information processing systems , 33:9459–9474.
Yucheng Li, Bo Dong, Frank Guerin, and Chenghua Lin.
2023. Compressing context to enhance inference
efficiency of large language models. In Proceedings
of the 2023 Conference on Empirical Methods in
Natural Language Processing , pages 6342–6353.
Zhonghao Li, Xuming Hu, Aiwei Liu, Kening Zheng,
Sirui Huang, and Hui Xiong. 2024. Refiner : Restruc-
ture retrieved content efficiently to advance question-
answering capabilities. In Findings of the Associa-
tion for Computational Linguistics: EMNLP 2024 ,
pages 8548–8572.
Barys Liskavets, Maxim Ushakov, Shuvendu Roy,
Mark Klibanov, Ali Etemad, and Shane Luke. 2024.
Prompt compression with context-aware sentence en-
coding for fast and improved llm inference. arXiv
preprint arXiv:2409.01227 .
Nelson F. Liu, Kevin Lin, John Hewitt, Ashwin Paran-
jape, Michele Bevilacqua, Fabio Petroni, and Percy
Liang. 2024. Lost in the middle: How language mod-
els use long contexts. Transactions of the Association
for Computational Linguistics , 12:157–173.
Kevin Meng, David Bau, Alex Andonian, and Yonatan
Belinkov. 2022. Locating and editing factual associ-
ations in gpt. Advances in Neural Information Pro-
cessing Systems , 35:17359–17372.
OpenAI. 2024. Gpt-4 technical report.
Zhuoshi Pan, Qianhui Wu, Huiqiang Jiang, Menglin Xia,
Xufang Luo, Jue Zhang, Qingwei Lin, Victor Rühle,
Yuqing Yang, Chin-Yew Lin, H. Vicky Zhao, Lili Qiu,
and Dongmei Zhang. 2024. LLMLingua-2: Data dis-
tillation for efficient and faithful task-agnostic prompt
compression. In Findings of the Association for Com-
putational Linguistics: ACL 2024 , pages 963–981.
Weijia Shi, Sewon Min, Michihiro Yasunaga, Min-
joon Seo, Richard James, Mike Lewis, Luke Zettle-
moyer, and Wen-tau Yih. 2024. REPLUG: Retrieval-
augmented black-box language models. In Proceed-
ings of the 2024 Conference of the North American
Chapter of the Association for Computational Lin-
guistics: Human Language Technologies (Volume 1:
Long Papers) , pages 8371–8384.
Seungwoo Son, Wonpyo Park, Woohyun Han, Kyuyeun
Kim, and Jaeho Lee. 2024. Prefixing attention
sinks can mitigate activation outliers for large
language model quantization. arXiv preprint
arXiv:2406.12016 .
Wenshan Wang, Yihang Wang, Yixing Fan, Huaming
Liao, and Jiafeng Guo. 2024. Quito: Accelerating
long-context reasoning through query-guided context
compression. arXiv preprint arXiv:2408.00274 .Wenhao Wu, Yizhong Wang, Guangxuan Xiao, Hao
Peng, and Yao Fu. 2024. Retrieval head mechanisti-
cally explains long-context factuality. arXiv preprint
arXiv:2404.15574 .
Fangyuan Xu, Weijia Shi, and Eunsol Choi. 2024. RE-
COMP: Improving retrieval-augmented LMs with
context compression and selective augmentation. In
The Twelfth International Conference on Learning
Representations .
Ori Yoran, Tomer Wolfson, Ori Ram, and Jonathan Be-
rant. 2024. Making retrieval-augmented language
models robust to irrelevant context. In The Twelfth
International Conference on Learning Representa-
tions .
Qianchi Zhang, Hainan Zhang, Liang Pang, Hong-
wei Zheng, Yongxin Tong, and Zhiming Zheng.
2025. Finefilter: A fine-grained noise filtering mech-
anism for retrieval-augmented large language models.
arXiv preprint arXiv:2502.11811 .
A Dataset Details
We provide a detailed description of the datasets
used in our experiments, based on the English sub-
set of LongBench (Bai et al., 2024). LongBench is
a long-context benchmark covering diverse tasks
designed to evaluate the capabilities of language
models in understanding and reasoning over ex-
tended textual inputs. It consists of six task cat-
egories, each comprising multiple representative
datasets:
•Single-Document QA :
–NARRATIVE QA: Answer questions
based on a single narrative document,
such as a story or movie script.
–QASPER : Answer questions grounded in
a scientific paper.
–MULTI FIELD QA- EN: Answer factual
questions from a long structured ency-
clopedic entry.
•Multi-Document QA :
–HOTPOT QA,2W IKIMULTIHOP QA,
and MUSIQUE: Multi-hop QA tasks
requiring reasoning across multiple
passages to answer complex factoid
questions.
•Summarization :
–GOVREPORT : Summarize long govern-
ment reports.

–QMS UM: Summarize meeting tran-
scripts.
–MULTI NEWS: Summarize multi-source
news articles.
•Few-shot Reasoning :
–TREC: Classify question types.
–TRIVIA QA: Answer trivia-style factual
questions.
–SAMS UM: Summarize short dialogues.
–LSHT : Classify Chinese news headlines
into topic categories.
•Synthetic Retrieval :
–PASSAGE COUNT : Count the number of
unique paragraphs among potentially du-
plicated inputs.
–PASSAGE RETRIEVAL -EN: Identify the
source paragraph corresponding to a
given abstract.
•Code Completion :
–LCC : Predict the next line of code given
a code block, without an explicit natural
language query.
–REPOBENCH -P: Predict the next line of
a function given multi-file code context
and the function signature.
Dataset Filtering. We exclude two tasks— LCC
andPASSAGE COUNT —from our evaluation due
to incompatibility with query-conditioned com-
pression. LCC lacks an explicit query, requir-
ing the model to complete the final line of code
based solely on preceding lines. Without a query
to anchor attention, our method may prune essen-
tial lines that appear semantically uninformative.
While this can potentially be addressed by treating
the instruction “Next line of code” as a synthetic
query, we leave this for future work. The PAS-
SAGE COUNT task involves counting exact dupli-
cates, which is incompatible with lossy compres-
sion: small differences between seemingly redun-
dant paragraphs can lead to incorrect counts.
B Baseline Descriptions
We compare Sentinel against the following baseline
methods, grouped by their design paradigms:
•LLMLingua-1/2 (Jiang et al., 2023; Pan et al.,
2024): Token-level compression methodsbased on saliency estimation via perplexity
and LLM distillation. These methods are task-
agnostic and do not condition on the query.
•Selective-Context (Li et al., 2023): A
sentence-level, task-agnostic method that
scores context segments based on general in-
formativeness, independent of the question.
•LongLLMLingua (Jiang et al., 2024a): A
query-aware, multi-stage compression system
using query-conditioned perplexity scoring,
document reordering, and adaptive compres-
sion ratios.
•CPC (Liskavets et al., 2024): A contrastively
trained sentence-ranking model that selects
sentences based on semantic similarity to the
query in embedding space. It is query-aware
and trained on synthetic QA data.
•Raw Attention (Wang et al., 2024; Fang et al.,
2025): A non-learning baseline that selects
sentences by averaging attention weights from
the final decoder token. This mimics attention-
based heuristics used in prior work such as
QUITO and AttentionRAG.
•Random Selection : Sentences are sampled
uniformly at random until the token budget is
met. Serves as a lower-bound reference.
•Empty Context : The model receives only the
question without any retrieved context, serv-
ing as a zero-context baseline.
All baselines are evaluated under the same to-
ken budget and LLM generation setting for fair
comparison.
C mRMR Feature Selection Details
To construct a compact attention-based feature set,
we use the Minimum Redundancy Maximum Rel-
evance (mRMR) algorithm. We first compute mu-
tual information between each feature (i.e., atten-
tion head statistics) and the binary relevance label,
selecting the most informative one. We then itera-
tively add features that maximize relevance while
minimizing redundancy, measured via Pearson cor-
relation with already selected features. The number
of features is capped at the number of heads in a
single decoder layer to ensure compactness and
interpretability.

D Chunk Size and Compression Ratio
Details
We provide a subset of Table 2 to highlight the
effect of chunk size and compression ratio in our
ablation studies. The following tables report de-
tailed task-level performance using the 0.5B proxy
model.
Chunk Size. Table 5 reports results with dif-
ferent chunk sizes (512 to 4096 tokens), under a
fixed 2000-token constraint. Despite being trained
on short contexts, the model benefits from longer
chunks at inference, suggesting gains from preserv-
ing broader intra-chunk coherence.
Compression Ratio. Table 6 reports re-
sults with varying compression ratios
(τ∈ {0.1,0.2,0.3,0.4,0.5}), under a fixed
chunk size of 1024. Sentinel remains robust
even at high compression, while Raw attention
deteriorates significantly.
E Additional Chinese Results with
GPT-3.5-Turbo
To assess the cross-lingual robustness of our
method, we evaluate Sentinel on LongBench-Zh
using GPT-3.5-Turbo as the inference model. We
compare against LLMLingua and LLMLingua-2
baselines, which are evaluated under a 3,000-token
input constraint. Sentinel uses only 2,000 tokens
but consistently outperforms the baselines across
all task categories.
F Sentence-Level Overlap Across Proxy
Models
To examine the consistency of attention-based rele-
vance signals across proxy model scales, we com-
pute the pairwise sentence-level overlap of selected
sentences. Figure 3 shows the overlap heatmaps
between the 0.5B, 1.5B, and 3B models at differ-
ent token budgets (1000, 2000, and 3000). Token-
level overlap increases with budget, ranging from
0.63–0.70 at 1000 tokens to 0.74–0.78 at 3000 to-
kens, suggesting consistent relevance estimation
across proxy model scales.
G t-SNE Visualization of Probing
Features
To investigate how attention-derived sentence
features vary across datasets and proxy modelsizes, we visualize probing features using t-
SNE. Each point represents a sentence-level ex-
ample (either positive or negative), based on
decoder attention from different proxy models:
Qwen-2.5-0.5B-Instruct ,1.5B , and 3B. Visual-
izations are generated from 6,000 samples across
SQuAD, NewsQA, and HotpotQA.
Figure 4a shows the feature space from the 0.5B
proxy. We observe that positive and negative ex-
amples in SQuAD and NewsQA form distinguish-
able clusters, while HotpotQA features are more
entangled—likely due to the diffuse nature of multi-
hop supervision. This aligns with our observation
that multi-hop sentence relevance is harder to learn
from attention alone.
Figures 4b and 4c depict the same projection
under larger proxies. Notably, the overall structure
remains consistent, supporting our hypothesis that
attention-based relevance behavior is stable across
model scales.
H Additional Results: Training Size
Ablation
Effect of Probing Data Size. We evaluate how
training size affects probing quality. As shown
in Table 8, performance remains stable across
500–3000 training examples, with only marginal
gains. This suggests that even a small probing set
can support effective compression.
I Attention Visualization Examples
We provide qualitative visualizations of two ad-
jacent chunks from a 2WikiMultihopQA exam-
ple, processed by the Qwen-2.5-0.5B-Instruct
proxy model with chunk size 1024. Each visualiza-
tion illustrates three relevance estimation strategies:
•Top: Token-level attention weights from the
decoder’s final token.
•Middle: Averaged sentence attention, com-
puted by averaging token-level attention over
each sentence (Raw Attention baseline).
•Bottom: Sentence-level relevance predictions
from our probing-based classifier.
As shown in Figures 5 and 6, token-level at-
tention displays strong sink behavior, with most
weights concentrated on the final input token.
Sentence-level averaging slightly reduces noise but
remains sensitive to this sink effect and lacks se-
mantic alignment. In contrast, Sentinel’s probing

MethodsLongBench-En (Qwen-2.5-7B-Instruct, 2000-token constraint) LongBench-Zh (Qwen-2.5-7B-Instruct, 2000-token constraint)Overall A VG
SingleDoc MultiDoc Summ. FewShot *Synth. *Code En-A VG SingleDoc MultiDoc Summ. FewShot Synth. Zh-A VG
Raw Attention (512) 31.67 36.01 24.19 64.37 72.92 62.93 48.68 46.50 19.32 12.77 39.25 55.90 34.75 45.70
Raw Attention (1024) 33.11 38.39 24.22 63.27 77.67 62.14 49.80 56.24 20.99 13.61 44.50 72.76 41.62 48.47
Raw Attention (2048) 34.51 36.54 24.17 65.21 77.50 61.35 49.88 56.17 23.14 13.36 40.50 75.39 41.71 48.63
Raw Attention (4096) 34.79 41.17 24.20 65.59 77.08 62.46 50.88 56.90 23.94 13.42 43.50 75.70 42.69 49.66
Sentinel (512) 36.97 40.60 24.73 64.80 78.42 63.71 51.54 57.88 23.17 13.50 42.50 68.40 41.09 49.51
Sentinel (1024) 37.11 44.98 25.02 64.88 85.54 63.04 53.43 59.18 24.15 13.19 43.00 74.68 42.84 51.23
Sentinel (2048) 36.15 43.20 25.09 65.03 90.50 61.67 53.60 61.41 25.07 13.06 42.50 80.26 44.46 51.68
Sentinel (4096) 37.79 44.36 25.13 65.88 88.33 61.66 53.86 59.07 25.14 12.81 41.50 81.12 43.93 51.87
Table 5: Performance across chunk sizes with a fixed 2000-token constraint.
MethodsLongBench-En (Qwen-2.5-7B-Instruct, chunk size = 1024) LongBench-Zh (Qwen-2.5-7B-Instruct, chunk size = 1024)Overall A VG
SingleDoc MultiDoc Summ. FewShot *Synth. *Code En-A VG SingleDoc MultiDoc Summ. FewShot Synth. Zh-A VG
Raw Attention (ratio 0.1) 24.83 33.73 21.84 60.56 63.67 58.66 43.88 33.17 18.95 12.65 34.77 31.82 26.27 39.52
Raw Attention (ratio 0.2) 30.88 36.64 23.21 64.61 81.33 60.57 47.19 45.43 20.90 13.33 42.82 49.60 32.88 43.62
Raw Attention (ratio 0.3) 32.50 39.85 24.04 64.93 90.50 61.62 52.24 50.97 20.05 14.16 39.81 61.84 37.37 48.39
Raw Attention (ratio 0.4) 34.85 40.49 24.35 66.10 94.50 61.81 53.68 54.50 19.77 14.11 41.61 70.70 40.14 50.26
Raw Attention (ratio 0.5) 35.44 38.87 24.86 67.54 94.83 62.48 54.00 57.33 20.52 14.31 45.13 78.51 43.16 51.48
Sentinel (ratio 0.1) 34.28 38.14 22.93 60.69 75.04 58.86 48.33 56.91 22.89 13.06 38.64 56.04 37.51 45.78
Sentinel (ratio 0.2) 36.22 42.70 24.17 64.72 85.17 61.08 52.34 58.71 21.68 13.84 42.84 71.51 41.72 49.99
Sentinel (ratio 0.3) 36.79 42.42 24.66 66.72 92.00 62.11 54.12 58.05 22.37 14.31 41.02 77.78 42.71 51.54
Sentinel (ratio 0.4) 37.31 41.35 24.89 66.82 92.92 61.71 54.17 57.66 22.40 14.31 43.55 83.51 44.29 52.09
Sentinel (ratio 0.5) 38.22 40.98 24.90 65.56 94.75 61.58 54.10 59.21 21.18 14.68 43.40 83.59 44.41 51.99
Table 6: Performance across compression ratios (chunk size = 1024).
classifier produces sparse and interpretable predic-
tions that reliably highlight answer-supporting sen-
tences across chunks.
J LLM Evaluation Settings
For LLM-based evaluation, we adopt the official
prompt templates and decoding settings used in
LongBench (Bai et al., 2024) to ensure consistency
and comparability. The decoding parameters are
fixed across all datasets as follows:
•temperature : 0.0
•top_p : 1.0
•seed : 42
•n: 1
•stream : False
•max_tokens : dataset-specific (see Table 9)

MethodsLongBench-Zh (GPT-3.5-Turbo, 3000-token constraint) Compression Stats
SingleDoc MultiDoc Summ. FewShot Synth. A VG Tokens 1/ τ
LLMLingua 35.2 20.4 11.8 24.3 51.4 28.6 3,060 5x
LLMLingua-2 46.7 23.0 15.3 32.8 72.6 38.1 3,023 5x
Evaluated under 2000-token constraint
Sentinel (Qwen-2.5-0.5B-Instruct) 64.8 25.1 14.3 38.0 89.0 46.2 1,932 5x
Sentinel (Qwen-2.5-1.5B-Instruct) 63.3 24.9 14.8 40.3 95.0 47.6 1,929 5x
Original Prompt 61.2 28.7 16.0 29.2 77.5 42.5 14,940 -
Table 7: Performance comparison on LongBench-Zh using GPT-3.5-Turbo. LLMLingua baselines are evaluated un-
der a 3,000-token budget. Sentinel uses only 2,000 tokens but consistently outperforms the baselines, demonstrating
effective compression across languages.
(a) Token budget = 1000
 (b) Token budget = 2000
 (c) Token budget = 3000
Figure 3: Pairwise sentence-level overlap between proxy models at different token budgets. Higher overlap indicates
stronger alignment in relevance estimation.
(a)Qwen-2.5-0.5B-Instruct
 (b)Qwen-2.5-1.5B-Instruct
 (c)Qwen-2.5-3B-Instruct
Figure 4: t-SNE visualization of probing features across three proxy model scales. Each point represents a
sentence from SQuAD, NewsQA, or HotpotQA. Despite model size differences, the distributional structure remains
stable—supporting scale-invariant attention behavior.
MethodsLongBench-En (Qwen-2.5-7B-Instruct, 2000-token constraint) LongBench-Zh (Qwen-2.5-7B-Instruct, 2000-token constraint)Overall A VG
SingleDoc MultiDoc Summ. FewShot *Synth. *Code En-A VG SingleDoc MultiDoc Summ. FewShot Synth. Zh-A VG
Qwen-2.5-0.5B-Instruct (500) 36.42 44.54 24.99 66.01 89.58 63.12 54.11 60.60 24.52 13.22 43.25 75.42 43.40 51.73
Qwen-2.5-0.5B-Instruct (1000) 36.11 44.50 24.95 66.06 88.75 62.10 53.74 60.13 23.53 13.38 42.50 77.69 43.45 51.55
Qwen-2.5-0.5B-Instruct (2000) 36.67 44.58 24.88 64.52 87.04 62.91 53.43 60.55 25.42 13.11 41.25 74.13 42.89 51.16
Qwen-2.5-0.5B-Instruct (3000) 37.11 44.98 25.02 64.88 85.54 63.04 53.43 59.18 24.15 13.19 43.00 74.68 42.84 51.23
Table 8: Performance of 0.5B models with different probing sizes (500, 1000, 2000, 3000) on LongBench.

0 10 20 30 40 50 60 70 80
T oken Position in Sentence She met her father-in-law the King and her brothers-in-law at Stäket Manor on 27 October, and she continued to be well-treated and liked by them all during her life in Sweden. 
Thereafter, she met her mother-in-law the Queen and her sister-in-law at Säby Manor, and on the 28th, she was formally presented for the Swedish royal court at Drottningholm Palace. 
At this occasion, Countess Ebba Bonde noted that the impression about her was: "By God, how beautiful she is!
", but that her appearance was affected by the fact that she had a: "terrible fear of the Queen". 
On 4 November 1766, she was officially welcomed to the capital of Stockholm, where she was married to Gustav in person in the Royal Chapel at Stockholm Royal Palace.Sophia Magdalena initially made a good impression upon the Swedish nobility with her beauty, elegance and skillful dance; but her shy, silent, and reserved nature soon made her a disappointment in the society life. 
Being of a reserved nature, she was considered cold and arrogant. 
Her mother-in-law Queen Louisa Ulrika, who once stated that she could comprehend nothing more humiliating than the position of a Queen Dowager, harassed her in many ways: a typical example was when she invited Gustav to her birthday celebrations, but asked him to make Sophia Magdalena excuse herself by pretending to be too ill to attend. 
Louisa Ulrika encouraged a distance between the couple in various ways, and Gustav largely ignored her so as not to make his mother jealous.
Sophia Magdalena was known to be popular with the Caps, who were supported by Denmark, while Louisa Ulrika and Gustav sided with the Hats. 
The Caps regarded Sophia Magdalena to be a symbol of virtue and religion in a degenerated royal court, and officially demonstrated their support. 
Sophia Magdalena was advised by the Danish ambassador not to involve herself in politics, and when the spies of Louisa Ulrika reported that Sophia Magdalena received letters from the Danish ambassador through her Danish entourage, the Queen regarded her to be a sympathizer of the Danish-supported Caps: she was isolated from any contact with the Danish embassy, and the Queen encouraged Gustav to force her to send her Danish servants home. 
This she did not do until 1770, and his demand contributed to their tense and distant relationship. 
In 1768, Charlotta Sparre tried to reconcile the couple at their summer residence Ekolsund Castle, but the marriage remained unconsummated.After King Adolf Frederick of Sweden died in 1771, Gustav III became King of Sweden. 
The following year, on 29 May, Sophia Magdalena was crowned Queen.
Early reign as Queen
The coronation of Gustav III and Sophia Magdalena took place on 29 May 1772. 
She was not informed about the coup of Gustav III, which reinstated absolute monarchy and ended the parliamentary rule of the Estates in the revolution of 1772. 
At the time she was deemed as suspicious and politically untrustworthy in the eyes of the King, primarily by her mother-in-law, who painted her as pro-Danish. 
Denmark was presumed to oppose the coup; there were also plans to conquer Norway from Denmark.
Sophia Magdalena was informed about politics nonetheless: she expressed herself pleased with the 1772 parliament because Count Fredrik Ribbing, for whom she had taken an interest, had regained his seat. 
The conflict between her and her mother-in-law was publicly known and disliked, and the sympathies were on her side. 
In the contemporary paper Dagligt Allehanda, a fable was presented about Rävinnan och Turturduvan ("The She Fox and the Turtle Dove"). 
The fable was about the innocent turtle dove (Sophia Magdalena) who was slandered by the wicked she fox (Louisa Ulrika), who was supported by the second she fox (Anna Maria Hjärne) and the other foxes (the nobility). 
The fable was believed to have been sent from the Caps party.Queen Sophia Magdalena was of a shy and reserved character, and was never a member of the King's inner circle. 
At the famous amateur court theater of Gustav III, Sophia Magdalena is occasionally named as participator in the documents. 
In 1777, for example, she dressed as an Italian sailor and participated in a battle between Italian and Spanish sailors. 
Usually it was rather her role to act as the passive lady of games and tournaments, and to decorate the winner with the award. 
She did her ceremonial duties, but disliked the vivid lifestyle of the court around her outgoing spouse.As queen, she was expected to do a great deal of representation  more than what had been expected from previous queens due to her husband's adoration of representation.
Sentence IndexRaw Attention Values (T oken Level)
0.020.040.060.080.100.120.14
Raw Attention Value
0 10 20 30 40 50 60 70 80
T oken Position in Sentence She met her father-in-law the King and her brothers-in-law at Stäket Manor on 27 October, and she continued to be well-treated and liked by them all during her life in Sweden. 
Thereafter, she met her mother-in-law the Queen and her sister-in-law at Säby Manor, and on the 28th, she was formally presented for the Swedish royal court at Drottningholm Palace. 
At this occasion, Countess Ebba Bonde noted that the impression about her was: "By God, how beautiful she is!
", but that her appearance was affected by the fact that she had a: "terrible fear of the Queen". 
On 4 November 1766, she was officially welcomed to the capital of Stockholm, where she was married to Gustav in person in the Royal Chapel at Stockholm Royal Palace.Sophia Magdalena initially made a good impression upon the Swedish nobility with her beauty, elegance and skillful dance; but her shy, silent, and reserved nature soon made her a disappointment in the society life. 
Being of a reserved nature, she was considered cold and arrogant. 
Her mother-in-law Queen Louisa Ulrika, who once stated that she could comprehend nothing more humiliating than the position of a Queen Dowager, harassed her in many ways: a typical example was when she invited Gustav to her birthday celebrations, but asked him to make Sophia Magdalena excuse herself by pretending to be too ill to attend. 
Louisa Ulrika encouraged a distance between the couple in various ways, and Gustav largely ignored her so as not to make his mother jealous.
Sophia Magdalena was known to be popular with the Caps, who were supported by Denmark, while Louisa Ulrika and Gustav sided with the Hats. 
The Caps regarded Sophia Magdalena to be a symbol of virtue and religion in a degenerated royal court, and officially demonstrated their support. 
Sophia Magdalena was advised by the Danish ambassador not to involve herself in politics, and when the spies of Louisa Ulrika reported that Sophia Magdalena received letters from the Danish ambassador through her Danish entourage, the Queen regarded her to be a sympathizer of the Danish-supported Caps: she was isolated from any contact with the Danish embassy, and the Queen encouraged Gustav to force her to send her Danish servants home. 
This she did not do until 1770, and his demand contributed to their tense and distant relationship. 
In 1768, Charlotta Sparre tried to reconcile the couple at their summer residence Ekolsund Castle, but the marriage remained unconsummated.After King Adolf Frederick of Sweden died in 1771, Gustav III became King of Sweden. The following year, on 29 May, Sophia Magdalena was crowned Queen.
Early reign as Queen
The coronation of Gustav III and Sophia Magdalena took place on 29 May 1772. 
She was not informed about the coup of Gustav III, which reinstated absolute monarchy and ended the parliamentary rule of the Estates in the revolution of 1772. 
At the time she was deemed as suspicious and politically untrustworthy in the eyes of the King, primarily by her mother-in-law, who painted her as pro-Danish. 
Denmark was presumed to oppose the coup; there were also plans to conquer Norway from Denmark.
Sophia Magdalena was informed about politics nonetheless: she expressed herself pleased with the 1772 parliament because Count Fredrik Ribbing, for whom she had taken an interest, had regained his seat. 
The conflict between her and her mother-in-law was publicly known and disliked, and the sympathies were on her side. 
In the contemporary paper Dagligt Allehanda, a fable was presented about Rävinnan och Turturduvan ("The She Fox and the Turtle Dove"). 
The fable was about the innocent turtle dove (Sophia Magdalena) who was slandered by the wicked she fox (Louisa Ulrika), who was supported by the second she fox (Anna Maria Hjärne) and the other foxes (the nobility). 
The fable was believed to have been sent from the Caps party.Queen Sophia Magdalena was of a shy and reserved character, and was never a member of the King's inner circle. 
At the famous amateur court theater of Gustav III, Sophia Magdalena is occasionally named as participator in the documents. 
In 1777, for example, she dressed as an Italian sailor and participated in a battle between Italian and Spanish sailors. 
Usually it was rather her role to act as the passive lady of games and tournaments, and to decorate the winner with the award. 
She did her ceremonial duties, but disliked the vivid lifestyle of the court around her outgoing spouse.As queen, she was expected to do a great deal of representation  more than what had been expected from previous queens due to her husband's adoration of representation.
Sentence Index0.1476
0.0286
0.2409
0.2211
0.0567
0.0172
0.0000
0.1196
0.0916
0.0217
0.0211
0.0475
0.0565
0.2676
0.1745
0.0469
0.0803
0.1546
0.0644
0.0357
0.0925
0.0875
0.2450
0.1413
0.1562
0.1223
1.0000Raw Attention Filter Scores (Sentence Level applied to tokens)
0 10 20 30 40 50 60 70 80
T oken Position in Sentence She met her father-in-law the King and her brothers-in-law at Stäket Manor on 27 October, and she continued to be well-treated and liked by them all during her life in Sweden. 
Thereafter, she met her mother-in-law the Queen and her sister-in-law at Säby Manor, and on the 28th, she was formally presented for the Swedish royal court at Drottningholm Palace. 
At this occasion, Countess Ebba Bonde noted that the impression about her was: "By God, how beautiful she is!
", but that her appearance was affected by the fact that she had a: "terrible fear of the Queen". 
On 4 November 1766, she was officially welcomed to the capital of Stockholm, where she was married to Gustav in person in the Royal Chapel at Stockholm Royal Palace.Sophia Magdalena initially made a good impression upon the Swedish nobility with her beauty, elegance and skillful dance; but her shy, silent, and reserved nature soon made her a disappointment in the society life. 
Being of a reserved nature, she was considered cold and arrogant. 
Her mother-in-law Queen Louisa Ulrika, who once stated that she could comprehend nothing more humiliating than the position of a Queen Dowager, harassed her in many ways: a typical example was when she invited Gustav to her birthday celebrations, but asked him to make Sophia Magdalena excuse herself by pretending to be too ill to attend. 
Louisa Ulrika encouraged a distance between the couple in various ways, and Gustav largely ignored her so as not to make his mother jealous.
Sophia Magdalena was known to be popular with the Caps, who were supported by Denmark, while Louisa Ulrika and Gustav sided with the Hats. 
The Caps regarded Sophia Magdalena to be a symbol of virtue and religion in a degenerated royal court, and officially demonstrated their support. 
Sophia Magdalena was advised by the Danish ambassador not to involve herself in politics, and when the spies of Louisa Ulrika reported that Sophia Magdalena received letters from the Danish ambassador through her Danish entourage, the Queen regarded her to be a sympathizer of the Danish-supported Caps: she was isolated from any contact with the Danish embassy, and the Queen encouraged Gustav to force her to send her Danish servants home. 
This she did not do until 1770, and his demand contributed to their tense and distant relationship. 
In 1768, Charlotta Sparre tried to reconcile the couple at their summer residence Ekolsund Castle, but the marriage remained unconsummated.After King Adolf Frederick of Sweden died in 1771, Gustav III became King of Sweden. The following year, on 29 May, Sophia Magdalena was crowned Queen.
Early reign as Queen
The coronation of Gustav III and Sophia Magdalena took place on 29 May 1772. 
She was not informed about the coup of Gustav III, which reinstated absolute monarchy and ended the parliamentary rule of the Estates in the revolution of 1772. 
At the time she was deemed as suspicious and politically untrustworthy in the eyes of the King, primarily by her mother-in-law, who painted her as pro-Danish. 
Denmark was presumed to oppose the coup; there were also plans to conquer Norway from Denmark.
Sophia Magdalena was informed about politics nonetheless: she expressed herself pleased with the 1772 parliament because Count Fredrik Ribbing, for whom she had taken an interest, had regained his seat. 
The conflict between her and her mother-in-law was publicly known and disliked, and the sympathies were on her side. 
In the contemporary paper Dagligt Allehanda, a fable was presented about Rävinnan och Turturduvan ("The She Fox and the Turtle Dove"). 
The fable was about the innocent turtle dove (Sophia Magdalena) who was slandered by the wicked she fox (Louisa Ulrika), who was supported by the second she fox (Anna Maria Hjärne) and the other foxes (the nobility). 
The fable was believed to have been sent from the Caps party.Queen Sophia Magdalena was of a shy and reserved character, and was never a member of the King's inner circle. 
At the famous amateur court theater of Gustav III, Sophia Magdalena is occasionally named as participator in the documents. 
In 1777, for example, she dressed as an Italian sailor and participated in a battle between Italian and Spanish sailors. 
Usually it was rather her role to act as the passive lady of games and tournaments, and to decorate the winner with the award. 
She did her ceremonial duties, but disliked the vivid lifestyle of the court around her outgoing spouse.As queen, she was expected to do a great deal of representation  more than what had been expected from previous queens due to her husband's adoration of representation.
Sentence Index0.4311
0.2796
0.4373
0.2262
0.3019
0.2085
0.2686
0.3107
0.4137
0.2399
0.2500
0.2227
0.3147
0.2553
0.2972
0.2609
0.2282
0.1650
0.2262
0.2624
0.2459
0.4302
0.3051
0.2184
0.1641
0.1658
0.3403LR Model Predicted Probabilities (Sentence Level applied to tokens)
0.00.20.40.60.81.0
Score / Similarity (0-1)Question: Who is the spouse of the director of film Streets Of Blood?
Gold Answer: Sandra Nelson
Sample ID: 1000_2 - T otal Sentences: 27 - T oken Counts: [43, 47, 27, 24, 83, 14, 70, 29, 34, 29, 88, 24, 55, 18, 31, 36, 37, 19, 44, 25, 37, 58, 41, 26, 28, 27, 51]Figure 5: Visualization of the second chunk in a 2WikiMultihopQA example. Top: token-level attention weights
from the final decoder token. Middle: sentence-level scores from averaged token attention (Raw Attention baseline).
Bottom: sentence-level relevance predictions from the probing-based classifier.
0 20 40 60 80
T oken Position in Sentence On formal occasions, she was at her best: she performed beautifully according to royal court etiquette, and was seen as dignified and impressive. 
For instance, on 17 September 1784, she cut the cord to let off the first air balloons from the Stockholm observatory. 
During the King's Italian journey in 1783 84, she hosted a grand formal public dinner every two weeks. 
During that time, she appeared at the Royal Swedish Opera and at the French Theater, but otherwise preferred her solitude. 
This attracted attention as during the absence of the King she had been expected to represent the royal couple all the more.
Sophia appeared to have enjoyed nature trips in the country side with only one lady-in-waiting and two footmen, however, her country side visitations were stopped because it was deemed 'unsuitable'. 
Several of her ladies-in-waiting were well known Swedish women of the era, among them The Three Graces: Augusta von Fersen, Ulla von Höpken and Lovisa Meijerfelt, as well as Marianne Ehrenström and Charlotta Cedercreutz, who were known artists.
Sophia Magdalena was a popular Queen: on 22 July 1788, for example, during the absence of her spouse in Finland, several members of the Royal Dramatic Theater and the musical society Augustibröder, among them Bellman, took a spontaneous trip by boat from the capital to Ulriksdal Palace, where she was, and performed a poem by Bellman to her honor at the occasion of her name day.
In the famous diary of her sister-in-law, Princess Hedwig Elizabeth Charlotte, Sophia Magdalena is described as beautiful, cold, silent and haughty, very polite and formal, reserved and unsociable. 
When she performed her duties as Queen, her sister-in-law, Hedwig Elizabeth Charlotte of Holstein-Gottorp, described her as "Forced to meet people".Sophia Magdalena preferred to spend her days in solitude whenever she could. 
She had two very intimate friends, Maria Aurora Uggla and Baroness Virginia Charlotta Manderström, but otherwise rarely participated in any social life outside of what was absolutely necessary to perform her representational  duties. 
She frequently visited the theater, and she also had a great interest for fashion. 
As a result of this, she was somewhat criticized for being too vain: even when she had no representational duties to dress up for and spend her days alone in her rooms, she is said to have changed costumes several times daily, and according her chamberlain Adolf Ludvig Hamilton, she never passed a mirror without studying herself in it. 
She was also interested in literature, and educated herself in various subjects: her library contained works about geography, genealogy and history. 
She educated herself in Swedish, English, German and Italian, and regularly read French magazines. 
According to Augusta von Fersen, Sophia Magdalena was quite educated, but she was not perceived as such because she rarely engaged in conversation.In 1784, after the King had returned from his trip to Italy and France, the relationship between the King and Queen soured. 
At this time, Gustav III spent more and more time with male favorites. 
In 1786, this came to an open conflict. 
The King had taken to spend more time at intimate evenings with his favorite Gustaf Mauritz Armfelt, from which he excluded her company. 
When he gave some of her rooms at the Royal Palace to Armfelt, Sophia Magdalena refused to participate in any representation until the rooms were given back to her, and she also banned her ladies-in-waiting from accepting his invitations without her permission.
In 1787, she threatened him with asking for the support of the parliament against him if he took their son with him to Finland, which she opposed, and the year after, she successfully prevented him from doing so. 
She also reprimanded him from allowing his male favorites to slander her before him.
Queen Sophia Magdalena was never involved in politics, except for one on one occasion. 
In August 1788, during the Russo-Swedish War (1788 1790), the King gave her the task to enter in negotiations with Denmark to prevent a declaration of war from Denmark during the ongoing war against Russia. 
He asked her to call upon the Danish ambassador Reventlow and give him a letter to be read in the Danish royal council before her brother, the Danish King. 
He gave her the freedom to write as she wished, but to use the argument that she spoke as a sister and mother to a son with the right to the Danish throne and upon her own initiative.
Sophia Magdalena called upon the Danish ambassador, held a speech to him followed by a long conversation and then handed him a letter written as a "warm appeal" to her brother.Sentence IndexRaw Attention Values (T oken Level)
0.020.040.060.080.10
Raw Attention Value
0 20 40 60 80
T oken Position in Sentence On formal occasions, she was at her best: she performed beautifully according to royal court etiquette, and was seen as dignified and impressive. 
For instance, on 17 September 1784, she cut the cord to let off the first air balloons from the Stockholm observatory. 
During the King's Italian journey in 1783 84, she hosted a grand formal public dinner every two weeks. 
During that time, she appeared at the Royal Swedish Opera and at the French Theater, but otherwise preferred her solitude. 
This attracted attention as during the absence of the King she had been expected to represent the royal couple all the more.
Sophia appeared to have enjoyed nature trips in the country side with only one lady-in-waiting and two footmen, however, her country side visitations were stopped because it was deemed 'unsuitable'. 
Several of her ladies-in-waiting were well known Swedish women of the era, among them The Three Graces: Augusta von Fersen, Ulla von Höpken and Lovisa Meijerfelt, as well as Marianne Ehrenström and Charlotta Cedercreutz, who were known artists.
Sophia Magdalena was a popular Queen: on 22 July 1788, for example, during the absence of her spouse in Finland, several members of the Royal Dramatic Theater and the musical society Augustibröder, among them Bellman, took a spontaneous trip by boat from the capital to Ulriksdal Palace, where she was, and performed a poem by Bellman to her honor at the occasion of her name day.
In the famous diary of her sister-in-law, Princess Hedwig Elizabeth Charlotte, Sophia Magdalena is described as beautiful, cold, silent and haughty, very polite and formal, reserved and unsociable. 
When she performed her duties as Queen, her sister-in-law, Hedwig Elizabeth Charlotte of Holstein-Gottorp, described her as "Forced to meet people".Sophia Magdalena preferred to spend her days in solitude whenever she could. 
She had two very intimate friends, Maria Aurora Uggla and Baroness Virginia Charlotta Manderström, but otherwise rarely participated in any social life outside of what was absolutely necessary to perform her representational  duties. 
She frequently visited the theater, and she also had a great interest for fashion. 
As a result of this, she was somewhat criticized for being too vain: even when she had no representational duties to dress up for and spend her days alone in her rooms, she is said to have changed costumes several times daily, and according her chamberlain Adolf Ludvig Hamilton, she never passed a mirror without studying herself in it. 
She was also interested in literature, and educated herself in various subjects: her library contained works about geography, genealogy and history. 
She educated herself in Swedish, English, German and Italian, and regularly read French magazines. 
According to Augusta von Fersen, Sophia Magdalena was quite educated, but she was not perceived as such because she rarely engaged in conversation.In 1784, after the King had returned from his trip to Italy and France, the relationship between the King and Queen soured. 
At this time, Gustav III spent more and more time with male favorites. 
In 1786, this came to an open conflict. 
The King had taken to spend more time at intimate evenings with his favorite Gustaf Mauritz Armfelt, from which he excluded her company. 
When he gave some of her rooms at the Royal Palace to Armfelt, Sophia Magdalena refused to participate in any representation until the rooms were given back to her, and she also banned her ladies-in-waiting from accepting his invitations without her permission.
In 1787, she threatened him with asking for the support of the parliament against him if he took their son with him to Finland, which she opposed, and the year after, she successfully prevented him from doing so. 
She also reprimanded him from allowing his male favorites to slander her before him.
Queen Sophia Magdalena was never involved in politics, except for one on one occasion. 
In August 1788, during the Russo-Swedish War (1788 1790), the King gave her the task to enter in negotiations with Denmark to prevent a declaration of war from Denmark during the ongoing war against Russia. 
He asked her to call upon the Danish ambassador Reventlow and give him a letter to be read in the Danish royal council before her brother, the Danish King. 
He gave her the freedom to write as she wished, but to use the argument that she spoke as a sister and mother to a son with the right to the Danish throne and upon her own initiative.
Sophia Magdalena called upon the Danish ambassador, held a speech to him followed by a long conversation and then handed him a letter written as a "warm appeal" to her brother.Sentence Index0.2709
0.0765
0.0205
0.0744
0.3829
0.1547
0.0925
0.0581
0.0895
0.0640
0.0784
0.0232
0.0180
0.0000
0.0048
0.1832
0.1559
0.0286
0.0420
0.1502
0.1110
0.1360
0.3251
0.1238
0.1116
0.3287
1.0000Raw Attention Filter Scores (Sentence Level applied to tokens)
0 20 40 60 80
T oken Position in Sentence On formal occasions, she was at her best: she performed beautifully according to royal court etiquette, and was seen as dignified and impressive. 
For instance, on 17 September 1784, she cut the cord to let off the first air balloons from the Stockholm observatory. 
During the King's Italian journey in 1783 84, she hosted a grand formal public dinner every two weeks. 
During that time, she appeared at the Royal Swedish Opera and at the French Theater, but otherwise preferred her solitude. 
This attracted attention as during the absence of the King she had been expected to represent the royal couple all the more.
Sophia appeared to have enjoyed nature trips in the country side with only one lady-in-waiting and two footmen, however, her country side visitations were stopped because it was deemed 'unsuitable'. 
Several of her ladies-in-waiting were well known Swedish women of the era, among them The Three Graces: Augusta von Fersen, Ulla von Höpken and Lovisa Meijerfelt, as well as Marianne Ehrenström and Charlotta Cedercreutz, who were known artists.
Sophia Magdalena was a popular Queen: on 22 July 1788, for example, during the absence of her spouse in Finland, several members of the Royal Dramatic Theater and the musical society Augustibröder, among them Bellman, took a spontaneous trip by boat from the capital to Ulriksdal Palace, where she was, and performed a poem by Bellman to her honor at the occasion of her name day.
In the famous diary of her sister-in-law, Princess Hedwig Elizabeth Charlotte, Sophia Magdalena is described as beautiful, cold, silent and haughty, very polite and formal, reserved and unsociable. 
When she performed her duties as Queen, her sister-in-law, Hedwig Elizabeth Charlotte of Holstein-Gottorp, described her as "Forced to meet people".Sophia Magdalena preferred to spend her days in solitude whenever she could. 
She had two very intimate friends, Maria Aurora Uggla and Baroness Virginia Charlotta Manderström, but otherwise rarely participated in any social life outside of what was absolutely necessary to perform her representational  duties. 
She frequently visited the theater, and she also had a great interest for fashion. 
As a result of this, she was somewhat criticized for being too vain: even when she had no representational duties to dress up for and spend her days alone in her rooms, she is said to have changed costumes several times daily, and according her chamberlain Adolf Ludvig Hamilton, she never passed a mirror without studying herself in it. 
She was also interested in literature, and educated herself in various subjects: her library contained works about geography, genealogy and history. 
She educated herself in Swedish, English, German and Italian, and regularly read French magazines. 
According to Augusta von Fersen, Sophia Magdalena was quite educated, but she was not perceived as such because she rarely engaged in conversation.In 1784, after the King had returned from his trip to Italy and France, the relationship between the King and Queen soured. 
At this time, Gustav III spent more and more time with male favorites. 
In 1786, this came to an open conflict. 
The King had taken to spend more time at intimate evenings with his favorite Gustaf Mauritz Armfelt, from which he excluded her company. 
When he gave some of her rooms at the Royal Palace to Armfelt, Sophia Magdalena refused to participate in any representation until the rooms were given back to her, and she also banned her ladies-in-waiting from accepting his invitations without her permission.
In 1787, she threatened him with asking for the support of the parliament against him if he took their son with him to Finland, which she opposed, and the year after, she successfully prevented him from doing so. 
She also reprimanded him from allowing his male favorites to slander her before him.
Queen Sophia Magdalena was never involved in politics, except for one on one occasion. 
In August 1788, during the Russo-Swedish War (1788 1790), the King gave her the task to enter in negotiations with Denmark to prevent a declaration of war from Denmark during the ongoing war against Russia. 
He asked her to call upon the Danish ambassador Reventlow and give him a letter to be read in the Danish royal council before her brother, the Danish King. 
He gave her the freedom to write as she wished, but to use the argument that she spoke as a sister and mother to a son with the right to the Danish throne and upon her own initiative.
Sophia Magdalena called upon the Danish ambassador, held a speech to him followed by a long conversation and then handed him a letter written as a "warm appeal" to her brother.Sentence Index0.3533
0.2930
0.2769
0.2825
0.4591
0.3381
0.3506
0.3231
0.3861
0.3286
0.2667
0.2366
0.2245
0.2191
0.2335
0.3128
0.6918
0.2176
0.3442
0.2147
0.2102
0.1426
0.5137
0.2556
0.2042
0.1477
0.1977LR Model Predicted Probabilities (Sentence Level applied to tokens)
0.00.20.40.60.81.0
Score / Similarity (0-1)Question: Who is the spouse of the director of film Streets Of Blood?
Gold Answer: Sandra Nelson
Sample ID: 1000_3 - T otal Sentences: 27 - T oken Counts: [29, 32, 28, 24, 23, 43, 65, 93, 44, 51, 46, 17, 69, 27, 19, 59, 17, 15, 29, 52, 48, 17, 19, 54, 34, 40, 38]
Figure 6: Visualization of the third chunk in a 2WikiMultihopQA example. Top: token-level attention weights from
the final decoder token. Middle: sentence-level scores from averaged token attention (Raw Attention baseline).
Bottom: sentence-level relevance predictions from the probing-based classifier.

Dataset Max Tokens
narrativeqa 128
qasper 128
multifieldqa_en 64
multifieldqa_zh 64
hotpotqa 32
2wikimqa 32
musique 32
dureader 128
gov_report 512
qmsum 512
multi_news 512
vcsum 512
trec 64
triviaqa 32
samsum 128
lsht 64
passage_count 32
passage_retrieval_en 32
passage_retrieval_zh 32
lcc 64
repobench-p 64
Table 9: Maximum number of generation tokens for each dataset.