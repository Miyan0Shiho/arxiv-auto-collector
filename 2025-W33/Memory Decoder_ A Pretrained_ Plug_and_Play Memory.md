# Memory Decoder: A Pretrained, Plug-and-Play Memory for Large Language Models

**Authors**: Jiaqi Cao, Jiarui Wang, Rubin Wei, Qipeng Guo, Kai Chen, Bowen Zhou, Zhouhan Lin

**Published**: 2025-08-13 15:16:29

**PDF URL**: [http://arxiv.org/pdf/2508.09874v1](http://arxiv.org/pdf/2508.09874v1)

## Abstract
Large Language Models (LLMs) have shown strong abilities in general language
tasks, yet adapting them to specific domains remains a challenge. Current
method like Domain Adaptive Pretraining (DAPT) requires costly full-parameter
training and suffers from catastrophic forgetting. Meanwhile,
Retrieval-Augmented Generation (RAG) introduces substantial inference latency
due to expensive nearest-neighbor searches and longer context. This paper
introduces Memory Decoder, a plug-and-play pretrained memory that enables
efficient domain adaptation without changing the original model's parameters.
Memory Decoder employs a small transformer decoder that learns to imitate the
behavior of an external non-parametric retriever. Once trained, Memory Decoder
can be seamlessly integrated with any pretrained language model that shares the
same tokenizer, requiring no model-specific modifications. Experimental results
demonstrate that Memory Decoder enables effective adaptation of various Qwen
and Llama models to three distinct specialized domains: biomedicine, finance,
and law, reducing perplexity by an average of 6.17 points. Overall, Memory
Decoder introduces a novel paradigm centered on a specially pretrained memory
component designed for domain-specific adaptation. This memory architecture can
be integrated in a plug-and-play manner, consistently enhancing performance
across multiple models within the target domain.

## Full Text


<!-- PDF content starts -->

Memory Decoder : A Pretrained, Plug-and-Play
Memory for Large Language Models
Jiaqi Cao1∗, Jiarui Wang1∗, Rubin Wei1, Qipeng Guo2, Kai Chen2,
Bowen Zhou2,3, Zhouhan Lin1,2†
1LUMIA Lab, Shanghai Jiao Tong University, Shanghai, China
2Shanghai AI Laboratory, Shanghai, China
3Department of Electronic Engineering, Tsinghua University, Beijing, China
Abstract
Large Language Models (LLMs) have shown strong abilities in general language
tasks, yet adapting them to specific domains remains a challenge. Current method
like Domain Adaptive Pretraining (DAPT) requires costly full-parameter train-
ing and suffers from catastrophic forgetting. Meanwhile, Retrieval-Augmented
Generation (RAG) introduces substantial inference latency due to expensive nearest-
neighbor searches and longer context. This paper introduces Memory Decoder , a
plug-and-play pretrained memory that enables efficient domain adaptation without
changing the original model’s parameters. Memory Decoder employs a small trans-
former decoder that learns to imitate the behavior of an external non-parametric
retriever. Once trained, Memory Decoder can be seamlessly integrated with any pre-
trained language model that shares the same tokenizer, requiring no model-specific
modifications. Experimental results demonstrate that Memory Decoder enables
effective adaptation of various Qwen and Llama models to three distinct specialized
domains: biomedicine, finance, and law, reducing perplexity by an average of 6.17
points. Overall, Memory Decoder introduces a novel paradigm centered on a
specially pretrained memory component designed for domain-specific adaptation.
This memory architecture can be integrated in a plug-and-play manner, consistently
enhancing performance across multiple models within the target domain.1
1 Introduction
Large Language Models (LLMs) have demonstrated remarkable capabilities across a wide range of
natural language processing tasks (Grattafiori et al., 2024; Yang et al., 2024; Liu et al., 2024; Guo
et al., 2025). Pretrained on vast corpora of general text data, LLMs have revolutionized how we
approach language understanding and generation tasks. However, despite their impressive general
capabilities, adapting LLMs to perform optimally in specific domains remains a significant challenge.
Domain-specific adaptation is crucial for applications in specialized fields such as biomedicine,
finance, and law (Chen et al., 2023; Liu et al., 2023b; Colombo et al., 2024), where domain expertise
and terminology are essential for accurate and reliable performance.
*Equal contribution.
†Corresponding author.
1Code and checkpoints will be released upon acceptance.
Preprint. Under review.arXiv:2508.09874v1  [cs.CL]  13 Aug 2025

Figure 1: Comparison of domain adaptation approaches. DAPT (left) requires separate pre-training
for each model size, modifying original parameters. RAG (middle) maintains model parameters
but requires expensive retrieval from external datastores during inference. Memory Decoder (right)
offers a plug-and-play solution where a single pretrained memory component can be interpolated
with models of different sizes, avoiding both parameter modification and retrieval overhead.
Figure 2: Perplexity comparison of
Qwen2.5 models augmented by Memory
Decoder and LoRA adapter of the same
param count on the finance domain.Domain adaptation for pretrained language models has
traditionally followed several approaches, each with dis-
tinct advantages and limitations. Domain Adaptive Pre-
Training (DAPT) involves continued pre-training of the
LLM on domain-specific corpora (Gururangan et al.,
2020). While effective, this approach suffers from sub-
stantial computational costs associated with full-parameter
training, especially as model sizes continue to grow into
billions of parameters. Furthermore, adapting multiple
models to the same domain requires separate training runs
for each model, leading to resource inefficiency. Even
with successful DAPT implementation, these models often
encounter catastrophic forgetting, where the adaptation
process diminishes the model’s general capabilities (Kirk-
patrick et al., 2017; Ven van de et al., 2024).
Retrieval-Augmented Generation (RAG) offers an alterna-
tive approach by enhancing model outputs with relevant
retrieved information (Lewis et al., 2020a; Izacard et al., 2023a). While this method preserves the
original model parameters, it introduces substantial computation overhead during inference due to
expensive nearest neighbor ( kNN) searches across large datastores and extended context (He et al.,
2021a).
These two approaches present a fundamental dilemma in domain adaptation: DAPT requires costly
training procedures and cannot efficiently adapt multiple models to the same domain, while RAG
introduces significant computation and storage overhead during inference. This inherent trade-off
between the plug-and-play nature of RAG and the inference efficiency of DAPT highlights the
research gap for a solution offering both adaptability across models and computational efficiency
during deployment. To address this challenge, we propose Memory Decoder (MemDec), a plug-and-
play pretrained memory designed for efficient domain adaptation of large language models without
modifying their parameters. Our approach draws inspiration from retrieval-based methods like kNN-
LM (Khandelwal et al., 2019a), but overcomes their limitations through a different paradigm. Rather
than building and searching model-specific datastores during inference, Memory Decoder employs a
small transformer decoder that is specially pretrained to imitate the behavior of non-parametric
retrievers by aligning its output distribution with the ones of non-parametric retrievers. Figure 1
illustrates how our approach differs fundamentally from both DAPT and RAG.
The key innovation of our approach lies in its plug-and-play functionality: once trained, a single
Memory Decoder can be seamlessly integrated with any large language model that shares the same
tokenizer, without requiring model-specific adaptations or additional training. This architectural
design enables immediate deployment across diverse model architectures, significantly reducing
2

the computational resources needed for domain adaptation pre-training. Furthermore, unlike RAG
methods, Memory Decoder achieves domain-specific performance improvements with minimal
impact on inference latency, combining versatility with computational efficiency.
Experimental results across three specialized domains (biomedicine, finance, and law) and multiple
model architectures demonstrate the versatility of Memory Decoder. As shown in Figure 2. the
same Memory Decoder with only 0.5B parameters consistently enhances performance across seven
different models from the Qwen2.5 model family on the finance domain. Our comprehensive analysis
confirms that Memory Decoder successfully preserves the advantages of non-parametric approaches
while eliminating their computational overhead, establishing a new paradigm for efficient domain
adaptation of LLMs.
Our contributions can be summarized as follows:
•We introduce Memory Decoder , a plug-and-play pretrained memory that enables efficient
domain adaptation for large language models without modifying their original parameters.
•We present the first approach that replaces traditional non-parametric retrievers with a
compact parametric model, achieving superior performance while eliminating costly retrieval
operations during inference.
•We demonstrate Memory Decoder ’s generalizability, where a single domain-specific pre-
trained memory can be seamlessly integrated across all models with the same tokenizer.
2 Background
2.1 Problem Formulation
Domain adaptation aims to enhance a pretrained language model’s performance on specialized text.
Formally, given a pretrained model MPLMwith parameters θand a domain corpus Ddomain , the
goal is to optimize the next-token prediction distribution pPLM(yt|x;θ)for the target domain. Here,
x= (x1, x2, ..., x t−1)represents the context sequence and ytdenotes the target token.
2.2 Nearest Neighbor Language Models
Thek-nearest neighbor language model (kNN-LM) (Khandelwal et al., 2019a) enables non-parametric
domain adaptation without modifying the pretrained model’s parameters.
For a domain corpus, kNN-LM first constructs a key-value datastore:
(K, V) ={(ϕ(xi), yi)|(xi, yi)∈ D domain} (1)
where ϕ(·)extracts hidden representations from the pretrained model.
During inference, for context x, it computes kt=ϕ(x), retrieves k-nearest neighbors, and constructs
a probability distribution:
pkNN(yt|x)∝X
(ki,vi)∈N(kt,k)1yt=viexp(−d(kt, ki)/τ) (2)
The final prediction interpolates between the pretrained model and kNN distributions:
pkNN-PLM (yt|x) =λ·pkNN(yt|x) + (1 −λ)·pPLM(yt|x) (3)
While effective, kNN-LM introduces substantial computational and storage overhead during infer-
ence. For instance, the Wikitext-103 datastore requires nearly 500GB storage even for GPT2-small
model (He et al., 2021a). These limitations motivate our Memory Decoder, a compact parametric
model pretrained to mimic retrieval behavior while eliminating the need for large datastores.
3 Memory Decoder
In this section, we present Memory Decoder (MemDec), a plug-and-play pretrained memory designed
for efficient domain adaptation of large language models. Our method consists of two primary com-
ponents: a specialized pre-training procedure that aligns the output distribution of Memory Decoder
3

Figure 3: Overview of Memory Decoder architecture. Upper § 3.1: During pre-training, Memory
Decoder learns to align its output distributions with those generated by non-parametric retrievers
through distribution alignment loss. Lower § 3.2: During inference, Memory Decoder processes input
in parallel with the base LLM, and their distributions are interpolated to produce domain-enhanced
predictions without retrieval overhead.
with those of non-parametric retrievers (Section 3.1), and an efficient inference mechanism that
enables plug-and-play domain adaptation (Section 3.2). As illustrated in Figure 3, Memory Decoder
first learns to mimic non-parametric retrieval distributions during pre-training (upper part), then
seamlessly integrates with any compatible language model during inference (lower part), eliminating
the computational overhead associated with datastore maintenance and nearest neighbor search.
3.1 Pre-training
Our primary goal during pre-training is to enable Memory Decoder MMemto produce probability
distributions that closely resemble those generated by non-parametric retrievers when encountering
the same context. This approach effectively encodes the domain knowledge captured in large
key-value datastores into the parameters of our compact model.
Data Construction Since we require non-parametric distributions as supervision signals, we
construct training pairs of (xi, pkNN(·|xi))in advance to enable efficient pre-training. Here, xi
represents the input context and pkNN(·|xi)denotes the probability distribution generated by the non-
parametric retriever for that context. First, we build a key-value datastore (K, V) ={(ϕ(xi), yi)|
(xi, yi)∈ D train}using our domain-specific corpus, where ϕ(·)extracts hidden representations from
a specific layer of the pretrained model. For each context xiin the corpus, we then perform k-nearest
neighbor search against this datastore to identify similar contexts. To avoid trivial self-retrieval that
would contaminate the learning signal, we exclude the top-1 neighbor when its key exactly matches
the query key. Finally, we compute the non-parametric distribution pkNN(·|xi)for each context using
the retrieved neighbors and cache these context-distribution pairs for training.
Pre-training Objective Unlike traditional language modeling with single-label targets, kNN
distributions offer richer supervision signals by capturing the diversity of plausible continuations in
the domain (Xu et al., 2023)(see Appendix G for detailed analysis on kNN distributions). Through
extensive experimentation, we have identified that a hybrid objective yields optimal performance.
Our approach centers on a Distribution Alignment Loss that minimizes the KL divergence (Van Erven,
Harremos, 2014) between Memory Decoder’s output distribution and the cached kNN distributions
4

Figure 4: Inference latency comparison across domain adaptation methods. These measurements
were conducted on Qwen2.5-1.5B (Yang et al., 2024) for biomedicine domain text, augmented by a
0.5B Memory Decoder.
for each sample:
LKL(xi) =KL(pkNN(·|xi)∥pMem(·|xi)) (4)
To prevent excessive deviation from the underlying corpus distribution, we integrate a complementary
standard Language Modeling objective (Zhang, Sabuncu, 2018):
LLM(xi) =−logpMem(yi|xi) (5)
The final loss function balances these two objectives through a hyperparameter β:
L(xi) =β· LKL(xi) + (1 −β)· LLM(xi) (6)
Previous failed attempts to learn kNN distributions and our conjecture on why vanilla KL divergence
with cross-entropy regularization succeeds are detailed in Appendix H.
3.2 Inference
Once pretrained, Memory Decoder exhibits a key plug-and-play capability that allows it to adapt any
language model with a compatible tokenizer to the target domain via simple interpolation. During
inference, both the pretrained language model MPLMand Memory Decoder MMemprocess the same
input context in parallel, and their output distributions are interpolated:
pMem-PLM (yt|x) =α·pMem(yt|x) + (1 −α)·pPLM(yt|x) (7)
where α∈[0,1]controls the influence of domain-specific knowledge.
Unlike traditional retrieval-augmented approaches that introduce substantial latency from nearest
neighbor search and extended context processing, Memory Decoder requires only a single forward
pass through a relatively small transformer decoder. As demonstrated in Figure 4, our method
achieves significant improvements in inference efficiency compared to alternative domain adaptation
techniques. With just 1.28 ×overhead relative to the base model, Memory Decoder substantially
outperforms both In-Context RAG (Ram et al., 2023) (1.51 ×) and kNN-LM (Khandelwal et al.,
2019a) (2.17 ×), with the efficiency gap widening as the number of processed tokens increases.
Our method achieves even higher relative speedup (10.0 ×) when augmenting larger base models
with a 500-million-entry datastore, as the inter-process communication overhead between Memory
Decoder and LLM becomes amortized by longer inference times, unlike kNN search which scales
linearly with datastore size. This computational advantage, combined with Memory Decoder’s
model-agnostic design, makes our approach particularly valuable for production environments where
both performance and efficiency are critical considerations.
5

GPT2-small GPT2-med GPT2-large GPT2-xl
base 24.89 18.29 15.80 14.39
Non-parametric methods
+In-Context RAG 18.46 14.01 12.09 11.21
+kNN-LM 15.62 12.95 12.21 11.30
Parametric methods
+DAPT 15.74 12.78 11.10 10.16
+LoRA 18.63 13.88 11.77 10.67
+MemDec 13.36 12.25 11.53 10.93
Table 1: Perplexity comparison of domain adaptation methods across GPT2 model sizes on Wikitext-
103. The best performing results are highlighted in bold, while the second-best results are underlined .
Notably, applying our Memory Decoder( 124M ) to GPT2-medium( 345M ) outperforms DAPT of
GPT2-medium(345M), demonstrating the effectiveness of our approach in capturing domain knowl-
edge without modifying original parameters.
4 Experimental Setup
Overview We evaluate Memory Decoder across six complementary settings: (1) Language model-
ing on WikiText-103 (§5.1) to demonstrate effectiveness across GPT-2 model scales; (2) Downstream
tasks (§5.2) to verify preservation of general capabilities during domain adaptation; (3) Cross-model
adaptation (§5.3) showing a single Memory Decoder enhancing Qwen models from 0.5B to 72B
parameters; (4) Cross-vocabulary adaptation (§5.4) demonstrating efficient transfer between tokenizer
families; and (5) Knowledge-intensive QA (§5.5) proving Memory Decoder maintains reasoning
abilities while enhancing factual recall—a key limitation of retrieval methods. (6) Domain-specific
downstream tasks (Appendix D) confirming preservation of in-context learning capabilities across 13
real-world benchmarks. These experiments establish Memory Decoder as a versatile, plug-and-play
solution for efficient domain adaptation across diverse architectures and applications.
Datasets For language modeling experiments, we use Wikitext-103 (Merity et al., 2016), a standard
benchmark containing over 100M tokens from Wikipedia articles. For downstream evaluation,
following the kNN-prompt framework, we assess performance across nine NLP tasks: sentiment
analysis (SST2 (Socher et al., 2013), MR (Pang, Lee, 2005b), CR (Hu, Liu, 2004), RT (Pang, Lee,
2005a)), textual entailment (HYP (Kiesel et al., 2019), CB (De Marneffe et al., 2019), RTE (Dagan
et al., 2010)), and text classification (AGN (Zhang et al., 2015a), Yahoo (Zhang et al., 2015b)). For
domain-specific adaptation, we utilize three specialized corpora: (1) biomedical text from MIMIC-
III (Johnson et al., 2016) clinical notes covering over 46,000 patients, (2) financial news (Liu et al.,
2023a) from April 2024 to February 2025 for over 5,000 stocks, and (3) legal text from the Asylex
corpus (Barale et al., 2023) containing 59,112 documents of refugee status determination in Canada
from 1996 to 2022.
Baselines We compare Memory Decoder against several established domain adaptation methods:
In-Context RAG (Ram et al., 2023), which implements a BM25 retriever that processes 32 query
tokens, with retrieval occurring every 4 tokens. kNN-LM (Khandelwal et al., 2019a), configured with
interpolation parameter λ= 0.25and temperature settings of τ= 1for GPT-2 small and medium,
andτ= 13 for large and xl models. LoRA (Hu et al., 2022), applied to query, key, value and
MLP layers, with rank adjusted for each model to achieve parameter counts comparable to Memory
Decoder. Domain Adaptive Pretraining(DAPT) (Gururangan et al., 2020), which involves complete
retraining of all model parameters on the domain-specific corpus.
Training Details We conduct our experiments on an 8 ×A800 80GB GPU setup. For language
modeling and downstream evaluations, we use GPT2-xl (Radford et al., 2019) to build the key-value
datastore and non-parametric distributions for training, and continue training on a GPT2-small
model with learning rate 1e-3. For cross-model adaptation, we use Qwen2.5-1.5B (Yang et al.,
2024) to build the datastore, and continue training on Qwen2.5-0.5B with learning rate 1e-4. For
cross-vocabulary adaptation, we use Llama3.2-1B (Grattafiori et al., 2024) to build the datastore, and
continue training on the Memory Decoder trained from cross-model experiments, with its embedding
layer and language model head re-initialized. All experiments use a training budget equivalent to the
6

SST2 MR CR RT HYP CB RTE AGN Yahoo Avg
base 81.98 78.40 84.40 76.54 63.75 41.07 52.70 78.79 49.40 67.45
Non-parametric methods
+kNN-LM 81.98 77.95 83.80 77.95 64.14 39.28 52.70 77.73 49.63 67.24
Parametric methods
+DAPT 83.52 80.15 80.45 77.39 36.04 50.00 51.26 64.31 24.40 60.84
+LoRA 80.88 76.90 83.95 76.07 64.14 39.28 53.79 81.06 49.46 67.28
+MemDec 82.43 78.35 84.35 77.30 64.15 57.14 55.24 79.80 49.37 69.79
Table 2: Performance on nine diverse NLP tasks including sentiment analysis, textual entailment, and
text classification.
computational cost of training a 7B parameter model for 1 epoch, with DAPT and LoRA baselines
using the same maximum training FLOPS but early stopped to prevent overfitting. The training
hyperparameter βis set to 0.5across all tasks.
Evaluation Metrics For language modeling, cross-model, and cross-tokenizer experiments, we use
sliding window perplexity. Following Baevski, Auli (2018), in each test example, the context length
is set to 1024 where only the latter 512 tokens are scored. For downstream evaluation, following
methodology from Shi et al. (2022), we report results using the domain-conditional PMI scoring
rule (Holtzman et al., 2021). The interpolation hyperparameter αis tuned on the validation split of
each task following Khandelwal et al. (2019a), see more details in Appendix A.
5 Results
5.1 Language Modeling on Wikitext-103
Table 1 demonstrates the exceptional effectiveness of Memory Decoder across all GPT2 model
sizes. A single Memory Decoder with only 124M parameters consistently enhances the entire GPT2
family, showcasing its plug-and-play capability regardless of base model size. For smaller models,
our approach delivers superior results compared to all adaptation methods—notably outperforming
DAPT by 15.1% for GPT2-small and maintaining an advantage for GPT2-medium despite utilizing
only a fraction of the parameters. Even when applied to larger models where DAPT has inherent
advantages due to full model updates, Memory Decoder remains highly competitive while consistently
outperforming all other parameter-efficient methods without modifying any original parameters.
These results validate that a small parametric decoder can effectively capture the benefits of non-
parametric retrieval while eliminating computational overhead.
5.2 Downstream Performance
Table 2 reveals Memory Decoder’s ability to enhance domain adaptation while preserving general
language capabilities in zero-shot evaluation settings. Unlike DAPT, which suffers catastrophic
forgetting on several tasks (particularly HYP and Yahoo where performance drops by nearly half; see
Appendix B for detailed analysis), Memory Decoder maintains or improves performance across all
evaluated tasks. Our approach achieves the highest average score across all nine tasks, outperforming
the base model, kNN-LM, and LoRA while demonstrating particular strength on textual entailment
tasks like CB and RTE. These results validate a key advantage of our plug-and-play architecture:
by keeping the original model parameters intact while augmenting them with domain knowledge,
Memory Decoder achieves domain adaptation without sacrificing general capabilities. Importantly,
all experiments are conducted in a zero-shot setting, and our method should be viewed as orthogonal
to in-context learning approaches, which we analyze in Appendix F.
5.3 Cross-Model Adaptation
Table 3 demonstrates Memory Decoder’s exceptional plug-and-play capabilities across diverse
model sizes and architectures. A single Memory Decoder (0.5B parameters) consistently enhances
performance across all models in both the Qwen2 and Qwen2.5 families, spanning from 0.5B to
72B parameters. For smaller models like Qwen2-0.5B, our approach achieves dramatic perplexity
reductions—transforming domain-specific performance from near-baseline levels to state-of-the-art
7

Model Bio Fin Law Avg
Qwen2 Family
Qwen2-0.5B 18.41 16.00 10.23 14.88
+LoRA 7.28 9.70 5.82 7.60
+MemDec 3.75 3.84 4.57 4.05
Qwen2-1.5B 12.42 10.96 7.69 10.36
+LoRA 5.73 7.37 4.84 5.98
+MemDec 3.68 3.61 4.32 3.87
Qwen2-7B 8.36 8.31 5.92 7.53
+LoRA 4.47 5.64 4.02 4.71
+MemDec 3.59 3.38 4.00 3.66
Qwen2-72B 6.15 6.62 4.84 5.87
+MemDec 3.45 3.20 3.69 3.45
Qwen2.5 Family
Qwen2.5-0.5B 17.01 16.04 9.86 14.30
+LoRA 7.02 9.88 5.75 7.55
+MemDec 3.74 3.87 4.57 4.06
Qwen2.5-1.5B 11.33 11.20 7.42 9.98
+LoRA 5.59 7.50 4.82 5.97
+MemDec 3.67 3.61 4.29 3.86
Qwen2.5-3B 9.70 9.83 6.68 8.74
+LoRA 5.07 6.71 4.45 5.41
+MemDec 3.63 3.52 4.16 3.77
Qwen2.5-7B 8.19 8.61 5.94 7.58
+LoRA 4.03 5.31 3.81 4.38
+MemDec 3.57 3.42 4.01 3.67
Qwen2.5-14B 7.01 7.60 5.35 6.65
+MemDec 3.51 3.31 3.86 3.56
Qwen2.5-32B 6.65 7.38 5.18 6.40
+MemDec 3.48 3.29 3.81 3.53
Qwen2.5-72B 5.90 6.80 4.84 5.85
+MemDec 3.44 3.23 3.70 3.46
Table 3: Cross-model adaptation results across
three specialized domains. A single 0.5B Mem-
ory Decoder enhances models ranging from 0.5B
to 72B parameters.Model Bio Fin Law Avg
Llama3 Family
Llama3-8B 7.95 8.63 5.96 7.51
+LoRA 4.38 5.68 4.12 4.73
+MemDec 3.92 4.32 4.46 4.23
Llama3-70B 5.92 6.87 4.90 5.90
+MemDec 3.74 4.01 4.07 3.94
Llama3.1 Family
Llama3.1-8B 7.82 8.46 5.88 7.39
+LoRA 4.38 5.72 4.10 4.73
+MemDec 3.91 4.30 4.42 4.21
Llama3.1-70B 5.85 6.68 4.89 5.81
+MemDec 3.73 3.97 4.06 3.92
Llama3.2 Family
Llama3.2-1B 12.81 11.85 8.23 10.96
+LoRA 5.97 7.83 5.21 6.34
+MemDec 4.06 4.85 5.11 4.67
Llama3.2-3B 9.83 9.70 6.83 8.79
+LoRA 5.11 6.55 4.59 5.42
+MemDec 3.99 4.45 4.76 4.40
Table 4: Cross-vocabulary adaptation results
demonstrating efficient knowledge transfer be-
tween model families. Memory Decoder trained
on Qwen2.5 can be adapted to Llama models
with minimal additional training (10% of original
budget), achieving substantial perplexity reduc-
tions across all Llama variants and consistently
outperforming LoRA in biomedical and financial
domains.
results on both biomedical and financial text. Even for the largest models in the family, Memory
Decoder provides substantial improvements, demonstrating that retrieval-augmented knowledge
remains valuable regardless of model scale. Most impressively, a 0.5B model augmented with our
Memory Decoder surpasses vanilla 72B models across all evaluated domains, achieving over 140 ×
parameter efficiency. These results validate Memory Decoder’s core strength: a single pretrained
memory component can enhance multiple models sharing the same tokenizer, providing efficient
domain adaptation that scales from the smallest to the largest models while consistently outperforming
existing approaches.
5.4 Cross-Vocabulary Adaptation
Table 4 demonstrates Memory Decoder’s ability to generalize across different tokenizers and model
architectures. By re-initializing only the embedding layer and language model head of our Qwen2.5-
trained Memory Decoder, we successfully adapt it to the Llama model family with just 10% of the
original training budget. This efficient transfer enables substantial performance improvements across
all Llama variants. For Llama3-8B, Memory Decoder achieves roughly 50% perplexity reduction on
both biomedical and financial domains. Similar improvements extend to the Llama3.1 and Llama3.2
families, with our method consistently outperforming LoRA on biomedical and financial domains,
though showing room for improvement on legal text. These findings illustrate Memory Decoder’s
versatility beyond a single tokenizer family, demonstrating that domain knowledge learned from one
architecture can be efficiently transferred to another with minimal additional training. This capability
expands the practical utility of our approach, offering a streamlined path to domain adaptation across
diverse model ecosystems.
5.5 Knowledge-Intensive Reasoning Tasks
While retrieval-augmented methods excel at improving factual recall, they often struggle with tasks
requiring both knowledge retrieval and complex reasoning. Prior work (Geng et al., 2024) has shown
that kNN-LM can actually harm performance on knowledge-intensive QA tasks, despite retrieving
from relevant Wikipedia corpora.
To evaluate Memory Decoder’s ability to handle such challenging scenarios, following the experi-
mental setup of Geng et al. (2024), we trained a 1B parameter Memory Decoder on the same large
8

NQ HotpotQA
Llama3-8B 23.64 25.14
+ kNN-LM 24.00 (+0.36) 24.48 (-0.66)
+ MemDec (1B) 28.01 (+4.37) 27.72 (+2.58)
Table 5: Performance on knowledge-intensive QA tasks. Memory Decoder provides substantial
improvements while kNN-LM shows minimal gains or degradation.
Long-tail Knowledge Learning
Context (target token underlined ) MemDec kNN Base LM
he starred alongside actors Mark Strong and Derek
Jacobi68.94% 9.39% 0.12%
The launch of HMS Dreadnought in 1906 by the
Royal Navy raised the stakes98.65% 40.62% 1.57%
Semantic Coherence and Reasoning
Context (target token underlined ) MemDec kNN Base LM
In 2000 Boulter had a guest-starring role onthe tele-
vision series The Bill40.11% 8.07% 45.51%
...three tank squadrons for special overseas operations,
known as ’A’, ’B’ and ’ C’ Special Service Squadrons50.10% 10.76% 63.04%
Table 6: Probability assignments for specific tokens by different methods. Orange section :
Memory Decoder excels at capturing long-tail factual knowledge, assigning dramatically higher
probabilities than the base model. Cyan section : For semantic coherence, Memory Decoder
intelligently balances between kNN-LM and base model probabilities, preserving linguistic fluency.
heterogeneous corpus (see detailed corpus composition in Appendix C) and evaluated on Natural
Questions (NQ) and HotpotQA benchmarks.
As shown in Table 5, Memory Decoder achieves substantial improvements on both benchmarks, in
stark contrast to kNN-LM which shows marginal improvement on NQ and degradation on HotpotQA.
Our approach successfully enhances the model’s ability to access factual knowledge without compro-
mising its reasoning capabilities—addressing a fundamental limitation of traditional retrieval methods.
These results demonstrate that by learning to internalize retrieval patterns rather than relying on
explicit retrieval at inference time, Memory Decoder maintains the compositional reasoning abilities
necessary for complex multi-hop questions while still benefiting from expanded knowledge access.
6 Analysis
6.1 Case Study: Bridging Parametric and Non-Parametric Methods
Memory Decoder fundamentally learns to compress the knowledge stored in large non-parametric
datastores into a compact parametric model, combining the memorization capabilities of retrieval
methods with the efficiency and generalization of parametric approaches. To validate this hypothesis,
we conducted case studies on WikiText-103 examining how different methods assign probabilities to
specific tokens.
As shown in Table 6, Memory Decoder exhibits two crucial capabilities:
Long-tail Knowledge: For factual information like "Jacobi" and "1906", Memory Decoder assigns
dramatically higher probabilities than the base model (68.94% vs. 0.12% and 98.65% vs. 1.57%),
successfully capturing the memorization benefits of non-parametric methods while far exceeding
even kNN-LM’s retrieval capabilities.
Semantic Coherence: For function words and logical continuations like "on" and "C", Memory
Decoder maintains probabilities closer to the base model rather than following kNN-LM’s lower
probabilities, demonstrating its ability to preserve coherent language modeling capabilities that pure
retrieval methods sacrifice.
9

α 0.40 0.45 0.50 0.55 0.60 0.65 0.70 0.75 0.80
Avg PPL 4.21 4.17 4.14 4.119 4.110 4.112 4.126 4.155 4.201
Relative 1.024 1.015 1.007 1.002 1.000 1.001 1.004 1.011 1.022
Table 7: Sensitivity analysis of interpolation parameter αacross 11 Qwen models on the law domain.
Performance remains stable within 2.5% across the entire range (0.4–0.8), demonstrating robustness
to hyperparameter selection.
GPT2-S GPT2-M GPT2-L GPT2-XL Avg
Base 24.89 18.29 15.80 14.39 18.34
DAPT 15.47 12.78 11.10 10.16 12.38
+ MemDec-S (117M) 13.36 12.25 11.53 10.93 12.01
+ MemDec-M (345M) 12.08 11.59 10.92 10.43 11.26
+ MemDec-L (774M) 11.67 11.23 10.83 10.28 11.00
Table 8: Performance comparison with different Memory Decoder sizes. Even small Memory
Decoders achieve competitive performance with full-parameter DAPT while maintaining plug-and-
play capability.
These observations confirm that Memory Decoder occupies a unique position: it enhances memoriza-
tion of domain-specific and long-tail knowledge like non-parametric methods, while maintaining the
generalization and reasoning capabilities inherent to parametric models. This best-of-both-worlds
behavior explains Memory Decoder’s strong performance across diverse evaluation settings, from
domain adaptation to knowledge-intensive reasoning tasks.
6.2 Sensitivity Analysis of Interpolation Parameter
To assess the sensitivity of our approach to hyperparameter selection, we conducted comprehensive
experiments varying the interpolation parameter αacross all 11 Qwen models on the law domain.
As shown in Table 7, Memory Decoder demonstrates remarkable robustness: performance varies
by less than 2.5% across the entire αrange (0.4–0.8), with optimal performance at α=0.6. The
relative performance row shows that even at the extremes of the range, perplexity increases by only
2.2% and 2.4% respectively. This stability is crucial for practical deployment, as it indicates that
Memory Decoder does not require extensive hyperparameter tuning to achieve strong performance.
The shallow degradation around the optimum further suggests that practitioners can confidently
deploy Memory Decoder with minimal domain-specific calibration.
6.3 Impact of Memory Decoder Size
Table 8 examines how Memory Decoder size affects performance across the GPT-2 family. Re-
markably, even our smallest Memory Decoder (117M parameters) achieves competitive results with
full-parameter DAPT—for instance, GPT2-medium with a small Memory Decoder outperforms
GPT2-medium DAPT (12.25 vs. 12.78). As Memory Decoder size increases, performance consis-
tently improves across all base models, with the large variant achieving the best average perplexity.
These results validate that Memory Decoder provides an efficient alternative to full model fine-tuning:
practitioners can choose the decoder size based on their computational constraints while maintaining
the crucial advantage of preserving the original model’s capabilities.
6.4 Ablation on Pre-training Objective
Table 9 compares our hybrid training approach against single-objective variants on the biomedical
domain. Memory Decoder consistently outperforms both alternatives across all model configurations,
with substantial improvements over both KL-only (distribution alignment) and CE-only (standard
language modeling) training. Notably, the CE-only baseline corresponds to the DAPT setup, demon-
strating that simply interpolating predictions with a domain-adapted model cannot achieve the same
benefits as our integrated approach. More comprehensive DAPT model interpolation ablation can be
found in Appendix E. The gap between our method and KL-only training highlights the importance
of maintaining direct corpus supervision, while the improvement over CE-only validates the value
of incorporating non-parametric distribution knowledge. These results confirm that combining both
10

Biomed
Qwen2.5-3B Qwen2.5-7B Qwen2-7B
+KL Only 3.93 3.84 3.86
+CE Only 4.81 4.66 4.69
+MemDec 3.64 3.57 3.59
Table 9: Ablation study on Memory Decoder’s pretraining objective components.
objectives enables Memory Decoder to capture diverse continuation patterns from retrieval-based
distributions while preserving the structural patterns of the domain corpus.
7 Related Work
Retrieval-Augmented Generation Retrieval-Augmented Generation (RAG) enhances language
models by incorporating knowledge from external sources, with retrieval granularity ranging from
documents (Chen et al., 2017) to passages (Guu et al., 2020; Lewis et al., 2020b; Izacard et al.,
2023b) to tokens (Khandelwal et al., 2019b; He et al., 2021b; Min et al., 2022; Yogatama et al.,
2021). Token-level retrieval achieves superior performance for rare patterns and out-of-domain
scenarios but introduces substantial computation overhead during inference. While non-differentiable
retrieval mechanisms prevent end-to-end optimization and memory token approaches (Chevalier et al.,
2023) enable differentiable access but are limited to local contexts, Memory Decoder provides both
differentiable optimization and full-dataset knowledge access without expensive retrieval operations
or model-specific datastores.
Domain Adaptation Domain adaptation techniques have evolved from domain-specific pre-training
(SciBERT (Beltagy et al., 2019), BioBERT (Lee et al., 2020), ClinicalBERT (Huang et al., 2019))
to parameter-efficient methods like LoRA (Hu et al., 2022) and adapters (Wang et al., 2020; Diao
et al., 2021, 2023). However, these approaches require model-specific modifications, preventing
generalization across architectures. Memory Decoder addresses this limitation by providing a
domain-specific memory module that enhances multiple frozen language models without parameter
modifications, enabling cross-model adaptation within tokenizer families and efficient cross-tokenizer
transfer with minimal additional training.
8 Conclusion
In this paper, we introduced Memory Decoder, a novel plug-and-play approach for domain adaptation
of large language models. By pre-training a small transformer decoder to emulate the behavior of
non-parametric retrievers, Memory Decoder effectively adapts any compatible language model to a
specific domain without modifying its parameters. Our comprehensive experiments across multiple
model families and specialized domains demonstrate that Memory Decoder consistently outperforms
both parametric adaptation methods and traditional retrieval-augmented approaches.
The key innovation of Memory Decoder lies in its versatility and efficiency. A single pretrained
Memory Decoder can seamlessly enhance any model that shares the same tokenizer, and with
minimal additional training, can be adapted to models with different tokenizers and architectures.
This capability enables efficient domain adaptation across model families, dramatically reducing the
resources typically required for specialized model development. Our results confirm that Memory
Decoder preserves the performance benefits of retrieval-augmented methods while maintaining the
general capabilities of the base models, avoiding the catastrophic forgetting often observed with
parameter fine-tuning approaches.
Memory Decoder introduces a new paradigm for domain adaptation that fundamentally reimagines
how we specialize language models for particular domains. By decoupling domain expertise from
model architecture through a pretrained memory component, our approach creates a more modular,
efficient, and accessible framework for enhancing language model performance in specialized fields.
11

9 Limitations
While Memory Decoder demonstrates significant advantages for domain adaptation, we acknowledge
several limitations in our current approach. The pre-training phase requires searching in key-value
datastores to obtain kNN distributions as training signals, introducing computational overhead during
the training process. Although this cost is incurred only once per domain and is amortized across all
adapted models, it remains a bottleneck in the pipeline. Additionally, while cross-tokenizer adaptation
requires minimal training compared to training from scratch, it still necessitates some parameter
updates to align embedding spaces, preventing truly zero-shot cross-architecture transfer.
12

References
Baevski Alexei, Auli Michael . Adaptive input representations for neural language modeling // arXiv preprint
arXiv:1809.10853. 2018.
Barale Claire, Rovatsos Michael, Bhuta Nehal . Automated refugee case analysis: An nlp pipeline for supporting
legal practitioners // arXiv preprint arXiv:2305.15533. 2023.
Beltagy Iz, Lo Kyle, Cohan Arman . SciBERT: A pretrained language model for scientific text // arXiv preprint
arXiv:1903.10676. 2019.
Chen Danqi, Fisch Adam, Weston Jason, Bordes Antoine . Reading wikipedia to answer open-domain questions
// arXiv preprint arXiv:1704.00051. 2017.
Chen Yirong, Wang Zhenyu, Xing Xiaofen, Xu Zhipei, Fang Kai, Wang Junhong, Li Sihang, Wu Jieling, Liu
Qi, Xu Xiangmin, others . Bianque: Balancing the questioning and suggestion ability of health llms with
multi-turn health conversations polished by chatgpt // arXiv preprint arXiv:2310.15896. 2023.
Cheng Daixuan, Huang Shaohan, Wei Furu . Adapting large language models via reading comprehension // The
Twelfth International Conference on Learning Representations. 2023.
Chevalier Alexis, Wettig Alexander, Ajith Anirudh, Chen Danqi . Adapting language models to compress contexts
// arXiv preprint arXiv:2305.14788. 2023.
Colombo Pierre, Pires Telmo Pessoa, Boudiaf Malik, Culver Dominic, Melo Rui, Corro Caio, Martins Andre FT,
Esposito Fabrizio, Raposo Vera Lúcia, Morgado Sofia, others . Saullm-7b: A pioneering large language
model for law // arXiv preprint arXiv:2403.03883. 2024.
Dagan Ido, Dolan Bill, Magnini Bernardo, Roth Dan . Recognizing textual entailment: Rational, evaluation and
approaches–erratum // Natural Language Engineering. 2010. 16, 1. 105–105.
De Marneffe Marie-Catherine, Simons Mandy, Tonhauser Judith . The commitmentbank: Investigating projection
in naturally occurring discourse // proceedings of Sinn und Bedeutung. 23, 2. 2019. 107–124.
Diao Shizhe, Xu Ruijia, Su Hongjin, Jiang Yilei, Song Yan, Zhang Tong . Taming pre-trained language models
with n-gram representations for low-resource domain adaptation // Proceedings of the 59th Annual Meeting
of the Association for Computational Linguistics and the 11th International Joint Conference on Natural
Language Processing (V olume 1: Long Papers). 2021. 3336–3349.
Diao Shizhe, Xu Tianyang, Xu Ruijia, Wang Jiawei, Zhang Tong . Mixture-of-domain-adapters: Decoupling and
injecting domain knowledge to pre-trained language models memories // arXiv preprint arXiv:2306.05406.
2023.
Geng Shangyi, Zhao Wenting, Rush Alexander M . Great Memory, Shallow Reasoning: Limits of kNN-LMs //
arXiv preprint arXiv:2408.11815. 2024.
Grattafiori Aaron, Dubey Abhimanyu, Jauhri Abhinav, Pandey Abhinav, Kadian Abhishek, Al-Dahle Ahmad,
Letman Aiesha, Mathur Akhil, Schelten Alan, Vaughan Alex, others . The llama 3 herd of models // arXiv
preprint arXiv:2407.21783. 2024.
Guo Daya, Yang Dejian, Zhang Haowei, Song Junxiao, Zhang Ruoyu, Xu Runxin, Zhu Qihao, Ma Shirong, Wang
Peiyi, Bi Xiao, others . Deepseek-r1: Incentivizing reasoning capability in llms via reinforcement learning //
arXiv preprint arXiv:2501.12948. 2025.
Gururangan Suchin, Marasovi´ c Ana, Swayamdipta Swabha, Lo Kyle, Beltagy Iz, Downey Doug, Smith Noah A .
Don’t stop pretraining: Adapt language models to domains and tasks // arXiv preprint arXiv:2004.10964.
2020.
Guu Kelvin, Lee Kenton, Tung Zora, Pasupat Panupong, Chang Mingwei . Retrieval augmented language model
pre-training // International conference on machine learning. 2020. 3929–3938.
He Junxian, Neubig Graham, Berg-Kirkpatrick Taylor . Efficient nearest neighbor language models // arXiv
preprint arXiv:2109.04212. 2021a.
He Junxian, Neubig Graham, Berg-Kirkpatrick Taylor . Efficient nearest neighbor language models // arXiv
preprint arXiv:2109.04212. 2021b.
Holtzman Ari, West Peter, Shwartz Vered, Choi Yejin, Zettlemoyer Luke . Surface form competition: Why the
highest probability answer isn’t always right // arXiv preprint arXiv:2104.08315. 2021.
13

Hu Edward J, Shen Yelong, Wallis Phillip, Allen-Zhu Zeyuan, Li Yuanzhi, Wang Shean, Wang Lu, Chen Weizhu,
others . Lora: Low-rank adaptation of large language models. // ICLR. 2022. 1, 2. 3.
Hu Minqing, Liu Bing . Mining and summarizing customer reviews // Proceedings of the Tenth ACM SIGKDD
International Conference on Knowledge Discovery and Data Mining. New York, NY , USA: Association for
Computing Machinery, 2004. 168–177. (KDD ’04).
Huang Kexin, Altosaar Jaan, Ranganath Rajesh . Clinicalbert: Modeling clinical notes and predicting hospital
readmission // arXiv preprint arXiv:1904.05342. 2019.
Izacard Gautier, Lewis Patrick, Lomeli Maria, Hosseini Lucas, Petroni Fabio, Schick Timo, Dwivedi-Yu Jane,
Joulin Armand, Riedel Sebastian, Grave Edouard . Atlas: Few-shot learning with retrieval augmented language
models // Journal of Machine Learning Research. 2023a. 24, 251. 1–43.
Izacard Gautier, Lewis Patrick, Lomeli Maria, Hosseini Lucas, Petroni Fabio, Schick Timo, Dwivedi-Yu Jane,
Joulin Armand, Riedel Sebastian, Grave Edouard . Atlas: Few-shot learning with retrieval augmented language
models // Journal of Machine Learning Research. 2023b. 24, 251. 1–43.
Johnson Alistair EW, Pollard Tom J, Shen Lu, Lehman Li-wei H, Feng Mengling, Ghassemi Mohammad, Moody
Benjamin, Szolovits Peter, Anthony Celi Leo, Mark Roger G . MIMIC-III, a freely accessible critical care
database // Scientific data. 2016. 3, 1. 1–9.
Khandelwal Urvashi, Levy Omer, Jurafsky Dan, Zettlemoyer Luke, Lewis Mike . Generalization through
memorization: Nearest neighbor language models // arXiv preprint arXiv:1911.00172. 2019a.
Khandelwal Urvashi, Levy Omer, Jurafsky Dan, Zettlemoyer Luke, Lewis Mike . Generalization through
memorization: Nearest neighbor language models // arXiv preprint arXiv:1911.00172. 2019b.
Kiesel Johannes, Mestre Maria, Shukla Rishabh, Vincent Emmanuel, Adineh Payam, Corney David, Stein Benno,
Potthast Martin . SemEval-2019 Task 4: Hyperpartisan News Detection // Proceedings of the 13th International
Workshop on Semantic Evaluation. Minneapolis, Minnesota, USA: Association for Computational Linguistics,
VI 2019. 829–839.
Kirkpatrick James, Pascanu Razvan, Rabinowitz Neil, Veness Joel, Desjardins Guillaume, Rusu Andrei A, Milan
Kieran, Quan John, Ramalho Tiago, Grabska-Barwinska Agnieszka, others . Overcoming catastrophic
forgetting in neural networks // Proceedings of the national academy of sciences. 2017. 114, 13. 3521–3526.
Lee Jinhyuk, Yoon Wonjin, Kim Sungdong, Kim Donghyeon, Kim Sunkyu, So Chan Ho, Kang Jaewoo . BioBERT:
a pre-trained biomedical language representation model for biomedical text mining // Bioinformatics. 2020.
36, 4. 1234–1240.
Lewis Patrick, Perez Ethan, Piktus Aleksandra, Petroni Fabio, Karpukhin Vladimir, Goyal Naman, Küttler
Heinrich, Lewis Mike, Yih Wen-tau, Rocktäschel Tim, others . Retrieval-augmented generation for knowledge-
intensive nlp tasks // Advances in neural information processing systems. 2020a. 33. 9459–9474.
Lewis Patrick, Perez Ethan, Piktus Aleksandra, Petroni Fabio, Karpukhin Vladimir, Goyal Naman, Küttler
Heinrich, Lewis Mike, Yih Wen-tau, Rocktäschel Tim, others . Retrieval-augmented generation for knowledge-
intensive nlp tasks // Advances in neural information processing systems. 2020b. 33. 9459–9474.
Liu Aixin, Feng Bei, Xue Bing, Wang Bingxuan, Wu Bochao, Lu Chengda, Zhao Chenggang, Deng Chengqi,
Zhang Chenyu, Ruan Chong, others . Deepseek-v3 technical report // arXiv preprint arXiv:2412.19437. 2024.
Liu Xiao-Yang, Wang Guoxuan, Yang Hongyang, Zha Daochen . Data-centric FinGPT: Democratizing Internet-
scale Data for Financial Large Language Models // NeurIPS Workshop on Instruction Tuning and Instruction
Following. 2023a.
Liu Xiao-Yang, Wang Guoxuan, Yang Hongyang, Zha Daochen . Fingpt: Democratizing internet-scale data for
financial large language models // arXiv preprint arXiv:2307.10485. 2023b.
Merity Stephen, Xiong Caiming, Bradbury James, Socher Richard . Pointer Sentinel Mixture Models. 2016.
Min Sewon, Shi Weijia, Lewis Mike, Chen Xilun, Yih Wen-tau, Hajishirzi Hannaneh, Zettlemoyer Luke . Nonpara-
metric masked language modeling // arXiv preprint arXiv:2212.01349. 2022.
Pang Bo, Lee Lillian . Seeing stars: Exploiting class relationships for sentiment categorization with respect to
rating scales // Proceedings of the ACL. 2005a.
Pang Bo, Lee Lillian . Seeing stars: exploiting class relationships for sentiment categorization with respect to
rating scales // Proceedings of the 43rd Annual Meeting on Association for Computational Linguistics. USA:
Association for Computational Linguistics, 2005b. 115–124. (ACL ’05).
14

Radford Alec, Wu Jeffrey, Child Rewon, Luan David, Amodei Dario, Sutskever Ilya, others . Language models
are unsupervised multitask learners // OpenAI blog. 2019. 1, 8. 9.
Ram Ori, Levine Yoav, Dalmedigos Itay, Muhlgay Dor, Shashua Amnon, Leyton-Brown Kevin, Shoham Yoav . In-
context retrieval-augmented language models // Transactions of the Association for Computational Linguistics.
2023. 11. 1316–1331.
Shi Weijia, Michael Julian, Gururangan Suchin, Zettlemoyer Luke . Nearest neighbor zero-shot inference //
Proceedings of the 2022 Conference on Empirical Methods in Natural Language Processing. 2022. 3254–3265.
Socher Richard, Perelygin Alex, Wu Jean, Chuang Jason, Manning Christopher D., Ng Andrew, Potts Christopher .
Recursive Deep Models for Semantic Compositionality Over a Sentiment Treebank // Proceedings of the 2013
Conference on Empirical Methods in Natural Language Processing. Seattle, Washington, USA: Association
for Computational Linguistics, X 2013. 1631–1642.
Van Erven Tim, Harremos Peter . Rényi divergence and Kullback-Leibler divergence // IEEE Transactions on
Information Theory. 2014. 60, 7. 3797–3820.
Ven Gido M van de, Soures Nicholas, Kudithipudi Dhireesha . Continual learning and catastrophic forgetting //
arXiv preprint arXiv:2403.05175. 2024.
Wang Ruize, Tang Duyu, Duan Nan, Wei Zhongyu, Huang Xuanjing, Cao Guihong, Jiang Daxin, Zhou Ming, oth-
ers. K-adapter: Infusing knowledge into pre-trained models with adapters // arXiv preprint arXiv:2002.01808.
2020.
Xu Frank F , Alon Uri, Neubig Graham . Why do nearest neighbor language models work? // International
Conference on Machine Learning. 2023. 38325–38341.
Yang An, Yang Baosong, Zhang Beichen, Hui Binyuan, Zheng Bo, Yu Bowen, Li Chengyuan, Liu Dayiheng,
Huang Fei, Wei Haoran, others . Qwen2. 5 technical report // arXiv preprint arXiv:2412.15115. 2024.
Yogatama Dani, Masson d’Autume Cyprien de, Kong Lingpeng . Adaptive semiparametric language models //
Transactions of the Association for Computational Linguistics. 2021. 9. 362–373.
Zhang Xiang, Zhao Junbo, LeCun Yann . Character-level convolutional networks for text classification //
Advances in neural information processing systems. 2015a. 28.
Zhang Xiang, Zhao Junbo, LeCun Yann . Character-level convolutional networks for text classification //
Advances in neural information processing systems. 2015b. 28.
Zhang Zhilu, Sabuncu Mert . Generalized cross entropy loss for training deep neural networks with noisy labels
// Advances in neural information processing systems. 2018. 31.
15

A Interpolation hyperparameter αof all tasks
A.1 Language Modeling on Wikitext-103
For language modeling on WikiText-103 (section 5.1), we use the following αvalues for different
GPT-2 model sizes:
Model α
GPT-2-small 0.80
GPT-2-medium 0.60
GPT-2-large 0.55
GPT-2-xl 0.55
Table 10: Interpolation hyperparameter αfor GPT-2 models on WikiText-103.
The trend of smaller αfor larger GPT models aligns with intuition—stronger base models require less
augmentation from the memory component. The general pattern centers around α=0.6, confirming it
as a robust default choice.
A.2 Downstream Performance
Table 11 presents the optimal αvalues for downstream tasks in section 5.2.
Task α
SST-2 0.30
MR 0.30
CR 0.05
RT 0.20
HYP 0.20
CB 0.30
RTE 0.60
AGN 0.20
Yahoo 0.20
Table 11: Optimal interpolation hyperparameter αfor downstream tasks.
The general pattern centers around α=0.3, which is consistent with the findings in Shi et al. (2022).
A.3 Cross-Model and Cross-Vocabulary Adaptation
For domain-specific language modeling tasks (section 5.3 and 5.4), we tune αon the validation set by
searching over {0.4,0.6,0.8,0.9}. We find that α= 0.6yields the best performance in most cases.
B Analysis of DAPT Performance on Downstream Tasks
Previous work has shown that domain-adaptive pretraining can adversely affect a model’s prompting
ability (Cheng et al., 2023). Our experiments reveal that this effect is particularly pronounced when
using domain-conditional PMI (DCPMI) scoring for evaluation, especially on tasks where label
verbalizers overlap with the pretraining domain vocabulary.
As shown in Table 12, while direct language modeling evaluation reveals only modest performance
drops with DAPT, the DCPMI scores show dramatic degradation for smaller models. This discrepancy
arises because we employ fuzzy verbalizers following Shi et al. (2022), and the label spaces for Yahoo
and AGN tasks contain terms (e.g., “politics,” “technology”) that appear frequently in WikiText-103.
When DAPT increases the domain probability for these terms, it causes the conditional PMI scores to
drop substantially, as the denominator in the DCPMI calculation becomes inflated.
The results for GPT-2-xl demonstrate that larger models exhibit greater robustness to this evaluation
artifact, maintaining relatively stable DCPMI scores after domain adaptation. This suggests that
16

Model Yahoo (LM) Yahoo (DCPMI) HYP (LM) HYP (DCPMI) Avg
GPT-2-small 0.466 0.495 0.639 0.638 0.559
+DAPT 0.429 0.244 0.608 0.361 0.410
∆ -0.037 -0.251 -0.031 -0.277 -0.149
GPT-2-xl 0.520 0.499 0.628 0.609 0.564
+DAPT 0.490 0.491 0.624 0.618 0.556
∆ -0.030 -0.008 -0.004 +0.009 -0.008
Table 12: Comparison of standard language modeling (LM) scores versus domain-conditional PMI
(DCPMI) scores for DAPT models on Yahoo and HYP tasks.
the apparent failure of DAPT on certain downstream tasks is partly an artifact of the evaluation
methodology rather than a fundamental limitation of the approach, though the phenomenon highlights
an important interaction between domain adaptation and prompt-based evaluation methods.
C Knowledge-Intensive Reasoning Task Corpus Composition
For the knowledge-intensive reasoning experiments in Section 5.5, we constructed a large hetero-
geneous corpus following Geng et al. (2024). The corpus combines diverse text sources to provide
broad coverage of factual knowledge and various writing styles. The complete corpus is publicly
available at https://huggingface.co/datasets/wentingzhao/knn-prompt-datastore .
Corpus Size
WikiText-103 181MB
Amazon Reviews 89MB
CC-NEWS 457MB
IMDB 45MB
Total 722MB
Table 13: Composition of the heterogeneous corpus used for training Memory Decoder on knowledge-
intensive tasks.
D Domain-Specific Downstream Tasks
To comprehensively evaluate Memory Decoder’s ability to maintain in-context learning capabilities
while achieving domain adaptation, we conducted extensive experiments on 13 real-world domain-
specific tasks across biomedical, financial, and legal domains, following the same domain corpus and
evaluation framework from Cheng et al. (2023). These benchmarks test both zero-shot and few-shot
performance, providing a rigorous assessment of whether domain adaptation preserves the model’s
prompting abilities—a known weakness of traditional DAPT approaches (Cheng et al., 2023).
D.1 Biomedical Domain Tasks
ChemProt MQP PubmedQA RCT USMLE Avg
(13-shot) (4-shot) (0-shot) (10-shot) (0-shot)
Qwen2.5-7B 24.40 83.44 63.70 70.10 36.92 55.71
+ DAPT 17.59 76.22 65.70 21.00 34.64 43.03
+ MemDec (0.5B) 24.40 84.09 64.40 74.06 36.84 56.76
Table 14: Performance on biomedical domain-specific tasks. Memory Decoder preserves in-context
learning capabilities while DAPT shows severe degradation, particularly on RCT (70.10 →21.00).
17

D.2 Financial Domain Tasks
FiQA_SA FPB Headline NER ConvFinQA Avg
(5-shot) (5-shot) (5-shot) (20-shot) (0-shot)
Qwen2.5-7B 80.46 70.96 87.08 68.92 60.53 73.59
+ DAPT 75.59 66.39 86.03 69.32 58.52 71.17
+ MemDec (0.5B) 81.34 71.25 87.95 69.21 63.69 74.68
Table 15: Performance on financial domain-specific tasks. Memory Decoder achieves consistent
improvements across all tasks while DAPT degrades performance on most benchmarks.
D.3 Legal Domain Tasks
SCOTUS (micro) SCOTUS (macro) CaseHOLD (micro) CaseHOLD (macro) UNFAIR-ToS Avg
(0-shot) (0-shot) (0-shot) (0-shot) (4-shot)
Qwen2.5-7B 26.66 17.90 35.92 35.93 87.05 40.69
+ DAPT 28.33 16.82 35.70 35.69 87.18 40.74
+ MemDec (0.5B) 31.66 21.05 37.58 37.59 87.05 42.99
Table 16: Performance on legal domain-specific tasks. Memory Decoder shows substantial improve-
ments on case-based reasoning tasks while maintaining strong performance on contract analysis.
D.4 Analysis
Across all three domains, Memory Decoder demonstrates a crucial advantage: it successfully enhances
domain-specific performance while preserving the model’s in-context learning capabilities. DAPT,
in contrast, shows significant degradation on many tasks, particularly those requiring few-shot or
zero-shot reasoning. The most dramatic example occurs in the biomedical RCT task, where DAPT’s
performance collapses from 70.10 to 21.00, while Memory Decoder improves it to 74.06.
This preservation of prompting abilities is essential for practical deployment, as real-world applica-
tions often require models to handle both domain-specific knowledge and general reasoning tasks.
Memory Decoder achieves this balance by augmenting rather than modifying the base model, ensur-
ing that the original capabilities remain intact while adding specialized domain knowledge through
the learned retrieval patterns.
E Comparison with DAPT Model Interpolation
A natural baseline for domain adaptation is to train a small model with domain-adaptive pretraining
(DAPT) and interpolate its predictions with the base LLM. This approach, while conceptually similar
to Memory Decoder’s inference-time interpolation, relies on standard language modeling objectives
rather than our novel distribution alignment training. To thoroughly evaluate this alternative, we
conducted extensive comparisons between Memory Decoder and logit ensembling with DAPT
models.
Base Model Baseline PPL + DAPT (Small) + MemDec (Small)
GPT2-small 24.89 15.95 13.36 (+2.59)
GPT2-medium 18.29 14.26 12.25 (+2.01)
GPT2-large 15.80 13.13 11.53 (+1.60)
GPT2-xl 14.39 12.30 10.93 (+1.37)
Average 18.34 13.91 12.01 (+1.90)
Table 17: Comparison between Memory Decoder and DAPT model interpolation on WikiText-103.
Both approaches use the same small model architecture (124M parameters) but differ in training
objectives.
As shown in Table 17, Memory Decoder consistently outperforms the DAPT model interpolation
baseline across all GPT-2 variants, achieving an average improvement of 1.90 perplexity points.
18

This gap is particularly pronounced for smaller base models, where Memory Decoder provides
2.59 perplexity improvement over DAPT interpolation for GPT2-small. These results validate that
our hybrid training objective—combining distribution alignment with language modeling—enables
superior memorization and retrieval of domain knowledge compared to standard DAPT approaches.
The performance difference stems from Memory Decoder’s ability to learn from non-parametric
distribution patterns during training. While a DAPT model only learns to maximize likelihood on the
domain corpus, Memory Decoder additionally learns to approximate the retrieval-based distributions
from the datastore, capturing richer continuation patterns that better represent the domain knowledge.
This finding aligns with our ablation results in Section 6.4, where the CE-only baseline (equivalent to
DAPT) significantly underperforms our full method.
Furthermore, the consistent improvement across different model scales demonstrates that this advan-
tage is not merely an artifact of model size or specific architectural choices, but rather a fundamental
benefit of our training methodology. Even when applied to the largest GPT2-xl model, where DAPT
already provides substantial gains, Memory Decoder achieves additional improvements of 1.37 per-
plexity points, confirming the complementary nature of distribution alignment and standard language
modeling objectives.
F In-Context Learning Performance Analysis
Memory Decoder is designed as a complementary approach to existing adaptation methods, including
in-context learning (ICL). While our primary comparisons focus on domain-adaptive pretraining due
to our emphasis on learning from unlabeled corpora, it is crucial to verify that Memory Decoder
preserves and potentially enhances the model’s ability to leverage in-context examples—a capability
often compromised by traditional domain adaptation methods.
To thoroughly evaluate this aspect, we conducted experiments on the CaseHOLD legal reasoning
benchmark with varying numbers of in-context examples.
0-shot 4-shot 8-shot 16-shot Best
CaseHOLD (micro)
Qwen2.5-7B 35.92 36.34 36.34 36.09 36.34
+ MemDec (0.5B) 37.58 (+1.66) 37.95 (+1.61) 38.29 (+1.95) 37.53 (+1.44) 38.29
CaseHOLD (macro)
Qwen2.5-7B 35.93 36.35 36.35 36.10 36.35
+ MemDec (0.5B) 37.59 (+1.66) 37.95 (+1.60) 38.30 (+1.95) 37.53 (+1.43) 38.30
Table 18: In-context learning performance on CaseHOLD legal reasoning task across different shot
settings. Memory Decoder consistently improves performance, with peak performance at 8-shot.
Table 18 reveals several critical insights about Memory Decoder’s interaction with in-context learning:
Zero-shot superiority: Memory Decoder’s zero-shot performance (37.58/37.59) already surpasses
the base model’s best few-shot performance (36.34/36.35), demonstrating that our method effectively
encodes domain knowledge that would otherwise require multiple demonstration examples to achieve.
Complementary benefits: Memory Decoder continues to benefit from in-context examples, with
performance improving from 0-shot (37.58) to 8-shot (38.29) settings. This improvement pattern
confirms that our method preserves the model’s ability to learn from demonstrations while adding
orthogonal domain knowledge.
Consistent advantages: Across all tested shot settings (0, 4, 8, 16), Memory Decoder maintains a
stable improvement margin over the base model, ranging from 1.43 to 1.95 points. This consistency
indicates that the domain knowledge encoded by Memory Decoder complements rather than interferes
with in-context learning mechanisms.
Optimal shot selection: Both methods achieve peak performance at 8 shots, with slight degradation
at 16 shots—likely due to noise from increased context length. Importantly, Memory Decoder’s
pattern mirrors the base model’s, suggesting that our method does not introduce additional brittleness
to long-context scenarios.
19

These results definitively establish that Memory Decoder not only provides strong zero-shot domain
adaptation but also preserves and enhances the model’s ability to leverage in-context examples—a
critical capability for practical deployment. This stands in stark contrast to DAPT, which, as shown
in our domain-specific evaluations (Appendix D), often compromises in-context learning abilities
while pursuing domain adaptation. The preservation of ICL capabilities makes Memory Decoder
particularly valuable for real-world applications where models must handle both domain-specific
tasks and general reasoning with varying amounts of available demonstrations.
G Characteristics of k-NN Distributions
G.1 Extreme Sparsity and Concentration
k-NN distributions exhibit fundamentally different characteristics from standard language model out-
puts. While LM distributions maintain smooth probability mass across vocabulary with extensive long
tails,k-NN distributions demonstrate extreme sparsity—typically assigning non-zero probabilities to
only 2–3 tokens from a 50,257-dimensional vocabulary.
Figure 5: Probability distributions from k-NN retrieval, standard LM, and Memory Decoder for
GPT-2-Large. The k-NN distribution shows extreme sparsity with concentrated probability mass.
This concentration emerges from two factors: (1) the hard constraint of selecting only knearest
neighbors eliminates low-probability candidates, and (2) high-dimensional embedding spaces (e.g.,
1280 dimensions for GPT-2-Large) amplify distance relationships through the curse of dimensionality,
causing nearest neighbors to dominate disproportionately.
G.2 Scale-Dependent Behavior
Model scale dramatically affects k-NN distribution quality. GPT-2-Small (117M) produces dis-
tributions marginally different from its LM outputs (top-1 probability 50%), while GPT-2-Large
(1.5B) generates radically sparse distributions with 93.48% average top-1 probability—a 67% relative
increase over its baseline.
Larger models benefit from: (1) higher-dimensional spaces where distance concentration intensifies,
and (2) superior contextual representations that better disambiguate polysemous tokens and preserve
semantic distinctions, leading to more coherent nearest neighbor retrievals.
20

Figure 6: k-NN distribution sparsity across model scales. Despite identical retrieval parameters
(k= 1024 ), larger models produce substantially sparser distributions.
Base Model Baseline PPL with MemDec from:
PPL Small Medium Large
GPT-2-Small 24.89 14.01 13.80 13.77
GPT-2-Medium 18.29 12.88 12.74 12.70
GPT-2-Large 15.80 12.05 11.95 11.93
Table 19: Perplexity with Memory Decoder (125M) trained using k-NN distributions from different
source models
G.3 Domain Adaptation Effects
Fine-tuned models produce sharper k-NN distributions than base models. Domain adaptation
creates specialized embedding clusters with reduced intra-cluster variance and increased inter-cluster
separation, leading to more decisive retrievals. Memory Decoders trained with fine-tuned distributions
consistently achieve lower perplexity, validating that domain-adapted representations provide superior
retrieval targets.
H Alternative Loss Functions for Imitating k-NN Distributions
H.1 Failed Approaches
While KL divergence (combined with cross-entropy regularization) successfully matches k-NN
distributions, we systematically evaluated several alternative loss functions that all demonstrated
inferior performance:
H.1.1 Focal Loss
Adapted from object detection to handle class imbalance through gradient rescaling:
LFocal=−X
i[α(1−pθ(i))γpkNN(i) logpθ(i) + (1 −α)pθ(i)γ(1−pkNN(i)) log(1 −pθ(i))]
(8)
With α= 0.5andγ= 2, focal loss theoretically emphasizes hard-to-classify sparse regions but
failed to achieve sufficient distribution concentration in practice.
21

H.1.2 Jensen-Shannon Divergence
A symmetric alternative to KL divergence:
JSD(P∥Q) =1
2DKL(P∥M) +1
2DKL(Q∥M), M =1
2(P+Q) (9)
Despite avoiding the directional bias of KL divergence, JSD provided no advantage for our extremely
sparse target distributions.
H.1.3 Bi-directional Logits Difference (BiLD)
BiLD focuses on relative rankings by computing pairwise differences among top- klogits:
LBiLD=DKL[pkNN
led∥pθ
cor] +DKL[pkNN
cor∥pθ
led] (10)
While theoretically suited for distributions where relative ordering matters more than exact probabili-
ties, BiLD consistently underperformed standard KL divergence.
H.1.4 Explicit Sparsity Penalty
Direct penalization of non-zero predictions in zero-probability regions:
Lsparse=LKL+αX
iI{pkNN(i)=0}·pθ(i), α = 0.01 (11)
This approach created training instability without meaningfully improving output sparsity.
H.2 Why KL Divergence Succeeds
The superior performance of KL divergence (with cross-entropy regularization) for matching k-NN
distributions likely stems from its unique mathematical properties that align with the retrieval-based
nature of the target:
Asymmetric penalty structure : KL divergence DKL(P||Q) =P
iP(i) logP(i)
Q(i)heavily penalizes
placing probability mass where the target has none (when P(i)≈0butQ(i)>0), while being
more forgiving of missing mass where the target has some. This asymmetry naturally encourages
sparsity—the model learns to aggressively zero out predictions outside the k-NN support.
Mode-seeking behavior : The forward KL divergence DKL(P||Q)is inherently mode-seeking,
preferring to capture a few high-probability modes rather than covering the entire distribution. For
k-NN distributions with 2-3 dominant modes, this bias perfectly matches the desired behavior, unlike
symmetric losses (JSD) or mode-covering alternatives.
Information-theoretic optimality : KL divergence directly minimizes the expected encoding length
difference between distributions. For k-NN distributions that encode "retrieval-aware uncertainty,"
KL naturally preserves the information structure—maintaining both the sharp peaks (high retrieval
confidence) and the specific ranking among top candidates that emerges from the datastore’s empirical
distribution.
The cross-entropy regularization component anchors the model to linguistically valid outputs, pre-
venting collapse to degenerate solutions while the KL term drives sparsity. This combination uniquely
balances the competing demands of extreme concentration and semantic coherence.
22