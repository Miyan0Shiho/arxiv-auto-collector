# Grounded in Context: Retrieval-Based Method for Hallucination Detection

**Authors**: Assaf Gerner, Netta Madvil, Nadav Barak, Alex Zaikman, Jonatan Liberman, Liron Hamra, Rotem Brazilay, Shay Tsadok, Yaron Friedman, Neal Harow, Noam Bresler, Shir Chorev, Philip Tannor

**Published**: 2025-04-22 10:28:23

**PDF URL**: [http://arxiv.org/pdf/2504.15771v1](http://arxiv.org/pdf/2504.15771v1)

## Abstract
Despite advancements in grounded content generation, production Large
Language Models (LLMs) based applications still suffer from hallucinated
answers. We present "Grounded in Context" - Deepchecks' hallucination detection
framework, designed for production-scale long-context data and tailored to
diverse use cases, including summarization, data extraction, and RAG. Inspired
by RAG architecture, our method integrates retrieval and Natural Language
Inference (NLI) models to predict factual consistency between premises and
hypotheses using an encoder-based model with only a 512-token context window.
Our framework identifies unsupported claims with an F1 score of 0.83 in
RAGTruth's response-level classification task, matching methods that trained on
the dataset, and outperforming all comparable frameworks using similar-sized
models.

## Full Text


<!-- PDF content starts -->

Grounded in Context: Retrieval-Based Method
for Hallucination Detection
Assaf Gerner1, Netta Madvil1, Nadav Barak1, Alex Zaikman1, Jonatan
Liberman1, Liron Hamra1, Rotem Brazilay1, Shay Tsadok1, Yaron Friedman1,
Neal Harow1, Noam Bresler1, Shir Chorev1, and Philip Tannor1
Deepchecks, Ramat Gan, Israel
Abstract. Despite advancements in grounded content generation, pro-
duction Large Language Models (LLMs) based applications still suf-
fer from hallucinated answers. We present ”Grounded in Context” -
Deepchecks’ hallucination detection framework, designed for production-
scale long-context data and tailored to diverse use cases, including sum-
marization, data extraction, and RAG. Inspired by RAG architecture,
our method integrates retrieval and Natural Language Inference (NLI)
models to predict factual consistency between premises and hypotheses
using an encoder-based model with only a 512-token context window.
Our framework identifies unsupported claims with an F1 score of 0.83
in RAGTruth’s response-level classification task, matching methods that
trained on the dataset, and outperforming all comparable frameworks
using similar-sized models.
1 Introduction
In natural language generation tasks such as Retrieval-Augmented Generation
(RAG) and abstractive summarization, hallucinations—instances where gener-
ated text contains contradictory or fabricated information in comparison to a
reference text—persist as a significant challenge in practical applications, de-
spite considerable advancements in grounded content generation [1]. In RAG
systems, these hallucinations emerge from inconsistencies between retrieved data
and generated content, substantially undermining output reliability. Similarly,
in abstractive summarization, models may produce information not directly in-
ferable from source text, resulting in summaries that introduce spurious details
or contradict the original document.
At Deepchecks, our primary objective is to evaluate Large Language Model
(LLM) based applications with particular emphasis on detecting and mitigating
such hallucinations. In practical scenarios, users retrieve data fragments from
diverse sources that are frequently unstructured or ”noisy”. Furthermore, con-
temporary LLMs can process extensive contexts, necessitating evaluation frame-
works capable of handling substantial data volumes both efficiently and precisely.
To address these challenges, we developed Grounded in Context, a RAG-inspired
methodology for hallucination detection. Our approach decomposes output intoarXiv:2504.15771v1  [cs.LG]  22 Apr 2025

factual statements and retrieves a dedicated context for each statement. Statement-
context pairs are evaluated for factual consistency, with individual scores ag-
gregated into a comprehensive metric. Our methodology demonstrates superior
performance compared to models of comparable size when evaluated on the
RAGTruth dataset [2], despite the exclusion of its training corpus from our
method’s training data.
2 Related Work
Research in hallucination detection has grown significantly in recent years, mainly
driven by the increasing deployment of RAG systems. Detection methods gen-
erally focus on identifying factual inaccuracies by comparing generated content
with retrieved evidence, highlighting discrepancies between them.
Some approaches have incorporated transformer-based models trained for Natu-
ral Language Inference (NLI) to evaluate consistency between generated claims
and external data. Such models can be encoder or decoder-based. NLI mod-
els categorize hypotheses as entailed, neutral, or contradictory in relation to a
given premise. By establishing entailment relationships while disregarding the
distinction between neutrality and contradiction [3][4], NLI-based methodologies
provide a structured framework to assess the substantiation of generated text
by source data.
Nevertheless, processing extensive documents within the constrained context
window of NLI models presents a significant practical limitation. Several method-
ological solutions have emerged to address this challenge, ranging from comput-
ing token-level entailment scores for individual context documents and subse-
quently aggregating these values [5], to leveraging advancements in encoder-
based models to accommodate expanded context windows [6].
3 Method
3.1 Problem Statement
To explain our architectural approach, we need to address several key challenges
in designing an effective hallucination detection system:
1.Non-factual statements: LLMs often generate many non-factual state-
ments to make content more readable, such as titles and greetings. These
statements aren’t information-dense enough to count as hallucinations, but
they typically don’t align factually with the retrieved context.
2.Long context: Leading encoder-based models for classification tasks had
relatively limited context windows until recent developments [7]. Even now,
contexts retrieved in RAG systems tend to exceed what encoder-based mod-
els can process.
2

3.Prediction resolution: Current methods vary in how finely they detect
hallucinations—whether at the token level, proposition level, or across the
entire sequence.
3.2 The Grounded in Context Solution
Grounded in Context takes inspiration from RAG by addressing the long con-
text problem through creating specific premises for each claim in the output,
pulling the most relevant sections from the document collection. This represents
a straightforward approach when implementing proposition-level analysis.
We focus on proposition-level analysis because we assume that models can learn
the task more easily since entailment is fundamentally a relationship between
propositions, while still offering enough explainability for root cause analysis.
However, it’s worth noting that token-level classification offers better explain-
ability. While proposition-level methods can highlight hallucinated statements,
token-level approaches can pinpoint exactly which parts of a statement con-
tribute to hallucinations.
Fig. 1. : Distribution of the number of tokens in a claim after LLM generated sequences
were split by our chunker without setting a maximum token size.
Our method consists of the following steps:
1. Split the output Ointo claims C=c1, c2, ..., c nusing our recursive text
chunker Twith parameters maximum chunk size smaxand maximum chunk
overlap omax. Our experiments indicate that setting smax= 60 tokens cap-
tures almost all of cases without unnaturally segmenting sentences, as Figure
1 shows.
2. Filter out non-factual claims from Cusing a compact encoder-based factual
claims classifier F, resulting in filtered set C′⊆C.
3

3. Split the context Dinto chunks D=d1, d2, ..., d musing the same chunker
T. The chunking parameters are calibrated to fit a number of chunks that
is chosen dynamically depending on the claim length within a 512-token
context window after accounting for claim token length.
4. For each claim ci∈C′, retrieve the (dynamically chosen) kmost relevant
chunks Ri=di1, di2, ..., d ik⊂D.
5. Score each claim-chunk pair ( ci, dij) using the entailment probability pent(ci, dij)
from an NLI model M.
6. Aggregate the entailment scores using function Athat applies greater weight-
ing to negative classifications, effectively penalizing hallucinations.
4 Evaluation
We initially assessed the method on our private evaluation datasets, which more
closely simulate production-like use cases, where it demonstrated strong perfor-
mance. In addition, we benchmark our method on the RAGTruth dataset [2],
a popular word-level hallucination detection dataset which chooses 3 recognized
tasks for response generation: Question Answering, Data-to-text Writing, and
News Summarization.
Benchmarking the method exactly as described above with [8] used for retrieval
and [3] used as an NLI model yields an F1 score of 0.72, surpassing Luna, which
was the state-of-the-art until recently.
QUESTION ANSWERING DATA-TO-TEXT WRITING SUMMARIZATION OVERALL
Method Prec. Rec. F1 Prec. Rec. F1 Prec. Rec. F1 Prec. Rec. F1
Trained on the Dataset
Finetuned Llama-2-13B 61.6 76.3 68.2 85.4 91.0 88.1 64.0 54.9 59.1 76.9 80.7 78.7
RAG-HAT 76.5 73.1 74.8 92.9 90.3 91.6 77.7 59.8 67.6 87.3 80.8 83.9
Luna 37.8 80.0 51.3 64.9 91.2 75.9 40.0 76.5 52.5 52.7 86.1 65.4
lettucedetect-large-v1 65.9 75.0 70.2 90.4 86.7 88.5 64.0 55.9 59.7 80.4 78.05 79.2
Not Trained on the Dataset
Prompt gpt-4-turbo 33.2 90.6 45.6 64.3 100.0 78.3 31.5 97.6 47.6 46.9 97.9 63.4
SelCheckGPT gpt-3.5-turbo 35.0 58.0 43.7 68.2 82.8 74.8 31.1 56.5 40.1 49.7 71.9 58.8
LMvLM gpt-4-turbo 18.7 76.9 30.1 68.0 76.7 72.1 23.2 81.9 36.2 36.2 77.8 49.4
ChainPoll gpt-3.5-turbo 33.5 51.3 40.5 84.6 35.1 49.6 45.8 48.0 46.9 54.8 40.6 46.7
RAGAS Faithfulness 31.2 41.9 35.7 79.2 50.8 61.9 64.2 29.9 40.8 62.0 44.8 52.0
Trulens Groundedness 22.8 92.5 36.6 66.9 96.5 79.0 40.2 50.0 44.5 46.5 85.8 60.4
Grounded in Context (Ours) 88.4 91.9 90.1 64.4 82.2 72.2 85.35 87.1 86.2 79.4 87.1 83.05
Table 1. Performance comparison across the various tasks in RAGTruth. We compare
our results with several approaches presented in Luna [5], RAGTruth [2], RAG-HAT
[9] and LettuceDetect [6].
Deepchecks’ current method builds upon the principles in Section 3.2 and incor-
porates additional improvements in chunking, retrieval, and context construc-
tion, alongside a proprietary NLI model. It achieves an F1 score of 0.83,
second only to RAG-HAT (0.84), a substantially larger model that was trained on
RAGTruth, unlike our model, for which the dataset remains out-of-distribution.
4

It is notable that our framework’s improvements were less pronounced in the
data-to-text assignment, which we attribute to a combination of challenges in
effectively chunking formatted data and the limited representation of such data
in our training set, rather than an inherent deficiency in the methodology itself.
As our training data regarding structured text primarily consists of feature ex-
traction use-cases where entities are extracted in isolation, we hypothesize that
these specific use-cases explain the low-precision, high-recall ratio observed in
the data-to-text task.
5 Conclusion
In this paper, we presented Grounded in Context, an approach for hallucination
detection that effectively addresses the challenges of long contexts and varying
statement types. By decomposing outputs into discrete factual statements and
retrieving dedicated context for each, our method demonstrates strong perfor-
mance on hallucination detection tasks and the best results on RAGTruth for a
model of its size even though it wasn’t trained on the dataset.
5.1 Future Work
As our research predates the release of newer comparable encoder models with
enhanced long-context capabilities [7], we faced constraints in selecting optimal
chunking and retrieval parameters. These limitations may lead to suboptimal
contextualization of document chunks and necessitate a low retrieval value k,
reducing the probability of identifying the correct chunks for grounding output
claims. Such limitations become particularly pronounced when applying mod-
els to real-world, noisy data environments, which present significantly greater
challenges than the controlled conditions of curated benchmark datasets.
We hypothesize that integrating our method with models such as ModernBERT,
which efficiently processes sequences of up to 8,192 tokens, has the potential to
substantially enhance performance by overcoming existing retrieval and contex-
tual limitations. We plan to investigate this prospect in the coming months.
5

Bibliography
[1] Ziwei Ji, Nayeon Lee, Rita Frieske, Tiezheng Yu, Dan Su, Yan Xu, Etsuko
Ishii, Ye Jin Bang, Andrea Madotto, and Pascale Fung. Survey of hallucina-
tion in natural language generation. ACM Computing Surveys , 55(12):1–38,
March 2023. ISSN 1557-7341. https://doi.org/10.1145/3571730 . URL
http://dx.doi.org/10.1145/3571730 .
[2] Cheng Niu, Yuanhao Wu, Juno Zhu, Siliang Xu, Kashun Shum, Randy
Zhong, Juntong Song, and Tong Zhang. Ragtruth: A hallucination corpus
for developing trustworthy retrieval-augmented language models, 2024. URL
https://arxiv.org/abs/2401.00396 .
[3] Wenhao Wu, Wei Li, Xinyan Xiao, Jiachen Liu, Sujian Li, and Yajuan Lv.
Wecheck: Strong factual consistency checker via weakly supervised learning,
2023. URL https://arxiv.org/abs/2212.10057 .
[4] Zorik Gekhman, Jonathan Herzig, Roee Aharoni, Chen Elkind, and Idan
Szpektor. Trueteacher: Learning factual consistency evaluation with large
language models, 2023. URL https://arxiv.org/abs/2305.11171 .
[5] Masha Belyi, Robert Friel, Shuai Shao, and Atindriyo Sanyal. Luna: An
evaluation foundation model to catch language model hallucinations with
high accuracy and low cost, 2024. URL https://arxiv.org/abs/2406.
00975 .
[6]´Ad´ am Kov´ acs and G´ abor Recski. Lettucedetect: A hallucination detection
framework for rag applications, 2025. URL https://arxiv.org/abs/2502.
17125 .
[7] Benjamin Warner, Antoine Chaffin, Benjamin Clavi´ e, Orion Weller, Oskar
Hallstr¨ om, Said Taghadouini, Alexis Gallagher, Raja Biswas, Faisal Ladhak,
Tom Aarsen, Nathan Cooper, Griffin Adams, Jeremy Howard, and Iacopo
Poli. Smarter, better, faster, longer: A modern bidirectional encoder for
fast, memory efficient, and long context finetuning and inference, 2024. URL
https://arxiv.org/abs/2412.13663 .
[8] Xianming Li and Jing Li. Angle-optimized text embeddings, 2024. URL
https://arxiv.org/abs/2309.12871 .
[9] Juntong Song, Xingguang Wang, Juno Zhu, Yuanhao Wu, Xuxin Cheng,
Randy Zhong, and Cheng Niu. RAG-HAT: A hallucination-aware tuning
pipeline for LLM in retrieval-augmented generation. In Franck Dernon-
court, Daniel Preot ¸iuc-Pietro, and Anastasia Shimorina, editors, Proceedings
of the 2024 Conference on Empirical Methods in Natural Language Process-
ing: Industry Track , pages 1548–1558, Miami, Florida, US, November 2024.
Association for Computational Linguistics. https://doi.org/10.18653/

v1/2024.emnlp-industry.113 . URL https://aclanthology.org/2024.
emnlp-industry.113/ .
7