# Toward Agentic RAG for Ukrainian

**Authors**: Marta Sumyk, Oleksandr Kosovan

**Published**: 2026-04-16 11:40:06

**PDF URL**: [https://arxiv.org/pdf/2604.14896v1](https://arxiv.org/pdf/2604.14896v1)

## Abstract
We present an initial investigation into Agentic Retrieval-Augmented Generation (RAG) for Ukrainian, conducted within the UNLP 2026 Shared Task on Multi-Domain Document Understanding. Our system combines two-stage retrieval (BGE-M3 with BGE reranking) with a lightweight agentic layer performing query rephrasing and answer-retry loops on top of Qwen2.5-3B-Instruct. Our analysis reveals that retrieval quality is the primary bottleneck: agentic retry mechanisms improve answer accuracy but the overall score remains constrained by document and page identification. We discuss practical limitations of offline agentic pipelines and outline directions for combining stronger retrieval with more advanced agentic reasoning for Ukrainian.

## Full Text


<!-- PDF content starts -->

Toward Agentic RAG for Ukrainian*
Marta Sumyk and Oleksandr Kosovan
Ukrainian Catholic University, Lviv, Ukraine
sumyk.pn@ucu.edu.ua, o.kosovan@ucu.edu.ua
Abstract
We present an initial investigation into Agen-
tic Retrieval-Augmented Generation (RAG) for
Ukrainian, conducted within the UNLP 2026
Shared Task on Multi-Domain Document Un-
derstanding. Our system combines two-stage
retrieval (BGE-M3 with BGE reranking) with
a lightweight agentic layer performing query
rephrasing and answer-retry loops on top of
Qwen2.5-3B-Instruct. Our analysis reveals that
retrieval quality is the primary bottleneck: agen-
tic retry mechanisms improve answer accuracy
but the overall score remains constrained by
document and page identification. We discuss
practical limitations of offline agentic pipelines
and outline directions for combining stronger
retrieval with more advanced agentic reasoning
for Ukrainian.
1 Introduction
Retrieval-Augmented Generation (RAG) has
emerged as a key paradigm for grounding large
language model (LLM) outputs based on external
data (Lewis et al., 2020; Gao et al., 2024). Tradi-
tional RAG follows a fixed retrieve-then-generate
pipeline, which often falls short when tasks require
multi-step reasoning or iterative answer refinement.
These limitations have motivated Agentic RAG,
which embeds autonomous agent behaviors like
planning, reflection, tool use, and multi-step re-
trieval (Singh et al., 2025; Li et al., 2025).
The application of RAG to languages beyond
English remains uneven. Despite its large speaker
base and growing NLP ecosystem driven by the
UNLP workshop series (Romanyshyn et al., 2024),
agentic retrieval systems for Ukrainian have not
been studied. The availability of multilingual mod-
els and embeddings has made basic RAG feasible
for many languages, yet the agentic extensions that
push the state of the art in English have not been
*This paper is a research report based on our participation
in the UNLP 2026 Shared Task.explored for Ukrainian. The concept of agentic
retrieval is closely related to established NLP prob-
lems: fact-checking systems require both retrieval
and verification of factual correctness (Trokhy-
movych and Saez-Trumper, 2021), while two-stage
de-duplication systems mirror the retrieve-then-
reason architecture central to RAG (Trokhymovych
and Kosovan, 2023).
We present an initial investigation into Agentic
RAG for Ukrainian within the UNLP 2026 Shared
Task on Multi-Domain Document Understanding,
which requires systems to answer multiple-choice
questions and identify the supporting document and
page1. The competition runs on Kaggle in code-
only mode on a single P100 GPU with a 9-hour
time limit, imposing realistic constraints on agentic
pipeline design.
Our contributions are: (1) a modular pipeline
combining dense retrieval with reranking, open-
weight LLMs, and agentic retry mechanisms; (2)
discussion of practical challenges and future direc-
tions for agentic RAG for Ukrainian.
2 Related Work
RAG was introduced by Lewis et al. (2020) to en-
hance LLM outputs with retrieved documents, and
has been extended with adaptive retrieval and self-
reflection mechanisms (Gao et al., 2024; Asai et al.,
2023). Agentic RAG goes further by embedding
autonomous agents into the pipeline. Singh et al.
(2025) provide a taxonomy of architectures based
on reflection, planning, tool use, and multi-agent
collaboration. Li et al. (2025) synthesize retrieval-
reasoning interplay under a unified framework,
identifying synergized approaches where agentic
LLMs iteratively interleave search and reasoning.
Recent benchmarks have begun evaluating agen-
tic RAG. Xi et al. (2025) introduce InfoDeepSeek
1https://github.com/unlp-workshop/
unlp-2026-shared-taskarXiv:2604.14896v1  [cs.AI]  16 Apr 2026

for assessing agentic information seeking in dy-
namic web environments, finding that search qual-
ity significantly impacts performance. Ning et al.
(2026) propose MC-Search for evaluating mul-
timodal agentic RAG with structured reasoning
chains. Both works highlight that retrieval quality
remains a key bottleneck: a finding consistent with
our results.
For Ukrainian NLP we can analyze, the UNLP
2024 Shared Task focused on LLM fine-tuning (Ro-
manyshyn et al., 2024), with the winning system us-
ing RAG alongside fine-tuning (Boros et al., 2024).
3 Shared Task
The UNLP 2026 Shared Task requires systems to:
(1) predict the correct answer to a multiple-choice
question given a document collection, and (2) iden-
tify the supporting document and page. The data
spans multiple domains, with dev/test_public
containing two domains and test_private con-
taining three (including one unseen domain). The
final score combines answer accuracy ( ai), docu-
ment correctness (d i), and page proximity (p i):
Metric = 0.5· a+ 0.25· d+ 0.25· p,
where a,d, andpdenote the averages over all ques-
tions. Page proximity decreases linearly with page
distance when the document is correct, and is zero
otherwise.
4 Methodology
4.1 Retrieval
We evaluated sparse methods (TF-IDF with word,
character, and hybrid n-grams) and dense re-
trieval (multilingual-e5-small, xlm-roberta-base,
gte-multilingual-base, bge-m3). Character-level
TF-IDF outperforms word-level, likely due to ro-
bustness to Ukrainian morphological variation. Our
best configuration combines BGE-M3 for initial re-
trieval with BGE-reranker-v2-m3 for reranking.
4.2 LLM-Based Answer Generation
We evaluated five open-weight LLMs (Gemma,
LapaLLM, Llama-2-7b, MamayLM, Qwen2.5-3B-
Instruct) in LLM-only and LLM+retrieval settings.
Qwen2.5-3B-Instruct performed best in both set-
tings.
4.3 Agentic Layer
Our agentic layer implements two lightweight retry
mechanisms: (1)Query rephrasing: when modelTable 1: Search quality on the development set. Higher
values indicate better retrieval.
Method Meand iMeanp i
tfidf_word 0.7202 0.5855
tfidf_char 0.8742 0.7161
tfidf_hybrid 0.8438 0.6878
multilingual-e5-small 0.9109 0.7895
xlm-roberta-base 0.7202 0.5804
gte-multilingual-base 0.6074 0.4923
bge-m3 0.9002 0.7365
bge-m3 + bge reranker0.9219 0.8111
confidence is low, the agent rephrases the ques-
tion and performs a second retrieval pass; (2)An-
swer retry: the agent re-prompts the LLM with
additional instructions, selecting the final answer
based on confidence across attempts. Both are con-
strained by the single-GPU, 9-hour time limit.
5 Evaluation
5.1 Retrieval Quality
Table 1 reports retrieval performance on the devel-
opment set. BGE-M3 with reranking achieves the
best results (mean di= 0.922 , mean pi= 0.811 ),
substantially outperforming all single-model ap-
proaches.
5.2 Answer Selection
Table 2 compares LLM-only and LLM+retrieval
performance. All models improve substantially
with retrieved context, with Qwen2.5-3B-Instruct
reaching 0.69 accuracy (+27 points over LLM-
only). A simple answer-page similarity baseline
achieves 0.49, outperforming all LLM-only con-
figurations. Ukrainian-specific models (LapaLLM,
MamayLM) perform comparably to multilingual
baselines, suggesting that domain knowledge from
retrieval is more important than language-specific
pretraining.
5.3 Effect of Agentic Mechanisms
Table 3 shows the impact of agentic retry on the
final metric (Qwen2.5-3B-Instruct + BGE-M3 with
reranker). Combining both query rephrasing and
answer retry achieves 0.7704, a gain of ∼1 point
over the base pipeline. The improvements are con-
sistent but modest, reflecting the constraints of our
minimal agentic design compared to the sophisti-

Table 2: Answer accuracy on the development set: LLM-
only (no context) vs. LLM with retrieved context.
Method LLM-only +Retrieval
Baseline (similarity) — 0.49
Gemma 0.38 0.62
LapaLLM 0.35 0.66
Llama-2-7b 0.39 0.63
MamayLM 0.39 0.64
Qwen2.5-3B-Instruct0.42 0.69
Table 3: Effect of agentic mechanisms on the final met-
ric (dev set, Qwen2.5-3B-Instruct + BGE-M3 reranker).
Method Final metric
LLM-only answering 0.7607
+ query rephrasing 0.7615
+ answer retry 0.7629
+ both0.7704
cated patterns described in recent surveys (Li et al.,
2025; Singh et al., 2025).
5.4 Leaderboard Results and Analysis
Table 4 reports our two best submissions. The non-
agentic pipeline achieves a private score of 0.7664
with strong retrieval ( di= 0.907 ,pi= 0.814 ) but
moderate answer accuracy ( ai= 0.633 ). The agen-
tic pipeline scores slightly higher (0.7668) with
substantially better answer accuracy ( ai= 0.814 )
but lower page precision ( pi= 0.625 ), suggesting
that query rephrasing retrieves relevant but differ-
ent pages.
Our results (12th place on the leaderboard) con-
firm several findings. First,retrieval is the pri-
mary bottleneck: the largest performance gains
come from improving retrieval, not from chang-
ing the LLM or adding agentic mechanisms that
consistent with InfoDeepSeek findings (Xi et al.,
2025). Second,agentic mechanisms present a
trade-off: higher answer accuracy comes at the
cost of page localization precision. Third,small
LLMs are effective with good retrieval: the 27-
point accuracy improvement from LLM-only to
LLM+retrieval shows that retrieval quality matters
more than model size for this task.
6 Conclusion
We have presented an initial investigation into
Agentic RAG for Ukrainian, using the UNLP 2026Table 4: Leaderboard submissions: non-agentic vs.
agentic.
Metric Non-agentic Agentic
Public score 0.7365 0.7368
Private score 0.7664 0.7668
Meana i 0.6334 0.8142
Meand i 0.9067 0.9110
Meanp i 0.8142 0.6247
Final metric 0.7469 0.7437
Shared Task as a testbed. Our results show that re-
trieval quality dominates performance, with agentic
retry mechanisms providing modest but consistent
improvements. The competition’s offline, single-
GPU constraints limit the agentic capabilities that
can be explored; advanced patterns such as multi-
agent collaboration and iterative deep search (Li
et al., 2025; Xi et al., 2025) remain out of reach in
this setting.
Future work should address three directions: (1)
improving retrieval through domain-adaptive em-
beddings and Ukrainian-specific rerankers; (2) ex-
ploring sophisticated agentic architectures (multi-
step planning, self-reflection, chain-of-retrieval) in
unconstrained settings; and (3) combining shared
task insights with larger, API-accessible models
to quantify the full potential of Agentic RAG for
Ukrainian.
Limitations
Our agentic capabilities are minimal (single-step
retry) and do not capture the full spectrum of agen-
tic behaviors (Singh et al., 2025). The single-GPU,
9-hour constraint precludes larger models, multi-
agent architectures, or API-based services. Our
evaluation is limited to the UNLP 2026 dataset,
and we do not fine-tune models on Ukrainian data,
relying on multilingual pretraining. Generalizabil-
ity to other Ukrainian NLP tasks remains to be
established.
Ethical Consideration
We used Claude Opus 4.6 (Anthropic) and Gram-
marly for assistance with text editing and improv-
ing the writing of this paper.

References
Akari Asai, Zeqiu Wu, Yizhong Wang, Avirup Sil, and
Hannaneh Hajishirzi. 2023. Self-RAG: Learning to
retrieve, generate, and critique through self-reflection.
arXiv preprint arXiv:2310.11511.
Tiberiu Boros, Radu Chivereanu, Stefan Dumitrescu,
and Octavian Purcaru. 2024. Fine-tuning and re-
trieval augmented generation for question answering
using affordable large language models. InProceed-
ings of the Third Ukrainian Natural Language Pro-
cessing Workshop (UNLP) @ LREC-COLING 2024,
pages 75–82, Torino, Italia. ELRA and ICCL.
Yunfan Gao, Yun Xiong, Xinyu Gao, Kangxiang Jia,
Jinliu Pan, Yuxi Bi, Yi Dai, Jiawei Sun, Meng Wang,
and Haofen Wang. 2024. Retrieval-augmented gener-
ation for large language models: A survey.Preprint,
arXiv:2312.10997.
Patrick Lewis, Ethan Perez, Aleksandra Piktus, Fabio
Petroni, Vladimir Karpukhin, Naman Goyal, Hein-
rich Küttler, Mike Lewis, Wen-tau Yih, Tim Rock-
täschel, Sebastian Riedel, and Douwe Kiela. 2020.
Retrieval-augmented generation for knowledge-
intensive nlp tasks. InProceedings of the 34th Inter-
national Conference on Neural Information Process-
ing Systems, NIPS ’20, Red Hook, NY , USA. Curran
Associates Inc.
Yangning Li, Weizhi Zhang, Yuyao Yang, Wei-Chieh
Huang, Yaozu Wu, Junyu Luo, Yuanchen Bei,
Henry Peng Zou, Xiao Luo, Yusheng Zhao, Chunkit
Chan, Yankai Chen, Zhongfen Deng, Yinghui Li,
Hai-Tao Zheng, Dongyuan Li, Renhe Jiang, Ming
Zhang, Yangqiu Song, and Philip S. Yu. 2025. To-
wards agentic rag with deep reasoning: A sur-
vey of rag-reasoning systems in llms.Preprint,
arXiv:2507.09477.
Xuying Ning, Dongqi Fu, Tianxin Wei, Mengting Ai,
Jiaru Zou, Ting-Wei Li, Hanghang Tong, Yada Zhu,
Hendrik Hamann, and Jingrui He. 2026. Mc-search:
Evaluating and enhancing multimodal agentic search
with structured long reasoning chains.Preprint,
arXiv:2603.00873.
Mariana Romanyshyn, Oleksiy Syvokon, and Roman
Kyslyi. 2024. The UNLP 2024 shared task on fine-
tuning large language models for Ukrainian. InPro-
ceedings of the Third Ukrainian Natural Language
Processing Workshop (UNLP) @ LREC-COLING
2024, pages 67–74, Torino, Italia. ELRA and ICCL.
Aditi Singh, Abul Ehtesham, Saket Kumar, and Tala Ta-
laei Khoei. 2025. Agentic retrieval-augmented
generation: A survey on agentic rag.Preprint,
arXiv:2501.09136.
Mykola Trokhymovych and Oleksandr Kosovan. 2023.
Geodd: End-to-end spatial data de-duplication sys-
tem. InData Science and Algorithms in Systems,
pages 717–727, Cham. Springer International Pub-
lishing.Mykola Trokhymovych and Diego Saez-Trumper. 2021.
Wikicheck: An end-to-end open source automatic
fact-checking api based on wikipedia. InProceed-
ings of the 30th ACM International Conference on
Information & Knowledge Management, CIKM ’21,
page 4155–4164, New York, NY , USA. Association
for Computing Machinery.
Yunjia Xi, Jianghao Lin, Menghui Zhu, Yongzhao Xiao,
Zhuoying Ou, Jiaqi Liu, Tong Wan, Bo Chen, Wei-
wen Liu, Yasheng Wang, Ruiming Tang, Weinan
Zhang, and Yong Yu. 2025. Infodeepseek: Bench-
marking agentic information seeking for retrieval-
augmented generation.Preprint, arXiv:2505.15872.