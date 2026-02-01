# XProvence: Zero-Cost Multilingual Context Pruning for Retrieval-Augmented Generation

**Authors**: Youssef Mohamed, Mohamed Elhoseiny, Thibault Formal, Nadezhda Chirkova

**Published**: 2026-01-26 19:00:11

**PDF URL**: [https://arxiv.org/pdf/2601.18886v1](https://arxiv.org/pdf/2601.18886v1)

## Abstract
This paper introduces XProvence, a multilingual zero-cost context pruning model for retrieval-augmented generation (RAG), trained on 16 languages and supporting 100+ languages through effective cross-lingual transfer. Motivated by the growing use of RAG systems across diverse languages, we explore several strategies to generalize the Provence framework-which first integrated efficient zero-cost context pruning directly into the re-ranking model-beyond English. Across four multilingual question answering benchmarks, we show how XProvence can prune RAG contexts with minimal-to-no performance degradation and outperforms strong baselines. Our model is available at https://huggingface.co/naver/xprovence-reranker-bgem3-v2.

## Full Text


<!-- PDF content starts -->

January, 2025
XProvence: Zero-Cost Multilingual Context Pruning for
Retrieval-Augmented Generation
Youssef Mohamed1†Mohamed Elhoseiny1Thibault Formal2Nadezhda Chirkova2
1KAUST2NAVER LABS Europe
https://huggingface.co/naver/xprovence-reranker-bgem3-v2
11100ProvenceBinary mask used for pruning[BOS]0.8Rerank scoreHow many …Inﬂu  ence of … The most…QuestionContext sentence 1Context sentence 2 
Targets for training are obtained from a pretrained rerankerTargets for training are obtained by prompting an LLM
ﬁnetuned from BGE-reranker-v2-m3xRetrieverUser questionHow is Santa called in different countriesOther names are Sinterklaas or Pere NoelMany countries celebrate Christmas with diﬀerent traditions, and their Santa’s look diﬀerent too. The Dutch version of Santa Claus is called Sinterklaas. In France kids wait for Pere Noel. Click here to learn more!The Dutch version of Santa Claus is called Sinterklaas. In France kids wait for Pere Noel. Data storeRetrieved passages (100+ langs supported)Reranked and pruned passagesGen.  LLM
multilingual context pruner &  reranker
x(100+ langs supported)(a) Inference(b) Model architecture
Figure 1: XProvence speeds up generation in multilingual RAG pipelines through zero-cost context pruning, using
an extra prediction head on a multilingual reranker to classify each sentence as relevant or irrelevant to the query.
Abstract
This paper introduces XProvence, a multilingual zero-cost context pruning model for Retrieval-
Augmented Generation (RAG), trained on 16 languages and supporting 100+ languages through
effective cross-lingual transfer. Motivated by the growing use of RAG systems across diverse languages,
we explore several strategies to generalize the Provence framework—which first integrated efficient
zero-cost context pruning directly into the re-ranking model—beyond English. Across four multilingual
Question Answering benchmarks, we show how XProvence can prune RAG contexts with minimal-
to-no performance degradation and outperforms strong baselines. Our training code is available at
https://github.com/naver/bergen/tree/main/scripts/xprovence.
1. Introduction
Retrieval-Augmented Generation (RAG) has emerged
as a powerful paradigm for grounding Large Language
Models (LLMs) in external knowledge (Lewis et al.,
2020). By retrieving and conditioning on domain-
specific contexts, RAG systems have demonstrated
strong performance across a wide range of applications.
The increasing capabilities of LLMs have in turn made
RAG pipelines more effective.
However,thebenefitsofRAGcomewithsignificantcom-
putational costs. Retrieved documents substantially in-
crease the input context length, leading to quadratic
growth in inference time, higher deployment costs,
and a larger carbon footprint. Consequently, reduc-
ing the size of the context fed to the LLM has become a
key research focus. Among various compression strate-
gies (Wang et al., 2023; Yoon et al., 2024; Cheng et al.,
2024; Louis et al., 2025; Rau et al., 2025), the selec-
tive removal of irrelevant content from retrieved docu-
ments, known as context pruning, has shown particular
promise (Jiang et al., 2023; Xu et al., 2023; Chirkova
et al., 2025; Hwang et al., 2024).A state-of-the-art approach in this space is Provence
(Chirkova et al., 2025), which introduces azero-cost
pruning mechanism integrated directly into the rerank-
ing stage of the RAG pipeline. By leveraging the
reranker’s query-aware representations, Provence la-
bels sentences as relevant or irrelevant and prunes
non-relevant content prior to generation. This sim-
ple yet effective design yields significant runtime im-
provements without compromising performance. How-
ever, Provence remains limited to English, constrain-
ing its applicability in multilingual settings (Chirkova
et al., 2024). In this work, we address these limitations
by introducing XProvence, a multilingual extension of
Provence. Our contributions can be summarized as
follows:
•Optimal recipe:We study various strategies for
extending the language coverage of Provence, in-
cluding(i)cross-lingual transfer,(ii)multilingual
data annotation, and(iii)data translation, aiming
to find the optimal recipe for training XProvence;
•Multilingual context pruning:We introduce
XProvence, the first multilingual zero-cost context
Corresponding author(s): youssef.mohamed@kaust.edu.sa, nadia.chirkova@naverlabs.comarXiv:2601.18886v1  [cs.IR]  26 Jan 2026

XProvence: Zero-Cost Multilingual Context Pruning for Retrieval-Augmented Generation
pruning model, supporting 100+ languages.
•Extensive evaluation on 4 datasets: We show
that XProvence effectively compresses RAG con-
texts across languages, achieving minimal-to-no
degradation in downstream quality and outper-
forming strong baselines on non-English bench-
marks.
Our model is available at https://huggingface.
co/naver/xprovence-reranker-bgem3-v2
and our training code and data are available at
https://github.com/naver/bergen/tree/
main/scripts/xprovence.
2. Background
Reranking.Production-ready RAG systems typically
consist of three main components: a retriever, which
coarsely identifies relevant passages from a datastore; a
reranker, which refines the context selection; and a gen-
erator LLM, which produces the final output (Rau et al.,
2024). Rerankers score candidate passages by jointly
encoding the query and each passage, which is com-
putationally more expensive than bi-encoder retrieval
but typically yields more accurate rankings1. While
decoder-style LLM rerankers (Zhuang et al., 2024; Qin
et al., 2024; Sun et al., 2023) have recently advanced
the state of the art, BERT-style cross-encoder models
remain strong baselines and are often preferred in prac-
tice due to their efficiency and ease of deployment.
Cross-encoderstakeasinputaconcatenationofa <BOS>
token, a tokenized query, a <SEP>token, and a tok-
enized passage (Nogueira & Cho, 2020). A linear head
applied to the representation of the <BOS>token out-
puts a relevance score used to rank passages. Rerankers
are typically trained on query–document pairs to opti-
mize the relevance of retrieved passages.
Provence.Provence (Chirkova et al., 2025) proposes
to enhance rerankers with context pruning capabilities
by applying an additional linear head on top of the
representations of thepassage tokensas shown in Fig-
ure 1 (b). A linear head outputs per-token values in the
range(0,1), which are then binarized into {0,1}using
a predefined pruning threshold. Each sentence in a
passage is then classified as relevant if and only if more
than50%of its tokens are relevant. The irrelevant sen-
tences are removed from the passage before passing it
to the generator LLM, which results in reduced context
lengths and subsequent speed-ups in generation. By in-
tegrating context pruning directly into the reranker, the
1By contrast, retrievers encode the query and each passage in-
dependently, which allows passages to be pre-computed offline and
makes retrieval substantially faster, but generally at the cost of rank-
ing precision.approach introduces no additional cost to a standard
RAG pipeline.
Training a Provence model involves fine-tuning a
pretrained reranker with a context pruning objec-
tive—formulated as per-token binary classification us-
ing cross-entropy loss—and a regularization term that
preservesrerankingcapabilitiesthrougharegressionob-
jectivewithmeansquarederrorloss. Targetsforrerank-
ing are simply obtained from the pretrained reranker,
andtargetsforcontextpruningareobtainedbyprompt-
ing a strong LLM, e.g., Llama-3-8B (AI@Meta, 2024).
In particular, an LLM is provided with a query and a
passage, and prompted to answer the query using only
information in the passage, while citing sentences used
in the answer with a[i]template.
Prior context pruning approaches such as RECOMP (Xu
et al., 2023) or DSLR (Hwang et al., 2024) encode
sentences in a passage independently of each other.
In contrast, Provence encodes all the sentences in a
retrieved passage together with a query, in a single
reranker forward pass. It makes context pruning zero-
cost in the RAG pipeline and enables more precise con-
text pruning, since information about coreferences be-
tween sentences can now be used to make decisions
about sentence pruning. Collectively, these properties
establish Provence as aneffective and efficient practical
solutionforcontextpruning. However, itremainslimited
to English, constraining its applicability in multilingual
RAG settings.
3. Training Methodology for XProvence
We aim to extend Provence beyond English, enabling
effective zero-cost pruning across multiple languages.
As a foundation for our XProvence model, we rely on
the BGE-M3 reranker2(Chen et al., 2024) which sup-
ports 100+ languages. In search of the optimal train-
ing recipe, we empirically compare three strategies de-
scribed below.
Cross-Lingual Transfer.The simplest way to train
XProvence is to replicate the original Provence training
procedure—using the same English training data—but
initialize it from a multilingual reranker. The multilin-
gual context pruning capabilities of the final model
emerge from cross-lingual transfer (Conneau et al.,
2020; Artetxe et al., 2020; Wu & Dredze, 2019) and the
extensive multilingual pretraining of the base model.
In our experiments, we rely on the MS MARCO-based
dataset (Bajaj et al., 2018) available in the Provence
2In particular, we useBAAI/bge-reranker-v2-m3.
2

XProvence: Zero-Cost Multilingual Context Pruning for Retrieval-Augmented Generation
repository3.
Data Translation.An alternative strategy, commonly
used to extend NLP models beyond English is to trans-
late the English data into multiple languages (Boni-
facio et al., 2022). For each language, we translate
125𝑘randomly sampled query-context pairs from the
MS MARCO-based dataset used to train Provence. We
perform sentence-by-sentence translation into 16 lan-
guages using a strong translator LLM, namely Gem-
maX2 9B (Cui et al., 2025).
Multilingual Data Annotation.A third strategy is to
rely on multilingual-by-design data. For this purpose,
we use the MIRACL dataset (Zhang et al., 2023), com-
monly employed to train multilingual retrievers and
rerankers. MIRACL contains queries in 16 languages,
paired with passages in the same languages sourced
from Wikipedia. We utilize the multilingual Aya Ex-
panse 8B (Dang et al., 2024) model to generate syn-
thetic labeling for context pruning, following the same
procedure as in Provence. We provide instructions in
English and ask the model to answer in the query lan-
guage, as this strategy worked best in our preliminary
experiments.
4. Experimental Setup
Training details.We train all our models on 4 Nvidia
A100 GPUs, and run evaluations using a single A100
GPUper experiment. Similarto Provence, we train for a
total of 5 epochs, setting the learning rate to1 𝑒−5and
batch size to64. We utilize spaCy’s multilingual sen-
tence tokenizer to split passages into sentences. Trans-
lated and annotated datasets include 16 languages4.
Evaluation.We use the Bergen library (Rau et al.,
2024) for evaluation, with Aya Expanse-8B (Dang et al.,
2024) as a generator LLM for all datasets. We run
generation with greedy decoding and only evaluate on
23 languages supported by Aya Expanse 8B.
Evaluationdatasets.Weconductevaluationsonstan-
dard Multilingual Question Answering (MQA) bench-
marks: MKQA (Longpre et al., 2021) (16 langs5, 2.8𝑘
ex./lang.) and TydiQA (Clark et al., 2020) (5 langs6,
400-700 ex./lang.), both relying on a Wikipedia datas-
tore (passages in the same language as the query lan-
3https://github.com/naver/bergen/tree/main/
scripts/provence
4Training data languages: ar,bn,en,es,fa,fi,fr,hi,id,ja,
ko,ru,sw,te,th,zh.
5Considered MKQA languages: seen: ar,en,es,fr,ja,ko,ru,
zh; unseen:de,he,it,nl,pl,pt,tr,vi
6Considered TydiQA languages:ar,en,id,ko,ruguage or in English). We follow the RAG experimental
setup of the Bergen library (Chirkova et al., 2024; Rau
et al., 2024), employing both the BGE-M3 multilingual
retriever and reranker (Chen et al., 2024) (top-5 pas-
sages per query are fed to the LLM), along with the
character 3-gram evaluation metric (Chirkova et al.,
2024). This metric measures the proportion of charac-
ter 3-grams from the short ground-truth label that are
present in the LLM-generated answer.
To evaluate XProvence beyond the Wikipedia domain,
we further consider two other MQA datasets: MedEx-
pQA(Alonsoetal.,2024)(multiplechoicemedicalques-
tions, 4 languages7, 125 ex./lang.) and XPQA (Shen
et al., 2023) (questions about e-commerce products, 11
languages8, 1.2𝑘-1.9𝑘ex./lang.). Both datasets pro-
vide a gold context for each query—retrieval is thus not
needed. We evaluate MedExpQA using accuracy and
XPQA using LLM-as-a-judge (Rau et al., 2024).
Baselines.We compare training strategies described
in Section 3. We also include DSLR—the most effec-
tive baseline from the Provence paper (Hwang et al.,
2024).DSLRsegments input contexts into individual
sentences and encodes each sentence together with the
query using a pretrained reranker. Sentences whose
reranking scores exceed a predefined threshold are re-
tained in their original order. However, DSLR faces
a key limitation addressed by XProvence: it encodes
sentences independently, thereby neglecting semantic
relationships across sentences. For a fair comparison,
we evaluate DSLR using the BGE-M3 reranker.
5. Experiments
Results are shown in Fig. 29. For each dataset, we plot
the task metric on the 𝑥axis, the context compression
rate10onthe𝑦axis, andpresentaParetofrontobtained
by varying the pruning threshold. Lines located closer
to the top right corner are best performing. We answer
several research questions in the following.
RQ1: WhatisthebesttrainingrecipeforXProvence?
We compare the three training strategies described
in Section 3:Cross-Lingual Transfer (CLT)vsData
Translation (DT)vsMultilingual Data Annotation
(MDT). For simplicity, all models in this subsection are
trainedforcontextcompressiononly, withoutreranking
loss.
7MedExpQA languages:en,es,fr,it
8Considered XPQA languages: ar,de,es,fr,hi,it,ja,ko,pl,
pt,zh
9Per-language results are provided in the code repository.
10Context compression is defined as the average portion of pruned-
out context.
3

XProvence: Zero-Cost Multilingual Context Pruning for Retrieval-Augmented Generation
Figure 2: Main results: for each dataset, we present a Pareto front obtained by varying the pruning threshold
and averaged over languages. Lines located closer to the top right corner are best performing. Notation 𝐿𝑞/𝐿𝑐𝑛𝑡𝑥
denotes the language of query/passage. For MKQA, we perform controlled experiments with various language
settings, including settings 𝐿𝑐𝑛𝑡𝑥=𝐿𝑞and𝐿𝑐𝑛𝑡𝑥=𝐸𝑛. For the former setting, we present separately the results for
languages seen and unseen in the training data. For the remaining datasets, we follow their original setting.
Reranker MKQA (R@20) MIRACL (nDCG@10)
BGE-M3 70.3 74.5
XProvence (ours) 70.7 74.3
Table 1: Reranking Performance averaged across languages.
Our conclusions are as follows:(1)TheCLTstrategy
is surprisingly effective and reaches similar or supe-
rior performance compared to other strategies on all
datasets;(2)ComparingDTvsCLT, we observe that
translating MS MARCO data does not bring consistent
improvements;(3)ComparingMDTvsCLT, we observe
that training on multilingual by-design but limited-
domain data (Wikipedia-based MIRACL) leads to lower
performance compared to training on the more diverse
MS MARCO data, i.e., theCLTlines are located closer
to the top right corners on Pareto fronts.
For our final model,XProvence (w/reranking), we use
the data translation strategy trained with Provence’s
joint objective. Our results on the effectiveness of cross-
lingualtransferalignwellwiththepriorresultsreported
in the literature (Wu & Dredze, 2019; Pires et al., 2019;
K et al., 2020).RQ2: How does XProvence compare against DSLR?
Comparing our final model,XProvence (w/ rerank-
ing)andDSLR, we observe thatXProvenceoutper-
formsDSLRin all cases, except XPQA. We note that
XProvenceis zero-cost, whileDSLRincurs extra com-
putational costs in the RAG pipeline, as reranking and
context pruning require separate forward passes.
RQ3: Does XProvence enable context compression
with minimal-to-no performance loss?The pruning
strength of XProvence can be adjusted through a prun-
ing threshold. We observe that for MKQA and TyDiQA,
XProvenceenables the compression rate of 40—60%,
with the same performance as the full context setting.
For XPQA and MedExpQA, the achieved pruning com-
pression is lower, due to the nature of gold contexts
which are provided as parts of the datasets.
4

XProvence: Zero-Cost Multilingual Context Pruning for Retrieval-Augmented Generation
RQ4: Does XProvence perform well on languages
unseen in the training data and when the context
language differs from the query language?We con-
sider these settings for MKQA in Figure 2 (b) and
(c) correspondingly and find that the performance of
XProvenceis robust to such settings. This is the result
of the effective cross-lingual transfer discussed above.
RQ5: Does integration of context pruning preserve
reranking performance?We evaluate in Table 1 the
reranking performance of XProvence using FlagEmbed-
ding11with default settings. We show that XProvence
maintainsthererankingperformanceofthebasemodel,
while being able to perform context pruning.
6. Conclusion
In this work, we present XProvence, a reranking model
equipped with integrated context pruning, capable of
handling queries and contexts in over 100+ languages.
Throughanempiricalinvestigationoftrainingstrategies
on multiple MQA benchmarks, we show the effective-
ness of an easy-to-implement strategy, consisting of tun-
ing a multilingual reranker on English context-pruning
data and exploiting cross-lingual transfer.
References
AI@Meta. Llama 3 model card. 2024. URL
https://github.com/meta-llama/llama3/
blob/main/MODEL_CARD.md. 2
Iñigo Alonso, Maite Oronoz, and Rodrigo Agerri. Med-
expqa: Multilingual benchmarking of large language
models for medical question answering.Artificial
intelligence in medicine, 155:102938, 2024. 3
Mikel Artetxe, Sebastian Ruder, and Dani Yogatama.
On the cross-lingual transferability of monolingual
representations. In Dan Jurafsky, Joyce Chai, Na-
talie Schluter, and Joel Tetreault (eds.),Proceedings
of the 58th Annual Meeting of the Association for
Computational Linguistics, pp. 4623–4637, Online,
July 2020. Association for Computational Linguistics.
doi: 10.18653/v1/2020.acl-main.421. URL https:
//aclanthology.org/2020.acl-main.421/ . 2
Payal Bajaj, Daniel Campos, Nick Craswell, Li Deng,
Jianfeng Gao, Xiaodong Liu, Rangan Majumder, An-
drew McNamara, Bhaskar Mitra, Tri Nguyen, Mir
Rosenberg, Xia Song, Alina Stoica, Saurabh Tiwary,
and Tong Wang. Ms marco: A human generated
machine reading comprehension dataset, 2018. URL
https://arxiv.org/abs/1611.09268. 2
LuizBonifacio,VitorJeronymo,HugoQueirozAbonizio,
11https://github.com/FlagOpen/FlagEmbeddingIsrael Campiotti, Marzieh Fadaee, Roberto Lotufo,
and Rodrigo Nogueira. mmarco: A multilingual ver-
sion of the ms marco passage ranking dataset, 2022.
URLhttps://arxiv.org/abs/2108.13897. 3
Jianlv Chen, Shitao Xiao, Peitian Zhang, Kun Luo,
Defu Lian, and Zheng Liu. Bge m3-embedding:
Multi-lingual, multi-functionality, multi-granularity
text embeddings through self-knowledge distillation.
CoRR, 2024. 2, 3
XinCheng,XunWang,XingxingZhang,TaoGe,Si-Qing
Chen, Furu Wei, Huishuai Zhang, and Dongyan Zhao.
xRAG: Extreme context compression for retrieval-
augmented generation with one token. InThe Thirty-
eighth Annual Conference on Neural Information Pro-
cessing Systems, 2024. URL https://openreview.
net/forum?id=6pTlXqrO0p. 1
NadezhdaChirkova, DavidRau, HervéDéjean, Thibault
Formal, Stéphane Clinchant, and Vassilina Nikoulina.
Retrieval-augmented generation in multilingual set-
tings. In Sha Li, Manling Li, Michael JQ Zhang,
Eunsol Choi, Mor Geva, Peter Hase, and Heng
Ji (eds.),Proceedings of the 1st Workshop on To-
wards Knowledgeable Language Models (KnowLLM
2024), pp. 177–188, Bangkok, Thailand, August
2024. Association for Computational Linguistics. doi:
10.18653/v1/2024.knowllm-1.15. URL https://
aclanthology.org/2024.knowllm-1.15/ . 1, 3
Nadezhda Chirkova, Thibault Formal, Vassilina
Nikoulina, and Stéphane CLINCHANT. Provence:
efficient and robust context pruning for retrieval-
augmented generation. InThe Thirteenth Interna-
tional Conference on Learning Representations, 2025.
1, 2
Jonathan H. Clark, Eunsol Choi, Michael Collins, Dan
Garrette, Tom Kwiatkowski, Vitaly Nikolaev, and
Jennimaria Palomaki. TyDi QA: A benchmark for
information-seeking question answering in typolog-
ically diverse languages.Transactions of the As-
sociation for Computational Linguistics, 8:454–470,
2020. doi: 10.1162/tacl_a_00317. URL https:
//aclanthology.org/2020.tacl-1.30/. 3
Alexis Conneau, Kartikay Khandelwal, Naman Goyal,
Vishrav Chaudhary, Guillaume Wenzek, Francisco
Guzmán, Edouard Grave, Myle Ott, Luke Zettle-
moyer, and Veselin Stoyanov. Unsupervised cross-
lingual representation learning at scale. InProceed-
ings of the 58th Annual Meeting of the Association for
Computational Linguistics, pp. 8440–8451, Online,
July 2020. Association for Computational Linguistics.
doi: 10.18653/v1/2020.acl-main.747. URL https:
5

XProvence: Zero-Cost Multilingual Context Pruning for Retrieval-Augmented Generation
//aclanthology.org/2020.acl-main.747. 2
Menglong Cui, Pengzhi Gao, Wei Liu, Jian Luan, and
Bin Wang. Multilingual machine translation with
open large language models at practical scale: An
empirical study. In Luis Chiruzzo, Alan Ritter, and
Lu Wang (eds.),Proceedings of the 2025 Conference
of the Nations of the Americas Chapter of the Asso-
ciation for Computational Linguistics: Human Lan-
guageTechnologies(Volume1: LongPapers),pp.5420–
5443, Albuquerque, New Mexico, April 2025. Asso-
ciation for Computational Linguistics. ISBN 979-8-
89176-189-6. doi: 10.18653/v1/2025.naacl-long.
280. URL https://aclanthology.org/2025.
naacl-long.280/. 3
John Dang, Shivalika Singh, Daniel D’souza, Arash
Ahmadian, Alejandro Salamanca, Madeline Smith,
Aidan Peppin, Sungjin Hong, Manoj Govindassamy,
Terrence Zhao, Sandra Kublik, Meor Amer, Viraat
Aryabumi, Jon Ander Campos, Yi-Chern Tan, Tom
Kocmi, Florian Strub, Nathan Grinsztajn, Yannis
Flet-Berliac, Acyr Locatelli, Hangyu Lin, Dwarak
Talupuru, Bharat Venkitesh, David Cairuz, Bowen
Yang, Tim Chung, Wei-Yin Ko, Sylvie Shang Shi,
Amir Shukayev, Sammie Bae, Aleksandra Piktus, Ro-
man Castagné, Felipe Cruz-Salinas, Eddie Kim, Lu-
cas Crawhall-Stein, Adrien Morisot, Sudip Roy, Phil
Blunsom, Ivan Zhang, Aidan Gomez, Nick Frosst,
MarziehFadaee,BeyzaErmis,AhmetÜstün,andSara
Hooker. Aya expanse: Combining research break-
throughs for a new multilingual frontier, 2024. URL
https://arxiv.org/abs/2412.04261. 3
TaehoHwang,SoyeongJeong,SukminCho,SeungYoon
Han, and Jong C Park. Dslr: Document refinement
with sentence-level re-ranking and reconstruction to
enhance retrieval-augmented generation. InProceed-
ings of the 3rd Workshop on Knowledge Augmented
Methods for NLP, pp. 73–92, 2024. 1, 2, 3
Huiqiang Jiang, Qianhui Wu, Chin-Yew Lin, Yuqing
Yang, and Lili Qiu. LLMLingua: Compressing
prompts for accelerated inference of large lan-
guage models. In Houda Bouamor, Juan Pino,
and Kalika Bali (eds.),Proceedings of the 2023
Conference on Empirical Methods in Natural Lan-
guage Processing, pp. 13358–13376, Singapore, De-
cember 2023. Association for Computational Lin-
guistics. doi: 10.18653/v1/2023.emnlp-main.
825. URL https://aclanthology.org/2023.
emnlp-main.825/. 1
Karthikeyan K, Zihan Wang, Stephen Mayhew, and
Dan Roth. Cross-lingual ability of multilingual bert:
An empirical study. InInternational Conference onLearning Representations, 2020. URL https://
openreview.net/forum?id=HJeT3yrtDr. 4
Patrick Lewis, Ethan Perez, Aleksandra Piktus, Fabio
Petroni, Vladimir Karpukhin, Naman Goyal, Heinrich
Küttler, Mike Lewis, Wen-tau Yih, Tim Rocktäschel,
et al. Retrieval-augmented generation for knowledge-
intensive nlp tasks.Advances in neural information
processing systems, 33:9459–9474, 2020. 1
Shayne Longpre, Yi Lu, and Joachim Daiber. MKQA:
A linguistically diverse benchmark for multilin-
gual open domain question answering.Transac-
tions of the Association for Computational Linguis-
tics, 9:1389–1406, 2021. doi: 10.1162/tacl_a_
00433. URL https://aclanthology.org/2021.
tacl-1.82/. 3
Maxime Louis, Hervé Déjean, and Stéphane Clin-
chant. PISCO: Pretty simple compression for
retrieval-augmented generation. In Wanxiang Che,
Joyce Nabende, Ekaterina Shutova, and Moham-
mad Taher Pilehvar (eds.),Findings of the Associ-
ation for Computational Linguistics: ACL 2025, pp.
15506–15521, Vienna, Austria, July 2025. Associ-
ation for Computational Linguistics. ISBN 979-8-
89176-256-5. doi: 10.18653/v1/2025.findings-acl.
800. URL https://aclanthology.org/2025.
findings-acl.800/. 1
Rodrigo Nogueira and Kyunghyun Cho. Passage re-
ranking with bert, 2020. URL https://arxiv.
org/abs/1901.04085. 2
Telmo Pires, Eva Schlinger, and Dan Garrette. How
multilingual is multilingual BERT? In Anna Korho-
nen, David Traum, and Lluís Màrquez (eds.),Pro-
ceedings of the 57th Annual Meeting of the Associa-
tion for Computational Linguistics, pp. 4996–5001,
Florence, Italy, July 2019. Association for Computa-
tional Linguistics. doi: 10.18653/v1/P19-1493. URL
https://aclanthology.org/P19-1493/. 4
Zhen Qin, Rolf Jagerman, Kai Hui, Honglei Zhuang,
Junru Wu, Le Yan, Jiaming Shen, Tianqi Liu, Jialu
Liu, Donald Metzler, Xuanhui Wang, and Michael
Bendersky. Large language models are effective
text rankers with pairwise ranking prompting. In
Kevin Duh, Helena Gomez, and Steven Bethard
(eds.),Findings of the Association for Computational
Linguistics: NAACL 2024, pp. 1504–1518, Mex-
ico City, Mexico, June 2024. Association for Com-
putational Linguistics. doi: 10.18653/v1/2024.
findings-naacl.97. URL https://aclanthology.
org/2024.findings-naacl.97/. 2
DavidRau, HervéDéjean, NadezhdaChirkova, Thibault
6

XProvence: Zero-Cost Multilingual Context Pruning for Retrieval-Augmented Generation
Formal, Shuai Wang, Stéphane Clinchant, and Vas-
silina Nikoulina. BERGEN: A benchmarking library
for retrieval-augmented generation. In Yaser Al-
Onaizan, Mohit Bansal, and Yun-Nung Chen (eds.),
Findings of the Association for Computational Linguis-
tics: EMNLP 2024, pp. 7640–7663, Miami, Florida,
USA, November 2024. Association for Computational
Linguistics. doi: 10.18653/v1/2024.findings-emnlp.
449. URL https://aclanthology.org/2024.
findings-emnlp.449/. 2, 3
David Rau, Shuai Wang, Hervé Déjean, Stéphane
Clinchant, and Jaap Kamps. Context embed-
dings for efficient answer generation in retrieval-
augmented generation. InProceedings of the Eigh-
teenth ACM International Conference on Web Search
and Data Mining, WSDM ’25, pp. 493–502, New
York, NY, USA, 2025. Association for Computing
Machinery. ISBN 9798400713293. doi: 10.1145/
3701551.3703527. URL https://doi.org/10.
1145/3701551.3703527. 1
Xiaoyu Shen, Akari Asai, Bill Byrne, and Adria De Gis-
pert. xPQA: Cross-lingual product question an-
swering in 12 languages. In Sunayana Sitaram,
Beata Beigman Klebanov, and Jason D Williams
(eds.),Proceedings of the 61st Annual Meeting of
the Association for Computational Linguistics (Vol-
ume 5: Industry Track), pp. 103–115, Toronto,
Canada, July 2023. Association for Computational
Linguistics. doi: 10.18653/v1/2023.acl-industry.
12. URL https://aclanthology.org/2023.
acl-industry.12/. 3
Weiwei Sun, Lingyong Yan, Xinyu Ma, Shuaiqiang
Wang, Pengjie Ren, Zhumin Chen, Dawei Yin, and
Zhaochun Ren. Is ChatGPT good at search? investi-
gating large language models as re-ranking agents.
In Houda Bouamor, Juan Pino, and Kalika Bali (eds.),
Proceedings of the 2023 Conference on Empirical Meth-
odsinNaturalLanguageProcessing,pp.14918–14937,
Singapore, December 2023. Association for Com-
putational Linguistics. doi: 10.18653/v1/2023.
emnlp-main.923. URL https://aclanthology.
org/2023.emnlp-main.923/. 2
Zhiruo Wang, Jun Araki, Zhengbao Jiang, Md Rizwan
Parvez, and Graham Neubig. Learning to filter con-
text for retrieval-augmented generation, 2023. URL
https://arxiv.org/abs/2311.08377. 1
Shijie Wu and Mark Dredze. Beto, bentz, becas: The
surprising cross-lingual effectiveness of BERT. In
Kentaro Inui, Jing Jiang, Vincent Ng, and Xiaojun
Wan (eds.),Proceedings of the 2019 Conference on
Empirical Methods in Natural Language Processingand the 9th International Joint Conference on Natu-
ral Language Processing (EMNLP-IJCNLP), pp. 833–
844, Hong Kong, China, November 2019. Association
for Computational Linguistics. doi: 10.18653/v1/
D19-1077. URL https://aclanthology.org/
D19-1077/. 2, 4
Fangyuan Xu, Weijia Shi, and Eunsol Choi. Recomp:
Improving retrieval-augmented lms with compres-
sion and selective augmentation.arXiv preprint
arXiv:2310.04408, 2023. 1, 2
Chanwoong Yoon, Taewhoo Lee, Hyeon Hwang, Min-
byul Jeong, and Jaewoo Kang. CompAct: Com-
pressing retrieved documents actively for question
answering. In Yaser Al-Onaizan, Mohit Bansal,
and Yun-Nung Chen (eds.),Proceedings of the 2024
Conference on Empirical Methods in Natural Lan-
guage Processing, pp. 21424–21439, Miami, Florida,
USA, November 2024. Association for Computational
Linguistics. doi: 10.18653/v1/2024.emnlp-main.
1194. URL https://aclanthology.org/2024.
emnlp-main.1194/. 1
Xinyu Zhang, Nandan Thakur, Odunayo Ogundepo,
Ehsan Kamalloo, David Alfonso-Hermelo, Xiaoguang
Li, Qun Liu, Mehdi Rezagholizadeh, and Jimmy Lin.
MIRACL: A multilingual retrieval dataset covering
18 diverse languages.Transactions of the Associa-
tion for Computational Linguistics, 11:1114–1131,
2023. doi: 10.1162/tacl_a_00595. URL https:
//aclanthology.org/2023.tacl-1.63/. 3
Honglei Zhuang, Zhen Qin, Kai Hui, Junru Wu, Le Yan,
Xuanhui Wang, and Michael Bendersky. Beyond yes
and no: Improving zero-shot LLM rankers via scoring
fine-grained relevance labels. In Kevin Duh, Helena
Gomez, and Steven Bethard (eds.),Proceedings of the
2024 Conference of the North American Chapter of the
AssociationforComputationalLinguistics: HumanLan-
guageTechnologies(Volume2: ShortPapers), pp.358–
370, Mexico City, Mexico, June 2024. Association for
Computational Linguistics. doi: 10.18653/v1/2024.
naacl-short.31. URL https://aclanthology.
org/2024.naacl-short.31/. 2
7