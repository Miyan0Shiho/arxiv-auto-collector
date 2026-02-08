# Less Finetuning, Better Retrieval: Rethinking LLM Adaptation for Biomedical Retrievers via Synthetic Data and Model Merging

**Authors**: Sameh Khattab, Jean-Philippe Corbeil, Osman Alperen Koraş, Amin Dada, Julian Friedrich, François Beaulieu, Paul Vozila, Jens Kleesiek

**Published**: 2026-02-04 16:36:00

**PDF URL**: [https://arxiv.org/pdf/2602.04731v1](https://arxiv.org/pdf/2602.04731v1)

## Abstract
Retrieval-augmented generation (RAG) has become the backbone of grounding Large Language Models (LLMs), improving knowledge updates and reducing hallucinations. Recently, LLM-based retriever models have shown state-of-the-art performance for RAG applications. However, several technical aspects remain underexplored on how to adapt general-purpose LLMs into effective domain-specific retrievers, especially in specialized domains such as biomedicine. We present Synthesize-Train-Merge (STM), a modular framework that enhances decoder-only LLMs with synthetic hard negatives, retrieval prompt optimization, and model merging. Experiments on a subset of 12 medical and general tasks from the MTEB benchmark show STM boosts task-specific experts by up to 23.5\% (average 7.5\%) and produces merged models that outperform both single experts and strong baselines without extensive pretraining. Our results demonstrate a scalable, efficient path for turning general LLMs into high-performing, domain-specialized retrievers, preserving general-domain capabilities while excelling on specialized tasks.

## Full Text


<!-- PDF content starts -->

Less Finetuning, Better Retrieval: Rethinking LLM Adaptation
for Biomedical Retrievers via Synthetic Data and Model Merging
Sameh Khattab1*,Jean-Philippe Corbeil2*,Osman Alperen Kora¸ s1,Amin Dada1
Julian Friedrich1,François Beaulieu2,Paul Vozila2,Jens Kleesiek1†
1IKIM, University Hospital Essen, Germany2Microsoft Healthcare & Life Sciences
Abstract
Retrieval-augmented generation (RAG) has be-
come the backbone of grounding Large Lan-
guage Models (LLMs), improving knowledge
updates and reducing hallucinations. Recently,
LLM-based retriever models have shown state-
of-the-art performance for RAG applications.
However, several technical aspects remain un-
derexplored on how to adapt general-purpose
LLMs into effective domain-specific retriev-
ers, especially in specialized domains such
as biomedicine. We present Synthesize-Train-
Merge (STM), a modular framework that en-
hances decoder-only LLMs with synthetic hard
negatives, retrieval prompt optimization, and
model merging. Experiments on a subset of
12 medical and general tasks from the MTEB
benchmark show STM boosts task-specific ex-
perts by up to 23.5% (average 7.5%) and pro-
duces merged models that outperform both sin-
gle experts and strong baselines without exten-
sive pretraining. Our results demonstrate a scal-
able, efficient path for turning general LLMs
into high-performing, domain-specialized re-
trievers, preserving general-domain capabilities
while excelling on specialized tasks.
1 Introduction
Retrieval-augmented generation (RAG) (Lewis
et al., 2020) has become a standard approach for
grounding Large Language Models (LLMs), lead-
ing to improved knowledge updating and reduced
hallucinations (Xiong et al., 2024; Fan et al., 2024;
Abo El-Enen et al., 2025; Ayala and Bechard, 2024;
Barry et al., 2025). RAG systems typically rely
on lexical and semantic search methods (Sawarkar
et al., 2024; Wang et al., 2024c), implemented via
sparse or dense retrievers. Dense retrievers based
*Corresponding authors:sameh.khattab@uk-essen.de,
jcorbeil@microsoft.com
†Other affiliations: Cancer Research Center Cologne Essen
(CCCE), German Cancer Consortium (DKTK, Partner site Es-
sen) and Department of Physics of TU Dortmund (Dortmund,
Germany).
Real
MedicalSynthetic
MedicalNLU Search
Models
- Qwen3 0.6B
- Gemma 2B
- Phi4 3.8B2. LoRA
Fine-tuningLLMPositive
NegativeQuery
Harder
Negative
3. Merge1.1 Hard Negative
Generation
Prompt1.2 Prompt
Optimization
Better
Prompt1. Synthetic Data
STMFigure 1: Diagram of our recipe to obtain theSTM
retrievers:1)synthetic data — including1.1)hard nega-
tive generation and1.2)retrieval prompt optimization
—,2)LoRA fine-tuning, and3)model merging. We
segment the BMRetriever dataset into four splits:Real
Medical,Synthetic Medical,NLU, andSearch.
on decoder-only LLMs achieve state-of-the-art per-
formances on embedding-related tasks (Wang et al.,
2024a; BehnamGhader et al., 2024). These results
suggest that general-purpose LLMs already provide
a strong foundation for retrieval.
However, important questions remain on how
such models should be adapted, especially for
domain-specific retrieval such as in the biomedical
field. While zero-shot approaches (Li and Zhou,
2025; Springer et al., 2025) reported some suc-
cesses, fine-tuned methods (BehnamGhader et al.,
2024; Ni et al., 2022; Wang et al., 2022) are at the
top of the MTEB leaderboard (Muennighoff et al.,
2022). However, certain technical aspects remain
underexplored about how to best convert general-
purpose LLMs into domain-specific retrievers.
1arXiv:2602.04731v1  [cs.CL]  4 Feb 2026

Previous work (Xu et al., 2024) has shown that
contrastive pre-training of LLMs on a large corpus,
followed by instruction tuning, yields strong dense
retrieval models for the medical domain. Hard
negative mining has also been shown to substan-
tially improve retriever performance (Moreira et al.,
2024; Lee et al., 2025; Shao et al., 2025), and
Weller et al. (2025) highlight the importance of
prompts for retriever models. Despite this progress,
several questions remain open: is continual pre-
training or all the finetuning data necessary to ob-
tain strong retrievers? Which subsets of the data
mix are the most effective for fine-tuning? Can
prompt optimization lead to further gains? Can
top-tier LLMs be used to synthesize effective hard
negatives?
In parallel, model merging (Goddard et al., 2024)
has emerged as techniques to compose robust mod-
els (Wortsman et al., 2022; Ahmadian et al., 2024)
from expert models, enabling modular develop-
ment (Corbeil et al., 2025) and efficient data abla-
tion (Na et al., 2024). Basic model-merging tech-
niques such as ModelSoup (Wortsman et al., 2022)
have been incorporated into the training recipes
of two recent models: EmbeddingGemma (Vera
et al., 2025), and Qwen3 Embedding (Zhang et al.,
2025). Nonetheless, questions remain about using
model merging to build retriever models: are there
clear gains compared to full fine-tuning? Which
data subsets are most effective? Which merging
technique offers the best performance?
In this work, we present Synthesize-Train-Merge
(STM), a modular framework, for enhancing LLM-
based dense retrievers along three axes: synthetic
hard negatives, retriever prompt optimization, and
model merging. We focus on biomedical retrieval
while maintaining performance on general do-
mains.
Our contributions are as follows:
•We present the first systematic evaluation
of twomodel-merging techniques for LM-
based retrievers, demonstrating significant
gains over fine-tuning, see Figure 2.
•We conduct a systematic study on two under-
explored axes for retriever models:synthetic
hard negatives, andprompt optimization.
• We achievebetter results with less data: no
pre-training (i.e.. 11.4M to 1.4M pairs), merg-
ing 3 experts out of 4 (i.e., -29% of pairs), and
fine-tuning on less than 10% of the pairs.
Qwen 0.6B Gemma 2B Phi4 3.8B0.40.50.60.7Avg NDCG@10
STM
FT0.616
0.5990.622
0.6000.646
0.621Figure 2: Performance comparison of STM Merged
Models versus models fine-tuned on the combined
datasets of all merged experts, across three base models,
using the average NDCG@10 metric across all datasets.
LLM2Vec BMRetriever STM(Ours)
Attention MaskBidir. Causal Bidir.
PoolingAverage EOS EOS
Training SetupLoRA LoRA LoRA
Training RecipeMNTP +
SimSCEPT + FTFT +
Merging
NegativesSimSCESampled
Top-kSynthetic
Dataset size1.5M 11.4M 1.4M
DomainGeneral BiomedicalGeneral &
Biomedical
Table 1: Comparison of attributes between previous
methods (LLM2Vec, BMRetriever) and ours. LLM2Vec
is a multi-task contrastive embedding model, and BM-
Retriever a biomedical dense retriever pretrained with
pseudo-queries (PT) and finetuned on a mix of real and
synthetic labeled data with mined hard negatives (FT).
•We release three types of artifacts:source
code, improved retriever fine-tuningdataset,
and STMmodel(s).1
2 Related Works
2.1 Retrievers from Decoder-Only Models
E5 (Wang et al., 2024b) first demonstrated state-
of-the-art performances on the MTEB benchmark
(Muennighoff et al., 2022) from fine-tuning Mistral
7B (Jiang et al., 2023), a decoder-only model, on
real and synthetic paired retrieval data. LLM2Vec
(BehnamGhader et al., 2024) showed that using
bidirectional attention masking with average pool-
ing on LLMs, and training them withmasked next
token predictionand SimCSE (Gao et al., 2021)
led to improvements on retrieval tasks. NV-Embed
(Lee et al., 2025) introduced thelatent attention
1Code, data and model(s) will be released upon acceptance.
2

pooling layer during training with positive-aware
hard negatives and non-retrieval task data to reach
stronger performances. BMRetriever (Xu et al.,
2024) leveraged a vast contrastive pre-training, fol-
lowed by instruction tuning on a mix of real and
synthetic datasets, yielding strong dense retrieval
models for the biomedical domain.
2.2 Hard Negative Mining
Hard negative mining has become a crucial de-
sign choice for training modern dense retrievers.
Classical negative mining schemes (Xiong et al.,
2021; Zhan et al., 2021) showed that retrieving
top-ranked passages during training can accelerate
contrastive learning. Despite its limitation to ex-
act word overlap, BM25 (Robertson and Zaragoza,
2009) is still widely used to surface hard negatives
(Karpukhin et al., 2020; Zhan et al., 2021). NV-
Retriever (Moreira et al., 2024) revisits this space
withpositive-awarehard negative mining, which
boosts the performance on the MTEB benchmark.
Beyond mined hard negatives, a small but grow-
ing line of work starts to exploregeneratedhard
negatives using LLMs. SyNeg (Li et al., 2024)
uses an LLM with a multi-attribute self-reflection
prompting strategy to synthesize hard negatives and
combines them with retrieved negatives in a hybrid
sampling scheme. Their ablation study shows that
gains only arise from the hybrid method. In con-
trast, we show that only prompting a top-tier LLM
to generate hard negatives yields substantial gains.
2.3 Model Merging
Linear-mode connectivity (Frankle et al., 2020;
Mirzadeh et al.) established that independently
trained solutions can be connected by low-loss
paths in parameter space, motivating model combi-
nation methods based on interpolation. Building on
this insight, Model Soup (Wortsman et al., 2022)
showed that averaging multiple training check-
points can outperform selecting a single one. These
ideas naturally extend from combining checkpoints
of a single model to merging distinct expert mod-
els:task arithmetic(Ilharco et al., 2023) formulates
merging as adding and subtracting task-specific
deltas from a base model, while methods such as
Ties-merging (Yadav et al., 2023) and DARE (Yu
et al., 2024) explicitly tackle parameter interfer-
ence when merging multiple experts. Recent work
further shows that such parameter-level merging
can be competitive with data-mixing strategies (Ah-
madian et al., 2024; Na et al., 2024).Authors (Labrak et al., 2024; Corbeil et al., 2025)
employed merging to build robust medical LLMs.
EmbeddingGemma (Vera et al., 2025) and Qwen3
Embedding (Zhang et al., 2025) exploit merging in
their recipe without studying its impact.
2.4 Prompt Optimization
Prior work has extensively studied automatic
prompt optimization for LLMs (Ramnath et al.,
2025). Automatic Prompt Engineer (APE) success-
fully performs black-box optimization over gener-
ated candidate prompts from a handwritten seed
prompt (Zhou et al., 2022). PromptWizard (Agar-
wal et al., 2025) and GEPA (Agrawal et al., 2025)
extend this line of work by coordinating multiple
agents or applying reflective feedback, respectively.
Although Promptriever (Weller et al., 2025)
shows that prompting can substantially affect em-
bedding quality, systematic studies of prompt opti-
mization specifically for retrievers remain limited.
3 Methods
3.1 Synthetic Data Utilization
We leverage LLMs to synthetically augment exist-
ing datasets along two complementary dimensions:
(1) generating synthetic hard negatives, and (2) op-
timizing retrieval prompts.
3.1.1 Hard Negative Generation
Training dense retrievers with contrastive objec-
tives requires negatives that are both topically re-
lated and semantically distinct from the positives.
Existing hard negative mining strategies frequently
struggle to balance informativeness and correct-
ness, often introducing either trivial negatives or
false negatives that are actually relevant (Moreira
et al., 2024).
To alleviate this issue, we employ GPT-4.1 to
generate synthetic hard negatives. Given a query
q, a positive passage p+, and an existing mined
negative p−, we prompt the LLM to generate a
new negative passage ˜p−that remains lexically and
topically aligned with qwhile being semantically
irrelevant or contradictory. The prompt provides
the full context (q, p+, p−)and explicitly instructs
the model to preserve surface-level similarity while
altering semantic intent. The exact prompt template
is provided in Appendix B.1.
3.1.2 Prompt Optimization
Prompting plays a critical role in decoder-only
embedding models (Moreira et al., 2024), as the
3

prompt directly conditions the resulting representa-
tion space. To investigate this effect systematically,
we first use the DSpy framework (Khattab et al.,
2023) to apply GEPA for automatically optimizing
retrieval prompts. Starting from an initial prompt
π, GEPA iteratively proposes refined prompts π′
aimed at improving downstream retrieval perfor-
mance on a held-out validation set. We employ
two instances of LLaMA-70B for this: an fp8-
quantized model for prompt generation and an fp16
model for reflective evaluation. The full GEPA
configuration details are reported in Table 8 of Ap-
pendix A.
Second, we examine the impact of randomly
sampled retrieval prompts. Using an fp8-quantized
LLaMA-70B, we generate sets of 10, 20, 50, and
100 generic retrieval prompts, which are randomly
assigned to queries during fine-tuning. Represen-
tative examples of both optimized and randomly
generated prompts are provided in Appendix B.2.
In all prompt-based settings, prompts are
prepended to queries during fine-tuning and ap-
plied consistently at inference time. Full templates
and example LLM-generated prompts are provided
in Appendix B.2.1.
3.2 Instruction Fine-Tuning
We fine-tune decoder-only backbone models using
a contrastive learning objective to obtain dense re-
trievers. We adopt the InfoNCE loss (Henderson
et al., 2017), which encourages each query em-
bedding to be closer to its corresponding positive
passage than to all other passages in the batch and
to any provided hard negative passage.
Formally, given a batch of Ntriplets
{(qi, p+
i, p−
i)}N
i=1, the loss term for a given
queryq iis defined as:
Li=−logesim(q i,p+
i)
esim(q i,p−
i)+PN
j=1esim(q i,p+
j),
where sim(·,·) denotes cosine similarity be-
tween embeddings, and the denominator includes
two terms: the first one uses a hard negative p−
i,
and the second one employs other passages p+
jfor
j̸=i as in-batch negatives. We average over the
loss terms in the batch to obtain the total loss.
3.2.1 Expert Model Fine-Tuning
To enable modular specialization, we fine-tune mul-
tiple expert retrievers on different, coherent subsetsof the BMRetriever fine-tuning dataset (Xu et al.,
2024):medical synthetic,medical real,natural
language understanding(NLU), andsearch.
Themedical realsubset includes sentence-level
biomedical inference and similarity datasets such
as MedNLI (Shivade, 2017) and Medical Ques-
tion Pairs (McCreery et al., 2020), as well as
passage-level biomedical QA benchmarks includ-
ing MEDIQA (Ben Abacha et al., 2019), medi-
cal StackExchange QA (Team, 2021), and med-
ical dialogue data (Li et al., 2023). Themedi-
cal syntheticsubset consists of LLM-generated
biomedical retrieval pairs. To improve general-
domain relevance modeling, the dataset further in-
corporates NLU benchmarks such as Natural Ques-
tions (Kwiatkowski et al., 2019), FEVER (Thorne
et al., 2018a), ELI5 (Fan et al., 2019), SNLI (Bow-
man et al., 2015), and the MS MARCO passage
ranking dataset (Bajaj et al., 2016).
We train four experts per backbone model, each
emphasizing a distinct data composition or train-
ing configuration (e.g., synthetic hard negatives,
prompt optimization).
3.3 Model Merging
Model merging aims to combine multiple expert
retrievers into a single model that inherits com-
plementary strengths without additional training.
Given a set of expert models {Mk}K
k=1with pa-
rameters {θk}, merging methods compute a unified
model ˆMby operating directly in parameter space.
In the simplest case, linear merging (Wortsman
et al., 2022) is defined as:
θˆM=KX
k=1αkθk,(1)
where αkare theweightcoefficients with 0≤
αk≤1.
The task arithmetic approach (Ilharco et al.,
2023) relies instead on a linear combination of task
vectors τk=θk−θB, which is the delta of param-
eters between the kthexpert and the parameters of
the base modelθ B. The merged model becomes
θˆM=θB+KX
k=1αkτk,(2)
Ties merging (Yadav et al., 2023) also leverages
task vectors τkwith two strategies to mitigate the
task-interferencephenomenon: keeping only high-
magnitude parameter changes by introducing a sec-
ond parameter named density δk∈[0,1] , and the
4

sign agreement algorithm which is a majority vote
on the signs across allτ k.
4 Experiments
4.1 Datasets
For continual pre-training experiments, we follow
the BMRetriever setup (Xu et al., 2024) in which
they employ large-scale unlabeled biomedical and
scientific corpora. For fine-tuning, we also use their
fine-tuning data mixture. We separate it into four
coherent subsets as shown in Table 2.
Medical General
SplitMed-Synth Med-Real Search NLU
#Pairs431,000 306,000 438,000 251,000
Table 2: Pair counts for our custom splits of the BMRe-
triever fine-tuning dataset, comprising four splits: two
in the medical domain and two in the general domain.
4.2 Training Setup
We experiment with three decoder-only backbone
architectures of varying sizes; Qwen3 (Yang et al.,
2025), Gemma (Team et al., 2024), and Phi-4 (Ab-
din et al., 2024); the configurations are summarized
in Table 3.
Qwen3 Gemma Phi4 mini instruct
Parameters 0.6B 2B 3.8B
Dimensions 1,024 2,048 3,072
Table 3: Backbone model used for expert training.Pa-
rametersindicate the total model size, andDimensions
refers to the hidden size.
Following prior work (BehnamGhader et al.,
2024; Lee et al., 2025), we disable causal attention
masking during fine-tuning to enable bidirectional
attention. In preliminary experiments, we consider
both EOS-token pooling and mean pooling strate-
gies. However, we observe that the former yields
slightly stronger performance. Therefore, we adopt
EOS pooling for all reported results.
During fine-tuning, retrieval prompts are
prepended to each query, and models are trained
using the InfoNCE loss. We fine-tune all mod-
els using LoRA adapters (Hu et al., 2022) ap-
plied to all linear layers following prior works
(BehnamGhader et al., 2024; Lee et al., 2025; Xu
et al., 2024). Hyperparameters, including learningrate, batch size, number of steps, and LoRA con-
figuration, are summarized in Table 9 of Appendix
A.
We conduct fine-tuning under several configura-
tions. We begin with standard fine-tuning on the
base dataset. We then fine-tune models on datasets
augmented with synthetic hard negatives, followed
by datasets generated using optimized prompts. Fi-
nally, we evaluate a configuration that combines
synthetic hard negatives with the best optimized
prompts.
4.3 Model Merging
We perform model merging using MergeKit (God-
dard et al., 2024), which supports parameter-space
merging for HuggingFace-compatible models. We
evaluate two merging strategies: linear interpola-
tion (Wortsman et al., 2022) and Ties merging (Ya-
dav et al., 2023).
We select the best merged models following
a grid search approach. For linear merging, we
sweep weight coefficients α∈ {0,0.1, . . . ,0.9} .
For Ties merging, we sweep the weight coefficients
over the same range, and vary the density parameter
overρ∈ {0.1,0.2, . . . ,0.9}.
All merged models are evaluated without fur-
ther training on development sets from four BEIR
benchmark datasets (Thakur et al., 2021): NFCor-
pus (Boteva et al., 2016), FiQA-2018 (Thakur et al.,
2021), Quora (DataCanary et al., 2017), and DB-
Pedia (Hasibi et al., 2017). We use the officialdev
splits and evaluate performance using NDCG@10
and Recall@10 metrics. To enable scalable evalua-
tion across numerous merged models, we evaluate
on a sampled subset of queries and documents for
each dataset.
4.4 Baselines
We compare our models against a diverse set of
retrieval baselines, including BM25 (Lù, 2024),
Contriever(Izacard et al., 2021), E5-v2 (Wang
et al., 2022), GTR (Ni et al., 2021), LLM2Vec
3B (BehnamGhader et al., 2024), and BMRetriever
2B (Xu et al., 2024). Baselines are selected to
match our models in parameter scale or architec-
tural family.
4.5 Evaluation
We evaluate retrieval performance on the En-
glish medical subset of the Medical MTEB bench-
mark (Muennighoff et al., 2022; Enevoldsen et al.,
2025), which includes TREC-COVID (Roberts
5

Qwen3 0.6B Gemma 2B Phi4 3.8B
VariantMed-
SynthMed-
RealSearch NLU AvgMed-
SynthMed-
RealSearch NLU AvgMed-
SynthMed-
RealSearch NLU Avg
FT 0.583 0.568 0.506 0.546 0.551 0.5810.5910.503 0.520 0.549 0.611 0.6110.497 0.585 0.576
FTSHN 0.592 0.583 0.557 0.563 0.574 0.598 0.575 0.508 0.559 0.560 0.605 0.600 0.591 0.566 0.590
FTPO 0.5940.568 0.566 0.583 0.578 0.5990.5690.567 0.613 0.587 0.6220.5740.614 0.619 0.607
FTSHN+PO 0.5860.587 0.5710.555 0.575 0.588 0.549 0.511 0.569 0.554 0.607 0.608 0.599 0.575 0.597
Table 4: NDCG@10 scores of experts averaged over 12 MTEB tasks. Each expert is trained on one of the four
subsets of the BMRetriever fine-tuning dataset. FT refers to fine-tuning on the corresponding data subset. SHN
denotes fine-tuning with synthetic hard negatives, PO applies the best-performing prompt optimization per model
family, and SHN+PO combines both. Bold and underlined entries indicate the best and second-best performance
within each expert column.
et al., 2021), SciFact (Cohan et al., 2020a), NF-
Corpus (Boteva et al., 2016), Cure (Athar Sheikh
et al., 2025), PublicHealthQA (Xing Han Lu, 2024),
and MedicalQA (Asma and Dina, 2019). To
assess general-domain generalization, we addi-
tionally evaluate on five general-domain MTEB
datasets, including FiQA (Thakur et al., 2021), Ar-
guAna (Wachsmuth et al., 2018), SciDocs (Co-
han et al., 2020b), and two NanoMTEB subsets
(FEVER (Thorne et al., 2018b) and Quora (Data-
Canary et al., 2017)).
We use the official MTEB evaluation pipeline
(Muennighoff et al., 2022), and report nDCG@10
as the evaluation metric. We average the scores
across three possible splits: medical subset, general
subset, and all datasets. At evaluation time, we
use the same retrieval prompt for all the models
finetuned with instructions.
5 Results and Analysis
5.1 Pre-training
Qwen3 0.6B Gemma 2B Phi4 3.8B0.00.10.20.30.40.50.60.7Avg NDCG@100.495 0.4980.5130.595 0.588 0.597 0.599 0.6000.621PT (10M) PT+FT (11.4M) FT (1.4M)
Figure 3: Performance averages of three base models
pre-trained (PT) and/or fine-tuned (FT) on the BMRe-
triever datasets with 10M and 1.4M samples, respec-
tively.
We displayed in Figure 3 results comparing pre-
training (PT) and fine-tuning (FT) setups. We ob-
serve that pretraining on 10M unlabeled pairs un-
derperforms models fine-tuned only on 1.4M pairs,despite BMRetriever (Xu et al., 2024) previously
benefitting from a pretraining phase. We therefore
drop the pretraining step and use fine-tuning only
in the following experiments.
5.2 Fine-Tuning Experts with Synthetic Data
5.2.1 Prompt Optimization
Method Qwen3 0.6B Gemma 2B Phi4 3.8B
FT-all 0.599 0.600 0.621
FT-all GP10 0.581†0.557 0.604
FT-all GP20 0.571 0.556 0.603
FT-all GP50 0.575 0.560 0.604†
FT-all GP100 0.580 0.555 0.604
FT-all GEPA−l 0.559 0.571 0.592
FT-all GEPA−m 0.563 0.577†0.595
FT-all GEPA−h 0.560 0.568 0.597
Med-Real 0.568 0.591 0.611
Med-Real PO 0.568 0.569 0.574
Med-Synth 0.583 0.581 0.611
Med-Synth PO 0.594 0.599 0.622
Search 0.506 0.503 0.497
Search PO 0.566 0.567 0.614
NLU 0.546 0.520 0.585
NLU PO 0.583 0.613 0.619
STM Ties 0.615 0.619 0.643
STM Linear 0.616 0.622 0.646
†Highest-performing prompt optimization for this base model.
Table 5: Prompt Optimization results across models
(average NDCG@10 over 12 MTEB tasks). Bold and
underlined entries indicate the best and second-best per-
formance within each backbone group. See full Table
10 in Appendix C.
We displayed results for two prompt optimiza-
tion techniques (generic prompts 10/20/50/100 and
GEPA light/medium/heavy) in Table 5. While fine-
tuning on all the dataset is surpassing optimized
FT-all variants, we observe substantial improve-
ments when applying the top-performing technique
per model architecture (noted by†in the first sec-
tion of the table) at the expert level (noted by the
6

POin the second section) of the table. The gains
are more considerable for the non-medical experts,
while the Med-Real expert without prompt opti-
mization remains superior. We carry over the best
prompt optimization technique per model in the
next section.
5.2.2 Expert Optimization
In Table 4, both SHN and PO consistently improve
retrieval performance compared to standard fine-
tuning when averaged over all experts. PO yields
the strongest overall gains — average improvement
of 5-7% over FT. This suggests that prompt-level
adaptations are generally effective strategy across
heterogeneous retrieval settings.
However, we observe that combining SHN and
PO does not reliably outperform SHN alone. In
fact, except for the Med-Real and Search experts
within the Qwen family, the SHN+PO variant con-
sistently ranks below the PO-only approach, indi-
cating that the benefits of SHN and PO are not
additive and may partially interfere.
Looking at individual experts, the largest rel-
ative gains are observed for the Search expert,
particularly for Phi-4 Mini, where PO achieves
a +23.5% relative improvement over standard fine-
tuning. The second-largest improvement is seen
for the NLU expert of the Gemma family, with
a +17.9% relative gain. More generally, PO con-
sistently delivers larger relative improvements for
general domain experts (Search & NLU) than for
medical domain experts.
In contrast, SHN tends to degrade performance
for medical experts as model size increases, most
notably for Med-Real retrieval. This trend suggests
that high-quality medical training data may already
contain sufficiently challenging negatives, which
larger models are better able to exploit.
Overall, while the results are not consistent,
prompt optimization emerges as the more robust
and effective method compared to SHN. SHN can
provide gains in selected settings, but its impact
is model- and task-dependent, and its combination
with PO rarely yields additional benefits.
5.3 Model Merging
As summarized in Table 6, linear interpolation con-
sistently yields the strongest performance when
combining the four individual experts across all
model families. Linear merging slightly but con-
sistently outperforms Ties merging, indicating that
simple weighted interpolation is sufficient to effec-ModelAvg
MedicalAvg
GeneralAvg
All
Qwen3 0.6B
Med-Real SHN+PO 0.613 0.551 0.587
Med-Synth PO 0.623 0.553 0.594
Search SHN+PO 0.597 0.535 0.571
NLU PO 0.606 0.550 0.583
FT-all 0.633 0.551 0.599
FT-all SHN 0.632 0.555 0.600
STM Qwen3−Ties 0.637 0.585 0.615
STM Qwen3−Linear 0.638 0.585 0.616
Gemma 2B
Med-Real 0.625 0.542 0.591
Med-Synth PO 0.625 0.564 0.599
Search PO 0.577 0.554 0.567
NLU PO 0.647 0.566 0.613
FT-all 0.638 0.548 0.600
FT-all SHN 0.637 0.561 0.605
STM Gemma−Ties 0.651 0.576 0.619
STM Gemma−Linear 0.654 0.577 0.622
Phi4 3.8B
Med-Real†0.643 0.567 0.611
Med-Synth PO 0.654 0.577 0.622
Search PO 0.636 0.583 0.614
NLU PO 0.647 0.580 0.619
FT-all 0.655 0.573 0.621
FT-all SHN 0.661 0.585 0.629
STM Phi4−Ties 0.669 0.6060.643
STM Phi4−Linear 0.6770.603 0.646
†For the STM Linear Merge, the PO variation was used.
Table 6: Best-performing experts and their STM-merged
results (average NDCG@10 across 12 MTEB tasks).
FT-all SHN indicates the model finetuned on the full
dataset along with synthetic hard negatives. Bold indi-
cates the best average, underline the second best within
each backbone group.
tively integrate complementary expert representa-
tions. Overall, merged models uniformly outper-
form fully fine-tuned counterparts across all back-
bones as shown in Figure 2 as well. We note that
these gains are consistent across model sizes.
When compared against the strongest individ-
ual expert, merged models also achieve superior
performance. In all three model families, the linear-
merged STM surpasses the best-performing single
expert, confirming that merging captures comple-
mentary strengths across domain-specialized ex-
perts rather than amplifying a single dominant ex-
pert.
5.4 Comparison with Prior Retrievers
We further evaluate the best merged STMs along
baselines from the literature on our targeted 12
7

Model SizeAvg
MedAvg
GeneralAvg
All
BM25 - 0.532 0.515 0.525
Contriever 150M 0.508 0.533 0.519
E5Large V2335M 0.654 0.576 0.622
GTRT5 XL1.2B 0.581 0.586 0.583
BMRetriever 2B 0.645 0.560 0.609
LLM2Vec 3B 0.635 0.597 0.619
STM Qwen3 0.6B 0.638 0.585 0.616
STM Gemma 2B 0.654 0.577 0.622
STM Phi4 3.8B0.677 0.603 0.646
Table 7: Summary of retrieval performance (averages)
against base lines across medical and general domains.
Bold indicates the best average, underline the second
best. Full results are reported in Appendix C.
datasets from MTEB. As shown in Table 7,
STM Phi4−Linear achieves the strongest perfor-
mance across both medical and general tasks, out-
performing all baselines. In particular, it surpasses
BMRetriever 2BandLLM2V ec , demonstrating
that expert merging scales effectively to multi-
domain retrieval and remains competitive with
state-of-the-art retrievers trained on large and di-
verse corpora.
Notably, STM Gemma−Linear also delivers
strong performance despite its smaller model size.
It consistently outperforms BMRetriever 2B,
which shares the same base model. These results
highlight the efficiency of the proposed approach
without relying on larger backbones or additional
pre-training.
5.5 Merging Coefficients
We provide the mergingweightcoefficients2in
Figure 5 of Appendix C for each model. We notice
similar coefficients for Qwen3 and Phi4 in contrast
to the ones used for Gemma. Qwen3 and Phi4
did not use the Search expert at all to build their
respective STM final models, and both utilize with
higher amplitudes the medical experts along with
the NLU expert at a weight of 0.5. For Gemma, the
weight coefficients tend to be lower than 0.5 with
no use of Med-Real or Med-Synth experts for linear
merging or Ties, respectively. Overall, Ties-merged
optimal models have lower coefficients compared
to the linear merging ones, but coefficients of both
methods are correlated.
2We ignore the density coefficients for the Ties method in
this analysis since the weight coefficients modulate directly
the final amplitude of that expert in the merged models.From analyzing the weight coefficients in terms
of data ablation, we note that generally all optimal
configurations for each model remove one expert.
Thus, we infer that removing one of the expert
could reduce the overall data budget from 18% for
the NLU subset up to 29% for the Search subset
out of the 1.4M available pairs.
5.6 Training Data Size Considerations
10K 100K 1.4M
Dataset Size0.00.10.20.30.40.50.60.7Avg NDCG@10
0.3080.603 0.599 0.3910.608 0.600
0.0200.633 0.621
Qwen3 0.6B
Gemma 2B
Phi4 3.8B
Figure 4: Performance averages across 3 runs of three
base models fine-tuned on three different sample sizes
of the BMRetriever dataset. Standard deviations are not
displayed since they are below 0.01.
Ablations on dataset sizes visualized in Figure 4
reveal clear patterns. Models’ performances aver-
aged across 3 runs saturate at around 100K samples
outperforming those trained on the full 1.4M sam-
ples across all three base models. Therefore, cu-
rated high-quality data can be more effective than
large-scale datasets; in line with the trends of the
pretraining and merging coefficient results in sec-
tions 5.1 and 5.5, respectively. While experiments
of previous sections did not leverage this finding,
future works could further explore this direction.
6 Conclusion
We presented Synthesize-Train-Merge (STM), a
modular framework for adapting decoder-only
LLMs into effective dense retrievers for domain-
specific tasks. By combining synthetic hard neg-
atives, retrieval prompt optimization, and model
merging, STM improves task-specific experts by
up to 23.5% and produce unified models that out-
perform both individual experts and baselines fine-
tuned on the experts datasets combined. Our re-
sults show that careful dataset selection and modu-
lar merging can yield strong retrieval performance
without extensive pre-training or larger backbones.
These findings suggest a scalable, efficient path for
8

adapting LLMs to specialized retrieval tasks while
maintaining general-domain generalization.
Limitations
Despite strong empirical results, our study has sev-
eral limitations. First, we only explore two merg-
ing strategies (linear interpolation and Ties); more
adaptive or task-aware merging approaches could
provide further gains but are beyond the scope of
this work. Second, our synthetic hard negative
generation and prompt optimization rely on large
LLMs, adding computational cost and potential
sensitivity to the choice of generator model. We do
not evaluate robustness across different LLMs or
prompt variants.
Acknowledgments
Three AI assistants were utilized to accomplish
parts of this work for writing and coding purposes.
ChatGPT 5was used for proofreading.Cursorwas
leveraged while coding the source code, specifi-
cally to draft routine functions and code blocks.
GitHub Copilotwas employed while coding fig-
ures. All outputs were thoroughly edited, revised,
fact checked, and/or debugged.
References
Marah Abdin, Jyoti Aneja, Harkirat Behl, Sébastien
Bubeck, Ronen Eldan, Suriya Gunasekar, Michael
Harrison, Russell J Hewett, Mojan Javaheripi, Piero
Kauffmann, and 1 others. 2024. Phi-4 technical re-
port.arXiv preprint arXiv:2412.08905.
Mohamed Abo El-Enen, Sally Saad, and Taymoor
Nazmy. 2025. A survey on retrieval-augmentation
generation (rag) models for healthcare applications.
Neural Computing and Applications, 37(33):28191–
28267.
Eshaan Agarwal, Raghav Magazine, Joykirat Singh,
Vivek Dani, Tanuja Ganu, and Akshay Nambi. 2025.
Promptwizard: Optimizing prompts via task-aware,
feedback-driven self-evolution. InFindings of the As-
sociation for Computational Linguistics: ACL 2025,
pages 19974–20003.
Lakshya A Agrawal, Shangyin Tan, Dilara Soylu,
Noah Ziems, Rishi Khare, Krista Opsahl-Ong, Ar-
nav Singhvi, Herumb Shandilya, Michael J Ryan,
Meng Jiang, and 1 others. 2025. Gepa: Reflec-
tive prompt evolution can outperform reinforcement
learning.arXiv preprint arXiv:2507.19457.
Arash Ahmadian, Seraphina Goldfarb-Tarrant, Beyza
Ermis, Marzieh Fadaee, Sara Hooker, and 1 others.
2024. Mix data or merge models? optimizing fordiverse multi-task learning. InSafe Generative AI
Workshop.
Ben Abacha Asma and Demner-Fushman Dina. 2019. A
question-entailment approach to question answering.
BMC Bioinform., 20(1):511:1–511:23.
Nadia Athar Sheikh, Daniel Buades Marcos, Anne-
Laure Jousse, Akintunde Oladipo, Olivier Rousseau,
and Jimmy Lin. 2025. Cure: A dataset for clinical
understanding &amp; retrieval evaluation. InPro-
ceedings of the 31st ACM SIGKDD Conference on
Knowledge Discovery and Data Mining V .2, KDD
’25, page 5270–5277. ACM.
Orlando Ayala and Patrice Bechard. 2024. Reduc-
ing hallucination in structured outputs via retrieval-
augmented generation. InProceedings of the 2024
Conference of the North American Chapter of the
Association for Computational Linguistics: Human
Language Technologies (Volume 6: Industry Track),
pages 228–238, Mexico City, Mexico. Association
for Computational Linguistics.
Payal Bajaj, Daniel Campos, Nick Craswell, Li Deng,
Jianfeng Gao, Xiaodong Liu, Rangan Majumder, An-
drew McNamara, Bhaskar Mitra, Tri Nguyen, and 1
others. 2016. MS MARCO: A human generated ma-
chine reading comprehension dataset. arXiv preprint
arXiv:1611.09268.
Mariam Barry, Gaetan Caillaut, Pierre Halftermeyer,
Raheel Qader, Mehdi Mouayad, Fabrice Le Deit,
Dimitri Cariolaro, and Joseph Gesnouin. 2025.
GraphRAG: Leveraging graph-based efficiency to
minimize hallucinations in LLM-driven RAG for fi-
nance data. InProceedings of the Workshop on Gen-
erative AI and Knowledge Graphs (GenAIK), pages
54–65, Abu Dhabi, UAE. International Committee
on Computational Linguistics.
Parishad BehnamGhader, Vaibhav Adlakha, Marius
Mosbach, Dzmitry Bahdanau, Nicolas Chapados, and
Siva Reddy. 2024. Llm2vec: Large language models
are secretly powerful text encoders. InFirst Confer-
ence on Language Modeling.
Asma Ben Abacha, Chaitanya Shivade, and Dina
Demner-Fushman. 2019. Overview of the MEDIQA
2019 shared task on textual inference, question entail-
ment and question answering. InProceedings of the
18th BioNLP Workshop and Shared Task, pages 370–
379, Florence, Italy. Association for Computational
Linguistics.
Vera Boteva, Demian Gholipour, Artem Sokolov, and
Stefan Riezler. 2016. A full-text learning to rank
dataset for medical information retrieval.
Samuel R. Bowman, Gabor Angeli, Christopher Potts,
and Christopher D. Manning. 2015. A large anno-
tated corpus for learning natural language inference.
InProceedings of the 2015 Conference on Empiri-
cal Methods in Natural Language Processing, pages
632–642, Lisbon, Portugal. Association for Compu-
tational Linguistics.
9

Arman Cohan, Sergey Feldman, Iz Beltagy, Doug
Downey, and Daniel S. Weld. 2020a. Specter:
Document-level representation learning using
citation-informed transformers. InACL.
Arman Cohan, Sergey Feldman, Iz Beltagy, Doug
Downey, and Daniel S. Weld. 2020b. Specter:
Document-level representation learning using
citation-informed transformers. InACL.
Jean-Philippe Corbeil, Amin Dada, Jean-Michel At-
tendu, Asma Ben Abacha, Alessandro Sordoni, Lu-
cas Caccia, Francois Beaulieu, Thomas Lin, Jens
Kleesiek, and Paul V ozila. 2025. A modular ap-
proach for clinical SLMs driven by synthetic data
with pre-instruction tuning, model merging, and
clinical-tasks alignment. InProceedings of the 63rd
Annual Meeting of the Association for Computational
Linguistics (Volume 1: Long Papers), pages 19352–
19374, Vienna, Austria. Association for Computa-
tional Linguistics.
DataCanary, Lili Jiang hilfialkaff, Meg Risdal, Nikhil
Dandekar, and tomtung. 2017. Quora question pairs.
Kenneth Enevoldsen, Isaac Chung, Imene Kerboua,
Márton Kardos, Ashwin Mathur, David Stap,
Jay Gala, Wissam Siblini, Dominik Krzemi ´nski,
Genta Indra Winata, Saba Sturua, Saiteja Utpala,
Mathieu Ciancone, Marion Schaeffer, Gabriel Se-
queira, Diganta Misra, Shreeya Dhakal, Jonathan
Rystrøm, Roman Solomatin, and 67 others. 2025.
Mmteb: Massive multilingual text embedding bench-
mark.arXiv preprint arXiv:2502.13595.
Angela Fan, Yacine Jernite, Ethan Perez, David Grang-
ier, Jason Weston, and Michael Auli. 2019. ELI5:
Long form question answering. InProceedings of
the 57th Annual Meeting of the Association for Com-
putational Linguistics, pages 3558–3567, Florence,
Italy. Association for Computational Linguistics.
Wenqi Fan, Yujuan Ding, Liangbo Ning, Shijie Wang,
Hengyun Li, Dawei Yin, Tat-Seng Chua, and Qing
Li. 2024. A survey on rag meeting llms: Towards
retrieval-augmented large language models. InPro-
ceedings of the 30th ACM SIGKDD conference on
knowledge discovery and data mining, pages 6491–
6501.
Jonathan Frankle, Gintare Karolina Dziugaite, Daniel
Roy, and Michael Carbin. 2020. Linear mode con-
nectivity and the lottery ticket hypothesis. InInter-
national Conference on Machine Learning, pages
3259–3269. PMLR.
Tianyu Gao, Xingcheng Yao, and Danqi Chen. 2021.
Simcse: Simple contrastive learning of sentence em-
beddings. InProceedings of the 2021 Conference on
Empirical Methods in Natural Language Processing,
pages 6894–6910.
Charles Goddard, Shamane Siriwardhana, Malikeh
Ehghaghi, Luke Meyers, Vladimir Karpukhin, Brian
Benedict, Mark McQuade, and Jacob Solawetz. 2024.Arcee’s mergekit: A toolkit for merging large lan-
guage models. InProceedings of the 2024 Confer-
ence on Empirical Methods in Natural Language
Processing: Industry Track, pages 477–485.
Faegheh Hasibi, Fedor Nikolaev, Chenyan Xiong, Krisz-
tian Balog, Svein Erik Bratsberg, Alexander Kotov,
and Jamie Callan. 2017. Dbpedia-entity v2: A test
collection for entity search. InProceedings of the
40th International ACM SIGIR Conference on Re-
search and Development in Information Retrieval,
SIGIR ’17, pages 1265–1268. ACM.
Matthew Henderson, Rami Al-Rfou, Brian Strope, Yun-
Hsuan Sung, László Lukács, Ruiqi Guo, Sanjiv Ku-
mar, Balint Miklos, and Ray Kurzweil. 2017. Effi-
cient natural language response suggestion for smart
reply.arXiv preprint arXiv:1705.00652.
Edward J Hu, Phillip Wallis, Zeyuan Allen-Zhu,
Yuanzhi Li, Shean Wang, Lu Wang, Weizhu Chen,
and 1 others. 2022. Lora: Low-rank adaptation of
large language models. InInternational Conference
on Learning Representations.
Gabriel Ilharco, Marco Tulio Ribeiro, Mitchell Worts-
man, Ludwig Schmidt, Hannaneh Hajishirzi, and Ali
Farhadi. 2023. Editing models with task arithmetic.
InThe Eleventh International Conference on Learn-
ing Representations.
Gautier Izacard, Mathilde Caron, Lucas Hosseini, Se-
bastian Riedel, Piotr Bojanowski, Armand Joulin,
and Edouard Grave. 2021. Unsupervised dense infor-
mation retrieval with contrastive learning.
Albert Q. Jiang, Alexandre Sablayrolles, Arthur Men-
sch, Chris Bamford, Devendra Singh Chaplot, Diego
de las Casas, Florian Bressand, Gianna Lengyel,
Guillaume Lample, Lucile Saulnier, Lélio Re-
nard Lavaud, Marie-Anne Lachaux, Pierre Stock,
Teven Le Scao, Thibaut Lavril, Thomas Wang, Timo-
thée Lacroix, and William El Sayed. 2023. Mistral
7b. ArXiv:2310.06825 [cs].
Vladimir Karpukhin, Barlas Oguz, Sewon Min, Patrick
Lewis, Ledell Wu, Sergey Edunov, Danqi Chen, and
Wen-tau Yih. 2020. Dense passage retrieval for open-
domain question answering. InProceedings of the
2020 Conference on Empirical Methods in Natural
Language Processing (EMNLP), pages 6769–6781.
Association for Computational Linguistics.
Omar Khattab, Arnav Singhvi, Paridhi Maheshwari,
Zhiyuan Zhang, Keshav Santhanam, Sri Vard-
hamanan, Saiful Haq, Ashutosh Sharma, Thomas T.
Joshi, Hanna Moazam, Heather Miller, Matei Za-
haria, and Christopher Potts. 2023. Dspy: Compiling
declarative language model calls into self-improving
pipelines.Preprint, arXiv:2310.03714.
Tom Kwiatkowski, Jennimaria Palomaki, Olivia Red-
field, Michael Collins, Ankur Parikh, Chris Alberti,
Danielle Epstein, Illia Polosukhin, Jacob Devlin, Ken-
ton Lee, Kristina Toutanova, Llion Jones, Matthew
Kelcey, Ming-Wei Chang, Andrew M. Dai, Jakob
10

Uszkoreit, Quoc Le, and Slav Petrov. 2019. Natu-
ral questions: A benchmark for question answering
research.Transactions of the Association for Compu-
tational Linguistics, 7:452–466.
Yanis Labrak, Adrien Bazoge, Emmanuel Morin, Pierre-
Antoine Gourraud, Mickaël Rouvier, and Richard
Dufour. 2024. Biomistral: A collection of open-
source pretrained large language models for medical
domains. InFindings of the Association for Compu-
tational Linguistics: ACL 2024, pages 5848–5864.
Chankyu Lee, Rajarshi Roy, Mengyao Xu, Jonathan
Raiman, Mohammad Shoeybi, Bryan Catanzaro, and
Wei Ping. 2025. Nv-embed: Improved techniques
for training llms as generalist embedding models. In
The Thirteenth International Conference on Learning
Representations.
Patrick Lewis, Ethan Perez, Aleksandra Piktus, Fabio
Petroni, Vladimir Karpukhin, Naman Goyal, Hein-
rich Küttler, Mike Lewis, Wen-tau Yih, Tim Rock-
täschel, and 1 others. 2020. Retrieval-augmented gen-
eration for knowledge-intensive nlp tasks.Advances
in neural information processing systems, 33:9459–
9474.
Xiaopeng Li, Xiangyang Li, Hao Zhang, Zhaocheng Du,
Pengyue Jia, Yichao Wang, Xiangyu Zhao, Huifeng
Guo, and Ruiming Tang. 2024. Syneg: Llm-driven
synthetic hard-negatives for dense retrieval.arXiv
preprint arXiv:2412.17250.
Yunxiang Li, Zihan Li, Kai Zhang, Ruilong Dan, Steve
Jiang, and You Zhang. 2023. Chatdoctor: A medical
chat model fine-tuned on a large language model
meta-ai (llama) using medical domain knowledge.
Cureus, 15(6).
Ziyue Li and Tianyi Zhou. 2025. Your mixture-of-
experts llm is secretly an embedding model for free.
InThe Thirteenth International Conference on Learn-
ing Representations.
Xing Han Lù. 2024. Bm25s: Orders of magnitude faster
lexical search via eager sparse scoring.Preprint,
arXiv:2407.03618.
Clara H McCreery, Namit Katariya, Anitha Kannan,
Manish Chablani, and Xavier Amatriain. 2020. Ef-
fective transfer learning for identifying similar ques-
tions: matching user questions to covid-19 faqs. In
Proceedings of the 26th ACM SIGKDD international
conference on knowledge discovery & data mining,
pages 3458–3465.
Seyed Iman Mirzadeh, Mehrdad Farajtabar, Dilan Gorur,
Razvan Pascanu, and Hassan Ghasemzadeh. Linear
mode connectivity in multitask and continual learn-
ing. InInternational Conference on Learning Repre-
sentations.
Gabriel de Souza P Moreira, Radek Osmulski, Mengyao
Xu, Ronay Ak, Benedikt Schifferer, and Even
Oldridge. 2024. Nv-retriever: Improving text em-
bedding models with effective hard-negative mining.
arXiv preprint arXiv:2407.15831.Niklas Muennighoff, Nouamane Tazi, Loïc Magne, and
Nils Reimers. 2022. Mteb: Massive text embedding
benchmark.arXiv preprint arXiv:2210.07316.
Clara Na, Ian Magnusson, Ananya Harsh Jha, Tom Sher-
borne, Emma Strubell, Jesse Dodge, and Pradeep
Dasigi. 2024. Scalable data ablation approximations
for language models through modular training and
merging. InProceedings of the 2024 Conference on
Empirical Methods in Natural Language Processing,
pages 21125–21141.
Jianmo Ni, Chen Qu, Jing Lu, Zhuyun Dai, Gustavo Her-
nandez Abrego, Ji Ma, Vincent Zhao, Yi Luan, Keith
Hall, Ming-Wei Chang, and 1 others. 2022. Large
dual encoders are generalizable retrievers. InPro-
ceedings of the 2022 Conference on Empirical Meth-
ods in Natural Language Processing, pages 9844–
9855.
Jianmo Ni, Chen Qu, Jing Lu, Zhuyun Dai, Gus-
tavo Hernández Ábrego, Ji Ma, Vincent Y . Zhao,
Yi Luan, Keith B. Hall, Ming-Wei Chang, and Yinfei
Yang. 2021. Large dual encoders are generalizable
retrievers.Preprint, arXiv:2112.07899.
Kiran Ramnath, Kang Zhou, Sheng Guan, Soumya Sm-
ruti Mishra, Xuan Qi, Zhengyuan Shen, Shuai Wang,
Sangmin Woo, Sullam Jeoung, Yawei Wang, Haozhu
Wang, Han Ding, Yuzhe Lu, Zhichao Xu, Yun Zhou,
Balasubramaniam Srinivasan, Qiaojing Yan, Yueyan
Chen, Haibo Ding, and 2 others. 2025. A systematic
survey of automatic prompt optimization techniques.
InProceedings of the 2025 Conference on Empiri-
cal Methods in Natural Language Processing, pages
33066–33098, Suzhou, China. Association for Com-
putational Linguistics.
Kirk Roberts, Tasmeer Alam, Steven Bedrick, Dina
Demner-Fushman, Kyle Lo, Ian Soboroff, Ellen
V oorhees, Lucy Lu Wang, and William R Hersh.
2021. Searching for scientific evidence in a pan-
demic: An overview of trec-covid.Preprint,
arXiv:2104.09632.
Stephen Robertson and Hugo Zaragoza. 2009. The
probabilistic relevance framework: BM25 and be-
yond.Foundations and Trends in Information Re-
trieval, 3(4):333–389.
Kunal Sawarkar, Abhilasha Mangal, and Shivam Raj
Solanki. 2024. Blended rag: Improving rag
(retriever-augmented generation) accuracy with se-
mantic search and hybrid query-based retrievers. In
2024 IEEE 7th international conference on multi-
media information processing and retrieval (MIPR),
pages 155–161. IEEE.
Rulin Shao, Rui Qiao, Varsha Kishore, Niklas Muen-
nighoff, Xi Victoria Lin, Daniela Rus, Bryan
Kian Hsiang Low, Sewon Min, Wen-tau Yih,
Pang Wei Koh, and 1 others. 2025. Reasonir: Train-
ing retrievers for reasoning tasks.arXiv preprint
arXiv:2504.20595.
11

Chaitanya Shivade. 2017. Mednli — a natural language
inference dataset for the clinical domain.
Jacob Mitchell Springer, Suhas Kotha, Daniel Fried,
Graham Neubig, and Aditi Raghunathan. 2025. Rep-
etition improves language model embeddings. In
The Thirteenth International Conference on Learn-
ing Representations.
Flax Sentence Embeddings Team. 2021. Stack ex-
change question pairs.
Gemma Team, Thomas Mesnard, Cassidy Hardin,
Robert Dadashi, Surya Bhupatiraju, Shreya Pathak,
Laurent Sifre, Morgane Rivière, Mihir Sanjay Kale,
Juliette Love, and 1 others. 2024. Gemma: Open
models based on gemini research and technology.
arXiv preprint arXiv:2403.08295.
Nandan Thakur, Nils Reimers, Andreas Rücklé, Ab-
hishek Srivastava, and Iryna Gurevych. 2021. BEIR:
A heterogeneous benchmark for zero-shot evaluation
of information retrieval models. InThirty-fifth Con-
ference on Neural Information Processing Systems
Datasets and Benchmarks Track (Round 2).
James Thorne, Andreas Vlachos, Christos
Christodoulopoulos, and Arpit Mittal. 2018a.
FEVER: a large-scale dataset for fact extraction
and VERification. InProceedings of the 2018
Conference of the North American Chapter of
the Association for Computational Linguistics:
Human Language Technologies, Volume 1 (Long
Papers), pages 809–819, New Orleans, Louisiana.
Association for Computational Linguistics.
James Thorne, Andreas Vlachos, Christos
Christodoulopoulos, and Arpit Mittal. 2018b.
FEVER: a large-scale dataset for fact extraction
and VERification. InProceedings of the 2018
Conference of the North American Chapter of
the Association for Computational Linguistics:
Human Language Technologies, Volume 1 (Long
Papers), pages 809–819, New Orleans, Louisiana.
Association for Computational Linguistics.
Henrique Schechter Vera, Sahil Dua, Biao Zhang,
Daniel Salz, Ryan Mullins, Sindhu Raghuram Pa-
nyam, Sara Smoot, Iftekhar Naim, Joe Zou, Feiyang
Chen, and 1 others. 2025. Embeddinggemma: Pow-
erful and lightweight text representations.arXiv
preprint arXiv:2509.20354.
Henning Wachsmuth, Shahbaz Syed, and Benno Stein.
2018. Retrieval of the best counterargument without
prior topic knowledge. InACL.
Liang Wang, Nan Yang, Xiaolong Huang, Binxing
Jiao, Linjun Yang, Daxin Jiang, Rangan Majumder,
and Furu Wei. 2022. Text embeddings by weakly-
supervised contrastive pre-training.arXiv preprint
arXiv:2212.03533.
Liang Wang, Nan Yang, Xiaolong Huang, Linjun Yang,
Rangan Majumder, and Furu Wei. 2024a. Improv-
ing text embeddings with large language models. InProceedings of the 62nd Annual Meeting of the As-
sociation for Computational Linguistics (Volume 1:
Long Papers), pages 11897–11916, Bangkok, Thai-
land. Association for Computational Linguistics.
Liang Wang, Nan Yang, Xiaolong Huang, Linjun Yang,
Rangan Majumder, and Furu Wei. 2024b. Improv-
ing text embeddings with large language models. In
Proceedings of the 62nd Annual Meeting of the As-
sociation for Computational Linguistics (Volume 1:
Long Papers), pages 11897–11916.
Xiaohua Wang, Zhenghua Wang, Xuan Gao, Feiran
Zhang, Yixin Wu, Zhibo Xu, Tianyuan Shi,
Zhengyuan Wang, Shizheng Li, Qi Qian, and 1 oth-
ers. 2024c. Searching for best practices in retrieval-
augmented generation. InProceedings of the 2024
Conference on Empirical Methods in Natural Lan-
guage Processing, pages 17716–17736.
Orion Weller, Benjamin Van Durme, Dawn Lawrie, Ash-
win Paranjape, Yuhao Zhang, and Jack Hessel. 2025.
Promptriever: Instruction-trained retrievers can be
prompted like language models. InThe Thirteenth
International Conference on Learning Representa-
tions.
Mitchell Wortsman, Gabriel Ilharco, Samir Ya Gadre,
Rebecca Roelofs, Raphael Gontijo-Lopes, Ari S Mor-
cos, Hongseok Namkoong, Ali Farhadi, Yair Car-
mon, Simon Kornblith, and 1 others. 2022. Model
soups: averaging weights of multiple fine-tuned mod-
els improves accuracy without increasing inference
time. InInternational conference on machine learn-
ing, pages 23965–23998. PMLR.
Xing Han Lu. 2024. publichealth-qa (revision
3b67b6b).
Guangzhi Xiong, Qiao Jin, Zhiyong Lu, and Aidong
Zhang. 2024. Benchmarking retrieval-augmented
generation for medicine. InFindings of the Associa-
tion for Computational Linguistics ACL 2024, pages
6233–6251, Bangkok, Thailand and virtual meeting.
Association for Computational Linguistics.
Lee Xiong, Chenyan Xiong, Ye Li, Kwok-Fung Tang,
Jialin Liu, Paul N. Bennett, Junaid Ahmed, and
Arnold Overwijk. 2021. Approximate nearest neigh-
bor negative contrastive learning for dense text re-
trieval. InProceedings of the 9th International Con-
ference on Learning Representations.
Ran Xu, Wenqi Shi, Yue Yu, Yuchen Zhuang, Yanqiao
Zhu, May Dongmei Wang, Joyce C Ho, Chao Zhang,
and Carl Yang. 2024. Bmretriever: Tuning large
language models as better biomedical text retrievers.
InProceedings of the 2024 Conference on Empiri-
cal Methods in Natural Language Processing, pages
22234–22254.
Prateek Yadav, Derek Tam, Leshem Choshen, Colin A
Raffel, and Mohit Bansal. 2023. Ties-merging: Re-
solving interference when merging models.Ad-
vances in Neural Information Processing Systems,
36:7093–7115.
12

An Yang, Anfeng Li, Baosong Yang, Beichen Zhang,
Binyuan Hui, Bo Zheng, Bowen Yu, Chang
Gao, Chengen Huang, Chenxu Lv, and 1 others.
2025. Qwen3 technical report.arXiv preprint
arXiv:2505.09388.
Le Yu, Bowen Yu, Haiyang Yu, Fei Huang, and Yongbin
Li. 2024. Language models are super mario: Absorb-
ing abilities from homologous models as a free lunch.
InForty-first International Conference on Machine
Learning.
Jingtao Zhan, Jiaxin Mao, Yiqun Liu, Jiafeng Guo, Min
Zhang, and Shaoping Ma. 2021. Optimizing dense
retrieval model training with hard negatives. InPro-
ceedings of the 44th International ACM SIGIR Con-
ference on Research and Development in Information
Retrieval, pages 1503–1512. ACM.
Yanzhao Zhang, Mingxin Li, Dingkun Long, Xin Zhang,
Huan Lin, Baosong Yang, Pengjun Xie, An Yang,
Dayiheng Liu, Junyang Lin, and 1 others. 2025.
Qwen3 embedding: Advancing text embedding and
reranking through foundation models.arXiv preprint
arXiv:2506.05176.
Yongchao Zhou, Andrei Ioan Muresanu, Ziwen Han,
Keiran Paster, Silviu Pitis, Harris Chan, and Jimmy
Ba. 2022. Large language models are human-level
prompt engineers. InThe eleventh international con-
ference on learning representations.
A Implementation Details
Parameter Value
LLM Configuration
Prompt Generation Model LLaMA 70B (FP8)
Reflection Model LLaMA 70B (FP16)
Temperature 0.7
Max Tokens 1000
GEPA Hyperparameters
Budget Level Auto
(Light / Medium / Heavy)
Training Examples 100 / 200 / 300
Evaluation Metric NDCG@10
Validation Examples 100 / 200 / 300
Reflection Minibatch Size 3 / 5 / 8
Number of Threads 1
Table 8: GEPA configuration details.Hyperparameter
Maximum Tokens 512
Optimizer AdamW
LR Scheduler Linear Warmup
# Warmup Steps 100
bf16 True
Learning Rate
Phi4 mini instruct (3.8B)2×10−5
Gemma (2B)1×10−5
Qwen3 (0.6B)5×10−5
# Epochs 1
LoRA Rank 16
LoRAα32
Total Batch Size
Phi4 mini instruct (3.8B) 64
Gemma (2B) 128
Qwen3 (0.6B) 256
Table 9: Fine-tuning Hyperparameters.
B Prompt Examples
B.1 Hard Negative Generation Prompts
The following prompt template was used for hard
negative mining:
Hard Negative Generation Prompt
Given this query:"{query}"
Original prompt:"{original prompt}"
Original positive document:"{positive doc}"
Original negative document: "{original
negative}"
Generate a HARD negative document that is:
1. Related to the same domain (extract domain from
the query)
2. Contains similar terminology and concepts as the
positive document
3. But is NOT relevant to answering the specific
query
4. Should be moderately challenging to distinguish
from the positive document
5. Should be a realistic document in the domain
6. Should be harder than the original negative docu-
ment but not as hard as a super hard negative
IMPORTANT: Return ONLY the document text. Do
not include any introductory text, explanations, sum-
maries, or meta-commentary. Just return the raw
document content.
B.2 Optimized Prompts
B.2.1 Generic Prompts
We generated random generic prompts using the
following template.
13

Generic Prompts Generation Prompt
You are an expert in information retrieval prompt
engineering.
Generate a creative and diverse prompt specifically
for DOCUMENT RETRIEV AL tasks.
The prompt should help a retrieval model find rele-
vant documents for any query.
Use a different style/approach from these examples:
- Direct retrieval instruction format
- Document ranking format
- Relevance scoring format
- Query-document matching format
- Information seeking format
- Context-aware retrieval format
- Domain-specific retrieval format
- Simple document finding format
Make it unique and varied. The prompt should be
effective for DOCUMENT RETRIEV AL.
Generate only the prompt text, no explanations, do
not specify any domain or document type to avoid
confusing the model:
Generic Retrieval Prompts for General and
Medical Domains
Baseline (all variations)
General Domain (MS MARCO):“Given a web search
query, retrieve relevant passages that answer the query. ”
Medical Domain (Synthetic):“Given a query, find
articles that discuss the correlation between a specific
lifestyle factor and a disease. ”
10 Generic prompts (all domains)
“Imagine you’re a curator of a vast library, tasked with
uncovering hidden gems that shed new light on a specific
topic of interest. Given the query, navigate through the
shelves of knowledge to gather a collection of documents
that not only resonate with the inquiry but also offer
diverse perspectives, insightful analysis, and thought-
provoking discussions. The goal is to assemble a compre-
hensive anthology that enriches understanding, sparks
curiosity, and fuels further exploration, ensuring that
every included document contributes a unique voice to
the chorus of knowledge on the subject at hand. ”
20 Generic prompts (all domains)
“Imagine you’re a librarian tasked with curating a person-
alized bookshelf for a curious reader. Given the topic of
interest, navigate through a vast library and handpick
a selection of texts that would spark fascinating discus-
sions, provide insightful knowledge, and resonate deeply
with the reader’s query, as if you were recommending
books to a close friend. ”
50 Generic prompts (all domains)
“Imagine you’re a librarian tasked with curating a person-
alized anthology for a curious reader; given the threads
of inquiry woven into the phrase, navigate the vast ex-
panse of written works to unearth the most enlightening
and informative texts that intricately weave together con-
cepts, ideas, and narratives, and present a collection
that not only resonates with the essence of the inquirybut also expands its boundaries, fostering a deeper un-
derstanding and sparking further exploration. ”
100 Generic prompts (all domains)
“Imagine you’re a librarian tasked with uncovering hid-
den gems in a vast archive, and someone has whispered
a cryptic clue in your ear; given this whispered clue,
what documents would you pull from the shelves to un-
ravel the mystery, and what threads of connection would
you follow to weave together a tapestry of insight and
understanding?”
B.2.2 GEPA-Optimized Retrieval Prompts
GEPA-Optimized Retrieval Prompts for
General and Medical Domains
Baseline (all budgets)
General Domain (MS MARCO):“Given a web
search query, retrieve relevant passages that answer
the query. ”
Medical Domain (Synthetic):“Given a query, find
articles that discuss the correlation between a specific
lifestyle factor and a disease. ”
GEPA-Light Budget
General Domain (MS MARCO):“Design an effi-
cient algorithm to retrieve relevant textual passages
from the web that directly answer a given search
query, considering factors such as query clarity, con-
textual relevance, and the specificity of the information
sought, and optimize the search results based on the
type of information required, such as factual, explana-
tory, or comparative analysis, to achieve high scores
in evaluation metrics like NDCG@10, MAP@10, and
Recall@10. ”
Medical Domain (Synthetic):“Retrieve recent, high-
quality Pubmed passages that directly answer the
given biomedical question, prioritizing peer-reviewed
articles from reputable journals published within the
last five years, and considering specific contexts such
as demographics or health conditions if applicable. ”
GEPA-Medium Budget
General Domain (MS MARCO):“Retrieve fac-
tual and explanatory passages from reputable online
sources that directly answer the given web search
query, ensuring the passages include specific keywords
related to the query topic, are relevant to the context
or domain of interest, and provide a clear, concise,
and relevant response that matches the desired type of
answer. ”
Medical Domain (Synthetic):“Retrieve recent, peer-
reviewed Pubmed articles or passages that directly an-
swer the given biomedical question, focusing on high-
quality studies published within the last five years, and
provide abstracts or summaries of these articles in the
search results. ”
GEPA-Heavy Budget
General Domain (MS MARCO):“Design an effi-
cient algorithm to retrieve relevant textual passages
14

from the web that directly answer a given search query,
considering factors such as query clarity, contextual
relevance, and the specificity of the information sought. ”
Medical Domain (Synthetic):“Retrieve recent, rele-
vant Pubmed passages that directly answer the given
biomedical question, prioritizing articles from rep-
utable journals published within the last five years,
and providing accurate and up-to-date information on
the topic. ”
C Detailed Results
15

Method Qwen3 0.6B Gemma 2B Phi4 3.8B
Avg
MedAvg
GenAvg
AllAvg
MedAvg
GenAvg
AllAvg
MedAvg
GenAvg
All
Med-Real 0.609 0.510 0.568 0.625 0.542 0.591 0.643 0.567 0.611
Med-Real PO 0.581 0.550 0.568 0.586 0.545 0.569 0.566 0.585 0.574
Med-Synth 0.614 0.540 0.583 0.606 0.546 0.581 0.642 0.567 0.611
Med-Synth PO 0.623 0.553 0.594 0.625 0.564 0.599 0.654 0.577 0.622
Search 0.527 0.477 0.506 0.523 0.475 0.503 0.515 0.472 0.497
Search PO 0.583 0.542 0.566 0.577 0.554 0.567 0.636 0.583 0.614
NLU 0.573 0.509 0.546 0.546 0.484 0.520 0.628 0.525 0.585
NLU PO 0.606 0.550 0.583 0.647 0.566 0.613 0.647 0.580 0.619
FT-all 0.633 0.551 0.599 0.638 0.548 0.600 0.655 0.573 0.621
FT-all GP10 0.609 0.543 0.581†0.579 0.526 0.557 0.641 0.552 0.604
FT-all GP20 0.595 0.537 0.571 0.576 0.530 0.556 0.643 0.547 0.603
FT-all GP50 0.603 0.535 0.575 0.586 0.524 0.560 0.642 0.552 0.604†
FT-all GP100 0.608 0.541 0.580 0.575 0.528 0.555 0.641 0.551 0.604
FT-all GEPA−l 0.592 0.513 0.559 0.607 0.521 0.571 0.632 0.537 0.592
FT-all GEPA−m 0.600 0.510 0.563 0.608 0.533 0.577†0.631 0.545 0.595
FT-all GEPA−h 0.594 0.513 0.560 0.600 0.523 0.568 0.636 0.544 0.597
STM Ties 0.637 0.585 0.615 0.651 0.576 0.619 0.669 0.6060.643
STM Linear 0.638 0.585 0.616 0.654 0.577 0.622 0.6770.603 0.646
†Highest-performing prompt optimization for this base model; used as the main prompt for experts.
Table 10: Prompt Optimization results across the base models (average NDCG@10 over 12 MTEB tasks).
Medical General
Model SizeFeedback
QAMedical
QACUREv1Public
HealthQANF
CorpusTREC
COVIDSci
FactSCI
DOCSNano
FEVERArgu
AnaNano
QuoraFiQA
2018Avg
Med.Avg
Gen.Avg
All
BM25 - 0.563 0.458 0.355 0.718 0.321 0.623 0.686 0.158 0.809 0.492 0.863 0.251 0.532 0.515 0.525
Contriever 150M 0.505 0.592 0.351 0.694 0.313 0.448 0.655 0.171 0.794 0.484 0.944 0.274 0.508 0.533 0.519
E5Large V2335M 0.704 0.6990.5620.856 0.372 0.666 0.722 0.205 0.889 0.464 0.912 0.411 0.654 0.576 0.622
GTRT5 XL1.2B 0.577 0.692 0.507 0.713 0.333 0.601 0.642 0.157 0.846 0.528 0.9570.442 0.581 0.586 0.583
BMRetriever 2B 0.587 0.727 0.471 0.812 0.3470.8390.729 0.1860.9400.356 0.960 0.357 0.645 0.560 0.609
LLM2Vec 3B 0.7080.731 0.490 0.8140.3850.5720.746 0.190 0.864 0.553 0.953 0.423 0.635 0.597 0.619
STM Qwen3 0.6B 0.681 0.697 0.487 0.819 0.340 0.761 0.681 0.190 0.865 0.548 0.967 0.354 0.638 0.585 0.616
STM Gemma 2B 0.621 0.717 0.5030.8640.368 0.793 0.715 0.201 0.898 0.479 0.969 0.338 0.654 0.577 0.622
STM Phi4 3.8B 0.6970.7320.531 0.860 0.382 0.791 0.744 0.2140.8530.562 0.9690.414 0.677 0.603 0.646
Table 11: Comparison of retrieval performance on English tasks from MTEB. Results are reported primarily on
the medical subset, with additional evaluation on five general-domain subsets. Performance is measured using
NDCG@10.
0.00.20.40.60.81.0Weight
0.41.0
0.20.7
0.51.0
0.5
0.10.30.30.30.5
0.40.8
0.20.9
0.10.50.5Med Real
Med Synth
Search
NLU
Med Real
Med Synth
Search
NLU
Med Real
Med Synth
Search
NLUQwen 0.6B Gemma 2B Phi4 3.8BLinear
TIES
Figure 5: Mergingweightcoefficients for each expert forLinearandTIEStechniques for each model.
16