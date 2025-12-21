# Exploration of Augmentation Strategies in Multi-modal Retrieval-Augmented Generation for the Biomedical Domain: A Case Study Evaluating Question Answering in Glycobiology

**Authors**: Primož Kocbek, Azra Frkatović-Hodžić, Dora Lalić, Vivian Hui, Gordan Lauc, Gregor Štiglic

**Published**: 2025-12-18 17:35:04

**PDF URL**: [https://arxiv.org/pdf/2512.16802v1](https://arxiv.org/pdf/2512.16802v1)

## Abstract
Multi-modal retrieval-augmented generation (MM-RAG) promises grounded biomedical QA, but it is unclear when to (i) convert figures/tables into text versus (ii) use optical character recognition (OCR)-free visual retrieval that returns page images and leaves interpretation to the generator. We study this trade-off in glycobiology, a visually dense domain. We built a benchmark of 120 multiple-choice questions (MCQs) from 25 papers, stratified by retrieval difficulty (easy text, medium figures/tables, hard cross-evidence). We implemented four augmentations-None, Text RAG, Multi-modal conversion, and late-interaction visual retrieval (ColPali)-using Docling parsing and Qdrant indexing. We evaluated mid-size open-source and frontier proprietary models (e.g., Gemma-3-27B-IT, GPT-4o family). Additional testing used the GPT-5 family and multiple visual retrievers (ColPali/ColQwen/ColFlor). Accuracy with Agresti-Coull 95% confidence intervals (CIs) was computed over 5 runs per configuration. With Gemma-3-27B-IT, Text and Multi-modal augmentation outperformed OCR-free retrieval (0.722-0.740 vs. 0.510 average accuracy). With GPT-4o, Multi-modal achieved 0.808, with Text 0.782 and ColPali 0.745 close behind; within-model differences were small. In follow-on experiments with the GPT-5 family, the best results with ColPali and ColFlor improved by ~2% to 0.828 in both cases. In general, across the GPT-5 family, ColPali, ColQwen, and ColFlor were statistically indistinguishable. GPT-5-nano trailed larger GPT-5 variants by roughly 8-10%. Pipeline choice is capacity-dependent: converting visuals to text lowers the reader burden and is more reliable for mid-size models, whereas OCR-free visual retrieval becomes competitive under frontier models. Among retrievers, ColFlor offers parity with heavier options at a smaller footprint, making it an efficient default when strong generators are available.

## Full Text


<!-- PDF content starts -->

Exploration of Augmentation Strategies in
Multi-modal Retrieval-Augmented Generation for
the Biomedical Domain*
*A Case Study Evaluating Question Answering in Glycobiology
Primo ˇz Kocbek
University of Maribor, Faculty of Health Sciences
Maribor, Slovenia
University of Ljubljana, Medical Factory
Ljubljana, Slovenia
primoz.kocbek@um.siAzra Frkatovi ´c-Hod ˇzi´c
Genos Ltd
Zagreb, Croatia
afrkatovic@genos.hrDora Lali ´c
Genos Ltd
Zagreb, Croatia
dora@glycanage.com
Vivian Hui
Center for Smart Health, School of Nursing
The Hong Kong Polytechnic University
Hong Kong, China
vivianc.hui@polyu.edu.hkGordan Lauc
University of Zagreb,
Faculty of Pharmacy and Biochemistry
Zagreb, Croatia
Genos Ltd
Zagreb, Croatia
glauc@genos.hrGregor ˇStiglic
University of Maribor,
Faculty of Health Sciences
Maribor, Slovenia
Usher Institute
University of Edinburgh
Edinburgh, UK
gregor.stiglic@um.si
Abstract—Multi-modal retrieval-augmented generation (MM-
RAG) promises grounded biomedical QA, but it is unclear when
to (i) convert figures/tables into text versus (ii) use optical
character recognition (OCR)-free visual retrieval that returns
page images and leaves interpretation to the generator. We
study this trade-off in glycobiology, a visually dense domain.
We built a benchmark of 120 multiple-choice questions (MCQs)
from 25 papers, stratified by retrieval difficulty (easy text,
medium figures/tables, hard cross-evidence). We implemented
four augmentations—None, Text RAG, Multi-modal conversion,
and late-interaction visual retrieval (ColPali)—using Docling
parsing and Qdrant indexing. We evaluated mid-size open-
source and frontier proprietary models (e.g., Gemma-3-27B-
IT, GPT-4o family). Additional testing used the GPT-5 family
and multiple visual retrievers (ColPali/ColQwen/ColFlor). Ac-
curacy with Agresti–Coull 95% confidence intervals (CIs) was
computed over 5 runs per configuration. With Gemma-3-27B-
IT, Text and Multi-modal augmentation outperformed OCR-free
retrieval (0.722–0.740 vs. 0.510 average accuracy). With GPT-4o,
Multi-modal achieved 0.808, with Text 0.782 and ColPali 0.745
close behind; within-model differences were small. In follow-on
experiments with the GPT-5 family, the best results with ColPali
and ColFlor improved by 2% to 0.828 in both cases. In general
across the GPT-5 family, ColPali, ColQwen, and ColFlor were
statistically indistinguishable; ColFlor matched ColPali while
being far smaller. GPT-5-nano trailed larger GPT-5 variants by
roughly 8–10%. Pipeline choice is capacity-dependent: converting
visuals to text lowers the reader burden and is more reliable
This work was supported by European Union under Horizon Europe
[grant number 101159018] and EuroHPC JU [grant number 101101903];
by University of Maribor under the 2023 internal call Strengthening Re-
searchers’ Programme Cores, field of Data Science and Artificial Intelligence
in Biomedicine; Slovenian Research Agency [grant number GC-0001].for mid-size models, whereas OCR-free visual retrieval becomes
competitive under frontier models. Among retrievers, ColFlor
offers parity with heavier options at a smaller footprint, making
it an efficient default when strong generators are available.
I. INTRODUCTION
The rapid proliferation of Large Language Models (LLMs)
has transformed numerous domains, including biomedical
question answering (QA). Advanced systems such as Med-
PaLM 2 have demonstrated expert-level performance on stan-
dardized medical examinations [1], leveraging well-established
biomedical benchmarks such as BioASQ [2] and Pub-
MedQA [3]. Despite these successes, their reliability in spe-
cialized scientific domains remains limited due to training data
constraints. This highlights the need for external knowledge
grounding to mitigate hallucinations and improve factual ac-
curacy [4].
Retrieval-Augmented Generation (RAG) has emerged as a
robust alternative or complement to fine-tuning (FT), dynami-
cally providing relevant contextual information to LLMs dur-
ing inference [5]. Compared with FT, RAG typically achieves
stronger performance at substantially lower computational cost
and does not require retraining when new knowledge becomes
available [6]. This property makes RAG particularly suitable
for rapidly evolving biomedical literature.
While text-based RAG systems are well established, sci-
entific research often involves multi-modal content extending
beyond text. Our study focuses on glycobiology—a visu-
ally dense and technically demanding domain encompassingarXiv:2512.16802v1  [cs.CL]  18 Dec 2025

complex molecular structures, pathway diagrams, and tabular
datasets [7]. Prior work has shown that even advanced LLMs
struggle with glycobiology-related queries, frequently produc-
ing inconsistent or fabricated responses [8]. This underscores
the need for multi-modal RAG (MM-RAG) systems capable
of processing heterogeneous data modalities.
In this preliminary study, we compare two primary MM-
RAG paradigms. The first converts multiple modalities into
text following a conventional pipeline: PDF documents are
parsed into structured elements—via direct text extraction
or Optical Character Recognition (OCR)—while figures and
tables are transformed into textual descriptions using summa-
rization techniques [9]. The textual content is subsequently
chunked and embedded for retrieval via standard vector
databases. This approach simplifies decoupling between re-
trieval and generation but risks information loss and pipeline
complexity.
The second paradigm employs vision-based document re-
trieval to circumvent conversion-related limitations. These
methods treat entire document pages as images and utilize
Vision-Language Models (vision-language models (VLMs))
to generate embeddings directly. A notable example is Col-
Pali [10], which adapts the late-interaction mechanism from
ColBERT [11] to the visual domain, enabling fine-grained
similarity matching between visual and textual embeddings.
Related approaches such as VisRAG also leverage VLM-based
retrievers and generators to preserve maximal information
from original documents [12]. In such setups, the downstream
LLM assumes greater responsibility for visual reasoning, ef-
fectively replacing traditional PDF parsing and summarization
components.
This study investigates whether the performance of OCR-
free, vision-based document retrieval depends on the multi-
modal reasoning capacity and scale of the downstream gen-
erative model in the context of multiple-choice question
(MCQ) answering from glycobiology literature. We evaluate
three RAG strategies—standard text-based RAG, modality-
converting MM-RAG, and vision-based ColPali—across both
large proprietary models (e.g., the OpenAI GPT-4o family)
and smaller open-source alternatives (e.g., Gemma-3-27B-IT).
Furthermore, we assess different late-interaction visual retriev-
ers—ColPali [10], ColFlor [13], and ColQwen [14]—using the
recently released OpenAI GPT-5 family.
Our findings reveal a trade-off between pipeline simplicity
and dependence on model reasoning capacity, establishing an
initial baseline for developing reliable and trustworthy multi-
modal RAG systems in specialized biomedical domains.
II. RELATEDWORK
The biomedical domain is inherently multi-modal, as clin-
icians and researchers routinely integrate information from
medical imaging, laboratory results, electronic health records
(EHRs), and genomics [15], [16]. This complexity makes it
a natural setting for developing multi-modal artificial intel-
ligence (AI) systems. Notable progress has been achieved
in multi-modal question answering (MMQA), with modelssuch as LLaV A-Med [17], Med-Flamingo [18], and Med-
PaLM M [19], which jointly reason across textual and visual
modalities in clinical contexts. However, public benchmarks
have revealed potential data contamination in newer LLMs,
motivating the use of private, domain-specific datasets for
evaluation.
Biomedical AI represents a key application area for
Retrieval-Augmented Generation (RAG), characterized by a
vast and rapidly evolving knowledge base where factual accu-
racy and evidence-based reasoning are paramount [20], [21].
Standard LLMs are static and prone to hallucinations, making
RAG critical for building reliable clinical systems [20]. A
systematic review of 30 studies identified diagnostic support,
EHR summarization, and medical QA as the most preva-
lent RAG applications [21]. A meta-analysis of 20 studies
comparing baseline LLMs with RAG-augmented counterparts
reported a pooled odds ratio of 1.35 (95% CI: 1.19–1.53),
demonstrating significant performance gains through RAG
integration [20].
For instance, a biomedical QA system employing a fine-
tuned Mistral-7B model that retrieved information from
PubMed and medical encyclopedias achieved a BERTScore F1
of 0.843 [22], [23]. Similarly, systems such as MEDGPT [24]
have demonstrated the practical utility of RAG for diagnostics
and automated report generation. Despite these advances, sev-
eral challenges persist. The technical terminology and struc-
tural density of biomedical literature often introduce retrieval
noise, resulting in suboptimal generation [21]. Furthermore,
biomedical content is fundamentally multi-modal: research ar-
ticles, clinical guidelines, and EHRs frequently include visual
components such as graphs, flowcharts, and tables that are
integral to interpretation. Yet, most existing biomedical RAG
implementations remain text-focused. The same systematic
review confirming RAG’s clinical benefits revealed that only
three of the twenty analyzed studies explicitly incorporated
tabular or visual data—and typically by converting these
modalities into text [20]. This limited engagement with native
multi-modal content underscores the need for architectures
capable of directly processing complex visual information.
To address this gap, the MRAG-Bench benchmark was
introduced to evaluate vision-centric multi-modal retrieval-
augmented models [25]. It contains multiple-choice questions
requiring visual reasoning across scenarios involving perspec-
tive shifts, occlusion, and temporal or geometric transfor-
mations [25]. The benchmark highlights a substantial gap
between machine and human visual reasoning capabilities:
when provided with ground-truth visual knowledge, GPT-4o’s
accuracy improved by only 5.82%, whereas human partici-
pants improved by 33.16% [25]. These findings quantitatively
support the hypothesis that the performance of vision-based
RAG pipelines depends strongly on the scale and multi-modal
reasoning ability of the underlying generative model.

III. MATERIALS ANDMETHODS
A. Benchmark
A private benchmark dataset was constructed by two domain
experts, comprising 120 multiple-choice questions (MCQs)
derived from 25 original research and review manuscripts.
Each question contained four possible answers and the corpus
spanned core glycobiology concepts and applications, ranging
from population-scale IgG glycomics, immune and inflam-
matory regulation, and endocrine/aging effects, to cardio-
metabolic, gastrointestinal, pulmonary, and oncologic disease
phenotypes (Appendix A). Questions were also categorized
by retrieval difficulty by consensus by the two domain ex-
perts:easywhen the answer appeared directly in the text,
mediumwhen it was presented in tables or figures, andhard
when it required integrating information across text, figures,
supplementary tables, or cited references. Iterative manual re-
finements were performed through review of model-generated
explanations, during which five items were reclassified to
ensure consistent difficulty labeling.
B. Vision-Language Model Selection
The selection of open-source vision-language mod-
els (VLMs) was constrained by local inference re-
sources—specifically a single NVIDIA H100 80GB PCIe
GPU. Considering activation memory and framework over-
head, models up to approximately 30 billion parameters
were feasible under 16-bit precision. We focused on mod-
els adapted from general multi-modal LLMs to biomedical
applications [26], employing a generate-then-filter pipeline to
synthesize diverse visual-instruction data from biomedical im-
age–caption pairs. The evaluated models included Qwen2-VL-
2B-Instruct [27], LLaV A-NeXT-Llama3-8B [28], and Llama-
3.2-11B-Vision-Instruct [29]. We additionally tested Google’s
Gemma 3 modelgemma-3-27b-it[30]. The were de-
ployed using vLLM [31] via docker (example in Appendix C)).
For proprietary baselines, the OpenAI GPT-4o fam-
ily (gpt-4o, version 2024-11-20;gpt-4o-mini, ver-
sion 2024-07-18) and GPT-5 family (gpt-5,gpt-5-mini,
gpt-5-nano; all version 2025-08-07) were accessed via
API under a GDPR-compliant Data Processing Addendum
(DPA) [32].
C. Multi-Modal RAG Framework
The multi-modal RAG framework consisted of three mod-
ular components: a document parser, an open-source vector
store, and a vision-language model for text–visual alignment at
inference. IBM’sDoclingserved as the document parser [33],
andQdrantas the vector database [34], supporting both
single- and multi-vector representations. Four vector store
configurations were developed: (i)text-only conversion, in
which all modalities were summarized into text; and (ii)visual
late-interactionvariants (ColPali), in which document pages
were represented as image embeddings. Retrieval employed
a late-interaction mechanism [11] for fine-grained similarity
matching. Standard semantic embeddings (BAAI/bge-base-en-
v1.5) [35] were used for the text-only and multi-modal–textconfigurations. A general prompt template was used for cre-
ating table and figure summaries (Appendix B). Docling,
Qdrant and the open-source models were deployed on-premise
on a GPU server (NVIDIA H100 80GB PCIe GPU) using
docker and accessed through api (example configurations
are in Appendix C). For chunking, we adopted Docling’s
HierarchicalChunker[33], which uses the structural
information encoded in theDoclingDocumentto produce
one chunk per detected document element. We set a token
budget of 16,000 tokens and capped image resolution at 1,300
pixels on the longer side.
We compared multiple augmentation strategies supplied to
the LLM:None(query only),Text(nearest text chunks via
standard RAG),Multi-Modal(raw figures and tables with cor-
responding summaries), and vision-based retrievals (ColPali,
ColQwen,ColFlor), in which the most similar document pages
were retrieved and passed directly to the LLM.
D. Vision-Based Retrievers
We selected vision-based retrievers that demonstrated
strong performance on the ViDoRe v2 benchmark [36]—par-
ticularly on healthcare-related subsets—and that were
parameter-efficient, such as ColFlor. Specifically, we used
the repositoriesvidore/colpali-v1.3-mergedfor
ColPali,vidore/colqwen2-v0.2for ColQwen, and
ahmed-masry/ColFlorfor ColFlor.
ColPali(vidore/colpali-v1.3-merged) is
a vision-based document retrieval model built on the
PaliGemma-3B vision–language backbone. It extends SigLIP
by feeding patch embeddings into PaliGemma to produce
ColBERT-style multi-vector representations of pages [37].
The model (2.92 billion parameters) encodes entire page
images—including text, layout, figures, and tables—into
patch embeddings, then matches queries to pages using
late-interaction scoring (e.g., MaxSim) over token–patch
pairs [38]. Strengths include preservation of visual layout
and non-textual information, elimination of OCR and layout
parsing, and strong retrieval accuracy on visual document
benchmarks. Limitations include high memory and storage
requirements due to multi-vector embeddings and increased
computational cost for large document collections.
ColQwen2(vidore/colqwen2-v0.2) follows the Col-
Pali paradigm but leverages the Qwen2-VL-2B backbone.
It adopts a multi-vector retrieval scheme and incorporates
adapters on top of Qwen2-VL-2B [39]. The Qwen2 backbone
provides vision–language alignment across image and text
modalities. Owing to its modular, adapter-based architecture,
ColQwen2 facilitates efficient model switching and reduced
update cost. However, its performance depends on the quality
of adapter tuning; suboptimal alignment can degrade retrieval
quality relative to ColPali. It also inherits the same storage and
matching complexity constraints associated with multi-vector
retrieval.
ColFlor(ahmed-masry/ColFlor) is a lightweight vi-
sual retriever optimized for footprint and speed. According to

its model card, ColFlor contains approximately 174 million pa-
rameters—roughly 17×smaller than ColPali [40]. It achieves
query encoding∼9.8×faster and image encoding∼5.25×
faster than ColPali, with only a∼1.8% performance drop on
text-rich English documents [41]. Architecturally, ColFlor em-
ploys Florence-2’s DaViT vision encoder followed by a BART
text encoder to produce contextualized visual embeddings,
projected into a 128-dimensional latent space for retrieval via
late interaction [41]. Advantages include low computational
requirements, high throughput, and suitability for resource-
constrained environments, whereas limitations include reduced
performance on highly visual or non-English documents and
limited capacity for capturing fine-grained visual detail due to
its smaller backbone.
E. Evaluation
Performance was quantified using mean accuracy, defined as
the proportion of correctly answered MCQs. With four equally
likely options, random guessing yields an expected accuracy
of 0.25. We evaluated both overall and difficulty-stratified
performance. Two primary experiments were conducted with
a general prompt template used (Appendix D).
First, we compared proprietary GPT-4o family of models
with the open-source Gemma-3-27B-IT across all augmenta-
tion strategies, performing ten runs per configuration—five
with permuted answer orders and five without—to assess
potential benchmark contamination. The selection of open
source model was due to the superior performance among
other open-source models (Table I). Second, we compared
vision-based retrievers (ColPali, ColQwen, ColFlor) using the
GPT-5 family of models, with five evaluation runs each. In
both cases we used default temperature and seed values -
details for first case in Table I, which was gathered from
model documentation and for the second case we note that the
temperature parameter was removed from the GPT-5 family of
models, seed was random. All accuracy point estimates were
reported using Agresti–Coull [42] 95% confidence intervals
(CIs) aggregated across runs. This method provides a robust
and stable estimation for binomial proportions, particularly
when sample sizes are finite or proportions are near the interval
boundaries of 0 or 1.
We checked for benchmark contamination by compar-
ing answer-order permutation or shuffling. For each model-
augmentation combination a paired t-test was performed with
a conservative Bonferroni correction for multiple testing (Fig-
ure 1).
Because each augmentation–model combination was eval-
uated on the identical set of 120 MCQs, per-run accuracies
form natural paired observations; we therefore used paired
non-parametric tests (Wilcoxon signed-rank) for within-model
comparisons with Bonferroni correction was used. The result-
ing test values and p-values are summarized in Table III.
We lastly looked at some retrieval metrics (precision@5),
cost (cost per run, cost-per-correct comparison) and latency
(tokens per second) analysis aggregated across runs focusingon vision-based retrievers and GPT-5 family of models. We re-
ported point estimates using bootstrapped 95% CIs aggregated
across runs.
IV. RESULTS
TABLE I
INITIAL MODEL EVALUATION WITH REGARD TO THE PROPOSED
FRAMEWORK.
Model Temp.* Seed* Augmentation
None TextMulti
-modalColPali**
AdaptLLM/biomed
-LLaV A-NeXT-Llama3-8B0.6 0 0.258 0.192 0.283 0.258
AdaptLLM/biomed
-Qwen2-VL-2B-Instruct0.01 0 0.217 0.283 0.283 /
AdaptLLM/biomed
-Llama-3.2-11B
-Vision-Instruct0.6 0 0.383 0.508 0.542 0.475
google/gemma-3-27b-it 0.7 0 0.483 0.692 0.767 0.558
gpt-4o-mini 0.7 rnd. 0.458 0.708 0.675 0.558
gpt-4o0.7 rnd.0.592 0.792 0.833 0.758
*default values of temperature and seeds were used (rnd. is random)
**AdaptLLM/biomed-Qwen2-VL-2B-Instruct ColPali augmentation did not
produce relevant answers.
From the initial testing (Table I), smaller open-source
models—even when fine-tuned—struggled on domain-specific
MCQs. Augmentation improved accuracy by roughly 12–26%.
Among open-source models,gemma-3-27b-itachieved
the strongest performance and was selected for detailed eval-
uation. ColPali augmentation performed well primarily with
gpt-4o.
.0066**NoneT extMulti-modalColPaligemma-3-27bgpt-4o-minigpt-4oY es No Y es No Y es No Y es No0.40.50.60.70.8
0.40.50.60.70.8
0.40.50.60.70.8
Permuted answersAccuracy
Fig. 1. Boxplot and significantp-values for accuracy across selected models
and augmentations.
Because contamination in public benchmarks can under-
mine evaluation robustness, we tested whether answer-order

permutations affected performance. As expected (Figure 1), we
found no statistically significant differences atα= 0.05. The
closest case wasgemma-3-27b-itwith no augmentation
(p= 0.0066), consistent with its lower accuracy and higher
variability (0.477; 95% CI [0.437, 0.517]) versus 0.400 (95%
CI [0.363, 0.440]).
TABLE II
MODEL EVALUATION OF MODELS WITH REGARD TO THE PROPOSED
FRAMEWORK.
Model Aug.Easy
(n=69)Medium
(n=24)Hard
(n=27)Average
(n=120)
gemma-3
-27b-itNone 0.414 0.407 0.350 0.400
[0.364, 0.467] [0.328, 0.492] [0.270, 0.439] [0.362, 0.440]
Text 0.780 0.711 0.567 0.722
[0.733, 0.820] [0.629, 0.781] [0.477, 0.652] [0.684, 0.756]
Multi 0.786 0.741 0.608 0.740
-modal[0.739, 0.826] [0.661, 0.808] [0.519, 0.691] [0.703, 0.774]
ColPali 0.548 0.459 0.458 0.510
[0.495, 0.600] [0.377, 0.543] [0.372, 0.547] [0.470, 0.550]
gpt-4o
-miniNone 0.568 0.422 0.325 0.487
[0.515, 0.619] [0.342, 0.507] [0.248, 0.413] [0.447, 0.527]
Text 0.745 0.600 0.525 0.668
[0.696, 0.788] [0.516, 0.679] [0.436, 0.612] [0.630, 0.705]
Multi 0.754 0.563 0.633 0.687
-modal[0.705, 0.796] [0.479, 0.644] [0.544, 0.714] [0.648, 0.723]
ColPali 0.609 0.667 0.500 0.600
[0.556, 0.659] [0.583, 0.741] [0.412, 0.588] [0.560, 0.638]
gpt-4oNone 0.664 0.630 0.358 0.595
[0.612, 0.712] [0.546, 0.707] [0.278, 0.447] [0.555, 0.634]
Text 0.832 0.830 0.583 0.782
[0.789, 0.868] [0.757, 0.884] [0.494, 0.668] [0.747, 0.813]
Multi0.835 0.874 0.658 0.808
-modal[0.792, 0.870] [0.807, 0.921] [0.570, 0.737] [0.775, 0.838]
ColPali 0.748 0.837 0.633 0.745
[0.699, 0.791] [0.765, 0.891] [0.544, 0.714] [0.709, 0.778]
Note: [95% CI] - Agresti-Coull 95% confidence interval; underline - best
performing augmentation for open-source model(gemma3-27b-it); bold
- best performing augmentation for proprietary model (gpt-4o); Aug. -
augmentation.
In the first main experiment,gpt-4oachieved the high-
est overall accuracy with multi-modal augmentation (0.808;
95% CI [0.775, 0.838]) (Table II). Text augmentation (0.782;
95% CI [0.747, 0.813]) and ColPali (0.745; 95% CI [0.709,
0.778]) were slightly lower; pairwise differences were not
statistically significant forgpt-4o. Forgemma-3-27b-it,
multi-modal (0.740; 95% CI [0.703, 0.774]) and text (0.722;
95% CI [0.684, 0.756]) significantly outperformed ColPali
(0.510; 95% CI [0.470, 0.550]). By difficulty,gpt-4owith
ColPali (0.745; 95% CI [0.709, 0.778]) was comparable to
gemma-3-27b-itwith its best augmentation (0.740; 95%
CI [0.703, 0.774]).
Within-model model pairwise comparisons using the
Wilcoxon signed-rank test with Bonferroni correction are
summed in Table III. We can observe that all augmentations
(Text, ColPali, Multi-modal) outperform no augmentation andTABLE III
PAIRWISE COMPARISONS OF STATISTICALLY SIGNIFICANT DIFFERENCE
BETWEEN AUGMENTATION AND MODELS(p <0.05).
Model Augmentations Test-value (V) p-value
gemma-3-27b-itNone<Text 597.0<0.001
None<ColPali 291.5<0.001
None<Multi-modal 1078.5<0.001
ColPali<Text 3448.5<0.001
ColPali<Multi-modal 3442.0<0.001
gpt-4o-miniNone<Text 676.5<0.001
None<Multi-modal 597.0<0.001
None<ColPali 718.0<0.001
ColPali<Text 1997.00.012
ColPali<Multi-modal 2291.50.001
gpt-4oNone<Text 319.0<0.001
None<Multi-modal 215.0<0.001
None<ColPali 359.0<0.001
ColPali<Multi-modal 959.50.014
only forgpt-4oText augmentation does not outperformed
ColPali.
TABLE IV
PERFORMANCE OF VISUAL RETRIEVERS(COLPALI, COLQWEN,
COLFLOR)ACROSS THEGPT-5MODEL FAMILY. VALUES ARE MEAN
ACCURACY WITHAGRESTI–COULL95% CIS.
LLM
modelRetrieval
modelEasy
(n=69)Medium
(n=24)Hard
(n=27)Average
(n=120)
gpt-5ColPali 0.835 0.830 0.8080.828
[0.792, 0.870] [0.757, 0.884] [0.728, 0.869][0.796, 0.856]
ColQwen 0.835 0.822 0.783 0.822
[0.792, 0.870] [0.748, 0.878] [0.701, 0.848] [0.789, 0.850]
ColFlor 0.823 0.837 0.833 0.828
[0.779, 0.860] [0.765, 0.891] [0.756, 0.890] [0.796, 0.856]
gpt-5
-miniColPali 0.817 0.859 0.717 0.807
[0.773, 0.855] [0.790, 0.909] [0.630, 0.790] [0.773, 0.836]
ColQwen 0.817 0.830 0.708 0.798
[0.773, 0.855] [0.757, 0.884] [0.621, 0.782] [0.764, 0.829]
ColFlor 0.823 0.859 0.758 0.818
[0.779, 0.860] [0.790, 0.909] [0.674, 0.827] [0.785, 0.847]
gpt-5
-nanoColPali 0.797 0.763 0.600 0.750
[0.751, 0.836] [0.684, 0.827] [0.511, 0.683] [0.714, 0.783]
ColQwen 0.786 0.793 0.575 0.745
[0.739, 0.826] [0.716, 0.853] [0.486, 0.660] [0.709, 0.778]
ColFlor 0.759 0.770 0.592 0.728
[0.712, 0.802] [0.692, 0.834] [0.502, 0.675] [0.691, 0.762]
Note: [95% CI] - Agresti-Coull 95% confidence interval; underline - best
performing retrieval model for a specific model; bold - best performing
retrieval model and model combination
In the second experiment (Table IV) comparing visual re-
trievers with thegpt-5family, the highest overall mean accu-
racy was tied between ColPali (0.828; 95% CI [0.796, 0.856])
and ColFlor (0.828; 95% CI [0.796, 0.856]) usinggpt-5. For
gpt-5-mini, medium-difficulty items reached the highest
category-wise accuracy with ColPali and ColFlor (both 0.859;
95% CI [0.790, 0.909]). In contrast,gpt-5-nanounderper-
formed the larger variants by∼8–10 %.
Within eachgpt-5model, differences among visual re-
trievers were small and not statistically significant. The small-

est unadjustedp-value was 0.430 (ColPali vs. ColFlor on
gpt-5-nano); after Bonferroni correction across within-
model retriever comparisons,p= 1.0.
TABLE V
PERFORMANCE OF RETRIEVAL FOR THEGPT-5FAMILY WITH RESPECT TO
LATENCY AND COST.
ModelRetrieval
modelP@5 Latency Tokens TTFT* Cost** P/C***
gpt-5ColPali 0.020 20.12 4643.05 417.88 5.57 5.72
[0.000, 0.094] [0.00, 163.66] [4269.59, 5016.51] [0.00, 3739.00] [5.12, 6.02] [4.14, 7.29]
ColQwen 0.025 22.76 4385.31 366.17 5.26 5.28
[0.017, 0.033] [0.00, 81.50] [4273.68, 4496.94] [0.00, 960.92] [5.13, 5.39] [5.08, 5.48]
ColFlor 0.018 17.82 4626.25 431.14 5.55 5.67
[0.018, 0.018] [0.00, 151.84] [4378.85, 4873.66] [0.00, 3754.74] [5.25, 5.85] [5.36, 5.97]
gpt-5
-miniColPali 0.022 13.80 7787.21 962.55 1.87 1.87
[0.017, 0.026] [0.00, 44.56] [7112.34, 8462.07] [0.00, 2396.78] [1.71, 2.03] [1.59, 2.15]
ColQwen 0.024 24.34 8920.67 754.34 2.14 2.23
[0.019, 0.030] [0.00, 89.54] [8867.94, 8973.40] [0.00, 2050.25] [2.13, 2.15] [1.98, 2.47]
ColFlor 0.018 21.33 7592.96 388.19 1.82 1.89
[0.018, 0.018] [0.00, 85.57] [7562.46, 7623.45] [0.00, 1423.52] [1.82, 1.83] [1.65, 2.14]
gpt-5
-nanoColPali 0.022 14.08 9909.67 1283.69 0.47 0.54
[0.011, 0.034] [0.00, 48.16] [9408.38, 10410.96] [0.00, 3391.69] [0.45, 0.50] [0.48, 0.59]
ColQwen 0.026 23.66 11755.52 866.97 0.56 0.67
[0.024, 0.028] [0.00, 78.55] [11688.70, 11822.33] [0.77, 2364.71] [0.56, 0.57] [0.62, 0.72]
ColFlor 0.018 13.07 10031.55 1363.26 0.48 0.55
[0.018, 0.018] [0.00, 44.49] [9928.94, 10134.17] [0.00, 3523.51] [0.48, 0.49] [0.48, 0.62]
Note: [95% CI] – bootstrap 95% confidence intervals for each metric;
P@5 – precision at 5; *Throughput; ** Cost estimate for a run in USD;
***Price-per-cost (US cents) – cost estimate per correctly answered question.
Because each question was derived from a single source
article rather than a multi-document corpus, the study was not
designed to primarily evaluate retrieval performance. Accord-
ingly, retrieval quality, as measured by P@5 (Precision@5),
was uniformly low with only modest variation across retrieval
models (range 0.02–0.026; Table V). This is expected for a
single-document retrieval setting, as any additional returned
documents beyond the relevant one are counted as false posi-
tives, which compresses P@5 even for strong visual retrievers.
In contrast, computational footprint and cost varied
markedly across base models. Latency was similar across
the GPT-5 family (13–24 s per request), but tokens per run
increased as model size decreased, from roughly 4,500 for
gpt-5to 8,000 forgpt-5-miniand about 11,000 for
gpt-5-nano. Throughput followed the same trend with
roughly 400, 700, and 1,300 tokens/s, respectively). Despite
using fewer tokens,gpt-5was far more expensive, with mean
per-run costs of $5.6 versus $1.9 and $0.5, i.e., about 2.5× and
10× higher, respectively. Price-per-correctly answered query
showed the same pattern: roughly 5.7, 2.0, and 0.6 US cents for
gpt-5,gpt-5-mini, andgpt-5-nano. A more detailed
breakdown stratified by retrieval difficulty is in Appendix E.
V. DISCUSSION
Multi-modal RAG and OCR-free visual retrieval aim at
the same goal: grounding generation on document evidencerather than model memory. Both decouple knowledge from
weights and benefit from fine-grained, late-interaction match-
ing. Where they differ iswhenvisual content is inter-
preted. Modality-converting pipelines interpret earlier (via
OCR/layout analysis and summarization). Visual retrievers
interpret later (the LLM reads retrieved page images). This
placement matters in practice.
Experiment 1 (capacity×pipeline).When model capacity
is limited, direct visual retrieval underperforms. In our initial
screen,gemma-3-27b-itdid poorly with ColPali compared
with text or multi-modal conversion. This shows a failure
mode: the system can retrieve the right page but the generator
cannot reliably read it. With a stronger model (GPT-4o),
the gap closes and ColPali becomes competitive. The design
lesson is conservative: conversion to text lowers the burden
on the generator; OCR-free pipelines lean on LLM visual
reasoning.
Experiment 2 (retriever×model family).Within the GPT-
5 family, retriever choice mattered less than model scale.
ColPali, ColQwen, and ColFlor delivered near-identical means
with overlapping CIs. ColFlor matched ColPali while being far
smaller and faster, making it an efficient default with frontier
models. Using ColFlor,gpt-5exceededgpt-5-miniby
only about 1–2% on average, whilegpt-5-miniis roughly
5×cheaper per token. For many deployments, that trade-off
favorsgpt-5-miniwith ColFlor.
Pipeline simplicity vs. robustness.Visual retrievers avoid
OCR and fragile PDF parsing (table reconstruction, layout
heuristics), simplifying ingestion and limiting parser-induced
errors. The cost is a higher reliance on the LLM’s visual
reasoning. Conversion-based pipelines add engineering steps
and may lose detail, but generalize better to mid-sized models
and yield clearer evidence traces.
Implication.Choose capacity-aware designs. Under ample
compute or hosted APIs, OCR-free retrieval is attractive. Un-
der tighter budgets or on-prem constraints, conversion-based
multi-modal RAG remains the safer and more interpretable
default.
A. Limitations
This study has several limitations. The benchmark is small
(120 MCQs) and focused on glycobiology, so broader valida-
tion in other biomedical subfields (e.g., radiology, genomics) is
needed. The item mix is skewed toward “easy” cases (57.5%),
which reduces power for medium and hard analyses; we
treat those strata as exploratory. We checked for benchmark
contamination answer-order permutation, however we plan on
a more robust strategy in the future, such as paraphrasing
questions, adding non-relevant retrieval information, generat-
ing non correct answers. We analyzed one open-source model
in depth (gemma-3-27b-it) because its initial performance
was superior to other options; future work should span a wider
range of open-source VLMs to map scaling and architecture
effects. Finally, we reported multiple-choice accuracy; adding
factual consistency and evidence-use metrics would give a
fuller behavioral picture.

B. Future Research Directions
Retriever design.We will systematically compare modern
visual retrievers—ColPali [10], ColQwen [39], and ColFlor
[13]—and newer vision/video variants to see how retrieval
detail (page-, patch-, or token-level) and index compression
affect accuracy, speed, and storage on visually rich biomed-
ical articles. Prior work shows that late-interaction, OCR-
free retrievers can be both simple and highly competitive,
which makes them strong candidates for evidence retrieval in
practice. We will evaluate on document-centric and vision-
centric benchmarks to ensure results transfer beyond a single
dataset.
Lightweight and cost-aware MM-RAG.We will pair smaller,
faster retrievers with mid-sized generators to reduce cost and
latency while tracking accuracy and “cost per correct answer.”
ColFlor is an example of a compact, OCR-free retriever (about
174M parameters) that approaches the performance of heavier
models like ColPali, but runs substantially faster, suggesting
good cost–utility for institutional deployments.
Trustworthy grounding.Beyond accuracy, we will stress-
test how reliably models use retrieved evidence—especially
images, tables, and complex page layouts. Recent vision-
centric studies find that even strong models under-use retrieved
visuals, highlighting the need for strict citation, showing the
exact snippets or page crops used, and simple checks that
link each answer back to its sources. These measures support
auditability in clinical and research settings.
Grounded platform for researchers.We intend to turn
these ideas into a simple, secure platform that helps re-
searchers in their own domain ask better questions, find
the right evidence, and draft stronger proposals. The sys-
tem will use RAG so every answer isgroundedin trusted
sources the user selects/provides (papers, guidelines, lab doc-
uments, PDFs—including figures and tables). For visuals, it
will either convert images/tables to text or pass the page
image—whichever is clearer—and always show the exact
snippets used so claims can be checked. The platform will
scale from a single lab to an institution, support on-premise
options to protect sensitive data, and let teams choose smaller
or larger models to match budget and accuracy needs.
VI. CONCLUSION
This study compared text-centric and OCR-free visual re-
trieval strategies for multi-modal grounding on a glycobiology
MCQ benchmark. We found a clear capacity-dependent pat-
tern: conversion-based pipelines (text and multi-modal sum-
maries) are more reliable with mid-size VLMs (e.g., Gemma-
3-27B-IT), whereas OCR-free late-interaction retrieval be-
comes competitive with frontier models (e.g., GPT-4o / GPT-
5). Among visual-retrievers, ColFlor matched ColPali under
GPT-5 while offering a smaller footprint, suggesting an at-
tractive efficiency-accuracy trade-off. Practically, these results
argue for capacity-aware MM-RAG design: convert earlier
when model visual reasoning is constrained; retrieve page
images directly when model capacity allows.Our work is preliminary. The benchmark is small and single-
domain. Broader evaluations across modalities and domain
will test the generality of these findings and support trust-
worthy multi-modal assistants in biomedicine.
GENAI USAGE DISCLOSURE
In this manuscript we used Gemini and ChatGPT to help
improve its language and overall presentation. Our aim with
these tools was to enhance the text’s clarity, coherence, and
correctness, and we carefully reviewed and edited any sugges-
tions they provided. We want to clearly state that all scientific
content, core ideas, and analyses are entirely our own original
work, developed without relying on Gemini or ChatGPT for
any conceptual or analytical aspects of the research. As au-
thors, we are responsible for the manuscript’s final content and
affirm the originality of its scientific contributions stemming
from our own efforts.
CODEAVAILABILITY
A reproducibility package (benchmark template, scripts and
prompts) are available at: https://github.com/pkocbek/multi-
modal-RAG-biomed.
REFERENCES
[1] K. Singhal, S. Azizi, T. Tu, and et al., “Large language models encode
clinical knowledge,”Nature, vol. 620, no. 7974, pp. 172–180, 2023.
[2] G. Tsatsaronis, G. Balikas, P. Malakasiotis, and et al., “An overview
of the bioasq large-scale biomedical semantic indexing and question
answering competition,”BMC Bioinformatics, vol. 16, p. 138, 2015.
[3] Q. Jin, B. Dhingra, Z. Liu, W. Cohen, and X. Lu, “Pubmedqa: A dataset
for biomedical research question answering,” inProceedings of the 2019
Conference on Empirical Methods in Natural Language Processing
and the 9th International Joint Conference on Natural Language
Processing (EMNLP-IJCNLP). Hong Kong, China: Association for
Computational Linguistics, 2019, pp. 2567–2577. [Online]. Available:
https://aclanthology.org/D19-1259/
[4] O. Ovadia, M. Brief, M. Mishaeli, and O. Elisha, “Fine-tuning or
retrieval? comparing knowledge injection in llms,” inProceedings of
the 2024 Conference on Empirical Methods in Natural Language
Processing (EMNLP), 2024. [Online]. Available: https://aclanthology.
org/2024.emnlp-main.15/
[5] P. Lewis, E. Perez, A. Piktus, F. Petroni, V . Karpukhin, N. Goyal,
H. K”uttler, M. Lewis, W. tau Yih, T. Rockt”aschel, S. Riedel, and
D. Kiela, “Retrieval-augmented generation for knowledge-intensive nlp
tasks,” inAdvances in Neural Information Processing Systems 33
(NeurIPS 2020), 2020.
[6] M. Editors, “Retrieval-augmented generation in biomedical informatics,”
Biomedicines, vol. 12, no. 7, p. 687, 2024. [Online]. Available:
https://www.mdpi.com/2306-5354/12/7/687
[7] M. Frank and S. Schloissnig, “Bioinformatics and molecular modeling
in glycobiology,”Cellular and Molecular Life Sciences, vol. 67, no. 16,
pp. 2749–2772, 2010.
[8] D. O. Williams and E. Fadda, “Can chatgpt pass glycobiology?”
Glycobiology, vol. 33, no. 8, pp. 606–614, 2023. [Online]. Available:
https://academic.oup.com/glycob/article/33/8/606/7235670
[9] M. Alkhalaf, P. Yu, M. Yin, and C. Deng, “Applying generative ai with
retrieval augmented generation to summarize and extract key clinical
information from electronic health records,”Journal of Biomedical
Informatics, vol. 156, p. 104662, 2024.
[10] M. Faysse, A. Loison, Q. Mac ´eet al., “Colpali: Efficient document
retrieval with vision language models,” inInternational Conference
on Learning Representations (ICLR), 2025. [Online]. Available:
https://arxiv.org/abs/2407.01449
[11] O. Khattab and M. Zaharia, “Colbert: Efficient and effective passage
search via contextualized late interaction over bert,” inProceedings
of the 43rd International ACM SIGIR Conference on Research and
Development in Information Retrieval, 2020, pp. 39–48.

[12] S. Yu, C. Tang, B. Xu, J. Cui, J. Ran, Y . Yan, Z. Liu, S. Wang,
X. Han, Z. Liu, and M. Sun, “Visrag: Vision-based retrieval-augmented
generation on multi-modality documents,” inInternational Conference
on Learning Representations (ICLR), 2025, iCLR 2025. [Online].
Available: https://openreview.net/forum?id=zG459X3Xge
[13] A. Masry and et al., “Colflor: Towards bert-size vision-language
document retrieval models,”arXiv preprint arXiv:2405.05666, 2024.
[Online]. Available: https://huggingface.co/blog/ahmed-masry/colflor
[14] V . Team, “Colqwen2-v0.2: Vision-language document retrieval,” https:
//huggingface.co/vidore/colqwen2-v0.2, 2025.
[15] Google Research, “Multimodal medical ai,” 2023, accessed September
29, 2025. [Online]. Available: https://research.google/
[16] Y . Li and et al., “A comprehensive review of ai in multimodal biomedical
data analysis,”Genomics, Proteomics & Bioinformatics, vol. 23, no. 1,
p. qzaf011, 2024.
[17] C. Li, C. Wong, S. Zhang, N. Usuyama, H. Liu, J. Yang, T. Naumann,
H. Poon, and J. Gao, “Llava-med: Training a large language-and-
vision assistant for biomedicine in one day,” inAdvances in Neural
Information Processing Systems 36 (NeurIPS 2023), Datasets and
Benchmarks Track, 2023. [Online]. Available: https://aka.ms/llava-med
[18] M. Moor, S. Pineda, K. Kreis, M. Welling, M. Kraus, D. Rueckert,
and G. R”atsch, “Med-flamingo: a multimodal medical few-shot
learner,” inProceedings of the Machine Learning for Health (ML4H)
2023, vol. 225. PMLR, 2023, pp. 353–367. [Online]. Available:
https://proceedings.mlr.press/v225/moor23a.html
[19] T. Tu and et al., “Towards generalist biomedical ai,”arXiv preprint
arXiv:2307.14334, 2023. [Online]. Available: https://arxiv.org/abs/2307.
14334
[20] S. Liu, A. B. McCoy, and A. Wright, “Improving large language model
applications in biomedicine with retrieval-augmented generation: A
systematic review, meta-analysis, and clinical development guidelines,”
Journal of the American Medical Informatics Association, vol. 32, no. 4,
pp. 605–615, 2025.
[21] M. F. Aljunid and et al., “Retrieval-augmented generation (rag) in
healthcare: A comprehensive review,”AI, vol. 6, no. 9, p. 226, 2025.
[22] Emergent Mind Team, “Biomedical literature q&a system using
retrieval-augmented generation (rag),”arXiv preprint arXiv:2509.05505,
2025. [Online]. Available: https://arxiv.org/abs/2509.05505
[23] Emergent Mind, “Biomedical literature q&a system using retrieval-
augmented generation (rag),” 2025. [Online]. Available: https://www.
emergentmind.com/biomedragqa
[24] V . S. Sree and et al., “Medgpt: A modular chatbot for medical
diagnostics,” inProceedings of the 2024 International Conference on
Intelligent Systems and Machine Learning (ISML), 2024. [Online].
Available: https://ieeexplore.ieee.org/document/medgpt2024
[25] W. Hu and et al., “Mrag-bench: Vision-centric evaluation for
retrieval-augmented multimodal models,” inInternational Conference
on Learning Representations (ICLR), 2025. [Online]. Available:
https://iclr.cc/virtual/2025/poster/2410.08182
[26] D. Cheng, S. Huang, Z. Zhu, X. Zhang, W. X. Zhao, Z. Luan, B. Dai,
and Z. Zhang, “On domain-specific post-training for multimodal large
language models,”arXiv preprint arXiv:2411.19930, 2024. [Online].
Available: https://arxiv.org/abs/2411.19930
[27] P. Wang, S. Bai, S. Tan, S. Wang, Z. Fan, J. Bai, K. Chen,
X. Liu, J. Wang, W. Ge, Y . Fan, and et al., “Qwen2-vl:
Enhancing vision-language model’s perception of the world at any
resolution,”arXiv preprint arXiv:2409.12191, 2024. [Online]. Available:
https://arxiv.org/abs/2409.12191
[28] F. Li, R. Zhang, H. Zhang, Y . Zhang, B. Li, W. Li, Z. Ma, and C. Li,
“Llava-next-interleave: Tackling multi-image, video, and 3d in large
multimodal models,”arXiv preprint arXiv:2407.07895, 2024. [Online].
Available: https://arxiv.org/abs/2407.07895
[29] Meta AI, “Llama 3.2: Revolutionizing edge ai and vision with open,
customizable models,” 2024. [Online]. Available: https://ai.meta.com/
blog/llama-3-2-connect-2024-vision-edge-mobile-devices/
[30] A. Kamath, J. Ferret, S. Pathak, N. Vieillard, R. Merhej, S. Perrin,
T. Matejovicova, A. Ram ´e, M. Rivi `ere, L. Rouillard, and et al.,
“Gemma 3 technical report,”arXiv preprint arXiv:2503.19786, 2025.
[Online]. Available: https://arxiv.org/abs/2503.19786
[31] W. Kwon, Z. Li, S. Zhuang, Y . Sheng, L. Zheng, C. H. Yu, J. E.
Gonzalez, H. Zhang, and I. Stoica, “Efficient memory management for
large language model serving with pagedattention,” inProceedings of
the ACM SIGOPS 29th Symposium on Operating Systems Principles,
2023.[32] OpenAI, “Enterprise privacy at openai,” 2025. [Online]. Available:
https://openai.com/enterprise-privacy/
[33] C. Auer, M. Lysak, A. Nassar, M. Dolfi, N. Livathinos, P. Vagenas,
C. Berrospi Ramis, M. Omenetti, F. Lindlbauer, K. Dinkla, and et al.,
“Docling technical report,”arXiv preprint arXiv:2408.09869, 2024.
[Online]. Available: https://arxiv.org/abs/2408.09869
[34] Qdrant Team, “Qdrant: Open-source vector database and vector search
engine,” 2025. [Online]. Available: https://qdrant.tech/
[35] S. Xiao, Z. Liu, P. Zhang, N. Muennighoff, D. Lian, and J.-Y .
Nie, “C-pack: Packed resources for general chinese embeddings,” in
Proceedings of the 47th International ACM SIGIR Conference on
Research and Development in Information Retrieval, 2024, pp. 1–9.
[Online]. Available: https://doi.org/10.1145/3626772.3657878
[36] Q. Mac ´e, A. Loison, and M. Faysse, “Vidore benchmark v2: Raising
the bar for visual retrieval,”arXiv preprint arXiv:2505.17166, 2025.
[Online]. Available: https://arxiv.org/abs/2505.17166
[37] “vidore/colpali-v1.3-merged,” https://huggingface.co/vidore/colpali-v1.
3-merged, 2025, model card of ColPali (2.92B parameters).
[38] “Colpali model documentation in hugging face transformers,” https://
huggingface.co/docs/transformers/en/model doc/colpali, 2025.
[39] “manu/colqwen2-v0.2,” https://huggingface.co/manu/colqwen2-v0.2,
2025, model card of ColQwen2 (adapter over Qwen2-VL).
[40] “ahmed-masry/colflor model card,” https://huggingface.co/
ahmed-masry/ColFlor, 2024, 174M-parameter OCR-free visual
retriever.
[41] “Colflor: Towards bert-size vision-language document retrieval models,”
https://huggingface.co/blog/ahmed-masry/colflor, 2024, details on archi-
tecture, speeds, and performance.
[42] A. Agresti and B. A. Coull, “Approximate is better than ”exact” for
interval estimation of binomial proportions,”The American Statistician,
vol. 52, no. 2, pp. 119–126, 1998.
APPENDIXA
LIST OF DOI LINKS FOR THE25ORIGINAL RESEARCH AND
REVIEW MANUSCRIPTS
https://doi.org/10.1186/s12967-018-1695-0
https://doi.org/10.1097/hjh.0000000000002963
https://doi.org/10.1186/s12967-018-1616-2
https://doi.org/10.3390%2Fbiom13020375
https://doi.org/10.1016/j.bbagen.2017.06.020
https://doi.org/10.1172%2Fjci.insight.89703
https://doi.org/10.1016%2Fj.isci.2022.103897
https://doi.org/10.1002%2Fart.39273
https://doi.org/10.1016/j.bbadis.2018.03.018
https://doi.org/10.1097/MIB.0000000000000372
https://doi.org/10.1053%2Fj.gastro.2018.01.002
https://doi.org/10.1186/s13075-017-1389-7
https://doi.org/10.1021/pr400589m
https://doi.org/10.1161/CIRCRESAHA.117.312174
https://doi.org/10.2337/dc22-0833
https://doi.org/10.1097%2FMD.0000000000003379
https://doi.org/10.1158/1078-0432.CCR-15-1867
https://doi.org/10.1093/gerona/glt190
https://doi.org/10.1111/imr.13407
https://doi.org/10.1053/j.gastro.2018.05.030
https://doi.org/10.1016/j.csbj.2024.03.008
https://doi.org/10.1016/j.cellimm.2018.07.009
https://doi.org/10.1016/j.biotechadv.2023.108169
https://doi.org/10.4049/jimmunol.2400447
APPENDIXB
GENERAL PROMPT TEMPLATE WAS USED FOR CREATING
TABLE AND FIGURE SUMMARIES
You are an AI assistant specialized
in summarizing tables and figures for
efficient retrieval. \n\nInstructions:
\n\nIdentify Input Type: Explicitly
state whether the input provided is a
table or a figure.\nScientific Abstract:
Summarize the contents concisely in the
style of a scientific abstract. Include
relevant numeric values and key findings.
\nRetrieval Optimization: Structure your

summary clearly, optimizing keywords and
phrasing to enhance retrieval and indexing.
\nLength Constraint: Your summary must
strictly adhere to a maximum of 300 words
or 250 tokens. Do not exceed this limit
under any circumstances. Any text
exceeding will be just cutoff post
generation.\nAvoid Generic Openings: Do
not start your summary with generic
phrases such as "The image provided is,
" "The table shows," or similar
introductory sentences. Instead,
immediately describe the core content.
\nPrevent Redundancy: Write succinctly,
avoiding repetition of concepts or data
points.\nFinal output: Only summary text.
If no relevant data is present,
output \’\’.
APPENDIXC
EXAMPLE DOCKER CONFIGURATIONS FORDOCLING,
QDRANT AND GEMMA-3-27B-IT
Dockling:
docker run \
--rm --gpus all \
-e DOCLING_SERVE_ENABLE_UI=true \
-e DOCLING_SERVE_MAX_SYNC_WAIT=600 \
-e DOCLING_SERVE_ENABLE_REMOTE_SERVICES=true \
-e "DOCLING_SERVE_API_KEY=${DOCLING_API_KEY}" \
--name docling_serve \
-p 5001:5001 \
ghcr.io/docling-project/docling-serve-cu124:latest
Qdrant:
docker run \
--rm \
--name qdrant_vd \
--gpus=all \
-p 6333:6333 \
-p 6334:6334 \
-e QDRANT__GPU__INDEXING=1 \
-e "QDRANT__SERVICE__API_KEY=${QDRANT_API_KEY}" \
--ulimit nofile=65536:65536 \
-v ./src/vectordb/storage:/qdrant/storage \
-v ./src/vectordb/custom_config.yaml:
/qdrant/config/custom_config.yaml \
qdrant/qdrant:gpu-nvidia-latest
Gemma-3-27B-IT (vLLM):
docker run --gpus all -it --rm --pull=always \
--name gemma_27b \
-v "${HF_DIR}:/root/.cache/huggingface" \
--env "HUGGING_FACE_HUB_TOKEN=
${HUGGING_FACE_HUB_TOKEN}" \
--env TRANSFORMERS_OFFLINE=1 \
--env VLLM_RPC_TIMEOUT=180000 \
--env HF_DATASET_OFFLINE=1 \
-p 8006:8000 \
--ipc=host \
vllm/vllm-openai:latest \
--model "google/gemma-3-27b-it" \
--limit_mm_per_prompt ’{"image": 8}’ \
--gpu-memory-utilization 0.82 \\
--max_model_len 16000 \
--enable-sleep-mode
APPENDIXD
GENERAL PROMPT TEMPLATE FOR EVALUATION
Generate a JSON with the query_answer,
the answer provided behind the letters:
A, B, C, and D. These are the values.
Additional information if provided in
the Context below. If the Context isnot empty, analyse it and choose from
the letters. MAKE SURE your output is
one of the four values stated.
Here is the query: {question}.
Here are the choices: {question_string}
Context:
APPENDIXE
PERFORMANCE OF VISUAL RETRIEVERS ACROSS THE
GPT-5MODEL FAMILY STRATIFIED BY RETRIEVAL
DIFFICULTY.
TABLE VI
PRECISION@5ACROSS THEGPT-5MODEL FAMILY WITH BOOTSTRAPPED
95% CIS.
LLM
modelRetrieval
modelEasy
(n=69)Medium
(n=24)Hard
(n=27)Average
(n=120)
gpt-5ColPali 0.022 0.017 0.018 0.020
[0.000, 0.115] [0.000, 0.070] [0.000, 0.049] [0.000, 0.094]
ColQwen 0.032 0.003 0.027 0.025
[0.023, 0.042] [0.000, 0.015] [0.019, 0.034] [0.017, 0.033]
ColFlor 0.021 0.017 0.010 0.018
[0.021, 0.021] [0.017, 0.017] [0.010, 0.010] [0.018, 0.018]
gpt-5
-miniColPali 0.021 0.025 0.020 0.022
[0.011, 0.032] [0.000, 0.056] [0.000, 0.045] [0.017, 0.026]
ColQwen 0.031 0.004 0.025 0.024
[0.026, 0.037] [0.000, 0.015] [0.025, 0.025] [0.019, 0.030]
ColFlor 0.021 0.017 0.010 0.018
[0.021, 0.021] [0.017, 0.017] [0.010, 0.010] [0.018, 0.018]
gpt-5
-nanoColPali 0.027 0.017 0.012 0.022
[0.014, 0.041] [0.017, 0.017] [0.000, 0.031] [0.011, 0.034]
ColQwen 0.033 0.006 0.027 0.026
[0.027, 0.038] [0.000, 0.012] [0.019, 0.034] [0.024, 0.028]
ColFlor 0.021 0.017 0.010 0.018
[0.021, 0.021] [0.017, 0.017] [0.010, 0.010] [0.018, 0.018]
TABLE VII
LATENCY PER QUERY(S)ACROSS THEGPT-5MODEL FAMILY WITH
BOOTSTRAPPED95% CIS.
LLM
modelRetrieval
modelEasy
(n=69)Medium
(n=24)Hard
(n=27)Average
(n=120)
gpt-5ColPali 20.50 19.31 19.93 20.12
[0.00, 153.60] [0.00, 175.57] [0.00, 179.20] [0.00, 163.66]
ColQwen 22.89 22.15 23.11 22.77
[0.00, 81.84] [0.00, 78.91] [0.00, 83.48] [0.00, 81.50]
ColFlor 18.02 18.63 16.37 17.83
[0.00, 154.33] [0.00, 154.54] [0.00, 141.62] [0.00, 151.84]
gpt-5
-miniColPali 13.85 13.26 14.26 13.80
[0.00, 44.10] [0.00, 41.85] [0.00, 48.99] [0.00, 44.56]
ColQwen 24.26 24.46 24.45 24.34
[0.00, 88.52] [0.00, 91.19] [0.00, 90.60] [0.00, 89.54]
ColFlor 21.72 21.81 19.68 21.33
[0.00, 89.65] [0.00, 83.66] [0.00, 75.96] [0.00, 85.56]
gpt-5
-nanoColPali 14.12 14.39 13.64 14.09
[0.00, 48.75] [0.00, 50.35] [0.00, 43.99] [0.00, 48.16]
ColQwen 23.44 24.30 23.57 23.66
[0.00, 77.03] [0.00, 81.46] [0.00, 79.67] [0.00, 78.55]
ColFlor 13.10 13.54 12.45 13.07
[0.00, 44.53] [0.00, 45.47] [0.00, 43.28] [0.00, 44.49]

TABLE VIII
TOKENS PER QUERY ACROSS THEGPT-5MODEL FAMILY WITH
BOOTSTRAPPED95% CIS.
LLM
modelRetrieval
modelEasy
(n=69)Medium
(n=24)Hard
(n=27)Average
(n=120)
gpt-5ColPali 4588.51 4431.44 5037.90 4643.05
[4171.88, 5005.15] [3524.13, 5338.76] [4686.62, 5389.17] [4269.59, 5016.51]
ColQwen 4296.22 4392.33 4633.54 4385.31
[4177.71, 4414.73] [4249.02, 4535.65] [4407.49, 4859.59] [4273.68, 4496.94]
ColFlor 4570.55 4544.65 4878.21 4626.25
[4181.45, 4959.66] [4333.58, 4755.71] [4522.44, 5233.98] [4378.85, 4873.66]
gpt-5
-miniColPali 7764.92 7766.67 7874.39 7787.21
[7212.10, 8317.74] [7101.18, 8432.15] [6572.19, 9176.59] [7112.34, 8462.08]
ColQwen 8827.38 8908.42 9202.65 8920.67
[8681.47, 8973.29] [8821.23, 8995.61] [9031.50, 9373.81] [8867.94, 8973.40]
ColFlor 7604.41 7910.82 7202.44 7592.96
[7498.25, 7710.57] [7880.70, 7940.93] [6778.63, 7626.24] [7562.46, 7623.45]
gpt-5
-nanoColPali 9634.73 10141.63 10439.17 9909.67
[9303.77, 9965.69] [9735.92, 10547.34] [8831.33, 12047.01] [9408.38, 10410.96]
ColQwen 11601.02 11705.74 12255.69 11755.52
[11469.15, 11732.89] [11689.46, 11722.03] [12200.02, 12311.37] [11688.70, 11822.33]
ColFlor 9927.24 10378.53 9941.10 10031.55
[9916.64, 9937.85] [10103.47, 10653.59] [9650.87, 10231.32] [9928.94, 10134.17]
TABLE IX
THROUGHPUT PER QUERY ACROSS THEGPT-5MODEL FAMILY WITH
BOOTSTRAPPED95% CIS.
LLM
modelRetrieval
modelEasy
(n=69)Medium
(n=24)Hard
(n=27)Average
(n=120)
gpt-5ColPali 405.45 415.76 455.99 417.88
[0.00, 3561.15] [0.00, 3761.30] [0.00, 4225.25] [0.00, 3739.00]
ColQwen 353.52 383.96 382.56 366.17
[0.00, 931.45] [0.00, 1005.56] [0.00, 1000.34] [0.00, 960.92]
ColFlor 420.42 390.53 507.63 431.14
[0.00, 3678.67] [0.00, 3355.48] [0.00, 4422.60] [0.00, 3754.74]
gpt-5
-miniColPali 939.25 984.49 1004.85 962.55
[0.00, 2336.21] [0.00, 2395.39] [0.00, 2577.26] [0.00, 2396.77]
ColQwen 726.47 794.62 789.15 754.34
[0.00, 1952.89] [0.00, 2228.31] [0.00, 2144.00] [0.00, 2050.25]
ColFlor 379.56 391.09 409.73 388.19
[0.00, 1495.23] [0.00, 1403.71] [0.00, 1239.65] [0.00, 1423.52]
gpt-5
-nanoColPali 1282.09 1292.56 1278.30 1283.69
[0.00, 3431.41] [0.00, 3398.66] [0.00, 3280.06] [0.00, 3391.69]
ColQwen 849.58 847.58 938.78 866.97
[0.00, 2304.27] [0.00, 2327.62] [0.00, 2580.72] [0.00, 2364.71]
ColFlor 1332.50 1324.49 1495.30 1363.26
[0.00, 3426.45] [0.00, 3408.69] [0.00, 3940.36] [0.00, 3523.50]TABLE X
COST(USD)PER RUN ACROSS THEGPT-5MODEL FAMILY WITH WITH
BOOTSTRAPPED95% CIS.
LLM
modelRetrieval
modelEasy
(n=69)Medium
(n=24)Hard
(n=27)Average
(n=120)
gpt-5ColPali 3.17 1.20 1.21 5.57
[2.88, 3.45] [0.95, 1.44] [1.13, 1.29] [5.12, 6.02]
ColQwen 2.96 1.19 1.11 5.26
[2.88, 3.05] [1.15, 1.22] [1.06, 1.17] [5.13, 5.40]
ColFlor 3.15 1.23 1.17 5.55
[2.89, 3.42] [1.17, 1.28] [1.09, 1.26] [5.26, 5.85]
gpt-5
-miniColPali 1.07 0.42 0.38 1.87
[1.00, 1.15] [0.38, 0.46] [0.32, 0.44] [1.71, 2.03]
ColQwen 1.22 0.48 0.44 2.14
[1.20, 1.24] [0.48, 0.49] [0.43, 0.45] [2.13, 2.15]
ColFlor 1.05 0.43 0.35 1.82
[1.04, 1.06] [0.43, 0.43] [0.33, 0.37] [1.82, 1.83]
gpt-5
-nanoColPali 0.27 0.11 0.10 0.48
[0.26, 0.28] [0.11, 0.11] [0.09, 0.12] [0.45, 0.50]
ColQwen 0.32 0.13 0.12 0.56
[0.32, 0.32] [0.13, 0.13] [0.12, 0.12] [0.56, 0.57]
ColFlor 0.27 0.11 0.10 0.48
[0.27, 0.27] [0.11, 0.12] [0.09, 0.10] [0.48, 0.49]
TABLE XI
PRICE-PER-CORRECT ANSWER(USCENTS)ACROSS THEGPT-5MODEL
FAMILY WITH BOOTSTRAPPED95% CIS.
LLM
modelRetrieval
modelEasy
(n=69)Medium
(n=24)Hard
(n=27)Average
(n=120)
gpt-5ColPali 5.61 5.70 6.05 5.72
[3.21, 8.01] [4.53, 6.86] [5.62, 6.47] [4.14, 7.29]
ColQwen 5.08 5.31 5.85 5.28
[4.82, 5.34] [4.95, 5.67] [5.57, 6.14] [5.08, 5.48]
ColFlor 5.74 5.46 5.72 5.67
[4.90, 6.57] [4.17, 6.74] [3.53, 7.90] [5.36, 5.97]
gpt-5
-miniColPali 1.84 1.88 1.97 1.87
[1.64, 2.04] [1.61, 2.15] [1.29, 2.65] [1.59, 2.15]
ColQwen 2.10 2.13 2.84 2.23
[1.92, 2.29] [1.76, 2.50] [2.11, 3.56] [1.98, 2.47]
ColFlor 1.88 2.03 1.82 1.90
[1.42, 2.33] [2.03, 2.04] [1.71, 1.93] [1.66, 2.14]
gpt-5
-nanoColPali 0.50 0.54 0.69 0.54
[0.42, 0.58] [0.49, 0.59] [0.61, 0.76] [0.48, 0.60]
ColQwen 0.61 0.67 0.91 0.67
[0.57, 0.65] [0.58, 0.75] [0.74, 1.08] [0.62, 0.72]
ColFlor 0.51 0.54 0.71 0.55
[0.45, 0.57] [0.49, 0.60] [0.45, 0.96] [0.48, 0.62]