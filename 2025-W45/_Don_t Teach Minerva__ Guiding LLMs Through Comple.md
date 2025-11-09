# "Don't Teach Minerva": Guiding LLMs Through Complex Syntax for Faithful Latin Translation with RAG

**Authors**: Sergio Torres Aguilar

**Published**: 2025-11-03 11:11:27

**PDF URL**: [http://arxiv.org/pdf/2511.01454v1](http://arxiv.org/pdf/2511.01454v1)

## Abstract
Translating a morphology-rich, low-resource language like Latin poses
significant challenges. This paper introduces a reproducible draft-based
refinement pipeline that elevates open-source Large Language Models (LLMs) to a
performance level statistically comparable to top-tier proprietary systems. Our
method first uses a fine-tuned NLLB-1.3B model to generate a high-quality,
structurally faithful draft. A zero-shot LLM (Llama-3.3 or Qwen3) then polishes
this draft, a process that can be further enhanced by augmenting the context
with retrieved out-context examples (RAG). We demonstrate the robustness of
this approach on two distinct benchmarks: a standard in-domain test set
(Rosenthal, 2023) and a new, challenging out-of-domain (OOD) set of
12th-century Latin letters (2025). Our central finding is that this open-source
RAG system achieves performance statistically comparable to the GPT-5 baseline,
without any task-specific LLM fine-tuning. We release the pipeline, the
Chartres OOD set, and evaluation scripts and models to facilitate replicability
and further research.

## Full Text


<!-- PDF content starts -->

”Don’t Teach Minerva”: Guiding LLMs Through Complex Syntax
for Faithful Latin Translation with RAG
Sergio Torres Aguilar (sertor01@ucm.es)
Abstract
Translating a morphology-rich, low-resource language like Latin poses significant challenges.
This paper introduces a reproducible draft-based refinement pipeline that elevates open-source Large
Language Models (LLMs) to a performance level statistically comparable to top-tier proprietary
systems. Our method first uses a fine-tuned NLLB-1.3B model to generate a high-quality, structurally
faithful draft. A zero-shot LLM (Llama-3.3 or Qwen3) then polishes this draft, a process that can
be further enhanced by augmenting the context with retrieved out-context examples (RAG). We
demonstrate the robustness of this approach on two distinct benchmarks: a standard in-domain test
set (Rosenthal, 2023) and a new, challenging out-of-domain (OOD) set of 12th-century Latin letters
(2025). Our central finding is that this open-source RAG system achieves performance statistically
comparable to the GPT-5 baseline, without any task-specific LLM fine-tuning. We release the pipeline,
the Chartres OOD set, and evaluation scripts and models to facilitate replicability and further research.
1 Introduction
Machine translation (MT) of Latin faces a unique confluence of challenges. The language itself is
inherently difficult, with its rich morphology, complex case-markings, and flexible word order. This
linguistic complexity is then magnified by a vast diachronic landscape: a textual tradition spanning
two millennia has produced an immense diversity of styles, from classical prose to medieval law and
Humanistic neolatin, that no single model can easily master. Yet, the high-quality parallel corpora needed
for training are both scarce and stylistically narrow, often relying on archaic English references that
penalize modern surface-level metrics like BLEU [ 10], while the scholarly practice of permitting multiple
legitimate interpretations makes any single reference a sparse and potentially biased target. While massive
proprietary language models have set a new standard for zero-shot translation, they present their own
challenges regarding reproducibility, cost, and control for specialized scholarly applications that demand
philological precision within budget limitations.
This paper addresses a key question: can a fully open-source and reproducible pipeline achieve per-
formance comparable to these state-of-the-art proprietary systems? We argue that it can, by strategically
combining the strengths of specialized models and general-purpose reasoners. We propose a two-stage
pipeline: a fine-tuned NMT model provides a domain-specialized draft, which a zero-shot Large Language
Model (LLM) then refines, guided by relevant in-context examples retrieved via semantic search. Our
core hypothesis is that this method of providing targeted, inference-time guidance allows open-source
LLMs to be on par with top-tier proprietary models, particularly on challenging, out-of-domain texts.
Contributions.
•We introduce a transparent, two-stage refinement pipeline for Latin-to-English translation. The system
refines drafts from a specialized NLLB model [ 3] with a zero-shot LLM (Llama-3.3 [ 4] or Qwen3
[18]), achieving state-of-the-art results.
•We introduce and evaluate performance on a new, challenging out-of-domain (OOD) test of 12th-
century Latin letters, based on contemporary scholarly translations [6] in a diplomatic style.
1arXiv:2511.01454v1  [cs.CL]  3 Nov 2025

•We demonstrate that our open-source Llama-3.3-70B+RAG system reaches the GPT-5 baseline on the
ID and OOD benchmarks without specific fine-tuning, validating the viability of guided open models
for specialized MT tasks.
•A detailed error taxonomy and qualitative analysis highlighting philological adequacy (e.g., handling
of dative/ablative roles, negation, tense) alongside traditional metrics.
2 Related Work
Latin NLP and resources.While Classical Latin benefits from linguistic resources like treebanks
and lexica (e.g., PROIEL, Perseus/CLTK), Latin–English parallel data remain scarce. Rosenthal [ 14]
assembled 100k pairs spanning Vulgate and Loeb classical texts; OPUS Bible adds 63k pairs [ 2], but
they are largely drawn from 19th-century English references. This creates a domain and style mismatch
for much of the Latin corpus, a challenge our new medieval OOD test set is designed to address. The
diachronic depth of Latin and editorial variance complicate alignment and evaluation using surface-
oriented MT metrics and models that prefer English-like word order, motivating our use of semantic
metrics and detailed qualitative analysis.
Neural MT for Historical Languages.Traditional rule-based pipelines and early statistical systems
offered coverage but struggled with complex syntax [ 17]. Domain-adaptive pre-training (DAPT) [ 7]
can help low-resource MT [ 8], but overfitting to reference style remains a risk. Many established NMT
pipelines rely on seq2seq architectures like NLLB and mBART. More recently, proprietary LLMs have
redefined the state-of-the-art, with models like GPT-4 achieving significant gains in zero-shot translation
over previous NMT systems [ 16]. However, the potential of applying advanced prompting techniques to
open-source LLMs for Latin has remained largely unexplored. Our work fills this gap, demonstrating that
a zero-shot RAG approach could be competitive with a strong proprietary ML baseline.
Retrieval augmentation and translation memory. RAG and translation-memory-style prompting
improve factuality and terminology by injecting context at inference. For MT, retrieval has been shown to
stabilize lexical choice and local syntax, but Latin poses a particularly clause-centric challenge: ablative
absolutes, case-licensed predicates, and relative-clause anchoring. We operationalize retrieval at this
level by pairing a draft NMT (for coverage) with a zero-shot refiner guided by exemplars of semantically
similar source texts and their corresponding NLLB drafts.
Latin MT benchmarks and recent systems.For Latin-English, Rosenthal [ 14] and Fischer et al. [ 5]
established strong NMT baselines with BLEU scores in the low 20s in 2023. Proprietary models soon
pushed this boundary, with GPT-4 reaching a BLEU of 34.5 [ 16] on 16th Latin texts. More recently,
Littera (2025) [ 15] reported exceptionally high scores using a complex, multi-call proprietary pipeline
(12 LLM calls), though its reliance on a tiny, custom test set and massive computational overhead makes
comparison difficult. Our work positions itself differently: we aim for an efficient and fully open-source
system that proves its robustness against a strong, single-call proprietary baseline (GPT-5 [ 9]) across
multiple, distinct benchmarks, including a new OOD set.
3 Methodology
Our approach is centered on a two-stage, retrieval-augmented pipeline designed to leverage the com-
plementary strengths of a specialized NMT model and a general-purpose LLM. This section details the
models used, the pipeline architecture, and the training protocol for our specialized components.
3.1 Model Architecture and Baselines
We evaluate four model families to cover a range of architectures, sizes, and access paradigms, as
summarized in Table 1. Our strategy involves three distinct roles:
2

•A specialized drafter:A fine-tuned NLLB-200-1.3B, a traditional encoder-decoder NMT model,
chosen for its strong multilingual foundation and efficiency.
•Open-source refiners:Decoder-only LLMs of varying scales (Llama-3.3-70B, Qwen3-14B/30B)
used in a zero-shot capacity to preserve their general reasoning capabilities.
•A proprietary benchmark:A single-call API to GPT-5 to contextualize our results against the current
state-of-the-art.
Model Params Arch. Open wts. Multilingual Type Release
NLLB-200-1.3B 1.3B Enc–Dec✓200 seq2seq 2023-05
Llama-3.3-70B 70B Dec-only✓* Strong (8 main) Instruct 2024-12
Qwen3-14B 14B Dec-only✓Strong (+100) Thinking 2025-05
Qwen3-30B 30B†Dec-only✓Strong (+100) Instruct 2025-05
GPT-5 (API)>2T? Undisclosed✗Strong (+100) Thinking 2025-08
Table 1: Model families compared and key properties. * gated download under Meta’s community terms.
†Mixture-of-Experts with approximately 3 billion active parameters per forward pass.
3.2 Two-Stage Retrieval-Augmented Refinement
Stage 1: Draft Generation.Our fine-tuned NLLB-1.3B model first produces an initial, structurally
faithful draft of the source Latin text.
Stage 2: Retrieval and Augmented Refinement.The refiner LLM receives the original Latin and
this NLLB draft ( k= 1 ). For augmentation, we retrieve the top 5 most similar Latin neighbors from our
50M-token corpus and generate their drafts on-the-fly with the same NLLB model ( k >1 ). The final
prompt instructs the LLM to polish the main draft using these five (neighbor, draft) pairs for guidance.
Rationale.This two-stage process is deliberate. The NLLB draft acts as anad verbumanchor,
effectively conditioning the LLM refiner, which better handles instructions and refinement, to a controlled
post-editing task. The final prompt, shown below, instructs the LLM to refine the provided semantically
enriched draft into a single, faithful English translation.
Prompt shape (refiner).The top k=>1 Neighbors are provided as analogical exemplars (Latin +
NLLB draft). The model outputs asingleEnglish line :
System: You are an expert classicist translator. Produce ONE faithful,
English translation. Preserve case roles and polarity. No extra text.
User:
- Revise the Draft Translation to be a more accurate and fluent version of
the Latin source text.
Latin text: <latin>
NMT draft (NLLB): <draft>
Use the Analogous examples for guidance:
[EX1] LATIN: <neighbor_latin>
[EX1] DRAFT: <neighbor_draft>
... (k=5)
Final translation:
3

Figure 1: Overview of our two-stage RAG pipeline. A specialized NLLB model first generates a high-
quality draft. This draft is then refined by a zero-shot LLM, which is augmented with in-context examples
retrieved from a non-parallel corpus via semantic search.
3.3 Training Protocol
NLLB Domain Adaptation.Our only trained component is the NLLB-1.3B drafter, which undergoes
two phases. First, we perform domain-adaptive pre-training (DAPT) using LoRA on our 50M-token raw
Latin corpus. Second, the adapted model is fully fine-tuned on our 160k-pair parallel corpus (Rosenthal
+ OPUS Bible) to specialize it for Latin-to-English translation.
Zero-Shot LLM Refiners.The Llama and Qwen models are used without any task-specific fine-
tuning and no gradient updates. This is a deliberate choice to preserve their generalization capabilities.
3.3.1 Hyperparameters
•DAPT LoRA:rank 16,α=32, attention+MLP, LR2×10−4cosine, bf16, seq len 1024.
•NLLB FT:LR5×10−5, 3 epochs, completion-only loss, standard seq2seq template.
•RAG:k=5neighbors, embeddingBAAI/bge-m3[1], Gen. Temp.0.0(ID),0.0/0.5(OOD).
•Hardware: 2X GPU (RTX A6000 48GB); batches: 128 (NLLB), 16 (Qwen3-14B), 4 (LLama3.3)
•Libraries: Transformers 4.55.1; Peft 0.17.1. 4-bit bitsandbytes.
•Sequences length:input (1100-1300 tokens fork= 1 + 5), output (256)
3.4 Inference Algorithm
We implement retrieval-augmented refinement as a light-weight wrapper around the refiner API:
1.Draft:d←NLLB(x)wherexis the Latin source.
2.Retrieve Neighbors:N←TopKDretrieval(Embed(x), k=5)with lemma Jaccard≥0.3
3.Assemble Context:C← {(LATIN=n i,DRAFT=NLLB(n i))}k
i=1forn i∈N.
4.Refine:ˆy←LLM θ(SystemPrompt, x, d, C).
4

Cost comparison.(10-2025)
•From our usage logs, GPT-5 ( k= 0 ) costs about$1.16per 100 sentences (23k input, 191k output
tokens at $1.25M/$10M rates), or$0.58with batching (asynchronous).
•Local: Llama-3.3-70B+RAG system on a single A6000 GPU runs at$0.15–0.20for the same
workload ( ≈18 minutes), assuming GPU amortization at $0.50/h plus $0.10/kWh, excluding labor.
• Cloud: Assuming similar hardware (A6000 + 10vCPU) at $0.50–0.79 / hr :$0.15–0.24
Thus local inference is roughly3–6 ×cheaperat observed output lengths, though costs remain
sensitive to output size and batching efficiency.
4 Experimental Setup
This section details the datasets, metrics, and settings used to evaluate our pipeline.
4.1 Corpora
•Parallel Corpus ( 160k pairs):A combination of the Rosenthal classical/biblical corpus [ 14] and the
OPUS bible-uedin corpus [2]. This pool is used exclusively to fine-tune the NLLB drafter model.
•Domain-Adaptation Corpus (50M tokens):A collection of non-parallel Latin texts from classical,
medieval, and early modern collections. This corpus serves two critical functions: it is used for the
initial DAPT phase of the NLLB model, and it serves as the retrieval index for our RAG pipeline,
providing a vast pool of semantically rich contexts. (OOD test set is excluded to prevent data leakage).
•In-Domain Test Set (Rosenthal):The standard test split from Rosenthal (2023), stylistically aligned
with the parallel training data.
•Out-of-Domain Test Set (Yves de Chartres):Our new 110-sentence test set (3500 tokens) from
the letters of the bishop Yves de Chartres (1090-1115), using contemporary scholarly translations as
references. [6].
4.2 Evaluation Metrics and Settings
We report standard lexical metrics (SacreBLEU [ 12], chrF++ [ 11]) and semantic metrics (BERTScore [ 19],
COMET-22 [ 13]), both using the multilingual XLM-RoBERTa model). Semantic metrics are better for
Latin, as they capture meaning equivalence even when surface forms diverge from the archaic references.
Besides, COMET is fine-tuned on human ratings of machine translations. For inference, we used greedy
decoding (temperature 0.0), but also explored temperature 0.5 on the OOD set to assess its impact on
fluency. The GPT-5 baseline was prompted with a simple, direct instruction (”Translate the following
Latin text to English:”) and greedy decoding (temp=0.0, top p=1.0). We used the GPT-5-large model
version available in October 2025.
5 Experiments
5.1 Results on the In-Domain Test Set
In-domain test results are summarized in Table 2. As expected, zero-shot LLMs are outperformed by the
specialized, fine-tuned NLLB-1.3B model. The proprietary GPT-5 model sets a high bar, establishing the
state-of-the-art for single-call systems, achieving a COMET score of 73.0 vs. the 70.1 of the baseline.
Our two-stage RAG pipeline, however, decisively improves performance across all open-source
models. The augmentation provides a substantial boost, with Llama-3-70B gaining over 5.5 points
5

System (Rosenthal Test set) BLEU↑chrF++↑METEOR↑BERTScore↑COMET↑Speed↓
Baselines(GPU)
NLLB-1.3B (fine-tuned) 25.61 48.17 0.539 0.920 0.701 1.1
Llama-3.3-70B (zero-shot) 20.50 45.73 0.493 0.917 0.688 15.2
Qwen3-30B (zero-shot) 19.12 43.50 0.473 0.911 0.673 8.2
Qwen3-14B (zero-shot) 19.35 44.01 0.479 0.912 0.674 5.4
Proprietary Models(API)
GPT-5-mini 22.74 47.35 0.524 0.919 0.722 11.5
GPT-5 25.2349.87 0.5480.922 0.730 16.9
Open RAG Systems(GPU)
Llama-3.3-70B + RAG26.5549.860.548 0.925 0.74318.3
Qwen3-30B + RAG 24.17 48.42 0.523 0.923 0.738 11.6
Qwen3-14B + RAG 24.97 48.79 0.532 0.923 0.728 8.6
Table 2: Results on the in-domain Rosenthal test set. Speed is expressed as the time in minutes required
to process 100 items at batch 1. We measure end-to-end latency per sentence, including draft generation,
embedding, ANN retrieval and filtering, neighbor draft generation and LLM decoding.
in both BLEU and COMET, while the smaller Qwen3-14B sees a similar increment in both metrics,
standing as a highly competitive alternative to the much larger and slower 70B model (2,3X slowest). This
confirms the efficacy of providing in-context examples to guide the refinement process. Crucially, our
Llama-3-70B+RAG system emerges as the top-performing model overall, surpassing both the specialized
NLLB baseline and the powerful GPT-5 on all semantic metrics and BLEU.
It is noteworthy that while RAG achieves the highest scores, the BLEU values for all systems remain
in the mid-20s. This likely represents a ceiling imposed by the 19th-century English references. In
contrast, the semantic metrics tell a more revealing story: high BERTScore ( >0.92) and COMET ( >0.72)
values, indicating that the core meaning is being translated accurately, even if the surface form differs.
5.2 Results on the Out-of-Domain Test Set
System (on Chartres OOD Test Set) BLEU↑chrF++↑METEOR↑BERTScore↑COMET↑
Baselines(GPU)
NLLB-1.3B (fine-tuned) 24.04 48.59 0.546 0.918 0.697
Llama-3.3-70B (zero-shot) 28.17 52.68 0.587 0.927 0.714
Qwen3-14B (zero-shot) 26.78 50.95 0.571 0.924 0.710
Proprietary Models(API)
GPT-5 34.05 57.05 0.6410.938 0.760
Our Open RAG Systems(GPU)
Llama-3.3-70B + RAG (temp 0.5)34.68 57.33 0.6460.936 0.757
Llama-3.3-70B + RAG (temp 0.0) 34.28 57.00 0.642 0.934 0.753
Qwen3-14B + RAG (temp 0.0) 32.81 55.28 0.629 0.931 0.746
Qwen3-14B + RAG (temp 0.5) 32.66 55.33 0.625 0.930 0.744
Single augmented System(GPU)
Llama-3.3-70B +k= 1(temp 0.0) 30.28 54.60 0.612 0.930 0.742
Qwen3-14B +k= 1(temp 0.0) 29.15 52.63 0.605 0.928 0.735
Table 3: Results on the out-of-domain Yves de Chartres test set. Is retrieval necessary? Comparing raw
LLM→k=1→full RAG shows that k=1 delivers the bulk of the improvement in semantics (COMET;
BERT), while adding neighbors yields diminishing but positive returns in surface metrics (BLEU, chrF).
6

The OOD results (Table 3) crystallize the core findings of this paper and allow us to dissect the contribu-
tions of our pipeline’s components. The first notable trend is the higher scores across the board, likely
due to the contemporary English references that better match the LLMs’ native style. Here, the zero-shot
LLMs already outperform the fine-tuned NLLB model.
Mirroring the trend observed in in-domain, the refinement component provides incremental on top of
draft-conditioning, but with a key insight: Simply providing the zero-shot Llama-3.3-70B model with the
NLLB draft (the k= 1 condition) boosts its COMET score from 0.714 to 0.742 (+2.8). This single draft
is responsible for the vast majority of the semantic improvement, confirming that the specialized NMT
model provides a powerful structural and semantic anchor.
The retrieval component ( k >1 ) acts as a final polishing step. Adding the five retrieved examples lifts
the COMET score further to a peak of 0.753, adding another 1 point. Notably, this RAG step provides an
even larger boost to lexical metrics like BLEU (+4 points) and METEOR (+3.5 points). This is consistent
with our hypothesis: the draft sets the semantic foundation, while the retrieved examples help the refiner
polish lexical choice and improve semantic fidelity. Ultimately, our full pipeline achieves performance
statistically on par with the GPT-5 benchmark.
Furthermore, the results highlight the value of smaller models within this framework. While the
Qwen3-14B+RAG system does not surpass the GPT-5 baseline, it remains highly competitive, achieving
a COMET score of 0.746. This shows that the RAG methodology is not solely dependent on massive
model scale, effectively elevating more accessible models into a competitive performance tier.
5.3 Ablation: Fine-Tuning vs. Zero-Shot RAG for Generalization
For contrast with zero-shot RAG, we conducted an ablation study. We fine-tuned the Llama and Qwen
models on a small portion (10%) of our parallel corpus and evaluated them on both test sets. As shown
in Table 4: while fine-tuning provides a modest BLEU/COMET in-domain boost, it yields negligible
COMET change, a signature of overfitting on archaic style and vocabulary from the training data. By
losing its broad generalization capabilities, its ability to adapt to the new domain and style of the Chartres
letters is consequently harmed.
This finding supports our draft-based methodology. Instead of forcing the model to fine-tune on a
specific style, our zero-shot RAG approach preserves the LLM’s vast prior knowledge and guides it at
inference time. This makes it a far more robust solution for real-world scenarios across diverse historical
periods and textual genres.
Model ConditionRosenthal (In-Domain) Chartres (OOD)
BLEU↑COMET↑BLEU↑COMET↑
Llama3.3-70BZero-shot 20.50 0.688 28.17 0.712
FT (10% corpus)22.15(+1.7)0.698(+1.0)25.28(-2.9)0.712(-0.01)
Qwen3-14BZero-shot 19.350.673 26.78 0.710
FT (10% corpus)20.67(+1.3)0.688(+1.5)23.92(-2.9)0.708(-0.02)
Table 4: Effect of fine-tuning on a small data portion. Scores on the in-domain test set are improved but
generalization to the OOD set are unchanged, a classic sign of overfitting.
6 Qualitative Analysis
To illustrate the qualitative impact of our pipeline, this section analyzes two challenging translations from
our OOD dataset (Tables 5 and 6). The examples were chosen to demonstrate how our method addresses
distinct linguistic phenomena in Latin that cause predictable errors in base models. We compare the
outputs of Llama3.3 and Qwen3 against a state-of-the-art proprietary benchmark (GPT-5).
7

Example 1: Complex Pronoun Linkage
Source (Latin) inspirante Dei gratia et prosequente, decrevi pati pro lege Dei mei nec ulla ratione cogente volo
ei esse consentaneus in culpa, qui nolo esse consors in poena.
ReferenceThe grace of God inspiring and sustaining me, I have resolved to suffer for the law of my God,
and no reason will compel me to agree in sin with one whose punishment I do not wish to share.
System Prediction Output
Llama3.3-70B Inspired by the grace of God and guided by it, I have decided to suffer for the law of my God
and I do not wish to be in agreement with him in guilt for any reason,I who do not wishto be a
partner in punishment.
Llama3.3-70B
+ RAGBy the inspiration and guidance of God’s grace, I have resolved to suffer for the law of my God,
and without being compelled by any reason, I am unwilling to be a partner in guiltwith whom I
do not wish toshare the punishment.
Qwen3-14B Inspired by the grace of God and pursuing it, I have decided to suffer for the law of my God, and
I do not wish to be in agreement with anyone in guilt,who I do not wishto be a companion in
punishment.
Qwen3-14B
+ RAGInspired by the grace of God and following it, I have determined to suffer for the law of my God,
and not being compelled by any reason,I do not wish to be in agreement in sin with whoI do
not wish to share the punishment.
GPT-5
(Benchmark)With the grace of God inspiring and accompanying me, I have resolved to suffer for the law of
my God; nor, when no reason compels me, do I wish to be in league with him in guilt—whose
punishment I do not wish to share.
Table 5: Comparison of model outputs with and without our RAG augmentation. For each example, we
highlight key phrases where RAG induces significant changes in syntactic or grammatical fidelity.
Example 1: Complex Pronoun Linkage.The source in Table 5 bundles two ablative absolutes
(inspirante . . . prosequente; nec ulla ratione cogente), a dative-anchored predicate (consentaneus + ei),
and a relative that must corefer to that dative (ei . . . qui) within a negated volitional frame. Baselines
struggle with this density: Llama3.3-70B (base) gets the gist but simplifies the coercion template (“for
any reason”) and produces a garden-path relative (“I who do not wish”). Qwen3-14B (base) literalizes
prosequente(“pursuing it”) and repeats the same mis-scoped relative.
With augmentations, Llama3.3-70B correctly restores the coercion template and the “accompany-
ing/guidance” sense ofprosequente; crucially, it resolves the pronominal linkage with a correct syntactic
form (“with whom I do not wish to share the punishment”) avoiding earlier semantic drift. By contrast,
Qwen3-14B + RAG learns the coercion template, but tangles the relative (“in agreement in sin with who I
do not wish to share. . . ”), showing a who/whom error and a redundant preposition.
GPT-5 serves as a fluency benchmark, resolving the linkage with an idiomatic possessive (“. . . whose
punishment. . . ”). Our augmented LLama3.3 output produces an equally correct but more literal rendering
(“. . . with whom. . . ”) closer to the source’s syntactic structure, a quality often preferred for scholarly use.
Example 2: Enforcing Grammatical Fidelity.The second example in Table 6 tests a different cluster:
fidelity to the passive voice (a humility topos withmagis doceri indigeo), a dative governed bycongaudere
(fraternis profectibus), and an additive construction withapponi+ dative (meliores exitus . . . bonis
principiis), while also requiring the correct resolution of the polysemy inprincipiis(“beginnings,” not the
doctrinal “principles”).
The base models stumble in predictable simplifications: both Llama3.3 and Qwen3 switch the passive
infinitive to active (“need to learn”), Llama3.3 readsprincipiisas “principles”, and Qwen3 turns the
additive relation into a causal one (“follow from”).
The augmentations correct these deficits. Both Llama3.3 and Qwen3 restore the passive (“need to
8

Example 2: Passive Voice Fidelity
Source (Latin)Ita dicens, Minervam quidem non doceo, a qua magis doceri indigeo, sed fraternis profectibus
congaudens, bonis principiis meliores exitus apponi desidero.
Reference In speaking thus, I am assuredly not teaching Minerva, by whom I rather need to be taught, but in
rejoicing in the progress of the brethren, I desire for better ends to be joined to good beginnings.
System Prediction Output
Llama3.3-70B Saying this, I do not teach Minerva, from whom I moreneed to learn, but rejoicing in fraternal
progress, I desire better outcomes to be joined to good principles.
Llama3.3-70B
+ RAGThus saying, I do not teach Minerva, from whom I have greaterneed to be instructed, but
rejoicing in the progress of my brothers, I desire that better outcomes be assigned to good
beginnings.
Qwen3-14B So saying, I do not teach Minerva, from whom Ineed to learn more, but being delighted with
the good progress of my brother, I desire that better results may follow from good beginnings.
Qwen3-14B
+ RAGThus speaking, I do not teach Minerva, from whom I have greaterneed to be taught; but
rejoicing in my brothers’ progress, I desire better results to be attached to good beginnings.
GPT-5
(Benchmark)In saying this I do not teach Minerva—by whom I ratherneed to be taught—but, rejoicing in a
brother’s progress, I desire that better outcomes be added to good beginnings.
Table 6: Comparison of model outputs with and without our RAG augmentation. For each example, we
highlight key phrases where RAG induces significant changes in syntactic or grammatical fidelity.
be taught”), keep the dative roles intact, and mapapponito an additive verb targeting a dative goal
(“assigned / attached to good beginnings”). GPT-5 is also correct on passive and additivity; its singular “a
brother’s” is a harmless stylistic choice relative to the Latin plural.
Where the first example showed our pipeline resolving a critical dependency failure, this one demon-
strates its ability to provide the lexical and syntactic scaffolding needed to correct more pervasive patterns
of simplification. With this guidance, both open-source models reach grammatical parity with the bench-
mark on all key features. The remaining differences in verb choice (assigned / attached / added) are again
stylistic rather than philological.
Two Paths to Correctness.While quantitative metrics show benchmark parity, the paths to correct-
ness differ. Across both examples, retrieval consistently stabilizes the parts of Latin that most often fail in
open models: clause scope, case governance, and formulaic patterns, bringing COMET scores in line
with closed models (76.0 vs. 75.7 in OOD). Proprietary models often retain a lead in idiomaticity; while
RAG-augmented outputs tend to preserve the source’s syntactic structure. In other words, our pipeline
does not replace strong closed systems; it complements them by offering a source-faithful alternative at
comparable adequacy, thereby adding stylistic diversity without sacrificing correctness.
7 Discussion
Our results demonstrate that the draft-based refinement pipeline, enhanced by retrieval augmentation
(RAG) can elevate open-source models to a level of performance that is quantitatively on par with,
and qualitatively distinct from, state-of-the-art proprietary systems. This section discusses the broader
implications of these findings.
The Two Paths to Correctness.The central insight from our analysis is that benchmark parity does
not imply stylistic identity. The qualitative comparison reveals that our retrieval augmented models and
GPT-5 follow ”two paths to correctness.” GPT-5 excels at producing dynamic, idiomatic translations that
prioritize fluency for a modern reader. Our pipeline, by contrast, consistently yields translations that
are more structurally faithful to the Latin source. One might interpret the greater structural fidelity of
our augmented outputs as a sign of rigidity compared to GPT-5’s dynamic fluency. However, we argue
9

this is a feature, not a limitation. For scholarly contexts or legal applications, this ”rigidity” translates to
traceability and philological precision, demonstrating that RAG can be used to instill specific, desirable
constraints on a model’s output.
Architectural Synergy and the Role of the Drafter.The success of our pipeline is rooted in
its architectural synergy. The fine-tuned NLLB model acts as a specialized ”navigator,” producing
syntactically plausible drafts that anchor the much larger LLM refiners in the correct semantic space. The
LLM then acts as a general reasoner, applying its vast pre-trained knowledge and linguistic fluency to
polish this draft. Our methodological choice to use NLLB drafts as RAG examples helps the refiner to be
an expert post-editor, specializing it on-the-fly to the specific error patterns and style of the drafter.
The NLLB Draft as an ”Ad Verbum” Anchor.A deeper reason for our pipeline’s success lies
in the NLLB draft’s function as anad verbum(literal) anchor. The fine-tuned NMT model produces
structurally conservative drafts that, while less fluent than a massive LLM, faithfully preserve core Latin
syntax (case roles, clause dependencies). This aligns with the reasoning of Rosu (2025) [ 15]: a literal
first pass constrains the LLM’s interpretative role. The draft provides the syntactic core, while examples
retrieved from our vast diachronic corpus help when surface similarity or terminological consistency
matters. This theory is empirically supported by our OOD results, where the NLLB draft alone accounted
for the majority of the semantic performance gain over the zero-shot baseline.
Limitations and Future Work.This study has several limitations that open avenues for future
research. Our pipeline’s performance is contingent on the quality of the retrieved examples; future work
could explore more robust retrieval, filtering mechanisms, and new ”thinking” capacities of LLMs. While
we demonstrate strong performance on Latin-to-English, testing this architecture on other low-resource or
morphologically-rich languages is a logical next step. Furthermore, the emergence of two distinct, high-
quality translation styles suggests a promising direction for controllable MT, where a user could specify
their desired level of ”idiomaticity” vs. ”fidelity” at inference time. Finally, while our ablation confirms
the value of the draft-based approach over fine-tuning, a more granular analysis, such as a k-sweep to
measure the marginal contribution of each retrieved neighbor, could further clarify the trade-offs between
performance and latency.
8 Conclusion
We addressed the challenge of high-fidelity Latin translation by developing a reproducible, draft-based
refinement pipeline, enhanced with retrieval augmentation. Our analysis reveals that a high-quality initial
draft provides the primary boost in semantic accuracy, while the retrieval of in-context examples acts as a
final polish agent for lexical choice and fluency, together elevating open-source LLMs to quantitative
parity with a top-tier proprietary benchmark. More significantly, our qualitative analysis demonstrates
that our system produces a philologically grounded alternative to the more idiomatic output of proprietary
models. This work validates a viable path toward creating diverse, specialized, and state-of-the-art
translation tools outside of closed ecosystems, offering researchers a meaningful choice in how they
engage with texts about the past.
Ethics Statement
Our system is designed for translating historical text. We acknowledge that historical documents,
including Latin texts, may contain outdated or offensive viewpoints (e.g., colonial or sexist language
in Victorian translations, or propagandistic statements in ancient Roman texts). Our approach focuses
on literal translation and does not aim to filter or alter such content. Users of the system for public-
facing translations should be aware of the potential for problematic content and might need to add
post-processing or content warnings. Finally, by using open-source models and data, we adhere to the
ethos of transparency and reproducibility, and avoid ethical concerns associated with proprietary AI
deployment (such as hidden biases introduced by unseen training data).
10

Reproducibility Statement
We have described the data sources and training procedures in detail. In Section 3, we include hyperpa-
rameter settings, algorithms, and examples of prompts used. The release of models and the LoRA adapter
weights for the models can be found on our Huggingface repository https://huggingface.co/magistermilitum.
The code to deploy our RAG system can be forked from our Gitlab: https://gitlab.com/magistermilitum
References
[1]CHEN, J., XIAO, S., ZHANG, P., LUO, K., LIAN, D.,ANDLIU, Z. Bge m3-embedding: Multi-
lingual, multi-functionality, multi-granularity text embeddings through self-knowledge distillation,
2023.
[2]CHRISTODOULOUPOULOS, C.,ANDSTEEDMAN, M. A massively parallel corpus: the bible in
100 languages.Language resources and evaluation 49, 2 (2015), 375–395.
[3]COSTA-JUSS `A, M. R., CROSS, J., C ¸ELEBI, O., ELBAYAD, M., HEAFIELD, K., HEFFERNAN, K.,
KALBASSI, E., LAM, J., LICHT, D., MAILLARD, J.,ET AL. No language left behind: Scaling
human-centered machine translation.arXiv preprint arXiv:2207.04672(2022).
[4]DUBEY, A., JAUHRI, A., PANDEY, A., KADIAN,ET AL. The llama 3 herd of models.arXiv
e-prints(2024), arXiv–2407.
[5]FISCHER, L., SCHEURER, P., SCHWITTER, R.,ANDVOLK, M. Machine translation of 16th cen-
tury letters from latin to german. InProceedings of the Second Workshop on Language Technologies
for Historical and Ancient Languages(2022), pp. 43–50.
[6]GIORDANENGO, G. Lettres d’Yves de Chartres. https://telma.irht.cnrs.fr/
chartes/yves-de-chartres/, jun, 2017. TELMA.
[7]IYER, V., MALIK, B.,ANDSTEPACHEV,E.A. Quality or quantity? on data scale and diversity in
adapting large language models for low-resource translation. InProceedings of the Ninth Conference
on Machine Translation(2024), pp. 1393–1409.
[8]KOCMI, T.,ANDBOJAR, O. One model to learn them all: Multilingual transfer with sparse mixture
of experts. InEMNLP(2021).
[9]OPENAI. GPT-5 system card. https://cdn.openai.com/gpt-5-system-card.pdf ,
august, 2025.
[10] PAPINENI, K., ROUKOS, S., WARD, T.,ANDZHU, W.-J. BLEU: a method for automatic
evaluation of machine translation. InACL(2002).
[11] POPOVI ´C, M. chrF: character n-gram f-score for automatic mt evaluation. InProceedings of the
tenth workshop on statistical machine translation(2015), pp. 392–395.
[12] POST, M. A call for clarity in reporting bleu scores. InWMT(2018).
[13] REI, R., DESOUZA, J. G., ALVES, D., ZERVA, C., FARINHA, A. C., GLUSHKOVA, T., LAVIE,
A., COHEUR, L.,ANDMARTINS, A. F. Comet-22: Unbabel-ist 2022 submission for the metrics
shared task. InProceedings of the Seventh Conference on Machine Translation (WMT)(2022),
pp. 578–585.
[14] ROSENTHAL, G.Machina cognoscens: Neural machine translation for latin, a case-marked
free-order language. PhD thesis, Master’s thesis, University of Chicago, 2023.
11

[15] ROSU, P. LITERA: An LLM based approach to Latin-to-English translation. InFindings of
the Association for Computational Linguistics: NAACL 2025(Albuquerque, New Mexico, Apr.
2025), L. Chiruzzo, A. Ritter, and L. Wang, Eds., Association for Computational Linguistics,
pp. 7781–7794.
[16] VOLK, M., FISCHER, D. P., FISCHER, L., SCHEURER, P.,ANDSTR ¨OBEL, P. B. LLM-based
machine translation and summarization for Latin. InProceedings of the Third Workshop on
Language Technologies for Historical and Ancient Languages (LT4HALA) @ LREC-COLING-2024
(Torino, Italia, May 2024), ELRA and ICCL, pp. 122–128.
[17] WHITE, J. F. Blitz latin revisited.Journal of Classics Teaching 16, 32 (2015), 43–49.
[18] YANG, A., LI, A., YANG, B., ZHANG, B., HUI, B., ZHENG, B., YU, B., GAO, C., HUANG, C.,
LV, C.,ET AL. Qwen3 technical report.arXiv preprint arXiv:2505.09388(2025).
[19] ZHANG, T., KISHORE, V., WU, F., WEINBERGER, K. Q.,ANDARTZI, Y. BERTScore: Evaluating
text generation with bert. InInternational Conference on Learning Representations(2020).
12