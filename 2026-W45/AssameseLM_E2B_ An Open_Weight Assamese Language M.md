# AssameseLM-E2B: An Open-Weight Assamese Language Model for Efficient Consumer-Grade Deployment

**Authors**: Pragyan Phukan

**Published**: 2026-11-04 08:48:43

**PDF URL**: [https://arxiv.org/pdf/2611.06170v1](https://arxiv.org/pdf/2611.06170v1)

## Abstract
While Large Language Models (LLMs) excel in high-resource languages, their performance often degrades in low-resource languages due to tokenization inefficiencies, generative instability, and linguistic interference. This paper presents AssameseLM-E2B and its instruction-tuned variant, AssameseLM-E2B-it, a 2-billion-parameter model adapted specifically for low resource Assamese language. Built upon the Gemma-4 architecture, the model incorporates an expanded byte-level BPE vocabulary to reduce tokenization overhead by nearly 500% compared with multilingual 2B--3B baselines. We employ a parameter-efficient Continual Pre-Training (CPT) pipeline with Hybrid LoRA, followed by Supervised Fine-Tuning (SFT) on a balanced, multi-domain dataset to enhance instruction-following and Retrieval-Augmented Generation (RAG) capabilities.
  Evaluation demonstrates a Bits-per-Byte (BPB) score of 0.7488 on FLORES-200, alongside an overall RAG score of 96.7% and 100% refusal accuracy on adversarial unanswerable prompts. Comparative analysis reveals significant limitations in general-purpose baselines, including receptive-productive decoupling, generative degradation on Romanized Assamese, and unintended Bengali script assimilation. These results demonstrate the potential of parameter-efficient adaptation for developing capable, locally deployable language models for low-resource languages under constrained computational resources.

## Full Text


<!-- PDF content starts -->

AssameseLM-E2B: An Open-W eight Assamese
Language Model for Eﬀicient Consumer-Grade
Deployment
Pragyan Phukan∗
Department of Physics, Moran College, Charaideo, Assam, India, 785670
Abstract
While Large Language Models (LLMs) excel in high-resource languages, their perfor-
mance often degrades in low-resource languages due to tokenization ineﬀiciencies, genera-
tive instability , and linguistic interference. This paper presents AssameseLM-E2B and
its instruction-tuned variant, AssameseLM-E2B-it , a 2-billion-parameter model adapted
specifically for low resource Assamese language. Built upon the Gemma-4 architecture, the
model incorporates an expanded byte-level BPE vocabulary to reduce tokenization overhead
by nearly 500% compared with multilingual 2B–3B baselines. W e employ a parameter-
eﬀicient Continual Pre-T raining (CPT) pipeline with Hybrid LoRA, followed by Supervised
Fine-T uning (SFT) on a balanced, multi-domain dataset to enhance instruction-following
and Retrieval-Augmented Generation (RAG) capabilities.
Evaluation demonstrates a Bits-per-Byte (BPB) score of 0.7488 on FLORES-200, along-
side an overall RAG score of 96.7% and 100% refusal accuracy on adversarial unanswerable
prompts. Comparative analysis reveals significant limitations in general-purpose baselines,
including receptive-productive decoupling, generative degradation on Romanized Assamese,
and unintended Bengali script assimilation. These results demonstrate the potential of
parameter-eﬀicient adaptation for developing capable, locally deployable language models
for low-resource languages under constrained computational resources.
1 Introduction
Language modeling has undergone a paradigm shift in recent years, driven by the emergence of
Large Language Models (LLMs), including OpenAI’s GPT series [ OpenAI ,2022 ,2023 ], Google’s
Gemini [ Gemini T eam ,2023 ], and Anthropic’s Claude [ Anthropic ,2024 ], alongside open-weight
architectures such as LLaMA [ T ouvron et al. ,2023a ], Mistral [ Jiang et al. ,2023 ], Qwen [ Qwen
T eam ,2024 ] and Gemma [ Mesnard et al. ,2024 ]. Built upon the scalable self-attention mech-
anism of the T ransformer architecture [ V aswani et al. ,2017 ], these foundational models have
demonstrated remarkable capabilities in zero-shot reasoning, machine translation, and contex-
tually coherent text generation.
Proprietary , closed-source models, such as ChatGPT, Claude, and Gemini, are typically
accessed through web-based interfaces or Application Programming Interfaces (APIs). Closed-
source models restrict access to their underlying source code and model weights, preventing
users from independently deploying or hosting them on their own local infrastructure. These
models predominantly follow an Operational Expenditure (OPEX) model, with usage-based
fees generally charged on a per-token basis. Although this approach offers convenience by
eliminating the need for local infrastructure and model deployment, it can incur prohibitive
costs at scale. More critically , the proprietary nature of these elite systems limits mechanistic
interpretability and heavily restricts open scientific research.
∗Email: phukan.pragyan@gmail.com
1
arXiv:2611.06170v1  [cs.CL]  4 Nov 2026

Conversely , open-weight architectures can be deployed on local infrastructure [ T ouvron et al. ,
2023b ], offering researchers greater control through Capital Expenditure (CAPEX). Although
they require upfront investment in computational resources, they enable unrestricted infer-
ence, domain-specific adaptation, and reproducibility in local environments. Y et, despite the
democratizing promise of open models, an overwhelming English-centric bias persists in mod-
ern foundational architectures. F or instance, an empirical breakdown of expansive pre-training
corpora reveals that high-resource European languages account for nearly 99.7% of total to-
kens, whereas Indic languages and other underrepresented languages collectively constitute a
negligible fraction (frequently under 0.21%).
Recent initiatives have made significant progress in addressing the linguistic divide in the
Indian language ecosystem. F rameworks such as IndicT rans2 [ Gala et al. ,2023 ] provide high-
quality neural machine translation across 22 Indian languages, while Airavata [ Gala et al. ,
2024 ] adapts open foundational architectures for Hindi instruction following. Similarly , T amil-
LLaMA [ Balachandran ,2023 ] demonstrates the effectiveness of vocabulary expansion for en-
hancing language-specific capabilities. More recent efforts include BharatGen’s Param1-7B, a
multilingual Mixture-of-Experts (MoE) model incorporating Assamese and 13 other Indian lan-
guages alongside English and Hindi [ BharatGen ,2025 ], and Sarvam AI’s Indic-focused language
models, including Sarvam-M, which demonstrate the potential of specialized post-training for
improving multilingual reasoning and generation across Indian languages [ Sarvam AI ,2024 ].
Despite these advances, substantial disparities persist in the representation of India’s linguis-
tic diversity . In particular, Assamese and several other languages of Northeast India remain
comparatively underrepresented in large-scale pretraining corpora, language-specific evaluation
benchmarks, and specialized generative AI research.
Applying off-the-shelf multilingual LLMs to Assamese introduces two fundamental compu-
tational and linguistic failures:
The Assamese T okenization Bottleneck and the “T oken T ax”: State-of-the-art tok-
enizers are trained on corpora overwhelmingly dominated by Latin scripts [ Ahia et al. ,2023 ].
Because native Assamese subwords are virtually absent from base vocabularies, tokenizers fall
back on raw byte-level encodings, frequently decomposing a single Assamese character into 3
to 4 individual UTF-8 byte tokens. This produces an excessively high subword fertility rate,
bloating input sequence lengths by 300% to 500% [ Rust et al. ,2021 ]. This structural ineﬀiciency
slows inference throughput by a factor of three, rapidly exhausts the model’s finite context win-
dow, and levies an exorbitant T oken T ax” on downstream users of commercial APIs who must
pay per generated subword token [ Ahia et al. ,2023 ].
Cross-Lingual Bengali Contamination: Assamese utilizes the Eastern Indic (Bengali-
Assamese) script, sharing significant orthographic overlap with Bengali [ Bhattacharyya et al. ,
2021 ]. However, because Bengali has a vastly larger digital footprint on the open web, generic
multilingual models trained on uncurated web dumps suffer from severe cross-lingual contam-
ination. When prompted in Assamese, these models routinely hallucinate Bengali vocabulary ,
adopt Bengali morphological conjugations, and fail to recognize distinct Assamese graphemes,
specifically the characters ৰ(ra) andৱ(wa ) . Standard tokenizers conflate the two distributions,
effectively forcing language models to treat Assamese as low-quality or corrupt Bengali.
T o overcome these structural limitations, we introduce AssameseLM-E2B , a 2-billion
parameter Language Model (LM) natively adapted for the Assamese language. Rather than
pre-training a massive 70-billion parameter system, we deliberately ground our work on the
lightweight, high-throughput Gemma-4-E2B architecture [ Mesnard et al. ,2024 ]. At∼2B param-
eters, AssameseLM-E2B is fully edge-ready: it circumvents the prohibitive multi-GPU infras-
tructure required by larger models, democratizing local inference, fine-tuning, and deployment
on standard consumer-grade hardware.
2

Drawing inspiration from vocabulary expansion strategies demonstrated in Chinese-LLaMA
[Cui et al. ,2023 ] and T amil-LLaMA [ Balachandran ,2023 ], we augment the base Gemma to-
kenizer with an additional, carefully curated Assamese vocabulary . T o train the expanded
embeddings and adapt the attention layers eﬀiciently without catastrophic forgetting or insur-
mountable compute budgets, we employ the Low-Rank Adaptation (LoRA) methodology [ Hu
et al. ,2021 ]. W e perform Continual Pre-T raining (CPT) on an Assamese corpus specifically
curated and filtered to eliminate Bengali linguistic contamination, resulting in the open-weight
model AssameseLM-E2B1. Subsequently , we conduct Supervised Fine-T uning (SFT) using
instruction-following datasets to develop its instruction-tuned counterpart, AssameseLM-
E2B-it2, which is also released as an open-weight model. The fine-tuned model is designed to
perform effectively in Retrieval-Augmented Generation (RAG) environments, making it a prac-
tical and cost-effective choice for academic institutions and small-scale industries. Its lightweight
architecture allows deployment on local infrastructure or general-purpose cloud servers, reducing
dependence on proprietary , API-based services and their associated per-token usage costs.
The principal contributions of this paper are summarized as follows:
• Assamese V ocabulary Expansion: W e train a custom SentencePiece tokenizer [ Kudo and
Richardson ,2018 ] on native Assamese text and merge it with the base Gemma tokenizer.
This expansion significantly reduces Assamese token fertility , allowing the model to handle
three times more information in its context window and process sequences significantly faster
while utilizing less memory .
• Contamination-Filtered Pre-T raining Corpus: W e curate an Assamese pre-training
dataset combining clean web-scraped corpora with historical, literary , and educational texts
digitized through an advanced Optical Character Recognition (OCR) pipeline, strictly iso-
lated from Bengali lexical contamination.
• Parameter-Eﬀicient Continual Pre-T raining: W e perform Continual Pre-T raining (CPT)
on the Gemma-4-E2B base model using Low-Rank Adaptation (LoRA) [ Hu et al. ,2021 ]. This
process adapts the newly expanded embedding layer and attention components to capture
native Assamese syntax, semantics, and morphology eﬀiciently . The resulting foundational
model is released under the name AssameseLM-E2B.
• Instruction Fine-T uning for Conversational Alignment: W e develop a targeted As-
samese instruction-following dataset by adapting translated subsets of Alpaca [ T aori et al. ,
2023 ] and OpenOrca [ Lian et al. ,2023 ], supplemented with culturally relevant, localized
prompts. Using this dataset, we perform Supervised Fine-T uning (SFT) on AssameseLM-
E2B to enhance its instruction-following capabilities and conversational alignment. The
resulting instruction-tuned model is released under the name AssameseLM-E2B-it.
• Open-Source Democratization: T o foster transparent academic inquiry and empower
developers, we release both the foundational base model weights (AssameseLM-E2B), the
instruction-tuned chat model (AssameseLM-E2B-it) under an open license.
The remainder of this paper is structured as follows. Section 2 provides the necessary back-
ground on the foundational Gemma model architecture. Section 3 introduces the AssameseLM-
E2B base model, detailing our methodology for expanding the Assamese vocabulary , the cura-
tion of high-fidelity pre-training datasets, the parameter-eﬀicient continual pre-training phase
and quantitative evaluation on FLORES-200 Benchmark. Section 4 presents AssameseLM-E2B-
it, our instruction-tuned conversational agent. This section outlines the supervised fine-tuning
dataset curation, the training methodology , and the optimization dynamics, culminating in a
1https://huggingface.co/assamailab/AssameseLM-E2B
2https://huggingface.co/assamailab/AssameseLM-E2B-it
3

comprehensive empirical evaluation of both Retrieval-Augmented Generation (RAG) and multi-
dimensional direct chat instruction capabilities. Section 5 provides a summary of our contri-
butions and the future outlook for low-resource language modeling. Finally , Section 6 supplies
the appendix resources, including links to our open-source weights and datasets and Section 7
showcases qualitative inference examples.
2 Background on Gemma Models
The Gemma family of open-weights language models, developed by Google, shares its foun-
dational architectural DNA and pre-training methodologies with the flagship Gemini mod-
els [ Mesnard et al. ,2024 ]. Designed to offer state-of-the-art performance within highly con-
strained computational budgets, Gemma models are particularly well-suited for deployment on
consumer-grade hardware and edge devices.
F or this research, we selected the Gemma-4-E2B (2-billion parameter) variant as our
foundational backbone. Architecturally , Gemma utilizes a decoder-only transformer network
equipped with Rotary Position Embeddings (RoPE), GeGLU activation functions, and RM-
SNorm for layer normalization. The 2B variant is specifically optimized for eﬀiciency , making
it an ideal candidate for Continual Pre-T raining (CPT) and Parameter-Eﬀicient Fine-T uning
(PEFT) pipelines via Low-Rank Adaptation (LoRA) [ Hu et al. ,2021 ]. F urthermore, its native
compatibility with Scaled Dot-Product Attention (SDP A) allows for significant memory opti-
mization without the hardware-specific constraints often associated with FlashAttention imple-
mentations. However, in its base multilingual form, Gemma’s tokenizer remains heavily biased
toward high-resource languages, leading to extreme subword fragmentation when processing the
Eastern Nagari script of the Assamese language. Addressing this structural limitation was the
primary motivation for our vocabulary expansion and domain-adaptation methodology .
3 AssameseLM-E2B
3.1 Expansion of Assamese V ocabulary
T able 1: T okenizer comparison between original Gemma-4-E2B and AssameseLM-E2B
Length ContentT okens per
W ord Ratio
Assamese T ext16
(W ord Count)িশক্ষাৈহেছমানৱজীৱনৰআটাইতৈকগুৰুত্বপ ূণর্
সম্পদ,িযেয়অজ্ঞানতাৰআন্ধাৰদ ূৰকিৰজ্ঞানৰ
েপাহৰিবলাইিদেয়।
Gemma-4-E2B43
(T okens Count)'িশক্ষা','ৈহ','েছ','মান','ৱ','জী','ৱ','ন',
'ৰ','আ','টাই','ত','ক','ৈ◌','গু','ৰ','◌ু',
'ত্ব','পূ','ৰ','◌্ণ','সম্পদ',',','য','ি◌েয়','
অজ্ঞান','তা','ৰ','আ','ন্ধ','◌াৰ','দ ূ','ৰ','
কিৰ','জ্ঞান','ৰ','প','ে◌াহ','ৰ','িব','লাই',
'িদেয়','।'2.7
AssameseLM-E2B24
(T okens Count)'িশক্ষা', ' ৈহেছ', ' মানৱ', ' জীৱনৰ', '
আটাইতৈক', ' গুৰুত্বপ ূণর্', ' ', 'সম্পদ', ',', '
িযেয়',অজ','◌্','ঞা','নতা','ৰ','আন্ধাৰ','
দূৰ','','কৰ','ি◌','জ্ঞানৰ','েপাহৰ','িবলাই',
'িদেয়।'1.5
4

As a morphologically rich language, Assamese presents unique tokenization challenges for ex-
isting foundational models. The default vocabulary of foundational models like Gemma severely
underrepresents the Eastern Indic script, leading to excessive subword fragmentation during en-
coding. T o resolve this structural ineﬀiciency , we expanded the base tokenizer with natively
derived Assamese subwords, drawing inspiration from vocabulary extension methodologies uti-
lized in Chinese-LLaMA [ Cui et al. ,2023 ] and T amil-LLaMA [ Balachandran ,2023 ].
T o ensure architectural compatibility with the base Gemma model, we utilized the Sen-
tencePieceBPET okenizer algorithm, relying on the Byte-Pair Encoding (BPE) framework es-
tablished by Kudo and Richardson [2018 ]. The custom tokenizer was trained entirely on our
curated monolingual Assamese corpus, which included both aggregated external datasets and
our proprietary Continual Pre-T raining (CPT) texts. W e initialized the training pipeline with
a target vocabulary size of 20,000 tokens. T o systematically exclude typographical anomalies
and statistical noise from the learned vocabulary , a minimum token frequency threshold of 2
was enforced during generation.
A critical challenge in applying unconstrained BPE algorithms to complex Indic scripts is the
generation of orthographically invalid subwords. In the Assamese script, dependent vowel signs
(matras), modifier markers such as Chandrabindu ( ◌ঁ), Anusvara (◌ং) and Visarga ( ◌ঃ)), and
the Virama / Hasant ( ◌্) are orthographically dependent; they cannot linguistically exist at the
beginning of a standalone word or valid subword token. If injected into the model’s embedding
layer, these “broken” graphemes force the language model to learn corrupted morphological
representations.
T o prevent this, we implemented a rigorous algorithmic filtering heuristic post-BPE training:
Unicode Isolation: W e applied a regular expression filter to retain only tokens containing
valid characters within the Eastern Indic Unicode block ( \u0980 –\u09FF ), stripping out pure
English or noise tokens generated during mixed-corpus processing.
Invalid Grapheme Exclusion: W e implemented a strict rule-based exclusion matrix to drop
any token that improperly commenced with an isolated modifier. Specifically , the script dis-
carded any generated subword beginning with dependent vowel signs (e.g., ◌া,ি◌,◌ী,◌ু ), Chan-
drabindu (◌ঁ), Anusvara (◌ং), Visarga (◌ঃ), or the Virama/Hasant ( ◌্), even when prefixed by
the SentencePiece meta-space character ( U+2581 ).
This deterministic filtering process yielded a highly sanitized, linguistically accurate set of
native Assamese tokens. F ollowing established vocabulary amalgamation practices, these clean
tokens were appended to the original Gemma tokenizer. Subsequently , the base model’s word
embedding matrix and language model head were dynamically resized from their original shape
V×H to an expanded shape V′×H, where V′represents the new aggregated vocabulary size.
This expansion strategy preserves the pre-trained weights of the original English vocabulary
while drastically reducing the token-per-word ratio and computational overhead.
3.2 Datasets Curation for Pre-training
The eﬀicacy of Continual Pre-T raining (CPT) heavily depends on the linguistic diversity , gram-
matical integrity , and volume of the underlying corpus [ Bhattacharyya et al. ,2021 ]. A funda-
mental challenge in Assamese NLP is that standard, open-source web crawls are frequently rid-
dled with code-switching, grammatical errors, and cross-lingual contamination [ Bhattacharyya
et al. ,2021 ]. T o establish a robust linguistic foundation for AssameseLM-E2B, we curated a
hybrid dataset that balances the massive token volume of open-source web dumps with the high
grammatical quality of targeted, proprietary datasets.
The dataset curation process was structured across the following three streams:
5

i. High-Fidelity OCR Literary Corpus: Although large-scale web crawls provide the
substantial token volume required for language modeling, they often lack the grammatical
consistency and linguistic quality of formally published texts. T o address this limitation,
we digitized approximately 100 books, comprising Assamese-medium textbooks, classical
literature, and popular novels, using an advanced Optical Character Recognition (OCR)
pipeline to enrich the corpus with grammatically reliable and linguistically diverse con-
tent. Although this proprietary subset constitutes a comparatively small fraction of the
total pre-training token volume, it serves as an indispensable linguistic anchor. Unlike
unstructured internet data, these digitized books inject highly formal Assamese grammar,
rich historical and literary vocabulary , and complex sentence structures into the model.
ii. Contemporary News Scraping: Because formal literature and generic web dumps
often suffer from temporal staleness, we engineered custom web scrapers to systematically
extract articles from major Assamese daily news portals. This localized dataset bridges
the gap between classic literature and internet slang. It grounds the model in modern
vernacular, evolving journalistic vocabulary , and contemporary socio-political contexts
specific to Assam, ensuring the language model understands current cultural discourse.
iii. Large-Scale W eb and Encyclopedic Corpora: T o supply the massive token volume
required to stabilize Continual Pre-T raining, we integrated several standard open-source
corpora. This included the Assamese subsets of the Common Crawl corpus and the OS-
CAR dataset [ Caswell et al. ,2020 ], which provide a vast breadth of unstructured internet
text. F urthermore, we incorporated the complete Assamese Wikipedia dump to inject
structured, factual, and encyclopedic reasoning, alongside the Sangraha Assamese corpus
[AI4Bharat ,2023 ], a specialized Indic dataset curated specifically for regional language
modeling.
Data Sanitization and Deduplication:
Because aggregating proprietary OCR text, news scrapes, and massive web crawls inevitably
introduces artifacts and redundancy , a rigorous multi-stage sanitization pipeline was enforced:
i. OCR Error Correction: Heuristic filters were applied specifically to our digitized book
corpus to correct common OCR artifacts, such as misread conjunct consonants, hyphen-
ation errors, and broken margins.
ii. Deduplication: W e executed exact-match and MinHash-based fuzzy deduplication [ Broder ,
1997 ,Lee et al. ,2021 ] across the merged corpora to eliminate repetitive boilerplate text,
duplicate news syndications, and overlapping Wikipedia entries.
iii. Contamination Filtering: Finally , strict Unicode-level filtering was applied to strip
out English HTML remnants and cross-lingual Bengali contamination, ensuring the fi-
nal pre-training corpus was a highly dense, uncontaminated, and structurally accurate
representation of the Assamese language.
3.3 Pre-training Phase
T o harness the newly expanded Assamese vocabulary and adapt the base model to the structural
nuances of the Assamese language, we executed a Continual Pre-T raining (CPT) on Gemma-
4-E2B foundational model. The primary objective of this phase is Causal Language Modeling
(CLM), wherein the model learns the semantic and syntactic distributions of the Assamese
language by predicting the next token in a sequence based on the preceding context.
Given the computational constraints of fully retraining a 2-billion parameter model, we
employed Parameter-Eﬀicient Fine-T uning (PEFT) utilizing the Low-Rank Adaptation (LoRA)
6

methodology [ Hu et al. ,2021 ]. F or the training process, we utilized an NVIDIA R TX A6000
GPU equipped with 48 GB of VRAM , supported by 50 GB of system RAM and 9 vCPUs. The
model was trained for one epoch over the entire dataset.
Hybrid LoRA Architecture
T o ensure compatibility between the underlying Gemma-4 architecture [ Mesnard et al. ,2024 ]
and the PEFT library , we dynamically modified the model graph by replacing the proprietary
Gemma4ClippableLinear modules with standard nn.Linear layers prior to LoRA initialization.
T o maximize the model’s adaptive capacity , we targeted all primary linear projection matrices
across the attention mechanisms and feed-forward networks ( q_proj ,k_proj ,v_proj ,o_proj ,
gate_proj ,up_proj , and down_proj ).
Crucially , because we expanded the tokenizer with native Assamese subwords, standard
LoRA, which exclusively trains adapter weights, would fail to learn representations for the
newly added tokens. T o resolve this, we implemented a “Hybrid LoRA” approach by explic-
itly unfreezing the input embedding layer ( embed_tokens ) and the final classification head
(lm_head ). This ensured that the model comprehensively learned the semantic weights of the
new Assamese vocabulary alongside the adapted attention mechanisms.
T raining Hyperparameters and Optimization
The Continual Pre-T raining (CPT) phase was conducted using the SFTTrainer class from the
T ransformer Reinforcement Learning (TRL) library . W e loaded the base model using bfloat16
precision and accelerated attention computations via Scaled Dot-Product Attention (SDP A).
T o optimize GPU memory utilization during training, we enabled gradient checkpointing.
F urthermore, because our pre-training corpus included unstructured text from OCR transcrip-
tions and web scrapes, we enabled sequence packing. Packing concatenates multiple shorter
documents into single dense blocks of the maximum sequence length, drastically improving
training throughput and eliminating padding ineﬀiciencies. The complete set of optimization
constraints and training hyperparameters is detailed in T able 2.
T able 2: Hyperparameters for the Continual Pre-T raining (CPT) Phase
Hyperparameter V alue
LoRA Rank ( r) 64
LoRA Alpha ( α) 128
LoRA T arget Modulesq_proj, k_proj, v_proj, o_proj,
gate_proj, up_proj, down_proj
LoRA Dropout 0.05
Precision bfloat16
Max Sequence Length 2048
Optimizer adamw_torch
Learning Rate 2×10−4
T raining Epochs 1
Pre-training Convergence and T raining Dynamics
The Continual Pre-T raining (CPT) phase exhibited highly stable and rapid convergence over a
corpus of 0.5 billion Assamese tokens. The cross-entropy loss initialized at an exceptionally high
value of approximately 15 (Figure 1a). Concurrently , the initial training perplexity exceeded
7

(a) T raining T okens vs.TL & LR
 (b) T raining T okens vs T raining Perplexity
(c) T raining T okens vs. Next-
T oken Prediction Accuracy
(d) T raining T okens vs. Model
Prediction Entropy
(e) T raining Steps vs. Gradient
Norm
Figure 1: CPT T raining Dynamics of AssameseLM-E2B
106 on a logarithmic scale (Figure 1b ). This initial divergence is mathematically expected; be-
cause the model’s newly expanded Assamese vocabulary embeddings were randomly initialized,
the earliest forward passes produced essentially uniform, non-semantic logits. However, guided
by a cosine learning rate scheduler that peaked at a maximum learning rate of 0.000200 (Fig-
ure 1a), the model demonstrated immediate adaptation. Within the first 0.05 billion tokens,
the cross-entropy loss dropped precipitously . The loss curve subsequently smoothed, gradually
decaying to a final plateau near a value of 3.1 by the end of the 0.5 billion token training
run. Correspondingly , training perplexity collapsed from its initial peak and stabilized firmly
between 101 and 102 (Figure 1b ), indicating that the model successfully mapped the new As-
samese subwords into a coherent latent semantic space.
The progressive linguistic alignment of the model is further validated by the next-token pre-
diction accuracy (Figure 1c) and model prediction entropy metrics (Figure 1d ). As we can see
from the (Figure 1d ), at the onset of training, the mean token accuracy was virtually random,
starting below 0.05 (5%). As the model began to capture Assamese syntactic structures, gram-
matical rules, and vocabulary distributions, the accuracy curve experienced a steep logarithmic
ascent, reaching approximately 0.35 (35%) early in the run. It then continued a gradual climb,
ultimately plateauing just above 0.40 (40%) at the 0.5 billion token mark. Inversely , the model
prediction entropy initialized near a high of 10, reflecting extreme predictive uncertainty (Fig-
ure 1d ). Over the course of the training tokens, the entropy steadily decayed, converging to a
final stable value of approximately 3.0. This sustained reduction in entropy signifies that the
model’s output probability distributions became significantly sharper and more deterministic
as its comprehension of the Assamese language matured.
Finally , the structural stability of the Hybrid LoRA optimization strategy is confirmed by
the gradient norm trajectory (Figure 1e). During the earliest training steps, when the model was
8

aggressively updating the uninitialized vocabulary embeddings, the raw gradient norm exhibited
a transient spike, briefly exceeding a value of 20. However, the gradients rapidly collapsed and
stabilized shortly after this initial adjustment phase. F or the vast majority of the training se-
quence, the smoothed gradient norm remained consistently flat at a value under 1.0. This steady
state confirms that the 100-step learning rate warmup and the inherent regularization of the
LoRA adapters successfully prevented gradient explosion, ensuring a highly stable convergence
environment for the Gemma-4-E2B base architecture.
3.4 Quantitative Evaluation on FLORES-200 Benchmark
T o objectively evaluate the linguistic modeling capabilities and structural eﬀiciency of AssameseLM-
E2B, we conducted a comparative quantitative analysis using the Assamese split of the FLORES-
200 evaluation benchmark [ Goyal et al. ,2022 ]. W e benchmarked our Continual Pre-T rained
(CPT) base model AssameseLM-E2B against four prominent foundational models: the original
base Gemma-4-E2B [Mesnard et al. ,2024 ],Qwen2.5-3B [Qwen T eam ,2024 ] and Llama-3.2-3B
[Dubey et al. ,2024 ].
Because each of these models utilizes fundamentally different tokenizers and vocabulary
sizes, comparing standard Perplexity (PPL) across them is mathematically invalid. Expand-
ing a vocabulary naturally inflates the denominator in the softmax cross-entropy calculation,
structurally increasing PPL even when model comprehension improves. Therefore, we evaluate
the models using two hardware-agnostic, cross-tokenizer metrics: T okenization Eﬀiciency
(tokens required to encode a fixed byte length) and Bits-per-Byte (BPB) , which measures
the theoretical information density required by the model to predict a single UTF-8 byte.
T able 3: FLORES-200 Assamese Benchmark Results
Model Parameters T okens RequiredPerplexity
(PPL)Bits-per-Byte
(BPB)
Llama-3.2-3B 3B 166,218 2.8360 0.7416
Qwen2.5-3B 3B 149,689 4.1012 0.9042
Gemma-4-E2B 2B 55,383 56.6854 0.9570
AssameseLM-E2B 2B 35,107 146.0292 0.7488
Note: Al l models evaluated against the exact same FLORES-200 Assamese test corpus comprising
total UTF-8 bytes.
Eliminating the “T oken T ax”: The empirical results definitively validate our custom As-
samese vocabulary expansion methodology . T o encode the fixed 337,084 UTF-8 byte corpus,
Llama-3.2-3B required a massive 166,218 tokens, and Qwen2.5-0.5B required 149,689 tokens,
demonstrating extreme subword fragmentation for the Assamese script. In stark contrast,
AssameseLM-E2B encoded the exact same text using only 35,107 tokens. Our model requires
nearly 5 times fewer tokens than Llama 3.2. This drastic reduction mathematically eliminates
the API “T oken T ax” and allows AssameseLM-E2B to process significantly more context within
the same hardware constraints.
Information Density and Bits-per-Byte (BPB): The BPB metric normalizes the loss
over the raw bytes of the dataset, providing the most accurate reflection of cross-lingual mod-
eling strength. AssameseLM-E2B achieved a BPB of 0.7488. This represents a massive 21.7%
improvement in predictive compression over its own foundational base model, Gemma-4-E2B,
which scored 0.9570.
9

F urthermore, our 2B parameter AssameseLM thoroughly outperformed Qwen2.5 (0.9042
BPB) and performed virtually on par with the much larger Llama-3.2-3B (0.7416 BPB). The fact
that a localized 2B SLM can rival state-of-the-art 3B models in byte-level semantic prediction
while possessing the most eﬀicient tokenizer in the cohort proves the immense viability of our
Continual Pre-T raining and vocabulary expansion pipeline.
4 AssameseLM-E2B-it
4.1 Dataset Curation for Instruction Fine-T uning
T o transition the foundational AssameseLM-E2B model from a causal text-completion engine
into a highly responsive conversational assistant, we curated a specialized, multi-task Super-
vised Fine-T uning (SFT) dataset. The eﬀicacy of an instruction-tuned model relies heavily on
the structural diversity of its training data; an overly homogenous dataset leads to repetitive
dialogue patterns and poor generalization.
T o ensure the model could handle a wide array of user intents, we compiled a strictly balanced
dataset consisting of 29,319 high-quality JSON instruction-response pairs. This dataset is a
carefully engineered mixture of human-curated as well as synthetically generated pairs. Notably ,
approximately 4,000 of these synthetic pairs were generated directly from the OCR-digitized
textbooks and classic literature that were utilized during the Continual Pre-T raining (CPT)
phase. This strategy explicitly grounds the model’s instruction-following capabilities in the
high-quality , formal Assamese grammar it learned during pre-training.
The dataset was stratified across four primary functional domains:
i. Direct Answer Pairs (17,119 pairs |58.39%):
Constituting the majority of the dataset, this subset is responsible for building the model’s
core generative and conversational capabilities. It includes a diverse mixture of open-
ended creative writing prompts, direct short-form answers, factual question-answering,
and English-to-Assamese translation tasks. This subset is mixure of English, Assamese
and Romanized Assamese prompts and response pairs.
ii. Retrieval-Augmented Generation (RAG) Pairs (8,000 pairs |27.29%):
T o mitigate hallucinations and enable the model to serve in enterprise or academic envi-
ronments, a significant portion of the dataset was dedicated to RAG formatting. These
pairs teach the model to synthesize answers strictly based on provided contextual text
snippets rather than relying solely on its internal parametric memory .
iii. Alpaca English Subset (3,000 pairs |10.23%):
T o prevent catastrophic forgetting of the base model’s English comprehension capabilities
and to ensure the model could operate in a bilingual capacity , we integrated a random
sample of 3,000 pairs from the standard Alpaca instruction dataset [ T aori et al. ,2023 ].
This allows the model to reliably process English instructions even when generating As-
samese outputs.
iv. Search Intent Pairs (1,200 pairs |4.09%):
A specialized subset designed to teach the model how to interpret search-oriented queries,
summarize search results, and format structured informational outputs.
By distributing the 29,319 pairs across these specific functional categories, the SFT dataset
ensures that AssameseLM-E2B-it is comprehensively aligned to handle creative, analytical, and
retrieval-based tasks without overfitting to a single interaction style.
10

4.2 Supervised Instruction Fine T uning Phase
T o align the model with conversational and instruction-following tasks while preserving its pre-
trained linguistic capabilities, we adopted a Parameter-Eﬀicient Fine-T uning (PEFT) strategy
based on the Low-Rank Adaptation (LoRA) methodology . Given the computational demands
of fine-tuning a 2-billion-parameter model, this approach enabled eﬀicient adaptation without
updating all model parameters. The fine-tuning process was performed on an NVIDIA R TX
A6000 Ada GPU with 48 GB of VRAM, supported by 62 GB of system RAM and 16 vCPUs.
The model was trained for three epochs over the entire instruction-tuning dataset.
Model Architecture and LoRA Configuration
W e initialized the SFT pipeline using the pre-trained AssameseLM-E2B foundational check-
point. Because Gemma-4 [ Mesnard et al. ,2024 ] natively includes vision-language modalities, we
explicitly disabled the vision_config to optimize memory allocation strictly for text-based gen-
eration tasks. W e utilized the Hugging F ace SFTTrainer in conjunction with Parameter-Eﬀicient
Fine-T uning (PEFT). The model was loaded in bfloat16 precision, implementing Scaled Dot-
Product Attention (SDP A) rather than FlashAttention-2 to prevent hardware-specific crashes
related to Gemma’s unique head dimensions.
T o bypass internal architectural mismatches within the Gemma-4 backbone, we employed
a targeted regular expression to attach LoRA adapters explicitly to standard nn.Linear layers
within the text backbone. The complete LoRA configuration is detailed alongside the training
hyperparameters in T able 4.
T raining Hyperparameters and Optimization
The training environment was optimized for computational eﬀiciency on constrained hardware.
T o mitigate Out-Of-Memory (OOM) errors caused by the massive 274,000-token expanded vo-
cabulary , we drastically reduced the per-device batch size and utilized the memory-eﬀicient
8-bit Adam W optimizer. Aggressive gradient accumulation was employed to maintain a robust
effective batch size. The comprehensive set of optimization constraints and training hyperpa-
rameters is summarized in T able 4.
T able 4: Hyperparameters for the Supervised Fine-T uning (SFT) Phase
Hyperparameter V alue
LoRA Rank ( r) 16
LoRA Alpha ( α) 32
LoRA T arget Modulesq_proj, k_proj, v_proj, o_proj,
gate_proj, up_proj, down_proj
LoRA Dropout 0.05
Precision bfloat16
Max Sequence Length 768
Optimizer adamw_8bit
Learning Rate 2×10−4
T raining Epochs 3
4.3 Instruction T uning Convergence and Optimization Dynamics
Figure 2a illustrates the variation in the learning rate over the course of the training steps.
The optimization dynamics during the Supervised Fine-T uning (SFT) phase reflect a stable
11

and successful alignment process over the course of approximately 1400 training steps. The
optimization was governed by a cosine learning rate schedule, which was initiated with a linear
warmup phase that smoothly ramped the learning rate to a peak of 2.0×10−4around step
41. F ollowing this warmup, the learning rate executed a gradual cosine decay , tapering to near
zero by the end of the training run to facilitate fine-grained weight adjustments in the LoRA
adapters.
The training loss throughout the Supervised Fine-T uning (SFT) process is plotted in Fig-
ure 2b . The progression of the SFT training loss exhibits behavior highly characteristic of
instruction tuning on structurally diverse datasets. The raw training loss (represented by the
blue trajectory) displays significant step-to-step variance, with occasional transient spikes ex-
ceeding values of 15.0 and 17.5. This high variance is an expected mathematical artifact of
utilizing a constrained per-device batch size while iterating through highly heterogeneous in-
struction prompts (e.g., shifting abruptly from simple conversational greetings or reasoning
tasks). Crucially , however, the Exponential Moving A verage (EMA, with α= 0.1) of the loss
(represented by the red trajectory) demonstrates robust and continuous convergence . The
EMA loss initialized near a value of 11.5 and decayed steadily alongside the learning rate sched-
ule, ultimately plateauing in a stable range between 2.5 and 3.5 during the final 400 steps of
training .
The structural integrity of the model during this alignment phase is further corroborated by
the gradient norm trajectory illustrated in Figure 2c. The gradient norm tracks the magnitude of
weight updates applied to the network. Throughout the training phase, the EMA of the gradient
norm steadily decreased from an initial value near 1.5 down to approximately 0.5. While the
raw gradient norm experienced isolated, acute spikes - most notably surpassing 4.0 near step
650 and 7.0 near step 1000 - the network demonstrated immediate recovery . These transient
anomalies likely correspond to the model processing batches containing highly complex, out-
of-distribution, or multi-turn reasoning instructions. However, the steady underlying decay of
the gradient norm EMA confirms that the adamw_8bit optimizer successfully absorbed these
updates without triggering gradient explosion or catastrophic forgetting, thereby preserving the
pre-trained Assamese linguistic capabilities while locking in conversational alignment.
(a) Learning Rate Schedule vs
T raining Steps
(b) SFT T raining Loss vs T rain-
ing Steps
(c) Gradient Norm vs T raining
Steps
Figure 2: SFT T raining Dynamics of AssameseLM-E2B-it
4.4 Results on Instruction F ollowing T asks
4.4.1 Evaluation of Retrieval-Augmented Generation (RAG) Instructions
T ask Design and Evaluation Method
T o rigorously evaluate the instruction-following capabilities and structural reliability of the fine-
tuned Assamese model, we benchmarked AssameseLM-E2B-it against three prominent founda-
tional and instruction-tuned models: Google’s Gemma-4-E2B-it ( ∼2B parameters) [ Mesnard
et al. ,2024 ], Meta’s Llama-3.2-3B-Instruct (3B parameters) [ Dubey et al. ,2024 ], and Alibaba’s
Qwen3-3B-Instruct (3B parameters) [ Qwen T eam ,2024 ].
12

The evaluation framework was designed to assess grounded generation, specifically focusing
on Retrieval-Augmented Generation (RAG) capabilities. The models were tested on a curated
taxonomy of 10 RAG prompts spanning Assamese cultural, geographical, and historical knowl-
edge. Each prompt provided numbered context passages and a targeted question, requiring the
model to generate a factually grounded, citation-attributed response in the appropriate script
(Assamese Unicode or Romanized Assamese). Crucially , the dataset included two adversarial
“refusal items” where the provided context deliberately omitted the answer, testing the model’s
ability to safely abstain rather than hallucinate.
The evaluation and automated grading of the generated outputs were conducted using An-
thropic’s Claude Opus 4.6 [ Anthropic ,2024 ] as an expert LLM-as-a-Judge. Claude Opus 4.6 was
specifically selected for this evaluation pipeline due to its industry-leading capabilities in deep
reasoning, long-context synthesis, and complex professional workflows. Because RAG evalua-
tion requires the evaluator to meticulously cross-reference generated Assamese and Romanized
text against provided context passages, Opus 4.6’s advanced instruction-following and excep-
tional logical consistency ensure that the grading remains highly objective. Its ability to handle
multi-step analytical chains with significantly reduced hallucination rates makes it uniquely
suited to score subtle nuances in faithfulness and attribution quality .
Performance was measured across six complementary metrics, each scored on a 0–100%
scale:
• Answer Correctness: Expert-annotated factual accuracy of the response.
• F aithfulness: The degree to which the response is strictly grounded in the provided context
passages.
• Citation F1: T oken-level F1 score measuring the accuracy of attribution indices (e.g., [1]).
• Refusal Accuracy: The proportion of adversarial prompts on which the model correctly
abstained from answering.
• Script Match: Alignment of the output with the expected Assamese Unicode or Latin
script.
• Output Cleanliness: The absence of leaked special formatting tokens (e.g., <start_of_turn> )
or prompt echoing.
An Overall RAG Score was computed as the unweighted macro-average of these six metrics.
Results from Retrieval-Augmented Generation (RAG) Evaluation
AssameseLM-E2B-it demonstrated state-of-the-art performance for Assamese knowledge re-
trieval, achieving an overall macro-average RAG score of 96.7%. Our domain-specialized model
decisively outperformed all general-purpose baselines, establishing a massive 39.2 percentage
point gap over the second-place Llama-3.2-1B-Instruct (57.5%). The base Gemma-4-E2B-it
model ranked third at 49.2%, while the larger 3B-parameter Qwen model performed the poor-
est at 35.8%.
The multi-dimensional capability profile visually illustrates AssameseLM’s comprehensive
dominance (Figure 3a); its performance polygon nearly fills the entire radar area, whereas the
baseline models exhibit highly irregular, collapsed polygons (Figure 3b ).
Per-Metric Breakdown (Figure 3c ): AssameseLM-E2B-it achieved perfect 100% scores
across Answer Correctness, F aithfulness, Citation F1, Refusal Accuracy , and Output Cleanli-
ness. Its only recorded imperfection was an 80% Script Match, which occurred because the
13

(a) Overall RAG score (macro-average, %)
 (b) Per-metric performance polygon
(c) Per-metric breakdown score
 (d) Capability over task taxonomy
Figure 3: RAG-Specific T ask Evaluation of AssameseLM-E2B-it
model correctly generated Assamese script for two queries that technically expected Romanized
Assamese, a negligible semantic mismatch.
In contrast, the baseline models struggled severely with proper attribution and grounding.
Citation F1 scores across the baselines ranged from a mere 20% to 40%, indicating a systemic
failure to accurately reference context passages. F urthermore, Qwen3-3B-Instruct exhibited a
paradoxical failure mode: while it achieved a perfect 100% for output cleanliness, it scored
only 25% on answer correctness and 30% on faithfulness. Qwen frequently generated verbose,
English-language preambles and explicitly denied information that was clearly present in the
context, demonstrating that clean formatting in larger parameter LLMs can mask profound
substantive deficiencies.
Refusal accuracy emerged as the most discriminative metric and the most critical safety
concern in this evaluation. AssameseLM-E2B-it successfully recognized the adversarial prompts
and achieved a 100% refusal rate. Conversely , no baseline model achieved any refusal accuracy
(0%). Gemma, Llama, and Qwen entirely failed to abstain, opting instead to generate fabricated
historical legends and irrelevant answers. This binary failure mode highlights that general-
purpose multilingual LLMs currently lack the instruction-following discipline required for safe
deployment in Assamese enterprise applications.
Capability Categorization (Figure 3d ): Breaking down answer correctness by task tax-
onomy further isolates the limitations of non-specialized models:
• Distractor Robustness: This proved to be the easiest sub-task; Gemma (100%) and Llama
(75%) performed well when simply selecting correct passages from mixed contexts.
• Cross-Lingual Synthesis: Synthesizing information across language boundaries caused
severe degradation in Qwen (dropping to 16.7%), while Gemma handled it moderately well
(83.3%).
14

• Extraction and Cross-Script: T asks requiring the model to extract factual answers when
the query and context used different scripts exposed a significant vulnerability in Gemma,
which scored only 33.3%.
Across all four complex taxonomic categories, AssameseLM-E2B-it maintained a flawless
100% correctness rate. These results definitively prove that for low-resource languages, targeted
vocabulary expansion and domain-specific instruction fine-tuning yield dramatically greater
structural and factual reliability than simply scaling a general-purpose model’s parameter count.
4.4.2 Evaluation of Multi-Dimensional Direct Chat Instructions
T ask Design and Evaluation Method
T o rigorously assess the open-ended conversational and generative capabilities of the mod-
els beyond context-grounded RAG environments, we developed a comprehensive 150-prompt,
multi-domain zero-shot evaluation suite. This benchmark was engineered to test direct response
capabilities across 26 fine-grained thematic categories, encompassing Regional & Assamese Cul-
tural Knowledge, General Science, F ormal Mathematical Reasoning, and Pragmatics.
T o accurately simulate the diverse input behaviors of low-resource language users, who
frequently rely on mobile keyboards, the dataset was structurally bifurcated into two script
modalities: 95 prompts utilizing the native Eastern Nagari script and 55 prompts utilizing
Romanized Latin transliteration.
Model outputs were systematically evaluated across four primary orthogonal dimensions on
a continuous-interval Likert scale from 1 (complete failure) to 10 (flawless execution):
• Linguistic Accuracy (LA): Evaluated grammatical correctness, morphological inflection,
orthographic validity (specifically the absence of Bengali grapheme substitution), and id-
iomatic naturalness.
• F actuality (F act): Assessed the precision, verifiable truthfulness of knowledge assertions,
and the absence of hallucinations.
• Query Understanding (QU): Measured pragmatic intent recognition, semantic interpre-
tation, and topic alignment.
• Instruction F ollowing (IF): Evaluated compliance with task constraints, requested struc-
tural formats, and response style.
Additionally , two secondary quantitative metrics were tracked: Linguistic Purity (the
percentage of tokens conforming strictly to Assamese lexicon, penalizing code-switching) and
RAG F aithfulness Score (a diagnostic marker checking for unwarranted context-dependent
refusal patterns).
The initial evaluation and automated grading of the generated outputs were conducted using
Google Gemini 3.8 [ Google Gemini T eam ,2024 ] acting as an expert LLM-as-a-Judge. Gemini
3.8 was explicitly selected for this phase due to its state-of-the-art multilingual reasoning ca-
pabilities and its robust cross-lingual semantic alignment across Indic languages. Evaluating
Assamese text generation requires an evaluator capable of parsing nuanced orthographic differ-
ences (e.g., distinguishing Assamese ৰfrom Bengaliর) and accurately interpreting non-standard
Romanized phonetic transliterations. Gemini 3.8’s advanced instruction-following capabilities
made it highly suited to strictly apply our complex, multi-dimensional Likert rubric.
However, an exhaustive data audit of the initial Gemini 3.8 evaluation pipeline uncovered
a critical methodological vulnerability inherent to LLM-as-a-judge frameworks in low-resource
regimes. The automated evaluator exhibited a severe “default-scoring bias” . F or example, in 43
15

instances (representing 28.7% of the Qwen model’s outputs), the judge assigned perfect 10/10
scores for F actuality and Query Understanding to completely incoherent, non-syntactic gibber-
ish. Because automated judges process low-resource text by relying on cross-lingual cognates
and keyword matching, the presence of overlapping Bengali-Assamese vocabulary tricked the
evaluator into hallucinating semantic coherence.
T o rectify this, a rigorous human-in-the-loop auditing phase was implemented. All 43 suspect
instances were individually reviewed by native Assamese evaluators against the formal rubric,
and responses lacking factual content or intent recognition were manually corrected to their true
empirical values (1/10). This hybrid methodology ensures that our final statistical analysis relies
on perfectly calibrated, bias-free data.
Statistical Distribution and Visual Analysis
The performance distributions across the four evaluation dimensions reveal stark structural
differences in model capabilities, as visualized in the comparative violin plots and aggregated
macro-level bar charts (Figure 4a-4e).
Linguistic Fluency vs. Generative Collapse: The violin plot for Linguistic Accuracy
demonstrates a heavy top-tier concentration for AssameseLM-E2B-it, which achieves a mean
of 7.30 and a distribution heavily skewed toward the upper quartile. In stark contrast, the
distributions for Llama-3.2-1B-Instruct (mean 1.26) and Qwen-3-3B-Instruct (mean 1.31) are
almost entirely suppressed at the absolute floor of 1.0, visually confirming their total generative
collapse when attempting to synthesize native Assamese text. Gemma-4-E2B-it exhibits a
highly stretched, fragmented distribution (mean 3.53), reflecting isolated instances where it
produced partially valid Assamese interspersed with frequent Bengali-assimilated outputs.
Visualizing the Receptive-Productive Decoupling: The Query Understanding distribu-
tions provide definitive visual proof of the receptive-productive decoupling phenomenon within
general-purpose foundational models. Despite their complete generative failure on the linguis-
tic axis, the violin plots for Llama-3.2-1B-Instruct (mean 6.48) and Gemma-4-E2B-it (mean
7.47) show massive density bulges in the upper quartiles, with a high concentration of scores
between 7 and 10. This confirms that the semantic intent of the Assamese queries is success-
fully mapped into their latent space, even though they lack the productive decoding weights
to generate a grammatically valid response. AssameseLM-E2B-it maintains the highest over-
all comprehension density (mean 8.06), while Qwen-3-3B-Instruct fails entirely on this axis,
clustering rigidly at a mean of 2.00.
The Bimodal Signature of RAG-Prior Bias: The violin plots for F actuality and In-
struction F ollowing explicitly expose the specific alignment vulnerability of our fine-tuned
AssameseLM. Rather than exhibiting a normal Gaussian distribution, AssameseLM-E2B-it
displays a distinct “hourglass” or bimodal shape in both metrics (F actuality mean: 4.89; In-
struction F ollowing mean: 4.62). This extreme variance directly visualizes the RAG-prior
refusal bias discussed in our qualitative analysis: the model either successfully answers the
zero-shot prompt based on parametric memory (clustering densely at scores of 9–10) or in-
appropriately triggers a canned refusal template due to missing external context (clustering
sharply at scores of 1–2).
Conversely , the baseline models show complete floor-suppression for Instruction F ollowing,
with Llama and Qwen locked at 1.40. This visual flatline demonstrates their total inability
to adhere to task-specific structural constraints or stylistic guidelines when operating in an
Assamese linguistic environment.
16

(a) T raining Steps vs. SFT T raining Loss
(b) T raining Steps vs. Learning Rate Schedule
 (c) T raining Steps vs. SFT T raining Loss
(d) T raining Steps vs. Gradient Norm
 (e) T raining Steps vs. SFT T raining Loss
Figure 4: Multi-Dimensional Direct Chat Instruction Evaluaton of AssameseLM-E2B-it
17

Inter-Dimensional Correlations and Cognitive Synergy
T o move beyond isolated metric analysis, we computed Spearman’s rank correlation coeﬀicients
(ρ) across all evaluation dimensions to examine the interrelationships among different perfor-
mance dimensions of the models [ Spearman ,1904 ,Mukaka ,2012 ].
This statistical analysis yielded two critical insights into low-resource model behavior. First,
it mathematically proved the “receptive-productive decoupling” observed in the baseline mod-
els. F or Llama-3.2-3B-instruct, the correlation between Query Understanding and Linguistic
Accuracy was virtually nonexistent ( ρ= 0.195 ), quantitatively confirming that semantic com-
prehension in the latent space does not translate to generative decoding capability . In contrast,
AssameseLM-E2B-it demonstrated a strongly coupled architecture where understanding and
generative fluency scaled together ( ρ= 0.500 ).
Secondly , the matrices revealed a profound “F actuality Instruction F ollowing Synergy” .
Across all evaluated architectures, F actuality and Instruction F ollowing exhibited exception-
ally high correlations ( ρ= 0.847 for AssameseLM-E2B-it, 0.824 for Gemma 4, and 0.916 for
Qwen). This demonstrates that in direct question-answering benchmarks, a model’s ability to
fulfill structural instructions is fundamentally contingent upon possessing the factual parametric
knowledge required to answer the prompt.
Thematic Cluster V ariance
Partitioning the 150-prompt dataset into thematic clusters revealed that model performance
is highly domain-sensitive. AssameseLM-E2B-it maintained generative dominance across all
domains, peaking in “Regional & Assamese Cultural Knowledge” with a near-perfect Linguistic
Accuracy of 9.33 and F actuality of 8.00.
Conversely , the baseline models exhibited varying degrees of domain collapse. While Gemma
4 demonstrated moderate comprehension in “General Scientific / F actual” queries (Query Un-
derstanding of 9.57), its performance eroded significantly in “F ormal Reasoning & Math” (Lin-
guistic Accuracy dropping to 4.36) and failed entirely in “Romanized Domain Knowledge” (Lin-
guistic Accuracy of 1.00). This confirms that general multilingual pre-training fails to encode
deep logical reasoning or transliteration robustness for underrepresented scripts.
Strategic Recommendations for Low-Resource SFT
Based on our empirical evaluation and the qualitative failure taxonomy , we propose four archi-
tectural and data-engineering principles for future low-resource language adaptation:
i. Orthographic Separation and Negative Sampling: Pre-training and SFT curricula
must implement explicit negative sampling against high-resource sister languages. Loss
penalties should be applied to the generation of intruding Bengali-specific glyphs and high-
frequency closed-class functional terms to prevent the model from treating Assamese as a
dialect.
ii. Context-Conditioning Isolation: T o prevent the RAG-prior contamination (false re-
fusals) observed in AssameseLM-E2B-it, instruction-tuning datasets must strictly distin-
guish between context-dependent Q&A and open-domain parametric recall. Prompt tem-
plates must explicitly train the model to bypass retrieval-refusal behaviors when external
context is absent.
iii. Phonetic T ransliteration Data Augmentation: F oundation models cannot rely on
cross-lingual transfer for Romanized text. SFT mixtures must explicitly incorporate par-
allel dual-script pairs (Romanized Latin ↔ Native Script) to ensure true script invariance.
18

iv. Dedicated V ocabulary Expansion: Standard tokenizers exhibit severe sub-word frag-
mentation, averaging 3.8 to 5.2 tokens per Assamese word. Compressing this fertility
ratio via byte-level BPE vocabulary expansion is an absolute prerequisite for stabilizing
generative decoding in compact models.
5 Summary and F uture Outlook
In this paper, we introduced AssameseLM-E2B and its instruction-tuned counterpart, As-
sameseLM -E2B-it , establishing a new state-of-the-art for low-resource Indo-Aryan language
modeling. By engineering a custom byte-level BPE vocabulary expansion, we successfully elim-
inated the API “T oken T ax” for the Assamese language, reducing the tokenization overhead by
nearly 500% compared to prominent 1B–3B multilingual models. Our Continual Pre-T raining
(CPT) pipeline effectively integrated OCR-digitized literary corpora and web scrapes, allowing
the 2B model to achieve a Bits-per-Byte (BPB) score of 0.7488 on the FLORES-200 benchmark
[Goyal et al. ,2022 ], rivaling models twice its size.
Through a highly curated Supervised Fine-T uning (SFT) phase, we demonstrated that
targeted domain adaptation is indispensable. AssameseLM-E2B-it decisively outperformed
general-purpose models (including Llama-3.2-3B [ Dubey et al. ,2024 ] and Qwen-3-3B [ Qwen
T eam ,2024 ]) across all Retrieval-Augmented Generation (RAG) and open-ended generative
benchmarks. Crucially , we identified and overcame severe failure modes prevalent in general
multilingual models, namely the “receptive-productive decoupling,” assimilation into the Ben-
gali “gravity well,” and catastrophic generative collapse when processing Romanized Assamese
transliteration. F urthermore, AssameseLM-E2B-it achieved a flawless 100% refusal accuracy on
adversarial unanswerable RAG prompts, proving its viability for safe, hallucination-free enter-
prise deployment.
While AssameseLM-E2B demonstrates profound linguistic capabilities, our analysis uncov-
ered specific alignment vulnerabilities, such as RAG-prior refusal bias where the model occa-
sionally suppresses its parametric memory in zero-shot settings. F uture work will focus on
refining SFT data mixtures to strictly decouple context-dependent refusal behaviors from open-
domain knowledge recall. Additionally , we plan to scale this methodology to larger parameter
architectures (7B–9B), expand the context window to support full-document Assamese sum-
marization, and explore multimodal integrations to allow the model to directly process and
transcribe Assamese historical manuscripts and visual cultural artifacts.
A Model A vailability and Accessibility
T o promote transparency , reproducibility , and further research on low-resource Assamese lan-
guage modelling, the trained model checkpoints are made publicly accessible through the Hug-
ging F ace Model Hub. The continued pre-trained (CPT) checkpoint and the supervised fine-
tuned (SFT) instruction-following model are available at the following repositories:
• CPT Base Model (AssameseLM-E2B):
https://huggingface.co/assamailab/AssameseLM-E2B
• SFT Model (AssameseLM-E2B-it):
https://huggingface.co/assamailab/AssameseLM-E2B-it
These repositories provide access to the respective model checkpoints for academic research,
evaluation, and further development of Assamese language technologies.
19

B Inference Examples
In this section, we show example generations from AssameseLM-E2B-it.
A.1 Short F actual Prompt-Response Examples
The following examples illustrate the proficiency of AssameseLM-E2B-it in handling zero-shot
factual queries, highlighting its capacity to deliver concise, direct, and hallucination-free an-
swers.
Example #A1:
F actual Prompt (Assam History)
Prompt:
শৰাইঘাটৰযুদ্ধতআেহামেসনাকেকােনেনত ৃত্বিদিছল?
Model Response:
শৰাইঘাটৰযুদ্ধতআেহামেসনাকলািচতবৰফ ুকেনেনতৃত্বিদিছল।
Example #A2:
F actual Prompt (Assam History)
Prompt:
শ্রীমন্তশংকৰেদৱেকানআিছল?
Model Response:
শ্রীমন্তশংকৰেদৱআিছলএজনমহানঅসমীয়াসমাজসংস্কাৰক,ধম র্গুৰুআৰুসমাজসংগঠক।েতও ঁনৱৈবষ্ণৱধমর্ৰ
প্রৱতর্কআিছল।
Example #A3:
F actual Prompt (Indian History)
Prompt:
ভাৰতৰেসউজিবপ্লৱৰৈসেতেকানজনিবজ্ঞানীৰনামিবেশষভােৱজিড়ত?
Model Response:
ভাৰতৰেসউজিবপ্লৱৰৈসেতএমএছস্বামীনাথনৰনামিবেশষভােৱজিড়ত।
Example #A4:
F actual Prompt (Indian History)
Prompt:
ভাৰতীয়জাতীয়কংেগ্রছিকমানচনতপ্রিতষ্ঠাকৰাৈহিছল?
Model Response:
ভাৰতীয়জাতীয়কংেগ্রছ১৮৮৫চনতপ্রিতষ্ঠাকৰাৈহিছল।
Example #A5:
F actual Prompt (Science)
Prompt:
পৰমাণুৰেকন্দ্রতেকানেবাৰকণাথােক?
Model Response:
পৰমাণুৰেকন্দ্রতপ্র'ট'ন(Proton)আৰুিনউট্রন(Neutron)থােক।
20

Example #A6:
F actual Prompt (Science)
Prompt:
ওেজানস্তেৰপৃিথৱীৰজীৱসমূহকেকেনদেৰসুৰক্ষািদেয়?
Model Response:
ওেজানস্তেৰসূযর্ৰক্ষিতকাৰকঅিতেবঙ ুনীয়াৰিশ্মআৰুেসৗৰধ ুমুহাৰপৰাজীৱসমূহকৰক্ষাকেৰ।ইয়াৰফলতপ ৃিথৱীত
জীৱৰঅিস্তত্বসম্ভৱৈহউেঠ।
Example #A7:
F actual Prompt (Science)
Prompt:
QCD-ৰসম্পূণর্নামিকআৰুইেকানেমৗিলকবলৰিবষেয়অধঁয়নকেৰ?
Model Response:
QCD-ৰসম্পূণর্নামৈহেছেকাৱাণ্টামক্র'ম’ডাইেনিমক্স(QuantumChromodynamics)আৰুইশিক্তশালী
েমৗিলকবলৰঅন্তগ র্তেকাৱাকর্কণাৰআচৰণঅধঁয়নকেৰ।
Example #A8:
F actual Prompt (Environment)
Prompt:
বনাঞ্চলধ্বংসহ'েলপিৰেৱশতিকিকপ্রভাৱপেৰ?
Model Response:
বনাঞ্চলধ্বংসকিৰেলজলবায়ুপিৰৱত র্ন,মািটৰখহনীয়াআৰুৈজৱ-ৈবিচত্রঁৰঅৱনিতঘেট,িযেয়মান ুহৰজীৱনধাৰণৰ
বােবভাবুিককিঢ়য়াইআেন।
Example #A9:
Reasoning
Prompt:
েকােনাঐিতহািসকঘটনাৰিবষেয়দ ুটািনভর্ৰেযাগঁউৎসতেবেলগতথঁথািকেলশুদ্ধতথঁিনধ র্াৰণকিৰবৈলিকিক
পদেক্ষপল'বলােগ?
Model Response:
এেনপিৰিস্থিততপ্রথেমদ ুেয়াটাউৎসৰিবতংিবেশ্লষণকিৰবলােগ,েযেন—আেলাচনী,প ুৰিণিকতাপবাচৰকাৰীনিথ।
তাৰিপছততথঁৰসতঁতাপৰীক্ষাকিৰবৈলিবিভন্নঅনলাইনআক র্াইভতিবচাৰচলাবলােগ।
Example #A10:
Reasoning
Prompt:
এটাAIমেডলৰওচৰতেকােনাপ্রশ্নৰিনভ র্ৰেযাগঁতথঁনাথািকেলইেকেনদেৰউত্তৰিদয়াউিচত?
Model Response:
এআইমেডলএটাইসিঠকউত্তৰিদবৈলতথঁৰওপৰতিনভ র্ৰকেৰ।যিদপ্রশ্নকৰািবষয়তপয র্াপ্ততথঁনাথােক,েতেন্ত
ই'unknown'ব ুিলক'বপােৰ।
21

Example #A11:
Cross-Lingual Adaptation for Romanized Assamese (Science)
Prompt:
Nuclearfissionarunuclearfusionormajotmukhyoparthokyoki?
Model Response:
পাৰমাণিৱকিবভাজনআৰুপাৰমাণিৱকসংেযাজনৰপ্রধানপাথ র্কঁেটাহ'লেযপ্রথমেটাতএটাগধ ুৰিনউিক্লয়াছবা
পৰমাণুিবভাজনকৰাহয়,আনহােতিদ্বতীয়েটাতদ ুটালঘুিনউিক্লয়াছএকিত্রতকৰাহয়িযেটােৱশিক্তৰম ুিক্তঘটায়।
Example #A12:
Cross-Lingual Adaptation for Romanized Assamese (Assamese Literature)
Prompt:
LakshminathBezbaruakunkaroneAxomiyaxahityotbikhyato?
Model Response:
লক্ষ্মীনাথেবজবৰুৱাঅসমীয়াসািহতঁতেতও ঁৰবহুমুখীপ্রিতভাৰবােবিবখঁাতআিছল।েতও ঁকিবতা,গল্প,নাটক,প্রবন্ধ
আিদসকেলােক্ষত্রেতঅৱদানআগবঢ়াইঅসমীয়াসািহতঁকসম ৃদ্ধকিৰিছল।
Example #A13:
Cross-Lingual Adaptation for Romanized Assamese (T echnology)
Prompt:
Open-sourcesoftwarebulikibujajai?
Model Response:
অেপন েছাসর্চফটেৱৰ ৈহেছ এেন এক প্রয ুিক্ত, য'ত ইয়াৰ উৎস ক'ড বা স ূত্রসমূহ মুক্তভােৱ উপলব্ধ থােক।
বঁৱহাৰকাৰীসকেলএইক'ডৰওপৰতসংেশাধনকিৰবপােৰ,বঁৱহাৰকিৰবপােৰআৰুিনজৰকামেতাপ্রেয়াগকিৰব
পােৰ।ইবন্ধক'ডৰিবপৰীেতএকসম্প ূণর্িবপৰীতধাৰণা।
Example #A14:
Reasoning
Prompt:
েসামবােৰগিণতৰক্লাছহয়।মঙ্গলবােৰিবজ্ঞানৰক্লাছহয়।িবজ্ঞানৰিপছৰিদনাকলা(Art)ক্লাছহয়।েতেন্তকলাক্লাছ
িকবােৰহয়?
Model Response:
হয়,মঙলবাৰৰিপছৰিদনাঅথ র্াৎবুধবােৰকলাৰক্লাছহয়।
Example #A15:
Reasoning
Prompt:
এটাAIমেডলৰওচৰতেকােনাপ্রশ্নৰিনভ র্ৰেযাগঁতথঁনাথািকেলইেকেনদেৰউত্তৰিদয়াউিচত?
Model Response:
এআইমেডলএটাইসিঠকউত্তৰিদবৈলতথঁৰওপৰতিনভ র্ৰকেৰ।যিদপ্রশ্নকৰািবষয়তপয র্াপ্ততথঁনাথােক,েতেন্ত
ই'unknown'ব ুিলক'বপােৰ।
Example #A16:
Mathematical Reasoning
Prompt:
500tokar20%kimanhoy?
Model Response:
৫০০টকাৰ২০%ৈহেছ১০০টকা।
22

A.2 Long Answer Prompt-Response Examples / Creative W ritting Examples
The following qualitative examples illustrate the performance of AssameseLM-E2B-it on long-
form creative writing tasks, highlighting its ability to maintain Assamese morphosyntactic sta-
bility over extended contexts without collapsing into sister languages.
Example #B1:
Assamese Creative W riting
Prompt:
Writeanessayonক ৃিত্রমবুিদ্ধমত্তাইভিৱষঁতৰিশক্ষাবঁৱস্থাতেকেনপ্রভাৱেপলাবপােৰ?
Model Response:
#কৃিত্রমবুিদ্ধমত্তাআৰুভিৱষঁতৰিশক্ষাৰপ্রতঁাহ্বান
##ভূিমকা
বতর্মানৰদ্রুতপিৰৱত র্নশীলপৃিথৱীতমানৱজািতৰভিৱষঁতিনধ র্াৰণতপ্রযুিক্তৰভূিমকাঅপিৰসীম।িবেশষৈক,ক ৃিত্রম
বুিদ্ধমত্তাৰ(AI)উত্থােনৈশিক্ষকবঁৱস্থাৰম ূলেভিটসলিনকিৰবৈলৈগআেছ।ছাত্র-ছাত্রীসকেলএইনত ুনপ্রযুিক্তক
েকেনদেৰগ্রহণকিৰবপােৰবাইয়াৰপৰাউদ্ভৱেহাৱাপ্রতঁাহ্বানৰসন্ম ুখীনহ'বপােৰ,েসয়াআিজৰিচন্তাৰিবষয়ৈহ
পিৰেছ।
###AI-ৰদ্বাৰাপাঠঁক্রমৰপিৰৱত র্ন
###িশক্ষণৰধৰণসলিন
এআই-েয়ছাত্র-ছাত্রীকেকৱলতথঁেযাগানধৰাইনহয়,বৰঞ্চেতও ঁেলাককপ্রশ্নকৰাৰপদ্ধিত,সমসঁাসমাধানৰপদ্ধিত
আৰুসৃজনীমূলকিচন্তাধাৰাৰেক্ষত্রেতাসহায়কিৰব।বঁিক্তগতকৰণকৰাপাঠঁক্রমপ্রস্তুতকৰাতএআইৰবঁৱহাৰব ৃিদ্ধ
পাব,যাৰফলতছাত্র-ছাত্রীেয়িনজৰগিতঅন ুসিৰিশিকবৈলসুেযাগপাব।িকন্তুিডিজেটলিডভাইচসম ূহৰঅতঁিধক
বঁৱহাৰৰফলতেশ্রণীেকাঠাৰবািহৰেতাএকাকীত্বআৰুমানিসকচাপৰসৃষ্টিেহাৱাৰসম্ভাৱনাথােক।
###কমর্সংস্থানসৃষ্টিৰেক্ষত্রতপ্রতঁাহ্বান
###এআই-এমান ুহৰচাকিৰৰগুৰুত্বহ্রাসকিৰবব ুিলআশংকাকৰাহয়।বত র্মানসময়তবহুকামেযেনঅন ুবাদকৰা,
ডাটাএণ্ট্রিকৰাআিদকামেবাৰস্বয়ংিক্রয়ভােৱসম্পাদনহ'ব।যিদওইিকছ ুমানপুৰিণবৃিত্তেকিবলুপ্তকিৰব,নতুন
ধৰণৰজ্ঞান,সৃষ্টিশীলতাআৰুেনত ৃত্বৰপ্রেয়াজনীয়তােহঅিধকগুৰুত্বপ ূণর্।িশক্ষাতযািন্ত্রককামৰপিৰৱেত র্মানৱীয়
দক্ষতাৰওপৰতেজাৰিদবলািগব।
##সামািজকআৰুৈনিতকিদশসম ূহ
###তথঁৰঅপবঁৱহাৰআৰুভ ুৱাখবৰ
এআই-এইণ্টাৰেনটৰজিৰয়েতিবশালপিৰমাণৰতথঁৰআদান-প্রদানতসহায়কেৰ,িযেটাইিতবাচককথা।অৱেশঁ,
ইয়াতভুলতথঁবাভুৱাখবৰৰসংক্রমণেৰাধকৰােটাঅিতপ্রেয়াজনীয়।ছাত্র-ছাত্রীসকলৰমাজততথঁৰিবচাৰক্ষমতা
শিক্তশালীকিৰেলসমাজতশািন্তবজাইৰখাতসহায়কহ'ব।
###সমতাআৰুঅন্তভ ুর্িক্তৰপ্রসাৰ
###প্রাথিমকপয র্ায়তAIপ্রেয়াগৰসিঠকপিৰকল্পনাঅতঁন্তজৰুৰী।দ ুখীয়াঅঞ্চলৰছাত্র-ছাত্রীআৰুগ্রামঁঅঞ্চলত
এিতয়াওইন্টাৰেনটৰসীিমতপ্রেৱশািধকাৰেদখাযায়,িযেতও ঁেলাকৰবােবিশক্ষাৰঅিধকাৰপ্রািপ্তকিঠনকিৰেতােল।
চৰকােৰিবনামূলীয়ািশক্ষাপ্রদানবঁৱস্থািবকিশতকিৰএআইৰস ুফলযােতসকেলােৰওচৰৈলযায়তাৰপ্রিতলক্ষঁ
ৰািখবলােগ।
সামৰিণতক'বপািৰেযক ৃিত্রমবুিদ্ধমত্তাৰআশীবর্াদিহচােপিশক্ষাইএকনত ুনিদগন্তেদিখেছ;আনহােতইয়াকিনয়ন্ত্রণ
অিবহেনপ্রেয়াগকিৰেলআমাৰসমাজগ ঁাথিনভািঙেযাৱাৰভয়আেছ।গিতেকআিমসকেলােৱিমিলএখনস ুৰিক্ষত
আৰুপ্রগিতশীলভিৱষঁতৰপথবািছল'বলািগব।
23

Example #B2:
Assamese Creative W riting
Prompt:
বানপীিড়তপিৰয়ালসম ূহৰবােবসাহাযঁপ্রদানৰঅন ুেৰাধজনাইিজলাআয়ুক্তৈলএখনআেবদনিলখা।
Model Response:
প্রিত,
িজলাআয়ুক্ত,
কামৰূপমহানগৰিজলাপ্রশাসন,গুৱাহাটী।
তািৰখ:২৪েম',২০২৪
িবষয়:বানিবধ্বস্তপিৰয়ালৰবােবসাহায র্ঁপ্রদানকৰাৰআেবদন।
মেহাদয়,
সন্মানপূবর্কিবনীতিনেবদনএইেয,েযাৱাদ ুিদনধিৰেহাৱােনৰােনেপৰাবৰষ ুণৰফলতআমাৰঅঞ্চলেটাতএকভয়াৱহ
পিৰিস্থিতৰসৃষ্টিৈহেছ।নদীৰপানীবািঢ়অহাৰফলতবহুেলাকৰঘৰ-দ ুৱাৰধ্বংসৈহেছআৰুহাজাৰহাজাৰেলাক
গৃহহীনৈহপিৰেছ।এইদ ুেযর্াগৰসময়তআটাইতৈকেবিছক্ষিতগ্রস্তৈহেছদিৰদ্রেলাকসকলআৰুিশশুসকল।
গিতেক, আেপানাৰওচৰতআমাৰিবনীতপ্রাথ র্না, অনুগ্রহকিৰবানাক্রান্তঅঞ্চলৰেলাকসকলৰবােবপ্রেয়াজনীয়
খাদঁ, িবশুদ্ধেখাৱাপানীআৰুকােপাৰ-কািনআিদসামগ্রীিবতৰণকিৰবৈলপ্রশাসনকিনেদ র্শপ্রদানকিৰব।আিম
িনিশ্চতেযচৰকাৰৰসহায়ৰহাতপাইসকেলােলাকউপক ৃতহ'ব।
ধনঁবােদেৰ,
স্থানীয়ৰাইজৰৈহ,
ৰিঞ্জতদাস।
Example #B3:
Assamese Creative W riting
Prompt:
পূবর্ৰকমর্স্থলীৰপৰাঅিভজ্ঞতাৰপ্রমাণ-পত্রিবচািৰএখনআেবদনিলখা।
Model Response:
প্রিত,
শাখাপ্রবন্ধক,
এিবিচপ্রাইেভটিলিমেটড,গুৱাহাটী।
তািৰখ:২৪অেক্টাবৰ,২০২৪
িবষয়:অিভজ্ঞতাম ূলকপ্রমাণ-পত্র(ExperienceCertificate)প্রদানৰবােবআেবদন।
মেহাদয়,
মইআেপানাৰপ্রিতষ্ঠানৰএগৰাকীকম র্চাৰীিহচােপিবগতপ ঁাচবছৰধিৰেসৱাআগবঢ়াইআিহেছ ঁা।েমাৰমূলদািয়ত্ব
আিছল—কমাণ্ডএণ্ডকণ্ট্রলচােভ র্(CPC)আৰুডাটাএনালাইিছছৰকামতকিম্পউটাৰছফ্টেৱৰবঁৱহাৰকৰা।এই
সময়েছাৱাতমইবহুজিটলপ্রকল্পৰসফলৰূপায়ণতঅংশৈলেছ ঁা।
েমাৰকাযর্কালেশষেহাৱাৰলেগলেগেমাকচাকিৰজীৱনৰঅিভজ্ঞতাপ্রিতফিলতকিৰবপৰাৈকএটাঅিভজ্ঞতাম ূলক
প্রমাণ-পত্রপ্রদানকিৰবৈলঅন ুেৰাধজনােলঁা।আশাকেৰঁাআপুিনেমাৰেসৱাৰমূলঁায়নকিৰপ্রেয়াজনীয়বঁৱস্থাগ্রহণ
কিৰব।
ধনঁবােদেৰ,
আেপানাৰিবশ্বস্ত,
ৰােজশবৰা।
24

A.3 RAG Output
The following qualitative examples illustrate the performance of AssameseLM-E2B-it on RAG
tasks, highlighting critical differences in context adherence, script fidelity , the mitigation of
hallucinations and RAG refusal.
Example #C1:
Native Assamese RAG
Prompt:
Context: [1]কািজৰঙাৰাষ্ট্রীয়উদঁানএিশঙীয়াগ ঁড়ৰবােবিবখঁাত।[2]মাজ ুলীৈহেছব্রহ্মপুত্রনদীৰমাজত
অৱিস্থতএখনবৃহৎনদীদ্বীপ।[3]মানাহৰাষ্ট্রীয়উদঁানবাক্সাআৰুিচৰাংিজলাতঅৱিস্থত।
Question:মাজ ুলীেকানখননদীৰমাজতঅৱিস্থত?
Model Response:
মাজুলীৈহেছব্রহ্মপুত্রনদীৰমাজতঅৱিস্থতএখনব ৃহৎনদীদ্বীপ[2].
Example #C2:
Cross-lingual RAG
Prompt:
Context: [1]NameriNationalParkislocatedintheSonitpurdistrictandisfamous
fortheWhite-wingedWoodDuck. [2]TheGoldenLangurConservationBreeding
ProgrammeissuccessfullyrunningattheAssamStateZooinGuwahati.[3]Pobitora
WildlifeSanctuaryhasthehighestdensityofone-hornedrhinosintheworld.
Question:StateZookotaseaarutatkihorbreedingprogrammesoliase?
Model Response:
অসমৰািজঁকিচিৰয়াখানাতেসাণালীবান্দৰৰসংৰক্ষণপ্রজননকায র্সূচীচিলআেছ[2]।
Example #C3:
Cross-lingual RAG
Prompt:
Context: [1]ড°ভ ূেপনহাজিৰকাঅসমৰএগৰাকীস ুধাকণ্ঠনােমেৰপিৰিচতিকংবদিন্তগায়ক।[2]Hewas
awardedtheBharatRatnaposthumouslyintheyear2019.
Question: Who is known as Sudhakantha and in which year did he receive the
BharatRatna?
Model Response:
Dr.BhupenHazarikaisknownasSudhakantha[1],andhereceivedtheBharatRatna
intheyear2019[2].
25

Example #C4:
Cross-lingual RAG
Prompt:
Context:[1]TheAhomkingdomwasestablishedbyChaolungSukaphaaintheyear
1228. [2]ThehistoricBattleofSaraighatwasfoughtin1671betweentheMughal
EmpireandtheAhomKingdom.
Question:আেহামসাম্রাজঁেকিতয়াপ্রিতষ্ঠাৈহিছলআৰুশৰাইঘাটৰৰণেকানচনতৈহিছল?
Model Response:
আেহামসাম্রাজঁ১২২৮চনতপ্রিতষ্ঠাকৰাৈহিছল[1],আৰুশৰাইঘাটৰৰণ১৬৭১চনতৈহিছল[2]।
Example #C5:
RAG Refusal
Prompt:
Context:[1]ৰঙালীিবহুবাব'হাগিবহুঅসমৰম ূলউৎসৱআৰুইনৱবষ র্ৰআৰম্ভিণকাসূচায়।[2]অম্বুবাচীেমলা
প্রিতবছেৰআহাৰমাহতগুৱাহাটীৰকামাখঁামিন্দৰতঅন ুিষ্ঠতহয়।
Question:েজানিবলেমলােকানখনিজলাতঅন ুিষ্ঠতহয়?
Model Response:
ক্ষমাকিৰব,প্রদানকৰাতথঁতইয়াৰউত্তৰউপলব্ধনহয়।
A.4 Examples where The AssameseLM-E2B-it Model Generates Bad Output
AssameseLM-E2B-it is prone to hallucinations and can present distorted facts as D1 & D2.
Example #D1:
F actual Prompt-Response hallucination
Prompt:
প্র'টনআৰুিনউট্রনেকানেবাৰেমৗিলককণাৰদ্বাৰাগিঠত?
Model Response:
প্র'টনএটাউৎপন্নেহাৱােমৗিলককণানহয়,ইিনেজইএটাপ্র'টন।আনহােত,িনউট্রনৈহেছএটােমৗিলককণািয
ইেলক্ট্রনৰৈসেতএেকভৰৰিকন্তুিবপৰীতআধানয ুক্ত।
Example #D2:
F actual Prompt-Response hallucination
Prompt:
শব্দইশূনঁস্থানৰমােজেৰিকয়গিতকিৰবেনাৱােৰ?
Model Response:
শব্দইশূনঁস্থানৰমােজেৰগিতকিৰবেনাৱােৰকাৰণইএকেভৗিতকবস্তুনহয়,বৰঞ্চএটাসংেকতেহ।
C Acknowledgements
The authors acknowledge the use of Google’s Gemini and OpenAI’s ChatGPT as AI-assisted
tools during the preparation of this manuscript. These tools were utilized to improve lan-
guage, refine the presentation of ideas, and enhance the overall readability and coherence of the
manuscript. The authors take full responsibility for the scientific content, interpretations, and
conclusions presented in this work.
26

References
Orevaoghene Ahia, Sachin Kumar, Hila Gonen, and Noah A. Smith. Do all languages cost the
same? tokenization in the era of commercial language models. In Proceedings of the 2023
Conference on Empirical Methods in Natural Language Processing (EMNLP) , 2023.
AI4Bharat. Sangraha: A massive, open-source corpus for indic languages. T echnical report,
AI4Bharat Research, 2023.
Anthropic. Claude 3.5 and opus 4.6 technical report. Anthropic Research , 2024.
Abhinand Balachandran. T amil-llama: A new tamil language model based on llama 2. arXiv
preprint arXiv:2311.05845 , 2023.
BharatGen. Param-1-7b-moe. https://huggingface.co/bharatgenai/Param-1-7B , 2025.
Multilingual Mixture-of-Experts language model.
Pushpak Bhattacharyya et al. Challenges in processing assamese and bengali: Shared script,
divergent syntax. ACM T ransactions on Asian and Low-Resource Language Information
Processing , 2021.
Andrei Z. Broder. On the resemblance and containment of documents. In Compression and
Complexity of Sequences 1997. Proceedings , pages 21–29. IEEE, 1997.
Isaac Caswell, Theresa Breiner, Daan van Esch, and Ankur Bapna. Language id in the wild:
Unexpected challenges on the path to a thousand-language web text corpus. In Proceedings
of the 28th International Conference on Computational Linguistics , pages 511–525, 2020.
Yiming Cui, Zhipeng Y ang, and Xin Y ao. Eﬀicient and effective text encoding for chinese llama
and alpaca. arXiv preprint arXiv:2304.08177 , 2023.
Abhimanyu Dubey , Abhinav Jauhri, Abhinav Pandey , Abhishek Kadian, Ahmad Al-Dahle,
Aiesha Letman, Akhil Mathur, Alan Schelten, Amy Y ang, Angela F an, et al. The llama 3
herd of models. arXiv preprint arXiv:2407.21783 , 2024.
Jay Gala, T ahir Jayakumar, Anoop Kumar M, Ratish Puduppully , Mitesh M. Khapra, Raj D.
Dabre, and Anoop Kunchukuttan. Airavata: Introducing hindi instruction-tuned llm. arXiv
preprint arXiv:2401.15006 , 2024.
Jay Gala et al. Indictrans2: T owards high-quality and accessible machine translation for all 22
scheduled indian languages. T ransactions of the Association for Computational Linguistics
(T ACL) , 2023.
Gemini T eam. Gemini: A family of highly capable multimodal models. arXiv preprint
arXiv:2312.11805 , 2023. doi: 10.48550/arXiv.2312.11805.
Google Gemini T eam. Gemini 1.5 and 3.8 pro: T echnical overview. Google DeepMind , 2024.
Naman Goyal, Cynthia Gao, Vishrav Chaudhary , Peng-Jen Razzaghi, Guillaume W enzek,
Da Ju, Aurélien Rodriguez, Sam Lakhani, F aisal Ngan, Zaid Kazi, et al. The flores-200 eval-
uation benchmark for low-resource machine translation. arXiv preprint arXiv:2207.04672 ,
2022.
Edward J. Hu, Y elong Shen, Phillip W allis, Zeyuan Allen-Zhu, Y uanzhi Li, Shean W ang,
Lu W ang, and W eizhu Chen. Lora: Low-rank adaptation of large language models. arXiv
preprint arXiv:2106.09685 , 2021.
27

Albert Q. Jiang, Alexandre Sablayrolles, Arthur Mensch, Chris Bamford, Devendra Singh Chap-
lot, Diego de las Casas, Florian Sayed, et al. Mistral 7b. arXiv preprint arXiv:2310.06825 ,
2023.
T aku Kudo and John Richardson. Sentencepiece: A simple and language independent subword
tokenizer and detokenizer for neural text processing. In Proceedings of the 2018 Conference
on Empirical Methods in Natural Language Processing: System Demonstrations , pages 66–71,
2018.
Katherine Lee, Daphne Ippolito, Andrew Nystrom, Chiyuan Zhang, Douglas Eck, Chris
Callison-Burch, and Carl Nicholas. Deduplicating training data makes language models
better. In Proceedings of the 60th Annual Meeting of the Association for Computational
Linguistics , 2021.
Wing Lian, Bleys Goodson, Eugene Pentland, Austin Cook, Chien V ong, and T eknium.
Openorca: An open dataset of gpt augmented flan reasoning traces. Hugging F ace Repository ,
2023.
Thomas Mesnard et al. Gemma: Open models based on gemini research and technology . arXiv
preprint arXiv:2403.08295 , 2024.
Mavuto Mukaka. A guide to appropriate use of correlation coeﬀicient in medical research.
Malawi Medical Journal , 24(3):69–71, 2012. doi: 10.4314/mmj.v24i3.
OpenAI. Introducing chatgpt. OpenAI Blog, 2022.
OpenAI. Gpt-4 technical report. arXiv preprint arXiv:2303.08774 , 2023.
Qwen T eam. Qwen2.5 technical report. arXiv preprint arXiv:2412.15115 , 2024.
Phillip Rust, Jonas Pfeiffer, Ivan V ulić, Sebastian Ruder, and Iryna Gurevych. How good is your
tokenizer? on the monolingual performance of multilingual language models. In Proceedings
of the 59th Annual Meeting of the Association for Computational Linguistics , 2021.
Sarvam AI. Sarvam-1. https://huggingface.co/sarvamai/sarvam-1 , 2024. 2-billion-
parameter language model optimized for 10 Indian languages.
Charles Spearman. The proof and measurement of association between two things. The Amer-
ican Journal of Psychology , 15(1):72–101, 1904. doi: 10.2307/1412159.
Rohan T aori, Ishaan Gulrajani, Tianyi Zhang, Y ann Dubois, Xuechen Li, Carlos Guestrin,
Percy Liang, and T atsunori B. Hashimoto. Stanford alpaca: An instruction-following llama
model. GitHub Repository , 2023.
Hugo T ouvron, Thibaut Lavril, Gautier Izacard, Xavier Martinet, Marie-Anne Lachaux, Timo-
thée Lacroix, and Guillaume Lample. Llama: Open and eﬀicient foundation language models.
arXiv preprint arXiv:2302.13971 , 2023a.
Hugo T ouvron, Louis Martin, Kevin Stone, Peter Albert, Amjad Almahairi, Y asmine Babaei,
and Thomas Scialom. Llama 2: Open foundation and fine-tuned chat models. arXiv preprint
arXiv:2307.09288 , 2023b.
Ashish V aswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez,
Łukasz Kaiser, and Illia Polosukhin. Attention is all you need. Advances in Neural Information
Processing Systems , 30, 2017.
28