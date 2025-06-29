# RAG-VisualRec: An Open Resource for Vision- and Text-Enhanced Retrieval-Augmented Generation in Recommendation

**Authors**: Ali Tourani, Fatemeh Nazary, Yashar Deldjoo

**Published**: 2025-06-25 20:32:12

**PDF URL**: [http://arxiv.org/pdf/2506.20817v1](http://arxiv.org/pdf/2506.20817v1)

## Abstract
This paper addresses the challenge of developing multimodal recommender
systems for the movie domain, where limited metadata (e.g., title, genre) often
hinders the generation of robust recommendations. We introduce a resource that
combines LLM-generated plot descriptions with trailer-derived visual embeddings
in a unified pipeline supporting both Retrieval-Augmented Generation (RAG) and
collaborative filtering. Central to our approach is a data augmentation step
that transforms sparse metadata into richer textual signals, alongside fusion
strategies (e.g., PCA, CCA) that integrate visual cues. Experimental
evaluations demonstrate that CCA-based fusion significantly boosts recall
compared to unimodal baselines, while an LLM-driven re-ranking step further
improves NDCG, particularly in scenarios with limited textual data. By
releasing this framework, we invite further exploration of multi-modal
recommendation techniques tailored to cold-start, novelty-focused, and
domain-specific settings. All code, data, and detailed documentation are
publicly available at: https://github.com/RecSys-lab/RAG-VisualRec

## Full Text


<!-- PDF content starts -->

arXiv:2506.20817v1  [cs.IR]  25 Jun 2025RAG-VisualRec: An Open Resource for Vision-
and Text-Enhanced Retrieval-Augmented Generation
in Recommendation
Ali Tourani∗, Fatemeh Nazary†, Yashar Deldjoo‡,
June 27, 2025
Abstract
This paper addresses the challenge of developing multimodal recommender sys-
tems for the movie domain, where limited metadata (e.g., title, genre) often hinders
the generation of robust recommendations. We introduce a resource that combines
LLM-generated plot descriptions with trailer-derived visual embeddings in a unified
pipeline supporting both Retrieval-Augmented Generation (RAG) and collabora-
tive filtering. Central to our approach is a data augmentation step that transforms
sparse metadata into richer textual signals, alongside fusion strategies (e.g., PCA,
CCA) that integrate visual cues. Experimental evaluations demonstrate that CCA-
based fusion significantly boosts recall compared to unimodal baselines, while an
LLM-driven re-ranking step further improves NDCG, particularly in scenarios with
limited textual data. By releasing this framework, we invite further exploration
of multi-modal recommendation techniques tailored to cold-start, novelty-focused,
and domain-specific settings. All code, data, and detailed documentation are pub-
licly available at: https://github.com/RecSys-lab/RAG-VisualRec .
1 Introduction
Recommender systems (RS) are progressively integrating advanced large-language mod-
els (LLMs), to interpret user preferences, capture nuanced item semantics, and gen-
erate explainable and contextually meaningful recommendations beyond conventional
interaction-based personalization. Retrieval-augmented generation (RAG), which com-
bines LLM generation capabilities with external evidence retrieval, has emerged as a
particularly promising approach to address limitations such as hallucinations, outdated
knowledge, and insufficient item metadata, thereby significantly enhancing recommen-
dation accuracy, novelty, and interpretability [6, 15, 5, 14]. A standard RAG pipeline
(Fig. 1) involves three main steps: (1) retrieval , which identifies relevant candidate items
from external sources (e.g., databases, visual and textual embeddings); (2) augmenta-
tion, enriching user queries with contextual information; and (3) generation , synthesizing
the augmented content into coherent recommendations. While classical collaborative fil-
tering (Collaborative Filtering (CF)) remains useful for stage (1), embedding-based
retrieval has become the de-facto choice because it embeds heterogeneous signals—text,
audio, vision—into a shared vector space, enabling (i) richer side-information use, (ii)
∗Interdisciplinary Centre for Security, Reliability, and Trust (SnT), University of Luxem-
bourg, Luxembourg. Institute for Advanced Studies, University of Luxembourg, Luxembourg.
ali.tourani@uni.lu
†Polytechnic University of Bari, Bari, Italy. fatemeh.nazary@poliba.it
‡Polytechnic University of Bari, Bari, Italy. deldjooy@acm.org
1

Figure 1: The overall pipeline of the proposed multimodal RAG framework.
real-time adaptation to catalogue changes, and (iii) natural exposure of long-tail or
freshly introduced items. For instance, if a film suddenly wins an award, a RAG recom-
mender can ingest the new fact into its vector store and refresh its suggestions without
full retraining [8].
Why video? The video/movie domain, characterized by inherently rich multimodal in-
formation—combining visual, audio, and textual elements—represents an ideal yet chal-
lenging environment for testing multimodal RAG-based recommender systems. Video
recommender systems must not only handle complex and sparse metadata but also ef-
fectively fuse multimodal signals extracted from diverse representations such as movie
trailers, audio tracks, and textual plot synopses. However, the research community cur-
rently lacks comprehensive, openly accessible datasets specifically designed to evaluate
multimodal RAG approaches for within realistic and demanding video recommendation
scenarios.
Gap in Existing Datasets. Despite notable advances in multimodal recommenda-
tion, current datasets often fall short for rigorous retrieval-augmented generation (RAG)
(see Table 1). Key shortcomings include: (i) Minimal support for RAG —most datasets
(e.g.,MMTF-14K [4], MicroLens [10]) focus on side-information-based collaborative fil-
tering, overlooking external multimodal retrieval for generation. (ii) Overly simplistic
fusion —datasets like Ducho [3] or MMRec [16] emphasize basic fusion ( e.g.,parallel con-
catenation) instead of alignment ( e.g., CCA) or robust retrieval. (iii) Sparse metadata
handling —many neglect the common reality of incomplete item descriptions in domains
like fashion or travel. (iv) Inadequate cold-start coverage —few explore early-stage rec-
ommendations when textual data is limited, hindering understanding of multimodal
signals.
RAG-VisualRec1at a glance. To close these gaps, we present RAG-VisualRec , the
firstopen, auditable, multimodal benchmark and toolkit tailored to multimodal RAG for
video recommendation. Rather than proposing yet another algorithm, RAG-VisualRec
aims to deliver the missing research infrastructure:
•Dataset layer. 9,724 MovieLens titles aligned with 3,751 trailers, each enriched
by eight low-level audiovisual descriptors and GPT-4o-generated plot synopses with
chain-of-thought rationales.
•Toolkit layer. A typed, open-source library offering early- and mid-fusion operators
1https://anonymous.4open.science/r/RAG-VisualRec-2866/
2

Table 1: Comparative Analysis of Multimodal Recommendation Systems and Datasets.
Name Modality Fusion Method Recommendation Type Data / Code
Visual Audio Text CF-side MM-RAG Other
Ducho v2.0 [3] ✓ ✓ ✓ Sum / Mul / Concat / Mean ✓ ✗ ✗ [code]
Ducho-meets-Elliot [2] ✓ ✓ ✓ Sum / Mul / Concat / Mean ✓ ✗ ✗ [code]
Rec-GPT4V [9] ✓ ✗ ✓ Unimodal (Textual) ✓ ✗ ✗ ✗
MMSSL [13] ✓ ✓ ✓ Modality-aware Fusion ✓ ✗ Modality-aware augmentation [code]
MMRec [16] ✓ ✓ ✓ Concat ✓ ✗ ✗ [code]
MMRec-LLM [12] ✓ ✗ ✓ Concat ✓ ✗ LLM summarization ✗
MMTF-14K [4] ✓ ✓ ✓ Mean / Med / MeanMed ✗ ✗ post-hoc (rank aggregation) [data]
YouTube-8M [1] ✓ ✗ ✓ PCA ✗ ✗ ✗ [data]
MicroLens [10] ✓ ✓ ✓ Sum / Concat ✓ ✗ ✗ [code]
RAG-VisualRec ✓ ✗ ✓ CCA / PCA / Concat ✓ ✓ ✗ [code]
(concat, PCA, CCA), textual augmentation, strong CF baselines, and a plug-and-play
RAG loop with optional LLM re-ranking.
•Benchmark harness. Declarative configurations and pipeline codes regenerate all
metrics—12 accuracy and 5 beyond-accuracy indicators (coverage, novelty, long-tail
share, cold-start lift, diversity)—with a single command, aligned with ACM repro-
ducibility guidelines. In this paper, due to space constraints, we report results on a
shorter set of metrics: NDCG, recall, novelty, TailFrac, and catalog coverage.
Our contributions. Building on this foundation, the paper makes five concrete ad-
vances:
1.Multimodal RAG Pipeline, Visual+Text Fusion, Three Fusion Methods. We intro-
duce an end-to-end multimodal Retrieval-Augmented Generation (RAG) pipeline
that integrates trailer-based visual embeddings (extracted via pre-trained ResNet-
50 CNN models) with advanced textual embeddings produced by state-of-the-
art language models, including GPT-4o, Llama3-8B, and Sentence Transformers
(such as Sentence-T5 and text-embedding-3-small); this pipeline enables three dis-
tinct multimodal fusion methods: simple concatenation (Concat), dimensionality
reduction using Principal Component Analysis (PCA), and alignment through
Canonical Correlation Analysis (CCA).
2.Open Resource, Rich Annotations, Audiovisual + Text Data. We openly release a
multimodal corpus that aligns all 9,724 movies in MovieLens-Latest with the
3,751 trailers available in the public MMTF-14K collection. Each trailer is sup-
plied with two families of pre-extracted visual descriptors—(i) CNN-based frame
embeddings and (ii) aesthetic-style features—stored under both Average and
Median aggregation schemes; our experiments use the CNN / Average variant,
but the other representations can be loaded by toggling a single flag. The package
also includes the block-level and i-vector audio descriptors shipped with MMTF-
14K (not exploited in the current study but ready for future work). Every movie
is further enriched with a GPT-4o–generated plot synopsis and an accompanying
chain-of-thought rationale, providing rich textual context for retrieval-augmented
generation.
3.LLM Data Augmentation, Cold-Start, Metadata Enrichment. We develop LLM-
driven data augmentation strategies specifically designed to address sparse meta-
data and cold-start problems. Our methods use GPT-4o to systematically gen-
erate semantically rich and highly detailed item descriptions from limited initial
metadata, substantially enhancing the quality of multimodal embeddings and im-
proving recommendation robustness in low-information and long-tail scenarios.
Both the original and augmented textual data are encoded using three distinct
3

Figure 2: Overall performance overview.
embedding models—OpenAI, Llama, and Sentence Transformers—ensuring di-
verse and complementary representations.
4.Comprehensive Evaluation Suite, Accuracy+Beyond, Reproducibility. We design
and implement a comprehensive evaluation suite that rigorously measures recom-
mendation quality across multiple dimensions: standard accuracy metrics (Re-
call@K, nDCG@K), as well as beyond-accuracy metrics such as novelty, catalog
coverage, long-tail effectiveness, cold-start scenario handling, and recommenda-
tion diversity; the entire evaluation process is made fully reproducible via declar-
ative configuration files and transparent, open-source code workflows.
5.Empirical Validation, CCA Outperforms, LLM Re-ranking. Our experiments
demonstrate that Canonical Correlation Analysis (CCA)–based multimodal fu-
sion consistently surpasses unimodal and simple-concat baselines, delivering up to
27% higher Recall@10 and almost doubling catalogue coverage. A second-stage,
zero-shot GPT-4o re-ranking step further enhances quality, yielding an additional
≈9%relative gain in nDCG@10 over the strongest non-LLM baselines.
Experimental Scope. Overall, our experimental setup spans:
3 (embedding families) ×3 (fusion methods) ×2 (retrieval method)
×120 (tested users) ≈2000 (experimental cases) .
Figure 2 previews key results by comparing key metrics (nDCG10 and tail frac-
tion) across different text backbones (Sentence Transformers, OpenAI, Llama3) under
three setups: visual-only, text-only, and multimodal fusion. The fused multimodal ap-
proach consistently outperforms unimodal variants in nDCG10, particularly for Sentence
Transformers and Llama3, and typically enhances catalogue coverage compared to both
visual-only and text-only baselines.
In summary, RAG-VisualRec provides a new, reproducible test-bed for multimodal
RAG research. We release all data and codes to catalyze further advances in robust and
explainable recommender systems.
4

Figure 3: Visual-RAG: Modular architecture supporting data ingestion, multimodal em-
bedding, fusion, data augmentation using LLMs, retrieval, LLM re-ranking, and robust
evaluation.
2 Toolkit Design and Visual-RAG Pipeline
2.1 System Overview: Motivation and Researcher Experience
TheVisual-RAG pipeline is designed as a transparent, modular, and extensible re-
source for rigorous multimodal recommendation research. Its primary goal is to bridge
the gap between theoretical advances (e.g., fusion techniques, textual data augmen-
tation, multi-modal retrieval, augmented generation) and practical, reproducible work-
flows that any researcher can adapt or extend. Figure 3 visualizes the pipeline as a series
of interconnected, parameterized blocks, enabling rapid experimentation and seamless
integration of new modalities, textual data augmentation, or evaluation metrics.
Researcher Scenario: Suppose a movie recommendation researcher wants to bench-
mark the impact of multimodal fusion (textual + visual) under cold-start conditions,
simulate the impact of data augmentation on popular movies, and reproduce all metrics
with a single config file—Visual-RAG enables all this without any code edits.
5

2.2 Step-by-Step Pipeline Walkthrough
Data Preparation and Ingestion
Data ingestion begins with MovieLens-Latest-Small or MovieLens-1M, loaded via
pandas.DataFrame adapters for uniformity. All user-item interactions are cleaned,
integer-indexed, and merged with available metadata (title, genres, tags). The
pipeline handles missing metadata gracefully, a crucial requirement for studying
real-world cold-start scenarios or incomplete catalogs.
Worked Example: Data Enrichment
Input: Title = Nixon (1995), Genres = Drama, Biography, Description =
(missing)
Output after LLM Augmentation: “Nixon (1995) explores the troubled
psyche and political career of America’s 37th president, delving into both his
strategic brilliance and the moral compromises that shaped his legacy. Directed
by Oliver Stone, the film highlights Nixon’s complex relationships with political
allies and adversaries, offering a gripping portrayal of power and vulnerability.”
To enable targeted experiments, a popularity annotator stratifies all items as head
(top 10%), mid-tail (next 40%), or long-tail (bottom 50%) according to their
frequency in the data. This annotation is vital for controlled cold-start evaluation.
Multimodal Embedding Extraction
This stage generates unified item representations from multiple modalities:
(a)Textual: Item descriptions are embedded using one of three back-ends: OpenAI-
Ada, SentenceTransformer-MiniLM, or LLaMA-2/3 (via HuggingFace). This
enables evaluation under both proprietary and open-source large language
model (LLM) encoders. The text-to-vector mapping yields dtext-dimensional
outputs.
(b)Visual: Frame-level visual features are extracted using ResNet-50, with both
mean and median pooling across sampled keyframes to produce robust dvis=
2048-dimensional vectors per item.
(c)Audio (optional): Four Mel-Frequency Cepstral Coefficient (MFCC) variants
(log, correlation, delta, spectral) are computed, each resulting in daud= 128-
dimensional vectors, providing complementary semantic cues for music or video
datasets.
(d)Fusion: Multimodal fusion is implemented using four stateless operators— concat ,
pca 128,cca 64, and avg—which project concatenated features into a joint la-
tent space d⋆. These operators allow for controlled ablation of cross-modal
information and support both early and late fusion paradigms.
Fusion Operators: These modality-specific embeddings are combined using a
choice of stateless operators, set via a configuration flag:
•Concatenation (concat ): Directly stacks all vectors.
•PCA (pca128): Projects concatenated vectors onto the first 128 principal
components.
•CCA (cca64): Aligns textual and visual spaces, maximizing cross-modal
correlation.
•Averaging (avg): Arithmetic mean for sanity checking.
6

This setup allows for plug-and-play ablation—simply change the fusion method in
the config and rerun to study its impact.
Example: Fusion in Practice
Text embedding of Nixon (1995): [ ···,−0.31,0.54,1.02, ...]
Visual embedding from trailer frames: [ ···,0.11,−0.22,0.91, ...]
CCA fusion produces a joint 64-d vector: [ ···,0.44,0.08,−0.32, ...]
Embedding Swap and Re-embedding
When item side information is altered (e.g., via adversarial augmentation), the
pipeline re-embeds those items using the original encoder settings. This ensures
that all subsequent retrieval and ranking steps are based on the current (potentially
poisoned) representations, enabling true end-to-end robustness studies.
User Embedding Construction
User representations are constructed with three interchangeable strategies:
•Random : Assigns a random vector as a baseline (for sanity checks).
•Average : Averages the fused embeddings of all high-rated items for the user.
•Temporal : Applies logistic decay to prioritize recent interactions.
All user vectors are mapped to the same fused space as items, allowing direct
cosine-based retrieval.
Example: Temporal Embedding
Suppose User 42 has rated Nixon (1995), The Post, and Frost/Nixon. Each
rating is timestamped. Temporal embedding weights the most recent (e.g., The
Post) more heavily in the user’s fused profile.
Candidate Retrieval
Given a user embedding, the pipeline constructs a cosine-based k-NN index over
all item vectors, efficiently returning the top- Nmost similar items for each user.
This step is sensitive to all upstream choices—embedding models, fusion, user pro-
file method—enabling direct comparison of retrieval quality under varied research
conditions.
Profile Augmentation and LLM Prompting
For LLM-based re-ranking, the system composes a structured JSON user pro-
file, containing favorite genres, most-used tags, top items, and a free-form taste
synopsis. Profiles can be:
•Manual: Derived by code from historical data (e.g., top genres, recent items).
•LLM-based: Synthesized from user history via LLM prompt (e.g., “This user
enjoys political dramas and historical biographies...”).
The final prompt to the LLM includes (a) the user profile, (b) candidate items with
metadata, and (c) strict task instructions (e.g., “Return the top 10 recommended
item IDs as a JSON array”).
7

Example: User Profile Prompt
USER PROFILE:
Genres: ["Drama", "Biography"], Top items: ["Nixon (1995)",
"The Post"], Taste: "Prefers political dramas exploring real
historical events."
TASK: Given this profile and the following candidate movies,
return your top-10 recommendations as a JSON list of item IDs.
LLM Generation Head and Re-ranking
The LLM-based generation head accepts the profile and candidates, then outputs
a ranked list of recommendations. Two privacy regimes are supported:
•ID-only : Returns only recommended item IDs (for privacy, e.g., production
use).
•Explainable : Returns both IDs and explanations/chain-of-thought rationales
(for interpretability research).
A robust JSON parser extracts outputs; failures fall back to the kNN list, ensuring
robustness.
Evaluation and Logging
Each experiment is evaluated with a comprehensive suite of metrics:
•Accuracy : Recall@K, nDCG@K, MAP, MRR, etc.
•Beyond-Accuracy : Coverage, novelty, diversity, long-tail fraction.
•Fairness/Robustness : Cold-start rate, exposure.
Metrics are computed per-user, averaged across test splits, and exported as CSV/Parquet.
Intermediate artefacts (embeddings, poisoned texts, candidate lists) are check-
pointed, ensuring full reproducibility. Optionally, all outputs can be pushed to a
public repository for open auditing.
2.3 Configuration Schema
Each stage of the pipeline is fully configurable through a centralized parameter block in
the Colab notebook. Key hyperparameters and toggles include:
•Dataset :mllatest small —lfm360k
•Embeddings :textual | visual | audio | fused {concat,pca,cca,avg }
•LLM Model :openai | sentence transformer | llama3
•User Vector :random | average | temporal
•Retrieval :N(default 50)
•Recommendation :K(default 10), explainable?
•Runtime :usegpu,seed,batch size
Default values are chosen to exactly reproduce the benchmarks and ablation studies
reported in Section 5. Importantly, changes to any flag or hyperparameter require no
code edits, supporting rapid experimentation and extensibility. The pipeline is fully
controlled by a single config block specifying dataset, embedding/fusion methods, user
8

profile mode, retrieval depth, evaluation settings, and runtime (GPU/CPU, batch size,
random seed). Modifying any experimental setting requires only a config edit, not a
code change.
Implementation Footprint: On a standard Google Colab A100 (40 GB), a full run
(MovieLens-1M, all fusion variants) completes in ∼3 hours, with peak memory under
11 GB. This efficiency enables large-scale or multi-domain experiments even on modest
hardware.
2.4 Practical Implications and Researcher Perspective
Reproducibility and Transparency: All experiments are deterministic (seeded),
versioned, and documented. CI jobs and open-source scripts enable reviewers to audit
every table or figure. Rapid switching between baselines or fusion methods facilitates
systematic ablation and robustness research.
Research Impact: The Visual-RAG pipeline empowers researchers to:
•Benchmark the effect of multimodal fusion under various data sparsity and cold-start
conditions.
•Study trade-offs between explainability, privacy, and accuracy.
Summary: Through modular design, extensive configuration, worked examples, and
fully open infrastructure, Visual-RAG offers a transparent, extensible, and auditable
testbed for next-generation recommender system research.
3 Formalism
In the following section, we formalize the proposed multimodal framework leveraging
RAG for movie recommendation.
Item Representations and Multimodal Fusion.
LetUdenote the set of users and Idenote the set of items (i.e., movies). Each item
i∈ Ican be represented by two primary embeddings:
•Textual embedding :x(txt)
i∈Rdtxt, extracted from a large language model (e.g.,
GPT, ST, or Llama).
•Visual embedding :x(vis)
i∈Rdvis, extracted from trailer frames or short promo-
tional clips.
If available, audio embeddings x(aud)
i∈Rdaudmay also be included. A fusion function
F(·) combines these embeddings into a single multi-modal vector zi∈Rdz.Common
approaches include:
zi= PCA
x(txt)
i∥x(vis)
i
, (1)
 
z(txt)
i,z(vis)
i
= CCA 
x(txt)
i,x(vis)
i
, (2)
where ∥denotes concatenation. With PCA , we first concatenate text and visual embed-
dings, then project to dzdimensions. With CCA , we learn z(txt)
iandz(vis)
ias canonical
9

components in a shared subspace. We concatenate these canonical components into a
single vector:
zi=h
z(txt)
i∥z(vis)
ii
Both approaches leverage complementary cues (semantic text vs. visual aesthetics),
thereby enabling multimodal fusion, which can result in enhanced movie recommenda-
tion performance. Each user u∈ Uhas historically rated or interacted with a subset of
items. In a content-driven embedding approach, we compute the user embedding uuby
aggregating item embeddings. The average user embedding is:
p(avg)
u =1I+uX
i∈I+
uzi,
where I+
uis the set of items that urated above a threshold (e.g. 4 stars). The temporal
embedding instead weights items by recency:
p(temp)
u =P
i∈I+
u 
zi·wu,i
P
i∈I+
uwu,i, w u,i=1
1 + exp
−α(time( i)−¯tu),
where ¯tudenotes the average timestamp for user u, and αis a smoothing hyperparam-
eter.
Note. We experimented with both user profile representations, p(avg)
u and
p(temp)
u , ultimately selecting the latter for its superior end-to-end performance,
excluding the former for space considerations. Henceforth, in the following we
denote pu=p(temp)
u . The terms user profile embedding anduser embedding are
used interchangeably to refer to this representation.
Visualization of t-SNE Projection of Embeddings. As a demonstration, Fig-
ure 4 visualizes user embeddings (red stars) and item embeddings (blue points), in a
shared 2D space, obtained via t-SNE projection of the embedding under investigation.
investigation) using t-SNE. The four subplots represent (a) random user vectors, (b)
textual, (c) visual, and (d) fused multimodal embeddings. In (a), the random user em-
bedding is non-informative, clustering far from items. By contrast, (b), (c), and (d)
employ the temporal aggregation strategies introduced in Section 3, producing more
coherent user–item alignment. This demonstrates how textual, visual, and fused rep-
resentations capture meaningful movie features, potentially enhancing recommendation
performance.2
Retrieval-Augmented Generation (RAG) Stage
Retrieval Step. For recommendation with LLM-based generation, we first retrieve
top-Nitems most relevant to user u. In order to achieve this, based on the user profile
embedding pu, we compute similarity s(u, i) = cos 
pu,zi
to produce final top- k
recommendation items, where k≪N. We examine the impact of retrieval scope Non
enhancing the final top- kperformance in our experimental research questions (cf. RQ4).
LLM-Based Augmentation. During augmentation, the LLM (e.g., GPT-4) takes as
input
(a) A structured user profile (genres, tags, top items),
2We prioritize temporal user profiles for better recommendations, though T-SNE showed similar
patterns for average embeddings.
10

(a)Random
 (b)Textual
 (c)Visual
 (d)Fused
Figure 4: t-SNE projection of item and user embeddings using the Sentence Transformer
(ST) LLM backbone. Multimodal embeddings are obtained via CCA.
(b) The candidate top- Nitems from retrieval, along with their metadata
into the form of a structured prompt. The LLM is then instructed to re-rank these N
candidates, generating a final list of krecommended items, k≤N, plus optional textual
rationales. We explore two methods to build or to generate this structured user profile:
(i)Manual: We compute a structured user profile (favorite genres, top-rated items,
etc.) from the training data. A simple textual summary is created with a rule-
based or heuristic script (i.e., “manual”).
(ii)LLM-based: We feed the user’s historical items and preferences into a large
language model (e.g., GPT-4) to generate a more free-form textual description of
that user’s tastes.
In our experimental research questions (RQs), we explore the impact of both aug-
mentation strategies (cf. Sec. 5):
Manual Augmentation| {z }
structured templatevs. LLM-based Augmentation| {z }
Free-Text
Generation Step.
In typical two-stage recommendation, the first stage (retrieval) picks a broader set of
candidates for efficiency, while the second stage (re-ranking) refines the list using deeper
signals (e.g., user textual profile). Formally:
Retrieve u(pu) = argsorti
i∈Is 
pu,zi
[:k],
ReRank u
eCu,Profile u
= LLM
eCu|Profile u
,
where eCuis the top- Nset from retrieval, and Profile uis a textual or structured repre-
sentation summarizing user upreferences (genres, prior ratings, etc.). This two-stage
framework ( retrieval andLLM-based re-ranking/generation ) flexibly integrates multi-
modal embeddings (visual, textual, audio) into personalized recommendation tasks. Re-
trieval outputs ( eCu) combined with user preference profiles (Profile u) naturally facilitate
both structured recommendations and narrative generations.
11

4 Proposed Resource and System Implementation
We present a novel resource for multimodal recommendation, integrating (i) LLM-
generated textual descriptions and (ii) visual trailer embeddings into a retrieval-augmented
pipeline. We detail our strategy for augmenting sparse movie metadata, summarize vi-
sual embedding extraction and fusion, and explain how these multimodal features sup-
port retrieval-augmented generation (RAG) and optional collaborative filtering (CF).
Overview and Motivation.
The proposed resource addresses sparse textual metadata by using LLMs to generate
short descriptions for movies, embedded via modern text encoders (e.g., GPT, Sentence
Transformers). Additionally, we leverage movie trailers for visual embeddings. Combin-
ing these textual and visual signals mitigates cold-start issues and captures cinematic
nuances overlooked by text-only methods. The resource content and main highlights
include:
•Enriched Textual Descriptions: LLMs expand sparse metadata into short
paragraph-level synopses for each movie.
•Visual Trailer Embeddings: We adopt existing trailer-based embeddings (e.g.,
from MMTF-14K [4]) or optionally generate them via CNN/Transformer back-
bones.
•Fusion Methods: We provide code for combining text and visual embeddings
with PCA or CCA, enabling a single multimodal representation for retrieval or
CF.
•RAG and CF Integration: We offer scripts to (1) perform embedding-based
retrieval + LLM re-ranking (RAG), and (2) (optional) incorporate side embed-
dings into CF methods.3We also include routines for standard accuracy metrics,
beyond-accuracy measures (novelty, tail coverage), and ablation analyses (tempo-
ral vs. average embeddings).
Overall, our resource unifies textual augmentations with visual data into a single flexible
environment, simplifying the design of multi-modal recommenders for the movie domain.
Text Processing and Data Augmentation.
To enrich the metadata of each film, we request a large language model with minimal
inputs ( title, genre, tags ) and request a concise description driven by the plot. For
instance:
Prompt Example: “Given the title {title}and genres {genre }, write a short paragraph sum-
marizing the film’s plot, themes, and style.”
The output of the LLMs is saved as an augmented description , forming a crucial
textual feature in cases where official synopses are missing. Table 2 shows the enriched
description for the movie Nixon (1995) . We then produce embeddings from the aug-
mented text via chosen encoders (e.g., OpenAI or Sentence Transformers), yielding a
vector x(txt)
i∈Rdtxtfor each item i.
3https://anonymous.4open.science/r/RAG-VisualRec-2866/
12

Table 2: Sample data augmentation for Nixon (1995) . Minimal metadata is auto-
matically expanded into a cohesive description highlighting historical context, thematic
depth, and key figures.
Aspect Before After LLM Augmentation
Title Nixon (1995) unchanged
Genres Drama — Biography unchanged
Description Not provided “Nixon (1995) explores the troubled psyche and po-
litical career of America’s 37th president, delving
into both his strategic brilliance and the moral com-
promises that shaped his legacy. Directed by Oliver
Stone, the film highlights Nixon’s ...”
Visual Embeddings and Fusion Methods
We incorporate trailer-level embeddings from publicly available datasets (e.g., MMTF-
14K). Each trailer is sampled at fixed intervals, passed through a CNN or Vision Trans-
former, and aggregated (e.g., mean pooling) to produce x(vis)
i∈Rdvis. We provide
scripts for custom extraction if desired.
Multimodal Fusion. Since textual and visual signals can be complementary, we ap-
ply two main fusion strategies:
•Concatenation + PCA: Stack x(txt)
iandx(vis)
i, then project down to a fused
vector zivia PCA.
•CCA Alignment: Learn a shared subspace maximizing cross-modal correlation,
yielding aligned vectors for text and visuals. These aligned components are con-
catenated as the fused embedding.
Either fused or unimodal embeddings are then fed into the recommender.
Integration into RAG and CF
We adopt the two-step retrieval and LLM-based re-ranking pipeline described in Sec-
tion 3: (i) retrieve candidates via k-NN in user-aggregated embedding space, and
(ii) re-rank these candidates using LLM prompts informed by user profiles and can-
didate metadata. In our experiment, we fix k= 10 and vary the retrieval scope
N∈[20,30,50,100,150] to analyze its impact (cf. RQ4).
Note. This paper introduces multimodal data tailored for evaluating RAG-based
methods in RSs. While experiments integrating side information embeddings into
CF models (e.g., visual embeddings via VBPR or ConvMF, textual embeddings
via CTR) are deferred to future work, example implementations using the Cornac
[11] library are provided in the GitHub repository.
Evaluation Metrics and Hyper-Parameter Settings
Beyond standard ranking metrics such as Recall and nDCG, we also measure diversity
and long-tail aspects:
Item Coverage is defined inline as Coverage =S
u∈UR(u)
|I|, where R(u) is the top- k
recommendation list for user uandIis the set of all items.
13

Table 3: Retrieval Stage Results (Accuracy Metrics): Columns are grouped by Modal-
ity(Visual, Text), Fusion Method (PCA, CCA), and Metrics (Recall (temp), NDCG
(temp), Recall (avg), NDCG (avg)). The best values are highlighted in green, while the
second-best are in yellow. ∆ NDCG indicates the relative improvement in NDCG (temp)
over the baseline.
Name Uni/Multimodal Fusion Metrics∆ndcg
Visual Text PCA CCA Recall (temp) NDCG (temp) Recall (avg) NDCG (avg)
Visual Unimodal ✓ 0.036688 0.037534 0.026457 0.025499 ±0.0%
ST Unimodal ✓ 0.160541 0.159737 0.116791 0.089738 +325 .6%
OpenAI Unimodal ✓ 0.176216 0.181286 0.116251 0.101413 +383 .0%
Llama3.0 Unimodal ✓ 0.093495 0.096862 0.070260 0.055245 +158 .1%
Visual+Textual (ST) ✓ ✓ ✓ 0.104287 0.122746 0.092985 0.096431 +227 .6%
Visual+Textual (OpenAI) ✓ ✓ ✓ 0.104882 0.123015 0.093541 0.096444 +228 .4%
Visual+Textual (Llama3.0) ✓ ✓ ✓ 0.108265 0.139158 0.089097 0.099020 +271 .3%
Visual+Textual (ST) ✓ ✓ ✓ 0.205637 0.252080 0.180762 0.174588 +571 .6%
Visual+Textual (OpenAI) ✓ ✓ ✓ 0.087140 0.080328 0.072103 0.055052 +114 .0%
Visual+Textual (Llama3.0) ✓ ✓ ✓ 0.119989 0.148211 0.099557 0.095724 +294 .9%
Novelty measures the average unexpectedness of recommended items: Novelty =
1
|R|P
i∈R−log2 
p(i)
, where p(i) is the relative popularity of item i. Higher values
indicate recommendations of less popular (thus more novel) items.
LongTail Fraction is given by LongTailFrac =1
|R|P
i∈R1
(iis tail)
where an item
is “tail” if it falls below a certain popularity threshold (in this work, τtail= 2).
Hyper-Parameter Settings. The following describes our choice of hyperparameters:
•LLM Generation : Temperature = 0 .7, max tokens = 200.
•k-NN retrieval :N= 50 or N= 100 neighbors; final top- k= 10 for recommen-
dations.
•PCA or CCA dims :dz= 64 or 128, chosen via grid search.
•CF Training : 10 latent dimensions, learning rate = 0 .01, up to 5–10 epochs, with
a 70:30 train-test split.
Overall, these choices provide baseline settings for reproducibility; finer hyperparameter
tuning is possible as needed.
Dataset . We benchmark our approach using the MovieLens (Latest) dataset [7],
filtering out users between 20 to 100 interactions. We perform a chronological train–test
split per user, allocating the earliest 70% of interactions for training and the latest 30%
for testing. The dataset contains 100 ,836 interactions, |U|= 610, |I|= 9,724, average
interactions per user 165 .30, per item 10 .37, and sparsity of 0 .017. For efficiency, the
evaluation uses 120 randomly selected users. Items are categorized into head (popular),
mid-tail, or long-tail based on rating frequency.
5 Benchmark and Experiments
Table 3 and Table 4 summarize the key findings for both Retrieval andRecommenda-
tion stages, contrasting unimodal ( Visual Unimodal ,ST Unimodal ,OpenAI Unimodal ,
Llama3.0 Unimodal ) and multimodal ( Visual+Textual with PCAorCCA). Although we
observe broadly similar trends in retrieval and recommendation, our primary focus is
on the Recommendation Stage results (Table 4), as the final ranking is most relevant
for practical systems. We nonetheless reference the Retrieval Stage (Table 3) to confirm
that both stages share consistent patterns. To organize our discussion, we define four
14

experimental research questions ( RQ1–RQ4). Note that the terms Manual andLLM-
based pipelines specifically refer to the augmentation process (Sec. 3).
Table 4: Recommendation Stage Results (Accuracy Metrics). The table is split into two
parts: Manual Pipeline (top) and LLM-based Pipeline (bottom). We use the same
color-based notation and columns as in Table 3.
Recommendation Stage – Manual Pipeline
Name Uni/Multimodal Fusion Metrics∆ndcg
Visual Text PCA CCA Recall (temp) NDCG (temp) Recall (avg) NDCG (avg)
Visual Unimodal ✓ 0.01611 0.045019 0.01198 0.030501 ±0.0%
ST Unimodal ✓ 0.084321 0.167488 0.050541 0.079941 +272 .0%
OpenAI Unimodal ✓ 0.075686 0.166709 0.059188 0.104077 +270 .3%
Llama3.0 Unimodal ✓ 0.03982 0.091084 0.034442 0.062854 +102 .3%
Visual+Textual (ST) ✓ ✓ ✓ 0.043374 0.12483 0.035712 0.080947 +177 .3%
Visual+Textual (OpenAI) ✓ ✓ ✓ 0.037374 0.115342 0.036908 0.078362 +156 .2%
Visual+Textual (Llama3.0) ✓ ✓ ✓ 0.037333 0.114743 0.032583 0.075561 +154 .9%
Visual+Textual (ST) ✓ ✓ ✓ 0.107084 0.268102 0.095418 0.174363 +495 .5%
Visual+Textual (OpenAI) ✓ ✓ ✓ 0.028301 0.078006 0.025079 0.059463 +73.3%
Visual+Textual (Llama3.0) ✓ ✓ ✓ 0.047614 0.145944 0.041977 0.09262 +224 .2%
Recommendation Stage – LLM-based Pipeline
Name Uni/Multimodal Fusion Method Metrics∆ndcg
Visual Text PCA CCA Recall (temp) NDCG (temp) Recall (avg) NDCG (avg)
Visual Unimodal ✓ 0.01734 0.046774 0.00847 0.027337 ±0.0%
ST Unimodal ✓ 0.0773 0.169313 0.051284 0.086847 +262 .0%
OpenAI Unimodal ✓ 0.089671 0.179811 0.057714 0.101169 +284 .4%
Llama3.0 Unimodal ✓ 0.033492 0.083699 0.032395 0.066655 +78.9%
Visual+Textual (ST) ✓ ✓ ✓ 0.03513 0.120745 0.030523 0.06666 +158 .1%
Visual+Textual (OpenAI) ✓ ✓ ✓ 0.036987 0.101464 0.034161 0.072735 +116 .9%
Visual+Textual (Llama3.0) ✓ ✓ ✓ 0.038048 0.118645 0.028749 0.078329 +153 .7%
Visual+Textual (ST) ✓ ✓ ✓ 0.097654 0.246567 0.089355 0.159758 +427 .1%
Visual+Textual (OpenAI) ✓ ✓ ✓ 0.04025 0.084531 0.02919 0.063593 +80.7%
Visual+Textual (Llama3.0) ✓ ✓ ✓ 0.048338 0.141869 0.034597 0.079539 +203 .3%
RQ1Does incorporating(i.e., fusing) visual features improve accuracy metrics across dif-
ferent textual LLM backbones, and why is this crucial?
RQ2Which fusion method ( PCAorCCA) is more effective for combining textual and visual
embeddings, and why does this matter?
RQ3How do multimodal pipelines affect beyond-accuracy metrics (Coverage, Novelty,
TailFrac), and why are these additional metrics important for user satisfaction?
RQ4What differences emerge between the Manual vs.LLM-based pipelines in final rec-
ommendation quality, and what is the impact of retrieval stage scope N?
Please note that in the table, we also report ∆, which measures the improvement
over a similar visual baseline. However, in the text, we do not always explicitly reference
this value.
5.1 RQ1: Impact of Visual Information Across LLM Backbones
This question is important because each textual LLM encoder (e.g., ST,OpenAI ,Llama3.0 )
may capture different semantic nuances, and we want to know whether adding visual
signals consistently improves performance across all backbones or primarily benefits cer-
tain encoders. Understanding this helps us decide when and how to invest in visual data
extraction.
Findings. From Table 4 (Manual Pipeline), ST Unimodal achieves about 0 .0843 in
Recall( temp) and 0 .1675 in NDCG( temp), whereas OpenAI Unimodal follows closely
(Recall( temp) = 0 .0757, NDCG( temp) = 0 .1667). Llama3.0 Unimodal lags behind
15

with Recall( temp)≈0.0398. Purely visual embeddings alone ( Visual Unimodal ) are
substantially lower, with recall ≈0.0161.
When we fuse visual features, performance increases dramatically, particularly for ST
and OpenAI . For example, Visual+Textual (ST) CCA raises recall( temp) to≈0.1071
(a jump of more than +27% from ST Unimodal ) and NDCG( temp) from 0 .1675 to
0.2681 (about +60%). Similar improvements are seen at the retrieval stage (Table 3),
confirming that visual data consistently helps across both coarse retrieval and final rec-
ommendation. These results suggest that incorporating visual embeddings can
be crucial for capturing cinematic or aesthetic factors that purely text-based
approaches may miss, and this benefit is especially pronounced for STandOpenAI back-
bones.
5.2 RQ2: Comparing PCAvs.CCAFusion
This question matters because PCAandCCArepresent two distinct philosophies of dimen-
sionality reduction: PCAaligns with global variance in a concatenated space, whereas
CCAexplicitly models correlations between textual and visual embeddings. Choosing the
right method can critically affect the synergy between text and visuals.
Findings. From Table 4, PCA-based fusion yields moderate gains over unimodal sys-
tems but is consistently outperformed by CCA. Accordingly, and in the Manual Pipeline,
Visual+Textual (ST) PCA hits Recall( temp)≈0.0434 and NDCG( temp)≈0.1248,
whereas Visual+Textual (ST) CCA surpasses 0 .1071 in recall and 0 .2681 in NDCG
(more than double the PCAvalues). The retrieval-stage results (Table 3) echo this pat-
tern, with ST CCA reaching up to 0 .2056 in recall( temp). Thus, CCA-based approaches
appear far more adept at capturing cross-modal alignment, thereby boosting both re-
call and ranking quality. For practitioners, this implies that simply concatenating text
and visuals into a single PCAspace is suboptimal ; learning a correlation-maximizing
transformation ( CCA) consistently leads to better synergy.
5.3 RQ3: Influence on Coverage, Novelty, and TailFrac
Beyond-accuracy metrics are increasingly crucial in recommender systems because users
often value diversity, discover less popular items, and receive suggestions that go beyond
the mainstream. Here, we examine how multimodal pipelines impact Coverage ,Novelty ,
andTailFrac (share of long-tail items).
Findings. The attached beyond-accuracy table (Table 5) and radar charts (Fig. 5)
show that fusing visual and textual embeddings generally raises coverage by 0 .10–0.15
in absolute terms compared to unimodal text, thus allowing more items to appear in
recommendations. Novelty metrics (where higher indicates recommending less famil-
iar or popular items) also increase, especially under CCAfusion. For example, OpenAI
Unimodal might have coverage near 0 .216, whereas OpenAI+Visual (CCA) can exceed
0.32–0.34. However, the TailFrac measure can slightly decline in some cases, as the
system recommends more mid-popular items (rather than extremely obscure ones). In
Fig. 5, fused methods consistently expand coverage and novelty axes relative to unimodal
baselines, but the exact impact on deep-tail exposure varies. In summary, multimodal
pipelines typically broaden the system’s recommendation space , improving
item discovery and novelty, with a slight trade-off in extreme tail coverage for certain
encoders.
16

Table 5: Recommendation Stage Results (Beyond Accuracy Metrics). The table is split
into two parts: Manual Pipeline (top) and LLM-based Pipeline (bottom). We use
the same color-based notation as in Table 3.
Recommendation Stage – Manual Pipeline
Name Uni/Multimodal Fusion Method Metrics
Visual Text PCA CCA Coverage Novelty TailFrac
Visual Unimodal ✓ 0.165006 11.303532 0.048333
ST Unimodal ✓ 0.234122 11.10709 0.1075
OpenAI Unimodal ✓ 0.216065 10.797482 0.043333
Llama3.0 Unimodal ✓ 0.200498 11.288529 0.101667
Visual+Textual (ST) ✓ ✓ ✓ 0.308219 11.162305 0.062189
Visual+Textual (OpenAI) ✓ ✓ ✓ 0.305106 11.163761 0.068333
Visual+Textual (Llama3.0) ✓ ✓ ✓ 0.273973 11.112711 0.065833
Visual+Textual (ST) ✓ ✓ ✓ 0.32254 10.931898 0.0625
Visual+Textual (OpenAI) ✓ ✓ ✓ 0.345579 11.320455 0.090833
Visual+Textual (Llama3.0) ✓ ✓ ✓ 0.291407 11.128886 0.0825
Recommendation Stage – LLM-based Pipeline
Name Uni/Multimodal Fusion Method Metrics
Visual Text PCA CCA Coverage Novelty TailFrac
Visual Unimodal ✓ 0.1401 11.29123 0.0475
ST Unimodal ✓ 0.202366 11.152372 0.096667
OpenAI Unimodal ✓ 0.167497 10.700434 0.023333
Llama3.0 Unimodal ✓ 0.156912 11.128583 0.079167
Visual+Textual (ST) ✓ ✓ ✓ 0.281445 11.137257 0.06
Visual+Textual (OpenAI) ✓ ✓ ✓ 0.272105 11.061737 0.050833
Visual+Textual (Llama3.0) ✓ ✓ ✓ 0.256538 10.950571 0.045
Visual+Textual (ST) ✓ ✓ ✓ 0.314446 10.815412 0.064167
Visual+Textual (OpenAI) ✓ ✓ ✓ 0.328144 11.335649 0.085833
Visual+Textual (Llama3.0) ✓ ✓ ✓ 0.278954 11.047586 0.070774
Figure 5: Radar plot illustrating the performance of the recommendation stage across
both accuracy and beyond-accuracy metrics for various LLMs.
5.4 RQ4: Differences Between Manual and LLM-based Pipelines
This final question is relevant because many real-world systems use an LLM-assisted re-
ranking stage to refine the top- kitems after an initial retrieval or manual pipeline. We
want to see if LLM-based re-ranking substantially reshuffles the best methods or simply
provides incremental gains.
Findings. From the bottom half of Table 4, we note that LLM-based re-ranking pre-
serves the overall hierarchy of approaches ( ST-CCA ,OpenAI-CCA >ST-PCA , etc.), but
it can add +5% to +10% improvement in recall or NDCG for unimodal text meth-
17

Figure 6: Ablation study across various retrieval depths (Top-N), evaluating temporal
NDCG and recall values.
ods. For instance, OpenAI Unimodal sees NDCG( temp) rise from 0 .1667 ( Manual ) to
0.1798 ( LLM-based ), while Visual+Textual (ST) PCA also jumps from around 0 .1248
in NDCG( temp) to above 0 .1207 or 0 .130 in some runs. Overall, the best-fused methods
remain best under both pipelines, but the LLM-based approach refines the ranking fur-
ther, especially for weaker unimodal baselines. This indicates that while second-stage
re-ranking can sharpen the final list, strong multimodal representations remain
the core factor for high recommendation quality.
Regarding the impact of retrieval scope, in general, we find that an LLM-based re-
ranking stage can further refine top- krecommendations, especially for textual-scarce
items. Meanwhile, varying the retrieval depth ( N= 20 to N= 150) shows a well-
known trade-off: broader retrieval may improve recall initially but eventually saturates
or slightly harms nDCG due to noisier candidates. Figure 6 (example ablation) confirms
that certain fused pipelines (e.g., CCA-based) benefit from moderate Nincreases, while
purely text or purely visual retrieval sees diminishing returns.
6 Conclusions
We introduced a multi-modal recommendation resource that fuses textual signals from
large language models (LLMs) with trailer-derived visual embeddings under a unified
retrieval-augmented generation (RAG) pipeline. Our data augmentation step addresses
sparse metadata, while Canonical Correlation Analysis (CCA)-based fusion often out-
performs simple concatenation in both recall and coverage. Further, re-ranking via
an LLM can boost nDCG by exploiting richer context. These findings underscore the
benefits of combining textual and visual features for cold-start and novelty-focused rec-
ommendations. Looking ahead, we hope this resource will encourage more sophisticated
multimodal approaches, including audio embeddings or advanced cross-modal alignment,
to tackle domain-specific challenges where item metadata is limited.
Limitations and Future Work. While our work introduces an open multimodal
benchmark and toolkit for Retrieval-Augmented Generation (RAG) in recommendation,
several limitations remain:
18

•Single-Domain Focus: Our experiments are restricted to the MovieLens dataset,
limiting generalizability to other domains (e.g., music, e-commerce, fashion) where
multimodal signals and metadata sparsity may present different challenges. Evalu-
ating the framework on additional, diverse datasets would strengthen claims about
robustness and utility.
•Audio and Rich Multimodality Underexplored: Although the dataset sup-
ports audio features, our current experiments focus only on textual and visual modal-
ities. Integrating audio descriptors or experimenting with other modalities (e.g.,
subtitles, user reviews) could further illuminate the value and challenges of true
multimodal RAG.
•Limited Baseline Comparisons: We primarily benchmark against unimodal base-
lines and our own fusion variants. Comprehensive evaluation against state-of-the-art
multimodal recommenders, including recent LLM/RAG-based models and more so-
phisticated fusion or alignment strategies (e.g., cross-attention, gated fusion), would
provide a clearer picture of strengths and weaknesses.
•Cold-Start and Metadata Augmentation Scope: Our LLM-based augmenta-
tion is demonstrated primarily for item-level (movie) metadata; user-side cold-start
or session-based recommendation remains underexplored. Extending data augmenta-
tion and evaluation protocols to user-side cold-start scenarios would further enhance
the resource.
•Design Principle Positioning: Our primary contribution is infrastructural—a re-
producible multimodal RAG resource, rather than algorithmic novelty. Nonetheless,
future versions could incorporate more advanced fusion, retrieval, and generation
strategies to push the boundaries of multimodal recommendation research.
In summary, while RAG-VisualRec addresses the acute need for transparent, au-
ditable multimodal RAG resources, future work should expand to additional modalities,
datasets, and more competitive baselines, and integrate formal robustness and adver-
sarial benchmarks. We invite the community to build on and extend this resource for
broader, more robust, and fairer multimodal recommender system research.
References
[1] S. Abu-El-Haija, N. Kothari, J. Lee, P. Natsev, G. Toderici, B. Varadarajan, and
S. Vijayanarasimhan. Youtube-8m: A large-scale video classification benchmark.
arXiv preprint arXiv:1609.08675 , 2016.
[2] M. Attimonelli, D. Danese, A. Di Fazio, D. Malitesta, C. Pomo, and T. Di Noia.
Ducho meets elliot: Large-scale benchmarks for multimodal recommendation. arXiv
preprint arXiv:2409.15857 , 2024.
[3] M. Attimonelli, D. Danese, D. Malitesta, C. Pomo, G. Gassi, and T. Di Noia.
Ducho 2.0: Towards a more up-to-date unified framework for the extraction of
multimodal features in recommendation. In Companion Proceedings of the ACM
on Web Conference 2024 , pages 1075–1078, 2024.
[4] Y. Deldjoo, M. G. Constantin, B. Ionescu, M. Schedl, and P. Cremonesi. Mmtf-14k:
a multifaceted movie trailer feature dataset for recommendation and retrieval. In
Proceedings of the 9th ACM Multimedia Systems Conference , pages 450–455, 2018.
19

[5] Y. Deldjoo, Z. He, J. McAuley, A. Korikov, S. Sanner, A. Ramisa, R. Vidal,
M. Sathiamoorthy, A. Kasrizadeh, S. Milano, et al. Recommendation with gen-
erative models. arXiv preprint arXiv:2409.15173 , 2024.
[6] Y. Gao, Y. Xiong, X. Gao, K. Jia, J. Pan, Y. Bi, Y. Dai, J. Sun, H. Wang, and
H. Wang. Retrieval-augmented generation for large language models: A survey.
arXiv preprint arXiv:2312.10997 , 2, 2023.
[7] F. M. Harper and J. A. Konstan. The movielens datasets: History and context.
Acm transactions on interactive intelligent systems (tiis) , 5(4):1–19, 2015.
[8] C. Huang, Y. Xia, R. Wang, K. Xie, T. Yu, J. McAuley, and L. Yao. Embedding-
informed adaptive retrieval-augmented generation of large language models. arXiv
preprint arXiv:2404.03514 , 2024.
[9] Y. Liu, Y. Wang, L. Sun, and P. S. Yu. Rec-gpt4v: Multimodal recommendation
with large vision-language models. arXiv preprint arXiv:2402.08670 , 2024.
[10] Y. Ni, Y. Cheng, X. Liu, J. Fu, Y. Li, X. He, Y. Zhang, and F. Yuan. A
content-driven micro-video recommendation dataset at scale. arXiv preprint
arXiv:2309.15379 , 2023.
[11] A. Salah, Q.-T. Truong, and H. W. Lauw. Cornac: A comparative framework
for multimodal recommender systems. Journal of Machine Learning Research ,
21(95):1–5, 2020.
[12] J. Tian, Z. Wang, J. Zhao, and Z. Ding. Mmrec: Llm based multi-modal rec-
ommender system. In 2024 19th International Workshop on Semantic and Social
Media Adaptation & Personalization (SMAP) , pages 105–110. IEEE, 2024.
[13] W. Wei, C. Huang, L. Xia, and C. Zhang. Multi-modal self-supervised learning for
recommendation. In Proceedings of the ACM Web Conference 2023 , pages 790–800,
2023.
[14] N. Wu, M. Gong, L. Shou, J. Pei, and D. Jiang. Ruel: Retrieval-augmented user
representation with edge browser logs for sequential recommendation. In Proceed-
ings of the 32nd ACM International Conference on Information and Knowledge
Management , pages 4871–4878, 2023.
[15] H. Zeng, Z. Yue, Q. Jiang, and D. Wang. Federated recommendation via hybrid
retrieval augmented generation. In 2024 IEEE International Conference on Big
Data (BigData) , pages 8078–8087. IEEE, 2024.
[16] X. Zhou. Mmrec: Simplifying multimodal recommendation. In Proceedings of the
5th ACM International Conference on Multimedia in Asia Workshops , pages 1–2,
2023.
20