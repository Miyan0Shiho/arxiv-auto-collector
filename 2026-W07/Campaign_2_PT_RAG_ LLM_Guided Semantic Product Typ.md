# Campaign-2-PT-RAG: LLM-Guided Semantic Product Type Attribution for Scalable Campaign Ranking

**Authors**: Yiming Che, Mansi Mane, Keerthi Gopalakrishnan, Parisa Kaghazgaran, Murali Mohana Krishna Dandu, Archana Venkatachalapathy, Sinduja Subramaniam, Yokila Arora, Evren Korpeoglu, Sushant Kumar, Kannan Achan

**Published**: 2026-02-11 07:03:08

**PDF URL**: [https://arxiv.org/pdf/2602.10577v1](https://arxiv.org/pdf/2602.10577v1)

## Abstract
E-commerce campaign ranking models require large-scale training labels indicating which users purchased due to campaign influence. However, generating these labels is challenging because campaigns use creative, thematic language that does not directly map to product purchases. Without clear product-level attribution, supervised learning for campaign optimization remains limited. We present \textbf{Campaign-2-PT-RAG}, a scalable label generation framework that constructs user--campaign purchase labels by inferring which product types (PTs) each campaign promotes. The framework first interprets campaign content using large language models (LLMs) to capture implicit intent, then retrieves candidate PTs through semantic search over the platform taxonomy. A structured LLM-based classifier evaluates each PT's relevance, producing a campaign-specific product coverage set. User purchases matching these PTs generate positive training labels for downstream ranking models. This approach reframes the ambiguous attribution problem into a tractable semantic alignment task, enabling scalable and consistent supervision for downstream tasks such as campaign ranking optimization in production e-commerce environments. Experiments on internal and synthetic datasets, validated against expert-annotated campaign--PT mappings, show that our LLM-assisted approach generates high-quality labels with 78--90% precision while maintaining over 99% recall.

## Full Text


<!-- PDF content starts -->

Campaign-2-PT-RAG: LLM-Guided Semantic Product Type
Attribution for Scalable Campaign Ranking
Yiming Che∗
Walmart Global Tech
Bentonville, AR, USA
yiming.che@walmart.comMansi Mane∗
Walmart Global Tech
Sunnyvale, CA, USA
mansi.ranjit.mane@walmart.comKeerthi Gopalakrishnan∗
Walmart Global Tech
Sunnyvale, CA, USA
k.goppalakrishnan@walmart.com
Parisa Kaghazgaran∗
Walmart Global Tech
Sunnyvale, CA, USA
parisa.kaghazgaran@walmart.comMurali Mohana Krishna Dandu∗
Walmart Global Tech
Sunnyvale, CA, USA
murali.dandu@walmart.comArchana Venkatachalapathy∗
Walmart Global Tech
Sunnyvale, CA, USA
Archana.Venkatachala@walmart.com
Sinduja Subramaniam
Walmart Global Tech
Sunnyvale, CA, USA
sinduja.subramaniam@walmart.comYokila Arora
Walmart Global Tech
Sunnyvale, CA, USA
yokila.arora@walmart.comEvren Korpeoglu
Walmart Global Tech
Sunnyvale, CA, USA
ekorpeoglu@walmart.com
Sushant Kumar
Walmart Global Tech
Sunnyvale, CA, USA
sushant.kumar@walmart.comKannan Achan
Walmart Global Tech
Sunnyvale, CA, USA
kannan.achan@walmart.com
Abstract
E-commerce campaign ranking models require large-scale training
labels indicating which users purchased due to campaign influence.
However, generating these labels is challenging because campaigns
use creative, thematic language that does not directly map to prod-
uct purchases. Without clear product-level attribution, supervised
learning for campaign optimization remains limited. We present
Campaign-2-PT-RAG, a scalable label generation framework that
constructs user–campaign purchase labels by inferring which prod-
uct types (PTs) each campaign promotes. The framework first in-
terprets campaign content using large language models (LLMs) to
capture implicit intent, then retrieves candidate PTs through seman-
tic search over the platform taxonomy. A structured LLM-based
classifier evaluates each PT’s relevance, producing a campaign-
specific product coverage set. User purchases matching these PTs
generate positive training labels for downstream ranking models.
This approach reframes the ambiguous attribution problem into a
tractable semantic alignment task, enabling scalable and consistent
supervision for downstream tasks such as campaign ranking opti-
mization in production e-commerce environments. Experiments on
internal and synthetic datasets, validated against expert-annotated
campaign–PT mappings, show that our LLM-assisted approach gen-
erates high-quality labels with 78–90% precision while maintaining
over 99% recall.
∗equal contribution
This work is licensed under a Creative Commons Attribution 4.0 International License.
WWW Companion ’26, Dubai, United Arab Emirates.
©2026 Copyright held by the owner/author(s).
ACM ISBN 979-8-4007-2308-7/2026/04
https://doi.org/10.1145/3774905.3795730CCS Concepts
•Information systems →Retrieval models and ranking;Rec-
ommender systems;•Computing methodologies →Natu-
ral language processing;•Social and professional topics →E-
commerce.
Keywords
e-commerce, campaign ranking, recommendation system, retrieval-
augmented generation, large language models
ACM Reference Format:
Yiming Che, Mansi Mane, Keerthi Gopalakrishnan, Parisa Kaghazgaran,
Murali Mohana Krishna Dandu, Archana Venkatachalapathy, Sinduja Sub-
ramaniam, Yokila Arora, Evren Korpeoglu, Sushant Kumar, and Kannan
Achan. 2026. Campaign-2-PT-RAG: LLM-Guided Semantic Product Type
Attribution for Scalable Campaign Ranking. InCompanion Proceedings
of the ACM Web Conference 2026 (WWW Companion ’26), April 13–17,
2026, Dubai, United Arab Emirates.ACM, New York, NY, USA, 8 pages.
https://doi.org/10.1145/3774905.3795730
1 Introduction
E-commerce platforms increasingly rely on campaign experiences
delivered through multiple user-facing channels, such as splash
pages, home pages, and cart pages, to promote curated collections,
seasonal promotions, and other thematic campaigns that guide
product discovery and drive user conversion. Figure 1 (left) illus-
trates an example of such campaigns presented as instant messages
shown to users upon opening the app (i.e., a splash or pop-up page).
Unlike item-level recommendations, campaigns are broad in scope:
a single campaign may promote dozens of product types, often
conveyed through thematic, creative, or metaphorical language
rather than explicit product descriptions. This heterogeneity makesarXiv:2602.10577v1  [cs.IR]  11 Feb 2026

WWW Companion ’26, April 13–17, 2026, Dubai, United Arab Emirates. Che et al.
it difficult to determine whether a user has “purchased from” a
campaign and, consequently, to construct reliable user–campaign
labels for supervised campaign-ranking models.
In current industry practice, building such labels requires man-
ual curation. Analysts examine a campaign’s content, determine
which product types it intends to promote, and then inspect post-
exposure user purchases. While feasible for a small number of
curated campaigns, this approach is labor-intensive, inconsistent
across annotators, slow to update, and impractical at the scale
and pace of modern e-commerce environments. A key barrier to
automation is the semantic alignment problem. Campaigns are
written using creative or thematic language, e.g., taglines, mood
statements, lifestyle imagery, which rarely map cleanly onto the
structured product type (PT) taxonomy used in production systems.
Embedding-based retrieval partially mitigates this gap but often
over-retrieves broad concepts (e.g., “home goods”) or misses subtle
associations (e.g., “festival essentials” implicitly promoting portable
speakers and hydration gear). Embedding models lack the ability
to explicitly reason about implicit intent, contextual cues, and hier-
archical relationships within the product taxonomy, often leading
to either over-generalized or incomplete mappings.
To address this, we allow an LLM to interpret the campaign
holistically and generate a natural-language explanation of what
the campaign is promoting. This interpretation step captures la-
tent themes and inferred product semantics that are not explicitly
mentioned in the campaign text. When combined with retrieval re-
sults, it enables more accurate and interpretable mappings between
campaign content and specific product types.
Building on this insight, we introduce Campaign-2-PT-RAG , a
Retrieval-Augmented Generation (RAG) framework that infers cam-
paign–PT alignments using semantic retrieval and structured LLM
reasoning. The PT taxonomy provides canonical descriptions and hi-
erarchical relationships that enable consistent semantic grounding.
The pipeline retrieves candidate PTs based on the cosine similarity
of the campaign interpretation embedding and the PT embeddings
with a low threshold to include possible relevant PTs and then ap-
plies the LLM-based relevance classification to determine whether
each PT is strongly relevant, weakly relevant or irrelevant. This
multi-level relevance judgment mitigates retrieval noise, resolves
ambiguous matches, and captures nuanced thematic associations
that embedding models alone cannot. The resulting PT coverage is
then used to compute user–campaign labels based on post-exposure
purchases.
Experiments on both internal and synthetic datasets show that
Campaign-2-PT-RAG substantially improves PT-mapping precision,
recall, and coverage relative to traditional non-LLM-assisted base-
lines. These improvements translate to reliable and scalable label
construction suitable for downstream campaign-ranking applica-
tions in production. Our main contributions are:
•To the best of our knowledge, this work is the first to sys-
tematically study user–campaign purchase labeling in e-
commerce, a problem that has been largely overlooked com-
pared to item-level recommendation and ranking.
•We propose Campaign-2-PT-RAG , a scalable framework that
combines LLM-based campaign interpretation with retrievaland structured LLM relevance reasoning over a product-type
taxonomy to infer campaign-level product coverage.
•We demonstrate substantial improvements in PT coverage
precision, recall, and semantic coherence over traditional
lexical and embedding-based baselines using both real-world
campaigns and synthetic evaluations
•Beyond label construction, Campaign-2-PT-RAG opens the
door to campaign-aware learning in large-scale recommender
systems by transforming unstructured campaign content
into structured, interpretable product-type representations
that support campaign-aware ranking, evaluation, and long-
horizon optimization.
2 Related Work
Our work relates to previous research in three areas: (1) retrieval
methods for semantic matching, (2) retrieval-augmented generation
and LLM reasoning, and (3) LLM-based evaluation.
2.1 Retrieval for Semantic Matching
Semantic matching in information retrieval is commonly addressed
through lexical or dense retrieval approaches. Classical lexical re-
trieval methods such as BM25 [ 14] rank documents based on exact
term matching and remain strong baselines in many retrieval set-
tings. Dense bi-encoder models [ 6,13] enable efficient approximate
nearest-neighbor retrieval by embedding queries and candidates
into a shared vector space, allowing semantic similarity to be com-
puted at scale.
These retrieval techniques have proven effective for search and
recommendation tasks where user intent is explicitly expressed.
However, they primarily optimize similarity under surface-level
or distributional matching assumptions. In settings such as cam-
paign understanding, where intent is often implicit, thematic, and
multi-faceted, retrieval alone tends to over-select broadly related
product types and lacks an explicit mechanism to distinguish rele-
vance strength or reason over hierarchical relationships within a
product taxonomy. Our work builds on this retrieval foundation
but augments it with structured LLM-based reasoning to explicitly
model relevance strength and reduce semantic ambiguity.
2.2 Retrieval-Augmented Generation and LLM
Reasoning
RAG integrates large language models with external knowledge
sources to improve grounding and reasoning [ 2,7,15]. Recent
surveys provide a comprehensive overview of RAG architectures,
challenges, and performance trends in modern recommendation
systems [ 4,5]. RAG-based methods have been widely applied to
knowledge-intensive tasks such as open-domain question answer-
ing and contextual generation, where retrieved context guides gen-
eration and evidence aggregation.
In e-commerce applications, LLMs have been explored for con-
tent and user intent understanding [ 3,10,18] and conversational
recommendation [ 9,16,17]. Most existing work, however, focuses
on generating responses or ranking items rather than explicitly rea-
soning about structured relevance within a predefined taxonomy. In
contrast, our approach applies RAG to infer campaign-level product-
type coverage and leverages LLMs to perform explicit relevance

Campaign-2-PT-RAG: LLM-Guided Semantic Product Type Attribution for Scalable Campaign Ranking WWW Companion ’26, April 13–17, 2026, Dubai, United Arab Emirates.
reasoning, categorizing product types as strongly relevant, weakly
relevant, or irrelevant. This formulation allows LLMs to go beyond
surface-level semantic matching and address the granularity and
ambiguity inherent in campaign content.
2.3 LLM-based Evaluation
Evaluating semantic relevance is challenging when ground truth
is incomplete, subjective, or expensive to obtain. Recent studies
show that large language models can serve as effective evaluators
for tasks such as summarization quality, reasoning correctness, and
retrieval relevance, often correlating well with human judgments
[1,8,12]. LLM-based evaluation has therefore emerged as a practical
complement to traditional human annotation.
Following this line of work, we adopt an LLM-as-judge protocol
to evaluate semantic alignment between campaign content and
predicted product types, particularly in settings where explicit
labels are unavailable. This approach provides an additional layer of
semantic assessment, helping to validate model predictions beyond
standard precision and recall metrics.
3 Methodology
We present Campaign-2-PT-RAG , a retrieval-augmented framework
for inferring product-type coverage from campaign content and con-
structing scalable user–campaign purchase labels. The framework
addresses the semantic gap between creative campaign content
and structured product taxonomies by combining LLM-based cam-
paign interpretation, semantic retrieval, and structured relevance
reasoning. We first formalize the problem and then describe each
component of the pipeline. Finally, we provide an overview of the
end-to-end system architecture, illustrating how campaign signals
are ingested, processed through the LLM-assisted campaign-to-PT
inference pipeline, and integrated into downstream ranking and
serving components
3.1 Problem formulation
Let𝑐denote a campaign containing marketing content in the format
Campaign Title | Campaign Content . LetTbe the product-
type taxonomy, where each node consists of a product category,
family, and type (see Sec. 3.3). The concepts in Tare structured
hierarchically, with higher-level categories encompassing broader
product families and lower-level types representing fine-grained
distinctions. Users generate purchase events over time. For a user
𝑢, let𝑃 𝑢denote the set of purchased PTs. We aim to:
(1) Infer the set of PTs promoted by campaign𝑐:
PT(𝑐)⊆T,(1)
(2)Construct a user-campaign label indicating whether user 𝑢
purchased from the inferred PTs:
𝑦𝑢,𝑐=1(𝑃𝑢∩PT(𝑐)≠∅).(2)
The challenge lies in mapping the content of the creative cam-
paign to the correct PTs inT.3.2 Framework Overview
Campaign-2-PT-RAG comprises four sequential stages: (1) LLM-
based campaign interpretation to capture explicit and implicit in-
tent; (2) semantic retrieval of candidate PTs from the taxonomy; (3)
candidates reranking; and (4) LLM-based relevance classification of
retrieved PTs. Fig. 1 (right) illustrates the overall architecture. The
framework leverages the structured PT taxonomy as a knowledge
base, enabling consistent semantic grounding. The LLM interpreta-
tion step enriches the retrieval query with contextual understand-
ing, while the relevance classification step refines candidate selec-
tion through explicit reasoning. This combination addresses the
granularity mismatch between campaign text and structured prod-
uct representations, yielding robust and interpretable PT coverage
estimates.
3.3 Product-Type Knowledge Base
The product-type taxonomy Tserves as a structured knowledge
base providing hierarchical relationships. Each PT node consists
ofCategory : high-level grouping, Family : intermediate semantic
grouping and Type : fine-grained product type. They are represented
as concatenated text fields:
Category|Family|Type
For example, a PT node might be represented as Electronics
| Audio Equipment | Wireless Headphones . This structured
representation provides a compact yet expressive semantic back-
bone. Because campaigns promote concepts rather than specific
stock keeping unit (SKU), PT-level alignment is both more stable
across catalog updates and more robust to item-level textual noise.
In our setting, the knowledge base contains7 ,147PTs, making
manual campaign labeling extremely labor-intensive and impracti-
cal.
3.4 Campaign-2-PT Inference via RAG
Campaign content often employs thematic or creative language that
does not directly reference specific PT names. To address this, we
apply an LLM to interpret the campaign holistically and generate
a semantic summary 𝑠𝑐=LLM_Interpret(𝑐) . The summary cap-
tures latent themes and both explicit and implicit product intents,
providing a richer and more context-aware retrieval query than the
raw campaign text alone.
Using the interpreted summary 𝑠𝑐, we retrieve a set of candidate
PT nodes by computing cosine embedding similarity. Each PT node
𝑡is embedded using its structured fields, and we retain all PTs
whose cosine similarity with the campaign embedding exceeds a
predefined threshold𝜏:
𝑅(𝑐)={𝑡∈T|𝐶𝑜𝑠(Embed(𝑠 𝑐),Embed(𝑡))≥𝜏}.(3)
We choose𝜏to favor high recall, with precision improved by
subsequent processing. In addition to embedding-based retrieval,
we apply a cross-encoder semantic reranker [ 11] that computes
pairwise relevance scores between the campaign text and PT de-
scriptions. Cross-encoders capture fine-grained semantic interac-
tions through joint encoding of input pairs and have been shown
to be effective for neural reranking in information retrieval tasks.
In our pipeline, reranking refines the candidate set and provides
a more semantically ordered input for downstream LLM-based

WWW Companion ’26, April 13–17, 2026, Dubai, United Arab Emirates. Che et al.
Campaign
InterpreterCampaign Intent
FAISS RetrieverRaw Campaign
"bring the stadium experience home |
Score big savings on large screen TVs.
,Plus, free wall mounting for select
65"+ V izio TVs through 4/7,Mount not
included. ,75" or larger when purchasing
through the W almart App or online"
[2338] | Electronics | Home A udio | FM T ransmitters
[2918] | Electronics | Home A udio | Digital-to- Analog Con verters
[132]   | Electronics | TV & Video | P ortable Blu R ay & DVD Pla yers
[1368] | Electronics | Home A udio | A udio P ower Amplifiers
[1084] | Electronics | TV & Video | Analog-to-Digital Con verters
...{Category | Family | Type}RerankerThe campaign mentions large
screen TVs, wall mounts, and
specific TV sizes, which imply
home entertainment products.
Classifier [729] | Electronics | TV & Video | TV & Monitor Mounts
[1710] | Electronics | Default | Other TV & Video Accessories
[687] | Electronics | TV & Video | Home Theater S ystems
[1004] | Electronics | TV & Video | T elevisions
...Strong + Weak RelevantIrrelevant 
JudgePT Data
LLM
Optional Component
Figure 1: (Left) Illustration of a real-world e-commerce marketing campaign as presented on a splash page. (Right) Architecture
of the proposed Campaign-2-PT-RAG framework for automated product curation.
relevance classification. While reranking provides consistent but
modest performance improvements (see Sec. 4.6), it incurs addi-
tional computational cost and latency. Hence, this component is
treated as optional depending on deployment constraints.
For each retrieved PT 𝑡∈𝑅(𝑐) , we use an LLM to classify its
relevance to campaign𝑐. Given(𝑠 𝑐,𝑡), the LLM outputs one of:
{strong relevance,weak relevance,irrelevant}.
Formally,
Rel(𝑐,𝑡)=LLM_Classify(𝑠 𝑐,𝑡).(4)
The LLM-based relevance classifier performs explicit semantic rea-
soning beyond surface-level similarity. Specifically, it infers implicit
product intent expressed through creative or thematic campaign
language, reasons over the hierarchical structure of the product-
type taxonomy to resolve granularity mismatches, distinguishes
transactional relevance from loosely related thematic associations,
and incorporates negative evidence to filter product types that are
semantically similar but not promoted by the campaign. This rea-
soning step corrects retrieval noise, resolves hierarchical ambiguity,
and identifies implicit thematic relationships. We define the final
PT coverage as:
PT(𝑐)={𝑡∈𝑅(𝑐):Rel(𝑐,𝑡)∈{strong,weak} }.(5)
Given the inferred PT coverage, we assign a binary label indicat-
ing whether user 𝑢made a relevant purchase after exposure to
campaign𝑐:𝑦 𝑢,𝑐=1(𝑃𝑢∩PT(𝑐)≠∅).
3.5 System Architecture
Figure 2 presents the end-to-end system architecture integrating
real-time user interaction signals with the Campaign-2-PT-RAG
pipeline for scalable campaign attribution and ranking. The overall
workflow can be divided into three stages: signal ingestion and
triggering, campaign-to-PT mapping, and decisioning and serving.
Signal Ingestion and Triggering.User impression and click events
are ingested through a unified Kafka stream. Click signals are pro-
cessed via an hourly Spark job, while impression signals are for-
warded directly to the orchestration layer to minimize latency. Both
signals are consolidated inOrion, an internal event orchestration
service that coordinates downstream processing.Orion, a real-timeorchestration and triggering service, that emits notifications that
trigger inference inNeuronandHydra, which are the campaign
inference and decisioning service, which also consumes batch fea-
tures and historical aggregates stored inCassandra, a distributed
key–value store.
Campaign-to-PT Mapping via LLM-RAG..Campaign-to-PT map-
ping is performed using the proposed Campaign-2-PT-RAG pipeline
described in Section 3.4. For each incoming campaign, the LLM
first interprets campaign metadata to infer implicit intent, retrieves
candidate product types from the taxonomy, and applies structured
relevance classification to assign strong, weak, or irrelevant labels.
The resulting campaign-level PT coverage provides a compact and
interpretable representation of campaign intent, which serves as a
key semantic signal for downstream task.
Decisioning and Serving. Neuroncombines real-time triggers,
batch features, and campaign–PT signals to produce candidate
scores, which are ranked by Hydra and cached for low-latency serv-
ing through the personalization layer under campaign eligibility
constraints.
While we refer to internal system names for concreteness, these
components correspond to standard functions in large-scale rec-
ommender architectures, including event orchestration, model in-
ference, and ranking, and can be implemented using equivalent
services in other production environments.
4 Experiments
We evaluate Campaign-2-PT-RAG on the task of campaign-PT map-
ping, focusing on the intrinsic quality of inferred PT sets. Our
evaluation framework consists of three components: (1)Human-
labeled campaigns: measures accuracy against curated ground
truth, (2)Synthetic campaigns: measures recovery of implicit PT
associations, and (3)LLM-as-judge evaluation: provides seman-
tic relevance judgments to complement incomplete human labels.
Finally, the ablation studies are detailed in the Section 4.6.
4.1 Experiment Setup
We use GPT-4o (API version 2024–10–21) as the LLM for campaign
interpretation and relevance classification. For neural reranking,

Campaign-2-PT-RAG: LLM-Guided Semantic Product Type Attribution for Scalable Campaign Ranking WWW Companion ’26, April 13–17, 2026, Dubai, United Arab Emirates.
Figure 2: End-to-end system architecture integrating click signal processing with campaign-to-PT mapping. Named components
(e.g., Orion, Neuron) denote logical orchestration and inference services.
we employ the cross-encoder/ms-marco-MiniLM-L6-v2 model
from Hugging Face, a lightweight cross-encoder trained on the MS
MARCO passage ranking task. The embedding model used for re-
trieval is sentence-transformers/all-mpnet-base-v2 , a widely
used bi-encoder model for semantic search. The PT knowledge base
includes7,147internal PTs.
We compare Campaign-2-PT-RAG against the following base-
lines:
•BM25 Retrieval: A lexical retrieval baseline that ranks prod-
uct types using BM25 over PT text fields. This method serves
as a strong keyword-based baseline.
•LLM Zero-Shot Classification: The LLM is provided with
the full product-type list and asked to directly select relevant
PTs for a campaign. This approach is susceptible to position
bias and long-context degradation.
•Embedding Retrieval: Product types are directly retrieved
using cosine similarity between embeddings of raw cam-
paign and PT embeddings, with multiple similarity thresh-
olds.
All LLM components are used in a zero-shot or prompt-based man-
ner, without fine-tuning on campaign or purchase data.
4.2 Evaluation for Human-Labeled Campaigns
LetPTtrue(𝑐)denote the ground truth PT set for campaign 𝑐from
human labeling and PTpred(𝑐)the predicted set, i.e., the union of
strongly relevant set and weakly relevant set from LLM classifica-
tion. We report the following metrics.
Precision.
Precision(𝑐)=|PT pred(𝑐)∩PT true(𝑐)|
|PT pred(𝑐)|.(6)Recall.
Recall(𝑐)=|PT pred(𝑐)∩PT true(𝑐)|
|PT true(𝑐)|.(7)
F1 Score.
F1(𝑐)=2·Precision(𝑐)·Recall(𝑐)
Precision(𝑐)+Recall(𝑐).(8)
Semantic Coherence.It measures whether the predicted PT set is
semantically consistent:
Coherence(𝑐)=1
|𝑆|∑︁
(𝑡1,𝑡2)∈𝑆cosine(𝑡 1,𝑡2),(9)
where𝑆is the set of unordered PT embedding pairs in PTpred(𝑐).
Semantic coherence complements precision/recall by assessing the-
matic consistency even when ground truth is incomplete.
4.3 Evaluation for LLM-as-Judge
While human-labeled PT sets provide an essential source of ground
truth, they are inherently incomplete: annotators may overlook
weakly relevant PTs, disagree on implicit product associations, or
miss long-tail categories that are semantically aligned with the
campaign. To complement these limitations, we adopt anLLM-
as-judgeprotocol to provide an additional semantic evaluation of
model predictions. Given a campaign 𝑐and predicted PT 𝑡, an LLM
assigns: strongly relevant, weakly relevant and irrelevant. This
produces a relevance label LLMJudge(𝑐,𝑡) that serves as an external
semantic assessment. We compute the following metrics:
LLM-Precision.The fraction of predicted PTs judged as strongly
or weakly relevant, i.e.,
LLM-Precision(𝑐)=|{𝑡:LLMJudge(𝑐,𝑡)≠irrele.∧𝑡∈PT pred(𝑐)}|
|PT pred(𝑐)|
(10)

WWW Companion ’26, April 13–17, 2026, Dubai, United Arab Emirates. Che et al.
LLM-Recall.The proportion of LLM-relevant PTs recovered by
the model, computed over the union of PTs predicted by all systems,
i.e.,
LLM-Recall(𝑐)=|{𝑡:LLMJudge(𝑐,𝑡)≠irrele.∧𝑡∈PT pred(𝑐)}|
|{𝑡:LLMJudge(𝑐,𝑡)≠irrele.}|.
(11)
LLM Score.A holistic 0–1 rating of how well the predicted PT
set matches campaign intent.
4.4 Walmart Internal Campaign Evaluation
We evaluate Campaign-2-PT-RAG on a set of real-world Walmart
campaigns manually annotated by domain experts at Walmart. In
total, we sample eight campaigns covering diverse Walmart themes
such as Electronics, Fresh produce, Home Outdoor, Auto-Care, Back-
to-School preparation, and Home Essentials.
Fresh produceGroceries guaranteed fresh or your money back |
Fresh produce, delivered daily to our stores.
Each campaign is independently annotated by at least two ex-
perts, who identify product types relevant to the campaign intent.
The final ground-truth PT set is constructed by aggregating the rel-
evant PTs identified by all annotators and resolving disagreements
through discussion. Inter-annotator agreement is measured using
Jaccard distance (0.9412±0.1326).
Table 1 summarizes precision, recall, and F1 scores, reported as
mean±standard deviation across campaigns, using human annota-
tions as ground truth. Retrieval-only baselines achieve high recall
but relatively low precision, indicating substantial over-selection
of loosely related product types. For example, in the fresh pro-
duce campaign (see Section 4.4), which is interpreted by the LLM
asfocusing on groceries, including fresh produce and daily essen-
tials, retrieval-based methods using cosine similarity thresholds of
0.5and0.3incorrectly include product types such as snack boxes
and emergency food. Although these categories are broadly food-
related, they are not aligned with the specific intent of the campaign.
In addition, some clearly unrelated product types, such as Office
& Stationery | Money Handling | Cash Registers , appear
close to the raw campaign embedding in the vector space due to
superficial lexical or distributional similarity, yet are semantically
distant from the campaign intent. In contrast, Campaign-2-PT-RAG
effectively filters out both loosely related and spurious matches
through structured LLM-based relevance reasoning, resulting in
substantially improved precision while preserving high recall. We
also observe that using a high retrieval threshold (e.g.,0 .7) yields
very small predicted PT sets, which results in coincident precision,
recall, and F1 values.
Lexical retrieval method BM25 is particularly prone to spuri-
ous matches. In the same fresh produce campaign, BM25 retrieves
product types like Office & Stationery | Money Handling |
Money Deposit Bags solely due to lexical overlap with the term
“money” despite being entirely unrelated to food or grocery con-
tent. In contrast, Campaign-2-PT-RAG successfully filters out these
false positives through LLM-based relevance reasoning, resulting
in substantially improved precision while preserving recall.
The zero-shot LLM baseline performs poorly across all metrics,
reflecting the difficulty of directly reasoning over thousands of prod-
uct types without retrieval or structured context. When presentedTable 1: PT-mapping quality on real Walmart campaigns with
human annotations. For retrieval-based models, the value in
parentheses denotes the cosine similarity threshold used for
candidate selection. The best performance for each metric is
shown in bold.
Model Precision Recall F1
BM250.2013±0.1246 0.9532±0.0120 0.3276±0.1293
Zero-Shot LLM0.0415±0.0312 0.5043±0.4992 0.0932±0.0913
Retrieval(0.3)0.4015±0.28290.9973±0.00730.5263±0.2640
Retrieval(0.5)0.4020±0.2523 0.9922±0.0117 0.5328±0.2421
Retrieval(0.7)0.2500±0.4629 0.2500±0.4629 0.2500±0.4629
Campaign-2-PT0.8934±0.08250.9756±0.01720.9412±0.0376
with long, unfiltered PT lists, the LLM exhibits degraded precision
due to limited context capacity and a lack of explicit mechanisms
to compare relevance across many candidates. In addition, this ap-
proach incurs substantially higher token usage, as the entire PT
list must be included in the prompt for each campaign, making it
computationally inefficient.
To further assess semantic quality beyond exact overlap with
human labels, we apply the LLM-as-judge evaluation described
in Section 4.3. We first measure the agreement between human
annotations and LLM-judge annotations across eight Walmart in-
ternal campaigns using Jaccard similarity. The resulting agreement
score of0.9376±0.1120indicates strong consistency between the
LLM judge and human annotators, suggesting that the LLM reliably
captures campaign-level product-type relevance in this setting.
The results in terms of LLM-as-judge are shown in Table 2.
Campaign-2-PT-RAG achieves the best results in all metrics, indi-
cating that the predicted PT sets are not only accurate with respect
to human annotations but also semantically consistent and well-
aligned with campaign intent. These results highlight the benefit
of combining semantic retrieval with structured LLM reasoning for
campaign-level product type inference in real-world settings.
4.5 Synthetic Campaign Evaluation
In addition to human-labeled real-world campaigns, we evaluate
Campaign-2-PT-RAG using synthetically generated campaign de-
scriptions designed to resemble real-world e-commerce marketing.
These campaigns are generated by ChatGPT 5.2 in a concise, trans-
actional style (e.g., campaign title and brief description), similar to
Walmart campaigns used in production systems, e.g.,
Small-Space Living Solutions Essentials | Entertain easily with
casual pieces and accessories that are ready when guests arrive, with
free delivery available.
On-the-Go Essentials | Snack healthier with convenient options you
can grab on busy days without sacrificing taste Easy returns available.
Travel Light & Ready | Pack lighter for weekend trips with travel-
ready items that save space and simplify transitions. Enjoy simple,
reliable choices for every day.
The synthetic campaigns are not constructed with explicit product-
type labels, and therefore do not provide direct ground truth for PT
relevance.

Campaign-2-PT-RAG: LLM-Guided Semantic Product Type Attribution for Scalable Campaign Ranking WWW Companion ’26, April 13–17, 2026, Dubai, United Arab Emirates.
Table 2: PT-mapping quality on real Walmart campaigns with LLM judge. For retrieval-based models, the value in parentheses
denotes the cosine similarity threshold used for candidate selection. The best performance for each metric is shown in bold.
Model Precision Recall F1 Coherence LLM Score
BM250.2100±0.0946 1.0000±0.0000 0.3386±0.1285–0.5714±0.1955
Zero-Shot LLM0.0525±0.0514 0.5153±0.4896 0.0949±0.0919–0.1500±0.1309
Retrieval (0.3)0.4075±0.2914 0.9972±0.0077 0.5284±0.2740 0.6730±0.0654 0.6500±0.2000
Retrieval (0.5)0.4041±0.2543 0.9942±0.0119 0.5359±0.2466 0.6743±0.0644 0.7250±0.1982
Retrieval (0.7)0.2500±0.4629 0.2500±0.4629 0.2500±0.4629 0.2253±0.1355 0.2125±0.4016
Campaign-2-PT0.9054±0.0600 1.0000±0.0000 0.9495±0.0326 0.7976±0.0332 0.8900±0.0566
Because the promoted product domains are implicit and not di-
rectly observable from the generation process, we rely on the LLM-
as-judge protocol described in Section 3.3 to evaluate PT-mapping
quality. After each method predicts a set of relevant product types
for a campaign, an independent LLM assesses the semantic align-
ment between the campaign content and the predicted PTs, pro-
ducing relevance judgments and a set-level quality score. Given
the high agreement between LLM judge assessments and human
annotations observed on real-world campaigns, we consider the
LLM judge to be a reliable proxy for semantic evaluation in this
setting.
Table 3 summarizes the results of synthetic campaigns in terms of
LLM-as-judge. Lexical and embedding-only baselines again exhibit
high recall but lower precision and semantic coherence, reflecting
their tendency to retrieve broadly related PTs under ambiguous
campaign language. The zero-shot LLM baseline performs poorly,
underscoring the difficulty of directly reasoning over PTs without
retrieval or structured filtering. Campaign-2-PT-RAG consistently
achieves the highest F1 score, semantic coherence, and LLM score,
demonstrating improved robustness to implicit, abstract, and multi-
intent campaign language. These results suggest that LLM-assisted
campaign interpretation and relevance reasoning are critical for
effective PT inference when explicit supervision is unavailable.
4.6 Ablation Studies
We conduct ablation studies to quantify the contribution of each
component in Campaign-2-PT-RAG , with results summarized in
Table 4 using the LLM-as-judge evaluation on synthetic campaigns.
A retrieval-only baseline achieves high recall but low precision
and semantic coherence, indicating substantial over-selection of
loosely related product types and confirming that dense retrieval
alone is insufficient for disambiguating campaign intent. Adding
LLM-based campaign interpretation before retrieval significantly
improves precision and coherence by producing a semantically en-
riched query that better captures implicit campaign intent, though
some weakly related PTs remain without explicit filtering. Intro-
ducing LLM-based relevance classification yields the largest gains
in precision, F1, and LLM score, demonstrating the importance
of structured reasoning over retrieved candidates to distinguish
strong relevance from weak or irrelevant associations. The full
Campaign-2-PT-RAG pipeline, which combines campaign interpre-
tation, retrieval, reranking, and LLM-based relevance classification,
achieves the best performance across all metrics, confirming thatthese components provide complementary benefits and that ex-
plicit campaign interpretation and LLM reasoning are the primary
drivers of improved semantic alignment.
5 Discussion and Limitations
While Campaign-2-PT-RAG improves campaign–PT mapping, it re-
lies on LLMs whose performance and cost may vary across deploy-
ments. Although LLM-based relevance reasoning reduces semantic
ambiguity, relevance judgments remain inherently subjective, even
among human annotators. In addition, the optional reranking and
LLM inference stages introduce latency that may not be suitable
for all real-time settings. Future work will explore lighter-weight
reasoning models and tighter integration with downstream ranking
objectives.
6 Conclusion
We introduced Campaign-2-PT-RAG , a RAG framework for scalable
construction of user–campaign purchase labels through seman-
tic alignment of campaign content with product-type taxonomies.
By combining LLM interpretation, semantic retrieval, and struc-
tured relevance classification, the framework significantly improves
the quality of campaign–PT mappings. By enabling scalable and
interpretable campaign labeling, Campaign-2-PT-RAG provides a
practical foundation for deploying user-level campaign ranking in
large-scale e-commerce systems.
References
[1]Negar Arabzadeh and Charles LA Clarke. 2025. A human-ai comparative analysis
of prompt sensitivity in llm-based relevance judgment. InProceedings of the 48th
International ACM SIGIR Conference on Research and Development in Information
Retrieval. 2784–2788.
[2]Akari Asai, Zeqiu Wu, Yizhong Wang, Avi Sil, and Hannaneh Hajishirzi. 2024.
Self-RAG: Learning to Retrieve, Generate, and Critique through Self-Reflection.
InInternational Conference on Learning Representations.
[3]Alexander Brinkmann, Nick Baumann, and Christian Bizer. 2024. Using LLMs
for the extraction and normalization of product attribute values. InEuropean
Conference on Advances in Databases and Information Systems. Springer, 217–230.
[4]Yunfan Gao, Yun Xiong, Xinyu Gao, Kangxiang Jia, Jinliu Pan, Yuxi Bi, Yixin
Dai, Jiawei Sun, Haofen Wang, and Haofen Wang. 2023. Retrieval-augmented
generation for large language models: A survey.arXiv preprint arXiv:2312.10997
2, 1 (2023).
[5]Shailja Gupta, Rajesh Ranjan, and Surya Narayan Singh. 2024. A comprehensive
survey of retrieval-augmented generation (rag): Evolution, current landscape
and future directions.arXiv preprint arXiv:2410.12837(2024).
[6]Vladimir Karpukhin, Barlas Oguz, Sewon Min, Patrick SH Lewis, Ledell Wu,
Sergey Edunov, Danqi Chen, and Wen-tau Yih. 2020. Dense Passage Retrieval for
Open-Domain Question Answering.. InEMNLP (1). 6769–6781.
[7]Patrick Lewis et al .2020. Retrieval-Augmented Generation for Knowledge-
Intensive NLP Tasks. InNeurIPS.

WWW Companion ’26, April 13–17, 2026, Dubai, United Arab Emirates. Che et al.
Table 3: PT-mapping quality on synthetic campaigns with LLM judge. For retrieval-based models, the value in parentheses
denotes the cosine similarity threshold used for candidate selection. The best performance for each metric is shown in bold.
Model Precision Recall F1 Coherence LLM Score
BM250.3442±0.2127 0.9837±0.1050 0.4738±0.2364–0.5869±0.1868
Zero-Shot LLM0.0568±0.0496 0.4416±0.3623 0.0990±0.0844–0.3147±0.1338
Retrieval (0.3)0.3632±0.1691 0.9893±0.1001 0.5106±0.1825 0.7001±0.0386 0.6693±0.1373
Retrieval (0.5)0.3536±0.1680 0.9953±0.0308 0.5002±0.1790 0.7009±0.0385 0.6522±0.1395
Retrieval (0.7)0.2081±0.3908 0.2300±0.4230 0.2163±0.4012 0.1238±0.2975 0.1940±0.3668
Campaign-2-PT0.8219±0.1614 0.9926±0.0032 0.8718±0.1003 0.7876±0.0335 0.8250±0.0894
Table 4: Ablation studies on PT-mapping quality on synthetic campaigns with LLM judge. For retrieval-based models, the value
in parentheses denotes the cosine similarity threshold used for candidate selection. The best performance for each metric is
shown in bold.
Model Precision Recall F1 Coherence LLM Score
Retrieval Only (0.5)0.3536±0.1680 0.9953±0.0308 0.5002±0.1790 0.7009±0.0385 0.6522±0.1395
Des. + Retrieval0.4690±0.1848 0.9972±0.0083 0.6163±0.1746 0.7294±0.0381 0.7533±0.1124
Des. + Retrieval + LLM0.7708±0.15160.9996±0.00270.8656±0.10410.7607±0.04630.8286±0.0847
Des. + Retrieval + Reranker + LLM0.7819±0.18140.9976±0.00220.8688±0.10130.7576±0.04350.8350±0.0896
[8]Dawei Li, Bohan Jiang, Liangjie Huang, Alimohammad Beigi, Chengshuai Zhao,
Zhen Tan, Amrita Bhattacharjee, Yuxuan Jiang, Canyu Chen, Tianhao Wu, et al .
2025. From generation to judgment: Opportunities and challenges of llm-as-
a-judge. InProceedings of the 2025 Conference on Empirical Methods in Natural
Language Processing. 2757–2791.
[9]Yuanxing Liu, Weinan Zhang, Yifan Chen, Yuchi Zhang, Haopeng Bai, Fan Feng,
Hengbin Cui, Yongbin Li, and Wanxiang Che. 2023. Conversational recommender
system and large language model are made for each other in E-commerce pre-
sales dialogue. InFindings of the Association for Computational Linguistics: EMNLP
2023. 9587–9605.
[10] Robyn Loughnane, Jiaxin Liu, Zhilin Chen, Zhiqi Wang, Joseph Giroux, Tianchuan
Du, Benjamin Schroeder, and Weiyi Sun. 2024. Explicit Attribute Extraction in
E-Commerce Search. InProceedings of the Seventh Workshop on e-Commerce and
NLP@ LREC-COLING 2024. 125–135.
[11] Rodrigo Nogueira and Kyunghyun Cho. 2019. Passage Re-ranking with BERT.
arXiv:1901.04085(2019).
[12] Hossein A Rahmani, Emine Yilmaz, Nick Craswell, and Bhaskar Mitra. 2024.
JudgeBlender: Ensembling Judgments for Automatic Relevance Assessment.CoRR
(2024).
[13] Nils Reimers and Iryna Gurevych. 2019. Sentence-BERT: Sentence Embeddings
using Siamese BERT-Networks. InProceedings of the 2019 Conference on EmpiricalMethods in Natural Language Processing and the 9th International Joint Conference
on Natural Language Processing (EMNLP-IJCNLP). 3982–3992.
[14] Stephen Robertson and Hugo Zaragoza. 2009. The Probabilistic Relevance Frame-
work: BM25 and Beyond.Foundations and Trends in Information Retrieval(2009).
classic BM25 reference.
[15] Qiaoyu Tang, Jiawei Chen, Zhuoqun Li, Bowen Yu, Yaojie Lu, Haiyang Yu, Hongyu
Lin, Fei Huang, Ben He, Xianpei Han, et al .2024. Self-retrieval: End-to-end infor-
mation retrieval with one large language model.Advances in Neural Information
Processing Systems37 (2024), 63510–63533.
[16] Xiaolei Wang, Xinyu Tang, Wayne Xin Zhao, Jingyuan Wang, and Ji-Rong Wen.
2023. Rethinking the Evaluation for Conversational Recommendation in the Era
of Large Language Models. InProceedings of the 2023 Conference on Empirical
Methods in Natural Language Processing. 10052–10065.
[17] Zihuai Zhao, Wenqi Fan, Jiatong Li, Yunqing Liu, Xiaowei Mei, Yiqi Wang, Zhen
Wen, Fei Wang, Xiangyu Zhao, Jiliang Tang, et al .2024. Recommender systems
in the era of large language models (llms).IEEE Transactions on Knowledge and
Data Engineering36, 11 (2024), 6889–6907.
[18] Tiangang Zhu, Yue Wang, Haoran Li, Youzheng Wu, Xiaodong He, and Bowen
Zhou. 2020. Multimodal Joint Attribute Prediction and Value Extraction for
E-commerce Product. InProceedings of the 2020 Conference on Empirical Methods
in Natural Language Processing (EMNLP). 2129–2139.