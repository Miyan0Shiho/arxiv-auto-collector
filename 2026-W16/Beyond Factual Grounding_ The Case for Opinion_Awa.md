# Beyond Factual Grounding: The Case for Opinion-Aware Retrieval-Augmented Generation

**Authors**: Aditya Agrawal, Alwarappan Nakkiran, Darshan Fofadiya, Alex Karlsson, Harsha Aduri

**Published**: 2026-04-13 23:39:39

**PDF URL**: [https://arxiv.org/pdf/2604.12138v1](https://arxiv.org/pdf/2604.12138v1)

## Abstract
RAG systems have transformed how LLMs access external knowledge, but we find that current implementations exhibit a bias toward factual, objective content, as evidenced by existing benchmarks and datasets that prioritize objective retrieval. This factual bias - treating opinions and diverse perspectives as noise rather than information to be synthesized - limits RAG systems in real-world scenarios involving subjective content, from social media discussions to product reviews. Beyond technical limitations, this bias poses risks to transparent and accountable AI: echo chamber effects that amplify dominant viewpoints, systematic underrepresentation of minority voices, and potential opinion manipulation through biased information synthesis. We formalize this limitation through the lens of uncertainty: factual queries involve epistemic uncertainty reducible through evidence, while opinion queries involve aleatoric uncertainty reflecting genuine heterogeneity in human perspectives. This distinction implies that factual RAG should minimize posterior entropy, whereas opinion-aware RAG must preserve it. Building on this theoretical foundation, we present an Opinion-Aware RAG architecture featuring LLM-based opinion extraction, entity-linked opinion graphs, and opinion-enriched document indexing. We evaluate our approach on e-commerce seller forum data, comparing an Opinion-Enriched knowledge base against a traditional baseline. Experiments demonstrate substantial improvements in retrieval diversity: +26.8% sentiment diversity, +42.7% entity match rate, and +31.6% author demographic coverage on entity-matched documents. Our results provide empirical evidence that treating subjectivity as a first-class citizen yields measurably more representative retrieval-a first step toward opinion-aware RAG. Future work includes joint optimization of retrieval and generation for distributional fidelity.

## Full Text


<!-- PDF content starts -->

Beyond Factual Grounding: The Case for Opinion-Aware
Retrieval-Augmented Generation
Aditya Agrawal, Alwarappan Nakkiran, Darshan Fofadiya, Alex Karlsson, Harsha Aduri
Amazon.com
USA
Abstract
Retrieval-Augmented Generation (RAG) systems have transformed
how Large Language Models access external knowledge, but we
find that current implementations exhibit a bias toward factual, ob-
jective content, as evidenced by existing benchmarks and datasets
that prioritize objective retrieval. This factual bias—treating opin-
ions and diverse perspectives as noise rather than information to
be synthesized—limits RAG systems in real-world scenarios involv-
ing subjective content, from social media discussions to product
reviews and policy debates. Beyond technical limitations, this bias
poses risks to transparent and accountable AI: echo chamber effects
that amplify dominant viewpoints, systematic underrepresentation
of minority voices, and potential opinion manipulation through
biased information synthesis. We formalize this limitation through
the lens of uncertainty: factual queries involve epistemic uncer-
tainty reducible through evidence, while opinion queries involve
aleatoric uncertainty reflecting genuine heterogeneity in human
perspectives. This distinction implies that factual RAG should mini-
mize posterior entropy, whereas opinion-aware RAG must preserve
it. Building on this theoretical foundation, we present an Opinion-
Aware RAG architecture featuring LLM-based opinion extraction,
entity-linked opinion graphs, and opinion-enriched document in-
dexing. We evaluate our approach on e-commerce seller forum
data, comparing an Opinion-Enriched knowledge base against a
traditional baseline. Experiments demonstrate substantial improve-
ments in retrieval diversity: +26.8% sentiment diversity, +42.7%
entity match rate, and +31.6% author demographic coverage on
entity-matched documents. Our results provide empirical evidence
that treating subjectivity as a first-class citizen yields measurably
more representative retrieval—a first step toward opinion-aware
RAG systems. Future work includes custom retrievers optimized for
our formalized opinion-aware loss function and end-to-end learn-
ing approaches that jointly optimize retrieval and generation for
distributional fidelity.
CCS Concepts
•Information systems →Retrieval models and ranking;Sen-
timent analysis;•Computing methodologies →Natural lan-
guage processing.
Keywords
Retrieval-Augmented Generation, Opinion Mining, Large Language
Models, Bias Evaluation, Perspective Diversity
1 Introduction
Retrieval-Augmented Generation has become an important method
for making Large Language Models more reliable and factuallygrounded [ 1,2]. By conditioning generation on external evidence re-
trieved at inference time, RAG systems have shown success in reduc-
ing hallucinations and improving performance across knowledge-
intensive tasks [ 3,4]. Recent evaluations have confirmed RAG’s
effectiveness in enabling accurate, up-to-date, and verifiable text
generation across diverse applications [ 5]. This approach has en-
abled LLMs to access current information, provide transparent
sourcing, and ground their outputs in verifiable evidence.
However, beneath this success lies a fundamental assumption
that has largely gone unexamined: that knowledge bases should
contain factually correct, objective information. This assumption
manifests across multiple levels of the RAG ecosystem—from the
datasets used for evaluation, to the retrieval mechanisms optimized
for factual recall, to the generation objectives that prioritize con-
sistency with ground truth answers. While this factual orientation
has driven advances in question answering and knowledge synthe-
sis, it leaves the handling of inherently subjective content—which
pervades real-world information landscapes—as an unexamined,
non-systematic capability rather than an explicit design goal.
Consider the disparity between how current systems handle
these two queries:“What is the corporate tax rate in the United
States?”versus“What do small business owners think about the cur-
rent tax policy?”The first query has a definitive, factual answer
that existing RAG systems excel at retrieving and presenting. The
second query requires synthesizing diverse, potentially conflicting
opinions from multiple sources. While some large language models
demonstrate emergent opinion synthesis capabilities—potentially
drawing from patterns learned during training (see Appendix C
for an example from contemporary LLMs)—systematic RAG ar-
chitectures designed explicitly for transparent, reproducible, and
evaluable opinion synthesis remain largely underexplored. The
difference between accidental capability and intentional design be-
comes important when deploying systems in applications where
understanding the methodology, sources, and potential biases of
opinion synthesis is essential.
This limitation becomes particularly pronounced in domains
where understanding human opinion is crucial. Despite growing en-
terprise adoption across customer service, knowledge management,
and domain-specific applications [ 6,7,8], RAG systems remain op-
timized primarily for factual accuracy, creating a systematic gap in
their ability to handle the subjective, opinion-rich content prevalent
in market research, policy analysis, social listening, customer feed-
back synthesis, and personalized content generation. The timing is
critical: as RAG deployments expand beyond traditional knowledge
bases into social media, customer forums, and user-generated con-
tent [ 9,10], the mismatch between factual-oriented architectures
and opinion-rich data sources risks creating echo chambers that ex-
clude minority perspectives and present artificial consensus where
genuine disagreement exists. In these contexts, the goal is not toarXiv:2604.12138v1  [cs.AI]  13 Apr 2026

Aditya Agrawal, Alwarappan Nakkiran, Darshan Fofadiya, Alex Karlsson, Harsha Aduri
findthetruth, but to understand and representmultiple truthsas
expressed through human experiences, preferences, and judgments.
Addressing this gap requires rethinking how systems handle
different information types. Humans employ distinct cognitive
mechanisms when processing factual versus socially-transmitted
information [11, 12]—understanding not justwhatis said butwho
says it andwhy. Current RAG systems lack these mechanisms, and
approaches to “robustness” primarily focus on filtering contradic-
tory information [ 13,14], treating opinions as defects rather than
valuable information to be synthesized.
This paper introducesOpinion-Aware RAG, a framework that
explicitly models and leverages subjectivity throughout the retrieval-
generation pipeline. Rather than seeking consensus or filtering
disagreement, opinion-aware systems embrace the complexity of
human discourse, providing users with nuanced understanding of
how different groups, individuals, or contexts shape perspectives
on topics of interest.
Our contributions are: (1) A novel theoretical framework for-
malizing opinion-aware retrieval through the lens of epistemic vs.
aleatoric uncertainty (2) An Opinion-Aware RAG architecture—
the first systematic approach to making retrieval systems opinion-
aware—featuring LLM-based opinion extraction and entity-linked
opinion graphs; (3) Empirical evaluation on e-commerce forum
data demonstrating +26.8% sentiment diversity and +42.7% entity
match rate, showing that opinion enrichment yields measurably
more representative retrieval than semantic similarity alone; and
(4) An evaluation framework measuring opinion coverage, per-
spective diversity, and representation fairness. While this paper
presents a complete system—from theoretical framework to em-
pirical validation—we view our most important contribution as
drawing systematic attention to opinion-aware RAG as a research
priority.
2 Motivation
2.1 Factual Bias Across the RAG Ecosystem
Surveys of RAG systems [ 2,5] document a field-wide consensus
around factual accuracy as the primary evaluation criterion. Of
35 major RAG benchmarks surveyed (Appendix A), only 1 (2.9%)
addresses opinion synthesis. As Table 1 demonstrates, the entire
RAG ecosystem has evolved around assumptions of objective truth:
benchmarks like Natural Questions [ 15], MS MARCO [ 16], and
HotpotQA [ 17] all assume questions have single correct answers;
evaluation frameworks treat contradictory information as noise
rather than valuable diversity; and retrieval architectures optimize
for semantic convergence rather than perspective coverage. Only
OpinioRAG [ 18] addresses opinion synthesis. This factual orienta-
tion becomes problematic when users seek to understand public
sentiment, explore arguments on controversial topics, or gauge
community reactions, using contemporary RAG systems.
2.2 Emergent Synthesis and Evaluation Blind
Spots
Some LLMs demonstrate emergent opinion synthesis capabilities—
responding to queries like“What do small business owners think
about current tax policy?”with nuanced summaries capturing di-
verse viewpoints, acknowledging tensions, and presenting balancedTable 1: Representative 7 of 35 RAG evaluation datasets sur-
veyed (2018-2025). Only OpinioRAG addresses opinion syn-
thesis; others prioritize factual accuracy. See Appendix A for
complete analysis.
Dataset Year Evaluation Focus Opinion
OpinioRAG 2025 Sentiment Alignment Yes
RAGBench 2024 TRACe Metrics No
CRAG 2024 Accuracy, Hallucination No
FActScore 2023 Atomic Fact Accuracy No
BEIR 2021 nDCG, MAP, Recall No
Natural Questions 2019 EM, Token F1 No
HotpotQA 2018 EM, F1, Support F1 No
perspectives rather than seeking single “correct” answers (see Ap-
pendix C). However, when responses sound balanced, it remains
unclear whether this reflects genuine diversity in retrieved con-
tent or the model’s learned tendency—often attributed to RLHF—to
present information evenhandedly regardless of input [ 19]. Without
transparency into retrieval, users cannot distinguish substantive
balance from stylistic hedging. This raises fundamental questions:
Retrieval Opacity.How do retrieval systems translate opin-
ion queries into search strategies that surface diverse perspectives
rather than just semantically similar content? Current embedding
models, trained primarily on factual content, may embed opposing
opinions similarly if they discuss the same topics—making it diffi-
cult to distinguish “Tax cuts help small businesses” from “Tax cuts
primarily benefit large corporations.” Retrieval systems provide
no transparency into which perspectives were considered versus
systematically excluded.
Synthesis Opacity.Even when diverse opinions are retrieved,
users cannot understand how conflicting perspectives were weighted
or what constitutes “balanced representation.” Unlike factual re-
trieval where citation suffices, opinion synthesis requires under-
standing how conflicting viewpoints are reconciled.
The limitations of current evaluation approaches become ap-
parent when examining how traditional RAG metrics can assign
high scores to obviously biased outputs. Recent work on bias am-
plification [ 20,21,22] demonstrates that RAG systems can inherit
and amplify biases present in their retrieval corpora—for instance,
leadership discussions that overrepresent masculine-coded traits
like “assertiveness” and “aggressive persistence” while underrepre-
senting collaborative qualities. If a knowledge base contains such
skewed content, a query like“What leadership qualities are most
important for startup CEOs?”would retrieve and synthesize this
bias into the response (see Appendix C.2). Such a response could
score excellently on traditional metrics—high ROUGE-L from lexi-
cal overlap with retrieved sources, strong context relevance from
semantic alignment, high answer faithfulness from accurately re-
flecting the retrieval set—yet systematically underrepresent certain
perspectives. This illustrates how RAG systems can produce out-
puts that are simultaneously high-quality by traditional measures
and biased by human judgment.

Opinion-Aware RAG
3 Related Works
Recent RAG research frames subjectivity as a problem to be solved
rather than information to be leveraged. RAGuard [ 23] treats Reddit
discussions as “naturally occurring misinformation” to be filtered;
CONFLICTS [ 24] focuses on conflict resolution rather than preserv-
ing perspective diversity; and robustness approaches [ 25,26,27]
emphasize filtering and source verification. While valuable for fac-
tual domains, these approaches are misaligned with scenarios where
understanding diverse perspectives is the goal.
Work on argument-based retrieval has begun exploring how
systems can handle argumentative content. RAFTS [ 28] synthesizes
contrastive arguments for fact verification, explicitly retrieving both
supporting and refuting evidence, though it focuses on claim cred-
ibility rather than synthesizing diverse perspectives. Other work
explores argument mining [ 29] and perspective-aware retrieval,
but typically addresses isolated components. A deeper challenge
emerges from findings that LLMs struggle with conflicting infor-
mation [ 30,31]—even state-of-the-art models perform only slightly
better than random guessing at contradiction detection, exhibiting
high precision but low recall [ 31]. This limitation makes opinion-
aware RAG particularly challenging, since opinions are inherently
contradictory.
Opinion mining and stance detection research has matured sig-
nificantly [ 32,33,34], but these capabilities remain largely isolated
from RAG frameworks [ 35,36,37]. OpinioRAG [ 18] bridges this
gap with RAG-based opinion summarization from product reviews.
While OpinioRAG focuses onsummarization—condensing retrieved
opinions into coherent narratives—our work addresses the com-
plementary challenge ofretrieval diversity: ensuring the retrieved
document set faithfully represents the full spectrum of opinions
before summarization.
Sophisticated approaches have been developed for opinion-rich
content on social networking websites, including stance detec-
tion [ 38,39], multimodal opinion analysis [ 40], and comment senti-
ment analysis [ 41,42,43]. CrediRAG [ 44] demonstrates how social
network structures can enhance misinformation detection. How-
ever, these contributions remain disconnected from broader RAG
research, and the wealth of naturally occurring opinion data re-
mains underutilized for training opinion-aware systems.
Simply combining these fragmented approaches is insufficient:
factual RAG optimizes for convergence toward ground truth, while
opinion-aware systems must preserve divergence, and no unified
framework exists for jointly optimizing retrieval diversity, synthesis
fidelity, and demographic fairness. This gap motivates our formal
problem formulation.
4 Theoretical Framework
We formalize the distinction between factual and opinion-aware
retrieval through uncertainty quantification and information theory.
Throughout, we use: 𝑞(query),𝑦(response),D(corpus),𝜃(latent
state),𝑃pop(𝜃)(population opinion distribution), R(𝑞) (retrieved
documents), 𝜙:D→R𝑚(opinion embedding), 𝐻(·) (entropy),
𝐷KL(KL divergence), andW 𝑝(Wasserstein distance).4.1 Epistemic vs. Aleatoric Uncertainty
Consider a query 𝑞and corpusD. The posterior predictive distri-
bution is:
𝑃(𝑦|𝑞,D)=∫
𝑃(𝑦|𝜃,𝑞)𝑃(𝜃|D)𝑑𝜃(1)
This computes the probability of response 𝑦by considering all
possible underlying states 𝜃, weighted by their likelihood given
evidenceD.
Forfactual queries, a unique true state 𝜃∗exists. Uncertainty
isepistemic—reducible through additional evidence. As corpus size
grows, the posterior collapses: e.g., for “What is 2+2?”, 𝑃(𝜃|D)→
𝛿(𝜃−4), where𝛿(·)is the Dirac delta:
lim
|D|→∞𝑃(𝜃|D)=𝛿(𝜃−𝜃∗)(2)
Foropinion queries, uncertainty isaleatoric—reflecting gen-
uine heterogeneity rather than ignorance. The posterior converges
to the population distribution, not a point: e.g., for “Is pineapple on
pizza good?”, we converge to perhaps 40% favor, 45% oppose, 15%
indifferent:
lim
|D|→∞𝑃(𝜃|D)=𝑃 pop(𝜃)(3)
This distinction has profound implications for system design:fac-
tual RAG minimizes posterior entropy; opinion-aware RAG must pre-
serve it.We note that opinion queries still involve epistemic uncer-
tainty about 𝑃popitself—O-RAG aims to reduce this epistemic com-
ponent through comprehensive retrieval while preserving aleatoric
uncertainty by representing discovered diversity rather than col-
lapsing to a single view.
4.2 Divergent Optimization Objectives
Standard RAG maximizes information gain by minimizing condi-
tional entropy:
Lfactual=𝐻(𝑌|𝑞)−𝐻(𝑌|𝑞,R(𝑞))(4)
This rewards retrieval that eliminates uncertainty—correct for facts,
counterproductive for opinions.
For opinions, we seekdistributional fidelity. Let 𝑣denote a posi-
tion in opinion space and 𝑃pop(𝑣|𝑞) the true opinion distribution.
The opinion-aware objective is:
LOA=𝐷 KL 𝑃pop(𝑣|𝑞)∥𝑃 model(𝑣|𝑞)−𝜆𝐻(𝑃 model)(5)
The first term drives distributional accuracy; the second (maximiz-
ing entropy) penalizes artificial consensus.
4.3 Retrieval as Coverage Optimization
LetR(𝑞)={𝑑 1,...,𝑑𝑘}. The embedding 𝜙:D→R𝑚maps doc-
uments to opinion space; in our implementation (Section 5), 𝜙
encodes sentiment, sentiment polarity, and topical framing. The
empirical opinion distribution from retrieval is:
ˆ𝑃R=1
𝑘𝑘∑︁
𝑖=1𝛿𝜙(𝑑 𝑖) (6)
where𝛿𝜙(𝑑 𝑖)is a point mass at document𝑑 𝑖’s opinion embedding.
We measure coverage using the𝑝-Wasserstein distance:
W𝑝(ˆ𝑃R,𝑃pop)= 
inf
𝛾∈Γ( ˆ𝑃R,𝑃pop)∫
∥𝜙−𝜙′∥𝑝𝑑𝛾(𝜙,𝜙′)!1/𝑝
(7)

Aditya Agrawal, Alwarappan Nakkiran, Darshan Fofadiya, Alex Karlsson, Harsha Aduri
Intuitively,W𝑝(earth mover’s distance) measures the minimum
“transportation cost” to reshape one distribution into another—
naturally penalizing retrieval that misses regions of opinion space.
We prefer it over Jensen-Shannon divergence as it remains well-
defined when distributions have non-overlapping support, as we
demonstrate empirically in Section 6.
4.4 Generation as Calibrated Synthesis
Let𝑃reader(𝜙|𝑦) represent the opinion distribution a reader infers
from response𝑦. Faithful generation requires:
𝑃reader(𝜙|𝑦)≈𝑘∑︁
𝑖=1𝑤𝑖𝛿𝜙(𝑑 𝑖) (8)
where𝑤𝑖are importance weights (Í
𝑖𝑤𝑖=1). This induces a cali-
bration requirement:
Var𝑃reader[𝜙|𝑦]≈Var 𝑃pop[𝜙|𝑞](9)
Overconfident synthesis—low-variance output for high-variance
opinion landscapes—misrepresents societal consensus.
4.5 Unified Objective
Opinion-Aware RAG minimizes:
L=W 2(𝑃pop,ˆ𝑃R)
|           {z           }
coverage+𝛼𝐷 KL(𝑃pop∥𝑃reader)
|                {z                }
fidelity+𝛽max
𝑔∈𝐺Δ𝑔
|  {z  }
fairness(10)
The first term ensures retrieval captures the opinion spectrum
via 2-Wasserstein distance. The second ensures faithful communi-
cation to readers ( 𝛼>0). The third enforces demographic equity:
Δ𝑔=𝐷 KL(𝑃pop(𝜙|𝑔)∥ ˆ𝑃R(𝜙|𝑔)) measures representation disparity
for demographic group 𝑔∈𝐺 (e.g., business size, tenure, region).
By minimizing maximum group disparity, we adopt a minimax fair-
ness criterion ensuring no subpopulation is systematically excluded
(𝛽>0).
We note key assumptions: (1) 𝑃popis estimated from corpus D,
treating it as a population sample; (2) the framework assumes static
opinions (see Section 10); (3) query classification (a pre-processor)
handles factual/opinion ambiguity (Section 5). Computationally,
directly optimizing Equation 10 is prohibitive; our architecture
approximates this through opinion-enriched indexing with author
attributes that support fairness analysis, achieving strong coverage
without explicit Wasserstein computation.
This formulation reveals Opinion-Aware RAG as adistribution
estimation and communication problem, fundamentally distinct from
the point-estimation paradigm underlying factual retrieval.
5 Methodology
We propose a general framework for opinion-aware RAG that intro-
duces anopinion enrichment stepbefore indexing. This section
presents the framework abstractly, then instantiates it for our ex-
perimental domain. Figure 1 illustrates the architecture.
5.1 General Framework
The key insight is that traditional RAG optimizes for semantic
similarity, which tends to retrieve documents expressing similarviewpoints. By enriching documents with structured opinion meta-
databeforeindexing, we enable retrieval that accounts for opinion
diversity—operationalizing the coverage objective from Section 4.
The framework consists of three domain-agnostic components:
Entity Registry.A hierarchical taxonomy of opinion targets
organized by granularity. Entities represent the “what” of opinions—
topics, policies, products, or concepts that people hold views about.
A tiered structure (general →specific→granular) enables flexible
aggregation and analysis.
Opinion Attributes.Structured metadata capturing the nature
of expressed opinions. Core attributes include:
•Sentiment & Intensity: Polarity and strength of the opinion
•Stance: Position on a spectrum (e.g., strongly against to
strongly for)
•Impact Assessment: Perceived severity or importance
•Evidence Type: Basis for the opinion (personal experience,
anecdotal, data-driven)
Author Attributes.Contextual factors about opinion holders
that enable fairness analysis. These capture demographic or ex-
periential characteristics relevant to the domain, supporting the
fairness termmax 𝑔∈𝐺Δ𝑔in our objective (Section 4).
Per-Entity Document Splitting.Each source document is split
into multiple enriched documents—one per mentioned entity. This
ensures that when a document discusses multiple topics with dif-
ferent sentiments, each opinion is indexed separately with its cor-
responding metadata. The enriched documents embed opinion and
author attributes alongside content, making this information avail-
able during retrieval and generation.
This architecture directly operationalizes the theoretical frame-
work: per-entity splitting with opinion metadata enables retrieval
that minimizes distributional divergence W2(𝑃pop,ˆ𝑃R), while struc-
tured attributes support analysis of demographic representation.
5.2 Case Study: E-commerce Seller Forums
We instantiate this framework for e-commerce seller discussion
forums, where merchants discuss policies, fees, and operational
challenges. This domain is well-suited for studying opinion-aware
RAG because: (1) sellers hold genuinely diverse views on policies, (2)
author characteristics (business size, tenure) meaningfully influence
perspectives, and (3) the content is naturally opinionated with real
business implications.
Entity Registry.We construct a 98-entity registry organized
across three tiers: Tier 1 covers broad categories (e.g., “Shipping”,
“Fees”), Tier 2 captures intermediate topics (e.g., “Fulfillment Fees”),
and Tier 3 addresses specific aspects (e.g., “Storage Fee Changes”).
The registry is constructed through domain expert curation and
LLM-assisted extraction, seeding Tier 1 from service taxonomy and
using clustering to identify finer-grained entities.
Opinion Attributes.We extract five attributes per entity men-
tion:sentiment(positive, negative, neutral, or mixed),intensity(low,
medium, high),stance(five-level scale from strongly against to
strongly for),business impact(assessed severity), andevidence type
(personal experience, anecdotal, or data-driven).
Author Attributes.We capture three seller characteristics:busi-
ness size(revenue band when disclosed),tenure(experience level
with the service), andregion(geographic location when available).

Opinion-Aware RAG
Raw
DocumentsOpinion
Enrichment
(LLM-based)Enriched
DocumentsVector
StoreLLM
GenerationEntity Registry (Tier 1→2→3)
Opinion Attributes
•Sentiment
•Intensity
•Stance
•Business Impact
•Evidence TypeAuthor Attributes
•Business Size
•Tenure
•Region
Traditional RAG:Raw
Documents→Vector
Store→ LLMOpinion Enrichment Step
Figure 1: Opinion-Aware RAG Architecture. Traditional RAG (bottom, grayed) indexes raw documents directly. Our approach
introduces an opinion enrichment step (highlighted) that extracts per-entity opinion and author attributes before indexing.
Implementation.We use Claude Sonnet 4.5 with a structured
output schema for extraction (see Appendix C.3). Validation against
manually annotated samples yields>90% precision, recall, and F1,
confirming reliable metadata extraction. For our 6-month corpus,
covering 10K+ discussions, avg. 300 words per discussion, enrich-
ment took approximately 4 hours using parallelized LLM calls at a
cost of approx. $200 using AWS Bedrock.1
6 Experiments
We design experiments to evaluate whether opinion enrichment
improves the diversity of perspectives surfaced by RAG systems,
guided by hypotheses derived from our theoretical framework (Sec-
tion 4).
6.1 Hypotheses
Our theoretical framework identifies three components of the opinion-
aware objective (Equation 10): coverage, fidelity, and fairness. We
translate these into testable hypotheses:
H1 (Coverage):Opinion enrichment improves retrieval coverage
of the underlying opinion distribution. W2(𝑃pop,ˆ𝑃R), the coverage
term measures how well retrieved documents represent the full
opinion spectrum. We hypothesize that embedding opinion meta-
data enables retrieval that better spans diverse viewpoints.
H2 (Fidelity):Opinion-enriched retrieval leads to responses that
better communicate opinion diversity to readers.The fidelity term
𝐷KL(𝑃pop∥𝑃reader)captures whether generated responses faithfully
convey the retrieved opinion distribution.
H3 (Fairness):Opinion-enriched retrieval improves demographic
representation in retrieved content.The fairness term max𝑔∈𝐺Δ𝑔
penalizes under-representation of demographic groups. We hy-
pothesize that structured author attributes enable more equitable
retrieval across subpopulations.
1For confidentiality reasons, we do not share the true distribution.6.2 Evaluation Framework
6.2.1 Dataset.We use seller discussion data similar to publicly
accessible e-commerce seller forums such as Amazon Seller Fo-
rums, eBay Community, and Etsy Community. Our dataset spans
six months of discussions, comprising approximately 8,000 unique
sellers and averaging 500 discussions per week. These forums con-
tain rich opinionated content where sellers discuss policies, fees,
operational challenges, and business strategies—making them ideal
for studying opinion diversity in RAG systems.
6.2.2 Knowledge Base Configurations.We construct two knowl-
edge base configurations using identical source content:
Raw Discussions KB (Baseline):Documents are indexed with
their original content only, without any opinion metadata. This
represents a traditional RAG setup where retrieval is based purely
on semantic similarity.
Opinion-Enriched KB (Ours):Documents are enriched with
structured opinion and author attributes as described in Section 5,
then indexed with embedded metadata. Each document is split
per-entity, with opinion attributes included in the indexed content.
Both configurations use Amazon OpenSearch Serverless with Ti-
tan Text Embeddings v2 (1024 dimensions) and fixed-size chunking
(4000 tokens, 15% overlap). We compare against Raw KB rather than
diversity-aware retrieval (e.g., MMR) because our contribution is at
indexing, not retrieval—and MMR optimizes for semantic diversity,
not opinion diversity.
6.2.3 Evaluation Protocol.Our evaluation targets entities with
high opinion diversity—topics where the underlying corpus con-
tains genuinely mixed perspectives. We follow a four-stage proto-
col:
1. Entity Selection:We select entities with genuine opinion
diversity by computing sentiment entropy. For each entity 𝑒, let
𝑝𝑠denote the proportion of opinions with sentiment 𝑠(positive,

Aditya Agrawal, Alwarappan Nakkiran, Darshan Fofadiya, Alex Karlsson, Harsha Aduri
negative, or neutral). We compute:
𝐻(𝑒)=−∑︁
𝑠𝑝𝑠log2(𝑝𝑠)(11)
We additionally require that entities have both positive and negative
sentiments represented. Entities with 𝐻(𝑒)≥ 0.6and at least 5
associated opinions are selected for evaluation.
2. Question Generation:For each selected entity, we generate
natural language questions across six question types using Claude
Sonnet 4.5, with manual review for quality. We select 4 entities and
generate 2 questions per type (48 total):
•Detailed Overview: “What do sellers think about X?”
•Experience: “What are sellers’ experiences with X?”
•Business Impact: “How does X affect sellers’ businesses?”
•Specific Examples: “What examples do sellers give about
X?”
•Process: “What do sellers say about the X process?”
•Emotional: “How do sellers feel about X?”
3. Retrieval and Response Generation:Each question is sub-
mitted to both KB configurations. We retrieve the top- 𝑘documents
using different retrieval algorithms and generate responses using
Claude Sonnet 4.5 with the retrieved context. Ablation studies (Sec-
tion 8) examine the impact of retrieval depth and different retrieval
algorithms.
4. Metrics Computation:Retrieved documents are matched
back to the enriched opinion metadata to compute diversity metrics.
We treat the LLM-extracted opinion metadata as ground truth based
on manual validation showing >90% precision, recall, and F1 on
manually annotated samples.
6.3 Metrics
The theoretical objectives (Equation 10) involve distributional quan-
tities that are difficult to compute directly. We design practical proxy
metrics to operationalize each hypothesis:
Coverage Metrics (H1).We approximate the coverage term
W2(𝑃pop,ˆ𝑃R)using variance-based measures that capture how well
retrieval spans the opinion space:
•Sentiment-Intensity 𝜎: Combines sentiment polarity and
intensity into a single score (e.g., Positive+High = +30, Neg-
ative+High = -30, Neutral = 0). Higher 𝜎indicates broader
coverage of the sentiment-intensity space.
•Business Impact 𝜎: Maps impact levels to numeric values
(High = 30, Medium = 20, Low = 10). Higher 𝜎indicates
retrieval of both high-impact and low-impact perspectives.
•Entity Match Rate: Fraction of retrieved documents that
mention the queried entity, measuring retrieval precision
for opinion-relevant content.
Fidelity Metric (H2).The fidelity term 𝐷KL(𝑃pop∥𝑃reader)mea-
sures whether readers perceive the full opinion spectrum—a quan-
tity we cannot compute directly. We approximate this through
human evaluation: five annotators compared response pairs (KB
identity hidden) on sentiment diversity, providing binary prefer-
ences with ties allowed.
Fairness Metrics (H3).We approximate the fairness term using
categorical coverage (max 𝑔∈𝐺Δ𝑔):•GMS Band Coverage: Fraction of business size categories
represented
•Tenure Coverage: Fraction of experience levels represented
•Evidence Type Coverage: Fraction of evidence types (per-
sonal, anecdotal, data-driven) represented
7 Results
We evaluate our hypotheses using hybrid retrieval with 𝑘= 20
documents and Claude Sonnet 4.5 for retrieval & response genera-
tion. Results are averaged across 48 evaluation questions spanning
4 entities and 6 question types. Statistical significance is assessed
via Wilcoxon signed-rank test ( 𝑛= 48paired questions), a non-
parametric test appropriate for paired samples without normality
assumptions.
7.1 H1: Coverage Improvement
Finding: Supported.Opinion enrichment significantly improves
retrieval coverage of the opinion distribution. Table 2 shows that at
𝑘=20, Opinion-Enriched KB achieves +26.8%, +14.5%, and +42.7%
improvements in sentiment-intensity 𝜎, business impact 𝜎, and
entity match rate respectively, all at p<0.05.
These improvements indicate that embedding opinion metadata
enables retrieval that better spans the sentiment-intensity space,
operationalizing the coverage term W2(𝑃pop,ˆ𝑃R)from our theo-
retical framework. The entity match rate improvement suggests
that per-entity splitting provides better semantic anchoring for
opinion-relevant content.
7.2 H2: Fidelity to Readers
Finding: Supported.Human annotators confirmed that opinion-
enriched retrieval produces responses that better communicate
opinion diversity. Of 48 question-response pairs, annotators pre-
ferred Opinion-Enriched KB responses in 38 cases (79.2% win rate,
95% CI: [65.0%, 89.5%], p<0.001).
This result validates that the automated coverage metrics (H1)
translate to perceptible improvements in response quality—readers
infer a more diverse opinion distribution from enriched responses,
reducing the fidelity gap𝐷 KL(𝑃pop∥𝑃reader).
7.3 H3: Fairness
Finding: Not supported.Contrary to our hypothesis, opinion
enrichment reduces demographic coverage. At 𝑘=20, GMS Band
coverage drops by 13.0% and tenure coverage by 6.9%, both at
p<0.05.
This reveals a trade-off in our current approach: per-entity split-
ting improves opinion coverage but concentrates retrieval on docu-
ments mentioning specific entities, which may come from a nar-
rower author demographic. Evidence type coverage shows slight
improvement (+4.7%), suggesting this trade-off is specific to author
attributes.
Addressing this limitation requires retrieval strategies that explic-
itly optimize the fairness term max𝑔∈𝐺Δ𝑔—an important direction
for future work (Section 10). Section 8 examines how this trade-off
varies with retrieval depth, showing that larger 𝑘partially mitigates
the fairness penalty while preserving coverage gains.

Opinion-Aware RAG
Table 2: Diversity metrics at different retrieval depths ( 𝑘) using hybrid retrieval. Opinion-Enriched KB consistently outperforms
Raw Discussions KB on scored diversity metrics while showing trade-offs on categorical coverage. Statistical significance via
Wilcoxon signed-rank test (𝑛=48): bold = p<0.05, bold* = p<0.001.
Entity Scored Diversity (𝜎) Categorical Coverage (%) Emot.
𝑘KB ConfigMatch % Sent-Int Biz Impact GMS Band Tenure Evidence Marker
10Raw Discussions 32.7 10.10 11.48 63.7 65.7 38.0 5.5
Opinion-Enriched 48.5 13.51 12.73 51.8 59.2 40.1 7.0
Δ+48.3%* +33.8%* +10.9% -18.7%* -9.9%+5.5%+27.3%*
20Raw Discussions 24.8 9.12 10.55 78.3 81.2 44.3 7.2
Opinion-Enriched 35.4 11.56 12.08 68.1 75.6 46.4 8.9
Δ+42.7%* +26.8%* +14.5% -13.0%* -6.9%+4.7%+23.6%*
40Raw Discussions 16.9 8.11 9.45 87.8 89.3 49.5 9.1
Opinion-Enriched 24.6 9.68 10.88 83.6 85.1 53.1 11.0
Δ+45.6%* +19.4%* +15.1%* -4.8% -4.7%*+7.3%+20.9%*
60Raw Discussions 13.3 7.28 8.52 90.5 89.6 50.5 10.1
Opinion-Enriched 19.0 8.52 9.78 89.0 87.5 53.6 12.2
Δ+42.9%* +17.0%* +14.8%*-1.7%-2.3%+6.1%+20.8%*
7.4 Summary
H1 (coverage) and H2 (fidelity) are supported, validating that opin-
ion enrichment improves both retrieval diversity and response qual-
ity. H3 (fairness) is not supported—demographic coverage decreases,
revealing a limitation of our current approach that motivates future
work on multi-objective retrieval. Appendix D provides qualita-
tive examples showing how enriched responses capture sentiment
distributions and constructive suggestions missing from baseline
responses.
8 Ablation Studies
We conduct ablation studies to understand how different design
choices affect system performance. Unless otherwise noted, experi-
ments use hybrid search with 𝑘=20retrieved documents. Retrieval
diversity performance is consistent across different state of art LLMs
for opinion extraction (Appendix B). We focus on Claude Sonnet
4.5 performance for this study.
8.1 Retrieval Depth Analysis
Can the Raw Discussions KB achieve comparable diversity by sim-
ply retrieving more documents? Table 2 compares metrics across
retrieval depths.
Notably, Raw Discussions KB diversitydecreaseswith more re-
trieval: sentiment diversity drops from 10.10 at 𝑘= 10to 9.12
at𝑘=20. This suggests semantic similarity-based retrieval con-
verges toward opinion-homogeneous content as more documents
are retrieved. In contrast, Opinion-Enriched KB maintains strong
diversity at both depths, demonstrating that opinion metadata helps
surface varied perspectives regardless of retrieval depth.
8.2 Impact of Question Type
Table 3 shows how question formulation affects diversity.Table 3: Sentiment-Intensity 𝜎by question type ( 𝑘= 20).
Largest gains for Business Impact and Emotional queries.
Question Type Raw Discussions EnrichedΔ
Detailed Overview 9.14 11.80+29.1%
Experience 10.64 11.30+6.2%
Business Impact 8.56 11.89+38.9%
Specific Examples 9.02 11.14+23.5%
Process 7.82 10.44+33.5%
Emotional 9.56 12.82+34.1%
Experience questions show the smallest improvement (+6.2%),
likely because they already surface diverse personal perspectives
without opinion metadata.
8.3 Entity-Matched Document Analysis
Section 7 showed that opinion enrichment reduces demographic
coverage overall. To understand why, we analyze only the subset
of retrieved documents that actually mention the queried entity
(Table 4).
Table 4: Diversity metrics on entity-matched documents only.
Metric Raw Enr.Δ
Matched Docs (of 960) 238 340+42.9%*
Sent-Int𝜎14.29 16.57 +16.0%
Biz Impact𝜎4.45 5.47+22.9%
GMS Band Cov. (%) 42.9 47.0+9.6%
Tenure Cov. (%) 40.5 53.3+31.6%*
Evidence Cov. (%) 44.3 46.4 +4.7%
Emot. Markers 7.2 8.9+23.6%*

Aditya Agrawal, Alwarappan Nakkiran, Darshan Fofadiya, Alex Karlsson, Harsha Aduri
On entity-matched documents, Opinion-Enriched KB improves
demographic coverage (GMS Band +9.6%, Tenure +31.6%)—the op-
posite of the overall result. This suggests the fairness penalty stems
from non-matched documents: opinion enrichment retrieves more
on-topic content from diverse authors, but also surfaces off-topic
documents from a narrower demographic that dilutes overall cov-
erage.
8.4 Retrieval Method Variations
Table 5 compares semantic-only and hybrid retrieval.
Table 5: Semantic-only vs. hybrid retrieval. Hybrid search
shows larger diversity gains.
Semantic Hybrid
MetricRaw Enr. (Δ) Raw Enr. (Δ)
Sent-Int𝜎8.93 10.84 (+21.4%*) 9.12 11.56 (+26.8%*)
Biz Impact𝜎10.47 11.63 (+11.1%) 10.55 12.08 (+14.5%)
GMS Band Cov. (%) 79.1 68.7 (-13.1%*) 78.3 68.1 (-13.0%*)
Tenure Cov. (%) 81.5 76.8 (-5.8%) 81.2 75.6 (-6.9%)
Evidence Cov. (%) 44.3 43.8 (-1.1%) 44.3 46.4 (+4.7%)
Emot. Markers 7.1 8.4 (+18.3%) 7.2 8.9 (+23.6%*)
Hybrid search with opinion enrichment achieves the best results
(+26.8% vs +21.4% for sentiment-intensity), suggesting keyword-
based retrieval complements semantic similarity by surfacing doc-
uments with matching entity mentions.
8.5 Summary of Ablations
A1: Can Raw KB match diversity by retrieving more?No. Raw
KB diversitydecreaseswith larger 𝑘(sentiment𝜎: 10.10→9.12 from
𝑘=10 to 20), while Opinion-Enriched KB maintains gains across all
depths.
A2: Which question types benefit most?Business Impact
(+38.9%) and Emotional (+34.1%) queries show largest gains; Expe-
rience queries (+6.2%) benefit least, likely already surfacing diverse
perspectives.
A3: Does retrieval method matter?Yes. Hybrid search am-
plifies opinion enrichment benefits (+26.8%) over semantic-only
(+21.4%), as keyword matching surfaces entity-relevant documents.
9 Challenges and Call to Action
Technical Challenges.Current RAG benchmarks lack datasets
with naturally occurring subjective content and rich annotations
for stance, sentiment, and perspective diversity. Opinion-aware
architectures must balance faithfulness to diverse sources with
coherent generation, avoiding bias toward dominant opinions while
maintaining efficiency.
Ethical Considerations.Without opinion-aware capabilities,
systems risk algorithmic opinion manipulation through training
biases, echo chamber amplification by retrieving only confirm-
ing information, and minority voice erasure by underrepresenting
marginalized viewpoints. The goal is not eliminating bias but mak-
ing it transparent and auditable.Community Call to Action.We call on the research commu-
nity to pursue: (1) benchmark development with subjective annota-
tions and opinion synthesis metrics; (2) foundation model research
on opinion-aware tasks; (3) retrieval algorithms for diverse per-
spective identification; and (4) evaluation protocols for subjective
content synthesis.
10 Future Work
Our experiments validate key aspects of the theoretical frame-
work in Section 4: opinion enrichment improves retrieval coverage
(W2reduction through increased sentiment diversity) and entity-
relevant retrieval (better alignment with 𝑃pop). However, fully re-
alizing opinion-aware RAG requires addressing several open chal-
lenges.
Representation Learning for Subjective Content.Our opin-
ion extraction relies on LLM-based annotation, but the theoretical
framework’s opinion embedding function 𝜙:D→R𝑚suggests
the need for learned representations that jointly encode seman-
tic content with subjective attributes. Future work should explore
multi-task learning objectives and hierarchical embedding spaces
where documents can be similar at the topic level while differing
at the stance level.
Optimizing the Coverage Term.While our diversity metrics
approximate the coverage objective W2(𝑃pop,ˆ𝑃R), we do not di-
rectly optimize this during retrieval. Query expansion techniques
generating stance-diverse queries, and re-ranking mechanisms us-
ing submodular optimization, could more directly minimize distri-
butional divergence.
Fairness Constraints.The framework’s fairness term max𝑔∈𝐺Δ𝑔
addresses demographic equity, but our ablations reveal trade-offs be-
tween opinion diversity and author demographic coverage. Future
work must develop retrieval strategies that jointly optimize for both,
ensuring no subpopulation is systematically under-represented.
Temporal Dynamics.The framework assumes a static 𝑃pop, but
opinions evolve continuously. Extending the theoretical foundation
to handle temporal opinion dynamics—distinguishing temporary
reactive opinions from lasting attitudinal changes—remains an im-
portant direction.
11 Conclusion
We presented Opinion-Aware RAG, an architecture that treats sub-
jectivity as a first-class citizen rather than noise to be filtered. Our
novel theoretical framework distinguishes factual queries (mini-
mize posterior entropy) from opinion queries (preserve entropy
reflecting genuine perspective heterogeneity), formalizing opinion-
aware retrieval through coverage, fidelity, and fairness objectives.
Experiments on e-commerce seller forum data demonstrate that
opinion enrichment yields measurably more representative retrieval:
+26.8% sentiment diversity, +42.7% entity match rate, and +4.7%
evidence type coverage. Human evaluation confirms these gains
translate to perceptible quality improvements, with annotators pre-
ferring Opinion-Enriched responses 79.2% of the time (p <0.001).
Ablations reveal that simply retrieving more documents does not
achieve comparable diversity—opinion metadata is essential for
surfacing varied perspectives.

Opinion-Aware RAG
Our work represents a first step. The trade-off between opinion
and author demographic diversity motivates future research on
multi-objective retrieval. Fully realizing the theoretical framework
requires learned opinion embeddings, direct optimization of distri-
butional coverage, and evaluation metrics for generation fidelity.
As AI systems increasingly mediate access to information, their
ability to fairly represent diverse perspectives becomes essential
for transparent and accountable AI.
Acknowledgments
Acknowledgments will be added after the review process.
References
[1]Patrick Lewis, Ethan Perez, Aleksandra Piktus, Fabio Petroni, Vladimir Karpukhin,
Naman Goyal, Heinrich Küttler, Mike Lewis, Wen-tau Yih, Tim Rocktäschel, et al.
Retrieval-augmented generation for knowledge-intensive nlp tasks. InAdvances
in Neural Information Processing Systems, volume 33, pages 9459–9474, 2020.
[2]Yunfan Gao, Yun Xiong, Xinyu Gao, Kanghao Jia, Jinfeng Pan, Yuxi Bi, Yi Dai,
Jiawei Sun, and Haofen Wang. Retrieval-augmented generation for large language
models: A survey.arXiv preprint arXiv:2312.10997, 2023.
[3]Fabio Petroni, Tim Rocktäschel, Patrick Lewis, Anton Bakhtin, Yuxiang Wu,
Alexander H Miller, and Sebastian Riedel. Language models as knowledge bases?
InProceedings of the 2019 Conference on Empirical Methods in Natural Language
Processing, pages 2463–2473, 2019.
[4]Adam Roberts, Colin Raffel, and Noam Shazeer. How much knowledge can
you pack into the parameters of a language model? InProceedings of the 2020
Conference on Empirical Methods in Natural Language Processing, pages 5418–5426,
2020.
[5]Aoran Gan, Hao Yu, Kai Zhang, Qi Liu, Wenyu Yan, Zhenya Huang, Shiwei Tong,
and Guoping Hu. Retrieval augmented generation evaluation in the era of large
language models: A comprehensive survey.arXiv preprint arXiv:2504.14891, 2024.
[6]Penghao Zhao, Hailin Zhang, Qinhan Yu, Zhengren Wang, Yunteng Geng,
Fangcheng Fu, Ling Yang, Wentao Zhang, and Bin Cui. Retrieval-augmented
generation for ai-generated content: A survey.arXiv preprint arXiv:2402.19473,
2024.
[7]Yucheng Wang, Xiaohan Li, Yongbin Gao, Jiawei Chen, and Zhiyuan Liu. A
systematic literature review of rag: Techniques, metrics, and challenges.arXiv
preprint arXiv:2501.13958, 2025.
[8]Wenqi Fan, Yujuan Ding, Liangbo Ning, Shijie Wang, Hengyun Li, Dawei Yin,
Tat-Seng Chua, and Qing Li. A survey on rag meeting llms: Towards retrieval-
augmented large language models.Proceedings of the 30th ACM SIGKDD Confer-
ence on Knowledge Discovery and Data Mining, pages 6491–6501, 2024.
[9]Md Rizwan Parvez Rony, Thiago Ferreira, Mohammed Abuhamad, and Preslav
Nakov. A benchmark for social-web indirect prompt injection in rag.arXiv
preprint arXiv:2601.10923, 2025.
[10] Yifan Chen, Wei Zhang, and Xiaoming Liu. From unstructured communication
to intelligent rag: Multi-agent automation for supply chain knowledge bases.
Proceedings of the ICML 2025 Workshop on Foundation Models in the Wild, 2025.
[11] Danilo Bzdok, Leonhard Schilbach, Kai Vogeley, Karoline Schneider, Angela R
Laird, Robert Langner, and Simon B Eickhoff. Parsing the neural correlates of
moral cognition: Ale meta-analysis on morality, theory of mind, and empathy.
Brain Structure and Function, 217(4):783–796, 2013.
[12] Dan Sperber, Fabrice Clément, Christophe Heintz, Olivier Mascaro, Hugo Mercier,
Gloria Origgi, and Deirdre Wilson. Epistemic vigilance.Mind & Language,
25(4):359–393, 2010.
[13] Hung-Yu Chen, Ramakanth Pasunuru, Jason Weston, and Mohit Bansal. Rich
knowledge sources bring complex knowledge conflicts: Recalibrating models to
reflect conflicting evidence. InProceedings of the 2022 Conference on Empirical
Methods in Natural Language Processing, pages 2292–2307, 2022.
[14] Peitian Zhang, Shitao Gu, Yongkang Cao, Yangyifei Xu, Kun Zhang, Zheng Zhao,
Huawei Qin, Zhiyuan Liu, Maosong Sun, and Chenyan Xiong. Retrieve anything
to augment large language models.arXiv preprint arXiv:2310.07554, 2023.
[15] Tom Kwiatkowski, Jennimaria Palomaki, Olivia Redfield, Michael Collins, Ankur
Parikh, Chris Alberti, Danielle Epstein, Illia Polosukhin, Jacob Devlin, Kenton
Lee, et al. Natural questions: a benchmark for question answering research. In
Transactions of the Association for Computational Linguistics, volume 7, pages
453–466, 2019.
[16] Tri Nguyen, Mir Rosenberg, Xia Song, Jianfeng Gao, Saurabh Tiwary, Rangan
Majumder, and Li Deng. Ms marco: A human generated machine reading com-
prehension dataset. InProceedings of the Workshop on Cognitive Computation:
Integrating Neural and Symbolic Approaches, 2016.
[17] Zhilin Yang, Peng Qi, Saizheng Zhang, Yoshua Bengio, William W Cohen, Ruslan
Salakhutdinov, and Christopher D Manning. Hotpotqa: A dataset for diverse,explainable multi-hop question answering. InProceedings of the 2018 Conference
on Empirical Methods in Natural Language Processing, pages 2369–2380, 2018.
[18] Mir Tafseer Nayeem and Davood Rafiei. Opiniorag: Towards generating user-
centric opinion highlights from large-scale online reviews.arXiv preprint
arXiv:2509.00285, 2025.
[19] Mrinank Sharma, Meg Tong, Tomasz Korbak, David Duvenaud, Amanda
Askell, Samuel R Bowman, Newton Cheng, Esin Durmus, Zac Hatfield-Dodds,
Scott R Johnston, Shauna Kravec, Timothy Maxwell, Sam McCandlish, Kamal
Ndousse, Oliver Rausch, Nicholas Schiefer, Da Yan, Miranda Zhang, and Ethan
Perez. Towards understanding sycophancy in language models.arXiv preprint
arXiv:2310.13548, 2023.
[20] Linlin Wang, Tianqing Zhu, Laiqiao Qin, Longxiang Gao, and Wanlei Zhou. Bias
amplification in rag: Poisoning knowledge retrieval to steer llms.arXiv preprint
arXiv:2506.11415, 2025.
[21] Tianhui Zhang, Yi Zhou, and Danushka Bollegala. Evaluating the effect of retrieval
augmentation on social biases.arXiv preprint arXiv:2502.17611, 2025.
[22] Matheus Vinicius da Silva de Oliveira, Jonathan de Andrade Silva, and Aw-
dren de Lima Fontao. Fairness testing in retrieval-augmented generation:
How small perturbations reveal bias in small language models.arXiv preprint
arXiv:2509.26584, 2025.
[23] Linda Zeng, Rithwik Gupta, Divij Motwani, Diji Yang, and Yi Zhang. Worse than
zero-shot? a fact-checking dataset for evaluating the robustness of rag against
misleading retrievals.arXiv preprint arXiv:2502.16101, 2025.
[24] Arie Cattan, Alon Jacovi, Ori Ram, Jonathan Herzig, Roee Aharoni, Sasha Gold-
shtein, Eran Ofek, Idan Szpektor, and Avi Caciularu. Dragged into conflicts:
Detecting and addressing conflicting sources in search-augmented llms.arXiv
preprint arXiv:2506.08500, 2025.
[25] Eunseong Choi, June Park, Hyeri Lee, and Jongwuk Lee. Conflict-aware soft
prompting for retrieval-augmented generation.arXiv preprint arXiv:2508.15253,
2025.
[26] Qinggang Zhang, Zhishang Xiang, Yilin Xiao, Le Wang, Junhui Li, Xinrun Wang,
and Jinsong Su. Faithfulrag: Fact-level conflict modeling for context-faithful
retrieval-augmented generation.arXiv preprint arXiv:2506.08938, 2025.
[27] Amin Bigdeli, Negar Arabzadeh, Ebrahim Bagheri, and Charles LA Clarke. Ad-
versarial attacks against neural ranking models via in-context learning.arXiv
preprint arXiv:2508.15283, 2025.
[28] Zhenrui Yue, Huimin Zeng, Lanyu Shang, Yifan Liu, Yang Zhang, and Dong Wang.
Retrieval augmented fact verification by synthesizing contrastive arguments.
arXiv preprint arXiv:2406.09815, 2024.
[29] Hao Li, Viktor Schlegel, Yizheng Sun, Riza Batista-Navarro, and Goran Ne-
nadic. Large language models in argument mining: A survey.arXiv preprint
arXiv:2506.16383, 2025.
[30] Jungyeon Lee, Kangmin Lee, and Taeuk Kim. Magic: A multi-hop and graph-based
benchmark for inter-context conflicts in retrieval-augmented generation.arXiv
preprint arXiv:2507.21544, 2025.
[31] Vignesh Gokul, Srikanth Tenneti, and Alwarappan Nakkiran. Contradiction
detection in rag systems: Evaluating llms as context validators for improved
information consistency.arXiv preprint arXiv:2504.00180, 2025.
[32] Abeer AlDayel and Walid Magdy. Stance detection on social media: State of the
art and trends.Information Processing & Management, 58(4):102597, 2021.
[33] Rabab Alkhalifa and Arkaitz Zubiaga. Capturing stance dynamics in social media:
Open challenges and research directions.arXiv preprint arXiv:2109.00475, 2021.
[34] Dimitrios Tsirmpas, Ioannis Gkionis, Georgios Th Papadopoulos, and Ioannis
Mademlis. Neural natural language processing for long texts: A survey on clas-
sification and summarization.Engineering Applications of Artificial Intelligence,
133:108231, 2024.
[35] Marek Šuppa, Daniel Skala, Daniela Jaššová, Samuel Sučík, Andrej Švec, and Peter
Hraška. Bryndza at climateactivism 2024: Stance, target and hate event detection
via retrieval-augmented gpt-4 and llama.arXiv preprint arXiv:2402.06549, 2024.
[36] Lata Pangtey, Omkar Kabde, Shahid Shafi Dar, and Nagendra Kumar. Two stage
context learning with large language models for multimodal stance detection on
climate change.arXiv preprint arXiv:2509.08024, 2025.
[37] Bohan Zhang, Daoan Ding, and Liyao Jiao. A survey on stance detection for mis-
and disinformation identification.arXiv preprint arXiv:2408.16906, 2024.
[38] Gaurab Chhetri, Anandi Dutta, and Subasish Das. Cognitivesky: Scalable sen-
timent and narrative analysis for decentralized social media.arXiv preprint
arXiv:2509.11444, 2025.
[39] Gaurab Chhetri, Darrell Anderson, Boniphace Kutela, and Subasish Das. A
transformer-based cross-platform analysis of public discourse on the 15-minute
city paradigm.arXiv preprint arXiv:2509.11443, 2025.
[40] Chhavi Sharma, Deepesh Bhageria, William Scott, Srinivas PYKL, Amitava Das,
Tanmoy Chakraborty, Viswanath Pulabaigari, and Bjorn Gamback. Semeval-2020
task 8: Memotion analysis – the visuo-lingual metaphor! 2020.
[41] Tao Li, Lei Lin, Minsoo Choi, Kaiming Fu, Siyuan Gong, and Jian Wang. Youtube av
50k: An annotated corpus for comments in autonomous vehicles. InProceedings of
the Thirteenth International Joint Symposium on Artificial Intelligence and Natural
Language Processing, 2018.

Aditya Agrawal, Alwarappan Nakkiran, Darshan Fofadiya, Alex Karlsson, Harsha Aduri
[42] Sarah Weißmann, Aaron Philipp, Roland Verwiebe, Chiara Osorio Krauter, Nina-
Sophie Fritsch, and Claudia Buder. Clicks, comments, consequences: Are content
creators’ socio-structural and platform characteristics shaping the exposure to
negative sentiment, offensive language, and hate speech on youtube?arXiv
preprint arXiv:2504.07676, 2025.
[43] Yifan Sun, Hirofumi Tsuruta, Masaya Kumagai, and Ken Kurosaki. Topic modeling
and sentiment analysis on japanese online media’s coverage of nuclear energy.
arXiv preprint arXiv:2411.18383, 2024.
[44] Ashwin Ram, Yigit Ege Bayiz, Arash Amini, Mustafa Munir, and Radu Marculescu.
Credirag: Network-augmented credibility-based retrieval for misinformation
detection in reddit.arXiv preprint arXiv:2410.12061, 2024.
[45] Jonathan Berant, Andrew Crivat, and Percy Liang. Semantic parsing on freebase
from question-answer pairs. InProceedings of the 2013 Conference on Empirical
Methods in Natural Language Processing, pages 1533–1544, 2013.
[46] Mandar Joshi, Eunsol Choi, Daniel S Weld, and Luke Zettlemoyer. Triviaqa: A
large scale distantly supervised challenge dataset for reading comprehension.
InProceedings of the 55th Annual Meeting of the Association for Computational
Linguistics, volume 1, pages 1601–1611, 2017.
[47] Pranav Rajpurkar, Jian Zhang, Konstantin Lopyrev, and Percy Liang. Squad:
100,000+ questions for machine reading comprehension.Proceedings of the 2016
Conference on Empirical Methods in Natural Language Processing, pages 2383–2392,
2016.
[48] Alex Mallen, Akari Asai, Victor Zhong, Rajarshi Das, Daniel Khashabi, and
Hannaneh Hajishirzi. Popqa: A dataset for population-based question answering.
Proceedings of the 2023 Conference on Empirical Methods in Natural Language
Processing, pages 14914–14924, 2023.
[49] Johannes Welbl, Nelson F. Liu, and Matt Gardner. Constructing a multi-hop qa
dataset for comprehensive evaluation of reasoning steps.Proceedings of the 2018
Conference on Empirical Methods in Natural Language Processing, pages 3219–3229,
2018.
[50] Harsh Trivedi, Niranjan Balasubramanian, Tushar Khot, and Ashish Sabharwal.
Musique: Multihop questions via single-hop question composition.Transactions
of the Association for Computational Linguistics, 10:539–554, 2022.
[51] Tomas Kocisky, Jonathan Schwarz, Phil Blunsom, Chris Dyer, Karl Moritz Her-
mann, Gabor Melis, and Edward Grefenstette. The narrativeqa reading compre-
hension challenge.Transactions of the Association for Computational Linguistics,
6:317–328, 2018.
[52] Angela Fan, Yacine Jernite, Ethan Perez, David Grangier, Jason Weston, and
Michael Auli. Eli5: Long form question answering.Proceedings of the 57th Annual
Meeting of the Association for Computational Linguistics, pages 3558–3567, 2019.
[53] Ming Zhong, Da Yin, Tao Yu, Ahmad Zaidi, Mutethia Mutuma, Rahul Jha,
Ahmed Hassan Awadallah, Asli Celikyilmaz, Yang Liu, Xipeng Qiu, and Dragomir
Radev. Qmsum: A new benchmark for query-based multi-domain meeting sum-
marization.Proceedings of the 2021 Conference of the North American Chapter of
the Association for Computational Linguistics, pages 5905–5921, 2021.
[54] Ivan Stelmakh, Yi Luan, Bhuwan Dhingra, and Ming-Wei Chang. Asqa: A dataset
for ambiguous question answering with long-form answers.Proceedings of the
2022 Conference on Empirical Methods in Natural Language Processing, pages
3866–3885, 2022.
[55] Timo Möller, Anthony Reina, Raghavan Jayakumar, and Malte Pietsch. Covid-
qa: A question answering dataset for covid-19.ACL 2020 Workshop on Natural
Language Processing for COVID-19, 2020.
[56] Pradeep Dasigi, Kyle Lo, Iz Beltagy, Arman Cohan, Noah A. Smith, and Matt
Gardner. A dataset of information-seeking questions and answers anchored in
research papers.Proceedings of the 2021 Conference of the North American Chapter
of the Association for Computational Linguistics, pages 4599–4610, 2021.
[57] Xidong Wang, Guiming H. Chen, Dingjie Song, Zhiyi Zhang, Zhihong Chen,
Qingying Xiao, Feng Jiang, Jianquan Li, Xiang Wan, Benyou Wang, and Haizhou
Li. Cmb: A comprehensive medical benchmark in chinese.arXiv preprint
arXiv:2308.08833, 2023.
[58] Peter Clark, Isaac Cowhey, Oren Etzioni, Tushar Khot, Ashish Sabharwal, Carissa
Schoenick, and Oyvind Tafjord. Think you have solved question answering? try
arc, the ai2 reasoning challenge.arXiv preprint arXiv:1803.05457, 2018.
[59] Alon Talmor, Jonathan Herzig, Nicholas Lourie, and Jonathan Berant. Common-
senseqa: A question answering challenge targeting commonsense knowledge.
Proceedings of the 2019 Conference of the North American Chapter of the Association
for Computational Linguistics, pages 4149–4158, 2019.
[60] Dan Hendrycks, Collin Burns, Steven Basart, Andy Zou, Mantas Mazeika, Dawn
Song, and Jacob Steinhardt. Measuring massive multitask language understanding.
Proceedings of the International Conference on Learning Representations, 2021.
[61] Richard Yuanzhe Pang, Alicia Parrish, Nitish Joshi, Nikita Nangia, Jason Phang,
Angelica Chen, Vishakh Padmakumar, Johnny Ma, Jana Thompson, He He, et al.
Quality: Question answering with long input texts, yes!Proceedings of the 2022
Conference of the North American Chapter of the Association for Computational
Linguistics, pages 5336–5358, 2022.
[62] Fabio Petroni, Aleksandra Piktus, Angela Fan, Patrick Lewis, Stas Borzunov,
Edouard Grave, et al. Kilt: a benchmark for knowledge intensive language tasks.Proceedings of the 2021 Conference of the North American Chapter of the Association
for Computational Linguistics, pages 2523–2544, 2021.
[63] Nandan Thakur, Nils Reimers, Andreas Rücklé, Abhishek Srivastava, and Iryna
Gupta. Beir: A heterogeneous benchmark for zero-shot evaluation of information
retrieval models.Proceedings of the Neural Information Processing Systems Track
on Datasets and Benchmarks, 1, 2021.
[64] James Thorne, Andreas Vlachos, Christos Christodoulopoulos, and Arpit Mittal.
Fever: a large-scale dataset for fact extraction and verification.Proceedings of the
2018 Conference of the North American Chapter of the Association for Computational
Linguistics, 1:809–819, 2018.
[65] Neema Kotonya and Francesca Toni. Explainable automated fact-checking for
public health claims.Proceedings of the 2020 Conference on Empirical Methods in
Natural Language Processing, pages 7740–7754, 2020.
[66] Stephanie Lin, Jacob Hilton, and Owain Evans. Truthfulqa: Measuring how models
mimic human falsehoods.Proceedings of the 60th Annual Meeting of the Association
for Computational Linguistics, pages 3214–3252, 2022.
[67] Nandan Thakur, Luiz Bonifacio, Xinyu Zhang, Odunayo Ogundepo, Ehsan Ka-
malloo, David Alfonso-Hermelo, Xiaoguang Li, Qian Liu, Boxing Chen, Mehdi
Rezagholizadeh, et al. Nomiracl: Knowing when you don’t know for robust mul-
tilingual retrieval-augmented generation.arXiv preprint arXiv:2312.11361, 2023.
[68] Xiao Yang, Kai Sun, Hao Xin, Yushi Sun, Nikita Bhalla, Xiangsen Chen, Sajal
Choudhary, Rongze Jiang, Hanwen Cai, Michael Deng, et al. Crag – compre-
hensive rag benchmark. InAdvances in Neural Information Processing Systems,
2024.
[69] Tianyu Gao, Howard Yen, Jiatong Yu, and Danqi Chen. Enabling large language
models to generate text with citations.Proceedings of the 2023 Conference on
Empirical Methods in Natural Language Processing, pages 6465–681, 2023.
[70] Sewon Min, Kalpesh Krishna, Xinxi Lyu, Mike Lewis, Wen-tau Yih, Pang Wei
Koh, et al. Factscore: Fine-grained atomic evaluation of factual precision in long
form text generation.Proceedings of the 2023 Conference on Empirical Methods in
Natural Language Processing, pages 12076–12100, 2023.
[71] Zhenrui Yue, Patrick Schramowski, Antonios Anastasopoulos, and Kristian Kerst-
ing. Attrscore: Reference-free evaluation of generated text as attribution.Proceed-
ings of the 61st Annual Meeting of the Association for Computational Linguistics,
pages 5956–5969, 2023.
[72] Robert Friel, Tho Tran, Xiaoyu Zhao, Yifei Cheng, Zheyuan Jiang, and Anh Totti
Vu. Ragbench: Explainable benchmark for retrieval-augmented generation sys-
tems.arXiv preprint arXiv:2407.11005, 2024.

Opinion-Aware RAG
A Extended Benchmark Analysis
Table 6 provides a comprehensive analysis of major evaluation
datasets used in RAG research from 2019-2025, demonstrating the
dominance of factual evaluation paradigms.
B Opinion Extraction Model Variations
Table 7 compares retrieval diversity when using different LLMs for
opinion extraction. All configurations use hybrid search at 𝑘=20.
Results show consistent improvements across models, demonstrat-
ing the approach is not dependent on a specific extraction model.
C Extended Response Examples
C.1 Emergent Opinion Synthesis Example
The following is a complete example response from a contemporary
LLM (e.g., GPT-4.5, Claude Sonnet 4.5) to the query:“What do small
business owners think about the current tax policy?”
Small business owners express mixed views on cur-
rent tax policy. Many appreciate efforts toward per-
manence and certainty, particularly wanting key pro-
visions from the 2017 Tax Cuts and Jobs Act made
permanent to aid long-term planning. They support
targeted deductions and pass-through relief, espe-
cially proposals allowing deduction of the first $25,000
of qualifying business income. However, concerns
persist: many feel previous tax cuts primarily bene-
fited larger corporations rather than small businesses.
Common complaints include tax code complexity,
high compliance costs, and perceived unfairness in
how large firms can leverage specialists and loop-
holes. Small business owners consistently advocate
for simplified reporting, scaled incentives that actu-
ally help smaller firms, and measures ensuring “fair
play” against large corporation advantages.
C.2 Biased Leadership Response Example
The following illustrates how traditional RAG metrics can assign
high scores to biased outputs. In response to the query“What
leadership qualities are most important for startup CEOs?”, a system
might generate:
Successful startup CEOs demonstrate strong analyti-
cal thinking, decisive action under pressure, and nat-
ural assertiveness in competitive markets. Research
shows that effective leaders maintain composure dur-
ing high-stakes negotiations, exhibit confidence in
pitching to investors, and demonstrate the aggressive
persistence needed to disrupt established industries.
The most successful CEOs are typically those who
embrace calculated risks, maintain unwavering con-
viction in their vision, and possess the competitive
drive necessary to outperform rivals.
While scoring excellently across standard metrics, this response
amplifies gender bias by emphasizing traditionally masculine-coded
traits while omitting collaborative, empathetic, or inclusive leader-
ship qualities.C.3 Opinion Extraction Prompt
We use Claude Sonnet 4.5 with a structured output schema for
opinion extraction. The prompt template is shown below (with
service-specific terms anonymized):
You are an expert at analyzing seller opinions
from e-commerce service content. Analyze the
following post and extract structured opinion
information for EACH entity (business topic/issue)
mentioned.
POST CONTENT: {content}
SERVICE: {service_type}
CATEGORY: {service_category}
{entity_registry}
Extract the following information and respond
ONLY with valid JSON:
{"entities": [{
"entity_name": "entity name",
"confidence": "Low|Medium|High",
"sentiment": "positive|negative|neutral|mixed",
"sentiment_intensity": "Low|Medium|High",
"emotional_markers": ["marker1", "marker2"],
"evidence": "supporting evidence",
"business_impact": "Low|Medium|High",
"context_snippet": "relevant quote",
"evidence_type": "personal|anecdotal|data"
}]}
Key extraction guidelines include:
•Entity matching: If the entity matches one from the hier-
archical registry, use that exact name; prefer granular (Tier
3) over general (Tier 1) entities
•Per-entity extraction: Each entity receives its own com-
plete opinion structure; a single post may yield multiple
entities with different sentiments
•Evidence typing: Classify as personal (first-hand experi-
ence), anecdotal (heard from others), or data (cites specific
statistics)
•Emotional markers: Common markers include Frustra-
tion, Anger, Resignation, Adaptation, Hope, Satisfaction,
Confusion, Anxiety
•Business impact: Consider financial mentions, operational
disruption, and account risks when assigning impact levels
The model outputs a JSON array with one object per entity-
opinion pair, enabling downstream per-entity document splitting.
D Qualitative Response Comparison
The following comparison illustrates how opinion enrichment af-
fects generated responses. Key phrases are highlighted inred.
Query:“Give me a detailed overview of what sellers think about
community management & moderators specifically on seller forums?”
D.1 Analysis of Key Differences
1. Opening Framing:The Raw KB opens with neutral organiza-
tion (“there are several perspectives”), treating the response as a
categorization exercise. The Enriched KB immediately conveys the
sentiment distribution (“mixed but generally positive”), answering
the actual question about what sellersthink.

Aditya Agrawal, Alwarappan Nakkiran, Darshan Fofadiya, Alex Karlsson, Harsha Aduri
Dataset Year Task Category Evaluation Focus Opinion Reference
Single-hop Question Answering
WebQuestions 2013 Knowledge Base QA EM, F1 No [45]
MS MARCO 2016 Passage Retrieval MRR, Recall@K No [16]
TriviaQA 2017 Reading Comprehension EM, F1 No [46]
SQuAD 2.0 2018 Reading Comprehension EM, F1 No [47]
Natural Questions 2019 Open-domain QA EM, Token F1 No [15]
PopQA 2023 Entity-centric QA EM, F1 No [48]
Multi-hop Question Answering
HotpotQA 2018 Multi-hop Reasoning EM, F1, Support F1 No [17]
2WikiMultiHopQA 2020 Multi-hop Reasoning EM, F1 No [49]
MuSiQue 2022 Multi-hop Reasoning EM, F1 No [50]
Long-form Question Answering
NarrativeQA 2018 Long-form QA ROUGE, BLEU No [51]
ELI5 2019 Long-form QA ROUGE-L No [52]
QMSum 2021 Meeting Summarization ROUGE No [53]
ASQA 2022 Ambiguous QA ROUGE, QA-based No [54]
Domain-specific Question Answering
COVID-QA 2020 Medical Domain EM, F1 No [55]
Qasper 2021 Scientific Papers EM, F1 No [56]
CMB 2023 Chinese Medical Accuracy No [57]
Multi-choice Question Answering
ARC 2018 Science Questions Accuracy No [58]
CommonsenseQA 2019 Commonsense Reasoning Accuracy No [59]
MMLU 2020 World Knowledge Accuracy No [60]
QuALITY 2021 Reading Comprehension Accuracy No [61]
Information Retrieval & Knowledge Tasks
KILT 2020 Knowledge-Intensive Tasks EM, F1, ROUGE-L No [62]
BEIR 2021 Information Retrieval nDCG, MAP, Recall No [63]
Fact Verification & Robustness
FEVER 2018 Fact Verification Accuracy, FEVER Score No [64]
PubHealth 2020 Health Fact-checking Accuracy No [65]
TruthfulQA 2021 Truthfulness Evaluation % True, % Informative Partial [66]
NoMIRACL 2023 Multilingual Robustness Accuracy No [67]
CRAG 2024 Comprehensive QA Accuracy, Hallucination No [68]
RAGuard 2025 Robustness Testing Resistance to Misinfo No [23]
CONFLICTS 2025 Conflict Resolution Appropriateness Score No [24]
Generation & Attribution
ALCE 2023 Citation Generation Citation Quality, Fluency No [69]
FActScore 2023 Factual Precision Atomic Fact Accuracy No [70]
AttrScore 2023 Attribution Quality Attribution Accuracy No [71]
RAGBench 2024 Industry RAG TRACe Metrics No [72]
OpinioRAG 2025 Opinion Summarization Sentiment Alignment Yes [18]
Table 6: Complete analysis of major RAG/LLM evaluation datasets (2013-2025). All high-impact benchmarks prioritize factual
accuracy over opinion synthesis, with only TruthfulQA and OpinioRAG showing partial opinion-awareness.
Table 7: Retrieval diversity improvements ( Δvs Raw KB)
across opinion extraction LLMs (𝑘=20, hybrid).
Metric Mistral Large Qwen 235B Sonnet 4.5
Sent-Int𝜎+24.2% +27.4% +26.8%*
Biz Impact𝜎+18.4% +12.4% +14.5%
GMS Band Cov. -17.5% -19.0%-13.0%*
Tenure Cov. -3.9% -3.6%-6.9%
Evidence Cov.+9.4% +8.9%+4.7%
Emot. Markers+28.1% +20.0% +23.6%*2. Success Attribution:The Raw KB never states whether mod-
erators actually help—it even admits “search results don’t show
clear feedback on whether moderators consistently respond.” The
Enriched KB explicitly captures positive outcomes: “moderators
are the only ones who have successfully helped.”
3. Emotional Language Preservation:The Raw KB sanitizes
seller language to “automated response loops. ” The Enriched KB pre-
serves the seller voice—“horrible loops”—providing more visceral
understanding of seller frustration.

Opinion-Aware RAG
Raw Discussions KB Response Opinion-Enriched KB Response
Based on seller discussions in the forums,there are several perspectives
on community management and moderators:
Requests for Moderator Intervention:Sellers frequently turn to mod-
erators when they feel stuck inautomated response loopsor cannot get
resolution through standard support channels. Multiple sellers specifically
call out moderators in their posts, asking for help with account issues,
policy violations, and reimbursement disputes.
Frustration with Support Systems:There’s significant frustration with
the overall support system that leads sellers to seek moderator help. Sellers
report dealing with what they describe as a “broken system.”
Moderator Responsiveness:Sellers appear to view moderators as a
potential escalation path when standard channels fail. However,the search
results don’t show clear feedbackon whether moderators consistently
respond or resolve these escalated issues.
Community Guidelines Enforcement:At least one seller reported
frustration with the moderation process for customer reviews, noting they
had reported guideline violations but received no response after 17 days.Sellers havemixed but generally positiveviews of community mod-
erators, often seeing them as acritical escalation pathwhen standard
support channels fail. Multiple sellers report thatmoderators are the
only ones who have successfully helpedthem resolve issues that auto-
mated systems could not address. Sellers turn to moderators when trapped
in what they describe as“horrible loops”of unhelpful automated re-
sponses.
However, sellersexpress frustrationwith community management’s
responsiveness to guideline violations. One seller reported waiting 17 days
without response. The reviewscontinued to negatively affecttheir
listings and sales.
Sellers alsorequest improvementsto forum functionality. One seller sug-
gested individual user accounts because the current setup causes irrelevant
notifications. This seller indicated they“rarely ever post”specifically
because of this issue.
When sellers reach out to moderators, it’s typicallyas a last resortafter
exhausting other options. They view moderator intervention asnecessary
for protectingtheir accounts from false claims and removing malicious
feedback.
Table 8: Side-by-side comparison. The Raw KB organizes byaction categoriesand admits “search results don’t show clear
feedback” on outcomes. The Opinion-Enriched KB capturessentiment distribution(mixed but positive),success attribution
(only ones who helped),behavioral explanations(rarely post, last resort), andconstructive suggestionsmissing from Raw KB.
4. Constructive Feedback:The Raw KB entirely omits con-
structive suggestions. The Enriched KB includes the forum function-
ality improvement request (individual user accounts, notification
issues)—a whole category of opinion the Raw KB missed.
5. Behavioral Insight:The Raw KB reports actions without
explaining motivations. The Enriched KB explainswhysellers be-
have certain ways: one seller “rarely ever posts” due to notificationissues; sellers turn to moderators “as a last resort after exhausting
other options.”
6. Narrative Coherence:The Raw KB reads as a categorized list
organized by topic buckets. The Enriched KB flows as a narrative
with contrast (“mixed but generally positive... However... ”), reading
like understanding rather than mere reporting.