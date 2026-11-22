# LSP-RAG: Latent Scoring Perspectives for Knowledge Graph Retrieval-Augmented Generation

**Authors**: Haochen Liu, Guanghui Min, Song Wang, Yada Zhu, Chen Chen, Jundong Li

**Published**: 2026-11-18 22:01:25

**PDF URL**: [https://arxiv.org/pdf/2611.25229v1](https://arxiv.org/pdf/2611.25229v1)

## Abstract
Large Language Models (LLMs) have demonstrated strong capabilities in open-domain question answering, but often struggle with factual accuracy and multi-hop reasoning due to the incompleteness of the training corpora. A promising solution is Knowledge Graph Retrieval-Augmented Generation (KG-RAG), which supplements LLMs with structured knowledge retrieved from external knowledge graphs (KGs). However, existing KG-RAG methods either rely on large-scale language models (e.g., ChatGPT) to guide the retrieval process, which leads to high computational costs, or suffer from limited retrieval quality when using lightweight language models, particularly under multi-hop scenarios. We propose LSP-RAG (Latent Scoring Perspectives for Retrieval-Augmented Generation over Knowledge Graphs), a novel KG-RAG framework that enhances hop-wise KG knowledge retrieval through multi-perspective prompted scoring. Each latent scoring perspective is guided by a combination of two types of soft prompts: a perspective-specific soft prompt induces a distinct prompt-conditioned scoring function over candidate triplets, while a contextual soft prompt evolves with each reasoning hop by encoding the query and previously explored KG triplets, enabling the model to preserve consistency and relevance across multi-hop retrieval. Rather than assuming manually defined semantic roles, these latent perspectives are adaptively aggregated to capture heterogeneous relevance signals during retrieval. This design allows LSP-RAG to perform accurate and robust retrieval using lightweight language models. LSP-RAG achieves superior performance across multiple KG-based Question Answering benchmarks compared to existing retrieval systems, including those that rely on much larger language models, demonstrating its effectiveness under limited computational budgets.

## Full Text


<!-- PDF content starts -->

LSP-RAG: Latent Scoring Perspectives for
Knowledge Graph Retrieval-Augmented Generation
Haochen Liu1, Guanghui Min1, Song Wang2, Yada Zhu3, Chen Chen1, Jundong Li1
1University of Virginia2University of Central Florida3IBM Research
sat2pv@virginia.edu, jjm8vr@virginia.edu, song.wang@ucf.edu,
yzhu@us.ibm.com, zrh6du@virginia.edu, jundong@virginia.edu
Abstract—Large Language Models (LLMs) have demonstrated
strong capabilities in open-domain question answering, but often
struggle with factual accuracy and multi-hop reasoning due to the
incompleteness of the training corpora. A promising solution is
Knowledge Graph Retrieval-Augmented Generation (KG-RAG),
which supplements LLMs with structured knowledge retrieved
from external knowledge graphs (KGs). However, existing KG-
RAG methods either rely on large-scale language models (e.g.,
ChatGPT) to guide the retrieval process, which leads to high
computational costs, or suffer from limited retrieval quality when
using lightweight language models, particularly under multi-hop
scenarios. We propose LSP-RAG (Latent Scoring Perspectives
for Retrieval-Augmented Generation over Knowledge Graphs), a
novel KG-RAG framework that enhances hop-wise KG knowledge
retrieval through multi-perspective prompted scoring. Each latent
scoring perspective is guided by a combination of two types of
soft prompts: aperspective-specific soft promptinduces a distinct
prompt-conditioned scoring function over candidate triplets, while
acontextual soft promptevolves with each reasoning hop by
encoding the query and previously explored KG triplets, enabling
the model to preserve consistency and relevance across multi-
hop retrieval. Rather than assuming manually defined semantic
roles, these latent perspectives are adaptively aggregated to
capture heterogeneous relevance signals during retrieval. This
design allows LSP-RAG to perform accurate and robust retrieval
using lightweight language models. LSP-RAG achieves superior
performance across multiple KG-based Question Answering
benchmarks compared to existing retrieval systems, including
those that rely on much larger language models, demonstrating
its effectiveness under limited computational budgets.
Index Terms—Knowledge Graph Retrieval-Augmented Gen-
eration, Knowledge Graph Question Answering, Multi-Hop
Reasoning, Latent Scoring Perspectives, Soft Prompting
I. INTRODUCTION
Large Language Models (LLMs) [1], [2] have demonstrated
impressive capabilities across a wide range of natural language
processing tasks, including text comprehension [3], open-
domain question answering [4], and natural language gen-
eration [5]. Despite these advances, LLMs remain limited due
to the incompleteness of knowledge encoded during pretraining,
which restricts their ability to provide factually accurate and
up-to-date answers [6], [7]. This limitation is particularly
pronounced in knowledge-intensive scenarios that demand
factual precision, reasoning over complex multi-hop relations,
and adaptation to continually evolving information [6], [7].
Consequently, LLM generations often contain hallucinated
content, outdated facts, or unverifiable claims, undermining
their reliability in real-world applications.To mitigate these issues, Retrieval-Augmented Generation
(RAG) [8] has been proposed to enhance LLM reliability by
incorporating external knowledge during inference. RAG allows
models to access up-to-date and factual information beyond
what was memorized during pretraining. While early RAG
systems primarily relied on unstructured text corpora, recent
efforts have explored structured sources of knowledge for more
precise and interpretable reasoning. Among these, Knowledge
Graphs (KGs) offer a highly structured and semantically
rich format, representing facts astripletsin the format of
(subject,relation,object) . This format supports explicit relation
modeling and multi-hop relational reasoning, making KGs
especially suitable for question answering tasks that require
factual consistency and transparency [9], [10]. In the KG-RAG
paradigm [11], models typically follow a two-step process:
(1)knowledge retrieval, where a subgraph containing query-
relevant triplets is constructed by expanding from entities
appearing in the query; and (2)prompt integration, where
triplets in the retrieved subgraph are presented as input to
the LLM as evidence for answer generation. The performance
of KG-RAG methods largely relies on the ability to retrieve
and rank relevant triplets effectively, especially under multi-
hop settings where irrelevant information can accumulate and
mislead reasoning.
Despite recent progress, current KG-RAG approaches still
face limitations that hinder their effectiveness. First, most
methods assess candidate triplet relevance through a single lens,
such as semantic similarity or prompt-based scoring [9], [12].
This rigid design fails to accommodate heterogeneous relevance
signals across different candidate triplets, lacking the flexibility
to account for diverse relational semantics [13]. A unified
scoring strategy cannot effectively handle such diversity, leading
to suboptimal triplet selection. Second, many candidate triplets
may appear irrelevant when judged in isolation, especially those
involving underspecified relations, but actually play a critical
role in connecting earlier reasoning steps to the final answer.
Existing methods typically treat each KG triplet independently
and fail to model such multi-hop dependencies [14], [15]. As
a result, they struggle to maintain contextual continuity, often
discarding useful evidence prematurely. Third, while most
recent systems attempt to address these issues by leveraging
large-scale language models such as ChatGPT to guide retrieval,
which are computationally expensive and impractical for
scalable or real-time applications [16]. Examples of the first
arXiv:2611.25229v1  [cs.IR]  18 Nov 2026

Can air pollution influence cognitive decline?
airpollutionincreaseKGOxidativestressdamageneuralfunctionleadtocognitivedeclineaffectsleeppatternslinkedtodaytimefatigue
Which actor starred in The Revenant?The Revenantstarred actorL. DiCaprioreleased year2015has genreAdventureChallenge 1: Heterogeneous SignalsChallenge 2: Contextual CoherenceWhich city is home to who collaborated with the artist of 'Empire State of Mind'?relatedtoEmpire State of Mindperformed byAliciaKeysNew YorkCity
What award did the actor of The Revenant win?directtemporalindirect
Irrelevant when judged in isolationRelevant to the questionFig. 1: Illustration of two key challenges in KG-RAG. In
Challenge 1, candidate triplets from the same entity may reflect
heterogeneous matching signals, such as direct factual rele-
vance, temporal association, or indirect relational connection,
making a single scoring strategy less flexible for effective
retrieval. In Challenge 2, a relation like related to may
appear irrelevant when judged in isolation, but becomes crucial
when combined with prior steps.
two limitations are shown in Figure 1.
To address these challenges, we propose LSP-RAG, a
lightweight but effective KG-RAG framework that enables
lightweight language models to perform accurate and multi-
hop retrieval. LSP-RAG contains amulti-perspective prompted
scoringmodule, which scores candidate KG triplets based
on the query and previously explored knowledge. Each latent
scoring perspective is parameterized by a learnableperspective-
specific soft prompt, inducing a distinct prompt-conditioned
scoring function over candidate triplets. Since supervision is
provided at the level of triplet relevance rather than manually
defined perspective labels, these latent scoring perspectives are
not intended to correspond to predefined semantic categories.
Instead, they serve as complementary scoring components that
are adaptively aggregated for flexible relevance estimation. In
addition, LSP-RAG incorporates acontextual soft promptthat is
dynamically constructed from the triplets in previous reasoning
steps. This prompt provides evolving contextual signals that
help the scoring of new candidate triplets using earlier reasoning
steps, allowing the model to assess each candidate not in
isolation, but in light of the evidence already accumulated,
addressing the limitation of prior work. By combining latent
scoring perspectives with contextual adaptation, our model
robustly filters irrelevant or noisy triplets. Notably, this frame-
work is fully compatible with lightweight models, avoiding the
reliance on expensive LLMs such as ChatGPT for retrieval,
thereby offering both efficiency and strong performance. Our
main contributions are listed as follows:•We propose LSP-RAG, a KG-RAG framework based on
latent scoring perspectives that enables lightweight language
models to conduct multi-hop KG retrieval, reducing the cost
of retrieval.
•We design two types of soft prompts to support our retrieval
process: (1)perspective-specific soft prompt, which induces
multiple latent scoring perspectives over candidate triplets,
and (2)contextual soft promptthat incorporates evidence
from the prior hop to support contextual retrieval. This
combination brings both flexible relevance estimation and
adaptivity into the retrieval process.
•We conduct extensive experiments on KGQA benchmark
datasets. LSP-RAG consistently outperforms existing KG-
RAG baselines across all settings. Remarkably, it even
surpasses ChatGPT retrieval despite using only lightweight
language models for retrieval, achieving better accuracy and
lower inference cost.
II. PROBLEMFORMULATION
We study the task of Knowledge Graph Retrieval-Augmented
Generation (KG-RAG), where structured knowledge from a
knowledge graph is selectively retrieved to assist a language
model in answering natural language questions. Formally, a
knowledge graph is represented as G= (E,R,T) , where
Edenotes the set of entities, Ris the set of relations, and
T={(h, r, t)|h, t∈ E, r∈ R} is the set of directed triplets.
Each triplet (h, r, t) encodes the fact that head entity his
connected to tail entity tvia relation r. Given a language
model and a question q, the objective is to identify a subset
of triplets from Tthat are relevant to answering q, and use
them as evidence to enhance the language model’s generation.
Formally, if T∗
q⊆ T denotes the selected triplet set, the final
answera qis generated as:
aq= LLM 
[q;T∗
q]
.(1)
HereT∗
qcontains the most relevant information for question
answering.
III. LSP-RAG
We propose LSP-RAG, a compact and adaptive framework
for Knowledge Graph Retrieval-Augmented Generation (KG-
RAG), designed to enhance multi-hop reasoning and knowledge
retrieval using lightweight language models. LSP-RAG operates
in a multi-hop retrieval setting, where relevant triplets from
the knowledge graph are selected iteratively based on both the
input query qand the reasoning context of explored knowledge.
As shown in Figure 2, the framework introduces two key
components: (1) amulti-perspective prompted scoringmodule,
where multiple latent scoring perspectives guided by dis-
tinctperspective-specific soft promptsprovide complementary
prompt-conditioned scoring functions; and (2)contextual soft
promptsthat are dynamically constructed from the query and
previously explored knowledge, enabling hop-aware retrieval
and maintaining coherence across multi-hop reasoning steps.
These components work together to provide a flexible and

context-sensitive evaluation of candidate triplets, allowing LSP-
RAG to robustly retrieve useful knowledge without relying on
enormous-scale LLMs.
A. Multi-Hop Retrieval Framework
Given a knowledge graph Gand a natural language question
q, our objective is to retrieve a set of triplets T∗
qfromGthat
provide faithful evidence for answering q. To achieve this,
we design an iterative retrieval framework that incrementally
explores the KG in a step-wise manner.
Step 1: Candidate Construction.To initiate retrieval, we
first identify a set of query entities Eqmentioned in the
question q, and initialize the first-hop frontier as E(0)=Eq.
At the t-th retrieval hop, we construct a set of candidate
triplets C(t)by expanding outward from the current frontier
E(t−1). Specifically, we include all outgoing triplets (e, r, e′)
fore∈ E(t−1), as well as reversed triplets (e, r−1, e′)whenever
(e′, r, e)appears in the KG:
C(t)={(e, r, e′)|e∈ E(t−1)} ∪
{(e, r−1, e′)|(e′, r, e)∈ G, e∈ E(t−1)}.(2)
This candidate construction step ensures broad coverage of
possible evidence paths, supporting both forward and backward
reasoning chains. The resulting set C(t)serves as the input to
the scoring module of hoptdescribed in the next step.
Step 2: Multi-Perspective Scoring and Triplet Selection.
Each candidate triplet x∈ C(t)is evaluated for its relevance to
the question qand the context knowledge explored before. To
model heterogeneous relevance signals and maintain contextual
consistency across hops, we employ a modular scoring mecha-
nism based on multi-perspective prompted scoring with soft
prompts (detailed in subsection III-B ). This module outputs a
contextualized score S(q, x, t) for each candidate. We retain
the top-Mtriplets byS(q, x, t)for further exploration:
T(t)
sel=Top-M{S(q, x, t)|x∈ C(t)}.(3)
This step prioritizes the relevant triplets while controlling the
expansion size and noise accumulation. We maintain a visited
setV(t)to track explored entities. At the beginning, we initialize
T∗
q=∅ andV(0)=E(0). At each hop t, the selected triplets
T(t)
selare added to the evidence set, and the set of new tail
entities is set as the frontier for the next hop:
T∗
q← T∗
q∪ T(t)
sel,(4)
E(t)={e′|(h, r, e′)∈ T(t)
sel} \ V(t−1),(5)
V(t)=V(t−1)∪ E(t).(6)
By keeping track of visited nodes, we avoid redundancy and
loops.
Iterative Retrieval.The above process is repeated for a fixed
number of hops H, progressively building an evidence graph
rooted in the query. The final triplet set T∗
qcontains the most
relevant facts and is passed to the downstream answer generator.
Unlike methods that rely on large-scale language models to
implicitly model reasoning chains, our framework performsretrieval in an explicit and structured way using only lightweight
language models for triplet scoring. This enables efficient
and scalable multi-hop retrieval with better evidence-level
interpretability and lower cost. The design of our scoring
module is key to enabling such structured reasoning and will
be detailed in the next subsection.
B. Multi-Perspective Prompted Scoring
To robustly evaluate candidate triplets in multi-hop retrieval,
we design a compact neural scoring module based on multi-
perspective prompted scoring with contextualized soft prompts.
Asoft promptrefers to a learnable continuous embedding
prepended to the input of a language model, allowing efficient
task adaptation without fine-tuning the model parameters.
This module addresses two key challenges in KG-RAG by
applying different soft prompts: (1) modeling heterogeneous
relevance signals with multiple latent scoring perspectives, and
(2) maintaining contextual coherence across hops. To this end,
each latent scoring perspective is equipped with two types
of trainable soft prompts: aperspective-specific soft prompt
that parameterizes a prompt-conditioned scoring function, and
acontextual soft promptthat evolves over retrieval steps by
summarizing the perspective’s prior output representations.
These components allow the model to flexibly score candidates
while remaining sensitive to context evolution across hops.
Each latent scoring perspective operates as an independent
scorer built on a shared, lightweight language model. Given a
question qand a candidate triplet xat hop t, thek-th of Klatent
scoring perspectives constructs its input by concatenating the
perspective-specific soft prompt pper
k, thecontextual soft prompt
pcxt
k(t), the question q, and a natural language serialization of
the candidate tripletxto obtain a new output embedding:
zk(q, x, t) = LM 
[pcxt
k(t);pper
k;q;x]
,(7)
where LM denotes the lightweight language model shared
across all latent scoring perspectives. pper
kis a learnable
embedding sequence that is unique to perspective kandpcxt
k(t)
encodes knowledge from the last hop. zk(q, x, t) encodes
prompt-conditioned relevance under the current perspective
and reasoning state.
To maintain temporal continuity, thecontextual soft prompt
is updated after each hop by aggregating perspective outputs
from the previous step:
pcxt
k(t) =g
1
|C(t−1)|X
x∈C(t−1)zk(q, x, t−1)
,(8)
where g(·) is a learned projection MLP layer. For the first
hop ( t= 1 ), this prompt is omitted in the input of LM.
This update allows each latent scoring perspective to track
its own contextual state across multiple retrieval steps. The
k-th perspective transforms its embedding into a scalar score
sk(q, x, t)∈(0,1) with a learnable linear layer wkand a
sigmoid layerσ:
sk(q, x, t) =σ(w⊤
kzk(q, x, t)),(9)

Subgraph Extraction
red meatfound inhealthy dietbe inaffectformbloodsupportcontainironsaladpopular inbemetalspinachlifeTriplets: (iPhone, made by, Apple); (Apple, hq in, Cupertino)What does spinach help with?
Spinach can support blood forming.What does spinach help with?Question:
GNN𝒔𝒓,𝒔𝒆LM encoderHop 1
𝒔𝒓𝒔𝒊𝒎containwater𝒔𝒓,𝒔𝒆𝑹𝒄𝒂𝒏𝒅: contain, be in …  𝑬𝒄𝒂𝒏𝒅: water, iron …
supportconcrete
spinachSelected path:Selected path:containironspinachcontainironsupportformblood
Question-Relevance ScoringLLM Stepwise Retrieval
Knowledge GraphGNN Score Filtering:Hop 2
Relation Selection𝒔𝒓,𝒔𝒆𝑹𝒄𝒂𝒏𝒅: support, be …  𝑬𝒄𝒂𝒏𝒅: formblood …supportEntity Selectionformblood
𝑹𝒄𝒂𝒏𝒅+𝒔𝒓𝒔𝒊𝒎𝑬𝒄𝒂𝒏𝒅+𝒔𝒆GNN Score Filtering:
Relation SelectioncontainEntity Selectioniron
𝑹𝒄𝒂𝒏𝒅+𝒔𝒓𝒔𝒊𝒎𝑬𝒄𝒂𝒏𝒅+𝒔𝒆Where is the company that makesiPhones headquartered?iPhonemade byApplecategorySmartphonereleasedin2007
E(1)={Apple}hq in
AppleCupertinofoundSteve Jobshas ceoTim Cook
popularinAsianis year of
FinancialCrisisFinancial CrisisiPhonemade byApplecategorySmartphonereleasedin2007hq inCupertinofoundSteve Jobshas ceoTim Cook
popularinAsianis year ofFinancial CrisisAppleAppleAppleiPhoneiPhoneiPhone𝓔(𝟎): 𝓔(𝟏): Apple
Where is the company that makes iPhones headquartered?The company that makes iPhones headquartered in Cupertino.
Candidate Construction𝓒(0)
𝓒(2)
MoE scoringTriplet 1: 0.92 Triplet 2: 0.36Triplet 3: 0.44
LightweightLM+Perspective-Specific SP
Latent Scoring Perspectives
Mixture-of-ExpertsTriplet 1: 0.95 Triplet 2: 0.58Triplet 3: 0.53
𝐩𝐤𝐜𝐱𝐭 updates!Contextual SP  𝐩𝐤𝐜𝐱𝐭(𝟏) :    ∅, ∅, ∅Contextual SP  𝐩𝐤𝐜𝐱𝐭(𝟐) :
selected
selected
在此处键入公式。MoE scoringMulti-Perspective Scoring
Candidate Construction
Answer GenerationCandidate Construction
Multi-Perspective Scoring
Retrieval ends!𝓣𝐤𝐜𝐱𝐭selectedFig. 2: Overview of LSP-RAG framework for KG-RAG. The framework consists of: (1) Candidate Construction: From the
frontier entities E(t−1), initially the entities in question q, the framework iteratively expands the candidate triplet set C(t)from
the KG. (2) Multi-Perspective Prompted Scoring: Candidate triplets are evaluated by a lightweight language model (LM) with
two types of soft prompts (SP): Perspective-specific soft prompts induce multiple latent scoring perspectives over candidate
triplets. Contextual soft prompt is updated across hops to propagate reasoning history and maintain coherence during multi-hop
inference. Perspective scores are adaptively aggregated into a final score for each candidate. Top-scoring triplets are selected at
each hop and accumulated into the evidence set. (3) Answer Generation: The collected evidence is provided to a large generator
for answer generation.
and an adaptive aggregation network assigns weights to
each perspective. Specifically, the lightweight language model
encoder produces a representation of the question and candidate
asf(q, x) , and a trainable aggregation matrix Wrmaps this
to a weight distribution onKperspectives for tripletx:
α(q, x, t) = softmax(W r·f(q, x)).(10)
The final score for candidate xis the weighted sum over latent
scoring perspectives:
S(q, x, t) =KX
k=1α(q, x, t)[k]·s k(q, x, t).(11)
This scoring architecture directly addresses the key chal-
lenges in multi-hop KG retrieval. Theperspective-specific soft
promptsinduce multiple prompt-conditioned scoring functions,
enabling flexible matching of heterogeneous relevance signals
without requiring predefined semantic roles. Since supervision
is provided at the level of triplet relevance rather than per-
spective labels, these latent scoring perspectives should not be
interpreted as human-defined reasoning categories. Instead, they
serve as latent scoring components that are adaptively aggre-
gated for each question-triplet pair. Thecontextual soft prompts
preserve reasoning continuity across hops by incorporating step-
wise retrieval history into scoring. Together, these components
allow the retriever to adapt its behavior based on both the
question semantics and the evolving context. Importantly, the
entire module operates over lightweight language models withsoft prompt tuning, offering an efficient and scalable alternative
to LLM-based retrievers. The overall retrieval procedure is
summarized in Algorithm 1.
C. Answer Generation
After the multi-hop selection process, we obtain the final
evidence triplet set T∗
qthat aggregates promising triplets across
hops. To generate the final answer, we leverage a powerful
large-scale language model, such as ChatGPT, which excels at
reasoning over textualized knowledge. Each triplet xis first
serialized into a natural language sentence. The evidence set
is then concatenated into a textual context. The LLM receives
the questionqand the retrieved evidenceT∗
qas input:
aq= LLM 
[q;T∗
q]
,(12)
where aqdenotes the generated answer. Since the LLM is
conditioned on the selected evidence, it grounds its reasoning
on relevant multi-hop chains rather than the entire KG.
While this generation step can leverage powerful LLMs,
our framework is designed to minimize reliance on them.
Due to the LSP-RAG retriever’s multi-perspective prompted
scoring mechanism built from compact language models and
soft prompts, the system can already filter highly relevant and
coherent evidence chains. This allows downstream generation
to focus on language fluency, rather than compensating for
missing knowledge or incorrect reasoning paths.
By explicitly addressing key challenges in KG-RAG, such
as heterogeneous relevance estimation via latent scoring

Algorithm 1LSP-RAG Retrieval (per question q; inference-
time scoring with learned prompts)
Input: KGG, query q, hops H, top- M, latent scoring per-
spectivesK
Input: Frozen LM encoder; learned parameters Θ =
{{pper
k}K
k=1, g,{w k}K
k=1, Wr}
1://Notes: pper
k∈RLper×d(e.g., Lper=5tokens), pcxt
init∈
RLcxt×d(e.g., Lcxt=1token), fis the encoding of the
LM encoder.
2:E(0)←LinkEntities(q) ,V(0)← E(0),T∗
q← ∅ ,
pcxt
init← ∅
3:fork= 1toKdo
4:pcxt
k(1)←pcxt
init
5:end for
6:fort= 1toHdo
7:C(t)←ExpandCandidates(G,E(t−1))
8:for allx∈ C(t)do
9:fork= 1toKdo
10:z k(q, x, t)←LM 
[pcxt
k(t);pper
k;q;x]
11:s k(q, x, t)←σ(w⊤
kzk(q, x, t))
12:end for
13:α(q, x, t)←softmax 
Wrf(q, x)
14:S(q, x, t)←PK
k=1α(q, x, t)[k]·s k(q, x, t)
15:end for
16:T(t)
sel←Top-Mcandidates inC(t)byS(q,·, t)
17:T∗
q← T∗
q∪ T(t)
sel
18:E(t)← {e′|(h, r, e′)∈ T(t)
sel} \ V(t−1)
19:V(t)← V(t−1)∪ E(t)
20:ift < Hthen
21:fork= 1toKdo
22: ¯zk(t)←1
|C(t)|X
x∈C(t)zk(q, x, t)
23:pcxt
k(t+1)←g ¯zk(t)
24:end for
25:end if
26:end for
Output:T∗
q
perspectives, cross-hop consistency via contextual soft prompts,
and LLM scale dependency via lightweight model retrieval, our
approach provides a scalable and interpretable solution. The
triplet evidence offers transparent reasoning traces, while the
latent scoring perspectives remain internal scoring components
and the generation stage remains modular and adaptable to
different deployment needs.
D. Training Objective
Our learnable components in LSP-RAG are trained by a con-
trastive learning process. Importantly, the underlying pretrained
language models (both the lightweight language model and
the large answer generator) remain frozen throughout training.
This ensures that learning focuses on adapting the lightweight
modules for scoring and information summarization.Supervision is provided by question-answer pairs together
with ground-truth subgraphs. For a given question q, in the
hopt, letT+
q,t∈ C(t)denote the set of gold triplets that lie
on reasoning paths leading a query entity in Eqto any answer
entity within a predefined path length, and let T−
q,t∈ C(t)denote
triplets sampled from the candidates that are not part of any
gold path. These triplets introduce useful contextual information
directly or indirectly relevant to the final answer. The objective
is to assign higher scores toT+
q,twhile suppressingT−
q,t.
We adopt a cross-entropy loss over triplets at hopt:
L=−1
|T+
q,t|X
x∈T+
q,tlog(S(q, x, t))
−1
|T−
q,t|X
x∈T−
q,tlog(1−S(q, x, t)),(13)
All trainable parameters are optimized jointly by backpropa-
gation. Through this objective, the scorer learns to emphasize
relevant triplets and suppress distractors, providing reliable
multi-hop evidence for the downstream answer generator.
IV. EXPERIMENTS
We evaluate the effectiveness of our proposed approach
across multiple QA benchmarks.
A. Datasets
We evaluate our method on four Question Answering
datasets: WebQuestionsSP (WebQSP) [17], WikiMovies [18],
and the datasets of MetaQA [19] in both 1-hop and 2-hop
settings. WebQSP contains 4,737 natural language questions
annotated with corresponding SPARQL queries over Free-
base [20], a broad-coverage knowledge graph. This dataset
requires both single- and multi-hop structured reasoning
and is commonly used to evaluate complex KG-based QA.
WikiMovies is a domain-specific dataset focused on the movie
domain, constructed from WikiData and Wikipedia, with around
100k question-answer pairs over 75,000 movie-related entities
targeting factual reasoning over a curated movie KG. MetaQA
builds upon WikiMovies by introducing paraphrased and more
compositional questions. In the 1-hop setting, each question is
grounded on a single KG fact, while the 2-hop version requires
2-hop inference to connect two supporting facts. Compared
to WikiMovies, MetaQA includes more natural language
variation and higher reasoning complexity, making it suitable
for evaluating generalization under compositionality. Together,
these datasets span both broad-domain and specialized-domain
KGs, and cover reasoning from simple fact retrieval to
multi-hop compositional queries. We follow the standard
train/validation/test splits provided by each benchmark.
B. Baselines
We evaluate our method against a broad set of baselines.
These baselines differ in how they retrieve and reason over
KG evidence, ranging from retrieval-free prompting to iterative
graph reasoning with large LLMs. In contrast, our method

achieves effective multi-hop reasoning using lightweight mod-
els. We evaluate our method against several widely-used and
representative baselines.IO-prompt[1] uses ChatGPT-3.5 to
directly generate answers without the assistance of any external
evidence, often resulting in hallucinations or factual errors due
to limited LLM internal knowledge.CoT-prompt[4] improves
upon this by encouraging the model to generate intermediate
reasoning steps, but its performance is still bounded by the
LLM’s pretraining knowledge.Self-Consistency[21] further
extends CoT by sampling multiple reasoning paths and selecting
the most frequent answer, improving robustness but still
lacking access to external facts.Sim-Retrieve[22] retrieves
KG triplets by computing embedding similarity between
questions and candidate triplets using LLaMA or Flan-T5.
While simple and efficient, it often retrieves noisy or irrelevant
facts, especially under multi-hop queries.G-Retriever[23]
formulates graph retrieval as a prize objective to select a
compact, relevant subgraph as context, improving scalability
and reducing hallucinations.GNN-RAG[24] uses a GNN
as a dense subgraph reasoner to retrieve answer candidates,
extracts KG paths connecting question entities to candidates,
verbalizes these paths, and feeds them to an LLM for final
reasoning. This improves faithfulness without extra LLM
calls for retrieval.GNN-Scoringintroduces a graph neural
network (GNN) with the same size as GNN-RAG to score
and filter triplets with a lightweight language model encoding
the query within the input of the GNN, but it still struggles
with long-range dependencies and lacks semantic adaptability.
Think-on-Graph (ToG)[9] improves retrieval reasoning by
iteratively selecting and scoring subgraphs using LLaMA, Flan-
T5, or ChatGPT-3.5, and generating answers with ChatGPT-
3.5. While effective, it incurs a higher computational cost
and still depends heavily on the size of LLMs.Reasoning
on Graphs (RoG)[25] adopts a planning-retrieval-reasoning
pipeline: it first generates KG-grounded relation-path plans,
then retrieves instantiated reasoning paths from the KG to
support faithful and interpretable reasoning. In contrast, our
method uses lightweight models to score and retrieve, achieving
accurate reasoning without relying on expensive models.
C. Experimental Setup
To evaluate the effectiveness of our proposed framework, we
conduct experiments usingChatGPT-3.5[26],LLaMA2-7B,
LLaMA3-8B[27], andFlan-T5-3B[28] as backbone language
models for retrieval and answering. In our framework, LLaMA
and Flan-T5 are used as retriever models for scoring and
selecting KG triplets, while ChatGPT-3.5 is used only during
the final answer generation stage for some baselines. For triplet
selection, the multi-perspective prompted scoring module is
configured with K= 3 latent scoring perspectives by default.
The multi-hop expansion process is controlled by a hop budget
H= 1 for MetaQA-1hop and H= 2 for other datasets,
balancing retrieval depth and noise. We cap the number of
retrieved triplets to 20 per query to maintain prompt length
constraints. Perspective-specific and contextual soft prompts
are jointly used to condition perspective scoring and improverelevance estimation under multi-hop contexts. All experiments
are conducted with 5 random runs, and we report the average
accuracy to mitigate variance.
D. Backbone Models
Our retrieval module operates on top of lightweight pre-
trained language models, enabling efficient and scalable scoring
without relying on large generative models during retrieval.
We experiment with three representative retriever backbones:
LLaMA2-7B, LLaMA3-8B, and Flan-T5-3B.
LLaMA.LLaMA [27] is a family of decoder-only transformer
models pre-trained on large-scale public corpora. We adopt the
7B/8B variants of LLaMA2/3 as feature extractors for triplet
scoring. In our setup, the LLaMA parameters are frozen, and
only the soft prompts, projection layers, and lightweight multi-
perspective scoring components are trained. As a decoder-only
model, LLaMA encodes the concatenated prompt and triplet
input into hidden representations, from which we use the latent
output representation for downstream scoring.
Flan-T5.Flan-T5 [28] is an instruction-tuned extension of
the T5 encoder-decoder architecture. The Flan variant is
trained on a large collection of prompting tasks, making it
effective under instruction-style input formats. We use the 3B
variant in encoder-only mode to encode the concatenated soft
prompts, question, and candidate triplet. Similar to LLaMA,
the backbone parameters of Flan-T5 are frozen throughout
training, with only the soft prompts, projection layers, and
lightweight scoring components updated.
Soft Prompt Integration.For both model families, we prepend
soft prompts as continuous embedding vectors at the input
layer. This design does not modify the tokenizer or the
internal architecture of the backbone language model. During
training, gradients are propagated through the perspective-
specific soft prompts, contextual soft prompts, projection
layer, and adaptive aggregation module. This makes LSP-RAG
compatible with different pre-trained language models while
keeping the backbone frozen.
E. Training Details
We train all models using the Adam optimizer with learning
rate1×10−4, batch size 20, and linear learning rate decay
over 5 epochs. Prompt parameters are initialized randomly and
trained from scratch. Each latent scoring perspective uses two
separate prompt embeddings: one perspective-specific and one
contextual. The underlying language models are kept frozen
throughout training, and only the soft prompts, projection layers,
and lightweight scoring components are updated. We train all
experiments on 4 NVIDIA A100 GPUs with 80GB memory
using mixed-precision training enabled via PyTorch. During
inference, retrieval is performed up to H= 1 hop for the
MetaQA-1hop dataset and H= 2 for others, with M= 20
triplets selected per hop. We set the number of latent scoring
perspectives to K= 3 . The dimension of the latent embeddings
in our method is set to the same dimension as the retrieval
LLM, i.e., 2,048 for Flan-T5-3B and 4,096 for LLaMA2/3.
All KG triplets are preprocessed into a bidirectional index,

TABLE I: Experimental results (accuracy in %) with standard deviation. We group methods into retrieval-free prompting,
ChatGPT retrieval, and lightweight LM retrieval. The best result within each lightweight LM group is shown inbold. The best
result among all compared methods is shown with an underline .
Category Method WebQSP WikiMovies MetaQA-1hop MetaQA-2hop
Retrieval-freeIO-prompt 62.32±0.41 69.24±0.35 60.02±0.60 23.01±0.79
CoT-prompt 62.74±0.46 72.45±0.42 64.98±0.52 26.71±0.88
Self-Consistency 61.11±0.53 73.64±0.37 71.38±0.39 30.52±1.01
ChatGPT retrievalToG 74.24±0.91 95.77 ±0.63 96.33 ±0.77 94.43 ±0.89
RoG 74.27 ±0.94 95.62±0.51 96.15±0.60 94.06±0.92
LLaMA2 retrievalSim-Retrieve 46.98±1.89 50.09±1.03 47.44±0.99 21.02±1.93
G-retriever 69.02±1.44 80.95±1.11 77.77±0.92 67.08±1.67
GNN-Scoring 63.86±0.73 78.32±0.48 83.82±0.52 71.63±0.91
GNN-RAG 63.43±0.34 77.52±0.62 82.89±0.63 70.35±0.84
ToG 66.70±0.85 86.62±0.59 88.63±0.67 78.01±0.93
RoG 64.19±0.89 85.14±0.74 86.08±0.73 76.34±0.90
LSP-RAG 75.02±0.88 98.29±0.76 97.25±0.99 94.91±0.72
LLaMA3 retrievalSim-Retrieve 47.16±1.01 51.02±1.08 47.46±0.70 22.96±1.49
G-retriever 70.12±1.74 82.04±1.32 76.21±0.90 69.23±1.87
GNN-Scoring 65.77±0.71 80.24±0.50 86.12±0.47 74.92±0.99
GNN-RAG 67.49±0.75 80.25±0.66 86.01±0.59 73.82±0.84
ToG 64.60±0.90 88.44±0.77 90.02±0.81 83.89±0.96
RoG 66.99±0.94 87.93±0.72 89.79±0.81 82.02±0.90
LSP-RAG 76.63±0.53 98.33±0.62 97.65±1.08 94.99±0.78
Flan-T5 retrievalSim-Retrieve 44.78±1.35 52.39±0.99 47.93±0.85 20.17±1.72
G-retriever 66.03±1.51 79.11±1.04 76.21±0.88 69.23±1.47
GNN-Scoring 64.56±0.65 79.10±0.44 84.69±0.49 73.68±0.86
GNN-RAG 62.51±0.62 78.06±0.38 83.98±0.54 73.86±0.91
ToG 72.25±0.77 94.80±0.78 90.67±0.66 89.60±0.98
RoG 50.95±1.57 96.11±0.66 90.12±0.68 88.18±0.91
LSP-RAG 88.59±0.83 98.22±0.74 96.84±0.87 94.95±0.95
which maps each entity to its outgoing and incoming relations.
This index is loaded into memory as a dictionary for efficient
candidate expansion at each hop.
F . Evaluation Protocol
We follow standard entity-level accuracy for all datasets,
considering a prediction correct if it matches any of the gold
answers. Each model is evaluated with 5 different random seeds,
and we report the average accuracy and standard deviation
across runs. For fair comparison, all baseline methods use
the same KG and entity linker. The knowledge graph used
for WikiMovies and WebQSP is provided by [29], while for
MetaQA we use the set from [30]. ToG baselines are re-
implemented with the same number of hops and selection
budget as LSP-RAG.Our implementation is based on PyTorch and Python 3.11.7.
We release our code at https://github.com/HaochenLiu2000/
LSP-RAG.
G. Results and Analysis
Table I shows that LSP-RAG consistently achieves the
highest accuracy across all datasets and model backbones,
significantly outperforming both retrieval-free prompting meth-
ods and alternative retrievers. Retrieval-free baselines such
as IO-prompt and CoT-prompt fall short due to their lack of
access to structured evidence. While embedding similarity-
based retrievers often fail to model multi-hop dependencies,
GNN-based retrievers cannot capture semantic alignment.
Despite modeling relationships of question and knowledge,
they provide limited flexibility in matching complex, com-

1 2 3 4
Number of Perspectives (K)657075808590Accuracy (%)
WebQSP
LLaMA2-7B
Flan-T5-3B
1 2 3 4
Number of Perspectives (K)93949596979899Accuracy (%)
WikiMovies
LLaMA2-7B
Flan-T5-3BFig. 3: QA accuracy with varying numbers of latent scoring perspectives K. We report results on WebQSP and WikiMovies
using both LLaMA2-7B and Flan-T5-3B. Increasing Kfrom 1 to 3 improves performance across all settings, as multiple
perspectives provide additional prompt-conditioned scoring capacity. Performance saturates or slightly drops at K= 4 , indicating
that a small number of latent scoring perspectives is sufficient for flexible triplet scoring without unnecessary redundancy.
TABLE II: Ablation results (accuracy in %) comparing the full LSP-RAG model with variants that remove the perspective-
specific (per) and/or contextual (cxt) soft prompts. We report accuracy on four datasets: WebQSP (WQ), WikiMovies (WM),
MetaQA-1hop (MQ-1), and MetaQA-2hop (MQ-2). The best results are shown inbold.
MethodLLaMA2-7B LLaMA3-8B Flan-T5-3B
WQ WM MQ-1 MQ-2 WQ WM MQ-1 MQ-2 WQ WM MQ-1 MQ-2
Full Model 75.02 98.29 97.25 94.91 76.63 98.33 97.65 94.99 88.59 98.22 96.84 94.95
w/o per 67.83 94.27 88.14 84.63 68.62 95.01 92.13 90.00 83.74 95.08 90.82 88.94
w/o cxt 70.01 95.46 N/A 90.91 71.04 95.47 N/A 91.48 86.23 97.31 N/A 91.18
w/o Both 63.98 86.01 N/A 77.86 65.01 87.35 N/A 79.71 71.65 93.88 N/A 86.97
positional semantics. In contrast, LSP-RAG demonstrates
strong performance even under lightweight language models,
outperforming ToG/RoG even when using ChatGPT as its
retriever. This highlights the effectiveness of LSP-RAG: the
perspective-specific soft prompts induce multiple latent scoring
perspectives for flexible relevance estimation, while the hop-
aware contextual soft prompts maintain coherence across
reasoning steps. Interestingly, Flan-T5-3B achieves substantially
higher accuracy on WebQSP, suggesting that the encoder
architecture like Flan-T5 is particularly well-suited for latent
triplet scoring and multi-hop relevance estimation in the open
domain. Importantly, LSP-RAG requires no LLM finetuning or
large-scale language model inference during retrieval, showing
that compact models, when equipped with the right inductive
biases, can also support accurate and robust reasoning.
H. Parameter Study
We further study the impact of the number of latent scoring
perspectives Kin the multi-perspective prompted scoring
module, focusing on WebQSP and WikiMovies with both
LLaMA2-7B and Flan-T5-3B as the LLMs for retrieval. The
results in Figure 3 show a clear upward trend when increasing
Kfrom 1 to 3: the models consistently achieve better QA
accuracy as multiple latent perspectives provide additional
prompt-conditioned scoring capacity. However, the benefit
quickly saturates beyond this point. In fact, while K= 4
performs slightly better than K= 3 on WikiMovies, theimprovement is marginal, and on WebQSP with LLaMA2-
7B the accuracy atK= 4even drops compared toK= 3.
This observation suggests that a moderate number of
latent scoring perspectives is sufficient to provide flexible
relevance estimation for effective triplet scoring. Using too
many perspectives may introduce redundancy and increases
training as well as inference cost, without yielding consistent
performance gains. For this reason, we set K= 3 as the
default configuration in our method, which achieves nearly the
same accuracy as K= 4 while being more cost-effective and
requiring lower computing cost.
I. Ablation Study
We examine the impact of removing two key components:
the perspective-specific soft prompts and the contextual soft
prompts. Results in Table II show that across all datasets
and LLM settings, removing either consistently degrades
performance. Without perspective-specific prompts, the model
collapses multiple latent scoring perspectives into a less flexible
scoring function, leading to weaker discrimination especially on
multi-hop datasets such as WebQSP and MetaQA-2hop where
heterogeneous relevance signals must be handled. Without
contextual prompts, the model loses continuity across hops and
struggles to select the right chain of evidence. This becomes
severe in multi-hop reasoning QA settings. Notably, removing
either type of prompt leads to consistent performance drops

across both models. Removing perspective-specific soft prompts
results in a larger degradation on both models, particularly
on complex reasoning settings like WebQSP, indicating the
importance of maintaining multiple prompt-conditioned scoring
functions. In contrast, the removal of contextual soft prompts
hurts both models on multi-hop datasets, highlighting the impor-
tance of preserving inter-hop coherence for complex reasoning.
Both components play complementary and indispensable roles
in enabling robust multi-hop retrieval.
V. RELATEDWORKS
A. Retrieval-Augmented Generation
Retrieval-Augmented Generation (RAG) [31] combines
LLMs with external memory to address knowledge staleness
and hallucination issues. The framework retrieves relevant pas-
sages from external corpora and conditions generation on both
the input query and retrieved context. Early variants include end-
to-end training approaches [32], dense passage retrieval opti-
mization [33], and fusion-in-decoder architectures [34]. Recent
advances have addressed key limitations through sophisticated
retrieval strategies: Self-RAG [35] enables adaptive retrieval
decisions through reflection tokens, RAPTOR [36] constructs
hierarchical tree structures via recursive clustering, Anthropic’s
Contextual Retrieval [37] reduces retrieval failures through con-
textual chunk embedding, GraphRAG [38] leverages knowledge
graphs for global reasoning, and CRAG [39] introduces retrieval
quality assessment mechanisms. Other developments include
HyDE [40] and agentic systems [41], [42]. However, most
methods are text-centric and do not fully leverage structured
knowledge graphs, limiting the reasoning transparency and
multi-hop capability.
B. Conditional Computation and Multi-View Scoring
Mixture of Experts (MoE) has emerged as a powerful
architecture for scaling model capacity while maintaining
computational efficiency through conditional computation [43],
[44]. Originally proposed to divide problem spaces into
homogeneous regions handled by specialized sub-networks [45],
MoE has been successfully adapted to large language models
where different experts are routed conditionally based on
the input. Recent advances in MoE architectures include
Mixtral 8x7B [46], which demonstrated that sparse MoE
models can outperform dense counterparts on knowledge-
intensive tasks, and Switch Transformer [43], which scales
to trillion-parameter models by routing tokens to different
expert networks through learned gating functions. However,
MoE models have been shown to excel at memorization-
intensive tasks but struggle with complex reasoning compared
to dense models [47], suggesting that conditional computation
alone does not necessarily provide interpretable reasoning
ability. In knowledge graph reasoning specifically, MoMoK [48]
introduces relation-guided modality knowledge experts for
multi-modal knowledge graph completion to handle different
relational contexts and modality-specific signals.
Different from conventional MoE architectures that rely
on specialized expert networks, our work uses lightweight
latent scoring perspectives induced by perspective-specificsoft prompts. These perspectives are not designed as human-
defined semantic experts; instead, they provide multiple prompt-
conditioned scoring functions that are adaptively aggregated
for KG triplet relevance estimation. This design preserves the
efficiency benefits of lightweight conditional scoring while
keeping the reasoning trace explicit through retrieved KG
evidence paths, offering an effective way for KG-RAG.
VI. LIMITATIONS
Although LSP-RAG achieves strong retrieval and QA per-
formance with lightweight language models, it still has several
limitations. First, its computational cost increases with the
number of latent scoring perspectives, since each candidate
triplet is evaluated by multiple prompt-conditioned components.
Second, LSP-RAG relies on hop-wise KG traversal with a fixed
hop budget, which may miss evidence requiring long, indirect,
or weakly connected reasoning paths. Third, our experiments
mainly focus on relatively clean and well-structured KGs;
extending the framework to incomplete, noisy, or large-scale
heterogeneous graphs remains future work. Finally, latent
scoring perspectives are not intended to correspond to human-
defined semantic categories. The interpretability of LSP-RAG
mainly comes from explicit KG triplets and hop-wise evidence
paths, while controllable or semantically supervised scoring
perspectives are left for future work.
VII. CONCLUSION
In this work, we proposed a novel knowledge graph-
augmented retrieval framework for LLM-based question an-
swering called LSP-RAG. Our approach introduces a multi-
perspective prompted scoring module that leverages both
perspective-specific and contextual soft prompts to evaluate
candidate triplets, enabling robust multi-hop retrieval across
knowledge graphs. Through the designed hop-aware retrieval
and selection process, our method ensures contextual and
adaptive evidence exploration while maintaining efficiency.
Importantly, the design does not require fine-tuning large
backbone language models; instead, it relies on a compact
scoring architecture while still fully exploiting the capabilities
of LLMs for final answer generation. Extensive experiments on
the benchmarks demonstrate the effectiveness of our approach.
It consistently outperforms retrieval-based baselines, including
ToG/RoG, even when the latter are powered by larger models
such as ChatGPT. Our ablation studies further confirm the
critical roles of perspective-specific and contextual prompts
in improving multi-hop retrieval performance. Overall, our
retrieval and scoring mechanisms enhance the reasoning ability
of LLMs without relying on large models. We believe that
LSP-RAG provides a general and cost-effective paradigm for
future research in knowledge-enhanced LLM reasoning.
VIII. ACKNOWLEDGEMENT
This work was supported in part by the National Science
Foundation (NSF) under Grants IIS-2144209, IIS-2223769,
BCS-2228534, CMMI-2411248, and IIBR-2601942; by the
Office of Naval Research (ONR) under Grant N000142412636;
and by the Commonwealth Cyber Initiative (CCI) under Grant
HV-4Q26-073.

REFERENCES
[1]T. B. Brown, B. Mann, N. Ryder, M. Subbiah, J. Kaplan, P. Dhariwal,
A. Neelakantan, P. Shyam, G. Sastry, A. Askell, S. Agarwal, A. Herbert-
V oss, G. Krueger, T. Henighan, R. Child, A. Ramesh, D. M. Ziegler,
J. Wu, C. Winter, C. Hesse, M. Chen, E. Sigler, M. Litwin, S. Gray,
B. Chess, J. Clark, C. Berner, S. McCandlish, A. Radford, I. Sutskever,
and D. Amodei, “Language models are few-shot learners,” inNeurIPS,
2020.
[2]H. Touvron, T. Lavril, G. Izacard, X. Martinet, M.-A. Lachaux, T. Lacroix,
B. Rozi `ere, N. Goyal, E. Hambro, F. Azharet al., “Llama: Open and
efficient foundation language models,”arXiv preprint arXiv:2302.13971,
2023.
[3]M. Lewis, Y . Liu, N. Goyal, M. Ghazvininejad, A. Mohamed, O. Levy,
V . Stoyanov, and L. Zettlemoyer, “Bart: Denoising sequence-to-sequence
pre-training for natural language generation, translation, and comprehen-
sion,” inACL, 2020.
[4] J. Wei, X. Wang, D. Schuurmans, M. Bosma, F. Xia, E. Chi, Q. V . Le,
D. Zhouet al., “Chain-of-thought prompting elicits reasoning in large
language models,” inNeurIPS, 2022.
[5]S. Cheng, B. Tian, Q. Liu, X. Chen, Y . Wang, H. Chen, and N. Zhang,
“Can we edit multimodal large language models?” inEMNLP, 2023.
[6]C. Zheng, L. Li, Q. Dong, Y . Fan, Z. Wu, J. Xu, and B. Chang, “Can
we edit factual knowledge by in-context learning?” inEMNLP, 2023.
[7]S. Wang, Y . Zhu, H. Liu, Z. Zheng, C. Chenet al., “Knowledge editing
for large language models: A survey,”ACM Computing Surveys, 2024.
[8]Y . Gao, Y . Xiong, X. Gao, K. Jia, J. Pan, Y . Bi, Y . Dai, J. Sun, H. Wang,
and H. Wang, “Retrieval-augmented generation for large language models:
A survey,”arXiv preprint arXiv:2312.10997, 2023.
[9]J. Sun, C. Xu, L. Tang, S. Wang, C. Lin, Y . Gong, H.-Y . Shum, and J. Guo,
“Think-on-graph: Deep and responsible reasoning of large language model
with knowledge graph,” inICLR, 2024.
[10] J. Jiang, K. Zhou, X. Zhao, Y . Song, C. Zhu, H. Zhu, and J. Wen, “Kg-
agent: An efficient autonomous agent framework for complex reasoning
over knowledge graph,” inACL, 2025.
[11] D. Sanmartin, “Kg-rag: Bridging the gap between knowledge and
creativity,”arXiv preprint arXiv:2405.12035, 2024.
[12] W. Shi, S. Min, M. Yasunaga, M. Seo, R. James, M. Lewis, L. Zettle-
moyer, and W.-t. Yih, “REPLUG: Retrieval-augmented black-box lan-
guage models,” inNAACL, 2024.
[13] A. Saxena, A. Tripathi, and P. Talukdar, “Improving multi-hop question
answering over knowledge graphs using knowledge base embeddings,”
inACL, 2020.
[14] Y . Feng, X. Chen, B. Y . Lin, P. Wang, J. Yan, and X. Ren, “Scalable
multi-hop relational reasoning for knowledge-aware question answering,”
inEMNLP, 2020.
[15] X. Zhang, A. Bosselut, M. Yasunaga, H. Ren, P. Liang, C. D. Manning,
and J. Leskovec, “Greaselm: Graph reasoning enhanced language models
for question answering,”arXiv preprint arXiv:2201.08860, 2022.
[16] S. Min, D. Chen, H. Hajishirzi, and L. Zettlemoyer, “A discrete hard
EM approach for weakly supervised question answering,” inEMNLP,
2019.
[17] W.-t. Yih, M. Richardson, C. Meek, M.-W. Chang, and J. Suh, “The
value of semantic parse labeling for knowledge base question answering,”
inACL, 2016.
[18] A. Miller, A. Fisch, J. Dodge, A.-H. Karimi, A. Bordes, and J. Weston,
“Key-value memory networks for directly reading documents,” inEMNLP,
2016.
[19] Y . Zhang, H. Dai, Z. Kozareva, A. J. Smola, and L. Song, “Variational
reasoning for question answering with knowledge graph,” inAAAI, 2018.
[20] K. Bollacker, C. Evans, P. Paritosh, T. Sturge, and J. Taylor, “Freebase: a
collaboratively created graph database for structuring human knowledge,”
inSIGMOD, 2008.
[21] X. Wang, J. Wei, D. Schuurmans, Q. Le, E. Chi, S. Narang, A. Chowdhery,
and D. Zhou, “Self-consistency improves chain of thought reasoning in
language models,”arXiv preprint arXiv:2203.11171, 2022.
[22] J. Baek, A. F. Aji, and A. Saffari, “Knowledge-augmented language
model prompting for zero-shot knowledge graph question answering,” in
ACL Workshop on Matching Entities, 2023.
[23] X. He, Y . Tian, Y . Sun, N. V . Chawla, T. Laurent, Y . LeCun, X. Bresson,
and B. Hooi, “G-retriever: Retrieval-augmented generation for textual
graph understanding and question answering,” inNeurIPS, 2024.[24] C. Mavromatis and G. Karypis, “GNN-RAG: graph neural retrieval for
efficient large language model reasoning on knowledge graphs,” inACL,
2025.
[25] L. Luo, Y . Li, G. Haffari, and S. Pan, “Reasoning on graphs: Faithful
and interpretable large language model reasoning,” inICLR, 2024.
[26] OpenAI, “ChatGPT: Optimizing language models for dialogue.” 2022.
[27] H. Touvron, L. Martin, K. R. Stone, P. Albert, A. Almahairi, Y . Babaei,
N. Bashlykov, S. Batra, P. Bhargava, S. Bhosale, D. M. Bikel, L. Blecher,
C. Cant ´on Ferrer, M. Chen, G. Cucurull, D. Esiobu, J. Fernandes, J. Fu,
W. Fu, B. Fuller, C. Gao, V . Goswami, N. Goyal, A. S. Hartshorn,
S. Hosseini, R. Hou, H. Inan, M. Kardas, V . Kerkez, M. Khabsaet al.,
“Llama 2: Open foundation and fine-tuned chat models,”arXiv preprint
arXiv:2307.09288, 2023.
[28] J. Wei, M. Bosma, V . Y . Zhao, K. Guu, A. W. Yu, B. Lester, N. Du,
A. M. Dai, and Q. V . Le, “Finetuned language models are zero-shot
learners,” inICLR, 2022.
[29] H. Sun, B. Dhingra, M. Zaheer, K. Mazaitis, R. Salakhutdinov, and
W. W. Cohen, “Open domain question answering using early fusion of
knowledge bases and text,”EMNLP, 2018.
[30] J. Kim, Y . Kwon, Y . Jo, and E. Choi, “KG-GPT: A general framework
for reasoning on knowledge graphs using large language models,” in
EMNLP, 2023.
[31] P. Lewis, E. Perez, A. Piktus, F. Petroni, V . Karpukhin, N. Goyal,
H. K ¨uttler, M. Lewis, W.-t. Yih, T. Rockt ¨aschelet al., “Retrieval-
augmented generation for knowledge-intensive nlp tasks,” inNeurIPS,
2020.
[32] G. Izacard, P. Lewis, M. Lomeli, L. Hosseini, F. Petroni, T. Schick,
J. Dwivedi-Yu, A. Joulin, S. Riedel, and E. Grave, “Atlas: Few-shot
learning with retrieval augmented language models,”Journal of Machine
Learning Research, 2023.
[33] V . Karpukhin, B. Oguz, S. Min, P. S. Lewis, L. Wu, S. Edunov, D. Chen,
and W.-t. Yih, “Dense passage retrieval for open-domain question
answering.” inEMNLP, 2020.
[34] G. Izacard and E. Grave, “Leveraging passage retrieval with generative
models for open domain question answering,” inEACL, 2021.
[35] A. Asai, Z. Wu, Y . Wang, A. Sil, and H. Hajishirzi, “Self-rag: Learning
to retrieve, generate, and critique through self-reflection,” 2024.
[36] P. Sarthi, S. Abdullah, A. Tuli, S. Khanna, A. Goldie, and C. D. Manning,
“Raptor: Recursive abstractive processing for tree-organized retrieval,” in
ICLR, 2024.
[37] Anthropic, “Introducing contextual retrieval,” https://www.anthropic.com/
news/contextual-retrieval, 2024.
[38] D. Edge, H. Trinh, N. Cheng, J. Bradley, A. Chao, A. Mody, S. Truitt,
D. Metropolitansky, R. O. Ness, and J. Larson, “From local to global:
A graph rag approach to query-focused summarization,”arXiv preprint
arXiv:2404.16130, 2024.
[39] S.-Q. Yan, J.-C. Gu, Y . Zhu, and Z.-H. Ling, “Corrective retrieval
augmented generation,”arXiv preprint arXiv:2401.15884, 2024.
[40] L. Gao, X. Ma, J. Lin, and J. Callan, “Precise zero-shot dense retrieval
without relevance labels,” inACL, 2023.
[41] A. Singh, A. Ehtesham, S. Kumar, and T. T. Khoei, “Agentic retrieval-
augmented generation: A survey on agentic rag,”arXiv preprint
arXiv:2501.09136, 2025.
[42] G. Dong, J. Jin, X. Li, Y . Zhu, Z. Dou, and J.-R. Wen, “Rag-
critic: Leveraging automated critic-guided agentic workflow for retrieval
augmented generation,” inACL, 2025.
[43] W. Fedus, B. Zoph, and N. Shazeer, “Switch transformers: Scaling to
trillion parameter models with simple and efficient sparsity,”JMLR, 2022.
[44] W. Cai, J. Jiang, F. Wang, J. Tang, S. Kim, and J. Huang, “A survey on
mixture of experts in large language models,”TKDE, 2025.
[45] R. A. Jacobs, M. I. Jordan, S. J. Nowlan, and G. E. Hinton, “Adaptive
mixtures of local experts,”Neural computation, 1991.
[46] A. Q. Jiang, A. Sablayrolles, A. Roux, A. Mensch, B. Savary, C. Bamford,
D. S. Chaplot, D. d. l. Casas, E. B. Hanna, F. Bressandet al., “Mixtral
of experts,”arXiv preprint arXiv:2401.04088, 2024.
[47] S. Jelassi, C. Mohri, D. Brandfonbrener, A. Gu, N. Vyas, N. Anand,
D. Alvarez-Melis, Y . Li, S. M. Kakade, and E. Malach, “Mixture of
parrots: Experts improve memorization more than reasoning,” inICLR,
2025.
[48] Y . Zhang, Z. Chen, L. Guo, Y . Xu, B. Hu, Z. Liu, W. Zhang, and
H. Chen, “Multiple heads are better than one: Mixture of modality
knowledge experts for entity representation learning,” inICLR, 2025.