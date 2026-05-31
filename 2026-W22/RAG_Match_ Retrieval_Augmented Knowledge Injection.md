# RAG-Match: Retrieval-Augmented Knowledge Injection and Hierarchical Reasoning for Calibrated Semantic Relevance

**Authors**: Hengjun Jiang, Liansheng Sun, Yan Jiang, Xiaojie Ke, Yongjin Wang, Xiangkun Liu, Cunxin Gu, Jian Xu, Guanjun Jiang

**Published**: 2026-05-25 06:42:39

**PDF URL**: [https://arxiv.org/pdf/2605.25486v1](https://arxiv.org/pdf/2605.25486v1)

## Abstract
Semantic relevance judgment for search is particularly challenging in knowledge-intensive scenarios, where accurate ranking requires not only semantic matching but also background grounding, multi-step reasoning, and well-calibrated decision boundaries. Existing relevance models mainly rely on direct label supervision or shallow semantic similarity, which limits their ability to handle implicit intent, factual equivalence, and fine-grained relevance distinctions. To address this issue, we propose \textsc{RAG-Match}, a three-stage framework that integrates knowledge-augmented pretraining, hierarchical reasoning alignment, and preference-based decision calibration for relevance modeling. The key idea is to first strengthen query-centered semantic grounding, then align the model with structured relevance reasoning, and finally correct decision-level inconsistencies in difficult boundary cases. Experimental results on a real-world search relevance benchmark show that \textsc{RAG-Match} consistently outperforms strong LLM-based baselines across multiple ranking metrics, demonstrating the effectiveness of combining knowledge injection, reasoning supervision, and preference optimization for fine-grained relevance judgment.

## Full Text


<!-- PDF content starts -->

RAG-Match: Retrieval-Augmented Knowledge Injection and
Hierarchical Reasoning for Calibrated Semantic Relevance
Judgment
Hengjun Jiang∗,†Liansheng Sun∗Yan Jiang∗,‡Xiaojie Ke Yongjin Wang
Xiangkun Liu Cunxin Gu Jian Xu
Guanjun Jiang
Qwen Applications Business Group of Alibaba
May 26, 2026
Abstract
Semantic relevance judgment for search is partic-
ularly challenging in knowledge-intensive scenarios,
where accurate ranking requires not only seman-
tic matching but also background grounding, multi-
step reasoning, and well-calibrated decision bound-
aries. Existing relevance models mainly rely on di-
rect label supervision or shallow semantic similarity,
which limits their ability to handle implicit intent,
factual equivalence, and fine-grained relevance dis-
tinctions. To address this issue, we proposeRAG-
Match, a three-stage framework that integrates
knowledge-augmented pretraining, hierarchical rea-
soning alignment, and preference-based decision cal-
ibration for relevance modeling. The key idea is to
first strengthen query-centered semantic grounding,
then align the model with structured relevance rea-
soning, and finally correct decision-level inconsisten-
cies in difficult boundary cases. Experimental results
on a real-world search relevance benchmark show
thatRAG-Matchconsistently outperforms strong
LLM-based baselines across multiple ranking metrics,
demonstrating the effectiveness of combining knowl-
edge injection, reasoning supervision, and preference
∗Equal contribution.
‡Work done during an internship at Alibaba.
†Corresponding author:
jianghengjun.jhj@alibaba-inc.com.optimization for fine-grained relevance judgment.
Keywords:semantic relevance judgment; infor-
mation retrieval; retrieval-augmented generation;
retrieval-augmented pretraining; hierarchical reason-
ing; chain-of-thought; preference alignment; large
language models; knowledge-intensive search
1 Introduction
Relevance judgment is a core problem in information
retrieval, especially in search systems where rank-
ing quality directly determines user experience[1].
While traditional retrieval and ranking methods have
achieved strong performance on head queries and lit-
eral matching scenarios[2], they often struggle when
relevance depends on implicit user intent, factual
equivalence, or external domain knowledge. For ex-
ample, a query such as “2024 Olympics” may need
to be matched with documents referring to “the 33rd
Olympic Games” even though the lexical overlap is
limited[3]. Recent advances in neural language mod-
els have enabled more sophisticated semantic match-
ing [4], yet integrating explicit knowledge and reason-
ing capabilities into relevance models remains a sig-
nificant challenge. In such cases, accurate relevance
judgment requires not only semantic matching but
also knowledge grounding and reasoning over latent
constraints.
1arXiv:2605.25486v1  [cs.IR]  25 May 2026

Early relevance models mainly relied on lexical
overlap and statistical scoring functions such as TF-
IDF and BM25[5, 6]. Later, neural ranking mod-
els and pretrained language models substantially im-
proved semantic matching by learning richer inter-
actions between queries and documents[7, 8]. How-
ever, these approaches still face clear limitations in
knowledge-intensive search scenarios. First, many
difficult relevance decisions depend on background
knowledge that is not explicitly present in the query-
document pair. Second, direct label supervision of-
ten encourages shortcut learning, where models over-
rely on surface-level similarity rather than reason-
ing about whether the document truly satisfies the
query intent. Third, although large language mod-
els (LLMs) exhibit strong reasoning ability[9, 10],
naively applying them to relevance judgment often
leads to unstable intermediate reasoning, halluci-
nated constraints[11, 12], and overestimation of doc-
uments with high keyword overlap[13].
A key challenge is that retrieved evidence is often
fragmented, whereas relevance judgment requires a
compact and query-focused understanding of the un-
derlying knowledge. Simply exposing the model to
retrieved passages is insufficient. Instead, the model
should learn to map a query to a synthesized support-
ing document that organizes the most relevant evi-
dence into a coherent form. Moreover, this grounded
semantic understanding should be further translated
into faithful document-level reasoning and calibrated
against hard-boundary errors.
A straightforward alternative would be to adopt
RAG with prompting and attach retrieved evidence
at inference time. However, this approach remains
limited by context window length, inference latency,
and the transient nature of in-context knowledge use.
In production ranking systems, repeatedly concate-
nating large amounts of external evidence is often
infeasible, and in our deployment setting we do not
request a RAG-synthesized document during online
inference. This motivates our Phase I design: in-
stead of relying on external knowledge only at pre-
diction time, we use retrieval-enhanced synthesized
documents as a training signal so that query-relevant
knowledge can be internalized into the model param-
eters. In this way, the model acquires a strongerquery-centered semantic prior that supports more ef-
ficient and accurate downstream relevance judgment.
Motivated by these observations, we propose
RAG-Match, a multi-stage framework for seman-
tic relevance judgment that combines retrieval-
augmented pretraining, hierarchical reasoning align-
ment, and preference-based calibration.RAG-
Matchconsists of three phases. In the first phase,
we introduce a query-to-synthesized-document pre-
training objective. For each query, a RAG system
first retrieves the top-kmost relevant evidence doc-
uments and then synthesizes a document that best
answers the query. By training the model to generate
this RAG-synthesized document from the query, we
inject relevant background knowledge into the model
and enable it to learn the mapping from queries to
their most informative supporting content. In the
second phase, we propose a Hierarchical Reasoning
Alignment (HRA) framework that decomposes rel-
evance judgment into two stages. The model first
performs query grounding using retrieved knowledge
and constructs a structured semantic representation
of the query intent and constraints. It then eval-
uates the candidate document under this grounded
frame through multi-dimensional relevance analysis.
In the third phase, we introduce Discrepancy-Guided
Preference Optimization (DGPO), which uses a small
amount of human-annotated data to calibrate hard-
boundary decisions and reduce the systematic over-
estimation tendency of LLM-based relevance predic-
tors.
Compared with prior relevance models,RAG-
Matchoffers three advantages. First, it incorpo-
rates external knowledge into the model parame-
ters through scalable retrieval-augmented supervi-
sion, rather than relying solely on runtime retrieval
or label-only supervision. Second, it transforms rel-
evance judgment from a direct classification prob-
lem into a structured reasoning process, improving
both prediction quality and interpretability. Third,
it achieves effective calibration with limited human
annotations by focusing preference optimization on
cases that are most easily confused.
Our main contributions are summarized as follows:
•We proposeRAG-Match, a three-stage frame-
2

work for knowledge-intensive relevance model-
ing that progressively integrates query-centered
grounding, hierarchical reasoning alignment,
and decision calibration. The framework is de-
signed to address relevance judgment scenarios
where accurate ranking requires not only seman-
tic matching, but also background knowledge,
multi-step reasoning, and fine-grained boundary
discrimination.
•We introduce a knowledge-augmented pretrain-
ing stage that improves downstream relevance
modeling by learning query-to-synthesized-
document generation. Rather than serving as a
standalone ranking objective, this stage provides
stronger semantic grounding and knowledge-
aware initialization for subsequent reasoning
alignment, leading to improved ranking quality
after Phase II.
•We design a hierarchical reasoning alignment
stage and a discrepancy-guided preference op-
timization stage to improve relevance deci-
sion quality from complementary perspectives.
Phase II distills structured relevance reasoning
from a strong teacher model, while Phase III fur-
ther calibrates final relevance judgments by cor-
recting the systematic overestimation tendency
observed after supervised reasoning alignment.
•We conduct extensive experiments and analy-
sis on a real-world search relevance benchmark.
The results show thatRAG-Matchconsistently
outperforms strong LLM-based baselines across
multiple ranking metrics. Additional analysis
further clarifies the distinct roles of the three
stages, including the contribution of Phase I to
downstream alignment, the high reliability of
teacher-generated CoT supervision, and the ef-
fectiveness of DGPO in improving decision cali-
bration.
Extensive experiments on a search relevance bench-
mark demonstrate thatRAG-Matchconsistently
improves ranking quality and ranking consistency
over strong LLM baselines. Ablation studies fur-
ther show that the three phases are complemen-tary, with the largest gains coming from hierarchi-
cal reasoning alignment and additional improvements
from retrieval-augmented pretraining and preference-
based calibration.
2 Related Work
This work lies at the intersection of relevance mod-
eling in information retrieval, large language models
for relevance judgment, retrieval-augmented reason-
ing, and preference-based model alignment. Below,
we review the most relevant lines of research and po-
sitionRAG-Matchamong them.
2.1 Relevance Modeling in Informa-
tion Retrieval
Relevance modeling has long been a central prob-
lem in information retrieval (IR). Early approaches
mainly relied on lexical matching and statistical scor-
ing functions such as TF-IDF and BM25 [5, 6, 14, 15].
These methods are effective for exact matching and
head queries, but they often struggle when relevance
depends on paraphrasing, implicit intent, or factual
equivalence.
To overcome the limitations of lexical match-
ing, neural retrieval and ranking models were intro-
duced to learn distributed semantic representations
of queries and documents. Representative early mod-
els include DSSM [16] and CDSSM [17], which project
query-document pairs into dense vector spaces for se-
mantic matching. Later work explored richer inter-
action architectures, including DRMM [18], Match-
Pyramid [19], and attention-based matching mod-
els [20, 21], which improved fine-grained interaction
modeling between queries and documents.
The emergence of pretrained language models fur-
ther advanced neural relevance modeling. BERT-
based rerankers, such as BERT-Ranker [22] and
BERT-PLI [23], demonstrated that deep contex-
tual representations can significantly improve ranking
quality. Subsequent work also proposed more efficient
architectures for large-scale retrieval and reranking,
including ColBERT [24] and SPLADE [25]. De-
spite these advances, most relevance models are still
3

trained with direct label supervision and therefore
remain limited in knowledge-intensive settings where
the query-document pair alone does not provide suf-
ficient evidence for accurate judgment.
In contrast to prior relevance models that pri-
marily focus on semantic matching,RAG-Match
emphasizes query-centered knowledge grounding and
structured reasoning. Our framework uses retrieval-
augmented pretraining to inject external knowledge
into the model and then performs relevance judgment
through hierarchical reasoning rather than direct la-
bel prediction alone.
2.2 Large Language Models for Rele-
vance Judgment
Large language models (LLMs) have recently
emerged as powerful tools for ranking and relevance
judgment. Models such as GPT-3 [26], T5 [27], and
later instruction-tuned variants have shown strong
zero-shot and few-shot capabilities, making them at-
tractive for low-resource relevance tasks [28, 29, 30].
In the retrieval community, works such as RankGPT
[31] demonstrated that LLMs can perform competi-
tive ranking through pairwise or listwise prompting,
while instruction-based retrieval approaches such as
TART [32] showed that task-aware prompting can
dynamically adjust matching behavior.
To reduce deployment costs, several studies have
explored fine-tuning or distilling smaller LLMs for
ranking and reranking. RankLLaMA [33] showed
that decoder-only LLMs can be adapted to re-
trieval tasks through supervised fine-tuning, while
other work distilled ranking knowledge from stronger
teacher models into more efficient students [34, 35].
These studies indicate that LLMs provide a promis-
ing foundation for relevance modeling beyond tradi-
tional discriminative architectures.
However, applying LLMs directly to relevance
judgment also introduces new challenges. Their pre-
dictions can be sensitive to prompting style, their
intermediate reasoning may drift without sufficient
grounding, and they often overestimate documents
with strong lexical overlap even when key seman-
tic constraints are violated.RAG-Matchaddresses
these issues by combining LLM-based reasoning withretrieval-augmented pretraining and targeted prefer-
ence alignment, thereby improving both the ground-
ing and calibration of relevance decisions.
2.3 Retrieval-Augmented Reasoning
and Chain-of-Thought Supervisi-
on
Retrieval-Augmented Generation (RAG) has be-
come a standard paradigm for incorporating external
knowledge into language models [36]. By retrieving
evidence from an external corpus and conditioning
generation on that evidence, RAG improves factu-
ality and helps language models handle knowledge-
intensive tasks. Most prior work has focused on ques-
tion answering, open-domain generation, or factual
reasoning, where retrieved passages are used as con-
text at inference time. In contrast, our work uses
retrieval not only as runtime evidence, but also as a
source of weak supervision for pretraining.
Chain-of-Thought (CoT) prompting and supervi-
sion have further expanded the reasoning capabili-
ties of LLMs [37]. By eliciting intermediate reasoning
steps before producing a final answer, CoT often im-
proves performance on tasks requiring multi-step in-
ference. Recent studies have explored CoT-style rea-
soning in ranking and retrieval settings, prompting
models to generate rationales before outputting rele-
vance labels [38]. Other work has shown that struc-
tured or knowledge-driven CoT can improve faithful-
ness by grounding intermediate reasoning in external
evidence.
Despite these advances, most existing approaches
either use retrieved evidence as auxiliary context or
use CoT as a generic prompting strategy.RAG-
Matchdiffers in two important respects. First,
in the pretraining stage, we use retrieved evidence
documents to construct a RAG-synthesized super-
vision target and train the model with a query-
to-synthesized-document objective, thereby injecting
query-centered background knowledge into the model
parameters. Second, in the reasoning stage, we orga-
nize CoT supervision hierarchically: the model first
grounds the query with retrieved evidence and then
performs multi-dimensional document analysis under
4

that grounded frame. This explicit decomposition
is designed for relevance judgment, where query in-
tent, latent constraints, and partial relevance must
be jointly considered.
2.4 Multi-Stage Training and Align-
ment for Generative Relevance
Modeling
Recent work has moved beyond zero-shot LLM rank-
ing and begun to optimize large models as dedicated
relevance engines through multi-stage training and
post-hoc alignment. This line of research is par-
ticularly relevant to industrial search, where rele-
vance judgment often involves hard cases, implicit
attributes, and ambiguous decision boundaries.
LREF [39] is one of the representative frameworks
in this direction. It combines high-quality data se-
lection, multi-perspective CoT tuning, and DPO-
based de-biasing for e-commerce relevance predic-
tion. LREF shows that reasoning supervision and
post-training alignment can substantially improve
LLM-based relevance prediction. However, its em-
phasis is primarily on improving reasoning trajecto-
ries and reducing bias after supervised adaptation.
By contrast,RAG-Matchintroduces an additional
retrieval-augmented pretraining phase before reason-
ing supervision. In particular, our Phase I trains the
model to generate a RAG-synthesized document that
best answers the query, allowing the model to in-
ternalize query-centered background knowledge be-
fore downstream reasoning alignment. This makes
RAG-Matchexplicitly knowledge-grounded at the
pretraining level, rather than relying only on reason-
ing supervision at later stages.
ADORE [40] further extends multi-stage rele-
vance modeling by combining rule-aware relevance
discrimination, error-type-aware hard sample syn-
thesis, and key-attribute-enhanced knowledge dis-
tillation. Its design highlights the importance of
domain-specific hard cases and explicit attribute
grounding. Compared with ADORE,RAG-Match
is less centered on handcrafted rule modeling and
synthetic hard-sample construction, and instead fo-
cuses on two complementary mechanisms: retrieval-augmented knowledge injection through query-to-
synthesized-document pretraining, and hierarchical
reasoning decomposition through query grounding
followed by multi-dimensional document analysis.
In other words, while ADORE emphasizes rule-
aware discrimination and hard-case synthesis,RAG-
Matchemphasizes compact knowledge grounding
and structured reasoning under a query-centered se-
mantic frame.
LORE [41] provides a more systematic perspective
by arguing that relevance should be decomposed into
several core capabilities, including knowledge and
reasoning, multimodal matching, and rule adherence.
It accordingly proposes a capability-oriented train-
ing and evaluation framework. This view is highly
aligned with our motivation that relevance judgment
is not a single-step matching problem. However,
RAG-Matchdiffers from LORE in two key aspects.
First, our framework operationalizes knowledge ca-
pability through an explicit retrieval-augmented pre-
training objective, rather than treating knowledge as
one capability dimension in a broader taxonomy. Sec-
ond, our reasoning process is explicitly hierarchical:
the model first constructs a grounded semantic un-
derstanding of the query from retrieved evidence and
then evaluates the candidate document under that
semantic frame. This query-first reasoning protocol
is central to our design and is more tightly coupled
to graded relevance judgment.
TaoSR1 [42] pushes this line of work toward on-
line deployment by combining CoT-based supervised
fine-tuning, pass@N-based preference optimization,
and GRPO-based refinement to improve both rea-
soning quality and final relevance decisions. Sim-
ilarly, reinforcement-learning-based generative rele-
vance modeling has also been explored in open-
domain and industrial search settings, where rele-
vance assessment is optimized as a multi-step rea-
soning process with process-aware supervision [43].
These approaches demonstrate the value of post-
training optimization for relevance alignment. Com-
pared with them,RAG-Matchplaces greater em-
phasis on the role of retrieval-augmented seman-
tic grounding before post-training alignment. Our
framework does not start from reasoning optimiza-
tion alone; instead, it first builds a compact query-
5

centered knowledge prior through RAG-synthesized
document generation, and then uses hierarchical rea-
soning supervision and discrepancy-guided preference
optimization to refine the final relevance boundary.
Overall, prior work has explored reasoning dis-
tillation, rule-aware relevance modeling, capability-
oriented training, and preference-based calibration
for LLM-based ranking.RAG-Matchshares the
multi-stage spirit of these methods, but differs
in its combination of three design choices: (1)
retrieval-augmented pretraining through query-to-
synthesized-document generation, (2) hierarchical
query-first reasoning alignment, and (3) discrepancy-
guided calibration on adjacent label confusions. We
believe this combination is particularly well suited for
semantic relevance judgment in knowledge-intensive
search, where external evidence, structured reason-
ing, and hard-boundary calibration are all essential.
3 Method
In this section, we presentRAG-Match, a multi-
stage framework for semantic relevance judgment in
knowledge-intensive search. The framework consists
of three phases: (1) retrieval-augmented pretraining
for query-centered semantic grounding, (2) hierarchi-
cal reasoning alignment for structured relevance pre-
diction, and (3) discrepancy-guided preference opti-
mization for hard-boundary calibration. Figure 1 il-
lustrates the overall pipeline, including RAG-based
synthesized document construction, hierarchical rea-
soning annotation, and the three-stage training pro-
cess.
3.1 Problem Definition
We study query-document relevance judgment with
retrieval-augmented evidence. Given a queryq∈ Q,
a candidate documentd∈ D, and a set of retrieved
evidence documentsE={e 1, . . . , e m}obtained from
a knowledge baseB, the goal is to predict an ordinal
relevance labely∈ {0,1,2,3}, where 3, 2, 1, and 0
denoteExcellent,Good,Fair, andIrrelevant, respec-
tively.
Rather than treating relevance prediction as adirect classification problem,RAG-Matchformu-
lates it as a structured generative reasoning pro-
cess. Specifically, the model first generates a query-
grounding rationalec 1conditioned on the retrieved
evidence, and then generates a document-level ratio-
nalec 2together with the final labelyconditioned on
the candidate document and the grounding rationale.
The joint probability is factorized as
pθ(c1, c2, y|q, d,E) =p θ(c1|q,E)·p θ(c2, y|q, d, c 1).
(1)
During inference, the relevance score is derived
from the generated labely, whilec 1andc 2serve as
interpretable rationales for the final decision.
3.2 Retrieval-Augmented Knowledge
Construction
Retrieved knowledge plays two roles inRAG-
Match: it provides weak supervision for pretrain-
ing and serves as grounding evidence during reason-
ing. To obtain such knowledge, we build a retrieval
module over a domain knowledge baseB, which con-
sists of structured or semi-structured text fragments
extracted from the search corpus and auxiliary re-
sources.
For each queryq, the retriever returns a set of top-
kevidence documents:
E= Top-k(R(q,B)),(2)
whereRdenotes the retrieval function. In our imple-
mentation, the retriever is a bi-encoder trained with
contrastive learning, and approximate nearest neigh-
bor search(e.g., FAISS) is used for efficient retrieval.
The retrieved evidence documents are not treated
as final supervision targets by themselves. Instead,
they are first used by a RAG generator to synthe-
size a compact query-focused document. This RAG-
synthesized document then serves different roles in
the subsequent stages: in Phase I, it is used as the
supervision target for query-to-document pretrain-
ing, while in Phase II, it is further used in KIG to
support query grounding before document-level rele-
vance analysis.
6

Figure 1: Overview ofRAG-Match. For each query, a RAG system retrieves top-kevidence documents
and synthesizes a document for Phase I pretraining. In Phase II, GPT-5.2 generates hierarchical reasoning
supervision through KIG and HMDA. In Phase III, preference pairs generated by the Phase II model are
used for discrepancy-guided preference optimization.
3.3 Knowledge-Augmented Semantic
Grounding via Query-to-Synthe-
sized-Document Pretraining
Literal query-document matching is often insuffi-
cient for complex or underspecified search queries,
especially when the query intent depends on dis-
persed background knowledge rather than surface
lexical overlap. To address this issue, we intro-
duce a knowledge-augmented pretraining stage based
on query-to-synthesized-document generation. In-
stead of directly predicting relevance labels, this
stage trains the model to map a query to a com-
pact document synthesized by a RAG system from
top-kretrieved evidence documents. In this way,
the model learns to internalize query-centered back-
ground knowledge and acquires grounded semanticpriors before entering downstream relevance reason-
ing.
3.3.1 RAG-Synthesized Pretraining Corpus
Construction
We construct a large-scale pretraining corpusD pre
from 10 million search queries sampled from histor-
ical logs. Since these queries do not have manual
relevance annotations, we derive weak supervision
through a retrieval-augmented generation pipeline.
For each queryq i, we first use a retriever to obtain
the top-kmost relevant evidence documents from the
knowledge baseB:
Ei= Top-k(R(q i,B)).(3)
We then feed the queryq itogether with the re-
trieved evidence setE iinto a RAG generator, which
7

synthesizes a single documents ithat best answers
the query:
si=G RAG(qi,Ei).(4)
The resulting synthesized documents iis not nec-
essarily identical to any individual retrieved docu-
ment. Instead, it provides a compact, query-focused,
and knowledge-grounded summary of the evidence
most relevant to the query. We therefore use the pair
(qi, si) as a weakly supervised pretraining instance.
3.3.2 Query-to-Synthesized-Document Pre-
training Objective
To inject query-centered background knowledge
into the model, we design a query-to-synthesized-
document objective. Given a queryq i, the model
is trained to generate the corresponding RAG-
synthesized documents i:
pθ(si|qi) =|si|Y
t=1pθ(st
i|s<t
i, qi).(5)
The pretraining loss is defined as
Lpre(θ) =−E (q,s)∈D pre
|s|X
t=1logp θ(st|s<t, q)
.
(6)
This objective encourages the model to learn the
latent mapping from a query to its most informative
supporting document, rather than merely memoriz-
ing lexical correlations. Since each synthesized target
is constructed from top-kretrieved evidence docu-
ments, the model is exposed to query-relevant enti-
ties, attributes, and factual relations in a compact
form.
Phase I is not designed to directly optimize the fi-
nal ranking metrics. Instead, its role is to improve
the model initialization by reshaping the intermedi-
ate semantic representation used in downstream rele-
vance modeling. The query-to-synthesized-document
objective encourages the model to construct a query-
centered information state that captures implicit in-
tent, relevant background knowledge, and the seman-
tic attributes that a relevant document is expected to
satisfy. Compared with direct relevance labels, thisgenerative objective provides denser and more struc-
tured supervision, because it teaches the model not
only whether a document is relevant, but also what
information a query should evoke.
From this perspective, Phase I can be interpreted
as learning a stronger query representation before dis-
criminative alignment. Letz q=fθ(q) denote the in-
ternal query representation. By requiringz qto sup-
port the generation of a query-focused synthesized
document, the model is encouraged to preserve query
intent, background facts, and semantically salient
content structure inz q. This provides a stronger ba-
sis for downstream relevance predictiony=g(z q, d)
than direct label supervision alone. In other words,
Phase I helps discrimination not by directly learning
the final ranking boundary, but by learning a query-
conditioned semantic scaffold that makes later rele-
vance reasoning easier and more stable.
From an optimization perspective, Phase I also
serves as a task-relevant warm start. Without this
stage, the model in Phase II must simultaneously
infer latent query requirements, organize support-
ing knowledge, and learn decision-oriented reasoning
from relatively sparse supervision. By pretraining
the model to first recover a query-centered semantic
support space, Phase I decouples semantic grounding
from later reasoning alignment, thereby making the
downstream optimization problem easier. This also
explains why the benefit of Phase I is mainly reflected
in improved NDCG after Phase II, while its addi-
tional effect on nPNR is limited: Phase I primarily
strengthens semantic grounding and graded ranking
quality, whereas pairwise ranking consistency is more
directly shaped by the explicit hierarchical supervi-
sion introduced in Phase II.
3.4 Hierarchical Reasoning Alignme-
nt
After retrieval-augmented pretraining, the model has
acquired query-centered semantic priors, but it still
needs to transform these priors into faithful rele-
vance decisions. Directly reasoning over the full tu-
ple (q, d, s) can be suboptimal, because the model
may overfit superficial overlap between the query and
the candidate document before establishing a precise
8

understanding of the query intent. To address this
problem, we introduce Hierarchical Reasoning Align-
ment (HRA), which decomposes relevance judgment
into two stages: query grounding and document-level
alignment.
3.4.1 Stage A: Knowledge-Infused Ground-
ing
In the first stage, the model performs Knowledge-
Infused Grounding (KIG) using only the queryq
and the RAG-synthesized document s. The purpose
of this stage is to construct a grounded semantic
state before the candidate document is introduced.
This design reduces reasoning drift[44] and forces the
model to explicitly organize the latent constraints un-
derlying the query.
We instantiate the grounding process using a
teacher LLM, which maps (q, s) to a structured six-
field representation:
G= [I, T, O, E, C, S],(7)
whereIdenotes the query intent,Tdenotes the ex-
pected answer type,Odenotes the core object range,
Edenotes the minimal answer elements,Cdenotes
critical constraints, andSdenotes a compact synop-
sis of the retrieved knowledge. Formally,
G=f(1)
teach(q, s).(8)
This representation serves as an explicit semantic
frame for the next stage. It summarizes what the
user is asking for, what constraints must be satisfied,
and what background knowledge is relevant to the
judgment.
3.4.2 Stage B: Holistic Multi-Dimensional
Alignment
In the second stage, the model evaluates the candi-
date documentdunder the grounded semantic frame
G. Instead of making a direct holistic judgment, we
decompose relevance into four complementary dimen-
sions:
•Intent Consistency(Φ IC): whether the docu-
ment satisfies the expected answer type and core
intent of the query;•Entity Fidelity(Φ EF): whether the enti-
ties and concepts in the document match the
grounded object range, including aliases and hi-
erarchical relations;
•Logical Completeness(Φ LC): whether the
document covers the minimal answer elements
required by the query;
•Information Density(Φ ID): whether the doc-
ument contains concentrated relevant evidence
rather than noisy or misleading content.
We use a teacher LLM to generate a structured
alignment rationale
A=f(2)
teach(q, d,G),(9)
which summarizes the document analysis across these
dimensions and supports the final relevance label.
This decomposition is designed to capture com-
mon failure modes in relevance judgment, includ-
ing superficial topic overlap, partial answer coverage,
entity mismatch, and noisy content. By enforcing
dimension-wise analysis, HRA improves both deci-
sion quality and interpretability.
3.4.3 Reasoning Distillation via Supervised
Fine-Tuning
To transfer the HRA reasoning protocol to the stu-
dent model, we build a reasoning-annotated dataset
Dcotusing the teacher LLM. Each instance contains
a queryq, a candidate documentd, a grounding ra-
tionaleG, a document-level alignment rationaleA,
and the final relevance labely.
The student model is then fine-tuned to generate
the full reasoning targett= (G,A, y) conditioned on
the inputx= (q, d):
Lsft(θ) =−E (x,t)∈D cotlogp θ(t|x).(10)
By distilling structured reasoning trajectories
rather than labels alone, HRA teaches the model to
perform relevance judgment through explicit query
grounding and document-level analytical reasoning.
9

3.5 Discrepancy-Guided Preference
Optimization
Although HRA equips the model with hierarchi-
cal reasoning supervision, it does not fully resolve
decision-level calibration errors in final relevance pre-
diction. In particular, we observe that the Phase II
model tends to overestimate relevance on ambiguous
or boundary cases, resulting in an upward scoring
tendency in its final labels. To address this issue,
we introduce Phase III, namely discrepancy-guided
preference optimization (DGPO)[45], which further
calibrates the model’s decision boundaries by encour-
aging preferences toward more reliable relevance out-
comes. As shown later in our analysis, this stage
effectively reduces systematic overestimation and im-
proves the robustness of final relevance judgments.
3.5.1 Preference Pair Construction
We construct preference pairs from a small set
of human-annotated samples. For each sample
(q, d, y∗), wherey∗is the gold relevance label, we
treaty∗as the preferred labely wand sample an ad-
jacent incorrect labely lfrom the valid neighborhood
ofy∗. This design focuses the preference learning
process on hard confusions rather than obviously in-
correct alternatives.
Using the Phase-II model, we generate two reason-
ing trajectories conditioned on these labels:
•a preferred trajectory (y w, rw), wherer wis the
rationale consistent with the gold label;
•a dispreferred trajectory (y l, rl), wherer lcorre-
sponds to a plausible but incorrect judgment.
The resulting dataset
Dpair={(q, d, y w, rw, yl, rl)}(11)
contains preference pairs centered on difficult decision
boundaries.
3.5.2 Preference-Based Calibration
We optimize the student policyπ θagainst a refer-
ence policyπ ref. For a reasoning trajectory (y, r),we define
h(y, r) = logπθ(y, r|q, d)
πref(y, r|q, d).(12)
The DPO objective is
Ldpo(θ) =−E Dpair[logσ(βh(y w, rw)−βh(y l, rl))],
(13)
whereσis the sigmoid function andβcontrols the
alignment strength.
Because online deployment primarily depends on
the final relevance label, we further add an auxiliary
label prediction term:
Ldpo+(θ) =L dpo−E (q,d,y w)∈D pairlogπ θ(yw|q, d).
(14)
This final calibration stage encourages the model
to prefer more faithful reasoning trajectories and
sharper decision boundaries, thereby reducing the
systematic overestimation tendency on ambiguous
relevance cases.
4 Experiments
In this section, we evaluate the performance ofRAG-
Matchthrough comparative benchmarks and abla-
tion studies. We show that the proposed combina-
tion of retrieval-augmented pretraining, hierarchical
reasoning alignment, and discrepancy-guided prefer-
ence optimization substantially improves both rank-
ing quality and ranking consistency in knowledge-
intensive relevance judgment.
4.1 Datasets and Metrics
For the pretraining stage, we sampled 10 million
search queries from historical logs. For each query,
we first retrieved the top-kmost relevant evidence
documents using a retriever and then used a RAG
generator to synthesize a document that best an-
swers the query. The resulting query-document pairs
(q, s) were used to train the query-to-synthesized-
document objective in Phase I.
For the training set of the Hierarchical Reasoning
Alignment task, we used a strong teacher LLM to
10

annotate and reason over 25,102 unlabeled samples.
Each sample is a triplet⟨q, d, s⟩consisting of a query
q, a candidate documentd, and a RAG-synthesized
document s. Given this triplet, the teacher model
produces the structured grounding rationaleG, the
document-level alignment rationaleA, and the final
relevance labely.
We use a small set of 3,000 human-annotated sam-
ples to train the Discrepancy-Guided Preference Op-
timization task. For each sample, preference pairs
are constructed around adjacent label confusions, and
the corresponding reasoning trajectories are gener-
ated by the Phase II model.
The label distribution is detailed in Table 1, show-
ing a representative mix of typical search scenarios.
To evaluate the model’s generalization in real-
world scenarios, we construct a manually annotated
test set of 1,728 query-document pairs collected from
search logs. The test set covers 300 queries and is
designed to include both randomly sampled queries
and medium-/long-tail search queries, so as to reflect
realistic and challenging search conditions, including
ambiguous user intents. We note that the test set
size is limited by the high cost and relatively low ef-
ficiency of professional manual annotation, especially
for fine-grained relevance judgment. Therefore, we
focus on building a carefully curated evaluation set
with high annotation quality rather than a larger but
noisier benchmark. All query-document pairs in the
test set were annotated by professional judges to pro-
vide a gold standard for evaluation.
We adopt two categories of evaluation metrics:
•Ranking Quality:We report NDCG@K (K∈
{1,3,5,10}) to assess the quality of top-ranked
results [46].
•Ranking Consistency (nPNR):Following
prior work, we also consider pairwise ranking
consistency as an auxiliary evaluation signal.
Instead of directly reporting the raw Positive-
Negative Ratio (PNR), which is sensitive to the
number of positive and negative document pairs
and lacks a standardized range, we report its nor-
malized form:
nPNR =PNR
1 + PNR=N+
N++N−,(15)whereN +andN −denote the numbers of cor-
rectly and incorrectly ordered document pairs,
respectively. This normalization maps the met-
ric into the range (0,1), making it easier to inter-
pret and compare across models. A higher nPNR
indicates better pairwise ranking consistency.
4.2 Baselines
To ensure a rigorous and fair evaluation, we com-
pareRAG-Matchagainst several strong large lan-
guage model baselines of comparable scale. All base-
line models are fine-tuned on the same labeled train-
ing data using a standard label-only supervised fine-
tuning objective before evaluation.
•DeepSeek-R1-0528 (8B)[47]: a distilled rea-
soning model optimized for complex logical in-
ference.
•GLM-4-9B-0414 (9B)[48]: a dense bilingual
model with strong semantic understanding and
long-context modeling ability.
•MiniMax-SynLogic (8B)[49]: a model spe-
cialized in structural logic and symbolic reason-
ing for text matching.
•Qwen3-8B-Reranker (8B)[50]: our backbone
model trained with label-only supervised fine-
tuning, serving as the main base model compar-
ison.
4.3 Implementation Details
We employ Qwen3-Reranker-8B as the backbone
model forRAG-Match. Phase I (knowledge-
augmented pretraining) is conducted on 10M queries.
For each query, we retrieve top-kevidence docu-
ments and use a RAG generator to synthesize a
query-focused supervision document. The backbone
model is then trained with the query-to-synthesized-
document objective using the AdamW optimizer with
a learning rate of 1.5×10−5.
Phase II (Hierarchical Reasoning Alignment) uses
GPT-5.2 as the teacher LLM to construct struc-
tured reasoning trajectories. Specifically, given a
11

Table 1: Label distribution of the training and test sets. The relevance labels are defined as 3 = Excellent,
2 = Good, 1 = Fair, and 0 = Irrelevant.
Split Excellent (3) Good (2) Fair (1) Irrelevant (0)
HRA Training Set 27.09% 34.57% 24.20% 14.14%
DGPO Training Set 18.76% 37.84% 22.74% 20.67%
Test Set 13.60% 32.18% 23.67% 30.56%
triplet⟨q, d, s⟩, GPT-5.2 generates the grounding ra-
tionaleG, the document-level alignment rationaleA,
and the final relevance labely, which together form
the reasoning-annotated datasetD cot. The student
model is then fine-tuned onD cotwith a maximum
sequence length of 2048. We choose GPT-5.2 as the
teacher model in Phase II because of its strong rea-
soning capability and high-quality CoT generation.
In our manual inspection, GPT-5.2 achieves 93.3%
accuracy on CoT generation, substantially higher
than its relevance label accuracy of 76.6%, suggest-
ing that it is particularly suitable for providing struc-
tured reasoning supervision.
Phase III (Discrepancy-Guided Preference Opti-
mization) is built on top of the Phase II student
model. Specifically, for each human-annotated sam-
ple, we construct a preferred label and an adjacent
dispreferred label, and then use the Phase II model
to generate the corresponding reasoning trajecto-
ries. These preferred–dispreferred trajectory pairs
are used for DPO-based calibration withβ= 0.1,
allowing the model to further refine subtle relevance
boundaries while remaining consistent with the rea-
soning protocol learned in Phase II.
Training is performed on an 8×NVIDIA A800
cluster.
4.4 Main Results
Table 2 presents the main comparison results.RAG-
Matchconsistently outperforms all baseline models
across all reported NDCG positions and nPNR. In
particular,RAG-Matchachieves the best perfor-
mance in both top-rank quality and ranking consis-
tency, indicating that the proposed framework effec-
tively bridges the gap between retrieved knowledge,structured reasoning, and final relevance judgment.
Notably,RAG-Matchbrings clear improvements
on both NDCG and nPNR, indicating that the pro-
posed framework improves not only top-ranked re-
trieval quality but also pairwise ranking consistency.
This suggests that the model is better able to capture
fine-grained relevance distinctions and maintain more
reliable decision boundaries in hard and ambiguous
search scenarios.
4.5 Ablation Study
Table 3 shows the macro-level ablation results of dif-
ferent components inRAG-Match. Overall, the
full model achieves the best performance across all
ranking metrics, indicating that the three stages con-
tribute complementary improvements.
Compared with the base model (Row 1), directly
appending the RAG-synthesized document at infer-
ence time (Row 2) substantially improves all metrics,
confirming that retrieved external knowledge is use-
ful for relevance estimation. However, this inference-
time RAG baseline still underperforms the label-only
SFT model (Row 3) and remains clearly weaker than
all multi-stage variants (Rows 4–6). This suggests
that simple RAG-style prompting provides only tran-
sient in-context evidence and cannot replace task-
specific training.
Starting from the label-only SFT baseline, intro-
ducing Phase II (Row 4) brings consistent gains on
both NDCG and nPNR, demonstrating the effec-
tiveness of hierarchical reasoning alignment. Adding
Phase I before Phase II (Row 5) further improves
all NDCG metrics, while nPNR remains nearly un-
changed, indicating that Phase I mainly strength-
ens query-centered semantic grounding and improves
12

Table 2: Main results on the search relevance benchmark. All baseline models are fine-tuned on the same
labeled training data using label-only supervised fine-tuning. The best results are in bold.
Model NDCG@1↑NDCG@3↑NDCG@5↑NDCG@10↑nPNR↑
DeepSeek-R1-0528 (8B) 0.815 0.867 0.883 0.906 0.783
GLM-4-9B-0414 (9B) 0.846 0.877 0.891 0.915 0.801
MiniMax-SynLogic (8B) 0.837 0.87 0.891 0.908 0.792
Qwen3-8B-Reranker (8B) 0.826 0.878 0.892 0.912 0.796
RAG-Match(8B)0.902 0.907 0.916 0.935 0.833
overall ranking quality rather than directly enhancing
pairwise consistency. Finally, incorporating Phase III
(Row 6) yields the best overall results, showing that
discrepancy-guided preference optimization further
improves difficult relevance judgments through better
decision calibration.
Taken together, these results show that the gain of
RAG-Matchdoes not come merely from attaching
extra retrieved context at inference time. Instead,
it comes from progressively internalizing retrieval-
enhanced knowledge, aligning the model with struc-
tured relevance reasoning, and calibrating final rele-
vance decisions.
4.5.1 Effect of Phase I on Phase II Alignment
We do not report standalone results for Phase I, be-
cause its objective is not to directly optimize the final
ranking metrics. Instead, Phase I is designed to pro-
vide a knowledge-augmented initialization through
the query-to-synthesized-document objective, which
is expected to facilitate the subsequent hierarchical
reasoning alignment in Phase II. Therefore, we as-
sess its contribution by comparing a Phase II-only
model with a model trained using Phase I followed
by Phase II.
As shown in Table 3, incorporating Phase I leads to
further improvements in NDCG after Phase II train-
ing, while the nPNR results remain largely compa-
rable. This suggests that the synthesized-document
pretraining stage mainly benefits the model’s overall
ranking quality by strengthening query-centered se-
mantic grounding and enriching background knowl-
edge before reasoning alignment. A plausible expla-nation for the relatively stable nPNR is that pairwise
ranking consistency is more directly shaped by the ex-
plicit hierarchical reasoning supervision introduced in
Phase II. In addition, Phase I may primarily refine
graded relevance estimation and the relative ordering
of top-ranked documents, which is more likely to im-
prove NDCG than to substantially change the binary
pairwise outcomes reflected by nPNR.
4.5.2 Analysis of Phase II: Hierarchical Rea-
soning Alignment
Phase II is a major source of performance gain in the
overall framework. Compared with the label-only su-
pervised fine-tuning baseline, enabling HRA substan-
tially improves both NDCG and nPNR, showing that
structured reasoning alignment plays a critical role
in relevance prediction quality. At the same time,
the later Phase III stage brings gains of a compara-
ble scale on NDCG and even larger improvements on
nPNR, suggesting that reasoning alignment and de-
cision calibration contribute in complementary ways.
These results indicate that explicit reasoning super-
vision is much more effective than direct label fitting
in complex search scenarios, where the model must
verify intent satisfaction, entity scope, and answer
completeness rather than rely on shallow semantic
overlap.
We further analyze the internal structure of
Phase II in Table 4. Adding Stage A to Stage B
consistently improves both ranking quality and rank-
ing consistency, indicating that query grounding pro-
vides an essential semantic frame for subsequent doc-
ument analysis. Without Stage A, the model tends to
13

Table 3: Macro ablation results ofRAG-Match. Phase I denotes query-to-synthesized-document pre-
training, Phase II denotes Hierarchical Reasoning Alignment, and Phase III denotes Discrepancy-Guided
Preference Optimization. For fairness, we additionally evaluate an inference-time RAG baseline that ap-
pends the RAG-synthesized document during prediction only, without applying the proposed multi-stage
training pipeline.
Row Phase I Phase II Phase III Infer-time RAG NDCG@1↑NDCG@3↑NDCG@5↑NDCG@10↑nPNR↑
1× × × × 0.668 0.744 0.784 0.827 0.618
2× × ×✓ 0.764 0.842 0.866 0.892 0.736
3×Label-only SFT× × 0.826 0.878 0.892 0.912 0.796
4×✓× × 0.853 0.891 0.901 0.914 0.810
5✓ ✓× × 0.882 0.895 0.908 0.925 0.809
6✓ ✓ ✓× 0.902 0.907 0.916 0.935 0.833
align directly against the candidate document with-
out first establishing a precise understanding of the
query, which increases the risk of reasoning drift and
superficial matching.
4.5.3 Analysis of Phase III: Discrepancy-
Guided Preference Optimization
Phase III (DGPO) serves as the final calibration
layer. Comparing the full framework with its su-
pervised fine-tuning predecessor shows that DGPO
further suppresses the systematic overestimation ten-
dency inherent in LLM-based relevance predictors.
By maximizing the preference margin between the
faithful trajectory and the plausible but incorrect
trajectory, DGPO improves consistency on hard-
boundary cases and leads to the best overall rank-
ing performance. This confirms that discrepancy-
guided preference learning is particularly effective for
resolving subtle label confusions, such asFairvs.
Good, which are common in real-world search rele-
vance judgment.
The motivation for introducing DGPO in Phase III
is also supported by our manual analysis of the
Phase II teacher annotations. Specifically, GPT-5.2
achieves high accuracy in generating CoT rationales
(93.3%), but its relevance label accuracy is noticeably
lower (76.6%). Moreover, after Phase II training, the
student model attains a CoT accuracy that is largely
comparable to that of the teacher, suggesting that
the reasoning patterns are effectively distilled. How-
ever, our case analysis reveals that some examples
contain correct CoT explanations but incorrect finallabels. This indicates that even high-quality reason-
ing supervision does not fully eliminate label noise
or decision-boundary errors. DGPO is therefore in-
troduced to further correct such discrepancies by op-
timizing preference signals over competing reasoning
outcomes, thereby improving the robustness of final
relevance decisions.
Beyond the gains on the main ranking metrics,
Phase III also improves the calibration of final rel-
evance decisions. We further quantify whether the
Phase II model exhibits an upward scoring tendency
in final relevance prediction. Specifically, given the
predicted label ˆyand the gold labely, we compute
three statistics: (1)OverScore Rate, the propor-
tion of cases with ˆy > y; (2)UnderScore Rate, the
proportion of cases with ˆy < y; and (3)Mean Score
Bias, defined as the average of ˆy−yover the evalu-
ation set. As shown in Table 5, the Phase II model
exhibits a pronounced upward scoring tendency, with
an OverScore Rate of 0.5023, an UnderScore Rate
of 0.0683, and a Mean Score Bias of 0.5602. This
indicates that after supervised reasoning alignment,
the model still tends to assign overly high relevance
labels, especially on ambiguous or hard-boundary
cases.
After applying DGPO in Phase III, the error pro-
file becomes substantially more balanced. The Over-
Score Rate drops from 0.5023 to 0.3356, and the
Mean Score Bias decreases from 0.5602 to 0.2378,
showing that Phase III effectively suppresses system-
atic overestimation of relevance. Meanwhile, the Un-
derScore Rate increases from 0.0683 to 0.1423, sug-
14

Table 4: Ablation study on the internal structure of Phase II. Stage A denotes Knowledge-Infused Grounding
(KIG), and Stage B denotes Holistic Multi-Dimensional Alignment (HMDA).
Setting Stage A Stage B NDCG@1↑NDCG@3↑NDCG@5↑NDCG@10↑nPNR↑
Direct reasoning baseline× × 0.826 0.878 0.892 0.912 0.796
Stage B only×✓ 0.840 0.867 0.885 0.907 0.799
Stage A + Stage B✓ ✓ 0.853 0.891 0.901 0.914 0.810
Table 5: Quantification of the upward scoring tendency for the Phase II model and the model after Phase III.
Phase III substantially reduces overestimation errors, as reflected by lower OverScore Rate and Mean Score
Bias.
Model OverScore Rate UnderScore Rate Mean Score Bias
Phase I + Phase II 0.5023 0.0683 0.5602
Phase I + Phase II + Phase III0.33560.14230.2378
gesting that the model becomes more conservative in
a subset of cases. Nevertheless, this trade-off is ben-
eficial overall, as it reduces the strong one-sided up-
ward bias of the Phase II model and leads to better-
calibrated final relevance decisions. These results
suggest that the gain of DGPO comes not only from
generic preference learning, but also from its ability
to correct systematic decision-level deviations that
remain after supervised reasoning alignment.
5 Discussion
Our results suggest that knowledge-intensive rele-
vance modeling benefits from being decomposed into
three complementary stages: query-centered ground-
ing, reasoning alignment, and decision calibration.
Rather than treating relevance judgment as a di-
rect label prediction problem,RAG-Matchim-
proves it progressively by first strengthening semantic
grounding, then aligning the model with structured
relevance reasoning, and finally correcting residual
decision-level deviations. This staged design helps
explain why the full framework consistently improves
ranking performance on challenging search scenarios
that require background knowledge, implicit intent
understanding, and fine-grained relevance discrimi-
nation.
A first important observation concerns the role ofPhase I. We do not interpret Phase I as a standalone
ranking stage, since its objective is not to directly op-
timize the final ranking metrics. Instead, its value lies
in providing a knowledge-augmented initialization for
downstream reasoning alignment. More specifically,
the query-to-synthesized-document objective encour-
ages the model to build a query-centered semantic
scaffold that captures implicit intent, relevant back-
ground knowledge, and the semantic content struc-
ture that a relevant document is expected to satisfy.
Compared with direct label supervision, this genera-
tive objective offers denser and more structured train-
ing signals, because it teaches the model not only
whether a document is relevant, but also what infor-
mation a query should evoke.
This perspective helps explain the empirical pat-
tern observed in our experiments. Adding Phase I
before Phase II leads to further improvements in
NDCG, while the nPNR results remain largely com-
parable to those of Phase II alone. This suggests
that Phase I mainly improves overall ranking quality
by strengthening semantic grounding and enriching
query-centered background knowledge before reason-
ing alignment. In contrast, pairwise ranking consis-
tency appears to be more directly shaped by the ex-
plicit hierarchical reasoning supervision introduced in
Phase II, which may explain why Phase I brings only
limited additional gains on nPNR. In this sense, the
15

main contribution of Phase I is not direct decision-
boundary learning, but the construction of a more
informative semantic starting point for downstream
discriminative relevance modeling.
A second observation concerns the role and limi-
tation of the teacher model in Phase II. Our manual
inspection shows that GPT-5.2 produces highly reli-
able reasoning trajectories, achieving a CoT accuracy
of 93.3%, while its relevance label accuracy is lower
at 76.6%. This suggests that the teacher is partic-
ularly effective for structured reasoning supervision,
but its final label decisions still contain non-negligible
noise. Consistent with this observation, we find that
after Phase II training, the student model reaches
a CoT accuracy that is largely comparable to that
of GPT-5.2, indicating that the reasoning capability
is effectively distilled. However, case analysis reveals
that some examples contain correct CoT explanations
but incorrect final labels. This mismatch implies that
reasoning quality and decision quality are not always
perfectly aligned, and also reveals a potential source
of error propagation in supervised reasoning align-
ment.
This issue becomes clearer when we examine the er-
ror distribution of the Phase II model. Our additional
analysis shows that Phase II exhibits a pronounced
upward scoring tendency in final relevance prediction,
with an OverScore Rate of 0.5023, an UnderScore
Rate of 0.0683, and a Mean Score Bias of 0.5602.
In other words, the model is much more likely to
overestimate relevance than to underestimate it, es-
pecially on ambiguous or hard-boundary cases. This
provides direct empirical support for the motivation
of Phase III. By introducing discrepancy-guided pref-
erence optimization, we encourage the model to pre-
fer more reliable relevance outcomes when competing
predictions reveal boundary-level inconsistencies. As
a result, Phase III substantially reduces the Over-
Score Rate to 0.3356 and the Mean Score Bias to
0.2378, while moderately increasing the UnderScore
Rate to 0.1423. Although this makes the model some-
what more conservative in a subset of cases, it effec-
tively mitigates the strong one-sided overestimation
tendency inherited from Phase II and leads to better-
calibrated final relevance judgments.
Taken together, these findings suggest that thethree phases ofRAG-Matchplay distinct but
complementary roles. Phase I improves semantic
grounding and benefits downstream ranking quality;
Phase II contributes the core reasoning capability
through hierarchical supervision; and Phase III serves
as a calibration stage that corrects decision-level in-
consistencies that cannot be fully resolved by super-
vised reasoning alignment alone. This division of la-
bor helps explain why the full framework achieves
stronger and more stable performance across both
ranking quality and relevance decision consistency.
At the same time, our study also highlights several
limitations. First, the effectiveness of the framework
still depends on the quality of retrieved evidence, and
noisy or incomplete retrieval may affect both ground-
ing and reasoning. Second, the current pipeline relies
on multi-stage training and teacher-generated super-
vision, which increases annotation and optimization
cost compared with simpler single-stage approaches.
Third, although DGPO improves decision calibra-
tion, some difficult cases may still contain residual
uncertainty, especially when retrieval evidence is in-
complete or when the relevance boundary itself is
highly ambiguous. These limitations point to the
need for more robust retrieval-aware modeling and
more efficient alignment strategies in future work.
6 Conclusion
In this paper, we proposedRAG-Match, a three-
stage framework for knowledge-intensive relevance
modeling that integrates knowledge-augmented pre-
training, hierarchical reasoning alignment, and
discrepancy-guided preference optimization. The
central idea is that accurate relevance judgment
should be learned progressively: first by strengthen-
ing query-centered grounding, then by aligning the
model with explicit multi-step relevance reasoning,
and finally by calibrating difficult boundary deci-
sions. Experimental results on a real-world search
relevance benchmark show thatRAG-Matchcon-
sistently outperforms strong LLM-based baselines
across multiple ranking metrics, demonstrating the
value of combining knowledge injection, structured
reasoning supervision, and preference-based calibra-
16

tion for fine-grained relevance estimation.
Our analysis further shows that the three stages
contribute in different ways. Phase I mainly improves
downstream ranking quality by providing stronger
semantic grounding for Phase II, rather than serv-
ing as an independently optimized ranking stage.
Phase II effectively distills high-quality reasoning
patterns from the teacher model, but still inherits
some label-level noise and systematic overestimation
in final relevance decisions. Phase III addresses this
issue by reducing the upward scoring tendency of the
Phase II model and producing better-calibrated rele-
vance judgments.
Despite these encouraging results, several limita-
tions remain. The framework is still sensitive to
retrieval quality, and noisy or incomplete evidence
may weaken downstream grounding and reasoning.
In addition, the multi-stage pipeline introduces ex-
tra annotation, training, and inference complexity,
which may limit efficiency in large-scale or latency-
sensitive deployment settings. Future work may ex-
plore more robust handling of retrieval noise, for ex-
ample through evidence filtering, retrieval confidence
modeling, or tighter joint optimization between re-
trieval and relevance estimation. Another important
direction is to reduce the latency and complexity of
multi-stage relevance modeling, such as by develop-
ing more efficient alignment strategies or more unified
end-to-end training schemes. We hope this work can
provide a useful step toward more accurate, robust,
and interpretable relevance modeling for knowledge-
intensive search.
A Prompt Template for Phase
II Reasoning Annotation
This appendix provides the prompt template used
in Phase II, i.e., Hierarchical Reasoning Alignment
(HRA), to construct teacher-generated reasoning tra-
jectories. As described in Section Method, Phase II
consists of two stages: Knowledge-Infused Ground-
ing (KIG) and Holistic Multi-Dimensional Alignment
(HMDA). We use GPT-5.2 as the teacher LLM to
generate structured rationales and relevance labelsfor triplets of the form⟨q, d, s⟩, whereqis the query,
dis the candidate document, and s denotes the RAG
synthesized document.
To improve readability and compatibility with our
annotation pipeline, the original prompt was written
in Chinese. Below we provide its English version,
translated as faithfully as possible while aligning ter-
minology with the main paper.
A.1 English Prompt Schema Used for
Phase II
Role / Instruction.You are an expert in seman-
tic understanding and relevance analysis. Given a
Query,Retrieved Evidence, and aCandidate
Document, determine the relevance score of the can-
didate document with respect to the query and ex-
plain the scoring logic in detail.
Important Constraints.
•Avoid explicitly mentioning terms such as “re-
trieved evidence,” “reference knowledge,” or
“RAG” in the reasoning whenever possible.
•Judge relevance by the extent to which the docu-
ment satisfies the information need of the query.
Retrieved evidence is only used for factual sup-
plementation and verification rather than for lit-
eral similarity comparison.
•Avoid overly strict entity matching. If the docu-
ment addresses the same category of core object,
the same event chain, or the same thematic set
as the query, reasonable partial alignment should
still receive partial credit.
•If the query does not explicitly specify year, ver-
sion, region, educational stage, or similar con-
straints, do not penalize a document merely be-
cause other versions may exist, unless there is a
clear contradiction or conflict.
•Downgrade the score if the document contains
off-topic extensions, insufficient coverage, or re-
quires substantial reader-side filtering or correc-
tion even when some core information is correct.
Step A: Knowledge-Infused Grounding.Ex-
tract from the retrieved evidence the information that
is directly relevant to the query and useful for rel-
evance judgment. Summarize it concisely and ab-
17

stractly rather than copying passages or listing ex-
cessive details. The extracted information should in-
clude:
•query intent and task type,
•expected answer type,
•core object and scope,
•minimal answer elements,
•key constraints and common pitfalls,
•core knowledge points that directly support an-
swering the query.
Step B: Holistic Multi-Dimensional Align-
ment.Evaluate the candidate document based on
the results of Step A. Focus on semantic alignment
and degree of information satisfaction rather than me-
chanical matching. The evaluation should consider:
•intent alignment,
•entity and key information matching,
•knowledge completeness and logical consistency,
•proportion of relevant content.
Scoring Criteria.
•3 (Fully Satisfied):Covers the minimal an-
swer elements, aligns well with the query intent
and object scope, and most of the document is
directly useful for answering the query.
•2 (Mostly Satisfied):Covers most key ele-
ments and answers the core question, but may
have minor omissions, slight formulation devia-
tions, or limited extension to closely related sec-
ondary objects.
•1 (Partially Satisfied):Only touches periph-
eral aspects, or is related to the topic but misses
many key conclusions or essential points.
•0 (Not Satisfied):Fundamentally irrelevant
to the query, or contains conflicting information
that prevents support for the query.
Input Variables.
•Query:{query}
•Retrieved Evidence:{rag doc}
•Candidate Document:{"title":{title},
"content":{content}}Output Format.Return strict JSON with the fol-
lowing schema:
{"score": "relevance score (0-3)",
"cot": "First provide the Step A knowledge
extraction in bullet points, then provide the
Step B alignment analysis and final scoring
rationale in bullet points"
}
A.2 Relation to the Main Framework
The schema above directly corresponds to the
Phase II design described in the main paper. Specif-
ically, Step A operationalizes Knowledge-Infused
Grounding (KIG) by extracting query intent, ex-
pected answer type, core object range, minimal an-
swer elements, key constraints, and core knowledge
points from the retrieved evidence. Step B op-
erationalizes Holistic Multi-Dimensional Alignment
(HMDA) by evaluating the candidate document with
respect to intent alignment, entity-level matching,
knowledge completeness, and the proportion of rel-
evant content. The resulting teacher-generated out-
puts are then used to construct the reasoning-
annotated dataset for supervised fine-tuning in
Phase II.
References
[1] C. D. Manning, P. Raghavan, and H. Sch¨ utze,
Introduction to Information Retrieval. Cam-
bridge, UK: Cambridge University Press, 2008.
[2] J. M. Ponte and W. B. Croft, “A language
modeling approach to information retrieval,” in
Proceedings of the 21st Annual International
ACM SIGIR Conference on Research and De-
velopment in Information Retrieval (SIGIR ’98).
ACM, 1998, pp. 275–281.
[3] W. tau Yih, M.-W. Chang, C. Meek, and
J. Pasternack, “The value of semantic parsing for
the QA task,” inProceedings of the Tenth Inter-
national Conference on Language Resources and
Evaluation (LREC 2016). European Language
Resources Association (ELRA), 2016.
18

[4] S. Humeau, K. Shuster, M. Ranzato, and
J. Weston, “Poly-encoders: Architectures
for real-time strategy game state encod-
ing and multi-segment matching,” inIn-
ternational Conference on Learning Repre-
sentations (ICLR), 2020. [Online]. Available:
https://openreview.net/forum?id=SkxS8yS6SF
[5] S. Robertson and H. Zaragoza,The probabilistic
relevance framework: BM25 and beyond. Now
Publishers Inc, 2009, vol. 4.
[6] H. Sch¨ utze, C. D. Manning, and P. Ragha-
van,Introduction to information retrieval.
Cambridge University Press Cambridge, 2008,
vol. 39.
[7] J. Devlin, M.-W. Chang, K. Lee, and
K. Toutanova, “Bert: Pre-training of
deep bidirectional transformers for language
understanding,” 2019. [Online]. Available:
https://arxiv.org/abs/1810.04805
[8] R. Nogueira and K. Cho, “Passage re-ranking
with bert,” 2020. [Online]. Available: https:
//arxiv.org/abs/1901.04085
[9] W. Sun, L. Yan, and X. e. a. Ma, “Is Chat-
GPT good at search? investigating large lan-
guage models as re-ranking agents,” inProceed-
ings of the 2023 Conference on Empirical Meth-
ods in Natural Language Processing. Singapore:
Association for Computational Linguistics, Dec.
2023, pp. 14 918–14 937.
[10] L. Bonifacio, H. Abonizio, M. Fadaee, and
R. Nogueira, “Inpars: Data augmentation
for information retrieval using large language
models,” 2022. [Online]. Available: https:
//arxiv.org/abs/2202.05144
[11] Q. Lyu, S. Havaldar, A. Stein, L. Zhang, D. Rao,
E. Wong, M. Apidianaki, and C. Callison-Burch,
“Faithful chain-of-thought reasoning,” inPro-
ceedings of the 13th International Joint Confer-
ence on Natural Language Processing and the
3rd Conference of the Asia-Pacific Chapter of
the Association for Computational Linguistics
(Volume 1: Long Papers), 2023, pp. 305–329.[12] Z. Ji, N. Lee, R. Frieske, T. Yu, D. Su, Y. Xu,
E. Ishii, Y. J. Bang, A. Madotto, and P. Fung,
“Survey of hallucination in natural language
generation,”ACM computing surveys, vol. 55,
no. 12, pp. 1–38, 2023.
[13] P. Wang, L. Li, L. Chen, Z. Cai, D. Zhu, B. Lin,
Y. Cao, L. Kong, Q. Liu, T. Liu, and Z. Sui,
“Large language models are not fair evaluators,”
inProceedings of the 62nd Annual Meeting of
the Association for Computational Linguistics
(Volume 1: Long Papers). Association for
Computational Linguistics, 2024, pp. 9440–
9450. [Online]. Available: https://aclanthology.
org/2024.acl-long.511/
[14] S. E. Robertson, S. Walker, H.-B. Micheline,
M. Gatford, and A. Payne, “Okapi at TREC-
3,” inProceedings of the Third Text REtrieval
Conference (TREC-3), 1995.
[15] S. Robertson and H. Zaragoza, “The proba-
bilistic relevance framework: BM25 and be-
yond,”Foundations and Trends in Information
Retrieval, vol. 3, no. 4, pp. 333–389, 2009.
[16] P.-S. Huang, X. He, J. Gao, L. Deng, A. Acero,
and L. Heck, “Learning deep structured seman-
tic models for web search using clickthrough
data,” inProceedings of the 22nd ACM interna-
tional conference on Information and Knowledge
Management (CIKM), 2013, pp. 2333–2338.
[17] Y. Shen, X. He, J. Gao, L. Deng, and G. Mes-
nil, “A latent semantic model with convolu-
tional pooling for information retrieval,” inPro-
ceedings of the 23rd ACM International Con-
ference on Information and Knowledge Manage-
ment (CIKM), 2014, pp. 101–110.
[18] J. Guo, Y. Fan, Q. Ai, and W. B. Croft, “A
deep relevance matching model for ad-hoc re-
trieval,” inProceedings of the 25th ACM Inter-
national Conference on Information and Knowl-
edge Management (CIKM), 2016, pp. 665–674.
[19] L. Pang, Y. Lan, J. Guo, J. Xu, S. Wan, and
X. Cheng, “Text matching as image recogni-
19

tion,” inProceedings of the Thirtieth AAAI Con-
ference on Artificial Intelligence (AAAI), 2016.
[20] Z. Yang, X. He, J. Gao, L. Deng, and A. Smola,
“Stacked attention networks for image question
answering,” inProceedings of the IEEE Confer-
ence on Computer Vision and Pattern Recogni-
tion (CVPR), 2016, pp. 21–29.
[21] D. Chen, J. Bolton, and C. D. Manning, “A
thorough examination of the CNN/Daily Mail
reading comprehension task,” inProceedings of
the 54th Annual Meeting of the Association for
Computational Linguistics (ACL), 2016.
[22] R. Nogueira and K. Cho, “Passage re-ranking
with bert,” 2020. [Online]. Available: https:
//arxiv.org/abs/1901.04085
[23] Z. Dai and J. Callan, “Deeper text understand-
ing for ir with contextual neural language mod-
eling,” inProceedings of the 42nd International
ACM SIGIR Conference on Research and Devel-
opment in Information Retrieval, ser. SIGIR ’19.
ACM, 2019, p. 985–988. [Online]. Available:
http://dx.doi.org/10.1145/3331184.3331303
[24] O. Khattab and M. Zaharia, “ColBERT: Effi-
cient and effective passage search via contextual-
ized late interaction over BERT,” inProceedings
of the 43rd International ACM SIGIR Confer-
ence on Research and Development in Informa-
tion Retrieval, 2020, pp. 39–48.
[25] T. Formal, B. Piwowarski, and S. Clinchant,
“SPLADE: Sparse lexical and expansion model
for first stage retrieval,” inProceedings of the
44th International ACM SIGIR Conference on
Research and Development in Information Re-
trieval, 2021, pp. 2288–2292.
[26] T. Brown, B. Mann, N. Ryderet al., “Lan-
guage models are few-shot learners,” inAd-
vances in Neural Information Processing Sys-
tems (NeurIPS), vol. 33, 2020, pp. 1877–1901.
[27] C. Raffel, N. Shazeer, A. Robertset al., “Explor-
ing the limits of transfer learning with a unified
text-to-text transformer,”Journal of MachineLearning Research, vol. 21, no. 140, pp. 1–67,
2020.
[28] P. Liu, W. Yuan, J. Fu, Z. Jiang, H. Hayashi,
and G. Neubig, “Pre-train, prompt, and predict:
A systematic survey of prompting methods in
natural language processing,”ACM Computing
Surveys, 2023.
[29] T. Gao, A. Fisch, and D. Chen, “Making pre-
trained language models better few-shot learn-
ers,” inProceedings of the 59th Annual Meeting
of the Association for Computational Linguistics
(ACL), 2021.
[30] H. W. Chung, L. Hou, S. Longpre, B. Zoph,
Y. Tay, W. Fedus, Y. Li, X. Wang, M. Dehghani,
S. Brahma, A. Webson, S. S. Gu, Z. Dai,
M. Suzgun, X. Chen, A. Chowdhery, A. Castro-
Ros, M. Pellat, K. Robinson, D. Valter,
S. Narang, G. Mishra, A. Yu, V. Zhao,
Y. Huang, A. Dai, H. Yu, S. Petrov, E. H. Chi,
J. Dean, J. Devlin, A. Roberts, D. Zhou, Q. V.
Le, and J. Wei, “Scaling instruction-finetuned
language models,” 2022. [Online]. Available:
https://arxiv.org/abs/2210.11416
[31] W. Sun, L. Yan, Z. Chenet al., “Is ChatGPT
good at search? investigating ChatGPT as a re-
ranker with progressive stacking,” inProceedings
of the 46th International ACM SIGIR Confer-
ence on Research and Development in Informa-
tion Retrieval, 2023.
[32] A. Asai, T. Schick, P. Lewiset al., “Task-aware
retrieval with instructions,” inProceedings of
the 61st Annual Meeting of the Association for
Computational Linguistics (ACL), 2023.
[33] X. Ma, L. Wang, N. Yang, F. Wei, and
J. Lin, “Fine-tuning llama for multi-stage text
retrieval,” 2023. [Online]. Available: https:
//arxiv.org/abs/2310.08319
[34] N. Choi, Y. Lee, G.-H. Cho, H. Jeong,
J. Kong, S. Kim, K. Park, S. Cho, I. Jeong,
G. Nam, S. Han, W. Yang, and J. Choi,
“Rradistill: Distilling llms’ passage ranking
20

ability for long-tail queries document re-ranking
on a search engine,” 2024. [Online]. Available:
https://arxiv.org/abs/2410.18097
[35] Y. Zhu, H. Yuan, S. Wang, J. Liu, W. Liu,
C. Deng, H. Chen, Z. Liu, Z. Dou, and J.-R.
Wen, “Large language models for information
retrieval: A survey,”ACM Transactions on
Information Systems, vol. 44, no. 1, p.
1–54, Nov. 2025. [Online]. Available: http:
//dx.doi.org/10.1145/3748304
[36] Y. Gao, Y. Xiong, X. Gao, K. Jia, J. Pan,
Y. Bi, Y. Dai, J. Sun, M. Wang, and
H. Wang, “Retrieval-augmented generation
for large language models: A survey,” 2024.
[Online]. Available: https://arxiv.org/abs/2312.
10997
[37] J. Wei, X. Wang, D. Schuurmanset al., “Chain-
of-thought prompting elicits reasoning in large
language models,” inAdvances in Neural Infor-
mation Processing Systems (NeurIPS), 2022.
[38] H. Zhuang, Z. Qin, K. Huiet al., “Beyond
yes and no: Improving zero-shot LLM rankers
via scoring fine-grained relevance indicators,” in
Proceedings of the 46th International ACM SI-
GIR Conference on Research and Development
in Information Retrieval, 2023.
[39] T. Tang, Z. Tian, Z. Zhu, C. Wang, H. Hu,
G. Tang, L. Liu, and S. Xu, “Lref: A novel
llm-based relevance framework for e-commerce
search,” inCompanion Proceedings of the ACM
on Web Conference 2025, 2025, pp. 468–475.
[40] Z. Fang, D. Xie, M. Pang, C. Yuan, X. Jiang,
C. Peng, Z. Lin, and Z. Luo, “Adore: Au-
tonomous domain-oriented relevance engine for
e-commerce,” inProceedings of the 48th Interna-
tional ACM SIGIR Conference on Research and
Development in Information Retrieval, 2025, pp.
4259–4263.
[41] C. Lu, Z. Chen, H. Zhao, Z. Zeng, G. Zhao,
J. Ren, R. Xu, H. Li, S. Liu, P. Wang, J. Xu,
and B. Zheng, “Lore: A large generative modelfor search relevance,” 2026. [Online]. Available:
https://arxiv.org/abs/2512.03025
[42] C. Dong, S. Yao, P. Jiao, J. Yang, Y. Jin,
Z. Huang, X. Zhou, D. Ou, H. Tang, and
B. Zheng, “Taosr1: The thinking model for
e-commerce relevance search,” 2026. [Online].
Available: https://arxiv.org/abs/2508.12365
[43] Z. Zeng, H. Jing, J. Chen, X. Li, H. Liu,
Y. He, Z. Li, Y. Sun, Z. Xie, Y. Yang,
S. Cao, J. Fan, Y. Wu, and Y. Hu,
“Optimizing generative ranking relevance via
reinforcement learning in xiaohongshu search,”
Proceedings of the 32nd ACM SIGKDD Con-
ference on Knowledge Discovery and Data
Mining V.1, 2025. [Online]. Available: https:
//api.semanticscholar.org/CorpusID:283450007
[44] K. Wang, F. Duan, S. Wang, P. Li, Y. Xian,
C. Yin, W. Rong, and Z. Xiong, “Knowledge-
driven cot: Exploring faithful reasoning in llms
for knowledge-intensive question answering,”
2023. [Online]. Available: https://arxiv.org/
abs/2308.13259
[45] R. Rafailov, A. Sharma, E. Mitchell, C. D. Man-
ning, S. Ermon, and C. Finn, “Direct preference
optimization: Your language model is secretly a
reward model,”Advances in neural information
processing systems, vol. 36, pp. 53 728–53 741,
2023.
[46] Y. Wang, L. Wang, Y. Li, D. He, and T.-Y.
Liu, “A theoretical analysis of ndcg type rank-
ing measures,” inConference on learning theory.
PMLR, 2013, pp. 25–54.
[47] DeepSeek-AI, D. Guo, and D. Y. et al.,
“Deepseek-r1 incentivizes reasoning in llms
through reinforcement learning,”Nature, vol.
645, pp. 633 – 638, 2025.
[48] T. GLM, A. Zeng, and B. X. et al., “Chatglm: A
family of large language models from glm-130b
to glm-4 all tools,” 2024.
[49] J. Liu, Y. Fan, and Z. J. et al., “Synlogic:
Synthesizing verifiable reasoning data at scale
21

for learning logical reasoning and beyond,”
2025. [Online]. Available: https://arxiv.org/
abs/2505.19641
[50] Y. Zhang, M. Li, D. Long, X. Zhang,
H. Lin, B. Yang, P. Xie, A. Yang, D. Liu,
J. Lin, F. Huang, and J. Zhou, “Qwen3
embedding: Advancing text embedding and
reranking through foundation models,” 2025.
[Online]. Available: https://arxiv.org/abs/2506.
05176
22