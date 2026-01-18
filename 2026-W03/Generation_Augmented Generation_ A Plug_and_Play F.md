# Generation-Augmented Generation: A Plug-and-Play Framework for Private Knowledge Injection in Large Language Models

**Authors**: Rongji Li, Jian Xu, Xueqing Chen, Yisheng Yang, Jiayi Wang, Xingyu Chen, Chunyu Xie, Dawei Leng, Xu-Yao Zhang

**Published**: 2026-01-13 04:23:36

**PDF URL**: [https://arxiv.org/pdf/2601.08209v1](https://arxiv.org/pdf/2601.08209v1)

## Abstract
In domains such as biomedicine, materials, and finance, high-stakes deployment of large language models (LLMs) requires injecting private, domain-specific knowledge that is proprietary, fast-evolving, and under-represented in public pretraining. However, the two dominant paradigms for private knowledge injection each have pronounced drawbacks: fine-tuning is expensive to iterate, and continual updates risk catastrophic forgetting and general-capability regression; retrieval-augmented generation (RAG) keeps the base model intact but is brittle in specialized private corpora due to chunk-induced evidence fragmentation, retrieval drift, and long-context pressure that yields query-dependent prompt inflation. Inspired by how multimodal LLMs align heterogeneous modalities into a shared semantic space, we propose Generation-Augmented Generation (GAG), which treats private expertise as an additional expert modality and injects it via a compact, representation-level interface aligned to the frozen base model, avoiding prompt-time evidence serialization while enabling plug-and-play specialization and scalable multi-domain composition with reliable selective activation. Across two private scientific QA benchmarks (immunology adjuvant and catalytic materials) and mixed-domain evaluations, GAG improves specialist performance over strong RAG baselines by 15.34% and 14.86% on the two benchmarks, respectively, while maintaining performance on six open general benchmarks and enabling near-oracle selective activation for scalable multi-domain deployment.

## Full Text


<!-- PDF content starts -->

Generation-Augmented Generation: A Plug-and-Play Framework for
Private Knowledge Injection in Large Language Models
Rongji Li1,2,3, Jian Xu1,2,3B, Xueqing Chen3, Yisheng Yang1,4, Jiayi Wang1,4
Xingyu Chen3, Chunyu Xie5, Dawei Leng5B, Xu-Yao Zhang1,2,3
1MAIS, Institute of Automation, Chinese Academy of Sciences
2School of Artificial Intelligence, University of Chinese Academy of Sciences
3Zhongguancun Academy
4School of Advanced Interdisciplinary Sciences, University of Chinese Academy of Sciences
5360 AI Research
Abstract
In domains such as biomedicine, materials, and
finance, high-stakes deployment of large lan-
guage models (LLMs) requires injecting pri-
vate, domain-specific knowledge that is pro-
prietary, fast-evolving, and under-represented
in public pretraining. However, the two dom-
inant paradigms for private knowledge injec-
tion each have pronounced drawbacks: fine-
tuning is expensive to iterate, and continual
updates risk catastrophic forgetting and general-
capability regression; retrieval-augmented gen-
eration (RAG) keeps the base model intact
but is brittle in specialized private corpora due
to chunk-induced evidence fragmentation, re-
trieval drift, and long-context pressure that
yields query-dependent prompt inflation. In-
spired by how multimodal LLMs align het-
erogeneous modalities into a shared semantic
space, we proposeGeneration-Augmented
Generation (GAG), which treats private ex-
pertise as an additional expert modality and
injects it via a compact, representation-level in-
terface aligned to the frozen base model, avoid-
ing prompt-time evidence serialization while
enabling plug-and-play specialization and scal-
able multi-domain composition with reliable se-
lective activation. Across two private scientific
QA benchmarks (immunology adjuvant and
catalytic materials) and mixed-domain evalua-
tions, GAG improves specialist performance
over strong RAG baselines by15.34%and
14.86%on the two benchmarks, respectively,
while maintaining performance on six open
general benchmarks and enabling near-oracle
selective activation for scalable multi-domain
deployment.
1 Introduction
In recent years, large language models (LLMs)
have demonstrated strong capabilities across a wide
BCorresponding author.
Figure 1:Three paradigms for private knowledge
injection.Fine-tuning is expensive and risky; RAG is
complex and fragile due to retrieval and long-context
pressure.GAGinjects private expertise through a
constant-budget, modular interface with selective ac-
tivation.
range of natural language processing tasks, includ-
ing text understanding, generation, and instruction
following (Grattafiori et al., 2024; Yang et al., 2025;
Liu et al., 2024a; Guo et al., 2025). Pretrained on
vast corpora of general text data, LLMs have pro-
foundly impacted various aspects of daily life and
professional environments. However, despite these
impressive general capabilities, enabling LLMs to
perform optimally in private domains remains a
significant challenge. In private-domain deploy-
ments such as biomedicine, materials, and finance
(Bao et al., 2023; Chen et al., 2023b,a), reliable
performance often requires incorporating domain-
specific knowledge beyond open-domain pretrain-
ing, where expert terminology and conventions are
critical for accurate and dependable outputs.
Two dominant paradigms are commonly used
to inject private knowledge into LLMs.(i) Do-
main fine-tuningcan internalize domain knowl-
edge, but it is costly to iterate, requires careful
validation, and risks general-capability regression
and catastrophic forgetting under continual updates
(Gururangan et al., 2020; Hu et al., 2022; Dettmers
et al., 2023).(ii) Retrieval-augmented generation
(RAG)preserves the base model by retrieving tex-
1arXiv:2601.08209v1  [cs.CL]  13 Jan 2026

tual evidence at inference time (Lewis et al., 2020;
Guu et al., 2020; Izacard and Grave, 2021; Izacard
et al., 2023). However, in private domains RAG
is often brittle: evidence is fragmented by chunk-
ing, retrieval can drift or miss crucial context, and
even relevant passages must compete for limited
context budget and are unevenly utilized by long-
context LLMs (Liu et al., 2024b). Moreover, be-
cause evidence must be serialized into the prompt,
RAG induces query-dependent context expansion,
making inference behavior less predictable as the
private corpus scales. Figure 1 shows these trade-
offs and positionsGeneration-Augmented Gen-
eration (GAG)as a constant-budget, modular al-
ternative to both fine-tuning and retrieval-based
injection.
In this work, we reformulate private knowledge
injection from a multimodal perspective, concep-
tualizing such knowledge not as textual snippets
but as an auxiliary modality. This modality can
be aligned and fused into the general LLM via
parameter-efficient interfaces (Alayrac et al., 2022;
Li et al., 2023; Liu et al., 2023; Huang et al.,
2023). To overcome the limitations of both fine-
tuning and RAG, we introduceGAG: a retrieval-
free, plug-and-play framework that performs pri-
vate knowledge injection through a constant-budget
interface, without updating the parameters of the
frozen base model. The framework adopts a de-
coupled architecture that partitions the system
into a general-purpose base model and a set of
lightweight domain-specific expert modules. A
routing mechanism dynamically selects between a
general route and domain-specific routes. This de-
sign transforms private-knowledge injection into a
constant-budget operation by injecting a single con-
tinuous token into the frozen base model, avoiding
both base-model fine-tuning and retrieval-time evi-
dence serialization, thereby enabling efficient and
scalable integration of knowledge across multiple
domains while preserving general-domain capabil-
ity.
Contributions.(1) We introduce GAG, a
retrieval-free and plug-and-play framework for pri-
vate knowledge injection into a frozen base model
via a single-token continuous interface. (2) We pro-
pose a prototype-based global router that enables
reliable selective activation and incremental multi-
domain expansion without router training. (3) We
validate GAG on two private scientific domains and
mixed-domain settings, demonstrating substantialspecialist improvements while maintaining general-
domain performance.
2 Related Work
Fine-tuning-based knowledge injection.Para-
metric adaptation injects domain knowledge via
continued pretraining or supervised fine-tuning on
domain data (Gururangan et al., 2020). A key
challenge is catastrophic forgetting and general-
capability regression under continual updates,
unless continual-learning controls are applied
(Kirkpatrick et al., 2017; Li and Hoiem, 2017).
To reduce cost, parameter-efficient fine-tuning
(PEFT) updates only small parameter subsets
(e.g., adapters, prefix/prompt tuning, sparse up-
dates) (Houlsby et al., 2019; Li and Liang, 2021;
Lester et al., 2021; Zaken et al., 2022). Low-
rank and quantization-aware PEFT further improve
efficiency and memory (Hu et al., 2022; Zhang
et al., 2023; Mao et al., 2022; Pfeiffer et al., 2021;
Dettmers et al., 2023; Lialin et al., 2023). However,
they still require iterative training/re-validation and
maintain evolving domain parameters, which con-
flicts with deployments requiring a strictly frozen
base model for governance and regression control.
This motivates modular knowledge injection mech-
anisms that preserve a frozen base model while
supporting plug-and-play domain expansion.
Retrieval-augmented knowledge injection.
Retrieval-augmented generation (RAG) injects
external knowledge by retrieving evidence from a
corpus and conditioning an LLM on the retrieved
text, and it is widely adopted for knowledge-
intensive QA (Lewis et al., 2020; Guu et al.,
2020; Izacard and Grave, 2021). A large body
of work improves the retrieve-then-read pipeline
via stronger dense retrieval and late-interaction
matching, better training objectives, and more
effective reader-side fusion (Karpukhin et al.,
2020; Xiong et al., 2020; Khattab and Zaharia,
2020). Other lines integrate retrieval more tightly
into training/inference, or plug retrieval into
pretrained LMs to improve factual grounding and
sample efficiency (Izacard et al., 2023; Borgeaud
et al., 2022; Shi et al., 2024; Khandelwal et al.,
2019). More recently, LM-driven generation or
verification signals have been explored to improve
retrieval robustness and attribution faithfulness
(Gao et al., 2023b,a; Asai et al., 2024). Despite
these advances, RAG can be particularly challeng-
ing in private, fast-evolving domains: evidence is
2

fragmented by chunking, top- kretrieval does not
guarantee complete coverage, and long-context
LMs may under-utilize or misinterpret relevant
spans when multiple passages compete within
a finite context budget (Liu et al., 2024b; Bai
et al., 2024). Context-compression methods (e.g.,
one-token interfaces) reduce prompt overhead, but
remain retrieval-dependent and thus still hinge on
retrieval coverage and indexing quality (Cheng
et al., 2024). Meanwhile, auxiliary-model-based
knowledge transfer can inject domain signals
through prompt-time mediation (Li et al., 2025),
but still relies on textual handoff and typically
requires non-trivial organization of training data
and prompts, while remaining subject to context-
budget pressure, making modular multi-domain
scaling less straightforward. In contrast, our work
targets a retrieval-free, constant-budget injection
interface under a frozen base model, avoiding
prompt-time evidence serialization while enabling
predictable inference behavior, requiring only min-
imal per-domain data preparation, and supporting
plug-and-play multi-domain composition with
reliable selective activation via routing.
3 Problem Formulation
We consider question answering with a frozen base
model. Let pθ(y|x) denote the conditional distri-
bution induced by a pretrained model with param-
etersθ, where xis a user query and yis the target
answer. After deployment, θis not allowed to be
updated.
Multi-domain private knowledge.Queries are
drawn from a mixture of one general distribution
D0andNprivate-domain distributions {Di}N
i=1.
For each domain i, samples (x, y)∼ D ireflect
domain-specific knowledge needs (e.g., enterprise
or vertical expertise) that are not reliably covered
by public pretraining. We assume each private
domain iis associated with a private knowledge
source Ki(e.g., internal documents or curated re-
sources), but we do not retrain the base model on
Ki.
Knowledge injection as conditional generation
with side information.Our objective is to enable
the frozen base model to answer domain-specific
queries without modifying its parameters. To this
end, we allow the system to condition the base
model on an auxiliary injected signal zderived
Figure 2:GAG overview.A training-free prototype
router selects either the general route or one of Nplug-
and-play domain modules. Each module derives an
expert readout from LLM domain,i and projects it into
LLM base’s embedding space as a single continuous in-
jected token, enabling constant-budget, retrieval-free
knowledge injection under a frozen base model.
from(x,K i):
z=A i(x,K i),ˆy∼p θ(y|x, z),(1)
where Aiis an abstract domain-specific injection
mechanism. Eq. (1)abstracts away the form of
zand its integration, requiring only an external
modular interface that can influence generation.
Objective and constraints.Our goal is to im-
prove private-domain QA quality while preserving
the base model’s general capability. Let L(ˆy, y) be
a task loss (or an evaluation-aligned surrogate). We
seek injection mechanisms{A i}N
i=1such that:
min
{Ai}N
i=1NX
i=1E(x,y)∼D i[L(ˆy i(x), y)]
s.t.E (x,y)∼D 0[L(ˆy 0(x), y)]≤R 0+ϵ.(2)
where R0denotes the baseline risk of the frozen
base model on D0(without injection) and ϵis an
allowable regression margin.
Plug-and-play domain expansion.We further
require modular multi-domain expansion: when a
3

Figure 3:GAG two-stage learning. Stage Idistills domain competence into LLM domain,i ’s internal representations.
Stage IIlearns a lightweight projector Πithat maps the expert readout ki(x)intoLLM base’s embedding space,
yielding a single continuous injected token for constant-budget, plug-and-play knowledge injection.
new domain karrives, the system should incorpo-
rateAkwithout modifying the base-model parame-
tersθor previously deployed mechanisms {Ai}i<k.
This captures the practical constraint that private
knowledge evolves across domains while the base
model must remain stable and reusable.
4 Method
We proposeGAG, which injects private knowl-
edge into a strictly frozen base model by attaching
lightweight domain modules that synthesize holis-
tic domain background and align it via a learned
one-token interface, with a prototype router acti-
vating the appropriate route on demand. Figure 2
depicts the end-to-end routed pipeline.
4.1 GAG Architecture
LetLLM basebe a frozen base model with parame-
tersθand hidden size d1. Given a query x, GAG
first predicts a route index
ˆi=r(x)∈ {0,1, . . . , N},(3)
then produces an answer by conditioning LLM base
on a route-specific side signalz ˆi(x)∈Rd1:
ˆy∼p θ 
y|x, z ˆi(x)
, z 0(x)≜NULL.(4)
Routes i≥1 correspond to domain expert modules
(Section 4.2); r(x) is realized by PPR (Section 4.3).
4.2 Generation Augmentation for Each
Specialist Domain
We instantiate the injected signal zi(x)as a com-
pact representation-level summary of domain ex-
pertise, obtained from a domain expert model and
aligned to LLM base’s input embedding space via a
lightweight projector.Expert background synthesis and semantic read-
out.For each domain i, a lightweight causal LM
LLM domain,i (parameters ϕi, hidden size d2) gener-
ates a background sequence:
b1:T∼pϕi(b|x).(5)
Leth(ℓ)
t∈Rd2denote the hidden state at step tand
layer ℓ. We compress the generated background
into a single expert vector via a late-layer readout:
ki(x) =h(ℓ⋆)
T∈Rd2, ℓ⋆∈ {1, . . . , L 2},
(6)
where L2is the number of transformer layers in
LLM domain,i andℓ⋆is a readout layer (ablation in
Section 7.1).
Token-geometry alignment and one-token in-
jection.We learn a lightweight projector Πi:
Rd2→Rd1(parameters ψi) to map the expert vec-
tor intoLLM base’s embedding space:
zi(x) = Π i(ki(x))∈Rd1.(7)
To conditionLLM base, we fix an anchor slot at po-
sition ain the prompt template s1:nand substitute
its input embedding. Let Eθ(s1:n)∈Rn×d 1denote
LLM base’s input embeddings; we form
E(i)
θ(x) =E θ(s1:n)withE θ(sa)←z i(x),(8)
and decode with the frozen base model:
ˆy∼p θ 
y|E(i)
θ(x)
.(9)
Eq.(8)defines a constant-budget knowledge in-
terface (one injected token), related to continuous
prompting (Li and Liang, 2021; Lester et al., 2021)
but expert-generated and cross-model aligned.
Figure 3 illustrates the two-stage learning proce-
dure.
4

Figure 4:Prototype Plug-and-Play Routing (PPR).
Offline:embed historical queries with a frozen encoder
gηand cluster them into per-domain prototype banks.
Online:route each query by nearest-prototype match-
ing to select the general path or a domain module, en-
abling training-free, scalable multi-domain expansion
without updating the base model.
Stage I: domain expert acquisition via QA adap-
tation.We first endow LLM domain,i with do-
main expertise by adapting it on in-domain QA
pairs. Concretely, given (x, y)∼ D i, we train
LLM domain,i to model the domain-conditional an-
swer distribution, thereby internalizing salient do-
main regularities that later serve as a strong expert
prior for background synthesis. Formally, we opti-
mize the standard autoregressive objective
min
ϕiE(x,y)∼D ih
−|y|X
t=1logp ϕi(yt|y<t, x)i
.(10)
After this adaptation, LLM domain,i can be prompted
to produce holistic domain background as an inter-
mediate signal.
Stage II: projector alignment under a frozen
base model.We then freeze θandϕi, and learn
only the projector parameters ψiby maximizing
the likelihood of the gold answer under injected
decoding:
min
ψiE(x,y)∼D ih
−|y|X
t=1logp θ 
yt|y<t,E(i)
θ(x)i
.
(11)
This objective directly optimizes the injection in-
terface in LLM base’s native representational space,
enabling domain specialization without updating
the base-model parameters.
Overall, in Stage I, we adopt the simplest train-
ing scheme to endow the lightweight LLM domain,i
with domain knowledge; in Stage II, we further
enable it to produce background knowledge while
aligning its outputs to the frozen LLM base. This de-
sign is simple yet effective, and it naturally supports
reusing domain models released by prior work.4.3 Prototype Plug-and-Play Routing
To enable plug-and-play multi-domain expansion
and reliable selective activation, we incorporate
Prototype Plug-and-Play Routing (PPR) as a built-
in component of GAG. PPR implements r(x) in
Eq.(3)via a training-free prototype-based deci-
sion rule, serving as a non-parametric alternative to
learned routers in conditional computation / MoE
systems (Lepikhin et al., 2020; Fedus et al., 2022).
Embeddings and prototypes.Let gηbe a frozen
encoder and Pool(·) a fixed pooling operator. We
embed and normalize:
e(x) =Pool(g η(x))
∥Pool(g η(x))∥ 2∈Rd.(12)
Offline, for each route i, we cluster historical query
embeddings intoC iprototypes:
Pi={p i,1, . . . ,p i,Ci}
= KMeans({e(x)} x∈Qi;Ci),∥p i,c∥2= 1.
(13)
Nearest-prototype routing.At inference time,
PPR routes by nearest-prototype cosine similarity:
si(x) = max
ce(x)⊤pi,c,
r(x) = arg max
i∈{0,...,N}si(x).(14)
Figure 4 visualizes the embedding space, do-
main prototypes, and the nearest-prototype deci-
sion boundary.
Modularity and incremental deployment.
GAG supports domain expansion by construction:
adding a new domain konly requires attaching
(LLM domain,k ,Πk)and computing prototypes Pk,
while keeping LLM base and existing modules
unchanged. This isolates domain updates from
the base model, mitigating the iteration cost and
regression risks typically associated with repeated
fine-tuning.
5 Experimental Setup
5.1 Datasets and Metrics
We evaluate GAG on bothgeneral-domain QA
andspecialist private-domain QAto quantify
whether modular knowledge injection improves
domain expertise without compromising broad us-
ability. For general QA, we follow prior work
and report performance on six widely used open-
domain benchmarks—FreebaseQA (Jiang et al.,
5

2019), HotpotQA (Yang et al., 2018), Natural Ques-
tions (Kwiatkowski et al., 2019), TriviaQA (Joshi
et al., 2017), WebQuestions (Berant et al., 2013),
and PopQA (Mallen et al., 2022)—usingExact
Match (EM)with standard answer normalization,
which provides a stringent measure of factual cor-
rectness under canonical string matching. To study
domain knowledge injection, we focus on two spe-
cialist domains:immunology adjuvantandcat-
alytic materials. Concretely, we treat (Anony-
mous, 2025b) and (Anonymous, 2025a) as the su-
pervision sources for domain expert knowledge
injection, and evaluate on their held-out test sets
to quantify specialist QA quality. Because refer-
ence answers in these domains are often free-form
and allow surface variation, we reportBERTScore
(Zhang et al., 2019) computed with SciBERT (Belt-
agy et al., 2019), a scientific-domain encoder that
better aligns evaluation with semantic faithfulness
in technical language. More detailed dataset statis-
tics are provided in Appendix A.
5.2 Implementation Details
We instantiate the frozen base model LLM basewith
Qwen3-8Band domain expert models LLM domain,i
withQwen3-1.7B(Yang et al., 2025). The projec-
torΠiis implemented as a lightweight two-layer
MLP with a GELU nonlinearity. For routing, we
useQwen3-1.7Bas a frozen query encoder and
treat general as a peer route: all domains compete
via maximum prototype cosine similarity, requir-
ing neither router training nor threshold tuning. All
experiments are run on 8 ×NVIDIA A100 GPUs
with bfloat16 precision and FlashAttention-2 (Dao,
2024) when available; full training hyperparame-
ters, routing configuration, inference and decoding
settings, and prompt templates are provided in Ap-
pendix B and Appendix C.5.3 Baselines
We compare against representative and competitive
knowledge-injection baselines under a frozen base
model constraint:(i) Base-Model-Only, where
Qwen3-8B answers directly without any external
knowledge;(ii) RAG, which builds domain cor-
pora by parsing scientific papers with MinerU2.5
(Niu et al., 2025), retrieves top- kdomain evidence
via ColBERTv2 (Santhanam et al., 2022), and con-
ditions the same base model on the retrieved text;
(iii) xRAG(Cheng et al., 2024), which performs
retrieval augmentation by retrieving the top-1 back-
ground passage from the corresponding domain
knowledge base and compressing it under an ex-
treme budget; and(iv) Expert-Generated Context
(EGC), where a domain expert model first gener-
ates explicit background text that is appended to
the base model prompt (i.e., text-level knowledge
transfer without a learned knowledge injection in-
terface). Oracle routing is treated as a diagnostic
upper bound and is reported in Section 6.1.
6 Experimental Results
6.1 Overall Performance
Table 1 reports overall performance on two special-
ist domains (Adjuvant, Materials) and six general-
domain QA benchmarks under a strictly frozen
Qwen3-8B base model. We omit RAG/xRAG on
general-domain benchmarks because they serve
as closed-book regression checks; adding open-
domain retrieval would change the setting, while
disabling retrieval makes RAG/xRAG identical to
the base-model-only route. Overall,GAG delivers
the strongest specialist gains among all baselines
while preserving general-domain capability.
On specialist QA,GAG delivers large and con-
sistent gainsover both retrieval-based baselines
and text-level expert transfer. RAG improves only
marginally, and xRAG remains close to RAG even
SystemSpecialist Domain General DomainAdded Tokens
Adjuvant Materials FreebaseQA HotpotQA Natural Questions TriviaQA WebQuestions PopQA Average
Base-Model-Only 56.12 60.01 61.06 28.72 35.15 54.3649.96 23.70 42.16 0
RAG 59.97(6.86%) 62.13(3.53%) — 375.17
xRAG 59.12(5.35%) 62.24(3.72%) — 1
EGC (Adjuvant) 64.07(14.17%) 58.60(- 2.35%) 38.50(- 36.95%) 18.24(- 36.49%) 17.09(- 51.38%) 34.54(- 36.46%) 30.22(- 39.51%) 14.36(- 39.41%) 25.49(- 39.54%) 148.48
EGC (Materials) 54.22(- 3.39%) 66.47(10.76%) 38.59(- 36.80%) 21.23(- 26.08%) 18.85(- 46.37%) 35.24(- 35.17%) 33.22(- 33.51%) 14.89(- 37.17%) 27.00(- 35.96%) 165.21
GAG (Gen+Adj) 69.16(23.24%) —61.41(0.57%)28.46(- 0.91%)35.59(1.25%)54.01(- 0.64%) 49.96(0.00%) 24.41(3.00%) 42.31(0.36%) 1
GAG (Gen+Adj+Mat)69.17(23.25%) 71.36(18.91%) 61.41(0.57%) 29.16(1.53%)34.98(- 0.48%) 53.57(- 1.45%)50.31(0.70%) 24.67(4.09%) 42.35(0.45%)1
Table 1:Overall performance across specialist and general domains.Specialist-domain QA is evaluated on
Adjuvant and Materials (BERTScore with SciBERT, ×100 ), and general-domain QA is evaluated with Exact Match
(EM) on six open benchmarks.Boldindicates the best result in each column and underline indicates the second best.
Numbers in parentheses denote relative improvements over the Base-Model-Only (Qwen3-8B) baseline. Added
Tokens reports the average number of additional knowledge tokens injected into the base-model prompt.
6

under a one-token budget, indicating that the dom-
inant bottleneck is retrieval reliability rather than
context length alone: chunk fragmentation, entity
drift, and incomplete coverage mean that compress-
ing retrieved text cannot fix missing or misaligned
evidence. EGC exposes a complementary failure
mode: although a matched expert can help, directly
appending generated background is an inherently
high-variance text-level intervention—its relevance
and granularity are not guaranteed—so the added
context can dilute or misguide the base model’s fo-
cus under attention competition, limiting in-domain
robustness. In contrast, GAG transfers domain
competence through a representation-level inter-
face—aligning an expert readout to the frozen base
model’s embedding geometry—yielding a com-
pact, high-signal conditioning token that avoids
prompt-time evidence competition and consistently
improves specialist performance across domains.
RAG results under different retrieval depths are
reported in Appendix D.
Importantly, these specialist improvements do
not come at the expense of general QA.GAG main-
tains the general-domain averagerelative to the
base model, whereas EGC serves as a stress test
showing that indiscriminate text-level injection can
severely degrade open-domain performance. This
contrast highlights the practical necessity of reli-
able selective activation. Finally, GAG operates
with aconstant knowledge budgetyet closely
approaches the oracle-route upper bounds on spe-
cialist domains (69.72 on Adjuvant and 71.53 on
Materials under oracle domain routing), offering a
deployment-friendly alternative to variable-length
retrieval and text conditioning.
Router configuration Active routes Micro Per-route acc. (%)
acc. (%) Gen Adj Mat
PPR (2 routes; base) Gen + Adj 99.78 99.65 99.91 —
PPR (3 routes; incremental +Mat) Gen + Adj + Mat 99.55 99.65 99.38 99.69
Table 2:Plug-and-play routing accuracy of PPR.Ac-
tive routes list the domains with loaded prototype banks.
Micro acc. is micro-averaged route-selection accuracy
over all evaluated queries. Per-route acc. reports class-
wise accuracy for each route (Gen/Adj/Mat).
6.2 Routing Accuracy of PPR
Reliable selective activation is a prerequisite for
plug-and-play expert deployment, since misrouting
can turn knowledge injection into harmful inter-
ference. Table 2 shows thatPPR achieves near-
oracle routing under a fully frozen setup: using afrozen Qwen3-1.7B encoder and nearest-prototype
matching, PPR attains99.78%micro-averaged ac-
curacy for Gen+Adj, and remains99.55%after
incrementally adding Mat without modifying any
existing routes. Per-route accuracy stays uniformly
high, indicating that PPR provides a stable, non-
parametric routing interface for scalable plug-and-
play expert composition; additional routing results
under broader incremental domain expansion are
deferred to the appendix E.
LLM domain,i readout layer BERTScore↑∆
L2−169.12−0.60
L2−269.20−0.52
L2−4(default) 69.72+0.00
L2−669.36−0.36
L2−869.38−0.34
L2−1269.14−0.58
L2−1668.87−0.85
L2−2066.69−3.03
L2−2458.42−11.30
Table 3:Readout-layer ablation for one-token GAG
injection.We form the injected token by projecting the
last-token hidden state from LLM domain,i layerℓinto the
frozen base model LLM base. We disable routing and al-
ways activate the Adjuvant module (oracle domain rout-
ing) to isolate the effect of readout depth. BERTScore ↑
(SciBERT) is evaluated on Adjuvant; ∆is the difference
relative to the best readout depth ( ℓ=L 2−4), which
we use as default.
7 Analysis
7.1 WhichLLM domain,i layer yields the best
one-token readout?
GAG relies on a single-vector readout ki(x)from
LLM domain,i (Eq. 6), making the readout depth a
key determinant of injection bandwidth and stabil-
ity. We ablate the readout layer by extracting the
last-token hidden state from ℓ∈ {L 2−1, L 2−2, L 2−
4, . . . , L 2−24} , projecting it into LLM base, and in-
jecting exactly one continuous token. All other
factors (frozen LLM base/LLM domain,i , prompts, de-
coding) are held constant.
Table 3 shows a sharp depth effect on Adjuvant:
ℓ=L 2−4is optimal and a narrow late-layer band
remains competitive, while substantially earlier lay-
ers collapse performance (e.g., L2−20/L2−24).
This pattern suggests that effective one-token trans-
fer prefers representations that are semantically
consolidated yet not overly specialized to next-
token prediction. Accordingly, we set ℓ⋆=L 2−4
as the default readout layer in all experiments for ro-
7

Figure 5:Case study (RAG vs. GAG).We contrast RAG’s retrieved evidence and answer with GAG’s injected-
token route and answer for the same query and reference. The displayed “Generated Expert Background” is an
analysis-only visualization of the injected signal (not exposed in deployment).
Variant Stage I Stage II BERTScore↑
w/o Stage I×✓57.14
w/o Stage II✓×55.64
Full✓ ✓69.72
Table 4:Two-stage ablation of GAG (Adjuvant).All
variants are evaluated with oracle domain routing.
bust plug-and-play deployment under a fully frozen
base model.
7.2 Are both Stage I and Stage II necessary?
We conduct a two-stage training ablation of GAG
on Adjuvant to quantify the necessity of Stage I
and Stage II (Table 4).w/o Stage Ikeeps the align-
ment objective but starts from a non-specialized
LLM domain,i , reducing BERTScore to 57.14, which
indicates that the injected representation lacks
domain-sufficient signal.w/o Stage IIpreserves
an adapted expert but replaces alignment with an
untrained mapping, further degrading to 55.64, sug-
gesting that LLM domain,i representations are not di-
rectly usable in LLM base’s embedding space. The
full model reaches 69.72, confirming that both
stages are complementary prerequisites for effec-
tive one-token injection.
7.3 Case study
Figure 5 illustrates retrieval brittleness in chunked
private corpora: the query targets AbISCO-300 and
a T-cell/APC mechanism, yet RAG’s evidence is
entity-mismatched(AbISCO-100) and dominated
by humoral readouts, so the frozen base model
cannot ground the requested mechanism and ef-fectively abstains. GAG instead conditions the
base model via a constant-budget injected token
derived from the domain expert module, avoiding
prompt-time evidence serialization and its cover-
age gaps. Importantly, the expert background text
shown in the figure is only an analysis-time probe
for interpretability: in the actual GAG pipeline,
LLM domain,i producesno explicit text outputto
the user and contributes solely a single continuous
embedding. This case supports our claim that GAG
mitigates retrieval fragmentation/mismatch while
keeping the interface fixed and predictable under
a frozen base model. In Appendix F, we include
more interesting cases including error analysis.
8 Conclusion
We proposedGAG, a retrieval-free, plug-and-
play framework for injecting private, domain-
specific knowledge into a frozen base model via a
constant-budget one-token interface aligned from
a lightweight domain expert model’s hidden repre-
sentation. By moving from text-level evidence seri-
alization to representation-level knowledge transfer,
GAG directly targets key deployment bottlenecks
of prevailing paradigms: it avoids RAG’s brittle-
ness under chunking, context-window pressure,
and retrieval mismatch/noise, while sidestepping
the iteration cost and general-capability regression
risks that often accompany fine-tuning in continu-
ously evolving private domains. Across two private
scientific QA benchmarks and a mixed-domain set-
ting, GAG yields strong specialist gains while pre-
serving general QA, highlighting a practical route
toward modular, scalable, and governance-friendly
8

private-knowledge deployment in real-world LLM
systems.
9 Limitations
While GAG demonstrates strong effectiveness for
private knowledge injection, we note two limita-
tions. First, our formulation assumes that each
query is predominantly associated with a single do-
main. For genuinely cross-domain questions that
require composing knowledge from multiple pri-
vate domains, GAG typically injects knowledge
from only one selected private domain, which may
limit multi-domain knowledge fusion and lead to
degraded performance on such mixed queries. In
future work, probabilistic multi-domain joint in-
jection may help mitigate this limitation. Sec-
ond, because GAG injects knowledge via a single
continuous token rather than verbatim evidence
text, it may occasionally be less precise on ex-
act surface-form numerics/units when the evalu-
ation hinges on copying a specific number (as il-
lustrated in Appendix F); in practice, lightweight
post-hoc numeric/unit normalization or verification
can complement the constant-budget interface with-
out changing the core method.
References
Ankush Agarwal, Sakharam Gawade, Sachin
Channabasavarajendra, and Pushpak Bhattacharyya.
2022. There is no big brother or small brother:
knowledge infusion in language models for link
prediction and question answering. InProceedings
of the 19th International Conference on Natural
Language Processing (ICON), pages 204–211.
Jean-Baptiste Alayrac, Jeff Donahue, Pauline Luc,
Antoine Miech, Iain Barr, Yana Hasson, Karel
Lenc, Arthur Mensch, Katherine Millican, Malcolm
Reynolds, and 1 others. 2022. Flamingo: a visual
language model for few-shot learning.Advances in
neural information processing systems, 35:23716–
23736.
Anonymous. 2025a. Catalystbench: A comprehensive
multi-task benchmark for advancing language models
in catalysis science. InSubmitted to The Fourteenth
International Conference on Learning Representa-
tions. Under review.
Anonymous. 2025b. An open-ended benchmark and
formal framework for adjuvant research with MLLM.
InSubmitted to The Fourteenth International Confer-
ence on Learning Representations. Under review.
Akari Asai, Zeqiu Wu, Yizhong Wang, Avirup Sil, and
Hannaneh Hajishirzi. 2024. Self-rag: Learning to re-
trieve, generate, and critique through self-reflection.Yushi Bai, Xin Lv, Jiajie Zhang, Hongchang Lyu,
Jiankai Tang, Zhidian Huang, Zhengxiao Du, Xiao
Liu, Aohan Zeng, Lei Hou, and 1 others. 2024. Long-
bench: A bilingual, multitask benchmark for long
context understanding. InProceedings of the 62nd
Annual Meeting of the Association for Computational
Linguistics (Volume 1: Long Papers), pages 3119–
3137.
Zhijie Bao, Wei Chen, Shengze Xiao, Kuang Ren, Jiaao
Wu, Cheng Zhong, Jiajie Peng, Xuanjing Huang, and
Zhongyu Wei. 2023. Disc-medllm: Bridging gen-
eral large language models and real-world medical
consultation.arXiv preprint arXiv:2308.14346.
Iz Beltagy, Kyle Lo, and Arman Cohan. 2019. Scibert:
A pretrained language model for scientific text. In
EMNLP. Association for Computational Linguistics.
Jonathan Berant, Andrew Chou, Roy Frostig, and Percy
Liang. 2013. Semantic parsing on Freebase from
question-answer pairs. InProceedings of the 2013
Conference on Empirical Methods in Natural Lan-
guage Processing, pages 1533–1544, Seattle, Wash-
ington, USA. Association for Computational Linguis-
tics.
Sebastian Borgeaud, Arthur Mensch, Jordan Hoff-
mann, Trevor Cai, Eliza Rutherford, Katie Milli-
can, George Bm Van Den Driessche, Jean-Baptiste
Lespiau, Bogdan Damoc, Aidan Clark, and 1 others.
2022. Improving language models by retrieving from
trillions of tokens. InInternational conference on
machine learning, pages 2206–2240. PMLR.
Wei Chen, Qiushi Wang, Zefei Long, Xianyin Zhang,
Zhongtian Lu, Bingxuan Li, Siyuan Wang, Jiarong
Xu, Xiang Bai, Xuanjing Huang, and 1 others. 2023a.
Disc-finllm: A chinese financial large language
model based on multiple experts fine-tuning.arXiv
preprint arXiv:2310.15205.
Zi-Yi Chen, Fan-Kai Xie, Meng Wan, Yang Yuan, Miao
Liu, Zong-Guo Wang, Sheng Meng, and Yan-Gang
Wang. 2023b. Matchat: A large language model and
application service platform for materials science.
Chinese Physics B, 32(11):118104.
Xin Cheng, Xun Wang, Xingxing Zhang, Tao Ge, Si-
Qing Chen, Furu Wei, Huishuai Zhang, and Dongyan
Zhao. 2024. xrag: Extreme context compression
for retrieval-augmented generation with one token.
Advances in Neural Information Processing Systems,
37:109487–109516.
Karl Cobbe, Vineet Kosaraju, Mohammad Bavarian,
Mark Chen, Heewoo Jun, Lukasz Kaiser, Matthias
Plappert, Jerry Tworek, Jacob Hilton, Reiichiro
Nakano, Christopher Hesse, and John Schulman.
2021. Training verifiers to solve math word prob-
lems.arXiv preprint arXiv:2110.14168.
Tri Dao. 2024. FlashAttention-2: Faster attention with
better parallelism and work partitioning. InInter-
national Conference on Learning Representations
(ICLR).
9

Tim Dettmers, Artidoro Pagnoni, Ari Holtzman, and
Luke Zettlemoyer. 2023. Qlora: Efficient finetuning
of quantized llms.Advances in neural information
processing systems, 36:10088–10115.
William Fedus, Barret Zoph, and Noam Shazeer. 2022.
Switch transformers: Scaling to trillion parameter
models with simple and efficient sparsity.Journal of
Machine Learning Research, 23(120):1–39.
Luyu Gao, Zhuyun Dai, Panupong Pasupat, Anthony
Chen, Arun Tejasvi Chaganty, Yicheng Fan, Vincent
Zhao, Ni Lao, Hongrae Lee, Da-Cheng Juan, and
1 others. 2023a. Rarr: Researching and revising
what language models say, using language models.
InProceedings of the 61st Annual Meeting of the
Association for Computational Linguistics (Volume
1: Long Papers), pages 16477–16508.
Luyu Gao, Xueguang Ma, Jimmy Lin, and Jamie Callan.
2023b. Precise zero-shot dense retrieval without rel-
evance labels. InProceedings of the 61st Annual
Meeting of the Association for Computational Lin-
guistics (Volume 1: Long Papers), pages 1762–1777.
Aaron Grattafiori, Abhimanyu Dubey, Abhinav Jauhri,
Abhinav Pandey, Abhishek Kadian, Ahmad Al-
Dahle, Aiesha Letman, Akhil Mathur, Alan Schelten,
Alex Vaughan, and 1 others. 2024. The llama 3 herd
of models.arXiv preprint arXiv:2407.21783.
Daya Guo, Dejian Yang, Haowei Zhang, Junxiao
Song, Ruoyu Zhang, Runxin Xu, Qihao Zhu, Shi-
rong Ma, Peiyi Wang, Xiao Bi, and 1 others. 2025.
Deepseek-r1: Incentivizing reasoning capability in
llms via reinforcement learning.arXiv preprint
arXiv:2501.12948.
Suchin Gururangan, Ana Marasovi ´c, Swabha
Swayamdipta, Kyle Lo, Iz Beltagy, Doug Downey,
and Noah A Smith. 2020. Don’t stop pretraining:
Adapt language models to domains and tasks.arXiv
preprint arXiv:2004.10964.
Kelvin Guu, Kenton Lee, Zora Tung, Panupong Pasu-
pat, and Mingwei Chang. 2020. Retrieval augmented
language model pre-training. InInternational confer-
ence on machine learning, pages 3929–3938. PMLR.
Neil Houlsby, Andrei Giurgiu, Stanislaw Jastrzebski,
Bruna Morrone, Quentin De Laroussilhe, Andrea
Gesmundo, Mona Attariyan, and Sylvain Gelly. 2019.
Parameter-efficient transfer learning for nlp. InIn-
ternational conference on machine learning, pages
2790–2799. PMLR.
Edward J Hu, Yelong Shen, Phillip Wallis, Zeyuan
Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang,
Weizhu Chen, and 1 others. 2022. Lora: Low-rank
adaptation of large language models.ICLR, 1(2):3.
Shaohan Huang, Li Dong, Wenhui Wang, Yaru Hao,
Saksham Singhal, Shuming Ma, Tengchao Lv, Lei
Cui, Owais Khan Mohammed, Barun Patra, and 1
others. 2023. Language is not all you need: Align-
ing perception with language models.Advances inNeural Information Processing Systems, 36:72096–
72109.
Gautier Izacard and Edouard Grave. 2021. Leveraging
passage retrieval with generative models for open do-
main question answering. InProceedings of the 16th
conference of the european chapter of the association
for computational linguistics: main volume, pages
874–880.
Gautier Izacard, Patrick Lewis, Maria Lomeli, Lucas
Hosseini, Fabio Petroni, Timo Schick, Jane Dwivedi-
Yu, Armand Joulin, Sebastian Riedel, and Edouard
Grave. 2023. Atlas: Few-shot learning with retrieval
augmented language models.Journal of Machine
Learning Research, 24(251):1–43.
Kelvin Jiang, Dekun Wu, and Hui Jiang. 2019. Free-
baseQA: A new factoid QA data set matching trivia-
style question-answer pairs with Freebase. InPro-
ceedings of the 2019 Conference of the North Amer-
ican Chapter of the Association for Computational
Linguistics: Human Language Technologies, Volume
1 (Long and Short Papers), pages 318–323, Min-
neapolis, Minnesota. Association for Computational
Linguistics.
Mandar Joshi, Eunsol Choi, Daniel Weld, and Luke
Zettlemoyer. 2017. triviaqa: A Large Scale Distantly
Supervised Challenge Dataset for Reading Compre-
hension.arXiv e-prints, arXiv:1705.03551.
Vladimir Karpukhin, Barlas Oguz, Sewon Min,
Patrick SH Lewis, Ledell Wu, Sergey Edunov, Danqi
Chen, and Wen-tau Yih. 2020. Dense passage re-
trieval for open-domain question answering. In
EMNLP (1), pages 6769–6781.
Urvashi Khandelwal, Omer Levy, Dan Jurafsky, Luke
Zettlemoyer, and Mike Lewis. 2019. Generalization
through memorization: Nearest neighbor language
models.arXiv preprint arXiv:1911.00172.
Omar Khattab and Matei Zaharia. 2020. Colbert: Effi-
cient and effective passage search via contextualized
late interaction over bert. InProceedings of the 43rd
International ACM SIGIR conference on research
and development in Information Retrieval, pages 39–
48.
James Kirkpatrick, Razvan Pascanu, Neil Rabinowitz,
Joel Veness, Guillaume Desjardins, Andrei A Rusu,
Kieran Milan, John Quan, Tiago Ramalho, Ag-
nieszka Grabska-Barwinska, and 1 others. 2017.
Overcoming catastrophic forgetting in neural net-
works.Proceedings of the national academy of sci-
ences, 114(13):3521–3526.
Tom Kwiatkowski, Jennimaria Palomaki, Olivia Red-
field, Michael Collins, Ankur Parikh, Chris Alberti,
Danielle Epstein, Illia Polosukhin, Jacob Devlin, Ken-
ton Lee, and 1 others. 2019. Natural questions: a
benchmark for question answering research.Trans-
actions of the Association for Computational Linguis-
tics, 7:453–466.
10

Dmitry Lepikhin, HyoukJoong Lee, Yuanzhong Xu,
Dehao Chen, Orhan Firat, Yanping Huang, Maxim
Krikun, Noam Shazeer, and Zhifeng Chen. 2020.
Gshard: Scaling giant models with conditional com-
putation and automatic sharding.arXiv preprint
arXiv:2006.16668.
Brian Lester, Rami Al-Rfou, and Noah Constant. 2021.
The power of scale for parameter-efficient prompt
tuning.arXiv preprint arXiv:2104.08691.
Patrick Lewis, Ethan Perez, Aleksandra Piktus, Fabio
Petroni, Vladimir Karpukhin, Naman Goyal, Hein-
rich Küttler, Mike Lewis, Wen-tau Yih, Tim Rock-
täschel, and 1 others. 2020. Retrieval-augmented gen-
eration for knowledge-intensive nlp tasks.Advances
in neural information processing systems, 33:9459–
9474.
Haitao Li, Qingyao Ai, Jia Chen, Qian Dong, Zhi-
jing Wu, and Yiqun Liu. 2025. Blade: Enhancing
black-box large language models with small domain-
specific models. InProceedings of the AAAI Con-
ference on Artificial Intelligence, volume 39, pages
24422–24430.
Junnan Li, Dongxu Li, Silvio Savarese, and Steven Hoi.
2023. Blip-2: Bootstrapping language-image pre-
training with frozen image encoders and large lan-
guage models. InInternational conference on ma-
chine learning, pages 19730–19742. PMLR.
Xiang Lisa Li and Percy Liang. 2021. Prefix-tuning:
Optimizing continuous prompts for generation.arXiv
preprint arXiv:2101.00190.
Zhizhong Li and Derek Hoiem. 2017. Learning without
forgetting.IEEE transactions on pattern analysis
and machine intelligence, 40(12):2935–2947.
Vladislav Lialin, Vijeta Deshpande, and Anna
Rumshisky. 2023. Scaling down to scale up: A guide
to parameter-efficient fine-tuning.arXiv preprint
arXiv:2303.15647.
Aixin Liu, Bei Feng, Bing Xue, Bingxuan Wang,
Bochao Wu, Chengda Lu, Chenggang Zhao, Chengqi
Deng, Chenyu Zhang, Chong Ruan, and 1 others.
2024a. Deepseek-v3 technical report.arXiv preprint
arXiv:2412.19437.
Haotian Liu, Chunyuan Li, Qingyang Wu, and Yong Jae
Lee. 2023. Visual instruction tuning.Advances in
neural information processing systems, 36:34892–
34916.
Nelson F Liu, Kevin Lin, John Hewitt, Ashwin Paran-
jape, Michele Bevilacqua, Fabio Petroni, and Percy
Liang. 2024b. Lost in the middle: How language
models use long contexts.Transactions of the Asso-
ciation for Computational Linguistics, 12:157–173.
Alex Mallen, Akari Asai, Victor Zhong, Rajarshi Das,
Hannaneh Hajishirzi, and Daniel Khashabi. 2022.
When not to trust language models: Investigating
effectiveness and limitations of parametric and non-
parametric memories.arXiv preprint.Yuning Mao, Lambert Mathias, Rui Hou, Amjad Alma-
hairi, Hao Ma, Jiawei Han, Scott Yih, and Madian
Khabsa. 2022. Unipelt: A unified framework for
parameter-efficient language model tuning. InPro-
ceedings of the 60th Annual Meeting of the Associa-
tion for Computational Linguistics (Volume 1: Long
Papers), pages 6253–6264.
Junbo Niu, Zheng Liu, Zhuangcheng Gu, Bin Wang,
Linke Ouyang, Zhiyuan Zhao, Tao Chu, Tianyao
He, Fan Wu, Qintong Zhang, Zhenjiang Jin, Guang
Liang, Rui Zhang, Wenzheng Zhang, Yuan Qu, Zhifei
Ren, Yuefeng Sun, Yuanhong Zheng, Dongsheng
Ma, and 42 others. 2025. Mineru2.5: A decoupled
vision-language model for efficient high-resolution
document parsing.Preprint, arXiv:2509.22186.
Jonas Pfeiffer, Aishwarya Kamath, Andreas Rücklé,
Kyunghyun Cho, and Iryna Gurevych. 2021.
Adapterfusion: Non-destructive task composition for
transfer learning. InProceedings of the 16th con-
ference of the European chapter of the association
for computational linguistics: main volume, pages
487–503.
Keshav Santhanam, Omar Khattab, Jon Saad-Falcon,
Christopher Potts, and Matei Zaharia. 2022. Col-
bertv2: Effective and efficient retrieval via
lightweight late interaction. InProceedings of the
2022 Conference of the North American Chapter of
the Association for Computational Linguistics: Hu-
man Language Technologies, pages 3715–3734.
Weijia Shi, Sewon Min, Michihiro Yasunaga, Min-
joon Seo, Richard James, Mike Lewis, Luke Zettle-
moyer, and Wen-tau Yih. 2024. Replug: Retrieval-
augmented black-box language models. InProceed-
ings of the 2024 Conference of the North American
Chapter of the Association for Computational Lin-
guistics: Human Language Technologies (Volume 1:
Long Papers), pages 8371–8384.
Lee Xiong, Chenyan Xiong, Ye Li, Kwok-Fung Tang,
Jialin Liu, Paul Bennett, Junaid Ahmed, and Arnold
Overwijk. 2020. Approximate nearest neighbor neg-
ative contrastive learning for dense text retrieval.
arXiv preprint arXiv:2007.00808.
An Yang, Anfeng Li, Baosong Yang, Beichen Zhang,
Binyuan Hui, Bo Zheng, Bowen Yu, Chang
Gao, Chengen Huang, Chenxu Lv, and 1 others.
2025. Qwen3 technical report.arXiv preprint
arXiv:2505.09388.
Zhilin Yang, Peng Qi, Saizheng Zhang, Yoshua Ben-
gio, William W. Cohen, Ruslan Salakhutdinov, and
Christopher D. Manning. 2018. HotpotQA: A dataset
for diverse, explainable multi-hop question answer-
ing. InConference on Empirical Methods in Natural
Language Processing (EMNLP).
Elad Ben Zaken, Yoav Goldberg, and Shauli Ravfogel.
2022. Bitfit: Simple parameter-efficient fine-tuning
for transformer-based masked language-models. In
11

Proceedings of the 60th Annual Meeting of the As-
sociation for Computational Linguistics (Volume 2:
Short Papers), pages 1–9.
Qingru Zhang, Minshuo Chen, Alexander Bukharin,
Nikos Karampatziakis, Pengcheng He, Yu Cheng,
Weizhu Chen, and Tuo Zhao. 2023. Adalora: Adap-
tive budget allocation for parameter-efficient fine-
tuning.arXiv preprint arXiv:2303.10512.
Tianyi Zhang, Varsha Kishore, Felix Wu, Kilian Q
Weinberger, and Yoav Artzi. 2019. Bertscore: Eval-
uating text generation with bert.arXiv preprint
arXiv:1904.09675.
Lucia Zheng, Neel Guha, Javokhir Arifov, Sarah Zhang,
Michal Skreta, Christopher D Manning, Peter Hen-
derson, and Daniel E Ho. 2025. A reasoning-focused
legal retrieval benchmark. InProceedings of the 2025
Symposium on Computer Science and Law, pages
169–193.
12

A Dataset Details
Table 5 summarizes the datasets used in our study.
For general-domain QA, we report EM on six pub-
lic benchmarks—FreebaseQA (Jiang et al., 2019),
HotpotQA (Yang et al., 2018), Natural Questions
(Kwiatkowski et al., 2019), TriviaQA (Joshi et al.,
2017), WebQuestions (Berant et al., 2013), and
PopQA (Mallen et al., 2022). Because the official
dev/test splits of these benchmarks are substantially
larger than needed to characterize general QA be-
havior under a fixed inference setup, we evaluate
on a single fixed random subset from each dataset’s
official dev/test split and reuse the same subsets
across all methods, ensuring a consistent, repro-
ducible, and benchmark-balanced probe of general
capability. For specialist knowledge injection, we
use two expert benchmarks in immunology adju-
vant and catalytic materials (Anonymous, 2025b,a),
paired with private literature corpora (813 papers
for adjuvants; 986 papers for materials). We eval-
uate specialist QA with BERTScore (Zhang et al.,
2019) computed using SCIBERT (Beltagy et al.,
2019) to better reflect semantic faithfulness in sci-
entific language.
B Additional Implementation Details
All experiments are run on 8 ×NVIDIA A100
GPUs with bfloat16 precision (FlashAttention-2
is enabled when available). We report the com-
plete hyperparameter settings for the two training
stages in Table 6 (Stage I) and Table 7 (Stage II) to
facilitate reproducibility across domains.
For PPR, query representations are obtained by
attention-masked mean pooling over the encoder’s
last-layer states followed by ℓ2normalization. Pro-
totype banks are built using 32 prototypes per do-
main, subsampling up to 10k in-domain queries for
clustering, and settingn_init=10.
Unless otherwise stated, we disable the model’s
explicit “thinking” mode during both training and
evaluation (i.e., we use the non-thinking chat for-
mat). At inference, we apply nucleus/top- ksam-
pling for both background synthesis and answer
generation with temperature 0.7, top- p0.8 , and
top-k20; the maximum generation lengths follow
the benchmark configuration.
C Prompt Templates
To ensure a controlled and reproducible evaluation,
we employ a small set of fixed chat-style prompt
templates across all experiments. The templates areintentionally minimal to reduce incidental prompt
variance, while still enforcing domain-appropriate
behavior for expert routes. Unless otherwise stated,
we disable the model’s explicit “thinking” mode
and use the non-thinking chat format throughout.
General route.Figure 6 shows the prompt used
for the general route, where the frozen base model
LLM baseanswers directly without any domain in-
jection. This template is deliberately concise to
serve as a clean probe of the base model’s general
QA capability under a stable instruction format.
Domain routes.For each private domain i,
prompting is factorized into two stages: (i) a
background-synthesis prompt for the domain ex-
pertLLM domain,i to elicit domain-relevant context
conditioned on the user question, and (ii) an an-
swering prompt for the frozen base model LLM base
that produces the final response. Figures 7 and 8
present the background-synthesis prompts for the
adjuvant and materials experts, respectively. Fig-
ures 9 and 10 present the corresponding answering
prompts forLLM base.
Figure 6:General-route prompt template.The frozen
base model LLM baseanswers the query directly using
a minimal instruction format. Placeholders (shown in
blue) indicate runtime fields (e.g.,<query>).
13

Figure 7:Adjuvant-domain background-synthesis
prompt.Template for the domain expert LLM domain,adj
to generate domain-relevant background knowledge
conditioned on the user question. Placeholders (shown
in blue) are filled at runtime.
Figure 8:Materials-domain background-synthesis
prompt.Template for the domain expert LLM domain,mat .
Placeholders (shown in blue) are filled at runtime.
Figure 9:Adjuvant-domain answering prompt
with GAG injection.Template for the frozen base
model LLM baseunder the adjuvant expert route. The
Knowledge: field denotes the single-token injection
slot in GAG: it is instantiated by a one-token, retrieval-
free injected interface. This contrasts with RAG-style
prompting that serializes chunked top- kevidence into
text and can introduce fragmented or incomplete con-
text. Placeholders (shown in blue) are filled at runtime.
Figure 10:Materials-domain answering prompt with
GAG injection.Template for the frozen base model
LLM baseunder the materials expert route. Placeholders
(shown in blue) are filled at runtime.
14

Category Dataset Split / Size Metric
General QAFreebaseQA (Jiang et al., 2019) Eval: 1135 EM
HotpotQA (Yang et al., 2018) Eval: 1135 EM
Natural Questions (Kwiatkowski et al.,
2019)Eval: 1135 EM
TriviaQA (Joshi et al., 2017) Eval: 1135 EM
WebQuestions (Berant et al., 2013) Eval: 1135 EM
PopQA (Mallen et al., 2022) Eval: 1135 EM
Private expert QAAdjuvant (Anonymous, 2025b)Train: 38,277
Test: 1,135
Corpus: 813 papersBERTScore
(SCIBERT)
Materials (Anonymous, 2025a)Train: 3,661
Test: 646
Corpus: 986 papersBERTScore
(SCIBERT)
Table 5: Dataset summary. “Eval” denotes the number of the evaluated questions used for general QA benchmarks.
Expert benchmarks are paired with private literature corpora and evaluated with BERTScore using SCIBERT.
Hyperparameter Value
Optimizer AdamW (β 1=0.9, β 2=0.999, ϵ=10−8)
Learning rate5×10−6
LR schedule cosine, warmup ratio 0.1
Epochs 8
Per-device train batch size 8
Gradient accumulation 2
GPUs 8
Global train batch size 128
Precision bfloat16
Seed 42
Table 6: Stage I hyperparameters for domain expert adaptation.
Hyperparameter Value
Max sequence length 2048
Optimizer AdamW
Learning rate6×10−3
LR schedule linear, warmup ratio 0.03
Weight decay 0.0
Epochs 5
Per-device train batch size 1
Gradient accumulation 4
GPUs 8
Global train batch size 32
Precision / acceleration bfloat16; FlashAttention-2 when available
Seed 980406
Table 7: Stage II hyperparameters for projector alignment.
15

D RAG Sensitivity to Retrieval Depth
As shown in Table 8, we further examine how re-
trieval depth affects RAG in specialized private
domains by varying k, the number of retrieved pas-
sages concatenated to the base-model prompt. For
brevity, Table 1 reports the best-performing k, and
we provide the full sweep here.
Discussion.Increasing kdoes not monotonically
improve RAG on either domain; in our setting, k=1
is best and largerkslightly degrades performance.
This behavior is consistent with (i) long-context in-
terference, where additional retrieved text increases
distraction and dilutes the signal needed for gener-
ation, and (ii) retrieval brittleness in specialized
corpora, where adding more passages may am-
plify partially mismatched or fragmented evidence
rather than improving reference-critical coverage.
By contrast, GAG avoids prompt-time evidence se-
rialization and provides a constant-budget injection
interface, yielding substantially stronger specialist
QA with more predictable inference behavior.
System Top-kBERTScore↑(SciBERT)
Adjuvant Materials
Base-Model-Only (Qwen3-8B) 0 56.12 60.01
RAG (Qwen3-8B)1 59.97 62.13
3 58.68 61.17
5 58.76 60.87
7 58.27 60.10
9 58.22 60.29
GAG (Gen+Adj+Mat) –69.17 71.36
Table 8:Effect of retrieval depth in RAG.We re-
port BERTScore (SciBERT) on Adjuvant and Materials
when concatenating the top-kretrieved passages.
E Additional Results: Incremental
Multi-domain Routing with PPR
Section 6.2 shows that PPR enables reliable selec-
tive activation in the main routed setting. Here we
further examine incremental multi-domain expan-
sion under a strictly plug-and-play protocol. Start-
ing fromGeneral+Adjuvant, we progressively add
Materials,Aviation(Agarwal et al., 2022),Law
(Zheng et al., 2025), andMath(Cobbe et al., 2021).
Crucially, at each step we construct and load only
the prototype bank of the newly introduced route,
while keeping the query encoder and all previously
deployed prototype banks fully frozen. Routing is
performed by nearest-prototype cosine similarity,
so performance directly reflects how well domain
query manifolds remain separable in a shared em-bedding space without router retraining or thresh-
old tuning.
Route Source / composition #Queries
GeneralFreebaseQA (189)
HotpotQA (196)
Natural Questions (193)
TriviaQA (184)
WebQuestions (183)
PopQA (190)1,135
Adjuvant Adjuvant expert QA 1,135
Materials Materials expert QA 646
Aviation sakharamg/AviationQA 1,135
Law reglabs/housing_qa 1,135
Math openai/gsm8k 1,135
Total—6,321
Table 9:Evaluation pool for incremental multi-
domain routing.The General route is a balanced union
of six public QA subsets (sizes in parentheses).
Evaluation composition.Table 9 summarizes
the evaluation pool (6,321 queries in total) span-
ning six routes. The General route is constructed as
a balanced union of six public QA subsets (1,135
queries), while each specialist route contributes a
domain-specific evaluation set of comparable scale
(except Materials, 646 queries).
#Routes New domain Micro Per-route acc. (%)
acc. (%) Gen Adj Mat Avi Law Math
2 — 99.78 99.65 99.91 — — — —
3 + Mat 99.55 99.65 99.38 99.69 — — —
4 + Avi 99.63 99.47 99.38 99.69 100.00 — —
5 + Law 99.71 99.47 99.38 99.69 100.00 100.00 —
6 + Math 99.72 99.47 99.38 99.69 100.00 100.00 99.74
Table 10:Incremental multi-domain scalability of
PPR.We progressively add new routes by loading only
the corresponding prototype bank (encoder and exist-
ing routes are frozen). Micro acc. is overall routing
accuracy; Per-route acc. is class-wise accuracy for each
active route.
Incremental routing stability.Table 10 reports
routing accuracy as the active route set grows from
2 to 6. PPR exhibits strong scalability under in-
cremental expansion: micro-averaged accuracy re-
mains above 99.5% across all stages, despite re-
peatedly increasing the decision space. Moreover,
per-route accuracy stays uniformly high for both
newly added routes and previously deployed routes,
with no meaningful degradation as new prototype
banks are attached. Together, these results position
PPR as a non-parametric, deployment-friendly rout-
ing interface for modular expert systems: domain
expansion is realized by a lightweight prototype up-
16

date, preserving the stability guarantees of a frozen
base model while maintaining near-saturated rout-
ing reliability at scale.
F More Interesting Cases
Note on visualization.In Figures 11–14, the
“Generated Expert Background” on the GAG side
is shown only as an analysis-time probe for inter-
pretability. In the actual GAG pipeline, the domain
expert model ( LLM domain,i ) doesnotexpose any
such text to users; it produces asingle continuous
embeddingthat is projected into LLM base’s token
space and injected asone token(cf. §4.2). Thus,
the user-visible interface remains constant-budget
and retrieval-free.
Legend (highlight colors).Across Fig-
ures 11–14, green highlights denote ground-
truth-critical key factors, red highlights mark
off-target/mismatched or misleading retrieved
details that can derail RAG, and gray text indicates
irrelevant/noisy content that is not required by the
reference answer.
Case 1 (Adjuvant): noisy retrieval →incomplete
evidence grounding.As shown in Figure 11, this
case illustrates a practical brittleness of RAG in pri-
vate scientific corpora: the retrieved top passage
is partially corrupted and fragmented, so the base
model is forced to answer under incomplete and
unstable evidence support. Even when the RAG
answer captures the coarse direction (Th1 bias),
retrieval noise can suppress explicit coverage of all
reference-critical markers. GAG avoids this failure
mode by decoupling domain knowledge transfer
from snippet quality: a single injected expert to-
ken provides a holistic domain prior in LLM base’s
representational space, enabling more reliable cov-
erage of the key Th1 evidence emphasized by the
ground truth.
Case 2 (Adjuvant): retrieval drift →objec-
tive misalignment.As shown in Figure 12,
this example exposes a high-stakes RAG fail-
ure mode in private scientific corpora: objective-
misaligned evidence can be topically relevant yet
steer generation toward the wrong criterion. The
ground truth is explicitly titer-based (higher env-
specific IgG/IgA under DNA–VLP than VLP–
VLP) and attributes the gap to a coherent prime–
boost mechanism. However, retrieved snippets
foreground adjuvant/mucosal and neutralization-
related details (plus off-target readouts like anti-gag), so the base model over-focuses on epitope
breadth/neutralization narratives and under-serves
the titer-focused comparison the question demands.
GAG avoids this drift by replacing snippet-level
evidence serialization with a representation-level
expert prior: a single injected token encodes the
causal chain needed for the titer claim, delivering
higher intent fidelity under a constant knowledge
budget.
Case 3 (Materials): wrong-entity retrieval →
mechanism collapse.As shown in Figure 13,
mechanism questions are particularly vulnerable
to RAG’s entity mismatch: when retrieval locks
onto an adjacent but different experimental setup
(here, Cu/Au catalyst characterization), the base
model is steered into an evidence-consistent yet
question-inconsistent explanation. Consequently,
RAG shifts to a Cu/Au-specific story and fails to
cover the ground-truth mechanism checklist (trans-
port effects, overlapping-field charge transfer, in-
termediate stabilization, and aggregation/sintering
tradeoffs). GAG mitigates this by injecting a
domain-conditioned representation that is not tied
to a single retrieved entity or paper chunk, allowing
LLM baseto synthesize the intended cross-concept
mechanism and maintain high-level faithfulness
under retrieval mismatch.
Case 4 (Adjuvant; error analysis): a minor
numeric-scale slip.As shown in Figure 14, this
error case reflects a common pattern in scientific
QA: retrieval can directly surface and copy exact
numeric details when they appear verbatim in the
retrieved span. GAG still provides strong proce-
dural correctness (the preparation method aligns
at a high level), but shows a small unit/scale slip
(nm vs. µm) on the mean size. In practice, this
is a lightweight edge case: when exact numeric
fidelity is paramount, simple post-hoc numeric/unit
normalization can be layered on top of the injected
expert signal without changing the core constant-
budget design.
17

Figure 11:Case 1 (Adjuvant): Robust key-factor coverage under noisy/fragmented retrieval.RAG retrieves
a corrupted/noisy snippet, which weakens evidence completeness for reference-critical cytokine signals, while
GAG yields a more faithful Th1 comparison via constant-budget expert-token injection. Green highlights indicate
reference-critical factors.
18

Figure 12:Case 2 (Adjuvant): Titer-critical QA under retrieval drift.The question targets env-specific serum
IgG/IgA titers and their mechanistic drivers. RAG retrieval over-emphasizes Eurocine L3/mucosal framing, anti-gag
signals, and neutralization/epitope-mapping anecdotes, inducing a neutralization-centric answer that drifts from the
requested titer comparison. GAG instead yields a titer-aligned, mechanism-grounded explanation (DNA priming
for CD4 help and broadened naive B-cell recruitment; VLP boosting for strong memory B-cell expansion/affinity
maturation; heterologous clades broadening epitope coverage) via expert-token injection.
19

Figure 13:Case 3 (Materials): Mechanism-level synthesis under wrong-entity retrieval.RAG retrieval centers
on a Cu/Au-XPS characterization narrative, inducing an off-target answer that misses the reference mechanisms
(mass transport, overlapping fields, intermediate stabilization, aggregation effects). GAG recovers the intended
ensemble-interaction mechanisms via one-token injection and explicitly covers the reference factors.
Figure 14:Case 4 (Adjuvant; error analysis): Exact numeric fidelity vs. knowledge transfer.RAG can
precisely copy an explicitly stated mean size from retrieved text, while GAG correctly captures the preparation
procedure but exhibits a minor nm/ µm scale slip on the numeric value. This suggests that lightweight numeric
normalization/verification can complement constant-budget injection when exact numbers dominate the evaluation.
20