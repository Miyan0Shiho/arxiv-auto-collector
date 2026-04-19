# Transforming External Knowledge into Triplets for Enhanced Retrieval in RAG of LLMs

**Authors**: Xudong Wang, Chaoning Zhang, Qigan Sun, Zhenzhen Huang, Chang Lu, Sheng Zheng, Zeyu Ma, Caiyan Qin, Yang Yang, Hengtao Shen

**Published**: 2026-04-14 11:36:29

**PDF URL**: [https://arxiv.org/pdf/2604.12610v1](https://arxiv.org/pdf/2604.12610v1)

## Abstract
Retrieval-Augmented Generation (RAG) mitigates hallucination in large language models (LLMs) by incorporating external knowledge during generation. However, the effectiveness of RAG depends not only on the design of the retriever and the capacity of the underlying model, but also on how retrieved evidence is structured and aligned with the query. Existing RAG approaches typically retrieve and concatenate unstructured text fragments as context, which often introduces redundant or weakly relevant information. This practice leads to excessive context accumulation, reduced semantic alignment, and fragmented reasoning chains, thereby degrading generation quality while increasing token consumption. To address these challenges, we propose Tri-RAG, a structured triplet-based retrieval framework that improves retrieval efficiency through reasoning-aligned context construction. Tri-RAG automatically transforms external knowledge from natural language into standardized structured triplets consisting of Condition, Proof, and Conclusion, explicitly capturing logical relations among knowledge fragments using lightweight prompt-based adaptation with frozen model parameters. Building on this representation, the triplet head Condition is treated as an explicit semantic anchor for retrieval and matching, enabling precise identification of query-relevant knowledge units without directly concatenating lengthy raw texts. As a result, Tri-RAG achieves a favorable balance between retrieval accuracy and context token efficiency. Experimental results across multiple benchmark datasets demonstrate that Tri-RAG significantly improves retrieval quality and reasoning efficiency, while producing more stable generation behavior and more efficient resource utilization in complex reasoning scenarios.

## Full Text


<!-- PDF content starts -->

1
Transforming External Knowledge into Triplets for
Enhanced Retrieval in RAG of LLMs
Xudong Wang, Chaoning Zhang*,Senior Member, IEEE, Qigan Sun, Zhenzhen Huang, Chang Lu, Sheng Zheng,
Zeyu Ma, Caiyan Qin, Yang Yang,Senior Member, IEEE, Hengtao Shen,Fellow, IEEE
Abstract—Retrieval-Augmented Generation (RAG) mitigates
hallucination in large language models (LLMs) by incorporating
external knowledge during generation. However, the effectiveness
of RAG depends not only on the design of the retriever and
the capacity of the underlying model, but also on how retrieved
evidence is structured and aligned with the query. Existing
RAG approaches typically retrieve and concatenate unstructured
text fragments as context, which often introduces redundant or
weakly relevant information. This practice leads to excessive
context accumulation, reduced semantic alignment, and frag-
mented reasoning chains, thereby degrading generation quality
while increasing token consumption. To address these chal-
lenges, we propose Tri-RAG, a structured triplet-based retrieval
framework that improves retrieval efficiency through reasoning-
aligned context construction. Tri-RAG automatically transforms
external knowledge from natural language into standardized
structured triplets consisting of Condition, Proof, and Conclusion,
explicitly capturing logical relations among knowledge fragments
using lightweight prompt-based adaptation with frozen model
parameters. Building on this representation, the triplet head
Condition is treated as an explicit semantic anchor for retrieval
and matching, enabling precise identification of query-relevant
knowledge units without directly concatenating lengthy raw texts.
As a result, Tri-RAG achieves a favorable balance between
retrieval accuracy and context token efficiency. Experimental
results across multiple benchmark datasets demonstrate that
Tri-RAG significantly improves retrieval quality and reasoning
efficiency, while producing more stable generation behavior and
more efficient resource utilization in complex reasoning scenarios.
Index Terms—retrieval-augmented generation (RAG), struc-
tured knowledge, triplet extraction, soft prompt tuning, semantic
retrieval, reasoning
I. INTRODUCTION
Retrieval-Augmented Generation (RAG) equips large lan-
guage models (LLMs) with access to external knowl-
edge, improving factual grounding and supporting reasoning-
intensive applications such as multi-hop question answering,
knowledge-intensive retrieval, and structured decision support
Xudong Wang and Qigan Sun are with the School of Computing, Kyung
Hee University, Yongin-si, South Korea (e-mail: wl200203@khu.ac.kr; sun-
qigan0206@gmail.com). Chaoning Zhang, Chang Lu, Sheng Zheng, Zeyu
Ma, and Yang Yang are with School of Computer Science and Engineer-
ing, University of Electronic Science and Technology of China, Chengdu,
China (email: chaoningzhang1990@gmail.com; george.changlu@gmail.com;
zszhx2021@gmail.com; mazeyu@uestc.edu.cn; yang.yang@uestc.edu.cn).
Zhenzhen Huang is with School of Information and Software Engineering,
University of Electronic Science and Technology of China, Chengdu, China
(email: alley10086@gmail.com). Caiyan Qin is with School of Robotics and
Advanced Manufacture, Harbin Institute of Technology, Shenzhen, China
(email: qincaiyan@hit.edu.cn). Hengtao Shen is with School of Computer
Science and Technology, Tongji University, Shanghai, China (e-mail: shen-
hengtao@hotmail.com).
* Corresponding Author
Fig. 1. Triplet Matching for Efficient Reasoning. The input query is
mapped to a condition representation that is matched against triplet heads
in the knowledge base. The matched triplet (Condition, Proof, Conclusion)
is returned to support structured reasoning with reduced unnecessary context
expansion.
[1]–[4]. While modern LLMs are capable of generating flu-
ent and context-aware responses, their performance in RAG
settings is shaped not only by the retriever and the backbone
model, but also by how retrieved evidence is organized and
presented to the generator [5]. In most existing pipelines, ex-
ternal knowledge is retrieved in the form of raw text segments,
where semantic relations and logical dependencies remain im-
plicit rather than explicitly structured. This mismatch between
query form and evidence organization can weaken query–
evidence alignment and reduce the reliability of downstream
reasoning [6]–[11].
A common RAG workflow retrieves top-kpassages and
concatenates them into a single context window. This
fragment-level retrieval unit can introduce redundancy, di-
lute salient evidence with weakly related content, and yield
contexts that are long yet semantically diffuse. The resulting
context construction may hinder coherent reasoning and in-
crease token usage, which can further amplify computation
and latency costs [12]–[15]. These limitations motivate a
representation-aware view of RAG: improving retrieval is not
only a matter of ranking documents, but also a matter of
organizing knowledge into units that better match the structure
of queries and the needs of reasoning.
Prior work has explored structured processing of external
knowledge to improve its usability within RAG. Represen-
tative directions include template-based extraction, schema-
guided information structuring, and graph-based represen-
tations that organize entities and relations for downstream
retrieval and reasoning. Such approaches can be effective
in constrained settings, but they frequently rely on heuristic
rules, domain-specific assumptions, or manual annotation,
which complicates deployment under diverse natural language
expressions and open-domain corpora [16], [17]. Another linearXiv:2604.12610v1  [cs.CL]  14 Apr 2026

2
of research leverages formal representations and verification
systems [18]–[21] to provide precise semantics and proof
checking. These systems offer strong guarantees, yet their
formal languages differ substantially from natural language,
and building formalized resources typically requires substan-
tial expert effort. This gap makes direct integration with large-
scale RAG pipelines challenging, especially for knowledge
that is expressed in varied, informal, or application-specific
language.
To address these challenges, we propose Tri-RAG, a struc-
tured knowledge transformation retrieval framework tailored
for LLM-based RAG. Tri-RAG converts external knowledge
written in natural language into standardized triplets ofCon-
dition,Proof, andConclusion, enabling retrieval over seman-
tically focused units that expose key relational structure. The
transformation is learned via soft prompt tuning [22]: with the
LLM backbone frozen, a small set of trainable prompt vectors
guides the model to extract explicit triplet representations
from unstructured knowledge sources. During retrieval, Tri-
RAG treats the triplet head (Condition) as a dynamic semantic
anchor for matching and returns the full triplet to support
inference. This design aims to reduce ineffective concatenation
of lengthy raw texts while preserving the information required
for downstream reasoning.
We conduct empirical evaluations of Tri-RAG across mul-
tiple reasoning tasks. The results indicate that Tri-RAG im-
proves retrieval quality and reasoning efficiency, and supports
more stable generation under challenging retrieval conditions.
The main contributions of this work are as follows:
•Tri-RAG is introduced as a structured knowledge trans-
formation retrieval framework, representing external
knowledge as triplets (Condition, Proof, Conclusion) to
enhance retrieval and reasoning in RAG.
•A soft prompt tuning approach for triplet extraction is
presented, which updates only a small set of prompt
parameters while keeping the LLM backbone frozen,
ensuring lightweight and scalable knowledge structuring.
•Tri-RAG is evaluated against representative RAG base-
lines across multiple reasoning benchmarks, demonstrat-
ing that triplet-based retrieval improves both retrieval
quality and end-task performance.
II. RELATEDWORK
A. RAG in Large Language Models
Large Language Models (LLMs) exhibit strong reasoning
and generation capabilities but often suffer from outdated
internal knowledge and limited factual recall, especially in
tasks requiring precise and up-to-date external information
[23]. Retrieval-Augmented Generation (RAG) addresses this
limitation by equipping LLMs with access to external knowl-
edge sources [24]–[26]. A typical RAG pipeline retrieves top-k
relevant documents using dense encoders [27], [28] or sparse
models such as ColBERT [29], and concatenates the retrieved
text with the query for generation [30]. However, using un-
structured text fragments as retrieval units often introduces
redundancy, weak semantic alignment, and irrelevant content,
which can degrade downstream reasoning quality [31], [32].Recent work improves retrieval through reranking [33],
[34], multi-document fusion [35], and dynamic retrieval strate-
gies [36], [37]. Nevertheless, these efforts mainly focus on
retrieval algorithms, while the structural representation and
granularity of retrieved content remain largely unaddressed.
More fundamentally, the lack of explicit structural modeling
hinders semantic alignment between queries and evidence
[38]. In reasoning-intensive tasks such as theorem proving or
multi-hop question answering, plain-text retrieval often does
not explicitly expose logical elements such as assumptions,
conclusions, and intermediate steps, making downstream rea-
soning less stable or harder to control [39], [40].
Beyond text-centric retrieval, related research explores
structured reasoning over knowledge graphs and relational
data. Prior work investigates few-shot and low-resource rea-
soning over knowledge graphs [41], structured graph or hyper-
graph modeling for complex learning systems [42], and reli-
able structure learning for causal reasoning [43]. While these
approaches highlight the benefits of explicit structure, they
typically assume pre-existing structured inputs, whereas Tri-
RAG focuses on transforming unstructured natural language
knowledge into retrieval-ready structured evidence.
B. Soft Prompt Tuning for Large Language Models
Prompt-based learning has emerged as an efficient alter-
native to full-parameter fine-tuning for adapting LLMs. Soft
Prompt Tuning introduces a small set of trainable continuous
prompt vectors prepended to the input sequence while keeping
all model parameters frozen. By optimizing only prompt
parameters, this approach significantly reduces training cost
and memory overhead, and has demonstrated strong perfor-
mance and generalization across models such as BERT [44],
GPT [45], and T5 [46]. Subsequent work further extends
this paradigm to improve expressiveness and adaptability.
P-Tuning [47] combines continuous prompts with discrete
templates, while Prefix-Tuning [44] injects trainable prefix
vectors into Transformer layers to better capture task-specific
semantics. Recent studies also show that prompt-based adap-
tation can be effectively extended to specialized multimodal
scenarios [48]. Together, these advances establish soft prompt
tuning as a practical and scalable strategy for controlling LLM
behavior under parameter-efficient constraints.
However, in many reasoning-intensive domains, knowledge
expressed in natural language often contains implicit struc-
tural semantics, such as premises, intermediate evidence, and
conclusions, that are difficult for existing models to reliably
identify and utilize. Traditional structure extraction methods
typically rely on manually designed rules or templates, which
limit scalability and generalization across diverse natural lan-
guage expressions. To address this challenge, we leverage soft
prompt tuning to guide frozen LLMs toward generating stan-
dardized triplet representations (Condition, Proof, Conclusion)
from unstructured text. By introducing a small set of trainable
prompt vectors, the model can be steered to produce structured
outputs without modifying backbone parameters, enabling
efficient and scalable construction of structured knowledge
resources.

3
III. METHOD
Tri-RAG is a structured knowledge-transformation retrieval
framework designed for retrieval-augmented generation. The
method is organized as follows: The Triplet Schema sec-
tion (III-A) explains how unstructured external knowledge
is transformed into standardized triplets (Condition, Proof,
Conclusion) with semantic constraints. The Triplet Extraction
via Soft Prompt Tuning section (III-B) details triplet extraction
using soft prompt tuning. In the Triplet Knowledge Base Con-
struction section (III-C), a knowledge base indexed by triplet
heads is built. Finally, the Inference: Condition-Anchored Re-
trieval and Full-Triplet Conditioning section (III-D) describes
condition-anchored retrieval and full triplet conditioning for
generation. An overview of the pipeline is shown in Figure 1
and Figure 2.
A. Triplet Schema
To improve the alignability and compositionality of retrieval
units, Tri-RAG re-parameterizes external knowledge from
free-form text into a structured triplet. Given a raw knowledge
unitx(e.g., a paragraph or passage), we map it to:
T= (T c, Tp, Tr),(1)
where each field is defined with explicit semantic constraints:
•Tc(Condition).T cspecifies theapplicability scopeunder
which the claim holds, including entity definitions, con-
textual constraints (time/place/scenario), preconditions,
assumptions, and input restrictions.T cis designed to
provide the minimal context required for reliable query–
knowledge alignment and serves as thetriplet headused
for indexing and retrieval.
•Tp(Proof).T pprovidessupporting rationaleconnecting
TctoTr, such as a concise causal chain, rule-based justi-
fication, key intermediate steps, or evidence description.
When the source text does not contain explicit rationale
or when faithful extraction is not possible, we allow
Tp=∅while preserving structural tags for consistent
serialization.
•Tr(Conclusion).T rstates thecore claimthat holds under
Tc—a self-contained and verifiable proposition (e.g., a
factual statement, rule outcome, operational result, or
decision recommendation).T ravoids extraneous back-
ground not required for the claim, reducing noise and
improving reuse across queries.
a) Interface to retrieval and generation:Tri-RAG de-
couples thematching signalfrom thegeneration evidence: we
retrieve by the triplet headT cto maximize alignment, and
condition generation on the full triplet(T c, Tp, Tr)under a
fixed context budget.
b) Granularity and compositionality:Triplets are ex-
tracted at the same granularity as the downstream retrieval
unit (e.g., paragraph/passage). If a raw unit contains multiple
relatively independent claims, it can be decomposed into
multiple triplets to reduce topic mixing and improve evidence
compositionality.c) Problem Formulation:Given a corpus of raw knowl-
edge unitsC={x j}|C|
j=1, Tri-RAG transforms each unitx jinto
a tripletT j= (T c,j, Tp,j, Tr,j)and builds a triplet knowledge
base:
K={T j= (T c,j, Tp,j, Tr,j)}|K|
j=1.(2)
At inference time, given a queryQ, Tri-RAG retrieves can-
didate triplets by matchingQagainst triplet headsT c,j, and
conditions generation on the retrieved full triplet(s) to produce
the final answerˆy(details in Section III-D).
B. Triplet Extraction via Soft Prompt Tuning
To enable scalable tripletization without full-parameter fine-
tuning, we adopt soft prompt tuning [22] to train a structured
extractor. Letf extdenote an open pre-trained Transformer
used for triplet extraction, withLlayers and hidden sized.
During training, we freeze all backbone parameters off extand
optimize only a small set of continuous prompt embeddings.
a) Layer-wise soft prompts.:Instead of using a single
input-level prompt, we employ layer-wise (deep) soft prompts
to increase extraction capacity while keeping the backbone
frozen. LetS ⊆ {1, . . . , L}denote the set of layers where
prompts are injected, and letL p=|S|. We introduce trainable
prompt embeddings
Esoft=n
E(l)
soft∈Rℓ×do
l∈S,(3)
whereℓis the number of prompt tokens inserted per selected
layer. The number of trainable parameters is thereforeℓ·d·L p.
For each raw knowledge unitx(i), we obtain the initial token
embeddings
H(0)= Φ
x(i)
,(4)
whereΦ(·)denotes tokenization and embedding lookup. At
each Transformer layerl, we prepend the corresponding soft
prompt to the hidden states:
˜H(l−1)=h
E(l)
soft;H(l−1)i
,(5)
and apply the frozen layer transformation
˜H(l)= Trans(l)
˜H(l−1)
.(6)
To keep the sequence length stable across layers, we discard
the prompt positions after each layer and retain only the hidden
states corresponding to the original tokens:
H(l)=˜H(l)[ℓ+ 1 : ].(7)
b) Structured generation target.:The extractor is trained
to generate a fixed structured template:
y(i)= Fmt
T(i)
c, T(i)
p, T(i)
r
,(8)
whereFmt(·)serializes a triplet into a tagged string
[C]T c[P]T p[R]T r. When the source text does not
contain explicit rationale, the model may output an empty
or minimalT(i)
p, which is later normalized during knowledge
base construction (Section III-C).

4
Fig. 2. Soft Prompt–Driven Triplet Structuring and Retrieval-Augmented Inference. The upper part shows how soft prompts guide frozen LLMs to extract
structured triplets (Condition, Proof, Conclusion). The lower part uses triplet retrieval to enhance LLM inference via relevance ranking and head matching.
c) Supervision construction.:We optimize the layer-wise
soft promptsE softon
Dext=n
x(i), T(i)
c, T(i)
p, T(i)
roN
i=1.(9)
If human-annotated triplets are available, we use them di-
rectly. Otherwise, we construct weak supervision aligned with
the downstream corpus: we deriveT ctargets from evidence
annotations as retrieval-oriented semantic anchors, extract
short rationales asT pfrom explanatory sentences or evidence
summaries when available, and defineT ras the core claim
stated in the unit. We use a held-out validation split for
model selection (and early stopping when applicable), and
filter malformed or degenerate outputs during knowledge base
construction (Section III-C).
d) Training objective:Lety(i)denote the target for-
matted sequence. With layer-wise soft prompts, the extractor
defines
Pθ
y(i)|x(i);Esoft
=|y(i)|Y
t=1Pθ
y(i)
t|y(i)
<t, x(i);Esoft
,(10)
and we minimize token-level cross-entropy:
L=−1
NNX
i=1|y(i)|X
t=1logP θ
y(i)
t|y(i)
<t, x(i);Esoft
.(11)
Importantly, gradients are computedonlywith respect toE soft:
∇EsoftL,(12)while all backbone parameters off extremain frozen.
C. Triplet Knowledge Base Construction
We construct a triplet knowledge baseKfrom the extracted
triplets for retrieval and downstream conditioning.
a) Unit definition:We segment the raw corpus into
knowledge units{x j}that match the downstream retrieval
granularity. For each unitx j, the extractor produces:
Tj= (T c,j, Tp,j, Tr,j).(13)
b) Normalization:We standardize tags and surface forms
across fields and apply light text cleaning (e.g., removing
redundant whitespace/punctuation and duplicated tags). We
discard degenerate outputs (e.g., emptyT c,jorT r,j) and
remove samples with abnormal field lengths based on prede-
fined bounds. We also de-duplicate near-identical triplet heads
(Conditions) to reduce index redundancy and improve retrieval
stability.
c) Quality control:We apply a rule-based validator re-
quiring (i) a complete tagged structure and (ii) field lengths
within preset ranges. Optionally, we apply a lightweight con-
fidence filter to remove low-quality outputs.
d) Indexing:We construct the retrieval index exclusively
over the triplet headsT c,j. Specifically, eachT c,jis encoded
into a dense vector representation and inserted into a dense
vector index for similarity-based retrieval at query time. Each
index entry is uniquely associated with its corresponding full

5
Algorithm 1Layer-wise Soft Prompt Tuning for Triplet
Extraction
1:Input:Supervision setD ext =
{(x(i), T(i)
c, T(i)
p, T(i)
r)}N
i=1; extractorf ext (frozen);
layer-wise soft promptsE soft={E(l)
soft}l∈S; learning rate
η; epochsK
2:Output:Optimized soft promptsE⋆
soft
3:# OnlyE softis trainable; all parameters off extare frozen.
4:forepoch= 1toKdo
5:foreach sample(x(i), T(i)
c, T(i)
p, T(i)
r)∈ D extdo
6:y(i)←Fmt(T(i)
c, T(i)
p, T(i)
r)
7:# Forward pass with layer-wise prompt injection.
8:ˆy(i)←f ext 
x(i);Esoft
9:L ←CE(ˆy(i), y(i))
10:g← ∇ EsoftL
11:E soft← E soft−ηg
12:end for
13:end for
14:E⋆
soft← E soft
15:Return:E⋆
soft
tripletT j= (T c,j, Tp,j, Tr,j). Upon retrieval, the complete
triplet is returned as a structured evidence unit and used for
subsequent generation conditioning.
D. Inference: Condition-Anchored Retrieval and Full-Triplet
Conditioning
Given a queryQ, Tri-RAG follows a unified two-stage
inference protocol: retrieval is driven solely by the triplet
headT c, while generation is conditioned on the full triplet
(Tc, Tp, Tr). Letkdenote the retrieval depth,mthe number
of triplets selected for generation conditioning (m≤k), andB
the token budget allocated to the evidence block. The overall
inference procedure is summarized in Algorithm 2.
a) Condition-Anchored Retrieval:During retrieval, Tri-
RAG uses only the triplet headT cas the matching unit.
Specifically, the queryQand each triplet headT c,jare
encoded into a shared vector space using the same retrieval
encoderEnc R(·):
q= Enc R(Q),k j= Enc R(Tc,j).(14)
Candidate triplets are retrieved via similarity search over the
triplet knowledge baseK:
Ktop-k= TopK
q,{k j}|K|
j=1
.(15)
All candidates are ranked according to their similarity scores
between the query and triplet heads, and the topmtriplets
are selected for downstream generation conditioning. No ad-
ditional reranking module is introduced, ensuring a simple and
controlled retrieval pipeline.
b) Evidence Construction and Full-Triplet Conditioning:
For generation, the selectedmtriplets are serialized using a
fixed template and concatenated into an evidence blockC.
Each triplet is represented in a structured textual form with
explicit field tags, namely[C]T c,[P]T p, and[R]T r,
corresponding to the condition, rationale, and conclusionAlgorithm 2Tri-RAG Inference
1:Input:QueryQ; triplet KBK={(T c,j, Tp,j, Tr,j)}|K|
j=1;
retrieval encoderEnc R; retrieval depthk; selected triplets
m(m≤k); token budgetB; generatorf gen
2:Output:Answerˆy
3:q←Enc R(Q)
4:# Retrieval is performed on triplet heads (T c) only.
5:K top-k←TopK
q,{Enc R(Tc,j)}|K|
j=1
6:Select the topmtripletsTby similarity score
7:# Full triplets are used as generation evidence.
8:C←Format(T;B)
9:# Enforce token budget with field-level truncation (trun-
cateT pfirst).
10:x←[Q;C]
11:ˆy←f gen(x)
12:Return:ˆy
fields, respectively. During evidence construction, we enforce
a unified token budgetBmeasured using the tokenizer of the
generation model. If the concatenated evidence exceeds the
budget, truncation is applied strictly within field boundaries to
preserve structural integrity. By default, truncation is applied
to the rationale fieldT pfirst, in order to maximize effective in-
formation density under a fixed context length constraint while
preserving alignment-critical conditions and core conclusions.
The generator input and output are defined as:
x= [Q;C],ˆy=f gen(x),(16)
wheref gendenotes the language model used for answer
generation.
c) Model Usage:Soft prompt tuning requires trainable
prompt embeddings; therefore, triplet extraction is performed
using an open, trainable language model. The generation
modelf genmay be either an open-source model or a closed
API-based LLM, but it is not involved in training or updating
the soft prompts.
IV. EXPERIMENT
A. Experimental Setup
1) Datasets:To comprehensively evaluate the effectiveness
of Tri-RAG in long-context question answering and multi-hop
reasoning, we conduct experiments along two complementary
axes:Long-context and multi-hop QA benchmarks, and Hot-
potQA Robustness Variants.
a) Long-context and multi-hop QA benchmarks:We
evaluate end-to-end answering performance on LongBench
[49] and a set of widely used QA benchmarks, including Hot-
potQA [50], 2WikiMultihopQA [51], MuSiQue [52], Natural
Questions (NQ) [53], and SQuAD [54]. Collectively, these
datasets span single-hop factual QA, compositional multi-hop
reasoning, and reading comprehension with varying context
lengths, enabling assessment of generalization across different
reasoning depths and evidence aggregation patterns.

6
b) HotpotQA Robustness Variants.:We evaluate robust-
ness on four variants derived from HotpotQA [50] that cover
different retrieval scales and prior-knowledge conditions:
•Hotpot-Dist (HotpotQA-Distractor): each query is paired
with a small candidate set (typically 10 documents)
containing supporting and distractor documents.
•Hotpot-Full (HotpotQA-Fullwiki): evidence is retrieved
from the full Wikipedia corpus.
•Shuffle-Hotpot-Dist: an entity-replacement robustness
variant introduced by Zhu et al. [55], where entities in
questions and contexts are replaced with type-consistent
alternatives to suppress memorized factual priors, forcing
reliance on retrieved evidence.
•Shuffle-Hotpot-Full: the entity-replacement counterpart
of Hotpot-Full with the same substitution strategy at full-
corpus scale.
2) Baselines Adopted:We design baselines from three
perspectives: search and reasoning strategies, RAG framework
baselines, and parameter-efficient adaptation settings. This de-
sign aims to make the source of performance gains attributable
and interpretable.
a) Search and reasoning strategies:We evaluate Tri-
RAG under four representative retrieval–reasoning strategies
used as the “Method” dimension: Direct, ReAct [56], Self-Act,
and Aflow. Direct performs single-pass generation without ex-
plicit planning, while ReAct interleaves reasoning and actions
across multiple steps. Self-Act focuses on self-initiated action
selection with iterative evidence completion, and Aflow fol-
lows a staged action-flow procedure for evidence aggregation.
All strategies are evaluated under matched inference settings;
Tri-RAG differs only in evidence representation and the use
of the Condition field as the retrieval key.
b) RAG framework baselines:To compare Tri-RAG
against established retrieval-augmented systems in terms of
retrieval quality and answer quality, we adopt the following
framework-level baselines:
•LLM-only: no retrieval; the model answers solely based
on parametric knowledge.
•Semantic RAG [33]: dense semantic retrieval over un-
structured passages, used as context for generation.
•Hybrid RAG [57]: hybrid sparse and dense retrieval to
improve coverage and relevance.
•LightRAG [58]: a simplified RAG design optimized for
low latency and lightweight deployment.
•GraphRAG [59]: organizes knowledge in graph structures
to support entity/relation chaining in multi-hop reasoning.
•KG2RAG [55]: extends graph-based retrieval by coupling
knowledge-graph reasoning with the RAG pipeline to
improve grounding.
•Tri-RAG: transforms external knowledge into a triplet
representation (Condition, Proof, Conclusion), retrieves
by matching Condition as a semantic anchor, and condi-
tions generation on the full triplet evidence.
c) Parameter-efficient adaptation baselines.:Tri-RAG
depends on a parameter-efficient extractor to produce triplet-
structured evidence at low adaptation cost. To ensure a fair
comparison under a constrained trainable-parameter budget,we evaluate several parameter-efficient adaptation schemes
for the extractor while keeping the backbone model frozen
whenever applicable:
•Pretrained: using the frozen backbone directly without
any task-specific adaptation.
•Prefix tuning [44]: prepending a small number of train-
able continuous prefix vectors to the input sequence,
while leaving all backbone parameters unchanged.
•LoRA [60]: inserting low-rank adaptation modules into
selected layers to enable efficient fine-tuning with a
limited number of trainable parameters.
•Tri-RAG (Ours): our extractor configuration, evaluated
end-to-end against the above baselines on the same
datasets to assess both the effectiveness and parameter
efficiency of structured triplet extraction.
3) Experimental Details:To ensure fairness and repro-
ducibility, we decompose the system into two components:
the generator module and the retrieval–structuring module (Re-
triever + Triplet KB). Key variables are controlled across meth-
ods so that performance differences primarily reflect retrieval
behavior and evidence organization rather than generation-side
configurations.
a) Generator backbones.:We evaluate end-to-end per-
formance using multiple generator backbones, including
GPT-4o-mini, GPT-4.1-nano, Qwen-Long, Qwen-Max, and
Gemma-2-27B. For each dataset, all methods share the same
prompt template, maximum output length, and decoding con-
figuration, ensuring that comparisons isolate the effect of
retrieval and evidence construction.
b) Triplet extraction backbones.:Since triplet extraction
involves trainable parameters, the structured-output extractor
is trained on open-weight backbones. We use DeepSeek-7B,
Llama-3-8B, and Qwen3-8B, following their official tokeniza-
tion and preprocessing pipelines. During training, backbone
parameters are frozen and only parameter-efficient components
(e.g., soft prompts) are optimized, yielding a lightweight
extractor for Triplet KB construction.
c) Training setup.:All experiments are conducted on a
single NVIDIA RTX 4090 GPU. We use AdamW with an
initial learning rate of1×10−4and weight decay of 0.01.
The learning rate is linearly warmed up over the first 10% of
training steps and then linearly decayed. The batch size is 6,
with up to 20 epochs and early stopping based on validation
performance (patience = 5).
d) Retrieval and evaluation.:All methods share the same
retrieval encoder, index implementation, retrieval depthk, and
evidence token budgetB. When the budget is exceeded, Tri-
RAG applies field-aware truncation that prioritizes the Condi-
tion and Conclusion fields, whereas passage-based baselines
use passage-level truncation. Generation is performed with
identical settings across methods. End-to-end QA performance
is evaluated using standard exact-match or F1 metrics, and
structured extraction quality is assessed via a unified LLM-as-
a-Judge protocol (Section IV-D).
B. Performance Evaluation
In this section, we evaluate Tri-RAG on long-context ques-
tion answering and multi-hop reasoning, focusing on effec-

7
TABLE I
PERFORMANCE OF SEARCH STRATEGIES(DIRECT, REACT, SELF-ACT, AFLOW,ANDTRI-RAG)ACROSS FIVEQABENCHMARKS. ALL METRICS ARE
REPORTED ASF1SCORES(%).
Model Method LongBench 2WikiMultihopQA MuSiQue NQ SQuAD A VG
GPT-4o-miniDirect 58.1 48.9 40.1 36.4 51.2 46.9
ReAct 62.7 52.3 53.6 48.9 55.3 54.6
Self-Act 68.9 69.2 67.3 62.4 69.5 67.5
Aflow 61.0 65.7 71.6 67.9 73.7 68.0
Tri-RAG 70.9 73.3 74.6 69.8 75.3 72.8
GPT-4.1-nanoDirect 56.3 45.3 38.2 33.7 49.3 44.6
ReAct 59.7 48.9 47.3 45.4 50.9 50.4
Self-Act 64.5 62.2 61.1 59.8 62.7 62.1
Aflow 66.1 70.1 69.4 66.5 72.1 68.8
Tri-RAG 68.2 73.5 71.2 69.8 77.1 72.0
Qwen-LongDirect 59.3 51.9 55.4 43.4 54.3 52.9
ReAct 64.2 57.2 58.9 56.3 58.7 59.1
Self-Act 68.3 73.1 72.3 68.9 70.4 70.6
Aflow 70.8 76.8 78.9 74.6 77.1 75.6
Tri-RAG 73.5 78.3 81.3 78.7 79.2 78.2
Qwen-MaxDirect 47.2 46.6 49.9 53.3 48.5 49.1
ReAct 52.3 53.1 53.6 59.8 54.5 54.7
Self-Act 58.7 64.8 61.8 67.8 68.3 64.3
Aflow 61.2 68.1 65.3 73.9 74.0 68.5
Tri-RAG 62.8 70.9 68.3 78.8 76.5 71.5
Gemma-2-27BDirect 49.3 43.4 48.4 51.3 48.5 48.2
ReAct 52.5 50.8 51.7 57.8 54.5 53.5
Self-Act 56.4 59.7 59.7 65.3 65.7 61.4
Aflow 60.8 63.2 63.4 69.4 71.9 65.7
Tri-RAG 63.3 67.9 67.1 72.5 75.8 69.3
tiveness, reliability, and robustness to contextual noise. The
evaluation spans three dimensions: end-to-end answer qual-
ity, retrieval quality, and robustness under perturbations. We
examine whether the gains of Tri-RAG persist across diverse
generator backbones and representative search or reasoning
strategies, mitigating reliance on specific model families or
inference heuristics. We further analyze a controlled multi-hop
benchmark that allows retrieval and generation to be assessed
separately, enabling a clearer connection between evidence
selection and answer correctness.
Table I summarizes end-to-end F1 scores on five QA
benchmarks under multiple generator backbones, including
GPT-4o-mini, GPT-4.1-nano, Qwen-Long, Qwen-Max, and
Gemma-2-27B. To ensure methodological consistency, all
methods use the same prompt template, output-length budget,
and decoding configuration for each dataset; consequently,
performance differences are attributable primarily to retrieval
behavior and evidence organization rather than generation-side
tuning. Across all backbones, Tri-RAG attains the strongest
average performance and remains consistently competitive on
each benchmark. For GPT-4o-mini, Tri-RAG improves the
average F1 from 67.5% to 72.8% relative to the best competing
strategy, while achieving 73.3% on 2WikiMultihopQA and
74.6% on MuSiQue. The trend persists for GPT-4.1-nano,
where the average F1 increases from 68.8% to 72.0% and per-
formance on 2WikiMultihopQA rises from 70.1% to 73.5%.
Comparable gains are observed for open-weight long-context
models. On Qwen-Long, Tri-RAG increases the average F1
from 75.6% to 78.2%; on Qwen-Max and Gemma-2-27B, theaverage F1 increases from 68.5% to 71.5% and from 65.7%
to 69.3%, respectively. Collectively, these results indicate that
the benefits of Tri-RAG are not contingent on a particular
generator backbone or task distribution, and instead generalize
across model capacities and benchmark characteristics.
End-to-end metrics, however, do not by themselves iden-
tify the source of improvement. We therefore perform a
mechanism-oriented evaluation on HotpotQA, a standard
multi-hop benchmark with annotated supporting evidence,
which supports a direct assessment of both retrieval and
answering. Table II reports response quality and retrieval
quality on four HotpotQA variants. Hotpot-Dist and Hotpot-
Full correspond to the standard distractor and full-wiki set-
tings, respectively. Shuffle-Hotpot-Dist and Shuffle-Hotpot-
Full apply an entity-replacement perturbation, where entities
in both questions and contexts are substituted with type-
consistent alternatives. This perturbation weakens direct entity-
specific cues and provides a controlled setting for assessing
robustness under reduced entity-level priors.
Tri-RAG yields consistent improvements in response quality
in both standard and perturbed settings. On Hotpot-Dist, Tri-
RAG achieves an answer F1 of 67.8%, exceeding KG2RAG at
66.3% and Semantic RAG at 61.7%. On Hotpot-Full, Tri-RAG
reaches 64.2%, again outperforming the strongest baseline
at 63.1%. Under perturbation, the advantage remains, with
Tri-RAG attaining 55.9% on Shuffle-Hotpot-Dist and 51.7%
on Shuffle-Hotpot-Full, compared with 54.5% and 50.7%
for KG2RAG. This indicates that Tri-RAG exhibits reduced
sensitivity to contextual shuffling and maintains more stable

8
TABLE II
PERFORMANCE OF DIFFERENT METHODS ONHOTPOTQADATASETS AND THEIR RESPECTIVE RESPONSE AND RETRIEVAL QUALITIES.
MethodsHotpot-Dist Hotpot-Full Shuffle-Hotpot-Dist Shuffle-Hotpot-Full
F1 Precision Recall F1 Precision Recall F1 Precision Recall F1 Precision Recall
Response Quality
LLM-only 23.7 25.9 23.4 23.7 25.9 23.4 15.8 17.5 15.8 15.8 17.5 15.8
Semantic RAG 61.7 64.6 64.3 52.8 55.8 53.5 50.8 53.3 52.4 42.2 44.9 43.3
+ Rerank 65.2 68.5 66.5 58.7 61.3 60.3 53.2 56.0 54.6 44.7 47.6 45.6
Hybrid RAG 65.3 67.6 65.5 55.1 58.2 55.8 52.0 54.8 53.4 44.3 47.3 44.6
LightRAG 29.3 28.8 48.0 26.1 25.9 36.4 28.5 28.4 40.4 20.2 19.9 29.3
GraphRAG 40.0 40.8 49.1 16.9 15.7 42.9 35.1 36.5 40.1 16.3 15.5 36.2
KG2RAG 66.3 69.0 68.3 63.1 66.5 64.3 54.5 57.2 56.6 50.7 53.9 51.2
Tri-RAG(Ours) 67.8 69.5 69.3 64.2 67.8 64.8 55.9 58.5 57.8 51.7 55.6 52.4
Retrieval Quality
Semantic RAG 34.3 20.6 89.4 30.0 17.8 79.0 32.1 20.1 83.7 26.8 16.7 70.8
+ Rerank 35.7 22.4 93.2 30.6 19.7 83.3 33.9 21.3 88.6 28.6 17.9 75.4
Hybrid RAG 35.4 22.2 92.1 30.2 18.9 79.5 33.4 21.0 83.7 27.9 17.4 73.9
LightRAG 23.4 15.0 63.8 13.2 8.3 34.0 22.7 14.8 53.5 11.6 7.3 29.5
GraphRAG 25.5 16.7 59.4 18.0 11.3 47.0 21.0 13.8 48.2 19.9 12.6 51.0
KG2RAG 43.6 30.1 90.8 31.0 20.3 83.8 40.5 27.9 84.0 30.5 19.3 79.0
Tri-RAG(Ours) 44.1 31.9 91.4 32.0 21.4 83.1 41.9 28.9 85.7 31.5 20.5 79.2
TABLE III
COMPARISON OF SYSTEM COST FOR STRUCTURED KNOWLEDGE CONSTRUCTION AND QUERY-TIME INFERENCE.
Method Knowledge construction Inference
Input tokens Output tokens LLM calls Time Retrieval Generation
LightRAG 1,269 381 1 3 s 40 ms 5,600 ms
GraphRAG 2,791 629 5 6 s 42 ms 5,500 ms
KG2RAG 561 22 1 1 s 25 ms 2,300 ms
Tri-RAG (Ours) 640 96 1 1 s 18 ms 1,900 ms
performance in the presence of noisy or reordered evidence.
Retrieval-quality comparisons provide a direct account of
this behavior. On Hotpot-Dist, Tri-RAG achieves retrieval
precision of 31.9% and retrieval F1 of 44.1%, improving upon
KG2RAG at 30.1% and 43.6% while maintaining high recall
at 91.4%. On Shuffle-Hotpot-Dist, Tri-RAG remains superior
in retrieval F1, with 41.9% compared to 40.5% for KG2RAG.
The combination of high recall with improved precision sug-
gests that Tri-RAG does not rely on narrowing retrieval to
gain apparent relevance; instead, it increases evidence purity
while preserving coverage, thereby mitigating context-budget
dilution and reducing distracting content during generation.
These observations are consistent with the Tri-RAG design:
Condition-anchored retrieval constrains matching and reduces
drift, while full-triplet conditioning supplies structured, ver-
ifiable evidence for downstream reasoning. Taken together,
Table I establishes cross-backbone effectiveness under diverse
inference strategies, and Table II provides a diagnostic link
from improved evidence selection to improved answer quality
and robustness under perturbations.
C. System Cost Analysis: From Construction to Inference
To examine whether structured evidence introduces addi-
tional system overhead, we conduct a cost-diagnostic study
that decomposes the pipeline into two stages: offline knowl-edge construction (transforming raw text into structured ev-
idence units) and online inference (query-time retrieval and
generation under a unified serving setting). Table III sum-
marizes the results, enabling a direct assessment of whether
Tri-RAG’s efficiency gains come at the expense of extra
preprocessing cost or increased latency.
a) Offline construction.:We report per-unit input/output
tokens, the number of LLM calls, and the wall-clock time
required to construct one knowledge unit. GraphRAG incurs
the highest fixed cost, as it relies on multi-round LLM inter-
actions to generate and complete graph structure (5 calls on
average), resulting in 2,791/629 input/output tokens and about
6 s per unit. LightRAG uses a single call but still requires long-
context prompting, with 1,269/381 tokens per unit. In contrast,
Tri-RAG constructs compact triplet units with a single call,
requiring 640/96 tokens and about 1 s per unit. Its construction
cost is close to KG2RAG (561/22 tokens; 1 call; about 1 s),
indicating that tripletization can be achieved with a controlled,
lightweight preprocessing budget rather than a multi-round or
long-context construction pipeline.
b) Online inference.:We further measure average re-
trieval and generation time per query. Tri-RAG reduces re-
trieval latency to 18 ms, compared with 25 ms for KG2RAG
and 40–42 ms for LightRAG/GraphRAG, consistent with using
Conditionas a tight semantic key that shrinks the matching
space and limits low-utility candidates. On the generation

9
TABLE IV
END-TO-ENDF1 (%)ON MULTI-HOPQABENCHMARKS FOR DIFFERENT EXTRACTOR BACKBONES UNDER PARAMETER-EFFICIENT TUNING METHODS.
MethodDeepSeek-7B Llama-3-8B Qwen3-8BAvg.
Hotpot 2Wiki MuSiQue Hotpot 2Wiki MuSiQue Hotpot 2Wiki MuSiQue
Pretrained 49.8 46.5 44.1 52.1 48.7 46.0 53.5 50.2 48.4 48.8
Prefix Tuning 56.7 52.8 50.3 58.9 55.1 52.2 60.1 58.4 55.6 55.6
LoRA 63.9 60.4 57.1 66.2 62.9 59.5 68.4 67.1 64.8 63.4
Tri-RAG (SP) 66.8 63.7 60.2 69.0 73.3 74.6 72.5 71.8 70.4 69.6
side, Tri-RAG achieves 1,900 ms, improving over KG2RAG
(2,300 ms;∼17% reduction) and substantially outperform-
ing LightRAG/GraphRAG (5,600/5,500 ms;∼65% reduction).
This gap aligns with the evidence organization effect: triplet
evidence provides clearer boundaries and higher information
density, reducing the fraction of the context window occupied
by redundant or weakly relevant fragments.
Overall, the study supports the interpretation that structure
does not necessarily imply higher cost. When structured units
expose an explicit retrieval anchor and maintain compact
evidence boundaries, both offline construction (single-call,
moderate token budgets) and online inference (lower retrieval
and generation latency) can benefit concurrently, yielding a
more deployable quality–efficiency trade-off for reasoning-
intensive settings.
D. Structured Extraction and Efficiency
Tri-RAG incorporates a structured pre-processing stage that
compresses and reorganizes unstructured external knowledge
units into retrievable and compositional triplet evidence. We
therefore analyze its quality–cost trade-off along three com-
plementary axes: the adaptation strategy used for structured
extraction, the end-to-end downstream gains on multi-hop
question answering, and the associated system overhead. Table
IV compares representative adaptation paradigms for triplet
extraction on multiple open-weight backbones and evaluates
how each choice affects downstream multi-hop QA. Across
HotpotQA, 2WikiMultihopQA, and MuSiQue, Tri-RAG yields
the best or consistently superior average performance. The
trend is stable across backbones, indicating that effective
structured extraction can be learned under a strictly parameter-
efficient regime: the backbone remains frozen and only a soft
prompt is optimized. Importantly, the improvements are not
confined to extraction accuracy in isolation, but translate into
measurable gains in retrieval-augmented reasoning end to end.
This evidence reduces the plausibility of attributing the gains
primarily to heavy parameter updates, and instead supports the
interpretation that the dominant benefit comes from improved
retrieval alignment and evidence organization enabled by the
triplet representation.
To jointly examine extraction quality and system efficiency,
Table V reports triplet-quality indicators alongside cost-related
metrics. Tri-RAG achieves leading or near-leading scores on
logical faithfulness, structural integrity, and condition align-
ment, while substantially reducing both the average context-
token budget and generation-time latency with only a minimalTABLE V
TRIPLET EXTRACTION QUALITY(1–5,MEAN±STD)AND EFFICIENCY
(TRAINABLE PARAMETERS)FORPREFIXTUNING, LORA,ANDTRI-RAG.
Metric Prefix Tuning LoRA Tri-RAG
Quality Metrics
Logical Faithfulness 4.12±0.35 4.41±0.124.48±0.18
Structural Integrity 3.75±0.82 4.75±0.104.82±0.05
Condition Alignment 3.98±0.44 4.51±0.234.55±0.21
Efficiency Metrics
Params 4.3 M 43.1 M3.7 M
number of trainable parameters. This combined pattern indi-
cates that triplet-structured evidence improves not only the
reliability of provided support, but also end-to-end efficiency.
By increasing evidence density, compressing redundant con-
text, and limiting the portion of the context window occupied
by low-utility retrieved fragments, Tri-RAG yields a more
favorable quality–cost trade-off. Conceptually, it uses explicit
structure to enable stronger compression and uses better align-
ment to promote more stable reasoning, which is particularly
beneficial under tight long-context budgets or latency-sensitive
deployments.
Since Tri-RAG converts unstructured passages into triplets
(Tc, Tp, Tr), we evaluate theintrinsic precisionof structured
evidence independently of downstream QA accuracy. Intrinsic
precision is operationalized along three complementary dimen-
sions that capture the core requirements of the triplet repre-
sentation:faithfulness, which assesses whether the extracted
proof and conclusion are supported by the source passage;
integrity, which evaluates whether the output conforms to a
stable and usable triplet structure with well-defined functional
roles; andalignment, which measures whether the condition
field provides a reliable and compact semantic anchor for
query matching. The corresponding results are reported in
Table V.
All three metrics are computed using a unifiedLLM-as-a-
Judgeprotocol. The evaluator employs a fixed judging prompt
and remains strictly frozen throughout evaluation. To ensure
comparability across methods, we use the same evaluation
prompt template, input serialization format, and decoding
configuration for all settings. The evaluator is constrained to
output a discrete integer score on a 5-point scale (1–5), without
generating free-form explanations. Deterministic decoding is
adopted to eliminate sampling-induced variance. Each instance

10
all Tp Tr Tc0.81.01.21.4ms1.3596
0.9830
0.9387
0.8871Retrieval Time
all Tp Tr Tc114116118120122MB120.8
116.2115.7
114.9Memory Usage
all Tp Tr Tc7071727374%73.80
70.90
70.5072.30Hit@1
Fig. 3. Retrieval-key ablation comparing ALLretrieval using the concatena-
tion[T c;Tp;Tr]and single-field retrieval using TC, TP, or TR. We report
Hit@1 together with index-level retrieval latency and memory footprint.
is evaluated once, and the “±” in Table V denotes the stan-
dard deviation across evaluated instances. For each evaluation
instance, the judge receives the source passage, the extracted
triplet serialized with explicit[C](Condition),[P](Proof),
and[R](Conclusion) tags, and, for condition alignment only,
the corresponding queryq. To avoid systematic bias caused
by length differences, all inputs are processed using the same
formatting and truncation rules prior to evaluation.
E. Ablation Study
To isolate how theretrieval keyaffects matching quality
and system cost, we perform a retrieval-key ablation under
a fully controlled configuration. Across all settings, we keep
the retrieval encoder, index implementation, retrieval depth,
and indexing hyperparameters identical, and varyonlythe
text used to form the retrieval key. Specifically, we compare
ALL, which retrieves using the concatenation[T c;Tp;Tr],
against three single-field variants that retrieve using onlyT c
(TC),T p(TP), orT r(TR). We report top-1 retrieval hit
rate (Hit@1) together with index-level retrieval latency and
memory footprint, capturing the trade-off between alignment
quality and computational overhead. Hit@1 is a retrieval-layer
metric that directly reflects query-key alignment.
The results in Fig. 3 reveal a clear quality–efficiency sep-
aration among retrieval keys. All attains the highest Hit@1
of 73.8%, but also incurs the largest retrieval latency (around
1.360 ms) and the highest memory footprint (120.8 MB). In
contrast, TCoffers the best overall balance, achieving a Hit@1
of 72.3% while reducing retrieval latency to 0.887 ms and
memory footprint to 114.9 MB. Using TPor TRas the retrieval
key further decreases Hit@1 to 70.9% and 70.5%, respectively,
while yielding higher latency and larger memory footprint
than TC. Overall, these results suggest that incorporating
longer and stylistically diverse fields into the retrieval key
does not necessarily improve matching, but can increase both
computational and memory overhead.
This behavior is consistent with the functional roles of the
triplet fields.T cis explicitly constructed to encode the appli-
cability scope and constraints of a claim in a compact, query-
aligned form, making it a stable and discriminative anchor
for retrieval. By contrast,T ptypically contains explanatory
language with higher variability in length and expression,
which weakens semantic targeting when used as a key.T r
is often concise but lacks the constraint information needed
for precise query alignment, making it prone to semantic
drift when retrieved in isolation. Taken together, the ablationsupports our design choice to anchor retrieval onT cfor
efficient and reliable matching, while reserving full-triplet
conditioning for downstream reasoning and traceability.
F . Case Studies
To illustrate the retrieval–reasoning–generation loop of Tri-
RAG, we present a representative two-hop example. The query
exhibits an explicit dependency chain: it first resolves the
relationwork→authorto obtain an intermediate entity, and
then queriesauthor→birth yearto produce the final answer.
This setup is well-suited for analyzing how early retrieval
decisions affect downstream reasoning, since an incorrect
intermediate entity typically leads to an irrecoverable failure
at the next hop.
Tri-RAG Inference Example
Source:2Wiki
Question:What is the birth year of the author ofHarry
Potter?
Tri-RAG Protocol:Retrieve by triplet head[C](Condition),
then condition generation on the retrieved full triplet(s)
[C,P,R].
Tri-RAG Trace:
(Head Retrieval; Hop 1)
Query intent: (Harry Potter, author,
?) # match against [C] only
Hit A: [C] work={Harry Potter};
rel=author [R] author(Harry
Potter)=J.K. Rowling
(Head Retrieval; Hop 2)
Intermediate: J.K. Rowling
Query intent: (J.K. Rowling,
birth_year, ?) # match against [C]
only
Hit B: [C] subj={J.K. Rowling};
attr=birth_year [R] birth_year(J.K.
Rowling)=1965
(Full-Triplet Conditioning)#
generator sees full structured
evidence
Input:[Q; Format(A,B)]
Final Answer:1965
Conventional passage-based RAG retrieves long spans by
semantic similarity, which often introduces co-occurrence
noise (e.g., multiple entities, several dates, or loosely related
biographical facts). Such clutter increases ambiguity during
generation and can lead to spurious attribution, especially in
multi-hop settings where early errors propagate and contami-
nate subsequent hops. In addition, passage retrieval typically
mixes the matching signal and the generation evidence: the
model must both infer which parts of a passage justify an
answer and decide whether the passage is even relevant,
making reasoning less controllable and harder to audit.

11
Tri-RAG mitigates this issue via Condition-anchored re-
trieval. At each hop, matching is restricted to the Condition
field, which provides a more regular and tightly scoped
retrieval key. In this example, the first hop retrieves triplets
aligned with thework→authorintent to yield the intermediate
entity, and the second hop uses that entity as the next condi-
tion anchor to retrieve the birth year evidence. This design
reduces retrieval drift and limits low-utility context injection,
while enabling a clearer separation between (i) retrieval-time
alignment and (ii) generation-time justification.
During generation, Tri-RAG conditions on the full re-
trieved triplets rather than the matched field alone, presenting
premises, supporting rationale, and conclusions in an explicit
structured format. This formatting encourages the generator
to rely on compact, verifiable evidence rather than incidental
cues from verbose passages, improving both reliability and
traceability. Importantly, even when multiple candidate entities
or dates co-occur in the raw corpus, the structured evidence
clarifies which entity is introduced as the intermediate result
and which attribute is queried at the next hop, thereby stabi-
lizing multi-step reasoning.
V. CONCLUSION
This paper proposes a structured output method that lever-
ages Soft Prompt Tuning to transform natural language ex-
pressions into standardized triplets consisting of a Condition,
a Proof, and a Conclusion. By integrating trainable soft
prompts into the input, the method enables large language
models to generate structured representations without fine-
tuning the model’s core parameters. This triplet-based ap-
proach provides a robust foundation for downstream tasks such
as knowledge retrieval, automated reasoning, and complex
question answering. It not only improves the efficiency and
accuracy of structure modeling, but also offers a scalable
solution for constructing large-scale structured knowledge
corpora from unstructured inputs. Empirical results show that
our method consistently outperforms rule-based and template-
based baselines in both structural consistency and seman-
tic alignment, demonstrating its effectiveness in enhancing
reasoning-oriented knowledge extraction. As a next step, we
plan to extend the triplet format to accommodate more com-
plex reasoning structures and broader application domains.
VI. LIMITATIONS
Tri-RAG relies on stable structured extraction to ensure
reliable retrieval and evidence auditability. In practice, we
observe that extracting well-formed triplets remains challeng-
ing for certain types of inputs, particularly narrative-heavy
passages and cases where key premises or conclusions are
distributed across multiple sentences. In such settings, the
boundaries between condition, rationale, and conclusion may
be less explicit, which can lead to incomplete or ambiguous
triplet representations.
Future work could refine Tri-RAG at both extraction and
retrieval. On the extraction side, self-verification or consis-
tency checks may help validate triplets against the source text,improving output stability. On the retrieval side, complemen-
tary signals (e.g., multi-granular or adaptive representations)
could better capture implicit cross-sentence evidence while
preserving the triplet schema, improving robustness without
sacrificing efficiency or interpretability.
ACKNOWLEDGMENTS
This work was supported by the National Natural Science
Foundation of China (NSFC) under the General Program
(Grant No. 62572104).
This article used large language models (such as ChatGPT)
as an auxiliary tool in the language polishing process, but did
not use them in research conception and academic content
generation.
REFERENCES
[1] P. Dixit and T. Oates, “Sbi-rag: Enhancing math word problem solving
for students through schema-based instruction and retrieval-augmented
generation,”arXiv preprint arXiv:2410.13293, 2024.
[2] Z. Levonian, C. Li, W. Zhu, A. Gade, O. Henkel, M.-E. Postle, and
W. Xing, “Retrieval-augmented generation to improve math question-
answering: Trade-offs between groundedness and human preference,”
arXiv preprint arXiv:2310.03184, 2023.
[3] P. Zhao, H. Zhang, Q. Yu, Z. Wang, Y . Geng, F. Fu, L. Yang, W. Zhang,
J. Jiang, and B. Cui, “Retrieval-augmented generation for ai-generated
content: A survey,”arXiv preprint arXiv:2402.19473, 2024.
[4] H. Li, C. Li, W. Xing, S. Baral, and N. Heffernan, “Automated feedback
for student math responses based on multi-modality and fine-tuning,” in
Proceedings of the 14th learning analytics and knowledge conference,
2024, pp. 763–770.
[5] J. Zhang, C. Zhang, S. Chen, Z. Huang, P. Zheng, Z. Wang, P. Guo,
F. Mo, S.-H. Bae, J. Zouet al., “Lightweight llm agent memory with
small language models,”arXiv preprint arXiv:2604.07798, 2026.
[6] Y . Wu, A. Q. Jiang, W. Li, M. Rabe, C. Staats, M. Jamnik, and
C. Szegedy, “Autoformalization with large language models,”Advances
in neural information processing systems, vol. 35, pp. 32 353–32 368,
2022.
[7] D. Ferreira and A. Freitas, “Premise selection in natural language
mathematical texts,” inProceedings of the 58th Annual Meeting of the
Association for Computational Linguistics, 2020, pp. 7365–7374.
[8] W. Chen, M. Yin, M. Ku, P. Lu, Y . Wan, X. Ma, J. Xu, X. Wang,
and T. Xia, “Theoremqa: A theorem-driven question answering dataset,”
arXiv preprint arXiv:2305.12524, 2023.
[9] K. Yang, G. Poesia, J. He, W. Li, K. Lauter, S. Chaudhuri, and D. Song,
“Formal mathematical reasoning: A new frontier in ai,”arXiv preprint
arXiv:2412.16075, 2024.
[10] J. Zhang, C. Zhang, S. Chen, X. Wang, Z. Huang, P. Zheng, S. Yuan,
S. Zheng, Q. Sun, J. Zouet al., “Learning global hypothesis space for en-
hancing synergistic reasoning chain,”arXiv preprint arXiv:2602.09794,
2026.
[11] J. Zhang, Q. Sun, C. Zhang, X. Wang, Z. Huang, Y . Zhou, P. Zheng,
C.-l. A. Tai, S.-H. Bae, Z. Maet al., “Tda-rc: Task-driven alignment
for knowledge-based reasoning chains in large language models,”arXiv
preprint arXiv:2604.04942, 2026.
[12] J. Zhang, C. Zhang, S. Chen, Y . Liu, C. Li, Q. Sun, S. Yuan, F. D.
Puspitasari, D. Han, G. Wanget al., “Text summarization via global
structure awareness,”arXiv preprint arXiv:2602.09821, 2026.
[13] P. Zheng, C. Zhang, J. Mo, G. Li, J. Zhang, J. Zhang, S. Cao,
S. Zheng, C. Qin, G. Wanget al., “Llava-fa: Learning fourier ap-
proximation for compressing large multimodal models,”arXiv preprint
arXiv:2602.00135, 2026.
[14] S. Cao, J. Zhang, P. Zheng, J. Yan, C. Qin, Y . Ye, W. Dong, P. Wang,
Y . Yang, and C. Zhang, “Language-guided token compression with
reinforcement learning in large vision-language models,”arXiv preprint
arXiv:2603.13394, 2026.
[15] P. Zheng, X. Pu, K. Chen, J. Huang, M. Yang, B. Feng, Y . Ren,
J. Jiang, C. Zhang, Y . Yanget al., “Joint lossless compression and
steganography for medical images via large language models,”arXiv
preprint arXiv:2508.01782, 2025.

12
[16] A. Amini, S. Gabriel, P. Lin, R. Koncel-Kedziorski, Y . Choi, and H. Ha-
jishirzi, “Mathqa: Towards interpretable math word problem solving with
operation-based formalisms,”arXiv preprint arXiv:1905.13319, 2019.
[17] Y . Gao, Y . Xiong, X. Gao, K. Jia, J. Pan, Y . Bi, Y . Dai, J. Sun,
H. Wang, and H. Wang, “Retrieval-augmented generation for large
language models: A survey,”arXiv preprint arXiv:2312.10997, vol. 2,
2023.
[18] L. De Moura, S. Kong, J. Avigad, F. Van Doorn, and J. von Raumer, “The
lean theorem prover (system description),” inInternational Conference
on Automated Deduction. Springer, 2015, pp. 378–388.
[19] G. Bancerek, C. Byli ´nski, A. Grabowski, A. Korniłowicz, R. Ma-
tuszewski, A. Naumowicz, and K. P ˛ ak, “The role of the mizar mathe-
matical library for interactive proof development in mizar,”Journal of
Automated Reasoning, vol. 61, no. 1, pp. 9–32, 2018.
[20] B. Barras, S. Boutin, C. Cornes, J. Courant, J.-C. Filliatre, E. Gimenez,
H. Herbelin, G. Huet, C. Munoz, C. Murthyet al., “The coq proof
assistant reference manual: Version 6.1,” Ph.D. dissertation, Inria, 1997.
[21] S. Xu, Z. Wu, X. Jia, Y . Wang, K. Liu, and A. X. Dong, “Self-correcting
rag: Enhancing faithfulness via mmkp context selection and nli-guided
mcts,” 2026. [Online]. Available: https://arxiv.org/abs/2604.10734
[22] B. Lester, R. Al-Rfou, and N. Constant, “The power of scale for
parameter-efficient prompt tuning,”arXiv preprint arXiv:2104.08691,
2021.
[23] A. Roberts, C. Raffel, and N. Shazeer, “How much knowledge can
you pack into the parameters of a language model?”arXiv preprint
arXiv:2002.08910, 2020.
[24] P. Lewis, E. Perez, A. Piktus, F. Petroni, V . Karpukhin, N. Goyal,
H. Küttler, M. Lewis, W.-t. Yih, T. Rocktäschelet al., “Retrieval-
augmented generation for knowledge-intensive nlp tasks,”Advances in
neural information processing systems, vol. 33, pp. 9459–9474, 2020.
[25] K. Guu, K. Lee, Z. Tung, P. Pasupat, and M. Chang, “Retrieval
augmented language model pre-training,” inInternational conference
on machine learning. PMLR, 2020, pp. 3929–3938.
[26] S. Borgeaud, A. Mensch, J. Hoffmann, T. Cai, E. Rutherford, K. Milli-
can, G. B. Van Den Driessche, J.-B. Lespiau, B. Damoc, A. Clarket al.,
“Improving language models by retrieving from trillions of tokens,” in
International conference on machine learning. PMLR, 2022, pp. 2206–
2240.
[27] V . Karpukhin, B. Oguz, S. Min, P. S. Lewis, L. Wu, S. Edunov,
D. Chen, and W.-t. Yih, “Dense passage retrieval for open-domain
question answering.” inEMNLP (1), 2020, pp. 6769–6781.
[28] J. Johnson, M. Douze, and H. Jégou, “Billion-scale similarity search
with gpus,”IEEE Transactions on Big Data, vol. 7, no. 3, pp. 535–547,
2019.
[29] O. Khattab and M. Zaharia, “Colbert: Efficient and effective passage
search via contextualized late interaction over bert,” inProceedings
of the 43rd International ACM SIGIR conference on research and
development in Information Retrieval, 2020, pp. 39–48.
[30] G. Izacard and E. Grave, “Leveraging passage retrieval with gener-
ative models for open domain question answering,”arXiv preprint
arXiv:2007.01282, 2020.
[31] S. Pan, L. Luo, Y . Wang, C. Chen, J. Wang, and X. Wu, “Unifying
large language models and knowledge graphs: A roadmap,”IEEE
Transactions on Knowledge and Data Engineering, vol. 36, no. 7, pp.
3580–3599, 2024.
[32] H. Soudani, E. Kanoulas, and F. Hasibi, “Fine tuning vs. retrieval
augmented generation for less popular knowledge,” inProceedings of
the 2024 Annual International ACM SIGIR Conference on Research and
Development in Information Retrieval in the Asia Pacific Region, 2024,
pp. 12–22.
[33] Z. Jiang, F. F. Xu, L. Gao, Z. Sun, Q. Liu, J. Dwivedi-Yu, Y . Yang,
J. Callan, and G. Neubig, “Active retrieval augmented generation,” in
Proceedings of the 2023 Conference on Empirical Methods in Natural
Language Processing, 2023, pp. 7969–7992.
[34] Y . Mao, P. He, X. Liu, Y . Shen, J. Gao, J. Han, and W. Chen, “Rider:
Reader-guided passage reranking for open-domain question answering,”
arXiv preprint arXiv:2101.00294, 2021.
[35] R. Zhang, Y . Xu, Y . Xiao, R. Zhu, X. Jiang, X. Chu, J. Zhao, and
Y . Wang, “Knowpo: Knowledge-aware preference optimization for con-
trollable knowledge selection in retrieval-augmented language models,”
inProceedings of the AAAI Conference on Artificial Intelligence, vol. 39,
no. 24, 2025, pp. 25 895–25 903.
[36] W. Shi, S. Min, M. Yasunaga, M. Seo, R. James, M. Lewis, L. Zettle-
moyer, and W.-t. Yih, “Replug: Retrieval-augmented black-box language
models,”arXiv preprint arXiv:2301.12652, 2023.
[37] R. S. Sutton, A. G. Bartoet al.,Reinforcement learning: An introduction.
MIT press Cambridge, 1998, vol. 1, no. 1.[38] G. Mialon, R. Dessì, M. Lomeli, C. Nalmpantis, R. Pasunuru,
R. Raileanu, B. Rozière, T. Schick, J. Dwivedi-Yu, A. Celikyil-
mazet al., “Augmented language models: a survey,”arXiv preprint
arXiv:2302.07842, 2023.
[39] Z. Chai, T. Zhang, L. Wu, K. Han, X. Hu, X. Huang, and Y . Yang,
“Graphllm: Boosting graph reasoning ability of large language model,”
IEEE Transactions on Big Data, 2025.
[40] P. Zheng, C. Zhang, M. Cui, G. Chen, Q. Sun, J. Huang, J. Zhang, T.-H.
Kim, C. Qin, Y . Renet al., “Towards visual chain-of-thought reasoning:
A comprehensive survey,” 2026.
[41] C. Yan, F. Zhao, X. Tao, and X. Zhu, “Multi-view few-shot reasoning
for emerging entities in knowledge graphs,”IEEE Transactions on Big
Data, 2024.
[42] M. Li, Z. Li, C. Huang, Y . Jiang, and X. Wu, “Edugraph: Learning path-
based hypergraph neural networks for mooc course recommendation,”
IEEE Transactions on Big Data, 2024.
[43] S. Yang, F. Cao, K. Yu, and J. Liang, “Learning causal chain graph
structure via alternate learning and double pruning,”IEEE Transactions
on Big Data, vol. 10, no. 4, pp. 442–456, 2023.
[44] X. L. Li and P. Liang, “Prefix-tuning: Optimizing continuous prompts
for generation,”arXiv preprint arXiv:2101.00190, 2021.
[45] X. Liu, Y . Zheng, Z. Du, M. Ding, Y . Qian, Z. Yang, and J. Tang, “Gpt
understands, too,”AI Open, vol. 5, pp. 208–215, 2024.
[46] L. Wang, S. Chen, L. Jiang, S. Pan, R. Cai, S. Yang, and F. Yang,
“Parameter-efficient fine-tuning in large language models: a survey of
methodologies,”Artificial Intelligence Review, vol. 58, no. 8, p. 227,
2025.
[47] X. Liu, K. Ji, Y . Fu, W. L. Tam, Z. Du, Z. Yang, and J. Tang, “P-tuning
v2: Prompt tuning can be comparable to fine-tuning universally across
scales and tasks,”arXiv preprint arXiv:2110.07602, 2021.
[48] Q. Sun, C. Zhang, J. Zhang, X. Wang, J. Xie, P. Zheng, H. Wang,
S. Lee, C.-l. A. Tai, Y . Yanget al., “Grasp: Guided region-aware
sparse prompting for adapting mllms to remote sensing,”arXiv preprint
arXiv:2601.17089, 2026.
[49] Y . Bai, X. Lv, J. Zhang, H. Lyu, J. Tang, Z. Huang, Z. Du, X. Liu,
A. Zeng, L. Houet al., “Longbench: A bilingual, multitask benchmark
for long context understanding,”arXiv preprint arXiv:2308.14508, 2023.
[50] Z. Yang, P. Qi, S. Zhang, Y . Bengio, W. W. Cohen, R. Salakhutdinov, and
C. D. Manning, “Hotpotqa: A dataset for diverse, explainable multi-hop
question answering,”arXiv preprint arXiv:1809.09600, 2018.
[51] X. Ho, A.-K. D. Nguyen, S. Sugawara, and A. Aizawa, “Constructing a
multi-hop qa dataset for comprehensive evaluation of reasoning steps,”
arXiv preprint arXiv:2011.01060, 2020.
[52] H. Trivedi, N. Balasubramanian, T. Khot, and A. Sabharwal, “Musique:
Multihop questions via single-hop question composition,”Transactions
of the Association for Computational Linguistics, vol. 10, pp. 539–554,
2022.
[53] T. Kwiatkowski, J. Palomaki, O. Redfield, M. Collins, A. Parikh,
C. Alberti, D. Epstein, I. Polosukhin, J. Devlin, K. Leeet al., “Natural
questions: a benchmark for question answering research,”Transactions
of the Association for Computational Linguistics, vol. 7, pp. 453–466,
2019.
[54] P. Rajpurkar, J. Zhang, K. Lopyrev, and P. Liang, “Squad: 100,000+
questions for machine comprehension of text,”arXiv preprint
arXiv:1606.05250, 2016.
[55] X. Zhu, Y . Xie, Y . Liu, Y . Li, and W. Hu, “Knowledge graph-guided
retrieval augmented generation,” inProceedings of the 2025 Conference
of the Nations of the Americas Chapter of the Association for Compu-
tational Linguistics: Human Language Technologies (Volume 1: Long
Papers), 2025, pp. 8912–8924.
[56] S. Yao, J. Zhao, D. Yu, N. Du, I. Shafran, K. Narasimhan, and Y . Cao,
“React: Synergizing reasoning and acting in language models,” in
International Conference on Learning Representations (ICLR), 2023.
[57] L. Gao, Z. Dai, T. Chen, Z. Fan, B. Van Durme, and J. Callan, “Com-
plement lexical retrieval model with semantic residual embeddings,” in
European Conference on Information Retrieval. Springer, 2021, pp.
146–160.
[58] Z. Guo, L. Xia, Y . Yu, T. Ao, and C. Huang, “Lightrag: Simple and
fast retrieval-augmented generation,”arXiv preprint arXiv:2410.05779,
2024.
[59] D. Edge, H. Trinh, N. Cheng, J. Bradley, A. Chao, A. Mody, S. Truitt,
D. Metropolitansky, R. O. Ness, and J. Larson, “From local to global:
A graph rag approach to query-focused summarization,”arXiv preprint
arXiv:2404.16130, 2024.
[60] E. J. Hu, Y . Shen, P. Wallis, Z. Allen-Zhu, Y . Li, S. Wang, L. Wang,
W. Chenet al., “Lora: Low-rank adaptation of large language models.”
ICLR, vol. 1, no. 2, p. 3, 2022.