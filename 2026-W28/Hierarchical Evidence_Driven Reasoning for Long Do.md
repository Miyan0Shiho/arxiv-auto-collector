# Hierarchical Evidence-Driven Reasoning for Long Document Understanding

**Authors**: Junyu Xiong, Yonghui Wang, Rongjian Gu, Chenyu Liu, Bing Yin, Wengang Zhou, Houqiang Li

**Published**: 2026-07-06 03:08:28

**PDF URL**: [https://arxiv.org/pdf/2607.04625v1](https://arxiv.org/pdf/2607.04625v1)

## Abstract
Retrieval-Augmented Generation (RAG) streamlines long-document understanding by leveraging retrieval mechanisms to restrict input images to a highly curated subset. However, existing multimodal RAG pipelines primarily face two critical challenges: first, standard semantic similarity retrievers frequently fetch topically overlapping yet answer-void distractor pages that mislead downstream generation; second, rigid single-pass pipelines heavily depend on initial retrieval success, where any omission of core evidence inevitably causes cascading errors. To address these challenges, we introduce HIEVI-RAG, a hierarchical, evidence-driven multimodal RAG framework for closed-domain document understanding. HIEVI-RAG systematically factorizes complex queries into a cooperative four-stage pipeline: (1) hierarchical question decomposition to break multi-hop root queries into atomic child questions; (2) coarse visual page retrieval leveraging a multimodal retriever to fetch candidate pages based on semantic similarity; (3) fine-grained page verification via EVIAGENT, a specialized multi-page verifier trained with GRPO to execute cross-page reasoning over multi-image blocks; and (4) memory-guided iterative generation that leverages accumulated sub-question context to execute multi-round, dynamic reasoning over the prioritized sequence. Extensive evaluations across four benchmarks demonstrate the robust efficacy and synergy of our framework, which significantly outperforms existing open-source baselines and exceeds the strongest reported baseline by an average of 8.05% in accuracy.

## Full Text


<!-- PDF content starts -->

HIEVI-RAG: Hierarchical Evidence-Driven Reasoning for Long Document
Understanding
Junyu Xiong1Yonghui Wang1Rongjian Gu1Chenyu Liu2
Bing Yin2Wengang Zhou1*Houqiang Li1
1University of Science and Technology of China
2iFLYTEK Research
xiongjyu@mail.ustc.edu.cn, zhwg@ustc.edu.cn
Abstract
Retrieval-Augmented Generation (RAG)
streamlines long-document understanding by
leveraging retrieval mechanisms to restrict in-
put images to a highly curated subset. However,
existing multimodal RAG pipelines primarily
face two critical challenges: first, standard
semantic similarity retrievers frequently
fetch topically overlapping yet answer-void
distractor pages that mislead downstream
generation; second, rigid single-pass pipelines
heavily depend on initial retrieval success,
where any omission of core evidence inevitably
causes cascading errors. To address these
challenges, we introduceHIEVI-RAG, a
hierarchical, evidence-driven multimodal
RAG framework for closed-domain document
understanding. HIEVI-RAG systematically
factorizes complex queries into a cooperative
four-stage pipeline: (1) hierarchical question
decomposition to break multi-hop root queries
into atomic child questions; (2) coarse visual
page retrieval leveraging a multimodal retriever
to fetch candidate pages based on semantic
similarity; (3) fine-grained page verification via
EVIAGENT, a specialized multi-page verifier
trained with GRPO to execute cross-page
reasoning over multi-image blocks; and (4)
memory-guided iterative generation that
leverages accumulated sub-question context to
execute multi-round, dynamic reasoning over
the prioritized sequence. Extensive evaluations
across four benchmarks demonstrate the robust
efficacy and synergy of our framework, which
significantly outperforms existing open-source
baselines and exceeds the strongest reported
baseline by an average of 8.05% in accuracy.
*Corresponding author.1 Introduction
Closed-domain document understanding has be-
come an indispensable core capability for practical
AI systems deployed across various sectors, includ-
ing science, finance, enterprise knowledge manage-
ment, and technical support (Hui et al., 2024; Suri
et al., 2025; Deng et al., 2025; Dong et al., 2024;
Xia et al., 2024). In these scenarios, users typically
pose queries over a proprietary document collec-
tion, where the answers are sparsely distributed
across hundreds of pages containing diverse and
rich information such as plain text, tables, charts,
and illustrations (Wang et al., 2025a; Chia et al.,
2025). However, due to the prohibitive number of
images in these closed-domain documents, feeding
the entire collection into a model in an end-to-end
manner easily exceeds the context window limits
of existing LLMs.
To address this bottleneck, a growing body of
work on Retrieval-Augmented Generation (RAG)
has emerged, which leverages retrieval mechanisms
to restrict the number of input images to a highly
curated subset prior to answer generation (Lewis
et al., 2020; Cho et al., 2024; Suri et al., 2025;
Xiong et al., 2026). Initially, to bypass tradi-
tional OCR limitations and capture multimodal in-
formation, ColPali (Faysse et al., 2025) extends
late interaction to rendered page images, while
M3DocRAG (Cho et al., 2024) establishes this
method as the core interface for multi-page QA.
Subsequently, research shifted toward filtering dis-
tractors and optimizing ranking precision: MDocA-
gent (Han et al., 2025) and SimpleDoc (Jain et al.,
2025) refine results via module collaboration or
iterative cascading; RagVL (Chen et al., 2025)
and MM-R5 (Xu et al., 2025) employ instruction-
tuning or reinforcement learning to train strongarXiv:2607.04625v1  [cs.CV]  6 Jul 2026

VLM rerankers; and MoLoRAG (Wu et al., 2025)
leverages page topology graphs for logic-aware
traversal. Concurrently, DocAgent (Sun et al.,
2025) and ALDEN (Yang et al., 2026) utilize mem-
ory feedback or active exploration protocols to
gather evidence dynamically, while DMAP (Fu
et al., 2026) constructs human-aligned structural
maps for global document comprehension. De-
spite these advancements, two challenges remain:
first, relying on semantic similarity misleads mod-
els with topically related yet answer-void pages;
second, single-pass pipelines rigidly depend on ini-
tial retrieval, where any omission of core evidence
makes correct generation impossible.
To address these challenges, we proposeHIEVI-
RAG, a four-step framework for long-document
understanding. First,hierarchical question de-
compositionbreaks down the complex, multi-hop
root question into multiple atomic sub-questions,
reducing retrieval difficulty while assisting the root
question’s resolution from different logical levels.
Second,coarse-grained page retrievalutilizes
a multimodal retriever (e.g., ops-ColQwen3) to
fetch the top- kpages based on semantic similar-
ity for each question, including both root and sub-
questions. Third, forfine-grained evidence veri-
fication, inspired by DocR1 (Xiong et al., 2025),
we employ GRPO (Shao et al., 2024) to trainEVI-
AGENT, an evidence-aware multi-page reasoning
model capable of cross-page reasoning and ground-
ing over multi-image inputs to pinpoint question-
relevant evidence. In this stage, for sub-questions,
we strictly retain only the evidence pages identified
by EVIAGENT; for the root question, EVIAGENT’s
verified evidence pages are prioritized at the front,
while non-evidence pages follow in order of their
initial semantic similarity, thereby achieving pre-
cise reranking. Fourth, duringmemory-guided
iterative generation, an empty memory is initial-
ized. The sub-questions and their corresponding
evidence pages are fed into the model, with the
sub-questions, generated reasoning traces, and in-
termediate answers subsequently accumulated in
the memory. For the root question, the model takes
the root query, the accumulated memory, and a
batch of images as input. If the current memory
and image batch are insufficient to resolve the ques-
tion, the system appends the latest reasoning trace
to the memory and fetches the next batch of images
for iterative reasoning.
To verify the effectiveness of our approach, we
conduct extensive experiments and analyses acrossfour distinct datasets. The results demonstrate that
our method significantly outperforms existing open-
source baselines, exceeding the previous state-of-
the-art by an average of 8.05%. Furthermore, we
perform comprehensive ablation studies to validate
the individual efficacy of each proposed module.
Our contributions can be summarized as follows:
•We propose HIEVI-RAG, a four-step multi-
modal RAG framework for long-document
QA that unifies hierarchical question decom-
position, coarse-to-fine retrieval, and memory-
guided iterative generation.
•We introduce a fine-grained verification mech-
anism via EVIAGENT, which performs cross-
page reasoning over multi-image inputs to pre-
cisely ground answer-supporting evidence.
•We formulate a memory-guided iterative gen-
eration process that utilizes sub-question rea-
soning traces to assist the root query while dy-
namically updating an explicit memory across
multiple rounds to eliminate cascading re-
trieval errors.
2 Related Work
2.1 Document Visual Question Answering
Advancements in DocVQA have bifurcated into
OCR-based and OCR-free paradigms. OCR-based
methods convert pages into structured textual and
spatial tokens to feed downstream reasoning frame-
works (Tang et al., 2023; Wang et al., 2024a; Zhu
et al., 2025), yet their performance remains inher-
ently bounded by compounding parsing errors and
the serialization loss of visual context. Conversely,
OCR-free approaches directly train VLMs on raw
document images, evolving from architectural scal-
ing for resolution management (Ye et al., 2023a;
Feng et al., 2023; Hu et al., 2024b) to evidence-
aware reasoning via reinforcement learning (Xiong
et al., 2025; Zheng et al., 2026). While these meth-
ods substantially enhance single- or multi-page
reader execution, effectively navigating and dis-
tilling massive long-document repositories remains
a largely orthogonal challenge.
2.2 Retrieval-Augmented Generation
Multimodal RAG methodologies have transitioned
from coarse-grained text-retrieval extensions to
visual-centric, layout-aware document discovery.

Early pipelines leverage visual retrievers and VLM-
based rerankers to screen candidate pages inde-
pendently based on superficial similarity scor-
ing (Faysse et al., 2025; Cho et al., 2024; Chen
et al., 2025; Xu et al., 2025). More recent efforts
incorporate iterative multi-agent navigation, mem-
ory feedback, or topological page graphs to cap-
ture structural inter-page logic (Wu et al., 2025;
Sun et al., 2025; Yang et al., 2026; Fu et al.,
2026). Despite these scaling improvements, ex-
isting pipelines still evaluate retrieved pages as iso-
lated query-image pairs, leaving them highly sus-
ceptible to topically overlapping but answer-void
distractors. To bridge these gaps, HIEVI-RAG
casts validation as a joint, cross-page verification
problem and bypasses cascading failure modes via
memory-guided iterative reasoning. Detailed re-
lated works are deferred to Appendix A.
3 Method
3.1 Overview and Task Formulation
Let a document be represented as an ordered se-
quence of page images D={p 1, p2, . . . , p N}and
qdenote a user question in a closed-domain set-
ting. The final answer is assumed to be grounded
inD, although the supporting evidence may be
sparsely distributed across multiple non-contiguous
pages. When the document length Nscales to hun-
dreds of pages, direct end-to-end VLM ingestion
becomes computationally prohibitive due to con-
text window constraints. To bypass this bottleneck,
HIEVI-RAG decomposes long-document QA into
four cooperative stages: (1) hierarchical question
decomposition, (2) coarse visual page retrieval, (3)
evidence-aware page verification, and (4) memory-
guided iterative generation. A core design rationale
of our framework is the multi-granularity opera-
tion across stages: coarse retrieval is strictly recall-
oriented, EVIAGENTconducts fine-grained verifi-
cation over grouped page candidates via cross-page
reasoning, and the final generator iteratively con-
sumes the verified evidence sequence backed by an
explicit memory. The overall pipeline is illustrated
in Figure 1.
3.2 Hierarchical Question Decomposition
Given a complex root question q, this stage con-
structs a shallow reasoning hierarchy, as illustrated
in Figure 1(1). Specifically, a lightweight LLM ex-
tracts key entities, structural attributes, constraints,
and logical relations from qto instantiate a set ofatomic child questions:
Q(q) ={q 1, q2, . . . , q m}, m≤5.(1)
Each child question qiis formulated as a distinct
natural query covering a non-redundant subset of
the global information need, while strictly avoiding
near-paraphrases of the full prompt.
Crucially, partitioning the global query into lo-
calized targets reduces the inherent difficulty of
multi-hop reasoning. This upfront decomposition
prevents retrieval omissions typical of standalone
root queries, ensuring core evidence is captured
before downstream generation. The correspond-
ing prompt template and example are detailed in
Appendix B.1 and E.
3.3 Coarse Visual Page Retrieval
Following decomposition, we employ
Ops-ColQwen3-4B (OpenSearch-AI, 2026),
a ColPali-style late-interaction multimodal re-
triever (Faysse et al., 2025; Khattab and Zaharia,
2020), to construct a candidate page pool from
rendered document images. As illustrated in
Figure 1(2), this retrieval stage is split into offline
index construction and online query matching.
Offline Index Construction.Each document
page image pi∈D is encoded offline into a multi-
vector representation to preserve fine-grained vi-
sual features:
vpi=E page(pi), i= 1,2, . . . , N.(2)
The derived page embeddings are subsequently
stored to build the global document index.
Online Query Matching.Upon receiving a root
question q, the stage online encodes both qand its
derived atomic child questions qi∈ Q(q) to obtain
their respective text embeddings:
vq=E query(q),v qi=E query(qi).(3)
These query embeddings are then matched against
the offline page index using the late-interaction
operator to compute similarity scores across all
document pages. Based on these scores, we rank
and retain the Top- K(K= 20 ) candidate pages for
the root question and each child question, denoted
asTqandT qirespectively.

1. Hierarchical Ques-tionDecomposition
LLM2. Coarse Visual RetrievalEncoderVector DB
…$!$"$#
$&
offline: index construction
Encoderonline: query matching!!"!!!#…
match
…Top K (K=20)3. Evidence-Aware Verification 
1!1"1&1'…think
evidence page
!/
……
KeepDiscard
…
EviAgent
1!1"1&1'…think
evidence page
Evidence First
…EviAgent!Child question
Root question……
Others Later4.Memory-GuidedIterative Generation 
Final Answer
!0
VLM
answer
Memoryinitialize!
Answerableyesfetch the next block.noupdate memory
Question: 2
!!!!"!#…"≤5Initializing Memory
Iterative Reasoning
!!/Child questionRoot questionFigure 1: Overview of HIEVI-RAG. The system first builds a shallow hierarchy from the original question to
atomic child questions, retrieves Top- Kcandidate pages for both root and child questions with a visual retriever,
verifies and reranks grouped- kcandidates with EVIAGENT, and performs memory-guided iterative reasoning over
the verified evidence order.
3.4 Evidence-Aware Page Verification
To filter out topically related but answer-void dis-
tractors from the coarse retrieval pool, we introduce
EVIAGENT, a specialized model trained to perform
binary evidence page verification via cross-page
reasoning. As illustrated in Figure 1(3), this stage
takes the coarse candidate lists TqandTqias input,
partitions them into grouped- ksliding windows,
and leverages the trained agent to output structured
decisions.
Training EVIAGENTwith Evidence-Aware
GRPO.Inspired by DocR1 (Xiong et al., 2025),
we train EVIAGENTusing an Evidence-Aware
Group Relative Policy Optimization (EviGRPO)
pipeline to specialize a VLM for multi-page docu-
ment verification. The policy is optimized across
six open-source multi-page datasets (Van Lan-
deghem et al., 2023; Tito et al., 2022; Zhao et al.,
2022; Schimanski et al., 2026; Yu et al., 2025a;
Tanaka et al., 2023) with statistics detailed in Ap-
pendix D.
During the training phase, the VLM is con-
strained to generate a structured content contain-ing three mandatory fields: o= (τ,y, a) . The
corresponding prompt template is detailed in Ap-
pendix B.2.
•Reasoning Trace( τ): Contained within the
<think> tag,τencapsulates intra-page visual
inspection and cross-page reasoning.
•Evidence Decisions( y∈ {T, F}k): En-
closed within the <evidence_page> tag,y
enforces a strict one-to-one binary sequence
matching the exact order of input images to
prevent evaluation omissions.
•Final Answer( a): Wrapped within the
<answer> tag,adelivers the terminal text re-
sponse derived from the verified evidence.
The policy is optimized via a weighted joint re-
ward function:
R=λ fmtrfmt+λ evirevi+λ accracc,(4)
where rfmtenforces schema adherence, and racc
evaluates final answer correctness. The evidence
reward revicomputes the F1score against ground-
truth page labels to accurately measure verification
performance and prevent reward hacking.

Fine-Grained Page Validation.During on-
line inference, each retrieved candidate list is
split into L=⌈K/k⌉ blocks of size k:
bk
t={p (t−1)k+1 , . . . , p min(tk,K) }, where t∈
{1, . . . , L} is the block index. Each block is veri-
fied jointly by the trained EVIAGENT. To prevent
information fragmentation across boundaries, all
pages verified as true evidence are aggregated into
a final consolidated block for a global cross-block
synthesis. The refined signals are then routed to
two distinct downstream policies:
•Discarding for Child Questions: For each
child question qi, we extract only the verified
evidence pages:
E(qi) ={p∈T qi:yp=T}.(5)
Non-evidence pages are discarded to restrict
downstream generation strictly to localized,
verified facts.
•Reranking for the Root Question: For the
root question q, letE(q) denote the verified
evidence page set. We construct a prioritized
sequence Π(q) while preserving the original
retriever ranking order within each partition:
Π(q) =E(q)⊕(T q\E(q)),(6)
where ⊕denotes sequence concatenation.
BothE(q) and the remaining subset Tq\E(q)
strictly inherit the original ranking order de-
termined by the coarse visual retriever. Non-
evidence pages are retained as a backup suffix
to secure recall for downstream iterative rea-
soning.
3.5 Memory-Guided Iterative Generation
This final stage leverages the resolved child ques-
tions and the prioritized sequence Π(q) to gener-
ate the terminal answer via a two-step, memory-
enhanced pipeline. As illustrated in Figure 1(4),
the process consists of initializing an explicit mem-
ory layer with sub-question contexts and iteratively
reasoning over the root question.
Initializing Memory with Sub-Questions.Be-
fore addressing the root query, the framework se-
quentially resolves each child question qi∈ Q(q)
over its verified evidence block E(qi)using EVIA-
GENT:
(τi,yi, ai) =A 
qi, E(q i)
,(7)where Adenotes the generative policy of EVIA-
GENT. Initializing the memory as M0=∅, the
state updates cumulatively by appending the com-
plete triplets:
Mi=M i−1∪ {(q i, τi, ai)}.(8)
This mechanism distills sub-question verification
into informative textual facts, seeding the working
memory with both mid-level execution traces and
terminal local answers.
Iterative Reasoning for the Root Question.
Upon consolidating the sub-question memory Mm
(where m=|Q(q)| ), HIEVI-RAG evaluates the
root question qby scanning the prioritized se-
quence Π(q) via a sliding window of size k. At ex-
ecution round t, EVIAGENTingests the root query,
the persistent memory state, and the t-th evidence
blockbk
t⊂Π(q):
(τt,yt, at) =A 
q, bk
t, Mm+t−1
.(9)
The execution terminates immediately, returning
atas the global response if at̸=α ∅, where
α∅denotes the designated abstention token (i.e.,
NOT_ANSWERABLE ). Otherwise, the current reason-
ing trace τtis appended to the memory to guide the
next window:
Mm+t=M m+t−1 ∪ {(q, τ t)}.(10)
This iterative accumulation continues until a valid
answer is produced or Π(q) is exhausted. By main-
taining text-based memory across rounds, this re-
current design forces the model to synthesize his-
torical context globally, preventing critical informa-
tion omission inherent in local window constraints.
4 Experiments
Benchmark # QA Samples Min. Pages Max. Pages Avg. Pages
PaperTab 393 2 152 10.7
FetaTab 1016 2 216 16.3
MMLongBench 1,082 9 468 47.8
LongDocURL 2,325 51 149 89.0
Table 1: Benchmark statistics for the four-dataset evalu-
ation protocol. Average page counts are rounded to one
decimal place.
4.1 Experimental Setup
Benchmarks.We evaluate HIEVI-RAG on four
benchmarks: PaperTab and FetaTab (Hui et al.,

Method Param. PaperTab FetaTab MMLongBench LongDocURL
(Acc.) (Acc.) (Acc.) (Acc.)
M3DocRAG (Cho et al., 2024) 7B 28.5 63.8 36.2 49.0
MDocAgent (Han et al., 2025) 7B 30.0 66.3 38.5 46.9
MoLoRAG+ (Wu et al., 2025) 7B 31.0 69.2 41.0 51.9
ALDEN (Yang et al., 2026) 7B 24.5 62.3 39.2 55.1
URaG (Shi et al., 2026) 7B – – 33.8 52.2
Doc-V⋆(Zheng et al., 2026) 7B – – 42.1 56.3
HIEVI-RAG 8B44.0 72.9 48.2 65.7
Table 2: Main experimental results across the four benchmarks. The best results are highlighted inbold, and the
second-best results are underlined. The symbol “–” indicates that the baseline result is not reported. ALDEN uses
the ColQwen+ColBERT pipeline, and Doc-V⋆refers to the GRPO-optimized model.
Variant PaperTab FetaTab MMLongBench LongDocURL Avg.
HIEVI-RAG44.0 72.9 48.2 65.7 57.7
w/o Evidence-Aware Verification 39.7 (-4.3) 69.6 (-3.3) 44.5 (-3.7) 59.1 (-6.6) 53.2 (-4.5)
w/o Hierarchical Decomposition 41.3 (-2.7) 70.5 (-2.4) 45.8 (-2.4) 62.4 (-3.3) 55.0 (-2.7)
w/o Iterative Reasoning 41.9 (-2.1) 72.0 (-0.9) 46.1 (-2.1) 63.2 (-2.5) 55.8 (-1.9)
Table 3: Ablation analysis of individual architectural modules. Forw/o Evidence-Aware Verification, HIEVI-RAG’s
Step 3 is bypassed; sub-questions in Step 4 initialize memory directly using the unverified Top- Kcoarse retrieval,
and the root question iteratively reasons over the un-reranked sequence. Forw/o Hierarchical Decomposition,
HIEVI-RAG’s Step 1 and the memory initialization phase in Step 4 are omitted, while the iterative reasoning
loop remains active. Forw/o Iterative Reasoning, the sliding window in HIEVI-RAG’s Step 4 is deactivated,
force-feeding the Top-5 candidate pages into the generator in a single-pass manner.
2024), which target question answering over
scientific and Wikipedia-style tables; and MM-
LongBench (Wang et al., 2025a) and Long-
DocURL (Deng et al., 2025), which feature ex-
tended multimodal documents stressing cross-page
retrieval, visual grounding, and multi-hop reason-
ing. Structural statistics are provided in Table 1.
Implementation Details.For hierarchical ques-
tion decomposition, a lightweight Qwen3.5-4B
model is deployed by default to generate at most
five child questions. The baseline coarse retriever
is configured as ops-ColQwen3-4B (OpenSearch-
AI, 2026). Both root and child queries retrieve
the Top- K(K= 20 ) candidate pages, which
are subsequently verified using a grouped window
size of k= 7 . We instantiate EVIAGENTusing
Qwen3-VL-8B-Instruct (Qwen Team, 2025) as
the foundational backbone. For the joint reward
function, the balancing scaling weights are explic-
itly set as (λfmt, λevi, λacc) = (0.1,0.5,0.4) , priori-
tizing the evidence verification performance. Train-
ing is executed on an 8×NVIDIA A100 GPU clus-
ter.Metrics.For MMLongBench and LongDocURL,
we follow their original evaluation protocols, em-
ploying standard accuracy and rule-based string
matching tailored to various answer types. For Pa-
perTab and FetaTab, we adopt an LLM-as-a-judge
paradigm leveraging Qwen3.5-27B as the evalua-
tor to compute binary accuracy (0 or 1) by assess-
ing whether the generated response semantically
matches the ground truth. Additionally, retrieval-
related performance is quantified via Recall@K,
NDCG@K, and MRR@K, with detailed mathe-
matical formulations deferred to Appendix C.
4.2 Main Results
Table 2 presents a performance comparison across
the four benchmarks. HIEVI-RAG consistently
outperforms all competitive baselines, validating
the synergistic effects of its core architectural com-
ponents.
On PaperTab and FetaTab, HIEVI-RAG
achieves44.0%and72.9%accuracy, yielding ab-
solute improvements of +13.0% and +3.7% over
the strongest baseline (MoLoRAG+), respectively.
Tabular documents typically require exact cell
alignment; standard retrievers frequently introduce

Method Generator Retriever PaperTab FetaTab MMLongBench LongDocURL
(Acc.) (Acc.) (Acc.) (Acc.)
MoLoRAG+ (Wu et al., 2025)Qwen2.5-VL-7B-Instruct ColQwen2.531.0 69.2 41.0 51.9
HIEVI-RAGQwen2.5-VL-7B-Instruct ColQwen2.534.2 69.1 42.6 55.4
Qwen2.5-VL-7B-Instruct Ops-ColQwen3-4B38.3 71.4 45.0 60.2
Qwen3-VL-8B-Instruct ColQwen2.536.0 70.8 43.8 57.1
Qwen3-VL-8B-Instruct Ops-ColQwen3-4B44.0 72.9 48.2 65.7
Table 4: Sensitivity analysis over generation backbones and visual retrievers. MoLoRAG+ is included as the
external baseline. The last row corresponds to the main HIEVI-RAG configuration in Table 2. The remaining
HIEVI-RAG rows are controlled variants that change only the listed generator or retriever while keeping the same
evidence-verification and iterative-generation pipeline.
false positives due to repetitive schema structures.
HIEVI-RAG mitigates this bottleneck by execut-
ing fine-grained verification via EVIAGENT, effec-
tively isolating answer-void distractors.
Similarly, on MMLongBench and Long-
DocURL, HIEVI-RAG secures48.2%and65.7%
accuracy, outperforming Doc- V⋆by +6.1% and
+9.4%, respectively. These tasks necessitate cross-
page retrieval and multi-hop reasoning, where
single-pass pipelines easily omit critical informa-
tion. The consistent performance gains demon-
strate the efficacy of our memory-guided iterative
generation.
4.3 Ablation Studies
Contribution of Individual Modules.To isolate
the empirical impact of each architectural phase in
HIEVI-RAG, we evaluate three ablation variants
across all benchmarks (Table 3). Disabling any
single stage yields a consistent performance degra-
dation, validating the synergy of our core compo-
nents.
Specifically,w/o Evidence-Aware Verification
drops the average score by -4.5% absolutely. With-
out Step 3 verification, answer-void distractors oc-
cupy the front of the sequence, misleading the gen-
erator into premature early stopping and missing
genuine evidence. Furthermore, initializing mem-
ory over unverified Top- Kinputs introduces catas-
trophic cascading noise. Meanwhile,w/o Hierar-
chical Decompositioncompromises the average
score by -2.7%. This demonstrates that execut-
ing complex queries as standalone prompts without
HIEVI-RAG’s Step 1 induces severe retrieval omis-
sions, whereas atomic decomposition guarantees
the upfront recall of dispersed evidence. Lastly,
w/o Iterative Reasoningconsistently degrades re-
sults by -1.9% on average. This variant deactivates
the sliding window in HIEVI-RAG’s Step 4 and
force-feeds the Top-5 pages in a single pass. Theresult proves that while verification serves as an
effective filter, iterative reasoning acts as an indis-
pensable safety net against information omission
under local window constraints.
Backbone and Retriever Sensitivity.To evalu-
ate the robustness of HIEVI-RAG under varying
underlying architectures, we perform a grid sensi-
tivity analysis by cross-combining different gener-
ative backbones and visual retrievers (Table 4).
Specifically, when deploying the identical back-
bone and retriever configuration as MoLoRAG+
(Qwen2.5-VL-7B-Instruct + ColQwen2.5 ),
HIEVI-RAG outperforms this baseline in almost
all benchmarks. It delivers clear absolute im-
provements of +3.2% on PaperTab and +3.5% on
LongDocURL, demonstrating that our primary
advantages stem strictly from architectural
innovations rather than the scaling of foundational
models. Furthermore, scaling up either the
retriever to Ops-ColQwen3-4B or the generator to
Qwen3-VL-8B-Instruct provides independent
performance increments across benchmarks.
The combination of both upgraded components
yields the best performance, proving that our
framework effectively leverages stronger retrieval
and generation capabilities to maximize final
accuracy.
Effectiveness of Fine-Grained Page Validation.
To evaluate the impact of the verification stage on
retrieval quality, we compare the retrieval metrics
of the baseline against the reranked outputs gener-
ated by our verifier both before and after training
under various Top- Ksettings on MMLongBench
and LongDocURL (Table 5).
Specifically, deploying an untunedQwen3-VL-
8B-Instructyields marginal or even detrimental
performance shifts. While it provides negligi-
ble fluctuations on MMLongBench (e.g., a mod-
est+0.2 point increase in NDCG@1 at Top-1), it

Top-KMethodMMLongBench LongDocURL
Recall NDCG MRR Recall NDCG MRR
1Ops-ColQwen3-4B (baseline) 54.2 70.4 70.4 53.5 73.4 73.4
+ Qwen3-VL-8B-Instruct 54.2(+0.0)70.6(+0.2)70.6(+0.2)52.9(−0.6)72.1(−1.3)72.1(−1.3)
+ EVIAGENT59.4(+5.2)78.3(+7.9)78.3(+7.9)57.3(+3.8)78.8(+5.4)78.8(+5.4)
3Ops-ColQwen3-4B (baseline) 75.2 73.8 77.2 74.8 73.7 81.1
+ Qwen3-VL-8B-Instruct 75.6(+0.4)74.4(+0.6)77.6(+0.4)73.7(−1.1)72.5(−1.2)79.8(−1.3)
+ EVIAGENT80.1(+4.9)80.6(+6.8)83.7(+6.5)77.5(+2.7)77.3(+3.6)84.9(+3.8)
5Ops-ColQwen3-4B (baseline) 81.4 76.2 78.2 81.0 76.4 82.0
+ Qwen3-VL-8B-Instruct 81.8(+0.4)76.6(+0.4)78.5(+0.3)80.2(−0.8)75.4(−1.0)80.7(−1.3)
+ EVIAGENT85.2(+3.8)82.2(+6.0)84.4(+6.2)83.2(+2.2)79.7(+3.3)85.6(+3.6)
Table 5: Retrieval performance on MMLongBench and LongDocURL under different Top- Ksettings. Values in
parentheses denote the absolute difference compared with the Ops-ColQwen3-4B baseline under the same Top- K
setting.
Figure 2: Effect of block size kon MMLongBench retrieval quality. A moderate block size provides the best
balance between cross-page comparison and visual-context overload.
consistently degrades retrieval precision on Long-
DocURL, with NDCG slipping by −1.3 points at
Top-1 and −1.0 points at Top-5. This demonstrates
that off-the-shelf VLMs lack the necessary align-
ment to distinguish authentic evidence from com-
plex multimodal distractors. In sharp contrast, our
trained EVIAGENTconsistently and substantially
outperforms thebaseline. It elevates the NDCG@1
from 70.4% to 78.3% on MMLongBench, and from
73.4% to 78.8% on LongDocURL. Furthermore, at
Top-5, EVIAGENTpushes the final Recall to 85.2%
and 83.2% on the respective benchmarks. These
comprehensive gains across Recall, NDCG, and
MRR solidly confirm EVIAGENT’s superior capac-
ity in identifying authentic evidence and filtering
out answer-void noise.
Sensitivity to Block Size.To evaluate the impact
of the block size kdefined in HIEVI-RAG’s Step
3, we analyze the retrieval quality under varying
scales across MMLongBench (Figure 2). The em-
pirical results indicate that both excessively large
and excessively small values of kdegrade retrieval
quality.
Specifically, a moderate block size (e.g., k= 7 )
achieves the optimal balance and yields the high-
est retrieval metrics. Setting ktoo small isolatescandidate pages. This constrains the model’s ca-
pacity for cross-page reasoning and comparison.
Conversely, expanding kexcessively degrades per-
formance. This drop is driven by visual context
overload, where the excessive input scale dilutes
model attention. Therefore, maintaining a tightly
bounded block size kis essential to maximize page
validation accuracy.
5 Conclusion
In this work, we presented HIEVI-RAG, a hierar-
chical, evidence-driven multimodal RAG frame-
work for closed-domain long-document under-
standing, whose core innovation lies in its cooper-
ative four-stage pipeline. Specifically, the frame-
work first employs hierarchical question decom-
position to partition complex multi-hop queries
into atomic child questions, drastically reducing
initial retrieval difficulty. Second, coarse visual
retrieval leverages rendered page indexing to guar-
antee upfront evidence recall. Third, fine-grained
page verification utilizes the EVIAGENTto perform
cross-page reasoning over multi-image blocks, pre-
cisely suppressing topically related yet answer-
void distractors. Fourth, memory-guided iterative
generation accumulates sub-question traces into

an explicit text-based memory, dynamically exe-
cuting sliding-window reasoning to eliminate cas-
cading information omissions. Extensive experi-
ments across four benchmarks demonstrate that
HIEVI-RAG significantly outperforms existing
open-source baselines. Furthermore, comprehen-
sive ablation studies analyses solidly validate the
individual efficacy and systemic synergy of each
cooperative phase within our pipeline.
Limitations
Despite the state-of-the-art performance achieved
by HIEVI-RAG, several inherent limitations war-
rant further investigation.
First, our framework introduces non-trivial com-
putational overhead during the inference stage.
Specifically, the hierarchical question decompo-
sition, the multi-block page verification via the
EVIAGENT, and the subsequent memory-guided
generation inherently mandate multiple sequential
LLM/VLM invocations. This compound multi-
stage architecture inevitably increases cumulative
response latency and inference costs compared to
standard single-pass pipelines.
Second, the memory-guided generation mecha-
nism relies heavily on the quality of textual interme-
diate reasoning traces; if the initial sub-questions
yield severely hallucinated or biased summaries,
such errors may cascade into the persistent mem-
ory layer and occasionally misguide the terminal
root-question synthesis.
Third, our framework was optimized and evalu-
ated predominantly on born-digital electronic doc-
uments with pristine rendering. Its zero-shot gener-
alization capabilities across real-world handwritten
inputs, heavily degraded physical scans with text
artifacts, or scene-text images captured in natural
environments require broader empirical validation
in future work.
Ethical Considerations
HIEVI-RAG is designed for closed-domain long-
document understanding and may be applied to
documents containing sensitive or proprietary in-
formation. All experiments in this paper are con-
ducted on publicly available benchmarks and open-
source datasets. We do not collect new user data
or perform human-subject annotation. Neverthe-
less, deployment in real-world enterprise or finan-
cial scenarios should ensure proper access control,
data privacy protection, and auditing of generatedanswers. Because the system may still produce
incorrect or unsupported answers when retrieval or
evidence verification fails, we recommend using
the framework as an assistive tool rather than an au-
tonomous decision-making system in high-stakes
domains.
References
Srikar Appalaraju, Bhavan Jasani, Bhargava Urala
Kota, Yusheng Xie, and R. Manmatha. 2024. Doc-
Formerv2: Local features for document understand-
ing. InProceedings of the AAAI Conference on Arti-
ficial Intelligence.
Zhanpeng Chen, Chengjin Xu, Yiyan Qi, Xuhui Jiang,
and Jian Guo. 2025. VLM is a strong reranker:
Advancing multimodal retrieval-augmented genera-
tion via knowledge-enhanced reranking and noise-
injected training. InFindings of the Association
for Computational Linguistics: EMNLP 2025, pages
8140–8158, Suzhou, China. Association for Compu-
tational Linguistics.
Yew Ken Chia, Liying Cheng, Hou Pong Chan, Mao-
jia Song, Chaoqun Liu, Mahani Aljunied, Soujanya
Poria, and Lidong Bing. 2025. M-LongDoc: A
benchmark for multimodal super-long document un-
derstanding and a retrieval-aware tuning framework.
InProceedings of the 2025 Conference on Empiri-
cal Methods in Natural Language Processing, pages
9233–9250, Suzhou, China. Association for Compu-
tational Linguistics.
Jaemin Cho, Debanjan Mahata, Ozan Irsoy, Yujie
He, and Mohit Bansal. 2024. M3DocRAG: Multi-
modal retrieval is what you need for multi-page
multi-document understanding.arXiv preprint
arXiv:2411.04952.
Chao Deng, Jiale Yuan, Pi Bu, Peijie Wang, Zhong-
Zhi Li, Jian Xu, Xiao-Hui Li, Yuan Gao, Jun Song,
Bo Zheng, and Cheng-Lin Liu. 2025. LongDocURL:
a comprehensive multimodal long document bench-
mark integrating understanding, reasoning, and locat-
ing. InProceedings of the 63rd Annual Meeting of
the Association for Computational Linguistics (Vol-
ume 1: Long Papers), pages 1135–1159, Vienna,
Austria. Association for Computational Linguistics.
Haoyu Dong, Jianbo Zhao, Yuzhang Tian, Junyu Xiong,
Mengyu Zhou, Yun Lin, José Cambronero, Yeye
He, Shi Han, and Dongmei Zhang. 2024. Encoding
spreadsheets for large language models. InProceed-
ings of the 2024 Conference on Empirical Methods in
Natural Language Processing, pages 20728–20748.
Manuel Faysse, Hugues Sibille, Tony Wu, Bilel Omrani,
Gautier Viaud, Céline Hudelot, and Pierre Colombo.
2025. ColPali: Efficient document retrieval with
vision language models. InThe Thirteenth Interna-
tional Conference on Learning Representations.

Hao Feng, Qi Liu, Hao Liu, Jingqun Tang, Wengang
Zhou, Houqiang Li, and Can Huang. 2023. DocPe-
dia: Unleashing the power of large multimodal model
in the frequency domain for versatile document un-
derstanding.arXiv preprint arXiv:2311.11810.
ShunLiang Fu, Yanxin Zhang, Yixin Xiang, Xiaoyu
Du, and Jinhui Tang. 2026. DMAP: Human-aligned
structural document map for multimodal document
understanding. InProceedings of the ACM Web Con-
ference 2026.
Siwei Han, Peng Xia, Ruiyi Zhang, Tong Sun, Yun Li,
Hongtu Zhu, and Huaxiu Yao. 2025. MDocAgent:
A multi-modal multi-agent framework for document
understanding.arXiv preprint arXiv:2503.13964.
Anwen Hu, Haiyang Xu, Jiabo Ye, Ming Yan, Liang
Zhang, Bo Zhang, Chen Li, Ji Zhang, Qin Jin, and Fei
Huang. 2024a. mPLUG-DocOwl 1.5: Unified struc-
ture learning for OCR-free document understanding.
arXiv preprint arXiv:2403.12895.
Anwen Hu, Haiyang Xu, Jiabo Ye, Ming Yan, Liang
Zhang, Bo Zhang, Chen Li, Ji Zhang, Qin Jin, and Fei
Huang. 2024b. mPLUG-DocOwl2: High-resolution
compressing for OCR-free multi-page document un-
derstanding.arXiv preprint arXiv:2409.03420.
Yulong Hui, Yao Lu, and Huanchen Zhang. 2024. UDA:
A benchmark suite for retrieval augmented generation
in real-world document analysis. InAdvances in
Neural Information Processing Systems 37: Datasets
and Benchmarks Track.
Chelsi Jain, Yiran Wu, Yifan Zeng, Jiale Liu, Shengyu
Dai, Zhenwen Shao, Qingyun Wu, and Huazheng
Wang. 2025. SimpleDoc: Multi-modal document un-
derstanding with dual-cue page retrieval and iterative
refinement. InProceedings of the 2025 Conference
on Empirical Methods in Natural Language Process-
ing, pages 28410–28427, Suzhou, China. Association
for Computational Linguistics.
Omar Khattab and Matei Zaharia. 2020. ColBERT: Effi-
cient and effective passage search via contextualized
late interaction over BERT. InProceedings of the
43rd International ACM SIGIR Conference on Re-
search and Development in Information Retrieval,
pages 39–48.
Patrick Lewis, Ethan Perez, Aleksandra Piktus, Fabio
Petroni, Vladimir Karpukhin, Naman Goyal, Hein-
rich Küttler, Mike Lewis, Wen-tau Yih, Tim Rock-
täschel, Sebastian Riedel, and Douwe Kiela. 2020.
Retrieval-augmented generation for knowledge-
intensive NLP tasks. InAdvances in Neural Informa-
tion Processing Systems 33.
Yuliang Liu, Zhang Li, Biao Yang, Chunyuan Li, Xu-
Cheng Yin, Cheng-Lin Liu, Lianwen Jin, and Xiang
Bai. 2024. TextMonkey: An OCR-free large mul-
timodal model for understanding document.arXiv
preprint arXiv:2403.04473.Chuwei Luo, Yufan Shen, Zhaoqing Zhu, Qi Zheng,
Zhi Yu, and Cong Yao. 2024. LayoutLLM: Lay-
out instruction tuning with large language models
for document understanding. InProceedings of the
IEEE/CVF Conference on Computer Vision and Pat-
tern Recognition.
Tengchao Lv, Yupan Huang, Jingye Chen, Yuzhong
Zhao, Yilin Jia, Lei Cui, Shuming Ma, Yaoyao Chang,
Shaohan Huang, Wenhui Wang, Li Dong, Weiyao
Luo, Shaoxiang Wu, Guoxin Wang, Cha Zhang, and
Furu Wei. 2023. KOSMOS-2.5: A multimodal liter-
ate model.arXiv preprint arXiv:2309.11419.
Ahmad Mohammadshirazi, Pinaki Prasad Guha Neogi,
Ser-Nam Lim, and Rajiv Ramnath. 2024. DLaV A:
Document language and vision assistant for answer
localization with enhanced interpretability and trust-
worthiness.arXiv preprint arXiv:2412.00151.
Mor Shpigel Nacson, Aviad Aberdam, Roy Ganz, Elad
Ben Avraham, Alona Golts, Yair Kittenplon, Shai
Mazor, and Ron Litman. 2025. DocVLM: Make
your VLM an efficient reader. InProceedings of
the IEEE/CVF Conference on Computer Vision and
Pattern Recognition, pages 29005–29015.
OpenSearch-AI. 2026. Ops-Colqwen3: State-of-
the-Art Multimodal Embedding Model for Visual
Document Retrieval. https://huggingface.co/
OpenSearch-AI/Ops-Colqwen3-4B.
Qwen Team. 2025. Qwen3-VL technical report.arXiv
preprint arXiv:2511.21631.
Tobias Schimanski, Imene Kolli, Yu Fan, Ario Saeid
Vaghefi, Jingwei Ni, Elliott Ash, and Markus Leip-
pold. 2026. pdfQA: Diverse, challenging, and real-
istic question answering over PDFs.arXiv preprint
arXiv:2601.02285.
Zhihong Shao, Peiyi Wang, Qihao Zhu, Runxin Xu,
Junxiao Song, Xiao Bi, Haowei Zhang, Mingchuan
Zhang, Y . K. Li, Y . Wu, and Daya Guo. 2024.
DeepSeekMath: Pushing the limits of mathematical
reasoning in open language models.arXiv preprint
arXiv:2402.03300.
Yaya Shi, Jingyun Zhang, Yuliang Liu, Hongliang Li,
and Xiang Bai. 2026. URaG: Unified retrieval and
generation for multimodal document understanding.
arXiv preprint arXiv:2603.12189.
Li Sun, Liu He, Shuyue Jia, Yangfan He, and Chenyu
You. 2025. DocAgent: An agentic framework for
multi-modal long-context document understanding.
InProceedings of the 2025 Conference on Empiri-
cal Methods in Natural Language Processing, pages
17701–17716, Suzhou, China. Association for Com-
putational Linguistics.
Manan Suri, Puneet Mathur, Franck Dernoncourt,
Kanika Goswami, Ryan A. Rossi, and Dinesh
Manocha. 2025. VisDoM: Multi-document QA with
visually rich elements using multimodal retrieval-
augmented generation. InProceedings of the 2025

Conference of the Nations of the Americas Chap-
ter of the Association for Computational Linguistics:
Human Language Technologies (Volume 1: Long Pa-
pers), pages 6088–6109, Albuquerque, New Mexico.
Association for Computational Linguistics.
Ryota Tanaka, Kyosuke Nishida, Kosuke Nishida, Taku
Hasegawa, Itsumi Saito, and Kuniko Saito. 2023.
SlideVQA: A dataset for document visual ques-
tion answering on multiple images.arXiv preprint
arXiv:2301.04883.
Zineng Tang, Ziyi Yang, Guoxin Wang, Yuwei Fang,
Yang Liu, Chenguang Zhu, Michael Zeng, Cha
Zhang, and Mohit Bansal. 2023. UDOP: Unified
document processing with vision, text and layout. In
Proceedings of the IEEE/CVF Conference on Com-
puter Vision and Pattern Recognition, pages 24423–
24433.
Rubèn Tito, Dimosthenis Karatzas, and Ernest Val-
veny. 2022. Hierarchical multimodal transform-
ers for multi-page DocVQA.arXiv preprint
arXiv:2212.05935.
Jordy Van Landeghem, Rubén Tito, Łukasz Borchmann,
Michał Pietruszka, Paweł Józiak, Rafał Powalski,
Dawid Jurkiewicz, Mickaël Coustaty, Bertrand Ack-
aert, Ernest Valveny, Matthew Blaschko, Sien Moens,
and Tomasz Stanisławek. 2023. Document under-
standing dataset and evaluation (DUDE).arXiv
preprint arXiv:2305.08455.
Dongsheng Wang, Natraj Raman, Mathieu Sibue,
Zhiqiang Ma, Petr Babkin, Simerjot Kaur, Yulong
Pei, Armineh Nourbakhsh, and Xiaomo Liu. 2024a.
DocLLM: A layout-aware generative language model
for multimodal document understanding.arXiv
preprint arXiv:2401.00908.
Yonghui Wang, Shi-Yong Chen, Zhenxing Zhou, Siyi Li,
Haoran Li, Wengang Zhou, and Houqiang Li. 2024b.
Root: Vlm based system for indoor scene understand-
ing and beyond.arXiv preprint arXiv:2411.15714.
Yonghui Wang, Shaokai Liu, Li Li, Wengang Zhou,
and Houqiang Li. 2024c. Swinshadow: Shifted win-
dow for ambiguous adjacent shadow detection.ACM
Transactions on Multimedia Computing, Communi-
cations and Applications, 20(11):1–20.
Yonghui Wang, Wengang Zhou, Hao Feng, Keyi
Zhou, and Houqiang Li. 2023a. Towards im-
proving document understanding: An exploration
on text-grounding via mllms.arXiv preprint
arXiv:2311.13194.
Yonghui Wang, Wengang Zhou, Zhenbo Lu, and
Houqiang Li. 2022. Udoc-gan: Unpaired document
illumination correction with background light prior.
InProceedings of the 30th ACM International Con-
ference on Multimedia, pages 5074–5082.
Yonghui Wang, Wengang Zhou, Yunyao Mao, and
Houqiang Li. 2023b. Detect any shadow: Segmentanything for video shadow detection.IEEE Transac-
tions on Circuits and Systems for Video Technology,
34(5):3782–3794.
Zhaowei Wang, Wenhao Yu, Xiyu Ren, Jipeng Zhang,
Yu Zhao, Rohit Saxena, Liang Cheng, Ginny Wong,
Simon See, Pasquale Minervini, Yangqiu Song,
and Mark Steedman. 2025a. MMLongBench:
Benchmarking long-context vision-language mod-
els effectively and thoroughly.arXiv preprint
arXiv:2505.10610.
Zining Wang, Tongkun Guan, Pei Fu, Chen Duan,
Qianyi Jiang, Zhentao Guo, Shan Guo, Junfeng Luo,
Wei Shen, and Xiaokang Yang. 2025b. Marten: Vi-
sual question answering with mask generation for
multi-modal document understanding. InProceed-
ings of the IEEE/CVF Conference on Computer Vi-
sion and Pattern Recognition.
Haoran Wei, Chenglong Liu, Jinyue Chen, Jia Wang,
Lingyu Kong, Yanming Xu, Zheng Ge, Liang Zhao,
Jianjian Sun, Yuang Peng, Chunrui Han, and Xiangyu
Zhang. 2024. General OCR theory: Towards OCR-
2.0 via a unified end-to-end model.arXiv preprint
arXiv:2409.01704.
Xixi Wu, Yanchao Tan, Nan Hou, Ruiyang Zhang, and
Hong Cheng. 2025. MoLoRAG: Bootstrapping doc-
ument understanding via multi-modal logic-aware
retrieval. InProceedings of the 2025 Conference on
Empirical Methods in Natural Language Processing,
pages 14024–14045, Suzhou, China. Association for
Computational Linguistics.
Shiyu Xia, Junyu Xiong, Haoyu Dong, Jianbo Zhao,
Yuzhang Tian, Mengyu Zhou, Yeye He, Shi Han,
and Dongmei Zhang. 2024. Vision language models
for spreadsheet understanding: Challenges and op-
portunities. InProceedings of the 3rd Workshop on
Advances in Language and Vision Research (ALVR),
pages 116–128.
Junyu Xiong, Yuan Pu, Jia Tang, and Yazhe Niu.
2026. Priorzero: Bridging language priors and
world models for decision making.arXiv preprint
arXiv:2605.12289.
Junyu Xiong, Yonghui Wang, Weichao Zhao, Chenyu
Liu, Bing Yin, Wengang Zhou, and Houqiang Li.
2025. DocR1: Evidence page-guided GRPO for
multi-page document understanding.arXiv preprint
arXiv:2508.07313.
Mingjun Xu, Jinhan Dong, Jue Hou, Zehui Wang, Si-
hang Li, Zhifeng Gao, Renxin Zhong, and Hengxing
Cai. 2025. MM-R5: Multimodal reasoning-enhanced
reranker via reinforcement learning for document re-
trieval.arXiv preprint arXiv:2506.12364.
Tianyu Yang, Terry Ruas, Yijun Tian, Jan Philip Wahle,
Daniel Kurzawe, and Bela Gipp. 2026. ALDEN:
Reinforcement learning for active navigation and evi-
dence gathering in long documents.arXiv preprint
arXiv:2510.25668.

Jiabo Ye, Anwen Hu, Haiyang Xu, Qinghao Ye, Ming
Yan, Yaya Dan, Chenliang Zhao, Guohai Xu, Chen
Li, Junfeng Tian, Qi Qian, Ji Zhang, and Fei Huang.
2023a. UReader: Universal OCR-free visually-
situated language understanding with multimodal
large language model. InFindings of the Associ-
ation for Computational Linguistics: EMNLP 2023,
pages 2841–2858.
Jiabo Ye, Anwen Hu, Haiyang Xu, Qinghao Ye,
Ming Yan, Guohai Xu, Chen Li, Junfeng Tian,
Qi Qian, Ji Zhang, and Fei Huang. 2023b. mPLUG-
DocOwl: Modularized multimodal large language
model for document understanding.arXiv preprint
arXiv:2307.02499.
Wenhan Yu, Zhaoxi Zhang, Wang Chen, Guanqiang Qi,
Weikang Li, Lei Sha, Deguo Xia, and Jizhou Huang.
2025a. SciEGQA: A dataset for scientific evidence-
grounded question answering and reasoning.arXiv
preprint arXiv:2511.15090.
Xinlei Yu, Chengming Xu, Zhangquan Chen, Yudong
Zhang, Shilin Lu, Cheng Yang, Jiangning Zhang,
Shuicheng Yan, and Xiaobin Hu. 2025b. Visual doc-
ument understanding and reasoning: A multi-agent
collaboration framework with agent-wise adaptive
test-time scaling.arXiv preprint arXiv:2508.03404.
Yilun Zhao, Lyuhao Chen, Arman Cohan, and Chen
Zhao. 2024. TaPERA: Enhancing faithfulness and in-
terpretability in long-form table QA by content plan-
ning and execution-based reasoning. InProceedings
of the 62nd Annual Meeting of the Association for
Computational Linguistics (Volume 1: Long Papers),
pages 12824–12840, Bangkok, Thailand. Association
for Computational Linguistics.
Yilun Zhao, Yunxiang Li, Chenying Li, and Rui Zhang.
2022. MultiHiertt: Numerical reasoning over multi
hierarchical tabular and textual data. InProceedings
of the 60th Annual Meeting of the Association for
Computational Linguistics (Volume 1: Long Papers),
pages 6588–6600, Dublin, Ireland. Association for
Computational Linguistics.
Yuanlei Zheng, Pei Fu, Hang Li, Ziyang Wang, Yuyi
Zhang, Wenyu Ruan, Xiaojin Zhang, Zhongyu Wei,
Zhenbo Luo, Jian Luan, Wei Chen, and Xiang Bai.
2026. Doc-V*: Coarse-to-fine interactive visual
reasoning for multi-page document VQA.arXiv
preprint arXiv:2604.13731.
Zhaoqing Zhu, Chuwei Luo, Zirui Shao, Feiyu Gao,
Hangdi Xing, Qi Zheng, and Ji Zhang. 2025. A
simple yet effective layout token in large language
models for document understanding. InProceedings
of the IEEE/CVF Conference on Computer Vision
and Pattern Recognition.
A Detailed Related Work
A.1 Document Visual Question Answering
Recent advancements in DocVQA primar-
ily diverge into OCR-based and OCR-freeparadigms (Wang et al., 2022, 2023b, 2024a,b,c,
2023a). OCR-based methods serialize document
pages into structured textual representations
to serve as inputs for downstream reasoning.
Representative architectures leverage multimodal
pretraining or specialized attention mechanisms
to fuse textual, visual, and layout features,
as exemplified by UDOP (Tang et al., 2023),
DocFormerv2 (Appalaraju et al., 2024), and Do-
cLLM (Wang et al., 2024a). To enhance efficiency
and structured generation, LayoutLLM (Luo
et al., 2024) frames layout-sensitive reasoning
as an instruction-tuning problem, whereas Lay-
TokenLLM (Zhu et al., 2025) employs compact
layout tokens to represent spatial bounding boxes
without expanding lengthy coordinate sequences.
For structured data, TaPERA (Zhao et al., 2024)
incorporates content planning and execution-based
reasoning to improve faithfulness in long-form
table QA. Although highly effective when struc-
tural parsing is reliable, the performance of these
methods remains intrinsically upper-bounded by
cascading parsing errors and the inevitable loss of
visually-situated evidence during text serialization.
OCR-free methods bypass external parsing
pipelines by training VLMs to comprehend docu-
ment images end-to-end. Early foundational archi-
tectures prioritized enhancing reading capabilities
under constrained token budgets; within this cohort,
UReader (Ye et al., 2023a) introduces auxiliary
reading and key-point generation tasks, KOSMOS-
2.5 (Lv et al., 2023) pretrains on text-intensive im-
ages for structured text generation, DocPedia (Feng
et al., 2023) processes visual inputs in the fre-
quency domain, TextMonkey (Liu et al., 2024)
incorporates high-resolution modeling with token
filtering, and the mPLUG-DocOwl series (Ye et al.,
2023b; Hu et al., 2024a,b) employs feature com-
pression to scale from single-page to multi-page
document understanding. More recently, frame-
works have evolved toward specialized reasoning
and optimization protocols. Specifically, GOT (Wei
et al., 2024) frames vision-based reading as a uni-
fied OCR-free generator, whereas DLaV A (Moham-
madshirazi et al., 2024) and Marten (Wang et al.,
2025b) implement visual answer localization and
mask prediction to enhance model interpretabil-
ity. To boost efficiency, DocVLM (Nacson et al.,
2025) uses compact textual queries to lower visual
token costs, while MACT (Yu et al., 2025b) orches-
trates collaborative multi-agent reasoning at test
time. To shift from passive reading to active pol-

icy optimization, DocR1 (Xiong et al., 2025) and
Doc-V⋆(Zheng et al., 2026) leverage evidence-
guided GRPO to explicitly train visual reasoning
trajectories for multi-page environments.
A.2 Retrieval-Augmented Generation
RAG methodologies have evolved from traditional
text passage retrieval to multi-modal and structure-
aware document discovery. Following the text-
centric foundations laid by passage retrieval (Lewis
et al., 2020) and token-level late interaction (Khat-
tab and Zaharia, 2020), visual RAG systems have
increasingly focused on page-level document in-
terfaces. ColPali (Faysse et al., 2025) extends
late interaction to rendered page images, while
M3DocRAG (Cho et al., 2024) establishes this ap-
proach as a core interface for multi-page QA. To
decouple retrieval from reading, MDocAgent (Han
et al., 2025) and SimpleDoc (Jain et al., 2025) uti-
lize multi-module collaboration and iterative cas-
cading, while RagVL (Chen et al., 2025) and MM-
R5 (Xu et al., 2025) employ instruction-tuning
or reinforcement learning to train specialized vi-
sual rerankers. Concurrently, efforts have been
dedicated to structured or active navigation to
handle complex layouts: MoLoRAG (Wu et al.,
2025) leverages page topology graphs for logic-
aware traversal, DocAgent (Sun et al., 2025) and
ALDEN (Yang et al., 2026) introduce memory
feedback and active exploration protocols, and
DMAP (Fu et al., 2026) constructs human-aligned
structural maps to guide global understanding.
B Prompt Templates
B.1 Prompt For Hierarchical Question
Decomposition
The system prompt for hierarchical question de-
composition is detailed in Figure 3. Given the root
question as input, the model is strictly constrained
to output a maximum of five atomic child questions,
each formatted strictly as an interrogative sentence.
B.2 Prompt For EviAgent
The system prompt used for both training and infer-
ence of EVIAGENTis provided in Figure 4. Given
a question and a set of input pages, EVIAGENTis
required to produce structured reasoning, make a
binary evidence decision for each page, and gener-
ate the final answer.C Retrieval Metrics
LetGqdenote the gold evidence-page set for ques-
tionq, and let RK
q= (r 1, . . . , r K)denote the
ranked top- Kretrieved pages. We report three
standard retrieval metrics.
Recall@K.Recall@K measures the fraction of
gold evidence pages recovered in the top- Kresults:
Recall@K(q) =|Gq∩RK
q|
|Gq|.(11)
Higher Recall@K indicates better evidence cover-
age.
NDCG@K.Normalized Discounted Cumulative
Gain emphasizes retrieving gold pages early in the
ranked list. With binary relevance labels reli=
I(ri∈Gq), we compute
DCG@K(q) =KX
i=12reli−1
log2(i+ 1),(12)
and normalize it by the ideal ranking:
NDCG@K(q) =DCG@K(q)
IDCG@K(q).(13)
This metric rewards both correctness and ranking
quality.
MRR@K.Mean Reciprocal Rank focuses on
how early the first relevant page appears:
RR@K(q) =(1
min{i|r i∈Gq},ifG q∩RK
q̸=∅,
0,otherwise.
(14)
MRR@K is the average reciprocal rank over all
questions.
D Training Datasets
Table 6 details the statistics of the multi-page train-
ing data utilized for EVIAGENT. To ensure a bal-
anced and comprehensive optimization, we uni-
formly sample 1,000 instances from each of the
six constitutive datasets, with all selected samples
inherently featuring multi-page visual contexts:
•DUDE(Van Landeghem et al., 2023) and
MP-DocVQA(Tito et al., 2022) supply rich
structural information from real-world com-
plex electronic layouts and scanned industry
documents.

SystemPrompt:You are a retrieval query decomposition assistant.Your task is to decompose a user question into multiple concise search queries that help retrieve documents containing the key information needed to answer the question.Guidelines:1. First identify all key elements in the question, including entities (people, organizations, locations), topics or concepts, numbers or statistics, dates or time constraints, and conditions or qualifiers.2. After identifying these key elements, construct retrieval queries by using one or two of the identified elements to form concise and searchable queries.3. Each query should focus on retrieving documents that mention those specific elements or their relationships.4. Preserve all quantitative constraints, temporal conditions, and logical relationships from the original question.5. Queries should be concise and phrased in a way that could realistically appear in documents.6. Avoid redundant queries that express the same concept.7. Keep the total number of generated queries within {MAX_QUERIES}.8. Return JSON only in the exact format:  {"queries": ["...", "..."]}Do not include explanations, comments, or markdown.Example:Input: "What are the views of 20% of the people in California regarding autonomous vehicles in 2023?"Output: {"queries": ["California residents views on autonomous vehicles", "Where 20% of California residents are mentioned?", "Public opinion on autonomous driving in 2023"]}Figure 3: System prompt for hierarchical question decomposition.
Input Page Evidence Page
Dataset # Samples Min Max Avg Min Max Avg
DUDE 1000 2 20 6.93 1 8 1.16
MP-DocVQA 1000 2 20 6.94 1 1 1.00
MultiHiertt 1000 3 7 4.06 1 3 1.48
pdfQA 1000 4 10 9.95 1 10 1.42
SciEGQA 1000 2 20 11.42 1 2 1.50
SlideVQA 1000 20 20 20.00 1 3 1.40
Table 6: Training statistics for EVIAGENT.
•MultiHiertt(Zhao et al., 2022) and
pdfQA(Schimanski et al., 2026) inject
demanding financial reports and scientific
literature requiring cross-page reasoning over
hierarchical tables and textual narratives.
•SciEGQA(Yu et al., 2025a) provides dense
academic papers with tightly coupled illustra-
tions, charts, and mathematical proofs.
•SlideVQA(Tanaka et al., 2023) introduces
sequential presentation slides characterized
by sparse, highly stylized cross-page visual
elements.
E Case Study
Figure 5 illustrates a concrete execution trace of
our decomposition layer. The complex, multi-hop
root query requires concurrent visual counting andcross-year mathematical synthesis. Directly retriev-
ing pages using this intricate query typically mis-
leads standard semantic retrievers. By partitioning
the root query into three localized, atomic child
questions, our framework effectively simplifies the
retrieval objectives, ensuring all disjoint tabular
and visual evidence pages are fully recalled with-
out cascading omissions.

SystemPrompt:Youareaspecializedmultimodalmodelforfine-grainedmulti-imagevisualunderstandingandevidenceimagelocalization.Youareanexpertatcarefullyinspectingmultipleimages,understandingdetailedvisualandtextualinformationwithineachimage,andidentifyingwhichimagescontainevidencerelevanttoagivenquestion.Youwillbegivensomeimagesalongwithaquestion.Carefullyexamineeveryimage. Foreachimage,performdetailedvisualanalysis,includingtext,layout,objects,diagrams,tables,andanyotherpotentiallyinformativeelements.Clearlydescribewhatinformationappearsineachimageandwhetheritmaycontributetoansweringthequestion.Afteranalyzingallimages,performcross-imagereasoningtodeterminewhichimagescontainrelevantevidencerelatedtothequestion.Someimagesmaynotdirectlyanswerthequestionbutmaystillprovidenecessarysupportinginformation.However,oncetheexistingevidenceissufficienttoanswerthequestiondirectlyandcompletely,donotselectanyadditionalimages.Afteridentifyingtheevidenceimages,youmustuseonlytheinformationfromtheseevidenceimagestofurtherreasonandderivethefinalanswertothequestion.Donotuseinformationfromnon-evidenceimages.Iftheanswercanbedetermined,outputaconciseanswerusingashortphraseorafewwordsonly.Iftheanswercannotbedeterminedbasedontheevidenceimages,output"NotAnswerable".Youroutputmustcontainexactlythreeparts. First,provideyourreasoningprocesswrappedin<think></think>.Inside<think>,yourreasoningMUSTbestructuredintothreeclearlyseparatedparts:     (1)Image-levelanalysis:analyzeeachimageonebyone.     (2)Cross-imagereasoning:explainwhichimagesarerelevantandwhy.     (3)Answerreasoning:basedonlyontheselectedevidenceimages,derivethefinalanswerstepbystep.Second,providetheevidence-imagejudgmentwrappedin<evidence_page></evidence_page>asacomma-separatedsequenceofTorFinimageorder.Thenumberofentriesmustexactlymatchthenumberofinputimages.Third,providethefinalanswerwrappedin<answer></answer>followingtherulesabove.Donotoutputanythingoutsidethesethreesections.Figure 4: System prompt used for EVIAGENTtraining and inference.
Question:What is the sum of the total number of paid search's conversions in the year of 2007, 2008 and the number of green bars in the heroes happen here launch?
Sub-question 1: What is the number of paid search conversions in 2007?Sub-question 2: What is the number of paid search conversions in 2008?Sub-question 3: How many green bars are shown in the “Heroes Happen Here” launch chart?
LLM
9
Figure 5: Case study of hierarchical question decomposition in HIEVI-RAG.