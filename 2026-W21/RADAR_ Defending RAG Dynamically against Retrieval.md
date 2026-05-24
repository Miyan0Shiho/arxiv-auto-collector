# RADAR: Defending RAG Dynamically against Retrieval Corruption

**Authors**: Ziyuan Chen, Yueming Lyu, Yi Liu, Weixiang Han, Jing Dong, Caifeng Shan, Tieniu Tan

**Published**: 2026-05-21 06:25:46

**PDF URL**: [https://arxiv.org/pdf/2605.22041v1](https://arxiv.org/pdf/2605.22041v1)

## Abstract
While RAG systems are increasingly deployed in dynamic web search, temporal volatility amplifies their vulnerability to adversarial attacks. Existing static-oriented defenses struggle to handle evolving threats and incur prohibitive storage costs in dynamic settings. We propose RADAR, a framework that models reliable context selection as a graph-based energy minimization problem, solved exactly via Max-Flow Min-Cut. By incorporating a Bayesian memory node, RADAR recursively updates a belief state instead of archiving raw historical documents, effectively balancing stability against attacks with adaptability to genuine knowledge shifts. Experiments on a novel dynamic dataset show that RADAR achieves superior robustness and response quality with minimal storage overhead compared to the baselines.

## Full Text


<!-- PDF content starts -->

RADAR: Defending RAG Dynamically against Retrieval Corruption
Ziyuan Chen1Yueming Lyu1Yi Liu2Weixiang Han1Jing Dong3Caifeng Shan1Tieniu Tan1
Abstract
While RAG systems are increasingly deployed
in dynamic web search, temporal volatility am-
plifies their vulnerability to adversarial attacks.
Existing static-oriented defenses struggle to han-
dle evolving threats and incur prohibitive storage
costs in dynamic settings. We propose RADAR , a
framework that models reliable context selection
as a graph-based energy minimization problem,
solved exactly via Max-Flow Min-Cut. By incor-
porating a Bayesian memory node, RADAR recur-
sively updates a belief state instead of archiving
raw historical documents, effectively balancing
stability against attacks with adaptability to gen-
uine knowledge shifts. Experiments on a novel
dynamic dataset show that RADAR achieves su-
perior robustness and response quality with mini-
mal storage overhead compared to the baselines.
Codes are available at https://github.
com/Etherealllllll/RADAR_code.
1. Introduction
Large Language Models (LLMs) have demonstrated re-
markable capabilities in natural language understanding
and generation. However, they remain prone to generat-
ing hallucinated or ungrounded content when faced with
knowledge-intensive queries. Retrieval-Augmented Gener-
ation (RAG) (Lewis et al., 2020b; Guu et al., 2020; Asai
et al., 2024) addresses this by incorporating external ev-
idence retrieval into the generation process. Early RAG
systems operated in static settings with a fixed corpus as
the knowledge base. More recently, research has shifted
toward dynamic RAG-based web search (Reddy et al., 2025;
Arora et al., 2025; Zhu, 2025), which continuously absorbs
1School of Intelligence Science and Technology, Nanjing Uni-
versity, Suzhou, China2City University of Hong Kong, Hong
Kong, China3Institute of Automation, Chinese Academy of
Sciences, Beijing, China. Correspondence to: Yueming Lyu
<ymlv@nju.edu.cn >, Yi Liu <97liuyi@ieee.org >, Caifeng Shan
<cfshan@nju.edu.cn>.
Proceedings of the 43rdInternational Conference on Machine
Learning, Seoul, South Korea. PMLR 306, 2026. Copyright 2026
by the author(s).updated information from evolving sources such as the web.
A prominent application is LLM-augmented search engines:
a web search engine retrieves documents relevant to the
user’s query, and the retrieved content is fed into an LLM to
produce a final response grounded in that evidence. Notable
examples include Deepseek (DeepSeek-AI, 2024), Chat-
GPT (OpenAI, 2024), and Grok (xAI, 2025).
Despite its potential, RAG-based web search remains vul-
nerable to adversarial attacks. Specifically, corpus poison-
ing (Zou et al., 2025; Hu et al., 2026) and prompt injec-
tion (Clop & Teglia, 2024) can manipulate LLMs into gen-
erating incorrect or malicious outputs. In dynamic settings,
these threats are compounded by temporal volatility, such as
mutating or transient content, leading to continuous corrup-
tion. This expanded attack surface necessitates defenses that
are resilient not only to static attacks but also to evolving
adversarial strategies.
Existing defense mechanisms are largely designed for static
settings. Heuristic aggregation or filtering (Xiang et al.,
2024) often causes utility loss, while optimization-based
consistency selection (Shen et al., 2025) typically relies on
approximations without strong guarantees. While they al-
leviate certain vulnerabilities and can be naively extended
to dynamic settings, such adaptations are often suboptimal.
Without explicit consideration of temporal dynamics, these
methods fail to maintain robust performance against contin-
uously mutating threats, resulting in a significant drop in
defensive efficacy within dynamic web search contexts.
To address these gaps, we introduce RADAR , a robust frame-
work for dynamic RAG. It formulates reliable context se-
lection as a graph-based energy minimization problem,
solved exactly via max-flow Min-Cut. RADAR utilizes a
Bayesian memory node to recursively update a belief state,
enabling the system to weigh historical consistency against
new evidence. This design effectively resolves the stability-
plasticity dilemma, balancing stability against attacks with
adaptability to legitimate knowledge shifts.
Our contributions are summarized as follows:
•We propose RADAR , which models RAG defense as a
Min-Cut problem, delivering exact and efficient inference
with superior robustness against corpus-based attacks.
•We design a novel dynamic graph construction augmented
with a Bayesian memory node. To the best of our knowl-
1arXiv:2605.22041v1  [cs.CR]  21 May 2026

RADAR: Defending RAG Dynamically against Retrieval Corruption
edge, this is the first defensive approach explicitly tailored
for continuous time-step attacks. By maintaining a recur-
sive belief state, RADAR achieves an effective balance
between historical consistency and newly observed evi-
dence.
•We construct a comprehensive dataset for dynamic RAG
security, simulating evolving adversarial scenarios. Ex-
tensive experiments demonstrate that RADAR achieves
superior defense success rates and response quality com-
pared to existing baselines in both static and dynamic
scenarios.
2. Related Work
2.1. Attacks against RAG-based Web Search
Retrieval-Augmented Generation (RAG) systems are vulner-
able to adversarial attacks (Bagwe et al., 2025; Chaturvedi
et al., 2025; Cho et al., 2024; Jiao et al., 2025; Nazary et al.,
2025) that exploit their reliance on external retrieval. We
focus on corpus-based attacks, which can be commonly
grouped into Prompt Injection Attacks (PIA) and Corpus
Poisoning Attacks.
Prompt Injection Attacksembed malicious instructions
in retrieved documents to override system prompts and hi-
jack outputs. For instance, Backdoored Retrievers (Clop &
Teglia, 2024) use implanted backdoors to prioritize injection-
carrying passages for link insertion or DoS hijacking. Sim-
ilarly, Hidden Parrotet al.(Prompt Security, 2025) poi-
sons vector stores to covertly steer generation via similarity
search.
Corpus Poisoning Attacksinject deceptive documents into
knowledge bases to manipulate retrieved contexts and down-
stream outputs. For instance, PoisonedRAG (Zou et al.,
2025) optimizes minimal injections for effective knowledge
corruption, while BadRAG (Xue et al., 2024) poisons the
retrieval process to induce harmful generations. Further-
more, Topic-FlipRAG (Gong et al., 2025) employs two-
stage perturbations to reverse topic-specific opinions, and
DeRAG (Wang & Yu, 2025) leverages black-box differential
evolution to hijack rankings across diverse RAG systems.
2.2. Defenses
RAG systems have inspired a range of robustness-enhancing
frameworks, which can be broadly categorized into two
classes: document filtering prior to generation and defenses
against adversarial attacks.
Document Pre-processing and Filteringevaluates and
refines retrieved content prior to generation. Self-RAG (Asai
et al., 2024) employs reflection tokens for adaptive retrieval
and self-critique to boost quality; Chain-of-Note (Yu et al.,
2024) generates reading notes to filter noise and enhancerobustness. CRAG (Xiang et al., 2024) utilizes lightweight
evaluators to trigger corrective actions like web search for
reliability, while RA-RAG (Hwang et al., 2025) assesses
source credibility and applies weighted majority voting for
reliability-aware aggregation.
Adversarial Defense Strategiesaim to mitigate retrieval
corruption and ensure reliability. RobustRAG (Xiang et al.,
2024) provides certifiable robustness via formal proofs; In-
structRAG (Wei et al., 2025) enables retrieval denoising
through self-synthesized rationales. AstuteRAG (Wang
et al., 2025) consolidates internal LLM knowledge with re-
trieved data to resolve conflicts, while ReliabilityRAG (Shen
et al., 2025) uses MIS algorithm to filter malicious content
with provable guarantees. However, extending these static
defenses to dynamic settings often incurs high storage over-
head and suboptimal performance.
3. Background and Defense Goals
3.1. RAG Workflow
Static Workflow.A standard RAG system consists of a
retriever Rand a generator G(typically an LLM). Given a
user query q, the retriever searches a corpus Cto return a set
of top-krelevant documents (or passages):
D=R(q,C) ={d 1, d2, ..., d k}.(1)
The generator then produces an answer abased on the query
and the retrieved context:
a=G(q,D).(2)
The goal is for Dto provide reliable external knowledge that
grounds the LLM’s response and reduces hallucinations.
Dynamic Workflow.In real-world applications, a RAG
system typically relies on web search, so its evidence corpus
is a continuously evolving set of documents returned by
the search engine rather than a fixed local collection. We
consider a dynamic setting over discrete time steps t∈
{0,1, . . . , T} . At each step, the retriever returns the top- k
documents D(t)=R(q) , and the generator produces an
updated answer a(t)=G(q,D(t))to reflect the most recent
information. The key challenge is to maintain both accuracy
and robustness when the reliability of D(t)varies over time.
3.2. Threat Model
Attacker’s Goal.The attacker aims to induce the LLM
to produce incorrect answers that directly contradict the
ground truth. For a factual query like “Who is the CEO of
Company X?”, the attacker seeks to mislead the model into
outputting a wrong name, which is an adversarial target.
Attacker’s Background Knowledge.In practice, attackers
can often infer a target RAG system’s external knowledge
2

RADAR: Defending RAG Dynamically against Retrieval Corruption
Figure 1.Overview of RADAR . It generates an atomic answer for each retrieved document, scores entailment and contradiction using an
NLI model, and applies an s-tMin-Cut to select a consistent, reliable subset for final answer generation. The dynamic variant augments
the graph with a memory node to balance stability and plasticity across time steps.
sources (e.g., Wikipedia) through repeated interactions, en-
abling them to craft malicious knowledge artifacts. They
also understand standard RAG workflows, common retrieval
methods and similarity metrics. However, they only have
black-box query access: no model parameters, no retriever
or generator modification, and no ability to intercept or
manipulate user queries.
Attacker’s Capabilities.The attacker can inject and opti-
mize malicious passages in the external corpus Cto max-
imize retrieval likelihood, keeping the attack covert at the
system-state level: defenders is largely unaware of whether
an attack is happening, how many documents are poisoned
(denoted by k′orkt), or which retrieved items are poisoned.
Attack Scenarios.We categorize attacks intostaticand
dynamicsettings based on whether the adversary can contin-
uously inject malicious documents into the RAG system. In
the static setting, the attacker performs a one-time injection
into a fixed corpus snapshot, resulting in a constant number
of malicious documents in retrieval. In the dynamic setting,
the attacker adapts over time, causing the number of mali-
cious retrieved documents kt, and thus the attack scale, to
vary across time steps.
3.3. Defense Goals
RADAR ’s main objective is to sanitize the retrieved con-
textDbefore feeding it to the generator, thereby ensuring
robustness: the RAG system should maintain high utility
by correctly answering queries based on Dclean while ef-
fectively neutralizing the influence of Dadv, achieving both
high response accuracy and low Attack Success Rate (ASR).
4. ProposedRADAR
Overview.To achieve the above goals, we propose
RADAR , a robust defense framework tailored for dynamicRAG (see Figure 1). RADAR first generates an atomic an-
swer for each retrieved document and computes entailment
(M) and contradiction ( C) matrices using an NLI model (He
et al., 2023). These semantic relations are encoded into an
s-tgraph, and a Min-Cut partitions the document nodes
into source and sink sets; the source-side nodes are retained
as reliable evidence for answer generation (§4.1–4.2). In
dynamic settings,RADARintroduces amemory noderepre-
senting the previous answer, with terminal edges updated
via Bayesian inference. This enables the system to adap-
tively retain or discard historical information in response to
new evidence, effectively balancing the stability–plasticity
trade-off (§4.3).
4.1. Reliable Subset Formulation
Given a user query qand a set of retrieved documents D=
{d1, d2, . . . , d k}, our goal is to select a subset of reliable
documents Drel⊆ D to generate the final answer. We
use binary labeling to annotate whether each document is
trustworthy. Let y={y 1, y2, . . . , y k}be a label vector,
where yi= 1indicates that the document diis correct and
reliable, andy i= 0indicates it is incorrect and unreliable.
To assign a label yito each document di, we propose to
minimize an energy function E(y) , formulated as a Markov
Random Field (MRF) to find the Maximum A Posteriori
(MAP) estimate (Boykov et al., 2002), which balances indi-
vidual document confidence with pairwise consistency:
minE(y) =kX
i=1ψu(yi) +X
i,jψp(yi, yj).(3)
Here,unary potential ψu(yi)represents the cost of as-
signing a label yito a document, diandpairwise potential
ψp(yi, yj)represents the penalty for assigning conflicting la-
bels to semantically similar documents. Specifically, ψu(yi)
3

RADAR: Defending RAG Dynamically against Retrieval Corruption
is defined as:
ψu(yi) =y i·Fi+ (1−y i)·Si,(4)
where Sirepresents the benefit of labeling dias correct and
Fithe benefit of labeling dias incorrect. By minimizing
this term, the model assigns yi= 1 whenever Si> F i,
indicating that the document’s reliability exceeds its risk.
Forψp(yi, yj), it is defined using the consistency score Mij:
ψp(yi, yj) =M ij· |yi−yj|.(5)
This term contributes to the energy cost only when yi̸=yj,
which means one document is labeled correct and the other
incorrect, thereby encouraging logical consistency.
Thus, this optimization problem is optimized using the Max-
Flow Min-Cut theorem (Elias et al., 1956). The proof is
provided in Appendix B. By constructing a flow network,
the minimum capacity cut directly corresponds to the mini-
mum energyE(y∗).
4.2. Static Defense via Single-Step Min-Cut
Static Graph Construction.Before static graph construc-
tion, we assess the semantic and logical relationships among
retrieved documents. For each document di, we prompt an
LLM to generate an atomic answer aiand discard diifai
is uninformative. For the remaining documents, we employ
an NLI model to compute two matrices: (i) a similarity
matrix M∈[0,1]k×k, where Mijquantifies the degree of
logical entailment between aiandaj; and (ii) a conflict
matrix C∈[0,1]k×k, where Cijmeasures their logical
contradiction. Details of NLI scoring and atomic answer
generation are provided in Appendices D and E. To capture
global consensus, we compute the eigenvector centrality
v∈RkofM. A higher viindicates that diis more central
to the logical consensus of the retrieved evidence.
Then, we construct a directed graph G= (V, E) , where the
node set Vcomprises a source node s, a sink node t, and
nodes representing each retrieved document {d1, . . . , d k}.
The edge set Eand their capacities are meticulously de-
signed to encode the energy minimization objective from
Eq.(3), ensuring an exact correspondence between the graph
cut and the target energy E(y) . Thus, the edges and their
capacities are defined as follows:
(i) Source Edges ( s→d i).The capacity Siof a source
edge represents the benefit of retaining a document di(i.e.,
labeling it as correct, yi= 1). A higher capacity indicates
a stronger pull to include diin the reliable set, making it
more costly to cut this edge (which corresponds to filtering
the document out). We define Siby integrating the docu-
ment’s eigenvector centrality vi, which reflects its alignment
with the global consensus, with a decay function wrank(i)
of its original retrieval rank, emphasizing higher-rankeddocuments:
Si=vi·wrank(i),(6)
where wrank(i) = exp(−i
k). A large Sistrongly attracts di
to the source side (the “Correct” partition). The denominator
serves as a normalization factor.
(ii) Sink Edges ( di→t).The capacity Fiof a sink edge
represents the benefit of filtering document di(i.e., labeling
it as incorrect, yi= 0). A higher capacity indicates a
stronger push to exclude difrom the reliable set, making it
costly to cut this edge (which corresponds to retaining the
document). We define Fito quantify the conflict between
diand other highly central, and thus presumably reliable,
documents in the retrieved set. It is computed as the average
contradiction score Cijbetween diand all other documents
dj(j̸=i), weighted by the centralityv jofdj:
Fi=P
j̸=iCij·vjP
j̸=ivj.(7)
Intuitively, if diconflicts with high-centrality documents,
Fibecomes large, strongly pushing ditowards the sink side
(the “Incorrect” partition).
(iii) Inter-Document Edges ( di↔d j).To enforce con-
sistency between semantically related documents, we con-
nect every pair of document nodes dianddjwith a pair of
anti-parallel edges, each assigned capacity Mij. This con-
struction emulates an undirected edge with equivalent cut
properties: if dianddjare assigned to different partitions,
the cut incurs a penalty of Mij, thereby discouraging logical
inconsistencies. Here, Mij∈[0,1] is the symmetric consis-
tency score derived from the entailment between the atomic
answers of dianddj. Consequently, edges from document
nodes to terminals sandtencode the unary potential ψu,
while inter-document edges realize the pairwise potential
ψp.
Inference via Min-Cut and Answer Generation.Fol-
lowing the graph construction above, we detail the inference
and answer generation pipeline: solving the Min-Cut prob-
lem, deriving optimal document labels, and producing the
final answer from the selected reliable evidence. Any s-t
cut partitions the document nodes into a source-side set Vs
and a sink-side setV t, inducing a natural binary labeling:
yi=(
1, d i∈ Vs(correct)
0, d i∈ Vt(incorrect).(8)
The total capacity of the cut can be decomposed into three
components:
•P
di∈VtSi, corresponding to unary costsψ u(yi= 0);
•P
di∈VsFi, corresponding to unary costsψ u(yi= 1);
•P
di∈Vs,dj∈VtMij, corresponding to pairwise penalties
ψp(yi̸=yj).
4

RADAR: Defending RAG Dynamically against Retrieval Corruption
Thus, the cut capacity is exactly equal to the energy function:
Capacity(Cut) =E(y).(9)
The set of reliable documents is defined as those assigned
to the source partition: Drel={d i|y∗
i= 1} . To ensure
semantic consistency, we post-process Drelby calculating
the average pairwise cosine similarity of the selected re-
sponses. For each document di∈ D rel, we compute its
average cosine similarity siwith the other selected docu-
ments and exclude those with low agreement, as determined
by a hyperparameter λ. The documents are retained if their
average similaritys iis greater than or equal toλ:
D′
rel={d i∈ Drel|si≥λ}.(10)
The details are in Appendix G. The remaining documents in
D′
relare then concatenated with the original query and used
to prompt the LLM to generate the final answer.
4.3. Dynamic Defense with Memory Node
As defined in Sec. 3.1, real-world RAG systems operate
over continuous time steps t= 1,2, . . . , T , processing an
evolving evidence stream D(t). Na¨ıvely reapplying a static
model at each step disregards temporal continuity, often
yielding unstable predictions when new evidence is sparse
or noisy. Conversely, over-reliance on past answers impedes
adaptation to genuine knowledge updates. To resolve this
stability–plasticity trade-off, we augment the static graph
with aMemory Nodethat encapsulates the system’s state
from the previous time step, enabling coherent integration
of historical knowledge and incoming evidence.
Dynamic Graph Construction.At the initial time step
t= 0 , no historical answer exists. In this case, the dynamic
mechanism naturally reduces to the static variant. We exe-
cute static defense on the initial retrieval set D(0)to generate
the first reliable answer a(0). Then, based on the generated
answer and the single-document answers from the current
step, we compute the prior probabilities π(0)
Sandπ(0)
Ffor
the next round. The priors are obtained by computing the
average similarity and conflict:
π(0)
S=1
|D(0)|X
di∈D(0)M(a(0), di),(11)
π(0)
F=1
|D(0)|X
di∈D(0)C(a(0), di).(12)
Fort >0 , it is necessary to use the dynamic mechanism.
The new dynamic graph G(t)= (V(t), E(t))contains all
nodes from the static case—source s, sink t, and current
retrieved documents D(t), plus the memory node a(t−1).
a(t−1)represents the reliable answer generated at time t−1 .Thus, the edge capacities are defined as follows, with edges
among retrieved new documents, source, and sink following
the same definitions as in Sec. 4.2. The key additions are
the edges connected to the memory nodea(t−1):
(i) Memory-Source Edges ( s→a(t−1)).The capacity of this
edge S(t)
oldrepresents the updated belief that the historical
answer a(t−1)remains correct given the new evidence D(t).
We model this using a Bayesian framework since it naturally
integrates prior beliefs with new evidence. Let π(t−1)
S be the
prior probability of correctness, derived from the average
consistency of a(t−1)within its previous context at t−1 .
The likelihood L(t)
Sis the average entailment score between
the old answer and the new documents, indicating how well
the new evidence supports the history:
L(t)
S=1
|D(t)|X
dk∈D(t)M(a(t−1), dk).(13)
The posterior belief (edge capacity) is computed via Bayes’
theorem:
S(t)
old=π(t−1)
S· L(t)
S
π(t−1)
S· L(t)
S+ (1−π(t−1)
S)·(1− L(t)
S).(14)
A high capacity attracts the memory node to the source side
(the “Correct” partition), signaling that the historical answer
is validated by the new information.
(ii) Memory-Sink Edges ( a(t−1)→t).The capacity of
this edge F(t)
oldrepresents the updated probability that the
historical answer is incorrect (i.e., should be filtered out).
Similarly, we define a prior conflict π(t−1)
F and a likelihood
of conflict L(t)
F, which is the average contradiction score
between the new documents and the old answer:
L(t)
F=1
|D(t)|X
dk∈D(t)C(a(t−1), dk).(15)
The updated capacityF(t)
oldis:
F(t)
old=π(t−1)
F· L(t)
F
π(t−1)
F· L(t)
F+ (1−π(t−1)
F)·(1− L(t)
F).(16)
If new documents explicitly contradict the old answer, this
capacity increases, pushing the memory node a(t−1)to-
wards the “Incorrect” partition.
(iii) Memory-Document Edges ( a(t−1)↔d i).To enforce
logical consistency between history and the present, we
add undirected edges between the memory node a(t−1)and
every current document di∈ D(t). The capacity is set to
their pairwise consistency M(a(t−1), di). This treats the old
answer as a “super-document” that participates in the global
consensus. If the old answer is semantically aligned with
the majority of valid new documents, these edges reinforce
their mutual selection.
5

RADAR: Defending RAG Dynamically against Retrieval Corruption
Inference via Min-Cut and Answer Generation.With
the dynamic graph constructed, we solve the Min-Cut prob-
lem to obtain the optimal partition ( Vt
s,Vt
t). The process
automatically determines whether the historical information
should be retained or discarded. If the memory node a(t−1)
remains on the source side ( Vs), the previous conclusion
is validated by the new evidence; otherwise, if it is cut to
the sink side (V t), it indicates concept falsification, and the
system removes the historical belief.
LetA(t)
rel={a(t)
1, . . . , a(t)
m}denote the set of reliable atomic
answers induced by the source partition,i.e., the atomic
answers associated with all selected nodes in Vs, includ-
ing the memory answer a(t−1)if it is retained. Then we
post-process A(t)
relas in Sec. 4.2 by computing their average
pairwise cosine similarity and excluding isolated atomic an-
swers with low similarity to the rest, forming A′(t)
rel. Finally,
we prompt the generator Gto strictly synthesize a single
coherent conclusion based only on these reliable atomic
answers, producing the final answer:
a(t)=G 
q,A′(t)
rel
.(17)
The coherence of a(t)is then computed and used to update
the priors π(t)
Sandπ(t)
Ffor the next time step, with the
update formulas being the same as in Eqs. (11) and(12),
thereby creating a continuous learning loop. Overall, we
presentRADARin Algo. 1.
5. Experiments
5.1. Experimental Setup
Static Evaluation Datasets.We evaluate RADAR on four
benchmark datasets: RealTimeQA (RQA) (Kasai et al.,
2023) for regular real-time QA snapshots; Natural Questions
(NQ) (Lee et al., 2019) for answering via full Wikipedia
articles; TriviaQA (TQA) (Joshi et al., 2017) for evidence-
grounded trivia; and Bio (Lebret et al., 2016) for generating
long-form biographies from Wikipedia infoboxes.
Dynamic Evaluation Datasets.To evaluate the robustness
of RAG systems in dynamic environments, we construct
a time-evolving QA benchmark of 500 open-domain ques-
tions whose ground-truth answers change over time. For
each question and timestamp, we retrieve the top-50 relevant
webpages via SerpApi’s Google Search API (SerpApi, LLC,
2026), forming temporally indexed evidence snapshots. Us-
ing DeepSeek (DeepSeek-AI, 2024), we inject two types of
adversarial artifacts into these snapshots: (i) poisoned doc-
uments containing fabricated but plausible claims aligned
with specific questions, and (ii) prompt-injection payloads
embedded in retrieved text that hijack the generator to dis-
regard the user query and output attacker-specified content.
Representative examples are provided in Appendix P, and
dataset statistics are provided in Appendix O.Algorithm 1RobustRADARDefense for Dynamic RAG
1:Input:queryq, retrieval stream{D(t)}T
t=0
2:Output:answers{a(t)}T
t=0
3:Initializeπ S←0, π F←0
4:fort= 0toTdo
5:A(t)←AtomicGen(D(t))
6:(M(t), C(t))←NLI(A(t)),v(t)←eigcen(M(t))
7:G(t)←BuildGraph(D(t), M(t), C(t), v(t))
8:ift >0then
9:G(t)←AddHistory(G(t), a(t−1),D(t), πS, πF)
10:// Inject Bayesian memory node for temporal
consistency
11:end if
12:(A(t)
rel,D(t)
rel)←Select(MinCut(G(t)))
13:DropIsolated(A(t)
rel,D(t)
rel)
14:ift= 0then
15:a(t)← G(q,D(t)
rel)
16:else
17:a(t)← G(q,A(t)
rel)
18:// Generate from reliable atomic claims
19:end if
20:π S←1
|D(t)|P
diM(a(t), di)
21:π F←1
|D(t)P
diC(a(t), di)
22:// Update belief priors for next time step
23:Returna(t)
24:end for
Baselines.We compare RADAR against a standard Vanilla
RAG pipeline, which directly prompts the generator with
retrieved documents, and several robustness-oriented base-
lines: RobustRAG (Xiang et al., 2024), AstuteRAG (Wang
et al., 2025), InstructRAG (Wei et al., 2025), and Reliabili-
tyRAG (Shen et al., 2025).
RAG Settings.We employ three LLMs as generators in our
RAG: DeepSeek (DeepSeek-AI, 2024), GPT-4o (OpenAI,
2024), and Grok-4-fast (xAI, 2025). We use DeBERTa-
v3 (He et al., 2023) and NLI model to compute MandC.
We also conducted experiments under two retrieval settings:
top-k= 10and top-k= 50retrieved documents.
Attack Settings.We evaluate Prompt Injection Attacks
(PIA) and Corpus Poisoning Attacks across different re-
trieval depths: targeting rank 1 (highest-ranked) and rank
10 (lowest-ranked) for k= 10 , and ranks 1, 25, and 50 for
k= 50 . Multi-position attacks are further detailed in Ap-
pendix L to simulate comprehensive adversarial scenarios.
Metrics.For QA datasets, we employ Answer Accuracy
(Acc.) to match time-specific ground truth and Attack
Success Rate (ASR) to measure the frequency of attacker-
targeted outputs. For long-form Bio generation, we use
DeepSeek as an LLM-as-a-Judge to score accuracy, rele-
6

RADAR: Defending RAG Dynamically against Retrieval Corruption
Table 1.Performance ofRADARand baseline methods on the RQA dataset using DeepSeek.
Scenario PosVanilla RAG AstuteRAG InstructRAG RobustRAG ReliabilityRAG RADAR(Ours)
Acc.↑ASR.↓ Acc.↑ASR.↓ Acc.↑ASR.↓ Acc.↑ASR.↓ Acc.↑ASR.↓ Acc.↑ASR.↓
Top-k= 10
Benign – 75.0 – 35.0 – 73.0 – 69.0 – 75.0 – 75.0 –
PIAPos 1 25.0 74.0 25.0 1.0 23.0 68.0 64.0 7.0 69.015.0 69.011.0
Pos 10 59.0 28.0 23.0 1.0 66.0 5.0 69.0 4.0 74.0 6.0 75.05.0
PoisonPos 1 38.0 57.0 21.0 15.0 29.0 50.0 62.0 13.0 70.014.0 70.011.0
Pos 10 58.0 35.0 36.0 4.0 54.0 14.0 70.0 5.0 75.0 6.0 75.06.0
Top-k= 50
Benign – 75.0 – 41.0 – 62.0 – 71.0 – 75.0 – 75.0 –
PIAPos 1 35.0 65.0 21.0 5.0 33.0 46.0 69.0 9.0 67.0 18.0 72.05.0
Pos 25 68.0 15.0 38.0 2.0 61.0 3.0 69.0 4.0 74.0 3.0 76.03.0
Pos 50 60.0 23.0 34.0 2.0 63.0 4.0 71.0 4.0 76.03.0 76.03.0
PoisonPos 1 44.0 53.0 21.0 10.0 39.0 35.0 67.0 20.0 64.0 19.0 71.07.0
Pos 25 65.0 26.0 37.0 3.0 57.0 21.0 71.0 5.0 70.0 7.0 76.05.0
Pos 50 74.0 12.0 29.0 2.0 55.0 17.0 71.0 5.0 75.0 3.0 76.03.0
Table 2.Performance ofRADARand baseline methods on Bio dataset using DeepSeek.
Method MetricBenign PIA Poison
k= 10k= 50k= 10k= 50 k= 10k= 50
Pos 1 Pos 10 Pos 1 Pos 25 Pos 50 Pos 1 Pos 10 Pos 1 Pos 25 Pos 50
Vanilla RAGAcc.↑ 77.2 78.4 24.8 9.6 19.0 21.2 10.2 59.6 31.0 57.6 48.2 29.2
Rel.↑ 77.8 79.2 24.2 9.6 18.2 21.0 10.0 62.6 38.2 58.4 52.0 38.8
Coh.↑ 85.0 84.4 28.4 13.8 22.6 24.8 12.4 72.4 48.0 70.4 57.4 46.2
AstuteRAGAcc.↑ 87.8 86.0 54.8 63.8 67.6 68.2 73.8 66.0 68.6 65.0 70.0 65.4
Rel.↑ 88.2 85.2 57.8 65.2 73.077.2 77.8 67.6 69.8 63.0 68.2 65.0
Coh.↑ 90.8 88.0 63.4 70.2 77.8 82.8 82.6 74.6 75.6 73.2 79.6 76.6
InstructRAGAcc.↑ 74.2 73.8 68.6 71.8 74.076.676.4 74.4 73.2 77.4 76.680.2
Rel.↑ 69.4 70.4 64.6 64.6 70.0 73.2 77.4 72.2 70.4 74.0 74.0 77.8
Coh.↑ 78.4 78.2 73.0 75.6 78.8 81.4 82.4 80.4 78.0 80.8 81.4 84.2
RobustRAGAcc.↑ 60.4 60.6 57.6 55.8 57.2 61.2 56.0 60.4 52.8 52.2 65.4 68.0
Rel.↑ 51.4 62.2 49.2 47.6 57.0 62.0 57.6 53.8 47.4 55.2 67.2 68.0
Coh.↑ 72.4 71.6 70.2 66.2 69.0 73.0 68.0 73.4 66.6 67.6 77.0 80.4
ReliabilityRAGAcc.↑ 70.6 75.4 66.6 67.0 71.8 74.4 74.6 65.2 75.0 68.4 80.0 73.0
Rel.↑ 70.6 78.4 67.6 68.0 71.2 76.0 76.4 67.0 76.8 71.481.874.6
Coh.↑ 78.4 83.2 75.2 74.0 77.4 83.0 81.4 75.8 84.0 77.887.479.6
RADAR(Ours)Acc.↑ 76.6 76.8 72.0 75.6 76.875.880.6 84.4 79.0 81.2 80.478.6
Rel.↑ 74.0 77.0 71.0 76.4 75.875.8 77.0 82.6 77.2 80.681.279.4
Coh.↑ 81.4 83.4 79.2 83.0 83.4 83.4 84.2 88.2 84.4 85.485.685.6
vance, and coherence based on Wikipedia references.
5.2. Defense Performance in Static Environments
Our static experiments highlight two RADAR strengths: (i)
preserving utility in benign settings and (ii) enhancing ro-
bustness against prompt injection and corpus poisoning with-
out compromising quality.
Benign Performance. RADAR maintains competitive accu-
racy, matching Vanilla RAG on RQA (75.0%) as shown in
Table 1. In long-form Bio tasks, while AstuteRAG leads
due to its specialized noise mitigation, RADAR consistentlyexceeds defense-oriented baselines RobustRAG and Reli-
abilityRAG in accuracy, relevance, and coherence. Our
sanitization mechanism thus avoids the utility loss typical
of heuristic filtering.
Robustness under Poisoning Attacks. RADAR yields the
best robustness-utility trade-off across datasets and attack
positions, achieving low ASR while maintaining the highest
accuracy. It remains effective even when Pos 1 is com-
promised. As shown in Table 1, for top- k= 10 on RQA
with PIA at Pos 10, it achieves 75.0% accuracy with 5.0%
ASR. And for top- k= 50 on RQA with PIA at Pos 1, it
maintains 72.0% accuracy and 5.0% ASR despite increased
7

RADAR: Defending RAG Dynamically against Retrieval Corruption
Table 3.Performance under evolving evidence streams with top-k= 50using DeepSeek under the cumulative snapshot setting.
Attack PosVanilla RAG AstuteRAG InstructRAG RobustRAG ReliabilityRAG RADAR(Ours)
Acc.↑ASR.↓ Acc.↑ASR.↓ Acc.↑ASR.↓ Acc.↑ASR.↓ Acc.↑ASR.↓ Acc.↑ASR.↓
Benign – 70.76 – 73.48 – 73.41 – 67.50 – 70.63 – 74.02–
PIAPos 1 12.41 87.01 54.25 12.99 61.42 34.46 61.29 18.62 53.61 42.67 63.6017.85
Pos 25 29.88 67.37 62.51 9.40 65.77 28.79 66.92 9.98 67.62 9.34 70.128.94
Pos 50 19.32 79.40 59.18 9.66 59.69 34.68 67.43 7.87 68.71 5.95 70.056.01
PoisonPos 1 25.72 53.17 47.92 16.63 53.49 24.50 44.98 34.29 53.87 27.51 63.6017.53
Pos 25 35.44 43.44 58.41 10.62 55.34 25.46 66.28 11.00 67.69 7.55 69.417.22
Pos 50 30.77 46.51 57.77 9.60 55.92 23.74 66.92 8.64 67.88 5.25 70.375.95
Table 4.Performance under evolving evidence streams with top-k= 50using DeepSeek under the lightweight history setting.
Attack PosVanilla RAG AstuteRAG InstructRAG RobustRAG ReliabilityRAG RADAR(Ours)
Acc.↑ASR.↓ Acc.↑ASR.↓ Acc.↑ASR.↓ Acc.↑ASR.↓ Acc.↑ASR.↓ Acc.↑ASR.↓
Benign – 70.70 – 64.17 – 83.10– 59.88 – 72.55 – 74.02 –
PIAPos 1 37.68 60.72 60.08 7.74 57.38 32.69 53.49 20.15 50.74 43.12 63.6017.85
Pos 25 36.98 58.35 64.68 8.25 55.34 25.46 59.31 12.92 66.53 10.81 70.128.94
Pos 50 14.84 82.47 62.06 9.85 42.22 49.07 59.12 12.80 68.45 6.78 70.056.01
noise and attack surface. Due to space constraints, the test
results for NQ and TQA and the test results for GPT-4o and
Grok-4-fast are presented in Appendix K.
Long-form Generation Robustness under Poisoning At-
tacks.As shown in Table 2, RADAR dominates Bio task
performance under PIA and poisoning across most positions.
Notably, under PIA attacks at position 50 when top- k= 50 ,
RADAR maintains a high factual accuracy of 80.6%, whereas
Vanilla RAG’s performance collapses to 10.2%. Further-
more, in poisoning scenarios when top- k= 10 at Pos 1,
RADAR achieves a peak accuracy of 84.4%, outperforming
all baseline models. It consistently improves factual ac-
curacy, relevance, and coherence. The method effectively
blocks adversarial inputs while maintaining high-quality
discourse in long-form outputs.
5.3. Defense Performance in Dynamic Environments
Experimental Protocol.Most existing RAG defenses are
designed for static snapshots of retrieved evidence and do
not naturally handle time-evolving streams. To enable a
fair comparison in dynamic scenarios, we explore two ways
to adapt static baselines to the temporal setting: (1) cu-
mulative snapshot: at each time step t, newly retrieved
documents are prepended to all previously seen documents
{D0, D1, . . . , D t−1}, forming an expanding evidence pool
D≤t; (2) lightweight history: instead of storing the full doc-
ument history, only the answer from the previous time step
is appended to the current prompt. Both approaches are
applied to all baselines.
Benign Performance.As shown in Table 3 and Table 4, inthe no-attack setting, RADAR achieves a peak accuracy of
74.02%, surpassing vanilla RAG in both temporal settings.
This is because the cumulative snapshot approach may feed
obsolete or incorrect evidence from earlier time steps into
the LLM, while the lightweight history approach may mis-
lead the model with previous answers that are no longer
valid. These results show that RADAR not only defends
against attacks but also improves correctness in dynamic
RAG.
Performance Under Time-Evolving Attacks.Table 3
presents a comparison between RADAR and other baselines
under the cumulative snapshot setting, reporting accuracy
and ASR on evolving evidence streams with time-varying
attacks. RADAR offers a better robustness–utility trade-off
than RobustRAG and ReliabilityRAG, achieving higher ac-
curacy and maintaining lower or stable ASR across injec-
tion positions. Notably, under the most challenging Pos 1
PIA attack, RADAR achieves 63.60% accuracy, significantly
outperforming RobustRAG (61.29%) and ReliabilityRAG
(53.61%). RADAR excels when adversarial content is in-
jected into high-ranked evidence, better protecting critical
passages. For mid-rank and tail-rank injections, RADAR
sustains about 70% accuracy with low ASR. Table 4 further
compares RADAR with baselines under the lightweight his-
tory setting, where only the previous answer is appended to
the current prompt. Under PIA attacks, RADAR consistently
achieves the highest accuracy across injection positions, sur-
passing the best baseline by 3.52% at Pos 1 and 5.44% at
Pos 25, while maintaining low ASR (17.85% and 8.94%,
respectively), demonstrating its superior robustness against
adversarial injections even under lightweight history adapta-
tion.
8

RADAR: Defending RAG Dynamically against Retrieval Corruption
Table 5.RADAR’s performance on different NLI models under PIA attack on RQA using Deepseek.
NLITop-k= 10 Top-k= 50
Pos 1 Pos 10 Pos 1 Pos 25 Pos 50
Acc.↑ASR.↓ Acc.↑ASR.↓ Acc.↑ASR.↓ Acc.↑ASR.↓ Acc.↑ASR.↓
DeBERTa-v3 69.0 11.0 75.0 5.0 72.0 5.0 76.0 3.0 76.0 3.0
BART 69.0 11.0 75.0 6.0 72.0 7.0 76.0 4.0 76.0 4.0
ModernBERT 69.0 12.0 75.0 6.0 72.0 7.0 76.0 4.0 76.0 4.0
Figure 2.Sensitivity of the post-processing thresholdλ.
Table 6. RADAR ’s performance under random perturbations of NLI
with Top-k= 10 under PIA attack on RQA using Deepseek.
Perturbation RatePos 1 Pos 10
Acc.↑ASR.↓ Acc.↑ASR.↓
0 69.0 11.0 75.0 5.0
0.1 67.0 10.0 74.0 6.0
0.3 68.0 12.0 73.0 6.0
0.5 64.0 18.0 78.0 7.0
5.4. Hyperparameter Sensitivity
We evaluate the sensitivity of the post-processing threshold
λon RQA under prompt injection attacks using DeepSeek.
As shown in Figure 2, varying λfrom 0.1to0.5has min-
imal impact on accuracy and ASR, with performance re-
maining stable across injection positions. ASR is slightly
more sensitive, following a decrease-then-degrade pattern
asλincreases. Across retrieval sizes and injection ranks,
λ= 0.3 provides the most consistent robustness gains with
negligible accuracy loss, so we adoptλ= 0.3by default.
5.5. NLI Sensitivity
We ablate different NLI models (DeBERTa-v3 (He
et al., 2023), BART (Lewis et al., 2020a), and Modern-
BERT (Warner et al., 2025)) under PIA on RealTimeQA in
Table 5 and observe only minor changes in Acc. and ASR,
indicating low sensitivity to the NLI choice. We further
perturb NLI outputs by randomly replacing them with prob-
abilities of 0.1, 0.3, and 0.5 in Table 6, and observe only
slight accuracy drops, reinforcing this low sensitivity.Table 7. RADAR ’s performance under none-adaptive attack and
adaptive attack on RQA using Deepseek.
Top-k PositionNone-Adaptive Adaptive
Acc.↑ASR.↓ Acc.↑ASR.↓
10Pos 1 69.0 11.0 69.0 10.0
Pos 10 75.0 5.0 74.0 6.0
50Pos 1 72.0 5.0 72.0 8.0
Pos 25 76.0 3.0 75.0 5.0
Pos 50 76.0 3.0 74.0 5.0
5.6. NLI Robustness under Adaptive Attack
To directly assess whether the NLI-based entailment/contra-
diction signals remain reliable under adversarially written
or stylistically camouflaged injected text, we additionally
evaluate RADAR under the adaptive attack setting proposed
by ReliabilityRAG. Specifically, besides the standard non-
adaptive injection, we consider the adaptive attack that in-
duces ambiguous answers (e.g., “A or B” while the correct
answer is A) to bypass NLI contradiction checks.
Our results in Table 7 show that RADAR remains largely
stable under this stronger attack: across both top- k= 10 and
top-k= 50 settings, the performance under adaptive attack
is very close to that under non-adaptive attack, with only
marginal differences. This suggests that RADAR remains
effective even when the injected text is adversarially crafted
to camouflage itself against NLI-based defenses.
6. Conclusion
In this paper, we present RADAR , a defense framework for
dynamic RAG that treats context sanitization as a Min-
Cut problem. By merging graph-theoretic inference with
a Bayesian Memory Node,RADARbalances adversarial re-
silience and knowledge adaptation with minimal storage
overhead. We also provide a comprehensive dynamic at-
tack dataset as a benchmark. As AI systems pivot toward
real-time data, such mathematically grounded, time-aware
resilience is vital for next-generation trustworthy AI agents.
9

RADAR: Defending RAG Dynamically against Retrieval Corruption
Impact Statement
This work introduces RADAR , a robust and storage-efficient
defense framework for dynamic Retrieval-Augmented Gen-
eration (RAG) systems operating in adversarial, time-
evolving environments such as web search. By formulating
context sanitization as a graph-based energy minimization
problem solved via Min-Cut and incorporating a Bayesian
memory node, RADAR effectively balances stability against
retrieval corruption with adaptability to genuine knowledge
updates, without archiving historical documents. This ap-
proach significantly reduces storage overhead (from tens
of MBs to ∼1 KB per query) while improving robustness
and answer accuracy under evolving attacks. As real-time
RAG becomes integral to AI assistants, search engines, and
decision-support tools, RADAR provides a practical, mathe-
matically grounded mechanism to enhance trustworthiness,
mitigate prompt injection and corpus poisoning risks, and
promote the safe deployment of LLM-powered systems in
dynamic real-world settings.
Acknowledgements
This work was supported by the New Generation Arti-
ficial Intelligence-National Science and Technology Ma-
jor Project (2025ZD0123504), the National Natural Sci-
ence Foundation of China (Grants 62502200), the Jiangsu
Provincial Science and Technology Major Project (Grant
BG2024042), and the Natural Science Foundation of
Jiangsu Province (Grants BK20251203).
References
Arora, S., Khan, H., Sun, K., Dong, X. L., Choudhary, S.,
Moon, S., Zhang, X., Sagar, A., Appini, S. T., Patnaik, K.,
et al. Stream rag: Instant and accurate spoken dialogue
systems with streaming tool usage. InarXiv preprint
arXiv:2510.02044, 2025.
Asai, A., Wu, Z., Wang, Y ., Sil, A., and Hajishirzi, H. Self-
RAG: Learning to retrieve, generate, and critique through
self-reflection. InProc. of ICLR, 2024.
Bagwe, G., Chaturvedi, S. S., Ma, X., Yuan, X., Wang,
K.-C., and Zhang, L. E. Your rag is unfair: Exposing
fairness vulnerabilities in retrieval-augmented generation
via backdoor attacks. InProc. of EMNLP, 2025.
Boykov, Y ., Veksler, O., and Zabih, R. Fast approximate
energy minimization via graph cuts.IEEE Transactions
on pattern analysis and machine intelligence, 23(11):
1222–1239, 2002.
Chaturvedi, S. S., Bagwe, G., Zhang, L. E., and Yuan, X.
Aip: Subverting retrieval-augmented generation via ad-
versarial instructional prompt. InProc. of EMNLP, 2025.Cho, S., Jeong, S., Seo, J., Hwang, T., and Park, J. C. Typos
that broke the rag’s back: Genetic attack on rag pipeline
by simulating documents in the wild via low-level pertur-
bations. InFindings of EMNLP, 2024.
Clop, C. and Teglia, Y . Backdoored retrievers for prompt in-
jection attacks on retrieval augmented generation of large
language models. InarXiv preprint arXiv:2410.14479,
2024.
DeepSeek-AI. DeepSeek-V3 technical report. Technical
Report arXiv:2412.19437, DeepSeek-AI, 2024. URL
https://arxiv.org/abs/2412.19437 . Ac-
cessed: 2026-01-27.
Elias, P., Feinstein, A., and Shannon, C. A note on the
maximum flow through a network.IRE Transactions on
Information Theory, 2(4):117–119, 1956.
Gong, Y ., Chen, Z., Chen, M., Yu, F., Lu, W., Wang, X., Liu,
X., and Liu, J. Topic-fliprag: Topic-orientated adversar-
ial opinion manipulation attacks to retrieval-augmented
generation models.arXiv preprint arXiv:2502.01386,
2025.
Guu, K., Lee, K., Tung, Z., Pasupat, P., and Chang, M.
Retrieval augmented language model pre-training. In
Proc. of ICML, 2020.
He, P., Gao, J., and Chen, W. Debertav3: Improving
deberta using electra-style pre-training with gradient-
disentangled embedding sharing. InProc. of ICLR, 2023.
Hu, H., Jiang, Z., Lyu, Y ., Zhang, J., Liu, Y ., and Chow, K.-
H. Confundo: Learning to generate robust poison for prac-
tical rag systems. InarXiv preprint arXiv:2602.06616,
2026.
Hwang, J., Park, J., Park, H., Kim, D., Park, S., and Ok,
J. Retrieval-augmented generation with estimation of
source reliability. InProc. of EMNLP, 2025.
Jiao, Y ., Wang, X., and Yang, K. Pr-attack: Coordinated
prompt-rag attacks on retrieval-augmented generation in
large language models via bilevel optimization. InProc.
of SIGIR, 2025.
Joshi, M., Choi, E., Weld, D., and Zettlemoyer, L. TriviaQA:
A large scale distantly supervised challenge dataset for
reading comprehension. InProc. of ACL, 2017.
Kasai, J., Sakaguchi, K., Le Bras, R., Asai, A., Yu, X.,
Radev, D., Smith, N. A., Choi, Y ., Inui, K., et al. Realtime
qa: What’s the answer right now? 2023.
Kolmogorov, V . and Zabin, R. What energy functions can be
minimized via graph cuts?IEEE transactions on pattern
analysis and machine intelligence, 26(2):147–159, 2004.
10

RADAR: Defending RAG Dynamically against Retrieval Corruption
Lebret, R., Grangier, D., and Auli, M. Neural text generation
from structured data with application to the biography
domain. InProc. of EMNLP, 2016.
Lee, K., Chang, M.-W., and Toutanova, K. Latent retrieval
for weakly supervised open domain question answering.
InProc. of ACL, 2019.
Lewis, M., Liu, Y ., Goyal, N., Ghazvininejad, M., Mo-
hamed, A., Levy, O., Stoyanov, V ., and Zettlemoyer, L.
Bart: Denoising sequence-to-sequence pre-training for
natural language generation, translation, and comprehen-
sion. InProc. of ACL, 2020a.
Lewis, P., Perez, E., Piktus, A., Petroni, F., Karpukhin, V .,
Goyal, N., K ¨uttler, H., Lewis, M., Yih, W.-t., Rockt ¨aschel,
T., et al. Retrieval-augmented generation for knowledge-
intensive nlp tasks. InProc. of NeurIPS, 2020b.
Nazary, F., Deldjoo, Y ., and Noia, T. d. Poison-rag: Ad-
versarial data poisoning attacks on retrieval-augmented
generation in recommender systems. InProc. of ECIR,
2025.
OpenAI. Gpt-4o system card. Technical Report GPT-4o
System Card, OpenAI, 2024. URL https://cdn.
openai.com/gpt-4o-system-card.pdf . Ac-
cessed: 2026-01-27.
Prompt Security. The hidden parrot: Stealthy prompt
injection and poisoning in rag systems via vector
database embeddings. GitHub repository, 2025. URL
https://github.com/prompt-security/
RAG_Poisoning_PoC . Proof-of-concept for stealthy
prompt injection in RAG. Accessed January 2026.
Reddy, R. G., Dixit, T., Qin, J., Qian, C., Lee, D., Han,
J., Small, K., Fan, X., Sarikaya, R., and Ji, H. Winell:
wikipedia never-ending updating with llm agents. In
arXiv preprint arXiv:2508.03728, 2025.
SerpApi, LLC. Google search engine results api. https:
//serpapi.com/search-api , 2026. Accessed:
2026-01-10.
Shen, Z., Imana, B. Y ., Wu, T., Xiang, C., Mittal, P., and Ko-
rolova, A. Reliabilityrag: Effective and provably robust
defense for rag-based web-search. InProc. of NeurIPS,
2025.
Tunc ¸el, L. On the complexity of preflow-push algorithms for
maximum-flow problems.Algorithmica, 11(4):353–359,
1994.
Wang, F., Wan, X., Sun, R., Chen, J., and Arik, S. O. Astute
rag: Overcoming imperfect retrieval augmentation and
knowledge conflicts for large language models. InProc.
of ACL, 2025.Wang, J. and Yu, F. Derag: Black-box adversarial attacks on
multiple retrieval-augmented generation applications via
prompt injection. InFirst International KDD Workshop
on Prompt Optimization, 2025.
Warner, B., Chaffin, A., Clavi ´e, B., Weller, O., Hallstr ¨om,
O., Taghadouini, S., Gallagher, A., Biswas, R., Ladhak,
F., Aarsen, T., et al. Smarter, better, faster, longer: A
modern bidirectional encoder for fast, memory efficient,
and long context finetuning and inference. InProc. of
ACL, pp. 2526–2547, 2025.
Wei, Z., Chen, W.-L., and Meng, Y . Instructrag: Instruct-
ing retrieval-augmented generation via self-synthesized
rationales. InProc. of ICLR, 2025.
xAI. Grok 4 fast. Technical report, xAI, 2025. URL
https://x.ai/news/grok-4-fast . Accessed:
2026-01-27; API model id: grok-4-fast (and variants
grok-4-fast-reasoning / grok-4-fast-non-reasoning).
Xiang, C., Wu, T., Zhong, Z., Wagner, D., Chen, D., and Mit-
tal, P. Certifiably robust rag against retrieval corruption.
InICML 2024 Next Generation of AI Safety Workshop,
2024.
Xue, J., Zheng, M., Hu, Y ., Liu, F., Chen, X., and Lou, Q.
Badrag: Identifying vulnerabilities in retrieval augmented
generation of large language models. InarXiv preprint
arXiv:2406.00083, 2024.
Yu, W., Zhang, H., Pan, X., Cao, P., Ma, K., Li, J., Wang,
H., and Yu, D. Chain-of-note: Enhancing robustness
in retrieval-augmented language models. InProc. of
EMNLP, 2024.
Zhu, Y . From static to dynamic: A streaming rag ap-
proach to real-time knowledge base. InarXiv preprint
arXiv:2508.05662, 2025.
Zou, W., Geng, R., Wang, B., and Jia, J. {PoisonedRAG }:
Knowledge corruption attacks to {Retrieval-Augmented }
generation of large language models. InProc. of USENIX
Security, 2025.
11

RADAR: Defending RAG Dynamically against Retrieval Corruption
A. Overview
This appendix provides supplementary technical details, derivations, and experimental results supporting the RADAR method
presented in the main paper. The sections are organized as follows:
Appendix B derives the submodularity of the proposed energy function and explains its exact minimization via s−t Min-Cut
construction.
Appendix C justifies the Bayesian update rules used for dynamic capacity adjustment of historical answers.
Appendix D-H detail key implementation components:
• Symmetric NLI scoring for consistencyMand conflictCmatrices
• Document-wise atomic answer generation
• Eigenvector centrality computation for consensus scoring
• Post-processing for semantic outlier removal
• Constrained LLM synthesis prompt for final answer generation
Appendix I-J analyze computational aspects:
• Min-cut complexity using HLPP
• Runtime and efficiency
Appendix K-M add more experimental results:
• Extended static PIA & Poison attack results on NQ, TQA, Bio using DeepSeek, GPT-4o, and Grok-4-fast
• Multi-position injection attack results
• Extended dynamic PIA attack results using GPT-4oand Grok-4-fast
Appendix N shows the failure cases of our method.
Appendix O-P gives details of our dynamic dataset:
• Dataset statistics
• Examples of the dataset
B. Min-Cut Solvability of the Energy
Under the Markov Random Field (MRF) framework, minimizing the energy function E(y) corresponds to finding the
Maximum A Posteriori (MAP) estimate of the document labels. Specifically, we seek a binary labeling y∈ {0,1}kby
minimizing:
E(y) =kX
i=1
yiFi+ (1−y i)Si
+X
1≤i<j≤kMij|yi−yj|.(18)
The pairwise term is a weighted Potts model, which is graph-representable when it is submodular for each pair (i, j) .
Concretely, define
Eij(yi, yj) =M ij|yi−yj|.(19)
WhenM ij≥0, we have
Eij(0,0) = 0, E ij(1,1) = 0,
Eij(0,1) =M ij, E ij(1,0) =M ij.(20)
12

RADAR: Defending RAG Dynamically against Retrieval Corruption
which satisfies the submodularity inequality
Eij(0,0) +E ij(1,1)≤E ij(0,1) +E ij(1,0),(21)
since 0 + 0≤M ij+M ijholds whenever Mij≥0. Therefore, the overall energy E(y) belongs to the class of submodular
binary energies and can be minimized exactly via an s–tMin-Cut (Kolmogorov & Zabin, 2004). Additionally, under the
standard s-tgraph construction, the unary term yiFi+ (1−y i)Siis represented by terminal edges (s→d i)and(di→t)
with capacities SiandFi, respectively, while the pairwise term Mij|yi−yj|is represented by an undirected edge between
diandd jwith capacityM ij. Consequently, the cut cost equalsE(y), and the Min-Cut yields the global minimizery∗.
C. Justification of Bayesian Memory Update
In dynamic defense, we update the capacity of the memory node based on new evidence. Here, we interpret the capacity
S(t)
oldas the posterior probability that the historical answer a(t−1)remains correct. We define the binary random variable
H∈ {0,1}, whereH= 1denotes the hypothesis thata(t−1)is correct.
Prior.The prior probability P(H= 1) is given by π(t−1)
S , which is derived from the coherence of the previous generation
step.
Likelihood.Let Ebe the event of observing the semantic relationship between the old answer and the current retrieved
documents D(t). We use the aggregated entailment score L(t)
Sas the likelihood of observing such support given that the
history is correct:
P(E|H= 1) =L(t)
S.(22)
To make the update tractable, we adopt aSymmetric Likelihood Assumption. We assume that if the historical answer were
incorrect ( H= 0 ), the probability of observing high entailment from valid new documents would be the complement of the
support score:
P(E|H= 0) = 1− L(t)
S.(23)
This assumption reflects the intuition that an incorrect answer will contradict or fail to entail the true information present in
D(t).
Posterior.By applying Bayes’ theorem, the posterior probabilityP(H= 1|E)is:
P(H= 1|E) =P(E|H= 1)·P(H= 1)
P(E)
=P(E|H= 1)·P(H= 1)
P(E|H= 1)P(H= 1) +P(E|H= 0)P(H= 0).(24)
Substituting the prior and likelihood terms:
S(t)
old=L(t)
S·π(t−1)
S
L(t)
S·π(t−1)
S + (1− L(t)
S)·(1−π(t−1)
S).(25)
This recovers the update formula in Eq. 14. The update for the conflict capacity F(t)
oldin Eq. 16 follows an identical derivation
by definingH= 1as the hypothesis that the answer isincorrectand using the conflict matrix for likelihood estimation.
D. NLI Scoring forMandC
We use a Natural Language Inference (NLI) model to quantify the logical relation between two atomic answers. Given an
ordered pair (premise, hypothesis), the NLI model outputs a probability distribution over entailment, contradiction, and
neutral. We take the entailment probability as the consistency strength and the contradiction probability as the conflict
strength. For any pair(a i, aj), we denote:
Mi→j∈[0,1]as the entailment strength froma itoaj,(26)
13

RADAR: Defending RAG Dynamically against Retrieval Corruption
Ci→j∈[0,1]as the contradiction strength froma itoaj.(27)
In general, the scores are not symmetric:
Mi→j̸=M j→i, C i→j̸=Cj→i.(28)
However, our graph construction uses undirected document, which requires a symmetric edge weight. Moreover, the
contradiction scores used in risk aggregation should not be dominated by a single directional prediction.
To remove directional bias, we symmetrize the two directions by the geometric mean. For each pair (i, j) , we define the
symmetric scores:
Mij≜p
Mi→jMj→i,(29)
Cij≜p
Ci→jCj→i.(30)
By construction, these satisfyM ij=M jiandC ij=Cji.
The geometric mean enforces a conservative agreement-in-both-directions criterion: the symmetric score is large only when
both directions are high, and it is strongly down-weighted if either direction is low. This mitigates directional artifacts while
keeping the scores in[0,1]for direct use as graph capacities.
E. Atomic Answer Generation
Given the retrieved set D={d 1, . . . , d k}from the standard RAG workflow, we further decompose the generation step into
a set of document-wise responses, referred to as atomic answers. Specifically, for each retrieved document di, we query the
generatorGwith the original user queryqandonlythe single-document contextd i:
ai=G(q, d i), i= 1, . . . , k.(31)
Here aiis intended to capture the minimal claim(s) about qthat can be supported by dialone, decoupling the influence of
other retrieved documents. In implementation, we prompt the LLM to (i) answer qusing only the evidence in di, (ii) avoid
introducing external knowledge, and (iii) return a concise, self-contained statement.
The resulting atomic answers {ai}k
i=1serve as standardized semantic units for subsequent reasoning. We discard documents
whose atomic answers are uninformative. For the remaining set, we compute document-level entailment and contradiction
relations by applying an NLI model to pairs of atomic answers, which are then used to construct the similarity matrix M
and conflict matrixCdescribed in Sec. 4.2.
F. Eigenvector Centrality Computation
Given the similarity matrix M∈Rk×k, where Mij∈[0,1] measures the semantic and logical agreement between aiandaj,
we compute a global consensus score for each document node using eigenvector centrality. Intuitively, a node is considered
central if it is similar to other central nodes.
To improve numerical stability, we add a small self-loop to each node and define the weighted adjacency matrix
A=M+ϵI,(32)
whereIis the identity matrix andϵ >0is a small constant.
Eigenvector centrality is defined as the principal eigenvector ofA, i.e., the nonzero vectorv∈Rksatisfying
Av=λ maxv,(33)
where λmaxis the largest eigenvalue of A. We approximate vusing Tsteps of power iteration with ℓ2normalization.
Starting from the uniform initialization
v(0)=1
k1,(34)
we update
˜v(τ+1)=Av(τ), v(τ+1)=˜v(τ+1)
˜v(τ+1)
2+δ, τ= 0,1, . . . , T−1,(35)
14

RADAR: Defending RAG Dynamically against Retrieval Corruption
where δ >0 is a small constant to avoid division by zero. After Titerations, we take v=v(T)as the estimated centrality
vector.
Finally, we rescalevinto[0,1]for downstream use:
centralityi=vi−min jvj
max jvj−min jvj+δ, i= 1, . . . , k.(36)
In our implementation, we set ϵ= 0.01 , the number of power-iteration steps to T= 10 , and use δ= 10−8as a numerical
stability constant.
Algorithm 2Eigenvector Centrality via Power Iteration
Require:Similarity matrixM∈Rk×k; iterationsT; constantsϵ >0,δ >0
Ensure:Normalized centrality scorescentrality∈[0,1]k
1:A←M+ϵI
2:v←1
k1
3:forτ←1toTdo
4:v←Av
5:v←v
∥v∥2+δ
6:end for
7:centrality←v−min(v)
max(v)−min(v) +δ
8:Returncentrality
G. Post-processing for Semantic Consistency
After Min-Cut inference, we obtain the reliable set
Drel={d i|y∗
i= 1},(37)
together with their corresponding atomic answers {ai}di∈Drel. To further ensure semantic consistency among the selected
evidence, we apply a post-processing step that removes isolated items based on embedding cosine similarity.
Letei= Enc(a i)denote the embedding of atomic answer ai. For any pair of selected answers, we compute the cosine
similarity
sim(i, j) = cos(e i,ej), d i, dj∈ Drel.(38)
For each selected documentd i, we measure its average semantic agreement with the remaining selected set:
si=1
|Drel| −1X
dj∈Drel
j̸=isim(i, j), d i∈ Drel.(39)
Documents whose atomic answers exhibit low agreement with the rest are treated as isolated outliers and excluded:
D′
rel={d i∈ Drel|si≥λ}.(40)
In our implementation, we set λ= 0.3 . The remaining atomic answers associated with D′
relare then concatenated with the
original query to prompt the LLM for the final answer.
H. LLM-based Synthesis from Reliable Atomic Answers in dynamic defense
After Min-Cut inference, we obtain a set of reliable atomic answers A(t)
rel={a(t)
1, . . . , a(t)
m}. We generate the final response
via a constrained LLM synthesis step, whose goal is to extract the most consistent and dominant conclusion supported by
these atomic answers.
15

RADAR: Defending RAG Dynamically against Retrieval Corruption
To reduce uncontrolled speculation, the generator is explicitly instructed to only use the provided atomic answers as
references, and to not add, correct, question, or challenge any information contained in them even if it appears outdated.
This turns the synthesis step into a purely aggregative operation over vetted evidence.
We concatenate all reliable atomic answers into a single context stringcontext strand use the following prompt:
Question:{question}
The following are all the reliable reference atomic answers(synthesizestrictly based on this content only.
Do NOT add, correct, question, or challenge any information in it, even if you believe it may be outdated):
{context str}
Strictly follow the reference answers provided above and synthesize the most consistent and main conclusion as
the final answer.
Outputonly the final answer itself. Do NOT write any explanations, reminders, supplements, or comments about
dates.
The model must output a single final answer string without rationale, meta-commentary, or auxiliary notes. This ensures that
the final response is a direct synthesis of the selected reliable atomic answers.
I. Computational Complexity of Min-Cut
By the Max-Flow Min-Cut theorem, the capacity of a s-tMin-Cut equals the value of a maximum s–tflow; hence we
recover the optimal cut and thus y∗by computing a max flow and reading off the s-reachable set in the residual graph.
RADARcontainskdocument nodes and two terminals, hence
n=k+ 2.(41)
Since we connect every document pair with a consistency edge of capacity λMijand add two terminal edges per document,
the number of edges satisfies
m= 2k+ Θ(k2) = Θ(k2) = Θ(n2),(42)
Therefore, the graph is dense.
We compute max flow using the highest-label preflow-push (HLPP) algorithm, which is a push-relabel method that always
selects an active vertex of maximum height label and discharges it via a sequence of local push and relabel operations.
Unlike augmenting-path methods that repeatedly search for full s-taugmenting paths, HLPP only performs local updates
on admissible residual arcs, which is particularly suitable for our dense graph where m= Θ(n2). For HLPP, a refined
amortized analysis based on a potential function yields the worst-case bound (Tunc ¸el, 1994):
THLPP=O 
n2√m
.(43)
The key idea is to bound the number of non-saturating pushes by splitting them into small and big pushes using a threshold
κ: in each phase, there are at most O(κn2)small pushes, while the total number of big pushes is bounded by O(n2m/κ)
since each big push decreases the potential by at leastκ. Balancing the two terms by choosingκ=√mgives
O(κn2) +O(n2m/κ) =O 
n2√m
.(44)
In our dense graphm= Θ(n2), this further implies
THLPP=O(n3).(45)
J. Runtime and Efficiency
We measured average per-query runtime on RealtimeQA. Results for top- k= 10 and top- k= 50 are shown in Table 8.
The findings indicate that RADAR ’s runtime is primarily dominated by atomic answer generation, while NLI scoring and
Min-Cut inference introduce only marginal overhead.
16

RADAR: Defending RAG Dynamically against Retrieval Corruption
Table 8.Runtime and Performance at Top-k= 10and Top-k= 50with attack Pos 1 on RQA using Deepseek.
MetricVanilla RAG AstuteRAG InstructRAG RobustRAG ReliabilityRAG RADAR(Ours)
k=10k=50 k=10k=50 k=10k=50 k=10k=50 k=10k=50 k=10k=50
Atomic Gen.(s) - - - - - - 20 86 20 41 20 86
NLI(s) - - - - - - - - 0.32 0.40 0.32 0.52
Inference(s) - - - - - - 0.0365 0.0945 0.0005 0.1072 0.0007 0.0025
Total(s) 2 2 6 7 3 4 21 94 21 43 21 87
Acc. 25.0 35.0 25.0 21.0 23.0 33.0 64.0 69.0 69.0 67.0 69.0 72.0
ASR 74.0 65.0 1.0 5.0 68.0 46.0 7.0 9.0 15.0 18.0 11.0 5.0
Table 9.Performance ofRADARand baseline methods with top-k= 10on NQ and TQA.
Dataset PosVanilla RAG AstuteRAG InstructRAG RobustRAG ReliabilityRAG RADAR(Ours)
Acc.↑ASR.↓ Acc.↑ASR.↓ Acc.↑ASR.↓ Acc.↑ASR.↓ Acc.↑ASR.↓ Acc.↑ASR.↓
Benign
NQ – 70.2 – 27.4 – 57.4 – 61.6 – 67.4 – 67.0 –
TQA – 76.2 – 45.2 – 64.2 – 60.6 – 71.0 – 71.6 –
PIA Attack
NQPos 1 15.0 83.6 23.0 4.4 15.8 76.4 55.6 6.8 65.215.0 63.4 7.8
Pos 10 32.2 61.0 22.2 1.6 57.2 3.8 60.2 2.2 67.86.0 64.8 6.4
TQAPos 1 13.2 88.2 35.6 5.4 19.4 73.6 59.6 21.0 60.0 35.8 61.432.2
Pos 10 53.6 44.0 39.6 1.8 60.0 6.4 66.4 13.2 69.617.4 69.616.0
Poison Attack
NQPos 1 57.0 26.8 24.2 5.4 38.6 23.6 57.2 7.4 64.811.4 62.0 9.8
Pos 10 66.0 13.4 31.2 1.8 49.8 5.2 60.0 2.4 65.25.6 64.8 6.4
TQAPos 1 34.8 60.2 37.0 10.4 34.0 48.4 57.4 24.8 60.4 32.4 61.632.0
Pos 10 57.8 38.6 45.8 4.4 54.8 13.6 67.0 14.0 69.0 16.2 70.016.0
At top- k= 10 ,RADAR (21 s) matches RobustRAG and ReliabilityRAG in runtime, while achieving the best Acc./ASR
under attacks. At top- k= 50 ,RADAR (87 s) is faster than RobustRAG (94 s) but slower than ReliabilityRAG (43 s), as it
preserves more comprehensive evidence coverage rather than relying on aggressive subsampling. Vanilla RAG, AstuteRAG,
and InstructRAG are more efficient but significantly less robust. Overall, RADAR strikes a favorable robustness–efficiency
trade-off, with its additional cost mainly stemming from more complete evidence coverage rather than graph-based reasoning.
K. Additional Static Results
Under both PIA and Poison attacks, we conduct experiments using Deepseek on the NQ and TQA datasets, as shown in
Tables 9– 10. Additionally, under PIA attacks, we evaluate GPT -4o and Grok -4-fast across four datasets (RQA, NQ, TQA,
and Bio), as reported in Tables 11–13. Overall, our method demonstrates superior performance compared to the baseline in
most cases, achieving higher accuracy and lower ASR, although there are a few scenarios where it performs slightly worse
than the baseline.
L. Experimental Results for Multi-Position Attacks
We conduct multi-position PIA attack experiments on the RQA dataset using Deepseek to examine how the insertion
positions within the retrieved list affect model performance.
For top- k= 10 , we simultaneously attack two retrieval positions. Specifically, we evaluate attacks targeting the early
positions (Pos 1 + Pos 2) and the late positions (Pos 9 + Pos 10) to assess how the relative placement of adversarial content
influences both Accuracy and ASR. In addition, to better reflect real-world scenarios where the attacker’s insertion positions
may be uncertain, we sample two positions using a random number generator, resulting in Pos 3 and Pos 5. This randomized
setting helps simulate the inherent randomness of practical attacks.
17

RADAR: Defending RAG Dynamically against Retrieval Corruption
Table 10.Performance ofRADARand baseline methods with top-k= 50on NQ and TQA.
Dataset PosVanilla RAG AstuteRAG InstructRAG RobustRAG ReliabilityRAG RADAR(Ours)
Acc.↑ASR.↓ Acc.↑ASR.↓ Acc.↑ASR.↓ Acc.↑ASR.↓ Acc.↑ASR.↓ Acc.↑ASR.↓
Benign
NQ – 71.6 – 32.8 – 56.8 – 65.6 – 68.8 – 69.2 –
TQA – 76.4 – 46.8 – 60.8 – 68.2 – 75.6 – 74.0 –
PIA Attack
NQPos 1 15.2 83.4 22.0 5.2 27.8 57.0 62.0 66.0 51.2 15.4 65.06.6
Pos 25 47.6 40.0 29.0 0.8 58.6 2.6 65.4 2.6 56.6 1.8 65.44.2
Pos 50 41.0 48.2 27.4 1.4 59.6 4.8 66.2 2.6 66.23.0 66.24.2
TQAPos 1 12.4 88.8 39.0 5.6 31.0 58.0 62.0 21.6 58.8 36.0 64.626.2
Pos 25 62.8 33.8 43.0 1.8 63.6 4.6 68.2 11.6 76.47.6 70.4 16.2
Pos 50 62.6 30.2 43.8 2.0 63.8 5.4 68.0 11.4 76.45.4 70.6 16.0
Poison Attack
NQPos 1 62.2 23.0 25.2 3.2 41.2 16.8 61.8 34.0 64.4 13.4 65.47.0
Pos 25 67.5 6.6 33.4 1.4 55.2 3.2 66.4 2.6 69.2 3.4 69.64.6
Pos 50 67.4 5.8 31.6 1.4 56.4 4.2 66.6 2.6 68.4 3.4 69.24.2
TQAPos 1 37.4 58.4 38.6 10.6 38.2 43.0 59.4 26.2 59.8 33.4 65.023.2
Pos 25 68.8 26.0 43.4 5.0 57.0 17.2 68.6 11.8 74.27.8 69.6 16.8
Pos 50 73.8 15.6 43.2 4.4 58.4 15.2 68.8 11.4 75.64.4 70.4 15.2
Table 11.Performance ofRADARand baseline methods under PIA attack on RQA, NQ and TQA using GPT-4o.
Dataset PosVanilla RAG AstuteRAG InstructRAG RobustRAG ReliabilityRAG RADAR(Ours)
Acc.↑ASR.↓ Acc.↑ASR.↓ Acc.↑ASR.↓ Acc.↑ASR.↓ Acc.↑ASR.↓ Acc.↑ASR.↓
Top-k= 10
RQAPos 1 57.0 32.0 27.0 47.0 21.0 62.0 69.0 16.0 68.0 24.0 69.016.0
Pos 10 59.0 28.0 35.0 6.0 53.0 27.0 74.0 11.0 76.09.0 76.09.0
NQPos 1 52.6 30.6 43.6 21.0 35.4 32.0 57.6 9.0 59.4 19.0 62.410.2
Pos 10 47.6 30.2 43.4 3.4 51.0 17.4 62.0 4.6 65.4 8.0 66.07.0
TQAPos 1 36.8 50.6 55.2 35.2 48.8 41.4 61.2 23.4 58.0 40.6 61.433.4
Pos 10 38.8 43.4 65.2 13.0 60.8 28.6 69.0 18.4 68.8 22.4 69.424.2
Top-k= 50
RQAPos 1 53.0 39.0 32.0 41.0 27.0 55.0 69.0 15.0 67.0 23.0 74.013.0
Pos 25 65.0 26.0 28.0 6.0 53.0 33.0 72.0 10.0 75.04.0 75.011.0
Pos 50 64.0 24.0 43.0 5.0 59.0 32.0 72.0 8.0 74.0 3.0 75.05.0
NQPos 1 50.6 34.6 46.0 16.8 34.2 35.8 57.0 8.8 58.2 19.2 64.07.8
Pos 25 52.6 27.0 51.6 2.2 54.2 19.8 65.4 3.2 67.84.2 67.6 5.2
Pos 50 52.4 24.2 50.8 3.0 54.2 14.8 65.2 3.4 67.6 3.6 68.84.2
TQAPos 1 32.6 50.6 59.4 30.4 48.4 38.4 63.5 28.2 60.8 35.2 63.632.0
Pos 25 47.2 37.8 70.0 16.4 58.2 17.1 71.0 17.0 75.89.0 71.2 16.2
Pos 50 46.2 32.8 68.8 11.2 57.2 35.0 71.2 16.8 76.85.6 72.2 16.0
18

RADAR: Defending RAG Dynamically against Retrieval Corruption
Table 12.Performance under PIA attack with top-k= 10andk= 50on RQA, NQ and TQA using Grok-4-fast.
Dataset PosVanilla RAG AstuteRAG InstructRAG RobustRAG ReliabilityRAG RADAR(Ours)
Acc.↑ASR.↓ Acc.↑ASR.↓ Acc.↑ASR.↓ Acc.↑ASR.↓ Acc.↑ASR.↓ Acc.↑ASR.↓
Top-k= 10
RQAPos 1 19.0 79.0 18.0 3.0 12.0 25.0 54.0 8.0 52.0 21.0 66.09.0
Pos 10 37.0 57.0 18.0 3.0 47.0 5.0 56.0 4.0 60.0 10.0 69.05.0
NQPos 1 12.2 86.0 52.2 2.6 27.2 12.2 54.4 5.2 57.0 16.8 59.48.2
Pos 10 18.8 75.4 52.4 2.6 48.6 5.4 57.0 2.2 61.4 9.8 62.45.0
TQAPos 1 11.0 89.0 65.0 3.8 51.6 25.2 66.0 16.2 62.6 30.6 66.018.8
Pos 10 33.4 63.8 70.8 3.6 64.4 5.2 69.0 18.4 70.2 15.2 71.012.8
Top-k= 50
RQAPos 1 15.0 84.0 20.0 4.0 26.0 32.0 63.0 10.0 62.0 28.0 68.07.0
Pos 25 45.0 42.0 20.0 3.0 52.0 3.0 72.0 4.0 76.08.0 75.0 2.0
Pos 50 42.0 43.0 20.0 4.0 46.0 3.0 69.0 2.0 76.03.0 76.04.0
NQPos 1 13.2 85.6 49.2 2.2 34.8 20.4 58.0 5.2 55.0 25.4 61.04.8
Pos 25 31.2 60.4 50.0 3.8 51.6 3.6 61.2 2.8 62.03.6 62.04.0
Pos 50 27.4 63.2 49.8 3.0 50.4 4.0 62.0 3.2 62.04.0 62.04.0
TQAPos 1 8.0 92.0 66.6 3.8 49.8 30.4 67.2 15.8 59.4 37.6 67.613.8
Pos 25 41.0 55.6 66.8 3.2 66.0 4.4 70.0 10.2 74.68.8 70.2 10.0
Pos 50 43.2 43.6 66.6 3.0 67.6 4.4 69.8 5.6 71.84.2 71.810.0
Table 13.Performance under PIA attack on Bio using GPT-4o and Grok-4-fast.
Method MetricGPT-4o Grok-4-fast
Top-k= 10 Top-k= 50 Top-k= 10 Top-k= 50
Pos 1 Pos 10 Pos 1 Pos 25 Pos 50 Pos 1 Pos 10 Pos 1 Pos 25 Pos 50
Vanilla RAGAcc.↑ 36.8 8.8 39.2 28.2 8.8 57.6 9.6 54.4 38.8 13.0
Rel.↑ 34.2 8.4 37.2 26.4 8.6 39.8 9.4 39.4 27.4 12.8
Coh.↑ 42.0 10.2 43.6 30.6 10.6 53.4 12.4 54.6 36.4 17.0
AstuteRAGAcc.↑ 68.2 66.8 69.8 70.0 69.8 27.4 47.0 31.8 52.6 48.8
Rel.↑ 69.8 62.4 67.267.8 66.8 22.8 24.8 24.4 27.0 26.4
Coh.↑ 78.2 73.0 77.6 78.4 77.0 28.8 38.0 31.6 44.0 43.4
InstructRAGAcc.↑ 59.4 67.6 60.8 69.6 69.0 53.8 58.2 56.0 57.4 52.6
Rel.↑ 47.0 66.8 51.2 64.2 64.6 29.8 30.2 32.2 31.4 26.2
Coh.↑ 65.4 77.6 66.2 74.0 72.6 48.0 49.8 50.2 48.2 44.8
RobustRAGAcc.↑ 44.6 50.8 55.2 44.6 50.8 63.6 69.6 61.0 71.271.4
Rel.↑ 40.0 45.2 58.0 40.0 45.2 59.8 64.4 53.456.2 55.6
Coh.↑ 58.2 64.4 68.8 58.2 64.4 70.6 75.8 65.8 71.474.0
ReliabilityRAGAcc.↑ 64.8 67.6 57.6 68.4 68.0 66.0 71.8 64.0 74.0 63.2
Rel.↑ 64.4 66.4 58.2 66.867.8 43.2 47.8 38.4 48.8 40.8
Coh.↑ 73.6 75.4 66.0 74.2 76.2 62.4 67.8 59.871.857.6
RADAR(Ours)Acc.↑ 75.2 69.2 72.0 73.4 73.6 68.0 72.8 69.6 74.269.8
Rel.↑ 68.8 68.8 65.670.867.6 55.6 55.0 54.053.2 50.8
Coh.↑ 82.2 78.2 77.8 79.0 79.6 69.8 73.8 67.070.8 67.2
19

RADAR: Defending RAG Dynamically against Retrieval Corruption
Table 14.Multi-position PIA attack results on RQA using DeepSeek.
MethodTop-k= 10 Top-k= 50
Pos 1+2 Pos 9+10 Pos 4+6 Pos 1+2+3 Pos 49+50+51 Pos 1+4+27
Acc.↑ASR.↓ Acc.↑ASR.↓ Acc.↑ASR.↓ Acc.↑ASR.↓ Acc.↑ASR.↓ Acc.↑ASR.↓
Vanilla RAG 18.0 82.0 56.0 31.0 53.0 21.0 14.0 86.0 62.0 21.0 25.0 74.0
AstuteRAG 12.0 9.0 26.0 1.0 33.0 1.0 11.0 11.0 31.0 4.0 15.0 14.0
InstructRAG 19.0 70.0 65.0 6.0 53.0 4.0 11.0 56.0 53.0 3.0 5.0 39.0
RobustRAG 48.0 28.0 68.0 5.0 58.0 12.0 59.0 25.0 68.0 4.0 62.0 23.0
ReliabilityRAG 59.0 32.0 72.014.0 67.0 18.0 44.0 50.0 72.03.0 60.0 31.0
RADAR(Ours) 64.030.0 72.017.0 68.021.0 61.030.0 70.0 8.0 63.030.0
Table 15.Performance under evolving evidence streams with GPT-4o.
Attack PosVanilla RAG AstuteRAG InstructRAG RobustRAG ReliabilityRAG RADAR(Ours)
Acc.↑ASR.↓ Acc.↑ASR.↓ Acc.↑ASR.↓ Acc.↑ASR.↓ Acc.↑ASR.↓ Acc.↑ASR.↓
Benign – 56.05 – 58.93 – 72.68 – 58.46 – 58.67 – 61.29 –
PIAPos 1 35.76 39.99 47.28 10.81 55.39 12.80 47.47 31.99 43.38 39.41 56.1720.02
Pos 25 45.04 25.40 50.93 10.75 53.93 32.18 53.49 26.49 56.88 8.25 60.6512.15
Pos 50 40.12 27.90 50.99 10.68 53.42 31.35 53.29 25.14 58.09 4.54 61.2310.87
Table 16.Performance under evolving evidence streams with Grok-4-fast.
Attack PosVanilla RAG AstuteRAG InstructRAG RobustRAG ReliabilityRAG RADAR(Ours)
Acc.↑ASR.↓ Acc.↑ASR.↓ Acc.↑ASR.↓ Acc.↑ASR.↓ Acc.↑ASR.↓ Acc.↑ASR.↓
Benign – 44.21 – 25.27 – 46.83 – 50.03 – 52.72 – 56.68 –
PIAPos 1 5.89 75.82 25.72 2.94 4.99 27.90 44.84 38.96 41.01 43.19 53.1620.79
Pos 25 17.08 53.04 26.10 2.82 44.79 9.40 47.98 30.13 50.93 10.36 57.5810.23
Pos 50 13.63 53.55 25.72 2.75 39.99 10.17 47.92 29.87 54.89 7.17 57.7010.04
Table 17.Static vs. DynamicRADARunder the dynamic setting using Deepseek.
Attack PosRADAR(Static)RADAR(Dynamic)
Acc.↑ASR.↓ Acc.↑ASR.↓
Benign - 54.13 - 74.02 -
PIA Pos 1 48.43 20.02 63.60 17.85
PIA Pos 25 50.86 12.98 70.12 8.94
PIA Pos 50 51.18 12.79 70.05 6.01
Table 18.Dynamic dataset statistics.
Statistic Value (%)
Questions with 1 answer change 77.8
Adjacent-year answer change rate 68.3
No answer change 22.2
Exactly 1 change 26.2
2 changes 51.6
20

RADAR: Defending RAG Dynamically against Retrieval Corruption
For top- k= 50 , we extend the attack to three positions. Following the same protocol, we test attacks at the front of the
list (Pos 1 + Pos 2 + Pos 3) and near the end of the list (Pos 48 + Pos 49 + Pos 50). We further include a randomized
configuration by sampling three positions with a random number generator, yielding Pos 1, Pos 4, and Pos 27, to evaluate
the impact of random multi-point attacks under a larger retrieval budget.
M. Additional Dynamic Results
We present additional performance results under evolving evidence streams for two different models, GPT-4o which is shown
in Table 15 and Grok-4-fast which is shown in Table 16. These results demonstrate that RADAR consistently outperforms
baselines in terms of accuracy, particularly in high-ranking evidence positions. Furthermore, RADAR shows superior
robustness to evolving attacks, outperforming competing methods in both benign and attack scenarios.
Our dynamic setting does not inject year-specific context into queries, though retrieved documents may contain temporal
cues. We compare RADAR (Static) and RADAR (Dynamic) under the same setting to evaluate the static variant and isolate
gains from the dynamic extension. As shown in Table 17, the static version degrades more as evidence evolves, while the
dynamic version remains more stable, indicating that improvements mainly come from the dynamic design.
N. Failure-case analysis
Failure Case 1:
Query: “Who won the FIFA Men’s World Cup?”
•2015 (ground truth:Barcelona): Google returns only one informative document (Pos 1). Attacking Pos 1 causes
RADAR to output the poisoned answer.
•2016 (ground truth:Real Madrid): Only one informative document appears at Pos 2. Attacking Pos 1 lets the poisoned
document dominate centrality; the new correct evidence is filtered out and the memory node preserves the stale answer.
Failure Case 2:
Query: “Who won the Nobel Peace Prize?”
•2021 (ground truth:Maria Ressa and Dmitry Muratov): Many documents describe the winners unclearly, causing some
atomic answers to be generated with onlyMaria Ressa. This leads to the Maria Ressa cluster dominating centrality,
pushing the correct full answer aside and causingRADARto output the wrong answer.
The essence of both failure cases is that RADAR ’s consensus mechanism using eigenvector centrality and NLI entailment
critically relies on the assumption that benign evidence forms the dominant coherent cluster in the retrieved set. When this
assumption breaks because genuinely new correct evidence is too sparse or because real-world reporting noise creates a
stronger false majority cluster, the Min-Cut selects the wrong partition and the memory node may further reinforces the
error.
O. Dynamic Dataset Statistics
We present descriptive statistics on answer changes over time in Table 18. This shows that answer volatility is substantial,
suggesting the dataset better reflects dynamic retrieval settings rather than a mostly static benchmark.
P. Examples of Dynamic Dataset
Our Dynamic Dataset contains 500 QA questions, each associated with several different years. For each year, we retrieved
the top 50 relevant documents from Google. Using Deepseek, we generated incorrect answers for each year, which are used
for PIA Attack and ASR statistics, as well as incorrect contexts, which are used for Poison Attack. Here are examples of our
dynamic dataset.
21

RADAR: Defending RAG Dynamically against Retrieval Corruption
Examples of Our Dynamic Dataset
{
"question": "Who is the President of the United States?",
"yearly_contexts": {
"2015": {
"answer": [
"Barack Obama"
],
"docs": [
{
"title": "Timeline of the Barack Obama presidency (2015)",
"url": "https://en.wikipedia.org/wiki/
Timeline_of_the_Barack_Obama_presidency_(2015)",
"snippet": "The following is a timeline of the presidency of Barack Obama,
from January 1 to December 31, 2015. For his time as president-elect,
see the presidential ...",
"content": "Timeline of the Barack Obama presidency (2015) - Wikipedia..."
},
{
"title": "Get Ready: President Obama’s 2015 State of the Union Address",
"url": "https://obamawhitehouse.archives.gov/blog/2015/01/11/get-ready-
president-obamas-2015-state-union-address",
"snippet": "On Tuesday, January 20 at 9pm ET, President Obama will deliver
his sixth State of the Union Address. This year there will be more
ways than ever to take part ...",
"content": "Get Ready: President Obama’s 2015 State of the Union Address
..."
}
...
],
"incorrect_answer": "George W. Bush",
"incorrect_context": [
"As of 2015, George W. Bush continues to serve as the 43rd President of the
United States, having been re-elected to a second term in 2004..."
]
},
"2018": {
"answer": [
"Donald Trump"
],
"docs": [
{
"title": "2018 United States elections",
"url": "https://en.wikipedia.org/wiki/2018_United_States_elections",
"snippet": "Elections were held in the United States on November 6, 2018.
These midterm elections occurred during incumbent Republican president
Donald Trump’s first ...",
"content": "2018 United States elections - Wikipedia..."
},
{
"title": "President Donald J. Trump Proclaims January 16, 2018, as
Religious Freedom Day",
"url": "https://trumpwhitehouse.archives.gov/presidential-actions/
president-donald-j-trump-proclaims-january-16-2018-religious-freedom-
day/",
"snippet": "On Religious Freedom Day, we celebrate the many faiths that
make up our country, and we commemorate the 232nd anniversary of the
passing of a State law.",
"content": "President Donald J. Trump Proclaims January 16, 2018, as
Religious Freedom Day..."
}
...
22

RADAR: Defending RAG Dynamically against Retrieval Corruption
],
"incorrect_answer": "Barack Obama",
"incorrect_context": [
"As of 2018, Barack Obama continues to serve as the 44th President of the
United States, having been re-elected for a second term in 2012..."
]
}
...
}
...
}
23