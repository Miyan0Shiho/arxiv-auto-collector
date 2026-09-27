# LEGO: Synergizing Expert GraphRAG and Expert Chain-of-Thought for Legal Reasoning

**Authors**: Qingjing Chen, Junkai Zhang, Shaochun Wang, Jiahao Ding, Siyuan Zheng, Yukun Yan, Zhi Zheng, Antonino Rotolo, Yun Liu, Weixing Shen

**Published**: 2026-09-22 19:48:52

**PDF URL**: [https://arxiv.org/pdf/2609.27009v1](https://arxiv.org/pdf/2609.27009v1)

## Abstract
Large language models are increasingly applied to high-risk domains such as law, yet complex legal reasoning remains limited by two structural challenges. First, existing RAG and GraphRAG methods emphasize lexical or semantic similarity while overlooking normative relations among legal provisions. Second, vanilla Chain-of-Thought prompting may generate plausible rationales without enforcing the normative structure of legal reasoning. To deal with the bottleneck of pipelines in the legal reasoning domain, we propose LEGO, a dual-module framework that synergizes Legal Expert GraphRAG and expert Chain-of-thought for complex legal reasoning. ExpertGraphRAG uses an expert-annotated civil code graph encoding these normative relations with a greedy normative-coverage retrieval algorithm to dynamically extract instance-specific provision subgraphs, while ExpertCoT organizes the retrieved provisions and case facts into structured Provision-Fact-Conclusion reasoning. With a Qwen3-8B backbone, LEGO achieves 40.53% exact-match accuracy on LawExamQA_Civil, outperforming the evaluated RAG and CoT baselines and performing comparably to the evaluated larger models, while remaining robust on multi-hop questions. It also achieves the best results among the evaluated baselines on the open-ended benchmarks. Ablation studies confirm the individual and complementary contributions of both modules, demonstrating LEGO's effectiveness in improving LLMs' complex legal reasoning ability. Code and dataset can be found in the link: https://github.com/BLK-WHT/LEGO

## Full Text


<!-- PDF content starts -->

LEGO: Synergizing Expert GraphRAG and Expert Chain-of-Thought for
Legal Reasoning
Qingjing Chen1*, Junkai Zhang2*, Shaochun Wang5†, Jiahao Ding3, Siyuan Zheng4,
Yukun Yan2,Zhi Zheng5,Antonino Rotolo1†,Yun Liu2,Weixing Shen2
1Alma AI, University of Bologna, Italy
2Tsinghua University, China
3Xiamen University, China
4Shanghai Jiao Tong University, China
5Modelbest Inc.
{qingjing.chen2,antonino.rotolo}@unibo.it
jj-zhang25@mails.tsinghua.edu.cn
wangshaochun@modelbest.cn
Abstract
Large language models are increasingly ap-
plied to high-risk domains such as law, yet
complex legal reasoning remains limited by
two structural challenges. First, existing RAG
and GraphRAG methods emphasize lexical or
semantic similarity while overlooking norma-
tive relations among legal provisions. Sec-
ond, vanilla Chain-of-Thought prompting may
generate plausible rationales without enforc-
ing the normative structure of legal reason-
ing. To deal with the bottleneck of pipelines
in the legal reasoning domain, we propose
LEGO, a dual-module framework that syn-
ergizesLegalExpertGraphRAG and ex-
pert Chain-of-thought for complex legal rea-
soning. ExpertGraphRAG uses an expert-
annotated Civil Code graph encoding these
normative relations with a greedy normative-
coverage retrieval algorithm to dynamically
extract instance-specific provision subgraphs,
while ExpertCoT organizes the retrieved pro-
visions and case facts into structured Pro-
vision–Fact–Conclusion reasoning. With a
Qwen3-8B backbone, LEGO achieves 40.53%
exact-match accuracy on LawExamQA_Civil,
outperforming the evaluated RAG and CoT
baselines and performing comparably to the
evaluated larger models, while remaining ro-
bust on multi-hop questions. It also achieves
the best results among the evaluated base-
lines on the open-ended benchmarks. Abla-
tion studies confirm the individual and com-
plementary contributions of both modules,
demonstrating LEGO’s effectiveness in improv-
ing LLMs’ complex legal reasoning ability.
Code and dataset can be found in the link:
https://github.com/BLK-WHT/LEGO
*These authors contributed equally to this work.
†Corresponding Author.1 Introduction
The application of LLMs to high-risk professional
domains has accelerated rapidly, including law,
where dedicated benchmarks now measure legal
language understanding and reasoning (Chalkidis
et al., 2022; Guha et al., 2023; Li et al., 2025b;
Shi et al., 2026). Yet reliable legal reasoning re-
mains challenging: where LLM reasoning in legal
domain is not robust and easily misled by typos
or irrelevant cues ( (Hu et al., 2025), models that
perform well on statutes seen during training falter
on unseen ones (Blair-Stanek et al., 2023), they can
reach the right answer without producing a sound
reasoning path (Kang et al., 2023), and they strug-
gle to recover the provisions a question actually
depends on once several are involved (Lee et al.,
2025). These problems involve two main pipelines:
Retrieval-Augmented Generation (RAG) grounds
generation in external evidence (Lewis et al., 2020),
while Chain-of-Thought (CoT) prompting has the
model follow intermediate reasoning steps (Wei
et al., 2022). although their strengths and weak-
nesses have begun to be examined jointly in the
general domain (Li et al., 2025a) and in medicine
(Ge et al., 2026), legal NLP applications have not
yet combined them in a way that addresses the lim-
itations of each, Both are largely evaluated and
improved in isolation. (Liu et al., 2025; Chen et al.,
2026b),
The first bottleneck is retrieval. Standard RAG
and many GraphRAG systems (Edge et al., 2024;
Zhang et al., 2025; Zhuang et al., 2026; Chen et al.,
2026a; Gutiérrez et al., 2025; Guo et al., 2025)
still rely heavily on lexical or embedding similar-
ity, even with induced graph neighborhoods. Yet
arXiv:2609.27009v1  [cs.CL]  22 Sep 2026

Figure 1: LEGO’s architecture. ExpertGraphRAG uses the fact-side instance representation, implemented as a
natural-language query containing the case facts, question, and options to retrieve an instance-specific provision
subgraph from the static Expert Provision Graph. ExpertCoT then composes the retrieved provisions and case
facts into a structured P–F–C analysis. “Fact Graph” and “Conclusion Graph” in the figure denote conceptual
representations rather than separately materialized graph objects.
legal knowledge is semantically sparse: textually
distant provisions may be tightly linked through
prerequisites, exceptions, general–special relations,
or legal effects, while textually similar ones may
distract because they belong to different constituent
elements (Blair-Stanek et al., 2023; Lee et al.,
2025). The complex case of Figure 1 shows this
issue: repeated transport-related terms may lead
similarity-based models to select the transport com-
pany (Option D). Yet the Fact Graph shows that B
has no contract with the transport company. Cor-
respondingly, the Provision Graph must capture
the general–specific relation between contractual
privity under Art. 465 and carrier liability under
Art. 832, as well as Art. 593’s exclusion of third-
party-caused breach as a basis for bypassing con-
tractual privity. Lexical similarity alone captures
neither these factual relations nor the normative
links among provisions.
The second bottleneck is reasoning. Vanilla CoT
can produce plausible intermediate text, but it does
not guarantee that the model follows the normative
order required by a professional domain (Wei et al.,
2022; Yao et al., 2023). Chain-of-Thought (CoT) is
extremely frag- ile and lacks robustness in high-risk
domains such as law (Yu et al., 2025; Kang et al.,
2023). The reasoning trace can drift by skipping
an exception, treating an unfulfilled prerequisite assatisfied, or affirming the consequent from a legal
effect back to its condition (Servantez et al., 2024).
The risk also noted in faithfulness and robustness
studies of reasoning in RAG, helps only if the re-
trieved sources are authoritative and structurally
relevant; otherwise, each generated step may am-
plify noise from the previous step, (Zhou et al.,
2026). Research highlights the need to constrain
and regulate the models’ CoT using knowledge in-
jection or expert-guided legal structures (Liu et al.,
2025).
To address these bottlenecks, we propose LEGO,
a dual-module framework that injects expert le-
gal knowledge into both stages of the pipeline:
an expert-annotated knowledge structure into re-
trieval (ExpertGraphRAG), and the expert mode of
legal reasoning into generation (ExpertCoT). Ex-
pertGraphRAG retrieves over an expert-annotated
Civil Code graph, in which doctrinal rule cards writ-
ten by legal experts encode the normative relations
among provisions. Rather than ranking articles
by semantic similarity alone, its Normative Cover-
age Greedy (NCG) component grows a provision
set by marginal legal utility, so that each article it
adds covers a legal issue the case raises but the cur-
rent set does not. This yields a compact, instance-
specific provision subgraph G(i)
Pinstead of a list
of lexically similar articles. ExpertCoT then orga-

nizes the retrieved provisions and case facts into
structured Provision–Fact–Conclusion reasoning,
with the controlling rules as the major premise, the
case facts as the minor premise, and the per-option
judgments as the conclusion. With a Qwen3-8B
backbone, LEGO reaches 40.53% exact-match ac-
curacy on LawExamQA_Civil, outperforming the
evaluated RAG and CoT baselines and performing
comparably to far larger models such as GPT-5 and
DeepSeek-V3 671B, while remaining robust on
multi-hop complex reasoning involving multiple
provisions. It also achieves the best results among
the evaluated baselines on two open-ended bench-
marks, and ablations confirm the individual and
complementary contributions of the two modules.
The structured syllogistic output further makes the
rationale linking provisions, facts, and conclusions
easier to inspect and audit.
2 Related Work
2.1 GraphRAG and Legal-Domain Retrieval
Flat-vector RAG can struggle when evidence is
distributed across passages because similarity rank-
ing alone does not guarantee retrieval of complete
reasoning chains (Han et al., 2026), a challenge
isolated by multi-hop QA benchmarks such as Hot-
potQA, MuSiQue, and 2WikiMultiHopQA (Yang
et al., 2018; Trivedi et al., 2022; Ho et al., 2020).
GraphRAG instead structures corpus information
as graphs and performs graph-aware retrieval or
aggregation (Edge et al., 2024; Zhang et al., 2025).
Recent work explores efficient or ontology-guided
graph construction (Zhuang et al., 2026; Wang
et al., 2026), inference-time reasoning structures
without pre-built graphs (Chen et al., 2026a),
graph-free triplet retrieval (Gong et al., 2026),
dependency-aware reranking (Li et al., 2026b),
memory-based HippoRAG 2 (Gutiérrez et al.,
2025), Tree Retrieval RAPTOR (Sarthi et al., 2024),
subgraph-retrieval-based G-Retriever (He et al.,
2024), and LightRAG incorporating graph struc-
tures (Guo et al., 2025). LegalGraphRAG further
organizes legal knowledge into Fact, Ontology, and
Rule subgraphs (Chen et al., 2026b). However,
neither these general-purpose baselines nor Legal-
GraphRAG evaluates retrieval on complex or mul-
tihop legal reasoning tasks involving normative
relations among provisions. However, retrieving
the right provisions does not ensure reasoning to
the correct final answer, as legal conclusions often
depend on reasoning over the relational interac-tions among multiple provisions and performing
deductive inference from facts to the applicable
provisions.
2.2 Chain-of-Thought and Legal Reasoning
Chain-of-Thought (Wei et al., 2022) and Tree-of-
Thought (Yao et al., 2023) prompting externalise
intermediate steps but offer no guarantee of nor-
mative correctness and reliability in specific do-
mains. Verifier-driven reasoning, popularised by
HuatuoGPT-o1 (Chen et al., 2024) for medicine, au-
dits each step with a True/False check and triggers
backtracking, path-exploration, or self-correction
on failure. Legal prompting studies have directly
evaluated IRAC-based reasoning on the COLIEE
entailment task (Yu et al., 2023) and on expert-
annotated legal scenarios (Kang et al., 2023). More
recently, MSLR introduced expert-derived IRAC
traces for evaluating multi-step legal reasoning (Yu
et al., 2025). IRAC-style structured legal reasoning
has also been explored on top of ChatLaw (Cui
et al., 2026), Lawformer (Xiao et al., 2021), and
LexGLUE (Chalkidis et al., 2022), However, these
studies lack grounding in reliable retrieved evi-
dence, leaving CoT vulnerable to instability and
hallucination. Research of CoT in legal domain re-
main unexplored on reasoning over inter-provision
relations and incorporating expert legal reasoning
patterns, such as legal syllogisms.
Recent RAG–CoT systems interleave retrieval
with reasoning (Trivedi et al., 2023), revise thought
steps with retrieved evidence (Wang et al., 2024),
or use knowledge graphs to constrain CoT genera-
tion (Li et al., 2025a); expert-guided medical QA
further shows that domain attributes can constrain
both retrieval and CoT in high-risk settings (Ge
et al., 2026). However, legal reasoning is uniquely
challenging because the system must first retrieve
the correct set of relevant provisions, then accu-
rately identify the normative links and reasoning re-
lations among them, and finally apply expert -level
legal reasoning to derive the conclusion.
3 Methodology
We instantiate LEGO as a pipeline following le-
gal syllogistic reasoning (Patzig, 2013), which de-
composes legal inference into the major premise
(normative Provision), minor premise (Fact), and
conclusion (Figure 1). For each query qi, com-
prising the case facts and question, with answer
options Oi, LEGO forms a fact-side query G(i)
F.

ExpertGraphRAG uses this query to retrieve a set
of relevant provisions Pifrom the global Provision
Graph GP. ExpertCoT then combines the original
query, answer options, and retrieved provision texts
to generate a structured P–F–C analysis G(i)
Cand
the final answer ˆAi:
G(i)
F:= Serialize(q i, Oi),
Pi:=f RAG
G(i)
F, GP
,

ˆAi, G(i)
C
=fCoT(qi, Oi, Pi).(1)
Here, G(i)
Fis implemented as the natural-
language retrieval query, Pidenotes the retrieved
provision texts, and G(i)
Cdenotes the generated P–
F–C response rather than a separately materialized
graph. LEGO therefore combines graph-structured
provision retrieval with structured syllogistic gen-
eration. Unlike prior expert-conditioned QA that
uses flat attributes such as subject-area tags (Ge
et al., 2026), LEGO represents expert knowledge
as a relational legal subgraph and uses CoT as a
constrained syllogistic reasoning process over that
graph.
3.1 ExpertGraphRAG
LEGO retrieves Civil-Code articles from an of-
flineProvision Graph, rather than from free-form
chunks. Let Rbe expert-defined legal rule units
andAbe Civil-Code articles:
GP= (V P, EP;B),
VP=R ∪ A,
EP=ERA∪E norm.(2)
HereERAis the bipartite rule–article layer, Enorm
stores expert provision–provision relations, and
Br,ais the support weight from article ato rule
unitr. For input xi, let di(a) = sim(x i, a),
gi(a) = max r∈Rsim(x i, r)B r,a, and Γ(a) ={r:
Br,a>0}.
Normative Coverage Greedy.We useNorma-
tive Coverage Greedy(NCG) to select articles.
Given a partial set S, NCG scores a candidate arti-
cleaby:
∆i(a|S) =d i(a) +λ aligngi(a)
+λcovni(a, S)−λ redoi(a, S),(3)
where Γ(S) =S
a∈SΓ(a) ,ni(a, S) =|Γ(a)\
Γ(S)| is newly covered legal-rule evidence, andoi(a, S) =|Γ(a)∩Γ(S)| is already-covered evi-
dence. The greedy update is:
at= arg max
a∈A\S t−1∆i(a|S t−1),
St=St−1∪ {a t}, S∗
i=SK.(4)
The first three terms are the marginal gain of a
relevance-weighted coverage objective over expert
rule units. This is the same monotone submodular
family underlying maximum coverage and facility-
location retrieval, for which greedy selection is the
standard approximation strategy (Nemhauser et al.,
1978; Lin and Bilmes, 2011). The final term is
a lightweight redundancy regularizer: it discour-
ages repeatedly selecting provisions that explain
the same legal issue, but we do not use it to claim
a new approximation bound. In effect, NCG op-
timizes themarginal legal utilityof each provi-
sion: it keeps provisions on topic, expands nor-
mative coverage, and yields a compact provision
subgraph that serves as the legal major-premise con-
text for downstream Syllogistic CoT. The weights
λalign, λcov, λredand the article budget Kused in
our experiments are listed in Appendix B.3.
The selected set induces the instance-level pro-
vision subgraph:
G(i)
P= (V(i)
P, E(i)
P),
V(i)
P=S∗
i∪ R∗
i,
E(i)
P=E P∩ 
V(i)
P×V(i)
P
.(5)
whereR∗
i={r∈ R: max a∈S∗
iBr,a>0}.
3.2 ExpertCoT
Letxidenote the input instance, comprising the
case description, question, and answer options. We
conceptually represent its factual structure as
G(i)
F=
V(i)
F, E(i)
F
,(6)
where V(i)
Fcomprises the legally relevant entities,
events, and factual propositions of the case, and
E(i)
Frepresents the relations among them. This
graph provides a conceptual description of the case
rather than a separately constructed input to the
model.
ExpertGraphRAG uses the serialized case, ques-
tion, and options as the retrieval query. ExpertCoT
then takes the original instance and the retrieved
provision texts as input and generates the P–F–C
analysis and final answer in a single pass. For

each option t, the applicable legal rules provide
the major premise and the corresponding case facts
provide the minor premise; the sub-judgment ˆctis
derived by applying those rules to the facts through
syllogistic reasoning. An option-level inference
may involve multiple provisions.
To describe the organization of the resulting ra-
tionale, we combine the provision nodes V(i)
Pand
edge set E(i)
Pof the retrieved Provision Graph in
Eq. 2 with the case structure inG(i)
F:
G(i)
C=
V(i)
F∪V(i)
P∪C(i), E(i)
F∪E(i)
P∪E(i)
C
,
(7)
where C(i)={ˆc 1, . . . ,ˆc T}contains the judgments
for the Tanswer options, and E(i)
Crepresents the
links between each judgment and the provisions
and facts invoked to support it. This graph is a con-
ceptual representation of the generated rationale;
the implementation does not explicitly construct or
store it.
To reduce reasoning errors and unsupported as-
sertions, ExpertCoT combines a syllogistic P–F–C
chain of thought with eleven explicit prompt in-
structions. The analysis first identifies the legal
issue and what the question asks. For the major
premise P, the model identifies controlling rules
only from the retrieved provisions and explains
how rule priority, provisos, exceptions, limiting
conditions, and remedies govern their application.
It must distinguish legal effects, such as contract
validity, real-rights transfer, opposability, and li-
ability, and avoid misreading limiting clauses as
grounds for invalidity. These instructions encour-
age reasoning grounded in the legal relations rep-
resented in the Civil Law ExpertGraph. For the
minor premise F, the model identifies legally rele-
vant facts without drawing legal conclusions. For
the conclusion C, it reviews the applicable pro-
vision relations and derives a judgment for each
option by applying the rules to the corresponding
facts. It checks the option’s subject, legal predicate,
required elements, and asserted legal effect, while
avoiding over-selection and respecting whether the
question asks for correct or incorrect statements.
The instructions were iteratively refined through
expert-in-the-loop error analysis. The full prompt
and an end-to-end example are reproduced in Ap-
pendix A.4 Experiments
4.1 Setup
Datasets.We evaluate LEGO on three open
datasets in the civil -law domain, ensuring consis-
tency with the expert -constructed civil -law knowl-
edge graph used in our pipeline. (1)LawEx-
amQA_Civil, a subset of the Chinese questions
from the National Judicial Examination of China
(NJEC), where we categorize questions by the
number of provisions referenced in the official
rationale. This hop count serves as a proxy for
the complexity of multi -hop legal reasoning. (2)
LexRAG_Civil(Li et al., 2025b), a civil -law sub-
set ofLexRAGused to assess multi -turn legal con-
sultation performance.LexRAG_Civilevaluates
open-ended, multi-turn legal consultation. Mod-
els generate free-form responses to successive user
queries rather than selecting from predefined an-
swer options. (3)PLawBench_Civil(Shi et al.,
2026), a rubric -based benchmark for evaluating
LLMs in real -world legal practice. We use the
Practical Case Analysissection, which consists
of complex civil -law cases annotated with reason-
ing rubrics, enabling assessment of legal reason-
ing ability in realistic scenarios. Our evaluation
is not limited to multiple-choice question answer-
ing. Only LawExamQA_Civil uses a multiple-
choice format; LexRAG_Civil evaluates free-form
responses in multi-turn legal consultations, and
PLawBench_Civil evaluates generated practical le-
gal analyses using task-specific rubrics. All of them
are evaluated on complex legal -reasoning tasks in-
volving multiple provisions and long case texts.
The data details can be found in Appendix D.
Civil law ExpertGraph.Our expert-annotated
Civil Law graph encodes the PRC Civil Code as
318 concepts over 409 rule cards, linked to 1,211
of its 1,260 articles by 4,152 support edges. Each
card states one doctrinal rule—its conditions, ef-
fects and exceptions—embedded whole, so queries
route on rules rather than article text. Built over
7 months by trained legal experts, it serves solely
to enhance RAG signal over the civil-code corpus
(Appendix E).
Language model Baseline(1) Closed -source
and large general LMs. We choose
closed -source models GPT -5 (OpenAI, 2025) and
DeepSeek -V3 671B (DeepSeek-AI, 2024) as large
general -purpose LMs, which have strong zero -shot
and long -context abilities, to test the performance

ceiling achievable through purely parametric
reasoning without external retrieval. (2) Open
Larger LMs. We choose Qwen3-30B-A3B (Qwen
Team, 2025) and GLM-4.7-Flash (zai-org, 2026)
as Open Larger LMs, which offer a strong balance
between computational cost and performance
at the 30B scale. (3) Small Open-source LMs.
We use Qwen3 -8B (Qwen Team, 2025) and
GLM -4-9B-Chat (zai-org, 2024) as base models
whose 8B–9B parameter scale matches our RAG
generators for direct comparison.(4) We select
DISC -LawLLM (7B) (Yue et al., 2024) and
LegalOne (8B) (Li et al., 2026a), two legal domain
finetuned open -source models tailored for legal
reasoning.
RAG baselineWe compare LEGO with sev-
eral representative RAG architectures: flat top-
k dense retrieval with Qwen3-Embedding-8B;
auto-induced semantic graph (LightRAG (Guo
et al., 2025)); hierarchical summary tree (RAP-
TOR (Sarthi et al., 2024)); Steiner-tree graph re-
trieval (G-Retriever (He et al., 2024)); long-range
memory / associative graph (HippoRAG2 (Gutiér-
rez et al., 2025)). All RAG baselines use the same
Qwen3-8B backbone and the same Civil-Code re-
trieval corpus as LEGO, so accuracy differences
are attributable to retrieval / reasoning design rather
than to scale or external knowledge.
The experiments use the method-specific con-
figurations documented in Appendix B because
heterogeneous retrievers define and expand re-
trieval units differently. These units include chunks,
statute articles, PCST seed nodes, tree nodes, and
entity–relation candidates, followed by method-
specific expansion and serialization procedures.
Therefore, no single measure, such as top- k, ar-
ticle count, node count, or token count can fully
equalize their retrieval budgets. We accordingly
compare complete pipelines under a shared reader,
corpus, evaluation set, and decoding configuration,
rather than retrieval algorithms using numerically
identical but semantically different retrieval units.
This follows prior GraphRAG evaluation practice:
LegalGraphRAG reports method-specific configu-
rations, while GraphRAG-Bench standardizes top-
kwhere directly applicable and otherwise retains
method-specific settings. Details for each baseline
are provided in Appendix B.
CoT baselineZero -shot-CoT (Kojima et al.,
2022) induces models to generate reasoning steps
by adding “Let’s think step by step,” while
Figure 2:Multi-hop performance trend.Accuracy
breakdown across hop counts on LawExamQA_Civil.
IRAC -CoT is a typical legal reasoning method
with an Issue–Rule–Application–Conclusion loop
demonstrated by MSLR (Yu et al., 2025).
Evaluation metrics.For LawExamQA_Civil,
we report exact-match accuracy (Acc.) and set-
level F1 over predicted and gold option sets. For
each item i, accuracy is 1 if the predicted set Pi
equals the gold set Gi, and 0 otherwise; set-level
F1 is F1i= 2|P i∩Gi|/(|P i|+|G i|). Both met-
rics are averaged over items rather than classes.
Empty predictions or outputs from which no an-
swer can be parsed receive zero for both metrics.
For LexRAG_Civil, we report Factuality, Satisfac-
tion, Clarity, Coherence, Completeness, and Over-
all scores. For PLawBench_Civil, we report Rea-
soning and Overall scores; judge models and scor-
ing details are given in Appendix D.
For provision retrieval, we report Recall@ kat
k∈ {8,10,20} to measure coverage of gold provi-
sions cited in the official rationales. We also report
the Gap Closing Rate (GCR), defined in Sec. 4.4, to
quantify the fraction of the accuracy gap between
the zero-shot and gold-article settings recovered by
each method.
We also compare LEGO with all 13 systems in
Table 1 using two-sided exact McNemar tests on
paired per-item predictions, percentile 95% boot-
strap confidence intervals from 10,000 item-level
resamples, and Holm correction across the com-
plete comparison family (Appendix L).

Method SizeLawExamQA_Civil Accuracy (%) by Hop Count
Overall 1-hop 2-hop 3-hop≥4-hop
Closed-source & Large General LMs
GPT-5 API 37.48 41.23 37.67 29.81 29.03
DeepSeek-V3 671B 39.97 41.23 39.53 41.3532.26
Open Larger LMs
Qwen3-30B-A3B 30B 36.93 39.18 38.14 30.77 30.65
GLM-4.7-Flash 30B 30.98 36.26 27.44 24.04 25.81
Small Open-source LMs
Qwen3-8B 8B 25.45 25.44 27.44 26.92 16.13
GLM-4-9B-chat 9B 29.18 30.41 27.44 30.77 25.81
Legal Domain LMs
DISC-LawLLM 7B 18.12 17.84 19.53 15.38 19.35
LegalOne 8B 30.29 32.16 31.63 20.19 32.26
RAG Baselines
Naive RAG 8B 28.91 30.41 29.30 26.92 22.58
HippoRAG 2 8B 29.88 30.70 29.30 29.81 27.42
RAPTOR 8B 29.32 28.95 31.16 30.77 22.58
G-Retriever 8B 31.67 31.58 35.81 26.92 25.81
LightRAG 8B 28.77 28.65 30.70 26.92 25.81
LEGO(ours) 8B 40.53 41.81 40.4737.50 38.71
Table 1:Main results stratified by multi-hop reasoning complexity on LawExamQA_Civil.Performance is
measured by exact-match accuracy (%).Boldindicates the absolute best performance across all evaluated models,
while underline indicates the second-best. By integrating ExpertGraphRAG and ExpertCoT, the 8B-parameter
LEGO achieves the highest observed overall accuracy and exhibits remarkable robustness in deep reasoning chains
(≥4-hop).
4.2 Main Results
Table 1 reports the main results on LawEx-
amQA_Civil. LEGO achieves the best overall
exact-match accuracy among the evaluated sys-
tems, reaching 40.53% with an 8B backbone. This
point estimate is numerically higher than those of
DeepSeek-V3 (39.97%) and GPT-5 (37.48%), al-
though paired tests do not establish statistically
significant superiority over these larger models.
By contrast, LEGO significantly outperforms all
five evaluated same-backbone RAG systems after
Holm correction, including G-Retriever (31.67%;
∆ = 8.86 pp, adjusted p= 4.4×10−6). The results
therefore suggest that external expert structure can
compensate for parameter scale, while statistically
supported superiority is limited to comparisons un-
der the shared Qwen3-8B backbone and Civil Code
corpus. Complete per-comparison statistics are re-
ported in Appendix L.
Hop-robustness.As shown in Figure 2, LEGO’s
advantage is most pronounced on deeper reason-
ing cases. On ≥4-hop questions, LEGO reaches
38.71%, outperforming the second-best result of
32.26% by 6.45 points. It also shows strongerhop-resilience: accuracy remains relatively sta-
ble across 1/2/3/4+-hop items (41.8 →40.5→
37.5→38.7), while GPT-5 drops by 12.2 points
from 1-hop to 4+-hop cases, and all five RAG base-
lines decline from 3-hop to ≥4-hop questions (by
1.1–8.2 points), LEGO remains stable and even
rises slightly. To better understand this robustness,
we analyze items correctly answered by LEGO but
missed by all 13 baselines. The qualitative anal-
ysis highlights four reasoning patterns: resolving
general and special rules, selecting among closely
related liability regimes, checking constitutive el-
ements before recognizing a claim, and identify-
ing the relevant right holder in multi-party settings.
These patterns align with the priority, prerequisite,
exception, and entitlement relations represented
in the Civil Law ExpertGraph, suggesting that le-
gal knowledge structure helps LEGO identify and
apply the controlling provisions. Appendix H pro-
vides detailed analyses of representative cases.
LexRAG_Civil and PLawBench_Civil.As
shown in Table 2, LEGO achieves the highest
scores among all RAG baselines on both bench-
marks. On LexRAG_Civil, LEGO achieves the

best score in every evaluated dimension, includ-
ing factuality (4.486), satisfaction (4.600), clarity
(6.879), coherence (5.700), completeness (5.093),
and overall quality (5.164). Compared with the
strongest baseline, HippoRAG 2, LEGO improves
the overall score by 0.335 points and completeness
by 0.407 points. On PLawBench_Civil, LEGO
obtains the highest reasoning score (47.40) and
overall score (59.68), suggesting that its expert-
structured retrieval and reasoning framework ben-
efits long-form legal reasoning. These results
demonstrate that LEGO improves not only sub-
jective answer quality, but also the robustness and
practical usability of legal reasoning in multi-turn
and long-form civil-law QA. Appendix K and Ap-
pendix J expand these two rows into case-level
analyses of multi-turn consultations and practical
case analyses, respectively.
4.3 Ablation Study
Table 3 shows that the full LEGO system consis-
tently outperforms all baselines and variants in both
Accuracy and F1, validating the contribution of
each core component. When the RAG module is re-
moved, or when the model relies only on standard
prompting strategies such as Qwen3-8B zero-shot,
CoT, and IRAC-CoT, performance drops substan-
tially. Compared with these No-RAG baselines,
LEGO improves Accuracy by 9.41–15.08 percent-
age points (pp) and F1 by 3.05–9.59 pp, highlight-
ing the difficulty of complex legal reasoning with-
out external legal knowledge retrieval. Even with a
retrieval module, general RAG frameworks paired
with standard prompts, such as Naive RAG and
G-Retriever with CoT/IRAC-CoT, still lag behind
the full system by 6.23–8.30 pp in Accuracy and
3.56–6.13 pp in F1. This suggests that generic re-
trieval pipelines struggle to capture the rigorous
structure required for statutory application. More-
over, although RAG + ExpertCoT variants benefit
from expert-annotated reasoning guidance, they
remain 4.71-4.98 pp lower in Accuracy and 2.72–
4.07 pp lower in F1 than LEGO, underscoring the
importance of our structurally optimized Expert-
GraphRAG for reducing semantic noise and sup-
porting deeper reasoning chains. Figure 3 further
shows that the benefit of each component is larger
when the other is present: ExpertGraphRAG im-
proves Accuracy by 1.25 pp with Plain CoT but by
4.71 pp with ExpertCoT, while ExpertCoT yields
gains of 1.52 and 4.98 pp with Naive RAG and Ex-
pertGraphRAG, respectively. The corresponding
Figure 3:Interaction between ExpertGraph and
Expert CoT .Accuracy improves most when both com-
ponents are enabled.
interaction analysis is reported in Appendix I.
4.4 Recall of Legal Retrieval
Table 4 shows that ExpertGraph substantially im-
proves gold-context approximation over plain re-
trieval. Compared with Naive RAG, CivilCode-
RAG + ExpertGraph raises Recall@8 from 27.99%
to 72.53%, Recall@10 from 34.70% to 74.58%,
and Recall@20 from 50.52% to 81.07%. This
improved provision coverage also translates into
stronger downstream QA performance, increasing
accuracy from 25.45% in the zero-shot setting to
30.29%. Notably, ExpertGraph achieves substan-
tially higher coverage of gold provisions while re-
lying entirely on automatically retrieved context,
indicating that structured legal-graph expansion
helps recover relevant provisions missed by plain
BM25 or dense retrieval. To diagnose why Expert-
Graph helps, Table 4 decomposes retrieval quality
into four facets and reports theGap Closing Rate
GCR(M) =ACC(M)−ACC zeroshot
ACC gold−ACC zeroshot.(8)
LEGO performs close to the gold-article ref-
erence.The ExpertGraph is built once, at corpus
level, independently of the benchmark questions,
answer labels and official rationales, and is then
reused across downstream legal reasoning tasks
without any task-specific annotation of gold provi-
sions. Under that setting LEGO reaches 40.53%
accuracy, against 41.22% for the gold-article refer-
ence in Table 3 (Gold Article + ExpertCoT), which
is handed the controlling provisions directly. The
0.69 pp gap corresponds to five items out of 723:
automatically retrieved provisions recover vast ma-
jority of the benefit of manually supplied ones.

Benchmark Metric Naive RAG RAPTOR G-Retriever HippoRAG 2 LightRAG LEGO
LexRAG_CivilFactuality 3.707 3.564 3.829 4.329 4.1214.486
Satisfaction 3.829 3.814 3.986 4.300 4.0434.600
Clarity 6.029 6.029 6.136 6.586 6.2866.879
Coherence 4.814 4.743 4.921 5.329 5.2575.700
Completeness 4.193 4.129 4.307 4.686 4.2795.093
Overall 4.314 4.293 4.450 4.829 4.5435.164
PLawBench_CivilReasoning 46.50 44.70 43.00 45.30 43.5347.40
Overall 59.05 58.02 57.22 57.93 58.4359.68
Table 2:Cross-benchmark evaluation on LexRAG_Civil and PLawBench_Civil.
MethodContext Coverage(%)QA GCR
Recall@8 Recall@10 Recall@20 Acc (%) F1
Naive RAG 27.99 34.70 50.52 28.91 52.16 0.61
BM25 (CivilCode, plain) 45.55 48.21 57.90 29.05 52.96 0.64
Dense (CivilCode, plain) 68.36 71.93 79.19 29.60 55.07 0.73
LEGO (CivilCode,w/ ExpertGraph)72.53 74.58 81.07 30.29 54.780.85
Gold Article 100 100 100 31.12 55.41 1.00
Table 4:Gold Context Approximation on LawExamQA_Civil.Coverage is reported as Recall@ katk∈
{8,10,20} . QA: Qwen3-8B zero-shot reader (no CoT), top-8 provisions. GCR = Gap Closing Rate on QA accuracy.
Model DescriptionLEGO Improvement (∆)
Acc F1∆Acc∆F1
No-RAG Baselines
Qwen3-8B zero-shot 0.2545 0.5382 ↑0.1508 ↑0.0305
Qwen3-8B CoT 0.2918 0.4728 ↑0.1135 ↑0.0959
Qwen3-8B IRAC-CoT 0.3112 0.4973 ↑0.0941 ↑0.0714
RAG + Standard CoT
Qwen3-8B Naive RAG + CoT 0.3430 0.5159 ↑0.0623 ↑0.0528
Qwen3-8B Naive RAG + IRAC-CoT 0.3223 0.5312 ↑0.0830 ↑0.0375
Qwen3-8B G-Retriever + CoT 0.3375 0.5074 ↑0.0678 ↑0.0613
Qwen3-8B G-Retriever + IRAC-CoT 0.3347 0.5331 ↑0.0706 ↑0.0356
RAG + ExpertCoT
Qwen3-8B Naive RAG + ExpertCoT 0.3582 0.5280 ↑0.0471 ↑0.0407
Qwen3-8B G-Retriever + ExpertCoT 0.3555 0.5415 ↑0.0498 ↑0.0272
ExpertGraphRAG + Standard CoT
LEGO CoT 0.3555 0.5312 ↑0.0498 ↑0.0375
LEGO IRAC-CoT 0.3734 0.5456 ↑0.0319 ↑0.0231
ExpertGraphRAG + ExpertCoT
LEGO Full System 0.4053 0.5687- -
Gold Article
Gold Article + CoT 0.3472 0.5257 - -
Gold Article + IRAC-CoT 0.3790 0.5468 - -
Gold Article + ExpertCoT 0.4122 0.5894 - -
Table 3:End-to-end RAG performance and abla-
tion study.Each row reports the full LEGO system’s
∆Acc/∆F1 over that configuration.
Constructing the ExpertGraph requires non-trivial
upfront effort, but because that effort is corpus-
level rather than item-level, it is amortised across
additional datasets and tasks rather than repaid for
each new benchmark.
4.5 Error Analysis
Beyond aggregate scores, comparing LEGO’s out-
puts with those of the baselines shows that it mainly
reduces three recurring failures of generic RAG sys-
tems: retrieving lexically similar but legally irrel-
evant provisions, producing partial answers based
on a single provision, and giving hedged conclu-
sions where the legal issue requires a determinate
judgment. By combining ExpertGraphRAG withExpertCoT, LEGO retrieves structurally relevant
provisions, composes them into more complete rea-
soning chains, and reaches determinate final judg-
ments. The errors that remain are concentrated in
the same structural steps: Appendix G codes the
items LEGO still mis-answers by the earliest rea-
soning step that makes the answer unrecoverable,
they arise mainly from incorrect provision selection
and prerequisite violation, with priority misappli-
cation, concept conflation, and exception bypass
accounting for most of the rest.
5 Conclusion
This paper presents LEGO, a legal domain
expertise-aware dual-module framework that syn-
ergizes expert GraphRAG and expert chain-of-
thought to solve the challenge of complex legal rea-
soning. ExpertGraphRAG selects provisions from
an expert-annotated Civil Code graph by normative
coverage rather than similarity alone, and Expert-
CoT organizes the retrieved provisions and case
facts into structured Provision-Fact-Conclusion rea-
soning. Experiments show that, with an 8B back-
bone, LEGO outperforms the evaluated RAG and
CoT baselines, performs comparably to the eval-
uated larger models, and remains robust on multi-
hop questions, while also leading the evaluated
baselines on two open-ended benchmarks. Abla-
tions confirm the individual and complementary
contributions of both modules. LEGO thus im-
proves both the complex legal reasoning of LLMs
and the interpretability of the analyses they pro-
duce.

Limitations
This paper focuses on a methodological frame-
work rather than proposing a new legal reason-
ing benchmark. LawExamQA_Civil is used only
as an evaluation set to validate expert-structured
retrieval and syllogistic reasoning, not as a stan-
dalone public multi-hop benchmark. Its hop count
is approximated by the number of Civil Code
provisions cited in official rationales, which does
not yet capture fine-grained inter-provision transi-
tions such as prerequisite, exception, priority, con-
dition–consequence, or general–special relations.
Our experiments evaluate answer accuracy, but do
not establish either the faithfulness of the gener-
ated analyses to the model’s internal reasoning
process or their correctness as post-hoc explana-
tions. Moreover, although the Civil Law Expert-
Graph is constructed independently of benchmark
questions, answer labels, and official rationales,
building and maintaining such a normative graph
requires substantial expert effort and may limit scal-
ability across jurisdictions and legal domains. The
Fact Graph is a conceptual representation; the im-
plementation passes the case text to the model and
does not construct an explicit fact graph.
Ethics Statement
LEGO is a methodological research framework for
evaluating expert-structured retrieval and syllogis-
tic reasoning in legal question answering. It is
not intended to provide legal advice, replace quali-
fied legal professionals, or support automated legal
decision-making. All evaluation questions are de-
rived from publicly available judicial-examination
materials and are used only for controlled exper-
imental evaluation. This work does not propose
LawExamQA_Civil as a new public benchmark,
nor does the evaluation encode or promote any par-
ticular legal interpretation. The Civil Law Expert-
Graph is constructed independently of benchmark
questions, answer labels, and official rationales.
Expert annotators were compensated at fair market
rates.
Use of AI Assistants
We acknowledge the use of several large lan-
guage models, including Claude, Gemini, GPT,
and Qwen, as well as GitHub Copilot, to assist
with text editing and code development during this
research. All outputs were thoroughly verified bythe authors, who maintain full accountability for
the paper’s contents.
Acknowledgements
We thank Modelbest for their technical support. We
also acknowledge the Tsinghua University Initia-
tive Scientific Research Program (20255080016)
for its support. In addition, we express our grati-
tude to the legal experts who collected the data and
analyzed the cases.
Antonino Rotolo was supported by the projects
EUSAiR “EU Regulatory Sandboxes for AI”
(DIGITAL-2024-AI-ACT-06-SANDBOX),
IT4LIA “Italy for Artificial Intelligence”(Grant
agreement ID: 101234224), and AISHA “Artificial
Intelligence Skills Hub Academy” (DIGITAL-
2025-SKILLS-08-GENAI-ACADEMY-STEP).
References
Andrew Blair-Stanek, Nils Holzenberger, and Benjamin
Van Durme. 2023. Can gpt-3 perform statutory rea-
soning? InProceedings of the Nineteenth Interna-
tional Conference on Artificial Intelligence and Law,
pages 22–31.
Ilias Chalkidis, Abhik Jana, Dirk Hartung, Michael
Bommarito, Ion Androutsopoulos, Daniel Martin
Katz, and Nikolaos Aletras. 2022. LexGLUE: A
benchmark dataset for legal language understanding
in English. InProceedings of the 60th Annual Meet-
ing of the Association for Computational Linguistics
(Volume 1: Long Papers), pages 4310–4330. Associ-
ation for Computational Linguistics.
Junying Chen, Zhenyang Cai, Ke Ji, Xidong Wang,
Wanlong Liu, Rongsheng Wang, Jianye Hou, and
Benyou Wang. 2024. Huatuogpt-o1, towards
medical complex reasoning with llms.Preprint,
arXiv:2412.18925.
Shengyuan Chen, Chuang Zhou, Zheng Yuan, Qing-
gang Zhang, Zeyang Cui, Hao Chen, Yilin Xiao,
Jiannong Cao, and Xiao Huang. 2026a. You don’t
need pre-built graphs for RAG: Retrieval augmented
generation with adaptive reasoning structures. InPro-
ceedings of the 40th AAAI Conference on Artificial
Intelligence.
Zerui Chen, Qinggang Zhang, Zhishang Xiang, Zhimin
Wei, Linfeng Gao, Xiao Huang, Zhihong Zhang,
and Jinsong Su. 2026b. LegalGraphRAG: Multi-
agent graph retrieval-augmented generation for re-
liable legal reasoning. InProceedings of the 64th
Annual Meeting of the Association for Computational
Linguistics (Volume 1: Long Papers), pages 37455–
37484, San Diego, California, United States. Associ-
ation for Computational Linguistics.

Jiaxi Cui, Munan Ning, Zongjian Li, Hao Li, Yang
Ya, Bohua Chen, Bin Ling, Yonghong Tian, and
Li Yuan. 2026. Chatlaw: A multi-agent legal as-
sistant based on a role-aligned mixture-of-experts
architecture.Fundamental Research.
DeepSeek-AI. 2024. Deepseek-v3 technical report.
arXiv preprint arXiv:2412.19437.
Darren Edge, Ha Trinh, Newman Cheng, Joshua
Bradley, Alex Chao, Apurva Mody, Steven Truitt,
and Jonathan Larson. 2024. From local to global: A
graph RAG approach to query-focused summariza-
tion. arXiv:2404.16130.
Xueren Ge, Sahil Murtaza, Anthony Cortez, and Homa
Alemzadeh. 2026. Expert-guided prompting and
retrieval-augmented generation for emergency med-
ical service question answering. InProceedings of
the AAAI Conference on Artificial Intelligence, vol-
ume 40, pages 30798–30806. Accepted at AAAI-26;
extended version: arXiv:2511.10900.
Shengbo Gong, Xianfeng Tang, Qi He, Carl Yang, and
Wei Jin. 2026. Beyond chunks and graphs: Retrieval-
augmented generation through triplet-driven thinking.
InFindings of the Association for Computational Lin-
guistics: ACL 2026, pages 26282–26308, San Diego,
California, United States. Association for Computa-
tional Linguistics.
Neel Guha, Julian Nyarko, Daniel Ho, Christopher Ré,
Adam Chilton, Alex Chohlas-Wood, Austin Peters,
Brandon Waldon, Daniel Rockmore, Diego Zam-
brano, and 1 others. 2023. Legalbench: A collab-
oratively built benchmark for measuring legal reason-
ing in large language models.Advances in neural
information processing systems, 36:44123–44279.
Zirui Guo, Lianghao Xia, Yanhua Yu, Tu Ao, and Chao
Huang. 2025. LightRAG: Simple and fast retrieval-
augmented generation. InFindings of the Associa-
tion for Computational Linguistics: EMNLP 2025,
pages 10746–10761, Suzhou, China. Association for
Computational Linguistics.
Bernal Jiménez Gutiérrez, Yiheng Shu, Weijian Qi,
Sizhe Zhou, and Yu Su. 2025. From RAG to mem-
ory: Non-parametric continual learning for large lan-
guage models. InProceedings of the 42nd Inter-
national Conference on Machine Learning (ICML).
ArXiv:2502.14802.
Haoyu Han, Li Ma, Yu Wang, Harry Shomer, Kai Guo,
Yongjia Lei, Zhisheng Qi, Zhigang Hua, Bo Long,
Hui Liu, Charu Aggarwal, and Jiliang Tang. 2026.
Rag vs. graphrag: A systematic evaluation and key
insights. InProceedings of the 32nd ACM SIGKDD
Conference on Knowledge Discovery and Data Min-
ing V .2, KDD ’26, pages 8966–8977, New York, NY ,
USA. Association for Computing Machinery.
Xiaoxin He, Yijun Tian, Yifei Sun, Nitesh V Chawla,
Thomas Laurent, Yann LeCun, Xavier Bresson, and
Bryan Hooi. 2024. G-retriever: Retrieval-augmentedgeneration for textual graph understanding and ques-
tion answering.Advances in Neural Information
Processing Systems, 37:132876–132907.
Xanh Ho, Anh-Khoa Duong Nguyen, Saku Sugawara,
and Akiko Aizawa. 2020. Constructing a multi-
hop QA dataset for comprehensive evaluation of
reasoning steps. InProceedings of the 28th Inter-
national Conference on Computational Linguistics,
pages 6609–6625.
Yiran Hu, Huanghai Liu, Qingjing Chen, Ning Zheng,
Chong Wang, Yun Liu, Charles LA Clarke, and Weix-
ing Shen. 2025. J&h: Evaluating the robustness of
large language models under knowledge-injection at-
tacks in legal domain. InProceedings of the AAAI
Conference on Artificial Intelligence, volume 39,
pages 28106–28115.
Xiaoxi Kang, Lizhen Qu, Lay-Ki Soon, Adnan Tra-
kic, Terry Zhuo, Patrick Emerton, and Genevieve
Grant. 2023. Can ChatGPT perform reasoning using
the IRAC method in analyzing legal scenarios like
a lawyer? InFindings of the Association for Com-
putational Linguistics: EMNLP 2023, pages 13900–
13923, Singapore. Association for Computational
Linguistics.
Takeshi Kojima, Shixiang Shane Gu, Machel Reid, Yu-
taka Matsuo, and Yusuke Iwasawa. 2022. Large lan-
guage models are zero-shot reasoners.Advances in
Neural Information Processing Systems, 35:22199–
22213.
Jihyung Lee, Daehui Kim, Seonjeong Hwang, Hy-
ounghun Kim, and Gary Lee. 2025. Koblex: Open
legal question answering with multi-hop reasoning.
InProceedings of the 2025 Conference on Empiri-
cal Methods in Natural Language Processing, pages
4019–4053.
Patrick Lewis, Ethan Perez, Aleksandra Piktus, Fabio
Petroni, Vladimir Karpukhin, Naman Goyal, Hein-
rich Küttler, Mike Lewis, Wen-tau Yih, Tim Rock-
täschel, Sebastian Riedel, and Douwe Kiela. 2020.
Retrieval-augmented generation for knowledge-
intensive nlp tasks. InAdvances in Neural Infor-
mation Processing Systems, volume 33, pages 9459–
9474. Curran Associates, Inc.
Feiyang Li, Peng Fang, Zhan Shi, Arijit Khan, Fang
Wang, Dan Feng, Weihao Wang, Xin Zhang, and
Yongjian Cui. 2025a. CoT-RAG: Integrating chain-
of-thought and retrieval-augmented generation to en-
hance reasoning in large language models. InFind-
ings of the Association for Computational Linguistics:
EMNLP 2025, pages 3119–3171.
Haitao Li, Yifan Chen, Yiran Hu, Qingyao Ai, Jun-
jie Chen, Xiaoyu Yang, Jianhui Yang, Yueyue Wu,
Zeyang Liu, and Yiqun Liu. 2025b. Lexrag: Bench-
marking retrieval-augmented generation in multi-turn
legal consultation conversation. InProceedings of
the 48th International ACM SIGIR Conference on
Research and Development in Information Retrieval,
pages 3606–3615.

Haitao Li, Yifan Chen, Shuo Miao, Qian Dong, Jia
Chen, Yiran Hu, Junjie Chen, Minghao Qin, Yueyue
Wu, Yujia Zhou, Qingyao Ai, Yiqun Liu, Cheng Luo,
Quan Zhou, Ya Zhang, and Jikun Hu. 2026a. Lega-
lone: A family of foundation models for reliable legal
reasoning.ArXiv, abs/2602.00642.
Ningyuan Li, Junrui Liu, Yi Shan, Minghui Huang,
Ziren Gong, and Tong Li. 2026b. Pankrag: Enhanc-
ing graph retrieval via globally aware query reso-
lution and dependency-aware reranking mechanism.
InICASSP 2026 – 2026 IEEE International Con-
ference on Acoustics, Speech and Signal Process-
ing (ICASSP), pages 19157–19161, Barcelona, Spain.
IEEE.
Hui Lin and Jeff Bilmes. 2011. A class of submodular
functions for document summarization. InProceed-
ings of the 49th annual meeting of the association
for computational linguistics: human language tech-
nologies, pages 510–520.
Huanghai Liu, Quzhe Huang, Qingjing Chen, Yiran
Hu, Jiayu Ma, Yun Liu, Weixing Shen, and Yansong
Feng. 2025. Jurex-4e: Juridical expert-annotated
four-element knowledge base for legal reasoning.
InProceedings of the 2025 Conference on Empir-
ical Methods in Natural Language Processing, pages
3794–3814.
George L. Nemhauser, Laurence A. Wolsey, and Mar-
shall L. Fisher. 1978. An analysis of approximations
for maximizing submodular set functions—i.Mathe-
matical Programming, 14(1):265–294.
OpenAI. 2025. GPT-5 System Card. Technical report.
Available at https://cdn.openai.com/gpt-5-s
ystem-card.pdf.
Günther Patzig. 2013.Aristotle’s theory of the syllo-
gism: A logico-philological study of book A of the
Prior Analytics. Springer Science & Business Media.
Qwen Team. 2025. Qwen3 technical report.arXiv
preprint arXiv:2505.09388.
Parth Sarthi, Salman Abdullah, Aditi Tuli, Shubh
Khanna, Anna Goldie, and Christopher D. Manning.
2024. Raptor: Recursive abstractive processing for
tree-organized retrieval.Preprint, arXiv:2401.18059.
Sergio Servantez, Joe Barrow, Kristian Hammond, and
Rajiv Jain. 2024. Chain of logic: Rule-based rea-
soning with large language models. InFindings of
the Association for Computational Linguistics: ACL
2024, pages 2721–2733, Bangkok, Thailand. Associ-
ation for Computational Linguistics.
Yuzhen Shi, Huanghai Liu, Yiran HU, Song Gao-
jie, Xu Xinran, Yubo Ma, Tianyi Tang, Li Zhang,
Qingjing Chen, Feng Di, Wenbo Lv, Weiheng Wu,
Kexin Yang, Sen Yang, Wei Wang, Rongyao Shi,
Qiu Yuanyang, Yuemeng Qi, Zhang Jingwen, and
11 others. 2026. PLAWBENCH: A rubric-based
benchmark for evaluating LLMs in real-world legalpractice. InProceedings of the 64th Annual Meet-
ing of the Association for Computational Linguis-
tics (Volume 1: Long Papers), pages 10067–10116,
San Diego, California, United States. Association for
Computational Linguistics.
Harsh Trivedi, Niranjan Balasubramanian, Tushar Khot,
and Ashish Sabharwal. 2022. MuSiQue: Multi-
hop questions via single hop question composition.
Transactions of the Association for Computational
Linguistics, 10:539–554.
Harsh Trivedi, Niranjan Balasubramanian, Tushar Khot,
and Ashish Sabharwal. 2023. Interleaving retrieval
with chain-of-thought reasoning for knowledge-
intensive multi-step questions. InProceedings of
the 61st Annual Meeting of the Association for Com-
putational Linguistics.
Jie Wang, Honghua Huang, Xi Ge, Jianhui Su, Wen
Liu, and Shiguo Lian. 2026. Omd-graphrag: En-
hancing graphrag with ontology-guided extraction,
multi-dimensional clustering and dual-channel fusion.
Preprint, arXiv:2603.25152.
Zihao Wang, Anji Liu, Haowei Lin, Jiaqi Li, Xi-
aojian Ma, and Yitao Liang. 2024. RAT: Re-
trieval augmented thoughts elicit context-aware rea-
soning in long-horizon generation.arXiv preprint
arXiv:2403.05313.
Jason Wei, Xuezhi Wang, Dale Schuurmans, Maarten
Bosma, Brian Ichter, Fei Xia, Ed Chi, Quoc V . Le,
and Denny Zhou. 2022. Chain-of-thought prompt-
ing elicits reasoning in large language models. In
Advances in Neural Information Processing Systems.
Chaojun Xiao, Xueyu Hu, Zhiyuan Liu, Cunchao Tu,
and Maosong Sun. 2021. Lawformer: A pre-trained
language model for Chinese legal long documents.
InAI Open, volume 2, pages 79–84.
Zhilin Yang, Peng Qi, Saizheng Zhang, Yoshua Ben-
gio, William W. Cohen, Ruslan Salakhutdinov, and
Christopher D. Manning. 2018. HotpotQA: A dataset
for diverse, explainable multi-hop question answer-
ing. InProceedings of the 2018 Conference on Em-
pirical Methods in Natural Language Processing,
pages 2369–2380. Association for Computational
Linguistics.
Shunyu Yao, Dian Yu, Jeffrey Zhao, Izhak Shafran,
Tom Griffiths, Yuan Cao, and Karthik Narasimhan.
2023. Tree of thoughts: Deliberate problem solving
with large language models. InAdvances in Neural
Information Processing Systems.
Fangyi Yu, Lee Quartey, and Frank Schilder. 2023.
Exploring the effectiveness of prompt engineering
for legal reasoning tasks. InFindings of the Asso-
ciation for Computational Linguistics: ACL 2023,
pages 13582–13596, Toronto, Canada. Association
for Computational Linguistics.

Wenhan Yu, Xinbo Lin, Lanxin Ni, Jinhua Cheng,
and Lei Sha. 2025. Benchmarking multi-step le-
gal reasoning and analyzing chain-of-thought ef-
fects in large language models.arXiv preprint
arXiv:2511.07979.
Shengbin Yue, Shujun Liu, Yuxuan Zhou, Chenchen
Shen, Siyuan Wang, Yao Xiao, Bingxuan Li, Yun
Song, Xiaoyu Shen, Wei Chen, Xuanjing Huang, and
Zhongyu Wei. 2024. Lawllm: Intelligent legal sys-
tem with legal reasoning and verifiable retrieval. In
Database Systems for Advanced Applications, pages
304–321, Singapore. Springer Nature Singapore.
zai-org. 2024. GLM-4-9B-Chat model card. Hugging
Face repository. Available at https://huggingfac
e.co/zai-org/glm-4-9b-chat.
zai-org. 2026. GLM-4.7-Flash model card. Hugging
Face repository. Available at https://huggingfac
e.co/zai-org/GLM-4.7-Flash.
Qinggang Zhang, Shengyuan Chen, Yuan-Qi Bei, Zheng
Yuan, Huachi Zhou, Zijin Hong, Junnan Dong, Hao
Chen, Yi Chang, and Xiao Huang. 2025. A survey of
graph retrieval-augmented generation for customized
large language models.ArXiv, abs/2501.13958.
Dongzhuoran Zhou, Yuqicheng Zhu, Xiaxia Wang,
Hongkuan Zhou, Yuan He, Jiaoyan Chen, Steffen
Staab, and Evgeny Kharlamov. 2026. What breaks
knowledge graph based rag? benchmarking and
empirical insights into reasoning under incomplete
knowledge. InProceedings of the 19th Conference of
the European Chapter of the Association for Compu-
tational Linguistics (Volume 1: Long Papers), pages
2522–2538.
Luyao Zhuang, Shengyuan Chen, Yilin Xiao, Huachi
Zhou, Yujing Zhang, Hao Chen, Qinggang Zhang,
and Xiao Huang. 2026. Linearrag: Linear graph
retrieval augmented generation on large-scale cor-
pora. InInternational Conference on Learning Rep-
resentations, volume 2026, pages 147053–147075.
ArXiv:2510.10114.
Appendix Contents
•Appendix A:Syllogistic CoT Prompt Card
(p. 13).
•Appendix B:Baseline Details and Hyperparam-
eters (p. 16).
•Appendix C:P/F/C Case Studies with Pipeline-
Level Reasoning (p. 17).
•Appendix D:Dataset Card (p. 21).
•Appendix E:LegalExpert-Domain Graph Con-
struction (p. 23).
•Appendix F:Civil Law ExpertGraph: Source
Statistics (p. 28).•Appendix G:Detailed Error Analysis (p. 28).
•Appendix H:Multi-Hop Case Studies on LawEx-
amQA_Civil (p. 30).
•Appendix I:Component-Synergy Case Studies
on LawExamQA_Civil (p. 34).
•Appendix J:PLawBench_Civil Case Studies
(p. 38).
•Appendix K:LexRAG_Civil Case Studies
(p. 40).
•Appendix L:Statistical Validation of the Main
Results (p. 44).
A Syllogistic CoT Prompt Card
This appendix reproduces the full prompts used
by LEGO’s Syllogistic CoT step (§3.2) on the pro-
duction run that obtains 0.4053/0.5687 on LawEx-
amQA_Civil. Both theSYSTEMandUSERprompts
are passed to the same Qwen3-8B backbone in a
single call with greedy decoding ( temperature =
0.0,max_tokens= 1800).
A.1.1 System prompt
PROMPT:You are a careful multiple-choice
assistant for PRC civil-law questions. You
must reason strictly from the case facts, the
options, and the provided statutes. Do not
invent facts or statutes that are not given.
A.1.2 User prompt template
The user prompt is a fixed template into which three
blocks are filled at run-time: {law_context} (the
top-Nretrieved Civil-Code articles, each rendered
as “[k] PRC Civil Code Art. c: ... ”);{case}
(the case body); and {question_with_options}
(the question stem followed by the four options
A–D).
PROMPT:
Please solve this PRC civil-law multiple-
choice question using the P/F/C (Provision–
Fact–Conclusion) method.
Role:write as a neutral adjudicator / exam
grader.
Requirements:
1.In one sentence, state the legal object to be
decided and what the question asks.
2.P-Provision: list only the controlling rules
among the[Provided Statutes]; do not add

any new statutes.
3.In P, explicitly state the provision relations
(general vs. special rule, proviso/exception,
limiting condition, remedy) and which one
controls the conclusion.
4.Carefully distinguish: contract validity,
real-rights transfer, opposability, liability
allocation, damages, priority rights / de-
fenses; do not expand one legal effect into
another.
5.When you see limiting clauses (e.g., “shall
not be deemed invalid solely because...”,
“does not affect the validity of...”, “may
claim compensation”), interpret them as
limitations; do not reverse-infer invalidity
or automatic extinction of rights.
6.F-Fact: list only the legally relevant facts;
do not write the final legal conclusion.
7.C-Conclusion: first perform a “provision-
relation self-check”, then judge options
A/B/C/D. Use at most one sentence per
option.
8.Select options according to the question
polarity. If it asks for “incorrect / unlaw-
ful / not established”, choose the incorrect
option(s).
9.Do not multi-select for coverage. Select an
option only when its subject, legal predi-
cate, required elements, and claimed legal
effect all match the Provision–Fact map-
ping.
10. Keep each section to 1–3 sentences; focus
on comparing options rather than repeating
the case.
11.The last line must be ex-
actly [FINAL_ANSWER]X<eoa> ,
e.g., [FINAL_ANSWER]A<eoa> or
[FINAL_ANSWER]BD<eoa>.
[Provided Statutes]
{law_context}
[Case]
{case}
[Question/Options]
{question_with_options}
Output format:
Object:
P-Provision:
Provision relations:F-Fact:
C-Conclusion:
Provision-relation self-check:
[FINAL_ANSWER]...<eoa>
A.1.3 End-to-end example (qid 0)
The following is the user prompt as actually
sent for question qid 0 (2002 Paper III No. 1,
the transport-delay / cargo-damage case used
as the worked example in Appendix C). Only
the three slot fillers ( {law_context} ,{case} ,
{question_with_options} ) differ from the tem-
plate above.
{law_context}.
{law_context}:
[1]PRC Civil CodeArt. 832: The carrier shall
be liable for compensation for any damage to
or loss of the goods during transport. How-
ever, where the carrier proves that such dam-
age or loss was caused by force majeure, the
natural properties of the goods or reasonable
wear and tear, or the fault of the consignor or
consignee, the carrier shall not be liable for
compensation.
[2]PRC Civil CodeArt. 834: Where two or
more carriers undertake through-transport by
the same mode of transport, the carrier that
entered into the contract with the consignor
shall be liable for the whole course of trans-
port; where the loss occurs on a particular seg-
ment, the carrier that entered into the contract
with the consignor and the carrier responsible
for that segment shall be jointly and severally
liable.
[3]PRC Civil CodeArt. 825: Where the con-
signor handles the carriage of goods, it shall
accurately state to the carrier the name of the
consignee, the name or the consignee as in-
structed, and the necessary information about
the goods for carriage such as the name, na-
ture, weight, quantity, and place of delivery.
Where the consignor makes a false declaration
or omits important information and thereby
causes losses to the carrier, the consignor shall
be liable for compensation.
[4–8]PRC Civil CodeArts. 841 / 842 / 824
/ 607 / 837 (multimodal transport, passenger
transport, risk allocation in sales contracts, de-
posit/consignation, etc.; full text omitted; not

on the main reasoning path of this question).
{case}.
{case}:
Company A needs to ship a batch of goods
to Company B (the consignee). The parties
agreed that Company A would arrange the
logistics. Company A’s legal representative,
C, contacted by phone and commissioned a
trucking company holding a road-transport
business license to transport the goods. The
trucking company assigned its full-time em-
ployee driver, Liu, to drive a dedicated vehicle.
During transport, a traffic accident occurred
due to Liu’s negligence; the traffic police ac-
cident report found Liu fully at fault, and the
goods were damaged. Company B had previ-
ously queried the trucking company directly
about the location of the goods. Company B
suffered losses because it failed to receive the
goods in time.
{question_with_options}.
{question_with_options}:
Since the damage was directly caused by the
transport side, who should Company B claim
compensation from?
Options:
A. Company A
B. C
C. Liu
D. The trucking company, as the actual carrier
that directly caused the damage; holding it
liable for damages accords with tort-law prin-
ciples.
Expected output skeleton.The model re-
turns the five labelled sections of §3.2 popu-
lated against this input, ending with the token
[FINAL_ANSWER]A<eoa> (the gold answer for this
item). The full P / F / C body is stored in the predic-
tion log and is the source for the worked example
in Appendix C.

B Baseline Details and Hyperparameters
This section lists the configurations used for base-
lines reported in Table 1, Table 2, and Table 3.
B.1 RAG Baselines
Naive RAG.Naive RAG (Lewis et al., 2020) re-
trieves the top- kCivil-Code chunks by dense sim-
ilarity and concatenates them as context for the
generator.
Naive RAG Configuration
{
embedding_model: Qwen3-Embedding-8B,
retrieval_topk: 5,
chunk_token_size: 1000,
chunk_overlap_token_size: 200
}
G-Retriever.G-Retriever (He et al., 2024) solves
a Prize-Collecting Steiner Tree over the Civil-Code
knowledge graph, returning a connected sub-tree
spanning the top-kprize nodes and their relations.
G-Retriever Configuration
{
embedding_model: Qwen3-Embedding-8B,
retrieval_topk: 10,
chunk_token_size: 1200,
chunk_overlap_token_size: 100,
entities_max_tokens: 3000,
relationships_max_tokens: 2000
}
LightRAG.LightRAG (Guo et al., 2025) per-
forms hybrid retrieval over an auto-induced entity–
relation graph, scoring queries against a global-
relation channel and a local-entity channel with
separate token budgets.
LightRAG Configuration
{
embedding_model: Qwen3-Embedding-8B,
query_type: hybrid,
retrieval_topk: 20,
chunk_token_size: 1200,
chunk_overlap_token_size: 100,
max_token_global_context: 2000,
max_token_local_context: 2000,
max_token_text_unit: 2000
}
HippoRAG 2.HippoRAG 2 (Gutiérrez et al.,
2025) links query mentions into a pre-built fact
/ passage hybrid graph and accumulates evidence
via multi-step traversal before returning the final
passages.HippoRAG 2 Configuration
{
embedding_model: Qwen3-Embedding-8B,
retrieval_top_k: 10,
linking_top_k: 7,
qa_top_k: 10,
max_qa_steps: 3,
graph_type: facts_and_sim_passage_
node_unidirectional
}
RAPTOR.RAPTOR (Sarthi et al., 2024) con-
structs a hierarchical summary tree by recursively
clustering and summarising chunks, enabling re-
trieval at multiple levels of abstraction; we use the
collapsed-tree variant.
RAPTOR Configuration
{
embedding_model: Qwen3-Embedding-8B,
retrieval_topk: 10,
chunk_token_size: 1200,
chunk_overlap_token_size: 100,
num_layers: 5,
max_length_in_cluster: 3500,
threshold: 0.1,
cluster_metric: cosine,
threshold_cluster_num: 5000,
max_context_tokens: 4500
}
B.2 CoT Baselines
CoT baselines pair with any retriever in the ab-
lation: the retrieved articles fill the law_context
block while the scaffold controls how the generator
consumes them.
Zero-shot CoT.Zero-shot CoT (Kojima et al.,
2022) prepends “Let’s think step by step” and lets
the model freely generate a reasoning chain before
committing an answer.
Zero-shot CoT Configuration
{
backbone: Qwen3-8B,
sampling: greedy,
max_tokens: 1800,
top_n_articles: 8,
scaffold: "Let’s think step by step",
output_skeleton: [Reasoning]→
[Option check]→
[Answer]X<eoa>
}
IRAC-CoT.IRAC-CoT (Yu et al., 2025) forces a
four-section Issue–Rule–Application–Conclusion
trace before the answer token.

IRAC-CoT Configuration
{
backbone: Qwen3-8B,
sampling: greedy,
max_tokens: 1800,
top_n_articles: 8,
scaffold: Issue→Rule→
Application→Conclusion,
output_skeleton: [Issue]→[Rule]→
[Application]→
[Conclusion]→
[Answer]X<eoa>
}
B.3 Experiment Settings
For LEGO and all retrieval-based baselines, we
use Qwen3-8B as the generator to ensure a fair
backbone-controlled comparison. During genera-
tion, all open-source experiments use greedy decod-
ing with temperature =0 and max_tokens =1800
(512 for the closed-book zero-shot setting, whose
output is a bare answer tag); the Qwen3-
8B reader is run with thinking mode disabled
(enable_thinking=False ) in all configurations;
API-based experiments use the providers’ default
decoding configurations. For LEGO’s provision
retrieval on LawExamQA_Civil, we instantiate the
Normative Coverage Greedy (NCG) objective of
Eq. 3–4 with the weights listed below; all di/gi
similarities are min–max normalized to [0,1] per
query before the greedy selection. To guarantee
reproducibility, we fix the random seed at 42. All
open-source model inferences and evaluations are
conducted on a single server equipped with 8 ×
NVIDIA H100 (80GB HBM3) GPUs.
LEGO Configuration
{
backbone: Qwen3-8B,
embedding_model: Qwen3-Embedding-8B,
sampling: greedy,
temperature: 0.0,
max_tokens: 1800,
seed: 42,
λalign: 0.55, // Eq. 3
λcov: 0.18, // Eq. 3
λred: 0.06, // Eq. 3
article_budget K: 8, // Eq. 4
top_n_articles N: 8
}
The three weights are lightly tuned on a held-out
development split, giving the largest weight to
query–article alignment ( λalign= 0.55 ), a mod-
erate weight to marginal normative-rule coverage
(λcov= 0.18 ), and a small redundancy penalty
(λred= 0.06 ). The article budget K=N= 8
means NCG selects eight articles per query and alleight are passed into the Syllogistic-CoT prompt
as{law_context}.
CP/F/C Case Studies with Pipeline-Level
Reasoning
This appendix expands six representative ablation
examples into case-level P/F/C (Provision–Fact–
Conclusion) comparisons. The original benchmark
cases are Chinese civil-law multiple-choice ques-
tions; to avoid encoding problems in the appendix,
the case facts and options below are English ren-
derings of the original items. For each case, green
cells mark legally correct P/F/C steps, red cells
mark the step that causes or directly contributes to
the wrong answer, and yellow cells mark partially
correct reasoning that still misses the decisive legal
boundary.
C.1 QID 19: Contract Rights and Obligations
Transfer
Case.Company A and Korean Company B
formed a Chinese-foreign joint venture, and the
joint-venture contract had already been approved
by the competent authority. Company C is a large
state-owned enterprise with the relevant qualifica-
tions. Company A wrote to Company B: “We plan
to transfer our equity to Company C. If you object,
please raise a written objection within 30 days;
otherwise you will be deemed to have consented.”
Company A then transferred its contractual rights
and obligations to Company C without Company
B’s consent. Company B did not reply in writing
for three months after receiving the notice. Com-
pany C had already participated in the joint ven-
ture’s operations for one year, with good business
results, and Company B had once signed a board
resolution together with Company C.
Question and options.Given that Company C
has actually performed contractual obligations and
Company B did not object in time, is the transfer
of the contract illegal?
A. Yes.
B.No. Based on commercial efficiency and
actual performance, the transfer should be
treated as valid, and Company B impliedly
recognized it by joining board actions.
C.It is illegal only if Company B expressly ob-
jects.

Observation QID Gold Prediction contrast Dominant P/F/C difference
Obs. 1 retrieval
helps19 A zero-shot B vs. RAG/LEGO A P: retrieved transfer-consent rules block
implied-consent intuition.
Obs. 1 retrieval
insufficient4 D plain RAG B vs. LEGO Full D P: apparent agency must override bare no-
authority reasoning.
Obs. 2 Expert-
CoT at fixed re-
trieval1 C CoT CD vs. ExpertCoT C C: termination damages are not full agreed
remuneration.
Obs. 3 Expert-
Graph alone36 B plain RAG/RAG+ExpertCoT C vs.
LEGO CoT BP: graph supplies the negative rule that no
remuneration claim exists.
Obs. 4 synergy 7 D single components A/B vs. LEGO Full
DP+F: the good-faith, no-present-benefit ex-
ception controls the result.
Obs. 5 strongest
baseline16 C G-Retriever+ExpertCoT D vs. LEGO
Full CC: the private return claim is limited to
principal plus legal fruits.
Table 4:Six representative P/F/C case studies.
D.It depends on whether the approval authority
later ratifies the transfer.
Difference analysis.The legal core is consent
for a combined transfer of contractual rights and
obligations. Civil Code Articles 551, 555, and
556 require consent by the other party; Article 140
permits implied expression but does not make si-
lence equal consent by default. RAG pipelines im-
prove because retrieval supplies this major premise.
LEGO Full adds a relation check in the P stage,
preventing later performance or business efficiency
from being converted into statutory consent.
C.2 QID 4: Stamped Blank Contract and
Apparent Agency
Case.Zhang was a salesperson of an enterprise
and carried blank contract forms bearing the en-
terprise’s official seal for external contracting. Be-
cause Zhang accepted kickbacks, the enterprise
formally removed him and circulated an internal
email about the removal, but it did not recover the
stamped blank contracts. On the day after leaving
the enterprise, Zhang concealed his departure and
used one of the stamped blank contracts to sign a
purchase and sale agreement with a counterparty.
The counterparty only checked the authenticity of
the seal and did not ask Zhang to produce proof of
employment. The enterprise’s internal rules state
that former employees may not use company con-
tract forms. The enterprise also copied Zhang’s
removal notice to all long-term business partners,
including the counterparty.
Question and options.Because Zhang had been
removed and had no current authorization, and be-
cause he did not produce any updated authorization
document, how should the purchase and sale agree-
ment be characterized?A. It was not formed.
B.It is invalid because Zhang lacked agency au-
thority, and the counterparty was clearly negli-
gent in ignoring the enterprise’s formal notice
and relying only on past impressions.
C. It is voidable.
D. It was formed and became effective.
Difference analysis.Article 172 provides that an
unauthorized agency act is effective if the counter-
party has reason to believe authority exists. The
civil-law graph lists stamped blank contracts as
a typical apparent-authority appearance. Correct
pipelines succeed because the P stage treats appar-
ent agency as a special rule that can override bare
no-authority reasoning. Plain RAG and LEGO CoT
fail by overweighting absence of current authority.
Doctrinal caveat.The rewritten facts include no-
tice of Zhang’s removal to the counterparty. If that
notice is treated as effectively received and under-
stood, reasonable reliance may be weakened. The
benchmark gold and source explanation preserve
the original apparent-agency conclusion, so this
case should be presented as LEGO Full matching
the benchmark’s rule hierarchy, not as a universal
conclusion for all notice variants.
C.3 QID 1: Mandate Termination and Loss
Compensation
Case.Party A orally authorized Party B to pur-
chase timber on A’s behalf, and the parties agreed
that B would receive remuneration for the service.
B spent time and effort on the matter, repeatedly
visited timber markets, and contacted several sup-
pliers. B had already selected specific timber spec-
ifications and reported quotations and stock infor-
mation to A, who raised no objection. Later, A no

Pipeline Pred. P-Provision F-Fact C-Conclusion
Qwen3-8B zero-shot B Does not anchor the answer
in Articles 551, 555, and 556.Overweights actual operation,
silence, and board participa-
tion.Treats commercial efficiency
and silence as implied consent.
Naive RAG + CoT A Identifies consent as required
for transfer of rights and obli-
gations.Makes the lack of Company
B’s consent decisive.Finds the transfer illegal and se-
lects A.
G-Retriever + Expert-
CoTD Lists the transfer-consent
rules.Extracts the key facts cor-
rectly.Shifts the answer from consent
to later administrative ratifica-
tion.
LEGO Full A Combines Articles 551, 555,
and 556 as validity condi-
tions.Treats silence and perfor-
mance as insufficient substi-
tutes for consent.Concludes that the missing con-
sent makes the transfer illegal.
Table 5: P/F/C comparison for QID 19.
Pipeline Pred. P-Provision F-Fact C-Conclusion
Naive RAG + CoT B Stops at the general no-
authority rule and misses ap-
parent agency.Overweights removal, no up-
dated authorization, and lack
of checking.Infers invalidity directly from
lack of authority.
Naive RAG + Expert-
CoTD Treats Article 172 apparent
agency as the special effec-
tiveness rule.Uses the stamped blank con-
tract as an appearance at-
tributable to the enterprise.Finds apparent agency and se-
lects D.
LEGO CoT B Mixes Articles 171, 172, and
504 without stable priority.Treats failure to verify current
employment as decisive fault.Incorrectly classifies the agree-
ment as invalid.
LEGO Full D Gives Article 172 priority
over the general no-authority
rule.Focuses on the unrecovered
stamped blank contract as ap-
parent authority.Finds the agreement formed
and effective.
Table 6: P/F/C comparison for QID 4.
longer wanted the timber and telephoned B to can-
cel the mandate without mentioning compensation.
B objected.
Question and options.A orally mandated B to
purchase timber and agreed to pay remuneration.
A now cancels the mandate. Given that B has per-
formed substantial work and completed the main
entrusted affairs, which statements are correct?
A.A has no right to unilaterally cancel the
mandate; otherwise A must compensate B’s
losses.
B.A may unilaterally cancel the mandate, but
only in writing.
C.A may unilaterally cancel the mandate, but
must compensate B’s losses.
D.Because B has performed substantial work, A
must still pay the agreed remuneration so that
B receives consideration for the work product.
Difference analysis.Article 933 gives both par-
ties a discretionary termination right and then allo-
cates loss compensation. The improvement is not
retrieval but C-stage legal-effect matching: “com-
pensate loss” and “pay agreed remuneration” aredistinct consequences. ExpertCoT succeeds be-
cause it checks the option predicate against the
exact statutory consequence.
C.4 QID 36: Emergency Rescue and
Negotiorum Gestio
Case.Zhang was travelling in a scenic area when
he saw a woman standing alone near a cliff with
an unusual expression. When the woman jumped,
Zhang urgently grabbed her clothing and pulled
her back. During the rescue, Zhang’s camera was
damaged, his arm was scratched, and he advanced
medical, bandaging, food, and lodging expenses.
The woman’s family later expressed gratitude and
said they were willing to compensate lost work
time. Zhang was a medical professional who en-
countered the incident on his way home from work.
Question and options.Given that Zhang spent
time and money on the rescue, may Zhang ask the
woman to pay a certain remuneration?
A. He may request full remuneration.
B. He may not request remuneration.
C.Based on fairness, he may request appropriate
remuneration.
D.He may request remuneration from the
woman’s family.

Pipeline Pred. P-Provision F-Fact C-Conclusion
Naive RAG + CoT CD Finds Article 933 on discre-
tionary termination and loss
compensation.Correctly identifies paid man-
date and work already done.Expands loss compensation
into full agreed remuneration.
G-Retriever + CoT CD Identifies the mandate termi-
nation rule.Uses work-performance facts
too strongly to support D.Overselects D and broadens the
legal effect.
Naive RAG + Expert-
CoTC Separates the termination
right from the compensation
duty.Uses the work facts only to
support loss, not full remuner-
ation.Selects only C.
LEGO Full C Keeps the paid-mandate loss
rule clear.Does not convert substantial
work into a full remuneration
claim.Matches the option predicate
and rejects D.
Table 7: P/F/C comparison for QID 1.
Pipeline Pred. P-Provision F-Fact C-Conclusion
Naive RAG + CoT C Conflates necessary expenses,
appropriate compensation,
and remuneration.Overweights advanced ex-
penses, lost work time, and
family gratitude.Uses fairness to create a remu-
neration claim.
Naive RAG + Expert-
CoTC Uses P/F/C but misses the no-
remuneration legal effect.Treats compensation facts as
a remuneration basis.Selects C.
LEGO CoT B Graph expansion supplies the
negative rule: no remunera-
tion claim.Treats family gratitude as nei-
ther mandate nor public re-
ward.Because the question asks
about remuneration, selects B.
LEGO Full B Separates rescue immunity,
expenses, compensation, and
remuneration.Expense facts do not change
the legal nature of remunera-
tion.Rejects C and D, selects B.
Table 8: P/F/C comparison for QID 36.
Difference analysis.Article 979 allows reim-
bursement of necessary expenses and appropriate
compensation for losses; it does not create remu-
neration. The civil-law graph explicitly states that
the manager has no remuneration claim. LEGO
CoT already succeeds because ExpertGraph pro-
vides this negative legal effect. Plain RAG and
RAG+ExpertCoT fail because they see expense or
compensation facts but miss the legal boundary of
remuneration.
C.5 QID 7: Misdelivered Milk and
Good-Faith Recipient
Case.Because of a delivery worker’s negligence,
milk ordered by Wang was mistakenly placed in
the milk box of Wang’s neighbor Zhang. Zhang
did not understand why the milk was there and,
without checking, directly took it out and discarded
it. The milk package bore Wang’s name, address,
and order number. Wang discovered the missing
milk the next day and demanded compensation
from Zhang. The milk was indeed Wang’s lawful
property.
Question and options.After Zhang discarded
another person’s milk that had been placed in his
milk box, and given that Wang did suffer property
loss and Zhang did discard the milk, how should
Zhang’s conduct be characterized?
A.It constitutes unjust enrichment becauseZhang obtained a benefit without legal basis
and caused loss.
B.It constitutes tort liability, either as intentional
destruction of another’s property or as negli-
gent disposal causing loss.
C. It constitutes unauthorized agency.
D. It involves no legal wrong.
Difference analysis.The gold explanation treats
Zhang’s lack of awareness as decisive. Zhang was
a good-faith recipient; the milk no longer existed;
no present benefit remained. The graph lists unjust-
enrichment elements and the good-faith recipient’s
present-benefit limitation. This case demonstrates
synergy: ExpertCoT alone is pulled to A by the gen-
eral unjust-enrichment rule, while graph-alone/free
CoT drifts to B by over-reading negligence. LEGO
Full combines the exception and the fact filter.
C.6 QID 16: Bank Overpayment and
Business Profit
Case.When Party A withdrew money from a
bank, a bank employee mistakenly overpaid A by
10,000 yuan because of a counting error. A knew
the extra money was not owed, but used the 10,000
yuan as capital for business, quickly bought goods
and resold them, and made a profit of 5,000 yuan,

Pipeline Pred. P-Provision F-Fact C-Conclusion
Naive RAG + Expert-
CoTA Uses only the general
unjust-enrichment elements
and misses the good-faith
limitation.Emphasizes name, address,
and Wang’s loss while weak-
ening Zhang’s good faith.Incorrectly finds unjust enrich-
ment.
G-Retriever + Expert-
CoTB Shifts to tort fault without sta-
bilizing the good-faith recipi-
ent rule.Treats failure to check as suf-
ficient legal fault.Incorrectly finds tort liability.
LEGO CoT B Rejects unjust enrichment but
drifts to tort.Still treats non-checking as
enough fault.Selects B.
LEGO Full D Checks both the general
unjust-enrichment rule and
the good-faith exception.Centers good faith, no re-
tained benefit, and no inten-
tional destruction.Rejects A and B, selects D.
Table 9: P/F/C comparison for QID 7.
of which 2,000 yuan was cost for labor and man-
agement. After discovering a cash shortage, the
bank traced the flow of funds and confirmed that
the profit was directly generated from the overpaid
money. One month later the bank discovered the
overpayment and asked A to return the principal
and corresponding gains. A refused.
Question and options.Given that A used the
bank’s funds for profit and that the bank suffered
an interest loss, which statement is correct?
A.A does not need to return anything because
the problem was entirely caused by the bank’s
own mistake.
B.A should return the overpaid 10,000 yuan,
with no other liability.
C.A should return the overpaid 10,000 yuan and
one month’s interest.
D.A should return the overpaid 10,000 yuan, one
month’s interest, and the entire 3,000 yuan net
profit generated by using the money.
Difference analysis.The gold explanation gives
the bank principal plus one month’s interest. The
interest is the legal fruit of money; downstream
business profit is not awarded to the bank under the
original exam’s private return claim. Wrong traces
have the right broad domain, unjust enrichment,
but over-extend the legal consequence. LEGO Full
succeeds by checking whether option D’s conse-
quence is strictly authorized by the controlling rule
and the benchmark explanation.
Doctrinal caveat.The civil-law graph contains
a broad note that returnable benefit may include
gains obtained by using the original object. Some
modern unjust-enrichment analyses may therefore
support a broader answer. The benchmark gold
follows the original exam explanation; this caseshould be used to illustrate legal-effect boundary
control, not a universal profit-return rule.
C.7 Cross-Case Lessons
Across the six cases, the observed failures are
mostly P/F/C mismatches rather than answer-
format errors. Retrieval helps when the missing
major premise is the bottleneck (QID 19). Expert-
CoT helps when a retrieved rule must be mapped
to the exact option predicate (QID 1). ExpertGraph
helps when the missing information is a negative
legal effect or exception (QID 36). LEGO Full is
strongest when both are needed: the graph supplies
the right rule neighborhood, and P/F/C prevents the
C step from expanding one legal consequence into
another (QID 7 and QID 16).
D Dataset Card
LawExamQA_Civil

Pipeline Pred. P-Provision F-Fact C-Conclusion
G-Retriever + Expert-
CoTD Uses an overbroad rule requir-
ing return of all gains from
the object.Overweights that the business
profit was produced by the
overpaid money.Awards the business profit to
the bank.
LEGO CoT D Expands damages under Arti-
cle 987 into full return of net
profit.Underweights labor, manage-
ment, and business interven-
tion.Selects D.
Naive RAG + Expert-
CoTC Limits the return scope to
principal and legal fruits.Distinguishes the bank’s use-
of-money loss from business
profit.Selects C.
LEGO Full C Uses Articles 122 and 985 for
return, without expanding Ar-
ticle 987 to D.Treats principal and interest
as the private return scope.Rejects D, selects C.
Table 10: P/F/C comparison for QID 16.
Statistic Value
Total items 723
Question type distribution
Single-select 365
Multi-select 254
True/False 104
Subjective (converted to options) 26
Hop count distribution
1-hop 342
2-hop 215
3-hop 104
≥4-hop 62
Subject area distribution
Contracts 291
Property 167
Torts 84
Marriage & Family 48
Succession 25
Other 108
Avg. tokens 158.06
Avg. option tokens 14.82
V ocabulary Hit Rate (median) 0.095
Table 11:LawExamQA_Civil Dataset Card.Detailed
statistics for the 723-item release; the 26 subjective ques-
tions converted to options form an overlapping subset.
We construct LawExamQA_Civil from publicly
available Chinese National Judicial Examination
questions in the civil -law domain. The dataset
contains both objective multiple -choice items and
subjective questions. To enable unified automatic
evaluation, subjective questions are converted into
option -style items while preserving their original
legal-reasoning requirements.
For each question, we use the accompanying offi-
cial explanation to identify the statutory provisions
involved in the reasoning chain. The number of
distinct provisions cited or required by the explana-
tion is treated as thehop count, providing a direct
measure of statutory reasoning complexity.
LexRAG_CivilStatistic Value
Total items 140
Avg. question chars 21.0
Table 12:LexRAG_Civil Dataset Card.Statistics for
the LexRAG_Civil subset used in our cross-benchmark
evaluation.
LexRAG (Li et al., 2025b) is the first benchmark
targeting RAG systems in multi-turn legal consulta-
tion, containing 1,013 expert-annotated five-round
dialogues paired with a 17,228-article candidate
pool; each generated response is scored by an
LLM-as-judge along five rubrics—factuality,user
satisfaction,clarity,logical coherence, andcom-
pleteness—with an expert-response anchor of 8/10 .
LexRAG-Civil is the civil-law subset obtained by
filtering LexRAG for civil-law items and randomly
sampling.
PLawBench_Civil
Statistic Value
Total items 114
Avg. question chars 67.4
Table 13:PLawBench_Civil Dataset Card.Statis-
tics for the PLawBench_Civil subset used in our cross-
benchmark evaluation.
PLawBench (Shi et al., 2026) is a rubric-based
benchmark with ∼850 expert-curated questions
across 13 practical legal scenarios, spanning three
task families (public legal consultation, practi-
cal case analysis, legal document generation) and
grounded in ∼12,500 fine-grained expert rubric
items. We use itsPractical Case Analysissec-
tion, where each item is structured into four
components—Fact,Law/Provision,Reasoning,
andConclusion—and each component is scored
against dedicated rubrics; PLawBench-Civil is the
civil-law subset obtained by filtering and randomly
sampling.
Judge models and scoring protocol.For
LexRAG, we used Qwen3-30B-A3B as the eval-

uator and followed the original LexRAG five-
dimensional LLM-as-a-judge rubric and scoring
prompt. The same evaluator configuration and
prompt were applied to all compared systems. For
PLawBench, we used Gemini-3.0-Pro-Preview as
the evaluator and followed the scoring protocol
specified in the original PLawBench paper. We
did not conduct a human-agreement calibration of
either judge, the reported margins on these two
benchmarks should therefore be read as evidence
that LEGO is competitive with the strongest RAG
baselines under each benchmark’s native protocol,
not as a calibrated measure of superiority.
E LegalExpert-Domain Graph
Construction
Expert-annotated legal graph
Expert annotators were compensated at fair mar-
ket rates. They were licensed legal professionals
who had passed the national judicial examination,
graduated from reputable law schools, and were
paid at the standard hourly rate for legally trained
research assistants at the authors’ institution.
The following protocol reproduces the complete
instructions provided to annotators. Annotators
were instructed to identify the applicable Civil
Code provisions and doctrinal concepts; annotate
constitutive elements, factual conditions, excep-
tions, defenses, and legal effects; connect them
using the predefined relation schema; consult ju-
dicial interpretations and representative cases only
when statutory provisions were underspecified; use
mainstream doctrinal views as the default while
recording disputed alternatives in notes; and re-
solve disagreements through expert discussion. No
task-related risks beyond those ordinarily associ-
ated with scholarly legal annotation were antici-
pated.
1.Node extraction.Chinese Civil Code parsed
into article nodes with metadata (Part, Chapter,
Section, Article number, full text); legal-concept
nodes were extracted from a curated civil-law
taxonomy and linked to the relevant articles.
2.Edge annotation.Our expert -curated legal
graph organizes all civil code concepts in a hi-
erarchical structure (Part →Chapter→Section
→Article). Each node is annotated with its def-
inition, constitutive elements, comparisons with
related provisions, and temporally ordered legal
conditions and consequences. Beyond struc-tural organization, we further label legally oper-
ative relations among concepts, including order
of application, rule priority, hierarchical depen-
dencies, prerequisites, statutory exceptions, and
other doctrinal links, each accompanied by a
brief legal justification.
Civil Law ExpertGraph Construction
To construct the Civil Law ExpertGraph, we fol-
low a hierarchical legal interpretation process in-
spired by source validity and doctrinal legal reason-
ing. The goal is not to annotate benchmark-specific
answers, but to build a general-purpose normative
graph that captures the structure of PRC civil law.
The graph encodes legal concepts, statutory pro-
visions, constitutive elements, exceptions, priority
rules, condition–consequence relations, and legal
effects. These annotations are constructed indepen-
dently of LawExamQA_Civil questions, answer
labels, and official rationales.
Hierarchical Legal Interpretation Frame-
work
Civil-law knowledge is organized according
to the validity and interpretive function of legal
sources. We use the following source hierarchy:
Civil Code Articles →Judicial Interpretations →
Guiding / Reference Cases →Civil-law Textbooks
and Academic Doctrines.
Civil Code articles provide the primary statutory
basis. Judicial interpretations clarify the scope and
application of statutory provisions. Guiding and
reference cases illustrate how abstract norms oper-
ate in concrete factual settings. Civil-law textbooks
and academic doctrines are used to refine concep-
tual boundaries, resolve doctrinal ambiguities, and
standardize expert annotations.
Correspondingly, different interpretive methods
are applied at different stages. Literal interpretation
is used to extract concepts and elements directly
from statutory text. Systematic interpretation is
used to locate each provision within the structure of
the Civil Code and to identify relations among pro-
visions. Purposive and sociological interpretation
are used to clarify how rules function in practice.
Doctrinal and comparative interpretation are used
to refine disputed or abstract civil-law concepts.
Annotation Process
Stage One: Article-level extraction through
literal interpretation.
Annotators first examine Civil Code provisions
and extract basic legal units from the statutory text.
These units include legal concepts, subjects, ob-

jects, constitutive elements, legal acts, conditions,
exceptions, and legal effects. For example, a provi-
sion on contract validity may be decomposed into
nodes such as legal act, expression of intent, ca-
pacity, legality, validity, invalidity, and revocability.
At this stage, the annotation remains close to the
statutory language and avoids adding case-specific
reasoning.
Stage Two: Systematic interpretation within
the Civil Code.
Annotators then situate each provision within
the broader structure of the Civil Code, includ-
ing Book, Chapter, Section, and neighboring pro-
visions. This stage identifies inter-provision re-
lations such as general–special rule, prerequisite,
limitation, exception, priority, parallel application,
and condition–consequence relation. For example,
rules on contract formation, contract validity, real-
right transfer, liability allocation, and remedies are
linked according to their doctrinal order of applica-
tion rather than surface textual similarity.
Stage Three: Refinement using judicial inter-
pretations and representative cases.
Where statutory provisions are abstract or un-
derspecified, annotators consult judicial interpreta-
tions and representative cases to clarify the prac-
tical scope of legal elements. This stage is used
to refine borderline concepts, exceptions, defenses,
and legal consequences. For example, judicial in-
terpretations may clarify when a third party is pro-
tected, when a contract-related remedy is available,
or when a general civil-law rule is displaced by a
more specific rule.
Stage Four: Doctrinal consolidation through
civil-law theory.
Finally, annotators consult civil-law textbooks
and academic doctrines to standardize concept
boundaries and relation types. This stage is es-
pecially important for distinguishing doctrinally
close concepts, such as validity versus effective-
ness, contract obligation versus real-right transfer,
termination versus rescission, liability for breach
versus tort liability, and invalidity versus revocabil-
ity. When multiple doctrinal views exist, annotators
record the mainstream view as the default graph
relation and preserve disputed views as notes or
alternative annotations.
Quality Control
To ensure consistency, all annotations follow
a predefined schema of node types and edge
types. Node types include legal concept, statu-
tory provision, constitutive element, factual condi-tion, exception, defense, and legal effect. Edge
types include hierarchy, prerequisite, specifica-
tion, exception, priority, parallel application, con-
dition–consequence, and remedy relation. Dis-
agreements are resolved through expert discussion,
with emphasis on whether the annotated relation
reflects a general civil-law structure rather than a
benchmark-specific answer path.
Independence from Evaluation Data
The Civil Law ExpertGraph is constructed in-
dependently from benchmark questions, answer
labels, options, and official explanations. Public
civil-law examination questions are used only for
evaluating whether the graph-enhanced retrieval
and reasoning framework improves multi-provision
legal reasoning. They are not used as annotation
sources for graph nodes or edges. Thus, the graph
provides general normative signals for retrieval and
reasoning, rather than item-level supervision.
Node types and error-oriented semantic links.
Our Civil Law Legal Knowledge Graph encodes
expert legal knowledge through typed nodes and se-
mantic links. These links are represented by parent–
child structures, sibling structures, and explicit se-
mantic annotations such as prerequisite, distinction,
exception, priority, and legal effect.
In this design,Conceptnodes guide the model
to the correct legal institution;Elementnodes force
prerequisite checking;Distinctionnodes prevent
confusion between similar doctrines;Exception
nodes expose provisos and defenses;Effectnodes
constrain the interpretation of legal consequences;
andprioritynodes impose the proper order of legal
reasoning.
Figure 6: Semantic Cloud of legal knowledge graph

Figure 4: Legal Concept Tree of Civillaw KnowledgeGraph

Figure 5: Legal relation in legal concept

Node
TypeFunction in the
GraphTargeted Error Types
Concept Identifies the legal
domain or institution,
e.g., contract, tort lia-
bility, property right,
agency, marital prop-
erty.Legal Concept Confla-
tion; Incorrect Provi-
sion Selection; Subordi-
nation Confusion.
Element Decomposes a rule
into required legal
elements or prereq-
uisites, e.g., valid
claim, manifestation
of intent, registration,
damage, causation.Prerequisite Violation;
Fact Subsumption; Rea-
soning Consistency; In-
correct Provision Selec-
tion.
Distinction Marks easily con-
fused doctrines, e.g.,
voidness vs. void-
ability, termination
vs. revocation, prop-
erty right vs. creditor
right.Legal Concept Con-
flation; Legal Effect
Misinterpretation; In-
correct Provision Selec-
tion.
Effect Encodes legal con-
sequences after
rule application,
e.g., restitution,
damages, revocation,
invalidity, specific
performance.Legal Effect Misinter-
pretation; Reasoning
Consistency; Fact Sub-
sumption.
Exception Represents statutory
provisos, defenses,
and limiting condi-
tions, e.g., good-
faith third party, lim-
itation period, force
majeure.Exception Bypass; In-
correct Provision Selec-
tion; Legal Effect Mis-
interpretation.
priority Specifies the norma-
tive order of legal
analysis, e.g., rela-
tionship before claim
basis, prerequisites
before effects, spe-
cial rules before gen-
eral rules.Priority Misappli-
cation; Prerequisite
Violation; Subordi-
nation Confusion;
Reasoning Consis-
tency.
Table 14: Node types in the Civil Law Legal Knowledge
Graph and their corresponding error targets.
Figure 7: legal relation types of civillaw knowledge-
graph

Level H1 H2 H3 H4 H5 H6 H7 H8 H9
Count 1 10 44 162 558 1,490 2,682 2,155 1,109
Table 15: Heading-depth distribution of the Expert-
Graph source corpus.
F Civil Law ExpertGraph: Source
Statistics
This appendix describes the structured Markdown
corpus from which the Civil Law ExpertGraph
is constructed. These figures characterise the up-
stream source and are not directly comparable with
the deployed-graph statistics, because the down-
stream construction pipeline changes the unit of
analysis through rule-unit typing, vocabulary nor-
malisation, and article linking.
Overall.The source is a single hierarchical Mark-
down document of roughly 391K characters over
17,190 lines (8,717 non-empty), organised as a
heading tree of nine depth levels. Its 8,211 heading
nodes—one document title, ten book- or major-
section headings, and 8,200 chapter-level and
deeper nodes—form the candidate knowledge units
consumed by the construction pipeline. The corpus
resolves to 217 distinct Civil Code articles span-
ning the full range of the code (articles 1–1260).
Heading-tree depth.Heading depth is the pri-
mary structural cue. As Table 15 shows, the mass
of the tree sits at H6–H9, where the corpus records
legal-effect statements, exception clauses, and pre-
requisite enumerations—the material the pipeline
maps onto normative edges.
Major-section breakdown.Table 16 reports
heading counts for the ten top-level sections: a
preface on the civil-law system, the seven books
of the Civil Code, a supplementary chapter on in-
tellectual property, and a notes section. Contracts,
General Provisions, and Property Rights together
account for roughly 71.8% of all hierarchical head-
ing nodes, reflecting the doctrinal density of these
three books.
Priority annotations.The corpus carries 357 in-
textPriority: N annotations marking doctrinally
salient nodes (208 at Priority 1, 149 at Priority 2).
These These annotations are recorded in the source
corpus.
High-frequency legal concepts.Table 17 reports
frequencies for 22 core civil-law concepts. The dis-
tribution makes the doctrinal focus explicit: the cor-
pus is anchored on contract, creditor’s right, succes-Section Headings (H3+)
Civil Law System 44
Book I General Provisions 1,912
Book II Property Rights 1,615
Book III Contracts 2,362
Book IV Personality Rights 376
Book V Marriage and Family 490
Book VI Succession 410
Book VII Tort Liability 607
Ch. 17 Intellectual Property 378
Notes 6
Total 8,200
Table 16: Per-section structural footprint of the source
corpus.
Term Freq. Term Freq.
Contract 1,514 Formation 261
Creditor’s right 713 Damages 260
Succession 708 Rescission 204
Property right 568 Termination 202
Agency 502 V oid 196
Tort 385 Take effect 180
Security 365 Extinction 174
Mortgage 359 Constitutive element 143
Validity 351 Limitation period 121
Transfer 276 Claim right 113
Lien 106
Defence 83
Table 17: Top-22 legal concept frequencies. Counts
are over the original Chinese terms; English glosses are
given for readability.
sion, property right, agency, and tort, with substan-
tial coverage of the machinery connecting them—
validity, transfer, formation, damages, rescission,
termination, and the validity-modality vocabulary.
The presence of procedural and defensive vocabu-
lary (limitation period, claim right, lien, defence)
indicates that the source captures not only first-
order legal concepts but the operational apparatus
used in reasoning.
From source to deployed graph.The corpus
above is the structured input to the construction
pipeline. The deployed ExpertGraph is produced
by typing each candidate heading node (concept
/ element / effect / exception-defence / relation /
article), normalising vocabulary across hierarchi-
cal levels, resolving article references against the
Civil Code text, and adding the normative edges—
general/special, principle/exception, prerequisite/-
consequence, and related types—used to route
queries to rule cards. The figures in this appendix
therefore characterise the corpus from which the
graph is built, not the graph itself.
G Detailed Error Analysis
Table 18 reports the primary error labels for the
430 items mis-answered by the full LEGO setting.
We code each item by the earliest reasoning step
that makes the final answer unrecoverable. This

Figure 8:Error-type distributionon items mis-
answered by full LEGO. These seven failure modes mo-
tivate the Syllogistic-CoT design choices in Section 3.2.
convention is important because legal reasoning
errors often cascade: a wrong provision may later
produce an apparent priority conflict, or a missed
prerequisite may later look like an erroneous legal
effect. The analysis below therefore focuses on
root causes rather than surface symptoms.
Figure 8 summarises the error-type distribution.
Most failures occur before final answer selection:
incorrect provision selection (36.74%) and pre-
requisite violation (20.70%) together account for
57.44% of all errors. The top seven categories ac-
count for 91.4%, indicating that wrong answers are
concentrated in a small set of structural reasoning
failures rather than diffuse factual uncertainty.
Incorrect Provision Selection.This is the largest
category, covering 158 errors (36.74%). These
cases usually arise when the model anchors the fact
pattern to provisions that are lexically or topically
similar but legally inapplicable. For example, a
question involving carrier liability may trigger a
nearby contractual damages provision while omit-
ting the special transportation rule that controls the
case. This error shows that retrieval is not merely
a semantic matching problem: the selected provi-
sion must occupy the correct position in the legal
topology and must be licensed by the facts.
Prerequisite Violation.Prerequisite violations
account for 89 errors (20.70%). In these cases,
the model applies a downstream legal consequence
before verifying an upstream legal state, such as dis-
cussing breach liability before establishing contract
validity, or assigning tort liability before checking
the required causal relation. These errors directly
motivate explicit prerequisite gates in Syllogistic-
CoT: before deriving a conclusion Q, the model
must verify every necessary premise Piencoded
by prerequisite edges.Priority Misapplication.Priority misapplication
appears in 42 errors (9.77%). The model often cites
both a general rule and a special rule but applies
them in the wrong order, or treats them as parallel
rather than hierarchical. Such errors are especially
common when civil law provisions contain nested
exceptions, special chapters, or mandatory rules
that override default contractual autonomy. This
category motivates an explicit priority-resolution
step before answer commitment.
Legal Concept Conflation.Legal concept con-
flation covers 33 errors (7.67%). These errors occur
when the model merges formally distinct concepts,
such as prerequisite versus trigger, presumption
versus legal fiction, or solidarity versus supplemen-
tary liability. Although the generated explanation
may sound legally plausible, the logical operator is
wrong, which changes the validity of the inference.
This supports our use of the 22-concept taxonomy
as typed reasoning operations rather than loose tex-
tual labels.
Exception Bypass.Exception bypass accounts
for 30 errors (6.98%). Here the model correctly
identifies a general rule but fails to test whether
an exception defeats it. This is a non-monotonic
reasoning failure: adding an exception-bearing fact
can reverse a conclusion that would otherwise fol-
low. This motivates the exception-checking micro-
constraint in ExpertCoT (§3.2), which requires the
model to test the retrieved provisos and defences
before accepting any conclusion derived from a
general rule.
Reasoning Consistency.Reasoning consistency
errors cover 21 cases (4.88%). These include con-
tradictions between intermediate conclusions and
the final answer, switching parties mid-chain, or
using one interpretation in the explanation and an-
other in option selection. These errors motivate a
final trace-consistency check that verifies whether
the selected option is actually entailed by the stated
premises.
Subordination Confusion.Subordination con-
fusion appears in 20 errors (4.65%). The model
mistakes dependent legal relations for independent
ones, such as treating accessory obligations, deriva-
tive rights, or supplementary liabilities as if they
could exist without their principal legal relation.
This category shows why the graph must encode
structural dependence, not only topical relatedness
among provisions.

Error Type Count %
Incorrect Provision Selection 158 36.74
Prerequisite Violation 89 20.70
Priority Misapplication 42 9.77
Legal Concept Conflation 33 7.67
Exception Bypass 30 6.98
Reasoning Consistency 21 4.88
Subordination Confusion 20 4.65
Legal Effect Misinterpretation 16 3.72
Fact Subsumption 15 3.49
Others 6 1.40
Table 18:Error-type distributionon items mis-
answered by full LEGO. These failure modes motivate
the Syllogistic-CoT design choices in Section 3.2.
Legal Effect Misinterpretation and Fact Sub-
sumption.The remaining substantive categories
are legal effect misinterpretation (16 errors, 3.72%)
and fact subsumption (15 errors, 3.49%). The for-
mer involves deriving the wrong consequence from
an otherwise correct rule, while the latter involves
mapping facts to the wrong constitutive elements.
These are smaller but still important categories be-
cause they occur after provision retrieval succeeds,
indicating that closed-set legal reasoning still re-
quires careful rule application.
Design Implications.The distribution suggests
that Syllogistic-CoT should be organized around
four safeguards. First, a provision-grounding step
reduces incorrect provision selection by forcing
the model to state the controlling rule as the ma-
jor premise. Second, a prerequisite-verification
step blocks downstream conclusions until all nec-
essary legal states are established. Third, a priority-
and-exception step handles special rules, manda-
tory rules, and defeaters before the final inference.
Fourth, a consistency check aligns the intermediate
legal trace with the selected answer. Together, these
checks target the concentrated failure mass: the top
seven categories cover 91.4% of all observed er-
rors.
H Multi-Hop Case Studies on
LawExamQA-Civil
This appendix complements the aggregate error-
type taxonomy of Appendix G. There we coded
LEGO’s 430mis-answered items by failure mode;
here we examine the opposite population—items
on which LEGO answers correctly while every one
of13strong baselines fails—to identify the spe-
cific reasoning patterns that drive LEGO’s hop-
resilience.QID Hop Gold LEGO-unique
627 1-hop ABD✓
716 2-hop ABD✓
544 3-hop CD✓
22 4+-hop A✓
Table 19: Four LawExamQA-Civil cases on which
LEGO is correct andall 13baselines (GPT-5, DeepSeek-
V3, Qwen3-30B-A3B, GLM-4.7-Flash, Qwen3-8B,
GLM-4-9B-chat, DISC-LawLLM, LegalOne, Naive
RAG, HippoRAG 2, RAPTOR, G-Retriever, LightRAG)
are wrong.
Selection.We isolate the 20items on which
LEGO is correct while all 13baselines are wrong,
and select four cases spanning the four hop strata
and four distinct reasoning failure patterns. For
each case we report the case facts, the question, the
answer distribution across the 14systems, and the
controlling provisions.
H.1 QID 7 (1-hop): Good-Faith Recipient
Exception to Unjust Enrichment and Tort
Case.Because of a delivery worker’s negligence,
milk ordered by Wang was mistakenly placed in the
milk box of Wang’s neighbour Zhang. Zhang did
not understand why the milk was there and, without
checking, took it out and discarded it. The pack-
age bore Wang’s name, address and order number.
Wang discovered the missing milk the next day and
demanded compensation from Zhang. (Here we
compare it with the 13 external baselines.)
Question.After Zhang discarded another per-
son’s milk that had been placed in his milk box,
and given that Wang did suffer a property loss, how
should Zhang’s conduct be characterised?
Options.
•A.It constitutes unjust enrichment, because
Zhang obtained a benefit without legal basis
and caused a loss.
•B.It constitutes a tort, either as intentional de-
struction of another’s property or as negligent
disposal causing loss.
•C.It constitutes unauthorised agency.
•D.It involves no legal wrong.
Gold.D. Civil Code Art.122(unjust enrichment).
Zhang did not know, and had no reason to know,
that the milk was not his; he retained no benefit,
because the milk no longer exists; and he neither
intended nor foresaw harm to Wang. Under the

good-faith-recipient limitation, a recipient who is
unaware of the lack of legal basis and whose benefit
has ceased to exist owes no restitution, so neither
unjust enrichment nor tort liability arises.
Answer distribution.
Answer Systems
B GPT-5, DeepSeek-V3, GLM-4.7-Flash,
Qwen3-8B, GLM-4-9B-chat, DISC-
LawLLM, LegalOne, G-Retriever,
LightRAG
AB Naive RAG, HippoRAG 2, RAPTOR
A Qwen3-30B-A3B
D LEGO✓
Baseline failure mode.Every baseline attaches
liability to Zhang. Nine choose tort alone (B), one
chooses unjust enrichment alone (A), and three
RAG baselines choose both (AB). The facts sup-
ply everything needed to trigger a general liability
frame—another person’s property, a loss, and an
act by Zhang—and the baselines stop there, without
testing whether the good-faith-recipient limitation
removes liability.
LEGO’s contribution.LEGO’s retrieved
set contains the unjust-enrichment provisions
(Arts. 122, 985–988), and its conclusion tests
the general rule against the good-faith limitation
before judging the options (translated from the
Chinese output):
Art. 987 does not apply, because he was unaware
and the benefit has ceased to exist; Art. 986 ap-
plies, so Zhang bears no duty of restitution. . . . A
is incorrect: Zhang obtained no benefit, so there
is no unjust enrichment. B is incorrect: Zhang
did not intentionally destroy the property. . . . D
is correct: Zhang’s conduct constitutes neither a
tort nor unjust enrichment.
Trace caveat.LEGO rejects option B only on the
ground that there was no intentional destruction; it
does not separately address the negligent-disposal
branch of B.
Pattern.EXCEPTION THAT DEFEATS A GEN-
ERAL LIABILITY RULE: a 1-hop item can still
require testing whether a statutory limitation de-
feats the general rule that the facts appear to trigger.
Here the general unjust-enrichment and tort frames
are displaced by the good-faith-recipient limitation,
which the ExpertGraph represents as an exception
to restitution.
H.2 QID 716 (2-hop): Threshold-Gated
Mental-Distress Claim
Case.A and B are neighbours. A’s house leaks
chronically, flooding the corridor B uses daily. Brepeatedly demands repair; A refuses, and later
publicly mocks B (“you can’t even walk past a
puddle”). B develops anxiety symptoms (medi-
cal record on file); the village committee fronts
RMB 1,000 to clear the water.
Question.Among the following claims, which
are correct?
Options.
•A.B may demand removal of nuisance.
•B.B may demand elimination of the hazard.
•C.B may demand a public apology, because
the prolonged flooding caused obvious mental
pressure.
•D.B may demand reimbursement of the
RMB 1,000.
Gold. ABD. Civil Code Arts.236(real-right pro-
tection: removal of nuisance / hazard elimination)
and980(necessary expenses incurred by a third
party may be recovered). Option C requires the
additional threshold ofseriousmental harm under
Art. 1183; the facts do not meet that bar.
Answer distribution.
Answer Systems
ABCD GPT-5, DeepSeek-V3, Qwen3-30B-A3B, GLM-
4.7-Flash, Qwen3-8B, DISC-LawLLM, Naive
RAG, HippoRAG 2, RAPTOR, G-Retriever,
LightRAG
ABC GLM-4-9B-chat
AB LegalOne
ABD LEGO
Baseline failure mode.Eleven of thirteen base-
lines pickeverything—ABCD—failing to test op-
tion C against the constitutive-element gate of
Art. 1183. “Obvious mental pressure” (the word-
ing in option C) is below the statute’s threshold of
“serious mental harm”; granting an apology remedy
without checking this element is a classicalexcep-
tion bypass(cf. Appendix G, error type “Exception
Bypass”).
LEGO’s contribution.LEGO explicitly verifies
the Art. 1183 threshold before accepting C, and
rejects it: “B’s claim of prolonged mental pressure
does not, on the facts, satisfy the ‘serious mental
harm’ requirement of Art. 1183, so the apology
remedy is not available even though the underlying
tort is established.” Options A, B, and D each map
cleanly onto a controlling provision (Art. 236 for
A and B; Art. 980 for D), so LEGO accepts them.

Pattern.THRESHOLD-GATED REMEDY: when
a remedy depends on satisfying a constitutive el-
ement (here, “serious” mental harm), explicitly
check the element against the facts before granting
the remedy. The Syllogistic-CoT major-premise
expansion makes the threshold visible; baselines
that skip the check end up over-claiming.
H.3 QID 544 (3-hop): Joint Tort vs. Separate
Tort with Divisibility
Case.A and B are neighbours, each keeping one
goat. The two goats are known to roam together.
One morning both pens are left unlocked; the goats
together eat C’s rare medicinal herbs bare. The
damage scene shows interleaved hoofprints and
mixed eating marks; the share consumed by each
goat cannot be distinguished. A and B had a verbal
agreement to take turns supervising, but neither
honoured it on the day.
Question.Regarding A’s and B’s liability:
Options.
•A.A and B may each escape liability by proving
they discharged the duty of care.
•B.Under Civil Code Art. 1168, A and B
bear joint and several liability for the jointly-
committed tort.
•C.If the quantity eaten by each goat can be de-
termined, A and B bear corresponding several
liability.
•D.If the quantity cannot be determined, A and B
bear liability in equal shares.
Gold. CD. Civil Code Arts.1171, 1172, 1245.
The two goat-owners did not jointly commit a tort
under Art. 1168; their tortious acts areseparate,
with damage that happens to combine (Art. 1172).
When the divisible share can be proved, each bears
proportional liability; when it cannot, the default is
equal apportionment.
Answer distribution.
Answer Systems
BCD GPT-5, DeepSeek-V3, Qwen3-8B, LegalOne,
Naive RAG, HippoRAG 2, RAPTOR, G-
Retriever, LightRAG
BD Qwen3-30B-A3B, GLM-4-9B-chat
BC GLM-4.7-Flash
D DISC-LawLLM
CD LEGO
Baseline failure mode.Nine of thirteen base-
lines pick BCD: they retain options C and D (thecorrect divisibility logic under Art. 1172) butalso
pick B (joint tort under Art. 1168). This is inter-
nally inconsistent—Art. 1168’s joint liability is in-
compatible with Art. 1172’s proportional liability—
but baselines do not detect the conflict because they
retrieve and apply each option’s apparent statute
independently. The error reflects what Appendix G
labelspriority misapplication: when a general rule
(joint tort) and a special rule (separate-act tort with
divisibility) both surface in retrieval, baselines treat
them as parallel rather than hierarchical.
LEGO’s contribution.LEGO’s response explic-
itly distinguishes three statutes: Art. 1168 (joint
tort, requires concerted action), Art. 1171 (concur-
rent dangerous acts, each act alone sufficient to
cause the entire damage), and Art. 1172 (separate-
act tort, damage combines). It then applies the facts
to Art. 1172 specifically (acts are separate; damage
is combined and only conditionally divisible), and
rejects Art. 1168 because the verbal supervision
agreement was not actually executed—there is no
concerted action. The conclusion CD follows from
Art. 1172’s two-branch structure (provable share
→proportional; unprovable→equal).
Pattern.GENERAL-VS-SPECIAL SELECTION
AMONG TORT REGIMES: ExpertGraph’s normative
edges (joint vs. separate vs. concurrent-dangerous)
gate which liability rule applies, preventing the
silent merger of Art. 1168 with Art. 1172 that traps
every baseline.
H.4 QID 22 (4+-hop): Independent Mortgage
Transfer (Art. 406 vs. Art. 407)
Case.D holds a mortgage over C’s house. To
secure a friend’s bank loan, D signs a written agree-
ment transferringthe mortgage right alone(not
the underlying debt) to the bank. The transfer is
duly registered; the registry lists the bank as the
new mortgagee. The bank, relying on the registry’s
public-faith principle, accepts the mortgage as se-
curity and disburses the loan.
Question.Given that the bank has completed reg-
istration and relied in good faith, is the mortgage-
transfer contract unlawful?
Options.
•A.Yes.
•B.No — the mortgage is a freely transfer-
able property right; registration completes the

property-right transfer, and the bank has lawfully
acquired the mortgage.
•C.No — effective upon notice to the debtor.
•D.No — effective upon C’s consent.
Gold. A. Civil Code Arts.153, 216, 217, 407.
Art. 407 is the controlling special rule: a mort-
gage right cannot be transferred independently of
the underlying debt; the general transferability and
registration-effect rules (Arts. 216, 217) are over-
ridden in this configuration.
Answer distribution.
Answer Systems
B All13baselines: GPT-5, DeepSeek-V3, Qwen3-
30B-A3B, GLM-4.7-Flash, Qwen3-8B, GLM-
4-9B-chat, DISC-LawLLM, LegalOne, Naive
RAG, HippoRAG 2, RAPTOR, G-Retriever,
LightRAG
A LEGO
This is the most striking unanimous-baseline-
failure case in the dataset: every closed-model base-
line (including GPT-5 and DeepSeek-V3), every
law-tuned LLM, and every RAG baseline picks B.
Baseline failure mode.Baselines apply the gen-
eral rule (Arts. 216/217: registration completes
real-property right transfer; public-faith protection
for good-faith reliance on the registry) without
checking whether a more specific rule constrains
the rule’s domain. Art. 407, located in the mortgage
chapter, explicitly removesindependentmortgage
transfers from the general transferability regime.
The error is a textbookpriority misapplication: the
general rule is correctly identified but the special
rule that overrides it is silently omitted.
LEGO’s contribution.LEGO’s response makes
the rule hierarchy explicit: “Art. 406 is the gen-
eral rule (transferability of the mortgaged property
with the mortgage following); Art. 407 is the spe-
cial rule (the mortgage right cannot be separated
from the secured debt); the special rule prevails.”
Applied to the facts (only the mortgage was trans-
ferred; the secured debt was not), the contract is
unlawful and the registration cannot cure the sub-
stantive invalidity. Note that the public-faith princi-
ple (Art. 216) protectsproceduralreliance on the
registry, not the substantive validity of an act pro-
hibited by Art. 407—this nested distinction is what
every baseline misses.
Pattern.GENERAL-VS-SPECIAL RULE RESOLU-
TION AT THE4-HOP LEVEL: the case requirescomposing four provisions (153 invalidity, 216
public faith, 217 registration effect, 407 mortgage-
debt inseparability) and resolving their priority.
Without an explicit rule-hierarchy step—which
Syllogistic-CoT enforces and ExpertGraph sup-
ports with normative edges—retrieval that surfaces
only Arts. 216/217 yields a confident but wrong
answer.
H.5 Cross-Case Synthesis
The four cases map onto four reasoning failure pat-
terns that ExpertGraph and Syllogistic-CoT are
specifically designed to address. Three of the
four also appear among the top error types in
Appendix G’s aggregate taxonomy (the analysis
of items on which LEGOfails), confirming that
LEGO’s wins and losses share a common skeleton:
success requires explicit rule-hierarchy resolution
and constitutive-element checking, while failure
occurs when those steps are skipped or applied to
the wrong rule.
Reasoning pattern Case Aggregate-
taxonomy ana-
logue
Right-holder disambiguation
among topically related par-
tiesQID 174 Legal Concept Con-
flation
Threshold-gated remedy
(constitutive-element check
before grant)QID 716 Exception Bypass
General-vs-special selection
among tort regimesQID 544 Priority Misapplica-
tion
General-vs-special rule reso-
lution at4-hopQID 22 Priority Misapplica-
tion / Incorrect Pro-
vision Selection
Two observations follow. First, the unanimous-
baseline-failure pattern on QID 22 is not noise:
Art. 407 is doctrinally required but lexically remote
from the case facts, so similarity-driven retrieval
does not surface it, whereas rule-card routing does.
Second, the 4+-hop margin of +6.4 pp over the
strongest LLM baseline is not driven by retrieving
more provisions, but by retrieving the rightconfig-
urationof provisions and resolving their hierarchy.
The aggregate hop-resilience of LEGO (Table 1)
is therefore a population-level signature of the per-
case mechanisms documented in this appendix.

Configuration 1h 2h 3h 4+h All
Qwen3-8B (zero-shot) 25.4 27.4 26.9 16.1 25.4
Qwen3-8B + CoT 28.9 27.0 31.7 33.9 29.2
Qwen3-8B + IRAC-CoT 30.1 34.4 26.0 33.9 31.1
Naive RAG + CoT 34.2 33.0 35.6 37.1 34.3
Naive RAG + IRAC-CoT 32.5 31.2 37.5 25.8 32.2
G-Retriever + CoT 34.2 34.0 30.8 35.5 33.7
G-Retriever + IRAC-CoT 33.3 34.9 27.9 38.7 33.5
Naive RAG + ExpertCoT 33.3 40.5 34.6 35.5 35.8
G-Retriever + ExpertCoT 34.8 38.1 32.7 35.5 35.5
ExpertGraph + CoT 36.5 33.5 37.5 33.9 35.5
ExpertGraph + IRAC-CoT 36.5 39.1 35.6 38.7 37.3
Full system (LEGO) 41.8 40.5 37.5 38.7 40.5
Table 20: Accuracy (%) on LawExamQA-Civil by hop
count for twelve component configurations. Single-
component upgrades give modest gains; combining Ex-
pertGraph with ExpertCoT yields a gain.
I Component-Synergy Case Studies on
LawExamQA-Civil
This appendix isolates the internal contribution of
the two LEGO components — the ExpertGraph re-
trieval module and the ExpertCoT reasoning mod-
ule — by examining a 5×3 ablation grid and the
case-level behaviour at its extremes. It comple-
ments Appendix H (cases where LEGO beats ex-
ternal baselines) and Appendix G (aggregate error
taxonomy on LEGO failures): together the three
appendices document, respectively, what LEGO
adds over baselines, what LEGO adds over its own
ablations, and where LEGO still fails.
Setup.We instantiate twelve configurations
crossing five retrieval modules (none, Naive RAG,
G-Retriever, ExpertGraph, Full) with three reason-
ing modules (plain CoT, IRAC-CoT, ExpertCoT).
Twelve of the fifteen cells were run; the missing
three are not informative for our hypothesis.
Accuracy matrix.Table 20 reports per-
configuration accuracy on each hop stratum. Three
patterns are visible. First, swapping Naive RAG for
ExpertGraph (with the same plain CoT) yields a
small +1.2 pp gain on the aggregate but a +2.3 pp
gain on 1-hop. Second, swapping plain CoT for
ExpertCoT (with the same Naive RAG) yields a
+1.5 pp gain on the aggregate but a +7.5 pp gain
on2-hop. Third, combining both yields +6.2 pp
on the aggregate and +7.6 pp on 1-hop — more
than the sum of the two single-component gains.
Synergy decomposition.Taking Naive RAG +
plain CoT as the baseline ( 34.30% ), we isolate the
contribution of each LEGO component and their
interaction:Hop Items Full-unique-correct
1-hop 342 5
2-hop 215 6
3-hop 104 1
4+-hop 62 2
Total 723 14
Table 21: Items on which the Full LEGO system is
correct and every one of the eleven ablations is incorrect.
•ExpertGraph retrieval alone (replacing Naive
RAG, keeping plain CoT):35.55%,+1.24pp.
•ExpertCoT alone (keeping Naive RAG, replacing
plain CoT):35.82%,+1.52pp.
• Both together (Full):40.53%,+6.22pp.
•Synergy:6.22−(1.24 + 1.52) =+3.46pp.
The interaction is positive in point estimate, but its
95% interval includes zero (Appendix L).
Case selection.We then identify items where the
Full system answers correctly whileevery one of
the eleven ablations(including the strong Expert-
Graph + IRAC-CoT variant and the two Expert-
Graph alternatives) is incorrect. Across the 723
items there are 14such cases (Table 21). We ex-
pand three of them spanning the 2/3/4+-hop strata.
I.1 QID 237 (2-hop): Constitutum
Possessorium
Case.A owns a jade worth RMB 10,000. A and
B sign a sale contract for RMB 11,000, “effective
upon signature”; B is to take delivery in three days.
After signing, A asks to keep the jade for personal
enjoyment a few days more; B agrees but no sepa-
rate property-transfer document is signed. The jade
remains in A’s physical custody. The next day A
re-sells and delivers the jade to a knowing buyer C,
and the chain continues to D, E (a finder after loss),
F.
Question.Given that the sale contract is effective
and B has confirmed ownership at signature, which
statements about B’s acquisition of ownership and
delivery are correct?
Options.
•A.B acquires ownership upon contract validity.
•B.B acquires ownership when the borrowing
agreement takes effect.
•C.Because there is no actual delivery, ownership
has not transferred.
•D.A delivered the jade to B by means ofconsti-
tutum possessorium.

Gold. BD. Civil Code Arts.224(general rule:
ownership of movables transfers upon delivery) and
228(constitutum possessorium: when the parties
agree the transferor continues to possess, owner-
ship transfers at the moment the agreement takes
effect). Art. 228 is the controllingspecialrule.
Ablation behaviour.The agreement to “borrow
back” the jade after the sale is the textbookconsti-
tutum possessoriumpattern that activates Art. 228;
eight of eleven ablations select CD — correctly
rejecting A butalsorejecting B and accepting C.
They thus apply the general delivery rule (Art. 224)
and miss the special rule. Even the strongest ab-
lation, ExpertGraph + IRAC-CoT, retrieves both
Arts. 228 and 224 but still answers CD: retrieval
is sufficient but reasoning fails to resolve their hi-
erarchy. Only the Full system (ExpertGraph + Ex-
pertCoT) explicitly invokes the general-vs-special
priority and selects BD.
Per-configuration outcomes.
Configuration Pred OK
Qwen3-8B (zero-shot) AD✗
Qwen3-8B + CoT CD✗
Qwen3-8B + IRAC-CoT CD✗
Naive RAG + CoT C✗
Naive RAG + IRAC-CoT CD✗
G-Retriever + CoT CD✗
G-Retriever + IRAC-CoT CD✗
Naive RAG + ExpertCoT CD✗
G-Retriever + ExpertCoT CD✗
ExpertGraph + CoT C✗
ExpertGraph + IRAC-CoT CD✗
Full (LEGO) BD✓
Full-system trace (excerpt).
Art. 228 is the special rule and overrides
Art. 224’s general delivery requirement. Since
A and B agreed that A would continue to possess
the jade after the sale, ownership transfers at the
moment of the agreement (B), not upon physical
delivery (rejecting C). The delivery itself takes the
form ofconstitutum possessorium(D).
What the ablation isolates.ExpertGraph +
IRAC-CoT receives both articles but does not ap-
ply their general–special priority; the Full system’s
P–F–C prompt, which asks for provision relations
explicitly, produces the correct ordering.
I.2 QID 509 (3-hop): Title-Retention Sale plus
Good-Faith Third Party
Case.Zhou sells a computer to Wu under a title-
retention contract for RMB 6,000 in five monthly
instalments of RMB 1,200; until full payment, titleremains with Zhou. Wu pays the first four instal-
ments on time but defaults on the fifth. The com-
puter malfunctions; Wu hands it to Zhou for repair.
After repair Zhou, citing Wu’s breach, re-sells the
computer to Wang for RMB 6,200; Wang checks
Zhou’s original purchase receipt and accepts in
good faith.
Question.Based on the title-retention rules,
which statements are correct?
Options.
•A.Wang may acquire ownership of the computer.
•B.Because title remains with Zhou, Zhou may
exercise the right of recovery when Wu cannot
pay the final instalment.
•C.If Wu’s unpaid due amount reaches
RMB 1,800, Zhou may demand the full purchase
price.
•D.If Wu’s unpaid due amount reaches
RMB 1,800, Zhou may rescind the contract and
demand a use fee for the computer.
Gold. ACD. Civil Code Arts.311(good-faith ac-
quisition),634(one-fifth default-acceleration rule
in instalment sales),642(seller’s right of recovery
under title retention, conditional on notice and rea-
sonable period). The case requires composing three
rules: B is incorrect because the right of recovery
requiresnotice plus reasonable period(Art. 642),
not just inability to pay; A is correct via Art. 311
(Wang’s good-faith reliance on the receipt); C and
D both follow from Art. 634 (1/5 threshold of
RMB 1,200 is exceeded by RMB 1,800).
Ablation behaviour.The ablations fragment
across the four options because each lacks one of
the three doctrinal pieces:
•Six of eleven ablations selectAD(good-faith
acquisition + rescission), missing Art. 634’s full-
price branch (C).
•Naive RAG + ExpertCoT selectsBCD, retaining
the wrong recovery-right framing (B).
•Two G-Retriever variants selectAB(mixing
good-faith acquisition with the unconditional re-
covery framing).
ExpertGraph + IRAC-CoT retrieves Arts. 642 and
634 but still answers AD; the Full system is the only
configuration that retrieves the same articlesand
composes them into the three-rule chain (Art. 311
for A; Art. 634 for both C and D, distinguishing its
two branches; Art. 642 as the gate that rejects B).

Per-configuration outcomes.
Configuration Pred OK
Qwen3-8B (zero-shot) ABCD✗
Qwen3-8B + CoT A✗
Qwen3-8B + IRAC-CoT AD✗
Naive RAG + CoT AD✗
Naive RAG + IRAC-CoT AD✗
G-Retriever + CoT AD✗
G-Retriever + IRAC-CoT AB✗
Naive RAG + ExpertCoT BCD✗
G-Retriever + ExpertCoT AB✗
ExpertGraph + CoT AD✗
ExpertGraph + IRAC-CoT AD✗
Full (LEGO) ACD✓
What the ablation isolates.This case demon-
strates the limit of single-component upgrades.
With ExpertGraph retrieval alone, the model has ac-
cess to Art. 634 but does not unpack its two-branch
structure (full-price accelerationvs.rescission-
plus-use-fee); with ExpertCoT alone over Naive
retrieval, the model develops the right reasoning
skeleton but misses the controlling articles. The
Full system both retrieves and decomposes, recov-
ering all three of A, C, and D.
I.3 QID 652 (4+-hop): Will Validity vs.
Mandatory Reservation for a Fetus
Case.Zhou and Wu, married for years and child-
less, signed an artificial-insemination consent at
a hospital; Wu became pregnant. In April 2021,
Zhou (now hospitalised with cancer) wrote and
signed a handwritten holographic will: “All hous-
ing purchased after marriage is inherited by my
parents.” The will is signed and dated, read aloud
to the parents, and otherwise satisfies every form
requirement of Civil Code Art. 1134. Zhou died in
May 2021; Wu gave birth in October 2021. Zhou’s
parents immediately listed the house for sale.
Question.Given that the will reflects Zhou’s gen-
uine intent and is formally complete, which of the
following statements areincorrect?
Options.
•A.Because artificial insemination raises ethical
issues, the AI consent is void as contrary to pub-
lic order and good morals.
•B.Being a formally complete and genuinely
intended holographic will, Zhou’s will is fully
valid.
•C.Because Wu’s child was conceived by artifi-
cial insemination rather than naturally, the child
is not a marital child of Zhou and Wu.•D.When dividing Zhou’s estate, an inheritance
share must be reserved for the fetus in Wu’s
womb.
Gold. ABC. Civil Code Arts.153(juristic-act
validity),1071(parent-child relationship covers
AI children),1153(inheritance division),1155
(mandatory share reservation for a fetus). State-
ment D is the only correct one; A, B, and C are each
separately wrong. The non-obvious wrong is B: a
will may be formally completeandsubstantively
over-broad if it disposes of estate without reserving
the share that Art. 1155 makes mandatory for a
conceived but unborn heir.
Ablation behaviour.All eleven ablations fail to
selectB: ten answerACand Naive RAG + IRAC-
CoT answersACD. They correctly reject A (AI
consent is not contra public order) and C (AI chil-
dren are marital children under Art. 1071) but ac-
cept B: they treat “formally complete” as a suffi-
cient condition for “fully valid,” silently overlook-
ing the mandatory-reservation requirement under
Art. 1155. The error pattern isexception bypass
(cf. Appendix G): even after retrieving the form-
validity provisions, the ablations fail to test the will
against the substantive limit imposed by Art. 1155
on every disposition of an estate when a heir is
in gestation. The Full system, which receives the
same ExpertGraph retrieval as the two ExpertGraph
ablations, is the only configuration that applies
Art. 1155 to option B.
Per-configuration outcomes.
Configuration Pred OK
Qwen3-8B (zero-shot) AC✗
Qwen3-8B + CoT AC✗
Qwen3-8B + IRAC-CoT AC✗
Naive RAG + CoT AC✗
Naive RAG + IRAC-CoT ACD✗
G-Retriever + CoT AC✗
G-Retriever + IRAC-CoT AC✗
Naive RAG + ExpertCoT AC✗
G-Retriever + ExpertCoT AC✗
ExpertGraph + CoT AC✗
ExpertGraph + IRAC-CoT AC✗
Full (LEGO) ABC✓
Full-system trace (excerpt).
Art. 1134 governs the formal validity of a holo-
graphic will; Art. 1155 is the special rule that re-
quires a reservation for an in-gestation heir. The
two are not mutually exclusive: a will may pass
formal validity (under Art. 1134) but fail substan-
tive validity because it disposes of the entire estate
without reserving the mandatory fetus share. The
disposition to Zhou’s parents leaves nothing for
Wu’s unborn child, violating Art. 1155.

Case ExpertGraph
suppliesExpertCoT
suppliesJoint effect
QID 237
(2-hop)Both Arts. 224 and
228 in the retrieval
setGeneral-vs-
special priority
resolutionSelects BD instead
of CD
QID 509
(3-hop)Arts. 311, 634,
642 in a single
retrievalTwo-branch
decomposition of
Art. 634Selects ACD
instead of AD
QID 652
(4+-hop)Art. 1155
alongside
form-validity
articlesSubstantive-limit
check on a
formally valid actSelects ABC
instead of AC
What the ablation isolates.Both components
are necessary on this case. ExpertGraph alone
(with plain CoT or IRAC-CoT) receives the same
provisions, including Art. 1155, but does not apply
it as a substantive limit on the will’s scope; Expert-
CoT alone (with Naive or G-Retriever retrieval)
supplies the constitutive-element checking frame-
work but lacks the article. Only the Full system
applies Art. 1155 through the element check that
connects “unborn fetus” to the reservation require-
ment.
I.4 Cross-Case Synthesis
The three ablation cases instantiate a common pat-
tern: each component supplies a necessary but
insufficient capability, and the Full system’s win
arises from their interaction rather than from either
alone.

ID Label LEGO Best B.∆
civil-107 Family/Marriage 90.9 81.8 +9.1
civil-682 Legal-theory Appl. 50.7 30.7 +20.0
civil-56 Individual Life 78.6 92.9 –14.3
civil-633 Cross-border Affairs 80.0 58.3 +21.7
Table 22: Four PLawBench-Civil case studies. civil-
56 is included as a partial win: LEGO ranks second
among three methods that retrieve the controlling statute,
contrasting with three baselines that retrieve none.
J PLawBench-Civil Case Studies
This appendix expands the PLawBench-Civil row
of Table 2 into case-level analyses. PLawBench-
Civil scores each response on four rubric compo-
nents (LAW, FACT, REASONING, CONCLUSION)
and a composite Overall score; component scores
are rationals in [0,1] and the Overall is reported
in[0,100] . We pick four cases spanning the four
largest labels, including one partial win (civil-56)
where LEGO trails the best baseline.
Selection.We enumerate all 114 samples, rank
byOverall LEGO−max bOverall b, and pick the
four cases below to span the four largest dataset la-
bels (family/marriage,individual life,legal-theory
application,cross-border affairs) and four distinct
baseline failure modes.
J.1 civil-107 (Family/Marriage): Joint Marital
Debt under Long Separation
Question.Do debts incurred by one spouse in
their own name constitute joint marital debt?A
divorcing husband claims six personal-name debts
totalling over RMB 1.7M (bank loans, supplier
debts, family-friend loans, online lending) are joint
marital debts because they were used for “house-
hold consumption, business turnover, rent, medical,
child education.” The wife denies knowledge of
every debt. The spouses had been separated since
2017.
Reference.Notjoint debt. The required anal-
ysis integrates “debt formation time, nature, use,
household-necessity test, joint-business test, joint
intent, and separation status.” Controlling provi-
sions: Civil Code Art.1064(three categories of
joint debt) and Art. 1089 (post-divorce repayment).
Per-method scores.MethodL F R COverall
Naive RAG 0.50 1.00 1.00 0 81.8
HippoRAG 2 0.50 0.75 0.50 0 54.5
RAPTOR 0.50 0.75 0.75 1 72.7
G-Retriever 0.50 1.00 0.75 0 72.7
LightRAG 0.50 1.00 0.75 0 72.7
LEGO 0.50 1.00 1.00 1 90.9
Baseline failure mode.Four of five baselines re-
treat into a non-committal conditional conclusion
(“partially joint, partially not” / “may constitute
joint debt subject to further review”). The PLaw-
Bench judge marks all four as conclusion-incorrect
because the reference requires a directional judg-
ment (notjoint debt) qualified by the Art. 1064
but-clause.
LEGO’s contribution.LEGO’s conclusion mir-
rors the reference’s logical form (negation + excep-
tion clause):
The debts incurred by Zhang in his own name do
not constitute joint marital debt, except for those
debts that can be proven to have been used for
joint marital life, joint production and operation,
or to be based on joint marital intent.
In the reasoning section, LEGO is the only method
that explicitly assigns the burden of proof to the
creditor and connects this to the long-separation
fact, aligning with the reference’sactori incumbit
probatiorequirement under Art. 1064 §2. Naive
RAG cites the correct Supplementary Interpreta-
tion Art. 35 but dilutes the same content into a
conditional sentence, costing the conclusion point.
Takeaway.On bivalent legal judgments with a
but-clause, baseline RAG models often hedge;
LEGO commits to the directional answer while pre-
serving the exception. This is the cleanest LEGO
win in the PLawBench dataset.
J.2 civil-682 (Legal-theory Application):
Lease Continuity after
Partnership-to-LLC Conversion
Question.A general partnership (“Hangzhou
Qiantang Metal Products Factory”) leases a factory
from Z at RMB 100k/month. The three partners
later convert the partnership into a limited-liability
company (“Qiantang Metal Products Co., Ltd.”)
with the same core trade name; on May 2 the LLC’s
legal representative declares: “Qiantang Co., Ltd.
ratifies all transactions previously signed by A and
B.” The LLC pays rent on time but Z now wants to
reclaim the factory. May Z reclaim it?

Reference.Z cannot reclaim. The LLC’s posses-
sion isrightful possession. Required provisions:
Civil Code Arts.235(return claim against wrong-
ful possession),462(possessor’s return right),733
(lease return on expiry),967(definition of part-
nership contract), and551(debt transfer requires
creditor consent).
Per-method scores.
MethodL F R COverall
Naive RAG 0.00 0.75 0.00 0.40 22.7
HippoRAG 2 0.25 0.75 0.00 0.60 30.7
RAPTOR 0.00 0.70 0.00 0.40 21.3
G-Retriever 0.00 0.70 0.00 0.40 21.3
LightRAG 0.00 0.75 0.00 0.40 22.7
LEGO 0.25 1.00 0.33 0.60 50.7
Baseline failure mode.All six methods correctly
conclude “Z cannot reclaim,” so conclusion polar-
ity is shared. The discriminator is doctrinal fram-
ing. Five of six baselines collapse the argument
onto Civil Code Art. 75 (acts performed during
legal-person setup are inherited by the legal per-
son). This anchors the case in asetup-actdoctrine,
but the reference requires thepartnership-contract
conversiondoctrine. Failure-mode variants: Ligh-
tRAG invokes Arts. 96/98 to argue the partnership
and the LLC are distinct subjects, then must re-
verse itself to accept the ratification, producing
an internally contradictory argument; RAPTOR re-
trieves Art. 547 (assignment of claims) and Art. 985
(unjust enrichment), both topical mismatches; G-
Retriever retrieves Art. 141 (withdrawal of intent),
unrelated.
LEGO’s contribution.LEGO is the only
method that retrieves Art. 967—one of the five
reference provisions—anchoring the analysis in
the partnership-contract doctrine rather than the
generic setup-act doctrine. It additionally cites
Art. 75 (setup-act inheritance) and Art. 703 (lease
form), producing a three-tier argument:partner-
ship contract →conversion + ratification →on-
going lease. LEGO is also the only method
whose case-facts section preserves the pivotal da-
tum “Qiantang Co., Ltd. has paid rent on time,”
achieving Fact = 1.00 where baselines score 0.70–
0.75.
Takeaway.This case illustrates retrieval-
precision benefits beyond conclusion correctness.
LEGO finds asmall but pivotalprovision (Art. 967)
that anchors the correct doctrinal frame. Baselines
retrieve a generally relevant but doctrinallyoff-target provision (Art. 75) and amplify its
prominence at the cost of the controlling rule.
J.3 civil-56 (Individual Life): Self-help in a
Neighbour Dispute
Question.Plaintiff smashed 2.5 m of his neigh-
bour’s stone railing with a hammer, claiming the
railing blocked ventilation and light; he was ad-
ministratively detained for 8 days. Plaintiff argues
the act was lawful self-help. Does the self-help
defence stand?
Reference.Self-help does not stand. Controlling
provision: Civil Code Art.1177(self-help requires
urgency,necessity, andimmediate post-act notifi-
cation of state authorities).
Per-method retrieval and scores.
Method Core articles cited 1177? Overall
Naive RAG 1165 / 296 / Public Security 49✗50.0
HippoRAG 21177+ 1165 + 293✓78.6
RAPTOR 1165 / 288 / 296✗50.0
G-Retriever 233 / 235 / Criminal 275✗50.0
LightRAG1177+ 1184✓92.9
LEGO 1177+ 238 + Public Security 49✓78.6
Baseline failure mode.All six methods reach
the correct conclusion (self-help fails). The dis-
criminator is whether Art. 1177—theonlycontrol-
ling statute—is retrieved. Three of five baselines
miss it entirely and instead retrieve general-tort
(Art. 1165), neighbour-relations (Arts. 288/296),
property-protection (Arts. 233/235), or even Crim-
inal Code Art. 275. These three baselines com-
pensate by writing a generic “tort plus neighbour”
argument that the judge marks as off-rubric, cap-
ping Overall at 50.
LEGO’s contribution.Once Art. 1177 is re-
trieved, LEGO is the only method besides the ref-
erence itself thatexplicitly decomposes the article
into three sub-elements—urgency, necessity, im-
mediate notification—and checks each against the
facts:
Self-help requires three conditions: “urgent cir-
cumstances, ” “inability to obtain timely protec-
tion from state authorities, ” and “failure to act
would cause irreparable damage. ” . . . The con-
duct occurred during daytime, and Y had already
reported to the police . . . so it does not satisfy “ur-
gent circumstances” or “inability to obtain timely
state-authority protection. ”
This element-by-element structure mirrors the ref-
erence reasoning’s three subsections. Baselines
that retrieve Art. 1177 (HippoRAG 2, LightRAG)
achieve comparable or slightly higher Overall on

this single case via a more terse statute applica-
tion; LEGO’s structured element-check, while not
the absolute best on this individual case, is the
form that the PLawBench rubricgenerallyrewards
across the rest of the benchmark.
Takeaway.When the controlling statute is reach-
able by single-shot retrieval, LEGO is one of three
methods that find it. LEGO’s distinctive contribu-
tion is then theelement-by-element checkgrounded
in the major-premise structure of the syllogism
(Section 3.2), not simply locating the article.
J.4 civil-633 (Cross-border Affairs): Validity
of Cross-border Marriage Brokerage
Question.A Chinese plaintiff pays
RMB 168,000 to an intermediary for a “Viet-
namese bride” arrangement; the bride disappears
one month after the marriage registration. The
intermediary refunds only RMB 18,000. Is the
brokerage valid? Must the intermediary refund the
remainder?
Reference.The brokerage isinvalid(violates
Civil Code Art.1042ban on marriage-trade and
the State Council 1994 Notice on cross-border
marriage brokerage). The intermediary must re-
fund partially, since the plaintiff bears co-fault as
a legally competent adult. Controlling provisions:
Civil Code Arts.1042,153,157.
Statute-retrieval pattern.
Method Marriage-prohibition article 157? Overall
Naive RAG1048✗(consanguinity, wrong topic)✓58.3
HippoRAG 21048✗ ✓50.0
RAPTOR1048✗ ✓41.7
G-Retriever1048✗ ✗50.0
LightRAG1042✓ ✗41.7
LEGO 1042✓ ✓80.0
Baseline failure mode.Four of five baselines re-
trieve Civil Code Art.1048(prohibition of consan-
guineous marriage), which istopically adjacent(lo-
cated in the same chapter as Art. 1042) butdoctri-
nally unrelatedto the case. This is a textbook near-
neighbour retrieval miss: the embedding space con-
fuses two provisions about “marriage prohibitions”
of fundamentally different kinds. LightRAG re-
trieves the correct Art. 1042 but does not compose
it with the invalidity-effects chain (Art. 157), and
so its overall reasoning chain is incomplete.
LEGO’s contribution.LEGO is the only
method that simultaneously retrieves the correctArt. 1042andthe correct Art. 157 (effects of in-
valid juristic acts including fault-based loss alloca-
tion), and additionally cites Art. 964 (intermediary
not entitled to fee for failed brokerage). The com-
position produces a three-step invalidity argument—
prohibition →invalidity →restitution with fault
allocation—that no baseline assembles. Reason-
ing component score is 0.75 versus ≤0.50 for all
baselines; Law component is 0.75 versus 0.50 for
baselines. The +21.7 Overall margin is the largest
LEGO win in the PLawBench dataset and isolates
two reinforcing mechanisms:
•Near-neighbour avoidance.ExpertGraph’s
normative edges distinguish “prohibition by
marriage-trade” from “prohibition by consan-
guinity,” allowing LEGO to bypass the lexical
attraction to Art. 1048 that traps four baselines.
•Multi-statute composition.Syllogistic CoT ex-
plicitly chains Arts. 1042 →157→964 along
an invalidity-restitution path, rather than treating
each article as an independent citation.
J.5 Cross-Case Synthesis for
PLawBench-Civil
The four cases isolate four distinct baseline failure
modes and the corresponding LEGO mechanisms:
Baseline failure mode Example
Hedged conditional conclusion
on bivalent questioncivil-107
Doctrinally off-target anchor
statutecivil-682
Missing controlling statute un-
der noisy distractorscivil-56
Near-neighbour statute confu-
sion (Art. 1048 vs. 1042)civil-633
Across these cases, LEGO’s gains arise from
(i) normative-edge retrieval that distinguishes lex-
ically similar but doctrinally distinct provisions,
(ii) element-by-element decomposition of the
major-premise statute, and (iii) multi-statute com-
position along invalidity, restitution, and fault-
allocation chains. 65
K LexRAG_Civil Case Studies
This appendix expands the LexRAG_Civil row of
Table 2 into case-level analyses. Unlike PLaw-
Bench’s case-analysis format, LexRAG asks short
consultation questions in the context of a multi-
turn dialogue history, and judges responses on five
dimensions (Factuality, Satisfaction, Clarity, Coher-
ence, Completeness) plus Overall, each in [1,10]

ID Category LEGO Best B.∆
972_turn2 Real-property Dispute 8 4 (LightRAG) +4
173_turn4 Public Security 9 6 (HippoRAG 2) +3
301_turn2 Inheritance 9 7 (HippoRAG 2) +2
639_turn1 Land Dispute 9 6 (G-Retriever) +3
Table 23: Four LexRAG_Civil case studies, drawn from
the top of the LEGO-vs-best-baseline margin distribu-
tion.
Selection.We enumerate all 140 samples, rank
byOverall LEGO−max bOverall b, and pick the
four cases below to span four distinct consultation
categories and four distinct baseline failure modes.
K.1 972_turn2 (Real Property): The
Effect-of-Registration Doctrine
Question (turn 2 of 2).Does failure to obtain
a property certificate affect the transfer of real-
property rights?
Dialogue context.Turn 1 established that the
user is asking about a so-called “triple-receipt
house” (sanlian-dan fang)—a residential property
held under informal documentation rather than a
regular real-estate certificate.
Gold articles.Civil Code Arts.209(registration
effects real-property right transfer) and210(regis-
tration procedures).
Reference answer.The establishment, modifica-
tion, transfer, and extinction of real-property rights
must be registered in accordance with law to take
effect. If the property certificate has not been ob-
tained, the actual transaction can form only a cred-
itor’s right between the parties; no property-right
transfer occurs.
Per-method scores and verdicts on the core legal
question.
MethodO F SVerdict
Naive RAG 3 2 3 “does not affect”—incorrect
RAPTOR 2 2 2 “does not affect”—incorrect
G-Retriever 3 2 2 internally contradictory
HippoRAG 2 2 2 2 “does not affect”—incorrect
LightRAG 4 3 3 “does not affect”—incorrect
LEGO 8 8 8“generally does affect”—correct
Baseline failure mode.All five baselines commit
the same doctrinal error: they conflateregistration
as a condition of validityfor real-property right
transfer (Art. 209, applicable here) withregistra-
tion as a condition for assertability against third
parties(Art. 225, applicable to specific movables
such as vehicles and vessels). By writing “the ab-
sence of registration does not affect the validity of
the property-right transfer, but merely deprives itof third-party assertability,” the baselines apply the
wrong doctrine to real property—a classical text-
book error reflecting embedding-space proximity
between two structurally similar but legally distinct
rules.
LEGO’s contribution.LEGO’s response cor-
rectly invokes Art. 209 andseparates the contract
from the property right:
If a housing sale-purchase contract has been
signed but the transfer-of-title registration has
not yet been completed, then although the con-
tract itself is valid (under Art. 215), the property
right (i.e., ownership of the house) has not actu-
ally been transferred to the buyer. . .
This is exactly the reference’s distinction between
the creditor’s right arising from the contract and
the property-right transfer requiring registration.
LEGO is the only method that anchors the answer
in theprinciple of separation(between the obliga-
tory act and the dispositive act), which is the con-
trolling doctrine here.
K.2 173_turn4 (Public Security): Joint
Liability among Multiple Tortfeasors
Question (turn 4 of 4).I wasn’t the only one in
the fight—do I have to bear all the compensation
alone?
Dialogue context.Turns 1–3 established: a fight
occurred, the user was hit first, the user sustained
injury, and the assistant already explained Art. 234
(criminal assault), Art. 20 (self-defence), Art. 1173
(comparative fault), and Art. 1165 (general tort
liability). The user is now narrowing the question
tomulti-actor liability apportionment.
Gold articles.Civil Code Arts.1168(joint tort
yields joint and several liability) and1170(multi-
person endangerment with possibly identifiable spe-
cific tortfeasor).
Statute retrieval and Overall.
Method Cited Overall
Naive RAG 1174 / 1175 (wrong: third-party / victim fault) 1
RAPTOR 973 (partnership debt!) 2
G-Retriever 178 / 518 (general joint liability) 3
HippoRAG 21170(one of two gold) 6
LightRAG1168(the other gold) 3
LEGO 1170 + 178 (recourse) 9
Baseline failure mode.Three of five baselines
retrievewrongarticles: Naive RAG goes to victim-
fault rules, RAPTOR jumps to partnership liability
(Art. 973 governs commercial partnerships and is ir-
relevant), G-Retriever picks a general joint-liability

article without multi-tortfeasor specificity. Two
methods (HippoRAG 2, LightRAG) each retrieve
one of the two gold articles but cannot compose a
complete answer.
LEGO’s contribution.LEGO’s response (i) di-
rectly answers the user’s framing (“you do not nec-
essarily have to bear it all alone”), (ii) decomposes
Art. 1170 into its two sub-cases—identifiable spe-
cific tortfeasorvs.unidentifiable—and (iii) closes
with therecourse rightunder Art. 178 (a joint
debtor who pays over their share may recover from
co-debtors). LEGO also leverages dialogue his-
tory: turns 1–3 already cited Art. 1173 (compar-
ative fault), so LEGO does not redundantly re-
cite it and instead introduces the new doctrine
(Art. 1170) the user actually needs. On multi-turn
consultations, LEGO’s advantage is partly retrieval
(avoiding wrong-statute traps) and partly conversa-
tional pragmatics (answering the newly-narrowed
question rather than re-deriving the full multi-turn
framework).
K.3 301_turn2 (Inheritance): Separating the
Inheritance Act from Property Transfer
Question (turn 2 of 2).Does subrogated inheri-
tance require transfer-of-title procedures?
Dialogue context.Turn 1 confirmed that the
user is entitled to subrogated inheritance under
Art. 1128 because her mother predeceased her
grandmother.
Gold articles.Civil Code Arts.1124(acceptance
of inheritance) and208(real-property registration).
Reference answer.Two layers must be sepa-
rated: (i) inheritance acceptance is governed by
Art. 1124—silence equals acceptance, no procedu-
ral act required; (ii) the subsequent real-property
registration is governed by Art. 208 andisrequired
for the heir to better exercise the right.
Method outputs and Overall.
Method Answer to the core question O
Naive RAG flat “no” 3
RAPTOR flat “no” 2
G-Retriever hedged “not necessarily” 3
HippoRAG 2 distinguishes via Arts. 1128 + 230 7
LightRAG flat “no” 2
LEGOseparates inheritance act from property transfer9
Baseline failure mode.Three of five baselines
give a flat “no” answer—technically correct for the
inheritanceacceptanceact but misleading because
the user is asking in the context of inheriting areal estate. The user’s practical question isdo I
need to do paperwork?A flat-no answer leaves
her unaware that property-register transfer is still
required for her to exercise ownership.
LEGO’s contribution.LEGO is the only
method besides HippoRAG 2 to separate the two
legal events the question conflates:
Subrogated inheritance itself does not require
transfer-of-title procedures. . . In practice, how-
ever, if the estate involves real property (such as
housing), transfer-of-title procedures are required
to complete the change of ownership. This proce-
dure is needed for the property-right transfer, not
as a legal effect of subrogated inheritance itself.
This mirrors the reference’s two-layer structure,
and explains the Overall margin of +2over Hip-
poRAG 2 and+6over the three flat-no baselines.
K.4 639_turn1 (Land Dispute):
Indefinite-Term Lease of Private Land
Question (turn 1 of N).If I lease my private land
to a tenant and we do not specify a fixed term, what
governs the lease?
Gold articles.Civil Code Arts.704(required
clauses of a lease) and730(termination rights for
indefinite-term leases).
Reference answer.Three layers: (i) Art. 704 re-
quires parties to include a lease term; (ii) under
Art. 730, either party may terminate with reason-
able notice if no term is specified; (iii) practical
advice—sign a supplementary written agreement.
Statute retrieval and Overall.
Method Cited Overall
Naive RAG 705 (20-yr cap) 5
RAPTOR 721 (rent-payment timing) — off-topic 4
G-Retriever 707 (writing requirement) 6
HippoRAG 2 707 + 510 + 734 (renewal) 5
LightRAG 705 (20-yr cap) 5
LEGO 707 + 730 (indefinite-term lease) + 705 9
Baseline failure mode.Each baseline retrieves a
single topically relevant provision but no method
assembles the complete chain that the reference
requires: form requirement, indefinite-term conse-
quence, termination right, and priority renewal for
the existing tenant. RAPTOR’s choice of Art. 721
(rent-payment timing) is a particularly clear re-
trieval failure—the lease chapter was located, but
the wrong article within it was selected.

LEGO’s contribution.LEGO integrates three
provisions (Arts. 707, 730, 705) and frames the
answer as structured advice with explicit action
items: sign a written contract, fix a term of ≤20
years, prioritize the existing tenant on renewal.
The LexRAG UserSatisfaction dimension explic-
itly rewards actionable practical guidance, which
only LEGO provides. When multiple provisions
interact—form rule, default rule, and right-of-first-
refusal—LEGO’s multi-hop retrieval composes
them into a coherent advisory response; baselines
cite fewer provisions.
K.5 Cross-Case Synthesis for LexRAG_Civil
The four LexRAG_Civil cases isolate four LEGO
mechanisms, three of which echo the PLawBench
findings (Appendix J) and one of which is specific
to multi-turn consultation:
Mechanism LexRAG case PLawBench ana-
logue
Near-neighbour doctrinal disam-
biguation (validity-registration
vs. opposability-registration)972_turn2 civil-633 (Art. 1042
vs. 1048)
Avoidance of wrong-statute re-
trieval traps173_turn4 civil-56 (Art. 1177
vs. 1165)
Multi-layer concept disambigua-
tion301_turn2 civil-107 (joint debt
+ exception clause)
Multi-statute composition +
dialogue-history awareness639_turn1,
173_turn4(new on LexRAG)
The mechanisms instantiate the two design
choices of LEGO: ExpertGraph’s normative edges
disambiguate provisions that are lexically close but
doctrinally distinct, and Syllogistic CoT composes
multiple provisions into structured advisory chains
rather than presenting them as parallel citations.
The same two mechanisms account for both the
rubric-based gains on PLawBench-Civil and the
dialogue-quality gains on LexRAG_Civil.

L Statistical Validation of the Main
Results
This appendix reports the paired significance anal-
ysis for the main results in Table 1 and for the
ablation study in Table 3. The analysis is post
hoc: it re-uses the stored per-item predictions and
changes no prediction, score, or reported accuracy.
Point estimates of accuracy differences are com-
puted from the rounded entries of Tables 1 and 3,
whereas confidence intervals and p-values are com-
puted from the per-item predictions; the two may
therefore differ by up to 0.01 pp.
Procedure.All comparisons are paired at the
item level over the n= 723 LawExamQA_Civil
items. For each pair of systems we treat the two
binary per-item correctness vectors as matched ob-
servations and apply a two-sidedexactMcNemar
test, which conditions on the discordant pairs and
so does not rely on a large-sample approximation.
Confidence intervals are unadjusted percentile in-
tervals obtained by resampling items with replace-
ment 10,000 times and recomputing the paired ac-
curacy difference on each resample; the resampling
unit is the item, so pairing is preserved within every
resample. Intervals describe effect-size uncertainty,
while family-wise decisions use Holm-adjusted p-
values.
Comparison families.We treat the system-level
comparisons and the component-level ablations as
two separate families and apply the Holm correc-
tion within each. The two families answer different
questions – whether LEGO improves on externally
published systems, and whether each of its own
components contributes – and pooling them would
penalise both for the size of the other. Family 1 is
the thirteen comparisons of Table 1; family 2 is the
eleven non-full configurations of Table 3.
System-level results.LEGO’s advantage over
all five same-backbone RAG systems survives cor-
rection, with adjusted pbetween 1.5×10−9and
4.4×10−6and intervals well clear of zero. The
margins over the larger general-purpose models do
not: the intervals for Qwen3-30B-A3B, GPT-5 and
DeepSeek-V3 all span zero, and the DeepSeek-V3
margin corresponds to four items out of 723. We
scope the claim accordingly. The comparison with
larger models is not evidence that an 8B backbone
is intrinsically stronger; it indicates that expert-
structured retrieval and syllogistic reasoning supply
domain signal that lets a small open model reachLEGO vs.∆Acc (pp) 95% CIp Holm
Same-backbone RAG systems
Naive RAG11.62 [+8.30,+15.08] 1.5×10−9
LightRAG11.76 [+8.16,+15.35] 5.1×10−9
RAPTOR11.20 [+7.75,+14.66] 8.4×10−9
HippoRAG 210.65 [+7.19,+14.11] 3.4×10−8
G-Retriever8.86 [+5.39,+12.31] 4.4×10−6
Larger general models
Qwen3-30B-A3B3.60 [−0.41,+7.47] 0.27
GPT-53.04 [−1.11,+7.05] 0.33
DeepSeek-V30.55 [−3.46,+4.56] 0.84
Table 24: System-level comparison family: paired ac-
curacy differences between LEGO and other systems
in Table 1, with exact McNemar tests, percentile boot-
strap intervals ( 10,000 item-level resamples), and Holm
correction applied across all thirteen comparisons in the
family. Positive ∆favours LEGO. The eight rows listed
cover every same-backbone RAG system together with
the three closest larger models; the five comparisons not
shown have larger margins.
Configuration∆Acc (pp) 95% CIp Holm
Naive RAG + plain CoT6.23 [+2.49,+9.82] 0.005
ExpertGraphRAG + plain CoT4.98 [+1.38,+8.58] 0.019
Naive RAG + ExpertCoT4.70 [+1.52,+7.88] 0.019
ExpertGraphRAG + IRAC-CoT3.18 [−0.14,+6.50] 0.067
Table 25: Component-level comparison family: accu-
racy drop of each non-full configuration relative to the
full LEGO system, with Holm correction applied across
all eleven non-full configurations of Table 3. Full LEGO
is significantly better than ten of the eleven; the single
exception is the closest configuration, ExpertGraphRAG
+ IRAC-CoT. The four rows listed are the three non-full
cells of the 2×2 crossed design analysed below, plus
that closest configuration.
the accuracy band of models one to two orders of
magnitude larger. The statistically supported claim
is the fixed-backbone one: under a shared Qwen3-
8B reader and a shared Civil Code corpus, LEGO
improves on every RAG pipeline we evaluate.
Component-level results.Every ablation that re-
moves an expert component costs accuracy, and ten
of the eleven differences clear the corrected thresh-
old. The single exception is instructive rather than
damaging: ExpertGraphRAG + IRAC-CoT, the
strongest non-full configuration, is 3.18 pp below
the full system with an interval that just includes
zero ( pHolm = 0.067 ). Once expert-structured re-
trieval is in place, the further benefit of replacing
a strong generic legal prompt with ExpertCoT is
positive in point estimate but not separable from
noise at this sample size.
Interaction between the two modules.Two
questions have to be kept apart: whether both mod-

ules materially contribute, and whether their com-
bination establishes statistical super-additivity. The
crossed ablation supports the first; the second re-
mains inconclusive. On the complete 2×2 subset
{Naive RAG, ExpertGraphRAG } × { plain CoT,
ExpertCoT }, the conditional gains reinforce one an-
other: the ExpertGraphRAG gain rises from +1.25
pp under plain CoT to +4.71 pp under ExpertCoT,
and the ExpertCoT gain rises from +1.52 pp under
Naive RAG to +4.98 pp under ExpertGraphRAG.
The corresponding interaction point estimate is
+3.46 pp. It is estimated from the four paired out-
come vectors over all 723items, not from a small
subset of them, but its jointly paired-bootstrap 95%
interval includes zero ( [−0.97,+7.75] ;10,000 re-
samples). The results therefore support architec-
tural integration and complementary conditional
gains, and we make no formal claim of statistically
established super-additivity.
Neither module is dominant.The crossed ab-
lation also does not support reading Expert-
GraphRAG as a modest retrieval add-on to the
prompt. With ExpertCoT held fixed, replacing Ex-
pertGraphRAG with Naive RAG reduces accuracy
from 40.53% to35.82% (∆ = 4.70 pp,95% CI
[+1.52,+7.88] ,pHolm = 0.019 ). The factorial
average marginal effects are 3.25 pp for Expert-
CoT and 2.97 pp for ExpertGraphRAG; their 0.28
pp difference – algebraically the paired contrast
between the two off-diagonal conditions – is not
distinguishable from zero (paired-bootstrap 95%
CI[−3.60,+4.15] ; exact McNemar p= 0.944 ).
The evidence does not identify either module as
dominant. Functionally, ExpertGraphRAG raises
Recall@8 over the Civil Code corpus from 27.99%
to72.53% , while ExpertCoT structures the appli-
cation of the retrieved provisions.
Discordant-pair counts.Because the exact Mc-
Nemar test conditions on discordant pairs, we re-
port them for the two closest same-backbone com-
parisons. Against G-Retriever, LEGO is correct
where G-Retriever is wrong on 116 items and
wrong where G-Retriever is correct on 52, a net
gain of 64items; against Naive RAG the split is
129against 45, a net gain of 84. These are the
quantities the tests above rest on. They should
not be confused with the twenty cases discussed
in Appendix H, which are a deliberately stringent
fourteen-system intersection – LEGO correct and
all thirteen alternatives wrong – used for qualita-
tive mechanism analysis rather than as a count ofpairwise wins.