# PAR$^2$-RAG: Planned Active Retrieval and Reasoning for Multi-Hop Question Answering

**Authors**: Xingyu Li, Rongguang Wang, Yuying Wang, Mengqing Guo, Chenyang Li, Tao Sheng, Sujith Ravi, Dan Roth

**Published**: 2026-03-30 23:52:54

**PDF URL**: [https://arxiv.org/pdf/2603.29085v1](https://arxiv.org/pdf/2603.29085v1)

## Abstract
Large language models (LLMs) remain brittle on multi-hop question answering (MHQA), where answering requires combining evidence across documents through retrieval and reasoning. Iterative retrieval systems can fail by locking onto an early low-recall trajectory and amplifying downstream errors, while planning-only approaches may produce static query sets that cannot adapt when intermediate evidence changes. We propose \textbf{Planned Active Retrieval and Reasoning RAG (PAR$^2$-RAG)}, a two-stage framework that separates \emph{coverage} from \emph{commitment}. PAR$^2$-RAG first performs breadth-first anchoring to build a high-recall evidence frontier, then applies depth-first refinement with evidence sufficiency control in an iterative loop. Across four MHQA benchmarks, PAR$^2$-RAG consistently outperforms existing state-of-the-art baselines, compared with IRCoT, PAR$^2$-RAG achieves up to \textbf{23.5\%} higher accuracy, with retrieval gains of up to \textbf{10.5\%} in NDCG.

## Full Text


<!-- PDF content starts -->

PAR2-RAG: Planned Active Retrieval and Reasoning for
Multi-Hop Question Answering
Xingyu Li, Rongguang Wang, Yuying Wang, Mengqing Guo,
Chenyang Li, Tao Sheng, Sujith Ravi, Dan Roth
Oracle AI
Abstract
Large language models (LLMs) remain
brittle on multi-hop question answering
(MHQA), where answering requires com-
bining evidence across documents through
retrieval and reasoning. Iterative retrieval
systems can fail by locking onto an early
low-recall trajectory and amplifying down-
stream errors, while planning-only ap-
proaches may produce static query sets that
cannot adapt when intermediate evidence
changes. We proposePlanned Active Re-
trieval and Reasoning RAG (PAR2-
RAG), a two-stage framework that sepa-
ratescoveragefromcommitment. PAR2-
RAG first performs breadth-first anchor-
ing to build a high-recall evidence frontier,
then applies depth-first refinement with ev-
idence sufficiency control in an iterative
loop. Across four MHQA benchmarks,
PAR2-RAG consistently outperforms ex-
isting state-of-the-art baselines, compared
with IRCoT, PAR2-RAG achieves up to
23.5%higher accuracy, with retrieval gains
of up to10.5%in NDCG.
1 Introduction
Recent LLMs have achieved expert-level per-
formance in domains such as mathematics
and coding (OpenAI, 2023; DeepSeek-AI, 2025;
Team and DeepMind, 2024; OpenAI, 2025), yet
they still struggle with multi-hop question an-
swering (MHQA), where systems must gather
evidence from multiple documents and com-
pose those facts into a coherent answer (Ho
et al., 2020; Trivedi et al., 2022; Schnitzler
et al., 2024; Krishna et al., 2025). Retrieval-
Augmented Generation (RAG) and related ap-
proaches were designed to address this gap,
but they still fail frequently in multi-hop set-
tings when evidence coverage is incomplete or
reasoning trajectories drift (Lewis et al., 2020;
Gao et al., 2023; Trivedi et al., 2023).A common failure mode in MHQA isprema-
ture commitment, where retrieval approaches
interleave reasoning with retrieval in a mostly
greedy, depth-first manner (Yao et al., 2023;
Trivedi et al., 2023). When early steps latch
onto a distractor, later hops amplify that er-
ror and recovery becomes difficult. Meanwhile,
planning-heavy methods can produce broad
but static query sets that become misaligned
when intermediate evidence changes (Cheng
et al., 2025; Liu et al., 2025).
This motivates a central question:can we
improve multi-hop robustness by jointly con-
trolling retrieval coverage and the timing of
reasoning commitment?We proposePlanned
Active Retrieval and Reasoning RAG
(PAR2-RAG), an agentic and modular two-
stage evidence search approach that separates
retrieval coveragefromreasoning commitment.
Stage 1 performscoverage anchorto expand
the evidence pool. Stage 2 performstterative
chain refinementwithin that anchored pool to
construct a coherent retrieval evidence chain
with intermediate sub-thoughts. The genera-
tion module then conditions on this structured
evidence, which improves final answer quality
by making synthesis more evidence-grounded.
We evaluate PAR2-RAG on four MHQA
benchmarksusingmultipleanswerandretrieval
quality metrics. PAR2-RAG consistently out-
performs strong training-free baselines on both
retrieval coverage and answer accuracy, with
23.5% gain over IRCoT in accuracy and re-
trieval gains of10.3% in recall and 10.5% in
NDCG.
We make three primary contributions:
•We introduce PAR2-RAG, an agentic
modular two-stage framework for MHQA
through planned active retrieval and
evidence-aware reasoning.
1arXiv:2603.29085v1  [cs.AI]  30 Mar 2026

•We provide empirical study under both
non-reasoning and reasoning settings, in-
cluding reasoning-intensive MHQA bench-
marks, showing consistent gains over
strong training-free baselines.
•We present mechanism-level diagnostics
that connect answer gains to retrieval be-
havior, together with robustness analyses.
2 Related Work
2.1 Retrieval-Augmented Generation
Retrieval-Augmented Generation (RAG) im-
proves factual grounding by coupling gener-
ation with external evidence retrieval (Lewis
et al., 2020; Guu et al., 2020). Follow-up work
improves retrieval quality through dense en-
coders, re-ranking, and retrieval-aware prompt-
ing (Karpukhin et al., 2020; Izacard et al., 2021;
Shi et al., 2023; Izacard et al., 2022; Asai et al.,
2024). However, strong first-hop retrieval does
not by itself solve multi-hop reasoning: sys-
tems can still fail when evidence is incomplete
or poorly composed across steps (Gao et al.,
2023; Tang and Yang, 2024).
2.2 Training-free Multi-Hop QA
Interleaving retrieval with reasoning is a domi-
nant training-free strategy for multi-hop ques-
tion answering (MHQA). ReAct alternates
thought and action to iteratively collect ev-
idence (Yao et al., 2023), and IRCoT inter-
leaves chain-of-thought with retrieval to im-
prove compositional QA performance (Trivedi
et al., 2023). These methods are flexible and
often strong, but retrieval and commitment are
coupled in a single loop. Recent analyses of
multi-hop behavior and retrieval difficulty in-
dicate that errors in early hops can propagate
and degrade later reasoning, especially when
bridge evidence is fragile or missing (Biran
et al., 2024; Zhu et al., 2025).
Other RAG-based approaches improve
MHQA by decomposing questions into sub-
queries and searching broader evidence fron-
tiers before deep reasoning. Decomposition
and structured retrieval can improve coverage,
especially for bridge-fact questions, but purely
static plans can become mismatched when in-
termediateevidenceshiftstheinformationneed.
Recent modular and structured pipelines (e.g.,explicit retrieval-in-context search or retrieval–
inference coupling) make this trade-off be-
tween breadth and commitment control ex-
plicit (Chen et al., 2025; Li et al., 2024; Cheng
et al., 2025; Liu et al., 2025).
3 Methods
Multi-hop QA systems often fail when they
commit to a reasoning path before evidence
coverage is sufficient. This premature commit-
ment can lock retrieval into low-recall trajec-
tories and amplify downstream errors. Our
goal is to address this failure mode by explic-
itly controllingwhenthe system commits to
reasoning andhowretrieval is expanded before-
hand. We therefore proposePlanned Active
Retrieval and Reasoning RAG (PAR2-
RAG), a two-stage framework that separates
coverage acquisitionfromreasoning commit-
ment. The key design principle iscoverage
first, commitment late: first build a broad evi-
dence frontier, then perform controlled chain
refinement with explicit sufficiency checks.
3.1 PAR2-RAG
To present PAR2-RAG, we first formulate the
multi-hop RAG. Let qdenote the input multi-
hop question and D={d1, . . . , d N}the docu-
ment corpus, where each dnis a chunked pas-
sage. A multi-hop RAG process can be viewed
as iterative sub-question planning, retrieval,
and response generation. Within this setting,
PAR2-RAGusesfiveagents: (1)aPlanner P(q)
that generates coverage-oriented sub-queries to
expand the evidence frontier; (2) a Retriever
R(q, D, k )that returns top- kranked passage
evidence Eifor a query qi; (3) a Query For-
mulatorQthat rewrites follow-up queries for
commitment-stage refinement; (4) an Evidence
Sufficiency Controller (ESC) Ethat decides
whether to continue retrieval via Qor stop
and finalize the answer; and (5) Writer Wto
generate the intermediate response or the final
answer with prompt-instructed LLM calls.
Figure 1 provides an overview of PAR2-RAG.
The architecture has two coordinated stages.
Stage 1 (Coverage Anchor) performs breadth-
first evidence expansion: the planner P(q)de-
composes the query into complementary sub-
queries, the retriever Rgathers candidate pas-
sages, and the system merges them into an
anchored context Canchor. Stage 2 (Iterative
2

Figure 1: Overall architecture of PAR2-RAG. Stage 1 (Coverage Anchor) expands evidence breadth to
build Canchor, and Stage 2 (Iterative Chain) performs ESC-gated refinement to either continue targeted
retrieval or stop with the final answer.
Chain) performs depth-first refinement over
this anchored context. At each hop, the writer
Wproduces a step response and the ESC E
decides whether to continue retrieval with a
reformulated query or finalize the answer. This
controlloopmakesretrievaladaptivetomissing
evidence while preventing premature commit-
ment to an incomplete reasoning path.
Stage 1: Coverage AnchoringStage 1
implements theCoverage Anchormodule,
whose objective is to construct a high-recall
evidence frontier before the system commits
to a narrow reasoning path. This stage uses
two agents: a planner P(q)that produces di-
verse decomposed sub-queries, and a retriever–
rerankerRthat executes these queries and
returns ranked, deduplicated passages. Note
that both agents are necessary: decomposed
queries define retrieval intents, but they do
not provide factual grounding by themselves.
The retrieved chunks instantiate an anchor con-
textCanchorgrounded in corpus evidence. This
grounding reduces premature commitment, im-
proves bridge-fact coverage, and provides a
stable initialization for Stage 2 refinement.
Concretely, given q, the planner generates a
diverse set of msub-queries{qi}m
i=1targeting
complementaryaspectsoftheinformationneed.
For each qi, the system retrieves top- kcandi-
dates, reranks, deduplicates, and merges the
retained passages into Canchor. The resulting
anchored context serves as a global evidencefrontier, increasing the likelihood that required
supporting facts are available before deep rea-
soning begins.
Stage 2: Iterative Chain Refinement
Stage 2 corresponds to theIterative Chain
module, which aims to convert broad anchored
evidence into a precise, answer-supporting rea-
soning chain. In contrast to Stage 1, this
stage prioritizes commitment quality: decid-
ing whether current evidence is sufficient for
the next reasoning step, or whether targeted
retrieval still needs refinement. TheIterative
Chainstage uses three complementary agent:
a generatorGto produce step-level responses
from the current context, a query formulator Q
to propose focused follow-up query formulation
when evidence is incomplete, and an ESC E
to decide whether to query reformulation or
answer generation. Considering at hop t, the
generatorGa step response rtfrom the current
context Ct(when t=1, C1is equal to Canchor),
thenEis invoked to decide the next step as:
E(q, r t, Ct)→(action, q⋆
t+1),(1)
where action∈ {CONTINUE,STOP} and a
prompt-instructed LLM makes the decision for
E. IfEoutputs CONTINUE , the system issues
a reformulated query q⋆
t+1and retrieves addi-
tional evidence for Ct+1; ifEoutputs STOP, the
loop terminates and returns the answer.
Algorithm 1 summarizes the implementation
of PAR2-RAG. After stage 1, at each hop t,
3

Algorithm 1PAR2-RAG
Require:Queryq, corpusD, hop budgetH
Ensure:Final answera
1:{q i}m
i=1←P(q)
2:InitializeC anchor
3:foreachq ido
4:E i←R(q i, D, k)
5:C anchor←C anchor∪Ei
6:end for
7:t←1,C 1←C anchor
8:whilet≤Hdo
9:r t←G(q, C t)
10:(action, q⋆
t+1, mt)←E(q, r t, Ct)
11:ifaction=STOPthen
12:break
13:end if
14:E t+1←R(q⋆
t+1, D, k)
15:C t+1←C t∪Et+1
16:end while
17:a←r t
18:returna
Gproduces rt=G(q, Ct), then ESC returns
(action, q⋆
t+1, mt). If action =CONTINUE , the
generated q⋆
t+1fromQinvokesR, and updates
context to Ct+1; otherwise the loop terminates
and returns the latest response as the final
answer. This keeps retrieval adaptive while
explicitly controlling continuation to reduce
premature commitment and off-path reasoning.
3.2 Discussion
PAR2-RAG is designed around a simple control
principle:retrieve broadly before committing
narrowly. Different from RAG methods such
as ReAct and IRCoT, PAR2-RAG separates
coverage expansion from commitment-time re-
finement, so early low-recall evidence is less
likely to lock the system into an off-path tra-
jectory. Comparing to planning-only decom-
position, PAR2-RAG remains adaptive during
refinement through ESC-guided continuation
and query reformulation.
This agentic decomposition also clarifies the
role of each component. Coverage Anchor in-
creases evidence breadth and bridge-fact recall;
Iterative Chain improves local trajectory qual-
ity; ESC decides when additional retrieval is
necessary versus when evidence is sufficient to
stop. Compared with single-module variants,
the combined design offers complementary ben-efits: stronger initial recall, controlled correc-
tion when evidence is incomplete, and more
stable behavior across multi-agents.
4 Experiments
4.1 Experimental Setups
BenchmarksWe evaluate PAR2-RAG on
four benchmarks spanning two categories:
2WikiMultiHopQA (Ho et al., 2020) and
MuSiQue (Trivedi et al., 2022), which empha-
size connected multi-step evidence composition;
and MoreHopQA (Schnitzler et al., 2024) and
FRAMES (Krishna et al., 2025), which fur-
ther stress generative reasoning and end-to-end
retrieval-grounded factuality. For each dataset,
we sample 500 queries from the validation split.
Dataset details are provided in Appendix A.1.
BaselinesTo provide evaluation of PAR2-
RAG, we compare against three-types of
training-free methods: (1)Non-retrievalbase-
lines: Direct Inference and Chain-of-Thought
(CoT) (Wei et al., 2022). (2)Iterative retrieval–
reasoningbaselines: ReAct (Yao et al., 2023)
and IRCoT (Trivedi et al., 2023), which we
treat as strong training-free RAG baselines.
(3)PAR2-RAG module variants: Coverage An-
chor and Iterative Chain, which correspond
to modular components of PAR2-RAG rather
than independent external methods.
Evaluating these two module variants sepa-
rately helps clarifywhyPAR2-RAG works: (a)
it isolates gains from planning-oriented cover-
age expansion versus iterative chain refinement,
(b) it reveals whether errors come from weak
global evidence coverage or weak local chain
updates, and (c) it quantifies the complemen-
tarity between planning and reasoning-chain
control when both modules are combined.
Evaluation MetricsWe evaluate both an-
swer quality and retrieval quality. For an-
swer quality, we use binary correctness judged
against ground truth by OpenAI GPT-5-
mini (Singh et al., 2025). For retrieval qual-
ity, we compare deduplicated retrieved chunk
IDs against ground-truth documents using Re-
call@k, NDCG@k (Jeunen et al., 2024), and
All-Pass (1 only when all required documents
are retrieved). We report query-level averages
and per-required-length diagnostics. Full met-
ric definitions are provided in Appendix A.2.
4

Method MuSiQue 2Wiki MoreHopQA FRAMES Average
Direct Inference 0.204 0.418 0.144 0.294 0.265
CoT 0.338 0.724 0.594 0.631 0.572
ReAct 0.432 0.724 0.740 0.745 0.660
IRCoT 0.498 0.820 0.754 0.726 0.700
Coverage Anchor 0.539 0.835 0.753 0.774 0.725
Iterative Chain 0.516 0.868 0.800 0.788 0.743
PAR2-RAG0.615 0.896 0.826 0.811 0.787
Table 1: Non-reasoning generation answer-quality
results on the considered MHQA benchmarks.
Method MuSiQue 2Wiki MoreHopQA FRAMES Average
Direct Inference 0.428 0.794 0.708 0.739 0.667
CoT 0.440 0.804 0.688 0.745 0.669
ReAct 0.575 0.864 0.832 0.833 0.776
IRCoT 0.590 0.830 0.742 0.754 0.729
Coverage Anchor 0.6270.9120.860 0.8600.815
Iterative Chain 0.628 0.884 0.830 0.834 0.794
PAR2-RAG0.6390.904 0.8640.858 0.816
Table 2: Reasoning generation answer-quality re-
sults on the considered MHQA benchmarks.
Implementation DetailsOur pipeline fol-
lows ingestion, retrieval, and generation
stages. During ingestion, each benchmark
corpus is split into chunks and indexed in
OpenSearch with 1024-dimensional embed-
dings. At inference time, retrieval uses
two stages: broad hybrid candidate re-
trieval (TopK=500) followed by reranking with
e5-mistral-7b-instruct (Wang et al., 2024)
to produce a compact evidence set (TopK=5).
We use GPT-4.1 for the Coverage Anchor
and Iterative Chain agents, and GPT-o4-mini
as the evidence controller. To test whether
retrieval-policy gains persist under stronger
generation-time reasoning, we evaluate both
non-reasoning and reasoning generation set-
tings using GPT-4.1 and GPT-o3, respectively.
Detailed settings appear in Appendix A.4.
4.2 Main Results
Tables 1 and 2 show that PAR2-RAG con-
sistently delivers the strongest overall answer
quality across benchmarks and generator set-
tings. Under non-reasoning generation, PAR2-
RAG achieves the best performance on all four
datasets. In particular, on MuSiQue, PAR2-
RAG yields its largest relative improvements:
+42.4% over ReAct and +23.5% over IRCoT.
Under reasoning generation, PAR2-RAG re-
mainsbestonaverage. Thelargestrelativegain
over Coverage Anchor is +1.9% on MuSiQue,
while the largest relative gain over Iterative
Chain is +4.1% on MoreHopQA. The runner-
up pattern also differs by regime: IterativeChain is second-best under non-reasoning gen-
eration, while Coverage Anchor is second-best
under reasoning generation. A plausible expla-
nation is that the dominant bottleneck shifts
with generator capability: with non-reasoning
generation, stronger chain refinement helps
more after retrieval, while with reasoning gen-
eration, broader evidence coverage becomes
relatively more important.
Table3providesdirectevidenceforthemech-
anism. PAR2-RAG achieves the best retrieval
performance on Recall, NDCG, and All-Pass
across all four datasets. These retrieval gains
align with the answer-quality improvements.
We hypothesize that ReAct and IRCoT are
less competitive in this setting because itera-
tive retrieval can suffer fromearly commitment:
reasoning starts before evidence coverage is suf-
ficiently broad, and early low-recall steps can
bias later retrieval. In contrast, PAR2-RAG ex-
plicitly separates coverage expansion and chain
refinement, which provides more controllable
behavior under fixed retrieval budgets.
4.3 Ablation Study
We conduct ablations to analyze how robust-
ness and where PAR2-RAG’s gains come from.
Ablation on GPT-5.2 SettingsWe eval-
uate PAR2-RAG and considered RAG-based
methods with GPT-5.2 model under two rea-
soning configurations ( noneand medium) for
both reasoning and generation. The experimen-
tal results in Table 4 show that PAR2-RAG
maintains clear gains over IRCoT and both
module variants in both reasoning settings. Un-
der reasoning= none, the largest relative gain
over Coverage Anchor is +9.3% on FRAMES,
and the largest gain over Iterative Chain is also
+7.2% on FRAMES. Under reasoning= medium,
the largest relative gain over Coverage Anchor
is +3.5% on 2Wiki, while the largest gain over
Iterative Chain is +4.8% on MuSiQue.
This extension provides a refined view of the
runner-up hypothesis. At the average level,
Iterative Chain remains the runner-up in both
settings. However, when moving from reason-
ing= noneto reasoning= medium, the gap be-
tween Iterative Chain and Coverage Anchor
narrows substantially, and Coverage Anchor be-
comes competitive on multiple datasets. This
trend is consistent with our hypothesis: as
5

MethodMuSiQue 2Wiki MoreHopQA FRAMES Average
NDCG Recall All Pass NDCG Recall All Pass NDCG Recall All Pass NDCG Recall All Pass NDCG Recall All Pass
ReAct 0.596 0.581 0.240 0.724 0.683 0.386 0.896 0.891 0.782 0.717 0.671 0.355 0.733 0.707 0.441
IRCoT 0.611 0.692 0.420 0.750 0.795 0.622 0.822 0.884 0.771 0.804 0.806 0.565 0.747 0.794 0.595
Coverage Anchor 0.610 0.737 0.428 0.708 0.804 0.588 0.881 0.944 0.888 0.785 0.794 0.544 0.746 0.820 0.612
Iterative Chain 0.615 0.691 0.392 0.762 0.768 0.548 0.898 0.895 0.790 0.785 0.794 0.545 0.765 0.787 0.569
PAR2-RAG0.636 0.747 0.460 0.784 0.877 0.728 0.908 0.954 0.908 0.834 0.874 0.677 0.791 0.863 0.693
Table 3: Retrieval performance on the considered multi-hop QA benchmarks. We report NDCG, Recall,
and All Pass for the retrieval stage. Best results are in bold and second-best are underlined.
Figure 2: Step robustness results on answer quality, each panel compares IRCoT, Coverage Anchor,
Iterative Chain, and PAR2-RAG across step counts
3,5,7,10.
Method MuSiQue 2Wiki MoreHopQA FRAMES Average
OpenAI-GPT-5.2Reasoning=None
IRCoT 0.542 0.747 0.698 0.533 0.630
Coverage Anchor 0.603 0.785 0.757 0.707 0.713
Iterative Chain 0.607 0.827 0.782 0.721 0.734
PAR2-RAG0.622 0.838 0.808 0.773 0.760
OpenAI-GPT-5.2Reasoning=Medium
IRCoT 0.622 0.807 0.682 0.590 0.675
Coverage Anchor 0.661 0.859 0.845 0.849 0.803
Iterative Chain 0.652 0.879 0.8640.829 0.806
PAR2-RAG0.683 0.8890.862 0.857 0.823
Table 4: GPT-5.2 results across MHQA bench-
marks with two reasoning configurations.
generation-time reasoning becomes stronger,
broader evidence coverage contributes rela-
tively more to final accuracy.
Ablation on Step RobustnessFigure 2
shows step/sub-query scaling for all consid-
ered methods. The experimental results show
the following observations: the performance of
all methods generally improves from 3 to 5/7
steps/sub-queries, then saturates or declines
at 10 steps on some datasets. We can tell that
the proposed PAR2-RAG remains the strongest
overall performance, achieving the highest an-
swer quality score at 12 of 16 benchmark-step
points and the highest average performance
across all benchmarks.
The results suggests that PAR2-RAG is ro-
bust in terms of maintaining high quality underbudget changes, even when deeper search does
not help uniformly. Variance-based stability
also highlights method differences: IRCoT and
Iterative Chain exhibit larger step-wise fluctu-
ations than Coverage Anchor, indicating that
retrieval-first anchoring yields more stable be-
havior under budget variation. Overall, these
patterns support the adaptive retrieval control
view: additional depth helps only when newly
retrieved evidence remains on-path.
5 Conclusions
Finding high-quality evidence across multi-
source corpora for multi-hop question answer-
ing remains challenging. We proposed Planned
Active Retrieval and Reasoning RAG (PAR2-
RAG), a two-stage framework that delays rea-
soning commitment until sufficient evidence
coverage is achieved for MHQA. Across four
MHQA benchmarks, PAR2-RAG is consis-
tently best for answer quality and achieves the
strongest retrieval performance. Together with
step-depth ablations, these results support a
practical design rule for industry MHQA sys-
tems: use coverage-first anchoring before deep
refinement, and prefer adaptive depth control
to fixed large hop budgets.
6

References
Akari Asai, Zeqiu Wu, Yizhong Wang, Avirup Sil,
and Hannaneh Hajishirzi. 2024. Self-rag: Learn-
ing to retrieve, generate, and critique through
self-reflection.
Eden Biran, Daniela Gottesman, Sohee Yang, Mor
Geva, and Amir Globerson. 2024. Hopping too
late: Exploring the limitations of large language
models on multi-hop queries. InProceedings
of the 2024 Conference on Empirical Methods
in Natural Language Processing, pages 14113–
14130.
Jiabei Chen, Guang Liu, Shizhu He, Kun Luo, Yao
Xu, Jun Zhao, and Kang Liu. 2025. Search-in-
context: Efficient multi-hop QA over long con-
texts via Monte Carlo tree search with dynamic
KV retrieval. InFindings of the Association
for Computational Linguistics: ACL 2025, pages
26443–26455, Vienna, Austria. Association for
Computational Linguistics.
Rong Cheng, Jinyi Liu, Yan Zheng, Fei Ni, Jiazhen
Du, Hangyu Mao, Fuzheng Zhang, Bo Wang,
and Jianye Hao. 2025. Dualrag: A dual-process
approach to integrate reasoning and retrieval for
multi-hop question answering.arXiv preprint
arXiv:2504.18243.
DeepSeek-AI. 2025. Deepseek-r1: Incentivizing rea-
soning capability in llms via reinforcement learn-
ing.arXiv preprint arXiv:2501.12948.
Yunfan Gao, Yun Xiong, Xinyu Gao, Kangxiang
Jia, Jinliu Pan, Yuxi Bi, Yixin Dai, Jiawei Sun,
HaofenWang, andHaofenWang.2023. Retrieval-
augmented generation for large language models:
A survey.arXiv preprint arXiv:2312.10997, 2(1).
Kelvin Guu, Kenton Lee, Zora Tung, Panupong Pa-
supat, and Mingwei Chang. 2020. Retrieval aug-
mented language model pre-training. InInter-
national conference on machine learning, pages
3929–3938. PMLR.
Xanh Ho, Anh-Khoa Duong Nguyen, Saku Sug-
awara, and Akiko Aizawa. 2020. Constructing
a multi-hop qa dataset for comprehensive eval-
uation of reasoning steps. InProceedings of the
28th International Conference on Computational
Linguistics, pages 6609–6625.
Gautier Izacard, Mathilde Caron, Lucas Hosseini,
Sebastian Riedel, Piotr Bojanowski, Armand
Joulin, and Edouard Grave. 2021. Unsuper-
vised dense information retrieval with contrastive
learning.arXiv preprint arXiv:2112.09118.
Gautier Izacard, Patrick Lewis, Maria Lomeli, Lu-
cas Hosseini, Fabio Petroni, Timo Schick, Jane
Dwivedi-Yu, Armand Joulin, Sebastian Riedel,
and Edouard Grave. 2022. Few-shot learning
withretrievalaugmentedlanguagemodels.arXiv
preprint arXiv:2208.03299, 1(2):4.Olivier Jeunen, Ivan Potapov, and Aleksei Usti-
menko. 2024. On (normalised) discounted cu-
mulative gain as an off-policy evaluation metric
for top-n recommendation. InProceedings of
the 30th ACM SIGKDD conference on knowledge
discovery and data mining, pages 1222–1233.
Vladimir Karpukhin, Barlas Oguz, Sewon Min,
Patrick SH Lewis, Ledell Wu, Sergey Edunov,
Danqi Chen, and Wen-tau Yih. 2020. Dense
passage retrieval for open-domain question an-
swering. InEMNLP (1), pages 6769–6781.
Satyapriya Krishna, Kalpesh Krishna, Anhad Mo-
hananey, Steven Schwarcz, Adam Stambler,
Shyam Upadhyay, and Manaal Faruqui. 2025.
Fact, fetch, and reason: A unified evaluation
of retrieval-augmented generation. InProceed-
ings of the 2025 Conference of the Nations of the
Americas Chapter of the Association for Compu-
tational Linguistics: Human Language Technolo-
gies (Volume 1: Long Papers), pages 4745–4759.
Patrick Lewis, Ethan Perez, Aleksandra Piktus,
Fabio Petroni, Vladimir Karpukhin, Naman
Goyal, Heinrich Küttler, Mike Lewis, Wen-
tau Yih, Tim Rocktäschel, and 1 others. 2020.
Retrieval-augmented generation for knowledge-
intensive nlp tasks.Advances in neural informa-
tion processing systems, 33:9459–9474.
Yuankai Li, Jia-Chen Gu, Di Wu, Kai-Wei Chang,
and Nanyun Peng. 2024. Brief: Bridging re-
trieval and inference for multi-hop reasoning via
compression.arXiv preprint arXiv:2410.15277.
Hao Liu, Zhengren Wang, Xi Chen, Zhiyu Li, Feiyu
Xiong, Qinhan Yu, and Wentao Zhang. 2025.
Hoprag: Multi-hop reasoning for logic-aware
retrieval-augmented generation.arXiv preprint
arXiv:2502.12442.
OpenAI. 2023. Gpt-4 technical report.arXiv
preprint arXiv:2303.08774.
OpenAI. 2025. Openai o3 and o4-
mini. https://openai.com/index/
introducing-openai-o3-and-o4-mini/ .
Accessed: 2026-02-08.
Julian Schnitzler, Xanh Ho, Jiahao Huang, Flo-
rian Boudin, Saku Sugawara, and Akiko Aizawa.
2024. Morehopqa: More than multi-hop reason-
ing.arXiv preprint arXiv:2406.13397.
Weijia Shi, Sewon Min, Michihiro Yasunaga,
Minjoon Seo, Rich James, Mike Lewis, Luke
Zettlemoyer, and Wen-tau Yih. 2023. Replug:
Retrieval-augmented black-box language models.
arXiv preprint arXiv:2301.12652.
Aaditya Singh, Adam Fry, Adam Perelman, Adam
Tart, Adi Ganesh, Ahmed El-Kishky, Aidan
McLaughlin, Aiden Low, AJ Ostrow, Akhila
Ananthram, Akshay Nathan, Alan Luo, Alec
7

Helyar, Aleksander Madry, Aleksandr Efre-
mov, Aleksandra Spyra, Alex Baker-Whitcomb,
Alex Beutel, Alex Karpenko, and 465 others.
2025. Openai gpt-5 system card.Preprint,
arXiv:2601.03267.
Yixuan Tang and Yi Yang. 2024. Multihop-
rag: Benchmarking retrieval-augmented gener-
ation for multi-hop queries.arXiv preprint
arXiv:2401.15391.
Gemini Team and Google DeepMind. 2024. Gemini
1.5: Unlocking multimodal understanding across
millions of tokens of context.arXiv preprint
arXiv:2403.05530.
Harsh Trivedi, Niranjan Balasubramanian, Tushar
Khot, and Ashish Sabharwal. 2022. Musique:
Multihop questions via single-hop question com-
position.Transactions of the Association for
Computational Linguistics, 10:539–554.
Harsh Trivedi, Niranjan Balasubramanian, Tushar
Khot, and Ashish Sabharwal. 2023. Interleav-
ing retrieval with chain-of-thought reasoning for
knowledge-intensivemulti-stepquestions. InPro-
ceedings of the 61st Annual Meeting of the Asso-
ciation for Computational Linguistics (Volume
1: Long Papers), pages 10014–10037.
Liang Wang, Nan Yang, Xiaolong Huang, Linjun
Yang, Rangan Majumder, and Furu Wei. 2024.
Improving text embeddings with large language
models. InProceedings of the 62nd Annual Meet-
ing of the Association for Computational Lin-
guistics (Volume 1: Long Papers), pages 11897–
11916.
Jason Wei, Xuezhi Wang, Dale Schuurmans,
Maarten Bosma, Fei Xia, Ed Chi, Quoc V Le,
Denny Zhou, and 1 others. 2022. Chain-of-
thought prompting elicits reasoning in large lan-
guage models.Advances in neural information
processing systems, 35:24824–24837.
Shunyu Yao, Jeffrey Zhao, Dian Yu, Nan Du, Izhak
Shafran, Karthik Narasimhan, and Yuan Cao.
2023. React: Synergizing reasoning and acting
in language models. InInternational Conference
on Learning Representations (ICLR).
Rongzhi Zhu, Xiangyu Liu, Zequn Sun, Yiwei
Wang, and Wei Hu. 2025. Mitigating lost-
in-retrieval problems in retrieval augmented
multi-hop question answering.arXiv preprint
arXiv:2502.14245.A Appendix
A.1 Dataset
Dataset Queries Docs Avg Text Length
MuSiQue 500 19990 489.52
2Wiki 500 10000 116.07
MoreHopQA 500 9942 132.44
FRAMES 500 9942 132.44
Average 500 13311 246.01
Table 5: Statistics of the multi-hop QA datasets.
’Query Number’ and ’Docs Number’ refer to the
total count of queries and documents, respectively.
’Avg Text Length’ is the average word count per
document.
A.2 Evaluation Metrics
Correctness EvaluationTo further evalu-
ate the generated answer quality beyond string
matching, we use a Correctness metric to mea-
sure the proportion of generated responses that
are judged as correct by an LLM evaluator us-
ing a structured prompt template:
Correct =|{r|LLM-Judge(q, p, g) =“yes”}|
|P|
(2)
where qdenotes the query, gdenotes the
ground-truth answer, and LLM-Judge (q, p, g )
is a language model that evaluates whether
the response pcorrectly answers query q
given ground truth g. The evaluation uses
GPT-4 with a predefined prompt template
in Prompt 3, which outputs binary judg-
ments, with responses containing "yes" (case-
insensitive) scored as 1 and others as 0.
NDCGThe Normalized Discounted Cumula-
tiveGain(NDCG)metricevaluatestheranking
quality of retrieved documents by considering
both relevance and position:
NDCG@k =DCG@k
IDCG@k(3)
where DCG@k is the Discounted Cumulative
Gain at rank k, calculated as:
DCG@k =k/summationdisplay
i=12reli−1
log2(i+ 1)(4)
and IDCG@k is the ideal DCG@k obtained by
sorting all relevant documents in descending
order of relevance. Here, relirepresents the
8

Prompt for Correctness judgment
You are a helpful research assistant. Your task is to evaluate an LLM’s answer against a
ground-truth answer and decide whether the ground-truth content is present in the model’s
response.
Instructions:
1. Carefully compare thePredicted Answerwith theGround-Truth Answer.
2.Judge based on substance and equivalence of meaning; do not require identical wording
unless wording is crucial to meaning.
3.Make a binary decision on whether the vital facts of the ground-truth are contained in
the predicted answer.
Input Data:
Question: {query}
Predicted Answer: {response}
Ground-Truth Answer: {answer}
Output Format:
Provide your final evaluation in the following format:
Explanation: <brief rationale for the decision>
Decision: <yes|no>
Output:
Figure 3: Prompt for correctness judgment.
relevance score of the document at position i,
with binary relevance where documents con-
taining ground-truth context receive a score of
1 and others receive 0.
Retrieval RecallThe Retrieval Recall met-
ricmeasurestheproportionofqueriesforwhich
atleastoneground-truthcontextissuccessfully
retrieved in the top-k results:
Recall@k =|q|GT q∩Retrieved@k q̸=∅|
|Q|
(5)
where Qdenotes the set of all queries, GTq
represents the set of ground-truth for query q,
andRetrieved@k qrepresents the set of docu-
ments in the top-k retrieved results for query q.
Each query receives a binary score of 1 if any
ground-truth appears in the retrieval results,
and 0 otherwise.
A.3 Prompts
A.4 Implementation Details
All main experiments use OpenSearch
TopK=100 and rerank TopK=5. Unlessotherwise stated, the step/sub-query bud-
get is 5 and retrieval summary is disabled
(self.nosummary_3 ). The retriever and
reranker use openai.o4-mini-2025-04-16 .
Main settings are:
•Non-reasoning generation setting: reason-
ing model openai.gpt-4.1-2025-04-14 ,
generation model
openai.gpt-4.1-2025-04-14.
•Reasoning generation set-
ting: reasoning model
openai.gpt-4.1-2025-04-14 , gener-
ation modelopenai.o3-2025-04-16.
•Extension setting: reason-
ing and generation models
openai.gpt-5.2-2025-12-11 with
reasoning mode variations.
A.5 Ablation Details
Step/sub-query without reasoning gener-
ation (300-sample study)For PAR2-RAG,
correctness on MuSiQue increases from 0.610
9

Prompt for Planner
You are a helpful research assistant. Given a query, come up with a set of database searches
to perform to best answer the query. Output 5 terms to query for.
Format your response as JSON (No code snippet) with a list of database searches needed to
answer the query. Each search should include:
1.Reason:A brief explanation of why this search is necessary.
2.Query:The exact search term to use.
Example output:
{
"searches": [
{
"reason": "Identify the best Caribbean destinations for surfing in April.",
"query": "best Caribbean surfing spots April"
},
{
"reason": "Find hiking trails in the Caribbean suitable for April vacations.",
"query": "hiking trails Caribbean April"
}
]
}
Figure 4: Prompt for planner.
(3 steps) to 0.680 (7 steps), then drops to 0.630
(10 steps). FRAMES increases from 0.793 (3)
to 0.800 (7–10). Similar non-monotonic trends
appear for IRCoT, Multi-step RAG, and Query
Decomposition.
Step/sub-query with reasoning genera-
tion (300-sample study)With reasoning
generation, PAR2-RAG reaches 0.940 on 2Wiki
at 7 steps and 0.700 on MuSiQue at 10 steps,
while FRAMES peaks at 0.866 at 7 steps. The
overall trend remains non-monotonic, reinforc-
ing the need for adaptive stopping.
A.6 Additional Results
GPT-5.2 extensionWith reasoning mode
set to none, PAR2-RAG reports 0.838 (2Wiki),
0.808 (MoreHopQA), 0.622 (MuSiQue), and
0.773 (FRAMES). With reasoning mode set
to medium, PAR2-RAG reports 0.889, 0.862,
0.683, and 0.857, respectively.
10

Prompt for Searcher
You are a database search interface. Your ONLY responsibility is to return the FIVE
database results exactly as provided, without any modification.
Strict Rules:
1. Output must consist ONLY of the raw database results.
2.Preserve the original wording, spelling, punctuation, capitalization, line breaks, and
formatting of the database entries.
3. Do NOT summarize, rephrase, explain, interpret, or add commentary.
4. Do NOT add introductions, conclusions, or transitional text.
5. Do NOT merge, reorder, or alter results beyond their original sequence.
6.If multiple entries are returned, output them in exactly the same order and format as
they are given by the database.
7. MUST output all five retrieved entries.
Example (for illustration only — do not generate similar text unless the database
returns it):
[Document1.txt]
Document1 is an example entry from the database.
[Document2.txt]
Document2 is other entry in the database.
[Document3.txt]
Document3 is other entry in the database.
[Document4.txt]
Document4 is other entry in the database.
[Document5.txt]
Document5 is other entry in the database.
Figure 5: Prompt for searcher.
Prompt for Writer
You are a senior researcher tasked with providing a comprehensive answer to a research
query. You will be provided with the original query, and initial research done by a research
assistant.
DO NOT WRITE A SUMMARY,directly provide a complete, accurate answer
to the original question.
Output Format
Format your response asJSON (No code snippet)with:
•answer : An answer that directly addresses the research query using all available evidence.
Example output:
{
"answer": "Based on the research findings, the Caribbean in April offers exceptional
conditions for outdoor activities..."
}
Figure 6: Prompt for Writer.
11