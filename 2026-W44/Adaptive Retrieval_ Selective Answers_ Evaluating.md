# Adaptive Retrieval, Selective Answers: Evaluating Conformal Factuality in Agentic RAG

**Authors**: Muhammad Aimal Rehman, Chi-Kuang Yeh

**Published**: 2026-10-26 05:37:03

**PDF URL**: [https://arxiv.org/pdf/2610.34872v1](https://arxiv.org/pdf/2610.34872v1)

## Abstract
Retrieval-augmented generation (RAG) can improve factual grounding without guaranteeing that every generated claim is supported. Previous work has studied claim-level conformal factuality in single-stage RAG and, more recently, multi-hop RAG. We examine its behavior in agentic RAG, where a language-model policy adaptively issues follow-up retrieval queries or terminates retrieval before response generation. We retain the retrieval-based relevance-scoring formulation and post-generation conformal filtering and evaluate six model-dataset configurations: GPT-6 Luna and Llama 3.1 8B on HotpotQA, Natural Questions, and TriviaQA. Each configuration uses 60 fixed questions, 100 seeded 30/30 calibration-test partitions, five conformal target levels, and a no-conformal-prediction baseline. Across the six configurations, higher target levels are accompanied by higher observed full-support rates and reduced answer availability. At the 95% target, non-empty response coverage ranges from 17.23% to 60.17%, claim retention from 10.72% to 43.16%, and full support among non-empty outputs from 89.14% to 95.56%. We formulate the adaptive retrieval trajectory explicitly and characterize the reliability-informativeness trade-off when conformal factuality is applied after agent termination.

## Full Text


<!-- PDF content starts -->

Adaptive Retrieval, Selective Answers:
Evaluating Conformal Factuality in Agentic RAG
Muhammad Aimal Rehman
Department of Mathematics & Statistics
Georgia State University
Atlanta, GA, USA
mrehman3@gsu.eduChi-Kuang Yeh
Department of Mathematics & Statistics
Georgia State University
Atlanta, GA, USA
cyeh@gsu.edu
Abstract
Retrieval-augmented generation (RAG) can improve factual grounding without guaranteeing
that every generated claim is supported. Previous work has studied claim-level conformal
factuality in single-stage RAG and, more recently, multi-hop RAG. We examine its behavior
inagenticRAG, where a language-model policy adaptively issues follow-up retrieval queries
or terminates retrieval before response generation. We retain the retrieval-based relevance-
scoring formulation and post-generation conformal filtering and evaluate six model–dataset
configurations: GPT-6 Luna and Llama 3.1 8B on HotpotQA, Natural Questions, and TriviaQA.
Each configuration uses 60 fixed questions, 100 seeded 30/30 calibration–test partitions, five
conformal target levels, and a no-conformal-prediction baseline. Across the six configurations,
higher target levels are accompanied by higher observed full-support rates and reduced answer
availability. At the 95% target, non-empty response coverage ranges from 17.23% to 60.17%,
claim retention from 10.72% to 43.16%, and full support among non-empty outputs from
89.14% to 95.56%. We formulate the adaptive retrieval trajectory explicitly and characterize the
reliability–informativeness trade-off when conformal factuality is applied after agent termination.
1 Introduction
Retrieval-augmented generation (RAG) conditions language-model responses on external documents,
but evidence retrieval does not ensure that every generated claim is factual or supported [ 4]. Claim-
level conformal factuality provides a complementary mechanism: after generating and decomposing
a response into atomic claims, a calibration-derived score threshold determines which claims are
retained [ 1,5]. Our previous study extended this approach to multi-hop RAG, in which successive
retrieval operations construct evidence for compositional questions [7].
The present study extends that evaluation toagentic RAG. Instead of following only a bounded
query-rewriting procedure, an agent makes explicit retrieve orfinishdecisions as evidence
accumulates. Its follow-up queries, retrieval trajectory, and stopping time depend on the question
and evidence observed so far. This adaptivity changes how the evidence used to generate and
score claims is acquired, while the conformal post-processing stage remains conceptually the same.
Decision-making interleaved with access to external information is also central to earlier agent-
oriented language-model research [9].
We address two empirical questions: (i) whether increasing conformal factuality targets produces
the same observed improvement in response-level support when evidence acquisition is agent-driven;
1
arXiv:2610.34872v1  [cs.CL]  26 Oct 2026

Table 1:Operational differences between the evaluated multi-hop and agentic retrieval procedures. Both
apply post-generation claim scoring and conformal filtering.
Dimension Multi-hop RAG Agentic RAG
Follow-up query Produced by a query-rewriting stage using
prior evidenceSelected as the argument of an LLM-policy
retrieveaction
Retrieval control Iterative refinement within a hop budget Explicit retrieve /finishdecision after
accumulated evidence is inspected
Early termination Configured hop/novel-evidence stopping
conditionsPolicy finish, repeated-query safeguards,
or configured decision/retrieval limits
Evidence Aggregated across a sequence of retrieval
hopsAggregated across adaptively selected re-
trieval actions
Response/CP stage Generation, atomic-claim scoring, confor-
mal filteringSame downstream stages after the agent
terminates
and (ii) how many claims and non-empty answers survive filtering across models and QA datasets.
Our goal is an experimental extension of conformal factuality, not a comparison intended to
rank agentic retrieval against multi-hop retrieval. We contribute a formal representation of the
implemented agentic retrieval process and a six-configuration evaluation with complete target sweeps
and unfiltered baselines.
2 Background and Relation to Prior Work
RAG and adaptive retrieval.RAG conditions generation on retrieved evidence [ 4]. For multi-
step questions, iterative retrieval can use information from earlier steps to form subsequent queries.
Agentic retrieval adds an explicit decision policy that can continue searching or terminate, as in
general reasoning-and-acting paradigms [ 9]. It is important to distinguish abounded decision policy
from an oracle for evidence sufficiency: a decision to finish does not certify that all relevant evidence
has been retrieved.
Conformal factuality.Conformal factuality uses a held-out calibration sample to select a nested
refinement or filtering operation for language-model outputs [ 5]. The SIGIR 2025 Conformal-RAG
study [1] introduced retrieval-informed claim scores, including a product-based score followed
by max aggregation. Our multi-hop work examined this framework when evidence comes from
multiple retrieval hops [ 7]. Here we retain the same relevance-scoring formulation and split-conformal
experimental protocol, while replacing the upstream evidence-construction procedure with an agentic
retriever.
3 Agentic Retrieval and Conformal Factuality
3.1 Policy-driven retrieval with an adaptive stopping time
Letx∈Xbe a question and Ka document corpus. We indexretrieval operations, rather than
agent-policy calls, by t. The initial query is the original question, q(1)=x. Given a top- kretriever
with similarity cutoffθ, the initial document sequence is
D1= Unique/parenleftig
Retrieve(K,q(1);k,θ)/parenrightig
.(1)
2

Aftertretrievals, let Qt= (q(1),...,q(t))and letDtcontain the distinct passages accumulated in
retrieval order. Write the decision state as
Ht=/parenleftbigx,Ψ(Dt),Qt,t,B−t/parenrightbig, at=πω(Ht)∈{Finish}∪{Retrieve(q) :q∈Q}.(2)
Hereπωis the agent’s policy, Bis the retrieval-operation budget, Qis the space of follow-up queries,
andΨ(Dt)is thebounded evidence viewsupplied to the policy, not necessarily the full accumulated
corpus. Ifa t= Retrieve(q(t+1))is admissible, then
Et+1= Retrieve(K,q(t+1);k,θ),(3)
Dt+1=Dt⊕New(E t+1,Dt),(4)
where Newremoves duplicates already present in Dtand⊕appends previously unseen passages.
The order matters because it can affect the evidence view available for the next policy decision.
Letτ(x)be the number of completed retrieval operations when the process terminates. Termina-
tion can result from a finishdecision, the retrieval or decision-step limit, or safeguards against
repeated queries; it need not be a calibrated confidence event. With an initial retrieval and bounded
operations,
1≤τ(x)≤B, D π(x) :=Dτ(x).(5)
In the experiments B= 3and the policy-decision budget is also three. The retriever performs
one initial retrieval before the policy makes follow-up decisions. This description permits different
questions to have different retrieval trajectories without assuming independence between retrieval
stages.
3.2 Generation, claim decomposition, and relevance scoring
After agent termination, the model generates a response and the claim decomposer extracts its
atomic statements:
ˆy=G/parenleftbigx,Dπ(x)/parenrightbig, C(x) =g(ˆy) ={c 1,...,cp(x)}.(6)
The number of extracted claims p(x)varies across questions. The factuality annotation for each
claim belongs to{S,I,U,N} , where only Scounts as supported. Write A(c)∈{0,1}for this support
indicator; reference answers and support annotations enter calibration and evaluation, not test-time
threshold selection.
The evaluated relevance scorer follows the product-and-maximum structure of Feng et al. [1].
A document dfirst added at retrieval operation j(d)carries a stored retrieval similarity sret(d),
evaluated for the query q(j(d))that retrieved it. Theimplementedagentic relevance score can be
expressed as
Ragent(c;x) = max/braceleftigg
0,max
d∈Dπ(x)sret(d) sim(c,d)/bracerightigg
.(7)
The similarity sim(c,d)measures semantic alignment between claim and passage. When retrieval
similarities are cosine similarities in the same embedding space, sret(d) =sim(q(j(d)),d). Documents
retrieved during follow-up steps are scored against their corresponding agent-generated retrieval
queries, whereas documents from the initial retrieval are scored against the original question x. This
contrasts with the idealized original-question factor sim(x,d)in the single-query relevance-score
expression of Feng et al. [1], even though the product/max scoring strategy is unchanged.
3

For tie handling, let/tildewideR(c) =Ragent(c;x) +ε(c)denote the score passed to conformal filtering,
including any configured tie-breaking perturbation. Therecorded experimentsuse the thresholded
retained set
Fλ(C(x)) ={c∈C(x) :/tildewideR(c)≥λ}.(8)
Thresholding is post-generation: the conformal target does not control when the retrieval agent
stops.
3.3 Calibration and response-level factuality
For a calibration question i, letCibe the extracted claims and rij=/tildewideR(cij)their relevance scores.
PutAij= 1if claim cijis supported and zero otherwise. The response is fully supported at threshold
λwhen every retained claim is supported:
Bi(λ) =I[∀c ij∈Fλ(Ci), Aij= 1].(9)
The all-claims-supported requirement corresponds to the within-response factuality parameter a= 1
in our earlier study [7]. A response retaining no claims satisfies eq. (9) by convention.
A useful explicit nonconformity score forall-claims-supportedcalibration is
Vi= max
j:Aij=0rij,max∅=−∞.(10)
ThusViis the largest relevance score of any unsupported claim. If V(1)≤···≤V (n)are the ordered
calibration scores, the usual split-conformal order-statistic threshold is
kα=⌈(n+ 1)(1−α)⌉, ˆλα=/braceleftigg
V(kα), kα≤n,
+∞, k α=n+ 1.(11)
The five evaluated targets1 −α∈{ 0.80,0.85,0.90,0.925,0.95}correspond, for n= 30, tokα∈
{25,27,28,29,30}.
4 Experimental Design
4.1 Datasets and models
We evaluate HotpotQA [ 8], Natural Questions (NQ) [ 3], and TriviaQA [ 2], using KILT-linked
validation resources and Wikipedia provenance [ 6]. Each experiment uses a deterministic 60-question
subset selected under its dataset-specific answer/provenance and extractable-Wikipedia-evidence
eligibility rules. NQ and TriviaQA selections are matched across the two model experiments using
the saved GPT-6 reference samples. These eligibility conditions define the evaluated population and
should be retained when reproducing the study.
The two generation backbones are the GPT-6 Luna API model ( gpt-6-luna ) and
meta-llama/Llama-3.1-8B-Instruct . Each model family supplies the generator, agent-policy
decisions, claim decomposition, and factuality verification within its own experimental pipeline. The
common retrieval stack uses local sentence-transformer embeddings ( all-MiniLM-L6-v2 ), a FAISS
index, top-k= 10documents per retrieval, similarity cutoff θ= 0.3, initial retrieval enabled, and
at most three retrieval operations and three policy decisions. The policy conditions on up to five
evidence documents, each truncated to 1200 characters. The local Llama experiments used bounded
retries for malformed structured agent/verifier outputs.
4

4.2 Calibration and evaluation protocol
For each of the six model–dataset configurations, we generate, decompose, score, and annotate 60
question-level examples. A no-CP baseline retains all generated claims. For each of the five CP
targets, the same 60 scored/annotated examples are repeatedly divided into 30 calibration and 30
test cases over 100 seeded splits (base seed 42). Only the conformal threshold varies across target
conditions within a split. The 100 runs arerepeated partitions of fixed generated outputs, not 100
independent retrieval-and-generation runs.
The experimental matrix is
{GPT-6 Luna,Llama 3.1 8B}×{HotpotQA,NQ,TriviaQA}×{No CP,80,85,90,92.5,95%}.(12)
All six experiments use the product-based relevance score with maximum aggregation.
4.3 Reported metrics
For test example i, writeCi(λ) =Fλ(Ci)andni(λ) =|Ci(λ)|. Letm= 30denote the test split size.
The response-level full-support fraction and non-empty response coverage are
FS(λ) =1
mm/summationdisplay
i=1I/bracketleftbig∀c∈Ci(λ), A(c) = 1/bracketrightbig,(13)
NE(λ) =1
mm/summationdisplay
i=1I[ni(λ)>0].(14)
The mean-per-response fraction of retained claims is
Ret(λ) =1
mm/summationdisplay
i=1ni(λ)
|Ci|.(15)
The supported-claim fraction is the number of supported retained claims divided by the total
number of retained claims in the split; the summary tables average this split-level ratio over the 100
repetitions. The conditional full-support fraction is
FS|NE(λ) =/summationtext
iI[ni(λ)>0]I[∀c∈C i(λ),A(c) = 1]/summationtext
iI[ni(λ)>0],(16)
when the denominator is nonzero. The no-CP baseline always has 100% non-empty coverage and
100% claim retention for the successfully decomposed examples. We report percentages averaged
across the 100 calibration–test partitions; they should not be treated as independent sample
replicates.
5 Results
5.1 Reliability–coverage trade-off across six configurations
Figure 1 shows a consistent direction across the six settings: increasing the conformal target raises
observed full support while reducing the fraction of responses retaining any claim. The no-CP
baselines lie at 100% non-empty coverage. The magnitude of the resulting selectivity cost depends
strongly on the dataset/model combination; in particular, NQ reaches much lower coverage than
TriviaQA at stringent targets.
5

0 10 20 30 40 50 60 70 80 90 100
Non-empty response coverage (%)6065707580859095100Responses fully supported (%)
GPT-6 Luna · HotpotQA
Llama 3.1 8B · HotpotQA
GPT-6 Luna · Natural QuestionsLlama 3.1 8B · Natural Questions
GPT-6 Luna · TriviaQA
Llama 3.1 8B · TriviaQA
CP target (left to right on each curve): 95%, 92.5%, 90%, 85%, 80%.   Crosses: No CP baseline (X: GPT-6; ×: Llama).Figure 1:Agentic RAG using the relevance-scoring formulation in eq. (7), evaluated across GPT-6 Luna
and Llama 3.1 8B on three QA datasets. The five points on each curve are the conformal targets 80%, 85%,
90%, 92.5%, and 95%; separate crosses at 100% non-empty coverage denote the no-CP baselines. The y-axis
is response-level full support and includes vacuously supported empty filtered outputs. The vertically zoomed
scale emphasizes reliability–coverage differences.
5.2 Results at the 95% target
Table 2 summarizes the most stringent target. It is computed from the exact target-sweep tables,
not by reading locations from the plotted curves.
At the 95% target, non-empty coverage spans 17.23% (GPT-6 Luna, NQ) to 60.17% (Llama
3.1 8B, TriviaQA); claim retention spans 10.72% to 43.16%. The full-support fractionamong
non-empty outputsspans 89.14% to 95.56%, while supported-claim fractions among retained claims
span 90.16% to 97.72%. These are complementary measures: the response-level full-support rate
in the figure additionally counts empty filtered responses as fully supported under the standard
all-retained-claims event. The complete sweeps, including no-CP rows for all datasets and both
models, are supplied in tables 3 and 4.
5.3 Conditional reliability and answer availability
To separate selectivity from factuality of substantive answers, let Ndenote the event that filtering
retains at least one claim and Sthe full-support event. Since an empty claim set is vacuously fully
supported,
Pr(S) = 1−Pr(N) + Pr(N) Pr(S|N).(17)
6

Table 2:Agentic RAG using the relevance-scoring formulation in eq. (7) at the 95% CP target. All values
are percentages averaged over 100 calibration–test partitions. “Full support |non-empty” conditions on
retaining at least one claim.
Model DatasetNon-
emptyClaims
retainedSupported
claimsFull support
|non-empty
GPT-6 Luna HotpotQA 36.47 23.57 97.14 95.23
NQ 17.23 10.72 90.16 89.14
TriviaQA 51.20 43.16 96.59 95.56
Llama 3.1 8B HotpotQA 38.73 27.06 97.33 93.26
NQ 21.00 11.77 96.27 91.47
TriviaQA 60.17 39.13 97.72 94.56
Thus the vertical coordinate of a response-level full-support plot and the conditional full-support
metric answer different questions. The tables report both useful answer coverage and supported
content retention; the reliability–informativeness trade-off is therefore visible without reducing the
evaluation to a single reliability number.
6 Discussion
Across both model families, the same post-generation conformal filtering procedure produces
progressively more selective outputs as the target is tightened. This pattern extends the observations
from conventional and multi-hop RAG [ 1,7] to evidence acquired by a model-controlled retrieval
policy. The evidence trajectory is adaptive, but the CP operation is applied only after the agent
has terminated and generated a response.
The results also show that the utility cost of a given target is not uniform. For example, 95%-target
non-empty coverage exceeds 50% for both TriviaQA settings, but falls below 22% for both NQ
settings. Answer availability and retained-claim fraction should therefore accompany response-level
factuality in deployment-facing reports. We do not treat the variation across settings as evidence that
one retrieval architecture universally outperforms another; the models and experimental populations
differ from our previous multi-hop study.
7 Limitations
Our evaluation is based on controlled subsets of 60 questions from each of the three QA benchmarks:
HotpotQA, Natural Questions, and TriviaQA. The relatively small sample size may limit the
generalizability of the findings to larger and more diverse question sets.
The agentic retrieval process is limited to a maximum of three retrieval operations and three
policy decisions, leaving the effects of longer retrieval trajectories and alternative agent policies
unexplored. Furthermore, our experiments consider two language models, GPT-6 Luna and Llama
3.1 8B. Extending the evaluation to other model families, such as Qwen and DeepSeek, would help
establish the robustness and generalizability of the observed conformal factuality improvements
across different language-model architectures.
7

8 Conclusion
We formulated bounded Agentic RAG as an evidence-dependent sequence of retrieval-policy deci-
sions and evaluated retrieval-based conformal claim filtering after the agent stopped. Across six
configurations spanning two LLMs and three QA benchmarks, higher conformal targets yielded
higher observed response-level support together with lower claim retention and non-empty answer
coverage. These results extend the empirical investigation of conformal factuality from single-stage
and multi-hop RAG to agent-driven evidence acquisition, and motivate further research on improving
the reliability–informativeness trade-off through broader evaluations involving larger datasets, longer
retrieval trajectories, and additional language-model families.
Code Availability
The implementation of the Agentic RAG framework evaluated in this study is publicly available at
https://github.com/aimalrehman92/conformal-rag-agentic.
A Complete Target Sweeps and No-CP Baselines
The following tables present the complete experimental results for GPT-6 Luna and Llama 3.1 8B
across HotpotQA, Natural Questions, and TriviaQA. For each configuration, we report the no-CP
baseline and five conformal targets (80, 85, 90, 92.5, and 95%). The reported metrics are defined in
section 4.3.
Table 3:Agentic RAG with GPT-6 Luna using the relevance-scoring formulation in eq. (7): no-CP baseline
and complete conformal target sweep (%).
Dataset Target Non-empty Claims retained Supported claimsFully supported
|non-empty
HotpotQANo CP100.00 100.00 87.68 81.40
80% 97.73 92.57 88.63 82.66
85% 88.30 73.98 91.51 86.41
90% 79.60 62.28 93.33 88.82
92.5% 67.30 49.10 95.08 91.50
95% 36.47 23.57 97.14 95.23
NQNo CP100.00 100.00 76.36 65.33
80% 79.27 62.95 86.78 78.06
85% 63.03 46.56 88.71 82.23
90% 51.37 36.04 89.03 84.53
92.5% 35.77 23.74 89.20 86.15
95% 17.23 10.72 90.16 89.14
TriviaQANo CP100.00 100.00 87.63 85.33
80% 98.47 98.22 87.91 85.66
85% 84.60 82.51 89.22 87.59
90% 74.43 69.74 90.70 89.40
92.5% 64.80 58.34 93.16 91.99
95% 51.20 43.16 96.59 95.56
8

Table 4:Agentic RAG with Llama 3.1 8B using the relevance-scoring formulation in eq. (7): no-CP baseline
and complete conformal target sweep (%).
Dataset Target Non-empty Claims retained Supported claimsFully supported
|non-empty
HotpotQANo CP100.00 100.00 87.98 70.87
80% 90.73 84.19 91.66 78.26
85% 83.17 69.32 94.54 84.82
90% 77.63 60.33 95.61 87.88
92.5% 72.17 52.62 96.73 91.13
95% 38.73 27.06 97.33 93.26
NQNo CP100.00 100.00 86.44 64.83
80% 82.47 62.21 88.75 77.54
85% 62.67 43.64 91.70 80.44
90% 51.23 34.42 93.05 82.16
92.5% 40.73 24.73 94.60 86.30
95% 21.00 11.77 96.27 91.47
TriviaQANo CP100.00 100.00 86.38 66.80
80% 86.33 75.45 88.13 78.04
85% 77.03 61.78 91.58 83.69
90% 73.47 56.20 93.73 86.86
92.5% 70.43 50.02 95.82 90.51
95% 60.17 39.13 97.72 94.56
B Experimental Configuration Summary
Table 5:Shared conditions across the six prioritized Agentic RAG experiments.
Component Specification
Experiment IDs E25, E27, E29 (GPT-6 Luna); E31, E33, E35 (Llama 3.1 8B)
Dataset KILT-linked HotpotQA, Natural Questions, TriviaQA
Retrieval Agentic retrieve /finish; initialretrievalenabled; top-10; similaritycutoff
0.3
Budgets Maximum three retrieval operations and three policy decisions
Embedding / indexsentence-transformers/all-MiniLM-L6-v2/ FAISS
Score / filter Product and max aggregation; experimental retention uses /tildewideR(c)≥ ˆλ
Conformal setting Within-response factuality requirement a= 1; targets 80, 85, 90, 92.5, 95%
Protocol 60 fixed questions per configuration; 30 calibration, 30 test; 100 seeded
splits; seed 42
Baseline No CP: original scored response with no claim removal
9

References
[1]Naihe Feng, Yi Sui, Shiyi Hou, Jesse C. Cresswell, and Ga Wu. Response quality assessment for
retrieval-augmented generation via conditional conformal factuality. InProceedings of the 48th
International ACM SIGIR Conference on Research and Development in Information Retrieval,
pages 2832–2836, 2025. doi: 10.1145/3726302.3730244. URL https://doi.org/10.1145/
3726302.3730244.
[2]Mandar Joshi, Eunsol Choi, Daniel Weld, and Luke Zettlemoyer. TriviaQA: A large scale
distantly supervised challenge dataset for reading comprehension. InProceedings of the 55th
Annual Meeting of the Association for Computational Linguistics, pages 1601–1611, 2017. doi:
10.18653/v1/P17-1147.
[3]Tom Kwiatkowski, Jennimaria Palomaki, Olivia Redfield, Michael Collins, Ankur Parikh, Chris
Alberti, Danielle Epstein, Illia Polosukhin, Jacob Devlin, Kenton Lee, Kristina Toutanova, Llion
Jones, Matthew Kelcey, Ming-Wei Chang, Andrew Dai, Jakob Uszkoreit, Quoc Le, and Slav
Petrov. Natural questions: A benchmark for question answering research.Transactions of the
Association for Computational Linguistics, 7:452–466, 2019. doi: 10.1162/tacl_a_00276.
[4]Patrick Lewis, Ethan Perez, Aleksandra Piktus, Fabio Petroni, Vladimir Karpukhin, Naman
Goyal, Heinrich Küttler, Mike Lewis, Wen-tau Yih, Tim Rocktäschel, Sebastian Riedel, and
Douwe Kiela. Retrieval-augmented generation for knowledge-intensive NLP tasks. InAdvances
in Neural Information Processing Systems, volume 33, 2020. URL https://arxiv.org/abs/
2005.11401.
[5]Christopher Mohri and Tatsunori Hashimoto. Language models with conformal factuality
guarantees. InProceedings of the 41st International Conference on Machine Learning, volume
235 ofProceedings of Machine Learning Research, pages 36029–36047, 2024. URL https:
//proceedings.mlr.press/v235/mohri24a.html.
[6]Fabio Petroni, Aleksandra Piktus, Angela Fan, Patrick Lewis, Majid Yazdani, Nicola De Cao,
James Thorne, Yacine Jernite, Vladimir Karpukhin, Jean Maillard, Vassilis Plachouras, Tim
Rocktäschel, and Sebastian Riedel. KILT: a benchmark for knowledge intensive language
tasks. InProceedings of the 2021 Conference of the North American Chapter of the Association
for Computational Linguistics: Human Language Technologies, pages 2523–2544, 2021. doi:
10.18653/v1/2021.naacl-main.200.
[7]Muhammad Aimal Rehman and Chi-Kuang Yeh. Conformal factuality control for multi-
hop retrieval-augmented generation.arXiv preprint arXiv:2609.38222, 2026. URL https:
//arxiv.org/abs/2609.38222.
[8] Zhilin Yang, Peng Qi, Saizheng Zhang, Yoshua Bengio, William Cohen, Ruslan Salakhutdinov,
and Christopher D. Manning. HotpotQA: A dataset for diverse, explainable multi-hop question
answering. InProceedings of the 2018 Conference on Empirical Methods in Natural Language
Processing, pages 2369–2380, 2018. doi: 10.18653/v1/D18-1259.
[9]Shunyu Yao, Jeffrey Zhao, Dian Yu, Nan Du, Izhak Shafran, Karthik R. Narasimhan, and Yuan
Cao. ReAct: Synergizing reasoning and acting in language models. InInternational Conference
on Learning Representations, 2023. URLhttps://arxiv.org/abs/2210.03629.
10