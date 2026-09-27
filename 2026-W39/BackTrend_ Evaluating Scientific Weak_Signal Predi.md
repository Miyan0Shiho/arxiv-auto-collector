# BackTrend: Evaluating Scientific Weak-Signal Prediction via Backward Reconstruction

**Authors**: Xiao Zhou, Yilun Zhao, Owen Jiang, Tiansheng Hu, Cai Xu, Manasi Patwardhan, Arman Cohan

**Published**: 2026-09-21 17:18:19

**PDF URL**: [https://arxiv.org/pdf/2609.24921v1](https://arxiv.org/pdf/2609.24921v1)

## Abstract
Scientific weak signals are early, low-visibility research directions that later become central to mature scientific topics, yet existing resources such as trend tracking, citation forecasting, and foresight reports rarely provide validated reference sets that link concrete early precursors to later paradigms. We introduce BackTrend, a retrospective benchmark in which, given a mature target topic and a temporal evidence constraint, systems must recover two types of precursors: problem-space signals, underrecognized research problems, and solution-space signals, emerging methods for known problems. BackTrend contains 25 mature target topics in artificial intelligence and machine learning and 66 human-validated weak signals, reconstructed from large-scale literature by grounding each candidate in its 2019-2024 publication-frequency trajectory. We evaluate frontier LLMs, RAG systems, and agentic research systems using semantic matching and coverage-based metrics. Current systems often generate plausible but misaligned precursors, exhibiting topic drift, granularity mismatch, near-miss matching, and incomplete coverage; the strongest system achieves only 10.1% F1, while Coverage10 reaches at most 18.5% of the reference signals. Our budget analyses show that additional retrieval and web-search evidence can improve performance up to a moderate budget, but does not by itself close the substantial performance gap.

## Full Text


<!-- PDF content starts -->

BackTrend: Evaluating Scientific Weak-Signal Prediction via
Backward Reconstruction
Xiao ZhouY*Yilun ZhaoY*Owen JiangY*Tiansheng HuN
Cai XuYManasi PatwardhanTArman CohanY
YYale NLP LabNNew York UniversityTTCS Research
BackTrend Dataset
 BackTrend Code
Abstract
Scientific weak signals are early, low-visibility research directions that later become central to mature
scientific topics, yet existing resources such as trend tracking, citation forecasting, and foresight reports
rarely provide validated reference sets that link concrete early precursors to later paradigms. We introduce
BackTrend, a retrospective benchmark in which, given a mature target topic and a temporal evidence
constraint, systems must recover two types of precursors:problem-space signals, underrecognized research
problems, andsolution-space signals, emerging methods for known problems. BackTrend contains
25 mature target topics in artificial intelligence and machine learning and 66 human-validated weak
signals, reconstructed from large-scale literature by grounding each candidate in its 2019–2024 publication-
frequency trajectory. We evaluate frontier LLMs, RAG systems, and agentic research systems using
semantic matching and coverage-based metrics. Current systems often generate plausible but misaligned
precursors, exhibiting topic drift, granularity mismatch, near-miss matching, and incomplete coverage; the
strongest system achieves only 10.1% F1, while Coverage 10reaches at most 18.5% of the reference signals.
Our budget analyses show that additional retrieval and web-search evidence can improve performance up
to a moderate budget, but does not by itself close the substantial performance gap.
1 Introduction
Identifying early indicators of transformative research, or weak signals, has long been a difficult goal across scientific
disciplines (Ansoff, 1975; Porter et al., 2004). Detecting such signals is valuable for researchers, corporations, and
policymakers because it informs long-term investment, research prioritization, and policy design (Kajikawa et al.,
2008; Yoon and Kim, 2012; Lee et al., 2014; Ogawa and Kajikawa, 2015). As AI and machine learning accelerate
the pace of scientific research, effective foresight depends on detecting low-visibility but potentially high-impact
ideas within a large volume of background noise, and recent work continues to identify open issues in methods for
detecting such emerging themes (Ren and Zhao, 2021; Liu et al., 2023). Across scientific domains, LLM-based
research agents are increasingly deployed to search literature, synthesize emerging areas, and propose new methods
(Zhao et al., 2026a; Hu et al., 2026; Yu et al., 2025), and a growing body of benchmarks evaluates their scientific
literature understanding and research capabilities (Zhao et al., 2025a; Xu et al., 2025; Chen et al., 2026a; Zhao et al.,
2026b; Wang et al., 2025; Zhao et al., 2025b; Chen et al., 2026b), yet they are rarely tested on whether they can
recover low-visibility precursors that later become central, leaving an open challenge for long-context reasoning
and abstraction control.
We address this challenge withBackTrend, a retrospective benchmark for scientific weak-signal prediction
(Figure 1). Given amature target topicand a temporally restricted evidence window, systems must recover early
problem-spaceandsolution-spaceweak signals that later became important precursors of that topic. Because
prospective evaluation would require waiting years for outcomes to mature, BackTrend uses backward reconstruction
as a controlled proxy for scientific foresight. We construct the reference set by reconstructing candidate precursors
directly from large-scale Semantic Scholar literature: for each mature target topic, we mine candidate research
directions from its 2019–2023 papers and score them by their 2019–2024 publication frequency, so that a candidate
is retained only when a low-visibility early emergence is followed by a clear rise in its 2024 frequency. AI/ML
researchers among the authors then validate every retained candidate for topical relevance, temporal consistency,
and supporting evidence; at evaluation time, systems must recover the held-out signals using only evidence from
publications available before the end-of-2023 prediction cutoff.
Our results show that strong systems can produce plausible precursors but routinely miss the benchmark’s
intended structure. Across all seven evaluated systems, semantic-judge F1 and Coverage Kremain low: F1 reaches
*Equal contributions.
arXiv:2609.24921v1  [cs.AI]  21 Sep 2026

Mature target topic
Large Language
Models
the topic to reconstructBenchmark construction
1Corpus construction  retrieve 2019 2024 papers (topic + paraphrase queries)
2Candidate topic discovery  mine & cluster problem/solution candidates
3Weak-signal identification & validation  score each candidate's
2019 2024 frequency trajectory, apply four gates, then expert-verify
yields the weak signals
Human-validated weak signals
showing the top-ranked of 11 signals (5 problem, 6 solution)
Problem-space
privacy leakage in
NLP modelsSolution-space
retrieval-augmented
language modelsGPT-5.4 prediction (illustrative)
input: evidence up to the 2023 cutoff only   ·   P / S = problem- / solution-space
prediction outcome reference weak signal
PMemorization, privacy
leakage, and
regurgitation from
web-scale dataprivacy leakage in
NLP models
PBias, toxicity, and
representational harm
in general-purpose
text generation fairness  detection
machine-generated
text detection
SRetrieval-augmented
generationretrieval-augmented
language models
ST ool-augmented
language modeling
tool use  retrieval
retrieval-augmented
language models match
 topic drift
 match
 near-miss
'19 '21 '23012341e3
'19 '21 '2301231e2
Figure 1: BackTrend overview for the 2024 mature target topiclarge language models.Top:the benchmark-
construction pipeline (§3.2), in which candidateproblem-spaceandsolution-spaceprecursors are mined from the
topic’s 2019–2023 papers, scored by their 2019–2024 frequency trajectory, gated, and expert-validated.Lower-left:
two of the topic’s eleven human-validated weak signals (the top-ranked problem- and solution-space precursor),
each with its 2019–2024 frequency trajectory, which shows the low-then-rising signature that defines a weak signal.
Lower-right:four real GPT-5.4 predictions, verbatim from the prediction outputs and produced under a pre-2024
evidence cutoff, each listed with its outcome and with the reference weak signal it was scored against. The full error
taxonomy, which also covers granularity mismatch and coverage failure, is defined in §5.2.
at most 10.1% Coverage Kreaches at most 18.5%. The dominant errors are topic drift, granularity mismatch, lexical
near-miss, and coverage failure. Additional evidence can improve performance, but not uniformly: deeper retrieval
generally benefits both RAG models, with a stronger effect on Qwen3-8B, while DeepResearch performs best with
a moderate search budget rather than unlimited search. Weak-signal prediction therefore tests semantic alignment,
abstraction-level control, and coverage over concrete research precursors rather than fluency in generating scientific
phrases.
Our main contributions are summarized below:
•We formulateretrospective scientific weak-signal prediction: recovering early problem-space and solution-space
precursors of mature research topics under a temporal evidence constraint that simulates research foresight.
•We construct BackTrend, a benchmark of 25 mature target topics in artificial intelligence and machine learning
and 66 human-validated weak signals, reconstructed from large-scale literature by scoring each candidate’s
2019–2024 publication-frequency trajectory and verified by AI/ML researchers among the authors.
•We benchmark frontier LLMs, RAG systems, and DeepResearch agents on BackTrend, finding that current
systems produce plausible but misaligned signals and that semantic-judge F1 reaches at most 10.1% for every
system and Coverage@ 10at most 18.5%, with errors dominated by topic drift, granularity mismatch, lexical
near-miss, and coverage failure.
2 Related Work
Benchmark Construction for Emerging Topics and Weak Signals.Existing benchmarks for emerging-topic
analysis mainly evaluate whether systems can detect and track salient events or trends over time, rather than identify
early scientific ideas that later develop into dominant paradigms (Allan, 2002; Petrovi ´c et al., 2010; Deng et al.,
2022). In scientific domains, benchmark construction has also focused on trajectory labeling or future impact
prediction, such as classifying topics as rising or declining or forecasting citations (Prabhakaran et al., 2016;
Moiseeva and Schütze, 2020; Ofer et al., 2024; Gu and Krenn, 2025; Ajith et al., 2026). Foresight and horizon-
scanning reports offer another related resource, but the signals they curate are usually broad thematic areas rather
than concrete technical precursors (Day and Schoemaker, 2005; Saritas, 2013; van Rij, 2010). BackTrend addresses
2

Figure 2: Overview of the BackTrend construction pipeline. We first compile 25 mature target topics from the
Artificial Intelligence and Machine Learning domain of a 2024 JRC report (Eulaerts et al., 2025) and retrieve their
2019–2024 papers from Semantic Scholar. We then mine problem-space and solution-space candidate topics from
the 2019–2023 abstracts, consolidate them by embedding clustering, and score each candidate by its 2019–2024
publication frequency, keeping only candidates that pass all four frequency gates. Finally, two AI/ML researchers
among the authors validate the surviving candidates for topical relevance, correct problem/solution categorization,
temporal consistency, and evidence faithfulness to produce the final BackTrend benchmark.
these limitations by constructing benchmark instances retrospectively from mature research topics, defining weak
signals as empirically grounded early precursors rather than transient events, broad themes, or short-term impact
patterns. Each instance decomposes into problem-space and solution-space precursors and is evaluated under a
prediction cutoff that withholds post-cutoff information from the system.
Weak Signal Detection and Emerging Trend Analysis.Weak-signal detection has long been studied in foresight,
bibliometrics, and technology intelligence as the task of identifying faint early indicators of future change (Ansoff,
1975; Hiltunen, 2008; Holopainen and Toivonen, 2012). Early computational approaches relied on keyword
frequencies, citation networks, and co-occurrence statistics, while later work adopted topic models and contextual
embedding methods to better capture latent semantic structure and thematic evolution over time (Small, 1973;
Yoon, 2012; Song et al., 2018; Blei et al., 2003; Rudolph and Blei, 2018; Yao et al., 2018; Grootendorst, 2022;
Boutaleb et al., 2024; Ebadi et al., 2026). However, these methods are still typically evaluated through case studies
or proxy trend metrics, because existing resources rarely provide verified examples of concrete early precursors
to mature scientific themes. BackTrend addresses this gap with a precursor-recovery task featuring validated
targets and a unified protocol. We benchmark LLM-based research agents and omit traditional bibliometric or
topic-modeling systems because their outputs are not directly aligned with BackTrend ’s output format: given a
mature topic, BackTrend requires an explicit, topic-conditioned set of precursor hypotheses, whereas these methods
typically produce corpus-level trends, clusters, or topic trajectories. Adapting them would require an additional,
non-standardized mapping from such outputs to target-specific precursor labels, potentially confounding direct
comparison. We leave such adaptations to future work.
3 BackTrend Benchmark
BackTrend is a retrospective weak-signal prediction benchmark in which systems recover early problem-space and
solution-space precursors of mature research topics. Below we define the task and the two weak-signal categories
(§3.1), detail the construction pipeline with expert validation (§3.2), and report per-topic statistics (§3.3).
3.1 BackTrend Task
The BackTrend task frames scientific foresight as recovering a mature topic’s early, low-visibility precursors from
evidence available before it matured, using backward reconstruction as a controlled proxy for prediction. We define
a precursor as an early research direction that later becomes associated with or central to the mature topic, without
implying a causal relationship.
Mature Target Topics.Given a scientific domain, mature target topics serve as anchor concepts for weak-signal
construction. Each mature target topic represents a research direction that has already achieved substantial visibility
within its domain through sustained publication activity. For example,large language modelsconstitute a mature
target topic in 2024. In BackTrend, these topics are instantiated from externally validated AI and machine-learning
topics in the 2024 JRC weak-signal report (Eulaerts et al., 2025); we describe the selection process in §3.2.
Problem-space and Solution-space Weak Signals.We define a weak signal as a concrete research direction
that initially received limited attention but later grew exponentially in prominence. We distinguish two types: (1)
3

problem-space weak signalsare underrecognized research problems or problem formulations that later become
central to the mature target topic in a given field; (2)solution-space weak signalsare early or niche methods,
techniques, or design principles that were not yet widely adopted but later became important solutions to already
recognized problems. In short, problem-space signals surface new questions, while solution-space signals offer
emerging answers to existing ones.
Task Formulation.We formulate weak-signal discovery as the task of identifying historically grounded early
research directions that later contribute to the emergence of a given mature target topic. Formally, given a domain-
specific mature target topic M, a maturity year y, an evidence window t= [y−k, y−1] , and a designated signal
space (problem or solution), the model must retrieve or infer a set of candidate weak signals s={s 1, . . . , s n}. A
valid weak signal si∈smust (i) exhibit low visibility during its early stage, (ii) demonstrate growth in prominence
within the evidence window, and (iii) have this rise confirmed in the maturity-year corpus of M, with its frequency
in year yexceeding its highest frequency within the evidence window by a factor of at least λ= 1.2 (§3.2), where λ
denotes the maturity-year frequency lift threshold. The year y−1 is referred to as theprediction cutoff year: at
evaluation time, systems are given the mature target topicMbut may only use evidence available up to the end of
this year, simulating a research-foresight setting in which evidence from the maturity year about how Memerged is
unavailable to the model. In BackTrend, mature topics are drawn from the 2024 JRC report (Eulaerts et al., 2025),
wherey= 2024andk= 5, resulting in an evidence window of 2019–2023 and a prediction cutoff year of 2023.
3.2 Benchmark Construction
We next detail the BackTrend construction process, with an overview shown in Figure 2. The pipeline is grounded
in observed publication frequency: a candidate is retained only when a low-visibility, exponentially growing early
emergence is followed by a rise in its frequency in the 2024 mature-topic corpus.
Corpus Construction.To identify weak signals retrospectively, we begin with a set of research topics that have
already reached maturity. We obtain these mature topics from the 2024 JRC weak-signal report (Eulaerts et al.,
2025), which identifies science and technology topics through large-scale data analysis and expert validation.
Specifically, we focus on topics within the Artificial Intelligence and Machine Learning domain and backtrack their
historical development to uncover the weak signals that preceded them. This results in a benchmark comprising 25
mature topics spanning a diverse range of AI and machine learning research directions. For each mature target topic
M, we retrieve papers through the Semantic Scholar API using keyword queries derived from the topic name and
its manually curated paraphrases, with one query per year from 2019 to 2024, to capture alternative terminology
and improve retrieval coverage. These paraphrase queries expand corpus coverage and reduce the risk of missing
historically relevant papers. The retrieved papers form the topic/paraphrase paper corpus, whose members we refer
to asM-papers. Papers from 2019–2023 serve as the historical corpus for candidate-topic discovery, while 2024
papers provide the later-year frequencies used to test whether an early candidate’s frequency subsequently rose.
Candidate Topic Discovery.We next extract candidate topics from the 2019–2023 historical corpus. For each
paper abstract retrieved for a mature target topic, we extract up to two reusable literature-level candidate topics that
are explicitly grounded in the abstract and conceptually related to the target topic; the extraction prompts are shown
in Figure 5 and Figure 6. Each candidate is assigned to one of two spaces:problem-spacecandidates describe
research problems, limitations, risks, gaps, evaluation failures, or scientific questions, whereassolution-space
candidates describe reusable methods, system directions, defenses, benchmarks, datasets, or evaluation protocols.
We require candidate labels to be neither overly broad field names nor paper-specific implementation details, and
we avoid problem-solution phrases that conflate a method with the problem it addresses.
The extracted candidates are then consolidated within each mature target topic and candidate type. We first
exact-deduplicate normalized candidate strings while retaining their supporting source-paper IDs, years, evidence
snippets, and mention counts. We then encode candidate-topic strings with text-embedding-3-large1and cluster
semantically similar candidates using a cosine-similarity threshold of 0.85, chosen to balance semantic consolidation
of related candidate topics with separation of distinct research directions. Clustering is performed separately for
problem-space and solution-space candidates to avoid merging research problems with solution methods. For each
cluster, we select a canonical candidate-topic label based on supporting evidence, prioritizing candidates with more
source papers and mentions while favoring compact labels when support is comparable. The resulting clustered
candidate topics are the units considered in the final weak-signal identification step.
Weak Signal Identification and Validation.For each clustered candidate topic cof mature target topic M, we
compute its yearly frequency over 2019–2024. Letn y(c)be the number of non-survey source papers supportingc
1https://platform.openai.com/docs/guides/embeddings
4

in yeary, andN y(M)the number of non-survey papers retrieved forMin yeary; the frequency is
fy(c;M) =ny(c)
Ny(M).
We exclude survey papers from both counts: a single survey touches many topics at once and would spuriously
inflate a candidate’s frequency, so excluding surveys reduces false-positive matches. However, we retain them
during candidate discovery and clustering because surveys provide broad coverage of research directions. We further
validate this design choice in Appendix A.4.
For 2019–2023, a paper supports cwhen its abstract matches c’s cluster. For 2024, we instead count a paper as
supportingcwhen itcitesat least one ofc’s early 2019–2023 source papers:
f2024(c;M) =r2024(c)
N2024(M),
where r2024(c)is the number of such non-survey 2024 M-papers. We measure the 2024 frequency through citations
rather than by re-matching c’s wording because a research direction is often renamed or rephrased as it matures, and
a direct text or embedding match in 2024 would miss these drifted mentions; a citation to the candidate’s own early
papers tracks the same line of work however it is now phrased.
A candidate is selected only if it passes four gates g1–g4. Letτc∈ {2019, . . . ,2022} be its onset year, selected
as the maximizer of the growth score below, and ϵa small constant. The gates use four hyperparameters: a
2024-frequency lift λ= 1.2 , a year-to-year retention ρ= 0.8 , a pre-onset tolerance δ= 0.6 , and a sparse-year skip
κ= 1. We first define the auxiliary quantities
Fmax= max 2019≤y≤2023 fy(c;M),
Fτ= max τc≤y≤2023 fy(c;M),
F<τ= max y<τcfy(c;M),
Ay(c) =1
fy+1(c;M) +ϵ≥ρ f y(c;M)
∨n y(c)≤κ
,
The growth score in g1fits an exponential trend to the frequencies over [τc,2023] and rewards a well-fitted,
sustained rise from a low base, with τcthe onset year that maximizes it; Appendix A.3 gives the full definition. We
then require all four gates to hold:
g1(c) =1[score(c)>0],
g2(c) =1[f 2024(c;M) +ϵ≥λF max],
gd
3(c) =(
1[f2023(c;M) +ϵ≥ρF τ], d= prob,
1hQ2022
y=τcAy(c) = 1i
, d= sol,
g4(c) =1[F <τ≤max(f τc(c;M), δf 2023(c;M))].
The four gates operationalize the three clauses of the definition in §3.1, each ruling out one failure mode. g1requires
the onset-window trajectory to fit a rising exponential, ruling out flat, declining, and single-point traces. g2requires
the rise to be confirmed in the maturity year, ruling out candidates whose 2024 frequency never clears their own early
peak. g3requires the growth to persist to the cutoff rather than collapse along the way, ruling out candidates that
spike early and then fade. g4requires the candidate to have been faint before its onset year, ruling out directions that
were already established when they began to grow. Together, g1,g3, andg4characterize the observable early-stage
properties of a weak signal, whileg 2validates whether the signal eventually contributes to the mature topic.
A candidate is retained when g1(c)g 2(c)gd
3(c)g 4(c) = 1 , with d= prob for problem-space candidates and
d= sol for solution-space candidates. After this automatic filtering step, two AI/ML researchers among the authors
conduct human validation: they inspect the candidate label, source papers, its 2024 citation evidence, and frequency
trajectory, then verify topical relevance, correct problem/solution categorization, temporal consistency, and evidence
faithfulness. Both experts independently label all 125 gate-passing candidates as valid weak signals or not, with
an inter-annotator agreement of Cohen’s κ= 0.75 . Candidates that fail validation are revised or removed through
consensus adjudication, yielding the final 66 signals.
Appendix A.2 traces a single candidate,retrieval-augmented language modelsunder the mature target topiclarge
language models, through every stage of the pipeline as a running example.
5

3.3 Data Statistics
BackTrend covers 25 mature target topics and 66 human-validated weak signals (34 problem-space and 32 solution-
space), all within the Artificial Intelligence and Machine Learning domain of the 2024 JRC report. Per-topic
statistics are reported in Table 1: the 66 signals are distributed over 18 of the 25 topics, while the remaining seven
yield no candidate that survives both the four gates of §3.2 and expert validation; §A.7 traces each empty set to
an identifiable property of the topic’s literature. At evaluation time, systems are asked to recover the weak signals
associated with each mature target topic under an end-of-2023 temporal search cut-off, using only historically
available information, simulating prospective foresight.
Mature Target Topic # Papers # P-WS # S-WS Mature Target Topic # Papers # P-WS # S-WS
Artificial Intelligence of Things 25.8K 1 0 Machine Unlearning 1.9K 2 2
Asynchronous Federated Learning 3.0K 2 1 Masked Face Recognition 1.9K 1 1
Attention Mechanisms in CNN 29.5K 3 5 Masked Language Model 5.4K 0 3
Decentralized Federated Learning 4.3K 1 1 Multimodal AI 8.6K 0 0
Epistemic AI 1.2K 0 0 Multimodal Hate Speech 0.1K 0 0
Evolutionary Neural Arch. Search 1.1K 1 0 Privacy-Preserving Machine Learning 11.3K 1 0
Explainable AI 18.8K 2 1 Scientific Machine Learning 35.4K 0 1
Federated Deep Learning 5.5K 0 0 Self-Supervised CNN 2.6K 0 0
Federated Machine Learning 6.8K 4 6 Tiny Machine Learning 3.6K 1 0
Federated Reinforcement Learning 1.6K 0 0 Trustworthy AI 29.1K 4 2
Human–AI Interface 22.4K 1 1 Trustworthy Machine Learning 26.6K 0 0
Human-Centric AI 7.2K 1 1 Vertical Federated Learning 1.0K 4 1
Large Language Models 22.8K 5 6
Total(grand total across all 25 topics)277.7K 34 32
Table 1: Data statistics of BackTrend. All 25 mature target topics fall in the Artificial Intelligence and Machine
Learning domain.# Papers: unique non-empty-abstract papers retrieved for the topic and its paraphrases over
2019–2024 (thousands; entries are rounded to 0.1K and need not sum exactly to the total).# P-WS/# S-WS:
number of human-validated problem- / solution-space weak signals. Seven topics yield no validated signal: five
produce no gate-passing candidate, and for two, expert validation removes every gate-passing survivor (§A.7).
4 Experiment Setup
We next discuss the evaluated systems and our automated and human evaluation protocols.
4.1 Evaluated Systems
We evaluate three categories of weak-signal prediction systems. All systems operate under the end-of-2023
prediction cutoff defined in §3.1, enforced through prompt instructions for parametric systems and through hard
filtering of retrieval results for systems that can search external corpora at inference time.
Frontier LLMs.We include GPT-5.4, Qwen3.5-397B-A17B (Qwen3.5-397B) (Qwen Team, 2026), and DeepSeek-
R1-0528 (Guo et al., 2025). Temporal consistency is enforced via explicit prompt instructions specifying the
emergence time of weak signals.
Retrieval-Augmented LLMs (RAG).We evaluate Qwen3-8B and Qwen3-30B (Qwen Team, 2025) with retrieval
augmentation, which are equipped with retrieval mechanisms to better ground predictions in evidence from historical
scientific literature. For each mature target topic, we retrieve a set of candidate papers from Semantic Scholar using
the topic as a query. The retrieved papers are encoded using the bge-base-en-v1.5 embedding model (Xiao et al.,
2024) and ranked by cosine similarity. The top 50 most relevant papers are then selected and provided as external
context to support weak-signal prediction. To ensure temporal consistency, retrieved papers are filtered on the client
side to include only those published on or before the end of 2023. We deliberately keep this retrieval stage simple;
stronger instruction-following or reasoning-intensive retrievers and rerankers (Song et al., 2025a,b; Zhao et al.,
2026a), as well as alternative ways of constructing retrieval queries when the mature target topic is unavailable,
could be explored in future work.
DeepResearch Systems.We evaluate two agentic LLM systems, DR-Tulu-8B (Shao et al., 2025) and Tongyi-
DeepResearch-30B-A3B (Tongyi-DR-30B-A3B) (Tongyi DeepResearch Team et al., 2026), which integrate multi-
step reasoning with tool-augmented academic retrieval and evidence collection. Tongyi-DR-30B-A3B follows a
ReAct-style framework, where the model iteratively performs reasoning and tool calls to query Semantic Scholar
and refine its predictions. In contrast, DR-Tulu adopts a workflow-based agentic search pipeline that orchestrates
model inference and retrieval through a modular multi-service architecture. In both systems, given a mature target
topic, the model retrieves relevant papers and generates weak signals grounded in the collected academic evidence.
6

ModelSet LLM Signal LLM Cov@10Human
Problem Solution All Problem Solution All Problem Solution All
Frontier LLMs
GPT-5.4 7.4 3.6 5.7 7.0 1.6 4.5 14.3 14.5 14.4 6.0
Qwen3.5-397B 6.5 4.0 5.3 4.7 1.1 3.0 7.5 9.5 8.4 5.3
DeepSeek-R1-052814.1 5.4 10.1 9.8 4.1 7.1 23.213.118.5 9.0
Retrieval-augmented LLMs
Qwen3-30B (RAG) 9.3 3.5 6.6 6.7 3.1 5.0 10.1 9.5 9.8 8.0
Qwen3-8B (RAG) 6.5 2.1 4.4 5.6 2.1 4.0 13.3 11.9 12.7 5.0
DeepResearch agents
Tongyi-DR-30B-A3B 6.7 4.3 5.6 5.2 3.2 4.3 14.1 6.2 10.4 6.0
DR-Tulu-8B 9.0 2.5 6.0 4.0 2.2 3.2 13.615.214.4 6.9
Table 2: Main results over the 30 topic-direction pairs with validated weak signals, contributed by 18 of the 25
AI/ML target topics. We report set-level, signal-level LLM-judge F1 (Claude-Opus-4.8, 3 runs) and Coverage@10
(Cov@10), broken down by signal direction (Problem / Solution / All), together with human-judged F1 (Human)
from our annotation interface. The highest score in each column is in bold.
To ensure temporal consistency, retrieval is restricted to papers published on or before the end of 2023, preventing
information leakage from future publications during the multi-step search process.
4.2 Evaluation Protocol
For each model, we provide a mature target topic Mand the 2019–2023 time window t, and ask the model to predict
the weak signals that emerged during that period. The prediction prompt templates, which ask every system for at
least ten weak signals ordered from most to least confident, are shown in are shown in Figure 7 and Figure 8. Let
the predicted weak-signal set be s={s 1, . . . , s m}and the human-validated reference set be g={g 1, . . . , g n}. We
report F1 under two LLM-judge settings with a strict matching criterion: two signals match only if they denote the
same specific research topic, differing at most in wording; related, adjacent, or same-broad-area topics do not count.
Inset-level LLM evaluation, the judge sees sandgjointly and estimates set-level precision and recall, measuring
alignment of the two sets as wholes. Insignal-level LLM evaluation, the judge decides for each predicted signal
whether it matches any reference signal, and symmetrically for each reference signal whether any prediction covers
it, yielding a finer-grained measure of coverage. We also reportCoverage@K, which measures the fraction of
validated reference signals that are covered by the model’s top- Kpredictions. Specifically, for each reference
signal, the judge determines whether it is matched by at least one of the top- Kpredictions, and Coverage@K is the
fraction of reference signals judged to be covered. The predictions are ordered from most to least confident, and
we set K= 10 . Of the 50 topic–direction pairs formed by the 25 mature target topics and the two signal spaces,
both settings score the 30 that carry at least one validated reference signal; the remaining 20 pairs have an empty
reference set and are excluded because F1 is undefined when g=∅ , 14 of them belonging to the seven topics with
no validated signal at all (§A.7) and six being single-direction gaps within signal-bearing topics. All judgments use
Claude-Opus-4.8, drawn from a model family independent of both the benchmark constructor and the evaluated
systems; the judging prompts are shown in Figure 9 and Figure 10, and each setting is run 3 times with scores
averaged. Embedding-similarity variants of these metrics reward shared domain vocabulary rather than same-topic
identity, so we defer them to Appendix A.9.
To independently validate the automatic evaluation, two annotators assess every system’s predictions across all
30 topic-direction pairs with validated weak signals using the same matching criterion as in the signal-level LLM
evaluation. We report a Cohen’s κof 0.81, indicating substantial agreement between the two annotators. Human F1
in Table 2 is aggregated as in the LLM-judge columns: F1 per pair, then averaged across pairs.
5 Experiments
We use BackTrend to evaluate frontier LLMs, RAG systems, and DeepResearch agents along three axes: overall
accuracy over the 30 topic-direction pairs with validated weak signals (§5.1), the failure modes that arise when
predictions miss the reference set (§5.2), and analyses of the retrieval and web-search budgets (§5.3).
5.1 Main Results
Table 2 shows that BackTrend is challenging even for strong LLM and DeepResearch systems. Under the LLM
judge, set-level F1 ranges from 4.4% to 10.1% and signal-level F1 from 3.0% to 7.1%, with the human-judged F1
7

showing a similar range. Coverage@10 ranges from 8.4% to 18.5%, substantially higher than the corresponding
signal-level F1 scores.
Problem signals are easier to recover.According to the Table 2, weak-signal discovery is substantially harder in
the solution space than in the problem space. All seven systems score higher on Problem than Solution under both
set-level, signal-level F1 and Coverage@10.
No simple capability hierarchy.Table 2 shows that weak-signal discovery does not follow a clear model-
capability hierarchy. DeepSeek-R1-0528 is consistently the strongest system across the LLM-judge metrics, but
other frontier LLMs, retrieval-augmented models, and DeepResearch agents are interleaved rather than cleanly
ordered by system type.
LLM judgments agree with human evaluation.The human largely agrees with the primary LLM judge:
DeepSeek-R1-0528 and Qwen3-30B (RAG) rank first and second under both evaluations, while the remaining
systems show broadly similar relative performance. This supports the reliability of the LLM-based evaluation.
5.2 Error Analysis and Case Study
We analyze GPT-5.4’s predictions on all 30 topic–direction pairs with validated weak signals and identify four main
error types:
(1) Topic drift.Topic drift refers to cases where the prediction remains relevant to the target topic but shifts to an
adjacent subproblem, meaning a related but different research question within the same literature that is not among
the validated precursors. Although such predictions share vocabulary and context with the target topic, they fail to
match the reference signals under the strict criterion of §4.2.
(2) Granularity mismatch.Granularity mismatch refers to cases where the prediction is plausible but at the wrong
level of abstraction relative to the benchmark.
(3) Lexical near-miss.Lexical near-miss refers to cases where the prediction is topically close to the benchmark
but semantically different.
(4) Coverage failure.Coverage failure refers to cases where the model recovers one slice of the benchmark but
misses other central dimensions.
The central challenge is thussemantic alignment with the intended precursor structure: matching the right
problem framing, abstraction level, and coverage. Appendix A.10 provides full examples of each error type,
including the reference signals, model predictions, and their confidence ranks.
Model Depth Set F1 Signal F1 Cov@10
Qwen3-8Bk=102.7 2.5 9.3
k=304.0 3.1 9.8
k=504.4 4.0 12.7
Qwen3-30Bk=105.2 4.110.4
k=303.5 3.5 9.4
k=506.6 5.09.8
Table 3: Retrieval-budget ablation (set-level / signal-level LLM-judge F1 and Coverage@ 10, Claude-Opus-4.8, 3
runs; percentages). Thek=50rows match the Qwen3 (RAG) rows of Table 2; best per model and metric in bold.
5.3 Analysis
We conduct two analyses: examining the validity of unmatched predictions and probing evidence budgets in RAG
and DeepResearch.
Unmatched predictions are often defensible.To assess how many unmatched predictions may represent valid
weak signals absent from the reference set, we sampled 100 predictions from the 589 deduplicated unmatched
predictions across all seven systems, stratified by system and signal direction. Two annotators independently
judged whether each prediction was a defensible 2019–2023 weak signal for the given topic and direction, without
access to the reference set. The two human annotators judged 57% and 49% of the predictions as plausible,
respectively. This suggests that a substantial fraction of predictions treated as false positives by the reference-based
evaluation may instead correspond to plausible weak signals that are absent from the reference set, indicating that
the reference-based F1 may underestimate system performance.
8

Retrieval budget in RAG.We re-generate RAG predictions at retrieval depths k∈ {10,30,50} and evaluate
them with the same LLM judge (Table 3). Both models achieve their best F1 atk=50, while Qwen3-8B improves
consistently with increasing retrieval depth. Coverage@ 10also increases for Qwen3-8B (9.3 to 12.7), whereas
Qwen3-30B shows little benefit beyond k=10 . These results suggest that retrieving more documents generally
improves RAG performance, with a particularly strong effect on the smaller model.
Web search rounds for DeepResearch.We evaluate Tongyi-DR-30B-A3B with search-round budgets b∈
{0,3,8,∞} (Table 4). Performance improves from no search to b=8 across the evaluated metrics, but declines
under the unlimited policy. Coverage@ 10increases from 6.6 at b=0 to 18.2 at b=8, before dropping to 10.4 with
unlimited search. Thus, additional search rounds improve performance up to a point, while removing the search
budget does not provide further gains.
Search budget Set F1 Signal F1 Cov@10
b=04.9 2.7 6.6
b=35.7 2.8 16.6
b=87.9 4.4 18.2
b=∞5.6 4.3 10.4
Table 4: Web-search-round ablation for Tongyi-DR-30B-A3B, where bis the number of search rounds the agent
may issue; b=∞ is the default policy of the main results, so its row equals the Tongyi row of Table 2. Set-level /
signal-level LLM-judge F1 and Coverage@10(Claude-Opus-4.8, 3 runs; percentages); best per metric in bold.
6 Conclusion
We introduced BackTrend to test a notion of scientific foresight, centered on identifying the specific precursors that
later mattered and recognizing when none exists. Our results expose less a knowledge gap than a discrimination
gap: systems surface content in the right neighborhood but struggle with abstraction level, distinguishing genuine
precursors from adjacent look-alikes, and covering a full direction. These failures persist with additional retrieval
and search, suggesting that the bottleneck lies in evidence abstraction rather than access to evidence. The same
conclusion holds under the more lenient Coverage@ K: even without penalizing unmatched predictions, the best
system recovers under a fifth of the reference signals. Scientific foresight therefore requires precise precursor
identification beyond simply generating relevant scientific content. Progress on BackTrend will depend less on
model scale or evidence budget than on abstraction-level control. More broadly, backward reconstruction offers a
reusable recipe for foresight benchmarks with verifiable ground truth, applicable wherever a field’s later record can
adjudicate what was once a weak signal.
7 Limitations
BackTrend’s 25 mature target topics are all drawn from the Artificial Intelligence and Machine Learning domain of
the 2024 JRC weak-signal report (§3.2), providing a controlled but bounded view of established AI/ML research
themes, so conclusions drawn from BackTrend should be read as evidence about how systems recover precursors
of well-documented AI/ML topics rather than a measure of free-form scientific foresight, and extending the topic
pool to other scientific domains as well as to longer-tail or pre-paradigmatic areas is a promising future direction.
Candidate precursors are mined from paper abstracts with a GPT-series LLM and then filtered by their 2019–2024
corpus frequency before expert validation, so the pipeline could inherit extraction- or labeling-specific phrasing
biases from that model; we mitigate this by requiring every retained signal to be grounded in verifiable Semantic
Scholar frequency and citation evidence rather than in model output alone, and by judging predictions with an
independent Claude-Opus-4.8 evaluator drawn from a different model family than both the constructor and the
evaluated systems (§3.2, §4.2). A direct sensitivity study with alternative extraction models is a natural next step.
The end-of-2023 evidence cutoff is enforced by hard filtering of retrieved papers for the RAG and DeepResearch
systems, but for parametric LLMs it can only be requested in the prompt (§4.1). Their pretraining corpora extend
past 2023, so we cannot rule out that a frontier model recalls how a target topic actually matured, and the foresight
setting is therefore simulated rather than guaranteed for these systems. Any such leakage would inflate rather than
depress the reported scores, so it does not weaken our central finding that every system scores at a low level; it does
mean the parametric numbers should be read as an optimistic bound, and a strict test would require models whose
pretraining cutoff precedes the prediction cutoff. Following the practice of large-scale benchmark efforts (Hendrycks
et al., 2021; Srivastava et al., 2023; Rein et al., 2023), all validators are co-authors, and we reduce single-reviewer
bias by running independent reviews before adjudication (§3.2); recruiting a broader pool of external domain experts
is a useful future extension.
9

Acknowledgments
We thank TCS Research and the Yale NLP Lab for their support and helpful feedback.
References
H. Igor Ansoff. Managing strategic surprise by response to weak signals.California Management Review, 18(2):
21–33, 1975.
Alan L. Porter, W. Bradford Ashton, Guenter Clar, Joseph F. Coates, Kerstin Cuhls, Scott W. Cunningham, Ken
Ducatel, Patrick van der Duin, Luke Georghiou, Theodore Gordon, Harold Linstone, Vincent Marchau, Gilda
Massari, Ian Miles, Mary Mogee, Ahti Salo, Fabiana Scapolo, Ruud Smits, and Wil Thissen. Technology futures
analysis: Toward integration of the field and new methods.Technological Forecasting and Social Change, 71(3):
287–303, 2004. doi: 10.1016/j.techfore.2003.11.004. URL https://doi.org/10.1016/j.techfore.
2003.11.004.
Yuya Kajikawa, Junta Yoshikawa, Yoshiyuki Takeda, and Katsumori Matsushima. Tracking emerging technologies
in energy research: Toward a roadmap for sustainable energy.Technological Forecasting and Social Change,
75(6):771–782, 2008. ISSN 0040-1625. doi: https://doi.org/10.1016/j.techfore.2007.05.005. URL https:
//www.sciencedirect.com/science/article/pii/S0040162507001266.
Janghyeok Yoon and Kwangsoo Kim. Detecting signals of new technological opportunities using semantic patent
analysis and outlier detection.Scientometrics, 90:445–461, 02 2012. doi: 10.1007/s11192-011-0543-2.
Sangjae Lee, Wanki Kim, Young Min Kim, Hyoung Yong Lee, and Kyong Joo Oh. The prioritization and
verification of IT emerging technologies using an analytic hierarchy process and cluster analysis.Technological
Forecasting and Social Change, 87(C):292–304, None 2014. doi: 10.1016/j.techfore.2013.12.029. URL
https://ideas.repec.org/a/eee/tefoso/v87y2014icp292-304.html.
Takaya Ogawa and Yuya Kajikawa. Assessing the industrial opportunity of academic research with patent re-
latedness: A case study on polymer electrolyte fuel cells.Technological Forecasting and Social Change,
90:469–475, 2015. ISSN 0040-1625. doi: https://doi.org/10.1016/j.techfore.2014.04.002. URL https:
//www.sciencedirect.com/science/article/pii/S0040162514001231.
Haiying Ren and Yuhui Zhao. Technology opportunity discovery based on constructing, evaluating, and search-
ing knowledge networks.Technovation, 101:102196, 2021. ISSN 0166-4972. doi: https://doi.org/10.1016/
j.technovation.2020.102196. URL https://www.sciencedirect.com/science/article/pii/
S0166497220300687.
Zhenfeng Liu, Jian Feng, and Lorna Uden. Technology opportunity analysis using hierarchical semantic networks
and dual link prediction.Technovation, 128:102872, 2023. ISSN 0166-4972. doi: https://doi.org/10.1016/
j.technovation.2023.102872. URL https://www.sciencedirect.com/science/article/pii/
S0166497223001839.
Yilun Zhao, Jinbiao Wei, Tingyu Song, Siyue Zhang, Chen Zhao, and Arman Cohan. Rethinking reasoning-
intensive retrieval: Evaluating and advancing retrievers in agentic search systems. In Maria Liakata, Viviane P.
Moreira, Jiajun Zhang, and David Jurgens, editors,Proceedings of the 64th Annual Meeting of the Association
for Computational Linguistics (Volume 1: Long Papers), pages 36776–36806, San Diego, California, United
States, July 2026a. Association for Computational Linguistics. ISBN 979-8-89176-390-6. doi: 10.18653/v1/
2026.acl-long.1705. URLhttps://aclanthology.org/2026.acl-long.1705/.
Tiansheng Hu, Yilun Zhao, Canyu Zhang, Arman Cohan, and Chen Zhao. SAGE: benchmarking and improving
retrieval for deep research agents.CoRR, abs/2602.05975, 2026. doi: 10.48550/ARXIV .2602.05975. URL
https://doi.org/10.48550/arXiv.2602.05975.
Zhaojian Yu, Kaiyue Feng, Yilun Zhao, Shilin He, Xiao-Ping Zhang, and Arman Cohan. Alpharesearch: Accelerat-
ing new algorithm discovery with language models.CoRR, abs/2511.08522, 2025. doi: 10.48550/ARXIV .2511.
08522. URLhttps://doi.org/10.48550/arXiv.2511.08522.
Yilun Zhao, Weiyuan Chen, Zhijian Xu, Manasi Patwardhan, Chengye Wang, Yixin Liu, Lovekesh Vig, and Arman
Cohan. AbGen: Evaluating large language models in ablation study design and evaluation for scientific research.
In Wanxiang Che, Joyce Nabende, Ekaterina Shutova, and Mohammad Taher Pilehvar, editors,Proceedings
of the 63rd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages
12479–12491, Vienna, Austria, July 2025a. Association for Computational Linguistics. ISBN 979-8-89176-251-0.
doi: 10.18653/v1/2025.acl-long.611. URLhttps://aclanthology.org/2025.acl-long.611/.
10

Zhijian Xu, Yilun Zhao, Manasi Patwardhan, Lovekesh Vig, and Arman Cohan. Can LLMs identify critical
limitations within scientific research? a systematic evaluation on AI research papers. In Wanxiang Che, Joyce
Nabende, Ekaterina Shutova, and Mohammad Taher Pilehvar, editors,Proceedings of the 63rd Annual Meeting of
the Association for Computational Linguistics (Volume 1: Long Papers), pages 20652–20706, Vienna, Austria,
July 2025. Association for Computational Linguistics. ISBN 979-8-89176-251-0. doi: 10.18653/v1/2025.
acl-long.1009. URLhttps://aclanthology.org/2025.acl-long.1009/.
Ziyu Chen, Yilun Zhao, and Arman Cohan. Measuring the gap between human and LLM research ideas, 2026a.
URLhttps://arxiv.org/abs/2607.01233.
Yilun Zhao, Kaiyan Zhang, Tiansheng Hu, Sihong Wu, Ronan Le Bras, Charles McGrady, Taira Anderson, Jonathan
Bragg, Joseph Chee Chang, Jesse Dodge, Matt Latzke, Yixin Liu, Xiangru Tang, Zihang Wang, Chen Zhao,
Hannaneh Hajishirzi, Doug Downey, and Arman Cohan. SciArena: An open evaluation platform for non-
verifiable scientific literature-grounded tasks. InThe Thirty-ninth Annual Conference on Neural Information
Processing Systems Datasets and Benchmarks Track, 2026b. URL https://openreview.net/forum?
id=am6RR85mnc.
Chengye Wang, Yifei Shen, Zexi Kuang, Arman Cohan, and Yilun Zhao. SciVer: Evaluating foundation mod-
els for multimodal scientific claim verification. In Wanxiang Che, Joyce Nabende, Ekaterina Shutova, and
Mohammad Taher Pilehvar, editors,Proceedings of the 63rd Annual Meeting of the Association for Com-
putational Linguistics (Volume 1: Long Papers), pages 8562–8579, Vienna, Austria, July 2025. Associa-
tion for Computational Linguistics. ISBN 979-8-89176-251-0. doi: 10.18653/v1/2025.acl-long.420. URL
https://aclanthology.org/2025.acl-long.420/.
Yilun Zhao, Chengye Wang, Chuhan Li, and Arman Cohan. Can multimodal foundation models understand
schematic diagrams? an empirical study on information-seeking QA over scientific papers. In Wanxiang
Che, Joyce Nabende, Ekaterina Shutova, and Mohammad Taher Pilehvar, editors,Findings of the Association
for Computational Linguistics: ACL 2025, pages 18598–18631, Vienna, Austria, July 2025b. Association
for Computational Linguistics. ISBN 979-8-89176-256-5. doi: 10.18653/v1/2025.findings-acl.957. URL
https://aclanthology.org/2025.findings-acl.957/.
Ziyu Chen, Yilun Zhao, Chengye Wang, Rilyn Han, Manasi Patwardhan, and Arman Cohan. SciMDR: Advancing
scientific multimodal document reasoning. In Maria Liakata, Viviane P. Moreira, Jiajun Zhang, and David
Jurgens, editors,Proceedings of the 64th Annual Meeting of the Association for Computational Linguistics
(Volume 1: Long Papers), pages 44718–44742, San Diego, California, United States, July 2026b. Association for
Computational Linguistics. doi: 10.18653/v1/2026.acl-long.2070. URL https://aclanthology.org/
2026.acl-long.2070/.
James Allan, editor.Topic Detection and Tracking: Event-based Information Organization, volume 12 ofThe
Information Retrieval Series. Springer US, Boston, MA, 2002. ISBN 978-1-4615-0933-2. doi: 10.1007/
978-1-4615-0933-2. URLhttps://doi.org/10.1007/978-1-4615-0933-2.
Saša Petrovi ´c, Miles Osborne, and Victor Lavrenko. Streaming first story detection with application to Twitter.
In Ron Kaplan, Jill Burstein, Mary Harper, and Gerald Penn, editors,Human Language Technologies: The
2010 Annual Conference of the North American Chapter of the Association for Computational Linguistics,
pages 181–189, Los Angeles, California, June 2010. Association for Computational Linguistics. URL https:
//aclanthology.org/N10-1021/.
Haolin Deng, Yanan Zhang, Yangfan Zhang, Wangyang Ying, Changlong Yu, Jun Gao, Wei Wang, Xiaoling Bai, Nan
Yang, Jin Ma, Xiang Chen, and Tianhua Zhou. Title2Event: Benchmarking open event extraction with a large-
scale Chinese title dataset. In Yoav Goldberg, Zornitsa Kozareva, and Yue Zhang, editors,Proceedings of the 2022
Conference on Empirical Methods in Natural Language Processing, pages 6511–6524, Abu Dhabi, United Arab
Emirates, December 2022. Association for Computational Linguistics. doi: 10.18653/v1/2022.emnlp-main.437.
URLhttps://aclanthology.org/2022.emnlp-main.437/.
Vinodkumar Prabhakaran, William L. Hamilton, Dan McFarland, and Dan Jurafsky. Predicting the rise and fall of
scientific topics from trends in their rhetorical framing. In Katrin Erk and Noah A. Smith, editors,Proceedings of
the 54th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 1170–
1180, Berlin, Germany, August 2016. Association for Computational Linguistics. doi: 10.18653/v1/P16-1111.
URLhttps://aclanthology.org/P16-1111/.
Alena Moiseeva and Hinrich Schütze. TRENDNERT: A benchmark for trend and downtrend detection in a scientific
domain.Proceedings of the AAAI Conference on Artificial Intelligence, 34(05):8512–8519, Apr. 2020. doi: 10.
1609/aaai.v34i05.6372. URLhttps://ojs.aaai.org/index.php/AAAI/article/view/6372.
Dan Ofer, Hadasah Kaufman, and Michal Linial. What’s next? forecasting scientific research trends.Heliyon,
10(1):e23781, 2024. ISSN 2405-8440. doi: https://doi.org/10.1016/j.heliyon.2023.e23781. URL https:
//www.sciencedirect.com/science/article/pii/S2405844023109893.
11

Xuemei Gu and Mario Krenn. Forecasting high-impact research topics via machine learning on evolving knowledge
graphs.Machine Learning: Science and Technology, 6(2):025041, 2025. doi: 10.1088/2632-2153/add6ef. URL
https://doi.org/10.1088/2632-2153/add6ef.
Anirudh Ajith, Amanpreet Singh, Jay DeYoung, Nadav Kunievsky, Austin C. Kozlowski, Oyvind Tafjord, James
Evans, Daniel S. Weld, Tom Hope, and Doug Downey. PreScience: A dataset and benchmark for scientific
forecasting, 2026. URLhttps://arxiv.org/abs/2602.20459.
George S. Day and Paul J. H. Schoemaker. Scanning the periphery.Harvard Business Review, 83(11):135–140,
142, 144–148, 11 2005.
Ozcan Saritas. Systemic Foresight Methodology. In Dirk Meissner, Leonid Gokhberg, and Alexander Sokolov,
editors,Science, Technology and Innovation Policy for the Future: Potentials and Limits of Foresight Studies,
pages 83–117. Springer Berlin Heidelberg, Berlin, Heidelberg, 2013. ISBN 978-3-642-31827-6. doi: 10.1007/
978-3-642-31827-6_6. URLhttps://doi.org/10.1007/978-3-642-31827-6_6.
Victor van Rij. Joint horizon scanning: identifying common strategic choices and questions for knowledge.
Science and Public Policy, 37(1):7–18, 02 2010. ISSN 0302-3427. doi: 10.3152/030234210X484801. URL
https://doi.org/10.3152/030234210X484801.
Elina Hiltunen. The future sign and its three dimensions.Futures, 40:247–260, 04 2008. doi: 10.1016/j.futures.
2007.08.021.
Mari Holopainen and Marja Toivonen. Weak signals: Ansoff today.Futures, 44(3):198–205, 2012. ISSN 0016-
3287. doi: https://doi.org/10.1016/j.futures.2011.10.002. URL https://www.sciencedirect.com/
science/article/pii/S0016328711002540. Special Issue: Weak Signals.
Henry Small. Co-citation in the scientific literature: A new measure of the relationship between two documents.
Journal of the American Society for Information Science, 24:265 – 269, 07 1973. doi: 10.1002/asi.4630240406.
Janghyeok Yoon. Detecting weak signals for long-term business opportunities using text mining of Web news.
Expert Systems with Applications, 39:12543–12550, 11 2012. doi: 10.1016/j.eswa.2012.04.059.
Kisik Song, Kyuwoong Kim, and Sungjoo Lee. Identifying promising technologies using patents: A retrospective
feature analysis and a prospective needs analysis on outlier patents.Technological Forecasting and Social Change,
128(C):118–132, None 2018. doi: 10.1016/j.techfore.2017.11.008. URL https://ideas.repec.org/a/
eee/tefoso/v128y2018icp118-132.html.
David M. Blei, Andrew Y . Ng, and Michael I. Jordan. Latent Dirichlet allocation.J. Mach. Learn. Res., 3(null):
993–1022, March 2003. ISSN 1532-4435.
Maja Rudolph and David Blei. Dynamic embeddings for language evolution. InProceedings of the 2018 World
Wide Web Conference, WWW ’18, page 1003–1011, Republic and Canton of Geneva, CHE, 2018. International
World Wide Web Conferences Steering Committee. ISBN 9781450356398. doi: 10.1145/3178876.3185999.
URLhttps://doi.org/10.1145/3178876.3185999.
Zijun Yao, Yifan Sun, Weicong Ding, Nikhil Rao, and Hui Xiong. Dynamic word embeddings for evolving
semantic discovery. InProceedings of the Eleventh ACM International Conference on Web Search and Data
Mining, WSDM 2018, page 673–681. ACM, February 2018. doi: 10.1145/3159652.3159703. URL http:
//dx.doi.org/10.1145/3159652.3159703.
Maarten Grootendorst. BERTopic: Neural topic modeling with a class-based TF-IDF procedure, 2022. URL
https://arxiv.org/abs/2203.05794.
Allaa Boutaleb, Jerome Picault, and Guillaume Grosjean. BERTrend: Neural topic modeling for emerging
trends detection. In Joel Tetreault, Thien Huu Nguyen, Hemank Lamba, and Amanda Hughes, editors,Pro-
ceedings of the Workshop on the Future of Event Detection (FuturED), pages 1–17, Miami, Florida, USA,
November 2024. Association for Computational Linguistics. doi: 10.18653/v1/2024.futured-1.1. URL
https://aclanthology.org/2024.futured-1.1/.
Ashkan Ebadi, Alain Auger, and Yvan Gauthier. WISDOM: An AI-powered framework for emerging research
detection using weak signal analysis and advanced topic modelling.Journal of Informetrics, 20(1):101759,
March 2026. ISSN 1751-1577. doi: 10.1016/j.joi.2025.101759. URL http://dx.doi.org/10.1016/j.
joi.2025.101759.
Olivier Eulaerts, Marcelina Grabowska, and Michela Bergamini. Weak signals in Science and Technologies –
2024. Technical Report EUR 40213, Publications Office of the European Union, Luxembourg, 2025. URL
https://publications.jrc.ec.europa.eu/repository/handle/JRC140959.
12

Qwen Team. Qwen3.5: Towards native multimodal agents, February 2026. URL https://qwen.ai/blog?
id=qwen3.5.
Daya Guo, Dejian Yang, Haowei Zhang, Junxiao Song, Peiyi Wang, Qihao Zhu, Runxin Xu, Ruoyu Zhang,
Shirong Ma, Xiao Bi, et al. DeepSeek-R1 incentivizes reasoning in LLMs through reinforcement learning.Na-
ture, 645(8081):633–638, 2025. doi: 10.1038/s41586-025-09422-z. URL https://doi.org/10.1038/
s41586-025-09422-z.
Qwen Team. Qwen3 technical report, 2025. URLhttps://arxiv.org/abs/2505.09388.
Shitao Xiao, Zheng Liu, Peitian Zhang, Niklas Muennighoff, Defu Lian, and Jian-Yun Nie. C-Pack: Packed
resources for general Chinese embeddings, 2024. URLhttps://arxiv.org/abs/2309.07597.
Tingyu Song, Guo Gan, Mingsheng Shang, and Yilun Zhao. IFIR: A comprehensive benchmark for evaluating
instruction-following in expert-domain information retrieval. In Luis Chiruzzo, Alan Ritter, and Lu Wang, editors,
Proceedings of the 2025 Conference of the Nations of the Americas Chapter of the Association for Computational
Linguistics: Human Language Technologies (Volume 1: Long Papers), pages 10186–10204, Albuquerque, New
Mexico, April 2025a. Association for Computational Linguistics. ISBN 979-8-89176-189-6. doi: 10.18653/v1/
2025.naacl-long.511. URLhttps://aclanthology.org/2025.naacl-long.511/.
Tingyu Song, Yilun Zhao, Siyue Zhang, Chen Zhao, and Arman Cohan. LimRank: Less is more for reasoning-
intensive information reranking. In Christos Christodoulopoulos, Tanmoy Chakraborty, Carolyn Rose, and Violet
Peng, editors,Proceedings of the 2025 Conference on Empirical Methods in Natural Language Processing, pages
20625–20639, Suzhou, China, November 2025b. Association for Computational Linguistics. doi: 10.18653/v1/
2025.emnlp-main.1041. URLhttps://aclanthology.org/2025.emnlp-main.1041/.
Rulin Shao, Akari Asai, Shannon Zejiang Shen, Hamish Ivison, Varsha Kishore, Jingming Zhuo, Xinran Zhao,
Molly Park, Samuel G. Finlayson, David Sontag, Tyler Murray, Sewon Min, Pradeep Dasigi, Luca Soldaini, Faeze
Brahman, Wen-tau Yih, Tongshuang Wu, Luke Zettlemoyer, Yoon Kim, Hannaneh Hajishirzi, and Pang Wei Koh.
DR Tulu: Reinforcement learning with evolving rubrics for deep research.arXiv preprint arXiv:2511.19399,
2025. URLhttps://arxiv.org/abs/2511.19399.
Tongyi DeepResearch Team, Baixuan Li, Bo Zhang, Dingchu Zhang, Fei Huang, Guangyu Li, Guoxin Chen,
Huifeng Yin, Jialong Wu, Jingren Zhou, Kuan Li, Liangcai Su, Litu Ou, Liwen Zhang, Pengjun Xie, Rui Ye,
Wenbiao Yin, Xinmiao Yu, Xinyu Wang, Xixi Wu, Xuanzhong Chen, Yida Zhao, Zhen Zhang, Zhengwei Tao,
Zhongwang Zhang, Zile Qiao, Chenxi Wang, Donglei Yu, Gang Fu, Haiyang Shen, Jiayin Yang, Jun Lin, Junkai
Zhang, Kui Zeng, Li Yang, Hailong Yin, Maojia Song, Ming Yan, Minpeng Liao, Peng Xia, Qian Xiao, Rui Min,
Ruixue Ding, Runnan Fang, Shaowei Chen, Shen Huang, Shihang Wang, Shihao Cai, Weizhou Shen, Xiaobin
Wang, Xin Guan, Xinyu Geng, Yingcheng Shi, Yuning Wu, Zhuo Chen, Zijian Li, and Yong Jiang. Tongyi
DeepResearch technical report, 2026. URLhttps://arxiv.org/abs/2510.24701.
Dan Hendrycks, Collin Burns, Steven Basart, Andy Zou, Mantas Mazeika, Dawn Song, and Jacob Steinhardt. Mea-
suring massive multitask language understanding, 2021. URLhttps://arxiv.org/abs/2009.03300.
Aarohi Srivastava, Abhinav Rastogi, Abhishek Rao, Abu Awal Md Shoeb, Abubakar Abid, Adam Fisch, Adam R.
Brown, Adam Santoro, Aditya Gupta, Adrià Garriga-Alonso, Agnieszka Kluska, Aitor Lewkowycz, Akshat
Agarwal, Alethea Power, Alex Ray, Alex Warstadt, Alexander W. Kocurek, Ali Safaya, Ali Tazarv, Alice Xiang,
Alicia Parrish, Allen Nie, Aman Hussain, Amanda Askell, Amanda Dsouza, Ambrose Slone, Ameet Rahane,
Anantharaman S. Iyer, Anders Andreassen, Andrea Madotto, Andrea Santilli, Andreas Stuhlmüller, Andrew
Dai, Andrew La, Andrew Lampinen, Andy Zou, Angela Jiang, Angelica Chen, Anh Vuong, Animesh Gupta,
Anna Gottardi, Antonio Norelli, Anu Venkatesh, Arash Gholamidavoodi, Arfa Tabassum, Arul Menezes, Arun
Kirubarajan, Asher Mullokandov, Ashish Sabharwal, Austin Herrick, Avia Efrat, Aykut Erdem, Ayla Karaka¸ s,
B. Ryan Roberts, Bao Sheng Loe, Barret Zoph, Bartłomiej Bojanowski, Batuhan Özyurt, Behnam Hedayatnia,
Behnam Neyshabur, Benjamin Inden, Benno Stein, Berk Ekmekci, Bill Yuchen Lin, Blake Howald, Bryan
Orinion, Cameron Diao, Cameron Dour, Catherine Stinson, Cedrick Argueta, César Ferri Ramírez, Chandan
Singh, Charles Rathkopf, Chenlin Meng, Chitta Baral, Chiyu Wu, Chris Callison-Burch, Chris Waites, Christian
V oigt, Christopher D. Manning, Christopher Potts, Cindy Ramirez, Clara E. Rivera, Clemencia Siro, Colin Raffel,
Courtney Ashcraft, Cristina Garbacea, Damien Sileo, Dan Garrette, Dan Hendrycks, Dan Kilman, Dan Roth,
Daniel Freeman, Daniel Khashabi, Daniel Levy, Daniel Moseguí González, Danielle Perszyk, Danny Hernandez,
Danqi Chen, Daphne Ippolito, Dar Gilboa, David Dohan, David Drakard, David Jurgens, Debajyoti Datta, Deep
Ganguli, Denis Emelin, Denis Kleyko, Deniz Yuret, Derek Chen, Derek Tam, Dieuwke Hupkes, Diganta Misra,
Dilyar Buzan, Dimitri Coelho Mollo, Diyi Yang, Dong-Ho Lee, Dylan Schrader, Ekaterina Shutova, Ekin Dogus
Cubuk, Elad Segal, Eleanor Hagerman, Elizabeth Barnes, Elizabeth Donoway, Ellie Pavlick, Emanuele Rodola,
Emma Lam, Eric Chu, Eric Tang, Erkut Erdem, Ernie Chang, Ethan A. Chi, Ethan Dyer, Ethan Jerzak, Ethan
Kim, Eunice Engefu Manyasi, Evgenii Zheltonozhskii, Fanyue Xia, Fatemeh Siar, Fernando Martínez-Plumed,
Francesca Happé, Francois Chollet, Frieda Rong, Gaurav Mishra, Genta Indra Winata, Gerard de Melo, Germán
13

Kruszewski, Giambattista Parascandolo, Giorgio Mariani, Gloria Wang, Gonzalo Jaimovitch-López, Gregor Betz,
Guy Gur-Ari, Hana Galijasevic, Hannah Kim, Hannah Rashkin, Hannaneh Hajishirzi, Harsh Mehta, Hayden
Bogar, Henry Shevlin, Hinrich Schütze, Hiromu Yakura, Hongming Zhang, Hugh Mee Wong, Ian Ng, Isaac
Noble, Jaap Jumelet, Jack Geissinger, Jackson Kernion, Jacob Hilton, Jaehoon Lee, Jaime Fernández Fisac,
James B. Simon, James Koppel, James Zheng, James Zou, Jan Koco ´n, Jana Thompson, Janelle Wingfield, Jared
Kaplan, Jarema Radom, Jascha Sohl-Dickstein, Jason Phang, Jason Wei, Jason Yosinski, Jekaterina Novikova,
Jelle Bosscher, Jennifer Marsh, Jeremy Kim, Jeroen Taal, Jesse Engel, Jesujoba Alabi, Jiacheng Xu, Jiaming
Song, Jillian Tang, Joan Waweru, John Burden, John Miller, John U. Balis, Jonathan Batchelder, Jonathan Berant,
Jörg Frohberg, Jos Rozen, Jose Hernandez-Orallo, Joseph Boudeman, Joseph Guerr, Joseph Jones, Joshua B.
Tenenbaum, Joshua S. Rule, Joyce Chua, Kamil Kanclerz, Karen Livescu, Karl Krauth, Karthik Gopalakrishnan,
Katerina Ignatyeva, Katja Markert, Kaustubh D. Dhole, Kevin Gimpel, Kevin Omondi, Kory Mathewson, Kristen
Chiafullo, Ksenia Shkaruta, Kumar Shridhar, Kyle McDonell, Kyle Richardson, Laria Reynolds, Leo Gao,
Li Zhang, Liam Dugan, Lianhui Qin, Lidia Contreras-Ochando, Louis-Philippe Morency, Luca Moschella,
Lucas Lam, Lucy Noble, Ludwig Schmidt, Luheng He, Luis Oliveros Colón, Luke Metz, Lütfi Kerem ¸ Senel,
Maarten Bosma, Maarten Sap, Maartje ter Hoeve, Maheen Farooqi, Manaal Faruqui, Mantas Mazeika, Marco
Baturan, Marco Marelli, Marco Maru, Maria Jose Ramírez Quintana, Marie Tolkiehn, Mario Giulianelli, Martha
Lewis, Martin Potthast, Matthew L. Leavitt, Matthias Hagen, Mátyás Schubert, Medina Orduna Baitemirova,
Melody Arnaud, Melvin McElrath, Michael A. Yee, Michael Cohen, Michael Gu, Michael Ivanitskiy, Michael
Starritt, Michael Strube, Michał Sw˛ edrowski, Michele Bevilacqua, Michihiro Yasunaga, Mihir Kale, Mike
Cain, Mimee Xu, Mirac Suzgun, Mitch Walker, Mo Tiwari, Mohit Bansal, Moin Aminnaseri, Mor Geva,
Mozhdeh Gheini, Mukund Varma T, Nanyun Peng, Nathan A. Chi, Nayeon Lee, Neta Gur-Ari Krakover, Nicholas
Cameron, Nicholas Roberts, Nick Doiron, Nicole Martinez, Nikita Nangia, Niklas Deckers, Niklas Muennighoff,
Nitish Shirish Keskar, Niveditha S. Iyer, Noah Constant, Noah Fiedel, Nuan Wen, Oliver Zhang, Omar Agha,
Omar Elbaghdadi, Omer Levy, Owain Evans, Pablo Antonio Moreno Casares, Parth Doshi, Pascale Fung, Paul Pu
Liang, Paul Vicol, Pegah Alipoormolabashi, Peiyuan Liao, Percy Liang, Peter Chang, Peter Eckersley, Phu Mon
Htut, Pinyu Hwang, Piotr Miłkowski, Piyush Patil, Pouya Pezeshkpour, Priti Oli, Qiaozhu Mei, Qing Lyu,
Qinlang Chen, Rabin Banjade, Rachel Etta Rudolph, Raefer Gabriel, Rahel Habacker, Ramon Risco, Raphaël
Millière, Rhythm Garg, Richard Barnes, Rif A. Saurous, Riku Arakawa, Robbe Raymaekers, Robert Frank,
Rohan Sikand, Roman Novak, Roman Sitelew, Ronan LeBras, Rosanne Liu, Rowan Jacobs, Rui Zhang, Ruslan
Salakhutdinov, Ryan Chi, Ryan Lee, Ryan Stovall, Ryan Teehan, Rylan Yang, Sahib Singh, Saif M. Mohammad,
Sajant Anand, Sam Dillavou, Sam Shleifer, Sam Wiseman, Samuel Gruetter, Samuel R. Bowman, Samuel S.
Schoenholz, Sanghyun Han, Sanjeev Kwatra, Sarah A. Rous, Sarik Ghazarian, Sayan Ghosh, Sean Casey,
Sebastian Bischoff, Sebastian Gehrmann, Sebastian Schuster, Sepideh Sadeghi, Shadi Hamdan, Sharon Zhou,
Shashank Srivastava, Sherry Shi, Shikhar Singh, Shima Asaadi, Shixiang Shane Gu, Shubh Pachchigar, Shubham
Toshniwal, Shyam Upadhyay, Shyamolima, Debnath, Siamak Shakeri, Simon Thormeyer, Simone Melzi, Siva
Reddy, Sneha Priscilla Makini, Soo-Hwan Lee, Spencer Torene, Sriharsha Hatwar, Stanislas Dehaene, Stefan
Divic, Stefano Ermon, Stella Biderman, Stephanie Lin, Stephen Prasad, Steven T. Piantadosi, Stuart M. Shieber,
Summer Misherghi, Svetlana Kiritchenko, Swaroop Mishra, Tal Linzen, Tal Schuster, Tao Li, Tao Yu, Tariq
Ali, Tatsu Hashimoto, Te-Lin Wu, Théo Desbordes, Theodore Rothschild, Thomas Phan, Tianle Wang, Tiberius
Nkinyili, Timo Schick, Timofei Kornev, Titus Tunduny, Tobias Gerstenberg, Trenton Chang, Trishala Neeraj,
Tushar Khot, Tyler Shultz, Uri Shaham, Vedant Misra, Vera Demberg, Victoria Nyamai, Vikas Raunak, Vinay
Ramasesh, Vinay Uday Prabhu, Vishakh Padmakumar, Vivek Srikumar, William Fedus, William Saunders,
William Zhang, Wout V ossen, Xiang Ren, Xiaoyu Tong, Xinran Zhao, Xinyi Wu, Xudong Shen, Yadollah
Yaghoobzadeh, Yair Lakretz, Yangqiu Song, Yasaman Bahri, Yejin Choi, Yichi Yang, Yiding Hao, Yifu Chen,
Yonatan Belinkov, Yu Hou, Yufang Hou, Yuntao Bai, Zachary Seid, Zhuoye Zhao, Zijian Wang, Zijie J. Wang,
Zirui Wang, and Ziyi Wu. Beyond the imitation game: Quantifying and extrapolating the capabilities of language
models, 2023. URLhttps://arxiv.org/abs/2206.04615.
David Rein, Betty Li Hou, Asa Cooper Stickland, Jackson Petty, Richard Yuanzhe Pang, Julien Dirani, Julian
Michael, and Samuel R. Bowman. GPQA: A graduate-level Google-proof Q&A benchmark, 2023. URL
https://arxiv.org/abs/2311.12022.
Jean-Baptiste Alayrac, Jeff Donahue, Pauline Luc, Antoine Miech, Iain Barr, Yana Hasson, Karel Lenc, Arthur
Mensch, Katherine Millican, Malcolm Reynolds, Roman Ring, Eliza Rutherford, Serkan Cabi, Tengda Han,
Zhitao Gong, Sina Samangooei, Marianne Monteiro, Jacob Menick, Sebastian Borgeaud, Andy Brock, Aida
Nematzadeh, Sahand Sharifzadeh, Mikolaj Binkowski, Ricardo Barreira, Oriol Vinyals, Andrew Zisserman, and
Karen Simonyan. Flamingo: a visual language model for few-shot learning. InAdvances in Neural Information
Processing Systems, volume 35, 2022.
A Appendix
A.1 Expert Information
In this section, we provide an overview of the experts that contribute to the weak-signal validation and the human
evaluation of model predictions. All experts are authors of the paper, so we do not provide monetary compensation
14

to the experts.
Expert Level Domain Contribution
Expert 1 Postdoctoral researcher AI & ML Weak-signal validation
Expert 2 4th-year PhD student AI & ML Weak-signal validation
Expert 3 2nd-year PhD student AI & ML Human evaluation of model predictions
Expert 4 2nd-year MS student AI & ML Human evaluation of model predictions
Table 5: Experts involved in BackTrend. Experts 1–2 independently validate the gate-passing weak-signal candidates
(§3.2); Experts 3–4 independently annotate all model predictions in the human evaluation (Table 2). All 25 target
topics fall in the Artificial Intelligence and Machine Learning domain, and all experts are AI/ML researchers among
the authors; the two weak-signal validators (Experts 1–2) are the senior members of this group (§3.2).
A.2 A Running Example Through the Pipeline
As a running example, take the mature target topiclarge language models. Corpus construction retrieves 22.8K
papers for the topic and its paraphrases over 2019–2024 (Table 1). Candidate discovery mines solution-space
labels from the 2019–2023 abstracts, among themretrieval-augmented language models, and consolidation merges
its surface variants into a single cluster. Frequency computation then yields the trajectory plotted in Figure 4:
normalized to its own early peak, the candidate stands at 0.03 in 2019 and rises through 0.17,0.25, and 0.45 to1.00
in 2023, then reaches 2.95 in 2024, measured through citations to its early papers. All four gates pass, since the
early trace is faint, gap-free, and exponentially rising, and the 2024 lift of 2.95 clears λ= 1.2 ; expert validation
retains it, and it becomes one of the topic’s eleven validated signals. Figure 1 shows GPT-5.4 recovering exactly this
signal, phrased asretrieval-augmented generation, while missing others.
A.3 The Growth Score of Gateg 1
Gate g1of §3.2 tests score(c)>0 ; we define that score here. It is evaluated once per candidate onset year
τ∈ {2019, . . . ,2022} , and the candidate’s score and onset year are the maximum and the maximizer over τ, ties
broken toward the earliest onset.
Fix an onset τand write vτ= (f τ, . . . , f 2023)for the frequency vector over the onset window, dropping the
arguments of fy(c;M) . LetbτandR2
τbe the slope and the coefficient of determination of the least-squares fit of
log(f y+ϵ) against y−τ overvτ, so that bτis the fitted exponential growth rate and R2
τmeasures how well a pure
exponential describes the trajectory. Let f+
τandmτbe the first nonzero entry and the number of nonzero entries of
vτ, letF τ= max τ≤y≤2023 fyandF <τ= max y<τfywithF <2019 = 0, and set
γτ=f2023+ϵ
f+
τ+ϵ, π τ=1
2023−τP2022
y=τ1[fy+1> fy], η τ= min 
1,f2023+ϵ
Fτ+ϵ
,
θτ= min 
1,f2023+ϵ
f2022+ϵ
, ν τ= min 
1,mτ
32, ω τ= min 
1,fτ+ϵ
F<τ+ϵ
.
These factors measure, in order, the total growth over the window, the fraction of years in which the frequency
increases, how close the final year is to the window peak, how close it is to the preceding year, whether the candidate
is supported in at least three distinct years, and how faint the candidate is at onset relative to anything before it. The
onset score is
sd
τ(c) =

0, b τ≤0,
(R2
τ)2log+γτπτητωτ, d= prob,
(R2
τ)3log+γτπτητθτντω2
τ, d= sol,
wherelog+γ= logγforγ >1and0otherwise, and
score(c) = max
τsd
τ(c), τ c= arg max
τsd
τ(c).
The solution-space form is the stricter of the two: it additionally requires support in several years and penalizes a
final-year dip and pre-onset visibility more heavily, matching the fact that a method is adopted gradually whereas a
problem can be posed in a single influential paper.
Two properties of this construction matter for how g1behaves. First, every factor is non-negative, so score(c)>0
holds exactly when all of them are positive: the fitted growth rate must satisfy bτ>0, the trajectory must grow
overall ( γτ>1) and rise in at least one year ( πτ>0), the exponential fit must be non-degenerate ( R2
τ>0), and
the candidate must be present in 2023. Flat, declining, and single-point traces therefore score zero at every onset.
Second, because the gate tests only positivity, the exponents on R2
τandωτaffect only how surviving candidates are
ranked, never which ones pass.
15

A.4 Analysis of Survey Paper Exclusion Strategy
The frequency fy(c;M) of §3.2 counts non-survey papers only, so surveys are removed at the counting stage
rather than before candidate extraction. To examine this choice, we rerun the full pipeline over all 25 topics with
survey-titled papers excluded before candidate extraction.
Surveys do not contribute frequency evidence under either setting: ny(c)andNy(M) count non-survey papers
only. For the 99.68% of candidates whose cluster composition is unchanged, the entire 2019–2023 trajectory is
identical across the two runs. However, surveys can provide additional terminology during candidate discovery:
survey abstracts often summarize multiple research directions explicitly, supplying candidate labels that help clusters
match additionalnon-surveypapers. Excluding surveys earlier removes vocabulary from 453clusters, and 94of
these lose non-survey supporting papers as a result, whereas unchanged clusters lose none.
Under our benchmark, the earlier exclusion strategy recovers fewer weak signals ( 117vs.125) and fewer
human-validated signals ( 61vs.66). Survey-titled papers constitute only 1.36% of the retrieved corpus, so the
overall effect is limited; nevertheless, the counting-stage exclusion separates the two roles of surveys: they assist
candidate discovery while not contributing to the evidence used for validation.
A.5 Empirical Validation of the Growth Criterion
Unlike a purely qualitative judgment, BackTrend’s growth requirement is enforced directly during construction
rather than checked post hoc. A candidate is retained only if its 2024 frequency exceeds its peak 2019–2023
frequency by a factor of λ= 1.2 (gate g2in §3.2), so every validated weak signal is, by construction, more frequent
in 2024 than at any point during its early emergence window. Expert validation then confirms that this measured
rise reflects a genuine research trajectory rather than a measurement artifact. Because the criterion is embedded in
selection, all 66 validated signals satisfy the growth condition.
A.6 Two Cases That Warrant Explanation
A weak signal is defined relative to a mature target topic’s own corpus and to a designated signal space (§3.1).
Under this definition, two cases in the final 66-signal set may look surprising at first, and we explain them here.
A mature target topic can itself be a weak signal of another topic.Machine unlearning is one of our 25 mature
target topics, yet it is also a validated problem-space weak signal of privacy-preserving machine learning. The
underlying reason is that the JRC mature topics are not all at the same level of granularity: the report mixes broad
research areas with much narrower directions, so a narrower topic can sit inside a broader one. Our definition
handles this cleanly, because maturity and weak-signal frequency are measured in different corpora. A topic’s
maturity is established within its own corpus, whereas its weak-signal frequency is measured within the target topic’s
corpus (§3.2). As a standalone field machine unlearning is mature, but within the broader privacy-preserving-ML
literature it was a low-visibility precursor over 2019–2023 whose frequency rose sharply into 2024, which is exactly
the kind of signal the benchmark is meant to recover.
A single direction can occupy both signal spaces of one topic.Within machine unlearning, federated unlearning
appears as a weak signal in both the problem space and the solution space, the only such case among the 66 signals.
The two occurrences are reasonable because they carry a different focus. As a problem-space signal, federated
unlearning is the open question of how to guarantee that a client’s contribution can be removed from a trained
federated model and how to verify that the removal took place. As a solution-space signal, it is a concrete method
that erases a client’s data without retraining the model from scratch. To confirm that these are genuinely two signals
rather than one label counted twice, we inspected their supporting paper pools and found them disjoint, so each
occurrence rests on its own distinct evidence, and expert validation retained both.
A.7 Why Seven Mature Topics Carry No Certifiable Weak Signal
Of the 25 mature target topics, 66 validated weak signals are distributed over 18 topics, while seven carry none
(Table 1). All 25 target topics, absent and present alike, were fixed in advance by the same external JRC report
(§3.2); which of them turn out empty is determined by the construction alone. Nor is an empty set anomalous
under our definition. A weak signal is a discrete research direction that (i) was of low visibility early, (ii) grew
exponentially within the 2019–2023 window, and (iii) subsequently rose in the mature topic’s 2024 corpus (§3.1):
clauses (i)–(ii) require the emergence to be measurable in the historical record, so that a foresight system at the
end-2023 cutoff could in principle detect it, and clause (iii) verifies that the emergence was real. A topic can
therefore mature along paths that leave no such trace in its own corpus: by aggregating sub-areas that were already
visible, so nothing satisfies (i); by drawing its 2024 growth from work outside the topic’s corpus boundary, so
nothing satisfies (ii)–(iii) together; or by advancing on a corpus too small to measure any yearly frequency above
noise, so nothing satisfies (ii) credibly. This subsection shows that the seven empty sets are of exactly this structural
kind by ruling out each stage of the construction in order: the paper pools, the candidate clustering, the four gates,
16

and expert validation. Five of the seven nulls are decided by the same external g2frequency gate that certifies the
other 18 topics and two by the same expert criteria that curate every retained signal; for none does the pipeline find
a candidate satisfying the definition with a measurable 2024 rise, so we retain them with documented empty signal
sets.
Searching harder changes nothing.The most natural objection is that these topics were simply not searched
thoroughly enough. We therefore re-ran retrieval for every topic with substantially expanded paraphrase-query sets,
after a semantic-noise cleanup of overly broad queries, and re-ran theentirepipeline on the enlarged pools: candidate
extraction, clustering, frequency computation, and gating are all recomputed from scratch. The pools of four of the
five gate-failing topics grew by +24% to+93% , and those of the two expert-null topics by +37% and+232% , yet
none produced a validated weak signal; the fifth gate-failing topic,Federated RL, grew by under1%, so its corpus
is intrinsically small; and a twelve-variant leave-one-in paraphrase sweep for the most fragile case,Epistemic AI,
produced zero signals throughout. Because the deciding lift is a ratio of within-corpus frequencies, only a shift
in the topicalcompositionof the yearly corpora could move it; the expansion shifts composition substantially and
moved none of these topics across the gate.
Re-merging clusters at any granularity changes nothing.A subtler objection holds that the cosine- 0.85
consolidation (§3.2) shattered one genuine direction into near-duplicate labels, each individually too sparse to
certify. We therefore re-merged the final clusters of every absent topic by single-linkage in the original embedding
space at seven coarser thresholds, from 0.80 down to 0.50, recomputing each merged union’s trajectory exactly,
as deduplicated unions of supporting and 2024 citing papers under the pipeline’s counting rules. Single-linkage
merging and a permissive reading of the four gates both favor the objection, yet of the 5,411 merged unions
produced, only eleven union–threshold instances pass, all in the narrow band 0.80–0.75 and none below: coarser
merging adds early-window mass faster than 2024 mass, so the low-visibility requirement collapses first. Each
of the eleven reduces, on inspection, to a case adjudicated below. In nine, a single member carries 86–100% of
the union’s 2024 citations and every other member is a single-year blip contributing 0–2citations; the sharpest,
a few-/zero-shot multimodal learning union with trace 0,0,0,1,5 , drawsall29 of its 2024 citations from one
member’s single 2022 paper, Flamingo (Alayrac et al., 2022). Such a union staples four unrelated blips onto
the citation halo of an already-famous paper, the late-arriving pattern analyzed below. The remaining two pool
themultimodal reasoningfamily andFederated DL’s drift-and-heterogeneity vocabulary, and inherit those cases’
defects: the merged reasoning trace sits at 48% of its own peak frequency already in 2019, and the merged drift
union’s frequencyfallsinto the cutoff, 0.0217→0.0180 . A genuinely fragmented emergence would spread its 2024
citations across the reunited fragments and rise as a whole; no union at any granularity does. The sparsity is a
property of these literatures themselves.
The gates decide the five automatic nulls and separate the two groups cleanly.With the pools and the clustering
ruled out, the decision rests with the four gates of §3.2, which operationalize the three defining clauses: g1tests for
an exponentially rising early trajectory, g4for low pre-onset visibility, g3for persistent growth, and the decisive
g2requires a candidate’s 2024 frequency to exceed its own early peak by a factor λ= 1.2 . Gate g2is the only
externalcheck, grounded in the candidate’s independently measured 2024 frequency, and it is the one that decides
the five automatic nulls. Across the 25 topics the pipeline mines 133,426 candidate topics; for each topic we take
itsbestlow-visibility precursor, the candidate maximizing the 2024-frequency liftf 2024(c)/F max(c)among those
already passing g1andg4, and compare it against the g2threshold (Figure 3). All 18 signal-bearing topics clear
the threshold with best lifts of 1.60–28.7; each yields 2–24gate-passing candidates, from which expert validation
retains the final 66 signals. Five of the seven absent topics fallbelowthe threshold, all at best lift ≤1.05 . This
statistic is generous to them, since a per-topic maximum over many noisy per-candidate ratios is biasedupwardby
selection. The instrument itself holds up. Renaming cannot produce a null: the 2024 frequency is measured through
citations to the candidate’s early papers and follows the line of work however it is later phrased (§3.2); the one
thing no within-corpus measure can follow is work migrating across thetarget topic’scorpus boundary. This is by
design, since the task asks for precursorsof Mcertified in M’s own literature. The asymmetry of the measurement,
in-corpus matching for 2019–2023 and citation-based for 2024, cannot manufacture the shortfalls either: within
these very corpora the same instrument registers large 2024 rises for other candidates, lift 10.1 with CI [1.3,75.5] in
Multimodal AIand lift 10.0 inFederated DL, both from the riser audit below. What the gate actually observes is
decline: the point estimates fall for four of the five topics and are flat for the fifth,Federated RL, and where the
counts are large enough the decline is statistically certified below. Rejecting a topic whose best precursor shows
no 2024 rise is the same gate that guarantees the growth property for the present topics (Section A.5) behaving
correctlyunder the definition. One route remains open: a genuine riser could in principle have been discarded by
theothergates; the audit below closes it.
17

0.2 0.5 1 1.5 2 5 10 30Multimodal Hate SpeechEpistemic AIMultimodal AISelf-Supervised CNNFederated RLDecentralized FLPrivacy-Preserving MLAsync FLAIoTTrustworthy MLTiny MLScientific MLEvolutionary NASMasked LMVertical FLTrustworthy AIHuman-Centric AIHuman–AI InterfaceFederated MLMasked Face Rec.Attention in CNNFederated DLExplainable AIMachine UnlearningLLMs
λ= 1.2
Best 2024-frequency lift of a low-visibility precursor 
max c:g1∧g4f2024(c)/F max(c)
, log scale
Figure 3:Why seven mature topics carry no certifiable weak signal: absence is decided by the same 2024-
frequency gate that certifies the present topics.For each of the 25 target topics we plot thelargest2024-frequency
lift achieved by any candidate passing the low-visibility and exponential-growth gates ( g1∧g4); a validated weak
signal requires this lift to clear gate g2(λ= 1.2 , dashed line). •All 18 signal-bearing topics clear the line
(1.60–28.7).•Five absent topics fall below it (all ≤1.05 , even though a per-topic maximum over noisy candidate
ratios is biased upward): point estimates decline for four and are flat for Federated RL ( 1.05, CI[0.38,2.92] ); §A.7
additionally audits every candidate above the line that the other gates excluded. ▲Two absent topics clear the line,
but expert validation removes every surviving candidate: none of their year-by-year traces is a genuine exponential
rise (§A.7). Topic names are abbreviated on the axis; Table 1 lists them in full.
The gates discard no certifiable riser.The lift statistic above conditions on g1∧g4; could those gates themselves
be hiding a genuine signal? We therefore audit, for the five gate-failing topics,everymined candidate whose
measured 2024 frequency clears λwhen all other gates are ignored: 66 candidates in total, a count that only
coincidentally equals that of the validated signals. Each falls into one of three classes.(a) Single-year blips(57 of
66): candidates whose entire 2019–2023 record is at most three papers, all in a single year. The sharpest example
isdomain-specific multimodal reasoning(Multimodal AI): one 2023 paper, 18 citing papers in 2024, lift 10.1,
nominally even a statistically significant rise ( 95% confidence interval [1.3,75.5] ; Katz log-ratio interval for a ratio
of two binomial proportions, here and throughout). It still certifies nothing about emergence: growth, exponential
or otherwise, is not observable from a single point, so clause (ii) fails outright. Admitting such candidates would
turn ground truth into a post-hoc lottery: 86–92% ofallcandidates mined for these five topics have single-year
records ( 3,943 ofMultimodal AI’s 4,469 ), only 0.5–6%of those happen to rise in 2024, and at the end-2023 cutoff
nothing in the record separates the eventual winners from the rest. No validated signal is of this form: all 66 kept
signals have at least two nonzero in-window years.(b) Sparse broken traces(8 of 66): two or three nonzero years at
the 1–2-paper level whose trace is gapped ( 1,0,0,0,1 ;0,0,1,0,1 ) or flat-to-falling in frequency, so no positive
frequency-growth fit exists; these are exactly the shapes expert validation removes wherever they do pass the gates
(next paragraph).(c) Already visible(1 of 66):multimodal reasoning(Multimodal AI), at 61% of its own peak
frequency already in 2019, violating clause (i); its lift of 1.36 moreover carries a CI of [0.57,3.28] , so even its rise
is uncertifiable. Extending the audit to the two expert-null topics adds no new class: beyond the nine survivors
reviewed next, their 84 unconditional risers decompose into 67 single-year blips, 12 gapped 1–2-paper traces, four
candidates whosefrequencyis flat or falling even as raw counts grow, and one g4-rejected candidate,contrastive
learning in federated learningat lift 10.0, which anticipates the natural experiment below: the same direction passes
every gate in the siblingFederated MLcorpus and is kept there. Across all seven topics, the gates therefore discard
nothing that our definition, or any definition requiring emergence to be measurable in the 2019–2023 record, would
admit.
Expert validation decides the remaining two, on the grounds it applies everywhere.Trustworthy MLand
Federated DLclear the gates with 6 and 3 automatic survivors, and expert validation removes every one. This
stage applies the validation criteria of §3.2 everywhere; miscategorization and fixable evidence issues are handled
by revision, so a survivor isremovedonly on two grounds: (i) it does not name an AI/ML research direction, or
(ii) its year-by-year trajectory lacks the low-then-exponentially-rising shape the definition requires (§3.1). Across
the 18 signal-bearing topics these grounds keep 66 of 116 survivors; ground (i) accounts for a single removal,
18

transparent face masksunderMasked Face Recognition, which passes every statistical gate yet names a physical
object; every other removal is ground (ii). Ground (ii) judges the full trace shape, and its footprint is auditable. The
59 removed survivors comprise the 50 from the signal-bearing topics plus the nine here; 44 of them,75%, contain
an interior zero year, versus 2 of the 66 kept signals, and the other 64 kept traces run gap-free from onset to 2023.
All nine survivors of the two absent topics fall to ground (ii). EachTrustworthy MLsurvivor rests on one or two
papers per nonzero year with multi-year gaps, traces such as 1,0,0,0,1 or0,1,1,0,1 over 2019–2023: on counts
this small the log-space fit behind g1can turn positive, but the trace is visibly noise. In a corpus of 2,200 –4,600
papers per year over 2019–2023, that is itself the structural finding: no genuine precursor ever concentrates under
the umbrella vocabulary.Federated DL’s survivors fail identically:client driftrises in raw counts but is flat
in frequency, 0.0032→0.0041→0.0033 , whileprototype-based FLandbatch normalization in FLboth trace
0,0,1,0,3 .Prototype-based FLis instructive: thesameprecursor rises monotonically in the broaderFederated ML
corpus, tracing 0,0,0,1,3 at lift 2.13, and is keptthereas a validated signal; its certifiable rise simply lives in the
sibling’s corpus.
Each absence traces to a measurable structural cause.Table 6 assigns each absent topic to its dominant
cause, and Figure 4 shows the mechanism directly by plotting normalized frequency trajectories. (1)Umbrella or
narrowly scoped topics whose gate-passing survivors are sparse-count artifacts:Trustworthy MLandFederated
DL, as established above. (2)Scope migration: topics whose growth crosses the corpus boundary. Multimodal AI’s
vocabulary consolidated only recently: its corpus quadruples from 448 papers in 2019 to 1,924 in 2023, and its
2024 growth is carried by directions with no in-corpus history. Of its 29 unconditional risers, 25 are the single-year
blips of class (a), 19 of them entering only in 2023, yet drawing 3–29 citing 2024 papers each. Sub-areas such as
visual language models and vision-language instruction tuning arrive in the corpus already formed, which is why its
best low-visibility precursor reaches a lift of only 0.56.Self-Supervised CNNshows the outbound migration: its
genuine early risers, contrastive and masked pretraining, matured and moved to transformer backbones, so within
the CNN-scoped corpus their frequency collapses in 2024 (Figure 4), and the counts are large enough tocertifythe
collapse.Contrastive pretraininggrows from 1 to 65 supporting papers over 2019–2023, reaching 11% of the 2023
corpus, then falls to 7 of 664 in 2024, a lift of 0.10 with CI [0.05,0.21] ;masked image modelingbehaves identically
at lift 0.09, CI[0.01,0.71] . This topichadreal early risers; what it lacks is any precursor whose rise survives into
its own 2024 corpus, the defining outcome condition (iii). A 2023-vintage forecaster would have named contrastive
pretraining, and 2024 proved that call wrong; the empty set records precisely this. (3)Sparse or fragmented corpora,
small ineveryyear of 2019–2023, so no choice of onset year escapes the sparsity.Multimodal Hate Speech, with
3–29 papers per year, is noise-dominated: its best low-visibility candidate rests on 2 papers at its peak against a
single citing paper in 2024, a lift of 0.24 with CI [0.02,2.51] ; at counts this small no trajectory, rising or falling, can
be certified underanythreshold, and that impossibility is itself the intrinsic property.Epistemic AI, with 53–270
papers per year split across unrelated senses of “epistemic”, still yields a statistically solid negative: its strongest
candidate falls from 37/188 source papers at its 2022 peak to 30/400 in 2024, a lift of 0.38 whose CI [0.24,0.60]
lies entirely below the gate.Federated RL, with 69–381 papers per year, is the one borderline case: its corpus can
neither certify a rise nor rule one out, as quantified below.
Design choice: null sets over manufactured signals.We could have relaxed λor the onset gates until every
topic yielded a candidate, but the riser audit above shows exactly what relaxation would admit: single-year blips,
gapped one-paper traces, and already-visible risers, all false positives by definition, eroding the very property that
distinguishes BackTrend from unvalidated trend lists. The choice λ= 1.2 carries no responsibility for the null cases
either: sweeping the threshold anywhere above 1.05 leaves all seven topics empty. For the five gate-failing topics
this is immediate, since their best eligible lift is 1.047 ; for the two expert-null topics, every candidate that passes the
gates at any λ≥1.05 is one of the nine survivors already reviewed and removed above. An absent topic first gains
a candidate only when λis pushed below 1.05: the sole such topic isFederated RL, whose best candidate appears
in6/372 papers at its 2023 peak and 9/533 in 2024, a lift of 1.05 with a 95% confidence interval of [0.38,2.92] ,
statistically indistinguishable from flat on a 1,600 -paper corpus. We instead retain the seven topics as target topics
with documented empty signal sets: this preserves benchmark integrity and supplies honest negative cases against
which a well-calibrated foresight system shouldabstain. No evaluated system currently does: prompted on these
seven topics, every system of §4.1 still returns 3–13 predicted precursors per topic–direction pair, 6.1on average
across all 98 system–pair combinations, and abstains on none, precisely the behavior these documented nulls make
measurable. That seven of 25 externally chosen mature topics genuinely lack a certifiable precursor trace is itself a
finding about how fields mature: not all growth is preceded by a detectable weak signal, and a benchmark unable to
represent this outcome would presuppose exactly what foresight systems are meant to establish.
19

Absent topic # cand. best lift Outcome Dominant cause
Trustworthy ML 16776 2.00 6 survivors, 0 kept All 6 removed because their frequency trajectories
are not exponential rises: each rests on 1–2 papers per
nonzero year with multi-year gaps (e.g. 1,0,0,0,1 ),
visibly noise. Nothing concentrates under the um-
brella vocabulary.
Federated DL 1781 7.35 3 survivors, 0 kept All 3 removed because their frequency trajectories
are not exponential rises: erratic ( 0,0,1,0,3 ) or
frequency-flat; the one genuine precursor (prototype-
based FL) rises monotonically, and is kept, in the
broaderFederated MLcorpus.
Multimodal AI 4469 0.560passg 1···g 4 Late-consolidating container: corpus quadruples
2019–2023, and its 2024 growth is carried by can-
didates with single-year in-corpus records (25 of its
29 risers, mostly 2023-only); the one already-visible
riser (multimodal reasoning) is rejected by g4, and its
rise is uncertifiable (CI[0.57,3.28]).
Self-Supervised CNN 1960 0.600passg 1···g 4 Scope migration: precursors matured onto trans-
former backbones; their in-corpus collapse is statisti-
cally certified (contrastive pretraining 65→7 papers,
lift0.10, CI[0.05,0.21]).
Federated RL 539 1.050passg 1···g 4 Sparse niche (69–381 papers/yr, 2019–23): para-
phrase expansion adds <1% papers; best lift statisti-
cally indistinguishable from flat (CI[0.38,2.92]).
Epistemic AI 948 0.380passg 1···g 4 Sparse (53–270 papers/yr, 2019–23), fragmented
across unrelated senses of “epistemic”; strongest can-
didate’s decline is statistically certified (lift 0.38, CI
[0.24,0.60]).
Multimodal Hate Speech 112 0.240passg 1···g 4 Extreme sparsity (3–29 papers/yr, 2019–23): best
candidate’s CI [0.02,2.51] spans the gate: too few
papers to certify any trajectory, under any threshold.
Table 6: The seven mature topics with no validated weak signal.# cand.: candidate topics mined;best lift: largest
2024-frequency lift among low-visibility precursors ( g1∧g4), which gate g2requires to reach λ=1.2 ;Outcome:
result of the automatic gates and expert validation. Full analysis in §A.7.
A.8 Gate-Passing Candidates Removed by Expert Validation
The four automatic gates of §3.2 pass 125 candidates across the 25 mature target topics, of which expert validation
retains 66. Table 7 names the 59 removals, grouped by target topic and signal space, so that the outcome of validation
is auditable per topic rather than only in aggregate. Removals concentrate in the topics with the largest candidate
pools,Large Language Modelsalone accounting for 13, and they are dominated by trajectory shape rather than
topical error: exactly one candidate is removed for not naming a research direction,transparent face masksunder
Masked Face Recognition, while every other removal is a trace that lacks the low-then-exponentially-rising shape
the definition requires. That is the same ground on which §A.7 empties the signal sets ofTrustworthy MLand
Federated DL, so those two topics are not adjudicated by a stricter standard than the rest.
A.9 BERTScore Results
BERTScore-based Evaluation Details.For completeness, we additionally evaluate weak-signal discovery using
two BERTScore-based settings, which complement the LLM-based metrics reported in the main text. Let the
predicted weak-signal set be s={s 1, . . . , s m}and the reference set be g={g 1, . . . , g n}. Unlike the LLM-based
settings, which rely on explicit semantic judgments, the BERTScore-based settings use contextual embedding
similarity to quantify semantic overlap.
Inset-level BERTScore, we first collapse each set into a single text sequence by concatenating all signals with a
separator:
Concat(s) =s 1⊕s2⊕ ··· ⊕s m,
Concat(g) =g 1⊕g2⊕ ··· ⊕g n,
where⊕denotes concatenation. We then compute BERTScore between the aggregated prediction and the aggregated
20

Mature Target Topic # Gate-passing candidates removed by expert validation
Artificial Intelligence of Things 4(P)cyber-physical production systems interoperability; heterogeneity in
federated learning for IoT; intrusion detection in Internet of Vehicles.(S)
AI-assisted network slicing
Attention Mechanisms in CNN 2(P)local feature modeling in vision transformers.(S)deformable
self-attention
Evolutionary Neural Arch. Search 2 (P)neural architecture size–performance tradeoffs.(S)weight inheritance in
evolutionary neural architecture search
Explainable AI 6(P)XAI deployment challenges; actionability of AI explanations;
explainable fake news detection; explanation effectiveness.(S)explanation
method benchmarking; layer-wise relevance propagation
Federated Deep Learning†3(P)client drift in federated learning.(S)batch normalization in federated
learning; prototype-based federated learning
Federated Machine Learning 1(S)vehicular federated learning
Human–AI Interface 2(P)chatbot ethics.(S)therapeutic conversational agents
Large Language Models 13(P)LLM factuality; fairness in large language models; in-context learning
mechanisms; knowledge-based visual question answering; reasoning abilities
in large language models; zero-shot task generalization.(S)biomedical
domain-specific language models; code generation models;
instruction-following benchmarks; knowledge graph integration in language
models; long-context language models; retrieval-augmented question
answering; retrieval-augmented reasoning
Machine Unlearning 6(P)computationally efficient machine unlearning; privacy leakage in
machine unlearning; training data protection from unauthorized model
learning.(S)approximate unlearning; data-free machine unlearning;
privacy-preserving machine unlearning
Masked Face Recognition 2(S)face occlusion reconstruction; transparent face masks‡
Masked Language Model 2 (P)open-vocabulary object detection.(S)contrastive representation learning
Privacy-Preserving Machine Learning 1(S)graph neural networks
Scientific Machine Learning 4(P)toxicity prediction.(S)data-driven fluid modeling; differentiable
simulation; machine-learning-assisted statistical inference
Tiny Machine Learning 1(S)on-device gesture recognition
Trustworthy AI 2(P)human–AI decision-making.(S)sustainable AI
Trustworthy Machine Learning†6(P)ethical issues in healthcare AI; shortcut learning.(S)adversarial attacks
on model explanations; contrastive learning; explainable clinical decision
support systems; transparent machine learning
Vertical Federated Learning 2(S)communication-efficient vertical federated learning; mutual information
estimation in vertical federated learning
Total 59of 125 gate-passing candidates; the remaining 66 are the validated weak
signals of BackTrend
Table 7: Gate-passing candidates removed at expert validation, by mature target topic and signal space (P: problem
space,S: solution space). The four automatic gates of §3.2 pass 125 candidates across the 25 target topics; two
senior AI/ML researchers independently review all of them and retain 66, so the 59 listed here are the removals.
Eight target topics do not appear: five produce no gate-passing candidate at all (§A.7), and for the other three,
Asynchronous Federated Learning,Decentralized Federated Learning, andHuman-Centric AI, every gate-passing
candidate is retained.†The two topics whose every survivor is removed, leaving an empty signal set (§A.7).‡The
single removal on ground (i), naming a physical object rather than a research direction; every other removal is on
ground (ii), a trajectory that lacks the low-then-exponentially-rising shape the definition requires (§3.1).
21

2019 2020 2021 2022 2023 20240.050.10.20.511.235 gateg 2
year (2019–2023: in-corpus frequency|2024: citation-based frequency)freq. normalized to early peakF max
RAG language models (LLMs, present)
Graph unlearning (Mach. Unlearning, present)
Contrastive pretraining (Self-sup. CNN, absent)
Multimodal reasoning (Multimodal AI, absent)
Hateful-meme detect. (M. Hate Speech, absent)
Figure 4:The weak-signal signature and three absent-topic failure modes.Each candidate’s yearly frequency is
normalized to its own 2019–2023 peak Fmax, so every trajectory reaches 1.0in its early window; the 2024 point
is the citation-based 2024 frequency, and entering the shaded band ( > λF max,λ= 1.2 ) is exactly the gate g2.
Presentsignals rise from low visibility and then leap into the band in 2024. The threeabsent-topic candidates
each fail differently:contrastive pretrainingrose from low visibility to 11% of the CNN-scoped 2023 corpus, then
its frequencycollapsesin 2024 as the method matured onto transformers (lift 0.10, CI[0.05,0.21] : a statistically
certified failure of the outcome condition);multimodal reasoningwas never low-visibility, starting at 61% of its
own peak, so g4rejects it, and its 2024 rise is itself uncertifiable (CI [0.57,3.28] );hateful-meme detectionspikes on
a handful of papers in a tiny corpus and collapses, noise with no positive growth fit for g1. None completes the
low-visibility, in-window-rise, 2024-confirmation signature that defines a weak signal.
reference:
(Pset-bs, Rset-bs, F1 set-bs)
= BERTScore 
Concat(s),Concat(g)
.
This setting measures coarse-grained similarity between the two sets as wholes, but does not explicitly model
one-to-one or one-to-many correspondence between individual weak signals.
Insignal-level BERTScore, we instead compare weak signals individually. We first compute the full pairwise
similarity matrix
Sij= BERTScore F1(si, gj),
i= 1, . . . , m, j= 1, . . . , n,
where each entry quantifies the semantic similarity between predicted signal siand reference signal gj. We then
aggregate this matrix directionally. Precision is defined as the average best-match similarity from each prediction to
the reference set:
Psig-bs=1
mmX
i=1max
1≤j≤nSij.
Intuitively, this asks: for each predicted signal, how well does its closest reference counterpart match? Recall is
defined symmetrically as the average best-match similarity from each reference signal to the prediction set:
Rsig-bs=1
nnX
j=1max
1≤i≤mSij.
22

This asks: for each reference weak signal, how well is it covered by the best prediction? We then combine the two
using the harmonic mean:
F1sig-bs=2Psig-bsRsig-bs
Psig-bs+R sig-bs.
Compared with set-level BERTScore, this signal-level variant provides a finer-grained view of coverage by rewarding
predictions that closely match individual reference weak signals rather than only the overall set semantics.
The main text reports only the LLM-judge and human-proxy settings; we give the two BERTScore-based
settings here, together with the per-direction breakdown (Table 8) and, below, an analysis of why BERTScore is
uninformative for weak-signal matching.
ModelSet-level BERTScore F1 Signal-level BERTScore F1
Prob Sol All Prob Sol All
GPT-5.4 81.7 82.9 82.3 86.988.387.6
Qwen3.5-397B 82.3 82.5 82.4 87.0 87.1 87.0
DeepSeek-R1-0528 82.3 82.9 82.6 87.1 87.7 87.4
Tongyi-DR-30B-A3B 83.1 82.9 83.0 87.5 87.4 87.4
DR-Tulu-8B 80.9 80.8 80.8 86.1 86.0 86.1
Qwen3-30B (RAG)83.5 83.1 83.3 88.087.587.8
Qwen3-8B (RAG) 83.6 82.6 83.1 87.8 86.5 87.2
Table 8: Appendix BERTScore results on the current AI/ML benchmark, broken down by signal direction (Problem
/Solution /All) for the set-level and signal-level settings. All values are percentages (%). Higher is better; the
best score in each column is in bold. All systems cluster within a few points, confirming that BERTScore is poorly
discriminative for weak-signal matching (cf. the LLM-judge scores in Table 2). The spread across all seven systems
is 2.5 points at the set level and 1.7 at the signal level, an order of magnitude narrower than the LLM-judge spread
on the same predictions.
Why BERTScore is high and non-discriminative.Signal-level BERTScore aggregates token-level cosine
similarities between contextual roberta-large embeddings via greedy best-match matching. Every weak signal in
BackTrend is a short noun phrase from a single domain, assembled from a small shared vocabulary:learning,
model,language,federated,augmented,neural,training,transformer. Two such phrases therefore share many
high-similarity tokensregardlessof whether they denote the same research direction, and roberta-large places
domain terms close together in embedding space; without baseline rescaling, even unrelated same-domain phrases
score above 0.85. BERTScore thus measures“are these both AI/ML phrases”rather than“are these the same
specific precursor”, which is exactly the distinction the benchmark hinges on. This is why all seven systems cluster
at81–88(Table 8) irrespective of correctness.
BERTScore’s “matches” are frequently the wrong research direction.Table 9 makes the failure concrete:
for each reference signal we show the prediction that BERTScore scoreshighest, and it is repeatedly a different
research direction that the LLM judge (and inspection) reject. These false matches score 86.7–91.2, on par with or
above genuine matches. Most strikingly, for the referencelong-context understanding in large language models
BERTScore ranks the unrelatedinterpretability and mechanistic understanding deficits( 91.2)abovethe correct
long-context dependence and context window limitations( 90.1), and forretrieval-augmented language modelsit
ranksprogram-aided and code-interpreted reasoning( 89.8) above the correctretrieval-augmented generation( 89.6).
In both cases a shared abstract-noun template — “. . . understanding . . . ” or “. . . -aided/-augmented . . . reasoning” —
dominates the token overlap, while the one word that carries the meaning barely moves the score. A metric that
ranks a wrong precursor above the right one cannot separate systems on this task, which is why we treat the LLM
judge, whose near-floor verdict a judge from a third model family corroborates in Table 2, as the primary evaluator
and relegate BERTScore to this appendix.
A.10 Additional Error Cases
This appendix expands the discussion in Section 5.2 by providing concrete benchmark–prediction comparisons.
The goal is to show that many low-scoring cases are not random failures, but systematic mismatches in framing,
abstraction level, or semantic coverage. In particular, several examples are highly plausible on their own terms,
which helps explain why lexical similarity can remain high even when semantic judge-based matching is low.
The examples in Table 10 follow the same four error types discussed in the main text. First, some cases exhibit
topic drift, where the model stays within a reasonable neighboring area but shifts away from the benchmark’s
23

Reference weak signal Prediction BERTScore ranks highest F1 Why it is not a match
parameter-efficient tuning Parameter-efficient finetuning 92.0(the correct match,
shown for reference)
long-context understanding in large
language modelsInterpretability and mechanistic
understanding deficits91.2interpretability̸=long
context, and it outranks
the correctLong-context
dependence and context
window limitations(90.1)
retrieval-augmented language models Program-aided and code-interpreted
reasoning89.8code execution̸=
retrieval, and it outranks
the correct
Retrieval-augmented
generation(89.6)
post-training quantization for large
language modelsTool use via language-model
orchestration89.3 model compression̸=
agentic tool use
privacy leakage in NLP models Benchmark contamination and data
leakage88.9 test-set contamination̸=
leaking training data;
only the word “leakage”
overlaps
machine-generated text detection In-context generalization without
reliable evaluation87.6 detecting generated text
̸=few-shot
generalization
LLM-based automated program repair Scaling-law-guided model development 86.7 code repair̸=scaling
laws
Table 9: BERTScore false matches on thelarge language modelstopic (GPT-5.4 predictions, both signal directions).
For each reference weak signal we list the prediction that signal-level BERTScore scores highest and its F1 (%).
The first row is the genuine match, included as a baseline. Every other row is a different research direction that
the LLM judge rejects, yet all of them score between 86.7 and91.2, indistinguishable from that baseline. In two
cases the top-scoring prediction is not merely wrong but outranks the correct one: the judge acceptsRetrieval-
augmented generationandLong-context dependence and context window limitations, and BERTScore places a
rejected prediction above each. BERTScore rewards shared domain vocabulary and phrase templates rather than
same-topic identity.
intended problem definition. Second, some cases showgranularity mismatch, where the model predicts narrower
mechanisms or engineering details while the benchmark is formulated at the level of research paradigms or system
bottlenecks. Third, some examples arelexical near-misses: the predictions are clearly in the same field and
share substantial vocabulary with the benchmark, but the actual research framing is different. Finally, some cases
demonstratecoverage failure, where the model captures one dimension of the benchmark but omits other equally
central dimensions.
24

Topic Direction Reference signal(s) GPT-5.4 prediction(s) (rank) Error interpretation
Trustworthy AI Problem AI trust measurement; appropriate
reliance on AI; AI trust calibration;
ethical AI in chatbot systemsRobustness under distribution shift
(1); uncertainty calibration (2);
out-of-distribution detection (3);
spurious correlation dependence (4)Topic drift from human-facing trust
calibration and measurement to
generic model-robustness and safety
problems.Uncertainty calibrationis
the nearest miss, but calibrating a
model’s confidence is not calibrating
a user’s trust; none of the 13 ranked
predictions matches.
Attention Mechanisms
in CNNSolution Linear-complexity self-attention;
sparse attention in vision
transformers; large-kernel
convolutions; locality-enhanced
transformer architectures;
hierarchical vision transformersLightweight channel–spatial gating
modules (1); multi-scale feature
reweighting (2); dynamic convolution
via input-conditioned kernel
weighting (3); efficient attention
approximations for vision backbones
(9)Granularity mismatch: the top of the
ranking is fine-grained
attention-block variants rather than
the benchmark’s architecture-family
solutions. The single match,efficient
attention approximations, surfaces
only at rank 9.
Masked Language
ModelSolution Vision-language models; prompt
tuning for pretrained language
models; masked video modelingSpan corruption denoising objectives
(1); replaced-token detection
objectives (2); dynamic masking
during pretraining (3); multimodal
masking objectives (12)Lexical near-miss: all 14 predictions
stay inside masked-LM pretraining
objectives. The two that do point
outward,multimodal masking
objectivesandretrieval-augmented
encoder representations, sit at ranks
12 and 14 and still miss the
downstream paradigms the
benchmark tracks.
Vertical Federated
LearningProblem Straggler problem; privacy leakage;
feature selection; unaligned data (all
in vertical FL)Privacy-preserving entity resolution
across siloed parties (1);
partial-overlap sample populations
(3); leakage risk from intermediate
predictions and gradients (5); label
leakage from party roles and training
interactions (10)Coverage failure on the
highest-scoring pair in the benchmark
(F141.8 ): unaligned data and privacy
leakage are matched at ranks 3 and 5,
yet the top-ranked prediction reframes
alignment as entity resolution and
feature selection is absent from all 13.
Machine Unlearning Problem Federated unlearning; graph
unlearningCertified removal of data influence
(2); retraining cost after data deletion
(3); cross-silo deletion in federated or
distributed training (9)Coverage failure: the federated
setting is recovered only at rank 9,
graph unlearning is absent from all 12
predictions, and the top ranks are
generic erasure-compliance concerns.
Large Language
ModelsProblem Privacy leakage in NLP models;
long-context understanding; LLM
alignment with human feedback;
machine-generated text detection;
pre-trained LM compressionHallucination and factual unreliability
(2); benchmark contamination and
data leakage (3); long-context
dependence and context window
limitations (10); memorization,
privacy leakage, and unintended
training-data reproduction (11)Coverage failure: two of the five
references are recovered, but only at
ranks 10–11; text detection and
model compression are absent from
all 16 predictions, and the first nine
ranks are evaluation and reliability
concerns outside the reference set.
Table 10: Representative benchmark–prediction comparisons for GPT-5.4 predictions on the AI/ML target topics.
Predictions are quoted from the ranked lists scored in Table 2, with each one’s rank in the system’s own confi-
dence ordering in parentheses. The examples illustrate that many low-scoring outputs are semantically plausible
near-misses that diverge in framing, abstraction level, or benchmark coverage. The ranks also show where the
misalignment sits: when a reference signal is recovered at all, it is usually recovered late.
A.11 Prompts
In this section, we present the comprehensive prompts that we use to mine candidate topics (Figure 5, continued in
Figure 6), elicit weak-signal predictions from the evaluated systems (Figure 7, Figure 8), and conduct LLM-as-a-
judge evaluations (Figure 9, Figure 10). We use Artificial Intelligence and Machine Learning as our field, and the
prompts can be easily adapted for other fields.
25

Construction Prompt: Candidate Topic Extraction (System Prompt, and User Prompt Part 1)
System prompt.
You are an expert research topic extractor.
Your task is to extract literature-level research topics from paper abstracts.
Extract only topics that are explicitly discussed in the paper.
A good topic should be broad enough that multiple independent papers could study it,
but specific enough to be more informative than a general field label.
Prefer clean, compact topic labels that name the core research problem or method family.
Return only valid json.
User prompt.
Target established topic:
{target_topic}
This paper was retrieved as related to the target established topic above.
Use the target topic only as context for relevance filtering.
Do not output the target topic itself unless the abstract discusses a more specific reusable subtopic.
Paper metadata:
- Title: {title}
- Paper ID: {paper_id}
- Year: {year}
- Venue: {venue}
- Source query: {source_query}
Abstract:
{abstract}
Topic categories:
1. Problem-space topics:
Research problems, gaps, limitations, risks, bottlenecks, evaluation failures, or scientific questions discussed by the paper.
2. Solution-space topics:
Research methods, method families, system directions, evaluation approaches, defenses, or solution directions discussed by the paper.
Use solution-space only for standalone reusable methods, method families, systems, defenses, algorithms, datasets, benchmarks, or evaluation protocols, not for
the problem that motivates them.
Task:
Extract clean, reusable research topics from this paper abstract that are conceptually related to the target established topic: “{target_topic}”.
These candidates may be problem-space topics or solution-space topics.
Most abstracts describe both a problem/gap and a method/solution. Separate these roles.
Do not combine a method, remedy, evaluation detail, dataset, application setting, or implementation detail with the problem it addresses.
Specificity guidance:
- Too broad: a whole field (e.g., “machine learning”, “computer vision”), broad model family (e.g., “deep learning”), or generic category label (e.g.,
“optimization”).
- Too specific: a paper-specific method name (e.g., “CoPINet”), system name (e.g., “GPT Semantic Cache”), exact task setting (e.g., “Raven’s Progressive
Matrices”), implementation detail, single experimental finding (e.g., “68.8% API-call reduction”), or enumerating technical details (e.g., “using interventional
data”, “with linguistically regularized CNN”, “additive feature attribution”).
- Correct level: a reusable research direction or problem space that multiple independent papers could study using different methods, or systems. The topic should
capture the core research direction without enumerating specific techniques or data types.
- Prefer compact labels such as “causal model evaluation” over long contribution phrases such as “evaluation of causal models using interventional empirical
data”.
- Focus on the research problem or method family, not on the specific implementation details or data modalities.
- If the abstract discusses a narrow technique or case study, abstract it to the broader research problem or method family it addresses.
- Do not phrase topics as actions (e.g., “evaluation of”, “analysis of”) or as this specific paper’s contribution.
- A candidate topic must be a standalone research topic phrase, not a relation between a problem and a solution.
- Avoid “X for Y” topic names when X is a method and Y is a problem, goal, task, or desired property. Split them into separate candidate topics when both are
explicitly supported.
- If the paper proposes method X to solve, evaluate, improve, or measure problem Y , output Y as a problem-space topic and X as a solution-space topic only when
each is independently reusable. Do not output the combined phrase.
Figure 5: Construction prompt used to extract problem- and solution-space candidate topics from each retrieved
paper abstract, reproduced verbatim from the released pipeline code; braces mark the template slots filled in at
run time: the full system prompt, and the user prompt up to the specificity guidance. The remainder is shown in
Figure 6.
26

Construction Prompt: Candidate Topic Extraction (User Prompt Part 2)
Bad topic examples (wrong abstraction level):
- “machine learning” because it is too broad
- “Empirical evaluation of causal models using interventional data” because it enumerates technical details (“using interventional data”)
- “evaluation of causal modeling algorithms using interventional empirical data” because it mixes the core problem with proposed evaluation details; prefer
“causal model evaluation” as the problem-space topic.
- “CoPINet for Raven’s Progressive Matrices” because it is paper-specific
- “Aspect-based sentiment analysis with linguistically regularized CNN” because it enumerates method details
- “Explainability challenges in additive feature attribution” because it is too method-specific
- “deep reinforcement learning for federated edge learning resource management” because it combines a solution method with a problem/context; split into
cleaner topics only if each side is independently reusable and explicitly supported.
- “Using interaction-aware explanations to solve misleading model interpretability” because it mixes a solution and a problem in one extracted topic; extract the
problem topic and solution topic separately if both are explicitly supported.
- “uncertainty communication for AI trust calibration” because it mixes a solution method with the problem or goal it addresses; extract “AI trust calibration” as a
problem-space topic or “uncertainty communication in AI systems” as a solution-space topic if each is explicitly supported.
Requirements:
- Output a json object with a “topics” array, which may be empty.
- Extract up to 2 topics total.
- Each topic must be explicitly supported by the abstract.
- Each topic must be reusable across multiple papers.
- Each topic must be one abstraction level broader than the paper’s specific method, benchmark, dataset, or case study.
- Each topic must include exactly one “topic_type”: “problem-space” or “solution-space”. A mix is not allowed.
- Each topic must be conceptually related to the target established topic: “{target_topic}”.
- If a topic cannot be clearly classified as problem-space or solution-space, do not emit it.
- If a topic cannot be clearly related to the target established topic, do not emit it.
- Do not phrase topics as actions, paper contributions, or problem-solution relationships.
- Do not combine a method and the problem it addresses into one topic phrase.
- Do not include proposed measurement choices, dataset choices, application settings, or implementation details in the topic label unless they are themselves the
reusable research topic.
- Do not invent evidence beyond the abstract.
Return only json with this schema:
{
"topics": [
{
"topic": "<standalone literature-level candidate topic>",
"topic_type": "problem-space|solution-space",
"target_topic": "{target_topic}",
"evidence": "<short phrase grounded in the abstract>",
"confidence": "high|medium|low"
}
]
}
Figure 6: Continuation of the candidate-topic extraction user prompt (Figure 5): bad-topic examples, output
requirements, and the response schema. Extracted candidates are then deduplicated, embedding-clustered, and
scored by early frequency and 2024 reference adoption before expert validation (§3.2).
27

Prediction Prompt for Problem-Space Weak Signals
You are an expert analyst of frontier {domain} research. Your task is to identify early weak signals that later contributed to a specified mature target topic.
We distinguish two categories of weak signals. One category is the “solution-space weak signal”: an early research method, technique, or design principle that
was not yet widely adopted but later became an important solution to an already-recognized problem.
However, the category you are asked to identify here is the “problem-space weak signal”: an underrecognized research problem or problem formulation that was
not yet widely recognized by the research community at the time it emerged, but that later became central to the mature target topic [{mainframe_topic}].
To be more specific, problem-space weak signals include research problems, gaps, limitations, risks, bottlenecks, evaluation failures, or scientific questions that
emerged in the literature during the prediction window, rather than claims tied to a single paper.
Mature target topic:
[{mainframe_topic}]
Prediction window:
[{year_range}]
Retrospective setup:
The mature target topic should be treated as a topic that is already established or prominent by 2024. Your task is not to predict after 2024. Your task is to look
backward and identify what this 2024 mature topic looked like in the 2019-2023 literature while it was still emerging.
Use the mature target topic only as 2024 relevance context. The weak signals themselves must be research problems, gaps, limitations, risks, bottlenecks, or
scientific questions that appeared in [{year_range}]. Do not use 2024-or-later evidence, and do not simply restate the mature target topic unless you name a more
specific predecessor formulation.
Question:
What are the early problem-space weak signals in {domain} that emerged between [{year_range}] and later contributed to the mature target topic
[{mainframe_topic}]?
Specificity guidance:
- Use the mature target topic only as relevance context.
- Do not output the target topic itself unless you name a more specific reusable subtopic or predecessor direction.
- Too broad: a whole field (e.g., “machine learning”, “computer vision”), broad model family (e.g., “deep learning”), or generic category label (e.g.,
“optimization”).
- Too specific: a paper-specific method name, system name, exact dataset, benchmark instance, implementation detail, single experimental finding, or single case
study.
- Correct level: a reusable research direction or problem space topic that multiple independent papers could study using different methods or systems.
- Focus on the research problem, not on specific implementation details or data modalities.
- Do not phrase signals as actions, paper contributions, or problem-solution relationships.
- Avoid “X for Y” signal names when X is a method and Y is a problem, goal, task, or desired property. For problem-space output, keep only the problem side
(i.e., Y) when it is independently reusable and relevant.
Requirements:
- Return ONLY valid JSON, no markdown fences, no explanation.
- Output a JSON object with a “weak_signals” array. Order the “weak_signals” array from most to least confident: the array order is your ranking, strongest first.
- Output at least 10 weak signals unless you genuinely cannot name that many.
- Each weak signal must have exactly these fields: “signal”, “what_it_was”, “why_weak_signal”.
- Each weak signal must be explicitly tied to the prediction window [{year_range}].
- “what_it_was” must include the year or year range within [{year_range}].
- “why_weak_signal” must explain why this was a problem-space weak signal for [{mainframe_topic}].
- Each signal must be conceptually related to [{mainframe_topic}].
- Each signal must be reusable across multiple papers.
- Do not include solution methods.
- Do not invent evidence or overclaim certainty.
Return only JSON with this schema:
{
"weak_signals": [
{
"signal": "<short reusable problem-space weak signal name>",
"what_it_was": "<1-2 sentences describing what it was, including the year>",
"why_weak_signal": "<1-2 sentences explaining why it was a problem-space weak signal for
[{mainframe_topic}]>"
}
]
}
Figure 7: Prediction prompt template for problem-space weak signals, reproduced verbatim from the released
pipeline code; braces mark the template slots filled in at run time. domain is Artificial Intelligence and Machine
Learning, mainframe_topic the mature target topic M, andyear_range the 2019–2023 prediction window.
Every evaluated system receives this as a single user message with no system prompt; the RAG systems additionally
append the retrieved evidence block and the instruction “Use this evidence when generating the weak signals.”
(§4.1).
28

Prediction Prompt for Solution-Space Weak Signals
You are an expert analyst of frontier {domain} research. Your task is to identify early weak signals that later contributed to a specified mature target topic.
We distinguish two categories of weak signals. One category is the “problem-space weak signal”: an underrecognized research problem or problem formulation
that was not yet widely recognized by the research community at the time it emerged, but that later became central to the mature target topic [{mainframe_topic}].
However, the category you are asked to identify here is the “solution-space weak signal”: an early research method, technique, or design principle that was not
yet widely adopted but later became an important solution to an already-recognized problem.
To be more specific, solution-space weak signals include research methods, method families, system directions, evaluation approaches, defenses, or solution
directions that emerged in the literature during the prediction window, rather than claims tied to a single paper.
Mature target topic:
[{mainframe_topic}]
Prediction window:
[{year_range}]
Retrospective setup:
The mature target topic should be treated as a topic that is already established or prominent by 2024. Your task is not to predict after 2024. Your task is to look
backward and identify what this 2024 mature topic looked like in the 2019-2023 literature while it was still emerging.
Use the mature target topic only as 2024 relevance context. The weak signals themselves must be research methods, method families, system directions,
evaluation approaches, defenses, or solution directions that appeared in [{year_range}]. Do not use 2024-or-later evidence, and do not simply restate the mature
target topic unless you name a more specific predecessor formulation.
Question:
What are the early solution-space weak signals in {domain} that emerged between [{year_range}] and later contributed to the mature target topic
[{mainframe_topic}]?
Specificity guidance:
- Use the mature target topic only as relevance context.
- Do not output the target topic itself unless you name a more specific reusable subtopic or predecessor direction.
- Too broad: a whole field (e.g., “machine learning”, “computer vision”), broad model family (e.g., “deep learning”), or generic category label (e.g.,
“optimization”).
- Too specific: a paper-specific method name, system name, exact dataset, benchmark instance, implementation detail, single experimental finding, or single case
study.
- Correct level: a reusable research method, method family, system direction, evaluation approach, defense, or solution direction that multiple independent papers
could study.
- Focus on the research method or solution direction, not on specific implementation details, problem formulations, or data modalities.
- Do not phrase signals as actions, paper contributions, or problem-solution relationships.
- Avoid “X for Y” signal names when X is a method and Y is a problem, goal, task, or desired property. For solution-space output, keep only the solution side
(i.e., X) when it is independently reusable and relevant.
Requirements:
- Return ONLY valid JSON, no markdown fences, no explanation.
- Output a JSON object with a “weak_signals” array. Order the “weak_signals” array from most to least confident: the array order is your ranking, strongest first.
- Output at least 10 weak signals unless you genuinely cannot name that many.
- Each weak signal must have exactly these fields: “signal”, “what_it_was”, “why_weak_signal”.
- Each weak signal must be explicitly tied to the prediction window [{year_range}].
- “what_it_was” must include the year or year range within [{year_range}].
- “why_weak_signal” must explain why this was a solution-space weak signal for [{mainframe_topic}].
- Each signal must be conceptually related to [{mainframe_topic}].
- Each signal must be reusable across multiple papers.
- Do not include problem statements.
- Do not invent evidence or overclaim certainty.
Return only JSON with this schema:
{
"weak_signals": [
{
"signal": "<short reusable solution-space weak signal name>",
"what_it_was": "<1-2 sentences describing what it was, including the year>",
"why_weak_signal": "<1-2 sentences explaining why it was a solution-space weak signal for
[{mainframe_topic}]>"
}
]
}
Figure 8: Prediction prompt template for solution-space weak signals, reproduced verbatim from the released
pipeline code; braces mark the template slots filled in at run time. domain is Artificial Intelligence and Machine
Learning, mainframe_topic the mature target topic M, andyear_range the 2019–2023 prediction window.
Every evaluated system receives this as a single user message with no system prompt; the RAG systems additionally
append the retrieved evidence block and the instruction “Use this evidence when generating the weak signals.”
(§4.1).
29

Set-Level LLM-as-a-Judge Prompt
System prompt.
You are a research topic matcher. Two research topics match only if they denote the same specific research topic: the same core method or the same problem,
differing at most in wording, phrasing, or abbreviation. Topics that are merely related, adjacent, complementary, or from the same broad area do not match.
User prompt.
Compare the PREDICTED set against the GROUND TRUTH set of research topics.
Precision = fraction of predicted topics that match some ground-truth topic.
Recall = fraction of ground-truth topics that match some predicted topic.
Ground truth:
{gt_block}
Predicted:
{pred_block}
Return ONLY valid JSON:
{“precision”: <float 0-1>, “recall”: <float 0-1>, “matched_pairs”: [{“gt_index”: <int>, “pred_index”: <int>}]}
Give one matched_pairs entry per matching (ground-truth, predicted) pair; empty list if none.
Figure 9: Set-level LLM-as-a-judge prompt, reproduced verbatim from the released pipeline code; braces mark the
template slots filled in at run time. Both judges share the same system prompt and differ only in the user message.
gt_block/pred_block are the reference and prediction lists, rendered as 0-based numbered items. A match
requires the two topics to denote the same specific research topic, not merely a related one (§4.2).
Signal-Level LLM-as-a-Judge Prompt
System prompt.
You are a research topic matcher. Two research topics match only if they denote the same specific research topic: the same core method or the same problem,
differing at most in wording, phrasing, or abbreviation. Topics that are merely related, adjacent, complementary, or from the same broad area do not match.
User prompt.
For each candidate research topic, decide whether it matches any reference research topic.
Reference:
{ref_block}
Candidate:
{cand_block}
Return ONLY valid JSON: {“matches”: [<int>, ...]} with exactly {n_cand} elements, 1 if the candidate matches any reference else 0.
Figure 10: Signal-level LLM-as-a-judge prompt, reproduced verbatim from the released pipeline code; braces
mark the template slots filled in at run time. Both judges share the same system prompt and differ only in the user
message. ref_block/cand_block are the reference and prediction lists, rendered as 0-based numbered items.
A match requires the two topics to denote the same specific research topic, not merely a related one (§4.2).
30