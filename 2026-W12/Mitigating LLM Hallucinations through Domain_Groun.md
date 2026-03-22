# Mitigating LLM Hallucinations through Domain-Grounded Tiered Retrieval

**Authors**: Md. Asraful Haque, Aasar Mehdi, Maaz Mahboob, Tamkeen Fatima

**Published**: 2026-03-18 15:59:30

**PDF URL**: [https://arxiv.org/pdf/2603.17872v1](https://arxiv.org/pdf/2603.17872v1)

## Abstract
Large Language Models (LLMs) have achieved unprecedented fluency but remain susceptible to "hallucinations" - the generation of factually incorrect or ungrounded content. This limitation is particularly critical in high-stakes domains where reliability is paramount. We propose a domain-grounded tiered retrieval and verification architecture designed to systematically intercept factual inaccuracies by shifting LLMs from stochastic pattern-matchers to verified truth-seekers. The proposed framework utilizes a four-phase, self-regulating pipeline implemented via LangGraph: (I) Intrinsic Verification with Early-Exit logic to optimize compute, (II) Adaptive Search Routing utilizing a Domain Detector to target subject-specific archives, (III) Corrective Document Grading (CRAG) to filter irrelevant context, and (IV) Extrinsic Regeneration followed by atomic claim-level verification. The system was evaluated across 650 queries from five diverse benchmarks: TimeQA v2, FreshQA v2, HaluEval General, MMLU Global Facts, and TruthfulQA. Empirical results demonstrate that the pipeline consistently outperforms zero-shot baselines across all environments. Win rates peaked at 83.7% in TimeQA v2 and 78.0% in MMLU Global Facts, confirming high efficacy in domains requiring granular temporal and numerical precision. Groundedness scores remained robustly stable between 78.8% and 86.4% across factual-answer rows. While the architecture provides a robust fail-safe for misinformation, a persistent failure mode of "False-Premise Overclaiming" was identified. These findings provide a detailed empirical characterization of multi-stage RAG behavior and suggest that future work should prioritize pre-retrieval "answerability" nodes to further bridge the reliability gap in conversational AI.

## Full Text


<!-- PDF content starts -->

Mitigating LLM Hallucinations through Domain-Grounded
Tiered Retrieval
Md. Asraful Haquea,b,1, Aasar Mehdib,2, Maaz Mahboobb,3, and Tamkeen Fatimac,4
aComputational Unit, Z.H. College of Engineering & Technology, Aligarh Muslim
University, India
bInterdisciplinary Center for Artificial Intelligence, Aligarh Muslim University, India
cDepartment of Computer Engineering, Aligarh Muslim University, India
Email:1md_asraf@zhcet.ac.in,2asarmehdibaquri@gmail.com,
3maazmahboob4@gmail.com,4tamkeen.fatima@zhcet.ac.in
Abstract
Background: Large Language Models (LLMs) have achieved unprecedented fluency but remain
susceptible to “hallucinations”—the generation of factually incorrect or ungrounded content. This
limitation is particularly critical in high-stakes domains where reliability is paramount.
Objective: We propose a domain-grounded tiered retrieval and verification architecture designed
to systematically intercept factual inaccuracies by shifting LLMs from stochastic pattern-matchers
to verified truth-seekers.
Methodology: The proposed framework utilizes a four-phase, self-regulating pipeline imple-
mented via LangGraph: (I) Intrinsic Verification with Early-Exit logic to optimize compute, (II)
Adaptive Search Routing utilizing a Domain Detector to target subject-specific archives, (III) Cor-
rective Document Grading (CRAG) to filter irrelevant context, and (IV) Extrinsic Regeneration
followed by atomic claim-level verification. The system was evaluated across 650 queries from five
diverse benchmarks: TimeQA v2, FreshQA v2, HaluEval General, MMLU Global Facts, and Truth-
fulQA.
Results: Empirical results demonstrate that the pipeline consistently outperforms zero-shot
baselines across all environments. Win rates peaked at 83.7% in TimeQA v2 and 78.0% in MMLU
Global Facts, confirming high efficacy in domains requiring granular temporal and numerical preci-
sion. Groundedness scores remained robustly stable between 78.8% and 86.4% across factual-answer
rows.
Conclusion: While the architecture provides a robust fail-safe for misinformation, a persistent
failure mode of “False-Premise Overclaiming” was identified. These findings provide a detailed
empirical characterization of multi-stage RAG behavior and suggest that future work should prioritize
pre-retrieval “answerability” nodes to further bridge the reliability gap in conversational AI.
Keywords:Hallucination Mitigation, Large Language Models (LLMs), Retrieval-Augmented Genera-
tion (RAG), AI Reliability, Fact-checking, Multi-stage Verification.
1 Introduction
Large Language Models (LLMs), including OpenAI’s ChatGPT, Google’s Gemini, and Meta’s Llama 3,
have fundamentally transformed natural language processing. By training on massive datasets, these
models have achieved a level of linguistic fluency often indistinguishable from human communication,
excelling in tasks like reasoning, summarization, and creative writing [1–3]. Their rapid adoption across
medicine, law, and journalism underscores their significant societal and industrial impact [2]. However,
the credibility of LLMs is frequently undermined by hallucinations—a critical flaw where the model
generates syntactically correct and coherent text that is despite being factually incorrect or unsupported
by reliable sources [4, 5]. This issue is particularly severe in high-stakes environments where factual
deviations can result in ethical breaches or dangerous decision-making errors [6] (Fig. 1). The origins of
hallucinations are multifactorial, stemming from data quality issues, the probabilistic nature of autore-
gressive generation, and ambiguous user prompts [7, 8]. Because LLMs are designed to predict the most
likely next token rather than verify truth, they often prioritize plausibility over correctness [9].
1arXiv:2603.17872v1  [cs.CL]  18 Mar 2026

Decision-Making
ErrorsMisinformation Spread
Erosion of Trust
Safety Risks
(e.g.
Medical/Legal)Inefficiency
&
Resource WasteReputational
Damage
(for Businesses)LLM Hallucination
ProblemsFigure 1: Consequences of LLM Hallucinations
Researchers generally categorize hallucination problems into two principal types [10]:
•Intrinsic Hallucinations: These occur when a model’s internal limitations (such as data bias or
restricted training diversity) cause it to distort or contradict the source information it is intended to
process. This represents a failure of source fidelity, often resulting in outputs that directly oppose
the provided input (Fig. 2).
•Extrinsic Hallucinations: These emerge when a model is queried on information absent from its
training data or too recent for its internal knowledge base. Without access to external verification
tools, the model relies on linguistic patterns to generate plausible but entirely unverifiable content,
marking a fundamental failure in factual grounding (Fig. 2).
Figure 2: Intrinsic vs Extrinsic Hallucination
Beyond these two broad categories, hallucinations can also be sub-classified by the type of factual
2

error they exhibit as shown in Table 1. This fine-grained taxonomy helps identify the nature of the
factual deviation, which is crucial for designing appropriate mitigation mechanisms.
Table 1: Hallucination Types based on Factual Deviation
Type Nature of Error Example Primary Cause
Factual Incorrect state-
ments contradict-
ing truth“Albert Einstein won two Nobel Prizes
in Physics.”Inaccurate memory
/ data bias
Numerical Wrong figures or
quantities“The Earth’s atmosphere contains 40%
oxygen.”Lack of numerical
reasoning
Logical Invalid reasoning
chain“All doctors are scientists; Li is a scien-
tist; therefore, Li is a doctor.”Weak inferential
ability
Commonsense Violates basic
knowledge“The sun rises in the west and sets in the
east.”Lack of world
grounding
Temporal Outdated or mixed-
up timelines“Barack Obama is the current President
of the United States.”Static knowledge
Attributional Wrong author or
source“The theory of relativity was proposed
by Isaac Newton.”Entity confusion
Contextual Factually true but
irrelevantIn an article about climate change policy,
the model inserts accurate facts about
solar flares that are irrelevant to the pol-
icy discussion.Poor discourse con-
trol
Referential Fabricated entities
or studies“According to a 2023 report by Wipro
Tech., 500 million full-time jobs could be
impacted by AI systems.” (No such re-
port exists)Name pattern syn-
thesis
Spatial Wrong geographic
facts“Sydney is capital of Australia.” Geographic misas-
sociation
Multimodal Cross-modal incon-
sistencyAn image caption stating, “A man play-
ing guitar on stage,” when no guitar is
present.Misaligned multi-
modal training
AI shouldn’t just sound smart; it also needs to be honest and clear about where its information comes
from. When an AI model hallucinates, it ruins that trust. Recent benchmarks such as TruthfulQA
show that even state-of-the-art models like GPT-3 achieve only∼58% truthfulness, while humans reach
over 94%, revealing a substantial reliability gap [11]. While strategies such as Retrieval-Augmented
Generation (RAG) and Reinforcement Learning from Human Feedback (RLHF) have been introduced,
significant limitations persist:
(i)Static Scrutiny: Most frameworks apply the same level of verification to every response, leading
to inefficiency and redundant checks for reliable claims.
(ii)Lack of Feedback Loops: Current systems cannot dynamically adjust verification efforts based
on internal confidence or past results [12, 13].
(iii)Opacity: Users are rarely informed about which parts of a response are verified or uncertain,
reducing trust and interpretability.
(iv)Resource Intensity: High-quality fact-checking often relies on expensive models (e.g., using
GPT-4 to check GPT-3), making real-time deployment impractical [14–16].
To address these gaps, this paper introduces a multi-layered verification pipeline designed to be self-
regulating, adaptive, and resource-efficient. This framework moves beyond standard “search and answer”
methods by incorporating internal and external checkpoints to ensure both fluency and factual integrity.
2 Related Work
Hallucination evaluation requires assessing whether generated statements are consistent with verifiable
truth rather than simply fluent or coherent. Research on hallucinations in LLMs has grown rapidly, with
3

efforts directed toward characterizing their occurrence, identifying their underlying mechanisms, and
developing mitigation strategies. The first major approach to hallucination detection focuses on factual
consistency—the degree to which the generated output aligns with source data or established truth.
One of the earliest and most widely used tools is FactCC, a supervised model trained to detect factual
inconsistencies in summarization outputs [17]. FactCC leverages a Natural Language Inference (NLI)
framework to classify whether each generated sentence is supported, refuted, or unverifiable based on the
source document. Despite its simplicity, FactCC’s binary nature limits its ability to handle nuanced or
multi-sentence dependencies. To overcome these limitations, a later study introduced SummaC [18]. It
utilizes a sentence-level entailment model that aggregates consistency judgments across all aligned text
spans, thereby improving robustness in longer summaries and generative tasks. Several other widely used
approaches for detecting hallucinations include truthfulness benchmarks, knowledge-grounding metrics,
and Retrieval-Augmented Generation (RAG)–based methods. Ayala and Bechard [19] presented a RAG
system that successfully reduced hallucinations in structured outputs, like workflows, that are generated
from natural language. They developed a retriever that suggested steps from a knowledge base to
enhance the language model’s prompt and also enabled the system to generalize to new areas. However,
Cheng et al. [20] showed that even with RAG, models can still generate unsupported information. They
introduced RAGTruth, a new dataset of nearly 18,000 manually annotated responses to measure and
address hallucinations in Retrieval-Augmented Generation (RAG) models. Another prevalent strategy
for reducing hallucinations is to use Large Language Models (LLMs) to verify the output of other LLMs.
While these LLM evaluators are efficient for checking general coherence or relevance, relying on them to
judge factual accuracy is risky. This is because a faulty LLM may reinforce another’s errors, creating
an “echo chamber” of misinformation without a ground truth. Tang et al. [21] addressed the high
computational cost of fact-checking LLM outputs using other LLM like GPT-4. To solve this, they
created MiniCheck, a small fact-checking model that achieves GPT-4-level performance at a cost that
is 400 times lower. They accomplished this by generating a new synthetic training dataset using GPT-
4, which helped the model learn to identify realistic and challenging factual errors. Ouyang et al [22]
also noted that simply scaling up language models doesn’t guarantee they will follow a user’s intent,
as they can produce unhelpful, untruthful, or toxic content. They introduced InstructGPT, a family
of models fine-tuned on human feedback to better align with user intent. They collected a dataset of
human rankings of model outputs and trained a reward model (RM) to predict which output a human
would prefer. However, they acknowledged that InstructGPT still made mistakes. Farquhar et al.
[23] developed a new, unsupervised technique to spot a particular kind of factual error in LLMs called
“confabulations.” These errors are arbitrary, incorrect answers that are highly sensitive to irrelevant
details, like a random seed. The central idea of the suggested technique is semantic entropy, which is a
measure of uncertainty calculated based on the meaning of the generated sentences, not just the individual
words or tokens. This is achieved by generating several answers to a question and then grouping them
based on whether they share the same semantic equivalence. Lavrinovics et al. [24] addressed the key
flaw of hallucinations in Large Language Models (LLMs) and proposed Knowledge Graphs (KGs) as a
structured external knowledge source to mitigate this issue.
Some researchers believe that the key to eliminating hallucinations is to simply make models more
accurate, assuming that a perfectly accurate model would never invent information [25]. However, this
claim overlooks a fundamental limitation: 100% accuracy is an impossible goal [26, 27]. No matter how
large or capable a model becomes, it will always encounter questions that are inherently unanswerable
in the real world. So, does this mean hallucinations are an inevitable side effect of language models?
The answer is “NO”. We can teach LLMs to refrain from answering or simply reply with “I don’t
know” when they lack sufficient information. Kalai et al. [25] modified standard evaluation benchmarks
to encourage and reward Large Language Models (LLMs) for expressing uncertainty—for instance, by
giving partial credit for responding with “I don’t know” or choosing to abstain from answering. They
suggested adding explicit confidence targets to the evaluation prompts, which would guide models to
only respond when they are sufficiently confident, thereby discouraging them from overconfident bluffing
and, in turn, reducing hallucinations.
3 Proposed Method: Domain-Grounded Tiered Retrieval Tech-
nique
To mitigate the persistent issue of hallucinations in LLMs, we propose a domain-grounded, tiered retrieval
and verification architecture designed to detect and intercept the generation of factual inaccuracies.
4

Instead of blindly searching the web for every question (which is expensive and slow), it acts like a
human test-taker. First, it tries to answer from memory (closed-book). If it is absolutely certain it
knows the answer, it stops. If it is uncertain or fails a strict constraint check, it opens the book (searches
the web), but it does so in stages—checking trusted sources before falling back to the open web. If it
still can’t find the truth, it gracefully admits defeat rather than guessing. The proposed algorithm has
been divided into four distinct phases as follows:
•Phase I: Intrinsic Verification and Early-Exit Logic
To optimize computational efficiency and reduce unnecessary API overhead, the system initially
attempts to resolve the query using its internal parametric memory through a zero-shot baseline,
Ainit. This process begins by decomposing the generated response into discrete, verifiable atomic
claims, which are then cross-referenced against the original query to identify potential constraint
violations. If the response adheres to all predefined rules and constraints, a specialized Intrinsic
Critic evaluates the factual reliability of the claims [28]. If the resulting internal confidence score
Sintrinsic meets or exceeds the established thresholdτ, the system triggers an early-exit mechanism,
bypassing the retrieval pipeline to provide an immediate, high-confidence final response.
•Phase II: Adaptive Search Routing and Domain-Specific Retrieval
When the initial intrinsic confidence score fails to meet the required thresholdτ, the system transi-
tions from internal memory to an adaptive, tiered retrieval process designed to ground the response
in external evidence. This phase begins with the Domain Detector, which classifies the user query
Qto identify the most relevant specialized archives, such as medical or legal databases. The routing
logic then follows a prioritized hierarchy of trust, initially executing a TrustedSearch within these
curated, subject-specific authorities to ensure maximum data integrity. If this primary tier fails
to yield sufficient information, the system dynamically adapts by falling back to a GeneralWeb-
Search. This systematic progression through layered databases ensures that the model balances
the precision of authoritative “Gold Standard” sources with the broad coverage of the open web,
only moving to less restricted tiers when specialized resources are exhausted.
•Phase III: Corrective Document Grading (CRAG)
Once the raw contextD rawis retrieved from either the trusted or general search tiers, the system
implements a Corrective Document Grading mechanism [29] to ensure the relevance and reliability
of the external data. During this phase, a specialized grader evaluates each retrieved document
against the original user queryQ, filtering out noise, irrelevant snippets, or contradictory informa-
tion to produce a refined set of high-quality context,D filtered . If the grading process determines
that the retrieved documents are empty or do not meet the necessary relevance criteria, the sys-
tem triggers a recursive loop to the next search tier. This selective filtering prevents “distractor”
information from entering the generation stage, thereby significantly reducing the likelihood of
retrieval-based hallucinations.
•Phase IV: Extrinsic Regeneration and Verification
In the final stage of the pipeline, the system utilizes the refined contextD filtered to perform
Extrinsic Regeneration, where a new, grounded answerA regen is synthesized. To ensure absolute
factual integrity, this regenerated response is decomposed into Atomic Claims [30]C regen which
are then rigorously cross-referenced against the retrieved evidence. A secondary validation score
Sretrieved is calculated to measure how well the external data supports each individual claim.
If this score meets the required confidence thresholdτ, the system outputs the verified RAG
answer. However, if the verification fails or if all search tiers have been exhausted without finding
corroborating evidence, a “circuit breaker” is triggered to return a Graceful Apology, prioritizing
the prevention of hallucination over the delivery of potentially false information.
5

Algorithm
Input: User Query Q, Confidence Thresholdτ(e.g., 70)
Output: Final Factual Answer A*
Initialize: trusted done = False, general done = False
/* 1. Intrinsic Verification and Early Exit */
1:A init←Generate(Q)// Zero-shot baseline
2:C←ExtractClaims(A init)
3:V←CheckConstraints(C, Q)
4:ifV contains no constraint violationsthen
5:S intrinsic ←ScoreIntrinsic(C)// Closed-book evaluation
6:ifS intrinsic ≥τthen
7:returnA init // Bypass retrieval, exit early
8:end if
9:end if
10:Domain←DetectDomain(Q)
11:whileTruedo
/* 2. Adaptive Search Routing */
12:ifnot trusted donethen
13:D raw←TrustedSearch(Q, Domain)
14:trusted done←True
15:else ifnot general donethen
16:D raw←GeneralWebSearch(Q)
17:general done←True
18:else
19:returnGenerateApology() // Circuit breaker triggered
20:end if
/* 3. Corrective Document Grading (CRAG) */
21:D filtered ←GradeDocuments(D raw, Q)
22:ifD filtered is emptythen
23:continue // Loop to next search tier
24:end if
/* 4. Extrinsic Regeneration & Verification */
25:A regen←Regenerate(Q, D filtered )
26:C regen←ExtractClaims(A regen )
27:S retrieved ←ScoreRetrieved(C regen, Dfiltered )
28:ifS retrieved ≥τthen
29:returnA regen // Verified RAG answer
30:else iftrusted done and general donethen
31:returnGenerateApology() // Exhausted all resources
32:end if
33:end while
Fig. 3 depicts the workflow of the proposed method, beginning with an ‘Intrinsic Confidence & Con-
straint Check’ to assess if the model can generate an accurate response from its internal memory. If
the confidence is low, a “Domain Detector” classifies the query and routes it through “Tiered External
Databases”, starting with curated authorities before falling back to a general web search. The archi-
tecture emphasizes verification by breaking regenerated answers into “Atomic Claims” that undergo an
“Extrinsic Confidence Check” to score retrieved facts, ultimately leading to a verified RAG answer or a
safe refusal if all tiers are exhausted.
4 Performance Evaluation
This section details the experimental configuration, the benchmarks utilized for stress-testing the archi-
tecture, and the metrics employed to quantify hallucination mitigation.
6

Atomic ClaimsHigh Confidence
No ViolationsRoutes to matching
DatabaseLow Confidence
or Violations
Hallucinated (Low Score)
Loop to Next Tier
All Tiers Exhausted
Verified (High Score)
Regenerated Text Raw ContextUser QueryLLM Generator & Intrinsic
Critic
(Generates Answer,
Extracts Claims,
Scores Confidence)
Claim Extractor
(Breaks Regenerated Answer
into Atomic Facts)Domain Detector
(Classifies Query for
Targeted Routing)
Regeneration Agent
(Filters Docs & Writes New
Answer)Tiered External Databases
Tier 1: Curated Authorities
Tier 2: General Web Search
Final Response
• Early-Exit Answer
• Verified RAG Answer
• Graceful ApologyExtrinsic
Confidence Check
(Score Retrieved
Claims)Intrinsic
Confidence &
Constraint CheckFigure 3: Proposed Framework
4.1 Experimental Configuration
The evaluation framework is constructed as a nine-stage directed graph implemented in LangGraph.
Each node corresponds to a distinct processing step, with conditional routing edges determining whether
the pipeline escalates from parametric memory to retrieval-grounded generation. Most internal tasks,
including answer generation, atomic claim extraction, and document relevance grading, are executed by
Llama 3.1 8B. This model is deployed locally via Ollama with a temperature of 0 to ensure deterministic
and consistent outputs. To keep the data organized and accurate during the checking phase, the system
uses Pydantic for structured validation. External data retrieval is facilitated via the Tavily search API,
which queries both subject-specific trusted repositories and the broader web. For the final evaluation,
Gemma3 27B serves as an independent, asymmetric judge to compare the pipeline’s performance against
a zero-shot baseline.
4.2 Benchmark Selection and Taxonomy
The proposed pipeline was evaluated across 650 queries derived from five diverse benchmarks, specif-
ically selected to stress-test the architecture’s ability to handle distinct factual deviations, temporal
constraints, and logical complexities (Table 2). The selection includes TimeQA v2 [31], which targets
temporal biographical reasoning through questions regarding roles held within specific date ranges, often
characterized by sparse documentation. FreshQA v2 [32] assesses the retrieval of current knowledge and
robustness against false premises. To evaluate broader capabilities, HaluEval General [33] provides a
large-scale assessment of open-domain hallucinations in science, coding, and general knowledge, while
MMLU Global Facts [34] focuses on precise numerical statistics that typically exceed the parametric
memory of smaller 8B models. Finally, TruthfulQA [11] is employed to measure the system’s resistance
to popular myths and common misconceptions, ensuring the architecture can identify and deny false
premises rather than succumbing to confabulation.
4.3 Evaluation Metrics
We use several key metrics to determine how well the new RAG system works compared to a standard
model. Below is a detailed explanation of these metrics:
7

Table 2: Benchmark summary
Benchmark N Domain Query Type Key Challenges
TimeQA v2 86* Temporal
biographical‘Who held role X
between dates A–B?’Sparse time-window
documentation; correct
refusal on unknowable
queries
FreshQA v2 150 Current events
+ false premisesFactual + denial queries 39% of queries embed
false or unresolved
premises; requires
refusal, not inference
HaluEval General 150 Open domain
(mixed)General knowledge,
science, codingBroad coverage; retrieval
advantage lower vs.
parametrically-strong
baseline
MMLU Global
Facts50 Global statistics Precise numerical recall Exact figures unavailable
in small-model memory;
short answers
TruthfulQA 150 Myths /
misconceptionsMyth-busting; denial of
false premisesMust resist
confabulation;
false-premise resistance
essential
* TimeQA: 150 questions attempted; 86 valid pipeline runs (64 invalidated by API quota exhaustion, recorded
as Ties).
(i)Primary Performance Metrics
•Win Rate: This is the percentage of total queries where the independent judge (Gemma3
27B) preferred the answer generated by the RAG pipeline over the standard, “zero-shot”
answer provided by Llama 3.1 8B. A higher win rate proves that the retrieval steps successfully
added valuable information that the model did not already know.
•Groundedness Score: This metric tracks the portion of individual factual statements
(atomic claims) in the final answer that are directly supported by the retrieved documents.
It serves as a measure of how well the model actually uses the evidence it finds; therefore, a
higher score is better.
•Hallucination Rate: This is the opposite of groundedness. It represents the fraction of
factual claims made by the model that cannot be found in or verified by the retrieved evidence.
To ensure the system is reliable and not “making things up,” a lower value is better.
(ii)Supplementary Metrics
•Tie Rate: This indicates the frequency with which the judge found both the pipeline’s
answer and the baseline answer to be of equal quality. This often happens when the model is
already confident and skips retrieval, or when both models correctly identify that a question
is unanswerable.
•Baseline Win Rate: This records the instances where the judge actually preferred the
simple, zero-shot answer over the one provided by the complex RAG pipeline. This usually
occurs due to “overclaiming,” where the pipeline tries to force a factual answer onto a trick
question or a false premise.
5 Discussion and Analysis
The proposed architecture consistently outperformed the zero-shot baseline across all tested environ-
ments, with win rates ranging from 50.0% to 83.7% (Table 3). Fig. 4 illustrates the comparative
performance of the proposed RAG pipeline versus the zero-shot baseline. This section analyzes the per-
formance drivers, systemic failure modes, and the stability of groundedness across diverse NLP domains.
8

5.1 Retrieval Advantage and Query Specificity
The evaluation demonstrates that retrieval advantage is strongly correlated with the degree to which
a query exceeds the model’s internal training data. This relationship follows the “specificity-advantage
curve,” where the pipeline performs best on highly specific queries, particularly those involving narrow
temporal or numerical information. The curve is characterized by two factors: specificity (narrowness of
the requested fact), and recency (likelihood of post-training information change).
•High-Advantage Queries: The highest win rates were observed in TimeQA v2 (83.7%) and
MMLU Global Facts (78.0%). These benchmarks require granular temporal biographical data or
precise numerical statistics—information that is either absent or poorly encoded in a small model
like Llama 3.1 8B.
•Competitive Parametric Memory: In HaluEval General, the win rate dropped to 50.0%.
This suggests that for common-knowledge or procedural tasks (e.g., coding or general science), the
baseline’s parametric memory is a strong competitor, and retrieval provides a more modest benefit.
•Adaptive Efficiency: The pipeline successfully bypassed retrieval for 20% of HaluEval queries via
the intrinsic halting criterion, proving the architecture can avoid unnecessary latency for creative
or procedural tasks without sacrificing quality.
Table 3: Primary results across all five benchmarks
Benchmark NProposed
WinsTie Baseline
WinsWin
RateHallucination Groundedness
TimeQA v2 86 72 6 4 83.7% 13.6% 86.4%
MMLU Global
Facts50 39 8 3 78.0% 33.1% 66.9%
FreshQA v2 150 97 37 16 64.7% 3.5%† 19.2%†
TruthfulQA 150 82 56 12 54.7% 15.1% 84.9%
HaluEval Gen-
eral150 75 45 30 50.0% 21.2% 78.8%
Combined
(650)650 365 152 133 56.2% − −
†FreshQA all-150 figures include 116 correct denial answers (0/0 scores by design). FreshQA factual-answer
subset (n=34): hallucination 15.5%, groundedness 84.5%.
5.2 Groundedness Stability Across Domains
While hallucination rates appeared to vary, groundedness—the fraction of claims supported by retrieved
evidence—showed remarkable consistency.
•Consistent Performance: On factual-answer rows, groundedness remained stable between 78.8%
and 86.4% across all five benchmarks. This indicates that the retrieval-to-grounding mechanism is
robust and domain-agnostic.
•Measurement Sensitivity in MMLU: The higher reported hallucination rate in MMLU Global
Facts (33.1%) is attributed to measurement sensitivity rather than factual failure. Because numer-
ical answers are short (2–3 atomic claims), any slight difference in units or data vintage between
the retrieved source and the generated answer triggers a high hallucination score.
5.3 Failure Mode Characterization
The empirical evaluation of the 650 queries identified six primary categories of failure that account for
the 133 competitive losses recorded against the baseline. Table 4 presents these failure modes with
their frequency, affected benchmarks, and root causes. Understanding these patterns is essential for
transitioning the architecture toward a more robust, “denial-aware” state machine.
9

Figure 4: Benchmark Win Rate Comparison
(i)Open-Domain Parametric Competition: This failure mode occurs when the baseline model’s
internal training data is already sufficient to provide a high-quality answer without external assis-
tance. For well-established scientific facts or common historical knowledge, the retrieval process
adds computational overhead without providing a measurable improvement in accuracy. This
was most prevalent in the HaluEval benchmark, where the baseline often matched the quality of
retrieved web sources. To address such failures, the system should use a dynamic weighting mecha-
nism that compares the Intrinsic Confidence Score (S intrinsic ) with the potential information gain
from retrieval. If the model is highly confident about a common-knowledge query, the system
should prioritize a zero-shot response to reduce latency and avoid retrieval-induced noise.
(ii)False-Premise Overclaiming: The system occasionally prioritizes contextually related informa-
tion over the recognition of impossible or unanswerable query premises. This leads to the generation
of a confident but factually incorrect response that validates a false query. In FreshQA, when asked
about Lionel Messi winning a second World Cup, the pipeline retrieved data about the 2022 win
but failed to notice the count was incorrect, “confirming” the second win. Similarly, in TimeQA,
the system inferred roles from adjacent biographical data rather than admitting a lack of evidence
for a specific time window. A dedicated Answerability Node should act as a gatekeeper that checks
whether a query is based on a valid premise. It should utilize Pydantic-based structured outputs
to verify the truthfulness of the query. If the premise is false or impossible, the system redirects
the query to a refusal-generation path instead of producing an overclaim.
(iii)Vagueness (Verbose vs. Concise Denial): Vagueness occurs when the generator identifies
that an event did not happen but produces a long, heavily hedged response. While these multi-
paragraph answers are factually defensible, evaluation judges often prefer the baseline’s direct and
simple “No”. This behavior is largely a consequence of the smaller 8B parameter model’s limited
ability to synthesize concise refusals under complex retrieval constraints. Such verbosity can reduce
the perceived utility of the system for end-users seeking clear information. Therefore, scaling the
generation nodes to a mid-size model (such as Llama 3.3 70B) is recommended. Larger models
typically exhibit better instruction-following capabilities, allowing them to produce concise and
direct denials without excessive hedging.
(iv)Retrieval Distraction: In this failure mode, the system retrieves a document that contains tan-
gential or irrelevant context which misleads the generator. Even if the correct answer is present
in the source, the model may focus on a side topic and produce a response that diverges from the
10

user’s core intent [35]. This highlights a vulnerability in the model’s attention mechanism when
dealing with long or multi-topic retrieved passages. It demonstrates that more information does
not always result in a more accurate output if the model cannot distinguish signal from noise.
Enhancing the Corrective Document Grading (CRAG) logic is necessary to improve context isola-
tion. Future work should also explore more aggressive “literalness” instructions for the generator
to ensure it ignores tangential data in favor of the primary factual evidence.
(v)Numerical Precision and Data Mismatch: This category involves failures where the retrieved
source and the final generated claim use different units, aggregation methods, or timeframes. In
MMLU Global Facts, a source might provide data from one year while the query asks for another,
or use different statistical denominators. Because the system relies on exact matching for verifica-
tion, these minor discrepancies result in high hallucination scores even when the answer is nearly
correct. Therefore, evaluation metrics should transition from rigid keyword comparisons to numer-
ical proximity scoring. This would allow the system to better accommodate small discrepancies in
data chronology or precision that do not constitute a genuine hallucination.
(vi)Structured Data Extraction Errors: Failures in this mode result from the model’s inability to
correctly interpret or extract values from structured documents like markdown tables or lists. The
Llama 3.1 8B model occasionally misreads row/column relationships, leading to the extraction of
incorrect data points. This is an inherent limitation of smaller models, which lack the sophisticated
reasoning required to parse dense technical documentation. These errors suggest that scaling to
larger models is necessary for tasks involving complex document structures.
Table 4: Summary of Failure Modes
Failure Mode Frequency Primary Benchmarks Primary Root Cause
Open-Domain
Parametric
Competition∼30 HaluEval High-quality training data in the base-
line model matches the quality of re-
trieved web sources for common knowl-
edge.
False-Premise
Overclaiming∼14 TimeQA, FreshQA,
TruthfulQALack of a pre-retrieval answerability
check; the system answers the premise
instead of correcting it.
Vagueness
(Verbose vs.
Concise)∼12 FreshQA, TruthfulQA The generator produces long, hedged re-
sponses that the judge finds less useful
than the baseline’s concise denial.
Retrieval Dis-
traction∼9 TruthfulQA, HaluEval Tangential information in a retrieved
document leads the generator away from
the direct answer.
Numerical
precision /
Data mis-
match∼5 MMLU, FreshQA The retrieved source utilizes different ag-
gregation methods, timeframes, or units
than the generated claim.
Structured
Data Extrac-
tion Errors∼4 FreshQA, MMLU Small-model limitations (Llama 3.1 8B)
leading to misreading of markdown ta-
bles or extracting wrong values from doc-
uments.
Fig. 5 provides provides a granular visualization of how different failure modes cluster across our cho-
sen benchmarks. The heatmap reveals localized “stress points” in the architecture, allowing for targeted
optimization. False-Premise Overclaiming and Vagueness are most intense in refusal-heavy benchmarks
like TruthfulQA and FreshQA v2. Conversely, Numerical/Data Mismatch is almost exclusively con-
centrated in MMLU Global Facts, highlighting a measurement sensitivity issue in statistical queries.
Additionally, the high intensity of Parametric Competition in HaluEval General validates the efficiency
of the Intrinsic Halting Criterion, showing where internal memory is already sufficient. Ultimately, the
heatmap identifies specific targets for architectural remediation, such as the need for an “Answerability
Node” to handle the clusters of overclaiming errors.
11

Figure 5: Realibility Heatmap (Benchmarks vs Error Types)
6 Conclusions
The proposed algorithm functions as a robust, specialized retrieval framework designed for high-stakes
accuracy. By grounding its outputs in verified, domain-specific repositories, the system delivers author-
itative information instead of relying on broad, unverified web data. This tiered strategy prioritizes
the most reliable sources first, incorporating a built-in safety protocol to provide a polite disclaimer
if a factual basis cannot be confirmed. Evaluation of 650 queries across five diverse NLP benchmarks
demonstrates that a multi-stage RAG pipeline significantly boosts the factual grounding of local models.
The pipeline consistently and substantially outperforms the zero-shot baseline, particularly in domains
demanding high specificity and temporal recency. Win rates peaked at 83.7% for TimeQA v2 and 78.0%
for MMLU Global Facts, illustrating the pipeline’s effectiveness in retrieving granular biographical and
numerical data that typically exceeds a model’s parametric memory. Conversely, the modest 50.0% win
rate in HaluEval General confirms that retrieval augmentation provides a more limited benefit for open-
domain common knowledge already encoded in large-scale training data. A critical architectural success
was the implementation of adaptive halting criteria. The intrinsic halting mechanism correctly bypassed
retrieval for 20% of open-domain queries involving procedural or creative tasks, thereby reducing latency
and mitigating retrieval noise. Furthermore, the consistency of groundedness scores (78.8% to 86.4%)
on factual queries across all benchmarks indicates that the pipeline’s verification loops provide stable,
domain-agnostic grounding quality. However, the evaluation also identified few failure patterns including
the false-premise overclaiming issue. Across multiple benchmarks, the system occasionally prioritized
contextually related information over recognizing impossible or unanswerable query premises. This high-
lights a specific need for a dedicated pre-retrieval answerability check in future iterations. Bridging the
gap between human-level verification and automated generation is essential for neutralizing hallucina-
tions. This shift will allow the AI community to create conversational models that are as credible and
secure as they are articulate.
12

References
[1] Haque, M. “A Brief Analysis of “ChatGPT” – A Revolutionary Tool Designed by OpenAI”. In:
EAI Endorsed Transactions on AI and Robotics(Mar. 2023), e15.
[2] Haque, M. and Li, S. “Exploring ChatGPT and its impact on society”. In:AI and Ethics5 (Feb.
2024), pp. 791–803.
[3] Haque, M. A. “LLMs: A game-changer for software engineers?” In:BenchCouncil Transactions on
Benchmarks, Standards and Evaluations5.1 (2025), p. 100204.url:https://www.sciencedirect.
com/science/article/pii/S2772485925000171.
[4] Huang, L. et al. “A Survey on Hallucination in Large Language Models: Principles, Taxonomy,
Challenges, and Open Questions”. In:ACM Trans. Inf. Syst.43.2 (Jan. 2025).url:https://doi.
org/10.1145/3703155.
[5] Alansari, A. and Luqman, H.Large Language Models Hallucination: A Comprehensive Survey.
2025.url:https://arxiv.org/abs/2510.06265.
[6] Rawte, V. et al. “The Troubling Emergence of Hallucination in Large Language Models - An
Extensive Definition, Quantification, and Prescriptive Remediations”. In: ed. by Bouamor, H. et al.
Singapore, Dec. 2023, pp. 2541–2573.url:https://aclanthology.org/2023.emnlp-main.155/.
[7] Geman, S. et al. “Neural Networks and the Bias/Variance Dilemma”. In:Neural Computation4
(Jan. 1992), pp. 1–58.
[8] Barkley, L. and Van Der Merwe, B.Investigating the Role of Prompting and External Tools in
Hallucination Rates of Large Language Models. Oct. 2024.
[9] Bender, E. M. et al. “On the Dangers of Stochastic Parrots: Can Language Models Be Too Big?”
In: FAccT ’21. Virtual Event, Canada, 2021, pp. 610–623.url:https://doi.org/10.1145/
3442188.3445922.
[10] Maynez, J. et al. “On Faithfulness and Factuality in Abstractive Summarization”. In: ed. by Ju-
rafsky, D. et al. Online, July 2020, pp. 1906–1919.url:https://aclanthology.org/2020.acl-
main.173/.
[11] Lin, S. et al. “TruthfulQA: Measuring How Models Mimic Human Falsehoods”. In: ed. by Muresan,
S. et al. Dublin, Ireland, May 2022, pp. 3214–3252.url:https://aclanthology.org/2022.acl-
long.229/.
[12] Sun, J. et al.DynamicRAG: Leveraging Outputs of Large Language Model as Feedback for Dynamic
Reranking in Retrieval-Augmented Generation. 2025.url:https://arxiv.org/abs/2505.07233.
[13] Chen, Y. et al.Improving Retrieval-Augmented Generation through Multi-Agent Reinforcement
Learning. 2025.url:https://arxiv.org/abs/2501.15228.
[14] Zheng, L. et al. “Judging LLM-as-a-judge with MT-bench and Chatbot Arena”. In: NIPS ’23. New
Orleans, LA, USA, 2023.
[15] Manakul, P. et al. “SelfCheckGPT: Zero-Resource Black-Box Hallucination Detection for Genera-
tive Large Language Models”. In: ed. by Bouamor, H. et al. Singapore, Dec. 2023, pp. 9004–9017.
url:https://aclanthology.org/2023.emnlp-main.557/.
[16] Liu, Y. et al. “G-Eval: NLG Evaluation using Gpt-4 with Better Human Alignment”. In: ed. by
Bouamor, H. et al. Singapore, Dec. 2023, pp. 2511–2522.url:https://aclanthology.org/2023.
emnlp-main.153/.
[17] Kryscinski, W. et al. “Evaluating the Factual Consistency of Abstractive Text Summarization”.
In: ed. by Webber, B. et al. Online, Nov. 2020, pp. 9332–9346.url:https://aclanthology.org/
2020.emnlp-main.750/.
[18] Laban, P. et al. “SummaC: Re-Visiting NLI-based Models for Inconsistency Detection in Sum-
marization”. In:Transactions of the Association for Computational Linguistics10 (2022). Ed. by
Roark, B. and Nenkova, A., pp. 163–177.url:https://aclanthology.org/2022.tacl-1.10/.
[19] Ayala, O. and Bechard, P. “Reducing hallucination in structured outputs via Retrieval-Augmented
Generation”. In: ed. by Yang, Y. et al. Mexico City, Mexico, June 2024, pp. 228–238.url:https:
//aclanthology.org/2024.naacl-industry.19/.
13

[20] Niu, C. et al. “RAGTruth: A Hallucination Corpus for Developing Trustworthy Retrieval-Augmented
Language Models”. In: ed. by Ku, L.-W. et al. Bangkok, Thailand, Aug. 2024, pp. 10862–10878.
url:https://aclanthology.org/2024.acl-long.585/.
[21] Tang, L. et al. “MiniCheck: Efficient Fact-Checking of LLMs on Grounding Documents”. In:
ed. by Al-Onaizan, Y. et al. Miami, Florida, USA, Nov. 2024, pp. 8818–8847.url:https://
aclanthology.org/2024.emnlp-main.499/.
[22] Ouyang, L. et al.Training language models to follow instructions with human feedback. 2022.url:
https://arxiv.org/abs/2203.02155.
[23] Farquhar, S. et al. “Detecting hallucinations in large language models using semantic entropy”. In:
Nature630 (June 2024), pp. 625–630.
[24] Lavrinovics, E. et al. “Knowledge Graphs, Large Language Models, and Hallucinations: An NLP
Perspective”. In:Journal of Web Semantics85 (2025), p. 100844.url:https://www.sciencedirect.
com/science/article/pii/S1570826824000301.
[25] Kalai, A. T. et al.Why Language Models Hallucinate. 2025.url:https://arxiv.org/abs/2509.
04664.
[26] Xu, Z. et al.Hallucination is Inevitable: An Innate Limitation of Large Language Models. 2025.
url:https://arxiv.org/abs/2401.11817.
[27] Banerjee, S. et al.LLMs Will Always Hallucinate, and We Need to Live With This. 2024.url:
https://arxiv.org/abs/2409.05746.
[28] Asai, A. et al. “Self-RAG: Learning to Retrieve, Generate, and Critique through Self-Reflection”. In:
ed. by Kim, B. et al. Vol. 2024. 2024, pp. 9112–9141.url:https://proceedings.iclr.cc/paper_
files/paper/2024/file/25f7be9694d7b32d5cc670927b8091e1-Paper-Conference.pdf.
[29] Yan, S.-Q. et al.Corrective Retrieval Augmented Generation. 2024.url:https://arxiv.org/
abs/2401.15884.
[30] Chern, I.-C. et al.FacTool: Factuality Detection in Generative AI – A Tool Augmented Framework
for Multi-Task and Multi-Domain Scenarios. 2023.url:https://arxiv.org/abs/2307.13528.
[31] Chen, W. et al. “A Dataset for Answering Time-Sensitive Questions”. In: ed. by Vanschoren, J.
and Yeung, S. Vol. 1. 2021.url:https://datasets-benchmarks-proceedings.neurips.cc/
paper_files/paper/2021/file/1f0e3dad99908345f7439f8ffabdffc4-Paper-round2.pdf.
[32] Vu, T. et al. “FreshLLMs: Refreshing Large Language Models with Search Engine Augmentation”.
In: ed. by Ku, L.-W. et al. Bangkok, Thailand, Aug. 2024, pp. 13697–13720.url:https : / /
aclanthology.org/2024.findings-acl.813/.
[33] Li, J. et al. “HaluEval: A Large-Scale Hallucination Evaluation Benchmark for Large Language
Models”. In: ed. by Bouamor, H. et al. Singapore, Dec. 2023, pp. 6449–6464.url:https : / /
aclanthology.org/2023.emnlp-main.397/.
[34] Hendrycks, D. et al. “Measuring Massive Multitask Language Understanding”. In:Proceedings of
the International Conference on Learning Representations (ICLR)(2021).
[35] Amiraz, C. et al. “The Distracting Effect: Understanding Irrelevant Passages in RAG”. In: ed. by
Che, W. et al. Vienna, Austria, July 2025, pp. 18228–18258.url:https://aclanthology.org/
2025.acl-long.892/.
14