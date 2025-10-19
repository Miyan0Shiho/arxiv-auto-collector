# Mapping Smarter, Not Harder: A Test-Time Reinforcement Learning Agent That Improves Without Labels or Model Updates

**Authors**: Wen-Kwang Tsao, Yao-Ching Yu, Chien-Ming Huang

**Published**: 2025-10-16 17:17:00

**PDF URL**: [http://arxiv.org/pdf/2510.14900v1](http://arxiv.org/pdf/2510.14900v1)

## Abstract
The Enterprise Intelligence Platform must integrate logs from numerous
third-party vendors in order to perform various downstream tasks. However,
vendor documentation is often unavailable at test time. It is either misplaced,
mismatched, poorly formatted, or incomplete, which makes schema mapping
challenging. We introduce a reinforcement learning agent that can self-improve
without labeled examples or model weight updates. During inference, the agent:
1) Identifies ambiguous field-mapping attempts. 2) Generates targeted
web-search queries to gather external evidence. 3) Applies a confidence-based
reward to iteratively refine its mappings. To demonstrate this concept, we
converted Microsoft Defender for Endpoint logs into a common schema. Our method
increased mapping accuracy from 56.4\%(LLM-only) to 72.73\%(RAG) to 93.94\%
over 100 iterations using GPT-4o. At the same time, it reduced the number of
low-confidence mappings requiring expert review by 85\%. This new approach
provides an evidence-driven, transparent method for solving future industry
problems, paving the way for more robust, accountable, scalable, efficient,
flexible, adaptable, and collaborative solutions.

## Full Text


<!-- PDF content starts -->

Mapping Smarter, Not Harder: A Test-Time Reinforcement Learning
Agent That Improves Without Labels or Model Updates
Wen-Kwang Tsao *, Yao-Ching Yu *, Chien-Ming Huang
AI Lab, TrendMicro
{spark_tsao,yaoching_yu}@trendmicro.com
Abstract
The Enterprise Intelligence Platform must in-
tegrate logs from numerous third-party ven-
dors in order to perform various downstream
tasks. However, vendor documentation is of-
ten unavailable at test time. It is either mis-
placed, mismatched, poorly formatted, or in-
complete, which makes schema mapping chal-
lenging. We introduce a reinforcement learn-
ing agent that can self-improve without labeled
examples or model weight updates. During
inference, the agent first identifies ambiguous
field-mapping attempts, then generates targeted
web-search queries to gather external evidence,
and finally applies a confidence-based reward
to iteratively refine its mappings. To demon-
strate this concept, we converted Microsoft
Defender for Endpoint logs into a common
schema. Our method increased mapping accu-
racy from 56.4% (LLM-only) to 72.73% (RAG)
to 93.94% over 100 iterations using GPT-4o.
At the same time, it reduced the number of
low-confidence mappings requiring expert re-
view by 85%. This new approach provides an
evidence-driven, transparent method for solv-
ing future industry problems, paving the way
for more robust, accountable, scalable, efficient,
flexible, adaptable, and collaborative solutions.
1 Introduction
Enterprise IT environments are rapidly evolving
toward proactive, agent-driven workflows. At the
core of these systems lies a foundational require-
ment: the ability to efficiently process and seman-
tically interpret logs from a wide range of third-
party sources. Such capability enables agents to
remain context-aware, access real-time data, and
make well-informed decisions.
For cybersecurity use cases, enterprises must
ingest massive volumes of logs—often terabytes
per day—from heterogeneous sources such as fire-
walls, servers, endpoints, cloud applications, net-
*Corresponding author.work flows, policy events, file operations, and
API calls, in order to enable effective security
operations and real-time threat detection (Reid,
2023). The cost of Security Information and
Event Management (SIEM) ingestion is substan-
tial, exceeding $500k annually for just 500 GB
per day (Hazel, 2021; Microsoft, 2024). The chal-
lenge lies not only in scale, but also in achieving
semantic consistency across dozens to hundreds
of disparate event types. Failures in log correla-
tion and schema normalization have contributed
to catastrophic breaches—including Target (2013)
and Equifax (2017)—where overlooked alerts and
misaligned schemas resulted in damages exceeding
$1 billion (Krishna, 2023).
The emergence of large language models
(LLMs) with robust natural language processing ca-
pabilities has the potential to transform third-party
log integration. These models could dramatically
reduce the need for labor-intensive processes that
require experts. The full log integration pipeline
comprises four stages: processing raw logs into
structured data, mapping source schemas to target
schemas, generating data transformation code, and
deploying to production with ongoing monitoring.
Schema mapping is the critical decision point that
underpins the success of the entire integration pro-
cess. This paper focuses on the practical challenge
of schema mapping, a topic that is often overlooked
in current literature.
Our focus is on a scenario in which enterprises
ingest new data into their platforms. Therefore,
the target schema is usually well-documented and
stable. By contrast, incoming data source schemas
are often poorly documented, typically due to
their origin in legacy systems or outdated software
versions. Unlike conventional machine learning
tasks—where the challenge is extracting key fea-
tures from abundant data—the difficulty here is the
opposite: the source provides too little context. Be-
cause these vendor schemas lack labeled trainingarXiv:2510.14900v1  [cs.AI]  16 Oct 2025

data, fine-tuning or other supervised methods are
impractical. Although expert review remains neces-
sary, it is resource-intensive and must be carefully
prioritized.
To address these challenges, we propose a test-
time reinforcement learning (RL) agent that can
self-improve without updating model weights or re-
lying on pre-defined labeled data. Our approach is
inspired by the TTRL framework (Zuo et al., 2025),
which improves model performance on unlabeled
data through test-stage rewards. In the absence
of ground-truth labels, we introduce a confidence-
based score as a proxy reward signal to guide the
agent toward more accurate mappings. Our design
explicitly targets industrially practical constraints:
model updates are costly, GPU-intensive, and op-
erationally complex. Instead, our method adapts
the system prompt at inference time by iteratively
enriching the context in a verbal RL-driven manner,
thereby enabling continual self-improvement under
real-world deployment conditions.
The confidence-based proxy reward is both in-
tuitive and interpretable. The agent identifies con-
flicts and ambiguities in its prior mapping attempts
and formulates targeted search queries to gather
external evidence. Confidence scores then serve as
reward signals to determine whether the collected
evidence should be retained or discarded, guiding
iterative refinement. Furthermore, the agent pro-
duces transparent reasoning traces, enabling human
experts to focus their review on low-confidence
mappings, thereby reducing manual verification
costs while preserving overall reliability.
This method has several advantages over existing
approaches. First, it achieves significantly higher
mapping accuracy through iterative refinement, re-
ducing reliance on static, one-time attempts. Sec-
ond, it uses confidence-based rewards instead of
ground truth labels, enabling effective learning in
settings where labeled data is unavailable. Finally,
it promotes transparency by revealing its reasoning
and evidence collection processes. This empow-
ers security experts to understand decisions and
prioritize low-confidence mappings.
Overall, our key contributions are: 1) identifying
the challenge of schema mapping where prompting,
fine-tuning, and retrieval-augmented generation all
face limitations and no ground truth labels are avail-
able, and 2) proposing a test-time reinforcement
learning framework that enables an agent to im-
prove accuracy over time. This approach opens a
new research direction by using confidence as aproxy signal to guide the agent’s learning process.
1Log Raw
CEF:0|h1|h2|001|5|
LocalPort=443
Log Parsed
Vendor=h1
Product=h2
DeviceVersion=001
Severity=5
LocalPort=443Target schema
Common Schema fields
•vendor
•pname
•severity
•src
•spt
•dst
•dptKB
AI
Response: the answer is …src more context
- layer: network 
- type:array[string] 
- desc:
the source IP address
RQ1: Can we detect the ambiguity in the 
LocalPort  mapping?
RQ2: Can we improve without label data 
at test time?
RQ3: What if we don’t have high -quality 
documents in our internal KB?Prompt: 
Which field in target schema 
should  LocalPort  map to ?
Context:A: Log parsing
B: Field mappingC: Field retrieve for augment generation
D: E2e test -time reinforcement learning agent
Figure 1: Position of our test-time RL agent relative
to prior work in the schema matching pipeline. In
Section A, Logparser-LLM Zhong et al. (2024) han-
dles raw-to-structured parsing. In section B, Schema-
Matching-LLM Parciak et al. (2024) shows baseline
one-shot mapping capability. In section C, ReMatch
Sheetrit et al. (2024) adds retrieval when full documen-
tation exists. MatchMaker Seedat and van der Schaar
(2024) pre-embeds target schemas for reuse. In sec-
tion D, Self-consistency Wang et al. (2022) improves
chain-of-thought reasoning. Search-R1 Jin et al. (2025)
enhances LLM reasoning with search-augmented RL.
This example shows that the log LocalPort is ambiguous
for decision making; we need more context to deter-
mine whether this field should map to src or dst. Our
research uniquely extends rightward beyond traditional
enterprise knowledge bases to handle newly seen logs
with ill-formatted or incomplete documentation. Unlike
fine-tuning approaches that require labeled data and risk
overfitting, our agent operates without test-time labels,
conducting internet searches to gather evidence outside
the enterprise KB scope. This addresses the critical
gap where traditional methods fail on unseen vendor
schemas with insufficient documentation.
2 Related Work
The foundation of schema mapping begins with
converting raw logs into structured data. Zhong
et al. (2024) introduced LogParser-LLM, which
addresses the initial step of our pipeline by trans-
forming unstructured log messages into key-value
pairs. Their hybrid approach combines an LLM-
based template extractor with a prefix-tree cluster-
ing method, achieving efficient parsing; the authors
report ~272.5 LLM calls amortized across large
log sets. This work eliminates the need for hyper-

parameter tuning or labeled data, enabling rapid
adaptability to new log formats.
LogParser-LLM’s contribution is complemen-
tary to our work: while they focus on the raw-to-
structured parsing phase. In contrast, our approach
begins where their process ends: once logs have
already been parsed into structured records, we
take these structured logs and further map them
into standardized, common schemas that enable
consistent downstream analysis, correlation, and
integration across diverse sources.
Parciak et al. (2024) conducted a comprehensive
experimental study of LLM capabilities in schema
matching, focusing on off-the-shelf performance
using only element names and descriptions. Their
work provides crucial insights into the baseline
capabilities of LLMs for schema matching tasks,
demonstrating that context scope significantly af-
fects performance—neither too little nor too much
context leads to optimal results.
Their findings directly inform our approach in
several ways: (1) they validate that LLMs can per-
form meaningful schema matching without requir-
ing data instances, which aligns with our privacy-
sensitive enterprise scenarios; (2) their context opti-
mization insights guide our prompt engineering;
and (3) their baseline performance metrics pro-
vide a foundation for measuring the improvements
achieved by our reinforcement learning approach.
However, their work focuses on one-shot matching
capabilities, while our approach addresses the it-
erative improvement challenge through test-time
learning.
ReMatch, introduced by Sheetrit et al. (2024),
leverages retrieval-augmented prompting to reduce
the target schema search space before matching.
Their approach assumes the availability of compre-
hensive documentation and uses embedding-based
retrieval to balance recall and precision in schema
matching tasks. ReMatch demonstrates strong per-
formance in healthcare schema matching scenarios
where complete documentation is available.
Our work differs from ReMatch in a fundamental
assumption: while ReMatch operates in environ-
ments with well-documented schemas and compre-
hensive knowledge bases, our approach is designed
for practical enterprise scenarios where such docu-
mentation is often incomplete or unavailable. Re-
Match’s retrieval mechanism works within a closed
set of known mappings, whereas our agent dynam-
ically discovers and accumulates evidence from
external sources to handle previously unseen ven-dor schemas.
Seedat and van der Schaar (2024) introduced
MatchMaker, which decomposes schema match-
ing into candidate generation, refinement, and con-
fidence scoring steps using LLMs and vector re-
trieval. Their approach focuses on embedding tar-
get schemas for better retrieval and future reuse,
generating synthetic in-context examples on-the-fly
for zero-shot self-improvement.
MatchMaker’s compositional approach and con-
fidence scoring align with our methodology, but
their self-improvement mechanism differs signif-
icantly from ours. While MatchMaker generates
synthetic examples for in-context learning, our ap-
proach accumulates real-world evidence through
external search and interaction. MatchMaker’s fo-
cus on target schema embedding optimization com-
plements our source schema adaptation strategy,
suggesting potential for hybrid approaches in fu-
ture work.
Two key techniques underpin our reinforce-
ment learning framework: search-enhanced rea-
soning and self-consistency validation. Jin et al.
(2025) demonstrated how external search can en-
hance LLM reasoning capabilities, providing a
foundation for our evidence collection mechanism.
Their Search-R1 approach shows that targeted web
search can significantly improve reasoning accu-
racy in complex tasks.
Wang et al. (2022) established that consistency
across multiple reasoning paths serves as a reli-
able indicator of correctness, forming the basis of
our confidence-based reward system. Their self-
consistency approach demonstrates that taking the
majority answer from multiple sampled reasoning
paths significantly improves accuracy, which di-
rectly informs our confidence score calculation.
A closely related line of research is theReflexion
framework (Shinn et al., 2023), which introduces
verbal reinforcement learning as a way for language
agents to improve through self-generated feedback
rather than model weight updates. Reflexion agents
reflect on task feedback, store these reflections in
episodic memory, and leverage them to guide future
decisions. This approach demonstrates that linguis-
tic feedback—whether scalar or free-form—can
significantly enhance agent performance across se-
quential decision-making, coding, and reasoning
tasks, achieving state-of-the-art results on bench-
marks such as HumanEval. Our work shares a
similar philosophy in emphasizing memory- and
feedback-driven improvement at test time; how-

ever, we focus specifically on schema mapping un-
der industrial constraints where ground truth labels
and comprehensive documentation are unavailable.
In this setting, we extend the notion of verbal rein-
forcement learning by integrating external evidence
collection with confidence-based rewards to refine
mappings dynamically.
Our work addresses the critical gap where ex-
isting methods fail on newly encountered ven-
dor schemas with ill-formatted or incomplete doc-
umentation. Unlike prior approaches that op-
erate within controlled environments with well-
documented schemas, our test-time reinforcement
learning agent adapts dynamically by accumulating
external evidence without requiring model updates
or labeled training data.
3 Methodology
3.1 Problem Formulation
We formally define the schema mapping prob-
lem as follows. Given a source schema S=
{s1, s2, . . . , s n}and a target schema T=
{t1, t2, . . . , t m}, we seek to find a mapping func-
tionf:S→T∪ {∅} such that f(s i) =t jrepre-
sents a semantic correspondence between source
fieldsiand target field tj, orf(s i) =∅ indicates
no corresponding field exists in the target schema.
Our objective is to maximize the correctness
of this mapping function: max fPn
i=1 I[f(s i) =
f∗(si)]where f∗(si)represents the ground truth
mapping and I[·]is the indicator function. However,
in practical deployment scenarios—particularly
when processing third-party vendor logs ground
truth mappingsf∗are unavailable.
To address this challenge, we use confidence
scores as a proxy for correctness. We define con-
fidence C(f(s i))as the consistency of mapping
predictions across multiple inferences, serving as a
surrogate objective:max fPn
i=1C(f(s i)).
3.2 Test-Time Reinforcement Learning
Framework
Our test-time reinforcement learning agent im-
proves schema mapping policy’s accuracy by it-
eratively collecting evidence and reinforcing use-
ful context. The agent starts from an empty evi-
dence set, identifies inconsistencies through mul-
tiple mapping attempts, and formulates targeted
search queries for ambiguous fields. Collected evi-
dence is added to the context, and confidence-based
reward signals guide which evidence to retain ordiscard. The complete algorithm is detailed in Al-
gorithm 1.
3.3 Reinforcement Learning Formulation
To formalize our approach, we define the reinforce-
ment learning components as follows:
State: The current state stconsists of the schema
mapping hypothesis from source to target fields
and the collected evidence at time t. Specifically:
st={M t, Et}where Mtis the current mapping
hypothesis and Etis the set of evidence collected
up to timet.
Action: The action atinvolves selecting a field
or conflict to investigate and executing a targeted
search query. The action space includes all possible
fields that could be investigated and the possible
search queries that could be formulated.
Reward: The reward rtis derived from the
change in confidence score after adding new ev-
idence: rt=C t+1−C twhere Ctis the confidence
score at time t. The confidence score serves as a
proxy for accuracy when ground truth labels are
unavailable.
Policy: The agent’s policy π(a|s) specifies the
suggested mapping from source schema fields to
target schema fields. While the policy can be im-
plemented in various ways, one practical approach
is to leverage an LLM’s reasoning process. In this
setup, the LLM performs the mapping by combin-
ing its prior knowledge with contextual evidence,
enabling the system to detect conflicts and generate
targeted search queries. We define a conflict as
a disagreement among the nprompt-variant pre-
dictions for the same source field within a single
iteration (as opposed to cross-run drift).
Learning: The goal of learning is to adapt a
policy’s behavior in a beneficial direction by lever-
aging rewards that reflect the situation. Instead of
updating the LLM’s model weights, learning takes
place through the accumulation of useful evidence
in the agent’s context. Evidence that increases con-
fidence is preserved, while unhelpful evidence is
discarded. This process reflects a form of verbal,
memory-based learning in which the agent’s knowl-
edge base is continuously refined.
We use verbal RL in the Reflexion sense—policy
adaptation via context/memory rather than parame-
ter updates; rewards are intrinsic confidence deltas,
not ground-truth returns.

Algorithm 1Iterative RL Schema Mapping with Confidence Improvement
Inputs:Source schema S={s 1, . . . , s n}; target schema T={t 1, . . . , t m}; iteration limit α(default:
100); conflict detection attemptsn(default:3); initial evidence contextE=∅; generative LLMF.
Outputs:Mapping functionf; refined evidence contextE.
1:fori←1toαdo
2:f← F(S, E)// Generate initial mappings using current evidence
3:C←ConflictDetection(f, n)// Detect inconsistent mappings
4:foreach source fields i∈Cdo
5:Q si←QueryFormulation(s i)// Formulate search queries
6:e si←EvidenceCollection(Q si)// Collect external evidence
7:r si←ConfidenceEvaluation(f(s i), esi)// Evaluate new confidence
8:r prev←Confidence(f(s i))// Retrieve previous confidence
9:ifr si> r prevthen
10:E←ContextUpdate(E, e si, rsi)// Update context if confidence improves
11:end if
12:end for
13:end for
14:returnf,E
4 Experiment
4.1 Setup
We evaluated our approach using two schemas: the
sourceschema was theMicrosoft Defender for End-
pointschema containing 195 fields, and thetarget
was ourCommon Schemawith 137 fields. The
ground truthconsisted of 66 manually verified
field mappings, curated collaboratively by domain,
threat intelligence, and product experts to cover a
diverse range of field types and complexity levels.
We employedGPT-4oas the underlying model and
assessed performance using two key metrics: (1)
Confidence score—measuring the consistency of
predictions across multiple attempts within a single
iteration, with special handling for empty outputs;
and (2)Accuracy—defined as the percentage of
correctly predicted mappings among the 66 verified
pairs.
For each field mapping iteration, the model was
executed three times (n=3), and both the confidence
score and accuracy were computed. While accu-
racy depends on the ground truth and is used solely
for evaluation, confidence scores—which can be
calculated without ground truth—are leveraged to
guide the reinforcement learning process and are
therefore suitable for production deployment.
Confidence Score Calculation:We calcu-
late confidence as the consistency of predic-
tions across multiple attempts using a modified
frequency-based approach. For a given field, if
we collect predictions P= [p 1, p2, p3]acrossthree iterations, the confidence score is: C=
count(most_frequent_prediction)
adjusted_totalwhere the adjusted to-
tal accounts for empty predictions with reduced
weight (0.5 instead of 1.0). The design aligns with
the findings of Kalai et al. (2025), who argue that
current training and evaluation paradigms implic-
itly reward guessing behavior in language models,
leading to hallucinations. By contrast, our scoring
mechanism provides a quantitative incentive for
uncertainty acknowledgment, steering the model
toward more trustworthy behavior.
We prepare two baselines for comparison.Base-
line 1:We used GPT-4o with a single-shot prompt
containing only the field name and value, resulting
in an accuracy of 56.36%.Baseline 2:We used
GPT-4o with a single-shot prompt enhanced with
additional field descriptions, data types, and sam-
ple data context from our internal knowledge base,
resulting in an accuracy of 72.73%.
4.2 Performance Improvements
Starting from aBaseline 2:accuracy of 72.73%
with GPT-4o in the first iteration, our method
achieved significant improvements throughout the
experiment. By the end of the 100-iteration exper-
iment, the model demonstrated consistent perfor-
mance with accuracy reaching 93.94%.
Our analysis of the 100-iteration experiment re-
veals several key performance characteristics. GPT-
4o collects 81 evidence tuples across 100 iterations,
with the most significant accuracy gains occurring
in the early stages—for example, iteration 1 shows

Figure 2: Comprehensive performance analysis of the reinforcement learning agent over 100 iterations. Top:
Accuracy improves from 72.73% to 93.94%, while confidence trends upward and approaches 1.0, highlighting
a persistent overconfidence gap. Middle: Accuracy variability is high in early iterations but stabilizes over time.
Bottom: Conflict field count decreases from 26 to 4(85% reduction), demonstrating reduced ambiguity as evidence
accumulates.
a +21.21% gain, iteration 5 achieves +9.60%, and
iteration 6 yields +10.71%. The system makes de-
cisions solely based on confidence scores, without
referencing accuracy at any point during execution.
This reflects real-world conditions in which
ground truth labels are unavailable. In these sce-
narios, accuracy can only be measured retrospec-
tively to confirm whether the solution performed
correctly. Notably, in 19 of the 100 iterations, the
system rejected newly proposed evidence, demon-
strating that it learned to recognize when gather-
ing additional information was unlikely to improve
results. This selective behavior preserves a high-
quality evidence set, which is critical not only for
validating outcomes but also for providing trans-
parency into how the agent arrived at its decisions.
By the conclusion of the 100 iterations, the agent
resolved 22 conflicts, only 4 fields were flagged as
low-confidence and requiring expert review—down
from 26 initially—representing an 85% reductionin the verification burden on security analysts and
allowing them to focus on the most ambiguous or
critical cases.
To evaluate the robustness of our approach, we
ran the full 100-iteration process multiple times.
The final accuracy consistently reached 93-94%,
with minimal variance (standard deviation less than
0.01) in the later iterations. This demonstrates that
the improvement is reliable and not due to chance.
4.3 Evidence Collection Analysis
For this experiment, we equipped the AI agent with
a tool that enables internet searches for additional
facts and evidence. Specifically, we used a generic
search utility powered by Bing. In each iteration,
the agent analyzes conflicts identified across three
prompt variants, formulates a targeted search query,
and retrieves relevant results from the tool. This
setup demonstrates one possible method of evi-
dence collection; other approaches could include

Figure 3: The relationship between confidence and accu-
racy. Path 1 shows the starting point. Path 2 illustrates
how confidence as a proxy reward improves accuracy.
Path 3 highlights the challenge of overconfidence, where
confidence saturates while accuracy lags behind. There-
fore, more engineering and research efforts are needed
to bring the confidence curve down. The diagonal line
indicates perfect calibration, and the curves above it
show overconfidence.
consulting human experts, performing advanced
reasoning, or validating findings against produc-
tion data.
We organized the collected evidence into three-
element tuples, each comprising the detected con-
flict, the resolution plan, and the retrieved evidence.
This structured representation ensures systematic
evidence collection and evaluation. Evidence is re-
tained only if it demonstrably increases the LLM’s
confidence in its mapping decisions. Helpful ev-
idence contributes in 4 ways, (1) by enhancing
awareness through the identification of ambigu-
ous fields and conflicting mappings; (2) by en-
abling self-correction through improved reasoning
about relationships among fields; (3) by improving
clarity via authoritative information on how fields
are defined and used in both the common schema
and vendor-specific schemas (e.g., Microsoft De-
fender); and (4) by revealing context-dependent
mappings, in which the correct correspondence
varies by scenario and requires additional contex-
tual understanding.
5 Conclusion
This paper presents a test-time learning framework
in which a reinforcement learning agent improves
schema mapping accuracy through iterative evi-
dence collection and context refinement—without
requiring model weight updates or pre-collected
training data. Unlike conventional approaches
that rely on all labeled data before inference, ourmethod actively gathers new evidence during test-
time to refine its decisions. This design eliminates
the high computational cost, GPU dependency, and
potential instability associated with model retrain-
ing, making it more practical for real-world de-
ployment. A confidence-based score serves as a
proxy reward for accuracy, providing a novel mech-
anism that evaluates model consistency across mul-
tiple mapping attempts within a single iteration.
Through systematic evidence collection and evalu-
ation, the agent resolves mapping ambiguities with
transparent reasoning that facilitates expert review
and validation. Applied to Microsoft Defender
for Endpoint logs mapped to a common schema,
our approach improves accuracy from 72.73% to
93.94%, while reducing the number of fields requir-
ing expert review by 85%—from 26 to 4.
A key insight from our work is the use of confi-
dence scores as a proxy for ground-truth labels, as
conceptually illustrated in Figure 3. While confi-
dence can effectively guide accuracy improvement,
our results reveal persistent overconfidence, where
confidence values exceed actual accuracy. This
observation underscores the need for better confi-
dence definition and calibration. In our current ap-
proach, confidence is measured as the consistency
of predictions across three trials. Future research
could extend this by (1) employing multiple mod-
els to generate diverse predictions and compute
ensemble-based confidence, or (2) directly prompt-
ing LLMs to provide explicit self-assessed confi-
dence scores alongside their outputs. Additional
experiments extending the number of inference at-
tempts from three to ten show encouraging signs of
reducing the overconfidence gap (see Appendix B),
albeit with increased computational cost. These
enhancements could yield better-calibrated confi-
dence measures that more accurately reflect true
prediction quality.
Limitations
While our approach demonstrates significant im-
provements, several limitations should be acknowl-
edged:
1.Evidence Quality Dependency: The sys-
tem’s performance is dependent on the quality
and availability of external evidence sources.
In domains where documentation is sparse or
inconsistent, the improvement may be limited.
2.Computational Overhead: The iterative ev-
idence collection process requires multiple

LLM calls and external searches, which may
increase computational costs compared to
single-shot approaches.
3.Domain Specificity: Our evaluation focuses
on cybersecurity schemas. While the ap-
proach is general, validation in other domains
(healthcare, finance, etc.) would strengthen
the generalizability claims.
4.Scope Limitations: We focus on 1-to-1 map-
pings to address the core challenge of knowl-
edge scarcity. Extension to more complex
mapping cardinalities (1-to-N, N-to-M) re-
mains future work.
Ethical Considerations
This study does not involve human subjects or per-
sonal data. However, the method collects web
evidence at test time, which could occasionally
include malicious or adversarial content. Practi-
tioners reproducing this system should apply input
sanitization, source filtering, and sandboxing to
prevent prompt-injection or security risks.
References
Thomas Hazel. 2021. Harnessing the value of log data
analytics.Techstrong Research. Accessed: 2025-06-
23.
Bowen Jin, Hansi Zeng, Zhenrui Yue, Jinsung Yoon,
Sercan Arik, Dong Wang, Hamed Zamani, and Jiawei
Han. 2025. Search-r1: Training llms to reason and
leverage search engines with reinforcement learning.
arXiv preprint arXiv:2503.09516.
Adam Tauman Kalai, Ofir Nachum, Santosh S Vem-
pala, and Edwin Zhang. 2025. Why language models
hallucinate.arXiv preprint arXiv:2509.04664.
G. Krishna. 2023. Security logging and monitoring
failures: A comprehensive guide. Accessed: 2025-
06-23.
Microsoft. 2024. Microsoft sentinel pricing calculator.
Accessed: 2025-06-23.
Marcel Parciak, Brecht Vandevoort, Frank Neven,
Liesbet M Peeters, and Stijn Vansummeren. 2024.
Schema matching with large language models: an ex-
perimental study.arXiv preprint arXiv:2407.11852.
Colin Reid. 2023. Searching for a siem solution? here
are 7 things it likely needs.Gartner. Accessed:
2025-06-23.
Nabeel Seedat and Mihaela van der Schaar. 2024.
Matchmaker: Self-improving large language model
programs for schema matching.arXiv preprint
arXiv:2410.24105.Eitam Sheetrit, Menachem Brief, Moshik Mishaeli,
and Oren Elisha. 2024. Rematch: Retrieval en-
hanced schema matching with llms.arXiv preprint
arXiv:2403.01567.
Noah Shinn, Federico Cassano, Beck Labash, Ash-
win Gopinath, Karthik Narasimhan, and Shunyu
Yao. 2023. Reflexion: Language agents with ver-
bal reinforcement learning, 2023.URL https://arxiv.
org/abs/2303.11366, 1.
Xuezhi Wang, Jason Wei, Dale Schuurmans, Quoc Le,
Ed Chi, Sharan Narang, Aakanksha Chowdhery, and
Denny Zhou. 2022. Self-consistency improves chain
of thought reasoning in language models.arXiv
preprint arXiv:2203.11171.
Aoxiao Zhong, Dengyao Mo, Guiyang Liu, Jinbu Liu,
Qingda Lu, Qi Zhou, Jiesheng Wu, Quanzheng Li,
and Qingsong Wen. 2024. Logparser-llm: Advancing
efficient log parsing with large language models. In
Proceedings of the 30th ACM SIGKDD Conference
on Knowledge Discovery and Data Mining, pages
4559–4570.
Yuxin Zuo, Kaiyan Zhang, Li Sheng, Shang Qu, Ganqu
Cui, Xuekai Zhu, Haozhan Li, Yuchen Zhang, Xin-
wei Long, Ermo Hua, and 1 others. 2025. Ttrl:
Test-time reinforcement learning.arXiv preprint
arXiv:2504.16084.

A Case Study: Direction-Sensitive Port
Mapping (Iteration 49)
To illustrate the practical difficulty of schema map-
ping, we examine a representative example from
iteration 49 involving the common schema field
dpt(destination port). The field dptis defined in
the Trend Micro Common Schema as “the service
destination port of the private application server
(dstport).” In Microsoft Defender for Endpoint,
however, two candidate fields exist: LocalPort
(TCP port on the local device used during commu-
nication) and RemotePort (TCP port on the remote
device being connected to).
Mapping ambiguity.Both candidates are seman-
tically plausible depending on the traffic direction.
Foroutboundconnections, the local device is the
source, so the destination port resides on the remote
endpoint—corresponding to RemotePort . Con-
versely, forinboundconnections (e.g., an RDP
session initiated by a remote host), the local device
becomes the destination, and thus the correct map-
ping is LocalPort . Without an explicit indicator of
connection direction, an LLM can easily misassign
the field.
Why fine-tuning does not help.Simply fine-
tuning model weights cannot reliably resolve this
ambiguity. The correct mapping is not a mem-
orized association ( dpt→RemotePort ) but a
conditional rule that depends on runtime con-
text—information that is absent from the training
corpus. Updating parameters may reinforce spu-
rious correlations rather than teach the model to
reason over directionality, resulting in brittle behav-
ior across different network scenarios.
Evidence-based reasoning.During iteration 49,
the agent retrieved definitional evidence clarifying
that:
•LocalPort refers to the port on the local de-
vice.
•RemotePort refers to the port on the remote
device being connected to.
•dptrepresents the service destination port.
Although this evidence did not directly yield the
final mapping, it revealed that the correct cor-
respondence varies by communication context
and motivated the agent to infer direction from
auxiliary fields such as RemoteIP ,LocalIP , andActionType . The agent’s confidence increased
from 0.67 to 1.0, reflecting a clearer and more con-
sistent conceptual understanding.
Categorization of helpful evidence.Accord-
ing to the taxonomy defined in the main paper,
this instance exemplifiesCategory (4): revealing
context-dependent mappings. The evidence was
helpful because it exposed the conditional nature
of the mapping—showing that correctness depends
on dynamic context rather than static field align-
ment—and thereby guided subsequent reasoning
toward more generalizable, context-aware mapping
rules.
Derived practical rule.
dpt=RemotePort,if Direction = Outbound;
LocalPort,if Direction = Inbound;
(defer/flag),if Direction is unknown.
This case highlights that progress in schema align-
ment arises not from parameter updates but from
the integration of structured evidence and contex-
tual inference.
B Overconfidence Mitigation
Experiments
To address the overconfidence phenomenon ob-
served in our main experiments (Figure 3), we
conducted an additional study to examine how in-
creasing the number of inference attempts affects
confidence calibration. Our hypothesis was that
a larger number of inference samples would im-
prove statistical robustness and reduce systematic
overconfidence.
B.1 Experimental Setup
We extended the original setup by increasing the
number of inference attempts per iteration from 3
to 10. This change allows for more reliable con-
fidence estimation through better sampling of the
model’s output distribution. All other experimental
parameters remained identical to the main study:
the same source and target schemas (Microsoft De-
fender for Endpoint with 195 fields and the Com-
mon Schema with 137 fields), the same 66 manu-
ally verified ground-truth mappings, and the same
GPT-4o model configuration.
The confidence score computation followed the
same principle of prediction consistency but with

a larger sample size (10 vs. 3). This design pro-
vides a more statistically grounded measure of un-
certainty and is expected to yield better-calibrated
confidence estimates.
B.2 Results and Analysis
Table 1 presents a comparison between the 3- and
10-inference settings.
Increasing the number of inference attempts
from 3 to 10 effectively narrows the gap between
confidence and accuracy. With 10 inferences, the
mean confidence (89.3%) aligns much more closely
with the observed accuracy (92.1%), whereas with
3 inferences, confidence (95.2%) noticeably ex-
ceeds accuracy (93.94%), indicating mild overcon-
fidence.
This calibration improvement comes with a mod-
est cost: while the confidence estimates become
more realistic, overall accuracy decreases slightly
(93.94%→92.1%). Nevertheless, the more conser-
vative confidence levels offer better guidance for
expert review prioritization. The reduction in low-
confidence fields remains substantial under both
settings (85% and 80% reductions), demonstrat-
ing that the key benefit—reducing expert review
effort—is preserved.
B.3 Implications and Trade-offs
These findings highlight a clear balance between
calibration quality and computational cost. Increas-
ing the number of inference attempts leads to better-
calibrated confidence scores but requires more com-
putation. In practical terms:
•For high-precision scenarios, increasing at-
tempts (e.g., to 10) provides more reliable
uncertainty estimates and tighter alignment
between confidence and accuracy.
•For cost-sensitive deployments, even three
attempts already achieve strong accuracy and
substantial reduction in expert workload, mak-
ing it a highly efficient operational setting.
These results demonstrate improved calibra-
tion—confidence more accurately tracks empir-
ical accuracy as the number of inferences in-
creases—without the need for additional metric
reporting.
C Prompt Architecture
Table 2 summarizes the three prompts used in our
schema mapping system. Each serves a distinctfunction in guiding the agent’s reasoning process
across sessions and mapping iterations.
System Prompt Example.The system prompt de-
fines the AI agent’s expertise and reasoning frame-
work. It is shown below for reference.
You are a Trend Micro cybersecurity data expert spe-
cializing in Trend Micro’s Common Schema across
multiple layers, including endpoint, network, mes-
saging, and cloud. You have extensive expertise in
processing and mapping third-party product and log
schemas to Trend Micro’s Common Data Schema,
enabling cross-log correlation, advanced threat detec-
tion, and compliance reporting.
You routinely perform professional schema mapping
for new third-party log sources with a focus on accu-
racy. Your approach follows a layered reasoning pro-
cess: (1) identify core entities such as IP addresses,
filenames, and hashes (e.g., SHA1); (2) narrow down
candidate fields based on data flow direction and con-
text; (3) make precise mapping decisions supported
by semantic consistency.
For example: – src_ip vs. dst_ip de-
pends on whether traffic is inbound or
outbound. – SHA1 hashes may represent
parent_process_sha1 ,launched_process_sha1 ,
ordropped_object_sha1.
If no suitable mapping exists, respond professionally
withNOT_COVERED.

Setting Final Accuracy Mean Confidence Low-Confidence Fields
3 Inferences 93.94% 95.2% 26→4 (-85%)
10 Inferences 92.1% 89.3% 31→6 (-80%)
Table 1: Comparison of overconfidence mitigation between 3 and 10 inferences per iteration. The 10-inference
setting improves calibration by bringing confidence values more closely aligned with actual accuracy.
# Prompt Name Purpose Usage
1 System Prompt Defines the AI agent’s role as a cy-
bersecurity data expert specializing in
schema mapping. Establishes reasoning
rules and guidelines.Loaded once per ses-
sion
2 User Prompt Carries the per-request payload: (a)
curated facts from prior conflicts (if
any), (b) RAG-assembled source/target
schema context (descriptions, sample
values, types), and (c) the mapping
task/question. Enforces a strict XML re-
sponse (CSV decision, 1–5 confidence,
reasoning) for deterministic parsing.For every mapping
request
3 Search Prompt Generates targeted Internet search
queries string to resolve ambiguous
field mappings using prior conflict in-
formation.Invoked only during
conflict resolution
Table 2: Overview of the three prompts used in the schema mapping system, detailing their roles and usage.