# AgentSim: A Platform for Verifiable Agent-Trace Simulation

**Authors**: Saber Zerhoudi, Michael Granitzer, Jelena Mitrovic

**Published**: 2026-04-29 13:19:38

**PDF URL**: [https://arxiv.org/pdf/2604.26653v1](https://arxiv.org/pdf/2604.26653v1)

## Abstract
Training trustworthy agentic LLMs requires data that shows the grounded reasoning process, not just the final answer. Existing datasets fall short: question-answering data is outcome-only, chain-of-thought data is not tied to specific documents, and web-agent datasets track interface actions rather than the core retrieval and synthesis steps of a RAG workflow. We introduce AgentSim, an open-source platform for simulating RAG agents. It generates verifiable, stepwise traces of agent reasoning over any document collection. AgentSim uses a policy to ensure the agent widely explores the document set. It combines a multi-model validation pipeline with an active human-in-the-loop process. This approach focuses human effort on difficult steps where models disagree. Using AgentSim, we construct and release the Agent-Trace Corpus (ATC), a large collection of grounded reasoning trajectories spanning three established IR benchmarks. We make three contributions: (1) the AgentSim platform with two mechanisms, Corpus-Aware Seeding and Active Validation, that improve trace diversity and quality; (2) the Agent-Trace Corpus (ATC), over 103,000 verifiable reasoning steps spanning three IR benchmarks, with 100% grounding rate on substantive answers; and (3) a comparative behavioral analysis revealing systematic differences in how state-of-the-art models approach information seeking. Platform, toolkit, and corpus are publicly available.

## Full Text


<!-- PDF content starts -->

AgentSim: A Platform for Verifiable Agent-Trace Simulation
Saber Zerhoudi
University of Passau
Passau, Germany
saber.zerhoudi@uni-passau.deMichael Granitzer
University of Passau
Passau, Germany
Interdisciplinary Transformation
University Austria
Linz, Austria
michael.granitzer@uni-passau.deJelena Mitrović
University of Passau
Passau, Germany
jelena.mitrovic@uni-passau.de
Figure 1: Overview of the AgentSim Workflow and Agent-Trace Corpus.
Abstract
Training trustworthy agentic LLMs requires data that shows the
grounded reasoning process, not just the final answer. Existing
datasets fall short: question-answering data is outcome-only, chain-
of-thought data is not tied to specific documents, and web-agent
datasets track interface actions rather than the core retrieval and
synthesis steps of a RAG workflow. We introduce AgentSim, an
open-source platform for simulating RAG agents. It generates veri-
fiable, stepwise traces of agent reasoning over any document collec-
tion. AgentSim uses a policy to ensure the agent widely explores the
document set. It combines a multi-model validation pipeline with
an active human-in-the-loop process. This approach focuses human
effort on difficult steps where models disagree. Using AgentSim,
we construct and release the Agent-Trace Corpus (ATC), a large
collection of grounded reasoning trajectories spanning three es-
tablished IR benchmarks. We make three contributions: (1) the
AgentSim platform1with two mechanisms, Corpus-Aware Seeding
and Active Validation, that improve trace diversity and quality; (2)
the Agent-Trace Corpus (ATC)2, over 103,000 verifiable reasoning
1Platform: https://agentsim.searchsim.org
2
ATC Dataset: https://huggingface.co/datasets/searchsim/agentsim-atcsteps spanning three IR benchmarks, with 100% grounding rate
on substantive answers; and (3) a comparative behavioral analy-
sis revealing systematic differences in how state-of-the-art models
approach information seeking3. Platform, toolkit, and corpus are
publicly available.
CCS Concepts
•Computing methodologies →Simulation environments;
Modeling and simulation;•Human-centered computing →
Interactive systems and tools.
Keywords
Agentic AI, Simulation platform, Retrieval-augmented generation
(RAG), Data generation
1 Introduction
Large language models are evolving into autonomous agents capa-
ble of multi-step reasoning and tool use. This shift holds promise
3Repository: https://github.com/searchsim-org/sigir26-agentsim
©ACM, 2026. This is the author’s version of the work.
The definitive version was published in:Proceedings of the 49th
International ACM SIGIR Conference on Research and Development in
Information Retrieval (SIGIR ’26), July 20–24, 2026, Melbourne, VIC, Australia.
DOI: https://doi.org/10.1145/3805712.3808577arXiv:2604.26653v1  [cs.IR]  29 Apr 2026

Saber Zerhoudi, Michael Granitzer, and Jelena Mitrović
for information retrieval, yet it introduces a challenge: the internal
reasoning of these agents remains hidden. When an agent fails
to find relevant information or produces an incorrect synthesis,
practitioners cannot seewhy. This opacity makes agentic systems
hard to trust, debug, and improve.
A path toward more auditable agents lies in training them on
data that captures the full reasoningprocess, not just the finalout-
come. This is especially true for Retrieval-Augmented Generation
(RAG) [ 12], where retrieval-driven user-centric agents are emerg-
ing [ 21]. We identify what we call thecognitive RAG gap[ 22]:
existing datasets do not capture the intermediate steps (query for-
mulation, document evaluation, information synthesis, and strategy
adaptation) that make up effective information seeking.
We can characterize this gap by examining three cate-
gories of existing resources.Question-answering datasetssuch as
MS MARCO [ 3] and Natural Questions [ 11] provide query-answer
pairs but omit the retrieval process entirely; they are outcome-only.
Reasoning-trace datasetslike OpenThoughts [ 10] and PRM800K [ 13]
capture step-by-step reasoning for abstract tasks such as mathemat-
ics and coding, yet these traces are not grounded in external, verifi-
able documents.Web-agent benchmarksincluding WebArena [ 23]
and Mind2Web [ 9] focus on interface navigation rather than the
cognitive workflow of retrieval and synthesis.
This paper introducesAgentSim, an open-source platform de-
signed to bridge this gap (Figure 1). AgentSim enables the design,
simulation, and recording of grounded agent traces over document
collections. The system comprises a visual web interface for pro-
totyping and a command-line toolkit for scalable data generation.
Two mechanisms differentiate AgentSim from prior tools. First,
Corpus-Aware Seedingemploys embedding-based clustering and
novelty filtering to ensure simulations explore the full breadth of a
target corpus, avoiding the redundancy of random sampling. Sec-
ond, anActive Validationloop employs multi-model disagreement
to automatically flag ambiguous steps for human review, focusing
annotation effort where it matters most.
Using AgentSim, we construct and publicly release theAgent-
Trace Corpus (ATC)2, a collection of over 103,000 verifiable rea-
soning steps spanning the MS MARCO, Quasar-T, and CausalQA [ 6]
corpora. We use this corpus to conduct a comparative analysis of
three state-of-the-art models, identifying variations in exploration
strategy, query reformulation behavior, and synthesis under uncer-
tainty. The results support the utility of the platform as a research
tool and a training source for RAG agents.
2 Related Work
Our work sits at the intersection of reasoning-trace datasets, RAG
evaluation, agent simulation platforms, and synthetic data genera-
tion. We position AgentSim relative to each of these areas.
Reasoning-Trace Datasets.Recent research captures explicit
reasoning processes to improve model training. Datasets such as
OpenThoughts [ 10] and PRM800K [ 13] provide step-by-step traces
for mathematics and code generation, training models to “show
their work” [ 1,18]. However, such traces are self-contained: they
capture internal reasoning but are not tied to external, verifiable
knowledge sources. Our work differs in that every reasoning step inAgentSim links to concrete retrieval actions and specific documents,
allowing verification against an external corpus.
RAG Evaluation Benchmarks.The evaluation of retrieval-
augmented systems relies on established collections such as MS-
MARCO [ 3], Natural Questions [ 11], and domain-specific resources
like CausalQA [ 6]. These benchmarks provide query-answer pairs
for measuring system accuracy [ 8,20]. Our platform does not re-
place these resources but rather builds upon them: we use their
queries as starting points and their document collections as the
environment for simulation. The contribution is transforming
static query-answer pairs into dynamic traces that reveal the full
information-seeking process, including failed retrievals, query re-
formulations, and iterative refinement.
Agent Simulation Platforms.Interactive benchmarks such
as WebArena [ 23], Mind2Web [ 9], and ALFWorld [ 15] evaluate
agents through interface-level interactions (clicking, navigating,
form-filling). A parallel IR line produces user-simulation toolkits
for interaction logs and behavioural data [ 2,4], but targets session-
level user behaviour, not the cognitive workflow of an agentic
retriever. AgentSim instead simulates and records thecognitive
processbehind RAG tasks—how agents formulate queries, evaluate
documents, synthesize information, and adapt strategies—a focus
that distinguishes it from both world- and user-simulation tools.
Synthetic Data Quality.Using LLM-generated data for training
raises concerns about quality and diversity [ 16]. Common mitiga-
tions include majority voting [ 17] and consistency filtering. Our
Active Validation loop goes beyond passive voting by introduc-
ing targeted human review: rather than applying uniform quality
checks, we use model disagreement (measured via a Divergence
Score) to identify steps where automated validation falls short. This
active approach focuses human annotation on ambiguous cases, im-
proving cost-effectiveness while maintaining data quality [14, 19].
3 System Architecture and Components
AgentSim facilitates flexible and reproducible data generation. The
platform combines a visual front-end for workflow design with
a command-line toolkit for scalable trace creation. To improve
synthetic data quality and coverage, the system employs two mech-
anisms: Corpus-Aware Seeding and Active Validation.
The platform’s design follows recent findings that high-quality
reasoning traces are data-efficient for training capable models [ 14].
Rather than maximizing trace volume, AgentSim prioritizes trace
diversity and verifiability. The modular architecture lets researchers
combine tools for retrieval, generation, and verification, supporting
both behavioral analysis and systematic data creation.
3.1 Platform and Toolkit
ThePlatform1provides a web-based visual interface for designing
and testing agent workflows. Users construct pipelines by connect-
ing modular components (search, rerank, summarize, synthesize)
and can step through simulations interactively to prototype and
debug agent behavior. A demonstration walkthrough is available in
the repository documentation, guiding users through creating RAG
workflows, configuring models, and inspecting generated traces.
TheToolkit3is a command-line interface for large-scale data
generation. It executes simulations from YAML configuration files

AgentSim: A Platform for Verifiable Agent-Trace Simulation
Table 1: Hyperparameters for Corpus-Aware Seeding.
Parameter Value Description
𝐾(clusters) 50 Number of topic clusters
𝜏(novelty) 0.4 Min. fraction of new documents
𝜆(MMR) 0.7 Diversity–relevance trade-off
Embedding MiniLMall-MiniLM-L6-v2model
specifying the agent’s tools, models, corpus connection, and seed-
ing parameters. Core commands include agentsim simulate for
trace generation, agentsim seed-select for corpus-aware seed cre-
ation, and agentsim validate for configuration verification. These
components form a unified system for repeatable data generation.
3.2 Corpus-Aware Seeding Policy
Random sampling of simulation starting points is inefficient: it
tends to favor popular topics and produces redundant traces that
waste computational resources. Our platform implements aCorpus-
Aware Seedingpolicy to ensure systematic coverage of the target
document collection while maximizing trace novelty.
Algorithm 1 formalises the policy: given candidate queries 𝑄,
corpus𝐶, and budget 𝐵, it produces a seed set 𝑆that balances top-
ical coverage with document novelty by jointly optimising two
dimensions—query semantic diversity (different topics) and re-
trieval novelty (different regions of the document space). The en-
coder choice matters: retrieval behaviour varies substantially across
encoder families on identical corpora [ 7], so we expose it as a hy-
perparameter (Table 1) and report results under all-MiniLM-L6-v2 .
Algorithm 1Corpus-Aware Seed Selection
Require:Candidate queries𝑄, corpus𝐶, budget𝐵
Ensure:Diverse seed set𝑆
1:𝐸←Embed(𝑄)using SentenceTransformer
2:{𝐶 1,...,𝐶𝐾}←KMeans(𝐸,𝐾)
3:coverage[𝑐]←0for all clusters𝑐
4:𝑆←∅; seen_docs←∅
5:while|𝑆|<𝐵do
6:𝑐∗←arg min 𝑐coverage[𝑐]
7:𝑄𝑐←{𝑞∈𝑄:cluster(𝑞)=𝑐∗}
8:for𝑞∈𝑄 𝑐do
9:𝐷𝑞←Retrieve(𝑞,𝐶)
10:novelty(𝑞)←|𝐷 𝑞\seen_docs|/|𝐷 𝑞|
11:end for
12:Filter𝑄 𝑐to queries with novelty(𝑞)>𝜏
13:𝑞∗←MMR(𝑄 𝑐,𝑆,𝜆)
14:𝑆←𝑆∪{𝑞∗}
15:seen_docs←seen_docs∪𝐷 𝑞∗
16:coverage[𝑐∗]←coverage[𝑐∗]+1
17:end while
18:return𝑆
Table 1 lists the key hyperparameters. The policy outputs a
seeds.jsonl file that enables transparent coverage measurement
and supports controlled experiments where different agent config-
urations are compared over identical starting conditions.Table 2: Agent-Trace Corpus (ATC) statistics.
Statistic Value
Total trace steps 103,567
Supervised training pairs 20,548
Unique documents retrieved 199,968
Total queries generated 26,176
Source corpora MS MARCO, Quasar-T, CausalQA
Analyst modelsgpt-4o,mistral-large, anddeepseek-v3
3.3 Active Validation and Refinement Loop
Generating high-quality traces at scale requires balancing automa-
tion with human oversight. Our system implements anActive
Validationloop that uses model disagreement to focus human
review precisely where it is most needed.
The mechanism operates through an Analyst–Critic–Judge
pipeline. For each simulation step, an Analyst model proposes
an action (e.g., a query reformulation or synthesis). Two Critic
models independently evaluate and potentially revise this pro-
posal. A Judge component then computes aDivergence Score
DS(𝑠)= 1−max𝑎∈𝐴|{𝑚:𝑚chose𝑎}|/|𝑀| , where𝐴is the set of
distinct actions proposed and 𝑀is the set of participating models. A
step is flagged for human review when DS(𝑠)>𝜃 (we use𝜃=0.4).
For each flagged step, reviewers see the competing model outputs
and choose toPromotethe best one,Reviseit, orDiscardthe trajec-
tory, following a rubric on query–document relevance, evidence
sufficiency, and synthesis faithfulness. Inter-reviewer agreement is
tracked via 10% double-annotation, with disagreements adjudicated.
The loop has two independent triggers. TheDivergence Score
routes ambiguous steps to human review. A separate, automated
route uses a per-stepgrounding-confidence scorethat measures how
well the proposed answer is supported by its cited evidence: when
this score is low ( <0.3), the step is sent for automatic re-retrieval.
This split keeps human attention on cases where models genuinely
disagree, while still repairing low-confidence steps at scale [ 14,
19]. Across 5,320 ATC steps, 73.4% had low grounding confidence
and triggered automatic re-retrieval; 99.8% of those subsequently
received external verification; and 79.3% of seed queries showed
improved grounding after iterative validation.
4 The Agent-Trace Corpus (ATC)
Using AgentSim, we construct and release theAgent-Trace Corpus,
a large collection of grounded reasoning traces designed for train-
ing and analyzing RAG agents. This section describes the corpus
statistics, data formats, and intended use cases.
4.1 Corpus Statistics
The ATC comprises traces generated across three established IR
benchmarks, using three state-of-the-art language models. Table 2
summarizes the key statistics. The corpus contains 103,567 reason-
ing trace steps, each connecting an agent’s internal deliberation to
concrete retrieval actions and specific documents. From these traces,
we extract 20,548 supervised training pairs suitable for fine-tuning,
along with nearly 200,000 unique retrieved documents.

Saber Zerhoudi, Michael Granitzer, and Jelena Mitrović
The three source corpora provide complementary coverage:
MS MARCO contributes web passage retrieval tasks, Quasar-T
provides factual question answering, and CausalQA offers com-
plex causal reasoning challenges. This diversity ensures the corpus
exercises different aspects of agent reasoning.
4.2 Data Formats
The ATC is distributed in three formats, serving distinct purposes:
Traces( traces/*.jsonl.gz ) contain the full reasoning sequence
for each simulation: thought →action→observation chains with
complete LLM inputs and outputs. Each trace records the agent’s
internal deliberation (“I need to find information about X”), the
action taken (“search for Y”), and the retrieved results. These are
suitable for behavioral analysis and process-level supervision.
Trajectories( trajectories/*.jsonl.gz ) provide high-level action
sequences abstracted from the raw LLM outputs. Each trajectory
lists the sequence of tools called (search, rerank, summarize) along
with their inputs and outputs, without the full prompt text. These
compact representations are efficient for training action-prediction
models and studying exploration patterns.
Supervised pairs( supervised/*.jsonl.gz ) extract query–
document–answer triples with multi-hop reasoning chains. Each
record contains a question, the documents retrieved to answer it,
and the final synthesized answer. These are formatted for direct
use in supervised fine-tuning pipelines.
4.3 Intended Use Cases
The corpus supports multiple training paradigms. Forchain-of-
thought fine-tuning, the traces provide explicit reasoning grounded
in retrieved documents. Inimitation learning, the trajectories
demonstrate effective information-seeking behavior. Forquery re-
formulation training, the 26,176 generated queries, including both
successful and unsuccessful reformulations, facilitate the learning
of adaptive search strategies. The structured traces also support
student-teacher distillation, where smaller models learn to approxi-
mate the reasoning patterns of larger analysts.
A distinguishing feature of ATC isverifiability: every reasoning
step traces to specific documents in the source corpus, enabling
researchers to audit agent behavior and identify failure modes. We
verify this on 5,320 trace entries. Of 1,385 substantive answers (after
excluding 3,907 correct refusals where models identified insufficient
evidence), 100% are grounded in their cited documents, with a mean
token coverage of 0.872, wheretoken coverageis the fraction of con-
tent tokens (excluding stopwords) in the synthesized answer that
also appear within the cited evidence spans. The combined quality
rate, grounded answers plus correct refusals, reaches 99.5%. A finer-
grained NLI-based analysis confirms 83.3% of 6,125 atomic claims
are fully supported by cited evidence. To our knowledge, no existing
reasoning-trace dataset reports token-level grounding verification
against source documents; prior work such as PRM800K [ 13] veri-
fies step correctness but not document grounding.
To validate downstream utility, we conducted student-teacher
distillation using GPT-4o-generated ATC traces as the teacher signal.
We fine-tuned Qwen 2.5 (0.5B to 7B) and Gemma-3 (270M) on ATC-
derived supervised pairs using LoRA across 12 configurations. The
smallest Qwen model (0.5B) improved its abstain-detection F1 from0.362 to 0.815, matching the performance of a 3 ×larger base model.
Across all sizes and architectures, trained models learn to abstain
when evidence is insufficient (recall >0.88), demonstrating that the
corpus captures transferable reasoning behavior.
5 Evaluation and Results
We address two research questions: (1) Does Corpus-Aware Seeding
improve coverage compared to alternatives? (2) Can the platform
reveal behavioral patterns across state-of-the-art models? We com-
pare our seeding approach against three baselines, then analyze
behavior across exploration, reformulation, and synthesis.
5.1 Experimental Setup
We conducted a controlled experiment using 1,000 seed queries
each from three corpora: MS MARCO (web passages about gen-
eral knowledge), Quasar-T (factual questions from trivia sources),
and CausalQA (questions requiring causal reasoning). Seeds were
selected via our Corpus-Aware Seeding policy (Section 3.2). We
executed three simulation runs per dataset (9 total), holding
the retrieval backend (ChatNoir [ 5] with MS MARCO v2.1 and
ClueWeb12-22) and workflow configuration constant.
The single independent variable was the Analyst model: we
rotated gpt-4o ,mistral-large , and deepseek-v3 through this role,
with the remaining two serving as Critics in each run. For each seed
query, the agent performed up to 7 retrieval-reasoning cycles of
query formulation, document retrieval, relevance assessment, and
optional reformulation. Each of the 3,000 seed queries produced
7 explorations via the Active Validation loop, generating a corpus
of 21,000 trajectories. This design isolates behavioral differences
tied to the Analyst model. Generating the full ATC consumed ap-
proximately 20M API tokens across OpenAI, Mistral, and DeepSeek
endpoints, at an estimated total cost of $200–300 USD. Individual
simulations average 58 steps and $0.14 in API costs, making corpus
extension straightforward; no local GPU resources were required.
5.2 Seeding Policy Evaluation
We compare Corpus-Aware Seeding against three alternatives: Ran-
dom (uniform sampling), Stratified (K-Means clustering with round-
robin selection), and DPP (determinantal point process maximizing
embedding-space diversity). Each method was evaluated across
three datasets at budget 𝐵=500, with five independent runs per
configuration (60 total). Table 3 reports four metrics:cluster cover-
age(fraction of 𝐾topic clusters whose centroid is closest to at least
one selected seed),document redundancy(mean Jaccard overlap of
top-100 retrievals between selected seed pairs),semantic diversity
(mean pairwise cosine distance among seed embeddings), andcor-
pus coverage@100( |Ð
𝑞∈𝑆Top-100(𝑞)|/|𝐶| , the fraction of corpus
documents reached by the top-100 retrievals of any selected seed).
Corpus-Aware achieves perfect cluster coverage (1.000, zero vari-
ance) on every dataset, while Random covers only 95–97% and
DPP drops to 45% on CausalQA at lower budgets. All pairwise
comparisons involving Corpus-Aware are statistically significant
after correcting for multiple comparisons (Holm-Bonferroni), with
medium-to-large effect sizes (Cohen’s 𝑑= 0.61–1.60). DPP achieves
the highest semantic diversity, particularly on CausalQA (0.948 vs.
0.835 for Corpus-Aware at 𝐵=100), but this advantage does not

AgentSim: A Platform for Verifiable Agent-Trace Simulation
Figure 2: Exploration breadth and retrieval redundancy across three datasets (1,000 seed queries each). Bar height indicates
total retrieved documents; solid and faded regions represent unique and redundant retrievals, respectively. Percentages denote
Retrieval Redundancy, the fraction of retrievals revisiting previously seen documents.
transfer to cluster or corpus coverage, confirming that optimizing
for embedding-space diversity alone is insufficient for corpus con-
struction. Stratified sampling offers minimal benefit over Random
on redundancy and coverage metrics ( 𝑝>0.14, not significant), in-
dicating that cluster-based allocation without novelty filtering and
MMR is not enough. When matched against existing ATC traces,
Corpus-Aware seeds achieve 68–98% trace coverage compared to
2–13% for baselines, confirming that the policy selects seeds that
produce valid, non-degenerate simulations.
5.3 Comparative Exploration Behavior
We measured exploration behavior across all three datasets (Fig-
ure 2), assessing two metrics:Exploration Breadth(total unique
documents retrieved) andRetrieval Redundancy(ratio of repeated
to total retrievals).
The gpt-4o Analyst demonstrated the widest exploration breadth,
retrieving 31.7% more unique documents than deepseek-v3 on
MS MARCO (74,700 vs. 56,700), 20.0% more on Quasar-T, and 176.3%
more on CausalQA (162,250 vs. 58,730). Six of nine pairwise com-
parisons (Mann-Whitney 𝑈, corrected for multiple comparisons)
are statistically significant, with effect sizes up to Cohen’s 𝑑=1.59
(gpt-4o vs. deepseek-v3 on CausalQA). This gap is partly explained
by differences in validation pipeline engagement: deepseek-v3 trig-
gers adaptive mode in only 0.1% of steps, bypassing the multi-model
consultation that generates alternative queries for gpt-4o (65.1%)
andmistral-large(64.0%).
RegardingRetrieval Redundancy, mistral-large was the high-
est (14.2% on MS MARCO, 18.1% on Quasar-T, 10.4% on CausalQA),
implying a policy that favors re-validating documents. In con-
trast, gpt-4o maintained the lowest redundancy (11.8%, 11.6%, and
9.8% respectively), which aligns with its aggressive exploration.Table 3: Seeding strategy comparison ( 𝐵=500, 5 runs). Cov. =
cluster coverage, Red. = document redundancy, Div. = seman-
tic diversity, CCov. = corpus coverage@100. All comparisons
involving Corpus-Aware are significant ( 𝑝< 0.001, Holm-
Bonferroni corrected; Cohen’s𝑑= 0.61–1.60).
Method Cov.↑Red.↓Div.↑CCov.↑MS MARCORandom .986 .002 .982 .048
Stratified .994 .002 .981 .048
DPP .992 .001.987.047
Corpus-Aware1.00 .000 .987 .050Quasar-TRandom .968 .007 .971 .026
Stratified .992 .006 .972 .026
DPP .994 .004.987.030
Corpus-Aware1.00 .000.984.050CausalQARandom .968 .095 .785 .044
Stratified .992 .096 .802 .044
DPP .936 .008.859.049
Corpus-Aware1.00 .003.821.050
deepseek-v3 had the lowest redundancy on CausalQA (8.8%), though
this was accompanied by its reduced exploration breadth.
5.4 Analysis of Model Reasoning Differences
We analyzed 1,254 follow-up queries generated by gpt-4o ,
mistral-large , and deepseek-v3 . These reformulated queries are
important as they reveal each model’s innate strategy for informa-
tion seeking when deciding what information to seek next.
5.4.1 Query Reformulation Strategies.We grouped each query re-
formulation into three types:conceptual(adding new related terms),

Saber Zerhoudi, Michael Granitzer, and Jelena Mitrović
Table 4: Distribution of query reformulation strategies.
Model Conceptual Procedural Syntactic
mistral-large93.2% 6.3%0.5%
gpt-4o87.1% 4.8% 8.1%
deepseek-v372.4% 4.3%23.3%
procedural(notes about the search process), andsyntactic(taking
keywords or making the query simpler). Table 4 shows how each
model behaves. The distribution of reformulation strategies differs
significantly across models on all three datasets (chi-squared test,
𝑝<0.001; Cramér’s𝑉= 0.054–0.073).
While all models favor conceptual reformulation, deepseek-v3
shows a high rate of syntactic simplification (23.3%), often removing
natural language structure to create keyword phrases. For example,
given the query “what was the immediate impact of the success
of the manhattan project?”, gpt-4o reformulates to “how did the
manhattan project influence world politics” (conceptual expansion),
while deepseek-v3 reformulates to “Manhattan Project key scien-
tists” (syntactic reduction to keywords).
5.4.2 Impact on Exploration.This strategic difference is reflected
in query length. mistral-large and gpt-4o expand average query
length by 27% (from 6.8 to 8.2 words) and 11% (from 5.8 to 6.4 words)
respectively, while deepseek-v3 is the only model to consistently
shorten queries, from 5.2 to 3.9 words (−16%).
This syntactic simplification may contribute to reduced semantic
exploration. Shorter, keyword-based queries may match narrower
lexical spaces in the retrieval index. This aligns with our finding that
deepseek-v3 retrieved 64% fewer unique documents than gpt-4o
(58,730 vs 162,250) on the CausalQA dataset, despite generating
the same number of queries. The model’s reformulation strategy
directly impacts the diversity of the document sets it explores.
5.4.3 Synthesis Under Uncertainty.Models show divergent strate-
gies when tasked with synthesizing answers from conflicting evi-
dence. Our analysis of 87 contradictory retrieval events across all
runs reveals two distinct approaches. Models such as gpt-4o and
mistral-large employ aMajority-Driven Synthesis. They typically
identify the most common claim and present it as the decisive an-
swer. Thisconfidence-firstapproach prioritizes a clear response
but obscures the underlying conflict. In contrast, deepseek-v3 uses
anUncertainty-Aware Synthesis, frequently acknowledging contra-
dictions explicitly in its reasoning. Thiscaution-firststance aligns
with its tendency to retrieve more documents, suggesting a more
conservative policy. These are not isolated behaviors but reflect
systematic differences in how models are trained to balance deci-
siveness against caution when faced with uncertain information.
6 Discussion
We examine the strengths and limitations of the AgentSim ap-
proach, and position it relative to other methods for generating
agent training data.
Strengths.The platform offers several advantages for RAG agent
research. First,grounded verificationsets AgentSim apart from pure
reasoning-trace datasets: every step can be checked against specificdocuments in the source corpus, allowing systematic auditing of
agent behavior. Second, the Active Validation loop providescost-
effective quality controlby focusing human annotation on ambigu-
ous cases rather than applying uniform review. The Divergence
Score effectively identifies steps needing human judgment: 79.3%
of flagged seeds showed improved grounding after iterative review.
Third, the modular architecture and configuration-driven execu-
tion supportreproducible research: experiments can be replicated
by sharing configuration files and seed sets.
Limitations.Several constraints should be noted. The approach
inherits biases from its source corpora; traces generated over
MS MARCO will reflect the characteristics of web passage retrieval,
which may not transfer to specialized domains. The behavioral
findings we report are specific to the models evaluated; as founda-
tion models evolve, these patterns may change. While the Active
Validation loop reduces annotation burden, standardizing human
review guidelines for agent reasoning remains an open challenge.
Positioning.AgentSim complements, rather than competes
with, existing resources. World-simulation platforms like We-
bArena [ 23] capture interface-level interactions; AgentSim provides
cognitive-level traces. Static benchmarks provide ground-truth la-
bels; AgentSim transforms these into process-level supervision
tailored to specific corpora, tasks, and model configurations.
7 Conclusion
This paper introduces AgentSim, an open-source platform for gen-
erating verifiable reasoning traces for retrieval-augmented genera-
tion (RAG) agents. The system addresses the lack of training data
for intermediate information seeking through two mechanisms:
Corpus-Aware Seeding ensures systematic document exploration,
while Active Validation directs human review to ambiguous in-
stances identified by multi-model disagreement.
We release three artifacts to the research community: the
AgentSim platform with its visual workflow designer, a command-
line toolkit for scalable trace generation, and the Agent-Trace Cor-
pus with over 103,000 grounded reasoning steps across three estab-
lished benchmarks3. Our evaluation shows the platform’s utility
for behavioral analysis, revealing systematic differences in how
state-of-the-art models approach exploration, query reformulation,
and synthesis under uncertainty.
Several directions merit future work. Our distillation experi-
ments show that ATC traces improve sample efficiency across model
families; scaling to pre-training is a natural next step. Domain-
specific deployments for legal or scientific literature review need
careful corpus curation and may reveal new challenges in agent be-
havior. Extending the Judge module with access to hidden solutions
would let us train agents to reconstruct reasoning processes without
seeing final answers. Researchers can also use ATC to benchmark
their own models against our three-analyst baseline, train process
reward models on per-step divergence scores, or generate domain-
specific traces with the toolkit over their own corpora.
The platform, corpus, and all evaluation code are released un-
der the MIT license3; the underlying datasets retain their original
licenses. We invite the community to use AgentSim for analyzing
agent behavior and generating training data, and to contribute
traces back to the growing corpus.

AgentSim: A Platform for Verifiable Agent-Trace Simulation
References
[1] Marah Abdin, Sahaj Agarwal, Ahmed Awadallah, Vidhisha Balachandran, Harki-
rat Behl, Lingjiao Chen, Gustavo de Rosa, Suriya Gunasekar, Mojan Javaheripi,
Neel Joshi, et al. 2025. Phi-4-reasoning technical report.
[2]Leif Azzopardi, Timo Breuer, Björn Engelmann, Christin Kreutz, Sean MacA-
vaney, David Maxwell, Andrew Parry, Adam Roegiest, Xi Wang, and Saber
Zerhoudi. 2024. SimIIR 3: A framework for the simulation of interactive and con-
versational information retrieval. InProceedings of the 2024 Annual International
ACM SIGIR Conference on Research and Development in Information Retrieval in
the Asia Pacific Region. 197–202.
[3] Payal Bajaj, Daniel Campos, Nick Craswell, Li Deng, Jianfeng Gao, Xiaodong Liu,
Rangan Majumder, Andrew McNamara, Bhaskar Mitra, Tri Nguyen, et al .2016.
MS MARCO: A human generated machine reading comprehension dataset.
[4] Krisztian Balog, Nolwenn Bernard, Saber Zerhoudi, and ChengXiang Zhai. 2025.
Theory and Toolkits for User Simulation in the Era of Generative AI: User
Modeling, Synthetic Data Generation, and System Evaluation. InProceedings
of the 48th International ACM SIGIR Conference on Research and Development in
Information Retrieval. 4138–4141.
[5] Janek Bevendorff, Benno Stein, Matthias Hagen, and Martin Potthast. 2018. Elas-
tic chatnoir: Search engine for the clueweb and the common crawl. InEuropean
conference on information retrieval. Springer, 820–824.
[6]Alexander Bondarenko, Magdalena Wolska, Stefan Heindorf, Lukas Blübaum,
Axel-Cyrille Ngonga Ngomo, Benno Stein, Pavel Braslavski, Matthias Hagen, and
Martin Potthast. 2022. CausalQA: A benchmark for causal question answering.
InProceedings of the 29th International Conference on Computational Linguistics.
3296–3308.
[7] Laura Caspari, Kanishka Ghosh Dastidar, Saber Zerhoudi, Jelena Mitrovic, and
Michael Granitzer. 2024. Beyond benchmarks: Evaluating embedding model simi-
larity for retrieval augmented generation systems.arXiv preprint arXiv:2407.08275
(2024).
[8] Rafael Teixeira De Lima, Shubham Gupta, Cesar Berrospi Ramis, Lokesh Mishra,
Michele Dolfi, Peter Staar, and Panagiotis Vagenas. 2025. Know your RAG:
Dataset taxonomy and generation strategies for evaluating RAG systems. 39–
57 pages.
[9]Xiang Deng, Yu Gu, Boyuan Zheng, Shijie Chen, Sam Stevens, Boshi Wang,
Huan Sun, and Yu Su. 2023. Mind2web: Towards a generalist agent for the web.
Advances in Neural Information Processing Systems36, 28091–28114.
[10] Etash Guha, Ryan Marten, Sedrick Keh, Negin Raoof, Georgios Smyrnis, Hritik
Bansal, Marianna Nezhurina, Jean Mercat, Trung Vu, Zayne Sprague, et al .2025.Openthoughts: Data recipes for reasoning models.
[11] Tom Kwiatkowski, Jennimaria Palomaki, Olivia Redfield, Michael Collins, Ankur
Parikh, Chris Alberti, Danielle Epstein, Illia Polosukhin, Jacob Devlin, Kenton
Lee, et al .2019. Natural questions: a benchmark for question answering research.
Transactions of the Association for Computational Linguistics7, 453–466.
[12] Patrick Lewis, Ethan Perez, Aleksandra Piktus, Fabio Petroni, Vladimir
Karpukhin, Naman Goyal, Heinrich Küttler, Mike Lewis, Wen-tau Yih, Tim Rock-
täschel, et al .2020. Retrieval-augmented generation for knowledge-intensive nlp
tasks.Advances in neural information processing systems33 (2020), 9459–9474.
[13] Hunter Lightman, Vineet Kosaraju, Yuri Burda, Harrison Edwards, Bowen Baker,
Teddy Lee, Jan Leike, John Schulman, Ilya Sutskever, and Karl Cobbe. 2023. Let’s
verify step by step.
[14] Pleias. 2025. SYNTH: the new data frontier. https://pleias.fr/blog/blogsynth-the-
new-data-frontier. Accessed: 2025-11-12.
[15] Mohit Shridhar, Xingdi Yuan, Marc-Alexandre Côté, Yonatan Bisk, Adam
Trischler, and Matthew Hausknecht. 2020. Alfworld: Aligning text and em-
bodied environments for interactive learning.
[16] Ilia Shumailov, Zakhar Shumaylov, Yiren Zhao, Yarin Gal, Nicolas Papernot, and
Ross Anderson. 2023. The curse of recursion: Training on generated data makes
models forget.arXiv preprint arXiv:2305.17493(2023).
[17] Xuezhi Wang, Jason Wei, Dale Schuurmans, Quoc Le, Ed Chi, Sharan Narang,
Aakanksha Chowdhery, and Denny Zhou. 2022. Self-consistency improves chain
of thought reasoning in language models.arXiv preprint arXiv:2203.11171.
[18] Jason Wei, Xuezhi Wang, Dale Schuurmans, Maarten Bosma, Fei Xia, Ed Chi,
Quoc V Le, Denny Zhou, et al .2022. Chain-of-thought prompting elicits reason-
ing in large language models. 24824–24837 pages.
[19] Yexin Wu, Zhuosheng Zhang, and Hai Zhao. 2024. Mitigating misleading chain-
of-thought reasoning with selective filtering. 11325–11340 pages.
[20] Hao Yu, Aoran Gan, Kai Zhang, Shiwei Tong, Qi Liu, and Zhaofeng Liu. 2024.
Evaluation of retrieval-augmented generation: A survey. 102–120.
[21] Saber Zerhoudi and Michael Granitzer. 2024. Personarag: Enhancing retrieval-
augmented generation systems with user-centric agents.arXiv preprint
arXiv:2407.09394(2024).
[22] Saber Zerhoudi and Michael Granitzer. 2026. Beyond the Click: A Framework
for Inferring Cognitive Traces in Search. InEuropean Conference on Information
Retrieval. Springer, 626–640.
[23] Shuyan Zhou, Frank F Xu, Hao Zhu, Xuhui Zhou, Robert Lo, Abishek Sridhar,
Xianyi Cheng, Tianyue Ou, Yonatan Bisk, Daniel Fried, et al .2023. Webarena: A
realistic web environment for building autonomous agents.