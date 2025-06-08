# A Two-Staged LLM-Based Framework for CI/CD Failure Detection and Remediation with Industrial Validation

**Authors**: Weiyuan Xu, Juntao Luo, Tao Huang, Kaixin Sui, Jie Geng, Qijun Ma, Isami Akasaka, Xiaoxue Shi, Jing Tang, Peng Cai

**Published**: 2025-06-04 08:22:56

**PDF URL**: [http://arxiv.org/pdf/2506.03691v1](http://arxiv.org/pdf/2506.03691v1)

## Abstract
Continuous Integration and Continuous Deployment (CI/CD) pipelines are
pivotal to modern software engineering, yet diagnosing and resolving their
failures remains a complex and labor-intensive challenge. In this paper, we
present LogSage, the first end-to-end LLM-powered framework that performs root
cause analysis and solution generation from failed CI/CD pipeline logs. During
the root cause analysis stage, LogSage employs a specialized log preprocessing
pipeline tailored for LLMs, which extracts critical error logs and eliminates
noise to enhance the precision of LLM-driven root cause analysis. In the
solution generation stage, LogSage leverages RAG to integrate historical
resolution strategies and utilizes tool-calling to deliver actionable,
automated fixes. We evaluated the root cause analysis stage using a newly
curated open-source dataset, achieving 98\% in precision and 12\% improvement
over naively designed LLM-based log analysis baselines, while attaining
near-perfect recall. The end-to-end system was rigorously validated in a
large-scale industrial CI/CD environment of production quality, processing more
than 3,000 executions daily and accumulating more than 1.07 million executions
in its first year of deployment, with end-to-end precision exceeding 88\%.
These two forms of evaluation confirm that LogSage providing a scalable and
practical solution to manage CI/CD pipeline failures in real-world DevOps
workflows.

## Full Text


<!-- PDF content starts -->

arXiv:2506.03691v1  [cs.SE]  4 Jun 2025A Two-Staged LLM-Based Framework for CI/CD
Failure Detection and Remediation with Industrial
Validation
Weiyuan Xu1,2†, Juntao Luo2†, Tao Huang2, Kaixin Sui2, Jie Geng2,
Qijun Ma2, Isami Akasaka2, Xiaoxue Shi2, Jing Tang2, Peng Cai1∗
1East China Normal University, Shanghai, China
2ByteDance, Shanghai, China
Abstract —Continuous Integration and Continuous Deployment
(CI/CD) pipelines are pivotal to modern software engineering, yet
diagnosing and resolving their failures remains a complex and
labor-intensive challenge. In this paper, we present LogSage, the
first end-to-end LLM-powered framework that performs root
cause analysis and solution generation from failed CI/CD pipeline
logs. During the root cause analysis stage, LogSage employs a
specialized log preprocessing pipeline tailored for LLMs, which
extracts critical error logs and eliminates noise to enhance the
precision of LLM-driven root cause analysis. In the solution
generation stage, LogSage leverages RAG to integrate historical
resolution strategies and utilizes tool-calling to deliver actionable,
automated fixes. We evaluated the root cause analysis stage using
a newly curated open-source dataset, achieving 98% in precision
and 12% improvement over naively designed LLM-based log
analysis baselines, while attaining near-perfect recall. The end-
to-end system was rigorously validated in a large-scale industrial
CI/CD environment of production quality, processing more than
3,000 executions daily and accumulating more than 1.07 million
executions in its first year of deployment, with end-to-end
precision exceeding 88%. These two forms of evaluation confirm
that LogSage providing a scalable and practical solution to
manage CI/CD pipeline failures in real-world DevOps workflows.
Index Terms —Continuous Integration, Continuous Deploy-
ment, Large Language Models, Log Analysis, Root Cause Di-
agnosis, Failure Remediation
I. I NTRODUCTION
As the backbone of modern software engineering, Continu-
ous Integration and Continuous Deployment (CI/CD) has be-
come critical infrastructure for reliable, rapid software delivery
[1]. It enables higher release frequency, faster iteration, and
reduced operational risk, and is now widely adopted across
leading technology companies [2]–[4]. Large-scale empirical
studies also confirm its growing adoption in open-source com-
munities and its pivotal role in modern development practices
[5].
In parallel, large language models (LLMs) have achieved
impressive results across a range of software engineering tasks
[6], including requirements engineering [7], code retrieval
[8], automated code review [9], and unit test enhancement
†Equal contribution.∗Corresponding author.[10]. Encouraged by these advances, recent work has begun
exploring LLM-based approaches for failure detection and
remediation—opening up new possibilities for intelligent au-
tomation in software delivery.
Yet despite these advancements, real-world CI/CD pipelines
continue to suffer frequent failures. Diagnosing and fixing
these failures typically requires on-call engineers to manually
inspect lengthy, semi-structured logs filled with irrelevant or
misleading information. This process is time-consuming, error-
prone and often affects both development velocity and product
stability. Moreover, many of these failures follow recurring
patterns, and organizations have accumulated substantial res-
olution knowledge stored in unstructured formats that are
difficult to retrieve or reuse with traditional non-LLM methods.
Without automated tools to identify and apply prior solutions,
teams face redundant effort, wasted resources, and growing
developer frustration. Given these recurring and learnable
failure patterns, why hasn’t this process been automated?
One major barrier is the lack of academic focus on CI/CD
log analysis. While log-based anomaly detection is a mature
research area, most existing work targets streaming system
logs [11], and assumes structured formats or consistent tem-
plates [12]–[16], which do not generalize well to noisy,
context-rich, file-level CI/CD logs. These approaches typically
require large volumes of labeled data for model training, lack
the ability to generate interpretable diagnostic reports, and
are incapable of producing actionable remediation strategies.
Consequently, they fall short in supporting practical root cause
analysis and resolution in real-world CI/CD environments.
Even recent efforts specifically targeting CI/CD logs [17]
continue to rely on deep learning techniques that assume
template-based logs, suffer from poor generalizability, and lack
the ability to incorporate unstructured repair knowledge or
produce interpretable results. While LLMs offer promising
capabilities, their application to CI/CD root cause analysis
remains underexplored [18]. Existing baselines show consid-
erable room for improvement in diagnostic accuracy, repair
guidance, and end-to-end integration.
To bridge this gap, we present LogSage , the first LLM-
powered framework for end-to-end CI/CD failure detection

and remediation. LogSage performs fine-grained fault local-
ization, root cause analysis, and automated resolution gen-
eration directly from raw CI/CD logs. It produces human-
readable diagnostic reports and applies executable fixes via
tool-calling. The framework consists of two stages: in the
root cause analysis stage, a tailored log preprocessing pipeline
extracts high-signal input for a pre-trained LLM to reason
over; In the solution generation stage, LogSage leverages a
carefully designed multi-route retrieval-augmented generation
(RAG) module to retrieve and rank domain-specific knowledge
from diverse internal sources. This knowledge is subsequently
integrated with the diagnostic output to synthesize executable
remediation strategies. Through the built-in tool-calling ability
of LLM, LogSage supports automated recovery with minimal
human effort.
Our key contributions are as follows:
1) To the best of our knowledge, LogSage is the first
and only end-to-end LLM-based framework for fully
automated CI/CD failure detection and remediation,
validated by a year-long deployment at a major tech
company with over 1.07M executions and88%+ pre-
cision .
2) We design a token-efficient log preprocessing strategy
that filters noise, expands relevant context, and adheres
to the token constraints of LLMs. This enables accurate
root cause analysis without relying on log templates.
On a newly constructed CI/CD benchmark, our method
improves root cause analysis precision by 12% over
existing prompt-based LLM baselines, achieving over
98% precision andnear-perfect recall .
3) We propose a multi-route RAG-based solution genera-
tion approach that effectively reuses historical fixes from
internal knowledge sources and leverages the LLM’s
tool-calling capabilities to generate executable strategies
for automated CI/CD failure remediation.
4) We release a curated dataset of 367 real-world CI/CD
failure cases from popular GitHub repositories, each
annotated with key log evidence to facilitate root cause
analysis and foster future research.
The rest of this paper is organized as follows: Section II re-
views related work; Section III details the LogSage framework,
covering both the high-level architecture and implementation-
level design; Section IV presents experiments and evaluation;
Section V concludes with future directions.
II. R ELATED WORK
With the rise of AI techniques in software engineering,
Continuous Integration and Continuous Deployment (CI/CD)
pipelines have increasingly adopted machine learning (ML),
deep learning (DL), and more recently, LLMs to improve
efficiency, reliability, and fault tolerance [19]. In this section,
we review prior work from three perspectives: traditional
AI methods in CI/CD pipelines, log anomaly detection, and
the emerging application of LLMs for failure detection and
remediation.A. AI in CI/CD Pipelines
Traditional ML approaches in CI/CD focus primarily on
predicting test outcomes, build success, or optimizing test
execution to reduce cost. These methods typically rely on
structured features such as code metrics and commit history,
with limited support for runtime failure analysis or recovery.
For instance, CNNs have been used to identify false positives
in static code analysis [20], while other studies aim to skip un-
necessary tests or prioritize test selection to save time and cost
[21]–[25]. Additionally, defect prediction has been explored as
an indirect way to reduce failures, though these methods do
not directly address fault localization or remediation [26], [27].
DL methods enhance predictive accuracy and generalization
in failure prediction tasks [28], [29]. However, most models
lack semantic understanding of failure causes. Mahindru et
al. [30] proposed the LA2R system, which combines log
anomaly detection with metadata prediction to retrieve relevant
resolutions. While effective, this approach depends heavily on
templates and rule-based clustering, limiting its adaptability to
evolving log formats and unseen failures.
B. Log Anomaly Detection
Among CI/CD tasks, log anomaly detection is particularly
critical and challenging due to the unstructured and context-
dependent nature of log data. Early deep learning work by Du
et al. [13] used LSTMs to learn normal log sequences and
detect anomalies. Meng et al. [14] extended this idea with
Template2Vec to capture sequential and quantitative anoma-
lies. Yang et al. [15] employed a GRU-based attention model
and semi-supervised learning to reduce labeling cost, while
Zhang et al. [16] addressed unstable logs using attention-based
Bi-LSTMs. Recent work by Le et al. [11], [12] introduced
parser-independent Transformer-based methods that improve
generalization by avoiding reliance on log parsing.
Given their strong semantic understanding and reasoning
abilities, LLMs are well suited for log anomaly detection.
Studies have shown that even prompt-based LLMs can detect
anomalies in streaming logs [31]–[34]. Qi et al. [35] proposed
a GPT-4-based detection framework, while Shan et al. [36]
leveraged configuration knowledge for anomaly detection.
Almodovar et al. [37] fine-tuned LLMs for better performance,
and Ju et al. [38] adopted RAG to enhance log interpretation.
Hybrid methods have also been proposed: Guan et al. [39]
combined BERT and LLaMA for semantic vector extraction,
and Hadadi et al. [40] used LLMs to compensate for missing
context in unstable environments. In-context learning tech-
niques were further used to refine log representation and guide
fine-tuning [41], [42].
Despite these advances, most studies focus on streaming
logs, with limited attention to file-based CI/CD logs. More-
over, few works provide large-scale industrial validation or
full pipeline integration.
C. LLMs for Failure Detection and Remediation
Beyond anomaly detection, LLMs offer unique potential in
end-to-end failure diagnosis and automated remediation. A

Fig. 1: Overview of the LogSage framework. The system follows a two-stage pipeline: Stage 1 performs root cause analysis by
extracting and processing critical log blocks, which are then analyzed by an LLM to produce interpretable RCA reports. Stage
2 generates and executes repair solution by retrieving relevant domain knowledge and guiding the LLM to call appropriate
tools. Solid arrows represent data flow in Stage 1, while dashed arrows indicate data flow in Stage 2.
recent survey identified root cause analysis and remediation
as key use cases for LLMs in AIOps [43]. For root cause
analysis, Roy et al. [44] and Wang et al. [45] proposed tool-
augmented LLM agents, while Zhang et al. [46] and Goel et al.
[47] used in-context learning and RAG for incident diagnosis
with real-world data. On the remediation side, Sarda et al. [48],
[49] and Ahmed et al. [50] automated fault recovery in cloud
environments. Wang et al. [51] incorporated Stack Overflow
into LLM-based solution generation, and Khlaisamniang et al.
[52] applied generative AI for self-healing systems.
In the CI/CD context, however, LLM-based solutions re-
main underexplored. Chaudhary et al. [53] proposed an LLM-
based assistant for CI/CD operations, and Sharma et al. [54]
outlined key challenges in adopting LLMs for fault handling
in build pipelines. While promising, these efforts have yet to
address the complexity of real-world CI/CD logs or provide
comprehensive remediation workflows.
Overall, while ML and DL methods have laid the foundation
for predictive CI/CD analytics, LLMs bring new opportunities
for building robust, context-aware, and automated failure de-
tection and remediation systems. However, systematic integra-
tion of LLMs into CI/CD pipelines, especially for file-based
logs and knowledge-driven automated remediation, remains an
open research challenge.
III. M ETHODOLOGY
In this section, we first provide an overview of the LogSage
framework architecture in Section III-A, followed by a descrip-
tion of the required data sources in Section III-B. We then
elaborate on the detailed workflows and key components ofthe two stages of LogSage in Section III-C and Section III-D,
respectively.
A. Overview of LogSage
The two-stage architecture of LogSage is illustrated in
Figure 1. Given the complete logs from a failed CI/CD
pipeline run, the first stage— Root Cause Analysis —produces
a structured JSON report containing interpretable natural lan-
guage explanations and the corresponding line numbers of key
error entries. This report then serves as input to the second
stage— Solution Generation and Execution —where LogSage
retrieves relevant knowledge and invokes the appropriate tools
via the LLM’s tool-calling capability to automatically reme-
diate the failure.
B. Data Sources
LogSage relies on multiple types of input data to support
its two-stage pipeline for diagnosing and remediating CI/CD
failures. These data sources differ in structure and function
and are selectively used in different stages of the framework
to maximize diagnostic accuracy. Notably, all data are used
at inference time only—none are involved in model train-
ing—demonstrating LogSage’s plug-and-play usability.
InRoot Cause Analysis stage , the primary input is the
complete log file generated during a failed CI/CD pipeline
execution, referred to as the failed log . This file-level log
captures the entire lifecycle of a failed run—including code
checkout, static analysis, compilation, configuration loading,
and unit testing—and serves as the main source of ground
truth for root causes analysis. These logs are unstructured
and heterogeneous, containing messages of varying severity

levels (e.g., INFO ,WARNING ,ERROR ) as well as diverse
content types such as configuration details, command outputs,
and developer-inserted messages. To ensure generalizability,
LogSage imposes no restrictions on log format, allowing it to
ingest raw textual logs from arbitrary CI/CD systems without
requiring modifications to existing logging pipelines.
However, failed log often include irrelevant or mislead-
ing information. To extract high-quality diagnostic signals,
LogSage incorporates an auxiliary data source in this stage:
recent successful logs from the same CI/CD task, referred to
as the success logs . These are used to perform log difference
analysis (log diff) , which eliminates stable log entries that ap-
pear in both failed and successful executions. This significantly
improves the signal-to-noise ratio of the input, enabling the
LLM to concentrate on genuinely anomalous log entries.
InSolution Generation Stage , LogSage retrieves internal
domain knowledge that supplements the LLM’s general rea-
soning capabilities. This stage uses the following auxiliary data
sources:
•On-call Q&A records , which document how engineers
have historically diagnosed and resolved CI/CD failures.
•Internal engineering documentation , including platform-
specific repair guides, tool usage instructions, and con-
figuration manuals.
•Historical remediation logs or case studies , which pro-
vide concrete examples of previously applied fixes.
These resources are accessed via a multi-route RAG
pipeline. Based on the root cause analysis report and critical
log blocks from Stage 1, LogSage constructs semantic queries
to retrieve relevant content, which is then used to populate
LLM prompt templates, guiding the model to generate accurate
and practice-aligned solutions.
C. Root Cause Analysis Stage
In this section, we present the design and implementation
of the root cause analysis stage. The goal of this stage is to
extract the most relevant portions of raw CI/CD failure logs
and enable accurate reasoning by the LLM, which then returns
a structured diagnostic report containing key error lines and
an interpretable root cause summary. This stage must address
several practical challenges inherent in real-world CI/CD log
analysis:
First, log heterogeneity : CI/CD pipelines differ widely in
their execution steps, commands, output formats, and logging
styles, making it infeasible to rely on a unified, structured
template to accommodate all scenarios.
Second, informational noise and misleading entries : Raw
logs often contain numerous irrelevant WARNING orERROR
messages that do not reflect the actual cause of failure. Naively
feeding the full log into the LLM can lead to degraded
reasoning quality and even hallucinations. Conversely, filtering
logs too aggressively (e.g., by extracting only lines containing
keywords such as ERROR orFAILED ) may omit critical
contextual information, causing the model to speculate and
increasing the risk of misdiagnosis.Third, input length constraints : LLMs cannot accept ar-
bitrarily long inputs. Excessively long log sequences not only
increase inference cost and latency but also reduce reasoning
accuracy due to context dilution.
To address these challenges, we design a dedicated log
preprocessing pipeline prior to model invocation. This pipeline
consists of the following modules:
•Key Log Filtering : Extracts candidate error log lines
from the failed log based on keyword matching, log tail
prioritization and log diff against recent success logs .
•Key Log Expansion : Adds contextual lines surrounding
the extracted errors to preserve semantic coherence and
prevent information loss.
•Token Overflow Pruning : Ranks expanded log blocks
by weight calculation and prunes low-priority blocks to
ensure input remains within a predefined token limit.
•Prompt Template Design : Task-specific prompts are
customized using role-playing, chain-of-thought reason-
ing, few-shot learning and output format constraints to
guide the LLM in performing accurate, interpretable, and
reproducible reasoning.
The processed critical log blocks are delivered to the LLM
through a structured prompt, allowing the model to perform
high-quality and accurate reasoning and generate a root cause
analysis report. The detailed design of each module is as
follows.
1)Key Log Filtering Module :Key log filtering is one of
the core modules in the root cause analysis stage. This module
is responsible for accurately extracting relevant error log lines
from the failed log by combining multiple parallel strategies,
including keyword matching, log tail prioritization, log diff ,
andlog template deduplication . These strategies collectively
ensure broad and precise coverage of failure-relevant log
entries. The filtering process is formalized in Algorithm 1.
Keyword matching identifies log lines containing high-risk
terms based on a curated set of failure-related keywords mined
from historical CI/CD failure cases. Any log line matching one
or more of these keywords is added to a candidate pool for
further downstream processing. The keyword set, denoted as
Kerror, includes commonly observed failure indicators such as:
fatal, fail, panic, error,
exit, kill, no such file, err:,
err!, failures:, err , missing,
exception, cannot
The log tail prioritization strategy is motivated by the
empirical observation that most critical error logs tend to
appear near the end of the file, as CI/CD pipeline failures often
lead to abrupt termination. Accordingly, log lines appearing
toward the end of the failed log are prioritized during candidate
selection.
Log diff is the most critical strategy in the key log fil-
tering module. It builds on the repetitive nature of CI/CD
pipelines—where executions typically follow consistent con-
figurations across runs—by comparing the failed log with

Algorithm 1 Key Log Filtering Process
1:Input: failed log,success logs
2:Output: filtered log
3:Initialize candidate pool← {}
4:Extract template storage from latest x success logs
using Drain()
5:foreach line linfailed logdo
6:template ←extract template( l)
7:position ←getposition( l, failed log)
8: ifcontains keyword( l)then
9: Addltocandidate pool{Keyword matching }
10: end if
11: ifisinlogtail(position )then
12: Addltocandidate pool{Log tail prioritization }
13: end if
14: iftemplate / ∈template storage then
15: Addltocandidate pool{Log diff }
16: end if
17:end for
18:filtered log←deduplicate( candidate pool)
19:return filtered log
recent success logs of the same pipeline. This is achieved
through an offline process that uses the Drain algorithm [55] to
extract structural templates from historical success logs . These
templates are stored and used to identify stable log lines that
repeatedly appear in successful runs. Such lines are treated as
non-informative background and automatically excluded from
the error candidate set. This enables the system to filter out
misleading WARNING orERROR entries without relying on
rigid templates, thereby significantly reducing noise.
To maintain the freshness and relevance of filtering tem-
plates derived from success logs , LogSage adopts a log
template deduplication strategy: for each pipeline task, only
the most recent xsuccessful runs are retained for template
extraction, where xis a configurable parameter. Empirical
analysis across diverse CI/CD projects shows that setting
x= 3 achieves the best trade-off between template diversity
and noise reduction, and is used as the default in our system.
The value of xcan be tuned based on pipeline stability and
log variance.
Through these strategies, the Key Log Filtering module is
able to deconstruct the failed log with high precision and
identify all potentially problematic log lines. After removing
redundant entries, the filtered logs are structured as pairs of
line numbers and corresponding log entries, serving as input
for subsequent processing.
2)Key Log Expansion Module :Based on manual experi-
ence in analyzing CI/CD failure logs, it is often insufficient to
rely solely on the ERROR log lines that directly report failures.
The true root cause of a pipeline error is typically identified by
examining several lines before and after the failure line. This
observation motivated the design of the Key Log Expansion
module, which provides additional contextual information tosupport LLM-based root cause analysis, helping to mitigate
hallucinations and logical errors caused by missing context.
Starting from the key error lines identified by the Key
Log Filtering module, the expansion module includes mlines
before and nlines after each key line to form a log block.
We adopt an asymmetric expansion strategy where n > m :
the preceding mlines ensure that the operations leading to
the error are captured, while the succeeding nlines provide
richer post-error information such as stack traces and impacted
scope, information that is often more critical than the pre-
error context. Overlapping blocks resulting from multiple key
lines are merged into a single cohesive block to maintain
contextual continuity. This expansion ensures that the LLM
can access sufficient surrounding information to interpret the
key log entries accurately.
In practice, we set m= 4 andn= 6 for both online de-
ployments and offline experiments. Empirical results show that
this configuration is sufficient to retain most of the contextual
information necessary for accurate root cause analysis.
3)Token Overflow Pruning Module :In practical pro-
duction environments, the token length limitation of LLMs
imposes a critical constraint, as overly long input sequences
can reduce instruction adherence, introduce reasoning errors,
and increase computational cost and latency. To address this,
LogSage incorporates a Token Overflow Pruning module that
enforces a predefined token limit during root cause analy-
sis. This module leverages a structured log weighting and
enhancement algorithm to assess the relative importance of
expanded log blocks produced in the previous stage. The
process includes four components: initial weight assignment
to identify candidate lines, pattern-based weight enhancement
to emphasize critical errors, contextual window expansion to
preserve semantic continuity, and density-based block ranking
to prioritize the most informative blocks while satisfying the
token constraint.
Initial Weight Assignment: Given the failed log L=
{l1, l2, ..., l n}and log lines from the candidate pool I=
{i1, i2, ..., i m}, we define a weight list W={w1, w2, ..., w n},
where each wjrepresents the weight assigned to line lj:
wj=

3if|I|
|L|≤αand|I| ≤βandj∈candidate pool
1ifj∈candidate pool
0otherwise
(1)
This adaptive rule (Equation 1) adds an initial weight to
each log in the candidate pool and ensures that higher weights
are assigned to sparse yet informative candidate log entries.
The parameters αandβare tunable thresholds that control
the criteria for assigning high weights. Based on empirical
observations, we set α= 0.7andβ= 500 in practice.
Pattern-Based Weight Enhancement: To further emphasize
the importance of critical log entries, we apply the following
rules to enhance their weights:

•Failure pattern detection: Log lines containing com-
mon test cases failure markers (e.g., --- FAIL: ,
Failures: ,=== FAIL: ) receive the maximum
weight:
wi= 10
•Keyword and header enhancement: Lines matching cu-
rated keywords or starting with section headers (e.g., lines
starting with #) receive a moderate boost:
wi=wi+ 2
•Recall-based reinforcement: Lines in the candidate pool
that do not contain known keywords still gain a small
additional weight:
wi=wi+ 1
This enhancement scheme ensures that both explicit and
implicit indicators of failure are adequately prioritized for
downstream pruning and diagnosis.
Contextual Window Expansion: To ensure the contextual
integrity of high-weight critical log lines, log lines with
weights above the threshold θare expanded into log blocks
again. For any wi≥θ, the log blocks in its neighborhood
[i−m, i+n]are added to the candidate pool, where mand
nrepresent the number of previous and next lines as defined
previously.
The threshold θis adaptively defined by Equation 2 accord-
ing to two different situations:
θ=(
1ifmax( W) = 1 or|{wi≥1}| ≤γ
3otherwise(2)
In the first scenario, all key log lines receive a uniform
weight of 1, or the number of high-weight lines falls below
a threshold γ, suggesting that the preceding filtering step
has limited effectiveness in isolating truly critical entries. In
this case, broader contextual expansion is required to ensure
the LLM has sufficient information for accurate reasoning.
In contrast, when high-weight lines are adequately identified,
context expansion is applied selectively around those entries
to maintain focus and efficiency. In practice, we empirically
setγ= 500 .
Density-Based Block Ranking: In order to compare the
weights between different log blocks, we define the log block
weight density. For contiguous candidate log lines that are
grouped into blocks Bj= [sj, ej], their weight density is
computed as Equation 3:
density (Bj) =Pej
i=sjwi
ej−sj+ 1(3)
After computing the weight density for all candidate log
blocks, they are ranked in descending order of density and
indexed as B1, B2, . . . , B K, where kdenotes the rank of the k-
th block in this sorted list. A greedy selection algorithm is thenapplied to include as many blocks as possible while ensuring
that the total token count does not exceed the predefined limit:
Token Limit ≥k−1X
i=1Token (Bi) +Token (Bk)
Based on empirical observations, we set the token limit at
22,000, which provides sufficient contextual coverage while
maintaining a reasonable computational cost. Under this con-
straint, high-density log blocks are retained with priority, while
low-density blocks exceeding the token limit are discarded to
achieve effective pruning.
Overall, this weighted pruning strategy strikes a practical
balance between capturing critical error information and con-
trolling input length. It is well-defined, tunable, and suited for
efficient preprocessing in real-world CI/CD scenarios.
4)Root Cause Analysis Prompt Template Design :To
facilitate root cause analysis, LogSage employs a task-specific
prompt template that incorporates prompt engineering tech-
niques such as role-playing, few-shot learning, and output
format constraints. This template provides the LLM with
explicit task instructions and takes filtered critical error log
blocks as input, guiding the model to focus on diagnostic
reasoning.
The output is strictly structured in JSON format, consisting
of two parts: log analysis and root cause identification, as Fig.
2. The log analysis component highlights the line numbers
of log entries that are strongly correlated with the failure,
emphasizing patterns that may have triggered the issue. The
root cause identification summarizes the underlying cause in
clear natural language, serving as the foundation for down-
stream solution generation. This structured output improves
the interpretability and traceability of the diagnostic process.
1{
2 "log_analysis": [
3 {
4 "error_logs": [
5 [434,437],
6 [678],
7 [789,795]
8 ],
9 "analysis": "Unit test
TestNsqProducerSend
failed.",→
,→
10 }
11 ],
12 "root_cause": [
13 "Unit test execution failure",
14 "Unit test compilation error"
15 ]
16}
Fig. 2: Example output from the root cause analysis stage of
LogSage.
D. Solution Generation and Execution Stage
In this section, we present the design and implementation
of the solution generation and execution stage. In this stage,

LogSage combines the critical log blocks and root cause anal-
ysis report from the previous stage with domain knowledge
retrieved by the RAG module to populate a solution generation
prompt. Guided by this prompt, the LLM produces executable
remediation suggestions and leverages its tool-calling capabili-
ties to automatically select and invoke appropriate repair tools,
completing the remediation of the CI/CD pipeline failure.
1)Domain Knowledge Recall Module :To support accu-
rate and context-aware solution generation, LogSage incor-
porates a multi-route RAG module that supplements LLM’s
general knowledge with domain-specific information relevant
to CI/CD systems.
Knowledge Sources. The retrieval system draws from a
diverse knowledge base that includes curated Q&A docu-
mentation for CI/CD operations and platform-specific tools,
historical on-call summaries containing real-world incident
resolutions, and internal technical manuals detailing infrastruc-
ture configurations and engineering workflows. These sources
provide procedural expertise typically absent from general-
purpose LLM pre-training.
Query Construction: Based on the JSON-formatted diag-
nostic report, the system constructs a semantic query that
integrates readable root causes with critical log blocks. To
ensure conciseness and relevance, the query undergoes seman-
tic deduplication based on an 80% token overlap threshold,
is truncated to a maximum length of 3,000 tokens, and is
normalized into a standard format that combines a brief error
summary with the core CI/CD trace.
Retrieval Architecture: LogSage employs a hybrid retrieval
pipeline consisting of:
•Sparse retrieval: Keyword and BM25-based matching
for precise lexical alignment.
•Dense retrieval: Embedding-based semantic search using
K-th Nearest Neighbors (KNN).
•Relational retrieval: Structured query resolution from
diagnosis summary tables.
•Document retrieval: Wiki-style engineering documenta-
tion for procedural reference.
These strategies operate across both document- and query-
level tracks to capture varied granularity, enhanced further by
techniques such as query rewriting and HyDE (Hypothetical
Document Embeddings).
Multi-Route Recall and Reranking: During recall, up to
8 retrieval strategies are executed concurrently, covering both
semantic and lexical similarity at different abstraction levels.
Each strategy yields up to 60 candidates; the top 100 are se-
lected based on combined similarity scores. These are reranked
using a BGE model with timeout and retry safeguards. The
top-ranked entries are then truncated under a fixed token limit
and used to populate the final LLM prompt.
This pipeline ensures that the LLM receives targeted,
domain-specific knowledge tailored to the diagnostic context,
enabling it to generate precise and actionable remediation
strategies that align with organizational practices.2)Solution Automation :LogSage equips the LLM with
API-accessible tools commonly used for CI/CD issue reso-
lution. These tools are integrated into the prompt context,
allowing the model to autonomously select and invoke the
appropriate operations. As a result, the system can not only
identify the failure but also execute repairs in a single auto-
mated workflow.
IV. E XPERIMENTAL & INDUSTRIAL EVALUATION
To thoroughly evaluate the effectiveness of LogSage, we
conducted empirical studies addressing the following ques-
tions:
•RQ1 : How does LogSage perform in root cause analysis
precision compared to existing LLM-based baselines?
•RQ2 : How does LogSage’s cost efficiency in the root
cause analysis stage compare to LLM-based baselines?
•RQ3 : Are all modules in LogSage’s root cause analysis
phase essential?
•RQ4 : How effective is LogSage in solving CI/CD failure
issues in real-world enterprise environments?
We conducted the following experiments around the above
questions.
A. Experimental Setup
To comprehensively validate the performance of LogSage
across various CI/CD scenarios, we built a public dataset for
comparative evaluation against other LLM-based log anomaly
detection baselines. Additionally, to validate the effectiveness
of the root cause analysis phase modules, we designed ablation
experiments using the public dataset. Lastly, to assess the
end-to-end CI/CD issue resolution capability, we deployed
the LogSage-based tool in an enterprise environment and
observed operational results. The models selected for this
experiment are GPT-4o, Claude-3.7-Sonnet and Deepseek V3,
with hyperparameters set to temperature = 0.1, and all other
settings using the default values of the respective models.
1)Dataset Description :To obtain representative and di-
verse CI/CD failure log cases for evaluating the effectiveness
of LogSage and other baseline methods, referring to [56],
we crawled the top 1,000 GitHub repositories by star count
via the GitHub public API to gather their usernames and
repository names, which serve as unique identifiers. Non-
engineering-related repositories were filtered out based on
keyword filtering in README, repository description, and
tags (e.g. “awesome”, “book”, “list”). This ensured that the
cases have practical software engineering relevance. We then
used the GitHub Action public API to retrieve the latest 300
CI/CD run logs for each qualifying repository and identified
success and failure run pairs with the same workflow ID.
These pairs were crawled as individual cases. After filtering,
we obtained 367 cases from 76 repositories, each manually
analyzed to identify the key failing log lines and the root
causes. The first 117 cases were used to train the Drain log
extraction template and build the few-shot log pool, with the
remaining 250 cases used as the test set. Data collection was

completed on April 25, 2025, and the dataset will be open-
sourced on GitHub.
2)Baseline Setup :According to the research conducted in
Section 2, to the best of our knowledge, there are currently
no baseline methods using LLMs for CI/CD error log analysis
that can be directly compared. Instead, we select LogPrompt
[33] and LogGPT [35] as baselines, as they represent recent
LLM-based approaches for general log analysis and anomaly
detection:
•LogPrompt : LogPrompt is an interpretable log analysis
framework that utilizes prompt engineering techniques
to guide LLMs in detecting anomalies without requiring
any fine-tuning. It is designed for real-world online
log analysis scenarios and emphasizes interpretability
by generating natural language explanations alongside
detection results. Evaluations across nine industrial log
datasets show that LogPrompt outperforms traditional
deep learning methods in zero-shot settings.
•LogGPT : LogGPT explores ChatGPT for log-based
anomaly detection using a structured prompt-response
architecture. It integrates chain-of-thought reasoning and
domain knowledge injection to improve detection accu-
racy. The system generates structured diagnostic outputs
with explanations and remediation suggestions. Experi-
ments on benchmark datasets such as BGL and Spirit
demonstrate its competitiveness against state-of-the-art
deep learning baselines under few-shot and zero-shot
conditions.
Other CI/CD anomaly detection methods are excluded due
to their reliance on fixed templates or the need for model
training, which conflicts with LogSage’s plug-and-play design.
We evaluate LogPrompt and LogGPT using their original
prompt templates, with optimal settings: LogPrompt (few-shot
= 20, window = 100) and LogGPT (few-shot = 5, window =
30). As shown in Table I, LogSage is the only method that
supports the full pipeline from interpretable root cause analysis
to automated solution execution. LogPrompt can produce in-
terpretable diagnostic outputs but lacks the ability to generate
solutions. Although LogGPT claims to generate remediation
suggestions, it operates without any external knowledge inte-
gration and relies solely on the LLM’s pre-trained knowledge,
often resulting in hallucinated or impractical outputs.
TABLE I: Comparison of LogSage and Baseline Methods
Method Anomaly Interpretable Solution File-level Automated
Detection Root Cause Generation Processing Execution
LogSage ✓ ✓ ✓ ✓ ✓
LogGPT ✓ ✓ ◦ × ×
LogPrompt ✓ ✓ × × ×
3)Metrics Description :We evaluate root cause analysis
using standard metrics: Precision, Recall and F1-Score, with
TP, FP, FN and TN defined specifically for the log diagnosis
scenario as follows:
•TP: A correct detection. For LogSage, critical error log
lines must overlap ≥90% with ground truth; for Log-Prompt and LogGPT, they must fall within a predefined
window.
•FP: Detected lines that fail to meet the above criteria.
•FN: LogSage produces no output; LogPrompt/LogGPT
fail to detect anomalies despite overlap.
•TN: Only applicable to LogPrompt/LogGPT when no
detection intersects the ground truth window.
B.RQ1 : Root Cause Analysis Precision Comparison
To compare the root cause analysis performance of LogSage
with the two baseline models and four prompt settings, we
conducted experiments on three mainstream LLMs. The results
are shown in Table II.
It is evident that LogSage performs consistently well across
all models, with near-perfect scores and minimal fluctuation,
indicating high precision and robustness in handling diverse
CI/CD log formats, lengths, and error types. In contrast,
LogPrompt’s performance suffers in precision, while LogGPT
shows significant variability, with both methods exhibiting low
recall, which suggests that their window-based sampling might
miss key contextual information, leading to errors in anomaly
detection.
RQ1 Result : LogSage significantly outperforms the base-
line methods in root cause analysis, demonstrating high
precision, recall, and F1-score across various models,
ensuring stability and reliability.
C.RQ2 : Root Cause Analysis Cost Comparison
Considering the cost and time consumption associated with
LLMs, we analyzed the token usage and query rounds for each
method from the baseline experiments. The distribution of the
data is shown in the violin plots Figure 3 and 4.
Fig. 3: Token usage for RCA across methods and LLMs.
As shown in the token graph, the majority of CI/CD logs
in the dataset have a length ranging from 104to105tokens,
with a long-tail distribution for some projects with unusually
large logs. LogPrompt and LogGPT, which use stream-based
processing through a sliding window without token length
limits, show a rapid increase in token consumption as log
length increases, leading to unnecessary cost in real-world
scenarios. In contrast, LogSage not only achieves the best
overall performance, but also benefits from a predefined token

TABLE II: Comparison of methods performance (Precision, Recall, F1-score) across different LLMs
Group Prompt TypeGPT-4o Claude-3.7-Sonnet Deepseek V3
Precision Recall F1-score Precision Recall F1-score Precision Recall F1-score
LogSage N/A 0.9798 1.0000 0.9898 0.9837 0.9918 0.9878 0.9838 0.9959 0.9898
LogPrompt CoT 0.8580 0.4654 0.6035 0.8467 0.3361 0.4812 0.8620 0.4477 0.5893
InContext 0.6120 0.4530 0.5206 0.6864 0.4046 0.5091 0.6923 0.5018 0.5819
LogGPT Prompt-1 0.7280 0.3081 0.4330 0.8140 0.3075 0.4464 0.8006 0.3394 0.4767
Prompt-2 0.7185 0.3484 0.4693 0.7715 0.3073 0.4396 0.7380 0.4438 0.5543
Fig. 4: Query rounds for RCA across methods and LLMs.
limit, maintaining stable token consumption across different
models and ensuring more predictable costs.
As shown in the query round graph, LogPrompt and Log-
GPT exhibit a linear relationship between query rounds and
token length. For most logs, they require nearly 10 query
rounds to complete root cause analysis, with some extreme
cases needing up to 1,000 rounds. Such a high number of query
rounds is not only highly inefficient but also prone to failure
in real-world application scenarios. In comparison, LogSage
efficiently limits the query rounds to around 1 for most cases,
with minimal retries in edge cases, offering significant time
advantages.
TABLE III: Efficiency Comparison Across Methods
Method Avg. Tokens Avg. Queries Token Variability
LogSage 17,853 1.0 14.46%
LogPrompt 152,615 31.7 26.30%
LogGPT 122,353 89.4 19.00%
We also analyzed the average token consumption and
query rounds across the models in Table III. The average
token consumption of LogPrompt and LogGPT are 152k
and 122k tokens respectively, whereas LogSage maintains
an average consumption of under 18k tokens —amounting
to only 11.84% of LogPrompt’s and 14.75% of Log-
GPT’s . Moreover, LogSage demonstrates stable cross-model
efficiency, with a normalized token variability of just 14.46%,
significantly lower than LogPrompt’s 26.30% and LogGPT’s
19.00%.
In terms of query rounds, LogSage exhibits strong practical
applicability by requiring only 1.0 query round on average to
perform accurate and actionable root cause analysis for CI/CDfailures across the entire test set. In contrast, LogPrompt,
benefiting from a large window size, requires an average of
31.7 queries, while LogGPT, due to its smaller window size,
incurs an average of 89.4 queries, rendering it nearly infeasible
in real-world scenarios.
RQ2 Result : LogSage efficiently completes root cause
analysis in an average of 1 query round with 11. 84%
to 14. 75% token consumption compared to baseline
methods, making it a highly cost-effective solution in real-
world production environments.
D.RQ3 : Ablation Study
To validate the necessity and effectiveness of the key log
filtering ,key log expansion , and token overflow pruning mod-
ules in the root cause analysis stage, we conducted ablation
experiments using the GPT-4o model on the same dataset. The
experimental groups and results are shown in Figure 5.
Fig. 5: Comparison of metrics in ablation variants.
We observed that removing both modules ( w/o both ) caused
a significant drop in recall, confirming the importance of
both modules in identifying key error logs. The precision of
the group without log expansion ( w/o Log Expansion ) was
significantly higher than the group without log filtering ( w/o
Log Filter ), indicating that the log filtering module plays a
crucial role in accurate error log identification. Compared to
the group without log expansion, the full LogSage system
achieved a slightly higher precision, further demonstrating
that the contextual information introduced by the Key Log

Expansion module contributes to more accurate root cause
analysis.
RQ3 Result : The key log filtering module enables precise
identification of error logs, while the log expansion mod-
ule provides additional context, reducing misclassification.
E.RQ4 : End-to-End Effectiveness in Enterprise Environments
We deployed LogSage on an internal LLM service from
May 2024 to May 2025. Over this 52-week period, it han-
dled 1,070,613 CI/CD failures for 36,845 users across the
company, receiving consistently positive feedback. This large-
scale, year-long deployment demonstrates LogSage’s robust-
ness and practical value in an enterprise environment. Key
usage metrics are summarized below.
1)End-to-end Precision :Due to the lack of ground truth
for CI/CD log failure analysis and automated remediation, we
initially evaluated LogSage during its early deployment phase
through manual annotation on both offline tests and online
cases. As the framework matured, the end-to-end precision
improved from 61% to 88.89% , reaching a level suitable for
production deployment. The gap between this metric and the
RCA-stage results on public datasets may be the result of the
increased complexity of the complete end-to-end workflow,
where cumulative errors across multiple stages can impact
overall accuracy.
2)User Scale :As LogSage serves as a tool for CI/CD
pipeline failure diagnosis and repair, its impact is difficult
to quantify in terms of code volume. Instead, we used user
scale to indirectly reflect LogSage’s adoption in the company’s
development workflows. The weekly active user trend is in
Figure 6. Within one year of LogSage’s deployment, the
number of weekly users remained stable at more than 5,000
for most of the time, demonstrating a relatively stable user
base throughout the year. It is worth noting that the sharp
drops in user count coincide with national public holidays in
the company’s region.
3)User Coverage Rate :As shown in Figure 6, since
October 14, 2024, LogSage’s user coverage rate has consis-
tently remained above 80% , indicating high and sustained
user engagement with LogSage in real-world CI/CD failure
scenarios. A high coverage rate suggests that most developers
encountering CI/CD failures actively rely on LogSage for
diagnosis and resolution, demonstrating its effectiveness and
essential role in daily DevOps workflows.
RQ4 Result : Through extensive validation in a large-
scale, real-world development environment, the LogSage
framework demonstrated outstanding performance in pre-
cision, user scale and user coverage. This confirms
LogSage’s substantial practical value and its positive
reception among users.V. C ONCLUSION & F UTURE WORK
In this paper, we present LogSage, the first end-to-end
LLM-powered framework for CI/CD failure detection and
automated remediation, validated through large-scale deploy-
ment in a real-world industrial environment. By integrating
a token-efficient log preprocessing pipeline, structured diag-
nostic prompting, multi-route RAG-based knowledge retrieval,
and executable tool-calling strategies, LogSage enables accu-
rate, interpretable root cause analysis and effective, automated
resolution generation for complex CI/CD failures.
Empirical evaluations across multiple LLM backends and
baselines show that LogSage achieves significant improve-
ments in both performance and efficiency. It outperforms state-
of-the-art LLM-based methods in root cause precision (up
to 98%) while reducing token consumption by over 85%
compared to prior approaches. In industrial settings, LogSage
demonstrates sustained adoption, processing over 1 million
CI/CD failures and maintaining high user coverage with end-
to-end precision exceeding 88%.
In future work, we plan to upgrade LogSage into a more
autonomous and adaptive LLM-Agent capable of orchestrating
complex remediation workflows through iterative reasoning
and proactive decision-making. This would empower LogSage
to not only respond to failures but also proactively initiate
multi-step diagnostics and repairs. We also aim to extend its
scope beyond reactive failure handling to broader DevOps
scenarios, including fault prediction, anomaly prevention, and
automated incident response. These directions involve in-
tegrating with observability tools, modeling failure trends,
and aligning with real-world operational workflows—pushing
LogSage toward becoming an intelligent and proactive De-
vOps collaborator.
REFERENCES
[1] B. Fitzgerald and K.-J. Stol, “Continuous software engineering: A
roadmap and agenda,” Journal of Systems and Software , vol. 123, pp.
176–189, 2017. doi: 10.1016/j.jss.2015.06.063
[2] Microsoft Inside Track, “DevOps is sending engineering practices
up in smoke”, Microsoft, Apr. 15, 2024. [Online]. Available:
https://www.microsoft.com/insidetrack/blog/devops-is-sending-
engineering-practices-up-in-smoke/ [Accessed: May 10, 2025].
[3] Engineering at Meta, “Rapid release at massive scale”,
Meta Engineering Blog, Aug. 31, 2017. [Online]. Available:
https://engineering.fb.com/2017/08/31/web/rapid-release-at-massive-
scale/ [Accessed: May 10, 2025].
[4] GitHub Engineering, “Building organization-wide governance and
re-use for CI/CD and automation with GitHub Actions”, GitHub
Blog, Apr. 5, 2023. [Online]. Available: https://github.blog/enterprise-
software/devops/building-organization-wide-governance-and-re-use-
for-ci-cd-and-automation-with-github-actions/ [Accessed: May 10,
2025].
[5] M. Hilton, T. Tunnell, K. Huang, D. Marinov, and D. Dig, “Usage,
costs, and benefits of continuous integration in open-source projects”,
inProc. 31st IEEE/ACM Int. Conf. Automated Software Engineering
(ASE) , Singapore, 2016, pp. 426–437. doi: 10.1145/2970276.2970358.
[6] X. Hou, Y . Zhao, Y . Liu, Z. Yang, K. Wang, L. Li, X. Luo, D. Lo, J.
Grundy, and H. Wang, “Large language models for software engineering:
A systematic literature review”, arXiv preprint arXiv:2308.10620 , 2024.
[Online]. Available: https://arxiv.org/abs/2308.10620
[7] A. Hemmat, M. Sharbaf, S. Kolahdouz-Rahimi, K. Lano, and S. Y .
Tehrani, “Research directions for using LLM in software requirement
engineering: A systematic review”, Frontiers in Computer Science , vol.
7, 2025. doi: 10.3389/fcomp.2025.1519437.

Fig. 6: Weekly active user count and user coverage rate
[8] H. Li, X. Zhou, and Z. Shen, “Rewriting the code: A simple method
for large language model augmented code search”, in Proc. 62nd Annu.
Meeting Assoc. Comput. Linguistics (ACL) , Bangkok, Thailand, Aug.
2024, pp. 1371–1389. doi: 10.18653/v1/2024.acl-long.75.
[9] T. Sun, J. Xu, Y . Li, Z. Yan, G. Zhang, L. Xie, L. Geng, Z. Wang,
Y . Chen, Q. Lin, W. Duan, and K. Sui, “BitsAI-CR: Automated code
review via LLM in practice”, in Proc. 33rd ACM Int. Conf. Found.
Softw. Eng. (Ind. Track) , 2025. arXiv preprint arXiv:2501.15134 , 2024.
[Online]. Available:https://arxiv.org/abs/2501.15134.
[10] N. Alshahwan, J. Chheda, A. Finogenova, B. Gokkaya, M. Harman,
I. Harper, A. Marginean, S. Sengupta, and E. Wang, “Automated unit
test improvement using large language models at Meta,” in Companion
Proc. of the 32nd ACM Int. Conf. on the Foundations of Software
Engineering (FSE 2024) , Porto de Galinhas, Brazil, 2024, pp. 185–196.
doi: 10.1145/3663529.3663839
[11] V .-H. Le and H. Zhang, “Log-based anomaly detection with deep
learning: how far are we?,” in Proc. 44th Int. Conf. Software Engi-
neering (ICSE) , Pittsburgh, Pennsylvania, pp. 1356–1367, 2022. doi:
10.1145/3510003.3510155
[12] V .-H. Le and H. Zhang, “Log-based anomaly detection without log
parsing,” in Proc. 36th IEEE/ACM Int. Conf. Automated Software
Engineering (ASE) , Melbourne, Australia, pp. 492–504, 2022. doi:
10.1109/ASE51524.2021.9678773
[13] M. Du, F. Li, G. Zheng, and V . Srikumar, “DeepLog: anomaly
detection and diagnosis from system logs through deep learning,”
inProc. 2017 ACM SIGSAC Conf. Computer and Communications
Security (CCS) , Dallas, Texas, USA, 2017, pp. 1285–1298. doi:
10.1145/3133956.3134015
[14] W. Meng, Y . Liu, Y . Zhu, S. Zhang, D. Pei, Y . Liu, Y . Chen, R. Zhang,
S. Tao, P. Sun, and R. Zhou, “LogAnomaly: unsupervised detection
of sequential and quantitative anomalies in unstructured logs,” in Proc.
28th Int. Joint Conf. Artificial Intelligence (IJCAI) , 2019, pp. 4739–4745.
Available: https://www.ijcai.org/proceedings/2019/0658.pdf
[15] L. Yang, J. Chen, Z. Wang, W. Wang, J. Jiang, X. Dong, and W. Zhang,
“Semi-supervised log-based anomaly detection via probabilistic label
estimation,” in Proc. 2021 IEEE/ACM 43rd Int. Conf. Software Engineer-
ing (ICSE) , 2021, pp. 1448–1460. doi: 10.1109/ICSE43902.2021.00130
[16] X. Zhang, Y . Xu, Q. Lin, B. Qiao, H. Zhang, Y . Dang, C. Xie, X. Yang,
Q. Cheng, Z. Li, J. Chen, X. He, R. Yao, J.-G. Lou, M. Chintalapati, F.
Shen, and D. Zhang, “Robust log-based anomaly detection on unstable
log data,” in Proc. 27th ACM Joint Meeting Eur. Softw. Eng. Conf.
Symp. Found. Softw. Eng. , Tallinn, Estonia, 2019, pp. 807–817. doi:
10.1145/3338906.3338931
[17] F. Hassan, N. Meng, and X. Wang, “UniLoc: Unified fault localization
of continuous integration failures,” ACM Transactions on Software
Engineering and Methodology , vol. 32, no. 6, pp. 1–31, Article 136,
Sep. 2023. doi: 10.1145/3593799
[18] A. K. Arani, T. H. M. Le, M. Zahedi, and M. A. Babar, “Systematic
literature review on application of learning-based approaches in contin-
uous integration,” IEEE Access , vol. 12, pp. 135419–135450, 2024. doi:
ACCESS.2024.3424276[19] A. S. Mohammed, V . R. Saddi, S. K. Gopal, S. Dhanasekaran,
and M. S. Naruka, “AI-driven continuous integration and continu-
ous deployment in software engineering,” in Proc. 2024 2nd Int.
Conf. on Disruptive Technologies (ICDT) , 2024, pp. 531–536. doi:
10.1109/ICDT61202.2024.10489475
[20] S. Lee, S. Hong, J. Yi, T. Kim, C.-J. Kim, and S. Yoo, “Classifying
false positive static checker alarms in continuous integration using
convolutional neural networks,” in Proc. 2019 12th IEEE Conf. on
Software Testing, Validation and Verification (ICST) , 2019, pp. 391–401.
doi: 10.1109/ICST.2019.00048
[21] G. Grano, T. V . Titov, S. Panichella, and H. C. Gall, “How high will
it be? Using machine learning models to predict branch coverage in
automated testing,” in Proc. 2018 IEEE Workshop on Machine Learning
Techniques for Software Quality Evaluation (MaLTeSQuE) , 2018, pp.
19–24. doi: 10.1109/MALTESQUE.2018.8368454
[22] G. Grano, T. V . Titov, S. Panichella, and H. C. Gall, “Branch coverage
prediction in automated testing,” J. Softw. Evol. Process , vol. 31, no. 9,
pp. 1–18, Oct. 2019. doi: 10.1002/smr.2158
[23] K. Al-Sabbagh, M. Staron, M. Ochodek, and W. Meding, “Early pre-
diction of test case verdict with bag-of-words vs. word embeddings,” in
Proc. CEUR Workshop Proceedings , Dec. 2019. Available: https://ceur-
ws.org/V ol-2568/paper6.pdf
[24] K. W. Al-Sabbagh, M. Staron, R. Hebig, and W. Meding, “Predicting
test case verdicts using textual analysis of committed code churns,”
inProc. Int. Workshop on Software Measurement and Int. Conf. on
Software Process and Product Measurement (IWSM Mensura) , Haarlem,
The Netherlands, 2019, pp. 138–153. Available: https://ceur-ws.org/V ol-
2476/paper7.pdf
[25] R. Abdalkareem, S. Mujahid, and E. Shihab, “A machine learning ap-
proach to improve the detection of CI skip commits,” IEEE Transactions
on Software Engineering , vol. 47, no. 12, pp. 2740–2754, 2021. doi:
10.1109/TSE.2020.2967380
[26] Y . Koroglu, A. Sen, D. Kutluay, A. Bayraktar, Y . Tosun, M. Cinar, and
H. Kaya, “Defect prediction on a legacy industrial software: a case study
on software with few defects,” in Proc. 4th Int. Workshop on Conducting
Empirical Studies in Industry (CESI) , Austin, TX, USA, 2016, pp. 14–
20. doi: 10.1145/2896839.2896843
[27] R. Mamata, K. Smith, A. Azim, Y .-K. Chang, Q. Taiseef, R.
Liscano, and G. Seferi, “Failure prediction using transfer learn-
ing in large-scale continuous integration environments,” in Proc.
32nd Annu. Int. Conf. on Computer Science and Software Engi-
neering (CASCON) , Toronto, Canada, 2022, pp. 193–198. Available:
https://dl.acm.org/doi/10.5555/3566055.3566079
[28] A. Mishra and A. Sharma, “Deep learning based continuous integration
and continuous delivery software defect prediction with effective opti-
mization strategy,” in Knowledge-Based Systems , vol. 296, p. 111835,
2024. [Online]. doi: 10.1016/j.knosys.2024.111835
[29] I. Saidani, A. Ouni, and M. W. Mkaouer, “Improving the prediction of
continuous integration build failures using deep learning,” in Automated
Software Engineering , vol. 29, no. 1, pp. 1–61, May 2022. [Online]. doi:
10.1007/s10515-021-00319-5

[30] R. Mahindru, H. Kumar, and S. Bansal, “Log anomaly to resolution:
AI based proactive incident remediation,” in Proc. 36th IEEE/ACM Int.
Conf. on Automated Software Engineering (ASE) , 2021, pp. 1353–1357.
doi: 10.1109/ASE51524.2021.9678815
[31] Y . Xiao, V .-H. Le, and H. Zhang, “Demonstration-free: Towards more
practical log parsing with large language models,” in Proc. 39th
IEEE/ACM Int. Conf. on Automated Software Engineering (ASE) , Sacra-
mento, CA, USA, 2024, pp. 153–165. doi: 10.1145/3691620.3694994
[32] C. Egersdoerfer, D. Zhang, and D. Dai, “Early exploration of using
ChatGPT for log-based anomaly detection on parallel file systems
logs,” in Proc. 32nd Int. Symp. on High-Performance Parallel and
Distributed Computing (HPDC ’23) , Orlando, FL, USA, 2023, pp. 315–
316. [Online]. doi: 10.1145/3588195.3595943
[33] Y . Liu, S. Tao, W. Meng, J. Wang, W. Ma, Y . Chen, Y . Zhao, H. Yang,
and Y . Jiang, “Interpretable online log analysis using large language
models with prompt strategies,” in Proc. 32nd IEEE/ACM Int. Conf.
Program Comprehension (ICPC) , Lisbon, Portugal, pp. 35–46, 2024.
doi: 10.1145/3643916.3644408
[34] Y . Liu, S. Tao, W. Meng, F. Yao, X. Zhao, and H. Yang, “LogPrompt:
Prompt engineering towards zero-shot and interpretable log analysis,”
inProc. 2024 IEEE/ACM 46th Int. Conf. on Software Engineering:
Companion Proceedings (ICSE-Companion) , 2024, pp. 364–365. doi:
10.1145/3639478.3643108
[35] J. Qi, S. Huang, Z. Luan, S. Yang, C. Fung, H. Yang, D. Qian,
J. Shang, Z. Xiao, and Z. Wu, “LogGPT: exploring ChatGPT for
log-based anomaly detection,” in Proc. 2023 IEEE Int. Conf. High
Performance Computing & Communications, Data Science & Systems,
Smart City & Dependability in Sensor, Cloud & Big Data Systems
& Application (HPCC/DSS/SmartCity/DependSys) , pp. 273–280, 2023.
doi: 10.1109/HPCC-DSS-SmartCity-DependSys60770.2023.00045
[36] S. Shan, Y . Huo, Y . Su, Y . Li, D. Li, and Z. Zheng, “Face it
yourselves: an LLM-based two-stage strategy to localize configuration
errors via logs,” in Proc. 33rd ACM SIGSOFT Int. Symp. Software
Testing and Analysis (ISSTA) , Vienna, Austria, pp. 13–25, 2024. doi:
10.1145/3650212.3652106
[37] C. Almodovar, F. Sabrina, S. Karimi, and S. Azad, “LogFiT: Log
anomaly detection using fine-tuned language models,” in IEEE Trans-
actions on Network and Service Management , vol. 21, no. 2, pp. 1715–
1723, 2024. doi: 10.1109/TNSM.2024.3358730
[38] H. Ju, “Reliable online log parsing using large language models with
retrieval-augmented generation,” in Proc. 2024 IEEE 35th Int. Symp. on
Software Reliability Engineering Workshops (ISSREW) , 2024, pp. 99–
102. doi: 10.1109/ISSREW63542.2024.00055
[39] W. Guan, J. Cao, S. Qian, J. Gao, and C. Ouyang, “LogLLM: log-
based anomaly detection using large language models,” arXiv preprint
arXiv:2411.08561 , 2025. Available: https://arxiv.org/abs/2411.08561
[40] F. Hadadi, Q. Xu, D. Bianculli, and L. Briand, “LLM meets ML: Data-
efficient anomaly detection on unseen unstable logs,” arXiv preprint
arXiv:2406.07467 , 2025. Available: https://arxiv.org/abs/2406.07467
[41] A. Fariha, V . Gharavian, M. Makrehchi, S. Rahnamayan, S. Alwidian,
and A. Azim, “Log anomaly detection by leveraging LLM-based parsing
and embedding with attention mechanism,” in Proc. 2024 IEEE Cana-
dian Conf. on Electrical and Computer Engineering (CCECE) , 2024,
pp. 859–863. doi: 10.1109/CCECE59415.2024.10667308
[42] M. He, T. Jia, C. Duan, H. Cai, Y . Li, and G. Huang, “LLMeLog: An
approach for anomaly detection based on LLM-enriched log events,”
inProc. 2024 IEEE Int. Symp. on Software Reliability Engineering
(ISSRE) , pp. 132–143, 2024. doi: 10.1109/ISSRE62328.2024.00023
[43] L. Zhang, T. Jia, M. Jia, Y . Wu, A. Liu, Y . Yang, Z. Wu, X. Hu, P.
S. Yu, and Y . Li, “A survey of AIOps for failure management in the
era of large language models,” arXiv preprint arXiv:2406.11213 , 2024.
Available: https://arxiv.org/abs/2406.11213
[44] D. Roy, X. Zhang, R. Bhave, C. Bansal, P. Las-Casas, R. Fonseca, and
S. Rajmohan, “Exploring LLM-based agents for root cause analysis,”
inCompanion Proc. 32nd ACM Int. Conf. Found. Softw. Eng. , Porto de
Galinhas, Brazil, 2024, pp. 208–219. doi: 10.1145/3663529.3663841
[45] Z. Wang, Z. Liu, Y . Zhang, A. Zhong, J. Wang, F. Yin, L. Fan, L.
Wu, and Q. Wen, “RCAgent: Cloud root cause analysis by autonomous
agents with tool-augmented large language models,” in Proc. 33rd ACM
Int. Conf. on Information and Knowledge Management (CIKM) , Boise,
ID, USA, 2024, pp. 4966–4974. doi: 10.1145/3627673.3680016
[46] X. Zhang, S. Ghosh, C. Bansal, R. Wang, M. Ma, Y . Kang, and
S. Rajmohan, “Automated root causing of cloud incidents using in-
context learning with GPT-4,” in Companion Proc. 32nd ACM Int. Conf.Found. Softw. Eng. , Porto de Galinhas, Brazil, 2024, pp. 266–277. doi:
10.1145/3663529.3663846
[47] D. Goel, F. Husain, A. Singh, S. Ghosh, A. Parayil, C. Bansal, X.
Zhang, and S. Rajmohan, “X-lifecycle learning for cloud incident
management using LLMs,” arXiv preprint, arXiv:2404.03662, 2024.
Available: https://arxiv.org/abs/2404.03662
[48] K. Sarda, Z. Namrud, M. Litoiu, L. Shwartz, and I. Watts, “Leveraging
large language models for the auto-remediation of microservice appli-
cations: An experimental study,” in Companion Proc. of the 32nd ACM
Int. Conf. on the Foundations of Software Engineering (FSE) , Porto de
Galinhas, Brazil, 2024, pp. 358–369. doi: 10.1145/3663529.3663855
[49] K. Sarda, Z. Namrud, R. Rouf, H. Ahuja, M. Rasolroveicy, M. Litoiu, L.
Shwartz, and I. Watts, “ADARMA: Auto-detection and auto-remediation
of microservice anomalies by leveraging large language models,” in
Proc. 33rd Annu. Int. Conf. on Computer Science and Software En-
gineering (CASCON) , Las Vegas, NV , USA, 2023, pp. 200–205. doi:
10.5555/3615924.3615949
[50] T. Ahmed, S. Ghosh, C. Bansal, T. Zimmermann, X. Zhang, and S.
Rajmohan, “Recommending root-cause and mitigation steps for cloud
incidents using large language models,” in Proc. 45th Int. Conf. Software
Engineering (ICSE) , Melbourne, Victoria, Australia, pp. 1737–1749,
2023. doi: 10.1109/ICSE48619.2023.00149
[51] J. Wang, G. Chu, J. Wang, H. Sun, Q. Qi, Y . Wang, J. Qi, and J. Liao,
“LogExpert: log-based recommended resolutions generation using large
language model,” in Proc. 2024 ACM/IEEE 44th Int. Conf. Software
Engineering: New Ideas and Emerging Results (ICSE-NIER) , Lisbon,
Portugal, pp. 42–46, 2024. doi: 10.1145/3639476.3639773
[52] P. Khlaisamniang, P. Khomduean, K. Saetan, and S. Wonglapsuwan,
“Generative AI for self-healing systems,” in Proc. 2023 18th Int. Joint
Symp. on Artificial Intelligence and Natural Language Processing (iSAI-
NLP) , 2023, pp. 1–6. doi: 10.1109/iSAI-NLP60301.2023.10354608
[53] D. Chaudhary, S. L. Vadlamani, D. Thomas, S. Nejati, and M. Sa-
betzadeh, “Developing a Llama-based chatbot for CI/CD question an-
swering: A case study at Ericsson,” in Proc. 2024 IEEE Int. Conf. on
Software Maintenance and Evolution (ICSME) , Oct. 2024, pp. 707–718.
doi: 10.1109/ICSME58944.2024.00075
[54] P. Sharma and M. S. Kulkarni, “A study on unlocking the poten-
tial of different AI in continuous integration and continuous de-
livery (CI/CD),” in Proc. 2024 4th Int. Conf. on Innovative Prac-
tices in Technology and Management (ICIPTM) , 2024, pp. 1–6. doi:
10.1109/ICIPTM59628.2024.10563618
[55] P. He, J. Zhu, Z. Zheng, and M. R. Lyu, “Drain: An online log parsing
approach with fixed depth tree,” in Proc. 2017 IEEE Int. Conf. on Web
Services (ICWS) , 2017, pp. 33–40. doi: 10.1109/ICWS.2017.13
[56] C. E. Brandt, A. Panichella, A. Zaidman, and M. Beller, “LogChunks:
A data set for build log analysis,” in Proc. 17th Int. Conf. on Mining
Software Repositories (MSR) , Seoul, Republic of Korea, 2020, pp. 583–
587. doi: 10.1145/3379597.3387485