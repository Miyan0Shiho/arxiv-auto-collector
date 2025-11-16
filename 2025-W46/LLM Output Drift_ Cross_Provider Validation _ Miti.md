# LLM Output Drift: Cross-Provider Validation & Mitigation for Financial Workflows

**Authors**: Raffi Khatchadourian, Rolando Franco

**Published**: 2025-11-10 19:54:00

**PDF URL**: [https://arxiv.org/pdf/2511.07585v1](https://arxiv.org/pdf/2511.07585v1)

## Abstract
Financial institutions deploy Large Language Models (LLMs) for reconciliations, regulatory reporting, and client communications, but nondeterministic outputs (output drift) undermine auditability and trust. We quantify drift across five model architectures (7B-120B parameters) on regulated financial tasks, revealing a stark inverse relationship: smaller models (Granite-3-8B, Qwen2.5-7B) achieve 100% output consistency at T=0.0, while GPT-OSS-120B exhibits only 12.5% consistency (95% CI: 3.5-36.0%) regardless of configuration (p<0.0001, Fisher's exact test). This finding challenges conventional assumptions that larger models are universally superior for production deployment.
  Our contributions include: (i) a finance-calibrated deterministic test harness combining greedy decoding (T=0.0), fixed seeds, and SEC 10-K structure-aware retrieval ordering; (ii) task-specific invariant checking for RAG, JSON, and SQL outputs using finance-calibrated materiality thresholds (plus or minus 5%) and SEC citation validation; (iii) a three-tier model classification system enabling risk-appropriate deployment decisions; and (iv) an audit-ready attestation system with dual-provider validation.
  We evaluated five models (Qwen2.5-7B via Ollama, Granite-3-8B via IBM watsonx.ai, Llama-3.3-70B, Mistral-Medium-2505, and GPT-OSS-120B) across three regulated financial tasks. Across 480 runs (n=16 per condition), structured tasks (SQL) remain stable even at T=0.2, while RAG tasks show drift (25-75%), revealing task-dependent sensitivity. Cross-provider validation confirms deterministic behavior transfers between local and cloud deployments. We map our framework to Financial Stability Board (FSB), Bank for International Settlements (BIS), and Commodity Futures Trading Commission (CFTC) requirements, demonstrating practical pathways for compliance-ready AI deployments.

## Full Text


<!-- PDF content starts -->

LLM Output Drift: Cross-Provider Validation & Mitigation for
Financial Workflows
Raffi Khatchadourian∗
IBM – Financial Services Market
New York, USA
raffi.khatchadourian1@ibm.comRolando Franco
IBM – Financial Services Market
New York, USA
rfranco@us.ibm.com
Abstract
Financial institutions deploy Large Language Models (LLMs) for rec-
onciliations, regulatory reporting, and client communications, but
nondeterministic outputs (output drift) undermine auditability and
trust. We quantify drift across five model architectures (7B–120B
parameters) on regulated financial tasks, revealing a stark inverse
relationship: smaller models (Granite-3-8B, Qwen2.5-7B) achieve
100% output consistency at 𝑇=0.0 , while GPT-OSS-120B exhibits
only 12.5% consistency (95% CI: 3.5–36.0%) regardless of configura-
tion (p<0.0001, Fisher’s exact test). This finding challenges conven-
tional assumptions that larger models are universally superior for
production deployment.
Our contributions include: (i) a finance-calibrated determinis-
tic test harness combining greedy decoding ( 𝑇=0.0 ), fixed seeds,
and SEC 10-K structure-aware retrieval ordering; (ii) task-specific
invariant checking for RAG, JSON, and SQL outputs using finance-
calibrated materiality thresholds (±5%) and SEC citation valida-
tion; (iii) a three-tier model classification system enabling risk-
appropriate deployment decisions; and (iv) an audit-ready attesta-
tion system with dual-provider validation.
We evaluated five models—Qwen2.5-7B (Ollama), Granite-3-8B
(IBM watsonx.ai), Llama-3.3-70B, Mistral-Medium-2505, and GPT-
OSS-120B—across three regulated financial tasks. Across 480 runs
(n=16 per condition), structured tasks (SQL) remain stable even
at𝑇=0.2 , while RAG tasks show drift (25–75%), revealing task-
dependent sensitivity. Cross-provider validation confirms deter-
ministic behavior transfers between local and cloud deployments.
We map our framework to Financial Stability Board (FSB), Bank for
International Settlements (BIS), and Commodity Futures Trading
Commission (CFTC) requirements, demonstrating practical path-
ways for compliance-ready AI deployments.
CCS Concepts
•General and reference →Evaluation;•Computing method-
ologies→Natural language processing;•Information systems
→Enterprise applications;•Software and its engineering →
Software verification and validation.
Keywords
output drift, Large Language Models, financial services, nondeter-
minism, regulatory compliance, cross-provider validation, repro-
ducibility, model-tiers, slm-finance
∗Corresponding author.1 Introduction
The financial services industry’s adoption of LLMs for operational
tasks—from regulatory reporting to client communications—faces
a fundamental challenge: nondeterministic outputs that violate
audit and compliance requirements. Recent infrastructure incidents
dramatically underscore this issue.
On September 17, 2025, Anthropic reported that Claude pro-
duced random anomalies due to a miscompiled sampling algorithm
affecting only specific batch sizes [ 4]. Similarly, Thinking Machines
Lab demonstrated that a 235B parameter model at temperature=0
produced 80 unique completions across 1000 identical runs due to
batch variance effects [44].
Production incidents throughout 2024-2025 demonstrate persis-
tent challenges in LLM determinism. OpenAI’s investigation into
reported Codex degradation [ 33] reveals nondeterministic behav-
ior affecting code generation quality—paralleling our findings on
output drift. Notably, OpenAI’s release of GPT-OSS-Safeguard-20B
[34] signals industry recognition that smaller, purpose-built mod-
els may better serve compliance-critical workflows than frontier
models. Microsoft’s Azure outage on October 30, 2025, disrupted
AI services including Copilot due to global misconfiguration [ 42],
demonstrating infrastructure-induced nondeterminism in cloud
deployments. The Shadow Escape Attack exploit disclosed October
22 enables zero-click data extraction from LLMs via malicious PDFs
[35], creating compliance risks in audit-exposed processes. These
incidents align with our economic analysis of verification over-
head, as deployments like Chronograph’s Claude integration for
private equity portfolio analysis [ 13] demand deterministic controls
to prevent drift in high-stakes financial decisions.
The economic implications are substantial. Morgan Stanley es-
timates $920 billion in potential savings from AI automation of fi-
nancial knowledge work [ 28], yet as industry observer Gene Marks
notes: “Whatever time people are spending using AI is offset by
them actually verifying what AI did because no one yet trusts
AI” [ 27]. This verification overhead negates automation benefits
unless deterministic outputs can be guaranteed.
Our key empirical finding: Determinism is not universal across
model architectures. Evaluating five models across 480 runs (n=16
per condition), we demonstrate that well-engineered smaller mod-
els (7-8B parameters) achieve perfect output consistency, while
120B parameter models fail at 12.5% consistency even with identi-
cal configuration ( 𝑇=0.0 , greedy decoding, fixed seeds). This inverse
correlation between model scale and determinism fundamentally
alters deployment strategies for regulated applications, where audit
requirements mandate reproducibility over raw capability. Table 1
Results correspond to repository release v0.1.0 (commit c19dac5). For exact reproduction,
use git tagv0.1.0.
1arXiv:2511.07585v1  [cs.LG]  10 Nov 2025

AI4F @ ACM ICAIF ’25, November 15–18, 2025, Singapore Khatchadourian and Franco
Table 1:Model Tiers for Financial Compliance: Deployment Decision
Matrix
Tier Models Consistency Compliance Use Cases
Tier 17-8B100%Full All regulated tasks
Tier 240-70B 56-100% Limited Structured only
Tier 3120B12.5%Requires validation Non-compliant
Note:n=480 total runs (16 runs per condition across 5 models×3 tasks×2
temperatures); Tier 1 models (Granite-3-8B, Qwen2.5:7B) achieve audit-ready
determinism at𝑇=0.0
summarizes our model tier classification for financial AI deploy-
ment based on empirical consistency measurements.
Why Financial Services Require Different AI Standards:
Financial AI systems operate under unique constraints that distin-
guish them from general-purpose applications. First, regulatory
frameworks (Basel III, Dodd-Frank, MiFID II)[ 12,19,46] mandate
explainable and consistent decision-making for credit, trading, and
risk management. Second, financial institutions face strict audit re-
quirements where AI-driven decisions must be reproducible months
or years later for regulatory examination. Third, the high-stakes
nature of financial decisions-affecting customer creditworthiness,
investment recommendations, and market stability-demands reli-
ability levels far exceeding consumer applications. Finally, cross-
border operations require AI systems to demonstrate consistent
behavior across different regulatory jurisdictions and infrastructure
deployments.
Example: reconciliation analysts re-verify AI-produced break
explanations, turning drift into direct rework hours.
Our experimental investigation addresses this gap with four key
contributions:
•Empirical proof of achievable determinism: We demonstrate
100% output consistency at 𝑇=0.0 across varying concurrency
levels, establishing feasibility for production financial systems
•Quantified drift patterns: We measure task-specific sensitivity
to randomness, revealing that SQL generation maintains deter-
minism even at 𝑇=0.2 while RAG tasks show substantial drift
(25-75% consistency at𝑇=0.2)
•Practical mitigation framework: We provide a tiered control
system with measured effectiveness, enabling risk-appropriate
deployment strategies
•Finance-calibrated validation protocol: We introduce domain-
specific adaptations including SEC 10-K structure-aware retrieval
ordering, finance-calibrated tolerance thresholds (±5%) as accep-
tance gates, and bi-temporal audit trails mapped to FSB/CFTC
requirements—innovations requiring financial regulatory exper-
tise beyond general ML reproducibility techniques
We quantify how concurrency and sampler settings affectoutput
driftin tasks common to financial workflows (RAG, summariza-
tion, SQL). We also show how application-level controls—seeded
decoding, retrieval-order normalization, and schema-constrained
outputs—reduce drift without relying solely on infrastructure fixes.
Finally, we connect these controls to operational and regulatory
guidance for financial institutions.
Recent incident reports documenting intermittent answer vari-
ance at production scale and research on batch-invariant kernelsaddress infra-level nondeterminism. Our results complement these
by isolating and mitigatingapplication-layercontributors to drift
in finance workflows, offering controls that are deployable today.
2 Related Work
2.1 Technical Foundations of Nondeterminism
The breakthrough understanding of LLM nondeterminism comes
from Thinking Machines Lab’s September 2025 work [ 44], which
identified batch-size variation-not floating-point arithmetic-as the
primary cause. Their batch-invariant kernels achieve exact repro-
ducibility by ensuring operations yield identical results regardless
of batch composition. Recent work on SLM agents [ 8] complements
batch-invariant kernels by advocating smaller models for efficient,
deterministic agentic inference. Complementary research on de-
terministic decoding methods [ 40] and hardware-level approaches
including HPC reproducibility studies [ 5] provides additional foun-
dations. This finding shows that determinism is achievable through
engineering rather than fundamentally impossible.
Baldwin et al. [ 6] quantified the problem’s severity, demonstrat-
ing accuracy variations up to 15% across runs even at zero tempera-
ture, with performance gaps reaching 70% between best and worst
outcomes. They introduced the Total Agreement Rate (TAR) metric,
providing the quantitative framework we extend in our financial
context evaluation.
Recent research identifies additional practical nondeterminism
sources at𝑇=0.0 , including non-batch-invariant matrix multipli-
cation and attention kernels causing numeric divergence under
varying loads. Tokenizer drift—where text normalization changes
inflate token counts by up to 112% and enable command injection
vulnerabilities—further exacerbates output variability and opera-
tional costs [45].
Analysis of model drift in crisis response contexts [ 43] shows
subtle output shifts can alter narratives, paralleling financial compli-
ance requirements for stable, auditable responses. State-of-the-art
LLM observability tools [ 9] emphasize real-time monitoring of
prompt-response pairs to detect such issues.
The vLLM serving infrastructure [ 23] with PagedAttention pro-
vides efficient batching that can be extended with deterministic
kernels. Production deployments of models like Qwen with vLLM
[38] demonstrate the framework’s maturity, though as we demon-
strate, efficient serving alone doesn’t guarantee consistency-explicit
determinism controls are required.
2.2 Financial AI Requirements and Benchmarks
Recent financial AI benchmarks-FinBen [ 51], SEC-QA [ 24], and
DocFinQA [ 39]-focus on accuracy metrics while overlooking re-
producibility. While FinBen overlooks reproducibility, SLM agent
frameworks [ 8] suggest fine-tuned small models could enhance
consistency in financial QA. SEC-QA’s 1% margin tolerance for
numerical answers and DocFinQA’s 123,000-word contexts test
reasoning capability but not output stability.
Existing financial AI benchmarks evaluateaccuracybut notrepro-
ducibility—a fundamental gap for regulated deployments. A model
achieving 95% accuracy on SEC-QA but exhibiting 25% output vari-
ance across runs fails regulatory requirements despite strong per-
formance metrics. Our work addresses this overlooked dimension,
2

LLM Output Drift: Cross-Provider Validation & Mitigation for Financial Workflows AI4F @ ACM ICAIF ’25, November 15–18, 2025, Singapore
demonstrating that model selection for compliance requires deter-
minism evaluation alongside traditional accuracy benchmarks.
The regulatory landscape demands more. The Financial Stability
Board [ 16,17] requires “consistent and traceable decision-making,”
the Bank for International Settlements [ 7,36] mandates “clear ac-
countability across the AI lifecycle,” and CFTC guidance [ 10,11]
demands “proper documentation of AI system outcomes.” Addi-
tional regulatory oversight comes from the Federal Reserve [ 15]
and the Office of the Comptroller of the Currency [ 31]. Our work
bridges the gap between these requirements and technical capa-
bilities, incorporating reproducibility guidance from PyTorch and
vLLM [37, 49].
What changed since 2024?Greedy decoding at 𝑇=0.0 alone was
not sufficient in prior runs due to retrieval order nondeterminism
and batching variance. Our harness adds: (1) DeterministicRetriever
implementing multi-key ordering (score ↓, section_priority ↑, snip-
pet_id ↑, chunk_idx ↑) that encodes SEC 10-K disclosure precedence
rules, treating retrieval order as acompliance requirementunder
Basel III explainability mandates rather than merely a performance
optimization, (2) fixed seeds and sampler params in manifests, and
(3) schema/invariant checks (SQL) with finance-calibrated±5% tol-
erance thresholds to constrain decoding freedom. These domain-
specific adaptations—mapping document structure to regulatory
obligations—eliminate residual drift at 𝑇=0.0 across concurrencies.
3 Experimental Design
Models and decoding.We initially evaluate both local and cloud
deployment patterns:Qwen2.5-Instruct 7B via Ollama(local)[ 32][41]
andIBM Granite-3-8B-Instruct via IBM watsonx.ai(cloud)[ 22][21],
with temperature 𝑇∈{ 0.0,0.2}, fixed seed(s), and consistent de-
coding parameters. Our framework is provider-agnostic and com-
patible with other cloud APIs (e.g., Google Vertex AI, Amazon
Bedrock). We prioritized open-source models over closed-source
frontier models for greater transparency in validating determinism
mechanisms—enabling direct examination of seed handling, sam-
pling algorithms, and inference implementations that are critical
for proving reproducibility claims. We log complete run manifests
(Python/OS/lib versions, model digests, API versions, decoding
params) per trial to ensure reproducibility across environments.
Drift metric.Regulators demand proof that repeated runs pro-
duce identical outputs. To quantify how much outputs vary, we
measure string similarity: if two outputs differ by even small edits,
auditors may reject the system. We count two outputs as identical
if their normalized edit distance is ≤𝜖(here𝜖=0unless stated). The
normalized edit distance is defined as:
𝑑𝑛𝑜𝑟𝑚(𝑠1,𝑠2)=𝐸𝐷(𝑠 1,𝑠2)
max(|𝑠 1|,|𝑠2|)(1)
where𝐸𝐷is the Levenshtein edit distance [ 25] and|𝑠|denotes
string length. We report identity rate with Wilson 95% CIs[ 50] and
include𝑁for each condition.
Intuition:This metric controls audit risk by flagging any deviation
that would invalidate a reproducibility attestation.
Factual drift.Financial workflows require not just identical phras-
ing but identical facts—citation mismatches or numeric changesviolate regulatory sourcing rules. A model that reports $1.05M in-
stead of $1.00M fails the materiality threshold, triggering audit
re-work. Factual drift counts differ if (a) the set of citation IDs dif-
fers or (b) any extracted numeric value differs after canonicalization
(strip commas; normalize percents; signs). Formally:
𝐹𝐷(𝑂 1,𝑂2)=⊮[citations(𝑂 1)≠citations(𝑂 2)]
+⊮[|num(𝑂 1)−num(𝑂 2)|>𝜖](2)
where ⊮is the indicator function and 𝜖=0.05(our 5% tolerance
reflecting GAAP materiality thresholds). We report the fraction of
runs with any such mismatch.
Operational throughput.We report mean latency and tokens/sec
per condition and emphasize that, in our measurements, throughput
changes at fixed𝑇did not correlate with identity rate (see §4.4).
Finance-specific validation invariants.Our framework incorpo-
rates domain-calibrated acceptance gates reflecting regulatory re-
quirements rather than generic output matching: (i)SEC citation val-
idation—RAG outputs must preserve exact citation references (e.g.,
[citi_2024_10k] ) to satisfy regulatory sourcing requirements; (ii)
Finance-calibrated tolerance thresholds—SQL queries against P&L
data use±5% tolerance reflecting auditing materiality practice; (iii)
MiFID II cross-jurisdiction consistency—dual-provider validation re-
duces the likelihood of outputs varying across cloud (IBM wat-
sonx.ai) and local (Ollama) deployments to satisfy cross-border
regulatory requirements. These domain-specific tolerances encode
financial regulatory knowledge non-obvious to AI engineers with-
out FSM expertise.
3.1 Task Selection
We designed three tasks representing key financial operations:
Task 1: Securities and Exchange Commission (SEC) 2024
Filing RAG Q&A- Tests retrieval-augmented generation against
actual SEC 2024 10-K filings from Citigroup, Goldman Sachs, and
JPMorgan Chase (1.4M-1.7M characters each) available via SEC
EDGAR database [ 48]. The corpus includes real financial disclosures,
risk factors, and operational metrics. Success requires consistent
facts and citation references (e.g.,[citi_2024_10k]) across runs.
Task 2: Policy-Bounded JSON Summarization- Generates
structured client communications with required fields (client_name,
summary, compliance_disclaimer). The disclaimer must exactly
match regulatory templates.
Task 3: Text-to-SQL with Invariants- Converts natural lan-
guage to SQL queries against a simulated P&L database with post-
execution validation ensuring sums match known totals within 5%
tolerance.
3.2 Experimental Protocol
Variables: Concurrency ∈{1,4,16}, Temperature∈{0.0,0.2}, greedy
decoding, 0-100ms uniform random tool latency.
Metrics: Normalized edit distance [ 25], factual drift rate (citation
consistency), schema violation rate (JSON validity), decision flip
rate (binary outcomes), mean latency.
Infrastructure: Mixed deployment testing both local serving (Ol-
lama with Qwen2.5:7B-instruct) and cloud serving (IBM watsonx.ai
3

AI4F @ ACM ICAIF ’25, November 15–18, 2025, Singapore Khatchadourian and Franco
with IBM Granite-3-8B-instruct), enabling cross-provider determin-
ism validation and real-world deployment pattern analysis. Statis-
tical Methods: Each experimental condition was evaluated with
n=16 runs. We report 95% Wilson confidence intervals [ 1,50] for all
proportion estimates and use Fisher’s exact test [ 18] for pairwise
model comparisons. Statistical significance threshold was set at
𝛼=0.05, with𝑝<0.0001indicating highly significant differences
between model tiers.
Bi-Temporal Regulatory Audit System:All experimental
runs generate immutable audit logs stored as JSONL traces in
traces/*.jsonl , capturing complete prompt-response pairs, times-
tamps, latency metrics, model configurations, and citation sources.
Beyond standard logging, our framework captures decision-level
compliance metrics (citation_accuracy, schema_violation, decision_flip)
mapped to specific regulatory requirements (FSB “consistent deci-
sions,” CFTC 24-17 “document all AI outcomes”). Crucially, mani-
fests include corpus version IDs and snippet-level provenance, en-
abling replay and attestation months after decisions were made—even
if source documents evolved—satisfying financial audit timelines
that exceed typical ML experiment reproducibility windows.
3.3 Data Sources
The RAG task uses real SEC 10-K filings from Citigroup, Goldman
Sachs, and JPMorgan Chase (2024).1
The policy-bounded JSON summarization task uses synthetic
client communication templates generated via structured prompts
to the evaluated LLMs, ensuring compliance disclaimers match
regulatory standards (e.g., fixed templates for disclaimers).
The Text-to-SQL task uses a synthetic database toy_finance.sqlite
containing tables for financial transactions, accounts, and balances.
This database was generated using a Python script with the sqlite3
library to create the schema (tables: accounts, transactions, bal-
ances) and the faker library2to populate∼1000 realistic entries
with names, dates, amounts, categories, and descriptions. The gen-
eration script ( generate_toy_finance.py ) is available in the sup-
plementary materials and was run locally to ensure reproducibility.
Prompts for all tasks (e.g., “Generate a SQL query for: [query]” for
SQL, with invariants like “ensure sum matches total”) are versioned
and logged in traces/*.jsonl ; prompt templates are in Appendix
D.
3.4 Statistical Notation Used in This Paper
Understanding Statistical Notation
Throughout this paper, we report two key statistical measures:
•95% Confidence Interval (CI): The range within which we
are 95% confident the true consistency rate lies. For example,
“12.5% [3.5–36.0]” means the measured consistency was 12.5%,
but the true value likely falls between 3.5% and 36.0%.
•𝑝-value: Measures whether differences between models are
statistically significant. Values 𝑝< 0.05indicate significance;
𝑝< 0.0001indicateshighlysignificant differences unlikely
due to chance.
1Downloaded from the SEC EDGAR database: https://www.sec.gov/edgar/search/.
2Faker library for generating synthetic data: https://github.com/joke2k/faker [14].4 Results
Our evaluation focuses on deterministic behavior as a compliance
requirement rather than a performance optimization. This is not
a performance argument—it is a governance one. Financial insti-
tutions require reproducible outputs to meet audit requirements,
regardless of whether nondeterminism might improve model cre-
ativity or capability.
4.1 Cross-Provider Validation (Local vs. Cloud)
Table 2 presents our cross-provider validation results, demonstrat-
ing remarkable consistency between local (Ollama) and cloud (IBM
watsonx.ai) deployments at𝑇=0.0.
Table 2:Cross-Provider Multi-Model Validation at𝑇=0.0
Provider Model Task Consistency Latency (s)
Ollama Qwen2.5:7B RAG 100.0% 2.82
IBM watsonx.ai Granite-3-8B RAG 100.0% 3.12
Ollama Qwen2.5:7B SQL 100.0% 0.72
IBM watsonx.ai Granite-3-8B SQL 100.0% 1.24
Ollama Qwen2.5:7B Summary 100.0% 1.42
IBM watsonx.ai Granite-3-8B Summary 100.0% 2.18
4.2 Model-Dependent Determinism: Size vs.
Consistency Trade-offs
At𝑇=0.0 , Granite-3-8B and Qwen2.5:7B achieved 100% consistency,
while GPT-OSS-120B reached 12.5%; see Table 3 for values and
CIs. To validate this size-consistency relationship, we expanded
our evaluation to include Llama-3.3-70B[ 2] and Mistral-Medium-
2505[ 3], confirming intermediate consistency degradation (75% and
56% respectively for RAG tasks) as model scale increases. Tables 4
through 7 present comprehensive cross-model comparisons across
tasks and temperature settings, establishing our three-tier model
classification system.
Audit scenario: credit decision consistency.A bank using LLMs for
automated credit assessment must demonstrate to regulators that
identical customer profiles produce identical decisions. Empirically,
we observed that 𝑇=0.0 achieves this consistency only with prop-
erly engineered smaller models (Granite-3-8B, Qwen2.5:7B), while
larger models like GPT-OSS-120B fail regulatory audit requirements
regardless of configuration, highlighting the importance of model
selection for compliance.
Table 3:Baseline results at 𝑇=0.0 showing exact consistency (Qwen2.5:7B)
Task Conc. Identical (%) Mean drift Lat. (s)
rag 1 100.000 0.000 2.818
rag 4 100.000 0.000 9.509
rag 16 100.000 0.000 14.397
sql 1 100.000 0.000 0.718
sql 4 100.000 0.000 2.251
sql 16 100.000 0.000 3.237
summary 1 100.000 0.000 1.416
summary 4 100.000 0.000 4.518
summary 16 100.000 0.000 6.450
4

LLM Output Drift: Cross-Provider Validation & Mitigation for Financial Workflows AI4F @ ACM ICAIF ’25, November 15–18, 2025, Singapore
Table 4:Cross-provider validation results with 95% Wilson confidence intervals
Task Provider/Model𝑇=0.0𝑇=0.2
RAGOllama/Qwen2.5:7B 100.0% [80.6-100.0] 56.3% [35.3-76.9]
IBM watsonx.ai / Granite-3-8B 100.0% [80.6-100.0] 87.5% [64.0-97.8]
IBM watsonx.ai / Llama-3.3-70B 75.0% [50.9-91.3] 56.3% [35.3-76.9]
IBM watsonx.ai / Mistral-Medium-2505 56.3% [35.3-76.9] 25.0% [9.8-46.7]
IBM watsonx.ai / GPT-OSS-120B 12.5% [3.5-36.0] 12.5% [3.5-36.0]
SQLOllama/Qwen2.5:7B 100.0% [80.6-100.0] 100.0% [80.6-100.0]
IBM watsonx.ai / Granite-3-8B 100.0% [80.6-100.0] 100.0% [80.6-100.0]
IBM watsonx.ai / Llama-3.3-70B 100.0% [80.6-100.0] 100.0% [80.6-100.0]
IBM watsonx.ai / Mistral-Medium-2505 100.0% [80.6-100.0] 100.0% [80.6-100.0]
IBM watsonx.ai / GPT-OSS-120B 12.5% [3.5-36.0] 31.3% [13.7-54.7]
SummaryOllama/Qwen2.5:7B 100.0% [80.6-100.0] 100.0% [80.6-100.0]
IBM watsonx.ai / Granite-3-8B 100.0% [80.6-100.0] 100.0% [80.6-100.0]
IBM watsonx.ai / Llama-3.3-70B 100.0% [80.6-100.0] 100.0% [80.6-100.0]
IBM watsonx.ai / Mistral-Medium-2505 87.5% [64.0-97.8] 87.5% [64.0-97.8]
IBM watsonx.ai / GPT-OSS-120B 12.5% [3.5-36.0] 12.5% [3.5-36.0]
Key finding:Tier 1 models (Granite-3-8B, Qwen2.5:7B) maintain perfect consistency across deployments, while GPT-OSS-120B exhibits severe nondeterminism ( 𝑝< 0.0001, Fisher’s
exact test).∗GPT-OSS-120B significantly different from Tier 1 models (𝑝<0.0001, Fisher’s exact test)
Table 5:All experimental results across temperatures and concurrency
Task Temp Conc. Identical (%) Mean drift Lat. (s)
rag 0.000 1 100.000 0.000 2.818
rag 0.000 4 100.000 0.000 9.509
rag 0.000 16 100.000 0.000 14.397
rag 0.200 1 56.250 0.311 3.209
rag 0.200 4 93.750 0.044 10.867
rag 0.200 16 56.250 0.311 16.157
sql 0.000 1 100.000 0.000 0.718
sql 0.000 4 100.000 0.000 2.251
sql 0.000 16 100.000 0.000 3.237
sql 0.200 1 100.000 0.000 0.717
sql 0.200 4 100.000 0.000 2.296
sql 0.200 16 100.000 0.000 3.289
summary 0.000 1 100.000 0.000 1.416
summary 0.000 4 100.000 0.000 4.518
summary 0.000 16 100.000 0.000 6.450
summary 0.200 1 100.000 0.000 1.228
summary 0.200 4 100.000 0.000 4.016
summary 0.200 16 100.000 0.000 5.606
Note:Complete dataset (n=16 per condition, 480 total runs) across all temperature and
concurrency combinations. Architecture matters more than scale: smaller models
(7-8B) outperform larger models (120B) in deterministic behavior. RAG tasks most
sensitive to configuration; SQL generation remains robust.Table 6:Model Performance by Task Type at T=0.0
Model RAG SQL Summary Overall Rating
Tier 1: Excellence across all tasks
Granite-3-8B 100% 100% 100% Excellent
Qwen2.5:7B 100% 100% 100% Excellent
Tier 2: Task-dependent performance
Llama-3.3-70B 75% 100% 100% Good
Mistral-Medium 56% 100% 87% Fair
Tier 3: Unsuitable for regulatory use
GPT-OSS-120B 12.5% 12.5% 12.5% Poor
These results reveal extreme model-dependent variability in
deterministic behavior, fundamentally challenging assumptions
about LLM consistency in financial applications.
Our tiered classification framework (Table 10, Appendix E) cate-
gorizes models based on their compliance viability: Tier 1 models
(Granite-3-8B, Qwen2.5:7B) achieve perfect determinism at 𝑇=0.0 ,
while Tier 3 models like GPT-OSS-120B exhibit significant nonde-
terminism with only 12.5% consistency acrossall tasks and tem-
peratures-including 𝑇=0.0 . This suggests fundamental architectural
differences in inference implementation that render some models
unsuitable for financial compliance regardless of configuration.
Table 7:Model Selection Guidelines for Financial AI Deployment
Tier Consistency Risk Level Deployment Context
Tier 1 (7-8B) 100% at𝑇=0.0Low All regulated tasks
Tier 2 (40-70B) 56-100% Medium SQL/structured only
Tier 3 (120B) 12.5% High Non-compliant
5

AI4F @ ACM ICAIF ’25, November 15–18, 2025, Singapore Khatchadourian and Franco
We formalize this tier classification as:
Tier(𝑀)= 
1if𝐶(𝑀,𝑇=0.0)=1.0
2if0.5<𝐶(𝑀,𝑇=0.0)<1.0
3if𝐶(𝑀,𝑇=0.0)≤0.5(3)
where𝐶(𝑀,𝑇) denotes the consistency rate for model 𝑀at tem-
perature𝑇.
4.2.1 Statistical Significance of Model Differences.Fisher’s exact
tests confirm highly significant differences (p<0.0001) between Tier
1 models (100% consistency) and GPT-OSS-120B (12.5% consistency).
With n=16 runs per condition, we achieve 80% power to detect
differences≥30% at𝛼=0.05. The 87.5% observed difference far ex-
ceeds this threshold, confirming architectural rather than sampling
variance.
4.3 Task-Specific Sensitivity at𝑇=0.2
Introducing modest randomness ( 𝑇=0.2 ) reveals dramatic task-
dependent behavior (see Table 11 in Appendix E for complete
drift patterns). While 𝑇=0.0 for Qwen2.5:7B provides exact consis-
tency, slight temperature increases reveal significant task-specific
variations. RAG tasks show the highest sensitivity to temperature
changes, with consistency dropping to 56.25% at 𝑇=0.2 , while both
SQL and summarization remain 100% consistent at𝑇=0.2.
Audit scenario: regulatory reporting variance.During a regula-
tory examination, an institution must explain why quarterly risk
reports generated from identical data show textual variations. Our
framework identifies RAG-based document analysis as the primary
source of output variance, enabling targeted mitigation strategies.
Key finding.Structured generation (SQL) maintains perfect deter-
minism even with sampling randomness, while creative synthesis
tasks show substantial drift. RAG factual drift increases with con-
currency, suggesting batch effects compound temperature-induced
variation.
Our new SEC 2024 corpus experiments with IBM watsonx.ai
demonstrate that these determinism patterns hold across deploy-
ment architectures. RAG queries against 1.4M+ character financial
documents (Citi, Goldman Sachs, JPMorgan 10-K filings) produced
100% identical outputs at 𝑇=0.0 , including consistent citation pat-
terns ( [citi_2024_10k] ,[gs_2024_10k] ,[jpm_2024_10k] ). This
cross-provider consistency validates that deterministic financial
AI is achievable in both local (Ollama) and cloud (IBM watsonx.ai)
deployments, enabling hybrid strategies for production systems.
4.4 Performance Implications
Latency scales predictably with concurrency but doesn’t correlate
with drift. Mean latency increases from 1.35s (C=1) to 6.13s (C=16),
reflecting resource contention. Crucially, this 4.5 ×latency increase
occurs equally at both temperature settings, confirming that timing
variations don’t cause drift—only sampling randomness does.
This finding matters for production deployments: teams can
parallelize inference for throughput without introducing nondeter-
minism. The performance-determinism decoupling means banks
can scale processing capacity (e.g., batch reconciliation jobs) while
maintaining audit-ready consistency, as long as temperature re-
mains fixed at 0.0.As demonstrated in Figure 1, the relationship between drift and
latency shows no correlation, with performance scaling predictably
across concurrency levels while consistency remains stable within
temperature settings. Figures 2 through 5 further illustrate the stark
architectural differences in deterministic behavior across model
tiers.
Local deployment eliminates API rate limiting and network la-
tency variables, with our measurements showing identical drift
characteristics between cloud and on-premises inference. Institu-
tions can optimize deployment strategies based on data residency,
cost, and sovereignty requirements without compromising deter-
ministic behavior for compliance-critical workflows.
Figure 1:Drift and performance analysis.Top: Identity rate with 95% CIs
(n=16).Key finding: latency and drift are uncorrelated.Bottom: Throughput
scales predictably with concurrency (1.35s at C=1 to 6.13s at C=16) with no
impact on determinism.
Figure 2:Granite-3-8B drift analysis. Excellent deterministic behavior
similar to Qwen2.5:7B, achieving 100% consistency at temperature 0.0
across all task types.
6

LLM Output Drift: Cross-Provider Validation & Mitigation for Financial Workflows AI4F @ ACM ICAIF ’25, November 15–18, 2025, Singapore
Figure 3:Llama-3.3-70B drift analysis. Moderate nondeterminism with 75%
consistency at temperature 0.0 for RAG tasks, indicating inherent
architectural limitations.
Figure 4:Mistral-Medium-2505 drift analysis. Task-specific sensitivity
patterns: excellent SQL performance (100%) but significant RAG drift (56%
at𝑇=0.0, 25% at𝑇=0.2).
Figure 5:GPT-OSS-120B drift analysis. Nondeterminism with only 12.5%
consistency across all tasks and temperatures, demonstrating fundamental
architectural incompatibility with financial compliance requirements.5 Mitigation Framework
Based on our empirical findings and regulatory requirements, we
propose a three-tier mitigation framework for financial LLM de-
ployments:
5.1 Tier 1: Mandatory Infrastructure Controls
Deterministic Configuration Management:Enforce 𝑇=0.0 for
all production financial AI systems through infrastructure-level
controls. Implement configuration drift detection with automatic
remediation.
Cross-Provider Validation:Establish dual-provider validation
protocols using our framework to verify output consistency across
deployment environments before production deployment.
Hardware-Level Reproducibility:Implement deterministic
computing environments with fixed random seeds, consistent hard-
ware configurations, and isolated processing environments to elim-
inate infrastructure-induced variability.
5.2 Tier 2: Application-Level Mitigations
Output Versioning and Audit Trails:Implement comprehensive
logging of all LLM inputs, outputs, and configuration parameters
with immutable audit trails for regulatory examination using sys-
tems like ourtraces/*.jsonlframework.
Consistency Monitoring:Deploy real-time drift detection sys-
tems that flag output variations exceeding defined thresholds, with
automatic fallback to human review for critical decisions.
Multi-Run Consensus:For high-stakes decisions, implement
multi-run consensus mechanisms that require consistent outputs
across multiple model invocations before proceeding.
5.3 Tier 3: Governance and Oversight
Model Risk Management:Integrate LLM consistency testing
into existing model risk management frameworks, with regular
validation against our cross-provider testing protocol.
Regulatory Reporting:Establish standardized reporting mech-
anisms for LLM consistency metrics, aligned with FSB and BIS
guidance requirements.
Human Oversight Integration:Implement escalation proce-
dures for scenarios where consistency cannot be guaranteed, main-
taining human oversight for regulatory-sensitive decisions.
Cross-Provider Validation:Our testing protocol (§4.1) enables
consistent behavior across hybrid cloud architectures, allowing
institutions to maintain deterministic guarantees while preserv-
ing deployment flexibility for data residency and infrastructure
optimization strategies.
7

AI4F @ ACM ICAIF ’25, November 15–18, 2025, Singapore Khatchadourian and Franco
Implementation examples are shown in Listing 1 and Listing 2: 
1# Deterministic configuration
2params = {
3" temperature ": 0.0 ,
4" top_p ": 1.0 ,
5" seed ": 42, # Fixed seed
6" num_predict ": 512 ,
7" stop ": [" </ output >"]
8}
9
10response = ollama . generate (
11model =" qwen2 .5:7b- instruct ",
12prompt = versioned_prompt ,
13options = params
14)
15
16hash_output = hashlib . sha256 (
17response ['response']. encode ()
18). hexdigest () 
Listing 1: Deterministic configuration
 
1# IBM watsonx .ai configuration
2fromibm_watsonx_aiimportCredentials
3fromibm_watsonx_ai . foundation_modelsimport
ModelInference
4
5# Initialize credentials and project configuration
6credentials = Credentials (
7api_key ="your -api - key ",
8url =" https :// us - south .ml. cloud . ibm . com "
9)
10
11# Deterministic parameters for watsonx .ai
12params = {
13" decoding_method ": " greedy ",
14" temperature ": 0.0 ,
15" top_p ": 1.0 ,
16" random_seed ": 42,
17" max_new_tokens ": 512 ,
18" return_options ": {" input_tokens ": True , "
generated_tokens ": True }
19}
20
21# Initialize foundation model with Granite -3 -8B- Instruct
22model = ModelInference (
23model_id =" ibm / granite -3 -8b- instruct ",
24credentials = credentials ,
25project_id ="your - project -id",
26params = params
27)
28
29# Generate deterministic response
30response = model . generate_text ( versioned_prompt )
31generated_text = response . get ('generated_text','')
32
33hash_output = hashlib . sha256 (
34generated_text . encode ()
35). hexdigest () 
Listing 2: Cross-provider deterministic configuration: IBM
watsonx.ai
6 Regulatory Controls Mapping
Table 8 summarizes how our controls map to common finance
guidance.
Tier-3 governance controls can be directly aligned with existing
model risk management frameworks already familiar to financial
institutions. For example, the Federal Reserve’s SR 11-7 and OCC
model risk guidance both require independent validation, monitor-
ing, and documentation of model behavior. By embedding auditTable 8:Finance guidance mapped to concrete controls.
Guidance Control in our framework
FSB: consistent & trace-
able decisions𝑇=0.0 ; request/response hashing; im-
mutable prompt versions
BIS: accountability across
lifecycleRun manifests; deterministic retrieval;
schema-constrained decoding
CFTC 24-17: document AI
outcomesVersioned prompts & traces; dual execution;
drift monitoring
logs, dual execution checks, and determinism attestations into AI
workflows, our framework extends these established practices to
LLMs, ensuring continuity with regulatory expectations while ad-
dressing the unique risks of nondeterministic systems.
7 Threats to Validity
Determinism controls randomness, not truthfulness—we measure
repeatability, not correctness. Our evaluation covers five architec-
tures but other model families, sizes, or quantizations may behave
differently. Provider versioning, infrastructure effects (batching,
caching, rate limiting), and task corpus limitations (2024 10-K RAG,
policy JSON, Text-to-SQL) constrain generalizability. Statistical
power (𝑛=16) detects large effects robustly but may miss subtle
variations. Evaluation loads reflect mid-size bank deployments;
hyperscale environments may differ.
While SLMs achieve superior determinism in our evaluation,
Google’s November 2025 withdrawal of Gemma from AI Studio—driven
by hallucinations and non-developer misuse—demonstrates that de-
terminism alone is insufficient for production finance deployments
[20]. This incident reinforces the necessity of pairing deterministic
model selection with governance controls (§5.3) and continuous
monitoring.
8 Discussion
8.1 Cross-Provider Validation
Our cross-provider validation results provide the first empirical
evidence that financial institutions can achieve consistent LLM
behavior across deployment environments.
8.2 Practical Deployment Guidance
For financial institutions implementing LLM systems, our results
support a dual-strategy approach:
•Production workflows (credit adjudication, regulatory re-
porting, reconciliation):deploy Tier-1 7–8B models at 𝑇=0.0
with fixed seeds and invariant checks. These deterministic models
deliver audit-ready consistency for mission-critical operations.
•Frontier model experimentation:maintain sandbox environ-
ments for experimenting with larger models (40B-120B) in non-
critical workflows where audit trails are optional. This captures
innovation benefits while isolating nondeterminism from regu-
lated processes.
•Attestation:use cross-provider runs with ±5%materiality in-
variants and pinned retrieval ordering before go-live.
•Task selection:prioritize SQL and summarization over RAG
for consistency-critical applications. Structured outputs remain
8

LLM Output Drift: Cross-Provider Validation & Mitigation for Financial Workflows AI4F @ ACM ICAIF ’25, November 15–18, 2025, Singapore
deterministic even at modest temperature increases, while RAG
requires strict𝑇=0.0for compliance workflows.
•Audit infrastructure:establish comprehensive audit trails via
traces/*.jsonl framework, enabling replay and attestation
months after decisions were made.
This dual-track approach balances innovation with risk miti-
gation, allowing institutions to leverage frontier model advances
while maintaining regulatory compliance where required.
8.3 Regulatory Implications
Our framework directly addresses regulatory guidance from mul-
tiple jurisdictions [ 30]. The FSB’s emphasis on "consistent and
predictable" AI behavior aligns with our 𝑇=0.0 findings. The BIS
requirement for "robust validation processes" is supported by our
cross-provider testing protocol. Our empirical results in Section
4.1 demonstrate that deterministic behavior transfers between lo-
cal and cloud deployments. The CFTC’s focus on "audit trails and
explainability" maps to our Tier 2 application-level controls, with
implementation guidance aligned with NIST AI Risk Management
Framework [ 29]. Our multi-key retrieval ordering encodes SEC dis-
closure structure directly into infrastructure, ensuring regulatory
precedence governs model behavior rather than similarity scoring
alone.
8.4 Limitations and Future Work
Our findings demonstrate that larger model architectures (70B+
parameters) introduce consistency challenges unsuitable for regu-
lated applications. For example, while 120B models excel in creative
tasks, their inherent nondeterminism from batch effects makes
them unsuitable for credit assessments, where reproducibility is
non-negotiable. This suggests future work should focus on optimiz-
ing smaller, more deterministic architectures rather than pursuing
parameter maximization.
Our evaluation covers five architectures but may not generalize
to other model families or quantizations. Statistical power ( 𝑛=16)
detects large effects robustly but may miss subtle variations. Future
work should: (1) develop task-specific consistency benchmarks for
specialized financial workflows; (2) investigate fine-tuning effects
on output consistency in the 7B-8B parameter range; (3) extend to
emerging architectures including multimodal models, with over-
sight frameworks aligned with GAO guidance on AI use in financial
services [47].
Threats to model generalizabilityinclude the need to test our
framework on even larger models if they become available, though
our current results suggest architectural limitations may persist
regardless of scale. Our evaluation is currently limited to specific
model families and may not generalize to fundamentally different
architectures or training methodologies. We prioritized open-source
models for transparency, but this may not fully represent closed-
source model behavior.
Domain-Specific vs. General Applicability:Our framework’s
innovations—SEC structure-aware retrieval ordering, finance-calibrated
invariants, and FSB/CFTC-mapped audit trails—are purposefully
finance-specific. While themethodologygeneralizes to other reg-
ulated domains (healthcare HIPAA compliance, legal discovery),the specificparameters(5% materiality, SEC citation rules) encode
financial regulatory knowledge.
8.5 Can Larger Models Be Made Suitable for
Regulated Use?
While it is theoretically possible to constrain very large LLMs (70B-
120B) with engineered kernels or consensus mechanisms, our re-
sults suggest that nondeterminism at this scale is an emergent
property of model architecture. Even with 𝑇=0.0 and greedy de-
coding, GPT-OSS-120B achieved only 12.5% identical outputs. We
therefore conclude that, for regulated financial applications, smaller
7B-8B models are not only more efficient but uniquely positioned
to deliver the determinism required for compliance.
8.6 Fit-for-Purpose Model Selection in Financial
Services
While frontier models excel in creative and exploratory tasks, our
findings support a fit-for-purpose approach to model selection in
financial services. OpenAI’s release of GPT-OSS-Safeguard-20B [ 34]
exemplifies this trend—a smaller, specialized model designed for
safety-critical applications rather than general capability.3Recent
work demonstrates that small models (7B parameters) can match or
exceed larger models through test-time compute strategies, evalu-
ating multiple solution paths and selecting optimal outputs [ 26]. Fi-
nancial institutions should maintain dual strategies: (1) sandbox en-
vironments for experimenting with frontier models in non-critical
workflows where audit trails are optional, capturing innovation ben-
efits; and (2) deployment of smaller, deterministic models (7B-20B
parameters) for mission-critical workflows requiring regulatory
compliance. This dual-track approach balances innovation with
risk mitigation, allowing institutions to leverage AI advances while
maintaining audit-ready determinism where required.
9 Conclusion
This paper presents the first comprehensive framework for mea-
suring and mitigating LLM output drift in financial deployments,
revealing a key insight for model selection: architectural scale in-
versely correlates with regulatory compliance viability. While 𝑇=0.0
achieves perfect consistency in smaller, well-engineered models
(Granite-3-8B, Qwen2.5:7B), larger models like GPT-OSS-120B ex-
hibit 12.5% consistency regardless of configuration. This demon-
strates that architectural design and parameter efficiency—not scale
alone—determine compliance viability in financial applications.
The evidence strongly supports deploying smaller language mod-
els (7B-8B parameters) over larger counterparts for regulated fi-
nancial use cases. Models like Granite-3-8B and Qwen2.5:7B de-
liver 100% deterministic outputs essential for audit requirements,
while avoiding the consistency failures observed in 70B+ parameter
models. Our three-tier mitigation framework provides practical
guidance for institutions prioritizing regulatory compliance over
raw model capability. Our findings align with emerging SLM agent
paradigms [ 8], confirming that 7B–8B models optimize compliance
3Note: GPT-OSS-Safeguard-20B (20B parameters) represents OpenAI’s purpose-built
safety model, distinct from the GPT-OSS-120B (120B parameters) evaluated in our
experiments. Both exemplify the trend toward fit-for-purpose model design.
9

AI4F @ ACM ICAIF ’25, November 15–18, 2025, Singapore Khatchadourian and Franco
and efficiency in financial AI. Future work will extend these find-
ings to emerging architectures and deployment patterns, providing
the financial services industry with empirically grounded guidance
for AI adoption in regulated environments.
Acknowledgments
We thank the IBM watsonx.ai and TechZone teams for platform
access, and the Ollama and Qwen communities for model access.
We also thank Rica Craig (IBM AI/MLOps), Shashanka Ubaru and
Eda Kavlakoglu (IBM Research) for their guidance. We gratefully
acknowledge the late Jamiel Sheikh, who provided inspiration for
this work. This paper is dedicated to his memory.
A Appendix A: Seed Sweep @𝑇=0.0
Additional validation experiments with random seed variation (42,
123, 456, 789, 999) confirm that output consistency at 𝑇=0.0 remains
stable across different initialization parameters. All seed configura-
tions achieved 100% output consistency across our test corpus.
B Appendix B: Scheduling Jitter and Retrieval
Order
Testing different document retrieval orders and processing sched-
ules showed no impact on output consistency at 𝑇=0.0 . This finding
supports our conclusion that infrastructure-level determinism is
achievable in production financial environments.
C Appendix C: Regulatory Mapping
Detailed mapping of our framework to specific regulatory sections:
•FSB Section 3.2: Model Risk Management →Our Tier 3 Gover-
nance Controls
•BIS Article 15: Validation Requirements →Our Cross-Provider
Testing Protocol
•CFTC Section 4.1: Audit Trail Requirements →Our Tier 2 Log-
ging Framework
D Appendix D: Prompts
Sample prompt templates used in our experiments are shown in
Table 9:
Table 9:Sample prompt templates used in experiments
Task Prompt Template
RAG“What were JPMorgan’s net credit losses in 2023? Include a citation.”
“List Citigroup’s primary risk factors mentioned in the annual report. Include a citation.”
Summary“Client: Jane Doe, institutional. Needs a concise update on portfolio. Summarize neutrally.”
“Client: Acme Holdings. Provide 2-sentence update. Avoid PII, include disclaimer exactly.”
SQL“Compute totalamountacross all transactions.”
“Sumamountfor region =NAbetween 2025-01-01 and 2025-09-01.”
All prompts were versioned and logged in traces/*.jsonl files
with complete request-response pairs, enabling full reproducibility
and audit trail compliance for financial AI deployments.
E Appendix E: Detailed Model Classification
Tables
For reference, we provide the complete model tiered classification
table and task-specific drift analysis in Tables 10 and 11.Table 10:Model Tiered Classification for Financial AI Deployment
Model Parameters Consistency @𝑇=0.095% CI Deployment Tier Compliance Status
Tier 1: Production-Ready (Regulatory Compliant)
Granite-3-8B 8B 100% [80.6-100.0] Tier 1✓Full Compliance
Qwen2.5:7B 7B 100% [80.6-100.0] Tier 1✓Full Compliance
Tier 2: Limited Deployment (Task-Specific)
Mistral-Medium 40B+ 56–87% [35.3-97.8] Tier 2△Conditional Use
Llama-3.3-70B 70B 75% [50.9-91.3] Tier 2△Conditional Use
Tier 3: Non-Compliant (Unsuitable for Finance)
GPT-OSS-120B 120B 12.5% [3.5-36.0] Tier 3×Non-Compliant
Note:Classification based on 480 experimental runs (n=16 per condition) using Wilson
95% confidence intervals. Tier 1 models (7-8B parameters) achieve compliance-ready
determinism suitable for all audit-exposed financial processes. Tier 2 models (40-70B
parameters) show task-specific consistency appropriate for structured outputs only.
Tier 3 models unsuitable for audit-exposed deployments regardless of temperature or
decoding configuration.
Table 11:Drift patterns at𝑇=0.2by task type for Qwen2.5:7B
Task Identical Mean Drift Factual Drift Sensitivity
RAG 56.25% 0.081 0.000-0.375 High
SQL 100.00% 0.000 N/A None
Summary 100.00% 0.000 N/A None
Note:Task-specific drift analysis at𝑇=0.2demonstrates differential sensitivity to
temperature settings. RAG tasks show substantial drift (56.25% consistency, mean drift
0.081) with factual drift ranging 0.000-0.375, indicating high sensitivity to sampling
randomness. SQL generation and summarization maintain perfect consistency (100%),
confirming that structured outputs remain deterministic even with modest
temperature increases. These patterns inform deployment strategies: structured tasks
tolerate temperature >0, while RAG requires strict𝑇=0.0for compliance workflows.
F F Code and Data Availability
Code and artifacts are available at:
https://github.com/ibm-client-engineering/output-drift-financial-llms
To ensure reproducibility, use release v0.1.0 (commit c19dac5):
git clone https://github.com/ibm-client-engineering/output-drift-financial-llms
git checkout v0.1.0
The repository includes:
•Complete evaluation framework with cross-provider validation
•SEC 10-K test corpus and synthetic financial database
•480 experimental run traces with full reproducibility manifests
•Requirements: 16GB RAM, 8-core CPU (local); standard API ac-
cess (cloud)
See repository README for detailed setup instructions and repli-
cation steps.
References
[1]Alan Agresti and Brent A. Coull. 1998. Approximate is Better than “Exact” for
Interval Estimation of Binomial Proportions.The American Statistician52, 2
(1998), 119–126. doi:10.1080/00031305.1998.10480550
[2]Meta AI. 2025. Llama 3.3 70B Instruct — Model Card. https://huggingface.co/meta-
llama/Llama-3.3-70B-Instruct.
[3]Mistral AI. 2025. Mistral Medium (2505) — Model Card. https://docs.mistral.ai/
models/.
[4]Anthropic. 2025. A postmortem of three recent issues. https://www.anthropic.
com/engineering/a-postmortem-of-three-recent-issues
[5]Bruno Antunes and David R. C. Hill. 2021. Reproducibility, Replicability and
Repeatability: A survey of reproducible research with a focus on high perfor-
mance computing.Future Generation Computer Systems117 (2021), 95–107.
doi:10.1016/j.future.2020.11.016
[6]Jacob Baldwin et al .2024. Non-Determinism of “Deterministic” LLM Settings.
arXiv preprint arXiv:2408.04667(2024). https://arxiv.org/abs/2408.04667
10

LLM Output Drift: Cross-Provider Validation & Mitigation for Financial Workflows AI4F @ ACM ICAIF ’25, November 15–18, 2025, Singapore
[7]Bank for International Settlements. 2024. Regulating AI in the financial sector:
recent developments and main challenges. https://www.bis.org/fsi/publ/insight
s63.htm FSI Insights No. 63.
[8]Peter Belcak et al .2025. Small Language Models are the Future of Agentic AI.
arXiv preprint arXiv:2506.02153(2025). https://arxiv.org/abs/2506.02153
[9]Braintrust. 2025. Top 10 LLM Observability Tools: Complete Guide for 2025.
https://www.braintrust.dev/articles/top-10-llm-observability-tools-2025
[10] CFTC. 2024. Staff Letter No. 24-17: Digital Asset and AI Guidelines. https:
//www.cftc.gov/csl/24-17/download
[11] Commodity Futures Trading Commission, Technology Advisory Committee.
2024. Artificial Intelligence in Financial Markets. https://www.cftc.gov/media/1
0626/TAC_AIReport050224/download
[12] U.S. Congress. 2010. Dodd–Frank Wall Street Reform and Consumer Protection
Act. https://www.congress.gov/bill/111th-congress/house-bill/4173. Public Law
111–203.
[13] Digital Wealth News. 2025. AI & Finance News for the Week Ending 10/31/25.
https://dwealth.news/2025/10/ai-finance-news-for-the-week-ending-10-31-
25/
[14] Faker Team. 2025. Faker. https://github.com/joke2k/faker.
[15] Federal Reserve Board. 2024. Artificial Intelligence Program. https://www.fede
ralreserve.gov/ai.htm
[16] Financial Stability Board. 2017. Artificial Intelligence and Machine Learning in
Financial Services: Market Developments and Financial Stability Implications.
https://www.fsb.org/2017/11/artificial-intelligence-and-machine-learning-in-
financial-service/
[17] Financial Stability Board. 2024. The Financial Stability Implications of Artificial
Intelligence. https://www.fsb.org/2024/11/the-financial-stability-implications-
of-artificial-intelligence/
[18] Ronald A. Fisher. 1922. On the Interpretation of 𝜒2from Contingency Tables,
and the Calculation of P.Journal of the Royal Statistical Society85, 1 (1922), 87–94.
doi:10.2307/2340521
[19] Bank for International Settlements. 2025. Basel Framework (Basel III). https:
//www.bis.org/basel_framework/.
[20] Anthony Ha. 2025. Google pulls Gemma from AI Studio after Senator Blackburn
accuses model of defamation. TechCrunch. https://techcrunch.com/2025/11/02/
google-pulls-gemma-from-ai-studio-after-senator-blackburn-accuses-model-
of-defamation/
[21] IBM. 2024. Granite-3.0-8B-Instruct — Model Card. https://huggingface.co/ibm-
granite/granite-3.0-8b-instruct.
[22] IBM. 2024. watsonx.ai Foundation Model Catalog and Text Generation. https:
//www.ibm.com/products/watsonx-ai/foundation-models.
[23] Woosuk Kwon et al .2023. Efficient Memory Management for Large Language
Model Serving with PagedAttention. InProceedings of SOSP ’23. ACM, 611–626.
doi:10.1145/3600006.3613165
[24] Viet Dac Lai, Mahesh Nadendla, et al .2024. SEC-QA: A Systematic Evaluation
Corpus for Financial QA.arXiv preprint arXiv:2406.14394(2024). https://doi.org/
10.48550/arXiv.2406.14394
[25] Vladimir I. Levenshtein. 1966. Binary Codes Capable of Correcting Deletions,
Insertions, and Reversals.Soviet Physics Doklady10, 8 (1966), 707–710.
[26] Zhenyu Li et al .2025. Small Language Models Can Match or Exceed Large
Models Through Deep Thinking.arXiv preprint arXiv:2501.04519(2025). https:
//arxiv.org/abs/2501.04519
[27] Gene Marks. 2025. Small Business Technology Roundup: Microsoft Copilot Does
Not Improve Productivity and OpenAI Makes ChatGPT Project Free. https:
//www.forbes.com/sites/quickerbettertech/2025/09/14/small-business-
technology-roundup-microsoft-copilot-does-not-improve-productivity-and-
openai-makes-chatgpt-project-free/
[28] Morgan Stanley Research. 2025. AI Could Affect 90% of Occupations. https:
//www.morganstanley.com/insights/articles/ai-workplace-outlook-2H-2025/
[29] National Institute of Standards and Technology. 2024. Artificial Intelligence
Risk Management Framework: Generative Artificial Intelligence Profile. NIST AI
600-1. https://www.nist.gov/itl/ai-risk-management-framework[30] Brad O’Brien, Lisa Toth, and Son Huynh. 2024. AI risk management: are financial
services ready for AI regulation? Baringa Partners. https://www.baringa.com/
en/insights/de-risking-risk/ai-risk-management/
[31] Office of the Comptroller of the Currency. 2025. Interpretive Letter 1183: Su-
perseding Interpretive Letter 1179. https://www.occ.gov/topics/charters-and-
licensing/interpretations-and-actions/2025/int1183.pdf Supersedes IL 1179 on
AI use.
[32] Ollama. 2025. Ollama Documentation. https://github.com/ollama/ollama.
[33] OpenAI. 2025. Ghosts in the Codex Machine: Investigation into Reported Degra-
dation. https://docs.google.com/document/d/1fDJc1e0itJdh0MXMFJtkRiBcxGEF
tye6Xc6Ui7eMX4o
[34] OpenAI. 2025. GPT-OSS-Safeguard-20B: Purpose-Built Safety Model. https:
//huggingface.co/openai/gpt-oss-safeguard-20b.
[35] Operant AI. 2025. Shadow Escape: Zero-Click Agentic Attack via Model Context
Protocol. Technical Report. Referenced in Hackread, October 23, 2025. https:
//hackread.com/shadow-escape-0-click-attack-ai-assistants-risk/
[36] Jermy Prenio and Jeffery Yong. 2021. Humans Keeping AI in Check — Emerging
Regulatory Expectations in the Financial Sector. FSI Insights on policy imple-
mentation No. 35. https://www.bis.org/fsi/publ/insights35.pdf
[37] PyTorch Team. 2025. PyTorch Reproducibility. https://pytorch.org/docs/stable/
notes/randomness.html
[38] Red Hat. 2025. Run Qwen3-Next on vLLM with Red Hat AI: A step-by-step guide.
https://developers.redhat.com/articles/2025/09/12/run-qwen3-next-vllm-red-
hat-ai-step-step-guide
[39] Varshini Reddy et al .2024. DocFinQA: A Long-Context Financial Reasoning
Dataset. InProceedings of ACL 2024. doi:10.18653/v1/2024.acl-short.42
[40] Chufan Shi, Haoran Yang, Deng Cai, Zhisong Zhang, Yifan Wang, Yujiu Yang, and
Wai Lam. 2024. A Thorough Examination of Decoding Methods in the Era of LLMs.
arXiv preprint arXiv:2402.06925(Feb. 2024). https://arxiv.org/abs/2402.06925
[41] Qwen Team. 2025. Qwen2.5-7B-Instruct — Model Card. https://huggingface.co
/Qwen/Qwen2.5-7B-Instruct.
[42] TechCircle. 2025. It’s a Wrap: News This Week (Oct 25-31). https://www.techci
rcle.in/2025/10/31/it-s-a-wrap-news-this-week-oct-25-31
[43] The New Humanitarian. 2025. Model Drift: How Subtle Shifts in AI Responses
Could Undermine Crisis Response. https://www.thenewhumanitarian.org/opini
on/2025/10/08/model-drift-how-subtle-shifts-ai-responses-could-undermine-
crisis-response
[44] Thinking Machines Lab. 2025. Defeating Nondeterminism in LLM Inference.
https://thinkingmachines.ai/blog/defeating-nondeterminism-in-llm-inference/
[45] Trend Micro Research. 2025. When Tokenizers Drift: Hidden Costs and Security
Risks in LLM Deployments. https://www.trendmicro.com/vinfo/us/security/n
ews/cybercrime-and-digital-threats/when-tokenizers-drift-hidden-costs-and-
security-risks-in-llm-deployments
[46] European Union. 2014. Directive 2014/65/EU (MiFID II) on Markets in Financial
Instruments. https://eur-lex.europa.eu/legal-content/EN/TXT/?uri=celex:
32014L0065.
[47] U.S. Government Accountability Office. 2024. Artificial Intelligence: Use and
Oversight in Financial Services. GAO-25-107197. https://www.gao.gov/assets/g
ao-25-107197.pdf
[48] U.S. Securities and Exchange Commission. 2025. EDGAR Database: Annual
Reports (Form 10-K). https://www.sec.gov/edgar/search/. Companies include:
Citigroup Inc. (CIK 0000831001), JPMorgan Chase & Co. (CIK 0000019617), and
The Goldman Sachs Group, Inc. (CIK 0000886982). Filed 2025.
[49] vLLM Team. 2025. vLLM Documentation: Reproducibility and Determinism.
https://docs.vllm.ai/
[50] Edwin B. Wilson. 1927. Probable Inference, the Law of Succession, and Statistical
Inference.J. Amer. Statist. Assoc.22, 158 (1927), 209–212. doi:10.1080/01621459.1
927.10502953
[51] Qianqian Xie et al .2024. FinBen: A Holistic Financial Benchmark for Large
Language Models. InNeurIPS 2024 Datasets and Benchmarks Track. https:
//arxiv.org/abs/2402.12659 Datasets & Benchmarks Track; preliminary versions
available on arXiv.
11