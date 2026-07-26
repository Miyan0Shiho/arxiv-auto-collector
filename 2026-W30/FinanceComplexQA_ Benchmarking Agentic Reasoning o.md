# FinanceComplexQA: Benchmarking Agentic Reasoning on Industrial-grade Financial Documents

**Authors**: Xianfu Cheng, Shiwei Zhang, Jiyu Zhao, Jian Yang, Xinyuan Wang, Ming Zhou, Weixiao Zhou, Xiangyuan Guan, Xiang Li, Zhenhe Wu, Ziyi Ni, Zhoujun Li, Bingjing Xu

**Published**: 2026-07-21 16:08:27

**PDF URL**: [https://arxiv.org/pdf/2607.19238v1](https://arxiv.org/pdf/2607.19238v1)

## Abstract
Agentic Reasoning has become a transformative force in financial analysis due to its ability to integrate large-scale information and generate reliable and accurate content. However, when handling complex real-world problems, different agents still show significant performance variation. In this work, we design Finance-LaTeX SKILL, a skill for synthesizing financial documents with complex layouts based on expert knowledge. Using an agent workflow built on this skill, we generate 2,000 professional financial documents along with 6,000 high-quality question-answer pairs. To evaluate the overall capability of agents, we introduce FinanceComplexQA, a comprehensive open-ended generation benchmark for financial documents that closely resembles real-world scenarios. It contains 2,026 deep research tasks targeting 1009 financial documents. FinanceComplexQA has 8 key features: bilingual support; coverage of six mainstream scenarios and seven tasks; expert-level document reasoning questions; deep research of complex layouts; relatively stable and permanent reference answers; and precise evaluation through an Agent-as-a-Judge with multiple evaluation metrics. Using FinanceComplexQA, we conduct a comprehensive evaluation of leading RAG systems and agentic reasoning tools for financial document QA. Through identifying and analyzing failure cases, we provide an in-depth study of their capabilities in numerical computation, multi-hop reasoning, content summarization, and industry analysis.

## Full Text


<!-- PDF content starts -->

FinanceComplexQA: Benchmarking Agentic Reasoning on
Industrial-grade Financial Documents
Xianfu Cheng1, Shiwei Zhang2, Jiyu Zhao3, Jian Yang1†, Xinyuan Wang3, Ming Zhou4
Weixiao Zhou1,Xiangyuan Guan1,Xiang Li1,Zhenhe Wu1,Ziyi Ni3,Zhoujun Li1,5,Bingjing Xu2
1Beihang University;2Microsoft, China;3Multilingual-Multimodal-NLP;
4Langboat Technology, Beijing, China;5Shenzhen Intelligent Strong Technology Co.,Ltd.
{buaacxf,jiayang,lizj}@buaa.edu.cn,shiweizhang@microsoft.com
Abstract
Agentic Reasoning has become a transforma-
tive force in financial analysis due to its abil-
ity to integrate large-scale information and
generate reliable and accurate content. How-
ever, when handling complex real-world prob-
lems, different agents still show significant
performance variation. In this work, we de-
signFinance-LaTeXSKILL, a skill for syn-
thesizing financial documents with complex
layouts based on expert knowledge. Using
an agent workflow built on this skill, we gen-
erate 2,000 professional financial documents
along with 6,000 high-quality question-answer
pairs. To evaluate the overall capability of
agents, we introduceFinanceComplexQA, a
comprehensive open-ended generation bench-
mark for financial documents that closely re-
sembles real-world scenarios. It contains 2,026
deep research tasks targeting 1009 financial
documents. FinanceComplexQA has 8 key fea-
tures: bilingual support; coverage of six main-
stream scenarios and seven tasks; expert-level
document reasoning questions; deep research
of complex layouts; relatively stable and per-
manent reference answers; and precise evalua-
tion through an Agent-as-a-Judge with multiple
evaluation metrics. Using FinanceComplexQA,
we conduct a comprehensive evaluation of lead-
ing RAG systems and agentic reasoning tools
for financial document QA. Through identify-
ing and analyzing failure cases, we provide an
in-depth study of their capabilities in numeri-
cal computation, multi-hop reasoning, content
summarization, and industry analysis.
1 Introduction
In recent years, large language models
(LLMs) (DeepSeek-AI et al., 2025; Touvron
et al., 2023; Yang et al., 2024, 2025, 2026a,b)
have achieved remarkable progress in natural lan-
guage understanding, reasoning, and multimodal
generation, enabling their adoption in high-stakes
domains such as finance, law, and healthcare.Among these, finance presents unique challenges
due to its strong requirements for numerical
precision, domain expertise, and factual reliability.
Unlike open-domain question answering, financial
analysis often requires multi-step reasoning over
long documents, integration of structured and
unstructured information, and strict consistency
with objective evidence. Even minor factual or
reasoning errors may lead to flawed investment
decisions or risk assessments, making finance one
of the most demanding scenarios for reliable LLM
deployment.
Recent advances such as retrieval-augmented
generation (RAG) systems (Lewis et al., 2020b;
Guo et al., 2024; Xiang et al., 2025; Xiao
et al., 2025), Model Context Protocol (MCP)
tools (Schick et al., 2023; Li, 2025; Liu et al.,
2026), and Agentic Reasoning tools (Anthropic,
2026; OpenAI, 2026) have improved LLM perfor-
mance on knowledge-intensive tasks by enabling
external retrieval, dynamic tool invocation, and
multi-stage planning. However, evaluating such
systems in real financial scenarios remains diffi-
cult. Existing financial question-answering (QA)
benchmarks mainly focus on narrow tasks such
as table QA (Wu et al., 2025b, 2026; Shu et al.,
2026), numerical extraction (Cheng et al., 2025a;
Wu et al., 2025a), short-context fact retrieval (Wei
et al., 2024; He et al., 2025; Cheng et al., 2025b),
or Summarization (Zhou et al., 2023, 2025, 2026).
They rarely capture the complex reasoning process
required in real financial research, where analysts
must synthesize evidence across documents, tables,
charts, and domain knowledge. Moreover, many
existing benchmarks rely on synthetic questions or
template-based construction, limiting realism and
reasoning depth. Current evaluation protocols also
struggle to provide stable and consistent scoring
for open-ended analytical answers.
To address these limitations, we introduceFi-
nanceComplexQA, a large-scale benchmark for
1
arXiv:2607.19238v1  [cs.CE]  21 Jul 2026

open-ended complex question answering over fi-
nancial documents. FinanceComplexQA contains
2026 expert-level questions built from 1009 real-
world financial rich-text documents, covering 8
financial subdomains and 9 representative finan-
cial research tasks, including numerical analysis,
multi-hop reasoning, causal inference, industry
comparison, and cross-layout synthesis. The bench-
mark is designed to evaluate whether modern LLM-
based systems can perform document understand-
ing, deep reasoning, and analytical synthesis at a
level closer to that of professional financial ana-
lysts.
Before the benchmark, we proposeFinance-
LaTeXSKILL, a multi-agent framework for scal-
able financial QA data synthesis. The framework
combines expert knowledge, automated document
generation, terminology constraints, and financial
logic verification to generate high-quality finan-
cial documents and question-answer pairs. Using
this framework, we synthesize an additional 2,000
financial documents and 6,000 QA samples, pro-
viding a scalable and automated path for bench-
mark construction and future benchmark expansion.
FinanceComplexQA is built around four core de-
sign principles. First, we introduce a dual-context
reasoning paradigm, where solving each question
requires both explicit document evidence and im-
plicit financial domain knowledge. Models must
retrieve relevant evidence and combine it with fi-
nancial logic and industry knowledge for accurate
reasoning. Second, we emphasize cross-layout rea-
soning: each question requires a joint understand-
ing of textual passages, tables, forms, and contents,
forcing systems to aggregate evidence across het-
erogeneous document elements. Third, the bench-
mark covers diverse financial tasks and scenarios,
enabling fine-grained evaluation of different rea-
soning capabilities. Fourth, to ensure long-term
usability, all questions are grounded in relatively
stable and verifiable financial facts and are evalu-
ated with a unified framework that jointly measures
semantic correctness, numerical accuracy, and rea-
soning completeness through an Agent-as-a-Judge
pipeline.
We evaluate 2 mainstream RAG/MCP systems
and 2 agentic reasoning systems on FinanceCom-
plexQA. Results reveal a substantial performance
gap between current state-of-the-art systems and
real-world financial reasoning requirements. Even
advanced systems struggle with long-chain numer-
ical reasoning, cross-layout evidence fusion, andindustry-level analytical synthesis. Detailed error
analysis further exposes critical weaknesses in rea-
soning, planning, evidence utilization, and factual
consistency, highlighting key directions for future
research in reliable financial AI agents.
Our contributions are summarized as follows: (1)
We propose Finance-LaTeX SKILL, a multi-agent
framework for scalable generation of high-quality
financial documents with QA data; (2) We present
FinanceComplexQA, a large-scale benchmark for
complex financial document question answering
that covers diverse tasks, domains, and multi-step
reasoning scenarios; (3) We design a challenging
benchmark construction methodology centered on
dual-context reasoning and cross-layout evidence
aggregation, enabling more realistic evaluation of
financial analytical reasoning; (4) We conduct a sys-
tematic evaluation of state-of-the-art RAG, MCP,
and agentic systems, revealing major bottlenecks in
complex financial reasoning and providing insights
for future financial LLM research.
2 Finance-LaTeX SKILL
2.1 Design Motivation
Finance-LaTeX is designed as the document-
generation engine behind the benchmark construc-
tion process illustrated in Figure 1. Its goal is to
synthesize professional financial documents and
question-answer pairs that resemble realistic ana-
lyst workloads rather than isolated text snippets.
To achieve this, the skill combines expert finan-
cial knowledge, evidence planning, layout-aware
LaTeX writing, and multi-stage verification. The
generated data are used as a scalable development
source for stress testing, prompt iteration, and
benchmark expansion, while the held-out evalu-
ation benchmark remains grounded in curated fi-
nancial documents.
The skill is motivated by two gaps in existing fi-
nancial QA data. First, many datasets contain short
excerpts or simplified tables, while real financial
analysis often requires reading long reports with
paragraphs, tables, forms, captions, and special lay-
outs. Second, open-ended answers require more
than factual extraction: they must connect evidence
to financial logic, preserve units and time periods,
and avoid unsupported recommendations. Finance-
LaTeX therefore treats a document as a structured
evidence environment and makes every generated
QA pair pass explicit consistency and solvability
checks.
2

Benchmark LNG #Docs #QAs PDF parser Long doc Cross-Layout Open-ended Metrics
FinQA (Chen et al., 2021) EN 2789 pages 1147 × × ×✓Recall, EA, PA
ConvFinQA (Chen et al., 2022) EN 434 pages 1568 × ×✓× EA, PA
DocFinQA (Reddy et al., 2024) EN 100 922✓ ✓ ✓× HR, ACC
TAT-QA (Zhu et al., 2021) EN 275 context 1655 × ×✓× EM, F1 score
FinanceBench (Islam et al., 2023) EN 150 context 150 × ×✓ ✓CA, IA, F2A
officeQA (Databricks, 2024) EN 246 246✓ ✓ ✓× ACC, ARE
officeQA-Pro (Databricks, 2025) EN 696 133✓ ✓× × ACC, ARE
BizFinBench (Lu et al., 2025) EN/CN 6500 context 6500 × × ×✓ACC
BizFinBench v2 (Guo et al., 2026) EN/CN 29578 context 29578 ×✓×✓ACC
GDPval (Patwardhan et al., 2025) EN 423 220✓ ✓ ✓ ✓WO, WT
FinanceAgent (Bigeard et al., 2025) EN tool env. 200 × × ×✓CBA, ACC
FinanceComplexQA (Ours) EN/CN 1009 2026✓ ✓ ✓ ✓ACC, ROU, FS, Cov
Table 1: Comparison between FinanceComplexQA and representative financial and enterprise QA benchmarks
in terms of language coverage, scale, PDF parsing, long-document support, cross-layout reasoning, open-ended
answering, and evaluation metrics. Compared with prior datasets, FinanceComplexQA jointly targets bilingual
expert questions over parsed financial documents and evaluates open-ended analytical answers with accuracy,
overlap, faithfulness, and coverage metrics.
2.2 Expert Knowledge and Evidence Planning
The workflow begins with corpus collection and
expert knowledge acquisition. For each candidate
document, an agent samples a financial domain,
target audience, document type, terminology con-
straints, and reasoning targets. The selected do-
main concepts include stable accounting relations,
regulatory terms, market-analysis concepts, and
business-process facts that are unlikely to become
obsolete quickly. This design follows the left part
of Figure 1, where corpus evidence and expert
knowledge are combined before document genera-
tion.
Before writing a document, the skill constructs
an evidence plan. The plan specifies which facts
should appear in paragraphs, which quantities
should appear in tables, which assumptions should
be expressed in captions or notes, and which pieces
of evidence should later support questions. Evi-
dence is also classified by reasoning function, such
as direct lookup, numerical comparison, multi-hop
inference, summary evidence, or planning con-
straint. This makes the generated documents suit-
able for testing both explicit retrieval and implicit
financial reasoning.
2.3 Layout-Aware LaTeX Generation
The generation stage converts the evidence plan
into a LaTeX document with realistic financial
structure. Instead of producing plain text, the agent
writes sections, paragraphs, tables, special layout
blocks, formulas, captions, and table-adjacent de-
scriptions. Numeric cells are generated together
with row labels, column labels, periods, units, and
entity scopes, so that later questions can require the
system to bind each number to its correct context.The skill intentionally creates cross-layout de-
pendencies. For example, a paragraph may de-
scribe a change in operating strategy, a table may
give the corresponding revenue and margin values,
and a note may define the scope of consolidation.
A valid answer must connect these elements rather
than relying on a single sentence. This layout-
aware design is central to FinanceComplexQA, be-
cause Figure 1 emphasizes special layouts and para-
graph evidence as inputs to benchmark construc-
tion.
2.4 QA Pair Generation and Verification
After document generation, the workflow creates
candidate question-answer pairs from the planned
evidence. Each question is paired with required
evidence units, a reference answer, and reasoning
notes. The question generator rejects items that
can be answered by copying a single span, that
depend on volatile external facts, or that ask for
unsupported speculation. The answer generator
then writes analytical responses that include con-
clusions, evidence, calculations, and caveats.
The generated QA pairs are checked against the
criteria shown in Figure 1: answers should be con-
sistent with the questions and reference documents,
and questions should be solvable from the refer-
ence documents. Verification is performed through
multiple channels. LLM-based checks inspect co-
herence, completeness, and financial logic; web
search and MCP-style tools verify stable public
facts when needed; and harness tests check whether
the evidence-answer relation is reproducible. If a
document or answer fails these checks, the work-
flow returns to a check-and-correct stage before the
sample can enter the data pool.
3

Figure 1: Overview of the FinanceComplexQA construction pipeline. The workflow collects financial corpora,
incorporates expert knowledge, generates layout-rich documents with Finance-LaTeX, verifies question-answer
pairs through LLM, Web/MCP, and human checks, and organizes the resulting benchmark by financial domain and
judge-oriented evaluation metrics.
2.5 Quality Control and Data Usage
The final stage applies quality control at both doc-
ument and QA levels. The checks include inde-
pendent testing, cross evaluation, expert validation,
criteria verification, and multi-source provision,
matching the quality-control block in Figure 1. At
the document level, we remove samples whose lay-
out is broken, whose tables lose headers or units,
or whose financial relations are inconsistent. At
the QA level, we remove samples with ambiguous
evidence, irreproducible calculations, incomplete
reference answers, or weak reasoning depth.
Using this workflow, we generate 2,000 profes-
sional financial documents and 6,000 high-quality
QA pairs. These synthetic data are not mixed into
the held-out benchmark for reporting model per-
formance. Instead, they support development and
ablation studies, provide controlled cases for test-
ing retrieval and agent behavior, and help expose
failure modes before systems are evaluated on Fi-
nanceComplexQA.
3 FinanceComplexQA Pipeline
3.1 Overview
FinanceComplexQA is a bilingual benchmark for
open-ended generation over industrial-grade finan-
cial documents. As summarized in Figure 1, the
pipeline combines real financial corpora, expertevidence extraction, Finance-LaTeX-assisted doc-
ument and QA construction, LLM/web/MCP ver-
ification, human review, and judge-oriented eval-
uation. The resulting benchmark contains 2,026
deep research tasks over 1009 financial documents,
with 1,013 Chinese questions and 1,013 English
questions.
The benchmark is designed to test whether an
agent can perform the full reasoning loop expected
in financial analysis: locate evidence, preserve lay-
out context, apply domain knowledge, compute or
compare quantities, synthesize an answer, and re-
main faithful to the source. This is why the pipeline
records not only questions and reference answers,
but also evidence units, reasoning notes, scenario
labels, task labels, and evaluation dimensions.
3.2 Corpus and Financial Scenarios
FinanceComplexQA is constructed from profes-
sional financial documents rather than isolated ar-
ticle fragments. The corpus includes corporate fi-
nancial reports, investment strategy reports, mar-
ket trend analysis reports, FinTech research re-
ports, bank financial statements, customer service
records, supervision and compliance audit materi-
als, and government fiscal bulletins. The metadata
of these corpora are respectively derived from Con-
vFinQA (Chen et al., 2022), FinanceBench-test (Is-
lam et al., 2023), officeQA-Pro (Databricks, 2025),
4

BizFinBench v2 (Guo et al., 2026) (Counterfactual
Inference, Anomaly Information Tracing), and the
actual industrial financial knowledge base.
The financial domains shown in Figure 1, such as
commercial banking, corporate reports, investment
banking, stock-market analysis, customer service,
compliance audit, commercial insurance, Sci-Tech
innovation, and official bulletins, are normalized
into the scenario labels used in Table 2. These sce-
narios were selected because they require different
evidence habits. Corporate reports emphasize ac-
counting relations; investment reports require mar-
ket and industry synthesis; compliance documents
require careful rule interpretation; customer ser-
vice records stress event reconstruction and intent
understanding; and fiscal bulletins require linking
policy descriptions to amounts, periods, and pro-
gram structure.
The documents are parsed into layout-aware
units, including paragraphs, tables, forms, captions,
section titles, and table-adjacent descriptions. We
retain this structural information because losing
page and layout context often removes the decisive
evidence required by financial reasoning.
3.3 Task Design
The benchmark questions are written to represent
real analytical requests rather than template slots.
We use nine task labels. Chinese questions include
comparison (CMP), implicit reasoning (IR), knowl-
edge query (KQ), multi-hop reasoning (MR), sum-
marization (SUM), and planning (PLA). English
questions include numerical comparison (NC), im-
plicit reasoning (IR), multi-hop judgment (MJ), ex-
plicit reasoning (ER), summarization (SUM), and
planning (PLA). The labels are not meant to iso-
late mutually exclusive cognitive skills. Instead,
they provide a primary lens for evaluation and error
analysis.
Each question is checked against three design
requirements. First, it must have a stable reference
answer based on relatively persistent financial facts
or document-internal evidence. Second, it must
require more than one trivial retrieval step. Third,
it must be answerable by a careful analyst using the
supplied documents and common financial knowl-
edge. Questions that depend on live market prices,
rapidly changing policies, or unsupported specula-
tion are removed or rewritten.3.4 Document Parsing and Evidence Units
Each document is converted into a hierarchy of
evidence units before question writing. The hi-
erarchy preserves page order, section boundaries,
paragraph spans, table row and column headers,
captions, and nearby explanatory text. We treat ta-
bles as structured evidence rather than plain strings:
numeric cells are stored together with their row
labels, column labels, units, and period informa-
tion. This representation is important for financial
reasoning because the same number can have dif-
ferent meanings depending on whether it refers to
a quarter, a fiscal year, a consolidated entity, or a
business segment.
For long documents, evidence units are grouped
into page-local blocks and document-level topic
blocks. Page-local blocks support layout-sensitive
retrieval, while topic blocks support cross-page rea-
soning. During annotation, question writers can
mark which units are necessary, optional, or mis-
leading. This lets us distinguish a system that re-
trieves the right page from one that actually uses
the decisive evidence.
3.5 Annotation Schema
For each example, we record the question text,
language, scenario label, task label, required evi-
dence units, reference answer, and reasoning notes.
The reasoning notes include intermediate quanti-
ties, comparison targets, and domain assumptions
when they are needed to reproduce the answer. For
cross-layout questions, the schema also records the
evidence relation, such as text-to-table, table-to-
chart, or multi-page synthesis. This metadata is
used only for evaluation and analysis; models do
not receive the gold evidence relation at inference
time.
The schema supports fine-grained error diagno-
sis. If a model retrieves all required evidence but
reaches the wrong conclusion, the failure is classi-
fied as a reasoning or calculation error. If it misses
a table, chart, or footnote that the reference depends
on, the failure is classified as evidence omission.
This distinction is useful because retrieval failures
and reasoning failures require different system im-
provements.
3.6 Dual-Context and Cross-Layout
Reasoning
The most important property of FinanceCom-
plexQA is dual-context reasoning. In ordinary doc-
5

StatisticsCountStatisticsCount
#QAs #Docs #QAs #Docs
Data2026 1009 8 types of Scene Categories
- Chinsne (CN) 1013 625 - FinTech Research Report (FR), CN&EN 134+157 65+25
- English (EN) 1013 384 - Corporate Financial Report (CF), CN&EN 163+253 141+58
9 types of Task Categories - Investment Strategy Report (IS), CN&EN 246+276 138+136
- Planning (PLA), CN&EN 137+101 74+61 - Market Trend Analysis Report (MTA), CN&EN 91+192 43+30
- Summary (SUM), CN&EN 154+243 154+159 - Bank Financial Statements (BFS), CN 139 85
- Implicit Reasoning (IR), CN&EN 163+220 141+76 - Customer Service Records (CSR), CN 140 80
- Comparison (CMP), CN 190 119 - Supervision and Compliance Audit (SCA), CN 100 73
- Knowledge Query (KQ), CN 141 81 - Government Fiscal Bulletin (GFB), EN 135 135
- MultiHop Reasoning (MR), CN 228 158 (CN/EN) Words length of Question Golden
- Explicit Reasoning (ER), EN 195 167 - Maximum length 255/1141 5671/2623
- MultiHop Judgment (MJ), EN 141 76 - Minimum length 5/51 5/1
- Numerical Comparison (NC), EN 113 47 - Average length 63/286 283/366
Table 2: Dataset statistics of FinanceComplexQA, including the bilingual split, QA and document counts by task
category and financial scene, and question/reference-answer length distributions. Counts joined by “+” denote
Chinese and English subsets, respectively.
LNG Group Pages (avg.) Length (avg.)
FR 18.23 12127
CF 9.43 7359
IS22.3315191
MTA 21.4715803
BFS 19.87 12276
CSR 19.90 11606
SCA 19.99 12736CN
A VG18.01 12040
FR 17.72 30911
CF 14.78 24943
IS 14.23 27519
MTA21.30 33498
GFB 9.92 16978EN
A VG13.58 24112
Total16.33 16493
Table 3: Average document length by language and fi-
nancial scene in FinanceComplexQA. The table reports
mean page counts and mean token/word lengths for each
Chinese and English scenario group, showing that the
benchmark is dominated by long, layout-rich financial
documents.
ument QA, the relevant sentence may contain most
of the answer. In financial research, a sentence usu-
ally becomes meaningful only after it is interpreted
through accounting definitions, market structure,
risk logic, or regulatory background. For example,
a table may show that revenue grew while oper-
ating cash flow declined; the answer requires not
only extracting both numbers, but also explaining
why the divergence matters.
We therefore design questions that combine ex-
plicit and implicit contexts. Explicit context in-
cludes document spans, table cells, chart labels,
and stated assumptions. Implicit context includes
domain knowledge such as margin interpretation,debt pressure, sector cyclicality, accounting rela-
tions, or the difference between nominal growth
and real operating improvement. The reference
answer records both types of evidence, allowing
evaluators to penalize answers that are fluent but
incomplete.
Cross-layout reasoning is enforced by construc-
tion. A typical question may require retrieving a
management discussion paragraph, matching it to a
financial table, and using a chart or trend statement
to determine whether the conclusion is supported.
This is difficult for systems that flatten documents
into independent chunks. It is also difficult for
agents that retrieve evidence correctly but fail to
preserve the relationship between a row header, a
time column, and a surrounding narrative explana-
tion.
3.7 Reference Answers and Evaluation
Reference answers are written as analytical re-
sponses rather than minimal spans. They include
the conclusion, the key evidence, relevant calcu-
lations, and caveats. For numerical answers, the
reference records the formula or comparison logic
whenever possible. For planning and summary
tasks, the reference identifies mandatory coverage
points so that a shorter answer can still be credited
if it includes the essential reasoning.
We evaluate model outputs with an Agent-as-a-
Judge protocol. The judge receives the question,
reference answer, retrieved evidence where avail-
able, and the candidate answer. It assigns task-
specific scores for accuracy, semantic alignment,
numerical correctness, evidence coverage, faithful-
ness, and completeness. For planning and sum-
6

Figure 2: Task taxonomy and representative benchmark cases. The figure summarizes the nine Chinese and English
task labels, their related financial domains, brief reasoning requirements, and example questions, ranging from
implicit reasoning and planning to multi-hop judgment and numerical comparison.
marization tasks, coverage and faithfulness receive
higher weight. For numerical comparison and ex-
plicit reasoning, exact quantities and units receive
higher weight. We also record cost and latency,
because deployment in financial workflows is con-
strained by both quality and resource use.
3.8 Quality Control
The benchmark is filtered through both automatic
and expert-oriented checks. At the document level,
we remove files whose parsing output loses section
hierarchy, breaks table headers, or produces am-
biguous chart descriptions. At the question level,
we reject items that can be answered by copying a
single span, that depend on transient market data,
or that ask for subjective investment recommen-
dations without document support. At the answer
level, we check whether mandatory evidence points
are present and whether numeric operations are re-
producible from the cited values.
We also maintain a difficulty balance. If a sub-
set contains too many direct lookup questions, ad-
ditional multi-hop or cross-layout questions are
sampled from the same scenario. Conversely, if
a question requires private external knowledge or
speculative assumptions, it is rewritten to expose
enough evidence in the reference documents. This
keeps the benchmark difficult while preserving an-
swerability. The goal is not to make every examplemaximally hard, but to reflect the range of tasks
that a financial analyst would face in a realistic
research workflow.
4 Experiments
4.1 Systems
We evaluate systems that represent three common
ways of using LLMs for financial document QA.
The first is a lightweight retrieval baseline, repre-
sented by LightRAG (Guo et al., 2024). The second
is a layout-aware retrieval pipeline, represented by
PageIndex (Zhang et al., 2025), which is designed
to preserve page-level and table-adjacent struc-
ture. The third is an agentic setting, represented by
Codex-style and Claude Code-style agents (Ope-
nAI, 2024; Anthropic, 2024), where the system can
plan, call tools, inspect intermediate evidence, and
revise an answer before final generation.
We evaluate 7 representative LLMs in total, com-
prising 4 closed-source and 3 open-source models,
which provide a diverse evaluation of model capa-
bilities across different architectures and training
paradigms. The closed-source models include GPT-
5.4 and GPT-5.51, Claude-5-Sonnet2, Qwen3.7-
1https://platform.openai.com/docs/models#
gpt-5.4;#gpt-5.5
2https://www.anthropic.com/news/
claude-sonnet-5
7

Dimension Judgment criterion
Accuracy The final conclusion matches the
reference answer.
Numeric correctness Calculations preserve formula, unit,
period, and sign.
Evidence coverage Required text, table, and chart
evidence is used.
Faithfulness Claims are supported by the
provided documents.
Completeness The answer includes required
caveats and reasoning steps.
Cost Token usage and tool calls remain
practical for deployment.
Table 4: Evaluation dimensions used by the Agent-as-a-
Judge protocol.
plus3. The open-source models cover 2 frame-
works, including Qwen3.5-flash and Qwen3.5-
plus4, DeepSeek-v4-flash5.
Each system is paired with either closed-source
or open-source foundation models, as shown in
Tables 5–7. All systems receive the same docu-
ment collection and question set. We keep each
system’s native retrieval or agent loop, but require
a comparable final answer format. For every an-
swer, we record the question language, scenario,
task label, judge scores, token usage, and latency.
This design lets us compare overall quality while
also diagnosing whether a system fails because of
retrieval, calculation, answer synthesis, or cost.
4.2 Metrics
Table 4 summarizes the Agent-as-a-Judge proto-
col. Accuracy measures whether the conclusion
matches the reference answer. Numeric correct-
ness checks whether formulas, units, periods, signs,
and entity scopes are preserved. Evidence cover-
age measures whether required paragraphs, tables,
charts, and captions are used. Faithfulness penal-
izes claims that are not supported by the provided
documents. Completeness captures whether the an-
swer includes required caveats and reasoning steps.
Cost records whether the token usage and tool calls
are practical for deployment.
The task tables use metric combinations that
match the answer type. Accuracy (ACC) is used
across tasks as the primary correctness signal.
ROUGE-style overlap (ROU) is reported for re-
trieval and reasoning tasks as a lexical-alignment
diagnostic, but it is not treated as sufficient for
3https://dashscope.aliyuncs.com#/model-market
4https://huggingface.co/Qwen#Qwen3.5-35B-A3B;
#Qwen3.5-397B-A17B
5https://platform.deepseek.com/correctness. Coverage (Cov) is emphasized for
summarization and planning, where an answer may
be fluent but omit required evidence. Faithfulness
(FS) is reported for planning tasks, where generic
or unsupported recommendations are a major risk.
The aggregate averages in Table 7 are computed
over accuracy-style scores, while the detailed task
table preserves the metric-level differences needed
for diagnosis.
4.3 Main Findings
Tables 5–7 show four main patterns. First, layout-
aware retrieval is a strong baseline. PageIndex
consistently improves over LightRAG in overall
scene scores, even though the two rows use differ-
ent core models. This suggests that preserving page
structure and table context is important for Finance-
ComplexQA, especially when the answer depends
on long-document or cross-layout evidence.
Second, agentic systems often achieve the
strongest results, but the winning configuration de-
pends on the view. In Table 5, Claude Code with
Sonnet 5 has the best overall Chinese scene score
(76.01) and the best overall English scene score
(69.39). In Table 7, Codex with GPT-5.5 has the
strongest aggregate task averages among closed-
source settings, with 60.63 on Chinese and 62.97
on English. Among open-source settings, Codex
with DeepSeek-V4-Flash is the strongest aggregate
configuration, reaching 60.44 on Chinese and 61.55
on English. These results indicate that both the or-
chestration strategy and the core model matter.
Third, no system dominates every task or sce-
nario. PageIndex is especially competitive on
knowledge query, multi-hop reasoning, numerical
comparison, multi-hop judgment, and planning ac-
curacy. Codex-style agents are strong on implicit
reasoning, explicit reasoning, and aggregate av-
erages. Claude Code-style agents are strong on
several Chinese scene categories and on faithful-
ness and coverage in planning. This fragmentation
supports the central claim of the benchmark: finan-
cial QA requires several capabilities that do not
always improve together.
Fourth, quality must be considered together with
cost. Table 7 shows that LightRAG and PageIn-
dex use far fewer tokens and less time than agentic
systems. LightRAG uses 12.8k tokens and 8.6 sec-
onds per question on average, while PageIndex uses
15.6k tokens and 11.9 seconds. Agentic systems
often require 31.4k–45.8k tokens and 22.8–50.6
seconds. The quality gains therefore come with a
8

Agent Core ModelChinese Financial Scenes English Financial Scenes
FR CF IS MTA BFS CSR SCA OA FR CF IS MTA GFB OA
Closed-Source Large Language Models
LightRAG gpt-5.5 80.59 59.20 41.46 55.49 77.85 66.07 74.50 62.42 57.64 76.87 53.80 61.19 41.48 59.92
PageIndex gpt-5.4 86.92 57.05 64.78 63.41 85.56 70.15 85.35 72.15 67.6283.33 65.58 67.12 37.69 66.90
gpt-5.5 79.83 72.12 63.87 55.15 72.37 68.93 76.59 69.29 70.38 76.48 64.49 64.5870.37 69.20Codexqwen3.7-plus 88.05 69.32 65.04 65.38 84.39 72.14 85.50 74.48 64.33 77.57 66.30 65.36 62.22 68.08
sonnet 5 91.04 65.6468.69 67.03 87.41 73.21 87.0076.01 65.60 80.63 65.9467.96 61.8569.39Claude Codeqwen3.7-plus 89.83 69.62 67.06 66.87 85.65 72.3089.13 75.93 68.14 75.4267.05 64.08 62.40 68.08
Open-Source Large Language Models
qwen3.5-flash 81.71 55.21 63.61 62.63 76.97 65.35 79.50 68.21 63.69 73.71 55.61 62.50 49.25 61.84
qwen3.5-plus 81.34 52.45 59.34 63.73 80.21 63.21 74.00 66.38 65.28 75.09 34.05 63.54 47.40 56.51 Codex
deepseek-v4-flash 86.94 72.69 64.17 60.43 83.81 71.07 80.80 74.66 69.74 76.87 64.80 63.80 66.79 69.33
qwen3.5-flash 88.72 54.43 65.44 64.60 83.21 72.62 82.32 71.82 64.10 77.09 57.42 63.87 48.86 63.58
qwen3.5-plus 88.80 68.75 62.22 61.18 83.81 71.42 82.77 73.24 68.58 76.20 51.63 65.64 59.30 64.49Claude Codedeepseek-v4-flash 85.8273.14 57.11 61.79 82.01 67.66 85.00 71.58 70.38 77.86 56.02 66.40 61.36 66.40
Table 5: Scenario-level answer-evaluation results judged by GPT-5-mini. Scores are reported for RAG, page-indexed,
and agentic systems across Chinese and English financial scenes; OA denotes the overall average within each
language block, and underlined values highlight the strongest results in the corresponding scenario.
Chinese Financial Tasks
Agent Core ModelCMP IR KQ MR SUM PLA
ACC ROU ACC ROU ACC ROU ACC ROU ACC Cov ACC FS Cov
Closed-Source Large Language Models
LightRAG gpt-5.5 55.86 23.46 54.67 42.72 58.20 32.88 56.11 19.17 44.87 31.93 51.43 58.36 49.57
PageIndex gpt-5.4 56.63 26.47 54.29 56.41 63.19 35.77 65.19 27.93 58.36 49.43 56.53 67.71 61.56
gpt-5.5 56.66 27.24 73.33 71.17 62.0937.80 61.02 27.00 60.08 47.58 50.58 83.54 32.77Codexqwen3.7-plus 55.95 23.09 69.63 70.24 61.67 32.96 58.47 24.44 60.03 57.61 54.15 92.31 64.90
sonnet 5 55.53 22.42 61.10 58.36 62.63 32.43 55.73 16.92 60.29 60.57 53.7196.18 68.36Claude Codeqwen3.7-plus 50.42 20.06 60.84 55.99 60.22 31.38 55.55 22.92 51.87 45.63 51.92 84.59 60.85
Open-Source Large Language Models
qwen3.5-flash 52.99 22.99 55.48 57.23 57.74 31.68 56.40 25.03 54.52 46.78 52.58 87.19 60.43
qwen3.5-plus 55.94 24.73 52.73 54.23 56.79 29.21 61.0528.31 46.85 28.33 50.61 90.20 51.76 Codex
deepseek-v4-flash 57.10 27.71 73.37 70.61 62.95 35.04 62.87 27.19 51.82 41.14 54.55 83.96 55.81
qwen3.5-flash 52.91 21.34 48.53 41.19 59.49 31.53 57.59 24.80 61.17 58.86 53.88 87.40 62.44
qwen3.5-plus 53.38 23.70 63.09 54.62 62.58 32.91 58.72 21.52 58.63 52.84 46.49 79.81 52.03 Claude Code
deepseek-v4-flash 53.29 22.85 72.04 70.23 56.97 28.85 58.78 23.65 52.34 41.99 51.79 90.92 61.40
English Financial Tasks
Agent Foundation ModelNC IR MJ ER SUM PLA
ACC ROU ACC ROU ACC ROU ACC ROU ACC Cov ACC FS Cov
Closed-Source Large Language Models
LightRAG gpt-5.5 58.16 38.52 43.71 22.60 54.62 28.34 50.63 26.12 50.55 51.01 62.09 63.83 66.63
PageIndex gpt-5.4 61.12 44.54 56.30 35.33 62.69 35.90 54.51 36.35 55.95 56.29 71.35 73.09 72.48
gpt-5.5 60.3446.83 59.64 36.55 62.5739.59 70.21 56.82 58.25 52.51 66.82 86.41 65.06Codexqwen3.7-plus 55.61 38.96 59.33 31.87 56.80 34.37 68.6057.64 55.8661.30 63.00 92.31 73.15
sonnet 5 52.94 35.66 51.76 27.41 54.53 29.92 59.78 37.63 55.07 55.16 60.9892.76 74.74Claude Codeqwen3.7-plus 52.11 35.34 52.97 29.13 51.35 29.19 61.44 49.41 52.11 57.79 57.23 86.72 66.77
Open-Source Large Language Models
qwen3.5-flash 53.17 35.43 53.75 30.92 58.95 32.20 59.04 46.65 50.04 53.07 60.12 84.85 66.76
qwen3.5-plus 57.61 38.45 56.09 31.41 60.45 38.22 59.39 46.74 36.70 29.47 62.92 79.74 67.45 Codex
deepseek-v4-flash 59.39 45.85 59.23 36.45 60.02 33.79 64.65 48.76 60.13 49.81 65.85 86.56 62.55
qwen3.5-flash 51.42 33.86 52.42 27.69 55.65 30.37 58.95 44.65 49.81 52.32 59.58 86.22 67.79
qwen3.5-plus 50.33 35.41 54.29 30.86 56.97 33.81 61.04 45.77 43.39 34.76 61.49 83.43 70.45 Claude Code
deepseek-v4-flash 56.67 41.39 57.69 34.37 61.20 35.74 66.84 55.74 57.13 51.48 65.45 89.63 75.21
Table 6: Task-level generation-evaluation results judged by GPT-5-mini. The table reports task-specific ACC,
ROUGE-style overlap (ROU), faithfulness (FS), and evidence coverage (Cov) for Chinese and English task groups,
highlighting how system performance varies across retrieval, reasoning, summarization, planning, and numerical-
comparison tasks.
9

System Core model CN
avg.EN
avg.Best CN
taskBest EN
taskToken
useTime
spend (s)
Closed-Source Large Language Models
LightRAG gpt-5.5 53.52 53.29 KQ 58.20 PLA 62.09 12.8k 8.6
PageIndex gpt-5.4 59.03 60.32 MR 65.19 PLA 71.35 15.6k 11.9
Codex gpt-5.560.63 62.97IR 73.33 ER 70.21 42.5k 46.8
Codex qwen3.7-plus 59.98 59.87 IR 69.63 ER 68.60 39.2k 35.4
Claude Code sonnet 5 58.17 55.84 KQ 62.63 PLA 60.98 45.8k 50.6
Claude Code qwen3.7-plus 55.14 54.54 IR 60.84 ER 61.44 41.0k 38.7
Open-Source Large Language Models
Codex qwen3.5-flash 54.95 55.85 KQ 57.74 PLA 60.12 31.4k 22.8
Codex qwen3.5-plus 54.00 55.53 MR 61.05 PLA 62.92 34.7k 29.6
Codex deepseek-v4-
flash60.44 61.55IR 73.37PLA 65.85 36.9k 26.4
Claude Code qwen3.5-flash 55.60 54.64SUM 61.17PLA 59.58 33.2k 25.9
Claude Code qwen3.5-plus 57.15 54.59 IR 63.09 PLA 61.49 37.6k 32.1
Claude Code deepseek-v4-
flash57.54 60.83 IR 72.04 ER 66.84 39.1k 28.7
Table 7: Representative task-level results from the draft experiments. CN and EN averages are computed over
the accuracy-style columns for the task labels in each language subset. Token use and Time spent are estimated
per-question averages; token use denotes approximate input plus output tokens.
clear efficiency cost, motivating cost-aware routing
in real financial assistants.
4.4 Scenario-level Observations
Table 5 reveals large differences across document
types. PageIndex raises OA over LightRAG by
9.73 points on Chinese (72.15 vs. 62.42) and 6.98
points on English (66.90 vs. 59.92). The largest
scene-level difference between these two reported
configurations appears on Chinese investment strat-
egy reports, where the score rises from 41.46 to
64.78 (+23.32). Improvements are also clear on
Chinese SCA (+10.85) and English IS (+11.78).
These patterns are consistent with the value of pre-
serving document layout, although the different
core models prevent a controlled causal attribution
to indexing alone.
The strongest agentic configuration varies by
domain. Claude Code with Sonnet 5 is strongest on
five Chinese scenes: FR (91.04), IS (68.69), MTA
(67.03), BFS (87.41), and CSR (73.21). Claude
Code with Qwen3.7-Plus leads Chinese SCA at
89.13, and Claude Code with DeepSeek-V4-Flash
leads Chinese CF at 73.14. English results are more
fragmented: PageIndex leads CF at 83.33, Claude
Code with Qwen3.7-Plus leads IS at 67.05, Claude
Code with Sonnet 5 leads MTA at 67.96, and Codex
with GPT-5.5 leads GFB at 70.37. Codex with GPT-
5.5 and Claude Code with DeepSeek-V4-Flash tie
on English FR at 70.38.
Two scenario patterns are particularly informa-tive. Chinese IS is difficult for a light retrieval
pipeline but improves sharply with page-aware and
agentic configurations, which is consistent with the
long, cross-section structure of investment strategy
reports. English GFB separates systems in a dif-
ferent way: LightRAG and PageIndex score 41.48
and 37.69, whereas Codex with GPT-5.5 reaches
70.37. Fiscal bulletins often require linking policy
statements to amounts, periods, and program struc-
ture, so retrieving a locally similar span may not be
enough. Conversely, PageIndex’s strong English
CF score shows that structured retrieval can match
or exceed agentic systems when tables and their
surrounding context are indexed effectively.
4.5 Task-Level Observations
Table 6 confirms that task difficulty cannot be
captured by language-level averages alone. For
Chinese tasks, the best ACC ranges from 57.10
on comparison to 73.37 on implicit reasoning,
both achieved by Codex with DeepSeek-V4-Flash.
PageIndex is strongest on KQ (63.19) and MR
(65.19), and it also has the best Chinese planning
ACC (56.53). Claude Code with Qwen3.5-Flash
obtains the best Chinese summary ACC (61.17),
while Claude Code with Sonnet 5 obtains the best
Chinese planning FS (96.18) and Cov (68.36).
These splits show that answer correctness, faith-
fulness, and coverage are not interchangeable.
For English tasks, PageIndex leads numerical
comparison (61.12), multi-hop judgment (62.69),
10

and planning ACC (71.35). Codex with GPT-5.5
leads implicit reasoning (59.64) and explicit reason-
ing (70.21). Codex with DeepSeek-V4-Flash leads
summarization ACC (60.13), while Codex with
Qwen3.7-Plus has the best English summary Cov
(61.30). Claude Code with Sonnet 5 has the best
English planning FS (92.76), and Claude Code with
DeepSeek-V4-Flash has the best English planning
Cov (75.21). The task-level results therefore sepa-
rate retrieval accuracy, reasoning accuracy, ground-
edness, and answer completeness.
The gap between ACC and ROU shows why
lexical overlap is only a secondary diagnostic. For
example, PageIndex reaches 65.19 ACC but only
27.93 ROU on Chinese multi-hop reasoning. A
valid multi-step answer can use different wording
from the reference, whereas a high-overlap answer
can still use the wrong period, scope, or unit. Judge-
based accuracy and explicit numeric checks are
therefore more informative for complex financial
reasoning than surface overlap alone.
Planning and summarization expose a different
weakness: grounded language does not guarantee
complete task fulfillment. On Chinese planning,
Codex with GPT-5.5 receives 83.54 FS but only
32.77 Cov, indicating that the response is largely
faithful yet omits many mandatory analysis steps.
Claude Code with Sonnet 5 raises these values to
96.18 FS and 68.36 Cov, but its planning ACC is
still 53.71. The same pattern appears in English:
several systems exceed 85 FS while their cover-
age remains in the mid-60s or low-70s. A strong
planner must therefore satisfy three separate re-
quirements: remain faithful to the source, cover
the document-specific evidence plan, and reach the
correct analytical objective.
Numerical comparison and multi-hop judgment
create another separation. Their errors frequently
arise after relevant evidence has already been re-
trieved: systems compare incompatible periods,
confuse percentage changes with percentage-point
changes, select a consolidated value instead of a
segment value, or lose a negative sign in a cash-
flow table. These errors explain why tool access
by itself is insufficient. Reliable agents must bind
each number to its row label, column period, unit,
and entity scope, and then verify the calculation
before generation.Failure mode Typical symptom
Numeric drift Correct formula but wrong period,
denominator, unit, or sign.
Evidence omission Final answer cites one document
element while ignoring a required
table or chart.
Layout confusion Row headers, column years, captions,
or footnotes are detached from values.
Over-synthesis The model adds plausible industry
claims not supported by the
documents.
Weak planning The answer gives generic steps instead
of a document-grounded research
plan.
Table 8: Common failure modes observed in Finance-
ComplexQA error analysis.
5 Further Analysis
5.1 Failure Modes
Table 8 summarizes the main error categories ob-
served during manual inspection. Many failures
occur after a system has already found partially rel-
evant evidence. The model may retrieve the right
table but read the wrong row, retrieve a paragraph
about the right company but miss the time period,
or quote a trend without checking whether the nu-
merical table supports it. These errors are espe-
cially harmful in finance because an answer can be
fluent and still reverse the investment implication.
The taxonomy also explains the metric patterns
in Table 6. Numeric drift lowers ACC on com-
parison, explicit reasoning, and multi-hop judg-
ment. Evidence omission lowers Cov in summa-
rization and planning. Layout confusion affects
both scene-level and task-level performance be-
cause table headers, captions, footnotes, and page
context often determine the meaning of a number.
Over-synthesis and weak planning reduce faith-
fulness and completeness even when the answer
sounds professional.
5.2 Human Evaluation
Table 9 reports the sampled human evaluation
groups. Each participant group reviews 50 QA
pairs, but the language, domain, document length,
correctness-and-completeness score, and time cost
vary substantially. The table is therefore not a
leaderboard; it is a diagnostic view of annotation
difficulty across subsets.
The sampled results show that difficulty is not
determined by length alone. Participant 4 reviews
English IS documents with the longest average
time cost (34,045 seconds) and obtains the high-
11

Candidates LNG Domain #QAs Length (avg.) Corr&Comp Time Spent (s)
Partic. 1 CN CF 50 7343 76.0 9679
Partic. 2 CN FR,MTA,BFS,CSR,SCA 50 12909 2.0 8991
Partic. 3 CN IS 50 15519 75.0 5899
Partic. 4 EN IS 50 27191 92.0 34045
Partic. 5 EN FR,CF,MTA 50 29784 78.0 12278
Partic. 6 EN GFB 50 16992 0.0 30252
Table 9: Human evaluation sample statistics by participant group. Each row reports the language, covered domains,
number of reviewed QA pairs, average document length, correctness-and-completeness score (Corr&Comp), and
total annotation time for the sampled subset.
est Corr&Comp score (92.0), while Participant 6
reviews shorter English GFB documents but ob-
tains 0.0. Participant 2 also obtains a very low
score (2.0) on a mixed Chinese subset covering
FR, MTA, BFS, CSR, and SCA. These low-scoring
groups suggest that some domains require stricter
evidence tracing, clearer task decomposition, or
more careful reference-answer calibration. This
supports the need for the multi-dimensional judge
criteria in Table 4 and the failure taxonomy in Ta-
ble 8.
5.3 Why the Benchmark Is Difficult
FinanceComplexQA is difficult because it com-
bines three pressures that are usually tested sepa-
rately. The first is long-document grounding: ev-
idence may be distributed across distant sections.
The second is financial computation: answers must
preserve units, time periods, signs, and accounting
relationships. The third is professional synthesis:
the final response must explain why the evidence
matters. A system that succeeds at only one of
these pressures will often produce an answer that
is locally plausible but globally incomplete.
Cross-layout reasoning is the clearest example.
Suppose a question asks whether a firm’s profit im-
provement is supported by operating fundamentals.
A paragraph may attribute improvement to cost
control, a table may show that gross margin rose,
and a chart or note may reveal that revenue growth
slowed. A strong answer must combine these sig-
nals and avoid overstating the conclusion. Many
systems retrieve one or two relevant elements but
miss the element that changes the interpretation,
leading to an answer that is directionally plausible
yet analytically incomplete.
The experiment tables make this difficulty visi-
ble. PageIndex can be highly competitive when
layout-preserving retrieval is enough, as in En-
glish CF and several task-level reasoning columns.
Agentic systems improve some open-ended andimplicit-reasoning settings, but they pay higher cost
and still miss coverage requirements. Human eval-
uation further shows that some subsets remain hard
even when documents are shorter, which indicates
that domain structure and evidence relations matter
as much as raw document length.
5.4 Implications for Agent Design
The experiments suggest that future financial
agents should include explicit evidence plans. Be-
fore generating a final answer, the system should
identify the required evidence types, check whether
each has been retrieved, and verify calculations in-
dependently. A second useful direction is layout-
preserving retrieval, where table structure, cap-
tions, notes, and page-level context are indexed
together. A third direction is metric-aware self-
checking: planning and summarization should ex-
plicitly check coverage and faithfulness, while nu-
merical tasks should verify units, signs, and peri-
ods.
Finally, cost-aware routing is necessary. Table 7
shows that agentic systems can improve difficult
tasks but require much higher token usage and
latency. A production financial assistant should
route simple knowledge queries to cheaper retrieval
paths, send layout-sensitive questions to structured
retrieval, and reserve full agent loops for tasks that
require decomposition, calculation, or cross-layout
synthesis.
5.5 Ablation Directions
Although the current experiments focus on system-
level comparison, FinanceComplexQA also sup-
ports ablations that isolate individual design
choices. One ablation removes layout metadata
and indexes only plain text chunks, measuring how
much performance depends on table headers, cap-
tions, and page-local structure. Another disables
tool use and asks the foundation model to answer
12

from retrieved context alone, testing whether ex-
plicit calculation is necessary for numerical tasks.
A third ablation compares one-shot generation
with a plan-then-answer workflow. A fourth com-
pares full agent loops with cost-aware routing poli-
cies. These studies would clarify when agentic
reasoning is worth its additional token and latency
cost, and when structured retrieval already provides
the needed evidence.
6 Related Work
6.1 Financial and Enterprise QA
Financial QA benchmarks have evolved from short-
context numerical reasoning to longer and more
realistic document settings. FinQA (Chen et al.,
2021) and ConvFinQA (Chen et al., 2022) focus on
multi-step arithmetic over financial report excerpts,
while TAT-QA (Zhu et al., 2021) combines tabular
and textual evidence. These datasets established
numerical reasoning as a core financial capability,
but the context remains tightly scoped.
FinanceBench (Islam et al., 2023) introduced
open-ended questions grounded in SEC filings
and highlighted the gap between LLM-only
generation and document-grounded answering.
DocFinQA (Reddy et al., 2024) extends the context
to full filings, yet still focuses on single-document
reasoning. Enterprise-oriented benchmarks such
as OfficeQA (Databricks, 2024) and OfficeQA-
Pro (Databricks, 2025) evaluate grounded reason-
ing over large office corpora and long documents,
bringing document parsing closer to the evalua-
tion target. BizFinBench and BizFinBench v2 (Lu
et al., 2025; Guo et al., 2026) expand bilingual fi-
nancial coverage and include business-driven tasks,
but their evidence requirements are usually not de-
signed for cross-layout and cross-document synthe-
sis at scale.
6.2 RAG and Agentic RAG
RAG couples retrieval with generation (Lewis et al.,
2020a). Later work improves retrieval through re-
ranking (Nogueira and Cho, 2019), query rewrit-
ing (Ma et al., 2023), and modular pipelines (Gao
et al., 2023). Iterative methods such as Self-
RAG (Asai et al., 2024), CRAG (Yan et al., 2024),
Adaptive-RAG (Jeong et al., 2024), IRCoT (Trivedi
et al., 2023), and ITER-RETGEN (Shao et al.,
2023) interleave retrieval with reflection or reason-
ing. GraphRAG-style systems (Edge et al., 2024;
Peng et al., 2024) and page-level indexing (Vec-tifyAI, 2025) preserve higher-level structure and
multi-hop links. Yet most evaluations still rely on
general QA or scientific corpora rather than pro-
fessional bilingual financial documents with tables,
charts, and domain-specific inference.
Agentic RAG integrates planning, tool use, re-
flection, and multi-agent collaboration into the re-
trieval loop (Singh et al., 2025). Related agent
frameworks build on chain-of-thought (Wei et al.,
2022), ReAct (Yao et al., 2023b), Reflexion (Shinn
et al., 2023), Tree-of-Thoughts (Yao et al., 2023a),
and Plan-and-Solve prompting (Wang et al., 2023b).
AutoGen (Wu et al., 2023), V oyager (Wang et al.,
2023a), Claude Code (Anthropic, 2024), and Ope-
nAI Codex (OpenAI, 2024) demonstrate that tool-
using agents can orchestrate complex workflows.
FinanceComplexQA asks whether these capabili-
ties transfer to grounded reasoning with financial
documents.
7 Conclusion
We presented Finance-LaTeX, a scalable work-
flow for generating layout-rich financial documents
and verified QA pairs, and FinanceComplexQA, a
bilingual benchmark for agentic reasoning over
industrial-grade financial documents. Finance-
ComplexQA contains 2,026 deep research tasks
over 1009 documents and is designed around dual-
context reasoning, cross-layout evidence aggre-
gation, stable reference answers, and Agent-as-a-
Judge evaluation. Experiments on RAG, layout-
aware retrieval, and agentic systems show that pre-
serving document structure is crucial, while current
agents still struggle with numerical drift, evidence
omission, layout confusion, incomplete coverage,
and high inference cost. These findings suggest
that reliable financial agents need layout-preserving
retrieval, explicit evidence planning, calculation
verification, and cost-aware routing. We hope Fi-
nanceComplexQA supports future work on finan-
cial agents that are grounded, complete, and verifi-
able.
Limitations
FinanceComplexQA focuses on document-
grounded financial reasoning and does not cover
live trading, real-time market forecasting, or
personalized investment advice. Although the
benchmark is bilingual, it currently emphasizes
Chinese and English and does not test broader
multilingual transfer. The Agent-as-a-Judge
13

protocol improves scalability, but may inherit
evaluator bias; future releases should include
stronger human calibration and adversarial judge
checks. Synthetic data from Finance-LaTeX is for
development only.
References
Anthropic. 2024. Claude code: An agentic coding
assistant. https://code.claude.com/docs/en/
quickstart.
Anthropic. 2026. Claude code overview. https:
//code.claude.com/docs . Official Claude Code
documentation. Accessed April 12, 2026.
Akari Asai, Zeqiu Wu, Yizhong Wang, Avirup Sil, and
Hannaneh Hajishirzi. 2024. Self-RAG: Learning to
retrieve, generate, and critique through self-reflection.
InProceedings of ICLR.
Antoine Bigeard, Langston Nashold, Rayan Krishnan,
and Shirley Wu. 2025. Finance agent benchmark:
Benchmarking llms on real-world financial research
tasks.arXiv preprint arXiv:2508.00828.
Zhiyu Chen, Wenhu Chen, Charese Smiley, Sameena
Shah, Iana Borova, Dylan Langdon, Reema Moussa,
Matt Beane, Ting-Hao Huang, Bryan Routledge, and
William Yang Wang. 2021. FinQA: A dataset of nu-
merical reasoning over financial data. InProceedings
of EMNLP.
Zhiyu Chen, Shiyang Li, Charese Smiley, Zhiqiang Ma,
Sameena Shah, and William Yang Wang. 2022. Con-
vFinQA: Exploring the chain of numerical reasoning
in conversational finance question answering. InPro-
ceedings of EMNLP.
Xianfu Cheng, Hang Zhang, Jian Yang, Xiang Li, Weix-
iao Zhou, Fei Liu, Kui Wu, Xiangyuan Guan, Tao
Sun, Xianjie Wu, and 1 others. 2025a. Xformparser:
A simple and effective multimodal multilingual semi-
structured form parser. InProceedings of the 31st
International Conference on Computational Linguis-
tics, pages 606–620.
Xianfu Cheng, Wei Zhang, Shiwei Zhang, Jian Yang,
Xiangyuan Guan, Xianjie Wu, Xiang Li, Ge Zhang,
Jiaheng Liu, Yuying Mai, and 1 others. 2025b. Sim-
plevqa: Multimodal factuality evaluation for multi-
modal large language models. InProceedings of the
IEEE/CVF International Conference on Computer
Vision, pages 4637–4646.
Databricks. 2024. Introducing OfficeQA:
An end-to-end grounded reasoning bench-
mark. https://www.databricks.com/blog/
introducing-officeqa-benchmark-end-to-en\
d-grounded-reasoning.
Databricks. 2025. OfficeQA-Pro: An enterprise bench-
mark for long-document grounded reasoning. https:
//arxiv.org/html/2603.08655v1.DeepSeek-AI, Aixin Liu, Bei Feng, Bing Xue, Bingx-
uan Wang, Bochao Wu, Chengda Lu, Chenggang
Zhao, Chengqi Deng, Chenyu Zhang, Chong Ruan,
Damai Dai, Daya Guo, Dejian Yang, Deli Chen,
Dongjie Ji, Erhang Li, Fangyun Lin, Fucong Dai,
and 181 others. 2025. Deepseek-v3 technical report.
Preprint, arXiv:2412.19437.
Darren Edge, Ha Trinh, Newman Cheng, Joshua
Bradley, Alex Chao, Apurva Mody, Steven Truitt,
and Jonathan Larson. 2024. From local to global: A
GraphRAG approach to query-focused summariza-
tion.arXiv preprint arXiv:2404.16130.
Yunfan Gao, Yun Xiong, Xinyu Gao, Kangxiang Jia,
Jinliu Pan, Yuxi Bi, Yi Dai, Jiawei Sun, and Haofen
Wang. 2023. Retrieval-augmented generation for
large language models: A survey.arXiv preprint
arXiv:2312.10997.
Xin Guo, Rongjunchen Zhang, Guilong Lu, Xuntao
Guo, Shuai Jia, Zhi Yang, and Liwen Zhang. 2026.
Bizfinbench. v2: A unified dual-mode bilingual
benchmark for expert-level financial capability align-
ment.arXiv preprint arXiv:2601.06401.
Zirui Guo, Lianghao Xia, Yanhua Yu, Tian Ao, and
Chao Huang. 2024. Lightrag: Simple and fast
retrieval-augmented generation.arXiv preprint
arXiv:2410.05779, 2(3).
Yancheng He, Shilong Li, Jiaheng Liu, Yingshui Tan,
Weixun Wang, Hui Huang, Xingyuan Bu, Hangyu
Guo, Chengwei Hu, Boren Zheng, and 1 others. 2025.
Chinese simpleqa: A chinese factuality evaluation for
large language models. InProceedings of the 63rd
Annual Meeting of the Association for Computational
Linguistics (Volume 1: Long Papers), pages 19182–
19208.
Pranab Islam, Anand Kannappan, Douwe Kiela, Re-
becca Qian, Nino Scherrer, and Bertie Vidgen. 2023.
FinanceBench: A new benchmark for financial ques-
tion answering.arXiv preprint arXiv:2311.11944.
Soyeong Jeong, Jinheon Baek, Sukmin Cho, Sung Ju
Hwang, and Jong C. Park. 2024. Adaptive-RAG:
Learning to adapt retrieval-augmented large language
models through question complexity. InProceedings
of NAACL.
Patrick Lewis, Ethan Perez, Aleksandra Piktus, Fabio
Petroni, Vladimir Karpukhin, Naman Goyal, Hein-
rich Küttler, Mike Lewis, Wen-tau Yih, Tim Rock-
täschel, Sebastian Riedel, and Douwe Kiela. 2020a.
Retrieval-augmented generation for knowledge-
intensive NLP tasks. InAdvances in Neural Informa-
tion Processing Systems (NeurIPS).
Patrick Lewis, Ethan Perez, Aleksandra Piktus, Fabio
Petroni, Vladimir Karpukhin, Naman Goyal, Hein-
rich Küttler, Mike Lewis, Wen-tau Yih, Tim Rock-
täschel, and 1 others. 2020b. Retrieval-augmented
generation for knowledge-intensive nlp tasks.Ad-
vances in neural information processing systems,
33:9459–9474.
14

Xinzhe Li. 2025. A review of prominent paradigms for
llm-based agents: Tool use, planning (including rag),
and feedback learning. InProceedings of the 31st in-
ternational conference on computational linguistics,
pages 9760–9779.
Jiacheng Liu, Xiaohan Zhao, Xinyi Shang, and Zhiqiang
Shen. 2026. Dive into claude code: The design space
of today’s and future ai agent systems.arXiv preprint
arXiv:2604.14228.
Guilong Lu, Xuntao Guo, Rongjunchen Zhang, Wen-
qiao Zhu, and Ji Liu. 2025. Bizfinbench: A business-
driven real-world financial benchmark for evaluating
llms.arXiv preprint arXiv:2505.19457.
Xinbei Ma, Yeyun Gong, Pengcheng He, Hai Zhao,
and Nan Duan. 2023. Query rewriting for retrieval-
augmented large language models. InProceedings
of EMNLP.
Rodrigo Nogueira and Kyunghyun Cho. 2019. Pas-
sage re-ranking with BERT.arXiv preprint
arXiv:1901.04085.
OpenAI. 2024. OpenAI Codex: Cloud-based soft-
ware engineering agent. https://openai.com/
zh-Hans-CN/codex/get-started/.
OpenAI. 2026. Harness engineering: Leveraging
Codex in an agent-first world. OpenAI, https://
openai.com/index/harness-engineering/ . Ac-
cessed June 2026.
Tejal Patwardhan, Rachel Dias, Elizabeth Proehl, Grace
Kim, Michele Wang, Olivia Watkins, Simón Posada
Fishman, Marwan Aljubeh, Phoebe Thacker, Lau-
rance Fauconnet, and 1 others. 2025. Gdp-
val: Evaluating ai model performance on real-
world economically valuable tasks.arXiv preprint
arXiv:2510.04374.
Boci Peng, Yun Zhu, Yongchao Liu, Xiaohe Bo,
Haizhou Shi, Chuntao Hong, Yan Zhang, and Siliang
Tang. 2024. Graph retrieval-augmented generation:
A survey.arXiv preprint arXiv:2408.08921.
Revanth Gangi Reddy and 1 others. 2024. DocFinQA:
A long-context financial reasoning dataset. InFind-
ings of ACL.
Timo Schick, Jane Dwivedi-Yu, Roberto Dessì, Roberta
Raileanu, Maria Lomeli, Eric Hambro, Luke Zettle-
moyer, Nicola Cancedda, and Thomas Scialom. 2023.
Toolformer: Language models can teach themselves
to use tools.Advances in neural information process-
ing systems, 36:68539–68551.
Zhihong Shao, Yeyun Gong, Yelong Shen, Minlie
Huang, Nan Duan, and Weizhu Chen. 2023. Enhanc-
ing retrieval-augmented large language models with
iterative retrieval-generation synergy. InFindings of
EMNLP.Noah Shinn, Federico Cassano, Edward Berman, Ash-
win Gopinath, Karthik Narasimhan, and Shunyu Yao.
2023. Reflexion: Language agents with verbal rein-
forcement learning. InAdvances in Neural Informa-
tion Processing Systems (NeurIPS).
Daixin Shu, Jian Yang, Zhenhe Wu, Xianjie Wu, Xianfu
Cheng, Guan Xiangyuan, Yanghai Wang, Pengfei
Wu, Tingyang Yang, Hualei Zhu, and 1 others. 2026.
M3tqa: Massively multilingual multitask table ques-
tion answering. InFindings of the Association for
Computational Linguistics: ACL 2026, pages 22578–
22602.
Aditi Singh and 1 others. 2025. Agentic retrieval-
augmented generation: A survey on agentic RAG.
arXiv preprint arXiv:2501.09136.
Hugo Touvron, Thibaut Lavril, Gautier Izacard, Xavier
Martinet, Marie-Anne Lachaux, Timothée Lacroix,
Baptiste Rozière, Naman Goyal, Eric Hambro, Faisal
Azhar, Aurélien Rodriguez, Armand Joulin, Edouard
Grave, and Guillaume Lample. 2023. Llama: Open
and efficient foundation language models.CoRR,
abs/2302.13971.
Harsh Trivedi, Niranjan Balasubramanian, Tushar Khot,
and Ashish Sabharwal. 2023. Interleaving retrieval
with chain-of-thought reasoning for knowledge-
intensive multi-step questions. InProceedings of
ACL.
VectifyAI. 2025. PageIndex: Hierarchical page-level re-
trieval for long documents. https://github.com/
VectifyAI/PageIndex.
Guanzhi Wang, Yuqi Xie, Yunfan Jiang, Ajay Man-
dlekar, Chaowei Xiao, Yuke Zhu, Linxi Fan, and An-
ima Anandkumar. 2023a. V oyager: An open-ended
embodied agent with large language models.arXiv
preprint arXiv:2305.16291.
Lei Wang, Wanyu Xu, Yihuai Lan, Zhiqiang Hu,
Yunshi Lan, Roy Ka-Wei Lee, and Ee-Peng Lim.
2023b. Plan-and-solve prompting: Improving zero-
shot chain-of-thought reasoning by large language
models. InProceedings of ACL.
Jason Wei, Nguyen Karina, Hyung Won Chung,
Yunxin Joy Jiao, Spencer Papay, Amelia Glaese, John
Schulman, and William Fedus. 2024. Measuring
short-form factuality in large language models.arXiv
preprint arXiv:2411.04368.
Jason Wei, Xuezhi Wang, Dale Schuurmans, Maarten
Bosma, Brian Ichter, Fei Xia, Ed H. Chi, Quoc V . Le,
and Denny Zhou. 2022. Chain-of-thought prompt-
ing elicits reasoning in large language models. In
Advances in Neural Information Processing Systems
(NeurIPS).
Qingyun Wu, Gagan Bansal, Jieyu Zhang, Yiran Wu,
Shaokun Zhang, Erkang Zhu, Beibin Li, Li Jiang,
Xiaoyun Zhang, and Chi Wang. 2023. AutoGen:
Enabling next-gen LLM applications via multi-agent
conversation.arXiv preprint arXiv:2308.08155.
15

Xianjie Wu, Di Liang, Jian Yang, Xianfu Cheng,
LinZheng Chai, Tongliang Li, Liqun Yang, and Zhou-
jun Li. 2025a. Breaking size barrier: Enhancing rea-
soning for large-size table question answering. In
International Conference on Database Systems for
Advanced Applications, pages 241–256. Springer.
Xianjie Wu, Xiaohang Xu, Tingyu Jiang, Jian Yang,
Di Liang, Xianfu Cheng, Zhenhe Wu, Linzheng
Chai, Wei Zhang, Jiaheng Liu, and 1 others. 2026.
Mmtablebench: A multi-level multimodal bench-
mark for reasoning and layout complexity in table qa.
InProceedings of the ACM Web Conference 2026,
pages 3881–3892.
Xianjie Wu, Jian Yang, Linzheng Chai, Ge Zhang, Ji-
aheng Liu, Xeron Du, Di Liang, Daixin Shu, Xi-
anfu Cheng, Tianzhen Sun, and 1 others. 2025b.
Tablebench: A comprehensive and complex bench-
mark for table question answering. InProceedings
of the AAAI Conference on Artificial Intelligence,
volume 39, pages 25497–25506.
Zhishang Xiang, Chuanjie Wu, Qinggang Zhang,
Shengyuan Chen, Zijin Hong, Xiao Huang, and Jin-
song Su. 2025. When to use graphs in rag: A com-
prehensive analysis for graph retrieval-augmented
generation.arXiv preprint arXiv:2506.05690.
Yilin Xiao, Junnan Dong, Chuang Zhou, Su Dong, Qian-
wen Zhang, Di Yin, Xing Sun, and Xiao Huang. 2025.
Graphrag-bench: Challenging domain-specific rea-
soning for evaluating graph retrieval-augmented gen-
eration.arXiv preprint arXiv:2506.02404.
Shi-Qi Yan, Jia-Chen Gu, Yun Zhu, and Zhen-Hua Ling.
2024. Corrective retrieval augmented generation.
arXiv preprint arXiv:2401.15884.
An Yang, feng Li, and etc. 2024. Qwen3 technical
report.arXiv preprint arXiv:2505.09388.
Jian Yang, Xianglong Liu, Weifeng Lv, Ken Deng,
Shawn Guo, Lin Jing, Yizhi Li, Shark Liu, Xianzhen
Luo, Yuyu Luo, and 1 others. 2025. From code
foundation models to agents and applications: A
comprehensive survey and practical guide to code
intelligence.arXiv preprint arXiv:2511.18538.
Jian Yang, Wei Zhang, Shawn Guo, Zhengmao Ye, Lin
Jing, Shark Liu, Yizhi Li, Jiajun Wu, Cening Liu,
X Ma, and 1 others. 2026a. Iquest-coder-v1 technical
report.arXiv preprint arXiv:2603.16733.
Jian Yang, Wei Zhang, Shuyue Guo, Yizhi LI, Linzheng
Chai, Zhengmao Ye, Shukai Liu, Yuyang Song, Ji-
ajun Wu, Che Liu, Tianyu Zheng, Siwei Wu, Leo
L, Xudong Ma, Chuan Hao, Ran Tao, Yan Xing,
Jianzhou Wang, Mingjie Tang, and 5 others. 2026b.
Loopcoder: Scaling code intelligence via looped lan-
guage models. InFindings of the Association for
Computational Linguistics, ACL 2026, San Diego,
California, United States, July 2-7, 2026, pages
16209–16223. Association for Computational Lin-
guistics.Shunyu Yao, Dian Yu, Jeffrey Zhao, Izhak Shafran,
Thomas L. Griffiths, Yuan Cao, and Karthik
Narasimhan. 2023a. Tree of thoughts: Deliberate
problem solving with large language models. InAd-
vances in Neural Information Processing Systems
(NeurIPS).
Shunyu Yao, Jeffrey Zhao, Dian Yu, Nan Du, Izhak
Shafran, Karthik Narasimhan, and Yuan Cao. 2023b.
ReAct: Synergizing reasoning and acting in language
models. InProceedings of ICLR.
Mingtian Zhang, Yu Tang, and PageIndex Team.
2025. Pageindex: Next-generation vector-
less, reasoning-based rag.PageIndex Blog.
Https://pageindex.ai/blog/pageindex-intro.
Weixiao Zhou, Gengyao Li, Xianfu Cheng, Xinnian
Liang, Junnan Zhu, Feifei Zhai, and Zhoujun Li.
2023. Multi-stage pre-training enhanced by chatgpt
for multi-scenario multi-domain dialogue summa-
rization. InFindings of the Association for Computa-
tional Linguistics: EMNLP 2023, pages 6893–6908.
Weixiao Zhou, Gengyao Li, Xianfu Cheng, Junnan
Zhu, Feifei Zhai, and Zhoujun Li. 2026. A large-
scale multi-dimensional empirical study of llms
for conversation summarization.arXiv preprint
arXiv:2606.15974.
Weixiao Zhou, Junnan Zhu, Gengyao Li, Xianfu Cheng,
Xinnian Liang, Feifei Zhai, and Zhoujun Li. 2025.
What are they talking about? a benchmark of
knowledge-grounded discussion summarization. In
Proceedings of the 14th International Joint Confer-
ence on Natural Language Processing and the 4th
Conference of the Asia-Pacific Chapter of the Asso-
ciation for Computational Linguistics, pages 2172–
2191.
Fengbin Zhu, Wenqiang Lei, Youcheng Huang, Chao
Wang, Shuo Zhang, Jiancheng Lv, Fuli Feng, and Tat-
Seng Chua. 2021. TAT-QA: A question answering
benchmark on a hybrid of tabular and textual content
in finance. InProceedings of ACL-IJCNLP.
16

A Prompt Templates
This appendix collects the LLM prompt templates used in FinBench construction. All prompts are
presented in English. Placeholders appear in <ALL_CAPS_NAME> form and are substituted programmatically
before invocation. Document-generation prompts (Part A) are typically batched with up to ten QA pairs
per call and used together with thefinance-latexskill.
Shared Constraints for Reference Document Generation (Part A)
• Answers mustnotbe obtainable directly or in a single retrieval step from the generated document.
•Generated content must remain consistent with the provided Relevant_passage and must not alter answer correct-
ness.
• Do not add glossary-style explanations or LaTeX formulas for financial terminology.
• All content is assumed publicly accessible; confidentiality or classified disclaimers are prohibited.
•Each invocation writes layoutCoT (document navigation hints) and Reference_documents (path to the generated
.texfile).
Part A — Reference Document Generation
Overview of Part A Prompt Variants
All six Part A prompts (A1–A5, including A3a/A3b) share the same skeleton: a financial-advisor role, the
finance-latex skill, the constraint that answers must not be directly retrievable, and outputs layoutCoT
plus Reference_documents . They differ mainly in benchmark source, document type, input fields,
content-generation strategy,layoutCoTgranularity, and file-naming conventions.
Part A at a Glance
ID Benchmark / Use Case Document Type Input
A1 Chinese real-industry QA Variable<FINANCIAL_DOC_TYPE>Q+A+Passage+Additional_info
A2 BizFinBenchV2 counterfactual Corporate Financial Report Q+A+Passage
A3a Chinese anomaly tracing Investment Strategy Report Q+A+Passage
A3b English BizFinBenchV2 summary Investment Strategy Report Q+A+Passage
A4 English FinanceBench real-
industryOne of five English types Q+A+Passage
A5 OfficeQA announcements Government Fiscal BulletinOneQ+A+Passage
Input structure.OnlyA1accepts Additional_information , which carries SubQuestions (decom-
posed sub-queries) and Related_questions . The prompt requires expanding the document so that the
main question, each sub-question, and each related question can all be answered.A2–A5use the simpler
<Question, Answer, Relevant_passage>triple only.
Document type. A1leaves the document type to the <FINANCIAL_DOC_TYPE> placeholder (filled from
doc_type ).A2,A3a, andA3beach fix the report genre (financial report or investment strategy report).
A4restricts output to one of five English genres: Investment Strategy Report, Corporate Financial Report,
FinTech Research Report, Market Trend Analysis Report, or Market Regulation and Compliance Audit
Report.A5always produces a government fiscal bulletin.
Content-generation strategy.
•A1(richest): treat Relevant_passage as a writing theme; search and expand content; mandate
complex tables for numerical sub-questions and rich-text forms for entity-retrieval sub-questions.
•A2/A3: treat passage items asreference data/content; randomly convert some items into paragraphs
and others into tables/forms (A3 wording: “rewrite” rather than “convert”).
17

•A4: rewrite and polish reference content into a coherent English document; may insert transitional
paragraphs;must not use Chinese.
•A5(lightest): organize passage text into bulletin format using the knowledge base only—no search-
based expansion.
layoutCoT granularity. A1emits one navigation chain persub-question.A2–A5emit one chain per
main question.
Naming conventions.A1 and A4 embed the document type in the filename; A2/A3 use fixed genre
suffixes; A5 uses a<UID>instead of an institution–year identifier.
A3a vs. A3b.These two prompts are logically identical; they differ only in benchmark language and
data source (Chinese anomaly-tracing vs. English summary items).
Summary.Part A prompts share a common skeleton but differ in input richness (A1), document-type flexibility
(A1/A4), content expansion strategy (search-and-expand vs. reorganize-only), and layoutCoT granularity (sub-question
vs. question level). Generation intensity increases in the order A5→A4≈A2/A3→A1.
A.1 Real-Industry Document Generation
A1 | Real-Industry Document Generation
Purpose Generate Chinese real-industry financial documents in LaTeX from QA pairs with
sub-questions and related questions.
Input<Question, Answer, Relevant_passage, Additional_information> (up to 10
pairs per invocation)
OutputLaTeX source file,layoutCoT, andReference_documents
Suppose you are a senior financial advisor skilled in analyzing financial industry intelligence
and writing PDF-format [FINANCIAL_DOC_TYPE] documents. In this task, I provide several sets
of <Question, Answer, Relevant_passage, Additional_information> combinations from the
financial domain. You are required to use the [SKILL_NAME] skill and follow the given
requirements to write a professional [FINANCIAL_DOC_TYPE] in LaTeX that reflects the gist
of the Relevant_passage. Please ensure that for each Question, the corresponding Answer can
be obtained from the generated file through in-depth investigation or calculation.
## Specific requirements for the content to be written:
1. In Additional_information, I provide SubQuestions obtained by decomposing the Question into
key elements. Based on each SubQuestion and the writing theme provided by Relevant_passage,
use search engines and your knowledge base to expand the [FINANCIAL_DOC_TYPE] so that each
SubQuestion can be reasonably answered. The generated content must not contradict the
factual information in Relevant_passage.
2. Follow the instructions in the [SKILL_NAME] skill. In the LaTeX [FINANCIAL_DOC_TYPE] you
write, focus on complex tables and forms so that complex Questions can be answered by
extracting key information from paragraphs, tables, or forms separately, thereby increasing
difficulty. When Questions or SubQuestions involve numerical lookup or calculation, you
must include relevant complex tables. When Questions or SubQuestions involve named-entity
retrieval, add associated rich-text forms in the generated document.
3. In Additional_information, I provide several Related_questions associated with the Question.
Based on each Related_question, use search engines and your knowledge base to expand the [
FINANCIAL_DOC_TYPE] so that Related_questions can be reasonably answered. The generated
content must not contradict the factual information in Relevant_passage.
4. Do not generate explanations or LaTeX formulas for financial terms or quantitative
terminology involved in Questions or SubQuestions.
5. All written content should be publicly accessible; do not add confidentiality clauses or
classified statements.
## You must check the generated content to ensure that the Answer cannot be obtained directly or
in a single step from the document.
18

## For each SubQuestion, you must generate a solving hint based on the written document, forming
a path-style chain of thought using sentences of the form "how to obtain XXXX from XXXX
paragraph/table/form", and store this field under the name "layoutCoT" in the json
dictionary at [QAR_TRIPLES_PATH].
## The generated document naming convention is: "[INSTITUTION_AND_YEAR]_[FINANCIAL_DOC_TYPE]_{
document_title}.tex". Name the document by referencing relevant information. The storage
path is: "[STORAGE_PATH]". Then store the path "[STORAGE_PATH]/[INSTITUTION_AND_YEAR]/[
INSTITUTION_AND_YEAR]_[FINANCIAL_DOC_TYPE]_{document_title}.tex" in the "
Reference_documents" field of the json dictionary at [QAR_TRIPLES_PATH].
## Key information required for writing the LaTeX [FINANCIAL_DOC_TYPE]:
### [NUM_PAIRS] sets of <Question, Answer, Relevant_passage, Additional_information> data:
[TO_BE_FILLED]
## Now please begin calling [SKILL_NAME] to write the LaTeX document.
A.2 Corporate Financial Report Generation
A2 | Corporate Financial Report Generation (Chinese Benchmark)
Purpose Generate corporate financial reports in LaTeX for BizFinBenchV2 counterfactual QA
items.
Input<Question, Answer, Relevant_passage>reference data (up to 10 pairs)
OutputLaTeX financial report,layoutCoT, andReference_documents
Suppose you are a senior financial advisor skilled in analyzing financial industry intelligence
and writing PDF-format corporate financial reports. In this task, I provide several sets of
<Question, Answer, Relevant_passage> combinations from the financial domain. You are
required to use the [SKILL_NAME] skill and follow the given requirements to write a
professional financial report in LaTeX that reflects the gist of the Relevant_passage.
Please ensure that for each Question, the corresponding Answer can be obtained from the
generated file through in-depth investigation or calculation.
## Specific requirements for the content to be written:
1. Based on the Question and the [reference data] provided in Relevant_passage, use search
engines and your knowledge base to expand the [reference data] into the corporate financial
report. The generated content must not contradict the factual information in
Relevant_passage and must not affect the correctness of the Answer.
2. Follow the instructions in the [SKILL_NAME] skill. In the LaTeX corporate financial report
you write, focus on the integrated use of rich-text paragraphs, complex tables, and forms
to increase the complexity of resolving the Question. Specifically, for multiple [reference
data] items, randomly convert some into rich-text paragraphs and some into complex tables/
forms. The same [reference data] item need not be converted repeatedly.
3. Do not generate explanations or LaTeX formulas for financial terms or quantitative
terminology involved in the Question or Relevant_passage.
4. All written content should be publicly accessible; do not add confidentiality clauses or
classified statements.
## Please check the generated content to ensure that the Answer cannot be obtained directly or
in a single step from the document.
## For each Question, you must generate a solving hint based on the written document, forming a
path-style chain of thought using sentences of the form "how to obtain XXXX from XXXX
paragraph/table/form", and store this field under the name "layoutCoT" in the json
dictionary at [QAR_TRIPLES_PATH].
## The generated document naming convention is: "[INSTITUTION_AND_YEAR]
_Corporate_Financial_Report_{document_title}.tex". Name the document by referencing
relevant information. The storage path is: "[STORAGE_PATH]". Then store the path "[
STORAGE_PATH]/[INSTITUTION_AND_YEAR]/[INSTITUTION_AND_YEAR]_Corporate_Financial_Report_{
document_title}.tex" in the "Reference_documents" field of the json dictionary at [
QAR_TRIPLES_PATH].
## Key information required for writing the LaTeX corporate financial report:
### [NUM_PAIRS] sets of <Question, Answer, Relevant_passage> data:
19

[TO_BE_FILLED]
## Now please begin calling [SKILL_NAME] to write the LaTeX document.
A.3 Investment Strategy Report Generation (Chinese Source)
A3a | Investment Strategy Report (Translated from Chinese Benchmark)
Purpose Generate investment strategy reports in LaTeX from Chinese anomaly-tracing bench-
mark items.
Input<Question, Answer, Relevant_passage>reference content (up to 10 pairs)
OutputLaTeX investment strategy report,layoutCoT, andReference_documents
Suppose you are a senior financial advisor skilled in analyzing financial industry intelligence
and writing PDF-format investment strategy reports. In this task, I provide several sets of
<Question, Answer, Relevant_passage> combinations from the financial domain. You are
required to use the [SKILL_NAME] skill and follow the given requirements to write a
professional investment strategy report in LaTeX that reflects the gist of the
Relevant_passage. Please ensure that for each Question, the corresponding Answer can be
obtained from the generated file through in-depth investigation or calculation.
## Specific requirements for the content to be written:
1. Based on the Question and the [reference content] provided in Relevant_passage, use search
engines and your knowledge base to expand the [reference content] into this investment
strategy report. The generated content must not contradict the factual information in
Relevant_passage and must not affect the correctness of the Answer.
2. Follow the instructions in the [SKILL_NAME] skill. In the LaTeX investment strategy report
you write, focus on the integrated use of rich-text paragraphs, complex tables, and forms
to increase the complexity of resolving the Question. Specifically, for multiple [reference
content] items, randomly rewrite some into complex tables and some into rich-text
paragraphs/complex forms. The same [reference content] item need not be converted
repeatedly.
3. Do not generate explanations or LaTeX formulas for financial terms or quantitative
terminology involved in the Question or Relevant_passage.
4. All written content should be publicly accessible; do not add confidentiality clauses or
classified statements.
## Please check the generated content to ensure that the Answer cannot be obtained directly or
in a single step from the document.
## For each Question, you must generate a solving hint based on the written document, forming a
path-style chain of thought using sentences of the form "how to obtain XXXX from XXXX
paragraph/table/form", and store this field under the name "layoutCoT" in the json
dictionary at [QAR_TRIPLES_PATH].
## The generated document naming convention is: "[INSTITUTION_AND_YEAR]
_Investment_Strategy_Report_{document_title}.tex". Name the document by referencing
relevant information. The storage path is: "[STORAGE_PATH]". Then store the path "[
STORAGE_PATH]/[INSTITUTION_AND_YEAR]/[INSTITUTION_AND_YEAR]_Investment_Strategy_Report_{
document_title}.tex" in the "Reference_documents" field of the json dictionary at [
QAR_TRIPLES_PATH].
## Key information required for writing the LaTeX investment strategy report:
### [NUM_PAIRS] sets of <Question, Answer, Relevant_passage> data:
[TO_BE_FILLED]
## Now please begin calling [SKILL_NAME] to write the LaTeX document.
20

A.4 Investment Strategy Report Generation (English Benchmark)
A3b | Investment Strategy Report (English Benchmark)
Purpose Generate English investment strategy reports in LaTeX for BizFinBenchV2 summary
items.
Input<Question, Answer, Relevant_passage>reference content (up to 10 pairs)
OutputLaTeX investment strategy report,layoutCoT, andReference_documents
Suppose you are a senior financial advisor, skilled in analyzing financial industry intelligence
and writing investment strategy reports in PDF format. This time, by providing several
sets of <Question, Answer, Relevant_passage> content combinations from the financial
professional domain, I require you to use the [SKILL_NAME] skill and follow the given
requirements to write an investment strategy report in LaTeX syntax that reflects the gist
of the Relevant_passage. Please ensure that for each Question, the corresponding Answer can
be obtained from the generated document through in-depth investigation or calculation.
## Specific requirements for writing the content are given below:
1. Based on the requirements of the Question and the [reference content] provided in the
Relevant_passage, use search engines and your knowledge base to expand the [reference
content] as the content of this investment strategy report. Note that you must ensure the
generated content does not contradict the factual information in the Relevant_passage and
does not affect the correctness of the Answer;
2. Follow the instructions in the [SKILL_NAME] skill, and focus on the integrated use of rich-
text paragraphs, complex tables, and forms in the LaTeX investment strategy report you
write, so as to increase the complexity of resolving the Questions. Specifically, for the
multiple [reference content] items input, please randomly rewrite some of them into complex
tables and some into rich-text paragraphs/complex forms. Note that the same [reference
content] item does not need to be converted repeatedly;
3. Do not generate explanations or LaTeX formulas for financial professional terms or
quantitative terms involved in the Question or SubQuestion;
4. The written content should be publicly accessible; do not add confidentiality clauses or
classified statements.
## Please check the generated content to ensure that the Answer cannot be directly/instantly
obtained from the document.
## For each Question, you must generate a prompt for solving the problem based on the written
document content, i.e., form a problem-solving path thought chain composed of a series of
sentences in the format "how to obtain XXXX from XXXX paragraph/table/form", and store this
field with the name "layoutCoT" in the json dictionary at [QAR_TRIPLES_PATH].
## The generated document naming convention is: "[INSTITUTION_AND_YEAR]
_Investment_Strategy_Report_{document_title}.tex". Please name the document by referencing
the relevant information, and the storage path is: "[STORAGE_PATH]"; then store the path
name "[STORAGE_PATH]/[INSTITUTION_AND_YEAR]/[INSTITUTION_AND_YEAR]
_Investment_Strategy_Report_{document_title}.tex" into the "Reference_documents" field of
the json dictionary at [QAR_TRIPLES_PATH].
## Key information required for writing the LaTeX investment strategy report is given below:
### [NUM_PAIRS] sets of <Question, Answer, Relevant_passage> data:
[TO_BE_FILLED]
## Now please begin calling the [SKILL_NAME] to write the LaTeX document.
A.5 General Financial Document Generation
21

A4 | General Financial Document Generation (English)
Purpose Generate one of five English financial document types in LaTeX for Fi-
nanceBench real-industry items.
Input<Question, Answer, Relevant_passage> reference content (up to 10 pairs)
Document types Investment Strategy Report; Corporate Financial Report; FinTech Research
Report; Market Trend Analysis Report; Market Regulation and Compliance
Audit Report
OutputLaTeX document,layoutCoT, andReference_documents
Suppose you are a senior financial advisor, skilled in analyzing financial industry intelligence
and writing financial PDF documents. This time, I will provide you with several sets of <
Question, Answer, Relevant_passage> content combinations from the financial professional
domain, and require you to use the "[SKILL_NAME]" skill and follow the given requirements
to write a professional financial domain document that reflects the gist of the
Relevant_passages using LaTeX syntax. Please ensure that for each Question, the
corresponding Answer can be obtained through in-depth investigation or calculation from the
generated document. You must generate the LaTeX document entirely in an English context
and may not use Chinese.
## Specific requirements for the content to be written are given below:
1. Based on the requirements of the Questions and the "reference content" in the
Relevant_passage list, use search engines and your knowledge base to rewrite and polish
each set of "reference content" into a clearly structured and coherent professional
document. Note that you must ensure the generated content does not contradict the factual
information stated in the Relevant_passages around the Questions and does not affect the
correctness of the Answers;
2. Follow the instructions in the "[SKILL_NAME]" skill, and focus on the integrated use of rich-
text paragraphs, complex tables, and forms in the LaTeX financial professional document you
write, so as to increase the complexity of resolving the Questions. Specifically, for the
input "reference content" list, you may randomly add some transitional text segments to
promote contextual coherence and clarity. Note that the added content should not repeat the
main idea of the "reference content";
3. Do not generate explanations or LaTeX formulas for financial professional terms or
quantitative terminology involved in the Questions or Relevant_passages;
4. All written content should be publicly accessible; do not add confidentiality clauses or
classified statements;
5. The types of documents (document_type) you are to write are strictly limited to the following
five: [Investment Strategy Report], [Corporate Financial Report], [FinTech Research Report
], [Market Trend Analysis Report], [Market Regulation and Compliance Audit Report].
## Please check the generated content to ensure that the Answer cannot be directly/instantly
obtained from the document.
## For each Question, you must generate a prompt for solving the problem based on the written
document content, i.e., form a problem-solving path thought chain composed of a series of
statements in the format "how to obtain XXXX from XXXX paragraph/table/form", and store
this field as "layoutCoT" in the json dictionary at [QAR_TRIPLES_PATH].
## The generated document naming convention is: "[INSTITUTION_AND_YEAR]_{document_type}_{
document_title}.tex". Please name the document by referencing the relevant information, and
the storage path is: "[STORAGE_PATH]"; then store the path name "[STORAGE_PATH]/[
INSTITUTION_AND_YEAR]/[INSTITUTION_AND_YEAR]_{document_type}_{document_title}.tex" into the
"Reference_documents" field of the json dictionary at [QAR_TRIPLES_PATH].
## Key information required for writing the LaTeX financial document is given below:
### [NUM_PAIRS] sets of <Question, Answer, Relevant_passage> data:
[TO_BE_FILLED]
## Now please begin calling the "[SKILL_NAME]" to write the LaTeX document.
22

A.6 Government Fiscal Bulletin Generation
A5 | Government Fiscal Bulletin Generation (English)
PurposeGenerate government fiscal bulletins in LaTeX for OfficeQA announcement items.
InputOne<Question, Answer, Relevant_passage>triple per invocation
OutputLaTeX bulletin,layoutCoT, andReference_documents
Suppose you are a senior financial advisor, skilled in analyzing financial industry intelligence
and writing government fiscal bulletins in PDF format. This time, by providing several
sets of <Question, Answer, Relevant_passage> content combinations from the financial
professional domain, I require you to use the [SKILL_NAME] skill and follow the given
requirements to write a government fiscal bulletin in LaTeX syntax that reflects the gist
of the Relevant_passage. Please ensure that for each Question, the corresponding Answer can
be obtained from the generated document through in-depth investigation or calculation.
## Specific requirements for writing the content are given below:
1. Based on the requirements of the Question and the instructions in the [SKILL_NAME] skill, use
your knowledge base to organize the Relevant_passage content into a LaTeX government
fiscal bulletin. Note that you must ensure the generated report does not affect the
correctness of the Answer;
2. Do not generate explanations or LaTeX formulas for financial professional terms or
quantitative terms involved in the Question or SubQuestion;
3. The written content should be publicly accessible; do not add confidentiality clauses or
classified statements.
## Please check the generated content to ensure that the Answer cannot be directly/instantly
obtained from the document.
## For each Question, you must generate a prompt for solving the problem based on the written
document content, i.e., form a problem-solving path thought chain composed of a series of
statements in the format "how to obtain XXXX from XXXX paragraph/table/form", and store
this field with the name "layoutCoT" in the json dictionary at [QAR_TRIPLES_PATH].
## The generated document naming convention is: "[UID]_government_fiscal_bulletin_{
document_title}.tex". Please name the document by referencing the relevant information, and
the storage path is: "[STORAGE_PATH]"; then store the path name "[STORAGE_PATH]/[UID]/[UID
]_government_fiscal_bulletin_{document_title}.tex" into the "Reference_documents" field of
the json dictionary at [QAR_TRIPLES_PATH].
## Key information required for writing the LaTeX government fiscal bulletin is given below:
### One set of <Question, Answer, Relevant_passage> data:
[TO_BE_FILLED]
## Now please begin calling the [SKILL_NAME] to write the LaTeX document.
Part B — Question Processing and Quality Control
A.7 Question Rewriting and Metadata Extraction
B1 | Question Rewriting and Metadata Extraction
Purpose Rewrite FinanceBench questions and extract company, period, and document type
metadata.
InputRawquestionstring
OutputJSON withrewrite_question,company,period,doc_type
Analyze the input question, extract an institution or company named entity, the year or month of
interest in the question, and the document type of interest in the question (use "10K" for
annual reports, "10Q" for quarterly reports, "8K" for material event interim reports, and
23

"EARNINGS" for financial and earnings reports). Then rewrite the input question according
to the following rules:
1. Retain the main interrogative part of the question, the required calculation units, and the
required decimal places;
2. Remove query-target guidance before and after the question;
3. Remove explanations of financial concepts or financial terms.
## Below are several examples of question rewriting:
question 1: How much was Boeing's FY2017 total interest expense (in USD thousands)? Calculate
what was asked by utilizing the line items clearly shown in the statement of income.
rewrite_question 1: How much was Boeing's FY2017 total interest expense (in USD thousands)?
------
question 2: Considering the data in the P&L statement, what is 3M's FY2016 unadjusted operating
income (in USD millions)?
rewrite_question 2: what is 3M's FY2016 unadjusted operating income (in USD millions)?
------
question 3: What is FY2019 working capital ratio for Costco? Define working capital ratio as
total current assets divided by total current liabilities. Round your answer to two decimal
places. Give a response to the question by relying on the details shown in the statement
of financial position.
rewrite_question 3: What is FY2019 working capital ratio for Costco? Round your answer to two
decimal places.
## Organize the extracted fields and rewritten question into the following plain-text JSON
output. Do not include any code block markers (such as```json or```):
{
"question": XXXX,
"rewrite_question": XXXX,
"company": XXXX,
"period": XXXX,
"doc_type": XXXX
}
## Next, input the question:
[<question>]
## Analyze the question and return the processed result as required:
A.8 Complex Question Synthesis
B2 | Complex Question Synthesis
Purpose Merge 2–4 sub-questions into one complex merged question with task typing and
reasoning trace.
InputA batch of rewritten questions, optionally paired with answers
Output JSON array of merged_question ,sub_questions ,sub_questions_id ,task_type ,
reasoning
Pipeline note.In the full pipeline, this prompt is invoked in two input variants: (i) with sub-questions and their answers
as<question, answer> pairs, or (ii) with sub-questions only as <question> entries. Clustering by company or period
is handled outside the prompt. Rule 2 below always references “sub-questions and their answers”; when answers are
omitted from the input, they are retrieved separately in a prior pipeline step.
Rewrite a set of input questions according to the following rules:
1. Select 2-4 financial questions of the same type or similar domain as sub-questions, and merge
and rewrite them into one complex new question. The new question should preferably be one
sentence, with natural and fluent expression, and must not simply list sub-questions;
2. The answer to the new question must be clearly obtainable through reasoning or calculation
from the sub-questions and their answers;
3. According to the answer type and task type, the new question can be classified as: numerical
comparison, explicit reasoning, implicit reasoning, multi-hop judgment, summary, planning,
etc.;
24

4. The new question should unify the calculation unit requirements and required decimal places
of the sub-questions;
5. Note: all question data must be processed and generated in English.
## Below are the approaches and categories for rewriting new questions:
1. Numerical comparison: a new question type rewritten by comparing sub-questions and their
targets through answers, finding maximum/minimum values, differences, multiples, etc.;
2. Reasoning: a new question type constructed by combining multiple sub-questions through
merging, nesting, summing/averaging answers, etc., requiring multi-step calculation to
solve;
3. Implicit reasoning: a reasoning-type new question in which sub-questions contain financial
concepts or professional financial terms, and solving requires querying relevant knowledge
or formulas;
4. Multi-hop judgment: a judgment-type new question constructed by conditionally combining
multiple sub-questions, requiring multi-step reasoning to solve;
5. Summary: a new question type constructed by combining multiple sub-questions of the same kind
, requiring extraction of key information and summarization of reference documents to solve;
6. Planning: a new question type such as event prediction, investment advice, or development
decisions, constructed based on the objectives or answers of sub-questions.
## Name the synthesized new question "merged_question"; place the selected sub-questions from
the input in the "sub_questions" list field, and record the corresponding indices of sub-
questions in the input in "sub_questions_id"; explain the generation and answer-solving
approach of "merged_question" in English and record it in "reasoning"; organize the
generated result into the following plain-text JSON array output. Do not include any code
block markers (such as```json or```):
[
{
"merged_question": XXXX,
"sub_questions_id": [],
"sub_questions": [],
"task_type": XXXX,
"reasoning": XXXX
},
...
]
## Next, input a set of questions and corresponding answers:
[
<question, answer>,
<question, answer>,
...
]
## Please analyze the input content and generate as many results as possible according to the
requirements while ensuring quality:
When answers are omitted, the input block uses question entries only; all other instructions remain unchanged.
A.9 Merged Question Solving
B3 | Merged Question Solving
Purpose Solve a synthesized merged question using sub-questions, sub-answers, and a reasoning
trace.
Inputmerged_question, sub-questions with answers, andreasoning
Output Runnable Python code (numerical tasks) or JSON with thinking andanswer (logical
tasks)
You are a knowledgeable and helpful financial domain expert. I now have a Question composed of
at least 2 sub-questions. All sub-questions and their standard answers, as well as the
25

overall approach for synthesizing this Question, are known.
## Your task
First read and understand the Question and Reasoning, and determine the type of the Question.
Questions are divided into two categories: numerical calculation and logical reasoning. The
former aims to obtain answers by using relevant numerical values for calculation or
comparison; the latter cannot rely solely on numerical calculation, but requires reasoning
or summarization based on textual content to obtain answers.
Then, please combine the input sub-questions and sub-answers to generate a solution for the
Question.
1. For numerical calculation Questions, generate runnable Python code to solve them. Code
requirements: concise and clear, directly runnable, meaningful variable names, necessary
input data definitions included, code comments and final output included.
2. For logical reasoning Questions, generate a clear step-by-step thinking process, then
generate the answer, and return in the following plain-text JSON format. Do not include any
code block markers (such as```json or```):
{
"thinking": generate thinking process in English,
"answer": generate answer in English
}
Below I provide this information to you. Please help me answer the Question as required.
## Input information
Question: [<full question text>]
SubQuestions_and_Answers:
[
{subquestion: [<sub-question text>], answer: [<sub-question answer>]},
{subquestion: [<sub-question text>], answer: [<sub-question answer>]},
...
]
Reasoning: [<approach for synthesizing the Question>]
## Please solve the Question as required. Note: do not question the correctness of the sub-
question answers
A.10 Answer–Passage Consistency Checking
B4 | Answer–Passage Consistency Checking
PurposeVerify whether a manually annotated answer is supported by the relevant passage.
Inputquestion,relevant_passage,answer
Output JSON with thinking andjudgement∈ {Fully Consistent, Partially Consistent, Incon-
sistent}
You are a knowledgeable and helpful financial domain expert. I now have a financial Question.
Based on the Question, I found some relevant document passages (Relevant_passage) and
manually annotated an answer (Answer). I now provide this information to you. Please help
me check whether the Answer is consistent with the content described in Relevant_passage
based on the Question.
## Your task:
Read and understand the Question, Relevant_passage, and Answer. Think about and judge whether
the Answer conforms to Relevant_passage. First generate concise and clear judgment basis,
then give a judgment conclusion (only one of'Fully Consistent','Partially Consistent', or
'Inconsistent'). Return in the following plain-text JSON format. Do not include any code
block markers (such as```json or```):
{
"thinking": generate thinking process in English,
"judgement": give judgment result
}
Below I provide the information to you. Please help me make the judgment as required.
## Input information:
26

Question: [<question>]
Relevant_passage:
[<relevant_passage>]
Answer: [<answer>]
## Please complete the judgment task as required:
27