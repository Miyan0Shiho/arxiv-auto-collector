# Cited but Not Verified: Parsing and Evaluating Source Attribution in LLM Deep Research Agents

**Authors**: Hailey Onweller, Elias Lumer, Austin Huber, Pia Ramchandani, Vamse Kumar Subbiah, Corey Feld

**Published**: 2026-05-07 17:46:45

**PDF URL**: [https://arxiv.org/pdf/2605.06635v1](https://arxiv.org/pdf/2605.06635v1)

## Abstract
Large language models (LLMs) power deep research agents that synthesize information from hundreds of web sources into cited reports, yet these citations cannot be reliably verified. Current approaches either trust models to self-cite accurately, risking bias, or employ retrieval-augmented generation (RAG) that does not validate source accessibility, relevance, or factual consistency. We introduce the first source attribution evaluation framework that uses a reproducible AST parser to extract and evaluate inline citations from LLM-generated Markdown reports at scale. Unlike methods that verify claims in isolation, our framework closes the loop by retrieving the actual cited content, enabling human or model evaluators to judge each citation against its source. Citations are evaluated along three dimensions. (1) Link Works verifies URL accessibility, (2) Relevant Content measures topical alignment, and (3) Fact Check validates factual accuracy against source content. We benchmark 14 closed-source and open-source LLMs across three evaluation dimensions using rubric-based LLM-as-a-judge evaluators calibrated through human review. Our results reveal that even the strongest frontier models maintain link validity above 94% and relevance above 80%, yet achieve only 39-77% factual accuracy, while fewer than half of open-source models successfully generate cited reports in a one-shot setting. Ablation studies on research depth show that Fact Check accuracy drops by approximately 42% on average across two frontier models as tool calls scale from 2 to 150, demonstrating that more retrieval does not produce more accurate citations. These findings reveal a critical disconnect between surface-level citation quality and factual reliability, and our framework provides the evaluation infrastructure to assess the disconnect.

## Full Text


<!-- PDF content starts -->

Preprint.
Cited but Not Verified: Parsing and Evaluating
Source Attribution in LLM Deep Research Agents
Hailey Onweller*Elias Lumer Austin Huber
Pia Ramchandani Vamse Kumar Subbiah Corey Feld
Commercial Technology and Innovation Office, PricewaterhouseCoopers, U.S.
Abstract
Large language models (LLMs) power deep research agents that synthesize
information from hundreds of web sources into cited reports, yet these cita-
tions cannot be reliably verified. Current approaches either trust models to
self-cite accurately, risking bias, or employ retrieval-augmented generation
(RAG) that does not validate source accessibility, relevance, or factual con-
sistency. We introduce the first source attribution evaluation framework
that uses a reproducible AST parser to extract and evaluate inline citations
from LLM-generated Markdown reports at scale. Unlike methods that
verify claims in isolation, our framework closes the loop by retrieving the
actual cited content, enabling human or model evaluators to judge each
citation against its source. Citations are evaluated along three dimensions.
(1)Link Worksverifies URL accessibility, (2)Relevant Contentmeasures topi-
cal alignment, and (3)Fact Checkvalidates factual accuracy against source
content. We benchmark 14 closed-source and open-source LLMs across
three evaluation dimensions using rubric-based LLM-as-a-judge evaluators
calibrated through human review. Our results reveal that even the strongest
frontier models maintain link validity above 94% and relevance above 80%,
yet achieve only 39–77% factual accuracy, while fewer than half of open-
source models successfully generate cited reports in a one-shot setting.
Ablation studies on research depth show that Fact Check accuracy drops
by approximately 42% on average across two frontier models as tool calls
scale from 2 to 150, demonstrating that more retrieval does not produce
more accurate citations. These findings reveal a critical disconnect between
surface-level citation quality and factual reliability, and our framework
provides the evaluation infrastructure to assess the disconnect.
1 Introduction
Deep research agents powered by Large Language Models (LLMs) now synthesize infor-
mation from hundreds of web sources into comprehensive reports with inline citations,
enabling systems such as Perplexity AI, ChatGPT with web search, and Google Gemini
to promise verifiable research at scale (Perplexity AI, 2024; OpenAI, 2024). However, the
citations these agents produce cannot be reliably verified. Current approaches either trust
models to self-cite accurately, risking hallucinated or misattributed references (Ravichander
et al., 2025), or employ retrieval-augmented generation (RAG) that does not validate whether
cited sources are accessible, topically relevant, or factually consistent with the claims they
support (Lewis et al., 2020; Lumer et al., 2025a; Gulati et al., 2026b). Recent studies have
documented citation hallucination rates ranging from 11% to 57% across commercially
deployed models (Yuan et al., 2026), yet no existing framework evaluates citation quality
beyond binary attribution verification. This gap between deployment scale and evaluation
rigor raises a critical question.How reliable are the citations that millions of users encounter daily
in LLM-generated research?
Despite growing recognition of this problem, existing evaluation approaches remain insuf-
ficient. Benchmarks such as AttributionBench (Li et al., 2024), CiteME (Press et al., 2024),
and CiteEval (Xu et al., 2025) focus on binary attribution classification, citation matching,
1arXiv:2605.06635v1  [cs.CL]  7 May 2026

Preprint.
or paradigm comparison, while CiteAudit (Yuan et al., 2026) targets fabricated references
in scientific writing. Three critical gaps persist. (1) No end-to-end framework combines
citation extraction with multi-dimensional quality assessment across URL accessibility,
topical relevance, and factual accuracy, (2) no systematic comparison exists across major
LLM providers in deep research settings, and (3) the relationship between search depth and
citation quality remains unexplored. These gaps motivate the following research questions.
To what extent do frontier LLMs produce factually accurate citations, and does
surface-level citation quality (link validity, topical relevance) mask deeper factual
failures? How does increasing search depth affect citation quality across these
dimensions?
To address these questions, we introduce a source attribution evaluation framework that
extracts and evaluates inline citations from LLM-generated Markdown reports at scale. As
illustrated in Figure 1, the framework operates as a three-stage pipeline. First, a Markdown
Abstract Syntax Tree (AST) parser structurally extracts citation-claim pairs without requiring
LLM inference. Second, each cited URL is retrieved and its content extracted. Third, three
complementary evaluators assess citation quality. (1)Link Worksverifies URL accessibility,
(2)Relevant Contentmeasures topical alignment via LLM-as-a-judge, and (3)Fact Check
validates factual accuracy against retrieved source content via LLM-as-a-judge calibrated
through human review.
We benchmark 14 LLMs spanning open-source and closed-source frontier models across
diverse research queries. Our evaluation reveals three key findings. First, even the strongest
frontier models maintain link validity above 94% and content relevance above 80%, yet
achieve only 39–77% factual accuracy, exposing a critical disconnect between surface-level
citation quality and factual reliability. Second, fewer than half of open-source models
successfully generate cited reports (17–40% task success), compared to 83–100% for frontier
models. Third, our ablation study across two models demonstrates that Fact Check accuracy
drops approximately 42% on average as search depth scales from 2 to 150 tool calls, while
Link Works and Relevant Content remain stable, suggesting that information overload
impairs factual synthesis rather than improving it.
We make the following contributions:
•We introduce the first end-to-end source attribution evaluation framework that com-
bines deterministic citation extraction with multi-dimensional quality assessment
across link accessibility, topical relevance, and factual accuracy.
•We benchmark 14 LLMs across three major providers and open-source alternatives,
revealing that factual accuracy is the most challenging and differentiating dimension
of citation quality.
•We demonstrate through ablation studies that increased search depth consistently
degrades factual accuracy while surface-level citation metrics remain stable, provid-
ing evidence for an information overload effect in LLM research synthesis.
2 Related work
2.1 Attributed generation and deep research systems
LLMs can generate responses with inline citations through web search integration and
retrieval-augmented generation (RAG) (Lewis et al., 2020). Commercial deep research sys-
tems such as Perplexity AI, ChatGPT with web search, and Google Gemini with grounding
have made citation-generating LLMs widely accessible (Perplexity AI, 2024; OpenAI, 2024;
Lumer et al., 2026; 2025d). Model providers have also introduced API-level citation support,
such as Anthropic’s Citations API, which returns structured references with character-level
provenance to source documents (Anthropic, 2025). However, these features focus on cita-
tiongenerationrather than citationevaluation. Research on attributed generation has explored
both generation-time approaches that produce citations during output (Gao et al., 2023b;
Slobodkin et al., 2024) and post-hoc methods that add citations after text generation (Gao
2

Preprint.
Figure 1: Source attribution evaluation framework. A deep research agent generates Mark-
down reports with inline citations, which are parsed via a Markdown AST parser to extract
citation-claim pairs. Each pair is evaluated on Link Works (URL accessibility), Relevant
Content (topical alignment), and Fact Check (factual accuracy).
et al., 2023a). Wan et al. (2025) introduce a modular framework that decomposes attributed
generation into executable programs, achieving fine-grained sentence-level attribution.
Wang et al. (2025b) propose a unified generation-retrieval model for academic writing
that dynamically triggers citation retrieval during text generation. Despite these advances,
citation hallucination rates of 11–57% persist across commercially deployed models (Yuan
et al., 2026), and citations may link to inaccessible URLs, reference irrelevant content, or
misrepresent source claims (Ravichander et al., 2025). Unlike these works, which focus
on improving citation generation, our work focuses on evaluating the quality of citations
already produced by deployed systems.
2.2 Attribution evaluation benchmarks
Several benchmarks have been developed for evaluating LLM-generated attributions. Attri-
butionBench (Li et al., 2024) provides a benchmark for automatic binary attribution classifica-
3

Preprint.
tion, finding that even fine-tuned GPT-3.5 achieves only 80% macro-F1. CiteME (Press et al.,
2024) evaluates citation matching in academic contexts, revealing that LLMs achieve only
4–18% accuracy on identifying correct papers to cite. CiteEval (Xu et al., 2025) goes beyond
NLI-based approaches by conducting fine-grained citation assessment across full retrieval
contexts. CiteGuard (Choi et al., 2025) reframes citation evaluation as attribution alignment,
achieving 68% accuracy on CiteME through retrieval-aware verification. RefChecker (Hu
et al., 2024) presents a claim-triplet framework for fine-grained hallucination detection,
while CiteAudit (Yuan et al., 2026) addresses fabricated references in scientific writing
through multi-agent verification. Most recently, Saxena et al. (2025) compare generation-
time versus post-hoc citation paradigms, finding a consistent trade-off between coverage
and correctness. Seo et al. (2025) evaluate fact verifiers across 14 benchmarks and find that
annotation ambiguity substantially affects model rankings, highlighting the importance
of careful benchmark design for evaluation tasks. While these works advance attribution
evaluation, they primarily address binary verification, citation matching, or paradigm com-
parison. Our framework differs by evaluating citations across three complementary quality
dimensions simultaneously and by targeting live web citations in deep research contexts
rather than controlled document sets.
2.3 LLM-as-a-judge for evaluation
The use of LLMs as automated evaluators has gained significant attention as an alternative
to costly human annotation (Zheng et al., 2023). However, research has revealed system-
atic biases in LLM judges, including position bias, verbosity bias, and self-enhancement
effects (Wang et al., 2024; Ye et al., 2025). Wang et al. (2025a) demonstrate that even Large
Reasoning Models remain susceptible to evaluation biases despite advanced reasoning
capabilities. These findings are directly relevant to our framework, which relies on LLM-
as-a-judge evaluators for the Relevant Content and Fact Check dimensions. To mitigate
potential judge biases, we calibrate our evaluators through human review and employ
rubric-based scoring that constrains evaluation to specific factual criteria rather than open-
ended quality assessment. Our work extends the LLM-as-a-judge paradigm from general
text evaluation to the specific task of source attribution verification, where the evaluator
must assess whether a claim is supported by retrieved source content rather than judging
overall response quality.
3 Methodology
3.1 Overview
Our framework evaluates source attribution quality in LLM-generated research responses
through a three-stage pipeline (Figure 1). First, a deep research agent generates a compre-
hensive Markdown report with inline citations for a given query. Second, a Markdown
Abstract Syntax Tree (AST) parser structurally extracts citation-claim pairs without requiring
LLM inference. Third, three complementary evaluators assess each citation along distinct
quality dimensions. The pipeline processes attributions independently at the sentence
level, enabling fine-grained analysis rather than document-level assessment. This modular
design enables evaluation of any LLM capable of generating Markdown responses with
citations, requires no modification to the underlying model, and scales to thousands of
cited pages. In Algorithm 1, the deep research agent’s Markdown output is parsed into an
AttributionDocument of citation-claim pairs, and the evaluation runner then fetches each
cited source and scores every pair across the three dimensions.
3.2 Markdown AST parser
The parser transforms raw LLM-generated Markdown into structured citation-claim pairs
by constructing an abstract syntax tree and traversing it to extract citation nodes. Given a
Markdown document, it produces an AttributionDocument containing a deduplicated list
of citations (URLs), a list of attributions (text spans with associated citation references), and
placeholder structures for evaluation results.
4

Preprint.
Algorithm 1Source Attribution Evaluation Pipeline
Require:Research queryq, deep research agent with web search
Ensure:AttributionDocumentDwithcitations,attributions,evals
Phase 0: Report Generation
1:D←DEEPRESEARCHAGENT(q)▷Markdown report with inline citations
Phase 1: Markdown AST Parsing
2:D′←CANONICALIZE(D)▷Normalize whitespace, strip code blocks
3:T←BUILDAST(D′)▷Construct abstract syntax tree
4:C ←EXTRACTCITATIONS(T)▷Deduplicatedurl,raw labels
5:S ←SENTENCESEGMENT(T)▷Split into individual claims
6:A ←BACKWARDATTRIBUTE(S,C)▷text nocite,span,citation ids
7:D ←AttributionDocument(citations=C,attributions=A)
Phase 2: Evaluation Runner
8:for allunique citationc∈ Cin parallel do
9:c.url content←FETCH(c.url)▷Web content extraction
10:end for
11:for allattribution-citation pair(a i,cj)∈ Din parallel do
12:s 1←LINKWORKS(c j.url)▷HTTP accessibility
13:s 2←RELEVANTCONTENT(a i.text nocite,c j.url content)▷LLM-as-a-judge
14:s 3←FACTCHECK(a i.text nocite,c j.url content)▷LLM-as-a-judge
15:D.evals← D.evals∪ {(a i.id,c j.id,[s 1,s2,s3])}
16:end for
17:returnD
Parsing proceeds in four stages. Canonicalization normalizes line endings and whitespace.
Code block removal strips fenced code sections to prevent false citation matches. AST
construction and traversal identifies citation nodes across multiple formats, including
numbered references ( [1],[2]), footnote-style references ( [ˆnote] ), inline Markdown links
([text](url) ), autolinks ( <url> ), and ranges ( [1-3] expanding to individual references).
Registry building creates a deduplicated citation list with normalized URLs.
The parser performs sentence-level segmentation to split continuous text into individual
claims. It implements backward attribution logic. When a citation appears at the end of a
passage, it applies to all preceding uncited sentences in that passage. This design reflects
common LLM citation practices, where a single reference supports multiple related claims.
The final output associates each text span with its citation identifiers, enabling independent
evaluation of each claim-citation pair.
3.3 Evaluation dimensions
The framework evaluates citations across three complementary dimensions, each capturing
a distinct aspect of attribution quality. These dimensions form an increasing hierarchy of
verification difficulty.
3.3.1 Link Works
Link Works assesses URL accessibility without LLM inference. For each cited URL, a web
content extractor capable of handling JavaScript-rendered pages retrieves the content. The
evaluator produces a binary score of 1 if the URL returns accessible content and 0 if the
request fails due to HTTP errors (404, 403), timeouts, or blocked access. This dimension
identifies broken links, paywalled content, and URLs removed since the research was
conducted.
3.3.2 Relevant Content
Relevant Content measures topical alignment between the claim and the cited source using
an LLM-as-a-judge approach. Given the attribution text and retrieved source content
5

Preprint.
(truncated to 5,000 characters), the evaluator determines whether the source addresses the
same topic as the claim. The evaluator produces a binary score with a natural language
explanation. This dimension identifies citations that link to valid URLs but reference content
unrelated to the claim’s subject matter.
3.3.3 Fact Check
Fact Check verifies whether specific factual claims are accurately supported by the source
content. Using an LLM-as-a-judge approach, the evaluator examines facts, numbers, dates,
and assertions in the attribution text against the retrieved source. The evaluator produces
a binary score of 1 if the facts are supported or consistent and 0 if they are contradicted,
absent, or uncertain. This dimension represents the most stringent evaluation, identifying
citations where the source exists and is topically relevant but does not support the specific
claims attributed to it. To ensure alignment with human judgment, the Fact Check evaluator
was calibrated through manual review of 50–100 LLM judgments.
3.4 Experimental setup
Models and queries.We evaluate 14 LLMs spanning three major providers and open-source
alternatives on 130 research queries drawn from DeepResearch Bench (Du et al., 2025) and
BrowseComp (Wei et al., 2025). The models include OpenAI (GPT-5.2, GPT-5.4, GPT-5 Mini,
Codex), Anthropic (Claude Sonnet 4.5, Claude Sonnet 4.6, Claude Opus 4.5, Claude Opus
4.6, Claude Haiku 4.5), Google (Gemini 3.1 Pro, Gemini 3 Flash), and three open-source
models (OSS-120B, Llama 4 Maverick, Pixtral Large). Queries cover diverse topics requiring
multi-source synthesis.
Evaluation protocol.For each query, a deep research agent with web search capabilities
generates a Markdown response with inline citations. The agent is configured with a system
prompt enforcing citation format requirements and minimum search depth. Evaluation runs
asynchronously with concurrency limits (10 concurrent agents, 15 concurrent evaluators)
and retry logic (5 retries with 5-second delays) to handle transient failures.
Ablation study.To analyze the relationship between search depth and attribution quality,
we vary the maximum number of tool calls across two models (GPT-5.4, Claude Opus 4.6)
at seven intervals (2, 10, 30, 50, 70, 100, and 150). This controlled experiment isolates the
effect of search depth from model capability differences.
4 Evaluation
We report results organized around three key findings that address our research questions.
Section 4.1 examines the disconnect between surface-level citation metrics and factual
accuracy, Section 4.2 analyzes provider-level patterns in citation quantity versus quality,
and Section 4.3 investigates the effect of search depth on attribution quality.
4.1 Finding 1. Surface-level citation quality masks factual failures
Table 1 presents evaluation results across all 14 models.
High link validity and relevance coexist with low factual accuracy. Across all models, 12
of 14 exceed 94% on Link Works and all frontier models exceed 80% on Relevant Content.
However, Fact Check scores range from 24% (OSS-120B) to 77% (Claude Opus 4.5), a 53%
spread that makes factual accuracy the most differentiating dimension. This pattern means
that a user encountering a citation in an LLM-generated report will almost always find a
working link to a topically relevant page, yet the specific factual claims attributed to that
source may be unsupported nearly half the time.
Open-source models struggle with the citation generation task itself. Fewer than half
of open-source models successfully generate cited reports. OSS-120B achieves 40% task
success, Llama 4 Maverick 30%, and Pixtral Large only 17%, compared to 83–100% for
frontier models. These models also make substantially fewer tool calls (7–60 versus 73–211
6

Preprint.
Table 1: Source attribution quality across 14 LLMs ordered by Relevant Content. Success
Rate indicates percentage of queries producing valid citations.Boldindicates best per
metric.
Model Success Link Works Relevant Fact Check
Claude Opus 4.5 90.0% 98.7%95.7% 76.8%
GPT-5.4100.0% 100.0%93.7% 47.7%
GPT-5.2100.0%98.3% 92.3% 58.8%
Codex100.0%96.9% 91.9% 54.1%
Claude Haiku 4.5 83.3% 98.9% 91.1% 68.9%
Claude Sonnet 4.6 93.3% 99.2% 89.8% 58.7%
Claude Sonnet 4.5 96.7% 98.9% 88.3% 51.8%
GPT-5 Mini100.0%99.3% 87.4% 38.9%
Claude Opus 4.6 93.3% 97.2% 83.9% 54.2%
Gemini 3 Flash100.0%94.7% 82.9% 45.2%
Gemini 3.1 Pro 90.0% 94.1% 80.7% 48.5%
OSS-120B 40.0% 83.9% 68.7% 24.4%
Pixtral Large 16.7%100.0%64.9% 51.4%
Llama 4 Maverick 30.0% 80.8% 60.6% 34.3%
for frontier models), suggesting that limited search capabilities compound with weaker
generation abilities.
4.2 Finding 2. Citation quantity trades off against factual accuracy
Providers that generate more citations achieve lower factual accuracy. Performance pat-
terns differ substantially across providers. OpenAI models achieve 100% task success and
generate the most citations (GPT-5 Mini, 1,272 total attributions), but their Fact Check accu-
racy spans only 39–59%. Anthropic models show lower task success (83–97%) but excel in
factual accuracy, with Claude Opus 4.5 achieving 77% Fact Check and Claude Haiku 4.5
achieving 69%. Google Gemini models occupy a middle ground (45–49% Fact Check). This
inverse relationship between citation quantity and quality suggests that current LLMs face
a fundamental trade-off between research thoroughness and factual reliability.
We hypothesize that this trade-off arises from attention dilution during synthesis. Models
generating more citations must aggregate information from a larger number of retrieved pas-
sages, increasing the likelihood of misattribution or conflation of facts across sources (Lumer
et al., 2025b). Notably, Claude Opus 4.5 achieves the highest Fact Check score despite a
lower task success rate than any OpenAI model, suggesting that selective citation may be a
more effective strategy than exhaustive citation for maintaining factual accuracy.
4.3 Finding 3. More search degrades factual accuracy
To directly test the information overload hypothesis, we conducted an ablation study across
two models at seven search depth intervals (2–150 tool calls). Tables 2–3 and Figure 2 present
the results.
Factual accuracy generally degrades with search depth while surface metrics remain sta-
ble. Across both models, Fact Check accuracy drops approximately 42% on average from
minimal (2 calls) to maximal search depth (Figure 2). GPT-5.4 shows the steepest decline,
from 79% to 17% (62%). Claude Opus 4.6 demonstrates the greatest resilience, declining
from 80% to 58% (22%). Critically, Link Works and Relevant Content remain above 92% at
all search depths, indicating that the degradation is specific to factual synthesis rather than
source selection.
This asymmetric degradation pattern provides strong evidence for an information overload
effect in LLM research synthesis (Lumer et al., 2025c). Models can consistently identify
and cite accessible, topically relevant sources regardless of search depth, but accurately
7

Preprint.
Table 2: GPT-5.4 attribution quality across search depths.Boldindicates best performance.
Tool Calls Link Works Relevant Fact Check
2 100.0% 100.0%78.6%
10 100.0% 99.0% 45.9%
30 98.5% 97.8% 43.0%
50 98.6% 96.5% 38.0%
70 100.0% 99.1% 35.5%
100 97.7% 95.3% 37.2%
150 99.2% 99.2% 16.7%
Table 3: Claude Opus 4.6 attribution quality across search depths.Boldindicates best
performance.
Tool Calls Link Works Relevant Fact Check
2 100.0% 100.0%80.0%
10 92.3% 92.3% 74.4%
30 100.0% 100.0% 69.2%
50 98.0% 98.0% 61.2%
70 100.0% 97.9% 61.7%
100 100.0% 100.0% 58.7%
150 100.0% 100.0% 57.9%
synthesizing factual claims becomes increasingly difficult as the number of sources grows.
The sharpest Fact Check decline occurs between 2 and 10 tool calls for GPT-5.4 (79% to
46%, a 33% drop), suggesting that even modest increases in source volume can overwhelm
factual synthesis capabilities.
4.4 Error analysis
Link Works failures stem from three categories, including HTTP 404 errors (content removed
or URL changed), blocked access (paywalls, bot detection), and connection timeouts. GPT-
5.4 demonstrated the highest link reliability with only 1 failed link out of 2,159 evaluations,
while open-source models showed substantially higher failure rates (Llama 4 Maverick
at 19.2%, OSS-120B at 16.1%). Rate limit errors (HTTP 429) had minimal impact across
all models, affecting fewer than 0.3% of evaluations with differences of less than 0.5% in
adjusted pass rates.
5 Limitations
Our framework has several limitations that present opportunities for future work. First,
the LLM-as-a-judge approach for Relevant Content and Fact Check evaluations, despite
calibration through human review, may retain biases inherent to the judge model, including
position bias and self-enhancement effects (Wang et al., 2024; Zheng et al., 2023). Ensemble
judging with multiple LLM evaluators or hybrid approaches combining LLM judges with
rule-based verification could mitigate single-model biases.
Second, web citations are temporally unstable. URLs that were accessible during evalua-
tion may become unavailable due to content removal, domain expiration, or access policy
changes, and source content itself may change. This temporal instability affects reproducibil-
ity and motivates longitudinal studies tracking citation persistence over time (Sen et al.,
2026). Deploying LLM-generated research in high-stakes domains such as healthcare, legal,
or financial analysis without robust citation verification could propagate factual errors,
underscoring the importance of frameworks like ours as a safeguard.
8

Preprint.
Figure 2: Fact Check accuracy degradation as search depth increases. Both models show
declining factual accuracy with increased tool calls, while Link Works and Relevant Content
remain stable (above 92%). GPT-5.4 exhibits the steepest decline (79% to 17%).
Finally, the evaluation is limited to models with web search capabilities, excluding enter-
prise RAG deployments that cite internal document corpora. Extending the framework to
evaluate citation quality in private knowledge bases (Gulati et al., 2026a) would broaden
applicability to the growing ecosystem of enterprise AI assistants.
6 Conclusion
In this work, we introduce a source attribution evaluation framework that extracts and eval-
uates inline citations from LLM-generated deep research reports across three dimensions of
link accessibility, topical relevance, and factual accuracy. Our evaluation of 14 LLMs reveals
a critical disconnect between surface-level citation quality and factual reliability. Models
consistently produce working links to relevant pages, yet factual accuracy remains the
weakest dimension across all providers. We further demonstrate an information overload
effect, where increased search depth degrades factual accuracy while surface metrics remain
stable, providing evidence that more retrieval does not produce more accurate citations.
These findings have implications for both system designers and end users of deep research
agents. For system designers, our results suggest that citation quality monitoring should
be integrated into agent pipelines, and that retrieval strategies should prioritize depth of
source understanding over breadth of source coverage. For end users, the high link validity
and topical relevance scores may create a false sense of trust that our framework can help
calibrate. We hope that future work will extend this evaluation infrastructure to domain-
specific research tasks, enterprise RAG deployments, and longitudinal studies of citation
persistence, advancing the systematic assessment of attribution quality as LLM-generated
research becomes increasingly prevalent.
9

Preprint.
Ethics Statement
This work evaluates the factual reliability of citations produced by commercially deployed
LLM-based deep research agents. Our findings that surface-level citation quality masks
factual failures have direct implications for users who rely on these systems for research,
decision-making, and information synthesis. We highlight the risk that high link validity
and topical relevance scores may create a false sense of trust in LLM-generated citations,
particularly in high-stakes domains such as healthcare, legal, and financial analysis where
factual errors can cause tangible harm. Our framework is intended to serve as a safeguard
by providing transparent, reproducible evaluation of citation quality, and we encourage
system designers to integrate similar evaluation mechanisms into deployed research agents.
All models evaluated in this study are publicly accessible, and our evaluation methodology
does not involve human subjects, personally identifiable information, or sensitive data.
References
Anthropic. Introducing citations on the Anthropic API. https://anthropic.com/news/
introducing-citations-api, 2025. Accessed: 2026.
Yee Man Choi, Xuehang Guo, Yi R. Fung, and Qingyun Wang. CiteGuard: Faithful citation
attribution for LLMs via retrieval-augmented validation.arXiv preprint arXiv:2510.17853,
2025.
Mingxuan Du, Benfeng Xu, Chiwei Zhu, Xiaorui Wang, and Zhendong Mao. DeepRe-
search Bench: A comprehensive benchmark for deep research agents.arXiv preprint
arXiv:2506.11763, 2025. ICLR 2026.
Luyu Gao, Zhuyun Dai, Panupong Pasupat, Anthony Chen, Arun Tejasvi Chaganty, Yicheng
Fan, Vincent Zhao, Ni Lao, Hongrae Lee, Da-Cheng Juan, and Kelvin Guu. RARR: Re-
searching and revising what language models say, using language models. InProceedings
of the 61st Annual Meeting of the Association for Computational Linguistics, pp. 16477–16508,
2023a.
Tianyu Gao, Howard Yen, Jiatong Yu, and Danqi Chen. Enabling large language models to
generate text with citations. InProceedings of the 2023 Conference on Empirical Methods in
Natural Language Processing, pp. 6465–6488, 2023b.
Anmol Gulati, Sahil Sen, Waqar Sarguroh, and Kevin Paul. Beyond rows to reasoning:
Agentic retrieval for multimodal spreadsheet understanding and editing.arXiv preprint
arXiv:2603.06503, 2026a.
Anmol Gulati, Sahil Sen, Waqar Sarguroh, and Kevin Paul. From rows to reasoning:
A retrieval-augmented multimodal framework for spreadsheet understanding.arXiv
preprint arXiv:2601.08741, 2026b.
Xiangkun Hu, Dongyu Ru, Lin Qiu, Qipeng Guo, Tianhang Zhang, Yang Xu, Yun Luo,
Pengfei Liu, Yue Zhang, and Zheng Zhang. RefChecker: Reference-based fine-grained
hallucination checker and benchmark for large language models. InProceedings of the 2024
Conference on Empirical Methods in Natural Language Processing, 2024.
Patrick Lewis, Ethan Perez, Aleksandra Piktus, Fabio Petroni, Vladimir Karpukhin, Naman
Goyal, Heinrich K ¨uttler, Mike Lewis, Wen-tau Yih, Tim Rockt ¨aschel, Sebastian Riedel,
and Douwe Kiela. Retrieval-augmented generation for knowledge-intensive NLP tasks.
InAdvances in Neural Information Processing Systems, volume 33, pp. 9459–9474, 2020.
Yifei Li, Xiang Yue, Zeyi Liao, and Huan Sun. AttributionBench: How hard is automatic
attribution evaluation? InFindings of the Association for Computational Linguistics: ACL
2024, 2024.
Elias Lumer, Alex Cardenas, Matt Melich, Myles Mason, Sara Dieter, Vamse Kumar Subbiah,
Pradeep Honaganahalli Basavaraju, and Roberto Hernandez. Comparison of text-based
and image-based retrieval in multimodal retrieval augmented generation large language
model systems.arXiv preprint arXiv:2511.16654, 2025a.
10

Preprint.
Elias Lumer, Anmol Gulati, Vamse Kumar Subbiah, Pradeep Honaganahalli Basavaraju,
and James A Burke. Memtool: Optimizing short-term memory management for dynamic
tool calling in llm agent multi-turn conversations.arXiv preprint arXiv:2507.21428, 2025b.
Elias Lumer, Anmol Gulati, Vamse Kumar Subbiah, Pradeep Honaganahalli Basavaraju,
and James A Burke. Scalemcp: Dynamic and auto-synchronizing model context protocol
tools for llm agents. InInternational Joint Conference on Computational Intelligence, pp. 23–42.
Springer, 2025c.
Elias Lumer, Faheem Nizar, Anmol Gulati, Pradeep Honaganahalli Basavaraju, and
Vamse Kumar Subbiah. Tool-to-agent retrieval: Bridging tools and agents for scalable llm
multi-agent systems.arXiv preprint arXiv:2511.01854, 2025d.
Elias Lumer, Anmol Gulati, Faheem Nizar, Dzmitry Hedroits, Atharva Mehta, Henry
Hwangbo, Vamse Kumar Subbiah, Pradeep Honaganahalli Basavaraju, and James A.
Burke. Tool and agent selection for large language model agents in production: A survey.
Preprints, April 2026. doi: 10.20944/preprints202512.1050.v2. URL https://doi.org/10.
20944/preprints202512.1050.v2.
OpenAI. Introducing ChatGPT search. https://openai.com/index/introducing-chatgpt-
search, 2024. Accessed: 2026.
Perplexity AI. Perplexity AI: Answer engine. https://www.perplexity.ai , 2024. Accessed:
2026.
Ori Press, Andreas Hochlehnert, Ameya Prabhu, Vishaal Udandarao, Ofir Press, and
Matthias Bethge. CiteME: Can language models accurately cite scientific claims? In
Advances in Neural Information Processing Systems, volume 37, 2024. Datasets and Bench-
marks Track.
Abhilasha Ravichander, Shrusti Ghela, David Wadden, and Yejin Choi. HALoGEN: Fantastic
LLM hallucinations and where to find them.arXiv preprint arXiv:2501.08292, 2025. ACL
2025 Outstanding Paper.
Yash Saxena, Raviteja Bommireddy, Ankur Padia, and Manas Gaur. Generation-time vs.
post-hoc citation: A holistic evaluation of LLM attribution.arXiv preprint arXiv:2509.21557,
2025.
Sahil Sen, Elias Lumer, Anmol Gulati, and Vamse Kumar Subbiah. Chronos: Temporal-
aware conversational agents with structured event retrieval for long-term memory.arXiv
preprint arXiv:2603.16862, 2026.
Wooseok Seo, Seungju Han, Jaehun Jung, Benjamin Newman, Seungwon Lim, Seungbeen
Lee, Ximing Lu, Yejin Choi, and Youngjae Yu. Verifying the verifiers: Unveiling pitfalls
and potentials in fact verifiers.arXiv preprint arXiv:2506.13342, 2025.
Aviv Slobodkin, Eran Hirsch, Arie Cattan, Tal Schuster, and Ido Dagan. Attribute first, then
generate: Locally-attributable grounded text generation. InProceedings of the 62nd Annual
Meeting of the Association for Computational Linguistics, 2024.
David Wan, Eran Hirsch, Elias Stengel-Eskin, Ido Dagan, and Mohit Bansal. GenerationPro-
grams: Fine-grained attribution with executable programs.arXiv preprint arXiv:2506.14580,
2025.
Peiyi Wang, Lei Li, Liang Chen, Zefan Cai, Dawei Zhu, Binghuai Lin, Yunbo Cao, Qi Liu,
Tianyu Liu, and Zhifang Sui. Large language models are not fair evaluators. InProceedings
of the 62nd Annual Meeting of the Association for Computational Linguistics, 2024.
Qian Wang, Zhanzhi Lou, Zhenheng Tang, Nuo Chen, Xuandong Zhao, Wenxuan Zhang,
Dawn Song, and Bingsheng He. Assessing judging bias in large reasoning models: An
empirical study.arXiv preprint arXiv:2504.09946, 2025a.
11

Preprint.
Yubo Wang, Xueguang Ma, Ping Nie, Huaye Zeng, Zhiheng Lyu, Yuxuan Zhang, Benjamin
Schneider, Yi Lu, Xiang Yue, and Wenhu Chen. ScholarCopilot: Training large language
models for academic writing with accurate citations.arXiv preprint arXiv:2504.00824,
2025b. COLM 2025.
Jason Wei, Zhiqing Sun, Spencer Papay, Scott McKinney, Jeffrey Han, Isa Fulford,
Hyung Won Chung, Alex Tachard Passos, William Fedus, and Amelia Glaese. Browsec-
omp: A simple yet challenging benchmark for browsing agents, 2025. URL https:
//arxiv.org/abs/2504.12516.
Yumo Xu, Peng Qi, Jifan Chen, Kunlun Liu, Rujun Han, Lan Liu, Bonan Min, Vittorio
Castelli, Arshit Gupta, and Zhiguo Wang. CiteEval: Principle-driven citation evaluation
for source attribution. InProceedings of the 63rd Annual Meeting of the Association for
Computational Linguistics, 2025.
Jiayi Ye, Yanbo Wang, Yue Huang, Dongping Chen, Qihui Zhang, Nuno Moniz, Tian Gao,
Werner Geyer, Chao Huang, Pin-Yu Chen, Nitesh V . Chawla, and Xiangliang Zhang.
Justice or prejudice? quantifying biases in LLM-as-a-judge. InInternational Conference on
Learning Representations, 2025.
Zhengqing Yuan, Kaiwen Shi, Zheyuan Zhang, Lichao Sun, Nitesh V . Chawla, and Yanfang
Ye. CiteAudit: You cited it, but did you read it? A benchmark for verifying scientific
references in the LLM era.arXiv preprint arXiv:2602.23452, 2026.
Lianmin Zheng, Wei-Lin Chiang, Ying Sheng, Siyuan Zhuang, Zhanghao Wu, Yonghao
Zhuang, Zi Lin, Zhuohan Li, Dacheng Li, Eric P . Xing, Hao Zhang, Joseph E. Gonzalez,
and Ion Stoica. Judging LLM-as-a-judge with MT-Bench and Chatbot Arena. InAdvances
in Neural Information Processing Systems, volume 36, 2023.
12