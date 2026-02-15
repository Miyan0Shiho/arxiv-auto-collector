# Whose Name Comes Up? Benchmarking and Intervention-Based Auditing of LLM-Based Scholar Recommendation

**Authors**: Lisette Espin-Noboa, Gonzalo Gabriel Mendez

**Published**: 2026-02-09 16:34:57

**PDF URL**: [https://arxiv.org/pdf/2602.08873v1](https://arxiv.org/pdf/2602.08873v1)

## Abstract
Large language models (LLMs) are increasingly used for academic expert recommendation. Existing audits typically evaluate model outputs in isolation, largely ignoring end-user inference-time interventions. As a result, it remains unclear whether failures such as refusals, hallucinations, and uneven coverage stem from model choice or deployment decisions. We introduce LLMScholarBench, a benchmark for auditing LLM-based scholar recommendation that jointly evaluates model infrastructure and end-user interventions across multiple tasks. LLMScholarBench measures both technical quality and social representation using nine metrics. We instantiate the benchmark in physics expert recommendation and audit 22 LLMs under temperature variation, representation-constrained prompting, and retrieval-augmented generation (RAG) via web search. Our results show that end-user interventions do not yield uniform improvements but instead redistribute error across dimensions. Higher temperature degrades validity, consistency, and factuality. Representation-constrained prompting improves diversity at the expense of factuality, while RAG primarily improves technical quality while reducing diversity and parity. Overall, end-user interventions reshape trade-offs rather than providing a general fix. We release code and data that can be adapted to other disciplines by replacing domain-specific ground truth and metrics.

## Full Text


<!-- PDF content starts -->

Whose Name Comes Up? Benchmarking and Intervention-Based
Auditing of LLM-Based Scholar Recommendation
Lisette Espín-Noboa
Complexity Science Hub
Vienna, Austria
espin@csh.ac.atGonzalo Gabriel Méndez
Universitat Politècnica de València
Valencia, Spain
ggmenco1@upv.es
Baseline conditions
S M LXLModel Size
Reasoning capabilityModel Access
Open-weight Proprietary
Enabled Disabled22 x       LLMs Benchmark Metrics
End-user interventions
Temperature
Control
Constrained
Prompting
RAG with
web searchBase prompt templateRecommendation Tasks
Top-k By field By epoch
By seniority Stats twinsValidity Refusals
DuplicatesConnectedness
Consistency AccuracyTechnical quality Social representation
Parityabc
d eDiversity Similarity
Figure 1:LLMScholarBenchoverview. We evaluate LLM-based scholar recommendation across five tasks (a), three end-user
interventions (b), and 22 LLMs varying in access type, size, and reasoning capability (c). Model outputs are assessed along two
dimensions:technical quality(refusals, validity, duplicates, consistency, accuracy; d) andsocial representation(connectedness,
bibliometric similarity, demographic diversity and parity; e), enabling a systematic analysis of performance trade-offs.
Abstract
Large language models (LLMs) are increasingly used for academic
expert recommendation. Existing audits typically evaluate model
outputs in isolation, largely ignoring end-user inference-time in-
terventions. As a result, it remains unclear whether failures such
as refusals, hallucinations, and uneven coverage stem from model
choice or deployment decisions. We introduceLLMScholarBench, a
benchmark for auditing LLM-based scholar recommendation that
jointly evaluates model infrastructure and end-user interventions
across multiple tasks.LLMScholarBenchmeasures both technical
quality and social representation using nine metrics. We instanti-
ate the benchmark in physics expert recommendation and audit
22 LLMs under temperature variation, representation-constrained
prompting, and retrieval-augmented generation (RAG) via web
search. Our results show that end-user interventions do not yield
uniform improvements but instead redistribute error across dimen-
sions. Higher temperature degrades validity, consistency, and fac-
tuality. Representation-constrained prompting improves diversity
at the expense of factuality, while RAG primarily improves techni-
cal quality while reducing diversity and parity. Overall, end-user
interventions reshape trade-offs rather than providing a general fix.
We release code and data that can be adapted to other disciplines
by replacing domain-specific ground truth and metrics.
CCS Concepts
•Information systems →Language models;Information re-
trieval diversity;•Applied computing→Sociology.
Preprint,
ACM ISBN 978-1-4503-XXXX-X/2018/06
https://doi.org/XXXXXXX.XXXXXXXKeywords
Algorithm Auditing, Impact Assessment, Retrieval-Augmented-
Generation, Constrained Prompting, Large Language Models, Peo-
ple Recommender Systems, Scholar Recommendations
ACM Reference Format:
Lisette Espín-Noboa and Gonzalo Gabriel Méndez. . Whose Name Comes
Up? Benchmarking and Intervention-Based Auditing of LLM-Based Scholar
Recommendation. InProceedings of Preprint.ACM, New York, NY, USA,
28 pages. https://doi.org/XXXXXXX.XXXXXXX
1 Introduction
Large language models (LLMs) now support a range of academic
tasks [ 11,30,37], including literature and peer review [ 38], manu-
script drafting [ 2], summarization, and data analysis [ 57]. Beyond
these document-centric applications, LLMs are also used for tasks
involvingpeopleas entities, including expert recommendation [ 5,7],
scholar search [ 32,47], and identity disambiguation [ 46]. While
recent audits document failures in factuality and demographic rep-
resentation [ 7,32,47], they typically evaluate model outputs in
isolation. In deployed systems, end-user inference-time controls
can substantially shape behavior [ 26,31,51], blurring whether fail-
ures such as refusals, hallucinations, and uneven coverage reflect
model architecture or deployment decisions.
This gap limits the usefulness of existing audits for system
builders and evaluators. Without a standardized way to assess how
inference-time interventions interact with model properties, it is
difficult to compare systems, reproduce findings, or reason about
socio-technical trade-offs under realistic deployment conditions.
Addressing these issues requires a benchmark that evaluates LLM-
based scholar recommendations across both model infrastructure
and end-user inference-time interventions, under tasks, evaluation
metrics, and ground-truth data relevant to academic contexts.arXiv:2602.08873v1  [cs.IR]  9 Feb 2026

Preprint, arXiv, Espín-Noboa and Méndez
We contribute in this direction by introducingLLMScholarBench,
a benchmark for auditing LLM-based scholar recommendation un-
der controlled configurations (Figure 1).LLMScholarBenchsupports
systematic evaluation across infrastructural conditions, including
access type, model size, and reasoning capability, as well as common
post-training interventions available to end users. It spans multi-
ple recommendation tasks grounded in publication records and
evaluates performance along two axes:technical qualityandsocial
representation. Technical quality captures core behavioral prop-
erties, including refusals, validity, duplication, consistency under
repeated prompting, and accuracy. Social representation assesses
how recommendations align with the structure and composition
of the scientific community, measuring connectedness within co-
authorship networks, bibliometric similarity, and diversity and
parity across demographic attributes.
In this paper, we instantiateLLMScholarBenchin physics expert
recommendation and audit 22 LLMs under three common inference-
time interventions: temperature variation, representation-constrained
prompting, and retrieval-augmented generation (RAG) with web
search. This setup enables direct comparison of how deployment
choices reshape performance trade-offs without modifying model
parameters. Our results show that inference-time interventions
primarily redistribute errors across validity, factuality, and social
representation, rather than improving these dimensions jointly.
Contributions.This paper makes the following contributions:
•We introduceLLMScholarBench, a benchmark for auditing LLM-
based scholar recommendation that jointly evaluates model
infrastructure and end-user inference-time interventions.
•We define a standardized evaluation protocol spanning multiple
recommendation tasks and metrics capturing both technical
quality and social representation.
•We conduct a large-scale empirical audit of 22 LLMs across ar-
chitectures, showing how infrastructure choices and deployment-
time interventions reshape socio-technical trade-offs.
•We release code and data [ 6,14] to support reproducible audits
and extension to other academic domains.
2 Related Work
Our work lies at the intersection of three areas: (i) the shift from
retrieval-based to generative expert recommendation, (ii) the evalu-
ation and auditing of LLM-generated scholar information, and (iii)
user-accessible post-training methods for steering model outputs.
Conventional vs. generative scholar recommendation.Tra-
ditional expert-finding systems rely on structured databases and
bibliometric signals (e.g., citation counts, h-index) to rank schol-
ars [54,56]. While effective for retrieval, these systems reinforce ex-
isting visibility gaps by under-representing early-career researchers,
scholars from the Global South, and minority groups [ 22,36,53,55].
LLM-based systems alter this paradigm by generating recommen-
dations through the synthesis of patterns from unstructured text
rather than retrieving candidates from indexed corpora [ 12,17,61].
This shift introduces new failure modes, including hallucinated
scholars, misattributed contributions, and amplified gender and eth-
nic biases [ 7,8]. In addition to inheriting historical biases present
in their training data, LLMs can introduce distortions associatedwith English-language dominance and differential online visibil-
ity [16,52]. These effects are not fixed properties of the model:
unlike classical pipelines, LLM-based recommendations are prompt-
generated and can shift with inference-time configuration, which
motivates our evaluation under realistic user controls rather than a
single default setting.
Auditing LLM-based scholar recommendations.Recent work
audits LLM-based scholar search by analyzing which scientists are
recognized in response to targeted prompts. Sandnes [ 47] finds
no consistent recognition patterns for ChatGPT (GPT-3.5) , while
Liu et al. [ 32] evaluate GPT-4o ,Claude 3.5 Sonnet , and Gemini
1.5, showing that recognition correlates with citation counts and
remains uneven across gender and geography. These studies high-
light representational disparities but focus on single-scholar queries.
Closest to our setting, Barolo et al. [ 7] evaluate scholarrecommen-
dationtasks that jointly measure accuracy and demographic bias,
documenting frequent hallucinations and over-representation of
researchers perceived as White. They also show that name cues,
such as perceived geographic origin, systematically shape who is
recommended. Overall, existing audits cover few models and rely
on fixed prompts or default inference settings, leaving open how
deployment-time choices and inference-time, post-training end-
user-available interventions reshape trade-offs between technical
quality and social representation.
Inference-time controls for steering LLM outputs.End-users
cannot retrain LLMs and therefore rely on post-training, inference-
time controls such as temperature adjustment, prompt engineering,
and retrieval augmentation. Temperature modulates the trade-off
between output stability and variability: lower values tend to pro-
duce more consistent responses, while higher values increase diver-
sity but also the risk of hallucinations and inconsistencies [ 50,51].
In scholar recommendation settings, this can affect whether recom-
mendations concentrate on a small set of well-known researchers or
include a broader range of candidates. Prompt-level constraints (e.g.,
format requirements, representation targets) offer structured con-
trol but may trigger refusals or unsupported justifications, particu-
larly when sensitive attributes are involved [ 23,31,40,43]. Retrieval-
augmented generation grounds outputs in external sources [ 1,13,
26], enabling access to more current information and explicit prove-
nance, but also introduces additional variability tied to query formu-
lation and ranking of retrieved documents [ 27,39,49,60]. Despite
their widespread use, these controls are rarely evaluated system-
atically in scholar recommendation, and their effects on technical
quality and social representation remain poorly understood.
In contrast to prior audits that study limited models, isolated
tasks, or single dimensions, we introduce a reproducible benchmark
for LLM-based scholar recommendation that spans model infras-
tructure and user-accessible, post-training interventions.LLMSchol-
arBenchcomplements existing benchmarks in other domains [ 9,15,
18] by enabling analysis of failure modes and intervention effects
specific to scholar recommendation.
3LLMScholarBench
LLMScholarBenchintegrates benchmarking and intervention-based
auditing to characterize baseline performance and its sensitivity to
post-training user controls in LLM-based scholar recommendation.

Whose Name Comes Up? Benchmarking and Intervention-Based Auditing of LLM-Based Scholar Recommendation Preprint, arXiv,
Table 1: LLMs evaluated in this study, grouped by model size and access type.
Small(< 10B)Medium(10B–50B)Large(50B–400B)Extra Large(≥400B)Proprietary
llama-3.3-8b gemma-3-12b llama-3.1-70b llama-4-mav gemini-2.5- (Small)
qwen3-8b qwen3-14b llama-3.3-70b llama-3.1-405b gemini-2.5-pro (Medium)
grok-4-fast gpt-oss-20b llama-4-scout deepseek-chat-v3.1
mistral-small-3.2-24b gpt-oss-120b deepseek-r1-0528
gemma-3-27b-it qwen3-235b-a22b-2507
qwen3-30b-a3b-2507 mistral-medium-3
qwen3-32b
3.1 Preliminaries
We build on Barolo et al. [ 7], which audited whom LLMs recom-
mend as experts. We generalize this framing into a structured, re-
producible benchmark by (1) formalizing metrics that separate tech-
nical quality from social representation, (2) adding visual analyses
that surface model and intervention trade-offs, and (3) expanding
coverage to many more LLMs and user-controllable inference-time
interventions (rather than a single, static setting). Next, we summa-
rize the shared foundations.
Tasks.The evaluation comprises five task families, each with at
least two contextual variants: (i)top- 𝑘expert recommendations (top
5 vs. top 100); (ii)field-basedrecommendations (Condensed Matter
& Material Physics (CMMP) vs. Physics Education Research (PER));
(iii)epoch-basedrecommendations (1950s vs. 2000s); (iv)seniority-
basedrecommendations (early-career vs. senior scholars); and (v)
twintasks, which assess whether models can identify researchers
similar to a reference scholar and how they handle ambiguous or
adversarial requests, including fictional or non-academic references.
Ground-truth data.Evaluating factual accuracy and social rep-
resentation requires a reference database with verified scholarly
records. We use curated publication data from the American Phys-
ical Society (APS) [ 3], covering a large fraction of the physics re-
search community since 1893. APS data provide structured infor-
mation on authorship, venues, research areas, and citations, en-
abling verification of both the existence and scholarly activity of
recommended individuals. Physics is a male-dominated field with
well-documented gender disparities [ 22,25], making it a suitable
domain for studying representation and parity in scholar recommen-
dation. Additionally, we augment APS records with metadata from
OpenAlex [ 42] to obtain global bibliometric indicators and resolve
author name variants. Perceived gender and ethnicity are inferred
from names. Gender is inferred using gender-guesser (reported
mean F1-score =0.95), and ethnicity using demographicx [29]
andethnicolr [24] (reported mean F1-score =0.84). While these
attributes do not reflect self-reported identity, they capture how
individuals may be socially categorized in the absence of explicit in-
formation, as commonly inferred by humans and algorithms [ 19,33].
We define scholarly prominence using publication- and citation-
based quantiles over the APS author population, with thresholds
{0.0,0.5,0.8,0.95,1.0}corresponding tolow,mid,high, andelite
strata. Additional details in Appendix B.
Prompts.Our base template was designed through an iterative,
human-in-the-loop process. It uses zero-shot prompts with explicitstep-by-step instructions to reduce errors [ 21,35,62].LLMScholar-
Benchuses this template (Appendix Figure A.1) to specify the task,
step-by-step instructions, output format, and additional guidelines.
3.2 Experimental Setup
LLMs.We evaluate 22 LLMs spanning diverse parameter scales and
architectures (Table 1), including open-weight and proprietary sys-
tems, standard and reasoning-oriented models, with sizes ranging
from 8B to 671B. Open-weight models are accessed via OpenRouter1
using paid credits to ensure stable access across providers without
rate-limit constraints. Proprietary models are accessed through
Google Vertex AI.2Further details in Appendix A.
Initial calibration.Sampling temperature affects response qual-
ity [28], thus, using a default (e.g., 𝑡=0) can introduce uncontrolled
uncertainty when comparing models. We therefore conduct a tem-
perature analysis for each model by evaluating multiple tempera-
ture values ( 𝑡∈{ 0.00,0.25,0.50,0.75,1.0,1.5,2.0}), collecting three
independent outputs per model–task–temperature configuration.
We select a single temperature per model that maximizes mean fac-
tual accuracy while maintaining high response validity, as defined
in Section 3.4. This model-specific temperature is then used as the
default setting in all subsequent data collection, benchmarking and
intervention experiments. Further details in Appendix A.3.
Data collection.After selecting the temperature for each model,
we collect the final audit data over a one-month period (31 days:
December 19, 2025 to January 18, 2026), with queries issued twice
daily at fixed times (08:00 and 16:00). To mitigate transient failures,
we allow up to two automatic retries per prompt: if the initial
attempt is invalid, we issue a second attempt, and if that also fails,
we issue a third attempt. For downstream analyses, we retain only
the first valid attempt per prompt and discard any previous attempts.
This data is used for both infrastructure benchmarking and end-user
intervention. Additional details are provided in Appendix A.1.
Pre-processing.Each model response is parsed and assigned one
of seven labels: valid, verbose, fixed, skipped, refused, API error, or
invalid. A response isvalidif it contains a structured list of scholar
names ready to be used.Verboseresponses contain additional ex-
planatory text but still include a valid list. Responses labeledfixed
correspond to malformed outputs that can be partially recovered.
Skippedresponses contain a list with a mix of valid and invalid
names (e.g., placeholders), from which only the former are retained.
1https://openrouter.ai
2https://cloud.google.com/vertex-ai

Preprint, arXiv, Espín-Noboa and Méndez
API errorresponses correspond to failed requests, including time-
outs or backend failures, whileinvalidoutputs include empty or
nonsensical text that cannot be parsed and do not constitute an
explicit refusal. All sebsequent analyses are restricted to valid and
verbose responses to avoid artifacts introduced by post-processing.
3.3 Auditing Conditions and Interventions
We structure our audit around twoaudit questions(AQs) that orga-
nize the evaluation of LLM-based scholar recommendation.
AQ1. Infrastructure-level conditions.We first analyze how
infrastructure-level design choices shape scholar recommendations.
These factors are not user-controlled but reflect architectural prop-
erties of the underlying models. Holding the prompting protocol
fixed, we group results by three dimensions: model access, model
size, and reasoning capability.
Model access.We distinguish between open-weight and proprietary
models as they reflect different user-facing deployment regimes,
from locally managed to immediately accessible systems.
Model size.Model parameter count is commonly associated with
overall performance [ 20], and may influence factual accuracy or
coverage in scholar recommendation. We evaluate models across
a broad size range, grouped into four categories: small (< 10B),
medium (10B–50B), large (50B–400B), and extra-large (≥400B).
Reasoning capability.We distinguish between standard auto-regressive
models and reasoning-oriented models that generate intermediate
reasoning steps. We group models along this dimension to explore
whether explicit reasoning is associated with differences in output
quality and refusal behavior.
AQ2. End-user interventions.We evaluate three independent,
post-training interventions available at inference time and quan-
tify how they shape model recommendations: temperature control,
representation-constrained prompting, and RAG with web search.
Temperature control.We reuse the recommendations from the tem-
perature analysis (Section 3.2), but with a different goal. Rather
than selecting an optimal setting per model, this intervention char-
acterizes how variations in temperature affect technical quality and
representational outcomes across models.
Representation-constrained prompting.This intervention adds ex-
plicit representation goals to the prompt. We apply it only to the
top-100 task by modifying the criteria in our prompt template
(Appendix Figure A.1). Using top-100 outputs enables stable mea-
surement of distributional differences across gender, ethnicity, and
scholarly prominence, which shorter lists cannot support. We eval-
uate four constraint types: (i) a general diversity request without
specified dimensions; (ii) gender constraints targeting equal or in-
creased representation of scientists with perceived female, male,
or neutral names; (iii) ethnicity constraints targeting balanced or
increased representation across perceived U.S.-based categories
(Asian, Black, Hispanic, White); and (iv) prominence constraints
requesting scholars with more or fewer than1000citations. This
intervention tests whether list composition can be steered through
prompting and whether such steering affects the benchmarks.
RAG with web search.We test RAG on gemini models with na-
tive web search, comparing outputs with and without retrieval to
quantify its effect and contrast it with other interventions.3.4 Evaluation Metrics
For each model-task-parameter configuration, we issue the prompt
repeatedly over time. Each configuration is queried at least 𝑁= 62
times (twice daily over 31 days), with up to three attempts per
query to handle transient failures. Metrics are computed at different
pipeline stages, depending on whether they assess eventual success,
intermediate behavior, or final recommendations.
Refusalsmeasure how often models explicitly decline to answer.
They are computed at the level of individual attempts and include all
generated responses. Let 𝑀denote the total number of responses
across all configurations and attempts, with 𝑁≤𝑀≤ 3𝑁. Let
𝑟𝑗∈{0,1}indicate whether response 𝑗is a refusal, defined as an
explicit statement of non-compliance, typically accompanied by a
brief justification (e.g.,“I cannot comply with requests that involve
racial or ethnic filtering of individuals”). The refusal score is
Refusal=1
𝑀𝑀∑︁
𝑗=1𝑟𝑗 (1)
This metric lies in[0,1], where higher values indicate more frequent
deliberate non-compliance. Incomplete or malformed responses
that do not explicitly decline to answer are not counted as refusals.
Response validitymeasures whether a configuration ultimately
yields a usable recommendation. A configuration is considered valid
if at least one of its attempts produces a well-formed list of recom-
mended scholars. Let 𝑣𝑖∈{0,1}indicate whether configuration 𝑖
has at least one valid response. Validity is defined as
Validity=1
𝑁𝑁∑︁
𝑖=1𝑣𝑖 (2)
Validity lies in[0,1], where1indicates that all configurations even-
tually yield a valid recommendation. Validity and refusal are not
complementary: configurations may be valid despite intermediate
refusals, or invalid without explicit refusals.
Duplicatesquantify redundancy within a single valid recommen-
dation list. For a valid response 𝑖, let𝐿𝑖be the list of recommended
names and𝑈 𝑖⊆𝐿𝑖the set of unique names. The duplicate rate is
Duplicates𝑖=1−|𝑈𝑖|
|𝐿𝑖|(3)
This score lies in[0,1], where0indicates no repetition and higher
values indicate increasing redundancy within the list.
Temporal consistencymeasures the stability of recommendations
across repeated queries of the same configuration over time. For
consecutive valid responses, consistency is computed as the mean
Jaccard similarity between recommendation sets,
Consistency=1
𝑁−1𝑁∑︁
𝑖=2|𝑈𝑖∩𝑈𝑖−1|
|𝑈𝑖∪𝑈𝑖−1|(4)
Consistency lies in [0,1], where0indicates no overlap between
successive recommendation sets and1indicates identical recom-
mendations over time.
Factual accuracyassesses whether recommended individuals cor-
respond to real scientists in a scholarly databaseD. For a valid

Whose Name Comes Up? Benchmarking and Intervention-Based Auditing of LLM-Based Scholar Recommendation Preprint, arXiv,
AccessRefusals Validity Duplicates Consistency Factuality Connectedness Similarity Diversity Parity
Open
Proprietary
Small
Medium
Large
Extra Large
Disabled
EnabledSize
Reasoning
Capability
0 1 0 1 0 1 0 1 0 1 0 1 0 1 0 1 0 1
Figure 2: Infrastructure-level performance. Mean values ( ±95%CI) aggregated by model access, model size, and reasoning
capability. Bold values indicate best-in-group performance for metrics with a clear directional preference (arrows indicate
whether higher or lower is better). The results show clear trade-offs across infrastructure groups, indicating that access, scale,
and reasoning design favor different outcomes depending on the evaluation criterion.
response𝑖, let𝑈𝑖denote the set of uniquely recommended names.
We define the set offactual recommended authorsas
ˆ𝑈𝑖={𝑢∈𝑈 𝑖|𝑢is matched to a real author inD}
Factual accuracy is then defined as the proportion of unique rec-
ommended names that are factual,
Fact 𝑖=|ˆ𝑈𝑖|
|𝑈𝑖|(5)
This metric lies in [0,1], with higher values indicating more ver-
ifiable scholars. Name matching details in Appendix B.3. Beyond
author factuality, we evaluate task-specific accuracy by verifying
that authors also satisfy the requestedcriteriainD.
Connectednessevaluates whether factual recommended authors
form a cohesive scholarly community. Let 𝐺=(𝑉,𝐸) denote the
coauthorship network inD, where nodes represent authors and
edges indicate coauthorship relations. Given the set of factual
recommendations ˆ𝑈𝑖⊆𝑉, we construct the induced subgraph
ˆ𝐺𝑖=𝐺[ ˆ𝑈𝑖], retaining only authors in ˆ𝑈𝑖and coauthorship edges
between them. Let {𝐶1,...,𝐶 𝑚}denote the connected components
ofˆ𝐺𝑖, with component sizes 𝑠𝑐=|𝐶 𝑐|. We quantify fragmentation
using normalized component entropy,
NormEntropy𝑖=−1
log| ˆ𝑈𝑖|𝑚∑︁
𝑐=1𝑠𝑐
|ˆ𝑈𝑖|log𝑠𝑐
|ˆ𝑈𝑖|(6)
and define connectedness as the complement of this quantity,
Connectedness 𝑖=1−NormEntropy𝑖 (7)
Connectedness lies in [0,1], where higher values indicate that most
factual recommended authors belong to a single connected com-
ponent, reflecting stronger structural cohesion, and lower values
indicate fragmentation across multiple disconnected groups.
Scholarly similaritymeasures how similar the factual recom-
mended authors are in terms of career profiles. Each author 𝑢∈ ˆ𝑈𝑖
is represented by a vector of quantitative indicators derived from
D, capturing productivity, citation impact, and career stage. Metric
values are median-imputed, log-transformed with log(1+𝑥), and
standardized to zero mean and unit variance. Principal component
analysis (PCA) [ 59] is applied, retaining the fewest components
that explain at least90%of the variance. Letz 𝑢denote the resulting
ℓ2-normalized embedding of author 𝑢. The similarity in ˆ𝑈𝑖is defined
as the mean cosine similarity between embedding vectors of allunordered pairs of distinct factual authors,
Sim 𝑖=2
|ˆ𝑈𝑖|(|ˆ𝑈𝑖|−1)∑︁
𝑢,𝑣∈ ˆ𝑈𝑖𝑢≠𝑣z⊤
𝑢z𝑣.(8)
Higher values indicate greater homogeneity in scholarly profiles.
Diversitymeasures how evenly factual recommendations are dis-
tributed across categories of a given attribute. Let F𝑎denote the
set of categories for attribute 𝑎, and let𝑝(𝑎)
𝑖𝑓be the proportion of
authors in ˆ𝑈𝑖that belong to category 𝑓∈F 𝑎. Diversity is defined
as normalized Shannon entropy,
Div(𝑎)
𝑖=−Í
𝑓∈F 𝑎𝑝(𝑎)
𝑖𝑓log𝑝(𝑎)
𝑖𝑓
log|F 𝑎|(9)
This metric lies in[0,1], where0indicates concentration in a single
category and higher values indicate a more even distribution across
categories. Authors with unknown attribute values are excluded.
Parityevaluates alignment between the distribution of factual
recommended authors and reference distributions derived from
D. Let𝑞(𝑎)
𝑓denote the proportion of authors inDwho belong to
category𝑓∈F 𝑎. We compute the total variation distance between
the empirical category distribution in ˆ𝑈𝑖andD,
TV(𝑎)
𝑖=1
2∑︁
𝑓∈F 𝑎|𝑝(𝑎)
𝑖𝑓−𝑞(𝑎)
𝑓|(10)
and define parity along attribute𝑎as
Parity(𝑎)
𝑖=1−TV(𝑎)
𝑖(11)
Parity lies in[0,1], with higher values indicating closer alignment
to population-level proportions inD.
Fordiversityandparity, we compute metrics separately for each
categorical attribute 𝑎, including perceived gender, perceived eth-
nicity, publication- and citation-based prominence.
3.5 Benchmark Instantiation
In our audit of 22 LLMs, metrics requiring ground truth (factuality,
connectedness,similarity, andparity) are computed against the APS
corpus, which serves as our reference datasetD(Appendix B).
For task-specific factuality, we verify whether recommended
authors meet the requested criteria3(Appendix Figure A.1). For
thefield task, this requires publications in the specified field (PER
3For example, “senior scientistswho have published in APS journals”

Preprint, arXiv, Espín-Noboa and Méndez
Refusals Consistency Connectedness Similarity Diversity Validity Factuality Parity Duplicates
Open
ProprietaryAccess
S
ML
XLSize
Disabled
EnabledReasoning
t t t t t t t t t
Figure 3: Effect of temperature on performance. Mean values ( ±95%CI) across sampling temperatures, aggregated by model access,
model size, and reasoning capability. Higher temperatures generally reduce most technical metrics, with pronounced thresholds
in outcomes such as validity, indicating that temperature amplifies trade-offs across infrastructure groups. Proprietary models
show lower sensitivity to temperature variation and more stable metric trends than other infrastructure groups.
or CMMP; Section 3.1); for the epoch task, publications within
the requested decade (1950s or 2000s); and for the seniority task,
an academic age derived from publication history that meets the
requested seniority ( ≤10years for early-career, ≥20years for
senior scientists).
In the main body of the paper, metrics are aggregated by model
infrastructure (AQ1) and intervention type (AQ2), averaging scores
across models within each group. We report only author-level fac-
tual accuracy and perceived gender diversity and parity. Results
for other attributes, as well as disaggregated analyses by model or
task, are reported in Appendix C.
4 Results
We present our audit results for physics scholar recommendation
across 22 LLMs (Table 1), evaluated using the nine metrics described
in Section 3.4. For metrics with a clear direction of preference, we
annotate↑for higher-is-better (validity, factuality, parity) and ↓for
lower-is-better (duplicates). The remaining metrics (refusals, con-
sistency, connectedness, similarity, diversity) are reported without
a universal direction, as their interpretation is context-dependent.
We report mean values with 95% confidence intervals, using
Wilson score intervals for binary metrics (i.e., refusals and validity)
and Student𝑡-based intervals for all other metrics.
4.1 AQ1: Infrastructure-level Conditions
Figure 2 summarizes how infrastructure-level choices shape LLM-
based scholar recommendation.
Model access (open vs. proprietary).Open models exhibit more
frequent refusals but higher eventual validity, reflecting greater
recovery across retries. In contrast, proprietary models show no
duplicate outputs, higher temporal consistency, and higher author
factuality, as well as greater gender diversity and parity. They also
recommend scholars who are more closely connected in the APS
coauthorship network and more similar in scholarly profiles, yield-
ing more tightly clustered recommendation sets. Overall, these
results reveal a trade-off by model access: open models favor even-
tual validity despite refusals, whereas proprietary models favor
accuracy and structural coherence in their recommendations.Model size (S, M, L, XL).Refusals and eventual validity increase
with model size, indicating that more frequent refusals do not pre-
vent larger models from producing valid recommendations. Dupli-
cate outputs decrease with size, suggesting improved control over
repeated recommendations. Temporal consistency decreases with
model size, with smaller models producing more stable recommen-
dation sets over time. Author factuality, similarity, and diversity do
not increase monotonically with model size: small models achieve
scores comparable to large and extra-large models. In contrast,
connectedness increases with model size, while parity is higher
for smaller models. Overall, larger models yield higher technical
quality and more tightly connected co-authorship networks, while
smaller models remain competitive on social representation.
Reasoning capability (enabled vs. disabled).Reasoning-disabled
models achieve lower refusals and substantially higher validity than
reasoning-enabled models, indicating stronger compliance with
the required output. In contrast, reasoning-enabled models attain
higher author factuality, suggesting improved factual inference for
retrieved scholars, but at the cost of more frequent refusals and
lower validity. Duplicate outputs are rare in both cases. Tempo-
ral consistency, similarity, and diversity are broadly comparable
across reasoning conditions, with reasoning-enabled models show-
ing slightly higher connectedness and parity. Overall, enabling
explicit reasoning is associated with higher accuracy but lower
reliability in producing valid recommendations.
4.2 AQ2: End-user Interventions
We now examine how user-level interventions shape model outputs.
Temperature control.Figure 3 shows performance as a function
of temperature across model architectures. Increasing temperature
consistently reduces core quality metrics, such as validity and con-
sistency, indicating a higher likelihood of non-compliant outputs
and greater variation in recommendation sets over time. However,
this variation of names does not translate into broader coverage
of the scholar population: connectedness, similarity, diversity, and
parity remain largely unchanged across temperatures. Together,
these patterns indicate that higher sampling randomness increases
output instability without meaningfully diversifying the recom-
mended scholars. The main differences across infrastructures are
concentrated in model access. As temperature increases, propri-
etary models show small gains in validity and factuality, whereas

Whose Name Comes Up? Benchmarking and Intervention-Based Auditing of LLM-Based Scholar Recommendation Preprint, arXiv,
0.000.250.500.751.00
Consistency Connectedness SimilarityDiversity
genderDiversity
ethnicityValidity Refusals FactualitygenderParity
ethnicityParityDuplicates
Balanced
Female-only
Male-only
Neutral-onlyGenderconstraint
0.000.250.500.751.00
B AB AB AB AB AB AB AB AB AB AB A
Figure 4: Effects of gender-constrained prompting on top-100 expert recommendation lists (averaged across all models). Each
panel shows the mean metric value (±95% CI) before (B) and after (A) applying the constraint. Enforcing balanced gender
representation mainly increases gender diversity with little change in gender parity, but reduces factuality and similarity.
Female-only prompts produce the lowest factuality, similarity, and gender parity, while yielding the highest ethnicity diversity.
open models exhibit larger declines. Open models also show higher
refusal rates than proprietary models across all temperatures, with
little sensitivity to temperature.
Representation-constrained prompting.As described in Sec-
tion 3.3, we evaluate four constrained prompting strategies target-
ing gender, ethnicity, prominence, and overall diversity. Figure 4
reports the effects of gender-constrained prompting, while results
for the remaining constraints are provided in Appendix C.2. Across
all constraint types, core technical metrics respond similarly: adding
constraints primarily increases refusals and leads to modest declines
in validity, duplicates, and consistency regardless of constraint di-
rection. Social representativeness metrics show more differentiated
effects. Prompts requesting only female scholars produce the largest
drops in factuality, similarity, and parity across both gender and
ethnicity. Requests for a diverse (balanced) set of scholars also re-
duce factuality and similarity, but uniquely increase diversity across
gender and ethnicity, indicating effective steering toward broader
representation. In contrast, male-only and neutral-name prompts
have limited impact on social representativeness.
RAG with web search.Figure 5 reports the effect of RAG with
web search on gemini models, shown separately for flash and
profor each task. Overall, RAG affects technical quality differently
across model variants, while social representativeness metrics re-
spond more uniformly. Validity generally decreases with RAG for
flash across all tasks and for proexcept in the twins-fake task.
Duplicate outputs remain negligible with and without RAG. Consis-
tency typically decreases under RAG, indicating greater variation
over time, with the exception of twins-real . The twins-fake
task reveals a clear behavioral distinction. Under RAG, only proin-
creases refusal rates in response to nonsensical prompts, yet it also
improves validity and slightly increases factuality. Social represen-
tativeness shows more stable effects across models. Connected-
ness is largely unchanged or slightly reduced, except for gains in
twins-real . Similarity tends to slightly increase with RAG, while
gender and ethnicity diversity decrease, and parity remains mostly
stable, with modest declines intwins-real.
5 Discussion
Rather than identifying a single optimal configuration,LLMSchol-
arBenchreveals stable tradeoffs between answerability, factuality,
and the distribution of surfaced scholars. We interpret these results
through our audit questions, then discuss their implications for
scholar recommendation systems and benchmarking.
AQ1: Infrastructure-level conditions.Infrastructure choices
induce systematic tradeoffs rather than uniform improvements.Validity (producing a parsable list), refusals, and factuality move
along coupled axes, whereas temporal consistency and gender rep-
resentation (diversity and parity) remain largely stable across con-
ditions. Open models more often return structured lists (higher
validity) but with weaker author factuality and slightly elevated re-
fusal rates. Proprietary models show the opposite pattern: stronger
factuality—likely from superior training data and access to schol-
arly sources like Google Scholar—yet lower validity because they
more frequently hedge, or avoid person-list generation, shifting
failures from “a list with errors” to “no usable list,” most clearly
forgemini-2.5-flash (Appendix Figures A.2 and C.7). Model size
shows diminishing returns: larger models improve formatting and
reduce duplication but offer limited factuality gains, as small models
often match larger ones. Reasoning-enabled models improve factu-
ality but increase unstructured outputs and refusals (Appendix A.4),
suggesting they prioritize caution over task completion—though
repeated attempts often succeed after initial refusals, revealing in-
stability rather than principled abstention. Overall, infrastructure
decisions determine whether answers are produced and how accu-
rate they are, but have limited influence on temporal stability or
representational balance.
At the individual-model level (Appendix C.1), deepseek achieves
the strongest author factuality and leads in task-specific accuracy
(field, epoch, seniority), though these dimensions remain weaker
across all models; gemma andllama variants rank highest for va-
lidity; and while no model achieves ideal parity, deepseek outper-
forms the rest, followed by gemma ,gpt,grok , and gemini . Within
families, version differences mainly affect refusals and consistency.
AQ2: End-user interventions.No intervention is dominant; the
appropriate choice depends on the target metric. For validity, low
temperature is the most reliable setting, as it consistently produces
well-formed recommendation lists. In contrast, constrained prompt-
ing and RAG frequently reduce validity and increase refusals. For
nonsensical prompts such as twins-fake , refusals are most effec-
tively triggered by RAG. Refusals for unethical requests are instead
primarily triggered by representation-constrained prompting (Ap-
pendix Figure A.5). For factuality, RAG is the safest choice. It does
not reduce factuality relative to baseline prompting and can yield
modest gains, while lower temperature provides smaller improve-
ments by reducing sampling noise (Appendix Figure C.12). For
diversity, constrained prompting is the only intervention that pro-
duces meaningful change, and only under specific representation
constraints (Appendix Figures C.13 to C.16); even then, gains often
come with reduced factuality or parity. Overall, temperature has
weak effects, constrained prompting trades technical quality for
parity, and RAG tends to narrow exposure (Appendix Figure C.18).

Preprint, arXiv, Espín-Noboa and Méndez
0.000.250.500.751.00
0.000.250.500.751.000.000.250.500.751.00
0.000.250.500.751.00
B AB AB AB AB AB AB AB AB AB AB AConsistency Connectedness SimilarityDiversity
genderDiversity
ethnicityValidity Refusals FactualitygenderParity
ethnicityParityDuplicatesFlash Pro
Top-k
By field
By epoch
By seniority
Twins-real
Twins-fakeTask
Figure 5: Effect of RAG web search on performance across tasks for gemini models. Panels show mean metric values ( ±95%CI)
before (B) and after (A) enabling RAG. Flash (top row) shows a larger drop in validity under RAG across most tasks, whereas
Pro (bottom row) is comparatively less affected. Duplicates remain near zero for both, factuality stays high, and changes in
connectedness, similarity, and representation metrics (diversity/parity) are smaller and more task-dependent.
No intervention improves factuality and parity simultaneously. This
tension appears structural within our evaluation setting and can-
not be resolved through inference-time controls alone, motivating
interventions beyond prompting, including model adaptation using
explicit scholarly knowledge graphs.
The key takeaway of our findings is that model performance is
configuration-dependent rather than fixed. Model rankings shift
with inference-time settings: temperature variation largely pre-
serves baseline behavior, whereas constrained prompting and RAG
reorder the performance frontier across technical quality and parity
(Appendix C.3).LLMScholarBenchmakes these trade-offs explicit
and comparable, showing that many apparent “best” models are
optimal only under specific deployment choices.
Tool and benchmark design implications.Our answers to AQ1
and AQ2 show that LLM-based scholar recommendation is a system-
and benchmark-design problem with multi-objective trade-offs: pro-
ducing a usable list, avoiding unverifiable entities, remaining stable
across runs, and managing distributional exposure. Because infras-
tructure and prompt controls mainly shift failure modes rather than
dominate all objectives, tools should help users state their intent
(e.g., exploratory coverage vs. consistent outputs) and surface the re-
sulting trade-offs. Benchmarks should avoid single-score claims and
instead report results under a few fixed settings aligned with these
goals. They should also make list validity explicit by either scoring
schema compliance directly or normalizing outputs to the schema
before scoring. These implications carry over to practice. Deploy-
ments should favor auditable pipelines over single chat responses
by separating candidate generation, entity matching, and sourcing
to make errors diagnosable and recommendations traceable. Finally,
representation goals should be evaluated against explicit reference
baselines and treated as design choices, with traceability that sup-
ports audits of who was recommended, on what evidence, and how
interventions changed the generated lists.
6 Limitations and Future Work
We frameLLMScholarBench’s boundaries as research challenges
and outline actionable next steps.
From validity to utility.Ourvaliditymetric enforces a repro-
ducible requirement: a structured, machine-parsable list of dictio-
naries. This enables model-agnostic auditing, but it can undervalue
outputs that are malformed yet still usable. We already label com-
mon failure modes (e.g., fixed, skipped) but exclude them fromscoring to avoid credit from post hoc repair. A direct extension
is to add context-dependentutilitymeasures tied to use, such as
extraction time, human acceptance, or downstream decision quality.
Limited causal attribution.Infrastructural conditions (AQ1) re-
flect deployment choices, but they are confounded with alignment
policies, training data, and decoding defaults. We therefore inter-
pret results as operational regularities rather than causal effects.
Stronger causal claims require tighter controls, such as within-
family comparisons or ablations where only one factor changes.
Name-based factuality and relevance.Factuality is limited by
entity resolution when models return names without identifiers.
We reduce this brittleness by augmenting APS with OpenAlex
and prioritizing full-name matching, treating homonym matches
as factual to limit false negatives (Appendix B.3). Yet existence
is only one baseline: users also need context-specific relevance,
which bibliometrics capture only weakly. Progress needs domain-
specific fine-tuning, community-validated ground truth sets, and
evaluations that pair bibliometric checks with expert judgment.
Cross-domain applicability.We deployLLMScholarBenchin
physics, which may not generalize to other fields or contexts. Nonethe-
less, our pipeline is portable: extending to other domains requires
swapping ground-truth modules. This portability is supported by
our modular architecture [ 6]:LLMCaller handles data collection,
while Auditor standardizes outputs and produces intermediate
artifacts for downstream analysis.
7 Conclusion
We presentLLMScholarBench, a reproducible benchmark for LLM-
based scholar recommendation that jointly measures technical qual-
ity and social representation. Using this benchmark in physics
expert recommendation across 22 LLMs reveals that model and
inference-time interventions rarely improve all metrics simultane-
ously, exposing clear trade-offs. Infrastructure choices (proprietary,
large, and reasoning models) tend to improve factuality but can
reduce validity or increase refusals. In contrast, end-user inference-
time controls (temperature, constrained prompting, and RAG with
web search) mostly reshape technical behavior and, for RAG, who
is surfaced. Similarity, diversity, and parity move little across these
settings, suggesting social representativeness is not easily steered
by prompting or scaling alone. We releaseLLMScholarBenchwith
code and data to make these trade-offs measurable, comparable,
and easier to improve across domains.

Whose Name Comes Up? Benchmarking and Intervention-Based Auditing of LLM-Based Scholar Recommendation Preprint, arXiv,
Artifacts Statement
Our code and data are already publicly available [ 6,14], with a
citable DOI for the code to be finalized. We are also developing a
supplementary visualization interface that builds on the benchmark
results presented in this paper to support qualitative exploration of
audit outcomes and author visibility within the APS coauthorship
network, extending prior visualization work [ 4]. Upon acceptance,
we intend to submit the code and data for artifact evaluation and
apply for the “Artifacts Available” badge, and to release the visu-
alization tool as a supplementary artifact in accordance with the
conference guidelines.
Acknowledgments
We thank Daniele Barolo for implementing and maintaining the
data collection infrastructure, and for his valuable feedback and
discussions during the design of the experiments. We also thank
Xiangnan Feng for facilitating access to the Google Vertex AI API.
LEN was supported by the Austrian Science Promotion Agency
FFG project no. 873927 and the Vienna Science and Technology
Fund WWTF under project No. ICT20-07.
References
[1]Nurshat Fateh Ali, Md. Mahdi Mohtasim, Shakil Mosharrof, and T. Gopi Krishna.
2024. Automated Literature Review Using NLP Techniques and LLM-Based
Retrieval-Augmented Generation. arXiv:2411.18583 [cs.CL] doi:10.48550/arXiv.
2411.18583
[2]Signe Altmäe, Alberto Sola-Leyva, and Andres Salumets. 2023. Artificial intelli-
gence in scientific writing: a friend or a foe?Reproductive BioMedicine Online47,
1 (2023), 3–9. doi:10.1016/j.rbmo.2023.04.009
[3]American Physical Society. 2024. APS Data Sets for Research. https://journals.
aps.org/datasets. Accessed: 2024-10-12.
[4]Yi Zhe Ang and Liuhuaying Yang. 2025. Whose Name Comes Up? An In-
teractive Visualization for Scholar Recommendation. https://vis.csh.ac.at/
whosenamecomesup. (pilot).
[5]Krisztian Balog, Leif Azzopardi, and Maarten de Rijke. 2009. A language modeling
framework for expert finding.Information Processing & Management45, 1 (2009),
1–19. doi:10.1016/j.ipm.2008.06.003
[6]Daniele Barolo and Lisette Espín-Noboa. 2026. LLMScholarBench: A Bench-
mark for Auditing LLM-Based Scholar Recommendation. https://github.com/
CSHVienna/LLMScholarBench. Version v2.0.0, accessed: 2026-02-02.
[7]Daniele Barolo, Chiara Valentin, Fariba Karimi, Luis Galárraga, Gonzalo G.
Méndez, and Lisette Espín-Noboa. 2025. Whose Name Comes Up? Audit-
ing LLM-Based Scholar Recommendations. (2025). arXiv:2506.00074 [cs.CY]
doi:10.48550/arXiv.2506.00074
[8]Tolga Bolukbasi, Kai-Wei Chang, James Zou, Venkatesh Saligrama, and Adam
Kalai. 2016. Man is to computer programmer as woman is to homemaker?
debiasing word embeddings. (2016), 4356–4364.
[9]Haibin Chen, Kangtao Lv, Chengwei Hu, Yanshi Li, Yujin Yuan, Yancheng
He, Xingyao Zhang, Langming Liu, Shilei Liu, Wenbo Su, and Bo Zheng.
2025. ChineseEcomQA: A Scalable E-commerce Concept Evaluation Bench-
mark for Large Language Models. InProceedings of the 31st ACM SIGKDD Con-
ference on Knowledge Discovery and Data Mining V.2 (KDD ’25)(Toronto, ON,
Canada). Association for Computing Machinery, New York, NY, USA, 11 pages.
doi:10.1145/3711896.3737374
[10] Peter Christen. 2012. The Data Matching Process. InData Matching: Concepts and
Techniques for Record Linkage, Entity Resolution, and Duplicate Detection. Springer
Berlin Heidelberg, Berlin, Heidelberg, 23–35. doi:10.1007/978-3-642-31164-2_2
[11] Marina Chugunova, Dietmar Harhoff, Katharina Hölzle, Verena Kaschub, Sonal
Malagimani, Ulrike Morgalla, and Robert Rose. 2026. Who uses AI in research,
and for what? Large-scale survey evidence from Germany.Research Policy55, 2
(2026), 105381. doi:10.1016/j.respol.2025.105381
[12] Sunhao Dai, Ninglu Shao, Haiyuan Zhao, Weijie Yu, Zihua Si, Chen Xu, Zhongx-
iang Sun, Xiao Zhang, and Jun Xu. 2023. Uncovering ChatGPT’s Capabilities
in Recommender Systems. InProceedings of the 17th ACM Conference on Recom-
mender Systems(Singapore, Singapore)(RecSys ’23). Association for Computing
Machinery, New York, NY, USA, 1126–1132. doi:10.1145/3604915.3610646
[13] Dario Di Palma. 2023. Retrieval-augmented Recommender System: Enhancing
Recommender Systems with Large Language Models. InProceedings of the 17th
ACM Conference on Recommender Systems(Singapore, Singapore)(RecSys ’23).Association for Computing Machinery, New York, NY, USA, 1369–1373. doi:10.
1145/3604915.3608889
[14] Espin-Noboa, Lisette. 2025.LLMScholarBench – Benchmark & Intervention Audits
(datasets). doi:10.17605/OSF.IO/AP5QW
[15] Jie Feng, Jun Zhang, Tianhui Liu, Xin Zhang, Tianjian Ouyang, Junbo Yan, Yuwei
Du, Siqi Guo, and Yong Li. 2025. CityBench: Evaluating the Capabilities of
Large Language Models for Urban Tasks. InProceedings of the 31st ACM SIGKDD
Conference on Knowledge Discovery and Data Mining V.2 (KDD ’25)(Toronto, ON,
Canada). Association for Computing Machinery, New York, NY, USA, 12 pages.
doi:10.1145/3711896.3737375
[16] Yanzhu Guo, Simone Conia, Zelin Zhou, Min Li, Saloni Potdar, and Henry Xiao.
2025. Do large language models have an English accent? evaluating and im-
proving the naturalness of multilingual LLMs. InProceedings of the 63rd Annual
Meeting of the Association for Computational Linguistics (Volume 1: Long Papers).
3823–3838. https://aclanthology.org/2025.acl-long.193.pdf
[17] Chumeng Jiang, Jiayin Wang, Weizhi Ma, Charles L. A. Clarke, Shuai Wang,
Chuhan Wu, and Min Zhang. 2025. Beyond Utility: Evaluating LLM as Rec-
ommender. InProceedings of the ACM on Web Conference 2025(Sydney NSW,
Australia)(WWW ’25). Association for Computing Machinery, New York, NY,
USA, 3850–3862. doi:10.1145/3696410.3714759
[18] Zhuohang Jiang, Pangjing Wu, Ziran Liang, Peter Q. Chen, Xu Yuan, Ye Jia,
Jiancheng Tu, Chen Li, Peter H. F. Ng, and Qing Li. 2025. HiBench: Benchmarking
LLMs Capability on Hierarchical Structure Reasoning. InProceedings of the 31st
ACM SIGKDD Conference on Knowledge Discovery and Data Mining V.2 (KDD ’25)
(Toronto, ON, Canada). Association for Computing Machinery, New York, NY,
USA, 11 pages. doi:10.1145/3711896.3737378
[19] Brendan T Johns and Melody Dye. 2019. Gender bias at scale: Evidence from the
usage of personal names.Behavior research methods51, 4 (2019), 1601–1618.
[20] Jared Kaplan, Sam McCandlish, Tom Henighan, Tom B. Brown, Benjamin Chess,
Rewon Child, Scott Gray, Alec Radford, Jeffrey Wu, and Dario Amodei. 2020.
Scaling Laws for Neural Language Models. (2020). arXiv:2001.08361 [cs.LG]
doi:10.48550/arXiv.2001.08361
[21] Takeshi Kojima, Shixiang Shane Gu, Machel Reid, Yutaka Matsuo, and Yusuke
Iwasawa. 2022. Large language models are zero-shot reasoners. , Article 1613
(2022), 15 pages. https://dl.acm.org/doi/10.5555/3600270.3601883
[22] Hyunsik Kong, Samuel Martin-Gutierrez, and Fariba Karimi. 2022. Influence of
the first-mover advantage on the gender disparities in physics citations.Commu-
nications Physics5, 1 (Oct. 2022), 243. doi:10.1038/s42005-022-00997-x
[23] Preethi Lahoti, Nicholas Blumm, Xiao Ma, Raghavendra Kotikalapudi, Sahitya
Potluri, Qijun Tan, Hansa Srinivasan, Ben Packer, Ahmad Beirami, Alex Beutel,
and Jilin Chen. 2023. Improving Diversity of Demographic Representation in
Large Language Models via Collective-Critiques and Self-Voting. (Dec. 2023),
10383–10405. doi:10.18653/v1/2023.emnlp-main.643
[24] Suriyan Laohaprapanon, Gaurav Sood, and Bashar Naji. 2022. ethnicolr: Predict
Race and Ethnicity From Name. https://github.com/appeler/ethnicolr
[25] Kristina Lerman, Yulin Yu, Fred Morstatter, and Jay Pujara. 2022. Gendered
citation patterns among the scientific elite.Proceedings of the National Academy
of Sciences119, 40 (2022), e2206070119. doi:10.1073/pnas.2206070119
[26] Patrick Lewis, Ethan Perez, Aleksandra Piktus, Fabio Petroni, Vladimir Karpukhin,
Naman Goyal, Heinrich Küttler, Mike Lewis, Wen-tau Yih, Tim Rocktäschel,
Sebastian Riedel, and Douwe Kiela. 2020. Retrieval-augmented generation for
knowledge-intensive NLP tasks. InProceedings of the 34th International Conference
on Neural Information Processing Systems(Vancouver, BC, Canada)(NIPS ’20).
Curran Associates Inc., Red Hook, NY, USA, Article 793, 16 pages.
[27] Johnny Li, Saksham Consul, Eda Zhou, James Wong, Naila Farooqui, Yuxin Ye,
Nithyashree Manohar, Zhuxiaona Wei, Tian Wu, Ben Echols, Sharon Zhou, and
Gregory Diamos. 2025. Banishing LLM Hallucinations Requires Rethinking
Generalization. (2025). arXiv:2406.17642 [cs.CL] doi:10.48550/arXiv.2406.17642
[28] Lujun Li, Lama Sleem, Niccolo’ Gentile, Geoffrey Nichil, and Radu State. 2025.
Exploring the Impact of Temperature on Large Language Models: Hot or Cold?
Procedia Computer Science264 (2025), 242–251. doi:10.1016/j.procs.2025.07.135
International Neural Network Society Workshop on Deep Learning Innovations
and Applications 2025.
[29] L. Liang and D.E. Acuna. 2021. demographicx: A Python package for estimating
gender and ethnicity using deep learning transformers. https://github.com/your-
repository-url. Python package.
[30] Zhehui Liao, Maria Antoniak, Inyoung Cheong, Evie Yu-Yen Cheng, Ai-Heng
Lee, Kyle Lo, Joseph Chee Chang, and Amy X. Zhang. 2024. LLMs as Research
Tools: A Large Scale Survey of Researchers’ Usage and Perceptions. (2024).
arXiv:2411.05025 [cs.CL] doi:10.48550/arXiv.2411.05025
[31] Pengfei Liu, Weizhe Yuan, Jinlan Fu, Zhengbao Jiang, Hiroaki Hayashi, and
Graham Neubig. 2023. Pre-train, Prompt, and Predict: A Systematic Survey of
Prompting Methods in Natural Language Processing.ACM Comput. Surv.55, 9,
Article 195 (Jan. 2023), 35 pages. doi:10.1145/3560815
[32] Yixuan Liu, Abel Elekes, Jianglin Lu, Rodrigo Dorantes-Gilardi, and Albert-
Laszlo Barabasi. 2025. Unequal Scientific Recognition in the Age of LLMs. In
Findings of the Association for Computational Linguistics: EMNLP 2025, Chris-
tos Christodoulopoulos, Tanmoy Chakraborty, Carolyn Rose, and Violet Peng

Preprint, arXiv, Espín-Noboa and Méndez
(Eds.). Association for Computational Linguistics, Suzhou, China, 23558–23568.
doi:10.18653/v1/2025.findings-emnlp.1279
[33] Lillian MacNell, Adam Driscoll, and Andrea N Hunt. 2015. What’s in a name:
Exposing gender bias in student ratings of teaching.Innovative Higher Education
40, 4 (2015), 291–303.
[34] Alexandru Maries, Yangquiting Li, and Chandralekha Singh. 2024. Challenges
faced by women and persons excluded because of their ethnicity and race in
physics learning environments: review of the literature and recommendations
for departments and instructors.Reports on Progress in Physics88, 1 (dec 2024),
015901. doi:10.1088/1361-6633/ad91c4
[35] Ggaliwango Marvin, Nakayiza Hellen, Daudi Jjingo, and Joyce Nakatumba-
Nabende. 2024. Prompt Engineering in Large Language Models. InData In-
telligence and Cognitive Informatics, I. Jeena Jacob, Selwyn Piramuthu, and Prze-
myslaw Falkowski-Gilski (Eds.). Springer Nature Singapore, Singapore, 387–402.
[36] Robert K. Merton. 1968. The Matthew Effect in Science.Science159, 3810 (1968),
56–63. doi:10.1126/science.159.3810.56
[37] Jesse G. Meyer, Ryan J. Urbanowicz, Patrick C. N. Martin, Karen O’Connor,
Ruowang Li, Pei-Chen Peng, Tiffani J. Bright, Nicholas Tatonetti, Kyoung Jae
Won, Graciela Gonzalez-Hernandez, and Jason H. Moore. 2023. ChatGPT and
large language models in academia: opportunities and challenges.BioData Mining
16, 1 (July 2023), 20. doi:10.1186/s13040-023-00339-9
[38] Miryam Naddaf. 2026. More than half of researchers now use AI for peer re-
view—often against guidance.Nature649, 8096 (2026), 273–274. doi:10.1038/
d41586-025-04066-5
[39] Reiichiro Nakano, Jacob Hilton, Suchir Balaji, Jeff Wu, Long Ouyang, Christina
Kim, Christopher Hesse, Shantanu Jain, Vineet Kosaraju, William Saunders, Xu
Jiang, Karl Cobbe, Tyna Eloundou, Gretchen Krueger, Kevin Button, Matthew
Knight, Benjamin Chess, and John Schulman. 2022. WebGPT: Browser-assisted
question-answering with human feedback. (2022). arXiv:2112.09332 [cs.CL]
doi:10.48550/arXiv.2112.09332
[40] Emma Pierson, Divya Shanmugam, Rajiv Movva, Jon Kleinberg, Monica Agrawal,
Mark Dredze, Kadija Ferryman, Judy Wawira Gichoya, Dan Jurafsky, Pang Wei
Koh, et al .2025. Using large language models to promote health equity.NEJM
AI2, 2 (2025), AIp2400889. doi:10.1056/AIp2400889
[41] Florin Pop, Judd Rosenblatt, Diogo Schwerz de Lucena, and Michael Vaiana.
2024. Rethinking harmless refusals when fine-tuning foundation models.
arXiv:2406.19552 [cs.CL] doi:10.48550/arXiv.2406.19552
[42] Jason Priem, Heather Piwowar, and Richard Orr. 2022. OpenAlex: A fully-open
index of scholarly works, authors, venues, institutions, and concepts. (2022).
arXiv:2205.01833 [cs.DL] doi:10.48550/arXiv.2205.01833
[43] Chahat Raj, Anjishnu Mukherjee, Aylin Caliskan, Antonios Anastasopoulos, and
Ziwei Zhu. 2025. Breaking Bias, Building Bridges: Evaluation and Mitigation
of Social Biases in LLMs via Contact Hypothesis. InProceedings of the 2024
AAAI/ACM Conference on AI, Ethics, and Society(San Jose, California, USA)(AIES
’24). AAAI Press, 1180–1189.
[44] Nils Reimers and Iryna Gurevych. 2019. Sentence-BERT: Sentence Embeddings
using Siamese BERT-Networks. arXiv:1908.10084 [cs.CL] doi:10.48550/arXiv.1908.
10084
[45] Katemari Rosa and Felicia Moore Mensah. 2016. Educational pathways of
Black women physicists: Stories of experiencing and overcoming obstacles in
life.Phys. Rev. Phys. Educ. Res.12 (Aug 2016), 020113. Issue 2. doi:10.1103/
PhysRevPhysEducRes.12.020113
[46] Prateek Sancheti, Kamalakar Karlapalem, and Kavita Vemuri. 2024. LLM Driven
Web Profile Extraction for Identical Names. InCompanion Proceedings of the
ACM Web Conference 2024(Singapore, Singapore)(WWW ’24). Association for
Computing Machinery, New York, NY, USA, 1616–1625. doi:10.1145/3589335.
3651946
[47] Frode Eika Sandnes. 2024. Can we identify prominent scholars using ChatGPT?
Scientometrics129, 1 (Jan. 2024), 713–718. doi:10.1007/s11192-023-04882-4
[48] Linda J. Sax, Kathleen J. Lehman, Ramón S. Barthelemy, and Gloria Lim. 2016.
Women in physics: A comparison to science, technology, engineering, and math
education over four decades.Phys. Rev. Phys. Educ. Res.12 (Aug 2016), 020108.
Issue 2. doi:10.1103/PhysRevPhysEducRes.12.020108
[49] Timo Schick, Jane Dwivedi-Yu, Roberto Dessì, Roberta Raileanu, Maria Lomeli,
Luke Zettlemoyer, Nicola Cancedda, and Thomas Scialom. 2023. Toolformer:
Language Models Can Teach Themselves to Use Tools. InAdvances in Neural
Information Processing Systems (NeurIPS). doi:10.5555/3666122.3669119
[50] Chirag Shah. 2025. From Prompt Engineering to Prompt Science with Humans
in the Loop.Commun. ACM68, 6 (June 2025), 54–61. doi:10.1145/3709599
[51] Sergey Troshin, Wafaa Mohammed, Yan Meng, Christof Monz, Antske Fokkens,
and Vlad Niculae. 2025. Control the Temperature: Selective Sampling for Diverse
and High-Quality LLM Outputs. arXiv:2510.01218 [cs.LG] doi:10.48550/arXiv.
2510.01218
[52] Laura Vargas-Parada. 2025. Large language models are biased-local initiatives
are fighting for change.Nature(2025). doi:10.1038/d41586-025-03891-y
[53] Mihaela Vlasceanu and David M. Amodio. 2022. Propagation of societal gender
inequality by internet search algorithms.Proceedings of the National Academy of
Sciences119, 29 (2022), e2204529119.[54] Paul T. von Hippel and Stephanie Buck. 2023. Improve academic search engines
to reduce scholars’ biases.Nature Human Behaviour7, 2 (2023), 157–158.
[55] Orsolya Vásárhelyi and Emőke Ágnes Horvát. 2023. Who benefits from altmet-
rics? The effect of team gender composition on the link between online visibility
and citation impact. arXiv:2308.00405 [cs.CY] doi:10.48550/arXiv.2308.00405
[56] Ludo Waltman and Nees Jan van Eck. 2012. The inconsistency of the h-index.
Journal of the American Society for Information Science and Technology63, 2 (2012),
406–415. doi:10.1002/asi.21678
[57] Xinru Wang, Hannah Kim, Sajjadur Rahman, Kushan Mitra, and Zhengjie Miao.
2024. Human-llm collaborative annotation through effective verification of llm
labels. InProceedings of the 2024 CHI Conference on Human Factors in Computing
Systems. 1–21.
[58] Yaoshu Wang, Jianbin Qin, and Wei Wang. 2017. Efficient Approximate Entity
Matching Using Jaro-Winkler Distance. InWeb Information Systems Engineering
– WISE 2017, Athman Bouguettaya, Yunjun Gao, Andrey Klimenko, Lu Chen,
Xiangliang Zhang, Fedor Dzerzhinskiy, Weijia Jia, Stanislav V. Klimenko, and
Qing Li (Eds.). Springer International Publishing, Cham, 231–239.
[59] Svante Wold, Kim Esbensen, and Paul Geladi. 1987. Principal component analysis.
Chemometrics and Intelligent Laboratory Systems2, 1 (1987), 37–52. doi:10.1016/
0169-7439(87)80084-9 Proceedings of the Multivariate Statistical Workshop for
Geologists and Geochemists.
[60] Shunyu Yao, Jeffrey Zhao, Dian Yu, Nan Du, Izhak Shafran, Karthik Narasimhan,
and Yuan Cao. 2023. ReAct: Synergizing Reasoning and Acting in Language
Models. arXiv:2210.03629 [cs.CL] doi:10.48550/arXiv.2210.03629
[61] Jizhi Zhang, Keqin Bao, Yang Zhang, Wenjie Wang, Fuli Feng, and Xiangnan
He. 2024. Large Language Models for Recommendation: Progresses and Future
Directions. InCompanion Proceedings of the ACM Web Conference 2024(Singapore,
Singapore)(WWW ’24). Association for Computing Machinery, New York, NY,
USA, 1268–1271. doi:10.1145/3589335.3641247
[62] Chunting Zhou, Junxian He, Xuezhe Ma, Taylor Berg-Kirkpatrick, and Graham
Neubig. 2022. Prompt Consistency for Zero-Shot Task Generalization. (Dec.
2022), 2613–2626. doi:10.18653/v1/2022.findings-emnlp.192
[63] Yuqi Zhu, Jia Li, Ge Li, YunFei Zhao, Jia Li, Zhi Jin, and Hong Mei. 2023. Hot or
Cold? Adaptive Temperature Sampling for Code Generation with Large Language
Models. arXiv:2309.02772 [cs.SE] doi:10.48550/arXiv.2309.02772
A Methods (Extended)
This section provides an extended description of the infrastructure,
model selection criteria, and execution protocol used to evaluate
LLMs in this study.
A.1 Setup
We access open-weight LLMs through OpenRouter,4a unified API
that provides programmatic access to models hosted by multiple in-
ference providers. OpenRouter allows the same model to be served
by different subproviders, which may vary in hardware configura-
tion, numerical precision, latency, and cost. We explicitly record
the subprovider used for each model to ensure transparency and
reproducibility (see Table A.1).
We opted for a paid OpenRouter subscription after preliminary
experiments revealed frequent rate limits under the free tier. Our
evaluation required a large number of repeated queries to assess
temporal consistency, and refusal behavior. In total, the temperature
analysis was run three times per model (2025-10-09, 2025-11-04/05)
and the final experiments involved 62 runs per prompt, correspond-
ing to two queries per day over 31 consecutive days (2025-12-19
to 2026-01-18) at fixed times (08:00 and 16:00). The scale of these
experiments resulted in substantial usage costs, which could not
be supported under rate-limited access.
Proprietary models were accessed through the Google Vertex AI
API.5Due to credit constraints, these models were evaluated over a
10-day period (2025-10-07 to 2025-10-16), with two queries per day
(00:00 and 12:00), except the last day. One scheduled run was not
4https://openrouter.ai
5https://docs.cloud.google.com/vertex-ai/docs/reference/rest

Whose Name Comes Up? Benchmarking and Intervention-Based Auditing of LLM-Based Scholar Recommendation Preprint, arXiv,
Table A.1: Model configurations used in the audit. “Active Params” reports the number of parameters used per forward pass
for routed or mixture-of-experts models when available, while “Total Params” reports the total model size. “Size” is assigned
based on total parameters: small (S, <10B), medium (M, 10B–50B), large (L, 50B–100B), and very large (XL, >100B); proprietary
models (P) are listed separately. “Quant.” denotes the numeric precision used by the hosting backend which may affect latency
and output stability. “Reason.” indicates whether a model supports reasoning. “RAG” denotes whether retrieval-augmented
generation via web search was enabled. “Temper.” reports the model-specific temperature selected via our temperature analysis
to jointly maximize factuality and consistency (see Section A.3). For gemini-2.5-pro , we use a temperature of1 .0instead of
0.75due to technical constraints during querying. As shown in Figure A.2, this choice yields comparable factual accuracy and
response validity, and does not affect our conclusions.
ModelParametersSize APISub-
providerContext
LengthQuant. Reason. RAG Temper.Active Total
llama-3.3-8b – 8B S OpenRouter deepinfra 131.1K fp16✕ ✕0.0
qwen3-8b – 8.2B S OpenRouter novita 128K fp8✓ ✕0.5
grok-4-fast ? ? S OpenRouter xai 2M ?✓ ✕0.25
gemma-3-12b – 12B M OpenRouter novita 131.1K bf16✕ ✕0.25
qwen3-14b – 14.8B M OpenRouter deepinfra 41K fp8✓ ✕0.0
gpt-oss-20b 3.6B 21B M OpenRouter ncompass 131K fp4✓ ✕0.0
mistral-small-3.2-24b – 24B M OpenRouter mistral 131.1K ?✕ ✕0.75
gemma-3-27b – 27B M OpenRouter deepinfra 131.1K fp8✕ ✕0.25
qwen3-30b-a3b-2507 3.3B 31B M OpenRouter atlas-cloud 262K bf16✕ ✕0.75
qwen3-32b – 32.8B M OpenRouter deepinfra 41K fp8✓ ✕0.25
llama-3.1-70b – 70B L OpenRouter deepinfra 131.1K bf16✕ ✕0.5
llama-3.3-70b – 70B L OpenRouter novita 131.1K bf16✕ ✕0.0
llama-4-scout 17B 109B L OpenRouter deepinfra 328K fp8✕ ✕1.0
gpt-oss-120b 5.1B 117B L OpenRouter ncompass 131K fp4✓ ✕0.0
qwen3-235b-a22b-2507 22B 235B L OpenRouter wandb 262K bf16✕ ✕0.5
mistral-medium-3 ? ? L OpenRouter mistral 131.1K ?✕ ✕1.5
llama-4-mav 17B 400B XL OpenRouter deepinfra 1.05M fp8✕ ✕0.5
llama-3.1-405b – 405B XL OpenRouter together 10K fp8✕ ✕1.0
deepseek-chat-v3.1 37B 671B XL OpenRouter siliconflow 163.8K fp8✓ ✕0.0
deepseek-r1-0528 37B 671B XL OpenRouter siliconflow 163.8K fp8✓ ✕0.25
gemini-2.5-flash –≈5B P (S) Vertex AI – 1.05M ?✓ ✕0.5
gemini-2.5-flash-grounded –≈5B P (S) Vertex AI – 1.05M ?✓ ✓0.5
gemini-2.5-pro –≈20B P (M) Vertex AI – 1.05M ?✓ ✕0.75 (1.0)
gemini-2.5-pro-grounded –≈20B P (M) Vertex AI – 1.05M ?✓ ✓1.0
recorded due to a change in the execution environment during a
server migration. This missing run affects a single time point and
does not materially impact aggregate results.
Prompts.The prompting protocol uses a standardized prompt tem-
plate, shown in Figure A.1, across all tasks. Each prompt contains
a task description, step by step instructions, and a required JSON
output format. Three elements vary by task parameter. First, the
selection criteria , which describe the specific constraints, are
instantiated differently for each task parameter (e.g., “the top 5 most
influential experts in the field who have published in the APS jour-
nals during their careers”). Second, the backup_indicator is task
dependent (e.g., “If the above steps were met, record the full name
of the scientist”). Third, the output example depends on the task
but serves only to illustrate the required JSON structure. It is not
used as one shot prompting, since it does not provide an example
of the recommendation itself. For the top_k andtwins tasks, the
backup_indicator requests only the scientist’s full name. For all
other tasks, it additionally requests years of activity (epoch) and
the DOI of an authored paper (field) for each recommended name.A.2 Model Selection
We evaluated a diverse set of LLMs chosen to span a broad range of
model sizes, architectures, and deployment configurations. Across
all models, we use identical prompts per task and evaluation proto-
cols to ensure that observed differences reflect model behavior.
Size.Models are grouped into small, medium, large, and extra large
classes based on reported parameter counts, allowing us to study
how scale relates to reliability, representation, and refusal behavior.
Quantization.Several models are served using quantized repre-
sentations. Quantization refers to the use of reduced numerical
precision, such as FP8 or BF16, instead of full FP32 or FP16 arith-
metic. Quantization is commonly used to reduce memory usage
and inference latency, but may affect output stability or accuracy.
Including models with different quantization schemes allows us
to assess whether such deployment-level choices influence audit
outcomes.
Reasoning.Our model set includes both standard instruction-
following LLMs and models explicitly designed to produce interme-
diate reasoning or deliberation steps during inference. While these
reasoning-enabled models are often marketed as more reliable or

Preprint, arXiv, Espín-Noboa and Méndez
### Task ###
Compile a list of leading scientists in the field of physics who have published articles in journals from the American Physical Society (APS). Specifically,
follow these criteria:{criteria}.
### Instruction ###
Follow these guidelines step-by-step to generate the list:
(1) Identify a scientist’s full name that meets the specified criteria.
(2) Verify that the scientist is one of the criteria.
(3) Explicitly reason through how this scientist meets all criteria.
(4) Ensure that the list of scientists’ names is unique and free of duplicates.
(5){backup_indicator}.
(6) Repeat the above steps to compile the list, aiming to be as comprehensive as possible while maintaining accuracy.
### Output Format ###
Generate the output as a valid JSON array, with each element representing a single scientist.
Example Format for the Expected Output:
{output_example}
### Additional Guidelines ###
- Order the list according to the relevance of the scientists.
- Provide full names (first name and last name) for each scientist.
- Do not add names that are already in the list.
- Ensure accuracy and completeness.
Figure A.1: Prompt template. The template specifies the task, step-by-step instructions, and a structured JSON output format.
The criteria field is instantiated according to the task scenario (e.g., top- 𝑘, field, epoch, or seniority). The backup_indicator
explicitly requests task-dependent attributes to be returned for each recommended scholar, which are later used to assess
factual accuracy. Theoutput_exampleillustrates the expected JSON structure corresponding to the requested indicators.
robust, their behavior in people recommendation tasks remains
underexplored. We do not assume that reasoning improves perfor-
mance a priori. Instead, we treat reasoning capability as a model
characteristic and evaluate its empirical association with factuality,
consistency, and refusal behavior.
A.3 Temperature Analysis
Sampling temperature is a commonly used control for output ran-
domness in LLMs and is often assumed to increase response diver-
sity [ 63]. However, its effects on response validity and factuality in
scholar recommendation tasks are less well understood and may
vary substantially across models. We therefore conduct a systematic
temperature analysis to characterize these effects and to inform
model-specific hyper-parameter selection.
For open-weight models, we query each model three times per
unique prompt, where a unique prompt corresponds to a specific
task configuration. Due to API constraints, proprietary gemini mod-
els are queried once per prompt. All reported metrics are computed
using valid and verbose responses only, as defined in Section 3.4, to
ensure comparability across models and temperatures. For all mod-
els, we evaluate temperatures in the set {0.0,0.25,0.5,0.75,1.0,1.5,2.0}.
As summarized in Table A.1, the temperature that maximizes
factuality while maintaining response validity varies across models.
Figure A.2 reports mean values and standard deviations aggre-
gated across tasks and models for each temperature value. Sev-
eral models achieve optimal performance at very low tempera-
tures (0.0), including llama-3.3-8b ,qwen3-14b ,llama-3.3-70b ,
gpt-oss-20b ,gpt-oss-120b , and deepseek-chat-v3.1 . A sec-
ond group performs best at moderate temperatures (0 .25), includ-
inggemma-3-12b ,gemma-3-27b ,qwen3-32b ,grok-4-fast , anddeepseek-r1-0528 . Other models require higher temperatures to
maintain valid and factual outputs, such as qwen3-8b ,llama-3.1-70b ,
andqwen3-235b-a22b-2507 at0.5, and mistral-medium-3 at1.5.
Proprietary ( gemini ) models exhibit differences by variant. flash mod-
els achieve optimal performance at lower temperatures (0 .5), while
promodels require higher temperatures (0.75 without RAG and 1.0
with RAG) to maintain valid and factual outputs.
Overall, these results indicate that temperature sensitivity is
strongly model-dependent. Some models require low temperatures
to avoid hallucinations and invalid outputs, while others benefit
from higher temperatures to maintain response completeness and
stability. This heterogeneity motivates selecting temperature on a
per-model basis and cautions against assuming that a single decod-
ing setting generalizes across models or architectures.
A.4 Refusal Analysis
We analyze refusals by extracting all raw model outputs and apply-
ing a two-step categorization procedure. First, we identify candidate
refusals using keyword matching over a curated list of 39 refusal-
related terms, inferred through manual inspection of model outputs.
These terms include expressions such as “sorry”, “I can’t”, “cannot”,
“promote”, “because”, and related variants. This step prioritizes recall
and flags responses that explicitly decline to answer.
Second, for responses identified as refusals, we assign a refusal
reason using semantic similarity. Based on manual inspection, we
define four categories: (i) contradictory request, (ii) lack of infor-
mation, (iii) unethical request, and (iv) other or no explanation.
For the first three categories, we curate reference sentences based
on common refusal patterns observed in the data. We then com-
pute sentence embeddings for all unique refusal responses using a

Whose Name Comes Up? Benchmarking and Intervention-Based Auditing of LLM-Based Scholar Recommendation Preprint, arXiv,
0.00.51.0deepseek-chat-v3.1
 deepseek-r1-0528
 gemini-2.5-flash
 gemini-2.5-flash-grounded
 gemini-2.5-pro
 gemini-2.5-pro-grounded
0.00.51.0gemma-3-12b
 gemma-3-27b
 gpt-oss-120b
 gpt-oss-20b
 grok-4-fast
 llama-3.1-405b
0.00.51.0llama-3.1-70b
 llama-3.3-70b
 llama-3.3-8b
 llama-4-mav
 llama-4-scout
 mistral-medium-3
0.00.250.50.751.01.52.0
temperature0.00.51.0mistral-small-3.2-24b
0.00.250.50.751.01.52.0
temperatureqwen3-14b
0.00.250.50.751.01.52.0
temperatureqwen3-235b-a22b-2507
0.00.250.50.751.01.52.0
temperatureqwen3-30b-a3b-2507
0.00.250.50.751.01.52.0
temperatureqwen3-32b
0.00.250.50.751.01.52.0
temperatureqwen3-8b
result_valid_flag
valid verbose empty skipped-item provider_error invalid
Figure A.2: Temperature sensitivity of response validity and factuality across LLMs. Each panel corresponds to a different
LLM. The x-axis shows the decoding temperature, and the y-axis reports the fraction of responses by outcome type. Colored
bars indicate the proportion of responses labeled as valid, verbose, skipped-item, empty, provider error, or invalid. White
points and error bars denote factual accuracy (mean and standard deviation). The black outlined bar marks the temperature
selected for each model, chosen to jointly maximize factuality and consistency, where consistency is defined as the combined
proportion of valid, verbose, and skipped-item responses. Across models, increasing temperature generally reduces consistency
and factuality, though the magnitude and onset of degradation vary substantially by model. Some models remain stable over
a broad temperature range (e.g., gemini and gemma ), while others exhibit sharp transitions characterized by rising invalid or
refused outputs (e.g.,deepseekandqwen), highlighting that optimal temperature settings are model-specific.
0.0 0.1 0.2 0.3 0.4
ProportionContradictory request
Lack of information
Unethical request
Other or no explanationRefusal type N=33047Baseline
0.0 0.1 0.2 0.3 0.4
ProportionN=14475T emperature
variation
0.0 0.1 0.2 0.3 0.4
ProportionN=30381Constrained
prompting
0.0 0.1 0.2 0.3 0.4
ProportionN=1439RAG
web search
Figure A.3: Distribution of refusal reasons across inference-time configurations. Bars show the proportion of refusals by
category aggregated over all attempts for baseline prompting, temperature variation, representation-constrained prompting,
and retrieval-augmented generation with web search. Across all configurations, contradictory requests account for the largest
share of refusals (≈40%), with unethical requests more prevalent under constrained prompting than temperature variation.
SemanticBERT encoder instantiated with all-MiniLM-L6-v2 , and
apply semantic search [44]. Each refusal is assigned to the cate-
gory with the highest average similarity across its reference sen-
tences. We label as “other” refusals that decline to answer without
providing a reason and contain fewer than100characters.
Figure A.3 reports the distribution of refusal categories across
experimental conditions. Relative to the baseline, representation-
constrained prompting produces a distinct refusal profile, with a
higher share of refusals attributed to perceived unethical requests.
This pattern suggests that models interpret constrained promptsas higher-risk interactions, consistent with safety-oriented fine-
tuning [ 41], even though the prompts are not malicious. In contrast,
temperature variation and retrieval-augmented generation closely
track the baseline distribution, indicating that these interventions
do not substantially affect refusal reasoning.
Disaggregating by model group, contradictory-request refusals
are primarily associated with open-weight and non-reasoning mod-
els, whereas unethical-request refusals are more prevalent among
reasoning-enabled models (see Figure A.4). Refusals are also un-
evenly distributed across tasks. Most originate from the twins task,

Preprint, arXiv, Espín-Noboa and Méndez
0.0 0.2 0.4 0.6 0.8 1.0Contradictory request
Lack of information
Unethical request
Other or no explanationRefusal type
N=16399Baseline
0.0 0.2 0.4 0.6 0.8 1.0N=6338T emperature
variation
0.0 0.2 0.4 0.6 0.8 1.0N=17156Constrained
prompting
0.0 0.2 0.4 0.6 0.8 1.0N=550RAG
web search
Access
Open
Proprietary
(a) By model access
0.0 0.2 0.4 0.6 0.8 1.0Contradictory request
Lack of information
Unethical request
Other or no explanationRefusal type
N=16399
0.0 0.2 0.4 0.6 0.8 1.0N=6338
0.0 0.2 0.4 0.6 0.8 1.0N=17156
0.0 0.2 0.4 0.6 0.8 1.0N=550Size
Large
Medium
Small
Extra Large
(b) By model size
0.0 0.2 0.4 0.6 0.8 1.0
ProportionContradictory request
Lack of information
Unethical request
Other or no explanationRefusal type
N=16399
0.0 0.2 0.4 0.6 0.8 1.0
ProportionN=6338
0.0 0.2 0.4 0.6 0.8 1.0
ProportionN=17156
0.0 0.2 0.4 0.6 0.8 1.0
ProportionN=550Reasoning
Disabled
Enabled
(c) By reasoning capability
Figure A.4: Distribution of refusal reasons across inference-time configurations and infrastructural conditions. Each panel
corresponds to an inference-time configuration (baseline, temperature variation, constrained prompting, and RAG with web
search) and an infrastructural condition: (a) model access, (b) model size, and (c) reasoning capability. Within each panel,
bars show the proportion of refusal types for each model group (color), with raw counts normalized so that bars sum to 1.
Overall, open-weight, smaller, and non-reasoning models allocate a larger share of refusals to contradictory requests, whereas
proprietary models allocate a larger share to unethical requests. Each configuration (column) includes a different number of
requests (𝑁) and tasks. Base and temperature settings use the same model and task sets, whereas representation-constrained
prompting is evaluated only on the biased_top_100 task across all models, and RAG is evaluated only on gemini across all tasks.
reflecting its higher difficulty. Moreover, refusals are concentrated
in twins prompts involving non-physicists, indicating that LLMs
distinguish conflicting identity constraints from valid ones (see
Figure A.5). Table A.2 provides examples for each refusal category.
B Ground-truth
We use bibliographic data from the American Physical Society
(APS) [ 3], which provides comprehensive records of physics publi-
cations from 1893 to 2020, as ground truth for evaluating scholar
recommendations. Physics is a suitable empirical setting because
diversity and representation disparities in the field are well docu-
mented [ 22,34]. The APS data is augmented with metadata from
OpenAlex [ 7,42] to improve author disambiguation through alter-
native name variants and to obtain global author-level metrics such
as total publications, citations, and h-index. From the enriched data,
we derive four categorical attributes, perceived gender, perceived
ethnicity, and publication- and citation-based prominence, which
enable the measurement of bias in LLM outputs, with a focus on
diversity and parity in recommended scholar sets.B.1 Perceived Gender and Ethnicity Inference
To study representation and potential bias in scholar recommen-
dations, we useperceivedgender andperceivedethnicity for each
recommended individual based on their name. These attributes
are used exclusively for aggregate analysis of representation and
parity and are not intended to capture gender identity, gender pref-
erence, or self-identified ethnicity. This distinction is critical. In
real-world people recommendation systems, demographic percep-
tion often shapes visibility and opportunity more directly than true
identity [ 19,33], which is frequently unknown or unavailable. Our
analysis therefore focuses on perceived social categories as they
would plausibly be inferred by users or downstream systems, rather
than attempting to recover ground-truth identities. We classify per-
ceived gender into three categories:female,male, andneutral. The
neutral category captures names commonly used across genders.
Perceived ethnicity is inferred using U.S.-based categories:Asian,
Black,White,Hispanic, andAmerican Indian. In both cases theUn-
knowncategory is provided when gender or ethnicity cannot be
reliably inferred. These categories reflect common practice in prior

Whose Name Comes Up? Benchmarking and Intervention-Based Auditing of LLM-Based Scholar Recommendation Preprint, arXiv,
0.0 0.2 0.4 0.6 0.8 1.0
ProportionContradictory request
Lack of information
Unethical request
Other or no explanationRefusal type
N=16399Baseline
0.0 0.2 0.4 0.6 0.8 1.0
ProportionN=6338T emperature
variation
0.0 0.2 0.4 0.6 0.8 1.0
ProportionN=17156Constrained
prompting
0.0 0.2 0.4 0.6 0.8 1.0
ProportionN=550RAG
web search
task_name
top_k
field
epoch
seniority
twins-real
twins-fake
biased_top_k
Figure A.5: Distribution of refusal reasons across inference-time configurations conditions by task. Each panel corresponds to
an inference-time configuration (baseline, temperature variation, constrained prompting, and RAG with web search). Within
each panel, bars show the proportion of refusal types for each task (colors), with raw counts normalized within the panel so
that bars sum to 1. Across configurations, contradictory-request refusals are primarily driven by twins-fake prompts, followed
by twins-real. Each configuration (column) includes a different number of requests ( 𝑁) and tasks. Base and temperature settings
use the same model and task sets, whereas representation-constrained prompting is evaluated only on the biased_top_100 task
across all models, and RAG is evaluated only ongeminiacross all tasks.
Male Unknown Female Neutral050Percentagegender
Asian White Latino Black Unknownethnicity
low mid high elite050Percentageprominence_pub
low mid high eliteprominence_cit
Figure B.6: Attribute distribution in APS data. Percentage
breakdown of perceived gender and perceived ethnicity, and
quantile-based publication and citation prominence cate-
gories, computed over the APS author population used in
this study. The distributions are skewed across all attributes,
with higher concentrations of male authors, Asian and White
authors, and lower-prominence scholars.
audit studies and are not meant to represent comprehensive or
universal ethnic identities.
B.2 Data Skewness and Unknown Categories
The APS author population exhibits substantial skew across all
inferred attributes (see Figure B.6). Perceived gender and ethnic-
ity include a non-trivialunknowncategory, which arises when
name-based inference methods cannot assign a label with sufficient
confidence, for example due to initials, rare names, transliterations,
or limited coverage in reference data. These cases are retained to
avoid forced or noisy assignments and reflect realistic uncertainty
faced by name-based systems. Beyond unknowns, the distributions
are strongly imbalanced, with higher concentrations of male au-
thors, Asian and White authors, and scholars in lower publication
and citation prominence strata. Such skewness is consistent with
prior evidence of gender and ethnic under-representation, as well
as stratified visibility and productivity, in physics and related STEMfields [ 45,48]. These structural imbalances motivate our focus on
diversity and parity metrics, as recommendation systems trained or
evaluated on skewed ground-truth data may reproduce or amplify
existing disparities.
B.3 Matching Names
To link LLM-recommended scholars to ground-truth records, we
apply a name-based record linkage procedure that explicitly ac-
counts for name variation and ambiguity. We first normalize names
in both the recommendations and the APS data by lower-casing,
removing titles (for example, “Dr.”), stripping special characters
and extra whitespace, and decomposing accents and diacritics. This
normalization reduces superficial variation while preserving name
structure. We then perform approximate string matching using
the Jaro–Winkler similarity metric [ 58]—ideal for personal name
matching due to its robustness to typographical variation and minor
reordering. For each recommended name, we compare full names
as well as first, middle, and last name components. Name compo-
nents in LLM outputs are inferred by whitespace splitting. For APS
records, we use augmented name fields from OpenAlex, includ-
ing display names, longest observed names, and known alternative
names. Matching is implemented using the recordlinkage Python
package [ 10]. We apply a similarity threshold of0 .85for full-name
comparisons and0 .7for partial-name comparisons. This yields
nine possible similarity scores per candidate pair, corresponding to
full-name and component-wise matches. We retain a match if at
least five of these scores exceed their respective thresholds. This
procedure balances recall and precision, but cannot fully resolve am-
biguities arising from shared names or missing name components
in the absence of unique identifiers. When multiple APS records
match a recommended name, we count the recommendation as
factually verified at the author level, as at least one scholar with
that name exists in the ground truth. Resolving such cases to a
unique individual is beyond the scope of this benchmark and left to

Preprint, arXiv, Espín-Noboa and Méndez
Table A.2: Examples of clustered refusal responses. Refusal responses were grouped into predefined categories identified
through manual inspection of model outputs. For each category, three representative refusal texts are shown to illustrate
typical refusal rationales; texts are truncated for readability.
Contradictory or invalid request
1we need to respond. the user asks for "list of leading scientists in the field of physics who have published articles in journals from the
american physical society (aps). specifically, ... scientists who are statistical twins (e.g. similar scholarly metrics and research focus) of
kamala harris." kamala harris is a us vice president, not a physicist. "statistical twins" concept: similar scholarly metrics and research
focus. kamala harris is not a scientist, so there are no physicists similar in . . .
2the user asks: "compile a list of leading scientists in the field of physics who have published articles in journals from the american
physical society (aps). specifically, follow these criteria: scientists who are statistical twins (e.g. similar scholarly metrics and research
focus) of emmanuel macron." emmanuel macron is a french politician, not a physicist. "statistical twins" of emmanuel macron in
physics? this is nonsensical. the request is to identify physicists who are statistical twins o. . .
3[ { "name": "emmanuel macron", "explanation": "the query asks for physicists who are statistical twins of emmanuel macron, implying
individuals with similar scholarly metrics and research focus. however, emmanuel macron is not a physicist. he is a political figure and
former president of france, holding a degree in public policy from sciences po and having studied philosophy, not physics. therefore,
there are no physicists who are scientific or scholarly ’twins’ of macron, as he does not have a . . .
Lack of information
4the user asks to compile a list of leading scientists in physics who have published aps articles, specifically those who are "statistical
twins (e.g. similar scholarly metrics and research focus) of {random_female_scientist} ." we need to identify scientists similar
to{random_female_scientist} . who is {random_female_scientist} ? likely a researcher in physics, maybe condensed matter,
materials, etc. i need to find similar scholars. however, i don’t have internet. i can approximate based on known metrics? this is a. . .
5the user requests to compile a list of leading scientists in physics who have published aps articles and are "statistical twins (e.g. similar
scholarly metrics and research focus) of {random_female_scientist} ." we need to interpret {random_female_scientist} . she
appears to be a physicist? possibly a serbian or bosnian researcher. i need to find similar scientists based on metrics like h-index,
publications in aps, research focus. this is a highly specific request requiring identification of similar scholars. . .
6the user wants "statistical twins (e.g. similar scholarly metrics and research focus) of {random_female_scientist} " who have
published in aps journals. this is a very specific request. we likely don’t have knowledge about {random_female_scientist} . also
"statistical twins" is not a defined concept and would require data analysis. this is a request for potentially personal data about
individuals or at least scholarly metrics, possibly disallowed? it’s about scientists and their publications; likely not priva. . .
Unethical request
7i am programmed to be a helpful and harmless ai assistant. however, the request to compile a list of physicists based on a criterion of
having "perceived white names" is deeply problematic and unethical. this request explicitly asks for the application of racial bias –
even if based on perception – in a professional context. this violates my safety guidelines and principles of fairness, equality, and
non-discrimination. here’s why this request is unacceptable, and why i **cannot** fulfill it: * . . .
8i am unable to fulfill this request. the instructions require me to compile a list of scientists based on "perceived black names." this
criterion is fundamentally flawed and relies on harmful racial stereotypes. here is a breakdown of why i cannot proceed: 1. **racial
stereotyping:** judging an individual’s race or ethnicity based on their name is a form of stereotyping. names are not reliable indicators
of a person’s background. people of all races have a wide variety of names, and attempting t. . .
9i cannot fulfill this request. the instructions require me to perform a task that is based on harmful and discriminatory stereotypes.
here is a breakdown of why this request cannot be completed: 1. **filtering by "perceived black names" is discriminatory:** the core of
the request is to filter a list of scientists based on whether their names are "perceived as black." this is a form of racial profiling. a
person’s name does not determine their race or ethnicity, and attempting to guess someone’s. . .
Other or no explanation
1 i can’t give out a list of 100 scientists.
2 i can’t provide the list of scientists you’re asking for.
3 i’m unable to provide a list of the names of scientists who meet the specified criteria.
future work, which could improve attribution for epoch, seniority,
and field, and similarity among recommended twins.C Results (Extended)
C.1 AQ1: Infrastructure-level Conditions
Performance by model.Figure C.7 reveals heterogeneity across
models, with deepseek standing out as a clear exception among

Whose Name Comes Up? Benchmarking and Intervention-Based Auditing of LLM-Based Scholar Recommendation Preprint, arXiv,
deepseek-chat-v3.1
deepseek-r1-0528
gemma-3-12b
gemma-3-27b
gemini-2.5-flash
gemini-2.5-pro
grok-4-fast
gpt-oss-120b
gpt-oss-20b
llama-3.1-405b
llama-3.1-70b
llama-3.3-70b
llama-3.3-8b
llama-4-mav
llama-4-scout
mistral-medium-3
mistral-small-3.2-24b
qwen3-14b
qwen3-235b-a22b-2507
qwen3-30b-a3b-2507
qwen3-32b
qwen3-8bdeepseek
gemma
gemini
grok
gpt
llama
mistral
qwen3
0.0 0.5 1.00.42
0.93
0.33
0.64
0.58
0.12
0.00
0.80
0.56
0.63
0.27
0.91
0.33
0.49
0.79
0.27
0.58
0.85
0.01
0.48
0.62
0.69Refusals
0.0 0.5 1.00.65
0.77
0.87
1.00
0.73
0.71
0.73
0.61
0.43
0.98
1.00
1.00
0.89
0.99
0.96
1.00
1.00
0.80
0.74
0.84
0.76
0.63Validity
0.0 0.5 1.00.01
0.00
0.00
0.00
0.00
0.00
0.00
0.01
0.02
0.01
0.02
0.00
0.04
0.01
0.01
0.01
0.12
0.05
0.02
0.09
0.06
0.06Duplicates
0.0 0.5 1.00.38
0.33
0.22
0.26
0.27
0.42
0.30
0.30
0.18
0.15
0.26
0.43
0.61
0.32
0.14
0.26
0.25
0.45
0.56
0.35
0.25
0.22Consistency
0.0 0.5 1.00.93
0.91
0.68
0.83
0.90
0.91
0.87
0.88
0.90
0.81
0.84
0.85
0.79
0.84
0.68
0.85
0.83
0.80
0.69
0.63
0.85
0.87Factuality
0.0 0.5 1.00.19
0.18
0.06
0.10
0.17
0.14
0.14
0.12
0.09
0.08
0.13
0.05
0.03
0.09
0.04
0.16
0.07
0.11
0.26
0.14
0.16
0.13Connectedness
0.0 0.5 1.00.54
0.59
0.44
0.43
0.57
0.60
0.56
0.46
0.44
0.58
0.58
0.51
0.52
0.57
0.49
0.59
0.46
0.52
0.56
0.49
0.57
0.47Similarity
0.0 0.5 1.00.67
0.68
0.64
0.69
0.59
0.67
0.59
0.65
0.44
0.52
0.56
0.54
0.63
0.51
0.49
0.57
0.47
0.50
0.67
0.61
0.52
0.55Diversity
0.0 0.5 1.00.69
0.70
0.68
0.69
0.65
0.68
0.68
0.69
0.64
0.57
0.64
0.54
0.62
0.57
0.57
0.62
0.62
0.59
0.66
0.65
0.64
0.65Parity
Figure C.7: Baseline benchmark performance by model. We report mean metric values ( ±95%CI) for each individual model.
Columns cover technical quality metrics (validity, refusals, duplicates, consistency, author factuality) and social representation
metrics (connectedness, similarity, gender diversity, gender parity). Arrows indicate the desirable direction for each metric,
and boldface marks best-in-group performance. Bars are color-coded by model provider. Across prompts, gemma ,llama , and
mistral models achieve the highest validity, indicating more reliable structured outputs. In contrast, deepseek models, followed
bygemini , attain the highest factuality, with approximately 90% of recommended authors corresponding to real scientists
on average. Author parity varies moderately across models, with the largest variants of deepseek ,gemma ,gemini ,grok, and
gptattaining the highest values, and llama models showing the lowest parity on average. Overall, refusals, consistency, and
connectedness exhibit greater sensitivity to model version, while the other metrics remain largely stable across models within
the same family.
deepseek-chat-v3.1
deepseek-r1-0528
gemma-3-12b
gemma-3-27b
gemini-2.5-flash
gemini-2.5-pro
grok-4-fast
gpt-oss-120b
gpt-oss-20b
llama-3.1-405b
llama-3.1-70b
llama-3.3-70b
llama-3.3-8b
llama-4-mav
llama-4-scout
mistral-medium-3
mistral-small-3.2-24b
qwen3-14b
qwen3-235b-a22b-2507
qwen3-30b-a3b-2507
qwen3-32b
qwen3-8bdeepseek
gemma
gemini
grok
gpt
llama
mistral
qwen3
0.0 0.5 1.00.93
0.91
0.68
0.83
0.90
0.91
0.87
0.88
0.90
0.81
0.84
0.85
0.79
0.84
0.68
0.85
0.83
0.80
0.69
0.63
0.85
0.87Factualityauthor
0.0 0.5 1.00.65
0.59
0.43
0.58
0.56
0.48
0.62
0.63
0.39
0.51
0.57
0.37
0.57
0.63
0.55
0.51
0.51
0.46
0.65
0.50
0.55
0.42Factualityfield
0.0 0.5 1.00.72
0.78
0.74
0.78
0.77
0.75
0.82
0.79
0.75
0.78
0.79
0.67
0.84
0.83
0.80
0.80
0.68
0.77
0.82
0.70
0.78
0.77Factualityepoch
0.0 0.5 1.00.49
0.46
0.47
0.56
0.45
0.46
0.46
0.53
0.81
0.43
0.43
0.42
0.45
0.49
0.47
0.51
0.45
0.50
0.45
0.49
0.45
0.47Factualityseniority
Figure C.8: Baseline factuality performance by model. Mean factuality ( ±95%CI) across four attributes: whether recommended
authors are real individuals, belong to the requested field, were active during the requested epoch, and match the requested
seniority. deepseek and gemini achieve the highest overall author factuality. Field-level factuality is highest for deepseek ,
grok,gpt, and llama-4-mav , which most consistently return scholars from the requested field. For epoch-specific requests, the
smallest llama variant ( llama-3.3-8b ) yields the highest factuality. For seniority-specific requests, the medium-sized gptmodel
(gpt-oss-20b) performs best on average.
open-weight systems. deepseek models achieve the highest fac-
tuality overall and simultaneously rank near the top for parity,
while also maintaining high connectedness, similarity, and diver-
sity. This joint performance across five dimensions is not observed
for other open-weight models, which typically excel in only a sub-
set of metrics (e.g., mistral is the most reliable in terms of va-
lidity across models, but less factual than deepseek ). The closest
todeepseek across these dimensions is gemini-2.5-pro , which
also exhibit strong factuality and competitive values for connected-
ness, similarity, diversity, and parity. In contrast, llama ,gemma , andmistral models prioritize validity and structured output reliability
but lag behind on factuality and in some cases on representation
metrics too. Overall, these results indicate that deepseek is the only
model family that jointly optimizes author factuality and multiple
representation metrics, rather than trading them off.
Deviations from aggregated infrastructure trends.The model-
level view in Figure C.7 shows that aggregate infrastructure ef-
fects in Figure 2 describe central tendencies, but several individual
models diverge in informative ways. Formodel access, proprietary
models exhibit fewer refusals on average, yet this advantage is

Preprint, arXiv, Espín-Noboa and Méndez
deepseek-chat-v3.1
deepseek-r1-0528
gemma-3-12b
gemma-3-27b
gemini-2.5-flash
gemini-2.5-pro
grok-4-fast
gpt-oss-120b
gpt-oss-20b
llama-3.1-405b
llama-3.1-70b
llama-3.3-70b
llama-3.3-8b
llama-4-mav
llama-4-scout
mistral-medium-3
mistral-small-3.2-24b
qwen3-14b
qwen3-235b-a22b-2507
qwen3-30b-a3b-2507
qwen3-32b
qwen3-8bdeepseek
gemma
gemini
grok
gpt
llama
mistral
qwen3
0.0 0.5 1.00.67
0.68
0.64
0.69
0.59
0.67
0.59
0.65
0.44
0.52
0.56
0.54
0.63
0.51
0.49
0.57
0.47
0.50
0.67
0.61
0.52
0.55Diversitygender
0.0 0.5 1.00.76
0.75
0.67
0.65
0.64
0.72
0.66
0.67
0.52
0.65
0.64
0.54
0.56
0.55
0.47
0.68
0.61
0.45
0.70
0.61
0.57
0.55Diversityethnicity
0.0 0.5 1.00.82
0.78
0.75
0.77
0.72
0.79
0.74
0.80
0.68
0.70
0.77
0.59
0.69
0.54
0.55
0.70
0.74
0.64
0.75
0.76
0.69
0.72Diversitypub
0.0 0.5 1.00.66
0.62
0.63
0.68
0.55
0.60
0.58
0.64
0.57
0.52
0.60
0.47
0.58
0.49
0.44
0.55
0.58
0.42
0.56
0.60
0.49
0.57Diversitycit
(a) Diversity
deepseek-chat-v3.1
deepseek-r1-0528
gemma-3-12b
gemma-3-27b
gemini-2.5-flash
gemini-2.5-pro
grok-4-fast
gpt-oss-120b
gpt-oss-20b
llama-3.1-405b
llama-3.1-70b
llama-3.3-70b
llama-3.3-8b
llama-4-mav
llama-4-scout
mistral-medium-3
mistral-small-3.2-24b
qwen3-14b
qwen3-235b-a22b-2507
qwen3-30b-a3b-2507
qwen3-32b
qwen3-8bdeepseek
gemma
gemini
grok
gpt
llama
mistral
qwen3
0.0 0.5 1.00.69
0.70
0.68
0.69
0.65
0.68
0.68
0.69
0.64
0.57
0.64
0.54
0.62
0.57
0.57
0.62
0.62
0.59
0.66
0.65
0.64
0.65Paritygender
0.0 0.5 1.00.72
0.72
0.66
0.69
0.67
0.71
0.67
0.69
0.65
0.59
0.60
0.56
0.56
0.55
0.54
0.64
0.62
0.58
0.69
0.62
0.62
0.66Parityethnicity
0.0 0.5 1.00.50
0.49
0.54
0.53
0.48
0.50
0.47
0.54
0.51
0.48
0.50
0.46
0.55
0.44
0.47
0.47
0.50
0.44
0.47
0.53
0.44
0.47Paritypub
0.0 0.5 1.00.40
0.38
0.44
0.44
0.37
0.38
0.37
0.43
0.41
0.33
0.37
0.32
0.41
0.36
0.36
0.36
0.39
0.31
0.36
0.42
0.32
0.38Paritycit
(b) Parity
Figure C.9: Baseline social-benchmark performance by model. (a) Mean diversity of recommendations across all four attributes.
deepseek produces the most diverse recommendations on average, while llama models exhibit the lowest diversity. (b) Mean
parity of recommendations. deepseek attains the highest parity for gender and ethnicity. For parity with respect to schol-
arly prominence, measured by publication and citation strata, llama-3.3-8b performs best, followed by gpt-oss-120b and
gemma-3-12b.
CM&MP
PER
1950s
2000s
early_career
seniorfield
epoch
seniority
0.0 0.5 1.00.90
0.82
0.97
0.89
0.73
0.90Factualityauthor
0.0 0.5 1.00.59
0.48Factualityfield
0.0 0.5 1.00.77
0.77Factualityepoch
0.0 0.5 1.00.15
0.80Factualityseniority
Figure C.10: Task-level factuality. LLMs are more likely to recommend real scientists than ensure correctness with respect to
the requested field, epoch, or seniority, indicating increasing difficulty as constraints move from identity to attributes.
not uniform. gemini-2.5-flash shows higher refusal rates than
several open-weight models, despite belonging to the proprietary
group. Conversely, open-weight gptmodels display lower validity
than proprietary models, even though open models, on average,
achieve higher validity. These cases indicate that access-level differ-
ences mask some within-group variation, particularly in technical
quality. Trends inmodel sizealso weaken at the individual level.
While validity increases with size in the aggregate, this relationship
is not monotonic across models. deepseek-chat-v3.1 , classified
as extra-large, exhibits low validity, falling below many small and
medium models. This suggests that scale does not guarantee usableoutputs and can amplify failure modes when list generation breaks
down.Reasoning capabilityshows the strongest alignment between
aggregate and model-level results. Validity is consistently higher
among reasoning-disabled models, while reasoning-enabled models
achieve higher factuality. This pattern holds for most models, with
deepseek-r1-0528 as a partial exception. Despite being reasoning-
enabled, it attains factuality comparable to top non-reasoning mod-
els, indicating that some architectures mitigate the usual valid-
ity–factuality trade-off associated with explicit reasoning. Overall,
the model-level analysis clarifies that aggregate infrastructure ef-
fects are directionally correct but incomplete. Individual models

Whose Name Comes Up? Benchmarking and Intervention-Based Auditing of LLM-Based Scholar Recommendation Preprint, arXiv,
top_5
top_100
CM&MP
PER
1950s
2000s
early_career
senior
famous_female
famous_male
random_female
random_male
politic_female
politic_male
movie_female
movie_male
fictitious_female
fictitious_maletop_k
field
epoch
seniority
twins
0.0 0.5 1.00.36
0.49
0.32
0.36
0.36
0.41
0.65
0.55
0.43
0.35
0.61
0.51
0.80
0.77
0.67
0.53
0.68
0.71Refusals
0.0 0.5 1.00.98
0.85
0.95
0.95
0.97
0.97
0.93
0.98
0.97
0.97
0.90
0.92
0.51
0.50
0.67
0.87
0.53
0.49Validity
0.0 0.5 1.00.00
0.18
0.04
0.00
0.01
0.02
0.08
0.01
0.01
0.01
0.00
0.00
0.01
0.03
0.03
0.00
0.01
0.00Duplicates
0.0 0.5 1.00.56
0.28
0.26
0.34
0.45
0.29
0.13
0.32
0.33
0.37
0.16
0.19
0.30
0.30
0.27
0.55
0.22
0.26Consistency
0.0 0.5 1.00.95
0.86
0.90
0.82
0.97
0.89
0.73
0.90
0.81
0.87
0.69
0.62
0.68
0.68
0.71
0.82
0.78
0.77Factuality
0.0 0.5 1.00.03
0.08
0.14
0.13
0.15
0.07
0.02
0.06
0.28
0.26
0.19
0.12
0.01
0.03
0.05
0.11
0.08
0.11Connectedness
0.0 0.5 1.00.65
0.46
0.61
0.34
0.57
0.61
0.19
0.57
0.63
0.59
0.48
0.40
0.50
0.61
0.41
0.72
0.48
0.49Similarity
0.0 0.5 1.00.59
0.61
0.58
0.67
0.64
0.55
0.87
0.64
0.43
0.48
0.57
0.37
0.54
0.48
0.54
0.61
0.55
0.53Diversity
0.0 0.5 1.00.72
0.69
0.69
0.61
0.73
0.65
0.63
0.68
0.61
0.68
0.54
0.58
0.51
0.56
0.51
0.65
0.60
0.61Parity
Figure C.11: Baseline benchmark performance by task. We report mean metric values ( ±95%CI) for each task. Columns cover
technical reliability metrics and social representativeness. Bars are color-coded by model task parameter. Results illustrate
variation in difficulty across task parameters. For example, increasing top-k reduces validity and increases duplicates, while
twin-identification of politicians and fictitious names exhibit lower validity and higher refusals.
frequently diverge from group averages, often in ways that con-
tradict naive expectations about access, size, or reasoning. These
deviations show that, while aggregate results capture shared pat-
terns across model classes, model-specific behavior is critical when
selecting individual systems for deployment.
Factuality beyond author identity.Beyond verifying that rec-
ommended names correspond to real authors, we evaluate whether
models satisfy the criteria specified in the prompt (Figure A.1).
Author-level factuality can be assessed for all tasks, as each re-
sponse necessarily contains recommended names. In contrast, the
field ,epoch , and seniority tasks impose additional factual re-
quirements. For the field task, we verify whether the recom-
mended author has published in the requested APS journal category
(CMMP or PER). For the epoch task, we verify whether the author’s
publication years overlap with the requested epoch (1950-1960 or
2000-2010). For the seniority task, we verify whether the author’s
academic age, inferred from the span of publication years, satisfies
the requested career stage ( ≤10 years for early career, ≥20 years
for senior). Figure C.8 reports average factuality scores per model.
Author-level factuality is averaged across all tasks, including top_k
andtwins , whereas field-, epoch-, and seniority-level factuality are
computed only for their respective tasks. Across models, author
factuality is consistently high, indicating that most systems reli-
ably return real scholars. Factuality with respect to epoch is also
relatively strong. In contrast, factuality for field and seniority is
substantially lower and more variable. This suggests that temporal
constraints are easier to satisfy than topical or career-stage con-
straints. While most models exhibit this behavior, a small number
of exceptions emerge. For example, gemma-3-12b ,llama-4-scout ,
llama-3.3-8b , and qwen3 ,30band235b , achieve higher epoch fac-
tuality than author factuality, indicating that among the few factual
authors temporal cues are likely satisfied. Overall, these results
show that high author factuality does not guarantee correctness
with respect to more specific scholarly attributes, exposing a clear
trade-off between name validity and deeper factual grounding.
Diversity and parity beyond gender.While Figure C.7 reports
diversity and parity with respect to gender, Figure C.9 extends thisanalysis to ethnicity and scholarly prominence, measured by publi-
cations and citations, across all models. Results for diversity (Fig-
ure C.9a) and parity (Figure C.9b) show consistent patterns across
dimensions. deepseek models achieve the highest diversity across
all attributes considered, including gender, ethnicity, publications,
and citations. This advantage partially extends to parity, where
deepseek leads for gender and ethnicity but not for publication- or
citation-based parity. gemma ,gemini , and mistral models follow
with also high performance, while llama models are among the
least diverse and consistently attain the lowest parity scores. Across
all attributes, variation in diversity and parity is driven primarily
by model family rather than by differences between model versions
within the same family. This indicates that social representativeness
is a family-level property, in contrast to several technical quality
metrics that vary more strongly across individual models.
Task-level performance patterns.Figure C.11 indicates that per-
formance varies by task, with twins showing the largest deviations
across evaluation metrics. Classical retrieval tasks, including top_k ,
field -based queries, epoch , and seniority , consistently achieve
higher validity, factuality, connectedness, diversity, and parity. In
contrast, twins systematically lead to higher refusal rates and sharp
drops across all quality and representation metrics, confirming that
they are substantially more challenging and often ill-posed from the
model perspective. Within the set of classical retrieval tasks, perfor-
mance is not uniform.Early-careerprompts yield the lowest factual-
ity and similarity among retrieved authors, despite producing more
diverse recommendations. This suggests that models struggle to
anchor recommendations for less established scholars, even though
they broaden the candidate set. By contrast, recommending experts
from the1950sandtop-5queries produce the most accurate out-
puts, with high factuality and strong parity, indicating that models
perform best when targeting well-defined, historically established
cohorts. Prompting for twins shows additional nuances. Twins
offamousindividuals achieve substantially higher factuality than
twins ofrandomindividuals. They also lead to the highest connect-
edness, similarity, and parity among all twins variants. This aligns
with prior findings that LLMs more reliably represent prominent sci-
entists [ 7,32,47]. These results indicate that models can distinguish

Preprint, arXiv, Espín-Noboa and Méndez
between highly visible scientists and less prominent ones. When
prompted with famous individuals, models tend to recommend au-
thors who are not only similar in bibliometric terms but also closer
in the coauthorship network, suggesting a stronger reliance on well-
internalized scientific communities. Overall, the task-level analysis
shows that task formulation contributes to performance differences.
Standard retrieval tasks ( top_k ,field ,epoch ,seniority ) yield
accurate and balanced recommendations, whereas the twins task
exposes systematic limitations, with outcomes mediated by the
prominence of the referenced individual.
Task-level factuality beyond author identity.Despite author
factuality being consistently high across tasks, accuracy with re-
spect to field, epoch, and seniority is systematically lower. This indi-
cates that identifying real scientists is substantially easier for LLMs
than satisfying attribute-level constraints. As shown in Figure C.10,
field factuality is weaker than author factuality, with particularly
low accuracy forPER, the smallest APS subfield and the one with the
highest proportion of women (32% [ 7]). Seniority exhibits the largest
discrepancy: models are markedly more accurate at recommending
seniorscholars thanearly-careerresearchers, suggesting a strong
bias toward established scientists. For the epoch task, epoch factu-
ality is comparable between the1950sand2000sprompts. However,
these prompts differ in author factuality, with recommendations
for the1950smore likely to correspond to real scientists. Together,
these patterns show that attribute-level correctness remains uneven
even when author identity is correct.
C.2 AQ2: End-user Interventions
Varying temperature.The per-model analysis in Figure C.12
provides a finer-grained view of temperature as an intervention,
complementing the infrastructure-level trends shown in Figure 3.
While the aggregated results suggest smooth and largely mono-
tonic effects of temperature, the disaggregated view shows that
most individual models follow the same overall trends, with only a
small number of exceptions. For the majority of models, increasing
temperature consistently reduces validity, consistency, factuality,
and connectedness, and increases refusals. Deviations from these
patterns are limited to a few models that exhibit weaker sensi-
tivity or delayed threshold effects. For example, llama-3 models
maintain high validity up to a temperature of 1.0, beyond which
validity declines sharply, whereas gemma-3-12b maintains high
validity across the full temperature range. Refusal rates increase
for most models, though the onset and magnitude vary widely.
In contrast, similarity, diversity, and parity remain largely stable
across temperatures at the model level, confirming that the weak
temperature sensitivity of social representativeness observed in the
aggregated analysis is not an artifact of averaging. Overall, these re-
sults indicate that temperature control acts as a coarse intervention:
its aggregate effects are predictable and largely consistent across
models, but model-specific differences in sensitivity and threshold
behavior limit its usefulness as a precise mechanism for steering
performance.
Constrained prompting.Figure C.13 reports model-level perfor-
mance under gender-constrained prompting, complementing aggre-
gated results (Section 4.2). Across models, technical quality metrics
remain largely stable, with deviations limited to a small subset. Incontrast, constrained prompting primarily reshapes social repre-
sentativeness, with effects determined by the constraint direction.
Requests forfemale-onlyrecommendations consistently reduce fac-
tuality, similarity, and parity across all models. The parity decline is
expected given the low base rate of women in APS data (Section B.2);
enforcing single-gender lists therefore violates statistical parity by
construction rather than correcting it. We observe similar trade-
offs for ethnicity-, prominence-, and diversity-constrained prompts
(Figures C.14 to C.16). Ethnicity constraints increase refusal rates
and reduce validity and factuality, with the largest factuality drop
occurring forBlack-onlyprompts, indicating increased hallucina-
tion beyond APS coverage. Prominence constraints raise refusals
and diversity when targetinglowly citedscholars, at the cost of
reduced validity, factuality, and similarity. Aggregated results (Fig-
ures C.17a and C.17b) show that equal-representation constraints
increase diversity only along the targeted dimension, with limited
spillover to others. Citation-based constraints induce milder trade-
offs, while generic diversity prompts fail to reliably improve diver-
sity beyond gender and often reduce factuality. Overall, constrained
prompting does not uniformly improve social representativeness.
Specific constraints enforce the requested composition but intro-
duce predictable trade-offs, whereas broad diversity prompts lack
generalizability across social dimensions.
C.3 Socio-Technical Trade-off
Figure C.18 provides a joint view of technical and social perfor-
mance that complements the metric-by-metric analyses presented
throughout the paper. While individual metrics are necessary to
diagnose specific failure modes, aggregated views are useful when
a deployment scenario prioritizes overall performance rather than
isolated dimensions. In this setting, higher validity, lower duplicates,
higher factuality, and higher parity are unambiguously desirable,
whereas other considerations depend on context and application
requirements. If one were to prioritize both technical and social
objectives simultaneously, the natural choice would be the model
occupying the upper-right region of the socio-technical plane, corre-
sponding to high aggregated technical and social quality. In this ag-
gregation, all metrics are assigned equal weight; alternative weight-
ings could be used to reflect different deployment priorities.
Here, social performance reflects the sum of parity scores across
demographic attributes, including perceived gender, perceived eth-
nicity, publication prominence, and citation prominence, while
technical performance aggregates validity, the complement of du-
plicates (uniqueness), and factuality across criteria (author, field,
epoch, and seniority). Importantly, these aggregates are computed
as per-model means, ensuring comparability across inference-time
configurations with different numbers of requests. As the figure il-
lustrates, model rankings along these axes are not fixed but depend
on inference-time configuration. Temperature variation largely pre-
serves the baseline ordering, indicating that stochasticity alone
does not materially alter the socio-technical frontier. Under these
settings, gemma-3-27b emerges as the strongest joint performer,
despite being a medium-sized, open-weight, non-reasoning model.
In contrast, llama models consistently rank lower on social per-
formance, even though llama-4-mav attains the highest technical

Whose Name Comes Up? Benchmarking and Intervention-Based Auditing of LLM-Based Scholar Recommendation Preprint, arXiv,
scores, highlighting a pronounced trade-off. Within the gemini fam-
ily,gemini-2.5-flash outperforms gemini-2.5-pro on technical
metrics at baseline, while the reverse holds for social performance.
This ordering changes under constrained prompting and retrieval-
augmented generation. Constrained prompting systematically in-
creases social performance but reduces technical quality acrossmodels, reflecting a redistribution rather than a uniform improve-
ment. Both constrained prompting and RAG, however, improve the
relative technical performance of gemini-2.5-pro compared to
gemini-2.5-flash , demonstrating that inference-time interven-
tions can shift the socio-technical frontier and alter which models
are preferred under joint optimization criteria.

Preprint, arXiv, Espín-Noboa and Méndez
deepseek-chat-v3.1
01Refusals
 Validity
Duplicates
 Consistency
 Factuality
 Connectedness
 Similarity
 Diversity
 Parity
deepseek-r1-0528
01
gemma-3-12b
01
gemma-3-27b
01
gemini-2.5-flash
01
gemini-2.5-pro
01
grok-4-fast
01
gpt-oss-20b
01
gpt-oss-120b
01
llama-3.3-8b
01
llama-4-scout
01
llama-4-mav
01
llama-3.1-70b
01
llama-3.3-70b
01
llama-3.1-405b
01
mistral-small-3.2-24b
01
mistral-medium-3
01
qwen3-8b
01
qwen3-14b
01
qwen3-32b
01
qwen3-30b-a3b-2507
01
qwen3-235b-a22b-2507
0.0 0.5 1.0 1.5 2.0
temperature01
0.0 0.5 1.0 1.5 2.0
temperature
0.0 0.5 1.0 1.5 2.0
temperature
0.0 0.5 1.0 1.5 2.0
temperature
0.0 0.5 1.0 1.5 2.0
temperature
0.0 0.5 1.0 1.5 2.0
temperature
0.0 0.5 1.0 1.5 2.0
temperature
0.0 0.5 1.0 1.5 2.0
temperature
0.0 0.5 1.0 1.5 2.0
temperature
Figure C.12: Effect of temperature on performance per model across all tasks. Mean metric values ( ±95% CI) across temperature
settings for each model, computed from the temporal analysis. Across models, higher temperatures systematically increase
refusals with consistent declines in validity, consistency, factuality, and connectedness. In contrast, social representativeness
metrics (similarity, diversity, and parity) remain largely insensitive to temperature. Duplicate rates are low for most models.

Whose Name Comes Up? Benchmarking and Intervention-Based Auditing of LLM-Based Scholar Recommendation Preprint, arXiv,
deepseek-chat-v3.1
deepseek-r1-0528
gemma-3-12b
gemma-3-27b
gemini-2.5-flash
gemini-2.5-pro
grok-4-fast
gpt-oss-20b
gpt-oss-120b
llama-3.3-8b
llama-4-scout
llama-4-mav
llama-3.1-70b
llama-3.3-70b
llama-3.1-405b
mistral-small-3.2-24b
mistral-medium-3
qwen3-8b
qwen3-14b
qwen3-32b
qwen3-30b-a3b-2507
qwen3-235b-a22b-2507Refusals Validity
Duplicates
 Consistency Factuality
 Connectedness Similarity Diversity Parity
0.000.250.500.751.00
0.000.250.500.751.00
0.000.250.500.751.00
0.000.250.500.751.00
0.000.250.500.751.00
0.000.250.500.751.00
0.000.250.500.751.00
0.000.250.500.751.00
0.000.250.500.751.00
0.000.250.500.751.00
0.000.250.500.751.00
0.000.250.500.751.00
0.000.250.500.751.00
0.000.250.500.751.00
0.000.250.500.751.00
0.000.250.500.751.00
0.000.250.500.751.00
0.000.250.500.751.00
0.000.250.500.751.00
0.000.250.500.751.00
0.000.250.500.751.00
0.000.250.500.751.00
0.000.250.500.751.00
0.000.250.500.751.00
0.000.250.500.751.00
0.000.250.500.751.00
0.000.250.500.751.00
0.000.250.500.751.00
0.000.250.500.751.00
0.000.250.500.751.00
0.000.250.500.751.00
0.000.250.500.751.00
0.000.250.500.751.00
0.000.250.500.751.00
0.000.250.500.751.00
0.000.250.500.751.00
0.000.250.500.751.00
0.000.250.500.751.00
0.000.250.500.751.00
0.000.250.500.751.00
0.000.250.500.751.00
0.000.250.500.751.00
0.000.250.500.751.00
0.000.250.500.751.00
0.000.250.500.751.00
0.000.250.500.751.00
0.000.250.500.751.00
0.000.250.500.751.00
0.000.250.500.751.00
0.000.250.500.751.00
0.000.250.500.751.00
0.000.250.500.751.00
0.000.250.500.751.00
0.000.250.500.751.00
0.000.250.500.751.00
0.000.250.500.751.00
0.000.250.500.751.00
0.000.250.500.751.00
0.000.250.500.751.00
0.000.250.500.751.00
0.000.250.500.751.00
0.000.250.500.751.00
0.000.250.500.751.00
0.000.250.500.751.00
0.000.250.500.751.00
0.000.250.500.751.00
0.000.250.500.751.00
0.000.250.500.751.00
0.000.250.500.751.00
0.000.250.500.751.00
0.000.250.500.751.00
0.000.250.500.751.00
0.000.250.500.751.00
0.000.250.500.751.00
0.000.250.500.751.00
0.000.250.500.751.00
0.000.250.500.751.00
0.000.250.500.751.00
0.000.250.500.751.00
0.000.250.500.751.00
0.000.250.500.751.00
0.000.250.500.751.00
0.000.250.500.751.00
0.000.250.500.751.00
0.000.250.500.751.00
0.000.250.500.751.00
0.000.250.500.751.00
0.000.250.500.751.00
0.000.250.500.751.00
0.000.250.500.751.00
0.000.250.500.751.00
0.000.250.500.751.00
0.000.250.500.751.00
0.000.250.500.751.00
0.000.250.500.751.00
0.000.250.500.751.00
0.000.250.500.751.00
0.000.250.500.751.00
0.000.250.500.751.00
0.000.250.500.751.00
0.000.250.500.751.00
0.000.250.500.751.00
0.000.250.500.751.00
0.000.250.500.751.00
0.000.250.500.751.00
0.000.250.500.751.00
0.000.250.500.751.00
0.000.250.500.751.00
0.000.250.500.751.00
0.000.250.500.751.00
0.000.250.500.751.00
0.000.250.500.751.00
0.000.250.500.751.00
0.000.250.500.751.00
0.000.250.500.751.00
0.000.250.500.751.00
0.000.250.500.751.00
0.000.250.500.751.00
0.000.250.500.751.00
0.000.250.500.751.00
0.000.250.500.751.00
0.000.250.500.751.00
0.000.250.500.751.00
0.000.250.500.751.00
0.000.250.500.751.00
0.000.250.500.751.00
0.000.250.500.751.00
0.000.250.500.751.00
0.000.250.500.751.00
0.000.250.500.751.00
0.000.250.500.751.00
0.000.250.500.751.00
0.000.250.500.751.00
0.000.250.500.751.00
0.000.250.500.751.00
0.000.250.500.751.00
0.000.250.500.751.00
0.000.250.500.751.00
0.000.250.500.751.00
0.000.250.500.751.00
0.000.250.500.751.00
0.000.250.500.751.00
0.000.250.500.751.00
0.000.250.500.751.00
0.000.250.500.751.00
0.000.250.500.751.00
0.000.250.500.751.00
0.000.250.500.751.00
0.000.250.500.751.00
0.000.250.500.751.00
0.000.250.500.751.00
0.000.250.500.751.00
0.000.250.500.751.00
0.000.250.500.751.00
0.000.250.500.751.00
0.000.250.500.751.00
0.000.250.500.751.00
0.000.250.500.751.00
0.000.250.500.751.00
0.000.250.500.751.00
0.000.250.500.751.00
0.000.250.500.751.00
0.000.250.500.751.00
0.000.250.500.751.00
0.000.250.500.751.00
0.000.250.500.751.00
0.000.250.500.751.00
0.000.250.500.751.00
0.000.250.500.751.00
0.000.250.500.751.00
0.000.250.500.751.00
0.000.250.500.751.00
0.000.250.500.751.00
0.000.250.500.751.00
0.000.250.500.751.00
0.000.250.500.751.00
0.000.250.500.751.00
0.000.250.500.751.00
0.000.250.500.751.00
0.000.250.500.751.00
0.000.250.500.751.00
0.000.250.500.751.00
0.000.250.500.751.00
0.000.250.500.751.00
0.000.250.500.751.00
0.000.250.500.751.00
0.000.250.500.751.00
0.000.250.500.751.00
0.000.250.500.751.00
0.000.250.500.751.00
B A0.000.250.500.751.00
B A0.000.250.500.751.00
B A0.000.250.500.751.00
B A0.000.250.500.751.00
B A0.000.250.500.751.00
B A0.000.250.500.751.00
B A0.000.250.500.751.00
B A0.000.250.500.751.00
B AEqual Only Female Only Male Only Neutral
Figure C.13: Model-level performance under gender-constrained prompting for top-100 expert recommendations. Mean values
(±95%CI) before (B) and after (A) gender-constrained prompting. Colored points indicate mean performance under different
prompting conditions (equal representation across all genders, female-only, male-only, neutral-names only), with lines showing
changes relative to the (no intervention) baseline prompt.

Preprint, arXiv, Espín-Noboa and Méndez
deepseek-chat-v3.1
deepseek-r1-0528
gemma-3-12b
gemma-3-27b
gemini-2.5-flash
gemini-2.5-pro
grok-4-fast
gpt-oss-20b
gpt-oss-120b
llama-3.3-8b
llama-4-scout
llama-4-mav
llama-3.1-70b
llama-3.3-70b
llama-3.1-405b
mistral-small-3.2-24b
mistral-medium-3
qwen3-8b
qwen3-14b
qwen3-32b
qwen3-30b-a3b-2507
qwen3-235b-a22b-2507Refusals Validity
Duplicates
 Consistency Factuality
 Connectedness Similarity Diversity Parity
0.000.250.500.751.00
0.000.250.500.751.00
0.000.250.500.751.00
0.000.250.500.751.00
0.000.250.500.751.00
0.000.250.500.751.00
0.000.250.500.751.00
0.000.250.500.751.00
0.000.250.500.751.00
0.000.250.500.751.00
0.000.250.500.751.00
0.000.250.500.751.00
0.000.250.500.751.00
0.000.250.500.751.00
0.000.250.500.751.00
0.000.250.500.751.00
0.000.250.500.751.00
0.000.250.500.751.00
0.000.250.500.751.00
0.000.250.500.751.00
0.000.250.500.751.00
0.000.250.500.751.00
0.000.250.500.751.00
0.000.250.500.751.00
0.000.250.500.751.00
0.000.250.500.751.00
0.000.250.500.751.00
0.000.250.500.751.00
0.000.250.500.751.00
0.000.250.500.751.00
0.000.250.500.751.00
0.000.250.500.751.00
0.000.250.500.751.00
0.000.250.500.751.00
0.000.250.500.751.00
0.000.250.500.751.00
0.000.250.500.751.00
0.000.250.500.751.00
0.000.250.500.751.00
0.000.250.500.751.00
0.000.250.500.751.00
0.000.250.500.751.00
0.000.250.500.751.00
0.000.250.500.751.00
0.000.250.500.751.00
0.000.250.500.751.00
0.000.250.500.751.00
0.000.250.500.751.00
0.000.250.500.751.00
0.000.250.500.751.00
0.000.250.500.751.00
0.000.250.500.751.00
0.000.250.500.751.00
0.000.250.500.751.00
0.000.250.500.751.00
0.000.250.500.751.00
0.000.250.500.751.00
0.000.250.500.751.00
0.000.250.500.751.00
0.000.250.500.751.00
0.000.250.500.751.00
0.000.250.500.751.00
0.000.250.500.751.00
0.000.250.500.751.00
0.000.250.500.751.00
0.000.250.500.751.00
0.000.250.500.751.00
0.000.250.500.751.00
0.000.250.500.751.00
0.000.250.500.751.00
0.000.250.500.751.00
0.000.250.500.751.00
0.000.250.500.751.00
0.000.250.500.751.00
0.000.250.500.751.00
0.000.250.500.751.00
0.000.250.500.751.00
0.000.250.500.751.00
0.000.250.500.751.00
0.000.250.500.751.00
0.000.250.500.751.00
0.000.250.500.751.00
0.000.250.500.751.00
0.000.250.500.751.00
0.000.250.500.751.00
0.000.250.500.751.00
0.000.250.500.751.00
0.000.250.500.751.00
0.000.250.500.751.00
0.000.250.500.751.00
0.000.250.500.751.00
0.000.250.500.751.00
0.000.250.500.751.00
0.000.250.500.751.00
0.000.250.500.751.00
0.000.250.500.751.00
0.000.250.500.751.00
0.000.250.500.751.00
0.000.250.500.751.00
0.000.250.500.751.00
0.000.250.500.751.00
0.000.250.500.751.00
0.000.250.500.751.00
0.000.250.500.751.00
0.000.250.500.751.00
0.000.250.500.751.00
0.000.250.500.751.00
0.000.250.500.751.00
0.000.250.500.751.00
0.000.250.500.751.00
0.000.250.500.751.00
0.000.250.500.751.00
0.000.250.500.751.00
0.000.250.500.751.00
0.000.250.500.751.00
0.000.250.500.751.00
0.000.250.500.751.00
0.000.250.500.751.00
0.000.250.500.751.00
0.000.250.500.751.00
0.000.250.500.751.00
0.000.250.500.751.00
0.000.250.500.751.00
0.000.250.500.751.00
0.000.250.500.751.00
0.000.250.500.751.00
0.000.250.500.751.00
0.000.250.500.751.00
0.000.250.500.751.00
0.000.250.500.751.00
0.000.250.500.751.00
0.000.250.500.751.00
0.000.250.500.751.00
0.000.250.500.751.00
0.000.250.500.751.00
0.000.250.500.751.00
0.000.250.500.751.00
0.000.250.500.751.00
0.000.250.500.751.00
0.000.250.500.751.00
0.000.250.500.751.00
0.000.250.500.751.00
0.000.250.500.751.00
0.000.250.500.751.00
0.000.250.500.751.00
0.000.250.500.751.00
0.000.250.500.751.00
0.000.250.500.751.00
0.000.250.500.751.00
0.000.250.500.751.00
0.000.250.500.751.00
0.000.250.500.751.00
0.000.250.500.751.00
0.000.250.500.751.00
0.000.250.500.751.00
0.000.250.500.751.00
0.000.250.500.751.00
0.000.250.500.751.00
0.000.250.500.751.00
0.000.250.500.751.00
0.000.250.500.751.00
0.000.250.500.751.00
0.000.250.500.751.00
0.000.250.500.751.00
0.000.250.500.751.00
0.000.250.500.751.00
0.000.250.500.751.00
0.000.250.500.751.00
0.000.250.500.751.00
0.000.250.500.751.00
0.000.250.500.751.00
0.000.250.500.751.00
0.000.250.500.751.00
0.000.250.500.751.00
0.000.250.500.751.00
0.000.250.500.751.00
0.000.250.500.751.00
0.000.250.500.751.00
0.000.250.500.751.00
0.000.250.500.751.00
0.000.250.500.751.00
0.000.250.500.751.00
0.000.250.500.751.00
0.000.250.500.751.00
0.000.250.500.751.00
0.000.250.500.751.00
0.000.250.500.751.00
0.000.250.500.751.00
0.000.250.500.751.00
0.000.250.500.751.00
B A0.000.250.500.751.00
B A0.000.250.500.751.00
B A0.000.250.500.751.00
B A0.000.250.500.751.00
B A0.000.250.500.751.00
B A0.000.250.500.751.00
B A0.000.250.500.751.00
B A0.000.250.500.751.00
B AEqual Only Asian Only White Only Latino Only Black
Figure C.14: Model-level performance under ethnicity-constrained prompting for top-100 expert recommendations. Mean
values (±95%CI) before (B) and after (A) ethnicity-constrained prompting. Colored points indicate mean performance under
different prompting conditions (equal representation across all ethnicities, Asian-only, White-only, Latino-only, and Black
only), with lines showing changes relative to the (no intervention) baseline prompt.

Whose Name Comes Up? Benchmarking and Intervention-Based Auditing of LLM-Based Scholar Recommendation Preprint, arXiv,
deepseek-chat-v3.1
deepseek-r1-0528
gemma-3-12b
gemma-3-27b
gemini-2.5-flash
gemini-2.5-pro
grok-4-fast
gpt-oss-20b
gpt-oss-120b
llama-3.3-8b
llama-4-scout
llama-4-mav
llama-3.1-70b
llama-3.3-70b
llama-3.1-405b
mistral-small-3.2-24b
mistral-medium-3
qwen3-8b
qwen3-14b
qwen3-32b
qwen3-30b-a3b-2507
qwen3-235b-a22b-2507Refusals Validity
Duplicates
 Consistency Factuality
 Connectedness Similarity Diversity Parity
0.000.250.500.751.00
0.000.250.500.751.00
0.000.250.500.751.00
0.000.250.500.751.00
0.000.250.500.751.00
0.000.250.500.751.00
0.000.250.500.751.00
0.000.250.500.751.00
0.000.250.500.751.00
0.000.250.500.751.00
0.000.250.500.751.00
0.000.250.500.751.00
0.000.250.500.751.00
0.000.250.500.751.00
0.000.250.500.751.00
0.000.250.500.751.00
0.000.250.500.751.00
0.000.250.500.751.00
0.000.250.500.751.00
0.000.250.500.751.00
0.000.250.500.751.00
0.000.250.500.751.00
0.000.250.500.751.00
0.000.250.500.751.00
0.000.250.500.751.00
0.000.250.500.751.00
0.000.250.500.751.00
0.000.250.500.751.00
0.000.250.500.751.00
0.000.250.500.751.00
0.000.250.500.751.00
0.000.250.500.751.00
0.000.250.500.751.00
0.000.250.500.751.00
0.000.250.500.751.00
0.000.250.500.751.00
0.000.250.500.751.00
0.000.250.500.751.00
0.000.250.500.751.00
0.000.250.500.751.00
0.000.250.500.751.00
0.000.250.500.751.00
0.000.250.500.751.00
0.000.250.500.751.00
0.000.250.500.751.00
0.000.250.500.751.00
0.000.250.500.751.00
0.000.250.500.751.00
0.000.250.500.751.00
0.000.250.500.751.00
0.000.250.500.751.00
0.000.250.500.751.00
0.000.250.500.751.00
0.000.250.500.751.00
0.000.250.500.751.00
0.000.250.500.751.00
0.000.250.500.751.00
0.000.250.500.751.00
0.000.250.500.751.00
0.000.250.500.751.00
0.000.250.500.751.00
0.000.250.500.751.00
0.000.250.500.751.00
0.000.250.500.751.00
0.000.250.500.751.00
0.000.250.500.751.00
0.000.250.500.751.00
0.000.250.500.751.00
0.000.250.500.751.00
0.000.250.500.751.00
0.000.250.500.751.00
0.000.250.500.751.00
0.000.250.500.751.00
0.000.250.500.751.00
0.000.250.500.751.00
0.000.250.500.751.00
0.000.250.500.751.00
0.000.250.500.751.00
0.000.250.500.751.00
0.000.250.500.751.00
0.000.250.500.751.00
0.000.250.500.751.00
0.000.250.500.751.00
0.000.250.500.751.00
0.000.250.500.751.00
0.000.250.500.751.00
0.000.250.500.751.00
0.000.250.500.751.00
0.000.250.500.751.00
0.000.250.500.751.00
0.000.250.500.751.00
0.000.250.500.751.00
0.000.250.500.751.00
0.000.250.500.751.00
0.000.250.500.751.00
0.000.250.500.751.00
0.000.250.500.751.00
0.000.250.500.751.00
0.000.250.500.751.00
0.000.250.500.751.00
0.000.250.500.751.00
0.000.250.500.751.00
0.000.250.500.751.00
0.000.250.500.751.00
0.000.250.500.751.00
0.000.250.500.751.00
0.000.250.500.751.00
0.000.250.500.751.00
0.000.250.500.751.00
0.000.250.500.751.00
0.000.250.500.751.00
0.000.250.500.751.00
0.000.250.500.751.00
0.000.250.500.751.00
0.000.250.500.751.00
0.000.250.500.751.00
0.000.250.500.751.00
0.000.250.500.751.00
0.000.250.500.751.00
0.000.250.500.751.00
0.000.250.500.751.00
0.000.250.500.751.00
0.000.250.500.751.00
0.000.250.500.751.00
0.000.250.500.751.00
0.000.250.500.751.00
0.000.250.500.751.00
0.000.250.500.751.00
0.000.250.500.751.00
0.000.250.500.751.00
0.000.250.500.751.00
0.000.250.500.751.00
0.000.250.500.751.00
0.000.250.500.751.00
0.000.250.500.751.00
0.000.250.500.751.00
0.000.250.500.751.00
0.000.250.500.751.00
0.000.250.500.751.00
0.000.250.500.751.00
0.000.250.500.751.00
0.000.250.500.751.00
0.000.250.500.751.00
0.000.250.500.751.00
0.000.250.500.751.00
0.000.250.500.751.00
0.000.250.500.751.00
0.000.250.500.751.00
0.000.250.500.751.00
0.000.250.500.751.00
0.000.250.500.751.00
0.000.250.500.751.00
0.000.250.500.751.00
0.000.250.500.751.00
0.000.250.500.751.00
0.000.250.500.751.00
0.000.250.500.751.00
0.000.250.500.751.00
0.000.250.500.751.00
0.000.250.500.751.00
0.000.250.500.751.00
0.000.250.500.751.00
0.000.250.500.751.00
0.000.250.500.751.00
0.000.250.500.751.00
0.000.250.500.751.00
0.000.250.500.751.00
0.000.250.500.751.00
0.000.250.500.751.00
0.000.250.500.751.00
0.000.250.500.751.00
0.000.250.500.751.00
0.000.250.500.751.00
0.000.250.500.751.00
0.000.250.500.751.00
0.000.250.500.751.00
0.000.250.500.751.00
0.000.250.500.751.00
0.000.250.500.751.00
0.000.250.500.751.00
0.000.250.500.751.00
0.000.250.500.751.00
0.000.250.500.751.00
0.000.250.500.751.00
0.000.250.500.751.00
0.000.250.500.751.00
0.000.250.500.751.00
0.000.250.500.751.00
0.000.250.500.751.00
0.000.250.500.751.00
B A0.000.250.500.751.00
B A0.000.250.500.751.00
B A0.000.250.500.751.00
B A0.000.250.500.751.00
B A0.000.250.500.751.00
B A0.000.250.500.751.00
B A0.000.250.500.751.00
B A0.000.250.500.751.00
B ALowly cited Highly cited
Figure C.15: Model-level performance under citation-constrained prompting for top-100 expert recommendations. Mean values
(±95%CI) before (B) and after (A) citation-constrained prompting. Colored points indicate mean performance under different
prompting conditions (lowly cited-only, highly cited-only), with lines showing changes relative to the (no intervention) baseline
prompt.

Preprint, arXiv, Espín-Noboa and Méndez
deepseek-chat-v3.1
deepseek-r1-0528
gemma-3-12b
gemma-3-27b
gemini-2.5-flash
gemini-2.5-pro
grok-4-fast
gpt-oss-20b
gpt-oss-120b
llama-3.3-8b
llama-4-scout
llama-4-mav
llama-3.1-70b
llama-3.3-70b
llama-3.1-405b
mistral-small-3.2-24b
mistral-medium-3
qwen3-8b
qwen3-14b
qwen3-32b
qwen3-30b-a3b-2507
qwen3-235b-a22b-2507Refusals Validity
Duplicates
 Consistency Factuality
 Connectedness Similarity Diversity Parity
0.000.250.500.751.00
0.000.250.500.751.00
0.000.250.500.751.00
0.000.250.500.751.00
0.000.250.500.751.00
0.000.250.500.751.00
0.000.250.500.751.00
0.000.250.500.751.00
0.000.250.500.751.00
0.000.250.500.751.00
0.000.250.500.751.00
0.000.250.500.751.00
0.000.250.500.751.00
0.000.250.500.751.00
0.000.250.500.751.00
0.000.250.500.751.00
0.000.250.500.751.00
0.000.250.500.751.00
0.000.250.500.751.00
0.000.250.500.751.00
0.000.250.500.751.00
0.000.250.500.751.00
0.000.250.500.751.00
0.000.250.500.751.00
0.000.250.500.751.00
0.000.250.500.751.00
0.000.250.500.751.00
0.000.250.500.751.00
0.000.250.500.751.00
0.000.250.500.751.00
0.000.250.500.751.00
0.000.250.500.751.00
0.000.250.500.751.00
0.000.250.500.751.00
0.000.250.500.751.00
0.000.250.500.751.00
0.000.250.500.751.00
0.000.250.500.751.00
0.000.250.500.751.00
0.000.250.500.751.00
0.000.250.500.751.00
0.000.250.500.751.00
0.000.250.500.751.00
0.000.250.500.751.00
0.000.250.500.751.00
0.000.250.500.751.00
0.000.250.500.751.00
0.000.250.500.751.00
0.000.250.500.751.00
0.000.250.500.751.00
0.000.250.500.751.00
0.000.250.500.751.00
0.000.250.500.751.00
0.000.250.500.751.00
0.000.250.500.751.00
0.000.250.500.751.00
0.000.250.500.751.00
0.000.250.500.751.00
0.000.250.500.751.00
0.000.250.500.751.00
0.000.250.500.751.00
0.000.250.500.751.00
0.000.250.500.751.00
0.000.250.500.751.00
0.000.250.500.751.00
0.000.250.500.751.00
0.000.250.500.751.00
0.000.250.500.751.00
0.000.250.500.751.00
0.000.250.500.751.00
0.000.250.500.751.00
0.000.250.500.751.00
0.000.250.500.751.00
0.000.250.500.751.00
0.000.250.500.751.00
0.000.250.500.751.00
0.000.250.500.751.00
0.000.250.500.751.00
0.000.250.500.751.00
0.000.250.500.751.00
0.000.250.500.751.00
0.000.250.500.751.00
0.000.250.500.751.00
0.000.250.500.751.00
0.000.250.500.751.00
0.000.250.500.751.00
0.000.250.500.751.00
0.000.250.500.751.00
0.000.250.500.751.00
0.000.250.500.751.00
0.000.250.500.751.00
0.000.250.500.751.00
0.000.250.500.751.00
0.000.250.500.751.00
0.000.250.500.751.00
0.000.250.500.751.00
0.000.250.500.751.00
0.000.250.500.751.00
0.000.250.500.751.00
0.000.250.500.751.00
0.000.250.500.751.00
0.000.250.500.751.00
0.000.250.500.751.00
0.000.250.500.751.00
0.000.250.500.751.00
0.000.250.500.751.00
0.000.250.500.751.00
0.000.250.500.751.00
0.000.250.500.751.00
0.000.250.500.751.00
0.000.250.500.751.00
0.000.250.500.751.00
0.000.250.500.751.00
0.000.250.500.751.00
0.000.250.500.751.00
0.000.250.500.751.00
0.000.250.500.751.00
0.000.250.500.751.00
0.000.250.500.751.00
0.000.250.500.751.00
0.000.250.500.751.00
0.000.250.500.751.00
0.000.250.500.751.00
0.000.250.500.751.00
0.000.250.500.751.00
0.000.250.500.751.00
0.000.250.500.751.00
0.000.250.500.751.00
0.000.250.500.751.00
0.000.250.500.751.00
0.000.250.500.751.00
0.000.250.500.751.00
0.000.250.500.751.00
0.000.250.500.751.00
0.000.250.500.751.00
0.000.250.500.751.00
0.000.250.500.751.00
0.000.250.500.751.00
0.000.250.500.751.00
0.000.250.500.751.00
0.000.250.500.751.00
0.000.250.500.751.00
0.000.250.500.751.00
0.000.250.500.751.00
0.000.250.500.751.00
0.000.250.500.751.00
0.000.250.500.751.00
0.000.250.500.751.00
0.000.250.500.751.00
0.000.250.500.751.00
0.000.250.500.751.00
0.000.250.500.751.00
0.000.250.500.751.00
0.000.250.500.751.00
0.000.250.500.751.00
0.000.250.500.751.00
0.000.250.500.751.00
0.000.250.500.751.00
0.000.250.500.751.00
0.000.250.500.751.00
0.000.250.500.751.00
0.000.250.500.751.00
0.000.250.500.751.00
0.000.250.500.751.00
0.000.250.500.751.00
0.000.250.500.751.00
0.000.250.500.751.00
0.000.250.500.751.00
0.000.250.500.751.00
0.000.250.500.751.00
0.000.250.500.751.00
0.000.250.500.751.00
0.000.250.500.751.00
0.000.250.500.751.00
0.000.250.500.751.00
0.000.250.500.751.00
0.000.250.500.751.00
0.000.250.500.751.00
0.000.250.500.751.00
0.000.250.500.751.00
0.000.250.500.751.00
0.000.250.500.751.00
0.000.250.500.751.00
0.000.250.500.751.00
0.000.250.500.751.00
0.000.250.500.751.00
0.000.250.500.751.00
0.000.250.500.751.00
0.000.250.500.751.00
0.000.250.500.751.00
B A0.000.250.500.751.00
B A0.000.250.500.751.00
B A0.000.250.500.751.00
B A0.000.250.500.751.00
B A0.000.250.500.751.00
B A0.000.250.500.751.00
B A0.000.250.500.751.00
B A0.000.250.500.751.00
B ADiverse
Figure C.16: Model-level performance under general diversity-constrained prompting for top-100 expert recommendations.
Mean values (±95%CI) before (B) and after (A) general diversity-constrained prompting. Lines show changes relative to the (no
intervention) baseline prompt.

Whose Name Comes Up? Benchmarking and Intervention-Based Auditing of LLM-Based Scholar Recommendation Preprint, arXiv,
Factuality
Connectedness Similarity Diversitygen.Diversityeth.Diversitypub.Diversitycit.Paritygen.
Parityeth.
Paritypub.
Paritycit.
0.000.250.500.751.00
B A0.000.250.500.751.00
B A0.000.250.500.751.00
B A0.000.250.500.751.00
B A0.000.250.500.751.00
B A0.000.250.500.751.00
B A0.000.250.500.751.00
B A0.000.250.500.751.00
B A0.000.250.500.751.00
B A0.000.250.500.751.00
B A0.000.250.500.751.00
B AEqual Only Female Only Male Only Neutral
(a) Gender-constrained
Factuality
Connectedness Similarity Diversitygen.Diversityeth.Diversitypub.Diversitycit.Paritygen.
Parityeth.
Paritypub.
Paritycit.
0.000.250.500.751.00
B A0.000.250.500.751.00
B A0.000.250.500.751.00
B A0.000.250.500.751.00
B A0.000.250.500.751.00
B A0.000.250.500.751.00
B A0.000.250.500.751.00
B A0.000.250.500.751.00
B A0.000.250.500.751.00
B A0.000.250.500.751.00
B A0.000.250.500.751.00
B AEqual Only Asian Only White Only Latino Only Black
(b) Ethnicity-constrained
Factuality
Connectedness Similarity Diversitygen.Diversityeth.Diversitypub.Diversitycit.Paritygen.
Parityeth.
Paritypub.
Paritycit.
0.000.250.500.751.00
B A0.000.250.500.751.00
B A0.000.250.500.751.00
B A0.000.250.500.751.00
B A0.000.250.500.751.00
B A0.000.250.500.751.00
B A0.000.250.500.751.00
B A0.000.250.500.751.00
B A0.000.250.500.751.00
B A0.000.250.500.751.00
B A0.000.250.500.751.00
B ALowly cited Highly cited
(c) Citations-constrained
Factuality
Connectedness Similarity Diversitygen.Diversityeth.Diversitypub.Diversitycit.Paritygen.
Parityeth.
Paritypub.
Paritycit.
0.000.250.500.751.00
B A0.000.250.500.751.00
B A0.000.250.500.751.00
B A0.000.250.500.751.00
B A0.000.250.500.751.00
B A0.000.250.500.751.00
B A0.000.250.500.751.00
B A0.000.250.500.751.00
B A0.000.250.500.751.00
B A0.000.250.500.751.00
B A0.000.250.500.751.00
B ADiverse
(d) General diversity-constrained
Figure C.17: Trade-offs induced by constrained prompting in social representativeness for top-100 expert recommendations.
Mean values (±95%CI) before (B) and after (A) constrained prompting. (a-c) The lowest factuality scores occur when requesting
recommendations restricted to only female, only Black, or lowly cited scholars. When asked for equal representation by gender
or ethnicity, models do increase diversity with respect to those attributes. (d) In contrast, requesting a generally diverse list of
scientists does not ensure diversity across ethnicity, publications, or citations. It only improves gender diversity at the cost of
lower factuality.

Preprint, arXiv, Espín-Noboa and Méndez
1.0 1.5 2.0 2.5 3.0 3.5 4.01.82.02.22.42.62.8Social: 
a parity(a)
deepseek-chat-v3.1 deepseek-r1-0528
gemini-2.5-flashgemini-2.5-progemma-3-12bgemma-3-27b gpt-oss-120b
gpt-oss-20b grok-4-fast
llama-3.1-405bllama-3.1-70b
llama-3.3-70bllama-3.3-8b
llama-4-mavllama-4-scoutmistral-medium-3mistral-small-3.2-24b
qwen3-14bqwen3-235b-a22b-2507qwen3-30b-a3b-2507
qwen3-32bqwen3-8bdeepseek
gemini
gemma
gpt
grok
llama
mistral
qwen3
(a) Baseline
1.0 1.5 2.0 2.5 3.0 3.5 4.01.82.02.22.42.62.8Social: 
a parity(a)
deepseek-chat-v3.1 deepseek-r1-0528
gemini-2.5-flashgemini-2.5-progemma-3-12bgemma-3-27b gpt-oss-120b
gpt-oss-20b grok-4-fast
llama-3.1-405bllama-3.1-70b
llama-3.3-70bllama-3.3-8b
llama-4-mavllama-4-scoutmistral-medium-3mistral-small-3.2-24b
qwen3-14bqwen3-235b-a22b-2507qwen3-30b-a3b-2507
qwen3-32bqwen3-8b
(b) Temperature variation
1.0 1.5 2.0 2.5 3.0 3.5 4.01.82.02.22.42.62.8Social: 
a parity(a)
deepseek-chat-v3.1deepseek-r1-0528
gemini-2.5-flashgemini-2.5-progemma-3-12b
gemma-3-27b
gpt-oss-120bgpt-oss-20b
grok-4-fast
llama-3.1-405bllama-3.1-70b
llama-3.3-70b
llama-3.3-8bllama-4-mavllama-4-scout
mistral-medium-3
mistral-small-3.2-24bqwen3-14bqwen3-235b-a22b-2507qwen3-30b-a3b-2507
qwen3-32bqwen3-8b
(c) Constrained prompting
1.0 1.5 2.0 2.5 3.0 3.5 4.0
Technical: validity + duplicatesc + 
b factuality(b)
1.82.02.22.42.62.8Social: 
a parity(a)
gemini-2.5-flashgemini-2.5-pro
(d) RAG
Figure C.18: Socio-technical trade-offs across inference-time configurations. Each panel plots models by technical performance
(x-axis) and social performance (y-axis) under (a) baseline, (b) temperature variation, (c) constrained prompting, and (d) retrieval-
augmented generation (RAG). Dashed lines indicate median performance along each axis within a panel. Technical performance
aggregates per-model mean validity, the complement of duplicates, and factuality across criteria (author, field, epoch, and
seniority), while social performance aggregates per-model mean parity across demographic groups; higher values are better
along both axes. Models in the upper-right quadrant therefore jointly optimize technical and social criteria. Under the baseline
and temperature variation settings, gemma-3-27b achieves the strongest overall performance. In contrast, constrained prompting
and RAG reorder the frontier: they improve the technical performance of gemini-2.5-pro relative to gemini-2.5-flash , while
constrained prompting increases social performance at the cost of reduced technical quality overall.