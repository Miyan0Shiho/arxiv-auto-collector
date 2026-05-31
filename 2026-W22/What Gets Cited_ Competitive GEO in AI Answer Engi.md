# What Gets Cited: Competitive GEO in AI Answer Engines

**Authors**: Rahul Vishwakarma, Shushant Kumar, Ratnesh Jamidar

**Published**: 2026-05-25 07:20:08

**PDF URL**: [https://arxiv.org/pdf/2605.25517v1](https://arxiv.org/pdf/2605.25517v1)

## Abstract
AI answer engines generate answers from retrieved pages but cite only a few sources. This makes visibility depend not just on ranking, but on being cited. We study competitive Generative Engine Optimization (GEO): when two retrieved candidates compete, what makes one more likely to be cited first? We build a controlled two-document retrieval-augmented generation (RAG) testbed that injects exactly two candidate sources into the model context and measures which source is referenced by the first citation marker in the output. Across six LLMs we execute 252,000 trials, repeated paired comparisons under one factorial program over 18 content factors. In each trial the two sources differ in exactly one factor; we use brand anonymization and counterbalanced source order to separate content effects from position bias. Mixed-effects models show that topical relevance and list position are the biggest drivers of being cited first. Including explicit price information and a recent timestamp also helps consistently. Completeness and trust cues add smaller gains, while formatting-only edits have little impact. We release a reproducible evaluation protocol and a prioritized GEO checklist for practitioners, and we exercised it in an early internal pilot at Sprinklr, where teams reported positive qualitative feedback on workflow usability.

## Full Text


<!-- PDF content starts -->

What Gets Cited: Competitive GEO in AI Answer Engines
Rahul Vishwakarma
rahul.vishwakarma@sprinklr.com
Sprinklr
Gurugram, IndiaShushant Kumar
shushant.k@sprinklr.com
Sprinklr
Dubai, UAERatnesh Jamidar
ratnesh.jamidar@sprinklr.com
Sprinklr
Gurugram, India
Abstract
AI answer engines generate answers from retrieved pages but cite
only a few sources. This makes visibility depend not just on rank-
ing, but on being cited. We study competitive Generative Engine
Optimization (GEO): when two retrieved candidates compete, what
makes one more likely to be cited first? We build a controlled
two-document retrieval-augmented generation (RAG) testbed that
injectsexactly two candidate sources into the model context and
measures which source is referenced by the first citation marker
in the output. Across six LLMs we execute 252,000trials, repeated
paired comparisons under one factorial program over 18 content
factors. In each trial the two sources differ in exactly one factor;
we use brand anonymization and counterbalanced source order to
separate content effects from position bias. Mixed-effects models
show that topical relevance and list position are the biggest drivers
of being cited first. Including explicit price information and a recent
timestamp also helps consistently. Completeness and trust cues add
smaller gains, while formatting-only edits have little impact. We
release a reproducible evaluation protocol and a prioritized GEO
checklist for practitioners, and we exercised it in an early internal
pilot at Sprinklr, where teams reported positive qualitative feedback
on workflow usability.
CCS Concepts
•Information systems→Language models.
Keywords
Generative Engine Optimization, Language Models, Information
Retrieval, AI Search, Retrieval-Augmented Generation
ACM Reference Format:
Rahul Vishwakarma, Shushant Kumar, and Ratnesh Jamidar. 2026. What
Gets Cited: Competitive GEO in AI Answer Engines. InProceedings of the
49th International ACM SIGIR Conference on Research and Development in
Information Retrieval (SIGIR ’26), July 20–24, 2026, Melbourne, VIC, Aus-
tralia.ACM, Melbourne, Australia, 5 pages. https://doi.org/10.1145/3805712.
3808445
1 Introduction
AI-powered search systems increasingly use retrieval-augmented
generation (RAG) [ 12]: they retrieve candidate documents and use
an LLM to synthesize an answer with selective source citations.
Because only a small subset of retrieved sources are cited, citation
selection becomes a visibility bottleneck. Traditional SEO focused
This work is licensed under a Creative Commons Attribution 4.0 International License.
SIGIR ’26, Melbourne, VIC, Australia
©2026 Copyright held by the owner/author(s).
ACM ISBN 979-8-4007-2599-9/2026/07
https://doi.org/10.1145/3805712.3808445on ranking positions [ 3], but in search with LLMs, content owners
must also optimize for beingcitedin the generated response. We
follow Aggarwal et al. [ 1] in calling thisGenerative Engine Opti-
mization (GEO).
Although GEO is gaining attention, most evaluations do not
match the competitive citation setting practitioners face. Answer
engines cite only a handful of sources, so a page must be not only
"good enough" in isolation, butpreferredover other plausible can-
didates competing for the same citation slot. Aggarwal et al. [ 1]
take an important first step by introducing GEO and proposing
benchmark-style visibility metrics, showing that targeted rewrites
can increase how much a model reflects a given source. However,
these evaluations primarily quantifysingle-source visibilitywithin
a fixed retrieved set and do not directly estimatecitation preference:
which of two similar candidates wins a citation when they com-
pete directly. Beyond evaluation gaps, observational analyses on
live systems further complicate causal inference by mixing content
effects with retrieval rank, presentation order, and other interface
artifacts. Without controlled experiments that isolate individual
content factors, practitioners risk misidentifying citation drivers,
leading to wasted optimization effort while missing gatekeeping
factors that determine whether a source is cited at all.
We studycompetitiveGEO using repeated head-to-head matched-
pair comparisons in a simulated RAG setting where two candidates
are injected as the only retrieved results. To remove two dominant
confounds, we (i) counterbalance source order to separate content
effects from position bias in prompts [ 13,17] and (ii) anonymize
brands and publishers to reduce familiarity effects from pretraining.
From 100 anonymized product review articles across 50 categories,
we construct 1,440 scenarios in which two variants differ in exactly
one of 18 factors spanning content match, completeness, trustwor-
thiness, readability, competitive standing, freshness, and position.
We then execute 252,000trialsacross six LLMs and model citation
preference using logistic mixed-effects models.
We make three contributions:
(1)We propose a simple, controlled way to test GEO under com-
petition: show an LLM two sources at a time, change only one
content factor between them, anonymize brands/publishers,
and swap their order to control for position bias.
(2)We use this protocol at scale to measure how 18 content
factors change which source is cited first across six LLMs,
using mixed-effects models.
(3)We translate the results into a prioritized, evidence-based
checklist that teams can use to improve citation visibility in
AI search.
2 Methodology
Identifying which content attributes drive LLM citations requires
controlled experiments because real-world content varies in multi-
ple attributes simultaneously. We employ head-to-head A/B testsarXiv:2605.25517v1  [cs.AI]  25 May 2026

SIGIR ’26, July 20–24, 2026, Melbourne, VIC, Australia Rahul Vishwakarma, Shushant Kumar, and Ratnesh Jamidar
Table 1: GEO Taxonomy of 18 Tested Factors with Research Backing
Category Factor Description Supporting Evidence
Content Match Topic Mismatch Content discusses unrelated products instead of query-
relevant itemsBM25 [16], TF-IDF, semantic relevance
Content Match Keyword Gap Content lacks key query terms present in competing
sourcesHybrid search in RAG, DPR [11]
Completeness Price Not Mentioned Product price information absent Decision-making research, purchasing fac-
tors
Completeness Missing Specifications Technical specifications omitted Information completeness standards
Completeness No Comparisons No comparison with alternative products AI synthesis capability unclear
Trustworthiness Hedged Language Heavy use of uncertain qualifiers (might, possibly,
could)Linguistic hedging and certainty [8]
Trustworthiness Claims With Evidence Claims lack supporting evidence (tests, certifications) Information credibility research [14]
Trustworthiness Internal Contradictions Content contains conflicting statements Information consistency principles
Trustworthiness Overly Promotional Excessively enthusiastic or sales-focused tone AI safety training observations
Readability Content Structure Dense paragraph format versus organized sections Document understanding research
Readability Scattered Information Related information dispersed throughout text Information grouping principles
Competitive Standing Weaker Value Proposition Less compelling benefits versus competitor Marketing communication research
Competitive Standing Less Comprehensive Shallower analysis versus detailed alternative Information completeness preference
Competitive Standing Weaker Social Proof Fewer or lower ratings/reviews Social proof in online reviews [6]
Competitive Standing Lower List Position Source appears in Position 2 versus Position 1 Lost in the Middle[13], position bias [17]
Freshness Recent vs Old Timestamp Content dated 2026 versus 2019 Temporal IR research [4], recency bias
Freshness No vs Old Timestamp No timestamp versus 2019 date Staleness signals
Freshness Recent vs No Timestamp Recent (2026) timestamp versus no timestamp Competing theories on timestamp absence
in an injected two-source RAG setup: for each query, the LLM re-
ceives two candidate sources (Variant A and Variant B) differing in
exactly one characteristic, and generates an answer with citations.
We recordcitation preference: which source is cited first.
We first construct a controlled dataset that isolates individual
content factors while removing brand familiarity confounds, then
execute experiments at scale across six LLMs and analyze results
with mixed-effects models.
2.1 Dataset Creation
Figure 1: Dataset creation pipeline
The pipeline has four stages (Figure 1): define which factors to
test, curate real-world seed content, anonymize brands to prevent
familiarity from affecting results, and generate matched pairs that
differ in exactly one factor.
18-Factor Testing Framework.To focus on attributes that con-
tent creators can directly control, we synthesized a taxonomy of 18
factors from information quality frameworks [ 19], RAG systems re-
search [ 12], position bias studies [ 13,17], and information retrieval(IR) principles [ 11,15,16]. The factors span Content Match, Com-
pleteness, Trustworthiness, Readability, Competitive Standing, and
Freshness, plus a position-bias manipulation (Lower List Position).
Table 1 presents the complete list.
Seed Corpus Curation.We selected 50 diverse B2C product cat-
egories (consumer technology, home goods, fitness equipment, etc.)
and used GPT-4o with web-search support to curate two representa-
tive product review blog posts per category, 100 in total, capturing
2026 market prices, key technical specifications, and publication
timestamps.
Familiarity Bias Control.LLMs can favor well-known entities
based on pretraining exposure [ 10]. To remove this confound, we
used GPT-4o to replace brand names, product models, and pub-
lisher names with fictional aliases across all 100 blogs, preserving
factual content (prices, specifications, and timestamps). This en-
sures citation outcomes reflect content characteristics, not brand
recognition.
Matched-Pair Design.To isolate each factor’s causal effect,
for each of the 18 factor-wise hypotheses we randomly selected
20 blogs and used GPT-4o to generate 4 scenarios per blog, pro-
ducing 80 scenarios per factor and 1,440 across all factors. Each
scenario pairs two variants that differ in exactly one factor while
matching in all facts, prices, specifications, and length ( ±5%). To
ensure results generalize across query wording, we generated 3
query paraphrases for each of the 1,440 base scenarios, yielding

What Gets Cited: Competitive GEO in AI Answer Engines SIGIR ’26, July 20–24, 2026, Melbourne, VIC, Australia
4,320 scenario-query instances across all 18 tests. All three authors
independently reviewed a stratified sample of 300 scenarios (21%
of 1,440), checking anonymization completeness, single-factor iso-
lation, factual consistency between variants, and length parity. No
brand leakage or unintended factor differences were found.
Variant A (Confident):The CleanBot Aroma Pro X3 delivers excep-
tional cleaning with 30,000 Pa suction and 99.99% germ elimination.
Variant B (Hedged):The CleanBot Aroma Pro X3 might possibly
deliver cleaning with what could be around 30,000 Pa suction...
2.2 Execution
We first describe how trials were run with counterbalanced ordering
across six LLMs, then how we modeled citation outcomes.
Why Two Sources?A pairwise design yields a clean forced
choice that isolates one content factor at a time. It also makes
position control tractable: with two sources there are only two
orders, whereasfully counterbalancing 𝑘sources requires 𝑘!per-
mutations (6 for 𝑘=3, 24 for 𝑘=4), quickly multiplying executions
(API/compute budget) and complicating attribution.
Experimental Execution.Each trial follows the usual turn
order:systemmessage,usermessage, a web_search tool call, the
tool response, then the model answers with citations. We use one
fixed system prompt for every trial:
“You are a helpful assistant. You MUST use the web_search tool to find
information before answering any query. Always cite the exact URLs from
the search results in your response. Do not use any other sources or prior
knowledge. ”
The user message carries only the scenario question. Next we in-
sert the web_search tool callandtool response. In a live prod-
uct the model usually writes the tool call and a backend runs the
search to build the response. Here we supply both: the call carries
a query string chosen when we built the plan for that scenario-
query pair, and we reuse that same call (including the query) on
every repeated run of the pair. The response always lists exactly
two variant sources, each with a title, a URL, and the full text, in
counterbalanced order. We never call a search engine. The model
then answers with citations and sees only this pair. All six APIs
follow the same pattern. Vendors differ only in how they format
tool calls, not in the text the model reads.
Figure 2: Experimental execution scale
Since LLMs can favor earlier-listed sources [ 13,17], we coun-
terbalanced source orders ([A,B] and [B,A]) for 17 content-quality
factors, but not for Lower List Position, where order is the treat-
ment. We ran each combination of scenario, query, and order 5
times for statistical stability. This yielded 2,400 trials per factor for
17 factors, plus 1,200 for the position factor, totaling 42,000 trialsTable 2: Quantitative Effect Sizes: Odds Ratios Across Models
Factor (A vs B) Diverse Providers GPT Models
Gemini
2.5
FlashClaude
3.5
Son-
netKimi
K2
Think-
ingGPT
5
NanoGPT
5
MiniGPT
5.2
On-Topic vs Off-Topic >10k >10k >10k 221†>10k >10k
Query Terms vs Missing 9.41 17.0†16.4 5.99 14.4 40.0
Price vs No Price >10k >10k 36.1 7.82 6.26 30.4
Specs vs No Specs 8.63>10k 238 11.5 15.4 243
With vs No Comparisons 7.45* 4.55* 4.72 2.14 1.78* 1.61
Confident vs Hedged 59975410.6 2.67*5.44 4.75
Evidence vs No Evidence 8.00* >10k 46.3 2.73 5.91 2.09
Consistent vs Contradictory 2.812.81 2.72 2.19 4.091.74
Neutral vs Promotional 1.45* 362* 2.02 1.31 1.81 2.08*
Structured vs Dense 1.68 1.03†0.79 0.90 0.78 1.25
Organized vs Scattered 2.21 3.87 2.49 1.19 1.13 1.57
Strong vs Weak Value Prop 5.22 5.667.24 2.79 2.681.53*
Deep vs Shallow Coverage 1,480†>10k 132 5.33 19.5 3.98
Strong vs Weak Social Proof 6.61*>10k25.4 3.13 4.782.14
Position 1 vs 2 >10k* >10k >10k 2,002 1,795 >10k
Recent vs Old Timestamp >10k>10k 68.7 14.4 1,494> 10k
No vs Old Timestamp 2.32 2.28 1.33* 1.31 1.55 1.48*
Recent vs No Timestamp 1.99 13.03.67 1.153.53 1.99
Bold= statistically significant (𝑝<0.05). OR>1favors variant A.
Convergence warnings:*Degenerate Hessian,†Singular fit.
Effect Size Categories:Very strong (OR>100), Strong (OR 10–100),
Moderate (OR 3–10), Weak (OR 1.5–3), Negligible (OR<1.5)
per model. Six LLMs (Gemini-2.5-Flash, GPT-5-Nano, GPT-5-Mini,
GPT-5.2, Claude-3.5-Sonnet, Kimi-K2-Thinking) produced 252,000
trials in total (Figure 2). We held each model’s output-length limit
and, where the API allows it, sampling temperature fixed; across
trials only the scenario question, the two source texts, their order
in the tool response, and stochastic variation across repeated runs
changed.
Table 2 reports fitted odds ratios by factor and model for the trials
just described (same fixed system prompt as in the framed quote).
Statistical Analysis.From each LLM response, we extracted
which source was cited first, giving a binary outcome 𝑌𝑖=1if
Variant A is cited first in trial 𝑖, and 0 otherwise. To avoid pseu-
doreplication [ 7] from repeated runs nested within position orders
within scenarios, we applied logistic GLMM with nested random
effects [2, 9]:
logit P(𝑌 𝑖=1)=𝛽 0+𝛽 1𝑋𝑖+𝑢𝑠+𝑣𝑠𝑜,(1)
where 𝑋𝑖is a centered position indicator ( +0.5if Variant A is
listed first,−0.5otherwise) and 𝛽1captures presentation-order bias.
We fit one model per content factor; Variant A is the expected
winner (as in Table 2). 𝛽0is the fixed intercept, so exp( ˆ𝛽0)is the
odds ratio in the table after adjusting for order. 𝑢𝑠and 𝑣𝑠𝑜are
random intercepts by scenario and by scenario-order combination,
capturing correlation across repeated runs with the same content

SIGIR ’26, July 20–24, 2026, Melbourne, VIC, Australia Rahul Vishwakarma, Shushant Kumar, and Ratnesh Jamidar
Figure 3: Content optimization workflow
and order. For the Lower List Position test, we exclude the position
covariate.
logit P(𝑌 𝑖=1)=𝛽 0+𝑢𝑠+𝑣𝑠𝑜.(2)
We fitted all models in R using lme4 with maximum likelihood
estimation and the BOBYQA optimizer (𝛼=0.05).
First link, multiple URLs, and exclusions.Across successful
runs for every model, answers contained exactly one distinct URL
about 86.4% of the time, two or more URLs about 10.5%, and no URL
about 3.1%. The GLMMs keep only trials where the first URL in the
answer exactly matches one of the two injected variants. About 3.4%
of trials have no URL or a first URL that matches neither variant; we
drop those from the regression. If several URLs appear, we assign
the outcome from the first URL in reading order.
Effect Size Interpretation.Table 2 presents each of the 18 fac-
tor tests asA vs B, where A is the expected-stronger variant, and
reports Odds Ratios (OR =exp( ˆ𝛽0)) [9,18] across all six models
after controlling for position effects ( 𝛽1). Bold entries are statis-
tically significant ( 𝑝< 0.05), confirming that variant A reliably
wins citation preference. For moderate effects, the OR summarizes
strength (e.g., OR =5≈fivefold higher odds of being cited first).
When preference is nearly deterministic, estimated ORs become
very large under quasi-separation [ 5]; we report them as >10k
where applicable and treat them as indicating adecisivewin for
variant A rather than a finely resolved numeric ratio. Non-bold ORs
lack reliable evidence of an effect.
3 Results and Discussion
We organize findings by strength of cross-model agreement: factors
that consistently drive citations, factors without reliable effects,
and how models differ in sensitivity.
Gatekeepers and Differentiators.Eleven of 18 factors (61%)
reached significance in 4+ models, forming a clear hierarchy. At
the top, four gatekeepers were unanimous across all six models
with large effects (OR >100): Topic Mismatch, Price Not Men-
tioned, Recent vs Old Timestamp, and Lower List Position. Failing
on any one can eliminate citation odds regardless of other content
strengths. Once these prerequisites are met, seven additional factors
provide secondary differentiation (OR 2.1–243): completeness (Miss-
ing Specifications, Less Comprehensive), trust (Hedged Language,
Claims With Evidence, Internal Contradictions), and competitive
positioning (Keyword Gap, No Comparisons).
Factors Without Consistent Effects.Seven factors (39%) had
weak or no effects. Formatting choices (Content Structure, Scat-
tered Information) had no impact, suggesting LLMs parse contentregardless of visual organization. The remaining five (Overly Pro-
motional, Weaker Value Proposition, Weaker Social Proof, No vs
Old Timestamp, Recent vs No Timestamp) reached significance in
only two or three of six models, insufficient for consensus.
Model Behavior Patterns.Models varied substantially in con-
tent sensitivity. Kimi-K2 showed the highest sensitivity (83% of
factors significant), followed by the GPT family, while Claude-3.5
(50%) and Gemini-2.5 (33%) were more selective. Models also dif-
fered in response magnitude: Gemini and Claude exhibited cate-
gorical patterns, with 67–78% of their significant factors producing
extremely large effects (OR >10,000), suggesting binary decision
boundaries. Despite these differences, all six models agreed on the
four gatekeeper factors, pointing to universal citation drivers in-
dependent of model provider or training. Within the GPT family,
all three models showed consistent behavior across different scales,
suggesting citation patterns are shaped by architecture rather than
model size.
4 Practical Implications
Figure 3 operationalizes our findings into a practitioner-facing
workflow. Given an AI answer engine response, the workflow first
checks whether the brand is the top recommendation. If not, it
extracts the sources the AI cited and checks whether the brand’s
content appears among them. If the brand’s content is cited but the
AI still did not recommend it first, the issue is content quality, so
the workflow evaluates it against our 11-factor taxonomy (factors
significant in at least four of six models) organized into four cate-
gories: Trust, Completeness, Relevance, and Context. If the brand
is absent from citations entirely, the bottleneck is retrieval and the
action is to improve SEO. The content evaluation produces a ranked
list of weak factors with effect sizes, guiding what to fix first.
Implementation Strategy.The workflow applies both reac-
tively, when citation visibility drops, and proactively for product
launches or periodic content refreshes. For owned content, sur-
face core topic terms early, add explicit price and key specs, in-
clude comparisons, keep dates current, and replace hedging with
evidence-backed claims. For earned or third-party content, provide
missing data to reviewers, enable independent testing, or support
side-by-side comparison pieces. SEO teams drive retrieval rank,
content/product teams own on-page fixes, and PR/partnerships
own third-party outreach.
Prioritization and Quick Wins.Start with gatekeepers (topic
match, price, recency, position) because failing on any one can
eliminate citation odds. Quick wins are usually editorial: add con-
crete pricing, update timestamps, and close keyword gaps. Once the

What Gets Cited: Competitive GEO in AI Answer Engines SIGIR ’26, July 20–24, 2026, Melbourne, VIC, Australia
baseline is met, invest in differentiators such as spec tables, com-
prehensive comparisons, and competitive positioning. Formatting
changes showed minimal return and can be deprioritized.
Current Content:The Zephyr Pulse 2 fitness tracker features
8-9 day battery life and offline maps with turn-by-turn navigation.
Contact us for pricing details.
Issues Detected:Price Not Mentioned, Missing Specifications,
Weaker Value Proposition
5 Limitations and Future Work
Beyond two injected documents.Production RAG often retrieves
five to ten or more pages, so real citation pools are larger than our
testbed. We nevertheless injectedexactly twocandidates because
the study targetsfactor-levelattribution: we ask which single con-
tent attributes move first-citation preference when everything else
is held fixed. Pairwise variants that differ in only one factor (with
counterbalanced list order) support that identification. Each ad-
ditional retrieved page would introduce simultaneous changes in
overlap, redundancy, and many quality cues, so citation shifts could
no longer be traced to one lever. The estimates are therefore pair-
wise preferences over a controlled slate, not full multi-document
competition. Extending the same factorial discipline to larger candi-
date sets explodes conditions and orderings; we leave that extension
to future work.
Anonymization and brand or domain trust.We anonymized
brands and publishers so citation choices reflect the text, not a
famous name or a well-known site the model may have seen often.
Production systems may still favor trusted domains or strong brands
when they pick sources. A natural complement to this work is a
second factorial layer that holds our pairwise structure but swaps
anonymized entities for matched real brands or domains, quantify-
ing how much residual preference remains after the content signals
we already isolate.
LLM involvement in the corpus.GPT-4o was used to generate
seeds, perform anonymization, and create paired rewrites. The pro-
cess of building the corpus is distinct from how citation preferences
are evaluated. In each run, two injected segments are presented
under a single protocol, and we record which of the two is preferred.
This setup should not be interpreted as a judgment of the model’s
overall writing quality. To further ensure reliability, we manually
reviewed a subset of the data. As detailed in Section 2.1, three au-
thors conducted a stratified review of 300 scenarios, focusing on
isolation of variables, alignment, and length parity, thereby reduc-
ing the likelihood that unintended differences affect the intended
comparison.
6 Conclusion
Our 252,000-trial study shows that LLM citation preference fol-
lows a clear hierarchy: four gatekeeper factors (topic match, price,
recency, position) dominate across all six models, completeness
and trust cues add secondary gains, and formatting has negligi-
ble impact. Where traditional SEO targets ranking, these findings
guide what content signals win citations in AI search. We pack-
aged them into a diagnostic workflow and validated it through
a beta deployment at Sprinklr. Future work will extend to larger
candidate sets, multiple citation slots, and end-to-end retrieval and
recommendation pipelines.Presenter Bio
Shushant Kumaris an Associate Director of Engineering at Sprin-
klr, working on generative AI and agentic systems at scale. He holds
an M.S. in Computer Science from Georgia Institute of Technology.
His interests include retrieval-augmented generation, multimodal
AI, and productionizing large-scale machine learning systems.
References
[1]Pranjal Aggarwal, Vishvak Murahari, Tanmay Rajpurohit, Ashwin Kalyan,
Karthik Narasimhan, and Ameet Deshpande. 2024. Generative Engine Opti-
mization. InProc. 30th ACM SIGKDD Conf. on Knowledge Discovery and Data
Mining (KDD ’24). doi:10.1145/3637528.3671900
[2]Douglas Bates, Martin Mächler, Ben Bolker, and Steve Walker. 2015. Fitting
Linear Mixed-Effects Models Using lme4.Journal of Statistical Software67, 1
(2015). doi:10.18637/jss.v067.i01
[3]Sergey Brin and Lawrence Page. 1998. The Anatomy of a Large-Scale Hyper-
textual Web Search Engine.Computer Networks and ISDN Systems30, 1 (1998).
doi:10.1016/S0169-7552(98)00110-X
[4]Miles Efron and Gene Golovchinsky. 2011. Estimation Methods for Ranking Re-
cent Information. InProc. 34th Int’l ACM SIGIR Conf. on Research and Development
in Information Retrieval (SIGIR ’11). doi:10.1145/2009916.2009984
[5]Kamil Fijorek and Andrzej Sokolowski. 2012. Separation-Resistant and Bias-
Reduced Logistic Regression: STATISTICA Macro.Journal of Statistical Software,
Code Snippets47, 2 (2012). doi:10.18637/jss.v047.c02
[6]Nan Hu, Paul A. Pavlou, and Jennifer Zhang. 2006. Can Online Reviews Reveal a
Product’s True Quality? Empirical Findings and Analytical Modeling of Online
Word-of-Mouth Communication. InProc. 7th ACM Conf. on Electronic Commerce
(EC ’06). doi:10.1145/1134707.1134743
[7]Stuart H. Hurlbert. 1984. Pseudoreplication and the Design of Ecological Field
Experiments.Ecological Monographs54, 2 (1984). doi:10.2307/1942661
[8]Ken Hyland. 1994. Hedging in academic writing and EAP textbooks.English for
Specific Purposes13, 3 (1994). doi:10.1016/0889-4906(94)90004-3
[9]T. Florian Jaeger. 2008. Categorical data analysis: Away from ANOVAs (transfor-
mation or not) and towards logit mixed models.Journal of Memory and Language
59, 4 (2008). doi:10.1016/j.jml.2007.11.007
[10] Mahammed Kamruzzaman, Hieu Minh Nguyen, and Gene Louis Kim. 2024.
“Global is Good, Local is Bad?”: Understanding Brand Bias in LLMs. InProc.
2024 Conf. on Empirical Methods in Natural Language Processing. ACL, Miami, FL,
USA. doi:10.18653/v1/2024.emnlp-main.707
[11] Vladimir Karpukhin, Barlas Oguz, Sewon Min, Patrick Lewis, Ledell Wu, Sergey
Edunov, Danqi Chen, and Wen-tau Yih. 2020. Dense Passage Retrieval for Open-
Domain Question Answering. InProc. 2020 Conf. on Empirical Methods in Natural
Language Processing (EMNLP). doi:10.18653/v1/2020.emnlp-main.550
[12] Patrick Lewis, Ethan Perez, Aleksandra Piktus, Fabio Petroni, Vladimir
Karpukhin, Naman Goyal, Heinrich Küttler, Mike Lewis, Wen-tau Yih,
Tim Rocktäschel, Sebastian Riedel, and Douwe Kiela. 2020. Retrieval-
Augmented Generation for Knowledge-Intensive NLP Tasks. InProc. 34th
Int’l Conf. on Neural Information Processing Systems(Vancouver, BC, Canada)
(NeurIPS ’20). https://proceedings.neurips.cc/paper_files/paper/2020/file/
6b493230205f780e1bc26945df7481e5-Paper.pdf
[13] Nelson F. Liu, Kevin Lin, John Hewitt, Ashwin Paranjape, Michele Bevilacqua,
Fabio Petroni, and Percy Liang. 2024. Lost in the Middle: How Language Models
Use Long Contexts.Trans. ACL12 (2024). doi:10.1162/tacl_a_00638
[14] Miriam J. Metzger and Andrew J. Flanagin. 2013. Credibility and trust of in-
formation in online environments: The use of cognitive heuristics.Journal of
Pragmatics59 (2013). doi:10.1016/j.pragma.2013.07.012
[15] Benjamin Reichman and Larry Heck. 2024. Dense Passage Retrieval: Is it Retriev-
ing?. InFindings of ACL: EMNLP 2024. doi:10.18653/v1/2024.findings-emnlp.791
[16] Stephen Robertson and Hugo Zaragoza. 2009. The Probabilistic Relevance Frame-
work: BM25 and Beyond.Found. Trends Inf. Retr.3, 4 (2009). doi:10.1561/
1500000019
[17] Patrick Schilcher, Dominik Karasin, Michael Schöpf, Haisam Saleh, Antonela
Tommasel, and Markus Schedl. 2025. Characterizing Positional Bias in Large
Language Models: A Multi-Model Evaluation of Prompt Order Effects. InFindings
of ACL: EMNLP 2025. doi:10.18653/v1/2025.findings-emnlp.1124
[18] Magdalena Szumilas. 2010. Explaining Odds Ratios.Journal of the Canadian
Academy of Child and Adolescent Psychiatry19, 3 (2010). https://pmc.ncbi.nlm.
nih.gov/articles/PMC2938757/
[19] Richard Y. Wang and Diane M. Strong. 1996. Beyond Accuracy: What Data
Quality Means to Data Consumers.Journal of Management Information Systems
12, 4 (1996). doi:10.1080/07421222.1996.11518099