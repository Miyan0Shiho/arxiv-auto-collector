# Do Large Language Models Know Conflict? Investigating Parametric vs. Non-Parametric Knowledge of LLMs for Conflict Forecasting

**Authors**: Apollinaire Poli Nemkova, Sarath Chandra Lingareddy, Sagnik Ray Choudhury, Mark V. Albert

**Published**: 2025-05-14 23:24:22

**PDF URL**: [http://arxiv.org/pdf/2505.09852v1](http://arxiv.org/pdf/2505.09852v1)

## Abstract
Large Language Models (LLMs) have shown impressive performance across natural
language tasks, but their ability to forecast violent conflict remains
underexplored. We investigate whether LLMs possess meaningful parametric
knowledge-encoded in their pretrained weights-to predict conflict escalation
and fatalities without external data. This is critical for early warning
systems, humanitarian planning, and policy-making. We compare this parametric
knowledge with non-parametric capabilities, where LLMs access structured and
unstructured context from conflict datasets (e.g., ACLED, GDELT) and recent
news reports via Retrieval-Augmented Generation (RAG). Incorporating external
information could enhance model performance by providing up-to-date context
otherwise missing from pretrained weights. Our two-part evaluation framework
spans 2020-2024 across conflict-prone regions in the Horn of Africa and the
Middle East. In the parametric setting, LLMs predict conflict trends and
fatalities relying only on pretrained knowledge. In the non-parametric setting,
models receive summaries of recent conflict events, indicators, and
geopolitical developments. We compare predicted conflict trend labels (e.g.,
Escalate, Stable Conflict, De-escalate, Peace) and fatalities against
historical data. Our findings highlight the strengths and limitations of LLMs
for conflict forecasting and the benefits of augmenting them with structured
external knowledge.

## Full Text


<!-- PDF content starts -->

arXiv:2505.09852v1  [cs.CL]  14 May 2025Do Large Language Models Know Conflict? Investigating Parametric vs.
Non-Parametric Knowledge of LLMs for Conflict Forecasting
Apollinaire Poli Nemkova1, Sarath Chandra Lingareddy1, Sagnik Ray Choudhury1, Mark V . Albert1
1University of North Texas, USA
Correspondence: poli.nemkova@unt.edu, sarathchandralingareddy@my.unt.edu, sagnik.raychoudhury@unt.edu, mark.albert@unt.edu
Abstract
Large Language Models (LLMs) have shown
impressive performance across natural lan-
guage tasks, but their ability to forecast violent
conflict remains underexplored. We investi-
gate whether LLMs possess meaningful para-
metric knowledge—encoded in their pretrained
weights—to predict conflict escalation and fa-
talities without external data. This is critical for
early warning systems, humanitarian planning,
and policy-making. We compare this paramet-
ric knowledge with non-parametric capabili-
ties, where LLMs access structured and un-
structured context from conflict datasets (e.g.,
ACLED, GDELT) and recent news reports
via Retrieval-Augmented Generation (RAG).
Incorporating external information could en-
hance model performance by providing up-to-
date context otherwise missing from pretrained
weights. Our two-part evaluation framework
spans 2020–2024 across conflict-prone regions
in the Horn of Africa and the Middle East. In
the parametric setting, LLMs predict conflict
trends and fatalities relying only on pretrained
knowledge. In the non-parametric setting, mod-
els receive summaries of recent conflict events,
indicators, and geopolitical developments. We
compare predicted conflict trend labels (e.g.,
Escalate, Stable Conflict, De-escalate, Peace)
and fatalities against historical data. Our find-
ings highlight the strengths and limitations of
LLMs for conflict forecasting and the benefits
of augmenting them with structured external
knowledge.
1 Introduction
Forecasting violent conflict is a critical yet persis-
tently challenging task. Humanitarian organiza-
tions, early warning systems, and policy actors rely
on timely conflict predictions to guide resource al-
location, diplomatic engagement, and civilian pro-
tection strategies. Traditionally, these forecasts are
derived from structured statistical models, domain
expert input, and region-specific knowledge, oftenleveraging structured event data such as ACLED or
GDELT. However, these approaches require man-
ual feature engineering, are brittle to new contexts,
and may lack the capacity to generalize across di-
verse geopolitical settings.
With the rapid progress of large language mod-
els (LLMs), a new question emerges: can LLMs
forecast conflict directly from their parametric
knowledge—the information implicitly encoded in
their pretrained weights? Given that many LLMs
are trained on massive corpora of news articles,
Wikipedia, and geopolitical documents, it is plau-
sible that they have internalized latent patterns re-
lated to conflict escalation, resolution, and regional
dynamics. The question of event forecasting using
LLMs has been approached from a similar perspec-
tive by several authors (Kauf et al., 2022; Ye et al.,
2024).
In this work, we explore the feasibility of using
LLMs to forecast conflict dynamics across two
experimental settings:
- Parametric forecasting, where LLMs generate
predictions using only their internal knowledge,
without any access to recent or region-specific data.
- Retrieval-Augmented Generation (RAG),
where models are supplied with structured indi-
cators and a news-derived summary covering the
previous three months, and are asked to forecast
the following month.
We frame conflict forecasting as a dual task:
(1) classifying the expected direction of vio-
lence—escalation, de-escalation, stable conflict,
or peace; and (2) predicting the estimated number
of fatalities in the forecast window.
Our experiments span multiple conflict-prone re-
gions—including the Horn of Africa and the Mid-
dle East —over a five-year period (2020–2024).
We evaluate several LLMs (GPT-41, LLaMA-22)
1https://openai.com/index/hello-gpt-4o/
2https://huggingface.co/meta-llama/Llama-2-7b

using both zero-shot and RAG-based approaches.
Results show that zero-shot LLMs capture broad
conflict patterns (especially peaceful or stable dy-
namics), but struggle with finer-grained trends
and numeric estimation. RAG-based augmenta-
tion, particularly in the monthly setting, improves
macro-level metrics and fatalities regression.
Interestingly, we find that while GPT-4 generally
outperforms LLaMA-2 across most metrics, the
open-source model shows promising capabilities
in specific contexts, particularly in categorical pre-
dictions for certain regions. This suggests potential
for open-source LLMs in humanitarian forecasting
applications, especially in resource-constrained en-
vironments where proprietary model access may
be limited. However, significant performance gaps
remain that will require further research to address.
Our contributions are threefold: (1) We develop
a structured framework for evaluating LLMs in
conflict forecasting tasks. (2) We compare para-
metric and RAG-based capabilities across multiple
models and regional contexts. (3) We provide qual-
itative and quantitative insights into the limitations
and strengths of current LLMs in high-stakes fore-
casting tasks, offering guidance for future work in
AI-assisted early warning systems.
2 Literature Review
Forecasting political violence and armed conflict
has long been a priority for scholars and prac-
titioners of early warning systems. Traditional
approaches often rely on statistical models us-
ing structured conflict data, such as ACLED3or
UCDP-GED4, combined with socio-political and
economic indicators. Recent advances in machine
learning have enriched this landscape by enabling
models to integrate large-scale unstructured data.
For instance, the ViEWS project (Hegre et al.,
2019) demonstrated that political violence could
be forecasted with subnational granularity using
an ensemble of ML classifiers trained on conflict
and governance data. Similarly, Attina et al. intro-
duced the Dynamic Elastic Net (DynENet), which
leverages over 700 variables—including event data
from GDELT5and ACLED—to capture changes
in conflict fatalities across countries. Other authors
would utilize protest data for the task of conflict
prediction (Rød et al., 2023) or utilize transformer
3https://acleddata.com/
4https://ucdp.uu.se/downloads/
5https://www.gdeltproject.org/architecture for human rights violation detection
(Nemkova et al., 2023) as a part of conflict moni-
toring.
Text-based modeling has emerged as a partic-
ularly powerful tool in this space. Mueller and
Rauh (2018) pioneered a method that forecasts con-
flict onset by extracting topic distributions from
millions of newspaper articles. These topic fea-
tures proved predictive of violent escalation even
in countries with no prior conflict, offering early
warning capability unattainable through histori-
cal trends alone. Their openly released dataset
of monthly country- and district-level forecasts
(Mueller et al., 2024) has become a valuable bench-
mark for evaluating risk-aware conflict interven-
tions.
While traditional ML approaches require task-
specific engineering, Large Language Models
(LLMs) offer a flexible alternative by generaliz-
ing across domains. Recent work has shown that
LLMs, such as GPT-3 and LLaMA-2, are capable
of performing temporal reasoning and event fore-
casting in zero-shot settings (Chang et al., 2024;
Yu et al.; Wang et al., 2024). The RAF framework
(Tire et al., 2024) shows that foundation models
can generate accurate continuations of time series
without dedicated fine-tuning. However, LLMs
trained purely on static corpora tend to struggle
with up-to-date, domain-specific predictions. As
such, their utility in high-stakes tasks like conflict
forecasting remains underexplored.
Retrieval-Augmented Generation (RAG) offers
one solution to the limitations of static parametric
models by injecting fresh, context-rich informa-
tion into the generation pipeline. Yang et al. in-
troduced TimeRAG, which retrieves relevant time
series segments to boost LLM forecasting perfor-
mance. Similarly, TS-RAG (Ning et al., 2025)
combines time series memory with large-scale re-
trieval to improve accuracy in zero-shot prediction
tasks. These RAG-based frameworks are espe-
cially valuable in settings where recent develop-
ments can rapidly shift forecast dynamics. In the
conflict domain specifically, Wood and Joshi pro-
posed Conflict-RAG, a multilingual RAG pipeline
that enhances LLMs’ ability to understand cur-
rent conflict events by retrieving regional news
sources, including Arabic-language articles. Their
approach improves situational awareness in rapidly
evolving crises and demonstrates the promise of
RAG-augmented LLMs for early warning systems.
Additionally, the phenomenon of knowledge

conflict in LLMs (Xu et al., 2024) — where para-
metric knowledge encoded during pretraining con-
tradicts non-parametric information retrieved at in-
ference time—raises important questions for con-
flict forecasting. This makes it especially valu-
able to experimentally compare LLM performance
in parametric-only versus retrieval-augmented set-
tings, to evaluate not only overall accuracy but also
the models’ ability to reconcile conflicting informa-
tion (Mallen et al., 2022; Wang et al., 2023; Jiang,
2024) . Recent work by Wu et al. demonstrates that
some LLMs exhibit a stronger bias toward their
parametric knowledge, even when provided with
relevant external context via RAG, highlighting the
need for model-specific evaluation in high-stakes
domains.
Together, these lines of research reveal four im-
portant trends: (1) media-derived features can pro-
vide strong signals for early conflict prediction,
particularly for first-onset cases; (2) LLMs can
offer flexible, language-aware forecasting capabil-
ities, but require external knowledge to achieve
reliability; (3) RAG is emerging as a bridge be-
tween deep language models and dynamic real-
world data, with growing relevance to humanitar-
ian and geopolitical forecasting tasks; and (4) the
strength of context knowledge is limited when it
conflicts with parametric contextual evidence.
While prior work has explored conflict forecast-
ing using structured models, media-based signals,
and more recently, large language models, limited
attention has been given to systematically compar-
ing parametric and non-parametric LLM reasoning
for this task (Zhang et al., 2024b,a). In particular,
the extent to which LLMs encode useful forecast-
ing priors in their pretrained weights versus their
ability to synthesize external geopolitical context
remains underexplored. Our work addresses this
gap by evaluating LLMs across three settings: (1)
zero-shot forecasting using only parametric knowl-
edge; (2) retrieval-augmented forecasting based
on recent news summaries and structured indi-
cators over a 3-month window. We benchmark
multiple LLMs (GPT-4, LLaMA-2) across regions
in the Horn of Africa and the Middle East from
2020–2024, analyzing both trend classification and
fatality prediction. This study provides a practical
evaluation framework and new empirical insights
into how well LLMs can support real-world early
warning efforts—either on their own or when aug-
mented with timely conflict data.3 Method
We conduct a structured set of experiments to eval-
uate large language models (LLMs) in forecasting
violent conflict. Specifically, we assess both para-
metric reasoning (pretrained knowledge alone) and
non-parametric reasoning (via retrieval-augmented
generation, or RAG). All experiments are con-
ducted using two LLMs: GPT-4 (via OpenAI API
6) and LLaMA-2-13B-chat (via Hugging Face
Transformers7).
3.1 Data Sources
Our dataset integrates both structured conflict data
and open-source news articles. We use the Global
Database of Events, Language, and Tone (GDELT)
(Leetaru and Schrodt, 2013) as the primary source
of unstructured information. GDELT provides
metadata on global news coverage, including arti-
cle URLs, actor country codes, event tone scores,
and Goldstein Scale values (a proxy for coopera-
tion–conflict dynamics). We queried GDELT for
multiple regions—including the Horn of Africa
and the Middle East —between 2020 and 2024, and
scraped article content using the provided URLs.
Structured conflict outcome data comes from the
Armed Conflict Location and Event Data Project
(ACLED), which records real-world conflict inci-
dents, including the number of fatalities. Fatality
data is used to construct ground truth for both clas-
sification and regression tasks. In Experiment 2,
we also use average article tone and average Gold-
stein scores (McClelland, 1984) from GDELT as
numerical features to contextualize retrieved sum-
maries.
3.2 Experiment 1: Parametric Forecasting
In the first setting, we test whether LLMs can fore-
cast conflict using only their internal parametric
knowledge. The model is prompted to generate a
one-month forecast for a given country, based on
recent conflict and social dynamics inferred from
its pretrained corpus. It must classify the conflict
trend using one of four predefined labels: Escalate,
De-escalate, Peace/No Conflict, Stable Conflict.
(We follow the labeling system as temporal states
utilized in (Croicu and von der Maase, 2025)).
Additionally, the model is asked to estimate the
expected number of fatalities during that period,
6https://platform.openai.com/docs/overview
7https://huggingface.co/docs/transformers/en/index

Table 1: GPT-4 Evaluation Metrics – Experiment 1 (Parametric) vs. Experiment 2 (RAG)
Experiment Metric Ethiopia Sudan Somalia Israel Iran
Exp 1: Class (Categorical)Accuracy 0.2712 0.7458 0.1695 0.4098 0.4667
Precision (macro) 0.2496 0.1864 0.3092 0.1096 0.1750
Recall (macro) 0.4690 0.2500 0.3118 0.2232 0.1556
F1 (macro) 0.2091 0.2136 0.1171 0.1471 0.1647
F1 (weighted) 0.2728 0.6372 0.1682 0.2700 0.4941
Exp 2: Class (Categorical)Accuracy 0.2712 0.7288 0.1356 0.3770 0.3385
Precision (macro) 0.2496 0.2995 0.2112 0.2738 0.2741
Recall (macro) 0.4690 0.3773 0.1505 0.3736 0.2257
F1 (macro) 0.2091 0.3333 0.0964 0.2710 0.1980
F1 (weighted) 0.2728 0.6638 0.1693 0.3781 0.4285
Exp 1: Class (From Fatalities)Accuracy 0.7288 0.7458 0.8305 0.4590 0.7500
Precision (macro) 0.1822 0.1864 0.2768 0.1148 0.1875
Recall (macro) 0.2500 0.2500 0.3333 0.2500 0.2500
F1 (macro) 0.2108 0.2136 0.3025 0.1573 0.2143
F1 (weighted) 0.6145 0.6372 0.7536 0.2888 0.6429
Exp 2: Class (From Fatalities)Accuracy 0.7288 0.7288 0.8305 0.4590 0.5231
Precision (macro) 0.1822 0.2995 0.2188 0.2524 0.2470
Recall (macro) 0.2500 0.3773 0.2500 0.4643 0.4222
F1 (macro) 0.2108 0.3333 0.2333 0.3252 0.2782
F1 (weighted) 0.6145 0.6638 0.7751 0.3440 0.5118
Exp 1: MAE Fatalities MAE 446.59 512.78 326.41 218.48 176.25
Exp 2: MAE Fatalities MAE 446.59 517.56 362.51 93.54 96.80
Exp 1: Binned RegressionAccuracy 0.2542 0.2203 0.3220 0.0000 0.1833
Precision (macro) 0.1918 0.1214 0.1671 0.0000 0.0458
Precision (weighted) 0.2081 0.1235 0.2062 0.0000 0.0336
Recall (macro) 0.2417 0.2167 0.2929 0.0000 0.2500
F1 (weighted) 0.2225 0.1464 0.2203 0.0000 0.0568
Exp 2: Binned RegressionAccuracy 0.2542 0.3390 0.2373 0.3443 0.3385
Precision (macro) 0.1918 0.4404 0.2542 0.3166 0.3036
Precision (weighted) 0.2081 0.4425 0.1895 0.5918 0.4033
Recall (macro) 0.2417 0.3429 0.2577 0.2072 0.3221
F1 (weighted) 0.2225 0.3483 0.1671 0.4280 0.3337
expressed either as a numerical range or specific
count.
3.2.1 Experiment 2: Retrieval-Augmented
Forecasting (RAG)
In the second setting, we assess whether access
to structured and unstructured context improves
forecasting accuracy. Using a RAG pipeline, the
LLM is supplied with retrieved summaries and
metadata covering the previous three months and
is asked to forecast the next month using the same
label set and fatality estimate. The retrieval step
is performed using the FAISS library (Facebook
AI Similarity Search)8, while summarization is
8https://ai.meta.com/tools/faiss/conducted using GPT-3.5. The provided context
includes:
•A summary of the most relevant news ex-
cerpts (retrieved and summarized via seman-
tic search);
•The average tone of retrieved articles (from
GDELT);
•The average Goldstein Scale score (from
GDELT);
•Weekly fatality counts over the past 12 weeks
(from ACLED).

3.3 Evaluation Framework
We evaluate model outputs across three prediction
types:
1. Categorical label prediction (e.g., Escalate)
2. Fatality range prediction (binned using quan-
tiles)
3.Label derived from predicted fatalities,
matched to true class thresholds (e.g., 400
fatalities corresponds to Stable Conflict)
Metrics include accuracy, precision, recall, and
F1-score, reported in micro, macro, and weighted
forms for classification tasks. For regression, we
report Mean Absolute Error (MAE).
3.4 Implementation Details
All experiments are run on Google Colab Pro+9
using NVIDIA A100 GPUs. GPT-4 queries are
queried via OpenAI’s GPT-4 API endpoint with
temperature set to 0.2. The total OpenAI API cost
for this study was $150. Prompt templates and all
source code is available on the author’s GitHub
repository10.
4 Results
The results of the experiments with GPT model
can be seen in Table 1 and in Table 2 for Llama.
This study compares two forecasting paradigms
in LLMs: parametric reasoning, where models
rely solely on internal knowledge (Experiment 1),
and non-parametric reasoning, where they are aug-
mented with retrieved data via RAG (Experiment
2). The results highlight key differences in model
capability and integration.
Parametric Forecasting: Internal Knowledge
Boundaries In the parametric setting, GPT-4 outper-
formed LLaMA across all tasks. GPT-4’s accuracy
was particularly high for class-from-fatalities (e.g.,
0.83 for Somalia), suggesting it encodes useful gen-
eral patterns from pretraining. However, categor-
ical class prediction remained weak (macro-F1 <
0.22 across countries), revealing limits in nuanced
multi-class reasoning based on internal knowledge
alone. LLaMA’s parametric performance was con-
sistently low, underscoring its limited capability
without external context.
9https://colab.research.google.com/
10https://github.com/anonymous-authorNon-Parametric Forecasting: Benefits of RAG
Adding retrieved context via RAG improved GPT-
4’s performance across most tasks and coun-
tries. Notable gains were observed in class-from-
fatalities (e.g., F1-macro in Israel: 0.16 →0.33)
and binned regression, with reduced MAE in coun-
tries like Israel and Iran. These results show GPT-
4’s strong ability to integrate structured external
inputs.
LLaMA, however, showed minimal or incon-
sistent gains from RAG. In some cases (e.g.,
Ethiopia), performance worsened or outputs
failed entirely. This suggests that effective non-
parametric forecasting requires not just retrieval,
but also sufficient model capacity to interpret and
incorporate the new information.
Key Takeaways Parametric forecasting is limited
in precision, especially for fine-grained classifica-
tion.
Non-parametric augmentation improves perfor-
mance—but only when the model can meaning-
fully integrate retrieved context.
GPT-4 benefits consistently from non-
parametric inputs; LLaMA does not.
5 Discussion
Our results shed light on the contrasting capabili-
ties of large language models (LLMs) when relying
solely on their parametric knowledge (Experiment
1) versus when enhanced with contextual retrieval
via a RAG pipeline (Experiment 2). The findings
suggest a persistent yet uneven performance gap
between these two paradigms, with the degree of
improvement varying by model, task formulation,
and region.
Parametric vs. Contextual Knowledge Over-
all, the integration of non-parametric context via
RAG yielded modest but consistent improvements
in several tasks, particularly for GPT-4. When com-
paring classification performance using explicit
labels (Experiment 1 – Class Categorical), RAG
provided marginal gains or maintained compara-
ble performance for GPT-4. For example, macro
F1 scores in Ethiopia and Israel improved (from
0.2091 to 0.2710 and from 0.1471 to 0.2710, re-
spectively), while Sudan and Iran also showed
slight improvements. However, in Somalia, perfor-
mance decreased, indicating that contextual infor-
mation is not universally beneficial, possibly due
to noisy or sparse retrieved content.
The gains were more pronounced in the binned

Table 2: LLama Evaluation Metrics – Experiment 1 (Parametric) vs. Experiment 2 (RAG)
Experiment Metric Ethiopia Sudan Somalia Israel Iran
Exp 1: Class (Categorical)Accuracy 0.0714 0.3898 0.0536 0.3770 0.0536
Precision (macro) 0.0374 0.1878 0.0144 0.2738 0.0153
Recall (macro) 0.3333 0.1667 0.2500 0.3736 0.2500
F1 (macro) 0.0670 0.1633 0.0273 0.2710 0.0288
F1 (weighted) 0.0144 0.4443 0.0058 0.3781 0.0062
Exp 2: Class (Categorical)Accuracy 0.0000 0.2542 0.4107 0.5439 0.0526
Precision (macro) 0.2083 0.2180 0.2054 0.3576 0.0150
Recall (macro) 0.2229 0.1655 0.1173 0.4018 0.2500
F1 (macro) 0.1933 0.1476 0.1494 0.3411 0.0283
F1 (weighted) 0.4732 0.3328 0.5227 0.5236 0.0060
Exp 1: Class (From Fatalities)Accuracy 0.4821 0.7288 0.8393 0.4590 0.7321
Precision (macro) 0.2148 0.1853 0.2217 0.2524 0.1830
Recall (macro) 0.1869 0.2443 0.2398 0.4643 0.2500
F1 (macro) 0.1949 0.2108 0.2304 0.3252 0.2113
F1 (weighted) 0.5373 0.6288 0.8064 0.3440 0.6189
Exp 2: Class (From Fatalities)Accuracy 0.7288 0.6949 0.8750 0.4912 0.7368
Precision (macro) 0.3114 0.2708 0.2917 0.1228 0.1842
Recall (macro) 0.4050 0.3216 0.3333 0.2500 0.2500
F1 (macro) 0.3520 0.2940 0.3111 0.1647 0.2121
F1 (weighted) 0.6389 0.6367 0.8167 0.3236 0.6252
Exp 1: MAE Fatalities MAE 624.66 575.25 344.91 93.54 370.80
Exp 2: MAE Fatalities MAE 572.49 649.92 369.63 229.23 376.84
Exp 1: Binned RegressionAccuracy 0.2679 0.2203 0.2500 0.3443 0.1786
Precision (macro) 0.1494 0.1533 0.3671 0.3166 0.0446
Precision (weighted) 0.1434 0.2302 0.3532 0.5918 0.0319
Recall (macro) 0.2692 0.1892 0.3036 0.2072 0.2500
F1 (weighted) 0.1335 0.1919 0.1744 0.4280 0.0541
Exp 2: Binned RegressionAccuracy 0.2203 0.2542 0.3214 0.0000 0.1754
Precision (macro) 0.1271 0.1834 0.4450 0.0000 0.0439
Precision (weighted) 0.1547 0.2417 0.5075 0.0000 0.0308
Recall (macro) 0.2042 0.2237 0.3155 0.0000 0.2500
F1 (weighted) 0.1485 0.2191 0.2302 0.0000 0.0524
regression task, where RAG substantially im-
proved macro precision and F1 for GPT-4 in re-
gions like Sudan and Israel (e.g., F1 from 0.1464
to 0.3483 and from 0.0000 to 0.4280, respectively),
highlighting the value of article-informed context
in more granular conflict prediction. Interestingly,
RAG did not consistently reduce MAE in fatalities
prediction, suggesting that while context can im-
prove classification of conflict intensity, it may not
translate directly into better numeric estimation.
LLaMA, in contrast, exhibited much lower base-
line performance across all tasks and regions, un-
derscoring its weaker parametric understanding.
While RAG improved classification and regression
metrics in select cases—such as macro F1 in Soma-lia and Israel (e.g., F1 macro from 0.0273 to 0.1494
and from 0.2710 to 0.3411, respectively)—the
overall performance remained limited. This con-
trast reaffirms the superior zero-shot reasoning ca-
pabilities of GPT-4 and suggests that LLaMA bene-
fits from retrieval mostly in cases where parametric
knowledge is clearly insufficient.
Model-Specific Trends GPT-4 consistently out-
performed LLaMA across both experiments and all
task formulations. This was especially evident in
the categorical classification task, where LLaMA
struggled to achieve even modest accuracy scores
in regions like Ethiopia and Iran (e.g., 0.0000 and
0.0526 in Exp 2), while GPT-4 remained above
0.25 in all regions. The disparity was also evident

in fatalities-derived classes, where GPT-4 main-
tained strong accuracy (>0.72 in three regions)
even in the parametric setting, showing its abil-
ity to generalize conflict trends from internalized
patterns alone.
Another key difference lies in how the models
handle retrieved context. GPT-4 appears to lever-
age it more reliably, suggesting better instruction-
following, contextual grounding, and inference ca-
pabilities. LLaMA, while modestly improved with
RAG, still struggled with precision and recall, and
even regressed in some cases (e.g., binned regres-
sion in Israel).
Regional Variation vs. Generalization Re-
gional differences were pronounced. GPT-4 per-
formed best in Sudan, Iran, and Ethiopia, showing
consistent macro F1 improvements across experi-
ments. In contrast, Somalia and Israel were more
volatile, possibly due to less consistent article qual-
ity, varying event dynamics, or weaker patterns in
historical data. This highlights a limitation of both
parametric and RAG-enhanced models: context
quality and topical consistency are critical.
Interestingly, the performance gap between Ex-
periment 1 and Experiment 2 was not uniform
across countries or metrics, indicating that gen-
eralization is region-sensitive. For example, in
fatalities-based classification, GPT-4 and LLaMA
both saw improvements in F1 macro in Sudan and
Somalia with RAG, while accuracy and MAE re-
mained stable. This suggests that regional context
quality, rather than a global trend, determines the
effectiveness of RAG.
6 Conclusion
This study explores the capacity of large language
models (LLMs) to forecast conflict trends and fa-
talities across multiple regions and timeframes. We
compare parametric (zero-shot) forecasting with
retrieval-augmented generation (RAG), evaluating
two prominent models: GPT-4 and LLaMA-2. Our
experiments demonstrate that while both models
exhibit some ability to capture broad conflict dy-
namics, performance improves significantly when
external context—structured and unstructured—is
incorporated through RAG. This is especially evi-
dent in forecasting minority classes such as escala-
tion and de-escalation.
Categorical label prediction remains challeng-
ing, however, and numeric fatality estimates are
often noisy without careful prompt design andpostprocessing. Notably, open-source models
like LLaMA-2, when paired with retrieval, can
perform competitively with GPT-4 in select con-
texts—offering promise for use in low-resource or
access-constrained environments. Our short-term
forecasting setup further suggests that LLMs can
be sensitive to recent trends when grounded in lo-
calized data.
Overall, this work highlights both the promise
and current limitations of LLMs for high-stakes
humanitarian forecasting. Future research should
explore multilingual retrieval, fine-tuning with
domain-specific signals, and human-in-the-loop
pipelines to increase reliability and applicability in
real-world early warning systems.
Limitations
This study presents a first step toward evaluating
LLMs for conflict forecasting, but several limita-
tions remain. Our test sets are relatively small at
the country level (59 monthly points), though ex-
panded across regions for broader coverage. Class
definitions rely on slope thresholds, which may
oversimplify complex conflict dynamics. Addi-
tionally, the quality of RAG outputs varies de-
pending on retrieved content and summarization,
sometimes leading to imprecise or hallucinated es-
timates. Finally, we evaluate only general-purpose
LLMs in zero-shot settings; fine-tuned models or
multilingual pipelines could yield stronger results.
Key limitations:
•Small per-country test size; regional hetero-
geneity introduces noise;
•Skewed data: instances of Stable/Conflict and
Peace are more common than Escalate or De-
escalate;
•Threshold-based labeling may not capture nu-
anced escalation patterns;
•RAG summaries can introduce ambiguous or
noisy context;
• No fine-tuning or few-shot adaptation used;
•English-only evaluation; multilingual sources
not yet explored.
Acknowledgments
The authors appreciate the advice and guidance of
Sun-joo Lee.
References
Fulvio Attina, Marcello Carammia, and Stefano M Ia-
cus. 2022. Forecasting change in conflict fatalities

with dynamic elastic net. International Interactions ,
48(4):649–677.
He Chang, Chenchen Ye, Zhulin Tao, Jie Wu, Zheng-
mao Yang, Yunshan Ma, Xianglin Huang, and Tat-
Seng Chua. 2024. A comprehensive evaluation of
large language models on temporal event forecasting.
arXiv preprint arXiv:2407.11638 .
Mihai Croicu and Simon Polichinel von der Maase.
2025. From newswire to nexus: Using text-based ac-
tor embeddings and transformer networks to forecast
conflict dynamics. arXiv preprint arXiv:2501.03928 .
Håvard Hegre, Marie Allansson, Matthias Basedau,
Michael Colaresi, Mihai Croicu, Hanne Fjelde, Fred-
erick Hoyles, Lisa Hultman, Stina Högbladh, Remco
Jansen, et al. 2019. Views: A political violence
early-warning system. Journal of peace research ,
56(2):155–174.
Zhengbao Jiang. 2024. Towards More Factual Large
Language Models: Parametric and Non-parametric
Approaches . Ph.D. thesis, Carnegie Mellon Univer-
sity.
Carina Kauf, Anna A. Ivanova, Giulia Rambelili, Em-
manuele Chersoni, Jingyuan Selena She, Zawad
Chowdhury, Evelina Fedorenko, and Alessandro
Lenci. 2022. Event knowledge in large language
models: the gap between the impossible and the un-
likely. Cognitive science , 47 11:e13386.
Kalev Leetaru and Philip A Schrodt. 2013. Gdelt:
Global data on events, location, and tone, 1979–2012.
InISA annual convention , volume 2, pages 1–49.
Citeseer.
Alex Troy Mallen, Akari Asai, Victor Zhong, Rajarshi
Das, Hannaneh Hajishirzi, and Daniel Khashabi.
2022. When not to trust language models: Investigat-
ing effectiveness of parametric and non-parametric
memories. pages 9802–9822.
Charles A McClelland. 1984. World Event/Interaction
Survey (WEIS) Project, 1966-1978 . Inter-University
Consortium for Political and Social Research.
Hannes Mueller and Christopher Rauh. 2018. Reading
between the lines: Prediction of political violence
using newspaper text. American Political Science
Review , 112(2):358–375.
Hannes Mueller, Christopher Rauh, and Ben Seimon.
2024. Introducing a global dataset on conflict fore-
casts and news topics. Data & Policy , 6:e17.
Poli Nemkova, Solomon Ubani, Suleyman Olcay Po-
lat, Nayeon Kim, and Rodney D Nielsen. 2023.
Detecting human rights violations on social me-
dia during russia-ukraine war. arXiv preprint
arXiv:2306.05370 .
Kanghui Ning, Zijie Pan, Yu Liu, Yushan Jiang,
James Y Zhang, Kashif Rasul, Anderson Schneider,
Lintao Ma, Yuriy Nevmyvaka, and Dongjin Song.2025. Ts-rag: Retrieval-augmented generation based
time series foundation models are stronger zero-shot
forecaster. arXiv preprint arXiv:2503.07649 .
E. Rød, Håvard Hegre, and M. Leis. 2023. Predicting
armed conflict using protest data. Journal of Peace
Research .
Kutay Tire, Ege Onur Taga, Muhammed Emrullah Ildiz,
and Samet Oymak. 2024. Retrieval augmented time
series forecasting. arXiv preprint arXiv:2411.08249 .
Xinlei Wang, Maike Feng, Jing Qiu, Jinjin Gu, and Jun-
hua Zhao. 2024. From news to forecast: Integrating
event analysis in llm-based time series forecasting
with reflection. Advances in Neural Information Pro-
cessing Systems , 37:58118–58153.
Yile Wang, Peng Li, Maosong Sun, and Yang Liu. 2023.
Self-knowledge guided retrieval augmentation for
large language models. pages 10303–10315.
Jacob Wood and Deepti Joshi. 2024. Conflict-rag: Un-
derstanding evolving conflicts using large language
models. In 2024 IEEE International Conference on
Big Data (BigData) , pages 5459–5467. IEEE.
Kevin Wu, Eric Wu, and James Y Zou. 2024. Clasheval:
Quantifying the tug-of-war between an llm’s internal
prior and external evidence. Advances in Neural
Information Processing Systems , 37:33402–33422.
Rongwu Xu, Zehan Qi, Zhijiang Guo, Cunxiang Wang,
Hongru Wang, Yue Zhang, and Wei Xu. 2024.
Knowledge conflicts for llms: A survey. arXiv
preprint arXiv:2403.08319 .
Silin Yang, Dong Wang, Haoqi Zheng, and Ruochun Jin.
2025. Timerag: Boosting llm time series forecasting
via retrieval-augmented generation. In ICASSP 2025-
2025 IEEE International Conference on Acoustics,
Speech and Signal Processing (ICASSP) , pages 1–5.
IEEE.
Chenchen Ye, Ziniu Hu, Yihe Deng, Zijie Huang,
Mingyu Derek Ma, Yanqiao Zhu, and Wei Wang.
2024. Mirai: Evaluating llm agents for event fore-
casting. ArXiv , abs/2407.01231.
X Yu, Z Chen, Y Ling, S Dong, Z Liu, and Y Lu.
Temporal data meets llm–explainable financial time
series forecasting. arxiv 2023. arXiv preprint
arXiv:2306.11025 .
Hao Zhang, Yuyang Zhang, Xiaoguang Li, Wenxuan
Shi, Haonan Xu, Huanshuo Liu, Yasheng Wang,
Lifeng Shang, Qun Liu, Yong Liu, and Ruiming
Tang. 2024a. Evaluating the external and parametric
knowledge fusion of large language models. ArXiv ,
abs/2405.19010.
Ying Zhang, YangPeng Shen, Gang Xiao, and Jinghui
Peng. 2024b. Leveraging non-parametric reason-
ing with large language models for enhanced knowl-
edge graph completion. IEEE Access , 12:177012–
177027.