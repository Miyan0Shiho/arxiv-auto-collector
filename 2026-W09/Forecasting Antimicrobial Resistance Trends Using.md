# Forecasting Antimicrobial Resistance Trends Using Machine Learning on WHO GLASS Surveillance Data: A Retrieval-Augmented Generation Approach for Policy Decision Support

**Authors**: Md Tanvir Hasan Turja

**Published**: 2026-02-26 06:45:08

**PDF URL**: [https://arxiv.org/pdf/2602.22673v1](https://arxiv.org/pdf/2602.22673v1)

## Abstract
Antimicrobial resistance (AMR) is a growing global crisis projected to cause 10 million deaths per year by 2050. While the WHO Global Antimicrobial Resistance and Use Surveillance System (GLASS) provides standardized surveillance data across 44 countries, few studies have applied machine learning to forecast population-level resistance trends from this data. This paper presents a two-component framework for AMR trend forecasting and evidence-grounded policy decision support. We benchmark six models -- Naive, Linear Regression, Ridge Regression, XGBoost, LightGBM, and LSTM -- on 5,909 WHO GLASS observations across six WHO regions (2021-2023). XGBoost achieved the best performance with a test MAE of 7.07% and R-squared of 0.854, outperforming the naive baseline by 83.1%. Feature importance analysis identified the prior-year resistance rate as the dominant predictor (50.5% importance), while regional MAE ranged from 4.16% (European Region) to 10.14% (South-East Asia Region). We additionally implemented a Retrieval-Augmented Generation (RAG) pipeline combining a ChromaDB vector store of WHO policy documents with a locally deployed Phi-3 Mini language model, producing source-attributed, hallucination-constrained policy answers. Code and data are available at https://github.com/TanvirTurja

## Full Text


<!-- PDF content starts -->

Forecasting Antimicrobial Resistance Trends
Using Machine Learning
on WHO GLASS Surveillance Data: A
Retrieval-Augmented Generation
Approach for Policy Decision Support
Md Tanvir Hasan Turja
MSc Data Science, Middlesex University London
London, United Kingdom
https://github.com/TanvirTurja
February 2026
Abstract
Antimicrobial resistance (AMR) is a growing global crisis projected to cause 10
million deaths per year by 2050. While the WHO Global Antimicrobial Resistance
and Use Surveillance System (GLASS) provides standardized surveillance data across
44 countries, few studies have applied machine learning to forecast population-level
resistance trends from this data. This paper presents a two-component framework for
AMR trend forecasting and evidence-grounded policy decision support. We benchmark
six models—Naive, Linear Regression, Ridge Regression, XGBoost, LightGBM, and
LSTM—on 5,909 WHO GLASS observations across six WHO regions (2021–2023).
1arXiv:2602.22673v1  [cs.LG]  26 Feb 2026

XGBoost achieved the best performance with a test MAE of 7.07% and R2of 0.854,
outperforming the naive baseline by 83.1%. Feature importance analysis identified the
prior-year resistance rate as the dominant predictor (50.5% importance), while regional
MAE ranged from 4.16% (European Region) to 10.14% (South-East Asia Region),
reflecting disparities in GLASS data quality. We additionally implemented a Retrieval-
Augmented Generation (RAG) pipeline combining a ChromaDB vector store of WHO
policy document excerpts with a locally deployed Phi-3 Mini language model, injecting
forecast context into each query to produce source-attributed, hallucination-constrained
policy answers. Together, these components demonstrate the feasibility of integrating
AMR surveillance forecasting with intelligent policy question answering using openly
available data and locally deployable models.
Keywords:antimicrobial resistance, WHO GLASS, machine learning, XGBoost,
LSTM, time series forecasting, retrieval-augmented generation, public health surveil-
lance, policy decision support
1 Introduction
Antimicrobial resistance (AMR) represents one of the most urgent threats to global health in
the twenty-first century. In 2019, AMR was directly responsible for an estimated 1.27 million
deaths worldwide and was associated with an additional 4.95 million deaths, with the highest
burden concentrated in Sub-Saharan Africa and South Asia [Murray et al., 2022]. Without
coordinated international action, AMR is projected to cause 10 million deaths per year by
2050, surpassing cancer as the leading cause of death globally [O’Neill, 2016, de Kraker et al.,
2016]. These projections carry wide-ranging consequences across patient outcomes, healthcare
systems, and national economies [Dadgostar, 2019, Tang et al., 2023], and have prompted
global political responses including the WHO Global Action Plan on Antimicrobial Resistance
[World Health Organization, 2015, Sugden et al., 2016].
A critical prerequisite for effective AMR governance is the ability to monitor and forecast
2

resistance trends across pathogens, antibiotics, and geographic regions. The WHO Global
Antimicrobial Resistance and Use Surveillance System (GLASS) was established to provide
standardized, comparable AMR data from national surveillance networks worldwide [World
Health Organization, 2022, 2023]. GLASS data captures resistance rates across multiple
pathogen-antibiotic combinations and has been used to demonstrate the relationship between
antibiotic consumption and resistance across participating countries [Ajulo and Awosile,
2024]. However, GLASS data availability remains uneven across regions, with participation
concentrated in higher-income countries and more established surveillance systems.
Machine learning has attracted growing interest as a tool for AMR prediction, primarily
applied to clinical isolate datasets to classify susceptibility or resistance at the level of
individual bacterial samples [Sakagianni et al., 2023, Kim et al., 2022]. These approaches
have shown promise within clinical laboratory settings but do not directly address population-
level forecasting of resistance trends from surveillance data. The temporal and geographic
structure of GLASS data, covering resistance rates over time and across countries, calls for
time series forecasting approaches rather than isolate-level classification.
Beyond forecasting, there is a parallel need for tools that make AMR surveillance findings
interpretable and actionable for policymakers and public health practitioners. Large language
models (LLMs) and Retrieval-Augmented Generation (RAG) systems [Lewis et al., 2020]
offer a promising direction for this purpose, enabling natural-language policy queries to be
answered with evidence grounded in specific source documents rather than model parameters
alone.
This paper addresses both gaps with a two-component framework. First, we benchmark six
machine learning models on WHO GLASS surveillance data (2021–2023) across 44 countries
to forecast resistance rates at the country-pathogen-antibiotic level. Second, we implement a
RAG pipeline that combines forecast results with retrieved WHO policy documents to answer
natural-language questions about AMR policy and resistance trends.
Our main contributions are:
3

1.A multi-model benchmark (Naive, Linear Regression, Ridge Regression, XGBoost,
LightGBM, LSTM) on WHO GLASS surveillance data for population-level resistance
rate forecasting, identifying XGBoost as the best-performing approach with a test MAE
of 7.07% and R2of 0.854.
2. A feature importance and regional error analysis identifying the resistance lag feature
as the dominant predictor and highlighting geographic disparities in forecast accuracy
tied to GLASS participation rates.
3.A locally deployed RAG system grounding AMR policy answers in retrieved WHO
document chunks and forecast context, without requiring proprietary cloud APIs.
2 Related Work
2.1 Machine Learning for AMR Prediction
The application of machine learning to AMR has grown substantially over the past decade.
Sakagianni et al. [2023] provided a comprehensive literature review of ML approaches to
AMR prediction, covering random forests, support vector machines, neural networks, and
gradient boosting methods. Kim et al. [2022] reviewed current practice and limitations from
a clinical perspective, noting that most existing work focuses on isolate-level susceptibility
classification rather than population-level temporal forecasting. Both reviews identified a
gap in approaches using surveillance data for anticipatory forecasting of resistance trends,
which is the primary focus of this work.
2.2 Gradient Boosting for Epidemiological Forecasting
Gradient boosting methods have demonstrated consistent advantages over classical statistical
models for infectious disease time series. Alim et al. [2020] compared XGBoost and ARIMA
for predicting human brucellosis incidence in China, finding that XGBoost achieved a test
4

MAPE of 7.64% compared to 17.68% for ARIMA, attributing XGBoost’s advantage to its
ability to model nonlinear patterns without stationarity assumptions. Fang et al. [2022]
replicated this finding for COVID-19 transmission forecasting in the United States. Ahn
et al. [2023] demonstrated that ensemble gradient boosting (XGBoost, LightGBM, CatBoost)
combined with attention-based CNN-LSTM achieved strong performance for environmental
time series forecasting, validating both XGBoost [Chen and Guestrin, 2016] and LightGBM
[Ke et al., 2017] in a forecasting context with temporal structure similar to our AMR dataset.
2.3 LSTM Networks for Epidemic Forecasting
Chimmula and Zhang [2020] applied LSTM to model COVID-19 transmission dynamics
in Canada, demonstrating the feasibility of LSTM for short-horizon epidemic forecasting
and motivating its inclusion in our model comparison. In our evaluation, LSTM achieves
performance comparable to gradient boosting (MAE: 7.16%), confirming that the limited
temporal dataset is the constraining factor rather than model architecture choice.
2.4 Retrieval-Augmented Generation
Lewis et al. [2020] introduced the RAG framework, combining a dense retrieval component
with a sequence-to-sequence language model to answer knowledge-intensive questions by
grounding responses in retrieved document passages. Our work applies RAG to the AMR
domain, integrating WHO policy documents with ML-derived forecast context to support
evidence-based policy queries. To our knowledge, this is the first application of RAG
specifically for AMR policy question answering informed by GLASS surveillance forecasts.
5

3 Data and Methods
3.1 Dataset
We used the WHO Global Antimicrobial Resistance and Use Surveillance System (GLASS)
dataset covering the years 2021–2023 [World Health Organization, 2022, 2023]. The dataset
contains standardized resistance observations collected from national surveillance networks
across 44 countries spanning six WHO geographic regions. The final processed dataset
comprised 5,909 observations after cleaning. Each observation records a unique combination
of country, pathogen, antibiotic, and year. The target variable is the resistance percentage,
defined as the proportion of tested isolates that demonstrated resistance to the corresponding
antibiotic (Table 1).
Table 1: Dataset characteristics.
Attribute Value
Years covered 2021–2023
Number of countries 44
WHO regions 6
Total observations 5,909
Target variable Resistance percentage (%)
Income groups High, Upper-middle, Lower-middle, Low, Unknown
The AMR-consumption relationship documented by Ajulo and Awosile [2024] across
GLASS-participating countries informed our feature selection, particularly the inclusion of
antibiotic consumption as a predictive feature.
6

3.2 Features
Input features for all supervised models included: pathogen name, antibiotic name, country
or territory, WHO region, income group (World Bank classification), year, resistance lag
(resistance percentage in the prior year for the same pathogen-antibiotic-country combination),
and antibiotic consumption (defined daily doses per 1,000 inhabitants per day). High-
cardinality categorical features were encoded using target encoding applied within the
training set to prevent data leakage.
3.3 Preprocessing
Missing values in the target variable and lag features were imputed using the median resistance
rate for the corresponding pathogen-antibiotic pair. The dataset was split temporally: 2021–
2022 observations for training, 2022 observations for validation, and 2023 observations for
the held-out test set, preventing information leakage from future time points into training.
3.4 Models
We compared six models of increasing complexity:
Naive Baseline.Predicts each observation’s resistance rate as equal to the prior year’s
resistance rate (lag1). This represents the minimum performance threshold any model must
surpass.
Linear Regression.Ordinary least squares regression on all encoded features.
Ridge Regression.L2-regularized linear regression; regularization parameter selected
via cross-validation.
XGBoost.Scalable gradient boosting framework using decision trees as base learners
[Chen and Guestrin, 2016]. XGBoost has consistently outperformed classical statistical
models for infectious disease time series forecasting [Alim et al., 2020, Fang et al., 2022].
LightGBM.Histogram-based gradient boosting algorithm offering efficient training
7

on tabular data [Ke et al., 2017]. Gradient boosting ensembles including LightGBM have
demonstrated strong performance on biological time series [Ahn et al., 2023].
LSTM.Single-layer Long Short-Term Memory network with sequences of length 1 given
the limited temporal span of the dataset [Chimmula and Zhang, 2020]. Trained on CPU due
to GPU driver incompatibility with the available PyTorch build.
Hyperparameters for XGBoost and LightGBM were tuned via grid search over learning
rate, maximum depth, number of estimators, and subsampling ratio. LSTM training used
the Adam optimizer (lr= 0.001) with early stopping on validation MAE.
3.5 Evaluation Metrics
All models were evaluated on the held-out 2023 test set using Mean Absolute Error (MAE),
Root Mean Square Error (RMSE), and R2. Regional MAE was additionally computed by
disaggregating test predictions by WHO region.
3.6 RAG Pipeline
To augment model forecasts with interpretable policy context, we implemented a Retrieval-
Augmented Generation (RAG) system [Lewis et al., 2020].
Knowledge base.Six curated WHO policy document excerpts covering: carbapenem
resistance and ESKAPE pathogens, the Global Action Plan on AMR [World Health Organi-
zation, 2015], the AWaRe antibiotic classification, AMR surveillance investment priorities for
LMICs, and regional resistance trends from GLASS data.
Embedding and retrieval.Documentchunkswereencodedusingthe all-MiniLM-L6-v2
sentence transformer model and stored in a ChromaDB vector database. The top 3 most
semantically relevant chunks were retrieved per query using cosine similarity.
Language model.Retrieved chunks were combined with XGBoost forecast findings
and regional MAE results to construct a structured prompt submitted to Phi-3 Mini (3.8B
parameters) via Ollama. The system prompt explicitly prohibited fabricated citations,
8

restricting source references to retrieved document chunk labels only.
Forecast context injection.Each query received an appended context block sum-
marizing XGBoost feature importances, regional MAE values, and the model performance
comparison table.
4 Results
4.1 Model Performance Comparison
Table 2 presents the performance of all six models on the 2023 held-out test set.
Table 2: Model performance on the 2023 held-out test set.
Model Test MAE (%) Test RMSE (%) Test R2vs. Naive
Naive Baseline 41.83 48.15 — Reference
Linear Regression 8.23 11.96 0.821 80.3%
Ridge Regression 8.25 11.95 0.821 80.3%
LightGBM 7.17 10.84 0.853 82.9%
LSTM 7.16 10.10 0.872 82.9%
XGBoost 7.07 10.80 0.854 83.1%
9

Figure 1: Model performance comparison across all six models on validation and test sets.
XGBoost achieves the lowest test MAE (7.07%), with LightGBM and LSTM closely behind.
XGBoost achieved the lowest test MAE of 7.07% and R2of 0.854, outperforming the naive
baseline by 83.1%. This is consistent with prior findings demonstrating XGBoost superiority
over statistical baselines for infectious disease time series [Alim et al., 2020, Fang et al., 2022].
Gradient boosting models and the LSTM achieved comparable performance with a spread of
less than 0.15 percentage points in MAE. The convergence of deep learning and tree-based
performance is expected given the limited three-year temporal span; LSTM networks typically
provide greater advantage over longer sequences [Chimmula and Zhang, 2020].
4.2 Feature Importance Analysis
Table 3 shows the top five XGBoost features by gain-based importance.
10

Table 3: XGBoost feature importances (gain-based, top 5).
Rank Feature Importance (%)
1 Resistance_lag1 (prior year resistance) 50.5
2 CountryTerritoryArea 9.0
3 AntibioticName 6.7
4 PathogenName 6.0
5 Antibiotic consumption (DID) 4.8
Figure 2: XGBoost feature importance (gain-based). Resistance_lag1 dominates at 50.5%,
confirming strong temporal autocorrelation in AMR resistance rates.
The dominance of the resistance lag feature (50.5%) reflects strong temporal autocorrela-
tion: a country’s resistance in a given year is the single strongest predictor of its resistance
the following year. Country-level heterogeneity (9.0%) and antibiotic identity (6.7%) account
for the next largest shares. The inclusion of antibiotic consumption as the fifth-ranked
feature (4.8%) is consistent with the AMR-consumption relationship documented by Ajulo
and Awosile [2024].
11

4.3 Regional Error Analysis
Table 4 presents the disaggregated test MAE by WHO region.
Table 4: XGBoost test MAE by WHO region (2023 test set). The Region of the Americas
had no observations in the 2023 test set and is excluded.
WHO Region Test MAE (%) Observations
European Region 4.16 809
Western Pacific Region 7.30 181
African Region 8.70 265
Eastern Mediterranean Region 9.00 709
South-East Asia Region 10.14 161
Figure 3: XGBoost test MAE disaggregated by WHO region. The European Region achieves
the lowest error (4.16%), while South-East Asia shows the highest (10.14%), reflecting
disparities in GLASS data coverage.
12

Forecast accuracy varied substantially across regions. The European Region achieved
the lowest MAE (4.16%), reflecting comprehensive GLASS participation and high data
consistency. The South-East Asia Region showed the highest MAE (10.14%), approximately
2.4 times higher than Europe. This regional gradient closely mirrors the global AMR burden
distribution documented by Murray et al. [2022], in which high-burden regions also tend to
have weaker surveillance infrastructure.
4.4 RAG Policy Q&A System
Figure 4: XGBoost residual and error analysis on the 2023 test set. Most predictions fall
within a narrow error range, with larger errors concentrated in high-resistance observations.
We evaluated the RAG system on five predefined policy questions spanning antibiotic priori-
tization by region, treatment guidelines for resistant pathogens, regional forecast uncertainty,
income-stratified resistance trends, and surveillance investment priorities. All five responses
demonstrated: (1) accurate retrieval of relevant WHO policy excerpts, (2) correct attribution
13

of retrieved sources without fabricated references, and (3) coherent integration of forecast
findings into policy recommendations.
Representative example.Question:Which antibiotics should Southeast Asia prioritize
preserving based on resistance forecasts?The system correctly identified carbapenem-resistant
ESKAPE pathogens as critical threats, recommended prioritizing Access-group antibiotics
per the AWaRe classification, connected the region’s high forecast uncertainty (MAE: 10.1%)
to surveillance infrastructure gaps, and recommended targeted investment in antibiotic
stewardship and laboratory capacity consistent with WHO LMIC guidance.
5 Discussion
5.1 Forecasting Performance
Our results confirm XGBoost as an effective model for short-term AMR resistance rate
forecasting from WHO surveillance data (MAE: 7.07%, R2: 0.854). The dominance of the
resistance lag feature (50.5%) indicates that near-term AMR forecasting is primarily autore-
gressive: past resistance strongly predicts future resistance. This has practical implications
for resource-constrained settings, as lag-only features capture the majority of the predictable
signal.
The near-equivalent performance of gradient boosting and LSTM models (MAE spread
<0.15percentage points) suggests that the three-year temporal window does not provide
sufficient depth for LSTM to exploit its sequential modeling advantage. Extended GLASS
time series may reveal more meaningful gains from deep learning, consistent with findings
from longer epidemic datasets [Chimmula and Zhang, 2020].
The regional MAE gradient reflects inequities in GLASS participation. Expanding GLASS
enrollment in high-burden regions would simultaneously improve AMR surveillance and
forecasting accuracy.
14

5.2 RAG System Evaluation
The RAG pipeline produced grounded, policy-relevant answers by combining retrieved WHO
document excerpts with XGBoost forecast context. The prompt-level citation constraint
effectively prevented hallucinated references, a known failure mode of large language models
[Lewis et al., 2020]. Expanding the knowledge base with complete WHO GLASS annual
reports and national AMR action plans would substantially improve retrieval coverage.
5.3 Limitations
Several limitations should be acknowledged. First, the dataset spans only three years (2021–
2023), restricting the model’s ability to capture multi-year emergence trajectories. Second,
GLASS data represents a geographically non-random sample weighted toward countries with
existing surveillance capacity, introducing selection bias in regional comparisons. Third, the
RAG knowledge base is small and manually curated. Fourth, Phi-3 Mini has lower reasoning
capacity than larger frontier models for complex multi-step queries.
5.4 Future Directions
Future work should explore: (1) integration of 2024 GLASS data to extend the temporal
window; (2) inclusion of complete WHO policy documents in the RAG knowledge base; (3)
evaluation of the Temporal Fusion Transformer for longer-range forecasting; (4) domain-
adapted sentence embeddings trained on AMR literature; and (5) formal automated evaluation
of RAG response faithfulness using frameworks such as RAGAS.
6 Conclusion
This paper presented an integrated framework for AMR resistance rate forecasting and
evidence-grounded policy question answering using WHO GLASS surveillance data. Across
six benchmarked models, XGBoost [Chen and Guestrin, 2016] achieved the best forecasting
15

performance (MAE: 7.07%, R2: 0.854), representing an 83.1% improvement over the naive
baseline. The prior-year resistance lag accounted for over half of model predictive power, and
regional forecast accuracy tracked closely with GLASS data quality.
We additionally implemented a RAG pipeline [Lewis et al., 2020] producing grounded,
source-attributed responses to AMR policy questions without fabricated citations, demon-
strating the feasibility of integrating ML-based surveillance forecasts into a locally deployable
policy decision support tool.
As AMR is projected to cause 10 million deaths per year by 2050 [O’Neill, 2016, de Kraker
et al., 2016], tools that translate global surveillance data into actionable policy insights are
urgently needed. Expanding GLASS participation and coupling those data streams with
forecasting and retrieval systems represents a concrete step toward evidence-based AMR
governance at global scale.
References
Jae Min Ahn, Jinhyun Kim, and Kyunghyun Kim. Ensemble machine learning of gradient
boosting (XGBoost, LightGBM, CatBoost) and attention-based CNN-LSTM for harmful
algal blooms forecasting.Toxins, 15(10):608, 2023. doi: 10.3390/toxins15100608.
Samuel Ajulo and Babafela Awosile. Global antimicrobial resistance and use surveillance
system (GLASS 2022): Investigating the relationship between antimicrobial resistance and
antimicrobial consumption data across the participating countries.PLOS ONE, 19(2):
e0297921, 2024. doi: 10.1371/journal.pone.0297921.
Mohammad Alim, Guo-Hong Ye, Peng Guan, De-Sheng Huang, Bao-Sen Zhou, and Wei Wu.
Comparison of ARIMA model and XGBoost model for prediction of human brucellosis
in mainland China: a time-series study.BMJ Open, 10:e039498, 2020. doi: 10.1136/
bmjopen-2020-039498.
16

Tianqi Chen and Carlos Guestrin. XGBoost: A scalable tree boosting system. InProceedings
of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data
Mining, pages 785–794, 2016. doi: 10.1145/2939672.2939785.
Vinay Kumar Reddy Chimmula and Lei Zhang. Time series forecasting of COVID-19
transmission in Canada using LSTM networks.Chaos, Solitons and Fractals, 135:109864,
2020. doi: 10.1016/j.chaos.2020.109864.
Porin Dadgostar. Antimicrobial resistance: Implications and costs.Infection and Drug
Resistance, 12:3903–3910, 2019. doi: 10.2147/IDR.S234610.
Marlieke E. A. de Kraker, Andrew J. Stewardson, and Stephan Harbarth. Will 10 million
people die a year due to antimicrobial resistance by 2050?PLOS Medicine, 13(11):e1002184,
2016. doi: 10.1371/journal.pmed.1002184.
Zhili Fang, Shuai Yang, Chenglong Lv, Su An, and Wei Wu. Application of a data-driven
XGBoost model for the prediction of COVID-19 in the USA: a time-series study.BMJ
Open, 12(7):e056685, 2022. doi: 10.1136/bmjopen-2022-056685.
Guolin Ke, Qi Meng, Thomas Finley, Taifeng Wang, Wei Chen, Weidong Ma, Qiwei Ye, and
Tie-Yan Liu. LightGBM: A highly efficient gradient boosting decision tree. InAdvances in
Neural Information Processing Systems (NeurIPS), volume 30, 2017.
Jong In Kim, Finlay Maguire, Kara K. Tsang, Theodore Gouliouris, Sharon J. Peacock,
Tim A. McAllister, Andrew G. McArthur, and Robert G. Beiko. Machine learning for
antimicrobial resistance prediction: Current practice, limitations, and clinical perspective.
Clinical Microbiology Reviews, 35(3):e00179–21, 2022. doi: 10.1128/cmr.00179-21.
Patrick Lewis, Ethan Perez, Aleksandra Piktus, Fabio Petroni, Vladimir Karpukhin, Naman
Goyal, Heinrich Kuttler, Mike Lewis, Wen-tau Yih, Tim Rocktaschel, Sebastian Riedel,
and Douwe Kiela. Retrieval-augmented generation for knowledge-intensive NLP tasks. In
17

Advances in Neural Information Processing Systems (NeurIPS), volume 33, 2020. URL
https://arxiv.org/abs/2005.11401.
Christopher J. L. Murray et al. Global burden of bacterial antimicrobial resistance in 2019:
a systematic analysis.The Lancet, 399(10325):629–655, 2022. doi: 10.1016/S0140-6736(21)
02724-0.
Jim O’Neill. Tackling drug-resistant infections globally: Final report and recommendations.
Review on Antimicrobial Resistance, 2016.
AikateriniSakagianni, ChristinaKoufopoulou, GeorgiosFeretzakis, DimitrisKalles, VassiliosS.
Verykios, Pavlos Myrianthefs, and Galanos Fildisis. Using machine learning to predict
antimicrobial resistance: A literature review.Antibiotics, 12(3):452, 2023. doi: 10.3390/
antibiotics12030452.
Ruth Sugden, Rebecca Kelly, and Sally Davies. Combatting antimicrobial resistance globally.
Nature Microbiology, 1:16187, 2016. doi: 10.1038/nmicrobiol.2016.187.
Kin Weng Kong Tang, Brian C. Millar, and John E. Moore. Antimicrobial resistance (AMR).
British Journal of Biomedical Science, 80:11387, 2023. doi: 10.3389/bjbs.2023.11387.
World Health Organization. Global action plan on antimicrobial resistance. Technical report,
WHO, 2015. URLhttps://www.who.int/publications/i/item/9789241509763.
World Health Organization. Global antimicrobial resistance and use surveillance system
(GLASS) report: 2021–2022. Technical report, WHO, 2022. URL https://www.who.int/
publications/i/item/9789240062702.
World Health Organization. GLASS manual for antimicrobial resistance surveillance in
common bacteria causing human infection. Technical report, WHO, 2023.
18