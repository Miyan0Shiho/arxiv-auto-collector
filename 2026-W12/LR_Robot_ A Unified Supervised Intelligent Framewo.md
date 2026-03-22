# LR-Robot: A Unified Supervised Intelligent Framework for Real-Time Systematic Literature Reviews with Large Language Models

**Authors**: Wei Wei, Jin Zheng, Zining Wang

**Published**: 2026-03-18 13:43:07

**PDF URL**: [https://arxiv.org/pdf/2603.17723v1](https://arxiv.org/pdf/2603.17723v1)

## Abstract
Recent advances in artificial intelligence (AI) and natural language processing (NLP) have enabled tools to support systematic literature reviews (SLRs), yet existing frameworks often produce outputs that are efficient but contextually limited, requiring substantial expert oversight.The framework employs a human-in-the-loop process to define sub-SLR tasks, evaluate models, and ensure methodological rigor, while leveraging structured knowledge sources and retrieval-augmented generation (RAG) to enhance factual grounding and transparency. LR-Robot enables multidimensional categorization of research, maps relationships among papers, identifies high-impact works, and supports historical, fine-grained analyses of topic evolution. We demonstrate the framework using an option pricing case study, enabling comprehensive literature analysis. Empirical results reveal the current capabilities of AI in understanding and synthesizing literature, uncover emerging trends, reveal topic connections, and highlight core research directions. By accelerating labor-intensive review stages while preserving interpretive accuracy, LR-Robot provides a practical, customizable, and high-quality approach for AI-assisted SLRs. Key contributions: (1) a novel framework combining AI and expert supervision for contextually informed SLRs, (2) support for multidimensional categorization, relationship mapping, and fine-grained topic evolution analysis, and (3) empirical demonstration of AI-driven literature synthesis in the field of option pricing.

## Full Text


<!-- PDF content starts -->

LR-Robot: A Unified Supervised Intelligent
Framework for Real-Time Systematic
Literature Reviews with Large Language
Models
Wei Weia
Jin Zhenga
Zining Wanga
aSchool of Engineering Mathematics and Technology, University of Bristol, Ada
Lovelace Building, Tankard’s Close, Bristol, BS8 1TW, England, United Kingdom
Corresponding author:Jin Zheng
Email: jin.zheng@bristol.ac.ukarXiv:2603.17723v1  [q-fin.GN]  18 Mar 2026

Abstract
Recent advances in artificial intelligence (AI) and natural language processing
(NLP) have enabled tools to support systematic literature reviews (SLRs), yet
existing frameworks often produce outputs that are efficient but contextually
limited, requiring substantial expert oversight. To address these challenges, we
propose LR-Robot, a novel supervised–augmented framework that integrates
expert judgment with AI capabilities to enhance literature synthesis. The frame-
work employs ahuman-in-the-loopprocess to define sub-SLR tasks, evaluate
models, and ensure methodological rigor, while leveraging structured knowledge
sources and retrieval-augmented generation (RAG) to enhance factual ground-
ing and transparency. LR-Robot enables multidimensional categorization of
research, maps relationships among papers, identifies high-impact works, and
supports historical, fine-grained analyses of topic evolution.
We demonstrate the framework using an option pricing case study, enabling
comprehensive literature analysis. Empirical results reveal the current capabili-
ties of AI in understanding and synthesizing literature, uncover emerging trends,
reveal topic connections, and highlight core research directions. By accelerating
labor-intensive review stages while preserving interpretive accuracy, LR-Robot
provides a practical, customizable, and high-quality approach for AI-assisted
SLRs.
Key contributions: (1) a novel framework combining AI and expert supervision
for contextually informed SLRs, (2) support for multidimensional categorization,
relationship mapping, and fine-grained topic evolution analysis, and (3) empirical
demonstration of AI-driven literature synthesis in the field of option pricing.
Keywords:Systematic Literature Review, Large Language Models, Bibliometric
Network, Knowledge Graph
1 Introduction
A systematic literature review (SLR) is not merely a preparatory step in academic
research but a fundamental methodological process that underpins all forms of schol-
arly inquiry. It serves as the intellectual foundation upon which rigorous and credible
research is constructed, enabling scholars to consolidate existing knowledge, iden-
tify theoretical and methodological gaps, and position new contributions within the
broader disciplinary landscape (Snyder 2019; Webster and Watson 2002). When con-
ducted systematically, a SLR offers both a comprehensive overview of the state of
2

knowledge and a critical evaluation of its limitations. Conversely, insufficient or unsys-
tematic reviews often lead to redundant, fragmented, or poorly contextualized studies
that lack theoretical coherence and fail to engage meaningfully with the research fron-
tier (Boell and Cecez-Kecmanovic 2015). Beyond its value for researchers, literature
synthesis also plays a vital role for learners and practitioners, providing structured
access to complex bodies of knowledge, supporting evidence-based understanding, and
facilitating the transfer of insights across domains. In this sense, SLR serves not only
as a research instrument but also as a powerful learning and knowledge integration
mechanism for the wider academic and professional community.
Over the past decades, the academic publishing landscape has expanded exponen-
tially, producing millions of new articles each year across disciplines. This explosive
growth has made traditional SLR methods of relying heavily on human judgment,
manual screening, close reading, classifying studies by topic, method, or contribution,
and finally synthesizing the findings into coherent narratives or frameworks (Okoli
2015) increasingly time-consuming and cognitively demanding, often struggling to
maintain completeness, timeliness, and reproducibility. As highlighted by Tsertsvadze
et al. (2015), conventional expert-driven reviews can take months or even years to com-
plete, rendering them increasingly impractical for keeping pace with rapidly evolving
research domains.
In response to these challenges, numerous scholars have developed methodological
innovations to make literature reviews more systematic, quantitative, and scalable.
Techniques such as citation network analysis (Shi et al. 2021), co-citation clustering
(ˇSubelj et al. 2016), keyword co-occurrence and thematic clustering (Lages et al. 2023),
and co-authorship network analysis (Donthu et al. 2021) have been widely applied
to uncover structural patterns, thematic clusters, influential works, and collaboration
networks within large bodies of research. Even though they offer clear advantages
in scalability and objectivity, they have inherent limitations. They primarily rely on
bibliometric metadata, which may overlook nuanced conceptual contributions, sub-
tle methodological differences, and evolving semantic shifts over time (Baccini and
De Nicolao 2016). Additionally, clustering results can be sensitive to parameter set-
tings and data quality, and quantitative patterns alone cannot fully capture the depth,
context, or critical appraisal that expert-driven synthesis provides (Allen et al. 2009).
Consequently, while bibliometric and network-based methods provide valuable struc-
tural insights, they remain complementary to, rather than a replacement for, expert
judgment in conducting comprehensive and rigorous literature reviews.
Recent advances in artificial intelligence (AI) and natural language processing
(NLP) have facilitated the development of tools to assist SLR (Bolanos et al. 2024).
Some studies have further proposed automated frameworks for SLR, designed to
streamline the entire review process from the screening and data extraction of relevant
studies to the generation of synthesized content. In these frameworks, large language
models (LLMs) are typically employed to interpret user requirements, retrieve and
classify relevant articles, and generate preliminary summaries or conceptual overviews
based on the selected corpus (Han et al. 2024; Dennst¨ adt et al. 2024; Agarwal et al.
2024). While these approaches can substantially improve the efficiency of literature
synthesis, several challenges remain. First, user intent is often not clearly defined,
3

particularly when users lack sufficient domain knowledge to formulate precise or com-
prehensive search criteria. This may result in overly narrow, incomplete, or even
misleading document retrieval. Second, as most existing frameworks generate con-
tent solely in response to user-specified topics, they tend to overlook the need for a
broader, forward-looking, and integrative synthesis of the research field. In contrast,
a rigorous literature review requires situating focused topics within a comprehen-
sive and theoretically informed understanding of the domain (Snyder 2019; Boell
and Cecez-Kecmanovic 2015). Consequently, current AI-driven LR systems often pro-
duce outputs that are efficient but contextually limited, emphasizing the continuing
necessity of expert supervision to ensure conceptual depth, methodological rigor, and
interpretive accuracy.
To address these limitations, we have developed a supervised–augmented frame-
work that integrates expert understanding with AI capabilities. During system
development, all processes are conducted under expert supervision, with researchers
validating outputs, refining prompts, and ensuring contextual and methodologi-
cal accuracy. The integration of structured knowledge sources such as knowledge
graphs and retrieval-augmented generation (RAG) frameworks further enhances fac-
tual grounding and transparency (Meloni et al. 2023; Salatino et al. 2022). Leveraging
AI’s ability to rapidly process large volumes of documents, complete classification, and
perform large-scale text analysis, the system can analyze all relevant literature within
a field to provide a comprehensive overview of the topics. While AI accelerates labor-
intensive review stages and supports content synthesis, it is used to augment expert
judgment rather than replace it. Once established, the framework enables end-users to
automatically generate customized literature syntheses and targeted document selec-
tion, producing their own systematic literature reviews tailored to specific research
needs.
The remainder of the paper is organized as follows. Section 2 introduces the
supervised–augmented framework, LR-Robot for SLR, which standardizes a human-
in-the-loop process to define sub-SLR tasks and identify the best-performing models
through systematic evaluation. Section 3 presents the implementation of the human-
in-the-loop process using an option pricing sample dataset. Section 4 demonstrates
the comprehensive application of the framework to the entire option pricing litera-
ture, enabling detailed literature analyses, including network analysis and temporal
evolution studies, which help researchers understand emerging trends, topic connec-
tions, and core directions in the field. Finally, Section 5 summarizes our conclusions
and outlines directions for future research.
2 Framework: LR-Robot
2.1 Overview
This section presents the framework (see Fig. 1) for an open-source, RAG-based
LLM platform that supports real-time SLRs with human-in-the-loop supervision. The
framework orchestrates data gathering, pre-processing, task definition, prompt design,
model evaluation, and output synthesis, while enabling iterative refinement through
4

human feedback. The architecture separates developer-mode operations (for task cre-
ation, data curation, and model evaluation) from user mode interactions (querying
and receiving outputs).The framework comprises four layers: Data Retrieval, Human-
in-the-loop Processing, RAG Database with ALL Output, and Interactive Output
Visualization. The first three layers pertain to developer mode, while the fourth layer
supports user mode.
Fig. 1Framework of LR-Robot
2.2 Architecture and Components
2.2.1 Developer mode
Data Retrieval.In the data retrieval layer, we provide a data collection protocol
capable of aggregating information from multiple sources. Initially, a query is gen-
erated from the user’s prompt to target an LLM. This prompt may specify topics,
domains and research ideas. The LLM, in collaboration with a human operator, iden-
tifies keywords and automatically searches across the databases. Once the keywords
are selected, the system performs daily checks for data updates and automatically
stores relevant papers and metadata to our RAG databases.
Human-in-the-loop Processing.The process starts with LR-Robot formulates
multiple sub-tasks of the SLR (e.g., classification, ranking, relationship analysis, data
5

extraction, et al.) based on human instructions. It subsequently defines prompts for
the tasks. A human researcher reviews and approves the LLM-generated instructions
and provides more precise guidance to shape the LLM’s output, i.e. constraints in
the prompt. Evaluation is conducted on a representative sample dataset, with the
researcher assessing outputs for accuracy, relevance, completeness, and other prede-
fined evaluation metrics. The evaluation loop informs refinements to the prompts to
support iterative improvement. The research may also compare the performance of a
variety of LLMs, and offer recommendations for the best-performing models.
RAG Database with All Outputs.After identifying the best-performing mod-
els, we generated outputs across the entire dataset and stored the results in our RAG
database. These aggregated outputs enable downstream SLR-related tasks—such as
network analysis, topic evolution, and others—to be conducted effectively. Further-
more, the outputs are automatically updated on a daily basis to incorporate newly
available data.
2.2.2 User mode
Interactive Output Visualization.In the user mode, a user submits questions
on the topic, and the LLM interprets these queries and extract relevant data from the
RAG database. The system then generates an organized output, and the user may
choose among multiple visualization options to view the results.
3 Framework Implementation and Evaluation in
Option Pricing Research
3.1 Data Overview
To investigate the practical applicability of the proposed framework, we employ it to
analyze the body of literature related to option pricing. We collect bibliometric data
from the Scopus database, encompassing all publications related to option pricing up
to April 2, 2025. The dataset includes comprehensive metadata such as article titles,
author names, publication years, source titles, abstracts, and references. Both journal
papers and conference papers are included to ensure broad coverage of the research
landscape.1
The initial search yield a total of 15,424 articles from the Scopus database. After
removing records with missing or incomplete bibliometric information (such as absent
abstracts), 11,916 valid samples are retained for subsequent analysis. These articles
are published across a wide range of journals and conference proceedings, reflecting the
interdisciplinary nature of research on option pricing. The distribution of publications
1The data are retrieved using the following query: TITLE-ABS-KEY (”pric* of option*” OR ”option
pric*” OR ”implied volatility” OR ”European option*” OR ”American option*” OR ”stock option*” OR
”crypto* option*” OR ”interest option*” OR ”interest rate option*” OR ”exotic option*” OR ”basket
option*” OR ”barrier option*” OR ”binary option*” OR ”Bermuda option*” OR ”compound option*”)
AND (LIMIT-TO (LANGUAGE, ”English”)) AND (LIMIT-TO (DOCTYPE, ”ar”) OR LIMIT-TO (DOC-
TYPE, ”cp”) OR LIMIT-TO (DOCTYPE, ”re”) OR LIMIT-TO (DOCTYPE, ”cr”)). This query ensure the
inclusion of English-language research articles and conference proceedings relevant to option pricing and its
associated domains.
6

across source titles is illustrated in Fig.2, which presents the top 10 most frequent
source titles in the dataset.
Fig. 2Number of Papers Distributed Across Journals
3.2 Multi-Dimensional Topic Modelling and LLMs
We select four dimensions for analyzing the option pricing problem: (1)whether pric-
ing models are used or developed, (2) underlying types, (3) option types, and (4)
model types. The dimension set is grounded in the existing literature (Sharma and
Kumar Verma 2025; Ruf and Wang 2020; Broadie and Detemple 2004) and aims to
provide a comprehensive framework for analyzing option pricing problems.
3.2.1 Option Pricing Models
In the first dimension, we classify each paper based on whether it discusses the develop-
ment or comparison of pricing or volatility models. This classification is further refined
by incorporating human expertise to guide the AI-based categorization. Specifically,
the AI is instructed to respond “no” when a paper primarily focuses on empirical stud-
ies, market microstructure, credit risk, risk management, non-traditional markets, or
other unrelated topics. The full list is in Appendix A.1.
To evaluate the performance of different LLMs, we randomly select a sample of
1,000 papers and manually labeled them. We then assess several LLMs—including
GPT-4o, Gemini Flash 2.0, DeepSeek-8B, DeepSeek V3, and DeepSeek R1—based on
their prediction accuracy and consistency. From the experiments, we selected the best
results. A total of 417 papers were labeled by an LLM with positive classification, and
we will use those 417 papers for the remaining three dimensions of analysis.
Table 1 presents the performance of each model. Each model is run three times,
and the reported values are the averages of accuracy and F1 score across the runs.
The rate of consistency is defined as the proportion of papers that a model assigns to
the same category in all runs (i.e., no category changes across the three executions).
As a result, both Gemini Flash 2.0 and DeepSeek R1 outperform in terms of accuracy
7

and F1 score. However, Gemini Flash 2.0 shows a 5.9% higher self-consistency rate,
which defined as the percentage of identical responses produced by the LLM across
three runs. Therefore, we use Gemini Flash 2.0 to categorize both samples and the
entire dataset for all dimensions2.
We also compare the results obtained without human-in-the-loop instructions for
classification (see Table 2). Across all models, we observe a decline in accuracy, F1
score, and self-consistency. This degradation suggests that incorporating human-in-
the-loop guidance enables AI systems to better understand domain-specific academic
concepts and make more accurate and consistent classification decisions.
Table 1Performance Across Various Models with Constraints.
Model Average Accuracy Average F1 Self-consistency
GPT-4o 0.8273 0.7906 0.914
Gemini Flash 2.0 0.8327 0.8152 0.947
DeepSeek8b 0.7187 0.7090 0.672
DeepSeek V3 (API) 0.7857 0.7010 0.978
DeepSeek R1 (API) 0.8403 0.8175 0.888
Table 2Performance Across Various Models without Constraints.
Model Average Accuracy Average F1 Self-consistency
GPT-4o 0.7110 0.6471 0.782
Gemini Flash 2.0 0.7281 0.7419 0.905
DeepSeek8b 0.6750 0.6672 0.660
DeepSeek V3 (API) 0.6470 0.3418 0.899
DeepSeek R1 (API) 0.6837 0.6484 0.880
3.2.2 Option Underlying Types
In the second dimension, we classify the selected papers according to their underly-
ing asset types. We determine whether each paper focuses on one or more common
categories of underlying assets, including Stocks, Indices, Commodities, Currencies,
Interest Rates, and Cryptocurrencies. Some papers, however, primarily develop math-
ematical frameworks (e.g., stochastic differential equation (SDE) solvers) or option
pricing models (e.g., alternative assumptions for constructing SDEs) without refer-
encing or implying any specific underlying assets. These papers are categorized as Not
Specified. We allow for multi-label classification, as some studies examine multiple
types of underlying assets simultaneously. The detailed prompt design is in Appendix
A.2.
2We further validated the results via robustness checks under several constraint configurations. Gemini
Flash 2.0 provided the strongest performance across all tested settings.
8

Using the Gemini Flash 2.0 model, we conducted both consistency and accuracy
evaluations on the sample of 417 papers. As shown in Table 3, the model demon-
strates high internal reliability and moderate agreement with human annotations.
Specifically, the self-consistency across repeated AI responses reached 0.9517, indi-
cating strong stability in the model’s predictions across runs. When compared with
human labels, the mean Jaccard similarity between AI and human annotations was
0.8185, and the Lenient Accuracy, which measures whether the model correctly pre-
dicted at least one human-labeled category per sample, was notably higher as 0.8753.
This reflects that the AI responses has good performance to match the human annota-
tions. Regarding classification performance, the model achieved a micro-averaged F1
score of 0.8048 (Precision = 0.7490, Recall = 0.8697) and a sample-averaged F1 score
of 0.8329 (Precision = 0.8158, Recall = 0.8717)3. Overall, these results indicate that
the AI model exhibits excellent reproducibility and robust performance in underlying
asset classification.
Table 3Evaluation of AI Labeling in Underlying Types.
Metrics Score
Self-consistency (AI repeatability) 0.9594
AI vs. Human (Mean Jaccard similarity) 0.8185
AI vs. Human (Lenient Accuracy) 0.8753
Precision Recall F1 Score
Micro-averaged 0.7490 0.8697 0.8048
Sample-averaged 0.8158 0.8717 0.8329
3.2.3 Option Types
In the third dimension, we classify the papers by option type. We group papers into
topics such as European, American, and Exotic. As with the second dimension, papers
that do not mention or imply any option type are labeled as “Not Specified” . Our
classification relies on text mapping to assign papers.
It is noteworthy that some papers do not explicitly mention Exotic options but
discuss related concepts such as Asian, Barrier, or Basket options, et al. We take all
3TheMicro-F1score is defined as:
Micro-F1 =2×TP
2×TP+FP+FN
which computes precision and recall globally by counting all true positivesTP, false positivesFP, and
false negativesFNaggregated across all classes.
TheSample-F1score, in contrast, is calculated for each sample and then averaged:
Sample-F1 =1
NN/summationdisplay
i=12×|y i∩ˆyi|
|yi|+|ˆy i|
whereNis the total number of samples,y idenotes the true label set for samplei, ˆy ithe predicted label
set, and|y i∩ˆyi|is the number of correctly predicted labels for samplei.
9

such option types into account during the text-mapping process and classify these
papers as Exotic.
3.2.4 Option Model Types
In the fourth dimension, we classify the papers based on the type of option pric-
ing model described in their abstracts. The classification framework was developed
through extensive consultation involving AI-assisted analyses and expert domain dis-
cussions, resulting in eight primary categories: (1) Analytical Models, (2) Numerical
Methods, (3) Multi-Factor and Hybrid Models, (4) Market Imperfections and Fric-
tions, (5) Calibration and Model Estimation, (6) Machine Learning and Data-Driven
Approaches, (7) Behavioral and Alternative Paradigms, (8) Emerging and Niche
Approaches or Others. Each category encompasses more fine-grained subclasses that
capture specific modeling nuances within the broader conceptual framework. The
detailed prompt design and classification procedure are documented in Appendix A.3 .
Given that some papers contribute to multiple modeling perspectives simultaneously,
the classification adopts a multi-label scheme, ensuring that overlapping or integrative
methodologies are appropriately represented.
Similarly, we also evaluate the reliability and validity of the fourth-dimension
classification on the sample. Table 4 shows that the Gemini Flash 2.0 achieves a self-
consistency score of 0.9472, indicating highly stable predictions across repeated runs.
When compared with human annotations, the Lenient Accuracy is 0.6739 but the
mean Jaccard similarity was 0.5545, suggesting a moderate level of agreement between
AI-generated and human-labeled classifications but difficult to perfectly match the
human annotations. In terms of classification performance, the model achieved a
micro-averaged F1 score of 0.6586 (Precision = 0.5505, Recall = 0.8196) and a sample-
averaged F1 score of 0.6426 (Precision = 0.5884, Recall = 0.7846). These results
indicate that while the model demonstrates excellent internal consistency and strong
recall performance, there remains a moderate precision gap, reflecting the inherent
complexity and overlap among option pricing model categories.
Table 4Evaluation Results of AI Labeling in Model Types.
Metrics Score
Self-consistency (AI repeatability) 0.9472
AI vs. Human (Mean Jaccard similarity) 0.5545
AI vs. Human (Lenient Accuracy) 0.8657
Precision Recall F1 Score
Micro-averaged 0.5505 0.8196 0.6586
Sample-averaged 0.5884 0.7846 0.6426
10

4 Comprehensive Application to Option Pricing
Literature
In this section, we apply the best-performing model, refined through human-in-
the-loop instruction, to the full dataset. We perform analyses, including category
classifications across four dimensions, citation network analysis, and the evolution
of option pricing literature, demonstrating the effectiveness and versatility of our
framework.
4.1 Static Analysis
Using the best-performing prompt and LLM model, we identified 5,942 of 11,916
papers (49.86%) as focusing on pricing or volatility model development and compari-
son, a proportion close to our sample estimate of 41.7%. Building on this classification,
subsequent analyses focused on the second through fourth dimensions, using these
5,942 papers as the target dataset.
The frequency distribution of underlying type combinations reveals several notable
patterns. Table 5 shows that a majority of studies do not specify the underlying asset,
addressing option pricing in general terms. Among papers with specified assets, stocks,
indices, and interest rates dominate, while currencies, commodities, and cryptocur-
rencies are far less represented. Multi-asset studies are rare, with the most common
combinations being indices + stocks and interest rates + stocks. Currencies, commodi-
ties, and cryptocurrencies appear far less frequently, likely due to their more limited
representation or narrower research coverage in the literature.
Table 5Label Frequency of Option Underlying
Single Label Freq. Multi-Label Combinations Freq.
Not Specified 3347 Indices + Stocks 261
Stocks 1144 Interest Rates + Stocks 57
Indices 405 Currencies + Stocks 30
Interest Rates 281 Currencies + Interest Rates 20
Currencies 161 Commodities + Currencies + Stocks 10
Commodities 135 Indices + Interest Rates 10
Cryptocurrencies 11
The combinations less than 10 are omitted
In addition to variations across underlying assets, the literature also exhibits
diverse modeling interests with respect to option contract types, spanning both stan-
dard and path-dependent instruments. As summarized in Table 6, European, exotic,
and American options occur with comparable frequencies, indicating balanced research
attention across these categories. A substantial proportion of papers (36.05%) are
marked as not specified, likely reflecting cases where the option type was not explicitly
mentioned in the source material.
Turning to model classifications, Table 7 summarizes the distribution of model
types in option pricing research. Given the complexity of analyzing all eight types,
11

Table 6Label Frequency of Option Type
Single Label Freq. Multi-Label Combinations Freq.
Not Specified 2142 American + European 368
European 1128 European + Exotic 296
Exotic 890 American + Exotic 195
American 815 American + European + Exotic 108
Table 7Label Occurrence of Option Model Types
Model Types Occurrence Proportion
(1) Analytical Models 3956 66.58%
(2) Numerical Methods 4038 67.96%
(3) Multi-Factor and Hybrid Models 1682 28.31%
(4) Market Imperfections and Frictions 373 6.28%
(5) Calibration and Model Estimation 213 3.58%
(6) Machine Learning and Data-Driven Approaches 1492 25.11%
(7) Behavioral and Alternative Paradigms 377 6.34%
(8) Emerging and Niche Approaches or Others 158 2.66%
we focus on the proportion of each type’s occurrence, even when multiple types
appear within a single paper. The results reveal that (1) Analytical Models and (2)
Numerical Methods overwhelmingly dominate the literature, highlighting the endur-
ing influence of classical mathematical and computational approaches that form the
methodological core of option pricing research. In contrast, (3) Multi-Factor and
Hybrid Models and (5) Calibration and Model Estimation occupy a secondary yet
significant role, reflecting growing attention to empirical validation and model inte-
gration. Meanwhile, (4) Market Imperfections and Frictions, (6) Machine Learning
and Data-Driven Approaches, and (7) Behavioral and Alternative Paradigms remain
comparatively underrepresented, suggesting that these emerging directions are still at
a developmental stage within the broader literature.
Allowing multi-label assignments in model type classification, we constructed a
chord diagram (Fig. 3) to visualize co-occurrence relationships and identify the most
frequently overlapping types. The strongest co-occurrences are observed between
(1)&(2), (1)&(3), and (1)&(5) . The linkage between (1) and (2) indicates that
analytical formulations often provide the theoretical foundation for computational
implementations. The connection of (1) and (3) reflects efforts to enrich analytical
frameworks with multi-factor structures capturing complex market dynamics, while
the association of (1) and (5) highlights the practical importance of aligning models
with empirical data. Collectively, these dominant interactions show that (1) Analytical
Models remain the conceptual core of option pricing research, frequently integrated
with (2), (3), and (5) types to achieve both theoretical rigor and empirical relevance.
12

Fig. 3Model Types Chord Diagram
4.2 Network Analysis
To further analyze the structural relationships within the collected literature, we con-
structed a citation network based on the bibliographic information of the full dataset
comprising 11,916 articles. Among these, a subset of 11810 papers contained detailed
reference information, which enabled the development of the citation network. In this
network, each node represents an individual article, and each directed edge indicates
a citation relationship from one paper to another.
Given our focus on studies specifically addressing option pricing models, and the
categorization of these articles by underlying asset types, option types, and model
types, we refined the citation network to include only option pricing papers as source
nodes (i.e., citing papers). Target nodes (i.e., cited papers) include either other option
pricing studies or works related to their applications, ensuring the network captures
the intellectual linkages and citation dynamics underlying option pricing research
within the broader financial modeling literature.
We applied several widely used network analysis algorithms—degree centrality
(Freeman 1978), PageRank (Page et al. 1999), and betweenness centrality (Freeman
1977)—to identify the most influential articles in option pricing modeling. As shown
in Table 8, degree centrality and PageRank rankings are dominated by classical foun-
dational works, notably Black and Scholes (1973), Heston (1993), and Merton (1976),
reflecting their central role in shaping the theory. In contrast, betweenness central-
ity highlights studies that bridge theoretical, numerical, and empirical strands of
research, such as Broadie and Detemple (2004); Bakshi et al. (1997); Carr and Wu
(2003), emphasizing computational and hybrid modeling approaches linking theory
with practice.
Using the labels for option type, underlying type, and model type, we constructed
category-specific citation networks and applied PageRank centrality to rank papers by
their structural importance within each category. As shown in Table 9, this ranking
successfully captures influential works across all major categories, encompassing both
classical foundational studies (e.g., Heston (1993), Longstaff and Schwartz (2001))
and more recent contributions (e.g., Hutchinson et al. (1994), Amilon (2003)). The
13

Table 8Top 10 Articles in Option Pricing Modeling Based on Different Centrality Measures.
Rank Degree Centrality PageRank Betweenness Centrality
1 Black and Scholes (1973) Black and Scholes (1973) Broadie and Detemple (2004)
2 Heston (1993) Heston (1993) Bakshi et al. (1997)
3 Merton (1976) Merton (1976) Carr and Wu (2003)
4 Cox et al. (1979) Cox et al. (1979) Broadie and Detemple (1996)
5 Hull and White (1987) Cox and Ross (1976) Bates (1996)
6 Longstaff and Schwartz (2001) Hull and White (1987) Boyle et al. (1997)
7 Bakshi et al. (1997) Harrison and Kreps (1979) Fang and Oosterlee (2009a)
8 Kou (2002) Harrison and Pliska (1981) Lord et al. (2008)
9 Bates (1996) Longstaff and Schwartz (2001) Kou and Wang (2004)
10 Carr and Madan (1999) Merton (1974) Fang and Oosterlee (2009b)
Fig. 4Cumulative Co-Occurrence of Model Type Pairs Over Time (Log Scale)
inclusion of key papers across multiple categories reflects the broad recognition and
impact of these studies within the option pricing literature (Ruf and Wang 2019;
Bates 2022; Sharma et al. 2024). These findings are consistent with previous literature
reviews in the field, which identify these works as seminal and widely cited across
theoretical, numerical, and applied research on option pricing models.
Table 9Top 5 Papers in Option Pricing Modeling under PageRank Centrality, with Labels Representing Thematic Categories.
Rank Indices as Underlying American options Model Type (3) Model Type (6)
1 Bakshi et al. (1997) Longstaff and Schwartz (2001) Heston (1993) Hutchinson et al. (1994)
2 Hull and White (1987) Heston (1993) Margrabe (1978) Yao et al. (2000)
3 Hutchinson et al. (1994) Jaillet et al. (1990) Heath et al. (1992) Tsitsiklis and Van Roy (2001)
4 Heston and Nandi (2000) Geske and Johnson (1984) Longstaff and Schwartz (2001) Garcia and Gen¸ cay (2000)
5 A¨ ıt-Sahalia and Lo (1998) Kim (1990) Ho and Lee (1986) Amilon (2003)
4.3 Topic Evolutions
To uncover the long-term evolutionary trajectory and intellectual synergies, we study
the co-occurrence frequency of model-type pairs over time. Co-occurrence is defined
as the frequency with which specific pairs of model types are studied together and Fig.
14

Fig. 5Cumulative Citations of Model Type Pairs Over Time (Log Scale)
4 shows the cumulative frequency of co-occurrence pairs over time. We can clearly see
that the co-occurrence of (1) and (2) has consistently high frequency from the 1970s
to the present. This observation aligns with the reality that these two methods are
cornerstones of option pricing research. We also observe an increasing co-occurrence
of the pairs (1)&(5), (1)&(3), (2)&(5), and (2)&(3) starting around 1995, as multi-
factor and hybrid models, and model calibration gained popularity. These approaches
blended with analytical and numerical methods. This is also consistent with the find-
ings in Section 4.2, where the paper by Heston (1993) is published and quickly becomes
influential in this domain. (5) has a similiar pattern where most high-ranked papers
are published around 1995. Another trend is the emergence of (6) as a focal point for
cross-thematic co-occurrence from the 2010s to 2025. This aligns with the data and
machine learning boom of the last decades, when many research, including in option
prices, has shifted to data-driven methodologies.
Building on the preceding network analysis of model-type co-occurrence, Fig. 5
presents the cumulative directed citation relationships among model types. The nota-
tion ‘1→2’ indicates that a paper labeled as model type (1) cites a paper labeled
as model type (2). A prominent pattern emerges in the sustained volume of citations
between (1), (2), (3) and (5). These four categories maintain relatively strong citation
relationships among one another, suggesting that modern research in this field relies
on a collaborative methodological ecosystem. Analytical models provide theoretical
foundations that numerical methods operationalize, while multi-factor/hybrid models
and calibration techniques refine these tools to reflect real-world complexity, creating
a feedback loop that reinforces methodological rigor across all four areas. From the
mid-2000s onward, we observe the emergence of (6), (7), and (8); these three cate-
gories have established themselves as both recipients and sources of citations. This
pattern is consistent with the anticipated shift in the methodology for option pricing.
5 Conclusion
In this study, we introduced LR-Robot, a supervised–augmented framework that com-
bines AI capabilities with human expertise to enhance systematic literature reviews.
15

Applied to the option pricing literature, the framework demonstrated strong per-
formance in accurately categorizing papers, mapping citation networks, identifying
high-impact studies, and tracing the historical evolution of research topics. The case
study illustrates LR-Robot’s ability to accelerate labor-intensive review processes
while maintaining interpretive accuracy, providing comprehensive insights into both
classical and emerging trends in option pricing research.
Beyond this domain, LR-Robot offers considerable potential for application in
other research fields, particularly those characterized by large, complex, and rapidly
evolving literature. By integrating structured knowledge and human-in-the-loop
evaluation, the framework can support multidimensional analyses, uncover key contri-
butions, and facilitate historical and thematic mapping in a wide range of disciplines.
Overall, LR-Robot provides a practical, flexible, and high-quality approach for AI-
assisted systematic reviews, bridging the gap between efficiency and methodological
rigor.
Future research could extend the application of LR-Robot through longitudinal
and cross-disciplinary studies to evaluate its generalizability and refine best practices
for human–AI collaboration in literature synthesis. Expanding the framework to mul-
tiple research domains would also allow the exploration of interconnections between
disciplines and the evolution of scientific knowledge across fields. Moreover, integrat-
ing LR-Robot with dynamic knowledge graphs or citation-based forecasting models
could enable the identification of emerging research frontiers and the prediction of
future thematic shifts.
AcknowledgementsWW is supported by Individual Funding Awards, School of
Engineering Mathematics and Technology, University of Bristol.
Author ContributionsAll authors conceived and designed the experiments, per-
formed the experiments, and contributed to the main manuscript as well as to the
discussion and revision of the content.
Data AvailabilityData and code will be available upon request
Declarations
Conflict of interestThe authors declare no conflict of interest.
References
Agarwal S, Sahu G, Puri A, et al (2024) Litllms, llms for literature review: Are we
there yet? arXiv preprint arXiv:241215249
A¨ ıt-Sahalia Y, Lo AW (1998) Nonparametric estimation of state-price densities
implicit in financial asset prices. The journal of finance 53(2):499–547
Allen L, Jones C, Dolby K, et al (2009) Looking for landmarks: the role of expert
review and bibliometric analysis in evaluating scientific publication outputs. PloS
16

one 4(6):e5910
Amilon H (2003) A neural network versus black–scholes: a comparison of pricing and
hedging performances. Journal of Forecasting 22(4):317–335
Baccini A, De Nicolao G (2016) Do they agree? bibliometric evaluation versus
informed peer review in the italian research assessment exercise. Scientometrics
108(3):1651–1671
Bakshi G, Cao C, Chen Z (1997) Empirical performance of alternative option pricing
models. The Journal of finance 52(5):2003–2049
Bates DS (1996) Jumps and stochastic volatility: Exchange rate processes implicit in
deutsche mark options. The Review of Financial Studies 9(1):69–107
Bates DS (2022) Empirical option pricing models. Annual review of financial
economics 14(1):369–389
Black F, Scholes M (1973) The pricing of options and corporate liabilities. Journal of
political economy 81(3):637–654
Boell SK, Cecez-Kecmanovic D (2015) On being ‘systematic’in literature reviews in
is. Journal of Information Technology 30(2):161–173
Bolanos F, Salatino A, Osborne F, et al (2024) Artificial intelligence for literature
reviews: Opportunities and challenges. Artificial Intelligence Review 57(10):259
Boyle P, Broadie M, Glasserman P (1997) Monte carlo methods for security pricing.
Journal of economic dynamics and control 21(8-9):1267–1321
Broadie M, Detemple J (1996) American option valuation: new bounds, approxi-
mations, and a comparison of existing methods. The Review of Financial Studies
9(4):1211–1250
Broadie M, Detemple JB (2004) Option pricing: Valuation models and applications.
Management Science 50(9):1145–1177. https://doi.org/10.1287/mnsc.1040.0275
Carr P, Madan D (1999) Option valuation using the fast fourier transform. Journal
of computational finance 2(4):61–73
Carr P, Wu L (2003) The finite moment log stable process and option pricing. The
journal of finance 58(2):753–777
Cox JC, Ross SA (1976) The valuation of options for alternative stochastic processes.
Journal of financial economics 3(1-2):145–166
Cox JC, Ross SA, Rubinstein M (1979) Option pricing: A simplified approach. Jour-
nal of Financial Economics 7(3):229–263. https://doi.org/https://doi.org/10.1016/
17

0304-405X(79)90015-1
Dennst¨ adt F, Zink J, Putora PM, et al (2024) Title and abstract screening for liter-
ature reviews using large language models: an exploratory study in the biomedical
domain. Systematic Reviews 13(1):158
Donthu N, Kumar S, Mukherjee D, et al (2021) How to conduct a bibliometric analysis:
An overview and guidelines. Journal of business research 133:285–296
Fang F, Oosterlee CW (2009a) A novel pricing method for european options based on
fourier-cosine series expansions. SIAM Journal on Scientific Computing 31(2):826–
848
Fang F, Oosterlee CW (2009b) Pricing early-exercise and discrete barrier options by
fourier-cosine series expansions. Numerische Mathematik 114(1):27–62
Freeman LC (1977) A set of measures of centrality based on betweenness. Sociometry
pp 35–41
Freeman LC (1978) Centrality in social networks conceptual clarification. Social
networks 1(3):215–239
Garcia R, Gen¸ cay R (2000) Pricing and hedging derivative securities with neural
networks and a homogeneity hint. Journal of Econometrics 94(1-2):93–115
Geske R, Johnson HE (1984) The american put option valued analytically. The Journal
of Finance 39(5):1511–1524
Han B, Susnjak T, Mathrani A (2024) Automating systematic literature reviews
with retrieval-augmented generation: a comprehensive overview. Applied Sciences
14(19):9103
Harrison JM, Kreps DM (1979) Martingales and arbitrage in multiperiod securities
markets. Journal of Economic theory 20(3):381–408
Harrison JM, Pliska SR (1981) Martingales and stochastic integrals in the theory of
continuous trading. Stochastic processes and their applications 11(3):215–260
Heath D, Jarrow R, Morton A (1992) Bond pricing and the term structure of interest
rates: A new methodology for contingent claims valuation. Econometrica: Journal
of the Econometric Society pp 77–105
Heston SL (1993) A closed-form solution for options with stochastic volatility with
applications to bond and currency options. The review of financial studies 6(2):327–
343
Heston SL, Nandi S (2000) A closed-form garch option valuation model. The review
of financial studies 13(3):585–625
18

Ho TS, Lee SB (1986) Term structure movements and pricing interest rate contingent
claims. the Journal of Finance 41(5):1011–1029
Hull J, White A (1987) The pricing of options on assets with stochastic volatilities.
The journal of finance 42(2):281–300
Hutchinson JM, Lo AW, Poggio T (1994) A nonparametric approach to pricing
and hedging derivative securities via learning networks. The journal of Finance
49(3):851–889
Jaillet P, Lamberton D, Lapeyre B (1990) Variational inequalities and the pricing of
american options. Acta Applicandae Mathematica 21(3):263–289
Kim IJ (1990) The analytic valuation of american options. The Review of Financial
Studies 3(4):547–572
Kou SG (2002) A jump-diffusion model for option pricing. Management science
48(8):1086–1101
Kou SG, Wang H (2004) Option pricing under a double exponential jump diffusion
model. Management science 50(9):1178–1192
Lages CR, Perez-Vega R, Kadi´ c-Maglajli´ c S, et al (2023) A systematic review and
bibliometric analysis of the dark side of customer behavior: An integrative customer
incivility framework. Journal of Business Research 161:113779
Longstaff FA, Schwartz ES (2001) Valuing american options by simulation: A simple
least-squares approach. The review of financial studies 14(1):113–147
Lord R, Fang F, Bervoets F, et al (2008) A fast and accurate fft-based method
for pricing early-exercise options under l´ evy processes. SIAM Journal on Scientific
Computing 30(4):1678–1705
Margrabe W (1978) The value of an option to exchange one asset for another. The
journal of finance 33(1):177–186
Meloni A, Angioni S, Salatino A, et al (2023) Integrating conversational agents and
knowledge graphs within the scholarly domain. Ieee Access 11:22468–22489
Merton RC (1974) On the pricing of corporate debt: The risk structure of interest
rates. The Journal of finance 29(2):449–470
Merton RC (1976) Option pricing when underlying stock returns are discontinuous.
Journal of financial economics 3(1-2):125–144
Okoli C (2015) A guide to conducting a standalone systematic literature review.
Communications of the association for information systems 37
19

Page L, Brin S, Motwani R, et al (1999) The pagerank citation ranking: Bringing
order to the web. Tech. rep., Stanford infolab
Ruf J, Wang W (2019) Neural networks for option pricing and hedging: a literature
review. arXiv preprint arXiv:191105620
Ruf J, Wang W (2020) Neural networks for option pricing and hedging: a literature
review. arXiv:1911.05620
Salatino A, Osborne F, Motta E (2022) Cso classifier 3.0: a scalable unsupervised
method for classifying documents in terms of research topics. International Journal
on Digital Libraries 23(1):91–110
Sharma A, Kumar Verma C (2025) Non-parametric insights in option pricing:
a systematic review of theory, implementation and future directions. Artificial
Intelligence Review 58(8):252. https://doi.org/10.1007/s10462-025-11249-z
Sharma P, Sharma DK, Gupta P (2024) Review of research on option pricing: A
bibliometric analysis. Qualitative Research in Financial Markets 16(1):159–182
Shi J, Gao Y, Ming L, et al (2021) A bibliometric analysis of global research output
on network meta-analysis. BMC medical informatics and decision making 21(1):144
Snyder H (2019) Literature review as a research methodology: An overview and
guidelines. Journal of Business Research 104:333–339
ˇSubelj L, Van Eck NJ, Waltman L (2016) Clustering scientific publications based
on citation relations: A systematic comparison of different methods. PloS one
11(4):e0154404
Tsertsvadze A, Chen YF, Moher D, et al (2015) How to conduct systematic reviews
more expeditiously? Systematic reviews 4(1):160
Tsitsiklis JN, Van Roy B (2001) Regression methods for pricing complex american-
style options. IEEE Transactions on Neural Networks 12(4):694–703
Webster J, Watson RT (2002) Analyzing the past to prepare for the future: Writing
a literature review. MIS Quarterly pp xiii–xxiii
Yao J, Li Y, Tan CL (2000) Option price forecasting using neural networks. Omega
28(4):455–466
20

Appendix A Prompt for classification
A.1 Prompt for option pricing model classification:
Please clarify whether the abstract discusses developing or comparing pricing models or
volatility models. I need a response that uses only the options listed below: [Yes, No]. What
is your answer? Your answer should consist solely of the item from the list and nothing
else. Your answer should also follow the constraints below:
•You should answer No if the abstract primarily focuses on the application of
option pricing, rather than the development or comparison of option pricing models
themselves.
•You should answer Yes if the abstract focuses on methods of solving the existing option
pricing or volatility model.
•You should answer No if the abstract is about real estate investment or real option.
•You should answer No if the abstract is purely about volatility and does not mention
option pricing at all.
•You should answer No if the abstract is purely about Greeks and risk management
and does not mention option pricing at all.
•You should answer No if the abstract is purely about hedging strategies and does not
mention option pricing models at all.
•You should answer No if the abstract describes a application of option pricing
principles to a non traditional financial market.
•You should answer No if the abstract is purely an empirical study testing the per-
formance of existing, well-established option pricing models, without proposing any
modifications or new solution methods.
•You should answer No if the abstract focuses on market microstructure related
to options, such as bid-ask spreads or trading volume, without discussing model
development.
•You should answer No if the abstract applies option pricing theory to model or predict
bankruptcy or credit risk, without developing or comparing new option pricing models
or solution methods.
•You should answer No if the abstract primarily focuses on comparing or developing
volatility models without a direct focus on option pricing models or their solution
methods.
•You should answer Yes if the abstract focuses on comparing different option pricing
models, even if it involves an empirical study.
•You should answer No if the provided text is a list of diverse paper topics from
a proceedings or collection, rather than a single abstract focused on developing or
comparing pricing/volatility models.
•You should answer No if the abstract focuses on developing or comparing estima-
tion methods for implied volatility surfaces, without directly developing or comparing
option pricing models.
•You should answer No if the abstract focuses on developing or analyzing numeri-
cal methods for solving PDE used in option pricing, without directly developing or
comparing option pricing models.
•You should answer No if the abstract applies option pricing theory to model or analyze
insurance products, without developing or comparing new option pricing models or
solution methods.
21

•You should answer No if the abstract applies option pricing theory to model or analyze
real options or investment opportunities, without developing or comparing new option
pricing models or solution methods.
•You should answer No if the abstract talks about cash-settled American-style options
•You should answer No if the abstract talks about energy markets
•You should answer No if the abstract talks about weather derivatives
•You should answer No if the abstract talks about employee stock options
•You should answer No if the abstract talks about vulnerable chained options
•You should answer No if the abstract contains the phrase ’The proceedings contain’
A.2 Prompt for option underlying types classification:
Task: Classify Underlying Asset Type. Classify the underlying asset type of options men-
tioned in the abstract. We have six questions for you to answer. For each question, please
respond with only ’yes’ or ’no’ and nothing else.
Q1: Does this abstract specify Stocks as underlying assets?
Q2: Does this abstract specify Indexes as underlying assets?
Q3: Does this abstract specify Commodities as underlying assets?
Q4: Does this abstract specify Currencies as underlying assets?
Q5: Does this abstract specify Interest Rates as underlying assets?
Q6: Does this abstract specify Cryptocurrencies as underlying assets?
Please merge your responses to the final output as the following format{Stocks: your
response for Q1, Indexes: your response for Q2, Commodities: your response for Q3, Cur-
rencies: your response for Q4, Interest Rates: your response for Q5, Cryptocurrencies: your
response for Q6}.
A.3 Prompt for pricing model types classification:
Task: Classify this abstract of an academic paper into the option pricing methodology
taxonomy. Please only assign up to all applicable subclass from the taxonomy. Use the exact
subclass index [1.1, ...,8.3] provided below and give me just a list in form of [subclass index;
subclass index]. The taxonomy index and toxonomy are as followings:
1.1 Analytical Models: Black-Scholes Extensions
1.2 Analytical Models: Stochastic Volatility Models
1.3 Analytical Models: Jump/Discontinuity Models
1.4 Analytical Models: Regime-Switching Models
1.5 Other Analytical Models
2.1 Numerical Methods: PDE/PIDE Solvers
2.2 Numerical Methods: Monte Carlo Simulation
2.3 Numerical Methods: Lattice/Tree Methods
2.4 Numerical Methods: Transform Methods
2.5 Other Numerical Methods
3.1 Multi-Factor and Hybrid Models: Stochastic interest rates/term structure of
interest rates
3.2 Multi-Factor and Hybrid Models: Stochastic dividends
3.3 Multi-Factor and Hybrid Models: Multi-asset correlation
3.4 Multi-Factor and Hybrid Models: Hybrid local-stochastic volatility
3.5 Other Multi-Factor and Hybrid Models
4.1 Market Imperfections and Frictions: Transaction costs
4.2 Market Imperfections and Frictions: Illiquidity/funding costs
22

4.3 Market Imperfections and Frictions: Taxes/regulation
4.4 Other Market Imperfections
5.1 Calibration and Model Estimation: Implied volatility fitting
5.2 Calibration and Model Estimation: Density recovery
5.3 Calibration and Model Estimation: Statistical calibration
5.4 Other Calibration and Model Estimation
6.1 Machine Learning and Data-Driven Approaches: Neural PDE solvers/Deep
learning for pricing prediction
6.2 Machine Learning and Data-Driven Approaches: Reinforcement Learning for
optimal exercise
6.3 Machine Learning and Data-Driven Approaches: ML for calibration
6.4 Other Machine Learning and Data-Driven Approaches
7.1 Behavioral and Alternative Paradigms: Utility-based pricing
7.2 Behavioral and Alternative Paradigms: Behavioral biases
7.3 Behavioral and Alternative Paradigms: Ambiguity aversion
7.4 Other Behavioral and Alternative Paradigms
8.1 Emerging and Niche Approaches: Quantum computing
8.2 Emerging and Niche Approaches: ESG-adjusted models
8.3 Others (cannot find in the previous class)
23