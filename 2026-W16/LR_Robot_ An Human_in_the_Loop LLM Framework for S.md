# LR-Robot: An Human-in-the-Loop LLM Framework for Systematic Literature Reviews with Applications in Financial Research

**Authors**: Wei Wei, Jin Zheng, Zining Wang, Weibin Feng

**Published**: 2026-04-16 08:53:48

**PDF URL**: [https://arxiv.org/pdf/2604.14793v1](https://arxiv.org/pdf/2604.14793v1)

## Abstract
The exponential growth of financial research has rendered traditional systematic literature reviews (SLRs) increasingly impractical, as manual screening and narrative synthesis struggle to keep pace with the scale and complexity of modern scholarship. While the existing artificial intelligence (AI) and natural language processing (NLP) approaches often often produce outputs that are efficient but contextually limited, still requiring substantial expert oversight.
  To address these challenges, we propose LR-Robot, a novel framework in which domain experts define multidimensional classification taxonomies and prompt constraints that encode conceptual boundaries, large language models (LLMs) execute scalable classification across large corpora, and systematic human-in-the-loop evaluation ensures reliability before full-dataset deployment.The framework further leverages retrieval-augmented generation (RAG) to support downstream analyses including temporal evolution tracking and label-enhanced citation networks.
  We demonstrate the framework on a corpus of 12,666 option pricing articles spanning 50 years, designing a four-dimensional taxonomy and systematically evaluating up to eleven mainstream LLMs across classification tasks of varying complexity. The results reveal the current capabilities of AI in understanding and synthesizing literature, uncover emerging trends, reveal structural research patterns, and highlight core research directions. By accelerating labor-intensive review stages while preserving interpretive accuracy, LR-Robot provides a practical, customizable, and high-quality approach for AI-assisted SLRs.

## Full Text


<!-- PDF content starts -->

LR-Robot: An Human-in-the-Loop LLM
Framework for Systematic Literature Reviews
with Applications in Financial Research
Wei Weia
Jin Zhenga
Zining Wanga
Weibin Fenga
aSchool of Engineering Mathematics and Technology, University of Bristol, Ada
Lovelace Building, Tankard’s Close, Bristol, BS8 1TW, England, United Kingdom
Corresponding author:Jin Zheng
Email: jin.zheng@bristol.ac.ukarXiv:2604.14793v1  [q-fin.CP]  16 Apr 2026

Abstract
The exponential growth of financial research has rendered traditional systematic
literature reviews (SLRs) increasingly impractical, as manual screening and nar-
rative synthesis struggle to keep pace with the scale and complexity of modern
scholarship. While the existing artificial intelligence (AI) and natural language
processing (NLP) approaches often often produce outputs that are efficient but
contextually limited, still requiring substantial expert oversight.
To address these challenges, we propose LR-Robot, a novel framework in which
domain experts define multidimensional classification taxonomies and prompt
constraints that encode conceptual boundaries, large language models (LLMs)
execute scalable classification across large corpora, and systematic human-
in-the-loop evaluation ensures reliability before full-dataset deployment.The
framework further leverages retrieval-augmented generation (RAG) to support
downstream analyses including temporal evolution tracking and label-enhanced
citation networks.
We demonstrate the framework on a corpus of 12,666 option pricing articles
spanning 50 years, designing a four-dimensional taxonomy and systematically
evaluating up to eleven mainstream LLMs across classification tasks of varying
complexity. The results reveal the current capabilities of AI in understanding
and synthesizing literature, uncover emerging trends, reveal structural research
patterns, and highlight core research directions. By accelerating labor-intensive
review stages while preserving interpretive accuracy, LR-Robot provides a prac-
tical, customizable, and high-quality approach for AI-assisted SLRs.
Keywords:Systematic Literature Review, Large Language Models, Bibliometric
Network, Knowledge Graph
2

1 Introduction
The volume of financial research has expanded exponentially, with thousands of annual
publications across asset pricing, risk management, and econometrics. While Sys-
tematic Literature Reviews (SLRs) are the cornerstone of rigorous scholarship, the
current scale of the literature has rendered traditional, expert-driven methodologies
increasingly impractical. The logical bottleneck lies in the manual nature of con-
ventional approaches: relying on individual screening and narrative synthesis often
requires months or even years to conclude (Tsertsvadze et al. 2015; Okoli 2015). This
protracted timeline inevitably compromises the completeness, timeliness, and repro-
ducibility of the findings. This challenge is particularly acute in quantitative finance,
where the research landscape is fragmented across diverse methodological traditions,
ranging from classical stochastic calculus to modern machine learning. In such an
environment, a single review must synthesize thousands of papers across dense, over-
lapping conceptual categories, a task that has effectively outpaced human cognitive
capacity.
In response to this scalability crisis, bibliometric and network-based methods,
such as citation analysis (Shi et al. 2021) and keyword co-occurrence mapping (Lages
et al. 2023), have been adopted to provide quantitative field overviews. However, a
fundamental gap remains: these methods rely on metadata as a proxy for content,
rather than analyzing the research’s actual semantic substance. Consequently, they
fail to distinguish between papers that share identical keywords but offer fundamen-
tally different scholarly contributions. For instance, a theoretical paper deriving a
novel stochastic volatility model and an empirical study testing the Heston model on
index options may both utilize keywords like “option pricing” and “stochastic volatil-
ity” . Despite this overlap, they belong to distinct research paradigms: one focuses on
analytical development while the other on empirical validation. Because metadata-
driven clustering is blind to these nuanced methodological distinctions, it remains a
“surface-level” tool that cannot achieve the deep, content-level understanding required
for meaningful academic classification.
While advances in Natural Language Processing (NLP) and Large Language
Models (LLMs) offer a potential solution for content-level analysis, significant gaps
remain for financial research. Unsupervised topic modeling frameworks such as
LDA (Blei et al. 2003) and BERTopic (Grootendorst 2022) struggle to produce a
structured classification hierarchy due to the dense thematic interconnectedness of
financial scholarship. In domains like derivative pricing, a single study often inte-
grates stochastic modeling, numerical computation, and empirical calibration, causing
unsupervised methods to collapse distinct methodologies into undifferentiated clus-
ters. Furthermore, these data-driven approaches are inherently sensitive to publication
volume. Topics that are numerically underrepresented but conceptually pivotal, such
as behavioral pricing paradigms or quantum computing approaches, are frequently
absorbed into dominant clusters or overlooked entirely. Our preliminary experiments
(Section 2.1) confirm that purely data-driven methods fail to align with expert-defined
taxonomies, which necessitates an expert-supervised framework capable of capturing
the granular distinctions that define high-level financial research.
3

Beyond these algorithmic limitations, a functional gap exists in the current eval-
uation landscape. Existing AI-assisted SLR tools (Van De Schoot et al. 2021) focus
primarily on screening, while prominent FinLLM benchmarks including FinanceBench
(Islam et al. 2023) and the Open Financial LLM Leaderboard (Lin et al. 2025) are
uniformly oriented toward practical tasks like sentiment analysis. Consequently, these
models remain decoupled from the nuanced demands of scholarly categorization, which
requires the ability to distinguish between closely related yet conceptually distinct
research contributions.
To address these gaps, we propose LR-Robot, a supervised–augmented framework
that combines expert-designed classification taxonomies with LLM-based execution
and human-in-the-loop evaluation. The core design principle is a clear division of labor:
domain experts definewhat to classifyby constructing multi-dimensional taxonomies
grounded in the research literature, LLMs executehow to classifyby processing
abstracts at scale under expert-designed prompt constraints, and human evaluation
determineshow wellthe classification performs through systematic accuracy and con-
sistency assessment. This approach is motivated by our finding that unsupervised
methods cannot produce meaningful category structures in domains with high termi-
nological overlap (detailed in Section 2.1), and by the observation that simple linguistic
constraints in prompts can substantially improve LLM classification performance in
specialized domains without requiring model fine-tuning or domain adaptation. The
framework further integrates retrieval-augmented generation (RAG) to maintain an
up-to-date knowledge base that supports downstream analyses such as citation net-
work construction and topic evolution tracking (Meloni et al. 2023; Salatino et al.
2022). We demonstrate and evaluate LR-Robot on a corpus of 12,666 option pricing
articles from the Scopus database, designing a four-dimensional classification scheme
and systematically comparing five mainstream LLMs across dimensions of varying
complexity.
The main contributions of this work are fourfold. First, we provide the first sys-
tematic evaluation of LLM capabilities on financial academic literature classification
tasks, comparing five models across four dimensions of varying complexity and filling
a gap in existing FinLLM benchmarks, which focus exclusively on practical financial
tasks and do not address academic literature understanding. Second, we character-
ize the relationship between task complexity and LLM reliability in domain-specific
classification, and demonstrate that expert-designed prompt constraints yield dispro-
portionately large gains without model fine-tuning, offering practical guidance for
deploying LLMs in specialized domains. Third, we propose and validate a content-level
semantic classification methodology that combines expert-designed taxonomies with
LLM execution, showing how this approach overcomes the limitations of unsupervised
topic modeling in domains with high terminological overlap. Fourth, we apply the
framework to produce the largest multi-dimensional mapping of the option pricing
literature to date, and demonstrate that temporal co-occurrence analyses and label-
enhanced citation networks can reveal structural research trends that are invisible to
traditional bibliometric methods.
The remainder of this paper is organized as follows. Section 2 presents the LR-
Robot framework, including the design rationale for expert-designed taxonomies and
4

the human-in-the-loop evaluation pipeline. Section 3 describes the option pricing case
study, details the four classification dimensions, and reports the systematic LLM
evaluation results based on the selected sample. Section 4 applies the framework
to the entire option pricing literature, presenting classification distributions, tem-
poral evolution of research themes, and label-enhanced citation network analyses.
Section 5 discusses the implications, limitations, broader applicability of the findings
and concludes with a summary and directions for future research.
2 Framework: LR-Robot
2.1 Design Rationale: Why Expert-Designed Taxonomy?
A natural starting point for classifying a large body of academic literature is unsu-
pervised topic modeling, which can automatically discover latent thematic structures
without manual annotation. Methods such as Latent Dirichlet Allocation (LDA)
(Blei et al. 2003) and BERTopic (Grootendorst 2022) have been successfully applied
in domains where research sub-areas exhibit relatively distinct vocabularies. For
instance, Xue et al. (2026) demonstrated that BERTopic can produce coherent topic
structures in the AI and computer science literature, where sub-fields such as computer
vision, natural language processing, and reinforcement learning employ sufficiently
differentiated terminology.
Topic -1
5298 Topic 0
1408Topic 1
551Topic 2
504Topic 3
436
Topic 4
341Topic 5
329Topic 6
224
Topic 7
210Topic 8
194
Topic 9
182
Topic 10
181Topic 11
180
Topic 12
155Topic 13
Topic 14Topic 15
Topic 16Topic 17
Topic 18
Topic 19
Topic 20
Topic 21
Topic 22
Topic 23Topic 24Topic 25Topic 26
Topic 27
(a) Topic sizes. Bubble area is proportional
to document count.
0 0.2 0.4 0.6 0.8 1 1.2 1.4Topic 43
Topic 29
Topic 21
Topic 8
Topic 1
Topic 40
Topic 7
Topic 45
Topic 51
Topic 33
Topic 15
Topic 18
Topic 32
Topic 44
Topic 26
Topic 3
Topic 37
Topic 12
Topic 6
Topic 52
Topic 49
Topic 24
Topic 48
Topic 25
Topic 38
Topic 19
Topic 16
Topic 10
Topic 9
Topic 22
Topic 34
Topic 20
Topic 53
Topic 41
Topic 46
Topic 31
Topic 27
Topic 13
Topic 47
Topic 54
Topic 11
Topic 4
Topic 5
Topic 50
Topic 23
Topic 39
Topic 28
Topic 35
Topic 14
Topic 30
Topic 42
Topic 17
Topic 0
Topic 2
Topic 36Hierarchical Clustering(b) Hierarchical clustering of BERTopic-
derived topics
Fig. 1: Overview of BERTopic results. Left: topic sizes measured by document count.
Right: hierarchical clustering illustrating semantic relationships among topics.
5

Motivated by these advantages, we adopt BERTopic as a baseline and apply
it to the option pricing literature following the standard modeling pipeline, which
consists of document embedding, dimensionality reduction via Uniform Manifold
Approximation and Projection (UMAP), density-based clustering using Hierarchical
Density-Based Spatial Clustering of Applications with Noise (HDBSCAN), and topic
representation using class-based term frequency–inverse document frequency (c-TF-
IDF), with hyperparameter tuning applied to key components. Comprehensive results,
including topic outputs, hyperparameter settings, and evaluation metrics, are provided
in Appendix A.
However, our preliminary experiments with BERTopic on the option pricing corpus
reveal several fundamental limitations. As shown in Fig. 1(a), of the 12,666 abstracts
processed, BERTopic identified 54 topics1but assigned nearly half of all papers (5,298)
to the outlier category (-1), indicating that the model was unable to place a sub-
stantial portion of the literature into any coherent thematic cluster. To examine the
relationships among topics, we perform hierarchical clustering on the topic embed-
dings obtained from BERTopic using cosine distance and an agglomerative clustering
procedure, as shown in Fig. 1(b).
The resulting hierarchical structure exhibits two characteristic failure modes:
fragmentation and conflation. Monte Carlo simulation, a single core numerical
methodology, is fragmented across three separate topics (Topics 29, 21, and 40) merely
because the papers apply it to different option types (American options, general
finance, and barrier options respectively), while barrier option pricing is similarly
scattered across Topics 40, 7, and 45. At the same time, fundamentally different
computational paradigms are conflated into the same branch: Topics 43 (Quantum
Algorithms), 8 (GPU-Accelerated Algorithms), 1 (Neural Networks), and 29 (Monte
Carlo) are clustered together despite representing distinct methodological approaches.
These patterns arise because BERTopic clusters on lexical co-occurrence rather than
methodological substance, grouping papers that share similar terminology regardless
of their actual research contribution. This mismatch is not a deficiency of the algo-
rithm itself, but a structural incompatibility between unsupervised topic modeling
and the characteristics of this domain, where distinctions between research areas are
conceptual rather than lexical.
This analysis motivates the core design choice of LR-Robot. Instead of relying on
data-driven topic discovery, we adopt anexpert-designed taxonomy, in which domain
experts explicitly define the classification schema, while large language models (LLMs)
are used to perform scalable classification. In this framework, the primary role of the
expert is not to manually label instances, but to delineate the conceptual boundaries
that define meaningful categories—boundaries that cannot be reliably inferred from
the data alone.
2.2 Framework Overview
LR-Robot is developed to address the challenges of financial academic literature analy-
sis. While designed and validated in the financial domain, the underlying methodology
1The topic indices, keyword representations, and corresponding descriptions are reported in Table A2.
6

can be adapted to other research fields facing similar challenges of high terminological
overlap and complex multi-dimensional classification.
The framework comprises three architectural layers (Fig. 2): Data Retrieval,
Human-in-the-Loop Processing, and RAG Knowledge Base Construction. Together,
these layers define theFramework Development phase, a structured process in which
domain experts collaborate with LLMs to build a classified, queryable knowledge base
for a target research field.
Once development is complete, the system supports anApplication phase, which
supports the continuous ingestion of newly published articles, information retrieval,
and downstream analyses such as citation network construction and topic evolution
analysis. The primary contribution of the framework lies in the design of the develop-
ment process itself, while the application phase illustrates how the resulting knowledge
base can be used. A concrete demonstration is provided in Section 4.
Fig. 2: Overview of the LR-Robot framework. The three-layer development process
(Data Retrieval, Human-in-the-Loop Processing, and RAG Knowledge Base Construc-
tion) is shown in the upper portion, with the iterative evaluation loop within Layer 2
indicated by dashed arrows. The lower portion illustrates the application phase, in
which new articles are continuously ingested and classified, and users retrieve infor-
mation from the RAG knowledge base.
7

2.3 Framework Development Phase
2.3.1 Layer 1: Data Retrieval
In the first layer, domain experts collaborate with an LLM to formulate search
queries that specify relevant topics, keywords, and inclusion criteria. These queries
are then executed across major bibliographic databases (e.g., Scopus, Web of Science)
to retrieve relevant records. Depending on data availability and access conditions,
the retrieved content may consist of either full-text articles or structured metadata,
including titles, abstracts, author information, publication years, and reference lists.
All retrieved records are stored in a structured database, which serves as the input
corpus for subsequent classification and analysis. The choice of text granularity has
implications for downstream classification. While abstracts are often sufficient for cap-
turing high-level attributes, access to full-text content can provide additional detail
that may improve the identification of more nuanced methodological distinctions.
2.3.2 Layer 2: Human-in-the-Loop Processing
The second layer is the methodological core of the framework. It encompasses taxon-
omy design, prompt engineering, and model selection, all embedded within an iterative
cycle of expert-guided design, sample-based evaluation, and systematic refinement.
Importantly, evaluation is not treated as a discrete, terminal step, but as an integral
component of the entire process. At each iteration, domain experts assess LLM out-
puts on a representative, manually annotated sample and use the results to refine the
taxonomy, adjust prompt specifications, and compare candidate models.
Taxonomy Design and Prompt Engineering.Domain experts define classifi-
cation dimensions and their associated categories, grounded in existing literature and
disciplinary conventions. In the option pricing domain, for instance, we define four
dimensions (pricing model relevance, underlying asset type, option type, and model
type) drawing on established survey frameworks (Sharma and Kumar Verma 2025;
Ruf and Wang 2020; Broadie and Detemple 2004). Experts then design prompt con-
straints that encode domain-specific rules to address boundary cases where LLMs are
likely to misclassify (see Appendix B for complete prompt designs). This approach
encodes expert knowledge without requiring model fine-tuning.
Sample-Based Evaluation and Iterative Refinement.Because LLM classifi-
cation performance varies across domains, taxonomies, and prompt configurations,
systematic evaluation on a manually labeled representative sample is necessary to
validate reliability before full-dataset application. The evaluation results inform tax-
onomy adjustment, prompt constraint refinement, and model selection simultaneously.
This iterative loop between design and evaluation continues until classification qual-
ity meets predefined standards. Section 3 details the specific evaluation metrics and
procedures applied in our option pricing case study.
2.3.3 Layer 3: RAG Knowledge Base Construction
Once the best-performing model and prompt configuration have been identified,
the model generates classification outputs across the entire dataset. These outputs,
8

together with original bibliometric metadata, abstracts, and reference lists, are stored
in a structured retrieval-augmented generation (RAG) knowledge base (Meloni et al.
2023; Salatino et al. 2022). This design provides factual grounding by enabling
retrieval of classified records rather than reliance on parametric memory, supports
structured downstream analysis through multi-dimensional labels attached to each
article (enabling label-enhanced citation networks, co-occurrence mapping, and tem-
poral evolution tracking), and ensures transparency and reproducibility by persistently
storing all outputs for audit and re-evaluation.
2.4 Application Phase
Once the framework development is complete, the resulting RAG knowledge base
serves as a living infrastructure for ongoing research support. Newly published articles
matching the predefined search criteria are retrieved on a regular basis, automatically
classified using the validated model and prompt configuration, and appended to the
knowledge base without further expert intervention. As the classified corpus grows,
it supports a range of downstream uses, from targeted literature retrieval by classifi-
cation dimension to structural analyses such as label-enhanced citation networks and
temporal co-occurrence tracking. Section 4 demonstrates these capabilities through
an application to the option pricing literature.
3 Framework Development: An Option Pricing Case
Study
We apply the three-layer development process of LR-Robot (Section 2) to the litera-
ture on option pricing models. Option pricing provides a particularly suitable testbed:
the field has accumulated a large volume of publications over decades, its core con-
cepts (e.g., model derivation, numerical solution, and calibration) exhibit substantial
terminological overlap that challenges automated classification, and extensive domain
expertise is available to validate AI-generated outputs. Our review focuses specifi-
cally on papers that develop or compare pricing and volatility models, rather than the
broader option pricing literature that includes empirical studies, risk management,
and market microstructure. Through systematic evaluation of multiple LLMs across
classification tasks of varying complexity, this section establishes, to the best of our
knowledge, the first reference benchmarks for LLM accuracy and consistency in finan-
cial academic literature classification, providing empirical guidance on where AI can
be reliably deployed and where human oversight remains essential.
3.1 Data Overview
We collect bibliometric data from the Scopus database, encompassing all English-
language research articles, conference papers, and review articles related to option
pricing published up to March 5, 2026.2The search query combines general terms
2The data are retrieved using the following query: TITLE-ABS-KEY (”pric* of option*” OR ”option
pric*” OR ”implied volatility” OR ”European option*” OR ”American option*” OR ”stock option*” OR
”crypto* option*” OR ”interest option*” OR ”interest rate option*” OR ”exotic option*” OR ”basket
option*” OR ”barrier option*” OR ”binary option*” OR ”Bermuda option*” OR ”compound option*”) AND
9

(e.g., “option pricing”, “implied volatility”) with specific contract types (e.g., “barrier
option”, “Bermuda option”) to ensure broad coverage of both mainstream and niche
research areas. The retrieved metadata include article titles, author names, publication
years, source titles, abstracts, and reference lists.
The initial search yields 16,174 records. After removing entries with missing or
incomplete metadata (primarily absent abstracts, which are required for LLM-based
classification), 12,666 valid articles are retained. These articles span publications from
the early 1970s to 2026, reflecting over five decades of active research. As shown in
Fig. 4(a), annual output has grown from fewer than 50 papers in the early 1990s to
over 700 by 2024–2025, consistent with the broader expansion of quantitative finance
research (Donthu et al. 2021).
3.2 Multi-Dimensional Classification and Evaluation
We define four classification dimensions for analyzing the option pricing literature,
grounded in established survey works in the field (Sharma and Kumar Verma 2025; Ruf
and Wang 2020; Broadie and Detemple 2004): whether a paper develops or compares
pricing models, what underlying assets it considers, what option types it addresses,
and what modeling approaches it employs. These dimensions capture complementary
aspects of each paper’s contribution and they also span a natural gradient of classifica-
tion difficulty, from a binary decision with definable concept boundaries to multi-label
assignments over categories with substantial semantic overlap.
To evaluate LLM performance on these tasks, we randomly select 1,000 papers
from the full dataset and manually label them for Dims 1 to 4. From these, 417 papers
identified as addressing pricing or volatility model development are carried forward for
manual labeling across Dims 2 through 4. All labels are assigned by domain experts.
Each LLM is run three times on the same sample, and reported values are averages
across runs.
3.2.1 Dim 1: Option Pricing Models
The first dimension asks a binary question: does the paper develop or compare a pric-
ing or volatility model? Authors typically state their contribution in the abstract,
making this question answerable from abstract-level information. While seemingly
straightforward, this distinction requires domain knowledge to operationalize. Many
papers in the option pricing literature touch on pricing models without making model
development their central contribution, for instance, empirical studies that apply the
Black-Scholes model to test market efficiency, or risk management analyses that use
pricing outputs as inputs. To address this ambiguity, we design expert constraints
that explicitly instruct the LLM to classify such papers as negative. These constraints
cover common boundary cases including empirical applications, market microstruc-
ture, credit risk modeling, hedging strategies, real options, and energy or weather
derivatives. The complete list is provided in Appendix B.1.
(LIMIT-TO (LANGUAGE, ”English”)) AND (LIMIT-TO (DOCTYPE, ”ar”) OR LIMIT-TO (DOCTYPE,
”cp”) OR LIMIT-TO (DOCTYPE, ”re”) OR LIMIT-TO (DOCTYPE, ”cr”)).
10

We evaluate eleven LLMs across three families (GPT-4o/5/5.2, Gemini Flash
2.0/2.5/3.0, DeepSeek-8B/70B/V3/R1/V3.2), each run three times, reporting Aver-
age Accuracy, Average F1 Score, and Self-consistency.3All models are evaluated with
and without expert constraints. Table 1 presents the results.
Table 1: Dim 1 Performance with and without Expert Constraints.
Without Constraints With Constraints
Model Accuracy F1 Self-cons. Accuracy F1 Self-cons.
GPT-4o 0.7110 0.6471 0.782 0.8273 0.7906 0.914
GPT-5 0.6537 0.6379 0.939 0.8323 0.7905 0.951
GPT-5.2 0.6847 0.6545 0.901 0.8290 0.7942 0.968
Gemini Flash 2.0 0.7281 0.7419 0.905 0.8327 0.8152 0.947
Gemini Flash 2.5 0.6317 0.6829 0.899 0.7857 0.7812 0.871
Gemini Flash 3.0 0.6407 0.6944 0.959 0.8360 0.8186 0.933
DeepSeek-8B 0.6750 0.6672 0.660 0.7187 0.7090 0.672
DeepSeek-70B 0.7310 0.7102 0.684 0.7700 0.6793 0.741
DeepSeek V3 0.6470 0.3418 0.899 0.7857 0.7010 0.978
DeepSeek R1 0.6837 0.6484 0.880 0.8403 0.8175 0.888
DeepSeek V3.2 0.6603 0.5150 0.966 0.7813 0.6867 0.961
The results in Table 1 demonstrate that LLMs can classify option pricing papers at
the abstract level with considerable accuracy and stability. With expert constraints,
the best-performing models achieve F1 scores above 0.81 and self-consistency above
0.94, indicating that authors’ abstracts carry sufficient signal for distinguishing model-
development papers from those that merely apply existing models. Expert constraints
play a critical role in reaching this level of performance: all eleven models improve
when constraints are introduced, with gains ranging from moderate (Gemini Flash 2.0,
F1 from 0.7419 to 0.8152) to dramatic (DeepSeek V3, F1 from 0.3418 to 0.7010). With-
out explicit rules that delineate concept boundaries, even capable models frequently
misclassify papers that touch on pricing models without making model development
their primary contribution. Across model families, we also observe that newer gener-
ations tend to outperform their predecessors in both accuracy and consistency, which
informs our decision to focus subsequent evaluations on the latest available mod-
els. Among all candidates, Gemini Flash 3.0 achieves the strongest overall balance of
accuracy, F1, and self-consistency, and is therefore selected as the primary model for
full-dataset classification in this dimension.
3.2.2 Dim 2: Underlying Asset Types
Since our review focuses on pricing and volatility models, Dims 2 through 4 are applied
only to the 417 papers positively classified in Dim 1. The second dimension classifies
3For the binary classification in Dim 1, Accuracy is the proportion of correctly classified papers. The F1
Score is the harmonic mean of Precision and Recall: F1 = 2×Precision×Recall/(Precision+Recall), where
Precision =TP/(TP+FP) and Recall =TP/(TP+FN), withTP,FP, andFNdenoting true positives,
false positives, and false negatives respectively. Self-consistency is the proportion of papers assigned to the
same class in all three runs. All reported values are averages across three runs.
11

papers by the underlying assets they study: Stocks, Indices, Commodities, Curren-
cies, Interest Rates, Cryptocurrencies, or Not Specified (for papers developing general
frameworks without referencing specific assets). Multi-label assignment is permitted,
as some studies examine multiple asset types. These categories have clear semantic
boundaries, with asset names typically appearing explicitly in abstracts, making this
dimension well suited for abstract-level classification. The detailed prompt design is
provided in Appendix B.2. Based on the Dim 1 evaluation, which showed that newer-
generation models generally outperformed their predecessors, and given that several
earlier models had been retired or were no longer accessible via API, we focus the
remaining evaluations on five models: GPT-5, GPT-5.2, Gemini Flash 2.5, Gemini
Flash 3.0, and DeepSeek V3.2 and each model is run three times.
Since Dims 2 through 4 involve multi-label classification, the evaluation metrics
differ from those used in Dim 1. We measure classification quality against human
labels using four metrics: Mean Jaccard Similarity, Lenient Accuracy, Micro-averaged
F1, and Sample-averaged F1.4We assess self-consistency using two measures: Full
Agreement Rate, the proportion of samples receiving identical label sets across all
three runs, and Pairwise Jaccard, the average Jaccard similarity between label sets
from each pair of runs, averaged across samples.5We also include a text-mapping
baseline that assigns labels based on keyword matching in abstracts.
Table 2 presents the results across models and the text-mapping baseline. All
five LLM models achieve Sample F1 above 0.82 on this dimension, confirming that
underlying asset types can be reliably classified from abstracts alone when category
boundaries are semantically clear. Notably, all LLMs outperform the text-mapping
baseline (Sample F1 0.8273), indicating that even for relatively explicit categories,
semantic understanding provides added value beyond keyword matching. Among the
models, GPT-5 and DeepSeek V3.2 achieve the highest accuracy (Sample F1 0.8850
and 0.8845), while Gemini Flash 3.0 achieves the highest self-consistency (Pairwise
Jaccard 0.9859). Gemini Flash 2.5 is notably weaker in both accuracy (0.8222) and
consistency (0.9307).
3.2.3 Dim 3: Option Types
The third dimension classifies the same 417 papers by option type: European, Ameri-
can, Exotic, or Not Specified. Papers that do not explicitly mention exotic options but
discuss related instruments such as Asian, Barrier, or Basket options are classified as
Exotic. As with Dim 2, multi-label assignment is permitted, and we evaluate the same
five models using the accuracy and self-consistency metrics defined in Section 3.2.2.
Option type names are among the most explicit textual cues in financial abstracts,
4Mean Jaccard Similarity:1
N/summationtextN
i=1|yi∩ˆyi|/|yi∪ˆyi|, wherey iand ˆy iare the true and predicted label
sets for samplei. Lenient Accuracy: the proportion of samples for which the model correctly identifies at
least one human-labeled category. Sample-F1 computes F1 per sample and then averages:1
N/summationtextN
i=12|yi∩
ˆyi|/(|y i|+|ˆy i|). Micro-F1 aggregates true positives, false positives, and false negatives globally across all
classes: 2TP/(2TP+FP+FN). All accuracy values are reported as mean±standard deviation across
three runs.
5Pairwise Jaccard: for each sample, compute|L a∩Lb|/|La∪Lb|for all run pairs (a, b), average across
pairs, then average across samples.
12

Table 2: Dim 2: Accuracy and Self-Consistency Across Models.
Accuracy Self-consistency
Model Jaccard Len. Acc. Samp. F1 Micro F1 Full Ag. Pw. Jac.
GPT-5 0.8829 0.8897 0.8850 0.8824 0.9760 0.9856
GPT-5.2 0.8666 0.8793 0.8706 0.8663 0.9592 0.9771
Gemini Flash 2.5 0.8136 0.8401 0.8222 0.8135 0.8801 0.9307
Gemini Flash 3.0 0.8670 0.8809 0.8714 0.8649 0.9712 0.9859
DeepSeek V3.2 0.8809 0.8921 0.8845 0.8786 0.9568 0.9734
Text Mapping 0.8173 0.8489 0.8273 0.8152 — —
making this the most straightforward multi-label task in our framework and we also
include a simple text-mapping baseline for comparison.
Table 3 presents the results across models. Since option type names appear as
near-exact keywords in abstracts, we include a text-mapping baseline for comparison.
Text mapping achieves the highest accuracy on this dimension (Sample F1 0.9251),
surpassing all LLMs, which reflects the keyword-like nature of option type terminology.
However, text mapping is susceptible to false positives when a keyword appears in a
context unrelated to the paper’s focus, for example, a paper mentioning “European
option” only as a benchmark rather than as its subject. Apart from Gemini Flash
2.5, which is again the weakest (Sample F1 0.8225, Pairwise Jaccard 0.9079), the
remaining four models achieve comparable accuracy and consistency, with differences
within two percentage points.
Table 3: Dim 3: Accuracy and Self-Consistency Across Models.
Accuracy Self-consistency
Model Jaccard Len. Acc. Samp. F1 Micro F1 Full Ag. Pw. Jac.
GPT-5 0.8812 0.9241 0.8955 0.8933 0.9424 0.9742
GPT-5.2 0.8873 0.9265 0.9001 0.8990 0.9760 0.9864
Gemini Flash 2.5 0.8099 0.8489 0.8225 0.8304 0.8465 0.9079
Gemini Flash 3.0 0.8680 0.9137 0.8829 0.8826 0.9400 0.9670
DeepSeek V3.2 0.8820 0.9169 0.8933 0.8938 0.9305 0.9607
Text Mapping 0.9169 0.9424 0.9251 0.9221 — —
3.2.4 Dim 4: Option Model Types
In the fourth dimension, we classify the papers based on the type of option pric-
ing model described in their abstracts. The classification framework was developed
through extensive consultation involving AI-assisted analyses and expert domain dis-
cussions, resulting in eight primary categories: (1) Analytical Models, (2) Numerical
13

Methods, (3) Multi-Factor and Hybrid Models, (4) Market Imperfections and Fric-
tions, (5) Calibration and Model Estimation, (6) Machine Learning and Data-Driven
Approaches, (7) Behavioral and Alternative Paradigms, (8) Emerging and Niche
Approaches or Others. Each category encompasses more fine-grained subclasses that
capture specific modeling nuances within the broader conceptual framework. The
detailed prompt design and classification procedure are documented in Appendix B.3.
As with Dims 2 and 3, multi-label assignment is permitted, and we evaluate the same
five models using the metrics defined in Section 3.2.2.
In Table 4, “Direct” refers to the model’s class-level response when asked to classify
into the eight major categories directly, while “Sub→Class” derives class labels by first
classifying into the 33 subclasses and then mapping upward. The rationale is that finer-
grained prompts may force the model to engage more precisely with methodological
details in the abstract before committing to a broad category. The results confirm
this: Sub→Class consistently outperforms Direct across all models and all metrics,
with Sample F1 improving by 0.08 to 0.12 and Full Agreement Rate by 0.05 to 0.17.
We therefore adopt the finer-grained subclass-level prompt for Dim 4 when classifying
the full dataset.
Regarding model selection, both GPT-5.2 and Gemini Flash 3.0 emerge as strong
candidates: GPT-5.2 achieves the highest accuracy (Jaccard 0.5627, Micro F1 0.6688,
Full Agreement 0.8034), while Gemini Flash 3.0 achieves the highest Lenient Accuracy
(0.8449) and Pairwise Jaccard (0.9389), meaning it rarely misses relevant categories
and produces the most stable outputs across runs. DeepSeek V3.2 achieves competitive
accuracy but substantially lower consistency (Full Agreement 0.5204).
Table 4: Dim 4: Class-Level Accuracy and Self-Consistency Across Models (Direct
vs. Subclass-Derived).
Accuracy Self-consistency
Model Method Jac. Len. Sa.F1 Mi.F1 F.Ag. Pw.J.
GPT-5Direct 0.4432 0.7634 0.5340 0.5445 0.7050 0.9050
Sub→Class 0.5364 0.8114 0.6176 0.6371 0.7146 0.9119
GPT-5.2Direct 0.4418 0.7890 0.5396 0.5494 0.7194 0.9098
Sub→Class 0.5627 0.8329 0.6436 0.6688 0.8034 0.9380
Gemini Flash 2.5Direct 0.4291 0.8225 0.5364 0.5408 0.4988 0.8542
Sub→Class 0.5349 0.8377 0.6242 0.6462 0.6715 0.8996
Gemini Flash 3.0Direct 0.4183 0.8297 0.5305 0.5381 0.6043 0.8910
Sub→Class 0.5179 0.8449 0.6141 0.6360 0.7698 0.9389
DeepSeek V3.2Direct 0.4212 0.7514 0.5139 0.5212 0.3573 0.7523
Sub→Class 0.5575 0.8090 0.6327 0.6561 0.5204 0.8294
14

3.3 Error Distribution Analysis
To understand whether classification errors are driven by individual model weaknesses
or by inherent ambiguity in the papers, we visualize the per-sample, per-model error
pattern across all three multi-label dimensions. For each of the 417 papers and each
model, we record the number of runs that produce a complete misclassification (no
overlap with human labels, i.e., Lenient Accuracy = 0 for that sample. Fig. 3 displays
the results, with the bottom panels aggregating the total error count per paper across
all models.
Fig. 3: Per-sample error patterns across five models and three dimensions. Top: each
cell indicates whether a model classifies a paper correctly in all three runs (dark blue),
one run wrong (light blue), two runs wrong (orange), or all three runs wrong (red).
Bottom: total error count per paper aggregated across all models.
Across Dims 2 to 4, errors exhibit the same structural pattern: vertical red stripes
spanning all five models, indicating that misclassifications concentrate on the same
papers regardless of which model is used. This consistency across models suggests that
errors are driven by inherent ambiguity in the abstracts, such as insufficient method-
ological detail or papers that genuinely straddle multiple categories, rather than by
idiosyncratic model weaknesses. The difference across dimensions is one of frequency:
Dims 2 and 3 show only occasional red stripes, while Dim 4 shows substantially more,
consistent with the greater conceptual overlap among model-type categories.
Beyond aggregate accuracy and self-consistency, the error heatmap also informs
model selection. A model that errs primarily on universally hard papers (where all
models fail) is preferable to one that additionally produces unique errors on otherwise
easy papers. Examining the model-specific columns in Fig. 3, Gemini Flash 3.0 and
GPT-5.2 exhibit the fewest isolated errors outside the universally hard subset, sug-
gesting that their mistakes are largely confined to genuinely ambiguous cases. Gemini
Flash 3.0 shows the cleanest pattern for Dims 2 and 4, while GPT-5.2 performs best
on Dim 3. Given that Dim 4 is the most demanding task, Gemini Flash 3.0 also per-
forms strongly on Dim 2, and it maintains high accuracy and self-consistency across
all dimensions (Tables 2–4), we select Gemini Flash 3.0 as the primary model for
full-dataset classification.
15

4 Application to Option Pricing Literature
In this section, we apply the best-performing model Gemini Flash 3.0 and the prompt
refined through human-in-the-loop instruction to the full dataset. We perform analy-
ses, including category classifications across four dimensions, the evolution of option
pricing literature and citation network analysis, demonstrating the effectiveness and
versatility of our framework.
4.1 Research Landscape and Thematic Evolution
4.1.1 Temporal Distribution of Research Themes
Fig. 4 presents the temporal distribution of all four dimensions. Panel (a) shows annual
publication volume and the Dim 1 modeling share; we identified 6,766 of 12,666 papers
(53.42%) as focusing on pricing or volatility model development and comparison.
Panels (b)–(d) show occurrence rates for Dim 2–4 within these 6,766 papers. Since
Dim 2–4 allow multi-label assignments, rates can exceed 100%.
Fig. 4: Temporal distribution of the four classification dimensions. (a) Annual
publication volume and proportion of pricing/volatility model papers (Dim 1). (b)–
(d) Occurrence rates of each label within the 6,766 Dim 1-positive papers for
underlying asset type (Dim 2), option type (Dim 3), and model type (Dim 4), respec-
tively. Dim 2–4 allow multi-label assignments, so rates can exceed 100%. Years prior
to 1991 are grouped as “≤1990” .
16

Dim 1: Option pricing model.Despite the rapid growth in publication volume,
the share of pricing/volatility model papers has remained stable at approximately
50–55% throughout the entire period (Fig. 4a), indicating a persistent equilibrium
between theoretical model development and empirical or applied research in option
pricing.
Dim 2: Underlying asset types.Stocks dominated the early literature (over 40%
before 1992) but have declined steadily to roughly 17% by 2025, reflecting asset-class
diversification rather than diminished interest. Indexes maintain a stable 10–15%,
while Interest Rates and Currencies sustain modest presence. Cryptocurrencies appear
only after 2018 and remain marginal.
Dim 3: Option types.European and American options maintain stable occurrence
rates of approximately 30% and 20–30%, respectively. Exotic options show the clearest
trend, rising from under 10% in the early 1990s to approximately 25% by 2025, driven
by the proliferation of structured products in financial markets.
Dim 4: Model types.Analytical Models (1) and Numerical Methods (2) have
dominated throughout at 60–80%, though both show gradual relative decline as the
methodological landscape has diversified. The most significant shift is the rise of ML
and Data-Driven Approaches (6), from near zero before 2005 to approximately 25%
by 2025, coinciding with advances in deep learning and the increasing availability
of high-frequency financial data. Multi-Factor Models (3) have also grown steadily
to roughly 35%, reflecting the need to capture empirically observed features such as
stochastic volatility and jumps that single-factor models cannot accommodate. The
remaining categories each stay below 15%.
4.1.2 Cross-Dimensional Co-occurrence and Its Evolution
Fig. 5 presents the top-20 co-occurrence pairs for Dim 2×Dim 4 and Dim 3×Dim 4,
split by time period (before 2015 vs. 2015 and after). The percentage annotations
indicate each pair’s share of total papers in the corresponding period. Some patterns
emerge from the comparison.
Equity underlyings remain stable, interest-rate pairings decline.In
Fig. 5(a), the top five positions are occupied by Stocks and Indexes: Stocks×Analytical
(18.8%→16.6%), Indexes×Calibration (11.0%→11.8%), Indexes×Analytical
(10.9%→11.8%), Stocks×Numerical (8.8%→10.1%), and Stocks×Calibration
(7.3%→7.8%). These proportions remain largely stable across both periods, confirm-
ing that equity-linked underlyings studied through classical approaches constitute
the enduring core of the field. In contrast, all Interest Rate pairings decline: Inter-
est Rates×Multi-Factor from 7.8% to 5.2%, Interest Rates×Analytical from 6.9%
to 4.6%, and Interest Rates×Numerical from 4.2% to 3.3%, consistent with the
single-dimension trend observed in Fig. 4(b).
European and Exotic options drive growth, American options declines.
In Fig. 5(b), the four most frequent combinations all involve European or Exotic
options: European×Analytical (26.6%→30.8%), Exotic×Analytical (24.6%→28.6%),
17

020040060080010001200Pair count19%
11%11%
9%
7%8%7%
6%
4%5%4%
2%3%3%3%2%1%2%2%1%17%
12%12%
10%
8%
5%
5%6%
5%4%3%
4%3%3%2%2%3%2%2%2%Before 2015
2015 and after
(a) Underlying type × model class pairsDim2: Stocks
Dim2: Indices
Dim2: Currencies
Dim2: Interest Rates
Dim4(1):Analytical Models
Dim4(2):Numerical Methods
Dim4(3):Multi-Factor
Dim4(4):Market Imperfections
Dim4(5):Calibration
Dim4(6):Machine Learning
Dim4(7):Behavioral
Dim4(8):Emerging
025050075010001250150017502000Pair count27%25%
19%17%17%16%
10%10%
8%
5%6%
3%3%3%2%1%1%1%1%1%31%
29%
22%23%
14%14%
14%12%
11%
8%
4%
4%4%
3%2%2%2%3%2%3%Before 2015
2015 and after
(b) Option type × model class pairsDim3: European
Dim3: American
Dim3: Exotic
Dim4(1):Analytical Models
Dim4(2):Numerical Methods
Dim4(3):Multi-Factor
Dim4(4):Market Imperfections
Dim4(5):Calibration
Dim4(6):Machine Learning
Dim4(7):Behavioral
Dim4(8):Emerging
Fig. 5: Top-20 cross-dimensional co-occurrence pairs, split by time period (before
2015 vs. 2015 and after, corresponding to the most recent decade of research). Each
bar shows the absolute pair count; percentages above each segment indicate the pair’s
share of total papers in that period. (a) Underlying type×model class (Dim 2×
Dim 4). (b) Option type×model class (Dim 3×Dim 4).
Exotic×Numerical (18.5%→21.6%), and European×Numerical (16.7%→22.8%). All
four show growth in the post-2015 period, with Exotic options now rivalling European
options as a primary driver of methodological development. In contrast, American
option pairings with the same core methods decline: American×Numerical drops from
16.9% to 14.0%, and American×Analytical from 15.5% to 13.9%. This suggests that
American option pricing has reached a stage of relative methodological maturity, with
the expanding complexity of exotic products attracting a growing share of research
effort.
ML approaches demonstrate rapid single-dimension growth .ML-related
combinations appear only at the bottom of both Top-20 lists, with shares of 1–3%
in the post-2015 period: European×ML (0.8%→2.8%), American×ML (1.3%→2.4%),
Exotic×ML (0.6%→2.6%), and Indexes×ML (1.4%→2.9%). Despite ML reaching
approximately 25% as a single-dimension occurrence rate in Fig. 4(d), its cross-
dimensional shares remain low and notably uniform across option types and asset
classes. This indicates that current ML research in option pricing is predominantly
methodological, developing general-purpose algorithms rather than targeting spe-
cific products or assets. The application of ML to exotic option pricing, commodity
derivatives, and interest-rate products remains largely unexplored and represents a
significant opportunity for future research.
4.1.3 Cross-Dimensional Co-occurrence and Its Evolution
Fig. 5 presents the top-20 co-occurrence pairs for Dim 2×4 and Dim 3×4, split by
time period (before 2015 vs. 2015 and after). The percentage annotations indicate
each pair’s share of total papers in the corresponding period. Four patterns emerge
from the comparison.
18

Dominance of Analytical and Numerical pairings.The leading combinations
in both panels are dominated by pairings with Analytical Models and Numerical Meth-
ods. In Fig. 5(a), Stocks paired with Analytical and Numerical methods occupy the
top positions, followed by similar pairings with Indices and Interest Rates. In Fig. 5(b),
the six most frequent pairs are European with Analytical (27%→31%), European
with Numerical (25%→29%), American with Analytical (19%→22%), American with
Numerical (17%→23%), and Exotic with both Analytical and Numerical (each
10%→14%). Notably, the percentage shares of these dominant pairings remain largely
stable across the two time periods, confirming that Analytical Models and Numerical
Methods have served as the default methodological choices across all underlying asset
classes and option types for at least three decades. While these dominant pairings
show no sign of displacement, the emergence of new combinations in the lower-ranked
positions signals a gradual diversification of the methodological landscape.
Rapid growth of ML in isolation.Despite reaching approximately 25% occur-
rence rate as a single-dimension label (Fig. 4d), ML-related cross-dimensional combi-
nations remain sparse in the Top 20, with most appearing at only 1–3% after 2015.
This contrast indicates that current ML research in option pricing is predominantly
methodological, developing general-purpose pricing algorithms rather than target-
ing specific asset classes or option types. The application of ML-based approaches
to under-explored areas such as exotic option pricing, interest-rate derivatives, and
commodity options represents a significant opportunity for future work.
4.1.4 Global Citation Network
We constructed citation networks based on the reference information available for
12,560 of the 12,666 papers (99.2%). We first compute PageRank centrality (Page
et al. 1999) on the global network encompassing all 12,666 papers to identify the
most influential articles. To examine whether pricing model research constitutes the
intellectual core of the broader option pricing literature, we additionally constructed
a closed citation network restricted to the 6,766 Dim 1-positive papers and compared
its PageRank6with that of the global network.
As shown in Fig. 6 (left), the two Top 10 rankings share 8 out of 10 papers. The
overlapping works (Black and Scholes (1973), Heston (1993), Merton (1976), Cox et al.
(1979), Cox and Ross (1976), Merton (1974), Hull and White (1987), and Harrison and
Kreps (1979)) are foundational contributions whose influence extends equally across
modeling and non-modeling research.
Fig. 6 (right) extends this comparison beyond the Top 10 by plotting the overlap
rate between the two networks’ PageRank Top-Nlists asNincreases from 10 to 500.
A direct comparison of the global Top-Nwith the modeling Top-N(dashed line)
yields an overlap of roughly 65%. This gap partly arises because the global network
ranks all 12,666 papers, so non-modeling works can occupy Top-Npositions and
reduce the overlap mechanically. To isolate the effect of network structure from that
of set composition, we re-rank only the 6,766 pricing-model papers by their global-
network PageRank scores and compare the resulting list with the modeling Top-N
6We apply damping factor 0.85 for computing all PageRank ranking.
19

(solid line). This restricted overlap stabilises above 80%, confirming that, within the
same population of papers, the two networks produce highly consistent importance
rankings. The gap between the two curves is itself informative: it indicates that fewer
than 20% of the most influential papers in the global network are non-modeling works,
further confirming that pricing model research constitutes the intellectual core of the
option pricing field.
These results justify restricting the subsequent label-enhanced analyses to the
6,766-paper modeling network, whose citation structure faithfully reflects the intel-
lectual hierarchy of the entire field while providing the semantic labels necessary
for sub-domain analysis. However, the resulting rankings remain uniform across all
research directions. In the following subsection, we leverage the multi-dimensional
labels produced by LR-Robot to identify works that are distinctively important to
each sub-domain.
Glob. / Mod. Rank Paper
Overlapping papers (in both Top 10)
1 / 1 Black and Scholes (1973)
2 / 5 Cox et al. (1979)
3 / 3 Merton (1976)
4 / 2 Heston (1993)
5 / 4 Cox and Ross (1976)
6 / 7 Merton (1974)
7 / 6 Hull and White (1987)
8 / 8 Harrison and Kreps (1979)
In Global Top 10 only
9 / 11 Rubinstein (1994)
10 / 18 Geske (1979)
In Modeling Top 10 only
15 / 9 Broadie et al. (1997)
14 / 10 Harrison and Pliska (1981)
Fig. 6: Comparison of the global (12,666 papers) and modeling (6,766 papers) citation
networks. Left: PageRank Top 10 overlap (8/10 coverage). Right: Top-Noverlap rate.
Solid line restricts the global Top-Nto pricing-model papers; dashed line uses the
unfiltered global ranking.
4.1.5 Sub-network Divergence from Global Rankings
The modeling citation network established in Section 4.1.4 treats the modeling litera-
ture as a homogeneous body. We now leverage the multi-dimensional labels produced
by LR-Robot to construct category-specific citation sub-networks, restricting source
nodes to papers carrying a given thematic label while retaining all 6,766 mod-
eling papers as potential targets. We examine whether these sub-networks reveal
citation priorities that differ from the global consensus. All analyses in this subsec-
tion are based on the Top-200 papers by PageRank. This threshold balances two
20

considerations: it ensures adequate representation for smaller categories (e.g., Behav-
ioral (7) contributes only 2 papers to the global Top-200), while remaining selective
at approximately the top 3% of the 6,766-paper network.
Table 5 reports the overlap rate between each sub-network’s own Top-200 and
the global Top-200, i.e. the fraction of global Top-200 papers that also appear in
the sub-network’s Top-200. The results reveal a pronounced gradient, ranging from
89.5% (Analytical Models) to 29.0% (Emerging Approaches). Categories that form the
backbone of the field, namely Analytical Models (1) at 89.5%, Numerical Methods (2)
at 79.5%, and European options at 78.5%, exhibit high overlap, confirming that these
categories constitute the shared intellectual infrastructure of option pricing research.
Their citation priorities closely mirror the global consensus because nearly every sub-
field builds upon the same foundational works in these areas.
At the other end of the spectrum, Machine Learning (6) at 37.5%, Market Imper-
fections (4) at 44.0%, and Emerging Approaches (8) at 29.0% show that more than half
of these communities’ most-cited works differ from the global Top-200. This reflects
the fact that these sub-communities have distinct research agendas requiring different
bodies of knowledge, and because they constitute a small fraction of the overall liter-
ature, their internal citation priorities are diluted in the global ranking. The contrast
betweenn global andn localin Table 5 illustrates this point: Machine Learning (6) con-
tributes only 2 papers to the global Top-200 but 72 to its own sub-network Top-200;
Market Imperfections (4) rises from 6 to 73; Behavioral (7) from 16 to 70. These cat-
egories possess substantial bodies of internally important literature that are largely
absent from the global ranking. Label-enhanced sub-network analysis makes these
intrinsic patterns visible by allowing each sub-community’s citation behaviour to be
examined on its own terms, free from the dominance of larger categories.
4.1.6 Cross-Category Citation Preferences
The overlap analysis above examines whether sub-networks identify different sets of
important papers. We now ask a complementary question: holding the set of important
papers fixed, how does each sub-community re-order them? This isolates the effect of
citation preference from that of compositional change.
Concretely, we fix the global PageRank Top-200 as a common reference set and,
for each sub-networkA, re-rank these same 200 papers by sub-networkA’s PageRank
scores. Papers in the global Top-200 that receive no citations from sub-networkAare
assigned the lowest rank. For each paper categoryB, we then compute the mean rank
shift:
∆R(A,B) = ¯r global (B)−¯r A(B) (1)
where ¯r global (B) is the mean rank ofB-labeled papers in the original global ordering
and ¯r A(B) is their mean rank after re-ordering by sub-networkA. Because both
quantities are computed over the same 200 papers, a positive ∆Rpurely reflects that
sub-communityAassigns higher importance toB-papers than the global consensus
does, without conflation from changes in pool composition.
Fig. 7 presents the resulting matrix. The findings are robust under Top-100 and
Top-300 thresholds. We highlight five structural patterns. Several structural patterns
emerge.
21

Table 5: Sub-network divergence from the global citation structure
(Top-200).n global andn localare the number of papers belonging to each
category in the global and sub-network Top-200, respectively. ¯r global
and ¯r localare the mean ranks of those papers in their respective Top-
200 pools (lower = more important). ∆r= ¯r global−¯rlocal: positive
values indicate the category’s papers are ranked higher in its own sub-
network than globally. Overlap rate measures the fraction of the global
Top-200 also appearing in the sub-network’s Top-200.
Dim Categoryn global nlocal ¯rglobal ¯rlocal ∆rOverlap
2Stocks 64 83 87.0 99.7−12.7 69.0%
Indices 34 75 90.9 97.8−6.9 49.0%
Currencies 12 44 90.9 84.9 +6.1 53.0%
Interest Rates 22 57 111.2 93.5 +17.7 50.0%
3European 58 76 103.0 108.8−5.8 78.5%
American 37 99 109.1 109.2−0.2 61.0%
Exotic 36 87 121.9 109.3 +12.6 66.0%
4Analytical (1) 163 173 100.2 101.9−1.8 89.5%
Numerical (2) 93 117 110.0 109.2 +0.7 79.5%
Multi-Factor (3) 40 65 113.0 104.2 +8.8 73.0%
Mkt Imperf. (4) 6 73 124.2 99.9 +24.3 44.0%
Calibration (5) 70 122 96.4 106.3−9.9 60.0%
ML (6) 2 72 71.5 99.0−27.5 37.5%
Behavioral (7) 16 70 69.1 105.5−36.3 48.0%
Emerging (8) 0 30 — 78.3 — 29.0%
Column-level patterns: universal vs. polarising categories.The Analytical
Models (1) column exhibits near-zero ∆Rvalues across all rows (−5 to +5), confirming
that analytical works are equally valued by every sub-community regardless of research
direction. Machine Learning (6) presents the opposite pattern: its column is strongly
negative from most traditional perspectives (−94 from Analytical Models,−74 from
Stocks,−64 from Behavioral), indicating that classical sub-fields do not depend on
ML literature. Notably, American options (+24) and Market Imperfections (+17) are
exceptions, suggesting early engagement with ML through reinforcement learning for
optimal exercise and data-driven friction modelling, respectively.
Diagonal patterns: self-concentration gradient.The diagonal entries
∆R(A,A) reveal the degree to which each sub-community elevates its own litera-
ture when re-ranking the same papers. Analytical Models (+4) and Stocks (+0) show
near-zero diagonals, confirming that their internal priorities coincide with the global
consensus. Market Imperfections (+115), Machine Learning (+62), American options
(+51), and Exotic options (+45) show strongly positive diagonals, indicating that
these communities substantially re-order the shared Top-200 in favour of their own
works. Combined with the low overlap rates in Table 5, this confirms that these sub-
fields have developed distinctive citation ecosystems at both the set-composition and
rank-ordering levels.
Off-diagonal patterns: inter-community dependencies.Two notable off-
diagonal relationships emerge. Interest Rates and Multi-Factor Models (3) exhibit
22

Fig. 7: Rank concentration matrix for the global PageRank Top-200. Each cell shows
∆R(A,B), the mean rank change of categoryBpapers when the Top-200 are re-
ranked by sub-networkA’s PageRank. Positive values (blue) indicate thatB-papers
are ranked higher fromA’s perspective; negative values (red) indicate they are ranked
lower. Bottom annotations show each category’s mean global rank (µ) and paper
count (n) within the Top-200. Results are robust to Top-100 and Top-500 thresholds.
symmetric elevation: from the Interest Rates perspective, Multi-Factor papers rise by
+27 positions; from the Multi-Factor perspective, Interest Rates papers rise by +36.
This mutual dependence reflects the structural link between multi-factor stochastic
frameworks and term-structure modelling. In contrast, Behavioral (7) ranks Market
Imperfections (4) papers +56 positions higher, the second-largest off-diagonal value
in the matrix, while the reverse is only +2. This asymmetry indicates that Market
Imperfections literature is a prerequisite for Behavioral research but not vice versa.
These inter-community dynamics are structural relationships that a global ranking
cannot reveal. They become visible only when citation networks are decomposed by
the content-level labels produced through LR-Robot’s classification, without which it
23

would not be possible to partition the network by methodological type, option type,
or underlying asset.
5 Conclusion
This study introduces LR-Robot, a human-in-the-loop LLM framework for systematic
literature reviews that bridges two fundamental bottlenecks in large-scale academic
literature analysis. On one hand, traditional expert-driven reviews are increasingly
impractical given the exponential growth of publications, with manual screening and
synthesis requiring months or years to complete. On the other hand, purely data-
driven approaches, including unsupervised topic modeling and fully automated LLM
pipelines, lack the domain-specific understanding needed for reliable content-level clas-
sification, as confirmed by our BERTopic experiments. LR-Robot resolves this tension
through a clear division of labor: domain experts define classification taxonomies and
prompt constraints that encode conceptual boundaries, while LLMs execute scalable
classification under systematic human evaluation. The framework thus occupies a
methodological niche that neither manual review nor fully automated methods can
fill alone.
We validate LR-Robot on a corpus of 12,666 option pricing articles spanning
five decades by introducing a four-dimensional classification scheme and systemati-
cally benchmarking multiple LLMs across tasks of increasing complexity. The results
show that expert-designed prompt constraints significantly enhance classification per-
formance. The framework demonstrates strong reliability on tasks with well-defined
conceptual boundaries (Dims 1–3) and maintains stable, moderate performance on
more complex classifications (Dim 4), even when relying solely on abstract-level infor-
mation where category overlap is unavoidable. Error analysis indicates that most
misclassifications arise from genuinely ambiguous cases rather than systematic model
deficiencies, suggesting that performance limits are driven by the inherent difficulty of
the task. At scale, the multi-dimensional labels generated by LR-Robot enable anal-
yses that extend beyond traditional bibliometrics: temporal co-occurrence analysis
uncovers trajectories of methodological co-evolution that are not captured by tradi-
tional methods and label-enriched citation networks reveal sub-community citation
patterns.
Several directions for future work emerge from this study. The current application
relies on abstracts, which provide an efficient and scalable foundation for large-scale
classification. However, abstracts may omit important methodological details and
contextual nuances, which can limit the ability to distinguish between closely related
research approaches. Extending the input to introduction part or even full-text could
improve classification accuracy for more complex dimensions and enable a more
detailed analysis of methodological evolution. Additionally, cross-disciplinary valida-
tion in other fields characterised by high terminological overlap would help establish
the generalisability of the expert-guided approach, while integration with dynamic
knowledge graphs could support the identification of emerging research frontiers and
predictive tracking of thematic shifts.
24

AcknowledgementsThis work is funded by an Individual Funding Award from the
School of Engineering Mathematics and Technology, University of Bristol, awarded
to WW. WF’s PhD research is supported by UKRI EPSRC Doctoral Training
Partnership (EP/W524414/1).
Author ContributionsW.W. and J.Z. conceived and designed the framework,
performed the experiments, and contributed to the main manuscript as well as to
the discussion and revision of the content. Z.W. contributed to the design of the
framework and conducted the LLM experiments. W.B. conducted the topic modeling
experiments (BERTopic) and visualization.
Data AvailabilityData and code will be available upon request
Declarations
Conflict of interestThe authors declare no conflict of interest.
References
Black F, Scholes M (1973) The pricing of options and corporate liabilities. Journal of
political economy 81(3):637–654
Blei DM, Ng AY, Jordan MI (2003) Latent dirichlet allocation. Journal of machine
Learning research 3(Jan):993–1022
Broadie M, Detemple JB (2004) Option pricing: Valuation models and applications.
Management Science 50(9):1145–1177. https://doi.org/10.1287/mnsc.1040.0275
Broadie M, Glasserman P, Kou S (1997) A continuity correction for discrete barrier
options. Mathematical Finance 7(4):325–349
Cox JC, Ross SA (1976) The valuation of options for alternative stochastic processes.
Journal of financial economics 3(1-2):145–166
Cox JC, Ross SA, Rubinstein M (1979) Option pricing: A simplified approach. Jour-
nal of Financial Economics 7(3):229–263. https://doi.org/https://doi.org/10.1016/
0304-405X(79)90015-1
Dieng AB, Ruiz FJR, Blei DM (2020) Topic modeling in embedding spaces. Transac-
tions of the Association for Computational Linguistics 8:439–453. https://doi.org/
10.1162/tacl a00325, URL https://aclanthology.org/2020.tacl-1.29/
Donthu N, Kumar S, Mukherjee D, et al (2021) How to conduct a bibliometric analysis:
An overview and guidelines. Journal of business research 133:285–296
Geske R (1979) The valuation of compound options. Journal of financial economics
7(1):63–81
25

Grootendorst M (2022) Bertopic: Neural topic modeling with a class-based tf-idf
procedure. arXiv preprint arXiv:220305794
Harrison JM, Kreps DM (1979) Martingales and arbitrage in multiperiod securities
markets. Journal of Economic theory 20(3):381–408
Harrison JM, Pliska SR (1981) Martingales and stochastic integrals in the theory of
continuous trading. Stochastic processes and their applications 11(3):215–260
Heston SL (1993) A closed-form solution for options with stochastic volatility with
applications to bond and currency options. The review of financial studies 6(2):327–
343
Hull J, White A (1987) The pricing of options on assets with stochastic volatilities.
The journal of finance 42(2):281–300
Islam P, Kannappan A, Kiela D, et al (2023) Financebench: A new benchmark for
financial question answering. arXiv preprint arXiv:231111944
Lages CR, Perez-Vega R, Kadi´ c-Maglajli´ c S, et al (2023) A systematic review and
bibliometric analysis of the dark side of customer behavior: An integrative customer
incivility framework. Journal of Business Research 161:113779
Lau JH, Newman D, Baldwin T (2014) Machine reading tea leaves: Automati-
cally evaluating topic coherence and topic model quality. In: Conference of the
European Chapter of the Association for Computational Linguistics, URL https:
//api.semanticscholar.org/CorpusID:15702125
Lin SC, Tian F, Wang K, et al (2025) Open finllm leaderboard: Towards financial ai
readiness. arXiv preprint arXiv:250110963
Meloni A, Angioni S, Salatino A, et al (2023) Integrating conversational agents and
knowledge graphs within the scholarly domain. Ieee Access 11:22468–22489
Merton RC (1974) On the pricing of corporate debt: The risk structure of interest
rates. The Journal of finance 29(2):449–470
Merton RC (1976) Option pricing when underlying stock returns are discontinuous.
Journal of financial economics 3(1-2):125–144
Okoli C (2015) A guide to conducting a standalone systematic literature review.
Communications of the association for information systems 37
Page L, Brin S, Motwani R, et al (1999) The pagerank citation ranking: Bringing
order to the web. Tech. rep., Stanford infolab
Rubinstein M (1994) Implied binomial trees. The journal of finance 49(3):771–818
26

Ruf J, Wang W (2020) Neural networks for option pricing and hedging: a literature
review. arXiv:1911.05620
Salatino A, Osborne F, Motta E (2022) Cso classifier 3.0: a scalable unsupervised
method for classifying documents in terms of research topics. International Journal
on Digital Libraries 23(1):91–110
Sharma A, Kumar Verma C (2025) Non-parametric insights in option pricing:
a systematic review of theory, implementation and future directions. Artificial
Intelligence Review 58(8):252. https://doi.org/10.1007/s10462-025-11249-z
Shi J, Gao Y, Ming L, et al (2021) A bibliometric analysis of global research output
on network meta-analysis. BMC medical informatics and decision making 21(1):144
Tsertsvadze A, Chen YF, Moher D, et al (2015) How to conduct systematic reviews
more expeditiously? Systematic reviews 4(1):160
Van De Schoot R, De Bruin J, Schram R, et al (2021) An open source machine
learning framework for efficient and transparent systematic reviews. Nature machine
intelligence 3(2):125–133
Xue Z, Zhang J, Jiang J, et al (2026) Large-scale multidimensional knowledge profiling
of scientific literature. arXiv preprint arXiv:260115170
27

Appendix A BERTopic Modeling
A.1 Evaluation Metrics and Hyperparameter Tuning
To evaluate the performance of BERTopic models, we use topic quality (TQ), based on
two established metrics, namely topic coherence (TC) and topic diversity (TD) (Dieng
et al. 2020). Specifically, topic coherence is measured using normalized pointwise
mutual information (NPMI) (Lau et al. 2014), defined as
Topic Coherence =1
|T|/summationdisplay
t∈T2
N(N−1)/summationdisplay
i<jlogP(wi,wj)
P(wi)P(w j)
−logP(w i,wj),
whereTdenotes the set of topics,Nis the number of top words per topic, and
P(w i,wj) andP(w i) are the joint and marginal probabilities estimated from a
reference corpus. Topic diversity (Dieng et al. 2020) is defined as
Topic Diversity =/vextendsingle/vextendsingle/uniontext
t∈TWt/vextendsingle/vextendsingle
|T|×N,
whereW trepresents the set of top-Nwords for topict. The overall topic quality is
computed as the product of coherence and diversity, i.e.,
Topic Quality = Topic Coherence×Topic Diversity.
Using the above evaluation metrics, we adopt a random search–based strategy to
explore key hyperparameters across the main components of BERTopic, where the
optimal configuration is selected by maximizing overall topic quality. For UMAP,
the tuned parameters includen neighbors∈ {15,30,50,max nn}(where max nn =
min(n docs−1,100)),n components∈{5,8,10}, and min dist∈{0.0,0.1}. For HDB-
SCAN, min cluster size∈{10,15,20,30,40}and min samples∈{5,10,15,20}are
considered. For the vectorization module, min df∈ {1,2,3}and ngram range∈
{(1,1),(1,2)}are tuned. In each trial, a parameter configuration is randomly sampled
from the predefined search space, and the BERTopic model is fitted using precom-
puted document embeddings. A total of 24 trials were conducted, and the detailed
results are reported in Table A1.
A.2 Topic Modeling Results
Table A2 presents the topics identified by the BERTopic model, along with their
keyword representations and corresponding descriptions.
The column “Topic Keywords” lists the most representative terms for each topic,
which serve as the basis for the GPT-4.1-generated descriptions. Topic indices are
assigned by the clustering algorithm and do not imply any inherent ordering or
hierarchy.
The topic indexed as−1 corresponds to outliers, representing documents that
cannot be reliably assigned to any coherent cluster during the density-based clustering
28

Table A1: Results of 24 random search trials for BERTopic hyperparameter tuning.
The optimal configuration (Trial 8) achieves the best overall performance.
Trial #Topics n nb n comp min clust min samp TC TD TQ
0 97 15 5 15 10 0.0571 0.6505 0.0371
1 172 15 10 10 5 0.0054 0.6669 0.0036
2 45 30 10 40 10 0.1230 0.6978 0.0859
3 103 30 8 10 10 0.1320 0.7068 0.0933
4 73 50 8 15 15 0.0742 0.6658 0.0494
5 48 100 5 20 15 0.1198 0.7125 0.0853
6 50 15 8 40 15 0.1258 0.6620 0.0833
7 119 30 10 10 10 0.0256 0.6555 0.0168
8 55 30 5 20 200.13840.72000.0997
9 74 30 8 15 15 0.0765 0.6676 0.0510
10 69 30 10 15 20 0.1274 0.7232 0.0921
11 103 30 10 10 10 0.1254 0.7068 0.0886
12 64 100 8 15 15 0.1190 0.7328 0.0872
13 7 100 10 15 15 0.0768 0.7714 0.0592
14 5 50 10 40 20 0.05480.82000.0449
15 145 30 10 10 5 0.0172 0.6669 0.0115
16 40 30 10 40 5 0.1350 0.6725 0.0908
17 34 100 10 40 5 0.0992 0.6412 0.0636
18 6 50 10 10 15 0.0769 0.7833 0.0602
19 33 100 5 40 10 0.0887 0.6606 0.0586
20 67 50 10 15 15 0.0777 0.6552 0.0509
21 80 50 8 10 15 0.0673 0.6575 0.0442
22 88 15 5 10 20 0.0540 0.6705 0.0362
23 39 30 10 40 10 0.1231 0.7179 0.0884
stage. These documents typically exhibit heterogeneous content, overlapping themes,
or weak semantic signals, making them difficult to group with other topics. As a result,
the outlier category is relatively large compared to individual topics, reflecting the
diversity and complexity of the option pricing literature.
Table A2: Topics identified by the champion BERTopic model, along with their
keyword representations and GPT-4.1-generated descriptions.
Topic Index Topic Keywords GPT-4.1 Description Count
-1(Outliers)option, volatility, pric-
ing, model, options, price, prices,
risk, market, option pricing(Outliers)Option Pric-
ing under Stochastic
Volatility5298
0 stock, compensation, stock
options, firms, executive, ceo,
stock option, firm, employee,
incentivesExecutive Stock Option
Compensation1408
1 neural, learning, network, neural
network, networks, deep, neural
networks, model, data, pricingNeural Networks in
Option Pricing551
2 real, project, investment, value,
projects, real options, decision,
option, carbon, real optionReal Options in Project
Investment504
29

Topic Index Topic Keywords GPT-4.1 Description Count
3 method, finite, numerical, scheme,
order, american, difference, finite
difference, differential, methodsHigh-Order Numerical
Schemes for Options436
4 returns, market, implied, stock,
trading, options, volatility, index,
implied volatility, informationInformed Options Trading
Predictability341
5 oil, volatility, market, implied,
implied volatility, markets, indices,
uncertainty, returns, indexCrude Oil Implied
Volatility Dynamics329
6 fractional, fractional brown-
ian, brownian, rough, motion,
brownian motion, hurst, model,
volatility, stochasticFractional Brownian
Motion Option Pricing224
7 barrier, barrier options, barrier
option, double, double barrier,
options, barriers, option, pricing,
monitoredDouble Barrier Option
Pricing Models210
8 parallel, gpu, computing, imple-
mentation, performance, cpu,
architecture, hardware, pricing,
monte carloGPU-Accelerated Option
Pricing Algorithms194
9 jump, jumps, regime, switch-
ing, regime switching, model,
stochastic, markov, diffusion,
volatilityRegime-Switching Jump
Diffusion Pricing182
10 tree, lattice, binomial, trino-
mial, binomial tree, pricing, trees,
model, option, convergenceBinomial Tree Option
Pricing Methods181
11 volatility, forecasting, implied,
implied volatility, forecasts, real-
ized, forecast, index, realized
volatility, volatility forecastingImplied Volatility in
Forecasting180
12 fractional, time fractional, frac-
tional black, black, method,
order, time, numerical, fractional
derivative, black scholesTime-Fractional Black-
Scholes Methods155
13 fuzzy, numbers, option, pricing,
option pricing, model, price, fuzzy
set, european, pricing modelFuzzy Option Pricing
Models121
14 credit, default, spreads, cds, credit
default, credit risk, risk, credit
spreads, debt, default riskCredit Derivatives
Valuation and Volatility120
15 hedging, problem, transaction
costs, transaction, martingale,
superhedging, costs, optimal,
super, dualityHedging Options with
Transaction Costs119
16 rate, bond, rates, term structure,
term, model, structure, short rate,
factor, stochasticTerm Structure of Interest
Rates108
17 clinical, health, received, research,
patients, dr, fees, study, medical,
pharmaceuticalsMedical Research Funding
Disclosures108
18 perpetual, american, stopping,
optimal, perpetual american,
exercise, optimal stopping, amer-
ican options, american option,
boundaryPerpetual American
Option Stopping103
30

Topic Index Topic Keywords GPT-4.1 Description Count
19 futures, commodity, prices, con-
venience, futures prices, futures
price, convenience yield, price, oil,
modelCommodity Futures
Pricing Models95
20 density, risk neutral, neutral, den-
sities, neutral density, estimation,
nonparametric, neutral densities,
risk, option pricesNonparametric Risk-
Neutral Density
Estimation92
21 monte, monte carlo, carlo, quasi
monte, reduction, variance, qmc,
quasi, variance reduction, methodMonte Carlo Methods in
Finance91
22 garch, garch models, garch model,
models, garch option, model,
option pricing, pricing, option,
innovationsGARCH Models in
Option Pricing85
23 currency, foreign, exchange,
exchange rate, currency options,
foreign exchange, fx, rate,
currency option, ratesForeign Currency Option
Pricing Models83
24 inverse, inverse problem, regu-
larization, problem, local, local
volatility, volatility, volatility
function, ill, tikhonovLocal Volatility Surface
Calibration82
25 implied volatility, asymptotics,
asymptotic, volatility, implied,
small, sabr, maturity, large,
expansionSmall-Time Implied
Volatility Asymptotics78
26 radial, radial basis, rbf, basis,
basis function, basis functions,
method, functions, function,
numericalRadial Basis Functions for
Option Pricing78
27 uncertain, stock model, pricing
formulas, formulas, lookback,
uncertain stock, pricing, stock,
uncertain financial, optionUncertain Models for
Option Pricing77
28 insurance, life insurance, life, mor-
tality, guaranteed, linked, death,
contracts, insurance contracts,
guaranteesLife Insurance Valuation
and Risk72
29 monte carlo, monte, carlo, amer-
ican, squares, american options,
squares monte, regression, method,
simulationMonte Carlo American
Option Pricing71
30 vulnerable, default, counterparty,
vulnerable options, credit, risk,
default risk, credit risk, pricing
vulnerable, optionsVulnerable Options and
Credit Risk65
31 asian, asian options, asian option,
geometric, arithmetic, average,
geometric asian, options, price,
arithmetic asianAsian Options Pricing
Models58
32 american, boundary, exercise,
american options, exercise bound-
ary, american option, optimal
exercise, approximation, early,
early exerciseAmerican Put Option
Exercise Boundary58
33 processes, stochastic, black scholes,
scholes, martingale, black, process,
vy, equation, option pricingStochastic Processes in
Option Pricing56
31

Topic Index Topic Keywords GPT-4.1 Description Count
34 distribution, kurtosis, log, distri-
butions, gamma, model, density,
pricing, motion, option pricingVariance Gamma Option
Pricing49
35 bank, deposit, banks, deposit
insurance, capital, insurance, risk,
loan, government, loansDeposit Insurance and
Capital Regulation48
36 supply, supply chain, chain,
retailer, contract, option con-
tract, supplier, demand, financing,
procurementSupply Chain Option
Contracts41
37 wavelet, wavelets, method, wavelet
based, discretization, legendre,
fractional, equations, matrices,
schemeWavelet Methods for
Option Pricing39
38 stochastic, stochastic volatility,
volatility, heston, volatility model,
heston model, model, expansion,
multiscale, approximationMultiscale Stochastic
Volatility Option Pricing38
39 mortgage, prepayment, mortgages,
loan, house, value, rate, borrower,
house price, mortgage backedMortgage Default and
Option Pricing38
40 barrier, monte carlo, monte, carlo,
barrier options, simulation, algo-
rithm, pricing barrier, options,
methodEfficient Monte Carlo
Barrier Option Pricing38
41 basket, basket options, approxima-
tion, basket option, pricing basket,
log, moment matching, matching,
closed, closed formClosed-Form Basket
Option Pricing31
42 warrants, warrant, stock, pricing,
warrant prices, structured, market,
warrant pricing, stocks, issuedWarrant Pricing and
Valuation Models31
43 quantum, amplitude, algorithms,
algorithm, computing, circuit,
monte carlo, monte, carlo, classicalQuantum Algorithms for
Option Pricing30
44 fourier, cos, cos method, method,
cosine, fourier cosine, fourier
transform, transform, fast, cosine
seriesFourier-Cosine Option
Pricing Methods28
45 static, hedging, static hedging,
hedge, barrier, hedges, barrier
options, static hedges, options,
portfolioStatic Hedging of Barrier
Options28
46 bounds, upper, good deal, lower,
prices, bound, bounds option,
deal, upper bounds, lower boundsOption Pricing Bounds in
Incomplete Markets28
47 weather, temperature, rainfall,
weather derivatives, derivatives,
temperatures, cooling, daily,
process, degreeTemperature Derivatives
in Agriculture/Energy26
48 implied volatility, implied, volatil-
ity, formula, black, black scholes,
scholes, approximation, method,
scholes formulaImplied Volatility
Approximation Methods25
49 lie, symmetry, symmetries,
equation, group, invariant, clas-
sification, solutions, symmetry
analysis, blackLie Symmetry Analysis in
Finance25
32

Topic Index Topic Keywords GPT-4.1 Description Count
50 currency, volatility, iv, exchange,
implied, usd, implied volatility,
exchange rate, rate volatility,
monthCurrency Options
Volatility Modeling23
51 reflected, bsdes, backward
stochastic, stochastic differen-
tial, backward, reflected bsdes,
stochastic, differential, uniqueness,
solutionReflected G-BSDEs and
Applications23
52 wave, solutions, ivancevic, soliton,
ivancevic option, nonlinear, pricing
model, rogue, dark, option pricingIvancevic Option Pricing
Waves22
53 copula, copulas, dependence,
bivariate, spread, dynamic, spread
option, gaussian, model, pricingDynamic Copula Bivariate
Option Pricing21
54 volatility, implied, index, implied
volatility, 500, models, model,
stochastic, volatility function, dataStochastic Volatility in
Option Pricing21
Appendix B Prompt for classification
B.1 Prompt for option pricing model classification:
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
33

•You should answer No if the abstract primarily focuses on comparing or developing
volatility models without a direct focus on option pricing models or their solution
methods.
•You should answer Yes if the abstract focuses on comparing different option pricing
models, even if it involves an empirical study.
•You should answer No if the provided text is a list of diverse paper topics from a pro-
ceedings or collection, rather than a single abstract focused on developing or comparing
pricing/volatility models.
•You should answer No if the abstract focuses on developing or comparing estima-
tion methods for implied volatility surfaces, without directly developing or comparing
option pricing models.
•You should answer No if the abstract focuses on developing or analyzing numeri-
cal methods for solving PDE used in option pricing, without directly developing or
comparing option pricing models.
•You should answer No if the abstract applies option pricing theory to model or analyze
insurance products, without developing or comparing new option pricing models or
solution methods.
•You should answer No if the abstract applies option pricing theory to model or analyze
real options or investment opportunities, without developing or comparing new option
pricing models or solution methods.
•You should answer No if the abstract talks about cash-settled American-style options
•You should answer No if the abstract talks about energy markets
•You should answer No if the abstract talks about weather derivatives
•You should answer No if the abstract talks about employee stock options
•You should answer No if the abstract talks about vulnerable chained options
•You should answer No if the abstract contains the phrase ’The proceedings contain’
B.2 Prompt for option underlying types classification:
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
B.3 Prompt for pricing model types classification:
Class-Level Task: Classify this abstract of an academic paper into the option pricing
methodology taxonomy. Please only assign up to all applicable class from the taxonomy. Use
the exact subclass index 1-8 provided below and give me just a list in form of [class index;
class index].
Taxonomy index and Toxonomy name:
34

1 Analytical Models
2 Numerical Methods
3 Multi-Factor and Hybrid Models
4 Market Imperfections and Frictions
5 Calibration and Model Estimation
6 Machine Learning and Data-Driven Approaches
7 Behavioral and Alternative Paradigms
8 Emerging and Niche Approaches or Others(cannot find in the previous class)
Subclass-Level Task: Classify this abstract of an academic paper into the option pricing
methodology taxonomy. Please only assign up to all applicable subclass from the taxonomy.
Use the exact subclass index [1.1, ...,8.3] provided below and give me just a list in form of
[subclass index; subclass index]. The taxonomy index and toxonomy are as followings:
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
35