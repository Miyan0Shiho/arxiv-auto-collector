# AnnoABSA: A Web-Based Annotation Tool for Aspect-Based Sentiment Analysis with Retrieval-Augmented Suggestions

**Authors**: Nils Constantin Hellwig, Jakob Fehle, Udo Kruschwitz, Christian Wolff

**Published**: 2026-03-02 11:56:47

**PDF URL**: [https://arxiv.org/pdf/2603.01773v1](https://arxiv.org/pdf/2603.01773v1)

## Abstract
We introduce AnnoABSA, the first web-based annotation tool to support the full spectrum of Aspect-Based Sentiment Analysis (ABSA) tasks. The tool is highly customizable, enabling flexible configuration of sentiment elements and task-specific requirements. Alongside manual annotation, AnnoABSA provides optional Large Language Model (LLM)-based retrieval-augmented generation (RAG) suggestions that offer context-aware assistance in a human-in-the-loop approach, keeping the human annotator in control. To improve prediction quality over time, the system retrieves the ten most similar examples that are already annotated and adds them as few-shot examples in the prompt, ensuring that suggestions become increasingly accurate as the annotation process progresses. Released as open-source software under the MIT License, AnnoABSA is freely accessible and easily extendable for research and practical applications.

## Full Text


<!-- PDF content starts -->

AnnoABSA: A Web-Based Annotation Tool for Aspect-Based
Sentiment Analysis with Retrieval-Augmented Suggestions
Nils Constantin Hellwig1, Jakob Fehle1, Udo Kruschwitz2, Christian Wolff1
1Media Informatics Group, University of Regensburg, Regensburg, Germany
2Information Science Group, University of Regensburg, Regensburg, Germany
nils-constantin.hellwig@ur.de, jakob.fehle@ur.de, udo.kruschwitz@ur.de, christian.wolff@ur.de
Abstract
We introduce AnnoABSA, the first web-based annotation tool to support the full spectrum of Aspect-Based Sentiment
Analysis (ABSA) tasks. The tool is highly customizable, enabling flexible configuration of sentiment elements
and task-specific requirements. Alongside manual annotation, AnnoABSA provides optional Large Language
Model (LLM)-based retrieval-augmented generation (RAG) suggestions that offer context-aware assistance in a
human-in-the-loop approach, keeping the human annotator in control. To improve prediction quality over time, the
system retrieves the ten most similar examples that are already annotated and adds them as few-shot examples
in the prompt, ensuring that suggestions become increasingly accurate as the annotation process progresses.
Released as open-source software under the MIT License, AnnoABSA is freely accessible and easily extendable for
research and practical applications.
Keywords:Annotation Tool, Aspect-Based Sentiment Analysis, Retrieval-Augmented Generation, Large
Language Models, AI-Assistance
1. Introduction
Aspect-Based Sentiment Analysis (ABSA) consti-
tutes a fine-grained approach to sentiment analysis
that goes beyond document-level polarity classifi-
cation by identifying specific aspects within a text
and determining the sentiment orientation associ-
ated with each aspect (Pontiki et al., 2016). The
field encompasses various ABSA subtasks that
differ in their granularity of aspect identification.
These tasks involve combinations of the following
sentiment elements: aspect terma, aspect cate-
goryc, opinion termo, and sentiment polarityp.
For instance, in the sentence“The pizza was deli-
cious.”, “pizza” represents the aspect term, “food
general” could constitute the associated aspect cat-
egory, “delicious” serves as the opinion term, and
the sentiment polarity is positive. In cases where
no aspect term is given for an aspect (=implicit
aspect), the aspect term is set to “NULL ”, e.g.“It
was delicious.”. A sentence may contain several
aspects, resulting in several aspects that need to
be annotated.
Due to the granularity of ABSA, the creation
of annotated resources for training and evaluat-
ing ABSA-specific models remains highly time-
consuming and labour-intensive (Nasution and
Onan, 2024; Negi et al., 2024; Wang et al., 2024).
Resources are particularly scarce for low-resource
languages and domain-specific contexts (Xu et al.,
2025; Le Ludec et al., 2023; Fehle et al., 2025).
The scarcity of annotated datasets is also re-
flected in the limited availability of specialized an-
notation tools, with only a few dedicated solutions
currently existing. To date, only general-purposeannotation frameworks such asINCEpTION1(Wo-
jatzki et al., 2017),BRAT2(Pontiki et al., 2016,
2015, 2014) andLabel Studio3(Hellwig et al.,
2024; Fehle et al., 2025) have been reported to be
used for ABSA annotation tasks. However, these
lack essential functionalities required for certain
subtasks. For example, in Target Aspect Sentiment
Detection (TASD), a text may contain numerous
implicit aspects that do not correspond to specific
textual spans but can be assigned to an aspect cat-
egory and sentiment polarity. The aforementioned
tools cannot handle such dynamic lists of entries,
in this case aspect annotations of implicit opinions.
Given the challenges associated with manual
data annotation, recent research has increasingly
explored Large Language Models (LLMs) to min-
imize annotation effort across various NLP do-
mains, including social media analysis (Hasan
et al., 2024; Mu et al., 2024; Zhang et al., 2024),
(bio-)medicine (Labrak et al., 2024; Ateia and Kr-
uschwitz, 2023), and finance (Deußer et al., 2024;
Deng et al., 2024). For ABSA, few-shot learn-
ing has demonstrated competitive performance in
ABSA tasks, achieving micro-averaged F1 scores
approaching those of fine-tuned models while re-
quiring only a few examples (Hellwig et al., 2025;
Zhou et al., 2024). However, these approaches
still fall short of the performance levels reported
for models specifically fine-tuned for ABSA tasks
(Hellwig et al., 2025; Zhang et al., 2024; Zhou et al.,
2024).
1INCEpTION (Klie et al., 2018): https://inceptio
n-project.github.io/
2BRAT:https://github.com/nlplab/brat
3Label Studio:https://labelstud.io/arXiv:2603.01773v1  [cs.CL]  2 Mar 2026

In this work, we introduceAnnoABSA, a web-
based annotation tool designed for ABSA. The tool
provides extensive customization capabilities and
supports all major ABSA subtasks documented in
the literature (Zhang et al., 2022), including aspect
term extraction, aspect category classification, sen-
timent polarity detection, opinion term identifica-
tion, and their various combinations (pairs, triplets,
quadruples).
Following recent studies in other time-intensive
annotation tasks (Ghazouali and Michelucci, 2025;
Kim et al., 2024; Sahitaj et al., 2025; Li et al., 2025),
we aimed to integrate the capabilities of founda-
tional models to assist annotators in the annota-
tion process. Beyond traditional manual annota-
tion functionality, AnnoABSA optionally incorpo-
rates a Retrieval Augmented Generation (RAG)-
based suggestion mechanism that combines the
strengths of LLM-based predictions with human ex-
pertise. Our proposed RAG mechanism retrieves
the most semantically similar examples from the
pool of instances previously annotated during the
annotation process to guide the LLM in providing
suggestions. This hybrid approach leverages the
efficiency and consistency of large language mod-
els while preserving the nuanced judgment and
domain knowledge of human annotators, thereby
balancing annotation speed with quality.
Our main contributions are as follows:
•We present the first open-source annotation
tool for ABSA with comprehensive compati-
bility across all ABSA subtasks and release
it under the permissive MIT licence at https:
//github.com/NilsHellwig/AnnoABSA.
•We introduce a retrieval-augmented LLM-
based suggestion mechanism that leverages
the most similar annotated examples to en-
hance annotation efficiency while improving
suggestion quality over time.
•We demonstrate through systematic evalua-
tion that RAG-based suggestions significantly
outperform random sampling baselines in
terms of prediction performance.
•We provide evidence from a controlled study
showing that expert annotators achieve a sta-
tistically significant reduction in annotation
time (30.51%) when assisted by RAG-based
suggestions compared to unassisted manual
annotation.
2. System Description
2.1. Motivation
We present AnnoABSA, a novel annotation tool
designed to address the need to support all ABSAtasks (see Appendix A) while providing an accessi-
ble and intuitive user interface (UI) that facilitates
efficient annotation with minimal user interaction.
In this section, we detail the system architecture,
UI design decisions, and AnnoABSA’s data man-
agement. Additionally, Table 1 characterizes the
core features of AnnoABSA and compares them
with existing annotation tools previously utilized for
ABSA tasks.
2.2. System Architecture
AnnoABSA is implemented usingReact.js4, a fron-
tend framework specifically designed for Single-
Page Applications (SPAs). We utilizedTypeScript5
as the primary programming language for the
frontend, a statically-typed superset of JavaScript
that enhances code robustness and reduces error
susceptibility. The backend employsFastAPI6, a
Python RESTful API framework that enables the
integration of essential Python packages such as
Pandas7andNumPy8for efficient data processing
and manipulation.
AnnoABSA can be started through a command-
line interface by providing the text to be annotated
in either JSON or CSV format, along with a config-
uration file specifying the annotation parameters.
$ ./annoabsa reviews.json –load-config
config.json
The system offers extensive customization op-
tions that can be specified in a configuration file or
through CLI flags. These configuration parameters
include the input text file for annotation, the sen-
timent elements to be considered (aspect terms,
aspect categories, opinion terms, sentiment polar-
ity), a list of valid sentiment polarities and aspect
categories, boolean options for specifying if im-
plicit aspect terms and/or opinion terms are valid.
Hence, AnnoABSA supports the annotation of text
data in any language. A comprehensive overview
of all supported flags is provided in Appendix B.
After executing the CLI tool, the AnnoABSA UI is
opened in the web browser.
2.3. UI Design
The UI is presented in Figure 1. The interface
components were designed for comfortable use
on both desktop computers and tablets. Following
minimalist design principles (Sani and Shokooh,
4React.js:https://reactjs.org/
5TypeScript:https://www.typescriptlang.org/
6FastAPI:https://fastapi.tiangolo.com/
7Pandas:https://pandas.pydata.org/
8NumPy:https://numpy.org/

Feature Label Studio INCEpTION BRAT AnnoABSA
Technical requirements
Open source√(Apache 2.0)√(Apache 2.0)√(MIT)√(MIT)
Installation Docker, Python 3 Docker, Tomcat, Java Python 2 Python 3
Regular updates√ √×Oct. 2021√
Web-based√ √ √ √
Multi-user support√ √×√
User interface & usability
Annotation setup XML template Project setup Project setup JSON config/CLI
Interface design Highly customizable Technical, functional Technical, functional Modern, intuitive
(XML template)
Language translation√× ×√
Annotation guidelines√Configurable × ×√Integrated in the
with XML interface as PDF
Documentation√ √ √ √
Support Community forum, GitHub issues GitHub issues GitHub issues,
GitHub issues (inactive developers) email
Data management
Import/Export formats JSON, TXT, XML; UIMA, TSV, JSON, TXT TXT, ANN JSON, CSV
CSV export only (BRAT stand-off)
General annotation functionalities
Multi-labelling functionality√ √ √ √
Relationship tagging√ √ √ √
Label customization√XML√Project√Config√JSON
template configuration files configuration
Team collaboration√Full support × × ×
with role management
Token identification1√ √ √ √
ABSA-specific features
Validation2× × ×√
AI-based suggestions√Integration for√Integration for ×√
individual models/APIs individual models/APIs
Dynamic lists3× × ×√
Assigning (multiple)
categories to a × × ×√
text span
Table 1:Comparison of Annotation Tools for ABSA.Detailed comparison of four annotation platforms
across technical requirements, usability features, data management capabilities, and ABSA-specific
functionalities. The table constitutes an extension of the comparison provided by Colucci Cante et al.
(2024, p. 359), who evaluated non-semantic textual annotation tools based on the following criteria:
multi-labelling functionality, annotation suggestions, relationship tagging, label customization, and team
collaboration.
1Token Identification: Automatic identification of token boundaries in text to prevent annotation errors where spans
begin or end within words.
2Validation: Label Studio, INCEpTION, and BRAT do not enforce mandatory linking of sentiment elements within
aspect tuples. For example, in the TASD task (tuples consisting of aspect term, aspect category, and sentiment
polarity), annotators may mark an aspect term but forget to set the corresponding aspect category and sentiment
polarity. AnnoABSA prevents such errors through strict validation, ensuring each tuple contains exactly the required
number of sentiment elements for a specific task. Individual sentiment elements are also validated (e.g., sentiment
polarity must be "positive", "negative", or "neutral"; aspect categories must be from predefined lists; aspect/opinion
terms must be either NULL for implicit aspects/opinions or substrings of the text that should be annotated).
3Dynamic Lists: Annotations for ABSA tasks may comprise an unlimited number of aspects. As an example, for the
ACD task (which considers pairs of aspect category and sentiment polarity), a tool should offer the functionality to
assign unlimited combinations of aspect categories and sentiment polarities to documents. Label Studio,
INCEpTION, and BRAT cannot handle unlimited assignments of categorical variable combinations to documents.
2016), we focused on the main content, avoided re-
dundant visual elements, and employed flat design
aesthetics.2.3.1. Topbar
The topbar provides a convenient navigation be-
tween examples, using arrow keys to move to the
previous or next instance. Additionally, a double-

Figure 1:UI of AnnoABSA.The UI consists of four components: (1) top navigation bar to change the
currently displayed example, (2) annotated text with highlighted sentiment annotations, (3) aspect addition
panel, and (4) editable list of added annotations.
arrow button on the right allows users to jump di-
rectly to the unannotated example with the highest
index, and an input field enables navigation to a
specific index.
2.3.2. Form to Add New Aspect
The subsequent section in the UI enables adding
new aspect annotations. Depending on the CLI
specification, varying numbers of sentiment ele-
ments to be annotated are displayed. Figure 1
shows all four elements, though configurations with
fewer sentiment elements can also be configured.
Categorical sentiment elements, aspect category
and sentiment polarity, can be selected using drop-
down menus, while aspect term and opinion term
each trigger a popup interface.
As shown in Figure 2, the popup displays the text
to be annotated twice, once for aspect term anno-
tation and once for opinion term annotation. When
only one phrase type needs to be annotated, the
text is displayed once. We considered implement-
ing a tool-switching approach (separate selectabletools for aspect term and opinion term selection),
but this would have required additional clicks. Our
chosen interface requires only phrase marking and
clicking “Done”, thus minimizing user actions to
only those absolutely necessary.
Phrase boundaries are defined by clicking on in-
dividual characters or, if configured, tokens. Once
a phrase is marked, both the text and position are
displayed for confirmation. Notably, this popup
appears both when annotating new aspects and
when editing existing ones. After all sentiment el-
ements are specified, users click “Add aspect” to
append the sentiment element to the list of applied
annotations presented below the “Add aspect” sec-
tion.
2.3.3. Annotation List
All annotated aspects are displayed in a list and
can be modified at any time. We included a du-
plication button for aspects, designed to assist in
cases where aspects differ in only one sentiment
element. For example, in the sentence“The pizza

Figure 2:Popup to add or manipulate phrase
annotations.In case both aspect term and opinion
term annotations are required, the text is displayed
twice, once for each phrase annotation. Implicit
aspects can be marked using a checkbox.
and the burger were delicious”, the Aspect Senti-
ment Quad Prediction (ASQP) gold label consists
of two quadruples: (’pizza’, ’food quality’,
’delicious’, ’positive’) and(’burger’, ’food
quality’, ’delicious’, ’positive’) . During an-
notation, one quadruple can be duplicated and the
relevant sentiment element modified accordingly.
Annotations can be deleted as needed.
2.4. Data Handling
The CSV or JSON file containing all examples to
be annotated is directly modified during the anno-
tation process, with annotations being appended
as the annotation work progresses. In addition
to the sentiment elements and phrase positions
within the given text, annotation duration can op-
tionally be stored. The use of JSON format enables
straightforward integration for NoSQL databases if
required, with minimal code modifications.
3. Retrieval Augmented Suggestions
Given the absence of annotated training examples
for supervised models in a situation where a human
annotator starts annotating text for ABSA, we adopt
an LLM-based approach for generating annota-
tion suggestions. Previous research (Hellwig et al.,
2025; Zhou et al., 2024; Zhang et al., 2024) demon-
strated that LLMs utilizing a small, fixed set of few-
shot examples achieve performance approaching
fine-tuned models, particularly in low-resource sce-narios, and, that Retrieval Augmented Generation
(RAG)-based few-shot learning achieved higher
performance scores than random sampling across
various ABSA tasks.
This section examines technical considerations
including model support and prompting techniques,
followed by a comparative evaluation of random
and RAG-based sampling approaches for annota-
tion suggestion generation.
3.1. Model Support
AnnoABSA provides flexible model integration
through appropriate CLI parameters, supporting
both commercial models via the OpenAI API9and
open-source models through the locally hosted Ol-
lama API10. These toolkits were selected based on
their platform independence and native implemen-
tation of structured output capabilities. Structured
outputs via guided decoding ensure that LLM sug-
gestions are constrained to the aspect categories
and sentiment polarities defined in the CLI config-
uration, preventing the generation of hallucinated
labels. Furthermore, structured outputs guarantee
that predicted phrases are present in the text that
needs to be annotated.
3.2. Prompt
We adopted the prompt structure employed by Hell-
wig et al. (2025) and Gou et al. (2023), which
comprises a task description, demonstrations, and
the text to be annotated. A modification of the
prompt involves formatting the demonstration’s la-
bels in JSON, similar to the aforementioned en-
forced structured output. An example of the prompt
template is provided in Appendix C.
3.3. Demonstration Selection Strategy
Evaluation
Although the RAG-based approach by Zhou et al.
(2024) which utilized thekmost semantically simi-
lar training examples as few-shot demonstrations
achieved higher performance scores than random
selection, their methodology cannot be directly
transferred to the annotation process. Their ap-
proach supposes access to a fully annotated train-
ing corpus for example selection, which contrasts
with annotation scenarios where the set of human-
annotated instances (pool) expands during the la-
belling process. To address this discrepancy, we
conducted a performance analysis between ran-
dom sampling and RAG-based sampling strategies
within an annotation framework. All LLM execu-
tions in our performance analysis were conducted
9OpenAI API:https://openai.com/api/
10Ollama:https://ollama.com/

100 200 300 400 500 600 700 800 9001000 1100788082848688F1 Score (%)
t-test, padj<0.001 *** Mdiff=2.78
83.70 83.65 83.63 83.68 83.5184.0084.54 84.4584.84 84.7084.08
81.1080.7381.64 81.69 81.4780.9981.68 81.92
81.02 81.3180.66Rest16ACD
100 200 300 400 500 600 700 800 9001000 110054565860626466
t-test, padj<0.001 *** Mdiff=4.21
60.2361.5160.8161.6362.14 62.2963.08 63.07 63.05 63.1862.39
58.6158.15 57.93 57.8558.9458.46 58.2057.49 57.21 57.17 56.99Rest16TASD
100 200 300 400 500 600 700 800 9001000 1100384042444648505254
Wilcoxon, padj=0.005 ** Mdiff=5.53
46.7446.12 46.11 46.2649.1848.6749.6950.66 50.9250.2550.85
43.7342.8342.3541.8943.6242.7443.1842.74 42.9444.7743.85Rest16ASQP
100 200 300 400 500 600 700 800 9001000 1100788082848688F1 Score (%)
Wilcoxon, padj=0.005 ** Mdiff=2.98
83.59 83.7284.56 84.45 84.60
83.7584.52 84.38 84.39 84.61 84.34
81.27 81.13 80.8981.35 81.61 81.74 82.03
80.4781.35
80.3281.92FlightABSA
100 200 300 400 500 600 700 800 9001000 110054565860626466
t-test, padj<0.001 *** Mdiff=4.50
60.9861.53 61.58 61.6262.2461.79 62.0761.4161.97 62.31 62.61
57.65
56.3556.93 57.23 57.02 57.0458.20
57.1358.62
57.39 57.05FlightABSA
100 200 300 400 500 600 700 800 9001000 110032343638404244464850
Wilcoxon, padj=0.005 ** Mdiff=7.25
43.7242.7443.3742.7746.00 45.7646.89 47.08 47.23 47.64 47.71
38.4137.46
35.9436.9439.5138.49 38.7139.80 39.54
37.6538.71FlightABSA
100 200 300 400 500 600 700 800 9001000 110046485052545658F1 Score (%)
t-test, padj<0.001 *** Mdiff=5.28
53.8554.63 54.79 54.4253.8154.82 55.14 55.3855.98 55.86 55.89
50.61
48.9549.9049.43 49.69 49.8050.58
48.3450.24
48.9749.94Coursera
100 200 300 400 500 600 700 800 9001000 11003234363840424446
Wilcoxon, padj=0.005 ** Mdiff=3.63
40.1640.5341.00 40.9541.42 41.2042.2541.4741.9542.62
41.53
35.8737.79 38.02 38.06 37.7438.64
36.9637.80 38.05 37.8438.34Coursera
100 200 300 400 500 600 700 800 9001000 110018202224262830
t-test, padj<0.001 *** Mdiff=3.68
25.17
24.0625.25 25.06 24.8625.6124.8725.4025.85 26.0825.44
21.35 21.2720.6821.7221.3522.3021.92 22.12
21.2121.75 21.51Coursera
100 200 300 400 500 600 700 800 9001000 1100
Pool size7072747678808284F1 Score (%)
t-test, padj<0.001 *** Mdiff=4.66
76.2977.1278.05 78.35 78.1779.09 78.8980.16
78.9080.0080.49
74.70 74.60
73.3574.6773.8174.3373.3275.16
73.20 73.0074.10Hotels
100 200 300 400 500 600 700 800 9001000 1100
Pool size5052545658606264
Wilcoxon, padj=0.005 ** Mdiff=5.52
57.21 57.09 57.2258.45 58.80 59.0859.90
58.8259.7959.2860.07
52.7254.31
53.2052.7454.01
52.77 52.79 53.02 52.6553.33 53.45Hotels
100 200 300 400 500 600 700 800 9001000 1100
Pool size30333639424548
t-test, padj<0.001 *** Mdiff=7.50
41.41
37.6538.5539.6142.75 42.78 43.13 43.1944.3243.7644.72
33.66 33.2632.44 32.9136.4035.18 35.2834.4035.11 34.9035.81Hotels
Rest16 FlightABSA Coursera Hotels RAG RandomFigure 3:F1 score comparison of RAG-based and random sampling approaches.RAG consistently
outperforms random sampling across all configurations. Statistical tests (paired t-test or Wilcoxon signed-
rank test with Holm-Bonferroni correction) confirm all differences are significant. Mdiffshows mean
performance differences.
on an NVIDIA RTX PRO 6000 GPU with 96 GB
VRAM.
3.3.1. Methodology
Selection Strategy.Following the methodology
established by Zhou et al. (2024), we employ BM25
(Robertson et al., 2009) as a sparse retrieval algo-
rithm for RAG, as it enables rapid similarity com-
parisons, thereby facilitating fast suggestion gen-
eration. For random sampling, the examples are
randomly selected from the pool.
Pool Size.To investigate performance evolution
as the number of gold-labelled examples available
for few-shot demonstration retrieval increases, we
analysed a spectrum ranging from 0 to 1,100 avail-
able training examples in incremental steps of 100.
For instance, when the pool size is 300, the few-
shot examples are selected from 300 examples.
We considered 1,100 examples as the maximum
pool size, as this was the largest multiple of 100
available across all datasets.
Datasets.Performance was evaluated acrossfour datasets spanning diverse review domains.
These datasets encompass reviews on restau-
rants (SemEval 2016, Rest16) (Pontiki et al., 2016;
Zhang et al., 2021), e-learning courses (Coursera)
(Chebolu et al., 2024), airlines (FlightABSA) (Hell-
wig et al., 2025), and hotels (Chebolu et al., 2024).
We randomly selected 1,100 examples from the
respective training sets.
LLM Configuration.Similar to Hellwig et al.
(2025), we employedGoogle’s Gemma-3-27B
(Team et al., 2025) with the temperature set to
0, ensuring deterministic selection of the highest
probability token during next-token prediction. The
prompt context incorporated 10 few-shot exam-
ples. For each combination of pool size, task and
dataset, the LLM was executed five times. Each
time, a different random seed was employed to
ensure varied selections of the 1,100 examples
extracted from the respective training set.
Tasks.We evaluated three tasks of varying com-
plexity: one single-aspect task (Aspect Category
Detection, ACD) and two tuple prediction tasks:
Target Aspect Sentiment Detection (TASD), and

Aspect Sentiment Quad Prediction (ASQP). ACD
requires the identification of all aspect categories
addressed within a given text. TASD extracts opin-
ion triplets comprising aspect term, aspect cate-
gory, and sentiment polarity, while ASQP addition-
ally extracts opinion terms, representing the most
fine-grained ABSA task.
Evaluation metrics.Evaluation for each step
was performed on the full respective test set of
each dataset. As common in the field of ABSA, the
reported evaluation metric is the micro-averaged
F1 score (Zhang et al., 2022). We publish all pre-
dicted labels and provide performance scores of
the macro-averaged F1 score, precision, and recall
in our GitHub repository.
3.3.2. Results & Discussion
The results (see Figure 3) demonstrated that the
RAG approach consistently outperformed random
few-shot selection across all pool sizes and tasks
for all datasets. Notably, RAG achieved substan-
tial performance gains, with differences ( Mdiff) of
up to six percentage points compared to random
sampling in several instances.
Pool LLM Execution time (seconds)
ACD TASD ASQP
RAG Random RAG Random RAG Random
100 0.816 0.851 1.292 1.240 1.708 1.622
200 0.848 0.852 1.291 1.204 1.723 1.626
300 0.846 0.871 1.291 1.211 1.736 1.642
400 0.862 0.856 1.298 1.221 1.743 1.622
500 0.858 0.867 1.314 1.220 1.743 1.647
600 0.868 0.852 1.308 1.219 1.745 1.630
700 0.864 0.884 1.311 1.210 1.739 1.640
800 0.865 0.872 1.320 1.210 1.755 1.638
900 0.868 0.864 1.322 1.204 1.759 1.635
1,000 0.872 0.856 1.310 1.202 1.745 1.636
1,100 0.866 0.859 1.296 1.199 1.756 1.617
AVG 0.858 0.862 1.305 1.213 1.741 1.632
Table 2:LLM inference time comparison for
ABSA tasks.Average execution time per predic-
tion (in seconds) comparing RAG-based versus
random sampling strategies across varying pool
sizes for ACD, TASD, and ASQP tasks.
Statistical significance was tested using paired
t-tests for normally distributed differences and
Wilcoxon signed-rank tests for non-normally dis-
tributed sets, followed by Holm-Bonferroni correc-
tion (Holm, 1979) for multiple testing across 12
comparisons ( α= 0.05 ). For each task-dataset
combination, we compared the 11 performance
scores (one at each pool size) obtained under
RAG versus random sampling. Statistical signifi-cance was observed across all task-dataset com-
binations.
Overall, our findings demonstrated that a RAG-
based approach achieved higher performance
scores than random sampling in the context of
a growing pool from which few-shot examples are
drawn, with statistically significant differences ob-
served across all tasks and datasets. Accordingly,
the RAG approach was integrated into the final
version of AnnoABSA.
Further performance improvements could poten-
tially be achieved by incorporating a larger number
of few-shot examples, as previously demonstrated
for random sampling by Hellwig et al. (2025). How-
ever, such improvements would come at the cost of
increased prediction latency, as additional tokens
would need to be loaded into the model context.
On our NVIDIA RTX PRO 6000 hardware, predic-
tions for individual tasks were completed within
seconds, as shown in Table 2, but increasing the
number of few-shot examples would proportionally
extend processing time or financial costs in the
case of a commercial LLM.
3.4. User Study on Annotation Speed
As with other LLM-assisted annotation tools, the
main motivation behind AnnoABSA’s LLM-based
suggestions is to reduce annotation time (Kim et al.,
2024). We therefore conducted a user study to
evaluate whether AnnoABSA enables faster anno-
tation compared to manual annotation without AI
assistance. We selected ASQP as the target task,
which considers the most sentiment elements per
tuple among all ABSA tasks (aspect term, aspect
category, opinion term, and sentiment polarity) and
is therefore the most comprehensive annotation
task.
3.4.1. Methodology
We conducted a within-subjects user study with 8
expert annotators, comprising four PhD students
and four master’s students in computer science,
all with prior experience in ABSA annotation tasks.
The study was conducted in a controlled usability
laboratory environment to minimize external dis-
tractions, with a research supervisor available ex-
clusively during the initial briefing phase to answer
questions before the annotation tasks commenced.
The study employed a counterbalanced design,
where each participant completed two annotation
sessions on separate days to eliminate fatigue ef-
fects: one session with AI suggestions enabled and
one without suggestions. To control for potential or-
dering and learning effects, we implemented Latin
square counterbalancing, systematically varying
both the system order (AI-first vs. baseline-first)

and dataset assignment (subset A vs. subset B)
across participants.
Subsets A and B each consisted of 50 randomly
sampled examples from the restaurant dataset pub-
lished as part of SemEval 2016 Task 5 (Pontiki
et al., 2016). Participants were provided with the
corresponding annotation guidelines for reference.
Prior to annotating the 50 test examples in each
session, participants completed a familiarization
phase with 5 demonstration examples to become
acquainted with the tool interface, during which
they could ask questions and receive guidance
from the research supervisor.
We employed the same LLM (Gemma-3-27B)
and GPU configuration as used in our selection
strategy evaluation. Given that the time annotators
took per example served as our primary evalua-
tion metric, we implemented frontend-based time
tracking. The timer was started upon loading each
example and terminated when participants opened
the subsequent example in the interface.
3.4.2. Results & Discussion
A paired t-test revealed a statistically significant
difference in mean annotation time per example
between the AI-assisted condition ( M= 24.19 ,
SD= 3.49 ) and the baseline condition ( M= 34.80 ,
SD= 6.92 ),t(7) =−3.640 ,p= 0.008 . The
Shapiro-Wilk test confirmed that the distribution
of differences satisfied the normality assumption
(W= 0.944 ,p= 0.647 ). These findings demon-
strate that LLM-generated suggestions can sub-
stantially accelerate human annotation workflows,
yielding a 30.51% reduction in annotation time.
Our results align with prior work demonstrating
that AI-assisted annotation can reduce the tempo-
ral demands of human labelling tasks (Kim et al.,
2024; Ni et al., 2024; Sahitaj et al., 2025; Li et al.,
2025), though the reduction observed in our study
is more modest than that reported for other tasks.
Notably, Ni et al. (2024) demonstrated that nearly
half of factual claim annotations could be fully au-
tomated through consistency-checked GPT -4 out-
puts, thereby reducing expert annotation effort by
approximately 50%. Similarly, Sahitaj et al. (2025)
reported a 73% reduction in annotation time for
propaganda detection in tweets, while Li et al.
(2025) observed a 36% reduction for AI-assisted
pre-annotation of x-ray images.
4. Conclusion & Future Work
We presented AnnoABSA, the first open-source,
web-based annotation tool supporting all subtasks
of ABSA. AnnoABSA features an intuitive and cus-
tomizable interface with built-in validation mecha-
nisms and optional LLM-powered suggestions viafew-shot prompting. Our evaluation demonstrated
that the integrated RAG-based approach, which
dynamically leverages the growing pool of anno-
tated examples during the annotation process, sig-
nificantly outperformed random sampling. A user
study revealed significant reductions in annotation
time when AI-assisted suggestions are employed.
While this work demonstrates AnnoABSA’s func-
tional capabilities and its positive impact on anno-
tation efficiency, future research should systemat-
ically investigate how AI assistance affects anno-
tation time, quality, and annotator confidence in
ABSA tasks. Such investigations could consider
both crowd-sourced workers and domain experts
with established ABSA expertise.
Finally, we note that our approach requires no
task-specific model training and provides immedi-
ate utility upon deployment, making it readily adapt-
able to diverse NLP annotation scenarios beyond
ABSA.
We invite the community to contribute to An-
noABSA’s continued development through our
GitHub repository. Feature requests in the form
of issues and pull requests are welcome, with our
commitment to timely integration of valuable contri-
butions to benefit the broader research community.
5. Ethics Statement and Limitations
This research was conducted without industrial
funding or commercial sponsorship. We employed
AI coding agents Claude Sonnet 4.511and Ope-
nAI’s open-source LLM gpt-oss:20b12for program-
ming support. Claude Sonnet 4.5 was also used
to assist in the formulation of this publication.
Several limitations should be considered when
interpreting our results. First, our evaluation of
LLM-based suggestions was restricted to Gemma-
3-27B due to computational constraints. Larger
language models or increased few-shot example
sizes could potentially yield superior suggestion
quality, albeit at higher computational or financial
costs in the case of commercial proprietary LLMs.
We executed 580,690 prompts for the evaluation
of LLM-based suggestions, which would have in-
curred substantial costs when employing commer-
cial models. However, AnnoABSA supports arbi-
trary Ollama and OpenAI-compatible models if one
wishes to employ those.
Finally, while we demonstrated significant reduc-
tions in annotation time, our evaluation was limited
to a small-scale annotation task. The generaliz-
ability of these efficiency gains to large-scale real-
world scenarios involving thousands of examples
11Claude Sonnet: https://www.anthropic.com/clau
de/sonnet
12gpt-oss:20b: https://ollama.com/library/gpt-o
ss:20b

remains to be validated, particularly regarding the
potential compounding effects of annotator fatigue
over extended annotation sessions.
6. Bibliographical References
Samy Ateia and Udo Kruschwitz. 2023. Is chatgpt
a biomedical expert? InWorking Notes of the
Conference and Labs of the Evaluation Forum
(CLEF 2023), Thessaloniki, Greece, Septem-
ber 18th to 21st, 2023, volume 3497 ofCEUR
Workshop Proceedings, pages 73–90. CEUR-
WS.org.
Yoav Benjamini and Yosef Hochberg. 1995. Con-
trolling the false discovery rate: A practical and
powerful approach to multiple testing.Journal of
the Royal Statistical Society. Series B (Method-
ological), 57(1):289–300.
Siva Uday Sampreeth Chebolu, Franck Dernon-
court, Nedim Lipka, and Thamar Solorio. 2024.
Oats: A challenge dataset for opinion aspect tar-
get sentiment joint detection for aspect-based
sentiment analysis. InProceedings of the 2024
Joint International Conference on Computational
Linguistics, Language Resources and Evalu-
ation (LREC-COLING 2024), pages 12336–
12347.
Luigi Colucci Cante, Salvatore D’Angelo, Beni-
amino Di Martino, and Mariangela Graziano.
2024.Text annotation tools: A comprehensive
review and comparative analysis, pages 353–
362. Springer.
Yong Deng, Xintong Zhang, Danping Zhou, De-
quan Zhang, and Boya Huang. 2024. Leverag-
ing nlp in finance: A synergistic approach using
large language models and chain-of-thought rea-
soning. InProceedings of the 5th International
Conference on Artificial Intelligence and Com-
puter Engineering, pages 494–500.
Tobias Deußer, Cong Zhao, Daniel Uedelhoven,
Lorenz Sparrenberg, Lars Hillebrand, Christian
Bauckhage, and Rafet Sifa. 2024. Leveraging
large language models for few-shot kpi extraction
from financial reports. In2024 IEEE International
Conference on Big Data (BigData), pages 4864–
4868. IEEE.
Jakob Fehle, Niklas Donhauser, Udo Kruschwitz,
Nils Constantin Hellwig, and Christian Wolff.
2025. German aspect-based sentiment anal-
ysis in the wild: B2b dataset creation and cross-
domain evaluation. In21st Conference on Nat-
ural Language Processing (KONVENS 2025),
volume 9, page 213.Safouane El Ghazouali and Umberto Michelucci.
2025. Visiofirm: Cross-platform ai-assisted an-
notation tool for computer vision.
Zhibin Gou, Qingyan Guo, and Yujiu Yang. 2023.
Mvp: Multi-view prompting improves aspect sen-
timent tuple prediction. InProceedings of the
61st Annual Meeting of the Association for Com-
putational Linguistics (Volume 1: Long Papers),
pages 4380–4397.
Md Arid Hasan, Shudipta Das, Afiyat Anjum, Firoj
Alam, Anika Anjum, Avijit Sarker, and Sheak
Rashed Haider Noori. 2024. Zero-and few-shot
prompting with llms: A comparative study with
fine-tuned models for bangla sentiment analy-
sis. InProceedings of the 2024 Joint Interna-
tional Conference on Computational Linguistics,
Language Resources and Evaluation (LREC-
COLING 2024), pages 17808–17818.
Nils Constantin Hellwig, Jakob Fehle, Markus Bink,
and Christian Wolff. 2024. GERestaurant: A
German dataset of annotated restaurant reviews
for aspect-based sentiment analysis. InPro-
ceedings of the 20th Conference on Natural
Language Processing (KONVENS 2024), pages
123–133, Vienna, Austria. Association for Com-
putational Linguistics.
Nils Constantin Hellwig, Jakob Fehle, Udo Kr-
uschwitz, and Christian Wolff. 2025. Do we still
need human annotators? prompting large lan-
guage models for aspect sentiment quad predic-
tion. InProceedings of the 1st Joint Workshop on
Large Language Models and Structure Modeling
(XLLM 2025), pages 153–172, Vienna, Austria.
Association for Computational Linguistics.
Sture Holm. 1979. A simple sequentially rejective
multiple test procedure.Scandinavian journal of
statistics, pages 65–70.
Hannah Kim, Kushan Mitra, Rafael Li Chen,
Sajjadur Rahman, and Dan Zhang. 2024.
Meganno+: A human-llm collaborative annota-
tion system. InProceedings of the 18th Confer-
ence of the European Chapter of the Association
for Computational Linguistics: System Demon-
strations, pages 168–176.
Jan-Christoph Klie, Michael Bugert, Beto Boullosa,
Richard Eckart de Castilho, and Iryna Gurevych.
2018. The INCEpTION platform: Machine-
assisted and knowledge-oriented interactive an-
notation. InProceedings of the 27th International
Conference on Computational Linguistics: Sys-
tem Demonstrations, pages 5–9, Santa Fe, New
Mexico.
Yanis Labrak, Mickaël Rouvier, and Richard Du-
four. 2024. A zero-shot and few-shot study of

instruction-finetuned large language models ap-
plied to clinical and biomedical tasks. InProceed-
ings of the 2024 Joint International Conference
on Computational Linguistics, Language Re-
sources and Evaluation (LREC-COLING 2024),
pages 2049–2066.
Clément Le Ludec, Maxime Cornet, and Anto-
nio A Casilli. 2023. The problem with annota-
tion. human labour and outsourcing between
france and madagascar.Big Data & Society,
10(2):20539517231188723.
Y an Li, Hao Qiu, Xu Wang, Na Dong, and Xinghua
Yu. 2025. Rapidx annotator: A specialized soft-
ware tool for industrial radiographic image anno-
tation and enhancement.SoftwareX, 31:102328.
Yida Mu, Ben P . Wu, William Thorne, Ambrose
Robinson, Nikolaos Aletras, Carolina Scarton,
Kalina Bontcheva, and Xingyi Song. 2024. Nav-
igating prompt complexity for zero-shot classi-
fication: A study of large language models in
computational social science. InProceedings
of the 2024 Joint International Conference on
Computational Linguistics, Language Resources
and Evaluation (LREC-COLING 2024), pages
12074–12086, Torino, Italia. ELRA and ICCL.
Arbi Haza Nasution and Aytug Onan. 2024. Chat-
gpt label: Comparing the quality of human-
generated and llm-generated annotations in low-
resource language nlp tasks.IEEE Access.
Gaurav Negi, Rajdeep Sarkar, Omnia Zayed, and
Paul Buitelaar. 2024. A hybrid approach to as-
pect based sentiment analysis using transfer
learning. InProceedings of the 2024 Joint In-
ternational Conference on Computational Lin-
guistics, Language Resources and Evaluation
(LREC-COLING 2024), pages 647–658.
Jingwei Ni, Minjing Shi, Dominik Stammbach, Mrin-
maya Sachan, Elliott Ash, and Markus Leippold.
2024. Afacta: Assisting the annotation of factual
claim detection with reliable llm annotators. In
Proceedings of the 62nd Annual Meeting of the
Association for Computational Linguistics (Vol-
ume 1: Long Papers), pages 1890–1912.
Maria Pontiki, Dimitrios Galanis, Harris Papageor-
giou, Ion Androutsopoulos, Suresh Manandhar,
Mohammad AL-Smadi, Mahmoud Al-Ayyoub,
Y anyan Zhao, Bing Qin, Orphee De Clercq, et al.
2016. Semeval-2016 task 5: Aspect based sen-
timent analysis. InProceedings of the 10th In-
ternational Workshop on Semantic Evaluation
(SemEval-2016), pages 19–30.
Maria Pontiki, Dimitris Galanis, Haris Papageor-
giou, Suresh Manandhar, and Ion Androutsopou-
los. 2015. SemEval-2015 task 12: Aspect basedsentiment analysis. InProceedings of the 9th
International Workshop on Semantic Evaluation
(SemEval 2015), pages 486–495, Denver, Col-
orado. Association for Computational Linguistics.
Maria Pontiki, Dimitris Galanis, John Pavlopoulos,
Harris Papageorgiou, Ion Androutsopoulos, and
Suresh Manandhar. 2014. SemEval-2014 task
4: Aspect based sentiment analysis. InPro-
ceedings of the 8th International Workshop on
Semantic Evaluation (SemEval 2014), pages 27–
35, Dublin, Ireland. Association for Computa-
tional Linguistics.
Stephen Robertson, Hugo Zaragoza, et al. 2009.
The probabilistic relevance framework: Bm25
and beyond.Foundations and Trends® in Infor-
mation Retrieval, 3(4):333–389.
Ariana Sahitaj, Premtim Sahitaj, Veronika
Solopova, Jiaao Li, Sebastian Möller, and Vera
Schmitt. 2025. Hybrid annotation for propa-
ganda detection: Integrating llm pre-annotations
with human intelligence. InProceedings of the
Fourth Workshop on NLP for Positive Impact
(NLP4PI), pages 215–228.
Somayeh Sani and Y eganeh Shokooh. 2016. Mini-
malism in designing user interface of commercial
websites based on gestalt visual perception laws
(case study of three top brands in technology
scope). pages 115–124.
Hope Schroeder, Deb Roy, and Jad Kabbara. 2025.
Just put a human in the loop? investigating llm-
assisted annotation for subjective tasks. InFind-
ings of the Association for Computational Lin-
guistics: ACL 2025, pages 25771–25795.
Gemma Team, Aishwarya Kamath, Johan Fer-
ret, Shreya Pathak, Nino Vieillard, Ramona
Merhej, Sarah Perrin, Tatiana Matejovicova,
Alexandre Ramé, Morgane Rivière, et al. 2025.
Gemma 3 technical report.arXiv preprint
arXiv:2503.19786.
An Wang, Junfeng Jiang, Youmi Ma, Ao Liu, and
Naoaki Okazaki. 2024. Generative data augmen-
tation for aspect sentiment quad prediction.Jour-
nal of Natural Language Processing, 31(4):1523–
1544.
Michael Wojatzki, Eugen Ruppert, Sarah
Holschneider, Torsten Zesch, and Chris Bie-
mann. 2017. Germeval 2017: Shared task on
aspect-based sentiment in social media cus-
tomer feedback.Proceedings of the GermEval,
pages 1–12.
Hongling Xu, Yice Zhang, Qianlong Wang, and
Ruifeng Xu. 2025. DS2-ABSA: Dual-stream
data synthesis with label refinement for few-shot

aspect-based sentiment analysis. InProceed-
ings of the 63rd Annual Meeting of the Asso-
ciation for Computational Linguistics (Volume
1: Long Papers), pages 15460–15478, Vienna,
Austria. Association for Computational Linguis-
tics.
Wenxuan Zhang, Yang Deng, Xin Li, Yifei Yuan,
Lidong Bing, and Wai Lam. 2021. Aspect senti-
ment quad prediction as paraphrase generation.
InProceedings of the 2021 Conference on Em-
pirical Methods in Natural Language Processing,
pages 9209–9219.
Wenxuan Zhang, Yue Deng, Bing Liu, Sinno Pan,
and Lidong Bing. 2024. Sentiment analysis in the
era of large language models: A reality check.
InFindings of the Association for Computational
Linguistics: NAACL 2024, pages 3881–3906.
Wenxuan Zhang, Xin Li, Y ang Deng, Lidong Bing,
and Wai Lam. 2022. A survey on aspect-based
sentiment analysis: Tasks, methods, and chal-
lenges.IEEE Transactions on Knowledge and
Data Engineering, 35(11):11019–11038.
Changzhi Zhou, Dandan Song, Yuhang Tian, Zhi-
jing Wu, Hao Wang, Xinyu Zhang, Jun Yang,
Ziyi Y ang, and Shuhao Zhang. 2024. A compre-
hensive evaluation of large language models on
aspect-based sentiment analysis.arXiv preprint
arXiv:2412.02279.

A. ABSA Tasks
Task Type Task Name Elements Gold Label (Example)
SingleAspect Term Extraction (ATE) a [’waiter’]
Aspect Category Detection (ACD)c[’food quality’, ’service speed’]
Opinion Term Extraction (OTE) o [’delicious’, ’way too slow’]
CompoundAspect Sentiment Classification (ASC)a,s[(’waiter’, ’negative’)]
Aspect-Opinion Pair Extraction (AOPE) a,o [(’waiter’, ’way too slow’)]
End-to-End ABSA (E2E-ABSA)a,s[(’NULL’, ’positive’), (’waiter’, ’negative’)]
Aspect Category Sentiment Analysis (ACSA) c,s [(’food quality’, ’positive’), (’service speed’, ’negative’)]
Aspect Sentiment Triplet Extraction (ASTE)a,o,s[(’NULL’, ’delicious’, ’positive’),
(’waiter’, ’way too slow’, ’negative’)]
Target Aspect Sentiment Detection (TASD) a,c,s[(’NULL’, ’food quality’, ’positive’),
(’waiter’, ’service speed’, ’negative’)]
Aspect Sentiment Quad Prediction (ASQP)a,c,o,s[(’NULL’, ’food quality’, ’delicious’, ’positive’),
(’waiter’, ’service speed’, ’way too slow’, ’negative’)]
Aspect-Category-Opinion-Sentiment
Quad Extraction (ACOS)a,c,o,s[(’NULL’, ’food quality’, ’delicious’, ’positive’),
(’waiter’, ’service speed’, ’way too slow’, ’negative’)]
Table 3:Overview of ABSA tasks supported by AnnoABSA.For each task, a gold label is presented for
the sentence“It was really delicious, but the waiter was way too slow”. Aspect categories are commonly
selected from a predefined set ({food quality, service speed, ...}). Notation:a= aspect term,c= aspect
category,o= opinion term,s= sentiment polarity. In case of an implicit aspect, aspect termais set to
’NULL’. The listed tasks are equivalent to those reported in the literature review by Zhang et al. (2022)

B. CLI Flags
Option Description Default
Server Configuration
–backend Start only backend server –
–backend-portBackend server port8000
–frontend-port Frontend server port 3000
–backend-ipBackend server IP addresslocalhost
–frontend-ip Frontend server IP address localhost
Session Management
–session-idUnique identifier for an annotation session.None
Annotation Elements
–elements Sentiment elements to annotate aspect_term, aspect_category, sen-
timent_polarity, opinion_term
–polaritiesValid sentiment polarities positive, negative, neutral
–categories Valid aspect categories Restaurant domain (13 categories)
–implicit-aspectAllow implicit aspect terms Enabled
–disable-implicit-aspect Disable implicit aspect terms Disabled
–implicit-opinionAllow implicit opinion terms Disabled
–disable_implicit_opinion Disable implicit opinion terms Enabled
Interface and Processing
–disable_clean_phrases Disable automatic punctuation cleaning from phrase
start/endEnabled
–disable-save-positions Disable saving phrase positions Enabled
–disable-click-on-tokenDisable click-on-token feature Enabled
–auto-positions Enable automatic position filling on startup Disabled
–annotation-guidelinesPath to PDF file containing annotation guidelines Disabled
Analytics and Timing
–store-timeStore timing data for annotation sessions Disabled
–display-avg-annotation-time Display average annotation time in the interface Disabled
AI Integration
–ai-suggestions Enable AI-powered prediction suggestions using LLM Disabled
–disable-ai-automatic-predictionDisable automatic AI prediction triggering Disabled
–llm-model LLM employed for suggestions (e.g., gemma3:4b ,gpt-4o )gemma3:4b
–openai-keyOpenAI API key for using OpenAI models –
–n-few-shot Maximum number of few-shot examples in LLM context 10
Configuration Management
–save-configSave config to JSON file –
–show-config Display current configuration in console –
Table 4:CLI options for AnnoABSA.AnnoABSA offers extensive customization across server settings,
annotation logic, and AI integration.

C. Prompt for RAG-Based Suggestions
Accordingtothefollowingsentimentelementsdefinition: -The 'aspectterm' istheexactwordorphrasein thetextthatrepresentsa specificfeature, attribute, oraspectofa productorservicethata user mayexpress an opinionabout. The aspecttermmightbe'NULL' forimplicitaspect.-The 'aspectcategory' referstothecategorythataspectbelongsto, and theavailablecategoriesincludes: hotelcomfort, roomsdesign_features, facilitiesmiscellaneous, roomscleanliness, food_drinksmiscellaneous, food_drinksstyle_options, facilitiescomfort, room_amenitiesgeneral, servicegeneral, roomsquality, roomsgeneral, hotelgeneral, food_drinksprices, facilitiesgeneral, room_amenitiesquality, facilitiesquality, roomsmiscellaneous, facilitiesdesign_features, hotelprices, food_drinksquality, room_amenitiesprices, room_amenitiescomfort, roomsprices, hotelcleanliness, hotelmiscellaneous, facilitiesprices, roomscomfort, hotelquality, room_amenitiesdesign_features, locationgeneral, facilitiescleanliness, hoteldesign_features, room_amenitiescleanliness.-The 'sentimentpolarity' referstothedegreeofpositivity, negativityorneutralityexpressedin theopiniontowardsa particularaspectorfeature of a productorservice, and theavailablepolaritiesinclude: neutral, negative, positive.-The 'opinionterm' istheexactwordorphrasein thetextthatreferstothesentimentorattitudeexpressedbya usertowardsa particularaspector feature ofa productorservice. Recognizeall sentimentelementswiththeircorrespondingaspectterms, aspectcategorys, sentimentpolaritys, opiniontermsin thefollowingtextin the form ofa listofobjects, eachobjecthavingkey(s) 'aspectterm', 'aspectcategory', 'sentimentpolarity', 'opinionterm'.Here aresomeexamples:Text: This isa budgethotelin a goodlocationcentraltoeverything.Sentiment elements: [('aspectterm': 'hotel', 'aspectcategory': 'hotelprices', 'sentimentpolarity': 'positive', 'opinionterm': 'budget'), ('aspect term': 'hotel', 'aspectcategory': 'locationgeneral', 'sentimentpolarity': 'positive', 'opinionterm': 'goodlocation')]Text: itwas veryrelaxinggoodbuffet breakfastincludedand so closetoeverything.Sentiment elements: [('aspectterm': 'breakfast', 'aspectcategory': 'food_drinksquality', 'sentimentpolarity': 'positive', 'opinionterm': 'relaxing'), ('aspectterm': 'breakfast', 'aspectcategory': 'food_drinksquality', 'sentimentpolarity': 'positive', 'opinionterm': 'good')]Text: LovedourSpauldingHotel Very goodlocation, centrallylocatedclosetoTheaters and shopping.Sentiment elements: [('aspectterm': 'SpauldingHotel', 'aspectcategory': 'hotelgeneral', 'sentimentpolarity': 'positive', 'opinionterm': 'Loved'), ('aspectterm': 'location', 'aspectcategory': 'locationgeneral', 'sentimentpolarity': 'positive', 'opinionterm': 'Very good'), ('aspectterm': 'SpauldingHotel', 'aspectcategory': 'locationgeneral', 'sentimentpolarity': 'positive', 'opinionterm': 'centrallylocatedclosetoTheaters and shopping')]Text: Goodrestaurantsnearby(Sentiment elements: [('aspectterm': 'NULL', 'aspectcategory': 'locationgeneral', 'sentimentpolarity': 'positive', 'opinionterm': 'Goodrestaurantsnearby')]Text: wellthoughtout design toincorporatealotofextras.Sentiment elements: [('aspectterm': 'NULL', 'aspectcategory': 'roomsdesign_features', 'sentimentpolarity': 'positive', 'opinionterm': 'wellthought out design')]Text: Therearea numberofrestaurantsin walkingdistance.Sentiment elements: [('aspectterm': 'NULL', 'aspectcategory': 'locationgeneral', 'sentimentpolarity': 'positive', 'opinionterm': 'numberofrestaurantsin walkingdistance')]Text: The ownersand staffareextremelyaccomodating, offeringgoodadvicetorestaurantsSentiment elements: [('aspectterm': 'owners', 'aspectcategory': 'locationgeneral', 'sentimentpolarity': 'positive', 'opinionterm': 'extremelyaccomodating'), ('aspectterm': 'staff', 'aspectcategory': 'locationgeneral', 'sentimentpolarity': 'positive', 'opinionterm': 'offeringgoodadviceto restaurants')]Text: Outside isa magnificentviewoftheriverforwhichthehotelisnamedand parallel isa cobblestonestreetfilledwitha widerangeofcharming shops.Sentiment elements: [('aspectterm': 'hotel', 'aspectcategory': 'hoteldesign_features', 'sentimentpolarity': 'positive', 'opinionterm': 'magnificentviewoftheriver')]Text: The onlythingthatthehotelcouldimproveon was seatingin themainlobby--thereisa coupleofchairsand a couchand alotofemptyspace. Sentiment elements: [('aspectterm': 'hotel', 'aspectcategory': 'facilitiescomfort', 'sentimentpolarity': 'neutral', 'opinionterm': 'improveon was seatingin themainlobby')]Text: greatlocationclosetoamenities, but nice and quieton an evening.Sentiment elements: [('aspectterm': 'location', 'aspectcategory': 'locationgeneral', 'sentimentpolarity': 'positive', 'opinionterm': 'great')] Text: Itisin a goodlocationand closetoalotofshopsand restaurants.Sentiment elements: 
Figure 4:Prompt used for RAG-based suggestion prediction.The prompt includes a task description
with explanations of sentiment elements, ten in-context demonstrations, and the target text for aspect
prediction. The few-shot examples shown are taken from the Coursera dataset and include annotations
for ASQP