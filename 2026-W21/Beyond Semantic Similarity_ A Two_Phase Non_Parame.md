# Beyond Semantic Similarity: A Two-Phase Non-Parametric Retrieval Workflow for Corporate Credit Underwriting

**Authors**: Linus Ng Junjia, Ezekiel Tee Kongquan, Kelvin Heng, Kenneth Zhu Ke, Zhao Jing Yuan

**Published**: 2026-05-20 04:23:06

**PDF URL**: [https://arxiv.org/pdf/2605.20684v1](https://arxiv.org/pdf/2605.20684v1)

## Abstract
Corporate credit underwriting requires analysts to extract actionable evidence from long, heterogeneous financial documents spanning hundreds of pages and multiple languages. Standard Retrieval-Augmented Generation (RAG) pipelines optimize for semantic similarity, which frequently surfaces passages that are topically related but lack decision utility, a problem we term the similarity-utility gap. We propose a two-phase non-parametric retrieval architecture that separates high-recall candidate retrieval from high-precision utility ranking. The first phase combines lexical and dense multilingual retrieval to construct a broad candidate pool. The second phase applies an adaptive retrieval controller that filters candidates using query intent and document structure signals, followed by an LLM-as-a-Judge utility scoring mechanism that ranks passages by analytical usefulness rather than semantic proximity.
  A context-aware extraction module preserves structural fidelity across narrative text and complex financial tables. The system is deployed entirely on-premise to satisfy enterprise data governance requirements. Evaluated on a multilingual corpus of proprietary financial documents with analyst-curated relevance labels, the system significantly outperforms naive retrieval baselines. In production deployment across more than 800 credit analysts, document review time was reduced from several hours to approximately three minutes, demonstrating the practical value of utility-aware RAG architectures for document-intensive decision-support workflows.

## Full Text


<!-- PDF content starts -->

Beyond Semantic Similarity: A Two-Phase Non-Parametric Retrieval Workflow for
Corporate Credit Underwriting
Linus Ng Junjia1Ezekiel Tee Kongquan1 2Kelvin Heng1Kenneth Zhu Ke1Zhao Jing Yuan1
1OCBC, Singapore2Georgia Institute of Technology. linus.ng@ocbc.com , ezekieltee@ocbc.com ,
kelvinheng@ocbc.com, kennethzhu@ocbc.com, jingyuanzhao@ocbc.com.
1. Introduction
Corporate credit underwriting relies heavily on
the analysis of long-form financial documents such
as annual reports and industry reports [ 1]. Analysts
must extract relevant financial indicators, assess risk
disclosures, and synthesize insights from documents
that can span hundreds of pages and multiple lan-
guages [1].
Retrieval-Augmented Generation (RAG) systems
have emerged as a promising approach to assist
document-intensive workflows [ 2,3]. By retrieving
supporting passages from external corpora, RAG sys-
tems can improve factual grounding and reduce hal-
lucination in language model outputs [ 2,3]. However,
standard RAG pipelines typically prioritize semantic
similarity between queries and document passages
[2]. In financial analysis tasks, this objective often
fails to align with the needs of analysts [1].
Financial documents often contain narrative de-
scriptions, regulatory disclosures, and repetitive boil-
erplate language [ 1]. As a result, similarity-based re-
trieval systems may surface passages that share termi-
nology with the query but lack actionable analytical
value [ 4]. We refer to this issue as thesimilarity–utility
gap[4].
To address this challenge, we propose a retrieval
architecture designed specifically for enterprise
decision-support workflows [ 4]. Our system intro-
duces a two-phase pipeline that separates high-recall
retrieval from high-precision utility ranking [ 4]. The
architecture incorporates hybrid lexical-semantic re-
trieval, adaptive candidate refinement, and a utility-
grounded ranking framework in which a language
model evaluates passages according to their analyti-
cal usefulness [1, 4].
The system is designed for deployment in regu-
lated financial environments and operates entirely
on-premise using self-hosted open-source models.
This ensures compliance with strict data governance
policies while maintaining high retrieval perfor-
mance.
Our main contributions are:
•A utility-grounded retrieval framework that
aligns passage ranking with decision usefulness
in credit underwriting tasks.
•An adaptive retrieval controller that filters candi-
date passages using query intent and document
structure signals.
•A context-aware extraction module that pre-
serves structural information in narrative and
tabular financial content.•An enterprise deployment demonstrating large
productivity gains for analysts.
2. Related Work
Retrieval-Augmented Generation has become a
widely adopted approach for grounding large lan-
guage models in external knowledge sources [ 2,3].
Early work introduced dense retrieval methods using
neural embeddings to retrieve semantically relevant
passages from large corpora [ 2]. Hybrid retrieval ap-
proaches combining lexical search with dense em-
beddings have further improved recall in heteroge-
neous document collections [5].
Recent research has explored the use of language
models as evaluators for ranking retrieved content [ 6].
TheseLLM-as-a-Judgeapproaches leverage the rea-
soning capabilities of language models to assess the
quality and relevance of candidate passages [ 6]. Such
methods have been applied in question answering,
information retrieval evaluation, and ranking tasks.
In enterprise settings, RAG systems must also ad-
dress constraints related to data governance, privacy,
and auditability. Financial and legal institutions often
require on-premise deployment and traceable source
attribution, which introduces additional design con-
siderations beyond model performance.
Our work contributes to this literature by propos-
ing a retrieval architecture that explicitly optimizes
for analytical utility in enterprise workflows. By
combining hybrid retrieval, adaptive candidate con-
trol, and utility-based ranking, the system prioritizes
passages that contain actionable financial evidence
rather than merely semantically similar text.
3. Methodology
3.1 Problem Setting
Corporate credit underwriting requires analysts
to justify reported financial statements derived from
long, heterogeneous financial documents such as an-
nual reports and industry reports. These documents
often contain dense narrative sections, multilingual
commentary, and unstructured financial tables. Tra-
ditional RAG pipelines retrieve passages based pri-
marily on semantic similarity, which frequently re-
sults in content that is topically related but not useful
for decision-making.
We formalize the task as retrieving and ranking
a set of document segments that maximize decision
utility rather than semantic similarity. Given a user
query𝑞and a corpus of long financial documents
𝐷={𝑑 1,𝑑2,...,𝑑 𝑛}, the objective is to identify a setarXiv:2605.20684v1  [cs.CL]  20 May 2026

of passages 𝑃∗⊂𝐷that contain verifiable evidence
relevant to credit underwriting decisions.
Our system introduces a two-phase retrieval and
re-ranking architecture designed to bridge the gap be-
tween semantic similarity and decision utility while
operating entirely within an on-premise environ-
ment.
3.2 System Overview
The proposed architecture consists of five main
components: document ingestion, hybrid candi-
date retrieval, adaptive retrieval controller, utility-
grounded re-ranking, and context-aware evidence
extraction.
The overall pipeline is illustrated conceptually as:
𝑞statement→𝑅 hybrid(𝐷)→𝐶 adaptive→𝐽 utility→𝐸 context
where𝑞statement a query supplemented with its cor-
responding financial statement, 𝑅hybrid retrieves an
initial candidate pool, 𝐶adaptive filters candidates us-
ing query-aware reasoning, 𝐽utility ranks candidates
by decision usefulness, 𝐸context extracts the final evi-
dence
3.3 Document Ingestion
Corporate financial documents are segmented into
structured sections using document layout cues. Each
segment is indexed with metadata that include the
source of the document, the title of the section, and
the page references. This preprocessing step ensures
that downstream components can leverage structural
information during retrieval and extraction.
3.4 Phase 1: Hybrid Candidate Retrieval
The first phase performs broad retrieval to max-
imize recall across multilingual and heterogeneous
financial documents.
Given a query 𝑞, we retrieve an initial candidate
set𝐶0using a hybrid retrieval strategy that combines
keyword retrieval and dense semantic retrieval using
multilingual embeddings:
𝐶0=TopKkw(𝑞,𝐷)∪TopKembed(𝑞,𝐷)
In Hybrid retrieval, keyword retrieval preserves
precision for financial terminology, while semantic
retrieval captures paraphrased or contextual refer-
ences.
The choice of the TopK value represents a critical
hyperparameter at this stage: setting it too low risks
restricting the candidate pool passed to the subse-
quent phases, while setting it too high diminishes the
value of initial retrieval. With a TopK value set to 50,
the result is a high-recall candidate pool of passages
for further processing.
3.5Phase 2: Adaptive Retrieval Controller & Utility-
Grounded Re-ranking
Not all retrieved passages are useful for credit anal-
ysis. Financial reports contain boilerplate disclo-sures, legal notes, and narrative sections that may
be semantically related to a query, but irrelevant to
underwriting decisions.
To address this issue, we introduce an adaptive
retrieval controller that evaluates candidate passages
using query intent and document structure.
Given candidate passages 𝐶0={𝑝 1,...,𝑝 𝑘}, the con-
troller predicts relevance and support for a given pas-
sage:
𝑅𝑒𝑙𝑖=𝑓(𝑞 statement,𝑝𝑖,𝑚𝑖)
𝑆𝑖=𝑓(𝑞 statement,𝑝𝑖,𝑚𝑖)
𝑈𝑖=𝑓(𝑞 statement,𝑝𝑖,𝑚𝑖)
where:
•𝑞statement is a query supplemented its correspond-
ing financial statement
•𝑝𝑖is the candidate passage
•𝑚 𝑖represents structural metadata
•𝑅𝑒𝑙 𝑖represents relevancy (boolean) of candidate
passage
•𝑆𝑖represents evidence support (boolean) of can-
didate passage
•𝑈 𝑖represents utility score (numerical score) of
candidate passage
The controller is implemented using a lightweight
language model that evaluates whether a passage is
likely to contain relevant and supportive information.
This stage produces a refined candidate set:
𝐶1={𝑝 𝑖∈𝐶0|(𝑆 𝑖)·[𝑅𝑒𝑙 𝑖]}
where𝑆𝑖·[𝑅𝑒𝑙 𝑖]represents the logical condition
that𝑆 𝑖is effectively nullified unless𝑅𝑒𝑙 𝑖succeeds.
This mechanism is conceptually related to adaptive
retrieval strategies that condition retrieval on model
reasoning rather than fixed pipelines [7].
The remaining candidates are ranked according to
decision utility using an LLM-as-a-Judge framework.
Rather than measuring similarity to the query, the
judge evaluates each passage based on its usefulness
for underwriting decisions. Given a passage 𝑝𝑖, the
judge produces a utility score:
𝐽1={𝑝 𝑖∈𝐶1|𝑈𝑖≥𝑈 threshold}
where𝑈𝑖is the utility score and 𝑈threshold is the
tunable utility score threshold. 𝑈threshold serves as
a critical hyperparameter in the architecture, bal-
ancing precision and recall in the final evidence
set. A lower 𝑈threshold increases recall by admitting a
larger set of passages, which is advantageous for ex-
ploratory queries or scenarios where analysts require
broad and comprehensive coverage. Conversely, a
higher𝑈threshold prioritizes precision, returning only
passages with the highest decision utility, which is
ideal for targeted queries where concise evidence is
needed.

The utility-grounded re-ranking mechanism en-
ables the system to prioritize passages containing
financial indicators and industry signals.
3.6 Context-Aware Evidence Extraction
Financial documents frequently contain complex
tables, footnotes, and structured subsections. Simple
chunk extraction can distort meaning, break struc-
tural relationships, or cause loss of attribution. We
therefore introduce a context-aware extraction mod-
ule that dynamically selects the appropriate extrac-
tion strategy based on document structure.
Two extraction modes are employed depending
on the structure of the source content. For narra-
tive sections, relevant text spans are extracted us-
ing markdown-aware segmentation, which preserves
structural elements such as section headers, bullet
lists, and paragraph boundaries.
When information appears in tables or struc-
tured financial statements, the system distinguishes
between complex and non-complex tables. Non-
complex tables (tables with single-level headers and
regular grid structures) are parsed to extract the
relevant rows or cells. In contrast, complex tables
containing multi-level headers, hierarchical indices,
merged cells, or irregular layouts are preserved along
with source metadata, including the document name
and page reference, to support manual verification
and accurate source attribution.
Two extraction modes are used:
•Localized Passage Extraction
For narrative sections, relevant text spans are
extracted using markdown-aware segmentation.
This preserves structural markers such as:
–section headers
–bullet lists
–paragraph boundaries
•High-Fidelity Table Citation
When information resides within tables or struc-
tured financial statements, the system classifies
tables as either complex tables or non-complex
tables:
–Complex tables: Multi-level headers, hier-
archical row indices, irregular structures,
or merged cells that require specialized han-
dling
–Non-complex tables: Single-level headers
and row indices, well-structured grids with
regular formatting
For non-complex tables, the system performs
structured table parsing to extract the relevant rows
or cells. For complex tables, rather than attempting
to parse the structure, the system preserves the full
table context, including source metadata such as doc-
ument name, and page reference. This information isappended as supplementary information to support
manual verification and referencing.
This approach ensures that extracted financial
metrics remain interpretable and can be reliably
traced back to their original document context, while
accommodating the diversity and structural complex-
ity of tables across different reports.
4. System Architecture
The proposed system consists of five main compo-
nents: document ingestion, hybrid retrieval, adaptive
candidate control, utility-grounded re-ranking, and
context-aware extraction.
Figure 1 illustrates the overall pipeline.
To satisfy regulatory and data governance con-
straints in corporate credit underwriting, the pro-
posed system is deployed entirely within an on-
premise environment, with all components executed
under enterprise-controlled infrastructure. This
ensures that sensitive financial data remains local
across all stages of the pipeline.
The deployment architecture is structurally
aligned with the proposed formulation defined in
Section 3.2 and is realized through a collection of
self-hosted modules that implement each stage
without reliance on external services.
•Retrieval Infrastructure
Hybrid retrieval is implemented via the com-
bination of lexical and dense retrieval mecha-
nisms over locally indexed document collections.
Keyword-based retrieval operates on structured
indices, while dense retrieval is enabled through
multilingual embedding models deployed on-
premise. Document representations are stored
and queried through an internal vector database,
facilitating efficient construction and propaga-
tion of the candidate set𝐶 0.
•Agentic Reasoning Modules
The adaptive retrieval controller and utility-
grounded re-ranking stages are instantiated us-
ing self-hosted language models. Lightweight
language models are utilized for relevance and
support classification to enable efficient pruning
of candidates, while higher-capacity models per-
form utility scoring under the LLM-as-a-Judge
framework and support final response genera-
tion. This separation reflects the staged evalua-
tion functions defined in Section 3.5, while main-
taining computational tractability within an on-
premise setting.
•Context-Aware Extraction
The evidence extraction stage operates directly
on the filtered and ranked candidate set 𝐽1, apply-
ing structure-aware processing to heterogeneous
document segments. Narrative text and tabular
content are handled through distinct extraction
strategies executed locally, preserving structural

Fig. 1: Utility-grounded retrieval architecture for long-document financial analysis.
fidelity and source attribution without external
transformation.
•Data Flow and Persistence
System components are coordinated through in-
ternal messaging pipelines that support scalable
ingestion and processing of financial documents.
Intermediate artifacts (candidate passages, fil-
tering decisions, utility scores, and associated
metadata) are persistently maintained, enabling
reproducibility, traceability, and auditability of
all outputs.
This design ensures that all decision-relevant com-
putations remain verifiable, controllable, compliant
with enterprise governance requirements, and that
sensitive financial data never leaves the enterprise
infrastructure.
5. Results
We evaluated the system on a multilingual corpus
of financial documents with relevance labels curated
by credit analysts. Compared to traditional naive re-
trieval systems, the proposed approach significantly
improves retrieval performance.
Due to the proprietary nature of the financial doc-
uments used in this study, the evaluation was con-
ducted in a restricted enterprise environment. The
dataset consists of internal corporate credit docu-
ments including financial reports, industry analyses,
and related underwriting materials used in produc-
tion workflows. These documents contain confiden-
tial financial information and cannot be publicly re-
leased. As a result, the annotated dataset and detailed
case studies used for evaluation are not included in
the public version of this work.
To ensure meaningful evaluation despite these
constraints, financial statements and additional con-
texts were curated by senior credit analysts based on
real underwriting tasks. Queries were constructed toreflect analytical questions commonly encountered
in credit assessment workflows, such as identifying
components in balance sheets, income statements,
and industry outlook signals.
In real-world deployment across more than 800
analysts, the system reduced document review time
from several hours to approximately three minutes.
6. Summary
The results highlight the limitations of similarity-
based retrieval for analytical tasks. By incorporating
utility signals into the ranking process, the proposed
system surfaces passages containing actionable fi-
nancial evidence.
Adaptive candidate filtering further improves com-
putational efficiency by reducing noise prior to the
utility-based ranking stage.
7. Conclusion & Future Work
We present a utility-grounded retrieval architec-
ture for corporate credit underwriting workflows in-
volving long financial documents. The proposed sys-
tem combines hybrid retrieval, adaptive candidate
filtering, and LLM-based utility ranking to prioritize
decision-relevant evidence.
Evaluation results demonstrate significant im-
provements in retrieval accuracy and analyst produc-
tivity. The system has been successfully deployed in a
large enterprise environment, highlighting the poten-
tial of utility-aware RAG architectures for document-
intensive decision-support applications.
Future work will explore enhanced structured data
extraction from financial tables. The current system
preserves traceability when detecting and retrieving
tabular content, including document location and
structural context. This capability opens the pos-
sibility of integrating specialized Optical Character
Recognition (OCR) and Vision Language Model (VLM)
pipelines to extract structured numerical information

directly from table regions.
By combining table detection that leverages OCR
extraction together with VLMs, the system could cap-
ture additional financial metrics that are often em-
bedded within complex document layouts. This struc-
tured information could then be incorporated into
downstream retrieval and reasoning processes, en-
abling richer analytical queries that combine narra-
tive explanations with precise financial figures.
Acknowledgments
This work was conducted at OCBC AI Lab. The au-
thors would like to express their sincere gratitude to
Kok Ker Ern Kovan, Lee Sheng Kiat, and Germaine
Goh Yanshan for their valuable contributions and
support throughout the project. Their insights and
domain expertise were instrumental in the successful
outcomes presented in this paper. The authors would
also like to specifically thank Ren Xuezhe, Lisa Tan,
Weiyang Song, Qishuai Zhong, and Kenneth Loh Zhen
Xiang for their significant contributions to the suc-
cessful delivery and shipment of the product. Their
dedication and collaboration greatly enhanced the
impact and practical application of this work.
References
[1]Bhaskarjit Sarmah, Benika Hall, Rohan Rao, Sunil
Patel, Stefano Pasquali, and Dhagash Mehta. Hy-
bridrag: Integrating knowledge graphs and vector
retrieval augmented generation for efficient infor-
mation extraction, 2024.
[2]Patrick Lewis, Ethan Perez, Aleksandra Piktus,
Fabio Petroni, Vladimir Karpukhin, Naman Goyal,
Pavel Kuksa, and et al. Retrieval-augmented gen-
eration for knowledge-intensive NLP tasks. In
Advances in Neural Information Processing Systems
(NeurIPS), 2020.
[3]Yunfan Gao, Yun Xiong, Xinyu Gao, Kangxiang
Jia, Jinliu Pan, Yuxi Bi, Yi Dai, Jiawei Sun, Meng
Wang, and Haofen Wang. Retrieval-augmented
generation for large language models: A survey,
2024.
[4]Hengran Zhang, Minghao Tang, Keping Bi, Ji-
afeng Guo, Shihao Liu, Daiting Shi, Dawei Yin,
and Xueqi Cheng. Utility-focused llm annotation
for retrieval and retrieval-augmented generation,
2025.
[5]Keshav Santhanam, Omar Khattab, Jon Saad-
Falcon, Christopher Potts, and Matei Zaharia.
Colbertv2: Effective and efficient retrieval via
lightweight late interaction, 2022.
[6]Lianmin Zheng, Wei-Lin Chiang, Ying Sheng,
Siyuan Zhuang, Zhanghao Wu, Yonghao Zhuang,
Zi Lin, Zhuohan Li, Dacheng Li, Eric P. Xing,
Hao Zhang, Joseph E. Gonzalez, and Ion Stoica.
Judging llm-as-a-judge with mt-bench and chatbot
arena, 2023.[7]Akari Asai, Zeqiu Wu, Yizhong Wang, Avirup Sil,
and Hannaneh Hajishirzi. Self-rag: Learning
to retrieve, generate, and critique through self-
reflection, 2023.