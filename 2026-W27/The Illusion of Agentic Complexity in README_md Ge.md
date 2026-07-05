# The Illusion of Agentic Complexity in README.md Generation: Evaluating Single-Agent vs. Multi-Agent RAG Systems

**Authors**: Abu Saleh, Tesfay Welegebreal Tesfay, Phuong T. Nguyen, Juri Di Rocco, Muhammad Umar Zeshan, Davide Di Ruscio

**Published**: 2026-06-29 16:29:31

**PDF URL**: [https://arxiv.org/pdf/2606.30524v2](https://arxiv.org/pdf/2606.30524v2)

## Abstract
Large Language Models (LLMs) are increasingly utilized to automate several software engineering tasks, including code completion, code summarization, testing, and the generation of repository-level documentation. While Multi-Agent Systems (MAS) are often adopted to support such tasks under the premise that task decomposition improves performance, the impact of architectural complexity on practical efficiency remains under-examined. This study empirically evaluates Retrieval-Augmented Generation (RAG) dependent architectures for the generation of README files for GitHub repositories. In this work, we conducted a systematic comparison between a Single-Agent pipeline, a specialized MAS, and a developer-guided planning (DevPlan) variant, benchmarked against LARCH -- a state-of-the-art baseline -- and the original ground truth. Results indicate a critical architectural trade-off: the Single-Agent pipeline achieves lexical quality comparable to MAS while reducing token consumption by 86% and operating at twice the speed. In contrast, manual taxonomy analysis demonstrates that MAS achieves high structural consistency (98%), resolving formatting issues observed in single-agent approaches. Autonomous planning is identified as the primary pipeline bottleneck; incorporating lightweight developer-guided plans produces the highest overall documentation quality, surpassing all the analyzed configurations.

## Full Text


<!-- PDF content starts -->

The Illusion of Agentic Complexity in
README.md Generation: Evaluating Single-Agent
vs. Multi-Agent RAG Systems
Abu Saleh∗†, Tesfay Welegebreal Tesfay∗†, Phuong T. Nguyen∗,
Juri Di Rocco∗, Muhammad Umar Zeshan∗‡, Davide Di Ruscio∗
∗University of L’Aquila, L’Aquila, Italy
†˚Abo Akademi University, Turku, Finland
‡University of Pisa, Pisa, Italy
{abu.saleh, tesfaywelegebreal.tesfay, muhammadumar.zeshan}@student.univaq.it,
{phuong.nguyen, juri.dirocco, davide.diruscio}@univaq.it
Abstract—Large Language Models (LLMs) are increasingly
utilized to automate several software engineering tasks, including
code completion, code summarization, testing, and the generation
of repository-level documentation. While Multi-Agent Systems
(MAS) are often adopted to support such tasks under the premise
that task decomposition improves performance, the impact of
architectural complexity on practical efficiency remains under-
examined. This study empirically evaluates Retrieval-Augmented
Generation (RAG) dependent architectures for the generation
of README files for GitHub repositories. In this work, we con-
ducted a systematic comparison between a Single-Agent pipeline,
a specialized MAS, and a developer-guided planning (Dev-
Plan) variant, benchmarked against LARCH–a state-of-the-art
baseline–and the original ground truth. Results indicate a critical
architectural trade-off: the Single-Agent pipeline achieves lexical
quality comparable to MAS while reducing token consumption
by 86% and operating at twice the speed. In contrast, manual
taxonomy analysis demonstrates that MAS achieves high struc-
tural consistency (98%), resolving formatting issues observed in
single-agent approaches. Autonomous planning is identified as the
primary pipeline bottleneck; incorporating lightweight developer-
guided plans produces the highest overall documentation quality,
surpassing all the analyzed configurations.
I. INTRODUCTION
Software repositories hosted on platforms such as GitHub
rely heavily on README files to communicate project pur-
poses, installation procedures, usage examples, and contri-
bution guidelines. As the primary entry point for users and
contributors, README plays a critical role in project dis-
covery and adoption. However, README writing remains a
manual and often inconsistent process, frequently resulting in
incomplete, outdated, or poorly structured documentation [9].
Recent advances in Large Language Models (LLMs) have
significantly improved the ability to generate natural-language
text from source code, enabling automated documentation at
scale [1]. Empirical studies show that LLMs can generate
code documentation with improved readability and semantic
accuracy compared to traditional approaches [8], [26]. These
capabilities have motivated growing interest in applying LLMs
to higher-level documentation tasks, including repository-level
summarization and generation of README files.Koreeda et al. [16] proposed LARCH, a heuristic-guided
pipeline for automatic README generation, whereas RepoA-
gent by Luo et al. [20] introduces a framework that leverages
repository structure and inter-file dependencies to improve
documentation coverage. More recent approaches, such as Re-
poSummary [37], emphasize feature-oriented summarization
to better align generated documentation with user-relevant
functionality. Additionally, frameworks like ReadMe.LLM
[33] suggest that documentation itself can be optimized to
better support both human users and machine understanding.
There has been an increasing interest in multi-agent LLM
systems. Frameworks such as AutoGen [34], MetaGPT [12],
or ChatDev [25] decompose software engineering (SE) work-
flows into interacting roles, enabling iterative refinement and
division of labor. This paradigm assumes that tasks such
as README generation benefit from modular task decom-
position into subtasks. However, recent evidence suggests a
single capable LLM can achieve high-quality repository-level
reasoning simply by incorporating structural and dependency
cues into a unified context window [28].
Despite the contrasting nature with their own advan-
tages and limitations, single-agent and multi-agent paradigms
have not been empirically compared on the specific task of
README generation. Such an investigation is particularly
important because README files require not only accurate
summarization but also coherent organization, consistent style,
and user-oriented presentation properties that may be sensitive
to fragmentation across multiple agents [10], [23].
This work supports a call to critically evaluate the necessity
of agentic complexity in SE with the following contributions:
▷Comparative Framework.A controlled experimen-
tal setting was designed to investigate single- and multi-
agent approaches for README generation. Both approaches
were evaluated under identical repository contexts to deter-
mine whether modular decomposition outperforms monolithic
LLMs. To the best of our knowledge, this work represents
the first systematic comparison of agentic architectures for
automated README generation.arXiv:2606.30524v2  [cs.SE]  1 Jul 2026

▷Evaluation.An evaluation of the generated documentation
was conducted using standardized assessment methodologies,
including manual human inspection. This evaluation explicitly
focuses on macro-level coherence, structural consistency, and
overall quality of repository-level documentation.
▷Comparison.We benchmarked the single-agent and multi-
agent approaches against LARCH [16], the current state-of-
the-art technique, and against the ground truth represented by
the original README files.
▷Open Science.A complete replication package with code
and data has been published to facilitate future research [2].
II. RELATEDWORK
Automatic README generation has gained attention due
to its role in improving software usability and reuse. More
recent work has explored automatic README generation
using LLMs [7], [16]. LARCH [16] is an LLM-based system
that generates README files by selecting representative code
fragments using heuristic methods and synthesizing them
into natural language descriptions. LARCH demonstrates that
grounding generation on salient code elements significantly
improves README quality, but relies on fixed heuristics and
single-pass generation. RMGenie [7] advances this direction
by introducing an agent-based framework for README gen-
eration, where an LLM agent iteratively analyzes repository
structure, invokes tools, and refines README content through
multi-step reasoning. In addition to agent- and LLM-based ap-
proaches, fine-tuning and retrieval-augmented strategies have
been applied to automated README generation. For instance,
ReadmeReady [5] uses fine-tuned open-source LLMs and code
indexing to produce basic README content.
Multi-agent approaches have also been employed for soft-
ware documentation tasks. DocAgent [35] leverages multiple
specialized agents to collaboratively generate code documenta-
tion, demonstrating improved completeness and factual accu-
racy through iterative reasoning and verification. Metagente
[23] deploys LLM-based MAS to GitHub README sum-
marization, where cooperating agents iteratively evaluate and
refine summaries, outperforming single-agent baselines.
The usage of LLMs has been also investigated for code
summarization and automated documentation [24], [14]. Prior
work has shown that LLMs can effectively support code
understanding by generating natural language explanations and
answering questions about code behavior [18], [22]. Early
work such as [15] demonstrated that GPT-3 can generate
function- and module-level documentation across multiple
programming languages. Other approaches, such as context-
aware summarization, use broader program structure to gen-
erate more accurate and informative descriptions [29]. At the
repository level, RepoAgent [20] introduces an LLM-based
framework for generating and maintaining structured code
documentation through global repository analysis.
Unlike prior work, which focuses on code documentation
[35] or summarization [23], our approach aims to com-
pare single- and multi-agent–based approaches to generate
README files starting from the source code of the projects.
Repo Proﬁler AgentREADME Planner
Agent
Reviewer Agent Aggregator AgentCore W riter
Optional W riterApprovedGenerated
README.md
NoYes    Orchestrator (MAS/Dev)
 Single-AgentTarget Repo
Evaluation
LARCH MAS Single-Agent Dev-PlanBertScoreRougeManual Taxonomy
Coverage
LLM-as-judgeOriginal  READMELibrarian Indexing
Pattern Generator
Vector DB
GenerationRepo Preparation
Semantic
IndexingFig. 1. Overview of the proposed README generation pipeline, from
repository preparation and indexing to generation and evaluation.
III. METHODOLOGY
In this section, we describe the process that we developed
to support the comparison of single and multi-agent systems
for the generation of README files. Figure 1 presents an
overview of the process, which includes theRepository
Preparation,Semantic Indexing,Generation,
andEvaluation Pipelines, described below.1
A. Repository Preparation
Before generation, the framework prepares the repository
for downstream processing and extracts structural patterns
from original README files.
▷Acquisition and Isolation. The originalREADME.mdis
extracted as ground truth and excluded from the pipeline
to prevent access to it during generation. All other Mark-
down files are similarly removed prior to vector database
construction. This prevents system from accessing reference
documentation, ensuring that generated READMEs are derived
solely from source code and configuration files.
▷Pattern Extraction. A canonical README structure is
derived by extractingH1,H2headings from ground-truth
README . These headings are aggregated across repositories
and analyzed using an LLM to identify frequency and semantic
patterns. The final output is a JSON file that presents a
taxonomy of section headers and their recommended sequence.
B. Semantic Indexing
To distill raw source code into a queryable format without
overwhelming LLM context windows, we employ an intelli-
gent, programming language-aware ingestion pipeline.
▷Intelligent File Selection. A static scanner generates a
repository file tree, which is evaluated by a specialized
Librarianagent, which isolates essential files (e.g., entry
points, configuration manifests, core logic), filtering out noise
to significantly reduce embedding overhead.
▷Segmentation and Indexing. To preserve structural integrity
during ingestion, code files are dynamically mapped to their
programming language and partitioned using syntax-aware
1Due to the strict space limit, we cannot show the prompts and represen-
tative intermediate artifacts here. Interested readers are kindly referred to the
replication package available online [2].

recursive splitters. The splitter divides chunks at logical se-
mantic boundaries (e.g., classes or function blocks) rather
than arbitrarily mid-statement. Generic text and configuration
files default to a standard character splitter. These chunks are
vectorized and persisted in isolated collections within a local
vector database. Following prior RAG methods for code com-
prehension [7], [20], downstream agents query this index using
diversity-aware semantic search, ensuring a comprehensive,
non-redundant representation of the codebase.
C. Generation Pipelines
Utilizing the previously constructed vector databases to
provide repository-level context, we implemented single-agent
and multi-agent generative pipelines. Furthermore, a human-
in-the-loop phase is introduced within the multi-agent systems
to evaluate the impact of human oversight on output quality.
1) Single-Agent Pipeline:To test whether monolithic gen-
eration can succeed given sufficient context, the first paradigm
implements a unified RAG approach. The system performs
a predetermined set of semantic retrieval queries on the
vector store. These queries are based on the section taxonomy
generated during the pattern extraction phase, with each query
formulated as a keyword-level expansion of its respective
section type. The retrieved code chunks are concatenated into
a single context window. A single LLM is then prompted to
generate the fully formatted README with one single query.
2) MAS Pipeline:In addition, we designed a multi-agent
workflow supervised by a centralOrchestratorthat routes
execution through four phases:
▷Profiling and Planning. TheRepo Profileragent gen-
erates a structured project profile by combining the repository
file tree with retrieved context from its repository-specific
vector database. The profile summarizes README-relevant
information such as project type, dependencies, commands,
and key features. Afterwards, theREADME Planneragent
combines this profile with theREADME Patternextracted
during the pattern extraction step to produce a JSON file
specifying the ordered section structure of the README and
section-level instructions, which is used to guide writer agents.
▷Section-Level Generation. TheOrchestratordele-
gates drafting tasks to specialized writing agents: aCore
Writerfor technical sections (e.g., usage, installation) and
anOptional Writerfor administrative sections (e.g.,
licenses). Each writer performs localized retrieval using the
Planner’s instructions to obtain section-specific source evi-
dence from the repository-specific vector database.
▷Iterative Review. Drafts are routed to aRevieweragent,
which executes queries against the vector database to verify
the factual correctness of the generated content. The retrieved
source code chunks serve as the ground-truth reference for
assessing the technical accuracy of the drafted text. Sections
with inaccurate or incomplete content are returned to the
writers with a critique for a rewrite.
▷Artifact Aggregation. Once all sections pass review, an
Aggregatoragent merges them into a contiguous docu-ment, ensuring stylistic cohesion and stripping out any redun-
dantly generated command blocks.
3) Developer-Guided Planning (Dev-Plan):We introduced
a developer-guided variant of the MAS pipeline to deter-
mine whether performance limitations arise from the plan-
ning stage, since LLMs frequently struggle with strategic
planning in multi-step workflows [31]. In this configuration,
theREADME Planneragent is omitted, and its output is
replaced with a human-authored planning artifact. The human
developer creates a JSON file that adheres to thePlanner’s
JSON schema, but contains a manually curated sequence of
sections designed to meet repository-specific requirements.
This artifact is provided before execution to directly substitutes
thePlanner’s output. The downstream MAS components
(section-level generation, iterative review, and aggregation)
remain unchanged. This design enables evaluation of MAS
performance under human-planned conditions.
IV. EXPERIMENTS
A. Research questions
We answer the following research questions (RQs).
▷RQ 1: How do single-agent and multi-agent architectures
compare with LARCH in terms of README quality
and computational cost? (Efficacy and Efficiency): This
RQ evaluates the central goal of the paper: comparing a
single-agent RAG pipeline, a multi-agent RAG pipeline, and
LARCH [16] as the state-of-the-art baseline for README
generation. Since LARCH generates documentation by sum-
marizing a single heuristically selected code file, we assess
whether repository-level RAG pipelines improve effectiveness
and efficiency by overcoming the information loss inherent
in LARCH’s restricted single-file methodology, and whether
added multi-agent complexity yields measurable benefits over
a simpler single-agent design.
▷RQ 2: How do the generated README files com-
pare with the original ones in terms of taxonomic cov-
erage? (Structural Consistency): This RQ evaluates how
closely the generated READMEs align with the original ones
in terms of structural completeness and organizational quality.
To this end, we compare the outputs of the single-agent and
multi-agent approaches against the original README.md files
(golden reference) using manual taxonomy-based coverage
analysis and LLM-as-a-Judge.
B. Dataset and Evaluation Metrics
We considered repositories created after August 2025 to
reduce data contamination [4], [6]. Following common repos-
itories filtering practices, we retained non-fork, non-archived
software repositories with more than 500 stars, disk size
between 100 and 10,000 KB, at most 50 source files, and
READMEs longer than 1,000 characters; we also excluded
tutorials and “awesome-*” lists [3], [21]. The final dataset
contains 180 GitHub repositories (118 Python, 49 JavaScript,
13 Go). Though this is a considerably small number, resulting
in an evaluation sample size aligned with recent studies [7],
[35]. From this corpus, we randomly selected 20 repositories

for evaluating the developer-guided planning (Dev-Plan) con-
figuration, consistent with qualitative study sizes [20], [30].
To evaluate generated content, we use ROUGE and
BERTScore, which are widely adopted in documentation
generation [12], [25]. However, these metrics do not capture
structural properties such as organizational flow and consis-
tency [11], [13]. Addressing this limitation, we define an eval-
uation taxonomy of 12 commonly recommended README
sections [2], derived from prior literature [19], [32]. This
literature-derived taxonomy is used only for evaluation, while
the dataset-derived Pattern described in Section III-A is used
only for generation. Using this taxonomy as the evaluation
criteria, we employ an LLM-as-a-judge [36], which better
assesses structural quality and usefulness [35]. READMEs
are scored on a 10-point scale based on completeness and
adherence to the derived taxonomy. Scores are averaged over
four randomized runs [27] to mitigate bias in judging. We
additionally perform a manual analysis of coverage of those
derived taxonomies, which is reported in the results section.
C. Experimental Settings
We utilizegpt-5.1for all generative and reasoning
tasks to ensure a fair comparison. Both workflow and API
interactions are implemented using LangChain [17]. Source
code is partitioned using a language-aware recursive splitter
to preserve syntactic boundaries. Chunks are vectorized via
text-embedding-3-smalland stored in ChromaDB.
To guarantee computational tractability in MAS, the itera-
tive refinement loop is strictly capped. If a drafted section
fails the Reviewer agent’s critique three consecutive times,
theOrchestratoraccepts the final attempt. While this
guarantees pipeline termination and bounds token costs, we
acknowledge it may occasionally retain suboptimal content.
V. RESULTS ANDDISCUSSION
A. Result Analysis
▷RQ 1: How do single-agent and multi-agent architectures
compare with LARCH in terms of README quality and
computational cost?
The generated READMEs were evaluated across 180 repos-
itories using ROUGE, BERTScore, and token consumption
metrics (Figs. 2 and 4). The experiments challenge the assump-
tion that increased multi-agent complexity inherently produces
superior generation. The single-agent pipeline slightly outper-
forms the fully autonomous MAS in ROUGE-L F1 (0.2007 vs.
0.1964), with negligible differences observed in BERTScore.
Both pipelines significantly outperform LARCH. Specifically,
LARCH’s ROUGE-L F1 decreased substantially to 0.1022.
This suggests that LARCH’s single-file heuristic effectively
extracts keywords but does not capture macro-level coherence.
The data indicate a significant operational penalty associated
with MAS. Single-agent configuration required an average of
7,840 tokens and 40 seconds to generate each README file.
In contrast, MAS consumed 56,242 tokens, a 7.1x increase
driven entirely by prompt overhead, and required 78 seconds.Given the Single-agent’s competitive performance, we hypoth-
esized that the autonomous MAS is constrained by cascading
errors originating arising from generic AI-generated outlines.
To examine this, the Dev-Plan configuration was evaluated, as
shown in Fig. 3. Dev-Plan achieved the highest overall scores
(ROUGE-L F1: 0.2323, BERTScore F1: 0.8230). However,
this approach incurs substantial computational cost (Fig. 5).
Because human-authored plans contain more granular section
specifications, Dev-Plan triggers more section-level retrieval,
generation, and review steps, consuming 79,196 tokens and
148 seconds on average.
Answer to RQ 1:Both the single-agent and multi-agent pipelines
outperform LARCH in README generation. The single-agent
approach is the most cost-efficient solution, matching MAS quality
while using 86% fewer tokens. However, Dev-Plan successfully
resolves the AI-generated outlines limitation in the MAS pipeline
and achieves the highest overall quality.
▷RQ 2: How do the generated README files compare
with the original ones in terms of taxonomic coverage?
TABLE I
STRUCTURALEVALUATION: MANUALTAXONOMYCOVERAGE VS.
LLM-AS-A-JUDGE(N= 20).
Manual Coverage Analysis LLM-as-a-Judge
Method Precision Recall F1-Score Mean Score1st Place
Win RateFail Rate
(Score≤5)
Dev-Plan 0.8610.7500.8018.60 53.75% 1.25%
MAS0.9820.6960.8147.55 16.25% 2.50%
Single-Agent 0.776 0.492 0.602 7.25 10.00% 6.25%
Golden (Original) 0.724 0.667 0.694 7.36 20.00% 17.50%
A dual evaluation was performed on a dataset consisting
of 20 repositories (Table I) to assess practical utility beyond
n-gram overlap. The initial evaluation consists of a manual
analysis of section coverage. For each README, we manually
mapped its sections to the 12 literature-derived taxonomy, to
calculate structural Precision, defined as the proportion of gen-
erated sections that match the taxonomy, and Recall, defined
as the proportion of taxonomy sections covered. In the second
evaluation, an LLM-as-a-Judge assessed overall usefulness on
a 10-point scale using the same taxonomy as reference. This
combined analysis identifies a significant limitation in standard
lexical metrics. While the single-agent approach demonstrated
strong performance in ROUGE (RQ1), it underperformed in
structural generation, achieving only 49.2% manual Recall
and ranking lowest in LLM-as-a-Judge (Mean: 7.25). Without
the strict step by step delegation like multi-agent workflow,
single-prompt LLMs produce monolithic, poorly segmented
text blocks that omit essential documentation components.
The autonomous MAS demonstrated substantial structural
discipline, achieving a manual Precision of 98.2%. This sug-
gests that multi-agent routing effectively eliminates formatting
limitations and irrelevant content. The LLM-as-a-Judge metric
reflected this organizational strength, assigning MAS higher
scores than the Single-Agent and recording a consistently low
2.50% failure rate. The original README (Golden) was also
evaluated against the literature-derived taxonomy, resulting

Fig. 2. ROUGE/BERTScore: MAS, LARCH, Single-Agent.
 Fig. 3. ROUGE/BERTScore: MAS, Dev-Plan, Single-Agent.
Fig. 4. Token usage: MAS, LARCH, Single-Agent.
 Fig. 5. Token usage: MAS, Dev-Plan, Single-Agent.
in lower structural Precision (72.4%) and a failure rate of
17.50%, primarily due to project-specific sections and omitted
setup or prerequisite instructions. Dev-Plan demonstrated the
most comprehensive performance. Incorporating human plan-
ning enabled the system to achieve the highest structural Recall
(75.0%) and the highest LLM-as-a-Judge scores (Mean: 8.60),
while nearly eliminating structural failures (1.25%).
Answer to RQ 2:Single-agent approaches exhibit limited structural
recall, often omitting critical project dimensions. In contrast, the
MAS maintains strict adherence to formatting standards, achieving
98.2% precision. The integration of human strategic planning with
AI execution (DevPlan) produces the most comprehensive docu-
mentation. These automated systems achieve significantly higher
coverage of the elicited taxonomy compared to the README files
authored by human developers.
B. Discussion
▷Statistical Validation.Wilcoxon signed-rank test with
Holm-Bonferroni correction confirms the superiority of agen-
tic architectures.2Both Dev-Plan and MAS achieved signif-
icantly higher F1 scores compared to the Golden human-
authored baseline (p <0.01). Furthermore, while Single-
Agent performance was comparable to human effort in pre-
cision (p= 0.778), the MAS approach demonstrated a sta-
tistically significant leap in formatting precision (p <0.001),
validating the effectiveness of modular decomposition.
▷Applicability.The evaluation identifies a critical trade-off:
Single-Agent pipelines optimize cost-efficiency, whereas MAS
ensures structural completeness, exposing the limitations of
2Details of the statistical analysis is available in the replication package [2]lexical metrics for documentation assessment. Notably, the
performance of Dev-Plan indicates that AI planning constitutes
the primary bottleneck of fully autonomous systems, as generic
or inadequate document outlines constrain the quality of down-
stream drafting agents. These findings suggest a necessary
shift toward human-AI collaboration. Positioning developers
as strategic architects and LLMs as execution engines enables
practitioners to balance computational costs while consistently
generating industry-standard documentation.
▷Limitations.The Librarian agent filters the repository solely
based on file names and directory structures to reduce token
overhead. This heuristic may exclude essential logic present in
files with non-descriptive names. Although the Dev-Plan con-
figuration demonstrates the highest performance, it depends
on manually supplied blueprints and was assessed using a
smaller dataset. The proposed approach emphasizes structural
quality, which may not adequately address the nuances of
highly domain-specific documentation. Due to data collection
challenges, our evaluation was conducted on 180 repositories.
Future work should validate our approach on larger datasets.
▷Threats to Validity.(i)Internal validity: LARCH was
adapted to usegpt-5.1instead of its originalgpt-3imple-
mentation to ensure a fair comparison. (ii)External validity:
To prevent data leakage, only repositories created after August
2025 were included. Though the findings are valid for this
diverse dataset, they may not fully generalize to closed-source
enterprise monoliths or unconventional codebases.
VI. CONCLUSION ANDFUTUREWORK
This study empirically evaluates RAG-augmented architec-
tures for repository-level README generation, questioning

the necessity of complex MAS. The findings indicate that
a single-agent architecture achieves quality moderately com-
parable to MAS, reducing token costs, outperforming the
baseline. Integrating human planning with AI results in the
highest quality. For future work, we plan to study the appli-
cability of agentic architectures in other domains in Software
Engineering, such as automated bug fixing or test generation.
ACKNOWLEDGMENT
This paper has been partially supported by the MOSAICO
project (Management, Orchestration and Supervision of AI-
agent COmmunities for reliable AI in software engineering)
that has received funding from the European Union under the
Horizon Research and Innovation Action (Grant Agreement
No. 101189664). This work was partially supported by the
Erasmus+ Alliance for Innovation project AIM-PRO–AI Lit-
eracy for Multidisciplinary Professional Readiness and Out-
reach (Grant Agreement No. 101245864) and by Movetia, the
Swiss agency promoting exchange, mobility and cooperation
in education, continuing education and youth work.
REFERENCES
[1] Josh Achiam, Steven Adler, Sandhini Agarwal, Lama Ahmad, Ilge
Akkaya, Florencia Leoni Aleman, Diogo Almeida, Janko Altenschmidt,
Sam Altman, Shyamal Anadkat, et al. Gpt-4 technical report.arXiv
preprint arXiv:2303.08774, 2023.
[2] Anonymous Authors. Replication package for The Illusion of Agentic
Complexity in README.md Generation: Evaluating Single-Agent vs.
Multi-Agent RAG. https://anonymous.4open.science/r/ML4SE-8268/
README.md, 2026.
[3] Hudson Borges, Andre Hora, and Marco Tulio Valente. Understanding
the factors that impact the popularity of github repositories. In2016
IEEE international conference on software maintenance and evolution
(ICSME), pages 334–344. IEEE, 2016.
[4] Laura Caspari, Alexander Trautsch, Michael Granitzer, and Steffen
Herbold. Studying memorization of large language models using
answers to stack overflow questions.Transactions on Machine Learning
Research, 2025.
[5] Sayak Chakrabarty and Souradip Pal. Readmeready: Free and customiz-
able code documentation with llms-a fine-tuning approach.Journal of
Open Source Software, 10(108):7489, 2025.
[6] Yuxing Cheng, Yi Chang, and Yuan Wu. A survey on data contamination
for large language models.arXiv preprint arXiv:2502.14425, 2025.
[7] Xing Cui, Jingzheng Wu, Zhiyuan Li, Tianyue Luo, and Xiang Ling.
Rmgenie: An llm-based agent framework for open source software
readme generation. In2025 IEEE International Conference on Software
Maintenance and Evolution (ICSME), pages 505–516. IEEE, 2025.
[8] Shubhang Shekhar Dvivedi, Vyshnav Vijay, Sai Leela Rahul Pujari,
Shoumik Lodh, and Dhruv Kumar. A comparative analysis of large
language models for code documentation generation. InProceedings of
the 1st ACM international conference on AI-powered software, pages
65–73, 2024.
[9] Haoyu Gao, Hong Yi Lin, Christoph Treude, Gregory Gay, and Man-
sooreh Zahedi. Does my readme file need to be updated? exploring
llm-based readme maintenance.arXiv preprint arXiv:2603.00489, 2026.
[10] Mingyan Gao, Yanzi Li, Banruo Liu, Yifan Yu, Phillip Wang, Ching-Yu
Lin, and Fan Lai. Single-agent or multi-agent systems? why not both?
arXiv preprint arXiv:2505.18286, 2025.
[11] Anh Nguyen Hoang, Minh Le-Anh, Bach Le, and Nghi DQ Bui.
Codewiki: Evaluating ai’s ability to generate holistic documentation for
large-scale codebases.arXiv preprint arXiv:2510.24428, 2025.
[12] Sirui Hong, Mingchen Zhuge, Jonathan Chen, Xiawu Zheng, Yuheng
Cheng, Jinlin Wang, Ceyao Zhang, Zili Wang, Steven Ka Shing Yau,
Zijuan Lin, et al. Metagpt: Meta programming for a multi-agent
collaborative framework. InThe twelfth international conference on
learning representations, 2023.[13] Xing Hu, Qiuyuan Chen, Haoye Wang, Xin Xia, David Lo, and Thomas
Zimmermann. Correlating automated and human evaluation of code
documentation generation quality.ACM Transactions on Software
Engineering and Methodology (TOSEM), 31(4):1–28, 2022.
[14] Juyong Jiang, Fan Wang, Jiasi Shen, Sungju Kim, and Sunghun Kim. A
survey on large language models for code generation.ACM Transactions
on Software Engineering and Methodology, 35(2):1–72, 2026.
[15] Junaed Younus Khan and Gias Uddin. Automatic code documentation
generation using gpt-3. InProceedings of the 37th IEEE/ACM Inter-
national Conference on Automated Software Engineering, pages 1–6,
2022.
[16] Yuta Koreeda, Terufumi Morishita, Osamu Imaichi, and Yasuhiro So-
gawa. Larch: Large language model-based automatic readme creation
with heuristics. InProceedings of the 32nd ACM International Con-
ference on Information and Knowledge Management, pages 5066–5070,
2023.
[17] LangChain. LangChain. https://www.langchain.com/, 2026. [Online;
accessed May 6, 2026].
[18] Juho Leinonen, Paul Denny, Stephen MacNeil, Sami Sarsa, Seth Bern-
stein, Joanne Kim, Andrew Tran, and Arto Hellas. Comparing code
explanations created by students and large language models. InProceed-
ings of the 2023 Conference on Innovation and Technology in Computer
Science Education V . 1, pages 124–130, 2023.
[19] Yuyang Liu, Ehsan Noei, and Kelly Lyons. How readme files are
structured in open source java projects.Information and Software
Technology, 148:106924, 2022.
[20] Qinyu Luo, Yining Ye, Shihao Liang, Zhong Zhang, Yujia Qin, Yaxi
Lu, Yesai Wu, Xin Cong, Yankai Lin, Yingli Zhang, et al. Repoagent:
An llm-powered open-source framework for repository-level code doc-
umentation generation.arXiv preprint arXiv:2402.16667, 2024.
[21] Nuthan Munaiah, Steven Kroh, Craig Cabrey, and Meiyappan Nagappan.
Curating github for engineered software projects.Empirical Software
Engineering, 22(6):3219–3253, 2017.
[22] Daye Nam, Andrew Macvean, Vincent Hellendoorn, Bogdan Vasilescu,
and Brad Myers. Using an llm to help with code understanding.
InProceedings of the IEEE/ACM 46th International Conference on
Software Engineering, pages 1–13, 2024.
[23] Duc SH Nguyen, Bach G Truong, Phuong T Nguyen, Juri Di Rocco, and
Davide Di Ruscio. Teamwork makes the dream work: Llms-based agents
for github readme. md summarization. InProceedings of the 33rd ACM
International Conference on the Foundations of Software Engineering,
pages 621–625, 2025.
[24] Amirkia Rafiei Oskooei, Selcan Yukcu, Mehmet Cevheri Bozoglan, and
Mehmet S Aktas. Repository-level code understanding by llms via
hierarchical summarization: Improving code search and bug localiza-
tion. InInternational Conference on Computational Science and Its
Applications, pages 88–105. Springer, 2025.
[25] Chen Qian, Wei Liu, Hongzhang Liu, Nuo Chen, Yufan Dang, Jiahao
Li, Cheng Yang, Weize Chen, Yusheng Su, Xin Cong, et al. Chatdev:
Communicative agents for software development. InProceedings of the
62nd annual meeting of the association for computational linguistics
(volume 1: Long papers), pages 15174–15186, 2024.
[26] Swapnil Sharma Sarker and Tanzina Taher Ifty. Automated and context-
aware code documentation leveraging advanced llms. InProceedings of
the 18th International Natural Language Generation Conference, pages
486–498, 2025.
[27] Lin Shi, Chiyu Ma, Wenhua Liang, Xingjian Diao, Weicheng Ma, and
Soroush V osoughi. Judging the judges: A systematic study of position
bias in llm-as-a-judge. InProceedings of the 14th International Joint
Conference on Natural Language Processing and the 4th Conference of
the Asia-Pacific Chapter of the Association for Computational Linguis-
tics, pages 292–314, 2025.
[28] Disha Shrivastava, Hugo Larochelle, and Daniel Tarlow. Repository-
level prompt generation for large language models of code. InInterna-
tional Conference on Machine Learning, pages 31693–31715. PMLR,
2023.
[29] Chia-Yi Su, Aakash Bansal, Yu Huang, Toby Jia-Jun Li, and Collin
McMillan. Context-aware code summary generation.arXiv preprint
arXiv:2408.09006, 2024.
[30] Wei Tao, Yucheng Zhou, Yanlin Wang, Wenqiang Zhang, Hongyu
Zhang, and Yu Cheng. Magis: Llm-based multi-agent framework for
github issue resolution.Advances in Neural Information Processing
Systems, 37:51963–51993, 2024.

[31] Lei Wang, Chen Ma, Xueyang Feng, Zeyu Zhang, Hao Yang, Jingsen
Zhang, Zhiyuan Chen, Jiakai Tang, Xu Chen, Yankai Lin, et al. A
survey on large language model based autonomous agents.Frontiers of
Computer Science, 18(6):186345, 2024.
[32] Tianlei Wang, Shaowei Wang, and Tse-Hsun Peter Chen. Study
the correlation between the readme file of github projects and their
popularity.Journal of Systems and Software, 205:111806, 2023.
[33] Sandya Wijaya, Jacob Bolano, Alejandro Gomez Soteres, Shriyanshu
Kode, Yue Huang, and Anant Sahai. Readme. llm: A framework to help
llms understand your library.arXiv preprint arXiv:2504.09798, 2025.
[34] Qingyun Wu, Gagan Bansal, Jieyu Zhang, Yiran Wu, Beibin Li, Erkang
Zhu, Li Jiang, Xiaoyun Zhang, Shaokun Zhang, Jiale Liu, et al. Autogen:
Enabling next-gen llm applications via multi-agent conversations. In
First conference on language modeling, 2024.
[35] Dayu Yang, Antoine Simoulin, Xin Qian, Xiaoyi Liu, Yuwei Cao,
Zhaopu Teng, and Grey Yang. Docagent: A multi-agent system for auto-
mated code documentation generation.arXiv preprint arXiv:2504.08725,
2025.
[36] Lianmin Zheng, Wei-Lin Chiang, Ying Sheng, Siyuan Zhuang, Zhang-
hao Wu, Yonghao Zhuang, Zi Lin, Zhuohan Li, Dacheng Li, Eric Xing,
et al. Judging llm-as-a-judge with mt-bench and chatbot arena.Advances
in neural information processing systems, 36:46595–46623, 2023.
[37] Yifeng Zhu, Xianlin Zhao, Xutian Li, Yanzhen Zou, Haizhuo Yuan, Yue
Wang, and Bing Xie. Reposummary: Feature-oriented summarization
and documentation generation for code repositories.arXiv preprint
arXiv:2510.11039, 2025.