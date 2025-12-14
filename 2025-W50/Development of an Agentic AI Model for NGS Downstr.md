# Development of an Agentic AI Model for NGS Downstream Analysis Targeting Researchers with Limited Biological Background

**Authors**: Donghyeon Lee, Dongseok Kim, Seokhwan Ko, Seo-Young Park, Junghwan Cho

**Published**: 2025-12-10 03:43:46

**PDF URL**: [https://arxiv.org/pdf/2512.09964v1](https://arxiv.org/pdf/2512.09964v1)

## Abstract
Next-Generation Sequencing (NGS) has become a cornerstone of genomic research, yet the complexity of downstream analysis-ranging from differential expression gene (DEG) identification to biological interpretations-remains a significant barrier for researchers lacking specialized computational and biological expertise. While recent studies have introduced AI agents for RNA-seq analysis, most focus on general workflows without offering tailored interpretations or guidance for novices. To address this gap, we developed an Agentic AI model designed to automate NGS downstream analysis, provide literature-backed interpretations, and autonomously recommend advanced analytical methods. Built on the Llama 3 70B Large Language Model (LLM) and a Retrieval-Augmented Generation (RAG) framework, the model is deployed as an interactive Streamlit web application. The system integrates standard bioinformatics tools (Biopython, GSEApy, gProfiler) to execute core analyses, including DEG identification, clustering, and pathway enrichment. Uniquely, the agent utilizes RAG to query PubMed via Entrez, synthesizing biological insights and validating hypotheses with current literature. In a case study using cancer-related dataset, the model successfully identified significant DEGs, visualized clinical correlations, and derived evidence-based insights (e.g., linking BRAF mutations to prognosis), subsequently executing advanced survival modeling upon user selection. This framework democratizes bioinformatics by enabling researchers with limited backgrounds to seamlessly transition from basic data processing to advanced hypothesis testing and validation.

## Full Text


<!-- PDF content starts -->

Development of an Agentic AI Model for
NGS Downstream Analysis Targeting
Researchers with Limited Biological
Background
Donghyeon Lee1,2, Dongseok Kim1,2, Seokhwan Ko3, Seo-Young
Park1,2, *, and Junghwan Cho3, *
1Department of Biomedical Science, Kyungpook National University,
Daegu, Republic of Korea
2BK21 Four Program, School of Medicine, Kyungpook National
University, Daegu, Republic of Korea
3Clinical Omics Institute, Kyungpook National University, Daegu,
Republic of Korea
∗Corresponding authors: seoyoungyuri.park@gmail.com (S.-Y. Park),
cho.junghwan@gmail.com (J. Cho)
1arXiv:2512.09964v1  [q-bio.GN]  10 Dec 2025

Figure 1. Overview of the Proposed Agentic AI Framework.The
pipeline integrates a gene expression matrix, clinical metadata, and user-
defined prompts to generate condition-specific biological insights. Target
clinical variables for comparison are first selected from the metadata via
statistical testing and presented to the user. Using these results, the user
formulates a prompt that defines the clinical comparison of interest and rel-
evant target pathways or conditions, which then guides predefined analyses
such as differential expression, clustering, and pathway enrichment. The
resulting outputs are subsequently interpreted by an LLM (Llama 3 70B)
with retrieval-augmented generation using Google Serper, enabling contex-
tualization of findings based on existing literature. As advanced analysis,
additional procedures including survival analysis and machine-learning clas-
sifiers provide further validation and hypothesis testing. The overall frame-
work delivers gene- and pathway-level interpretations and insights explicitly
aligned with user-specified hypotheses and clinical conditions.
2

1. INTRODUCTION
Next-Generation Sequencing (NGS) has revolutionized genomic research, en-
abling high-throughput analysis of gene expression data. However, down-
stream analysis such as differential expression gene (DEG) identification,
clustering, and pathway analysis requires substantial biological background
knowledge and computational expertise. Researchers lacking this expertise
often face barriers in interpreting results and extending analyses to advanced
stages like predictive modeling or hypothesis generation.
Currently, numerous studies leverage agentic AI to process RNA-seq
data; for example, a variety of agents such as CellAgent, BioAgents, and
CompBioAgent have been proposed [1–3]. These AI agents assist with pre-
processing and analysis for either scRNA-seq and bulk RNA-seq data, and
they perform standard downstream analyses, such as trajectory analysis and
clustering. However, existing studies using Agentic AI did not provide spe-
cialized downstream analysis, focusing instead on general workflows without
tailored interpretations for novices.
To address this, we developed an Agentic AI model that automates
NGS downstream analysis, provides biological interpretations, and recom-
mends/executes advanced methods. Through this process, users can com-
pare their results with findings from previous studies, obtain meaningful
insights, generate and test hypotheses, and receive guidance on which fur-
ther analyses to perform for validation and additional exploration, thereby
specifically supporting downstream analysis. Built on Llama 3 70B for effi-
cientreasoningandRAGforliterature-backedinsights, themodelisdeployed
3

as an interactive Streamlit web app, making it accessible to newcomers in
bioinformatics.
The model’s novelty lies in its agentic nature: it autonomously plans,
executes, and reflects on analyses, using user-provided gene expression ma-
trices and clinical data. The model is designed to execute and present prior
statistical tests for each clinical variable and accept user prompts specify-
ing the perspective from which the data should be analyzed. The model
integrates bioinformatics tools (Biopython [6], GSEApy [8], gProfiler [7])
with LLM-driven code generation, allowing seamless transition from basic
to advanced analyses. This extends prior works like GenoMAS [4] for tran-
scriptomic analysis and AutoBA [5] for multi-agent omics, by incorporating
adaptive RAG-based literature search and user-selected advanced workflows.
2. MATERIALS AND METHODS
System Overview
The model follows a modular, agentic workflow implemented in Python,
with Streamlit providing an interactive user interface. Users can upload
their own gene expression matrices (RNA-seq data) and clinical information
in CSV format or alternatively use example data consisting of200genes×
100samples enriched for cancer-related genes. Genes are extracted to ensure
consistent handling of gene identifiers across all components of the pipeline,
and the expression matrix is normalized using alog2transformation to sta-
bilize variance and reduce the impact of extreme values. To relate clinical
features to the user-defined hypothesis, the system computes a statistical
4

significance table for the clinical data using Pearson correlation or t-tests
between hypothesis-related variables and other clinical columns, summariz-
ing the data type, test used, test statistic, andp-value in a single table.
Core Downstream Analysis and Visualization
Basedontheprocessedinputsandclinicalvariables, themodelperformscore
downstream analyses commonly used in RNA-seq workflows. For DEG anal-
ysis, a t-test is applied to compare groups defined by a user-selected clinical
variable, and log fold changes (logFC) are calculated to quantify expression
differences between groups. Unsupervised structure in the data is explored
via k-means clustering on the transposed expression matrix, with clusters
visualized using principal component analysis (PCA). To connect DEGs to
biological processes, pathway analysis is conducted using gProfiler (0.3.5)
and gene set enrichment analysis (GSEA) through GSEA.py (1.1.11) on sig-
nificantly altered genes, and the top enriched pathways are summarized in
bar plots. The model generates multiple visual outputs, including a volcano
plot annotated with the top 10 genes, a PCA plot colored by the selected
clinical variable and accompanied by a legend, and functional bar plots high-
lighting key pathways, thereby offering both statistical and biological views
of the data.
Interpretation, Literature Integration, and Advanced Agentic
Actions
To support users with limited bioinformatics or biology background, the
model employs an LLM (Llama 3.3-70b) to interpret DEG, clustering, and
5

pathwayresults. Themodelfurtherintegratesaliterature-awareRAGpipeline:
PubMed is queried via Entrez using terms derived from dysregulated genes
and enriched pathways, abstracts are embedded with sentence-transformer
models and stored in a FAISS vector database for retrieval and summariza-
tion. Building on these insights, the LLM recommends three advanced anal-
ysis methods tailored to the specific dataset and findings, each accompanied
by a description, rationale, and executable Python code. Users can select
one of these methods via a Streamlit selectbox and trigger execution with a
“run” button directly in the web interface. The LLM then summarizes the
advancedanalysisoutputsforbeginnersandgeneratesdownloadablereports.
Throughout, the workflow is organized into interactive stages with dedicated
buttons, preserving user control while maintaining an agentic, autonomous
orchestration of data processing, analysis, interpretation, and method rec-
ommendation.
3. RESULTS
Using example data, the model identifies∼40significant DEGs (p <0.05),
clusters samples into two groups correlated withDisease_Status, and sim-
ulates pathway IDs. Volcano plot highlights top DEGs (e.g.,Gene1with
high−log10(p-value)). PCA plot color points by selected variables (e.g.,
Disease_Status), revealing group separation. Clinical significance table
showsp-values (e.g.,Disease_Statusvs. Age:p= 0.3, not significant).
Based on genes obtained from basic analysis (e.g., upregulated oncogenes
likeBRAFin diseased samples), the LLM derives insights such as potential
6

immune pathway dysregulation, reasoned by logFC>1in DEG results indi-
cating tumor progression. RAG summaries from PubMed abstracts provide
literature-backedevidence, e.g.,BRAFmutationslinkedtopoorprognosisin
lungcancer(logicalground: correlationwithclinicalsurvivaldata,p <0.05).
Advanced recommendations include survival modeling (code:statsmodels
CoxPHFitter), executed on user selection, yielding hazard ratios grounded
in DEG-driven hypothesis testing.
4. CONCLUSION
This Agentic AI model democratizes NGS downstream analysis by automat-
ing complex tasks and providing interpretable insights for non-experts. By
integrating LLM for code generation and RAG for evidence-based recom-
mendations, it reduces barriers in bioinformatics. Future extensions could
includegenenetwork-baseddrugresponsepredictionfortargetdiscoveryand
multi-omics support. The Streamlit app ensures accessibility, potentially ac-
celerating discovery in precision medicine.
ACKNOWLEDGEMENT
This work was supported by the Brain Pool Program through the National
Research Foundation of Korea (NRF), funded by the Ministry of Science and
ICT(RS-2023-00283791),theMinistryofEducation,Korea(2021R1I1A3056903),
by Basic Science Research Program through the NRF funded by the Min-
istry of Education (RS-2024-00459836), and by a grant of the Korea Health
Technology R&D Project through the Korea Health Industry Development
7

Institute (KHIDI), funded by the Ministry of Health & Welfare, Republic
of Korea (RS-2025-02263414). The corresponding authors (S.-Y. Park and
J. Cho) declare conflicts of interest related to ongoing patent applications
associated with this research.
References
[1] Xiao, Y., Liu, J., Zheng, Y., Xie, X., Hao, J., Li, M., ... & Peng, J.
(2024). Cellagent: An llm-driven multi-agent framework for automated
single-cell data analysis.arXiv preprint arXiv:2407.09811.
[2] Mehandru, N., Hall, A. K., Melnichenko, O., Dubinina, Y., Tsirulnikov,
D., Bamman, D., ... & Malladi, V. S. (2025). BioAgents: Bridging the
gap in bioinformatics analysis with multi-agent systems.Scientific Re-
ports, 15(1), 39036.
[3] Zhang, H., Sun, Y. H., Hu, W., Cui, X., Ouyang, Z., Cheng, D., ... &
Zhang, B. (2025). CompBioAgent: An LLM-powered agent for single-
cell RNA-seq data exploration.bioRxiv, 2025-03.
[4] Liu, H., Li, Y., & Wang, H. (2025). Genomas: A multi-agent framework
for scientific discovery via code-driven gene expression analysis.arXiv
preprint arXiv:2507.21035.
[5] Zhou, J., Zhang, B., Li, G., Chen, X., Li, H., Xu, X., ... & Gao, X.
(2024). An AI Agent for Fully Automated Multi-Omic Analyses.Ad-
vanced Science, 11(44), 2407094.
8

[6] Cock, P. J. A., Antao, T., Chang, J. T., Chapman, B. A., Cox, C.
J., Dalke, A., Friedberg, I., Hamelryck, T., Kauff, F., Wilczynski, B.,
& de Hoon, M. J. L. (2009). Biopython: Freely available Python tools
forcomputationalmolecularbiologyandbioinformatics.Bioinformatics,
25(11), 1422–1423.
[7] Kolberg, L., Raudvere, U., Kuzmin, I., Vilo, J., & Peterson, H. (2023).
g:Profiler—interoperable web service for functional enrichment analysis
and gene identifier mapping (2023 update).Nucleic Acids Research.
[8] Fang, Z., Liu, X., & Peltz, G. (2023). GSEApy: a comprehensive pack-
age for performing gene set enrichment analysis in Python.Bioinfor-
matics, 39(1), btac757.
9