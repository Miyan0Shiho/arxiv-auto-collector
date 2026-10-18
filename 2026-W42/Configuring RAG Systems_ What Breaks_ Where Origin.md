# Configuring RAG Systems: What Breaks, Where Originates, and How to Fix

**Authors**: Liangyu Chen, Youpeng Ma, Yulong Ye, Zezhen Xiang, Tao Chen

**Published**: 2026-10-13 15:37:27

**PDF URL**: [https://arxiv.org/pdf/2610.18567v1](https://arxiv.org/pdf/2610.18567v1)

## Abstract
Retrieval-Augmented Generation (RAG) has emerged as a dominant paradigm for improving Large Language Model (LLM) by intertwining it with other components, such as a vector/graph database. Therefore, the resulting RAG systems are often highly configurable, under which a non-trivial amount of failures are related to the configurations. However, there remains a lack of systematic understanding regarding the configuration issues for RAG systems. To bridge this gap, in this paper, we conduct the first large-scale empirical study of 654 real-world configuration issues collected from 10 diverse RAG systems. Drawing on this, we provide a complete categorization of the symptoms, root causes, relevant stages, and fix patterns of configuration issues for RAG systems, as well as several new observations. Our findings not only reveal the characteristics of configuration issues, but also strongly link them to the unique pipeline stages of RAG, e.g., we found that most symptoms can be observed at the early or late stage, while the stage of the root cause might not be at the same stage as where the symptom is identified. We also provide nine actionable insights to the practitioners on testing, localizing, and fixing the configuration issues for RAG systems.

## Full Text


<!-- PDF content starts -->

Configuring RAG Systems: What Breaks, Where Originates,
and How to Fix
LIANGYU CHEN,University of Electronic Science and Technology of China, China
YOUPENG MA,University of Electronic Science and Technology of China, China
YULONG YE,University of Birmingham, United Kingdom
ZEZHEN XIANG,University of Electronic Science and Technology of China, China
TAO CHEN∗,University of Birmingham, United Kingdom
Retrieval-Augmented Generation (RAG) has emerged as a dominant paradigm for improving Large Language
Model (LLM) by intertwining it with other components, such as a vector/graph database. Therefore, the
resulting RAG systems are often highly configurable, under which a non-trivial amount of failures are related
to the configurations. However, there remains a lack of systematic understanding regarding the configuration
issues for RAG systems. To bridge this gap, in this paper, we conduct the first large-scale empirical study
of 654 real-world configuration issues collected from 10 diverse RAG systems. Drawing on this, we provide
a complete categorization of the symptoms, root causes, relevant stages, and fix patterns of configuration
issues for RAG systems, as well as several new observations. Our findings not only reveal the characteristics
of configuration issues, but also strongly link them to the unique pipeline stages of RAG, e.g., we found that
most symptoms can be observed at the early or late stage, while the stage of the root cause might not be at the
same stage as where the symptom is identified. We also provide nine actionable insights to the practitioners
on testing, localizing, and fixing the configuration issues for RAG systems.
CCS Concepts:•Software and its engineering →Empirical software validation;Software configuration
management and version control systems.
Additional Key Words and Phrases: Retrieval-augmented generation, large language models, configuration
bugs, misconfiguration
1 Introduction
Large Language Models (LLMs) have witnessed unprecedented growth and evolution in recent
years, demonstrating remarkable capabilities across a wide spectrum of tasks [ 6,37,58]. To mitigate
the issue of LLM hallucination [ 27,35,40] and outdated knowledge [ 26,44], Retrieval-Augmented
Generation (RAG) system has emerged as a dominant paradigm [ 21,31]. Typically, as a new type
of software systems, the RAG systems augment LLMs with external knowledge sources, such as
documents, vector databases, and retrieval pipelines, forming the backbone of many question
answering systems, developer assistants, and knowledge-based applications [2, 17, 18].
To ensure flexibility, RAG systems are designed to be highly configurable [ 3,5,19,41,42]—
a key property of many types of software systems. For example, changing the retrieval stride
length will significantly affect the quality of the generated text in RAG systems [ 32]. However,
configuration has long been recognized as a critical source of reliability issues in traditional
configurable software systems [ 16,22–25,33,36,45,50,51], RAG systems are of no exception. Yet,
the unique complex stages in the pipeline of RAG systems (e.g., indexing, retrieval, and generation)
make their configuration even more error-prone [ 3,5,41]. For example, it has been shown that
configuration issues of RAG systems (including both configuration bugs and misconfiguration) can
∗Corresponding author.
Authors’ Contact Information: Liangyu Chen, University of Electronic Science and Technology of China, Chengdu, China,
cly009@std.uestc.edu.cn; Youpeng Ma, University of Electronic Science and Technology of China, Chengdu, China, myp@
std.uestc.edu.cn; Yulong Ye, yxy382@student.bham.ac.uk, University of Birmingham, Birmingham, United Kingdom; Zezhen
Xiang, zz.xiang.work@gmail.com, University of Electronic Science and Technology of China, Chengdu, China; Tao Chen,
University of Birmingham, Birmingham, United Kingdom, t.chen@bham.ac.uk.
Manuscript submitted to ACM Trans. Softw. Eng. Methodol.
arXiv:2610.18567v1  [cs.AI]  13 Oct 2026

2 Chen et al.
cause a latency performance degradation of10 ×, leading to a two orders of magnitude cost [ 5].
Another example showcases the complex interaction of configuration options among the RAG
stages: a subtle change in the indexing strategy can unexpectedly invalidate assumptions in the
retrieval or generation stages, causing significant errors to propagate silently across the pipeline,
and hence a system failure. Therefore, understanding the configuration issues of RAG systems and
how they have been fixed is in high demand.
Despite the importance, our current knowledge of the configuration issues within RAG systems
remains rather limited. While there are prior empirical studies of RAG systems, they focus on two
different aspects: either investigates how configuration choices influence system performance [ 3,
5,41], typically from a configuration tuning and hyperparameter optimization perspective; or
examines failure cases and practical challenges in RAG systems through empirical analyses of
observed bugs in general, which are not configuration-related [ 1,4,30]. There has been no systematic
study of the configuration issues as the root of RAG systems’ failures, particularly with respect to
the RAG pipeline.
To bridge this gap, in this paper, we conduct a large-scale empirical study on RAG configuration
issues, covering 654 issues and 345 pull requests from 10 RAG systems on GitHub over six person-
months. We make several contributions:
•A Categorization:We summarize a comprehensive categorization for configuration issues
in RAG systems with respect to its pipeline stages, including five types of symptoms, nine
root causes, and six fix patterns.
•Empirical Findings:Our empirical study reveals ten previously unknown findings, the
most notable ones are:
–Core pipeline fault, in which failures occurred at a pipeline stage with exceptions, is the
predominant symptom of configuration issues in RAG systems.
–For configuration bugs, invocation of external libraries/systems is the most common cause;
while resource provision that does not fulfil the needs is the most common reason for user
misconfiguration.
–Most of the symptoms occurred at the initialization, indexing and generation stage of RAG
systems; while root causes mostly lie in the early stages.
–The stage where the symptom is observed might not necessarily correspond to the stage of
the root cause—as observed in one third of the cases.
–When the symptom and root cause deviate at more distant stages (e.g., from initialization
to the last stage of generation), the resolving cost (time) of real configuration bugs can
increase by up to2.83×.
–Patching the logic and algorithm is the most common fix pattern for configuration bugs.
•Insights:Based on these results, we distil several insights about configuration issues for
RAG systems on their testing/detection, localization, and fixing.
All the data and code can be accessed at our repository: https://github.com/ideas-labo/rag-config-
bugs.
This paper is organized as follows. Section 2 introduces the background. Section 3 presents
our methodology. Section 4 delineates the categorization of configuration-related issues and the
findings. Section 5 discusses actionable insights. Section 6 addresses threats to validity. Section 7
reviews related work. Finally, Section 8 concludes the paper.
Manuscript submitted to ACM Trans. Softw. Eng. Methodol.

Configuring RAG Systems: What Breaks, Where Originates, and How to Fix 3
S1: Indexing
S0: System 
Initialization
S2: Query
Transformation
S4: Post-Retrieval
ProcessingUser
Query
S3: Retrieval
S5: Generation
Document
PromptDatabase
LLMQuery
Result
Fig. 1. Workflow of RAG with different pipeline stages.
2 Background
2.1 RAG Pipeline
As shown in Figure 1, RAG systems operate in their unique pipeline to work with LLM. Specifically,
given a task (e.g., question answering), there are several key stages [18]:
S0:Initializationprepares, validates the environment, and fulfils the resources needed to boot
the system.
S1:Indexingtransforms raw data into a searchable knowledge base via chunking and embedding.
S2:Query Transformationrewrites or decomposes user inputs to improve retrieval capability.
S3: Retrievalfetches contexts from vector/graph database.
S4: Post-Retrieval Processingrefines results through re-ranking and compression.
S5:Generationsynthesizes the final prompt with contextual information for the LLM(s) to
process.
Each stage comes with its own configuration options. The final output serves as the answer to
different tasks.
2.2 Configuration Issues for RAG Systems
In this paper, we define aconfiguration issuein RAG systems as a defect that satisfies two conditions:
•The issue must manifest as an observable system-level problem: functional defects, e.g., exe-
cution failures, crashes, or core pipeline breakdowns, and performance-related degradations,
e.g., latency or answer quality.
•It must be directly related to configuration.
From the example in Table 1, we see that the reported issue exhibits a system-level problem and
is relevant to the configuration of the RAG system. Clearly, it indicates an observable system-level
problem, i.e., a degradation in document chunking quality during indexing, with clear statements
confirming that the problem is directly related to the configuration optionchunk_token_num.
Manuscript submitted to ACM Trans. Softw. Eng. Methodol.

4 Chen et al.
Table 1. A real-world configuration issue from RAG systemRAGFlow.
ID#7790:Potential Issue innaive_merge_docxFunction
Description (excerpt from the issue report):
...This can result in the first chunk being larger than intended, as there’s no
proper size control for the initial chunk...
...The function initializes tk_nums = [0] and uses tk_nums[-1] >
chunk_token_num as a condition for creating new chunks... Since0 <
chunk_token_num (default 128), the first section is always added to the first
chunk...
In general, configuration issues can be broadly distinguished by two types depending on their
root causes [25, 51], which are both considered in this work:
•Configuration bugs:Here, the issue is caused by system defects, where the system fails to
correctly handle specific configuration combinations due to flawed adaptation logic, incorrect
assumptions, or incomplete validation.
•Misconfigurations:The issues are mainly caused by incompatible or improper configuration
settings (e.g., model choices and retrieval strategies) made by the users.
3 Empirical Methodology
Figure 2 shows our methodology. As can be seen, we select representative RAG systems from
GitHub, followed by rigorous extraction and analysis of the issues. These then allow us to construct
a categorization of configuration issues for RAG systems, together with a rich set of findings. All of
those are systematically validated for ensuring inter-rater reliability.
3.1 Research Questions
We seek to answer the following research questions (RQs):
•RQ1:What common symptoms do configuration issues exhibit in RAG systems?
•RQ2:What commonly causes the configuration issues?
•RQ3:How do the deviated stages between symptom and root cause affect the resolution of
configuration issues?
•RQ4:How are the actual configuration bugs often fixed?
RQ1andRQ2summarize the general characteristics of configuration issues for RAG systems.
RQ3is derived from the fact that the stage of the exhibited symptom of a configuration issue might
deviate from that of its root cause (e.g., a symptom from theindexingstage might be caused by
bugs in thegenerationstage), hence we aim to study how such a phenomenon might affect its bug
fixing.RQ4investigates the patterns of actual configuration bug fixing, if any.
Notably, we centre around the above aspects because RAG developers typically reasoned about
configuration issues from three complementary pillars: (1) how the issue manifests in practice; (2)
where the issue originates; and (3) how the issue is eventually resolved. Thus, understanding them
enables us to analyze issue propagation across the RAG pipeline and derive actionable guidance for
debugging and maintenance.
Manuscript submitted to ACM Trans. Softw. Eng. Methodol.

Configuring RAG Systems: What Breaks, Where Originates, and How to Fix 5
SOTA RAG Repos
Identification
GitHub API
RQ1-3RAG System Selection and Data Collection
10 RAG SystemsFiltering
CriteriaAutomated Search
Potential Issues
Inclusion and
Exclusion Criteria
Collected 
654 Issues
Analysis and Validation
Classification and 
Taxonomization
Inter-Rater 
Reliability ValidationFinal DatasetWith PRRQ4
Fig. 2. The empirical study methodology.
3.2 RAG System Selection
To identify the RAG system, we use several keywords to search over GitHub, i.e.,“RAG”or“Retrieval-
Augmented Generation”. This returns more than a thousand projects. We then filter the systems
using the criteria below:
•Configuration Accessibility: The repository contains user-adjustable configuration arte-
facts, such as configuration files, environment variables, or documented options.
•Popularity: To ensure sufficient issue discussion and relevance, we remove repositories with
less than2 ,000stars1—a proxy for both community adoption and the likelihood of reporting
configuration-related issues.
•Issue Richness: Repositories with fewer than 20 issues are removed, as there is little infor-
mation to be analyzed.
After applying the above criteria, we identify several tens of repositories of the RAG systems2.
To ensure both representativeness and realism, we select the top 10 repositories ranked by the
number of GitHub stars among those eligible.
As can be seen from Table 2, the RAG systems are of distinct domains, including general end-to-
end RAG platforms3(e.g.,FastGPT), research prototypes (e.g.HippoRAG), and domain specific RAG
system (e.g.DocsGPT). Further, the selected RAG systems cover the major forms of popular open-
source RAG software, including end-to-end platforms, developer-oriented frameworks, research
prototypes, and user-facing or domain-specific applications. They also incorporate heterogeneous
RAG components and external services, such as LLMs, embedding models, vector databases, re-
trievers, rerankers, and data sources. This diversity provides broad coverage of the architectures,
application scenarios, and configuration surfaces found in popular and active open-source RAG
systems, making the selected repositories representative within the scope targeted by our study.
1This was chosen pragmatically as a general measurement of popularity.
2As of 1st Sep, 2025.
3Some of these can be used for building other types of LLM systems, but we explicitly extract the RAG-related issues only.
Manuscript submitted to ACM Trans. Softw. Eng. Methodol.

6 Chen et al.
Table 2. Statistics of the selected RAG systems.
RAG System StarsRetrieved Issues Filtered Issues
Total Closed Total with PR
RAGFlow61.1k 1,417 685 217 161
DB-GPT17.0k 812 559 120 76
FastGPT25.3k 494 412 74 19
GraphRAG26.9k 195 152 64 28
LightRAG18.9k 214 197 63 22
Obsidian-copilot5.0k 564 353 35 15
Private-GPT56.4k 255 225 31 3
Quivr38.0k 283 282 22 8
DocsGPT16.9k 332 318 21 10
HippoRAG2.6k 86 75 7 3
Total – 4,652 3,258 654 345
3.3 Data Collection and Filtering
We collected issues from the repositories of the 10 RAG systems (till 1st August 2025) using a
two-step process: automated retrieval and then manual validation by the authors.
3.3.1 Automatic Search.To comprehensively identify configuration issues under a repository
of RAG system, we constructed a set of search queries by combining two primary dimensions:
commonly used configuration terms in software systems (e.g., config ,setting ,parameter ,env)
and RAG-specific terms corresponding to core pipeline components (e.g., embedding ,vectorstore ,
retriever ,prompt ). Each pair of the above terms is used in isolation; the resulting issues are then
merged with redundancy removal.
We tailored our search scope based on the volume of issues and the effectiveness of labels in
each repository:
•Labeled Search: For projects with a large volume of issues and well-maintained labeling
systems (e.g.,LightRAG), we combine the keywords with the project-specific label (e.g.,
keyword AND label:bug) to locate related issues while discarding unrelated discussions.
•Keyword-Only Search: For projects that did not maintain a meaningful labeling protocol
(or those returned fewer than 20 issues), we conducted keyword-based searches paired with
manual screening.
As from Table 2, this process yields a preliminary dataset of 4,652 candidate issues. From this
pool, we focus on the 3,258 closed issues due to the needs forRQ3andRQ4.
3.3.2 Manual Filtering.Since the retrieved issues might contain noise, e.g., some of them could be
usage questions or feature requests. To mitigate this, we conducted a rigorous manual inspection
to identify genuine configuration-related issues. We say that an issue should be included in the
final dataset if and only if it meets all the criteria below:
•Validity: The issue must describe a confirmed system defect (e.g., crash, functional error,
performance degradation) or user misconfiguration (e.g., parameter/option misunderstanding,
usage errors, or documentation-induced mistakes), excluding feature requests and inquiries.
Manuscript submitted to ACM Trans. Softw. Eng. Methodol.

Configuring RAG Systems: What Breaks, Where Originates, and How to Fix 7
•Relevance: The issue’s trigger, manifestation, or expected resolution must be logically
attributable to at least one user-adjustable configuration option of RAG, such as those specified
in configuration files (e.g.,YAMLorTOML), environment variables, or API arguments.
•Traceability: Only closed issues are considered, and sufficient evidence (e.g., linked pull
requests, reproduction steps, or conclusive developer discussions) has to be available to
reasonably infer the root cause of the issues.
Applying these criteria resulted in 654 configuration issues for RAG systems in total, which we
use to addressRQ1–RQ3. Among them, 345 issues explicitly link to pull requests (PR), enabling a
deeper analysis of code-level fix patterns forRQ4.
3.4 Analysis and Validation Procedure
To systematically analyze configuration issues in RAG systems, we carefully read the issues’
description, comments/discussions, and pull requests (if any), which provide rich information
on the symptoms (RQ1), root causes (RQ2), and fix patterns (RQ4), following iterative open-
coding [15, 38].
3.4.1 Iterative Categorization.Four authors, who are experienced researchers of RAG systems,
lead the categorization. The analysis starts with an exploratory open-coding phase. We randomly
sample an initial 10% issues (stratified across systems) in an open-coded manner without predefined
categories. Based on recurring patterns, we propose preliminary categories (and sub-categories as
needed) forRQ1,RQ2, andRQ4. In the first iteration, the full dataset is then labeled, with iterative
refinements (merging, splitting, redefining categories); we stop at the level of categories where the
next level of sub-categories resulted in highly biased labeling—over 90% of the issues are condensed
to one sub-category—or no finer-grained common phenomenon across the RAG systems can be
observed. All authors need to agree upon a final categorization.
3.4.2 Inter-Rater Reliability Validation.With the final categories, each author then independently
conducts the second iteration, aiming to correct the labels of issues using those categories. Note
that in some cases, an issue might involve multiple categories/sub-categories, in which case we
label it with the most fundamental/crucial one, as agreed by all raters. We also seek to assign the
issues to the relevant RAG stages forRQ3. To measure inter-rater reliability, we use Cohen’s 𝜅[29].
Through multiple iterations of relabeling, agreeing, and consulting external experts, we seek to
reach𝜅≥0.7for all levels of the categories and stages4[14, 46].
As summarized in Table 3, we achieve 𝜅≥ 0.7at all levels, including all the categories and
identification of the RAG stage, with three iterations. In particular, we have perfect agreement for
the main category of symptoms and root causes.
3.4.3 Unweighted vs. Weighted Percentage.To provide overall statistics, we report on both the
unweighted and weighted proportions of issues per category/sub-category over all systems, i.e.,
the former is robust to the diverse sample sizes of the RAG systems, while the latter is sensitive to
the sizes.
4 Results
4.1 Categorization of RAG Configuration Issues
Unlike others, RAG systems operate in a multi-stage semantic pipeline that integrates heterogeneous
components, which have been captured by our categorization in Figure 3. It is worth noting
that our categorization contains both general software-failure manifestations and RAG-specific
40.7 is a pragmatic “good” agreement in software engineering [14, 46].
Manuscript submitted to ACM Trans. Softw. Eng. Methodol.

8 Chen et al.
Table 3. Inter-rater agreements (Cohen’s𝜅) recorded.
2nd Iteration 3rd Iteration 4th Iteration
Symptom category 0.78 1.00 1.00
Symptom sub-category 0.61 0.73 0.73
Symptom stage 0.89 0.89 0.89
Root cause category 0.68 1.00 1.00
Root cause sub-category 0.53 0.69 0.80
Root cause stage 0.62 0.74 0.74
Fix pattern 0.63 0.82 0.82
Symptom
RAG System
Root Cause
Fix Pattern
[SP-A1] System crash
and unavailability
[SP-A]
Functional Execution Fault
[SP-B]
Behavior Fault
[SP-A2] Core
pipeline fault[SP-A3] Configuration
adherence fault [SP-B1] Low-
quality result [SP-B2] Inefficient
performance 
[RC-A]
Configuration Bug
[RC-B]
User Misconfiguration
[RC-A1] 
Configuration
handling bug[RC-A2] 
RAG component
implementation bug[RC-A5] 
Resource provisioning 
and orchestration bug[RC-A3] 
Inter-component
integration bug[RC-A4] 
External system
adaptation bug[RC-B1] 
External configuration
mistake[RC-B3] 
Configuration
misunderstanding[RC-B2] 
Component
coordination mistake[RC-B4] 
Documentation-induced
misconfiguration[FP1] Data interaction
and mapping fix[FP2] Component logic 
and algorithm fix[FP3] Checking and
robustness fix
[FP4] Static definition
and prompt tuning[FP6] Lifecycle and
resource scheduling[FP5] Environment and
documentation maintenance
S0, S1, 
S3, S5All Stages All Stages All Stages S1--S5
All Stages All Stages All Stages All Stages S0, S1, S5 All Stages All Stages S0--S4, S5 S0--S2
Fig. 3. Categorization for configuration issues of RAG systems.
refinements. High-level symptoms such as crashes or performance degradation may also occur in
conventional software systems. However, their finer-grained manifestations in our dataset are tied
to the functional components of RAG systems. In particular,core pipeline faultsare distinguished
according to the RAG stage in which the affected functionality resides, whileexternal system
adaptation bugscapture incompatibilities involving LLMs, embedding models, vector databases,
external data sources or applications, and rerankers. These component-specific categories are
further connected to the symptom and root cause stages, providing a stage-aware characterization
that is not captured by generic configuration bug categories.
4.1.1 Symptoms.The Symptom dimension characterizes how configuration issues become exter-
nally observable from a user perspective. We divide the observable symptoms into two categories,
each of which has different sub-categories.
Functional Execution Fault.This corresponds to cases where the system or a specific pipeline
stage fails to fulfil the intended needs, including three sub-categories:
•System crash and unavailabilitymean the system terminates unexpectedly without explicit
exceptions or hangs indefinitely. The RAG system becomes entirely non-responsive without
reasons, preventing any further interaction: e.g., issue#822forObsidian-copilot.
Manuscript submitted to ACM Trans. Softw. Eng. Methodol.

Configuring RAG Systems: What Breaks, Where Originates, and How to Fix 9
•Core pipeline faultdenotes that the system remains functional, but fails to fully complete
a specific stage of the pipeline with explicit exceptions, causing the system to stop: e.g., issue
#1642forGraphrag.
•Configuration adherence faultmeans that the system has good quality/performance, but
violates the user’s configuration intent: e.g., issue#2636forDB-GPT.
Behavior Fault.Here, the RAG system suffers severe behavior-level degradations, where the
system deliverslow-quality results, i.e., the system does generate answers, but they are factually
incorrect/incomplete; or exhibitspoor performance: a good quality outcome with abnormal
resource consumption or excessive latency. For example, issue#1321forLightRAGand issue#7518
forRAGFlow.
4.1.2 Root Cause.Our results indicate that root causes of the configuration issues for RAG systems
can be divided into two categories, each of which contains several sub-categories.
Configuration Bug.The configuration issues whose root causes lie in the system’s code. There
are five sub-categories:
•Configuration handling bugarises when configuration options are incorrectly parsed,
transmitted, or bound to execution logic, e.g., issue#1313forDocsGPT.
•RAG component implementation bugoriginates from flawed internal logic at core RAG
data processing (such as chunking, retrieval strategy, graph clustering, etc.) related to config-
uration, e.g., issue#436forRAGFlow.
•Inter-component integration bugoccurs at component boundaries, where mismatches
between upstream outputs and downstream input assumptions violate implicit data con-
tracts within the RAG pipeline under specific configurations. For example, issue#1642for
GraphRAG.
•External system adaptation bugstems from incompatible configuration between the RAG
system and external artefacts such as LLMs, databases, embedding models, or re-rankers:
issue#990forPrivate-GPT.
•Resource provisioning and orchestration bugreflects failures in allocating, scheduling, or
utilizing underlying computational resources for fulfilling a given configuration. For example,
issue#1288forRAGFlow.
User Misconfiguration.Unlike configuration bugs, user misconfiguration mainly originates
from users’ misunderstanding of the RAG system’s configuration options:
•External configuration mistakeoccurs when users select or configure external artefacts—
such as LLMs or databases—that are incapable of supporting the intended workload. For
example, issue#1567forLightRAG.
•Component coordination mistakearises when users set the configuration to combine
incompatible internal RAG components, violating functional dependencies between pipeline
stages. For example, issue#614forFastGPT.
•Configuration misunderstandingreflects misunderstandings of configuration semantics,
constraints, or deprecated options, leading to behaviors that diverge from user intents despite
technically valid inputs. For example, issue#144forObsidian-copilot.
•Documentation-induced misconfigurationcaptures cases where users are misled by
incorrect, outdated, or ambiguous documentation, resulting in configurations that follow
official guidance but fail in practice. For example, issue#8556forRAGFlow.
Manuscript submitted to ACM Trans. Softw. Eng. Methodol.

10 Chen et al.
4.1.3 Stage Labeling.We also study how configuration issues are localized within and propagated
through the RAG pipeline with respect to the symptoms and root cause5:
•Symptom Stage:The stage where the abnormal behavior first becomes observable to the
user. Crucially, this measures failure observability rather than the exact moment an internal
pipeline data flow might first deviate silently.
•Root Cause Stage:This pinpoints where the configuration issue originates. For configuration
bugs, it is the stage of the flawed code logic; for user misconfiguration, it is the stage where
users set improper configuration options. We infer the root-cause stage by jointly analyzing
issue content and developer discussions. For issues with pull requests, we examine the
modified code and the comments to identify the originating pipeline stage.
We found that some symptoms/root causes are linked to all stages, while others are only related
to certain stages.
4.1.4 Fix Pattern.We extract six fix patterns for configuration issues in RAG systems based on 345
issues with pull requests:
•Data interaction and mapping:This pattern involves ensuring configuration-related data
mapping and interface alignment between disparate RAG components or external APIs (e.g.,
key renaming, context propagation).
•Component logic and algorithm:This involves modifying the core business logic or
internal algorithms (e.g., rewriting chunking flows) to better handle configurations.
•Checking and robustness:This pattern focuses on increasing system stability by adding
error checking/warning code, e.g., null-pointer checkers, data type validation, and improved
error-handling logic to prevent crashes under invalid or incompatible configurations.
•Static definition and prompt tuning:These fixes involve tuning static constants, default
values, or prompt templates. Developers often use this to refine hard-coded defaults or update
prompt constraints within thegenerationstage to align with the intents of the configurations.
•Environment and documentation maintenance:This covers configuration-related but
non-code interventions, such as fixing external dependencies, updating deployment (e.g.,
Docker), or correcting misleading documentation to ensure correct RAG system deployment.
•Lifecycle and resource schedulingThis involves governing configuration-related resources
for RAG’s lifecycle, including concurrency and consistency. Typical fixes include singleton
enforcement, implementing cache invalidation, and optimizing pooling for external databases.
4.2 RQ1: Symptoms of Configuration Issues
4.2.1 Commonality of Symptoms.As from Table 4, the observedfunctional execution fault(86%/88%
for SP-A1, SP-A2, and SP-A3) is predominantly more common thanbehavior fault(14%/12% for
SP-B1 and SP-B2) on both weighted and unweighted percentages. These results indicate that
configuration issues in RAG systems primarily act as execution blockers rather than behaviors
influencers. Within the former, for all RAG systems, thecore pipeline faultis the most widespread
exhibited issue, consisting of 64%, since the related options, if contain issues, often cause severe
results such as crashes; while for the latter,quality faultis more commonly observed, as it is often
more critical than performance.
There are also discrepancies among the systems, e.g.,Quivr—external tools-heavy RAG systems—
show particularly high proportions of core pipeline faults, as their orchestration logic is strongly
coupled with configuration.FastGPT, a lightweight RAG system, exhibits more behavior faults.
5We found that most symptoms/root causes can correspond to one stage only. For the rare cases of symptoms/root causes
that are related to multiple stages, we use the one that is the most relevant, as agreed by all authors.
Manuscript submitted to ACM Trans. Softw. Eng. Methodol.

Configuring RAG Systems: What Breaks, Where Originates, and How to Fix 11
Table 4. Symptoms proportion by systems/categories—% per system (count). The highest % is highlighted in
blue .
System SP-A1 SP-A2 SP-A3 SP-B1 SP-B2 Total
RAGFlow8% (17) 63% (137) 17% (37) 9% (19) 3% (7) 217
DB-GPT11% (13) 66% (79) 9% (11) 13% (16) 1% (1) 120
FastGPT4% (3) 55% (41) 18% (13) 20% (15) 3% (2) 74
GraphRAG6% (4) 59% (38) 23% (15) 9% (6) 2% (1) 64
LightRAG11% (7) 63% (40) 6% (4) 14% (9) 5% (3) 63
Obsidian-copilot6% (2) 77% (27) 6% (2) 11% (4) 0% (0) 35
Private-GPT6% (2) 74% (23) 3% (1) 6% (2) 10% (3) 31
Quivr5% (1) 86% (19) 5% (1) 5% (1) 0% (0) 22
DocsGPT14% (3) 62% (13) 14% (3) 0% (0) 10% (2) 21
HippoRAG29% (2) 57% (4) 14% (1) 0% (0) 0% (0) 7
Weighted 8% (54) 64% (421) 14% (88) 11% (72) 3% (19) 654
Unweighted10% (54) 66% (421) 12% (88) 9% (72) 3% (19) 654
Table 5. Overall symptoms proportion over pipeline stages. The highest is highlighted in blue .
Pipeline Stage Count Weighted Unweighted
S0: Initialization 153 24% 22%
S1: Indexing 208 32% 30%
S2: Query Transformation 23 3% 3%
S3: Retrieval 70 11% 11%
S4: Post-Retrieval Processing 23 3% 2%
S5: Generation 177 27% 32%
Finding 1
Configuration issues in RAG systems are more predominantly manifested as functional execution
faults (86%/88% ) rather than behavior faults (14%/12%). Faults related to the core pipeline and the
answers’ quality are the most common.
4.2.2 Symptoms-Stage Relations.Table 5 reveals two critical observations on the symptoms-stage
relations: (1) the stageinitializationaccounts for 24%/22% of the issues, meaning that users frequently
fail to even start the system. (2) The issues are more associated with either the early or late stage. That
is, we see that stageindexingandgenerationown 32%/30% and 27%/32% of the issues, respectively,
which are considerably higher than any other intermediate stages.
Finding 2
Configuration issues in RAG systems predominantly surface at three critical stages: theinitializa-
tion(24%/22%),indexing(32%/30%) andgeneration(27%/32%).
To take a closer look, as shown in Figure 4, there is a distinct temporal separation in symptom
manifestation. 54%-92%/36%-96% of thefunctional execution faultoccurs at the first two stages,
often blocking further execution. In contrast, thebehavior faultis related to the late stages, with
42%-64%/24%-59% at thegenerationstage, manifesting after the full pipeline has executed. This is
Manuscript submitted to ACM Trans. Softw. Eng. Methodol.

12 Chen et al.
/uni00000036/uni00000013 /uni00000036/uni00000014 /uni00000036/uni00000015 /uni00000036/uni00000016 /uni00000036/uni00000017 /uni00000036/uni00000018/uni00000036/uni00000033/uni00000010/uni00000024/uni00000014
/uni00000036/uni00000033/uni00000010/uni00000024/uni00000015
/uni00000036/uni00000033/uni00000010/uni00000024/uni00000016
/uni00000036/uni00000033/uni00000010/uni00000025/uni00000014
/uni00000036/uni00000033/uni00000010/uni00000025/uni00000015/uni0000001b/uni00000014/uni00000008
/uni0000000b/uni00000017/uni00000017/uni0000000c/uni00000014/uni00000014/uni00000008
/uni0000000b/uni00000019/uni0000000c/uni00000013/uni00000008
/uni0000000b/uni00000013/uni0000000c/uni00000017/uni00000008
/uni0000000b/uni00000015/uni0000000c/uni00000013/uni00000008
/uni0000000b/uni00000013/uni0000000c/uni00000017/uni00000008
/uni0000000b/uni00000015/uni0000000c
/uni00000014/uni0000001c/uni00000008
/uni0000000b/uni0000001a/uni0000001c/uni0000000c/uni00000016/uni0000001b/uni00000008
/uni0000000b/uni00000014/uni00000019/uni00000015/uni0000000c/uni00000017/uni00000008
/uni0000000b/uni00000014/uni00000018/uni0000000c/uni00000014/uni00000015/uni00000008
/uni0000000b/uni00000018/uni00000014/uni0000000c/uni00000016/uni00000008
/uni0000000b/uni00000014/uni00000016/uni0000000c/uni00000015/uni00000017/uni00000008
/uni0000000b/uni00000014/uni00000013/uni00000014/uni0000000c
/uni00000016/uni00000013/uni00000008
/uni0000000b/uni00000015/uni00000019/uni0000000c/uni00000015/uni00000017/uni00000008
/uni0000000b/uni00000015/uni00000014/uni0000000c/uni00000017/uni00000008
/uni0000000b/uni00000017/uni0000000c/uni00000014/uni00000015/uni00000008
/uni0000000b/uni00000014/uni00000014/uni0000000c/uni0000001a/uni00000008
/uni0000000b/uni00000019/uni0000000c/uni00000015/uni00000016/uni00000008
/uni0000000b/uni00000015/uni00000013/uni0000000c
/uni00000013/uni00000008
/uni0000000b/uni00000013/uni0000000c/uni00000015/uni00000015/uni00000008
/uni0000000b/uni00000014/uni00000019/uni0000000c/uni00000016/uni00000008
/uni0000000b/uni00000015/uni0000000c/uni0000001a/uni00000008
/uni0000000b/uni00000018/uni0000000c/uni00000017/uni00000008
/uni0000000b/uni00000016/uni0000000c/uni00000019/uni00000017/uni00000008
/uni0000000b/uni00000017/uni00000019/uni0000000c
/uni00000015/uni00000014/uni00000008
/uni0000000b/uni00000017/uni0000000c/uni00000014/uni00000019/uni00000008
/uni0000000b/uni00000016/uni0000000c/uni00000014/uni00000014/uni00000008
/uni0000000b/uni00000015/uni0000000c/uni00000018/uni00000008
/uni0000000b/uni00000014/uni0000000c/uni00000018/uni00000008
/uni0000000b/uni00000014/uni0000000c/uni00000017/uni00000015/uni00000008
/uni0000000b/uni0000001b/uni0000000c/uni0000003a/uni00000048/uni0000004c/uni0000004a/uni0000004b/uni00000057/uni00000048/uni00000047/uni00000003/uni0000000b/uni00000008/uni0000000c
/uni00000036/uni00000013 /uni00000036/uni00000014 /uni00000036/uni00000015 /uni00000036/uni00000016 /uni00000036/uni00000017 /uni00000036/uni00000018/uni00000036/uni00000033/uni00000010/uni00000024/uni00000014
/uni00000036/uni00000033/uni00000010/uni00000024/uni00000015
/uni00000036/uni00000033/uni00000010/uni00000024/uni00000016
/uni00000036/uni00000033/uni00000010/uni00000025/uni00000014
/uni00000036/uni00000033/uni00000010/uni00000025/uni00000015/uni0000001b/uni00000018/uni00000008
/uni0000000b/uni00000017/uni00000017/uni0000000c/uni00000014/uni00000014/uni00000008
/uni0000000b/uni00000019/uni0000000c/uni00000013/uni00000008
/uni0000000b/uni00000013/uni0000000c/uni00000016/uni00000008
/uni0000000b/uni00000015/uni0000000c/uni00000013/uni00000008
/uni0000000b/uni00000013/uni0000000c/uni00000015/uni00000008
/uni0000000b/uni00000015/uni0000000c
/uni00000014/uni00000018/uni00000008
/uni0000000b/uni0000001a/uni0000001c/uni0000000c/uni00000016/uni0000001b/uni00000008
/uni0000000b/uni00000014/uni00000019/uni00000015/uni0000000c/uni00000016/uni00000008
/uni0000000b/uni00000014/uni00000018/uni0000000c/uni00000014/uni00000016/uni00000008
/uni0000000b/uni00000018/uni00000014/uni0000000c/uni00000015/uni00000008
/uni0000000b/uni00000014/uni00000016/uni0000000c/uni00000015/uni0000001b/uni00000008
/uni0000000b/uni00000014/uni00000013/uni00000014/uni0000000c
/uni00000015/uni00000018/uni00000008
/uni0000000b/uni00000015/uni00000019/uni0000000c/uni00000014/uni00000014/uni00000008
/uni0000000b/uni00000015/uni00000014/uni0000000c/uni00000016/uni00000008
/uni0000000b/uni00000017/uni0000000c/uni00000014/uni00000016/uni00000008
/uni0000000b/uni00000014/uni00000014/uni0000000c/uni00000017/uni00000008
/uni0000000b/uni00000019/uni0000000c/uni00000017/uni00000017/uni00000008
/uni0000000b/uni00000015/uni00000013/uni0000000c
/uni00000013/uni00000008
/uni0000000b/uni00000013/uni0000000c/uni00000014/uni00000018/uni00000008
/uni0000000b/uni00000014/uni00000019/uni0000000c/uni00000014/uni00000008
/uni0000000b/uni00000015/uni0000000c/uni00000016/uni00000008
/uni0000000b/uni00000018/uni0000000c/uni00000015/uni00000008
/uni0000000b/uni00000016/uni0000000c/uni00000018/uni0000001c/uni00000008
/uni0000000b/uni00000017/uni00000019/uni0000000c
/uni00000014/uni00000019/uni00000008
/uni0000000b/uni00000017/uni0000000c/uni00000014/uni0000001a/uni00000008
/uni0000000b/uni00000016/uni0000000c/uni00000019/uni00000008
/uni0000000b/uni00000015/uni0000000c/uni00000014/uni00000008
/uni0000000b/uni00000014/uni0000000c/uni00000018/uni00000008
/uni0000000b/uni00000014/uni0000000c/uni00000015/uni00000017/uni00000008
/uni0000000b/uni0000001b/uni0000000c/uni00000038/uni00000051/uni0000005a/uni00000048/uni0000004c/uni0000004a/uni0000004b/uni00000057/uni00000048/uni00000047/uni00000003/uni0000000b/uni00000008/uni0000000c
/uni00000013/uni00000015/uni00000013/uni00000017/uni00000013/uni00000019/uni00000013/uni0000001b/uni00000013/uni00000014/uni00000013/uni00000013
/uni00000033/uni00000048/uni00000055/uni00000046/uni00000048/uni00000051/uni00000057/uni00000044/uni0000004a/uni00000048
Fig. 4. Detailed symptoms % per category over stages.
Table 6. Root cause proportion by systems/categories. The same format as Table 4.
System RC-A1 RC-A2 RC-A3 RC-A4 RC-A5 RC-B1 RC-B2 RC-B3 RC-B4 Total
RAGFlow18% (38) 28% (61) 10% (21) 26% (56) 3% (7) 5% (10) 1% (3) 7% (16) 2% (5) 217
DB-GPT12% (14) 13% (16) 13% (16) 32% (38) 12% (14) 8% (10) 2% (2) 8% (9) 1% (1) 120
FastGPT5% (4) 8% (6) 0% (0) 28% (21) 3% (2) 38% (28) 4% (3) 14% (10) 0% (0) 74
GraphRAG 33% (21) 19% (12) 9% (6) 19% (12) 2% (1) 6% (4) 3% (2) 9% (6) 0% (0) 64
LightRAG6% (4) 17% (11) 6% (4) 40% (25) 0% (0) 14% (9) 0% (0) 16% (10) 0% (0) 63
Obsidian-copilot29% (10) 3% (1) 0% (0) 17% (6) 6% (2) 34% (12) 3% (1) 9% (3) 0% (0) 35
Private-GPT6% (2) 0% (0) 10% (3) 23% (7) 10% (3) 42% (13) 3% (1) 6% (2) 0% (0) 31
Quivr9% (2) 9% (2) 0% (0) 59% (13) 0% (0) 18% (4) 0% (0) 5% (1) 0% (0) 22
DocsGPT24% (5) 0% (0) 0% (0) 24% (5) 10% (2) 29% (6) 5% (1) 10% (2) 0% (0) 21
HippoRAG14% (1) 14% (1) 0% (0) 14% (1) 0% (0) 43% (3) 0% (0) 14% (1) 0% (0) 7
Weighted 15% (101) 17% (110) 8% (50) 28% (184) 5% (31) 15% (99) 2% (13) 9% (60) 1% (6) 654
Unweighted16% (101) 11% (110) 5% (50) 28% (184) 4% (31) 24% (99) 2% (13) 10% (60) 0% (6) 654
because the early stages can easily exhibit errors related to preparations, while the late stage is
often where the problematic answers are observed.
Finding 3
Functional execution faultoften occurs much earlier than thebehavior faultat the pipeline stages
of RAG systems.
4.3 RQ2: Root Causes of Configuration Issues
4.3.1 Commonality of Root Causes.As shown in Table 6,configuration bug(RC-A1 to RC-A5)
dominates the root causes, accounting for 73%/64% of the weighted/unweighted percentage, while
user misconfigurationaccounts for the remaining 27%/36%. Notably,external system adaptation
bug(RC-A4) emerges as the most prevalent root cause of configuration bugs. This reflects the
fundamental nature of RAG systems as integration-centric architectures: they act as glue code that
orchestrates multiple heterogeneous components with evolving interfaces and implicit assumptions.
As a result, configuration bugs frequently surface at those boundaries.
As foruser misconfiguration, theexternal configuration mistake(RC-B1) is the most common
cause. This suggests that user mistakes are most likely to arise from configuring external artefacts
that do not meet the RAG system’s needs.
Manuscript submitted to ACM Trans. Softw. Eng. Methodol.

Configuring RAG Systems: What Breaks, Where Originates, and How to Fix 13
Table 7. Overall root cause proportion over pipeline stages. The highest is highlighted in blue .
Pipeline Stage Count Weighted Unweighted
S0: Initialization 255 39% 40%
S1: Indexing 177 27% 25%
S2: Query Transformation 27 4% 3%
S3: Retrieval 43 7% 5%
S4: Post-Retrieval Processing 23 3% 2%
S5: Generation 129 20% 24%
/uni00000036/uni00000013 /uni00000036/uni00000014 /uni00000036/uni00000015 /uni00000036/uni00000016 /uni00000036/uni00000017 /uni00000036/uni00000018/uni00000035/uni00000026/uni00000010/uni00000024/uni00000014
/uni00000035/uni00000026/uni00000010/uni00000024/uni00000015
/uni00000035/uni00000026/uni00000010/uni00000024/uni00000016
/uni00000035/uni00000026/uni00000010/uni00000024/uni00000017
/uni00000035/uni00000026/uni00000010/uni00000024/uni00000018
/uni00000035/uni00000026/uni00000010/uni00000025/uni00000014
/uni00000035/uni00000026/uni00000010/uni00000025/uni00000015
/uni00000035/uni00000026/uni00000010/uni00000025/uni00000016
/uni00000035/uni00000026/uni00000010/uni00000025/uni00000017/uni0000001b/uni00000015/uni00000008
/uni0000000b/uni0000001b/uni00000016/uni0000000c/uni00000019/uni00000008
/uni0000000b/uni00000019/uni0000000c/uni00000014/uni00000008
/uni0000000b/uni00000014/uni0000000c/uni00000014/uni00000008
/uni0000000b/uni00000014/uni0000000c/uni00000015/uni00000008
/uni0000000b/uni00000015/uni0000000c/uni0000001b/uni00000008
/uni0000000b/uni0000001b/uni0000000c
/uni00000014/uni00000017/uni00000008
/uni0000000b/uni00000014/uni00000018/uni0000000c/uni00000017/uni0000001c/uni00000008
/uni0000000b/uni00000018/uni00000017/uni0000000c/uni00000015/uni00000008
/uni0000000b/uni00000015/uni0000000c/uni00000014/uni00000014/uni00000008
/uni0000000b/uni00000014/uni00000015/uni0000000c/uni0000001a/uni00000008
/uni0000000b/uni0000001b/uni0000000c/uni00000014/uni0000001a/uni00000008
/uni0000000b/uni00000014/uni0000001c/uni0000000c
/uni00000015/uni00000013/uni00000008
/uni0000000b/uni00000014/uni00000013/uni0000000c/uni00000017/uni0000001b/uni00000008
/uni0000000b/uni00000015/uni00000017/uni0000000c/uni00000015/uni00000008
/uni0000000b/uni00000014/uni0000000c/uni00000014/uni00000019/uni00000008
/uni0000000b/uni0000001b/uni0000000c/uni00000015/uni00000008
/uni0000000b/uni00000014/uni0000000c/uni00000014/uni00000015/uni00000008
/uni0000000b/uni00000019/uni0000000c
/uni00000016/uni00000013/uni00000008
/uni0000000b/uni00000018/uni00000019/uni0000000c/uni00000016/uni00000015/uni00000008
/uni0000000b/uni00000018/uni0000001b/uni0000000c/uni00000017/uni00000008
/uni0000000b/uni0000001b/uni0000000c/uni0000001a/uni00000008
/uni0000000b/uni00000014/uni00000016/uni0000000c/uni00000017/uni00000008
/uni0000000b/uni0000001a/uni0000000c/uni00000015/uni00000016/uni00000008
/uni0000000b/uni00000017/uni00000015/uni0000000c
/uni0000001a/uni0000001a/uni00000008
/uni0000000b/uni00000015/uni00000017/uni0000000c/uni00000014/uni00000016/uni00000008
/uni0000000b/uni00000017/uni0000000c/uni00000013/uni00000008
/uni0000000b/uni00000013/uni0000000c/uni00000013/uni00000008
/uni0000000b/uni00000013/uni0000000c/uni00000013/uni00000008
/uni0000000b/uni00000013/uni0000000c/uni00000014/uni00000013/uni00000008
/uni0000000b/uni00000016/uni0000000c
/uni00000016/uni00000013/uni00000008
/uni0000000b/uni00000016/uni00000013/uni0000000c/uni00000015/uni00000016/uni00000008
/uni0000000b/uni00000015/uni00000016/uni0000000c/uni00000014/uni00000013/uni00000008
/uni0000000b/uni00000014/uni00000013/uni0000000c/uni00000014/uni00000008
/uni0000000b/uni00000014/uni0000000c/uni00000017/uni00000008
/uni0000000b/uni00000017/uni0000000c/uni00000016/uni00000015/uni00000008
/uni0000000b/uni00000016/uni00000014/uni0000000c
/uni00000016/uni0000001c/uni00000008
/uni0000000b/uni00000018/uni0000000c/uni00000015/uni00000016/uni00000008
/uni0000000b/uni00000016/uni0000000c/uni0000001b/uni00000008
/uni0000000b/uni00000014/uni0000000c/uni00000014/uni00000018/uni00000008
/uni0000000b/uni00000015/uni0000000c/uni00000013/uni00000008
/uni0000000b/uni00000013/uni0000000c/uni00000014/uni00000018/uni00000008
/uni0000000b/uni00000015/uni0000000c
/uni00000017/uni00000019/uni00000008
/uni0000000b/uni00000015/uni0000001b/uni0000000c/uni0000001a/uni00000008
/uni0000000b/uni00000017/uni0000000c/uni00000018/uni00000008
/uni0000000b/uni00000016/uni0000000c/uni00000014/uni00000013/uni00000008
/uni0000000b/uni00000019/uni0000000c/uni00000015/uni00000008
/uni0000000b/uni00000014/uni0000000c/uni00000016/uni00000013/uni00000008
/uni0000000b/uni00000014/uni0000001b/uni0000000c
/uni00000019/uni00000019/uni00000008
/uni0000000b/uni00000017/uni0000000c/uni00000014/uni0000001a/uni00000008
/uni0000000b/uni00000014/uni0000000c/uni00000014/uni0000001a/uni00000008
/uni0000000b/uni00000014/uni0000000c/uni00000013/uni00000008
/uni0000000b/uni00000013/uni0000000c/uni00000013/uni00000008
/uni0000000b/uni00000013/uni0000000c/uni00000013/uni00000008
/uni0000000b/uni00000013/uni0000000c/uni0000003a/uni00000048/uni0000004c/uni0000004a/uni0000004b/uni00000057/uni00000048/uni00000047/uni00000003/uni0000000b/uni00000008/uni0000000c
/uni00000036/uni00000013 /uni00000036/uni00000014 /uni00000036/uni00000015 /uni00000036/uni00000016 /uni00000036/uni00000017 /uni00000036/uni00000018/uni00000035/uni00000026/uni00000010/uni00000024/uni00000014
/uni00000035/uni00000026/uni00000010/uni00000024/uni00000015
/uni00000035/uni00000026/uni00000010/uni00000024/uni00000016
/uni00000035/uni00000026/uni00000010/uni00000024/uni00000017
/uni00000035/uni00000026/uni00000010/uni00000024/uni00000018
/uni00000035/uni00000026/uni00000010/uni00000025/uni00000014
/uni00000035/uni00000026/uni00000010/uni00000025/uni00000015
/uni00000035/uni00000026/uni00000010/uni00000025/uni00000016
/uni00000035/uni00000026/uni00000010/uni00000025/uni00000017/uni0000001b/uni00000017/uni00000008
/uni0000000b/uni0000001b/uni00000016/uni0000000c/uni00000017/uni00000008
/uni0000000b/uni00000019/uni0000000c/uni00000014/uni00000008
/uni0000000b/uni00000014/uni0000000c/uni00000013/uni00000008
/uni0000000b/uni00000014/uni0000000c/uni00000014/uni00000008
/uni0000000b/uni00000015/uni0000000c/uni0000001c/uni00000008
/uni0000000b/uni0000001b/uni0000000c
/uni0000001b/uni00000008
/uni0000000b/uni00000014/uni00000018/uni0000000c/uni00000017/uni00000016/uni00000008
/uni0000000b/uni00000018/uni00000017/uni0000000c/uni00000015/uni00000008
/uni0000000b/uni00000015/uni0000000c/uni0000001a/uni00000008
/uni0000000b/uni00000014/uni00000015/uni0000000c/uni00000017/uni00000008
/uni0000000b/uni0000001b/uni0000000c/uni00000014/uni0000001a/uni00000008
/uni0000000b/uni00000014/uni0000001c/uni0000000c
/uni0000001b/uni00000008
/uni0000000b/uni00000014/uni00000013/uni0000000c/uni00000015/uni00000015/uni00000008
/uni0000000b/uni00000015/uni00000017/uni0000000c/uni00000013/uni00000008
/uni0000000b/uni00000014/uni0000000c/uni0000001a/uni00000008
/uni0000000b/uni0000001b/uni0000000c/uni00000013/uni00000008
/uni0000000b/uni00000014/uni0000000c/uni00000014/uni00000015/uni00000008
/uni0000000b/uni00000019/uni0000000c
/uni00000015/uni0000001c/uni00000008
/uni0000000b/uni00000018/uni00000019/uni0000000c/uni00000017/uni00000015/uni00000008
/uni0000000b/uni00000018/uni0000001b/uni0000000c/uni00000016/uni00000008
/uni0000000b/uni0000001b/uni0000000c/uni00000017/uni00000008
/uni0000000b/uni00000014/uni00000016/uni0000000c/uni00000015/uni00000008
/uni0000000b/uni0000001a/uni0000000c/uni00000014/uni0000001c/uni00000008
/uni0000000b/uni00000017/uni00000015/uni0000000c
/uni00000017/uni0000001b/uni00000008
/uni0000000b/uni00000015/uni00000017/uni0000000c/uni00000014/uni00000017/uni00000008
/uni0000000b/uni00000017/uni0000000c/uni00000013/uni00000008
/uni0000000b/uni00000013/uni0000000c/uni00000013/uni00000008
/uni0000000b/uni00000013/uni0000000c/uni00000013/uni00000008
/uni0000000b/uni00000013/uni0000000c/uni0000001b/uni00000008
/uni0000000b/uni00000016/uni0000000c
/uni00000017/uni00000019/uni00000008
/uni0000000b/uni00000016/uni00000013/uni0000000c/uni00000014/uni0000001b/uni00000008
/uni0000000b/uni00000015/uni00000016/uni0000000c/uni00000017/uni00000008
/uni0000000b/uni00000014/uni00000013/uni0000000c/uni00000014/uni00000008
/uni0000000b/uni00000014/uni0000000c/uni00000014/uni00000008
/uni0000000b/uni00000017/uni0000000c/uni00000016/uni00000013/uni00000008
/uni0000000b/uni00000016/uni00000014/uni0000000c
/uni00000015/uni00000015/uni00000008
/uni0000000b/uni00000018/uni0000000c/uni00000015/uni00000016/uni00000008
/uni0000000b/uni00000016/uni0000000c/uni00000016/uni00000008
/uni0000000b/uni00000014/uni0000000c/uni00000014/uni00000018/uni00000008
/uni0000000b/uni00000015/uni0000000c/uni00000013/uni00000008
/uni0000000b/uni00000013/uni0000000c/uni0000001a/uni00000008
/uni0000000b/uni00000015/uni0000000c
/uni00000016/uni00000018/uni00000008
/uni0000000b/uni00000015/uni0000001b/uni0000000c/uni00000016/uni00000008
/uni0000000b/uni00000017/uni0000000c/uni00000016/uni00000008
/uni0000000b/uni00000016/uni0000000c/uni00000014/uni00000014/uni00000008
/uni0000000b/uni00000019/uni0000000c/uni00000014/uni00000008
/uni0000000b/uni00000014/uni0000000c/uni00000017/uni0000001b/uni00000008
/uni0000000b/uni00000014/uni0000001b/uni0000000c
/uni00000014/uni00000019/uni00000008
/uni0000000b/uni00000017/uni0000000c/uni00000015/uni00000008
/uni0000000b/uni00000014/uni0000000c/uni00000015/uni00000008
/uni0000000b/uni00000014/uni0000000c/uni00000013/uni00000008
/uni0000000b/uni00000013/uni0000000c/uni00000013/uni00000008
/uni0000000b/uni00000013/uni0000000c/uni00000013/uni00000008
/uni0000000b/uni00000013/uni0000000c/uni00000038/uni00000051/uni0000005a/uni00000048/uni0000004c/uni0000004a/uni0000004b/uni00000057/uni00000048/uni00000047/uni00000003/uni0000000b/uni00000008/uni0000000c
/uni00000013/uni00000015/uni00000013/uni00000017/uni00000013/uni00000019/uni00000013/uni0000001b/uni00000013/uni00000014/uni00000013/uni00000013
/uni00000033/uni00000048/uni00000055/uni00000046/uni00000048/uni00000051/uni00000057/uni00000044/uni0000004a/uni00000048
Fig. 5. Detailed root cause % per category over stages.
There is also a notable system discrepancy: highly-integrated and cloud-native assistants, e.g.,
Quivr, predominantly suffer from external system adaptation bugs due to their extensive reliance
on third-party APIs, whereas local-deployment or lightweight systems, e.g.,Private-GPTand
FastGPTexhibit a significantly higher proportion of external configuration mistakes (38%–42%) as
users frequently struggle with local environment and model provisioning.
Finding 4
Configuration bug(73%/64%) is the primary cause of configuration issues in RAG systems against
user misconfiguration(27%/36%). Notably,external system adaptation bugis the most predominant
configuration bug, while the most common user mistakes are from configuring external artefacts.
4.3.2 Root Cause-Stage Relation.We further analyze at which RAG stage the root causes are
identified. As shown in Table 7, the majority of root causes lie in the early stage (initializationand
indexing), which together account for over 66%/65% of the cases. The same can be observed over
the sub-categories (Figure 5). This highlights that many configuration issues are rooted in system
setup and data preparation.
Manuscript submitted to ACM Trans. Softw. Eng. Methodol.

14 Chen et al.
Table 8. Proportion of607error-prone configuration option over stages. The highest is highlighted in blue .
Pipeline Stage Count Weighted Unweighted
S0: Initialization 131 21% 22%
S1: Indexing 239 39% 38%
S2: Query Transformation 25 4% 3%
S3: Retrieval 40 7% 7%
S4: Post-Retrieval Processing 34 6% 4%
S5: Generation 138 23% 26%
Finding 5
All types of root causes of configuration issues in RAG are often related to early RAG pipeline
stages (66%/65% cases).
4.3.3 Configuration Options-Stage Relations.To understand the granularity of configuration risks,
we extracted the specific configuration options of root causes in the issues. Notblity, to prevent
repeatedly reported problems from disproportionately increasing the frequency of a configuration
option, we deduplicated the option-level data within each repository. Specifically, when multiple
issues from the same repository involved the same configuration option, that option was counted
only once. The same option name appearing in different repositories was retained separately because
its implementation, semantics, and affected component may differ across systems. Consequently, the
resulting 607 repository–option pairs characterize the breadth of distinct issue-linked configuration
options rather than the raw frequency of issue reports.
In total, we identify 607 unique options across the 654 issues (an issue may involve multiple
options). As in Table 8, we say that:
Finding 6
The complexity of RAG configuration is unevenly distributed: theindexingstage involves the
highest volume of error-prone options (39%/38%), such as embedding settings.
4.4 RQ3: Stage Gap and Resolving Costs
RQ3investigates how costly it is to resolve the configuration issues of RAG systems in practice,
depending on the deviation between the stage of symptoms and root causes. To that end, we
compute the stage gapGof each issue:
G=|S−R|(1)
wherebySandRdenote the stage level for symptom and root cause, respectively. Clearly, a
G= 0indicates the issue manifests at the same stage where it was introduced. A G> 0indicates
cross-stage propagation, where the configuration issues are propagated through other stages before
revealing a symptom. Note that the cause might be related to a stage earlier or later than the stage
of the symptom.
For example, the issue#144forObsidian-copilotis a misconfiguration issue. Here, the user
configured the RAG system to use the“Active Note”context mode, mistakenly believing it would
feed the entire document as full context. However, the system’s underlying logic uses similarity-
based retrieval. As such, the stage of root cause isretrieval, as the user’s misunderstanding led to
selecting a retrieval strategy unsuitable for the task. Yet, the symptom only became observable at the
generationstage, when the LLM produced a low-quality response. This results in a G=| 5−3|=2.
Manuscript submitted to ACM Trans. Softw. Eng. Methodol.

Configuring RAG Systems: What Breaks, Where Originates, and How to Fix 15
Table 9. Stage gap proportion across root cause categories; W and U denote weighted and unweighted
percentage, respectively. The highest is highlighted in blue .
Stage GapOverall
[Count (W/U%)]RC-A
[Count (W/U%)]RC-B
[Count (W/U%)]
Gap = 0 440 (67%/64%) 332 (70%/67%) 108 (61%/57%)
Gap = 1 93 (15%/13%) 61 (13%/8%) 32 (18%/21%)
Gap = 2 27 (4%/5%) 19 (4%/3%) 8 (5%/5%)
Gap = 3 19 (3%/5%) 17 (3%/9%) 2 (1%/2%)
Gap = 4 15 (2%/2%) 9 (2%/2%) 6 (3%/2%)
Gap = 5 60 (9%/11%) 38 (8%/11%) 22 (12%/13%)
Total Gap>0214 (33%/36%) 144 (30%/33%) 70 (39%/43%)
/uni00000036/uni00000013 /uni00000036/uni00000014 /uni00000036/uni00000015 /uni00000036/uni00000016 /uni00000036/uni00000017 /uni00000036/uni00000018
/uni00000036/uni0000005c/uni00000050/uni00000053/uni00000057/uni00000052/uni00000050/uni00000003/uni00000036/uni00000057/uni00000044/uni0000004a/uni00000048/uni00000036/uni00000013
/uni00000036/uni00000014
/uni00000036/uni00000015
/uni00000036/uni00000016
/uni00000036/uni00000017
/uni00000036/uni00000018/uni00000035/uni00000052/uni00000052/uni00000057/uni00000003/uni00000026/uni00000044/uni00000058/uni00000056/uni00000048/uni00000003/uni00000036/uni00000057/uni00000044/uni0000004a/uni00000048/uni00000014/uni0000001b/uni00000008
/uni0000000b/uni00000014/uni00000014/uni00000019/uni0000000c/uni00000014/uni00000014/uni00000008
/uni0000000b/uni00000019/uni0000001c/uni0000000c/uni00000013/uni00000008
/uni0000000b/uni00000016/uni0000000c/uni00000016/uni00000008
/uni0000000b/uni00000014/uni00000019/uni0000000c/uni00000014/uni00000008
/uni0000000b/uni00000017/uni0000000c/uni0000001a/uni00000008
/uni0000000b/uni00000017/uni0000001a/uni0000000c
/uni00000016/uni00000008
/uni0000000b/uni00000014/uni0000001b/uni0000000c/uni00000015/uni00000014/uni00000008
/uni0000000b/uni00000014/uni00000016/uni0000001b/uni0000000c/uni00000013/uni00000008
/uni0000000b/uni00000013/uni0000000c/uni00000015/uni00000008
/uni0000000b/uni00000014/uni00000019/uni0000000c/uni00000013/uni00000008
/uni0000000b/uni00000013/uni0000000c/uni00000014/uni00000008
/uni0000000b/uni00000018/uni0000000c
/uni00000013/uni00000008
/uni0000000b/uni00000013/uni0000000c/uni00000013/uni00000008
/uni0000000b/uni00000013/uni0000000c/uni00000016/uni00000008
/uni0000000b/uni00000015/uni00000013/uni0000000c/uni00000014/uni00000008
/uni0000000b/uni00000018/uni0000000c/uni00000013/uni00000008
/uni0000000b/uni00000013/uni0000000c/uni00000013/uni00000008
/uni0000000b/uni00000015/uni0000000c
/uni00000013/uni00000008
/uni0000000b/uni00000014/uni0000000c/uni00000013/uni00000008
/uni0000000b/uni00000013/uni0000000c/uni00000013/uni00000008
/uni0000000b/uni00000013/uni0000000c/uni00000018/uni00000008
/uni0000000b/uni00000016/uni00000016/uni0000000c/uni00000013/uni00000008
/uni0000000b/uni00000014/uni0000000c/uni00000014/uni00000008
/uni0000000b/uni0000001b/uni0000000c
/uni00000014/uni00000008
/uni0000000b/uni00000018/uni0000000c/uni00000013/uni00000008
/uni0000000b/uni00000013/uni0000000c/uni00000013/uni00000008
/uni0000000b/uni00000013/uni0000000c/uni00000013/uni00000008
/uni0000000b/uni00000013/uni0000000c/uni00000016/uni00000008
/uni0000000b/uni00000014/uni0000001b/uni0000000c/uni00000013/uni00000008
/uni0000000b/uni00000013/uni0000000c
/uni00000015/uni00000008
/uni0000000b/uni00000014/uni00000016/uni0000000c/uni00000013/uni00000008
/uni0000000b/uni00000014/uni0000000c/uni00000013/uni00000008
/uni0000000b/uni00000013/uni0000000c/uni00000013/uni00000008
/uni0000000b/uni00000013/uni0000000c/uni00000013/uni00000008
/uni0000000b/uni00000013/uni0000000c/uni00000014/uni0000001a/uni00000008
/uni0000000b/uni00000014/uni00000014/uni00000018/uni0000000c/uni0000003a/uni00000048/uni0000004c/uni0000004a/uni0000004b/uni00000057/uni00000048/uni00000047/uni00000003/uni0000000b/uni00000008/uni0000000c
/uni00000036/uni00000013 /uni00000036/uni00000014 /uni00000036/uni00000015 /uni00000036/uni00000016 /uni00000036/uni00000017 /uni00000036/uni00000018
/uni00000036/uni0000005c/uni00000050/uni00000053/uni00000057/uni00000052/uni00000050/uni00000003/uni00000036/uni00000057/uni00000044/uni0000004a/uni00000048/uni00000036/uni00000013
/uni00000036/uni00000014
/uni00000036/uni00000015
/uni00000036/uni00000016
/uni00000036/uni00000017
/uni00000036/uni00000018/uni00000035/uni00000052/uni00000052/uni00000057/uni00000003/uni00000026/uni00000044/uni00000058/uni00000056/uni00000048/uni00000003/uni00000036/uni00000057/uni00000044/uni0000004a/uni00000048/uni00000014/uni00000019/uni00000008
/uni0000000b/uni00000014/uni00000014/uni00000019/uni0000000c/uni00000014/uni00000013/uni00000008
/uni0000000b/uni00000019/uni0000001c/uni0000000c/uni00000013/uni00000008
/uni0000000b/uni00000016/uni0000000c/uni00000018/uni00000008
/uni0000000b/uni00000014/uni00000019/uni0000000c/uni00000013/uni00000008
/uni0000000b/uni00000017/uni0000000c/uni0000001b/uni00000008
/uni0000000b/uni00000017/uni0000001a/uni0000000c
/uni00000015/uni00000008
/uni0000000b/uni00000014/uni0000001b/uni0000000c/uni00000015/uni00000013/uni00000008
/uni0000000b/uni00000014/uni00000016/uni0000001b/uni0000000c/uni00000013/uni00000008
/uni0000000b/uni00000013/uni0000000c/uni00000016/uni00000008
/uni0000000b/uni00000014/uni00000019/uni0000000c/uni00000013/uni00000008
/uni0000000b/uni00000013/uni0000000c/uni00000014/uni00000008
/uni0000000b/uni00000018/uni0000000c
/uni00000013/uni00000008
/uni0000000b/uni00000013/uni0000000c/uni00000013/uni00000008
/uni0000000b/uni00000013/uni0000000c/uni00000016/uni00000008
/uni0000000b/uni00000015/uni00000013/uni0000000c/uni00000013/uni00000008
/uni0000000b/uni00000018/uni0000000c/uni00000013/uni00000008
/uni0000000b/uni00000013/uni0000000c/uni00000013/uni00000008
/uni0000000b/uni00000015/uni0000000c
/uni00000013/uni00000008
/uni0000000b/uni00000014/uni0000000c/uni00000013/uni00000008
/uni0000000b/uni00000013/uni0000000c/uni00000013/uni00000008
/uni0000000b/uni00000013/uni0000000c/uni00000016/uni00000008
/uni0000000b/uni00000016/uni00000016/uni0000000c/uni00000013/uni00000008
/uni0000000b/uni00000014/uni0000000c/uni00000015/uni00000008
/uni0000000b/uni0000001b/uni0000000c
/uni00000013/uni00000008
/uni0000000b/uni00000018/uni0000000c/uni00000013/uni00000008
/uni0000000b/uni00000013/uni0000000c/uni00000013/uni00000008
/uni0000000b/uni00000013/uni0000000c/uni00000013/uni00000008
/uni0000000b/uni00000013/uni0000000c/uni00000015/uni00000008
/uni0000000b/uni00000014/uni0000001b/uni0000000c/uni00000013/uni00000008
/uni0000000b/uni00000013/uni0000000c
/uni00000016/uni00000008
/uni0000000b/uni00000014/uni00000016/uni0000000c/uni00000013/uni00000008
/uni0000000b/uni00000014/uni0000000c/uni00000013/uni00000008
/uni0000000b/uni00000013/uni0000000c/uni00000013/uni00000008
/uni0000000b/uni00000013/uni0000000c/uni00000013/uni00000008
/uni0000000b/uni00000013/uni0000000c/uni00000015/uni00000014/uni00000008
/uni0000000b/uni00000014/uni00000014/uni00000018/uni0000000c/uni00000038/uni00000051/uni0000005a/uni00000048/uni0000004c/uni0000004a/uni0000004b/uni00000057/uni00000048/uni00000047/uni00000003/uni0000000b/uni00000008/uni0000000c
/uni00000013/uni00000015/uni00000013/uni00000017/uni00000013/uni00000019/uni00000013/uni0000001b/uni00000013/uni00000014/uni00000013/uni00000013
/uni00000033/uni00000048/uni00000055/uni00000046/uni00000048/uni00000051/uni00000057/uni00000044/uni0000004a/uni00000048
Fig. 6. Stage matrix for symptom and root cause.
4.4.1 Prevalence and Patterns of Stage Gap.As summarized in Table 9, 33%/36% (weighted/un-
weighted) of the configuration issues exhibit a non-zero stage gap between root cause and symptom.
This indicates that nearly one-third of configuration issues in RAG systems cannot be diagnosed by
inspecting the stage where the failure is observed alone. Notably, this gap is present in both primary
categories of the root cause, suggesting that cross-stage reasoning is a challenge in debugging the
configuration issues for RAG systems.
Figure 6 reveals that for those configuration issues with deviated stages, most commonly the
symptoms can be at eitherindexing(S1) orgeneration(S5) while the root cause is associated
withinitialization(S0) stage—the configuration issues introduced during systeminitializationonly
become apparent as abnormalities propagate to theindexingandgeneration.
Finding 7
Nearly one-third of RAG configuration issues exhibit a mismatch between the stages of symptom
and root cause. This often happens when root problems at theinitializationcause the symptoms
at eitherindexingorgenerationstage.
Manuscript submitted to ACM Trans. Softw. Eng. Methodol.

16 Chen et al.
Table 10. Median resolving cost over stage gap.
Stage gap 0 1 2 3 4 5
Median cost (h) 66.0 35.1 102.4 357.2 47.1 167.2
4.4.2 Impacts on Resolving Costs.To understand the impacts of the stage gap, we quantify the
duration required to resolve each configuration issue. Specifically, for each issue, the start time is
defined as the issue creation timestamp ( created_at ). The end time is determined depending on
two cases:
•if the issue is relevant to a fixing pull request, we use the pull request merge time ( merged_at );
•otherwise, we use the issue closing time (closed_at).
The resolving cost is calculated as the elapsed time between the start and end timestamps. From
Table 10, we observe that there seems to be a complex relationship between the stage gap and the
resolving cost. We hence take two further analyses.
Correlation Analysis.Our initial hypothesis is that there could be a monotonic relationship
between those two. To confirm this, we apply Spearman correlation 𝑟on the stage gap (G∈[ 0,5]) to
the resolving cost for all configuration issues. The result indicates that they have a trivial correlation
(𝑟≈ 0.068,𝑝-value≈0.08). As such, a larger stage gap might not mean that the configuration issue
takes longer to resolve.
Grouping Analysis.Based on the correlation results, we revised our hypothesis: there might
exist a“complexity threshold”where cross-stage debugging of configuration issues becomes sig-
nificantly harder. To that end, we apply the Mann-Whitney U test to compare the resolving costs
between two groups of issues separated by different thresholds, and we found that only when G= 2
yields statistically significant differences between the groups in the resolving cost: we evaluated
each feasible cutoff 𝑐∈{ 1,2,3,4,5}by comparing issues with G<𝑐 against those withG≥𝑐 . The
cutoffs 𝑐= 1,𝑐= 4, and 𝑐= 5did not produce statistically significant differences in resolving costs,
whereas both 𝑐= 2and 𝑐= 3produced significant differences. We selected 𝑐= 2for the reported
grouping because, compared with 𝑐= 3, it yielded a smaller overall 𝑝-value (0 .0025versus0 .0059),
a larger cost multiplier for configuration bugs (2 .83×versus2 .77×), and a more balanced high-gap
sample ( 𝑁= 121versus 𝑁= 94). Therefore, we dichotomize the configuration issues for RAG
systems as:
•Low gap (G∈[ 0,1]): Issues that are localized or manifest in the same or immediate adjacent
stage.
•High gap (G∈[2,5]): Issues that propagate more stages.
As shown in Table 11, we observe:
•Overall impact: For all configuration issues, those with the high gap have a median resolving
cost of 160.6 hours, which is 2.48 ×higher than that of those within a low gap (64.7 hours)
with statistical significance (𝑝<0.01).
•Configuration bugs (RC-A) as root causes: A deeper breakdown reveals that the large
resolving cost difference is primarily driven by configuration bugs: the cost multiplier rises
to 2.83×(𝑝< 0.01) in the high gap issues. This implies that when internal code logic fails and
propagates downstream, developers face substantial difficulty in tracing the RAG pipeline to
the root causes.
•Misconfiguration (RC-B) as root causes: For user misconfigurations, although the median
cost increases (1.87×) in the high gap group, the difference is not statistically significant. This
Manuscript submitted to ACM Trans. Softw. Eng. Methodol.

Configuring RAG Systems: What Breaks, Where Originates, and How to Fix 17
Table 11. Statistical difference of resolving costs (hours) on RAG configuration issues with low and high gap.
GroupLow Gap High Gap U Test
Count Median Count Median Increase𝑝-value
Overall533 64.7 121 160.6 2.48×0.003
RC-A: bugs 393 58.5 83 165.2 2.83×0.002
RC-B: misconfig. 140 78.3 38 146.1 1.87×0.273
confirms that while user misconfiguration issues exhibit large stage gaps, they are “cheaper”
to resolve (often a single option update).
Therefore, we conclude:
Finding 8
There is no monotonic correlation between stage gap size and resolving cost. Instead, we observe
a“threshold effect”: costs spike significantly when issues have G≥ 2, leading to largely deviated
costs mainly due to configuration bugs.
4.5 RQ4: Fix Pattern
We now focus on studying the 345 configuration issues for RAG systems, which are linked with
pull requests (including both configuration bugs and user misconfiguration). As shown in Table 12,
we see that for both weighted and unweighted percentages, thecomponent logic and algorithm
(FP2, 40%/43%) anddata interaction and mapping(FP1, 28%/26%) are among the most common
fix patterns. However, they are not necessarily those that require the greatest effort in terms of
fixing. Instead, patterns that involve deeper system-level intervention, such aslifecycle and resource
scheduling(FP6) exhibit noticeably higher median fix durations.
Specifically, graph-based and multi-source integrated systems, e.g.,GraphRAGandQuivr
requiredata interaction and mappingfix to resolve complex data structure mismatches, whereas
other systems heavily rely oncomponent logic and algorithm fixto repair their routing and execution
flows.
From Figure 7, we see several patterns between root causes and fix patterns, where the configu-
ration bugs are often linked todata interaction and mappingfixes (FP1) andcomponent logic and
algorithm fixes(FP2). Interestingly, misconfiguration can also lead to fixes, mainly on the improved
error checking/handling (FP3) and document/manual updates (FP5).
Finding 9
Thecomponent logic and algorithm(40%/43%) anddata interaction and mapping(28%/26%) are
the most common patterns. Yet, those patterns that require deeper system-level interaction often
require a higher cost to fix.
Finding 10
Different root causes indeed exhibit distinct repair tendencies: configuration bugs are often related
to data mapping and core logic fixes, while the misconfiguration can lead to improved error
checking and document updates.
5 Insights for Configuration of RAG Systems
Here, we discuss the insights derived from our empirical observations:
Manuscript submitted to ACM Trans. Softw. Eng. Methodol.

18 Chen et al.
Table 12. Fix pattern proportion by systems/categories. The same format as Table 4.
System FP1 FP2 FP3 FP4 FP5 FP6 Total
RAGFlow25% (40) 40% (64) 21% (34) 6% (9) 5% (8) 4% (6) 161
DB-GPT26% (20) 39% (30) 8% (6) 8% (6) 7% (5) 12% (9) 76
GraphRAG 50% (14) 32% (9) 11% (3) 4% (1) 0% (0) 4% (1) 28
LightRAG32% (7) 36% (8) 5% (1) 18% (4) 0% (0) 9% (2) 22
FastGPT26% (5) 47% (9) 5% (1) 11% (2) 0% (0) 11% (2) 19
Obsidian-copilot7% (1) 60% (9) 13% (2) 20% (3) 0% (0) 0% (0) 15
DocsGPT 40% (4) 20% (2) 10% (1) 0% (0) 20% (2) 10% (1) 10
Quivr 50% (4) 25% (2) 0% (0) 0% (0) 12% (1) 12% (1) 8
HippoRAG0% (0) 67% (2) 33% (1) 0% (0) 0% (0) 0% (0) 3
Private-GPT0% (0) 67% (2) 0% (0) 0% (0) 0% (0) 33% (1) 3
Weighted 28% (95) 40% (137) 14% (49) 7% (25) 5% (16) 6% (23) 345
Unweighted26% (95) 43% (137) 11% (49) 7% (25) 4% (16) 9% (23) 345
Median cost (h)31.7 58.6 41.0 30.4 63.1 103.2 –
/uni00000029/uni00000033/uni00000014 /uni00000029/uni00000033/uni00000015 /uni00000029/uni00000033/uni00000016 /uni00000029/uni00000033/uni00000017 /uni00000029/uni00000033/uni00000018 /uni00000029/uni00000033/uni00000019/uni00000035/uni00000026/uni00000010/uni00000024/uni00000014
/uni00000035/uni00000026/uni00000010/uni00000024/uni00000015
/uni00000035/uni00000026/uni00000010/uni00000024/uni00000016
/uni00000035/uni00000026/uni00000010/uni00000024/uni00000017
/uni00000035/uni00000026/uni00000010/uni00000024/uni00000018
/uni00000035/uni00000026/uni00000010/uni00000025/uni00000014
/uni00000035/uni00000026/uni00000010/uni00000025/uni00000015
/uni00000035/uni00000026/uni00000010/uni00000025/uni00000016
/uni00000035/uni00000026/uni00000010/uni00000025/uni00000017/uni00000016/uni00000017/uni00000008
/uni0000000b/uni00000015/uni00000019/uni0000000c/uni00000016/uni0000001a/uni00000008
/uni0000000b/uni00000015/uni0000001b/uni0000000c/uni00000014/uni00000014/uni00000008
/uni0000000b/uni0000001b/uni0000000c/uni00000014/uni00000016/uni00000008
/uni0000000b/uni00000014/uni00000013/uni0000000c/uni00000014/uni00000008
/uni0000000b/uni00000014/uni0000000c/uni00000017/uni00000008
/uni0000000b/uni00000016/uni0000000c
/uni00000014/uni00000015/uni00000008
/uni0000000b/uni00000014/uni00000013/uni0000000c/uni00000018/uni00000015/uni00000008
/uni0000000b/uni00000017/uni00000018/uni0000000c/uni00000015/uni00000019/uni00000008
/uni0000000b/uni00000015/uni00000015/uni0000000c/uni00000019/uni00000008
/uni0000000b/uni00000018/uni0000000c/uni00000014/uni00000008
/uni0000000b/uni00000014/uni0000000c/uni00000016/uni00000008
/uni0000000b/uni00000016/uni0000000c
/uni00000017/uni0000001b/uni00000008
/uni0000000b/uni00000014/uni0000001b/uni0000000c/uni00000016/uni0000001b/uni00000008
/uni0000000b/uni00000014/uni00000017/uni0000000c/uni00000014/uni00000014/uni00000008
/uni0000000b/uni00000017/uni0000000c/uni00000016/uni00000008
/uni0000000b/uni00000014/uni0000000c/uni00000013/uni00000008
/uni0000000b/uni00000013/uni0000000c/uni00000013/uni00000008
/uni0000000b/uni00000013/uni0000000c
/uni00000016/uni00000016/uni00000008
/uni0000000b/uni00000017/uni00000014/uni0000000c/uni00000016/uni0000001b/uni00000008
/uni0000000b/uni00000017/uni0000001a/uni0000000c/uni00000014/uni00000014/uni00000008
/uni0000000b/uni00000014/uni00000016/uni0000000c/uni00000019/uni00000008
/uni0000000b/uni0000001b/uni0000000c/uni00000019/uni00000008
/uni0000000b/uni0000001b/uni0000000c/uni00000019/uni00000008
/uni0000000b/uni0000001b/uni0000000c
/uni00000013/uni00000008
/uni0000000b/uni00000013/uni0000000c/uni00000014/uni0000001c/uni00000008
/uni0000000b/uni00000016/uni0000000c/uni00000019/uni00000008
/uni0000000b/uni00000014/uni0000000c/uni00000019/uni00000008
/uni0000000b/uni00000014/uni0000000c/uni00000014/uni00000016/uni00000008
/uni0000000b/uni00000015/uni0000000c/uni00000018/uni00000019/uni00000008
/uni0000000b/uni0000001c/uni0000000c
/uni00000013/uni00000008
/uni0000000b/uni00000013/uni0000000c/uni00000013/uni00000008
/uni0000000b/uni00000013/uni0000000c/uni00000014/uni00000013/uni00000013/uni00000008
/uni0000000b/uni00000014/uni0000000c/uni00000013/uni00000008
/uni0000000b/uni00000013/uni0000000c/uni00000013/uni00000008
/uni0000000b/uni00000013/uni0000000c/uni00000013/uni00000008
/uni0000000b/uni00000013/uni0000000c
/uni00000013/uni00000008
/uni0000000b/uni00000013/uni0000000c/uni00000013/uni00000008
/uni0000000b/uni00000013/uni0000000c/uni00000013/uni00000008
/uni0000000b/uni00000013/uni0000000c/uni00000013/uni00000008
/uni0000000b/uni00000013/uni0000000c/uni00000013/uni00000008
/uni0000000b/uni00000013/uni0000000c/uni00000013/uni00000008
/uni0000000b/uni00000013/uni0000000c
/uni00000013/uni00000008
/uni0000000b/uni00000013/uni0000000c/uni00000013/uni00000008
/uni0000000b/uni00000013/uni0000000c/uni00000013/uni00000008
/uni0000000b/uni00000013/uni0000000c/uni00000013/uni00000008
/uni0000000b/uni00000013/uni0000000c/uni00000013/uni00000008
/uni0000000b/uni00000013/uni0000000c/uni00000013/uni00000008
/uni0000000b/uni00000013/uni0000000c
/uni00000013/uni00000008
/uni0000000b/uni00000013/uni0000000c/uni00000013/uni00000008
/uni0000000b/uni00000013/uni0000000c/uni00000013/uni00000008
/uni0000000b/uni00000013/uni0000000c/uni00000013/uni00000008
/uni0000000b/uni00000013/uni0000000c/uni00000014/uni00000013/uni00000013/uni00000008
/uni0000000b/uni00000017/uni0000000c/uni00000013/uni00000008
/uni0000000b/uni00000013/uni0000000c/uni0000003a/uni00000048/uni0000004c/uni0000004a/uni0000004b/uni00000057/uni00000048/uni00000047/uni00000003/uni0000000b/uni00000008/uni0000000c
/uni00000029/uni00000033/uni00000014 /uni00000029/uni00000033/uni00000015 /uni00000029/uni00000033/uni00000016 /uni00000029/uni00000033/uni00000017 /uni00000029/uni00000033/uni00000018 /uni00000029/uni00000033/uni00000019/uni00000035/uni00000026/uni00000010/uni00000024/uni00000014
/uni00000035/uni00000026/uni00000010/uni00000024/uni00000015
/uni00000035/uni00000026/uni00000010/uni00000024/uni00000016
/uni00000035/uni00000026/uni00000010/uni00000024/uni00000017
/uni00000035/uni00000026/uni00000010/uni00000024/uni00000018
/uni00000035/uni00000026/uni00000010/uni00000025/uni00000014
/uni00000035/uni00000026/uni00000010/uni00000025/uni00000015
/uni00000035/uni00000026/uni00000010/uni00000025/uni00000016
/uni00000035/uni00000026/uni00000010/uni00000025/uni00000017/uni00000015/uni00000017/uni00000008
/uni0000000b/uni00000015/uni00000019/uni0000000c/uni00000016/uni00000016/uni00000008
/uni0000000b/uni00000015/uni0000001b/uni0000000c/uni00000019/uni00000008
/uni0000000b/uni0000001b/uni0000000c/uni00000014/uni0000001b/uni00000008
/uni0000000b/uni00000014/uni00000013/uni0000000c/uni00000018/uni00000008
/uni0000000b/uni00000014/uni0000000c/uni00000017/uni00000008
/uni0000000b/uni00000016/uni0000000c
/uni0000001b/uni00000008
/uni0000000b/uni00000014/uni00000013/uni0000000c/uni00000015/uni0000001c/uni00000008
/uni0000000b/uni00000017/uni00000018/uni0000000c/uni00000014/uni0000001a/uni00000008
/uni0000000b/uni00000015/uni00000015/uni0000000c/uni00000015/uni00000008
/uni0000000b/uni00000018/uni0000000c/uni00000014/uni00000008
/uni0000000b/uni00000014/uni0000000c/uni00000017/uni00000008
/uni0000000b/uni00000016/uni0000000c
/uni00000014/uni00000019/uni00000008
/uni0000000b/uni00000014/uni0000001b/uni0000000c/uni00000014/uni00000014/uni00000008
/uni0000000b/uni00000014/uni00000017/uni0000000c/uni00000015/uni00000008
/uni0000000b/uni00000017/uni0000000c/uni00000014/uni00000013/uni00000008
/uni0000000b/uni00000014/uni0000000c/uni00000013/uni00000008
/uni0000000b/uni00000013/uni0000000c/uni00000013/uni00000008
/uni0000000b/uni00000013/uni0000000c
/uni00000015/uni00000017/uni00000008
/uni0000000b/uni00000017/uni00000014/uni0000000c/uni00000017/uni0000001c/uni00000008
/uni0000000b/uni00000017/uni0000001a/uni0000000c/uni0000001a/uni00000008
/uni0000000b/uni00000014/uni00000016/uni0000000c/uni0000001a/uni00000008
/uni0000000b/uni0000001b/uni0000000c/uni00000018/uni00000008
/uni0000000b/uni0000001b/uni0000000c/uni0000001a/uni00000008
/uni0000000b/uni0000001b/uni0000000c
/uni00000013/uni00000008
/uni0000000b/uni00000013/uni0000000c/uni0000001c/uni00000008
/uni0000000b/uni00000016/uni0000000c/uni00000016/uni00000008
/uni0000000b/uni00000014/uni0000000c/uni00000018/uni00000008
/uni0000000b/uni00000014/uni0000000c/uni00000015/uni00000008
/uni0000000b/uni00000015/uni0000000c/uni00000016/uni00000013/uni00000008
/uni0000000b/uni0000001c/uni0000000c
/uni00000013/uni00000008
/uni0000000b/uni00000013/uni0000000c/uni00000013/uni00000008
/uni0000000b/uni00000013/uni0000000c/uni00000014/uni00000013/uni00000008
/uni0000000b/uni00000014/uni0000000c/uni00000013/uni00000008
/uni0000000b/uni00000013/uni0000000c/uni00000013/uni00000008
/uni0000000b/uni00000013/uni0000000c/uni00000013/uni00000008
/uni0000000b/uni00000013/uni0000000c
/uni00000013/uni00000008
/uni0000000b/uni00000013/uni0000000c/uni00000013/uni00000008
/uni0000000b/uni00000013/uni0000000c/uni00000013/uni00000008
/uni0000000b/uni00000013/uni0000000c/uni00000013/uni00000008
/uni0000000b/uni00000013/uni0000000c/uni00000013/uni00000008
/uni0000000b/uni00000013/uni0000000c/uni00000013/uni00000008
/uni0000000b/uni00000013/uni0000000c
/uni00000013/uni00000008
/uni0000000b/uni00000013/uni0000000c/uni00000013/uni00000008
/uni0000000b/uni00000013/uni0000000c/uni00000013/uni00000008
/uni0000000b/uni00000013/uni0000000c/uni00000013/uni00000008
/uni0000000b/uni00000013/uni0000000c/uni00000013/uni00000008
/uni0000000b/uni00000013/uni0000000c/uni00000013/uni00000008
/uni0000000b/uni00000013/uni0000000c
/uni00000013/uni00000008
/uni0000000b/uni00000013/uni0000000c/uni00000013/uni00000008
/uni0000000b/uni00000013/uni0000000c/uni00000013/uni00000008
/uni0000000b/uni00000013/uni0000000c/uni00000013/uni00000008
/uni0000000b/uni00000013/uni0000000c/uni00000015/uni00000013/uni00000008
/uni0000000b/uni00000017/uni0000000c/uni00000013/uni00000008
/uni0000000b/uni00000013/uni0000000c/uni00000038/uni00000051/uni0000005a/uni00000048/uni0000004c/uni0000004a/uni0000004b/uni00000057/uni00000048/uni00000047/uni00000003/uni0000000b/uni00000008/uni0000000c
/uni00000013/uni00000015/uni00000013/uni00000017/uni00000013/uni00000019/uni00000013/uni0000001b/uni00000013/uni00000014/uni00000013/uni00000013
/uni00000033/uni00000048/uni00000055/uni00000046/uni00000048/uni00000051/uni00000057/uni00000044/uni0000004a/uni00000048
Fig. 7. Distribution of fix patterns across different root causes.
Manuscript submitted to ACM Trans. Softw. Eng. Methodol.

Configuring RAG Systems: What Breaks, Where Originates, and How to Fix 19
5.1 For RAG System Developers
Insight 1: Prioritize detection on stages other than query or retrieval-related ones.
Most configuration issues in our dataset manifest as functional execution faults rather than
only as degraded answer quality. Core pipeline faults are the most common symptom category,
and symptoms concentrate at initialization, indexing, and generation. In contrast, query
transformation, retrieval, and post-retrieval processing account for a smaller share of directly
observable symptoms.
Guidance.Developers should treat configuration validation as a pipeline-level reliability mecha-
nism rather than as a retrieval-only quality check. A practical validation suite should include at
least three layers:
•Startup validation:check whether required environment variables, model credentials,
service endpoints, and database connections are available before the RAG service accepts
user requests.
•Indexing validation:run a small indexing job with a known document and verify that
chunking, embedding, metadata persistence, and vector-store insertion all complete success-
fully.
•Generation validation:run a short end-to-end query over a controlled fixture and verify
that the configured prompt template, LLM endpoint, context window, and output parser are
mutually compatible.
Retrieval metrics such as recall or ranking quality remain important, but they should not be the
only oracle. Many RAG configuration failures prevent the system from reaching the retrieval-quality
evaluation stage at all.
Example.Suppose a RAG framework allows users to configure embedding_model ,chunk_size ,
vector_store , and llm_provider . A developer can add a rag doctor command that performs the
following checks before deployment: instantiate the embedding model, index one sample document,
retrieve one chunk, and send a minimal prompt to the configured LLM. If the embedding dimension
does not match the vector database collection, the command should fail during indexing with an
explicit diagnostic, instead of letting the mismatch later appear as a retrieval or generation failure.
Insight 2: Focus debugging efforts on early-stage configurations.
Most root causes originate from the initialization and indexing stages. This means that
downstream failures are often caused by earlier configuration handling, resource binding,
model loading, document processing, or index construction decisions.
Guidance.When a configuration issue is reported, developers should first reconstruct the
early-stage configuration path:
(1)Identify which configuration source is authoritative, such as a YAML file, environment
variable, command-line argument, UI setting, or database record.
(2)Confirm how the option is parsed, normalized, and propagated into the initialization or
indexing component.
(3) Check whether the option silently falls back to a default value.
(4)Verify that derived artifacts, such as embedding dimensions, chunk metadata, index schema,
and collection names, match the configured values.
Manuscript submitted to ACM Trans. Softw. Eng. Methodol.

20 Chen et al.
This debugging order is especially useful when the symptom appears later in the pipeline. A
generation-stage error may still be caused by an incorrectly constructed index or by a missing
initialization-time dependency.
Example.A user reports that a generated answer ignores uploaded documents. Rather than
starting from prompt tuning, the maintainer first checks whether the configured document parser
and embedding model were used during indexing. The investigation reveals that the system accepted
a custom embedding model in the UI, but the indexing worker still used the default model because
the configuration was not propagated to the background job. The fix is to pass the selected model
ID into the indexing job and record the model metadata with the generated index.
Insight 3: Cross-stage bug localization is necessary.
Nearly one-third of configuration issues exhibit a non-zero stage gap between the root-cause
stage and the symptom stage. Therefore, inspecting only the stage where the failure becomes
visible is often insufficient.
Guidance.Developers should provide observability that connects configuration values to
pipeline artifacts across stages. Useful mechanisms include:
•a configuration provenance record that shows where each effective option came from;
•stage-level logs that include configuration fingerprints rather than only raw errors;
•metadata attached to indexed chunks, retrieved passages, and generated answers;
•an issue template that asks reporters for both runtime symptoms and upstream setup and
indexing details.
The goal is to make it possible to trace how a configuration value affects later-stage behavior.
Without this trace, maintainers may overfit the fix to the visible symptom and leave the upstream
cause unchanged.
Example.In a note-taking RAG application, a user chooses an “active note” mode expecting the
whole note to be passed to the LLM. The system actually interprets this mode as a retrieval strategy
over the note, and the symptom appears only when the LLM answers that it lacks enough context.
The visible failure is in generation, but the root cause is a retrieval-stage configuration misunder-
standing. A cross-stage trace should show: selected context mode →retrieval strategy→retrieved
passages→final prompt. This trace would guide both documentation and UI improvements.
Insight 4: Prioritize the fix of issues with large stage gaps.
Issues whose root causes and symptoms are separated by multiple pipeline stages incur
substantially higher resolving costs. In our analysis, high-gap issues ( G≥ 2) have a higher
median resolving cost than low-gap issues, and the effect is particularly strong for system
implementation defects.
Guidance.Maintainers should add stage-gap information to issue triage. Severity should still
matter, but a non-crashing issue with a large stage gap may consume more debugging time than a
localized crash with an obvious cause. A practical triage workflow can classify each report as:
•localized:symptom and suspected root cause are in the same or adjacent stages;
•propagated:symptom appears two or more stages away from the suspected root cause;
•unknown:insufficient evidence, requiring reproduction logs or configuration provenance.
Manuscript submitted to ACM Trans. Softw. Eng. Methodol.

Configuring RAG Systems: What Breaks, Where Originates, and How to Fix 21
Propagated issues should receive earlier maintainer attention because they often require under-
standing multiple components and data contracts.
Example.A generation-stage formatting failure may appear minor because the system still
returns an answer. However, if the actual cause is that indexing stored metadata under the wrong
key, then the issue spans indexing, retrieval, and generation. The maintainer should prioritize it over
a localized prompt typo because the metadata mismatch can affect multiple retrieval strategies and
future indexed data. A robust fix should migrate or normalize metadata at the indexing boundary,
not merely patch the final prompt.
Insight 5: Specific tools are required for configuration repair.
The most common fixes involve component logic and algorithms, followed by data interaction
and mapping. Therefore, RAG configuration repair requires tools that understand component
interactions, external dependencies, and pipeline data contracts, rather than only tools that
edit configuration files.
Guidance.Repair support should be configuration-aware and component-aware. Useful tool
features include:
•checking whether configuration keys are consistently mapped across UI, API, worker, and
storage layers;
•detecting incompatible component combinations, such as embedding models and existing
vector-store schemas;
•generating regression tests from real configuration issues;
•suggesting code-level repairs when the system mishandles valid configurations.
This is different from a generic linter. Many observed repairs modify internal logic, interface
mapping, or lifecycle management, which means the correct repair often belongs in framework
code rather than in the user’s configuration file.
Example.A vector database adapter changes its expected field name from metadata topayload .
Existing configurations still point to the same database, but retrieval fails because the RAG system
maps chunk metadata to the old field. A configuration repair assistant should detect the adapter
version, compare the expected schema with the configured mapping, and recommend an interface-
level fix. The repair may update the adapter code and add a compatibility test for both field names.
5.2 For RAG Systems Users
Insight 6: Validate external artefacts before deployment.
External configuration mistakes are the most common type of user misconfiguration. These
mistakes involve external artefacts such as LLMs, embedding models, vector databases, re-
rankers, local model files, credentials, and service endpoints.
Guidance.Users should validate external artefacts before relying on a full RAG deployment. A
lightweight pre-deployment checklist should include:
•the selected LLM endpoint is reachable and supports the configured context length, model
name, authentication mode, and streaming option;
•the embedding model is available and its output dimension matches the vector database
collection;
Manuscript submitted to ACM Trans. Softw. Eng. Methodol.

22 Chen et al.
•the vector database is reachable, writable, and uses the expected metric, schema, and collection
name;
•any re-ranker, parser, or external application integration uses a compatible API version.
The key is to test each external artefact independently before debugging the entire RAG pipeline.
Example.A user switches from one embedding provider to another because the new model is
cheaper. Before reusing the old vector collection, the user should check the new model’s embedding
dimension. If the old collection was built with 1536-dimensional vectors and the new model returns
1024-dimensional vectors, the correct action is to rebuild the index or create a separate collection.
Otherwise, the system may fail during insertion or silently retrieve poor matches.
Insight 7: Pay close attention to configuration semantics and documentation.
Some user misconfigurations arise from misunderstanding option semantics, constraints,
deprecated parameters, or misleading documentation. A configuration can be syntactically
valid but still conflict with the intended RAG behavior.
Guidance.Users should not treat defaults and option names as complete specifications. Before
production deployment, they should inspect the semantic effects of critical options, especially those
controlling context construction, chunking, retrieval mode, model selection, prompt templates, and
resource limits. When documentation is ambiguous, users should run a small controlled example
and inspect intermediate artifacts such as chunks, retrieved passages, and final prompts.
Example.A configuration option named context_mode=active may sound as if the system
passes the entire active document to the LLM. In a particular implementation, it may instead mean
that the active document is used as a search scope for similarity-based retrieval. A user preparing a
legal or academic assistant should verify the final prompt for a test query. If only a few retrieved
snippets appear, the user may need a different mode, a larger top_k , or a custom prompt assembly
strategy.
5.3 For RAG Researchers
Insight 8: Explore compatibility-sensitive configuration assistants.
External system adaptation bugs are the most prevalent system-side root cause. This suggests
that RAG systems need support for handling evolving APIs, changing model capabilities,
incompatible vector database schemas, and heterogeneous component assumptions.
Guidance.Researchers can design assistants that reason about compatibility as a first-class
property. Promising directions include:
•mining dependency versions, model cards, API specifications, and adapter code to infer
compatibility constraints;
•generating warnings when a selected LLM, embedding model, vector database, or re-ranker
violates known constraints;
•learning from historical issue-fix pairs to identify fragile configuration combinations;
•continuously checking compatibility as external services evolve.
Such assistants should explain why a configuration is risky, not merely report that it is invalid.
Many failures are caused by implicit assumptions that are not encoded in a single configuration
schema.
Manuscript submitted to ACM Trans. Softw. Eng. Methodol.

Configuring RAG Systems: What Breaks, Where Originates, and How to Fix 23
Example.A research prototype could monitor a RAG project’s dependency file, environment
variables, and adapter implementations. When the vector database client is upgraded, the assistant
checks whether the configured collection schema and metadata mapping still match the adapter’s
expected API. If not, it suggests a migration plan and generates a regression test that indexes,
retrieves, and generates over a small fixture.
Insight 9: Investigate pipeline-aware localization and repair techniques.
Cross-stage issue propagation and the high resolving cost of large-gap issues indicate that RAG
debugging tools should model the pipeline structure explicitly. Stage-agnostic bug localization
may miss upstream causes that only become visible after several transformations.
Guidance.Future localization and repair techniques should represent RAG execution as a
chain of configuration-dependent transformations: initialization, indexing, query transformation,
retrieval, post-retrieval processing, and generation. A pipeline-aware tool can then reason over
both control flow and data flow:
•which configuration value affects which component;
•which intermediate artifact was produced under which configuration;
•where the first divergence from an expected artifact occurs;
•whether the repair should update user documentation, configuration validation, adapter
mapping, or core pipeline logic.
Example.Consider an issue where the final answer cites irrelevant documents. A pipeline-aware
localizer should compare the execution trace against a known-good fixture: initialization loads the
expected model, indexing creates chunks with correct metadata, query transformation preserves
user intent, retrieval returns the expected candidates, post-retrieval filtering keeps valid passages,
and generation uses the intended prompt template. If the first divergence appears in post-retrieval
filtering because a configured metadata key was renamed during indexing, the repair should target
metadata propagation rather than the LLM prompt.
6 Threats to Validity
To mitigate threats tointernal validity, all authors are involved in the manual analysis for
reaching agreements. We also calculate the Cohen’s 𝜅coefficient to evaluate inter-rater reliability
iteratively—we stop only when there is 𝜅≥ 0.7for all categories. Our categorization is also
supported by discussion and agreement among all authors, together with external experts when
needed. Yet, we agree that we might not be able to completely rule out subjectivity.
External validityconcerns the generalizability. Although we analyzed 654 issues from 10
popular RAG systems, which serve as good representations, our results may not fully generalize to
all RAG systems, particularly proprietary systems. Our selection criteria favor popular and active
open-source repositories with sufficiently rich issue histories. Therefore, the findings may not
generalize to proprietary or long-tail RAG systems, newly emerging projects with fewer stars, or
systems whose configuration problems are not publicly reported. Moreover, we did not explicitly
select or analyze systems according to agentic or cyclic execution patterns. Caution is therefore
required when extending our findings to newly emerging types of RAG systems involving planning,
tool use, memory, iterative retrieval, or multi-agent orchestration.
Construct validityrelates to the metrics/procedure used in the study. We choose Cohen’s
𝜅as it has been widely used for empirical studies in software engineering [ 14,46]; the same
for Spearman correlation and Mann-Whitney U test. Further, we use elapsed calendar time as
Manuscript submitted to ACM Trans. Softw. Eng. Methodol.

24 Chen et al.
an observable proxy for resolving cost rather than a direct measurement of developers’ active
debugging effort. The interval may also include issue triage, waiting for maintainer responses,
implementation scheduling, code review, and periods of inactivity. It might be affected by repository
activity, maintainer availability, issue priority and severity, reporting quality, and project-specific
development practices.
7 Related Work
Empirical studies on AI systems.A growing body of empirical research investigates reliability
challenges in AI-driven software systems [ 13,34,43,47,52,57]. Long and Chen [ 34] investigate
the reporting and resolution of performance and accuracy bugs in deep learning frameworks. Shao
et al. [ 43] study LLM-integrated software applications and systematically analyzed their integration
defects, while Xue et al. [ 52] examine the symptoms and root causes of bugs in LLM agent orches-
tration frameworks. While these studies provide valuable empirical insights into the reliability and
development challenges of AI systems, they primarily focus on deep learning frameworks and LLM-
based applications, leaving the configuration of RAG systems largely unexplored. There are studies
that systematically characterized the RAG landscape, detailing its architectural evolution and core
paradigms [ 2,17,18,39,42]. Among them, Gao et al. [ 18] provide a survey of RAG, categorizing
its evolution into different types. Fan et al. [ 17] systematically review mainstream relevant work
by RAG architectures, training strategies, and application areas. Şakar and Emekci [ 42] address
the optimization of RAG processes by further exploring various methodologies. However, those
works do not focus on real-world bug patterns or configuration-induced issues of RAG systems.
Some other efforts are specific to RAG deployments [ 1,4,30]. Agrawal et al. [ 1] identify failure
points in knowledge graph-based RAG systems, while Leung et al. [ 30] develop a taxonomy of
RAG errors and propose an automated evaluation approach. Barnett et al. [ 4] identify seven failure
points in RAG engineering through three case studies. However, these studies primarily focus on
RAG failure modes and output errors, without systematically investigating configuration-related
issues and their relationships with the RAG pipeline.
Unlike these studies, our work conducts a systematic empirical investigation of real-world
configuration issues in RAG systems, characterizing their symptoms, root causes, and fix patterns.
In particular, we examine how configuration issues manifest and originate across different RAG
pipeline stages, revealing the discrepancies between symptom and root-cause locations and their
implications for issue resolution which are previously unknown.
Empirical studies on configurable systems.Configuration-related errors have long been
recognized as a key failure in general software systems. Classic studies [ 16,33] analyze real-world
misconfigurations to understand their root causes and impact on system availability. Design-
centric research [ 50,51] explore the challenges of configuration design and usage, highlighting
how inadequate configuration handling and excessive configuration options complicate system
management and increase the likelihood of user errors. Performance-focused studies [ 10,11,22,24,
25,28,55] characterize configuration-related performance bugs where internal code defects trigger
performance degradation. Other studies [22, 28] empirically characterize real-world performance
bugs, including the challenges of testing and debugging them in highly configurable systems.
Subsequent research [ 24,25] specifically investigates configuration-related performance bugs
and develops detection techniques based on unexpected performance behaviors under different
configuration settings.
Being a new type of system, RAG systems introduce distinct configuration challenges that may
limit the direct applicability of findings from traditional configurable systems, for two main reasons:
Manuscript submitted to ACM Trans. Softw. Eng. Methodol.

Configuring RAG Systems: What Breaks, Where Originates, and How to Fix 25
•Heterogeneous component integration.RAG systems are integration-driven applications
that depend on diverse external components, such as LLM APIs, embedding models, and
vector databases. Their configurations must accommodate evolving interfaces, component
dependencies, and compatibility requirements, which forms special categories for RAG
systems such asinter-component implementation bug.
•Multi-stage pipeline.RAG systems operate in their unique multi-stage pipelines, where
configuration options influence inter-stage data flow and semantic processing logic. Config-
uration choices made at one stage can influence downstream data processing and system
behavior, potentially causing failures to manifest at stages different from where their root
causes originate.
Those characteristics are what make our configuration study unique, as we unveil new, previously
unknown findings/insights specifically on RAG systems.
Configuration testing, tuning, and modeling.Configurable systems have been widely re-
searched, including performance modeling [ 20,45,48] for learning the effects of configuration
options and their interactions on system performance under varying conditions; configuration
tuning [ 8,9,12,49,53,54] for efficiently searching for configurations that optimize system perfor-
mance; and configuration testing [ 36] for efficiently detecting configuration-related performance
bugs. Additionally, a major line of work focuses on the automated diagnosis of configuration errors
and detection of performance bugs [ 23,36,56], leveraging program analysis, historical bug reports,
and testing prioritization techniques to identify and diagnose defects. Other studies investigate
automated configuration optimization and adaptation for RAG systems [ 3,5,7,41], aiming to
improve answer quality, latency, and cost efficiency through parameter optimization, configuration
selection, and workload-aware adaptation. Our study is complementary to those by revealing the
unique characteristics of configuration issues for RAG systems.
8 Conclusion
In the paper, we conduct an empirical study consisting of 10 RAG systems and 654 configuration
issues on GitHub. From this, we summarize a categorization that documents symptoms, root causes,
relevant stages, and fix patterns for configuration issues of RAG systems. In particular, we reveal ten
previously unknown findings. Drawing on those, we conclude several actionable insights for future
configuration engineering of RAG systems. These results highlight that configuration reliability
in RAG systems is not merely an option-selection concern, but a stage-dependent engineering
challenge rooted in their integration complexity. We hope this work lays the foundation for more
principled debugging and design practices of the next-generation RAG systems.
References
[1]Garima Agrawal, Tharindu Kumarage, Zeyad Alghamdi, and Huan Liu. 2024. Mindful-RAG: A Study of Points of
Failure in Retrieval Augmented Generation. In2nd International Conference on Foundation and Large Language Models,
FLLM 2024, Dubai, United Arab Emirates, November 26-29, 2024. IEEE, 607–611. doi:10.1109/FLLM63129.2024.10852457
[2]Muhammad Arslan, Hussam Ghanem, Saba Munawar, and Christophe Cruz. 2024. A Survey on RAG with LLMs.
Procedia computer science246 (2024), 3781–3790.
[3]Matthew Barker, Andrew Bell, Evan Thomas, James Carr, Thomas Andrews, and Umang Bhatt. 2025. Faster, cheaper,
better: Multi-objective hyperparameter optimization for LLM and RAG systems.arXiv preprint arXiv:2502.18635(2025).
[4]Scott Barnett, Stefanus Kurniawan, Srikanth Thudumu, Zach Brannelly, and Mohamed Abdelrazek. 2024. Seven
Failure Points When Engineering a Retrieval Augmented Generation System. InProceedings of the IEEE/ACM 3rd
International Conference on AI Engineering - Software Engineering for AI, CAIN 2024, Lisbon, Portugal, April 14-15, 2024,
Jane Cleland-Huang, Jan Bosch, Henry Muccini, and Grace A. Lewis (Eds.). ACM, 194–199. doi:10.1145/3644815.3644945
[5]Archisman Bhowmick and Rekha Singhal. 2025. OptiRAG: Optimized RAG Configuration Under Performance Con-
straints. In2025 5th International Conference on AI-ML-Systems (AIMLSystems). 318–323. doi:10.1109/AIMLSystems67835.
2025.11330856
Manuscript submitted to ACM Trans. Softw. Eng. Methodol.

26 Chen et al.
[6]Yupeng Chang, Xu Wang, Jindong Wang, Yuan Wu, Linyi Yang, Kaijie Zhu, Hao Chen, Xiaoyuan Yi, Cunxiang Wang,
Yidong Wang, et al .2024. A survey on evaluation of large language models.ACM transactions on intelligent systems
and technology15, 3 (2024), 1–45.
[7]Pengzhou Chen and Tao Chen. 2026. CDS4RAG: Cyclic Dual-Sequential Hyperparameter Optimization for RAG. In
Proceedings of the Thirty-Fifth International Joint Conference on Artificial Intelligence and the 29th European Conference
on Artificial Intelligence, IJCAI/ECAI 2026, Bremen, Germany, 15-21 August 2026. ijcai.org, 4188–4196. doi:10.24963/
IJCAI.2026/466
[8]Pengzhou Chen and Tao Chen. 2026. PromiseTune: Unveiling Causally Promising and Explainable Configuration
Tuning. InProceedings of the 2026 IEEE/ACM 48th International Conference on Software Engineering(Rio de Janeiro,
Brazil)(ICSE ’26). Association for Computing Machinery, New York, NY, USA, 14–26. doi:10.1145/3744916.3764552
[9]Pengzhou Chen, Tao Chen, and Miqing Li. 2024. MMO: Meta Multi-Objectivization for Software Configuration Tuning.
IEEE Trans. Software Eng.50, 6 (2024), 1478–1504. doi:10.1109/TSE.2024.3388910
[10] Pengzhou Chen, Jingzhi Gong, and Tao Chen. 2025. Accuracy Can Lie: On the Impact of Surrogate Model in Configu-
ration Tuning.IEEE Trans. Software Eng.51, 2 (2025), 548–580. doi:10.1109/TSE.2025.3525955
[11] Pengzhou Chen, Hongyuan Liang, and Tao Chen. 2026. Evaluating Useful Surrogate Models for Configuration
Tuning Beyond Accuracy: A Fitness Landscape Analysis Perspective.ACM Trans. Softw. Eng. Methodol.(Aug. 2026).
doi:10.1145/3844143 Just Accepted.
[12] Tao Chen and Miqing Li. 2021. Multi-objectivizing software configuration tuning. InProceedings of the 29th ACM
Joint Meeting on European Software Engineering Conference and Symposium on the Foundations of Software Engineering
(ESEC/FSE ’21). ACM, 453–465. doi:10.1145/3468264.3468555
[13] Xiang Chen, Chaoyang Gao, Chunyang Chen, Guangbei Zhang, and Yong Liu. 2025. An Empirical Study on Challenges
for LLM Application Developers.ACM Trans. Softw. Eng. Methodol.34, 7, Article 205 (Aug. 2025), 37 pages. doi:10.
1145/3715007
[14] Jithin Cheriyan, Bastin Tony Roy Savarimuthu, and Stephen Cranefield. 2021. Towards offensive language detection and
reduction in four software engineering communities. InProceedings of the 25th International Conference on Evaluation
and Assessment in Software Engineering. 254–259.
[15] Daniela S. Cruzes and Tore Dyba. 2011. Recommended Steps for Thematic Synthesis in Software Engineering. In
Proceedings of the 2011 International Symposium on Empirical Software Engineering and Measurement (ESEM ’11). IEEE
Computer Society, USA, 275–284. doi:10.1109/ESEM.2011.36
[16] Constanze Dietrich, Katharina Krombholz, Kevin Borgolte, and Tobias Fiebig. 2018. Investigating System Operators’
Perspective on Security Misconfigurations. InProceedings of the 2018 ACM SIGSAC Conference on Computer and
Communications Security(Toronto, Canada)(CCS ’18). Association for Computing Machinery, New York, NY, USA,
1272–1289. doi:10.1145/3243734.3243794
[17] Wenqi Fan, Yujuan Ding, Liangbo Ning, Shijie Wang, Hengyun Li, Dawei Yin, Tat-Seng Chua, and Qing Li. 2024. A
survey on rag meeting llms: Towards retrieval-augmented large language models. InProceedings of the 30th ACM
SIGKDD conference on knowledge discovery and data mining. 6491–6501.
[18] Yunfan Gao, Yun Xiong, Xinyu Gao, Kangxiang Jia, Jinliu Pan, Yuxi Bi, Yi Dai, Jiawei Sun, Qianyu Guo, Meng Wang,
and Haofen Wang. 2023. Retrieval-Augmented Generation for Large Language Models: A Survey.CoRRabs/2312.10997
(2023). arXiv:2312.10997 doi:10.48550/ARXIV.2312.10997
[19] Yunfan Gao, Yun Xiong, Meng Wang, and Haofen Wang. 2024. Modular rag: Transforming rag systems into lego-like
reconfigurable frameworks.arXiv preprint arXiv:2407.21059(2024).
[20] Jingzhi Gong, Tao Chen, and Rami Bahsoon. 2025. Dividable Configuration Performance Learning.IEEE Trans. Software
Eng.51, 1 (2025), 106–134. doi:10.1109/TSE.2024.3491945
[21] Kelvin Guu, Kenton Lee, Zora Tung, Panupong Pasupat, and Mingwei Chang. 2020. Retrieval augmented language
model pre-training. InInternational conference on machine learning. PMLR, 3929–3938.
[22] Xue Han and Tingting Yu. 2016. An empirical study on performance bugs for highly configurable software systems. In
Proceedings of the 10th ACM/IEEE International Symposium on Empirical Software Engineering and Measurement. 1–10.
[23] Xue Han, Tingting Yu, and David Lo. 2018. Perflearner: Learning from bug reports to understand and generate
performance test frames. InProceedings of the 33rd ACM/IEEE international conference on automated software engineering.
17–28.
[24] Haochen He. 2019. Tuning backfired? Not (always) your fault: Understanding and detecting configuration-related
performance bugs. InProceedings of the 2019 27th ACM Joint Meeting on European Software Engineering Conference and
Symposium on the Foundations of Software Engineering. 1229–1231.
[25] Haochen He, Zhouyang Jia, Shanshan Li, Erci Xu, Tingting Yu, Yue Yu, Ji Wang, and Xiangke Liao. 2020. Cp-detector:
Using configuration-related performance properties to expose performance bugs. InProceedings of the 35th IEEE/ACM
International Conference on Automated Software Engineering. 623–634.
Manuscript submitted to ACM Trans. Softw. Eng. Methodol.

Configuring RAG Systems: What Breaks, Where Originates, and How to Fix 27
[26] Hangfeng He, Hongming Zhang, and Dan Roth. 2022. Rethinking with retrieval: Faithful large language model
inference.arXiv preprint arXiv:2301.00303(2022).
[27] Ziwei Ji, Nayeon Lee, Rita Frieske, Tiezheng Yu, Dan Su, Yan Xu, Etsuko Ishii, Ye Jin Bang, Andrea Madotto, and
Pascale Fung. 2023. Survey of hallucination in natural language generation.ACM computing surveys55, 12 (2023),
1–38.
[28] Guoliang Jin, Linhai Song, Xiaoming Shi, Joel Scherpelz, and Shan Lu. 2012. Understanding and detecting real-world
performance bugs.SIGPLAN Not.47, 6 (June 2012), 77–88. doi:10.1145/2345156.2254075
[29] J Richard Landis and Gary G Koch. 1977. The measurement of observer agreement for categorical data.biometrics
(1977), 159–174.
[30] Kin Kwan Leung, Mouloud Belbahri, Yi Sui, Alex Labach, Xueying Zhang, Stephen Anthony Rose, and Jesse C. Cresswell.
2026. Classifying and Addressing the Diversity of Errors in Retrieval-Augmented Generation Systems. InProceedings
of the 19th Conference of the European Chapter of the Association for Computational Linguistics (Volume 1: Long Papers),
Vera Demberg, Kentaro Inui, and Lluís Marquez (Eds.). Association for Computational Linguistics, Rabat, Morocco,
3185–3207. doi:10.18653/v1/2026.eacl-long.147
[31] Patrick Lewis, Ethan Perez, Aleksandra Piktus, Fabio Petroni, Vladimir Karpukhin, Naman Goyal, Heinrich Küttler,
Mike Lewis, Wen-tau Yih, Tim Rocktäschel, Sebastian Riedel, and Douwe Kiela. 2020. Retrieval-Augmented Generation
for Knowledge-Intensive NLP Tasks. InAdvances in Neural Information Processing Systems 33: Annual Conference on
Neural Information Processing Systems 2020, NeurIPS 2020, December 6-12, 2020, virtual, Hugo Larochelle, Marc’Aurelio
Ranzato, Raia Hadsell, Maria-Florina Balcan, and Hsuan-Tien Lin (Eds.). https://proceedings.neurips.cc/paper/2020/
hash/6b493230205f780e1bc26945df7481e5-Abstract.html
[32] Siran Li, Linus Stenzel, Carsten Eickhoff, and Seyed Ali Bahrainian. 2025. Enhancing retrieval-augmented generation:
a study of best practices. InProceedings of the 31st International Conference on Computational Linguistics. 6705–6717.
[33] Yuhao Liu, Yingnan Zhou, Hanfeng Zhang, Zhiwei Chang, Sihan Xu, Yan Jia, Wei Wang, and Zheli Liu. 2024. Rethinking
software misconfigurations in the real world: an empirical study and literature analysis.arXiv preprint arXiv:2412.11121
(2024).
[34] Guoming Long and Tao Chen. 2022. On Reporting Performance and Accuracy Bugs for Deep Learning Frameworks:
An Exploratory Study from GitHub. InProceedings of the 26th International Conference on Evaluation and Assessment in
Software Engineering(Gothenburg, Sweden)(EASE ’22). Association for Computing Machinery, New York, NY, USA,
90–99. doi:10.1145/3530019.3530029
[35] Junliang Luo, Tianyu Li, Di Wu, Michael Jenkin, Steve Liu, and Gregory Dudek. 2024. Hallucination detection and
hallucination mitigation: An investigation.arXiv preprint arXiv:2401.08358(2024).
[36] Youpeng Ma, Tao Chen, and Ke Li. 2025.Faster Configuration Performance Bug Testing with Neural Dual-Level
Prioritization. IEEE Press, 988–1000. https://doi.org/10.1109/ICSE55347.2025.00201
[37] Humza Naveed, Asad Ullah Khan, Shi Qiu, Muhammad Saqib, Saeed Anwar, Muhammad Usman, Naveed Akhtar, Nick
Barnes, and Ajmal Mian. 2025. A comprehensive overview of large language models.ACM Transactions on Intelligent
Systems and Technology16, 5 (2025), 1–72.
[38] Joanne Neale. 2016. Iterative categorization (IC): a systematic technique for analysing qualitative data.Addiction111,
6 (2016), 1096–1106. arXiv:https://onlinelibrary.wiley.com/doi/pdf/10.1111/add.13314 doi:10.1111/add.13314
[39] Agada Joseph Oche, Ademola Glory Folashade, Tirthankar Ghosal, and Arpan Biswas. 2025. A systematic review of key
retrieval-augmented generation (rag) systems: Progress, gaps, and future directions.arXiv preprint arXiv:2507.18910
(2025).
[40] Vipula Rawte, Amit Sheth, and Amitava Das. 2023. A survey of hallucination in large foundation models.arXiv
preprint arXiv:2309.05922(2023).
[41] Siddhant Ray, Rui Pan, Zhuohan Gu, Kuntai Du, Shaoting Feng, Ganesh Ananthanarayanan, Ravi Netravali, and
Junchen Jiang. 2025. Metis: fast quality-aware rag systems with configuration adaptation. InProceedings of the ACM
SIGOPS 31st symposium on operating systems principles. 606–622.
[42] Tolga Şakar and Hakan Emekci. 2025. Maximizing RAG efficiency: A comparative analysis of RAG methods.Natural
Language Processing31, 1 (2025), 1–25.
[43] Yuchen Shao, Yuheng Huang, Jiawei Shen, Lei Ma, Ting Su, and Chengcheng Wan. 2025. Are LLMs Correctly Integrated
into Software Systems?. In47th IEEE/ACM International Conference on Software Engineering, ICSE 2025, Ottawa, ON,
Canada, April 26 - May 6, 2025. IEEE, 1178–1190. doi:10.1109/ICSE55347.2025.00204
[44] Xinyue Shen, Zeyuan Chen, Michael Backes, and Yang Zhang. 2023. In chatgpt we trust? measuring and characterizing
the reliability of chatgpt.arXiv preprint arXiv:2304.08979(2023).
[45] Norbert Siegmund, Alexander Grebhahn, Sven Apel, and Christian Kästner. 2015. Performance-influence models for
highly configurable systems. InProceedings of the 2015 10th Joint Meeting on Foundations of Software Engineering.
284–294.
Manuscript submitted to ACM Trans. Softw. Eng. Methodol.

28 Chen et al.
[46] Simin Wang, Liguo Huang, Amiao Gao, Jidong Ge, Tengfei Zhang, Haitao Feng, Ishna Satyarth, Ming Li, He Zhang,
and Vincent Ng. 2023. Machine/Deep Learning for Software Engineering: A Systematic Literature Review.IEEE
Transactions on Software Engineering49, 3 (2023), 1188–1231. doi:10.1109/TSE.2022.3173346
[47] Siyu Wu, Yulong Ye, Zezhen Xiang, Pengzhou Chen, Gangda Xiong, and Tao Chen. 2026. LLMSYS-HPOBench: Hyper-
parameter Optimization Benchmark Suite for Real-World LLM Systems.CoRRabs/2605.08305 (2026). arXiv:2605.08305
doi:10.48550/ARXIV.2605.08305
[48] Zezhen Xiang, Jingzhi Gong, and Tao Chen. 2026. Dually Hierarchical Drift Adaptation for Online Configuration
Performance Learning. In48th IEEE/ACM International Conference on Software Engineering (ICSE). ACM.
[49] Gangda Xiong and Tao Chen. 2025. CoTune: Co-evolutionary Configuration Tuning. In40th IEEE/ACM International
Conference on Automated Software Engineering, ASE 2025, Seoul, Korea, Republic of, November 16-20, 2025. IEEE, 1490–
1502. doi:10.1109/ASE63991.2025.00126
[50] Tianyin Xu, Long Jin, Xuepeng Fan, Yuanyuan Zhou, Shankar Pasupathy, and Rukma Talwadker. 2015. Hey, you
have given me too many knobs!: Understanding and dealing with over-designed configuration in system software. In
Proceedings of the 2015 10th Joint Meeting on Foundations of Software Engineering. 307–319.
[51] Tianyin Xu, Jiaqi Zhang, Peng Huang, Jing Zheng, Tianwei Sheng, Ding Yuan, Yuanyuan Zhou, and Shankar Pasupathy.
2013. Do not blame users for misconfigurations. InProceedings of the Twenty-Fourth ACM Symposium on Operating
Systems Principles(Farminton, Pennsylvania)(SOSP ’13). Association for Computing Machinery, New York, NY, USA,
244–259. doi:10.1145/2517349.2522727
[52] Ziluo Xue, Yanjie Zhao, Shenao Wang, Kai Chen, and Haoyu Wang. 2025. A Characterization Study of Bugs in LLM
Agent Workflow Orchestration Frameworks. In2025 40th IEEE/ACM International Conference on Automated Software
Engineering (ASE). IEEE, 3369–3380.
[53] Yulong Ye, Tao Chen, and Miqing Li. 2025. Distilled Lifelong Self-Adaptation for Configurable Systems. In47th
IEEE/ACM International Conference on Software Engineering, ICSE 2025, Ottawa, ON, Canada, April 26 - May 6, 2025.
IEEE, 1333–1345. doi:10.1109/ICSE55347.2025.00094
[54] Yulong Ye, Miqing Li, and Tao Chen. 2026. Less Is More: Tuning Configurable Systems with Imperfect Fidelity(ASE
’26). Association for Computing Machinery, New York, NY, USA, 575–587. doi:10.1145/3832783.3834376
[55] Yulong Ye, Hongyuan Liang, Chao Jiang, Miqing Li, and Tao Chen. 2026. Revealing Domain-Spatiality Patterns for
Configuration Tuning: Domain Knowledge Meets Fitness Landscapes.ACM Trans. Softw. Eng. Methodol.(March 2026).
doi:10.1145/3803859 Just Accepted.
[56] Sai Zhang and Michael D. Ernst. 2013. Automated diagnosis of software configuration errors. InProceedings of the 2013
International Conference on Software Engineering(San Francisco, CA, USA)(ICSE ’13). IEEE Press, 312–321.
[57] Tianyi Zhang, Cuiyun Gao, Lei Ma, Michael Lyu, and Miryung Kim. 2019. An Empirical Study of Common Challenges in
Developing Deep Learning Applications. In2019 IEEE 30th International Symposium on Software Reliability Engineering
(ISSRE). 104–115. doi:10.1109/ISSRE.2019.00020
[58] Wayne Xin Zhao, Kun Zhou, Junyi Li, Tianyi Tang, Xiaolei Wang, Yupeng Hou, Yingqian Min, Beichen Zhang, Junjie
Zhang, Zican Dong, et al .2023. A survey of large language models.arXiv preprint arXiv:2303.182231, 2 (2023), 1–124.
Manuscript submitted to ACM Trans. Softw. Eng. Methodol.