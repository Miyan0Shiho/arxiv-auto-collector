# Revisiting Vul-RAG: Reproducibility and Replicability of RAG-based Vulnerability Detection with Open-Weight Models

**Authors**: Sabrina Kaniewski, Fabian Schmidt, Tobias Heer

**Published**: 2026-06-03 11:20:37

**PDF URL**: [https://arxiv.org/pdf/2606.04739v1](https://arxiv.org/pdf/2606.04739v1)

## Abstract
Large language models (LLMs) have shown strong potential for automated software vulnerability detection, particularly in retrieval-augmented generation (RAG) settings. However, for approaches relying on proprietary models and APIs, reproducibility and replicability remain largely unexplored, raising the question of whether reported results generalize or depend primarily on specific model choices. In this work, we present a reproducibility study of Vul-RAG, a RAG-based framework for source code vulnerability detection that enhances LLMs with high-level vulnerability knowledge. We first replicate the results in a fully local and open-weights setting using the reported open-weight baseline models. We then extend the evaluation to a diverse set of recent open-weight LLMs, including code-specialized, general-purpose, and reasoning models of varying parameter sizes. The results confirm that the findings of Vul-RAG are reproducible under local deployment, but with minor deviations. Across all evaluated models, we observe a performance plateau at approximately 0.30 pairwise accuracy (code pairs for which both the vulnerable and the patched function are correctly classified). Notably, this plateau persists even for more recent and advanced models, indicating that improvements in model capacity alone do not substantially enhance performance. Finally, we discuss practical implications and trade-offs between detection effectiveness, model capabilities, and model scale. Implementation and evaluation artifacts are publicly available at https://github.com/hs-esslingen-it-security/revisiting-Vul-RAG.

## Full Text


<!-- PDF content starts -->

Revisiting Vul-RAG: Reproducibility and
Replicability of RAG-based Vulnerability
Detection with Open-Weight Models
Sabrina Kaniewski1, Fabian Schmidt2, and Tobias Heer1
1Institute for Secure Networked Systems, Esslingen University, Esslingen, Germany
2Institute for Intelligent Systems, Esslingen University, Esslingen, Germany
Contact:sabrina.kaniewski@hs-esslingen.de
Abstract. Large language models (LLMs) have shown strong potential
for automated software vulnerability detection, particularly in retrieval-
augmented generation (RAG) settings. However, for approaches relying
on proprietary models and APIs, reproducibility and replicability remain
largely unexplored, raising the question of whether reported results gen-
eralize or depend primarily on specific model choices. In this work, we
present a reproducibility study of Vul-RAG, a RAG-based framework for
source code vulnerability detection that enhances LLMs with high-level
vulnerability knowledge. We first replicate the results in a fully local
and open-weights setting using the reported open-weight baseline models.
We then extend the evaluation to a diverse set of recent open-weight
LLMs, including code-specialized, general-purpose, and reasoning mod-
els of varying parameter sizes. The results confirm that the findings of
Vul-RAG are reproducible under local deployment, but with minor devi-
ations. Across all evaluated models, we observe a performance plateau
at approximately 0.30 pairwise accuracy (code pairs for which both the
vulnerable and the patched function are correctly classified). Notably, this
plateau persists even for more recent and advanced models, indicating
that improvements in model capacity alone do not substantially enhance
performance. Finally, we discuss practical implications and trade-offs
between detection effectiveness, model capabilities, and model scale. Im-
plementation and evaluation artifacts are publicly available at https:
//github.com/hs-esslingen-it-security/revisiting-Vul-RAG.
Keywords:Software vulnerability detection·Large language models·
Retrieval-augmented generation·Open science·Reproducibility.
1 Introduction
Software vulnerability detection is a critical task in the software development
life cycle, enabling developers to identify and mitigate security issues before they
can be exploited in deployed systems [8]. Recent advances in Large Language
Models (LLMs) have opened new opportunities for automating vulnerability
detection. Due to their extensive world knowledge and ability to generalize acrossarXiv:2606.04739v1  [cs.SE]  3 Jun 2026

2 S. Kaniewski et al.
programming languages and code representations, LLMs have emerged as a
promising approach for analyzing the security of source code [9].
One particularly promising technique for adapting LLMs to the software
vulnerability detection task is Retrieval-Augmented Generation (RAG) [10].
RAG enhances LLMs by dynamically retrieving relevant information from a
knowledge base and incorporating it into the prompt during inference. By aug-
menting prompts with up-to-date vulnerability knowledge, RAG allows LLMs
to reason about vulnerabilities beyond the information contained in the model’s
training data [9]. Recent studies explore different knowledge representations
such as vulnerability descriptions or labeled code examples to improve detection
performance [2,5,11,16,19–23,25].
Despite advancements in LLM-based software vulnerability detection, several
challenges hinder performance validation. Many studies do not release their code
and artifacts as open-source or rely on proprietary models and APIs such as
OpenAI’s GPT. It remains unclear whether reported performance gains are a
resultoftheproposedmethodologiesortheselectionofhighlycapable,proprietary
models. Reproducibility, i.e., the ability for an independent group of researchers
to obtain the same results using the provided methods and data, is a fundamental
requirement for validating findings and incremental knowledge building [1]. In
LLM-based research, reproducibility is further complicated by rapidly evolving
model ecosystems, undocumented dependency changes, and the non-deterministic
nature of outputs [17]. In addition, the reliance on external APIs raises concerns
for security-critical domains such as vulnerability detection, where sensitive or
proprietary code may not be suited for transmission to third-party services.
Contributions. To address these challenges, we conduct a reproducibility
study of Vul-RAG [5], a recently proposed RAG-based framework for source code
vulnerability detection. The evaluation of Vul-RAG relied on a set of proprietary
and open-weight models. Given the rapid evolution of the model ecosystem, there
is a need to evaluate whether these findings hold for the latest generation of
models and to identify current limitations in cross-model replicability. To enable
reproducible experimentation, we adapt the Vul-RAG framework to operate
entirely with open-weight models and local inference, removing the dependency
on proprietary APIs. First, we reproduce the results using the reported open-
weight baseline models. Second, we systematically evaluate Vul-RAG using recent
open-weight LLMs, including code-specialized, general-purpose, and reasoning
models of various parameter sizes. Through this study, we aim to assess the
reproducibility of the reported results, investigate the sensitivity of Vul-RAG to
the underlying model, and discuss implications for selecting and deploying models
andRAG-basedsystemsindetectionpipelines.Wepublishallimplementationand
evaluation artifacts at /githubhttps://github.com/hs-esslingen-it-security/
revisiting-Vul-RAG.
The remainder of this work is structured as follows. In Section 2, we review
related studies on open science. In Section 3, we provide an overview of the
Vul-RAG framework. Section 4 outlines the experimental setup, including the
research questions, dataset, studied LLMs, metrics, and implementation details.

Revisiting Vul-RAG 3
In Section 5, we discuss the results. We address threats to validity (e.g., potential
evaluation bias) in Section 6 and conclude this work in Section 7.
2 Related Work
Despite the growing body of work on LLM-based software vulnerability de-
tection [9], only a limited number of studies examine the reproducibility or
replicability of existing approaches.
Nong et al. [13] investigated open science practices in deep learning (DL)-
based software vulnerability detection by analyzing 55 studies across four di-
mensions: availability, executability, reproducibility, and replicability. Only 14
studies provided publicly available artifacts. Among these, three lacked sufficient
documentation or complete implementations, rendering them non-executable. To
assess reproducibility, the authors measured the deviation between reported and
reproduced F1 scores. Artifacts were classified as reproducible, weakly repro-
ducible, or not reproducible if the absolute deviation was less than 1%, between
1% and 5%, or greater than 5%, respectively. Only 7 out of 8 executable artifacts
were reproducible, and merely two were fully replicable, i.e., reported performance
could be replicated against different training and testing datasets.
Chakraborty [3] revisited four DL-based vulnerability detection approaches.
When evaluated on the original datasets, the reproduced results largely matched
the reported ones, typically differing by about one percentage point. However,
when replicated on a real-world benchmark, the authors observed a substantial
performance drop, highlighting limitations in their practical applicability.
Similarly, Steenhoek et al. [18] reproduced nine state-of-the-art DL-based
vulnerability detection approaches using two common benchmark datasets. For
most models, reproduced results differed by less than 2% from those reported.
Recent work has also highlighted reproducibility challenges in LLM-based
software engineering research. Angermeir et al. [1] analyzed the reproducibility
of 85 LLM-centric studies published in 2024. Among these, 69 studies relied on
OpenAI services either as base models or for comparison. Of the 18 studies that
provided research artifacts, only five were complete and executable. However,
none could be fully reproduced: two were partially reproducible, while three could
not be reproduced at all. The authors identified several common obstacles to
reproducibility, including non-obvious missing artifacts (e.g., individual missing
scripts), unspecified dependency versions, code issues, and deprecation of LLMs.
Despite these efforts, no existing study has examined the reproducibility of
LLM-based vulnerability detection approaches that employ advanced adaptation
techniques such as RAG. Given the increasing adoption of RAG-based techniques
in LLM-assisted software analysis, understanding their reproducibility is essential
for ensuring reliable and verifiable research outcomes. In this work, we address
this gap by reproducing Vul-RAG [5], one of the most cited (at the time of
writing) and openly available RAG-based frameworks for vulnerability detection.

4 S. Kaniewski et al.
3 Vul-RAG: Knowledge-level RAG Framework
Vul-RAG [5] is a retrieval-augmented framework for source code vulnerability
detection that enhances LLMs with high-level, generalizable vulnerability knowl-
edge. Vul-RAG constructs a multi-dimensional representation of vulnerability
knowledge that focuses on semantic and causal characteristics of vulnerabilities.
The framework consists of three steps as visualized in Figure 1:
1In a preceding offline vulnerability knowledge base construction, Vul-RAG
extracts multi-dimensional knowledge from historical Common Vulnerabili-
ties and Exposures (CVEs) and their corresponding fixes. Specifically, each
knowledge item consists of three dimensions: (i)Functional Semantics, which
summarize the high-level functionality and detailed behavior of the vulner-
able code; (ii)Vulnerability Causes, which describe the root cause of the
vulnerability by comparing vulnerable and patched code; and (iii)Fixing
Solutions, which summarize how the vulnerability is fixed in the patch. Vul-
RAG abstracts method invocations, variable names, and types to create a
representation that is less dependent on code specifics.
2During online detection, Vul-RAG first prompts an LLM to extract the code’s
functional semantics. Vul-RAG then uses the code, its abstract purpose, and
detailed behavior to retrieve the top-n relevant items that have high functional
similarity with the input code. Retrieval is performed using probabilistic
ranking (BM25) followed by a re-ranking.
Vulnerability Knowledge Base ConstructionVulnerability Knowledge RetrievalKnowledge-Augmented Vulnerability DetectionVulnerabilityKnowledge BaseFunctional Semantics ExtractionKnowledge AbstractionVulnerability Causes and Fixing Solutions ExtractionVulnerability Knowledge ExtractionFunctional Semantics ExtractionVulnerability Knowledge RetrievalLLM-based Vulnerability DetectionRetrieval QueryVulnerability Detection PromptTop-N Related Vulnerability KnowledgeCVE DescriptionCode PairsCVE-Corpus
Source Code
Detection Results
123*P1*P2
Fig.1.Overview of Vul-RAG, adapted from [5]. The framework consists of three stages:
1offline vulnerability knowledge base construction; 2online retrieval of relevant
knowledge items based on functional semantics; and 3iterative, knowledge-augmented
vulnerability detection. The markers*P1and*P2highlight points of failure where
model performance frequently failed, as discussed in Sections 4.2 and 5.1.

Revisiting Vul-RAG 5
3Vul-RAG performs iterative, knowledge-augmented vulnerability detection.
Instead of injecting all retrieved knowledge items into a single detection
prompt, the LLM sequentially reasons over each knowledge item via (i) a
prompt for finding vulnerability causes, which checks whether the input code
matches the vulnerability cause, and (ii) a prompt for finding fixing solutions,
which checks whether the code applies the corresponding fixing solution. The
process terminates once the code is identified as vulnerable or all items have
been considered, in which case the code is classified as non-vulnerable.
In this study, we focus on the online retrieval and detection stages ( 2and
3in Figure 1). To ensure controlled comparison and isolate the impact of the
selected models, we keep the vulnerability knowledge base generated and provided
by the authors of Vul-RAG.
4 Experimental Setup
This study aims to assess the reproducibility and replicability of Vul-RAG.
Specifically, we first examine whether the reported results can be reproduced
when using the reported open-weight baseline models in a local deployment
setting. The Vul-RAG study notes that the generalization of its findings to
other LLMs requires further investigation [5]. Motivated by this limitation, we
systematically evaluate how Vul-RAG performs when replacing the reported
baseline models with more recent LLMs and models with different capabilities,
including code-specialized, general-purpose, and reasoning models of various
parameter sizes. We design the experiments to answer the following research
questions (RQs):
RQ1To what extent are the reported Vul-RAG results reproducible?
RQ2How does Vul-RAG perform with newer generation LLMs?
RQ3How does model specialization (code-specialized vs. general-purpose)
influence Vul-RAG performance?
RQ4Do reasoning capabilities improve the performance of Vul-RAG?
RQ5How does model parameter scale affect the performance of Vul-RAG?
4.1 Benchmark Dataset
We reuse the benchmark dataset introduced in the Vul-RAG study. The dataset
consists of pairs of vulnerable and patched functions extracted from real-world
CVEsreportedfortheLinuxkernel.EachpairisenrichedwithCommonWeakness
Enumeration (CWE) identifiers and descriptions. The dataset comprises 2,903
function pairs covering the ten most prevalent CWEs. It is randomly split in a
1:4 ratio into a test and training set. The test set contains 586 pairs and serves
as the evaluation benchmark (PairVul), while the remaining 2,317 pairs form the
training set used to construct the knowledge base.
The most prevalent vulnerability types in the datasets are shown in Table 1.
We categorize the CWEs according to the CWE-1000 Research View [12], which

6 S. Kaniewski et al.
Table 1.Number of vulnerable-patched code pairs per CWE in PairVul and the
knowledge base (KB), grouped by CWE-1000 research view pillars [12]:Improper Access
Control(Access),Improper Control of a Resource Through its Lifetime(Resource),
Improper Check or Handling of Exceptional Conditions(Exceptional), andImproper
Neutralization(Neutralization).
SetCWEAccess Resource Exceptional Neutralization
264 119 125 200 362 401 416 787 476 20
PairVul 31 44 35 39 81 26 166 47 71 46
KB 120 173 140 152 320 101 660 187 281 182
CWE-264: Permissions, Privileges, and Access Controls
CWE-119: Improper Restriction of Operations within the Bounds of a Memory Buffer
CWE-125: Out-of-bounds Read
CWE-200: Exposure of Sensitive Information to an Unauthorized Actor
CWE-362: Concurrent Execution using Shared Resource with Improper Synchronization
CWE-401: Missing Release of Memory after Effective Lifetime
CWE-416: Use After Free
CWE-787: Out-of-bounds Write
CWE-476: Null Pointer Dereference
CWE-20: Improper Input Validation
groups weaknesses into ten high-level research pillars. This categorization enables
analysis of Vul-RAG performance across vulnerability classes rather than indi-
vidual CWEs. The covered CWEs span four of ten pillars, i.e.,Improper Access
Control,Improper Control of a Resource Through its Lifetime,Improper Check
or Handling of Exceptional Conditions, andImproper Neutralization. The high
coverage of resource-related vulnerabilities is consistent with the Linux kernel
setting and its implementation in C, where handling of memory and system
resources is a common source of vulnerabilities.
4.2 Studied LLMs
Baseline Models. The Vul-RAG evaluation considered four state-of-the-art
LLMs that have been widely used for vulnerability detection [5]: two proprietary
models (GPT-4o and Claude Sonnet 3.5) and two open-weight models (DeepSeek-
Coder-V2-Instruct and Qwen2.5-Coder-32B-Instruct). While proprietary models
are subject to version deprecation and opaque updates (rendering long-term
reproducibility difficult), open-weight models remain available, ensuring that
experimental results can be independently verified and compared even as the
model ecosystem evolves.
Additional Models. To investigate the impact of different models, we
evaluate a broader set of open-weight LLMs. During preliminary experiments,
we tested open-weights models of various parameter sizes across the Qwen,
DeepSeek, Llama, and OpenAI model families, cf. online material. However,
some of these models frequently produced outputs that did not consistently
follow the given instruction format, i.e., missing <result>YES</result> or
<result>NO</result> tags in*P2(cf. Figure 1), and, thus, could not be reliably
parsed for evaluation. To ensure a stable evaluation pipeline, we restrict the
final set of models to those that processed all benchmark items successfully, as
summarized in Table 2.

Revisiting Vul-RAG 7
Table 2.Studied LLMs grouped by experiment. Full HuggingFace model identifier,
parameter count, maximum context length, and knowledge cutoff.
Model IdentifierParameter
[B]Context
[Token]Cutoff
Date
RQ1 Baseline models
deepseek-ai/DeepSeek-Coder-V2-Instruct[4] 236 131,072 Nov 2023
Qwen/Qwen2.5-Coder-32B-Instruct[7] 32.5 131,072 Feb 2024
RQ2 Newer generation of baseline models
Qwen/Qwen3-Coder-30B-A3B-Instruct [24] 30.5 262,144 –
RQ3 General-purpose models
Qwen/Qwen2.5-32B-Instruct [14] 32.5 131,072 –
RQ4 Reasoning models
deepseek-ai/DeepSeek-R1-Distill-Qwen-32B [6] 32.5 131,072 Jul 2024
Qwen/QwQ-32B [15] 32.5 131,072 Nov 2024
RQ5 Varying model parameter scale
deepseek-ai/DeepSeek-R1-Distill-Llama-8B [6] 8 131,072 Jul 2024
deepseek-ai/DeepSeek-R1-Distill-Qwen-14B [6] 14 131,072 Jul 2024
Qwen/Qwen2.5-Coder-3B-Instruct [7] 3.09 32,768 Feb 2024
Qwen/Qwen2.5-Coder-7B-Instruct [7] 7.61 131,072 Feb 2024
Qwen/Qwen2.5-Coder-14B-Instruct [7] 14.7 131,072 Feb 2024
Qwen/Qwen3-4B [24] 4 32,768 –
Qwen/Qwen3-8B [24] 8.2 32,768 –
Qwen/Qwen3-14B [24] 14.8 32,768 –
Qwen/Qwen3-30B-A3B-Instruct-2507 [24] 30.5 262,144 –
4.3 Metrics
A challenge in vulnerability detection is that models must reliably distinguish
between vulnerable code and its corresponding patch, which often differ only in
small but security-relevant details [5]. Misclassifying vulnerable code as safe (false
negatives) leaves exploitable weaknesses undetected, while incorrectly flagging
safe code as vulnerable (false positives) introduces unnecessary remediation effort
and may reduce developer trust in automated tools.
Vul-RAG introduces thepairwise accuracymetric, which measures the pro-
portion of code pairs for which both the vulnerable function and its patched
counterpart are correctly classified. In addition, the framework introducesbal-
anced recallandbalanced precision(see Equations (1) and (2), respectively)
to capture the model’s ability to detect vulnerabilities and avoid false alarms.
Balanced recall reflects the reliability of identifying vulnerable code, while bal-
anced precision measures the accuracy of predicted vulnerabilities relative to
actual weaknesses. In the Vul-RAG study, the best performance was achieved
using GPT-4o, with a pairwise accuracy of 0.32, a balanced recall of 0.58, and a
balanced precision of 0.63. To put these results in perspective, a random baseline
would achieve a recall and precision of 0.50, and a pairwise accuracy of 0.25.
These results highlight the complexity of the task even for state-of-the-art models.
Balanced Recall =#True vul
#Total vul+#True nvul
#Total nvul
/2(1)
Balanced Precision =#True vul
#Predict vul+#True nvul
#Predict nvul
/2(2)

8 S. Kaniewski et al.
4.4 Implementation Details
We adapted the Vul-RAG pipeline to use PyTorch and the HuggingFace Trans-
formers library, replacing the original API-based clients with local open-weight
models. The implementation supports configurable model parameters, such as
the maximum number of output tokens ( max_new_tokens=4096 ) and tempera-
ture, which regulates the randomness of the output by scaling the probability
distribution of the next-token prediction; a value close to zero ensures more deter-
ministic and consistent responses. For models without built-in prompt templates,
we implemented a fallback formatting routine to standardize inputs. All other
components, including source code, input data, prompts, and default Vul-RAG
hyperparameter (e.g., temperature=0.01 ,max_knowledge=3 ,retry_times=5 ),
remain unchanged.
All experiments were conducted on a high-performance computing cluster.
Experiments were executed on NVIDIA L40S 48GB (for experiments with models
with up to 16B parameters) or up to eight NVIDIA H100 80GB SXM5 GPUs.
5 Results
In this section, we present and discuss the experimental results with respect to
the research questions RQ1-RQ5 defined in Section 4.
5.1 RQ1 Reproducibility of Vul-RAG
To assess the reproducibility of Vul-RAG, we evaluate the framework using its
reported open-weight baseline models: DeepSeek-Coder-V2-Instruct and Qwen2.5-
Coder-32B-Instruct. Following best practices for LLM reproducibility [17], we
conducted three independent runs and report the mean performance.
Table 3 compares the reproduced results with the results reported in the Vul-
RAG study. For Qwen2.5-Coder-32B-Instruct, the results confirm the reported
performance: pairwise accuracy remains identical, and deviations in balanced
recall and precision are minor (at most two percentage points, which is negligi-
ble considering the reported standard deviation). Examining the CWE-specific
pairwise accuracy in Table 4, we observe higher variance, e.g., inCWE-125, we
see a difference of 13 percentage points. However, these discrepancies are likely
attributable to output variability.
Table 3.Vul-RAG results reported (*) and reproduced (r). We report the mean ±
standard deviation for the Qwen baseline. The DeepSeek baseline processed only 565/586
pairs; thus, the results are not representative and we report only a single run.
LLM Pair. Accuracy Recall Precision
DeepSeek-Coder-V2*0.30 0.61 0.62
r↓0.22↓0.52↓0.53
Qwen2.5-Coder-32B*0.26 0.59 0.61
r0.26±0.01 0.59±0.01↓0.59±0.01

Revisiting Vul-RAG 9
For DeepSeek-Coder-V2-Instruct, we observe a performance drop across all
metrics. DeepSeek-Coder-V2-Instruct in BF16 format requires 8 ×80GB GPUs
for inference according to its HuggingFace model card. Despite executing on
eight NVIDIA H100 GPUs, which provide the required resources, the model
successfully processed only 565 of 586 pairs. The remaining 21 pairs repeatedly
failed due to CUDA out-of-memory (OOM) errors, primarily during the early
retrieval phases, e.g., extraction of function purpose and functionality (cf.*P1
in Figure 1). As visualized in Figure 2, these failures affected all CWE types
and were particularly pronounced for long codes. In these cases, the quadratic
scaling of the self-attention mechanism exhausts available GPU memory. Due to
incomplete processing, Table 3 and Table 4 report only a single run for DeepSeek-
Coder-V2-Instruct. The Vul-RAG framework excludes failed items from the
evaluation. While excluding these longer and likely more complex items reflects
Table 4.Mean CWE-specific pairwise accuracy for the Vul-RAG results reported (*)
and reproduced (r). CWEs grouped by CWE-1000 research view pillars, cf. Table 1.
Model names abbreviated.
LLMCWE Access Resource Excep. Neut.
264 119 125 200 362 401 416 787 476 20
DeepSeek-Coder-V2* 0.35 0.30 0.26 0.38 0.25 0.54 0.29 0.32 0.27 0.26
r↓0.23 0.30↓0.21↓0.14↓0.18↓0.29↓0.21↓0.16↓0.25↑0.27
Qwen2.5-Coder-32B* 0.42 0.23 0.43 0.28 0.22 0.27 0.24 0.21 0.24 0.26
r↑0.49↓0.22↓0.30↓0.26↑0.25↓0.26↑0.26↑0.28↓0.21↓0.20
CWE-264 CWE-119 CWE-125 CWE-200 CWE-362 CWE-401 CWE-416 CWE-787 CWE-476CWE-20
Dataset05000100001500020000Tokens
Token Distribution
Failed Item
OVERALL
Fig.2.Distribution of tokens for the detection prompts across the PairVul CWE subsets,
calculated using the DeepSeek-Coder-V2-Instruct tokenizer. Red crosses indicate dataset
items where the model failed to generate a detection output. The right side shows the
distribution of input code lengths of PairVul in tokens.

10 S. Kaniewski et al.
the model’s capability on the subset it could successfully process, the results
on the remaining 565 pairs still show a substantial deviation from the reported
performance, with a drop of eight percentage points (0.08) in pairwise accuracy
and a drop of 9 percentage points (0.09) in both recall and precision.
RQ1To what extent are the reported Vul-RAG results reproducible?
We were able to reproduce the results for the Qwen2.5-Coder-32B-Instruct
baseline, with only minor deviations in precision. DeepSeek-Coder-V2-
Instruct exhibits a noticeable performance drop and fails to process all
input pairs due to GPU memory limitations. Importantly, failed items are
excluded from the evaluation; observed discrepancies persist even on the
successfully processed subset.
5.2 RQ2 Performance with Newer Generation Models
To assess whether improvements in models translate to enhanced vulnerability
detection performance, we evaluate Vul-RAG using the newer generation of
the Qwen baseline model, namely Qwen3-Coder-30B-A3B-Instruct. While other
newer open-weight models such as DeepSeek-V2.5 and Qwen3-Coder-Next exist,
their high parameter counts (236B and 80B, respectively) make them impractical
for on-premise deployment; thus, we focus on the 30B-parameter model.
The results are shown in Table 5 row RQ2. Qwen3-Coder-30B-A3B-Instruct
achieves a modest increase of two percentage points in pairwise accuracy (0.28 vs.
Table 5.Vul-RAG results reported (*) and reproduced (RQ1-RQ5) using open-weight
models. We highlight the first,second, andthirdbest reproduced results across
models per metric; incomplete item coverage is marked in red. Model names abbreviated.
RQ LLM Pair. Accuracy Recall Precision Pairs
*DeepSeek-Coder-V2 0.30 0.61 0.62
Qwen2.5-Coder-32B 0.26 0.59 0.61
RQ1DeepSeek-Coder-V2 0.22 0.52 0.53 565/586
Qwen2.5-Coder-32B 0.26 0.59 0.59 586/586
RQ2 Qwen3-Coder-30B 0.28 0.59 0.59 586/586
RQ3 Qwen2.5-32B 0.25 0.58 0.59 586/586
RQ4DeepSeek-R1-32B 0.290.57 0.58 586/586
QwQ-32B 0.29 0.60 0.61586/586
RQ5DeepSeek-R1-8B 0.290.57 0.58 586/586
DeepSeek-R1-14B 0.27 0.56 0.57 586/586
Qwen2.5-Coder-3B 0.24 0.54 0.54 586/586
Qwen2.5-Coder-7B 0.08 0.51 0.52 586/586
Qwen2.5-Coder-14B 0.24 0.57 0.57 586/586
Qwen3-4B 0.25 0.57 0.58 586/586
Qwen3-8B 0.24 0.57 0.58 586/586
Qwen3-14B 0.22 0.58 0.61586/586
Qwen3-30B 0.25 0.58 0.60 586/586

Revisiting Vul-RAG 11
0.26), while maintaining balanced recall and precision. However, this enhancement
remains limited, suggesting that while incremental model updates might provide
slight benefits in performance, they do not necessarily overcome the challenges
inherent in the detection task.
RQ2How does Vul-RAG perform with newer generation LLMs?
Incremental model upgrades yield only modest performance gains, with
Qwen3-Coder-30B providing an increase of only two percentage points in
pairwise accuracy over the Qwen2.5-Coder-32B baseline.
5.3 RQ3 Code-Specialized vs. General-Purpose LLMs
Toinvestigatewhetherstrongnaturallanguageunderstandingbenefitsabstraction-
based vulnerability detection, we evaluate Vul-RAG using the general-purpose
counterpart of the Qwen2.5-Coder baseline, namely Qwen2.5-32B-Instruct.
The results are shown in Table 5 row RQ3. We observe only a marginal perfor-
mance decrease of one percentage point in pairwise accuracy and recall compared
to the code-specialized baseline. By abstracting code and vulnerabilities into
high-level semantic descriptions, Vul-RAG reframes a specialized programming
task as a logical reasoning task. The small performance difference suggests that
this abstraction can compensate for the absence of explicit code specialization and
that, for RAG-based systems relying on high-level knowledge, natural language
understanding capabilities may be as critical as domain-specific exposure.
RQ3How does model specialization influence Vul-RAG performance?
The general-purpose model Qwen2.5-32B shows only a minimal performance
drop of one percentage point compared to the code-specialized baseline. The
results suggest that Vul-RAG’s abstraction allows general-purpose models to
compensate for a lack of code-specific training.
5.4 RQ4 Impact of Reasoning Capabilities
To evaluate whether explicit reasoning capabilities improve prediction, we analyze
reasoning LLMs from both baseline model families, i.e., QwQ-32B and a distilled
variant of the DeepSeek-R1 model (DeepSeek-R1-Distill-Qwen-32B), transferring
reasoning patterns into smaller models.
The results are shown in Table 5 row RQ4. The considered reasoning models
consistently outperform the baseline (RQ1), achieving the highest pairwise ac-
curacy of 0.29, and nearly matching the reported DeepSeek-Coder-V2 baseline
of 0.30. However, this performance gain entails a trade-off in computational
overhead. As shown in Figure 3, reasoning models generate substantially more
tokens than the non-reasoning baseline. At equal parameter scale, QwQ-32B

12 S. Kaniewski et al.
increases the median output length from 540 to 1385 tokens (+2.6 ×) and the
IQR from 166 to 1035 tokens (+6.2 ×) compared to Qwen2.5-Coder-32B-Instruct.
The heavy-tailed output distribution indicates that reasoning models do not
merely generate longer outputs but adopt more diverse generation behaviors.
Comparing different model families, QwQ shows extensive reasoning traces while
DeepSeek-R1 produces more compact outputs.
qwen2.5-coder-32b qwq-32b deepseek-r1-8b deepseek-r1-14b deepseek-r1-32b
Model50010001500200025003000350040004500Tokens
Fig.3.Comparison of output lengths in tokens between baseline and reasoning models
formax_new_tokens=4096 . Reasoning models exhibit higher token counts and longer
tails, reflecting the generation of reasoning steps.
1 2 30255075100
82%
36%100%
76%
40%100%CWE-264
1 2 370%
50%100%
86%
42%100%CWE-119
1 2 370%67%100%
79%
53%100%CWE-125
1 2 354%67%100%
71% 70%100%CWE-200
1 2 30255075100
80%
70%100%
83%
46%100%CWE-362
1 2 381%
40%100%
90%
40%100%CWE-401
1 2 375%
56%100%
83%
60%100%CWE-416
1 2 368%
53%100%
81%
56%100%CWE-787
1 2 30255075100
79%
57%100%
85%
55%100%CWE-476
1 2 372%
35%100%
74%
46%100%CWE-20
1 2 374%
56%100%
81%
53%100%OVERALL
IterationsConditional Success Rate (%)
Model
qwen2.5-coder-32b
qwq-32b
Fig.4.Conditional success rates for reasoning vs. baseline model. Values represent the
proportion of samples classified at a specific iteration, provided no decision was reached
earlier. Vul-RAG iterates through up to three items before enforcing a final decision.

Revisiting Vul-RAG 13
Further analysis of the iterative retrieval process in Figure 4 reveals that
reasoning models exhibit different conditional success rates (i.e., the proportion
of codes that are classified at a specific iteration step). The slight advantage in
the first iteration suggests that they are more effective when limited contextual
information is available. However, if no decision is reached in the first iteration,
subsequent iterations provide only limited additional benefit, suggesting that the
advantage of reasoning primarily lies in early-stage decision-making.
RQ4Do reasoning capabilities improve the performance of Vul-RAG?
Reasoning models achieve the highest pairwise accuracy but incur higher
inference costs due to increased token generation. Their advantage in iterative
detection is most pronounced in the first iteration, where they more effectively
translate retrieved knowledge into a correct prediction.
5.5 RQ5 Impact of Model Parameter Scale
To assess the impact of model parameter scale on performance, we evaluate
Vul-RAG across models from RQ1-RQ4 with parameters ranging from 3B to 32B.
The results are shown in Table 5 row RQ5. Overall, we do not observe a
consistent relationship between model scale and detection performance. Several
smaller models achieve a performance comparable to or exceeding that of larger
models. For example, for the Qwen2.5-Coder baseline model (RQ1), performance
does not drop significantly when using smaller variants, i.e., with 3B and 14B
parameters.Whilethe7Bvariantstruggledwithcorrectlyidentifyingpairs(biased
towards predicting non-vulnerable), the 3B and 14B variants both achieve a
pairwise accuracy of 0.24, only two percentage points below the baseline. Similarly,
within the Qwen3 model family (RQ2, RQ3), performance remains largely stable
across scales: The Qwen3-4B model achieves a pairwise accuracy of 0.25, matching
both Qwen3-30B and Qwen2.5-32B, while Qwen3-14B performs worse with a
pairwise accuracy of 0.22. Further, the DeepSeek-R1-Distill-Llama-8B reasoning
model (RQ4) reaches a pairwise accuracy of 0.29, matching the best-performing
32B reasoning models, and even outperforms much larger non-reasoning models.
These results indicate that architectural and training refinements in newer model
generations may have a greater impact than mere parameter count, and that
reasoning capabilities can outweigh the benefits of increased parameter scale.
RQ5How does model parameter scale affect the performance of Vul-RAG?
Increasing model scale does not consistently improve detection performance.
Smaller models (e.g., 4B–8B) achieve results comparable to larger 30B–32B
models, and in some cases even outperform them. The findings suggest that
model capabilities, such as reasoning, are more critical than parameter scale.

14 S. Kaniewski et al.
5.6 Discussion
The results of this work provide several insights into the current state of RAG-
based vulnerability detection and the specifics of the Vul-RAG framework re-
garding (i) detection effectiveness, (ii) knowledge representation and quality, and
(iii) practical deployment.
Detection effectiveness.A primary finding across all research questions
RQ1-RQ5 is the narrow performance difference despite the diversity of the
models evaluated, cf. Table 5. Notably, the highest achieved overall pairwise
accuracy of 0.29 is only four percentage points above the random baseline of
0.25. Hence, while the Vul-RAG framework enhances the ability of LLMs to
distinguish between vulnerable and non-vulnerable code [5], these capabilities do
not scale with model advancement; for instance, Qwen3-4B achieves the same
pairwise accuracy (0.25) as Qwen3-30B-A3B-Instruct-2507. The observation of a
performance plateau is also evident in the study by Antal et al. [2], where we
observe a similar ceiling in the context of Java vulnerabilities. Despite evaluating
(proprietary) frontier models with large parameter scales, e.g., DeepSeek-V3.1,
GPT-5, and Llama-4-Maverick, the authors reported pairwise accuracies of 16.7%,
23.1%, and 32.4% respectively. The persistence of this plateau across different
programming languages, datasets, knowledge bases, and models reinforces the
inherent difficulty of the pairwise discrimination task.
Table 6.CWE-specific pairwise accuracy across evaluated models. We highlight the
first,second,thirdbest and worstresult per model. CWEs grouped byCWE-
1000research view pillars, cf. Table 1. Model names abbreviated.
LLMCWEAccess Resource Excep. Neut.
RQ264 119 125 200 362 401 416 787 476 20
RQ1DeepSeek-Coder-V2 0.23 0.300.21 0.14 0.18 0.29 0.21 0.16 0.25 0.27
Qwen2.5-Coder-32B 0.490.22 0.30 0.26 0.25 0.26 0.26 0.28 0.21 0.20
RQ2 Qwen3-Coder-30B 0.390.25 0.370.23 0.28 0.23 0.25 0.34 0.31 0.28
RQ3 Qwen2.5-32B 0.450.20 0.23 0.28 0.25 0.27 0.24 0.23 0.24 0.20
RQ4DeepSeek-R1-32B 0.550.32 0.31 0.330.33 0.23 0.27 0.32 0.23 0.13
QwQ-32B 0.480.27 0.37 0.28 0.25 0.46 0.32 0.17 0.20 0.22
RQ5DeepSeek-R1-8B 0.29 0.30 0.26 0.33 0.25 0.420.28 0.30 0.320.24
DeepSeek-R1-14B 0.23 0.23 0.34 0.26 0.360.27 0.27 0.28 0.27 0.20
Qwen2.5-Coder-3B 0.26 0.20 0.140.360.22 0.35 0.20 0.21 0.27 0.33
Qwen2.5-Coder-7B 0.06 0.07 0.110.100.060.150.08 0.060.10 0.07
Qwen2.5-Coder-14B 0.23 0.270.340.21 0.23 0.23 0.24 0.21 0.270.17
Qwen3-4B 0.29 0.25 0.340.23 0.26 0.27 0.24 0.30 0.20 0.17
Qwen3-8B 0.320.30 0.26 0.23 0.220.31 0.23 0.23 0.23 0.22
Qwen3-14B 0.350.20 0.29 0.18 0.16 0.350.22 0.26 0.20 0.13
Qwen3-30B 0.550.25 0.23 0.23 0.20 0.35 0.23 0.190.25 0.20
CWE-264: Permissions, Privileges, and Access Controls
CWE-119: Improper Restriction of Operations within the Bounds of a Memory Buffer
CWE-125: Out-of-bounds Read
CWE-200: Exposure of Sensitive Information to an Unauthorized Actor
CWE-362: Concurrent Execution using Shared Resource with Improper Synchronization
CWE-401: Missing Release of Memory after Effective Lifetime
CWE-416: Use After Free
CWE-787: Out-of-bounds Write
CWE-476: Null Pointer Dereference
CWE-20: Improper Input Validation

Revisiting Vul-RAG 15
Performance also varies considerably across vulnerability types, as shown
in Table 6, indicating that certain vulnerabilities are inherently more difficult
to capture, even with abstract knowledge representations. Across all models,
performance is consistently highest forCWE-264, where access control vulnera-
bilities often follow well-defined patterns that can be captured through semantic
descriptions. In contrast, performance is lowest forCWE-20. Input validation
logic is typically distributed across multiple code locations and depends on
broader program context, making it difficult to represent through localized func-
tional semantics. For resource-related CWEs, performance remains inconsistent
across models. These vulnerabilities often involve complex interactions or implicit
assumptions about program state, which are not easily captured by static knowl-
edge snippets. Notably, reasoning models (RQ4) exhibit more stable performance
across CWE categories, suggesting that explicit reasoning helps mitigate some of
the limitations of incomplete or imperfect retrieval.
Knowledge representation and quality.For reproducibility, we reused
the Vul-RAG knowledge base, which was generated using GPT-3.5-turbo-0125.
While necessary to ensure a consistent baseline, this choice raises questions about
whether system performance is effectively capped by the quality of the initial
knowledge extraction. The analysis in Vul-RAG identified that false negatives
often stem from non-existent or vague vulnerability descriptions within the knowl-
edge base or from irrelevant retrieval, where the retrieved functional semantics
are too different from the input function. Similarly, false positives are frequently
triggered by mismatched fixing solutions, particularly when a vulnerability could
be addressed by alternative patches not represented in the retrieved context [5].
It remains unclear whether results would improve if the knowledge base were
generated by, e.g., a reasoning model. Future research should, therefore, investi-
gate where model capability matters most (whether in the construction of the
knowledge base, the summarization of code semantics, or the final vulnerability
prediction) and how to evaluate such capability regarding quality.
Practicaldeployment.Weconductedallexperimentsonahigh-performance
computing cluster. Such extensive computational resources are typically unavail-
able in industrial development environments, where hardware is often more
constrained. This discrepancy highlights the importance of the findings regarding
model scale: The small performance differences between models suggest that
selecting significantly larger models does not necessarily yield proportional gains.
Notably, several smaller models achieved comparable performance. These smaller
models are more attractive for on-premise deployment, as they can operate on
commodity hardware while maintaining privacy.
Overall, several challenges remain regarding the real-world practicality of
RAG-based systems, specifically Vul-RAG. The pairwise accuracy ceiling of ap-
proximately 0.30 (and the resulting false positives and false negatives) suggests
that Vul-RAG currently serves more as a research framework than a produc-
tion tool. In addition, its current scope is limited to the Linux kernel and ten
CWE types. The non-deterministic nature of LLMs and the strict formatting
requirements also pose a challenge for automated integration. Further, the use

16 S. Kaniewski et al.
of a binary output without providing verifiable reasoning makes it difficult for
developers to interpret or trust the results. Finally, further benchmarking is
required to determine whether these findings generalize to other programming
languages and vulnerability types.
6 Threats to Validity
Several factors may impact the validity of the findings.
Data leakage.A primary threat in LLM-based research is data leakage,
i.e., the possibility that a model has seen the evaluation samples during its
training and, thus, memorized correct answers rather than performing genuine
vulnerability detection. The PairVul dataset was initially published on GitHub
in April 2024. Given that several models used in this study were released or
updated after this date (specifically QwQ-32B, variants of Qwen-2.5 and Qwen3,
cf. Table 2), we cannot rule out the possibility that the evaluation samples were
included in their training data. The lack of transparency regarding the training
data of LLMs remains an open reseacrh challenge.
Bias in model selection.Vul-RAG relies on <result>YES</result> or
<result>NO</result> tags in the model response for automated parsing of the
prediction. During preliminary experiments, we excluded several models from
discussion in the RQs, as they failed to consistently adhere to this format. A
model’s failure to follow specific output formats does not necessarily imply an
inability to detect vulnerabilities. However, with the setup used by Vul-RAG,
such models are penalized, introducing a selection bias towards models with
superior instruction-following capabilities. Loosening the evaluation script to
handle free-form text might yield different results for less-instructed models but
would introduce non-deterministic parsing errors.
Computational resources and deployment.Hardware constraints af-
fected the evaluation of DeepSeek-Coder-V2, for which we encountered CUDA
out-of-memory errors on an 8 ×80GB H100 node. Consequently, the performance
reported for this model is not fully representative. More importantly, we high-
light a significant gap between research and practice: the infrastructure required
to run such large models exceeds the resources available to typical mid-sized
organizations for software vulnerability detection, underscoring the necessity
for evaluating smaller, more resource-efficient models that can be realistically
deployed in industry pipelines.
7 Conclusion
This work presents a reproducibility and replicability study of the Vul-RAG
framework in a fully local, open-weights setting. We find that Vul-RAG is re-
producible for Qwen2.5-Coder but not for DeepSeek-Coder-V2. Across a diverse
set of LLMs, including newer model generations (RQ2), general-purpose mod-
els (RQ3), and reasoning models (RQ4) with varying parameter scales (RQ5), we

Revisiting Vul-RAG 17
observe only small performance differences. Reasoning models achieve the highest
pairwise accuracy of 0.29. In contrast, neither increased model scale nor newer
model generations yield substantial improvements. These results consistently
indicate a performance plateau in Vul-RAG and highlight the inherent difficulty
of the pairwise discrimination task. From a practical perspective, the findings
show that large parameter scales are not required for competitive performance,
allowing for efficient deployment with smaller models. Future work should focus
on improving the knowledge base quality and retrieval strategies and investigate
where model capability matters most in RAG-based pipelines to overcome the
existing performance limitations.
Acknowledgments.This work has been funded by the Federal Ministry of Research,
Technology and Space (BMFTR) and the state of Baden-Württemberg (Program:
Forschung an HAW, Grant No. 13HAW26PX4), and in part by the German Research
Foundation (DFG) under project-ID 528745080 - FIP 68. The authors alone are respon-
sible for the content of this paper. We thank the DACHS data analysis cluster, hosted
at Hochschule Esslingen and co-funded by the MWK within the DFG’s "Großgeräte der
Länder" program, for providing the computational resources necessary for this research.
References
1.Angermeir, F., Amougou, M., Kreitz, M., Bauer, A., Linhuber, M., Fucci, D., et al.:
Reflections on the Reproducibility of Commercial LLM Performance in Empirical
Software Engineering Studies. arXiv:2510.25506 (2025)
2.Antal, G., Értekes, L., Szolnoki, N., Hegedűs, P.: Evaluating Retrieval-Augmented
Generation for LLM-Based Vulnerability Detection: An Empirical Study on Real-
World Java Vulnerabilities. IEEE Access14, 50878–50891 (2026). https://doi.
org/10.1109/ACCESS.2026.3676577
3.Chakraborty, P., Arumugam, K.K., Alfadel, M., Nagappan, M., McIntosh, S.:
Revisitingtheperformanceofdeeplearning-basedvulnerabilitydetectiononrealistic
datasets. IEEE Transactions on Software Engineering50(8), 2163–2177 (2024).
https://doi.org/10.1109/TSE.2024.3423712
4.DeepSeek-AI, Zhu, Q., Guo, D., Shao, Z., Yang, D., Wang, P., et al.: DeepSeek-
Coder-V2: Breaking the Barrier of Closed-Source Models in Code Intelligence.
arXiv:2406.11931 (2024)
5.Du, X., Zheng, G., Wang, K., Zou, Y., Wang, Y., Deng, W., et al.: Vul-RAG:
Enhancing LLM-based Vulnerability Detection via Knowledge-level RAG. ACM
Transactions on Software Engineering and Methodology (TOSEM) (2026). https:
//doi.org/10.1145/3797277, Just Accepted
6.Guo, D., Yang, D., Zhang, H., Song, J., Wang, P., Zhu, Q., et al.: DeepSeek-R1
incentivizes reasoning in LLMs through reinforcement learning. Nature645(8081),
633–638 (2025)
7.Hui, B., Yang, J., Cui, Z., Yang, J., Liu, D., Zhang, L., et al.: Qwen2.5-Coder
Technical Report. arXiv:2409.12186 (2024)
8.Kaniewski, S., Holstein, D., Schmidt, F., Heer, T.: Vulnerability Handling of AI-
Generated Code - Existing Solutions and Open Challenges. In: Conference on AI,
Science, Engineering, and Technology (AIxSET). pp. 145–148. Laguna Hills, CA,
USA (2024).https://doi.org/10.1109/AIxSET62544.2024.00026

18 S. Kaniewski et al.
9.Kaniewski, S., Schmidt, F., Enzweiler, M., Menth, M., Heer, T.: A Systematic
Literature Review on Detecting Software Vulnerabilities with Large Language
Models. ACM Transactions on Software Engineering and Methodology (TOSEM)
(2026).https://doi.org/10.1145/3815425, Just Accepted
10.Lewis, P., Perez, E., Piktus, A., Petroni, F., Karpukhin, V., Goyal, N., et al.:
Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks. In: Advances
in Neural Information Processing Systems (NeurIPS). vol. 33, pp. 9459–9474 (2020)
11.Lu, G., Ju, X., Chen, X., Pei, W., Cai, Z.: GRACE: Empowering LLM-based
Software Vulnerability Detection with Graph Structure and In-Context Learning.
Journal of Systems and Software (JSS)212, 112031 (2024)
12.MITRE: CWE-1000: Research Concepts. https://cwe.mitre.org/data/
definitions/1000.html(2026), accessed 2026-03-06
13.Nong, Y., Sharma, R., Hamou-Lhadj, A., Luo, X., Cai, H.: Open Science in Software
Engineering: A Study on Deep Learning-Based Vulnerability Detection. IEEE
Transactions on Software Engineering49(4), 1983–2005 (2023). https://doi.org/
10.1109/TSE.2022.3207149
14.Qwen: Qwen2.5-LLM: Extending the boundary of LLMs. https://qwen.ai/blog?
id=qwen2.5-llm, accessed 2026-03-04
15.Qwen: QwQ-32B: Embracing the Power of Reinforcement Learning. https://qwen.
ai/blog?id=qwq-32b, accessed 2026-03-04
16.Safdar, R., Mateen, D., Ali, S.T., Hussain, W.: Real-VulLLM: An LLM Based
Assessment Framework in the Wild. arXiv:2510.04056 (2025)
17.Sallou, J., Durieux, T., Panichella, A.: Breaking the Silence: the Threats of Using
LLMsinSoftwareEngineering.In:ACM/IEEEInternationalConferenceonSoftware
Engineering: New Ideas and Emerging Results (ICSE-NIER). p. 102–106. Lisbon,
Portugal (2024).https://doi.org/10.1145/3639476.3639764
18.Steenhoek, B., Rahman, M.M., Jiles, R., Le, W.: An Empirical Study of Deep
Learning Models for Vulnerability Detection. In: IEEE/ACM International Confer-
ence on Software Engineering (ICSE). pp. 2237–2248. Melbourne, Australia (2023).
https://doi.org/10.1109/ICSE48619.2023.00188
19.Sun, Y., Wu, D., Xue, Y., Liu, H., Ma, W., Zhang, L., et al.: LLM4Vuln: A
Unified Evaluation Framework for Decoupling and Enhancing LLMs’ Vulnerability
Reasoning. arXiv:2401.16185 (2025)
20.Tsai, C.N., Wang, X., Lee, C.H., Lin, C.S.: A Sequential Multi-Stage Approach for
Code Vulnerability Detection via Confidence- and Collaboration-based Decision
Making. In: Conference on Empirical Methods in Natural Language Processing
(EMNLP). pp. 21162–21168. Suzhou, China (2025). https://doi.org/10.18653/
v1/2025.emnlp-main.1071
21.Tsai, C.N., Xie, J., Lai, C.M., Lin, C.S.: Leveraging Intra-and Inter-References in
Vulnerability Detection using Multi-Agent Collaboration Based on LLMs. Cluster
Computing28(15), 1–12 (2025). https://doi.org/10.1007/s10586-025-05721-2
22.Wen, X.C., Wang, X., Chen, Y., Hu, R., Lo, D., Gao, C.: VulEval: Towards
Repository-Level Evaluation of Software Vulnerability Detection. arXiv:2404.15596
(2024)
23.Wu, Z., Xu, J., Peng, Y., Chong, C.Y., Jia, X.: MulVul: Retrieval-augmented
Multi-Agent Code Vulnerability Detection via Cross-Model Prompt Evolution.
arXiv:2601.18847 (2026)
24.Yang, A., Li, A., Yang, B., Zhang, B., Hui, B., Zheng, B., et al.: Qwen3 Technical
Report. arXiv:2505.09388 (2025)
25.Zhu, H., Li, J., Gao, C., Qian, J., Dong, Y., Liu, H., et al.: Specification-Guided
Vulnerability Detection with Large Language Models. arXiv:2511.04014 (2025)