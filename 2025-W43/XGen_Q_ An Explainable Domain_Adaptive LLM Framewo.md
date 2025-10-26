# XGen-Q: An Explainable Domain-Adaptive LLM Framework with Retrieval-Augmented Generation for Software Security

**Authors**: Hamed Jelodar, Mohammad Meymani, Roozbeh Razavi-Far, Ali A. Ghorbani

**Published**: 2025-10-21 18:35:38

**PDF URL**: [http://arxiv.org/pdf/2510.19006v1](http://arxiv.org/pdf/2510.19006v1)

## Abstract
Generative AI and large language models (LLMs) have shown strong capabilities
in code understanding, but their use in cybersecurity, particularly for malware
detection and analysis, remains limited. Existing detection systems often fail
to generalize to obfuscated or previously unseen threats, underscoring the need
for more adaptable and explainable models. To address this challenge, we
introduce XGen-Q, a domain-adapted LLM built on the Qwen-Coder architecture and
pretrained on a large-scale corpus of over one million malware samples,
spanning both source and assembly code. XGen-Q uses a multi-stage prompt
strategy combined with retrieval-augmented generation (RAG) to deliver reliable
malware identification and detailed forensic reporting, even in the presence of
complex code obfuscation. To further enhance generalization, we design a
training pipeline that systematically exposes the model to diverse obfuscation
patterns. Experimental results show that XGen-Q achieves significantly lower
perplexity than competitive baselines and exhibits strong performance on novel
malware samples, demonstrating the promise of LLM-based approaches for
interpretable and robust malware analysis.

## Full Text


<!-- PDF content starts -->

XGen-Q: An Explainable Domain-Adaptive LLM
Framework with Retrieval-Augmented Generation
for Software Security
Hamed Jelodar, Mohammad Meymani, Roozbeh Razavi-Far, Ali Ghorbani
Canadian Institute for Cybersecurity
Faculty of Computer Science
University of New Brunswick
Fredericton, Canada
{h.jelodar, mohammad.meymani79, roozbeh.razavi-far, ghorbani}@unb.ca
Abstract—Generative AI and large language models (LLMs)
have shown strong capabilities in code understanding, but
their use in cybersecurity, particularly for malware detection
and analysis, remains limited. Existing detection systems often
fail to generalize to obfuscated or previously unseen threats,
underscoring the need for more adaptable and explainable
models. To address this challenge, we introduce XGen-Q, a
domain-adapted LLM built on the Qwen-Coder architecture and
pretrained on a large-scale corpus of over one million malware
samples, spanning both source and assembly code. XGen-Q uses a
multi-stage prompt strategy combined with retrieval-augmented
generation (RAG) to deliver reliable malware identification and
detailed forensic reporting, even in the presence of complex
code obfuscation. To further enhance generalization, we design
a training pipeline that systematically exposes the model to
diverse obfuscation patterns. Experimental results show that
XGen-Q achieves significantly lower perplexity than competitive
baselines and exhibits strong performance on novel malware
samples, demonstrating the promise of LLM-based approaches
for interpretable and robust malware analysis.
INTRODUCTION
As software systems become more complex and
interconnected, the attack surface for security vulnerabilities
continues to grow. Modern threat actors increasingly exploit
these vulnerabilities using sophisticated malware capable
of evading traditional detection mechanisms [1, 2]. While
static and dynamic code analysis techniques remain essential,
they often struggle to detect obfuscated, polymorphic,
or previously unseen threats at scale. This has led to a
growing interest in AI-driven solutions that offer adaptability,
contextual reasoning, and semantic understanding of software
behavior [3–5]. Recent advances in LLMs have demonstrated
exceptional performance in code understanding, generation,
and reasoning tasks [6–8]. Also, there are some work related
to malware code analysis [9–11]. For instance, in [12], the
authors focused on overcoming the challenges of applying
LLMs to Android malware detection, such as large codebases
and complex app structures. In other work [13], the authors
focused on improving malware detection by using fine-grained
code features and expanding the dataset with LLM-translated
Fig. 1: This figure illustrates a simple interaction between an
input code and our proposed framework, where the system has
labeled the input as malware.
malicious functions from other languages.
However, their direct application to cybersecurity remains
limited, especially in areas requiring behavioral pattern
recognition, threat explanation, and integration with
operational workflows. These limitations originate from
challenges in dataset availability, model interpretability, and
the lack of end-to-end frameworks that map LLM outputs
with real-time cybersecurity workflows [6].
Although general-purpose LLMs have shown promises
on malware detection, malware classification, and malware
analysis, they are trained predominantly on large-scale but
generic programming corpora, often lack the specializedarXiv:2510.19006v1  [cs.IR]  21 Oct 2025

knowledge needed to assess malicious intent or accurately
interpret attacker’s strategies [8, 14, 15].
In this work, we propose XGen-Q (Explainable Generation-
Driven Qwen Model for Malware Behavior Analysis), a
domain-adaptive, retrieval-augmented generative AI frame-
work designed to strengthen prevention, mitigation, and preser-
vation strategies in software security. The primary contribu-
tions and innovations of this work are as follows:
•We introduce XGen-Q, a systematic large language model
framework for malware behavior analysis, trained on a
diverse corpus of real-world malware samples in both as-
sembly and source code. This domain-specific pretraining
allows the model to capture low-level behavioral patterns
that general-purpose LLMs often fail to recognize.
•We design a retrieval-augmented generation (RAG)
mechanism to dynamically incorporate external cyber-
security knowledge during inference. This improves the
model’s contextual awareness and adaptability to emerg-
ing threats through behavior keyword extraction and
prompt augmentation.
•We develop a two-stage prompt architecture that separates
structured forensic reporting from final behavior classifi-
cation. This promotes interpretability, transparency, and
operational flexibility by providing both human-readable
reports and system-ready verdicts (malware, benign, par-
tially malicious).
1The rest of the paper is organized as follows: In Section
related-works, we review the existing literature on LLMs for
code analysis and malware detection. In Section methodology,
the details of XGen-Q framework, including data prepara-
tion, domain-specific pre-training, model architecture, and
multi-stage prompt design are explained. In Section experi-
ments and implementation, we present our experimental setup,
evaluation metrics, and implementation details and discusses
XGen-Q’s performance on malware behavior classification
and forensic analysis tasks. In Section limitations and future
works, we examine the current limitations of our approach and
outlines directions for future improvement. Finally, in Section
conclusion, we conclude the paper and highlights promising
avenues for further research.
RELATEDWORKS
The intersection of LLMs and software security analysis has
recently attracted significant attention. Several works explore
leveraging pre-trained LLMs for detecting malicious code,
understanding malware behavior, and assisting in reverse en-
gineering tasks.
LLMs for Malware Detection and Analysis:Recent
studies [9, 11, 12, 16] have demonstrated the effectiveness of
adapting general-purpose LLMs to malware detection by pre-
training on domain-specific datasets. These approaches typi-
cally focus on improving detection accuracy and interpretabil-
ity by incorporating behavioral patterns extracted from static
1The code and LLM framework are open source and available here:
https://huggingface.co/JeloH/xGenq-qwen2.5-coder-1.5b-instruct-OKIand dynamic analysis. However, there existing challenges in
handling obfuscated and polymorphic malware, which demand
more nuanced contextual understanding.
Domain-Adaptive Pretraining and Retrieval-Augmented
Generation:To enhance LLM performance in specialized
domains, such as cybersecurity, domain-adaptive pretraining
has been proposed and is used by several works [17–20].
This involves continued training on security-specific corpora,
enabling better modeling of code semantics and malicious
patterns. RAG techniques have further improved inference by
integrating external knowledge dynamically [21]. These strate-
gies help models stay up-to-date with emerging threats and
mitigate knowledge cutoff issues inherent to static pretraining.
Perplexity as a Performance Metric in Code Modeling:
Perplexity has been widely used as an intrinsic metric
to evaluate the predictive capability of language models
on code and natural language tasks. Lower perplexity
generally correlates with better token prediction and semantic
understanding [22]. In the context of malware and code
analysis, perplexity helps quantify how well models grasp
complex, obfuscated code patterns, making it a valuable
benchmark for comparing competing models. Works such as
[23–26] explored this area. To the best of our knowledge,
this is the first work to unify domain-specific pretraining,
retrieval-augmented generation, and multi-stage prompt
engineering for malware analysis with LLMs. In addition
to achieving state-of-the-art performance on perplexity
benchmarks, our framework enhances interpretability and
resilience against evolving threat landscapes. This contribution
represents a meaningful stride toward integrating the progress
of modern language modeling with the practical demands of
cybersecurity.
METHODOLOGY
This section outlines the methodology behind the
development of XGen-Q, a novel large language model
framework specialized for malware behavior detection and
analysis in software security. XGen-Q uses the Qwen-Coder
architecture and is trained with malware samples from
assembly and source code. It combines RAG with a multi-
step prompt approach to provide accurate classification and
clear forensic explanations. Figure 1 illustrates a simple
interaction between input and our proposed model. The
following subsections describe the dataset preparation,
domain-specific pretraining, model architecture and training,
prompt engineering, and semantic post-processing methods
that contribute to XGen-Q’s effectiveness. Figure 2 provides
an overview of the research model.

Fig. 2: The general diagram of the proposed framework.
Phase 1: Dataset Collection and Preparation
In this study, we used SBAN [27] dataset. The dataset con-
tain diverse samples across multiple programming languages
and malware families, allowing the model to generalize across
real-world threats (see Table I). We utilize this dataset for pre-
training the LLM. Additional details will follow in Phase 2.
TABLE I: Size of malware datasets (source code and assembly
only) from PE files - source code (Src) and assembly (Assem).
Dataset Source NLD Assem Binary
1. BODMAS 93711 93711 92317 88605
2. MalwareBazzar 14746 14746 14051 13973
3. Sorel20m 81584 81584 81177 79166
4. Dike 17431 17431 12138 11726
5. xLangKode 468679 468679 5974 13299
Total 676151 676151 205657 206769
Phase 2: Domain-Specific Pre-Training
To effectively detect and analyze malware, models need
specialized training on security-focused data. Domain-specific
pre-training helps the model to learn malware behaviors and
threat patterns that general programming data misses. The next
subsections cover why domain adaptation matters and how
XGen-Q was pretrained on malware samples.Importance of Domain Adaptation:Generic LLMs which
are trained on broad programming corpora lack special-
ized knowledge of malware behaviors and threat signatures.
Domain adaptation through targeted pretraining on security
datasets enables the model to internalize malware-specific
tactics, obfuscation strategies, and exploitation methods. Such
adaptation improves the model’s understanding of subtle be-
havioral indicators that is missed by general-purpose models.
It also enhances the model’s ability to explain decisions using
the correct cybersecurity terminology, which is critical for trust
and adoption in real-world analysis settings.
Pre-training on Malware samples:To specializeXGen-
Q, we conduct domain-adaptive pretraining using malware
samples from Table I. We use Causal Language Modeling
(CLM) [28, 29] for training the model. Exposure to real-world
malware patterns enables the model to develop contextual
awareness that supports accurate detection and interpretation
of malicious behavior embedded in malware code analysis.
Phase 3: Model Architecture and Training
XGen-Q is built upon theQwen-Coderlanguage model,
specifically theQwen2.5-Coder-1.5B-Instructvari-
ant. This version balances performance and computational
efficiency, offering a 1.5B parameter model optimized for
code understanding with a large token context window. The
pretraining process is based on the malware samples .

Phase 4: LLM-Prompt Design for Security Tasks
This model employs a two-stage prompt strategy to combine
explainability with precision. Prompt 1 instructs the LLM
to generate a structured forensic analysis that includes con-
clusion, reasoning, evidence, and suspicious behavior expla-
nation. Prompt 2 compresses this reasoning into a single
actionable label: malware, benign, or partially malicious. This
approach allows flexible pipeline design, where analysts can
interpret full reasoning or rely solely on classification outputs,
depending on operational needs. Figure 3 illustrates how an
input code goes through the process.
Prompt 1 - Expert Malware Behavior Classification:
Prompt 1 is structured to simulate a human cybersecurity
expert’s thought process in static malware analysis. The model
outputs:
1.Conclusion: High-level decision on sample classification.
2.Reasoning: Justification based on observed behavior.
3.Evidence: Key code features such as suspicious function
calls or API usage.
4.Explanation of Suspicious Elements: Description of
why these features are concerning.
RAG-Based Behavior Extraction (Algorithm 1): To
enrich Prompt 1, we employ a RAG mechanism de-
scribed in Algorithm 1. This algorithm extracts the top 10
behavior-relevant keywords from external knowledge from
https://attack.mitre.org/, ensuring the model is guided by the
most contextually relevant security insights.
Algorithm 1: RAG-Based Behavior Keyword Extraction.
Require:Code snippetC
Ensure:Top 10 behavior-related keywordsK
1:EncodeCinto semantic vector representationE
2:Query a pre-indexed knowledge base (attack.mitre.org)
usingE
3:Retrieve topNrelevant documentsD={d 1, d2, . . . , d N}
4:Extract keyword candidates fromDusing TF-IDF
5:Rank all extracted keywords by semantic similarity toC
6:Select the top 10 most relevant keywords asK
7:returnK
Prompt 2 - Final Behavior Classification:Prompt 2
simplifies the structured forensic report by requesting a single
classification label: malware, benign, or partially malicious.
This helps to facilitate automated integration with real-time
pipelines, alert systems, and triage dashboards. By separating
analysis and labeling, the system allows the explanation
module and decision module to evolve independently,
improving flexibility for future updates or policy adjustments.
Multi-Stage Malware Analysis Pipeline (Algorithm 2):
Algorithm 2 outlines the complete multi-stage inference
pipeline of XGen-Q, combining RAG-based retrieval, multi-
step prompt generation, semantic parsing, and optional analystreview. This pipeline ensures robust, explainable classification
with feedback loops for continual improvement.
Algorithm 2: XGen-Q Multi-Stage Malware Analysis Pipeline.
Require:Static code snippetC
Ensure:Final classification labelL∈ {malware, benign, or
partially malicious.}
1:EncodeCinto semantic vector representationE
2:Query a cybersecurity knowledge index usingE
3:Retrieve topNrelevant documentsD
4:Extract behavior-related keywords using NLP-based rank-
ing
5:Select top 10 keywordsKfor prompt injection
6:Create Prompt 1 usingCandK
7:Generate detailed forensic reportRfrom LLM
8:ParseRinto structured outputs:
•Conclusion
•Reasoning
•Code Evidence
•Suspicious Element Explanation
9:Construct Prompt 2 using parsed output fromR
10:Query LLM with Prompt 2 to obtain labelL
11:VerifyL∈ {malware, benign, or partially malicious}
12:Extract and normalize fields for ingestion into SIEM/log
systems
13:Store(C, L, R)in a structured database
14:Use results for trend analysis and threat correlation
15:ifAnalyst review is enabledthen
16:Present structured report for human validation
17:Analyst accepts/modifies classificationL
18:Feedback is stored for pre-training
19:end if
20:ifFeedback availablethen
21:Append corrected samples to feedback set
22:Periodically fine-tune XGen-Q on new feedback
23:end if
24:returnFinal classification labelL
Phase 5: Semantic Handling and Post-Processing
As part of the final stage in the XGen-Q pipeline, semantic
handling and post-processing play a critical role in trans-
forming unstructured LLM outputs into actionable, machine-
readable formats.
Semantic Extraction:The LLM-generated free-text output
is parsed into structured fields: conclusion, reasoning, code
evidence, and suspicious indicators. This parsing enables
automated ingestion into monitoring systems and supports
downstream analytics such as pattern tracking and correlation
of threat events.
Analyst Review and Auditability:Structured outputs sim-
plify manual review, accelerating validation and improving
trust in AI-generated results. Separation between reasoning
and final decision supports auditability, compliance, and post-
incident investigation. Analyst’s feedback is looped into future

Fig. 3: Knowledge injection based on MITRE ATT&CK, to build guidance for Prompt 1, and prompt 2 creation based on
prompt 1’s output.
model updates, continuously improving performance and re-
ducing false outcomes.
EXPERIMENTS ANDIMPLEMENTATION
In this section, we firstly explain the configurations asso-
ciated with the training our model. Then, we demonstrate
the performance of our proposed model and illustrate an
output generated by our framework. Finally, we compare our
proposed model with similar competitors and summarize the
results.
Settings and Computing Configuration
All experiments were conducted on a server equipped
with NVIDIA H100 GPU, offering the computational power
required for large-scale model training. To optimize efficiency
and reduce memory usage, we employed mixed-precision
training [30, 31] during the pretraining phase. In this study,
we used LangChain [32] to orchestrate large language model
tasks and integratedllamaindexfor implementing the
framework and RAG mechanism, utilizing ’attack.mitre.org’
as the primary retrieval source.
The Performance of the Pre-training
Figure 4 presents three subplots that collectively illustrate
the model’s training dynamics over 3 epochs. The first subplot
shows a sharp drop in training loss from an initial value above
0.75 within the first 0.2 epochs, indicating rapid adaptation
and effective optimization. As training progresses, the losssteadily decreases and flattens, suggesting that the model
is approaching convergence with stable and robust learning
behavior. The second subplot, depicting the gradient norm,
begins with high values, typical for early training on new
data, but quickly stabilizes, reflecting a smooth transition from
large corrective updates to fine-tuning, which is a sign of
healthy training. The third subplot illustrates the learning rate
schedule, where an initial warmup phase allows the model to
begin learning gradually, followed by a linear decay that fine-
tunes updates as convergence is approached. Together, these
metrics demonstrate a well-controlled and effective training
process.
Example of the XGen-Q Output
XGen-Q generates detailed reports analyzing software sam-
ples for malicious behavior. Each report is structured to
provide a clear and concise understanding of the sample’s
nature by breaking the analysis into four key components:
conclusion, reasoning, evidence, and final Judgment as showed
in Figures 5 and 6. This structured output facilitates both
automated processing and human review, enabling security
analysts to quickly assess the characteristics of a given sample.
•Conclusionsummarizes the overall assessment of the an-
alyzed sample, directly stating whether the code appears
benign, malicious, or partially malicious. This summary
results from a comprehensive examination of the code’s
behavior and characteristics.

Fig. 4: Training metrics of XGen-Q showing loss reduction,
gradient behavior and, learning rate decay.
•Reasoningoutlines the rationale behind the conclusion,
describing the analytical process and highlighting specific
behaviors or patterns that led to the classification. This
section promotes transparency and helps users understand
the logic behind the model’s judgment.
•Evidencelists concrete findings that support the reason-
ing, such as suspicious API calls, code structures, or
detected obfuscation techniques.
•Final Judgmentprovides a definitive label, categorizing
the sample as malware, benign, or partially malicious.
This final decision streamlines response workflows by
offering a clear and actionable verdict.
1{
2"ID": "malware_sample_0645470.c",
3"conclusion": "Classified as MALWARE
.",
4"reasoning": "Suspicious use of
Windows Update... ",
5"evidence": [
6"CreateProcessA used to execute
update.exe.",
7],
8"final_Judgment": "MALWARE",
9"source_code": "oid
exploitWindowsUpdate() { ... }\
nint main() {
exploitWindowsUpdate(); return 0;
}"
10}
Fig. 5: Example of a sample considered as malware.
Comparing XGen-Q with Existing Code Models
To demonstrate how well the model understands malware
code, we use theperplexitymetric. We evaluated several
LLMs based on this metric to compare their performance.1{
2"ID": "malware_sample_0816286.c",
3"conclusion": "This code is neither
clearly MALWARE nor BENIGN. ",
4"reasoning": "The use of encrypted
DLL injection via
CreateRemoteThread suggests
evasive behavior.",
5"evidence": [
6"Encrypted DLL loaded using
LoadLibraryA.",
7],
8"final_Judgment": "PARTIALLY
MALICIOUS",
9"source_code": "void
inject_polymorphic_dll(DWORD pid)
{ ... }\nint main() { ... }"
10}
Fig. 6: Example of a sample considered as partially malicious.
For ease of reference, the modelsLM-3,DS-1.3B, andPhi-
4correspond respectively to the full model identifiers
meta-llama/Llama-3.1-8B-Instruct[33],
deepseek-ai/deepseek-coder-
1.3b-instruct[34] [35], and
microsoft/Phi-4-mini-instruct.
As shown in Figure 7, XGen-Q demonstrates superior
performance, consistently maintaining lower perplexity scores
across all sample sizes compared to LM-3, DS-1.3B, and Phi-
4. Its curve shows a steep initial decline and stabilizes at
an impressively low level, indicating excellent generalization
capability even as the dataset scales to 6,000 samples. This
robust performance suggests that XGen-Q is particularly well-
optimized for handling large-scale, complex tasks efficiently.
While LM-3 also performs respectably, its perplexity scores
remain slightly higher than XGen-Q’s, especially at larger
sample sizes. The gap between these two models widens
as data volume increases, reinforcing XGen-Q’s scalability
advantage. For applications requiring both precision and the
ability to process massive datasets, XGen-Q clearly emerges as
the top choice, its combination of low perplexity and stability
makes it the standout model in this comparison.
Figure 8 and Table II summarize the comparative perfor-
mance of all evaluated models on both assembly and source
code variants of the same malware samples. Across both code
types,XGen-Qachieves the lowest perplexity scores, out-
performing all baselines. On assembly code, XGen-Q scores
1.530, substantially better thanDS-1.3B(9.183) andPhi-4
(16.713). On source code, it again leads with 1.592, while
the next-best baseline (DS-1.3B) trails at 3.997. These results
clearly demonstrateXGen-Q’s domain-specific strength in
modeling malware code, which is often obfuscated, irregular,
and semantically complex.

Fig. 7: Perplexity values from four models on malware source
code samples.
Fig. 8: Line graph with distinct markers showing individual
perplexity scores.
DISCUSSION, LIMITATIONS ANDFUTUREWORKS
The XGen-Q framework demonstrates the feasibility and
benefits of combining domain-specific pre-training, RAG, and
multi-stage prompt engineering for enhanced software security
analysis. Our evaluation highlights that adapting a strong base
LLM like Qwen, using carefully curated malware and benign
code datasets, leads to substantial improvements in detection,
accuracy, and interpretability. The integration of RAG allows
the model to remain current with the evolving threat landscape
by dynamically incorporating external knowledge at inference
time.
We also mentioned that in this work, we limited the size of
the dataset used for pre-training the LLM. Future work will
explore incorporating multi-modal inputs, such as dynamic
execution traces, binary metadata, and network telemetry, to
further enrich the model’s contextual understanding [36–38].
We also aim to investigate continual learning strategies that
allow XGen-Q to incrementally update its internal knowl-
edge without requiring full retraining. Moreover, integrating
explainable AI techniques will improve the transparency of
the model’s predictions, making its outputs more actionable
and trustworthy for security analysts [39, 40].TABLE II: Perplexity comparison of language models on
assembly and source code levels. Lower perplexity indicates
better performance.
Data Model Perplexity↓Relative to XGen-Q (×)
AssemblyXGen-Q 1.530 1.00× (best)
LM-3 9.972 6.52×
Phi-4 16.713 10.93×
DS-1.3B 9.183 6.00×
Source CodeXGen-Q 1.592 1.00× (best)
LM-3 5.822 3.66×
Phi-4 7.739 4.86×
DS-1.3B 3.997 2.51×
CONCLUSION
This paper presents XGen-Q, a domain-adapted large lan-
guage model built on the Qwen-Coder architecture for robust
and interpretable malware analysis. Trained on a large-scale
dataset of over one million malware samples and enhanced
through domain-specific pretraining techniques, XGen-Q is
capable of capturing complex and subtle code patterns as-
sociated with malicious behavior. By incorporating retrieval-
augmented generation (RAG) and a multi-stage inference strat-
egy, the model supports detailed and context-aware forensic
reporting, even in the presence of advanced code obfuscation.
Experimental results demonstrate that XGen-Q achieves lower
perplexity and strong generalization to previously unseen
samples. These findings underscore the potential of XGen-Q
as a powerful and explainable tool for advancing automated
malware analysis.
REFERENCES
[1] M. B ¨ohme and E. Bodden, “Software security analysis in
2030 and beyond: A research roadmap,”ACM Transac-
tions on Software Engineering and Methodology, vol. 34,
no. 5, pp. 1–26, 2025.
[2] N. Rahimi, B.-A. Schuelke-Leech, and M. Mirhassani,
“A comprehensive review of security vulnerabilities in
heavy-duty vehicles: Comparative insights and current
research gaps,”Computers & Security, p. 104452, 2025.
[3] N. Mohamed, “Artificial intelligence and machine learn-
ing in cybersecurity: a deep dive into state-of-the-art
techniques and future paradigms,”Knowledge and Infor-
mation Systems, pp. 1–87, 2025.
[4] I. Aldasoro, S. Doerr, L. Gambacorta, S. Notra,
T. Oliviero, and D. Whyte, “Generative artificial intelli-
gence and cyber security in central banking,”Journal of
Financial Regulation, vol. 11, no. 1, pp. 119–128, 2025.
[5] M. M. Rahman, S. Hossain, B. Bhusal, and N. Kshetri,
“Cyber ai trends: Future trends in ai for cyberbullying
prevention,” inCombating Cyberbullying With Genera-
tive AI, pp. 279–298, IGI Global Scientific Publishing,
2025.
[6] H. Xu, S. Wang, N. Li, K. Wang, Y . Zhao, K. Chen,
T. Yu, Y . Liu, and H. Wang, “Large language models

for cyber security: A systematic literature review,”arXiv
preprint arXiv:2405.04760, 2024.
[7] S. Tian, T. Zhang, J. Liu, J. Wang, X. Wu, X. Zhu,
R. Zhang, W. Zhang, Z. Yuan, S. Mao,et al., “Explor-
ing the role of large language models in cybersecurity:
A systematic survey,”arXiv preprint arXiv:2504.15622,
2025.
[8] W. Kasri, Y . Himeur, H. A. Alkhazaleh, S. Tarapiah,
S. Atalla, W. Mansoor, and H. Al-Ahmad, “From vul-
nerability to defense: The role of large language models
in enhancing cybersecurity,”Computation, vol. 13, no. 2,
p. 30, 2025.
[9] A. A. Hossain, M. K. PK, J. Zhang, and F. Amsaad,
“Malicious code detection using llm,” inNAECON 2024-
IEEE National Aerospace and Electronics Conference,
pp. 414–416, IEEE, 2024.
[10] J. Al-Karaki, M. A.-Z. Khan, and M. Omar, “Explor-
ing llms for malware detection: Review, framework de-
sign, and countermeasure approaches,”arXiv preprint
arXiv:2409.07587, 2024.
[11] C. Zhou, Y . Liu, W. Meng, S. Tao, W. Tian, F. Yao, X. Li,
T. Han, B. Chen, and H. Yang, “Srdc: Semantics-based
ransomware detection and classification with llm-assisted
pre-training,” inProceedings of the AAAI Conference on
Artificial Intelligence, vol. 39, pp. 28566–28574, 2025.
[12] X. Qian, X. Zheng, Y . He, S. Yang, and L. Cav-
allaro, “Lamd: Context-driven android malware de-
tection and classification with llms,”arXiv preprint
arXiv:2502.13055, 2025.
[13] Z. Yu, M. Wen, X. Guo, and H. Jin, “Maltracker: A fine-
grained npm malware tracker copiloted by llm-enhanced
dataset,” inProceedings of the 33rd ACM SIGSOFT In-
ternational Symposium on Software Testing and Analysis,
pp. 1759–1771, 2024.
[14] I. Hasanov, S. Virtanen, A. Hakkala, and J. Isoaho,
“Application of large language models in cybersecurity:
A systematic literature review,”IEEE Access, 2024.
[15] J. Zhang, H. Bu, H. Wen, Y . Liu, H. Fei, R. Xi, L. Li,
Y . Yang, H. Zhu, and D. Meng, “When llms meet cyber-
security: A systematic literature review,”Cybersecurity,
vol. 8, no. 1, pp. 1–41, 2025.
[16] R. Feng, H. Chen, S. Wang, M. M. Karim, and Q. Jiang,
“Llm-maldetect: A large language model-based method
for android malware detection,”IEEE Access, 2025.
[17] D. Lee, J. Kim, J. Kim, S.-w. Hwang, and J. Park, “trag:
Term-level retrieval-augmented generation for domain-
adaptive retrieval,” inProceedings of the 2025 Confer-
ence of the Nations of the Americas Chapter of the Asso-
ciation for Computational Linguistics: Human Language
Technologies (Volume 1: Long Papers), pp. 6566–6578,
2025.
[18] Q. Long, W. Wang, and S. J. Pan, “Adapt in contexts:
Retrieval-augmented domain adaptation via in-context
learning,”arXiv preprint arXiv:2311.11551, 2023.
[19] T. Leemann, P. Petridis, G. Vietri, D. Manousakas,
A. Roth, and S. Aydore, “Auto-gda: Automatic do-main adaptation for efficient grounding verification
in retrieval augmented generation,”arXiv preprint
arXiv:2410.03461, 2024.
[20] R. Xu, H. Liu, S. Nag, Z. Dai, Y . Xie, X. Tang,
C. Luo, Y . Li, J. C. Ho, C. Yang,et al., “Simrag: Self-
improving retrieval-augmented generation for adapting
large language models to specialized domains,”arXiv
preprint arXiv:2410.17952, 2024.
[21] P. Lewis, E. Perez, A. Piktus, F. Petroni, V . Karpukhin,
N. Goyal, H. K ¨uttler, M. Lewis, W.-t. Yih,
T. Rockt ¨aschel,et al., “Retrieval-augmented generation
for knowledge-intensive nlp tasks,”Advances in neural
information processing systems, vol. 33, pp. 9459–9474,
2020.
[22] T. Brown, Mann,et al., “Language models are few-shot
learners,”Advances in neural information processing
systems, vol. 33, pp. 1877–1901, 2020.
[23] Z. Xu and V . S. Sheng, “Detecting ai-generated code
assignments using perplexity of large language models,”
inProceedings of the aaai conference on artificial intel-
ligence, vol. 38, pp. 23155–23162, 2024.
[24] N. Cooper and T. Scholak, “Perplexed: Understand-
ing when large language models are confused,”arXiv
preprint arXiv:2404.06634, 2024.
[25] J. Xu, H. Zhang, Y . Yang, Z. Cheng, J. Lyu, B. Liu,
X. Zhou, L. Yang, A. Bacchelli, Y . K. Chiam,et al.,
“Investigating efficacy of perplexity in detecting llm-
generated code,”arXiv preprint arXiv:2412.16525, 2024.
[26] M. A. Yusof and S. Saee, “Code switching: exploring
perplexity and coherence metrics for optimizing topic
models of historical documents,”International Journal of
Systematic Innovation, vol. 8, no. 4, pp. 103–118, 2024.
[27] H. Jelodar, M. Meymani, S. Bai, R. Razavi-Far, and A. A.
Ghorbani, “Sban: A framework & multi-dimensional
dataset for large language model pre-training and soft-
ware code mining,” inProceedings of the 2025 IEEE
International Conference on Data Mining Workshops
(ICDMW), IEEE, 2025.
[28] A. Wu, K. Kuang, M. Zhu, Y . Wang, Y . Zheng, K. Han,
B. Li, G. Chen, F. Wu, and K. Zhang, “Causality for large
language models,”arXiv preprint arXiv:2410.15319,
2024.
[29] Z. Zhu, H. Yu, C. Shen, J. Du, Z. Shen, and Z. Wang,
“Causal language model aided sequential decoding with
natural redundancy,”IEEE Transactions on Communica-
tions, vol. 71, no. 5, pp. 2685–2697, 2023.
[30] P. Micikevicius, S. Narang, J. Alben, G. Diamos,
E. Elsen, D. Garcia, B. Ginsburg, M. Houston,
O. Kuchaiev, G. Venkatesh,et al., “Mixed precision
training,”arXiv preprint arXiv:1710.03740, 2017.
[31] D. Das, N. Mellempudi, D. Mudigere, D. Kalamkar,
S. Avancha, K. Banerjee, S. Sridharan, K. Vaidyanathan,
B. Kaul, E. Georganas,et al., “Mixed precision training
of convolutional neural networks using integer opera-
tions,”arXiv preprint arXiv:1802.00930, 2018.
[32] H. Chase and L. Contributors, “Langchain,” 2022.

[33] H. Touvron, T. Lavril, G. Izacard, X. Martinet, M.-A.
Lachaux, T. Lacroix, B. Rozi `ere, N. Goyal, E. Hambro,
F. Azhar,et al., “Llama: Open and efficient founda-
tion language models,”arXiv preprint arXiv:2302.13971,
2023.
[34] A. Liu, B. Feng, B. Xue, B. Wang, B. Wu, C. Lu,
C. Zhao, C. Deng, C. Zhang, C. Ruan,et al., “Deepseek-
v3 technical report,”arXiv preprint arXiv:2412.19437,
2024.
[35] A. Abouelenin, A. Ashfaq, A. Atkinson, H. Awadalla,
N. Bach, J. Bao, A. Benhaim, M. Cai, V . Chaudhary,
C. Chen,et al., “Phi-4-mini technical report: Compact
yet powerful multimodal language models via mixture-
of-loras,”arXiv preprint arXiv:2503.01743, 2025.
[36] G. Gebrehans, N. Ilyas, K. Eledlebi, W. T. Lunardi,
M. Andreoni, C. Y . Yeun, and E. Damiani, “Generative
adversarial networks for dynamic malware behavior:
A comprehensive review, categorization, and analysis,”
IEEE Transactions on Artificial Intelligence, 2025.
[37] F. Khorrami, R. Karri, and P. Krishnamurthy, “Real-
time multi-modal subcomponent-level measurements for
trustworthy system monitoring and malware detection,”
arXiv preprint arXiv:2501.13081, 2025.
[38] M. Shafi, “Intruders’ behavior unveiled: A dual-tier
behavior-driven model for malicious activity detection in
iot network using graph learning,” 2024.
[39] H. Jelodar, S. Bai, P. Hamedi, H. Mohammadian,
R. Razavi-Far, and A. Ghorbani, “Large language
model (llm) for software security: Code analysis, mal-
ware analysis, reverse engineering,”arXiv preprint
arXiv:2504.07137, 2025.
[40] H. Jelodar, M. Meymani, and R. Razavi-Far, “Large
language models (llms) for source code analysis:
applications, models and datasets,”arXiv preprint
arXiv:2503.17502, 2025.