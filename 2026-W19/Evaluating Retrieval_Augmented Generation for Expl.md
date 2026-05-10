# Evaluating Retrieval-Augmented Generation for Explainable Malware Analysis

**Authors**: Jayson Ng, Amin Milani Fard

**Published**: 2026-05-04 20:29:38

**PDF URL**: [https://arxiv.org/pdf/2605.03140v1](https://arxiv.org/pdf/2605.03140v1)

## Abstract
Large Language Models (LLMs) are increasingly being used as security engineering tools to summarize and explain malware behavior to analysts. A common assumption is that Retrieval-Augmented Generation (RAG) improves explanation quality by injecting external security knowledge. In this work, we empirically evaluate this assumption for malware explanation using VirusTotal reports as structured input. Across multiple LLMs, we find that RAG frequently degrades explanation quality by introducing distracting or weakly related context and adding narrative noise or generic write-ups. Our results highlight a practical risk in security-critical pipelines for malware explanation that RAG can be counterproductive when structured security evidence is already sufficient. We argue that malware explanation is primarily a signal-extraction task, not a knowledge-retrieval problem, and outline design recommendations for secure development workflows.

## Full Text


<!-- PDF content starts -->

Evaluating Retrieval-Augmented Generation for Explainable
Malware Analysis
Jayson Ng
New York Institute of Technology
Vancouver, BC, Canada
jng10@nyit.eduAmin Milani Fard
New York Institute of Technology
Vancouver, BC, Canada
amilanif@nyit.edu
Abstract
Large Language Models (LLMs) are increasingly being used as secu-
rity engineering tools to summarize and explain malware behavior
to analysts. A common assumption is that Retrieval-Augmented
Generation (RAG) improves explanation quality by injecting ex-
ternal security knowledge. In this work, we empirically evaluate
this assumption for malware explanation using VirusTotal reports
as structured input. Across multiple LLMs, we find that RAG fre-
quently degrades explanation quality by introducing distracting
or weakly related context and adding narrative noise or generic
write -ups. Our results highlight a practical risk in security-critical
pipelines for malware explanation that RAG can be counterproduc-
tive when structured security evidence is already sufficient. We
argue that malware explanation is primarily a signal-extraction
task, not a knowledge-retrieval problem, and outline design recom-
mendations for secure development workflows.
Keywords
Malware Analysis, Secure Development, Explainable AI, LLM, RAG
ACM Reference Format:
Jayson Ng and Amin Milani Fard. 2026. Evaluating Retrieval-Augmented
Generation for Explainable Malware Analysis. InPoster at ACM Secure Devel-
opment Conference (SecDev ’26), July 5–9, 2026, Montreal, QC, Canada.ACM,
New York, NY, USA, 2 pages. https://doi.org/10.1145/nnnnnnn.nnnnnnn
1 Introduction
Understanding why a binary is malicious is as important as deter-
mining whether it is malicious. While platforms such as VirusTotal
provide rich detection labels, behavioral traces, and metadata, they
do not offer analyst -ready explanations that connect low -level indi-
cators to high -level attacker intent. These outputs lack a coherent
causal narrative explaining why certain behaviors matter, how ex-
ecution unfolds, or how signals indicate malicious activity. Such
explainability is essential for threat hunting, incident response, and
regulatory or audit requirements.
Recently, LLMs have been proposed as security engineering tools
to translate low-level indicators (e.g., API calls, registry edits, and
network activity) into natural-language explanations that support
triage, incident response, and reporting. For example, MalGPT [ 9]
demonstrates that generative models can produce meaningful ex-
planations from malware binaries by learning latent behavioral
This work is licensed under a Creative Commons Attribution 4.0 International License.
Poster at SecDev ’26, Montreal, QC, Canada
©2026 Copyright held by the owner/author(s).
ACM ISBN 978-x-xxxx-xxxx-x/YYYY/MM
https://doi.org/10.1145/nnnnnnn.nnnnnnn
LLMData from V irusT otal  API
+
System Prompt
Explanation
Output
VectorDB  Retrieved
  Context Knowledge
Base   Data
  ChunksEmbedding
ModelFigure 1: RAG LLM architecture for malware explanation.
patterns. RAG [ 6] is often treated as a default enhancement for
LLM-based systems, and has been applied to related pipelines such
as CVE association and reasoning over decompiled artifacts [ 2,7].
VulRAG [ 3] shows that incorporating structured domain knowledge
improves reasoning about software vulnerabilities and reduces hal-
lucinations. However, prior studies do not evaluate whether RAG
improves explanation quality in malware analysis. Unlike domains
that rely on external documentation, malware reports often encap-
sulate sufficient contextual evidence, raising the operational risk
that indiscriminate retrieval introduces distraction or noise leading
to misleading explanations in security workflows.
2 Proposed Approach
We evaluate the explainability of LLMs with and without RAG for
malware explanation tasks when structured evidence already exists.
Implementation.We use LlamaIndex as the RAG orchestration
framework, ChromaDB for persistent vector storage, and Open-
Router as the LLM gateway. We apply all -MiniLM -L6-v2 and Ope-
nAI’s text -embedding -3-large for embeddings. Documents are chun-
ked into 1024 tokens with a 100 -token overlap. Retrieval employs
top-𝑘similarity search with a 0.5 threshold. The knowledge base
consists of 26 documents parsed using LlamaIndex’s SimpleDirecto-
ryReader. Our dataset is derived from the MalGPT corpus [ 9] with
1,702 VirusTotal reports, including structured indicators such as
API calls and registry operations. MD5 hashes serve as identifiers.
Our implementation is available for download1.
System prompt.The prompt has 10 sections: (1) one -line verdict
with confidence, (2) non -technical summary, (3) evidence bullets
with JSON citations, (4) malware family mapping, (5) grouped Indi-
cator of Compromises (IoCs) with paths, (6) behavioral summary,
(7) confidence score, (8) recommended actions for technical and
non-technical audiences, (9) optional FAQ, and (10) raw evidence
appendix. This structure enforces consistency and traceability.
1https://github.com/nyit-vancouver/RAG-Explain-MalwarearXiv:2605.03140v1  [cs.CR]  4 May 2026

Poster at SecDev ’26, July 5–6, 2026, Montreal, QC, Canada Jayson Ng and Amin Milani Fard
Model # Parameters Embeddings BERTScore
GPT OSS 21B None 0.8407
GPT OSS 21B all-MiniLM-L6-v2 0.8184
GPT OSS 21B text-embedding-3-large 0.8493
DeepSeek-R1 671B None0.8617
DeepSeek-R1 671B all-MiniLM-L6-v2 0.8583
DeepSeek-R1 671B text-embedding-3-large 0.8582
GPT-5.1 est. 2.5T None1.0000
GPT-5.1 est. 2.5T all-MiniLM-L6-v2 0.8778
GPT-5.1 est. 2.5T text-embedding-3-large0.9047
Table 1: Malware explanation compared to GPT -5.1 w/o RAG.
Knowledge base.The knowledge base integrates four compo-
nents: (1) the MITRE ATT&CK framework (Enterprise, ICS, Mo-
bile); (2) research and best -practice guidance, including MalGPT
and CISA/MITRE mapping recommendations; (3) VirusTotal tech-
nical documentation covering behavioral schemas, reports, and
IoC definitions; and (4) malware intelligence datasets, including
malware family taxonomies and CVE records.
Evaluation.We conduct an ablation study comparing LLMs
with and without RAG against GPT -5.1 (Table 1). Each model runs
under identical prompts using either VirusTotal JSON alone or
JSON augmented with retrieved context. To isolate retrieval effects,
no fine -tuning is performed. We do not compare directly with
MalGPT, as it represents a different paradigm based on a trained
multi -modal architecture rather than external retrieval. Since our
focus is on retrieval -induced effects when explaining structured
sandbox evidence, such a comparison would not isolate the impact
of RAG. We measure explanation quality using BERTScore [ 10],
which captures semantic similarity using contextual embeddings
and is robust to paraphrasing and structural variation suitable for
evaluating malware explanations where equivalent meanings may
be expressed using different terminology. It is also sensitive to
semantic drift introduced by loosely related retrieved passages.
3 Discussion
Results analysis.Based on commonly observed ranges in the
literature, BERTScores 0.9-1 indicate excellent alignment, scores be-
tween 0.85 and 0.9 reflect strong semantic similarity, and scores be-
tween 0.8 and 0.85 indicate moderate quality. Differences as small as
0.01–0.02 are meaningful in technical domains. Excluding GPT -5.1,
the highest score is achieved by DeepSeek-R1 without RAG (0.8617),
indicating that retrieval does not effectively improve explanation
quality. As shown in Table 1, the all -MiniLM -L6-v2 retriever de-
grades performance for both models, particularly GPT -OSS-20B.
While text -embedding -3-large improves GPT -OSS-20B slightly, it
does not benefit DeepSeek-R1. Compact embeddings often retrieve
semantically similar but low -utility passages, such as generic mal-
ware descriptions, which distract the generator. Prior work formal-
izes this distracting effect and shows that weakly relevant retrieval
degrades performance [ 1]. Long -context RAG studies similarly ob-
serve performance decline as retrieved context accumulates hard
negatives [ 4]. These findings align with our observations. VirusTo-
tal JSON reports already provide tightly scoped behavioral evidence.
Injecting external descriptions shifts model attention toward plau-
sible but irrelevant narratives, reducing semantic alignment. For
strong models such as DeepSeek-R1, additional context appearsredundant or mildly contradictory [ 5]. In this setting, malware ex-
plainability is primarily a signal -extraction problem rather than a
knowledge -retrieval one. Retrieval thresholds were intentionally
permissive to expose potential distractions.
Security engineering implications.Our findings highlight
important implications for secure system design. The risk of au-
tomation bias increases when RAG-based explanations that drift
from the underlying evidence are not manually verified. Analysts
benefit more from consistency and traceability than from expanded
narratives. From a tooling perspective, malware explanation is bet-
ter framed as structured signal extraction rather than knowledge
synthesis. In security-critical pipelines, adding context without
clear necessity increases cognitive load and the risk of error.
Recommendations.Our results suggest that when base evi-
dence is already sufficient, RAG can degrade explanation quality. To
mitigate retrieval -induced noise, we recommend: (1) skipping RAG
when structured reports are sufficient using context -sufficiency pre-
dictors [ 5]; (2) applying domain -tuned re -rankers with strict 𝑘limits
to avoid hard negatives [ 4]; (3) replacing fixed -window chunking
with behavior -or IoC -centric chunks enriched with metadata [ 8];
(4) adversarial fine -tuning with hard distractors [ 1]; and (5) explor-
ing alternatives such as cache -augmented generation, structured
context engineering, or knowledge-graph-based augmentation.
4 Conclusion and Future Work
While RAG is often proposed to enhance LLMs with external knowl-
edge, our experiments show that low -relevance or poorly retrieved
context can degrade malware explanation quality. This aligns with
known limitations of RAG and suggests that retrieval is not uni-
versally beneficial—particularly in domains driven by structured
reasoning rather than knowledge completion. Malware triage us-
ing VirusTotal primarily involves extracting signals from struc-
tured artifacts such as behaviors, signatures, and hashes. When
this evidence is already sufficient, additional retrieved text can di-
lute salient signals, increase cognitive load, and distract the model.
As future work, we will explore alternatives to RAG, including
Cache -Augmented Generation and Context Engineering, which
better preserve structured evidence without introducing noise.
References
[1]Chen Amiraz, Florin Cuconasu, Simone Filice, and Zohar Karnin. 2025. The Distracting Effect:
Understanding Irrelevant Passages in RAG. InACL’25.
[2]Eduard Andrei Cristea, Petter Molnes, and Jingyue Li. 2026. MalCVE: Malware Detection and
CVE Association Using Large Language Models.arXiv(2026). https://arxiv.org/pdf/2510.
15567v2
[3]Xiaoyu Du, Guanyu Zheng, Kai Wang, Jie Feng, Wei Deng, Ming Liu, Bin Chen, Xin Peng, and
Yutong Lou. 2024. Vul-RAG: Enhancing LLM-Based Vulnerability Detection via Knowledge-
Level RAG.arXiv preprint arXiv:2406.11147(2024).
[4]Bowen Jin, Jinsung Yoon, Jiawei Han, and Sercan Arik. 2025. Long-Context LLMs Meet RAG:
Overcoming Challenges for Long Inputs in RAG. InICLR’25, Vol. 2025. 37784–37822.
[5]Hailey Joren, Jianyi Zhang, Chun-Sung Ferng, Da-Cheng Juan, Ankur Taly, and Cyrus
Rashtchian. 2025. Sufficient Context: A New Lens on Retrieval-Augmented Generation Systems.
InICLR’25.
[6]Patrick Lewis, Ethan Perez, Aleksandra Piktus, Fabio Petroni, Vladimir Karpukhin, Naman
Goyal, Heinrich Küttler, Mike Lewis, Wen-tau Yih, Tim Rocktäschel, et al .2020. Retrieval-
augmented generation for knowledge-intensive nlp tasks.NeurIPS’2033 (2020), 9459–9474.
[7]Moqsadur Rahman, Krish O. Piryani, Aaron M. Sanchez, Sai Munikoti, Luis De La Torre,
Maxwell S. Levin, Monika Akbar, Mahmud Hossain, Monowar Hasan, and Mahantesh Halap-
panavar. 2024.Retrieval Augmented Generation for Robust Cyber Defense. Technical Report
PNNL-36792.
[8]Elena Samuylova. 2025. A complete guide to RAG evaluation: metrics, testing and best practices.
Evidently AI. https://www.evidentlyai.com/llm-guide/rag-evaluation
[9]Mohd Saqib, Benjamin CM Fung, Steven HH Ding, and Philippe Charland. 2025. MalGPT: A
Generative Explainable Model for Malware Binaries. InECML PKDD’25. Springer, 130–148.
[10] Tianyi Zhang, Varsha Kishore, Felix Wu, Kilian Q. Weinberger, and Yoav Artzi. 2020. BERTScore:
Evaluating Text Generation with BERT. InICLR’20.