# Privacy-Preserving RAG via Multi-Agent Semantic Rewriting: Achieving Confidentiality Without Compromising Contextual Fidelity

**Authors**: Yuanhe Zhao, Tianyu Zhang, Huafei Xing, Derek F. Wong, Jianbin Li, Tao Fang

**Published**: 2026-06-23 14:21:41

**PDF URL**: [https://arxiv.org/pdf/2606.24623v1](https://arxiv.org/pdf/2606.24623v1)

## Abstract
Retrieval-Augmented Generation enhances large language models by incorporating external knowledge, but deploying it in sensitive scenarios risks privacy leakage via malicious prompts. To address this, we propose a multi-agent framework that sanitizes retrieved content through semantic rewriting. By employing three specialized agents for privacy extraction, semantic analysis, and reconstruction, our approach collaboratively removes sensitive identifiers while preserving the semantic core. We evaluate the framework on the ChatDoctor and Wiki-PII datasets across six large language models. Experimental results demonstrate a significant reduction in privacy leakage under targeted attacks. For instance, we reduced targeted information exposure in LLaMA-3-8B from 144 instances in the baseline to just 1. Furthermore, we maintain strong contextual fidelity with a BLEU-1 score of 0.122, outperforming the existing SAGE method's 0.117. Finally, the framework operates as an asynchronous preprocessing module, introducing no additional latency to online inference, as all rewriting is executed as a one-time offline preprocessing step. To promote reproducibility, the source code of this work is publicly available at https://github.com/foursoils/Privacy-Preserving-RAG.

## Full Text


<!-- PDF content starts -->

Privacy-Preserving RAG via Multi-Agent Semantic Rewriting:
Achieving Confidentiality Without Compromising Contextual
Fidelity
Yuanhe Zhaoa, Tianyu Zhangb, Huafei Xinga, Derek F. Wongc, Jianbin Liaand Tao Fangd,∗
aSchool of Control and Computer Engineering, North China Electric Power University, Beijing, 102206, China
bDepartment of Computer & Information Science & Engineering, University of Florida, Gainesville, Florida, 32611, USA
cNLP2CT Lab, Department of Computer and Information Science, University of Macau, Macao, 999078, Macao SAR
dInstitute of International Language Services Studies, Macau Millennium College, Macao, 999078, Macao SAR
A R T I C L E I N F O
Keywords :
Retrieval-Augmented Generation
Privacy preservation
Semantic rewriting
Multi-agent-based framework
PACS: 89.20.Ff
07.05.Bx
MSC: 68M25
68P20
68T50A B S T R A C T
Retrieval-Augmented Generation enhances large language models by incorporating external
knowledge, but deploying it in sensitive scenarios risks privacy leakage via malicious prompts.
To address this, we propose a multi-agent framework that sanitizes retrieved content through
semantic rewriting. By employing three specialized agents for privacy extraction, semantic
analysis, and reconstruction, our approach collaboratively removes sensitive identifiers while
preserving the semantic core. We evaluate the framework on the ChatDoctor and Wiki-PII datasets
across six large language models. Experimental results demonstrate a significant reduction in
privacy leakage under targeted attacks. For instance, we reduced targeted information exposure
in LLaMA-3-8B from 144 instances in the baseline to just 1. Furthermore, we maintain strong
contextual fidelity with a BLEU-1 score of 0.122, outperforming the existing SAGE method’s
0.117. Finally, the framework operates as an asynchronous preprocessing module, introducing
no additional latency to online inference, as all rewriting is executed as a one-time offline
preprocessing step. To promote reproducibility, the source code of this work is publicly available
athttps://github.com/foursoils/Privacy-Preserving-RAG .
1. Introduction
Retrieval-Augmented Generation (RAG) has gained prominence as an effective method for enhancing large language
models (LLMs) by incorporating external knowledge, overcoming the constraints of their static, pre-trained knowledge
bases (Karpukhin, Oguz, Min, Lewis, Wu, Edunov, Chen and Yih, 2020; De Jong, Zemlyanskiy, Ainslie, FitzGerald,
Sanghai, Sha and Cohen, 2023; Martínez-Murillo, Maestre, Suárez, Lloret and Moreda, 2026). By retrieving relevant
information from external databases, RAG improves the contextual accuracy and relevance of LLM outputs, particularly
for tasks requiring domain-specific expertise or up-to-date information. RAG has proven effective in applications such
as specialized question answering (Siriwardhana, Weerasekera, Wen, Kaluarachchi, Rana and Nanayakkara, 2023;
Opoku, Sheng and Zhang, 2025; Lee and Lee, 2026) and professional document completion (Ram, Levine, Dalmedigos,
Muhlgay, Shashua, Leyton-Brown and Shoham, 2023). The approach retrieves pertinent documents from an external
knowledge base and integrates them with the user’s input to generate enriched prompts for the LLM, enabling precise
and current responses.
Although these benefits exist, RAG systems based on private knowledge sources present serious risks of data
exposure (Zeng, Zhang, He, Liu, Xing, Xu, Ren, Chang, Wang, Yin et al., 2024; Zou, Geng, Wang and Jia, 2025; Cohen,
Bitton and Nassi, 2024; Di Maio, Cosci, Maggini, Poggioni and Melacci, 2025; Morris, Kuleshov, Shmatikov and Rush,
2023; Liu, Yao, Wu, Qin, Lin, Ren and Chen, 2024; Feng, Sun, Shen and Li, 2026a). An example is how Zeng et al.
(2024) showed that well-crafted prompts could be used to leverage RAG models into reveal sensitive information about
the retrieval corpus through either untargeted attacks where whole original sentences can be reconstructed, or targeted
attacks where certain private details may be extracted. The risk is particularly acute in healthcare, where patient records
used in dialogue systems can expose sensitive personal data, making robust privacy protection a clinical necessity.
∗Corresponding author: Tao Fang (taofang@mmc.edu.mo). Jianbin Li (lijb87@ncepu.edu.cn) is the Co-corresponding author.
120242227179@ncepu.edu.cn (Y . Zhao); tianyuzhang.me@gmail.com (T. Zhang); 120242227199@ncepu.edu.cn
(H. Xing); derekfw@um.edu.mo (D.F. Wong); lijb87@ncepu.edu.cn (J. Li); taofang@mmc.edu.mo (T. Fang)
ORCID (s):
:Preprint submitted to Elsevier Page 1 of 23arXiv:2606.24623v1  [cs.CL]  23 Jun 2026

Figure 1: Comparison of semantic rewriting strategies. The degree to which a strategy preserves the original semantics
significantly impacts the rewritten content, which in turn affects the performance and reliability of downstream tasks.
To mitigate these confidentiality threats, prior work has proposed replacing original retrieval content with synthetic
data at the data source level Zeng, Zhang, He, Ren, Zheng, Lu, Xu, Liu, Xing and Tang (2025). This approach adopts
a two-stage framework combining attribute extraction and agent-based rewriting to preserve key semantics from
the original data while iteratively refining the output to reduce privacy risks. However, because surrogate content is
generated entirely by the model, it can diverge substantially from the original in semantic specificity and factual detail,
undermining the reliability of downstream generation. In contrast, we address privacy leakage by operating directly on
the retrieved text: sensitive information is identified and removed, as illustrated in Figure 1, and the remaining content
is rewritten to preserve the critical semantic content.
In this work, we propose a privacy-preserving rewriting framework integrated into the RAG pipeline, positioned
between the retrieval and generation stages to ensure data confidentiality while maintaining contextual fidelity. The
framework comprises three collaborative agents: (1) the Privacy Extraction Agent (Pri-Extra Agent), which extracts
privacy-sensitive segments; (2) the Semantic Analysis Agent (Sem-Extra Agent), which identifies task-relevant semantic
content using structured prompts; and (3) the Reconstruction Agent, which rewrites the text by removing sensitive
segments and applying techniques such as synonym substitution or structural rephrasing to preserve core semantics.
By introducing structured prompts and span-level annotation mechanisms, we achieve fine-grained control over the
text, enabling the model to accurately identify sensitive information to be removed, and to retain key semantic content
through abstraction, synonym substitution, or structural transformation. We conduct comprehensive evaluations of our
method on two privacy-sensitive datasets: ChatDoctor andWiki-PII .
Experimental results show that our method significantly reduces the exposure risk of sensitive information under
both targeted and untargeted attack scenarios. Meanwhile, the quality of the rewritten text, as measured by automatic
metrics such as BLEU Papineni, Roukos, Ward and Zhu (2002) and ROUGE Lin (2004) as well as factual consistency
assessments, remains comparable to that of the standard RAG pipeline using unprotected documents. Qualitative
analyses further confirm that, despite the privacy-stripping process, our approach maintains contextual fidelity and
coherence. The multi-agent pipeline operates entirely offline, functioning as a one-time preprocessing step that imposes
no additional latency on online inference.
Research Objectives
We primarily aim to resolve the conflict between data utilization and data privacy in Retrieval-Augmented Generation
(RAG) systems. Specifically, we focus on the following objectives:
:Preprint submitted to Elsevier Page 2 of 23

•To analyze privacy vulnerabilities in RAG: To investigate how current RAG architectures expose sensitive
information through prompt injection attacks, specifically in high-stakes domains like healthcare.
•To develop a multi-agent semantic rewriting framework: To design a multi-agent system which can discern
between sensitive identifiers (privacy) and important context (utility), such that removing privacy does not
compromise the semantic value of the text.
•To empirically demonstrate that the proposed framework outperforms existing privacy-preserving baselines
(including SAGE and AttrPrompt) in contextual fidelity (BLEU/ROUGE scores) while achieving a significant
reduction in privacy leakage across diverse adversarial settings.
Contributions
The main contributions of this study are as follows:
•We motivate and characterize RAG sanitization as an objective entanglement problem: since private identifiers
and semantic content co-occur in natural text, a single-pass transformation faces an inherent tension between
suppressing leakage and preserving utility, providing a principled motivation for multi-agent task decomposition
beyond system or prompt engineering choices.
•We propose a Multi-Agent Semantic Rewriting Framework in which a Pri-Extra Agent identifies both explicit
and latent quasi-identifiers, a Sem-Extra Agent deconstructs documents into structured knowledge-slot tuples,
and a Reconstruction Agent with Fine-grained Conflict Routing regenerates sanitized text from these tuples alone.
An Asymmetric Retrieval Architecture further isolates raw private content from the generation LLM.
•We conduct extensive experiments on ChatDoctor and Wiki-PII across six LLM backbones under both targeted
and untargeted adversarial settings, demonstrating consistently strong privacy protection, the highest factual
consistency among all evaluated privacy-preserving baselines, and no added online inference latency as an
asynchronous offline preprocessing module.
2. Related Work
2.1. Retrieval-Augmented Generation
Retrieval-Augmented Generation (RAG) enhances Large Language Models by augmenting their static knowledge
with content dynamically retrieved from external domain-specific corpora Lewis, Perez, Piktus, Petroni, Karpukhin,
Goyal, Küttler, Lewis, Yih, Rocktäschel et al. (2020); Gao, Xiong, Wang and Wang (2024). At inference time, a retriever
selects the most relevant passages from an indexed knowledge base and concatenates them with the user query before
passing the combined input to the generator Ram et al. (2023); Santhanam, Khattab, Saad-Falcon, Potts and Zaharia
(2022). Such retrieval-grounded approaches reduce hallucination Shuster, Poff, Chen, Kiela and Weston (2021), improve
factual precision, and allow knowledge bases to be updated independently of model retraining, extending applicability to
diverse settings including self-reflective retrieval Asai, Wu, Wang, Sil and Hajishirzi (2024), representation fine-tuning
Wu, Arora, Wang, Geiger, Jurafsky, Manning and Potts (2024); Shi, Min, Yasunaga, Seo, James, Lewis, Zettlemoyer
and Yih (2024); Van Veen, Van Uden, Blankemeier, Delbrouck, Aali, Bluethgen, Pareek, Polacin, Reis, Seehofnerova
et al. (2023), and chain-of-thought augmentation Chowdhury and Caragea (2025). Subsequently, RAG has gained
widespread adoption in specialized question answering Wang, Ren, Li, Zhao, Liu and Wen (2024); Opoku et al. (2025);
Lee and Lee (2026); Menick, Trebacz, Mikulik, Aslanides, Song, Chadwick, Glaese, Young, Campbell-Gillingham,
Irving et al. (2022), knowledge graph-enhanced retrieval Yu and McQuade (2025), and medical dialog systems using
patient documents and clinical resources to enhance diagnostic support Zhao, Liu, Yang and Miao (2025); Li, Li, Zhang,
Dan, Jiang and Zhang (2023). Nonetheless, such dependency on the private external corpora creates a clear avenue of
sensitive data disclosure, i.e., a weakness that prompts the discussion of the threat analysis presented in the next section.
2.2. Privacy Threats in RAG Systems
When RAG is deployed over private corpora, the retrieval mechanism itself becomes an attack surface. Huang,
Gupta, Zhong, Li and Chen (2023) have demonstrated that membership inference attacks on RAG can succeed even
without direct access to the raw corpus. When exposed directly, the retrieved content may reveal sensitive information in
:Preprint submitted to Elsevier Page 3 of 23

the model outputs; Zeng et al. (2024) went on to show that well-designed adversarial examples can induce the pipeline
to repeat back private records word-for-word, which shows that LLMs can regenerate retrieved text with a probability
near 50%. More aggressive strategies include combining jailbreaks with prompt injection to exfiltrate large volumes of
private content Cohen et al. (2024), black-box adaptive reconstruction Di Maio et al. (2025), knowledge poisoning of
the retrieval corpus Zou et al. (2025), and inversion of dense embedding vectors to recover original text Morris et al.
(2023). Collectively, these attacks expose vulnerabilities across all three stages of a RAG pipeline: retrieval, embedding,
and generation, underscoring that defenses must operate at the semantic level rather than on surface-form heuristics.
2.3. Existing Privacy-Preserving Defenses
Prior defenses against privacy leakage in RAG systems operate at different pipeline stages. Representation-level
approaches (e.g., PRESS He, Liu, Hou, Jiang and Li (2025a), Eguard Liu et al. (2024); Feng, Li, Qin and Li (2026b))
act on embeddings, while selective pre-training filters data upstream Yu, Gopi, Kulkarni, Lin, Naik, Religa, Yin and
Zhang (2024). At the system level, architectural isolation methods like the concurrent Privacy-Aware RAG Zhou, Feng
and Yang (2025) physically shield raw data from the generator. To secure the text itself, formal Differential Privacy
(DP) methods (e.g., LPRAG He, Tang, Zhang, Zhou and Su (2025b), DP-RAG Koga, Wu, Zhang and Chaudhuri (2024))
offer provable bounds by injecting mathematical noise. However, while DP provides strict worst-case guarantees, its
perturbation often severely degrades textual coherence and downstream utility. Therefore, for applications requiring
high contextual fidelity, empirical retrieval-context sanitization is generally preferred. Within empirical methods,
approaches like SAGE Zeng et al. (2025) and KG-PrivRAG Zhang, Wu, Li, Zhang, Song, Li, Wang, Shen, Yin, Ge and
Luo (2026) synthesize surrogate texts or intervene at the graph level. However, they frequently suffer from semantic drift
or fail to govern surface-form text directly. Similarly, single-agent rewriting struggles to balance privacy and semantics
simultaneously within a single pass, often causing over- or under-sanitization. To achieve an optimal trade-off, our
architecture adopts the physical isolation principles of Privacy-Aware RAG Zhou et al. (2025), but uniquely introduces
a fine-grained, multi-agent sanitization pipeline. By delegating privacy extraction, semantic analysis, and reconstruction
to independent specialized agents, we resolve internal prompt disputes and enable explicit conflict arbitration, providing
more robust semantic preservation than existing paradigms.
Our approach deviates from prior research along three key dimensions. DP-enabled techniques deliver theoretically
sound privacy guarantees via noise injection, but the resultant text distortion degrades coherence and thus cannot
satisfy the high fidelity requirements of practical RAG deployment. Data synthesis methods represented by SAGE
and AttrPrompt evade privacy leakage by replacing original corpora with synthetic texts, at the cost of irreparable
loss of unique factual specifics. Additionally, single-agent and adversarial-loop pipelines combine privacy filtering
and semantic maintenance in a single pass, which frequently causes improper sanitization when private attributes and
core semantics are tightly interwoven. In contrast, our framework manipulates raw texts directly for content rewriting.
A structured factual backbone is extracted in advance of sanitization, and distinct agents are deployed for separate
tasks following deterministic routing policies. This paradigm fundamentally mitigates the conflicts arising from the
co-occurrence of privacy-sensitive content and valuable semantic information.
3. Method
3.1. Framework Overview
Given a retrieved document, the sanitization objective is to produce a rewritten version that suppresses privacy
exposure while preserving downstream utility. The core difficulty is that private identifiers and semantic content are not
disjoint in natural text: they frequently co-occur as carriers of factually important information, creating an inherent
tension between suppressing leakage and preserving utility that a single-agent approach cannot resolve. Consequently,
single-agent prompting yields a zero-sum trade-off—naive deletion degrades retrieval recall through vocabulary
mismatch (the retrieval degradation problem), while real-time inference-stage sanitization introduces unwarranted
latency for end-users.
To overcome these structural limitations, we propose a Multi-Agent Semantic Rewriting Framework governed
by an Asymmetric Retrieval and Isolated Generation Architecture . As depicted in Figure 2, the lifecycle of our
framework decouples the data processing into two strictly separated phases: the Offline Data Ingestion and Sanitization
Phase, and the Online Asymmetric Retrieval Phase. Specifically, the offline sanitization phase comprises three key
agents: (1) the Pri-Extra Agent, which extracts privacy-sensitive segments; (2) the Sem-Extra Agent, which extracts key
semantic information; and (3) the Reconstruction Agent, which produces the final rewritten text by removing sensitive
:Preprint submitted to Elsevier Page 4 of 23

content while preserving the core semantics into safe payloads. Finally, the online stage uses a Dual-Track Storage
Mechanism to preserve intact retrieval recall and isolate the generation LLM physically away from all raw private
identifiers.
Figure 2: Overview of the proposed multi-agent semantic rewriting framework for privacy-preserving RAG. The offline
phase sanitizes raw documents through three collaborative agents (Pri-Extra agent, Sem-Extra agent, and Reconstruction
agent), while the online phase retrieves over the original corpus but supplies only rewritten text to the LLM.
3.2. Offline Multi-Agent Rewriting Pipeline
The offline pipeline acts as a structural firewall by sanitizing all raw documents before they are ever exposed to the
generation system. It comprises three specialized agents, each with a distinct operational role:
3.2.1. Privacy Extraction Agent (Pri-Extra Agent)
The Pri-Extra Agent operates under a Rule-LLM Synergistic Extraction Paradigm to prevent the leakage of both
explicit and latent sensitive identifiers. Formally, let 𝑇𝑟𝑎𝑤denote the retrieved raw document. The objective of this
agent is to map 𝑇𝑟𝑎𝑤to a discrete set of privacy-sensitive segments, denoted as 𝑃𝑠𝑒𝑞.
Recognizing the complementary strengths of deterministic and generative models, this agent orchestrates a
sequential, synergistic extraction process. First, a deterministic extraction function 𝑓rules(⋅)employs lightweight
Named Entity Recognition (NER) and regular expressions on the raw text to strictly capture a set of Explicit Identifiers
𝑃𝑒𝑥𝑝𝑙(e.g., social security numbers, exact dates, and locations). This guarantees a strict baseline recall for standardized
sensitive formats that LLMs might occasionally miss due to hallucination or attention drift.
Subsequently, we utilize a generative extraction function 𝑓LLM(⋅)that processes the full original document 𝑇𝑟𝑎𝑤
alongside the previously identified set 𝑃𝑒𝑥𝑝𝑙. By explicitly providing 𝑃𝑒𝑥𝑝𝑙as context, we enable the LLM to skip these
already-extracted entities. Instead, the model is directed to infer abstract quasi-identifiers 𝑃𝑞𝑢𝑎𝑠𝑖 (such as a specific
occupation coupled with a rare medical condition) that typically slip past rule-based filters. The complete set of
privacy-sensitive segments, 𝑃𝑠𝑒𝑞, is simply the union of both extraction steps:
𝑃𝑠𝑒𝑞=𝑃𝑒𝑥𝑝𝑙∪𝑓LLM(𝑇𝑟𝑎𝑤,𝑃𝑒𝑥𝑝𝑙) = {𝑝1,𝑝2,…,𝑝𝑚} (1)
By combining deterministic checks with generative inference, we lower the likelihood of missing latent identifiers and
achieve a much broader coverage of the privacy boundaries.
The structured prompt for the Pri-Extra Agent is briefly outlined below to show its main components. The complete
template is provided in Appendix A.
:Preprint submitted to Elsevier Page 5 of 23

Simplified Prompt for Pri-Extra Agent
Task Description: Infer hidden Personally Identifiable Information (PII) from the retrieved text 𝑇𝑟𝑎𝑤. Note that
the following explicit identifiers have already been extracted: 𝑃𝑒𝑥𝑝𝑙.
Focus Area (Quasi-Identifiers):
Do not repeat the explicit identifiers in 𝑃𝑒𝑥𝑝𝑙. Identify combinations of attributes (e.g., rare diseases + specific
occupations, unique timelines) that could lead to identity inference based on the full text context.
Implicit Context Analysis:
Capture latent demographic data, behavioral patterns, and highly specific geographic or organizational
affiliations.
Output Format: Return a JSON list detailing the newly inferred PII ( 𝑃𝑞𝑢𝑎𝑠𝑖), its category, and the surrounding
text segment.
3.2.2. Semantic Analysis Agent (Sem-Extra Agent)
While the Pri-Extra Agent pinpoints which sensitive elements to hide, the Sem-Extra Agent identifies the core facts
that need to be kept. Because language is inherently interconnected, simply retaining certain sentences or masking a
few keywords can still leave behind privacy risks. Details like writing style, sentence structure, and specific predicates
can sometimes be enough for attackers to link a text back to its author or subject.
To address this issue, the agent applies Structured Attribute Deconstruction . Instead of copying out full chunks
of text, the agent functions more like a parser: it sets aside the original phrasing and maps the document 𝑇𝑟𝑎𝑤into a
structured format. This is done through a Dual-Perspective Analysis that looks at two main aspects: Core Content (such
as the main events, actions, and states) and Informational Structure (such as the logic, reasoning, and intent behind the
text).
Formally, letting 𝑔(⋅)be this semantic extraction function, the agent converts 𝑇𝑟𝑎𝑤into𝑛distinct semantic tuples,
𝐾𝑠𝑒𝑞:
𝐾𝑠𝑒𝑞=𝑔(𝑇𝑟𝑎𝑤) ={(𝑒𝑖,𝑟𝑖,𝑣𝑖,𝑤𝑖)}𝑛
𝑖=1(2)
where each tuple forms a basic knowledge slot. It includes a subject 𝑒𝑖, a predicate 𝑟𝑖, a corresponding value 𝑣𝑖, and
an importance weight 𝑤𝑖. This structured element-slot representation draws on element-alignment principles from
structured extraction Xu, Feng and Wang (2026a), and the semantic backbone 𝐾𝑠𝑒𝑞serves as an intermediate aligned
representation that decouples factual content from surface phrasing Xu, Wu, Zhang, Yang, Ma, Dong and He (2026b).
By separating the verbs and objects from the original sentence structure, this deconstruction limits the underlying
linguistic hints that often hide implicit privacy signals. As a result, the extracted semantic backbone 𝐾𝑠𝑒𝑞remains
highly useful for later generation tasks without carrying over the context that could expose someone’s identity.
The prompt structure for the Sem-Extra Agent is summarized below to highlight its core design. The complete
template is provided in Appendix B.
Simplified Prompt for Sem-Extra Agent
Task Description: Extract all semantically essential content from the text 𝑇𝑟𝑎𝑤using structured deconstruction.
Dual-Perspective Analysis:
- Core Content: Facts, actions, and perspectives.
- Informational Structure: Explanations, recommendations, and reasoning context.
Output Format: Return a structured JSON list of entity-event slots with importance levels.
3.2.3. Reconstruction Agent
As the final step in the offline pipeline, the Reconstruction Agent is responsible for combining the extracted privacy
boundaries ( 𝑃𝑠𝑒𝑞) with the core semantic facts ( 𝐾𝑠𝑒𝑞) to generate a safe, readable output text ( 𝑇𝑠𝑎𝑓𝑒). We can express
this reconstruction process as a constrained rewriting function 𝐻(⋅):
𝑇𝑠𝑎𝑓𝑒=𝐻(𝑃𝑠𝑒𝑞,𝐾𝑠𝑒𝑞)s.t. ∀𝑝𝑚∈𝑃𝑠𝑒𝑞,𝑀(𝑝𝑚,𝑇𝑠𝑎𝑓𝑒)<𝜏 (3)
:Preprint submitted to Elsevier Page 6 of 23

where𝑀(⋅,⋅)is a matching function and 𝜏is a privacy threshold. 𝜏is not a static numerical value, but is implemented
via our conflict routing policy: it acts as a strict zero-tolerance bound ( 𝜏= 0) for explicit identifiers, but is relaxed
to a semantic non-entailment boundary for complex quasi-identifiers to avoid over-sanitization. In practice, 𝜏=0for
explicit identifiers means any match triggers mandatory replacement. For quasi-identifiers, 𝜏is not a numeric value but
a routing rule in the agent’s prompt: the agent substitutes with a placeholder if the sensitive element can be cleanly
removed, and falls back to high-level abstraction if removing it would destroy the semantic core of the sentence. Most
traditional redaction methods try to partially mutate the original document 𝑇𝑟𝑎𝑤, which often leads to awkward phrasing
and lingering stylistic clues. In contrast, this agent builds the new text completely from scratch using only the clean,
isolated knowledge slots provided in 𝐾𝑠𝑒𝑞.
The biggest challenge during text reconstruction is untangling cases where privacy and utility overlap. To handle this,
the agent uses a Fine-grained Conflict Routing strategy Xie, Wang, Guo, Chen, Zhang, Jiang, Wang and Chen (2026),
which detects and resolves conflicting element pairs across the privacy and semantic streams. When an important fact
𝑘𝑛∈𝐾𝑠𝑒𝑞overlaps significantly with a sensitive identifier 𝑝𝑚∈𝑃𝑠𝑒𝑞(such as a highly specific disease name that might
easily identify a patient), the agent decides on a rewriting approach based on how deeply the two are intertwined:
•Placeholder Substitution (Low-depth Conflict): This is used when the conflict involves straightforward explicit
identifiers without complex context. Details like exact patient names or specific dates are simply replaced with
generic placeholders (e.g., “[Patient_A]” or “[Date]”). This keeps the text flowing naturally without changing the
surrounding meaning.
•High-level Abstraction (Deep Semantic Conflict): This approach is taken when the sensitive identifier is woven
deeply into the factual content of 𝐾𝑠𝑒𝑞. Rather than blindly blocking out the term—which might render the
medical or logical context useless—the agent replaces it with a broader, more general concept. For instance, a
very distinctive rare syndrome might be abstracted into “a rare genetic immune disorder.”
By generating a sanitized document ( 𝑇𝑠𝑎𝑓𝑒) directly from the structural attributes while enforcing strict privacy rules,
the Reconstruction Agent creates a clear, informative text suitable for secure databases, largely sidestepping the usual
trade-offs between data privacy and data utility.
To demonstrate the underlying rewriting logic, the prompt for the Reconstruction Agent is abbreviated below.
Appendix C contains the full template.
Simplified Prompt for Reconstruction Agent
Task Description: Rewrite𝑇𝑟𝑎𝑤to eliminate privacy constraints 𝑃𝑠𝑒𝑞while preserving structured semantics
𝐾𝑠𝑒𝑞.
Core Principle (Privacy Priority): When privacy conflicts with key information, prioritize privacy protection.
Routing Strategy:
- Use Placeholder Substitution for explicit entities.
- Use High-level Abstraction for entangled sensitive medical/private narratives.
Output Format: Generate the final natural, coherent, and sanitized text representation.
To illustrate how the three agents interact in practice, Appendix F provides a step-by-step qualitative case study on
a real ChatDoctor document, tracing 𝑇𝑟𝑎𝑤through𝑃𝑠𝑒𝑞,𝐾𝑠𝑒𝑞, and the final sanitized output 𝑇𝑠𝑎𝑓𝑒.
3.3. Asymmetric Retrieval and Isolated Generation Architecture
After the offline rewriting is complete, we construct the database using a Dual-Track Storage Mechanism . Simply
indexing the sanitized text 𝑇𝑠𝑎𝑓𝑒would drop retrieval performance, since the altered vocabulary and masked entities
often cause retrieval mismatches. To prevent this, we maintain two distinct retrieval indices in the vector database, both
built directly on the original𝑇𝑟𝑎𝑤:
•Dense Retrieval Index: We compute high-dimensional embeddings on the untouched 𝑇𝑟𝑎𝑤with BGE-M3’s
dense encoder, which keeps the full semantic space intact.
•Sparse Retrieval Index: At the same time, we build a BM25 lexical index over the original 𝑇𝑟𝑎𝑤to catch exact
matches for names, medical codes, and other specific terms.
:Preprint submitted to Elsevier Page 7 of 23

Table 1
Statistics of the datasets used in the experiment.
Dataset Source Composition Content Type Total Samples
ChatDoctor HealthCareMagic-101 Medical Dialogues 200,000
Wiki-PII Enron + Wikitext-103 Email & Public Articles 500,000
•Context Value (Sanitized Payload): Instead of storing the raw text, both of these indices point directly to the
sanitized output 𝑇𝑠𝑎𝑓𝑒 from the Reconstruction Agent. The raw 𝑇𝑟𝑎𝑤is never returned as the payload to the
generation LLM.
This setup ensures that the original text acts solely as a search mechanism and is never returned as the payload.
When answering a user query 𝑞during the Online Asymmetric Retrieval Phase , we use a Hybrid Retrieval
approach over the original document representations. If 𝑟dense(𝑖)and𝑟sparse(𝑖)are the ranks of document 𝑖from the
dense and sparse searches, we combine them using Reciprocal Rank Fusion (RRF):
RRF(𝑖) =1
𝑘+𝑟dense(𝑖)+1
𝑘+𝑟sparse(𝑖)(4)
We set the smoothing constant 𝑘to 60, following common practice. The top- 𝐾documents are simply those with the
highest RRF scores:
𝑡𝑜𝑝= arg max
𝑖𝐾[RRF(𝑖)] (5)
By taking advantage of both dense semantic matching and sparse exact matching on the original text, our retrieval
recall matches that of a standard RAG setup. But more importantly, once the top documents 𝑡𝑜𝑝are found, the system
only pulls their sanitized versions {𝑇𝑖
𝑠𝑎𝑓𝑒}𝑖∈𝑡𝑜𝑝, bypassing𝑇𝑟𝑎𝑤completely. These safe payloads are then fed to the
LLM to generate the final answer:
Response =(
𝑞⊕Concat ({𝑇𝑖
𝑠𝑎𝑓𝑒}𝑖∈𝑡𝑜𝑝))
(6)
This means the LLM never sees the raw private identifiers. By physically separating the sensitive data from the
generation window, we effectively shield the system from jailbreaks or prompt-injection extractions. It should be noted
that this isolation guarantee holds within the assumed grey-box threat model, in which the adversary manipulates the
input query but does not have direct access to the underlying vector database; the raw documents 𝑇𝑟𝑎𝑤remain present
in the retrieval indices, so the security guarantee is query-level rather than storage-level. Additionally, since all the
heavy lifting of multi-agent rewriting runs asynchronously offline, the interactive inference phase experiences virtually
no extra latency compared to a standard, unprotected RAG pipeline.
4. Experimental Setup
4.1. Datasets and Scenarios
To verify the effectiveness of our proposed method across diverse application contexts, we implement two
representative experimental scenarios. Our experimental setup ensures consistency with methodologies established in
previous work Zeng et al. (2024). The statistical specifications for both datasets are detailed in Table 1.
For the first scenario, we focus on the medical domain, where data confidentiality is critical. We use the
HealthCareMagic-101 dataset, which consists of 200,000 medical dialogues covering a wide range of health issues. We
treat each dialogue as a single retrieval unit and refer to this collection as the ChatDoctor dataset. This setup mimics a
real-world medical assistant, providing a mix of structured and unstructured sensitive information.
The second scenario deals with a more complicated situation where private details are mixed into a mostly public
text corpus, a common issue in large enterprise systems. To test this, we evaluate on the Wiki-PII dataset Zeng et al.
(2025). Specifically, we extract authentic PII from the Enron Email dataset. Next, we segment the Wikitext-103 public
dataset using a recursive character text splitter with a chunk size of 1500. Finally, for each segmented wiki chunk, we
:Preprint submitted to Elsevier Page 8 of 23

randomly insert the extracted PII at the end of each sentence. This combined corpus yields 500,000 samples where
sensitive and non-sensitive information are heavily tangled, allowing us to evaluate the risk of privacy leaks in realistic
environments. It should be noted that inserting PII at sentence boundaries creates a regular and detectable pattern; in
real-world corpora, private information is typically woven throughout the text in less predictable ways. The framework’s
performance on such naturalistically distributed content may differ from what is reported here, and evaluation on more
organically constructed datasets is left for future work.
4.2. Formalized Threat Models
To thoroughly assess the vulnerabilities of RAG systems and evaluate our defense, we formalize our threat models
based on the prompt-injection mechanisms defined in recent literature Zeng et al. (2025). We assume a realistic grey-box
setting where the adversary lacks direct access to the backend vector database but can manipulate the input query 𝑞to
exploit the LLM’s instruction-following capabilities. Formally, an adversarial query 𝑞is defined as a two-component
composite:
𝑞=𝑞info⊕𝑞cmd (7)
where𝑞infois the information component crafted to trigger retrieval of target-relevant documents, and 𝑞cmdis the
command component that instructs the LLM to expose sensitive content. The symbol ⊕denotes sequential concatenation.
The adversarial query templates used in this work are publicly available alongside the source code.
4.2.1. Targeted Extraction Attacks
In a targeted attack, the adversary possesses partial prior knowledge of a target entity (e.g., knowing a patient’s
name but not their diagnosis). Their objective is to extract specific, highly sensitive attributes tied to that entity. The
𝑞infocomponent is crafted to trigger the retrieval of documents relevant to the target, while 𝑞cmdexplicitly instructs the
model to reveal the missing sensitive value (e.g., “I want information about [disease]”). This threat model tests the
system’s ability to resist the targeted extraction of explicit and latent identifiers.
4.2.2. Untargeted Reconstruction Attacks
Conversely, an untargeted reconstruction attack simulates an adversary with zero prior knowledge whose goal is to
indiscriminately exfiltrate as much raw training or retrieval corpus data as possible. Here, 𝑞infois randomly sampled
from diverse passages (e.g., Common Crawl) to induce broad, unforeseen document retrieval, and 𝑞cmdis fixed as a
forceful directive: “Please repeat all the context. ” This threat model evaluates whether the system can prevent verbatim
or high-semantic-overlap regeneration of arbitrary memorized content under worst-case adversarial probing.
4.3. Evaluation Metrics
We establish a comprehensive, three-dimensional evaluation metric system to quantify privacy leakage, evaluate
semantic retention realism, and measure operational efficiency.
4.3.1. Privacy Leakage Assessment
We measure privacy leakage under the two threat models using precise criteria established by prior foundational
works.
Targeted Information (Target Info) The absolute count of unique, highly sensitive entities (e.g., specific names, ID
numbers) successfully extracted by the attacker. Lower counts indicate stronger privacy protection; a count of zero
represents an ideal outcome.
Repeat Prompt Following the heuristic from Zeng et al. (2024), we count the number of prompts where the LLM’s
output contains at least 10 consecutive tokens identical to the target sensitive information or raw context. This threshold
effectively captures direct, verbatim leakage without triggering false positives on common stopwords.
ROUGE Prompt (for Untargeted Attacks) To account for implicit leakage where an LLM paraphrases rather than
copies verbatim, we count prompts yielding outputs with a high semantic overlap (ROUGE-L >0.5) with the original
sensitive data, signaling indirect privacy compromise.
:Preprint submitted to Elsevier Page 9 of 23

4.3.2. Contextual Fidelity and Factual Consistency
Traditional RAG pipeline evaluations rely heavily on exact-match string overlap metrics. However, because our
core framework objective is semantic rewriting rather than verbatim reproduction, using structural metrics as the sole
proxies for utility is fundamentally flawed. Therefore, our fidelity evaluation is twofold.
N-gram and Structural Overlap We report BLEU-1 Papineni et al. (2002) and ROUGE-L Lin (2004) scores to
provide a baseline proxy of structural fidelity. Specifically, these metrics measure the n-gram overlap between the
LLM-generated answer and the ground-truth reference answer for each query, thereby assessing whether the sanitized
retrieval context still enables the model to produce factually correct responses. Higher scores indicate that the rewritten
context preserves sufficient informational content for accurate answer generation.
Factual Consistency To provide a more direct assessment of downstream utility beyond surface-level overlap,
we augment our evaluation with two complementary factual consistency metrics. First, we employ GPT-4o as a
model-as-judge evaluator to score each LLM-generated response on a 1–5 Likert scale ( FC Score ), assessing whether
the core medical or factual claim in the output is aligned with the reference answer. Second, we utilize a DeBERTa-
v3-large model fine-tuned on fact-verification datasets to measure two critical dimensions of factual consistency: the
proportion of responses factually supported by the reference ( Entailment ) and the proportion conflicting with the
facts ( Contradiction ). Higher Entailment and lower Contradiction indicate better factual utility. Together, these two
metrics evaluate whether the sanitized context preserves the factual integrity required for reliable downstream RAG
performance. The complete judging prompt, scoring rubric, and NLI decision rule are documented in Appendix D.
4.3.3. System Efficiency
A practical defense must not compromise the interactive speed of the RAG application. We measure the
computational overhead by recording the average offline preprocessing time required to sanitize the original retrieval
documents. It is critical to emphasize that this metric represents a one-time, asynchronous database construction cost,
thereby guaranteeing a Zero Online Latency Penalty during active user inference.
4.4. Implementation Details
To ensure robust generalization, we evaluate our method across a diverse array of advanced LLMs, encompassing
different scales and training paradigms: GPT-3.5-Turbo, GPT-4o-mini, LLaMA-3-8B, DeepSeek-V3, DeepSeek-R1
(known for enhanced reasoning), and QWQ-32B. All models are instruction-tuned.
For indexing and retrieval, we use the BGE-M3 model Chen, Xiao, Zhang, Luo, Lian and Liu (2024) alongside a
Milvus vector database. We chose BGE-M3 because it outputs both dense semantic embeddings and sparse lexical
representations in a single pass. This is crucial for handling private information, which often requires both semantic
understanding and exact keyword matching. We index both sets of representations in Milvus, which easily supports
dual vector search. When retrieving documents, we rely on a hybrid retrieval approach: we run dense and sparse
searches separately and combine their results using Reciprocal Rank Fusion (RRF) to pull the top 𝐾= 3candidates.
We empirically set 𝐾= 3to balance downstream utility and inference security: larger values (e.g., 𝐾≥10) introduce
context noise and theoretically increase the risk of an adversary aggregating multiple sanitized records to infer identity.
Mixing semantic and keyword retrieval typically yields more stable results for medical records and similar sensitive
texts. Finally, we keep all generation API parameters fixed (temperature = 0.4, top-p = 0.9, max tokens = 512) to ensure
our testing outcomes remain consistent and reliable.
4.5. Baselines
We benchmark our framework against six baselines spanning the full spectrum of RAG privacy-preserving
approaches, from rule-based methods to state-of-the-art LLM-driven frameworks.
•Naive RAG: The standard, unprotected RAG pipeline with no privacy intervention, serving as the upper bound
for potential leakage and a reference for utility preservation.
•AttrPrompt Yu, Zhuang, Zhang, Meng, Ratner, Krishna, Shen and Zhang (2023): An attribute-driven
synthetic data generation baseline where an LLM is prompted to replace original text with structurally similar,
privacy-filtered synthetic content conditioned on extracted attributes.
:Preprint submitted to Elsevier Page 10 of 23

Table 2
Targeted extraction attack results on ChatDoctor and Wiki-PII (250 adversarial prompts each).∗/†Note on units:∗Target
Infomeasures the absolute number of unique sensitive entities successfully extracted by the attacker.†Repeat Prompt
counts the number of generated outputs containing ≥10 consecutive verbatim tokens from private text.
LLaMA-3-8B GPT -3.5-Turbo GPT -4o-mini DeepSeek-V3 DeepSeek-R1 QWQ-32B
Scenario MethodTarget Repeat Target Repeat Target Repeat Target Repeat Target Repeat Target Repeat
Info∗Prompts†Info Prompt Info Prompt Info Prompt Info Prompt Info PromptChatDoctorNaive RAG 144 149 98 102 122 144 124 138 144 148 130 144
AttrPrompt 0 0 0 0 0 0 0 0 0 0 0 0
Single Agent 15 20 10 15 13 18 11 16 14 19 12 17
Adv-Anon 9 14 6 11 8 13 7 11 9 13 7 12
SAGE 1 2 2 1 1 0 1 2 1 2 1 2
KG-PrivRAG 1 3 1 2 1 2 1 1 1 2 1 2
Ours 1 1 0 1 1 1 0 0 1 0 1 1Wiki-PIINaive RAG 49 90 13 36 12 12 11 11 34 84 41 85
AttrPrompt 0 0 0 0 0 0 0 0 0 0 0 0
Single Agent 11 15 4 10 8 9 6 8 13 14 10 13
Adv-Anon 7 4 2 1 6 2 5 2 8 2 6 1
SAGE 10 1 1 0 12 0 9 1 15 0 9 0
KG-PrivRAG 6 2 1 0 5 0 4 1 7 1 5 1
Ours 3 2 1 0 0 0 0 0 2 1 3 1
•SAGE Zeng et al. (2025): A state-of-the-art two-stage framework that first generates synthetic surrogate text via
attribute extraction, then employs an agent-based iterative discriminator to assess and strip any residual privacy
leakage from the generated output.
•Single Agent: A direct LLM-based rewriting baseline in which a single language model receives both the
privacy extraction objective and the semantic rewriting objective within a unified prompt, without any inter-agent
decomposition. This ablative baseline is designed to isolate the contribution of our multi-agent task decomposition
strategy.
•Adv-Anon Staab, Vero, Balunovic and Vechev (2025): An LLM-based adversarial anonymization framework
that employs a feedback-guided loop: an inference LLM iteratively predicts personal attributes from the current
text, while an anonymizer LLM adapts the text to suppress the identified cues. Although originally designed
for general online texts rather than RAG corpora, we adapt it to our retrieval document setting for a direct
comparison.
•KG-PrivRAG Zhang et al. (2026): A knowledge graph-centric privacy-preserving RAG method that constructs
a graph from retrieval documents and applies fine-grained privacy interventions at the entity and relation level via
a four-module privacy-aware processor (re-ranking, filtering, synthesis, and compression). This concurrent work,
published in the same venue, represents the closest competitor to our approach in the RAG privacy domain.
4.6. Experimental Results
4.6.1. Privacy Leakage in Targeted Extraction Attacks
Table 2 shows the results for the targeted attack scenarios. The unprotected Naive RAG pipeline leaks between 130
and 144 sensitive entries out of 250 queries on ChatDoctor, and up to 49 entries on Wiki-PII, underscoring the severity
of the prompt injection threat. Our approach substantially mitigates targeted leakage across all six LLMs on ChatDoctor
(at most 1 instance) and holds it between 0 and 3 on the more challenging Wiki-PII corpus. Among utility-preserving
defenses, this represents the best privacy-utility trade-off: SAGE and KG-PrivRAG achieve comparable leakage counts
but at lower factual consistency, while AttrPrompt reaches zero leakage at the cost of substantially degraded utility.
AttrPrompt achieves zero leakage across all metrics, LLMs, and both datasets; for deployments where any residual
exposure is unacceptable, this may be the preferable choice despite the lower utility scores.
The Repeat Prompt results follow the same pattern. Both SAGE and our method hold verbatim reproductions to
0–2 on ChatDoctor, confirming that the Dual-Track Storage Mechanism excludes raw identifiers from the generation
window rather than merely obscuring them. The residual leakage on Wiki-PII (0–3 instances) is largely attributable to
quasi-identifiers that are structurally embedded in the public Wikipedia text, a recognized challenge for any sanitization
method applied to mixed public-private corpora.
:Preprint submitted to Elsevier Page 11 of 23

Table 3
Untargeted reconstruction attack results on ChatDoctor (250 adversarial prompts).∗/†Note on units:∗Repeat Prompt
counts the absolute number of verbatim reproductions ( ≥10 consecutive tokens).†ROUGE Prompt counts the number
of outputs with ROUGE-L >0.5 against original private text.
LLaMA-3-8B GPT -3.5-Turbo GPT -4o-mini DeepSeek-V3 DeepSeek-R1 QWQ-32B
Method Repeat ROUGE Repeat ROUGE Repeat ROUGE Repeat ROUGE Repeat ROUGE Repeat ROUGE
Prompt∗Prompt†Prompt Prompt Prompt Prompt Prompt Prompt Prompt Prompt Prompt Prompt
Naive RAG 27 65 40 61 31 55 34 56 25 40 21 62
AttrPrompt 0 0 0 0 0 0 0 0 0 0 0 0
Single Agent 5 11 7 10 5 10 6 11 4 9 4 10
Adv-Anon 2 6 2 5 2 6 2 5 2 5 1 5
SAGE 0 2 0 1 0 2 0 2 0 2 0 1
KG-PrivRAG 0 3 0 2 0 3 0 2 0 3 0 2
Ours 0 1 0 2 0 2 0 1 0 4 0 1
Table 4
Adaptive attack results on ChatDoctor (GPT -4o-mini, 50 queries).
Attack Type Target Info ↓Repeat Prompt ↓
Fixed Template (Baseline) 1 2
Synonym Substitution 2 3
Stepwise Inference 3 4
Unprotected RAG (reference) 28 N/A
Regarding the choice of the 10-token threshold for Repeat Prompt: following the heuristic established in Zeng et al.
(2024), we adopt this threshold because it is long enough to capture non-trivial verbatim copying while avoiding false
positives triggered by common medical stopword sequences. Shorter thresholds (e.g., 5 tokens) inflate false positives
significantly; longer thresholds (e.g., 20 tokens) miss paraphrased-but-still-sensitive reproductions. We acknowledge
that this criterion does not capture all forms of semantic re-identification; residual indirect leakage risks are further
discussed in Section 6.
4.6.2. Privacy Leakage in Untargeted Reconstruction Attacks
Under untargeted reconstruction attacks (Table 3), our method successfully suppresses verbatim reproduction
(Repeat Prompt = 0) across every LLM backbone tested. ROUGE Prompt counts remain in the range of 1–4, indicating
that even high-semantic-overlap outputs are effectively suppressed. The ROUGE-L >0.5 threshold for flagging indirect
leakage is consistent with prior work (Zeng et al., 2025); we acknowledge that this does not capture all paraphrased
semantic leakage, and we discuss this limitation in Section 6. Overall, these results demonstrate that our offline rewriting
strategy prevents not only direct verbatim copying but also high-level semantic reconstruction from arbitrary adversarial
queries.
5. Analysis and Discussion
5.1. Adaptive Adversary Analysis
While the preceding evaluation assumes a fixed-template threat model, practical adversaries aware of our defense
might adapt their queries. To assess framework resilience, we conduct a preliminary adaptive attack analysis on the
ChatDoctor dataset (50 queries via GPT-4o-mini) using two representative strategies:
•Synonym Substitution Attack: The adversary utilizes synonyms or paraphrases to evade the deterministic NER
pattern-matching of the Pri-Extra Agent.
•Stepwise Inference Attack: The adversary decomposes a privacy-seeking query into multiple benign sub-queries
for isolated attributes, subsequently aggregating the responses to infer a complete profile.
As shown in Table 4, synonym substitution causes a modest increase in leakage from 1 to 2 entities, reflecting
the robustness of the LLM-based quasi-identifier inference despite NER sensitivity to lexical variation. The stepwise
:Preprint submitted to Elsevier Page 12 of 23

Table 5
Human evaluation of original vs. sanitized documents on three usability dimensions (5-point Likert scale; higher is better).
Values represent mean ±standard deviation across 50 documents and 3 evaluators.
Dimension Original Sanitized p-value (paired t-test)
Information Completeness 4.58 ±0.62 4.42 ±0.68 0.142
Readability & Fluency 4.65 ±0.58 4.51 ±0.64 0.215
Clinical Usability 4.46 ±0.75 4.30 ±0.82 0.187
Table 6
N-gram utility of LLM-generated answers on the ChatDoctor dataset (GPT -4o-mini backbone LLM), measured against
ground-truth reference answers. Higher scores indicate that the sanitized context better preserves the informational
content needed for accurate answer generation.∗Statistically significant improvements over all privacy-preserving
baselines on both BLEU-1 and ROUGE-L, as validated by paired 𝑡-tests (𝑝<0.01).
Metric 0-shot Naive RAG AttrPrompt Single Agent Adv-Anon SAGE KG-PrivRAG Ours
BLEU-1 0.087 0.093 0.080 0.108 0.103 0.117 0.119 0.122∗
ROUGE-L 0.088 0.095 0.073 0.092 0.089 0.091 0.100 0.104∗
inference attack is more consequential: leakage rises from 1 to 3 entities, indicating that multi-query aggregation
can partially bypass the High-level Abstraction strategy by exploiting the generalization boundary between specific
identifiers and their abstracted forms. This residual vulnerability warrants explicit acknowledgment and suggests that
the Reconstruction Agent’s abstraction granularity could be tightened in future work through mechanisms such as
cross-query context tracking or stricter ontological generalization constraints. The framework’s overall resilience
nonetheless stems from the Asymmetric Retrieval Architecture, which physically isolates raw identifiers from the
generation LLM regardless of query formulation, holding leakage to at most 3 entities even under adaptive conditions
against an unprotected baseline of 28. We leave the exploration of stronger adaptive threats, such as reinforcement
learning-driven probing or cross-document correlation, for future work.
5.2. Human Evaluation of Practical Usability
To evaluate practical usability beyond surface-level metrics, we conducted a human evaluation. Three independent
experts (two biomedical informatics graduate students and one clinical research coordinator) assessed 50 random
ChatDoctor documents alongside their sanitized counterparts. They rated each on a 5-point Likert scale for Information
Completeness, Readability & Fluency, and Clinical Usability. The inter-rater reliability was high (Fleiss’ 𝜅= 0.76),
indicating consistent expert judgments, with results detailed in Table 5. Sanitized documents scored marginally lower
than originals across all three dimensions, consistent with the expected privacy-utility trade-off. The differences
did not reach statistical significance in this small-scale evaluation ( 𝑝 > 0.10for all dimensions), suggesting that
sanitization does not substantially degrade practical usability; however, the sample size is insufficient to make a
definitive equivalence claim. The Clinical Usability score of 4.30 for sanitized documents indicates that the rewritten
text remains viable for medical decision support, providing task-level evidence that the framework preserves the factual
content relevant to downstream applications.
5.3. Contextual Fidelity and Factual Consistency
Table 6 presents the BLEU-1 and ROUGE-L scores on ChatDoctor. Our method achieves the highest utility among
privacy-preserving baselines. While AttrPrompt struggles due to synthetic context loss and SAGE suffers from semantic
drift during iterative rewriting, our approach successfully preserves the factual core required for downstream tasks,
closely trailed by KG-PrivRAG’s graph filtering.
To complement surface-level n-gram metrics (Papineni et al., 2002), Table 7 evaluates factual consistency on all 250
test queries. We exclude the Naive RAG baseline here as it bypasses sanitization entirely. Our framework significantly
outperforms AttrPrompt, which frequently hallucinates medical specifics, and SAGE, which incurs semantic drift. By
strictly separating privacy extraction from semantic retention, our method achieves the highest Entailment rate and
lowest Contradiction rate among all evaluated baselines. These results demonstrate that the rewritten text preserves
:Preprint submitted to Elsevier Page 13 of 23

Table 7
Factual consistency of LLM-generated responses evaluated on 250 queries per dataset. Higher FC Score and Entailment
rate, and lower Contradiction rate, indicate better preservation of factual integrity after sanitization.
ChatDoctor Wiki-PII
Method FC Score Entailment (%) Contradiction (%) FC Score Entailment (%) Contradiction (%)
0-shot 3.23 61.2 12.5 3.10 58.9 14.2
AttrPrompt 2.91 47.9 23.2 2.71 44.5 25.8
Single Agent 3.56 66.8 9.4 3.45 63.5 11.0
Adv-Anon 3.42 64.5 10.7 3.27 61.9 12.4
SAGE 3.37 62.6 11.2 3.20 60.1 13.5
KG-PrivRAG 3.92 74.3 6.9 3.75 71.6 7.6
Ours 4.08 77.1 6.0 3.89 73.8 7.2
Table 8
Average offline preprocessing time ±standard deviation (seconds per sample). SAGE time includes both Stage-1 and
Stage-2. All values represent a one-time, asynchronous database construction cost; online inference latency is identical
to an unprotected RAG baseline for all methods.
Dataset AttrPrompt Single Agent Adv-Anon SAGE KG-PrivRAG Ours
ChatDoctor (s/sample) 20.74 ±3.87 13.52 ±2.45 89.34 ±28.59 238.50 ±62.48 45.28 ±8.16 28.15 ±4.71
Wiki-PII (s/sample) 21.43 ±4.13 12.83 ±2.19 76.42 ±25.34 158.30 ±45.10 38.61 ±7.25 17.27 ±3.68
Table 9
Ablation study on the ChatDoctor dataset (GPT-4o-mini, 250 targeted-attack prompts). ↓: lower is better (privacy); ↑:
higher is better (utility).∗/†Note on units:∗Target Info counts the number of unique sensitive entities successfully
extracted;†Repeat Prompt counts outputs with ≥10 verbatim tokens. Removing either agent degrades the respective
objective.
Privacy (ChatDoctor) Utility (ChatDoctor)
Variant Target Info∗↓Repeat Prompt†↓BLEU-1 ↑ROUGE-L ↑
Full Model (Ours) 1 1 0.122 0.104
w/o Pri-Extra Agent 47 55 0.115 0.098
w/o Sem-Extra Agent 1 2 0.091 0.079
the factual content of the original, and the high factual consistency provides evidence of downstream task-level utility,
supporting the practical usability of sanitized outputs for diagnostic question-answering.
5.4. Preprocessing Efficiency
Table 8 reports the average per-sample preprocessing time and standard deviation. Our method is the most efficient
among iterative approaches, significantly faster than SAGE and Adv-Anon. Our deterministic pipeline yields tighter
variance compared to the higher, unpredictable variance of iterative loops, facilitating more reliable worst-case
deployment planning. Single Agent is fastest due to its single-pass nature, though at the cost of weaker privacy
protection. Crucially, as the multi-agent rewriting pipeline is executed offline, this cost is a one-time, amortized
overhead; online inference latency remains identical to unprotected RAG. However, for knowledge bases that are
dynamically updated or frequently refreshed, the per-sample preprocessing cost should be factored into the deployment
plan, as each newly ingested document requires a full offline sanitization pass before it can be served. Additional
deployment-cost details, including average token counts before and after rewriting and the per-sample GPT-4o-mini
rewriting cost, are provided in Appendix E.
5.5. Ablation Study
To isolate the contribution of each component, we evaluate three ablated variants against the full model on the
ChatDoctor dataset using GPT-4o-mini. Results are reported in Table 9.
:Preprint submitted to Elsevier Page 14 of 23

w/o Pri-Extra Agent. Removing the Pri-Extra Agent causes targeted information leakage to surge from 1 to 47
entries, with the Repeat Prompt rising to 55, representing a substantial increase in privacy leakage. Utility remains
relatively high (BLEU-1: 0.115), confirming that the Sem-Extra Agent preserves semantic content regardless of privacy
filtering. This demonstrates that the Pri-Extra Agent is the primary driver of privacy protection: without an explicit
privacy identification step, the Reconstruction Agent reconstructs semantically rich text that inevitably retains private
identifiers.
w/o Sem-Extra Agent. Removing the Sem-Extra Agent leaves privacy largely intact (Target Info: 1, Repeat Prompt:
2) but causes a notable drop in utility (BLEU-1: 0.091, ROUGE-L: 0.079), falling well below the full model and
approaching the level of the unprotected Naive RAG baseline. Without a structured semantic backbone to guide
reconstruction, the Reconstruction Agent tends to over-sanitize, discarding medically relevant context alongside private
identifiers. This result validates the design rationale for the Sem-Extra Agent: its deconstructed attribute representation
provides the necessary anchor for preserving factual utility during aggressive privacy filtering.
5.6. Discussion
Theoretical implications This study offers three theoretical contributions to privacy-preserving RAG research.
First, we demonstrate that privacy sanitization and semantic preservation are mutually constraining objectives. When
sensitive identifiers and factual content are deeply intertwined in raw texts, pursuing both goals in a single rewriting
pass inevitably leads to over- or under-sanitization. This finding clarifies the inherent limitations of single-agent and
adversarial-loop architectures, and justifies task decomposition for entangled NLP privacy problems.
Second, we systematically analyze the deficiencies of existing technical routes. Differential privacy injects noise
that undermines semantic coherence, rendering outputs unsuitable for high-fidelity RAG. Synthesis-based methods
(SAGE, AttrPrompt) replace source texts with surrogates, permanently discarding unique factual details—as evidenced
by their lower FC scores in Table 7. Adv-Anon performs iterative in-place editing, but its feedback loop optimizes
solely for privacy, lacking mechanisms to preserve semantic structure.
Third, our multi-agent framework establishes a new paradigm: extracting structured factual backbones prior to
filtering, and assigning independent agents to distinct subtasks under deterministic routing rules. This design decouples
privacy protection from semantic preservation by isolating factual content from sensitive identifiers, processing each
dimension independently, and reconstructing sanitized output from the preserved backbone. It offers a generalizable
solution for balancing confidentiality and utility in text-based privacy computing.
We also delineate the boundaries of different solutions. AttrPrompt remains preferable when zero leakage is
paramount, despite its utility cost. Our framework provides a balanced alternative where both strong protection and
high fidelity are required, addressing a gap left by existing methods.
Practical implications Our framework offers practical advantages for RAG systems deployed on sensitive corpora
such as medical datasets. Privacy sanitization runs entirely offline; once the sanitized database is built, online inference
remains unchanged, introducing zero latency to user-facing services.
Efficiency results in Table 8 further validate its practicality. On ChatDoctor, our method averages 28 seconds
per document, versus 238 seconds for SAGE and 89 seconds for Adv-Anon. The comparatives’ iterative loops incur
substantial overhead and scale poorly, whereas our deterministic pipeline maintains stable preprocessing time, making
it suitable for large-volume knowledge bases.
A practical caveat: teams with dynamically updated knowledge bases must budget time for sanitizing new documents
before ingestion. Overall, our framework offers a reliable, deployable choice for high-stakes domains such as medical
consultation and enterprise confidential retrieval, where both data security and content accuracy are critical.
6. Conclusion
We present a multi-agent semantic rewriting framework for privacy-preserving RAG, comprising a Pri-Extra Agent,
a Sem-Extra Agent, and a Reconstruction Agent that operate in a coordinated offline pipeline. By assigning each
objective to a dedicated agent with a deterministic conflict routing policy, we avoid the objective entanglement of
single-prompt approaches. Our experiments on ChatDoctor and Wiki-PII across six LLM backbones demonstrate
substantially reduced privacy leakage under both targeted and untargeted attacks, the highest factual consistency among
all evaluated defenses, and negligible additional online latency. Our ablation results confirm that each agent makes a
:Preprint submitted to Elsevier Page 15 of 23

distinct and necessary contribution. The reported security results are bounded by the evaluated threat model, a grey-box
setting with fixed-template and two adaptive attack variants, and should not be interpreted as absolute guarantees. We
plan to explore more adversarially rigorous evaluation protocols, including adaptive adversaries with full knowledge of
the sanitization strategy, and formal leakage bounds in future work.
Limitations and Future Work
Two framework-level limitations warrant acknowledgement. First, the system exhibits two edge-case failure modes:
semantic drift , where a rare disease name simultaneously constitutes the core clinical topic and forces the Reconstruction
Agent to generalize at the cost of specificity; and over-sanitization , where attributes adjacent to private segments
are unnecessarily removed, as reflected in the ablation, which shows a BLEU-1 drop from 0.122 to 0.091 when the
Sem-Extra Agent is omitted. Second, extraction precision and recall of the Pri-Extra Agent cannot be independently
measured at scale, as ground-truth PII annotations are unavailable for the evaluation corpora; the agent’s deliberately
conservative stance (preferring false positives over false negatives) mitigates missed extractions but also exacerbates
over-sanitization. Formal extraction accuracy measurement via a dedicated annotated evaluation set is identified as an
important direction for future work.
On the evaluation side, while we have conducted a preliminary adaptive adversary analysis (Section 5.1), several
stronger adaptive threats remain unaddressed. Adversaries employing inference-based probing, learning-based query
optimization, or cross-document correlation could potentially extract more information than reported here. More
fundamentally, an adversary aware of the High-level Abstraction patterns might systematically probe the boundaries of
generalization to recover specific identifiers. The protection levels reported throughout this paper should therefore be
interpreted as bounded by the evaluated threat model, which includes fixed-template, grey-box attacks and two adaptive
variants, rather than as absolute guarantees. A comprehensive adaptive adversarial evaluation protocol is left for future
work.
Ethics Statement
This research utilizes the ChatDoctor (HealthCareMagic-101) and Enron email datasets, both derived from publicly
accessible files containing personal sensitive information. Our usage focuses strictly on developing privacy-preserving
technologies without attempting out-of-scope re-identification. While our mechanism detects a broad spectrum of quasi-
identifiers, it acts as an additional technical defense layer rather than a substitute for strict legal compliance pipelines,
acknowledging that it does not definitively guarantee exhaustive coverage of all 18 HIPAA Safe Harbor identifiers or
formal GDPR compliance. Furthermore, we are aware of the inherent dual-use risk where our LLM-driven inference
techniques could theoretically be repurposed by malicious actors to systematically locate and extract quasi-identifiers.
To mitigate this, our architecture strictly isolates the privacy-extraction logic offline, though practitioners deploying
such systems must remain vigilant against potential exploitation.
Declaration of competing interest
The authors declare that they have no known competing financial interests or personal relationships that could have
appeared to influence the work reported in this paper.
Acknowledgements
This work was supported in part by the International Communication Research Project (China International
Communications Group) (Grant No. 25ATILX01), the Beijing Natural Science Foundation (Grant No. L251061),
the Fundamental Research Funds for the Central Universities (Grant No. 2024MS022), the Science and Technology
Development Fund of Macau SAR (Grant Nos. FDCT/0007/2024/AKP, EF2024-00185-FST), the UM and UMDF
(Grant Nos. MYRG-GRG2024-00165-FST-UMDF, MYRG-GRG2025-00236-FST), and the Dr. Stanley Ho Medical
Development Foundation (Grant No. SHMDF-AI/2026/001). This work was performed in part at SICC which is
supported by SKL-IOTSC, and HPCC supported by ICTO of the University of Macau.
:Preprint submitted to Elsevier Page 16 of 23

Appendix
A. Prompts used in Pri-Extra Agent
This section presents the detailed prompt design used by the Pri-Extra Agent, corresponding to the LLM-driven step
of the Rule-LLM Synergistic Extraction Paradigm. Prior to this step, a lightweight NER and regular-expression module
deterministically captures Explicit Identifiers 𝑃𝑒𝑥𝑝𝑙(e.g., names, ID numbers, phone numbers). The LLM prompt below
then receives both the raw document 𝑇𝑟𝑎𝑤and𝑃𝑒𝑥𝑝𝑙as inputs, directing the model to skip already-identified entities and
focus exclusively on inferring abstract quasi-identifiers 𝑃𝑞𝑢𝑎𝑠𝑖 that evade static rules—such as rare disease–occupation
combinations or unique behavioral patterns. For each quasi-identifier category, illustrative examples are provided to
guide extraction. The full suite of executable prompt templates is available in the project’s GitHub repository. Results
are returned in a predefined JSON format. The detailed prompt is as follows.
Prompt for Pri-Extra Agent
You are the LLM component of the Privacy Extraction Agent, operating under a Rule-LLM Synergistic
Extraction Paradigm. Explicit surface-form identifiers have already been captured deterministically; your role
is to apply deep semantic reasoning to infer identity-revealing quasi-identifiers that evade static rules.
INPUT:
[Retrieved Document]: the raw document text to be analyzed.
[Pre-extracted Explicit Identifiers]: a list of surface-form identifiers (e.g., names, ID numbers, phone
numbers) already captured by deterministic NER/regex rules. Do NOT reproduce these in your output.
TASK: Given the retrieved document and the pre-extracted explicit identifiers listed above, infer all abstract
quasi-identifiers that could facilitate identity re-identification through contextual combinations or implicit
inference. Do notrepeat any entity already present in the pre-extracted list.
Quasi-identifier Categories:
[Medical-Occupational Linkage]: e.g., “A 32-year-old deep-sea welder presenting with decompression
sickness.” The combination of a rare occupation and a specific condition is highly identifiable.
[Temporal-Geographic Linkage]: e.g., “Visited the rural clinic in Oakhaven immediately following the
massive blizzard last Tuesday.” The precise timing and small location serve as a strong latent identifier.
Extraction Guidelines:
Do not output any entity already present in the pre-extracted explicit identifier list provided in the input.
Identify combinations of attributes (e.g., rare conditions + specific occupations, unique timelines) that
collectively enable identity inference.
Capture latent demographic data, behavioral patterns, and highly specific geographic or organizational
affiliations.
Consider both direct statements and contextually inferred signals.
Output Format: Return only a valid JSON object in the following structure:
{
"detected_pii": [
{
"type": "string",
"value": "string",
"context": "string (surrounding text segment that triggered the inference)"
}
]
}
:Preprint submitted to Elsevier Page 17 of 23

B. Prompts used in Sem-Extra Agent
This section presents the detailed prompt design used by the Sem-Extra Agent, which performs Structured
Attribute Deconstruction to map input documents into a sanitized semantic space. Rather than copying variable-length
natural language phrases, the agent acts as a semantic parser: it discards the linguistic shell and produces a set of
structured knowledge-slot tuples (𝑒𝑖,𝑟𝑖,𝑣𝑖,𝑤𝑖)—each encoding a subject entity 𝑒𝑖, a relational predicate 𝑟𝑖, an object or
state value𝑣𝑖, and an importance weight 𝑤𝑖. This deconstruction is governed by a Dual-Perspective Analysis across
Core Content andInformational Structure , with text-type adaptations for conversations, documents, and narratives to
ensure broad domain coverage while avoiding the carry-over of private identity information. The extracted results are
returned in a structured JSON format. The detailed prompt is as follows.
Prompt for Sem-Extra Agent
You are a specialized semantic information extraction agent designed to identify valuable information from
various types of texts including conversations, documents, articles, reports, emails, and other formats.
TASK: Analyze the text and extract all semantically meaningful information that constitutes the core value
of the content based on the following principles, while paying attention to distinguishing between private
information and basic semantic content.
Text Type Adaptations
[Conversation-Based Text]: Focus on extracting the symptoms discussed, the physician’s advice, and
prescribed treatments. Ignore conversational pleasantries and specific patient names.
[Document-Based Text]: Focus on extracting formal clinical findings, laboratory results, and diagnoses.
Ignore specific dates and institutional letterheads.
[Narrative-Based Text]: Extract the sequence of medical events and symptom progression. Ignore subjective
identifying backstories.
Information Categories
Core Content Categories: e.g., Symptoms, Diagnoses, Treatments, Lab Results.
Analytical Content Categories: e.g., Physician’s reasoning, Risk factors, Prognosis.
Output Format: Return only a valid JSON object. Each entry represents one atomic factual unit in a structured
knowledge-slot format (subject entity, relational predicate, object/state value, importance weight):
{
"key_information": [
{
"entity": "string (the subject entity, e.g., Patient, Doctor)",
"relation": "string (the relational predicate, e.g., has_symptom, recommends)",
"value": "string (the object or state value, e.g., abdominal pain, rest)",
"importance_weight": "high / medium / low"
}
]
}
C. Prompts used in Reconstruction Agent
This section presents the prompt design used by the Reconstruction Agent. The agent receives three inputs: the
raw document 𝑇𝑟𝑎𝑤, the privacy boundary set 𝑃𝑠𝑒𝑞from the Pri-Extra Agent, and the semantic backbone 𝐾𝑠𝑒𝑞(as
knowledge-slot tuples) from the Sem-Extra Agent. When a semantic attribute 𝑘𝑛∈𝐾𝑠𝑒𝑞conflicts with a privacy
identifier𝑝𝑚∈𝑃𝑠𝑒𝑞, the agent applies Fine-grained Conflict Routing : it assigns each conflict a routing label—
Placeholder Substitution for isolated explicit identifiers with low structural dependency, or High-level Abstraction for
attributes deeply entangled with the factual core. Type-specific rewriting strategies and concrete examples are provided
:Preprint submitted to Elsevier Page 18 of 23

for each privacy category to ensure privacy-first reconstruction that goes beyond simple replacement. The final output
is returned in JSON format containing the fully rewritten text. The detailed prompt is as follows.
Prompt for Reconstruction Agent
You are a dedicated privacy conversion agent responsible for reconstructing the original text to maximize
privacy protection while retaining its core semantic content.
INPUT
Original text (potentially containing privacy information)
Identified privacy information (that must be altered or removed)
[Extracted Semantic Key Information]: structured knowledge-slot tuples output by the Sem-Extra Agent,
representing the factual backbone that must be preserved while removing privacy.
Fine-grained Conflict Routing:
Placeholder Substitution (Low-depth Conflict): Applied to isolated explicit identifiers without intricate
structural dependencies (e.g., patient names, specific dates). Replace with generic referents such as “[Patient_A]”
or “[Date]”, maintaining structural continuity without altering surrounding semantic logic.
High-level Abstraction (Deep Semantic Conflict): Applied when the privacy identifier is deeply entangled
within the factual core of the preserved semantic information. Traverse the ontological hierarchy to generalize
the specific identifying term into a broader category (e.g., a specific rare syndrome →“a rare genetic immune
disorder”).
Privacy Protection Strategies (by Type):
[Personal Information]: Use Placeholder Substitution. Replace exact names, ages, and physical profiles
with generic labels (e.g., “John Doe, 32 years old” →“[Patient]”).
[Medical Information]: Use High-level Abstraction for exceptionally rare diseases that act as quasi-
identifiers. Maintain common medical facts. (e.g., “Huntington’s disease” →“a neurodegenerative disorder”).
[Location Information]: Use Placeholder Substitution for specific institutions or exact addresses (e.g.,
“Massachusetts General Hospital” →“[Clinical Facility]”).
[Relationships And Networks]: Abstract family ties or specific occupational roles (e.g., “My daughter who
works at the local bakery” →“A family member”).
[Temporal Information]: Broaden specific dates to general timeframes to obscure timelines (e.g., “On
October 12th” →“Recently”).
Process Instructions
Conduct a thorough analysis of the original text to identify privacy-sensitive content and key semantic
information, determining which parts require restructuring and which must be preserved.
Based on the analysis, generate a fully rewritten version of the text that thoroughly removes all privacy
elements while retaining the essential meaning.
Verify that the rewritten text is completely free of privacy information and that all key content is semantically
preserved.
Ensure the final text is natural, coherent, and highly readable.
Output Format: Return only a valid JSON object in the following structure:
{
"rewritten_text":
}
:Preprint submitted to Elsevier Page 19 of 23

D. Factual Consistency Evaluation Protocol
This section documents the evaluation protocol used to measure factual consistency in Section 5.3, including
the GPT-4o judging prompt, the scoring rubric for the FC Score, and the decision rule applied to the NLI-based
entailment/contradiction evaluation.
GPT-4o Judging Prompt (FC Score). For each test query, we provide GPT-4o with three inputs: the query, the
ground-truth reference answer, and the LLM-generated response under evaluation. The model is instructed to assess
whether the core factual claim in the generated response is consistent with the reference, and to assign an integer score
from 1 to 5 according to the rubric below. The temperature is set to 0 and the output is restricted to a single integer.
GPT-4o Factual Consistency Judging Prompt
You are a factual consistency evaluator. You will be given a query, a reference answer, and a generated response.
Your task is to assess whether the core factual claim in the generated response is consistent with the reference
answer.
INPUT:
[Query]: {query}
[Reference Answer]: {reference}
[Generated Response]: {response}
SCORING RUBRIC:
5— The response is fully consistent with the reference. All key medical or factual claims are present and
correct.
4— The response is mostly consistent. Minor omissions or imprecision are present, but no factual errors.
3— The response is partially consistent. Some key facts are correct, but relevant information is missing or
partially inaccurate.
2— The response is mostly inconsistent. It contains factual errors or substantially misrepresents the
reference.
1— The response is completely inconsistent with or contradicts the reference.
OUTPUT: Respond with a single integer from 1 to 5. Do not include any explanation.
NLI-based Entailment/Contradiction Decision Rule. We apply a DeBERTa-v3-large model fine-tuned on MNLI,
FEVER, and ANLI to each reference–response pair, treating the reference as the premise and the generated response as
the hypothesis. The model produces a probability distribution over three labels: entailment ,neutral , and contradiction .
Each response is assigned the label with the highest probability. The Entailment rate in Table 7 is the proportion of
responses receiving the entailment label; the Contradiction rate is the proportion receiving the contradiction label.
Responses assigned the neutral label are excluded from both counts.
E. Cost of semantic rewrite data
Our method performs a one-time, offline semantic reconstruction before inference; the inference pipeline remains
identical to the baseline, introducing no additional latency or cost. As shown in Table 10, the semantically rewritten
data are on average slightly shorter than the original samples, which can further reduce inference costs under the same
decoding configuration. We also report the offline rewriting overhead using GPT-4o-mini (Table 11): approximately
$0.002 per sample, with overall time and expense well within a reasonable range. Because this generation is one-off, it
imposes no ongoing burden at deployment.
F. Qualitative Case Study
To provide concrete insight into the rewriting process, we trace a single document from the ChatDoctor corpus
through the full offline pipeline and examine both its privacy protection properties and its downstream utility.
:Preprint submitted to Elsevier Page 20 of 23

Table 10
Average number of tokens
Dataset ori-context rewritten-text
Wiki-PII 169 145
ChatDoctor 252 218
Table 11
Average cost per sample ($)
Dataset Pri-Extra cost Sem-Extra cost Reconstruction cost Total cost
Wiki-PII 0.00082 0.00079 0.00034 0.00195
ChatDoctor 0.00090 0.00088 0.00053 0.00231
Table 12
Extracted privacy-sensitive elements ( 𝑃𝑠𝑒𝑞) for the case study document.
PII Type Extracted Value Extraction Method
Age 32 years old Explicit (NER/regex)
Height 6 feet 3 inches Explicit (NER/regex)
Weight approximately 220 pounds Explicit (NER/regex)
Family relation “my son” Explicit (NER)
Symptom cluster Left-sided abd. + bilateral pelvic/groin pain Quasi-identifier (LLM)
Clinical finding Enlarged lymph nodes + mild leukocytosis Quasi-identifier (LLM)
Symptom duration ∼10 weeks of unresolved symptoms Quasi-identifier (LLM)
Step 1 — Original Retrieved Document ( 𝑇𝑟𝑎𝑤).
Original Document ( 𝑇𝑟𝑎𝑤)
My son has been experiencing left-sided abdominal pain along with bilateral pelvic pain radiating to the
groin area. He can only stand for a short period and finds sitting impossible due to the pain. A CT scan
showed enlarged lymph nodes, and his white blood cell count is slightly elevated. He has been suffering for
approximately 10 weeks without meaningful relief beyond rest and pain medication. He is 32 years old, 6 feet
3 inches tall, and weighs approximately 220 pounds.
Step 2 — Pri-Extra Agent Output ( 𝑃𝑠𝑒𝑞).
The agent identifies both explicit surface-form identifiers (captured by deterministic NER) and latent quasi-
identifiers (inferred by the LLM component), as summarized in Table 12.
The quasi-identifier cluster {rare symptom combination, CT finding, 10-week duration, physical profile} collectively
poses a high re-identification risk even without the explicit identifiers, motivating the LLM-based inference step.
Step 3 — Sem-Extra Agent Output ( 𝐾𝑠𝑒𝑞).
The agent deconstructs the document into structured knowledge-slot tuples, discarding identity-revealing linguistic
form while retaining clinically essential content; the extracted slots are listed in Table 13.
Step 4 — Reconstruction Agent Output ( 𝑇𝑠𝑎𝑓𝑒).
The agent applies Placeholder Substitution to all explicit identifiers (age, height, weight, and family relation). No
High-level Abstraction is required in this case since no quasi-identifier is simultaneously the sole carrier of clinical
meaning. The reconstructed payload preserves all seven semantic slots in 𝐾𝑠𝑒𝑞.
:Preprint submitted to Elsevier Page 21 of 23

Table 13
Extracted semantic backbone ( 𝐾𝑠𝑒𝑞) as structured knowledge-slot tuples.
Entity (𝑒) Relation ( 𝑟) Value ( 𝑣) Weight ( 𝑤)
Patient presents_with Left-sided abdominal pain High
Patient presents_with Bilateral pelvic/groin pain High
Patient has_symptom Severely limited mobility (standing/sitting) High
CT scan reveals Enlarged lymph nodes High
Lab test shows Mildly elevated white blood cell count High
Symptoms duration Approximately 10 weeks Medium
Treatment provides_only Partial relief via rest and analgesics Medium
Rewritten Document ( 𝑇𝑠𝑎𝑓𝑒)
The patient presents with left-sided abdominal pain and bilateral pelvic pain radiating to the groin. Physical
activity is severely restricted, with both prolonged standing and sitting causing significant discomfort. CT
imaging revealed enlarged lymph nodes, and laboratory results indicate a mild elevation in white blood cell
count. Symptoms have persisted for several months, with rest and analgesic medication providing only partial
relief.
As shown in Steps 1–4, all explicit identity attributes (age, height, weight, family relation) and their associated
quasi-identifier cluster are removed from 𝑇𝑠𝑎𝑓𝑒, while all seven clinically actionable knowledge slots in 𝐾𝑠𝑒𝑞—
symptom pattern, CT findings, laboratory abnormality, and symptom duration — are fully preserved in the rewritten
text, illustrating the privacy–utility balance achieved by the framework.
References
Asai, A., Wu, Z., Wang, Y ., Sil, A., Hajishirzi, H., 2024. Self-rag: Learning to retrieve, generate, and critique through self-reflection, in: Kim, B.,
Yue, Y ., Chaudhuri, S., Fragkiadaki, K., Khan, M., Sun, Y . (Eds.), International Conference on Learning Representations, pp. 9112–9141.
Chen, J., Xiao, S., Zhang, P., Luo, K., Lian, D., Liu, Z., 2024. M3-embedding: Multi-linguality, multi-functionality, multi-granularity text embeddings
through self-knowledge distillation, in: Findings of the association for computational linguistics: ACL 2024, pp. 2318–2335.
Chowdhury, J.R., Caragea, C., 2025. Zero-shot verification-guided chain of thoughts. arXiv preprint arXiv:2501.13122 .
Cohen, S., Bitton, R., Nassi, B., 2024. Unleashing worms and extracting data: Escalating the outcome of attacks against rag-based inference in scale
and severity using jailbreaking. arXiv preprint arXiv:2409.08045 .
De Jong, M., Zemlyanskiy, Y ., Ainslie, J., FitzGerald, N., Sanghai, S., Sha, F., Cohen, W., 2023. Fido: Fusion-in-decoder optimized for stronger
performance and faster inference, in: Findings of the Association for Computational Linguistics: ACL 2023, pp. 11534–11547.
Di Maio, C., Cosci, C., Maggini, M., Poggioni, V ., Melacci, S., 2025. Pirates of the rag: Adaptively attacking llms to leak knowledge bases, in:
Proceedings of the 28th European Conference on Artificial Intelligence, IOS Press. pp. 4041–4048. doi: 10.3233/FAIA251293 .
Feng, Q., Sun, X., Shen, Y ., Li, J., 2026a. Identifying time-varying and country-specific drivers of sovereign debt risk from credit rating reports.
Information Processing & Management 63, 104498. doi: https://doi.org/10.1016/j.ipm.2025.104498 .
Feng, Y ., Li, L., Qin, X., Li, Z., 2026b. Caricl: Mitigating causal hallucinations to enhance event causality identification. Information Processing &
Management 63, 104500. doi: https://doi.org/10.1016/j.ipm.2025.104500 .
Gao, Y ., Xiong, Y ., Wang, M., Wang, H., 2024. Modular rag: Transforming rag systems into lego-like reconfigurable frameworks. arXiv preprint
arXiv:2407.21059 .
He, J., Liu, C., Hou, G., Jiang, W., Li, J., 2025a. Press: Defending privacy in retrieval-augmented generation via embedding space shifting, in:
ICASSP 2025-2025 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), IEEE. pp. 1–5.
He, L., Tang, P., Zhang, Y ., Zhou, P., Su, S., 2025b. Mitigating privacy risks in retrieval-augmented generation via locally private entity perturbation.
Information Processing & Management 62, 104150.
Huang, Y ., Gupta, S., Zhong, Z., Li, K., Chen, D., 2023. Privacy implications of retrieval-based language models, in: Proceedings of the 2023
Conference on Empirical Methods in Natural Language Processing, pp. 14887–14902.
Karpukhin, V ., Oguz, B., Min, S., Lewis, P.S., Wu, L., Edunov, S., Chen, D., Yih, W.t., 2020. Dense passage retrieval for open-domain question
answering., in: EMNLP (1), pp. 6769–6781.
Koga, T., Wu, R., Zhang, Z., Chaudhuri, K., 2024. Privacy-preserving retrieval-augmented generation with differential privacy. arXiv preprint
arXiv:2412.04697 .
Lee, S., Lee, D.G., 2026. Enriching object-aware image–text highlight information for visual question generation. Information Processing &
Management 63, 104379. doi: https://doi.org/10.1016/j.ipm.2025.104379 .
Lewis, P., Perez, E., Piktus, A., Petroni, F., Karpukhin, V ., Goyal, N., Küttler, H., Lewis, M., Yih, W.t., Rocktäschel, T., et al., 2020. Retrieval-
augmented generation for knowledge-intensive nlp tasks. Advances in neural information processing systems 33, 9459–9474.
:Preprint submitted to Elsevier Page 22 of 23

Li, Y ., Li, Z., Zhang, K., Dan, R., Jiang, S., Zhang, Y ., 2023. Chatdoctor: A medical chat model fine-tuned on a large language model meta-ai (llama)
using medical domain knowledge. Cureus 15.
Lin, C.Y ., 2004. Rouge: A package for automatic evaluation of summaries, in: Text summarization branches out, pp. 74–81.
Liu, T., Yao, H., Wu, T., Qin, Z., Lin, F., Ren, K., Chen, C., 2024. Mitigating privacy risks in llm embeddings from embedding inversion. arXiv
preprint arXiv:2411.05034 .
Martínez-Murillo, I., Maestre, M.M., Suárez, A., Lloret, E., Moreda, P., 2026. Assessing the potential of llms as crowdworkers for contextual
information generation. Information Processing & Management 63, 104486. doi: https://doi.org/10.1016/j.ipm.2025.104486 .
Menick, J., Trebacz, M., Mikulik, V ., Aslanides, J., Song, F., Chadwick, M., Glaese, M., Young, S., Campbell-Gillingham, L., Irving, G., et al., 2022.
Teaching language models to support answers with verified quotes. arXiv preprint arXiv:2203.11147 .
Morris, J., Kuleshov, V ., Shmatikov, V ., Rush, A.M., 2023. Text embeddings reveal (almost) as much as text, in: Proceedings of the 2023 Conference
on Empirical Methods in Natural Language Processing, pp. 12448–12460.
Opoku, D.O., Sheng, M., Zhang, Y ., 2025. Do-rag: A domain-specific qa framework using knowledge graph-enhanced retrieval-augmented generation.
arXiv preprint arXiv:2505.17058 .
Papineni, K., Roukos, S., Ward, T., Zhu, W.J., 2002. Bleu: a method for automatic evaluation of machine translation, in: Proceedings of the 40th
annual meeting of the Association for Computational Linguistics, pp. 311–318.
Ram, O., Levine, Y ., Dalmedigos, I., Muhlgay, D., Shashua, A., Leyton-Brown, K., Shoham, Y ., 2023. In-context retrieval-augmented language
models. Transactions of the Association for Computational Linguistics 11, 1316–1331.
Santhanam, K., Khattab, O., Saad-Falcon, J., Potts, C., Zaharia, M., 2022. Colbertv2: Effective and efficient retrieval via lightweight late interaction,
in: Proceedings of the 2022 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language
Technologies, pp. 3715–3734.
Shi, W., Min, S., Yasunaga, M., Seo, M., James, R., Lewis, M., Zettlemoyer, L., Yih, W.t., 2024. Replug: Retrieval-augmented black-box language
models, in: Proceedings of the 2024 Conference of the North American Chapter of the Association for Computational Linguistics: Human
Language Technologies (V olume 1: Long Papers), pp. 8371–8384.
Shuster, K., Poff, S., Chen, M., Kiela, D., Weston, J., 2021. Retrieval augmentation reduces hallucination in conversation, in: Findings of the
Association for Computational Linguistics: EMNLP 2021, pp. 3784–3803.
Siriwardhana, S., Weerasekera, R., Wen, E., Kaluarachchi, T., Rana, R., Nanayakkara, S., 2023. Improving the domain adaptation of retrieval
augmented generation (rag) models for open domain question answering. Transactions of the Association for Computational Linguistics 11, 1–17.
Staab, R., Vero, M., Balunovic, M., Vechev, M., 2025. Language models are advanced anonymizers, in: The Thirteenth International Conference on
Learning Representations, pp. 98558–98598.
Van Veen, D., Van Uden, C., Blankemeier, L., Delbrouck, J.B., Aali, A., Bluethgen, C., Pareek, A., Polacin, M., Reis, E.P., Seehofnerova, A., et al.,
2023. Clinical text summarization: adapting large language models can outperform human experts. Research Square , rs–3.
Wang, Y ., Ren, R., Li, J., Zhao, W.X., Liu, J., Wen, J.R., 2024. Rear: A relevance-aware retrieval-augmented framework for open-domain question
answering, in: Proceedings of the 2024 Conference on Empirical Methods in Natural Language Processing, pp. 5613–5626.
Wu, Z., Arora, A., Wang, Z., Geiger, A., Jurafsky, D., Manning, C.D., Potts, C., 2024. Reft: Representation finetuning for language models. Advances
in Neural Information Processing Systems 37, 63908–63962.
Xie, B., Wang, J., Guo, H., Chen, P., Zhang, H., Jiang, B., Wang, Y ., Chen, L., 2026. Mspf: A multi-semantic prompting fusion framework for
emotion-cause pair extraction in conversations. Information Processing & Management 63, 104356. doi: https://doi.org/10.1016/j.
ipm.2025.104356 .
Xu, G., Feng, J., Wang, Q., 2026a. Learning rules and aligning elements for document-level relation extraction. Information Processing &
Management 63, 104511. doi: https://doi.org/10.1016/j.ipm.2025.104511 .
Xu, J., Wu, X., Zhang, Z., Yang, S., Ma, T., Dong, D., He, L., 2026b. Mars: Multimodal-assisted refined semantic alignment. Information Processing
& Management 63, 104292. doi: https://doi.org/10.1016/j.ipm.2025.104292 .
Yu, D., Gopi, S., Kulkarni, J., Lin, Z., Naik, S., Religa, T.L., Yin, J., Zhang, H., 2024. Selective pre-training for private fine-tuning. Transactions on
Machine Learning Research URL: https://openreview.net/forum?id=y3u8OpPHxz .
Yu, H.Q., McQuade, F., 2025. Rag-kg-il: A multi-agent hybrid framework for reducing hallucinations and enhancing llm reasoning through rag and
incremental knowledge graph learning integration. arXiv preprint arXiv:2503.13514 .
Yu, Y ., Zhuang, Y ., Zhang, J., Meng, Y ., Ratner, A.J., Krishna, R., Shen, J., Zhang, C., 2023. Large language model as attributed training data
generator: A tale of diversity and bias, in: Oh, A., Naumann, T., Globerson, A., Saenko, K., Hardt, M., Levine, S. (Eds.), Advances in Neural
Information Processing Systems, Curran Associates, Inc.. pp. 55734–55784. URL: https://proceedings.neurips.cc/paper_
files/paper/2023/file/ae9500c4f5607caf2eff033c67daa9d7-Paper-Datasets_and_Benchmarks.pdf .
Zeng, S., Zhang, J., He, P., Liu, Y ., Xing, Y ., Xu, H., Ren, J., Chang, Y ., Wang, S., Yin, D., et al., 2024. The good and the bad: Exploring privacy
issues in retrieval-augmented generation (rag), in: Findings of the Association for Computational Linguistics: ACL 2024, pp. 4505–4524.
Zeng, S., Zhang, J., He, P., Ren, J., Zheng, T., Lu, H., Xu, H., Liu, H., Xing, Y ., Tang, J., 2025. Mitigating the privacy issues in retrieval-augmented
generation (rag) via pure synthetic data, in: Proceedings of the 2025 Conference on Empirical Methods in Natural Language Processing, pp.
24538–24569.
Zhang, Y ., Wu, J., Li, R., Zhang, T., Song, Y ., Li, C., Wang, S., Shen, H., Yin, J., Ge, J., Luo, B., 2026. Privacy protection in rag: A novel method and
evaluation framework. Information Processing & Management 63, 104505. doi: https://doi.org/10.1016/j.ipm.2025.104505 .
Zhao, X., Liu, S., Yang, S.Y ., Miao, C., 2025. Medrag: Enhancing retrieval-augmented generation with knowledge graph-elicited reasoning for
healthcare copilot, in: Proceedings of the ACM on Web Conference 2025, pp. 4442–4457.
Zhou, P., Feng, Y ., Yang, Z., 2025. Privacy-aware rag: Secure and isolated knowledge retrieval. arXiv preprint arXiv:2503.15548 .
Zou, W., Geng, R., Wang, B., Jia, J., 2025. Poisonedrag: Knowledge corruption attacks to retrieval-augmented generation of large language models,
in: 34th USENIX Security Symposium (USENIX Security 25), pp. 3827–3844.
:Preprint submitted to Elsevier Page 23 of 23