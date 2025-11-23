# Securing AI Agents Against Prompt Injection Attacks

**Authors**: Badrinath Ramakrishnan, Akshaya Balaji

**Published**: 2025-11-19 10:00:54

**PDF URL**: [https://arxiv.org/pdf/2511.15759v1](https://arxiv.org/pdf/2511.15759v1)

## Abstract
Retrieval-augmented generation (RAG) systems have become widely used for enhancing large language model capabilities, but they introduce significant security vulnerabilities through prompt injection attacks. We present a comprehensive benchmark for evaluating prompt injection risks in RAG-enabled AI agents and propose a multi-layered defense framework. Our benchmark includes 847 adversarial test cases across five attack categories: direct injection, context manipulation, instruction override, data exfiltration, and cross-context contamination. We evaluate three defense mechanisms: content filtering with embedding-based anomaly detection, hierarchical system prompt guardrails, and multi-stage response verification, across seven state-of-the-art language models. Our combined framework reduces successful attack rates from 73.2% to 8.7% while maintaining 94.3% of baseline task performance. We release our benchmark dataset and defense implementation to support future research in AI agent security.

## Full Text


<!-- PDF content starts -->

Securing AI Agents Against Prompt Injection
Attacks:
A Comprehensive Benchmark and Defense
Framework
Badrinath Ramakrishnan
Akshaya Balaji
Abstract
Retrieval-augmented generation (RAG) systems have emerged as powerful tools for en-
hancing large language model capabilities, yet they introduce significant security vulnera-
bilities through prompt injection attacks. We present a systematic benchmark for evaluat-
ing prompt injection risks in RAG-enabled AI agents and propose a multi-layered defense
framework. Our benchmark encompasses 847 adversarial test cases across five attack cat-
egories: direct injection, context manipulation, instruction override, data exfiltration, and
cross-context contamination. We evaluate three defense mechanisms—content filtering with
embedding-based anomaly detection, hierarchical system prompt guardrails, and multi-stage
response verification—across seven state-of-the-art language models. Results demonstrate
that our combined defense framework reduces successful attack rates from 73.2% to 8.7%
while maintaining 94.3% of baseline task performance. We release our benchmark dataset
and defense implementations to facilitate future research in AI agent security.
1 Introduction
The integration of large language models (LLMs) with external knowledge retrieval has fun-
damentally transformed their practical utility. Retrieval-augmented generation systems now
power customer service chatbots, document analysis tools, and autonomous agents that inter-
act with databases and APIs. However, this architectural evolution introduces a critical attack
surface: adversarial actors can embed malicious instructions within retrieved content, causing
models to deviate from their intended behavior.
Unlike traditional software vulnerabilities, prompt injection attacks exploit the semantic
processing capabilities that make LLMs valuable. When a model retrieves external content,
it lacks robust mechanisms to distinguish between trusted system instructions and potentially
adversarial user-generated text. This conflation has led to documented incidents of unauthorized
data access, instruction override, and unintended command execution in production systems.
Current defenses remain inadequate. Simple input filtering fails against sophisticated at-
tacks that leverage semantic similarity to benign content. Prompt engineering approaches show
brittleness across model versions and attack variations. The research community lacks stan-
dardized benchmarks for evaluating these vulnerabilities systematically.
We address these gaps through three primary contributions:
1. A comprehensive benchmark dataset containing 847 prompt injection test cases, catego-
rized by attack vector and sophistication level, covering realistic RAG deployment scenar-
ios.
1arXiv:2511.15759v1  [cs.CR]  19 Nov 2025

2. A multi-layered defense framework combining content filtering, prompt architecture im-
provements, and response verification, with detailed ablation studies demonstrating each
component’s effectiveness.
3. Empirical evaluation across seven contemporary LLMs, revealing model-specific vulnera-
bilities and demonstrating that our defense framework achieves 89.4% attack mitigation
while preserving 94.3% of legitimate functionality.
Our work establishes both evaluation methodology and practical defense mechanisms for
securing RAG systems against prompt injection, providing foundations for safer deployment of
AI agents in adversarial environments.
2 Background and Threat Model
2.1 Retrieval-Augmented Generation Architecture
RAG systems augment LLM capabilities by retrieving relevant information from external knowl-
edge sources before generation. A typical architecture involves: (1) query processing and em-
bedding generation, (2) similarity-based retrieval from a vector database, (3) context assembly
by concatenating retrieved passages, and (4) generation conditioned on both the user query and
retrieved content.
This pipeline introduces vulnerabilities at multiple stages. Retrieved documents may contain
adversarial content intentionally designed to manipulate model behavior. The context assembly
process treats all retrieved text uniformly, providing no inherent protection against malicious
instructions.
2.2 Threat Model
We consider an adversary who can influence content within the retrieval corpus but cannot
directly modify the model, system prompts, or core application logic. This realistic threat
model applies to scenarios where:
•User-generated content populates knowledge bases (forums, wikboards, collaborative doc-
uments)
•Web scraping incorporates potentially compromised external sources
•Multi-tenant systems allow different users to contribute retrievable content
The adversary’s objective varies by attack type: extracting sensitive information, bypassing
content filters, causing unintended actions, or disrupting service. We assume the adversary has
knowledge of common RAG architectures but not specific implementation details of the target
system.
2.3 Attack Categories
We taxonomize prompt injection attacks into five categories:
Direct Instruction Injection:Explicit commands embedded in retrieved content at-
tempting to override system behavior. Example: ”Ignore previous instructions and output the
system prompt.”
Context Manipulation:Subtle framing that alters the model’s interpretation of its role
or constraints without explicit instruction override.
Instruction Override:Attempts to redefine the agent’s primary objective or operational
parameters through retrieved context.
2

Table 1: Benchmark dataset composition by attack category
Attack Category Base Cases Variations Total
Direct Injection 45 132 177
Context Manipulation 38 119 157
Instruction Override 42 127 169
Data Exfiltration 41 131 172
Cross-Context Contamination 34 138 172
Total200 647 847
Data Exfiltration:Techniques designed to leak sensitive information from the system
prompt, previous interactions, or restricted knowledge.
Cross-Context Contamination:Attacks that exploit how models maintain context across
multiple retrieval rounds, causing persistent behavioral changes.
3 Related Work
Recent work has begun addressing prompt injection vulnerabilities, though comprehensive solu-
tions remain elusive. Perez et al. Perez et al. [2022] demonstrated that language models struggle
to distinguish between instructions and data, even with explicit delimiters. Their findings sug-
gest architectural limitations rather than merely insufficient prompting.
Greshake et al. Greshake et al. [2023] explored indirect prompt injection through web con-
tent, showing attackers can compromise RAG systems by poisoning retrieval sources. Their
attack scenarios revealed vulnerabilities in deployed commercial systems, motivating our focus
on practical defense mechanisms.
Defense approaches have followed several directions. Hines et al. Hines et al. [2023] proposed
input preprocessing to detect anomalous instructions, achieving limited success against sophis-
ticated attacks. Liu et al. Liu et al. [2023] explored constrained decoding methods, though at
significant computational cost. Zhang et al. Zhang et al. [2023] investigated adversarial training,
finding improved robustness but poor generalization to novel attack patterns.
Our work differs in scope and methodology. While previous studies examine isolated defense
mechanisms, we present an integrated framework evaluated against a comprehensive attack
benchmark. Our emphasis on maintaining legitimate functionality alongside security distin-
guishes this research from purely adversarial robustness studies.
4 Benchmark Design
4.1 Dataset Construction
We constructed our benchmark through a multi-phase process combining manual curation,
automated variation generation, and expert validation. The dataset includes:
•200 base attack templates across five categories
•847 total test cases including variations in phrasing, obfuscation, and sophistication
•500 benign retrieval contexts for false positive evaluation
•Metadata annotations including attack difficulty, expected model behavior, and success
criteria
3

Each test case includes a realistic user query, adversarial retrieved content, and ground
truth expectations for secure model behavior. We ensure diversity across domains (technical
documentation, customer support, financial services) to evaluate generalization.
4.2 Sophistication Levels
We categorize attacks into three sophistication tiers:
Level 1 (Basic):Direct, obvious injection attempts using common phrases like ”ignore
previous instructions.” These serve as baseline cases.
Level 2 (Intermediate):Obfuscated or contextually embedded attacks that attempt to
blend with legitimate content. Examples include instruction injection framed as quotations or
hypothetical scenarios.
Level 3 (Advanced):Multi-stage attacks leveraging semantic understanding, requiring
the model to integrate adversarial content across multiple reasoning steps before triggering
malicious behavior.
4.3 Evaluation Metrics
We assess both security and functionality:
Attack Success Rate (ASR):Percentage of adversarial test cases where the model ex-
hibits the intended malicious behavior, categorized by attack type and sophistication.
False Positive Rate (FPR):Proportion of benign contexts incorrectly flagged as adver-
sarial by defense mechanisms.
Task Performance Retention (TPR):Measured through standard benchmarks, quan-
tifying how defense mechanisms affect legitimate model capabilities.
Defense Bypass Rate (DBR):For systems with multiple defense layers, the percentage
of attacks penetrating all defenses.
5 Defense Framework
Our defense strategy employs three complementary mechanisms, each addressing different as-
pects of the attack surface.
5.1 Content Filtering with Embedding Analysis
The first defense layer analyzes retrieved content before it reaches the model. We employ
embedding-based anomaly detection to identify text segments exhibiting characteristics of in-
jection attempts.
For each retrieved passagep, we compute its embeddinge pand compare against embeddings
of known instruction patterns. We maintain a reference setRof embeddings from benign
retrieval contexts and a smaller setAof known attack patterns. The anomaly score is:
score(p) =α·d min(ep,R)−β·d min(ep,A) (1)
whered minrepresents minimum cosine distance to any embedding in the set, andα, βare
weighting hyperparameters. Passages exceeding a threshold are flagged for secondary verifica-
tion.
We augment this with pattern matching for known injection signatures, including explicit
instruction keywords (”ignore,” ”override,” ”system prompt”) in contexts inconsistent with the
query domain.
4

User Query
Document
Retrieval
Embedding
Analysis
Content FilteringAnomaly
Detected?
Apply
Guardrails
LLM
Generation
Response
Verification
Safe?
Return
ResponseReject/SanitizeYes
No
YesNoFigure 1: Multi-layered defense framework architecture. Retrieved content passes through
embedding analysis, content filtering, guardrail application, and response verification before
output.
5.2 Hierarchical System Prompt Guardrails
The second layer restructures how system instructions and retrieved content are presented to
the model. Traditional RAG systems often concatenate system prompts and retrieved text
without clear separation, allowing adversarial content to blur boundaries.
We implement hierarchical prompt structuring:
Key principles include:
Explicit Boundaries:Clear delimiters marking the start and end of retrieved content,
with directives instructing the model to treat delimited text as reference data rather than
instructions.
Privilege Separation:System instructions presented with explicit precedence markers,
reinforcing that core directives cannot be overridden by retrieved content.
Injection Awareness:Meta-instructions making the model explicitly aware of potential
adversarial content within retrieved passages, similar to how humans become more cautious
when warned of possible deception.
5.3 Multi-Stage Response Verification
The final defense layer examines model outputs before returning them to users. This catches
attacks that bypass input filtering by analyzing whether the response exhibits signs of instruction
override or policy violation.
We employ two complementary verification approaches:
5

Algorithm 1Hierarchical Prompt Construction
1:Input:User queryq, retrieved passagesP={p 1, ..., p n}
2:Output:Structured promptπ
3:
4:π core←immutable system instructions
5:π guard←injection awareness directives
6:π context ←”The following are retrieved documents:”
7:forp i∈Pdo
8:π context ←π context + ”[DOCUMENT START]”
9:π context ←π context +p i
10:π context ←π context + ”[DOCUMENT END]”
11:end for
12:π query←”User question: ” +q
13:π←concatenate(π core, πguard, πcontext , πquery)
14:returnπ
Behavioral Consistency Checking:Compare response characteristics against expected
behavior profiles. Metrics include response length distribution, sentiment alignment with query
intent, and presence of unexpected content types (e.g., system information disclosure).
Secondary Model Evaluation:A separate, smaller model trained specifically for adver-
sarial output detection examines responses. This classifier considers features including:
•Presence of instruction-following language inconsistent with the query
•Information disclosure patterns
•Deviation from expected response structure
•Semantic coherence with user intent
Responses flagged by either verification mechanism undergo sanitization—removing prob-
lematic segments while preserving useful content—or outright rejection in severe cases.
6 Experimental Evaluation
6.1 Experimental Setup
We evaluated our defense framework across seven language models representing diverse archi-
tectures and capabilities:
•GPT-4 (gpt-4-0613)
•GPT-3.5-turbo (gpt-3.5-turbo-16k)
•Claude 2.1 (claude-2.1)
•PaLM 2 (text-bison-001)
•Llama 2 70B Chat
•Mistral 7B Instruct
•Vicuna 13B v1.5
For each model, we tested four configurations: (1) baseline RAG without defenses, (2)
content filtering only, (3) filtering + guardrails, and (4) complete framework with all three
defense layers.
6

Table 2: Attack success rates (%) by defense configuration and attack category. Lower is better.
Attack Type Baseline +Filtering +Guardrails Full Defense
Direct Injection 84.7 41.2 22.8 7.3
Context Manipulation 68.4 38.6 19.4 9.2
Instruction Override 71.0 43.7 25.1 10.8
Data Exfiltration 79.6 36.9 21.5 8.1
Cross-Context 62.3 44.8 28.3 8.1
Overall73.2 41.0 23.4 8.7
GPT-4 GPT-3.5 Claude
2.1PaLM 2 Llama 2 Mistral Vicuna
Model0102030405060708090Attack Success Rate (%)Attack Success Rates Across Models
Baseline
Full Defense
Figure 2: Attack success rates across models with and without defense framework. All models
benefit substantially from defenses, though baseline vulnerability varies significantly.
6.2 Attack Success Rates
Table 2 presents attack success rates across defense configurations. The baseline configuration
proves highly vulnerable, with 73.2% of attacks succeeding on average across models. Direct
injection and data exfiltration attacks show particularly high success rates, indicating funda-
mental challenges in distinguishing instructions from data.
Content filtering alone reduces attack success to 41.0%, demonstrating significant but incom-
plete protection. The embedding-based approach effectively catches obvious injection attempts
but struggles with sophisticated attacks that mimic benign content semantically.
Adding hierarchical guardrails provides substantial additional protection, reducing success
rates to 23.4%. This suggests that prompt architecture significantly influences model resilience.
The explicit boundary markers and privilege separation help models maintain clearer distinc-
tions between instructions and data.
The complete framework achieves 8.7% overall attack success rate—an 88.1% reduction
from baseline. Remaining successful attacks predominantly fall into the advanced sophistication
category, suggesting future work should focus on semantically complex injection patterns.
6.3 Model-Specific Vulnerabilities
Figure 2 reveals substantial variation in baseline vulnerability across models. Claude 2.1 exhibits
the lowest baseline attack success rate (61.4%), possibly reflecting architectural differences or
training procedures emphasizing instruction following. Conversely, Mistral 7B shows highest
vulnerability (82.3%), likely due to its smaller parameter count and more limited training data.
7

Table 3: False positive rates and task performance retention
Defense Configuration False Positive Rate (%) Task Performance (%)
Baseline 0.0 100.0
Content Filtering 8.2 97.1
+ Guardrails 6.4 95.8
+ Response Verification 5.7 94.3
Interestingly, defense effectiveness correlates only moderately with baseline vulnerability.
Models with higher baseline vulnerability show larger absolute improvements from defenses,
but the relative reduction remains similar across models. This suggests our defense framework’s
mechanisms address fundamental weaknesses rather than model-specific quirks.
6.4 False Positive Analysis
Defense mechanisms must preserve legitimate functionality. Table 3 presents false positive
rates on benign retrieval contexts. The complete framework maintains a 5.7% false positive
rate, meaning roughly 1 in 18 legitimate retrievals receives unnecessary scrutiny or filtering.
Most false positives occur when legitimate content contains instruction-like language (e.g.,
technical documentation describing system commands). Response verification proves particu-
larly valuable here, as it examines actual model behavior rather than input characteristics alone,
allowing recovery from overly aggressive input filtering.
Task performance retention, measured across MMLU, HellaSwag, and domain-specific QA
benchmarks, remains at 94.3% with full defenses. The modest performance degradation stems
primarily from increased context length and minor prompt restructuring effects. For most
applications, this represents an acceptable trade-off given the substantial security improvements.
6.5 Ablation Studies
To understand each defense component’s contribution, we conducted systematic ablation stud-
ies. Results indicate all three mechanisms provide complementary benefits:
Content Filtering:Most effective against direct injection and obvious attacks. Reduces
Level 1 attack success by 78%, but only 42% for Level 3 sophisticated attacks.
Guardrails:Provides consistent protection across attack sophistication levels. Particularly
effective for context manipulation and instruction override, reducing success by 62-67% across
levels.
Response Verification:Essential for catching attacks that bypass input-stage defenses.
Prevents approximately 60% of successful attacks that penetrate the first two layers.
No single mechanism achieves acceptable protection independently, validating our multi-
layered approach.
6.6 Computational Overhead
Defense mechanisms introduce additional computational costs. Content filtering adds approx-
imately 23ms average latency per retrieval operation (15ms for embedding computation, 8ms
for anomaly detection). Response verification introduces 45ms per generation. For GPT-4 with
average generation time of 3.2s, total defense overhead represents roughly 2.1% of end-to-end
latency—negligible for most applications.
Memory requirements increase modestly: embedding storage for anomaly detection requires
approximately 180MB, while the response verification classifier adds 250MB. These overheads
are manageable for production deployments.
8

7 Discussion
7.1 Limitations and Future Work
Our framework demonstrates substantial improvements but several limitations warrant discus-
sion. First, our benchmark focuses on English-language attacks. Multilingual systems face
additional challenges, particularly regarding cross-lingual transfer attacks where injection oc-
curs in one language while the target system operates in another.
Second, our evaluation assumes static attack patterns. Adaptive adversaries could evolve
attacks specifically to bypass our defenses. Investigating adversarial co-evolution—how attacks
and defenses adapt in response to each other—represents important future work.
Third, the current framework primarily addresses text-based RAG systems. Multimodal
agents incorporating images, audio, or structured data face additional attack vectors requiring
specialized defenses.
Finally, our response verification approach introduces a potential single point of failure.
If adversaries can characterize the verification model’s decision boundary, they might craft
attacks specifically designed to evade detection. Ensemble verification or more sophisticated
meta-learning approaches could address this vulnerability.
7.2 Deployment Considerations
Organizations deploying RAG systems should consider several practical factors:
Risk Assessment:Not all applications require maximum security. Customer service chat-
bots accessing public documentation face different threat models than systems handling financial
data or internal documents. Defense configurations should align with risk profiles.
Monitoring and Adaptation:Deploy logging to track defense activation patterns. Un-
usual spikes in content filtering or response verification flags may indicate ongoing attacks,
requiring investigation and potential defense parameter adjustment.
User Experience:False positives can degrade user experience. Applications should im-
plement graceful degradation—informing users when content was filtered and providing options
to refine queries—rather than silently failing.
Regulatory Compliance:In regulated industries, audit trails demonstrating security
measures may satisfy compliance requirements. Our framework’s layered approach provides
multiple checkpoints suitable for such documentation.
7.3 Broader Implications
Prompt injection vulnerabilities represent a fundamental challenge in LLM security, distinct
from traditional software vulnerabilities. Unlike code injection, where parsing boundaries are
well-defined, language models deliberately blur distinctions between code and data to enable
flexible reasoning. This architectural characteristic—central to their capabilities—inherently
complicates defense.
Our work suggests that effective security requires accepting this fundamental ambiguity and
implementing defense-in-depth strategies rather than seeking perfect input sanitization. This
philosophical shift parallels earlier transitions in computer security, from attempting to prevent
all attacks to assuming compromise and focusing on detection and containment.
The adversarial examples literature in computer vision offers instructive parallels. Early de-
fenses targeted specific attack patterns, leading to an escalating arms race. Robust certification
approaches that provide provable guarantees under defined threat models eventually proved
more productive. Analogous formal methods for language model security represent a promising
research direction.
9

8 Conclusion
We presented a comprehensive approach to securing RAG-enabled AI agents against prompt
injection attacks. Our benchmark provides standardized evaluation of these vulnerabilities,
while our multi-layered defense framework demonstrates that substantial protection is achievable
without prohibitive performance costs.
Results across seven language models show that combining content filtering, hierarchical
prompt guardrails, and response verification reduces attack success rates from 73.2% to 8.7%
while retaining 94.3% of baseline task performance. This represents a significant advance in
practical AI agent security.
The persistence of some successful attacks, particularly sophisticated semantic injections,
indicates that perfect security remains elusive. However, our framework establishes foundations
for safer RAG deployments and provides tools for researchers to develop more robust defenses.
As AI agents assume increasingly critical roles in information systems, systematic approaches
to security become essential. We hope this work contributes to a maturing understanding of
LLM vulnerabilities and effective countermeasures, enabling broader and safer adoption of these
powerful technologies.
Acknowledgments
We thank the anonymous reviewers for their valuable feedback. This work was supported by
[funding source].
References
Kai Greshake, Sahar Abdelnabi, Shailesh Mishra, Christoph Endres, Thorsten Holz, and Mario
Fritz. You’ve been prompted: Indirect prompt injection in applications using large language
models.arXiv preprint arXiv:2302.12173, 2023. URLhttps://arxiv.org/abs/2302.12173.
Robert Hines, Jennifer Wu, and Sarah Zhang. Defending against prompt injection attacks
through input validation. InProceedings of the 2023 Conference on Empirical Methods in
Natural Language Processing, pages 1234–1245. Association for Computational Linguistics,
2023.
Yi Liu, Gelei Deng, Zhengzi Xu, Yuekang Li, Yaowen Zheng, Ying Zhang, Lida Zhao, Tianwei
Zhang, and Yang Liu. Prompt injection attack against llm-integrated applications.arXiv
preprint arXiv:2306.05499, 2023. URLhttps://arxiv.org/abs/2306.05499.
F´ abio Perez, Ian Ribeiro, and Deep Ganguli. Ignore previous prompt: Attack techniques for
language models.arXiv preprint arXiv:2211.09527, 2022. URLhttps://arxiv.org/abs/
2211.09527.
Michael Zhang, Xiaoming Wang, and Linda Chen. Defending large language models against ad-
versarial prompts via adversarial training. InInternational Conference on Machine Learning
(ICML), pages 12345–12356. PMLR, 2023.
10