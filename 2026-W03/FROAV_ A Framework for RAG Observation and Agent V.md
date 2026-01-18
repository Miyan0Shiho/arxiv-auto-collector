# FROAV: A Framework for RAG Observation and Agent Verification -- Lowering the Barrier to LLM Agent Research

**Authors**: Tzu-Hsuan Lin, Chih-Hsuan Kao

**Published**: 2026-01-12 13:02:32

**PDF URL**: [https://arxiv.org/pdf/2601.07504v1](https://arxiv.org/pdf/2601.07504v1)

## Abstract
The rapid advancement of Large Language Models (LLMs) and their integration into autonomous agent systems has created unprecedented opportunities for document analysis, decision support, and knowledge retrieval. However, the complexity of developing, evaluating, and iterating on LLM-based agent workflows presents significant barriers to researchers, particularly those without extensive software engineering expertise. We present FROAV (Framework for RAG Observation and Agent Verification), an open-source research platform that democratizes LLM agent research by providing a plug-and-play architecture combining visual workflow orchestration, a comprehensive evaluation framework, and extensible Python integration. FROAV implements a multi-stage Retrieval-Augmented Generation (RAG) pipeline coupled with a rigorous "LLM-as-a-Judge" evaluation system, all accessible through intuitive graphical interfaces. Our framework integrates n8n for no-code workflow design, PostgreSQL for granular data management, FastAPI for flexible backend logic, and Streamlit for human-in-the-loop interaction. Through this integrated ecosystem, researchers can rapidly prototype RAG strategies, conduct prompt engineering experiments, validate agent performance against human judgments, and collect structured feedback-all without writing infrastructure code. We demonstrate the framework's utility through its application to financial document analysis, while emphasizing its material-agnostic architecture that adapts to any domain requiring semantic analysis. FROAV represents a significant step toward making LLM agent research accessible to a broader scientific community, enabling researchers to focus on hypothesis testing and algorithmic innovation rather than system integration challenges.

## Full Text


<!-- PDF content starts -->

FROAV: A Framework for RAG Observation and Agent
Verification — Lowering the Barrier to LLM Agent Research
Tzu-Hsuan Lin∗1and Chih-Hsuan Kao†2
1AetheTech‡
January 13, 2026
Abstract
The rapid advancement of Large Language Models (LLMs) and their integration into
autonomous agent systems has created unprecedented opportunities for document analysis,
decision support, and knowledge retrieval. However, the complexity of developing, evaluat-
ing, and iterating on LLM-based agent workflows presents significant barriers to researchers,
particularly those without extensive software engineering expertise. We present FROAV
(Framework for RAG Observation and Agent Verification), an open-source research plat-
form that democratizes LLM agent research by providing a plug-and-play architecture com-
bining visual workflow orchestration, a comprehensive evaluation framework, and extensible
Python integration. FROAV implements a multi-stage Retrieval-Augmented Generation
(RAG) pipeline coupled with a rigorous “LLM-as-a-Judge” evaluation system, all accessible
through intuitive graphical interfaces. Our framework integrates n8n for no-code workflow
design, PostgreSQL for granular data management, FastAPI for flexible backend logic, and
Streamlit for human-in-the-loop interaction. Through this integrated ecosystem, researchers
can rapidly prototype RAG strategies, conduct prompt engineering experiments, validate
agent performance against human judgments, and collect structured feedback—all without
writing infrastructure code. We demonstrate the framework’s utility through its application
to financial document analysis, while emphasizing its material-agnostic architecture that
adapts to any domain requiring semantic analysis. FROAV represents a significant step
toward making LLM agent research accessible to a broader scientific community, enabling
researchers to focus on hypothesis testing and algorithmic innovation rather than system
integration challenges.
Keywords:LargeLanguageModels,Retrieval-AugmentedGeneration,LLM-as-a-Judge,
Human-in-the-Loop, Research Infrastructure, Workflow Orchestration, Agent Verification
1 Introduction
1.1 The Promise and Challenge of LLM Agents
Large Language Models have fundamentally transformed the landscape of natural language
processing, demonstrating remarkable capabilities in text generation, reasoning, and knowledge
synthesis [1,12]. The emergence of LLM-based autonomous agents—systems that leverage LLMs
to perceive, reason, and act upon complex information—has opened new frontiers in document
analysis, decision support, and automated research assistance [17]. Particularly compelling
is the application of Retrieval-Augmented Generation (RAG) techniques, which ground LLM
∗Corresponding author: thlin40210@aethetech.com
†Corresponding author: carolyn@aethetech.com
‡Website: www.aethetech.com
1arXiv:2601.07504v1  [cs.LG]  12 Jan 2026

outputs in external knowledge bases, significantly reducing hallucination and improving factual
accuracy [9].
However, the development and evaluation of LLM agents remains a formidable challenge.
Researchers face a complex technology stack requiring expertise in:
1.Workflow Orchestration: Designing multi-step reasoning pipelines with appropriate
error handling and state management.
2.Vector Database Management: Implementing and optimizing semantic search over
large document corpora.
3.Prompt Engineering: Iteratively refining prompts across multiple agent roles and eval-
uation dimensions.
4.Evaluation Methodology: Establishing rigorous metrics that capture both automated
assessments and human judgments.
5.Data Infrastructure: Managing persistent storage for experiments, logs, and feedback
collection.
Thismultifacetedcomplexitycreatessubstantialbarrierstoentry, particularlyfordomainex-
perts (e.g., financial analysts, medical researchers, legal scholars) who possess invaluable subject-
matter expertise but lack software engineering backgrounds. The result is a concerning gap:
those best positioned to validate LLM agent outputs in specialized domains often cannot par-
ticipate meaningfully in system development and evaluation.
1.2 The Need for Accessible Research Infrastructure
Current approaches to LLM agent development typically require researchers to either:
1.Build from scratch: Implementing custom pipelines using frameworks like LangChain or
LlamaIndex, which demands significant programming expertise and system design knowl-
edge.
2.Use closed platforms: Relying on proprietary solutions that limit reproducibility, cus-
tomization, and scientific transparency.
3.Cobble together tools: Manually integrating disparate components (databases, APIs,
frontends) with substantial integration overhead.
None of these approaches adequately serve the research community’s need for accessible,
reproducible, and extensible experimentation platforms. The scientific study of LLM agents
requires infrastructure that supports:
•Rapid prototypingof different RAG strategies and prompt configurations.
•Systematic evaluationacross multiple dimensions with both automated and human
assessments.
•Transparent loggingof all intermediate steps for debugging and analysis.
•Flexible extensionto accommodate novel techniques and domain-specific requirements.
•Collaborative workflowsenabling domain experts and ML researchers to contribute
according to their strengths.
2

1.3 Our Contribution: FROAV
We introduce FROAV (Framework for RAG Observation and Agent Verification), an integrated
research platform designed to lower the barrier to LLM agent experimentation while maintaining
the flexibility required for serious scientific inquiry. FROAV’s key contributions include:
1.Visual Workflow Orchestration: Integration with n8n enables researchers to design,
modify, and experiment with complex agent workflows through an intuitive drag-and-drop
interface, eliminating the need for boilerplate infrastructure code.
2.Multi-Dimensional Evaluation Framework: A comprehensive “LLM-as-a-Judge” sys-
tem that evaluates agent outputs across four theoretically-grounded dimensions (Reliabil-
ity, Completeness, Understandability, and Relevance), with multi-model consensus mech-
anisms to improve evaluation robustness.
3.Human-in-the-Loop Integration: A Streamlit-based frontend that enables domain
experts to review agent outputs, provide structured feedback, and contribute human judg-
ments that can be correlated with automated assessments.
4.Granular Data Management: PostgreSQL-based storage architecture that captures
execution traces, evaluation results, and human feedback with full provenance tracking,
enabling sophisticated post-hoc analysis.
5.Extensible Python Backend: FastAPI-based services that expose clean interfaces for
custom preprocessing, analysis routines, and integration with existing Python-based ML
pipelines.
6.Containerized Deployment: Docker Compose configuration enabling reproducible de-
ployment across different computing environments with minimal setup.
While we demonstrate FROAV through its application to financial document analysis (SEC
10-K, 10-Q filings), the architecture is deliberately material-agnostic, adaptable to any domain
requiring systematic semantic analysis and agent verification.1
Table 1: Development Effort Comparison: Manual Coding vs. FROAV Framework
Metric Manual Coding FROAV Framework
Infrastructure Setup40–50 hours (DB, Docker, API
auth, Environment)∼1 hour (Single Docker Com-
pose)
Workflow Logic1000+ Lines of Python (Orches-
tration, Error handling)0 Lines (Visual Drag-and-Drop)
Evaluation Logic1000+ Lines (Multi-model con-
sensus, Persistence)Pre-configured "Judge" nodes
HITL Interface80+ hours (Frontend dev, API
integration)2 hours (Modular Streamlit con-
fig)
Learning CurveHigh (Requires Senior Software
Engineering)Low (Domain Expert / Analyst
friendly)
*Estimates based on typical development cycles for a production-grade RAG pipeline with evaluation.
1https://github.com/tw40210/FROAV_LLM
3

2 Related Work
2.1 Retrieval-Augmented Generation
RAGsystemshaveemergedasaprimaryapproachforgroundingLLMoutputsinexternalknowl-
edge. The foundational work by Lewis et al. [9] demonstrated that combining retrieval mecha-
nisms with generative models significantly improves knowledge-intensive NLP tasks. Subsequent
developments have refined RAG architectures through improved retrieval mechanisms [7], multi-
hop reasoning [15], and adaptive retrieval strategies [8].
However, implementing production-quality RAG systems remains challenging. Frameworks
like LangChain [6] and LlamaIndex [10] provide building blocks but require substantial program-
ming expertise. Recent work has highlighted the sensitivity of RAG systems to retrieval quality,
chunk sizing, and prompt design [5], underscoring the need for systematic experimentation in-
frastructure.
2.2 LLM-as-a-Judge Evaluation
The “LLM-as-a-Judge” paradigm has gained traction as a scalable approach to evaluating LLM
outputs. Zheng et al. [18] demonstrated that strong LLMs can provide evaluations highly cor-
related with human judgments across various dimensions. This approach has been extended
to domain-specific evaluation [4], multi-dimensional assessment [11], and debate-based refine-
ment [3].
Critical to effective LLM-as-a-Judge systems is addressing known biases, including position
bias, verbosity bias, and self-enhancement [18]. Multi-model consensus approaches have shown
promise in mitigating these biases [16], though implementation complexity has limited their
adoption in research settings.
2.3 Human-in-the-Loop Systems
Human-in-the-loop (HITL) approaches remain essential for validating AI system outputs, par-
ticularly in high-stakes domains. Research has demonstrated the value of structured human
feedback for improving model performance [13], identifying failure modes [14], and establishing
evaluation baselines [2].
ToolslikeLabelStudioandArgillaprovideannotationinterfaces, butthesetypicallyfocuson
training data collection rather than production system evaluation. Platforms designed for LLM
evaluation often require significant integration effort and may not support custom evaluation
dimensions.
2.4 Research Infrastructure Gaps
Despite these advances, significant gaps remain in the research infrastructure landscape:
1.Integration Burden: Researchers must manually connect retrieval systems, LLMs, eval-
uation frameworks, and feedback collection tools.
2.Reproducibility Challenges: Custom implementations often lack standardized logging,
making experiments difficult to reproduce and compare.
3.Accessibility Barriers: Most tools assume programming expertise, excluding domain
experts from meaningful participation.
4.Evaluation Fragmentation: Automated and human evaluations are typically collected
through separate systems, complicating correlation analysis.
4

3 Methodology
FROAV’s architecture embodies a key design philosophy:maximum accessibility with max-
imum flexibility. We achieve this through a layered architecture that provides intuitive inter-
faces for common operations while exposing powerful extension points for advanced customiza-
tion. Figure 1 illustrates the high-level system architecture.
Figure 1: FROAV System Architecture showing the four primary components and their inter-
connections.
3.1 System Architecture Overview
FROAV comprises four primary components orchestrated through Docker Compose for repro-
ducible deployment:
3.1.1 n8n Workflow Orchestrator
n8nservesasthecentralnervoussystemofFROAV,providingvisualworkflowdesigncapabilities
that enable researchers to construct complex agent workflows without writing infrastructure
code. Key capabilities include visual pipeline design, subworkflow composition, code injection
points (JavaScript/Python), built-in AI nodes, and execution logging.
3.1.2 PostgreSQL Data Layer
PostgreSQL provides the persistent storage layer with a schema designed for research workflows
for correlation analysis, temporal analysis, and access control.
3.1.3 FastAPI Backend Services
The FastAPI backend provides Python-based services that extend n8n’s capabilities. This archi-
tecture enables researchers to implement custom preprocessing logic, integrate existing Python
libraries, and extend the system with new endpoints without modifying core workflows.
5

3.1.4 Streamlit Frontend
The Streamlit frontend provides an accessible interface for human-in-the-loop evaluation, includ-
ing an interactive report viewer, judgment browser, feedback collection, and user authentication.
3.2 RAG Pipeline Architecture
FROAV implements a sophisticated multi-stage RAG pipeline designed for complex document
analysis tasks. The pipeline architecture supports iterative refinement based on evaluation
feedback. While we use Supabase (PostgreSQL with pgvector) as the default third-party vector
store service provider, FROAV is highly customizable.
3.2.1 Core RAG Stages
The RAG pipeline operates in three core stages:
1.PDF Content Parsing: Utilizing custom FastAPI backend services to ingest PDF files
and convert raw document content into structured text.
2.Materials Chunking: Segmenting extracted text into contextually coherent chunks suit-
able for embedding.
3.Embedding and Vector Storage: Transforming chunks into high-dimensional vectors
and storing them with metadata in the vector database.
3.3 LLM-as-a-Judge Evaluation Framework
FROAV implements a multi-model, multi-dimensional evaluation framework that addresses
known limitations of single-model assessment.
3.3.1 Evaluation Dimensions
FROAV evaluates outputs across four dimensions (Table 2).
Table 2: Evaluation Dimensions
Dimension Definition Evaluation Focus
ReliabilityAccuracy of stated facts Data verification, calculation cor-
rectness, source fidelity
CompletenessPresence of required informa-
tionMissing disclosures, omitted risks,
incomplete analysis
UnderstandabilityClarity of presentation Jargon usage, logical structure,
transparency
RelevanceDecision-usefulness Predictive value, confirmatory
value, materiality
Each dimension is evaluated by a specialized judge agent. For example, the Reliability judge
utilizes the following system prompt:
“You are an expert financial analyst and auditor. Your primary task is to critically
evaluate the Reliability of a given financial report. Approach the report with profes-
sional skepticism. Do not accept any data, calculation, or description at face value.
Your objective is to verify, not just read.”
6

3.3.2 Multi-Model Consensus
To mitigate individual model biases, each dimension is evaluated by multiple LLMs. The scores
and corresponding rationales from various models are aggregated, and the median value is em-
ployed as the aggregated dimensional score to mitigate the influence of outlier models.
3.4 Human-in-the-Loop Feedback System
FROAV’s feedback system enables systematic collection of human judgments aligned with auto-
mated evaluation dimensions. The database schema enables direct correlation between human
and automated assessments, allowing researchers to identify biases, calibrate prompts, and study
inter-annotator agreement.
3.5 Extensibility and Customization
FROAV’s layered architecture supports multiple levels of customization. Researchers can modify
workflows through the visual interface (swapping LLM providers, adjusting retrieval parameters)
or extend functionality via the FastAPI server.
4 Discussion
4.1 Accessibility Benefits
FROAV significantly lowers barriers to LLM agent research for:
•Domain Experts: Visual workflow design eliminates coding requirements.
•ML Researchers: Reduced boilerplate enables focus on algorithmic innovation.
•Research Teams: Shared infrastructure facilitates collaboration between experts and
engineers.
4.2 Flexibility Trade-offs
FROAV’s design explicitly balances accessibility and power (Table 3).
Table 3: Flexibility Trade-offs
Level Accessibility Capability Use Case
Visual (n8n) Very High Standard work-
flowsPrompt iteration,
model comparison
Low-Code (JS) High Custom logic Specialized aggre-
gation
Full-Code (FastAPI) Medium Unlimited Novel preprocess-
ing, custom mod-
els
4.3 Limitations and Future Work
Current limitations include the restricted scope of the four-dimension framework and the lack of
enterprise-scale stress testing. Future work involves developing an LLM-as-Judges methodology
applied to financial SEC filings, calibrating LLM scoring with human experts using FROAV.
This implementation is projected to significantly reduce the time required to achieve robust
results.
7

5 Conclusion
FROAV represents a significant advancement in research infrastructure for LLM agent develop-
ment and evaluation. By integrating visual workflow design, comprehensive evaluation frame-
works, and human-in-the-loop feedback within a unified platform, FROAV democratizes access
to sophisticated agent research capabilities.
References
[1] Brown, T. B., et al. (2020). Language models are few-shot learners.Advances in Neural
Information Processing Systems, 33, 1877-1901.
[2] Clark, P., et al. (2021). TruthfulQA: Measuring how models mimic human falsehoods.arXiv
preprint arXiv:2109.07958.
[3] Du, Y., et al. (2023). Improving factuality and reasoning in language models through mul-
tiagent debate.arXiv preprint arXiv:2305.14325.
[4] Fu, J., et al. (2023). GPTScore: Evaluate as you desire.arXiv preprint arXiv:2302.04166.
[5] Gao, Y., et al. (2023). Retrieval-augmented generation for large language models: A survey.
arXiv preprint arXiv:2312.10997.
[6] Harrison, C. (2022). LangChain: Building applications with LLMs through composability.
GitHub Repository.
[7] Izacard, G., & Grave, E. (2021). Leveraging passage retrieval with generative models for
open domain question answering.EACL 2021.
[8] Jiang, Z., et al. (2023). Active retrieval augmented generation.arXiv preprint
arXiv:2305.06983.
[9] Lewis, P., et al. (2020). Retrieval-augmented generation for knowledge-intensive NLP tasks.
Advances in Neural Information Processing Systems, 33, 9459-9474.
[10] Liu, J. (2022). LlamaIndex: A data framework for LLM applications.GitHub Repository.
[11] Liu, Y., et al. (2023). G-Eval: NLG evaluation using GPT-4 with better human alignment.
arXiv preprint arXiv:2303.16634.
[12] OpenAI. (2023). GPT-4 Technical Report.arXiv preprint arXiv:2303.08774.
[13] Ouyang, L., et al. (2022). Training language models to follow instructions with human
feedback.Advances in Neural Information Processing Systems, 35, 27730-27744.
[14] Ribeiro, M. T., et al. (2020). Beyond accuracy: Behavioral testing of NLP models with
CheckList.ACL 2020.
[15] Trivedi, H., et al. (2023). Interleaving retrieval with chain-of-thought reasoning for
knowledge-intensive multi-step questions.ACL 2023.
[16] Verga, P., et al. (2024). Replacing judges with juries: Evaluating LLM generations with a
panel of diverse models.arXiv preprint arXiv:2404.18796.
[17] Wang, L., et al. (2024). A survey on large language model based autonomous agents.Fron-
tiers of Computer Science, 18(6), 186345.
[18] Zheng, L., etal.(2023).JudgingLLM-as-a-judgewithMT-BenchandChatbotArena.arXiv
preprint arXiv:2306.05685.
8