# Saarthi for AGI: Towards Domain-Specific General Intelligence for Formal Verification

**Authors**: Aman Kumar, Deepak Narayan Gadde, Luu Danh Minh, Vaisakh Naduvodi Viswambharan, Keerthan Kopparam Radhakrishna, Sivaram Pothireddypalli

**Published**: 2026-03-03 17:30:36

**PDF URL**: [https://arxiv.org/pdf/2603.03175v1](https://arxiv.org/pdf/2603.03175v1)

## Abstract
Saarthi is an agentic AI framework that uses multi-agent collaboration to perform end-to-end formal verification. Even though the framework provides a complete flow from specification to coverage closure, with around 40% efficacy, there are several challenges that need to be addressed to make it more robust and reliable. Artificial General Intelligence (AGI) is still a distant goal, and current Large Language Model (LLM)-based agents are prone to hallucinations and making mistakes, especially when dealing with complex tasks such as formal verification. However, with the right enhancements and improvements, we believe that Saarthi can be a significant step towards achieving domain-specific general intelligence for formal verification. Especially for problems that require Short Term, Short Context (STSC) capabilities, such as formal verification, Saarthi can be a powerful tool to assist verification engineers in their work. In this paper, we present two key enhancements to the Saarthi framework: (1) a structured rulebook and specification grammar to improve the accuracy and controllability of SystemVerilog Assertion (SVA) generation, and (2) integration of advanced Retrieval Augmented Generation (RAG) techniques, such as GraphRAG, to provide agents with access to technical knowledge and best practices for iterative refinement and improvement of outputs. We also benchmark these enhancements for the overall Saarthi framework using challenging test cases from NVIDIA's CVDP benchmark targeting formal verification. Our benchmark results stand out with a 70% improvement in the accuracy of generated assertions, and a 50% reduction in the number of iterations required to achieve coverage closure.

## Full Text


<!-- PDF content starts -->

Saarthi for AGI: Towards Domain-Specific General
Intelligence for Formal Verification
Aman Kumar1, Deepak Narayan Gadde2, Luu Danh Minh3,
Vaisakh Naduvodi Viswambharan2, Keerthan Kopparam Radhakrishna2, Sivaram Pothireddypalli1
1Infineon Technologies India Private Limited, India
2Infineon Technologies Dresden AG & Co. KG, Germany
3Infineon Technologies Vietnam Company Ltd., Vietnam
Abstract
Saarthi [1] is an agentic AI framework that uses multi-agent collaboration to perform end-to-end formal verification.
Even though the framework provides a complete flow from specification to coverage closure, with around40 %efficacy,
there are several challenges that need to be addressed to make it more robust and reliable. Artificial General Intelligence
(AGI) is still a distant goal, and current Large Language Model (LLM)-based agents are prone to hallucinations and making
mistakes, especially when dealing with complex tasks such as formal verification. However, with the right enhancements
and improvements, we believe that Saarthi can be a significant step towards achieving domain-specific general intelligence
for formal verification. Especially for problems that require Short Term, Short Context (STSC) capabilities, such as formal
verification, Saarthi can be a powerful tool to assist verification engineers in their work. In this paper, we present two key
enhancements to the Saarthi framework: (1) a structured rulebook and specification grammar to improve the accuracy
and controllability of SystemVerilog Assertion (SV A) generation, and (2) integration of advanced Retrieval Augmented
Generation (RAG) techniques, such as GraphRAG [2], to provide agents with access to technical knowledge and best
practices for iterative refinement and improvement of outputs. We also benchmark these enhancements for the overall
Saarthi framework using challenging test cases from NVIDIA’s CVDP benchmark [3] targeting formal verification. Our
benchmark results stand out with a70 %improvement in the accuracy of generated assertions, and a50 %reduction in
the number of iterations required to achieve coverage closure.
Index Terms
Generative AI, LLM, Agentic AI, AGI, Formal Verification, Saarthi
I. INTRODUCTION
The increasing complexity, configurability, and safety-criticality of modern semiconductor designs have intensified demands
on Artificial Intelligence (AI) workflows. Industry-standard methodologies rely on expert engineers to interpret specifications,
distill them into verification plans, author and refine SV As, and iteratively close proof, coverage, and vacuity gaps. While
Formal Verification (FV) offers exhaustive guarantees, it remains labor-intensive and difficult to scale. Advances in LLMs and
agentic AI suggest potential for automating substantial portions of this pipeline [1], [4]. However, limitations such as syntactic
instability, semantic misinterpretation, and shallow reasoning persist [5], [6].
Emerging work on multi-agent orchestration [1], [7] demonstrates that decomposing FV into specialized roles improves
robustness. Yet, gaps remain in assertion synthesis controllability, grounding agent reasoning in authoritative corpora, and
adaptive feedback mechanisms. Addressing these gaps is essential for progress toward Domain-Specific General Intelligence
(DSGI) in FV—a tractable milestone on the trajectory toward AGI [8], [9].
RAG [10] mitigates hallucinations and increases factual precision by conditioning generation on retrieved evidence. In
Electronic Design Automation (EDA), authoritative sources include IEEE SystemVerilog standards [6], ISA specifications [11],
and prior assertion libraries. GraphRAG [2] extends RAG by retrieving structured subgraphs, enabling multi-hop consistency
checks and traceability.
This paper advances AI-driven FV through: (1) a structured rulebook and specification grammar for SV A generation; (2)
integration of advanced RAG techniques, such as GraphRAG [2]; (3) automated coverage hole-filling; (4) Human-in-the-Loop
(HIL) data collection pipelines; and (5) benchmarking on diverse designs. Enhancements yield up to a70 %increase in assertion
accuracy and a50 %reduction in iterations to coverage closure. By coupling controllable generation with structured knowledge
grounding and experiential learning, Saarthi narrows the reliability gap in domain-specialized cognitive workflows, advancing
DSGI.
II. BACKGROUND
This paper represents a significant step toward achieving AGI in the domain of formal verification. By leveraging the
capabilities of LLMs and integrating advanced techniques such as multi-agent collaboration, structured rulebooks, and retrieval-
augmented generation, we aim to address the challenges of automating complex verification workflows. Unlike prior work thatarXiv:2603.03175v1  [cs.AI]  3 Mar 2026

highlighted the limitations of LLM-generated outputs [5], our approach focuses on iterative refinement, grounding in author-
itative knowledge, and systematic feedback loops to enhance reliability and accuracy. Use cases, such as formal verification,
that require STSC capabilities for LLMs are well-suited to the path toward achieving AGI [12]. These efforts align with the
broader vision of DSGI, where domain-specific intelligence can tackle intricate tasks like formal verification, paving the way
for scalable and robust AI-driven engineering solutions.
A. Retrieval-Augmented Generation (RAG)
RAG integrates information retrieval with neural text generation, enabling models to condition outputs on external sources
for accurate, context-aware results [10], [13]. In hardware verification, engineers consult standards like IEEE SystemVerilog
[6], Universal Verification Methodology (UVM) [14], and ISA manuals [11]. Grounding responses in cited passages, RAG
improves factuality and reduces hallucinations, supporting traceable workflows [15], [16].
User  Query
ResponseInformation
Extraction
Raw Data
Sour cesChunkingEmbeddingEmbedding
Additional
ContextAgentsVector  DB
 (e.g, Chr omaDB,
FAISS DB)
Relevant
Data
Data Pr eparation (Append) Query1  0  0
0  1  0
0  0  11  0  0
0  1  0
0  0  1
Fig. 1: Overview of the basic RAG pipeline [17]
A RAG pipeline combines offline indexing of domain corpora with hybrid retrieval and grounded generation. The corpus
includes standards, repositories, and logs. Documents are chunked, embedded, and stored in an index (e.g., Facebook AI
Similarity Search (FAISS)) [18] as shown in Figure 1. At query time, a hybrid retriever pairs dense retrieval with lexical
matching, reranking top candidates for precision [19]. The generator produces grounded outputs (e.g., SV As with citations)
[10]. Enhancements like query reformulation [20] and iterative reasoning [21] further improve results.
B. Knowledge Graph
A Knowledge Graph (KG) represents entities as nodes and their relationships as edges, structured by a schema or ontology
[22], [23]. KGs can be implemented as Resource Description Framework (RDF) triples or property graphs (e.g., Neo4j) and
accessed via query languages like SPARQL or Cypher [24], [25], [26]. They enable precise querying, inference, and integration
with vector-based similarity measures [22], [27].
Standard KG workflows unify diverse data sources (e.g., documents, logs) into a graph through entity and relationship
extraction, schema alignment, and enhancement [22], [27]. KGs support multi-hop queries, provenance tracing, and analytics,
facilitating traceability in hardware verification [6], [14], [15], [28].
In verification, KG-based traceability links requirements to Register Transfer Level (RTL) modules, assertions, tests, and
coverage, enabling queries like identifying unvalidated requirements or correlating failures with design hierarchies. GraphRAG
enhances retrieval by leveraging KG topology for multi-hop reasoning, improving factual consistency and recall on long-horizon
dependencies [10], [29].

C. GraphRAG
GraphRAG is a RAG approach that uses a knowledge graph as its evidence source, as illustrated in Figure 2. The pipeline
begins by processing a corpus of unstructured documents to construct a knowledge graph, where nodes represent formal
verification artifacts (requirements, signals, properties, proofs, counterexamples) and edges capture relations (e.g.,implements,
constrains,proven by,violated by). When a query arrives, the system retrieves a relevant subgraph containing interconnected
information rather than isolated text fragments. This structured evidence is then integrated into the prompt context for the LLM.
The model leverages both the query and the graph-derived knowledge to generate responses grounded in relational reasoning,
enabling multi-hop inference and producing more accurate, contextually coherent answers than traditional chunk-based retrieval.
QueryBuild
Graph
Adjusted Pr ompt
(with knowledge
graph)AgentRelevant
InformationGraph Corpus
Response
Fig. 2: Knowledge graph construction and retrieval in GraphRAG pipeline [30]
Example:Suppose you ask, “How should I check thatAXI WLASTmatches theAWLENburst length?”. The graph links
AWLENto the rule burst length =AWLEN+ 1,WLASTto “asserted on the last data beat,” and the handshake relation that each
WV ALID && WREADYadvances the beat count. Graph RAG retrieves this subgraph and produces a simple check:
1 A f t e r AWVALID && AWREADY handshake
2 Count data − b e a t h a n d s h a k e s
3 Check : WLAST = 1 on (AWLEN + 1) − t h b e a t
4 WLAST = 0 b e f o r e t h a t
5 Assume : AWLEN s t a b l e d u r i n g b u r s t
The answer cites the specific nodes (AWLEN,WLAST, handshake rule), preserving provenance for auditability.
III. SAARTHI: AGENTICAI-BASEDFORMALVERIFICATION
The framework presented in [7] [1] laid the groundwork for AI-driven formal verification, emphasizing agent coordination,
EDA tool integration, and the potential for explainable AI in hardware design workflows. Extending this foundation, the present
work advances the methodology through three primary directions. First, a structured rulebook and specification grammar
are introduced to enhance the accuracy and controllability of SV A generation. Second, advanced RAG approaches, such as
GraphRAG [2], are incorporated to improve the agents’ access to technical knowledge and iterative reasoning capabilities.
Third, an automated coverage hole–filling mechanism is proposed to identify and generate targeted assertions that address
unverified design regions, thereby accelerating coverage closure. Together, these enhancements significantly strengthen the
robustness and practical applicability of the Saarthi framework, as validated through benchmarks on NVIDIA’s Comprehensive
Verilog Design Problems (CVDP) dataset [3].
To realize our contributions and conduct the experiments, we implemented the flow shown in Figure 3. Upon task assignment,
AI agents assume primary control of the verification workflow. Saarthi facilitates formal verification through a multi-agent,
agentic AI approach that coordinates specialized agents. The framework incorporates design patterns for agentic reasoning and
safeguards to mitigate context limitations, hallucinations, and repetitive loops. Saarthi is built on Microsoft AutoGen, leveraging
its multi-agent orchestration capabilities to support formal verification. Its architecture provides a configurable orchestration
layer that can be tailored to diverse verification requirements while preserving process consistency and reliability.
A. Agent Orchestration
The orchestrator supports both sequential and hierarchical execution; for this formal verification process, agents are arranged
sequentially. After orchestration, the selected framework’s main module initializes the verification run and invokes the agents
in order.

During this process, the agents generate key artifacts such as Verification Plan (vPlan) and properties, logging their interactions
and collaborations as they proceed. Generated properties are evaluated by critic agents, who provide feedback to improve the
accuracy and correctness of SV As. This iterative loop continues until a convergence threshold is reached. If the agents cannot
finalize the SV As within a predefined iteration limit, human intervention (i.e., HIL) is triggered for further assessment. Once
finalized, the SV As undergo formal verification, and any Counter Examples (CEXs) are identified, and resolved by the agents.
B. Multi-Agent Workflow for AI-Driven Formal Verification
Saarthi Agents 
coverage agentsproperty generation
agents
syntax validatorsyntax err or fixing
agents
assertion integritycheckerassertion compliancereviewerassertion author
assertion validation executorsyntax analyzer
code extractorformal verification lead
location analyzer
SVA propertyengineercode fixer
Fig. 3: Saarthi: Agentic AI based formal verification using multi-agent collaboration
To implement our contributions and execute our experiments, we developed the multi-agent workflow illustrated in Figure 3.
The framework orchestrates the formal verification process through a coordinator/lead that manages the end-to-end formal
verification plan for the Design Under Verification (DUV).
Property-generation agentssystematically translate functional requirements and protocol rules into candidate SV As, an-
notating each assertion with metadata including the target module, interface, and signal bindings.Syntax-error-fixing agents
sanitize these candidates through a three-stage pipeline: a syntax analyzer detects parser and lint violations, a code fixer applies
canonical rewrites and enforces naming conventions, and a syntax validator verifies compliance against tool-chain compilers.
Coverage agentsperform gap analysis and strategic placement: a location analyzer identifies optimal insertion points within
the RTL hierarchy, while an SV A property engineer synthesizes additional assertions to address uncovered behaviors. This
decomposition enables concurrent progress across property authoring, correction, and placement phases, with clear ownership
boundaries and coordinator/lead oversight.
A code extractor/manager mediates interactions among agents, tools, and human reviewers. Its primary function is to
materialize agent-generated code and update the design repository and/or verification environment with the extracted artifacts.
The extractor returns integration-ready deliverables and performs binding into the verification environment or specific modules.
Coverage feedback and tool diagnostics drive the next iteration. Coverage agents analyze reports to identify gaps and missing
checks, then issue new property requests with target modules and signals. Compile errors and vacuity warnings are routed
back to the syntax-fixing loop for correction and revalidation.
Each cycle repairs the code, normalizes naming, updates the code index/graph, and re-runs proofs. This reduces authoring
time, minimizes pre-run syntax churn, accelerates coverage closure, maintains correct assertion placement, and preserves
provenance for auditability and reproducibility.
C. Dataset Collection during Human-in-the-Loop (HIL) Refinement
During the HIL phase, we systematically collect the validated response set produced following human interventions as shown
in Figure 4. Each corrected or approved agent output is logged into a structured dataset that records the refined response, its
originating prompt and context, observed error signatures (e.g., parse failures, vacuity, misbinding), and the applied resolution

HIL
Response
Captur eSaarthi's Agents
property generation agents
syntax error fixing agents
coverage agentsFig. 4: Data Collection Flow during HIL
pattern. This collection ensures that every human-validated example contributes to a growing corpus of high-quality artifacts
reflecting real failure recoveries. The resulting dataset underpins iterative improvement enabling prompt adjustments, template
extensions for previously uncovered properties, and targeted repair of syntactic and semantic defects identified during the
HIL cycle. By continuously enriching this corpus, the framework builds a self-improving feedback base that increases agents’
reliability and reduces the need for manual intervention over time.
Furthermore, the framework supports standalone invocation of individual agents without executing the full workflow to
address specific use cases or targeted applications.
D. Rulebooks
Inspired by the BugGen fault injection methodology [31], we propose a structured rulebook to standardize specification
grammar in assertion generation workflows. This rulebook bridges ambiguous natural language descriptions and machine-
readable formats, enhancing automation and interpretability.
Unlike prior approaches relying on natural language instructions, our methodology encodes specifications as concise key-
words, creating a deterministic pipeline for assertion generation. This uniform representation improves predictability and
facilitates debugging by human engineers.
Learning Cache: Documented Errors and Resolutions
Specification in keywords:
Signals: [clk, req, ack, error]
Property: [assert, concurrent, positive edge of clk]
Condition: [if req is high, then ack must be high within 2 cycles unless error is high]
Incorrect property:✖
property start_done_within_3_cycles_unless_reset;
@(negedge clk) start |-> $####$[1:3] done unless reset;
endproperty
assert property (start_done_within_3_cycles_unless_reset);
Explanation:
1. Uses negedge clk instead of posedge clk
2. Specifies 3-cycle window ####[1:3]) instead of 2-cycle ####[1:2])
3. Misinterprets logic; done unless reset should be ack within 2 cycles unless error
Corrected property:✔
property req_ack_unless_error;
@(posedge clk) req |-> (error or ####[1:2] ack);
endproperty
assert property (req_ack_unless_error);
Fig. 5: A snippet of the cached mistakes that all agents can learn from for future iterations
The rulebook includes best practices and common pitfalls, documented with corrections and explanations (Figure 5). These
insights assist agents and engineers in avoiding recurring errors. The rulebook is transferable across hardware designs, allowing
design-specific details to be appended.
Figure 6 illustrates the impact of structured grammar. Without it, manager agent notes vary, causing instability. Structured
grammar ensures consistent assertion generation, enabling engineers to refine specifications rather than adjust prompts. Cached
mistakes, such as handling reset conditions in assertions, are reused to enhance reliability and reduce debugging efforts.

Input stage
Specification:
Done Signal Validity: When ’i_start’ is asserted and
’enc_done’ is not, then ’o_done’ must deassert in the
next simulation tick.Input stage
Specification:
Done Signal Validity: When ’i_start’ is asserted and
’enc_done’ is not, then ’o_done’ must deassert in the
next simulation tick.
Intermediate stage (without rulebook enhancement)
Specification breakdown:
Trigger: ’i_start 1’ && ’enc_done 0’
Condition: ’o_done == 0’ in the next cycle
Scope: Immediate (next cycle after the trigger)
Generated property:✖
property done_signal_validity;
@(posedge clk)(i_start && !enc_done) |=> (!o_done);
endproperty
assert property (done_signal_validity) else
$error("o_done did not deassert as expected.");
Suggested improvement:
Use a stable condition for ’i_start’ to avoid false
negatives:
property done_signal_stability;
@(posedge clk) ($stable(i_start) && i_start &&
!enc_done)
|=> (!o_done);
endpropertyIntermediate stage (with rulebook enhancement)
Specification in keywords:
Signals: [clk, i_start, enc_done, o_done]
Property: [assert, concurrent, positive edge of clk]
Condition: [if i_start is high and enc_done is low,
o_done must be low in next cycle]
Generated property:✔
property done_signal_validity;
@(posedge clk)(i_start && !enc_done) |=> (!o_done);
endproperty
assert property (done_signal_validity);
Suggested improvement:
Add ’disable iff (!rst_async_n)’ to prevent assertion
failures during reset
property done_signal_validity;
@(posedge clk) disable iff (!rst_async_n)
(i_start && !enc_done) |=> !o_done;
endproperty
Output stage (without rulebook enhancement)
Final property:✖
property done_signal_stability;
@(posedge clk) ($stable(i_start) && i_start &&
!enc_done) |=> (!o_done);
endproperty
assert property (done_signal_stability) else
$error("o_done did not deassert with stable i_start
and enc_done == 0.");Output stage (With rulebook enhancement)
Final property:✔
property done_signal_validity;
@(posedge clk) disable iff (!rst_async_n) (i_start &&
!enc_done) |=> !o_done;
endproperty
assert property (done_signal_validity);
Fig. 6: A comparison of the iterative SV A generation process before and after adding the rulebook
Ultimately, the rulebook improves accuracy, controllability, and collaboration in SV A generation, creating a robust and
scalable solution.
E. Root Cause Analysis (RCA)
Diagnosing and resolving CEXs is a critical bottleneck in formal verification. To address this, we deploy multi-agent RCA
agents that analyze failures and propose corrections collaboratively, as shown in Figure 7.
Upon property violation, the formal engine generates Value Change Dump (VCD) traces capturing signal activity. These,
along with the specification, RTL sources, and failing SV As, serve as inputs to four specialized agents: the VCD Parser
extracts signal values and timestamps; the Specification–Assertion Analyst verifies property correctness; the RTL Analysis
Agent diagnoses signal dependencies and logic conditions; and the Verification Agent ensures consistency across analyses.
The agents iteratively refine diagnoses, generating structured reports with evidence analysis, bug locations, root cause
classifications, and proposed patches. If unresolved after three iterations, the framework transitions to HIL mode, where
human engineers review and validate the findings. This hybrid model balances automation with expert oversight for complex
cases.
IV. BENCHMARKING ANDRESULTS
To evaluate performance and benchmark capabilities, we used the new Saarthi to verify RTL designs of varying complexity.
Alongside our in-house designs, ECC and Automotive Intellectual Property (IP), we included three publicly available designs:
Memory Scheduler, AXI4Lite, and CIC Decimator, sourced from the NVIDIA CVDP agentic AI benchmark for assertion
generation [32]. For additional variety, we also incorporated a floating-point multiplier design from [33]. The agents utilized
four models: GPT-4.1 and GPT-5 from OpenAI [34] and LLama 3.3 from Meta [35].

VCD filesRTL sour ce
filesSpecification Assertions
VCD Parser
AgentSpecification–
Assertion
Analysis AgentRTL
Analysis AgentRCA  Verification
AgentInput Files
RCA  Agents
OutcomeVCD shows: `wr_en = 1`, `rd_en = 0`, `wr_pointer_next = b00000`, `rd_pointer = b00000`, `full = 0`  
Expected: `full = 1` when `wr_en = 1`, `rd_en = 0`, and `wr_pointer_next == rd_pointer`  
Discrepancy: `full` did not assert when `wr_pointer_next` equaled `rd_pointer`  during a write-only operationEvidence Analysis
File: fifo.sv  
Lines: 159-177 Problematic code: {(wr_en & (~full | rd_en)), (rd_en & ~empty)}
Issue: The condition `wr_en & (~full | rd_en)` incorrectly allows `full` to remain
deasserted when `wr_pointer_next == rd_pointer`.Bug Location
The `full` flag logic does not properly handle the case where `wr_pointer_next` equals `rd_pointer`
during a write-only operation. Root Cause - RTL Bug
Possible Patch
Modify the `full` flag logic to ensure it properly asserts when `wr_pointer_next` equals
`rd_pointer` during a write-only operation: {(wr_en & ~full), (rd_en & ~empty)} Fig. 7: Overview of the comprehensive multi-agent RCA setup
This paper evaluates performance across three comprehensive benchmarks: Key Performance Indicators (KPIs), HIL vs. No
HIL, and fully automated coverage improvement. We adopt the same KPIs from our previous work [1], with two additions:first
generation successandnumber of fix attempts after failure. First generation success measures whether the generated assertions
compile and run correctly on the first attempt, indicating the effectiveness of our assertion rulebooks in minimizing syntax errors.
If the initial attempt fails, the number of fix attempts tracks how many iterations the syntax-fixing agent requires to produce a
valid result, with a maximum of five attempts. The second benchmark compares assertion generation with and without human
feedback. In the HIL setting, after the coverage improvement agent runs and initial results are recorded (see the left side of
Table II), a human engineer provides targeted guidance to refine the model’s output, with final results recorded accordingly. This
setup allows us to assess the impact of human feedback on assertion quality and coverage. The third benchmark evaluates the
model’s ability to autonomously improve coverage over multiple iterations without human intervention. By tracking coverage
and assertion quality across five iterations, we aim to understand the effectiveness and limitations of the coverage improvement
agent in a fully automated setting.
ECC
Automotive
IP
Memory
Scheduler
AXI4LiteCIC
DecimatorFloat
Multiplier708090100
% Proven
ECC
Automotive
IP
Memory
Scheduler
AXI4LiteCIC
DecimatorFloat
Multiplier708090100
% Coverage
GPT-4.1 GPT-5 Llama3.3
Fig. 8: Radar chart for Pass@1 for different designs
The results in Table I present the KPIs of Saarthi. The results show that Saarthi can generate formal assertions for a wide
range of hardware designs. GPT-4.1 performs well on simpler modules, while GPT-5 significantly improves both assertion
proof rates and formal coverage for complex designs like the Float Multiplier. This improvement comes at the cost of higher

TABLE I: Key Performance Indicators of Saarthi on designs of various complexity
Design MetricPass@1 Pass@2 Pass@3
GPT-4.1 GPT-5 Llama3.3 GPT-4.1 GPT-5 Llama3.3 GPT-4.1 GPT-5 Llama3.3
ECC# Assertions 19 28 14 25 32 6 25 48 12
1st generation No Yes No Yes Yes Yes No Yes Yes
# attempts to fix 1 0 2 0 0 0 2 0 0
% Proven 71.86% 89.28% 57.14% 64.00% 75.00% 33.33% 64.00% 81.25% 50.00%
% Coverage 62.67% 58.57% 68.01% 60.21% 61.86% 2.18% 58.15% 93.74% 43.90%
Automotive IP# Assertions 36 58 9 64 76 16 45 54 11
1st generation No No No No No No No No No
# attempts to fix 4 3 5 5 3 5 3 3 5
% Proven 50% 84.48% 22.22% 50% 56.57% 18.75% 48.88% 55.55% 45.45%
% Coverage 64.29% 80.99% 6.99% 76.6% 77.02% 8.96% 72.75% 80.38% 42.44%
Memory Scheduler [32]# Assertions 24 22 13 16 28 7 23 29 17
1st generation No No No Yes Yes Yes No No Yes
# attempts to fix 2 2 3 0 0 0 1 1 0
% Proven 35.71% 40.91% 30.77% 50.00% 32.14% 71.43% 21.74% 32.01% 17.65%
% Coverage 42.93% 54.73% 39.47% 43.32% 58.21% 44.20% 46.39% 48.86% 37.30%
AXI4Lite [32]# Assertions 69 139 39 75 87 60 74 117 92
1st generation No No No No No No No No No
# attempts to fix 5 3 5 5 3 5 3 3 5
% Proven 46.37% 72.66% 64.10% 40% 68.96% 45.00% 62.16% 61.15% 23.90%
% Coverage 31.29% 44.06% 32.83% 35% 50% 36.42% 41.51% 41.18% 37.16%
CIC Decimator [32]# Assertions 16 36 10 16 37 10 19 30 10
1st generation Yes Yes No Yes No Yes Yes Yes No
# attempts to fix 0 0 1 0 1 0 0 0 1
% Proven 62.5% 75% 40% 31.25% 91.89% 30% 52.63% 86.67% 30%
% Coverage 66.67% 74.45% 50% 52.14% 77.39% 18.22% 56.67% 72.86% 48.42%
Float Multiplier# Assertions 19 49 10 17 42 0 23 57 15
1st generation Yes Yes Yes Yes Yes No Yes No Yes
# attempts to fix 0 0 0 0 0 3 0 2 0
% Proven 10.53% 51.02% 20% 23.53% 30.95% 0% 17.39% 42.11% 6.67%
% Coverage 4.97% 65.37% 12.14% 19.96% 83.71% 0% 9.44% 78.48% 3.15%
ECC
Automotive
IP
Memory
Scheduler
AXI4LiteCIC
DecimatorFloat
Multiplier708090100
% Proven
ECC
Automotive
IP
Memory
Scheduler
AXI4LiteCIC
DecimatorFloat
Multiplier708090100
% Coverage
GPT-4.1 GPT-5 Llama3.3
Fig. 9: Radar chart for Pass@2 for different designs
latency, as GPT-5 uses a reasoning-based approach that requires more processing time. This trade-off between accuracy and
speed is critical for practical deployment. Additionally, specification quality strongly impacts results: simple specs work for
basic designs (e.g., ECC, CIC decimator), but vague natural language descriptions in complex RTL lead to divergence from
ideal assertions. For example, the entire CVDP Memory Scheduler spec contained only a few lines of expected states. In

ECC
Automotive
IP
Memory
Scheduler
AXI4LiteCIC
DecimatorFloat
Multiplier708090100
% Proven
ECC
Automotive
IP
Memory
Scheduler
AXI4LiteCIC
DecimatorFloat
Multiplier708090100
% Coverage
GPT-4.1 GPT-5 Llama3.3Fig. 10: Radar chart for Pass@3 for different designs
real-world scenarios, continuous collaboration between design and verification teams is essential, so despite minimal guidance
and limited specification, we believe the LLMs achieved impressive outcomes.
TABLE II: Coverage improvement with and without HIL at Pass@1
Design MetricWithout HIL With HIL
GPT-4.1 GPT-5 Llama3.3 GPT-4.1 GPT-5 Llama3.3
ECC# Assertions 53 66 19 53 60 19
% Proven 84.90% 78.80% 68.42% 96.06% 80.00% 84.21%
% Coverage 89.31% 92.83% 90.63% 98.05% 95.22% 92.53%
Automotive IP# Assertions 36 58 19 36 58 19
% Proven 50% 84.48% 26.31% 72.22% 93.10% 63.15%
% Coverage 64.29% 80.99% 38.87% 73.91% 85.94% 64.91%
Memory Scheduler# Assertions 27 35 36 27 35 36
% Proven 62.96% 48.57% 33.33% 85.19% 57.14% 55.56%
% Coverage 42.35% 50.00% 41.11% 61.54% 62.02% 45.65%
AXI4Lite# Assertions 101 139 3 101 139 3
% Proven 70.29% 72.66% 33.33% 85.10% 83.45% 66.66%
% Coverage 38.90% 44.06% 28.23% 61.56% 67.06% 31.12%
CIC Decimator# Assertions 28 53 28 28 53 28
% Proven 67.86% 81.13% 50% 100% 100% 100%
% Coverage 79.69% 73.14% 55.64% 91.20% 91.60% 91.32%
Float Multiplier# Assertions 88 126 10 88 126 10
% Proven 79.55% 61.91% 20% 100% 100% 100%
% Coverage 27.46% 64.72% 12.23% 67.57% 92.83% 74.68%
Table II illustrates the impact of HIL collaboration in reducing the challenges of interpreting natural language specifications.
By improving specification visibility and providing timely feedback, the models achieved higher assertion quality and coverage
compared to the initial results. This interaction not only improved accuracy but also significantly reduced the time required,
transforming what would typically take hours of manual effort into a much faster, collaborative process. These findings indicate

that, even in their current state, LLMs can substantially enhance productivity by serving as intelligent assistants, allowing
engineers to focus on complex reasoning tasks while delegating repetitive steps to AI.
TABLE III: Coverage improvement without HIL across iterations (GPT-5)
Design MetricCoverage improvement without HIL
1stIter 2ndIter 3rdIter 4thIter 5thIter
ECC# Assertions 312 318 1258 N/A N/A
% Proven 91.67% 92.12% 97.93% N/A N/A
% Coverage 88.89% 86.86% 87.87% N/A N/A
Automotive IP# Assertions 59 71 134 185 233
% Proven 81.35% 77.46% 85.82% 87.75% 89.69%
% Coverage 80.56% 79.95% 75.14% 74.31% 72.69%
Memory Scheduler# Assertions 35 36 37 37 37
% Proven 40.00% 41.12% 43.24% 42.50% 43.80%
% Coverage 55.64% 56.10% 55.85% 56.40% 55.20%
AXI4Lite# Assertions 104 145 165 199 232
% Proven 62.50% 58.62% 59.39% 59.29% 59.99%
% Coverage 45.36% 45.36% 47.21% 52.99% 48.93%
CIC Decimator# Assertions 53 63 86 116 203
% Proven 81.13% 79.37% 80.23% 81.90% 80.30%
% Coverage 73.14% 74.77% 72.34% 72.44% 64.37%
Float Multiplier# Assertions 126 148 201 244 264
% Proven 61.91% 57.43% 67.66% 60.25% 58.71%
% Coverage 64.72% 71.60% 75.09% 76.84% 82.45%
Table III presents the progression of assertion quality and coverage across multiple iterations of the coverage improvement
agent operating in a fully autonomous setting. Some designs, such as Automotive IP and Float Multiplier, show steady
improvements in assertion count and coverage, whereas others, like Memory Scheduler and AXI4Lite, reach early saturation,
with minimal gains beyond the initial iterations. These trends indicate that repeated autonomous runs do not always provide
significant benefits, particularly when early iterations already achieve stable results. Interestingly, GPT-5 was able to generate
thousands of assertions in a single run for ECC before crashing due to the large token requirements. This result highlights the
practical limits of GPT-5 when handling extremely large assertion sets, even for moderately complex designs. In practice, a
more efficient approach is to perform one autonomous run and then apply HIL refinement to close remaining gaps.
ECC Automotive
IPMemory
SchedulerAXI4Lite CIC
DecimatorFloat
Multiplier5060708090100Pass@1 (%)
% Proven without HIL
ECC Automotive
IPMemory
SchedulerAXI4Lite CIC
DecimatorFloat
Multiplier5060708090100Pass@1 (%)
% Proven with HILGPT-4.1 GPT-5 Llama3.3
Fig. 11: % Proven comparison with and without HIL at Pass@1
V. CONCLUSION
The increasing complexity and safety-critical demands of semiconductor design have intensified the need for scalable,
reliable AI-driven formal verification. While multi-agent collaboration and orchestration offer promising directions, challenges

ECC Automotive
IPMemory
SchedulerAXI4Lite CIC
DecimatorFloat
Multiplier5060708090100Pass@1 (%)
% Coverage improvement without HIL
ECC Automotive
IPMemory
SchedulerAXI4Lite CIC
DecimatorFloat
Multiplier5060708090100Pass@1 (%)
% Coverage improvement with HILGPT-4.1 GPT-5 Llama3.3Fig. 12: Coverage comparison with and without HIL at Pass@1
in controllability and adaptive learning remain. Saarthi addresses these limitations by combining structured assertion generation,
domain-specific knowledge retrieval, and iterative coverage refinement to move toward DSGI in formal verification. Across
six RTL designs, Saarthi demonstrated consistent improvements in assertion quality, proof convergence, and coverage. In
particular, on NVIDIA CVDP designs such as AXI4Lite and CIC Decimator, the system achieved over 50% and 77%
coverage, respectively, for first generation, with further gains observed through human-in-the-loop refinement. GPT-5 delivered
the strongest results, especially on complex designs, though its advanced reasoning introduced moderate latency compared
to smaller models. These findings validate the effectiveness of our rulebook-guided generation, hybrid retrieval strategies,
and feedback-driven repair mechanisms. Similar to semiconductors, Moore’s Law-like trends apply to AI: over time, access
becomes faster, cheaper, and more widespread. As benchmarking continues, Saarthi offers a concrete and practical step toward
more dependable and efficient AI integration towards DSGI.
ACKNOWLEDGEMENT
Under grant 101194371, Rigoletto is supported by the Chips Joint Undertaking and its members, including the top-up funding
by the National Funding Authorities from involved countries.
Rigoletto is also funded by the Federal Ministry of Research, Technology and Space under the funding code 16MEE0548S.
The responsibility for the content of this publication lies with the author.
REFERENCES
[1] A. Kumar et al.,Saarthi: The first ai formal verification engineer, 2025. arXiv: 2502 . 16662[cs.AI]. [Online].
Available: https://arxiv.org/abs/2502.16662
[2] D. Edge et al.,From local to global: A graph rag approach to query-focused summarization, 2025. arXiv: 2404.16130
[cs.CL]. [Online]. Available: https://arxiv.org/abs/2404.16130
[3] N. Pinckney et al.,Comprehensive verilog design problems: A next-generation benchmark dataset for evaluating large
language models and agents on rtl design and verification, 2025. arXiv: 2506.14074[cs.LG]. [Online]. Available:
https://arxiv.org/abs/2506.14074
[4] A. Kumar et al.,Generative AI Augmented Induction-based Formal Verification, 2024. arXiv: 2407.18965[cs.AI].
[Online]. Available: https://arxiv.org/abs/2407.18965
[5] D. N. Gadde et al.,All Artificial, Less Intelligence: GenAI through the Lens of Formal Verification, 2024. arXiv:
2403.16750[cs.AI]. [Online]. Available: https://arxiv.org/abs/2403.16750
[6] “Ieee standard for systemverilog–unified hardware design, specification, and verification language,”IEEE Std 1800-2023
(Revision of IEEE Std 1800-2017), pp. 1–1354, 2024.DOI: 10.1109/IEEESTD.2024.10458102
[7] D. N. Gadde et al.,Hey ai, generate me a hardware code! agentic ai-based hardware design & verification, 2025. arXiv:
2507.02660[cs.AI]. [Online]. Available: https://arxiv.org/abs/2507.02660
[8] S. Bubeck et al.,Sparks of Artificial General Intelligence: Early experiments with GPT-4, 2023. arXiv: 2303.12712
[cs.CL]. [Online]. Available: https://arxiv.org/abs/2303.12712

[9] Leopold Aschenbrenner,Situational Awareness: The Decade Ahead, https://situational-awareness.ai/wp-content/uploads/
2024/06/situationalawareness.pdf, Jun. 2024.
[10] P. Lewis et al.,Retrieval-augmented generation for knowledge-intensive nlp tasks, 2021. arXiv: 2005.11401[cs.CL].
[Online]. Available: https://arxiv.org/abs/2005.11401
[11] R.-V . International,Risc-v ratified specifications, Accessed: 2025-11-06, RISC-V International, 2025. [Online]. Available:
https://riscv.org/specifications/ratified/
[12] Q. Xu et al.,Revolution or hype? seeking the limits of large models in hardware design, 2025. arXiv: 2509.04905
[cs.LG]. [Online]. Available: https://arxiv.org/abs/2509.04905
[13] G. Izacard et al.,Leveraging passage retrieval with generative models for open domain question answering, 2021. arXiv:
2007.01282[cs.CL]. [Online]. Available: https://arxiv.org/abs/2007.01282
[14] “Ieee standard for universal verification methodology language reference manual,”IEEE Std 1800.2-2020 (Revision of
IEEE Std 1800.2-2017), pp. 1–458, 2020.DOI: 10.1109/IEEESTD.2020.9195920
[15] I. 2. 32, “Road vehicles – functional safety – part 1: V ocabulary (iso 26262-1:2018),” International Organization for
Standardization, Geneva, Switzerland, Tech. Rep. ISO 26262-1:2018, 2018, Accessed: 2025-11-06. [Online]. Available:
https://www.iso.org/standard/68383.html
[16] I. (.-1. RTCA, “Design assurance guidance for airborne electronic hardware (do-254),” RTCA, Inc., Washington, DC,
USA, Tech. Rep. DO-254, 2000, Issue Date: 19 April 2000; Accessed: 2025-11-06. [Online]. Available: https://www.
rtca.org/publication/do-254
[17] C. Macpherson,Implementing RAG in LangChain with Chroma: A Step-by-Step Guide — callumjmac, https://medium.
com/@callumjmac/implementing- rag- in- langchain- with- chroma- a- step- by- step- guide- 16fc21815339, [Accessed
10-11-2025].
[18] J. Johnson et al.,Billion-scale similarity search with gpus, 2017. arXiv: 1702.08734[cs.CV]. [Online]. Available:
https://arxiv.org/abs/1702.08734
[19] V . Karpukhin et al.,Dense passage retrieval for open-domain question answering, 2020. arXiv: 2004.04906[cs.CL].
[Online]. Available: https://arxiv.org/abs/2004.04906
[20] L. Gao et al.,Precise zero-shot dense retrieval without relevance labels, 2022. arXiv: 2212.10496[cs.IR]. [Online].
Available: https://arxiv.org/abs/2212.10496
[21] S. Yao et al.,React: Synergizing reasoning and acting in language models, 2023. arXiv: 2210.03629[cs.CL]. [Online].
Available: https://arxiv.org/abs/2210.03629
[22] A. Hogan et al., “Knowledge graphs,”ACM Comput. Surv., vol. 54, no. 4, Jul. 2021,ISSN: 0360-0300.DOI: 10.1145/
3447772 [Online]. Available: https://doi.org/10.1145/3447772
[23] L. Ehrlinger et al., “Towards a definition of knowledge graphs,” Sep. 2016.
[24] R. Cyganiak et al., “Rdf 1.1 concepts and abstract syntax,”W3C Proposed Recommendation, Jan. 2014.
[25]SPARQL 1.1 Query Language — w3.org, https://www.w3.org/TR/sparql11-query/, [Accessed 10-11-2025].
[26] E. E. Ian Robinson Jim Webber,Graph Databases, 2nd Edition — oreilly.com, https://www.oreilly.com/library/view/
graph-databases-2nd/9781491930885/, [Accessed 10-11-2025].
[27] P. Cimiano et al., “Knowledge graph refinement: A survey of approaches and evaluation methods,”Semant. Web, vol. 8,
no. 3, pp. 489–508, Jan. 2017,ISSN: 1570-0844.DOI: 10.3233/SW-160218 [Online]. Available: https://doi.org/10.3233/
SW-160218
[28]ED-80 Design Assurance Guidance for Airborne Electronic Hardware - EUROCAE — eurocae.net, https://www.eurocae.
net/product/ed-80-design-assurance-guidance-for-airborne-electronic-hardware/, [Accessed 10-11-2025].
[29]GitHub - microsoft/graphrag: A modular graph-based Retrieval-Augmented Generation (RAG) system — github.com,
https://github.com/microsoft/graphrag, [Accessed 10-11-2025].
[30]Graph RAG: The upgrade that traditional RAG needed — rabiloo.com, https://rabiloo.com/blog/graph-rag-the-upgrade-
that-traditional-rag-needed, [Accessed 10-11-2025].
[31] S. Jasper et al., “Buggen: A self-correcting multi-agent llm pipeline for realistic rtl bug synthesis,” in2025 ACM/IEEE
7th Symposium on Machine Learning for CAD (MLCAD), 2025, pp. 1–9.DOI: 10.1109/MLCAD65511.2025.11189127
[32] N. Pinckney et al., “Comprehensive verilog design problems: A next-generation benchmark dataset for evaluating large
language models and agents on rtl design and verification,”arXiv preprint arXiv:2506.14074, 2025, Accessed: Nov. 3,
2025. [Online]. Available: https://arxiv.org/abs/2506.14074
[33] Tsarnadelis,Hw2project, https://github.com/tsarnadelis/HW2Project, [Online; accessed 3-November-2025], 2023.
[34] OpenAI,Gpt-4.1 and gpt-5, [Online; accessed 3-November-2025].
[35] M. AI,Llama 3.3 model card and prompt formats, https://www.llama.com/docs/model-cards-and-prompt-formats/
llama3 3/, 2024.