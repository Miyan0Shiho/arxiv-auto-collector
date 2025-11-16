# Vendor-Aware Industrial Agents: RAG-Enhanced LLMs for Secure On-Premise PLC Code Generation

**Authors**: Joschka Kersting, Michael Rummel, Gesa Benndorf

**Published**: 2025-11-12 08:56:11

**PDF URL**: [https://arxiv.org/pdf/2511.09122v1](https://arxiv.org/pdf/2511.09122v1)

## Abstract
Programmable Logic Controllers are operated by proprietary code dialects; this makes it challenging to train coding assistants. Current LLMs are trained on large code datasets and are capable of writing IEC 61131-3 compatible code out of the box, but they neither know specific function blocks, nor related project code. Moreover, companies like Mitsubishi Electric and their customers do not trust cloud providers. Hence, an own coding agent is the desired solution to cope with this. In this study, we present our work on a low-data domain coding assistant solution for industrial use. We show how we achieved high quality code generation without fine-tuning large models and by fine-tuning small local models for edge device usage. Our tool lets several AI models compete with each other, uses reasoning, corrects bugs automatically and checks code validity by compiling it directly in the chat interface. We support our approach with an extensive evaluation that comes with code compilation statistics and user ratings. We found that a Retrieval-Augmented Generation (RAG) supported coding assistant can work in low-data domains by using extensive prompt engineering and directed retrieval.

## Full Text


<!-- PDF content starts -->

Vendor-Aware Industrial Agents: RAG-Enhanced
LLMs for Secure On-Premise PLC Code Generation
Joschka Kersting
Centre for Machine Learning
Fraunhofer IOSB-INA
Lemgo, Germany
. r n e a . i r . s f d o k s r hk f a h c n i n e t i o u - g s e a @ o b jMichael Rummel
FA-EDC
Mitsubishi Electric Europe
Ratingen, Germany
e h e g a . c @ e u e e o l m r m m m c m l . mi .Gesa Benndorf
Centre for Machine Learning
Fraunhofer IOSB-INA
Lemgo, Germany
b s i - b e h na . . @ a d o a f e o u r n o s n n f . e i rf d e r g
Abstract—Programmable Logic Controllers are operated by
proprietary code dialects; this makes it challenging to train
coding assistants. Current LLMs are trained on large code
datasets and are capable of writing IEC 61131-3 compatible code
out of the box, but they neither know specific function blocks,
nor related project code. Moreover, companies like Mitsubishi
Electric and their customers do not trust cloud providers. Hence,
an own coding agent is the desired solution to cope with this. In
this study, we present our work on a low-data domain coding
assistant solution for industrial use. We show how we achieved
high quality code generation without fine-tuning large models
and by fine-tuning small local models for edge device usage.
Our tool lets several AI models compete with each other, uses
reasoning, corrects bugs automatically and checks code validity
by compiling it directly in the chat interface. We support our
approach with an extensive evaluation that comes with code
compilation statistics and user ratings. We found that a Retrieval-
Augmented Generation (RAG) supported coding assistant can
work in low-data domains by using extensive prompt engineering
and directed retrieval.
Index Terms—LLM, NLP, PLC, Automation Engineering,
Control Logic Generation, Structured Text (ST), IEC 61131-3
I. INTRODUCTION
Every day, millions of Programmable Logic Controllers
(PLCs) orchestrate production lines, manage power grids, and
safeguard critical infrastructure—from automotive assembly
plants to water treatment facilities [1]. Behind each auto-
mated process lies control logic programmed in specialized
languages, most prominently Structured Text (ST), a high-
level language standardized in IEC 61131-3 that combines
the expressiveness of modern programming with the real-time
determinism industrial systems demand [1].
Despite ST’s syntactic similarity to languages like Python
or Pascal, developing PLC control logic remains fundamen-
tally different. Where Python developers access thousands
of open-source libraries and Stack Overflow answers, ST
engineers face sparse code repositories and proprietary ven-
dor documentation. Moreover, each manufacturer – Siemens,
CODESYS, Mitsubishi Electric, Rockwell Automation – im-
plements distinct dialects with incompatible function blocks,
naming conventions, and compiler behaviors [2], [3]. A timer
implementation for Siemens TIA Portal will not compile on a
Mitsubishi Electric iQ-R controller; even seemingly portableconstructs like counter instructions exhibit vendor-specific
reset semantics and enable/disable conventions.
This fragmentation collides with industrial realities: manu-
facturing firms guard control logic as trade secrets, regulatory
frameworks restrict proprietary code sharing, and security-
critical infrastructure operators prohibit cloud-dependent AI
tools that could expose operational intelligence [4]. The re-
sult is a low-data domain where traditional large language
model training approaches—reliant on vast public code cor-
pora—struggle to generate vendor-compliant, compilable con-
trol logic [3]. Existing coding assistants trained on GitHub’s
predominantly web and enterprise software fail to recognize
thatRTRIG_Ptriggers on rising edges in Mitsubishi Electric
ST, that variable declarations must occur in external label
editors rather than inlineVARblocks, or that certain ladder
logic instructions are forbidden in ST contexts.
Nonetheless, recent research has demonstrated significant
advances in automated ST code generation using Large Lan-
guage Models (LLMs) [5]. Haag et al. [3] pioneered the
integration of compiler feedback with LLM training, achiev-
ing 70% compilation success rates through iterative Direct
Preference Optimization (DPO). Liu et al. [6] developed
Agents4PLC, achieving up to 68.8% formal verification suc-
cess across 23 programming tasks. Yang et al. [2] advanced
the field with AutoPLC, a framework for vendor-aware ST
code generation achieving 60% compilation success across
Siemens TIA Portal and CODESYS platforms. Meanwhile,
Ren et al. [7] introduced MetaIndux-PLC, employing control
logic-guided iterative fine-tuning.
However, despite these promising academic results, signifi-
cant gaps remain between research achievements and industrial
deployment requirements [4]. Brehme et al.’s [4] comprehen-
sive industry study revealed that most RAG applications in
industrial settings remain in prototype stages, with data pro-
tection, security, and quality identified as paramount concerns
(8.9, 8.5, and 8.7 of 10 points).
Existing research has primarily focused on either general
IEC 61131-3 compatibility or multiple vendor platforms si-
multaneously, but has not adequately addressed the challenge
of creating practical coding assistants for specific proprietary
implementations [4]. While Haag et al. [3] demonstrated com-
piler feedback integration, their approach requires extensivearXiv:2511.09122v1  [cs.SE]  12 Nov 2025

fine-tuning and relies on cloud-based services. Yang et al. [2]
addressed multi-vendor variations but depends on cloud-based
processing incompatible with industrial security requirements
[4]. Liu et al. [6] operates as a single-model solution lacking
comprehensive retrieval mechanisms for proprietary function
blocks. Ren et al. [7] requires substantial fine-tuning datasets
typically unavailable in proprietary environments.
Recent work has extended LLM-based control logic genera-
tion beyond textual notations. Koziolek et al. [8] demonstrated
Spec2Control, a workflow generating graphical DCS control
logic for ABB systems with a high connection accuracy
across 65 test cases, achieving 94-96% labor savings. While
Spec2Control targets multi-plant DCS graphical programming,
our work focuses on vendor-specific textual ST generation.
The specialized nature of industrial automation creates
a low-data domain scenario where comprehensive training
datasets for specific vendor implementations are scarce or
confidential [3], making traditional fine-tuning approaches
impractical [4]. Notably, no existing work has specifically
addressed Mitsubishi Electric’s ST implementation and MEL-
SOFT GX Works3, the engineering environment for Mitsubishi
Electric’s PLC series, which presents challenges due to its
proprietary function blocks, specific syntax variations, and
integration requirements. These are, however, compliant with
the IEC 61131-3 standard, since it leaves freedoms to vendors.
These freedoms and said function blocks, are not known to the
wider public and hence not known to LLMs.
This work addresses this critical gap through a novel RAG-
enhanced coding assistant specifically designed for Mitsubishi
Electric’s ST implementation with the development environ-
ment and compiler MELSOFT GX Works3. Unlike previous
approaches targeting multiple vendors [2] or require extensive
fine-tuning [3], [7], our solution focuses on one ecosystem.
The key contributions include: (1) the first RAG-based
coding assistant specifically designed for Mitsubishi Electric’s
ST implementation, addressing a significant gap in vendor-
specific industrial AI tools; (2) novel competitive multi-model
approach with MELSOFT GX Works3 integration enabling
real-time compilation feedback and error correction; (3) com-
prehensive evaluation using Mitsubishi Electric’s development
environment with both compilation statistics and expert assess-
ments from engineers familiar with the platform; and (4) prac-
tical demonstration that RAG-supported coding assistants can
achieve high-quality code generation for specific proprietary
platforms through targeted knowledge base construction and
vendor-specific prompt engineering. We also enhance earlier
work [9] that used translation transformers due to privacy
constraints and plan to further support local Small and Medium
Sized companies (SMEs) in the future with our results. Fig. 1
illustrates an ST program generated by our approach.
II. INDUSTRIALPROJECTBACKGROUND
ANDSYSTEMARCHITECTURE
GX Works3 emphasizes a label- and device-based model
in which variables (labels) are defined and managed in the
engineering tool and then referenced by ST programs; the
Fig. 1. Example of generated ST code using our GeneriST web app (built
with Streamlit [10]).

operating manual details dedicated editors and the standard
workflow for creating ST programs, running syntax checks,
and converting projects [11].
The ST programming guide presents a program creation
procedure that explicitly ties label definition, ST authoring,
and conversion (compilation) into an executable sequence
program, including correction steps when compile errors occur
[12]. Mitsubishi Electric instructions are exposed as functions
and function blocks callable from ST [13], [14].
Accordingly, the project scope is a specific copilot that
integrates with GX Works3 workflows: it uses directed re-
trieval over official documentation and internal code patterns to
emit code that compiles in GX Works3. Moreover, the copilot
validates outputs via the platform’s compilation and syntax-
check mechanisms [11], [12].
Finally, under low-data constraints, synthetic and customer-
derived task variants are iteratively refined using compiler
feedback, aligning with evidence that compilation-aware feed-
back loops yield measurable improvements in ST code gener-
ation quality [3].
The system architecture is determined by a RAG pipeline
that combines directed knowledge retrieval, competitive
multi-model orchestration, and compiler-in-the-loop validation
specifically for Mitsubishi Electric’s iQ-R ST dialect (Fig. 2).
The architecture prioritizes compilability, vendor-specific con-
straint enforcement, and practical deployment under industrial
security requirements.
The knowledge base consolidates three retrieval segments
indexed with metadata filtering: (1) function block definitions
for iQ-R CPUs, augmented with instruction suffix semantics
(_Pfor edge execution,_Ufor unsigned variants,EN/ENO
conventions); (2) specification excerpts from Mitsubishi Elec-
tric manuals encoding syntax rules, reserved words, and
Program Organization Unit (POU) structure constraints [11],
[13]; and (3) auxiliary context including user-contributed
examples and project-specific patterns. Documents are em-
bedded via semantic text representations using OpenAI’s
text-embedding-3-largemodel [15] and retrieved
through specialized query strategies tuned for definitions, rule
constraints, and usage examples respectively.
Prompt construction enforces vendor-specific hard con-
straints: reserved-word prohibition, declaration-before-use
mandates, restricted datatype vocabulary aligned with
GX Works3 semantics, and explicit POU structure templates.
Retrieved context from the knowledge base is concatenated
with a canonical example program encoding compliant timer
usage, state-machine logic, and edge-detection idioms.
The system supports three model configurations executed
concurrently on initial queries: (1)Azure RAG(GPT-4.1
with full retrieval), (2)Azure Standard(GPT-4.1, mini-
mal prompt, no retrieval), and (3)Local RAG(fine-tuned
quantized model with retrieval). This competitive architecture
enables systematic evaluation of retrieval impact and cloud
versus on-premises inference. Follow-up turns use a single
user-selected model to preserve conversational coherence. In
Sec. IV are further models tested for evaluation purposes.FailureKnowledge Base
Compiler-in-the-LoopFunction Blocks
(iQ-R CPU)Specifications
(MEC2023/24)Auxiliary Context
(Uploads, Chats)
Knowledge Retrieval
3 Specialized
Retrievers
(Libraries,
Specs, General)User Query
Optional Query
Expansion
(LLM Reasoning)
Prompt Construction
Hard Constraints +
Retrieved Context
+ POU Templates +
Canonical Example
Azure RAG
GPT-4.1
+ RetrievalAzure Standard
GPT-4.1
No RetrievalLocal RAG
Fine-tuned GGUF
+ Retrieval
Generated ST Code
(JSON or Markdown)
GX Works3
Compilation Server
(MCP)Diagnostic-Guided
Repair
(Max 3 Iterations)
Validated ST Code
+ Compilation ReportUser Interface
Model Selection,
Upload,
Chat PersistenceSuccessCompetitive Model Layer
Fig. 2. System architecture for Mitsubishi Electric ST coding assistant with
RAG, competitive multi-model orchestration, and compiler-driven iterative
repair.
Generated code is validated via a GX Works3 compila-
tion server that returns structured diagnostics. For Azure
RAG responses, the system performs bounded iterative re-
pair (maximum 3 compilation attempts). Failures trigger tar-
geted correction prompts addressing specific diagnostic cate-
gories—undeclared variables, reserved-word violations, type
mismatches, disallowed instructions—while preserving vali-
dated segments. The repair loop terminates on success, server
timeout, or budget exhaustion.
The interface supports model selection, optional query
expansion, draft mode, file uploads, and chat persistence.
Generated responses stream with compilation results displayed
in dedicated panels.
The system deploys via containerization with persistent
volumes for vector storage and model artifacts. User-uploaded
files undergo validation for binary content and encoding com-

pliance. All proprietary code remains local; external calls are
restricted to Azure OpenAI and the configured compilation
server. The fine-tuned local model enables fully offline oper-
ation for security-critical environments.
In contrast to Koziolek et al. [16], who demonstrated
retrieval-augmented generation for general IEC 61131-3 with
manual IDE import and OpenPLC validation, this architec-
ture addresses Mitsubishi Electric-specific constraints through
fine-grained retrieval segmentation (libraries/specs/general vs.
unified vector store), competitive multi-model orchestration
(three concurrent paths vs. single GPT-4), and automated
compiler-driven repair loops (bounded iteration vs. manual
import and spot testing). While Koziolek et al. focused on
proof-of-concept with OSCAT open-source libraries, this work
targets proprietary vendor ecosystems under low-data condi-
tions via strategic prompt engineering rather than exhaustive
document chunking.
Unlike approaches requiring exclusive cloud-dependent pro-
cessing [2], extensive fine-tuning and generating datasets [3],
[7], or single-model architectures [6], this system operates
effectively in low-data proprietary environments through di-
rected retrieval, strict vendor-specific guardrails, and optional
edge deployment.
III. IMPLEMENTATIONCHALLENGES ANDSOLUTIONS
The primary challenge stems from the scarcity of pro-
prietary training data for Mitsubishi Electric’s ST dialect.
While official documentation exists, it comprises thousands
of pages. To find corresponding descriptions, which may be
partly in images, after having found the correct function
block, is inefficient and error-prone. Each function block has
a brief, generic description text that does not sufficiently
distinguish instruction variants (e.g.,ZPUSHvs.ZPUSHP
for edge detection) for retrieval. We addressed this through
machine-augmented descriptions: a structured prompt encod-
ing Mitsubishi Electric-specific suffix semantics (ending in
Pfor rising-edge,_Ufor unsigned) was used to generate
enhanced function block descriptions that explicitly explain
variant behavior, enabling more precise retrieval.
Given industrial requirements for low latency, we opted for
a comprehensive single-prompt architecture rather than multi-
agent orchestration. The prompt embeds critical Mitsubishi
Electric rules: reserved word blacklists,VARblock semantics
(e.g.,VAR_INPUTavailable forFUNCTION), examples of
incorrect label usage, and mandates that declared function
blocks must be instantiated and invoked. This holistic approach
performs well in a single pass.
GX Works3’s label-based variable management diverges
standard code gerneration practices, requiring external label
registration rather than inlineVARdeclarations [11]. Generated
code must reference pre-defined device labels while avoiding
self-contained variable blocks that the GX Works3 converter
rejects. The prompt explicitly instructs models to assume
label pre-registration and provides templates demonstrating
compliant device referencing patterns. We set up an MCPserver that parsesVAR...END_VARlogic and imports it into
GX Works 3.
Absence of existing training corpora for the local fine-tuned
model necessitated synthetic data generation. We employed
the same comprehensive prompt used for inference to gen-
erate task-solution pairs, then validated each via GX Works3
compilation. A critical validation step enforces that generated
code actually invokes declared function blocks—static analysis
rejects samples where function blocks are defined but unused,
ensuring the compiler exercises full instruction coverage and
surfaces potential errors during training data curation.
Determining optimal model configurations (cloud vs. lo-
cal, RAG vs. baseline) required systematic comparison. The
architecture executes three concurrent model paths—Azure
RAG, Azure Standard (no retrieval), and Local RAG—on
identical queries, enabling direct evaluation of retrieval impact
and cloud-versus-edge trade-offs. This competitive approach
surfaces performance differences in compilation success, re-
sponse quality, and latency under real industrial workloads,
informing deployment decisions for privacy-sensitive versus
performance-critical environments.
Industrial environments often prohibit cloud dependen-
cies due to data protection regulations [4], necessitating on-
premises inference. The local model employs quantized GGUF
[17] format (deepseek-coder-v2-lite fine-tuned) [18] to enable
deployment on resource-constrained hardware including stan-
dard engineering laptops. Quantization reduces model size and
memory footprint while preserving sufficient code generation
capability for Mitsubishi Electric-specific tasks, balancing
privacy requirements against computational limitations (here
10GB VRAM are sufficient).
GX Works3 diagnostics employ vendor-specific formatting
that requires interpretation for automated repair. Rather than
implementing complex pattern-based diagnostic parsing, we
adopted a prompt-guided approach: compiler feedback is
extracted verbatim from the last compilation attempt and
embedded into a structured repair instruction that directs
the model to address common diagnostic categories (unde-
clared variables, reserved-word violations, type mismatches,
disallowed instructions) while preserving validated code seg-
ments. This approach delegates diagnostic interpretation to the
LLM’s reasoning capability, reducing maintenance overhead
for vendor-specific error schemas.
These challenges underscore the viability of prompt en-
gineering combined with strategic validation as an adapta-
tion pathway for low-data proprietary domains. By encoding
vendor-specific constraints directly into prompts, augmenting
documentation with machine-generated semantic annotations,
and validating synthetic data through actual compilation, the
system achieves practical code generation without large-scale
dataset collection or extensive model retraining [3], [7]. The
competitive evaluation framework further enables evidence-
based deployment decisions tailored to specific industrial
security and performance requirements.

IV. EVALUATION INMITSUBISHIELECTRIC
ENVIRONMENT
TABLE I
COMPILATIONSUCCESSRATESACROSSMODELCONFIGURATIONS
Model Configuration Compiled Repaired
Azure GPT-4.1 RAG73 %23 %
Azure GPT-4.1 Standard 38 % 5 %
Azure GPT-5 RAG87 %14 %
Azure GPT-5 Standard 56 % 25 %
Local Fine-tuned RAG86 %27 %
Local Fine-tuned Standard73 %22 %
Local RAG 25 % 12 %
Local Standard 23 % 17 %
With a focus on real-world applications, we incorporated
extensive human feedback by releasing an early demo app for
our industry partners. However, to provide a structured and
empirical evaluation, we generated 100 varied queries that
were used to generate 100 ST programs with each model
composition. We have evaluated the four earlier mentioned
models – Azure GPT-4.1 RAG — Standard (i.e., non-RAG);
Local Fine-tuned RAG — Standard (i.e., DeepSeek Coder v2
Lite Instruct) – and further non-fine-tuned models for a broader
view on our RAG approach. We got access to GPT-5 after
finishing the project, but we also included it in our evaluation
experiments (using reasoning set to low). All RAG configura-
tions employ OpenAI’stext-embedding-3-large[15]
for retrieval, ensuring consistent semantic search across model
variants. The results can be found in Tab. I.
We also generated queries to get training data: Our local
model was trained with generated ST programs that were
compilable by GX Works3. The generated code was produced
by the earlier described RAG system and its sophisticated
prompt. To achieve a high variability, we set up different
personas for the AI to match real-world query styles. Personas
were, e.g., ”casual developer”, ”electrician jargon”, ”control
engineer”, etc.. We added additional functional flags to make
sure we have a broad variety in program categories: ”timers”,
”communication”, ”array processing”, ”pid control”, etc. Since
the compiler got us a hard check, we aimed at a high number
of generated programs of which only the best (i.e., those that
passed) were saved for training.
Our quantitative evaluation results can be found in Tab. I. As
can be seen, the local fine-tuned RAG model performed best
(DeepSeek Coder v2 Lite Instruct). The second best is the fine-
tuned local model without RAG that achieved a 73 % compile
rate compared to 86% for the RAG. Interesting is the fast
inference time and high quality of the local standard model, i.e.
the one that did not use RAG. We also tested earlier versions
of the DeepSeek model and others like CodeLlama [19] and
Qwen-3 Coder [20], which performed less well. Our fine-tuned
DeepSeek model was able to write code in the Mitsubishi
Electric ST style and use the relevant function blocks due to
training. That is, it performed well with no overhead. However,
an overhead was necessary to generate curated training data.
The RAG version took much longer to generate code and wehad to shorten the system prompt to achieve reasonable answer
times (<1 min).
Interestingly, the large GPT-4.1 was outperformed quantita-
tively by the local model. The qualitative analysis showed that
users liked the larger GPT better. It delivered explanations for
each generated code and also answered in text-only manner, if
desired. These are things that the local model un-learned due to
fine-tuning. Moreover, users told us they also liked the coding
style of the larger model better: It generated longer and better
structured code with various variable blocks and functions.
Not all code that was compiled was reasonable, though. The
smaller model often did not produce coherent logic, generated
shorter programs and applied functions incorrectly. In the
generated code were, e.g., empty arrays at locations that, while
correct but useless, did not lead to compile errors.
However, we are comparing apples with oranges here:
Models like GPT-5 or even GPT-4.1, that actually followed
GPT-4.5 [21], have surely several hundred billion parameters
or more, the true number is not public, while DeepSeek Coder
v2 has 16 billion [22]. The latter model fits into an Apple
MacBooks memory, esp. when quantized, where it used about
10 GB. Hence, the users would opt for a hosted GPT-4.1/5
with RAG when data protection is not the primary concern.
But, as can be seen, a local option is also available, even for
highest security standards.
From Tab. I can be derived for sure that the fine-tuning and
RAG approaches perform well, since standard models do not
know Mitsubishi Electric ST style specifics. Moreover, the
Azure RAGs perform pretty well in the quantitative analysis
(73%, 87%), but convinces most in the qualitative analysis.
Those generations that set up a function block or function
but did not make use of it were regarded as failed attempt
(not compiled). The number of repaired attempts shows that,
e.g., of the 73 % compiled code attempts, 50% did not need a
repair, but 23% did need up to two repairs. RAG approaches
are promising; however, the latest GPT-5 performs even better
while requiring fewer repairs (87%; 14%).
V. DISCUSSION ANDINDUSTRIALIMPLICATIONS
This work demonstrates that vendor-specific coding as-
sistants can achieve practical utility in low-data industrial
domains through strategic prompt engineering and directed re-
trieval. Retrieval-augmented approaches substantially outper-
form non-RAG baselines. The 73% success rate via fine-tuning
alone indicates targeted synthetic data enables reasonable
baseline performance, yet the 38% non-RAG rate—despite
GPT-4.1’s size underscores the challenge of generating vendor-
compliant code without domain-specific context.
Industrial practitioners prioritize data protection, security,
and quality over raw performance [4]. This suggests modest
performance trade-offs between cloud and local models are
acceptable for eliminating external dependencies in security-
critical environments. Preliminary observations show local
models avoid API overhead, enabling faster iteration cycles
critical for interactive development.

While Mitsubishi Electric-focused, the architecture gener-
alizes to other ST dialects via vendor-specific documentation,
prompt templates encoding platform constraints, and vendor
compiler integration. This means that the solution could be
scaled to more CPU series apart from the used iQR series.
Limitations include Mitsubishi Electric-only scope (though
architectural components transfer), task coverage and ”spe-
cialist” local model behavior with occasional misinterpreta-
tion. Human evaluation from a single organizational context
may limit generalizability; longitudinal deployment studies are
needed to assess adoption barriers and real-world productivity
impacts. The local model limits the available context and thus
shortens the system prompt and synthetic training data may
have downsides not yet visible.
Deployment-wise, containerized on-premises configurations
with local models enable SME adoption without cloud de-
pendencies. Integration into GX Works3 (analogous to other
vendors) would reduce context-switching overhead. Synthetic
training data—validated via compiler—enables continuous im-
provement, while customer-specific uploads tailor the system
to organizational conventions without external data exposure,
addressing industrial security requirements.
Prompt engineering is cost-effective for low-data domains,
avoiding multi-month fine-tuning campaigns. Future work
includes static pre-compilation validation, coding environment
integration, but mostly a multi-agent approach for increased
performance in terms of quality and velocity. We also plan to
employ our findings to graphical IEC 61131-3 languages. Our
aim is to support the whole development cycle.
REFERENCES
[1] K. H. John and M. Tiegelkamp,IEC 61131-3: Programming Industrial
Automation Systems: Concepts and Programming Languages, Require-
ments for Programming Systems, Decision-Making Aids. Springer
Berlin Heidelberg, 2010.
[2] D. Yang, A. Wu, T. Zhang, L. Zhang, F. Liu, X. Lian, Y . Ren, J. Tian,
and X. Che, “Autoplc: Generating vendor-aware structured text for
programmable logic controllers,”arXiv preprint, Dec. 2024.
[3] A. Haag, B. Fuchs, A. Kacan, and O. Lohse, “Training llms for
generating iec 61131-3 structured text with online feedback,”arXiv
preprint, Oct. 2024.
[4] L. Brehme, B. Dornauer, T. Str ¨ohle, M. Ehrhart, and R. Breu, “Retrieval-
augmented generation in industry: An interview study on use cases,
requirements, challenges, and evaluation,”arXiv preprint, Aug. 2025.
[5] W. Xia, Y . Zhang, B. Zhao, W. Liu, L. Han, and Q. Ye, “Intelligent
plc code generation in hcps 2.0: A multi-dimensional taxonomy and
evolutionary framework,” inProceedings of the 2025 2nd International
Conference on Generative Artificial Intelligence and Information Secu-
rity, ser. GAIIS 2025. ACM, Feb. 2025, pp. 202–212.
[6] Z. Liu, R. Zeng, D. Wang, G. Peng, J. Wang, Q. Liu, P. Liu, and
W. Wang, “Agents4plc: Automating closed-loop plc code generation and
verification in industrial control systems using llm-based agents,”arXiv
preprint, Oct. 2024.
[7] L. Ren, H. Wang, J. Dong, H. Wang, S. Liu, Y . Laili, and L. Zhang,
“Metaindux-plc: A control logic-guided llm for plc code generation in
industrial control systems,”Applied Soft Computing, vol. 184, p. 113673,
Dec. 2025.
[8] H. Koziolek, T. Braun, V . Ashiwal, S. Linsbauer, M. A. Hansen, and
K. Grotterud, “Spec2control: Automating plc/dcs control-logic engi-
neering from natural language requirements with llms - a multi-plant
evaluation,”arXiv preprint, Oct. 2025.[9] D. Reinhardt, J. Jeschin, J. Jasperneite, and G. Benndorf, “Chatplc
– the potential of generative ai for control development [chatplc –
potenziale der generativen ki f ¨ur die steuerungsentwicklung],”Zeitschrift
f¨ur wirtschaftlichen Fabrikbetrieb, vol. 120, no. s1, pp. 196–201, Mar.
2025.
[10] Snowflake Inc., “streamlit/streamlit: Streamlit – a faster way to build
and share data apps.” https://github.com/streamlit/streamlit, 2019, last
accessed: 2025-10-16. Version 1.50.0.
[11] Mitsubishi Electric Corporation,GX Works3 Operating
Manual, 2024, manual No. SH-081215ENG. [Online].
Available: https://dl.mitsubishielectric.com/dl/fa/document/manual/plc/
sh081215eng/sh081215engan.pdf
[12] ——,Structured Text (ST) Programming Guide Book (Q/L Series), 2022,
manual No. SH(NA)-080368E-I. [Online]. Available: https://suport.
siriustrading.ro/01.DocAct/1.%20Automate%20programabile%20PLC/
1.6.%20Manuale%20comune/1.6.1.%20Manuale%20programare/Q,L%
20-%20Programming%20Guide%20Book%20(Structured%20Text%
20ST)%20SH(NA)-080368-I%20(11.22).pdf
[13] ——,MELSEC iQ-R/MELSEC iQ-F Structured Text (ST) Programming
Guide Book, 2023, manual No. SH-081483ENG. [Online].
Available: https://www.mitsubishielectric.com/dl/fa/document/manual/
plc/sh081483eng/sh081483engf.pdf
[14] ——,MELSEC iQ-R Programming Manual (CPU Module: Instructions
and Standard Functions/Function Blocks), 2024, manual No. SH-
081266ENG. [Online]. Available: https://www.mitsubishielectric.com/
dl/fa/document/manual/plc/sh081266eng/sh081266engac.pdf
[15] OpenAI, “New Embedding Models and API Updates — OpenAI,” https:
//openai.com/index/new-embedding-models-and-api-updates/, 2024, ac-
cessed: 2025-10-28.
[16] H. Koziolek, S. Gr ¨uner, R. Hark, V . Ashiwal, S. Linsbauer, and N. Es-
kandani, “Llm-based and retrieval-augmented control code generation,”
inProceedings of the 1st International Workshop on Large Language
Models for Code, ser. LLM4Code ’24. ACM, Apr. 2024, pp. 22–29.
[17] ggml, “ggml-org/llama.cpp: LLM inference in C/C++,” https://github.
com/ggml-org/llama.cpp, 2025, last accessed 2025-10-13.
[18] DeepSeek-AI, Q. Zhu, D. Guo, Z. Shao, D. Yang, P. Wang,
et al., “Deepseek-coder-v2: Breaking the barrier of closed-source
models in code intelligence,” 2024. [Online]. Available: https:
//arxiv.org/abs/2406.11931
[19] Baptiste Rozi `ere and Jonas Gehring and Fabian Gloeckle and Sten
Sootla and Itai Gat and Xiaoqing Ellen Tan et al., “Code llama: Open
foundation models for code,”arXiv preprint, Aug. 2023.
[20] An Yang and Anfeng Li and Baosong Yang and Beichen Zhang and
Binyuan Hui and Bo Zheng et al., “Qwen3 technical report,”arXiv
preprint, May 2025.
[21] OpenAI, “Introducing GPT-4.1 in the API — OpenAI,” https://openai.
com/index/gpt-4-1/, 2025, last accessed 2025-10-24.
[22] GitHub.com, “deepseek-ai/DeepSeek-Coder-V2: DeepSeek-Coder-V2:
Breaking the Barrier of Closed-Source Models in Code Intelligence,”
https://github.com/deepseek-ai/DeepSeek-Coder-V2, 2025, last accessed
2025-10-24.