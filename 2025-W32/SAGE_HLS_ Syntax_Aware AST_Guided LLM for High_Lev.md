# SAGE-HLS: Syntax-Aware AST-Guided LLM for High-Level Synthesis Code Generation

**Authors**: M Zafir Sadik Khan, Nowfel Mashnoor, Mohammad Akyash, Kimia Azar, Hadi Kamali

**Published**: 2025-08-05 15:28:13

**PDF URL**: [http://arxiv.org/pdf/2508.03558v1](http://arxiv.org/pdf/2508.03558v1)

## Abstract
In today's rapidly evolving field of electronic design automation (EDA), the
complexity of hardware designs is increasing, necessitating more sophisticated
automation solutions. High-level synthesis (HLS), as a pivotal solution,
automates hardware designs from high-level abstractions (e.g., C/C++). However,
it faces significant challenges, particularly in design space exploration and
optimization. While large language models (LLMs) have shown notable
capabilities in code generation, their application to HLS has been limited due
to the scarcity of (publicly) available HLS code datasets. Hence, research in
this domain has primarily focused on techniques such as prompt engineering and
retrieval-augmented generation (RAG). To overcome this limitation, this paper
introduces SAGE-HLS, the first-of-its-kind fine-tuned LLM specifically for HLS
code generation. Our method includes three key advancements: (i) We implement
Verilog-to-C/C++ porting, converting verified and synthesizable Verilog codes
into corresponding C, creating a dataset of 16.7K HLS codes; (ii) We implement
a fine-tuning strategy, which is based on instruction prompting to code
generation guided by abstract syntax tree (AST); (iii) We develop a
semi-automated evaluation framework using VerilogEval to assess the
functionality of the generated HLS code. Our experiments show that SAGE-HLS,
fined-tuned on the QwenCoder (2.5) 7B model, achieves a near 100% success rate
in code synthesizability and a 75% success rate in functional correctness.

## Full Text


<!-- PDF content starts -->

SAGE-HLS: Syntax-Aware AST-Guided LLM for
High-Level Synthesis Code Generation
M Zafir Sadik Khan, Nowfel Mashnoor, Mohammad Akyash, Kimia Azar, Hadi Kamali
Department of Electrical and Computer Engineering (ECE), University of Central Florida, Orlando, FL 32816, USA
{mzafirsadik.khan, nowfel.mashnoor, mohammad.akyash, azar, kamali }@ucf.edu
Abstract —In today’s rapidly evolving field of electronic design
automation (EDA), the complexity of hardware designs is in-
creasing, necessitating more sophisticated automation solutions.
High-level synthesis (HLS), as a pivotal solution, automates
hardware designs from high-level abstractions (e.g., C/C++).
However, it faces significant challenges, particularly in design
space exploration and optimization. While large language models
(LLMs) have shown notable capabilities in code generation, their
application to HLS has been limited due to the scarcity of
(publicly) available HLS code datasets. Hence, research in this
domain has primarily focused on techniques such as prompt
engineering and retrieval-augmented generation (RAG). To over-
come this limitation, this paper introduces SAGE-HLS, the first-
of-its-kind fine-tuned LLM specifically for HLS code generation.
Our method includes three key advancements: (i) We implement
Verilog-to-C/C++ porting, converting verified and synthesizable
Verilog codes into corresponding C, creating a dataset of 16.7K
HLS codes; (ii) We implement a fine-tuning strategy, which is
based on instruction prompting to code generation guided by
abstract syntax tree (AST); (iii) We develop a semi-automated
evaluation framework using VerilogEval to assess the functional-
ity of the generated HLS code. Our experiments show that SAGE-
HLS, fined-tuned on the QwenCoder (2.5) 7B model, achieves
a near 100% success rate in code synthesizability and a 75%
success rate in functional correctness1.
Index Terms —Large Language Model (LLM), Abstract Syntax
Tree (AST), High-level Synthesis (HLS), Synthesis.
I. I NTRODUCTION
High-Level Synthesis (HLS) was introduced to mitigate
the inefficiencies of traditional register-transfer level (RTL)
design and automate hardware generation from high-level pro-
gramming languages [1], [2], aiming to enhance productivity
and ease application-specific hardware design space explo-
ration (DSE) [3] without hand-coding in hardware description
languages (HDL). Over the past decade, HLS has matured,
enabling faster iterations and optimizations in hardware design
[4]–[7]. However, challenges persist in optimizing power,
performance, and area (PPA), often requiring extensive manual
intervention and expert knowledge [6], [8].
Recent advancements in large language models (LLMs)
have shown remarkable capabilities in natural language un-
derstanding [9], reasoning [10], and particularly code genera-
tion (as evidenced by models like GitHub Copilot [11] and
CodeGen [12]). In the domain of hardware designs, while
LLMs have been applied to tasks such as RTL code generation
[13]–[18], debugging and verification [19]–[21], security [22],
1The code and resources related to this work are publicly available at:
https://github.com/zfsadik/SAGEHLS[23], etc., their application in HLS has been comparatively
limited [24]. A few recent research endeavors have begun
to explore the integration of LLMs in HLS [25]–[33], yet
these efforts remain narrowly focused on specific aspects of
the design process. Some studies have investigated the use of
LLMs for optimizing existing HLS code by refining pragma
set, or coding styles to enhance synthesizability [26], [30].
Other studies have focused on converting software C/C++
codes into HLS-compatible C/C++ through refactoring or code
repair, aiming for code generation [28], [31]. Some studies
also investigated the use of LLM as an alternative for HLS to
perform C-to-HDL conversion through direct reasoning [27].
While these approaches have demonstrated promising re-
sults in their scope, they primarily rely on prompt engineer-
ing, whether using simple instructions or structured chain-
of-thought prompting, all on top of commercially available
models, e.g., GPT, Gemini, and Claude [30], [31]. Notably,
none of these studies have pursued the development of a
domain-specific model by fine-tuning a base LLM for HLS.
Additionally, existing LLM-driven HLS approaches face two
other key limitations: (i) reliance on raw text-based learning
without structural representations, leading to difficulties in
handling hierarchy, memory optimizations, and pipelining; and
(ii) the lack of a standardized evaluation framework, making
results difficult to compare due to variations in benchmarks,
synthesis tools, and optimization objectives.
To address these limitations, in this paper, we introduce
SAGE-HLS, a novel approach that moves beyond prompt
engineering by fine-tuning a base model for HLS code gen-
eration to improve the quality (functional correctness) and
Open -source RTL Designs
Prompting for SpecificationDataset of 
RTL Designs
Corresponded 
(generated) 
Instructions:
Specification, 
Func . Desc.,
Arch. Notes, 
etc.Open -source 
pre -trained LLM 
Instruction -tuning
(fine -tuning on pairs of 
codes/instructions) 
LLM
(on Commercial)
(on Open -Source)tuned
Open -source 
fine -tuned LLM 
RTL Code generation 
(using tuned model)
Testing 
(evaluation)
Refinement
Open -source  RTL  Design s
OpenAI
GPT -4o
RTL -to-C Porting
Verifying 
Synthesizability
using AMD Vitis 
HLS
If passedTree -sitter
OpenAI GPT -4o
Instruction for 
HLS C generation
Abs. syntax tree 
(AST) of HLS C
pre-trained LLM 
 fine -tuned LLM Open -source LLM fine -tuning  
(QwenCoder  2.5 - 7B)•Functional verification
•Synthesizability chec k
Fig. 1: Overview of SAGE-HLS: From Code Dateset Preparation using
Verilog-to-C Porting, to HLS Code Generation using Fine-Tuned LLM.arXiv:2508.03558v1  [cs.PL]  5 Aug 2025

TABLE I: Overview of Existing LLM-assisted Studies for HLS Code Generation.
Study LLM Method Key Novelty Evaluation Metrics HLS Tool Test/Synthesizability
Swaroopa et al. [25]Prompt engineering for HLS-C
(direct instruction-to-code)Introducing HLSEval
(functional correctness check)pass@1 functional correctness Vitis HLS
HLSPilot [26] Prompt Engineering for design space exploration Integrating profiling and task pipelining Area/Latency evaluation Vitis HLS
Liao et al. [27]Prompt engineering for HLS-C,
Prompt engineering for C-to-Verilog SynthesisLLM as the C-to-Verilog synthesis tool Area/Latency evaluationVitis HLS,
LLM as HLS
HLS Repair [28] Prompt Engineering for HLS-C RepairLLM-based Step-by-step Repair
(using RAG and bit-width optimization)Functional pass rate None
SynthAI [29] Recursive Prompting for HLS-CChain of Thoughts (CoT) Prompting
using multi-agent LLMGeneric pass/fail per prompt None
RALAD [30] Prompt Engineering forpargma managementDocument Splitting and Retrieval
forpargma -based OptimizationArea/Latency evaluation Vitis HLS
C2HLSC [31] Prompt Engineering for HLS CAutomated software C to HLS-C
(using iterative LLM-based refactoring)Functional correctness,
Area/Latency evaluationCatapult HLS
Agentic-HLS [32] Prompt Engineering for HLS Code EvaluationAgentic reasoning with
hierarchical graph embeddings (GNN)Area/Latency Evaluation None
SAGE-HLS
(Proposed)Fine-tuning code-based LLM for HLS-CVerilog-to-C for DB generation
AST-based context-aware fine-tuningFunctional correctness
Pass@1/5/10Vitis HLS
synthesizability of LLM-generated HLS code (see Fig. 1). By
integrating instruction-based learning and abstract syntax tree
(AST) representations, SAGE-HLS is built on the basis of
structured learning, leveraging both semantic and syntactical
perspectives of coding to enhance automation, robustness, and
correctness. Our main contributions are as follows:
1) We construct a large-scale dataset of 20,000 HLS-C codes
by reversely converting verifed and synthesizable Verilog
to C/C++. This Verilog-to-C porting, conducted by GPT-4,
creates a semi-synthetic data useful for fine-tuning.
2) We fine-tune a pre-trained model (QwenCoder 7B model
[34]) using instruction-based learning, initially mapping
instructions to code, then subsequently enhancing it with
AST representations of HLS codes, to capture functional
dependencies and improve synthesis outcomes.
3) We engage a semi-automated evaluation framework (by
using AMD Vitis HLS and VerilogEval [35]) to measure
both synthesizability and functional correctness, reflecting
the efficiency of the SAGE-HLS model.
II. B ACKGROUND AND RELATED WORK
A. LLMs for RTL Code Generation
Recent advancements in transformer-based hardware design
automation have demonstrated the potential of LLMs in vari-
ous EDA tasks, such as scripting [15], error diagnosis [36], and
AI-driven design assistants [37]. Among them, numerous stud-
ies have focused on fine-tuning and pre-training LLMs for RTL
code generation. Early efforts, such as VeriGen [13], com-
piled datasets from GitHub and textbooks but suffered from
inconsistencies due to insufficient pre-processing, resulting in
frequent syntax errors in generated Verilog code. To improve
data quality, RTLCoder [38] introduced RTL-specific keyword
extraction to synthesize code-instruction pairs, yet its reliance
on GPT-3.5 limited diversity in generated code. Addressing
this, OriGen [39] employed code-to-code augmentation and
self-reflection mechanisms, enabling dataset expansion withsyntactically varied but semantically equivalent Verilog while
refining code through compiler feedback loops.
Building upon these advancements, BetterV [40] refined
Verilog generation by aligning Verilog semantics with C-like
program structures to enhance LLM comprehension, introduc-
ing discriminative generation techniques for PPA optimiza-
tion. AutoVCoder [41] tackled domain-specific accuracy and
diversity using a two-stage fine-tuning process and retrieval-
augmented generation (RAG) for improved correctness.
Moving beyond generation, CodeV [42] shifted focus to
Verilog summarization, curating a dataset of 165K modules
to generate high-quality code-description pairs for training.
CraftRTL [17] integrated state-transition diagrams, Karnaugh
maps, and waveforms to enhance structured reasoning in
LLMs, while MAGE [43] applied multi-agent reinforcement
learning for RTL optimization.
B. LLMs for HLS (C/C++) Code Generation
Unlike RTL designs, which explicitly define cycle-accurate
concurrent behaviors, HLS-C introduces hierarchical (sequen-
tial) transformations, compiler-driven optimizations, and syn-
thesis constraints that are not explicitly encoded in the text rep-
resentation of the HLS code. Liao et al. [27] provide a broad
evaluation of LLM-based approaches for HLS, benchmarking
their effectiveness against standard HLS tools like Vitis HLS.
Their study highlights that while LLMs can translate C code
into hardware descriptions, their performance in terms of
power, area, and timing efficiency remains suboptimal due to
their lack of explicit structural guidance.
Similarly, HLSPilot [26] proposes an LLM-driven method-
ology that integrates profiling, kernel extraction, and DSE
to optimize C-to-HLS conversion, demonstrating that LLMs
can generate competitive HLS designs when guided with
appropriate synthesis constraints.
A key challenge in HLS-based LLM generation is automat-
ing the transformation of generic C/C++ programs into syn-
thesizable HLS-C, as explored by C2HLSC [31], which inves-

RTL Code Extraction
Small to large  size V codes
(resembling FPGA -style)
•DSP
•Systolic  arrays
•FIFO memories
•Peripherals (JTAG, USB)
•Crypto  cores
•Soft processor units
to GPT -4o
Verilog -to-C PortingRTL Code to C/C++ 
Conversion by GPT4 
(Equivalent C/C++)
 (w/ self refinement)
to Vitis HLSSynthesizability verification 
✓ 
✗Added to HLS C/C++ DB
Dropped from HLS C/C++ DB
HLC C/C++ 
code dataset
to GPT -4o
Code -to-Instruction Prompting
(+ AST Generation from HLS C/C++)
(1) Dataset GenerationInstruction 
Generation
(from  given  
HLS  C/C+ +AST Generation from 
HLS C/C++ codes
•Tree -sitter  AST
(parsing C to AST)
•Textualized  AST
(Structured text of AST)
(2) Fine Tuning
Instruction -tuning
on QWENCoder  7B
•Input:
Instruction + AST
•Output:
HLS C/C++ Code HLC C/C++ datasetInstruction generated  
per each  HLC C/C++AST generated  
per each  HLC C/C++(3) Evaluation16.7K HLS Codes
Trained QWENCoder  7B
VerilogEval  datasetInstruction
preparation
Instructions for HLS C 
Generation
Generated HLS C
to Vitis HLS
to GPT -4o
Generated 
Verilog
Testbench Generation(Functional )
pass @1
pass@5
pass@10Generated 
Testcases(constrained)
Constrained 
simulation(Synthesis ) checkFig. 2: Overview of SAGE-HLS: Fine-tuning an LLM for HLS-C generation leveraging AST.
tigates whether LLMs can refactor software-like C programs
into hardware-compatible representations. Their approach sug-
gests that while LLMs can perform basic transformations,
they struggle with hierarchical designs, memory optimizations,
and function refactoring, requiring additional preprocessing
steps. Xu et al. [28] expand on this challenge by introducing
an automated program repair framework that uses retrieval-
augmented generation (RAG) to correct C/C++ programs for
HLS synthesis, addressing common pitfalls such as dynamic
memory allocation, recursion, and improper data types.
Beyond syntax-level generation, recent works have explored
multi-agent structured reasoning for HLS. SynthAI employs
a structured decision graph with ReAct agents and Chain-
of-Thought (CoT) prompting to decompose complex HLS
design tasks into manageable subproblems, improving design
modularity and synthesis efficiency [29]. Agentic-HLS further
extends this concept by integrating agentic reasoning into
the HLS flow, using graph-based representations to optimize
performance predictions and pragma selection [32].
Despite advancements, prior studies rely on raw text-based
learning, limiting their ability to capture structural dependen-
cies in HLS designs. Our work bridges this gap by integrating
AST representations, enabling LLMs to better understand
function dependencies, loops, and memory access patterns for
more synthesis-friendly HLS code generation.
C. LLMs for Structural Data Analysis
With the rise of LLMs, researchers have explored ways to
incorporate graph-structured data into LLM inputs, either as
embeddings for in-context learning or as structured prompts to
improve reasoning over complex relationships [44]. Given that
hardware design inherently involves structural dependencies,
such as control flow graph (CFG), data flow graph (DFG),
and hierarchical relationships, effective integration of these
structural elements into LLM reasoning could be crucial for
improving synthesis and optimization tasks.
Several studies have investigated encoding graph informa-
tion into LLMs. Fatemi et al. [45] examined the impact of
different textual encoding schemes for graph representations,
showing that LLM performance in graph reasoning tasks ishighly sensitive to encoding strategies, task complexity, and
graph structure. Perozzi et al. [46] introduced GraphToken,
a parameter-efficient embedding method that enhances LLM
reasoning by learning structured graph representations instead
of relying on textual encoding. Their work suggests that aug-
menting prompts with explicit graph embeddings significantly
improves task performance. Alternatively, GraphLLM [47]
integrates graph learning models directly with LLMs, em-
ploying a graph transformer to process graph structures more
efficiently, thereby improving both accuracy and scalability.
III. P ROPOSED MODEL : SAGE-HLS
In SAGE-HLS, we aim to address the challenges in LLM-
based HLS code generation through three key stages:
1)Synthesis-friendly Dataset Creation : With the scarcity of
reliable (likely synthesizable) HLS code across varying
sizes and complexities, from small building blocks (e.g.,
filters, memory management units (MMUs), crypto cores,
etc.) to large-scale designs (e.g., AI accelerators), we have
created a dataset by posting verified and synthesizable
Verilog code to C/C++, ensuring a high-quality, synthesis-
friendly collection of HLS code sutiable for fine-tuning.
2)AST Generation from HLS Codes : We extracted structural
information from the HLS-C code to construct ASTs that
encode hierarchical and functional dependencies.
3)Model Fine-Tuning : Using both textual and AST-based
representations, we will fine-tune an LLM to enhance their
capability in generating synthesis-aware HLS-C code. This
dual-representation strategy enables the model to grasp
both the semantic and structural properties of the code.
These stages, explained in this section, collectively enable
the model to capture both semantic and structural properties
for more robust and efficient HLS-C code generation.
A. Verilog-to-C Porting: Creating and Filtering HLS-C
To construct a diverse and high-quality dataset for HLS code
generation, we first gathered 19K Verilog designs from open-
source repositories, e.g., GitHub and Hugging Face, covering
a wide range of circuit architectures, including arithmetics

and cryptographic cores2. Although there is an abundance of
C/C++ codes on open repositories, e.g., GitHub, most of it is
software-oriented and thus not directly suitable for HLS. Only
a small portion of general-purpose C/C++ codes is written in
a synthesizable, hardware-oriented manner style compatible
with HLS tools. Using the above RTL implementations as
a reference, we employ GPT-4o to generate corresponding
HLS-C code and natural language instructions (see Fig. 2 and
Listing 1), ensuring that the generated HLS-C adheres to high-
level programming paradigms while maintaining functional
equivalence with the original Verilog designs.
1SYSTEM_MSG = ’’’
2You are an expert in hardware design, Verilog, and High-
Level Synthesis (HLS). The task is to assist users in
converting Verilog into functionally equivalent HLS-C code
that can be synthesized using tools like Vivado HLS or
Catapult HLS.
3
4When given a Verilog code, you must always generate two
outputs:
5
6(1) Equivalent HLS Code: Convert the Verilog into HLS-C
while maintaining equivalent functionality. Ensure that the
generated HLS-C defines the top-level module as top_module
, serving as the entry function. Apply necessary #pragma
HLS directives for optimizations, e.g., loop unrolling,
pipelining, and memory interfaces. Maintain proper data
types and bit-widths to preserve accuracy.
7
8(2) Corresponding Prompt: Generate a generic and simple
prompt that describes the hardware functionality concisely.
The prompt must be structured in a way that any LLM,
including smaller models, can generate the correct HLS code
without requiring the original Verilog code. Ensure that
the generated HLS code always includes a top_module
function as the entry point. You must strictly adhere to
this format, ensuring clarity and correctness in both
outputs. Do not add unnecessary explanations - focus on
delivering precise and structured responses.
9’’’
Listing 1: C-to-Verilog Porting using GPT-4 Prompt Engineering.
Through this step, we ensure that the generated HLS code
are functionally equivalent with the original Verilog designs,
while it inherently preserves the synthesis-friendly character-
istics of the designs. Furthermore, upon manual inspection of
the HLS-C code generated by GPT-4o, we observe that pragma
directives are added which are valid and provide guidance
that is directionally consistent with typical HLS optimization
practices. Also, from Listing 1, it is evident that no platform or
board is specified during the dataset generation. This generates
a generic, algorithm-only implementation that can be compiled
and simulated in any HLS tool. So, the generated dataset is
not limited for a specific board or HLS tool, thus the resulting
HLS-C code is synthesizable. To enforce synthesis compatibil-
ity, we validated the synthesizability of the generated HLS-C
codes using Vitis HLS. This step filtered out non-synthesizable
or inefficient implementations, leaving 16.7K verified HLS-C
designs that met synthesis requirements.
B. AST Generation for AST-based Fine-tuning
To extract structural information from HLS-C code, we
utilize Tree-sitter [48], a widely used incremental parsing
2Given that HLS predominantly targets FPGA-based designs, our focus is
on array-style codes, e.g., systolic arrays, DSP engines, etc..Algorithm 1 AST Extraction and Control Flow Analysis
Require: HLS-C source file S
Ensure: Optimized AST T′
mainand Control Flow Graph CFG
1:function PARSE AST(S)
2: T←Tree-sitter.parse (S)
3: Nmain←FindNode (T,”main” )
4: return Subtree (T, N main )
5:function OPTIMIZE AST(Tmain )
6: foreach node N∈Tmain do
7: ifN∈RedundantNodes then
8: Tmain←RemoveNode (Tmain, N)
9: else if HasSingleChild (N)then
10: Tmain←CollapseNode (Tmain, N)
11: return Tmain
12:function ANALYZE CONTROL FLOW (T′
main)
13: CFG← ∅
14: foreach node N∈T′
maindo
15: handler ←Handlers (N)
16: CFG←CFG∪handler
17: return CFG
18:function HANDLERS (N)
19: return

{(N, T (N)),(N, E (N))}, N.type =if
{(N, L (N)),(L(N), N)}, N.type ∈ {for,while }
{(N, c i)|ci∈C(N)}, N.type =switch
{(N, F (N))}, N.type =function
{(N, R (N))}, N.type =return
{(N, Expr (N))}, N.type =expression
{(N, D (N))}, N.type =declaration
{(N, A (N))}, N.type =assignment
{(N, C (N))}, N.type =call
∅, otherwise
20:function MAIN(S)
21: Tmain←ParseAST (S)
22: T′
main←OptimizeAST (Tmain )
23: CFG←AnalyzeControlFlow (T′
main)
24: return (T′
main, CFG )
framework that efficiently generates ASTs. The AST repre-
sentation can provide a formal hierarchical decomposition of
the program, capturing essential structural properties such as
syntactic constructs, function call dependencies, control flow
structures, and memory access patterns3.
Unlike raw token-based representations, by using ASTs,
we leverage deeper structural insights when fine-tuning our
LLM. By wrapping our processing pipeline around Tree-
sitter, we systematically extract typed tree structures that
organize HLS-C code into a context-free grammar repre-
sentation, allowing for efficient traversal and transformation.
This approach preserves key hardware-relevant abstractions,
including loop unrolling, function inlining, pipeline direc-
tives, and memory partitioning strategies, which are crucial
for high-performance hardware synthesis. Additionally, ASTs
improves static analysis of code by explicitly capturing data
dependencies and control structures, enabling our model to
learn how different code components contribute to scheduling,
resource allocation, and computational parallelism in an HLS
design. Furthermore, AST-based representations provide a
layer of abstraction between software-like HLS-C code and it’s
underlying RTL implementation, helping the model bridge the
3Due to the inherently structured nature of HLS-C, the AST derived from
HLS-C provides a more precise and constrained representation of the code’s
structure compared to traditional software C [49].

gap between algorithmic specification and hardware synthesis
constraints whether those are pragmas, directives or external
tool constraints. Alg. 1 shows a high-level yet step-by-step
implementation of AST, followed by generic optimization,
leading to the extraction of the CFG per each HLS-C.
C. LLM Fine-tuning for HLS-C Generation
To enable LLM to generate synthesis-friendly HLS-C code,
we fine-tune QWENCoder (2.5) 7B [34], an advanced open-
source model designed for code generation. Our fine-tuning
strategy is designed to evaluate the impact of structural infor-
mation by training two separate variants: (i) a baseline model
fine-tuned only on raw HLS-C text (called QWEN-HLS ) and
(ii) an AST-enhanced model trained with both HLS-C text and
its corresponding AST representation (called SAGE-HLS) .
In the AST-augmented variant, (from Alg. 1) we prepend a
serialized AST representation to the input, allowing the model
to process hierarchical relationships, function dependencies,
and control flow structures that are otherwise lost in token-
based representations. This additional structured context pro-
vides the model with explicit syntax-awareness which helps it
make better synthesis decisions, such as loop transformations,
memory optimizations, and pipeline scheduling. Listing 2
illustrates the input format provided to the base model during
the fine-tuning process of the AST-augmented model.
1HLS_CODE = ’’’
2#include <ap_int.h>
3#include <hls_stream.h>
4// Top-level function for HLS synthesis
5void top_module(
6ap_uint<11> v_addr, ap_uint<8>& v_data,
7bool v_en, bool& v_rdy
8) {// ROM definition and initialization
9static const ap_uint<8> rom[2048] = {/ *Add ROM data here
*/};
10// Signal management
11#pragma HLS PIPELINE II=1
12v_rdy = v_en;
13if (v_en) {
14v_data = rom[v_addr];
15}}
16’’’
17
18HLS_INSTRUCTION = ’’’
19Create a C++ function named ‘top_module‘ that simulates a
ROM with the following behavior:
20- The top_module accepts 4 inputs: a 11-bit address (‘
v_addr‘), an 8-bit data output reference (‘v_data‘), a
boolean enable signal (‘v_en‘), and a boolean ready signal
output reference (‘v_rdy‘).
21- When ‘v_en‘ is high, the ROM outputs the data at the
location specified by ‘v_addr‘ and sets ‘v_rdy‘ to true.
22- The ROM should have a size of 2048 entries (addressable
using the 11-bit address) with 8-bit data in each entry.
23- Use a static array to represent the ROM content and
initialize it with some placeholder data.
24- Optimize the function by pipelining it with a single
initiation interval.
25’’’
26
27AST = ’’’
28FuncName: top_module, Params: ap_uint<11>, ap_uint<8>, bool
, bool
29VarTyp: ap_uint<8>
30Asgnmnt: v_rdy = v_en
31IfStmt: Contn: (v_en)
32Then:
33Asgnmnt: v_data = rom[v_addr]
34’’’
Listing 2: HLS Code, Instruction and AST for training SAGE-HLS.To ensure that the generated HLS-C code is not only struc-
turally correct but also synthesizable with efficient hardware
characteristics, we embed pragma annotations during both
dataset generation and model fine-tuning. These annotations
are selectively added based on AST-guided structural insights
such as loop depth, memory access patterns, and function
hierarchy. During fine-tuning, the input sequence is a concate-
nation of (i) instruction prompt, (ii) serialized AST, allowing
the model to learn contextual correlations between high-
level specifications, structural code properties, and hardware
optimization directives. This combination enables the model
to better infer where and how to place pragmas. For instance,
loop nodes with independent iterations in the AST are often
associated with unrolling or pipelining pragmas, while top-
level I/O functions receive interface-related annotations. The
model learns these associations during fine-tuning through a
diverse set of examples where pragma usage varies depending
on code topology. This results in HLS-C code that is not only
functionally correct but also optimized for performance, as
evidenced in our evaluation results.
D. Evaluation on Modified VerilogEval
A key challenge in evaluating HLS-C code is the absence
of a standardized benchmark. To address this, we establish
a structured evaluation framework derived from VerilogEval
[35], a dataset (followed by simulation-based verification)
originally designed for LLM-based RTL verification. However,
assuming that HLS-C is synthesized to RTL (e.g., using
Vitis HLS), VerilogEval’s direct application to these RTLs
is limited by differences in HLS behavior, particularly the
timing variations (e.g., introduced by Vitis HLS when targeting
FPGA architectures). It prevents one-to-one validation of HLS-
generated RTL vs. reference designs. To bridge this gap, we in-
troduce a semi-automated verification methodology that adapts
VerilogEval for HLS-C through constrained simulation4.
Our evaluation begins by transforming VerilogEval’s in-
structions into an HLS-compatible format. These adapted
instructions prompt our fine-tuned LLM to generate HLS-C
code. Now via constrained simulation (see Fig. 2), testbenches
from VerilogEval are modified using GPT-4 to incorporate
specific constraints that validate functional correctness based
on reference code, existing testbench, and instructions. The
number of constraints introduced corresponds to the number of
test cases defined by VerilogEval for each circuit. The modified
testbenches (containing constraints) are then executed using
a RTL simulator. If any constraint fails, the system flags the
design as incorrect. This structured verification process ensures
that our evaluation rigorously assesses the correctness of HLS-
generated designs while accommodating the synthesis-driven
differences inherent in targeted implementations.
IV. E XPERIMENTAL SETUP
We evaluate our approach by comparing the baseline
QWENCoder, text-only fined-tuned model (QWEN-HLS), and
4It derives from constrained random verification (CRV) [50] but generates
deterministic stimulus via reference code, existing testbench, and instructions.

the AST-enhanced fine-tuned model (SAGE-HLS). The eval-
uation phase includes three main steps: (i) HLS-Cgeneration:
We use HLS-aligned VerilogEval instructions as the prompts
to our fine-tuned model, which generate corresponding HLS-
C code; (ii) Synthesizabilitycheck : We run AMD Vitis HLS
on generated HLS-C codes, where the output would be syn-
thesized RTL code generated by Vitis HLS; (iii) Func tionality
correctness: We run our semi-automated constraint-based sim-
ulation on the HLS-generated RTL, where constraints (pass or
fail) show the correctness of the code.
Our experiments are conducted using QwenCoder (2.5) 7B
[34], a well-optimized model for code generation. To enable
efficient fine-tuning, we apply low-rank adaptation (LoRA),
which significantly reduces memory overhead while main-
taining fine-tuning effectiveness [51]. Also, we utilize 4-bit
quantization to minimize the memory footprint and accelerate
inference without compromising model performance. The fine-
tuning process is configured with a:
•Per-device train batch size of 2;
•Gradient accumulation steps of 4;
•One training epoch with a linear learning rate scheduler;
•The learning rate of 2e-4;
•AdamW 8-bit optimization;
•Weight decay of 0.01 (for better generalization);
•5-step warmup, with a seed of 3407 (for reproducibility).
To ensure a fair evaluation on the VerilogEval benchmark,
we conducted an experiment to verify that our dataset’s
instructions are not similar to those in VerilogEval. We cal-
culated the ROUGE-L [52] similarity scores between our
instructions and those from VerilogEval, and found that all
scores were below 0.4. This low similarity confirms that
our model learns to generalize rather than memorize specific
instruction patterns. For initial dataset collection and creation,
including Verilog-to-C porting, instruction generation, and
testbench augmentation for constrained simulation, we utilized
the OpenAI API5. For the synthesis (using AMD Vitis HLS
2024), we used AMD’s Zynq XCZU3EG-SBV A484 MPSoC
with standard speed grade and extended temperature range.
V. E XPERIMENTAL RESULTS
Table II shows a comparative analysis of QWENCoder 7B
(base, QWEN-HLS, and SAGE-HLS) in terms of synthesiz-
ability and functional correctness across multiple 1, 5, and
10 runs (to observing the likelihood of producing a correct
result after multiple attempts). First, we check synthesizability,
ensuring that the generated code can successfully compile into
5The total cost for processing the entire dataset amounted to around $480.RTL using Vitis HLS. As shown in Table II, while the pre-
trained (base) QWENCoder 7B model achieves low synthe-
sizability scores, both QWEN-HLS and SAGE-HLS demon-
strate a significant improvement, achieving nearly perfect
synthesizability. Additionally, SAGE-HLS performs similarly
to QWEN-HLS in terms of synthesizability6, suggesting that
AST-based structural learning does not significantly improve
syntactical formation of the HLS-C for synthesizability.
In terms of functional correctness, the pre-trained QWEN
model has the lowest correctness ratios, showing that most of
its generated code does not pass functional validation. While
the QWEN-HLS model significantly improves correctness,
the AST-Guided SAGE-HLS model consistently outperforms
QWEN-HLS in functional correctness, thanks to AST integra-
tion, which enables the model to learn structural dependency
for more robust (and consistent) code generation.
To further evaluate HLS-SAGE, we categorize benchmarks
into three difficulty tiers based on the number of characters
in the reference Verilog code from VerilogEval: (T1) easy
(shorter RTL codes); (T2) intermediate (moderate length of
RTL codes); and (T3) hard/complex (large codes, involving
loops, deep pipeline, huge state machines, etc.). Tables III
and IV assess how models handle varying complexity levels
in terms of synthesizability and functional correctness. As
shown in Table III, while QWEN base model struggles more
with synthesizability (in T2 and T3), even at @5 and @10,
synthesizability is nearly independent of difficulty tiers in
trained models7. As shown in Table IV, all circuits are cases
that failed in pass@1. As the number of samples increases to
@5 and @10, functional correctness rates improve, where T1
(Easy) benchmarks achieve correctness faster than T2 and T3,
and For the most T3 cases, SAGE-HLS consistently performs
the best (even for larger benchmarks).
In Fig. 3(a), we plot the training loss of two models (i.e.,
QWEN-HLS and SAGE-HLS) over 500 steps. As shown,
SAGE-HLS with the AST-enhanced model (red curve) con-
sistently converges faster, showing a more rapid drop in loss
during the initial training phase and maintaining a lower loss
value throughout the process compared to the QWEN-HLS
without AST (blue curve). Additionally, the SAGE-HLS curve
exhibits less fluctuation, suggesting a smoother optimization
trajectory. These observations indicate that including AST
information provides richer syntactic and semantic context,
thereby reducing ambiguity and improving the model’s ability
to learn code patterns and structural information more effi-
6Only 2 designs (synth@5) and 3 designs (synth@1) showed different
synthesizability results between QWEN-HLS and SAGE-HLS.
7Almost all circuits from trained models passed synthesizability in 2nd run.
TABLE II: SAGE-HLS Performance vs. Base LLM Model, Showcasing the Impact of Fine-tuning and AST-based Context Enahncent.
Evaluated ModelSynthesizability Ratio Functional Correctness Ratio
Synth@1 Synth@5 Synth@10 Pass@1 Pass@5 Pass@10
QWENCoder 7B Pre-trained ( QWEN Base Model) 52.56% 61.54% 70.51% 22.44% 38.46% 43.59%
QWENCoder 7B Fine-tuned using {HLS-C, Instruction }(QWEN-HLS ) 94.87% 98.72% 100% 56.41% 67.95% 71.79%
AST-Guide QWENCoder 7B Fine-tuned using {HLS-C, Instruction }(SAGE-HLS ) 92.95% 100% 100% 57.69% 70.51% 75.64%

100%
0%10%20%30%40%50%60%70%80%90%
SynthesisT2 Problems
0.000.250.500.751.001.25
Loss000
 QWEN HLS SAGE -HLS
0 100  200  300  400  500
(a)     (b)    (c)T1     T2     T3       T1     T2     T3      T1    T2    T3 
          01234567
Diff.
 Tiers8(out of 10)
Function passed
0
0
0
Steps910
QWEN HLSBase QWEN QWEN -HLS SAGE -HLSpass@1
pass@5
pass@10Fig. 3: Detailed Performance Analysis of SAGE-HLS: (a) Training Loss Comparison between QWEN-HLS and SAGE-HLS; (b) Synthesizability and Functional
Correctness on Different Tiers; (c) Detailed Functional Correctness for Pass@1/5/10 on Different Tiers.
ciently. Consequently, the model augmented with AST con-
verges to a lower loss, suggesting a more robust understanding
of the code and implying that AST-based training can offer
significant advantages in specialized language modeling tasks
such as HLS-C code generation.
Fig. 3(b) summarizes a radar plot comparing the synthesiz-
ability and functional correctness of models across different
benchmark difficulty tiers (for @1). This plot reflects the aver-
age synthesizability and functionality, accompanied with func-
tional correctness of different tiers. As shown, synthesizability
(for only one run) is effectively a solved problem for fine-
tuned models, achieving near-perfect results. For functional
correctness, there exists a significant gap between fine-tuned
models and the base one (from ∼20% to ∼60%). This anal-
ysis reinforces the necessity of structurally informed models
like SAGE-HLS, which consistently outperform standard fine-
tuned models (QWEN-HLS) in handling complex functional
dependencies in HLS-C generation.
Fig. 3(c) shows a detailed breakdown of functional cor-
rectness, showing how many of 10 different designs (per
tier) successfully passed functional verification in constrained
simulation. The designs are distributed across three difficulty
tiers (T1, T2, and T3 — each 10 separate designs) to illustrate
how model performance varies based on circuit complexity.
As shown T1 (and almost T2) circuits are easier to generate
functionally correct code for, while T3 see a sharp decline.
However, with multiple attempts (e.g., @10), T3 shows the
biggest improvement, particularly in SAGE-HLS, reinforcing
the importance of structural data analysis, such as AST guid-
ance, for more robust HLS-C code understanding.
TABLE III: Synth@1, @5, and @10 (Synthesis) for Selected Benchmarks.
BenchmarkQWEN Base QWEN-HLS SAGE-HLS
@1 @5 @10 @1 @5 @10 @1 @5 @10
popcount3 (T1) ✗ ✓ ✓ ✗ ✓ ✓ ✗ ✓ ✓
dff8r (T2) ✗ ✗ ✓ ✗ ✓ ✓ ✗ ✓ ✓
rule110 (T3) ✗ ✗ ✗ ✗ ✓ ✓ ✗ ✓ ✓
2013 q2bfsm (T3) ✗ ✗ ✗ ✗ ✓ ✓ ✗ ✓ ✓
lemmings3 (T3) ✗ ✗ ✗ ✗ ✓ ✓ ✗ ✓ ✓TABLE IV: Pass@1, @5, and @10 (Function) for Selected Benchmarks.
BenchmarkQWEN Base QWEN-HLS SAGE-HLS
@1 @5 @10 @1 @5 @10 @1 @5 @10
ringer (T1) ✗ ✓ ✓ ✗ ✓ ✓ ✗ ✓ ✓
countslow (T2) ✗ ✗ ✓ ✗ ✓ ✓ ✗ ✗ ✓
truthtable (T2) ✗ ✗ ✗ ✗ ✓ ✓ ✗ ✓ ✓
ece241 2014 q5b (T3) ✗ ✗ ✓ ✗ ✓ ✓ ✗ ✓ ✓
ece241 2013 q8 (T3) ✗ ✗ ✗ ✗ ✗ ✗ ✗ ✓ ✓
VI. C ONCLUSION
In this paper, we presented SAGE-HLS, a novel framework
that leverages AST-guided fine-tuning of an LLM to generate
synthesizable HLS-C code. By converting verified Verilog
designs into high-level C/C++ and enriching the training data
with their corresponding AST representations (control flow
analysis), our approach effectively enhances the alignment
between design intent and hardware implementation (and its
correctness). Our experimental evaluations on a modified set of
VerilogEval benchmark, coupled with constrained simulation,
demonstrate that SAGE-HLS improves the synthesizability of
HLS-C codes generated by LLM by 41%, reaching near-
perfect synthesis rates, while also boosting functional correct-
ness by more than 35%, i.e., more than doubling the number
of correctly verified cases compared to the baseline model.
REFERENCES
[1] P. Coussy et al. ,High-level synthesis . Springer, 2010, vol. 1.
[2] S. Lahti et al. , “Are we there yet? a study on the state of high-level
synthesis,” IEEE Transactions on Computer-Aided Design of Integrated
Circuits and Systems , vol. 38, no. 5, pp. 898–911, 2018.
[3] D. Gajski et al. ,High—Level Synthesis: Introduction to Chip and System
Design . Springer Science & Business Media, 2012.
[4] S. Liu et al. , “Accelerating fpga prototyping through predictive model-
based hls design space exploration,” in Proceedings of the 56th Annual
Design Automation Conference 2019 , 2019, pp. 1–6.
[5] J. Zhang et al. , “Towards automatic and agile ai/ml accelerator design
with end-to-end synthesis,” in 2021 IEEE 32nd International Conference
on Application-specific Systems, Architectures and Processors (ASAP) .
IEEE, 2021, pp. 218–225.
[6] J. Cong et al. , “Fpga hls today: successes, challenges, and opportunities,”
ACM Transactions on Reconfigurable Technology and Systems (TRETS) ,
vol. 15, no. 4, pp. 1–42, 2022.

[7] A. Cortes et al. , “High level synthesis using vivado hls for zynq soc:
Image processing case studies,” in 2016 Conference on design of circuits
and integrated systems (DCIS) . IEEE, 2016, pp. 1–6.
[8] S. Shi et al. , “Sechls: Enabling security awareness in high-level syn-
thesis,” in Proceedings of the 28th Asia and South Pacific Design
Automation Conference , 2023, pp. 585–590.
[9] TB. Brown et al. , “Language models are few-shot learners,” Advances
in neural information processing systems , vol. 33, pp. 1877–1901, 2020.
[10] J. Jiang et al. , “Structgpt: A general framework for large language model
to reason over structured data,” arXiv preprint arXiv:2305.09645 , 2023.
[11] A. M. Dakhel, V . Majdinasab, A. Nikanjam, F. Khomh, M. C. Desmarais,
Z. Ming, and Jiang, “Github copilot ai pair programmer: Asset or
liability?” 2023. [Online]. Available: https://arxiv.org/abs/2206.15331
[12] E. Nijkamp et al. , “Codegen: An open large language model for code
with multi-turn program synthesis,” arXiv preprint arXiv:2203.13474 ,
2022.
[13] S. Thakur et al. , “Verigen: A large language model for verilog code
generation,” 2023. [Online]. Available: https://arxiv.org/abs/2308.00708
[14] J. Blocklove et al. , “Chip-chat: Challenges and opportunities in conver-
sational hardware design,” in 2023 ACM/IEEE 5th Workshop on Machine
Learning for CAD (MLCAD) . IEEE, 2023, pp. 1–6.
[15] M. Liu et al. , “Chipnemo: Domain-adapted llms for chip design,” arXiv
preprint arXiv:2311.00176 , 2023.
[16] M. Akyash et al. , “Rtl++: Graph-enhanced llm for rtl code generation,”
inIEEE International Conference on LLM-Aided Design (ICLAD) , 2025,
pp. 1–7.
[17] M. Liu et al. , “Craftrtl: High-quality synthetic data generation
for verilog code models with correct-by-construction non-textual
representations and targeted code repair,” 2025. [Online]. Available:
https://arxiv.org/abs/2409.12993
[18] M. Akyash et al. , “Decortl: A run-time decoding framework for rtl code
generation with llms,” 2025, pp. 1–9.
[19] W. Fang et al. , “Assertllm: Generating and evaluating hardware verifica-
tion assertions from design specifications via multi-llms,” arXiv preprint
arXiv:2402.00386 , 2024.
[20] J. Bhandari et al. , “Llm-aided testbench generation and bug detection
for finite-state machines,” arXiv preprint arXiv:2406.17132 , 2024.
[21] N. Mashnoor et al. , “Llm-ift: Llm-powered information flow tracking
for secure hardware,” in IEEE 43rd VLSI Test Symposium (VTS) . IEEE,
2025, pp. 1–5.
[22] M. Akyash et al. , “Self-hwdebug: Automation of llm self-instructing for
hardware security verification,” in 2024 IEEE Computer Society Annual
Symposium on VLSI (ISVLSI) . IEEE, 2024, pp. 391–396.
[23] B. Ahmad et al. , “On hardware security bug code fixes by prompting
large language models,” IEEE Transactions on Information Forensics
and Security , vol. 19, pp. 4043–4057, 2024.
[24] M. M. Akyash and H. Kamali, “Evolutionary large language models for
hardware security: A comparative survey,” in Great Lakes Symposium
on VLSI (GLSVLSI) , 2024, pp. 496–501.
[25] S. Swaroopa et al. , “Evaluating large language models for automatic
register transfer logic generation via high-level synthesis,” arXiv preprint
arXiv:2408.02793 , 2024.
[26] X. Chenwei et al. , “Hlspilot: Llm-based high-level synthesis,” 2024.
[Online]. Available: https://arxiv.org/abs/2408.06810
[27] L. Yuchao et al. , “Are llms any good for high-level synthesis?” 2024.
[Online]. Available: https://arxiv.org/abs/2408.10428
[28] K. Xu et al. , “Automated c/c++ program repair for high-level synthesis
via large language models,” in Proceedings of the 2024 ACM/IEEE
International Symposium on Machine Learning for CAD , 2024, pp. 1–9.
[29] SA. Sheikholeslam et al. , “Synthai: A multi agent generative ai frame-
work for automated modular hls design generation,” arXiv preprint
arXiv:2405.16072 , 2024.
[30] H. Xu et al. , “Optimizing high-level synthesis designs with retrieval-
augmented large language models,” in 2024 IEEE LLM Aided Design
Workshop (LAD) . IEEE, 2024, pp. 1–5.[31] L. Collini et al. , “C2hlsc: Can llms bridge the software-to-hardware
design gap?” in 2024 IEEE LLM Aided Design Workshop (LAD) . IEEE,
2024, pp. 1–12.
[32] A. Oztas et al. , “Agentic-hls: An agentic reasoning based high-level
synthesis system using large language models (ai for eda workshop
2024),” arXiv preprint arXiv:2412.01604 , 2024.
[33] N. Mashnoor et al. , “Timelyhls: Llm-based timing-aware and
architecture-specific fpga hls optimization,” in IEEE International Con-
ference on Omni-layer Intelligent systems (COINS) . IEEE, 2025, pp.
1–6.
[34] A. Yang et al. , “Qwen2. 5 technical report,” arXiv preprint
arXiv:2412.15115 , 2024.
[35] M. Liu et al. , “Verilogeval: Evaluating large language models for
verilog code generation,” in 2023 IEEE/ACM International Conference
on Computer Aided Design (ICCAD) , 2023, pp. 1–8.
[36] K. Chang et al. , “Data is all you need: Finetuning llms for chip design
via an automated design-data augmentation framework,” in Proceedings
of the 61st ACM/IEEE Design Automation Conference , ser. DAC ’24.
New York, NY , USA: Association for Computing Machinery, 2024.
[Online]. Available: https://doi.org/10.1145/3649329.3657356
[37] H. Wu et al. , “Chateda: A large language model powered autonomous
agent for eda,” IEEE Transactions on Computer-Aided Design of Inte-
grated Circuits and Systems , 2024.
[38] S. Liu et al. , “Rtlcoder: Outperforming gpt-3.5 in design rtl generation
with our open-source dataset and lightweight solution,” in 2024 IEEE
International Workshop on LLM-Aided Design . IEEE, 2024.
[39] F. Cui et al. , “Origen:enhancing rtl code generation with code-
to-code augmentation and self-reflection,” 2024. [Online]. Available:
https://arxiv.org/abs/2407.16237
[40] Z. Pei et al. , “Betterv: Controlled verilog generation with discriminative
guidance,” 2024. [Online]. Available: https://arxiv.org/abs/2402.03375
[41] M. Gao et al. , “Autovcoder: A systematic framework for automated
verilog code generation using llms,” arXiv preprint arXiv:2407.18333 ,
2024.
[42] Y . Zhao et al. , “Codev: Empowering llms for verilog generation
through multi-level summarization,” 2024. [Online]. Available: https:
//arxiv.org/abs/2407.10424
[43] Z. Yujie et al. , “Mage: A multi-agent engine for automated rtl code
generation,” 2024. [Online]. Available: https://arxiv.org/abs/2412.07822
[44] J. Sun et al. , “Think-on-graph: Deep and responsible reasoning of large
language model on knowledge graph,” arXiv preprint arXiv:2307.07697 ,
2023.
[45] B. Fatemi et al. , “Talk like a graph: Encoding graphs for large language
models,” 2023. [Online]. Available: https://arxiv.org/abs/2310.04560
[46] B. Perozzi et al. , “Let your graph do the talking: Encoding structured
data for llms,” 2024. [Online]. Available: https://arxiv.org/abs/2402.
05862
[47] Z. Chai et al. , “Graphllm: Boosting graph reasoning ability of large
language model,” 2023. [Online]. Available: https://arxiv.org/abs/2310.
05845
[48] M. Brunsfeld et al. , “Tree-sitter: a parser generator tool and incremental
parsing library,” https://tree-sitter.github.io/tree-sitter/.
[49] H. Ye et al. , “Scalehls: A new scalable high-level synthesis framework
on multi-level intermediate representation,” in 2022 IEEE international
symposium on high-performance computer architecture (HPCA) . IEEE,
2022, pp. 741–755.
[50] F. Haedicke et al. , “Crave: An advanced constrained random verification
environment for systemc,” in 2012 International Symposium on System
on Chip (SoC) . IEEE, 2012, pp. 1–7.
[51] E. J. Hu, Y . Shen, P. Wallis, Z. Allen-Zhu, Y . Li, S. Wang, L. Wang,
W. Chen et al. , “Lora: Low-rank adaptation of large language models.”
ICLR , vol. 1, no. 2, p. 3, 2022.
[52] C. Lin, “Rouge: A package for automatic evaluation of summaries,” in
Text summarization branches out , 2004, pp. 74–81.