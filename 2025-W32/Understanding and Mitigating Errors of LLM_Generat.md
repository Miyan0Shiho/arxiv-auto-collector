# Understanding and Mitigating Errors of LLM-Generated RTL Code

**Authors**: Jiazheng Zhang, Cheng Liu, Huawei Li

**Published**: 2025-08-07 11:02:32

**PDF URL**: [http://arxiv.org/pdf/2508.05266v1](http://arxiv.org/pdf/2508.05266v1)

## Abstract
Despite the promising potential of large language model (LLM) based
register-transfer-level (RTL) code generation, the overall success rate remains
unsatisfactory. Errors arise from various factors, with limited understanding
of specific failure causes hindering improvement. To address this, we conduct a
comprehensive error analysis and manual categorization. Our findings reveal
that most errors stem not from LLM reasoning limitations, but from insufficient
RTL programming knowledge, poor understanding of circuit concepts, ambiguous
design descriptions, or misinterpretation of complex multimodal inputs.
Leveraging in-context learning, we propose targeted error correction
techniques. Specifically, we construct a domain-specific knowledge base and
employ retrieval-augmented generation (RAG) to supply necessary RTL knowledge.
To mitigate ambiguity errors, we introduce design description rules and
implement a rule-checking mechanism. For multimodal misinterpretation, we
integrate external tools to convert inputs into LLM-compatible meta-formats.
For remaining errors, we adopt an iterative debugging loop (simulation-error
localization-correction). Integrating these techniques into an LLM-based
framework significantly improves performance. We incorporate these error
correction techniques into a foundational LLM-based RTL code generation
framework, resulting in significantly improved performance. Experimental
results show that our enhanced framework achieves 91.0\% accuracy on the
VerilogEval benchmark, surpassing the baseline code generation approach by
32.7\%, demonstrating the effectiveness of our methods.

## Full Text


<!-- PDF content starts -->

1
Understanding and Mitigating Errors of
LLM-Generated RTL Code
Jiazheng Zhang, Cheng Liu, Senior Member, IEEE , and Huawei Li, Senior Member, IEEE
Abstract —Despite the promising potential of large language
model (LLM) based register-transfer-level (RTL) code gener-
ation, the overall success rate remains unsatisfactory. Errors
arise from various factors, with limited understanding of specific
failure causes hindering improvement. To address this, we con-
duct a comprehensive error analysis and manual categorization.
Our findings reveal that most errors stem not from LLM
reasoning limitations, but from insufficient RTL programming
knowledge, poor understanding of circuit concepts, ambiguous
design descriptions, or misinterpretation of complex multimodal
inputs. Leveraging in-context learning, we propose targeted error
correction techniques. Specifically, we construct a domain-specific
knowledge base and employ retrieval-augmented generation
(RAG) to supply necessary RTL knowledge. To mitigate ambigu-
ity errors, we introduce design description rules and implement a
rule-checking mechanism. For multimodal misinterpretation, we
integrate external tools to convert inputs into LLM-compatible
meta-formats. For remaining errors, we adopt an iterative debug-
ging loop (simulation-error localization-correction). Integrating
these techniques into an LLM-based framework significantly
improves performance. We incorporate these error correction
techniques into a foundational LLM-based RTL code generation
framework, resulting in significantly improved performance.
Experimental results show that our enhanced framework achieves
91.0% accuracy on the VerilogEval benchmark, surpassing the
baseline code generation approach by 32.7%, demonstrating the
effectiveness of our methods.
Index Terms —Large Language model, RTL code generation,
RTL error correction
I. I NTRODUCTION
RECENT advances in Large Language Models (LLMs) [1]
have significantly reshaped the landscape of automated
code generation. Specialized code generation models such as
Codex [2], CodeGen [3] and foundational LLMs such as
GPT [4] and DeepSeek [5] have demonstrated impressive
capabilities in producing syntactically correct and functionally
meaningful code across a wide range of programming lan-
guages and tasks. However, while LLMs achieve high success
rates for mainstream programming languages like Python, their
performance on hardware description languages (HDLs) [6]
such as Verilog remains noticeably lower. This discrepancy is
largely due to the scarcity of high-quality Verilog code in the
pretraining corpora, as well as the inherent domain-specific
complexity of RTL designs in general.
Nevertheless, errors in LLM-generated Verilog code arise
from a variety of sources. Some stem from a lack of fa-
The authors are with the State Key Laboratory of Processors, Institute
of Computing Technology, Chinese Academy of Sciences, Beijing 100190,
China, and also with the Department of Computer Science and Technology,
University of Chinese Academy of Sciences, Beijing 100190, China (e-mail:
zhangjiazheng24@mails.ucas.ac.cn, liucheng@ict.ac.cn).miliarity with Verilog-specific syntax and semantics [7]. For
example, the distinction between blocking and non-blocking
assignments is critical in Verilog but largely absent in general-
purpose programming languages. Other errors are rooted in
the model’s limited understanding of circuit-level concepts.
Concepts such as bit manipulation or sequential logic often
require domain-specific expertise that LLMs, trained primarily
on software code, may lack. For instance, operations based on
bit series or timing-sensitive constructs can be difficult for
LLMs to interpret and generate correctly. Given the diversity
and complexity of these errors, a one-size-fits-all solution is
unlikely to be effective. Therefore, a deeper understanding
of the types and causes of these errors is essential for sys-
tematically improving the quality of LLM-based RTL code
generation.
Recent research has taken important steps toward addressing
these RTL code generation challenges. Approaches such as
VeriGen [8] ,VerilogEval [9], RTLCoder [10], Betterv [11] and
DeepRTL [12] improve LLM performance by fine-tuning on
larger domain-specific Verilog datasets. Other methods, includ-
ing RTLFixer [13] and HDLDebugger [14], adopt retrieval-
augmented generation (RAG) strategies, incorporating external
knowledge bases or repair examples to compensate for the
models’ limitations in hardware design and debugging. Recog-
nizing the difficulty of resolving errors in a single step, recent
systems such as AutoChip [15], MEIC [16], and VeriAssistant
[17] employ iterative and multi-agent techniques to facilitate
multi-round refinement and repair of LLM-generated Verilog
code. Despite the varied methodologies, these approaches con-
sistently demonstrate improved RTL code quality, suggesting
that different techniques such as fine-tuning [18] can effec-
tively address similar errors. However, these advancements
often lack a systematic analysis of the root causes of the errors,
raising the question of whether such errors are truly resolved
or merely happen to be corrected by chance. This gap hinders
further improvement in LLM-based RTL code generation.
In this work, we present a comprehensive analysis of errors
in LLM-generated RTL code, categorizing them manually
according to their underlying causes. Our findings indicate
that the majority of errors do not stem from the reason-
ing limitations of LLMs, but rather from a lack of RTL
programming knowledge, inadequate understanding of cir-
cuit design principles, ambiguous design specifications, or
misinterpretation of complex multimodal inputs. Guided by
these insights, we leverage the in-context learning capabilities
of foundation models and introduce a set of targeted error
correction techniques. Specifically, we construct a domain-
specific knowledge base and apply retrieval-augmented gen-arXiv:2508.05266v1  [cs.AR]  7 Aug 2025

2
eration (RAG) to provide LLMs with the specialized RTL
programming and circuit design knowledge required for pre-
cise interpretation of design specification and accurate RTL
code generation. To address issues arising from ambiguous
design descriptions, we define a set of specification rules
and implement a rule-checking mechanism to ensure clarity
and completeness. For errors caused by misinterpretation of
multimodal data, we integrate external tools to convert such
inputs into meta-formats better suited for LLM processing.
Additionally, to handle residual errors such as those resulting
from incomplete contextual understanding in case of longer
context, we adopt a classic iterative debugging loop consisting
of simulation, error localization, and correction. We integrate
all these techniques into a foundational LLM-based RTL code
generation framework, achieving substantial improvements in
both code correctness and overall generation quality.
Our contributions are summarized as follows:
•We conduct an in-depth manual analysis of errors in
LLM-generated RTL code and categorize them based
on their root causes, revealing that most errors originate
from insufficient domain knowledge rather than reasoning
limitations of the models.
•Based on the identified causes of errors, we propose
a suite of correction techniques, including a domain-
specific knowledge base with RAG, a rule-based mech-
anism for description refinement, multimodal input con-
version tools, and an iterative simulation-based debugging
loop.
•We integrate the error correction mechanisms into a rep-
resentative LLM-based RTL code generation framework,
achieving 91.0% accuracy on the VerilogEval Benchmark
—— surpassing a series of RTL generation framework.
•We have open-sourced the erroneous RTL samples gener-
ated by GPT and other LLMs, along with their labeled re-
sults, analysis reports, and corresponding error-correction
code on GitHub1. We hope our work can contribute to
advancing LLM-assisted hardware design.
II. R ELATED WORK
A. LLMs for RTL Code Generation
With the rise of LLMs in the programming field, an in-
creasing number of researchers are considering using LLMs
for the automated generation of RTL designs from natural
language descriptions. DA VE [19] emerged as an early study
that fine-tuned GPT-2 [20] to automatically convert English
specifications into Verilog code. Along this line, VeriGen [8]
demonstrates that a fine-tuned open-source LLM (CodeGen-
16B [3]) can outperform GPT-3.5 Turbo [21] in generating
functionally correct Verilog code, highlighting the potential of
specialized smaller models for hardware design automation.
Subsequently, OriGen [22] enhances RTL dataset quality via
code-to-code improvement, while AutoVCoder [23] utilizes
data augmentation, two-phase fine-tuning, and domain-specific
1https://github.com/zjz1222/RTLErrorAnalysisRAG. Betterv [11] fine-tunes LLMs on domain-specific data
with generative discriminators for functionally correct code.
DeepRTL [12] proposes a unified CodeT5+-based model, fine-
tuned on Verilog code aligned with multi-level natural lan-
guage descriptions. Following this, methods like Aivril [24],
PromptV [25], and VerilogCoder [26] attempt to implement
RTL generation using multi-agent approach. VRank [27] and
MAGE [28] also employ multi-candidate sampling method
to assist high-quality RTL generation. PREFACE [29] intro-
duces a reinforcement learning based framework that enhances
LLMs in generating formally verifiable code by iteratively
refining prompts based on verification feedback.
B. LLMs for RTL Code Debugging
Despite the promising RTL code generation capabilities
of LLMs, the success rate of the LLM-based RTL code
generation remains unsatisfactory. An intuitive approach is
to repeat the code generation procedures multiple times and
select the best for the code generation, but this approach gen-
erally poses limited improvement and there is no guarantee for
better results due to the uncertainty of LLM-based generation.
Hence, debugging becomes a critical procedure to the LLM-
based RTL code generation, and tremendous efforts have been
devoted to fix the bugs and enhance the generated results
continuously. For example, RTLFixer [13] combines the ReAct
framework and RAG technology to fix compilation errors
using a knowledge base of expert insights. HDLDebugger [14]
enhances LLM error correction by integrating RAG with fine-
tuning and a code sample library. AutoChip [15] employs
a multi-round iterative approach guided by compilation and
simulation feedback, while MEIC [16] employs a multi-agent
framework with iterative refinement to correct both syntac-
tic and functional errors in Verilog code. VeriAssistant [17]
implements Verilog code repair through iterative LLM self-
verification and self-correction. VeriDebug [30] introduces a
contrastive representation and guided correction approach for
automated Verilog debugging.
Prior studies have demonstrated the significant potential of
using LLMs for RTL code generation and debugging. How-
ever, LLMs remain susceptible to introducing errors during
RTL generation, and the underlying reasons of these errors
remain insufficiently understood. Moreover, there is a lack of
systematic characterization of the causes and manifestations
of errors introduced during RTL generation with foundational
LLMs. Such characterization is crucial for identifying major
challenges and guiding future research on LLM-based RTL
code generation. In the next section, we will explore an
important question: Why do LLMs produce errors in RTL
generation?
III. E RROR ANALYSIS OF LLM-G ENERATED RTL C ODE
To understand the RTL code generation capabilities of
foundational LLMs, we analyze the errors in the RTL code
produced by these models and investigate their root causes. In
our study, we use GPT models as representative LLMs and
adopt VerilogEval as the benchmark suite for generating RTL
code from natural language specifications. A standard RTL

3
 System Pr ompt:
 You only complete chats with syntax correct Verilog code. End   
 the Verilog module code completion with ' endmodule '. Do not 
 include module, input and output definitions.
 User  Prompt:
 Implement the Verilog module based on the following description.
 Implement a 4-bit adder with full adders. The output sum should 
 include the overflow bit. 
 module TopModule (
   input [3:0] x,
   input [3:0] y ,
   output [4:0] sum
 );
Fig. 1: The basic prompt used for RTL code generation.
code generation prompt is shown in Fig. 1. The generated
RTL code is verified and simulated using Icarus Verilog.
Considering that models with different parameter scales and
types vary in their ability to generate RTL code, to ensure the
completeness of the analysis, we conducted experiments with
the Qwen-Coder-32B-Instruction model which has a small
scale and was additionally pre-trained on code, the general-
purpose GPT-3.5 Turbo model of ordinary size, and the more
capable GPT-4 Turbo model. And we respectively collected
83, 89, and 65 faulty programming samples. We manually
examine the failed cases and classify them into two primary
categories as listed below.
•Type I: Insufficient Knowledge of Specialized RTL
Programming (IKSP.): In these cases, the LLM cor-
rectly interprets the design specification but fails due to
a lack of specialized knowledge in RTL programming. If
the same specification were used to generate code in a
more widely used language like Python, the generation
would likely succeed.
•Type II: Misinterpretation of Design Specifications
(MDS.): Here, the LLM fails to correctly interpret the
design intent, which may stem from either insufficient
understanding of circuit design concepts or general mis-
comprehension of the specifications. This type of error
would likely persist even if a high-level language like
Python were used.
For different categories of LLMs, we conducted a detailed
statistical analysis of the distribution of the two aforemen-
tioned error types, as shown in Fig 2. The results indicate
that the majority of design errors (over 70%) stem from
misinterpretations of design specifications, and this propor-
tion increases with the model’s programming capability. In
contrast, errors related to RTL-specific programming account
for only a minor portion. A detailed analysis of both error
categories is provided in the following subsections.
A. Insufficient Knowledge of Specialized RTL Programming
Although Verilog shares similarities with C-like languages,
it incorporates specialized syntax and is deeply intertwined
with circuit design concepts, making it substantially different
from conventional high-level languages such as Python or
C/C++. For example, Verilog enforces distinct assignment
semantics for different contexts: blocking assignments andnon-blocking assignments are used for combinational and se-
quential logic, respectively. Foundational LLMs often struggle
with such distinctions due to the limited availability of Verilog-
specific data in their training corpora. As a result, many RTL
coding errors stem from insufficient understanding of RTL
programming. According to our experiments on VerilogEval,
we identified a total of 41 failed cases, which can be roughly
categorized into four major groups, as shown in Table I.
Wire in Always Block: The LLM often fails to correctly
interpret RTL variable usage specifications, leading to incor-
rect assignments to variables declared as wire types within
always blocks, as shown in Table I. This type of error is more
likely to occur in smaller-scale models such as Qwen-Coder-
32B-Instruction. However, for larger-scale models like GPT-4
Turbo, as the training corpus and model parameters increase,
the frequency of such errors decreases significantly.
Numerical Processing Logic Error: In some design con-
texts, LLMs are prone to errors in numerical processing logic.
For instance, as illustrated in the Table I. When implementing
the function of counting the number of 1’s in the vector in[1:0],
the LLM incorrectly equates in[1:0] with in[1] + in[0].
Vector Bit Selection Error: In some design scenarios,
LLMs are prone to boundary violations when selecting bits
from vectors.
Inversion of Vector Slice Selection: Likewise, in some
RTL design contexts, LLMs occasionally perform vector
slicing in a direction inconsistent with the original vector
definition, leading to compilation failures.
Incomplete RTL Code: The LLM correctly interprets the
design specifications, but generates only a portion of the
required code. It often omits the remaining code inappropri-
ately, particularly when the expected output is lengthy and
repetitive. As shown in Table I, the model uses a comment to
substitute the remaining branching cases in a typical switch
block, instead of generating all 256 possible branches.
Variable Redefinition: LLM-generated RTL code often
exhibits redefinitions of variables, particularly when the same
variables serve as both registers and output signals. This type
of error occurs relatively frequently in GPT-3.5 Turbo.
Using Undefined Variables: In some RTL designs, LLMs
may improperly reference undeclared variables during RTL
generation. As shown in Table I, when implementing the
mux in[0] signal, the LLM incorrectly uses undefined vari-
ables ’a’ and ’b’, resulting in compilation errors.
Incorrect Use of Generate Statement: In some cases,
LLMs may confuse ’generate for’ constructs with regular ’for’
statements. The case in Table I shows erroneous usage where
the LLM attempted to iteratively update individual bits of
register ’q’ through generate statements, violating synthesis
rules and leading to malfunction.
Mixing Blocking and Non-blocking Assignments: In cer-
tain always blocks, LLMs exhibit a tendency to mix blocking
and non-blocking assignments, resulting in functional errors.
Based on the statistics above, we can observe that Qwen-
Code-32B-Instruction tends to make errors in numerical and
vector processing logic, GPT-4 Turbo often generates incom-
plete code or misuses advanced RTL syntax like Generate,
while GPT-3.5 Turbo produces a wider variety of errors.

4
TABLE I: Analysis of errors induced by insufficient knowledge of specialized RTL programming
Error Category Error Examples Golden Generation QwenCoder
SamplesGPT-3.5
SamplesGPT-4
Samples
Wire in Always Blockoutput out;
always @( *) begin ...
case (state) ...
B: out = 1;
endoutput out;
assign out = (state == B);9 8 1
Numerical Processing
Logic Error// Count the number of 1s in the first two bits
assign count1 = in[1:0];
// Count the number of 1s in the last bit ...
assign count2 = count1 + in[2];
assign out = count2;assign out = in[0]+in[1]+in[2];2 1 0
Vector Bit Selection
Errormodule TopModule (input clk,
... output [3:1] g
);
assign g[1] = (state == B);
assign g[0] = 1’b0;module TopModule (input clk,
... output [3:1] g
);
assign g[1] = (state == B);1 3 0
Inversion of Vector
Slice Selectionmodule TopModule (
input [99:0] in, ...
output [99:1] out_any,
);
assign out_any = in[99:1] |in[0:98];module TopModule (
input [99:0] in, ...
output [99:1] out_any,
);
assign out_any = in[99:1] |in;1 1 0
Incomplete RTL Codeassign out = (sel == 0) ? in[0]:
(sel == 1) ? in[1]:
//Continue this pattern for all 256 inputs
in[255] ;assign out = in[sel] ;0 4 3
Variable Redefinitionmodule TopModule(input clk,
..., output reg [7:0] q);
reg [7:0] q;module TopModule(input clk,
..., output reg [7:0] q);0 4 0
Using Undefined
Variablesmodule TopModule (input c, d,
output [3:0] mux_in
);
wire w1, w2 ...;
assign mux_in[0] = a & b ? w1 : w2;
//The variables a and b are undefined.module TopModule (input c, d,
output [3:0] mux_in
);
assign mux_in[0] = c|d ;0 1 1
Incorrect Use of
Generate Statementmodule TopModule(input clk, ...,
output reg [511:0] q);
always @(posedge clk) begin ...
genvar i;
generate
for (i = 1; i < 511; i = i + 1)
begin: gen_rule90
always @(posedge clk)
q[i] <= q[i-1] ˆ q[i+1];
end
endgenerate ...
endmodule TopModule(input clk, ...,
output reg [511:0] q);
always @(posedge clk) begin
integer i;
for (i = 1; i < 511; i = i + 1)
begin
q[i] <= (q[i-1] ˆ q[i+1]);
end
end0 1 1
Mixing Blocking and
Non-blocking
Assignmentsalways @(posedge clk) begin ...
q[0] = (q[1] ˆ 0);
for (i = 1; i < 511; i = i + 1)
q[i] <= (q[i-1] ˆ q[i+1]);
endalways @(posedge clk) begin ...
q[0] <= (q[1] ˆ 0);
for (i = 1; i < 511; i = i + 1)
q[i] <= (q[i-1] ˆ q[i+1]);
end0 1 0
This indicates that both specialized training on code data for
smaller models and simply scaling up model parameters and
training data can help reduce errors caused by insufficient RTL
programming knowledge in LLMs.
B. Misinterpretation of Design Specifications
Beyond the errors resulting from insufficient knowledge of
specialized RTL programming, the majority of issues arise
from misinterpretation of design specifications. These misin-
terpretations can be categorized into four main types. Detailed
error categories will be illustrated in the rest of the subsections.
1) Insufficient Understanding of Circuit Concepts (IUCC.):
In this case, the LLM fails to correctly interpret the specifica-
tion due to the absence of key circuit design concepts in the
description, leading to incorrect code generation. Furthermore,
the lack of circuit-related concepts can be categorized into four
main causes: inadequate grasp of timing principles, insufficientunderstanding of specific design, limitations in numerical
and vector processing capabilities, and deficiencies in state
machine design for complex scenarios. These gaps collectively
lead to constrained comprehension of design specifications.
Inadequate Understanding of Timing Related Concepts
(TRC.): In this scenario, the LLM’s lack of domain-specific
knowledge in timing-related concepts leads to functional errors
in the generated RTL code. These errors include mishandling
synchronous/asynchronous logic, improper previous signal
latching, and incorrect timing distinctions in assign vs. always
block assignments. As shown in Fig. 3(a), the LLM incorrectly
sets the sensitivity signals in the always block due to a lack
of understanding of timing concepts in synchronous/asyn-
chronous programming. Additionally, as illustrated in Fig.
3(b), because the LLM lacks knowledge about tracking the
predecessor signal (prev in) in edge detection, it incorrectly
clears prev in within the reset block. This prevents the module

5
IKSP .7.2%MDS. 84.3%
IKSP . & MDS.8.4%Qwen-Coder-32B-Instruction
IKSP .
13.5%
MDS. 71.9%
IKSP . & MDS.14.6%GPT-3.5 Turbo
IKSP .
10.8%
MDS.87.7%IKSP . & MDS. 1.5%GPT-4 Turbo
Fig. 2: The distribution of error type ratios in RTL code generation scenarios.
 Design Description  (Pr ob035_count1to10 ）
 Make a decade counter that counts 1 through 10, inclusive. The reset input is active high
synchronous , and   should reset the counter to 1.
LLM's Response ：  
module TopModule ( input clk,  reset,
    output reg [3:0] q
);
    always @(posedge clk or posedge reset)  
    begin
        if (reset) begin
            q <= 4'b0001; // Reset to 1
        end else begin
            if (q == 4'b1010) begin
                 q <= 4'b0001; 
            end else begin
                 q <= q + 1; 
            end
        end
   end
endmoduleGolden Refer ence ：  
module TopModule (input clk, reset,
    output reg [3:0] q
);
    always  @(posedge clk)
if (reset || q == 10)
q <= 1;
else
q <= q+1;
endmodule
 
 Design Description  ( Prob066_edgecaptur e）
 For each bit in a 32-bit vector , capture when the input signal changes from 1 in one clock cycle to 0 
 the next. "Capture" means that the output will remain 1 until the register is reset (active high 
 synchronous reset).
LLM's Response ：  
module TopModule (
    input clk, reset,
    input [31:0] in,
    output reg [31:0] out
);
    reg [31:0] prev_in;
    always @(posedge clk) begin
        if (reset) begin
            out <= 32'b0;
            prev_in <= 32'b0;
        end else begin
            prev_in <= in;
            out <= out | (prev_in & ~in);
        end
    end
endmoduleGolden Refer ence ：  
module TopModule ( input clk,
  input reset,
  input [31:0] in,
  output reg [31:0] out
);
  reg [31:0] d_last;
  always @(posedge clk) begin
      d_last <= in;
      if (reset)
          out <= '0;
      else
          out <= out | (~in & d_last);
  end
endmodulea. An example of synchronous programming error 
b. An example of edge capture programming error 
Fig. 3: Examples of inadequate understanding of timing related
concept
from correctly recording the previous state of in after reset,
leading to functional errors.
Insufficient Understanding of Specific Design (SP.): In
such cases, LLMs may struggle to comprehend design specifi-
cations due to unfamiliarity with specific RTL design, leading
to coding errors. This typically manifests as inadequate un-
derstanding of specialized designs such as dual-edge triggered
circuits, LFSRs (Linear Feedback Shift Registers), adder carry
chains, and other domain-specific implementations. As shown
in Fig 4, due to limited exposure to specialized designs like
dual-edge during training, the LLM simply combines q pos
and q neg through an OR operation in its final output logic,
failing to fully account for the clock signal’s level state to
determine the correct output, thereby causing functional errors.
Design Description （ Prob078_dualedge ）：
A dual-edge triggered flip-flop is triggered on both edges of the clock. However ,
FPGAs don't have dual-edge triggered flip-flops, and always @(posedge clk or
negedge clk) is not accepted as a legal sensitivity list.
 Build a circuit that functionally behaves like a dual-edge triggered flip-flop.
LLM's Response:
module TopModule ( input clk,  d,
    output reg q
);
    reg q_pos,  q_neg;
    always @(posedge clk) begin
        q_pos <= d;
    end
    always @(negedge clk) begin
        q_neg <= d;
    end
    always @(*) begin
        q = q_pos | q_neg;
    end
endmoduleGolden Refer ence:
module TopModule ( input clk,  d,
    output reg q
);
  reg qp, qn;
  always @(posedge clk)
    qp <= d;
  always @(negedge clk)
    qn <= d;
  always @(*)
    q <= clk ? qp : qn;
endmoduleFig. 4: An example of insufficient understanding of specific design
 Design Description  ( Prob071_always_casez ）
 Build a priority encoder for 8-bit inputs. Given an 8-bit vector , the output should report the first 
 (least significant) bit in the vector that is 1. Report zero if the input vector has no bits that are high. 
 For example, the input 8'b10010000 should output 3'd4, because bit[4] is first bit that is high.
LLM's Response ：  
module TopModule (
    input [7:0] in,
    output reg [2:0] pos
);
    always @(*) begin
         case(in)
            8'b 00000001 : pos = 3'b0;
            8'b 00000010 : pos = 3'b1;
            8'b 00000100 : pos = 3' b2;
            8'b 00001000 : pos = 3' b3;
            8'b 00010000 : pos = 3' b4;
            8'b 00100000 : pos = 3' b5;
            8'b 01000000 : pos = 3' b6;
            8'b 10000000 : pos = 3' b7;
            default: pos = 3'b0;
        endcase
    end
endmoduleGolden Refer ence ：  
module RefModule (
    input [7:0] in,
    output reg [2:0] pos
);
    always @(*) begin
        casez (in)
            default : pos = 2'h0;
            8'b zzzzzzz1 : pos = 3'h0;
            8'b zzzzzz1z : pos = 3'h1;
            8'b zzzzz1zz : pos = 3' h2;
            8'b zzzz1zzz : pos = 3' h3;
            8'b zzz1zzzz : pos = 3' h4;
            8'b zz1zzzzz : pos = 3' h5;
            8'b z1zzzzzz : pos = 3' h6;
            8'b 1zzzzzzz : pos = 3' h7;
        endcase
    end
endmodule
Fig. 5: An example of limitations in numerical and vector processing
capabilities
Limitations in Numerical and Vector Processing Capa-
bilities (NVP.): Additionally, due to potential limitations in its
understanding of numerical and vector processing, the LLM
struggles to align high-level natural language descriptions with
low-level RTL code, leading to errors. Typical examples of
such errors include incorrect sign-bit handling logic, mis-
configured case-condition vectors, and misunderstandings of
least-significant-bit (LSB) concepts—all of which stem from
deficiencies in numerical and vector comprehension. As shown
in Fig 5, due to LLMs’ limited understanding of vector fuzzy

6
matching (casez), when processing input vectors containing
multiple ’1’s, the generated RTL code incorrectly selects the
default branch, leading to functional errors. Additionally, the
LLM-produced code contains numerical representation errors,
such as invalid binary formats like 3’b2, where digits other
than 0 and 1 are incorrectly included, further causing failures.
Design Description ( Prob089_ece241_2014_q5a) :
You are to design a one-input one-output serial 2's complementer Moore  state machine.
The input (x) is a series of bits (one per clock cycle)  beginning with the least-
significant bit of the number, and the output  (Z) is the 2's complement of the input.
The machine will accept input numbers of arbitrary length. The circuit requires a
positive edge  triggered asynchronous reset. The conversion begins when Reset
is released and stops when Reset is asserted.
LLM‘s Response ： Golden Refer ence ：
Fig. 6: An example of deficiencies in state machine design for
complex scenarios
Deficiencies in State Machine Design for Complex Sce-
narios (SMDC.): Moreover, when LLMs process high-level
functional descriptions involving state machine design—such
as sequence detection problems—their limited understanding
of state machine programming often results in errors in state
encoding, transition conditions, and output logic. A representa-
tive example from VerilogEval is Prob089 ece241 2014 q5a,
as illustrated in Fig. 6. The specification requires a Moore
machine that outputs the binary complement of the serial input.
However, the LLM fails to construct an appropriate finite
state machine based on this description, resulting in functional
errors in the generated RTL code.
TRC. SP. NVP. SMDC.051015202530# of error subtypes under ADD.23
461725
4820
34
314Qwen-Coder-32B GPT-3.5 Turbo GPT-4 Turbo
Fig. 7: Distribution of error subtypes under IUCC.
We analyzed the distribution of sub-error categories under
the ’Insufficient Understanding of Circuit Concepts’ scenario
(Fig. 7), revealing that GPT-4 Turbo consistently demonstrated
the lowest error rates across all subcategories, while GPT-
3.5 Turbo showed the highest frequency of errors; Qwen-
Coder-32B-Instruction’s performance fell between GPT-3.5
Turbo and GPT-4 Turbo. The analysis further reveals that
both Qwen-Coder-32B and GPT-3.5 Turbo show significantlyweaker comprehension of temporal concepts (particularly syn-
chronous/asynchronous understanding) compared to GPT-4
Turbo, while all models demonstrate similar performance in
special-design programming and state machine design for
complex scenarios, indicating these knowledge gaps represent
critical bottlenecks limiting LLMs’ programming accuracy.
 Design Description  ( Prob045_edgedetect2 ）
 For each bit in an 8-bit vector , detect when the input signal changes from one clock cycle to the 
 next  (detect any edge).  The output bit should be set the cycle after a  0 to 1 transition occurs .
LLM's Response ：  
module TopModule (
    input clk,
    input [7:0] in,
    output reg [7:0] anyedge
);
reg [7:0] in_prev;
always @(posedge clk) begin
    in_prev <= in;
    anyedge <= (in & ~in_prev); 
    // Detect rising edge
end
endmoduleGolden Refer ence ：  
module RefModule (
    input clk,
    input [7:0] in,
    output reg [7:0] anyedge
);
    reg [7:0] d_last;
    always @(posedge clk) begin
        d_last <= in;
        anyedge <= in ^ d_last;
    end
endmodule
  Design Description  (Pr ob028_m2014_q4a)   
  Implement a D latch using an always block. Design Description  ( Prob034_dff8 )     
 Create 8 D flip-flops. All DFFs should be   
 triggered by the positive edge of clk.
LLM's Response ：  
module TopModule (input clk, input [7:0] d,
       output reg [7:0] q
);
   always @(posedge clk) begin
       q <= d;
   end
endmoduleLLM's Response ：  
module TopModule (input d, input ena,
       output logic q
);
always @(posedge ena)  begin
    if (ena) q <= d;
end
endmodule
Golden Refer ence ：  
module TopModule (input d, input ena,
       output logic q
);
always @(*) begin
    if (ena) q <= d;
end
endmoduleGolden Refer ence ：  
module TopModule (input clk, input [7:0] d,
       output reg [7:0] q
);
     initial q = 8'h0;
     always @(posedge clk)
         q <= d;
endmodulea. An example of  unclear module's overall function description 
b. An example of  ambiguous module's
input/output signal descriptionc. An example of  missing module initialiaztion
Fig. 8: Examples of ambigious design descriptions
2) Ambiguous Design Descriptions (ADD.): We observe
that a significant portion of errors stem from ambiguous
design descriptions. In such cases, the LLM may interpret
the ambiguous information differently from the fixed golden
reference implementation. Ambigious design descriptions can
be primarily categorized into three aspects: unclear overall
module functionality, ambiguous module input/output signal
descriptions, and lack of initialization.
Unclear Overall Module Functionality (UOMF.): This
type of error primarily manifests as internal contradictions
in the overall functional description of the module or in-
consistencies between the description and the Golden Design
implementation. As shown in Fig. 8(a), the first part of the
design description requires detecting any edge, while the end
specifies capturing only rising edges (0 to 1). Due to this
inherent contradiction, the LLM fails to correctly interpret the
user’s intent, leading to design errors.
Ambiguous Module Input/Output Description (AIOD.):
This type of error manifests as insufficient description of the
basic functionality of each input/output signal. As shown in
Fig. 8(b), the design description does not clearly specify the
triggering mechanism of ena, leading the LLM to incorrectly

7
assume it is edge-triggered rather than level-triggered, thereby
introducing functional errors.
Missing Module Initialization Configuration (MMI.):
Some design errors are caused by the description omitting
module initialization information. As shown in Fig. 8(c), due
to missing information in the description, the LLM-generated
code lacks the initialization module compared to the Golden
Design, resulting in functional errors.
UOME. AIOD. MMI.012345# of error subtypes under ADD.4
3 34 4
3 3
23Qwen-Coder-32B GPT-3.5 Turbo GPT-4 Turbo
Fig. 9: Distribution of error subtypes under ADD.
We analyzed the distribution of subcategory errors under
ambiguous design descriptions, as shown in Fig 11. The
results indicate that, regardless of model capability, all models
exhibit similar performance across different error categories.
Comparatively, GPT-4 Turbo performs slightly better than the
other two models, suggesting it generates marginally more
robust code in scenarios with amibiguous descriptions.
a. Karnaugh Map b. Table
c.  State Transition Diagram (T extual Format) d. Waveform (T extual Format)
Fig. 10: Examples of multimodal data in hardware circuit design
3) Misinterpretation of Multimodal Data (MMD.): Another
significant source of code generation errors arises from the
presence of multimodal data in design descriptions, such as
Karnaugh maps (KMAP.) ,Tables (TB.) ,State Transition
Diagrams (ST.) , and Waveform Diagrams (WA V .) , as illus-
trated in Fig. 10. We observed that when LLMs process data in
the aforementioned format, they struggle to accurately convert
such multimodal information into correct RTL code due to
limitations in their training data distribution.
We analyzed the distribution of subcategory errors caused
by misinterpretation of multimodal information. The results
show that all LLMs performed poorly in comprehending
KMAP. TB. ST. WAV.02468101214# of error subtypes under MMD.7
48 8
7
3712
7
169Qwen-Coder-32B GPT-3.5 Turbo GPT-4 TurboFig. 11: Distribution of error subtypes under MMD.
multimodal information, with each model exhibiting over 25
design errors attributable to such misinterpretations.
Notably, GPT-4 Turbo demonstrated significantly better
performance in parsing tabular data compared to GPT-3.5
Turbo and Qwen-Coder-32B-Instruction. However, for Kar-
naugh maps and state transfer diagrams, all models showed
comparable performance. The identical error rates on KMAPs
suggest that neither Qwen nor GPT series models have re-
ceived specialized training on this data type.
Interestingly, in interpreting textual waveform descriptions
(textual format), Qwen-Coder-32B-Instruction slightly out-
performed GPT-4 Turbo, indicating that domain-specialized
smaller models may possess capabilities absent in general
large language models.
Design Description ( Prob094_gatesv )：
You are given a four -bit input vector in[3:0]. We want to know some
relationships between each bit and its neighbour .
....
 (3) out_dif ferent: Each bit of this output vector should indicate whether the
corresponding input bit is dif ferent from its neighbour  to the left.  
For example, out_dif ferent[2] should indicate if  in[2] is  differ ent fr om in[3] .
...
LLM's Response ：
module TopModule  (input [3:0] in,
  ...
  assign out_different[2] = in[2] ^ in[1];
  ...
endmoduleLong Context
Fig. 12: An example of code generation errors induced by long
descriptions
4) Missing Details of Long Descriptions (MDLD): Al-
though recent LLMs claim to support long-context under-
standing, we observe that they often focus primarily on the
main instructions in the design descriptions, while overlooking
finer details. While such omissions may not significantly
impact general question-answering tasks, they pose substantial
challenges in LLM-based code generation, where missing even
minor details can result in functional errors. As illustrated in
Fig. 12, the generated code fails to include the bit selection
logic for out difference, leading to incorrect behavior eventu-
ally.
5) Error Distribution Statistics Under Misinteroretation of
Design Specifications: We statistically analyzed the distribu-
tion of the aforementioned error types across different LLMs,
as illustrated in Fig 13.
Our analysis reveals that for all evaluated LLMs, the pri-
mary bottleneck in generating correct RTL code stems from

8
IUCC. ADD. MMI. MDLD.01020304050# of error subtypes under MDS.45
1027
1043
1129
1123
823
10Qwen-Coder-32B GPT-3.5 Turbo GPT-4 Turbo
Fig. 13: Errorous design distribution of error types under MDS.
insufficient circuit knowledge leading to design description
misinterpretations. This is particularly pronounced in Qwen-
Coder-32B and GPT-3.5 Turbo, where over 40 erroneous
designs originated from this limitation. In contrast, GPT-
4 Turbo demonstrates significantly better comprehension of
circuit programming concepts, suggesting that scaling model
parameters and training data - particularly to enhance circuit
knowledge - effectively reduces RTL generation errors.
A substantial portion of errors also arises from multimodal
information processing failures. The comparable performance
across models in this category indicates that neither Qwen
nor GPT series models have received adequate cross-modal
alignment training specific to hardware design domains.
Additionally, a smaller but consistent error category involves
ambiguous descriptions and limited long-context understand-
ing. Performance in this aspect remains similar across model
scales, from small models to larger GPT variants. Enhancing
LLMs’ robustness in generating RTL from vague specifi-
cations and improving their long-context comprehension in
hardware design contexts remain critical research challenges.
C. Combination of Multiple Errors
When generating larger-scale designs, LLM-produced RTL
code may contain multiple coexisting errors, rather than being
limited to a single error type.
As shown in Fig 14, the LLM initially sets the syn-
chronously triggered reset as asynchronously triggered due to
inadequate grasp of timing related concepts . Second, be-
cause of its limited understanding of numerical and vector
concepts , when handling data transmitted in least-significant-
first order, the LLM incorrectly configures the data recording
direction as left-shift instead of the required right-shift. Ad-
ditionally, due to deficiencies in RTL-specific programming
knowledge , it mistakenly assigns a value to ’done’ (a wire
type) inside an always block, leading to compilation errors.
Finally, in complex state machine design scenarios, the LLM
fails to account for error handling in the data transmission
protocol. Upon reaching the STOP state, it only considers
transitioning to DONE when the input is 1, neglecting the case
where the input is 0—which requires waiting for an additional
stop bit before returning to the START state. We attribute
this error to the LLM’s limited knowledge of state machine
design for complex scenarios . For such erroneous designs,
we manually categorize them under each of these root causes
simultaneously.
 Design Description  ( Prob146_fsm_serialdata ）
 In many (older) serial communications protocols, each data byte is sent along with a start bit and a  stop 
 bit, to help the receiver delimit bytes from the stream of bits. One common scheme is to use one  start 
 bit  (0), 8 data bits, and 1 stop bit (1). The line is also at logic 1 when nothing is being transmitted   
 (idle).  Design a finite state machine that will identify when bytes have been correctly received when 
 given a stream of bits. It needs to identify the start bit, wait for all 8 data bits, then verify that the stop
bit   was correct.  The module will also output the correctly-received data byte. out_byte needs to be valid 
 when done is 1, and is don't-care otherwise. If the stop bit does not appear when expected, the   FSM 
 must wait until it finds a stop bit before attempting to receive the next byte.  Include a active-high 
 synchronous reset . Note that the serial protocol sends the least significant bit first . It should assert   done 
 each time it finds a stop bit.
Golden Refer ence ：  
module TopModule (input clk, in, reset,
    output [7:0] out_byte,
    output done
);
...
always_comb begin
    case (state)
    START: next = in ? START : B0; 
    // start bit is 0
    B0: next = B1;
    B1: next = B2;
    B2: next = B3;
    B3: next = B4;
    B4: next = B5;
    B5: next = B6;
    B6: next = B7;
    B7: next = STOP;
    STOP: next = in ? DONE : ERR; 
    // stop bit is 1. Idle state is 1.
    DONE: next = in ? START : B0;
    ERR: next = in ? START : ERR;
    endcase
end
always @(posedge clk) begin
    if (reset) state <= START;
    else state <= next;
end
always @(posedge clk) begin
    byte_r <= {in, byte_r[9:1]};
end
assign done = (state==DONE);
assign out_byte = done ? byte_r[8:1] :  8'hx;
endmoduleLLM's Response ：  
module TopModule (input clk, in,reset,
    output [7:0] out_byte,
    output done
);
...
always @(posedge clk or posedge reset)  
begin
    if (reset) begin
    ...
    end
    else begin
         case (state)
         3'b000: begin 
         if (in == 1'b0) begin
             state <= 3'b001; 
             start_bit_detected <= 1'b1;
             end
         end
         3'b001: begin 
         if (bit_count < 4'd8) begin
             data_reg <= {data_reg[6:0], in};
             bit_count <= bit_count + 1;
         end
         else begin
             state <= 3'b010; 
        // Move to stop bit detection state
         end
         end
          3'b010: begin 
         if (in == 1'b1) begin
             stop_bit_detected <= 1'b1;
             out_byte <= data_reg;
             
             done <= 1'b1;
             state <= 3'b000; 
         // Move back to idle state
         end
         end
         endcase
   end
end
endmoduleError 1: synchronous reset
Error 2:  least signficant bit first
Error 3:  wire in alway block
Error 4:  lack of error state design
and handlingFig. 14: An example of combination of multiple errors
While some unclassified errors are categorized as miscella-
neous, this work focuses on the identified ones. When LLMs
understand design specifications, generated code typically
aligns with golden implementations, with most errors arising
from RTL programming knowledge gaps. For misinterpreted
designs, causes vary—from limited circuit design understand-
ing to ambiguous specifications or complex data formats.
Certain errors, like missing long-context details, stem from
inherent LLM limitations. Addressing these diverse errors re-
quires targeted strategies to enhance LLMs’ domain reasoning,
disambiguation, multimodal understanding, and long-context
retention.
IV. LLM- BASED ERROR CORRECTION
Based on the above error analysis, we propose a set of
correction mechanisms tailored to address different types
of errors. For issues stemming from a lack of specialized
RTL programming knowledge or circuit design concepts, we
generally employ Retrieval-Augmented Generation (RAG) to
supplement missing knowledge using representative examples,
which are then provided as context to the LLMs. To handle
ambiguities in design specifications, we introduce a rule-
based checking mechanism that clarifies and completes the

9
descriptions. For some of the knowledge-related errors such
as text misunderstanding that non-blocking assignments apply
only to registers in sequential logic design, it is more like a
design rule and is widely applied in the RTL design. In this
case, we encode such domain knowledge as explicit rules and
include them in the system context during code generation. To
overcome challenges posed by multimodal data, we integrate
external tools that assist in interpreting circuit design elements,
enabling LLMs to utilize these tools for more effective RTL
code generation. For errors caused by miscellaneous issues or
the limited reasoning capabilities of LLMs such as omitting
details in long-context scenarios, we adopt a classic iterative
debugging loop that includes simulation, error localization,
and correction. Finally, we integrate these error correction
mechanisms into a baseline LLM-based Verilog code gener-
ation workflow to improve the overall quality of RTL code
generation. The details of these mechanisms are presented in
the remainder of this section.
A. Error Correction Mechanisms
Description Compilation RTL Code
Simulation 
Error RepairCompilation      
Error RepairEndTestbench Basic RTL Generation And Repair Workflow
Simulation
Module II:  Rule-Guided Description RefinementLLM-Aided
Rule Checking
1. Initialization?
2. Reset Setting?
3. Enable Trigger?
…Description    
Refinement
1. Initialize to 0
2. Active -high / Syn 
3. Level trigger
…
…
Multimodal 
Data Extraction
Meta -Format
Module II:  Multimodal Data ConversionDescription
Erroneous RTL
Knowledge Base
Error Message
Keyword Matching 
Search
Vector Similarity 
SearchKeyword -Specific
Entry
Function -
Approximate RTLRTL Code
 LLMFail Test Case
Statement Error LocalizationDescription
LLM
Operation Error Localization And Repair
Module IV:   Simulation -Error Localization and Repair 
Erroneous RTL
Entry:
[keyword]: [“dual…”]
[Description]: The design…
[Example]: module……RTL Programming
Knowledge EntryCircuit Design
Knowledge Entry
Fig. 15: The workflow of RAG-based knowledge error mitigation
[Keyword]:  ["dual-edge trigger"]
[Description]:
This design is a dual-edge triggered flip-flop is triggered on both edges of the clock
that captures the input signal `d` on both the rising and falling edges of the clock
signal `clk`. ...
[Example]: 
module TopModule ( input clk,  input d,  output reg q );
  reg qp, qn;
  always @(posedge clk)  qp <= d;
  always @(negedge clk)  qn <= d;
  always @(*)  q <= clk ? qp : qn;
endmodule
Fig. 16: An example of circuit design knowledge entry
RAG-based Knowledge Error Mitigation: We leverage
RAG to supplement missing knowledge in both RTL pro-
gramming and circuit design, and enhance the RTL code
generation quality. To this end, we construct a domain-specific
knowledge base whose entries each consist of a keyword, a
concise description, and an illustrative example. Circuit design
knowledge is sourced from GitHub and StackOverflow and
organized into four categories—arithmetic, memory, control,
and miscellaneous modules—to enable efficient lookup (see
Fig. 16).
Our retrieval engine supports both semantic and
keyword-based queries. In semantic search, user design
descriptions are embedded and matched against the
knowledge-base vectors to retrieve the most relevant
entries. For clearly defined RTL terms, keywords extracted
from design specifications or compiler error messages
are employed to retrieve corresponding knowledge entry
via regular-expression matching. Ultimately, the retrievedinformation is deduplicated and integrated into the LLM’s
context to facilitate RTL generation, thereby mitigating
programming errors caused by insufficient knowledge.
Explicit Design Description ：
Implement a D latch using an always block.
module TopModule ( input d,  ena,     
                  output logic q );
// [Constraint] Registers/Logic initialization to 0
//[Constraint] Enable high-level triggerRule-Guided Design Description Refinement ：
Check the user's R TL design specification against the following rules and refine if necessary ：
[Rules]:
1. Check if register initialization is specified. If not specified, supplement the design
specification with default initialization to 0.
2. Check if the reset signal configuration is specified. If not specified, supplement the design
specification with a default high-level asynchronous trigger reset.
...                   
Ambigious Design Description ：
Implement a D latch using an always block.
module TopModule ( input d,  ena,
                  output logic q );
Corr ect R TL code ：
initial q = 0;
always @(*) begin
     if (ena) q<= d;
endErroneous R TL code ：
always @(posedge ena) begin
     if (ena) q <= d;
end
Fig. 17: An example of rule-based description refinement
Rule-based Description Refinement: To resolve ambigui-
ties in user-provided specifications, we introduce a rule-guided
refinement mechanism that leverages LLMs to enforce key
circuit design rules such as register initialization, reset/enable
signal integrity, and internal logical consistency. The LLM first
parses the initial specification, then checks it against these
rules. Whenever it detects ambiguities or contradictions, the
model automatically augments and clarifies the description
using contextual constraints, thereby eliminating gaps in the
design intent. Fig. 17 illustrates a typical case: by explicitly re-
fining the register initialization and enable signal requirements,
our mechanism ensures that the subsequent Verilog code
generation is both accurate and robust. Similarly, to mitigate
relatively common errors arising from LLMs’ unfamiliarity
with specialized RTL programming, based on the analysis in
Section III, we designed and integrated 10 programming rules
into the system prompt, such as ”Do not assign values to wire-
type variables within always blocks”.
(a) Kmap Example (Prob057_kmap2) 
Multimodal
DataMeta-Format RTL Code
(b) Waveform Example (Prob098_circuit7) 
Fig. 18: Examples of multimodal data conversion
Multimodal Data Conversion: To mitigate generation er-
rors arising from multimodal design inputs—such as Karnaugh
maps, waveform diagrams, and truth tables, we have devel-
oped tools that convert these heterogeneous formats into a
single, unified truth-table meta-representation as shown in Fig.
18. This meta-representation can then be directly translated
into Verilog via LLMs. Although the initially generated cir-
cuits may not be fully optimized, they can be automatically
streamlined using logic synthesis tools under the specified
design constraints. Importantly, when processing waveform

10
inputs for sequential circuits, our conversion tool captures the
clock triggering semantics to preserve correct timing behavior.
Likewise, for state transition diagrams, the intermediate rep-
resentation explicitly encodes both current and next states in
addition to the input and output signals.
Fig. 19: An example of two-stage debugging mechanism
Two-Stage Debugging: Localization and Correction:
To address the remaining errors such as missing details
of long context and miscellaneous errors, we employ a
simulation-based debugging loop. First, the generated Verilog
design is executed under a testbench to identify the earliest
failing test case. Next, an LLM analyzes the simulation trace
to pinpoint the Verilog statements most likely responsible for
the failure. The failing test case, candidate error locations,
and relevant design intent are then presented to the LLM,
which, using chain-of-thought reasoning, proposes corrective
modifications. This cycle of simulation, fault localization, and
automated repair iterates until all test cases pass, yielding a
robust final design. A representative debugging workflow is
illustrated in Fig. 19. In addition, specific debugging strategies
can also be integrated on top of the basic debugging loop. For
the missing details of long context, we may split the design
descriptions into smaller semantic chunks and conduct debug
of different chunks progressively.B. Application of the Error Correction Mechanisms
We integrate our error-correction mechanisms into a stan-
dard LLM-based design workflow that comprises code gen-
eration, compilation, and verification as shown in Fig. 20.
Before code generation, our RAG-based mechanism retrieves
relevant circuit designs and RTL programming knowledge
from a domain-specific knowledge base to enrich and disam-
biguate the user’s design descriptions. During the correction
phase, compilation-error messages similarly drive focused
lookups to fill any remaining knowledge gaps. Concurrently,
a rule-guided refinement module applies design rule checks
to the pre-generation descriptions, automatically refining or
repairing violations as needed. Whenever multimodal inputs
(e.g., K-maps, waveforms) are detected, a dedicated converter
replaces them with our unified truth-table meta-representation,
simplifying subsequent code generation. Finally, residual er-
rors such as missing long-context details are exposed via
simulation. Our debugger then iteratively localizes faults and
invokes LLM chain-of-thought reasoning to propose and apply
fixes until all test cases pass. This integrated framework
ensures robust and accurate RTL generation without manual
intervention.
V. E XPERIMENTS
A. Experiment Setups
LLM Setups: We evaluate Verilog code generation and
repair performance using GPT-based models. All experiments
are conducted with a fixed temperature of 0.1. For the multi-
round iterative RTL code repair method, we limit the maxi-
mum number of iterations to 10.
Benchmark: The evaluation is based on two widely
used RTL repair benchmarks: VerilogEval-v1-Human and
VerilogEval-v2 , both of which contain 156 representative real-
world designs spanning various difficulty levels.
Baselines: We compare against two categories of base-
lines: 1) Base models — single-pass RTL code generation
approaches using both general LLMs (e.g., GPT-4 Turbo [31]),
Code-Specific models (e.g. Qwen-Coder-32B-Instruction [32])
and RTL-specific models (e.g., ITERTL [33], CodeV [34]);
2)Agent-based systems — multi-agent frameworks de-
signed to enhance LLM-based RTL generation, including
LLM+RTLFixer [13], VeriAssistant [17], PromptV [25].
Methods Integration: In our experiment, we first refine all
design descriptions using rule-based method to facilitate initial
RTL generation or error correction. If the design still contains
errors, we perform a single-round correction via multimodal
data conversion if multimodal data is detected in the design
description; otherwise, we apply a single-round correction
using the RAG method. If errors persist, we then adopt the
two-stage debugging mechanism for iterative repair.
B. End-to-End Code Generation Comparison
Table II presents a comparison between our method and
existing baseline approaches. Our solution achieves 91.0% and
90.4% accuracy in pass@1 evaluations on VerilogEval-Human
andVerilogEval-v2 , respectively. Compared to strong vanilla

11
RTL Code 
GenerationCompilationLLM -based Design Workflow
RAG -based Knowledge Error MitigationPlease implement a D 
latch using Verilog.
….
…module TopModule (
input d; …
);…
endmoduleDescription
Input MessageVerificationCorrect RTL Code
Keyword 
Matching Search
Vector 
Similarity SearchKnowledge Base[keyword]: [“d latch…”]
[Description]: The 
design…
[Example]: module…Knowledge Entry
Meta Representation
Conversion ToolMultimodal Data
Kmap
Waveform
…Implement a D latch using an 
always block…
//[Constraint]: Ena high -level...
//[Design Rule]: Do not assign…1. Initialization?
2. Reset Setting?
…Multimodal Data Conversion
Rule Checking
 Refined DescriptionRule -based Description Refinement
Fail Test Case Erroneous RTL DescriptionTwo -Stage Debugging Mechanism
Statement Error Localization
Operation Error Localization 
and Repair
ENDA Set of Rules:
PassFail
Fig. 20: Overview of our framework
models such as GPT-4 Turbo, our method demonstrates signif-
icant improvements of 32.7% and29.5% on the two bench-
marks. In addition, our approach outperforms various agent-
based systems including LLM+RTLFixer ,VeriAssistant , and
PromptV . Meanwhile, our method achieves approximately a
30% improvement even on smaller-scale models like Qwen2.5-
Coder-32B-Instruction and GPT-3.5 Turbo.
TABLE II: Main results
Method LLM ModelVerilogEval-
Human
(Pass@1)VerilogEval-
v2 (Pass@1)
Generic LLMGPT-3.5 Turbo 42.9 41.0
GPT-4 Turbo 58.3 60.9
Code-Specified
LLMQwen-Coder-32B-
Instruction46.8 46.1
RTL-Specified
LLMRTLCoder [10] 41.6 36.5
ITERTL [33] 42.9 N/A
CodeV [34] 53.2 N/A
LLM +
RTLFixer [13]GPT-3.5 Turbo 46.4 44.9
GPT-4 Turbo 65.0 65.4
VeriAssist [17]Claude-3 41.6 N/A
GPT-3.5 34.4 N/A
GPT-4 50.5 N/A
PromptV [25] GPT-4 80.4 N/A
OursQwen-Coer-32B-
Instruction76.9 (+30.1) 78.1 (+32.0)
GPT-3.5 Turbo 73.1 (+30.2) 72.4 (+31.4)
GPT-4 Turbo 91.0 (+32.7) 90.4 (+29.5)
C. Error Correction Analysis
When utilizing foundational LLMs such as Qwen2.5-Coder-
32B-Instruction, GPT-3.5 Turbo and GPT-4 Turbo to generate
code on the VerilogEval-Human benchmark, we observe 243
erroneous RTL designs, respectively. Based on the integration
strategy mentioned above, we use each model to self-correct
their own erroneous designs. The number of successfully
repaired designs for each method is summarized in Fig 21.
Impact of RAG-based Knowledge Error Mitigation
(RAG.): Experimental results demonstrate that this methodsignificantly reduces knowledge-related errors in RTL designs
generated by GPT models. Specifically, it successfully cor-
rects 22 erroneous designs for GPT-3.5 Turbo and 25 for
GPT-4 Turbo. For the Qwen-Coder-32B-Instruction model, it
successfully corrected 16 erroneous designs. These findings
underscore that knowledge gaps remain a primary bottleneck
for large language models in RTL code generation.
RAG. RDR. MDC. TDM.051015202530Repair designs(#) of different methods16 16
12
322
913
325
613
7Qwen-Coder-32B GPT-3.5 Turbo GPT-4 Turbo
Fig. 21: Number of correctly repaired designs by different methods
Impact of Rule-based Description Refinement (RDR.):
We observe that some design descriptions are vague, such
as missing explanations of input signals. Additionally, we
identify several issues stemming from illegal assignments and
duplicate variable definitions. By clarifying descriptions and
enforcing RTL programming conventions, this method enables
GPT-3.5 Turbo and GPT-4 Turbo to correctly generate 9 and
6 additional designs, respectively. Using this approach, Qwen-
Coder-32B-Instruction corrected 16 design errors. This reveals
that compared to general-purpose LLMs, incorporating proper
coding rules brings more significant improvements to models
specifically pretrained on code.
Impact of Multimodal Data Conversion (MDC.): Our
experiments reveal that current LLMs encounter challenges in
semantically understanding multimodal hardware design data
(e.g., Karnaugh maps). However, our tool-based conversion ap-
proach mitigates these issues, reducing errors by 13 instances
in both GPT-3.5 Turbo and GPT-4 Turbo. For Qwen-Coder-
32B-Instruction, this approach reduced 12 such errors.

12
TABLE III: Correction outcomes for different categories of errors
Error
CategoryQwen-Coder-32B-Instruction GPT-3.5 Turbo GPT-4 Turbo
Total Corrected Accuracy(%) Total Corrected Accuracy(%) Total Corrected Accuracy(%)
IKSP. 13 8 61.5 25 14 56.0 8 7 87.5
IUCC. 45 23 51.1 43 18 41.9 23 18 78.3
ADD. 10 3 30.0 11 3 27.3 8 4 50.0
MMD. 27 18 66.7 28 19 67.9 23 17 73.9
MDLD. 10 1 10.0 11 5 45.5 10 4 40.0
Impact of Two-Stage Debugging Mechanism (TDM.):
We find that the debugging module performs better with
GPT-4 Turbo than with GPT-3.5 Turbo and Qwen-Coder-
32B-Instruction. This may be attributed to models like GPT-
3.5 Turbo tendency to produce larger and more error-prone
code blocks that require extensive revisions. Benefiting from
enhanced reasoning and programming capabilities, GPT-4
Turbo exhibits greater precision in identifying design flaws
and systematically correcting faulty RTL code.
Subsequently, we investigated the effectiveness of error
correction across different error categories, based on 243
erroneous RTL designs generated by the three different LLMs
above. The correction outcomes for each error category are
summarized in Table III.
It can be observed that for cases of Insufficient Knowledge
of Specialized RTL Programming, the repair results are no-
tably effective, with GPT-4 Turbo successfully fixing 87.5%
of design failures caused by this issue. Regarding Insufficient
Understanding of Circuit Concepts, all three models achieved
repairs for approximately 20 faulty designs by incorporating
additional external knowledge entries. When user descriptions
are ambiguous, the framework can still rectify a subset of
resulting errors, contributing to more robust code genera-
tion. For Misinterpretation of Multimodal Data, our method
consistently repaired around 18 error-inducing designs across
all three LLMs. In long-context scenarios, Qwen-Coder-32B-
Instruction repaired one design, while GPT-3.5 Turbo and
GPT-4 Turbo each fixed approximately five designs, indicating
that general-purpose LLMs represented by the GPT series
may demonstrate better long-context handling capabilities than
specifically trained smaller models.
We have found that complex design concepts combining
multiple knowledge components, such as the gshare branch
predictor with multiple design considerations, cannot guar-
antee correct RTL generation based solely on RAG. Simi-
larly, problems like Prob093 ece241 2014 q3—which require
implementing a given K-map logic function using a 4-to-1
multiplexer and minimal 2-to-1 multiplexers—demand deeper
multimodal data understanding to meet complex design con-
straints, and intermediate representation remains limited in
such cases. Additionally, LLMs’ lack of detailed long-context
understanding further limits the correctness of RTL generation.
Enhancing LLMs’ understanding and analytical capabilities
for such scenarios represents a key challenge and future
research direction.VI. D ISCUSSION
In this section, we will illustrate our error classification
criteria through several representative examples.
A. Classification criteria for the two main error categories
When given natural language design descriptions, we con-
sider an LLM to have correctly understood the requirements if
it generates accurate high-level pseudocode (natural language
format); We conclude the LLM failed to comprehend the
design requirements when its generated pseudocode exhibits
behavioral inconsistencies with the specifications. Conversely,
if the LLM violates RTL syntax rules when translating correct
pseudocode into low-level RTL code, we attribute this to
insufficient mastery of RTL programming knowledge.
Design Description  ( Prob100_fsm3comb ）
The following is the state transition table for a Moore state
machine with one input, one output, and four states. Use the
following state encoding: A=2'b00, B=2'b01, C=2'b10,
D=2'b1 1. Given the current state (state),  compute
the next_state and output (out) based on the state transition
table.
   State | Next state in=0, Next state in=1 | Output
   A     | A, B                             | 0
   B     | C, B                             | 0
   C     | A, D                             | 0
   D     | C, B                             | 1
LLM's Response ：  
module TopModule (
    input in,
    input [1:0] state,
    output reg [1:0] next_state,
    output out
);
    always @(*) begin
         case(state)
    2'b00: begin // State A
        next_state = in ? 2'b01 : 2'b00;
        out = 1'b0;
    end
    2'b01: begin // State B
        next_state = in ? 2'b01 : 2'b10;
        out = 1'b0;
    end
    2'b10: begin // State C
        next_state = in ? 2'b11 : 2'b00;
        out = 1'b0;
   end
   2'b11: begin // State D
        next_state = in ? 2'b01 : 2'b10;
        out = 1'b1;
  end
endcase
    end
endmoduleDesign Description  ( Prob071_always_casez ） 
Build a priority encoder for 8-bit inputs. Given an 8-bit vector ,
the output should report the first (least significant)  bit in the
vector that is 1. Report zero if the input vector has no bits that
are high.  For example, the input 8'b10010000 should output
3'd4, because bit[4] is first bit that is high.
LLM's Response ：  
module TopModule (
    input [7:0] in,
    output reg [2:0] pos
);
    always @(*) begin
        casez (in)
            8'b1??????? : pos = 3'd0;
             8'b01??????  : pos = 3'd1;
             8'b001?????  : pos = 3'd2;
            8'b0001????  : pos = 3'd3;
             8'b00001???  : pos = 3'd4;
            8'b000001??  : pos = 3'd5;
            8'b0000001?  : pos = 3'd6;
             8'b00000001  : pos = 3'd7;
           default : pos = 3'd0; // All zeros
       endcase
    end
endmodule
a. High-level code behavior matches the description. b. High-level code behavior does not match the description.
Fig. 22: Two distinct error type examples
As illustrated in Fig. 22, Sample a demonstrates correct
high-level code behavior (state design, transitions, and output
logic) that complies with the description, but contains RTL
implementation errors due to improper wire-type variable as-
signment within an always block. Sample b exhibits behavioral
inconsistencies with the description resulting from the LLM’s
incorrect understanding of the least significant bit concept -
representing two distinct error categories.
Notably, some larger designs may contain both error types
simultaneously (Fig. 14), where certain code segments comply

13
with behavioral descriptions but violate RTL conventions (e.g.,
done assignment), while others exhibit behavioral inconsisten-
cies due to knowledge limitations (e.g., data reg recording);
such designs are classified into both categories.
B. Subclassification Criteria for Misinterpretation of Design
Specifications
Design Description  ( Prob094_gatesv ）
You are given a four -bit input vector in[3:0]. We want to know
some relationships between each bit and its neighbour:
(1) out_both: Each bit of this output vector should indicate
whether both the corresponding input bit ...
(2) out_any: Each bit of this output vector should indicate
whether any of the corresponding input ...
(3) out_dif ferent: Each bit of this output vector should indicate
whether the corresponding input bit is dif ferent from its
neighbour to the left.  For example, out_dif ferent[2] should
indicate if in[2] is different from in[3] . For this part, treat the
vector as wrapping around, so in[3]'s neighbour to the left is
in[0]
a. Code generation errors in long-context scenariosLLM's Response ：  
module TopModule (
    input [3:0] in,
    output [2:0] out_both,
    output [3:1] out_any,
    output [3:0] out_different
);
    ...
// out_different
assign out_different[2] = in[2] ^ in[1];
    ...
endmoduleDesign Description  ( Prob094_gatesv ）
You are given a four -bit input vector in[3:0]. We want to know some
relationship between each bit and its neighbour:
out_dif ferent: Each bit of this output vector should indicate
whether the corresponding input bit is dif ferent from its neighbour to
the left. For example, out_dif ferent[2] should indicate if in[2] is
different from in[3].
LLM's Response ：  
module TopModule (
    input [3:0] in,
    output [2:0] out_both,
    output [3:1] out_any,
    output [3:0] out_different
);
    ...
// out_different
    assign out_different[1] = (in[1] !=
in[2]);    ...
endmodule
b.Correct code generation in decomposed-context scenarios  
Fig. 23: Long-context constraints hinder LLMs from correctly gen-
erating RTL code examples
Initially, regarding long-context scenarios, we observed that
directly prompting LLMs to generate RTL code based on
lengthy contexts often resulted in certain code segments failing
to accurately reflect high-level requirements. However, by
manually decomposing the long context—retaining only the
key descriptive text relevant to the erroneous segments while
omitting unnecessary details—the LLM regenerated RTL code
that correctly matched the specifications. As shown in Fig
23, after manual decomposition of the long context, the LLM
successfully generated the correct code. This demonstrates that
redundancy in long contexts is a limiting factor in LLMs’
ability to generate accurate RTL code.
Design Description  ( Prob050_kmap1 ）
Implement the circuit described by t he Karnaugh map below .
LLM's Response ：  
module TopModule (
    input a,
    input b,
    input c,
    output out
);
    assign out = (a & b);
endmoduleDesign Description  ( Prob050_kmap1 ）
Implement the circuit described below .                                   
a = 0, b = 0 c = 0, out = 0
a = 0, b = 0 c = 1, out = 1
a = 0, b = 1 c = 1, out = 1
a = 0, b = 1 c = 0, out = 1
a = 1, b = 0 c = 0, out = 1
a = 1, b = 0 c = 1, out = 1
a = 1, b = 1 c = 1, out = 1
a = 1, b = 1 c = 0, out = 1
LLM's Response ：  
module TopModule (
    input a,
    input b,
    input c,
    output out
);
    assign out = (a | b | c);
endmodule
a. Code generation errors in multimodal input scenarios b.Correct code generation by multimodal input conversion
Fig. 24: Multimodal input constraints hinder LLMs from correctly
generating RTL code examples
Similarly, in multimodal input scenarios, as illustrated in
Fig 24, when a KMAP table is directly provided as input, the
LLM struggles to correctly interpret the included information.
However, when the table’s content is converted into natural
language, the LLM successfully generates functionally accu-
rate RTL code. This reveals a bottleneck in the LLM’s ability
to comprehend tabular or other non-textual modalities.
 
 
 Design Description  ( Prob045_edgedetect2 ）
 For each bit in an 8-bit vector , detect when the input signal 
 changes from one clock cycle to the next (detect any edge). The 
 output bit should be set the cycle after a 0 to 1 transition occurs . 
 
 Design Description  ( Prob045_edgedetect2 ）
 For each bit in an 8-bit vector , detect when the input signal changes
 from one clock cycle to the next  (detect any edge). 
LLM's Response ：  
module TopModule (
    input clk,
    input [7:0] in,
    output reg [7:0] anyedge
);
    reg [7:0] prev_in; 
    always @(posedge clk) begin
       for (int i = 0; i < 8; i = i + 1)     
       begin
           if (in[i] & !prev_in[i]) begin
              anyedge[i] <= 1;
           end
           prev_in[i] <= in[i];
        end
    end
endmoduleLLM's Response ：  
module TopModule (
    input clk,
    input [7:0] in,
    output reg [7:0] anyedge
);
    reg [7:0] prev_in; 
    always @(posedge clk) begin
       for (int i = 0; i < 8; i = i + 1)     
       begin
           if (in[i] != prev_in[i]) begin
              anyedge[i] <= 1;
           end
           prev_in[i] <= in[i];
        end
    end
endmodule
a. Code generation errors in ambiguous description scenarios b. Code generation correct in clear description scenariosFig. 25: Ambiguous description constraints hinder LLMs from cor-
rectly generating RTL code examples
 
 
 Design Description  ( Prob033_ece241_2014_q1c ）
 Assume that you have two 8-bit 2's complement numbers, a[7:0] and 
 b[7:0]. These numbers are added to produce s[7:0]. Also compute 
 whether a  (signed) overflow  has occurred. 
 
 Design Description  ( Prob033_ece241_2014_q1c ）
 Assume that you have two 8-bit 2's complement numbers, a[7:0] and 
 b[7:0]. These numbers are added to produce s[7:0]. Also compute 
 whether a (signed) overflow has occurred.
 [How to determining whether  addition (signed) overflows]:
 When computing a + b, overflow occurs if the carry into the highest 
 bit dif fers from the carry out of the highest bit.
 
 Implementation:
 Sign-extend a and b to n+1 bits.
 Compute sum = {a[n-1], a} + {b[n-1], b}.
 Compare sum[n] and sum[n-1].
LLM's Response ：  
module TopModule (
    input [7:0] a,
    input [7:0] b,
    output [7:0] s,
    output reg overflow
);
    reg [8:0] temp_sum;
    always @* begin
        temp_sum = a + b;
        if (temp_sum[8] != temp_sum[7])        
             overflow = 1;
        else
            overflow = 0;
    end
    assign s = temp_sum[7:0];
endmoduleLLM's Response ：  
module TopModule (
    input [7:0] a,
    input [7:0] b,
    output reg [7:0] s,
    output reg overflow
);
    reg [8:0] a_ext, b_ext;
    reg [8:0] sum;
    assign a_ext = {a[7], a};
    assign b_ext = {b[7], b};
    always @* begin
       sum = a_ext + b_ext;
       overflow = sum[8] ^ sum[7];
    end
assign s = sum[7:0];
endmodule
a. Code generation errors  in cases where the model's internal
knowledge is insuf ficient  b. Code generation correct  with additional knowledge entries
provided
Fig. 26: Insufficient mastery of circuit knowledge constraints hinder
LLMs from correctly generating RTL code examples
As shown in Fig 25, ambiguous design descriptions required
manual refinement by removing contradictions and supple-
menting key information. This clarification significantly im-
proved RTL generation accuracy. Remaining errors primarily
stemmed from the LLM’s limited circuit knowledge, causing
RTL-description mismatches. Cases like 26 demonstrate that
providing essential circuit knowledge in the context helps
generate correct RTL, confirming this knowledge gap as a
major limitation.
Since we manually classify errors in LLM-generated RTL
code, and given the inherent randomness or potential halluci-
nations in LLM outputs, some inaccuracies and imperfections
in classification are inevitable. Nevertheless, it is undeniable
that the aforementioned factors represent key bottlenecks
limiting the LLM’s ability to generate correct RTL. Major
challenges and future directions for applying LLMs to hard-
ware design include determining effective methods to infuse
circuit knowledge into LLMs, improving their robustness in
generating code from ambiguous descriptions, and enhancing
their comprehension of multimodal information and long-
context understanding in hardware circuit design.
VII. C ONCLUSION
In this paper, we systematically analyzed and categorized
error causes in LLM-generated RTL code, and find that most

14
errors stem not from the reasoning capabilities of LLMs,
but rather from a lack of RTL programming knowledge,
insufficient understanding of circuit concepts, ambiguous de-
sign descriptions, or misinterpretation of complex multimodal
inputs. To address these, we propose a set of error correction
techniques including: 1) RAG-based knowledge error miti-
gation, 2) rule-based description refinement, 3) multimodal
data conversion, and 4) simulation-guided iterative debugging.
We integrate the proposed error correction mechanisms into a
representative LLM-based RTL code generation framework,
achieving 91.0% accuracy on the VerilogEval Benchmark.
This demonstrates that based on a series of strategies such
as knowledge retrieval, without requiring additional training,
can significantly mitigate errors in LLM-generated RTL code.
REFERENCES
[1] Humza Naveed, Asad Ullah Khan, Shi Qiu, Muhammad Saqib, Saeed
Anwar, Muhammad Usman, Naveed Akhtar, Nick Barnes, and Ajmal
Mian. A comprehensive overview of large language models. arXiv
preprint arXiv:2307.06435 , 2023.
[2] Mark Chen, Jerry Tworek, Heewoo Jun, Qiming Yuan, Henrique Ponde
De Oliveira Pinto, Jared Kaplan, Harri Edwards, Yuri Burda, Nicholas
Joseph, Greg Brockman, et al. Evaluating large language models trained
on code. arXiv preprint arXiv:2107.03374 , 2021.
[3] Erik Nijkamp, Bo Pang, Hiroaki Hayashi, Lifu Tu, Huan Wang, Yingbo
Zhou, Silvio Savarese, and Caiming Xiong. Codegen: An open large
language model for code with multi-turn program synthesis. arXiv
preprint arXiv:2203.13474 , 2022.
[4] Josh Achiam, Steven Adler, Sandhini Agarwal, Lama Ahmad, Ilge
Akkaya, Florencia Leoni Aleman, Diogo Almeida, Janko Altenschmidt,
Sam Altman, Shyamal Anadkat, et al. Gpt-4 technical report. arXiv
preprint arXiv:2303.08774 , 2023.
[5] Aixin Liu, Bei Feng, Bing Xue, Bingxuan Wang, Bochao Wu, Chengda
Lu, Chenggang Zhao, Chengqi Deng, Chenyu Zhang, Chong Ruan, et al.
Deepseek-v3 technical report. arXiv preprint arXiv:2412.19437 , 2024.
[6] Sumit K Ghosh. Hardware description languages: concepts and prin-
ciples . Wiley-IEEE Press, 1999.
[7] Zhenxing Fan. Automatically generating verilog rtl code with large
language models. Master’s thesis, New York University Tandon School
of Engineering, 2023.
[8] Shailja Thakur, Baleegh Ahmad, Hammond Pearce, Benjamin Tan,
Brendan Dolan-Gavitt, Ramesh Karri, and Siddharth Garg. Verigen:
A large language model for verilog code generation. ACM Transactions
on Design Automation of Electronic Systems , 29(3):1–31, 2024.
[9] Mingjie Liu, Nathaniel Pinckney, Brucek Khailany, and Haoxing Ren.
Verilogeval: Evaluating large language models for verilog code genera-
tion. In 2023 IEEE/ACM International Conference on Computer Aided
Design (ICCAD) , pages 1–8. IEEE, 2023.
[10] Shang Liu, Wenji Fang, Yao Lu, Qijun Zhang, Hongce Zhang, and
Zhiyao Xie. Rtlcoder: Outperforming gpt-3.5 in design rtl generation
with our open-source dataset and lightweight solution. In 2024 IEEE
International Workshop on LLM-Aided Design . IEEE, 2024.
[11] Zehua Pei, Hui-Ling Zhen, Mingxuan Yuan, Yu Huang, and Bei Yu.
Betterv: Controlled verilog generation with discriminative guidance.
arXiv preprint arXiv:2402.03375 , 2024.
[12] Yi Liu, Changran Xu, Yunhao Zhou, Zeju Li, and Qiang Xu. Deeprtl:
Bridging verilog understanding and generation with a unified represen-
tation model. arXiv preprint arXiv:2502.15832 , 2025.
[13] Yun-Da Tsai, Mingjie Liu, and Haoxing Ren. Rtlfixer: Automatically
fixing rtl syntax errors with large language models. arXiv preprint
arXiv:2311.16543 , 2023.
[14] Xufeng Yao, Haoyang Li, Tsz Ho Chan, Wenyi Xiao, Mingxuan Yuan,
Yu Huang, Lei Chen, and Bei Yu. Hdldebugger: Streamlining hdl de-
bugging with large language models. arXiv preprint arXiv:2403.11671 ,
2024.
[15] Shailja Thakur, Jason Blocklove, Hammond Pearce, Benjamin Tan, Sid-
dharth Garg, and Ramesh Karri. Autochip: Automating hdl generation
using llm feedback, 2024.
[16] Ke Xu, Jialin Sun, Yuchen Hu, Xinwei Fang, Weiwei Shan, Xi Wang,
and Zhe Jiang. Meic: Re-thinking rtl debug automation using llms. arXiv
preprint arXiv:2405.06840 , 2024.[17] Hanxian Huang, Zhenghan Lin, Zixuan Wang, Xin Chen, Ke Ding, and
Jishen Zhao. Towards llm-powered verilog rtl assistant: Self-verification
and self-correction. ArXiv , abs/2406.00115, 2024.
[18] Shang Liu, Wenji Fang, Yao Lu, Jing Wang, Qijun Zhang, Hongce
Zhang, and Zhiyao Xie. Rtlcoder: Fully open-source and efficient llm-
assisted rtl code generation technique. IEEE Transactions on Computer-
Aided Design of Integrated Circuits and Systems , 2024.
[19] Hammond Pearce, Benjamin Tan, and Ramesh Karri. Dave: Deriving au-
tomatically verilog from english. In Proceedings of the 2020 ACM/IEEE
Workshop on Machine Learning for CAD , pages 27–32, 2020.
[20] Alec Radford, Jeff Wu, Rewon Child, David Luan, Dario Amodei, and
Ilya Sutskever. Language models are unsupervised multitask learners.
2019.
[21] OpenAI. Gpt-3.5 turbo. https://platform.openai.com/docs/models/gpt-3.
5-turbo, 2024. [Accessed: 2024-06-20].
[22] Fan Cui, Chenyang Yin, Kexing Zhou, You lin Xiao, Guangyu Sun,
Qiang Xu, Qipeng Guo, Demin Song, Dahua Lin, Xingcheng Zhang,
and Yun Liang. Origen:enhancing rtl code generation with code-to-code
augmentation and self-reflection. ArXiv , abs/2407.16237, 2024.
[23] Mingzhe Gao, Jieru Zhao, Zhe Lin, Wenchao Ding, Xiaofeng Hou,
Yu Feng, Chao Li, and Minyi Guo. Autovcoder: A systematic framework
for automated verilog code generation using llms. arXiv preprint
arXiv:2407.18333 , 2024.
[24] Humza Sami, Pierre-Emmanuel Gaillardon, Valerio Tenace, et al. Aivril:
Ai-driven rtl generation with verification in-the-loop. arXiv preprint
arXiv:2409.11411 , 2024.
[25] Zhendong Mi, Renming Zheng, Haowen Zhong, Yue Sun, and Shaoyi
Huang. Promptv: Leveraging llm-powered multi-agent prompting for
high-quality verilog generation. arXiv preprint arXiv:2412.11014 , 2024.
[26] Chia-Tung Ho, Haoxing Ren, and Brucek Khailany. Verilogcoder:
Autonomous verilog coding agents with graph-based planning and
abstract syntax tree (ast)-based waveform tracing tool. In Proceedings of
the AAAI Conference on Artificial Intelligence , volume 39, pages 300–
307, 2025.
[27] Zhuorui Zhao, Ruidi Qiu, Ing-Chao Lin, Grace Li Zhang, Bing Li, and
Ulf Schlichtmann. Vrank: Enhancing verilog code generation from large
language models via self-consistency. arXiv preprint arXiv:2502.00028 ,
2025.
[28] Yujie Zhao, Hejia Zhang, Hanxian Huang, Zhongming Yu, and Jishen
Zhao. Mage: A multi-agent engine for automated rtl code generation.
arXiv preprint arXiv:2412.07822 , 2024.
[29] Manvi Jha, Jiaxin Wan, Huan Zhang, and Deming Chen. Preface-a
reinforcement learning framework for code verification via llm prompt
repair. In Proceedings of the Great Lakes Symposium on VLSI 2025 ,
pages 547–553, 2025.
[30] Ning Wang, Bingkun Yao, Jie Zhou, Yuchen Hu, Xi Wang, Nan
Guan, and Zhe Jiang. Veridebug: A unified llm for verilog debug-
ging via contrastive embedding and guided correction. arXiv preprint
arXiv:2504.19099 , 2025.
[31] OpenAI. Gpt-4 turbo. https://platform.openai.com/docs/models/
gpt-4-turbo, 2024. [Accessed: 2024-06-20].
[32] Binyuan Hui, Jian Yang, Zeyu Cui, Jiaxi Yang, Dayiheng Liu, Lei Zhang,
Tianyu Liu, Jiajun Zhang, Bowen Yu, Kai Dang, et al. Qwen2. 5-coder
technical report. arXiv preprint arXiv:2409.12186 , 2024.
[33] Peiyang Wu, Nan Guo, Xiao Xiao, Wenming Li, Xiaochun Ye, and
Dongrui Fan. Itertl: An iterative framework for fine-tuning llms for rtl
code generation. arXiv preprint arXiv:2407.12022 , 2024.
[34] Yang Zhao, Di Huang, Chongxiao Li, Pengwei Jin, Ziyuan Nan, Tianyun
Ma, Lei Qi, Yansong Pan, Zhenxing Zhang, Rui Zhang, Xishan Zhang,
Zidong Du, Qi Guo, Xingui Hu, and Yunji Chen. Codev: Empowering
llms for verilog generation through multi-level summarization. ArXiv ,
abs/2407.10424, 2024.