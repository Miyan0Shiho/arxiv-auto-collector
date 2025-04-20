# SymRTLO: Enhancing RTL Code Optimization with LLMs and Neuron-Inspired Symbolic Reasoning

**Authors**: Yiting Wang, Wanghao Ye, Ping Guo, Yexiao He, Ziyao Wang, Yexiao He, Bowei Tian, Shwai He, Guoheng Sun, Zheyu Shen, Sihan Chen, Ankur Srivastava, Qingfu Zhang, Gang Qu, Ang Li

**Published**: 2025-04-14 16:15:55

**PDF URL**: [http://arxiv.org/pdf/2504.10369v1](http://arxiv.org/pdf/2504.10369v1)

## Abstract
Optimizing Register Transfer Level (RTL) code is crucial for improving the
power, performance, and area (PPA) of digital circuits in the early stages of
synthesis. Manual rewriting, guided by synthesis feedback, can yield
high-quality results but is time-consuming and error-prone. Most existing
compiler-based approaches have difficulty handling complex design constraints.
Large Language Model (LLM)-based methods have emerged as a promising
alternative to address these challenges. However, LLM-based approaches often
face difficulties in ensuring alignment between the generated code and the
provided prompts. This paper presents SymRTLO, a novel neuron-symbolic RTL
optimization framework that seamlessly integrates LLM-based code rewriting with
symbolic reasoning techniques. Our method incorporates a retrieval-augmented
generation (RAG) system of optimization rules and Abstract Syntax Tree
(AST)-based templates, enabling LLM-based rewriting that maintains syntactic
correctness while minimizing undesired circuit behaviors. A symbolic module is
proposed for analyzing and optimizing finite state machine (FSM) logic,
allowing fine-grained state merging and partial specification handling beyond
the scope of pattern-based compilers. Furthermore, a fast verification
pipeline, combining formal equivalence checks with test-driven validation,
further reduces the complexity of verification. Experiments on the RTL-Rewriter
benchmark with Synopsys Design Compiler and Yosys show that SymRTLO improves
power, performance, and area (PPA) by up to 43.9%, 62.5%, and 51.1%,
respectively, compared to the state-of-the-art methods.

## Full Text


<!-- PDF content starts -->

SymRTLO: Enhancing RTL Code Optimization with LLMs and
Neuron-Inspired Symbolic Reasoning
Yiting Wang* 1Wanghao Ye* 1Ping Guo* 2Yexiao He* 1Ziyao Wang1Yexiao He1Bowei Tian1Shwai He1
Guoheng Sun1Zheyu Shen1Sihan Chen3Ankur Srivastava1Qingfu Zhang2Gang Qu1Ang Li1
Abstract
Optimizing Register Transfer Level (RTL) code
is crucial for improving the efficiency and per-
formance of digital circuits in the early stages
of synthesis. Manual rewriting, guided by syn-
thesis feedback, can yield high-quality results
but is time-consuming and error-prone. Most
existing compiler-based approaches have diffi-
culty handling complex design constraints. Large
Language Model (LLM)-based methods have
emerged as a promising alternative to address
these challenges. However, LLM-based ap-
proaches often face difficulties in ensuring align-
ment between the generated code and the pro-
vided prompts. This paper introduces Sym-
RTLO, a neuron-symbolic framework that inte-
grates LLMs with symbolic reasoning for the ef-
ficient and effective optimization of RTL code.
Our method incorporates a retrieval-augmented
system of optimization rules and Abstract Syn-
tax Tree (AST)-based templates, enabling LLM-
based rewriting that maintains syntactic correct-
ness while minimizing undesired circuit behav-
iors. A symbolic module is proposed for analyz-
ing and optimizing finite state machine (FSM)
logic, allowing fine-grained state merging and
partial specification handling beyond the scope of
pattern-based compilers. Furthermore, a fast veri-
fication pipeline, combining formal equivalence
checks with test-driven validation, further reduces
the complexity of verification. Experiments on
the RTL-Rewriter benchmark with Synopsys De-
sign Compiler and Yosys show that SymRTLO
improves power, performance, and area (PPA)
by up to 43.9%, 62.5%, and 51.1%, respectively,
compared to the state-of-the-art methods.
*Equal contribution1Department of Electrical Engineering,
University of Maryland, Maryland, United States2Department of
Computer Science, City University of Hong Kong, Hong Kong
3Viterbi School of Engineering, University of Southern California,
United States. Correspondence to: Ang Li <angliece@umd.edu >.
Under Review.1 Introduction
Register Transfer Level (RTL) optimization is a cornerstone
of modern circuit design flows, serving as the foundation
for achieving optimal Power, Performance, and Area (PPA).
As the earliest phase in the hardware design lifecycle, RTL
development provides engineers with the most significant
degree of flexibility to explore design patterns, make archi-
tectural trade-offs, and influence the overall design qual-
ity (Chu, 2006). Engineers use hardware description lan-
guages (HDLs) like Verilog to describe circuit functionality.
At this stage, decisions made have far-reaching implications,
as the quality of the RTL implementation directly impacts
subsequent stages, including synthesis, placement, and rout-
ing (Wang et al., 2009). A well-optimized RTL not only
ensures better design outcomes but also prevents suboptimal
designs from propagating through the flow, which can lead
to significant inefficiencies and costly iterations (Fang et al.,
2023; Zhou et al., 2002).
Despite its importance, RTL optimization remains a chal-
lenging and labor-intensive task. Engineers must iteratively
refine their designs through multiple rounds of synthesis and
layout feedback to ensure functionality and meet stringent
PPA targets. This process becomes increasingly cumber-
some as design complexity grows, with synthesis times
scaling disproportionately, often taking hours or even days
for a single iteration.(Fang et al., 2023) Consequently, de-
signers frequently face numerous synthesis cycles to evalu-
ate trade-offs and reach acceptable results. While modern
electronic design automation (EDA) tools provide compiler-
based methods to aid optimization, these approaches are
inherently limited. (Yao et al., 2024) They rely heavily on
predefined heuristics, making them ill-suited for adapting
to unconventional design patterns, complex constraints, or
dynamic optimization scenarios. As a result, the RTL opti-
mization process demands significant expertise and effort.
Recent advances in artificial intelligence, particularly the
advent of large language models (LLMs), have introduced
a new paradigm for automating and optimizing RTL code.
Leveraging the powerful generative capabilities of LLMs,
researchers have demonstrated their potential to rewrite
and optimize Verilog code automatically (Yao et al., 2024).
However, existing LLM-based approaches face critical chal-
lenges that limit their effectiveness. First, these models
1arXiv:2504.10369v1  [cs.AR]  14 Apr 2025

SymRTLO: Enhancing RTL Code Optimization with LLMs and Neuron-Inspired Symbolic Reasoning
often fail to align their generated outputs with specified
optimization objectives. The inherent limitations in log-
ical reasoning within LLMs can lead to deviations from
intended transformations, resulting in suboptimal or incor-
rect outputs. Second, despite their potential for automating
code generation, current methods still heavily rely on tra-
ditional synthesis feedback loops for optimization. This
reliance results in the inefficiencies of the synthesis process,
failing to address the core issue of long design cycles.
Our Proposed Framework. To address the critical chal-
lenges in RTL optimization, we introduce SymRTLO , the
first neuron-symbolic system that seamlessly integrates
LLMs with symbolic reasoning to optimize RTL code.
SymRTLO significantly reduces the reliance on repeated
synthesis tool invocations and enhances the alignment of
LLM-generated results with intended optimization rules.
Designing such a system introduces significant challenges,
which we address through the integration of carefully de-
signed modules. The first challenge lies in the generaliza-
tionof optimization rules. Traditionally, optimization pat-
terns are scattered across code samples, books, and informal
notes, making it difficult for compiler-based methods to for-
malize or apply them effectively. SymRTLO tackles this by
employing an LLM-based rule extraction system, combined
with a retrieval-augmented generation (RAG) mechanism
and a search engine. This ensures that optimization rules
are not only generalized but also efficiently retrievable from
a robust library built from diverse sources.
Another critical challenge is the alignment of LLM-
generated RTL code with the intended transformations, as
LLMs often struggle to produce outputs that strictly ad-
here to the specified optimization objective, leading to un-
reliable and unexplainable results. To ensure alignment,
SymRTLO employs Abstract Syntax Tree (AST)-based tem-
plates, which guide the LLM to generate code that satisfies
syntactic and semantic correctness. For complex control
flows or edge cases that exceed the capabilities of AST tem-
plates, the framework utilizes a symbolic generation module,
designed to handle such scenarios dynamically while main-
taining optimization quality.
In addition to alignment, conflicts often arise when differ-
ent design patterns are required to meet distinct PPA goals.
To address this, SymRTLO introduces a goal-oriented ap-
proach, where each optimization rule is explicitly tied to
its intended objective. This enables selective application
based on user-defined optimization goals, efficiently bal-
ancing these conflicts to deliver optimized designs without
disproportionately compromising PPA metrics.
Verification Traditional RTL verification requires extensive
manual test case creation. To address this, SymRTLO inte-
grates an automated test case generator, streamlining verifi-
cation while ensuring functional correctness.Our key contributions are summarized as follows:
1.LLM Symbolic Optimization: SymRTLO , the first
framework to combine LLM-based rewriting with sym-
bolic reasoning for RTL optimization.
2.Data Path and Control Path Optimization:
SymRTLO addresses critical challenges in both tra-
ditional EDA compilers and purely LLM-based ap-
proaches, particularly by aligning generated code with
FSM and data path algorithms, balancing conflicting
optimization rules, and improving explainability.
3.PPA Improvements: SymRTLO demonstrates its ef-
ficacy on industrial-scale and benchmark circuits, sur-
passing manual coding, classical compiler flows ( e.g.,
Synopsys DC Compiler), and state-of-the-art LLM-
based methods, achieving up to 43.9%, 62.5%, and
51.1% improvements in power, delay, and area.
2 Background and Motivation
LLMs have emerged as powerful tools for RTL design au-
tomation, with various approaches being developed since
2023. As summarized in Table 1, these approaches fall
into three primary categories: RTL code generation (Liu
et al., 2023a; 2024; Chang et al., 2023; Blocklove et al.,
2023), debugging (Tsai et al., 2024), and optimization (Yao
et al., 2024). This growing body of research demonstrates
the significant potential of LLMs to improve the efficiency
and effectiveness of EDA workflows. However, RTL code
optimization has always been a significant challenge in RTL
design, even for human experts, as it has the greatest impact
on the performance of downstream tasks.
Challenges in RTL Code Optimization with LLMs.
Aligning generated code with intended optimization goals
is a major challenge in LLM-based RTL optimization.
Due to inherent randomness, LLMs often produce incom-
plete, incorrect, or suboptimal results. For example, RTL-
Rewriter (Yao et al., 2024) employs retrieval-augmented
prompts and iterative synthesis-feedback loops to enhance
functional correctness but still struggles with fundamen-
tal misalignment between generated code and optimization
objectives. Additionally, the need for multiple synthesis
rounds significantly increases optimization time as design
complexity increases, limiting the scalability of current
LLM-based methods for large industrial-scale designs.
Current Approaches: Underutilization of Knowledge
and Manual Verification. Traditional RTL design opti-
mization relies on established patterns such as subexpres-
sion elimination (Pasko et al., 1999; Cocke, 1970), dead
code elimination (Knoop et al., 1994; Gupta et al., 1997),
strength reduction (Cooper et al., 2001), algebraic simpli-
fication (Buchberger & Loos, 1982; Carette, 2004), Mux
reduction (Chen & Cong, 2004; Wang et al., 2023), and
memory sharing (LaForest & Steffan, 2010; Ma et al., 2020).
While these techniques are effective, these optimizations
2

SymRTLO: Enhancing RTL Code Optimization with LLMs and Neuron-Inspired Symbolic Reasoning
Table 1. Comparative analysis of LLM-based methods for RTL
design. ✔indicates the presence of the feature, while ✘indicates
absence of the feature.
Category Method Verification Rule Output Conflict
Capability -Based Alignment Resolution
GenerationChipNeMo (Liu et al., 2023a) ✘ ✘ ✘ ✘
VeriGen (Thakur et al., 2024) ✘ ✘ ✘ ✘
VerilogEval (Liu et al., 2023b) ✘ ✘ ✘ ✘
RTLCoder (Liu et al., 2024) ✘ ✘ ✘ ✘
ChipChat (Blocklove et al., 2023) ✘ ✘ ✘ ✘
ChipGPT (Chang et al., 2023) ✘ ✘ ✘ ✘
Debug RTLFixer (Tsai et al., 2024) ✔ ✔ ✘ ✘
OptimizationRTLRewriter (Yao et al., 2024) ✔ ✔ ✘ ✘
SymRTLO (Ours) ✔ ✔ ✔ ✔
typically operate at the gate-level netlist, making the rela-
tionship between optimized output and original RTL code
less transparent. Additionally, optimization patterns from
design manuals and codebases remain underutilized due
to the lack of a centralized repository, forcing engineers
to rely on their expertise rather than automated tools. Fur-
thermore, verification requires the creation of test benches
and test cases manually, making the RTL design flow both
time-consuming and error-prone.
Table 2. Conflicts Between Design Goals & Optimization Patterns.
Design Pattern Goal Conflict Goal Conflict Design Pattern
Pipelining Low Timing Low Area Resource Sharing
Clock Gating Low Power Low Timing Retiming
Optimization Conflicts and Limited Compiler Capabili-
ties. Existing compiler-based methods face additional chal-
lenges, particularly in managing optimization goal con-
flicts and handling complex patterns. For instance, opti-
mizing for one metric, such as delay, often conflicts with
another, such as power consumption. Striking the right
balance between these competing objectives is crucial, espe-
cially as trade-offs between power and delay directly impact
overall system performance. As shown in Table 2, each
optimization method has its own specific goal, which often
clashes with others. Compiler-based methods also lack the
flexibility to adapt to such conflicts, limiting their effec-
tiveness in optimizing designs with diverse and competing
constraints.
Advancing LLM-based RTL Optimization with Neuro-
Symbolic Integration. Recent research has shown a grow-
ing trend toward combining symbolic reasoning with LLM
(Yang et al., 2023; Wan et al., 2024b; Diego Calanzone &
Vergari, 2024; Deepa Tilwani & Sheth, 2024; Wan et al.,
2024a), bringing new inspirations for more efficient and reli-
able LLM-based RTL code optimization with better prompt-
code alignment. These integrated approaches have seen
broad application across various fields, from automated the-
orem proof and knowledge representation to robotics and
medical diagnostics, demonstrating how the combination
of pattern recognition and generative capabilities of LLM
with the interpretability and logical rigor of symbolic sys-tems can significantly improve the alignment between LLM
output and the given prompt.
Motivation Experiments. To highlight the limitations of
current LLM-based RTL optimization, we conduct an exper-
iment using state-of-the-art commercial LLM, GPT-O1, to
optimize an 11-state FSM design. The goal was to minimize
and merge unnecessary states to enhance PPA metrics. GPT-
O1 receives a detailed state reduction algorithm to guide
the optimization process. We compare its results with an
optimized design that directly applied the state reduction al-
gorithm. As shown in Table 3, GPT-O1 struggles to align its
outputs with the algorithm, resulting in an under-optimized
FSM with minimal state reduction and improvement in PPA.
In contrast, algorithm-driven optimization achieves signif-
icantly better results, underscoring the challenges LLM-
based methods face in handling complex optimization tasks.
Table 3. Performance Comparison Across Different Approaches.
Approach # States Time (ns) Power (mW) Area (µm2)
Baseline 11 0.041 2.250 833.000
GPT-O1 10 0.041 2.280 993.480
Optimized 4 0.025 1.170 403.920
3 Methodology
SymRTLO takes a Verilog RTL module as input and opti-
mizes it for specific design goals, such as low power, high
performance, or reduced area. As illustrated in Figure 1,
the workflow begins by entering the RTL code and the user-
specified optimization goal ( 1) into the LLM Dispatcher
(2). This dispatcher analyzes the input circuit and deter-
mines the appropriate optimization path: either proceeding
solely with Data Flow Optimization (§3.2) or incorporat-
ingControl Flow Optimization (§3.3) as well, depending
on the characteristics of the design. For Data Flow Opti-
mization, a search engine with a retrieval-augmented mod-
ule extracts optimization rules and constructs AST-based
templates. For Control Flow Optimization, an LLM-driven
symbolic system generator performs FSM-specific transfor-
mations. Finally, the Final Optimization Module (§3.4)
integrates both paths and incorporates a verification system
to ensure the functional correctness of the optimized design.
3.1 LLM Dispatcher
The LLM Dispatcher (2) receives the input RTL code
and the specified optimization goal ( e.g., low power) ( 1)
before any optimization begins. It first summarizes the code
and generates potential optimization suggestions. These
suggestions are then passed to the Retrieval-Augmented
Generation (RAG) system to identify the relevant optimiza-
tion rules. Additionally, the Dispatcher evaluates the pres-
ence of a Finite State Machine (FSM) in the original code to
determine whether control flow optimization is necessary.
3

SymRTLO: Enhancing RTL Code Optimization with LLMs and Neuron-Inspired Symbolic Reasoning
IF: LLM 
recommends 
Control Flow 
Optimization 
Data Flow Optimization 
Books 
Code 
Manual 
Input 
Goal Input 
Code LLM 
Dispatcher 
Search 
Engine 
Rules 
extraction 
AST Templates 
Description 
Rule descriptions 
Optimization 
Library Template Writing 
Instructions 
Rule Search 
Results 
AST Optimization 
Template Module 
Equivalence 
Check 
AST-based 
Optimized 
Code 
IF: not  functional equivalent 
LLM Code 
Optimizer 
Optimized 
Code 
Equivalence 
Check 
IF: not  functional equivalent 
Final Output Code 1
23
64 RAG System 
IF: not  functional equivalent Symbolic 
Representation 
Symbolic 
Script 
Equivalence 
Check 
578 10 9
11 Final Optimization Module 
Optimized 
Symbolic 
Results 
Control Flow Optimization 
Figure 1. SymRTLO Architecture.
3.2 Data Flow Optimization
Data Flow represents the process by which information is
propagated, processed, and optimized within an RTL de-
sign. Effective data flow optimizations improve system
efficiency by simplifying computations, reducing redun-
dancy, and enhancing PPA metrics. Common techniques
include sub-expression elimination, constant folding, and
resource sharing. The proposed Data Flow Optimization
Module addresses three key challenges: (1) generalizing di-
verse optimization patterns into accurate, reusable rules; (2)
aligning LLM-generated optimizations with functional and
logical requirements; and (3) resolving conflicts between
optimization goals inherent in distinct design patterns.
Optimization Rule Search Engine Optimization knowl-
edge is often scattered across books, lecture notes, code-
bases, and design manuals, with no generalized repository
to serve as a unified knowledge base. Furthermore, opti-
mization patterns frequently conflict due to divergent goals,
for example, power reduction versus performance improve-
ment. To tackle these challenges, we developed a retrieval-
augmented generation (RAG) system equipped with an opti-
mization goal-based rule extraction module.
Optimization Library. The RAG system aggregates raw
RTL optimization data from sources such as lecture notes,
manuals, and example designs (Vahid, 2010; Taraate, 2022;
Schultz, 2023; Palnitkar, 2003) into a comprehensive knowl-
edge base ( 3). LLMs then summarize and structure these
data into an optimization library ( 4). Each rule is ab-
stracted to include its description, applicable optimization
goals ( e.g., area, power, or timing), and its category ( e.g.,
data flow, FSM, MUX, memory, or clock gating). A similar-
ity engine identifies overlaps with existing entries, prompt-
ing merges or exclusive labels to ensure the scalability of the
rule library. To align optimization goals with generated out-
puts, the rules specify detailed instructions for constructing
AST templates, enabling precise application of optimization
patterns. Rules with clearly defined requirements include
template-writing guidelines, while more abstract rules arestored as descriptive text and used directly as optimization
prompts. An example RAG is included in Appendix C.
Enforcing Rule Alignment and Resolving Conflicts To
improve the structure and alignment of optimization rules,
the LLM Dispatcher ( 1) provides both a summary of the
input RTL code and suggestions for potential optimizations.
These inputs are passed to the search engine along with user-
specified optimization goals, performing a similarity search
to identify the most relevant rules from the RAG system.
Given the potential for conflicts between optimization goals,
it is critical to prevent the inclusion of conflicting rules
while ensuring no critical optimizations are overlooked.
To achieve this balance, we employ the elbow method to
analyze the similarity scores between the query and the
candidate rules. This approach identifies a natural cutoff
point where adding more rules no longer yields significant
benefits. Let the similarity scores between the query and
candidate rules be ordered as:
s1≥s2≥ ··· ≥ sM, (1)
where sidenotes the similarity score for i-th rule, ordered
from highest to lowest, and Mis the total number of candi-
dates. The optimal cutoff index i∗is determined by maxi-
mizing the difference between consecutive similarity scores:
i∗= arg max
1≤i<M,(si−si+1). (2)
Rules with similarity scores above the threshold τelbow are
selected for application. The similarity between the query
embedding ( equery) and the a rule embedding ( erule) is com-
puted as shown below:
sim(equery,erule) =equery·erule
|equery||erule|≥τelbow. (3)
This method ensures that only the most relevant rules are
selected, striking an optimal balance between comprehen-
siveness and precision. The output of the search engine con-
tains two components: rules with detailed template-writing
instructions ( 7) and abstract rules described only by their
optimization properties ( 6).
4

SymRTLO: Enhancing RTL Code Optimization with LLMs and Neuron-Inspired Symbolic Reasoning
AST Template Building To ensure that LLM-generated
RTL code aligns with functional and logical optimization
goals, we enforce rules using AST-based symbolic systems,
which have been proven to be effective in hardware de-
bugging (Tsai et al., 2024). Compared to LLM-generated
symbolic systems, AST-based templates offer several advan-
tages: (1) parsing Verilog into an AST ensures accurate and
structured design representations; (2) limiting each template
to a single optimization goal maintains conciseness, facilitat-
ing correct generation and application by LLMs; and (3) the
modular approach allows selection of templates to balance
conflicting optimization patterns, enhancing flexibility.
For rules that include template-writing instructions, we
prompt the LLM to generate an AST-based template that
serves as a general optimization framework ( 8). Let A
denote the set of all AST nodes in the Verilog design, and
let the matching condition:
Φ :A → { true,false}, (4)
determine whether a node qualifies for optimization. The
process begins by identifying the Target Node Type ,
such as Always ,Instance ,Assign orModule
Instantiation . For each node of the specified type,
we apply Φto decide whether it requires rewriting. Once
the target nodes are identified, the Transformation Rule is
applied as follows:
τ:{a∈ A | Φ(a) =true} −→ A , (5)
where τreplaces the matched node with an optimized AST
subtree ( e.g., merging nested if-else statements, folding
constants, or simplifying expressions; code example see
Appendix B). To ensure functional correctness, the trans-
formed design undergoes an equivalence check using LLM-
generated testbenches. If the template passes verification, it
is stored in the RAG system as reusable content.
RTL 
Design Pyverilog AST Construction 
A[0:7] 
AND B[0:7] 
NOT 
MUX SHR C1[0] 
O[0:7] …Y[0:7] C2[0] 
Subexpression removal Templates 
Critical Path reduction Dead code elimination …
LLM Template 
selection Selected Reconstruct 
AST based 
Optimized code 
RAG system 
Feedback loop: Re-select on failed symbolic case 
Figure 2. SymRTLO AST template optimization workflow.
As shown in Figure 2, the RTL design is initially interpreted
as an AST representation. The RAG system provides the
LLM with multiple template options. Due to the varying op-
timization goals and scenarios, the system avoids relying on
a fixed sequence of templates. Instead, the LLM determines
which templates to apply and in what order, tailoring theoptimization process to the design’s specific requirements.
To further prevent conflicts between templates or failures in
the symbolic system, we introduce a feedback loop . This
loop allows the LLM to re-select templates and adjust its
strategy based on prior failures, ensuring robustness and
adaptability in the optimization process.
3.3 Control Flow Optimization
Control Flow, unlike Data Flow’s focus on how information
is processed and propagated, defines the execution paths
and sequencing of operations in RTL designs through finite-
state machines (FSMs) that capture states, transitions, and
outputs. These FSMs are tightly coupled with design con-
straints ( i.e.,partial specifications, clock gating, and reset
logic), making generic symbolic systems fragile or incom-
plete. Addressing these challenges requires deeper semantic
analysis beyond simple pattern matching or generic AST
templates. To enhance alignment between optimized code
and the FSM minimization algorithm, we propose a Control
Flow Optimization module utilizing an LLM-based sym-
bolic system. An FSM can be formally represented as:
M= (Q,Σ, δ, q 0, F), (6)
where Qis the finite set of states, Σis the input alphabet,
δ:Q×Σ→Qis the transition function, q0∈Qis the
initial state, and F⊆Qis the set of accepting states.
For a partially specified FSM Mp, the transition function is
extended to handle non-deterministic transitions:
δp:Q×Σ→2Q, (7)
where 2Qrepresents the power set of Q.
Classical minimization algorithms ( e.g., Hopcroft’s
(Hopcroft & Ullman, 1969) or Moore’s (Moore et al., 1956))
are effective for fully specified deterministic FSMs but are
limited by real-world complexities. Practical RTL designs
often integrate control logic with data path constraints, and
undefined states and transitions make FSM minimization an
NP-complete problem with a general complexity of O(2|Q|).
A single pre-built AST script cannot efficiently handle all
such partial specifications. Let:
ϕ:Q×D→B, (8)
represent the data path constraints, where Dis the data path
state space and Bis the boolean domain. Pure FSM-focused
AST-based optimization scripts can overlook these data path
side effects, failing to capture deeper control semantics.
Inspired by (Hu et al., 2023), we propose leveraging LLMs
to transform each circuit into a symbolic representation
focused solely on FSM components, i.e.,isolating states,
transitions, and relevant outputs, as illustrated in Figure 3.
Instead of relying on a one-size-fits-all script, we prompt the
LLM to dynamically generate a specialized minimization
5

SymRTLO: Enhancing RTL Code Optimization with LLMs and Neuron-Inspired Symbolic Reasoning
script tailored to the specific FSM structure and constraints.
The optimization is formulated as:
min
M′f(M′, C)subject to ϕ(M′, D) =true, (9)
where M′represents the optimized FSM, Crepresents the
design constraints, and ϕ(M′, D)ensures consistency with
the data path. Appendix A shows an example workflow.
RTL 
Design 2 1 3
5 6 4[2]
[4] [6]a
b
a,bLLM 
Transform LLM 
Symbolic 
System Transformed Symbolic 
Representation 
[1,3,5] Optimized Symbolic 
Representation 
aa
a
a a,bb b
bbbba
a
b
a
Figure 3. SymRTLO FSM optimization workflow.
3.4 Verification and Final Optimization
To address the challenges of manual verification and un-
reliable automated methods, we introduce an automated
verification module that integrates functionality testing and
formal equivalence checking. After AST-based optimiza-
tion and the LLM-assisted symbolic system generate ini-
tial results, the LLM combines extracted rules, template-
optimized code, and symbolic outputs to produce the final
optimized RTL design ( 11). Verification is essential to en-
sure correctness, as LLM-generated rewrites may introduce
unintended behavioral deviations. We employ a two-step
verification pipeline: (1) the LLM generates test benches to
validate basic functional correctness, acting as a rapid filter
to reject invalid rewrites early; and (2) for designs that pass
initial tests, we perform satisfiability (SAT)-based equiva-
lence checking to formally confirm functional equivalence
to the original design. Although SAT-based verification is
rigorous and ensures accuracy, it can be computationally
expensive, particularly for complex designs with arithmetic-
heavy circuits. By combining SAT checks with the quick,
randomized testing from the first step, we significantly re-
duce the number of candidates requiring formal verification.
This hybrid approach optimizes the trade-off between ver-
ification accuracy and computational cost, allowing us to
efficiently validate even complex optimizations.
4 Experiments
4.1 Experimental Setup
Baseline We compare SymRTLO with several state-of-the-
art LLM and open source RTL optimization frameworks.
The LLM baselines include GPT-4o, GPT-4(OpenAI et al.,
2023), GPT-3.5(Ye et al., 2023), and GPT-4o-mini. Ad-
ditionally, we include two specialized open-source LLM-
based tools: Verigen(Thakur et al., 2024) and RTL-Coder-
Deepseek (Liu et al., 2024). For a comprehensive evaluation,we analyze 11 RTLRewriter short-benchmark examples for
Wires and Cells and 10 complex FSM and algorithm ex-
amples from both short and long benchmarks for PPA. Al-
though the RTLRewriter environment is reproducible using
Yosys (Wolf et al., 2013) for wires and cells analysis, the ex-
act test case they use is not provided. Moreover, comparing
PPA results is even more challenging due to its reliance on
Yosys + ABC (Brayton & Mishchenko, 2010)with unknown
libraries. To demonstrate SymRTLO ’s capabilities, we sub-
ject it to a broader evaluation scope, selecting examples that
are diverse in size and functionality, while including cases
reported in RTLRewriter’s benchmark for direct compari-
son.
Implementations TheSymRTLO framework relies on
GPT-4o as its primary LLM for optimization strategy selec-
tion, symbolic system generation, and iterative HDL synthe-
sis, leveraging its robust inference and coding capabilities.
Other models like GPT-3.5 and GPT-O1 were excluded after
preliminary experiments revealing poor performance: GPT-
3.5 lacks sufficient coding capabilities, and GPT-O1 incurrs
high costs and long inference times, making our framework
less efficient. Pyverilog (Takamaeda-Yamazaki, 2015) is
used for AST extraction and code reconstruction.
To efficiently retrieve relevant transformation templates and
knowledge, we integrate OpenAI’s text embedding-3-small,
which excels in embedding-based retrieval tasks. For hard-
ware compilation and validation, we use a combination of
open-source and commercial tools. Yosys measures wires
and cells, while Synopsys DC Compiler 2019 (Synopsys),
paired with the Synopsys Standard Cell (SSC) library, per-
forms PPA analysis. GPT-4o generates test benches for
functional coverage, and Yosys + ABC serves as the logical
equivalence checker. For a fair comparison with standard
compiler workflows, we apply typical Synopsys DC Com-
piler optimizations, explicitly setting the area constraint to
zero and using medium mapping effort with incremental
mapping to reflect common practices.
Evaluation Metrics First, to evaluate generation qual-
ity and functional correctness, we use the pass@k metric
commonly employed in code generation tasks. This metric
captures the probability that at least one valid solution exists
within the top kgenerations:
pass@k =1
NNX
i=0(1−Ck
ni−ci
Ckni), (10)
where Nis the number of problems, niandcirepresent
the total and correct samples for the i-th problem, respec-
tively. Second, to test the performance of the synthesis
results, we use the best results of the 10 valid generations of
each model and our method. For smaller benchmarks, we
evaluate optimization results using Wires and Cells1
6

SymRTLO: Enhancing RTL Code Optimization with LLMs and Neuron-Inspired Symbolic Reasoning
Table 5. Comparison of Wire and Cell Counts. Gray highlighting denotes state-of-the-art results. GeoMean is the geometric mean of the
resource usage (wires or cells). Ratios are calculated by dividing the geometric mean of resource usage by the baseline’s usage.
BenchmarkYosys GPT-4-Turbo GPT-4o GPT-3.5-Turbo GPT-4o-mini RTLCoder-DS RTLrewriter SymRTLO
Wires Cells Wires Cells Wires Cells Wires Cells Wires Cells Wires Cells Wires Cells Wires Cells
adder subexpression 8 3 7 3 7 3 7 3 7 3 8 3 7 3 7 3
adder architecture 86 56 30 40 30 40 96 63 86 56 86 56 - - 14 16
multiplier subexpr 26 71 18 15 259 255 26 71 26 71 26 71 - - 18 15
constant folding raw 12 6 10 5 10 5 10 5 10 5 12 6 - - 8 5
subexpression elim 17 12 19 12 19 12 19 12 17 10 17 12 - - 14 8
alusubexpression 30 24 30 24 28 22 30 24 27 22 30 24 21 18 21 18
adder resource 13 3 6 3 9 4 6 3 7 3 13 3 - - 6 3
multiplier bitwidth 9 3 8 3 8 3 9 3 9 3 9 3 8 3 8 3
multiplier architect 4 2 14 36 4 2 16 20 18 36 4 2 - - 4 2
adder bitwidth 4 1 3 1 3 1 3 1 3 1 4 1 3 1 3 1
loop tiling raw 5 16 4 16 4 16 4 16 484 496 5 16 - - 3 16
GeoMean 15.49 8.96 13.45 9.81 16.35 9.97 16.40 11.49 16.31 11.49 15.49 8.96 - - 9.75 5.95
Ratio 1.00 1.00 0.87 1.10 1.06 1.11 1.06 1.28 1.11 1.31 1.15 0.91 0.69†10.77†10.63 0.67
Table 6. FSM Designs PPA Comparison. Gray highlights indicate state-of-the-art results. A ⇓marks improvement, while a ⇑denotes a
decline compare with the original design. Two comparison scenarios are shown: without compiler optimization (upper improvement) and
with compiler optimization (lower improvement). A - indicates that no code is available for analysis.
Model/Method example1 state example2 state example3 state example4 state example5 state
Power
(mW)Time
(ns)Area
(µm2)Power
(mW)Time
(ns)Area
(µm2)Power
(mW)Time
(ns)Area
(µm2)Power
(mW)Time
(ns)Area
(µm2)Power
(mW)Time
(ns)Area
(µm2)
Original 0.042 1.21 833.0 0.056 2.25 549.4 0.052 1.35 589.6 0.055 2.18 597.1 0.055 2.18 597.1
GPT-3.5 0.043 1.27 870.6 0.056 2.25 549.4 0.052 1.35 589.6 0.059 2.17 972.3 0.055 2.18 597.1
GPT4o-mini 0.055 1.08 1021.1 0.062 2.23 579.2 0.063 1.08 714.9 0.055 2.18 597.1 0.053 2.18 634.7
GPT-4-Turbo 0.053 2.97 993.5 0.067 2.28 737.6 0.065 1.22 810.3 0.055 2.18 273.5 0.029 2.25 366.8
GPT-4o 0.053 2.97 1002.5 0.056 2.25 549.4 0.052 1.35 589.6 0.055 2.18 273.5 0.055 2.18 597.1
RTLCoder-DS 0.042 1.21 833.0 0.056 2.25 549.4 0.052 1.35 589.6 0.055 2.18 597.1 0.055 2.18 597.1
SymRTLO 0.047 0.36 534.4 0.024 1.17 271.0 0.023 1.15 268.5 0.024 2.17 273.5 0.026 2.18 270.9
Improvement(%) ⇑11.90⇓70.25⇓35.85⇓57.14⇓48.00⇓50.67⇓55.77⇓14.81⇓54.46⇓56.36⇓0.46⇓54.1952 ⇓52.73 0.00 ⇓54.63
Original + Compiler Opt. 0.041 2.85 564.51 0.035 2.24 316.11 0.038 1.23 358.77 0.044 2.19 451.59 0.045 2.19 451.59
SymRTLO + Compiler Opt. 0.021 2.64 240.85 0.018 0.6 175.61 0.018 2.42 180.63 0.020 2.17 185.65 0.019 2.27 188.169
Improvement(%) ⇓48.78⇓7.37⇓57.33⇓48.57⇓73.21⇓44.45⇓52.63⇑96.74⇓49.65⇓54.55⇓0.91⇓58.89 ⇓57.78⇑3.65⇓58.33
Table 4. Pass Rate Results.
Method Pass@1 Pass@5 Pass@10
Ours 97.5 100.0 100.0
GPT-4o 45.9 60.0 72.7
GPT-4-Turbo 42.9 62.7 81.8
GPT-4o-mini 2.5 10.9 12.7
GPT-3.5-Turbo 28.6 42.7 54.5
RTL-Coder DeepSeek 8.8 18.2 27.3
Verigen-2B 0.0 0.0 0.0
Verigen-16B 0.0 0.0 0.0
metrics, which reflect low-level physical characteristics of
circuits. These metrics provide granular insights into routing
complexity (wires) and logical component count (cells),
offering a precise evaluation for isolated modules or blocks.
For larger designs, we focuse on PPA metrics to capture
high-level efficiency and real-world applicability. These
metrics offer a holistic view of resource usage and perfor-
mance for complex designs, where low-level metrics like
Wires and Cells become impractical.
4.2 Functional Correctness Analysis
To demonstrate that our method reduces synthesis time and
improves functional correctness, Table 4 presents the evalu-
ation results. SymRTLO achieves near-perfect pass rates in
first-attempt optimizations. This demonstrates exceptional
consistency in generating valid, optimized RTL code. This
significantly outperforms state-of-the-art language models,
particularly given the complexity of RTL optimization tasks
1†: Reported Results from (Yao et al., 2024).and the necessity of maintaining functional equivalence.
As a result, SymRTLO minimizes the number of synthesis
iterations required, reducing redundant computations and
streamlining the overall optimization process.
4.3 Circuit Optimization Performance
2.5 3.0 3.5
Time (ns)0.700.750.800.850.900.951.00Power (mW)Power vs Time
5000 6000
Area ( m²)
0.700.750.800.850.900.951.00Power (mW)Power vs Area
2.5 3.0 3.5
Time (ns)45004750500052505500575060006250Area ( m²)
Area vs Time
Original
SymRTLOGPT-4-Turbo
GPT-4oRTLCoder
GPT-4o-miniGPT-3.5-Turbo
Figure 4. The PPA overall improvement of benchmark cases.
To demonstrate SymRTLO ’s effectiveness in resolving cross-
rule conflicts and achieving optimization objectives, we con-
duct an experiment presented in Figure 4. Given the limited
benchmarks available for LLM-driven RTL design, we rely
on RTLRewriter’s benchmark, which primarily emphasizes
area optimization, leaving minimal room for improvements
in power and delay. To align with this limitation, we put area
optimization as our primary goal. Despite these constraints,
SymRTLO achieves substantial improvements, averaging
40.96% in power, 17.02% in delay, and 38.05% in area,
while maintaining a balanced optimization across all three
7

SymRTLO: Enhancing RTL Code Optimization with LLMs and Neuron-Inspired Symbolic Reasoning
Table 7. Algorithm Optimizations PPA Comparison. Gray highlights indicate state-of-the-art results. A ⇓marks improvement, while
a⇑denotes a decline compare with the original design. Two comparison scenarios are shown: without compiler optimization (upper
improvement) and with compiler optimization (lower improvement).
Model/Method sppm redundancy subexpression elim adder architecture vending fft
Power
(mW)Time
(ns)Area
(µm2)Power
(mW)Time
(ns)Area
(µm2)Power
(mW)Time
(ns)Area
(µm2)Power
(mW)Time
(ns)Area
(µm2)Power
(mW)Time
(ns)Area
(µm2)
Original 2.863 7.41 40102.6 5.265 11.09 10989.1 0.418 2.78 1023.5 7.61 227.86 176982.98 58.23 8.26 2255264.75
GPT-3.5-Turbo 2.863 7.41 40102.6 5.265 11.09 10989.1 0.418 2.78 1023.5 7.61 227.86 176982.98 58.23 8.26 2255264.75
GPT4o-mini 2.863 7.41 40102.6 5.265 11.09 10989.1 0.418 2.78 1023.5 7.61 227.86 176982.98 58.23 8.26 2255264.75
GPT-4-Turbo 2.863 7.41 40102.6 3.936 11.09 7783.03 0.392 2.74 1023.5 7.50 227.86 176982.98 58.23 8.26 2255264.75
GPT-4o 2.863 7.41 40102.6 5.296 11.09 8984.61 0.392 2.74 1023.5 7.61 227.86 176982.98 58.23 8.26 2255264.75
RTLCoder-DS 2.863 7.41 40102.6 5.265 11.09 10989.1 0.418 2.78 1023.5 7.61 227.86 176982.98 58.23 8.26 2255264.75
SymRTLO 1.762 7.29 29606.18 3.018 2.87 7358.8 0.328 1.97 762.6 6.97 227.86 164831.1 31.71 8.09 1726125.71
Improvement(%) ⇓38.46⇓1.62⇓26.17 ⇓42.68⇓74.12⇓33.04⇓21.53⇓29.14⇓25.49 ⇓8.41 ⇓0⇓6.87 ⇓45.54⇓2.06 ⇓23.46
Original + Compiler Opt. 1.46 7.95 22908.69 4.61 11.78 9484.15 0.17 2.29 541.920 11.46 7.90 240079.29 51.12 7.90 1857805.49
SymRTLO + Compiler Opt. 1.46 7.95 22908.69 3.53 11.78 6791.88 0.17 2.48 531.880 8.175 7.90 151593.86 26.32 8.98 1471378.46
Improvement(%) 0 0 0 ⇓23.4 0 ⇓28.39 0⇑8.30⇓1.86⇓28.66 0 ⇓36.86 ⇓48.51⇑13.67 ⇓20.8
metrics, highlighting its versatility and robustness.
Smaller benchmarks require only 1–2 optimization patterns
to optimize, making them ideal for testing alignment be-
tween LLM and the output to ensure our approach accu-
rately represents the optimization patterns. Table 5 shows
thatSymRTLO consistently outperforms baseline implemen-
tations across various test cases, With wire and cell ratios of
0.63 and 0.67 respectively, it surpasses the state-of-the-art
values of 0.69 and 0.77. While models like GPT-4 excel in
certain cases, they lack consistency across diverse optimiza-
tion tasks. Even RTL-focused models, despite their special-
ized training, exhibit limitations in optimization tasks.
In FSM PPA experiments, SymRTLO significantly outper-
forms existing approaches, particularly in relation to RTL-
Rewriter, the state-of-the-art solution, achieving an improve-
ment of up to 50.59%, 12.65%, 53.09% in power, time, and
area, respectively. As shown in Table 6, it effectively aligns
the FSM state reduction algorithm with optimized code,
minimizing all FSM states and achieving the best overall
PPA results. This demonstrates that the LLM-generated
symbolic system is both stable and aligned with intended
optimization goals.
To evaluate the effectiveness of generalized rules and AST
templates in balancing conflicting rules, we conduct algo-
rithm case PPA experiments involving complex Data Path
and Control Path scenarios. As shown in Table 7, SymRTLO
applies AST templates, optimized rules, and minimized
FSM states, achieving 30.34%, 21.37%, and 20.01% im-
provements in PPA over GPT-4o, our base model, on av-
erage. Since RTLRewriter’s generated code isn’t publicly
available, we cannot test or compare our results.
We test SymRTLO with Synopsys DC optimization work-
flows for both FSM and Algorithm cases, as shown in Table
6 and Table 7, demonstrating further balanced optimization
alongside compiler optimization processes, achieving over-
all improvements of 36.2% in power and 35.66% in area,
with only an 8.3% increase in time as a trade-off.4.4 Ablation Studies
Power Timing AreaPPA Improvement20%
8%5%37%
2%34%31%
1%32%41%
17%38%
1% 1%0%
Remove Template-Based Opt
Remove Symbolic ReasoningRemove Goal-Based Search
SymRTLOGPT-4o
Figure 5. Ablation results.
To assess the effectiveness of individual components in
SymRTLO , we conduct ablation studies by selectively re-
moving one module at a time, i.e.,either the AST-based
module, FSM symbolic system, or goal-based search engine,
while keeping the rest of the system intact. We then analyze
the impact of each removal across test cases, measuring
the resulting overall PPA improvements compared to the
baseline. Figure 5 summarizes these results, showing that
all three components contribute significantly to SymRTLO ’s
overall performance. Removing any one of them results
in substantial losses in optimization effectiveness, further
emphasizing the necessity of their integration. By contrast,
GPT-4o alone achieves minimal improvements, underscor-
ing the advantages of SymRTLO ’s tailored framework.
5 Conclusion
We present SymRTLO , a neuron-symbolic framework that
integrates LLM-based code rewriting and symbolic reason-
ing to optimize both data flow and control flow in RTL
designs. SymRTLO generalizes optimization rules, aligns
generated code with intended transformations, resolves con-
flicting optimization goals, and ensures reliable automated
verification. By combining retrieval-augmented guidance
with symbolic systems, SymRTLO automates complex struc-
tural rewrites while maintaining functional correctness. Ex-
tensive evaluations on industrial-scale designs demonstrate
significant PPA gains over state-of-the-art solutions.
8

SymRTLO: Enhancing RTL Code Optimization with LLMs and Neuron-Inspired Symbolic Reasoning
Impact Statement
SymRTLO advances EDA by integrating Large Language
Models with symbolic reasoning to automate and opti-
mize RTL code more efficiently than existing methods.
This innovation not only significantly enhances PPA met-
rics—enabling the development of more energy-efficient
and compact electronic devices—but also accelerates chip
design, prototyping, and the creation of specialized chips.
By reducing design time and reducing production costs,
SymRTLO facilitates faster industry growth and empow-
ers scientific exploration through more accessible and cost-
effective hardware development. Additionally, by lowering
the expertise barrier and reducing the educational costs asso-
ciated with chip design, SymRTLO democratizes hardware
development, allowing a broader range of engineers and
even non-experts to design their own chips. This democra-
tization fosters greater innovation, minimizes human error,
and contributes to economic growth and environmental sus-
tainability, ultimately transforming the landscape of chip
production and scientific research.
References
Blocklove, J., Garg, S., Karri, R., and Pearce, H. Chip-chat:
Challenges and opportunities in conversational hardware
design. In 5th ACM/IEEE Workshop on Machine Learn-
ing for CAD, MLCAD . IEEE, 2023.
Brayton, R. and Mishchenko, A. ABC: An academic
industrial-strength verification tool. In Computer Aided
Verification: 22nd International Conference, CAV 2010,
Edinburgh, UK, July 15–19, 2010. Proceedings 22 , pp.
24–40. Springer, 2010.
Buchberger, B. and Loos, R. Algebraic simplification. In
Computer algebra: symbolic and algebraic computation ,
pp. 11–43. Springer, 1982.
Carette, J. Understanding expression simplification. In Pro-
ceedings of the 2004 international symposium on Sym-
bolic and algebraic computation , pp. 72–79, 2004.
Chang, K., Wang, Y ., Ren, H., Wang, M., Liang, S., Han, Y .,
Li, H., and Li, X. Chipgpt: How far are we from natural
language hardware design. CoRR , abs/2305.14019, 2023.
Chen, D. and Cong, J. Register binding and port assignment
for multiplexer optimization. In Proceedings of the 2004
Asia and South Pacific Design Automation Conference ,
ASP-DAC ’04, pp. 68–73. IEEE Press, 2004. ISBN
0780381750.
Chu, P. P. RTL hardware design using VHDL: coding for
efficiency, portability, and scalability . John Wiley & Sons,
2006.Cocke, J. Global common subexpression elimination. In
Proceedings of a symposium on Compiler Optimization ,
pp. 20–24. ACM, 1970. doi: 10.1145/800028.808480.
Cooper, K. D., Simpson, L. T., and Vick, C. A. Operator
strength reduction. ACM Transactions on Programming
Languages and Systems (TOPLAS) , 23(5):603–625, 2001.
Deepa Tilwani, R. V . and Sheth, A. P. Neurosymbolic ai
approach to attribution in large language models. arXiv ,
2410.03726, September 2024. doi: 10.48550/arXiv.2410.
03726. Paper under review.
Diego Calanzone, S. T. and Vergari, A. Logically consistent
language models via neuro-symbolic integration. arXiv ,
2409.13724, September 2024. doi: 10.48550/arXiv.2409.
13724.
Fang, W., Lu, Y ., Liu, S., Zhang, Q., Xu, C., Wills, L. W.,
Zhang, H., and Xie, Z. Masterrtl: A pre-synthesis ppa esti-
mation framework for any rtl design. In 2023 IEEE/ACM
International Conference on Computer Aided Design (IC-
CAD) , pp. 1–9, 2023. doi: 10.1109/ICCAD57390.2023.
10323951.
Gupta, R., Benson, D., and Fang, J. Z. Path profile guided
partial dead code elimination using predication. In Pro-
ceedings 1997 International Conference on Parallel Ar-
chitectures and Compilation Techniques , pp. 102–113.
IEEE, 1997.
Hopcroft, J. E. and Ullman, J. D. Formal languages and
their relation to automata . Addison-Wesley Longman
Publishing Co., Inc., 1969.
Hu, Y ., Yang, H., Lin, Z., and Zhang, M. Code prompting:
a neural symbolic method for complex reasoning in large
language models, 2023. URL https://arxiv.org/
abs/2305.18507 .
Knoop, J., R ¨uthing, O., and Steffen, B. Partial dead code
elimination. ACM Sigplan Notices , 29(6):147–158, 1994.
LaForest, C. E. and Steffan, J. G. Efficient multi-ported
memories for fpgas. In Proceedings of the 18th An-
nual ACM/SIGDA International Symposium on Field Pro-
grammable Gate Arrays (FPGA ’10) , pp. 41–50, Mon-
terey, CA, USA, 2010. ACM. doi: 10.1145/1723112.
1723122.
Langley, P. Crafting papers on machine learning. In Langley,
P. (ed.), Proceedings of the 17th International Conference
on Machine Learning (ICML 2000) , pp. 1207–1216, Stan-
ford, CA, 2000. Morgan Kaufmann.
Liu, M., Ene, T., Kirby, R., Cheng, C., Pinckney, N. R.,
Liang, R., Alben, J., Anand, H., Banerjee, S., Bayrak-
taroglu, I., Bhaskaran, B., Catanzaro, B., Chaudhuri, A.,
9

SymRTLO: Enhancing RTL Code Optimization with LLMs and Neuron-Inspired Symbolic Reasoning
Clay, S., Dally, B., Dang, L., Deshpande, P., Dhodhi, S.,
Halepete, S., Hill, E., Hu, J., Jain, S., Khailany, B., Kunal,
K., Li, X., Liu, H., Oberman, S. F., Omar, S., Pratty, S.,
Raiman, J., Sarkar, A., Shao, Z., Sun, H., Suthar, P. P.,
Tej, V ., Xu, K., and Ren, H. Chipnemo: Domain-adapted
llms for chip design. CoRR , abs/2311.00176, 2023a.
Liu, M., Pinckney, N. R., Khailany, B., and Ren, H. Invited
paper: Verilogeval: Evaluating large language models
for verilog code generation. In IEEE/ACM International
Conference on Computer Aided Design, ICCAD . IEEE,
2023b.
Liu, S., Fang, W., Lu, Y ., Wang, J., Zhang, Q., Zhang, H.,
and Xie, Z. Rtlcoder: Fully open-source and efficient llm-
assisted rtl code generation technique. IEEE Transactions
on Computer-Aided Design of Integrated Circuits and
Systems , 2024.
Ma, J., Zuo, G., Loughlin, K., Cheng, X., Liu, Y ., Eneyew,
A. M., Qi, Z., and Kasikci, B. A hypervisor for shared-
memory fpga platforms. In Proceedings of the Twenty-
Fifth International Conference on Architectural Support
for Programming Languages and Operating Systems , pp.
827–844, 2020.
Moore, E. F. et al. Gedanken-experiments on sequential
machines. Automata studies , 34:129–153, 1956.
OpenAI, Achiam, J., Adler, S., Agarwal, S., Ahmad, L.,
Akkaya, I., Aleman, F. L., Almeida, D., Altenschmidt,
J., Altman, S., Anadkat, S., Avila, R., Babuschkin, I.,
Balaji, S., Balcom, V ., Baltescu, P., Bao, H., Bavarian, M.,
Belgum, J., Bello, I., Berdine, J., Bernadett-Shapiro, G.,
Berner, C., Bogdonoff, L., Boiko, O., Boyd, M., Brakman,
A.-L., Brockman, G., Brooks, T., Brundage, M., Button,
K., Cai, T., Campbell, R., Cann, A., Carey, B., Carlson,
C., Carmichael, R., Chan, B., Chang, C., Chantzis, F.,
Chen, D., Chen, S., Chen, R., Chen, J., Chen, M., Chess,
B., Cho, C., Chu, C., Chung, H. W., Cummings, D.,
Currier, J., Dai, Y ., and many others. Gpt-4 technical
report. arXiv , 2303.08774, March 2023. doi: 10.48550/
arXiv.2303.08774. We report the development of GPT-4,
a large-scale, multimodal model which exhibits human-
level performance on various professional and academic
benchmarks.
Palnitkar, S. Verilog HDL: a guide to digital design and
synthesis , volume 1. Prentice Hall Professional, 2003.
Pasko, R., Schaumont, P., Derudder, V ., Vernalde, S., and
Durackov ´a, D. A new algorithm for elimination of com-
mon subexpressions. Computer-Aided Design of Inte-
grated Circuits and Systems, IEEE Transactions on , 18:
58 – 68, 02 1999. doi: 10.1109/43.739059.Schultz, J. Optimizing the rtl design flow with real-
time ppa analysis. Synopsys Silicon to Systems
Blog , N/A, March 2023. doi: N/A. Blog Post.
URL: https://www.synopsys.com/blogs/silicon-to-
systems/optimizing-rtl-design-flow-real-time-ppa-
analysis/ (Accessed January 26, 2025).
Synopsys. Dc ultra for synthesis and
test. https://www.synopsys.com/
implementation-and-signoff/
rtl-synthesis-test/dc-ultra.html .
Takamaeda-Yamazaki, S. Pyverilog: A python-based
hardware design processing toolkit for verilog hdl.
InApplied Reconfigurable Computing , volume 9040
ofLecture Notes in Computer Science , pp. 451–460.
Springer International Publishing, Apr 2015. doi: 10.
1007/978-3-319-16214-0 42. URL http://dx.doi.
org/10.1007/978-3-319-16214-0_42 .
Taraate, V . Digital logic design using verilog . Springer,
2022.
Thakur, S., Ahmad, B., Pearce, H., Tan, B., Dolan-Gavitt,
B., Karri, R., and Garg, S. Verigen: A large language
model for verilog code generation. ACM Trans. Design
Autom. Electr. Syst. , 2024.
Tsai, Y ., Liu, M., and Ren, H. Rtlfixer: Automatically
fixing RTL syntax errors with large language model. In
Proceedings of the 61st ACM/IEEE Design Automation
Conference, DAC . ACM, 2024.
Vahid, F. Digital design with rtl design, vhdl, and verilog
(2nd edition). John Wiley & Sons Inc , 2, January 2010.
doi: N/A. ISBN-13: 978-0470531082, 575 pages.
Wan, Z., Liu, C.-f., Yang, H., Raj, R., Li, C., You,
H., Fu, Y ., Wan, C., Li, S., Kim, Y ., Samajdar, A.,
Lin, Y ., Ibrahim, M., Rabaey, J. M., Krishna, T.,
and Raychowdhury, A. Cross-layer design for neuro-
symbolic ai: From workload characterization to hard-
ware acceleration. arXiv , 2409.13153, September
2024a. doi: 10.48550/arXiv.2409.13153. Available at
https://arxiv.org/abs/2409.13153.
Wan, Z., Liu, C.-K., Yang, H., Raj, R., Li, C., You, H.,
Fu, Y ., Wan, C., Li, S., Kim, Y ., Samajdar, A., Lin,
Y . C., Ibrahim, M., Rabaey, J. M., Krishna, T., and
Raychowdhury, A. Towards efficient neuro-symbolic
ai: From workload characterization to hardware archi-
tecture. IEEE Transactions on Circuits and Systems for
Artificial Intelligence , 2409.13153, September 2024b. doi:
10.48550/arXiv.2409.13153.
Wang, L.-T., Chang, Y .-W., and Cheng, K.-T. T. Electronic
design automation: synthesis, verification, and test . Mor-
gan Kaufmann, 2009.
10

SymRTLO: Enhancing RTL Code Optimization with LLMs and Neuron-Inspired Symbolic Reasoning
Wang, Z., You, H., Wang, J., Liu, M., Su, Y ., and Zhang,
Y . Optimization of multiplexer combination in rtl logic
synthesis. In Proceedings of the 2023 International Sym-
posium of Electronics Design Automation (ISEDA) , pp.
N/A. IEEE, May 2023. doi: 10.1109/ISEDA59274.2023.
10218464.
Wolf, C., Glaser, J., and Kepler, J. Yosys-a free
verilog synthesis suite. In Proceedings of the Fo-
rum on Specification and Design Languages (FDL) ,
2013. URL https://api.semanticscholar.
org/CorpusID:202611483 . Yosys is the first open-
source Verilog synthesis suite supporting a wide range of
synthesizable Verilog features.
Yang, S., Li, X., Cui, L., Bing, L., and Lam, W. Neuro-
symbolic integration brings causal and reliable reasoning
proofs. arXiv , 2311.09802, November 2023. doi: 10.
48550/arXiv.2311.09802.
Yao, X., Wang, Y ., Li, X., Lian, Y ., Ran, C., Chen, L., Yuan,
M., Xu, H., and Yu, B. Rtlrewriter: Methodologies for
large models aided rtl code optimization, 09 2024.
Ye, J., Chen, X., Xu, N., Zu, C., Shao, Z., Liu, S., Cui, Y .,
Zhou, Z., Gong, C., Shen, Y ., Zhou, J., Chen, S., Gui, T.,
Zhang, Q., and Huang, X. A comprehensive capability
analysis of gpt-3 and gpt-3.5 series models, 2023. URL
https://arxiv.org/abs/2303.10420 .
Zhou, H., Lin, Z., and Cao, W. Research on vhdl rtl synthe-
sis system. In Proceedings First IEEE International Work-
shop on Electronic Design, Test and Applications ’2002 ,
pp. 99–103, 2002. doi: 10.1109/DELTA.2002.994596.
11

SymRTLO: Enhancing RTL Code Optimization with LLMs and Neuron-Inspired Symbolic Reasoning
A FSM Symbolic System
The following section demonstrates an example FSM in verilog. First the verilog is transformed to Symbolic Representation,
then the Symbolic system applied minimization algorithm to optimize the FSM.
1module example(
2 input wire clk,
3 input wire reset,
4 input wire [1:0] input_signal,
5 output reg output_signal);
6 parameter S0 = 3’b000,S1 = 3’b001,S2 = 3’b010,S3 = 3’b011,S4 = 3’b100,S5 = 3’b101;
7 reg [2:0] current_state, next_state;
8 always @(current_state) begin
9 output_signal = 0;
10 case (current_state)
11 S0: output_signal = 1;
12 S2: output_signal = 1;
13 S4: output_signal = 1;
14 default : output_signal = 0;
15 endcase
16 end
17 always @(posedge clk or posedge reset) begin
18 if(reset) begin
19 current_state <= S0; // Reset to state S0
20 end else begin
21 current_state <= next_state;
22 end
23 end
24 always @(*)begin
25 next_state = current_state;
26 case (current_state)
27 S0: case (input_signal)
28 2’b00: next_state = S0;
29 2’b01: next_state = S1;
30 2’b10: next_state = S2;
31 2’b11: next_state = S3;
32 endcase
33 S1: case (input_signal)
34 2’b00: next_state = S0;
35 2’b01: next_state = S3;
36 2’b11: next_state = S5;
37 endcase
38 S2: case (input_signal)
39 2’b00: next_state = S1;
40 2’b01: next_state = S3;
41 2’b10: next_state = S2;
42 2’b11: next_state = S4;
43 endcase
44 S3: case (input_signal)
45 2’b00: next_state = S1;
46 2’b01: next_state = S0;
47 2’b10: next_state = S4;
48 2’b11: next_state = S5;
49 endcase
50 S4: case (input_signal)
51 2’b00: next_state = S0;
52 2’b01: next_state = S1;
53 2’b10: next_state = S2;
54 2’b11: next_state = S5;
55 endcase
56 S5: case (input_signal)
57 2’b00: next_state = S1;
58 2’b01: next_state = S4;
59 2’b10: next_state = S0;
60 endcase
61 endcase
62 end
63endmodule
Listing 1: Example Test Case: example1 state.
12

SymRTLO: Enhancing RTL Code Optimization with LLMs and Neuron-Inspired Symbolic Reasoning
1 ’states’: [’S0’, ’S1’, ’S2’, ’S3’, ’S4’, ’S5’],
2 ’transitions’: {
3 ’S0’: {
4 ’input_signal=00’: {’next_state’: ’S0’},
5 ’input_signal=01’: {’next_state’: ’S1’},
6 ’input_signal=10’: {’next_state’: ’S2’},
7 ’input_signal=11’: {’next_state’: ’S3’}
8 },
9 ’S1’: {
10 ’input_signal=00’: {’next_state’: ’S0’},
11 ’input_signal=01’: {’next_state’: ’S3’},
12 ’input_signal=11’: {’next_state’: ’S5’}
13 },
14 ’S2’: {
15 ’input_signal=00’: {’next_state’: ’S1’},
16 ’input_signal=01’: {’next_state’: ’S3’},
17 ’input_signal=10’: {’next_state’: ’S2’},
18 ’input_signal=11’: {’next_state’: ’S4’}
19 },
20 ’S3’: {
21 ’input_signal=00’: {’next_state’: ’S1’},
22 ’input_signal=01’: {’next_state’: ’S0’},
23 ’input_signal=10’: {’next_state’: ’S4’},
24 ’input_signal=11’: {’next_state’: ’S5’}
25 },
26 ’S4’: {
27 ’input_signal=00’: {’next_state’: ’S0’},
28 ’input_signal=01’: {’next_state’: ’S1’},
29 ’input_signal=10’: {’next_state’: ’S2’},
30 ’input_signal=11’: {’next_state’: ’S5’}
31 },
32 ’S5’: {
33 ’input_signal=00’: {’next_state’: ’S1’},
34 ’input_signal=01’: {’next_state’: ’S4’},
35 ’input_signal=10’: {’next_state’: ’S0’}
36 }
37 },
38 ’outputs’: {’S0’: {’output_signal’: 1},
39 ’S1’: {’output_signal’: 0},
40 ’S2’: {’output_signal’: 1},
41 ’S3’: {’output_signal’: 0},
42 ’S4’: {’output_signal’: 1},
43 ’S5’: {’output_signal’: 0} }
Listing 2: LLM-generated symbolic representation for example1 state.
1State: S2, Output: 1
2 input_signal=00 -> S1
3 input_signal=01 -> S3_S5
4 input_signal=10 -> S2
5 input_signal=11 -> S0_S4
6State: S0_S4, Output: 1
7 input_signal=00 -> S0_S4
8 input_signal=01 -> S1
9 input_signal=10 -> S2
10 input_signal=11 -> S3_S5
11State: S1, Output: 0
12 input_signal=00 -> S0_S4
13 input_signal=01 -> S3_S5
14 input_signal=10 -> S1
15 input_signal=11 -> S3_S5
16State: S3_S5, Output: 0
17 input_signal=00 -> S1
18 input_signal=01 -> S0_S4
19 input_signal=10 -> S0_S4
20 input_signal=11 -> S3_S5
Listing 3: Reduced states of example1 state.
13

SymRTLO: Enhancing RTL Code Optimization with LLMs and Neuron-Inspired Symbolic Reasoning
B AST Template
The following section demonstrate how applying AST templates transforms the code and apply optimization patterns.
1module example_raw
2#( parameter BW = 8)
3(
4 input [BW-1:0] a,
5 input [BW-1:0] b,
6 input [BW-1:0] c,
7 input [BW-1:0] d,
8 output [BW-1:0] s1
9);
10 assign s2 = a *b;
11 assign s3 = a % b +d;
12 assign s4 = c + d + b *a;
13 assign s5 = a - b;
14 assign s6 = (b + 1) *a + d + c -b;
15 assign s1 = a + 23;
16endmodule
Listing 4: Example Test Case: dead code elimination.
1module example_raw #
2(parameter BW = 8)
3(
4 input [BW-1:0] a,
5 input [BW-1:0] b,
6 input [BW-1:0] c,
7 input [BW-1:0] d,
8 output [BW-1:0] s1
9);
10 assign s1 = a + 23;
11endmodule
Listing 5: Example Test Case: dead code elimination after applying the Dead Code Elimination AST template.
1module example_raw
2#( parameter BW = 8)
3(
4 input [BW-1:0] a,
5 input [BW-1:0] b,
6 input [BW-1:0] c,
7 input [BW-1:0] d,
8 output [BW-1:0] s1,
9 output [BW-1:0] s2,
10 output [BW-1:0] s3,
11 output [BW-1:0] s4,
12 output [BW-1:0] s5,
13 output [BW-1:0] s6
14);
15 assign s1 = a + b;
16 assign s2 = a *b;
17 assign s3 = a \% b +d;
18 assign s4 = c + d + b *a;
19 assign s5 = a - b;
20 assign s6 = (b + 1) *a + d + c -b;
21endmodule
Listing 6: Example Test Case: subexpression elimination.
14

SymRTLO: Enhancing RTL Code Optimization with LLMs and Neuron-Inspired Symbolic Reasoning
1module example
2#( parameter BW = 8)
3( input [BW-1:0] a,
4 input [BW-1:0] b,
5 input [BW-1:0] c,
6 input [BW-1:0] d,
7 output [BW-1:0] s1,
8 output [BW-1:0] s2,
9 output [BW-1:0] s3,
10 output [BW-1:0] s4,
11 output [BW-1:0] s5,
12 output [BW-1:0] s6
13);
14 assign s1 = a + b;
15 assign s2 = a *b;
16 assign s3 = a \% b + d;
17 assign s4 = c + d + s2;
18 assign s5 = a - b;
19 assign s6 = s4 + s5;
20endmodule
Listing 7: Example Test Case: subexpression elimination after applying the Common Sub-Expressions Elimination
template. The Common Sub-Expressions are reused in the states after it.
1module example_raw
2#( parameter BW = 8)
3(
4 input [BW-1:0] a,
5 input [BW-1:0] b,
6 output [BW-1:0] s1,
7 output [BW-1:0] s2
8);
9 wire [BW-1:0] t1, t2;
10 assign s1 = a + b;
11 assign t1 = s1 + 0;
12 assign t2 = s1 *1;
13 assign s2 = t1 + t2;
14endmodule
Listing 8: Example Test Case: algebraic simplification.
1
2module example_raw #
3(
4 parameter BW = 8
5)
6(
7 input [BW-1:0] a,
8 input [BW-1:0] b,
9 output [BW-1:0] s1,
10 output [BW-1:0] s2
11);
12
13 assign s1 = a + b;
14 assign s2 = s1 + s1;
15
16endmodule
Listing 9: Example Test Case: algebraic simplification after applying the Temporary Variable Elimination, Dead Code
Elimination, then Expression Simplification templates.
15

SymRTLO: Enhancing RTL Code Optimization with LLMs and Neuron-Inspired Symbolic Reasoning
C RAG Example
Sample Retrieval Augmented Optimization Rule
"name": " Zero Multiplication Elimination ",
"pattern ": " Detect multiplication by zero in expressions ( e.g., , 0 * c) ",
"rewrite ": " Eliminate multiplication by zero, replacing the entire expression with zero ",
"category ": " combinational/dataflow ",
"objective improvement ": " area",
"template guidance ": " Identify vast.Times nodes with a zero operand. Replace the node with a vast.IntConst node
representing zero. ",
"function name": " ZeroMultiplicationTemplate "
Figure 6. Sample Retrieval Augmented Optimization Rule 1: Zero Multiplication Rule.
Sample Retrieval Augmented Optimization Rule
"name": " IntermediateVariableExtraction ",
"pattern ": " Detect conditional assignments to a register based on a control signal ",
"rewrite ": " Extract common sub-expressions into intermediate variables to reduce redundant logic ",
"category ": " combinational/dataflow ",
"objective improvement ": " area",
"template guidance ": " To implement this rule in a Python template subclassing BaseTemplate, use pyverilog AST
manipulation to identify conditional assignments (vast.IfStatement) and extract the common sub-expressions into
separate assignments. Look for vast.Identifier nodes that are assigned conditionally and create new vast.Assign nodes
for the intermediate variables. Ensure that the new assignments are placed before the conditional logic to maintain
correct data flow ",
"function name": " IntermediateVariableExtractionTemplate "
Figure 7. Sample Retrieval Augmented Optimization Rule 2: Intermediate Variable Extraction.
Hardware Optimization Rule
"name": " ReplaceRippleCarryWithCarryLookahead ",
"pattern ": " Detects a ripple carry adder implementation using a series of full adders connected in sequence ",
"rewrite ": " Transforms the ripple carry adder into a carry lookahead adder by using partial full adders and generating
carry bits in parallel ",
"category ": " combinational/dataflow ",
"objective improvement ": " area, delay ",
"template guidance ": null ,
"function name": null
Figure 8. Sample Retrieval Augmented Optimization Rule 3: Replace Ripple Carry with Carry Lookahead, no template guidance is
needed since it is an abstract rule.
16