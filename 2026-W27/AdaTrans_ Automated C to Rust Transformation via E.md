# AdaTrans: Automated C to Rust Transformation via Error-Adaptive Repair

**Authors**: Xiaofan Liu, Zecan Li, Zhuang Zhao, Ziqi Shuai, Yanming Yang, Qi Xin, Jifeng Xuan

**Published**: 2026-06-30 14:11:54

**PDF URL**: [https://arxiv.org/pdf/2606.31706v1](https://arxiv.org/pdf/2606.31706v1)

## Abstract
The automated transformation of C code to Rust is challenging due to Rust's strict ownership and borrowing semantics. While Large Language Models (LLMs) show promise, they often produce code that violates these rules or relies on unsafe constructs. We propose AdaTrans, a framework that addresses these issues through three core mechanisms: a Strategy-Driven Retrieval-Augmented Generation (RAG) mechanism to map compiler errors to specific repairs, an Error-Stratified Transformation Strategy (ESTS) that adapts its behavior based on error types, and a multi-stage validation pipeline to ensure both compilability and functional equivalence. Evaluating on a dataset of 104 algorithmic problems, AdaTrans achieves a mean compilation pass rate of 95.51% and a mean solve rate of 81.09%, significantly outperforming existing tools while maintaining an unsafe file rate of only 1.19%.

## Full Text


<!-- PDF content starts -->

AdaTrans: Automated C-to-Rust Transformation via Error-Adaptive Repair
XIAOFAN LIU,School of Computer Science Wuhan University, China
ZHUANG ZHAO,School of Computer Science Wuhan University, China
ZECAN LI,School of Computer Science Wuhan University, China
ZIQI SHUAI,School of Computer Science Wuhan University, China
YANMING YANG,School of Computer Science Wuhan University, China
QI XIN,School of Computer Science Wuhan University, China
JIFENG XUAN∗,School of Computer Science Wuhan University, China
Automated transformation of C to Rust remains challenging due to the strict ownership system and borrowing semantics of Rust.
Large language models (LLMs) show potential for code generation, but frequently produce Rust code that violates ownership rules
or relies on unsafe blocks. Three factors compound this problem. First, the ownership system of Rust presents a semantic barrier
that general-purpose LLMs cannot reliably cross. Second, existing approaches underutilize compiler feedback for program repair.
Third, without targeted verification, generated code tends to circumvent safety constraints rather than satisfy them. We propose
AdaTrans, a framework that addresses these challenges by integrating three mechanisms. First, a Strategy-Driven Retrieval-Augmented
Generation (RAG) mechanism maps compiler errors to specific repair strategies. Second, an Error-Stratified Transformation Strategy
(ESTS) classifies compiler diagnostics into semantic error categories. Per-category temperature scheduling and stagnation detection
adapt the repair behavior to balance generation diversity with constraint satisfaction. Third, a multi-stage validation pipeline ensures
both compilability and functional equivalence through iterative repair. We focus on file-level transformation of self-contained C
modules with standard input/output behavior, a controlled setting that isolates the core semantic mapping from C to Rust and admits
differential-testing oracles for rigorous functional-equivalence checking. We evaluate AdaTrans on a dataset of 104 algorithmic
problems from LeetCode Weekly Contests. We compare it with three existing LLM-based C-to-Rust tools, a zero-shot LLM baseline,
and the c2rust AST-level transpiler. Across three independent runs, AdaTrans achieves a mean compilation pass rate of 95.51% ( ±
1.11%) and a mean solve rate of 81.09% ( ±3.09%) under a fuzz-based test oracle, with a mean unsafe file rate of 1.19%. AdaTrans
improves the solve rate over the strongest existing LLM-based tool by 59.94 percentage points. It also keeps the unsafe file rate at
1.19%, well below that of the c2rust AST-level transpiler. These results demonstrate that adapting repair strategies to the semantic
characteristics of compiler diagnostics can reconcile transformation correctness with memory safety.
Additional Key Words and Phrases: Code Transformation, Large Language Models, Error-Stratified Repair, Adaptive Temperature
Scaling, Memory Safety, Automated Program Repair
∗Corresponding author.
Authors’ Contact Information: Xiaofan Liu, xfliu@whu.edu.cn, School of Computer Science and Wuhan University, China, Wuhan; Zhuang Zhao,
zhao2z@whu.edu.cn, School of Computer Science and Wuhan University, China, Wuhan; Zecan Li, zecanli@whu.edu.cn, School of Computer Science
and Wuhan University, China, Wuhan; Ziqi Shuai, szq@whu.edu.cn, School of Computer Science and Wuhan University, China, Wuhan; Yanming Yang,
yanming.yang@whu.edu.cn, School of Computer Science and Wuhan University, China, Wuhan; Qi Xin, qxin@whu.edu.cn, School of Computer Science
and Wuhan University, China, Wuhan; Jifeng Xuan, jxuan@whu.edu.cn, School of Computer Science and Wuhan University, China, Wuhan.
Permission to make digital or hard copies of all or part of this work for personal or classroom use is granted without fee provided that copies are not
made or distributed for profit or commercial advantage and that copies bear this notice and the full citation on the first page. Copyrights for components
of this work owned by others than the author(s) must be honored. Abstracting with credit is permitted. To copy otherwise, or republish, to post on
servers or to redistribute to lists, requires prior specific permission and/or a fee. Request permissions from permissions@acm.org.
©2026 Copyright held by the owner/author(s). Publication rights licensed to ACM.
Manuscript submitted to ACM
Manuscript submitted to ACM 1arXiv:2606.31706v1  [cs.SE]  30 Jun 2026

2 Xiaofan Liu, Zhuang Zhao, Zecan Li, Ziqi Shuai, Yanming Yang, Qi Xin, and Jifeng Xuan
ACM Reference Format:
Xiaofan Liu, Zhuang Zhao, Zecan Li, Ziqi Shuai, Yanming Yang, Qi Xin, and Jifeng Xuan. 2026. AdaTrans: Automated C-to-Rust
Transformation via Error-Adaptive Repair.ACM Trans. Softw. Eng. Methodol.1, 1 (July 2026), 37 pages. https://doi.org/XXXXXXX.
XXXXXXX
1 Introduction
System programming languages underpin operating systems, embedded devices, and high-performance computing
infrastructure. Among these languages, C has maintained its position as the de facto standard for over five decades. This
enduring dominance stems from its low-level hardware control and high execution efficiency. However, the unrestrained
memory access capabilities of C represent a critical weakness, which manifests as serious security vulnerabilities,
including buffer overflows, null pointer dereferences, and data races. The prevalence of memory-related vulnerabilities
in system software motivates the search for safer programming language alternatives. Rust has emerged as a prominent
alternative to C, with widespread adoption in both academia and industry. The memory safety of Rust rests on three
mechanisms: ownership, borrowing, and lifetimes. These mechanisms enforce safety checks at compile time and
eliminate the majority of memory errors without sacrificing runtime performance.
The transformation of codebases from C to Rust remains a challenging engineering task that traditionally demands
extensive manual effort. Existing automated tools, such as c2rust [ 14] and Laertes [ 8], generate syntactically correct
Rust code that relies heavily on unsafe blocks to bypass the borrow checker. Such reliance on unsafe blocks effectively
defeats the original purpose of memory safety during migration to Rust [ 5]. Large Language Models (LLMs), pre-trained
on massive open-source codebases, have recently demonstrated potential in code generation and transformation [ 3,15]
by capturing complex syntactic structures across multiple programming languages. General-purpose LLMs, however,
still face severe challenges in C-to-Rust transformation. First, the ownership rules of Rust impose a semantic barrier that
general-purpose models cannot reliably cross. Generated code frequently violates borrowing and lifetime invariants
because the LLM fails to map manual memory management in C to the explicit ownership model of Rust. Second,
existing approaches underutilize compiler feedback for iterative repair. Rust compiler diagnostics encode rich semantic
information, yet LLMs struggle to interpret complex error messages, which restricts automated repairs to superficial
syntactic adjustments rather than structural corrections. Third, without targeted verification mechanisms, most gen-
eration methods prioritize syntactic correctness over functional integrity. Unsafe blocks and logical errors therefore
persist in the generated code.
Existing research on C-to-Rust transformation spans multiple granularities. Recent LLM-based approaches [ 13,42,54]
primarily target project-level transformation, improving code idiomaticity through techniques such as skeleton-guided
evolution, type-directed mutation, and pointer knowledge graphs. These advances are complementary to our work,
because the central difficulty of safe transformation persists across granularities. It lies in the semantic mapping between
manual memory management in C and the ownership model of Rust, where functional equivalence and memory safety
must hold simultaneously. We isolate this semantic core by studying file-level transformation of self-contained C
modules with standard input/output behavior. This scope is a deliberate methodological choice rather than a capability
limitation. Self-contained modules admit ground-truth differential-testing oracles, making functional equivalence
directly verifiable, whereas project-level migration rarely provides comparable validation mechanisms. Our repair
mechanism treats each file-unit as a minimal project, so it is not bound to this granularity. The controlled setting
removes confounding factors such as build configuration, cross-module dependencies, and external bindings, allowing
us to focus on semantic fidelity and safety. Such units also arise naturally during incremental migration of legacy C
Manuscript submitted to ACM

AdaTrans: Automated C-to-Rust Transformation via Error-Adaptive Repair 3
codebases [ 19], where standalone utilities and self-contained libraries are often migrated first before larger project-wide
integration, while remaining straightforward to validate in isolation [20].
We assess the current state of file-level C-to-Rust transformation under this controlled setting. For this purpose, we
construct a dataset of 104 algorithmic problems sourced from LeetCode Weekly Contests released after the knowledge
cutoff of the LLM snapshot used in our experiments (Section 4). This design mitigates, but does not eliminate, the risk
of benchmark contamination with respect to the evaluated model. On this dataset, a zero-shot LLM ( gpt-4o-mini )
reaches a pass@100 estimate of 70.58%. We compute this estimate from 200 independent samples per problem with the
unbiased estimator of Chen et al. [ 3]. We use this high-budget resampling result as a strong brute-force reference under
the same backbone. This reference shows that repeated independent sampling alone is still insufficient for reliable
transformation. We therefore introduce AdaTrans, an end-to-end framework that transforms individual C files into
predominantly safe and compilable Rust programs and preserves behavioral consistency.
The key design intuition of AdaTrans is to adapt repair behavior to the type of failure signal observed during
validation. Different error categories correspond to different repair needs, and a uniform strategy across heterogeneous
failures is often ineffective. Syntax errors, as relatively rigid defects with narrow repair spaces, tend to benefit from low-
temperature exploitation. Ownership-related errors often require moderate-temperature exploration over alternative
memory-management patterns. Behavioral failures, where the program compiles but produces incorrect outputs,
typically require higher-temperature exploration to escape locally consistent but semantically incorrect solutions [ 43,56].
AdaTrans operationalizes this intuition through an error-driven repair loop that converts validation feedback into
signals for retrieval and repair control (Section 3).
The main contributions of this paper are as follows:
(1)A compiler-feedback-driven framework for C-to-Rust transformation. We propose AdaTrans, a three-
phase generate-verify-repair pipeline designed for self-contained C files in a controlled file-level migration
setting.
(2)An operational error-stratified repair strategy. We introduce ESTS, which groups validation signals into
coarse error categories and adapts repair behavior through category-aware temperature scheduling and stagnation
recovery.
(3)A strategy-driven retrieval mechanism for repair guidance. We design a RAG module that maps diagnostic
signals to repair templates and Rust-specific knowledge, and provides structured guidance beyond raw compiler
messages.
(4)An empirical study of correctness-safety trade-offs under a fixed LLM backbone. On a 104-problem dataset
designed to mitigate contamination risk with respect to the evaluated model snapshot, AdaTrans achieves a mean
compilation pass rate of 95.51% ( ±1.11%) and a mean solve rate of 81.09% ( ±3.09%) across three independent
runs, while maintaining a meanunsafefile rate of 1.19%.
The remainder of this paper is organized as follows. Section 2 introduces the memory model differences between C
and Rust, analyzes the limitations of LLM-based code generation, and presents a motivating case study. Section 3 details
the problem formulation and the three-phase pipeline of the AdaTrans framework. Section 4 describes the experimental
setup, and Section 5 presents the evaluation results. Section 6 discusses the threats to validity. Section 7 reviews related
work. Finally, Section 8 concludes the paper.
Manuscript submitted to ACM

4 Xiaofan Liu, Zhuang Zhao, Zecan Li, Ziqi Shuai, Yanming Yang, Qi Xin, and Jifeng Xuan
2 Background
2.1 Memory Model Differences between C and Rust
The C memory model operates as a direct abstraction of the von Neumann architecture. Programmers manipulate a
linear address space through raw pointers. This design grants flexibility and performance but places the entire burden
of memory safety on the programmer. Memory allocated via malloc must be released through free exactly once. The
absence of compiler enforcement gives rise to three critical error classes:
•Use-After-Free (UAF): Dereference of a dangling pointer after its underlying allocation has been deallocated.
•Double Free: Release of the same allocation more than once, which corrupts heap metadata and enables
exploitation.
•Memory Leak: Failure to deallocate memory, which causes gradual resource exhaustion.
Rust prevents UAF and double-free errors at compile time through an ownership model rooted in an affine type
system. Automatic resource management (RAII) further mitigates memory leaks by tying deallocation to scope exit. For
every value𝑣, the compiler maintains a unique bindingO(𝑥,𝑣)and enforces three core axioms:
•Uniqueness:∀𝑣,∀𝑡 :∃!𝑥such thatO(𝑥,𝑣) holds at time 𝑡. Ownership of every resource is unambiguous at
every program point.
•Move Semantics: An assignment 𝑦=𝑥 transfers ownership so that O(𝑦,𝑣) holds and𝑥enters an uninitialized
state (⊥). This rule prevents the shallow-copy hazards inherent in C pointer aliasing.
•Borrowing Invariants: Within any given scope, a value may be borrowed through either multiple immutable
references (& 𝑥) or exactly one mutable reference ( &mut x ), but not both simultaneously. This mutual-exclusion
rule eliminates data races by construction.
The borrow checker verifies all memory accesses against these axioms through region-based lifetime analysis. C-to-
Rust transformation therefore demands more than syntactic remapping. It requires a structural transformation from
unconstrained pointer manipulation to strict ownership enforcement.
2.2 Limitations of LLM-Based Code Generation
LLMs demonstrate strong performance on general programming tasks, yet encounter severe bottlenecks when the
target language imposes deterministic static constraints. Three architectural conflicts underlie these failures.
Stochastic Generation vs. Deterministic Constraints.LLM code generation follows an autoregressive process
that predicts 𝑃(𝑥𝑡|𝑥<𝑡)from local context. This mechanism reproduces high-frequency syntactic patterns effectively
but struggles with the non-local nature of Rust ownership. Finite context windows impose a myopic bias on synthesis.
This bias yields code that reads fluently at the local level yet violates global borrowing invariants and lifetime rules.
Open-Loop Synthesis and Feedback Deficiency.The standard LLM generation pipeline operates as an open-loop
system with no integration of compiler feedback. In manual development, programmers perform root-cause analysis
with the aid of external documentation when they encounter errors. Without this iterative feedback loop, LLMs
produce syntactically plausible but semantically invalid code. Complex diagnostics such as borrowed value does not
live long enough remain opaque to general-purpose models. The consequence is repetitive, ineffective edits or the
introduction of new defects.
Unsafe Shortcuts in Generated Code.When the borrow checker rejects generated code, LLMs frequently resort
tounsafe blocks or raw pointers to suppress compiler errors. Qin et al. [ 35] document that unsafe usage is widespread
Manuscript submitted to ACM

AdaTrans: Automated C-to-Rust Transformation via Error-Adaptive Repair 5
in real-world Rust codebases. Because system-level training corpora contain abundant unsafe patterns, models learn to
reproduce them readily and prioritize syntactic acceptance over semantic safety. Such shortcuts silence compilation
failures but simultaneously circumvent the safety guarantees that motivate the C-to-Rust migration. Effective guidance
toward idiomatic safe Rust remains a central challenge for automated transformation.
2.3 Motivating Case Study
An illustrative case of string collection shows how these limitations manifest in practice. In typical C code (Fig. 1), the
programmer consolidates multiple strings into a heap-allocated buffer and stores their addresses in a pointer array:
Fig. 1. Typical C pattern for string collection.
Direct transformation attempts by general-purpose LLMs typically mimic the pointer-to-buffer logic of the C original,
which triggers immediate borrow-checker violations (Fig. 2):
Fig. 2. LLM-generated Rust code triggering E0597.
Analysis of Model Failure.General-purpose models struggle to resolve theE0597error for three reasons:
•Superficial Corrections: Models treat compiler errors as syntactic mismatches rather than structural ownership
conflicts. The result is trial-and-error edits on irrelevant code sections (e.g., substitution of different I/O methods).
•Safety Circumvention: When simple patches fail, models resort to wrapping code in unsafe blocks or calling
Box::leak . These modifications suppress the error but introduce latent memory leaks, which defeats the purpose
of the transformation.
•Insufficient Reasoning Depth: Resolution of this lifetime conflict requires a structural shift from reference
storage (Vec<&str> ) to ownership acquisition ( Vec<String> ). Without targeted guidance, models lack the
capacity to redesign data structures on the basis of variable lifetimes within iterative scopes.
Manuscript submitted to ACM

6 Xiaofan Liu, Zhuang Zhao, Zecan Li, Ziqi Shuai, Yanming Yang, Qi Xin, and Jifeng Xuan
This case illustrates a difficulty that goes beyond syntax: reconciling manual memory management in C with the
strict ownership constraints of Rust. The dataset-wide error analysis in Section 5 is consistent with this view, where
syntactic errors are resolved quickly, ownership-related errors require deeper structural repair, and behavioral errors
remain the dominant bottleneck. AdaTrans addresses these challenges through error-driven adaptation, as described in
Section 3.
3 Methodology
3.1 Problem Formulation
We formalize the C-to-Rust transformation as a constrained generation problem. Given a source C program 𝑆𝐶, the
objective is to generate a target Rust program 𝑆∗
𝑅that satisfies two strict criteria: compilability (static correctness) and
functional equivalence (I/O consistency).
3.1.1 Functional Equivalence (I/O Consistency).Direct proof of semantic equivalence between two programs remains
generally undecidable. Therefore, we define functional equivalence based on strict Input/Output (I/O) consistency
over a representative test suite I. Let Out(𝑃,𝑖) denote the standard output of program 𝑃under test case 𝑖∈I . The
functional consistency function SemEquiv is defined as:
SemEquiv(𝑆 𝑅,𝑆𝐶) ⇐⇒ ∀𝑖∈I,Norm(Out(𝑆 𝑅,𝑖))=Norm(Out(𝑆 𝐶,𝑖))(1)
where Norm(·) is a normalization operator that collapses whitespace, trims trailing newlines, and applies floating-point
epsilon comparison ( 𝜖≤ 10−6) when the output domain includes real-valued results. In the current evaluation, all
problem outputs are exact integers or strings, so only whitespace normalization is applied. The epsilon comparison is
retained in the formulation for generality.
3.1.2 Compilability and Memory Safety.The target program 𝑆𝑅must pass the strict static analysis of the Rust compiler.
LetV𝑅𝑢𝑠𝑡denote the verification process of the compiler, which encompasses type checking, borrow checking, and
lifetime analysis. The soundness constraint𝑉(𝑆 𝑅)is defined as:
𝑉(𝑆𝑅)=True⇐⇒ V 𝑅𝑢𝑠𝑡(𝑆𝑅)=Pass (2)
This constraint guarantees that the generated code contains no syntax errors and adheres to the ownership rules of
Rust, thereby ensuring memory safety for all code outsideunsafeblocks.
3.1.3 Iterative Repair Process.The generation of 𝑆∗
𝑅requires iterative repair to satisfy both criteria. We model this
as a sequence of states {𝑆(0)
𝑅,𝑆(1)
𝑅,...,𝑆(𝑡)
𝑅}. Each transition relies on the LLM generating a refined version based on
previous feedback:
𝑆(𝑡+1)
𝑅∼𝑃𝐿𝐿𝑀(𝑆𝑅|𝑆𝐶,𝐸(𝑡),K;𝜃𝑡)(3)
where𝐸(𝑡)represents the diagnostic feedback (compiler errors or test failures) from 𝑆(𝑡)
𝑅,Kdenotes external knowledge
retrieved via a Retrieval-Augmented Generation (RAG) module (detailed in Section 3.5), and 𝜃𝑡is the temperature
parameter controlling generation diversity. We use 𝜃for category-specific temperatures ( 𝜃𝑆𝐿,𝜃𝑀𝑆,𝜃𝐿𝐵) and𝑇for the
base temperature. In Algorithm 1, the variable𝑇is set by the ESTS mappingF 𝐸𝑆𝑇𝑆 and corresponds to𝜃 𝑡in Eq. (3).
The optimization goal finds the earliest iteration𝑡at which both compilability and I/O consistency are satisfied:
𝑡∗=min{𝑡≥0|𝑉(𝑆(𝑡)
𝑅)∧SemEquiv(𝑆(𝑡)
𝑅,𝑆𝐶)}, 𝑆∗
𝑅=𝑆(𝑡∗)
𝑅(4)
Manuscript submitted to ACM

AdaTrans: Automated C-to-Rust Transformation via Error-Adaptive Repair 7
In practice, AdaTrans terminates at the first successful iteration or after exhausting the budget 𝑀. The iteration count
𝑡∗is a termination criterion, not an explicit optimization target.
3.2 AdaTrans Framework Overview
The AdaTrans framework addresses the limitations of LLMs in C-to-Rust transformation through an iterative generate-
verify-repair paradigm. This paradigm guides candidate code toward compilation correctness and functional consistency.
AdaTrans iteratively integrates compiler diagnostics and execution feedback from a representative subset of test cases.
This selective testing strategy verifies core functional requirements without exhausting computational resources during
each iteration. Based on the test outcomes and compiler error messages, the framework dynamically modulates its
transformation strategies. These execution traces, embedded in the model context, provide explicit feedback for the
next repair iteration.
Phase 1. Multi -Stage Validation
(Deterministic Oracle)Phase 1. Multi -Stage Validation
(Deterministic Oracle)Phase 3. Error -Stratified
Transformation Strategy (ESTS)Phase 3. Error -Stratified
Transformation Strategy (ESTS)Phase 2. Strategy -Driven RAG
(Knowledge Enhancement)Phase 2. Strategy -Driven RAG
(Knowledge Enhancement)
C Source 
Code SCInitial Rust 
Candidate
SR(0)
Initial  LLM
Transformation
√ Syntax Errors
√ Linking Errors
√ Ownership Errors
√ Borrow Errors
√ Non-zero Exit CodeCompiler Verification
( cargo build)
√ Run on Test Suite  I
√ Capture Output Out
√ Monitor stderr 
√ Normalize & CompareExecution  Verification
( test runner)
Compile 
& All Test
Pass？True
FalseSuccess
Return SR(t)
Diagnostic
Signal
E(t)
Error Parsing
< Extract E rror Signature  ei >RVFG Lookup
< fretrieve (ei) >Retrieve Template  Ti
< Prompt Library  >Retrieve Knowledge Ki
< Rust Documentation  >Construct Augmented Prompt
< include SR(t), Ti, Ki, E(t) >
 Diagnostic Classification
 Local -Loop Stagnation Escape 
One error -category  repeats  > NL 
trigger RESET Temperature Scheduling
T = FESTS(Category)
LLM -based  
Repair  
New Rust 
Candidate
SR(t+1)Syntactic -Linking (SL)   θSL = 0.1
Memory -Semantic (MS)   θMS = 0.5
Logic -Behavioral (LB)   θLB = 1.2
Ambiguous -Fallback (AF)   θAF = 1.0
Fig. 3. Overall architecture and inter-phase dependencies of the AdaTrans framework.
Fig. 3 illustrates how diagnostic signals flow through the three phases within each repair iteration. This architecture
operates as a closed loop that consists of three sequential phases. Phase 1 implements a multi-stage validation pipeline
to extract diagnostic signals from both the compiler and the execution environment. Phase 2 employs a Strategy-Driven
Retrieval-Augmented Generation (RAG) module to construct adaptive prompt templates based on the diagnostic feedback.
This module constructs augmented prompts by retrieving error-specific repair templates and relevant documentation
from a Rust knowledge repository. Phase 3 implements the Error-Stratified Transformation Strategy (ESTS), which
classifies diagnostic signals into error categories and adapts the repair behavior accordingly through per-category
temperature scheduling and stagnation detection. The error category extracted during validation drives both the
retrieval function in Phase 2 and the temperature scheduling in Phase 3. Syntax errors trigger targeted templates
with low-entropy sampling for precise correction, while test failures activate reflection templates with high-entropy
sampling to escape local optima.
3.3 Taxonomy of Violation Signals
The complex space S𝑅𝑢𝑠𝑡of valid Rust programs demands a structured search strategy. We establish a taxonomy that
classifies the diagnostic feedback 𝐸(𝑡)(extracted during the validation phase) according to thescope of reasoningand
constraint rigidityrequired for repair. This taxonomy partitions the violation signals into four disjoint semantic domains:
Manuscript submitted to ACM

8 Xiaofan Liu, Zhuang Zhao, Zecan Li, Ziqi Shuai, Yanming Yang, Qi Xin, and Jifeng Xuan
•Syntactic-Linking (SL) Violations: These represent violations of theContext-Free Grammaror basic name
resolution (e.g., missing semicolons, unresolved crates). In the state space, SL errors indicate that 𝑆𝑅is not a
well-formed string in L(𝑅𝑢𝑠𝑡) . These defects aredeterministicand typically require only localized, low-entropy
sampling for repair.
•Memory-Semantic (MS) Violations: These are unique to the Rust affine type system and borrow checker (e.g.,
E0597 ,E0499 ). Unlike SL errors, MS violations often stem fromnon-local structural conflictsbetween variable
lifetimes and ownership transfer. Resolving an MS error requires the LLM to reconstruct the ownership structure.
This repair demands moderate-entropy sampling to explore alternative memory management patterns (e.g.,
cloning vs. borrowing).
•Logic-Behavioral (LB) Violations: These occur when 𝑉(𝑆𝑅)=True butSemEquiv(𝑆 𝑅,𝑆𝐶)=False . These
signals indicate that while 𝑆𝑅is a valid Rust program, itsexecution semanticsdeviate from the source. Since the
root cause (e.g., algorithmic flaws, off-by-one errors) is often decoupled from the syntax, escaping these local
optima requires high-entropy stochastic perturbation to trigger significant structural mutations.
•Ambiguous-Fallback (AF) Violations: For unseen error patterns, we define a fallback category for errors not
covered by the three categories above, such as those generated by external macros or third-party crates. When
an AF violation occurs, the system defaults to baseline-entropy sampling and relies on zero-shot LLM reasoning.
This taxonomy directly informs both the retrieval strategy in Phase 2 and the entropy mappingF 𝐸𝑆𝑇𝑆 in Phase 3.
3.4 Phase 1: Multi-Stage Validation Pipeline
Phase 1 implements a multi-stage validation pipeline to evaluate the generated Rust code. This pipeline functions as a
deterministic oracle that verifies whether the candidate code satisfies both the strict syntactic invariants of Rust and
the functional equivalence of the original C implementation.
Initially, the pipeline invokes the Rust compiler ( cargo build ) to perform strict static analysis. This compilation
step captures syntax errors, linking failures, and ownership violations. The compiler returns non-zero exit codes along
with standard error outputs when it detects these structural flaws. These outputs contain the exact error codes (e.g.,
E0597) required for the subsequent repair phases.
If the code compiles successfully, the pipeline proceeds to the dynamic execution stage to evaluate SemEquiv(𝑆 𝑅,𝑆𝐶).
The Test Runner executes the compiled binary over the input domain I(a predefined fuzzing suite comprising both
typical use cases and edge-case mutations) to collect the execution trace Out(𝑆𝑅,𝑖). The system monitors standard error
streams (stderr) during this process to intercept runtime panics (e.g., out-of-bounds access or unhandled Result types)
that bypass static analysis.
Output comparison applies the normalization operator Norm(·) (Equation (1)), which collapses whitespace and trims
trailing newlines to account for benign formatting differences across system architectures.
For example, during the evaluation of a dynamic programming transformation task, the generated code passed
the strict ownership checks of the compiler but failed the execution stage. As shown in Fig. 4, the validation pipeline
successfully caught a hidden panic that bypassed the static compiler checks.
Finally, the validation module categorizes these results into actionable diagnostic signals. These signals distinguish
between rigid compilation errors (SL, MS) and loose behavioral errors (LB) to provide targeted optimization feedback.
Manuscript submitted to ACM

AdaTrans: Automated C-to-Rust Transformation via Error-Adaptive Repair 9
Fig. 4. A hidden runtime panic captured by the validation pipeline during a transformation task.
3.5 Phase 2: Strategy-Driven RAG for Knowledge Enhancement
Phase 2 formalizes a Strategy-Driven Retrieval-Augmented Generation (RAG) mechanism to bridge the gap between raw
compiler diagnostics and the structured repair context that LLMs require. Recent LLM-based C-to-Rust transformation
tools [ 13,42] incorporate compiler feedback into their repair loops, yet they treat diagnostics uniformly without
distinguishing error categories. Direct injection of raw compiler feedback into LLMs is limited by three factors.
•Misdiagnosis of Root Causes: The Rust compiler evaluates ownership strictly at the point of violation, which
often misattributes the root cause of a lifetime conflict to a downstream usage rather than its structural origin.
•Suboptimal Heuristic Suggestions: The heuristic fixes provided by the compiler (e.g., suggesting .clone() or
adding lifetime annotations) prioritize immediate compilation over structural correctness, which misleads the
model into adopting suboptimal patterns such as unnecessary cloning orBox::leak().
•Absence of Global Context: Raw feedback lacks the architectural context necessary for global refactoring,
which constrains the LLM to localized, often conflicting patches.
We address these limitations through an error-to-template mapping formalized as aRust Violation-Fix Graph
(RVFG). We formalize the RVFG as a directed bipartite graph G=(V𝐸,V𝑆,E). Here,V𝐸represents the discrete space
of possible Rust compiler error signatures (e.g., E0597 ),V𝑆=T×K represents the set of Semantic Anchors combining
repair operators (Prompt Templates T) and official documentation snippets ( K), andErepresents the deterministic
mapping edges. As shown in the central component of Fig. 5, the violation nodes V𝐸(left) are deterministically linked to
the semantic anchors V𝑆(right) through the mapping edges E. The Template Library Tcomprises a set of role-specific
meta-prompts designed for different failure modalities, including a syntax-correction template ( T𝑆𝐿), an ownership-
reasoning template ( T𝑀𝑆), an algorithmic-reflection template ( T𝐿𝐵), and a generalized exploration template ( T𝐴𝐹). The
RVFG acts as a deterministic mapping from error signatures to repair strategies. In practice, this graph reduces to a
deterministic lookup table indexed by error code, with 𝑂(1)retrieval latency and straightforward extensibility as new
Manuscript submitted to ACM

10 Xiaofan Liu, Zhuang Zhao, Zecan Li, Ziqi Shuai, Yanming Yang, Qi Xin, and Jifeng Xuan
error codes are added. In this graph, nodes represent specific violation semantics defined by exact compiler diagnostic
codes (e.g., the 517 distinct Rust compiler error codes extracted from version 1.75), and edges represent the mapping to
their corresponding semantic anchors.
We define a deterministic retrieval function 𝑓𝑟𝑒𝑡𝑟𝑖𝑒𝑣𝑒 :V𝐸→V𝑆, which is constrained by the edges EinG. When
a new diagnostic error 𝑒𝑖is encountered from Phase 1, the module performs exact error-code matching against the
nodes inV𝐸. This approach aligns with the principles of deterministic static analysis, as compiler errors are strongly-
typed signals whose resolution demands precision rather than semantic approximation. If the system encounters an
Ambiguous-Fallback (AF) error (e.g., from an unseen third-party macro), 𝑓𝑟𝑒𝑡𝑟𝑖𝑒𝑣𝑒 gracefully degrades to the generalized
exploration template T𝐴𝐹and delegates the repair to zero-shot LLM reasoning. The RVFG is not a rigid code-replacement
table. Rather, it providessemantic anchors. The actual code transformation is not hard-coded but dynamically synthesized
by the LLM, which applies these semantic constraints to the specific local or inter-procedural context of the target
function.
Compiler Error Signature e i
Example:  E0597
borrowed value does not 
live long enough
Raw Compiler Output
error[E0597] : `local_vec`
does not live long enough
--> src/main.rs:6:19
6 | vec_ref = &local_vec;
| ^^^^^^^^^^
| borrowed value does not
| live long enough
7 | }
| - `local_vec` dropped
| here while borrowed
Legend  - violation categories
SL — Syntactic -Linking
MS — Memory -Semantic
LB — Logic -Behavioral
AF — Ambiguous -Fallback
Violation node
Semantic anchorMapping edge E          Rust Violation -Fix Graph (RVFG)
         G=(GE,GS,E )      Diagnostic  Signal
      from Phase 1 211          Retrieved Anchors
         Semantic ( Ti,Ki)3         Prompt Construction
         Augmentation4         Retrieved Anchors
         Semantic ( Ti,Ki)5
Violation Space 
𝓥ESemantic Anchors  
𝓥S=𝕋×𝕂
E0597 · MS
Lifetime MS
E0499 · MS
Mut borrow
E0597 · MS
aliasing
E0597 · SL
unresolved import
Test Failure  · LB
Lifetime ·  MS
Unknown Error · AF
macro / 3rd -party𝓣LB Reflection Template
logic / behavioral fix
𝓚LB Algorithmic Tips𝓣LB Reflection Template
logic / behavioral fix
𝓚LB Algorithmic Tips
𝓣AF Fallback Template
generalized exploratio n
𝓚AF General Docs𝓣AF Fallback Template
generalized exploratio n
𝓚AF General Docs𝓣SL Syntax Template
syntax / linking fix
𝓚SL Rust Reference𝓣SL Syntax Template
syntax / linking fix
𝓚SL Rust Reference𝓣MS Ownership Template
ownership / borrow repair
𝓚MS Lifetime + Borrow Docs𝓣MS Ownership Template
ownership / borrow repair
𝓚MS Lifetime + Borrow Docs
...
Deterministic Retrieval  fretriev e : 𝒱 E→𝒱S
Exact match on error code ei (O(1) lookup table)
Unseen code → fallback ( 𝓣AF , 𝓚AF )
517 Rust error codes (rustc 1.75 )Retrieved Template 𝓣i
Ownership Repair Strategy (MS)
• Extend scope/lifetime of
  `local_vec`
• Prefer borrowing over cloning
• Share via Rc<Vec<i32>> when
  ownership must be shared
• Avoid unnecessary .clone()
• Rebuild the ownership tree,
  not local patches
• ...
Retrieved Knowledge  𝓚i
Rust lifetime invariants
• A reference cannot outlive
  its referent.
• Lifetimes are enforced at
  compile time.
• A borrow must end before
  its owner is dropped.
• ...Source C SC
int a[]={1,2,3}; /* origin */
Current Rust Code S R(t)
let local_vec = vec![1,2,3];
Diagnostic Signal ℰ(t)
E0597: borrowed value …
Retrieved Template  𝓣i
Retrieved Knowledge 𝓚i
Repair Instruction 
Fix rules : …
Final Augmented Prompt  
semantic anchor for the LLMLLM Repair
Generate with the augmented
prompt under temperature
𝛳t (set by ESTS, Phase 3 )
SR(t+1) ~ PLLM( · | SC , ℰ(t), 𝒦 ; θt )
Updated Rust Candidate
SR(t+1)
fn process() {
      // expand local_vec scope
      let v = Rc::new(
      vec![1,2,3]);
      // ... use v ...
}
→ re-enter Phase 1 for validation
Iterative repair loop — back to  Phase 1
Fig. 5. Execution flow of the Strategy-Driven RAG mapping mechanism. The central panel depicts the Rust Violation-Fix Graph
G=(V𝐸,V𝑆,E), mapping compiler error signatures (V 𝐸) to semantic anchors (V 𝑆) via deterministic edges (E).
Fig. 5 illustrates the execution flow of this mapping strategy. During the initial step of each repair iteration, the RAG
module parses the diagnostic signal from the validation phase. The module then evaluates 𝑓𝑟𝑒𝑡𝑟𝑖𝑒𝑣𝑒(𝑒𝑖)to construct the
augmented prompt.
For instance, consider Fig. 6, which illustrates a common MS scenario where the generated code produces an E0597
error (“borrowed value does not live long enough”).
A naive compiler feedback approach would simply pass the raw message local_vec dropped here while still
borrowed to the LLM. This typically causes the LLM to apply a local patch, such as inserting .clone() inappropriately,
which may degrade performance or fail structurally. In contrast, our Strategy-Driven RAG maps E0597 to the Memory-
Semantic (MS) category. It retrieves the fix_ownership.j2 template (T𝑖) and injects the formal definition of Rust
lifetime invariants ( K𝑖) into the prompt. The augmented prompt serves as a semantic anchor that restricts the stochastic
Manuscript submitted to ACM

AdaTrans: Automated C-to-Rust Transformation via Error-Adaptive Repair 11
Fig. 6. Example of an MS error (E0597) triggering RAG injection.
Table 1. Entropy-Rigidity Mapping in ESTS
Error Category Constraint Type Target Strategy Temperature Level
SL (Syntactic-Linking) Rigid / Deterministic Exploitation (Precision) Low Entropy
MS (Memory-Semantic) Semantic / Logical Balanced (Multi-path Reasoning) Moderate Entropy
LB (Logic-Behavioral) Loose / Behavioral Exploration (Mutation) High Entropy
AF (Ambiguous-Fallback) Unseen / Fallback Generalized Exploration Baseline Entropy
search space to the domain of memory-safe Rust invariants. This constraint prevents the LLM from relying on unsafe
workarounds and steers it toward an idiomatic structural solution (e.g., expanding the scope of local_vec or utilizing
Rc<Vec<i32»).
3.6 Phase 3: Error-Stratified Transformation Strategy (ESTS)
Phase 3 implements the Error-Stratified Transformation Strategy (ESTS), which adapts repair behavior based on the
error category identified during Phase 1. ESTS integrates three mechanisms: (1) diagnostic classification into the four
categories defined in Section 3.3, (2) per-category temperature scheduling that maps each error stratum to an appropriate
level of sampling diversity, and (3) a Local-Loop stagnation escape that resets the repair trajectory when same-category
errors persist. The error category also drives template selection in Phase 2, so the ESTS classifier serves as the central
signal for both knowledge retrieval and sampling control. The per-category temperature schedule differs from the
static approaches (e.g., 𝑇= 0.0for greedy decoding or 𝑇= 1.0for default sampling) used in traditional code generation,
which cannot adapt to the semantic characteristics of different compiler signals.
Table 1 summarizes the Entropy-Rigidity mapping that underpins the temperature schedule. The effectiveness of ESTS
stems from the ordering of temperature tiers (low, moderate, and high entropy) rather than the exact parameter values.
The default parameter values ( 𝜃𝑆𝐿=0.1,𝜃𝑀𝑆=0.5,𝜃𝐿𝐵=1.2) are selected based on the directional entropy-rigidity
Manuscript submitted to ACM

12 Xiaofan Liu, Zhuang Zhao, Zecan Li, Ziqi Shuai, Yanming Yang, Qi Xin, and Jifeng Xuan
mapping and validated through a one-at-a-time sensitivity analysis (Section 5.3.1). This analysis provides diagnostic
support that the framework appears reasonably stable under moderate perturbations within each entropy tier.
•Syntax Precision (Low Entropy): Rigid, deterministic SL errors represent a point-to-point repair scenario
where the search space is minimal. The goal is to enforce exploitation and minimize invalid paths. We avoid a
purely greedy approach ( 𝜃=0.0) because zero-entropy sampling frequently traps the LLM in repetition loops. A
minimal stochasticity ( 𝜃𝑆𝐿=0.1) introduces just enough low-entropy noise to break these loops while preventing
deviations from the rigid syntactic constraints.
•Semantic Reasoning (Moderate Entropy): For semantic ownership and lifetime violations (MS), the system
adopts a balanced temperature. Resolving Rust ownership trees requires a balance between exploitation (retaining
existing logic) and exploration (restructuring the tree). Grounded in recent findings on Self-Consistency in
LLMs [ 43], a moderate entropy provides the model with a multi-path reasoning space to explore diverse resolution
strategies without fabricating nonexistent library functions.
•Algorithmic Exploration (High Entropy): Behavioral LB errors indicate that the algorithmic logic has diverged
from the source program. The current state is trapped in an incorrect attractor (local optimum). In the framework
of Search-Based Software Engineering (SBSE) theory [ 12], this high temperature functions analogously to a
genetic mutation operator that injectsstochastic perturbationinto the high-dimensional solution space. Such
high-entropy disruption is necessary to discard localized patches and escape the current attractor. Zhu et al. [ 56]
show that dynamically raised temperature for challenging code tokens expands the diversity of generated
candidates, which confirms the benefit of elevated entropy for difficult synthesis steps. Based on this finding,
we set a high-entropy threshold ( 𝜃𝐿𝐵=1.2) to trigger significant algorithmic shifts without loss of structural
coherence.
For instance, in our empirical evaluation (transforming a dynamic programming algorithm), the initial transformation
compiled successfully but panicked at runtime ( ParseIntError ) as shown previously in Fig. 4. A static, low-temperature
prompt repeatedly generated minor, ineffective edits that failed the exact same test case. A dynamic shift to high entropy
alongside a reflection prompt enabled the model to escape this local optimum. As demonstrated in Fig. 7, the high
entropy induced a complete refactoring of the input parsing mechanism and variable tracking logic. This refactored
solution ultimately passed the fuzzing suite.
The optimizer also incorporates aLocal-Loop Stagnation Escapemechanism to prevent infinite repair loops. When
the LLM repeatedly produces code that triggers the same error category for more than 𝑁𝐿consecutive iterations (i.e.,
𝜅𝑙𝑜𝑐𝑎𝑙>𝑁𝐿, default𝑁𝐿=3), the optimizer interprets this as a greedy deadlock. The local repair trajectory has converged
to a non-productive attractor. The system then fires a RESET that discards the failing Rust draft and re-derives a fresh
candidate from the C source at baseline temperature 𝑇= 1.0. The choice 𝑁𝐿=3is supported by an empirical sweet-spot
analysis and a threshold sensitivity study (Section 5). Both indicate that the conditional probability of subsequent
success drops sharply once the same error category repeats four or more times. The selected temperature and the
augmented prompt are passed to the LLM, which generates a new candidate program. This candidate re-enters Phase 1
for validation, which closes the iterative repair loop.
3.7 Algorithm Integration: The Closed-Loop Repair Cycle
Algorithm 1 presents the global execution flow of the AdaTrans framework, which integrates the Multi-Stage Validation
(Phase 1), the Strategy-Driven RAG (Phase 2), and the ESTS module (Phase 3).
Manuscript submitted to ACM

AdaTrans: Automated C-to-Rust Transformation via Error-Adaptive Repair 13
Fig. 7. High-entropy mutation escaping local optima by refactoring the parsing logic.
This algorithm serves as the central orchestrator. It highlights how the diagnostic signal 𝐸(𝑡)from the deterministic
oracle drives both the retrieval function 𝑓𝑟𝑒𝑡𝑟𝑖𝑒𝑣𝑒 and the entropy mapping F𝐸𝑆𝑇𝑆. The stagnation escape block (the
conditional check on 𝜅𝑙𝑜𝑐𝑎𝑙) implements the Local-Loop fallback mechanism, and ensures that the stochastic search
does not fall into an infinite deadlock when the same error category repeats.
4 Experimental Setup
4.1 Dataset Construction
The evaluation of C-to-Rust transformation tools in our file-level setting requires a dataset that satisfies three criteria.
First, the problems should reduce contamination risk with respect to the evaluated LLM snapshot. Second, the difficulty
spectrum should be sufficient to stress-test both syntactic and semantic transformation capabilities. Third, each
problem should have a machine-verifiable execution oracle for automated functional equivalence checking. Existing
resources [ 3,42] often rely on code snippets of uncertain provenance or problems that may overlap with LLM training
data. Such overlap makes it difficult to distinguish genuine transformation capability from memorized patterns.
We therefore constructed FuzzForLeetcode,1a dataset of 104 C programming problems sourced from LeetCode
Weekly Contests 413 through 438 (26 contests, 4 problems per contest). We selected contests released after Sep-
tember 2024, which postdates the knowledge cutoff (October 2023) of the LLM snapshot used in our experiments
(gpt-4o-mini-2024-07-18 ). This temporal separation mitigates, but does not eliminate, the risk that the evaluated
model has memorized similar solution patterns during pretraining. LeetCode Weekly Contests span a graduated difficulty
range from Easy to Hard and cover diverse algorithmic paradigms including dynamic programming, graph traversal,
combinatorial optimization, and string processing. While these tasks do not capture the full complexity of project-level
migration, they provide a controlled benchmark for studying file-level transformation of self-contained programs with
executable oracles.
The dataset construction followed a three-stage pipeline:
Stage 1: Problem acquisition.For each of the 104 problems, we downloaded the problem specification, including
input/output format, constraints, and examples. Where official C solutions were available from the LeetCode submission
1FuzzForLeetcode: https://github.com/SlainTroyard/FuzzForLeetcode_dev
Manuscript submitted to ACM

14 Xiaofan Liu, Zhuang Zhao, Zecan Li, Ziqi Shuai, Yanming Yang, Qi Xin, and Jifeng Xuan
Algorithm 1AdaTrans Iterative Repair with ESTS
Require:Source C code𝑆 𝐶; initial Rust program𝑆(0)
𝑅; max iterations𝑀
Ensure:Optimized Rust code𝑆∗
𝑅orFailure
1:𝑡←0;𝜅 𝑙𝑜𝑐𝑎𝑙←0
2:while𝑡<𝑀do
// Phase 1: Multi-Stage Validation
3:Status,𝐸(𝑡)←Validate(𝑆(𝑡)
𝑅)
4:ifStatus=Successthen
5:return𝑆(𝑡)
𝑅
6:end if
7:Category←Classify(𝐸(𝑡))∈{SL,MS,LB,AF}
// Update Local-Loop counter
8:if𝑡>0andCategory=Category(𝑡−1)then
9:𝜅 𝑙𝑜𝑐𝑎𝑙←𝜅𝑙𝑜𝑐𝑎𝑙+1
10:else
11:𝜅 𝑙𝑜𝑐𝑎𝑙←1
12:end if
// Local-Loop stagnation escape
13:if𝜅 𝑙𝑜𝑐𝑎𝑙>𝑁𝐿then
14:𝑇←1.0; Prompt←T 𝑅𝐸𝑆𝐸𝑇
15:𝜅 𝑙𝑜𝑐𝑎𝑙←0
16:else
// Phase 2: Strategy-Driven RAG
17:T 𝑖,K𝑖←𝑓𝑟𝑒𝑡𝑟𝑖𝑒𝑣𝑒(𝐸(𝑡))
18:Prompt←ConstructPrompt(𝑆(𝑡)
𝑅,T𝑖,K𝑖)
// Phase 3: ESTS temperature scheduling
19:𝑇←F 𝐸𝑆𝑇𝑆(Category)
20:end if
21:𝑆(𝑡+1)
𝑅←LLM.generate(Prompt, 𝑇)
22:𝑡←𝑡+1
23:end while
24:returnFailure
archive, we adopted them directly. For problems lacking C submissions, a researcher manually implemented a correct
C solution following the standard I/O interface ( scanf /printf ) and verified it by submission to the LeetCode online
judge. This manual intervention was necessary for approximately 30% of the problems.
Stage 2: Fuzzing infrastructure.For each problem, we developed a fuzzing script that encodes the input constraints
of each problem (e.g., array length bounds, value ranges, string character sets) and programmatically generates random
test cases. Fig. 8 illustrates the structure of such a script. Each script is manually crafted for the specific input structure
of its problem (e.g., arrays, graphs, trees, strings) and generates between 10 and 200 test cases depending on problem
complexity, with 100 cases for the majority of problems. The generated cases cover both typical inputs and boundary
conditions (e.g., minimum/maximum array lengths, extreme values). The script then compiles the C solution, feeds
each test case via stdin, and records the stdout output.
Manuscript submitted to ACM

AdaTrans: Automated C-to-Rust Transformation via Error-Adaptive Repair 15
Fig. 8. Example fuzzing script for test case generation.
Stage 3: Oracle construction.We executed each fuzzing script against its corresponding C solution to produce a
deterministic I/O oracle: a set of input–output pairs (𝑖,𝑜) where𝑜=Out(𝑆 𝐶,𝑖). This oracle serves as the ground truth
for evaluating functional equivalence of transformed Rust code. Any Rust program that produces matching outputs
for all test cases (after normalization) is considered functionally equivalent to the original C program. This evaluation
protocol follows the principle of differential testing [ 31]. The original C program and the transformed Rust program
are executed on the same randomly generated inputs, and output discrepancies indicate transformation errors. The
approach is analogous to how Csmith [ 52] [53] reveals compiler bugs by comparing the outputs of differently compiled
versions of the same program. In our setting, the C solution plays the role of the trusted reference against which the
Rust output is compared.
The resulting dataset comprises 104 problems across 26 weekly contests (4 problems per contest) and spans three
official LeetCode difficulty levels: 22 Easy, 48 Medium, and 34 Hard. The problems collectively cover a wide algorithmic
spectrum under a consistent evaluation protocol.
4.2 Baselines and Comparison Models
We compare AdaTrans against the following baseline methods and ablation variants:
•Zero-shot LLM: A baseline that transforms C code to Rust using gpt-4o-mini without any repair loop, iterative
feedback, or retrieval-augmented context. We sample 200 independent transformations per problem and report
pass@𝑘at𝑘∈{ 1,10,20,100}. The setting 𝑘=20matches the AdaTrans iteration budget 𝑀= 20. This enables a
budget-matched comparison between independent sampling and iterative repair.
•AdaTrans (full framework): Our complete system integrating Strategy-Driven RAG retrieval, ESTS (error
categorization, per-category temperature scheduling, and Local-Loop stagnation detection), and compiler-
feedback repair.
•AdaTrans w/o RAG: An ablation variant that retains ESTS (error categorization, per-category temperature
scheduling, and Local-Loop stagnation reset) but replaces every specialized repair template with a generic prompt
that contains the previous Rust draft and the compiler error only. The RVFG doc-tip injection is also disabled.
This variant is designed to assess the contribution of the Strategy-Driven RAG module.
Manuscript submitted to ACM

16 Xiaofan Liu, Zhuang Zhao, Zecan Li, Ziqi Shuai, Yanming Yang, Qi Xin, and Jifeng Xuan
•AdaTrans w/o ESTS: An ablation variant that retains RAG (specialized templates and doc-tip injection) but
removes the entire ESTS module, namely error categorization, per-category temperature scheduling, and Local-
Loop stagnation reset. All non-initial repair iterations are routed through a single uniform RAG-augmented
template at fixed temperature𝑇=1.0. This variant is designed to assess the contribution of the ESTS module.
•AdaTrans w/o ESTS & RAG: An ablation variant that removes both the ESTS and RAG modules and retains
only the iterative repair loop with compiler feedback. All non-initial repair iterations use a generic prompt at
fixed temperature 𝑇=1.0, without error categorization, temperature scheduling, stagnation reset, specialized
templates, or doc-tip injection. This variant also stops iterating as soon as the code compiles, without executing
test cases during the repair loop. Only the final output is validated against the fuzz oracle. This reduced variant
retains only the basic iterative compilation repair loop and serves as a baseline for the two core innovations of
AdaTrans.
We also evaluate three existing C-to-Rust transformation tools on the same 104-problem dataset:
•EvoC2Rust[ 42]: An LLM-based framework that translates C projects through a skeleton-guided strategy. It
first generates a compilable Rust skeleton of type-checked function stubs, then incrementally translates each
function into its stub, and finally repairs compilation errors by combining the LLM with static analysis. We
adapted the authors’ provided implementation for file-level evaluation by developing batch orchestration scripts
and configuring the LLM client for our evaluation infrastructure. The original implementation was designed
for multi-file C projects with build systems and header dependencies. Our adaptation applies its pipeline to
standalone single-file LeetCode solutions.
•Tymcrat[ 13]: An LLM-based type-migration tool that translates each C function into a Rust function with proper
Rust type signatures and iteratively repairs the resulting type errors using Rust compiler feedback. We adapted
the authors’ provided implementation for our dataset, which required resolving numerous crash-inducing edge
cases in the transformation engine (e.g., signature parsing panics, unhandled type transformation failures, and
keyword collisions with Rust identifiers).
•PtrTrans[ 54]: A project-level C-to-Rust transformation framework that constructs a C-Rust Pointer Knowledge
Graph to encode pointer ownership, mutability, nullability, and lifetime information. The enriched pointer
semantics are injected into LLM prompts to guide the generation of ownership-compliant Rust code. We adapted
the authors’ provided implementation for our file-level evaluation by packaging each standalone C solution as a
single-file project with the required build metadata.
All three tools primarily target project-level transformation. We nevertheless evaluate them under the same file-level
compilation and I/O consistency protocol used for AdaTrans so that all methods are compared on a common controlled
benchmark. This adaptation may disadvantage tools whose intended setting is multi-file migration, and we therefore
interpret the comparison as evidence within a shared file-level evaluation setup rather than as a definitive judgment of
each tool in its original deployment scenario. Additionally, we report unsafe code statistics for c2rust [ 14], a traditional
AST-based transpiler, on the same dataset.
LLM configuration.All LLM-based methods (AdaTrans, its ablation variants, and the zero-shot baseline) use the
same dated model snapshot, gpt-4o-mini-2024-07-18 [32] (knowledge cutoff: October 2023). We intentionally fix a
single backbone throughout the study to isolate the effect of the AdaTrans algorithmic components from differences
in underlying model capability. This design also makes full-dataset repeated runs, ablations, and sensitivity analyses
feasible within a reproducible evaluation budget. We therefore do not claim that the reported gains automatically transfer
Manuscript submitted to ACM

AdaTrans: Automated C-to-Rust Transformation via Error-Adaptive Repair 17
to all LLM backbones. The base sampling parameters are top_p= 1.0and max_tokens= 4096. The base temperature is
𝑇= 1.0. For AdaTrans and the w/oRAG ablation, the ESTS module overrides this value with per-category temperatures
(𝜃𝑆𝐿=0.1,𝜃𝑀𝑆=0.5,𝜃𝐿𝐵=1.2) as described in Section 3. The w/oESTS ablation and the zero-shot baseline use the
base temperature 𝑇= 1.0for all iterations. Full prompt templates and the error knowledge base are available in the
replication package [25].
4.3 Evaluation Metrics
We adopt the following metrics to evaluate the quality and safety of the transformed Rust code:
•Compilation Pass Rate: The percentage of the 104 problems for which the final output compiles successfully
without errors. For iterative methods (AdaTrans and its ablation variants), this is the compilation status of the
program returned by the optimizer at termination. A higher compilation pass rate indicates that the tool produces
syntactically and type-correct Rust code.
•Functional Equivalence (I/O Consistency) Rate: The percentage of problems where a compiled Rust program
produces output that exactly matches the expected output across all test cases in the evaluation oracle I. Output
comparison applies whitespace normalization (stripping leading/trailing whitespace and collapsing internal
whitespace) to account for benign formatting differences. All problems in our dataset produce exact integer or
string outputs, so floating-point tolerance and order-independent matching are not required. This metric serves
as the primary measure of transformation correctness.
•Unsafe Block Usage: We measure safety along two dimensions: (i) the unsafe filerate, defined as the percentage
of transformed files that contain at least one unsafe block, and (ii) the unsafe LOC rate, defined as the percentage
of total generated lines of code that reside withinunsafeblocks, aggregated across all output files per method.
These metrics quantify whether a transformation tool uses the Rust ownership system or falls back to unsafe
abstractions.
•pass@𝑘[3]: For the zero-shot LLM baseline, we report the unbiased estimator of the probability that at least
one of𝑘randomly drawn samples passes all test cases. Given 𝑛=200total samples per problem with 𝑐correct
samples, pass@ 𝑘is computed as pass@𝑘=1− 𝑛−𝑐
𝑘/ 𝑛
𝑘. This estimator follows the standard formulation from
Chen et al. [3].
5 Experimental Evaluation
This section presents the empirical evaluation of AdaTrans to answer the following Research Questions (RQs):
•RQ1 (Effectiveness): How does AdaTrans perform in terms of compilation pass rate and functional
equivalence compared to baseline transformation methods?To answer this, we evaluate AdaTrans against
existing C-to-Rust tools and a zero-shot LLM baseline on a 104-problem dataset. We then report both compilation
rates and fuzz-validated solve rates, supplementing these metrics with a difficulty-stratified evaluation and failure
mode analysis.
•RQ2 (Safety): To what extent does AdaTrans eliminate the reliance on unsafe blocks to ensure memory
safety in the transformed Rust code?We assess this by measuring the proportion of generated files that
containunsafe blocks and the proportion of total lines of code (LOC) that reside within them, comparing the
outputs of our framework against a traditional AST-level transpiler and the zero-shot baseline.
Manuscript submitted to ACM

18 Xiaofan Liu, Zhuang Zhao, Zecan Li, Ziqi Shuai, Yanming Yang, Qi Xin, and Jifeng Xuan
Table 2. Overall performance comparison on the 104-problem dataset. AdaTrans and its ablation variants report means across three
independent runs. pass@𝑘for𝑘∈{1,10,20,100}for the zero-shot LLM are estimated over 200 independent samples.
Method Compile Rate Solve Rate
c2rust [14] 96.15% 96.15%
EvoC2Rust [42] 25.00% 1.92%
Tymcrat [13] 81.73% 21.15%
PtrTrans [54] 94.23% 10.58%
Zero-shot LLM (pass@1) 57.00% 25.18%
Zero-shot LLM (pass@10) 90.78% 56.53%
Zero-shot LLM (pass@20) 94.67% 61.48%
Zero-shot LLM (pass@100) 98.98% 70.58%
AdaTrans w/o ESTS & RAG 96.79% 46.79%
AdaTrans w/o ESTS 97.76% 69.23%
AdaTrans w/o RAG 98.72% 71.15%
AdaTrans (Full) 95.51% 81.09%
•RQ3 (Ablation): What is the individual contribution of the Strategy-Driven RAG and the ESTS compo-
nents to the overall performance of the framework?We investigate this through a macro-level ablation
study by systematically disabling these components. Furthermore, we conduct micro-level sensitivity analyses
on key experimental configurations, specifically evaluating different per-category temperature settings and
local-loop stagnation reset thresholds.
•RQ4 (Convergence): How efficiently does the iterative repair cycle of AdaTrans converge to a valid
solution, and how does the error distribution evolve during this process?We track how the prevalence of
the four semantic error categories evolves across successive repair iterations at the dataset level. We also present
a detailed case study that traces the repair trajectory of a complex algorithm to illustrate how the adaptive
strategies drive convergence.
5.1 RQ1: Effectiveness Analysis
Table 2 presents the overall performance comparison between AdaTrans and the baseline methods on the 104-problem
dataset.
Across three independent runs, AdaTrans achieves a mean compile rate of 95.51% ±1.11% and a mean solve rate of
81.09%±3.09% (standard deviation), with individual run solve rates of 84.62%, 79.81%, and 78.85%. All solve rates in this
paper are validated against a fuzz oracle comprising 10–200 test cases per problem, generated independently of the
lightweight examples used for iteration feedback. The Zero-shot LLM pass@k metrics represent distinct usage scenarios.
The pass@1 value reflects single-shot performance (one API call, no retries) and approximates default practitioner
usage. The pass@10 and pass@20 values represent moderate resampling effort (10 and 20 independent attempts), and
pass@20 matches the AdaTrans iteration budget 𝑀= 20. The pass@100 value provides a high-budget reference point
for brute-force independent sampling under the same backbone. AdaTrans achieves its result through a single iterative
repair trajectory, a process distinct from brute-force independent sampling.
Compilation Pass Rate.Among existing tools, c2rust achieves a compile rate of 96.15% (100/104) when evaluated
on the nightly toolchain that its generated extern_types declarations require, followed by PtrTrans at 94.23% (98/104),
Tymcrat at 81.73% (85/104), and EvoC2Rust at 25.00% (26/104). The four remaining c2rust compilation failures occur
Manuscript submitted to ACM

AdaTrans: Automated C-to-Rust Transformation via Error-Adaptive Repair 19
because the C source uses GCC-extension nested function definitions (three problems) or variable-length array initializers
(one problem), constructs for which c2rust omits the affected function bodies entirely. The high compilation success of
PtrTrans stems from its SVF-based pointer analysis, which injects precise ownership, mutability, and lifetime annotations
into the LLM prompt. However, as discussed below, this compilation advantage does not extend to functional correctness.
The low compilation rate of EvoC2Rust stems from an architectural mismatch: it is designed for project-level C-to-Rust
transformation involving multi-file codebases with headers, build systems, and inter-module dependencies. Its pipeline
(extract metadata, generate Rust skeleton, incrementally transform functions, then repair) introduces failure points when
applied to standalone single-file programs. Specifically, EvoC2Rust generates code that relies on undefined convenience
macros such as c_qsort! ,c_malloc! ,c_sizeof! , andc_for! . These macros are part of its transformation utility
library but are not consistently included in the generated output. Its type mapping is also imprecise for isolated functions.
For example, it produces i64wherei32is expected, which causes systematic mismatched types errors. Its repair chain
(bracket fix→rule fix→LLM repair) was designed for large-project compilation errors rather than the systematic
type mismatches and missing definitions that arise from single-function transformation. We note that the reported
results (25.00% compile, 1.92% solve) are after adapting the EvoC2Rust infrastructure for our evaluation pipeline. The
original implementation produced even lower success rates due to hardcoded API configurations and missing batch
evaluation support.
AdaTrans achieves a mean compile rate of 95.51%, comparable to c2rust (96.15%) and surpassing Tymcrat by 13.78
percentage points and EvoC2Rust by 70.51 percentage points. The Zero-shot LLM pass@100 compilation estimate
reaches 98.98%, but this metric reflects the near-certainty of finding at least one compiling sample among 100 independent
draws rather than a per-attempt success rate. AdaTrans achieves this compilation rate through a single iterative repair
trajectory that systematically addresses compiler errors across multiple repair attempts. This approach avoids the cost
of generating and evaluating hundreds of candidate programs. The Strategy-Driven RAG module further contributes by
steering repairs toward functionally correct solutions (Section 5.3).
Functional Equivalence (Solve Rate).Under our file-level evaluation protocol, AdaTrans achieves a mean solve
rate of 81.09% and surpasses Tymcrat (21.15%) by 59.94 percentage points, PtrTrans (10.58%) by 70.51 percentage points,
and EvoC2Rust (1.92%) by 79.17 percentage points. c2rust achieves the highest solve rate among all tools at 96.15%
(100/104) because its deterministic AST transpilation preserves C semantics faithfully whenever the source compiles.
Every c2rust file that compiles also passes all fuzz tests, with a compile-to-solve gap of zero. However, this functional
fidelity comes at the cost of memory safety. As shown in Table 4, 100% of the c2rust output files contain unsafe
blocks (97.94% of generated LOC), effectively forfeiting the safety guarantees that motivate C-to-Rust migration. For
practitioners who prioritize correctness over safety, c2rust followed by manual unsafe cleanup represents a viable
alternative workflow. AdaTrans targets the complementary scenario where memory safety is a first-class requirement.
AdaTrans achieves a mean solve rate of 81.09% with a mean unsafe file rate of 1.19%, a result that shows that iterative
repair with semantic knowledge retrieval can achieve high functional equivalence rates while producing code that
satisfies the Rust ownership guarantees.
The large compile-to-solve gap of Tymcrat (81.73% compile vs. 21.15% solve) indicates that type-directed mutation
alone is insufficient for functional correctness. While Tymcrat can map C types to Rust equivalents well enough to
pass the compiler, it lacks a mechanism to verify and repair algorithmic behavior. PtrTrans exhibits an even wider gap
(94.23% compile vs. 10.58% solve), the largest among all tools. Although its SVF-based pointer analysis produces code
that satisfies Rust ownership constraints, the transformed programs diverge from the original C semantics in most
cases. Precise pointer ownership inference thus addresses the type-level challenge of C-to-Rust transformation but not
Manuscript submitted to ACM

20 Xiaofan Liu, Zhuang Zhao, Zecan Li, Ziqi Shuai, Yanming Yang, Qi Xin, and Jifeng Xuan
Table 3. Solve rates by problem difficulty. AdaTrans results are from one complete run. The pass@ 𝑘values are estimated from 200
independent samples.𝑛denotes the number of problems per difficulty level.
Difficulty𝑛AdaTrans pass@20 pass@100
Easy 22 95.45% 84.48% 94.55%
Medium 48 75.00% 57.72% 64.59%
Hard 34 91.18% 51.92% 63.52%
Overall 104 84.62% 61.48% 70.58%
the algorithmic fidelity required for functional equivalence. The low functional equivalence rate of EvoC2Rust (1.92%)
reflects the engineering challenges of adapting a project-level pipeline to file-level evaluation.
AdaTrans surpasses the Zero-shot LLM pass@1 rate (25.18%) by 55.91 percentage points. When the Zero-shot LLM is
allowed ten or twenty attempts (pass@10 = 56.53%, pass@20 = 61.48%), with pass@20 matching the AdaTrans iteration
budget𝑀= 20, AdaTrans still leads by 24.56 and 19.61 percentage points respectively. The single iterative repair
trajectory of AdaTrans also surpasses the Zero-shot LLM pass@100 estimate (70.58%) by 10.51 percentage points. This
result suggests that feedback-guided repair can be more effective than repeated independent sampling in this setting.
AdaTrans consumes an average of11,545 tokens per problemacross all 104 problems, while the zero-shot baseline
requires approximately 31,875 tokens per problem for 20 independent samples. AdaTrans thus achieves a higher mean
solve rate (81.09% vs. 61.48%) at roughlyone-third the token cost. Overall, AdaTrans processes the 104-problem
dataset at an estimated cost of $0.44 USD (629K prompt tokens + 571K completion tokens at gpt-4o-mini pricing).
This corresponds to $0.004 per problem and 5.47 iterations per problem on average.
The gap between compile rate and solve rate reveals the inherent difficulty of achieving functional equivalence in
C-to-Rust transformation. While syntactic and ownership violations can be systematically diagnosed and repaired
through compiler feedback, logic-behavioral errors, where the code compiles but produces incorrect results, require
deeper semantic understanding that AdaTrans addresses through its ESTS error categorization and adaptive repair
strategy.
Difficulty-Stratified Analysis.The following diagnostic analyses (difficulty stratification, failure analysis, sensitivity
studies, and convergence analysis) are based on one complete run. Table 3 stratifies solve rates by LeetCode difficulty
level for AdaTrans and the zero-shot LLM baseline.
AdaTrans outperforms pass@100 across all three difficulty levels, with the largest advantage on Hard problems
(91.18% vs. 63.52%, a 27.66 percentage-point gap). On Medium problems, AdaTrans achieves 75.00% versus 64.59% for
pass@100 (10.41 pp). The non-monotonic pattern (Hard problems yield a higher solve rate than Medium) indicates that
LeetCode difficulty, which reflects algorithmic complexity for human programmers, does not directly predict C-to-Rust
transformation difficulty. Transformation difficulty depends more on the prevalence of pointer-intensive idioms (e.g.,
linked lists, manual memory management) than on algorithmic sophistication.
Failure Analysis.We examine one complete run in detail to illustrate the distribution of failure modes. Of the 104
problems, 16 fail validation. These fall into two categories. Seven problems never pass even the lightweight example
tests during iteration, and nine problems pass examples but fail when validated against the full fuzz oracle.
The 7 unsolved problems exhibit two distinct failure modes by final compilation status. Four exhaust the 20-iteration
budget with a final compilation error, while three compile successfully but produce incorrect output. The ESTS diagnostic
classification labels each problem by the error category that drove the final repair attempt, which reflects the diagnostic
Manuscript submitted to ACM

AdaTrans: Automated C-to-Rust Transformation via Error-Adaptive Repair 21
Table 4. Safety comparison: unsafe block usage in transformed Rust code. The AdaTrans unsafe file rate is the three-run mean.
c2rust statistics are computed over 104 transformed files.
Method Unsafe File Rate Unsafe LOC Rate†
c2rust [14] 100.00% 97.94%
Zero-shot LLM (gpt-4o-mini) 1.05%<0.1%
AdaTrans (Full) 1.19%<0.1%
†Percentage of total generated LOC withinunsafeblocks, aggregated across all output files per method.
signal of the preceding iteration rather than the final compilation outcome. By this classification, logic-behavioral (LB)
errors dominate with 6 problems, followed by memory-semantic (MS, 1). Of the 6 LB-classified problems, 4 regressed to
compilation errors during their final repair attempt. The high-temperature exploration triggered by the LB classification
(𝜃𝐿𝐵=1.2) occasionally destabilizes previously working syntax while attempting algorithmic restructuring. The
prevalence of LB classifications indicates that algorithmic reconstruction, not syntax or ownership repair, is the primary
bottleneck for the remaining unsolved problems. All 4 compilation failures achieved successful compilation at earlier
iterations (compiling in 1–16 of 20 iterations) but regressed after high-temperature exploration or stagnation resets
redirected the search.
The 9 false-positive problems pass the 2–3 example test cases used for iteration feedback but fail on the full fuzz
oracle (10–200 test cases per problem). These failures indicate that the generated code overfits to the small example set
without capturing the full algorithmic specification. Increasing the number of iteration feedback test cases beyond the
current 2–3 examples would likely reduce the false positive rate, at the cost of increased execution time per iteration.
We leave the exploration of this trade-off to future work. The false positives concentrate at contest positions 2 and 4
(3 each), with 2 at position 3 and 1 at position 1, where positions 1–4 correspond to increasing difficulty within each
weekly contest (Section 4.1).
The difficulty-stratified analysis (Table 3) shows that Medium problems have the lowest solve rate among the three
difficulty levels, and suggests that Medium-difficulty algorithmic patterns pose the greatest challenge for C-to-Rust
transformation.
5.2 RQ2: Safety Analysis
A central promise of the Rust programming language is the guarantee of memory safety without runtime over-
head [ 29] [30]. However, this guarantee is contingent on the absence of unsafe blocks, which bypass the borrow
checker and reintroduce the memory vulnerabilities that Rust aims to eliminate [ 35]. We evaluate the safety of AdaTrans
output by measuring both the proportion of transformed files containing unsafe blocks and the proportion of generated
LOC residing withinunsafeblocks.
Table 4 presents the safety metrics for AdaTrans, the zero-shot LLM baseline, and c2rust.
AdaTrans introduces unsafe blocks in only 1.19% of successfully validated programs (three-run mean, with exactly
oneunsafe file per run), and less than 0.1% of total generated LOC resides inside unsafe blocks. The recurring unsafe
case arises in a problem whose original C solution relies on global mutable state (multiple statically allocated lookup
tables), and the LLM elected to mirror this pattern via static mut declarations rather than refactoring to interior-
mutability primitives such as LazyLock orOnceCell . The zero-shot LLM baseline shows a comparable unsafe file rate
(1.05%) and a similarly negligible unsafe LOC rate. By contrast, as an AST-level transpiler, c2rust preserves C semantics
Manuscript submitted to ACM

22 Xiaofan Liu, Zhuang Zhao, Zecan Li, Ziqi Shuai, Yanming Yang, Qi Xin, and Jifeng Xuan
Table 5. Ablation study: component-wise contribution analysis on the 104-problem dataset. All values are means across three
independent runs (±standard deviation).
Variant Compile Rate Solve RateΔSolve (pp)
AdaTrans w/o ESTS & RAG 96.79%±2.00 46.79%±3.89−34.30
AdaTrans w/o ESTS 97.76%±1.11 69.23%±7.51−11.86
AdaTrans w/o RAG 98.72%±1.47 71.15%±5.85−9.94
AdaTrans (Full) 95.51%±1.11 81.09%±3.09—
by wrapping nearly all transformed code in unsafe blocks (97.94% of generated LOC). This wrapping negates the
memory safety guarantees that motivate Rust adoption. c2rust also fails to produce compilable output for 4 of the
104 problems where the C source contains nested function definitions or variable-length array initializers with no
direct Rust equivalent. Even a single unsafe block can undermine the compiler’s ability to guarantee the absence of
use-after-free, double-free, and data-race bugs [35].
The near-elimination of unsafe code in AdaTrans is a direct consequence of the Strategy-Driven RAG mechanism.
When the LLM encounters memory-semantic (MS) violations, the RAG module retrieves ownership-aware repair
templates (T𝑀𝑆) that guide the model toward idiomatic safe Rust patterns, such as ownership scope restructuring,
reference counting ( Rc,Arc), or the Rust standard collection APIs, instead of unsafe escape hatches. These results
indicate that high transformation success rates and memory safety are largely compatible. By combining iterative repair
with semantic knowledge retrieval, the framework achieves a mean solve rate of 81.09% while keeping unsafe usage
two orders of magnitude below c2rust.
5.3 RQ3: Ablation Study
We conduct an ablation study to quantify the individual contribution of each core component, systematically removing
the Strategy-Driven RAG and the ESTS modules from the full AdaTrans framework. Table 5 presents the ablation
results.
A notable pattern in Table 5 is that the full framework has the lowest mean compile rate (95.51%) among all variants.
This reflects the cost of high-entropy LB exploration ( 𝜃𝐿𝐵=1.2): aggressive algorithmic restructuring occasionally
destabilizes previously compiling code, but the resulting solve-rate gain (up to 34.30 pp) far outweighs the modest
compile-rate reduction.
Impact of removing both components.The 𝑤/𝑜𝐸𝑆𝑇𝑆&𝑅𝐴𝐺 variant retains only the iterative repair loop with
compiler feedback: all errors use a generic prompt at fixed temperature 𝑇= 1.0, with no error categorization, no
stagnation reset, no specialized templates, and no doc-tip injection. This variant also stops iterating as soon as the
code compiles, without executing test cases during the repair loop, only the final output is validated against the
fuzz oracle. Across three runs, it achieves a mean solve rate of 46.79% ( ±3.89%), a 34.30 percentage-point drop from
the full framework. The large gap suggests that the ESTS and RAG components contribute substantially to the full
framework’s effectiveness beyond what the iterative repair loop alone provides. The 46.79% solve rate still substantially
underperforms the zero-shot LLM pass@20 estimate (61.48%). This result indicates that iterative compilation repair
without test-case feedback and adaptive strategies is less effective than independent sampling with equivalent budget.
Impact of Strategy-Driven RAG.Removing the RAG module ( 𝑤/𝑜𝑅𝐴𝐺 ) replaces every specialized repair template
with a generic prompt that exposes only the previous Rust draft and the compiler error, without doc-tip knowledge
Manuscript submitted to ACM

AdaTrans: Automated C-to-Rust Transformation via Error-Adaptive Repair 23
Table 6. Temperature sensitivity analysis on the full 104-problem C dataset (one complete run). Each variant overrides all three
category temperatures simultaneously. The Local-Loop threshold𝑁 𝐿=3is held constant.
Variant(𝜃 𝑆𝐿,𝜃𝑀𝑆,𝜃𝐿𝐵)SolveΔSolve (pp)
Default(0.1,0.5,1.2)84.62%—
Uniform-Moderate(0.5,0.5,0.5)69.23%−15.39
Uniform-High(1.2,1.2,1.2)74.04%−10.58
Reversed(1.2,0.5,0.1)67.31%−17.31
injection. The mean compile rate remains high at 98.72% ( ±1.47%), which indicates that the LLM can produce syntactically
valid Rust without targeted templates. The mean solve rate, however, falls from 81.09% to 71.15% ( ±5.85%), a 9.94
percentage-point reduction. This disparity reveals the distinct role of RAG. Raw compiler feedback suffices to drag
the program through Rust syntax and ownership checks, but the strategy-driven retrieval of error-category-specific
templates is essential for directing the LLM toward functionally correct algorithmic implementations. The RAG module
provides not only compilation guidance but also idiom-level repair knowledge.
Impact of ESTS.Removing the ESTS module ( 𝑤/𝑜𝐸𝑆𝑇𝑆 ) disables error categorization, the per-category temperature
schedule, and the Local-Loop stagnation reset. All errors are routed through a uniform RAG-augmented template at
fixed temperature 𝑇= 1.0. The mean compile rate remains high at 97.76% ( ±1.11%), but the mean solve rate drops from
81.09% to 69.23% (±7.51%), an 11.86 percentage-point reduction. Two observations emerge from this comparison. First,
ESTS is the larger contributor to functional correctness in our ablation, with a slightly greater drop than removing RAG.
Second, the high compile rate of the ablated variants indicates that without categorization, repair iterations gravitate
toward conservative, locally-correct edits that compile but fail to recover the intended algorithm.
Synergistic Effects.The two components are tightly coupled. ESTS supplies the error-category signal that selects
the appropriate RAG template. RAG supplies the structured repair guidance that turns the ESTS temperature schedule
into productive search. Removing either component reduces the mean solve rate by approximately 10–12 percentage
points, while removing both causes a 34.30 percentage-point drop, substantially larger than the sum of individual
removals would suggest if the components were independent. This super-additive degradation suggests that the two
modules reinforce each other. Adaptive temperature scheduling in ESTS is most effective when paired with matched
RAG templates, and vice versa. We note that part of the additional gap may also reflect the absence of test-case feedback
during the repair loop in the 𝑤/𝑜𝐸𝑆𝑇𝑆&𝑅𝐴𝐺 variant, which stops at first successful compilation rather than verifying
functional equivalence at each iteration. The synergy estimate is therefore an upper bound.
5.3.1 Temperature Sensitivity Analysis.A natural question is whether ESTS depends critically on the specific temperature
values (𝜃𝑆𝐿=0.1,𝜃𝑀𝑆=0.5,𝜃𝐿𝐵=1.2) or on the directional mapping itself, that is, low temperature for rigid syntax
fixes, moderate temperature for ownership reasoning, and high temperature for algorithmic exploration. We investigate
this by comparing the default configuration against three alternative settings on the full 104-problem dataset with
budget𝑀=20.
Discussion.Table 6 reveals two findings. First, the differentiated default (0.1,0.5,1.2)achieves the highest solve rate
among all tested variants. The advantage over Uniform-High is 10.58 percentage points, over Uniform-Moderate 15.39
percentage points, and over Reversed 17.31 percentage points. The consistent direction supports the entropy-rigidity
rationale. Syntax errors benefit from low-temperature determinism, while logic errors benefit from high-temperature
exploration. The Reversed configuration, which assigns high temperature to syntax errors and low temperature to logic
Manuscript submitted to ACM

24 Xiaofan Liu, Zhuang Zhao, Zecan Li, Ziqi Shuai, Yanming Yang, Qi Xin, and Jifeng Xuan
Table 7. Local-Loop threshold sensitivity (one complete run). Other ESTS settings (per-category temperatures) are held at their
defaults.
Variant𝑁 𝐿Compile Solve
Aggressive 1 86.54% 73.08%
Default 3 96.15% 84.62%
Lazy 5 91.35% 69.23%
Disabled∞85.58% 67.31%
errors, suffers the largest drop, which supports the conclusion that the mapping direction matters. Second, the ablation
study (Section 5.3) shows that the full ESTS module contributes an 11.86 percentage-point gain in mean solve rate
over the𝑤/𝑜𝐸𝑆𝑇𝑆 baseline. The temperature sensitivity analysis complements this finding. Differentiated per-category
temperatures outperform all uniform configurations tested and provide diagnostic support for the directional mapping
(low for syntax, high for logic) as a key driver within ESTS.
5.3.2 Local-Loop Threshold Sensitivity Analysis.ESTS triggers a RESET (fall back to a fresh transformation from the C
source) when the same error category repeats more than 𝑁𝐿times in succession. We justify the chosen value 𝑁𝐿=3by
evaluating four configurations on the full 104-problem dataset with budget𝑀=20.
Empirical sweet-spot.The value 𝑁𝐿=3is further justified by a post-hoc analysis of the AdaTrans repair logs.
For each iteration step, we compute the conditional probability that AdaTrans reaches a successful solution within
the next five iterations, conditioned on the current value of 𝜅𝑙𝑜𝑐𝑎𝑙. Fig. 9 shows the curve. The probability decreases
monotonically from 51.0% at𝜅 𝑙𝑜𝑐𝑎𝑙=1to 37.5% at𝜅 𝑙𝑜𝑐𝑎𝑙=3, then drops sharply to 9.1% at𝜅 𝑙𝑜𝑐𝑎𝑙=4(11 observations)
and 0.0% at𝜅𝑙𝑜𝑐𝑎𝑙=5(7 observations). The steep decline beyond 𝜅𝑙𝑜𝑐𝑎𝑙=3indicates that recovery probability drops by
a factor of four between the third and fourth consecutive same-category repetition. By 𝜅𝑙𝑜𝑐𝑎𝑙=5, no successful recovery
is observed. The marginal benefit of allowing a fourth same-category attempt (9.1%) is substantially outweighed by the
opportunity cost of consuming an iteration budget slot on a near-exhausted repair trajectory, which supports 𝑁𝐿=3as
the threshold.
1 2 3 4 50102030405060
RESET threshold 51
40.437.5
9.1
0−260
Consecutive same-category errors (𝜅 𝑙𝑜𝑐𝑎𝑙)𝑃(success in next 5 iters)(%)
Fig. 9. Sweet-spot analysis of the Local-Loop threshold from AdaTrans repair logs (434 observations across error-category iterations).
The conditional success probability decreases monotonically from 51.0% at 𝜅𝑙𝑜𝑐𝑎𝑙=1to 37.5% at 𝜅𝑙𝑜𝑐𝑎𝑙=3, then drops sharply to
9.1% at𝜅𝑙𝑜𝑐𝑎𝑙=4and 0.0% at𝜅 𝑙𝑜𝑐𝑎𝑙=5, which supports𝑁 𝐿=3as the threshold.
Manuscript submitted to ACM

AdaTrans: Automated C-to-Rust Transformation via Error-Adaptive Repair 25
Table 8. Distribution of Rust compiler error codes in the RVFG knowledge base across ESTS categories.
Category Count (%) Scope
SL (Syntactic-Linking) 301 (58.2%) Type system, module resolution, syntax
MS (Memory-Semantic) 105 (20.3%) Ownership, lifetime, borrow checker
LB (Logic-Behavioral) 1 (0.2%) Runtime logic (post-compilation)
AF (Ambiguous-Fallback) 9 (1.7%) External macros, crate interactions
Obsolete 101 (19.5%) Deprecated or version-specific codes
Discussion.The sensitivity table mirrors the sweet-spot analysis. 𝑁𝐿=3achieves the highest solve rate (84.62%).
Aggressive RESET ( 𝑁𝐿=1) loses 11.54 percentage points by interrupting still-productive repair sequences, while lazy
or disabled RESET ( 𝑁𝐿=5at 69.23%, 𝑁𝐿=∞ at 67.31%) wastes budget on saturated trajectories. The compile-rate
trade-off is gentler because abandoning a same-category repair sometimes regresses to a compile-error state on the
next iteration. The simplified single-trigger design is empirically motivated. An earlier version of AdaTrans included
secondary thresholds for “compilation hell” ( 𝑁𝐶) and “algorithmic deadlock” ( 𝑁𝐴). However, a log analysis of that
multi-trigger prototype shows that the Local-Loop trigger fires before either of the others in 95.9% of stagnation events
(118 of 123 trigger-condition evaluations across the full dataset). The two secondary triggers fire only 5 times combined.
Disabling them entirely yields the configuration adopted in this paper.
5.4 RQ4: Convergence and Error Repair Analysis
This RQ investigates the convergence behavior of the iterative repair cycle of AdaTrans and the evolution of error
categories across successive iterations. The analysis in this subsection is based on one complete run. We analyze the
error distribution through the lens of the four-category taxonomy defined in Section 3.3, namely Syntactic-Linking (SL),
Memory-Semantic (MS), Logic-Behavioral (LB), and Ambiguous-Fallback (AF).
5.4.1 Error Distribution in the Knowledge Base.The RVFG knowledge base encodes 517 distinct Rust compiler error
codes. Table 8 shows their distribution across the four ESTS categories.
SL errors dominate the knowledge base (58.2%), which reflects the syntactic rigidity of the Rust type system and
module resolution. The substantial MS proportion (20.3%) reflects the unique challenge of mapping manual memory
management in C to the Rust ownership model. LB errors are rare at the compiler level because logic defects manifest
only after successful compilation, during test execution.
5.4.2 Convergence Behavior.The iterative repair cycle of AdaTrans converges efficiently across the dataset. For the
100 problems that achieve compilation success, the majority require only a small number of iterations. Actual repair
trajectories are non-linear. Errors from different categories can interleave, recur after apparent resolution, or re-emerge
following stagnation resets (see Section 5.4.4). Even so, we observe three dominant patterns in how error categories
distribute across iterations (Fig. 10):
(1)All three major error categories co-occur from the first repair iteration.At iteration 1, LB errors account
for 42.3% of active problems (compilable code that fails test cases), MS errors for 38.5% (ownership and borrow-
checker violations), and SL errors for 19.2% (syntax and type mismatches). This early co-occurrence of diverse
error types motivates per-category temperature differentiation in ESTS rather than a one-size-fits-all repair
strategy.
Manuscript submitted to ACM

26 Xiaofan Liu, Zhuang Zhao, Zecan Li, Ziqi Shuai, Yanming Yang, Qi Xin, and Jifeng Xuan
(2)MS and SL errors decrease as repair progresses.MS errors decline steadily as the RAG module injects
ownership-reasoning templates ( T𝑀𝑆) at moderate entropy ( 𝜃𝑀𝑆=0.5). These templates guide the LLM away
fromunsafe escape hatches. SL errors resolve efficiently when they appear, aided by low-entropy deterministic
repair (𝜃𝑆𝐿=0.1) and syntax-correction templates (T 𝑆𝐿).
(3)LB errors are the most persistent and increasingly dominant.LB errors constitute 42.3% of active problems
at iteration 1 and rise to 85.7% by iteration 19. The root causes of these errors, such as algorithmic flaws, off-by-one
errors, and incorrect data structure choices, are decoupled from syntax and ownership. These errors constitute
the primary bottleneck for unsolved problems. The ESTS module activates high-entropy exploration ( 𝜃𝐿𝐵=1.2)
to escape local optima by generating structurally diverse alternatives. This mechanism functions as a mutation
operator in the search-based software engineering sense [12].
These patterns represent statistical tendencies across the dataset, not strict sequential phases within individual
trajectories. A single trajectory may revisit earlier error categories after a stagnation reset or high-temperature
exploration, as demonstrated by the C_422_4 case study (Section 5.4.4), whose repair path cycles through MS →
AF→LB×2→MS×3→RESET 1→SL×3→RESET 2→SUCCESS (where each RESET fires at the fourth consecutive
same-category occurrence, i.e., 𝜅𝑙𝑜𝑐𝑎𝑙=4>𝑁𝐿, and the×3 counts the stagnation iterations preceding the triggering
one).
1 3 5 7 9 11 13 15 17 19020406080104
Repair iterationNumber of problemsSolved SL MS
LB AF RESET
Fig. 10. Error category distribution across repair iterations (1–19) for all 104 problems. At each iteration, each problem is either Solved
(passed feedback tests at a prior iteration) or classified by its ESTS error category for the current repair attempt. RESET indicates
Local-Loop stagnation resets ( 𝜅𝑙𝑜𝑐𝑎𝑙 >𝑁𝐿). Twenty-five problems (24.0%) solve on the initial transformation (iteration 0, not shown).
5.4.3 Role of ESTS in Convergence.The ablation study (Section 5.3) confirms that removing ESTS causes the largest
single-component solve-rate drop ( −11.86 pp). The convergence perspective reveals why. Without error categorization,
the repair cycle oscillates between equivalent error states because uniform-temperature sampling fails to match the
exploration intensity required by different error categories. The Local-Loop stagnation escape ( 𝑁𝐿=3) further aids
convergence by triggering a context reset when the same category repeats. This reset functions as a random restart that
Manuscript submitted to ACM

AdaTrans: Automated C-to-Rust Transformation via Error-Adaptive Repair 27
prevents budget exhaustion on saturated trajectories (see sensitivity analysis in Section 5.3.2). Across the 104-problem
dataset, RESET events fire in 16 distinct problems (31 events total). Of these, 14 (87.5%) are ultimately solved. This
high recovery rate indicates that the stagnation escape mechanism effectively redirects the search toward productive
trajectories.
5.4.4 Case Study: Iterative Repair of Problem C_422_4.As a concrete illustration of the error-driven repair dynamics,
we trace the full transformation of Problem 4 from Weekly Contest 422. The C source implements a countBalanced
Permutations function that counts the distinct permutations of a digit string such that the sum of digits at even-indexed
positions equals the sum at odd-indexed positions, computed modulo109+7. The solution employs a memoized recursive
search over a three-dimensional state space (digit value, remaining sum, remaining count) with precomputed binomial
coefficients via Pascal’s triangle.
Baseline comparison on this problem.This problem is exceptionally difficult for zero-shot LLM transformation.
Across 200 independent samples, not a single transformation passes the test suite. The pass@1 through pass@100
estimates (following Chen et al. [ 3]) are all 0.0%, despite 42.0% of samples compiling successfully. The zero-shot LLM
can produce syntactically plausible Rust code for this problem but consistently fails to preserve the precise modular
arithmetic and combinatorial logic required for functional correctness. AdaTrans solves this problem in 14 iterations
(one initial generation plus 13 repair steps). The zero-shot approach fails on all 200 samples, yet the iterative repair
cycle converges to a correct solution. The following trace details how the framework converges through error-driven
adaptation. In the iteration headers below, the category label (e.g., MS, LB) denotes the ESTS error classification
that drove the repair attempt, derived from the diagnostic signal of the preceding iteration, and may differ from the
compilation outcome of the current attempt.
Iteration 0 (INIT, 𝜃=1.0).The initial zero-shot transformation mirrors the C global-variable pattern using Rust’s
static mutdeclarations, which requireunsafeblocks for every access:
static mutCNT: [usize; MAX_DIGITS] = [0; MAX_DIGITS];
static mutDP: [[[Option<i64>; MAX_COUNT]; MAX_SUM];
MAX_DIGITS] = [[[None; MAX_COUNT]; ...];
The compiler reports E0308 andE0277 : “cannot multiply i64byusize ”—a type mismatch between the i64accumulator
and theusize binomial coefficient array. This is classified as a Memory-Semantic (MS) error because it originates from
the interaction between Rust’s strict numeric type system and the ownership of mutable global state.
Iterations 1–4 (MS →AF→LB→LB).The ESTS module routes the MS error to 𝜃𝑀𝑆=0.5, and the RAG module
retrieves a type-repair template from T𝑀𝑆. Iteration 1 fails with the same type mismatch. At iteration 2, the error shifts
to an Ambiguous-Fallback (AF) classification ( 𝜃=1.0), and the code compiles but panics at runtime with “attempt to
subtract with overflow” in the recursive DP function. Iteration 3 produces a wrong answer (expected output 2, actual
output 0), classified as Logic-Behavioral (LB) at 𝜃𝐿𝐵=1.2. Iteration 4, still driven by the LB classification at 𝜃𝐿𝐵=1.2,
regresses to a compile error: the high-temperature exploration restructures the numeric types fromi64toisizeand
introduces newMul<usize>trait violations.
Iterations 5–7 (MS ×3, stagnation).Iterations 5–7 repeat the same MS-classified E0308 /E0277 compile error:
“cannot multiply isize byusize .” Despite RAG-guided ownership repair at 𝜃𝑀𝑆=0.5, the model cycles through
equivalent type-conversion attempts without resolving the underlying mismatch between signed and unsigned integer
types in the DP recurrence.
Manuscript submitted to ACM

28 Xiaofan Liu, Zhuang Zhao, Zecan Li, Ziqi Shuai, Yanming Yang, Qi Xin, and Jifeng Xuan
Iteration 8 (RESET 1,𝜃=1.0).Once the MS category persists beyond three consecutive repetitions ( 𝜅𝑙𝑜𝑐𝑎𝑙=4>
𝑁𝐿=3), the Local-Loop stagnation escape triggers a context reset. The system discards the accumulated repair context
and re-derives a fresh Rust transformation from the C source at baseline temperature 𝑇=1.0. The regenerated code
retainsstatic mut globals but shifts all arrays to i64, partially resolving the earlier type conflicts. A residual E0308
remains (“you can convert ausizeto ani64”).
Iterations 9–11 (SL ×3, stagnation).Iterations 9–11 attempt to fix the remaining type mismatch, now classified as
Syntactic-Linking (SL) at 𝜃𝑆𝐿=0.1. Low-temperature repairs produce minimal edits that fail to resolve the function
signature incompatibility and stagnate at the same error.
Iteration 12 (RESET 2,𝜃= 1.0).A second RESET fires once the SL category persists beyond three consecutive
repetitions (𝜅𝑙𝑜𝑐𝑎𝑙=4>𝑁𝐿). The fresh regeneration produces a restructured solution: static mut globals are eliminated
in favor of function parameters and heap-allocated Veccontainers, which removes all unsafe blocks. However, a new
E0308 error arises: “expected &[usize; 10] , found&Vec<usize> ”—a fixed-size vs. dynamic array mismatch in the
recursive function signature.
Iteration 13 (SUCCESS).The final iteration resolves the array type mismatch by unifying all container types as
Vec<usize> and restructuring the DP function to accept slice references. The solution passes all test cases with zero
unsafeblocks:
fndfs(i: usize, s: usize, c: usize,
left_s: &[usize], left_c: &[usize],
cnt: &[usize], cb: &[Vec<usize>],
dp: &mut Vec<Vec<Vec<Option<usize>>>>,
r1: &[usize]) -> usize { ... }
The transformed code replaces C-style global mutable state and pointer arithmetic with Rust-idiomatic parameter
passing,Vec-based dynamic allocation, andu128intermediate arithmetic for modular multiplication.
Key observations from this case study.(1) The problem is beyond zero-shot capability (pass@100 = 0.0% across
200 samples), yet AdaTrans solves it in 14 iterations through adaptive error-driven repair. (2) All four error categories
manifest in the repair trajectory (MS, AF, LB, SL), which exercises the full spectrum of adaptive temperature scheduling
in ESTS. (3) Two distinct RESET events occur: the first escapes MS-category stagnation (iterations 5–7), and the second
escapes SL-category stagnation (iterations 9–11). This confirms that the Local-Loop mechanism handles diverse error
types. (4) The second RESET (iteration 12) induces a qualitative architectural shift, namely a move from static mut
globals with pervasive unsafe blocks to safe, parameterized functions. This shift illustrates that stagnation escape can
trigger structural improvements beyond local repair. (5) The final solution eliminates all unsafe code and preserves the
modular arithmetic correctness of the original C implementation. This outcome shows that both safety and functional
equivalence are achievable on this combinatorially complex algorithm.
Comparative case: c2rust transpilation failures.Two further cases highlight the limitations of traditional
AST-level transpilation. In both, c2rust fails to produce valid Rust code, while AdaTrans generates safe, idiomatic
alternatives.
Problem 3 from Weekly Contest 418 ( constructGridLayout ) uses variable-length array (VLA) initializers, a C99
feature with no direct Rust equivalent:
intson[n][2] = {}, sou[n] = {};
Manuscript submitted to ACM

AdaTrans: Automated C-to-Rust Transformation via Error-Adaptive Repair 29
c2rust cannot represent this construct in Rust and omits the core constructGridLayout function entirely. It generates
only themain wrapper with raw libc pointers and unsafe blocks. AdaTrans, by contrast, transforms the VLA into
idiomatic Rust vectors:
let mutson:Vec<Vec<usize>> = vec![vec![0; 2]; n];
let mutsou:Vec<usize> = vec![0; n];
The resulting code uses safe slicing, standard I/O ( std::io ), and no unsafe blocks, while preserving the algorithm’s
correctness.
Problem 3 from Weekly Contest 430 ( numberOfSubsequences ) defines multiple helper functions ( hash ,hash_insert ,
hash_get ,gcd) as nested functions inside the main function, a GCC C extension not supported by standard C99. c2rust
reports “function definition is not allowed here” errors for each nested function and produces structurally invalid output.
AdaTrans successfully refactors the nested functions into top-level Rust functions and replaces the manual hash table
with Rust’sHashMap:
fngcd(muta: i32,mutb: i32) -> i32 {
whileb > 0 {lettemp = b; b = a % b; a = temp; }
a
}
fnnumber_of_subsequences(nums: &[i32]) -> i64 {
let muthash_table = HashMap::new();
// ...
}
These cases illustrate a key advantage of LLM-based transformation over deterministic AST rewriting. The LLM can
infer semantic intent from C idioms and re-express it with Rust-native abstractions, whereas the rigid syntax-level
mapping of c2rust fails when the source code relies on constructs that have no direct syntactic counterpart in Rust.
6 Threats to Validity
We organize potential threats to validity into external threats (factors that limit the generalizability of our findings) and
internal threats (factors that may affect the correctness of the experimental methodology).
6.1 External Validity
Dataset representativeness.Our evaluation uses 104 algorithmic problems from LeetCode Weekly Contests (Contests
413–438). These problems are self-contained, single-file programs with standard I/O interfaces. This scope may not
fully represent the challenges of systems programming or large-scale codebases with complex pointer arithmetic and
multi-file dependencies. We mitigate selection bias by sampling problems across 26 distinct contest sessions that cover
diverse algorithmic paradigms, including dynamic programming, graph traversal, and combinatorial optimization.
The file-level focus is deliberate, as it targets incremental migration, where Rust is gradually adopted within legacy C
environments [19].
Baseline tool adaptation.EvoC2Rust [ 42], Tymcrat [ 13], and PtrTrans [ 54] were designed for project-level or
function-level transformation. We adapted each tool for our file-level, single-file evaluation protocol. This adaptation may
disadvantage these tools relative to their intended use cases. We mitigated this concern by following the documentation
of each tool, resolving only the minimum set of integration issues required to run on our dataset, and applying the
same compilation and I/O consistency protocol to all methods.
Manuscript submitted to ACM

30 Xiaofan Liu, Zhuang Zhao, Zecan Li, Ziqi Shuai, Yanming Yang, Qi Xin, and Jifeng Xuan
LLM selection.All experiments employ gpt-4o-mini as the underlying LLM. The modular design of AdaTrans is
model-agnostic. However, performance may vary across LLM backends. We selected gpt-4o-mini for its balance of
cost efficiency and capability, which enabled the large number of experimental runs required for ablation and sensitivity
analyses. Commercial LLM providers may also silently update model weights behind a fixed API endpoint, which can
introduce run-to-run variance beyond sampling stochasticity. Pinning the dated snapshot ( gpt-4o-mini-2024-07-18 )
mitigates but cannot fully eliminate this risk. Future work should evaluate AdaTrans with other LLMs to assess
transferability.
Data contamination.LLM-based transformation results may be inflated if evaluation problems overlap with the
model’s pretraining data. We mitigate this risk by sourcing problems exclusively from LeetCode Weekly Contests
released after September 2024, which postdates the knowledge cutoff (October 2023) of gpt-4o-mini (Section 4.1).
While temporal separation cannot guarantee zero overlap, it provides a principled safeguard against memorization of
solution patterns.
6.2 Internal Validity
Evaluation oracle.We adopt I/O consistency over 10–200 fuzzing-generated test cases per problem (100 for the
majority) as the operational definition of functional equivalence. This choice follows established practice in the
code transformation literature [ 3,42]. While this test suite covers both typical and boundary-condition inputs, it
cannot guarantee equivalence on untested inputs. Our normalization operator Norm(·) mitigates benign formatting
discrepancies through whitespace collapsing. All outputs in our dataset are exact integers or strings, so only whitespace
normalization is applied.
Non-determinism.LLMs are inherently stochastic [ 43]. We quantify run-to-run variance by executing AdaTrans
three times on the full 104-problem dataset. The solve rates are 84.62%, 79.81%, and 78.85%. These runs yield a mean
of 81.09%±3.09% (standard deviation). The moderate variance indicates stable performance across runs. The core
components of AdaTrans, namely the deterministic RAG retrieval function 𝑓𝑟𝑒𝑡𝑟𝑖𝑒𝑣𝑒 and the ESTS mapping F𝐸𝑆𝑇𝑆, reduce
run-to-run variance compared to unconstrained LLM generation. The zero-shot LLM baseline mitigates non-determinism
through pass@𝑘estimation over 200 independent samples [3].
Temperature parameter sensitivity.The ESTS module relies on three temperature parameters ( 𝜃𝑆𝐿,𝜃𝑀𝑆,𝜃𝐿𝐵)
whose default values are selected based on the entropy-rigidity mapping. We do not claim these values are globally
optimal. A diagnostic sensitivity analysis on the full 104-problem dataset (Section 5.3.1) suggests that the framework
appears reasonably stable under moderate perturbations within each entropy tier and that the directional mapping (low
for syntax, moderate for ownership, high for logic) is a consistent factor in effectiveness.
7 Related Work
7.1 C-to-Rust Transformation Tools
Rule-based tools such as c2rust [ 14] transform C source code through AST-level rewriting. The output compiles but relies
heavily onunsafeconstructs. Extensive manual refactoring is therefore required to achieve safe Rust. Ling et al. [21]
proposed CRustS, which applies source-to-source transformation rules expressed in the TXL language [ 4] to reduce the
scope of unsafe expressions in c2rust output. Emre et al. [ 8] proposed Laertes, which lifts a subset of raw pointers in
c2rust output into safe Rust references by using the borrow checker as a lifetime oracle. Emre et al. [ 7] later examined
the limits of this strategy and found that imprecision in the Rust safety checker labels many safe pointer manipulations
Manuscript submitted to ACM

AdaTrans: Automated C-to-Rust Transformation via Error-Adaptive Repair 31
as potentially unsafe, which constrains the fraction of pointers that can be lifted into safe references. More recently,
LLM-based approaches have emerged. Wang et al. [ 42] introduced EvoC2Rust, which refines LLM-generated Rust
through an evolutionary, skeleton-guided strategy. Hong and Ryu [ 13] proposed Tymcrat, an LLM-based type-migration
tool that replaces C types with appropriate Rust types and iteratively fixes the resulting type errors using compiler
feedback. Yuan et al. [ 54] proposed PtrTrans, a framework that constructs a C-Rust Pointer Knowledge Graph to encode
pointer ownership and lifetime information. Luo et al. [ 26] proposed IRENE, which integrates rule-augmented retrieval,
structured code summarization, and error-driven repair to improve the safety and semantic consistency of LLM-based
C-to-Rust translation. Cai et al. [ 2] proposed RustMap, a project-scale approach that decomposes a C program into
dependency-guided units and uses input–output equivalence checks together with compiler and test feedback to refine
the translated Rust code. These approaches operate primarily at the project or function level. AdaTrans targets the
complementary file-level scenario.
7.2 LLM-Based Code Generation and Transformation
The Codex model [ 3] demonstrated that LLMs trained on large code corpora can solve non-trivial programming tasks
and established the basis for tools such as GitHub Copilot. Code Llama [ 37] and GPT-4 [ 33] further advanced the
state of the art in code generation and instruction following. Jiang et al. [ 15] provide a survey of LLM applications in
code generation. For cross-language transformation, Rozière et al. [ 36] proposed an unsupervised approach based on
back-transformation that does not require parallel corpora. Pre-trained code models such as CodeBERT [ 9], PLBART [ 1],
InCoder [ 10], CodeT5 [ 44], and GrammarT5 [ 55] provide strong code representations, yet they lack the compilation
feedback needed to reliably produce safe, compilable target code.
7.3 Neural and Cross-Language Code Translation
Cross-language code translation has been studied extensively beyond the C-to-Rust setting. Building on unsupervised
neural machine translation [ 36], Liu et al. [ 22] proposed SDA-Trans, which incorporates syntax structure and domain
knowledge to improve cross-lingual transfer, particularly for languages unseen during pre-training. Shi et al. [ 38]
introduced execution-result-based minimum Bayes risk decoding, which selects among candidate programs by ap-
proximating semantic equivalence through execution on a small number of test inputs. Rule-based and customizable
transpilers offer an alternative to purely neural translation. Wang et al. [ 40] proposed DuoGlot, which incrementally
constructs user-guided transpilation rules to translate Python to JavaScript with high accuracy. Doeraene [ 6] presented
Scala.js, a type-directed interoperability framework between statically typed Scala and dynamically typed JavaScript.
Sun et al. [ 39] applied context-aware code translation to the related task of code search by translating code snippets
into natural language descriptions.
Because automatically translated code frequently contains residual errors, several works focus on human collaboration
and error localization. Weisz et al. [ 45] studied how software engineers tolerate imperfect AI-generated translations
and identified interface features that aid error detection. Liu et al. [ 23] proposed hmCodeTrans, an interactive human-
machine method that feeds engineer edits back to the model for retranslation. Malyala et al. [ 28] analyzed common
failure patterns of unsupervised translators and combined rule-based pre- and post-processing with a neural model to
improve translation. Wang et al. [ 41] proposed TransMap, which pinpoints the location of semantic mistakes in neural
code translation to reduce debugging effort.
The evaluation of code translation has also received dedicated attention. Jiao et al. [ 16] developed a taxonomy
of translation tasks by complexity and knowledge dependence and showed that existing benchmarks are biased
Manuscript submitted to ACM

32 Xiaofan Liu, Zhuang Zhao, Zecan Li, Ziqi Shuai, Yanming Yang, Qi Xin, and Jifeng Xuan
toward trivial token-level translation. Macedo et al. [ 27] found that the output format of LLM-generated translations
substantially affects execution-based evaluation and proposed extraction methods to recover source code from model
responses. Pendyala and Thakur [ 34] presented an automated evaluation and explainability framework that combines
compilation and execution validation with token-level feature attribution across multiple code LLMs. These studies
inform our use of an execution-based oracle and exact output matching for evaluating functional equivalence (Section 4).
7.4 Retrieval-Augmented Generation for Code
Lewis et al. [ 18] introduced the RAG framework, which combines a pre-trained retriever with a generative model to
produce more factual and contextually grounded outputs. In the code domain, Xia et al. [ 46] demonstrated that large
pre-trained language models substantially improve automated program repair, especially when fix-template information
is incorporated. AdaTrans builds on this foundation with a retrieval module that maps compiler error signatures to
error-specific repair templates and official Rust documentation through a Rust Violation-Fix Graph.
7.5 Adaptive Strategies for LLM Generation
Wang et al. [ 43] introduced self-consistency decoding, which samples multiple reasoning paths and selects the most
consistent answer. Zhu et al. [ 56] showed that adaptive temperature control yields substantial gains over fixed-
temperature sampling in code generation. Harman et al. [ 12] survey search-based optimization for software engineering
and review how metaheuristic search can explore large solution spaces. Xuan et al. [ 49] applied a backbone-based
multilevel metaheuristic to the large-scale next release problem, illustrating how search scales to large software
optimization spaces. AdaTrans draws on these perspectives by treating temperature selection as a feedback-driven
search problem.
7.6 Automated Program Repair
Automated program repair (APR) generates patches for buggy programs by using test suites as correctness specifications.
Xuan et al. [ 50] proposed Nopol, which encodes conditional statement repair as a Satisfiability Modulo Theory (SMT)
problem and synthesizes patches from runtime traces. Xiong et al. [ 47] introduced ACS, which improves repair precision
through condition synthesis guided by code context and document analysis. Effective repair often depends on accurate
fault localization. Xuan and Monperrus [ 51] proposed test case purification, which decomposes failing tests to sharpen
spectrum-based fault localization. Gu et al. [ 11] predicted whether a crashing fault resides within the stack trace to
guide crash localization, and Liu et al. [ 24] studied nested exceptions in Java crash reports to support crash reproduction.
At the project level, Xuan et al. [ 48] applied data reduction techniques to improve the efficiency of bug triage. The
iterative repair cycle of AdaTrans shares the generate-and-validate paradigm with APR. Compiler diagnostics and test
feedback serve as the correctness specification, and the ESTS module adapts the repair strategy based on error category,
analogous to how APR techniques select repair templates based on fault localization.
7.7 Memory Safety in Rust
The Rust ownership and borrowing system provides compile-time memory safety guarantees without garbage collec-
tion [ 29] [30]. The formal soundness of this system has been verified through the RustBelt project [ 17]. Qin et al. [ 35]
studied how and why programmers write unsafe code in real-world Rust projects and found that unsafe is commonly
used to support low-level control that safe Rust does not permit. Cui et al. [ 5] further examined safety requirements
across unsafe API boundaries and identified a set of safety properties that programmers must satisfy. These findings
Manuscript submitted to ACM

AdaTrans: Automated C-to-Rust Transformation via Error-Adaptive Repair 33
highlight the challenge facing any C-to-Rust tool. The generated code must satisfy the Rust ownership rules rather
than circumvent them.
8 Conclusion
This paper presents AdaTrans, a framework for automated C-to-Rust code transformation that aligns LLM-generated
code with the ownership and memory safety guarantees of Rust. AdaTrans integrates three mechanisms. First, a
Strategy-Driven RAG mechanism maps compiler error signatures to targeted repair strategies through a Rust Violation-
Fix Graph. Second, an Error-Stratified Transformation Strategy (ESTS) classifies compiler diagnostics into semantic error
categories and adjusts the repair temperature per category to balance exploitation for syntax repairs with exploration for
algorithmic mutations. Third, a multi-stage validation pipeline enforces both compilability and functional equivalence
through a closed-loop generate-verify-repair cycle.
Our evaluation on 104 C algorithm problems demonstrates that AdaTrans achieves a mean compilation pass rate of
95.51% (±1.11%) and a mean solve rate of 81.09% ( ±3.09%) across three independent runs, validated against a fuzz oracle
of 10–200 test cases per problem, with a mean unsafe file rate of 1.19%. AdaTrans surpasses the pass@100 estimate
(70.58%), a high-budget reference point for brute-force independent sampling under the same backbone, through a
single iterative repair trajectory of at most 20 iterations. Ablation studies confirm the individual contributions of both
the RAG and ESTS components. Removing ESTS reduces the mean solve rate to 69.23%, removing RAG to 71.15%, and
removing both to 46.79%.
Several directions remain for future investigation. First, the current file-level transformation scope should be extended
to support project-level migration, where multi-file dependencies, build system integration, and cross-module type
sharing introduce additional complexities. Second, evaluating AdaTrans with stronger and more diverse LLM backends
(beyondgpt-4o-mini ) would provide deeper insights into the model-agnostic properties of the framework and its
scalability to frontier models. Third, expanding the error taxonomy and RAG knowledge base to cover a broader range
of C constructs, including POSIX system calls, hardware-specific intrinsics, and third-party library bindings, would
enhance applicability to real-world systems programming scenarios. Finally, integrating formal verification techniques
alongside the current testing-based validation could provide stronger guarantees of semantic equivalence and move
beyond I/O consistency toward provable correctness.
Declarations
Conflict of Interest
The authors declare that they have no competing interests.
Funding
This work was supported by [FUNDING INFORMATION REMOVED FOR REVIEW].
Data Availability
All data, code, and experimental artifacts are available in the replication package (see Section 4).
Manuscript submitted to ACM

34 Xiaofan Liu, Zhuang Zhao, Zecan Li, Ziqi Shuai, Yanming Yang, Qi Xin, and Jifeng Xuan
Author Contributions
Xiaofan Liu conceived and designed the study, performed the experiments, and wrote the original draft. Zecan Li
assisted with data collection. Zhuang Zhao participated in the discussion of the ideas and the revision of the manuscript.
Ziqi Shuai assisted with the experimental infrastructure. Qi Xin participated in the discussion of the ideas and the
revision of the manuscript. Jifeng Xuan supervised the project and contributed to the review and editing of the paper.
All authors read and approved the final manuscript.
Acknowledgments
References
[1]Wasi Ahmad, Saikat Chakraborty, Baishakhi Ray, and Kai-Wei Chang. 2021. Unified Pre-training for Program Understanding and Genera-
tion. In Proceedings ofthe2021 Conference oftheNorth American Chapter oftheAssociation forComputational Linguistics: Human Language
Technologies , Kristina Toutanova, Anna Rumshisky, Luke Zettlemoyer, Dilek Hakkani-Tur, Iz Beltagy, Steven Bethard, Ryan Cotterell, Tanmoy
Chakraborty, and Yichao Zhou (Eds.). Association for Computational Linguistics, Online, 2655–2668. doi:10.18653/v1/2021.naacl-main.211
[2]Xuemeng Cai, Jiakun Liu, Xiping Huang, Yijun Yu, Haitao Wu, Chunmiao Li, Bo Wang, Imam Nur Bani Yusuf, and Lingxiao Jiang. 2025. RustMap:
Towards Project-Scale C-to-Rust Migration via Program Analysis and LLM. In Engineering ofComplex Computer Systems: 29th International
Conference, ICECCS 2025, Hangzhou, China, July 2–4, 2025, Proceedings (Hangzhou, China). Springer-Verlag, Berlin, Heidelberg, 283–302. doi:10.
1007/978-3-032-00828-2_16
[3]Mark Chen, Jerry Tworek, Heewoo Jun, Qiming Yuan, Henrique Ponde de Oliveira Pinto, Jared Kaplan, Harri Edwards, Yuri Burda, Nicholas Joseph,
Greg Brockman, Alex Ray, Raul Puri, Gretchen Krueger, Michael Petrov, Heidy Khlaaf, Girish Sastry, Pamela Mishkin, Brooke Chan, Scott Gray,
Nick Ryder, Mikhail Pavlov, Alethea Power, Lukasz Kaiser, Mohammad Bavarian, Clemens Winter, Philippe Tillet, Felipe Petroski Such, Dave
Cummings, Matthias Plappert, Fotios Chantzis, Elizabeth Barnes, Ariel Herbert-Voss, William Hebgen Guss, Alex Nichol, Alex Paino, Nikolas Tezak,
Jie Tang, Igor Babuschkin, Suchir Balaji, Shantanu Jain, William Saunders, Christopher Hesse, Andrew N. Carr, Jan Leike, Josh Achiam, Vedant
Misra, Evan Morikawa, Alec Radford, Matthew Knight, Miles Brundage, Mira Murati, Katie Mayer, Peter Welinder, Bob McGrew, Dario Amodei,
Sam McCandlish, Ilya Sutskever, and Wojciech Zaremba. 2021. Evaluating Large Language Models Trained on Code. arXiv:2107.03374 [cs.LG]
https://arxiv.org/abs/2107.03374
[4]James R. Cordy. 2006. The TXL source transformation language. Science ofComputer Programming 61, 3 (2006), 190–210. doi:10.1016/j.scico.2006.
04.002 Special Issue on The Fourth Workshop on Language Descriptions, Tools, and Applications (LDTA ’04).
[5]Mohan Cui, Shuran Sun, Hui Xu, and Yangfan Zhou. 2024. Is unsafe an Achilles’ Heel? A Comprehensive Study of Safety Requirements in Unsafe Rust
Programming. In Proceedings oftheIEEE/ACM 46th International Conference onSoftware Engineering (Lisbon, Portugal) (ICSE ’24) . Association
for Computing Machinery, New York, NY, USA, Article 106, 13 pages. doi:10.1145/3597503.3639136
[6]Sébastien Doeraene. 2013. Scala.js: Type-Directed Interoperability with Dynamically Typed Languages. https://infoscience.epfl.ch/handle/20.500.
14299/97425
[7]Mehmet Emre, Peter Boyland, Aesha Parekh, Ryan Schroeder, Kyle Dewey, and Ben Hardekopf. 2023. Aliasing Limits on Translating C to Safe Rust.
Proc. ACM Program. Lang. 7, OOPSLA1, Article 94 (April 2023), 29 pages. doi:10.1145/3586046
[8]Mehmet Emre, Ryan Schroeder, Kyle Dewey, and Ben Hardekopf. 2021. Translating C to safer Rust. Proc. ACM Program. Lang. 5, OOPSLA, Article
121 (Oct. 2021), 29 pages. doi:10.1145/3485498
[9]Zhangyin Feng, Daya Guo, Duyu Tang, Nan Duan, Xiaocheng Feng, Ming Gong, Linjun Shou, Bing Qin, Ting Liu, Daxin Jiang, and Ming Zhou. 2020.
CodeBERT: A Pre-Trained Model for Programming and Natural Languages. In Findings oftheAssociation forComputational Linguistics: EMNLP
2020, Trevor Cohn, Yulan He, and Yang Liu (Eds.). Association for Computational Linguistics, Online, 1536–1547. doi:10.18653/v1/2020.findings-
emnlp.139
[10] Daniel Fried, Armen Aghajanyan, Jessy Lin, Sida Wang, Eric Wallace, Freda Shi, Ruiqi Zhong, Scott Yih, Luke Zettlemoyer, and Mike Lewis. 2023.
InCoder: A Generative Model for Code Infilling and Synthesis. In TheEleventh International Conference onLearning Representations, ICLR 2023,
Kigali, Rwanda, May 1-5,2023. OpenReview.net, Kigali, Rwanda, 26 pages. https://openreview.net/forum?id=hQwb-lbM6EL
[11] Yongfeng Gu, Jifeng Xuan, Hongyu Zhang, Lanxin Zhang, Qingna Fan, Xiaoyuan Xie, and Tieyun Qian. 2019. Does the fault reside in a stack trace?
Assisting crash localization by predicting crashing fault residence. Journal ofSystems andSoftware 148 (2019), 88–104. doi:10.1016/j.jss.2018.11.004
[12] Mark Harman, S. Afshin Mansouri, and Yuanyuan Zhang. 2012. Search-based software engineering: Trends, techniques and applications. ACM
Comput. Surv. 45, 1, Article 11 (Dec. 2012), 61 pages. doi:10.1145/2379776.2379787
[13] Jaemin Hong and Sukyoung Ryu. 2025. Type-migrating C-to-Rust translation using a large language model. Empir. Softw. Eng. 30, 1 (2025), 3.
doi:10.1007/S10664-024-10573-2
[14] Immunant, Inc. and Galois, Inc. 2018. C2Rust: Migrating C Code to Rust. https://github.com/immunant/c2rust. Open-source transpiler. Accessed:
2025.
Manuscript submitted to ACM

AdaTrans: Automated C-to-Rust Transformation via Error-Adaptive Repair 35
[15] Juyong Jiang, Fan Wang, Jiasi Shen, Sungju Kim, and Sung Hun Kim. 2026. A Survey on Large Language Models for Code Generation. ACM Trans.
Softw. Eng. Methodol. 35, 2 (2026), 58:1–58:72. doi:10.1145/3747588
[16] Mingsheng Jiao, Tingrui Yu, Xuan Li, Guanjie Qiu, Xiaodong Gu, and Beijun Shen. 2024. On the Evaluation of Neural Code Translation: Taxonomy
and Benchmark. In Proceedings ofthe38th IEEE/ACM International Conference onAutomated Software Engineering (Echternach, Luxembourg)
(ASE ’23). IEEE Press, Piscataway, NJ, USA, 1529–1541. doi:10.1109/ASE56229.2023.00114
[17] Ralf Jung, Jacques-Henri Jourdan, Robbert Krebbers, and Derek Dreyer. 2017. RustBelt: securing the foundations of the Rust programming language.
Proc. ACM Program. Lang. 2, POPL, Article 66 (Dec. 2017), 34 pages. doi:10.1145/3158154
[18] Patrick Lewis, Ethan Perez, Aleksandra Piktus, Fabio Petroni, Vladimir Karpukhin, Naman Goyal, Heinrich Küttler, Mike Lewis, Wen-tau Yih, Tim
Rocktäschel, Sebastian Riedel, and Douwe Kiela. 2020. Retrieval-augmented generation for knowledge-intensive NLP tasks. In Proceedings ofthe
34th International Conference onNeural Information Processing Systems (Vancouver, BC, Canada) (NIPS ’20) . Curran Associates Inc., Red Hook,
NY, USA, Article 793, 16 pages.
[19] Hongyu Li, Liwei Guo, Yexuan Yang, Shangguang Wang, and Mengwei Xu. 2024. An empirical study of rust-for-Linux: the success, dissatisfaction,
and compromise. In Proceedings ofthe2024 USENIX Conference onUsenix Annual Technical Conference (Santa Clara, CA, USA) (USENIX ATC’24) .
USENIX Association, USA, Article 27, 19 pages.
[20] Ruishi Li, Bo Wang, Tianyu Li, Prateek Saxena, and Ashish Kundu. 2025. Translating C To Rust: Lessons from a User Study. In 32nd Annual Network
andDistributed System Security Symposium, NDSS 2025, SanDiego, California, USA, February 24-28, 2025 . The Internet Society, Reston, VA, USA,
18 pages. https://www.ndss-symposium.org/ndss-paper/translating-c-to-rust-lessons-from-a-user-study/
[21] Michael Ling, Yijun Yu, Haitao Wu, Yuan Wang, James R. Cordy, and Ahmed E. Hassan. 2022. In rust we trust: a transpiler from unsafe C to safer
rust. In Proceedings oftheACM/IEEE 44th International Conference onSoftware Engineering: Companion Proceedings (Pittsburgh, Pennsylvania)
(ICSE ’22). Association for Computing Machinery, New York, NY, USA, 354–355. doi:10.1145/3510454.3528640
[22] Fang Liu, Jia Li, and Li Zhang. 2023. Syntax and Domain Aware Model for Unsupervised Program Translation. In Proceedings ofthe45th International
Conference onSoftware Engineering (Melbourne, Victoria, Australia) (ICSE ’23) . IEEE Press, Piscataway, NJ, USA, 755–767. doi:10.1109/ICSE48619.
2023.00072
[23] Jiaqi Liu, Fengming Zhang, Xin Zhang, Zhiwen Yu, Liang Wang, Yao Zhang, and Bin Guo. 2024. hmCodeTrans: Human–Machine Interactive Code
Translation. IEEE Trans. Softw. Eng. 50, 5 (May 2024), 1163–1181. doi:10.1109/TSE.2024.3379583
[24] Shaoting Liu, Haiyan Xu, Qi Xin, and Jifeng Xuan. 2026. Exceptions Can Be Nested: An Exploratory Study on Nested Exceptions in Java Crashes.
Journal ofSoftware: Evolution andProcess 38, 4 (2026), e70099. arXiv:https://onlinelibrary.wiley.com/doi/pdf/10.1002/smr.70099 doi:10.1002/smr.
70099
[25] Xiaofan Liu, Zecan Li, Zhuang Zhao, Ziqi Shuai, and Jifeng Xuan. 2025. AdaTrans Replication Package. https://github.com/SlainTroyard/adatrans_dev.
Source code, prompts, knowledge base, and experimental data..
[26] Feng Luo, Kexing Ji, Cuiyun Gao, Shuzheng Gao, Jia Feng, Kui Liu, Xin Xia, and Michael R. Lyu. 2025. Integrating Rules and Semantics for
LLM-Based C-to-Rust Translation . In 2025 IEEE International Conference onSoftware Maintenance andEvolution (ICSME) . IEEE Computer Society,
Los Alamitos, CA, USA, 685–696. doi:10.1109/ICSME64153.2025.00069
[27] Marcos Macedo, Yuan Tian, Filipe Cogo, and Bram Adams. 2024. Exploring the Impact of the Output Format on the Evaluation of Large
Language Models for Code Translation. In Proceedings ofthe2024 IEEE/ACM First International Conference onAIFoundation Models andSoftware
Engineering (Lisbon, Portugal) (FORGE ’24). Association for Computing Machinery, New York, NY, USA, 57–68. doi:10.1145/3650105.3652301
[28] Aniketh Malyala, Katelyn Zhou, Baishakhi Ray, and Saikat Chakraborty. 2023. On ML-Based Program Translation: Perils and Promises. In Proceedings
ofthe45th International Conference onSoftware Engineering: New Ideas andEmerging Results (Melbourne, Australia) (ICSE-NIER ’23) . IEEE Press,
Piscataway, NJ, USA, 60–65. doi:10.1109/ICSE-NIER58687.2023.00017
[29] Nicholas D. Matsakis and Felix S. Klock. 2014. The rust language. In Proceedings ofthe2014 ACM SIGAda Annual Conference onHigh Integrity
Language Technology (Portland, Oregon, USA) (HILT ’14) . Association for Computing Machinery, New York, NY, USA, 103–104. doi:10.1145/
2663171.2663188
[30] Nicholas D. Matsakis and Felix S. Klock. 2014. The rust language. Ada Lett. 34, 3 (Oct. 2014), 103–104. doi:10.1145/2692956.2663188
[31] William M. McKeeman. 1998. Differential Testing for Software. Digital Technical Journal 10, 1 (1998), 100–107.
[32] OpenAI. 2024. Models - GPT-4o-mini. https://platform.openai.com/docs/models/gpt-4o-mini. Accessed: 2026-06-08.
[33] OpenAI, Josh Achiam, Steven Adler, Sandhini Agarwal, Lama Ahmad, Ilge Akkaya, Florencia Leoni Aleman, Diogo Almeida, Janko Altenschmidt,
Sam Altman, Shyamal Anadkat, Red Avila, Igor Babuschkin, Suchir Balaji, Valerie Balcom, Paul Baltescu, Haiming Bao, Mohammad Bavarian, Jeff
Belgum, Irwan Bello, Jake Berdine, Gabriel Bernadett-Shapiro, Christopher Berner, Lenny Bogdonoff, Oleg Boiko, Madelaine Boyd, Anna-Luisa
Brakman, Greg Brockman, Tim Brooks, Miles Brundage, Kevin Button, Trevor Cai, Rosie Campbell, Andrew Cann, Brittany Carey, Chelsea Carlson,
Rory Carmichael, Brooke Chan, Che Chang, Fotis Chantzis, Derek Chen, Sully Chen, Ruby Chen, Jason Chen, Mark Chen, Ben Chess, Chester
Cho, Casey Chu, Hyung Won Chung, Dave Cummings, Jeremiah Currier, Yunxing Dai, Cory Decareaux, Thomas Degry, Noah Deutsch, Damien
Deville, Arka Dhar, David Dohan, Steve Dowling, Sheila Dunning, Adrien Ecoffet, Atty Eleti, Tyna Eloundou, David Farhi, Liam Fedus, Niko Felix,
Simón Posada Fishman, Juston Forte, Isabella Fulford, Leo Gao, Elie Georges, Christian Gibson, Vik Goel, Tarun Gogineni, Gabriel Goh, Rapha
Gontijo-Lopes, Jonathan Gordon, Morgan Grafstein, Scott Gray, Ryan Greene, Joshua Gross, Shixiang Shane Gu, Yufei Guo, Chris Hallacy, Jesse
Han, Jeff Harris, Yuchen He, Mike Heaton, Johannes Heidecke, Chris Hesse, Alan Hickey, Wade Hickey, Peter Hoeschele, Brandon Houghton,
Kenny Hsu, Shengli Hu, Xin Hu, Joost Huizinga, Shantanu Jain, Shawn Jain, Joanne Jang, Angela Jiang, Roger Jiang, Haozhun Jin, Denny Jin, Shino
Manuscript submitted to ACM

36 Xiaofan Liu, Zhuang Zhao, Zecan Li, Ziqi Shuai, Yanming Yang, Qi Xin, and Jifeng Xuan
Jomoto, Billie Jonn, Heewoo Jun, Tomer Kaftan, Łukasz Kaiser, Ali Kamali, Ingmar Kanitscheider, Nitish Shirish Keskar, Tabarak Khan, Logan
Kilpatrick, Jong Wook Kim, Christina Kim, Yongjik Kim, Jan Hendrik Kirchner, Jamie Kiros, Matt Knight, Daniel Kokotajlo, Łukasz Kondraciuk,
Andrew Kondrich, Aris Konstantinidis, Kyle Kosic, Gretchen Krueger, Vishal Kuo, Michael Lampe, Ikai Lan, Teddy Lee, Jan Leike, Jade Leung,
Daniel Levy, Chak Ming Li, Rachel Lim, Molly Lin, Stephanie Lin, Mateusz Litwin, Theresa Lopez, Ryan Lowe, Patricia Lue, Anna Makanju, Kim
Malfacini, Sam Manning, Todor Markov, Yaniv Markovski, Bianca Martin, Katie Mayer, Andrew Mayne, Bob McGrew, Scott Mayer McKinney,
Christine McLeavey, Paul McMillan, Jake McNeil, David Medina, Aalok Mehta, Jacob Menick, Luke Metz, Andrey Mishchenko, Pamela Mishkin,
Vinnie Monaco, Evan Morikawa, Daniel Mossing, Tong Mu, Mira Murati, Oleg Murk, David Mély, Ashvin Nair, Reiichiro Nakano, Rajeev Nayak,
Arvind Neelakantan, Richard Ngo, Hyeonwoo Noh, Long Ouyang, Cullen O’Keefe, Jakub Pachocki, Alex Paino, Joe Palermo, Ashley Pantuliano,
Giambattista Parascandolo, Joel Parish, Emy Parparita, Alex Passos, Mikhail Pavlov, Andrew Peng, Adam Perelman, Filipe de Avila Belbute Peres,
Michael Petrov, Henrique Ponde de Oliveira Pinto, Michael, Pokorny, Michelle Pokrass, Vitchyr H. Pong, Tolly Powell, Alethea Power, Boris Power,
Elizabeth Proehl, Raul Puri, Alec Radford, Jack Rae, Aditya Ramesh, Cameron Raymond, Francis Real, Kendra Rimbach, Carl Ross, Bob Rotsted, Henri
Roussez, Nick Ryder, Mario Saltarelli, Ted Sanders, Shibani Santurkar, Girish Sastry, Heather Schmidt, David Schnurr, John Schulman, Daniel Selsam,
Kyla Sheppard, Toki Sherbakov, Jessica Shieh, Sarah Shoker, Pranav Shyam, Szymon Sidor, Eric Sigler, Maddie Simens, Jordan Sitkin, Katarina Slama,
Ian Sohl, Benjamin Sokolowsky, Yang Song, Natalie Staudacher, Felipe Petroski Such, Natalie Summers, Ilya Sutskever, Jie Tang, Nikolas Tezak,
Madeleine B. Thompson, Phil Tillet, Amin Tootoonchian, Elizabeth Tseng, Preston Tuggle, Nick Turley, Jerry Tworek, Juan Felipe Cerón Uribe,
Andrea Vallone, Arun Vijayvergiya, Chelsea Voss, Carroll Wainwright, Justin Jay Wang, Alvin Wang, Ben Wang, Jonathan Ward, Jason Wei, CJ
Weinmann, Akila Welihinda, Peter Welinder, Jiayi Weng, Lilian Weng, Matt Wiethoff, Dave Willner, Clemens Winter, Samuel Wolrich, Hannah
Wong, Lauren Workman, Sherwin Wu, Jeff Wu, Michael Wu, Kai Xiao, Tao Xu, Sarah Yoo, Kevin Yu, Qiming Yuan, Wojciech Zaremba, Rowan
Zellers, Chong Zhang, Marvin Zhang, Shengjia Zhao, Tianhao Zheng, Juntang Zhuang, William Zhuk, and Barret Zoph. 2024. GPT-4 Technical
Report. arXiv:2303.08774 [cs.CL] https://arxiv.org/abs/2303.08774
[34] Vishnu S. Pendyala and Neha Bais Thakur. 2026. Rosetta-XAI: An automated evaluation and explainability framework for code translation models.
Software Impacts 27 (2026), 100811. doi:10.1016/j.simpa.2026.100811
[35] Boqin Qin, Yilun Chen, Zeming Yu, Linhai Song, and Yiying Zhang. 2020. Understanding memory and thread safety practices and issues in
real-world Rust programs. In Proceedings ofthe41stACM SIGPLAN Conference onProgramming Language Design andImplementation (London,
UK)(PLDI 2020). Association for Computing Machinery, New York, NY, USA, 763–779. doi:10.1145/3385412.3386036
[36] Baptiste Roziere, Marie-Anne Lachaux, Lowik Chanussot, and Guillaume Lample. 2020. Unsupervised translation of programming languages. In
Proceedings ofthe34th International Conference onNeural Information Processing Systems (Vancouver, BC, Canada) (NIPS ’20) . Curran Associates
Inc., Red Hook, NY, USA, Article 1730, 11 pages.
[37] Baptiste Rozière, Jonas Gehring, Fabian Gloeckle, Sten Sootla, Itai Gat, Xiaoqing Ellen Tan, Yossi Adi, Jingyu Liu, Romain Sauvestre, Tal Remez,
Jérémy Rapin, Artyom Kozhevnikov, Ivan Evtimov, Joanna Bitton, Manish Bhatt, Cristian Canton Ferrer, Aaron Grattafiori, Wenhan Xiong, Alexandre
Défossez, Jade Copet, Faisal Azhar, Hugo Touvron, Louis Martin, Nicolas Usunier, Thomas Scialom, and Gabriel Synnaeve. 2024. Code Llama: Open
Foundation Models for Code. arXiv:2308.12950 [cs.CL] https://arxiv.org/abs/2308.12950
[38] Freda Shi, Daniel Fried, Marjan Ghazvininejad, Luke Zettlemoyer, and Sida I. Wang. 2022. Natural Language to Code Translation with Execution.
InProceedings ofthe2022 Conference onEmpirical Methods inNatural Language Processing , Yoav Goldberg, Zornitsa Kozareva, and Yue Zhang
(Eds.). Association for Computational Linguistics, Abu Dhabi, United Arab Emirates, 3533–3546. doi:10.18653/v1/2022.emnlp-main.231
[39] Weisong Sun, Chunrong Fang, Yuchen Chen, Guanhong Tao, Tingxu Han, and Quanjun Zhang. 2022. Code search based on context-aware code
translation. In Proceedings ofthe44th International Conference onSoftware Engineering (Pittsburgh, Pennsylvania) (ICSE ’22) . Association for
Computing Machinery, New York, NY, USA, 388–400. doi:10.1145/3510003.3510140
[40] Bo Wang, Aashish Kolluri, Ivica Nikolić, Teodora Baluta, and Prateek Saxena. 2023. User-Customizable Transpilation of Scripting Languages. Proc.
ACM Program. Lang. 7, OOPSLA1, Article 82 (April 2023), 29 pages. doi:10.1145/3586034
[41] Bo Wang, Ruishi Li, Mingkai Li, and Prateek Saxena. 2023. TransMap: Pinpointing Mistakes in Neural Code Translation. In Proceedings ofthe31st
ACM Joint European Software Engineering Conference andSymposium ontheFoundations ofSoftware Engineering (San Francisco, CA, USA)
(ESEC/FSE 2023). Association for Computing Machinery, New York, NY, USA, 999–1011. doi:10.1145/3611643.3616322
[42] Chaofan Wang, Tingrui Yu, Chen Xie, Jie Wang, Dong Chen, Wenrui Zhang, Yuling Shi, Xiaodong Gu, and Beijun Shen. 2026. EvoC2Rust: A
Skeleton-guided Framework for Project-Level C-to-Rust Translation. 12 pages. https://arxiv.org/abs/2508.04295
[43] Xuezhi Wang, Jason Wei, Dale Schuurmans, Quoc V. Le, Ed H. Chi, Sharan Narang, Aakanksha Chowdhery, and Denny Zhou. 2023. Self-Consistency
Improves Chain of Thought Reasoning in Language Models. In TheEleventh International Conference onLearning Representations, ICLR 2023,
Kigali, Rwanda, May 1-5,2023. OpenReview.net, Amherst, MA, USA, 24 pages. https://openreview.net/forum?id=1PL1NIMMrw
[44] Yue Wang, Weishi Wang, Shafiq Joty, and Steven C.H. Hoi. 2021. CodeT5: Identifier-aware Unified Pre-trained Encoder-Decoder Models for Code
Understanding and Generation. In Proceedings ofthe2021 Conference onEmpirical Methods inNatural Language Processing , Marie-Francine
Moens, Xuanjing Huang, Lucia Specia, and Scott Wen-tau Yih (Eds.). Association for Computational Linguistics, Online and Punta Cana, Dominican
Republic, 8696–8708. doi:10.18653/v1/2021.emnlp-main.685
[45] Justin D. Weisz, Michael Muller, Stephanie Houde, John Richards, Steven I. Ross, Fernando Martinez, Mayank Agarwal, and Kartik Talamadupula.
2021. Perfection Not Required? Human-AI Partnerships in Code Translation. In Proceedings ofthe26th International Conference onIntelligent User
Interfaces (College Station, TX, USA) (IUI ’21) . Association for Computing Machinery, New York, NY, USA, 402–412. doi:10.1145/3397481.3450656
Manuscript submitted to ACM

AdaTrans: Automated C-to-Rust Transformation via Error-Adaptive Repair 37
[46] Chunqiu Steven Xia, Yuxiang Wei, and Lingming Zhang. 2023. Automated Program Repair in the Era of Large Pre-Trained Language Models. In
Proceedings ofthe45th International Conference onSoftware Engineering (Melbourne, Victoria, Australia) (ICSE ’23) . IEEE Press, Piscataway, NJ,
USA, 1482–1494. doi:10.1109/ICSE48619.2023.00129
[47] Yingfei Xiong, Jie Wang, Runfa Yan, Jiachen Zhang, Shi Han, Gang Huang, and Lu Zhang. 2017. Precise condition synthesis for program repair. In
Proceedings ofthe39th International Conference onSoftware Engineering (Buenos Aires, Argentina) (ICSE ’17) . IEEE Press, Piscataway, NJ, USA,
416–426. doi:10.1109/ICSE.2017.45
[48] Jifeng Xuan, He Jiang, Yan Hu, Zhilei Ren, Weiqin Zou, Zhongxuan Luo, and Xindong Wu. 2015. Towards Effective Bug Triage with Software Data
Reduction Techniques. IEEE Trans. Knowl. Data Eng. 27, 1 (2015), 264–280. doi:10.1109/TKDE.2014.2324590
[49] Jifeng Xuan, He Jiang, Zhilei Ren, and Zhongxuan Luo. 2012. Solving the Large Scale Next Release Problem with a Backbone-Based Multilevel
Algorithm. IEEE Trans. Software Eng. 38, 5 (2012), 1195–1212. doi:10.1109/TSE.2011.92
[50] Jifeng Xuan, Matias Martinez, Favio DeMarco, Maxime Clément, Sebastian Lamelas Marcote, Thomas Durieux, Daniel Le Berre, and Martin
Monperrus. 2017. Nopol: Automatic Repair of Conditional Statement Bugs in Java Programs. IEEE Transactions onSoftware Engineering 43, 1
(2017), 34–55. doi:10.1109/TSE.2016.2560811
[51] Jifeng Xuan and Martin Monperrus. 2014. Test case purification for improving fault localization. In Proceedings ofthe22nd ACM SIGSOFT
International Symposium onFoundations ofSoftware Engineering (Hong Kong, China) (FSE 2014) . Association for Computing Machinery, New
York, NY, USA, 52–63. doi:10.1145/2635868.2635906
[52] Xuejun Yang, Yang Chen, Eric Eide, and John Regehr. 2011. Finding and understanding bugs in C compilers. SIGPLAN Not. 46, 6 (June 2011),
283–294. doi:10.1145/1993316.1993532
[53] Xuejun Yang, Yang Chen, Eric Eide, and John Regehr. 2011. Finding and understanding bugs in C compilers. In Proceedings ofthe32nd ACM
SIGPLAN Conference onProgramming Language Design andImplementation (San Jose, California, USA) (PLDI ’11) . Association for Computing
Machinery, New York, NY, USA, 283–294. doi:10.1145/1993498.1993532
[54] Zhiqiang Yuan, Wenjun Mao, Zhuo Chen, Xiyue Shang, Chong Wang, Yiling Lou, and Xin Peng. 2026. Project-Level C-to-Rust Translation via
Synergistic Integration of Knowledge Graphs and Large Language Models. Proceedings oftheACM onSoftware Engineering 3, FSE, Article 162
(July 2026), 24 pages. Conference: ACM SIGSOFT International Symposium on the Foundations of Software Engineering (FSE 2026).
[55] Qihao Zhu, Qingyuan Liang, Zeyu Sun, Yingfei Xiong, Lu Zhang, and Shengyu Cheng. 2024. GrammarT5: Grammar-Integrated Pretrained Encoder-
Decoder Neural Model for Code. In Proceedings oftheIEEE/ACM 46th International Conference onSoftware Engineering (Lisbon, Portugal)
(ICSE ’24). Association for Computing Machinery, New York, NY, USA, Article 76, 13 pages. doi:10.1145/3597503.3639125
[56] Yuqi Zhu, Jia Li, Ge Li, Yunfei Zhao, Jia Li, Zhi Jin, and Hong Mei. 2024. Hot or Cold? Adaptive Temperature Sampling for Code Generation
with Large Language Models. In Thirty-Eighth AAAI Conference onArtificial Intelligence, AAAI 2024, Thirty-Sixth Conference onInnovative
Applications ofArtificial Intelligence, IAAI 2024, Fourteenth Symposium onEducational Advances inArtificial Intelligence, EAAI 2014, February
20-27, 2024, Vancouver, Canada , Michael J. Wooldridge, Jennifer G. Dy, and Sriraam Natarajan (Eds.). AAAI Press, Palo Alto, CA, USA, 437–445.
doi:10.1609/AAAI.V38I1.27798
Manuscript submitted to ACM