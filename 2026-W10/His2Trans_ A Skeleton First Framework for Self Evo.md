# His2Trans: A Skeleton First Framework for Self Evolving C to Rust Translation with Historical Retrieval

**Authors**: Shengbo Wang, Mingwei Liu, Guangsheng Ou, Yuwen Chen, Zike Li, Yanlin Wang, Zibin Zheng

**Published**: 2026-03-03 05:42:08

**PDF URL**: [https://arxiv.org/pdf/2603.02617v1](https://arxiv.org/pdf/2603.02617v1)

## Abstract
Automated C-to-Rust migration encounters systemic obstacles when scaling from code snippets to industrial projects, mainly because build context is often unavailable ("dependency hell") and domain-specific evolutionary knowledge is missing. As a result, current LLM-based methods frequently cannot reconstruct precise type definitions under complex build systems or infer idiomatic API correspondences, which in turn leads to hallucinated dependencies and unproductive repair loops. To tackle these issues, we introduce His2Trans, a framework that combines a deterministic, build-aware skeleton with self-evolving knowledge extraction to support stable, incremental migration. On the structural side, His2Trans performs build tracing to create a compilable Project-Level Skeleton Graph, providing a strictly typed environment that separates global verification from local logic generation. On the cognitive side, it derives fine-grained API and code-fragment rules from historical migration traces and uses a Retrieval-Augmented Generation (RAG) system to steer the LLM toward idiomatic interface reuse. Experiments on industrial OpenHarmony modules show that His2Trans reaches a 99.75% incremental compilation pass rate, effectively fixing build failures where baselines struggle. On general-purpose benchmarks, it lowers the unsafe code ratio by 23.6 percentage points compared to C2Rust while producing the fewest warnings. Finally, knowledge accumulation studies demonstrate the framework's evolutionary behavior: by continuously integrating verified patterns, His2Trans cuts repair overhead on unseen tasks by about 60%.

## Full Text


<!-- PDF content starts -->

His2Trans: A Skeleton First Framework for Self Evolving C to
Rust Translation with Historical Retrieval
SHENGBO WANG,Sun Yat-sen University, China
MINGWEI LIU∗,Sun Yat-sen University, China
GUANGSHENG OU,Sun Yat-sen University, China
YUWEN CHEN,Sun Yat-sen University, China
ZIKE LI,Sun Yat-sen University, China
YANLIN WANG,Sun Yat-sen University, China
ZIBIN ZHENG,Sun Yat-sen University, China
Automated C-to-Rust migration encounters systemic obstacles when scaling from code snippets to industrial
projects, mainly because build context is often unavailable (“dependency hell") and domain-specific evolu-
tionary knowledge is missing. As a result, current LLM-based methods frequently cannot reconstruct precise
type definitions under complex build systems or infer idiomatic API correspondences, which in turn leads to
hallucinated dependencies and unproductive repair loops. To tackle these issues, we introduceHis2Trans, a
framework that combines a deterministic, build-aware skeleton with self- evolving knowledge extraction to
support stable, incremental migration. On the structural side,His2Transperforms build tracing to create
a compilable Project-Level Skeleton Graph, providing a strictly typed environment that separates global
verification from local logic generation. On the cognitive side, it derives fine-grained API and code-fragment
rules from historical migration traces and uses a Retrieval-Augmented Generation (RAG) system to steer
the LLM toward idiomatic interface reuse. Experiments on industrial OpenHarmony modules show that
His2Transreaches a 99.75% incremental compilation pass rate, effectively fixing build failures where baselines
struggle. On general-purpose benchmarks, it lowers the unsafe code ratio by 23.6 percentage points compared
to C2Rust while producing the fewest warnings. Finally, knowledge accumulation studies demonstrate the
framework’s evolutionary behavior: by continuously integrating verified patterns,His2Transcuts repair
overhead on unseen tasks by about 60%.
CCS Concepts:•Software and its engineering→Software evolution;Automatic programming.
Additional Key Words and Phrases: C-to-Rust, LLM, RAG, incremental migration, compiler feedback
1 Introduction
Memory-safety issues in legacy C codebases remain a primary source of security vulnerabilities.
Industry reports from major technology companies, such as Microsoft and Google, indicate that
approximately 70% of security vulnerabilities stem from memory-safety errors [ 19,27]. Rust has
emerged as a robust solution to this persistent issue, offering memory-safety guarantees through
its ownership and borrowing models without sacrificing low-level control or performance [ 13,14].
Consequently, migrating legacy C systems to Rust has become a widespread industry trend [ 22,28].
However, manual migration is labor-intensive, error-prone, and prohibitively expensive for large-
scale systems, creating an urgent demand for automated translation tools [7, 25].
The field of automated C-to-Rust translation has evolved from rule-based systems to approaches
based on Large Language Models (LLMs). Traditional rule-based transpilers, such as C2Rust [ 7],
∗Corresponding Author
Authors’ Contact Information: Shengbo Wang, puppytagge@gmail.com, Sun Yat-sen University, Guangzhou, China; Mingwei
Liu, liumw26@mail.sysu.edu.cn, Sun Yat-sen University, Guangzhou, China; Guangsheng Ou, ougsh3@mail2.sysu.edu.cn,
Sun Yat-sen University, Guangzhou, China; Yuwen Chen, chenyw95@mail2.sysu.edu.cn, Sun Yat-sen University, Guangzhou,
China; Zike Li, lizk8@mail2.sysu.edu.cn, Sun Yat-sen University, Guangzhou, China; Yanlin Wang, wangylin36@mail.
sysu.edu.cn, Sun Yat-sen University, Guangzhou, China; Zibin Zheng, zhzibin@mail.sysu.edu.cn, Sun Yat-sen University,
Guangzhou, China.arXiv:2603.02617v1  [cs.SE]  3 Mar 2026

2 Wang, et al.
prioritize compilation guarantees but generate non-idiomatic code that relies heavily on unsafe
blocks, effectively carrying C’s memory risks into Rust syntax [ 4]. With the advent of Generative
AI, mainstream solutions have shifted from rule-based methods [ 7,35] to LLM-based approaches.
However, current LLM-based research is constrained by context window limitations and generation
instability. As input length increases, the probability of hallucinations and syntax errors rises
significantly [ 12]. Consequently, most existing works focus on function-level translation [ 21,37],
typically evaluating performance only on small-scale algorithms or competitive programming
datasets rather than complex software systems.
Furthermore, industrial migration is rarely a one-time total rewrite; it is typically a gradual process
where C and Rust coexist. An effective translation system must maximize the reuse of migrated
Rust APIs while maintaining compatibility with untranslated C code, ensuring that new code
integrates seamlessly into the evolving ecosystem. However, existing frameworks treat translation
as an isolated task, failing to address the systemic challenges inherent in this evolutionary context:
•Missing Build Context and “Dependency Hell”:Large projects rely on complex build
systems (e.g., Makefiles, CMake), where macro definitions and conditional compilation
options directly determine type availability. Simple isolated translation loses this context,
leading LLMs to hallucinate definitions.
•The Gap in Domain-Specific Evolutionary Knowledge:General-purpose LLMs excel at
standard library calls (libc) but fail when encountering domain-specific or private APIs (e.g.,
internal kernel drivers) [ 16,40]. In a gradual migration, new code must reuse previously
migrated Rust APIs. However, without access to the project’shistorical migration traces
or API mapping rules, LLMs cannot infer these idiomatic correspondences, leading to
fabricated calls and integration failures [16].
•Inefficient Repair Loops and Unscalable Costs:Existing frameworks often mishandle
dependencies, triggering cascading compilation errors. This causes the repair loop to diverge,
rendering automated migration of large-scale projects economically infeasible [40].
To address these challenges, we proposeHis2Trans, a framework designed to facilitate evolving
C-to-Rust translation by synergizing multi-level knowledge extraction with a compilable skeleton.
Unlike probabilistic approaches, our method operates on two dimensions to systematically resolve
the bottlenecks. First, regarding the structural dimension, we utilize build system tracing to construct
a compilable Project-Level Skeleton Graph, which explicitly resolves missing build contexts by
enforcing a strictly typed environment. Second, regarding the knowledge dimension, we mine
historical migration projects to extract fine-grained rules; this bridges the evolutionary knowledge
gap, guiding the LLM to reuse domain-specific APIs consistent with historical patterns. Ultimately,
by decoupling structural verification from logic generation, this skeleton-driven approach isolates
cascading errors, resolving the bottleneck of unscalable repair costs.
To situate our work within the existing landscape, Table 1 contrasts the design philosophies of
His2Trans against prior paradigms. While rule-based methods often compromise safety by relying
onunsafe blocks, and existing LLM-based approaches face challenges with probabilistic context
inference, His2Trans introduces a deterministic, build-aware architecture. Crucially, we implement
two fundamental shifts: (1) we transition the context source from speculative static analysis to
concrete Build Traces, establishing a Type-Consistent Skeleton that mitigates “dependency hell”;
(2) we replace static prompt engineering with accumulable multi-level knowledge, enabling aself-
evolvingworkflow that progressively improves through historical data mining, which represents
a capability absent in prior one-off translation methods.
Experimental results on real-world industrial projects demonstrate that His2Trans achieves a
superior incremental compilation pass rate of 99.75%, whereas baselines fail due to missing build
contexts. To validate the generalizability of our mined knowledge beyond the source domain, we

His2Trans: A Skeleton First Framework for Self Evolving C to Rust Translation with Historical Retrieval 3
Table 1. Comparison of Design Philosophies: His2Trans vs. Existing C-to-Rust Paradigms
Paradigm Representative MethodsTarget
ScopeContext SourceStructural
GuaranteeKnowledge
Source
Rule-based C2Rust [7] Project Local ASTNone
(Relies on
unsafe)Hard-coded
Rules
Unit-Level
LLMRustFlow [38],
SPECTRA [ 21], SafeTrans [ 6],
C2RustTV [39], VERT [32]Snippet /
FunctionMultimodal Spec
/ IRI/O EquivalenceLLM Internal
Knowledge +
Few-shot
Project-
Level LLMEvoC2Rust [29],
PTRMAPPER [34],
RustMap [1], LLMigrate [17],
Tymcrat [10]Full
Project /
ModuleGlobal Static
AnalysisGlobal
Consistency /
Compilable
SkeletonContext from
Static Analysis
Industrial
Project-
Level LLMHis2Trans(Ours)General &
Industrial
ProjectsBuild Trace
(Concrete
Dependencies)Type-Consistent
Skeleton
(Type-Truth)Accumulable
Knowledge
(Self-Evolving
RAG)
further evaluated the framework on general-purpose benchmarks, where it effectively reduced the
unsafe code ratio by 23.6 percentage points compared to C2Rust and achieves the lowest warning
count. Crucially, our knowledge accumulation experiments confirm the system’s evolutionary
capability: by continuously integrating verified translation pairs,His2Transreduces the repair
cost for unseen modules by approximately60%, proving that the framework becomes more efficient
as it evolves.
This paper makes the following contributions:
•A Self-Evolving Knowledge Extraction Framework:We propose a RAG system mining
API- and Fragment-Level rules from historical data. This creates an accumulable knowl-
edge base that facilitates interface reuse and progressively enhances efficiency through
verification feedback.
•A Build-Aware Incremental Architecture:We introduce a skeleton-based strategy utiliz-
ing build tracing to enforce a strictly typed environment. Decoupling structural verification
from logic generation resolves “dependency hell” and isolates errors at the function level.
•Extensive Evaluation:His2Transachieves a 99.75% compilation pass rate in industrial
settings while demonstrating robust generalizability on standard benchmarks. Furthermore,
knowledge accumulation reduces repair overhead by ∼60% on unseen tasks, validating the
framework’s evolutionary capability.
2 Methodology
This chapter details the overall framework ofHis2Trans. As illustrated in Figure 1,His2Trans
operates in three stages: (1)Knowledge Base Constructionmines historical mapping rules to
facilitate precise retrieval; (2)Rust Skeleton Constructionemploys static analysis and build trace
to establish a strictly typed project structure and dependency graph; and (3)Incremental Rust
Function Body Translationgenerates function logic bottom-up based on the topological order,
achieving automated project-wide migration.

4 Wang, et al.
Stage 2: Rust Skeleton Construction Stage 3: Incremental Rust Function Body Translation
RepairFailSuccessKnowledge  RetrievalFunction 
ExtractionMapping Rule 
Knowledge BaseStage 1: Knowledge Base Construction
Initial PairingFile-Level 
Re-rankingFunction -Level 
Re-rankingMapping Rule 
Mining
Layered File 
Skeleton 
Construction
Module 
DecompositionDependency IntegrationFile-Level Rust SkeletonsTranslate 
Bottom -UpSuccessfully 
Translated C-Rust 
Function Pairs
Rust File AssemblyProject 
Skeleton 
ConstructionKnowledge 
Accumulation
C Project
Dependency map
Function 
Bodys
Project 
Skeleton Graph
Mapping Rules
Topological 
Scheduler
Skeletons with 
Translated Functions
Definitions
Macros
Function 
SignatureRust Compiler
Rule-Based Repair
 LLM -Based Repair
Rust Project
C Repositories
Rust RepositoriesFile-Level C -Rust 
Candidate PairsFile-Level C -Rust 
Translation Pairs
Function -Level C -
Rust Candidate PairsFunction -Level C -
Rust Translation Pairs
API-Level Rules
Fragment -Level Rules
Fig. 1. Overview of the C-to-Rust translation framework.
OBJECTIVES
•Primary Goals:
• Structure Learning: Identify and classify structurally similar code blocks (Full/Partial).
• Dictionary Mining: Extract precise API mappings and idiom transformations between C and Rust.
•Granularity: [Target specific patterns ranging from whole functions to small loops and memory management idioms.]
•Value Criterion: [Assess utility for developers translating specific C snippets to Rust.]
PRE -JUDGMENT FILTERS
•Filter 1: Entity Name Check [Verify concept consistency for Full/Partial classification; reject clear domain mismatches.]
•Filter 2: Empty/Trivial Code [Discard empty function bodies, default constructors/destructors, or trivial single -line wrappers.]
•Filter 3: FFI Wrapper [Exclude Rust functions that serve solely as direct Foreign Function Interface wrappers without added logic.]
•Filter 4: Semantic Domain Mismatch [Reject pairs with fundamental conflicts, such as matching memory management logic with 
UI/Logging logic.]
•Filter 5: Empty Structs [Filter out struct definitions that lack fields or content.]
•Filter 6: Definition vs. Usage Asymmetry [Prevent matching a function definition in one language with a function call/test wrapper in 
the other.]
CLASSIFICATION & MINING
•3.1 Partial Match ( is_partial )[Extract matching sub -blocks or specific logic fragments even when the overall function structure differs.]
•3.2 API Mappings ( has_api_mappings )
• Context Symmetry: [Enforce matching of logical operations (Logic -to-Logic), prohibiting Definition -to-Usage matching.]
• Semantic Matching: [Prioritize functional equivalence over naming similarity (e.g., C add vs. Rust push).]
• Inference Strategy: [Infer mappings based on control flow position, operation type, and data flow analysis.]
• Extraction Scope: [Capture valid mappings despite different names; reject similar names with different functions.]
•3.3 None [Classify unrelated or trivial pairs that fail all criteria.]
INPUT DATA
•[C Code Snippet]
•[Rust Code Snippet]SYSTEM OBJECTIVES & CONSTRAINTS
•Primary Goal: [Generate code that strictly compiles, permitting unsafe operations where necessary to replicate C semantics.]
•Namespace Contract: [Strictly utilize crate::types for data structures and crate:: compat for external FFI calls; prohibit ad -hoc external 
definitions.]
•Safety & Type Discipline: [Mandate unsafe blocks for all pointer dereferences and enforce explicit type casting (e.g., size_t → usize ) to 
satisfy Rust's strict type system.]
•Critical Avoidance: [Prevent common hallucination errors such as inventing external APIs, invalid pointer method calls, or unwrapped 
function pointers.]
•Internal Resolution: [Resolve calls to other project functions using full module paths (e.g., crate:: src_module ::Func) based on internal 
mapping.]
DYNAMIC CONTEXT INJECTION
•Target Signature: [The mandatory Rust function signature that the generated implementation must match exactly.]
•Skeleton Context: [The existing compilation environment, including available type definitions, global symbols, and FFI compatibility 
layers.]
•RAG Knowledge: [Injected domain -specific translation rules and API mappings retrieved from the project's historical data.]
•Translation Hints: [Auxiliary metadata providing callee signatures, inline helper context, and field access patterns.]
OUTPUT SCOPING
•Scope Definition: [Generate exactly one Rust item (Function Body, Constructor, or Drop impl) as specifically requested.]
•Structural Constraints: [Strictly exclude enclosing impl blocks, module wrappers, helper functions, or import statements to ensure 
seamless injection into the skeleton.]
•Code Compactness: [Maintain concise implementation logic, avoiding excessive temporary variables or unused assignments.]
INPUT DATA
•[Target Rust Signature]
•[Preprocessed C Source Code]
•[Skeleton Context Snippet]
•[Retrieved RAG Rules]Knowledge Mining Prompt Function Generation Prompt
Fig. 2. Prompt templates utilized for knowledge mining (Left) and function generation (Right).
C Project
Rust Project
File1.c File2 .c File3. c File4 .c
File1.rs File2.rs File3.rs File4.rsInitial Pairing & File-Level Re-ranking
······File1. rs
File1 .c File4 .c
File4. rs
File1 .c File3 .cFile-Level C -Rust 
Translation Pairs
Fragment -Level Rules
Rust:
forbintext.bytes() {
accumdata =accumdata *10+
u32::from(b-b'0');
accumcount +=1;
ifaccumcount ==3{
bb.append_bits (accumdata , 10);
accumdata =0;
accumcount =0;}}
C:
for(; *digits !='\0'; digits ++) {
charc =*digits;
accumData =accumData *10+
(unsigned int)(c -'0');
accumCount ++;
if(accumCount ==3) {
appendBitsToBuffer (accumData , 
10, buf, &result.bitLength );
accumData =0;
accumCount =0;}}C functionRust function
letmutaccumdata :u32=0;
letmutaccumcount :u8=0;
forbintext.bytes() {
accumdata =accumdata *10+u32::from(b-
b'0');
accumcount +=1;
ifaccumcount ==3{
bb.append_bits (accumdata , 10);
accumdata =0;
accumcount =0;}}
ifaccumcount >0{
bb.append_bits (accumdata , accumcount *3+1);}
unsigned intaccumData =0;
intaccumCount =0;
for(; *digits!='\0'; digits++) {
charc=*digits;
accumData =accumData *10+(unsigned int)(c
-'0');
accumCount ++;
if(accumCount ==3) {
appendBitsToBuffer (accumData , 10, buf, 
&result.bitLength );
accumData =0;
accumCount =0;}}
if(accumCount >0)
appendBitsToBuffer (accumData , accumCount *3+
1, buf, &result.bitLength );Function Extraction  & Function -Level Re-rankingMapping Rule MiningFunction -Level C -Rust Translation Pairs
File4 .c File1.rs
usecore::convert::TryFrom;
usecore::fmt;
...
pubfnmake_bytes (...) ->
Self{
}
pubfnmake_numeric (...) ->
Self{
}
pubfnmake_alphanumeric (...) 
->Self{
}#include <assert.h >
#include <limits.h >
...
boolqrcodegen_encodeText (...) {
}
boolqrcodegen_encodeBinary (...) {
}
structqrcodegen_Segment
qrcodegen_makeNumeric (...) {
}API-Level Rules
Rust API : bb.append_bits
C API: appendBitsToBuffer
Fig. 3. Workflow of Knowledge Base Construction. The pipeline executes a coarse-to-fine mining process:
File-Level Pairing→Function-Level Re-ranking→Rule Extraction (API & Fragment).
2.1 Knowledge Base Construction
This phase aims to extract implicit translation knowledge from historical codebases into a structured
Mapping Rule Knowledge Base. As illustrated in Figure 3, we adopt a coarse-to-fine mining strategy
to progressively filter noise and extract precise patterns. The workflow proceeds in three stages:
first, File-Level Pairing narrows down the search space by identifying high-confidence file corre-
spondences from raw repositories; next, Function-Level Alignment performs semantic matching to

His2Trans: A Skeleton First Framework for Self Evolving C to Rust Translation with Historical Retrieval 5
pinpoint functionally equivalent C-Rust function pairs within these files; finally, Mapping Rule
Mining analyzes these aligned pairs to extract fine-grained API-Level Rules and Fragment-Level
Rules, which serve as the reference context for subsequent generation.
2.1.1 Data Collection and Initial Pairing.This stage extracts File-Level Candidate Pairs from
repositories using two strategies tailored to project evolution characteristics. For theGeneral Case
(where evolution history is absent), we adopt a global search strategy based on the Cartesian
product of all C and Rust files, creating a complete search space to ensure full recall. Conversely,
for theCo-evolution Case, where C and Rust co-evolve within the same codebase or possess explicit
migration records, we leverage Historical Migration Traces. By applying heuristic filtering rules
to Git metadata (refer to Table 2), we identify high-confidence pairs based on signals such as
simultaneous modifications or explicit migration keywords. This targeted approach significantly
prunes the search space, offering superior efficiency compared to the global search.
Table 2. Heuristic Rules for Mining File-level C-to-Rust Translation Pairs
Category Dimension Strategy & Rationale
Git History
(Dynamic)Synchronous
(Same Commit)Commit Message Keyword Matching:Detects explicit migration intent via key-
words (e.g., “rewrite”, “port”) in commit messages.
Build Config Switch:Identifies commits that remove C files and add Rust files
within the same build target definition (e.g., Makefile).
Interface Migration:Captures logic replacement where C function calls are substi-
tuted by Rust equivalents in caller code.
Code Churn Balance:Matches the magnitude of deleted C lines with added Rust
lines usingnumstatto find structural correspondence.
Asynchronous
(History Window)Sequential Re-implementation:Links a C file deletion to a subsequent Rust file
creation within a 365-day window (“Delete-then-Create” pattern).
Evolutionary Coupling:Identifies file pairs that are frequently modified together
over time, implying logical dependency.
Developer Identity:Prioritizes pairs where the Rust author matches the original C
maintainer (or recent contributor).
Codebase
Snapshot
(Static)Spatial Module Colocation:Scans build systems to find C and Rust files co-existing in the
same module/target, suggesting functional relevance.
SemanticKey Token Overlap:Calculates the intersection of unique identifiers (constants,
error codes). Pairs with≥3matches are retained.
Shared Literals:Matches long string literals ( >5chars) such as log messages or
error prompts identical in both files.
2.1.2 Multi-Stage Filtering and Re-ranking.We implement a cascading filtering mechanism to
refine coarse candidates into semantically equivalent translation pairs. At the file level, we first
employ BM25 to retrieve the top-20 candidates based on lexical overlap (e.g., identifiers). We then
apply the code-optimized jina-reranker-v3 [30] to capture deep semantic alignment, isolating
the top-5 high-confidence File-Level Translation Pairs. At the function level, we decompose these
file pairs and generate a Cartesian product of their internal C and Rust functions. By re-applying
jina-reranker-v3 to rank these combinations, we extract the top-5 Function-Level Translation
Pairs, completing the transition from coarse file matching to precise granular alignment.
2.1.3 Mapping Rule Mining and Knowledge Base Formulation.After acquiring high-confidence
function-level translation pairs, the framework executes mapping rule mining. We employ LLMs to
mine generalizable patterns from the aligned function pairs (see Figure 2). This process extracts
two key categories of knowledge: (1) API-Level Rules, which map C library calls to idiomatic Rust
expressions; and (2) Fragment-Level Rules, which capture syntactic transformations of algorithmic

6 Wang, et al.
logic. These rules populate the Knowledge Base, which additionally supports a “Human-in-the-loop”
mechanism, enabling experts to verify mined rules or inject manual priors to mitigate extraction
biases.
2.2 Rust Skeleton Construction
This stage reconstructs the C source into a structured Project-Level Skeleton Graph to serve as
the static foundation for migration. The process operates in three steps: (1) mapping the physical
directory hierarchy to establish the module structure; (2) employing hierarchical synthesis to
populate type definitions and interface contracts; and (3) resolving cross-module references to form
a statically consistent Skeleton, providing precise structural support for incremental translation.
2.2.1 Project-Level Skeleton Synthesis.We employ a direct mapping strategy to construct the
Project-Level Skeleton, mirroring the C directory hierarchy into a Rust module tree. The framework
traverses the source tree, converting subdirectories into parent modules and mapping individual C
source files one-to-one to independent Rust modules. During this process, filenames are normalized
tosnake_case to comply with Rust syntax, while a mapping table is maintained to ensure code
traceability. This establishes a structural container for the subsequent injection of definitions and
signatures.
2.2.2 Layered File-Level Skeleton Synthesis.Following the structural mapping, we populate the
modules with static symbol contracts via a layered synthesis strategy. First, we utilize Build Tracing
to preprocess sources and extract strict type definitions (structs, unions, enums), ensuring memory
layout consistency without relying on probabilistic inference. Second, we resolve global states by
encapsulating file-local statics as module-private storage and mapping global variables to public
mutable resources. Finally, we generate Rust function signatures that preserve C calling conventions,
initializing bodies with unimplemented! . This results in a statically complete Skeleton capable of
passing compilation checks.
2.2.3 Project-Level Skeleton Graph Generation.Although isolated module skeletons are internally
type-consistent, they do not yet constitute a complete software system. To integrate isolated
modules into a cohesive system, we first establish a Global Symbol Index by analyzing header
inclusions and external calls. Based on this index, we transform implicit C linking dependencies
into explicit Rust reference paths (e.g., use crate::... ) and inject necessary visibility modifiers to
bridge call chains. This construction yields the Project-Level Skeleton Graph, where nodes represent
typed modules and edges denote data dependencies. Crucially, this topology dictates the execution
order, providing a precise scheduling basis for the subsequent bottom-up incremental translation.
2.3 Incremental Rust Function Body Translation
Leveraging the static foundation, this stage executes the incremental generation and verification of
function bodies (Figure 4). The process operates within a strictly typed context composed of the
Shared Layer (consolidating ABI-critical definitions) and the Module Skeleton (providing typed
signatures initialized with unimplemented! and file-scope statics). Guided by these constraints, the
LLM generates function implementations that strictly adhere to the global memory layout defined
in the shared context.
2.3.1 Dependency-Aware Topological Scheduling.Based on the Project-Level Skeleton Graph con-
structed in the preceding stage, this framework implements a dependency-aware scheduling
strategy. To ensure contextual integrity, the framework employs a Topological Scheduler that
performs a topological sort on the Project-Level Skeleton Graph to resolve call dependencies. This
transforms the graph into a linear, bottom-up execution sequence. The scheduler prioritizes “leaf

His2Trans: A Skeleton First Framework for Self Evolving C to Rust Translation with Historical Retrieval 7
C Project#[repr(C)]
pubstructDefinition1 {
}
#[repr(C)]
pubstructDefinition2 {
}
pubstaticmutGLOBAL_VAR :i32=0;
/* ... */Shared Layer
usecrate::types::*;
pubunsafefnFunction1 () { 
unimplemented! ();
}
pubunsafefnFunction2 () {
unimplemented! ();
}
pubunsafefnFunction3 () {
unimplemented! ();
}File1.rs
usecrate::types::*;
pubunsafefnFunction1 () {
unimplemented! ();
}
pubunsafefnFunction2 () {
unimplemented! ();
}
pubunsafefnFunction3 () {
unimplemented! ();
}File2.rsModule Skeleton
Topological 
SchedulerUnlock DependentsRetryRepair
Accept
File1:: func2
File2:: func1File1::func1
LLM
TranslatorRust 
Compiler
File1:: func1
File1:: func3File2:: func1
File2::func2
File2::func3Dependency map
File1:: func2
LLM -Based Repair
Rule -Based repair
FailFail
Retry
Parallel Translation Batch
Mapping Rule 
Knowledge Base
Fig. 4. Workflow of Incremental Function Translation. The Topological Scheduler dispatches parallel tasks
based on the strictly typed Shared Layer and Module Skeleton, followed by a hybrid Rule/LLM-Based repair
loop.
nodes”—functions whose dependencies are fully resolved—and groups independent tasks into a
Parallel Translation Batch for concurrent processing. This mechanism guarantees that higher-level
functions are processed only after their dependencies are verified, ensuring the LLM accesses
accurate semantic context while strictly adhering to the Rust build order.
2.3.2 Function Translation.Upon task scheduling, we execute Context Retrieval to construct a strict
"Compilation Context." This involves loading the target C code and extracting the corresponding
Rust signatures, global type definitions, and external interfaces from the Skeleton. This context
enforces global type constraints, preventing errors typical of isolated translation.
To augment this structural context, we employ a RAG phase using the coarse-to-fine strategy (see
Section 2.1). We retrieve the top- 𝑘similar pairs from the Knowledge Base and extract their associated
API-Level Rules and Fragment-Level Rules. The framework synthesizes these historical rules with
the Compilation Context to construct a dual-source prompt (Figure 2). This design inject mined
API-Level rules and Fragment-Level rules as specific reference examples, while simultaneously
enforcing strict syntactic constraints to regulate the model’s coding behavior.
Guided by this prompt, the LLM generates function bodies that strictly adhere to project de-
pendencies, mitigating hallucinations. The generated implementation is then populated into the
Skeleton, completing the migration unit.
2.3.3 Compiler-Feedback Iterative Repair.To guarantee robustness, we implement a closed-loop
Compiler-Feedback Iterative Repair mechanism. Immediately post-generation, the framework
triggers verification to capture errors (e.g., ownership violations) early, acting as a quality gate to
prevent downstream propagation. Once a compilation error is detected, the right side of Figure 4
illustrates the repair mechanism, a prioritized strategy is invoked: first, Rule-Based Repair targets
deterministic patterns such as omitted type conversions; if unresolved, the system escalates to

8 Wang, et al.
LLM-Based Repair, injecting compiler error messages as feedback to guide deep semantic fixes.
Should the repair limit be exhausted, the system executes a Fallback to C2Rust-generated unsafe
code to ensure compilation integrity and functional correctness. Finally, successful verification
updates the Skeleton Graph, triggering the “Unlock Dependents” action to advance the scheduler.
2.3.4 Knowledge Accumulation.As incremental translation progresses, the framework extracts Rust
functions that have passed compilation verification, along with their source C functions, as high-
quality translation pairs. Subsequently, this empirically verified data is fed back into the mapping
rule mining module to expand API-Level Rules and Fragment-Level Rules within the knowledge
base. This closed-loop feedback mechanism achieves self-enhancement of the framework, effectively
improving the accuracy and efficiency of subsequent translation tasks through the continuous
accumulation of correct samples.
3 Experimental Setup
3.1 Experimental Configuration
Experiments were conducted on a server with dual Intel Xeon Gold 6430 CPUs, 503 GiB RAM, and
four NVIDIA RTX 5880 Ada GPUs, running Ubuntu 22.04. We used Clang 14.0.0, Rust 1.94.0-nightly,
and Python 3.13.5.
For LLMs, we employed Qwen2.5-Coder-32B for offline knowledge extraction and DeepSeek-V3 /Claude-Opus-4.5
for translation/repair (Temp=0.0). We configured inference with a temperature of 0.0, top- 𝑝of 1.0,
top-𝑘of -1, and a repetition penalty of 1.0. The maximum token limit per request was 8,192.
To ensure a rigorous assessment of the framework’s autonomous capabilities, we enforced a
strictzero-human-interventionpolicy across all experiments. All translation results and repair
iterations were executed entirely by the framework; no manual post-editing or cherry-picking was
performed during the evaluation.
3.2 Datasets
We employed two distinct datasets to evaluate the framework’s performance on both domain-
specific and general-purpose C projects. Table 3 presents the detailed statistics of these projects.
Domain-Specific Projects:We selectedOpenHarmony[ 23] as a representative domain-
specific codebase. As a large-scale distributed operating system widely deployed in industrial
embedded devices, OpenHarmony features intricate build systems (e.g., GN/Ninja), extensive
Hardware Abstraction Layers (HAL), and complex inter-component dependencies. Unlike isolated
algorithmic benchmarks, these characteristics provide a rigorous stress test for evaluating the
framework’s capability to handle real-world “dependency hell” and recover strictly typed contexts
in a production environment. From this codebase, we curated five production projects. The selection
criteria were as follows: (i) the codebases are primarily written in C, (ii) they possess executable
unit tests, and (iii) they contain at least one function. From the resulting candidate list, we randomly
sampled five submodules for evaluation. Note that these five submodules were strictly excluded
from the source repositories used for knowledge base construction (Section 2.1) to ensure zero data
leakage during evaluation.
General-Purpose Projects:We selected ten open-source C benchmark projects that are fre-
quently utilized as evaluation targets in prior studies[ 2,20,26,34,41], covering data structures,
small-scale programs, and larger projects.
3.3 Baselines
We evaluate our approach against seven representative baselines: C2Rust [ 7], SmartC2Rust [ 26],
RUSTINE [2], PTRMAPPER [34], C2SaferRust [20], EvoC2Rust [29], and Tymcrat [10].

His2Trans: A Skeleton First Framework for Self Evolving C to Rust Translation with Historical Retrieval 9
Table 3. Statistics of the datasets used in evaluation.
Category Project#LOC #Files #Funcs #Macros #Tests #Definitions
Domain-
Specificosal 161 1 8 3 4 1
uhdf2_shared 208 2 7 9 15 0
core_shared 338 7 26 1 5 1
host 1,892 14 127 16 8 2
appverify_lite 3,544 8 160 11 8 14
General-
Purposeht 15 1 2 2 3 0
qsort 27 1 3 0 6 0
quadtree 319 5 24 2 4 4
buffer 353 2 23 4 14 1
rgba 420 2 13 3 10 2
urlparser 426 1 20 6 3 1
genann 459 3 13 10 12 7
avl 836 1 28 29 2 9
bzip2 4,325 9 93 144 1 15
zopfli 6,115 29 113 47 1 41
Regarding data sourcing, we directly adopt results from the original papers for the closed-source
methods (SmartC2Rust, RUSTINE, and PTRMAPPER). For all other baselines, we conducted unified
reproduction runs in our environment. Entries marked with “–” indicate metrics that were either
unreported in the original literature or unobtainable in our setup (e.g., due to compilation failures).
3.4 Metrics
We report the following metrics: Incremental Compilation Pass Rate (ICompRate) [ 2,29], Functional
Correctness (FC) [2, 20, 26, 29, 34], Unsafe Ratio [2, 20, 26, 29, 34], and Warnings [2, 34].
(1) Incremental Compilation Pass Rate:We start with a verified “skeleton version” containing
placeholder bodies. Subsequently, we incrementally restore actual function implementations one-
by-one, triggering compilation checks and rolling back upon failure. The rate is defined as the ratio
of successfully restored functions to the total evaluated. Additionally, for our method, functions that
revert to C2Rust-generated unsafe code due to repair exhaustion are strictly counted as failures.
(2) Functional Correctness:We calculate FC based on the test pass rate. For each project, we
utilize a pre-maintained collection of project-level unit tests, temporarily incorporating them into
the translation artifact during evaluation and executing them via a standard test runner. FC is
defined as the number of passed tests divided by the total number of tests.
(3) Unsafe Ratio:We perform static statistics on the translated Rust source code, separately
counting lines of code containing the unsafe keyword and lines within unsafe syntactic scopes,
taking the union of the two as the number of unsafe-related lines. The Unsafe Ratio is defined as
the number of unsafe-related lines divided by the total number of lines of code (calculated based
on source lines, excluding comments and string literals to prevent counting interference).
(4) Warning:We run the Rust static checker on the translation artifacts and parse its structured
diagnostic output, counting the number of warnings generated to characterize code quality and
potential defect proneness.
We report warning counts exclusively for crates that compile successfully. For failed compilations,
where early termination precludes accurate diagnostic collection, we denote the entry as “–”.
For RQ3 and RQ4, we additionally report AvgRepair , which represents the average number of
compilation-guided repair rounds utilized to achieve a successful build.

10 Wang, et al.
3.5 Implementation
Following the Knowledge Base Construction process described in Section 2.1, we prioritized theCo-
evolution Casestrategy to leverage the rich history of the OpenHarmony codebase. OpenHarmony
serves as an ideal industrial testbed due to its active C-to-Rust migration. We targeted all 21
sub-repositories containing Rust implementations, identifying 229,127 commits where C modules
were explicitly refactored into Rust. The extracted knowledge primarily comprises two categories:
API-Level rules and Fragment-Level rules. Based on the current knowledge base, we extracted a
total of 81,850 API-Level rules and 19,685 Fragment-Level rules, amounting to 101,535 items in
total. Although extracted from a domain-specific codebase, this knowledge base captures universal
C-to-Rust translation patterns (e.g., standard library mappings and algorithmic idioms), enabling
us to apply it to the general-purpose benchmarks in RQ2.
To evaluate functional correctness, we prioritize the reuse of test code from the source implemen-
tation. For general-purpose benchmarks, we prioritize migrating tests from the original projects
to the Rust framework to verify semantic consistency. In cases where original tests are absent,
we supplement them by developing test cases based on core interfaces. Specifically, projects with
existing tests that we migrated include urlparser ,avl,buffer ,rgba ,genann , and ht. Projects
for which we developed or maintained tests include qsort ,bzip2 ,zopfli , and quadtree . For
domain-specific benchmarks, we similarly opt to reuse tests from the original projects. All selected
modules (including osal ,uhdf2_shared ,core_shared ,host , and appverify_lite ) directly reuse
their original unit test suites to ensure the validity of the verification.
4 Experimental Evaluation
We conduct experiments to evaluate the effectiveness of His2Trans, aiming to answer the following
research questions (RQs):
•RQ1:How effective is our approach in translating domain-specific C projects?
•RQ2:How effective is our approach in translating general-purpose projects?
•RQ3:How do mapping rules and repair loops impact the translation of function bodies?
•RQ4:How does knowledge accumulation affect translation performance?
4.1 RQ1: Domain-Specific Effectiveness
4.1.1 Design.RQ1 evaluates the end-to-end migration effectiveness on domain-specific projects
characterized by macro-intensive build contexts, external symbol dependencies, and long de-
pendency chains. We compared our approach against baseline methods on five OpenHarmony
submodules listed in Table 3. To evaluate the sensitivity to domain knowledge density, we swept
the retrieval depth𝐾∈{1,3,5,10}for theDeepSeek-V3.2model.
We prioritized DeepSeek-V3.2 for this parameter sweep due to its superior cost-effectiveness,
which allows for extensive experimentation without the prohibitive costs of proprietary SOTA
models. Furthermore, it has been widely used in recent code translation literature [ 2,29], ensuring
the representativeness of our findings.
4.1.2 Results and Analysis.Table 4 demonstrates a decisive advantage inbuild feasibilitywithin
complex environments. Baselines (C2Rust, C2SaferRust) failed to compile the majority of projects
(ICompRate≈40%), with 0% success on complex modules like host andappverify_lite due
to unresolved external macro dependencies. In contrast, our approach using Claude-Opus-4.5
achieved near-perfect stability (ICompRate=99.75%), successfully synthesizing compilable skeletons
where baselines failed completely.
Regarding retrieval depth for the DeepSeek-V3.2 backend, performance remained robust across
all settings ( 𝐾∈{ 1,3,5,10}), consistently exceeding 97%. Specifically, 𝐾= 5proved marginally

His2Trans: A Skeleton First Framework for Self Evolving C to Rust Translation with Historical Retrieval 11
Table 4. Performance comparison on domain-specific projects (RQ1).
Method ICompRate FC Unsafe Clippy
C2Rust 40.00 40.00 47.75 45.00
C2SaferRust 39.23 20.00 36.33 27.50
EvoC2Rust 0.00 0.00 2.20 228.75
Tymcrat 35.31 0.00 30.20 9.50
Ours(DeepSeek-V3.2 K = 1) 97.70 75.00 32.14 464.00
Ours(DeepSeek-V3.2 K = 3) 98.16 75.00 33.02 448.80
Ours(DeepSeek-V3.2 K = 5) 98.36 75.00 35.42 463.00
Ours(DeepSeek-V3.2 K = 10) 97.64 75.00 34.04 446.40
Ours(Claude-opus-4-5 K = 5) 99.75 75.00 37.09 315.40
osal
uhdf2_shared core_sharedhost
appverify_lite
C2Rust
C2SaferRust
EvoC2Rust
Tymcrat
Ours(DS K = 1)
Ours(DS K = 3)
Ours(DS K = 5)
Ours(DS K = 10)
Ours(Claude K = 5)100 -- 100 -- --
100 -- 96.15 -- --
-- -- -- -- --
42.86 28.57 51.85 53.28 --
100 100 100 98.3 90.2
100 100 100 100 90.8
100 100 100 99.2 92.6
100 100 100 99.2 89
100 100 100 100 98.77better
worse100
28.57
ICompRate
osal
uhdf2_shared core_sharedhost
appverify_lite
C2Rust
C2SaferRust
EvoC2Rust
Tymcrat
Ours(DS K = 1)
Ours(DS K = 3)
Ours(DS K = 5)
Ours(DS K = 10)
Ours(Claude K = 5)100 -- 100 -- --
100 -- 0 -- --
-- -- -- -- --
0 0 0 0 --
100 100 100 75 0
100 100 100 75 0
100 100 100 75 0
100 100 100 75 0
100 100 100 75 0better
worse100
0
FC
osal
uhdf2_shared core_sharedhost
appverify_lite
C2Rust
C2SaferRust
EvoC2Rust
Tymcrat
Ours(DS K = 1)
Ours(DS K = 3)
Ours(DS K = 5)
Ours(DS K = 10)
Ours(Claude K = 5)54.04 -- 41.46 -- --
45.61 -- 27.05 -- --
-- -- -- -- --
19.14 51.38 24.67 25.61 --
17.2 36.7 20.8 45 41
20.2 39 20.6 44.4 40.9
28.4 41.5 21.4 45.1 40.7
26.4 38.5 20.7 45.6 39
38.99 27.14 23.55 48.83 46.95better
worse17.2
54.04
Unsafe
osal
uhdf2_shared core_sharedhost
appverify_lite
C2Rust
C2SaferRust
EvoC2Rust
Tymcrat
Ours(DS K = 1)
Ours(DS K = 3)
Ours(DS K = 5)
Ours(DS K = 10)
Ours(Claude K = 5)23 -- 67 -- --
21 -- 34 -- --
-- -- -- -- --
4 0 4 30 --
37 69 106 721 1387
43 50 103 725 1323
53 68 103 696 1395
44 81 100 723 1284
27 41 65 423 1021better
worse0
1395
Warning
Fig. 5. Detailed performance metrics for domain-specific projects (RQ1). Darker shades denote better results;
“–” indicates compilation failure.
optimal (98.36%). Insufficient context ( 𝐾= 1) yielded 97.70%, while excessive context ( 𝐾= 10)
slightly introduced noise (97.64%), but the minimal variance confirms the framework’s insensitivity
to hyperparameter fluctuations. Although the average Functional Correctness (FC=75.00%) indicates
room for logic improvement, His2Trans effectively bridges the gap from “impossible to compile” to
“debuggable Rust code,” providing a viable starting point for migration.
Figure 5 visualizes the detailed results for each submodule (rows represent methods, columns
represent submodules). The most challenging modules (e.g., host andappverify_lite ) failed with
baseline methods primarily due to missing build context dependencies, whereas our “skeleton-first
+ repair loop” strategy maintained high compilation and test pass rates.
Why does domain-specific migration fall into a distinct category?OpenHarmony submod-
ules introduce system-level challenges that are absent in general-purpose independent benchmarks.

12 Wang, et al.
First, complex build contexts are implicit within extensive header include graphs and conditional
macros; without reconstructing this environment, the system cannot even synthesize a compilable
Rust crate. Second, many key definitions (such as platform enums, log constants, and opaque
handles) reside outside the local translation unit, rendering function-based prompts under-defined.
Third, correctness verification relies directly on the original gtest suites, which can rapidly expose
minute deviations in ABI compatibility and memory layout. Therefore, in this category, the core
bottleneck is no longer limited to the generation quality of individual function bodies, but rather
whether the system can assemble a coherent crate, thereby making compiler feedback actionable
for repair.
Impact of Skeleton-First Design.Our Rust Skeleton Construction module (see Section 2.2)
establishes and verifies a compile-safe skeleton (containing types, global variables, and compatibility
layers) prior to translating function bodies. This strategy transforms implicit C dependencies into
explicit Rust declarations, thereby significantly reducing the occurrence of non-local errors. Ulti-
mately, the repair loop receives highly localized and deterministic feedback, rather than cascading
failures stemming from missing global definitions.
fnpower_state_token_release_wake_lock (token: Option<&mut PowerStateToken >) 
{
const HDF_POWER_DYNAMIC_CTRL: u32 = /* appropriate constant value */ ;
let state_token = match token {
Some(t) => t,
None => return,    
};
if state_token.mode != HDF_POWER_DYNAMIC_CTRL {
return;
}
/* ... */
}Tymcrat
pub const HDF_POWER_DYNAMIC_CTRL: PowerManagementMode = 1;
fnPowerStateTokenReleaseWakeLock (token: *mut crate::types:: IPowerStateToken )
{
let stateToken = token as *mut crate::types:: PowerStateToken ;
if stateToken.is_null () {
return;
}
let stateToken = unsafe { &mut *stateToken };
if stateToken .mode!= crate::types ::HDF_POWER_DYNAMIC_CTRL {
return;
}
/* ... */
}His2Trans
Fig. 6. A representative definition-gap failure in host : the baseline inserts a placeholder constant (syntax
error), while our scaffold recovers a concrete domain constant and unblocks compilation.
Baseline Failure Mode Analysis.Figure 6 summarizes the primary failure modes exhibited by
the five OpenHarmony submodules across various baselines under a unified testing environment.
For C2Rust and C2SaferRust, their effectiveness relies on reconstructing a buildable Cargo project
during the conversion phase. However, in our experimental setup, this mechanism failed on
three submodules (manifested as a failure to generate Cargo.toml ), thereby directly blocking
any downstream compilation or testing processes. For LLM-centric baselines (EvoC2Rust and
Tymcrat), due to the lack of an explicit compile-safe skeleton, they failed to successfully compile
any submodules (0/5). Although these methods generated a large number of function bodies, residual
unresolved constants, type definitions, and placeholder tokens resulted in crate-level compilation
failures.
Case Study: Constant Recovery Prevents Cascading Failures.Figure 6 compares the output
of Tymcrat and our method on the host wakelock release routine. Tymcrat left a placeholder
constant initializer within the function body, which caused a syntax-level compilation block,
subsequently preventing the compiler-guided repair mechanism from initiating. Our Skeleton
Construction module pre-recovered the enum-based constant definitions, thereby eliminating
compilation barriers. This enabled the subsequent repair loop to focus on resolving local logical
defects rather than noise produced by missing definitions.
4.1.3 Summary.In domain-specific projects, His2Trans achieved a commanding 99.75% Incremen-
tal Compilation Pass Rate, effectively resolving the build failures that plagued baselines (which
failed completely on complex modules like host ). Coupled with a 75.00% Functional Correctness,

His2Trans: A Skeleton First Framework for Self Evolving C to Rust Translation with Historical Retrieval 13
Table 5. Performance comparison on general-purpose benchmarks (RQ2).
Method ICompRate FC Unsafe Clippy
C2Rust 100.00 100.00 69.48 134.70
SmartC2Rust 99.38 100.00 – –
RUSTINE 100.00 100.00 – –
PTRMAPPER 100.00 82.44 – –
C2SaferRust 53.12 62.74 47.56 163.70
EvoC2Rust 90.86 60.00 6.59 464.70
Tymcrat 62.46 10.00 4.59 2.50
Ours(DeepSeek-V3.2) 97.44 45.33 37.86 130.70
Ours(Claude-opus-4-5) 89.75 48.31 45.84 97.20
htqsortquadtree buffer rgbaurlparser genannavl bzip2 zopfli
C2Rust
SmartC2Rust
RUSTINE
PTRMAPPER
C2SaferRust
EvoC2Rust
Tymcrat
Ours(DS)
Ours(Claude)100 100 100 100 100 100 100 100 100 100
99.18 99 -- -- 99.2 99.77 -- 99.93 -- 99.22
100 100 100 100 100 100 100 100 100 100
100 -- 100 100 100 100 100 100 -- --
-- 66.67 12.5 95.65 38.46 95 83.33 51.72 87.91 --
100 100 70.8 100 100 40 100 100 97.8 100
-- 100 83.3 69.6 -- 85 75 37.9 80.2 93.6
100 100 100 91.3 100 96.15 100 100 87.91 99.03
100 100 100 100 100 100 100 98.28 -- 99.25better
worse100
12.5
ICompRate
htqsortquadtree buffer rgbaurlparser genannavl bzip2 zopfli
C2Rust
SmartC2Rust
RUSTINE
PTRMAPPER
C2SaferRust
EvoC2Rust
Tymcrat
Ours(DS)
Ours(Claude)100 100 100 100 100 100 100 100 100 100
100 100 -- -- 100 100 -- 100 -- 100
100 100 100 100 100 100 100 100 100 100
100 -- 87.5 75 100 36.8 77.8 100 -- --
-- 66.7 75 85.7 100 0 100 100 0 --
100 100 100 100 100 0 100 0 0 0
-- 100 0 0 -- 0 0 0 0 0
100 100 0 0 70 0 83.33 100 0 0
100 100 0 71.43 70 0 41.67 100 -- 0better
worse100
0
FC
htqsortquadtree buffer rgbaurlparser genannavl bzip2 zopfli
C2Rust
SmartC2Rust
RUSTINE
PTRMAPPER
C2SaferRust
EvoC2Rust
Tymcrat
Ours(DS)
Ours(Claude)51.85 74.42 61.19 67.44 26.01 84.45 82.61 82.02 83.42 81.41
-- -- -- -- -- -- -- -- -- --
-- -- -- -- -- -- -- -- -- --
-- -- -- -- -- -- -- -- -- --
-- 20 56.99 31.96 24.11 65.14 63.41 63.81 68.8 --
9.47 3.28 10.82 8.12 4.07 7.51 8.34 11.18 1.71 1.37
-- 0 0.35 19.78 -- 6.73 1.28 8.5 0.74 8.49
14.02 21.77 43.98 41.48 29.86 34.92 42.56 59.06 42.95 47.99
13.76 21.77 54.22 46.61 55.74 48.31 33.03 66.4 -- 58.39better
worse0
84.45
Unsafe
htqsortquadtree buffer rgbaurlparser genannavl bzip2 zopfli
C2Rust
SmartC2Rust
RUSTINE
PTRMAPPER
C2SaferRust
EvoC2Rust
Tymcrat
Ours(DS)
Ours(Claude)5 3 42 29 163 134 26 78 230 637
-- -- -- -- -- -- -- -- -- --
-- -- -- -- -- -- -- -- -- --
-- -- -- -- -- -- -- -- -- --
-- 5 54 29 164 66 20 48 577 --
425 426 424 423 447 424 444 781 425 428
-- 3 -- -- -- -- -- -- -- --
0 12 19 31 190 101 55 144 79 676
0 12 2 19 30 55 5 281 -- 505better
worse0
781
Warning
Fig. 7. Heatmap visualization of detailed performance metrics across general-purpose benchmarks (RQ2).
Darker shades indicate superior performance (higher Compilation/FC, lower Unsafe/warning).
these results quantitatively demonstrate that His2Trans overcomes the “dependency hell” barrier
of general-purpose benchmarks and is competent for complex engineering migration tasks.
4.2 RQ2: General-Domain Effectiveness
4.2.1 Design.RQ2 evaluates the general-purpose effectiveness on ten open-source C benchmarks
(Table 3). We operated the framework with two distinct model backends under an identical pipeline
configuration to assess model sensitivity. We compared our approach against rule-based baselines
as well as representative LLM-based methods.
4.2.2 Results and Analysis.Table 5 illustrates a distinct trade-off. While C2Rust achieves 100%
Functional Correctness by mechanically preserving C semantics, it retains a high Unsafe Ratio
(69.48%). Our approach ( Claude-Opus-4.5 ) prioritizes idiomatic safety, reducing the Unsafe Ratio
to45.84%and achieving the best maintainability among automated tools (lowest warnings:97.20).
Interestingly, while DeepSeek-V3.2 achieved a higher compilation pass rate (97.44% vs. 89.75%), it
produced significantly more warnings (130.70), suggesting that stronger reasoning models ( Claude )
may trade compilation aggressiveness for stricter adherence to Rust idioms.
Crucially, the performance variance across methods stems from our rigorous evaluation protocol.
We enforcezero human interventionand mandate the reuse of original test suites. For instance,
in compression utilities like bzip2 andzopfli , we require the translated code to produce bit-exact

14 Wang, et al.
output matches. High scores reported by baselines often result from relaxed constraints that hide
defects: (1)SmartC2Rustutilizes simplified checks ( ≈20 assertions) with limited assertions, failing
to detect internal state errors; (2)Rustineincorporates approximately four hours of manual repair
per project, violating the fully automated premise; (3)C2SaferRustfailed to optimize zopfli and
reverted to the unsafe C2Rust code. Consequently, our results reflect the realistic limit of fully
automated translation under strict verification standards.
Trade-offs and Consequences of Baseline Strategies.In general-purpose benchmarks, al-
though multiple systems are capable of generating compilable crates, they exhibit significant
differences in technical trade-offs. First, rule-based transpilers (such as C2Rust) prioritize com-
pilability by mechanically preserving C memory management patterns; however, this implies
retaining a high unsafe ratio (69.48%) and ignoring Rust idioms. Second, LLM-based baselines
show distinct capability skews: conservative repair-oriented systems like C2SaferRust attempt
to enforce safety constraints but struggle to preserve logic, achieving only 62.74% Functional
Correctness under the unified testing environment; meanwhile, generation-oriented systems like
EvoC2Rust and Tymcrat exhibit a “safety illusion,” where their exceptionally low unsafe ratios
(below 7%) are largely artifacts of under-translation, as they either fail to implement complex
functional logic (e.g., Tymcrat’s low 10% FC) or generate unmaintainable code laden with warnings
(e.g., EvoC2Rust’s 425.50 warning count). In contrast, His2Trans achieves a pragmatic balance: the
compile-safe skeleton establishes build stability, and while strictly adhering to safe idioms results
in moderate automated Functional Correctness (48.31%), our approach successfully reduces the
unsafe footprint by 23.6 percentage points compared to the mechanical baseline and achieves
the highest code maintainability (lowest warnings: 97.20), providing the cleanest foundation for
subsequent refinement.
It is worth noting that our approach’s retained unsafe ratio (45.84%) primarily stems from the
necessity of maintaining binary compatibility with untranslated C modules. These unsafe blocks
are predominantly localized at FFI boundaries and #[repr(C)] structures to ensure memory-layout
consistency, which is a deliberate trade-off in an evolutionary migration rather than a deficiency in
translation logic.
type BufferT = usize;
fnbuffer_fill (self_: &mut BufferT, c: u8) {
std::ptr::write_bytes (self_.data.as_mut_ptr (), c, self_. len);
}
fnbuffer_new_with_size (n: usize) -> Result<Box< BufferT>, ()> {
/* ... */
let mut buffer = Box::new( BufferT {
len: n,
data: data_ptr ,
alloc: data_ptr ,
});
Ok(buffer )
}Tymcrat
#[repr(C)]
pub struct buffer_t {
pub len: size_t,
pub alloc: *mut std:: ffi::c_char,
pub data: *mut std:: ffi::c_char,
}
pub extern "C" fnbuffer_fill (self_: *mut crate::types:: buffer_t , c: ::core:: ffi::c_int) {
if self_. is_null() { return; }
unsafe {
libc::memset((*self_).data as *mut ::core:: ffi::c_void, c, (*self_). lenas usize);
}
}His2Trans
Fig. 8. On buffer , the baseline collapses a struct into an integer handle, which breaks field-level reasoning;
our scaffold recovers the struct layout, enabling type-correct function bodies.
Case Study: Struct Layout Recovery vs. Pointer-as-Integer Collapse.Figure 8 illustrates a
representative failure case regarding buffer type recovery. Tymcrat collapsed the C struct into an
integer handle but subsequently treated it as a record, causing field access type errors that prevented
compilation. Our Stage 1 recovered the struct layout within the scaffold, rendering function body
translation and compilation-guided repair for typed fields a well-posed problem.
4.2.3 Summary.In general-purpose open-source benchmarks, our framework maintained a high
compilation success rate and achieved robust Functional Correctness.

His2Trans: A Skeleton First Framework for Self Evolving C to Rust Translation with Historical Retrieval 15
Table 6. Experimental Configurations and Results for RQ3.
Config IDConfiguration Setup Performance Metrics
API Rule Frag. Rule Repair Limit ICompRate FC AvgRepair
𝐵𝑎𝑠𝑒-1𝑆ℎ𝑜𝑡None None 0 68.29 100.00 0.00
𝐵𝑎𝑠𝑒-𝑅𝑒𝑝None None 5 91.00 100.00 2.10
𝑃𝑟𝑒𝑑-1𝑆ℎ𝑜𝑡Retrieved Retrieved 0 71.03 100.00 0.00
𝑃𝑟𝑒𝑑-𝑅𝑒𝑝Retrieved Retrieved 5 85.45 100.00 2.05
𝐺𝑇-𝐴𝑃𝐼Ground TruthRetrieved 5 83.60 100.00 1.70
𝐺𝑇-𝐹𝑟𝑎𝑔RetrievedGround Truth5 89.15 100.00 2.03
𝐺𝑇-𝐹𝑢𝑙𝑙Ground Truth Ground Truth5 88.95 100.00 1.97
4.3 RQ3: Mapping Rules and Repair Loops for Function Body Translation
4.3.1 Design.RQ3 aims to investigate how function body translation varies under differentmap-
ping rule channelsandrepair budgetswithin domain-specific projects. We kept the underlying
build context and toolchain constant, varying only the target mechanisms, with the model set to
Claude-Opus-4.5.
Due to the prohibitive token costs of larger modules, we restricted this ablation study to a
representative subset ( osal ,uhdf2_shared , and core_shared ). Consequently, the reported metrics
reflect this subset, where our full method achieves 100% correctness, consistent with RQ1.
We designed a controlled experiment with seven configurations based on three dimensions: (1)
API Rule, referring to where the API-Level rules come from (None, Retrieved via RAG, or Ground
Truth); (2)Fragment Rule, indicating where the Fragment-Level rules come from; and (3)Repair
Limit, representing the maximum number of compilation-guided repair iterations allowed (0 for
one-shot, 5 for repair loop).
The left side of Table 6 details the specific settings for each configuration. “Retrieved” denotes
rules retrieved from our constructed knowledge base using similarity search. “Ground Truth” refers
to oracle rules extracted from manual translations of the test projects (mined via theGeneral Case
strategy described in Section 2.1).
4.3.2 Results and Analysis.The Necessity of Iterative Repair.Our analysis of the average per-
formance across submodules (right side of Table 6) confirms that one-shot translation is insufficient
for production-level code generation. The one-shot baseline ( 𝐵𝑎𝑠𝑒− 1𝑆ℎ𝑜𝑡) achieved an average
Compilation Rate of only68.29%, struggling to satisfy Rust’s strict ownership rules in a single
pass. Introducing the compilation-guided repair loop ( 𝐵𝑎𝑠𝑒−𝑅𝑒𝑝 ) dramatically revitalized these
failed cases, boosting the average success rate to91.00%. This finding empirically validates that
the compiler feedback loop is not an optional optimization but acritical dependencyfor making
LLM-generated Rust code compilable.
Knowledge-Driven Efficiency.Comparing the repair costs, the injection of domain knowledge
demonstrated clear efficiency gains. While the final compilation ceiling was largely determined
by the repair loop (hovering around 89-91%), the knowledge base significantly reduced the effort
to get there. As shown in Table 6, the average repair rounds dropped from2.10( 𝐵𝑎𝑠𝑒−𝑅𝑒𝑝 ) to
1.70(𝐺𝑇−𝐴𝑃𝐼 ). This indicates that high-quality mapping rules act as a “warm start” for the
model. Specifically, 𝐺𝑇-𝐴𝑃𝐼 achieved the lowest repair cost (1.70 rounds), suggesting that resolving
external interface mapping is the most critical factor in unblocking the LLM’s initial reasoning
path. It is worth noting that 𝐺𝑇-𝐹𝑢𝑙𝑙 (1.97) incurred slightly higher repair costs than 𝐺𝑇-𝐴𝑃𝐼 , likely
because fragment-level rules introduce more complex syntactic structures that, while idiomatically
correct, require finer ownership adjustments during compilation.

16 Wang, et al.
Case Study: Domain-Specific Knowledge-Guided Solutions.To concretely demonstrate how
knowledge guides translation and reduces repair costs, Figure 9 presents two representative cases
from OpenHarmony. Each case displays (i) the injected structured knowledge and (ii) representative
translation results under 𝐵𝑎𝑠𝑒− 1𝑆ℎ𝑜𝑡 (no knowledge, no repair), 𝐵𝑎𝑠𝑒−𝑅𝑒𝑝 (no knowledge with
repair loop), and𝐺𝑇−𝐹𝑢𝑙𝑙(Ground Truth knowledge with repair loop).
/* ... */
// Fail: Dereferencing null to get offset is UB in Rust
letoffset=unsafe{
&(*(std::ptr::null::<crate::types::IoServiceStatusListener >())).svcstatListener as*const_asusize
};
letioserv_listener =unsafe{
(listener as*mutu8).sub(offset) as*mutcrate::types::IoServiceStatusListener
};
/* ... */
/* ... */
// Success: Direct usage of Rust native macro
letoffset=std::mem::offset_of !(crate::types::IoServiceStatusListener , svcstatListener );
letioserv_listener =unsafe{
(listener as*mutu8).offset(-(offsetasisize)) as*mutcrate::types::IoServiceStatusListener
};
/* ... *//* ... */
// Repair: Clumsy workaround to avoid null -deref
letoffset={
letdummy=std::mem::MaybeUninit ::<crate::types::IoServiceStatusListener >::uninit();
letbase_ptr =dummy.as_ptr();
unsafe{
letfield_ptr =std::ptr::addr_of!((*base_ptr ).svcstatListener );
(field_ptr as*constu8).offset_from (base_ptr as*constu8) asusize
}
};
/* ... */pubextern"C"fnHdfIoServiceRemove (service:*mutcrate::types::HdfIoService ) {
extern"C"{
fnHdfIoServiceAdapterRemove (...);
// Error: Hallucinated static causing duplicate symbol
staticHdfIoServiceAdapterRemove :Option<unsafeextern"C" fn(...)>;
}
/* ... */
}
pubextern"C"fnHdfIoServiceRemove (service:*mutcrate::types::HdfIoService ) {
extern"C"{
fnHdfIoServiceAdapterRemove (...);
}
// Success: Generated correct check immediately
unsafe{
letfunc_ptr :*const() =HdfIoServiceAdapterRemove as*const();
if!func_ptr .is_null() {
HdfIoServiceAdapterRemove (service);
}
}
}pubextern"C"fnHdfIoServiceRemove (service:*mutcrate::types::HdfIoService ) {
extern"C"{
fnHdfIoServiceAdapterRemove (...);
}
// Repair: Check function address for weak linking
unsafe{
letfunc_ptr :*const() =HdfIoServiceAdapterRemove as*const();
if!func_ptr .is_null() {
HdfIoServiceAdapterRemove (service);
}
}
}
API-Level rules:
-OsalMemFree ->OsalMemFree (Free memory via OSAL allocator )
Fragment -Level rules :
-C: CONTAINER_OF (p, T, field)
-Rust: (pas*mutu8).offset(-(offset_of !(T, field) asisize)) as*mutTAPI-Level rules :
-HdfIoServiceAdapterRemove ->HdfIoServiceAdapterRemove (Remove io service via adapter)voidIoServiceStatusListenerFree (structServiceStatusListener *listener ){
// The macro implies a layout offset calculation
structIoServiceStatusListener *ioservListener =CONTAINER_OF (listener, structIoServiceStatusListener , svcstatListener );
OsalMemFree (ioservListener );
}voidHdfIoServiceRemove (structHdfIoService *service){
if(HdfIoServiceAdapterRemove !=NULL) {
HdfIoServiceAdapterRemove (service);
}
}
Repair 1 timeRepair 3 times
Case 1 Case 2
C code Base -1Shot output Base -Rep output GT-Full output
Repair 1 times
Repair 0 time
Fig. 9. Comparison of baseline vs. knowledge-guided translation ( 𝐺𝑇-𝐹𝑢𝑙𝑙 ). Case studies include Memory
Layout Recovery (Left) and Weak Symbol Linking (Right).
In the case of memory layout recovery shown on the left of Figure 9, under the 𝐵𝑎𝑠𝑒− 1𝑆ℎ𝑜𝑡
configuration without prior knowledge, the model attempted to mimic the C implementation by
dereferencing a null pointer to calculate member offsets. This operation constitutes Undefined
Behavior (UB) in Rust and causes the compiler to report errors during the constant evaluation phase.
After entering the 𝐵𝑎𝑠𝑒−𝑅𝑒𝑝 repair loop, although the system bypassed the null pointer check using
MaybeUninit , it generated verbose and obscure pointer arithmetic code that only barely achieved
functional alignment. In contrast, the 𝐺𝑇−𝐹𝑢𝑙𝑙 configuration benefited from the injected code
fragment patterns, directly selecting the Rust standard library’s native std::mem::offset_of!
macro. This approach not only accurately captured the original semantics but also eliminated the
memory safety risks associated with manual offset calculation, yielding idiomatic code.
In the weak symbol linking case shown on the right of Figure 9, the 𝐵𝑎𝑠𝑒− 1𝑆ℎ𝑜𝑡 configuration, due
to a lack of context, misinterpreted the null-check logic of the function pointer as requiring a new
variable. This resulted in the simultaneous declaration of an external function and a static variable
with the same name, leading to a symbol redefinition error. In the 𝐵𝑎𝑠𝑒−𝑅𝑒𝑝 configuration, the
repair module underwent multiple trial-and-error attempts before finally deducing the correct logic
of casting the function item to a raw pointer for the null check. Under the 𝐺𝑇−𝐹𝑢𝑙𝑙 configuration,
the injected API-Level rules clarified the semantic intent of the weak symbol in advance, guiding the
model to adopt the correct function address check pattern ( as *const () ) in the very first generation
round. This prior guidance avoided syntactic conflicts in symbol definition and significantly reduced
the iteration cost required to recover from errors.

His2Trans: A Skeleton First Framework for Self Evolving C to Rust Translation with Historical Retrieval 17
Table 7. Impact of knowledge accumulation on unseen modules (RQ4).
Setting ICompRate FC AvgRepair
Base KB only 77.91 37.50 1.39
Base KB + Accumulated KB95.7037.500.57
Relative Improvement +22.83% – -59.0%
4.3.3 Summary.RQ3 demonstrates that a compile-safe skeleton is a prerequisite for stable in-
cremental translation; under this setting, mapping rules and compilation-guided repair primarily
influence therepair efficiency (AvgRepair)on smaller modules.
4.4 RQ4: Knowledge Accumulation
4.4.1 Design.To evaluate whether Knowledge Accumulation can yield transferable benefits, we
partitioned the OpenHarmony dataset into two mutually exclusive subsets (Group A and Group B).
Regarding the grouping strategy, we assigned osal ,uhdf2_shared , and core_shared to Group A.
Since His2Trans demonstrated optimal translation quality on these modules, they serve as high-
confidence sources of accumulated knowledge for subsequent experiments. Group B comprises
host andappverify_lite , serving as the target set for evaluation. In the evaluation process, we
first executed the framework on Group A to extract an Accumulated Knowledge Base (KB), and
subsequently compared the effectiveness of three configurations on Group B: (1) No Accumulation
(using only the baseline KB); (2) Accumulated KB Only; and (3) Merged KB (baseline + accumulated).
Specifically,No Accumulationrefers to using the initial knowledge base constructed in Section 2.1.
We summarize the specific migration benefit metrics for these two representative submodules
(appverify_liteandhost) in a dedicated table.
4.4.2 Results and Analysis.The accumulated knowledge experiment highlights significant gains in
bothbuild feasibilityandrepair efficiencywhen migrating to unseen targets. As reported in
Table 7, the Accumulated KB (Base + Sedimented) drastically improved the Incremental Compilation
Pass Rate from 77.91% to95.70%. This surge indicates that historical patterns effectively cover the
"long-tail" build errors found in new modules.
Simultaneously, the computational cost dropped drastically. The AvgRepair metric decreased
from1.39with no accumulation to0.57using the merged knowledge base—a reduction of approxi-
mately59%. This suggests that accumulated domain knowledge effectively “prunes” the search
space, allowing the repair module to resolve compilation errors with significantly fewer API queries
and iterations, effectively lowering the barrier for migrating new, complex modules.
4.4.3 Summary.Experimental results demonstrate that accumulated domain knowledge can sig-
nificantly reduce the workload of compilation-guided repair. Although the configurations exhibited
diminishing returns in terms of accuracy metrics, the substantial improvement in repair efficiency
(an approximate 60% reduction in repair rounds) proves that this mechanism can maintain equally
high effectiveness at a lower computational cost when migrating to unseen projects.
5 Threats to Validity
External Validity: Domain Bias.Our knowledge base is primarily mined from the OpenHarmony
ecosystem, potentially biasing rules toward embedded systems. While general-purpose benchmarks
(RQ2) demonstrate transferability, performance on distinct domains (e.g., high-level web services)
with different API patterns remains to be fully verified.

18 Wang, et al.
Internal Validity: Test Suite Adequacy.We rely on legacy test suites for verification. If original
tests are sparse, the Functional Correctness metric may overlook subtle Rust-specific logic errors,
although strict compilation checks partially mitigate this by catching borrow checker violations.
6 Related Work
Rust offers low-level control with performance comparable to C/C++ while providing strong
compile-time memory-safety guarantees through ownership and borrowing. These properties have
motivated substantial interest in migrating legacy systems code from C/C++ to Rust. Existing work
on C-to-Rust can be broadly categorized into rule-based, LLM-based, and hybrid approaches.
Rule-based methods.Rule-based approaches translate C to Rust via compiler front-ends and
handcrafted rules, typically using syntax-level rewrites to preserve structural correspondence. A
representative tool is C2Rust [ 7], built on Clang, which translates largely via C–Rust FFI and often
relies on unsafe , resulting in unsafe-heavy and non-idiomatic code. Subsequent work improves
safety and idiomaticity with stronger analyses and specialization: Emre et al. [ 3] use compiler-
feedback repair, Ling et al. [ 15] introduce syntactic heuristics, and Crown [ 36] applies static
ownership tracking for pointer conversion. Hong & Ryu [ 8,9,11] further target recurring C
idioms and API patterns. However, rule-based pipelines remain engineering-intensive and brittle
to project-specific APIs and build configurations, limiting scalability to large systems.
LLM-based methods.LLM-based approaches prompt an LLM to synthesize Rust from C, sometimes
augmented with tests or specifications. While more flexible than transpilers and often better at
capturing Rust idioms, LLM outputs can be incorrect and hard to trace, and high-quality parallel
C–Rust data remains limited [ 24]. Prior work therefore emphasizes verification/feedback and
structured context. FLOURINE combines fuzzing with error-message feedback for iterative repair,
though type serialization remains challenging [ 5]; UniTrans leverages test generation for iterative
refinement [ 33]. Others inject richer context: IRENE integrates rule-augmented retrieval and
structured summarization for function-level translation [ 18], and additional systems incorporate
static specifications and I/O tests [21] or data-flow graphs to improve type migration [31].
Hybrid methods.Hybrid approaches combine rule-based scaffolding with LLM refinement: rules
provide a deterministic, type-checked structure, while LLMs improve idiomaticity and reduce
unsafe usage. C2SaferRust [ 20] refactors C2Rust output [ 7] in test-verified chunks, SACTOR [ 41]
applies a two-stage LLM translation-and-refinement workflow with static/FFI-based validation,
and EvoC2Rust [ 29] migrates projects via a compilable skeleton and iterative stub replacement
with build-error repair.
7 Conclusion
In this paper, we introducedHis2Trans, an automated framework designed to overcome structural
and knowledge barriers in industrial C-to-Rust migration. By synergizing a strictly typed Project-
Level Skeleton with a Self-Evolving Knowledge Base, our approach effectively resolves “dependency
hell” and bridges the domain-specific knowledge gap. Evaluations on OpenHarmony and general
benchmarks demonstrate its superiority, achieving a 99.75% incremental compilation pass rate
while reducing repair costs by approximately 60% through knowledge accumulation. These results
suggest that future migration tools must prioritize deterministic build contexts and accumulable
domain knowledge, for whichHis2Transprovides a robust and practical foundation.
Data Availability
The raw data, test code, and framework source code used in this study are available at: https:
//anonymous.4open.science/r/His2Trans-F1F0/.

His2Trans: A Skeleton First Framework for Self Evolving C to Rust Translation with Historical Retrieval 19
References
[1]Xuemeng Cai, Jiakun Liu, Xiping Huang, Yijun Yu, Haitao Wu, Chunmiao Li, Bo Wang, Imam Nur Bani Yusuf,
and Lingxiao Jiang. 2025. RustMap: Towards Project-Scale C-to-Rust Migration via Program Analysis and LLM. In
Engineering of Complex Computer Systems: 29th International Conference, ICECCS 2025, Hangzhou, China, July 2–4,
2025, Proceedings(Hangzhou, China). Springer-Verlag, Berlin, Heidelberg, 283–302. doi:10.1007/978-3-032-00828-2_16
[2]Saman Dehghan, Tianran Sun, Tianxiang Wu, Zihan Li, and Reyhaneh Jabbarvand. 2025. Translating Large-Scale C
Repositories to Idiomatic Rust. arXiv:2511.20617 [cs.SE] https://arxiv.org/abs/2511.20617
[3]Mehmet Emre, Peter Boyland, Aesha Parekh, Ryan Schroeder, Kyle Dewey, and Ben Hardekopf. 2023. Aliasing limits
on translating C to safe Rust.Proceedings of the ACM on Programming Languages7, OOPSLA1 (2023), 551–579.
[4]Mehmet Emre, Ryan Schroeder, Kyle Dewey, and Ben Hardekopf. 2021. Translating C to safer Rust.Proc. ACM Program.
Lang.5, OOPSLA, Article 121 (Oct. 2021), 29 pages. doi:10.1145/3485498
[5]Hasan Ferit Eniser, Hanliang Zhang, Cristina David, Meng Wang, Maria Christakis, Brandon Paulsen, Joey Dodds,
and Daniel Kroening. 2025. Towards Translating Real-World Code with LLMs: A Study of Translating to Rust.
arXiv:2405.11514 [cs.SE] https://arxiv.org/abs/2405.11514
[6]Muhammad Farrukh, Smeet Shah, Baris Coskun, and Michalis Polychronakis. 2025. SafeTrans: LLM-assisted Transpi-
lation from C to Rust. arXiv:2505.10708 [cs.CR] https://arxiv.org/abs/2505.10708
[7] Galois, Inc. 2018. C2Rust. https://galois.com/blog/2018/08/c2rust/. Accessed: 2026-01-27.
[8]Jaemin Hong and Sukyoung Ryu. 2024. Don’t write, but return: Replacing output parameters with algebraic data types
in c-to-rust translation.Proceedings of the ACM on Programming Languages8, PLDI (2024), 716–740.
[9]Jaemin Hong and Sukyoung Ryu. 2024. To Tag, or Not to Tag: Translating C’s Unions to Rust’s Tagged Unions. In
Proceedings of the 39th IEEE/ACM International Conference on Automated Software Engineering. 40–52.
[10] Jaemin Hong and Sukyoung Ryu. 2024. Type-migrating C-to-Rust translation using a large language model.Empirical
Software Engineering30 (10 2024). doi:10.1007/s10664-024-10573-2
[11] Jaemin Hong and Sukyoung Ryu. 2025. Forcrat: Automatic I/O API Translation from C to Rust via Origin and Capability
Analysis.arXiv preprint arXiv:2506.01427(2025).
[12] Xinyi Hou, Yanjie Zhao, Yue Liu, Zhou Yang, Kailong Wang, Li Li, Xiapu Luo, David Lo, John Grundy, and Haoyu
Wang. 2024. Large Language Models for Software Engineering: A Systematic Literature Review.ACM Trans. Softw.
Eng. Methodol.33, 8, Article 220 (Dec. 2024), 79 pages. doi:10.1145/3695988
[13] Ralf Jung, Jacques-Henri Jourdan, Robbert Krebbers, and Derek Dreyer. 2017. RustBelt: securing the foundations of the
Rust programming language.Proc. ACM Program. Lang.2, POPL, Article 66 (Dec. 2017), 34 pages. doi:10.1145/3158154
[14] Ralf Jung, Jacques-Henri Jourdan, Robbert Krebbers, and Derek Dreyer. 2021. Safe systems programming in Rust.
Commun. ACM64, 4 (March 2021), 144–152. doi:10.1145/3418295
[15] Michael Ling, Yijun Yu, Haitao Wu, Yuan Wang, James R Cordy, and Ahmed E Hassan. 2022. In Rust we trust:
a transpiler from unsafe C to safer Rust. InProceedings of the ACM/IEEE 44th international conference on software
engineering: companion proceedings. 354–355.
[16] Tianyang Liu, Canwen Xu, and Julian McAuley. 2024. RepoBench: Benchmarking Repository-Level Code Auto-
Completion Systems. InInternational Conference on Learning Representations, B. Kim, Y. Yue, S. Chaudhuri, K. Fragki-
adaki, M. Khan, and Y. Sun (Eds.), Vol. 2024. 47832–47850. https://proceedings.iclr.cc/paper_files/paper/2024/file/
d191ba4c8923ed8fd8935b7c98658b5f-Paper-Conference.pdf
[17] Yuchen Liu, Junhao Hu, Yingdi Shan, Ge Li, Yanzhen Zou, Yihong Dong, and Tao Xie. 2025. LLMigrate: Transforming
"Lazy" Large Language Models into Efficient Source Code Migrators. arXiv:2503.23791 [cs.PL] https://arxiv.org/abs/
2503.23791
[18] Feng Luo, Kexing Ji, Cuiyun Gao, Shuzheng Gao, Jia Feng, Kui Liu, Xin Xia, and Michael R. Lyu. 2025. Integrating
Rules and Semantics for LLM-Based C-to-Rust Translation. arXiv:2508.06926 [cs.SE] https://arxiv.org/abs/2508.06926
[19] Matt Miller. 2019. Trends, Challenges, and Strategic Shifts in the Software Vulnerability Mitigation
Landscape. Presentation at BlueHat IL Conference. https://github.com/microsoft/MSRC-Security-
Research/blob/master/presentations/2019_02_BlueHatIL/2019_01%20-%20BlueHatIL%20-%20Trends%2C%
20challenge%2C%20and%20shifts%20in%20software%20vulnerability%20mitigation.pdf Microsoft Security
Response Center (MSRC).
[20] Vikram Nitin, Rahul Krishna, Luiz Lemos do Valle, and Baishakhi Ray. 2025. C2SaferRust: Transforming C Projects
into Safer Rust with NeuroSymbolic Techniques. arXiv:2501.14257 [cs.SE] https://arxiv.org/abs/2501.14257
[21] Vikram Nitin, Rahul Krishna, and Baishakhi Ray. 2025. SpecTra: Enhancing the Code Translation Ability of Language
Models by Generating Multi-Modal Specifications. arXiv:2405.18574 [cs.SE] https://arxiv.org/abs/2405.18574
[22] Miguel Ojeda et al .2022. Rust for Linux.Linux Kernel Mailing List(2022). https://rust-for-linux.com/ Rust support
merged in Linux 6.1.
[23] OpenAtom Foundation. [n. d.]. OpenHarmony: A distributed operating system for all scenarios. https://gitee.com/
openharmony.

20 Wang, et al.
[24] Guangsheng Ou, Mingwei Liu, Yuxuan Chen, Xin Peng, and Zibin Zheng. 2024. Repository-level code translation
benchmark targeting rust.arXiv preprint arXiv:2411.13990(2024).
[25] Baptiste Roziere, Marie-Anne Lachaux, Lowik Chanussot, and Guillaume Lample. 2020. Unsupervised Translation of
Programming Languages. InAdvances in Neural Information Processing Systems, H. Larochelle, M. Ranzato, R. Hadsell,
M.F. Balcan, and H. Lin (Eds.), Vol. 33. Curran Associates, Inc., 20601–20611. https://proceedings.neurips.cc/paper_
files/paper/2020/file/ed23fbf18c2cd35f8c7f8de44f85c08d-Paper.pdf
[26] Momoko Shiraishi, Yinzhi Cao, and Takahiro Shinagawa. 2024. SmartC2Rust: Iterative, Feedback-Driven C-to-Rust
Translation via Large Language Models for Safety and Equivalence. https://api.semanticscholar.org/CorpusID:
272689595
[27] The Chromium Project. 2020. Memory Safety. https://www.chromium.org/Home/chromium-security/memory-safety/
Accessed: 2026-01-27.
[28] The White House. 2024.Back to the Building Blocks: A Path Toward Secure and Measurable Software. Technical
Report. The White House. https://bidenwhitehouse.archives.gov/wp-content/uploads/2024/02/Final-ONCD-Technical-
Report.pdf
[29] Chaofan Wang, Tingrui Yu, Beijun Shen, Jie Wang, Dong Chen, Wenrui Zhang, Yuling Shi, Chen Xie, and Xiaodong
Gu. 2026. EvoC2Rust: A Skeleton-guided Framework for Project-Level C-to-Rust Translation. arXiv:2508.04295 [cs.SE]
https://arxiv.org/abs/2508.04295
[30] Feng Wang, Yuqing Li, and Han Xiao. 2025. jina-reranker-v3: Last but Not Late Interaction for Listwise Document
Reranking. arXiv:2509.25085 [cs.CL] https://arxiv.org/abs/2509.25085
[31] Qingxiao Xu and Jeff Huang. 2025. Optimizing Type Migration for LLM-Based C-to-Rust Translation: A Data Flow
Graph Approach. InProceedings of the 14th ACM SIGPLAN International Workshop on the State Of the Art in Program
Analysis. 8–14.
[32] Aidan Z. H. Yang, Yoshiki Takashima, Brandon Paulsen, Josiah Dodds, and Daniel Kroening. 2024. VERT: Verified
Equivalent Rust Transpilation with Large Language Models as Few-Shot Learners. arXiv:2404.18852 [cs.PL] https:
//arxiv.org/abs/2404.18852
[33] Zhen Yang, Fang Liu, Zhongxing Yu, Jacky Wai Keung, Jia Li, Shuo Liu, Yifan Hong, Xiaoxue Ma, Zhi Jin, and Ge Li.
2024. Exploring and unleashing the power of large language models in automated code translation.Proceedings of the
ACM on Software Engineering1, FSE (2024), 1585–1608.
[34] Zhiqiang Yuan, Wenjun Mao, Zhuo Chen, Xiyue Shang, Chong Wang, Yiling Lou, and Xin Peng. 2025. Project-Level C-to-
Rust Translation via Synergistic Integration of Knowledge Graphs and Large Language Models. arXiv:2510.10956 [cs.SE]
https://arxiv.org/abs/2510.10956
[35] Hanliang Zhang, Cristina David, Yijun Yu, and Meng Wang. 2023. Ownership Guided C to Rust Translation. In
Computer Aided Verification: 35th International Conference, CAV 2023, Paris, France, July 17–22, 2023, Proceedings, Part
III(Paris, France). Springer-Verlag, Berlin, Heidelberg, 459–482. doi:10.1007/978-3-031-37709-9_22
[36] Hanliang Zhang, Cristina David, Yijun Yu, and Meng Wang. 2023. Ownership guided C to Rust translation. In
International Conference on Computer Aided Verification. Springer, 459–482.
[37] Ruxin Zhang, Shanxin Zhang, and Linbo Xie. 2025. A systematic exploration of C-to-rust code translation based
on large language models: prompt strategies and automated repair.Automated Software Engineering33 (10 2025).
doi:10.1007/s10515-025-00570-0
[38] Ruxin Zhang, Shanxin Zhang, and Linbo Xie. 2025. A systematic exploration of C-to-rust code translation based
on large language models: prompt strategies and automated repair.Automated Software Engineering33, 1 (2025), 21.
doi:10.1007/s10515-025-00570-0
[39] Han Zhou, Yu Luo, Mengtao Zhang, and Dianxiang Xu. 2025. C2RustTV: An LLM-based Framework for C to Rust
Translation and Validation. In2025 IEEE 49th Annual Computers, Software, and Applications Conference (COMPSAC).
1254–1259. doi:10.1109/COMPSAC65507.2025.00158
[40] Shuyan Zhou, Uri Alon, Frank F. Xu, Zhengbao Jiang, and Graham Neubig. 2023. DocPrompting: Generating Code by
Retrieving the Docs. InThe Eleventh International Conference on Learning Representations. https://openreview.net/
forum?id=ZTCxT2t2Ru
[41] Tianyang Zhou, Ziyi Zhang, Haowen Lin, Somesh Jha, Mihai Christodorescu, Kirill Levchenko, and Varun Chan-
drasekaran. 2025. SACTOR: LLM-Driven Correct and Idiomatic C to Rust Translation with Static Analysis and
FFI-Based Verification. arXiv:2503.12511 [cs.SE] https://arxiv.org/abs/2503.12511
Received 20 February 2007; revised 12 March 2009; accepted 5 June 2009