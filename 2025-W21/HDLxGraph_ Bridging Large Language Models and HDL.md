# HDLxGraph: Bridging Large Language Models and HDL Repositories via HDL Graph Databases

**Authors**: Pingqing Zheng, Jiayin Qin, Fuqi Zhang, Shang Wu, Yu Cao, Caiwen Ding, Yang, Zhao

**Published**: 2025-05-21 16:14:10

**PDF URL**: [http://arxiv.org/pdf/2505.15701v1](http://arxiv.org/pdf/2505.15701v1)

## Abstract
Large Language Models (LLMs) have demonstrated their potential in hardware
design tasks, such as Hardware Description Language (HDL) generation and
debugging. Yet, their performance in real-world, repository-level HDL projects
with thousands or even tens of thousands of code lines is hindered. To this
end, we propose HDLxGraph, a novel framework that integrates Graph Retrieval
Augmented Generation (Graph RAG) with LLMs, introducing HDL-specific graph
representations by incorporating Abstract Syntax Trees (ASTs) and Data Flow
Graphs (DFGs) to capture both code graph view and hardware graph view.
HDLxGraph utilizes a dual-retrieval mechanism that not only mitigates the
limited recall issues inherent in similarity-based semantic retrieval by
incorporating structural information, but also enhances its extensibility to
various real-world tasks by a task-specific retrieval finetuning. Additionally,
to address the lack of comprehensive HDL search benchmarks, we introduce
HDLSearch, a multi-granularity evaluation dataset derived from real-world
repository-level projects. Experimental results demonstrate that HDLxGraph
significantly improves average search accuracy, debugging efficiency and
completion quality by 12.04%, 12.22% and 5.04% compared to similarity-based
RAG, respectively. The code of HDLxGraph and collected HDLSearch benchmark are
available at https://github.com/Nick-Zheng-Q/HDLxGraph.

## Full Text


<!-- PDF content starts -->

HDLxGraph: Bridging Large Language Models and
HDL Repositories via HDL Graph Databases
Pingqing Zheng
University of Minnesota, Twin Cities
Minneapolis, MN, USA
pingqingzheng13@gmail.comJiayin Qin
University of Minnesota, Twin Cities
Minneapolis, MN, USA
qin00162@umn.eduFuqi Zhang
University of Minnesota, Twin Cities
Minneapolis, MN, USA
zhan7076@umn.edu
Shang Wu
Northwestern University
Evanston, USA
swu@u.northwestern.eduYu Cao
University of Minnesota, Twin Cities
Minneapolis, MN, USA
yucao@umn.edu
Caiwen Ding
University of Minnesota, Twin Cities
Minneapolis, MN, USA
dingc@umn.eduYang (Katie) Zhao
University of Minnesota, Twin Cities
Minneapolis, MN, USA
zhao1948@umn.edu
Abstract —Large Language Models (LLMs) have demonstrated
their potential in hardware design tasks, such as Hardware
Description Language (HDL) generation and debugging. Yet,
their performance in real-world, repository-level HDL projects
with thousands or even tens of thousands of code lines is hindered.
To this end, we propose HDLxGraph, a novel framework that
integrates Graph Retrieval Augmented Generation (Graph RAG)
with LLMs, introducing HDL-specific graph representations by
incorporating Abstract Syntax Trees (ASTs) and Data Flow
Graphs (DFGs) to capture both code graph view and hardware
graph view. HDLxGraph utilizes a dual-retrieval mechanism that
not only mitigates the limited recall issues inherent in similarity-
based semantic retrieval by incorporating structural information,
but also enhances its extensibility to various real-world tasks
by a task-specific retrieval finetuning. Additionally, to address
the lack of comprehensive HDL search benchmarks, we intro-
duce HDLSearch, a multi-granularity evaluation dataset derived
from real-world repository-level projects. Experimental results
demonstrate that HDLxGraph significantly improves average
search accuracy, debugging efficiency and completion quality by
12.04%, 12.22% and 5.04% compared to similarity-based RAG,
respectively. The code of HDLxGraph and collected HDLSearch
benchmark are available at https://github.com/Nick-Zheng-Q/
HDLxGraph.
Index Terms —Graph RAG, Hardware description language,
LLM agent
I. I NTRODUCTION
Recent advances in Large Language Models (LLMs) for
software language understanding and generation [1], [2] have
inspired efforts to extend their capabilities to facilitate Hard-
ware Description Language (HDL) code designs. Prior works
have demonstrated LLMs’ potential in generating [3]–[5] and
debugging [6] HDL code [7], [8]. However, LLM performance
in HDL-related tasks remains hindered by limited training
data and degradation caused by long prompts. To addressthese issues, researchers have integrated Retrieval-Augmented
Generation (RAG) , which retrieves relevant HDL fragments
from high-quality HDL repositories to supplement knowledge
gaps and reduce prompt length [8], [9].
Despite its potential, existing RAG approaches in HDL pre-
dominantly rely on similarity-based semantic retrieval, which
exhibits low recall when encountering intricate queries or
large, complex HDL repositories. Figure 1 shows an HDL
debugging example for a CV32E40P RISC-V HDL implemen-
tation, which consists of over 30 modules [10]. The similarity-
based RAG approach relies solely on semantic similarity
between the user query and code module names, making it
vulnerable to vocabulary mismatches. For instance, a query
may contain only an ambiguous description, or the relevant
code may exist as an unnamed block within HDL repositories.
Inspired by recent advancements in Graph RAG [11], [12]
and the characteristics of HDL codes, we propose integrating
graph-based structures into HDL-specific RAGs to address the
aforementioned challenges. Specifically, we introduce HDLx-
Graph, a novel hybrid graph-enhanced RAG framework, which
incorporates two HDL-specific graphs: Abstract Syntax Trees
(ASTs) and Data Flow Graphs (DFGs). Using ASTs, we
partition the HDL repository with about several thousand lines
of code into a code graph view containing multi-level entity
relationships. While DFGs provide a more precise hardware
graph view of signal-level flow to reflect the circuit topology.
By integrating structural properties into semantic information,
HDLxGraph significantly enhances LLMs’ understanding of
code structures through AST retrieval, enabling multi-level
reasoning for complex HDL codes and ambiguous queries,
while demonstrating extensibility across three downstream
applications: code search, debugging, and completion, througharXiv:2505.15701v1  [cs.AR]  21 May 2025

Orig. Query
output logic            apu_read_dep_for_ jalr_o,
.........
apu_result_q <= 'b0;
          apu_flags_q  <= 'b0;
        end else begin
          if (apu_rvalid_i && apu_ multicycle  && 
(data_misaligned_i || data_misaligned_ex_i || 
(data_req_i && regfile_alu_we_i) || 
(mulh_active && (mult_operator_i == MUL_H)))) 
begin
     apu_rvalid_q <= 1'b1;
     apu_result_q <= apu_result_i;
    apu_flags_q  <= apu_flags_i;... HDLHardwar e Description
Language Database
Natural Language
Query
Similarity-based Sear ch
cv32e40p_ex_stage .svViolation of ALU operand forwarding  if
preceded by FPU multicycle instruction
There is PC mismatch in one of the fpu failing
tests. Initial debug shows that it could be a
potential bug in operand forwarding  “addi x16
… -> jar x0 x16 …”User  Debugging Query
HDLxGraph
module: ex_stage                        0.3
block: operand  forwarding        0.9
singal: apu_read..._ jalr_o           0.7wrong
location!!
operand in cv32e40p_alu.sv;
multicycle_o in cv32e40p_mult.sv Similarity-based:
Redundant & Inaccurate
Graph-based: 
Structural & Accurate
Multi-level Query in
HDLxGraph
jar & forwarding
 not consideredIn
Conventional
RAGVocabulary Mismatch
sub name(); always; <=
Consist; Instance; Edge
Trigger; SynchronizeStructurral Mismatch
Hierarchical & Structural
Flat & Token SequentialFig. 1. (Top) An illustration of the mismatch between HDL and natural lan-
guage in conventional RAG, including structural and vocabulary mismatches.
And (Bottom) a demonstration of HDLxGraph’s efficiency in bridging these
mismatches by incorporating graph information, using an HDL debugging
example for a CV32E40P RISC-V HDL implementation [10].
signal-level task-specific retrieval achieved by DFG. Further-
more, due to the identified absence of comprehensive HDL
code search benchmarks containing question-answer pairs with
multi-level relationships (as the example in Figure 1), we
extend HDLxGraph framework to address this gap, generating
a benchmark, dubbed as HDLSearch. Our key contributions
are summarized as follows:
•We propose HDLxGraph, a novel LLM-driven RAG
framework that leverages a dual-retrieval mechanism
based on AST and DFG retrieval. Specifically, it takes
into account the alignment across different hierarchical
levels in the AST and incorporates task-specific retrieval
on signal level within the DFG, thereby enabling more
fine-grained retrieval compared to conventional RAG and
demonstrating extensibility across various tasks. To the
best of our knowledge, HDLxGraph is the first framework
to integrate HDLs’ inherent graph structures with RAGs.
•HDLxGraph implements a repository-level HDL graph
database with hybrid graph view, where the AST graph
provides the code structure view while the DFG graph
represents the hardware graph view. The database con-
struction also considers cross-file relationship, thereby
providing a more accurate and consistent graph repre-
sentation of projects at repository level.
•Based on HDLxGraph, we further construct a new LLM-
generated dataset for HDL code search with data clean-
ing and evaluation, called HDLSearch, which derives
query benchmark from real-world repository-level HDL
projects, to solve the gap in insufficient search datasets
for HDL codes.
•Integrating HDLxGraph with three LLMs with various
scale and different coding abilities, we demonstrate the
versatility of HDLxGraph on three real-world HDL tasks,
i.e., code search, debugging, and completion. Experi-
ments demonstrate that our framework exhibits competi-tive performance on two widely-used benchmarks [13],
[14] for code completion and debugging as well as
HDLSearch for code search, respectively.
The remaining sections are organized as follows. Section
II provides an overview of the application of LLMs in hard-
ware design, alongside a review of conventional Verilog code
structural abstractions and graph-based RAG techniques. Sec-
tion III-B presents a detailed explanation of the HDLxGraph
workflow integrating AST and DFG abstraction and employs a
multi-hierarchy approach to generate the HDLSearch Bench-
mark regarding the benchmark gap in hardware searching.
Section IV reports thorough experimental results on three
hardware downstream tasks, and finally, Section V concludes
the paper.
II. P RELIMINARIES
A. LLM-aided HDL Tasks
Generation. Although LLMs excel in generating simple
HDL designs, they still struggle with complex repository-
level chip designs, as demonstrated in previous work [15]–
[23]. For example, state-of-the-art (SOTA) works [9], [22]–
[24] reply on the templates or customed RAG dataset provided
by human experts, using LLMs to fill in fixed-level content
while overlooking the entire generation.
Debugging. Existing work also exhibits certain limita-
tions when using LLMs for repository-level complex debug-
ging [13], [25]–[28]. The LLM4DV [25] framework utilizes
LLMs to generate test stimuli. Though performing well on
simple tasks, it fails to achieve high coverage in more complex
chip designs. Additionally, [26] integrates LLMs with RAG to
identify and patch functional HDL bugs. However, it still relies
on manually defined error types, limiting LLMs’ potential for
understanding-based bug fixing.
Search. Precise code search is the foundation to RAG for
both HDL generation and debugging. While no direct work has
focused on HDL search, recent studies have examined LLMs’
potential in HDL summarization [3], which is a pre-step for
HDL search, as well as EDA Q&A tasks [8]. However, these
works do not consider HDL’s inherent hierarchical structure,
preventing their direct application to precise code searches.
Additionally, previous work falls short in tasks beyond their
targeted objectives, limiting generalizability. Our proposed
HDLxGraph is a unified RAG-assisted framework designed
to address these three tasks while exploring LLMs’ potential
for repository-level HDL codes.
B. Graph Retrieval Augmented Generation
Graph Retrieval-Augmented Generation (Graph RAG)
leverages the structured nature of knowledge graphs and
integrates them into the RAG framework [11] to enable more
complex structured reasoning and context-aware responses.
Recent studies suggest that Graph RAG outperforms classical
RAG-based LLM systems in certain software code tasks [29]–
[31]. Inspired by this, our proposed HDLxGraph leverages the
unique structure of HDL’s ASTs and DFGs to optimize hard-
ware design via Graph RAG. Details shown in Section III-B.

AST Retrieval
QueryRTL
Repo
Multi-level Query
Module-level
Block-level
Signal-levelResult
Modules
Blocks
Signals
AST Parser
DFG Parser
Signal
Traverse
Primary
OperationsCode Debugging
DFG RetrievalInput
Code Completion
Retrieved
Results
DFG GraphAST GraphStep 1
InputStep 2 Step 3
Downstream T asks
Fusion
Module
Decomposer Task Solver
Code Semantic
SearchOutputModules
Blocks
Signals
Sim.-based
ExtractResult
Debugging
CompletionFig. 2. The overview of our proposed HDLxGraph framework.
C. HDL Code Structure
Graph code views, such as AST, DFG, and Control Flow
Graph (CFG), have been adopted for a more comprehensive
understanding of software programming language. Although
sharing syntactic similarities, HDLs introduce unique com-
plexities in representation characterized in three aspects: ex-
plicit timing modeling, inherent parallelism, and rigorous bit-
width specifications [32], [33]. Specifically, Verilog’s always
blocks enable concurrent execution, while the assign statement
facilitates continuous assignment, both differ from constructs
in software languages. Therefore, directly inheriting the graph
views from software code is infeasible. We conduct an in-depth
study of HDL-specific graphs and propose a graph database
with hybrid representations by AST and DFG in Section III-A.
D. Benchmarks/Datasets for LLM-aided Tasks
For HDL code generation benchmarks, RTLLM [34] con-
sists of 30 designs; and VerilogEval [14] presents an evaluation
dataset consisting of 156 problems from HDLBits. For HDL
code debugging benchmarks, LLM4SecHW [13] contains bug
localization and repair test sets from the version control
data in Github; RTLFixer [24] introduces a Verilog syntax
debugging dataset, derived from VerilogEval [14]; and CirFix
[35] includes a bug repair benchmark with testbenches.
No existing benchmark has been established for the HDL
search, which is an essential step for downstream tasks such as
generation and debugging. Therefore, we propose HDLSearch,
the first benchmark for HDL code search, which derives query
benchmark from real-world repository-level HDL projects.
III. M ETHODOLOGY
Figure 2 illustrates the comprehensive workflow of our
proposed HDLxGraph framework, which consists of three
steps: 1) Graph Database Preparation, 2) Multi-level Retrieval,
and 3) Downstream Task Completion. Beginning with Step 1 ,
we extract ASTs and DFGs from the input code repositories
through the AST and DFG parsers, then store HDL entities
and relationships as nodes and edges in a graph database (see
Section III-A). In Step 2 (see Section III-B), HDLxGraph
utilizes a Decomposer Agent in AST retrieval to extract the
input query into structural levels, which are later sent to pre-
defined searching paradigms to retrieve relevant fine-grained
code snippets. Additionally, code debugging and completiontasks trigger DFG retrieval in parallel to narrow the search
space or enable similarity matching between incomplete and
complete code snippets. HDLxGraph supports three real-world
HDL downstream tasks. Step 3 fuses the retrieved code
snippets with LLMs to support code debugging, completion,
and search, which further demonstrates the generality of our
framework (see Section III-B). In addition, due to the lack of
a code search benchmark in HDL repositories, we generated
a new benchmark, called HDLSearch, based on HDLxGraph,
as shown in Figure 5, composed of three steps: 1) Manual
Filtering, 2) Query Generation, 3) Benchmark Generation.
Details of benchmark generation are presented in Section
III-C.
A. Graph Database Preparation
As shown in Step 1 of Figure 2, the HDLxGraph RAG
framework begins with an off-line graph database construc-
tion. The graph database represents HDL repositories through
nodes and edges that correspond to HDL entities and their
relationships. Without losing representativity, we focus on
Verilog, a widely used HDL language, in our implementation.
Please note that, although different HDLs have different syn-
tactic properties, they share the same three-level structural ab-
straction, i.e., (module →block →signal) in Verilog.
Specifically, we use an AST to support the code graph view
that emphasizes multi-level structural relationships in HDL,
and a DFG to facilitate the hardware graph view focusing
on signal flow reflecting circuit topology, providing a com-
prehensive and tailored representation of the HDL repository.
The AST graph incorporates node types such as MODULE ,
BLOCK , and SIGNAL connected through CONTAINS and
INSTANTIATE edge types, whereas the DFG graph intro-
duces TEMP nodes alongside SIGNAL nodes, connected
viaFLOWS_TO ,TRUE ,FALSE , and COND edges. When
constructing the entire graph database, there are three main
sub-steps:
1) Parsing. The graph database construction begins with
analyzing individual HDL file in the repository using a
Pyverilog-based [36] AST and DFG parser. For AST pars-
ing, we extract the cross-level dependancy information of
MODULE ,BLOCK , and SIGNAL from each Verilog file to
represent the fine-grained hierarchical code structure. Note that
block level (always, assign, initial) here represent behavioral
abstraction at the register-transfer level, defining concurrent
hardware operations. Concurrently, we generate the hardware
signal flow for DFG parsing, which characterizes the trans-
mission and interaction between signals. The DFG graph
incorporates both the signal directions and the dependency
relationships between signals, reflecting the functionality and
processing flow of a circuit. This multi-granularity representa-
tion enables our database to store both the code structure and
the hardware behavior of a single HDL file, thereby facilitating
a more comprehensive graph abstraction of the HDL, as shown
in Figure 3.
2) Meta-data generation. After the parsing of graph data,
we generate embeddings for nodes (both MODULE and

module incrementer (input wire  clk, input
wire rst, input wire  [3:0] in_data, output
wire [3:0] out_data );
    reg [3:0] data_reg;
    
    always  @(posedge  clk or posedge  
    rst) begin
     if (rst)
       data_reg <=4'b0000;
     else
       data_reg <=in_data;
     end
     assign  out_data=data_reg+1;
endmodulemodule  top_module( input wire  clk, input
wire rst, input wire  [3:0] sw , output wire
[3:0] led );
    wire [3:0] inc_result;
    incrementer  u_incrementer (.clk(clk), 
        .rst(rst), .in_data(sw), 
         .out_data(inc_result) );
    reg[3:0] led_reg;
    always  @(posedge  clk or posedge  
    rst) begin
     if (rst)
       led_reg <=4'b0000;
     else
       led_reg <=inc_result;
     end
     assign  led=led_reg;
endmodule
(1) source code (2) nodes & edges# example nodes
MODULE : "incrementer",
"top_module"
Block : "always...", "assign..."
Signal : "clk","rst",
"in_data","out_data"...
Temp: "Branch_in_data_4'd0"
# example edges
# meta-data of an BLOCK node
type: "Always"
code_embedding : "0.08,0.028,-0.09
,0.003,..."
code : "always @(posedge clk or
posedge rst) begin ......."AST
DFG
AST
DFGCONT AINS : 
("top_module")->("always_led_reg")
INITIA TES :
 ("top_module")->("u_incrementer")
FLOWS_T O: 
("inc_result")->("led_reg")
COND :
("rst")->("Branch_incrementer .in_
data_ 4'd0") 
(3) visualizationFLOWS_TO
FLOWS_TOFALSEFLOWS_TO
F…
FLOWS_TO FLOWS_TO
C…F…
TRUEFLOWS_TO
FLOW
…CONTAINSCONTAINS
CONTAINSCONTAINSCONTAINS
CONTAINS
CONTAINSCONTAINS
CONTAI…
CONTAINSCONTAINS
CONTAINSCONTAINS
CONTAINS
INS…CONTAINSCONTAINSCONTAINS
CONTAINS
CO
NTAINSCONTAINS
CONTAINS
FLOWS_TO
FLOWS_TOFALSEFLOWS…
FLOWS_TO FLOWS_TO
C…FLOWS_TO
TRUE
CONTAINSCONTAINS
C…CONTAI…
CONTAINS
CONTAINSCONTAINS
CONTAINSCONTAINS
CONTAINS CONTAINS
CONTAINSCONTAINSCONTAINSCONTAINS
in_datadata_reg
rstout_data
Branch_…
4'd0_gra…Plus_gra…
'd1_grap…top_mo…
input
wire
clk, i…wire
[3:0]
inc_r…
u_incre…reg [3:0]
led_re…always
@(pos…
assign
led =
led_…
led
inc_resultclk
swled_reg
rstBranch_…
4'd0_gra…
increme…
input
wire
clk, i…reg [3:0]
data_r…
always
@(pos…assign
out_data
= d
clk
Module
Block
Signal
TempFig. 3. Visualization of an example in the graph database.
BLOCK ) via code encoding to facilitate semantic search.
These embeddings, together with the node attributes extracted
by the parser, are recorded in the database as part of the node
meta-data. Typically, meta-data will include multiple attributes
of each node, for instance, the meta-data for BLOCK nodes
contains attributes such as block type, code, and embedding,
as illustrated in Figure 3. We use CodeT5+ [1], a SOTA code
LLM model, to directly generate the embedding for our code,
avoiding description generation.
3) Cross-file Relationship Construction. Finally, we ad-
dress the absense of cross-file relationship, which is the mod-
uleINSTANTIATE relationships. We search for the module
node with same name recorded in the meta-info instance block
to establish the cross-file and cross-module relationships.
The developed graph database provides multi-level code
exploration spanning from module-level abstractions to signal-
level implementations, thereby positioning our HDL graph
database as a extensible framework for multiple downstream
tasks due to the modular fashion of database schema manage-
ment.
B. Multi-level Retrieval and Downstream Task Completion
Multiple-level Retrieval: In real-world hardware project
issues, user queries always contain rich contextual cues, such
as module names, functional descriptions, and sometimes
brief code snippets, offering hints for retrieval. Specifically,
user queries can be used to extract multi-level structural
information and then guide the following multiple-level AST
retrieval1. In addition, signal-level flow through DFG retrieval
is adopted for code completion and debugging tasks.
AST Retrieval. HDLxGraph constructs a hierarchical rep-
resentation of HDL codebases through an AST-based graph,
enabling multi-level HDL retrieval as depicted in Figure 4. For
AST retrieval, we follow three sub-steps:
1Figure 9 in Appendix A demonstrates this through two real-world issues
submitted to the CV A6 [37] and OpenTitan [38] projects, where highlighted
hints guide HDL retrieval.
Multi-level Query
Module-level
Block-level
Signal-level
TracerError
Signal
Unfinish
CodeDFG
ParserUnfinish 
GraphQuery
DFG Graph DatabaseAST Graph Database
DFG Graph
Traverse
Graph SimilarityAST Graph
Module
Block
SignalTop-k Nodes Rerank
Debug. Result
Completion
ResultDebug.
CompletionQuery DecompositionTop-k Selection &
FilteringCross-level 
RerankFig. 4. Flow of multi-level retrieval containing AST and DFG retrieval.
1) Query Decomposition. HDLxGraph employs an
LLM agent, called Decomposer , to decompose the original
query into three abstraction levels: module, block, and
signal, thereby extracting structural information. It supports
intricate queries from various downstream tasks such as:
‘Find some certain blocks under a certain
module’ in Search, or ‘Some functions in some
certain modules have led to the following
errors ...’ Therefore, we obtain multi-level queries
which have captured inherent structural information in the
original query.
2) Top-k Selection and Filtering. Leveraging Ver-
ilog’s inherent three-level abstraction (module →block →
signal) , HDLxGraph first retrieves top-k candidate modules
and blocks which have the highest similarity scores with the
decomposed query in corresponding levels based on semantic
matching, then filters valid module-block pairs through con-
tainment relationships. To facilitate precise code retrieval in
different levels, a suite of retrieval APIs is introduced, as
detailed in Table I of Appendix A. Since we select Neo4j
as the graph database, the query APIs are written in Cypher
to interact with the database.
3) Cross-level Rerank. Finally, we rerank results using av-
eraged similarity scores. Since the signal-level representation
lacks the code context, it is challenging to directly obtain an
accurate similarity score for the signal-level query. Therefore,
HDLxGraph extracts all filtered module-block pairs that con-
tain the signal and computes their average similarity scores as
the signal-level similarity score. Therefore, we prioritize signal
with the highest similarity score to be the retrieved signal. This
hierarchical approach ensures fine-grained retrieval of HDL’s
structural information across multiple abstraction layers while
maintaining compatibility with similarity-based semantic anal-
ysis, balancing precision and scalability in hardware database
exploration.
DFG Retrieval. The DFG is composed of signal-level
variables and relationships. As a result, it is useful when
signal-level information is needed and the utilization method
can vary greatly for different downstream tasks. In this work,
we utilize the signal-level flow to enhance code completion
and code debugging tasks, as illustrated in Figure 2. There
are two primary operations for DFG retrieval of different tasks,
which are Signal Traverse andSimilarity-based Extract :

Blocks
Modules
Signals
Query
Gen.
Query
Gen.
Internal
Eval.
Internal
Eval.
Bench-
mark
Step 2 Step 3
RTL
Repo
Input
Filtered
Repo
Manual FilteringStep 1
Fig. 5. HDLSearch benchmark generation flow.
1) Debugging. For debugging tasks, if a signal mismatch
is detected, the debugging process can iteratively traverse the
DFG upstream with Signal Traverse operation from the faulty
signal, inspecting each node (e.g., operators, multiplexers,
or instance outputs) to identify where the dataflow diverges
from expected behavior. This approach guides LLM debugging
by focusing only on the subgraph directly influencing the
problematic signal, filtering out irrelevant code regions. By
extracting the immediate upstream nodes and their associated
code blocks, the system generates a concise, context-rich error
candidate set.
2) Completion. While we want to retrieve the similar code
with the unfinished code, some reference code may look
different but still having the similar functionality because the
hardware (i.e. dataflow) described are very similar. Graph
embedding offers a viable approach for Verilog code com-
pletion by translating code’s structural and semantic relation-
ships into a unified mathematical framework. By leveraging
GraphSAGE [39], these graphs are compressed into low-
dimensional vector representations that preserve contextual
patterns, such as recurring HDL constructs (e.g., finite state
machines, pipelined operations) or common coding idioms
(e.g., non-blocking assignments in clock-driven blocks). When
a developer writes partial code, the corresponding subgraph is
embedded and compared against historical embeddings using
similarity metrics, enabling the system to infer likely comple-
tions—even with incomplete structures—by prioritizing nodes
critical to the current context. This allows real-time retrieval
of relevant patterns from large codebases while adhering to
Verilog-specific constraints.
Downstream Task Completion: The propagation trajec-
tory of error signals establishes causal dependencies within
hardware description constructs, enabling LLMs to trace
fault origins through backward-chaining analysis. Meanwhile,
Dataflow graph analysis enables LLMs to identify functionally
equivalent code patterns by detecting structural similarities in
hardware operations, even when surface code syntax differs.
This approach allows semantic-aware code completion beyond
literal text matching.
C. HDL Search Benchmark
Observing the absence of an HDL code search benchmark,
we aim to establish a specific benchmark to address this gap.
However, manually creating expert-annotated benchmark is
time-consuming and labor-intensive, posing it economicallyimpractical. Therefore, based on the multi-level hierarchical
framework of HDLxGraph, we propose to leverage LLMs to
construct a benchmark dubbed HDLSearch, as shown in Figure
5. The benchmark generation can be divided into mainly three
sub-steps:
1) Manual Filtering. Our corpus originates from RTL-Repo
[40], a collection of publicly accessible GitHub repositories
specializing in HDLs. Unlike conventional software reposito-
ries, HDL projects tend to lack structured documentation and
standardized code organization, making automated repository
filtering particularly challenging. To address this limitation, we
first implement a manual filtering and select 10 representative
repositories at different difficulty levels, ranging from educa-
tional FPGA projects, interconnection protocols to commercial
CPUs.
2) Query Generation. Adopting a hierarchical framework
where block serve as the fundamental level , we implement
a multi-stage generation process. Initial functional block de-
scriptions are first generated, then systematically propagated
through two parallel pathways, which are 1) Signal-level
annotation: through contextual information, the semantics of
a functional block can be inherited by its associated signals,
thereby effectively annotating these signals with specific func-
tionalities, and 2) Module-level abstraction by designing a set
of explicit and tailored prompts for the LLMs, we enable it
to analyze and summarize the interactions among individual
functional blocks as a module-level description. This dual-
path flow ensures consistent semantic alignment between fine-
grained signal behaviors and coarse-grained module opera-
tions. With all descriptions finished, repo-specific information
such as module and signal’s names are removed to generate a
relatively ambiguous query.
3) Benchmark Refinement. To further ensure benchmark
validity, we employ an iterative refinement process using tem-
plated instructions (shown in Appendix B). Through multiple
rounds of evaluation and regeneration, we gradually remove
unsuitable queries and align the LLM-generated query outputs
with practical engineering requirements till it reaches the
defined termination count 𝐾. After that, manual adjustments
are undertaken to address few gaps between LLM outputs and
the actual search intent.
IV. E XPERIMENTS
A. Experimental Configuration and Platforms
To explore the capabilities of the proposed HDLxGraph
framework, we evaluate it on three HDL downstream tasks:
code search, code debugging, and code completion. The task-
specific benchmarks and experimental metrics are detailed in
the following subsections. We equip HDLxGraph with three
LLMs with different model sizes: Claude-3.5-Sonnet [41],
a large model with strong coding ability; Qwen2.5-Coder-
7B [42], a coding-specific model of medium size; and
LLAMA-3.1 [43], a general-purpose model with a relatively
small size. We use top-p = 1.0 and temperature = 0.7 as our
basic configuration. All experiments are run on a 2xA6000
Linux GPU Server and all benchmark evaluations conduct 10

0.72
0.64
0.56
0.48
0.4BM25Sear ch
HDLxGraph0.54670.6671MRRSimilarity-
based RAG
0.5017Fig. 6. HDL search MRR comparison with baselines.
independent experimental trials per task to ensure statistical
robustness.
B. Code Semantic Search
Benchmark: Considering the absence of benchmarks for
HDL-specific code search, we use our proposed HDLSearch
(see Section III-C) as the benchmark with termination count
𝐾=7when generation. The generated benchmark comprises
40 module-level queries, 100 block-level queries and 200
signal-level queries, with 6,300 code blocks from 10 reposito-
ries serving as distractor, i.e., retrieval scope. The evaluation
focuses on block-level retrieval, which serves as a fundamental
level with highest extensibility to other downstream tasks as
mentioned before.
Metric: We adopt the widely used mean reciprocal rank
(MRR) in RAG as the primary metric, which assesses whether
the framework is capable of returning correct results within the
top-ranked outputs:
MRR =1
𝑁𝑁∑︁
𝑖=11
rank𝑖(1)
Baselines: We compare HDLxGraph against two commonly
used similarity-based RAG methods, BM25 [44] and CodeT5+
embeddings [1].
Evaluation Results: As shown in Figure 6, HDLxGraph
achieves superior performance in block-level search evaluated
with all the 100 block-level queries (averaged 12.04% MRR
improvement), demonstrating its potential in accurate HDL
search for complex repository-level codes.
C. Code Debugging
Benchmark: We evaluate HDLxGraph’s capability in han-
dling real-world, repository-level debugging challenges and
choose LLM4SecHW [13] as the benchmark, which extracts
and refines data from the version control systems of open-
source repository-level hardware designs. Specifically, we
choose the mor1kx repository [45], an OpenRISC processor IP
core, for our evaluation2. In the mor1kx repository [45], there
are 5 git commit SHAs covering different debugging issues.
Metric: Following LLM4SecHW, we choose ROUGE-N F1
score [46] as the evaluation metric, which refers to the direct
𝑁-gram overlap between a prediction and a reference word
considering precision and recall. The parameter 𝑁can be set
2Since HDLxGraph’s AST and DFG parsers currently do not support Sys-
temVerilog syntax, we leave further debugging evaluation on SystemVerilog
repositories as future work.
ROUGE F1 Scor eAccurate-RAG Similarity-based RAG HDLxGraphDebugging
0.6
0.48
0.36
0.24
0.12
0
ROUGE-1 ROUGE-2 ROUGE-L ROUGE-1 ROUGE-2 ROUGE-L ROUGE-1 ROUGE-2 ROUGE-L
LLaMA3.1 Qwen2.5-Coder -7B Claude-3.5-SonnetFig. 7. HDL debugging comparison with baselines.
to 1, 2 and L, corresponding to matching at unigram, bigram,
and longest common subsequence gram, respectively.
Baselines: We compare our framework with two RAG strate-
gies: the CodeT5+ embedding search strategy [1], denoted as
“Similarity-based RAG”, which represents the conventional
similarity-based RAG approach, and the accurate-RAG de-
bugging strategy, denoted as “Accurate-RAG”, which relies
on human effort to extract the exact buggy code segments to
be modified, serving as a theoretical top-tier RAG baseline.
Evaluation Results: As illustrated in Figure 7, HDLxGraph
achieves a higher score of ROUGE-1, ROUGE-2 and ROUGE-
L compared to similarity-based RAG under all scenarios, and
exhibits performance approaching that of the top-tier baseline.
This demonstrates HDLxGraph’s potential in handling real-
world debugging issues.
D. Code Completion
Benchmark: We evaluate code completion capabilities
using VerilogEval-Human v2 [14] with RTLLM [34] as a
reference implementation.
Metric: we apply Pass@k metrics [47] to assess the generation
pass rate:
pass@k =EProblems"
1− 𝑛−𝑐𝑝
𝑘
 𝑛
𝑘#
(2)
where𝑛is the total number of generations, 𝑐𝑝is the number
of successes, and 𝑘is the number of attempts considered. We
apply𝑃𝑎𝑠𝑠 @1in our experiment.
Baselines: We compare HDLxGraph against two baselines:
direct LLM completion without RAG and similarity-based
RAG using CodeT5+, as described in Section IV-C.
Evaluation Results: As shown in Figure 8, HDLxGraph
consistently improves 𝑃𝑎𝑠𝑠 @1 accuracy by 3-10% across
various LLMs. While our evaluation framework operates
at module granularity rather than full repository scope, we
strategically employ the RTLLM [48] codebase as a RAG
corpus, thereby maintaining repository-level evaluation. The
higher accuracy suggests HDLxGraph’s generalizability across
different abstraction levels, highlighting that structural code
understanding significantly benefits completion tasks, even at
sub-repository granularity.

Similarity-based RAG No RAGCompletion
HDLxGraphPass@1 Rate (%)100%
80%
60%
40%
20%
012.50%
LLaMA3.1 Qwen2.5-Coder -7B13.78%23.72%29.81% 29.82% 32.41%72.44% 73.40%76.00%
Claude-3.5-SonnetFig. 8. HDL completion Pass@1 comparison with baselines.
V. C ONCLUSION AND FUTURE WORK
In this work, we propose HDLxGraph, a novel hybrid graph-
enhanced RAG framework that innovatively combines AST-
based structural matching with DFG-aware code retrieval.
Experimental validation across semantic search, debugging,
and code completion tasks demonstrates improvements of
12.04%, 12.22%, and 5.04% respectively over conventional
methods, proving the effectiveness of joint structural-semantic
retrieval for HDL applications.
This work establishes graph-enhanced retrieval as a viable
paradigm for hardware engineering assistance, with broader
implications for code-intensive domains requiring precise pro-
gram analysis. Future directions may include multi-view HDL
representation learning to bridge the semantic gap between
natural language specifications and circuit implementations.
This direction could enable comprehensive support for het-
erogeneous downstream tasks in electronic design automation,
from specification validation to cross-module optimization.
REFERENCES
[1] Yue Wang, Hung Le, Akhilesh Gotmare, Nghi Bui, Junnan Li, and
Steven Hoi. CodeT5+: Open code large language models for code
understanding and generation. In Houda Bouamor, Juan Pino, and Kalika
Bali, editors, Proceedings of the 2023 Conference on Empirical Methods
in Natural Language Processing , December 2023.
[2] Anton Lozhkov, Raymond Li, Loubna Ben Allal, and et al. Starcoder 2
and the stack v2: The next generation. arXiv , 2024.
[3] Yang Zhao, Di Huang, Chongxiao Li, Pengwei Jin, Ziyuan Nan, Tianyun
Ma, Lei Qi, Yansong Pan, Zhenxing Zhang, Rui Zhang, Xishan Zhang,
Zidong Du, Qi Guo, Xing Hu, and Yunji Chen. Codev: Empowering
llms for verilog generation through multi-level summarization. arXiv ,
2024.
[4] Shailja Thakur, Baleegh Ahmad, Hammond Pearce, Benjamin Tan,
Brendan Dolan-Gavitt, Ramesh Karri, and Siddharth Garg. Verigen:
A large language model for verilog code generation. arXiv , 2023.
[5] Xi Wang, Gwok-Waa Wan, Sam-Zaak Wong, Layton Zhang, Tianyang
Liu, Qi Tian, and Jianmin Ye. Chatcpu: An agile cpu design and
verification platform with llm. In Proceedings of the 61st ACM/IEEE
Design Automation Conference , DAC ’24, New York, NY , USA, 2024.
Association for Computing Machinery.
[6] Xufeng Yao, Haoyang Li, Tsz Ho Chan, Wenyi Xiao, Mingxuan Yuan,
Yu Huang, Lei Chen, and Bei Yu. Hdldebugger: Streamlining hdl
debugging with large language models. arXiv , 2024.
[7] Luyao Shi, Michael Kazda, Bradley Sears, Nick Shropshire, and Ruchir
Puri. Ask-eda: A design assistant empowered by llm, hybrid rag
and abbreviation de-hallucination. In 2024 IEEE LLM Aided Design
Workshop (LAD) , pages 1–5, 2024.
[8] Yuan Pu, Zhuolun He, Tairu Qiu, Haoyuan Wu, and Bei Yu. Customized
retrieval augmented generation and benchmarking for eda tool documen-
tation qa. arXiv , 2024.[9] Mingzhe Gao, Jieru Zhao, Zhe Lin, Wenchao Ding, Xiaofeng Hou,
Yu Feng, Chao Li, and Minyi Guo. Autovcoder: A systematic framework
for automated verilog code generation using llms. arXiv , 2024.
[10] Michael Gautschi, Pasquale Davide Schiavone, Andreas Traber, Igor Loi,
Antonio Pullini, Davide Rossi, Eric Flamand, Frank Gurkaynak, and
Luca Benini. Near-Threshold RISC-V Core With DSP Extensions for
Scalable IoT Endpoint Devices, February 2017.
[11] Darren Edge, Ha Trinh, Newman Cheng, Joshua Bradley, Alex Chao,
Apurva Mody, Steven Truitt, and Jonathan Larson. From local to global:
A graph rag approach to query-focused summarization. arXiv , 2024.
[12] Zirui Guo, Lianghao Xia, Yanhua Yu, Tu Ao, and Chao Huang. Lightrag:
Simple and fast retrieval-augmented generation. arXiv , 2024.
[13] Weimin Fu, Kaichen Yang, Raj Gautam Dutta, Xiaolong Guo, and Gang
Qu. Llm4sechw: Leveraging domain-specific large language model for
hardware debugging. In 2023 Asian Hardware Oriented Security and
Trust Symposium (AsianHOST) , pages 1–6, 2023.
[14] Mingjie Liu, Nathaniel Pinckney, Brucek Khailany, and Haoxing Ren.
VerilogEval: evaluating large language models for verilog code genera-
tion. In 2023 IEEE/ACM International Conference on Computer-Aided
Design (ICCAD) , 2023.
[15] Jason Blocklove, Siddharth Garg, Ramesh Karri, and Hammond Pearce.
Chip-chat: Challenges and opportunities in conversational hardware
design. arXiv preprint arXiv:2305.13243 , 2023.
[16] Yongan Zhang, Zhongzhi Yu, Yonggan Fu, Cheng Wan, and
Yingyan (Celine) Lin. MG-Verilog: multi-grained dataset towards en-
hanced llm-assisted verilog generation. In The First IEEE International
Workshop on LLM-Aided Design (LAD’24) , 2024.
[17] Shailja Thakur, Baleegh Ahmad, Hammond Pearce, Benjamin Tan,
Brendan Dolan-Gavitt, Ramesh Karri, and Siddharth Garg. Verigen:
A large language model for verilog code generation. ACM Trans. Des.
Autom. Electron. Syst. , 29(3), April 2024.
[18] Erik Nijkamp, Bo Pang, Hiroaki Hayashi, Lifu Tu, Huan Wang, Yingbo
Zhou, Silvio Savarese, and Caiming Xiong. Codegen: An open large
language model for code with multi-turn program synthesis. arXiv
preprint arXiv:2203.13474 , 2022.
[19] Shailja Thakur, Baleegh Ahmad, Zhenxing Fan, Hammond Pearce,
Benjamin Tan, Ramesh Karri, Brendan Dolan-Gavitt, and Siddharth
Garg. Benchmarking large language models for automated verilog
rtl code generation. In 2023 Design, Automation & Test in Europe
Conference & Exhibition (DATE) , pages 1–6. IEEE, 2023.
[20] Shang Liu, Wenji Fang, Yao Lu, Qijun Zhang, Hongce Zhang, and
Zhiyao Xie. Rtlcoder: Outperforming gpt-3.5 in design rtl generation
with our open-source dataset and lightweight solution. In 2024 IEEE
LLM Aided Design Workshop (LAD) , pages 1–5. IEEE, 2024.
[21] Hasan Genc, Seah Kim, Alon Amid, Ameer Haj-Ali, Vighnesh Iyer,
Pranav Prakash, Jerry Zhao, Daniel Grubb, Harrison Liew, Howard
Mao, Albert Ou, Colin Schmidt, Samuel Steffl, John Wright, Ion
Stoica, Jonathan Ragan-Kelley, Krste Asanovic, Borivoje Nikolic, and
Yakun Sophia Shao. Gemmini: Enabling systematic deep-learning ar-
chitecture evaluation via full-stack integration. In 2021 58th ACM/IEEE
Design Automation Conference (DAC) , pages 769–774, 2021.
[22] Deepak Vungarala, Mahmoud Nazzal, Mehrdad Morsali, Chao Zhang,
Arnob Ghosh, Abdallah Khreishah, and Shaahin Angizi. Sa-ds: A dataset
for large language model-driven ai accelerator design generation, 2024.
[23] Yonggan Fu, Yongan Zhang, Zhongzhi Yu, Sixu Li, Zhifan Ye, Chaojian
Li, Cheng Wan, and Yingyan Celine Lin. Gpt4aigchip: Towards
next-generation ai accelerator design automation via large language
models. In 2023 IEEE/ACM International Conference on Computer
Aided Design (ICCAD) , pages 1–9, 2023.
[24] Yun-Da Tsai, Mingjie Liu, and Haoxing Ren. Rtlfixer: Automatically
fixing rtl syntax errors with large language models. arXiv , 2023.
[25] Zixi Zhang, Greg Chadwick, Hugo McNally, Yiren Zhao, and Robert
Mullins. Llm4dv: Using large language models for hardware test stimuli
generation. ArXiv , abs/2310.04535, 2023.
[26] Khushboo Qayyum, Muhammad Hassan, Sallar Ahmadi-Pour, Chan-
dan Kumar Jha, and Rolf Drechsler. From bugs to fixes: Hdl bug
identification and patching using llms and rag. In 2024 IEEE LLM
Aided Design Workshop (LAD) , pages 1–5, 2024.
[27] Ke Xu, Jialin Sun, Yuchen Hu, Xinwei Fang, Weiwei Shan, Xi Wang,
and Zhe Jiang. Meic: Re-thinking rtl debug automation using llms.
arXiv , 2024.
[28] Baleegh Ahmad, Shailja Thakur, Benjamin Tan, Ramesh Karri, and
Hammond Pearce. On hardware security bug code fixes by prompting

large language models. IEEE Transactions on Information Forensics
and Security , 19:4043–4057, 2024.
[29] Kounianhua Du, Jizheng Chen, Renting Rui, Huacan Chai, Lingyue Fu,
Wei Xia, Yasheng Wang, Ruiming Tang, Yong Yu, and Weinan Zhang.
Codegrag: Bridging the gap between natural language and programming
language via graphical retrieval augmented generation. arXiv , 2024.
[30] Xiangyan Liu, Bo Lan, Zhiyuan Hu, Yang Liu, Zhicheng Zhang, Fei
Wang, Michael Shieh, and Wenmeng Zhou. Codexgraph: Bridging large
language models and code repositories via code graph databases. arXiv ,
2024.
[31] Ibrahim Abdelaziz, Julian Dolby, James P McCusker, and Kavitha
Srinivas. A toolkit for generating code knowledge graphs. The Eleventh
International Conference on Knowledge Capture (K-CAP) , 2021.
[32] Ieee standard for verilog hardware description language. IEEE Std 1364-
2005 (Revision of IEEE Std 1364-2001) , pages 1–590, 2006.
[33] M. Gordon. The semantic challenge of verilog hdl. In Proceedings of
Tenth Annual IEEE Symposium on Logic in Computer Science , pages
136–145, 1995.
[34] Yao Lu, Shang Liu, Qijun Zhang, and Zhiyao Xie. Rtllm: An open-
source benchmark for design rtl generation with large language model.
In2024 29th Asia and South Pacific Design Automation Conference
(ASP-DAC) , pages 722–727. IEEE, 2024.
[35] Hammad Ahmad, Yu Huang, and Westley Weimer. Cirfix: automatically
repairing defects in hardware design code. In Proceedings of the 27th
ACM International Conference on Architectural Support for Program-
ming Languages and Operating Systems , ASPLOS ’22, page 990–1003,
New York, NY , USA, 2022.
[36] Shinya Takamaeda-Yamazaki. Pyverilog: A python-based hardware
design processing toolkit for verilog hdl. In Applied Reconfigurable
Computing , volume 9040 of Lecture Notes in Computer Science , pages
451–460. Springer International Publishing, Apr 2015.
[37] F. Zaruba and L. Benini. The cost of application-class processing:
Energy and performance analysis of a linux-ready 1.7-ghz 64-bit risc-v
core in 22-nm fdsoi technology. IEEE Transactions on Very Large Scale
Integration (VLSI) Systems , 27(11):2629–2640, Nov 2019.
[38] Scott Johnson, Dominic Rizzo, Parthasarathy Ranganathan, Jon Mc-
Cune, and Richard Ho. Titan: enabling a transparent silicon root of
trust for cloud. In Hot Chips: A Symposium on High Performance Chips ,
volume 194, page 10, 2018.
[39] William L. Hamilton, Rex Ying, and Jure Leskovec. Inductive repre-
sentation learning on large graphs, 2018.
[40] Ahmed Allam and Mohamed Shalan. Rtl-repo: A benchmark for
evaluating llms on large-scale rtl design projects. arXiv , 2024.
[41] Sonnet Anthropic. Model card addendum: Claude 3.5 haiku and
upgraded claude 3.5 sonnet.
[42] Binyuan Hui, Jian Yang, Zeyu Cui, Jiaxi Yang, Dayiheng Liu, Lei Zhang,
Tianyu Liu, Jiajun Zhang, Bowen Yu, Keming Lu, Kai Dang, Yang Fan,
Yichang Zhang, An Yang, Rui Men, Fei Huang, Bo Zheng, Yibo Miao,
Shanghaoran Quan, Yunlong Feng, Xingzhang Ren, Xuancheng Ren,
Jingren Zhou, and Junyang Lin. Qwen2.5-coder technical report. arXiv ,
2024.
[43] Aaron Grattafiori, Abhimanyu Dubey, Abhinav Jauhri, Abhinav Pandey,
Abhishek Kadian, Ahmad Al-Dahle, Aiesha Letman, Akhil Mathur, Alan
Schelten, Alex Vaughan, Amy Yang, and et al. The llama 3 herd of
models. arXiv , 2024.
[44] Stephen Robertson, Hugo Zaragoza, et al. The probabilistic relevance
framework: Bm25 and beyond. Foundations and Trends® in Information
Retrieval , 3(4):333–389, 2009.
[45] OpenRISC. mor1kx. https://github.com/openrisc/mor1kx, 2022.
[46] Chin-Yew Lin. ROUGE: A package for automatic evaluation of sum-
maries. In Text Summarization Branches Out , pages 74–81, Barcelona,
Spain, July 2004. Association for Computational Linguistics.
[47] Andre Nakkab, Sai Qian Zhang, Ramesh Karri, and Siddharth Garg.
Rome was not built in a single step: Hierarchical prompting for llm-
based chip design. In Proceedings of the 2024 ACM/IEEE International
Symposium on Machine Learning for CAD , MLCAD ’24, New York,
NY , USA, 2024. Association for Computing Machinery.
[48] Yao Lu, Shang Liu, Qijun Zhang, and Zhiyao Xie. Rtllm: An open-
source benchmark for design rtl generation with large language model.
In2024 29th Asia and South Pacific Design Automation Conference
(ASP-DAC) , pages 722–727, 2024.

APPENDIX
A. Multi-level Retrieval
Figure 9 demonstrates two real-world repository-level ex-
amples used for multi-level retrieval, which are CV A6 and
Opentitan, respectively. The highlighted part demonstrates the
retrieval query.
(Issue CVA6-2732 )
[BUG] Cr oss-privilege TLB leakage thr ough SLS
Our microarchitectural fuzzer has found that CV A6 is susceptible to SLS
(straight-line speculation [1]) and thus allows leakage  through the TLB
across privileges. Since speculatively issued loads and stor es from a
higher privilege access the TLB, their addresses can be recovered from a
lower privilege. Thus, privileged code that (architecturally) does not leak
any sensitive data through its control flow or memory operations, leaks
transiently to an unprivileged attacker .
We provide a snippet from the generated test case bellow:.......
(Issue Opentitan-26355 )
[sram_ctrl,rtl] Remove macr o timing assumptions
Some parts of the sram_ctrl  design (e.g., the r eadback featur e) make
assumptions about the timing of the underlying SRAM macr o (e.g., a
read always comes back at the next cycle).
We should identify those assumptions and rewrite the design such that the
controller can handle dif ferent SRAM macros.
Fig. 9. Two real-world issues posted in repository-level projects.
Table I shows the code retrieval APIs containing mainly
five APIs:𝑠𝑒𝑎𝑟𝑐ℎ𝑚𝑜𝑑𝑢𝑙𝑒 ,𝑠𝑒𝑎𝑟𝑐ℎ𝑏𝑙𝑜𝑐𝑘 ,𝑠𝑒𝑎𝑟𝑐ℎ𝑠𝑖𝑔𝑛𝑎𝑙 ,
𝑠𝑒𝑎𝑟𝑐ℎ𝑚𝑜𝑑𝑢𝑙𝑒𝑏𝑙𝑜𝑐𝑘 and𝑠𝑒𝑎𝑟𝑐ℎ𝑚𝑜𝑑𝑢𝑙𝑒𝑠𝑖𝑔𝑛𝑎𝑙 . The
output here can be either the name or code of different
abstraction levels, depending on the API definition.
TABLE I
LIST OF CODE RETRIEVAL API S
API name Description Output
𝑠𝑒𝑎𝑟𝑐ℎ𝑚𝑜𝑑𝑢𝑙𝑒 Search for module Name of module
𝑠𝑒𝑎𝑟𝑐ℎ𝑏𝑙𝑜𝑐𝑘 Search for block Code of block
𝑠𝑒𝑎𝑟𝑐ℎ𝑠𝑖𝑔𝑛𝑎𝑙 Search for signal Name of signal
𝑠𝑒𝑎𝑟𝑐ℎ𝑚𝑜𝑑𝑢𝑙𝑒𝑏𝑙𝑜𝑐𝑘 Search for block in the module Code of block
𝑠𝑒𝑎𝑟𝑐ℎ𝑚𝑜𝑑𝑢𝑙𝑒𝑠𝑖𝑔𝑛𝑎𝑙 Search for signal in the module Name of signal
B. HDLSearch Benchmark
Figure 10 demonstrates the template instructions used in the
HDLSearch benchmark refinement.
Through the annotations, we want to measure how relevant
would these results are to your Verilog design.
- You don't have to be absolutely certain about the correctness of
the code.
- You might be interested in copy-pasting the code, finding a
project to use or just getting some understanding about how
something is implemented.
- You might be searching within your project (e.g., to reuse
modules, signals, or testbench constructs) or to understand how a
particular digital circuit is structured.
Please annotate the results according to the following scheme:
- 3: Exact match . This Verilog snippet is exactly what I was
looking for . I would directly integrate it into my design with
minimal adaptations.
- 2: Str ong match . The snippet lar gely meets my requirements.
I might use it as a backbone for my hardware module, but some
modifications or additional verification might be needed.
- 1: Weak match . Although the snippet is not a perfect fit, it
contains useful structural elements, coding patterns, or testbench
ideas that could guide further exploration.
- 0: Totally irr elevant . This snippet does not address the query
or Verilog design challenge at all.Fig. 10. Two real-world issues posted in repository-level projects.
Table II describes the scales and classification of the pro-
posed HDLSearch benchmark. The benchmark contains 10
repository-level designs from FPGA projects to CPUs, includ-
ing coffee machine, CNN acc, image compression, AIB, IIC,
RIFFA, Ethernet, AXIS, MIPS and E203-hbirdv2.
TABLE II
HDLS EARCH BENCHMARK DESCRIPTION AND SCALES
Design Description Lines Modules Blocks
FPGA Project
coffee machine An FPGA-based coffee
machine control circuit890 8 26
CNN acc An FPGA-based CNN ac-
celerator866 9 22
image compression An FPGA-based JPEG
image compression circuit2340 18 79
Interconnection
AIB Advanced Interface Bus
protocol9994 14 184
IIC Inter-Integrated Circuit
protocol1540 2 11
RIFFA
Reusable Integration Framework for FPGA Accelerators protocol41279 5 24
Ethernet Ethernet protocol 130457 12 177
AXIS AXI-Stream protocol 15269 11 132
CPU
MIPS A MIPS RISC-V CPU 706 9 23
E203-hbirdv2 The second version of
the Hummingbird E203
RISC-V processor38577 6 88