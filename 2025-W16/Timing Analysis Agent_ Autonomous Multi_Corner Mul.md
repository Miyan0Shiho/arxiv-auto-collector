# Timing Analysis Agent: Autonomous Multi-Corner Multi-Mode (MCMM) Timing Debugging with Timing Debug Relation Graph

**Authors**: Jatin Nainani, Chia-Tung Ho, Anirudh Dhurka, Haoxing Ren

**Published**: 2025-04-15 04:14:36

**PDF URL**: [http://arxiv.org/pdf/2504.11502v1](http://arxiv.org/pdf/2504.11502v1)

## Abstract
Timing analysis is an essential and demanding verification method for Very
Large Scale Integrated (VLSI) circuit design and optimization. In addition, it
also serves as the cornerstone of the final sign-off, determining whether the
chip is ready to be sent to the semiconductor foundry for fabrication.
Recently, as the technology advance relentlessly, smaller metal pitches and the
increasing number of devices have led to greater challenges and longer
turn-around-time for experienced human designers to debug timing issues from
the Multi-Corner Multi-Mode (MCMM) timing reports. As a result, an efficient
and intelligent methodology is highly necessary and essential for debugging
timing issues and reduce the turnaround times. Recently, Large Language Models
(LLMs) have shown great promise across various tasks in language understanding
and interactive decision-making, incorporating reasoning and actions. In this
work, we propose a timing analysis agent, that is empowered by multi-LLMs task
solving, and incorporates a novel hierarchical planning and solving flow to
automate the analysis of timing reports from commercial tool. In addition, we
build a Timing Debug Relation Graph (TDRG) that connects the reports with the
relationships of debug traces from experienced timing engineers. The timing
analysis agent employs the novel Agentic Retrieval Augmented Generation (RAG)
approach, that includes agent and coding to retrieve data accurately, on the
developed TDRG. In our studies, the proposed timing analysis agent achieves an
average 98% pass-rate on a single-report benchmark and a 90% pass-rate for
multi-report benchmark from industrial designs, demonstrating its effectiveness
and adaptability.

## Full Text


<!-- PDF content starts -->

Timing Analysis Agent: Autonomous Multi-Corner
Multi-Mode (MCMM) Timing Debugging with
Timing Debug Relation Graph
Jatin Nainani
NVIDIA
Santa Clara, CA, USA
jnainani@nvidia.comChia-Tung Ho
NVIDIA Research
Santa Clara, CA, USA
chiatungh@nvidia.comAnirudh Dhurka
NVIDIA
Santa Clara, CA, USA
adhurka@nvidia.comHaoxing Ren
NVIDIA Research
Austin, TX, USA
haoxingr@nvidia.com
Abstract —Timing analysis is an essential and demanding
verification method for Very Large Scale Integrated (VLSI)
circuit design and optimization. In addition, it also serves as the
cornerstone of the final sign-off, determining whether the chip is
ready to be sent to the semiconductor foundry for fabrication.
Recently, as the technology advance relentlessly, smaller metal
pitches and the increasing number of devices have led to greater
challenges, and longer turn-around-time for experienced human
designers to debug timing issues from the Multi-Corner Multi-
Mode (MCMM) timing reports. As a result, an efficient and
intelligent methodology is highly necessary and essential for
debugging timing issues and reduce the turnaround times.
Recently, Large Language Models (LLMs) have shown great
promise across various tasks in language understanding and
interactive decision-making, incorporating reasoning and actions.
In this work, we propose a timing analysis agent, that is
empowered by multi-LLMs task solving, and incorporates a novel
hierarchical planning and solving flow to automate the analysis
of timing reports from commercial tool. In addition, we build a
Timing Debug Relation Graph (TDRG) that connects the reports
with the relationships of debug traces from experienced timing
engineers. The timing analysis agent employs the novel Agentic
Retrieval Augmented Generation (RAG) approach, that includes
agent and coding to retrieve data accurately, on the developed
TDRG. In our studies, the proposed timing analysis agent
achieves an average 98% pass-rate on a single-report benchmark
and a 90% pass-rate for multi-report benchmark from industrial
designs, demonstrating its effectiveness and adaptability.
Index Terms —Large Language Models, Autonomous Agents,
Multi-Agent Systems, VLSI, Static Timing Analysis
I. I NTRODUCTION
Very Large Scale Integration (VLSI) circuit design and
optimization require extensive timing analysis and debugging
to verify functional correctness. Today, timing analysis serves
as the cornerstone of the final sign-off process, determining
whether the design is ready for chip fabrication. As technology
has advanced beyond 5 nm, the growing number of transistors,
increasingly complex circuit designs, and the amplified effects
of cross-talk due to smaller metal pitches have introduced
significant challenges. These factors lead to longer debugging
times for experienced timing analysis engineers who must
address timing issues based on Multi-Corner Multi-Mode
(MCMM) timing reports.
UserDebug worst xtalk  net 
across all corners and 
modes?
MCMM Planning: 
MCMM Planner Agent
Timing Analysis Agent: Hierarchical Plan Solving + Multi -Agent Collaboration 
Multi -Report Debug 
Planning: TDRG 
Traversal AgentAgentic Retrieval: 
Expert Report Agent
Report Analysis ResultsAnalysis Results of 
each corner modeFinal Answer
Max
XtalkClk Min
LC WireStructural Report Database
(MCMM timing reports)Distilled Timing Debug Relation Graph
⋯
⋯
Multi -Report Debug 
Relation Knowledge Retrieval
Fig. 1: An illustration of Timing Analysis Agent, which integrates hierarchical
planning, multi-agent collaborations, and the novel distilled Timing Debug
Relation Graph to solve MCMM timing task.
Recent advances have explored the use of Large Language
Models (LLMs) as agents for interactive decision-making [1].
However, MCMM timing report analysis remains an unsolved
and challenging task because the context of multi-report
analysis exceeds the token limits of current state-of-the-art
LLMs and requires a wide variety of queries across different
types of timing reports. Although recent works [2], [3], [4]
have proposed repository-level agent frameworks for tasks like
code planning and coding assistance, these solutions lack the
domain knowledge necessary to handle various report types
(e.g., path delay, cross-talk, logic constraints) simultaneously.
Consequently, an efficient and intelligent methodology is es-
sential to streamline the debugging of timing issues and reduce
turnaround time.
In this paper, we propose a novel and intelligent Timing
Analysis Agent, which integrates multi-LLM task-solving,
hierarchical planning, and the expertise of experienced timing
engineers to generate insightful analysis and debug timing
issues with high accuracy and minimal variance, as shown in
Fig. 1. The proposed methodology aims at solving the general
timing analysis and debug problems of MCMM timing reports.
The contributions of our work include:
1) We propose a novel Timing Analysis Agent that in-
tegrates hierarchical plan solving and multi-agent col-
laboration to automate the analysis of MCMM timing
reports generated by timing verification tools (e.g., Static
Timing Analysis (STA), variation analysis, etc.).arXiv:2504.11502v1  [cs.SE]  15 Apr 2025

2) We distill the debugging trace into a Timing Debug
Relation Graph (TDRG), which connects individual re-
ports with debug knowledge from experienced timing
engineers. The proposed Timing Analysis Agent lever-
ages the TDRG to dynamically traverse and retrieve
information on nets, paths, or instances across multiple
timing reports to solve tasks.
3) We introduce a novel variation of Agentic Retrieval
Augmented Generation (RAG), which leverages the cod-
ing capabilities of LLMs to retrieve necessary timing
information from the reports, excluding irrelevant data.
This approach ensures the retrieval is adaptable to a wide
range of queries.
4) We develop the MCMM planner agent, the TDRG
traversal agent, and the expert report agent to solve tasks
hierarchically—starting with mode planning, followed
by multi-report traversal, single-report information re-
trieval, and culminating in providing the final answer.
5) We validate the effectiveness of our proposed Timing
Analysis Agent through extensive experiments. The re-
sults show that our approach outperforms other RAG
techniques by more than 46% and achieves a 98% pass-
rate on average for single-report benchmark and a 90%
pass-rate for multi-report benchmark.
The remaining sections are organized as follows: Section II
reviews the related works on leveraging LLMs for multi-
file analysis, retrieval, and applications in Electronic Design
Automation (EDA) field. Section III discusses the timing
analysis tasks in detail. Then, we introduces the proposed
Timing Analysis Agent methodology in Section IV. Section V
presents our main experiment and sensitivity studies. Lastly,
Section VI concludes the paper.
II. L ITERATURE REVIEW
We review related works on multi-file analysis, structural
data analysis, and LLM applications in EDA below.
A. Multi-file Analysis and Editing
Bairi et al. [2] implemented repository-level coding through
a task-agnostic framework called CodePlan, which frames
coding as a planning problem. This system synthesizes a
multi-step plan of interdependent edits across a repository
by combining static analysis, change-impact assessment, and
adaptive planning. Their method allows LLMs to handle com-
plex tasks such as package migration and temporal code ed-
its, outperforming simpler oracle-guided systems by ensuring
consistency throughout the repository. Liu et al. [3] extended
repository-level code completion with STALL+, which inte-
grates static analysis techniques to enhance LLM performance.
By leveraging static analysis for dependency identification,
STALL+ improves context comprehension and boosts LLM
accuracy on large repositories. Luo et al. [4], with RepoA-
gent, focused on repository-level documentation generation.
Their framework uses LLMs to analyze code and generate
comprehensive documentation across entire repositories, fur-
ther demonstrating that LLMs can manage complex multi-
file tasks. These works illustrate the growing capability of
Fig. 2: Hierarchical relations of the MCMM timing reports. There are multiple
PVT corners and modes, like TT mode1, SS mode3, etc. Each of corner and
mode has max, min, xtalk max, xtalk min, clk, freq, LC, and wire reports.
LLM agents in handling repository-wide tasks. However, the
domain-specific complexity of MCMM timing reports, which
requires expert trace distillation and cross-corner reasoning,
remains a challenge.
B. Structured Data Analysis with Large Language Models
The application of LLMs to structured data remains a
significant challenge due to the complexity and scale of such
datasets. Recently, several works have focused on efficient
retrieval methodologies, Ziletti et al. [5] showcase how LLM,
using the retrieval augmented text-to-SQL technique, can be
applied to answer epidemiological questions using EHR and
claims data. Dong and Wang [6] review the current state of
LLMs for tabular data, highlighting progress in fine-tuning
and data representation, while also emphasizing limitations
in handling large, heterogeneous datasets without significant
adaptation. Jiang et al. [7] propose StructGPT to improve
reasoning over structured data through prompt engineering.
These methods usually assume a relational database is already
present and therefore can not be directly applied to efficiently
retrieve data from MCMM timing reports. Additionally, Sui
et al. [8] benchmark LLMs on tabular data, revealing that
while LLMs are promising, they underperform on complex
datasets typical in EDA. These findings suggest that despite
recent advancements, LLMs require further refinement to be
effective in large-scale, multi-file analysis tasks.
C. Large Language Models in Electronic Design Automation
The integration of Large Language Models (LLMs) in EDA
has shown promise across various tasks. Chang et al. [9]
developed a framework that utilizes data augmentation to
fine-tune LLMs for chip design, particularly in generating
Verilog code and EDA scripts. This framework improves the
alignment between natural language and hardware description
languages, enhancing the model’s ability to generate and
repair Verilog code. Ho and Ren [10], [11] explored the
application of LLMs in optimizing standard cell layout design
and Verilog coding, demonstrating the potential for integrating
domain-specific knowledge into layout tasks. Similarly, Liu
et al. [12] introduced LayoutCopilot, a multi-agent collabo-
rative framework for interactive analog layout design, which
leverages LLMs to streamline the design process. Additionally,
Wu et al. [13] presented ChatEDA, an approach that uses

Path     Path
Item     Slack    Delay  Startpoint     Endpoint
--------  --------  --------   ---------------  ---------------
1         -0.1170   0.3067     < s_pt >     f  < e_pt >
2         -0.1169   0.3067     < s_pt >     f  < e_pt >
. . .
k        -0.1163   0.3067      < s_pt >     f  < e_pt >
. . .
Item:              5033
Path ID:           33126
Startpoint :        < input_port >
Endpoint:          < instance_pin >
Path Type:         max
Constraint:        <constraint>
                                              Wire                 Xtalk
    Path     Incr   Adjust    Delay      Var    Delta    Trans  …         Point
--------  --------  --------  --------  --------  --------  --------  -----------------------------
                             0.0000                                                                   clock (rise)
                             0.1840                                                             input external delay
  0.1840              0.0000                             0.0250   0.0015              <net>
 . . .
  0.8526                                                                                          data required time
------------------------------------------------------------------------------------------
  0.8526                                                                                          data required time
 -0.2016                                                                                          data arrival time
------------------------------------------------------------------------------------------
  0.6510                                                                                          slack (MET)(a) Max Report Example
****************************************
Report : <Report_Name >
Design : <Design_Name >
****************************************
                       rising   falling
 coupling   percent aggressor aggressor  max delta  max percent
      cap  of total     trans     trans      delay contribution Victim net    aggressor net   
---------  ---------  ---------  ---------  ---------  ------------  ------------------------ -------------------- --
   0.0004     24.96    0.0086    0.0073    0.0019      26.8936 <net3>             <net0>
   0.0004     24.95    0.0077    0.0065    0.0019      30.2786 <net3>    <net1>
   . . .
  
   0.0000      2.88    0.0089    0.0075   0.0019       2.9988 <net3>    <net2>
. . .
****************************************
Report : < Report_Name >
Design : < Design_Name >
****************************************
                                            rising        falling
 coupling   percent aggressor aggressor  max delta  max percent
           cap  of total     trans                trans           delay  contribution   Victim net   aggressor net  
---------  ---------  ---------  ---------  ---------  ------------  ----------------------------------------------
   0.0004     24.10       0.0085          0.0072      0.0019      26.1935           <net1>             <net5>
   0.0004     24.08       0.0076          0.0064      0.0019      29.5695           <net1>             <net2>
   . . .
  
   0.0000      2.98        0.0090          0.0074      0.0019         3.0867            <net1>           <net11>(b) Xtalk  Max Report ExampleFig. 3: Examples of max and xtalk max timing report for a specific corner and mode.
LLMs to autonomously generate and interact with EDA scripts
within the OpenROAD environment, showcasing the potential
for automating EDA workflows. While these advancements
highlight the capabilities of LLMs in EDA, the focus has
largely been on isolated tasks such as layout optimization,
script generation, and Verilog code synthesis.
In summary, the prior related works can not be applied for
complex timing debugging and analyses of MCMM timing
reports since it requires cross-referencing and multi-hop rea-
soning across reports with expert domain knowledge.
III. B ACKGROUND
Here, we introduce the background of timing reports and the
selected real world timing tasks from expert timing engineers.
We create the benchmark based on the selected timing tasks
for evaluation in the experiments.
A. Timing Reports
Timing analysis tools cover gate-level STA, detailed
transistor-level STA, and variation analysis for VLSI designs
including intricate custom designs. Silicon failures in advanced
technologies like FinFET can be costly, making rigorous
signoff analysis crucial to avoid critical timing and noise
issues. Timing analysis is typically run across the entire circuit,
which contains multiple corners. Each corner includes modes
like read, write, scan, etc. For each corner and mode, there are
max, min, xtalk max, xtalk min, clk, freq, Logic Constraints
(LC) and wire types of report for timing engineer to debug
the timing issues.
Fig 2 illustrates the hierarchical relations of the MCMM
timing reports. Fig 3 provides examples of max, and
xtalk max timing reports that usually contain more than
16,000 timing paths for an industrial design. In Fig 3, the
format and attributes of the timing reports vary, posing chal-
lenges for LLMs when retrieving the necessary information
for different timing tasks.B. Timing Tasks
We outline the timing tasks selected by experienced timing
engineers to create a benchmark from industrial designs for
evaluating performance, categorizing them into single-report
and multi-report tasks.
Single-Report Tasks : Single-report tasks require domain
knowledge of the type of report being analyzed and an
understanding of its intricate structure. We divide single-report
tasks into two categories and provide examples below.
Query or group timing paths based on specific criteria : These
tasks commonly involve retrieving paths between a ”start-
point” and an ”endpoint,” identifying paths with specific ”Con-
straint” values, or finding paths that start with specific edges
across the data and clock arcs. Some tasks require performing
complex queries to retrieve specific sets of problematic paths
that need further investigation, while others focus on deeply
examining individual paths to identify issues.
Perform mathematical or string operations on a group of
paths : These tasks usually involve performing operations (e.g.,
max, min, avg) on columns or high-level attributes. For
example, one might compare the worst values of different
attributes across multiple paths or find the top-k paths with
timing issues.
Multi-Report Tasks : The process of completing multi-report
tasks often involves gathering data from various types of
reports (e.g., max, xtalk max, etc.) and performing reasoning
to derive the final conclusion. For example, detecting missing
clock signals in a timing path for a specified corner and
mode requires cross-reference the clock signals in the clk
report and timing tables in the max report. Another example
involves identifying unusual RC values between victim and
aggressor nets in a timing path. This first requires finding
the most significant ”aggressor” and ”victim” nets in the
xtalk report. Then, the next step is investigating the high
RC mismatch and any abnormal logic constraints between
the identified aggressor and victim nets in the wire and LC

UserDetect unusual logic constraints between 
victim and aggressors of path A in TT_read  
and SS_read  corner mode.
MCMM Planning: MCMM Planner Agent (Level 1)
- Break up task into subtasks and make plan for each corner and mode
- Retrieve information from lower -level agent and Reason to provide Final 
AnswerTiming Analysis Agent: Hierarchical Plan Solving + Multi -Agent Collaboration 
Multi -Report Debug Planning: TDRG Traversal Agent (Level 2)
- Break up task into subtasks and make plan for retrieving information 
from reports (i.e., max, min, xtalk , etc.) on TDRG
- Retrieve information from lower -level agent and Reason to provide 
answer for each subtask
Agentic Retrieval: Expert Report Agent (Level 3)
- Write python code to retrieve information from various type of reports 
for each subtasks
- Return the retrieved information
Report Analysis 
ResultsAnalysis Results of 
each corner modeFinal Answer(a) Flow Overview
(c) Structural Report Database for Agentic Retrieval
Item 1
Item 2
Path kSummary:  { “item”: ..,
      “Start point”: ..,
      “slack”: .., 
       ….}
Data Stages: 
Data Info:  { “PBSA adjustment”: ..
                      “data arrival time”:
      … }
Clock Stages: 
Clock Info: { “PBSA adjustment”: ,
     “data arrival time”:     
… }Timing TableDelay Incr…… Point Net
0.13 0.0 … Start Pt Net1
0.31 0.29 … Net1 Net2
0.24 0.4 .. Net2 Net3
0.45 0.69 .. Net3 End Pt
Delay Incr……. Point Net
0.09 0.0 … Start Pt Net1
0.31 0.29 … Net1 Net2 
0.15 0.12 .. Net2 Net3
0.65 0.41 .. Net3 End PtData Arc
Clock Arc⋮Max Report
Timing TablePath 1
Path 2
⋮(b) Timing Debug Relation Graph (TDRG): Distilled Debug Traces
LC
Xtalk
Max/MinMax/Min
ClkWire
NetsAggressorsLogic 
Constraints
Missing Clocks
Clock InfoPair of Nets
RC mismatchProblematic Nets Report
Debug Relations 
between ReportsFig. 4: (a) Flow overview of Timing Analysis Agent with hierarchical plan solving, and multi-agent collaboration. (b) Timing Debug Relation Graph (TDRG)
from distilled debug traces from experienced timing engineer. (c) Structural report database for coding agentic retrieval.
reports, respectively. Here, the RC mismatch is determined
by subtracting the worst-case RC values. To perform analysis
across MCMM, the multi-report task must be conducted in
various corners and modes. The results are then summarized,
providing explanations for any potential causes of identified
problems across all corners and modes.
IV. T IMING ANALYSIS AGENT
The Timing Analysis Agent integrates hierarchical plan
solving, and multi-agent collaboration, as shown in Fig 4(a).
Given a user task (e.g., violation search or identifying nets
with significant cross-talk impact), the MCMM Planner Agent
first decomposes the task into a list of corner and mode
specified sub-tasks. Then, the TDRG Traversal Agent makes
a report retrieval plans, calls Expert Report Agents to retrieve
information, and summarizes the fetched net delay, path delay,
and cross-talk data from multiple reports on the proposed
novel distilled TDRG as shown in Fig 4(b). Finally, the
MCMM planner agent compiles the responses of sub-tasks,
and returns the Final Answer. We introduce the structural
report database and the framework in a bottom-up approach.
A. Structural Report Database
Given the MCMM timing reports as input, we utilize
the original structure of each timing report to construct a
structured report database, organized using nested dictionaries
for each report type. Figure 4(c) illustrates an example of
a structured database for a max report. In this database,
the max report is organized into key attributes: ”Summary,”
”Data Info,” ”Clock Info,” ”Data Stages,” and ”Clock Stages,”
allowing for efficient and informed access. ”Summary” at-
tribute contains high-level path information, such as startpoint,
endpoint, path ID, slack, etc. The ”Data Info” and ”Clock Info”
attributes summarize timing arc information, including PBSA
adjustments and data arrival times. Finally, the detailed timingtable for each timing arc is stored in the ”Data Stages” and
”Clock Stages” as shown in the rightmost part of Figure 4(c).
B. Timing Debug Relation Graph (TDRG)
We construct the TDRG based on a distilled debugging trace
provided by experienced timing engineers, enabling the agent
to develop a plan for retrieving information from multiple tim-
ing reports. In the TDRG, nodes represent the reports, while
edges define the relationships between the debug information,
as shown in Figure 4(b). For each node, we describe the
timing attributes, type of timing table, and the usage of the
corresponding report. For example, the description of a wire
report node includes details such as the ”Worst RC” attributes
and the calculation of RC mismatch. On the other hand,
the edge descriptions capture distilled knowledge from the
debugging trace, describing the relationships between different
types of reports. For instance, the edge relationship between
xtalk and LC reports is ”find the logic constraints on the
aggressor and victim nets.”
C. Expert Report Agent: Coding Agentic Retrieval (level 3)
We develop a novel Expert Report Agent that leverages
the coding capabilities of LLMs to retrieve the needed timing
information without including extraneous data that could fill
the context limit. Inspired by the findings of Wang et al. [14],
which demonstrate that agents capable of executing code
outperform other methods, we designed the Expert Report
Agent to tailor its functionality to each report type (e.g., max,
crosstalk, etc.). The agent generates flexible and adaptable
Python code to query the structural report database according
to the specific task requirements.
Figure 5 illustrates an example where the Expert Report
Agent writes Python code to retrieve the path ID of the
minimum slack from the max report. Given the task query,

Query TaskFig. 5: An example of Expert Report Agent writing Python code to retrieve
path ID of the minimum slack in max report.
Max Expert Report AgentTDRG Traversal 
AgentUser
MCMM Planner 
Agent
MaxXtalk
MaxLC
Xtalk  Max Expert Report Agent
 LC Expert Report AgentNets Problematic Nets
Planned Report Trajectory: Max -> 
Xtalk  Max -> LCAh! I should check ‘net_1’ and the logic 
constraints between ‘net_1’ and ‘ net_agg ’.
Planned Corner_Mode  Trajectory: 
TT_read
Final Answer: 
1. ‘net_1’ is the most unusual net in the path
2. The most significant aggressor is ‘ net_agg ’
3. The logic constraint between them is missing
Which might explain the unusual xtalk  of ’net1’Detect unusual logic constraints between 
victim and aggressors of path 282613 in 
TT_read  corner mode.
Thought: I am on Max report
Action: Call the Max report agent 
with task “Find the unusual net for 
Xtalk  for Path 282613” Thought: I am on Xtalk  Max report
Action: Call the Xtalk  Max report 
agent with task “Find most 
significant aggressors for net ‘net_1’” Thought: I am on LC report
Action: Call the LC report agent with 
task “Find logic constraints on ‘net1’ 
and ‘ net_agg ’” 
1. ‘net_1’ is the most unusual net in the path
2. The most significant aggressor is ‘ net_agg ’
3. The logic constraint between them is missing
Fig. 6: Illustration of the hierarchical plan solving and multi-agent collabora-
tion in solving the multi-report task.
the Expert Report Agent first writes the python code for
execution. After retrieving the information, the Expert Report
Agent summarizes the retrieved information and provides the
answer to the query.
D. TDRG Traversal Agent: Multi-Report Debugging (level 2)
The leverages the distilled TDRG to create a plan to retrieve
information (i.e., unusual nets, aggressors, and logic con-
straints, etc.), and then summarizes the results to complete the
task. The TDRG Traversal Agent calls Expert Report Agents
to retrieve the unusual nets in max report, aggressors in xtalk
max report, and logic constraints in LC report, respectively.
E. MCMM Planner Agent: MCMM Planning (level 1)
The MCMM Planner Agent decomposes the user-provided
MCMM timing task into sub-tasks for each corner and mode
based on the prompt from user. Then, MCMM Planner Agent
calls TDRG Traversal Agent to resolve each sub-task sequen-
tially. Finally, the MCMM Planner Agent collects the analysisTABLE I: Pass-rate (%) of keyword search [17], embed search [18], hybrid
search [19], functional agentic retrieval (Agentic Function), and proposed
approach (Agentic Coding) on single-report benchmark table.
General Task Category #TasksKeyword
[17]Embed
[18]Hybrid
[19]Agentic
FunctionAgentic
Coding
(proposed)
Check path for violation 10 0.0 0.0 0.0 100.0 100.0
Find worst case
<attribute >across paths10 0.0 0.0 0.0 90.0 100.0
Find worst case
<column >across paths10 0.0 0.0 0.0 70.0 90.0
Check if a given path is
external or internal10 0.0 0.0 0.0 80.0 100.0
Which is the slowest
stage in the whole path10 0.0 0.0 0.0 20.0 100.0
Net with max crosstalk
delta in path10 0.0 0.0 0.0 30.0 90.0
Slew on the <net>
in path10 0.0 0.0 0.0 0.0 100.0
Path goes through
<net>?10 0.0 0.0 0.0 50.0 100.0
Data arc goes through
<clk>rising?10 0.0 0.0 0.0 30.0 100.0
Average 10 0.0 0.0 0.0 52.0 97.8
results for each planned corner and mode from the TDRG
Traversal Agent and provides the final answer to the user.
Fig 6 illustrates the hierarchical plan solving and multi-
agent collaboration in solving the task ”Detect unusual logic
constraints between victim and aggressors of path 282613
in TT read corner mode” task. The MCMM Planner Agent
makes a planned trajectory of mode and corner and instructs
TDRG Traversal Agent for completing the task. The TDRG
Traversal Agent generates sub-tasks for multi-report retrieval.
The Expert Report Agents retrieve the unusual net, aggressor,
and logic constraint, then return this information to TDRG
Traversal Agent. Lastly, the MCMM Planner Agent summa-
rizes the analysis messages from TDRG Traversal Agent and
provides the Final Answer to user.
V. E XPERIMENTAL RESULTS
Our work is implemented in Python and is built on top
of the Autogen [15] multi-LLM agent framework. We use
Llama3 [16] for each agent. The temperature and top p param-
eters of the LLM are set to 0.3 and 1.0, respectively. We create
a single-report benchmark to evaluate retrieval effectiveness
and correctness, and a multi-report benchmark to assess the
hierarchical planning and reasoning capabilities of the agent
from industrial designs, as described in Section III. The pass
rates of the single-report and multi-report benchmarks are
compared to the golden answer and evaluated by experienced
human engineers. There are 10 tasks for each general task cat-
egory in the single-report benchmark. Firstly, we demonstrate
the effectiveness of the proposed coding agent retrieval on the
single-report benchmark. Next, we present the pass rate of the
proposed Timing Analysis Agent in multi-report debugging
and reasoning. Lastly, we conduct a sensitivity study on the
impact of information from TDRG on multi-report tasks.
A. Single-Report Experiment
We demonstrate the pass-rate of the novel coding agentic re-
trieval method and compare it with prior works using a single-
report benchmark. Each task is performed once, and the aver-
age pass rate for each general task category is calculated. we

TABLE II: Pass-rate (%) of the proposed approach, and a naive LLM planner
without TDRG on multi-report benchmark table.
Task ID Task DescriptionRequired types
of reportsLLM Planner
wo GraphProposed
M1Find missing clk signals that
have no rise/fall informationmax, clk X V
M2Identify pairs of nets with
high RC mismatchmax, wire X V
M3Detect unusual constraints
between victim and its aggressorsmax, xtalk,
LCX V
M4Identify unusual RC values
between victim and its aggressorsmax, wire,
xtalk, LCX V
M5Find the constraints of slowest
stages with highest RC valuesmax, wire,
xtalk, LCX V
M6Compare each timing table for
number of stages, point values
and timing mismatchmax X X
M7Task M2 and Task M3 for
specific stages in list of pathsmax, wire,
xtalk, LCX V
M8 Task M1 across all modes max, clk X V
M9 Task M3 across all modesmax, xtalk,
LCX V
M10Task M2 and Task M3 across
all modesmax, wire,
xtalk, LCX V
Average pass-rate (%) - 0.0 90.0
compare the proposed coding agentic retrieval approach with
Keyword Search [17], Embedding search [18], and a hybrid
search [19]. In addition, we compare the proposed approach
with functional agentic retrieval approach to further show the
flexibility and effectiveness of the proposed coding agentic
retrieval method for dynamic task content. Basic functions
(i.e., ”check path forviolation,” ”get specific attribute,” and
”get total column value”) are implemented to extract infor-
mation from the structural database of reports for the func-
tional agentic retrieval approach.
Table I shows the pass-rate of the proposed coding agentic
retrieval approach (i.e., Agentic Coding) and other baseline
methods. Keyword Search [17], Embedding search [18], and
hybrid search fail to retrieve the specific columns or values
required for the task, as these unstructured approaches either
retrieve entire timing tables or unrelated information. Com-
pared to functional agentic approach (i.e., Agentic Function),
the proposed coding agentic approach achieves 45.8% higher
pass-rate on average of all the general task categories since
functional agentic approach is limited by predefined functions
and cannot scale to the diverse types of queries that involve
many combinations of basic column or value extractions. On
the other hand, the proposed coding agentic approach scales
to various combinations of basic queries without requiring
predefined functions for retrieval. Overall, the proposed coding
agentic approach achieves 98% pass-rate on average for single-
report benchmark.
B. Multi-Report Experiment
We study the problem solving, and reasoning abilities of the
proposed Timing Analysis Agent on multi-report benchmark
in Table II, which includes task descriptions and the required
types of reports for each task. As we are first to approach
the multiple timing reports debugging and analysis tasks, we
compare the proposed Timing Analysis Agent with a naive
LLM planner that does not utilize TDRG. The proposed
Timing Analysis Agent achieves a 90% pass-rate, whereas
the naive LLM planner struggles to solve any of the multi-
report tasks due to its lack of debug trace integration between
Fig. 7: Pass-rate of varying node and edge descriptions for selected multi-
report tasks (i.e., from task M1 to task M5).
timing reports, which is necessary for creating accurate plans.
In summary, the proposed Timing Analysis Agent successfully
and effectively resolve the multi-report tasks using the novel
hierarchical plan solving process and distilled TDRG from
debug traces of experienced timing engineers.
C. Sensitivity Study of TDRG
We conducted an extensive study to evaluate the impact
and sensitivity of node descriptions and edge information
on TDRG across five multi-report tasks, each specified by a
particular corner and mode (i.e., from Task M1 to Task M5)
as detailed in Table II. We vary the node description and edge
information for agent at level 2, as shown in Figure 4(a),
to make a plan for solving multi-report tasks. The different
settings are as follows:
1)Set1: No information about reports and their relations.
2)Set2: Limited node description only.
3)Set3: Limited edge description only.
4)Set4: Limited node and edge descriptions.
5)Set5: Detailed node and limited edge descriptions.
6)Set6: Detailed edge limited node descriptions.
7)Proposed : Detailed description for node and edge.
The average number of words for limited descriptions of nodes
and edges are 12.5 and 8, respectively, while the average
number of words for detailed descriptions of nodes and edges
are 42.5 and 20, respectively.
Figure 7 shows the pass-rate of various settings for the node
and edge descriptions. Without plan examples for the multi-
report tasks (i.e., the red dashed line), the Proposed setting
achieves an 80% pass rate. Detailed node descriptions have a
significant impact on the pass rate, as seen in the improvements
from Set3 toSet5 and from Set6 toProposed . When plan
examples are provided (i.e., the blue solid line), the limited
edge and node descriptions (i.e., Set4) also achieve an 80%
pass rate. Adding any detailed information about nodes or
edges increases the pass rate to 100%. This sensitivity study
highlights the importance of node and edge descriptions in
solving multi-report tasks and motivates the continual learning
on optimizing the TDRG to further improve multi-report task.
VI. C ONCLUSION
Our proposed Timing Analysis Agent demonstrates ad-
vanced capabilities in hierarchical plan solving and reasoning
for complex multi-report tasks from industrial designs through

the innovative distilled Timing Debug Relation Graph (TDRG)
and coding agentic retrieval methodology. Firstly, we show
that the proposed flexible and adaptive coding agentic retrieval
method not only achieves a 98% pass rate but also outperforms
other retrieval methods by more than 46% pass-rate on single-
report benchmarks. Next, we demonstrate that the Timing
Analysis Agent achieves a 90% pass rate on multi-report
benchmarks, as evaluated by experienced human engineers.
Finally, we have extensively studied the effectiveness and
importance of node and edge descriptions within the proposed
Timing Debug Relation Graph (TDRG) through a sensitivity
analysis of the multi-report benchmarks.
REFERENCES
[1] Shunyu Yao, Jeffrey Zhao, Dian Yu, Nan Du, Izhak Shafran, Karthik
Narasimhan, and Yuan Cao. React: Synergizing reasoning and acting in
language models. arXiv preprint arXiv:2210.03629 , 2022.
[2] Ramakrishna Bairi, Atharv Sonwane, Aditya Kanade, Arun Iyer, Suresh
Parthasarathy, Sriram Rajamani, B Ashok, and Shashank Shet. Code-
plan: Repository-level coding using llms and planning. Proceedings of
the ACM on Software Engineering , 1(FSE):675–698, 2024.
[3] Junwei Liu, Yixuan Chen, Mingwei Liu, Xin Peng, and Yiling Lou.
Stall+: Boosting llm-based repository-level code completion with static
analysis. arXiv preprint arXiv:2406.10018 , 2024.
[4] Qinyu Luo, Yining Ye, Shihao Liang, Zhong Zhang, Yujia Qin, Yaxi
Lu, Yesai Wu, Xin Cong, Yankai Lin, Yingli Zhang, et al. Repoagent:
An llm-powered open-source framework for repository-level code doc-
umentation generation. arXiv preprint arXiv:2402.16667 , 2024.
[5] Angelo Ziletti and Leonardo D’Ambrosi. Retrieval augmented text-to-
sql generation for epidemiological question answering using electronic
health records. arXiv preprint arXiv:2403.09226 , 2024.
[6] Haoyu Dong and Zhiruo Wang. Large language models for tabular
data: Progresses and future directions. In Proceedings of the 47th
International ACM SIGIR Conference on Research and Development
in Information Retrieval , pages 2997–3000, 2024.
[7] Jinhao Jiang, Kun Zhou, Zican Dong, Keming Ye, Wayne Xin Zhao,
and Ji-Rong Wen. Structgpt: A general framework for large language
model to reason over structured data. arXiv preprint arXiv:2305.09645 ,
2023.
[8] Yuan Sui, Mengyu Zhou, Mingjie Zhou, Shi Han, and Dongmei Zhang.
Table meets llm: Can large language models understand structured table
data? a benchmark and empirical study. In Proceedings of the 17th
ACM International Conference on Web Search and Data Mining , pages
645–654, 2024.
[9] Kaiyan Chang, Kun Wang, Nan Yang, Ying Wang, Dantong Jin, Wen-
long Zhu, Zhirong Chen, Cangyuan Li, Hao Yan, Yunhao Zhou, et al.
Data is all you need: Finetuning llms for chip design via an automated
design-data augmentation framework. arXiv preprint arXiv:2403.11202 ,
2024.
[10] Chia-Tung Ho and Haoxing Ren. Large language model (llm) for stan-
dard cell layout design optimization. arXiv preprint arXiv:2406.06549 ,
2024.
[11] Chia-Tung Ho, Haoxing Ren, and Brucek Khailany. Verilogcoder:
Autonomous verilog coding agents with graph-based planning and
abstract syntax tree (ast)-based waveform tracing tool. arXiv preprint
arXiv:2408.08927 , 2024.
[12] Bingyang Liu, Haoyi Zhang, Xiaohan Gao, Zichen Kong, Xiyuan Tang,
Yibo Lin, Runsheng Wang, and Ru Huang. Layoutcopilot: An llm-
powered multi-agent collaborative framework for interactive analog
layout design. arXiv preprint arXiv:2406.18873 , 2024.
[13] Bing-Yue Wu, Utsav Sharma, Sai Rahul Dhanvi Kankipati, Ajay Yadav,
Bintu Kappil George, Sai Ritish Guntupalli, Austin Rovinski, and
Vidya A Chhabria. Eda corpus: A large language model dataset for
enhanced interaction with openroad. arXiv preprint arXiv:2405.06676 ,
2024.
[14] Xingyao Wang, Yangyi Chen, Lifan Yuan, Yizhe Zhang, Yunzhu Li, Hao
Peng, and Heng Ji. Executable code actions elicit better llm agents. arXiv
preprint arXiv:2402.01030 , 2024.[15] Qingyun Wu, Gagan Bansal, Jieyu Zhang, Yiran Wu, Shaokun Zhang,
Erkang Zhu, Beibin Li, Li Jiang, Xiaoyun Zhang, and Chi Wang. Au-
togen: Enabling next-gen llm applications via multi-agent conversation
framework. arXiv preprint arXiv:2308.08155 , 2023.
[16] Meta. meta-llama/llama3. Original-date: 2024-03-15T17:57:00Z. https:
//build.nvidia.com/meta/llama3-70b, 2024.
[17] K Sparck Jones, Steve Walker, and Stephen E. Robertson. A prob-
abilistic model of information retrieval: development and comparative
experiments: Part 2. Information processing & management , 36(6):809–
840, 2000.
[18] Patrick Lewis, Ethan Perez, Aleksandra Piktus, Fabio Petroni, Vladimir
Karpukhin, Naman Goyal, Heinrich K ¨uttler, Mike Lewis, Wen-tau Yih,
Tim Rockt ¨aschel, et al. Retrieval-augmented generation for knowledge-
intensive nlp tasks. Advances in Neural Information Processing Systems ,
33:9459–9474, 2020.
[19] Ashish Abraham, M ´or Kapronczay, and Robert Turner. Optimizing RAG
with Hybrid Search & Reranking. https://superlinked.com/vectorhub/
articles/optimizing-rag-with-hybrid-search-reranking, 2024.