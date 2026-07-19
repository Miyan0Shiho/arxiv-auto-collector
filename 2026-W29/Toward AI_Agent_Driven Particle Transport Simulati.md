# Toward AI-Agent-Driven Particle Transport Simulations: Implementation of AI-Assisted Workflows for PHITS

**Authors**: Tatsuhiko Sato, Shintaro Hashimoto, Tatsuhiko Ogawa, Takuya Furuta, Yuho Hirata, Seiki Ohnishi, Yasuhito Sakaki, Nobuhiro Shigyo

**Published**: 2026-07-13 09:22:19

**PDF URL**: [https://arxiv.org/pdf/2607.11309v1](https://arxiv.org/pdf/2607.11309v1)

## Abstract
Monte Carlo particle transport codes are powerful tools, but their use requires substantial knowledge of input preparation, execution, and result analysis. In this study, we present a code-side strategy for applying existing AI assistants and AI agents to PHITS. Two complementary sets of AI-ready resources were prepared from manuals, lecture materials, sample inputs, utility information, and developer-curated cautions: a bundled knowledge base for retrieval-augmented generation (RAG)-based assistants and a compact agent reference for direct use by AI agents. The knowledge base was loaded into NotebookLM to provide conversational PHITS support, while the agent reference was combined with PHITS-specific policies and execution rules to enable Codex and Claude Code to edit input files, execute calculations, inspect errors, analyze results, and assist with source-code modification and compilation. Five demonstration tasks covered input modification, repeated simulations, parameter optimization, program compilation, post-processing, and result interpretation. The results showed that AI agents could handle complex PHITS workflows when appropriate resources and rules were provided. Practical lessons included precise prompts, human verification, well-documented sample files, explicit execution policies, and command-line-accessible tools. These findings support bundling AI-ready resources with particle transport codes to enable the use of general-purpose AI tools without requiring dedicated code-specific applications.

## Full Text


<!-- PDF content starts -->

1 
 Toward AI -Agent -Driven Particle Transport Simulations: 
Implementation of AI- Assisted Workflows for PHITS  
Tatsuhiko Satoa*, Shintaro Hashimotoa, Tatsuhiko Ogawaa, Takuya Furutaa, 
Yuho Hirataa, Seiki  Ohnishia, Yasuhito Sakak ib, Nobuhiro Shigyoc 
aJapan Atomic Energy Agency, Shirakata 2- 4, Tokai, Ibaraki 319- 1195, Japan;  bHigh 
Energy Accelerator Research Organization, Oho 1- 1, Tsukuba, Ibaraki, 305- 0801, 
Japan; cKyushu University, Department of Applied Quantum Physics and Nuclear 
Engineering, Motooka 744, Nishi, Fukuoka, 819- 0395, Japan;  
*Corresponding author  
Abstract  
Monte Carlo particle transport codes are powerful tools, but their use requires 
substantial knowledge of input preparation, execution, and result analysis. In this study, 
we present a code- side strategy for applying existing AI assistants and AI agents to 
PHITS. Two complementary sets of AI -ready resources were prepared from manuals, 
lecture materials, sample inputs, utility information, and developer -curated cautions: a 
bundled knowledge base for retrieval -augmented generation (RAG) -based assistants and 
a compact agent reference for direct use by AI agents. The knowledge base was loaded 
into NotebookLM to provide conversational PHITS support, while the agent reference 
was combined with PHITS -specific policies and execution rules to enable Codex and 
Claude Code to edit input files, execute calculations, inspect errors, analyze results, and assist with source -code modification and compilation. Five demonstration tasks covered 
input modification, repeated simulations, parameter optimization, program compilation, post-processing, and result interpretation. The results showed that AI agents could 
handle complex PHITS workflows when appropriate resources and rules were provided. Practical lessons included precise prompts, human verification, well -documented 
sample files, explicit execution policies, and command- line-accessible tools. These 
findings support bundling AI -ready resources with particle transport codes to enable the 
use of general -purpose AI tools without requiring dedicated code -specific applications.  

2 
 1. Introduction 
Monte Carlo particle transport codes are widely used in accelerator facility design, 
nuclear technology, medical physics, space radiation dosimetry, and many other fields. 
General -purpose codes such as FLUKA [ 1], Geant4  [2], MCNP [ 3], OpenMC [ 4], 
PHITS [ 5], and TRIPOLI -4 [6] are powerful tools for detailed simulations of the 
transport of neutrons, photons, electrons, protons, heavy ions, and other particles in 
complex geometries. However, appropriate use of these codes requires substantial 
knowledge and experience, including input syntax, geometry modeling, material 
definition, source specification, tally selection, statistical uncertainty evaluation, and output analysis. 
Recent advances in large language models (LLMs) have rapidly expanded the 
use of natural -language interfaces for program generation, file editing, error correction, 
and data analysis [ 7]. In addition, LLM -based autonomous agents , often referred  to as 
AI agent s, have attracted increasing attention as a means of combining reasoning, 
planning, tool use, and iterative task execution [ 8]. Retrieval -augmented generation 
(RAG) further enables LLMs to answer questions while referring to external documents rather than relying solely on their internal parameters [ 9]. These technologies may also 
be useful for assisting users of particle transport codes. 
In the field of radiation- transport Monte Carlo simulations, AutoFLUKA has 
recently been reported as an LLM -agent framework for automating FLUKA workflows, 
including input preparation, simulation execution, and post -processing [ 10]. It is an 
important precedent showing that AI agents can be applied to Monte Carlo simulation workflows , and its authors suggested that such frameworks could be extended to other 
simulation packages. For such extensions to be realized in practice, however, code developers need to prepare and distribute AI -ready code -side resources, such as 

3 
 knowledge bases, sample -file descriptions, reference policies, and execution rules, that 
allow general -purpose AI assistants and AI coding agents to use each code reliably. 
Motivated by this perspective, we present an implementation strategy for 
applying existing LLMs and AI agents to particle transport simulations through the 
development of AI -assisted PHITS workflows. To achieve this, the following tasks 
were carried out: (1) PHITS manuals, lecture materials , sample input files, and other 
instructive materials were reorganized into AI -ready  text resources;  (2) a conversational 
support environment was constructed by loading these text resources into NotebookLM, a RAG -based AI assistant; and (3) an autonomous PHITS execution and analysis 
workflow was developed using AI agents such as Codex and Claude Code by making the text resources and additional PHITS -specific operational rules available to them.  It 
should be noted that the purpose of this study is not to develop a new LLM algorithm, but to clarify the AI -ready  knowledge resources and execution policies that code 
developers should prepare so that particle transport codes can be used safely and 
efficiently by AI agents.  
2. Methods  
2.1. Overview of the PHITS package with AI -assisted workflow components  
Figure 1 shows an overview of the PHITS package with AI -assisted workflow 
components introduced in this study. The PHITS package contains several directories that support users in preparing input files , such as  lecture, manual , recommendation, 
sample , and utility . These directories provide lecture materials, the official PHITS 
manual in HTML format, recommended input files for typical  simulations, sample input 
files and user -defined programs , and supporting tools such as PHITS Interactive 
Geometry Viewer in 3D (PHIG- 3D) [ 11] and the RadioT herapy package based on 
PHITS (RT -PHITS)  [12,13]. The package also contains other directories for executables, 

4 
 source code, and cross -section libraries, although they are not shown in the figure for 
simplicity.  
This study focuses on the workbench  directory, an AI -oriented directory newly 
introduced in the forthcoming version of PHITS. The workbench  directory contains AI -
ready  text resources generated from the PHITS manual and other user -support materials. 
These resources are intended for use by AI assistants and AI agents when answering 
PHITS -related questions or searching for practical examples. The directory also 
provides practical instructions and tools for an AI -agent -driven workflow , including 
agent entry -point files, reference policies, task definitions, PHITS -input language -
support settings, and setup scripts. Through this workflow , AI agents operating in  an 
integrated development environment ( IDE) such as VS Code can edit PHITS input files, 
execute PHITS, analyze output files, and iteratively repeat these procedures until the user’s request is satisfied. The preparation and use of the workbench  directory are 
described in the following subsections. 
2.2. Preparation of AI -ready  PHITS knowledge resources  
To use particle transport codes with AI, it is not sufficient simply to provide AI 
systems with manuals in HTML or PDF format. This is because manuals primarily serve as comprehensive reference documents; practical knowledge resources with contextual information, such as examples, tutorials, and recommended workflows, are also required. Therefore, we developed a Python script to convert information related to practical PHITS usage contained in the lecture, manual , recommendation, sample , and 
utility  directories into text formats that can be easily searched and referenced by AI 
systems. The resulting resources were stored in the knowledge_base  and 
agent_reference  directories under  workbench/AI  for use by RAG -based AI assistants 
and AI agents, respectively. 

5 
 In constructing these resources , we adopted the following seven  measures to 
make it easier for AI systems to search, interpret, and use the information:   
1. Overview descriptions were prepared for each source directory to summarize the 
types of information contained in the converted text files and to help AI systems determine where to search according to the user’s calculation purpose.  
2. Keywords were assigned to individual input files and related materials so that AI systems could more easily identify relevant information, such as calculation purposes, particle types, geometries, tallies, and output quantities.  
3. The lecture materials in the lecture directory were linked with slide texts, narration 
scripts  where available, and exercise input files so that practical workflows could 
be referred to together with the corresponding examples.  
4. Information on sample input files and user -defined programs in the sample  
directory was organized together with the relevant instruction texts, where available, allowing AI systems to refer to advanced functions when necessary.  
5. Lecture materials, sample files , and manuals  for supporting tools in the utility  
directory were systematically converted into text -based resources so that AI 
systems could understand how these tools should be used and could refer to their usage procedures . 
6. Developer -curated cautions were summarized as crucial notices to explicitly 
describe practical points that AI systems may easily misunderstand, such as input syntax, geometry checking, tally selection, and execution procedures.  
7. For direct use by AI agents, the agent_reference  directory was organized into small, 
categorized Markdown files. Because AI agents can directly access the original project files when necessary, input files, output files, and other plain- text source 

6 
 files were generally referenced by path rather than duplicated in this directory. This 
structure enables targeted searches and reduces unnecessary token consumption.  
It should be noted that t hese measures were developed through iterative consultation 
with AI systems, which provided suggestions on how information should be structured to match their own search and interpretation processes . More detailed strategies for 
preparing these AI -ready resources are provided in Appendix I.  
2.3. C onversational  PHITS support using a RAG -based AI assistant  
The constructed knowledge base was first loaded into NotebookLM, a RAG -based AI 
assistant, to establish conversational PHITS support  in which users can ask questions in 
natural language. In this approach, users do not need to search manuals, lecture materials, and sample files individually. Instead, they can ask the AI assistant questions such as “Which recommended input file is appropriate for accelerator shielding design?”, “What does this keyword mean?”, or “What is this sample input intended to do?” This is particularly useful when users are not familiar with the PHITS directory 
structure.  
An advantage of the conversational AI -assistant  approach is its ease of 
introduction. Users can access this environment simply by visiting an existing public website configured with the above knowledge base, named AI Phi -chan  [14], and 
signing in with a Google A ccount, without preparing a special programming 
environment. NotebookLM also supports multilingual output, allowing users to ask questions and receive explanations in a supported language even when the underlying PHITS resources are available only in English or Japanese. This capability may substantially reduce language barriers for users who are not proficient in either language. This approach is therefore useful for beginner education, self -study after PHITS 
tutorials, preliminary consultation before preparing input files, and efficient searching 

7 
 of the PHITS manual. Because the knowledge -base files are distributed with PHITS, 
users can also construct similar dedicated PHITS support assistants on other RAG -based 
AI platforms by uploading these files together with additional reference materials, such 
as lecture materials in their preferred language, project -specific documents, or 
frequently used input files. This flexibility is particularly useful in organizations or regions  where access to Google services is restricted or unavailable, as NotebookLM 
may not be accessible to all users.  
However, this  conversational support  approach also has limitations. In 
environments such as NotebookLM , the main functions are limited to dialogue with 
users, document summarization, explanation of sample files, and suggestions for input -
preparation strategies. Such environments do not directly edit PHITS input files, execute PHITS, correct errors, analyze output files, or visualize calculation results. Therefore, moving from conversational support to automated PHITS calculations requires an agentic workflow that allows AI systems to directly edit files, run 
commands, and inspect outputs. 
2.4. AI -agent -driven PHITS execution and analysis workflow  
To develop such an automated PHITS workflow, we prepared the workbench 
structure so that AI agents could access the  text resources, operational rules, task 
definitions, and user -specific cautions. In this study, Codex and Claude Code were used 
as AI agents, and VS Code served as the IDE providing the workspace interface. To use 
all functions of this workflow, users must install PHITS, a compatible Fortran compiler, 
an IDE , and the command -line interface (CLI) for at least one supported AI agent. They 
must also obtain an account or API access for the selected AI service.  Depending on the 
service and usage plan, a paid subscription or usage -based fees may be required; such 
external AI services are not included in the PHITS package.  

8 
 After these prerequisites have been satisfied  on a Windows, macOS, or Linux 
computer, the user first executes the setup script in the workbench /IDE directory. This 
script performs the following procedures: (1) it checks the availability of required tools, 
such as Fortran compilers, and configures VS Code tasks for building and executing PHITS; (2) it installs language -support functions for PHITS input files, such as syntax 
highlighting, auto- completion, and hover information; (3) it copies the common agent 
instruction file workbench/AI/AGENTS.md to the root of the workspace, i.e., the phits  
directory, as AGENTS.md  for Codex and duplicates the same file as CLAUDE.md  for 
Claude Code, so that both AI agents can follow the same PHITS -specific reference 
policies, cautions, and working rules; and (4) it creates a template for user -specific 
crucial notices, named crucial_notice.txt , in the user directory.  
The common agent instruction file, AGENTS.md , was intentionally kept simple. 
This file instructs the AI agents to read workbench/AI/reference_policy.md  as the 
highest -priority document for PHITS -related tasks, to perform targeted searches of 
workbench/AI/ agent_reference  when PHITS -specific information is required, and to 
refer to user/crucial_notice.txt for practical cautions. The reference_policy.md  file 
provides more detailed procedures for AI-agent -based editing and execution of PHITS 
calculations.  For example, after modifying geometry, the agent is instructed to run 
PHITS in geometry -check mode and confirm that there are no overlapping or undefined 
regions before proceeding to an actual particle transport calculation.  
When an AI agent is asked to modify a PHITS input file, it first reads the target 
input file and identifies the relevant sections, keywords, particle names, material numbers, cell numbers, tally names, and other related items. If the user’s request is sufficiently clear, the agent modifies only the relevant parts with the minimum necessary changes. If the request is ambiguous, the agent performs targeted searches of 

9 
 workbench/AI/ agent_reference  and other relevant resources, identifies the necessary 
input sections and keywords, and determines how the input file should be modified. 
This procedure was designed to prevent the agent from unnecessarily rewriting unrelated parts of the input file and to help maintain the consistency of PHITS input files.  
The AI -agent -driven workflow  can also handle more advanced operations. 
When custom functions, such as user -defined tallies or source routines, are required, the 
agent prepares the corresponding source files, builds  PHITS with them, and runs the 
resulting executable.  When optimization of a calculation parameter is requested, the 
agent can iteratively modify the relevant conditions, run PHITS, compare the obtained results, and identify the optimal setting. In addition, when an AI agent makes a mistake during input preparation, the corresponding caution can be added to user/crucial_notice.txt , either manually by the user or by the agent under the user’s 
instruction. This allows the user to incrementally customize the workflow  based on their 
own experience.  
2.5. Demonstration tasks  
To evaluate the capability of the AI -agent -driven workflow , the following prompts were 
applied as demonstration tasks. 1. Please calculate the response function of a Si detector to cosmic- ray irradiation, 
referring to recommendation/CosmicRay . Set up the following geometry and 
irradiation conditions:(1) Place a 5 cm × 5 cm × 0.1 cm silicon detector inside an aluminum spherical shell, (2) The aluminum shell has an inner radius of 10 cm and an outer radius of 15 cm, (3) irradiate the shell isotropically from the outside with galactic cosmic- ray in DeepSpace mode, (4) Consider protons only as the source. 
Estimate the detector response rate in counts/sec, assuming that the threshold 

10 
 deposition energy in the detector is 1 MeV. The statistical uncertainty of the 
estimated response rate should be less than 10%.  
2. Please calculate the optimum polyethylene shielding thickness for a neutron source, referring to recommendation/h10multiplier . Set up a 10 MBq Cf -252 source at the 
center and surround it with a spherical polyethylene shield. Variance reduction is not required. Calculate the ambient dose equivalent rate, H
*(10), on the outer 
surface of the polyethylene sphere. Find the radius of the polyethylene sphere required to reduce H
*(10) to 5 μSv/h or less. Adjust the polyethylene thickness in 1 
cm increments. The statistical uncertainty of the calculated H*(10) value should be 
within 5%. 
3. Please create a more realistic snowman geometry based on lecture/exercise/snowman . Assign realistic colors using [mat name color] . Then, 
estimate the proton energy that maximizes the deposited energy in the target region of the snowman geometry, adjusting the energy in 1 MeV increments. 
4. Please evaluate the correlations among the responses of three detectors. Use sample /UserDefined/ usrtally/t- deposit -event  as a reference. Construct a geometry 
with three cylindrical liquid scintillators placed consecutively along the same axis; each detector should have a radius of 10 cm and a thickness of 3 cm. Irradiate the 
detectors with 100 MeV neutrons along the central axis. Set the number of histories 
to 1000. Output the deposition energy in each of the three detectors for every event 
and calculate the probability that only the middle detector has a deposition energy 
above the 1 MeV threshold. 
5. Please calculate the number of DNA damage events using KURBUC. Estimate the 
number of DNA damage events per source particle (damage/source) for 3 MeV 
protons incident on a water sphere with a radius of 50 nm. Use 

11 
 sample /UserDefined/ usrtally/DNAdamage  as a reference for the calculation method. 
Keep the number of histories unchanged from the sample input. 
These tasks are hereafter referred to as cosmic- ray response, shielding optimization, 
melting -snowman exercise, detector coincidence, and DNA damage estimation, or 
simply as Tasks 1 –5. Task 1 was designed to represent a typical PHITS simulation 
workflow, in which the geometry, source, and tally are appropriately defined, PHITS is 
executed, and the calculation results are obtained. Tasks 2 and 3 were optimization problems, in which PHITS simulations were performed repeatedly under different conditions to search for the optimal setting. Tasks 4 and 5 could not be handled by the standard tallies alone and required compilation of PHITS with user -defined tally 
functions. In particular, Task 5 was a highly advanced task that would conventionally have been feasible only for advanced users, because it required compilation with the specialized PHITS -KURBUC library [ 15] and compilation of a post -processing 
program to estimate DNA damage from track -structure analysis results obtained by 
PHITS  [16]. 
Each task was performed using both Codex (GPT -5.5) and Claude Code (Opus 
4.8). For each completed task, the number of tokens used, the agent processing time, and the execution time of PHITS and related programs, hereafter referred to as the 
program execution time, were recorded and compared. All calculations were performed 
only once on a standard desktop PC running Windows 11 with an Intel Core i7- 9700K 
CPU at 3.60 GHz and 32 GB of RAM. 
3. Results  and Discussion  
3.1. Generated PHITS knowledge base  
Table 1 summarizes the file size, word count, and character count of the text files 
included in the PHITS knowledge base stored in the workbench/AI/knowledge_base  

12 
 directory and uploaded to the NotebookLM page  named AI Phi -chan. In NotebookLM, 
the maximum length of a single source file is limited to 500,000 words ; therefore, the 
lecture materials were divided into two files. The Japanese manual contains fewer 
words than the English manual, mainly because Japanese generally conveys more information per word. The content covered in the Japanese and English manuals is almost identical. In addition to these text files, the PDF manuals for ANGEL, a graph-drawing program for PHITS, and DCHAIN -PHITS  [17], an inventory calculation code, 
are also included in the knowledge base. These PDF manuals will be converted into HTML format and then textualized in the future. It should be noted that these data were 
generated based on PHITS version 3.36; therefore, the file sizes, word counts, character counts, and related values will be updated whenever a new version of PHITS is released.  
3.2. Results of demonstration tasks  
Table 2 shows the number of tokens used, the agent processing time, and the 
program execution time for the demonstration tasks. The agent processing time tended 
to be longer for Claude Code than for Codex, whereas no clear trend was observed in the number of tokens used by the agents. This difference in processing time is likely attributable to differences in the models used in this study. The program execution times for Codex and Claude Code differed substantially in Tasks 2 and 3. This is because these tasks involved optimization, and the execution time therefore depended not only on the number of Monte Carlo histories used in each PHITS simulation but also on the number of repeated PHITS runs. 
Table 2 also summarizes the answers obtained using Codex and Claude Code for 
each task. The answers obtained by the two agents were generally consistent, except for the optimization problems, i.e., Tasks 2 and 3. We also confirmed the adequacy of the answers and simulation procedures, including post -analysis, by checking the summary 

13 
 files generated by the AI agents.  These results suggest that AI agents can properly 
handle relatively complex particle transport problems through the AI -agent -driven 
workflow developed in this study, including tasks that involve code compilation and 
post-analysis of simulation results, when they are provided with appropriate  AI-ready 
resources  and reference policies.  
The differences between the answers obtained by the two agents for the 
optimization problems can be attributed to the large degrees of freedom in these tasks. In particular, the detailed geometry to be constructed was not strictly specified in the melting -snowman exercise, making it reasonable that the two agents generated different 
geometries and obtained different optimized proton energies. As an example, Fig. 2 shows three -dimensional visualizations of the original geometry included in the 
snowman exercise and the geometries generated by Codex and Claude Code, together with the particle flux or absorbed- dose distributions obtained by irradiation at the 
optimal proton energy. The lower panels of the figure indicate that the deposited energy 
is maximized when the proton range corresponds to the distal edge of the target. Note that Codex used a snowball density of 1 g/cm
3, as in the original exercise, whereas 
Claude Code adopted a more realistic value of 0.3 g/cm3. For the shielding optimization 
task, the difference in the answers was caused by the fact that Claude Code evaluated 
the combined neutron and photon dose, whereas Codex considered only the neutron 
dose and therefore underestimated the total dose. This restriction was explicitly 
described in the summary file generated by Codex, demonstrating the importance of manually checking the summary files to confirm the assumptions and calculation choices made by AI agents, which may not always fully match the user’s intention. 
3.3. Lessons learned for code users  

14 
 The prompts used in the demonstration tasks were not written perfectly from the 
beginning. Rather, they were refined through several trials in which unclear instructions caused the AI agents to perform simulations that were different from what we had intended. Based on these unsuccessful attempts, the following information should be included when asking an AI agent to perform Monte Carlo particle transport simulations:  
 the purpose of the calculation;  
 the name of the sample directory or input file to be used as a reference;  
 the components to be modified and those to be kept fixed, such as the geometry, source, tally, and calculation parameters;  
 the evaluation and optimization criteria, including the physical quantities to be 
evaluated and the required level of statistical uncertainty. 
The third item is particularly important, because ambiguous instructions may cause AI agents to modify unintended parts of the input file. Users should therefore specify the 
conditions in as much detail as possible, for example whether only the size of a particular region should be modified, or whether only the source energy should be changed while keeping the spatial and angular distributions fixed. 
It is efficient to provide all these items in a single prompt, but they can also be 
provided step by step if the user is unsure. For example, if the user does not know which sample should be used as a reference, the user can first describe the purpose of the calculation and ask the AI agent to suggest an appropriate sample. In many cases, the agent can identify a suitable reference input file. For optimization tasks that require repeated PHITS runs, it is also useful to first ask the agent to generate an input file for a specific condition and have it checked by a human before starting the repeated calculations. This procedure can reduce the risk of wasting computational time by 

15 
 repeatedly executing incorrect simulations. As an illustrative example of such step -by-
step use, Appendix II presents a detailed interaction between a user and an AI agent in 
which a PHITS calculation for passive -beam  proton therapy using an ICRP adult 
reference voxel phantom  [18] was progressively developed, from the selection of an 
appropriate recommended input to beam optimization and organ- dose assessment.  
Even if a calculation is completed successfully, as previously mentioned, human 
review, confirmation of statistical uncertainties, and assessment of physical validity are essential. It is also important to save the PHITS execution logs, input files, and output files to ensure reproducibility. When an AI agent makes a mistake, the user should ask the agent to analyze the cause of the mistake and accumulate preventive cautions for future tasks. In the present workflow , this role is served by the user/crucial_notice.txt  
file. 
3.4. Lessons learned for code developers  
Even if a user provides an appropriate prompt, the expected result cannot be obtained if 
the AI agent does not correctly understand the syntax and execution procedures of the particle transport code. In fact, during the demonstration tasks, the agents made several mistakes before the final prompts and reference policies were refined. Examples included using # as a comment character  in the [surface ] and [cell]  section s, where it is 
not allowed, and proceeding to a particle transport calculation without checking a newly generated geometry and thereby causing lost -particle errors . Another example was  
running multiple PHITS input files in the same directory simultaneously, which resulted in file -conflict errors.  
Based on these experiences, we consider the following points important for 
developers who wish to make particle transport codes more AI -friendly.  

16 
  Manuals and lecture materials  should be provided not only in image -based or PDF -
layout formats but also as text -based resources. Although AI systems can handle 
figures and images, input syntax, commands, keywords, file paths, and error 
messages are more reliably interpreted when they are provided as explicit text.  
 A sufficient number of sample input files should be provided, together with descriptions of their purpose, assumptions, physics models, tallies, output files, modifiable parts , recommended usage, and cautions. For AI agents, high- quality 
samples serve as starting points for input generation. In particle transport simulations, it is generally safer to modify validated samples with minimal changes than to generate input files from scratch.  
 Execution policies for AI agents should be clearly defined in advance. Working directories, file names, collision avoidance during repeated or parallel execution, execution commands, error -checking procedures, statistical -uncertainty checks, and 
geometry checks should be specified to prevent unintended agent behavior and 
accidental damage to files. These policies should also include typical mistakes to be checked and usages to be avoided, such as code -specific syntax restrictions, 
geometry -check requirements, and file -conflict risks.  
 Execution paths controllable from a command -line interface or scripts should be 
provided in addition to graphical user interfaces. AI agents are well suited to file editing, command execution, log inspection, and script execution, but are less suited to manual GUI operations. Tools that can be executed from the command line, such as PHITS itself, ANGEL, DCHAIN -PHITS, and RT -PHITS , are 
therefore compatible with AI -agent workflows. For visualization tools such as 
PHIG -3D, a future task is to enable AI agents to control display settings through 

17 
 scripts or command- line options, so that image generation and geometry checking 
can be automated.  
3.5. L imitations  
The present demonstration has two main limitations. First, each demonstration task was 
performed only once for each AI agent. Because the outputs of AI agents are not fully deterministic, the same prompt and environment may not always produce exactly the 
same input files, execution procedures, or answers. Therefore, the present results should 
be regarded as illustrative demonstrations of AI -agent -based PHITS workflows rather 
than statistical evaluations of agent performance. In particular, task success rates, 
reproducibility across repeated trials, and statistical differences among AI models or agents were not evaluated in this study.  
Second, the applicability of the workflow established in this study was examined 
using specific AI agents and a specific IDE, namely, Codex and Claude Code running in VS Code. As a preliminary check of the generality of this workflow, we also tested 
additional AI -agent configurations, including Google Antigravity IDE, GitHub Copilot 
in VS Code , and Claude Code CLI without VS Code . These tests suggested that similar 
AI-agent -driven workflows may also be possible in other environments and that the 
dependence on the specific AI -agent platform may be limited. However, these 
observations were preliminary, and more systematic comparisons among different AI -
agent configurations and execution environments are needed.  
Conclusions  
In this study, we presented a code -side implementation strategy for applying existing AI 
assistants and AI agents to PHITS. Two complementary sets of AI -ready resources were 
introduced in the workbench directory: a bundled knowledge base for RAG -based AI 
assistants and a compact agent reference for direct use by AI agents. Together with 

18 
 agent instruction files, PHITS -specific reference policies, execution rules, task 
definitions, and setup tools, these resources enabled conversational PHITS support and 
AI-agent -driven execution and analysis workflows. A key feature of this approach is 
that the resources and operational rules required for AI -assisted workflows are bundled 
with the PHITS package, rather than implemented in a dedicated PHITS -specific AI 
application. This allows existing general -purpose AI tools to support usage -related 
questions, input -file preparation, simulation execution, result verification, and analysis. 
Although this study demonstrated a code -side implementation strategy for 
applying existing AI tools to PHITS, further work is needed to generalize and optimize this approach for broader particle transport simulation workflows. As part of this effort, future studies should establish design principles and evaluation metrics for AI-ready  
knowledge bases, agent references , sample files, and execution policies that enable AI 
agents to perform particle transport simulations with fewer errors. Visualization and 
other supporting tools should also be made accessible  to AI agents through scripts or 
command -line interfaces, thereby enabling automated geometry checking, result 
visualization, and related operations. The practical effectiveness of such workflows should then be evaluated quantitatively using different AI agents, in terms of task success rates, error -correction capability, token consumption, human working time, and 
applicability to different simulation fields.  
If these future studies establish effective AI -assisted workflows for many 
particle transport codes, the practical boundaries between codes may gradually become less important for users. In such a future, users may be able to execute multiple particle transport codes, compare their results, and analyze the differences simply by submitting a single prompt to a single AI agent. 

19 
 Declaration of generative AI use 
During the preparation of this manuscript, the authors used ChatGPT (OpenAI) to assist 
with English editing, wording suggestions, and improvement of clarity. The authors 
reviewed and edited all AI -assisted outputs and take full responsibility for the content of 
the manuscript.  
  

20 
  
Figure 1. Overview of  the PHITS package with AI -assisted workflow components 
introduced in this study. 
 
 
Figure 2. PHIG -3D visualizations of the original geometry included in the snowman 
exercise and the geometries generated by Codex and Claude Code  (upper panels) , 
together with the particle flux or absorbed -dose distributions calculated at the optimized 
proton energy for each geometry (lower panels) . 
  


21 
 Table 1  File size, word count, and character count of the text files included in the 
PHITS knowledge base . 
File name  Description  File size  #Words  #Characters  
crucial_notice_forAI.txt  Crucial notice for AI  3.48 kB  571 3.56k  
lecture_forAI_part01.txt  Textualized lecture materials (Part I)  3.12 MB  431k  3.02M  
lecture_forAI_part02.txt  Textualized lecture materials (Part II)  1.15 MB  171k  1.15M  
manualE_forAI.txt  Textualized PHITS manual in English  932 kB  148k  955k  
manualJ_forAI.txt  Textualized PHITS manual in Japanese 1.10 MB  53.2k  797k  
sample_forAI.txt  Sample and recommendation- setting input files  4.32 MB  431k  4.37M  
utility.txt Instructions  for using supporting tools  2.88 MB  277k  2.90M  
 Total 13.5 MB  1.51M  13.2M  
 
Table 2 . Number of tokens used, the agent processing time, the program execution time , 
and the answers obtained using Codex and Claude Code  for each  demonstration task. 
  Token s used (M)  Agent processing 
time (sec)  Program execution 
time (sec)  Answer  
Task ID  Task name  Codex  Claude  Codex  Claude  Codex  Claude  Codex  Claude  (unit)  
1 cosmic -ray response  1.52  1.67  177  400  210  234  23.2  24.3  counts/sec  
2 shielding optimization  3.88  1.89  212  403  2036  1254  26 29 cm 
3 melting -snowman exercise  1.37  2.71  218  830  539  477  158 131 MeV  
4 detector coincidence  2.51  1.11  253  209  86  85  0.027  0.027  event s/source  
5 DNA damage estimation  3.75  4.94  380  912  242  282  0.60  0.60  damage/source  
 
References  
[1] Ahdida C, Bozzato D, Calzolari D, et al. New capabilities of the fluka multi -
purpose code. Front Phys -Lausanne. 2022;9. 
[2] Agostinelli S, Allison J, Amako K, et al. Geant4- a simulation toolkit. Nucl 
Instrum Meth A. 2003;506(3):250- 303. 
[3] Kulesza JA, Adams TR, Armstrong JC, et al. Mcnp® code version 6.3.1 theory 
& user manual. LA -UR-24-24602. USA: Los Alamos National Laboratory; 2024.  
[4] Romano PK, Horelik NE, Herman BR, et al. Openmc: A state -of-the-art monte 
carlo code for research and development. Ann Nucl Energy. 2015;82:90- 97. 
[5] Sato T, Iwamoto Y, Hashimoto S, et al. Recent improvements of the particle and heavy ion transport code system -  phits version 3.33. J Nucl Sci Technol. 
2024;61:127- 135. 
[6] Brun E, Damian F, Diop CM, et al. Tripoli -4®, cea, edf and areva reference 
monte carlo code. Ann Nucl Energy. 2015;82:151 -160. 
[7] Naveed H, Khan AU, Qiu S, et al. A comprehensive overview of large language models. ACM Transactions on Intelligent Systems and Technology. 2025;16(5):106, 1- 72. 

22 
 [8] Wang L, Ma C, Feng X, et al. A survey on large language model based autonomous agents. Front Comput Sci. 2024;18(6):186345. 
[9] Lewis P, Perez E, Piktus A, et al. Retrieval -augmented generation for 
knowledge -intensive nlp tasks. Adv Neur In. 2020;33. 
[10] Ndum Ndum Z, Tao J, Ford J, et al. Automating monte carlo simulations in nuclear engineering with domain knowledge -embedded large language model 
agents. Energy and AI. 2025;21:100555. 
[11] Ohnishi S. Gxsview: Geometry and cross section viewer for calculating radiation transport. Softwarex. 2021;14:100681.  
[12] Sato T, Furuta T, Liu Y, et al. Individual dosimetry system for targeted alpha therapy based on phits coupled with microdosimetric kinetic model. EJNMMI Phys. 2021;8:4.  
[13] Furuta T, Koba Y, Hashimoto S, et al. Development of the dicom -based monte 
carlo dose reconstruction system for a retrospective study on the secondary cancer risk in carbon ion radiotherapy. Phys Med Biol. 2022;67(14):145002. 
[14] AI phi-chan 2026 [cited 2026 Jun 21]. Available from: 
https://notebooklm.google.com/notebook/a15dee09- 1c7c -49be -96de -
d21d90813edb 
[15] Matsuya Y, Kai T, Sato T, et al. Verification of kurbuc -based ion track structure 
mode for proton and carbon ions in the phits code. Phys Med Biol. 2021;66(6):06NT02.  
[16] Matsuya Y, Kai T, Parisi A, et al. Application of a simple DNA damage model developed for electrons to proton irradiation. Phys Med Biol. 2022;67:215017. 
[17] Ratliff HN, Matsuda N, Abe S, et al. Modernization of the dchain- phits 
activation code with new features and updated data libraries. Nucl Instrum Meth B. 2020;484:29- 41. 
[18] International Commission on Radiological Protection. ICRP publication 110: Adult reference computational phantoms. Annals of the ICRP. 2009:39 (2). 
 
  

23 
 Appendix I. Strategies for Preparing the AI- Ready Knowledge Base 
and Agent Reference  
This appendix describes the strategy used in this study to prepare AI -ready reference 
resources from the PHITS package, which may also be applicable to other particle 
transport codes. The purpose was to reorganize the existing manuals, lecture materials, sample inputs, recommended calculations, utility programs, and developer -curated 
cautions into formats that can be efficiently searched and used by both r etrieval -
augmented generation (RAG) systems and AI agents. The original PHITS files were retained in their original locations; the AI -ready resources were generated as derived 
files by Python scripts. 
Two complementary outputs were prepared: a knowledge_base directory and an 
agent_reference directory. The knowledge_base directory was intended for RAG -based 
AI assistants. It contains multiple merged text bundles, each corresponding to a major source category, such as manuals, lectures, utilities, sample inputs, and recommended 
calculations. The bundles include file indexes, extracted document text, source paths, metadata, and, where appropriate, the contents of related input and supplemental files. Large collections of lecture materials were divided into multiple files to keep individual bundles manageable.  
The conversion scripts first scanned the relevant PHITS directories and prepared indexes of the available materials. Each entry records the source path and, where possible, a short description of its purpose. Text was extracted from DOCX, PPTX, and PDF files. For PPTX files, both slide text and speaker notes were collected when available. The manual sources were converted while retaining their section hierarchy, including linked or included files. These procedures made information originally 

24 
 distributed across different document formats accessible through text search. 
Additional metadata were prepared to improve retrieval. For individual input files, the 
intended purpose was inferred from title sections, explanatory comments, folder names, or file names. Keywords were generated from informative section names while excluding generic structural sections. The resulting metadata help AI systems identify examples related to specific calculation purposes, particle types, geometries, tallies, or output quantities. Related instructional documents and small supplemental text or program files were included when they were useful for interpreting an example. In contrast, binary files, images, calculation outputs, and excessively large auxiliary files were excluded or represented only by their paths. 
The agent_reference directory was prepared for direct use by AI agents operating in the 
PHITS workspace. It contains a compact top -level index and smaller Markdown files 
divided by category, source folder, or utility. These Markdown files provide extracted 
documentation, searchable headings, keywords, purpose descriptions, and paths to the original materials. However, input files, output files, source programs, and other plain-text files were generally not duplicated in this directory. Because an AI agent can directly read or copy the original files when necessary, duplication would unnecessarily increase token consumption. This lightweight structure supports targeted searches and 
selective reading during AI -agent -driven PHITS workflows.  
  

25 
 Appendix II. Illustrative User –AI Agent Interaction for PHITS -Based 
Proton Therapy Dose Assessment  
This appendix illustrates a step -by-step interaction between a user and an AI 
agent for developing a PHITS -based dose assessment of passive -beam  proton therapy. 
Codex (GPT -5.5) was used as the sole AI agent in this example. The elapsed times 
shown in parentheses indicate the actual wall -clock time required for each response, 
including both the time spent by Codex and the time required to execute PHITS and 
related programs.  
It should be noted that the PHITS package already contains sample input files 
for a ridge filter and the  ICRP adult reference voxel phantom s. Therefore, Codex did 
not construct these components entirely from scratch, but instead adapted and combined existing PHITS resources according to the user’s instructions. Figure A 2.1 shows the 
final geometry and particle tracks for 100 source particles rendered using PHIG -3D. 
Because PHIG -3D cannot yet be controlled directly by the AI agent, this visualization 
was generated manually by the authors after completion of the agent -assisted workflow.  
 
 
Figure A 2.1. Final geometry and particle tracks for 100 source particles rendered using 
PHIG -3D. 


26 
 User- AI Agent Interaction Log  
User:  
I would like to perform a calculation related to the medical physics of proton therapy. 
Please tell me the appropriate recommendation settings.  
 
AI Agent  (47 s):  
I checked the local PHITS references and recommended settings, and identified the 
particle therapy recommendation input as an appropriate starting point for proton 
therapy depth- dose calculations.  
 User:  
Copy the recommendation input file to user/protontherapy/water_phantom.inp, and calculate the depth -dose distribution for 200 MeV protons incident on a cylindrical 
water phantom. Set the cylinder radius to 10 cm and the depth to 30 cm.  
AI Agent  (4 m 14 s):  
I copied the recommended input file to user/protontherapy/water_phantom.inp and modified it for 200 MeV protons incident on a cylindrical water phantom. The phantom 
was set to a radius of 10 cm and a depth of 30 cm, with a tally configured to calculate the depth- dose distribution. 
 User:  
Set the beam radius to 2 cm, and adjust the tally region to match this range. 
 
AI Agent  (2 m 41 s):  
I changed the beam radius to 2 cm and adjusted the tally region to match the beam range.  
 
User:  
Place a ridge filter 10 cm upstream of the irradiation surface, and adjust the ridge filter geometry so that the SOBP width becomes 5 cm. Adjust the number of histories so that 
the statistical uncertainty in the SOBP region is 2% or less.  The absorbed dose variation 
within the SOBP region should be kept within 5%.  

27 
 AI Agent  (29 m 53 s):  
I placed a ridge filter 10 cm upstream of the irradiation surface and adjusted the ridge -
filter geometry so that the SOBP width became approximately 5 cm. I also increased the 
number of histories so that the statistical uncertainty in the SOBP region was 2% or less, 
and checked that the absorbed dose variation within the SOBP region was within 5%. 
 
User:  
Move the ridge filter to a position 1 m upstream of the irradiation surface, and place a PMMA range shifter with a radius of 10 cm and a thickness of 1 cm halfway between 
the ridge filter and the irradiation surface.  Then, check whether the SOBP satisfies the 
above conditions. If it does not, adjust the ridge filter geometry again.  
 
AI Agent  (24 m 30 s):  
I moved the ridge filter to 1 m upstream of the irradiation surface and placed a PMMA 
range shifter with a radius of 10 cm and a thickness of 1 cm halfway between the ridge 
filter and the irradiation surface. I then checked whether the SOBP satisfied the 
previous conditions and adjusted the ridge -filter geometry as needed.  
 
 
User:  
Without changing the source conditions at all, replace the water phantom with the ICRP adult male voxel phantom. Assuming stomach cancer, adjust the position and 
orientation of the phantom so that the proton SOBP beam is delivered from the anterior 
side of the body toward the stomach.  
 
AI Agent  (33 m 11 s):  
Done. I replaced the water phantom in C:/phits/user/protontherapy/water_phantom.inp with the ICRP adult male voxel phantom and copied the text lattice to 
C:/phits/user/protontherapy/male -lattice.inp.  
I preserved the source conditions unchanged: 200 MeV proton, 2 cm radius, z0/z1 = -120 cm, dir = 1.  
 The phantom was transformed so that the fixed +z beam enters from the anterior side 

28 
 and points toward the stomach. I added stomach and organ dose tallies.  
Verification: 
- The icntl=8 geometry check completed, with no *_geo.out files produced.  
- The full PHITS run completed: PHITS simulation with icntl = 0 is properly finished.  
- The stomach dose uncertainty was well below 2%:  
  - Stomach wall region 72: 9.9074E+00, r.err 0.0028 
  - Stomach contents region 73: 1.9990E+01, r.err 0.0028 
One caveat: the completed run reported recoverable geometry warnings and one lost 
particle in the ridge -filter region, not in the voxel phantom. PHITS still finished 
properly with Number of unrecovered errors = 0.  
 
User:  
Adjust only the thickness of the range shifter so that the dose to the stomach contents is 
maximized.  
 AI Agent  (3 h 31 m 16 s):  
Done. I adjusted only the PMMA range -shifter thickness in 
C:/phits/user/protontherapy/water_phantom.inp.  
Final setting: - Range shifter thickness: 9.25 cm  
- Surfaces: z = -54.625 cm to z = - 45.375 cm  
- Center remains at z = -50 cm  
I swept the thickness and found that the stomach- contents dose peaked around 9.0- 9.25 
cm; 9.25 cm was the highest value in the refined sweep. The production run with 100,000 histories finished properly.  
Final stomach contents dose:  
- Region 73: 2.4144E+01 MeV/source  
- Relative error: 0.0036  
For comparison, the previous 1 cm range shifter gave 1.9990E+01 MeV/source, so this 
increased the stomach -contents dose by about 21%. 
The source conditions were unchanged. There was still the pre -existing recoverable 
ridge -filter geometry warning with one lost particle and zero unrecovered errors.  
 

29 
 User:  
Assuming a beam current of 1 nA, calculate the irradiation time required to deliver an 
absorbed dose of 10 Gy to the stomach contents under this irradiation condition.  
Also, report the absorbed doses to the surrounding organs for this irradiation time. 
 
AI Agent  (1 m 55 s):  
Using the final 9.25 cm PMMA range -shifter condition:  
- Stomach contents energy deposition: 24.144 MeV/source  
- Stomach contents volume: 240.358 cm3 
- Density: 1.04 g/cm3 
- Mass: 0.24997 kg 
- Beam current: 1 nA = 6.2415E+09 protons/s  
Dose rate to stomach contents:  
0.09659 Gy/s  
Therefore, the irradiation time required to deliver 10 Gy to the stomach contents is:  
103.5 s = 1.73 min 
Approximate absorbed doses for the same irradiation time:  
Organ / tissue          Absorbed dose  
Stomach contents        10.0 Gy 
Stomach wall            7.25 Gy 
Pancreas                1.85 Gy 
Small intestine         0.727 Gy  
Liver                   0.562 Gy Spleen                  0.301 Gy Large intestine         0.217 Gy 
Oesophagus              0.210 Gy 
Adipose tissue          0.169 Gy 
Kidneys                 0.0118 Gy 
Urinary bladder         0.00480 Gy 
These are mass -averaged organ/material doses scaled from male- organ -dose.out. The 
target timing used the dedicated stomach -dose.out tally for region 73.   
 User:  

30 
 For this condition, I would like to visualize the particle tracks for 100 source particles 
using PHIG -3D. Please add [t -4dtrack] and run PHITS with a short history number. 
 
AI Agent  (6 m 4 s):  
Done. I created and ran the PHIG -3D track case here:  
C:/phits/user/protontherapy/phig3d_tracks/track_100.inp 
Changes in that visualization input:  
- maxcas = 100  
- maxbch = 1  
- Added [ T -4Dtrack ]  
- Output file: 4Dtrack.out  
- HistoryMax = 100  
- verbose = 0  
PHITS completed properly and produced:  
C:/phits/user/protontherapy/phig3d_tracks/4Dtrack.out  
Run check:  
- PHITS simulation with icntl = 0 is properly finished  
- Lost particles: 0  
- Unrecovered geometry errors: 0  
- No *_geo.out files were produced in the run folder.  
I also launched PHIG -3D with track_100.inp. In PHIG -3D, use the Particle tracks tab 
and open 4Dtrack.out to display the 100- source -particle trajectories.  
 