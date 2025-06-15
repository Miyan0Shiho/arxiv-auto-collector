# Repeton: Structured Bug Repair with ReAct-Guided Patch-and-Test Cycles

**Authors**: Nguyen Phu Vinh, Anh Chung Hoang, Chris Ngo, Truong-Son Hy

**Published**: 2025-06-09 19:36:40

**PDF URL**: [http://arxiv.org/pdf/2506.08173v1](http://arxiv.org/pdf/2506.08173v1)

## Abstract
Large Language Models (LLMs) have shown strong capabilities in code
generation and comprehension, yet their application to complex software
engineering tasks often suffers from low precision and limited
interpretability. We present Repeton, a fully open-source framework that
leverages LLMs for precise and automated code manipulation in real-world Git
repositories. Rather than generating holistic fixes, Repeton operates through a
structured patch-and-test pipeline: it iteratively diagnoses issues, proposes
code changes, and validates each patch through automated testing. This stepwise
process is guided by lightweight heuristics and development tools, avoiding
reliance on embedding-based retrieval systems. Evaluated on the SWE-bench Lite
benchmark, our method shows good performance compared to RAG-based methods in
both patch validity and interpretability. By decomposing software engineering
tasks into modular, verifiable stages, Repeton provides a practical path toward
scalable and transparent autonomous debugging.

## Full Text


<!-- PDF content starts -->

arXiv:2506.08173v1  [cs.SE]  9 Jun 2025Repeton: Structured Bug Repair with ReAct-Guided
Patch-and-Test Cycles
Nguyen Phu Vinh
Uppsala University
SwedenAnh Chung Hoang
Hanoi University of Science & Technology
Vietnam
Chris Ngo
Knovel Engineering Lab
SingaporeTruong-Son Hy
The University of Alabama at Birmingham
USA
Abstract
Large Language Models (LLMs) have shown strong capabilities in
code generation and comprehension, yet their application to com-
plex software engineering tasks often suffers from low precision and
limited interpretability. We present Repeton1, a fully open-source
framework that leverages LLMs for precise and automated code
manipulation in real-world Git repositories. Rather than generating
holistic fixes, Repeton operates through a structured patch-and-
test pipeline: it iteratively diagnoses issues, proposes code changes,
and validates each patch through automated testing. This stepwise
process is guided by lightweight heuristics and development tools,
avoiding reliance on embedding-based retrieval systems. Evaluated
on the SWE-bench Lite benchmark, our method shows good per-
formance compared to RAG-based methods in both patch validity
and interpretability. By decomposing software engineering tasks
into modular, verifiable stages, Repeton provides a practical path
toward scalable and transparent autonomous debugging.
CCS Concepts
•Software and its engineering →Software creation and man-
agement ; Software organization and properties; •Computing
methodologies →Artificial intelligence ; Artificial intelligence.
Keywords
Automated Software Engineering, LLM Agents, Code Generation,
Bug Fixing, Program Repair, Git Automation, Iterative Refinement,
Tool-Augmented LLMs
ACM Reference Format:
Nguyen Phu Vinh, Anh Chung Hoang, Chris Ngo, and Truong-Son Hy.
2025. Repeton: Structured Bug Repair with ReAct-Guided Patch-and-Test
Cycles. In .ACM, New York, NY, USA, 5 pages. https://doi.org/XXXXXXX.
XXXXXXX
1https://github.com/phuvinhnguyen/Repeton
Permission to make digital or hard copies of all or part of this work for personal or
classroom use is granted without fee provided that copies are not made or distributed
for profit or commercial advantage and that copies bear this notice and the full citation
on the first page. Copyrights for components of this work owned by others than the
author(s) must be honored. Abstracting with credit is permitted. To copy otherwise, or
republish, to post on servers or to redistribute to lists, requires prior specific permission
and/or a fee. Request permissions from permissions@acm.org.
Conference’17, Washington, DC, USA
©2025 Copyright held by the owner/author(s). Publication rights licensed to ACM.
ACM ISBN 978-x-xxxx-xxxx-x/YYYY/MM
https://doi.org/XXXXXXX.XXXXXXX1 Introduction
Software debugging remains one of the most labor-intensive and
cognitively demanding tasks in software development. As code-
bases grow in size and complexity, diagnosing and fixing bugs
increasingly requires deep domain knowledge, iterative testing,
and contextual reasoning. Recent advances in large language mod-
els (LLMs) have created new opportunities to automate parts of
this process. LLMs have demonstrated strong performance in code
generation, understanding, and repair, improving agents’ ability in
automating debugging workflows. However, many pipelines and
frameworks in software debugging are closed-source or rely on
closed-source LLMs, which restricts the development of this field
and causes problems such as the inability to deploy the pipeline
locally.
In this work, we introduce a fully open-source debugging agent
designed to overcome these limitations through a modular, embedding-
free pipeline that uses completely open-source LLMs solely for
structured reasoning. Instead of relying on closed-source LLM, our
system guides the agent through a step-by-step process inspired by
how developers debug: understanding and summarizing the prob-
lem, reproducing errors, narrowing the search space with keyword
heuristics, identifying suspect files and functions, and iteratively
generating and testing patches. This pipeline, illustrated in Figure 1,
organizes the agent’s reasoning in a transparent and interpretable
way, combining static inspection with runtime feedback.
Our approach focuses on practicality and efficiency. It avoids
neural embeddings and external retrieval systems, such as some
existing open-weight methods, instead using symbolic techniques
such as keyword extraction, structured file search, and test-driven
validation. This reduces the complexity of the framework and also
reduces the time to embed the entire codebase while still being
flexible enough to handle diverse software projects. After that, to
evaluate our system, we use SWE-bench Lite, a benchmark of real-
world GitHub issues that require multi-file reasoning and test-based
validation, to assess and compare our method with some existing
open-weight solutions. Our results show that the proposed agent
performs well compared to solely RAG-based methods despite be-
ing unable to surpass SweFixer. Lastly, we analyze failure cases
of our framework, which can provide useful information for fur-
ther research in this field. This work provides a reproducible and
transparent foundation for building autonomous debugging agents.
All components, including model weights, pipeline code, and eval-
uation scripts, are open-sourced to support further research and
real-world adoption.

Conference’17, July 2017, Washington, DC, USA Nguyen Phu Vinh, Anh Chung Hoang, Chris Ngo, and Truong-Son Hy
Problem
Input
Problem
SummaryPatch
Validation /
Test Re-
executionSuccessfully reproduce bug
No bugInitial Bug
ReproductionIterative Code
Search and
Repair Process
(ICSR)
Yes
NoBug 
resolved?
Process Complete
(Bug fixed)YesContinue 
Iteration?
Process Complete
(No action needed) Process Complete
(Bug not fixed)No
Figure 1: Iterative Repair and Validation workflow. The flow combines code search and patch generation with error reproduction
to ensure each patch resolves the bug without introducing new errors.
2 Related Works
Rapid advancement of large language models (LLMs) has led to
the emergence of coding agents, systems that go beyond simple
code generation to engage in autonomous software development
tasks. Models like Codex [ 3] and CodeT5+ [ 10] demonstrate strong
performance on code completion and synthesis benchmarks, while
frameworks such as GPT-Engineer [ 9] and Auto-GPT [ 13] showcase
the potential of chaining LLM actions to build software from natural
language specifications. However, these systems typically target
isolated or synthetic tasks and struggle to handle the complexity
of real-world software projects that require multistep reasoning,
context tracking, tool usage, and iterative testing.
This gap has raised interest in building autonomous agents that
can perform end-to-end bug fixing across entire codebases. Rather
than just generating snippets, these agents can identify relevant
files, interpret issues, locate faults, and fix the problem. In response
to this broader challenge, previous work explores a spectrum of
methods and tools. Classical approaches in automated program
repair (APR) separate fault localization from patch generation, using
techniques such as spectrum-based analysis, mutation testing, and
symbolic synthesis (e.g., GenProg [ 4], Angelix [ 8]). Learning-based
methods treat repair as translation from buggy code to fixed code,
but often fail to generalize beyond training distributions.
More recent systems adopt agentic architectures to coordinate
these steps. SWE-agent [ 14] and FixAgent [ 6] organize specialized
agents based on LLM (e.g., bug explainer, fault locator, patch pro-
poser) that collaboratively solve debugging tasks. RepairAgent [ 2]
and AutoCodeRover [ 16] treat the LLM as a high-level planner that
invokes tools such as search, compilation, and testing. VulDebug-
ger [7] combines dynamic run-time information with LLMs to guide
security patching. These systems show that agentic design enablesmore robust debugging, especially when combined with static and
dynamic analysis.
To rigorously evaluate such agents, benchmarks such as the
SWE benchmark [ 5] have emerged as critical infrastructure. De-
rived from real GitHub issues and pull requests, SWE-bench tests
agents on realistic bug-fixing tasks involving complex, multi-file
projects and test-driven validation. It has become the standard
benchmark for assessing autonomous repair agents, with a curated
subset (SWE-bench Lite) introduced for efficient experimentation.
Performance in the SWE benchmark reveals the limitations of mand
LLMs, with top models solving only a fraction of cases, highlighting
the challenge of scaling debugging to real-world codebases.
Given these challenges, open-source software agents have be-
come increasingly important. SWE-Fixer [ 12], built on InternLM,
demonstrates competitive performance using structured prompt-
ing, memory modules, and tool-augmented reasoning without rely-
ing on closed APIs. Moatless Tools [ 1] emphasizes simplicity and
reproducibility, offering lightweight, interpretable primitives for
debugging without embedding-heavy infrastructure. Our approach
shares this open, modular philosophy that extends it with run-
time feedback loops and symbolic heuristics to minimize inference
overhead. The SWE-agent project [ 14] further exemplifies open
innovation, allowing LLMs to autonomously fix GitHub issues with
pluggable model backends, achieving strong results with models
like LLaMA-32B. In contrast, proprietary systems like Cognition
Labs’ Devin [ 11] remain closed-source, despite their reported per-
formance (13.9% on SWE-bench). As such, open-source debugging
agents not only promote transparency and reproducibility but are
essential for democratizing progress in AI-assisted software engi-
neering. Our work builds on this ethos, advancing fully open agents
for practical, scalable, and interpretable automated debugging.

Repeton: Structured Bug Repair with ReAct-Guided Patch-and-Test Cycles Conference’17, July 2017, Washington, DC, USA
3 Methods
This section introduces Repeton , a framework powered by the
Large Language Model (LLM) designed for the rectification of au-
tonomous and precise codes.
3.1 Iterative Repair and Validation (IRV)
The Iterative Repair and Validation (IRV) process represents the
core patch-and-test strategy of Repeton , orchestrated through the
Testing andPatching modules. The process begins by summariz-
ing the reported issue using a large language model (LLM), which
distills the original problem description into a concise summary
denoted 𝑃sum. This summary serves as a guiding context for both
patch generation and testing.
TheTesting module uses a ReAct-style [ 15] reasoning flow to
iteratively build a test program that can reproduce the reported bug.
Using both the original problem description and its summary, it
generates a verifiable test case. Although the module also provides
feedback on the current patch and development state later on, its
main goal at this stage is to produce a consistently failing test that
confirms the presence of the bug. Once the test is generated, it is
executed, and the results are analyzed. If the test passes, indicating
that the bug is resolved and the patch does not negatively impact
the project, the current patch is finalized, and the process stops. If
the test fails, the system evaluates whether the failure is due to an
invalid test or an unresolved bug. Based on this, it either refines
the test or produces a diagnostic report with logs and suggestions
for further debugging. Throughout the process, the system also
considers the current patch state to assess how each change affects
the bug and the overall functionality, guiding whether to continue
with the current patching method or to find a new solution.
To support automated repair, the Patching module implements
theIterative Code Search and Repair (ICSR) process. This mul-
tistep framework searches for candidate fixes and applies them
iteratively. Crucially, ICSR supports rollback capabilities, allowing
the system to revisit earlier stages and explore alternative strategies
when the current direction is found to be ineffective or suboptimal.
3.2 Iterative Code Search and Repair Process
(ICSR)
The patching module is designed to methodically identify and apply
minimal modifications necessary to resolve a reported issue. This
process is organized into four sequential stages: identifying relevant
files, summarizing their structure, locating precise code regions, and
generating a patch. Initially, the agent extracts a list of keywords
based on both the problem description and the project’s directory
structure. These keywords are then used to query the project tree for
potentially relevant files. The resulting matches are displayed in a
tree-like format as shown in Figure 2, allowing the agent to visualize
the layout of the codebase. Furthermore, the agent can reassess
its choices and backtrack to refine the keyword list, leveraging
feedback from previous attempts to improve the relevance of the
search.
Upon identifying the appropriate files, the agent proceeds to
summarize the structure of each file by listing its classes, functions,
and their respective line spans. This structural overview helps the
agent reason about file content without parsing the full code in onepass. Using this high-level map, the agent then inspects the inter-
nal content of specific classes and functions, enabling the precise
localization of potential bug sites. If the existing implementation
appears correct or irrelevant to the issue, the agent can shift its
focus to alternative files or code regions using the system-provided
tools. Importantly, edits are made conservatively: only one code
region is modified per iteration, and viewing a different file resets
all prior modifications to ensure patches remain minimal.
The agent follows a strictly linear process, advancing to the next
stage only after completing the current one. However, many stages
include a rollback mechanism to handle flawed decisions, such as
incorrect keyword selection or file relevance. In such cases, the
agent explains the mismatch between expectations and outcomes
and justifies revisiting an earlier stage with improved criteria. This
adaptive correction improves the robustness of the patching strat-
egy without disrupting the structured workflow. Furthermore, the
entire process is guided by React-style prompting, where reasoning
and actions are tightly coupled at each step. A persistent problem
summary is included in all prompts to maintain focus, and a trunca-
tion mechanism retains only the most recent exchanges to prevent
history overload. When rolling back, the conversation history is
trimmed to that point, removing later interactions. Reflective feed-
back is essential for explaining failures and planning improvements,
avoiding repeated mistakes, and making informed decisions.
4 Experiments
4.1 Experimental Result
Method Score (%)
Moatless Tools + DeepSeekV3 0.00
Swe-Fixer 24.67
Repeton + DeepSeekR1 (Ours) 11.67
Table 1: Performance comparison on the Swebench-lite
benchmark. The score reflects the percentage of bugs suc-
cessfully fixed.
We evaluated our pipeline on the Swebench-lite benchmark
and compared it with other open-weight coding agents. Swebench-
lite includes 300 GitHub repositories with real-world software bugs
and is widely used to assess automated software engineering sys-
tems. The target is to generate code patches that fix the reported
issues. We compare our method against two open-weight baselines:
Moatless Tools + DeepSeekV3 andSwe-Fixer .Swe-Fixer is
trained and fine-tuned specifically for Swebench-style bug fixing
and uses an additional text retrieval module to enhance patch qual-
ity, though this adds complexity and reduces efficiency. Our method,
in contrast, uses only general-purpose, open-weight models such
as DeepSeek, without task-specific tuning or retrieval components.
As shown in Table 1, our system achieves a success rate of 11.67%,
outperforming Moatless Tools + DeepSeekV3 , which failed to
resolve any of the benchmark tasks. Although Swe-Fixer achieves
a higher overall success rate, our pipeline demonstrates competitive
performance, particularly considering its minimal architectural
assumptions and the lack of specialized fine-tuning.

Conference’17, July 2017, Washington, DC, USA Nguyen Phu Vinh, Anh Chung Hoang, Chris Ngo, and Truong-Son Hy
Stage 1:
Keyword
Extraction &
File
IdentificationNeed refinement
YesRelevant 
File(s)
Identified?Stage 2: File
Structure
Summarization
No, Inspect Different File
Stage 4: Patch
Generation &
ApplicationStage 3: Precise
Code Region
Localization 
No, Inspect Dif ferent Region
in Same File
Patch Applied & 
Agent Deems ICSR 
Iteration Complete?Output: Modified
Code State
To IVR Loop for
ValidationYesInput: Problem
Summary &
Feedback from
IVR Loop
Example relevant files list
+ tmp
  + tmpi6s8zh1d
    + mwaskom_seaborn
      + mwaskom__seaborn-3190
        + seaborn
          - __init__.py  
          + _core
            - __init__.py  
            - properties.py  
            - rules.py  
            - scales.py  (contain all
keywords)
            - typing.py  
          + external
            - __init__.py  
        + tests
          + _core
            - test_scales.py Example file summarization
File: /mwaskom_seaborn/mwaskom__seaborn-3190/seaborn/_core/scales.py
Classes:
- InternalScale (line 88)
    Methods:
        set_default_locators_and_formatters (line 89)
- Identity (line 134)
- Scale (line 54)
    Methods:
        __post_init__ (line 65)
        tick (line 71)
        label (line 74)
        _get_locators (line 77)
        _get_formatter (line 80)
        _get_scale (line 83)
- CatScale (line 182)
    Methods:
        set_default_locators_and_formatters (line 185)
Figure 2: Iterative Code Search and Repair workflow, which includes many steps from locating to fixing errors, and example
display of file summarization and relevant files list in the work.
5 Failure Case Analysis
Resolved Unresolved Empty Patch Total
# Instances 35 113 152 300
Table 2: Summary of model outcomes. Many failures are due
to the agent not generating any patch content.
We analyze the failure cases of our method using DeepSeek-R1,
as summarized in Table 2. One major issue is that the model some-
times does not generate any patch. This often happens when the
agent produces an overly long reasoning sequence, which causes it
to run out of context and stop before producing a final answer. This
highlights a risk when using general-purpose reasoning models
such as DeepSeek-R1 in this type of task.
Another common failure is the agent’s inability to identify the
correct files. This results in repeated unsuccessful searches without
any code modifications, revealing a weakness in the pipeline’s
ability to guide the agent effectively. In some cases, the agent alsofails to reproduce the reported error. Since the pipeline relies on
successful error reproduction to validate patches, this can cause
multiple valid patches to be rejected. Eventually, only the last patch
is accepted by default, even though earlier patches are often more
correct. This behavior can significantly reduce overall performance.
6 Conclusion
We present Repeton, a fully open-source framework for autonomous
software debugging. Unlike monolithic LLM-based systems, Repeton
structures the debugging process as a modular patch-and-test pipeline,
enabling precise, interpretable, and verifiable code edits in real-
world coding projects or repositories. By combining open-source
LLM reasoning with symbolic heuristics and developer tools, Repeton
efficiently navigates and modifies codebases. This research also ex-
amines cases where the pipeline failed to resolve issues, providing
a valuable resource for future development in this field. Finally,
Repeton offers a practical and extensible foundation for LLM-driven
software agents, prioritizing transparency and real-world applica-
bility.

Repeton: Structured Bug Repair with ReAct-Guided Patch-and-Test Cycles Conference’17, July 2017, Washington, DC, USA
References
[1]AorWall. 2024. Moatless Tools. GitHub repository. https://github.com/aorwall/
moatless-tools
[2]Islem Bouzenia, Premkumar Devanbu, and Michael Pradel. 2024. RepairA-
gent: An Autonomous, LLM-Based Agent for Program Repair. arXiv preprint
arXiv:2403.17134 (2024).
[3]Mark Chen, Jerry Tworek, Heewoo Jun, Qiming Yuan, Henrique P. Oliveira Pinto,
Jared Kaplan, Harri Edwards, Yuri Burda, Nicholas Joseph, Greg Brockman, Alex
Ray, Raul Puri, Gretchen Krueger, Michael Petrov, Heidy Khlaaf, Girish Sastry,
Pamela Mishkin, Brooke Chan, Scott Gray, Nick Ryder, Mikhail Pavlov, Alethea
Power, Lukasz Kaiser, Mohammad Bavarian, Clemens Winter, Philippe Tillet,
Felipe Petroski Such, Dave Cummings, Matthias Plappert, Fotios Chantzis, Eliza-
beth Barnes, Ariel Herbert-Voss, William Guss, Alex Nichol, Alex Paino, Nikolas
Tezak, Jie Tang, Igor Babuschkin, Suchir Balaji, Shantanu Jain, William Saun-
ders, Christopher Hesse, Andrew N. Carr, Jan Leike, Josh Achiam, Vedant Misra,
Evan Morikawa, Alec Radford, Matthew Knight, Miles Brundage, Mira Murati,
Katie Mayer, Peter Welinder, Bob McGrew, Dario Amodei, Sam McCandlish, Ilya
Sutskever, and Wojciech Zaremba. 2021. Evaluating Large Language Models
Trained on Code. arXiv preprint arXiv:2107.03374 (2021).
[4]Claire Le Goues, ThanhVu Nguyen, Stephanie Forrest, and Westley Weimer. 2012.
GenProg: A Generic Method for Automatic Software Repair. IEEE Transactions
on Software Engineering 38, 1 (2012), 54–72. doi:10.1109/TSE.2011.104
[5]Carlos E. Jimenez, John Yang, Alexander Wettig, Shunyu Yao, Kexin Pei, Ofir
Press, and Karthik Narasimhan. 2024. SWE-bench: Can Language Models Resolve
Real-World GitHub Issues? arXiv preprint arXiv:2310.06770 (2024).
[6]Cheryl Lee, Chunqiu S. Xia, Longji Yang, Jen tse Huang, Zhouruixing Zhu, Ling-
ming Zhang, and Michael R. Lyu. 2024. FixAgent: Hierarchical Multi-Agent
Framework for Unified Software Debugging. arXiv preprint arXiv:2404.17153
(2024).
[7]Zhengyao Liu, Yunlong Ma, Jingxuan Xu, Junchen Ai, Xiang Gao, Hailong Sun,
and Abhik Roychoudhury. 2025. Agent That Debugs: Dynamic State-GuidedVulnerability Repair. arXiv preprint arXiv:2504.07634 (2025).
[8]Sergey Mechtaev, Jooyong Yi, and Abhik Roychoudhury. 2016. Angelix: Scalable
Multiline Program Patch Synthesis via Symbolic Analysis. In Proceedings of the
38th International Conference on Software Engineering (ICSE) . ACM, 691–701.
[9]Anton Osika. 2023. GPT-Engineer: CLI Platform to Experiment with Code Gen-
eration. GitHub repository. https://github.com/AntonOsika/gpt-engineer
[10] Yue Wang, Hung Le, Akhilesh Deepak Gotmare, Nghi Q. D. Bui, Junnan Li, and
Steven C. H. Hoi. 2023. CodeT5+: Open Code Large Language Models for Code
Understanding and Generation. In Proceedings of the 2023 Conference on Empirical
Methods in Natural Language Processing (EMNLP) . Association for Computational
Linguistics, 1069–1088.
[11] Scott Wu. 2024. Introducing Devin, the First AI Software Engineer. Cognition.ai
blog. https://cognition.ai/blog/introducing-devin
[12] Chengxing Xie, Bowen Li, Chang Gao, He Du, Wai Lam, Difan Zou, and Kai
Chen. 2025. SWE-Fixer: Training Open-Source LLMs for Effective and Efficient
GitHub Issue Resolution. In Proceedings of the Deep Learning for Code (DL4C)
Workshop at ICLR 2025 .
[13] Hui Yang, Sifu Yue, and Yunzhong He. 2023. Auto-GPT for Online Decision
Making: Benchmarks and Additional Opinions. arXiv preprint arXiv:2306.02224
(2023).
[14] John Yang, Carlos E. Jimenez, Alexander Wettig, Kilian Lieret, Shunyu Yao,
Karthik Narasimhan, and Ofir Press. 2024. SWE-agent: Agent-Computer Inter-
faces Enable Automated Software Engineering. arXiv preprint arXiv:2405.15793
(2024).
[15] Shunyu Yao, Jeffrey Zhao, Dian Yu, Nan Du, Izhak Shafran, Karthik Narasimhan,
and Yuan Cao. 2023. ReAct: Synergizing Reasoning and Acting in Language
Models. arXiv:2210.03629 [cs.CL] https://arxiv.org/abs/2210.03629
[16] Yuntong Zhang, Haifeng Ruan, Zhiyu Fan, and Abhik Roychoudhury.
2024. AutoCodeRover: Autonomous Program Improvement. arXiv preprint
arXiv:2404.05427 (2024).
Received 8 June 2025