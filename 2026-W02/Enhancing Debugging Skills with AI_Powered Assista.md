# Enhancing Debugging Skills with AI-Powered Assistance: A Real-Time Tool for Debugging Support

**Authors**: Elizaveta Artser, Daniil Karol, Anna Potriasaeva, Aleksei Rostovskii, Katsiaryna Dzialets, Ekaterina Koshchenko, Xiaotian Su, April Yi Wang, Anastasiia Birillo

**Published**: 2026-01-05 19:20:59

**PDF URL**: [https://arxiv.org/pdf/2601.02504v1](https://arxiv.org/pdf/2601.02504v1)

## Abstract
Debugging is a crucial skill in programming education and software development, yet it is often overlooked in CS curricula. To address this, we introduce an AI-powered debugging assistant integrated into an IDE. It offers real-time support by analyzing code, suggesting breakpoints, and providing contextual hints. Using RAG with LLMs, program slicing, and custom heuristics, it enhances efficiency by minimizing LLM calls and improving accuracy. A three-level evaluation - technical analysis, UX study, and classroom tests - highlights its potential for teaching debugging.

## Full Text


<!-- PDF content starts -->

Enhancing Debugging Skills with AI-Powered Assistance:
A Real-Time Tool for Debugging Support
Elizaveta Artser
JetBrains Research
Munich, Germany
elizaveta.artser@jetbrains.comDaniil Karol
JetBrains Research
Berlin, Germany
daniil.karol@jetbrains.comAnna Potriasaeva
JetBrains Research
Belgrade, Serbia
anna.potriasaeva@jetbrains.com
Aleksei Rostovskii
JetBrains Research
Berlin, Germany
aleksei.rostovskii@jetbrains.comKatsiaryna Dzialets
JetBrains Research
Munich, Germany
katsiaryna.dzialets@jetbrains.comEkaterina Koshchenko
JetBrains Research
Amsterdam, Netherlands
ekaterina.koshchenko@jetbrains.com
Xiaotian Su
ETH Zurich
Zurich, Switzerland
xiaotian.su@inf.ethz.chApril Yi Wang
ETH Zurich
Zurich, Switzerland
april.wang@inf.ethz.chAnastasiia Birillo
JetBrains Research
Belgrade, Serbia
anastasia.birillo@jetbrains.com
Abstract
Debugging is a crucial skill in programming education and software
development, yet it is often overlooked in CS curricula. To address
this, we introduce an AI-powered debugging assistant integrated
into an IDE. It offers real-time support by analyzing code, suggest-
ing breakpoints, and providing contextual hints. Using RAG with
LLMs, program slicing, and custom heuristics, it enhances efficiency
by minimizing LLM calls and improving accuracy. A three-level
evaluation – technical analysis, UX study, and classroom tests –
highlights its potential for teaching debugging.
CCS Concepts
•Applied computing →Computer-assisted instruction;•So-
cial and professional topics →Software engineering educa-
tion.
Keywords
Intelligent Tutoring, Debugging, In-IDE Learning, Generative AI
ACM Reference Format:
Elizaveta Artser, Daniil Karol, Anna Potriasaeva, Aleksei Rostovskii, Kat-
siaryna Dzialets, Ekaterina Koshchenko, Xiaotian Su, April Yi Wang, and Anas-
tasiia Birillo. 2026. Enhancing Debugging Skills with AI-Powered Assis-
tance: A Real-Time Tool for Debugging Support. In2026 IEEE/ACM 48th
International Conference on Software Engineering (ICSE-SEET ’26), April
12–18, 2026, Rio de Janeiro, Brazil.ACM, New York, NY, USA, 6 pages.
https://doi.org/10.1145/3786580.3786976
1 Introduction
Debugging is an essential skill in programming education [ 13]
and professional software development [ 28], yet it is rarely taught
This work is licensed under a Creative Commons Attribution 4.0 International License.
ICSE-SEET ’26, Rio de Janeiro, Brazil
©2026 Copyright held by the owner/author(s).
ACM ISBN 979-8-4007-2423-7/2026/04
https://doi.org/10.1145/3786580.3786976explicitly in CS curricula [ 2]. Most studies focus on pre-designed
exercises that teach debugging concepts [ 16,23,27,30,34] but fail
to help students apply these skills to real coding tasks [ 35]. At the
same time, providing real-time feedback can significantly improve
students’ performance and enthusiasm for learning [24, 25, 33].
Recent advancements in Large Language Models (LLMs) offer
potential solutions for providing real-time debugging assistance
during programming exercises. LLMs are already widely used for
program repair [ 18] and generating personalized feedback in edu-
cation [ 24]. However, LLMs can generate inaccurate or unreliable
responses, making it difficult to fully depend on them [ 10,20,31,36].
A recent study presented a promising approach to enhance the qual-
ity of next-step hint state-of-the-art methods by integrating LLMs
with IDE internals such as static analysis and code quality checkers,
improving the quality of next-step hints [ 5]. The similar approach is
widely used in other fields to enhance the output of models [ 8,17].
Since debugging is typically performed within IDEs, this method
could potentially be combined with LLMs to assist students with
debugging.
This paper introduces an AI-powered debugging assistant inte-
grated into the open-source JetBrains Academy plugin for in-IDE
learning [ 6]. The tool provides real-time debugging support by
analyzing students’ code, suggesting breakpoints for buggy pro-
grams, and offering contextual explanations. It integrates Retrieval
Augmented Generation (RAG) [ 15] with LLMs to generate solu-
tions and uses program slicing and custom heuristics to identify
breakpoint locations efficiently. The source code of the tool can be
found in the supplementary materials [ 3]. A three-level evaluation,
including technical analysis, UX evaluation, and classroom pilot
tests, showed promising results, highlighting the tool’s potential to
assist students in debugging effectively.
2 Related Work
A common way to teach debugging is through debugging exer-
cises [ 30,32,35], where students identify and (sometimes) fix bugsarXiv:2601.02504v1  [cs.SE]  5 Jan 2026

ICSE-SEET ’26, April 12–18, 2026, Rio de Janeiro, Brazil Elizaveta Artser et al.
(a) Task panel with (1) Check button,
(2) failed test, (3) session-start message.
(b) Code editor during a guided debugging session with (1) breakpoints suggested by the AI Debugging Assistant,
(2) breakpoints set by the student, and (3) breakpoint hint.
Figure 1: Main components of the AI Debugging Assistant tool.
in faulty code. Previous studies have expanded on this with ap-
proaches like scaffolding exercises with debug prints [ 12] or logs [ 11],
treating as a hypothesis-driven process [ 26,32] or adapting a trou-
bleshooting framework [ 21]. Strategies also include learning ma-
terials, such as manuals [ 16] or cheatsheets [ 4], as well as struc-
tured instruction, like debugging courses or interactive demonstra-
tions [ 9,34]. These approaches are effective, but often require extra
materials and significant educator involvement for timely support.
The emergence of LLMs offers new opportunities to support
students in debugging and practicing strategies. Chat-based assis-
tants [ 18,22] provide guarded feedback through conceptual expla-
nations, pseudo code, and targeted annotations to help students find
errors. Tools likeBugSpotter[ 30] generate debugging tasks, while
HypoCompass[ 24] focuses on hypothesis construction through di-
agnosing buggy code. Despite their potential, LLM-based tools face
challenges with reliability and seamless IDE integration.
To address this, some works embed debugging support into IDEs.
For example,Ladebug[ 23] provided a browser-based IDE-like envi-
ronment where students practiced in predefined exercises. Recently,
Noller et al. combined fault localization techniques with LLMs to
guide students through the debugging process inside the IDE [ 29],
using automated breakpoints and a chatbot interface. While closely
related, our work takes a different approach to breakpoint sugges-
tion, focusing on performance, quality, and the cost of the solution.
Additionally, we provide contextual hints for breakpoints directly
within the editor, reducing reliance on a chatbot interface. This
distinction highlights that AI-powered debugging support can be
integrated into IDEs in different ways, making it valuable to study
a range of approaches.Our work integrates debugging support into the IDE, aligning
with students’ workflows and reducing context switching. Unlike
previous methods using predefined tasks, our tool works on stu-
dents’ code for personalized support. By combining IDE internals
such as static analysis with LLMs, it provides reliable, scalable
assistance without adding to educators’ workload.
3 AI-Debugging Tool
3.1 Usage Pipeline
The AI-Debugging Assistant is integrated into the JetBrains Acad-
emy plugin for in-IDE learning [ 6]. Students can view task de-
scriptions and check their progress by clicking the Check button
(Figure 1a-1), which runs tests locally. If an error happens (Figure 1a-
2), the tool offers students the option to start aguided debugging
session(Figure 1a-3).
During the debugging session, the tool highlights recommended
breakpoints in purple (Figure 1b-1) to distinguish them from student-
set breakpoints (Figure 1b-2). Students follow these breakpoints
and view explanation hints (Figure 1b-3). After the session, the AI-
recommended breakpoints convert to regular breakpoints, allowing
students to continue the regular debugging process.
3.2 Tool Pipeline
The general idea of the approach is to generate a correct version
of the student’s solution that passes the failed test. Based on this,
the AI-Debugging assistant generates and explains the breakpoints
that should be shown to the student. Figure 2 illustrates the tool’s
internal pipeline. The tool is composed of three main components:

Enhancing Debugging Skills with AI-Powered Assistance: A Real-Time Tool for Debugging Support ICSE-SEET ’26, April 12–18, 2026, Rio de Janeiro, Brazil
Breakpoints to recommend
Author solutionLLM: generate a fixK solutions “passing” failed test
N generated solutions
RAG Database
LLM: does solution pass failed test?Fixed student solution
Backward and forward slicingCustom heuristics
Breakpoints with explanations
LLM: generate explanationRun tests and update RAGTop-1 solutionComponent 1 Program RepairStudent solution
Failed test
Similar solution not found:  generate new solutionSimilar solution found: return a similar solution from the databaseComponent 3:  Breakpoints ExplanationsComponent 2:  Breakpoints Recommendations
Figure 2: The tool pipeline: (1) program repair; (2) breakpoints recommendations; (3) breakpoints explanations components. The
program repair component takes a student solution and a failed test, searching the RAG database for a similar correct solution.
If a match is found, the process ends; if not, a new fixed solution is generated alongside the author’s initial solution. The
breakpoint recommendations component analyzes the fixed solution, and the breakpoint explanations component generates
explanations for each recommended breakpoint.
program repair (Figure 2-1), breakpoints recommendations (Fig-
ure 2-2), and breakpoints explanations (Figure 2-3). Currently, the
tool supports only Kotlin, but can be extended for other languages
in the future.
3.2.1 Program repair.The first key component is program repair
(Figure 2-1), which takes the student’s solution and thefailedtest
as input and returns a corrected solution that passesthattest. To
generate a fix, we use a combined approach: retrieving a correct
solution from existing ones using a RAG system [ 15] or generating
a new solution with an LLM. This minimizes unnecessary LLM
calls and avoids hallucinations [ 36], while remaining usable for
new courses without prior student submissions.
RAG system.The RAG system uses existing solutions and infor-
mation about tests, stored as embeddings generated by the LaBSE
model1. The database can be empty (for new tasks) or pre-filled
with previous submissions. When a debugging session is requested,
embeddings are generated for the student’s solution and failed test.
The system searches for a similar solution that passes the same test,
using a cosine similarity threshold of ≥0.8. If found, that solution
is returned. Otherwise, the LLM generates a new solution.
LLM solution generation.The gpt-4o model is used to gener-
ate corrected solutions. Since LLMs are prone to hallucinations [ 10,
20,31,36], five solutions are generated in parallel, and the best one
is selected. Instead of running actual tests, the o3-mini model acts
as a binary classifier to predict whether each solution will pass the
failed test. Corrected solutions predicted topassare ranked using
embeddings, prioritizing those closest to the student’s original code.
The top-ranked solution is returned as the output. The full prompts
can be found in the supplementary materials [3].
1LaBSE model: https://huggingface.co/sentence-transformers/LaBSEUploading new solutions to the RAG system. The LLM solu-
tion generation creates new submissions, which are added to the
RAG system to expand the database and reduce future LLM calls.
Initially, we rely on the model’s predictions to save time (Figure 2-2).
To ensure only valid solutions are added, we later execute actual
tests. This step not only enhances the database but also allows us
to validate our approach and identify cases where the LLM’s test
failure predictions are inaccurate. This step is performed indepen-
dently of the rest of the AI-Debugging assistant tool and does not
affect its performance.
3.2.2 Breakpoints recommendations.When the correct version of
the solution is generated, the next step is to suggest aset of possi-
ble breakpoints locationsto the student (Figure 2-2). In our work,
we consider two types of breakpoints: (1) breakpoints on the lines
where thebug is locatedand needs to be fixed directly, and (2) break-
points on the lines that arelikely affected by the bug, which guide
the student to the areas needing fixes. The first type of breakpoints
is determined automatically by calculating the difference between
the student’s current solution and the correct one – these are the
exact locations that need to be changed for the solution to become
correct. The more interesting part is using this information to find
additional places the student should check to make these fixes.
We use a combined approach to identify the lines that are likely
affected by the bug. This includes programmingbackward and
forward slicing[ 1,19] as well ascustom heuristicsto limit the number
of recommended breakpoints. To show the final list of breakpoints
to the student, we take the intersection of the breakpoints. We
believe this approach works because it addresses two key problems:
1) it retains the most essential breakpoints, which are the most
likely to impact the bug, and 2) it limits their number. This ensures
that our system does not create an excessive number of breakpoints

ICSE-SEET ’26, April 12–18, 2026, Rio de Janeiro, Brazil Elizaveta Artser et al.
and allows the student to focus on debugging in the most relevant
parts of the program.
Backward and forward slicing. Code slicing identifies code
parts that influence (backward slice) or are influenced by (forward
slice) a specific program element. It leverages static information to
locate problem areas and is widely used in debugging tools [ 1,19].
We built data and control flow graphs based on Program Struc-
ture Interface (PSI)2from the IntelliJ IDEA API to extract data
control dependencies.Data dependenciesoccur when the value of
one variable depends on another,e.g.in properties, or function calls.
For example, in var c = a + b ,chas backward dependencies
onaandb, and aandbhave forward dependencies on c.Control
dependenciesoccur when the execution of one statement depends
on another,e.g.in ifstatements or loops. For example, in an if
statement, both conditional branches have backward control depen-
dencies on the ifstatement. As a result, this step produces forward
and backward dependencies to the incorrect parts of the student’s
code. However, slicing alone may recommend too many lines, as it
includesallaffected lines in the code.
Heuristics. Based on our programming experience and previous
research [ 14], we created a list of custom heuristics for possible
places of breakpoints. We avoided using LLMs to ensure a deter-
ministic approach, prevent model hallucinations, and reduce costs.
(1)Conditional Statement Heuristic.When the fix involves
changes to a condition statement,e.g., if,when , orwhile ,
place breakpoints at the beginning of all code blocks that are
executed based on these conditions. This allows monitoring
how the logic flows under different scenarios.
(2)Variable Modification Heuristic.If the fix involves a line
where a variable is modified, such as a property change, a
binary expression, a loop parameter in a for expression, or a
property in a while condition, place breakpoints to observe
the state of the variable before and after its modification.
(3)Function Scope Heuristic.When the fix occurs within
a function definition, place breakpoints at the beginning
of the function to trace its execution and at all calls to the
function in the code. This helps identify whether the function
is invoked and behaves correctly after the change.
3.2.3 Breakpoints explanations.The final step of the pipeline is the
generation of breakpoints explanations (Figure 2-3). We use gpt-4o
to generate explanations for each breakpoint, helping students focus
on specific code lines. The model identifies variables or objects being
modified and provides detailed instructions for understanding these
changes. For breakpoints on lines that arelikely affected by the bug,
we request explanations about how those lines relate to the final
error. For breakpoints on lines where thebug is located, the model
is instructed to explain why the error occurs [3].
4 Evaluation
4.1 RQ1: Breakpoints Recommendation Quality
This section presents the technical evaluation of the proposed ap-
proach to answerRQ1: How precise is the AI-powered debugging
system in recommending breakpoints?. The evaluation focuses on
2PSI: https://plugins.jetbrains.com/docs/intellij/psi.htmlTable 1: Execution time comparison across candidate models,
where P – Precision, R – Recall. Grey indicates the best model.
Model Avg (s) Max (s) P R F1
gpt-4o-mini(2048)9.11 12.18 - - -
gpt-4o-mini(1024)8.24 10.680.680.880.76
o3-mini (low) 6.50 10.23 0.70 0.87 0.78
o3-mini (medium)17.85 44.92 - - -
the program repair (Figure 2-1) and the breakpoints recommenda-
tion (Figure 2-2) modules.
4.1.1 Program Repair Evaluation.The developed system relies on
corrected student solutions that must pass the failed test. We use
an LLM-based model to predict whether this solution will pass the
failed test. To investigate how well such classifiers can replace actual
test execution, we evaluated their feasibility as binary predictors of
whether a generated fix passes the failed test.
Execution time.We benchmarked four models on a sample of
150 solutions generated for theKotlin Onboarding Introduction3. We
ran the classifier on each solution, using a MacBook Pro (Apple M3
Max chip and 64 GB of memory). Based on the measurements (see
Table 1), o3-mini (low) andgpt-4o-mini(1024) models were
selected for further evaluation because as the fastest ones.
Quality evaluation.To evaluate how reliably the models can
predict whether a generated fix is correct, we conducted an evalu-
ation using1 ,796incorrect Kotlin student submissions. For each
submission, a fix was generated by gpt-4o (see Figure 2-1). Then
we ran two fastest models from the previous experiment on each
fixed submission to predict if the fixed code isactuallyfixes the
student solution and validated the results by executing tests on each
fixed submission. The performance of the classifiers is summarized
in Table 1.o3-miniwas chosen, balancing efficiency and quality.
4.1.2 Breakpoints Recommendation Evaluation.Another key com-
ponent of the developed system is the breakpoints recommendation
module (Figure 2-2). We assume the student’s incorrect solution
and the corresponding fixed solution are available. We evaluated
the proposed approach, using a sample of 32 student solution pairs
and their fixed versions from the Kotlin course3. Two authors of
this paper with over two years of research experience and more
than four years of Kotlin programming experience independently
labeled the data by identifying potential breakpoint locations. After
labeling, they discussed the results to reach a consensus. Using the
labeled data, we evaluated the quality of the system. We achieved
a Precision of0 .9, Recall of0 .7, and F1 score of0 .79. These re-
sults demonstrate strong performance, so we decided to retain the
current approach for further evaluation with students.
4.2 RQ2: Tool’s Usability
The proposed system is embed to in-IDE environment, which might
be hard for students to use [ 7]. To address this problem and answer
RQ2: How do students perceive the tool’s user experience, and how
can it be improved?, we conducted a usability study with students.
3Kotlin Onboarding Introduction course: https://plugins.jetbrains.com/plugin/21067-
kotlin-onboarding-introduction

Enhancing Debugging Skills with AI-Powered Assistance: A Real-Time Tool for Debugging Support ICSE-SEET ’26, April 12–18, 2026, Rio de Janeiro, Brazil
Participants. We invited eight CS students with basic Kotlin
knowledge to participate in 15-minute semi-structured interviews.
All participants had limited experience with debugging,i.e., they
had tried using a debugger but did not use it regularly.
Experiment setting. The experiment had two phases: (1) three
pilot interviews, and (2) five interviews used for analysis. Interviews,
conducted in English, used an interactive Figma prototype with
limited functionality to allow interactions with the AI debugging
assistant, but did not allow code editing. Participation was voluntary.
We informed students about the study’s purpose, data collection,
and usage. Participants solved a beginner-level Kotlin problem with
a buggy solution, using the prototype to locate the bug and explain
how to fix it. The full interview script and the task can be found in
the supplementary materials [3].
Results. Most participants found the tool intuitive after initial
use. They appreciated the pre-configured breakpoints, which en-
couraged deeper exploration. Overall, the prototype was easy to
use. Many participants liked the idea of hints but found the content
insufficient. Five noted that generic hints, like “Check if this value is
correct” were insufficient. The final version was updated to include
more detailed explanations. Another issue noted by students was
that breakpoint explanations overlapped with the code. In the ini-
tial prototype, hints were displayed by default. In the final version,
hints now appear only when hovering over a breakpoint.
4.3 RQ3: Student Perspective
The last step of evaluation was conducted as a pilot study in a class-
room to answerRQ3: What are novice students’ initial impressions
of the tool in a classroom setting?
Participants.We invited 20 1st- and 2nd-year Engineering Sci-
ences bachelor’s students. All participants had basic programming
experience, having completed at least one semester of program-
ming. To ensure clarity in the experiment, we selected students
with limited debugging experience, assessed by their frequency of
usage and self-reported confidence in the debugging process.
Experiment setting.The students participated in a 90-minute
session conducted in a controlled lab environment. Participation
was voluntary, and students were informed about the study’s pur-
pose, data collection, and usage. No personal data was collected.
Students were provided with laptops preinstalled with the tool and
printed instructions, including a description of the tool’s concept
and key components. During the session, participants were tasked
with two types of programming exercises – several regular pro-
gramming tasks requiring to implement some functions, and two
debugging tasks – one before and one after – with multiple bugs in
each [3]. After completing the tasks, students filled out a survey.
Results.The experiments showed that only 8 out of 20 students
used the AI Debugging Assistant during their debugging sessions.
The most likely reason was a lack of Kotlin knowledge combined
with the limited session time, which led to syntax errors and pre-
vented students from fully utilizing the assistant. However, students
were generally positive about the tool’s concept.
According to results for the 1-5 Likert scale questions, the stu-
dents who used the proposed system provided moderately positive
feedback, with an average score of3 .13for the correctness of theprovided breakpoint locations and3 .25for the clarity of the break-
point explanations. In addition, three students highlighted that
the tool’s ability to localize and summarize the source of the bug
was helpful for debugging. At the same time, two students men-
tioned that the automatically suggested breakpoints were useful
and helped ease the debugging process.
Students suggested three main areas for improving the tool. First,
four students recommended integrating the tool with a chat-based
system to allow students to ask clarification questions. This idea
could be easily implemented in future updates. Second, the user
experience could be improved, as students found it unclear that
they needed to hover over breakpoints to see explanation messages.
This issue could likely be addressed with onboarding tools that
demonstrate this feature during the first few uses. Lastly, some
students mentioned adding a code generation feature to provide
corrected solutions. However, this would require future investiga-
tion, as the tool’s primary goal is to teach and assist with debugging
rather than directly providing correct solutions.
5 Threats to Validity
The proposed approach has several key limitations. First, the tech-
nical evaluation was limited in scale. A greater diversity of tasks
could enhance the custom heuristic set by identifying additional
cases. This remains an area for future investigation. Second, the
pilot classroom evaluation involved a small number of participants.
While 20 students were invited, only 8 used the tool during the
session. Although we gathered initial evidence suggesting students
appreciate the approach, we could not assess the actual learning
impact of the tool, and results may also have been influenced by
novelty effects. Future studies with a larger time for the experiment
and more participants are planned to address this limitation.
6 Conclusion
This study addresses the gap in teaching debugging skills in CS cur-
ricula by presenting an AI-powered debugging assistant integrated
into the IDE. The tool offers real-time, personalized debugging
support by identifying breakpoints, providing explanations, and
using techniques like RAG and program slicing to ensure accuracy
and efficiency. Our three-level evaluation demonstrated the feasi-
bility of this approach and suggested directions for future work on
assessing its impact on student debugging practices and learning
outcomes. This work provides a foundation for integrating real-
time debugging tools into programming education and encourages
further research on scalable solutions for teaching debugging.
Acknowledgments
We want to thank Evgenii Moiseenko for mentoring our team on
this project, especially in programming slicing, code review, and
discussing algorithms. A big thank you to Yaroslav Zharov and Rauf
Kurbanov for their ML mentorship and guidance in building the
system. We also thank the Human AI Experience team at JetBrains
Research for designing and conducting the UX study with students,
providing valuable insights to improve the system. Lastly, we thank
the ETH Zurich Decision Science Laboratory for coordinating the
student lab study.

ICSE-SEET ’26, April 12–18, 2026, Rio de Janeiro, Brazil Elizaveta Artser et al.
References
[1]Hiralal Agrawal and Joseph R Horgan. 1990. Dynamic program slicing.ACM
SIGPlan Notices25, 6 (1990), 246–256.
[2]Marzieh Ahmadzadeh, Dave Elliman, and Colin Higgins. 2005. An analysis of
patterns of debugging among novice computer science students.ACM SIGCSE
Bulletin37, 3 (2005), 84–88. doi:10.1145/1151954.1067472
[3]Elizaveta Artser, Daniil Karol, Anna Potriasaeva, Aleksei Rostovskii, Katsiaryna
Dzialets, Ekaterina Koshchenko, Xiaotian Su, April Yi Wang, and Anastasiia
Birillo. 2025.Supplementary materials. Retrieved January 4, 2026 from https:
//zenodo.org/records/18045875
[4]Andrew Ash and John Hu. 2025. WIP: Exploring the Value of a Debugging
Cheat Sheet and Mini Lecture in Improving Undergraduate Debugging Skills and
Mindset.arXiv preprint arXiv:2506.11339(2025).
[5]Anastasiia Birillo, Elizaveta Artser, Anna Potriasaeva, Ilya Vlasov, Katsiaryna
Dzialets, Yaroslav Golubev, Igor Gerasimov, Hieke Keuning, and Timofey Bryksin.
2024. One step at a time: Combining llms and static analysis to generate next-step
hints for programming tasks. InProceedings of the 24th Koli Calling International
Conference on Computing Education Research. 1–12.
[6]Anastasiia Birillo, Mariia Tigina, Zarina Kurbatova, Anna Potriasaeva, Ilya Vlasov,
Valerii Ovchinnikov, and Igor Gerasimov. 2024. Bridging education and develop-
ment: Ides as interactive learning platforms. InProceedings of the 1st ACM/IEEE
Workshop on Integrated Development Environments. 53–58.
[7]Anastasiia Birillo, Ilya Vlasov, Katsiaryna Dzialets, Hieke Keuning, and Timofey
Bryksin. 2025. In-IDE Programming Courses: Learning Software Development in
a Real-World Setting. In2025 IEEE/ACM Second IDE Workshop (IDE). IEEE, 1–6.
[8]Scott Blyth, Sherlock A Licorish, Christoph Treude, and Markus Wagner. 2025.
Static analysis as a feedback loop: Enhancing llm-generated code beyond cor-
rectness. (2025), 100–109.
[9]Liia Butler, Charlotte Kiesel, Dipayan Mukherjee, Mohammed Hassan, Mattox
Beckman, and Geoffrey Herman. 2025. ILDBug: A New Approach to Teaching
Debugging. InProceedings of the 56th ACM Technical Symposium on Computer
Science Education V. 2. 1731–1731.
[10] Yupeng Chang, Xu Wang, Jindong Wang, Yuan Wu, Linyi Yang, Kaijie Zhu, Hao
Chen, Xiaoyuan Yi, Cunxiang Wang, Yidong Wang, et al .2024. A survey on
evaluation of large language models.ACM transactions on intelligent systems and
technology15, 3 (2024), 1–45.
[11] Ryan Chmiel and Michael C Loui. 2004. Debugging: from novice to expert.Acm
Sigcse Bulletin36, 1 (2004), 17–21.
[12] Joel Fenwick and Peter Sutton. 2012. Using quicksand to improve debugging
practice in post-novice level students. InProceedings of the Fourteenth Australasian
Computing Education Conference-Volume 123. 141–146.
[13] Sue Fitzgerald, Renée McCauley, Brian Hanks, Laurie Murphy, Beth Simon, and
Carol Zander. 2009. Debugging from the student perspective.IEEE Transactions
on Education53, 3 (2009), 390–396.
[14] Eduardo Andreetta Fontana and Fabio Petrillo. 2021. Mapping breakpoint types:
an exploratory study. In2021 IEEE 21st International Conference on Software
Quality, Reliability and Security (QRS). IEEE, 1014–1023.
[15] Yunfan Gao, Yun Xiong, Xinyu Gao, Kangxiang Jia, Jinliu Pan, Yuxi Bi, Yixin
Dai, Jiawei Sun, Haofen Wang, and Haofen Wang. [n. d.]. Retrieval-augmented
generation for large language models: A survey.arXiv preprint arXiv:2312.10997
([n. d.]).
[16] Rita Garcia, Chieh-Ju Liao, and Ariane Pearce. 2022. Read the debug manual: A
debugging manual for CS1 students. In2022 IEEE Frontiers in Education Conference
(FIE). IEEE, 1–7.
[17] Imen Jaoua, Oussama Ben Sghaier, and Houari Sahraoui. 2025. Combining Large
Language Models with Static Analyzers for Code Review Generation. In2025
IEEE/ACM 22nd International Conference on Mining Software Repositories (MSR).
IEEE, 174–186.
[18] Majeed Kazemitabaar, Runlong Ye, Xiaoning Wang, Austin Zachary Henley,
Paul Denny, Michelle Craig, and Tovi Grossman. 2024. Codeaid: Evaluating
a classroom deployment of an llm-based programming assistant that balances
student and educator needs. InProceedings of the 2024 chi conference on human
factors in computing systems. 1–20.[19] Bogdan Korel and Janusz Laski. 1988. Dynamic program slicing.Information
processing letters29, 3 (1988), 155–163.
[20] Charles Koutcheme, Nicola Dainese, Sami Sarsa, Arto Hellas, Juho Leinonen,
Syed Ashraf, and Paul Denny. 2025. Evaluating language models for generating
and judging programming feedback. InProceedings of the 56th ACM Technical
Symposium on Computer Science Education V. 1. 624–630.
[21] Chen Li, Emily Chan, Paul Denny, Andrew Luxton-Reilly, and Ewan Tempero.
2019. Towards a framework for teaching debugging. InProceedings of the Twenty-
First Australasian Computing Education Conference. 79–86.
[22] Mark Liffiton, Brad E Sheese, Jaromir Savelka, and Paul Denny. 2023. Codehelp:
Using large language models with guardrails for scalable support in program-
ming classes. InProceedings of the 23rd Koli Calling International Conference on
Computing Education Research. 1–11.
[23] Andrew Luxton-Reilly, Emma McMillan, Elizabeth Stevenson, Ewan Tempero,
and Paul Denny. 2018. Ladebug: an online tool to help novice programmers
improve their debugging skills. InProceedings of the 23rd annual acm conference
on innovation and technology in computer science education. 159–164.
[24] Qianou Ma, Hua Shen, Kenneth R. Koedinger, and Sherry Tongshuang Wu. 2024.
How to Teach Programming in the AI Era? Using LLMs as a Teachable Agent for
Debugging. InProceedings of the 2024 ACM Conference on International Computing
Education Research (ICER). ACM.
[25] Samiha Marwan, Ge Gao, Susan Fisk, Thomas W Price, and Tiffany Barnes. 2020.
Adaptive immediate feedback can improve novice programming engagement
and intention to persist in computer science. InProceedings of the 2020 ACM
conference on international computing education research. 194–203.
[26] Tilman Michaeli and Ralf Romeike. 2019. Improving debugging skills in the
classroom: The effects of teaching a systematic debugging process. InProceedings
of the 14th workshop in primary and secondary computing education. 1–7.
[27] Michael A Miljanovic and Jeremy S Bradbury. 2017. Robobug: a serious game
for learning debugging techniques. InProceedings of the 2017 acm conference on
international computing education research. 93–100.
[28] Monika AF Müllerburg. 1983. The role of debugging within software engineering
environments. InProceedings of the symposium on High-level debugging. 81–90.
[29] Yannic Noller, Erick Chandra, Srinidhi Chandrashekar, Kenny Choo, Cyrille
Jegourel, Oka Kurniawan, and Christopher M Poskitt. 2025. Simulated interactive
debugging.arXiv preprint arXiv:2501.09694(2025).
[30] Victor-Alexandru Padurean, Paul Denny, and Adish Singla. 2025. BugSpotter:
Automated Generation of Code Debugging Exercises. InProceedings of the 56th
ACM Technical Symposium on Computer Science Education V. 1. 896–902.
[31] Samantha Boatright Smith, Heather Wei, Abby O’Neill, Aneesh Durai, John
DeNero, JD Zamfirescu-Pereira, and Narges Norouzi. 2025. Spotting AI Missteps:
Students Take on LLM Errors in CS1. InProceedings of the 56th ACM Technical
Symposium on Computer Science Education V. 2. 1627–1628.
[32] Jacqueline Whalley, Amber Settle, and Andrew Luxton-Reilly. 2021. Analysis
of a process for introductory debugging. InProceedings of the 23rd Australasian
Computing Education Conference. 11–20.
[33] Joseph B Wiggins, Fahmid M Fahid, Andrew Emerson, Madeline Hinckle, Andy
Smith, Kristy Elizabeth Boyer, Bradford Mott, Eric Wiebe, and James Lester. 2021.
Exploring novice programmers’ hint requests in an intelligent block-based coding
environment. InProceedings of the 52nd ACM technical symposium on computer
science education. 52–58.
[34] G Aaron Wilkin. 2025. "Debugging: From Art to Science" A Case Study on a
Debugging Course and Its Impact on Student Performance and Confidence. In
Proceedings of the 56th ACM Technical Symposium on Computer Science Education
V. 1. 1225–1231.
[35] Stephanie Yang, Miles Baird, Eleanor O’Rourke, Karen Brennan, and Bertrand
Schneider. 2025. Decoding debugging instruction: A systematic literature review
of debugging interventions.ACM Transactions on Computing Education24, 4
(2025), 1–44.
[36] Ziyao Zhang, Chong Wang, Yanlin Wang, Ensheng Shi, Yuchi Ma, Wanjun Zhong,
Jiachi Chen, Mingzhi Mao, and Zibin Zheng. 2025. Llm hallucinations in practical
code generation: Phenomena, mechanism, and mitigation.Proceedings of the
ACM on Software Engineering2, ISSTA (2025), 481–503.