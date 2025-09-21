# GPU Programming for AI Workflow Development on AWS SageMaker: An Instructional Approach

**Authors**: Sriram Srinivasan, Hamdan Alabsi, Rand Obeidat, Nithisha Ponnala, Azene Zenebe

**Published**: 2025-09-17 05:11:05

**PDF URL**: [http://arxiv.org/pdf/2509.13703v1](http://arxiv.org/pdf/2509.13703v1)

## Abstract
We present the design, implementation, and comprehensive evaluation of a
specialized course on GPU architecture, GPU programming, and how these are used
for developing AI agents. This course is offered to undergraduate and graduate
students during Fall 2024 and Spring 2025. The course began with foundational
concepts in GPU/CPU hardware and parallel computing and progressed to develop
RAG and optimizing them using GPUs. Students gained experience provisioning and
configuring cloud-based GPU instances, implementing parallel algorithms, and
deploying scalable AI solutions. We evaluated learning outcomes through
assessments, course evaluations, and anonymous surveys. The results reveal that
(1) AWS served as an effective and economical platform for practical GPU
programming, (2) experiential learning significantly enhanced technical
proficiency and engagement, and (3) the course strengthened students'
problem-solving and critical thinking skills through tools such as TensorBoard
and HPC profilers, which exposed performance bottlenecks and scaling issues.
Our findings underscore the pedagogical value of integrating parallel computing
into STEM education. We advocate for broader adoption of similar electives
across STEM curricula to prepare students for the demands of modern,
compute-intensive fields.

## Full Text


<!-- PDF content starts -->

GPU Programming for AI Workflow Development
on AWS SageMaker: An Instructional Approach
Sriram Srinivasan, Hamdan Alabsi, Rand Obeidat, Nithisha Ponnala, Azene Zenebe
Bowie State University, MD, USA
{ssrinivasan, halabsi, robeidat, azenebe}@bowiestate.edu,
ponnalan0103@students.bowiestate.edu
Abstract—We present the design, implementation, and compre-
hensive evaluation of a specialized course on GPU architecture,
GPU programming, and how these are used for developing AI
agents. This course is offered to undergraduate and graduate
students during Fall 2024 and Spring 2025. The course began
with foundational concepts in GPU/CPU hardware and parallel
computing and progressed to develop RAG and optimizing
them using GPUs. Students gained experience provisioning and
configuring cloud-based GPU instances, implementing parallel
algorithms, and deploying scalable AI solutions. We evaluated
learning outcomes through assessments, course evaluations, and
anonymous surveys. The results reveal that (1) A WS served as an
effective and economical platform for practical GPU program-
ming, (2) experiential learning significantly enhanced technical
proficiency and engagement, and (3) the course strengthened
students’ problem-solving and critical thinking skills through
tools such as TensorBoard and HPC profilers, which exposed
performance bottlenecks and scaling issues. Our findings under-
score the pedagogical value of integrating parallel computing
into STEM education. We advocate for broader adoption of
similar electives across STEM curricula to prepare students for
the demands of modern, compute-intensive fields.
Index Terms—GPU, Parallel Computing, A WS SageMaker,
HPC profiling, AI agents, Deep Learning
I. INTRODUCTION
We are currently entering an era where Artificial Intelligence
(AI) agents are capable of reasoning, planning, and acting
with unprecedented autonomy. To build these sophisticated AI
agents, High-Performance Computing (HPC), and specifically
Graphics Processing Unit (GPU) programming skills, play a
crucial role. Traditional computing paradigms simply cannot
handle the immense computational demands of modern AI.
Training deep learning models and deploying real-time, high-
fidelity AI solutions requires a level of parallel processing
power that only GPUs can deliver. There is an exploding
demand for AI engineers and data scientists [1] [2] [3] who
understand how GPUs work and how to exploit parallelism
in GPU architecture to design intelligent algorithms. Foun-
dational GPU programming skills are rapidly becoming an
indispensable competency for anyone pursuing a career in
AI. Consequently, infusing GPU programming into graduate
and undergraduate STEM curricula is a critical necessity to
prepare the next generation of scientists, engineers, and data
professionals.
In Fall 2024 and Spring 2025, we taught an elective
course (Special Topics). This course was taken by seniorsmajoring in Data Analytics and Information Systems who
have completed an Introduction to Programming course. It’s
also offered as a cross-listed elective for graduate students
in Information Systems. In both semesters, our focus was on
developing AI workflows and agents with GPU programming
using libraries like RAPIDS and Dask [4]. The first half of
the 16-week course concentrated on introducing basic HPC
concepts, demonstrating how to use accelerators to improve
performance, and implementing simple parallel algorithms on
accelerators. Previously, when this course was offered, we
never introduced HPC concepts. Two factors contributed to
revising the curriculum and infusing HPC for developing AI
agents: 1) The support received through The U.S. National
Science Foundation (NSF) funding, specifically the NSF Ex-
pand AI grant, which enabled us to procure Amazon Web
Services (AWS) credits to facilitate this transition [5]. 2)
The availability and accessibility of infrastructure provided
by AWS, which offers affordable, on-demand access to GPU
instances as a cloud service. This capability eliminated the
need for maintaining on-campus HPC infrastructure to support
the course.
Combining both semesters, we had about thirty-nine stu-
dents enrolled in this course. All students had a Python pro-
gramming background, as the prerequisite course (Introduction
to Programming) was taught in Python. Most students lacked
C programming knowledge, and since the course’s focus was
on developing AI agents using HPC, they all utilized Python
JIT (Just-in-Time) libraries such as Numba [6] [7] [8] and
CuPy [9] for parallel programming. As most of them had no
prior knowledge of parallel computing, the course began with
the basics of writing simple parallel programs and running
them on GPUs. Students were given AWS console/Educate
access during the first class, where they set up credentials and
were taught how to use Python scripts to spin up and terminate
instances. Students were familiar with AWS SageMaker, which
offers Jupyter Notebook, allowing them to write and run
code in one place. Starting week 2, we explored into the
foundations of GPU computing, where students learned the
anatomy of Compute Unified Device Architecture (CUDA)
threads and blocks and performed hands-on experiments with
Matrix operations using CuPy. As we progressed, students fo-
cused on data flow between the Central Processing Unit (CPU)
(host) and GPU (device). They started profiling and used
critical thinking and problem-solving skills to address memoryarXiv:2509.13703v1  [cs.DC]  17 Sep 2025

bottlenecks, which impact performance. Before midterm, we
introduced JIT libraries, as discussed, to exploit GPU par-
allelism for large datasets. After the midterm, students built
Graph Convolutional Networks (GCNs) and trained them on a
GPU; they explored various ways to use parallel programming
concepts to aggregate node features on multiple GPUs for
large-scale, real-world networks. Students were also required
to build Retrieval Augmented Generation (RAG) systems
and experiment with GPU-tuned retrievers and generators to
optimize latency and throughput. In this paper, we present the
evaluation results of the course based on multiple instruments:
students’ performance on assessments, their end-of-semester
course evaluations, and two anonymous surveys (mid and
post-course). Our takeaways from offering this course are: 1)
AWS or any cloud resources are great and effective tools for
infusing parallel programming with modern GPUs; 2) The best
way to infuse HPC concepts to STEM students with minimal
programming background is through hands-on learning with
a mix of lecture and in-class labs; 3) Infusing HPC concepts
into the curriculum to build AI agents has a positive impact on
student learning, enabling them to apply these concepts to real-
world data analysis tasks. To reinforce this learning, graded
activities that require independent work should be designed as
extensions of in-class labs, challenging students to apply their
critical thinking and problem-solving skills. Therefore, we
recommend the continued offering of this course and propose
exploring its designation as a General Education course, with
an introductory programming course as a prerequisite.
The remainder of the paper is organized as follows: Sec-
tion II reviews related work, Section III outlines the course
structure, Section IV presents evaluation results, and Section
V concludes with future directions.
II. RELATEDWORK
The increasing pervasiveness and ubiquity of HPC systems,
largely driven by the unprecedented growth and reliance on
AI, requires the efficient execution of complex computational
tasks. Consequently, GPUs have become indispensable compo-
nents for optimizing the computing paradigm within AI appli-
cations, with major platforms such as Google Cloud, NVIDIA,
and AWS SageMaker extensively utilizing GPU-accelerated
and cloud-based computing resources to meet these demands.
This technological evolution underscores the urgent need for
academic curricula to incorporate HPC concepts, particularly
parallel and GPU programming, to prepare students for current
and emerging computational challenges.
Prasad et al. proposed a practical framework as outlined
in the NSF/IEEE TCPP Curriculum Initiative on Parallel and
Distributed Computing, specifically the Core Topics for Under-
graduates [10]. This initiative provides a nationally endorsed
framework for embedding parallel and distributed computing
concepts into undergraduate education. Chen et al. further
support this initiative, highlighting the importance of com-
puting competency, particularly in the context of developing
practical skills for high-performance computing. Their support
reflects a growing consensus that undergraduate computerscience education must evolve to reflect the realities of modern
computational environments [11].
Arroyo echoes this need by promoting the integration
of HPC and distributed programming across core computer
science curricula. He proposes a progressive pedagogical
framework that bridges sequential and parallel paradigms
using a combination of traditional and novel teaching tools
to enhance student preparedness [12]. In alignment with Ar-
royo’s perspective, NVIDIA offers hands-on GPU-accelerated
training programs in AI, HPC, and workflow automation that
include teaching kits and hybrid learning formats tailored for
university integration, supporting the importance of aligning
academic instruction with evolving industry practices [13].
Further advancing educational strategies, Qasem et al.
present innovative approaches to parallel and distributed com-
puting education that resonate with the need for practical GPU
programming skills. Their work highlights GPU programming
courses modeled on frameworks such as Carpentries, alongside
specialized bootcamps focused on AI workloads in HPC
systems, which collectively support hands-on learning and
interdisciplinary student engagement [14]. In a related context,
Adams et al. analyze the rapid adoption of cloud computing
services and their implications for higher education curricula.
They argue that the growing demand for practical, hands-on
training in cloud computing, programming, and data science
requires structured teaching materials, canonical learning ob-
jectives, and modular course design. Their approach ensures
that cloud computing content can be seamlessly integrated into
a variety of computing courses, facilitating student exposure
to this rapidly evolving field [15].
Neelima et al. provide a case study of HPC integra-
tion into an undergraduate engineering curriculum over six
years, documenting a progressive expansion from shared and
distributed memory programming to accelerator and hybrid
programming models. Their findings reveal that deliberate
pedagogical strategies and institutional support can overcome
initial challenges, ultimately yielding measurable benefits to
both students and stakeholders [16]. Similarly, Qasem et
al. advocate for a module-driven approach to embedding
heterogeneous computing topics across the computer science
curriculum. This strategy addresses curricular constraints by
dispersing focused modules within existing courses, a method
that demonstrated increased student engagement and founda-
tional understanding of heterogeneous architectures pertinent
to HPC, mobile processing, and Internet of Things (IoT)
domains [17].
Bridging theory and practice, Xu’s senior elective course ef-
fectively teaches OpenMP (Open Multi-Processing), Message
Passing Interface (MPI), and CUDA using Google Colab and
hands-on Raspberry Pi cluster construction to teach heteroge-
neous and parallel computing. Xu’s evaluation reports that this
hybrid model significantly enhances students’ comprehension
of data organization and parallel programming concepts, high-
lighting the value of experiential learning in HPC education
[18].
In parallel, the increasing prominence of AI in diverse
2

sectors has prompted curricular innovations beyond traditional
computer science programs. Xu and Babaian address the
challenges of delivering AI education to interdisciplinary and
non-technical audiences by proposing a graduate-level curricu-
lum that balances foundational AI knowledge with cutting-
edge developments. Their study provides actionable guidelines
for curriculum design, emphasizing the need for pedagogical
approaches that engage students with varying backgrounds
[19].
Taken together, these works collectively argue for a cur-
riculum that incorporates GPU programming and HPC as part
of AI education, leveraging cloud-based platforms such as
AWS SageMaker. The integration of GPU-accelerated cloud
resources in academic training provides students with relevant,
hands-on experience essential for navigating AI workflows
in contemporary computational environments. Such curricular
developments are vital to align educational outcomes with the
technological demands of industry and research, ensuring that
graduates are equipped to contribute effectively to AI-driven
innovation.
III. COURSESTRUCTURE
This course plunges students into the fundamental concepts
of accelerators and empowers them to leverage these powerful
tools for creating sophisticated AI workflows and automated
AI agents. Crucially, it integrates modules that introduce core
HPC concepts, laying a robust foundation for modern AI
development. The course is flexibly offered across the fall,
spring, and summer semesters, contingent on enrollment. Each
weekly 120-minute session is dynamically split: the initial
half is dedicated to module lectures, while the subsequent
half transforms into an immersive, hands-on lab. For spring
and summer iterations, the course adopts a hybrid model,
featuring optional in-person lab sessions to accommodate
diverse student needs. This interdisciplinary offering caters
to undergraduate seniors from Data Analytics, Information
Systems, and Public Health Informatics programs, as well
as graduate students in Information Systems. Designed as a
comprehensive 16-week journey during the standard academic
year, it condenses into an intensive four-week program during
the summer. The detailed evaluation components are discussed
in section IV . Figure 1 showcases student enrollment across
Fall 2024, Spring, and Summer 2025, the Spring 2025 iteration
notably saw fifteen graduate students enroll. For the scope
of this paper, we will exclusively analyze course evaluation
and assessment results from the Fall 2024 and Spring 2025
sessions, as the Summer 2025 session is currently ongoing.
A. Infrastructure Setup
One of the cornerstones of our pedagogical approach was
providing students with unparalleled hands-on access to AWS
cloud resources. Each student was assigned a dedicated Iden-
tity and Access Management (IAM) role, empowering them to
independently launch instances, configure NVIDIA GPUs, and
provision the necessary computational power for their labs and
assignments. We further augmented this practical experience
Spring 2024 Fall 2024 Spring 2025 Summer 2025
Term051015202530EnrollmentTotal: 10Total: 9Total: 31
Total: 10Enrollment per Term (Graduate vs. Undergraduate)
Undergraduate
GraduateFig. 1: Enrollment per Term (Graduate vs. Undergraduate)
by leveraging AWS Educate resources, training students not
only on core AWS services but also on the intricacies of AWS
SageMaker and spinning up new instances. To ensure respon-
sible resource utilization, each student’s usage was capped for
all assessments, complemented by automated scripts designed
to terminate idle resources. Our current budget allows us to
provision ample AWS resources until July 2026, ensuring
sustained access. For efficient management and monitoring of
unused instances, all GPU instances are provisioned within the
US East (N. Virginia) region. Students were provided with a
bootstrap script that simplified resource configuration using
their AWS credentials for each assessment.
1) AWS Cost:The cost efficiency of this approach is signif-
icant: the average on-demand cost for a single-GPU instance
was approximately $1.262 per student per hour, while multi-
GPU instances (up to 3) averaged about $2.314 per student
per hour. Over the entire semester, students typically spent
around 40-45 hours utilizing AWS resources (including GPUs,
AWS SageMaker, and VPC) for their assessments, translating
to an average cost of roughly $50-60 per student for the entire
semester. Appendix A provides an overview of average AWS
GPU usage and cost during Fall 2024 and Spring 2025. For
certain assessments, we strategically utilized AWS Educate
resources, which are provided free of charge by AWS, further
enhancing cost-effectiveness and accessibility. Furthermore,
we allowed students to request additional resources, capped
at $100 per student for the semester; remarkably, no one
found it necessary to request additional funds, highlighting the
sufficiency of the allocated resources. While we also explored
Google Colab, a platform frequently discussed in academic
literature [18], its limitation to a single GPU per session made
AWS a superior choice for the diverse and demanding multi-
GPU AI workloads central to our curriculum.
B. Modules
Our primary aim is to immerse students in GPU program-
ming concepts (GPC), enabling them to confidently navigate
and utilize various powerful libraries such as RAPIDS for
GPU-accelerated data science and Dask for scalable comput-
ing. Students are expected to gain a profound understanding
of how GPU architecture fundamentally differs from CPU
architecture, exploring into the core components of GPUs like
Kernels, blocks, grids, and threads. Each course module was
3

developed using a variety of resources, including academic
peer-reviewed papers and official documentation from libraries
such as Nvidia [13], Dask [20], and PyTorch [21]. The inclu-
sion of academic papers as course materials was inspired by
curriculum development articles on High-Performance Com-
puting [22], [23]. To support student learning, we provided
access to lecture slides and supplementary reference materials
for each module. The textbook [24] was used to introduce
students to the fundamental concepts of GPU programming
and architecture.
Crucially, all labs, assignments, and exams require students
to employ a Python-based interface for GPU programming,
rather than native CUDA (C/C++). Our objective is to famil-
iarize them with the architecture and empower them to create
sophisticated AI workflows that are heavily supported by the
rich ecosystem of Python libraries. While we acknowledge the
inherent performance compromises of not using native CUDA
for this course, our pedagogical focus remains on accessibility
and practical application for this specific student demographic.
Table I outlines the modules covered during the 16-week
course, alongside their corresponding student learning out-
comes, and the deliverables for that week, which includes a
weekly quiz for each module except week 7, and the final
exam.
Students engaged in the development of several distributed
GPU programs, prominently featuring the partitioning of large-
scale, real-world networks such as PubMed [25] and Reddit
[26]. This foundational step enabled the subsequent execution
of Graph Convolutional Network (GCN) [27] training for node
label prediction across multiple GPUs.
Algorithm 1Distributed GCN Training Using METIS Parti-
tioning and Dask
Require:Undirected graphG= (V, E)
Require:Node features matrixX∈R|V|×d
Require:Label vectorY∈R|V|
Require:GCN modelMwith parametersθ
Require:Loss functionLand optimizerO
Require:Number of partitionsk, training epochsN epochs
Ensure:Trained model parametersθ
1:procedureTRAINDISTRIBUTEDGCN(G, X, Y, M,L,O, k, N epochs )
2: LoadG,X, andY; compute normalized adjacency matrix ˜A
3: PartitionGinto{G 1, . . . ,G k}using METIS
4: Initialize Dask cluster; assign each worker to a GPU
5:fori= 1tokdo
6: DistributeG i,Xi,Yito workeri
7: Initialize global modelM global with parametersθ
8: Broadcastθto all workers
9:forepoch= 1toN epochs do
10:fori= 1tokdo
11: On workeri: compute local loss and gradients
12: Aggregate gradients from all workers
13: Update global model parametersθusingO
14: Report epoch loss
15:returnTrained parametersθ
Algorithm 1 illustrates a classic node label problem. Here,
a GCN model processes node features,X∈R|V|×d, and
the graph’s structural information, typically represented by its
adjacency matrix, derived from the graphG= (V, E). The
model’s objective is to predict labels for nodes,Y∈R|V|. Akey strategy for infusing parallel processing into this workflow
involves distributing the graph across multiple GPUs, allowing
each device to train on a distinct subgraph. As delineated in
Algorithm 1, line 3, the graph is initially partitioned using
the highly efficient METIS algorithm [28]. Subsequently, lines
5 and 6 detail the distribution of these subgraphs across the
available GPUs, facilitating parallel training (lines 9-10). We
specifically leveraged the Dask framework [4] for orchestrating
this distributed training process. Within this parallel paradigm
(lines 10-12), each GPU worker independently processes its
assigned partition, computes the local loss by comparing
predicted node labels with ground truth, and calculates gra-
dients. These local gradients are then efficiently combined,
and a global optimizer updates the model’s parameters in a
synchronized fashion.
While students were encouraged to explore and devise their
own orchestration methods for graph splitting and parallel
training, our work illuminated several interesting challenges
inherent in distributed graph processing. Furthermore, to foster
a deeper understanding of partitioning effects, students were
encouraged to experiment with random graph partitioning as
an alternative to METIS [28] and to thoroughly analyze the
resulting GPU utilization patterns.
Parallelizing large-scale, often dense graphs presents sig-
nificant difficulties. In most cases, we observed that simply
splitting the graph and distributing the training yielded min-
imal performance improvement. However, a notable outcome
was the enhanced prediction accuracy scores after splitting and
training, particularly when compared to sequential approaches.
IV. EVALUATIONINSTRUMENTS ANDRESULTS
We utilize a multifaceted assessment approach, incorporat-
ing students’ grades, their end-of-semester course evaluations,
the comprehensive course project, and two anonymous surveys
(mid- and post-course) to thoroughly gauge the effectiveness
of our course modules and pedagogy.
A. Student Performance Assessment
Our rigorous assessment framework includes twelve to
fourteen dynamic in-class labs (flexibly adjusted to class
pace), four challenging assignments, and a collaborative group
project (capped at two members). The project that constitutes
15% of final grade. We also evaluate attendance and class par-
ticipation. Class participation is assessed through the submis-
sion of scribed notes for each lecture, along with a thoughtful
question related to the module covered. Both the notes and
the question are due immediately after class. This participation
component is complemented by two comprehensive exams, a
midterm and a final.
While all grading activities contribute to the final grade,
their weighting is intentionally uneven. For labs and assign-
ments, students will have direct access to both the instructor
and the Teaching Assistant (TA), fostering a supportive learn-
ing environment where assistance is readily available. These
highly interactive activities collectively constitute half of final
grade, emphasizing hands-on application. The remaining half
4

TABLE I: Course Modules, Student Learning Outcomes (SLOs), and Deliverables
Week / Topic Student Learning Outcome (SLO) Deliverable
Week 1: A WS GPU Setup + Course
IntroductionApply: Set up AWS EC2 GPU instances and configure Python environ-
mentsLab 1: AWS GPU instance setup with
Jupyter and SSH access
Week 2: CUDA Fundamentals &
GPU ParallelismUnderstand/Apply: Explain GPU architecture, grasp CUDA program-
ming basics, and implement parallel executionLab 2: CuPy vector/matrix operations &
parallel processing
Week 3: Memory Management &
GPU OptimizationAnalyze/Optimize: Manage and optimize memory transfers between
host and GPULab 3: Matrix multiplication with mem-
ory profiling using Numba
Assignment 1: GPU Matrix Multiplica-
tion and Profiling (Due Week 5)
Week 4: GPU Profiling Tools & Bot-
tleneck AnalysisAnalyze/Evaluate: Apply Nsight Systems, PyTorch profiler, and cPro-
file for comprehensive GPU workload analysisLab 4: Profiling GPU RL loop with
Nsight and PyTorch profiler
Assignment 2: Distributed GPU Data
Processing (Due Week 7)
Week 5: Custom CUDA Kernels with
PythonCreate/Integrate: Write, compile, and seamlessly integrate custom
CUDA kernels in Python workflowsLab 5: Custom CUDA kernel with
Numba + profiling
Week 6: RAPIDS + Dask for Scalable
Data PipelinesApply/Create: Process large datasets efficiently using RAPIDS cuDF
and Dask for distributed GPU workflowsLab 6: Parallel data processing using
Dask with RAPIDS cuDF
Week 7: Midterm Exam / Assessment No SLO (Assessment Week) Midterm Exam; Assignment 2 Due
Week 8: Deep Learning on GPUs
(PyTorch Focus)Apply/Optimize: Train and optimize neural networks using GPU ac-
celeration, specifically focusing on GCNsLab 7: CNN model training on GPU
using PyTorch
Week 9: Reinforcement Learning on
GPUsDevelop/Implement: Develop reinforcement learning agents acceler-
ated by GPUsLab 8: DQN agent training using
CUDA-enabled PyTorch
Week 10: Multi-GPU Training & Par-
allel StrategiesApply/Scale: Scale models efficiently using multi-GPU setups with
Distributed Data Parallel (DDP)Lab 9: PyTorch DDP implementation
across 2 GPUs
Week 11: AI Agent Foundations &
GPU BenefitsUnderstand/Describe: Describe AI agents and explain the GPU’s
critical role in training accelerationLab 10: Simple reinforcement agent us-
ing CuPy/Numba
Assignment 3: Multi-GPU AI Agent
(Due Week 13)
Week 12: Retrieval-Augmented Gen-
eration (RAG) BasicsUnderstand/Describe: Describe RAG architectures, combining re-
trieval and generation modules effectivelyLab 11: Basic RAG pipeline using
FAISS for retrieval
Week 13: GPU-Optimized RAG De-
velopmentConstruct/Optimize: Construct and optimize RAG models using GPU-
accelerated retrievers and generatorsLab 12: Build GPU-enabled RAG with
retriever + small LLM
Week 14: RAG Pipeline Optimization
& InferenceOptimize/Deploy: Optimize end-to-end RAG pipelines for efficient
real-time GPU inferenceLab 13: Deploy real-time RAG infer-
ence pipeline
Assignment 4: End-to-End RAG Sys-
tem (Due Week 16)
Week 15: Project Development &
SupportApply/Create: Apply GPU acceleration, AI agent techniques, and RAG
models in capstone projectsLab 14: Build your own Lab (Extra
Credit); Academic paper review (Extra
Credit)
Week 16: Final Project Presentations
& ExamShowcase/Demonstrate: Showcase final projects demonstrating GPU-
accelerated AI/RAG pipelinesFinal Project Presentation
comes from independent endeavors, including the assignments
and closed-book exams, designed to assess individual mastery,
along with the group project.
Appendix B lists the extra credit opportunities available
to students. Figure 2 visually depicts the grade distribution
across Fall 2024 and Spring 2025. In Fall 2024, the majority
of students achieved a ’B’ grade, with many encountering
difficulties with post-midterm modules. Despite receiving as-
sistance from the TA and instructor, a number of students were
only able to submit partial assignments, suggesting challenges
in independent completion. In stark contrast, Spring 2025 wit-
nessed a significant uplift, with over 60% of students securing
an ’A’ grade and demonstrating enhanced independence in
completing assignments. This marked improvement is directly
correlated with proactive revisions to our lab instructions,
meticulously designed to better prepare students for subse-
quent assignments. We further observed that a substantial ma-
jority of students proactively leveraged office hours throughout
the semester, not only to clarify complex assignments but
also to request crucial code reviews prior to submission.
Conversely, students earning a ’B’ or lower typically corre-lated with missed submissions or late lab/assignment turn-
ins. The exam average remained remarkably consistent across
both semesters, hovering between 75-80%. This reflects a
solid foundational understanding among the students. (See
Appendix C for further data analysis and findings).
A
33.3%B
66.7%
Total Students: 9Fall 2024
A
61.3%B
19.4%C
12.9%D
3.2%
F
3.2%
Total Students: 31Spring 2025
Fig. 2: Grade Distribution for Fall 2024 and Spring 2025
Course Offerings
5

B. End-of-Semester Student Evaluations
A robust 85% of students completed the anonymous online
evaluation form, providing valuable feedback on the course
using questions presented in Table II. All questions, which
are standard for all courses and designed by the university,
utilized a five-point Likert scale with response options in-
cluding ”Always,” ”Often,” ”Sometimes,” ”Seldom,” ”Never,”
and ”N/A.” Figure 3 provides an overview of the student
TABLE II: End-of-Semester Course Assessment Questions
Evaluation Questions
The course information further developed my knowledge in this area.
The course activities enhanced my learning of the course content.
The oral assignments improved my presentation skills.
The course activities improved my computer technology skills.
Lab or clinical experiences contributed to my understanding of the course
theories and concepts.
The instructor clearly explained laboratory or clinical experiments or
procedures.
feedback regarding course content, with the X-axis repre-
senting the percentage of responses for each type and the
Y-axis representing the questions from Table II. Figure 3
suggests that both undergraduate and graduate courses are
largely successful in delivering positive learning experiences.
However, there appear to be subtle differences in the perceived
strengths, with undergraduates valuing core course content
and graduates finding more significant gains in specific skill
development. Further investigation into the methodologies
and learning objectives of these course levels could provide
deeper insights into these observed differences. The ”Seldom,”
”Never,” and ”N/A” categories consistently represent a small
minority of responses, indicating that strongly negative or irrel-
evant feedback is not prevalent. For both groups, ”Lab/Clinical
Experiences Contributed to Understanding” and ”Instructor
Clearly Explained Lab/Clinical Procedures” tend to have lower
”Always” percentages compared to the course content aspects,
suggesting these might be areas for instructors or course
designers to explore for enhancement. We plan to experiment
in Fall 2025 by revising our labs and offering a more clear
explanation about labs and assignments during class sessions
and to explicitly highlight how assignments serve as extensions
of the lab experiences. Although the survey question wording
is standardized across disciplines, in this course “lab/clinical
procedures” specifically refers to programming and computa-
tional tasks rather than healthcare contexts. (See Appendix D
for further data analysis on student course satisfaction.)
C. Anonymous Surveys
During the Fall 2024 and Spring 2025 semesters, students
were invited to provide anonymous feedback. This feedback
was crucial for the department to evaluate existing course
offerings and strategically plan for future similar courses,
particularly concerning GPU programming content and peda-
gogical approaches. Feedback was systematically collected via
Google Forms at two key points: during the sixth week (prior
to the midterm exam) and the twelfth week (before studentsbegan their group projects). The primary objective of these
two surveys was to ascertain the extent to which students
had grasped the course material. Each survey question utilized
a five-point Likert scale. Participation in the surveys was
robust, with most students completing them. The mid-course
survey featured three questions assessing their confidence in
using Numba, building and configuring GPUs on AWS, and
utilizing HPC tools for profiling. The final survey repeated
these questions and additionally included a question on their
confidence level in using multiple GPUs for parallel algorithm
implementation.
Figure 4a compares student self-assessment of their abil-
ity to use Numba to implement a parallel algorithm using
CUDA across two semesters. In Fall 2024, responses were
relatively evenly distributed: two students strongly disagreed,
two disagreed, one was neutral, two agreed, and two strongly
agreed. Conversely, Spring 2025 shows a greater proportion
of students in the ’Neutral’ to ’Strongly Agree’ categories.
Specifically, nine students were neutral, seven agreed, and five
strongly agreed, making ’Neutral’ the largest single response
group. This suggests that while many students felt capable,
a significant portion remained in the middle, indicating they
felt neither fully proficient nor entirely lacking in this specific
skill.
Figure 4b illustrates student confidence level in building
and configuring GPU clusters. In Fall 2024, students initially
expressed weak confidence during the midterm. We believe
this stemmed from initial challenges in configuring GPUs
and ensuring instances were correctly connected within the
same Virtual Private Cloud (VPC) with appropriate subnet
addresses. By the final survey, confidence improved; we at-
tribute this to the automation of instance setup and networking,
although some students still faced challenges in updating and
executing these scripts correctly.
In Spring 2025, midterm confidence was mixed, with ap-
proximately twelve students expressing disagreement, eight
remaining neutral, and eleven showing agreement. By the final
survey, confidence had substantially improved, with students
reporting strong confidence. We posit this increase in confi-
dence resulted from the reinforcement of cluster creation tasks
throughout all in-class activities.
Figure 4c illustrates student confidence level in utilizing
PyTorch Profiler [21] and Nsight Systems [29] for GPU
profiling. In Fall 2024, students initially expressed strong
confidence during the midterm assessment; however, a clear
reduction in confidence was observed in the final survey
regarding their proficiency with GPU profiling tools. We
suggest this decline may be attributed to the increased inde-
pendent application of these profilers for project work after
the midterm. A similar trend was observed in Spring 2025;
however, the magnitude of the confidence dip (from ”agree”
to ”disagree”) was less pronounced compared to Fall 2024.
This attenuated decline in Spring 2025 might be a consequence
of incorporating additional hands-on profiling activities during
in-class sessions. Figure 4d presents final survey results on
student confidence in applying multi-GPU training and parallel
6

Fig. 3: Student Feedback on Course Content and Lab/Clinical Experiences
computing for AI models such as GCN. Fall 2024 responses,
from a small group of students, were largely positive. Spring
2025 results, from a larger group, were more varied, with ten
students expressing disagreement while most reported neutral
or higher confidence. The moderate confidence may reflect the
assignment’s difficulty, particularly issues installing PyTorch
and configuring Dask (Algorithm 1).
V. CONCLUSIONS
Despite the rapid advancements in AI and the growing
accessibility of GPU technologies, integrating GPU program-
ming and HPC concepts into the STEM curriculum remains
a significant challenge. We contend that the most effective
pedagogical approach to bridge this gap involves hands-on
laboratory exercises and assessments that not only impart
technical skills but also cultivate critical thinking and problem-
solving abilities among STEM students.
We successfully introduced GPU programming. Our expe-
rience in delivering these special topics courses, as detailed in
this paper, has been overwhelmingly positive. These offerings
have provided invaluable insights into student engagement and
learning outcomes.
Looking ahead, we plan to refine and expand our lab
modules and assessments based on these experiences. We are
actively exploring various ways to integrate specific course
modules into other Data Analytics offerings. To better pre-
pare students for these advanced topics, we are considering
a revision of the prerequisite course to infuse foundationalHPC concepts alongside traditional sequential programming.
Furthermore, we intend to explore alternative pedagogical
strategies and continuously evolve the curriculum to align with
the AI and HPC competencies currently in high demand in the
job market. Our overarching goal is to offer this course more
frequently, making it accessible to a broader and more diverse
STEM student population.
REFERENCES
[1] N. Corporation, “The state of ai infrastructure: Insights from
industry leaders,” 2023, white Paper. [Online]. Available: https:
//resources.nvidia.com/en-us-ai-infrastructure/ai-infrastructure-report
[2] Alooba, “Everything you need to know when assessing parallel
computing skills,”Alooba Blog, 2024, accessed: [Current Date, e.g.,
July 21, 2025]. [Online]. Available: https://www.alooba.com/skills/conc
epts/devops/parallel-computing/
[3] U.S. Bureau of Labor Statistics. (2024) Data Scientists : Occupational
Outlook Handbook.
[4] M. Rocklin, “Dask: Parallel computation with blocked algorithms
and task scheduling,”Proceedings of the 14th Python in Science
Conference (SciPy), pp. 130–136, 2015. [Online]. Available: https:
//conference.scipy.org/proceedings/scipy2015/pdfs/matthew rocklin.pdf
[5] National Science Foundation, “Cap: Expanding use-inspired ai research
and instruction across campus,” NSF Award #2401658, 2024, award
abstract retrieved from NSF Award Search.
[6] M. Cecchini, R. Ramaswamy, M. Bruenger, G. Hager, and G. Wellein,
“Implementation and evaluation of cuda-unified memory in numba,” in
Euro-Par 2020: Parallel Processing Workshops. Springer, 2020, pp.
120–133.
[7] M. Bruenger, M. Cecchini, G. Hager, and G. Wellein, “Lessons learned
from comparing c-cuda and python-numba for gpu-computing,” in
Workshop on Python for High-Performance and Scientific Computing
(PyHPC), 2020.
7

Strongly DisagreeDisagreeNeutralAgree
Strongly Agree0.02.55.07.510.012.5Number of Students0124
2 2
1
05
1Fall 2024: 9 Students
Mid
Final
Strongly DisagreeDisagreeNeutralAgree
Strongly Agree68
6
5610
8
5 5
3Spring 2025: 31 Students
Mid
FinalI feel confident in using Numba for Parallel CUDA Algorithms
Likert Scale(a) Numba-CUDA Parallel Programming Ability
Strongly DisagreeDisagreeNeutralAgree
Strongly Agree0510Number of Students7
0101 13
12 2Fall 2024: 9 Students
Mid
Final
Strongly DisagreeDisagreeNeutralAgree
Strongly Agree578 8
3 36
4711Spring 2025: 31 Students
Mid
FinalI feel confident in Building & Configuring AWS GPU Clusters
Likert Scale
(b) Confidence in Using AWS GPU Cluster
Strongly DisagreeDisagreeNeutralAgree
Strongly Agree0510Number of Students14
1 125
021 1Fall 2024: 9 Students
Mid
Final
Strongly DisagreeDisagreeNeutralAgree
Strongly Agree27
5611
6
47 7 7Spring 2025: 31 Students
Mid
FinalI feel confident in using PyTorch Profiler and Nsight Systems for GPU Profiling
Likert Scale
(c) Python/PyTorch and Nsight GPU Profiling Confi-
dence
Strongly DisagreeDisagreeNeutralAgree
Strongly Agree0246810Number of Students023
13Fall 2024: 9 Students
Final
Strongly DisagreeDisagreeNeutralAgree
Strongly Agree5 57 7 7Spring 2025: 31 Students
FinalI feel confident in applying multi-GPU training and parallel computing strategies to accelerate AI model development
Likert Scale
(d) Multi-GPU Programming Confidence
Fig. 4: Student survey results (Fall 2024 / Spring 2025)
[8] S. K. Lam, A. Pitrou, and S. Seibert, “Numba: A dynamic python
compiler for numerical functions,”Proceedings of the Second Workshop
on the LLVM Compiler Infrastructure in HPC, pp. 1–6, 2015. [Online].
Available: https://dl.acm.org/doi/abs/10.1145/2833157.2833162
[9] P. Holm, M. Bruenger, and G. Wellein, “Evaluating numba and cupy for
gpu-accelerated monte carlo transport,”Mathematics, vol. 11, no. 3, p.
612, 2023.
[10] S. K. Prasad, A. Chtchelkanova, S. Das, F. Dehne, M. Gouda, A. Gupta,
J. Jaja, K. Kant, A. La Salle, R. LeBlancet al., “Nsf/ieee-tcpp curriculum
initiative on parallel and distributed computing: core topics for under-
graduates,” inProceedings of the 42nd ACM technical symposium on
Computer science education, 2011, pp. 617–618.
[11] J. Chen, S. Ghafoor, and J. Impagliazzo, “Producing competent hpc
graduates,”Communications of the ACM, vol. 65, no. 12, pp. 56–65,
2022.
[12] M. Arroyo, “Teaching parallel and distributed computing to undergrad-
uate computer science students,” pp. 1297–1303, 2013.
[13] “Hands-on gpu-accelerated training in ai, hpc, and workflow automa-
tion,” https://www.nvidia.com/en-us/training/, accessed: July 19, 2025.
[14] A. Qasemet al., “Lightning talks of eduhpc 2022,” in2022 IEEE/ACM
International Workshop on Education for High Performance Computing
(EduHPC). IEEE, 2022, pp. 42–49.
[15] J. Adams, B. Hainey, L. White, D. Foster, N. Hall, M. Hills, C. Taglienti
et al., “Cloud computing curriculum: Developing exemplar modules for
general course inclusion,” inProceedings of the Working Group Reportson Innovation and Technology in Computer Science Education, 2020,
pp. 151–172.
[16] B. Neelima and J. Li, “Introducing high performance computing con-
cepts into engineering undergraduate curriculum: a success story,” in
Proceedings of the Workshop on Education for High-Performance Com-
puting, 2015, pp. 1–8.
[17] A. Qasem, D. P. Bunde, and P. Schielke, “A module-based introduction
to heterogeneous computing in core courses,”Journal of Parallel and
Distributed Computing, vol. 158, pp. 56–66, 2021.
[18] Z. Xu, “Teaching heterogeneous and parallel computing with google
colab and raspberry pi clusters,” inProceedings of the SC’23 Workshops
of The International Conference on High Performance Computing,
Network, Storage, and Analysis, 2023, pp. 308–313.
[19] J. J. Xu and T. Babaian, “Artificial intelligence in business curriculum:
The pedagogy and learning outcomes,”The International Journal of
Management Education, vol. 19, no. 3, p. 100550, 2021.
[20] “Dask tutorial,” https://tutorial.dask.org/, 2024, accessed: [Insert Access
Date].
[21] PyTorch Contributors. (2024) Profiling your PyTorch Module.
[22] S. Neuwirth, “Bridging the gap between education and research: A
retrospective on simulating an hpc conference,” inProceedings of the
Workshop on Education for High-Performance Computing (EduHPC).
IEEE, 2023.
[23] N. Kremer-Herman, “Challenges and triumphs teaching distributed
computing topics at a small liberal arts college,” inProceedings of the
Workshop on Education for High-Performance Computing (EduHPC).
IEEE, 2023.
[24] M. Dias and M. G. Dias,Hands-On GPU Programming with Python
and CUDA: Explore high-performance parallel computing with CUDA.
Packt Publishing Ltd, 2019.
[25] P. Sen, G. Namata, M. Bilgic, L. Getoor, B. Gallagher, and T. Eliassi-
Rad, “Collective classification in network data,” Carnegie Mellon
University, Tech. Rep., 2008, technical Report. [Online]. Available:
https://linqs.soe.ucsc.edu/data
[26] W. L. Hamilton, R. Ying, and J. Leskovec, “Inductive representation
learning on large graphs,” inAdvances in Neural Information
Processing Systems (NeurIPS), 2017, pp. 1024–1034. [Online].
Available: https://arxiv.org/abs/1706.02216
[27] T. N. Kipf and M. Welling, “Semi-supervised classification with
graph convolutional networks,” inInternational Conference on
Learning Representations (ICLR), 2017. [Online]. Available: https:
//arxiv.org/abs/1609.02907
[28] G. Karypis and V . Kumar,METIS–A Software Package for Partitioning
Unstructured Graphs, Partitioning Meshes, and Computing Fill-
Reducing Orderings of Sparse Matrices, University of Minnesota,
Department of Computer Science and Engineering, Minneapolis, MN,
1998, version 4.0. [Online]. Available: http://glaros.dtc.umn.edu/gkhom
e/metis/metis/overview
[29] NVIDIA Corporation. (2025) NVIDIA Nsight Systems.
8

APPENDIXA
AWS GPU COMPUTATIONALHOURS
Fall 2024 Spring 2025
Semester02004006008001000120014001600Total Computational Hours376.9 hrs
($142.83)
(9 students, 41.9 avg hrs/student)1387.4 hrs
($525.81)
(31 students, 44.8 avg hrs/student)GPU Computational Hours per Semester
Fig. 5: Average AWS GPU usage and cost for Fall 2024, and
Spring 2025.
Figure 5 provides an estimated average computational hours
of GPU instances for all labs and assignments in Fall 2024 and
Spring 2025. The notable increase in average hours per student
during Spring 2025 is due to the introduction of two additional
labs. For group projects, the average usage of GPU resources
was less than 2 hours in both semesters. We did not include
the computational hours of GPU instances from AWS Educate
because the instructor lacks access to resource usage insights
for that platform.
APPENDIXB
EXTRACREDIT
Two distinct extra credit opportunities were offered to
students.
1. Build Your Own Lab
Students were encouraged to design their own lab based
on the course modules. The only requirement was that the
lab could not replicate an existing one. The primary goal
of this initiative was to assess students’ ability to apply
core concepts and design a lab from scratch. A secondary
objective was to identify potential labs that could be rigorously
tested during the summer of 2025 and possibly integrated
into future semesters. No students attempted this extra credit
in Fall 2024. In Spring 2025, three students submitted the
lab; however, none of the submissions fully met the student
learning outcomes. We believe this may be due to students at-
tempting the design and preparation during finals week, which
prevented them from dedicating sufficient time to research and
development.
2. Academic Paper Review
In Spring 2025, an extra credit opportunity was offered
where students could select a peer-reviewed academic paper
published between 2020 and 2025 based on the modules
discussed. They were asked to provide a one-page summary,
discuss its shortcomings, and propose an extension of the
research. The goal was to help students understand the current
research landscape in their chosen modules. The inclusion of
this extra credit was inspired by an article [22]; where attemptwas made to bridge the gap between education and research.
Approximately 60% of students completed this activity. While
most provided excellent summaries, their explanations for
expanding on the proposed research were often vague. Based
on this outcome, we plan to explore integrating academic
papers directly into course modules to better promote student
research skills.
APPENDIXC
DATAANALYSIS ANDFINDINGS
We evaluated potential differences in academic performance
between graduate and undergraduate students enrolled in a
special topics course, specifically focused on GPU program-
ming, offered in Fall 2024 and Spring 2025. To assess this,
we tested the following hypotheses:
•Null Hypothesis (H 0): There is no difference in av-
erage performance between graduate and undergraduate
students.
•Alternative Hypothesis (H 1): There is a difference in
average performance between graduate and undergradu-
ate students.
As we did not specify a direction for the difference, a
two-tailed test was employed to assess whether the observed
performance differences reflected random variation or a sys-
tematic distinction between the two cohorts. Before conducting
hypothesis testing, we evaluated whether the data satisfied the
assumptions required for parametric tests—namely, normality
and homogeneity of variance.
Figure 6 displays histograms, while Figure 7 and Figure 8
present Q–Q plots. These visualizations indicate clear de-
partures from normality, particularly in the graduate group,
whose scores were tightly clustered near the upper end of the
distribution and exhibited noticeable skewness.
60 70 80 90 100
Score (%)0.000.020.040.060.080.10DensityHistogram and KDE
Graduate Undergraduate
Fig. 6: Histogram comparison of academic scores between
graduate and undergraduate student groups.
To statistically assess normality, we conducted Shapiro–
Wilk tests. The graduate group showed a substantial de-
viation from normality (W= 0.722,p < .001), while
the undergraduate group also deviated, though less severely
(W= 0.898,p=.037). In contrast, Levene’s test for equality
of variances revealed no significant difference between the
9

two groups (F= 2.437,p=.127), suggesting that the
assumption of homogeneity of variance was met. These results
are summarized in Table III.
TABLE III: Results of Assumption Tests for Normality and
Homogeneity of Variance
Assumption Test Statistic p-value
Shapiro–Wilk (Graduate) 0.722< .001
Shapiro–Wilk (Undergraduate) 0.898 .037
Levene’s Test 2.437 .127
In addition to assumption testing, descriptive statistics pro-
vided insight into the distribution and central tendency of
scores in each group, as shown in Table IV. Graduate students
achieved remarkably higher mean scores and demonstrated
a more compact distribution, reflected in lower variability
compared to undergraduates.
2
 1
 0 1 2
Theoretical Quantiles60708090100Sample Quantiles
Q-Q Plot: Undergraduate
Fig. 7: Q–Q plots of academic
scores for undergraduate stu-
dents.
2
 1
 0 1 2
Theoretical Quantiles7580859095100105Sample Quantiles
Q-Q Plot: GraduateFig. 8: Q–Q plots of academic
scores for graduate students.
TABLE IV: Descriptive statistics for academic performance
scores by group
Group Mean Std Dev Min Q1 Median Q3 Max Count
Graduate 94.36 6.91 74.38 90.06 97.92 98.80 99.17 20
Undergraduate 83.51 11.33 53.75 80.79 85.94 91.05 98.54 20
Although the variances were homogeneous, the pronounced
non-normality, especially in the graduate scores, made para-
metric tests (e.g., independent samplest-test) inappropriate.
Given the relatively small sample sizes (n= 20per group) and
non-normal distributions, we employed the Mann–WhitneyU
test, a non-parametric alternative suitable for comparing cen-
tral tendencies between two independent groups. The Mann–
WhitneyUtest revealed a statistically significant difference in
performance between the two groups, with graduate students
outperforming undergraduates (U= 332.00,p=.0004).
Figure 9 which show a higher median and a more compact
score distribution among graduate students compared to un-
dergraduates.
In conclusion, the data provide strong evidence to reject the
null hypothesis. Graduate students performed significantly bet-
ter than undergraduate students in terms of their weighted total
scores. By testing assumptions and applying a suitable non-
parametric method, we ensured the validity of our findings and
/uni0000002a/uni00000055/uni00000044/uni00000047/uni00000058/uni00000044/uni00000057/uni00000048 /uni00000038/uni00000051/uni00000047/uni00000048/uni00000055/uni0000004a/uni00000055/uni00000044/uni00000047/uni00000058/uni00000044/uni00000057/uni00000048
/uni00000036/uni00000057/uni00000058/uni00000047/uni00000048/uni00000051/uni00000057/uni00000003/uni0000002a/uni00000055/uni00000052/uni00000058/uni00000053/uni00000019/uni00000013/uni0000001a/uni00000013/uni0000001b/uni00000013/uni0000001c/uni00000013/uni00000014/uni00000013/uni00000013/uni00000036/uni00000046/uni00000052/uni00000055/uni00000048/uni00000003/uni0000000b/uni00000008/uni0000000c
/uni00000036/uni00000046/uni00000052/uni00000055/uni00000048/uni00000003/uni00000027/uni0000004c/uni00000056/uni00000057/uni00000055/uni0000004c/uni00000045/uni00000058/uni00000057/uni0000004c/uni00000052/uni00000051/uni00000003/uni00000045/uni0000005c/uni00000003/uni0000002a/uni00000055/uni00000052/uni00000058/uni00000053/uni00000003/uni0000000b/uni00000025/uni00000052/uni0000005b/uni00000003/uni00000033/uni0000004f/uni00000052/uni00000057/uni0000000c
/uni0000002a/uni00000055/uni00000044/uni00000047/uni00000058/uni00000044/uni00000057/uni00000048 /uni00000038/uni00000051/uni00000047/uni00000048/uni00000055/uni0000004a/uni00000055/uni00000044/uni00000047/uni00000058/uni00000044/uni00000057/uni00000048
/uni00000036/uni00000057/uni00000058/uni00000047/uni00000048/uni00000051/uni00000057/uni00000003/uni0000002a/uni00000055/uni00000052/uni00000058/uni00000053/uni00000019/uni00000013/uni0000001a/uni00000013/uni0000001b/uni00000013/uni0000001c/uni00000013/uni00000014/uni00000013/uni00000013/uni00000036/uni00000046/uni00000052/uni00000055/uni00000048/uni00000003/uni0000000b/uni00000008/uni0000000c
/uni0000002c/uni00000051/uni00000047/uni0000004c/uni00000059/uni0000004c/uni00000047/uni00000058/uni00000044/uni0000004f/uni00000003/uni00000036/uni00000046/uni00000052/uni00000055/uni00000048/uni00000056/uni00000003/uni00000045/uni0000005c/uni00000003/uni0000002a/uni00000055/uni00000052/uni00000058/uni00000053/uni00000003/uni0000000b/uni00000036/uni00000057/uni00000055/uni0000004c/uni00000053/uni00000003/uni00000033/uni0000004f/uni00000052/uni00000057/uni0000000cFig. 9: Boxplot and Stripplot of academic scores for graduate
and undergraduate student groups.
aligned our analytical approach with the data’s distributional
characteristics.
APPENDIXD
DESCRIPTIVESTATISTICS OFSTUDENTSATISFACTION
We analyzed student satisfaction for the GPU programming
course across Fall 2024 and Spring 2025 based on course
evaluations collected anonymously (n= 18) and found overall
positive results with some differences between the semesters.
In Fall 2024, most students (87.5%) reported “Very High”
satisfaction, while one student (12.5%) reported “Very Low”
satisfaction. In Spring 2025, 60% of students rated their
satisfaction as “Very High,” and the remaining 40% rated it
as “High,” with no “Very Low” ratings.
As shown in Figure 10 we counted the number of responses
in each satisfaction category by semester. When we exam-
ined the proportional breakdown in Figure 11 we saw that
Fall 2024’s responses clustered heavily around “Very High,”
whereas Spring 2025 had a more balanced split between “Very
High” and “High.”
Very Low Low High Very High
Satisfaction Level01234567Number of Students 17
46Satisfaction Counts by Semester
Semester
Fall2024 Spring2025
Fig. 10: Bar Plot of Satisfac-
tion Counts by Semester.
Fall2024 Spring2025
Semester0.00.20.40.60.81.0Proportion of Responses12.5%40.0%87.5%60.0%Percentage Satisfaction by Semester
Satisfaction Level
Very Low Low High Very HighFig. 11: Stacked Bar Chart
of Percentage Satisfaction by
Semester.
Overall, we conclude that students were very satisfied with
the GPU programming course in both semesters. While Fall
2024 included an isolated low satisfaction rating, Spring 2025
showed consistently positive and moderately varied satisfac-
tion levels. These findings suggest that the GPU programming
course effectively facilitated the acquisition of valuable new
skills, which contributed to high levels of student satisfaction,
reflecting the course’s successful delivery and instructional
quality.
10