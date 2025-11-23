# Build AI Assistants using Large Language Models and Agents to Enhance the Engineering Education of Biomechanics

**Authors**: Hanzhi Yan, Qin Lu, Xianqiao Wang, Xiaoming Zhai, Tianming Liu, He Li

**Published**: 2025-11-19 05:16:51

**PDF URL**: [https://arxiv.org/pdf/2511.15752v1](https://arxiv.org/pdf/2511.15752v1)

## Abstract
While large language models (LLMs) have demonstrated remarkable versatility across a wide range of general tasks, their effectiveness often diminishes in domain-specific applications due to inherent knowledge gaps. Moreover, their performance typically declines when addressing complex problems that require multi-step reasoning and analysis. In response to these challenges, we propose leveraging both LLMs and AI agents to develop education assistants aimed at enhancing undergraduate learning in biomechanics courses that focus on analyzing the force and moment in the musculoskeletal system of the human body. To achieve our goal, we construct a dual-module framework to enhance LLM performance in biomechanics educational tasks: 1) we apply Retrieval-Augmented Generation (RAG) to improve the specificity and logical consistency of LLM's responses to the conceptual true/false questions; 2) we build a Multi-Agent System (MAS) to solve calculation-oriented problems involving multi-step reasoning and code execution. Specifically, we evaluate the performance of several LLMs, i.e., Qwen-1.0-32B, Qwen-2.5-32B, and Llama-70B, on a biomechanics dataset comprising 100 true/false conceptual questions and problems requiring equation derivation and calculation. Our results demonstrate that RAG significantly enhances the performance and stability of LLMs in answering conceptual questions, surpassing those of vanilla models. On the other hand, the MAS constructed using multiple LLMs demonstrates its ability to perform multi-step reasoning, derive equations, execute code, and generate explainable solutions for tasks that require calculation. These findings demonstrate the potential of applying RAG and MAS to enhance LLM performance for specialized courses in engineering curricula, providing a promising direction for developing intelligent tutoring in engineering education.

## Full Text


<!-- PDF content starts -->

1   
Build AI Assistants using Large Language Models and Agents to 
Enhance the Engineering Education of Biomechanics  
Hanzhi Yan1, Qin Lu1, Xianqiao Wang2, Xiaoming Zhai3, Tianming Liu4, and He Li5* 
1School  of Electrical  and Computer  Engineering,  University  of Georgia,  Athens,  GA 30602  
2School  of Environmental,  Civil,  Agricultural  and Mechanical  Engineering,  University  of 
Georgia, Athens, GA 30602  
3AI4STEM  Education  Center,  University  of Georgia,  Athens,  GA 30602  
5School of  Computing, The  University  of Georgia,  Athens, 30602  
5School  of Chemical,  Materials,  and Biomedical  Engineering,  University  of Georgia, 
Athens, GA 30602  
*he.li3@uga.edu  
 
 
Abstract  
While  large  language  models  (LLMs)  have  demonstrated  remarkable  versatility  across  a wide 
range  of general  tasks,  their  effectiveness  often  diminishes  in domain -specific  applications  due to 
inherent  knowledge  gaps  and the tendency  to generate  hallucinated  or inaccurate  information. 
Moreover,  their  performance  typically  declines  when  addressing  complex  problems  that require 
multi -step reasoning and analysis, which is commonly encountered across many engineering 
disciplines.  In response to these challenges, AI agents, which often incorporate LLMs as core 
components, have  recently  been  introduced  to handle  more  sophisticated  tasks.  In this study, 
we propose leveraging both LLMs and AI agents to develop education assistants aimed at 
enhancing  undergraduate  learning  in biomechanics  courses  that focus  on analyzing  the force  and 
moment  in the musculoskeletal  system  of the human  body. To  achieve  our goal,  we construct  a 
dual -module  framework  to enhance  LLM  performance  in biomechanics  educational  tasks:  1) we 
apply  Retrieval -Augmented  Generation  (RAG)  to improve  the specificity  and logical  consistency 
of LLM’s responses to the conceptual true/false questions; 2) We build a Multi -Agent System 
(MAS) to solve calculation -oriented problems involving multi -step reasoning and code execution. 
Specifically,  we evaluate  the performance  of several  LLMs,  i.e., Qwen -1.0-32B,  Qwen -2.5-32B, 
and Llama -70B, on  a biomechanics dataset comprising 100 true/false conceptual questions and 
problems requiring equation derivation and calculation.  Our results demonstrate that RAG 
significantly  enhances  the performance  and stability  of LLMs  in answering  conceptual  questions, 
surpassing those of vanilla models.  On the other hand, the MAS constructed using multiple 
LLMs  demonstrates  its ability  to perform  multi -step  reasoning,  derive  equations,  execute  code, 
and generate  explainable  solutions  for tasks  that require  calculation.  These  findings  demonstrate 
the potential of applying RAG and MAS to enhance LLM performance for specialized courses 
in engineering  curricula,  providing  a promising  direction  for developing  intelligent  tutoring  in 
engineering education.  
 
1 Introduction  
In recent  years,  the rapid  advancement  of large  language  models  (LLMs)  in natural  language 
processing  has sparked  growing  interest  in their  potential  for domain -specific  applications,  including  

2  knowledge -based  question  answering  [1–7] and educational  assistance  [8–14]. In interdisciplinary 
fields  in engineering,  LLMs  are expected  not only  to comprehend  domain -specific  knowledge  but 
also to accurately complete tasks requiring logical reasoning, such as true/false questions and 
calculation problems that require multi -step analysis.  However, existing LLMs face significant 
challenges  in solving  engineering  applications,  including  insufficient  grasp  of specialized  concepts, 
frequent  hallucination  phenomena, and  a lack of domain -specific  contextual  support, compromising 
their performance in solving engineering problems [ 15–22]. 
One  major  cause of  the suboptimal  performance of  current LLMs  when  used  in advanced  en- 
gineering  or scientific  coursework  is that they  are primarily  pretrained  on broad,  general -purpose 
corpora.  Consequently, these models often lack the necessary domain -specific knowledge to provide 
accurate answers to specialized questions, frequently resulting in inaccurate or general, non -specific 
responses.  Furthermore, the  hallucination  problem, where  the model  generates  responses  that seem 
plausible  but are factually  incorrect,  could  potentially  mislead  students  [23–26]. To address  these 
issues,  Retrieval -Augmented  Generation  (RAG)  has become  a promising  domain -specific  enhance - 
ment  mechanism  [27–34]. The core  idea  of RAG  is to retrieve  relevant  passages  from  structured  or 
unstructured  knowledge  bases  prior  to text generation,  and then  feed  them,  along  with  the original 
query, into the LLM to support knowledge -grounded inference.  This approach aims to provide 
more  domain -knowledge  support,  reduce  hallucinations,  and improve  response  quality.  Although 
RAG has achieved success in domains such as medical [ 35–44] and legal question answering [ 45–51], 
its application  in educational  scenarios,  like answering  conceptual  questions  in specific  engineering 
courses, remains relatively unexplored.  
To this end, in this study, we propose and implement a complete RAG -based system tailored 
for knowledge -enhanced  answering  of true/false  questions  generated  based  on an undergraduate - 
level Biomechanics class, which aims to train students to apply mechanical principles to analyze 
human  posture  and movement.  Understanding  the biomechanics  of the human  body  requires  not 
only knowledge of the functional anatomy and physiology of each body component, but also a 
quantitative  calculation  of forces  and moments  based  on physical  principles  and mechanical  laws 
that govern  the body’s  balance  in various  static  and dynamic  conditions  [52, 53]. Unlike  conceptual 
questions, biomechanics calculation problems typically require the model to possess a broader set of 
capabilities,  including  accurately  interpreting  physical  conditions,  deriving  mathematical  formulas, 
and generating  and executing  computation  code  for the final  solution.  While  the RAG  architecture 
provides effective textual enhancement, it remains insufficient for handling tasks that involve multi - 
step reasoning and quantitative analysis.  
Relatively few studies have applied LLMs to tackle problems involving the derivation of gov - 
erning  equations  and execution  of numerical  calculations  in engineering  education,  which  require 
not only  language  understanding  but also abilities  in physical  reasoning,  equation  derivation,  unit 
conversion, and numerical computation.  For example, to solve a biomechanics calculation problem, 
the LLMs must be able to sequentially accomplish several tasks.  First, it must accurately under - 
stand  the problem,  correctly  identify  static  or dynamic  conditions,  and fully  define  the known  and 
unknown  variables.  Second,  it should  be capable  of deriving  appropriate  equations  from  physical 
laws  based  on the interpretation  of the problems.  Third,  the system  must  be capable  of generating 
and executing code, translating the reasoning process into runnable programs to compute unknown 
variables.  Lastly,  the solution  must  be presented  in a structured  and explainable  format,  including 
formulas,  calculations,  and final  answers  that facilitate  both  instructional  review  and learner  com - 
prehension.  Given  that traditional  LLMs  are primarily  optimized  for textual  reasoning  and natural 
language  generation,  they  tend  to be inconsistent  in code  generation  and struggle  to maintain  log- 
ical continuity  across  multi -step  problems  [54, 55]. Furthermore,  they  lack the ability  to verify  the 
correctness  of intermediate  steps,  which  undermines  their  reliability  in educational  environments  

3  where  both  the problem -solving  processes  and the accuracy  of the solutions  are significant  [56, 57]. 
To address  this limitation,  we propose  and implement  a Multi -Agent  System  (MAS)  [58, 59] 
specifically  designed  for solving  biomechanics  problems  that require  deriving  and solving  physics - 
based  equations.  MAS  are architectures  built  on the principle  of task decomposition  and role-based 
collaboration,  which  is widely  applied  in fields  such  as planning,  game  theory,  autonomous  driving, 
and complex dialogue systems [ 60–64]. Agents in MASs operate cooperatively, performing diverse 
tasks  and communication  protocols  in parallel,  enabling  them  to retrieve  data  and execute  tasks 
more  quickly  and efficiently  than  conventional  single -agent  systems  [60, 65]. This  feature  allows 
simultaneous  problem -solving  and enhanced  system  performance  in education  environment  [66]. 
In particular,  the MAS  architecture  could  be better  suited  for solving  biomechanics  calculation 
problems.  By simulating  the human  problem -solving  workflow  and dividing  the task into three 
stages,  including  understanding,  solving,  and reviewing,  MAS  establishes  a complete  loop  from 
natural  language  comprehension  to code  execution  and output  generation.  This  could  significantly  
improve  the stability,  transparency,  and traceability  of the solution  process.  
 
2 Methods  
2.1 Retrieval -Augmented  Generation,  RAG  
First, we  will focus  on assessing  automated  answering  for true/false  questions  in a biomechanics 
course  by LLM  and evaluate  whether  incorporating  external  knowledge  via RAG  can improve  the 
models’  performance,  stability,  and accuracy  in solving  domain -specific  questions.  Specifically,  we 
will 1) compare outputs of vanilla LLMs and RAG -enhanced LLMs on the same question and ex - 
plore  whether  RAG  can enhance  the performance  of LLMs  in specialized  true/false  question  tasks;  
2) analyze  factors  affecting  answer  accuracy,  including  retrieval  relevance,  hallucination  frequency, 
and output consistency.  The input to the LLMs will be 100 biomechanics conceptual true/false 
questions, while outputs are 1) baseline answers generated without any external knowledge sup - 
port, serving  as a control group  for comparison; 2)  RAG -enhanced  responses, which  incorporate 
retrieved  contextual  information  from  the domain -specific  knowledge  base;  3) confidence  scores  and 
supporting context for each response round.  
To support the RAG framework, we constructed a domain -specific vector knowledge base, which 
served  as the contextual  backbone  for retrieval  operations, allowing  the system  to fetch  semanti - 
cally relevant information prior to answer generation.  The source materials used for knowledge 
base  construction  included  biomechanics  textbooks  ”Fundamentals  of Biomechanics”  by Ozkaya 
et al. [ 67] (in PDF format), based on which the lecture notes are prepared.  These documents 
were  preprocessed  through  a multi -stage  pipeline.  First,  we applied  document  cleaning  using  the 
UnstructuredLoader,  which  parsed  PDFs  while  removing  non-informative  content  such  as headers, 
footers, tables of contents, and repetitive figures.  Next, the cleaned text was segmented using 
RecursiveCharacterTextSplitter,  with  a chunk  size of 1000  characters  and a 200-character  overlap. 
This  strategy  helped  preserve  semantic  continuity  between segments, ensuring  that retrieval  could 
capture  coherent  information  spans.  Each  chunk  was then  embedded  into a high -dimensional  vec- 
tor space using mxbai -embed -large -v1 [68]. Retrieval was then conducted using a FAISS vector 
store  [69], paired  with  Maximal  Marginal  Relevance  (MMR)  to maximize  both  relevance  and diver - 
sity in the retrieved  content.  For each  query,  the system  selected  the top 10 most  relevant  document 
chunks.  During  generation,  the original  prompt  and the retrieved  content  were  jointly  fed into the 
LLM,  which  produced  a structured  response  that included  the answer,  the supporting  context,  and 
a confidence  score.  This  design  allowed  us to systematically  evaluate  whether  enriching  the LLM’s 
input  with  retrieved  domain -specific  knowledge  could  improve  not only  answer  accuracy,  but also 

4  Total number of questions logical coherence and output consistency. 
 
2.2 Dataset Construction 
A custom-built domain-specific dataset was constructed, covering key knowledge points from biome- 
chanics courses focusing on human posture and movement.  The dataset consists exclusively of 
true/false conceptual questions, each accompanied by a ground truth label for evaluation purposes. 
The dataset includes a total of 100 questions, covering the three topics, namely, statics, dynamics, 
and mechanics of biomaterials, with an average length of 21.7 characters, written in a formal and 
domain-specific style. This dataset serves as the primary benchmark for evaluating system perfor- 
mance across three experiments: 1) baseline LLM performance without retrieval augmentation; 2) 
RAG-enhanced LLM outputs; 3) multi -round consistency analysis to assess output stability and 
reliability. 
 
2.3 Test the effects of prompts on model performance 
In the baseline group, LLMs (Qwen-2.5-32B and Llama-2-70B) were directly prompted to answer 
the biomechanics questions without access to any external knowledge base or contextual support. 
This setup simulates the model’s performance purely based on pertaining.  To investigate how 
prompts influence the model’s output, we also perform prompt template comparison by using three 
levels of prompts with increasing domain specificity:  
• “This is a question, help me answer it.” (generic); 
• “This is a biomechanics question, help me answer it.” (mild guidance); 
• “Please think carefully about this biomechanics question and give a professional answer.” 
(strong guidance + professional emphasis).  
 
 
2.4 Evaluation Metrics 
To comprehensively measure the performance of LLMs on true/false question tasks, two evaluation 
metrics were adopted:  
Accuracy is the primary metric used to assess whether the model’s output matches the ground 
truth answer, and it is evaluated using Accuracy =  Number of correct answers . In cases where models 
were executed in multi-round settings, we computed both the average accuracy and the standard 
deviation to assess not only correctness but also output stability across runs.  A comprehensive 
comparative analysis was conducted to evaluate the impact of key variables, including the pres - 
ence or absence of RAG, the type of language model used, and variations in prompt design.  The 
summary of the experimental variables is summarized in Table 1. This analysis enabled a quanti- 
tative understanding of how RAG, model architecture, and input phrasing affected overall system 
performance. 
Stability. LLMs are inherently probabilistic.  To measure their robustness, each question was 
answered three times per experiment. Stability  evaluation included: 1) Answer Consistency: Per- 
centage of questions where all three outputs are identical; 2) Confidence Variance: Analyzing fluc- 
tuation across the three trials; 3) Instability Diagnosis: Identifying causes of inconsistent outputs 
(e.g., retrieval changes); 4) RAG Impact:  Whether RAG improves answer consistency.  

5  Dimension  Values  Description  
LLM  Model  Qwen -2-5-32B / Llama -2- 
70B Two  LLMs  
Prompt  Template  Generic  / Domain -specific  
/ Structured  Tests  effect  of prompt  strength  
Temperature  0.6 /  0.8 Controls  generation  diversity  and 
consistency  
Knowledge  Support  No (Baseline)  / Yes (RAG)  Whether  external  knowledge  is 
introduced  
Answer  Rounds  3 responses  per question  For evaluating  output  stability  
Output  Format  answer  + context  + confi - 
dence  Supports  structured  evaluation  
and visualization  
 
Table 1:  Experimental settings and variables used to evaluate LLM performance of answering 
conceptual questions.  
 
2.5 Multi -Agent  System,  MAS  
To enable  automated  solving  of biomechanical  analysis  problems  requiring  equation  derivation  and 
numerical computation, we propose a MAS architecture based on LLMs.  Inspired by the human 
problem -solving  process,  the system  adopts  a modular  design  with  three  cooperative  agents:  namely, 
Manager, Solver, and Reviewer.  The system is built using modular tools, such as LangChain and 
LangGraph, and agents share context via a unified State object, which includes the conversation 
history,  current  execution  status,  and ground  truth  reference.  As shown  in Fig. 1, the system  starts 
with  the Manager,  which  interprets  the original  input  (text,  images,  or formulas)  and reformulates 
it into a clear, structured natural language question.  Based on the task type, the Manager either 
proceeds to the Solver (for multi -step reasoning and code execution) or directly routes to the 
Reviewer (for evaluation).  The Solver plans and executes each step, generating Python code and 
computing  results,  then  returns  to the Manager  for further  decision -making.  Once  complete,  the 
Reviewer evaluates the final solution against the ground truth, optionally scoring and providing 
feedback before the system terminates.  
 
2.6 Agent  Roles  and Prompt  Design  
2.6.1  Manager:  Problem  Interpretation  and Input  Normalization  
As the entry  point  of the MAS,  the Manager  agent  receives  the raw question  input  and transforms 
it into a well-structured,  clear  natural  language  description. For  multimodal  inputs,  we utilize  the 
Qwen2.5 -VL-72B-Instruct model to jointly interpret text, images, and embedded formulas, and 
convert the questions into a predefined prompt template.  
 
2.6.2  Solver:  Stepwise  Reasoning  and Code  Execution  
The Solver  serves  as the calculation  core  of the MAS,  and is implemented  using  a high -performance, 
text-only  language  model  such  as Mistral -3.2-24B-Instruct -2506.  It is responsible  for translating  the 
structured  problem  statement  into a logically  coherent  and executable  solution.  The Solver  follows 
a multi -stage  process:  First, it  performs  plan  generation,  where  it outlines  a structured  sequence 
of steps required to solve the problem.  Next, it proceeds with stepwise execution, carrying out 
each  reasoning  step  in order  and generating  Python  code  as needed.  Once  code  segments  are 

6   
 
Figure  1: Workflow  of MAS  for solving  biomechanics  problems  involving  the derivation  of governing 
equations and execution of numerical calculations.  
 
produced, they are passed to an integrated code runner, which executes the code and outputs 
solutions.  Finally, the Solver produces a structured Markdown -formatted output that includes 
natural language reasoning, code blocks, and the corresponding numerical results.  This design 
ensures both calculation accuracy and clear traceability of the problem -solving process.  
 
2.6.3  Reviewer:  Solution  Evaluation  and Scoring  
The Reviewer operates during the evaluation phase of the MAS system.  When a ground truth 
answer  is available,  the Reviewer  compares  the output  generated  by the Solver  with  the reference 
solution  and assigns  a corresponding  score.  Its role is to simulate  the judgment  and feedback  process 
typically  performed  by a human  grader.  Specifically,  the Reviewer  provides  a binary  judgment  of 
correctness, an  overall  score  out of 100, and  detailed  feedback  on any missing  steps, miscalculations, 
or logical flaws in the solution.  Additionally, it assesses the clarity of reasoning and expression, 
providing  insights  into the overall  quality  of the answer.  This  module  plays  a critical  role in enabling 
automated  grading,  formative  assessment,  and self-correction  feedback  in calculation  educational 
tasks.  
 
2.7 Dataset  Construction  
A custom dataset of biomechanics problems involving multi -step mathematical calculations was 
developed  to test the performance  of MAS.  The problems  were  sourced  from  the textbook,  lecture 
notes, homework, and prior midterm and final exams.  The resulting dataset reflects common 
problem types encountered in undergraduate and graduate -level biomechanics instruction.  The 
dataset  contains  a total  of 30 problems,  each  accompanied  by a text prompt,  an image  or diagram, 
a reference (ground truth) answer, and a set of standard solution steps used for performance 
evaluation.  


7  2.8 Task  setup  
The primary  goal  of the Task  is to assess  the capability  of MAS  to perform  reasoning,  mathemat - 
ical modeling,  code  execution,  and generate  structured  outputs,  and to evaluate  its effectiveness 
in real-world  educational  scenarios.  The MAS  accepts  as input  a biomechanics  problem  state - 
ment,  typically  in textual  form,  and optionally  accompanied  by images  or embedded  equations.  A 
reference  (ground  truth)  answer  is also provided  for evaluation.  The system’s  output  includes  a 
step-by-step reasoning trace, auto -generated and executed Python code, the final numerical result 
with  appropriate  units,  and,  when  the Reviewer  module  is active,  feedback  and scoring  information. 
The key objectives  of this task  are threefold:  (1) to verify  whether  the MAS  system  can ac- 
curately interpret and solve domain -specific calculation problems;  (2) to evaluate its performance 
in terms  of accuracy,  stability,  and explainability;  and (3) to compare  its results  with  those  of 
conventional  single -model  baselines  in order  to validate  the benefits  of the MAS  architecture.  
Experimental  Phase  1: MAS  Built  with  the Same  Multimodal  LLMs  
In the first phase  of experimentation,  all three  agents  in the MAS  architecture,  Manager,  Solver, 
and Reviewer, will be configured to use the same multimodal LLM. This setup is designed to 
evaluate the system’s performance under a fully unified configuration and to establish an upper - 
bound baseline.  Each biomechanics calculation problem will be solved five times, enabling both 
accuracy and output consistency to be assessed.  The primary goal of this phase is to determine 
how  well a single,  powerful  multimodal  model  could  perform  across  all roles  in the MAS  pipeline. 
Specifically,  we will test different  sizes  of Qwen2.5 -VL models. Key  metrics,  including  the per-run 
accuracy,  the average  accuracy  across  repetitions,  and the stability  of model  outputs  under  identical 
input conditions, will be recorded.  
Experimental  Phase  2: Hybrid  MAS  Configuration  
In the second phase of the experiment,  the best -performing multimodal model identified in Phase  
1 will be retained for the Manager agent, while the Solver and Reviewer agents will be replaced 
with  a more  lightweight  and efficient  text-only  model,  specifically  Mistral -3.2-24B-Instruct.  This 
configuration  established  a hybrid  MAS  architecture,  aiming  to balance  calculation  cost with  task - 
specific  effectiveness.  The goal  of this phase  is to assess  whether  text-based  models  could  reliably 
handle  core  responsibilities,  such  as logical  reasoning  and automated  evaluation,  while  preserving 
system performance.  As in Phase 1, each problem will be solved five times, and the resulting 
outputs will be analyzed for both accuracy and consistency of output.  Experimental results from 
this hybrid  setup  will be compared  with  those  from  the phase  1 experiments,  enabling  us to evaluate 
trade -offs in performance,  efficiency,  and stability  between  the two architectural  designs.  
 
3 Results  
3.1 The performance  of baseline  models  on answering  conceptual  questions  
This section presents and analyzes LLM performance under different settings with a focus on  
the following topics:  1) Performance differences between LLM architectures; 2) The influence of 
different  prompt  templates;  3) Baseline  models  vs. RAG -enhanced  models;  4) The impact  of RAG 
on accuracy,  stability,  and explainability.  To achieve  these  goals,  we designed  three  levels  of prompt 
instructions  (from  general  to domain -specific)  across  three  LLMs:  Llama -70B,  Qwen -1.0-32B,  and 
Qwen -2.5-32B.  We also compared  performance  under  different  temperature  settings.  Each  question 
was answered three times by each model to assess the stability, which refers to the percentage of 
cases  in which  all three  outputs  were  consistent,  and accuracy,  which  is assessed  by the proportion 
of outputs  matching  the ground  truth.  It is noted  that LLM  not only  responds  to the true/false  

8  
answer but also provides the rationale behind its answers.  An example of the results of answers 
generated using the three different prompts is listed below,  
Example  Question  
Quasi -static  equilibrium  is a good  assumption  for running.  
Ground  Truth  Answer:  False  
prompt 1:  ‘answer’: True, ‘context’: ‘Quasi -static equilibrium is often used in biomechanics to 
analyze forces at specific instants or averaged over a cycle where the changes aren’t too abrupt. 
While  running  is dynamic,  involving  inertia  and rapid  force  changes,  quasi -static  assumptions  can 
still be useful for understanding certain aspects like average forces during a stride’, ‘confidence’: 
‘0.8’.  
prompt 2:  ‘answer’: False, ‘context’: ‘Running involves dynamic movements with acceleration 
and deceleration phases, muscle activity, and inertial forces.  Quasi -static equilibrium assumes 
balanced  forces  with  no significant  changes  over  time,  which  doesn’t  align  with  the dynamic  nature 
of running’, ‘confidence’:  ‘0.9’  
prompt  3: ‘answer’: False,  ‘context’: ‘Running  involves  continuous  motion  with  phases  of accel - 
eration  and deceleration,  where  net forces  are not zero.  Therefore,  quasi -static  equilibrium  is not 
a good assumption’, ‘confidence’:  ‘0.8’  
As listed in the example above, the prompt framing had a clear impact on the model’s out - 
put. With a generic prompt (Prompt 1), the model incorrectly responded “True,” suggesting that 
quasi -static  assumptions  can be used  to analyze  average  forces,  which  clearly  oversimplifies  and mis- 
represents  the underlying  physics  of running.  Prompts  2 and 3, which  introduced  domain -specific 
framing  and professional  tone,  guided  the model  to conclude  that running  involves  dynamic  forces 
and is not a suitable assumption with quasi -static equilibrium.  This result highlights that insuf - 
ficient  or vague  prompts  can trigger  hallucinated  reasoning,  while  domain -specific  prompts  guide 
the model toward scientifically valid conclusions.  
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
Figure  2: Comparison  of stability  and accuracy  in Llama -70B models  under  three  types  of prompts 
and two temperature settings (0.6&0.8).  
 
Figure 2 shows the accuracy and stability of Llama -70B across the three prompt settings for 
answering  the examined  100 true/false questions.  It can be observed  that as  the prompt  became 
more  domain -specific,  model  performance  improved  accordingly.  At an appropriate  temperature 
(0.6),  accuracy increased from 62% to 70%,  and stability improved from 60% to 72%.  This indicates  

9  
that both  prompt  optimization  and temperature  adjustment  can effectively  reduce  answer  variance 
and enhance output quality.  
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
Figure  3: Stability  and accuracy  performance  of Qwen -1.0-32B under  three  types  of prompts  and 
two temperature settings (0.6&0.8).  
 
Figure  3 presents  the performance  of Qwen -1.0 under  the same  experimental  setting.  Similar 
to Llama, more domain -specific prompts and optimized temperature (0.6) led to noticeable im - 
provements.  Under the best -case scenario, the model achieved 63% accuracy and 71% stability. 
Figure 4 further illustrates the performance trends of Qwen -2.5. Among the three prompt varia - 
tions,  accuracy  increased  from  65%  to 67%,  while  stability  improved  from  85%  to 88%.  We noted 
that although the accuracy gain was modest, the model’s high output consistency suggests that 
Qwen -2.5 has a more robust architecture for specialized tasks than that of Qwen -1.0. 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
Figure  4: Stability  and accuracy  performance  of Qwen -2.5-32B under  three  types  of prompts  and 
two temperature settings (0.6&0.8).  
 
In summary, our numerical experiments demonstrated that as prompts became more tailored to 
the biomechanics  domain,  for example,  by explicitly  including  the term  “biomechanics”,  all three  

10  models  exhibited  consistent  improvements  in both  accuracy  and output  stability.  This  suggests  that 
domain -specific  language  cues  can help  LLMs  anchor  their  responses  more  reliably.  Additionally, 
adjustments  to the temperature  setting  also had a notable  impact  on stability.  Lower -temperature 
configurations  could  result  in 10–15% gains  in output  consistency,  making  these  settings  particu - 
larly effective for true/false questions where the consistency is important.  Third, the Llama -70B 
model  demonstrated  superior  baseline  performance,  achieving  strong  stability  even  in the absence 
of external  knowledge  support.  This  implies  that its underlying  architecture  or training  corpus  may 
be inherently  better  aligned  with  the conceptual  structure  characteristic  of biomechanics  reasoning 
tasks.  
 
3.2 The performance of models enhanced with RAG on answering conceptual 
questions  
After the Baseline experiments, we introduced the RAG mechanism into three baseline models, 
Qwen -1.0-32B, Qwen -2.5-32B, and Llama -70B, by providing the same retrieved knowledge base 
content as additional input context.  
 
 
Figure 5:  Impact of RAG on improving the accuracy and stability of Llama -70B, Qwen -1.0-32B, 
and Qwen -2.5-32B.  
 
Figure  5 compares  the accuracy  and stability  of all three  models  with  and without  RAG.  It is ev- 
ident  that RAG  significantly  improved  both  metrics  across  all models,  with  Qwen -2.5 achieving  the 
highest  stability  (97%).  Common  error  types  observed  in baseline  outputs  included  hallucinations, 
knowledge gaps (conceptual confusion or misinterpretation), and redundant content (irrelevant 
elaboration).  These  issues  were  significantly  reduced  as the retrieval  of relevant  passages  from  the 
domain  knowledge  base  helped  the model  to further  refine  its outputs.  These  results  demonstrate 
that the RAG framework notably improved accuracy on these conceptual questions.  
 
3.3 Results  of MAS  built  with  the same  multimodal  LLMs  
In the first experimental phase, all three agents in the MAS system, namely, manager, solver, 
and reviewer,  were  configured  to use the same  multimodal  LLMs  to solve  biomechanics  calculation 
problems.  The goal  was to evaluate  the upper -bound  performance  of a fully  multimodal  MAS  


11  pipeline.  To further  illustrate  the differences  in reasoning  quality  and numerical  accuracy  among 
the three  models  under  the same  MAS  configuration,  we present  a representative  example  question 
as shown in Figure 6(A). 
The example  involves  a biomechanics  computation  problem,  which  was independently  solved 
using three different multimodal models: Gemma -3-27B, Qwen2.5 -VL-32B, and GPT -4o. Their 
respective outputs are visualized below to highlight differences in problem interpretation, code 
generation,  and final  results.  In this comparison,  we only  display  the Manager  module’s  output  for 
each model to visualize how they interpret and formulate the problem (see Figure 6(B)). The full 
solution  process,  including  Solver  execution  and Reviewer  evaluation, is  provided  in the appendix.  
 
Figure  6: (A)  An example  of biomechanics  calculation  problems. What  is the horizontal  center  of 
gravity  (CG x) for a person  standing  on one foot with  an outstretched  arm,  and what  mass  must  a 
ball have  to maintain  equilibrium? (B)Manager  Analysis  by Gemma -3-27B,  Qwen2.5 -VL-32B and 
GPT -4o, respectively.  
 
Table  2 lists the key outputs  from  the three  examined  LLMs  for solving  the example  problem. 
The MAS system’s Reviewer module assigned scores of 42, 83, and 95 to Gemma, Qwen2.5, and 
GPT -4o, respectively.  The low score  for Gemma  stemmed  from  unrealistic  outputs  (e.g.,  negative 
mass), while Qwen2.5 performed generally well but did not achieve the correct center of gravity 
for this posture.  GPT -4 produced the most accurate and physically plausible solution, with the 
Reviewer  noting  well-structured  reasoning,  correct  unit handling,  and consideration  of anatomical 
balance.  To assess  the impact  of the MAS  system,  we evaluated  the performance  of using  GPT -4o 
alone  to solve  the same  problem.  As shown  in the bottom  row of Table  2, a single  GPT -4o failed  to 
solve  both  the center  of gravity  and force  and moment  balance  correctly,  underscoring  the critical 
role of the MAS system in guiding the solution process.  
Figure 7(A) illustrates the comparison of the accuracy of gemma -3-27b-it, qwen2.5 -vl-32b- 
instruct,  and GPT -4o when  applied  uniformly  across  all agents  (Manager,  Solver,  Reviewer)  in 


12  Table  2: Comparison  of performance  of three  MAS  with  different  LLMs  on the example  problem.  
 
Model  CGx  (m)  Ball  Mass  (kg)  Equations  Score  Summary  Evaluation  
Gemma -3-27B 1.32 -95.19  No 42 Incomplete   reasoning;  incorrect  
CGx  estimation  led to an invalid  
(negative)  result.  
Qwen2.5 -VL-32B 0.181  21.93  No 83 Clear  derivation  and correct  units;  
slightly  overestimated  due to wrong  
equations  in calculating  CGx.  
GPT -4o -0.0171  15.58  Yes 95 Most  accurate  and physically  real- 
istic;  correct  selection  of equations  
and reasoning  was complete.  
GPT -4o (no MAS)  0.022  2.21 No N/A  Incorrect  CGx  estimation  and did 
not use moment  balance  to esti- 
mate  the mass  of the ball,  leading  
to wrong  results.  
 
the MAS  framework. Accuracy  was averaged  over  five runs  per problem. Error  bars  represent  the 
standard deviation, indicating the output stability.  The results show that GPT -4o achieved the 
highest accuracy (82.0%), followed by Qwen2.5 (75.4%), with Gemma trailing at 68.6%.  These 
results  establish  a performance  baseline  for evaluating  hybrid  MAS  configurations  in subsequent 
experiments.  
 
 
Figure  7: Accuracy  comparison  of MAS  architectures  on solving  the 30 biomechanics  calculation 
questions,  including  (A) MAS  built  from  a single  multimodal  model  type  and (B) MAS  constructed 
from hybrid combinations of multimodal model types.  
 
 
3.4 Results  of MAS  Built  with  Hybrid  Models  
In the second phase of the experiment, the Qwen2.5 -VL model was retained for the Manager 
agent,  while  both  the Solver  and Reviewer  agents  were  replaced  with  a high -performance  text-only 
language  model,  Mistral -3.2-24B-Instruct.  This  hybrid  configuration  was designed  to assess  whether 
such a model combination could deliver high accuracy while reducing computational complexity 
and cost.  As summarized in Table 3, the hybrid MAS system, which combines Qwen2.5 -VL for 
multimodal  understanding  and Mistral  for symbolic  computation,  produced  the most  accurate  and 
structured  solution.  The complete  step -by-step  solution  process  is provided  in the appendix.  It 


13  achieved  full correctness  on both  center -of-gravity  and equilibrium  mass  calculations,  receiving  a 
100%  system  score.  In contrast,  the Qwen -only  model  showed  slight  inaccuracies  and less structured 
outputs,  while  GPT -4o demonstrated  strong  reasoning  but lacked  fine-grained  solution  traceability.  
 
Table  3: Evaluation  of different  MAS  structures  on the example  problem.  
 
Model  Configuration  
Type  Problem  Interpre - 
tation  Mathematical  
Modeling  Structured  Out- 
put System  
Score  
Qwen2.5 - 
VL-32B  Fully  Multimodal  
(MAS)  Moderate,  clear  but 
not deeply  technical  Basic  derivation,  
lacks  detail  Step -based  Mark - 
down;  somewhat  
brief  83 
Hybrid  
MAS  
(Qwen  + 
Mistral)  Multimodal  Man - 
ager  + Text -only 
Solver  Precise,  academic,  
rigorous  Complete  reason - 
ing chain;  accurate 
torque -based  formu - 
lation  Fully  structured  
Markdown  output  100 
GPT -4o Fully  Multimodal  
(MAS)  Most  professional  
and fluent  Symbolically  rich,  
concise,  clear  logic  Well -separated  rea- 
soning  steps,  but 
execution  trace  lim- 
ited 95 
 
Figure  7(B) summarizes  the comparison  of the accuracy  of the hybrid  MAS  configuration  with 
error  margins  for all the examined  problems.  This  bar chart  compares  the performance  of a hybrid 
MAS in which the Manager uses the multimodal model Qwen2.5 -VL, while Solver and Reviewer 
use the text -only model Mistral -3.2-24B-Instruct.  Compared with the fully multimodal baseline 
(Qwen2.5  only), the  hybrid  configuration  achieved  a notable  improvement  in accuracy, 81.7%  versus 
75.4%, while  approaching  the performance  of GPT -4o (82.0%).  The error  bars  indicate  standard 
deviation  across  five runs  per problem,  showing  strong  output  consistency.  
In summary,  our experimental  findings  revealed  several  key insights.  First,  hybrid  model  con- 
figurations  outperformed  fully  multimodal  setups,  particularly  on reasoning -intensive  tasks  such  as 
step-by-step mathematical  computations.  Text -based  models  could  offer  efficiency  and more  consis - 
tent performance  in executing  logic -driven  steps,  while  still maintaining  high  accuracy.  Second,  the 
Manager  agent  proved  to be a pivotal  component  in the MAS  architecture.  Its ability  to clearly  for- 
mulate  problem  inputs  had a direct  influence  on the performance  of downstream  agents,  especially 
the Solver.  Poorly  structured  inputs  often  led to misinterpretation  or incomplete  execution.  Third, 
structuring  the problem -solving  process  with  explicit  reasoning  steps,  formulas,  executable  code, 
and final outputs could greatly enhance the transparency and explainability of the system.  This 
format  not only  facilitated  debugging  and evaluation  but also made  the outputs  more  accessible  for 
educational use, including automated grading and student feedback.  Lastly, the agent -based ar - 
chitecture  significantly  improved  output  stability  and consistency,  demonstrating  the architectural 
advantages of MAS in maintaining reliability during complex, multi -stage problem solving.  
 
4 Summary  
In the current  study,  we evaluated  the extent  of RAG -induced  improvement  in LLM  performance 
on 100 biomechanics  true/false  questions  used  in the quizzes  and exams  of the Biomechanics  class 
under  both  RAG -enabled  and baseline  conditions.  Specifically,  we have  comprehensively  assessed 
the performance  gains  in terms  of accuracy,  confidence,  and contextual  grounding.  Main  contri - 
butions  of this  task  include: 1) Construction  of a high -quality,  domain -specific  biomechanics 
vector  knowledge  base  and the development  of a complete  RAG  application  pipeline  using  open - 
source  LLMs.  2) Implementation  of structured  answer  output  supporting  automated  evaluation  and 

14  visualized  result  export.  3) Quantitative  comparison  between  baseline  LLMs  and RAG -enhanced 
LLMs  with  a systematic  analysis  of error  types  and optimization  potential.  We expect  the proposed 
system to not only provide a novel approach to automated answering and assessment in educational 
scenarios but also serve as a technical prototype for the practical application of LLM + RAG in 
closed -domain, course -specific environments.  
To address the limitations of LLMs in solving problems involving multi -step mathematical 
calculations, we further propose and implement a MAS specifically designed for biomechanics cal - 
culation problems.  The system comprises three core modules, Manager, Solver, and Reviewer, 
each  assigned  explicit,  complementary  responsibilities.  By enforcing  a clear  division  of roles  and 
a structured output format, the MAS emulates a human -like problem -solving workflow, thereby 
enhancing  the reliability,  accuracy,  and interpretability  of solutions  to complex  biomechanics  prob - 
lems requiring quantitative calculation.  Specifically, The major contributions of this Task 
include:  1) It is among the first to systematically address the automated solving of calculation 
biomechanics  problems.  2) Our framework  introduces  a three -stage  agent  pipeline,  consisting  of a 
Manager,  Solver,  and Reviewer,  that simulates  the full human  problem -solving  process,  including 
understanding,  reasoning,  execution,  and evaluation.  3) Our framework  achieves  a deep  integration 
of LLM -based  reasoning  with  executable  tool modules,  allowing  it to generate,  execute,  and verify 
code as part of a structured solution workflow.  
In summary,  we present  an AI-driven  educational  framework  that leverages  LLMs,  RAG,  and 
MAS  to enhance  learning  and assessment  in undergraduate  biomechanics  courses.  Because  general - 
purpose  LLMs  struggle  with  domain -specific  reasoning,  physics -based  analysis,  and multi -step cal- 
culations, the authors design a two -module system:  a RAG pipeline for improving accuracy, sta - 
bility,  and contextual  grounding  on 100 conceptual  biomechanics  true/false  questions,  and a MAS 
composed of Manager, Solver, and Reviewer agents to solve complex calculation problems requiring 
equation  derivation,  code  execution,  and structured  reasoning.  Experiments  across  several  open - 
source  and commercial  LLMs  demonstrate  that RAG  substantially  increases  accuracy  and reduces 
hallucination  in conceptual  tasks,  while  MAS  significantly  improves  correctness  and explainabil - 
ity for multi -step computational  biomechanics  problems,  particularly  in hybrid  configurations  that 
combine multimodal and text -only models.  Together, these results highlight a practical and scalable 
approach for building intelligent AI assistants that can support specialized engineering education.  
 
References  
[1] Junnan  Dong, Qinggang Zhang, Chuang Zhou, Hao Chen, Daochen Zha, and Xiao Huang. 
Cost -efficient  knowledge -based  question  answering  with  large  language  models.  arXiv  preprint 
arXiv:2405.17337 , 2024.  
[2] Zhou Yu, Xuecheng Ouyang, Zhenwei Shao, Meng Wang, and Jun Yu.  Prophet:  Prompt - 
ing large  language  models  with  complementary  answer  heuristics  for knowledge -based  visual 
question  answering.  IEEE Transactions  on Pattern  Analysis  and Machine  Intelligence , 2025.  
[3] Wenbin An, Feng Tian, Jiahao Nie, Wenkai Shi, Haonan Lin, Yan Chen, QianYing Wang, 
Yaqiang Wu, Guang Dai, and Ping Chen.  Knowledge acquisition disentanglement for 
knowledge -based visual question answering with large language models.  arXiv preprint 
arXiv:2407.15346 , 2024.  
[4] Junnan Dong, Qinggang Zhang, Huachi Zhou, Daochen Zha, Pai Zheng, and Xiao Huang. 
Modality -aware  integration  with  large  language  models  for knowledge -based  visual  question 
answering.  arXiv  preprint  arXiv:2402.12728 , 2024.  

15  [5] Wenyu  Huang,  Guancheng  Zhou,  Mirella  Lapata,  Pavlos  Vougiouklis,  Sebastien  Montella,  and 
Jeff Z Pan.  Prompting  large  language  models  with  knowledge  graphs  for question  answering 
involving  long -tail facts.  Knowledge -Based  Systems , page  113648,  2025.  
[6] Zhongjian Hu, Peng Yang, Fengyuan Liu, Yuan Meng, and Xingyu Liu.  Prompting large 
language  models  with  knowledge -injection  for knowledge -based  visual  question  answering.  Big 
Data Mining  and Analytics , 7(3):843 –857,  2024.  
[7] Gyeonggeon  Lee and Xiaoming  Zhai.  Realizing  visual  question  answering  for education:  Gpt- 
4v as a multimodal ai.  TechTrends , pages 1 –17, 2025.  
[8] Xiaoming Zhai.  Chatgpt user experience: Implications for education.  Available at SSRN 
4312418 , 2022.  
[9] Xiaoming Zhai, Yue Yin, James W Pellegrino, Kevin C Haudek, and Lehong Shi.  Applying 
machine  learning  in science  assessment:  a systematic  review.  Studies  in Science  Education , 
56(1):111 –151, 2020.  
[10] Xiaoming  Zhai.  Chatgpt  for next  generation  science  learning.  XRDS:  Crossroads,  The ACM 
Magazine  for Students , 29(3):42 –46, 2023.  
[11] Xiaodan Tang, Yue Yin, Qiao Lin, Roxana Hadad, and Xiaoming Zhai.  Assessing computa - 
tional thinking:  A systematic review of empirical studies.  Computers  & Education , 148:103798, 
2020.  
[12] Ehsan  Latif,  Gengchen  Mai,  Matthew  Nyaaba,  Xuansheng  Wu, Ninghao  Liu, Guoyu  Lu, Sheng 
Li, Tianming  Liu, and Xiaoming  Zhai.  Artificial  general  intelligence  (agi)  for education.  arXiv 
preprint  arXiv:2304.12479 , 1, 2023.  
[13] Xiaoming  Zhai  and Min Lu. Machine  learning  applications  in educational  studies.  In Frontiers 
in Education , volume  8, page  1225802.  Frontiers  Media  SA, 2023.  
[14] Xiaoming  Zhai. Transforming  teachers’  roles  and agencies  in the era of generative  ai: Percep - 
tions,  acceptance,  knowledge,  and practices.  Journal  of Science  Education  and Technology , 
pages 1 –11, 2024.  
[15] Muhammad Usman Hadi, Rizwan Qureshi, Abbas Shah, Muhammad Irfan, Anas Zafar, 
Muhammad Bilal Shaikh,  Naveed Akhtar,  Jia Wu,  Seyedali Mirjalili,  et al.  A survey on 
large  language  models: Applications,  challenges,  limitations,  and practical  usage.  Authorea 
Preprints , 2023.  
[16] Mohaimenul  Azam  Khan  Raiaan,  Md Saddam  Hossain  Mukta,  Kaniz  Fatema,  Nur Mohammad 
Fahad,  Sadman  Sakib,  Most  Marufatul  Jannat  Mim,  Jubaer  Ahmad,  Mohammed  Eunus  Ali, 
and Sami  Azam.  A review  on large  language  models: Architectures,  applications,  taxonomies, 
open issues and challenges.  IEEE  access , 12:26839 –26874, 2024.  
[17] Muhammad Usman Hadi, Rizwan Qureshi, Abbas Shah, Muhammad Irfan, Anas Zafar, 
Muhammad  Bilal Shaikh,  Naveed Akhtar,  Jia Wu,  Seyedali Mirjalili,  et al.  Large lan - 
guage  models: a  comprehensive  survey  of its applications,  challenges,  limitations,  and future 
prospects.  Authorea  Preprints , 1:1–26, 2023.  
[18] Mert Karabacak and Konstantinos Margetis.  Embracing large language models for medical 
applications:  opportunities  and challenges.  Cureus , 15(5),  2023.  

16  [19] Rajvardhan  Patil  and Venkat  Gudivada. A  review  of current  trends,  techniques,  and challenges 
in large  language  models  (llms).  Applied  Sciences , 14(5):2074,  2024.  
[20] Yupeng Chang, Xu Wang, Jindong Wang, Yuan Wu, Linyi Yang, Kaijie Zhu, Hao Chen, 
Xiaoyuan Yi, Cunxiang Wang, Yidong Wang, et al. A survey on evaluation of large language 
models.  ACM transactions  on intelligent  systems  and technology , 15(3):1 –45, 2024.  
[21] Rui Yang, Ting Fang Tan, Wei Lu, Arun James Thirunavukarasu, Daniel Shu Wei Ting, and 
Nan  Liu. Large  language  models  in health  care: Development, applications, and  challenges. 
Health Care Science , 2(4):255 –263,  2023.  
[22] Shubo Tian, Qiao Jin, Lana Yeganova, Po -Ting Lai, Qingqing Zhu, Xiuying Chen, Yifan 
Yang, Qingyu Chen, Won Kim, Donald C Comeau, et al.  Opportunities and challenges for 
chatgpt  and large  language  models  in biomedicine  and health.  Briefings  in Bioinformatics , 
25(1):bbad493, 2024.  
[23] Yifan  Li, Yifan  Du, Kun  Zhou,  Jinpeng  Wang,  Wayne  Xin Zhao,  and Ji-Rong  Wen.  Evaluating 
object  hallucination  in large  vision -language  models.  arXiv  preprint  arXiv:2305.10355 , 2023.  
[24] Ziwei Xu, Sanjay Jain, and Mohan Kankanhalli.  Hallucination is inevitable: An innate limi - 
tation of large language models.  arXiv preprint arXiv:2401.11817 , 2024.  
[25] SMTI Tonmoy, SM Zaman, Vinija Jain, Anku Rani, Vipula Rawte, Aman Chadha, and Ami - 
tava Das.  A comprehensive survey of hallucination mitigation techniques in large language 
models.  arXiv  preprint  arXiv:2401.01313 , 6, 2024.  
[26] Lei Huang,  Weijiang  Yu, Weitao  Ma, Weihong  Zhong,  Zhangyin  Feng,  Haotian  Wang,  Qiang - 
long  Chen,  Weihua  Peng,  Xiaocheng  Feng,  Bing  Qin,  et al. A  survey  on hallucination  in large 
language  models:  Principles,  taxonomy,  challenges,  and open  questions.  ACM  Transactions 
on Information  Systems , 43(2):1 –55, 2025.  
[27] Patrick Lewis, Ethan Perez, Aleksandra Piktus, Fabio Petroni, Vladimir Karpukhin, Na - 
man  Goyal,  Heinrich  Ku¨ttler,  Mike  Lewis,  Wen -tau Yih, Tim Rockt¨aschel, et al. Retrieval - 
augmented generation for knowledge -intensive nlp tasks.  Advances  in neural  information  pro- 
cessing systems , 33:9459 –9474, 2020.  
[28] Yunfan Gao, Yun Xiong, Xinyu Gao, Kangxiang Jia, Jinliu Pan, Yuxi Bi, Yixin Dai, Jiawei 
Sun, Haofen Wang, and Haofen Wang.  Retrieval -augmented generation for large language 
models:  A survey.  arXiv  preprint  arXiv:2312.10997 , 2(1),  2023.  
[29] Alireza  Salemi  and Hamed  Zamani.  Evaluating  retrieval  quality  in retrieval -augmented  gen- 
eration.  In Proceedings  of the 47th International  ACM SIGIR  Conference  on Research  and 
Development in Information Retrieval , pages  2395 –2400,  2024.  
[30] Xiaohua Wang, Zhenghua Wang, Xuan Gao, Feiran Zhang, Yixin Wu, Zhibo Xu, Tianyuan 
Shi, Zhengyuan Wang, Shizheng Li, Qi Qian, et al.  Searching for best practices in retrieval - 
augmented  generation.  arXiv  preprint  arXiv:2407.01219 , 2024.  
[31] Hao Yu, Aoran Gan, Kai Zhang, Shiwei Tong, Qi Liu, and Zhaofeng Liu.  Evaluation of 
retrieval -augmented  generation:  A survey.  In CCF Conference  on Big Data , pages  102–120. 
Springer, 2024.  

17  [32] Guangzhi Xiong, Qiao Jin, Zhiyong Lu, and Aidong Zhang.  Benchmarking retrieval -augmented 
generation  for medicine.  In Findings  of the Association  for Computational  Linguistics  ACL 
2024 , pages  6233 –6251,  2024.  
[33] Shangyu Wu, Ying Xiong, Yufei Cui, Haolun Wu, Can Chen, Ye Yuan, Lianming Huang,  
Xue Liu, Tei-Wei Kuo,  Nan  Guan,  et al. Retrieval -augmented  generation  for natural  language 
processing:  A survey.  arXiv  preprint  arXiv:2407.13193 , 2024.  
[34] Shahul Es, Jithin James, Luis Espinosa Anke, and Steven Schockaert.  Ragas: Automated 
evaluation  of retrieval  augmented  generation.  In Proceedings  of the 18th Conference  of the 
European Chapter of the Association for Computational Linguistics:  System Demonstrations , 
pages 150 –158, 2024.  
[35] Yucheng  Shi, Tianze  Yang,  Canyu  Chen,  Quanzheng  Li, Tianming  Liu, Xiang  Li, and Ninghao 
Liu. Searchrag:  Can search  engines  be helpful  for llm-based  medical  question  answering?  arXiv 
preprint  arXiv:2502.13233 , 2025.  
[36] Yucheng  Shi, Shaochen  Xu, Tianze  Yang, Zhengliang  Liu, Tianming  Liu, Quanzheng  Li, Xiang 
Li, and Ninghao  Liu. Mkrag: Medical  knowledge  retrieval  augmented  generation  for medical 
question  answering.  arXiv  preprint  arXiv:2309.16035 , 2023.  
[37] Cyril Zakka, Rohan Shad, Akash Chaurasia, Alex R Dalal, Jennifer L Kim, Michael Moor, 
Robyn Fong, Curran Phillips, Kevin Alexander, Euan Ashley, et al.  Almanac —retrieval - 
augmented language models for clinical medicine.  Nejm  ai, 1(2):AIoa2300068, 2024.  
[38] Guangzhi Xiong, Qiao Jin, Xiao Wang, Minjia Zhang, Zhiyong Lu, and Aidong Zhang.  Im- 
proving retrieval -augmented generation in medicine with iterative follow -up questions.  In 
Biocomputing  2025:  Proceedings  of the Pacific  Symposium , pages  199–214. World  Scientific, 
2024.  
[39] Minbyul  Jeong,  Jiwoong  Sohn,  Mujeen  Sung,  and Jaewoo  Kang.  Improving  medical  reasoning 
through  retrieval  and self-reflection  with  retrieval -augmented  large  language  models.  Bioin - 
formatics , 40(Supplement  1):i119 –i129, 2024.  
[40] Lameck Mbangula Amugongo, Pietro Mascheroni, Steven Brooks, Stefan Doering, and Jan 
Seidel.  Retrieval  augmented  generation  for large  language  models  in healthcare:  A systematic 
review.  PLOS  Digital  Health , 4(6):e0000877,  2025.  
[41] Rui Yang, Yilin Ning, Emilia Keppo, Mingxuan Liu, Chuan Hong, Danielle S Bitterman, 
Jasmine  Chiat  Ling  Ong,  Daniel  Shu Wei Ting,  and Nan  Liu. Retrieval -augmented  generation 
for generative artificial intelligence in medicine.  arXiv  preprint  arXiv:2406.12449 , 2024.  
[42] Ozan Unlu, Jiyeon Shin, Charlotte J Mailly, Michael F Oates, Michela R Tucci, Matthew 
Varugheese,  Kavishwar  Wagholikar,  Fei Wang,  Benjamin  M Scirica,  Alexander  J Blood, 
et al.  Retrieval -augmented generation –enabled gpt -4 for clinical trial screening.  NEJM  AI, 
1(7):AIoa2400181, 2024.  
[43] Arunabh Bora and Heriberto Cuay´ahuitl.  Systematic analysis of retrieval -augmented 
generation -based llms for medical chatbot  applications.  Machine  Learning  and Knowledge 
Extraction , 6(4):2355 –2374,  2024.  

18  [44] Siru  Liu, Allison  B McCoy, and  Adam Wright.  Improving large language model  applications 
in biomedicine  with  retrieval -augmented  generation:  a systematic  review,  meta -analysis,  and 
clinical  development  guidelines.  Journal  of the American  Medical  Informatics  Association , 
page ocaf008, 2025.  
[45] Nirmalie  Wiratunga, Ramitha Abeyratne, Lasal Jayawardena, Kyle Martin, Stewart Massie, 
Ikechukwu Nkisi -Orji, Ruvan Weerasinghe, Anne Liret, and Bruno Fleisch.  Cbr-rag: case - 
based  reasoning  for retrieval  augmented  generation  in llms  for legal  question  answering.  In 
International Conference on Case -Based Reasoning , pages  445–460.  Springer,  2024.  
[46] Antoine  Louis, Gijs  van Dijck, and  Gerasimos  Spanakis.  Interpretable  long -form  legal  ques - 
tion answering  with  retrieval -augmented  large  language  models.  In Proceedings  of the AAAI 
Conference  on Artificial  Intelligence , volume 38, pages 22266 –22275, 2024.  
[47] Sanat Sharma, David Seunghyun Yoon, Franck Dernoncourt, Dewang Sultania, Karishma 
Bagga, Mengjiao Zhang, Trung Bui, and Varun Kotte.  Retrieval augmented generation for 
domain -specific  question  answering.  arXiv  preprint  arXiv:2404.14760 , 2024.  
[48] Nuo  Xu, Siben  Li, and Yufan  Xia. Research  on legal  question  answering  system  with  retrieval - 
augmented  large  language  models.  In International  Forum  on Digital  TV and Wireless  Mul- 
timedia  Communications , pages  137–151. Springer,  2024.  
[49] Ryan C Barron, Maksim E Eren, Olga M Serafimova, Cynthia Matuszek, and Boian S 
Alexandrov.  Bridging legal knowledge and ai: Retrieval -augmented generation with vector 
stores,  knowledge  graphs,  and hierarchical  non-negative  matrix  factorization.  arXiv  preprint 
arXiv:2502.20364 , 2025.  
[50] Dnyanesh Panchal, Aaryan Gole, Vaibhav Narute, and Raunak Joshi.  Lawpal: A retrieval 
augmented  generation  based  system  for enhanced  legal  accessibility  in india.  arXiv  preprint 
arXiv:2502.16573 , 2025.  
[51] Abe Bohan  Hou,  Orion  Weller,  Guanghui  Qin,  Eugene  Yang,  Dawn  Lawrie,  Nils Holzenberger, 
Andrew  Blair -Stanek,  and Benjamin  Van Durme. Clerc: A  dataset  for legal  case  retrieval  and 
retrieval -augmented analysis generation.  arXiv  preprint  arXiv:2406.17186 , 2024.  
[52] Jay D Humphrey  and Sherry  L Delange.  An introduction  to biomechanics.  Solids  and Fluids, 
Analysis  and Design.  Springer,  Heidelberg , page 10, 2004.  
[53] Tung -Wu Lu and Chu-Fen Chang.  Biomechanics  of human  movement  and its clinical  applica - 
tions.  The Kaohsiung  journal  of medical  sciences , 28(2):S13 –S25,  2012.  
[54] Aske Plaat, Annie Wong, Suzan Verberne, Joost Broekens, and Niki Van Stein.  Multi -step 
reasoning with large language models, a survey.  ACM Computing  Surveys , 2025.  
[55] Haolin  Jin, Linghan  Huang,  Haipeng  Cai, Jun Yan,  Bo Li, and Huaming  Chen. From  llms  to 
llm-based  agents  for software  engineering: A  survey  of current,  challenges  and future.  arXiv 
preprint  arXiv:2408.02479 , 2024.  
[56] Enkelejda  Kasneci,  Kathrin  Seßler,  Stefan  Ku¨chemann,  Maria  Bannert,  Daryna  Dementieva, 
Frank  Fischer,  Urs Gasser,  Georg  Groh,  Stephan  Gu¨nnemann,  Eyke  Hu¨llermeie r, et al. Chatgpt 
for good?  on opportunities  and challenges  of large  language  models  for education.  Learning 
and individual  differences , 103:102274, 2023.  

19  [57] Tariq Shahzad, Tehseen Mazhar, Muhammad Usman Tariq, Wasim Ahmad, Khmaies Oua - 
hada, and Habib Hamam.  A comprehensive review of large language models:  issues and 
solutions in learning environments.  Discover Sustainability , 6(1):27, 2025.  
[58] Taicheng Guo, Xiuying Chen, Yaqi Wang, Ruidi Chang, Shichao Pei, Nitesh V Chawla, Olaf 
Wiest,  and Xiangliang  Zhang.  Large  language  model  based  multi -agents:  A survey  of progress 
and challenges.  arXiv  preprint  arXiv:2402.01680 , 2024.  
[59] Yuheng  Cheng,  Ceyao  Zhang,  Zhengwen  Zhang,  Xiangrui  Meng,  Sirui  Hong,  Wenhao  Li, Zihao 
Wang, Zekai Wang, Feng Yin, Junhua Zhao, et al.  Exploring large language model based 
intelligent  agents:  Definitions,  methods,  and prospects.  arXiv  preprint  arXiv:2401.03428 , 
2024.  
[60] Ali Dorri, Salil S Kanhere, and Raja Jurdak.  Multi -agent systems: A survey.  Ieee Access , 
6:28573 –28593,  2018.  
[61] Enyo  Jos´e Tavares  Gon¸calves,  Mariela  I Cort´es, Gustavo  Augusto  Lima  Campos,  Yrleyjander  S 
Lopes, Emmanuel SS Freire, Viviane Torres da Silva, Kleinner Silva Farias De Oliveira, and 
Marcos Antonio De Oliveira.  Mas -ml 2.0: Supporting the modelling of multi -agent systems 
with different agent architectures.  Journal  of Systems  and Software , 108:77 –109, 2015.  
[62] Adelinde  M Uhrmacher  and Danny  Weyns.  Multi-Agent  systems:  Simulation  and applications . 
CRC press, 2018.  
[63] Viviane Torres Da Silva and Carlos JP de Lucena.  Mas -ml: a multi -agent system modeling 
language.  In Companion  of the 18th annual  ACM SIGPLAN  conference  on Object -oriented 
programming, systems, languages, and applications , pages 126 –127, 2003.  
[64] Diego  Maldonado, Edison Cruz, Jackeline Abad Torres, Patricio J Cruz, and Silvana del 
Pilar Gamboa Benitez.  Multi -agent systems:  A survey about its components, framework 
and workflow.  IEEE  Access , 12:80950 –80975,  2024.  
[65] Zool  Hilmi  Ismail,  Nohaidda  Sariff,  and E Gorrostieta  Hurtado.  A survey  and analysis  of co- 
operative  multi -agent  robot  systems:  challenges  and directions.  Applications  of Mobile  Robots , 
5:8–14, 2018.  
[66] Yuan -Hao  Jiang,  Tian -Yi Liu, Xinlin  Zhuang,  Hanglei  Hu, Ruijia  Li, and Rui Jia. Enhancing 
educational  practices  with  multi -agent  systems:  A review.  Enhancing  Educational  Practices: 
Strategies for Assessing and Improving Learning Outcomes , pages  47–65, 2024.  
[67] Nihat  O¨ zkaya,  Margareta  Nordin,  David  Goldsheyder,  and Dawn  Leger.  Fundamentals  of 
biomechanics , volume  86. Springer,  2012.  
[68] Chankyu  Lee, Rajarshi Roy, Mengyao Xu, Jonathan Raiman, Mohammad Shoeybi, Bryan 
Catanzaro, and Wei Ping.  Nv-embed:  Improved techniques for training llms as generalist 
embedding  models.  arXiv  preprint  arXiv:2405.17428 , 2024.  
[69] Matthijs Douze, Alexandr Guzhva, Chengqi Deng, Jeff Johnson, Gergely Szilvasy, Pierre - 
Emmanuel  Maza r´e, Maria  Lomeli,  Lucas  Hosseini,  and Her v´e J´egou.  The faiss  library.  IEEE 
Transactions  on Big Data , 2025.  

1APPENDIX

Manager
A person with a total body weight (M) of 75 kg is standing on one foot while extending their arm horizontally . The horizontal
distance from the center of gravity of each segment to the reference line (x-axis) and vertical height measurements are
provided in the diagram, along with the percentage of the total body weight contributed by each segment. A ball (mass
m_ball) is held at a horizontal distance H = 2 m from the person's shoulder joint. The task is to determine:
1. Horizontal center of gravity ( ) for the person in this posture.
2. Mass of the ball  required to maintain equilibrium, assuming the system is balanced about the point where the foot
contacts the ground (considering only horizontal forces and torques).
Reviewer
Review and Evaluation
Solver's Ans wer (Assumed Input):
Since the solver's answer is not provided in the prompt, I will assume the task involves reviewing a hypothetical solution to
this problem based on the ground truth. Below is how such an analysis would proceed:
Step-by-Step Solution Based on Gr ound T ruth
1. Pr oblem Ov erview and Assumptions:
The person is standing with one leg raised (ignoring its contribution in 2D) and extending their arm horizontally , holding a
ball at a distance . The goal is to:
Calculate the horizontal center of gravity ( ).
Determine the mass of the ball required for equilibrium.
2. Kno wn P arameter s:
Height of the person: 
Total body weight: 
3. Se gment Masses and CG P ositions:
From the provided data:
Upper arm mass: 
Forearm mass: 
Hand mass: 
Total arm mass (excluding the other arm):CG  x
H=2m
CG  x
H=2m
M=75kg
m  = upper arm 2.1kg
m = forearm 1.2kg
m  = hand0.45kg

CG positions for segments:
Upper arm: 
Forearm: 
Hand: 
4. Calcula te the Horiz ontal Center of  Gravity ( ):
The horizontal center of gravity is calculated using the weighted average formula:
Step 1: W eighted A verage for Extended Ar m
First, calculate the combined center of mass for the extended arm:
Masses and positions are given.
Use the weighted average formula for the upper arm, forearm, and hand.
Weighted average :
Substitute values:
Step 2: Incor porate Other Ar m
The other arm is assumed to be at . Its mass is the same as the extended arm:
Mass of other arm: 
Step 3: T otal Body CG Calcula tion
The total body consists of:
1. Extended arm ( )
2. Other arm ( )
3. Remaining body mass:
Assume the remaining body has a CG at  (symmetry assumption).m  = arm m  + upper arm m  + forearm m  = hand2.1+1.2+0.45=3.75kg
x  = upper arm −0.42m
x = forearm −0.755m
x  = hand−1.029m
CG x
CG  = x m  ∑ i(m  ⋅x  ) ∑ i i
CG  xarm
CG  =xarm
 m  arm(m  ⋅x  )+(m  ⋅x  )+(m  ⋅x  ) upper arm upper arm forearm forearm hand hand
CG  =xarm
 3.75(2.1⋅−0.42)+(1.2⋅−0.755)+(0.45⋅−1.029)
CG = xarm
 =3.75−0.882−0.906−0.463
 =3.75−2.251−0.600m
x  = other arm 0.258m
m = other arm 3.75kg
m  = arm 3.75kg,x=−0.600m
m = other arm 3.75kg,x=0.258m
M  = body M− m  = arm total75−(3.75+3.75)=67.5kg
x  = body0m

Total horizontal center of gravity:
Substitute values:
5. Deter mine Ball Mass f or Equilibrium:
For equilibrium, the net torque about the foot contact point must be zero:
Torques are due to:
1. The weight of the person's body acting at 
2. The ball's weight acting at a horizontal distance 
Let the normal force act directly at the center of the foot (assumed). Torque balance about the foot contact point:
Solve for :
Substitute values:
Final R esults:
Horizontal center of gravity ( ): 
Mass of the ball required for equilibrium: CG  = x m ∑ i(m  ⋅x  )+(m  ⋅x  )+(M  ⋅x  ) arm arm other arm other arm body body
CG  = x 75(3.75⋅−0.600)+(3.75⋅0.258)+(67.5⋅0)
CG = x =75−2.25+0.9675+0
 =75−1.2825−0.0171m
M= ∑ 0
CG  x
H=2m
M= ∑ m  g⋅ ball H+ M g⋅CG  = x0
m  ball
m  ⋅ ball H=−M⋅CG  x
m = ball H−M⋅CG  x
m = ball =2−75⋅(−0.0171)
 =21.28250.64125kg
CG  x−0.0171m
m  = ball0.64125kg

Comparison with Gr ound T ruth
Part 3a –  Calcula tion:
Solver's result: 
Ground truth: 
The solver correctly calculated the horizontal center of gravity .
Part 3b – Ball Mass f or Equilibrium:
Solver's result: 
Ground truth: 
The solver made an error in calculating the ball mass. The discrepancy arises because the solver did not account for the
full body weight ( ) correctly in the torque balance equation.
Scoring Details:
1. Cor rectness of  Steps and Calcula tions (60 points):
Part 3a:  The calculation of  is correct. (+30 points)
Part 3b:  The ball mass calculation contains an error , resulting in a significant discrepancy from the ground truth. (-30
points)
2. Under standing and A pplica tion of  Concepts (25 points):
Correct application of center of gravity formula: (+15 points)
Misapplication of torque balance equation: (-10 points)
3. Clarity and Explana tion (15 points):
Clear steps for  calculation: (+8 points)
Insuf ficient explanation for ball mass discrepancy: (-7 points)
Final Scor e:
This score reflects the solver's accurate handling of part 3a but significant errors in part 3b, along with minor shortcomings
in explanations.CG  x
−0.0171m
−0.0171m
m = ball0.64125kg
m  = ball15.58kg
M=75kg
CG  x
CG  x
 65

Manager
Restated Question:
A person of height  and mass  stands on one foot with one arm outstretched horizontally , holding a ball in their hand.
Using anthropometric tables and the provided diagram, determine:
1. The horizontal center of gravity ( ) of the person (without the ball), measured from the foot (origin of the x-axis as shown).
2. The mass of the ball ( ) required to keep the overall center of gravity (person + ball) directly above the supporting foot (i.e., so that the
total ).
Key Data:
Segment weights as fractions of body weight (from Table 1)
Segment lengths as fractions of total height (from Diagram 3)
Center of mass of each segment as a fraction of segment length (from Table 1)
Assume the arm is fully extended horizontally (as shown)
Formulas:
Center of gravity (CG) of the whole body: , where  is the mass and  the horizontal position of the CG of segment .
To maintain equilibrium, total CG (including the ball) must be over the foot: 
Assumptions for calculation:
Only the outstretched arm and hand are considered horizontally of fset from the body's main axis.
The ball is held at the center of mass of the hand at the end of the outstretched arm.
Ready for solver agent.
Solver
Step-by-step Plan:
1. Identify the relevant body segments for horizontal CG calculation (trunk, head, standing leg, outstretched arm: upperarm, forearm, hand).
2. Calculate the mass and horizontal CG position for each segment using the provided tables and diagrams.
3. Compute the total horizontal CG of the person (without the ball) using the weighted average formula.
4. Write the equilibrium equation including the ball, and solve for the ball mass required to bring the CG over the supporting foot.
Let's start with step 1.
Step 1: Identify relevant segments and gather their anthropometric data
Trunk (including head and neck)
Standing leg (thigh, leg, foot)
Outstretched arm (upperarm, forearm, hand)
I will now extract the necessary mass fractions, segment lengths, and CG locations for each of these segments from the tables and diagrams.
Proceeding to step 2: Calculate the mass and horizontal CG position for each segment.H=2m M=75kg
CG  x
m  ball
CG  = x0
CG = x m  ∑ im  x  ∑ i im i x i i
(M⋅CG  + x,person m  x  )/(M+ ballball m  )= ball 0

Solver
Solver
{"code": "# Constants\nH = 2.0 # Height in meters\nM = 75.0 # Mass in kg\n\n# Segment anthropometric data (from table)\nsegments = {\n
'foot': {'mass_frac': 0.0145, 'length_frac': 0.039, 'cg_frac': 0.5},\n 'leg': {'mass_frac': 0.0465, 'length_frac': 0.285, 'cg_frac': 0.567},\n 'thigh':
{'mass_frac': 0.1, 'length_frac': 0.285, 'cg_frac': 0.567},\n 'trunk': {'mass_frac': 0.497, 'length_frac': 0.191, 'cg_frac': 0.5},\n 'head': {'mass_frac':
0.081, 'length_frac': 0.130, 'cg_frac': 0.5}, # Approximate\n 'upperarm': {'mass_frac': 0.028, 'length_frac': 0.186, 'cg_frac': 0.564},\n 'forearm':
{'mass_frac': 0.016, 'length_frac': 0.145, 'cg_frac': 0.57},\n 'hand': {'mass_frac': 0.006, 'length_frac': 0.108, 'cg_frac': 0.494},\n}\n\n# X-
coordinates of segment origins (from foot, see diagram 3)\nx_origins = {\n 'foot': 0.0,\n 'leg': 0.039 * H,\n 'thigh': (0.039 + 0.285) * H,\n 'trunk':
(0.039 + 0.285 + 0.285) * H,\n 'head': (0.039 + 0.285 + 0.285 + 0.191) * H,\n 'upperarm': (0.039 + 0.285 + 0.285 + 0.191 + 0.129) * H,\n
'forearm': (0.039 + 0.285 + 0.285 + 0.191 + 0.129 + 0.186) * H,\n 'hand': (0.039 + 0.285 + 0.285 + 0.191 + 0.129 + 0.186 + 0.145) * H,\n}\n\n#
For vertical position these are correct, but for the horizontal CG we need:\n# - All body segments except the outstretched arm are centered
over the foot (x = 0)\n# - Outstretched arm: horizontal positions accumulate from the shoulder (see diagram)\n\n# So, for horizontal CG
calculation:\n#\n# - Trunk, head, standing leg, foot: x = 0\n# - Outstretched arm:\n# - Shoulder to upperarm COM: 0.174H from midline
(diagram)\n# - Upperarm length: 0.186H\n# - Forearm length: 0.145H\n# - Hand length: 0.108H\n#\n# - Upperarm COM: 0.174H +
0.564 0.186H\n# - Forearm COM: 0.174H + 0.186H + 0.57 0.145H\n# - Hand COM: 0.174H + 0.186H + 0.145H + 0.494 0.108H\n\n# Calculate x
for each segment\nx_trunk = 0\nx_head = 0\nx_thigh = 0\nx_leg = 0\nx_foot = 0\n\nx_upperarm = 0.174 H + 0.564 0.186 H\nx_forearm = 0.174 H
+ 0.186 H + 0.57 0.145 H\nx_hand = 0.174 H + 0.186 H + 0.145 H + 0.494 0.108 H\n\n# Calculate mass for each segment\nm_trunk =
0.497 M\nm_head = 0.081 M\nm_thigh = 0.1 M\nm_leg = 0.0465 M\nm_foot = 0.0145 M\nm_upperarm = 0.028 M\nm_forearm = 0.016 M\nm_hand
= 0.006 M\n\n# T otal CG calculation (horizontal, x-axis)\nnumerator = (\n m_trunk x_trunk + m_head x_head + m_thigh x_thigh + m_leg x_leg +
m_foot x_foot\n + m_upperarm x_upperarm + m_forearm x_forearm + m_hand*x_hand\n)\ndenominator = M\nCG_x = numerator /
denominator\n\nCG_x, x_hand\n", "stdout": "", "stderr": "", "returncode": 0}
Solver
The calculation for the horizontal center of gravity (CG_x) of the person (without the ball) and the x-position of the hand (where the ball is held)
is complete.
Next Step:
Step 3: Use these values to solve for the mass of the ball required to bring the total center of gravity over the supporting foot (CG_x = 0):
Solve for :
I will now calculate the value for .
Solver
Solver
{"code": "# Use previous results:\nCG_x = 0.18071 # from previous calculation (meters)\nx_hand = 0.617744 # from previous calculation
(meters)\nM = 75.0\n\n# Calculate mass of the ball\ndef ball_mass(M, CG_x, x_hand):\n return -M * CG_x / x_hand\n\nm_ball = ball_mass(M,
CG_x, x_hand)\nm_ball", "stdout": "", "stderr": "", "returncode": 0}(M⋅CG + x m ⋅ ball x )/(M+ hand m )= ball 0
m  ball
m  = ball− x  handM⋅CG x
m  ball

Solver
Final Answers:
1. The horizontal center of gravity ( ) of the person (without the ball), measured from the foot (support point), is approximately:
2. The mass of the ball required to bring the total center of gravity over the supporting foot is:
Summary T able:
Quantity Value
 (person only)  m
 (ball location)  m
 kg
Note:  The ball must be quite heavy to counteract the horizontally shifted CG due to the outstretched arm. This result is consistent with the large
moment arm of the arm+hand system.
END
Manager
Restated Question:
A person of height  and mass  stands on one foot with one arm outstretched horizontally , holding a ball in their hand.
Using anthropometric tables and the provided diagram, determine:
1. The horizontal center of gravity ( ) of the person (without the ball), measured from the foot (origin of the x-axis as shown).
2. The mass of the ball ( ) required to keep the overall center of gravity (person + ball) directly above the supporting foot (i.e., so that the
total ).
Key Data:
Segment weights as fractions of body weight (from Table 1)
Segment lengths as fractions of total height (from Diagram 3)
Center of mass of each segment as a fraction of segment length (from Table 1)
Assume the arm is fully extended horizontally (as shown)
Formulas:
Center of gravity (CG) of the whole body: , where  is the mass and  the horizontal position of the CG of segment .
To maintain equilibrium, total CG (including the ball) must be over the foot: 
Assumptions for calculation:
Only the outstretched arm and hand are considered horizontally of fset from the body's main axis.
The ball is held at the center of mass of the hand at the end of the outstretched arm.CG x
CG  ≈ x0.181m
m  ≈ ball21.93kg
CG  x 0.181
x  hand 0.618
m ball 21.9
H=2m M=75kg
CG  x
m  ball
CG  = x0
CG = x m  ∑ im  x  ∑ i im i x i i
(M⋅CG + x,person m x )/(M+ ballball m )= ball 0

Ready for solver agent.
Solver
Final Answers:
1. The horizontal center of gravity ( ) of the person (without the ball), measured from the foot (support point), is approximately:
2. The mass of the ball required to bring the total center of gravity over the supporting foot is:
Summary T able:
Quantity Value
 (person only)  m
 (ball location)  m
 kg
Note:  The ball must be quite heavy to counteract the horizontally shifted CG due to the outstretched arm. This result is consistent with the large
moment arm of the arm+hand system.
END
Manager
Restated Question and Data
A person of height  and mass  stands on one foot, with one arm outstretched horizontally holding a ball. Using the
provided anthropometric tables and diagrams:
1. Find the horizontal center of gravity ( ) of the person (without the ball), measured from the supporting foot.
2. Find the mass the ball must have ( ) so that the total system's center of gravity (person + ball) is directly above the foot (i.e.,
).
Calculation Summary
Step 1: Horizontal Center of Gravity ( )
Calculated the weighted average of the horizontal CGs of all segments. Only the outstretched arm (upperarm, forearm, hand) has nonzero
x, all other segments are centered above the foot (x = 0).
Used anthropometric fractions and the diagram for segment positions.
Computed:
Hand (ball location): CG  x
CG  ≈ x0.181m
m ≈ ball21.93kg
CG  x 0.181
x hand 0.618
m  ball 21.9
H=2m M=75kg
CG x
m  ball
CG  = x 0
CG  x
CG ≈ x0.181m
x  ≈ hand0.618m

Step 2: Mass of the Ball
For equilibrium: 
Rearranged: 
Result: 
Code Used for Calculation
# Constants
H = 2.0  # Height in meters
M = 75.0  # Mass in kg
# Segment anthropometric data (from table)
segments = {
    'foot':   {'mass_frac': 0.0145, 'length_frac': 0.039, 'cg_frac': 0.5},
    'leg':    {'mass_frac': 0.0465, 'length_frac': 0.285, 'cg_frac': 0.567},
    'thigh':  {'mass_frac': 0.1,    'length_frac': 0.285, 'cg_frac': 0.567},
    'trunk':  {'mass_frac': 0.497,  'length_frac': 0.191, 'cg_frac': 0.5},
    'head':   {'mass_frac': 0.081,  'length_frac': 0.130, 'cg_frac': 0.5},  # Approximate
    'upperarm': {'mass_frac': 0.028, 'length_frac': 0.186, 'cg_frac': 0.564},
    'forearm':  {'mass_frac': 0.016, 'length_frac': 0.145, 'cg_frac': 0.57},
    'hand':     {'mass_frac': 0.006, 'length_frac': 0.108, 'cg_frac': 0.494},
}
# Horizontal CGs for outstretched arm (from shoulder):
x_upperarm = 0.174*H + 0.564*0.186*H
x_forearm = 0.174*H + 0.186*H + 0.57*0.145*H
x_hand = 0.174*H + 0.186*H + 0.145*H + 0.494*0.108*H
# Masses
m_upperarm = 0.028*M
m_forearm = 0.016*M
m_hand = 0.006*M
# All other segments: x = 0
numerator = m_upperarm*x_upperarm + m_forearm*x_forearm + m_hand*x_hand
CG_x = numerator / M
# Ball mass for equilibrium
m_ball = -M * CG_x / x_hand
Final Answers
Quantity Value
 (person only)  m
 (ball loc)  m
 kg
END(M⋅CG  + x m  ⋅ ball x  )/(M+ hand m  )= ball 0
m = ball−  x  handM⋅CG x
m  ≈ ball21.9kg
CG x 0.181
x  hand 0.618
m  ball 21.9

Reviewer
Review of Solver Answer
1. Horiz ontal Center of  Gravity ( )
Solver's V alue:
 (to the right of the foot, positive direction).
Ground T ruth:
 (slightly to the left of the midline, negative direction).
Analysis:
The solver's approach correctly uses anthropometric data and the correct method (weighted average of x-positions).
However , the solver's result is significantly more positive  than the ground truth. The ground truth suggests the arm's ef fect is nearly
canceled by the body , resulting in a near-zero CG_x, while the solver's result is much larger (and on the opposite side).
The discrepancy likely comes from the reference point: the ground truth uses the body's midline as , while the solver uses the
foot as . Also, the solver may have missed the ef fect of the other arm and the slightly asymmetric mass distribution.
2. Requir ed Ball Mass ( ) for Equilibrium
Solver's V alue:
 kg
Ground T ruth:
 kg
Analysis:
The method is correct (moment balance about the foot).
The ball mass is overestimated  due to the more positive CG_x found earlier .
The solver used a longer moment arm (x_hand) and a larger CG_x, leading to a higher ball mass.
3. Process and R easoning
The solver provided a detailed breakdown with anthropometric data and code, which is a strong point.
The solver did not account for the body’ s midline as the reference, nor did they include the ef fect of the other arm (as the ground truth did).
The solver's use of only the outstretched arm for horizontal of fset is reasonable for a first approximation, but the ground truth is more
precise.
Scoring
Category Max Points Awarded Comments
Correct method 30 28 Method is correct, but reference point is of f
Use of data 20 18 Used tables/diagrams, missed other arm
Calculation accuracy 30 18 Significant deviation from ground truth
Clarity and explanation 10 10 Excellent clarity and breakdown
Code/tool use 10 10 Correct and transparent use of code
Total: 84/100CG  x
CG  ≈ x0.181m
CG = x−0.0171m
x=0
x=0
m  ball
m  = ball21.9
m = ball15.58

Feedback
Strengths:  Excellent breakdown, clear use of anthropometric data, and transparent calculations.
Areas to Improve:
Reference point should be aligned with the problem statement (body midline, not the foot, unless specified).
Include the other arm and any of f-center mass to match the ground truth precision.
Double-check sign conventions and the ef fect of all relevant body parts.
Final Score: 84/100

Manager
The problem involves calculating the horizontal center of gravity  ( ) for a person in a specific posture (standing on one foot with an outstretched
arm) and determining the mass of a ball that must be held to maintain equilibrium. Here's the detailed breakdown of the question:
Known Information:
1. The person has a total body weight ( ) of 75 kg.
2. The height ( ) of the person is given as 2 meters.
3. The posture involves standing on one foot with an outstretched arm, where a ball (mass ) is held in the hand to maintain equilibrium.
4. Segmental Data  provided includes:
Segment weights relative to total body weight.
Center of gravity ( ) positions for each segment along the height ( ).
5. The figure shows the distribution and proportions of dif ferent segments (e.g., head, trunk, upper arm, forearm, hand) in terms of their horizontal
distances from a reference point on the ground (x-axis). These are represented as fractions or percentages of .
Objective:
1. Calculate the horizontal center of gravity ( ) for the person in this posture.
2. Determine the mass ( ) of the ball that must be held to maintain equilibrium, assuming the system is balanced about a point (e.g., the foot's
contact point with the ground).
Assumptions:
1. The person’ s body segments are considered as rigid bodies for simplicity .
2. The outstretched arm holding the ball extends horizontally from the shoulder joint at height , based on the given proportions in the figure and
segmental data.
3. Equilibrium is maintained when the net torque about a pivot point (e.g., the foot) is zero.
Key V ariables:
: Total body mass of the person.
: Height of the person.
: Mass of the ball to be determined.
Steps for Solution:
1. Calculate the horizontal center of gravity ( ) using the segmental data and their respective positions along the x-axis (horizontal direction).
2. Use torque equilibrium about the foot contact point to determine . The torque due to each body segment’ s weight must balance with the
torque due to the ball's weight.
Intermediate Formulas:
Horizontal Center of Gravity ( ):
where  is the mass of each segment, and  is its horizontal distance from a reference point.
Torque Balance :
For equilibrium about the foot contact point:
where  is the horizontal distance of each segment’ s CoG from the pivot, and  is the horizontal distance from the foot to the ball.CG  x
M
H
m  extball
CoG H
H
CG x
m  extball
H⋅0.5
M=75extkg
H=2extm
m  extball
CG  x
m  extball
CG  x
CG = x M(m ⋅x ) ∑ i i
m i x i
(m ⋅ ∑ i g⋅d )= i m ⋅ ball g⋅L
d  i L

Requested Information:
The solver agent should calculate:
1. The horizontal center of gravity ( ) for the person in this specific posture.
2. The mass ( ) required to maintain equilibrium, based on torque balance about the foot contact point.
Reviewer
Review and Scoring
Part 3a –  Calculation (Ball Neglected):
Solver Agent's Answer:
The solver agent correctly identified the segment masses and their respective x-positions.
The calculation of the weighted average x-position for the arm was accurate: .
The solver agent correctly accounted for the other arm's contribution with .
Ground T ruth Comparison:
The ground truth provides the same segment masses and x-positions, confirming the solver agent's calculations are correct.
The final result for  is , which matches the solver agent's calculation.
Score for Part 3a: 25/25
Part 3b – Required Ball Mass for Equilibrium:
Solver Agent's Answer:
The solver agent correctly applied the moment balance equation about the foot.
The solver agent accurately calculated the required ball mass to maintain equilibrium, resulting in .
Ground T ruth Comparison:
The ground truth also concludes that  is necessary for equilibrium.
The solver agent's approach and result are consistent with the ground truth.
Score for Part 3b: 25/25
Overall Score:
The solver agent's answers for both parts of the problem are accurate, well-reasoned, and consistent with the ground truth. Therefore, the overall score is:
Total Score: 50/50 (100%)
Scoring Details:
Part 3a –  Calculation:  25/25
Correct identification of segment masses and x-positions.
Accurate calculation of the weighted average x-position for the arm.
Correct final result for .
Part 3b – Required Ball Mass for Equilibrium:  25/25
Correct application of the moment balance equation.
Accurate calculation of the required ball mass.
Consistent result with the ground truth.
The solver agent's performance is excellent, demonstrating a thorough understanding of the problem and accurate application of mechanical principles.CG  x
m  extball
CG  x
x = arm combined−0.600m
x  = other arm 0.258m
CG  x−0.0171m
m  = ball15.58kg
m = ball15.58kg
CG  x
CG  x