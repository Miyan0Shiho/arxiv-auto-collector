# RISE: Reasoning Enhancement via Iterative Self-Exploration in Multi-hop Question Answering

**Authors**: Bolei He, Xinran He, Mengke Chen, Xianwei Xue, Ying Zhu, Zhenhua Ling

**Published**: 2025-05-28 03:48:28

**PDF URL**: [http://arxiv.org/pdf/2505.21940v1](http://arxiv.org/pdf/2505.21940v1)

## Abstract
Large Language Models (LLMs) excel in many areas but continue to face
challenges with complex reasoning tasks, such as Multi-Hop Question Answering
(MHQA). MHQA requires integrating evidence from diverse sources while managing
intricate logical dependencies, often leads to errors in reasoning.
Retrieval-Augmented Generation (RAG), widely employed in MHQA tasks, faces
challenges in effectively filtering noisy data and retrieving all necessary
evidence, thereby limiting its effectiveness in addressing MHQA challenges. To
address these challenges, we propose RISE:Reasoning Enhancement via Iterative
Self-Exploration, a novel framework designed to enhance models' reasoning
capability through iterative self-exploration. Specifically, RISE involves
three key steps in addressing MHQA tasks: question decomposition,
retrieve-then-read, and self-critique. By leveraging continuous
self-exploration, RISE identifies accurate reasoning paths, iteratively
self-improving the model's capability to integrate evidence, maintain logical
consistency, and enhance performance in MHQA tasks. Extensive experiments on
multiple MHQA benchmarks demonstrate that RISE significantly improves reasoning
accuracy and task performance.

## Full Text


<!-- PDF content starts -->

arXiv:2505.21940v1  [cs.CL]  28 May 2025RISE: Reasoning Enhancement via Iterative Self-Exploration in Multi-hop
Question Answering
Bolei He1,2*Xinran He2*Mengke Chen2Xianwei Xue2
Ying Zhu2Zhen-Hua Ling1†
1University of Science and Technology of China, Hefei, China
2Baidu Inc., Beijing, China
hebl@mail.ustc.edu.cn ,zhling@ustc.edu.cn ,
{hexinran, xuexianwei, zhuying11}@baidu.com ,
anthreebody@gmail.com ‘
Abstract
Large Language Models (LLMs) excel in many
areas but continue to face challenges with com-
plex reasoning tasks, such as Multi-Hop Ques-
tion Answering (MHQA). MHQA requires in-
tegrating evidence from diverse sources while
managing intricate logical dependencies, of-
ten leads to errors in reasoning. Retrieval-
Augmented Generation (RAG), widely em-
ployed in MHQA tasks, faces challenges in
effectively filtering noisy data and retrieving
all necessary evidence, thereby limiting its ef-
fectiveness in addressing MHQA challenges.
To address these challenges, we propose RISE:
Reasoning Enhancement via Iterative Self-
Exploration , a novel framework designed to
enhance models’ reasoning capability through
iterative self-exploration. Specifically, RISE
involves three key steps in addressing MHQA
tasks: question decomposition, retrieve-then-
read, and self-critique. By leveraging continu-
ous self-exploration, RISE identifies accurate
reasoning paths, iteratively self-improving the
model’s capability to integrate evidence, main-
tain logical consistency, and enhance perfor-
mance in MHQA tasks. Extensive experiments
on multiple MHQA benchmarks demonstrate
that RISE significantly improves reasoning ac-
curacy and task performance.
1 Introduction
Large language models (LLMs) demonstrate out-
standing capabilities in natural language under-
standing and generation (Brown et al., 2020; Zhang
et al., 2022; Zeng et al., 2022; Chowdhery et al.,
2023; Touvron et al., 2023). However, LLMs still
face challenges with complex Multi-Hop Question
Answering (MHQA) tasks. MHQA requires mod-
els to integrate evidence from multiple sources
and manage intricate logical relationships. This
involves both retrieving and combining various
*Equal contributions.
†Corresponding author.
Sub Question: WhenwasTheBookOfElireleased?Sub Answer:2015Sub Answer:1990
Final Answer (ERROR): According to the evidences provided, The Book of Eli was released in 2009, while Fire Birds was released on Blu-ray in 2015. Therefore, Fire Birds was released more recently than The Book of Eli.Final Answer(CORRECT): Accordingtotheevidencesprovided,TheBookofEliwasreleasedin2010,whileFireBirdswasreleasedin1990.Therefore,TheBookofEliwasreleasedmorerecentlythanFireBirds.Sub Answer:2009Sub Answer:2010……Question:Whichfilmwasreleasedmorerecently,TheBookOfEliorFireBirds?Reference 1: Fire Birds released on Blu-raythis week on Kino Lorber’ s Studio Classics line. (A previous edition, now out-of-print, was released by Mill Creek in 2015).Reference 2: … Fire Birds Released on May 25, 1990. Directed by David Green. … Filmed on location in Texas and Arizona. Released on video in 1990.…Reference N: …Sub Question:What is the production year of TheBookOfEli?Reference 1: ... Reference 2: …Principal photography began in February 2009and took place in New Mexico. ……Reference N: …
Reference 1: ... The Book of Eli is a2010American post-apocalyptic, dystopia, neo ...Reference 2: ……Reference N: …
Sub Question: WhenwasFireBirdsreleased?Decomposition
Decomposition
GenerationGenerationRetrieve
RetrieveRetrieveDecompositionGenerationGeneration
Figure 1: The upper part of the figure (blue) illustrates
an Evidence Aggregation Error, where the Blu-ray re-
lease year of Fire Birds (2015) is mistaken for its theatri-
cal release year. The lower part (green and red) shows
a Reasoning Decomposition Error. The incorrect path
formulates the sub-question as the production year of
The Book of Eli (2009) instead of its release year (2010).
pieces of evidence and constructing coherent rea-
soning chains. Prompt-based methods, such as
Chain-of-Thought (CoT) (Wei et al., 2022b; Wang
et al., 2023a; Yu et al., 2023), are employed to
address MHQA by split complex problems into
smaller, thereby harnessing the reasoning poten-
tial of LLMs. However, these methods often lack
external knowledge, resulting in key evidence be-
ing overlooked and generate hallucinations (Rawte
et al., 2023; Ji et al., 2023; Ye et al., 2023).
Retrieval-Augmented Generation (RAG) meth-
ods (Guu et al., 2020; Lewis et al., 2020; Izac-
ard et al., 2022; Nakano et al., 2021; Asai et al.,
1

2023; Ma et al., 2023; Yu et al., 2024; Shi et al.,
2024a) have been proposed to address the afore-
mentioned challenges. By incorporating external
knowledge, RAG effectively mitigates hallucina-
tion phenomena and achieves significant results in
MHQA tasks through multiple retrievals. However,
RAG is constrained by the performance of the re-
trievers, inevitably introducing noise. Additionally,
the multi-round retrieval process may lead to error
propagation, resulting in two main types of errors:
Evidence Aggregation Errors andReasoning De-
composition Errors . As illustrated in Figure 1, Ev-
idence Aggregation Errors occur when the model
fails to accurately integrate evidence from multiple
evidences, leading to hallucinations. Reasoning
Decomposition Errors arise when problem decom-
position phase generates sub-questions that do not
align with original question’s intent. These issues
are particularly pronounced in smaller models with
weaker reasoning capabilities.
Distillation and fine-tuning (Uesato et al., 2022;
Luo et al., 2023; Shridhar et al., 2023) effectively
enhance the reasoning capabilities of LLMs by
leveraging large-scale models or high-quality, man-
ually annotated data to improve performance. How-
ever, biases brought by human subjective anno-
tations may undermine the performance of fine-
tuning (Casper et al., 2023; Lightman et al., 2023),
and these methods are costly, requiring substan-
tial human or computational resources. Mean-
while, self-iteration methods (Yuan et al., 2024;
Wang et al., 2024; Madaan et al., 2024) demon-
strate tremendous potential in complex reasoning
tasks. Unlike approaches that depend on large-
scale models and manual annotations, self-iteration
methods enable models to generate and learn from
their own data, achieving outstanding results in
complex tasks such as code generation and intel-
ligent agents (Jiang et al., 2023; Ni et al., 2024;
Qiao et al., 2024). Nevertheless, research on com-
bination self-iteration methods with RAG remains
limited. The integration of these two approaches
has the potential to improve performance in com-
plex reasoning tasks and leads to cost reduction.
In this paper, we introduce an innovative frame-
work, RISE (Reasoning Enhancement via Iterative
Self-Exploration), which combines the paradigms
of RAG and self-iteration to address key chal-
lenges in MHQA tasks. Specifically, RISE de-
fines three core actions: question decomposition,
retrieve-then-read, and self-critique. By repeatedly
executing these actions, the model autonomouslyexplores accurate reasoning paths for problems.
During this process, RISE accumulates experience
datasets for the three actions and updates the model
based on this experience. Through multiple iter-
ations, RISE significantly enhances the model’s
reasoning capabilities in MHQA tasks. Experi-
mental results demonstrate that RISE outperforms
baseline methods on several MHQA benchmark
datasets, strongly validating its effectiveness in
solving MHQA tasks while offering lower usage
costs. Our main contributions are as follows:
•We propose RISE, which combines RAG and
self-iteration to address two key challenges in
MHQA tasks: Evidence Aggregation Errors
and Reasoning Decomposition Errors.
•We design self-exploration mechanism, con-
verts MHQA in RAG into multi-objective op-
timization problem, thus improving model’s
reasoning capability and reducing costs.
•We integrate self-iteration paradigm with
RAG, bridging gap in applying self-iteration
strategies within MHQA RAG framework.
2 Methods
2.1 Overview
In this section, we provide a concise description
ofRISE , focusing on its algorithmic process. As
shown in algorithm 1, RISE begins with a seed
question set Q0and an initial model M0. The
model iteratively performs self-exploration for each
question q∈ Q i, with details presented in Sec-
tion 2.2. The exploration results are stored as his-
torical experience H. After completing the explo-
ration for all questions, the accumulated experi-
ences optimize the model through multi-objective
training, yielding an enhanced model Mi+1. Sub-
sequently, Mi+1expands the question set based on
the previous seed questions Qi, generating Qi+1
to initiate the next round of exploration. This
self-iterative process enables RISE to continuously
improve capabilities without external supervision,
leveraging the model’s intrinsic potential.
2.2 Self-Exploration Mechanism
The self-exploration mechanism enables the model
to address complex problems through iterative rea-
soning, comprising three core actions: question de-
composition, retrieve-then-read, and self-critique.
2

𝑄𝑖 𝑀𝑖𝑀𝑖+1𝑄𝑖+1
b) Self-Iterative  Optimization
𝑹𝒂𝒈{𝒔𝒖𝒃𝒒,𝒓,𝒔𝒖𝒃𝒂}𝟏𝟏
𝑹𝒂𝒈𝟏𝟏𝑹𝒂𝒈𝟏𝟏 𝑹𝒂𝒈𝟏𝟏 𝑹𝒂𝒈𝟏𝟏
𝑹𝒂𝒈𝟐𝟐 𝑹𝒂𝒈𝟐𝟐 𝑹𝒂𝒈𝟐𝟐
𝑹𝒂𝒈𝟑𝟏 𝑹𝒂𝒈𝟑𝟐𝑹𝒂𝒈𝟏𝟐 𝑹𝒂𝒈𝟏𝟏
𝑹𝒂𝒈𝟐𝟏𝒒𝟎 𝒒𝟎 𝒒𝟎 𝒒𝟎 𝒒𝟎 𝒒𝟎
{𝒔𝒖𝒃𝑸,𝑹,𝑨}𝟏𝟏 {𝒔𝒖𝒃𝑸,𝑹,𝑨}𝟐𝟐 {𝒔𝒖𝒃𝑸,𝑹,𝑨}𝟑𝟐{𝒒𝟎→𝒒𝟏𝟏→𝒒𝟐𝟐→𝒒𝟑𝟏}{𝒒𝟎→𝒒𝟏𝟏→𝒒𝟐𝟐}{𝒒𝟎→𝒒𝟏𝟏}𝒒𝟎
Seed Questions from 𝑄𝑖Updating
𝑻𝒊 𝑻𝒊+𝟏
True
True
TrueFalse
False
FalseSelf Critique  DataQuestion Decomposition  Data  
                        
                     
                        
                     
Retrieve -then-read Data Multi -Objective Optimization
Question  Expansion
𝑀𝑖+1In-context learning With 
Multi -Roundsa) Self-Exploration
𝑹𝒂𝒈{𝒔𝒖𝒃𝒒,𝒓,𝒔𝒖𝒃𝒂}𝟏𝟐
𝑹𝒂𝒈{𝒔𝒖𝒃𝒒,𝒓,𝒔𝒖𝒃𝒂}𝟐𝟏 𝑹𝒂𝒈{𝒔𝒖𝒃𝒒,𝒓,𝒔𝒖𝒃𝒂}𝟐𝟐
𝑹𝒂𝒈{𝒔𝒖𝒃𝒒,𝒓,𝒔𝒖𝒃𝒂}𝟑𝟏 𝑹𝒂𝒈{𝒔𝒖𝒃𝒒,𝒓,𝒔𝒖𝒃𝒂}𝟑𝟐
𝒂𝟎FalseFalse
FalseTrueTrue
True
                        
                     q𝒒𝟏𝒒𝟐𝒒𝟑𝒒𝟒
Figure 2: A complete iteration cycle in RISE. a) Self-Exploration : Model Midecomposes complex questions q0
into simpler sub-questions, generates sub-answers via retrieve-then-read, and evaluates their validity, leading to a
final answer a0. Interactions are stored as historical data D.b) Iterative Optimization : RISE optimizes model Mi
using historical data Dto create an enhanced model Mi+1, which generates new questions Qi+1for the next cycle.
These actions collectively form a structured explo-
ration pathway, with the resulting information col-
lected as historical data to support the model’s self-
improvement in complex problem-solving. The
related prompts are provided in Appendix A.1.1.
Question Decomposition. In this task, the
model incrementally decomposes the initial com-
plex question into fine-grained sub-questions.
At the t-th exploration node, the model uses
previously explored sub-questions and answers
as historical information, denoted as H=
(subq 1, suba 1),···,(subq t−1, suba t−1). The
original question q0is combined with Hand input
into model Mto generate the next sub-question.
The model ends exploration by generating the final
answer if the historical information suffices for q0.
Formally, this process is represented as Formula 1:
subq t=Fd(M,H, q0) (1)
a0=M(q0,H),ifHis sufficient. (2)
Additionally, all decomposition steps, includ-
ing the original question and generated sub-
questions, are recorded to form the dataset Dd=
{q0,H, subq}np
i=1	Nq. By leveraging this fine-
grained and structured dataset, the model learns
the logical dependencies and relationships betweenquestions and sub-questions, thereby improving its
ability to decompose complex problems.
Retrieve-then-Read. This task follows the stan-
dard RAG paradigm to provide evidence-based an-
swers for sub-questions. At the t-th exploration
node, a retriever obtains relevant fragments rt
based on the sub-question, and model Mgener-
ates answer using the retrieved evidence:
suba t=Fg(M, subqi
t, rt) (3)
Each sub-question and its answer form an ex-
ploration node (subq i, suba i), added to the histor-
ical information Ht+1=Ht∪ {(subq i, suba i)}.
All nodes are recorded to construct the dataset
Dr=
(subq, r, suba )np
i=1	Nq. Training on this
dataset helps the model integrate evidence into rea-
soning, improving answer accuracy and reliability.
Self-Critique. In this task, the model’s critique
capability is incorporated into the exploration pro-
cess. Specifically, after completing the question
decomposition and retrieve-then-read tasks at the
t-th exploration node, the model Mcritiques the
relevance and utility of the node for solving the
original question and outputs a binary decision. If
critiqued as True, it is retained, and exploration
proceeds to the next step. If critiqued as False , the
node is temporarily stored, and the process reverts
3

Algorithm 1 RISE
Input: Seed question set Q0, Initial model M0,
Retriever R, Maximum nodes Nmax= 20
1:Initialize: History H ← ∅ , Model index i←0
2:while True do
3: foreach question q∈ Qido
4: n←0 ▷Start self-exploration.
5: while Mi(q,H) =More information
needed andn < N maxdo
6: subq←Mi(H)
7: r←R(subq)
8: suba←Mi(subq, r )
9: σ←Mi(subq, suba )
10: ifσ= 1then
11: Add(subq, suba )toH
12: end if
13: n←n+ 1
14: end while
15: a←Mi(H)
16: σfinal←Mi(a, q,H)
17: end for ▷End self-exploration.
18: Mi+1←Multi-Objective Train (Mi,H)
19: Qi+1←Qustion Expansion (Mi+1,Qi)
20: i←i+ 1
21:end while
Output: Final model Mi
to the preceding valid node to generate a new node.
This process is formalized in Formula 4:
σt=Fc(M, subq t, suba t), σ t∈ {0,1}(4)
Similarly, we record critique historical infor-
mation and then construct the dataset Dc=
{⟨subq, suba ⟩, σ}np
i=1	Nq, to enhance the self-
critique capabilities of the model, ensuring logical
consistency and relevance within the exploration.
2.3 Self-Iterative Optimization
RISE is a self-iterative fine-tuning framework that
optimizes the model in each training round based
on data generated by the model itself, gradually en-
hancing its generalization ability and reasoning per-
formance. Through a closed-loop iteration of data
generation and model training, RISE effectively un-
covers the model’s reasoning potential in complex
tasks, driving continuous self-improvement.
Initialization. We initially use randomly sampled
question set Q0from the training sets of three
tasks—2Wiki, HotpotQA, and Musique, with 800weights ratios
α 1 1 4 1 1
β 1 4 1 2 3
γ 4 1 1 3 2
Accuracy 41.55 40.32 41.94 38.71 37.63
α 2 2 3 3 2
β 1 3 1 2 2
γ 3 1 2 1 2
Accuracy 39.78 40.32 43.01 41.13 44.27
Table 1: Performance on the 2WikiMultiHopQA dataset
under varying weight ratios of α,β, and γ.
examples from each. Subsequently, we employ
self-exploration mechanism to automatically ex-
pand and collect the three types of dataset. Subse-
quently, we employ self-exploration mechanism to
automatically expand and collect Dd,Dr, andDc
datasets for subsequent model training.
Multi-Objective Optimization. These three
datasets, Dd,Dr, andDc, are interconnected, with
sample sizes ranging from 2k to 8k (detailed statis-
tics are provided in Appendix 5). We believe that
joint training facilitates complementary learning
and enhances model capabilities. Therefore, we
adopt a multi-objective optimization approach to
integrate the objectives of different tasks into a uni-
fied optimization goal. The effectiveness of this
approach is validated through ablation study at Sec-
tion 4.3. The overall loss function is defined as
follows Formula 5:
L=αLd+βLr+γLc (5)
Here, α,β, and γrepresent task weights, deter-
mined by the proportion of each capability in the
training data. Specifically, Ld(Formula 6) models
the autoregressive loss for sub-question generation,
Lr(Formula 7) models the loss for sub-answer gen-
eration conditioned on retrieved context, and Lc
(Formula 8) denotes a binary classification loss for
self-critique judgments, where True and False rep-
resent the predicted probabilities of the respective
tokens.
Ld=−X
ilogP(subqi|q0,subq<i) (6)
Lr=−X
ilogP(suba i|subqi, ri) (7)
Lc=−σlogP(True|subqi,suba i)
−(1−σ) logP(False |subqi,suba i)(8)
0P(True ) =P(Token (True ))
P(Token (True ))+P(Token (False ))and
P(False )accordingly.
4

Meanwhile, Experimental results (Table 1) in-
dicate that the weights assigned to different tasks
have an impact on model performance, and appro-
priate weight adjustments facilitate fine-grained
performance optimization. Notably, to avoid po-
tential overfitting caused by manual weight tuning
which may affect the final evaluation we do not
perform any fine-tuning of the task weights in our
experiments. Instead, we adopt a uniform weight-
ing strategy, assigning equal weights to all tasks.
Question Expansion After completing multi-
objective optimization, we use the questions gener-
ated in the previous iteration as seed data for Mi+1
to perform question expansion, thereby acquiring
training data for the next iteration. This method is
inspired by (Wang et al., 2023c), leveraging multi-
round in-context learning to ensure the diversity
and richness of the newly generated questions. De-
tailed information about the question expansion
prompts is provided in the appendix Figure 10.
3 Experiments Setup
Datasets: For the main experiments, we use three
QA datasets: 2WikiMultiHopQA (2WIKI) (Ho
et al., 2020), HotpotQA (Hotpot) (Yang et al.,
2018), and MuSiQue (MSQ) (Trivedi et al., 2022),
which provide diverse reasoning challenges to eval-
uate the robustness of our framework. Addition-
ally, for the analysis experiments, we include Nat-
ural Questions (NQ) (Kwiatkowski et al., 2019),
Web Questions (WebQ) (Berant et al., 2013) and
TriviaQA (Joshi et al., 2017) to assess the model’s
performance on open-domain Question Answering
tasks, further extending the evaluation scope.
Models and Methods: In our experiments, we
use LLaMA-3.1-8B (Dubey et al., 2024) as the
base model for our method in main experiments.
Similarly, most of the reproduced methods are
also implemented using LLaMA-3.1-8B. Addi-
tionally, based on the characteristics of MHQA
tasks, we select and reproduce a variety of methods,
categorized into non-retrieval-based methods and
retrieval-based methods. Non-retrieval-based meth-
ods include Naive LLM (LLaMA-3.1-8B, GPT-3.5-
turbo), CoT (Wei et al., 2022b), CoT-SC (Wang
et al., 2023a) and GenRead (Yu et al., 2023), while
the retrieval-based methods consist of Naive RAG,
Self-Ask (Press et al., 2023), WebGLM (Liu et al.,
2023), Self-RAG (Asai et al., 2023), RRR (Ma
et al., 2023), and GenGround (Shi et al., 2024a). In
10203040Accuracy (%)
(a): Accuracy per Iteration
round0 round1 round2 round3 round4
Iteration Rounds2345Reasoning Length
(b): Reasoning Length per Iteration
2wiki
hotpot
msqFigure 3: Changes in model accuracy (a) and reasoning
length (b) across datasets. Accuracy consistently im-
proves across datasets, while reasoning length, despite
some fluctuations, shows an overall decreasing trend.
the analysis experiments, we employ GPT-4o1as
the evaluation model, combining subjective analy-
sis with specific metrics to comprehensively assess
model performance.
Retrieval: We adopt a two-stage retrieval frame-
work (Liu et al., 2023), consisting of coarse-grained
web search (via Chrome) followed by fine-grained
LLM-enhanced retrieval. We consistently use the
same retrieval method to reproduce results for other
approaches that incorporate retrievers.
Evaluation Metrics: We assess model perfor-
mance primarily using standard metrics for ques-
tion answering tasks: Accuracy (Acc), F1 score
(F1), and Exact Match (EM), which together pro-
vide a comprehensive measure of answer correct-
ness and completeness. In addition to answer qual-
ity, we evaluate the generated reasoning chains by
examining their length as an objective measure of
complexity. Furthermore, we conduct a qualita-
tive assessment of the reasoning chains from four
subjective perspectives: conciseness, rationality,
sequencing, and goal orientation.
We provide comprehensive experimental details
in Appendix A.2, including implementation details,
datasets, and other relevant information.
1We use GPT models accessed via the OpenAI API: https:
//openai.com/api/ .
5

Method ModelMHQA SHQA
2WIKI Hotpot MSQ NQ WebQ Trival
w/o retrieval
Naive LLM LLaMA-3.1-8B 35.90 27.30 11.30 57.50 61.25 71.50
GPT-3.5-turbo 47.10 41.50 19.10 57.25 58.30 80.25
CoT (Wei et al., 2022b) LLaMA-3.1-8B 43.00 34.60 16.20 56.75 62.00 71.75
CoT-SC*(Wang et al., 2023a) GPT-3.5-turbo 21.00 30.60 8.90 39.75 38.50 79.25
GenRead (Yu et al., 2023) LLaMA-3.1-8B 20.00 28.40 10.30 46.25 43.00 67.00
w retrieval
Naive RAG LLaMA-3.1-8B 43.50 37.50 15.00 51.75 57.75 73.75
Self-Ask (Press et al., 2023) LLaMA-3.1-8B 22.90 29.70 12.70 44.50 46.50 66.00
WebGLM (Liu et al., 2023) LLaMA-3.1-8B 37.60 36.50 13.30 55.75 62.25 73.75
Self-RAG*(Asai et al., 2023) LLaMA2-7B 32.00 30.20 8.00 54.50 60.25 74.25
LLaMA2-13B 30.80 29.40 8.80 55.25 58.75 75.00
RRR (Ma et al., 2023) LLaMA-3.1-8B 23.70 11.80 5.40 22.50 30.50 36.75
GPT-3.5-turbo 28.20 29.70 8.70 57.75 56.50 80.50
GenGround (Shi et al., 2024a) LLaMA-3.1-8B 37.90 36.10 17.80 48.50 44.50 75.25
RISE(Ours) LLaMA-3.1-8B 49.40 40.50 21.70 59.50 62.50 80.25
Table 2: Comparison of RISE’s accuracy with other methods on 2WikiMultiHopQA, HotpotQA, MuSiQue, Natural
Questions, Web Questions, and TriviaQA. Methods marked with an asterisk (*) involve specific considerations:
CoT-SC uses GPT-3.5 due to LLaMA-3.1’s instruction-following limitations, and Self-RAG employs public model
weights as its dataset is unavailable. Other methods use LLaMA-3.1-8B. denote Prompting-based Methods , while
denote Finetuning-based Methods . Due to space constraints, F1andEMmetrics are in Appendix A.4.
4 Results and Analysis
In this section, we evaluate RISE from three as-
pects. First, we validate effectiveness of multiround
self-iterative and compare RISE with mainstream
MHQA methods. Second, we conduct an in-depth
analysis of the performance of question decomposi-
tion, retrieve-then-read, and self-critique using ob-
jective metrics and AI-based evaluations. Finally,
we conduct ablation studies to verify the impor-
tance of different tasks in enhancing performance.
4.1 Overall Performance
RISE Outperforms Other Methods: Table 2
presents the experimental results across three
MHQA datasets and three SHQA datasets. We
observe that retrieval-based enhancement is cru-
cial for MHQA tasks. While CoT achieves rela-
tively good performance, other non-retrieval meth-
ods generally perform worse than most RAG ap-
proaches with the same model. For the relatively
simpler SHQA tasks, retrieval-based enhancement
does not seem to offer significant advantages. No-
tably, RISE achieves outstanding results in both
task types over all datasets, even surpassing GPT-
3.5. Furthermore, our method excels in F1 and EM
metrics, demonstrating its efficiency (additional
metrics are provided in Appendix A.4).
Steady Performance Improvement: Meanwhile,
as shown in Figure 3 (a) Accuracy per Iteration , weillustrate how the model’s accuracy evolves over
four iterations on multiple datasets. The results
demonstrate a consistent upward trend in accuracy
with each iteration, further validating the effective-
ness of our proposed self-training method in im-
proving the model’s overall performance.
4.2 Analysis Experiments
4.2.1 Question Decomposition Capability
To evaluate improvement in the model’s decompo-
sition capability for MHQA tasks, we first analyze
the changes in reasoning length. As shown in Fig-
ure 3 (b) Reasoning Length per Iteration , accuracy
steadily improves, while reasoning length initially
increases and then decreases, ultimately showing
downward trend. This trend reflects model’s de-
composition ability progressively improves over
iterations.
To further analyze changes in decomposition
ability, we using GPT-4o as a judge to evaluate the
model’s query decomposition across four dimen-
sions (including conciseness, rationality, sequenc-
ing and goal orientation, see Appendix A.1.2 for
more details.). As illustrated in Figure 4, we com-
pare the performance of the model across iterations
and observe newer model consistently outperforms
the previous iteration. These findings demonstrate
that self-training not only improves reasoning paths
but also enhances the rationality of decomposition.
6

0 20 40 60 80 100Round 1 vs. Round 0
Round 2 vs. Round 1
Round 3 vs. Round 2
Round 4 vs. Round 348.4% 20.3% 31.3%
49.8% 16.2% 34%
47.6% 23.0% 29.4%
57.3% 19.7% 23.0%Win Tie LossFigure 4: Evaluating the win rates between the current and previous iterations using GPT-4o to assess model’s
question decomposition capability. Results indicate that each new iteration consistently outperforms the previous
one in subjective effectiveness, demonstrating RISE’s continuously enhance the model’s decomposition capability.
NQ TriviaQA WebQ4550556065707580Accuracy (%)
49.254.556.258.2 58.273.874.5 74.575.577.2
54.560.0 59.862.0 62.2
(a) Simple QuestionsRound 0
Round 1
Round 2
Round 3
Round 4
2WIKI Hotpot MSQ1015202530354045
37.940.6 40.343.043.5
37.539.441.943.142.2
11.911.112.713.814.3
(b) Complex QuestionsRound 0
Round 1
Round 2
Round 3
Round 4
Figure 5: Changes in the model’s retrieve-then-read capability. (a) Results on simpler datasets (NQ, TriviaQA,
WebQ), (b) Results on more complex datasets (2Wiki, HotpotQA, MSQ), where accuracy shows consistent growth
with each iteration, even in challenging scenarios.
4.2.2 Retrieve-then-Read Capability
In MHQA tasks, models often struggle to integrate
logical information from extensive evidence, espe-
cially in filtering irrelevant content. To evaluate the
changes in the model’s summarization capability
over iterations, we disable the decomposition func-
tionality and instead allow model to perform single-
round retrieval and direct question-answering. To
ensure robustness in the experiments, we introduce
simpler datasets such as NQ, WebQ, and TriviaQA
(Figure 5 (a) Simple Questions ) while retaining the
complex datasets from main experiments (Figure 5
(b) Complex Questions ). The experimental results
show that, as iterations progress, RISE consistently
improves its performance across six datasets. This
demonstrates the advantage of RISE in MHQA
tasks and its effectiveness in conventional QA tasks,
further validating its generalizability.
4.2.3 Self-Critique Capability
To evaluate the changes in the model’s self-critique
capability, we designed a third set of experiments.In this experiment, both our model and GPT-4o as-
sess the same set of decomposition results, with
GPT-4o serving as a reference. By analyzing
the consistency between our model and GPT-4o
evaluations, we measure the improvement in the
model’s self-critique capability. As shown in Table
3, the consistency between our model and GPT-4o
steadily increases with each iteration. This indi-
cates that the iterative process in RISE effectively
enhances the model’s self-criticism capability. (For
more experiment details see Appendix A.2.3.)
4.3 Ablation Study
To evaluate the impact of each synthesized training
dataset on the model’s performance, we conduct an
ablation study. As shown in Table 4, the experiment
uses the same three MHQA datasets as before and
the three training datasets generated in the round1,
with accuracy as the primary evaluation metric.
Removing the question decomposition dataset
leads to accuracy drop of 3.5% on 2Wiki, highlight-
ing its importance in enabling effective multi-hop
7

Consistency with GPT-4o (%)
2WIKI HotpotQA MSQ
Round 1 74.30 64.70 60.00
Round 2 72.67 66.30 76.00
Round 3 79.67 77.33 79.33
Round 4 80.67 79.33 78.00
Table 3: Consistency analysis with GPT-4o on each
datasets. The results show progressive improvements
in consistency with GPT-4o, highlighting the model’s
enhanced self-critique ability through iterative training.
2WIKI Hotpot MSQ
Acc Acc Acc
w/o Decomp 37.63 33.89 11.08
w/o R-t-R 40.59 33.06 9.46
w/o Critique 38.98 33.89 10.27
Separate 40.86 34.72 10.54
RISE 41.13 35.83 11.89
Table 4: Ablation study on 2WIKI, HotpotQA, and
MSQ, showing the impact of removing individual tasks
(Question Decomposition ,Retrieve-then-Read , and Self-
Critique ) and comparing joint training ( RISE ) with sep-
arate training ( Separate ) of individual tasks.
reasoning. Excluding the retrieve-then-read dataset
causes accuracy declines on HotpotQA (2.77%)
and Musique (2.43%), highlighting the importance
of this dataset in synthesizing evidence from di-
verse sources to mitigate the impact of noise. The
removal of the self-critique dataset results in con-
sistent accuracy reductions across all three datasets,
emphasizing its pivotal function in refining reason-
ing paths processes. These results demonstrate the
complementary and indispensable contributions of
the question decomposition, retrieve-then-read, and
self-critique datasets to the model’s performance.
Furthermore, we conduct separate training for
the three tasks (Separate), where three LLMs are in-
dividually trained for decomposition, retrieve-then-
read, and self-critique tasks. Compared to joint
training (RISE), the accuracy of separate training
is consistently lower across all datasets.
5 Related Works
Multi-hop Question Answering: MHQA tasks ad-
dress questions that require integrating information
from multiple sources and performing multi-step
reasoning to produce a complete answer (Zhang
et al., 2024; Li and Du, 2023). Question decomposi-
tion has been a pivotal approach for understandingand solving multi-hop questions, some works (Wei
et al., 2022a; Wang et al., 2023b; Zhou et al., 2023;
Shi et al., 2024b) leverage LLMs to divide complex
questions into simpler single-hop sub-questions
that are solved sequentially. Self-Ask (Press et al.,
2023) uses LLMs to generate and resolve follow-
up sub-questions with an external search engine.
However, the effectiveness of these approaches de-
pends on LLM’s inherent question decomposition
capability, which constrained by hallucinations.
Retrieval-Augmented Generation for MHQA:
RAG (Guu et al., 2020; Lewis et al., 2020; Izac-
ard et al., 2022; Nakano et al., 2021; Asai et al.,
2023; Ma et al., 2023; Yu et al., 2024; Shi et al.,
2024a) integrates retrieval with generation to solve
knowledge-intensive tasks (Zhu et al., 2024; Feng
et al., 2024). The original RAG framework excels
at single-hop QA but faces significant challenges
in handling multi-hop QA and complex reasoning
tasks (Lewis et al., 2020; Xu et al., 2024).
To address these challenges, various methods
have been proposed. Chain of Thought (CoT) (Wei
et al., 2022b) and Tree of Thought (ToT) (Yao et al.,
2024) are integrated with RAG to enable multi-step
reasoning and iterative retrieval (Press et al., 2023;
Yao et al., 2023; Zhou et al., 2023; Khattab et al.,
2023), allowing the model to incorporate a broader
range of external knowledge and improve its rea-
soning capabilities. However, existing retrieval-
augmented systems are inevitably affected by the
limitations of retrievers, often introducing irrele-
vant or noisy information (Yin et al., 2023; Xu et al.,
2024; Ma et al., 2023). Enhancing the model’s rea-
soning capabilities to filter noise and focus on crit-
ical evidence is essential for accurate summaries,
which our method achieves through reasoning de-
composition, improving both logical reasoning and
QA performance.
Self-Improvement in Large Language Models:
Self-improvement refers to the process by which
models generate and utilize their own output data
to enhance performance (Zelikman et al., 2024;
Singh et al., 2024; Gülçehre et al., 2023). Exist-
ing approaches, such as self-training (Du et al.,
2021) and self-play (Yuan et al., 2024; Chen et al.,
2024), leverage pseudo-label generation and itera-
tive policy optimization to improve the utilization
of unlabeled data and enhance decision-making
capabilities. Self-Rewarding (Yuan et al., 2024)
employs the LLM-as-Judge paradigm to strengthen
reasoning abilities, while Self-Refine (Madaan
et al., 2024) iteratively optimizes generated out-
8

puts through self-feedback mechanisms.
In complex tasks like code generation and agent-
based learning, self-improvement proves effective.
Methods such as Self-Evolve (Jiang et al., 2023),
NExT (Ni et al., 2024), and AutoAct (Qiao et al.,
2024) leverage self-feedback, self-guided track-
ing, and self-planning to enhance performance.
However, the application of self-iterative tech-
niques in RAG scenarios remains underexplored.
Our method addresses this gap by integrating self-
exploration into RAG to generate diverse training
data, enabling continuous model evolution and en-
hancing performance in complex tasks.
6 Conclusion
We propose RISE, a framework that addresses two
key errors in MHQA tasks: Evidence Aggrega-
tion and Reasoning Decomposition. Through self-
exploration, RISE continuously enhances reason-
ing capabilities. Additionally, RISE integrates self-
iterative paradigm with RAG framework, bridg-
ing the gap in applying self-iterative strategies to
MHQA scenarios without requiring manual inter-
vention or reliance on large models, thereby offer-
ing a cost-effective solution. Experimental results
on MHQA benchmarks demonstrate significant im-
provements in reasoning accuracy and task perfor-
mance, highlighting RISE’s robustness and adapt-
ability in tackling complex reasoning challenges.
Limitation
While RISE achieves strong performance in com-
plex reasoning tasks, there remain opportunities
for further enhancement. The current framework
relies on external retrieval mechanisms without
explicit optimization, which may limit the qual-
ity of evidence for downstream reasoning. Future
work could explore self-improvement across the
entire pipeline—spanning question decomposition,
retrieval, generation, and reflection—to achieve
more seamless integration and efficiency.
References
Akari Asai, Zeqiu Wu, Yizhong Wang, Avirup Sil, and
Hannaneh Hajishirzi. 2023. Self-rag: Learning to
retrieve, generate, and critique through self-reflection.
InThe Twelfth International Conference on Learning
Representations .
Jonathan Berant, Andrew Chou, Roy Frostig, and Percy
Liang. 2013. Semantic parsing on freebase from
question-answer pairs. In Proceedings of the 2013conference on empirical methods in natural language
processing , pages 1533–1544.
Tom Brown, Benjamin Mann, Nick Ryder, Melanie
Subbiah, Jared D Kaplan, Prafulla Dhariwal, Arvind
Neelakantan, Pranav Shyam, Girish Sastry, Amanda
Askell, Sandhini Agarwal, Ariel Herbert-V oss,
Gretchen Krueger, Tom Henighan, Rewon Child,
Aditya Ramesh, Daniel Ziegler, Jeffrey Wu, Clemens
Winter, Chris Hesse, Mark Chen, Eric Sigler, Ma-
teusz Litwin, Scott Gray, Benjamin Chess, Jack
Clark, Christopher Berner, Sam McCandlish, Alec
Radford, Ilya Sutskever, and Dario Amodei. 2020.
Language models are few-shot learners. In Ad-
vances in Neural Information Processing Systems ,
volume 33, pages 1877–1901. Curran Associates,
Inc.
Stephen Casper, Xander Davies, Claudia Shi,
Thomas Krendl Gilbert, Jérémy Scheurer, Javier
Rando, Rachel Freedman, Tomasz Korbak, David
Lindner, Pedro Freire, et al. 2023. Open problems
and fundamental limitations of reinforcement
learning from human feedback. Transactions on
Machine Learning Research .
Zixiang Chen, Yihe Deng, Huizhuo Yuan, Kaixuan Ji,
and Quanquan Gu. 2024. Self-play fine-tuning con-
verts weak language models to strong language mod-
els. In Forty-first International Conference on Ma-
chine Learning .
Aakanksha Chowdhery, Sharan Narang, Jacob Devlin,
Maarten Bosma, Gaurav Mishra, Adam Roberts, Paul
Barham, Hyung Won Chung, Charles Sutton, Sebas-
tian Gehrmann, et al. 2023. Palm: Scaling language
modeling with pathways. Journal of Machine Learn-
ing Research , 24(240):1–113.
Jingfei Du, Edouard Grave, Beliz Gunel, Vishrav Chaud-
hary, Onur Celebi, Michael Auli, Veselin Stoyanov,
and Alexis Conneau. 2021. Self-training improves
pre-training for natural language understanding. In
Proceedings of the 2021 Conference of the North
American Chapter of the Association for Computa-
tional Linguistics: Human Language Technologies ,
pages 5408–5418, Online. Association for Computa-
tional Linguistics.
Abhimanyu Dubey, Abhinav Jauhri, Abhinav Pandey,
Abhishek Kadian, Ahmad Al-Dahle, Aiesha Letman,
Akhil Mathur, Alan Schelten, Amy Yang, Angela
Fan, et al. 2024. The llama 3 herd of models. arXiv
e-prints , pages arXiv–2407.
Zhangyin Feng, Xiaocheng Feng, Dezhi Zhao, Maojin
Yang, and Bing Qin. 2024. Retrieval-generation syn-
ergy augmented large language models. In ICASSP
2024 - 2024 IEEE International Conference on
Acoustics, Speech and Signal Processing (ICASSP) ,
pages 11661–11665.
Çaglar Gülçehre, Tom Le Paine, Srivatsan Srini-
vasan, Ksenia Konyushkova, Lotte Weerts, Abhishek
Sharma, Aditya Siddhant, Alex Ahern, Miaosen
9

Wang, Chenjie Gu, et al. 2023. Reinforced self-
training (rest) for language modeling. CoRR .
Kelvin Guu, Kenton Lee, Zora Tung, Panupong Pasu-
pat, and Mingwei Chang. 2020. Retrieval augmented
language model pre-training. In International confer-
ence on machine learning , pages 3929–3938. PMLR.
Xanh Ho, Anh-Khoa Duong Nguyen, Saku Sugawara,
and Akiko Aizawa. 2020. Constructing a multi-
hop QA dataset for comprehensive evaluation of
reasoning steps. In Proceedings of the 28th Inter-
national Conference on Computational Linguistics ,
pages 6609–6625, Barcelona, Spain (Online). Inter-
national Committee on Computational Linguistics.
Gautier Izacard, Patrick Lewis, Maria Lomeli, Lu-
cas Hosseini, Fabio Petroni, Timo Schick, Jane
Dwivedi-Yu, Armand Joulin, Sebastian Riedel, and
Edouard Grave. 2022. Few-shot learning with re-
trieval augmented language models. arXiv preprint
arXiv:2208.03299 .
Ziwei Ji, Nayeon Lee, Rita Frieske, Tiezheng Yu, Dan
Su, Yan Xu, Etsuko Ishii, Ye Jin Bang, Andrea
Madotto, and Pascale Fung. 2023. Survey of halluci-
nation in natural language generation. ACM Comput-
ing Surveys , 55(12):1–38.
Shuyang Jiang, Yuhao Wang, and Yu Wang. 2023. Self-
evolve: A code evolution framework via large lan-
guage models. Preprint , arXiv:2306.02907.
Mandar Joshi, Eunsol Choi, Daniel S Weld, and Luke
Zettlemoyer. 2017. Triviaqa: A large scale distantly
supervised challenge dataset for reading comprehen-
sion. arXiv preprint arXiv:1705.03551 .
Omar Khattab, Keshav Santhanam, Xiang Lisa
Li, David Hall, Percy Liang, Christopher Potts,
and Matei Zaharia. 2023. Demonstrate-search-
predict: Composing retrieval and language mod-
els for knowledge-intensive nlp. Preprint ,
arXiv:2212.14024.
Tom Kwiatkowski, Jennimaria Palomaki, Olivia Red-
field, Michael Collins, Ankur Parikh, Chris Alberti,
Danielle Epstein, Illia Polosukhin, Jacob Devlin, Ken-
ton Lee, et al. 2019. Natural questions: a benchmark
for question answering research. Transactions of the
Association for Computational Linguistics , 7:453–
466.
Patrick Lewis, Ethan Perez, Aleksandra Piktus, Fabio
Petroni, Vladimir Karpukhin, Naman Goyal, Hein-
rich Küttler, Mike Lewis, Wen-tau Yih, Tim Rock-
täschel, et al. 2020. Retrieval-augmented generation
for knowledge-intensive nlp tasks. Advances in Neu-
ral Information Processing Systems , 33:9459–9474.
Ruosen Li and Xinya Du. 2023. Leveraging structured
information for explainable multi-hop question an-
swering and reasoning. In Findings of the Associa-
tion for Computational Linguistics: EMNLP 2023 ,
pages 6779–6789, Singapore. Association for Com-
putational Linguistics.Hunter Lightman, Vineet Kosaraju, Yuri Burda, Harri-
son Edwards, Bowen Baker, Teddy Lee, Jan Leike,
John Schulman, Ilya Sutskever, and Karl Cobbe.
2023. Let’s verify step by step. In The Twelfth Inter-
national Conference on Learning Representations .
Xiao Liu, Hanyu Lai, Hao Yu, Yifan Xu, Aohan
Zeng, Zhengxiao Du, Peng Zhang, Yuxiao Dong,
and Jie Tang. 2023. Webglm: Towards an efficient
web-enhanced question answering system with hu-
man preferences. In Proceedings of the 29th ACM
SIGKDD Conference on Knowledge Discovery and
Data Mining , pages 4549–4560.
Haipeng Luo, Qingfeng Sun, Can Xu, Pu Zhao, Jian-
guang Lou, Chongyang Tao, Xiubo Geng, Qingwei
Lin, Shifeng Chen, and Dongmei Zhang. 2023. Wiz-
ardmath: Empowering mathematical reasoning for
large language models via reinforced evol-instruct.
Preprint , arXiv:2308.09583.
Xinbei Ma, Yeyun Gong, Pengcheng He, Hai Zhao,
and Nan Duan. 2023. Query rewriting in retrieval-
augmented large language models. In Proceedings of
the 2023 Conference on Empirical Methods in Natu-
ral Language Processing , pages 5303–5315, Singa-
pore. Association for Computational Linguistics.
Aman Madaan, Niket Tandon, Prakhar Gupta, Skyler
Hallinan, Luyu Gao, Sarah Wiegreffe, Uri Alon,
Nouha Dziri, Shrimai Prabhumoye, Yiming Yang,
et al. 2024. Self-refine: Iterative refinement with
self-feedback. Advances in Neural Information Pro-
cessing Systems , 36.
Reiichiro Nakano, Jacob Hilton, Suchir Balaji, Jeff Wu,
Long Ouyang, Christina Kim, Christopher Hesse,
Shantanu Jain, Vineet Kosaraju, William Saunders,
et al. 2021. Webgpt: Browser-assisted question-
answering with human feedback. arXiv preprint
arXiv:2112.09332 .
Ansong Ni, Miltiadis Allamanis, Arman Cohan, Yinlin
Deng, Kensen Shi, Charles Sutton, and Pengcheng
Yin. 2024. Next: Teaching large language models to
reason about code execution. In Forty-first Interna-
tional Conference on Machine Learning .
Ofir Press, Muru Zhang, Sewon Min, Ludwig Schmidt,
Noah A Smith, and Mike Lewis. 2023. Measuring
and narrowing the compositionality gap in language
models. In Findings of the Association for Computa-
tional Linguistics: EMNLP 2023 , pages 5687–5711.
Shuofei Qiao, Ningyu Zhang, Runnan Fang, Yujie Luo,
Wangchunshu Zhou, Yuchen Eleanor Jiang, Huajun
Chen, et al. 2024. Autoact: Automatic agent learning
from scratch for qa via self-planning. In ICLR 2024
Workshop on Large Language Model (LLM) Agents .
Vipula Rawte, Amit Sheth, and Amitava Das. 2023. A
survey of hallucination in large foundation models.
arXiv preprint arXiv:2309.05922 .
Zhengliang Shi, Shuo Zhang, Weiwei Sun, Shen Gao,
Pengjie Ren, Zhumin Chen, and Zhaochun Ren.
10

2024a. Generate-then-ground in retrieval-augmented
generation for multi-hop question answering. In Pro-
ceedings of the 62nd Annual Meeting of the Associa-
tion for Computational Linguistics (Volume 1: Long
Papers) , pages 7339–7353.
Zhengliang Shi, Shuo Zhang, Weiwei Sun, Shen Gao,
Pengjie Ren, Zhumin Chen, and Zhaochun Ren.
2024b. Generate-then-ground in retrieval-augmented
generation for multi-hop question answering. In Pro-
ceedings of the 62nd Annual Meeting of the Associa-
tion for Computational Linguistics (Volume 1: Long
Papers) , pages 7339–7353, Bangkok, Thailand. As-
sociation for Computational Linguistics.
Kumar Shridhar, Alessandro Stolfo, and Mrinmaya
Sachan. 2023. Distilling reasoning capabilities into
smaller language models. In Findings of the Associa-
tion for Computational Linguistics: ACL 2023 , pages
7059–7073.
Avi Singh, John D Co-Reyes, and Rishabh Agarwal.
2024. Beyond human data: Scaling self-training for
problem-solving with language models. In ICLR
2024 Workshop on Navigating and Addressing Data
Problems for Foundation Models .
Hugo Touvron, Louis Martin, Kevin Stone, Peter Al-
bert, Amjad Almahairi, Yasmine Babaei, Nikolay
Bashlykov, Soumya Batra, Prajjwal Bhargava, Shruti
Bhosale, et al. 2023. Llama 2: Open founda-
tion and fine-tuned chat models. arXiv preprint
arXiv:2307.09288 .
Harsh Trivedi, Niranjan Balasubramanian, Tushar Khot,
and Ashish Sabharwal. 2022. ♪MuSiQue: Multi-
hop questions via single-hop question composition.
Transactions of the Association for Computational
Linguistics , 10:539–554.
Jonathan Uesato, Nate Kushman, Ramana Kumar, Fran-
cis Song, Noah Siegel, Lisa Wang, Antonia Creswell,
Geoffrey Irving, and Irina Higgins. 2022. Solving
math word problems with process- and outcome-
based feedback. Preprint , arXiv:2211.14275.
Tianduo Wang, Shichen Li, and Wei Lu. 2024. Self-
training with direct preference optimization improves
chain-of-thought reasoning. In Proceedings of the
62nd Annual Meeting of the Association for Compu-
tational Linguistics (Volume 1: Long Papers) , pages
11917–11928.
Xuezhi Wang, Jason Wei, Dale Schuurmans, Quoc V Le,
Ed H Chi, Sharan Narang, Aakanksha Chowdhery,
and Denny Zhou. 2023a. Self-consistency improves
chain of thought reasoning in language models. In
The Eleventh International Conference on Learning
Representations .
Xuezhi Wang, Jason Wei, Dale Schuurmans, Quoc V Le,
Ed H. Chi, Sharan Narang, Aakanksha Chowdhery,
and Denny Zhou. 2023b. Self-consistency improves
chain of thought reasoning in language models. In
The Eleventh International Conference on Learning
Representations .Yizhong Wang, Yeganeh Kordi, Swaroop Mishra, Alisa
Liu, Noah A Smith, Daniel Khashabi, and Hannaneh
Hajishirzi. 2023c. Self-instruct: Aligning language
models with self-generated instructions. In The 61st
Annual Meeting Of The Association For Computa-
tional Linguistics .
Jason Wei, Xuezhi Wang, Dale Schuurmans, Maarten
Bosma, brian ichter, Fei Xia, Ed Chi, Quoc V Le,
and Denny Zhou. 2022a. Chain-of-thought prompt-
ing elicits reasoning in large language models. In
Advances in Neural Information Processing Systems ,
volume 35, pages 24824–24837. Curran Associates,
Inc.
Jason Wei, Xuezhi Wang, Dale Schuurmans, Maarten
Bosma, Fei Xia, Ed Chi, Quoc V Le, Denny Zhou,
et al. 2022b. Chain-of-thought prompting elicits rea-
soning in large language models. Advances in neural
information processing systems , 35:24824–24837.
Shicheng Xu, Liang Pang, Huawei Shen, Xueqi Cheng,
and Tat-Seng Chua. 2024. Search-in-the-chain: In-
teractively enhancing large language models with
search for knowledge-intensive tasks. In Proceed-
ings of the ACM Web Conference 2024 , WWW ’24,
page 1362–1373, New York, NY , USA. Association
for Computing Machinery.
Zhilin Yang, Peng Qi, Saizheng Zhang, Yoshua Bengio,
William Cohen, Ruslan Salakhutdinov, and Christo-
pher D. Manning. 2018. HotpotQA: A dataset for
diverse, explainable multi-hop question answering.
InProceedings of the 2018 Conference on Empiri-
cal Methods in Natural Language Processing , pages
2369–2380, Brussels, Belgium. Association for Com-
putational Linguistics.
Shunyu Yao, Dian Yu, Jeffrey Zhao, Izhak Shafran,
Tom Griffiths, Yuan Cao, and Karthik Narasimhan.
2024. Tree of thoughts: Deliberate problem solving
with large language models. Advances in Neural
Information Processing Systems , 36.
Shunyu Yao, Jeffrey Zhao, Dian Yu, Nan Du, Izhak
Shafran, Karthik R Narasimhan, and Yuan Cao. 2023.
React: Synergizing reasoning and acting in language
models. In The Eleventh International Conference
on Learning Representations .
Hongbin Ye, Tong Liu, Aijia Zhang, Wei Hua, and
Weiqiang Jia. 2023. Cognitive mirage: A review
of hallucinations in large language models. arXiv
preprint arXiv:2309.06794 .
Xunjian Yin, Baizhou Huang, and Xiaojun Wan. 2023.
ALCUNA: Large language models meet new knowl-
edge. In Proceedings of the 2023 Conference on
Empirical Methods in Natural Language Processing ,
pages 1397–1414, Singapore. Association for Com-
putational Linguistics.
Wenhao Yu, Dan Iter, Shuohang Wang, Yichong Xu,
Mingxuan Ju, S Sanyal, Chenguang Zhu, Michael
Zeng, and Meng Jiang. 2023. Generate rather than
retrieve: Large language models are strong context
11

generators. In International Conference on Learning
Representations .
Yue Yu, Wei Ping, Zihan Liu, Boxin Wang, Jiaxuan
You, Chao Zhang, Mohammad Shoeybi, and Bryan
Catanzaro. 2024. Rankrag: Unifying context ranking
with retrieval-augmented generation in llms. In The
Thirty-eighth Annual Conference on Neural Informa-
tion Processing Systems .
Weizhe Yuan, Richard Yuanzhe Pang, Kyunghyun Cho,
Xian Li, Sainbayar Sukhbaatar, Jing Xu, and Ja-
son E Weston. 2024. Self-rewarding language mod-
els. In Forty-first International Conference on Ma-
chine Learning .
Eric Zelikman, Eliana Lorch, Lester Mackey, and
Adam Tauman Kalai. 2024. Self-taught optimizer
(stop): Recursively self-improving code generation.
InOPT 2023: Optimization for Machine Learning .
Aohan Zeng, Xiao Liu, Zhengxiao Du, Zihan Wang,
Hanyu Lai, Ming Ding, Zhuoyi Yang, Yifan Xu,
Wendi Zheng, Xiao Xia, et al. 2022. Glm-130b: An
open bilingual pre-trained model. In The Eleventh In-
ternational Conference on Learning Representations .
Jiahao Zhang, Haiyang Zhang, Dongmei Zhang, Liu
Yong, and Shen Huang. 2024. End-to-end beam re-
trieval for multi-hop question answering. In Proceed-
ings of the 2024 Conference of the North American
Chapter of the Association for Computational Lin-
guistics: Human Language Technologies (Volume
1: Long Papers) , pages 1718–1731, Mexico City,
Mexico. Association for Computational Linguistics.
Susan Zhang, Stephen Roller, Naman Goyal, Mikel
Artetxe, Moya Chen, Shuohui Chen, Christopher De-
wan, Mona Diab, Xian Li, Xi Victoria Lin, et al. 2022.
Opt: Open pre-trained transformer language models.
arXiv preprint arXiv:2205.01068 .
Denny Zhou, Nathanael Schärli, Le Hou, Jason Wei,
Nathan Scales, Xuezhi Wang, Dale Schuurmans,
Claire Cui, Olivier Bousquet, Quoc V Le, and Ed H.
Chi. 2023. Least-to-most prompting enables com-
plex reasoning in large language models. In The
Eleventh International Conference on Learning Rep-
resentations .
Yutao Zhu, Huaying Yuan, Shuting Wang, Jiongnan Liu,
Wenhan Liu, Chenlong Deng, Haonan Chen, Zheng
Liu, Zhicheng Dou, and Ji-Rong Wen. 2024. Large
language models for information retrieval: A survey.
Preprint , arXiv:2308.07107.
12

A Appendix
A.1 Prompts
A.1.1 Self-Exploration Prompts
We designed detailed prompts for the three tasks in
the self-exploration phase: question decomposition
(Figure 6), retrieve-then-read(Figure 7), and self-
critique(Figure 8). The examples used in the de-
composition prompt are inspired by self-ask (Press
et al., 2023).
A.1.2 Self-Decomposition Evaluation Prompt
In this paper, the evaluation of the question decom-
position capability is conducted using GPT-4o with
prompt as shown in Figure 9. The analysis involves
assessing and scoring the decomposition results of
different iterations across multiple dimensions, ulti-
mately leading to a comparative analysis of the two
models. The dimensions of the analysis include:
•Conciseness : Whether the decomposition
avoids redundancy while ensuring comprehen-
siveness.
•Rationality : Whether the decomposed sub-
problems are closely related to the original
problem.
•Sequencing : Whether the decomposition of
sub-problems follows a logical order and fa-
cilitates the problem-solving process.
•Goal Orientation : Whether the decompo-
sition is clearly centered around addressing
the main problem’s objective. Are the sub-
problems closely aligned with the core goal
of the main problem? Does it avoid redundant
issues that deviate from the primary objective?
A.2 Experiment detail
A.2.1 Implementation Details
We conduct all experiments on a server equipped
with four NVIDIA A800 80G GPUs. For the ex-
perimental setup, we use the following hyperpa-
rameters: learning rate of 1×10−4, batch size of
64,and cut-off length of 8192. Furthermore, for the
weighting parameters α,β, andγin the overall loss
function, values of 1 are uniformly adopted in this
research.Datasets DdDrDc
Round 1 3276 2501 3925
Round 2 8309 6311 8074
Round 3 4858 2106 2312
Round 4 6913 4759 5307
Table 5: Number of samples accumulated in datasets
Dd,Dr, andDcafter each round of self-exploration.
2WIKI HotpotQA MSQ
Round 1 223 194 180
Round 2 218 199 228
Round 3 239 232 238
Round 4 242 238 234
Total 300 300 300
Table 6: Number of instances in each round’s self-
critique capability evaluation that aligned with GPT-4o.
A.2.2 Datasets
The cold-start dataset Q0consists of 800 randomly
sampled instances from the training sets of 2Wiki-
MultiHopQA, HotpotQA, and MuSiQue, totaling
2,400 cold-start samples. Table 5 provides detailed
information on the training datasets constructed
during each round of self-exploration. The eval-
uation datasets we used 2WikiMultiHopQA, Hot-
potQA, and MuSiQue each contain 1,000 samples,
Nature Questions, Web Questions, and TriviaQA
each contain 400 samples.
A.2.3 Self-Critique Capability Experiments
Details
To demonstrate the improvement in the self-critique
capability of the model across iterations, we sam-
pled 300 instances from the generated Dcat each
round and compared them with GPT-4o. The re-
sponses from GPT-4o were used as ground truth to
calculate the self-critique accuracy of our model.
In Table 6, we present the number of instances in
each round’s self-critique capability evaluation that
aligned with GPT-4o.
A.3 Additional Experiments
A.3.1 RISE Robustness
To further verify the robustness of our experimental
conclusions, we conducted additional experiments
using the Qwen2.5-7B model. Specifically, we
performed four rounds of self-exploration follow-
ing the same experimental setup as in our original
RISE framework. The results consistently demon-
strate the effectiveness of RISE, with performance
13

2WIKI HotpotQA MSQ
Acc F1 EM Acc F1 EM Acc F1 EM
Round 0 38.98 23.28 12.10 31.94 21.72 10.28 7.30 8.98 2.70
Round 1 45.97 46.49 32.50 35.83 44.55 32.50 10.54 18.65 8.92
Round 2 46.24 46.33 30.83 36.11 41.58 30.00 11.89 16.19 8.92
Round 3 47.31 47.10 34.41 38.33 42.82 31.39 13.24 20.70 10.81
Table 7: Performance of the Qwen2.5-7B model after four rounds of self-exploration on different datasets, showing
improvements in accuracy, F1, and EM scores across 2WIKI, HotpotQA, and MSQ.
Methods Average Input Token Average Output Tokens Number of LLMs Calls
Naive LLM 18 244 1
CoT 24 284 1
CoT-SC 240 2840 10
GenRead 307 233 2
Naive RAG 440 104 1
Self-Ask 782 172 2
WebGLM 436 103 1
Self-RAG 866 294 2
RRR 496 86 2
GenGround 2449 167 5
RISE 2881 192 5
Table 8: Token consumption comparison between RISE and other baseline methods, showing average input tokens,
average output tokens, and the number of LLMs calls required for each approach. RISE demonstrates a higher input
token consumption due to its multi-step reasoning, but maintains efficient reasoning performance.
improvements observed across multiple datasets
after each iteration. This confirms that RISE main-
tains strong generalization capabilities and stable
performance even when applied to different large
language models.
A.3.2 Token Consumption Details
In addition to performance evaluation, we analyzed
the token consumption of RISE compared to other
baseline methods. We measured both the average
input token consumption and the average output
token length, as well as the number of model calls
required in each approach. The results reveal that
while RISE consumes more input tokens due to its
multi-step reasoning process, it achieves higher ef-
ficiency in output generation and overall reasoning
effectiveness. This analysis highlights the trade-
off between token usage and model performance,
demonstrating that RISE achieves a balanced opti-
mization in complex reasoning tasks.
A.3.3 The necessity of multiple iterations
To further validate the necessity and effectiveness
of our multi-round training strategy, we conduct ad-
ditional experiments comparing single-round and
multi-round training setups on three benchmarkdatasets: 2Wiki, HotpotQA, and MSQ. The results
are summarized in Table ??.
These results demonstrate that multi-round train-
ing significantly improves accuracy (Acc) and ex-
act match (EM) metrics across all datasets, high-
lighting the advantage of iterative self-exploration
over static single-round training. Additionally, we
compare joint training with alternating training
strategies to clarify their differences. As shown in
Table ??, joint training better preserves learned ca-
pabilities and achieves performance gains through
synergistic task-chain interactions.
A.3.4 Reliability Analysis of Self-Evaluation
Mechanism
We further investigate the reliability of the model’s
self-critique mechanism by evaluating the align-
ment between self-assessment and actual correct-
ness. Table ??presents the distribution of cases
where the model’s self-judgment matches or mis-
matches the ground truth for both the baseline GPT-
4o and our method.
These results confirm a positive correlation
between the model’s self-assessment and actual
answer correctness, demonstrating the effective-
14

Setup2Wiki Hotpot MSQ
Acc F1 EM Acc F1 EM Acc F1 EM
Single-round 41.10 43.32 30.60 34.10 43.28 29.50 11.10 17.78 8.60
Multi-round 49.40 43.01 32.70 40.50 42.00 30.50 21.70 22.87 11.70
Table 9: Comparison between single-round and multi-round training on three datasets.
ness and reliability of the proposed self-evaluation
mechanism.
A.4 Supply Metrics of Main Results
This section provides additional details to sup-
plement the main results, including comprehen-
sive Exact Match (EM) and F1 scores across
six QA datasets: 2WikiMultiHopQA, HotpotQA,
MuSiQue, Natural Questions, Web Questions, and
TriviaQA. We compare RISE (Ours) with both
prompting-based and fine-tuning-based methods,
under settings with and without retrieval. The re-
sults offer a deeper understanding of RISE’s perfor-
mance, highlighting its consistent improvements
over baseline models.
A.5 Case Study
To further illustrate the design of self-exploration
process, we present representative case study.
Each task is marked with a task-specific prefix,
and follows a carefully curated instruction tem-
plate.(Shown in Figure 11, 12, 13, 14, 15, 16)
15

Model Correct and judged Correct Correct but judged Wrong Wrong but judged Correct Wrong and judged Wrong
GPT-4o 32.58% 9.71% 11.21% 46.50%
Ours 28.65% 7.58% 23.43% 40.33%
Table 10: Alignment between model self-judgment and ground truth correctness on initial question set Q0.
Method ModelMHQA SHQA
2WIKI Hotpot MSQ NQ WebQ Trival
w/o retrieval
Naive LLM LLaMA-3.1-8B 0.00 0.30 0.00 0.25 0.00 2.25
GPT-3.5-turbo 0.50 5.50 0.20 2.50 0.75 26.75
CoT (Wei et al., 2022b) LLaMA-3.1-8B 0.00 0.00 0.00 0.00 0.00 0.25
CoT-SC*(Wang et al., 2023a) GPT-3.5-turbo 18.40 31.90 9.60 30.75 22.75 71.75
GenRead (Yu et al., 2023) LLaMA-3.1-8B 17.00 25.60 7.60 35.75 25.50 61.75
w retrieval
Naive RAG LLaMA-3.1-8B 0.30 1.70 0.20 0.00 0.00 2.25
Self-Ask (Press et al., 2023) LLaMA-3.1-8B 17.00 25.60 9.60 28.50 21.50 58.75
WebGLM (Liu et al., 2023) LLaMA-3.1-8B 0.00 0.30 0.00 0.00 0.00 0.00
Self-RAG*(Asai et al., 2023) LLaMA2-7B 2.90 3.90 0.80 0.00 0.00 2.00
LLaMA2-13B 3.40 2.50 0.40 0.00 0.00 4.50
RRR (Ma et al., 2023) LLaMA-3.1-8B 0.00 0.40 0.00 0.75 0.75 2.75
GPT-3.5-turbo 3.20 2.20 0.30 2.50 2.00 20.00
GenGround (Shi et al., 2024a) LLaMA-3.1-8B 23.50 24.30 10.20 20.50 18.50 60.25
RISE(Ours) LLaMA-3.1-8B 32.70 30.50 11.70 28.50 19.25 59.50
Table 11: EM metrics of RISE with other methods on the 2WikiMultiHopQA, HotpotQA, MuSiQue, Natural
Questions, Web Questions and TriviaQA. Methods marked with asterisk (*) involve specific considerations: CoT-SC
uses GPT-3.5 due to LLaMA-3.1’s limitations in adhering to instructions, and Self-RAG employs publicly released
model weights because its dataset is unavailable. All other methods are reproduced with LLaMA-3.1-8B.
represent Prompting-based Methods , while represent Finetuning-based Methods .
Method ModelMHQA SHQA
2WIKI Hotpot MSQ NQ WebQ Trival
w/o retrieval
Naive LLM LLaMA-3.1-8B 7.99 5.49 2.67 4.09 5.46 17.55
GPT-3.5-turbo 13.38 17.66 8.10 16.53 19.14 44.11
CoT (Wei et al., 2022b) LLaMA-3.1-8B 2.95 2.31 1.53 1.77 2.19 4.93
CoT-SC*(Wang et al., 2023a) GPT-3.5-turbo 24.31 39.59 15.36 40.98 36.54 80.65
GenRead (Yu et al., 2023) LLaMA-3.1-8B 22.39 34.51 15.26 47.90 42.55 69.59
w retrieval
Naive RAG LLaMA-3.1-8B 6.39 9.60 4.02 9.02 9.72 19.37
Self-Ask (Press et al., 2023) LLaMA-3.1-8B 23.42 36.14 19.67 40.35 39.61 67.02
WebGLM (Liu et al., 2023) LLaMA-3.1-8B 8.27 6.03 3.76 5.57 6.68 9.30
Self-RAG*(Asai et al., 2023) LLaMA2-7B 17.38 15.44 6.02 25.56 21.45 11.34
LLaMA2-13B 14.82 13.41 6.70 15.65 9.34 15.34
RRR (Ma et al., 2023) LLaMA-3.1-8B 5.35 3.21 1.48 4.46 6.68 9.56
GPT-3.5-turbo 13.70 16.76 6.08 22.80 23.16 43.68
GenGround (Shi et al., 2024a) LLaMA-3.1-8B 34.33 34.53 19.81 36.29 28.93 67.20
RISE(Ours) LLaMA-3.1-8B 43.01 42.00 22.87 42.97 36.22 72.70
Table 12: F1metrics of RISE with other methods on the 2WikiMultiHopQA, HotpotQA, MuSiQue, Natural
Questions, Web Questions and TriviaQA. Methods marked with asterisk (*) involve specific considerations: CoT-SC
uses GPT-3.5 due to LLaMA-3.1’s limitations in adhering to instructions, and Self-RAG employs publicly released
model weights because its dataset is unavailable. All other methods are reproduced with LLaMA-3.1-8B.
represent Prompting-based Methods , while represent Finetuning-based Methods .
16

Decomposition Prompt
Instruction: Please answer the following questions according to the given format. Strictly follow
each format specification, as this will ensure consistency and clarity in your response.
- Only add follow-up questions if additional details are needed to arrive at the final answer.
- For each follow-up question, use exactly this format: ’Follow up: question’
- Ensure each follow-up question is direct and structured to be easily searchable, focusing on key
information for efficient search engine retrieval.
- For each answer to a follow-up question, use exactly this format: ’Intermediate answer: answer’
- Do not repeat or alter any previously generated follow-up questions or intermediate answers.
- Conclude with the final answer using this exact format: ’So the final answer is: final answer’ if
no further questions are needed.
Use the examples below to understand the expected structure, and follow this format
without deviating from these instructions.
Question: Who lived longer, Muhammad Ali or Alan Turing?
Are follow up questions needed here: Yes.
Follow up: How old was Muhammad Ali when he died?
Intermediate answer: Muhammad Ali was 74 years old when he died.
Follow up: How old was Alan Turing when he died?
Intermediate answer: Alan Turing was 41 years old when he died.
So the final answer is: Muhammad Ali.
Question: When was the founder of craigslist born?
Are follow up questions needed here: Yes.
Follow up: Who was the founder of craigslist?
Intermediate answer: Craigslist was founded by Craig Newmark.
Follow up: When was Craig Newmark born?
Intermediate answer: Craig Newmark was born on December 6, 1952.
So the final answer is: December 6, 1952.
...
—
Now, **continue the response** using the following question and information provided below.
Only add follow-up questions if necessary to reach the final answer.
**Ensure all follow-up questions are optimized for search engine queries, making each question
concise, direct, and easily searchable. Avoid modifying or repeating any existing content.**
—
Question (ORIGINAL): {question}
Are follow up questions needed here: Yes.
Figure 6: Question Decomposition prompt template.
17

Retrieve-then-Read Prompt
## Question-Answering-in-Reference-Task ##
Instruction:
- Use the references provided to answer the question as specifically and completely as possible.
- If the references do not directly answer the question, combine relevant information from multiple
references to create a well-supported answer.
- When references are Null or insufficient, use your own knowledge to provide a clear and relevant
answer.
- When a direct answer cannot be determined, list any information in the references that could be
relevant or provide partial insights related to the question. Avoid responses such as ’I don’t know’
or ’more information is needed.’
- Always prioritize specificity and relevance in your answer, providing helpful context or details
that approach a complete answer.
Reference [1]
Reference [2]
...
Question: {question}
Figure 7: Retrieve-then-Read prompt template.
18

Self-Critique Prompt
## Critique-Task ##
Main Question: {question}
Below is a list of previously generated subquestions and their intermediate answers, created as part
of a multi-step reasoning process to answer the main question.
Your task is to evaluate whether the information in the current subquestion is necessary and
contributes incrementally towards solving the main question.
Previously generated subquestions and answers:
{previous subquestions}
Current subquestion and answer candidate:
{subquestion and intermediate answer }
Instruction:
- Step 1: Check for Redundancy. Check if the current subquestion or answer repeats information
already provided in previous subquestions. If it does, return ’flag = False’ as this information is
redundant.
- Step 2: Assess Relevance. If the information is not a duplicate, analyze its relevance to the main
question. Determine whether it provides new, relevant information that helps move closer to
solving the main question, even if it only provides indirect context or background.
Note that information does not need to directly answer the main question to be considered relevant;
it can also support understanding or provide necessary context. Mark it as ’flag = True’.
- Step 3: Based on your analysis, provide a final judgment in the following format:
**Final Judgment**: [flag = True or flag = False]
Examples:
Main Question: “Who lived longer, Muhammad Ali or Alan Turing?”
•Follow up: “How old was Muhammad Ali when he died?” (Flag = True, relevant for lifespan
comparison.)
•Follow up: “How old was Alan Turing when he died?” (Flag = True, completes lifespan
comparison.)
•Redundant Example: “How old was Muhammad Ali when he passed?” (Flag = False, redundant
with earlier subquestion.)
Main Question: “Are both the directors of Jaws and Casino Royale from the same country?”
• Follow up: “Who directed Jaws?” (Flag = True, needed for director identification.)
• Follow up: “Where is Steven Spielberg from?” (Flag = True, relevant to nationality check.)
•Irrelevant Example: “What is Steven Spielberg’s favorite genre?” (Flag = False, not relevant to
nationality.)
Reminder: Use “flag = True” for any subquestion that provides useful information or
context toward solving the main question, even if indirectly. Set “flag = False” only if it is
redundant or entirely irrelevant.
Figure 8: Self-Critique prompt template.
19

Self-Decomposition Evaluation Prompt
You are given two problem decomposition results for the same complex problem. Your task is to
compare these results from Conciseness, Rationality, Sequencing and Goal Orientation. Analyze
the two decomposition results using the criteria above. Clearly explain which approach is more
effective for solving the problem and why, while highlighting the strengths and weaknesses of
each approach in detail.
# Scoring Criteria:
- Score each dimension on a scale of 1-5, where:
- 1: Poor
- 2: Needs Improvement
- 3: Average
- 4: Good
- 5: Excellent
# The output follows the format below. Do not add any additional text: {
"Conciseness": {
"Result 1 Score": X,
"Result 2 Score": Y ,
"Explanation": "How effectively does each decomposition avoid unnecessary complexity while
still addressing all relevant aspects of the problem? Is the explanation clear and straightforward?"
},
"Rationality": {
"Result 1 Score": X,
"Result 2 Score": Y ,
"Explanation": "Are the identified components logical and directly related to the problem? Do the
solutions align well with the identified components?"
},
"Sequencing": {
"Result 1 Score": X,
"Result 2 Score": Y ,
"Explanation": "Is the order of steps or components logical and easy to follow? Does the sequence
facilitate efficient problem-solving?"
},
"Goal Orientation": {
"Result 1 Score": X,
"Result 2 Score": Y ,
"Explanation": "Do the sub-questions stay aligned with the core goal of the main problem? Are
there any redundant sub-questions that deviate from the primary objective?"
},
"Result": "Decomposition Results 1 Decomposition Results 2 Tie"
}
# Problem:
{problem} # Decomposition Results to Compare:
- Decomposition Results 1:
{result1}
- Decomposition Results 2:
{result2} # Output:
Figure 9: GPT-4o decomposition prompt template.
20

Multi-Hop Question Generation Prompt
#Multi-Hop-Question-Generation-in-{Task}#
Instruction:
- You are an AI assistant tasked with generating multi-hop questions similar to those in the {task}
dataset.
- These questions require combining multiple pieces of information to reach the answer.
- Typically, these questions involve indirect references and nested relationships.
Examples from the {task} dataset:
Example 1: Question: [Insert Example Here]
Example 2: Question: [Insert Example Here]
Example 3: Question: [Insert Example Here]
Example 4: Question: [Insert Example Here]
Example 5: Question: [Insert Example Here]
Example 6: Question: [Insert Example Here]
Example 7: Question: [Insert Example Here]
Example 8: Question: [Insert Example Here]
Please generate a **new and unique multi-hop question** that meets the following crite-
ria:
- **Requires reasoning across multiple facts or entities**.
- **Asks for only one piece of information or answer** without multiple sub-questions.
- **Is a single, coherent question** that requires multi-step reasoning to answer.
- **Includes indirect references or nested relationships**.
- **Matches the complexity and structure** of questions in the {task} dataset.
- **Is concise** (one sentence) and clearly worded.
- **Covers new topics** or involves different entities from the examples above.
- **Avoids being a duplicate** of the examples above.
- **Avoids using conjunctions like ’and’, ’or’, or commas** to ask more than one thing.
Your response should **only contain the generated question** without any extra text
and follow the format:
Question: {question}
Figure 10: Multi-hop question generation prompt template.
21

Case Study: Question Decomposition Input
## Question-Decomposition-Task ##
Please answer the following questions according to the given format. Strictly follow each format specification, as
this will ensure consistency and clarity in your response.
- Only add follow-up questions if additional details are needed to arrive at the final answer.
- For each follow-up question, use exactly this format: ’Follow up: question’
- Ensure each follow-up question is direct and structured to be easily searchable, focusing on key information for
efficient search engine retrieval.
- For each answer to a follow-up question, use exactly this format: ’Intermediate answer: answer’
- Do not repeat or alter any previously generated follow-up questions or intermediate answers.
- Conclude with the final answer using this exact format: ’So the final answer is: final answer’ if no further
questions are needed.
Use the examples below to understand the expected structure, and follow this format without deviat-
ing from these instructions.
Question: Who lived longer, Muhammad Ali or Alan Turing?
Are follow up questions needed here: Yes.
Follow up: How old was Muhammad Ali when he died?
Intermediate answer: Muhammad Ali was 74 years old when he died.
Follow up: How old was Alan Turing when he died?
Intermediate answer: Alan Turing was 41 years old when he died.
So the final answer is: Muhammad Ali. ...
—
Now, continue the response using the following question and information provided below.
Only add follow-up questions if necessary to reach the final answer.
Ensure all follow-up questions are optimized for search engine queries, making each question concise, direct, and
easily searchable. Avoid modifying or repeating any existing content.
—
Question: Are the directors of both films Inter Nos and La Bandera (Film) from the same
country?
Are follow up questions needed here:
Figure 11: Example of Question Decomposition task input.
Case Study: Question Decomposition Output
Yes.
Follow up: Who is the director of Inter Nos?
Figure 12: Example of Question Decomposition task output.
22

Case Study: Retrieve-then-Read Input
## Question-Answering-in-Reference-Task ##
Instruction: - Use the references provided to answer the question as specifically and completely as possible. -
If the references do not directly answer the question, combine relevant information from multiple references to
create a well-supported answer. - When references are Null or insufficient, use your own knowledge to provide a
clear and relevant answer. - When a direct answer cannot be determined, list any information in the references
that could be relevant or provide partial insights related to the question. Avoid responses such as ’I don’t know’
or ’more information is needed.’ - Always prioritize specificity and relevance in your answer, providing helpful
context or details that approach a complete answer.
Reference [1]: Congratulations to the four Directors of Inter Nos...
Reference [2]: ...
...(More retrieve contents)
—
Question: Who is the director of Inter Nos?
Figure 13: Example of Retrieve-then-Read task input.
Case Study: Retrieve-then-Read Output
According to the references provided, the directors of Inter Nos include President Juan
Carlos Martínez. Additionally, Cynthia Vaughn has been named Associate Editor of Inter Nos,
and she will edit the publication’s “Independent Voices” section. It is not specified in
the references provided who the other directors of Inter Nos are.
Figure 14: Example of Retrieve-then-Read task output.
23

Case Study: Self-Critique Input
## Critique-Task ##
Main Question: Are the directors of both films Inter Nos and La Bandera (Film) from the same country?
Below is a list of previously generated subquestions and their intermediate answers, created as part
of a multi-step reasoning process to answer the main question. Your task is to evaluate whether the in-
formation in the current subquestion is necessary and contributes incrementally towards solving the main question.
Previously generated subquestions and answers: Null
Current subquestion and answer candidate:
Follow up: Who is the director of Inter Nos?
Intermediate answer: According to the references provided, the directors of Inter Nos
include President Juan Carlos Martínez. Additionally, Cynthia Vaughn has been named
Associate Editor of Inter Nos, and she will edit the publication’s “Independent Voices”
section. It is not specified in the references provided who the other directors of Inter
Nos are.
Instruction:
- Step 1: Check for Redundancy. Check if the current subquestion or answer repeats information already provided
in previous subquestions. If it does, return ’flag = False’ as this information is redundant.
- Step 2: Assess Relevance. If the information is not a duplicate, analyze its relevance to the main question.
Determine whether it provides new, relevant information that helps move closer to solving the main question,
even if it only provides indirect context or background.Note that information does not need to directly answer the
main question to be considered relevant; it can also support understanding or provide necessary context. Mark it
as “flag = True.
- Step 3: Based on your analysis, provide a final judgment in the following format:
**Final Judgment**: [flag = True or flag = False]
Examples:
Main Question: “Who lived longer, Muhammad Ali or Alan Turing?”
•Follow up: “How old was Muhammad Ali when he died?” (Flag = True, relevant for lifespan
comparison.)
•Follow up: “How old was Alan Turing when he died?” (Flag = True, completes lifespan
comparison.)
•Redundant Example: “How old was Muhammad Ali when he passed?” (Flag = False, redundant
with earlier subquestion.)
Main Question: “Are both the directors of Jaws and Casino Royale from the same country?”
•Follow up: “Who directed Jaws?” (Flag = True, needed for director identification.)
•Follow up: “Where is Steven Spielberg from?” (Flag = True, relevant to nationality
check.)
•Irrelevant Example: “What is Steven Spielberg’s favorite genre?” (Flag = False, not
relevant to nationality.)
Reminder:
Use “flag = True” for any subquestion that provides useful information or context toward solving the main
question, even if indirectly. Set “flag = False” only if it is redundant or entirely irrelevant.
Figure 15: Example of Self-Critique task input.
Case Study: Self-Critique Output
flag = True.
Figure 16: Example of Self-Critique task output.
24