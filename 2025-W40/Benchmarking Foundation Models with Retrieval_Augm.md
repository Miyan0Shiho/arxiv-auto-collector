# Benchmarking Foundation Models with Retrieval-Augmented Generation in Olympic-Level Physics Problem Solving

**Authors**: Shunfeng Zheng, Yudi Zhang, Meng Fang, Zihan Zhang, Zhitan Wu, Mykola Pechenizkiy, Ling Chen

**Published**: 2025-10-01 13:57:53

**PDF URL**: [http://arxiv.org/pdf/2510.00919v2](http://arxiv.org/pdf/2510.00919v2)

## Abstract
Retrieval-augmented generation (RAG) with foundation models has achieved
strong performance across diverse tasks, but their capacity for expert-level
reasoning-such as solving Olympiad-level physics problems-remains largely
unexplored. Inspired by the way students prepare for competitions by reviewing
past problems, we investigate the potential of RAG to enhance physics reasoning
in foundation models. We introduce PhoPile, a high-quality multimodal dataset
specifically designed for Olympiad-level physics, enabling systematic study of
retrieval-based reasoning. PhoPile includes diagrams, graphs, and equations,
capturing the inherently multimodal nature of physics problem solving. Using
PhoPile, we benchmark RAG-augmented foundation models, covering both large
language models (LLMs) and large multimodal models (LMMs) with multiple
retrievers. Our results demonstrate that integrating retrieval with physics
corpora can improve model performance, while also highlighting challenges that
motivate further research in retrieval-augmented physics reasoning.

## Full Text


<!-- PDF content starts -->

Benchmarking Foundation Models with Retrieval-Augmented Generation
in Olympic-Level Physics Problem Solving
Shunfeng Zheng1*, Yudi Zhang2*, Meng Fang3,
Zihan Zhang1,Zhitan Wu4,Mykola Pechenizkiy2,Ling Chen1
1AAII, University of Technology Sydney, New South Wales, Australia
2Eindhoven University of Technology, Eindhoven, The Netherlands
3University of Liverpool, Liverpool, United Kingdom
4University of New South Wales, New South Wales, Australia
Shunfeng.Zheng@student.uts.edu.au,y.zhang5@tue.nl,Meng.Fang@liverpool.ac.uk
Zihan.Zhang-5@student.uts.edu.au,zhitan.wu@student.unsw.edu.au,m.pechenizkiy@tue.nl
Ling.Chen@uts.edu.au
Abstract
Retrieval-augmented generation (RAG) with
foundation models has achieved strong perfor-
mance across diverse tasks, but their capac-
ity for expert-level reasoning—such as solving
Olympiad-level physics problems—remains
largely unexplored. Inspired by the way stu-
dents prepare for competitions by reviewing
past problems, we investigate the potential of
RAG to enhance physics reasoning in foun-
dation models. We introducePhoPile, a
high-quality multimodal dataset specifically de-
signed for Olympiad-level physics, enabling
systematic study of retrieval-based reasoning.
PhoPile includes diagrams, graphs, and equa-
tions, capturing the inherently multimodal na-
ture of physics problem solving. Using PhoPile,
we benchmark RAG-augmented foundation
models, covering both large language models
(LLMs) and large multimodal models (LMMs)
with multiple retrievers. Our results demon-
strate that integrating retrieval with physics cor-
pora can improve model performance, while
also highlighting challenges that motivate fur-
ther research in retrieval-augmented physics
reasoning.
1 Introduction
Physics plays a foundational role in natural sci-
ences and engineering, underpinning progress in
fields ranging from construction and aerospace to
electronics and materials science (Serway et al.,
2000). Mastering physics requires not only con-
ceptual understanding of natural laws but also the
ability to integrate them with quantitative analy-
sis, diagrams, and symbolic reasoning. Recent
years have witnessed the exceptional performance
of foundation models, including large language
models (LLMs) and large multimodal models
*Equal contribution.(LMMs), such as GPT-3 (Brown et al., 2020), Chat-
GLM (Du et al., 2022), GPT-3.5 (OpenAI, 2022),
GPT-4 (OpenAI, 2023a), and Gemini (Deepmind,
2023). These models demonstrate strong capabili-
ties in logic and mathematics (Imani et al., 2023;
Romera-Paredes et al., 2024; Liu et al., 2023; Shi
et al., 2023) and scientific domains (Singhal et al.,
2023; Bingler et al., 2022; Zheng et al., 2025), rais-
ing the prospect of AI agents that can support scien-
tific discovery through physics reasoning. However,
despite these advances, foundation models still face
serious limitations: they lack domain-specific ex-
pertise (Li et al., 2023a; Shen et al., 2023), fre-
quently hallucinate factual content (Ji et al., 2023;
Xiong et al., 2023), and struggle to consistently ap-
ply the appropriate physical principles in problem-
solving.
Retrieval-Augmented Generation (RAG) has re-
cently emerged as a promising approach to mitigat-
ing the limitations of LLMs by integrating external
knowledge sources into their workflows (Caffagni
et al., 2024; Gao et al., 2023). Yet, its effective-
ness for physics reasoning remains largely unex-
plored. In this context, RAG offers a natural solu-
tion: just as a student preparing for an exam con-
sults past competition problems to recall formulas
and problem-solving strategies, an LLM equipped
with retrieval can access relevant laws, examples,
and reasoning patterns from a curated retrieval cor-
pus. This mechanism not only improves factual
accuracy but also helps guide the selection and ap-
plication of physical principles in diverse contexts.
A key obstacle, however, is the absence of suit-
able benchmarks. Whereas mathematics has bene-
fited from a rich ecosystem of high-quality datasets
and benchmarks (Zheng et al., 2021; Hendrycks
et al., 2021; Cobbe et al., 2021; Bauer et al., 2023;
Wang et al., 2023b; Azerbayev et al., 2023; Fang
et al., 2025), physics-specific evaluations are scarce.arXiv:2510.00919v2  [cs.CL]  2 Oct 2025

PhoPile
Retrievers
BM25
MiniLM+cosine
Dragon+
Contriever
…New Question
Consider a uniformly charged metallic ring of radius 𝑅 and total 
charge 𝑞. The ring is a hollow toroid of thickness 2𝑎≪𝑅. This 
thickness can be neglected in parts 𝐴,𝐵,𝐶, and 𝐸 . The 𝑥𝑦 plane 
coincides with the plane of the ring, while the z -axis is 
perpendicular to it, as shown in Figure 1. In parts A and B you 
might need to use the formula (Taylor expansion): 1+𝑥𝜀≈1+
𝜀𝑥+0.5𝜀𝜀−1𝑥2, when 𝑥≪1. Calculate the electrostatic 
potential Φ𝑧 along the axis of the ring at a 𝑧 distance from its 
center  (point A in ###img_1 ###). 
LLMs
GPT-4
Gemini -Pro
Llama -3
Mistral
…AnswerRetrieval
Generation
img_1Figure 1: PhoPile and the overall workflow of foundation models with RAG.
Existing natural science datasets (Welbl et al., 2017;
Lu et al., 2022; Chen et al., 2023) contain only a
small number of low-difficulty, text-only physics
problems. OlympiadBench (He et al., 2024) raises
the level of challenge but evaluates models in iso-
lation, without retrieval. Furthermore, evaluation
itself poses unique challenges: physics answers
may take diverse forms—numerical, symbolic, or
diagrammatic—making automatic grading substan-
tially more difficult than in mathematics.
To address these gaps, we introducePhoPile1,
the first multimodal RAG benchmark for physics
Olympiad problems, as illustrated in Figure 1.
PhoPile consists of 390 Olympiad-level physics
problems from 2019–2021 for evaluation, along
with 2,662 problems from earlier years that serve
as an external retrieval database. Using PhoPile,
we benchmark RAG-augmented foundation mod-
els, covering both large language models (LLMs)
and large multimodal models (LMMs), together
with a variety of retrievers. We highlight two key
observations: competition problems share similar
concepts across years, and past problems capture
not only the necessary physics knowledge from ba-
sic to advanced levels, but also valuable strategies
for applying laws and formulas to novel scenar-
ios. Importantly, PhoPile incorporates multimodal
content—including diagrams, graphs, and equa-
tions—mirroring the real-world practice of physics
problem solving. We further design an LLM-as-
judge framework for evaluation. Our approach uses
instructions and reference solutions to automati-
1Data and code available at: https://github.com/
aialt/PhoPilecally score generated outputs, incorporating both
step-wise and solution-level assessments to capture
the richness of physics reasoning. This framework
enables scalable evaluation that is otherwise infea-
sible with traditional script-based methods.
Our contributions are threefold:
•We propose PhoPile, the first multimodal
benchmark for evaluating retrieval-augmented
physics reasoning.
•We introduce a new LLM-as-judge evaluation
framework tailored for physics, capable of
handling diverse solution formats.
•We conduct a comprehensive benchmark of
8 foundation models with 4 text-only retriev-
ers and 3 multimodal retrievers, providing the
first systematic study of RAG for physics rea-
soning.
2 The PhoPile dataset
2.1 Overview
PhOPile is structured to evaluate the performance
of the RAG pipeline in the domain of high-
level physics problem-solving. Therefore, we
collect Physics Olympiad questions from vari-
ous regions around the globe, including Interna-
tional Physics Olympiad (IPhO, 1967-2023), Asian
Physics Olympiad (APhO, 2000-2021), European
Physics Olympiad (EuPhO, 2017-2021), Nordic-
Baltic Physics Olympiad (NBPhO, 2003-2021),
Romanian Master of Physics (RMPhO, 2012-
2021), United States Physics Olympiad (AAPT,

2007-2019), and British Physics Olympiad (BPhO,
2001-2022), all of which are publicly available.
Our data collection is motivated by the real-
world practice of examinees reviewing past exam
problems when preparing for future tests. Physics
competition problems across years often share over-
lapping knowledge points, cover essential concepts
from basic to advanced levels, and showcase di-
verse strategies for applying physical formulas. Or-
ganizing the dataset in this way provides rich refer-
ences for tackling new problems by leveraging the
knowledge and methods embedded in past compe-
titions.
To evaluate foundation models with RAG in the
context of physics, we organize the collected data
as follows:
•Evaluation Set: 390Olympiad-difficulty-level
physics problems from 2019-2021, used to
evaluate the model’s performance on contem-
porary problems;
•Retrieval Corpus: 2,662 problems before
2019, used by the retriever to provide context
and reference for solving new problems.
The evaluation set is further divided into two sub-
sets:PhoPile-Test, which contains 267 questions
from 125 main problems in text-only form, and
PhoPile(V)-Test, which consists of 123 questions
from 77 main problems that include images either
in the question statement or in the reference solu-
tions.
2.2 Data Collection
In this section, we describe the process of collect-
ing and standardizing physics competition prob-
lems into a unified format. A typical physics prob-
lem consists of textual descriptions, mathematical
formulas, and images, and often exhibits hierarchi-
cal sub-question structures with multiple reference
solutions. All finalized samples are stored in JSON
format.
To construct PhoPile in a consistent and high-
quality manner, we applied the following prepro-
cessing steps:
(1) Text Cleaning.We delete extraneous elements
from the questions, like the historical background
introductions, scoring criteria, and regulations or
policies related to competitions. We filtered out
certain LaTeX commands solely involved in adjust-
ing the format, as they contribute nothing to the
essence of the question.(2) Formula Representation in LaTeX.As
physics problems often involve mathematical for-
mulas, we use LaTeX to formulate the solutions
with all the detailed information in plain text. Addi-
tionally, the text formulation of that physical prob-
lem usually are not accessible, therefore, to avoid
human effort and ensure uniformity and appropri-
ateness, we utilize an OCR recognition software,
MathPix2, to convert the content of images into
LaTeX code (Wang et al., 2023a).
(3) Image Processing.32% of problems in PhoPile
involve images, a detailed breakdown of image us-
age is provided in Figure 1 and Figure 2; we store
them in the local repository and list the local URLs
for the images associated with the questions and
the solutions in PhoPile, respectively. To highlight
the position of the image appearing in the text, we
mark n-th image as ###img_n### . Regarding the
captions of these images, similar to MathVista (Lu
et al., 2024), we omit the useless image labels (like
‘Figure 05’ or ‘figure 1’) and add the meaningful
captions that contain crucial content relevant to the
problem into the question description, such as ‘Fig-
ure 1: Isosceles glass prism with an apex angle of
90◦’.
(4) Hierarchical Question Structure.Unlike
mathematical datasets, physics problems often con-
tain several sub-questions that may need to be an-
swered in a specific order. We organize these sub-
questions using Arabic numerals as indices. For
the rest of the paper, we do not distinguish which
main question the sub-questions belong to, except
during evaluation.
(5) Multiple Solutions.For certain questions, the
source files provide multiple solutions. They of-
ten appear as ‘Solution 2’, ‘Another way to solve
this problem’. For the completeness of the question
and subsequent development, we also store them in-
dexed by ‘solution 1’, ‘solution 2’, etc. Illustrative
examples are provided in Appendix.
2.3 Data Analysis
Statistics.Token statistics are summarized in Ta-
ble 2. With the exception of EuPho, the majority
of questions and solutions contain fewer than 500
tokens. This size is well within the context window
of current popular LLMs such as Llama-2 (Touvron
et al., 2023) and its variants, allowing for complete
inference or training without the need for prompt
pruning.
2https://mathpix.com

Evaluation Set
w/ image 117
w/o image 273
Retrieval Corpus
w/ image 879
w/o image 1,783
Table 1: Number of questions with or without images.
050010001500
APhO EuPhO IPhO NBPhO RMPhO USAPhO WoPhO BPhONumber of Questions in PhOPile
w/ images w/o images34.31 %
26.34 %
24.10 %32.64 %
45.21 %12.95 %
52.17 %33.45%Figure 2: Proportion of image-containing questions
across different dataset sources.
Source # Questions # Tokens # Tokens per q (Max/Min/Ave) Years
APhOevaluation set 87 19,231 1,677/10/221 2019-2021
retrieval corpus 502 85,949 1,708/7/171 2000-2018
EuPhOevaluation set 15 2,489 3,951/20/902 2019-2021
retrieval corpus 8 1,434 278/24/179 2017-2018
IPhOevaluation set 93 16,871 854/15/181 2019-2021
retrieval corpus 854 120,995 846/8/142 1967-2018
NBPhOevaluation set 55 6,856 468/15/125 2019-2021
retrieval corpus 374 36,400 569/5/97 2003-2018
RMPhevaluation set 63 11,828 1,399/15/188 2019-2021
retrieval corpus 132 15,251 792/9/116 2012-2018
USAPhOevaluation set 77 6,373 262/10/82 2019-2021
retrieval corpus 646 58,908 739/4/91 2007-2019
WoPhOevaluation set 0 - - -
retrieval corpus 146 17,075 813/9/117 2011-2013
Totalevaluation set 390 63,648 3,951/10/ 2019-2021
retrieval corpus 2,662 336,012 1,708/4/ 1967 - 2019
Table 2: The token statistics ofPhOPile.
Images.The overall image statistics are sum-
marized in Table 1. Images are widely present in
PhoPile, appearing in both questions and solutions,
with 33% of problems containing at least one im-
age.
Their distribution across different sources is
shown in Figure 2. These images play an essential
role in conveying information such as experimental
setups, physical systems, and data visualizations
that cannot be fully captured by text alone.
In our PhoPile, about two-thirds of the solutions
include images, ranging from curve plots that illus-
trate variable relationships to structural diagrams
for force analysis, among other types. Although
most current LMMs are unable to generate images
that precisely meet the requirements posed by the
problems, we deliberately preserve these instances
to encourage and support future research on multi-
modal reasoning in physics.3 Experiments
In this section, we introduce the RAG pipeline,
describe the evaluation workflow, and present the
experimental results of foundation models.
3.1 RAG Pipeline
The RAG pipeline comprises two main compo-
nents—the retriever and the generator—and we
further incorporate a reflection mechanism to en-
hance performance.
Retriever.Given an input query q, the retriever
searches an external retrieval corpus Dto find the
most relevant problems and solutions. A scoring
function f(q, d i)assigns relevance scores to each
itemdi∈ D, and the top- kitems with the highest
scores are selected:
R(q) ={d 1, d2, . . . , d k}whered i∈ D.
In PhoPile, Dconsists of 2,662 physics compe-
tition problems with reference answers collected

Your task is to answer the physics questions. The
mathematical formulas are provided in Latex code.
There are some related questions and their an-
swers you may find helpful. \n Here are the exam-
ples:
Question: {Retrieved Question 1}
Reference answer: {Reference Answer to Ques-
tion 1}
Question: {Retrieved Question 2}
Reference answer: {Reference Answer to Ques-
tion 2}\n
The question that you need to solve is: \n {Ques-
tion to be answered} \n\n
Respond with the FINAL answer to the question
to get a higher score as possible as you can, rather
than only give directions or suggestions for solv-
ing the problem. Do NOT use the conditions in
the example questions to solve the question.
Figure 3: Instruction prompt template for the generator
to answer the question.
before 2019. The retriever returns the top- krele-
vant question–answer pairs R(q) ={(q i, ai)}k
i=1,
where qiandaidenote the retrieved question and
answer, respectively. The scoring function can thus
be rewritten asf(q, q i).
As there is no domain-specific retriever for
physics, we adopt general-purpose retrieval meth-
ods. For text-only retrieval, we adopt several
representative methods: (i) a sparse retriever,
BM25 (Robertson and Zaragoza, 2009); and (ii)
dense retrievers, including Emb-cos (embedding
modelall-MiniLM-L6-v2(Wang et al., 2020) with
cosine similarity), Dragon+ (Lin et al., 2023), and
Contriever (Izacard et al., 2022). For multimodal
retrieval, we employ CLIP (Radford et al., 2021),
ALIGN (Jia et al., 2021), and VisualBERT (Li et al.,
2019) to obtain joint text–image embeddings, and
use cosine similarity to identify the closest prob-
lems, which are likewise used as exemplars.
Generator.The generator takes the retrieved
pairsR(q) together with the original query qand
produces a coherent, contextually grounded re-
sponser:
r=G(p(q,R(q))),
where pdenotes the prompt template that integrates
the retrieved information, and Gis the generative
model. An illustration of the prompt design is pro-
vided in Figure 3. In our evaluation, we benchmark
both text-based LLMs and vision–language LMMs,
including Llama-2-13B (Touvron et al., 2023),
GPT-3.5, Llama-3-70B, DeepSeek-Math (Shao
et al., 2024), GPT-4 (OpenAI, 2023a), Gemini-Your task is to choose the answer with a higher
score of the given physics problem.\n\n
Question: {Question to be answered} \n
Answer 1: {Candidate answer without RAG} \n
Answer 2: {Candidate answer with RAG}\n\n
Please give a reason and output the final answer
number in side “##”, for example, ##1##.
Figure 4: Instruction prompt template for reflection to
choose the answer with or without RAG.
Pro (GoogleAI, 2024), GPT-4V (OpenAI, 2023a),
and Gemini-Pro-Vision.
Reflection.We use a reflection mechanism
based on GPT-4. While retrieved questions and
solutions can improve the generator’s performance,
they may also introduce noise that misleads rea-
soning. One cause is the long input context af-
ter retrieval, which can dilute focus and increase
distraction. The reflection mechanism can miti-
gate this issue. As illustrated in Figure 4, given a
physics problem and two candidate answers—one
generated with RAG and one without—the model
is prompted to compare their relative quality and
select the response it considers more accurate. This
self-reflection step reduces the negative impact of
noisy retrievals and enhances the robustness of the
final output.
Workflow.Given a problem, we apply our RAG
pipeline to solve it. For problems with a single
question, we directly prompt the LLM to solve
it while inserting the retrieved question and an-
swers into the prompt. For problems with multi-
ple sub-questions, we retrieve question and answer
pairs according to the sub-question to be solved
and incorporate the retrieved context into the query.
The LLM is then prompted to generate a solution,
which is appended to the query as additional con-
text. This updated query is fed back to the LLM to
solve the next sub-question. The process repeats
until all sub-questions under the same main prob-
lem are answered, enabling the model to leverage
both retrieved knowledge and its own prior reason-
ing history.
3.2 Evaluation
k 0 1 2 3
Accuracy (%) 37 49 73 87
Table 3: Accuracy of GPT-4 as a grader across different
tolerance thresholdsk.

You are a professional physicist and you will
grade answers provided by physics students by
reference to standard answers. The full score is
10 points, and the minimum score is 0 points.
If the student gives the final answer, full marks
will be awarded directly. If the student does
not give the final answer or the final answer is
incorrect, please score based on the proportion
of correct calculation steps given by the student.
You only need to output a score number.
Standard answer: {Reference Answer}
Student answer: {Candidate Answer from Gen-
erator}
Figure 5: Prompt template for the evaluator to score the
candidate answer, given the reference answer.
Marking Scheme.Physics Olympiad problems
typically require long chains of logical reasoning;
therefore, scoring must account not only for the
correctness of the final answer but also for the
quality of intermediate reasoning steps. Instead of
adopting the original competition-specific marking
schemes, which assign varying maximum scores
across problems, we standardize the scoring by set-
ting the maximum score of each problem to ten
points.
To this end, we design anLLM-as-judgeevalu-
ation framework tailored for physics, supporting
both holistic and step-wise scoring. As illustrated
in Figure 5, GPT-4 is employed as the grader: it
is prompted to compare candidate model solutions
against the reference solution and assign a score
between0and10.
To examine the reliability of GPT-4 as a grader,
we conducted a human evaluation on thePhoPile-
Testset. Three experienced instructors who regu-
larly provide training for the IPhO independently
graded the model-generated solutions based on the
same rubric that was provided to the LLMs for
scoring. The final human score for each solution
was obtained by averaging the three annotators’
scores. GPT-4 was prompted with the same rubric
and asked to assign scores to the same set of solu-
tions. We then compared GPT-4’s scores with the
averaged human annotations under varying toler-
ance thresholds k, where kdenotes the maximum
allowable difference between the GPT-4 score and
the human-assigned score, as shown in Table 3.
Results show that although exact agreement is not
achieved, GPT-4 provides sufficiently consistent
judgments to capture relative performance differ-
ences across models.Evaluation Metric.We report both the average
score (AS) the LLMs and LMMs earn and their
pass rate (PR) overPhoPile-TestandPhoPile(V)-
Test. The average score is defined as,
AS=Total points gained by candidate model
Number of questions×10×100%.(1)
We regard the generator (LLMs and LMMs) as
successfully passing a problem if they answer the
problem correctly and earn a score of 10. Therefore
pass rate is defined as,
PR= 100%·Nfull-score
Ntotal.(2)
Intuitively, the pass rate and average score reflect
the performance of the retriever and generator in a
coupled manner. The higher the value of the pass
rate and average score, the better the performance
of the retriever and generator.
Baselines.For thePhoPile-Testsubset contain-
ing text-only questions, we evaluate strong pub-
licly available instruction-tuned LLMs, includ-
ing LLaMA-3 (Meta, 2024), as well as propri-
etary models trained with private data, such as
GPT-3.5, GPT-4 (OpenAI, 2023a), and Gemini-
Pro (Deepmind, 2023). We additionally evaluate
DeepSeekMath-7B, which is trained with math-
related tokens and achieves comparable perfor-
mance of GPT-4.
For thePhoPile(V)-Testsubset with images, we
use LMMs including GPT-4V (OpenAI, 2023b)
and Genimi-Pro-Vision (Deepmind, 2023). By de-
fault, we evaluate the zero-shot CoT (Wei et al.,
2022) performance for the baselines. To eval-
uate RAG’s effectiveness in solving Olympic-
level mathematical physics problems, we exper-
iment with the different RAG methods, where a
model generates output given the input question
prepended with the top-K retrieved examples.
In addition, we finetune several open-source
models on the retrieval corpus, which consists
of question–answer pairs. The models include
Mistral-7B-v0.3, Phi-3.5-mini, LLaMA-3-8B, and
Mathstral-7B-v0.1. We denote the fine-tuned ver-
sions with the suffix ‘-FT’, for example: Mistral-
7B-v0.3-FT, Phi-3.5-mini-FT, LLaMA-3-8B-FT,
and Mathstral-7B-v0.1-FT.
3.3 Main Results
We report the evaluation results of the foundation
models with text-only retrieval under the 2-shot

Model Input w/o RAG Emb-cos BM25 Dragon+ Contriever
PR(AS) PR(AS) PR(AS) PR(AS) PR(AS)
OnPhoPile-Test
Llama-3-70B T 10.51(1.34) 5.4(1.84) 19.07(4.86) 13.62(4.83) 10.28(4.65)
Llama-3-70B w/ Reflection T 10.51(1.34) 19.38(4.35) 19.38(4.35) 14.51(4.81) 10.80(4.60)
DeepSeek-Math T 4.10(0.64) 2.06 (0.27) 2.06(0.29) 2.06(5.93) 3.08(0.38)
DeepSeek-Math w/
ReflectionT 4.10(0.64) 16.95(2.85) 3.59 (0.55) 2.83(6.085) 3.37 (0.54)
GPT-3.5 T 7.95(4.12) 8.72(4.02) 8.23(3.84) 10(3.75) 7.69(3.91)
GPT-3.5 w/ Reflection T 7.95(4.12) 11.79(4.9) 9.23(4.37) 10.26(4.25) 8.46(43.21)
Gemini-Pro T 17.18(5.30) 16.15(4.91) 15.90(4.93) 16.41(5.69)30.51(5.19)
Gemini-Pro w/ Reflection T 17.18(5.30) 21.54(5.72) 20.51(5.56) 18.72(5.65) 19.74(5.49)
GPT-4 T 26.41(6.27) 24.10(5.71) 25.19(5.92) 25.71(5.91) 25.19(5.82)
GPT-4 w/ Reflection T 26.41(6.27) 27.92(6.22) 27.69(6.37) 28.46(6.34) 26.99(6.23)
Mistral-7B-v0.3-FT T 1.47(2.10) 22.64(5.38) 20.90(4.10) 25.28(4.62) 23.28(6.15)
Phi-3.5-mini-FT T 2.56(1.95) 18.00(7.18) 16.25(6.43) 20.31(7.95) 21.44(7.46)
Llama-3-8B-FT T 5.86(2.17) 28.31(5.90) 26.44(5.38) 27.46(5.91) 25.39(6.19)
Mathstral-7B-v0.1-FT T 6.62(2.84) 27.17(5.91)29.02(9.28)28.90(8.21) 27.66(8.74)
OnPhoPile(V)-Test
Gemini-Pro-V T, I 12.82(5.09) 17.95(5.24) 12.82(4.78) 12.88(4.858) 14.96(5.04)
Gemini-Pro-V w/
ReflectionT, I 12.82(5.09) 19.23(5.24) 16.67(5.05) 15.38(4.83) 17.09(5.06)
GPT-4V T, I 21.79(6.26) 20.51(5.43) 7.69(2.50) 21.46(5.53) 21.79(5.65)
GPT-4V w/ Reflection T, I 21.79(6.26)21.89(6.09)19.31(5.25) 21.03(6.16) 21.46(6.20)
Table 4: Evaluation results on PhoPile.Input: T= question text only;T, I= question text with images. Values
indicate pass rates (PR) in percentages, with average scores (AS) in parentheses. Bold values denote the best
performance. All retrievers are text-only.
Model Input w/o RAG CLIP VisualBERT ALIGN
PR(AS) PR(AS) PR(AS) PR(AS)
Gemini-Pro-V T, I 12.82(5.09) 17.48(4.99) 13.59(3.42) 14.56(5.88)
Gemini-Pro-V w/ Reflection T, I 12.82(5.09) 14.56(5.12)17.48(5.28)15.53(5.35)
GPT-4V T, I 21.79(6.26)30.10(6.20)24.27(5.80) 15.53(5.79)
GPT-4V w/ Reflection T, I 21.79(6.26) 26.41(5.99) 22.33(5.58) 23.30(5.71)
Table 5: Evaluation results onPhoPile(V)-Testwith multimodal retrieval. Bold values denote the best-performing
retriever for each model.
setting in Table 4, including LLMs onPhoPile-
Testand LMMs onPhoPile(V)-Test. RAG provides
some insights to aid the generators’ physical rea-
soning. For example, Gemini-Pro combined with
Contriever improves substantially, from 17.18%
to 30.51%, while LLaMA-3-70B with BM25 in-
creases from 10.51% to 19.07%. However, not
all retrievers yield positive effects. In many cases,
performance decreases with RAG, primarily due
to noise and irrelevant content introduced by the
retrieved examples. For LMMs with RAG, GPT-4
consistently outperforms Gemini in both pass rate
and average score. With the incorporation of RAG,
Gemini-Pro improves from 12.82% to 17.95%. We
also investigate reflection in the RAG framework as
a means to mitigate the negative impact of retrieved
questions. This mechanism yields noticeable per-
formance improvements.Interestingly, among open-source models, we
observe substantial improvements after fine-tuning
even for those with fewer than 8 billion parame-
ters. Their overall performance increased by fac-
tors ranging from 5 to 17. The strongest model,
Mathstral, achieved an accuracy of 29.02, which is
already comparable to the best closed-source result
of 30.51.
The results of multimodal retrieval based on joint
text–image embeddings are presented in Table 5.
Both Gemini-Pro-V and GPT-4V show improve-
ments after applying multimodal RAG, though the
extent varies across retrievers. GPT-4V benefits
most from CLIP, reaching a pass rate of 30.10%,
while Gemini-Pro-V gains more from VisualBERT,
where reflection boosts its performance to 17.48%.
These results demonstrate that the choice of multi-
modal retriever significantly impacts performance,

Model #Shots Emb-cos BM25 Dragon+ Contriever
PR(AS) PR(AS) PR(AS) PR(AS)
GPT-3.51 8.97(3.88) 6.92(3.65) 9.74(3.87) 0.77(0.62)
2 8.72(4.02) 8.23(3.84) 10.00(3.75) 7.69(3.91)
3 9.74(3.90) 6.41(3.77) 7.44(3.70) 7.71(3.88)
GPT-41 26.74(6.00) 22.82(5.70) 26.41(6.01) 28.97(6.10)
2 24.10(5.71) 25.19(5.92) 25.71(5.91) 25.19(5.82)
3 25.90(6.01) 22.56(5.65) 22.37(5.89) 24.62(5.91)
Table 6: Evaluation results of GPT-3.5 and GPT-4 with different numbers of retrieved examples. The rows
correspond to different numbers of shots (1, 2, and 3), with values in parentheses indicating standard deviations.
with CLIP particularly effective for GPT-4V and
VisualBERT more favorable for Gemini-Pro-V .
Error analysis.In many cases, retrieved exam-
ples had a negative impact on performance, pri-
marily due to the following reasons: 1) The gen-
eral retriever was not effectively applied to physics
problems, as retriever specific to physics may con-
sider the questions that using the same theorem as
the top-k relevant ones, instead of those with high-
est semantic similarity. Therefore, it highlights the
significance of establishing domain-specific retriev-
ers. 2) The format in retrieved questions misleads
the candidate models’ answering. The retrieved
questions and their reference answer may provide
guidance answers instead of directly answering
the question. Therefore, the foundation models
may refuse to answer the final answer directly and
answer with some guidance for the question. 3)
Additionally, some wrong answers arise from us-
ing conditions in the retrieved questions as if they
were the known conditions in the current question,
demonstrating the significance of noise robustness
in foundation models with RAG (Gao et al., 2023).
Please refer to Appendix for the examples.
3.4 Ablation
Table 6 reports the evaluation results of GPT-3.5
and GPT-4 using different retrieval methods: Emb-
cos, BM25, Dragon+, and Contriever, with varying
numbers of retrieved examples: 1-shot, 2-shot, and
3-shot. The results show that the top-1 retrieved
example from Contriever can negatively affect the
performance of GPT-3.5, whereas Emb-cos and
Dragon+ provide more useful examples, particu-
larly in the 1-shot and 2-shot settings.
4 Related Work
RAG (Chen et al., 2024; Caffagni et al., 2024; Gao
et al., 2023; Zhang et al., 2024) has recently drawnsignificant attention in complimenting the domain-
specific expertise for LLMs (Li et al., 2023a; Shen
et al., 2023; Zhang et al.), or constructing demon-
strations for in-context learning (ICL) (Poesia et al.,
2022; Agrawal et al., 2023; Liu et al., 2022; Hu
et al., 2022; Li et al., 2023b), thus serves as a natu-
ral way to enhance foundation model’s capability
of physic reasoning by integrating external knowl-
edge sources. Let’s consider a high school student
preparing for an exam; it is natural for the student to
review past exam questions to find similar types of
problems to practice. These questions can provide
similar problem-solving approaches and relevant
knowledge applications, much like how RAG can
retrieve and incorporate pertinent information and
similar demonstrations to enhance the reasoning
and accuracy of LLMs, resulting in more informed
and contextually relevant responses.
In recent years, LLMs have developed very
rapidly, providing great convenience for people’s
needs in all aspects of life (Wang et al., 2025; Liu
et al., 2024; Hu et al., 2025; Fan, 2024; Schipper
et al., 2025). These models, like GPT-3 (Brown
et al., 2020), GPT-4 (OpenAI, 2023a) and Gem-
ini (Deepmind, 2023) have already shown great
performance in terms of accuracy, interpretability,
and multimodality, similarly as general LLMs, they
show outstandingly high performance of natural
science QA and mathematical reasoning. Mean-
while, a range of excellent open source models,
including T5 (Raffel et al., 2020), GPT-2 (Rad-
ford et al., 2019) and Llama-2, is available for re-
searchers to enhance further, by training them on
a specialized dataset to attain superior capabilities
compared to generalized models (Magister et al.,
2023; Shridhar et al., 2022; Zhang et al., 2025).
Consequently, a series of outstanding open source
models that are specifically trained and fine-tuned
on math have emerged, such as DeepSeekMath,
Llema (Azerbayev et al., 2023) and Goat (Liu and

Low, 2023). Additionally, there are also a few
models focusing on formal proof such as Lean-
Dojo (Yang et al., 2023); these are models trained
on math-specialized corpus or datasets. However,
in the expansive domain of mathematics, the multi-
tude of sub-disciplines presents a significant chal-
lenge for models with constrained parameters to
adequately address comprehensive mathematical
problems. Studies like Boosting LLM Reason-
ing (Huang et al., 2023a) and LeanDojo (Yang
et al., 2023) use a retrieval-augmented approach
to improve the accuracy of mathematical problem-
solving. It is noteworthy that research at the inter-
section of linguistics and natural sciences remains
relatively scarce. Scholars have placed a greater
emphasis on mathematical reasoning.
Models which demonstrate excellent perfor-
mance on mathematical ability are inseparable
from high-quality datasets and corpus such as
Mathpile (Wang et al., 2023b), proof-pile-2 (Azer-
bayev et al., 2023), MiniF2F (Zheng et al., 2021),
MATH (Hendrycks et al., 2021), GSM8K (Cobbe
et al., 2021), MLFMF (Bauer et al., 2023), Math-
Odyssey (Fang et al., 2025) and the corpus pro-
posed by DeepSeekMath. The aforementioned
datasets consist solely of textual data; however,
it is commonly understood that the interpretation
of mathematical problems often requires the analy-
sis of images. Consequently, MathVista (Lu et al.,
2024) introduced a specialized image-based mathe-
matical dataset and conducted evaluations of mod-
els such as GPT-4, GPT-3.5, Claude-2 (Anthropic,
2023), and mPLUG-Owl-Llama (Ye et al., 2023)
from various perspectives: purely textual input,
text with captions and image OCR (Augmented-
LLMs), and multimodal analysis. However, there is
a noticeable paucity of specialized research linking
LLMs with the discipline of physics. The relevant
work in this area is confined to a minimal subset
of physics-related data within certain natural sci-
ence datasets, such as SciQ Dataset (Welbl et al.,
2017), ScienceQA (Lu et al., 2022), C-eval (Huang
et al., 2023b), E-EV AL (Hou et al., 2024), and
TheoremQA (Chen et al., 2023).
5 Conclusion
In this work, we presentPhoPile, a benchmark de-
signed to comprehensively evaluate the ability of
foundation models to perform physics reasoning
with retrieval-augmented generation (RAG) across
both text-only and image-based questions. Webenchmark a range of mainstream foundation mod-
els, including both large language models (LLMs)
and large multimodal models (LMMs), together
with multiple retrievers. To ensure robust evalu-
ation, we introduce an LLM-as-judge framework
capable of assessing diverse solution formats. Our
results demonstrate that combining physics corpora
with retrieval can improve performance, while also
revealing challenges that motivate further research
in retrieval-augmented physics reasoning.
Limitations
Our retrieval corpus is limited in scale due to con-
straints in data acquisition, including restricted ac-
cess to diverse sources and practical challenges in
collection and integration. As future work, we plan
to incorporate multimodal cross-referencing, which
would enable richer interactions between text, for-
mulas, and images, and has the potential to further
improve the accuracy and robustness of foundation
models for physics reasoning.
References
2024. Large language models are neurosymbolic rea-
soners. 38:17985–17993.
AAPT. 2007-2019. United states physics olympiad
official website. Accessed: 2024-02-15.
Sweta Agrawal, Chunting Zhou, Mike Lewis, Luke
Zettlemoyer, and Marjan Ghazvininejad. 2023. In-
context examples selection for machine translation.
InFindings of the Association for Computational
Linguistics: ACL 2023, pages 8857–8873, Toronto,
Canada. Association for Computational Linguistics.
Anthropic. 2023. Claude 2: Improvements and
capabilities of anthropic’s second-generation ai
chatbot. https://www.anthropic.com/news/
claude-2. Accessed: 2024-02-14.
APhO. 2000-2021. Asian physics olympiad 2019 offi-
cial website. Accessed: 2024-02-15.
Zhangir Azerbayev, Hailey Schoelkopf, Keiran Paster,
Marco Dos Santos, Stephen McAleer, Albert Q.
Jiang, Jia Deng, Stella Biderman, and Sean Welleck.
2023. Llemma: An open language model for mathe-
matics.Preprint, arXiv:2310.10631.
Andrej Bauer, Matej Petkovi ´c, and Ljup ˇco Todor-
ovski. 2023. Mlfmf: Data sets for ma-
chine learning for mathematical formalization.
https://arxiv.org/abs/2310.16005.
Julia Anna Bingler, Mathias Kraus, Markus Leippold,
and Nicolas Webersinke. 2022. Cheap talk and

cherry-picking: What climatebert has to say on cor-
porate climate risk disclosures.Finance Research
Letters, 47:102776.
BPhO. 2001-2022. British physics olympiad official
website. Accessed: 2024-02-15.
Tom B. Brown, Benjamin Mann, Nick Ryder, Melanie
Subbiah, Jared Kaplan, Prafulla Dhariwal, Arvind
Neelakantan, Pranav Shyam, Girish Sastry, Amanda
Askell, Sandhini Agarwal, Ariel Herbert-V oss,
Gretchen Krueger, T. J. Henighan, Rewon Child,
Aditya Ramesh, Daniel M. Ziegler, Jeff Wu, Clemens
Winter, and 12 others. 2020. Language models are
few-shot learners.ArXiv, abs/2005.14165.
Davide Caffagni, Federico Cocchi, Nicholas Moratelli,
Sara Sarto, Marcella Cornia, Lorenzo Baraldi, and
Rita Cucchiara. 2024. Wiki-llava: Hierarchical
retrieval-augmented generation for multimodal llms.
arXiv preprint arXiv:2404.15406.
Jiawei Chen, Hongyu Lin, Xianpei Han, and Le Sun.
2024. Benchmarking large language models in
retrieval-augmented generation.Proceedings of
the AAAI Conference on Artificial Intelligence,
38(16):17754–17762.
Wenhu Chen, Ming Yin, Max Ku, Elaine Wan,
Xueguang Ma, Jianyu Xu, Tony Xia, Xinyi Wang,
and Pan Lu. 2023. Theoremqa: A theorem-
driven question answering dataset.arXiv preprint
arXiv:2305.12524.
Karl Cobbe, Vineet Kosaraju, Mohammad Bavarian,
Mark Chen, Heewoo Jun, Lukasz Kaiser, Matthias
Plappert, Jerry Tworek, Jacob Hilton, Reiichiro
Nakano, Christopher Hesse, and John Schulman.
2021. Training verifiers to solve math word prob-
lems.arXiv preprint arXiv:2110.14168.
Deepmind. 2023. Gemini: A family of highly capable
multimodal models.ArXiv, abs/2312.11805.
Zhengxiao Du, Yujie Qian, Xiao Liu, Ming Ding,
Jiezhong Qiu, Zhilin Yang, and Jie Tang. 2022. Glm:
General language model pretraining with autoregres-
sive blank infilling. InProceedings of the 60th An-
nual Meeting of the Association for Computational
Linguistics (Volume 1: Long Papers), pages 320–335.
EuPhO. 2017-2021. European physics olympiad official
website. Accessed: 2024-02-15.
Meng Fang, Xiangpeng Wan, Fei Lu, Fei Xing, and
Kai Zou. 2025. Mathodyssey: Benchmarking math-
ematical problem-solving skills in large language
models using odyssey math data.Scientific Data,
12(1):1392.
Yunfan Gao, Yun Xiong, Xinyu Gao, Kangxiang Jia,
Jinliu Pan, Yuxi Bi, Yi Dai, Jiawei Sun, and Haofen
Wang. 2023. Retrieval-augmented generation for
large language models: A survey.arXiv preprint
arXiv:2312.10997.GoogleAI. 2024. Gemini 1.5: Unlocking multimodal
understanding across millions of tokens of context.
ArXiv, abs/2403.05530.
Chaoqun He, Renjie Luo, Yuzhuo Bai, Shengding
Hu, Zhen Leng Thai, Junhao Shen, Jinyi Hu,
Xu Han, Yujie Huang, Yuxiang Zhang, and 1 oth-
ers. 2024. Olympiadbench: A challenging bench-
mark for promoting agi with olympiad-level bilin-
gual multimodal scientific problems.arXiv preprint
arXiv:2402.14008.
Dan Hendrycks, Collin Burns, Saurav Kadavath, Akul
Arora, Steven Basart, Eric Tang, Dawn Song, and
Jacob Steinhardt. 2021. Measuring mathematical
problem solving with the math dataset.NeurIPS.
Jinchang Hou, Chang Ao, Haihong Wu, Xiangtao Kong,
Zhigang Zheng, Daijia Tang, Chengming Li, Xiping
Hu, Ruifeng Xu, Shiwen Ni, and Min Yang. 2024. E-
eval: A comprehensive chinese k-12 education evalu-
ation benchmark for large language models.ArXiv,
abs/2401.15927.
Xueyu Hu, Tao Xiong, Biao Yi, Zishu Wei, Ruixuan
Xiao, Yurun Chen, Jiasheng Ye, Meiling Tao, Xi-
angxin Zhou, Ziyu Zhao, and 1 others. 2025. Os
agents: A survey on mllm-based agents for com-
puter, phone and browser use. InProceedings of the
63rd Annual Meeting of the Association for Compu-
tational Linguistics (Volume 1: Long Papers), pages
7436–7465.
Yushi Hu, Chia-Hsuan Lee, Tianbao Xie, Tao Yu,
Noah A. Smith, and Mari Ostendorf. 2022. In-
context learning for few-shot dialogue state tracking.
InFindings of the Association for Computational
Linguistics: EMNLP 2022, pages 2627–2643, Abu
Dhabi, United Arab Emirates. Association for Com-
putational Linguistics.
Xijie Huang, Li Lyna Zhang, Kwang-Ting Cheng, and
Mao Yang. 2023a. Boosting llm reasoning: Push the
limits of few-shot learning with reinforced in-context
pruning.ArXiv, abs/2312.08901.
Yuzhen Huang, Yuzhuo Bai, Zhihao Zhu, Junlei
Zhang, Jinghan Zhang, Tangjun Su, Junteng Liu,
Chuancheng Lv, Yikai Zhang, Jiayi Lei, Fan-
chao Qi, Yao Fu, Maosong Sun, and Junxian He.
2023b. C-eval: A multi-level multi-discipline chi-
nese evaluation suite for foundation models.ArXiv,
abs/2305.08322.
Shima Imani, Liang Du, and H. Shrivastava. 2023.
Mathprompter: Mathematical reasoning using large
language models. InAnnual Meeting of the Associa-
tion for Computational Linguistics.
IPhO. 1967-2023. International physics olympiad unof-
ficial website. Accessed: 2024-02-15.
Gautier Izacard, Mateusz Barto ´n, Marcin Junczys-
Dowmunt, Sebastian Ruder, Armand Joulin, and
Edouard Grave. 2022. Unsupervised dense infor-
mation retrieval with contrastive learning.Preprint,
arXiv:2202.06991.

Ziwei Ji, Nayeon Lee, Rita Frieske, Tiezheng Yu, Dan
Su, Yan Xu, Etsuko Ishii, Ye Jin Bang, Andrea
Madotto, and Pascale Fung. 2023. Survey of halluci-
nation in natural language generation.ACM Comput-
ing Surveys, 55(12):1–38.
Chao Jia, Yinfei Yang, Ye Xia, Yi-Ting Chen, Zarana
Parekh, Hieu Pham, Quoc V Le, Tom Duerig, and
Yunhui Song. 2021. Scaling up visual and vision-
language representation learning with noisy text su-
pervision.arXiv preprint arXiv:2102.05918.
Liunian Harold Li, Yen-Chun Chen Su, Jianwei Xing,
Xiaowei Li, and Jianfeng Gao. 2019. Visualbert: A
simple and performant baseline for vision and lan-
guage. InProceedings of the 33rd Conference on
Neural Information Processing Systems (NeurIPS).
Xianzhi Li, Xiaodan Zhu, Zhiqiang Ma, Xiaomo Liu,
and Sameena Shah. 2023a. Are chatgpt and gpt-
4 general-purpose solvers for financial text analyt-
ics? an examination on several typical tasks.arXiv
preprint arXiv:2305.05862.
Xiaonan Li, Kai Lv, Hang Yan, Tianyang Lin, Wei Zhu,
Yuan Ni, Guotong Xie, Xiaoling Wang, and Xipeng
Qiu. 2023b. Unified demonstration retriever for in-
context learning. InProceedings of the 61st Annual
Meeting of the Association for Computational Lin-
guistics (Volume 1: Long Papers), pages 4644–4668,
Toronto, Canada. Association for Computational Lin-
guistics.
Sheng-Chieh Lin, Akari Asai, Minghan Li, Barlas Oguz,
Jimmy Lin, Yashar Mehdad, Wen tau Yih, and Xilun
Chen. 2023. How to train your dragon: Diverse
augmentation towards generalizable dense retrieval.
Preprint, arXiv:2302.07452.
Jiachang Liu, Dinghan Shen, Yizhe Zhang, Bill Dolan,
Lawrence Carin, and Weizhu Chen. 2022. What
makes good in-context examples for GPT-3? In
Proceedings of Deep Learning Inside Out (DeeLIO
2022): The 3rd Workshop on Knowledge Extrac-
tion and Integration for Deep Learning Architectures,
pages 100–114, Dublin, Ireland and Online. Associa-
tion for Computational Linguistics.
Tiedong Liu and Kian Hsiang Low. 2023. Goat: Fine-
tuned llama outperforms gpt-4 on arithmetic tasks.
ArXiv, abs/2305.14201.
Xiao Liu, Hao Yu, Hanchen Zhang, Yifan Xu, Xuanyu
Lei, Hanyu Lai, Yu Gu, Hangliang Ding, Kaiwen
Men, Kejuan Yang, Shudan Zhang, Xiang Deng, Ao-
han Zeng, Zhengxiao Du, Chenhui Zhang, Sheng
Shen, Tianjun Zhang, Yu Su, Huan Sun, and 3 others.
2024. Agentbench: Evaluating LLMs as agents. In
The Twelfth International Conference on Learning
Representations.
Yixin Liu, Avi Singh, C. Daniel Freeman, John D. Co-
Reyes, and Peter J. Liu. 2023. Improving large lan-
guage model fine-tuning for solving math problems.
ArXiv, abs/2310.10047.Pan Lu, Hritik Bansal, Tony Xia, Jiacheng Liu, Chun-
yuan Li, Hannaneh Hajishirzi, Hao Cheng, Kai-
Wei Chang, Michel Galley, and Jianfeng Gao. 2024.
Mathvista: Evaluating mathematical reasoning of
foundation models in visual contexts. InInter-
national Conference on Learning Representations
(ICLR).
Pan Lu, Swaroop Mishra, Tony Xia, Liang Qiu, Kai-
Wei Chang, Song-Chun Zhu, Oyvind Tafjord, Peter
Clark, and Ashwin Kalyan. 2022. Learn to explain:
Multimodal reasoning via thought chains for science
question answering. InThe 36th Conference on Neu-
ral Information Processing Systems (NeurIPS).
Lucie Charlotte Magister, Jonathan Mallinson, Jakub
Adamek, Eric Malmi, and Aliaksei Severyn. 2023.
Teaching small language models to reason. InPro-
ceedings of the 61st Annual Meeting of the Associa-
tion for Computational Linguistics (Volume 2: Short
Papers), pages 1773–1781.
Meta. 2024. Introducing meta llama 3: The most capa-
ble openly available llm to date.
NBPhO. 2003-2021. Nordic-baltic physics olympiad
official website. Accessed: 2024-02-15.
OpenAI. 2022. Chatgpt. https://openai.com/blog/
chatgpt. Accessed: 2023-02-06.
OpenAI. 2023a. Gpt-4: Openai’s generative
pre-trained transformer 4 model.Preprint,
arXiv:arXiv:2301.00000.
OpenAI. 2023b. Gpt-4v(ision) system card.
Gabriel Poesia, Alex Polozov, Vu Le, Ashish Tiwari,
Gustavo Soares, Christopher Meek, and Sumit Gul-
wani. 2022. Synchromesh: Reliable code generation
from pre-trained language models. InInternational
Conference on Learning Representations.
Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya
Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sas-
try, Amanda Askell, Pamela Mishkin, Jack Clark,
Gretchen Krueger, and Ilya Sutskever. 2021. Learn-
ing transferable visual models from natural language
supervision. InProceedings of the 38th International
Conference on Machine Learning (ICML).
Alec Radford, Jeff Wu, Rewon Child, David Luan,
Dario Amodei, and Ilya Sutskever. 2019. Language
models are unsupervised multitask learners. Techni-
cal report, OpenAI. OpenAI Technical Report.
Colin Raffel, Noam Shazeer, Adam Roberts, Kather-
ine Lee, Sharan Narang, Michael Matena, Yanqi
Zhou, Wei Li, and Peter J. Liu. 2020. Exploring the
limits of transfer learning with a unified text-to-text
transformer.Journal of Machine Learning Research,
21(140):1–67.
RMPhO. 2012-2021. Romanian master of physics 2023.
Accessed: 2024-02-15.

Stephen Robertson and Hugo Zaragoza. 2009. The prob-
abilistic relevance framework: Bm25 and beyond.
Foundations and Trends® in Information Retrieval,
3(4):333–389.
Bernardino Romera-Paredes, Mohammadamin
Barekatain, Andrey Novikov, and 1 others. 2024.
Mathematical discoveries from program search with
large language models.Nature, 625:468–475.
Olivier Schipper, Yudi Zhang, Yali Du, Mykola Pech-
enizkiy, and Meng Fang. 2025. Pillagerbench:
Benchmarking llm-based agents in competitive
minecraft team environments. In2025 IEEE Confer-
ence on Games (CoG), pages 1–15. IEEE.
Raymond A Serway, John W Jewett, and Vahé Peroo-
mian. 2000.Physics for scientists and engineers,
volume 2. Saunders college publishing Philadelphia.
Zhihong Shao, Peiyi Wang, Qihao Zhu, Runxin Xu,
Junxiao Song, Mingchuan Zhang, Y . K. Li, Y . Wu,
and Daya Guo. 2024. Deepseekmath: Pushing the
limits of mathematical reasoning in open language
models.arXiv preprint arXiv:2402.03300, v1.
Xinyue Shen, Zeyuan Chen, Michael Backes, and Yang
Zhang. 2023. In chatgpt we trust? measuring
and characterizing the reliability of chatgpt.arXiv
preprint arXiv:2304.08979.
Zijing Shi, Meng Fang, Shunfeng Zheng, Shilong Deng,
Ling Chen, and Yali Du. 2023. Cooperation on the
fly: Exploring language agents for ad hoc teamwork
in the avalon game.Preprint, arXiv:2312.17515.
Kumar Shridhar, Alessandro Stolfo, and Mrinmaya
Sachan. 2022. Distilling reasoning capabilities
into smaller language models.arXiv preprint
arXiv:2212.00193.
Karan Singhal, Shekoofeh Azizi, Tao Tu, and et al. 2023.
Large language models encode clinical knowledge.
Nature, 620:172–180.
Hugo Touvron, Louis Martin, Kevin Stone, Peter Al-
bert, Amjad Almahairi, Yasmine Babaei, Nikolay
Bashlykov, Soumya Batra, Prajjwal Bhargava, Shruti
Bhosale, and 1 others. 2023. Llama 2: Open foun-
dation and fine-tuned chat models.arXiv preprint
arXiv:2307.09288.
Jiaqi Wang, Enze Shi, Huawen Hu, Chong Ma, Yiheng
Liu, Xuhui Wang, Yincheng Yao, Xuan Liu, Bao Ge,
and Shu Zhang. 2025. Large language models for
robotics: Opportunities, challenges, and perspectives.
Journal of Automation and Intelligence, 4(1):52–64.
Wenhui Wang, Hangbo Bao, Li Dong, and Furu Wei.
2020. Minilm: Deep self-attention distillation for
task-agnostic compression of pre-trained transform-
ers.Advances in Neural Information Processing Sys-
tems (NeurIPS), 33:5776–5788.Zengzhi Wang, Rui Xia, and Pengfei Liu. 2023a. Gen-
erative ai for math: Part i - mathpile: A billion-
token-scale pretraining corpus for math.ArXiv,
abs/2312.17120.
Zengzhi Wang, Rui Xia, and Liu Pengfei. 2023b. Gen-
erative ai for math: Part i – mathpile: A billion-token-
scale pretraining corpus for math.arXiv preprint
arXiv:2312.17120.
Jason Wei, Xuezhi Wang, Dale Schuurmans, Maarten
Bosma, Brian Ichter, Fei Xia, Ed Chi, Quoc V . Le,
and Denny Zhou. 2022. Chain of thought prompt-
ing elicits reasoning in large language models. In
Advances in Neural Information Processing Systems
(NeurIPS), volume 35, pages 24824–24837.
Johannes Welbl, Nelson F. Liu, and Matt Gardner. 2017.
Crowdsourcing multiple choice science questions. In
Proceedings of the 2017 Workshop on The Future of
AI: Language, Reasoning, and Cognition.
Miao Xiong, Zhiyuan Hu, Xinyang Lu, YIFEI LI, Jie
Fu, Junxian He, and Bryan Hooi. 2023. Can llms
express their uncertainty? an empirical evaluation of
confidence elicitation in llms. InThe Twelfth Inter-
national Conference on Learning Representations.
Kaiyu Yang, Aidan Swope, Alex Gu, Rahul Chala-
mala, Peiyang Song, Shixing Yu, Saad Godil, Ryan
Prenger, and Anima Anandkumar. 2023. LeanDojo:
Theorem proving with retrieval-augmented language
models. InNeural Information Processing Systems
(NeurIPS).
Qinghao Ye, Haiyang Xu, Guohai Xu, Jiabo Ye, Ming
Yan, Yi Zhou, Junyan Wang, Anwen Hu, Pengcheng
Shi, Yaya Shi, Chenliang Li, Yuanhong Xu, Hehong
Chen, Junfeng Tian, Qiang Qi, Ji Zhang, and Feiyan
Huang. 2023. mplug-owl: Modularization empowers
large language models with multimodality.ArXiv,
abs/2304.14178.
Yudi Zhang, Lu Wang, Meng Fang, Yali Du, Chenghua
Huang, Jun Wang, Qingwei Lin, Mykola Pech-
enizkiy, Dongmei Zhang, Saravan Rajmohan, and
Qi Zhang. 2025. Distill not only data but also re-
wards: Can smaller language models surpass larger
ones?Preprint, arXiv:2502.19557.
Yudi Zhang, Pei Xiao, Lu Wang, Chaoyun Zhang, Meng
Fang, Yali Du, Yevgeniy Puzyrev, Randolph Yao,
Si Qin, Qingwei Lin, and 1 others. Ruag: Learned-
rule-augmented generation for large language models.
InThe Thirteenth International Conference on Learn-
ing Representations.
Zihan Zhang, Meng Fang, and Ling Chen. 2024. Re-
trievalQA: Assessing adaptive retrieval-augmented
generation for short-form open-domain question an-
swering. InFindings of the Association for Compu-
tational Linguistics: ACL 2024, pages 6963–6975,
Bangkok, Thailand. Association for Computational
Linguistics.

Kunhao Zheng, Jesse Michael Han, and Stanislas Polu.
2021. Minif2f: a cross-system benchmark for for-
mal olympiad-level mathematics.arXiv preprint
arXiv:2109.00110.
Shunfeng Zheng, Meng Fang, and Ling Chen. 2025.
SpatialWebAgent: Leveraging large language mod-
els for automated spatial information extraction and
map grounding. InProceedings of the 63rd Annual
Meeting of the Association for Computational Lin-
guistics (Volume 3: System Demonstrations), pages
252–266, Vienna, Austria. Association for Computa-
tional Linguistics.

Appendix
1.Question Examples inPhOPile(Sec. A)
Dataset indexing scheme (“question_number”, “sub_question_number”,
“sub_sub_question_number”) and textual normalization of in-question indices.
Figure:Examples from the dataset (Figure 6).
2.Supplementary Experiments and Examples for GPT-4 Scoring(Sec. B)
Scoring rubric and qualitative marking cases for GPT-4 as evaluator.
Table:Candidate-answer categories and expected scores (Table 7).
Figures:10/5/0-score answer exemplars (Figures 7–9).
3.Prompt Examples(Sec. C)
Prompt formats for pure-text inference and 1-shot RAG.
Figures:Pure text prompt (Figure 10); 1-shot RAG prompt (Figure 11).
4.Gemini Example(Sec. D)
One worked Gemini inference case with 1-shot RAG.
Figure:Gemini output example (Figure 12).
5.Multimodality Example(Sec. E)
Vision-augmented physics problem; standard answer; GPT-4-V vs Gemini-Pro-Vision responses and
scores.
Figure:The image from the problem (Figure 13).
6.Runtime Analysis(Sec. F)
Retrieval vs retrieval+embedding timing across methods.
Table:RAG performance (Table 8).
7.Examples of Answer Errors(Sec. G)
(a)Error Type 1:Gives guidelines instead of a direct answer (projectile problem).
(b)Error Type 2:Misguided by retrieved examples (thermoacoustic temperature amplitude).
(c)Error Type 3:Wrong retrieval (dipole torque/power vs unrelated contexts).
8.Additional Results(Sec. H)
Difficulty distribution; RAG-driven zero→non-zero improvements; average scores by difficulty.
Tables:Difficulty distribution (Table 9); average scores by difficulty (Table 10); questions improved
from 0 to non-0 after RAG (Table 11).

A Question Examples in PhOPile
To fully record the details of the questions, we create ‘question_number’, ‘sub_question_number’,
and ‘sub_sub_question_number’ which stand for question number, first order sub-question number
and second order sub-question number in our dataset to facilitate distinction. Furthermore, within
the text of the questions, we replace the original question index, which typically consist of Arabic
numerals, English letters, Roman numerals, etc, with Arabic number in ‘ sub_question_number ’ or
‘sub_sub_question_number’. We provide examples in Figure 6.
B Supplementary Experiments and Examples for GPT-4 Scoring
As shown in Table 7, we verify the GPT4’s capability on marking the candidate answers by foundation
models, given the reference solution.
Figure 7, Figure 8 and Figure 9 are examples of evaluator marking three different types of answers.

Example 1:
“index”: 2882,
“problem”: "A metal ring of radius Ris made out of a wire of cross section aria s; the resistivity of the wire’s
material is given by ρ. The ring is place in a vertical plane. A small magnetic needle is placed in the center of
the ring. The magnetic needle can rotate freely around the vertical axis which passes through the center of the
ring. Due to mechanical inertia, the needle points in the direction of the average horizontal magnetic field at
the center of the ring. When the ring is motionless in the reference frame of the Earth, the magnetic needle
indicates the direction of the horizontal component of Earth’s magnetic field. When the ring is rotating around
its vertical axis with a constant angular velocity, ω, the magnetic needle deviates, in horizontal plane, from
this direction by an angle α. 1.a. Find the expression of angle of deviation, αof the magnetic needle. Write
your expression forαas a function ofs, ω, ρand of the magnetic permeabilityµ 0.”,
“question_number”: 978,
“sub_question_number”: 1,
“sub_sub_question_number”: null,
“source”: "RMPh”,
“year”: 2021,
“solution”: “For: ⃗BEarth =B h·ˆex+B v·ˆezBh- horizontal component of Earth’s magnetic
field Bv- vertical component of Earth’s magnetic field formula of surface vector ⃗Sfor the ring
⃗S=π·R2·(cos(ω·t)·ˆe x+ sin(ω·t)·ˆe y)expression of magnetic flux ϕof Earth’s magnetic field
through ring’s surface ...”,
“imgQ”: null,
“imgA”: null
Example 2:
“index”: 2883,
“problem”: “A thin ring of mass m, radius r0and inductance Lis maintained in a horizontal plane above a
cylindrical magnetic bar which is placed vertically (see ###img_1###). The vertical axis of symmetry of
the cylindrical magnetic bar is aligned with the center of the ring. The magnetic field due to the cylindrical
magnet is shown in ###img_1### and each of its components are given by: the radial component is given
byBr=B 0·β·r , while the vertical component is given by Bz=B 0·(1−α·z).B 0, α, β are all
positive constant with appropriate dimensions, while zandrdenotes the vertical and, respectively, the radial
coordinate of the system. Initially, there is no electric current passing through the ring and it is kept fixed
above the magnet. It is then allowed to fall due to the gravitational pull of the Earth, given by the gravitational
acceleration, g. During the fall the ring will still be in a horizontal plane and will have the same vertical axis.
Answer the following questions and write your results as a function of the variables specified above. Derive
the equations of motion for the ring in the reference frame specified in ###img_1###.”,
“question_number”: 978,
“sub_question_number”: 2,
“sub_sub_question_number”: 1,
“source”: "RMPh”,
“year”: 2021,
“solution”: “For: expression of magnetic flux through ring’s surface Φ =B z·π·r2
0+L·I
0 =R·I=dΦ
dt- voltage drop on superconducting ring is zero - magnetic flux inside the ring is constant
Φ =B 0·(1−α·z)·π·r2
0+L·I=constant Initial conditions
z(t= 0) = 0
l(t= 0) = 0constant
=B 0·π·r2
0expression of the intensity of electric current through the ring I=B0
L·α·π·r2
0·zradial
component of the force of interaction is zero - because of symmetry vertical component of the force of
interaction Fz=−2π2·α·β·B2
0·r4
0
L·zelastic constant k=2π2·α·β·B2
0·r4
0
Lequations of motion for the ring
m·d2z
dt2+k·z=−m·ggeneral solution of the equations of motion for the ring ...”,
“imgQ”: [ "18.png" ],
“imgA”: null
Figure 6: Examples from the dataset.

Candidate Answer Output
Fully correct solution 10
Fully incorrect solution but with the correct final answer 0
A completely wrong solution 0
Half of the fully correct solution 1-9
Fully correct solution but with a wrong final answer 9
Correct answers in various type (normal format, Latex format, or without units) 10
Table 7: By inputting the candidate solution to be marked and reference answer to GPT4, we can have detailed and
based-on-step scores without human intervention.
Figure 7: A 10 score example answer provided by GPT-3.5 with cosine similarity.

Figure 8: A 5 score example answer provided by GPT-3.5 with cosine similarity.

Figure 9: A 0 score example answer provided by GPT-3.5 with cosine similarity.

C Prompt Examples
In this section, we present the prompt examples used to guide LLMs in solving physical questions and
evaluating solutions. Figure 10 shows a prompt example of pure text inference, while Figure 11 illustrates
a prompt with 1-shot RAG.
Figure 10: Prompt example of pure text inference

Figure 11: Prompt example of pure text inference with 1-shot RAG

D Gemini Example
Figure 12 is an example of an answer provided by Gemini.
Figure 12: An example for Gemini inference with 1-shot RAG using cosine similarity.

E Multimodality Example
Questions: In a simple model, an atomic nucleus can be thought of as a ball consisting of closely packed
nucleons [see Figure 1(a)], where the nucleons are hard balls of radius rN= 0.85fm 
1fm = 10−15m
.
The nuclear force is present only for two nucleons in contact. The volume of the nucleus Vis larger than
the volume of all nucleons AVN, where VN=4
3r3
Nπ. The ratio f=AV N/Vis called the packing factor
and gives the percentage of space filled by the nuclear matter. Calculate what would be the packing factor
fif nucleons were arranged in a "simple cubic" (SC) crystal system, where each nucleon is centered on a
lattice point of an infinite cubic lattice [see Figure 1(b)]. (0.3 points)
Figure 13: The Image from the Question.
Standard Answer:In the SC-system, in each of 8 corners of a given cube there is one unit (atom,
nucleon, etc.), but it is shared by 8 neighboring cubes - this gives a total of one nucleon per cube. If
nucleons are touching, as we assume in our simplified model, then a= 2r Nis the cube edge length
a. The volume of one nucleon is then VN=4
3r3
Nπ=4
3 a
23π=4a3
3·8π=π
6a3from which we obtain
f=VN
a3=π
6≈0.52
GPT-4-Vision Output:(Score: 10)In the "simple cubic" (SC) crystal system, each nucleon is centered
on a lattice point, and because we’re assuming the nucleons to be hard spheres, the sides of the cubic
lattice cell must be twice the radius of a nucleon to accommodate one nucleon inside without overlapping
with others. Given that the radius of a nucleon rNis 0.85 fm (fermi), which is 0.85×10−15meters,
the side length aof the cubic cell will be: a= 2×r NThe volume VNof a single nucleon (since it’s
a sphere) is: VN=4
3πr3
NNow, the volume VSCof the cubic cell which contains one nucleon would
be:VSC=a3= (2r N)3= 8r3
NThe packing factor fwill be the ratio of the nucleon’s volume to
the volume of the cubic cell: f=AVN
V=VN
VSCSince each cell volume VSCcontains one nucleon for
the simple cubic packing, we don’t need to multiply VNbyA, because Awill cancel when we take
the ratio for one cubic cell. Plugging in the expressions for VNandVSCgives: f=4
3πr3
N
8r3
Nalculating
this gives: f=π
6Numerically, this evaluates to approximately: f≈3.14159
6≈0.5236 Or, in percentage
terms: f≈52.36% his means the packing factor or the percentage of space filled by the nuclear matter in
a simple cubic arrangement is approximately 52.36%.
Gemini-Pro-Vision Output:(Score: 2)I see the image you sent and can use it to understand the setup
of the Kelvin water dropper. The absolute value of the charge, Q0, of each droplet as they separate from
the tubes and at the instant the capacitor’s charge is qcan be expressed as: Q0=Cq/(2r max)where: C
is the capacitance. qis the charge on the capacitor. rmaxis the maximum radius of the droplet as it breaks
free from the nozzle This expression is derived based on the following: 1. Charge Quantization: Each
droplet acquires a quantized charge, meaning its charge is a multiple of the fundamental electron charge
(e). As the first droplet falls, it acquires a small charge imbalance, which is then shared equally between
the two electrodes due to the capacitor. 4. Maximum Radius: The maximum radius, rmax, relates to the
amount of charge a droplet can hold due to its surface tension. A larger droplet can hold more charge.
Therefore, the absolute value of the charge on each droplet depends on the capacitance, the capacitor’s
charge, and the maximum radius the droplet can achieve before breaking free. It’s important to note that
this expression neglects the effect mentioned in Part A-iii, where the charge on the droplet can induce a

charge on the metal nozzle. This additional effect can slightly modify the charge acquired by the droplet.
I hope this helps! Let me know if you have any other questions.
F Runtime Analysis
Algorithm Pure Retrieval (s) Retrieval + Embedding (s)
Contriver 23 255
Dragon+ 33 155
Cosine similarity 126 275
BM25 7 7.4
Table 8: Performance Comparison of Different RAG Algorithms in Second.

G Examples of answer errors
G.1 Error Type 1: Give guidelines instead of direct answer
ProblemA ball is launched at speed vat angle θabove the horizontal toward a vertical wall a distance L
away. It bounces elastically off the wall and falls back to its launch point. What was its initialspeed?
Reference Answer√2Lg√
sin(2θ)
GPT-3.5’s responseTo solve this problem, we can consider the motion in two parts: the motion from
the launch point to the wall, and then the motion from the wall back to the launch point.For the first part
of the motion (from the launch point to the wall), we can consider the horizontal motion and the vertical
motion separately. In the horizontal direction, the ball travels a distance Lwith a constant horizontal
velocity. In the vertical direction, the ball travels a vertical distance h, given by: h=1
2gt2where tis the
time taken to reach the wall, and gis the acceleration due to gravity.Since the ball lands back at the launch
point, the total time taken for the entire motion is 2t. The vertical speed of the ball when it reaches the wall
can be calculated as follows: vy=gt . From the kinematic equation, we know that the vertical component
of velocity when hitting the wall is equal to the initial vertical component of velocity multiplied by −1
(due to reflection of velocity), thus: vsin(θ) =−gt . Solving for tgives: t=vsin(θ)
g.Since an elastic
collision preserves kinetic energy, the speed of the ball upon reflection is the same as the incoming speed
vector. Therefore, the total horizontal distance traveled by the ball is 2L. The total horizontal distance
can be calculated as follows: 2L=vcos(θ)·2t . Substitute the expression for t:2L=vcos(θ)·2vsin(θ)
g.
Solving forvgives:v=q
gL
sin(2θ).Therefore, the initial speed of the ball isq
gL
sin(2θ).Score: 4
Example 1 reterieved by cosine similarityA projectile is launched across flat ground at an angle θto
the horizontal and travels in the absence of air resistance. It rises to a maximum height Hand lands a
horizontal distanceRaway. What is the ratioH/R? Reference answer:1
4tanθ.
Example 2 reterieved by cosine similarityA cannonball is launched with initial velocity of magnitude
v0over a horizontal surface. At what minimum angle θminabove the horizontal should the cannonball
be launched so that it rises to a height Hwhich is larger than the horizontal distance Rthat it will travel
when it returns to the ground? Reference answer:θ min= 76◦.
GPT-3.5’s response using two-shot Retrieval-Augmented GenerationWe can break down the motion
of the ball into horizontal and vertical components. When the ball hits the wall and bounces back, the
vertical component of its velocity remains the same while the horizontal component changes direction.
Letvxbe the horizontal component of the initial velocity, and vybe the vertical component of the initial
velocity. Since the motion is symmetrical, the time taken to return to the launch point is twice the
time taken to reach the wall: ttotal= 2·t to wall The time taken to reach the wall can be found using the
horizontal distance Land the x-component of the initial velocity: L=v x·tto wall From this, we solve for
tto wall :tto wall =L
vxNow, considering the vertical motion, at the maximum height Hthe vertical component
of the velocity is zero, hence: v2
y−2·g·H= 0 where gis the acceleration due to gravity. As the initial
velocity is the magnitude of the total velocity: v=q
v2x+v2yFrom the above two equations we can solve
forv.Score: 3
AnalysisIn this case, the LLM only provided advisory comments, stating ‘From the above two equations
we can solve forv’. without continuing to answer the question, resulting in a loss of some points.

G.2 Error Type 2: Misguided by the retrieved examples.
ProblemThe change in the gas temperature due to the adiabatic expansion and contraction, as a result
of the sound wave, takes the form: T(x, t) =T 0−T 1(x) cos(ωt) .Compute the amplitude T1(x)of the
temperature oscillations in terms ofT 0, γ,a, kandx.
Reference AnswerGiven: - Temperature perturbation due to the sound wave: T(x, t) =T 0−
T1(x) cos(ωt) , - Adiabatic constant: γ, - Displacement amplitude: a, - Wave parameters: kandω.
We know that for adiabatic processes, pVγ=const , and for ideal gases, pV=nRT where nis the
number of moles and Ris the gas constant. From the ideal gas law, we have p=nRT
V. Let’s consider the
equilibrium state where p=p 0,V=V 0, andT=T 0. Assuming small temperature perturbation from the
equilibrium state: T=T 0+T1(x) cos(ωt) . Substitute into the ideal gas law:nR(T 0+T1(x) cos(ωt))
V0=p0.
Expanding this expression to first order in the perturbations, we get: T1(x) =−γT 0asin(kx) . Therefore,
the amplitudeT 1(x)of the temperature oscillations is−γT 0asin(kx)in terms ofT 0,γ,a,k, andx.
GPT-3.5’s response (Score: 6)In a thermoacoustic system where the gas oscillates due to a standing
sound wave, and there is weak thermal interaction between the gas and the tube, let us analyze the
temperature changes at specific points on the tube. At point A, located at the edge of the tube, the gas
parcel experiences maximum compression and minimum volume. As the gas undergoes an adiabatic
expansion, it cools down, leading to a decrease in the tube’s temperature at this point over time. At
point B, the central part of the tube, the gas parcel is at its equilibrium position, neither at maximum
compression nor expansion. Here, the temperature tends to remain constant as the gas parcel neither gains
nor loses significant heat to the tube. At point C, another edge of the tube, the gas parcel is at maximum
expansion and volume. Upon adiabatic compression, the gas heats up, resulting in an increase in the
tube’s temperature at this point. Therefore, in this scenario of weak thermal interaction, over a prolonged
period, the temperature at point A decreases, remains unchanged at point B, and increases at point C.
Example 1 retrieved by cosine similarityConsider a longitudinal wave traveling along the xdirection
in the rod. For a cross section at xwhen the rod is unstressed, let ξ(x, t) be its displacement at time
tand assume ξ(x, t) =ξ 0sink(x−ut) where ξ0andkare constants. Determine the corresponding
velocity v(x, t) , strain S(x, t) , and pressure p(x, t) as a function of xandt. Reference answer: The
velocity vis related to the displacement ξas in a simple harmonic motion (or a uniform circular mo-
tion, as shown in Figure A2) of angular frequency ω=ku . Therefore, if ξ(x, t) =ξ 0sink(x−ut) ,
thenv(x, t) =−kuξ 0cosk(x−ut). The strain and pressure are related to velocity as in Problem (a).
Hence, S(x, t) =−v(x, t)/u=kξ 0cosk(x−ut)p(x, t) =ρuv(x, t) =−kρu2ξ0cosk(x−ut) =
−Y S(x, t) =−kY ξ 0cosk(x−ut).
Example 2 retrieved by cosine similarityIn the uniform background magnetic field, theplasma density
is nonuniform in x. For simplicity, we assume that the temperature and the distribution of the ions and
electrons are the same. Thus, the plasma pressure can be expressed as p(x) =kT[n i(x) +n e(x)] =
2kTn(x) = 2kT(n 0+αx) , Where B, T, k, n 0, and αare positive constants, ni(x)andne(x)are the
number densities of the ions and electrons. The diamagnetic field is given by Bmx=µoM. Please explain
the generation mechanism of the electric current by a schematic drawing. Reference answer: Since the
magnetic field and the plasma are uniform z, the orbits of ions and electrons can project into in the x−y
plane. From the results of sub_question_number: 1, we know that an ion has a left-hand circular motion
and an electron has a right-hand circular motion. Due to the linear increase of the plasma density in x,
the number of ions with upward motion is less than that with downward motion at a given xposition,
which leads a net upward ion flow. Similarly, electrons have a net downward flow. Combining the ion and
election flows, we have a net upward electric current as illustrated below in schematic drawing.
GPT-3.5’s response using two-shot Retrieval-Augmented Generation (Score: 4)Given: - Tem-
perature perturbation due to the sound wave: T(x, t) =T 0−T 1(x) cos(ωt) , - Adiabatic constant: γ,
- Displacement amplitude: a, - Wave parameters: kandω. We know that for adiabatic processes,
pVγ=const , and for ideal gases, pV=nRT where nis the number of moles and Ris the gas constant.
From the ideal gas law, we have p=nRT
V. Let’s consider the equilibrium state where p=p 0,V=V 0, and

T=T 0. Assuming small temperature perturbation from the equilibrium state:T=T 0+T1(x) cos(ωt).
Substitute into the ideal gas law:nR(T 0+T1(x) cos(ωt))
V0=p 0. Expanding this expression to first order in
the perturbations, we get: T1(x) =−γT 0asin(kx) . Therefore, the amplitude T1(x)of the temperature
oscillations is−γT 0asin(kx)in terms ofT 0,γ,a,k, andx.
AnalysisThe original question was relatively short, but the retrieved questions are quite lengthy,
inevitably focusing more attention on the retrieved content. After the RAG process, it is evident that the
LLM listed conditions that did not exist in the original question.

G.3 Error Type 3: Wrong Retrieval.
Problemhis part deals with the usage of microwave radiation (radiated by the magnetron antenna into
the food chamber) for cooking, that is, heating up a lossy dielectric material such as water, either pure or
salty (which is our model for, say, soup). An electric dipole is a configuration of two equal and opposite
electric chargesqand−qa small distancedapart. The electric dipole vector points from the negative to
the positive charge, and its magnitude is p=qd . A time-dependent electric field ⃗E(t) =E(t)ˆx is applied
on a single dipole of moment ⃗ p(t) with constant magnitude p0=|⃗ p(t)| . The angle between the dipole and
the electric field is θ(t).Write expressions for both the magnitude of the torque τ(t) applied by the electric
field on the dipole and the power Hi(t)delivered by the field to the dipole, in terms of p0, E(t), θ(t) and
their derivatives.
Reference AnswerThe torque at time tis given by τ(t) =−qdsin[θ(t)]E(t) =
−p0sin[θ(t)]E(t) ,hence the instantaneous power delivered to the dipole by the electric field is Hi(t) =
τ(t)˙θ(t) =−p 0E(t) sinθ(t) ˙θ(t) =E(t)d
dt(p0cosθ(t)) =E(t)dpx(t)
dt.
GPT-3.5’s response (Score: 9)The torque τ(t) applied by the electric field on the dipole can be
expressed as the cross product of the electric dipole moment⃗ p(t)and the electric field ⃗E(t):
τ(t) =⃗ p(t)× ⃗E(t)
Expanding the cross product using vector algebra gives:
τ(t) =p 0ˆp(t)×E(t)ˆx=p 0E(t) sin(θ(t))ˆz
where ˆp(t)is the unit vector pointing along the direction of the dipole moment ⃗ p(t) andˆzis the unit
vector along the z-axis.The power P(t) delivered by the field to the dipole can be calculated using the dot
product of the torqueτ(t)and the angular velocityω(t)of the dipole:
P(t) =⃗ τ(t)·⃗ ω(t)
As the electric field ⃗E(t) is time-dependent and not directly causing the dipole to rotate, the power
delivered to the dipole is due to changes in the potential energy of the system, which can be related to the
time derivative of the dipole moment ⃗ p(t). Therefore, the power Hi(t)delivered by the field to the dipole
can be expressed as:
Hi(t) =d
dt(⃗ p(t)· ⃗E(t))
By substituting the given values, we obtain the expression for the power:
Hi(t) =d
dt(p0E(t) cos(θ(t)))
Example 1 retrieved by cosine similarityIn the figure below, a simplified scheme of a mass-
spectrometer is given. It is a device for measuring the masses of molecules. The substance under
investigation is ionised by heating up to a temperature Ton a hot filament (molecules undergo a single-
electron ionisation). The ions are accelerated using voltage U. At first, let us neglect the thermal energy
of the ions (eU≫kT , where eis elementary charge and k- the Boltzmann’s constant). A narrow beam
of accelerated ions enters a region with magnetic field. For the sake of simplicity, let us assume that the
region has a rectangular shape, and the magnetic field is homogeneous inside it. The magnetic fields
deflects the ions and depending on their mass, they may hit the detector. Let us assume that those ions
which hit the centre of the detector enter and exit the region with magnetic field perpendicularly with its
boundary, and the distance between the entry and exit points is l (see figure). 1) Express the mass M
of those ions which hit the centre of the detector via the quantities B l, U ande.Reference answer:The
trajectory of a charged particle in the magnetic field is circle of radius R=l/√
2. Lorenz force is
responsible for the acceleration, Bev =Mv2/Rhence BeR=p . Substituting p2= 2MUe=B2e2R2,
we obtainM=B2l2e/4U.

Example 2 retrieved by cosine similarityOne way of measuring the atmospheric electric field, and
hence σ0, is with the system shown in the diagram. A pair of metal quadrants, insulated from ground
but connected to each other, are mounted just underneath a grounded uniformly rotating disk with two
quadrant-shaped holes cut in it. (In the diagram, the spacing has been exaggerated in order to show the
arrangement.) Twice in each revolution the insulated quadrants are completely exposed to the field, and
then (1/4 of a period later) are completely shielded from it. Let Tbe the period of revolution, and let
the inner and outer radii of the insulated quadrants be r1andr2as shown. Take t= 0 to be an instant
when the insulated quadrants are completely shielded. Obtain expressions that give the total charge q(t)
induced on the upper surface of the insulated quadrants as a function of time between t= 0 andt=T/2 ,
and sketch a graph of this variation. [The effects of the atmospheric ion current can be ignored in this
situation.]
GPT-3.5’s response using two-shot Retrieval-Augmented Generation (Score: 4)The torque applied
by an electric field on a dipole is given by τ(t) =⃗ p(t)× ⃗E(t) . Since ⃗ p(t) =p 0cos(θ(t))ˆy and
⃗E(t) =E(t)ˆx, the torque would be:
τ(t) =p 0E(t) cos(θ(t))ˆz
The power Pdelivered by the electric field to the dipole is given bydW
dt=⃗F·⃗ v , where ⃗Fis the force on
the dipole and⃗ vis its velocity. Since power is force times velocity, we get:
P= ⃗F·⃗ v=τ·⃗ ω
Where ⃗ ωis the angular velocity of the dipole. Substituting τ(t) =p 0E(t) cos(θ(t))ˆz and⃗ ω(t) =− ˙θ(t)ˆz ,
we find:
Hi(t) =τ(t)·⃗ ω(t) =−p 0E(t) ˙θ(t) cos(θ(t))
.
AnalysisAlthough the original topic is about wave radiation, and the searched topics are also related to
waves, the specific content and background of the topics are different: Example 1 describes a simplified
model of a mass spectrometer. Example 2 discusses a method for measuring the electric field of the
atmosphere.
H Additional Results
H.1 Experimental results of problems with different difficulty levels.
We provide the number of questions of different difficulty levels, as shown in Table 9. The number of
questions whose answer scores are improved from zero to non-zero after RAG are shown in Table 11.
Average score of LLMs with RAG over different difficulty level problems are provided in Table 10.
Category Easy Medium Hard Superhard
Count 51 74 148 117
Table 9: Distribution of Question Difficulty.

Model Method Easy Medium Hard Superhard
GPT-3.5 Base 9.41 5.89 3.76 1.15
GPT-3.5 BM25 6.51 4.74 3.55 2.45
GPT-3.5 Dragon+ 6.63 4.08 3.68 2.38
GPT-3.5 Contriver 6.35 4.85 3.72 2.48
GPT-3.5 Cos 6.57 5.04 3.75 2.62
GPT-4 Base 8.45 7.15 6.14 4.91
GPT-4 BM25 8.51 6.81 5.78 4.42
GPT-4 Dragon+ 8.53 7.07 5.60 4.38
GPT-4 Contriver 8.94 6.70 5.29 4.60
GPT-4 Cos 8.45 6.45 5.41 4.17
Table 10: Average score of GPT-3.5 and GPT4 with RAG in different difficulty level problems.
Model BM25 Cos Dragon+ Contriver Average
GPT-4 6 7 5 9 6.75
GPT-3.5 18 18 24 19 19.75
Gemini 26 22 23 22 23.25
DeepSeek-Math 10 11 12 12 11.25
Table 11: Number of questions whose answer scores are raised from zero to non-zero after RAG.