# MCTS-RAG: Enhancing Retrieval-Augmented Generation with Monte Carlo Tree Search

**Authors**: Yunhai Hu, Yilun Zhao, Chen Zhao, Arman Cohan

**Published**: 2025-03-26 17:46:08

**PDF URL**: [http://arxiv.org/pdf/2503.20757v1](http://arxiv.org/pdf/2503.20757v1)

## Abstract
We introduce MCTS-RAG, a novel approach that enhances the reasoning
capabilities of small language models on knowledge-intensive tasks by
leveraging retrieval-augmented generation (RAG) to provide relevant context and
Monte Carlo Tree Search (MCTS) to refine reasoning paths. MCTS-RAG dynamically
integrates retrieval and reasoning through an iterative decision-making
process. Unlike standard RAG methods, which typically retrieve information
independently from reasoning and thus integrate knowledge suboptimally, or
conventional MCTS reasoning, which depends solely on internal model knowledge
without external facts, MCTS-RAG combines structured reasoning with adaptive
retrieval. This integrated approach enhances decision-making, reduces
hallucinations, and ensures improved factual accuracy and response consistency.
The experimental results on multiple reasoning and knowledge-intensive datasets
datasets (i.e., ComplexWebQA, GPQA, and FoolMeTwice) show that our method
enables small-scale LMs to achieve performance comparable to frontier LLMs like
GPT-4o by effectively scaling inference-time compute, setting a new standard
for reasoning in small-scale models.

## Full Text


<!-- PDF content starts -->

MCTS-RAG: Enhance Retrieval-Augmented Generation with
Monte Carlo Tree Search
Yunhai HuN∗Yilun ZhaoY∗Chen ZhaoNArman CohanY
YYale UniversityNNew York University
https://github.com/yale-nlp/MCTS-RAG
Abstract
We introduce MCTS-RAG, a novel approach
that enhances the reasoning capabilities of
small language models on knowledge-intensive
tasks by leveraging retrieval-augmented gen-
eration (RAG) to provide relevant context and
Monte Carlo Tree Search (MCTS) to refine
reasoning paths. MCTS-RAG dynamically in-
tegrates retrieval and reasoning through an iter-
ative decision-making process. Unlike standard
RAG methods, which typically retrieve infor-
mation independently from reasoning and thus
integrate knowledge suboptimally, or conven-
tional MCTS reasoning, which depends solely
on internal model knowledge without exter-
nal facts, MCTS-RAG combines structured
reasoning with adaptive retrieval. This inte-
grated approach enhances decision-making, re-
duces hallucinations, and ensures improved fac-
tual accuracy and response consistency. The
experimental results on multiple reasoning
and knowledge-intensive datasets datasets ( i.e.,
ComplexWebQA, GPQA, and FoolMeTwice)
show that our method enables small-scale LMs
to achieve performance comparable to fron-
tier LLMs like GPT-4o by effectively scaling
inference-time compute, setting a new standard
for reasoning in small-scale models.
1 Introduction
Recent advancements in MCTS-based reason-
ing have demonstrated remarkable improvements
in structured decision-making and logical infer-
ence (Kocsis and Szepesvári, 2006; Browne et al.,
2012; Xie et al., 2024a). The rStar framework (Qi
et al., 2024), for instance, has shown that system-
atic search and exploration can significantly en-
hance reasoning performance, enabling small-scale
LMs ( i.e.,models with up to 7B parameters) to
compete with much larger models. However, a
key limitation of these approaches is their heavy
∗Equal contributions. Correspondence: Yilun Zhao
(yilun.zhao@yale.edu )reliance on internal knowledge, which hinders their
effectiveness in knowledge-intensive tasks.
On the other hand, RAG has been widely used
to solve knowledge-intensive tasks (Lewis et al.,
2020; Karpukhin et al., 2020; Izacard and Grave,
2021), but its effectiveness with small-scale LMs
remains limited. small-scale LMs struggle with
query formulation and retrieved content compre-
hension, often generating vague queries and misin-
terpreting key details (Fan et al., 2025). Moreover,
existing RAG systems do not dynamically adjust
their retrieval strategies based on changing infor-
mational or reasoning requirements, which results
in unnecessary or repetitive retrieval steps (Li et al.,
2024; Gao et al., 2024). For example, when an-
swering a multi-hop question like “Which novel in-
spired the movie that won Best Picture in 1994?”, a
standard retrieval system might retrieve documents
about Forrest Gump ( i.e.,Best Picture winner in
1994), but fail to recognize the need for additional
reasoning or retrieval steps to establish the connec-
tion between Forrest Gump and the novel written
by Winston Groom. This limitation arises because
small-scale language models often lack the ability
to refine queries iteratively and integrate retrieved
information into a coherent reasoning process.
To address the aforementioned limitations, we
propose MCTS-RAG, a novel framework that inte-
grates MCTS’s reasoning and search capabilities
with adaptive retrieval mechanisms. At a high level,
MCTS-RAG operates by iteratively refining both
retrieval and reasoning through a search-based pro-
cess. Given a query, it explores multiple reasoning
paths, dynamically incorporating retrieval actions
at key decision points. Retrieved knowledge is then
used to evaluate intermediate states, and beneficial
retrieval pathways are reinforced through backprop-
agation. This structured search mechanism ensures
that the model efficiently acquires and utilizes rele-
vant information for more accurate reasoning. In
contrast, by integrating retrieval with search-basedarXiv:2503.20757v1  [cs.CL]  26 Mar 2025

reasoning, MCTS-RAG is able to systematically
explore relevant knowledge and reason over it to
obtain the correct answer.
MCTS-RAG has the following key features: Im-
proved reasoning accuracy: New retrieval ac-
tions enable SLMs to acquire external knowledge
and enhance the quality of question answering
(§3.2). Optimized query formulation: The re-
finement process ensures that each query focuses
on specific information needs, improving the ef-
fectiveness of retrieval query generation (§3.3).
Enhanced retrieval quality: Reflecting on and
summarizing retrieved information helps reduce
semantic discrepancies and ensures alignment with
the core problem (§3.3). MCTS-RAG demon-
strates superior performance on various knowledge-
intensive benchmarks, including ComplexWebQA
(CMQA) (Talmor and Berant, 2018), GPQA (Rein
et al., 2024), and FoolMeTwice (FMT) (Eisensch-
los et al., 2021a). Specifically, it achieves over 20%
improvement with Llama 3.1-8B and 6% with
Qwen2.5-7B on CWQA, roughly 15% and 10%
gains on GPQA, and over 10% (Llama) and 4%
(Qwen) on FMT, while outperforming other base-
lines like Standard RAG, ReAct (Yao et al., 2023b),
Self-Ask (Press et al., 2023), Search-O1 (Li et al.,
2025), and rStar (Qi et al., 2024) by effectively
retrieving and integrating evidence through refined
multi-step reasoning that minimizes hallucinations.
2 Related Work
Inference-time Scaling. Inference-time scaling
enhances reasoning without modifying model pa-
rameters by optimizing computational allocation
during generation. A core approach involves rea-
soning diversification and selection: generating
multiple candidates (Wang et al., 2023) and choos-
ing optimal outputs via voting (Liang et al., 2024)
or verifier-guided ranking (Cobbe et al., 2021).
Structured search algorithms, such as beam search
(Xie et al., 2024b) and tree-of-thought frameworks
(Yao et al., 2023a), explicitly model reasoning
paths. Recently, Monte Carlo Tree Search (MCTS)
has been applied to balance exploration and ex-
ploitation in reasoning tasks, iteratively refining
solutions through selection, expansion, simulation,
and backpropagation (Hao et al., 2023). Further,
integrating MCTS with LLMs using value func-
tions (Zhang et al., 2024) or predefined reasoning
heuristics (Qi et al., 2024) has improved efficiency
in mathematical reasoning and code generation.Retrieval-Augmented Generation. The RAG
system enhances LLMs in knowledge-intensive
tasks by incorporating external information. Query
optimization techniques, including expansion and
transformation, improve retrieval quality (Ma et al.;
Jagerman et al., 2023). Iterative retrieval methods,
such as IRCoT (Trivedi et al., 2023) and ITER-
RETGEN (Shao et al.), refine retrieval and gen-
eration. LLM-driven retrieval strategies, such as
WebGPT (Nakano et al., 2021) and Toolformer
(Schick et al., 2023), have demonstrated notable
improvements in efficiency by leveraging large lan-
guage models to interact with external tools or
search engines, thus streamlining the process of
gathering relevant data. Meanwhile, self-reflection
mechanisms in systems like Self-RAG (Asai et al.;
Islam et al., 2024) and Auto-RAG (Yu et al., 2024)
further enhance retrieval relevance by employing
iterative introspection to refine intermediate out-
puts. To address this, reasoning-intensive retrieval
methods have emerged. For example, BRIGHT
(Su et al., 2024) introduces complex, reasoning-
driven queries that challenge traditional retrieval
approaches, while Rank1 (Weller et al., 2025) lever-
ages advanced inference-time reranking to identify
nuanced relationships missed by standard meth-
ods. Despite these advancements, however, these
methods often overlook alternative solutions due
to their linear reasoning approach and the limited
capabilities of small-scale LMs.
3 MCTS-RAG
3.1 Preliminaries
rStar (Qi et al., 2024) is a recently proposed self-
consistency framework designed to enhance the
reasoning capabilities of language models with-
out requiring additional fine-tuning or reliance on
stronger teacher models. rStar achieves this by
breaking down the reasoning process into two dis-
tinct yet interconnected phases: generation anddis-
crimination . In the Generation Phase , the model
proactively explores multiple reasoning trajectories
through human-like reasoning actions, including
step-by-step inference and question decomposition.
Subsequently, the Discrimination Phase evaluates
these candidate reasoning paths, selecting and re-
fining them to identify the most logically consistent
and accurate responses.
However, the original rStar framework is lim-
ited by its inability to dynamically acquire ex-
ternal knowledge, restricting its performance in

Question:  A textile dye containing an extensively conjugated pi-electrons emits light
with energy of 2.3393 eV . What color of light is absorbed by the organic compound?
Step 1:  To determine the color of light absorbed
by the organic compound, we need to
understand the relationship between the energy
of emitted light and the color of absorbed light.Question 1.1:  What is the relationship between the energy of emitted
light and the energy of absorbed light?  
The answer is:  The relationship between the energy of emitted light
and the energy of absorbed light is that they are complementary
Now that we have the [...]. 
 
The asnwer is: Red. Step 3:  Given that the energy of emitted light is 2.339
eV, we can use the relationship between the energy
of absorbed light and the energy of emitted light to
ﬁnd the energy of absorbed light.
The government type of  [...].   
The answer is:  RedQuestion 1.2:  What is the energy of
the light emitted by the textile dye?
The answer is:  The emitted light is
in the green  region of the spectrumThe retrieval process is shown in Figure 2
Answer from  Trace 1
Blue Answer from  Trace 2
BlueAnswer from  Trace 4
Red Answer from  Trace 5
Red Answer from  Trace 6
Red Answer from  Trace 6
Violet 
Answer 1
Blue
Conf: 0.25Answer 3
Red 
Conf: 0.125Answer 4
Red
Conf: 0.625  ✅
Final Selected AnswerThe answer is: 
BlueStep 1:  let's ﬁrst understand the relationship
between the color of light emitted and the
color of light absorbed by a molecule.The answer is:   
VioletStep2:  The energy of absorbed
light is complementary to the
energy of emitted light.
③ ④⑥
The answer is: 
BlueStep 2:  Now that we have established the
relationship between the energy of emitted
light and the energy of absorbed light, we
can use the concept of complementary
colors to determine the color of light
absorbed by the organic compound.
The government type of  [...].   
The answer is:  Red①
②
⑤A1
Quick
ReasoningA2
Retrieval
ReasoningA3
Decompose  
QuestionA4
Retrieval
DecomposeA5
Summary
AnswerA6Direct
AnswerFigure 1: An illustration of MCTS-RAG workflow for answering the question sampled from ComplexWebQA.
knowledge-intensive queries. To address the in-
herent limitations of rStar, we propose an inte-
grated reasoning framework that combines the it-
erative reasoning capabilities of rStar with RAG.
At a high level, our approach builds on the iter-
ative generative-discriminative structure of rStar
and introduces additional operations specifically
designed to facilitate dynamic external knowledge
retrieval. This enables the language model to seam-
lessly integrate relevant external information into
its reasoning process, significantly improving fac-
tual accuracy and decision robustness. The follow-
ing subsections detail the proposed MCTS-RAG
framework.
3.2 Action Space Definition
We design a set of discrete actions at each MCTS
decision point: A1–A3 from rStar (Qi et al., 2024),
along with two new RAG-related actions A4 and
A5 and a summary action A6, enabling dynamic
knowledge acquisition and enhanced reasoning syn-
ergy for improved decision-making.
A1: Direct Answer: Provide an immediate re-
sponse based on existing reasoning or pre-
viously known context, suitable for straight-
forward queries or when additional analysis
is unnecessary.A2: Quick Reasoning: Execute rapid, incremen-
tal reasoning steps based on the current con-
text, ideal for exploratory paths or preliminary
judgments to efficiently guide the search.
A3: Decompose Question: Break complex
queries into smaller, manageable sub-
questions, allowing for clearer problem-
solving pathways and improved reasoning
efficiency, particularly beneficial for multi-
part or intricate problems.
A4: Retrieval Reasoning: Actively retrieve rel-
evant knowledge from internal or external
sources before proceeding with the next rea-
soning step, critical for queries requiring sup-
plementary information or when existing con-
text is incomplete.
A5: Retrieval Decompose: Integrate both decom-
position and retrieval, first breaking down
complex questions and then acquiring rel-
evant knowledge to solve individual sub-
problems. This action is highly effective for
queries involving detailed context-dependent
sub-questions.
A6: Summarized Answer: Generate a concise,
structured summary that synthesizes results

from previous reasoning and retrieved infor-
mation, providing coherent and comprehen-
sive responses especially useful for queries
that demand summarization or integration of
multifaceted information.
Each action is designed to address specific as-
pects of the reasoning-retrieval interplay, ensuring
that the model can adapt its strategy dynamically as
it navigates through the problem space. To further
enhance exploration, we employ Upper Confidence
Bound for Trees (UCT) (Kocsis and Szepesvári,
2006) in our MCTS framework—a crucial method
that balances exploitation and exploration. The
UCT formula is:
UCT(s, a) =¯Q(s, a) +C·s
lnN(s)
N(s, a),
where ¯Q(s, a) =Q(s,a)
N(s,a)is the average reward for
action ain state s, with Q(s, a)as the cumulative
reward and N(s, a)as the visit count. N(s)is the
total number of visits to state s.Cis the explo-
ration constant, controlling the balance between
exploitation and exploration.
Within MCTS-RAG, search depth limits how
many levels are expanded from the root node to
control the search range, while the number of roll-
outs indicates how many times the simulation is
run from a selected node until termination or a
preset limit to estimate its value. By running simu-
lations within a controlled depth and updating node
statistics via UCT, MCTS effectively balances ex-
ploration and exploitation with finite computational
resources, continuously refining its search strategy.
3.3 Retrieval Process
Our approach dynamically retrieves information
within an evolving MCTS reasoning environment,
enabling timely and relevant integration of exter-
nal knowledge. The model autonomously deter-
mines when retrieval is required, generates targeted
queries, and critically integrates external knowl-
edge to improve reasoning accuracy. By interweav-
ing retrieval with reasoning, we streamline infor-
mation flow and produce concise yet informative
outputs. If previously retrieved data adequately an-
swers the current reasoning step—determined by
checking whether the information satisfies prede-
fined accuracy thresholds or resolves open reason-
ing paths—the model foregoes additional retrieval,
thus avoiding redundancy.
Retrieval Query
Generation
"Le Mali"  is the
national anthem of
Mali.Mali has "Le Mali"
as its national
anthem. It was
adopted in 1960.As of February
2025, General
Assimi Goïta  serves
as the interim
President of Mali.  
 We identiﬁed the country that has  Le
Mali as its national anthem.  According
to the given context , Mali is the country
with Le Mali  as its national anthem.✅ Useful ❌ Not Useful ✅ UsefulWhat is the government type where \"Le Mali\" is the
national anthem? Question 1.1: What is the name of the
country where "Le Mali" is the national anthem?Query
GenerationR1
R2
Retrieval
Execution
Knowledge
ReﬂectionR3
R4
Summary   
ReasoningIs "Le Mali" the
national anthem of
any country?Who is the leader
of Mali in 2025?Which country has
"Le Mali" as its
national anthem?Figure 2: An illustration of MCTS-RAG retrieval pro-
cess ( i.e.,R1-R4) within one step of the retrieval decom-
position action highlighted in Figure 2.
R1: Query Generation: If a knowledge gap is
detected, the model generates search queries.
R2: Query Execution: External retrieval tools are
used to obtain the most relevant information.
R3: Knowledge Reflection: Retrieved data is
evaluated for relevance and consistency to de-
termine its inclusion in the reasoning process.
R4: Summary Answer: Refined information is
integrated, enabling the model to answer sub-
questions or advance reasoning.
This interleaved retrieval process ensures that
the model’s reasoning is continuously updated and
validated against external data, thereby reducing
errors and enhancing the robustness of final output.
3.4 Determing Final Answer
At the conclusion of the MCTS exploration (il-
lustrated in the bottom part of Figure 2), the
best answer is selected through a voting mech-
anism and consistency analysis over candidate
solutions. Specifically, each reasoning trajec-
tory obtained from the MCTS yields a candi-
date answer cj, resulting in a candidate answer
setC={c1, c2, . . . , c M}. These candidate an-
swers are grouped into a set of unique answers
A={a1, a2, . . . , a N}based on semantic consis-
tency. The final score for each unique answer ak
is computed as the sum of the rewards of all candi-
dates grouped under ak, where the reward of each

candidate cjis the product of rewards for all nodes
along its corresponding reasoning trajectory.
Score( ak) =P
cj∈C(ak)Reward( cj)
P
cj∈CReward( cj)(1)
The best answer is then determined as
a∗= arg max
ak∈AScore( ak), (2)
ensuring that the most frequent and consistent rea-
soning trajectory is chosen.
4 Experiments
4.1 Experimental Setup
We evaluate Qwen2.5-7B and Llama 3.1-8B on
three complex reasoning tasks: ComplexWe-
bQA (CWQA) (Talmor and Berant, 2018),
which requires multi-step reasoning over web-
based queries; Graduate-Level Google-Proof QA
(GPQA) (Rein et al., 2023), which tests knowledge-
intensive science question answering; and Fool-
MeTwice (FMT) (Eisenschlos et al., 2021b), a chal-
lenging fact-checking benchmark that assesses the
model’s ability to verify factual claims.
Baselines. We evaluate the following baseline
methods for comparison: Chain-of-Thought
(CoT) (Wei et al., 2022) prompting encourages the
model to generate explicit step-by-step reasoning
to improve complex problem-solving. Standard
RAG (Ma et al., 2023) performs a single-pass re-
trieval to augment the model’s responses but lacks
iterative refinement. ReAct (Yao et al., 2023b) al-
ternates between reasoning and retrieval, allowing
the model to dynamically refine its understanding
based on external evidence. Self-Ask with Search
(Self-Ask) (Press et al., 2023) with Search decom-
poses complex queries into subquestions, retrieves
relevant external information, and synthesizes the
answers to enhance multi-step reasoning. Search-
O1(Li et al., 2025) executes a single retrieval step
before generating an answer, limiting its ability to
iteratively verify information. Finally, the original
rStar (Qi et al., 2024) algorithm employs a struc-
tured iterative reasoning process but does not fully
leverage dynamic retrieval or decomposition.
Implementation Details. We compare MCTS-
RAG with other baselines using the same lan-
guage models, Qwen2.5-7B and LLaMA 3.1-8B,
to ensure a fair comparison of reasoning capabili-
ties. To maintain consistency across methods, we
use the same retrieval corpus and retriever, thusavoiding discrepancies caused by varying infor-
mation access. Specifically, we rely on the Bing
Search Engine and LangChain for retrieval, as Bing
provides extensive and up-to-date web informa-
tion while LangChain supports robust retrieval-
augmented generation workflows. For different
datasets, CWQA is obtained from its web snippets,
GPQA uses a corpus sourced from Wikipedia and
Bing, and FMT draws on its own related documents.
This setup ensures that variations in performance
stem from differences in reasoning mechanisms.
To facilitate structured reasoning, we configure our
setup with a rollout of 4, allowing multiple steps
of reasoning expansion. Each query can be decom-
posed into at most two subquestions, ensuring a
controlled breakdown of complex queries. We set
the maximum reasoning depth to 5, enabling deep
but efficient multi-hop reasoning.
4.2 Main Findings
Table 1 compares reasoning methods on CWQA,
GPQA, and FMT for Llama 3.1-8B and Qwen2.5-
7B. Our approach consistently outperforms base-
lines, demonstrating strong multi-step reasoning
and retrieval capabilities. On CWQA, it achieves
over a 20% gain with Llama 3.1-8B and around 6%
with Qwen2.5-7B. Similarly, it surpasses competi-
tors on GPQA by roughly 15% and 10%, respec-
tively, benefiting from refined verification strate-
gies. On FMT, it leads by over 10% with Llama
3.1-8B and 4% with Qwen2.5-7B, proving its re-
silience against misleading distractors. These re-
sults highlight our method’s superior generaliza-
tion and efficiency, especially in fact-checking and
science-related tasks. Compared to baselines like
Standard RAG, ReAct, Self-Ask, and Search-O1,
Our structured multi-step reasoning can retrieve
and process evidence more accurately, and on av-
erage we improve the performance by about 14%
over the baseline under three datasets. Unlike rStar,
it enables broader retrieval, extracting critical in-
sights while minimizing hallucinations, achieving
an average improvement of 17%. This framework
sets a new benchmark for complex reasoning, de-
livering high accuracy and efficiency in diverse
problem-solving scenarios.
4.3 Fine-grained Analysis
We evaluate the effectiveness of retrieval actions
and rollout times in Table 2. Specifically, we con-
duct an ablation by disabling different retrieval
modules (A4, A5, or both) to gauge their impact

MethodsQwen2.5-7B Llama 3.1-8B
CWQA GPQA FMT CWQA GPQA FMT
CoT 34.65 35.00 57.25 27.72 28.71 56.50
GPT-4o 54.45 52.98 55.44 54.45 52.98 55.44
Qwen2.5-72B 44.55 40.59 58.41 44.55 40.59 58.41
rStar 55.45 32.32 55.94 37.62 28.71 56.42
Standard RAG 44.21 40.59 58.41 35.64 31.68 51.48
GPT-4o 59.40 54.90 61.38 59.40 54.90 61.38
Qwen2.5-72B 48.51 43.13 59.40 48.51 43.13 59.40
ReAct 45.54 41.58 62.37 47.52 34.31 55.44
Self-Ask 44.55 42.57 60.91 44.55 57.84 58.41
Search-O1 49.50 54.45 64.35 44.55 58.82 62.87
MCTS-RAG 61.38 64.64 68.28 67.32 74.25 74.25
Table 1: Answer accuracy of MCTS-RAG and other meth-
ods (both with and without retrieval modules).Settings CWQA GPQA FMT
Analysis of Retrieval Modules
Disable A4&A5 55.45 32.32 50.40
Disable A4 55.70 36.27 55.94
Disable A5 56.20 44.11 62.37
Enable All 56.70 53.20 66.50
Analysis of Rollout Numbers
4 rollout (main setting) 61.38 64.64 68.28
8 rollout 64.35 63.72 68.12
12 rollout 68.65 75.15 69.35
16 rollout 71.20 84.31 74.14
Table 2: Answer accuracy of Qwen2.5-7B-Instruct
under different retrieval and rollout settings.
on overall performance. In addition, we vary the
number of rollouts from 4 to 16 to investigate how
deeper search affects accuracy and efficiency.
Impact of Different Actions. Retrieval actions,
especially A4 and A5, are key for multi-step
reasoning. Enabling all retrievals boosts GPQA
(+20.88%) and FMT (+16.10%). Disabling A5
improves GPQA (+7.84%) and FMT (+6.43%)
over disabling A4, suggesting A4’s stronger role.
CWQA sees minimal impact (+1.25%). These find-
ings highlight retrieval trade-offs and the impor-
tance of recursive evidence aggregation.
Impact of Different Rollout Strategies. More
rollouts enhance performance, particularly for
GPQA. Increasing from 4 to 8 slightly aids CWQA
(+3%), while 8 to 12 boosts GPQA (+11%). Scal-
ing to 16 further improves GPQA (+9%) and FMT
(+5%), reinforcing the value of iterative reasoning.
4.4 Human Analysis and Case Study
To better understand the strengths and limitations of
MCTS-RAG, we conduct a comprehensive analysis
of its successful cases in comparison to baseline
methods, along with a thorough error analysis.
Successful Case Analysis. Our case study re-
veals the following two key improvements intro-
duced by MCTS-RAG: (1) Enhanced External
Knowledge Utilization : Compared to other rea-
soning methods, MCTS-RAG achieves higher ac-
curacy, primarily due to its richer reasoning space
and more effective utilization of external knowl-
edge. Figure 6 clearly illustrates how Monte Carlo
Tree Search tightly integrates reasoning and re-
trieval processes, significantly enhancing the qual-ity and richness of information used during reason-
ing, thereby substantially improving inference ac-
curacy. (2) Reduced Hallucination Risks : More-
over, MCTS-RAG mitigates hallucination risks
through detailed and explicit reasoning steps. On
one hand, the explicit reasoning pathways enable
the model to more accurately interpret retrieved
external knowledge, reducing errors arising from
ambiguity or misunderstanding (as illustrated in
Figure 7 in Appendix). On the other hand, these
thorough reasoning procedures generate clearer and
more contextually relevant queries, thus improving
the precision of external information retrieval (as
illustrated in Figure 8 in Appendix). Consequently,
MCTS-RAG demonstrates substantial advantages
over traditional reasoning methods in terms of im-
proved accuracy and robustness.
Error Case Analysis Our human analysis iden-
tifies the following three primary error types in
MCTS-RAG: (1) Amplification Error : As illus-
trated in Figure 3, early retrieval errors in MCTS-
RAG can be magnified, causing incorrect infor-
mation to dominate subsequent reasoning and ul-
timately leading to a incorrect final answer. (2)
Factual Confusion : We reveal that semantic mis-
matches between retrieved text and the reasoning
process can lead to conflations or hallucinations.
Figure 4 presents details on how semantically di-
vergent retrieval results can lead to incorrect final
answers. (3) Information Overload : Excessive
additional information in MCTS-RAG can cause
certain reasoning paths to deviate from the original
question, leading to incorrect conclusions. Fig-
ure 5 presents a detailed example of some reason-
ing paths that prioritize irrelevant aspects.

5 Conclusion
In this work, we propose MCTS-RAG, an ap-
proach that integrates Monte Carlo Tree Search
with Retrieval-Augmented Generation to improve
multi-step reasoning accuracy and reliability. By
effectively allocating search and retrieval processes,
MCTS-RAG excels at handling cross-domain tasks
that require in-depth external knowledge. It not
only enables flexible formulation of high-quality
retrieval queries but also refines the reasoning path
through iterative tree exploration, thus reducing hal-
lucinations caused by shallow retrieval or simplistic
reasoning. Experimental results show that MCTS-
RAG has achieved good results in scenarios such as
complex reasoning tasks, knowledge-enhanced sci-
entific question-answering tasks, and challenging
fact-checking tasks.
Acknowledgments
We are grateful to Google TRC program for pro-
viding computing resources and Together AI for
granting LLM API credits.
Limitations and Future Work
MCTS-RAG integrates MCTS-based reasoning
and RAG to enhance reasoning capabilities, but
several errors persist. Amplification errors occur
when early retrieval mistakes propagate through
search iterations. Factual confusion arises from
semantic mismatches leading to incorrect reason-
ing. Information overload happens when excessive
retrieval results cause reasoning to deviate from
the target. Additionally, search latency remains a
challenge, as deep MCTS search trees significantly
increase reasoning time, particularly with multiple
retrieval steps. Action selection complexity arises
because the optimal choice among A1-A6 depends
on query difficulty, necessitating a more adaptive
decision mechanism. Inefficient expansion occurs
when MCTS explores unnecessary branches due
to a lack of effective pruning based on retrieval
confidence or early error detection. Addressing
these issues is essential for improving efficiency
and reasoning accuracy.
We encourage future work to focus on optimiz-
ing search efficiency by developing adaptive action
selection strategies, confidence-based retrieval fil-
tering, and error-aware pruning mechanisms to im-
prove MCTS exploration. Additionally, integrating
reinforcement learning for dynamic search policy
refinement may further enhance reasoning accuracy.Addressing these challenges will contribute to the
development of more robust and scalable reasoning
models, bridging the gap between retrieval-based
methods and human-like problem-solving.
References
Akari Asai, Zeqiu Wu, Yizhong Wang, Avirup Sil, and
Hannaneh Hajishirzi. Self-rag: Learning to retrieve,
generate, and critique through self-reflection. In The
Twelfth International Conference on Learning Repre-
sentations .
Cameron B Browne, Edward Powley, Daniel White-
house, Simon M Lucas, Peter I Cowling, Philipp
Rohlfshagen, Stephen Tavener, Diego Perez, Spyri-
don Samothrakis, and Simon Colton. 2012. A survey
of monte carlo tree search methods. IEEE Transac-
tions on Computational Intelligence and AI in games ,
4(1):1–43.
Karl Cobbe, Vineet Kosaraju, Mohammad Bavarian,
Jacob Hilton, Reiichiro Nakano, Christopher Hesse,
and John Schulman. 2021. Training verifiers to solve
math word problems.
Julian Eisenschlos, Bhuwan Dhingra, Jannis Bulian,
Benjamin Börschinger, and Jordan Boyd-Graber.
2021a. Fool me twice: Entailment from Wikipedia
gamification. In Proceedings of the 2021 Conference
of the North American Chapter of the Association
for Computational Linguistics: Human Language
Technologies , pages 352–365, Online. Association
for Computational Linguistics.
Julian Eisenschlos, Bhuwan Dhingra, Jannis Bulian,
Benjamin Börschinger, and Jordan Boyd-Graber.
2021b. Fool me twice: Entailment from wikipedia
gamification. In Proceedings of the 2021 Conference
of the North American Chapter of the Association
for Computational Linguistics: Human Language
Technologies .
Tianyu Fan, Jingyuan Wang, Xubin Ren, and Chao
Huang. 2025. Minirag: Towards extremely simple
retrieval-augmented generation.
Yunfan Gao, Yun Xiong, Xinyu Gao, Kangxiang Jia,
Jinliu Pan, Yuxi Bi, Yi Dai, Jiawei Sun, Meng Wang,
and Haofen Wang. 2024. Retrieval-augmented gen-
eration for large language models: A survey.
Shibo Hao, Yi Gu, Haodi Ma, Joshua Hong, Zhen
Wang, Daisy Wang, and Zhiting Hu. 2023. Rea-
soning with language model is planning with world
model. In Proceedings of the 2023 Conference on
Empirical Methods in Natural Language Processing ,
pages 8154–8173, Singapore. Association for Com-
putational Linguistics.
Shayekh Islam, Md Asib Rahman, KSM Tozammel Hos-
sain, Enamul Hoque, Shafiq Joty, and Md Rizwan
Parvez. 2024. Open-rag: Enhanced retrieval aug-
mented reasoning with open-source large language

models. In Findings of the Association for Compu-
tational Linguistics: EMNLP 2024 , pages 14231–
14244.
Gautier Izacard and Edouard Grave. 2021. Leveraging
passage retrieval with generative models for open do-
main question answering. In Proceedings of the 16th
Conference of the European Chapter of the Associ-
ation for Computational Linguistics: Main Volume ,
pages 874–880, Online. Association for Computa-
tional Linguistics.
Rolf Jagerman, Honglei Zhuang, Zhen Qin, Xuanhui
Wang, and Michael Bendersky. 2023. Query expan-
sion by prompting large language models. arXiv
preprint arXiv:2305.03653 .
Vladimir Karpukhin, Barlas Oguz, Sewon Min, Patrick
Lewis, Ledell Wu, Sergey Edunov, Danqi Chen, and
Wen-tau Yih. 2020. Dense passage retrieval for open-
domain question answering. In Proceedings of the
2020 Conference on Empirical Methods in Natural
Language Processing (EMNLP) , pages 6769–6781,
Online. Association for Computational Linguistics.
Levente Kocsis and Csaba Szepesvári. 2006. Bandit
based monte-carlo planning. In European conference
on machine learning , pages 282–293. Springer.
Patrick Lewis, Ethan Perez, Aleksandra Piktus, Fabio
Petroni, Vladimir Karpukhin, Naman Goyal, Hein-
rich Küttler, Mike Lewis, Wen-tau Yih, Tim Rock-
täschel, et al. 2020. Retrieval-augmented generation
for knowledge-intensive nlp tasks. Advances in Neu-
ral Information Processing Systems , 33:9459–9474.
Jiatao Li, Xinyu Hu, and Xiaojun Wan. 2024. Smart-
rag: Selection using determinantal matrices for aug-
mented retrieval.
Xiaoxi Li, Guanting Dong, Jiajie Jin, Yuyao Zhang,
Yujia Zhou, Yutao Zhu, Peitian Zhang, and Zhicheng
Dou. 2025. Search-o1: Agentic search-enhanced
large reasoning models. CoRR , abs/2501.05366.
Tian Liang, Zhiwei He, Wenxiang Jiao, Xing Wang,
Yan Wang, Rui Wang, Yujiu Yang, Shuming Shi, and
Zhaopeng Tu. 2024. Encouraging divergent thinking
in large language models through multi-agent debate.
InProceedings of the 2024 Conference on Empiri-
cal Methods in Natural Language Processing , pages
17889–17904, Miami, Florida, USA. Association for
Computational Linguistics.
Xinbei Ma, Yeyun Gong, Pengcheng He, Nan Duan,
et al. Query rewriting in retrieval-augmented large
language models. In The 2023 Conference on Empir-
ical Methods in Natural Language Processing .
Xinbei Ma, Yeyun Gong, Pengcheng He, Hai Zhao,
and Nan Duan. 2023. Query rewriting in retrieval-
augmented large language models. In Proceedings of
the 2023 Conference on Empirical Methods in Natu-
ral Language Processing , pages 5303–5315, Singa-
pore. Association for Computational Linguistics.Reiichiro Nakano, Jacob Hilton, Suchir Balaji, Jeff Wu,
Long Ouyang, Christina Kim, Christopher Hesse,
Shantanu Jain, Vineet Kosaraju, William Saunders,
et al. 2021. Webgpt: Browser-assisted question-
answering with human feedback. arXiv preprint
arXiv:2112.09332 .
Ofir Press, Muru Zhang, Sewon Min, Ludwig Schmidt,
Noah Smith, and Mike Lewis. 2023. Measuring and
narrowing the compositionality gap in language mod-
els. In Findings of the Association for Computational
Linguistics: EMNLP 2023 , pages 5687–5711, Singa-
pore. Association for Computational Linguistics.
Zhenting Qi, Mingyuan Ma, Jiahang Xu, Li Lyna Zhang,
Fan Yang, and Mao Yang. 2024. Mutual reasoning
makes smaller llms stronger problem-solvers.
David Rein, Betty Li Hou, Asa Cooper Stickland, Jack-
son Petty, Richard Yuanzhe Pang, Julien Dirani, Ju-
lian Michael, and Samuel R Bowman. 2023. Gpqa: A
graduate-level google-proof q&a benchmark. arXiv
preprint arXiv:2311.12022 .
David Rein, Betty Li Hou, Asa Cooper Stickland, Jack-
son Petty, Richard Yuanzhe Pang, Julien Dirani, Ju-
lian Michael, and Samuel R. Bowman. 2024. GPQA:
A graduate-level google-proof q&a benchmark. In
First Conference on Language Modeling .
Timo Schick, Jane Dwivedi-Yu, Roberto Dessì, Roberta
Raileanu, Maria Lomeli, Eric Hambro, Luke Zettle-
moyer, Nicola Cancedda, and Thomas Scialom. 2023.
Toolformer: Language models can teach themselves
to use tools. Advances in Neural Information Pro-
cessing Systems , 36:68539–68551.
Zhihong Shao, Yeyun Gong, Minlie Huang, Nan
Duan, Weizhu Chen, et al. Enhancing retrieval-
augmented large language models with iterative
retrieval-generation synergy. In The 2023 Confer-
ence on Empirical Methods in Natural Language
Processing .
Hongjin Su, Howard Yen, Mengzhou Xia, Weijia Shi,
Niklas Muennighoff, Han-yu Wang, Haisu Liu, Quan
Shi, Zachary S Siegel, Michael Tang, Ruoxi Sun, Jin-
sung Yoon, Sercan O Arik, Danqi Chen, and Tao Yu.
2024. Bright: A realistic and challenging benchmark
for reasoning-intensive retrieval.
Alon Talmor and Jonathan Berant. 2018. The web as
a knowledge-base for answering complex questions.
InProceedings of the 2018 Conference of the North
American Chapter of the Association for Computa-
tional Linguistics: Human Language Technologies,
Volume 1 (Long Papers) , pages 641–651.
Harsh Trivedi, Niranjan Balasubramanian, Tushar Khot,
and Ashish Sabharwal. 2023. Interleaving retrieval
with chain-of-thought reasoning for knowledge-
intensive multi-step questions. In Proceedings of
the 61st Annual Meeting of the Association for Com-
putational Linguistics (Volume 1: Long Papers) ,
pages 10014–10037, Toronto, Canada. Association
for Computational Linguistics.

Xuezhi Wang, Jason Wei, Dale Schuurmans, Quoc V Le,
Ed H. Chi, Sharan Narang, Aakanksha Chowdhery,
and Denny Zhou. 2023. Self-consistency improves
chain of thought reasoning in language models. In
The Eleventh International Conference on Learning
Representations .
Jason Wei, Xuezhi Wang, Dale Schuurmans, Maarten
Bosma, Fei Xia, Ed Chi, Quoc V Le, Denny Zhou,
et al. 2022. Chain-of-thought prompting elicits rea-
soning in large language models. Advances in neural
information processing systems , 35:24824–24837.
Orion Weller, Kathryn Ricci, Eugene Yang, Andrew
Yates, Dawn Lawrie, and Benjamin Van Durme. 2025.
Rank1: Test-time compute for reranking in informa-
tion retrieval.
Yuxi Xie, Anirudh Goyal, Wenyue Zheng, Min-Yen
Kan, Timothy P Lillicrap, Kenji Kawaguchi, and
Michael Shieh. 2024a. Monte carlo tree search
boosts reasoning via iterative preference learning.
arXiv preprint arXiv:2405.00451 .
Yuxi Xie, Kenji Kawaguchi, Yiran Zhao, James Xu
Zhao, Min-Yen Kan, Junxian He, and Michael Xie.
2024b. Self-evaluation guided beam search for rea-
soning. Advances in Neural Information Processing
Systems , 36.
Shunyu Yao, Dian Yu, Jeffrey Zhao, Izhak Shafran,
Thomas L Griffiths, Yuan Cao, and Karthik
Narasimhan. 2023a. Tree of thoughts: deliberate
problem solving with large language models. In
Proceedings of the 37th International Conference
on Neural Information Processing Systems , pages
11809–11822.
Shunyu Yao, Jeffrey Zhao, Dian Yu, Nan Du, Izhak
Shafran, Karthik Narasimhan, and Yuan Cao. 2023b.
React: Synergizing reasoning and acting in language
models. In International Conference on Learning
Representations (ICLR) .
Tian Yu, Shaolei Zhang, and Yang Feng. 2024.
Auto-rag: Autonomous retrieval-augmented gener-
ation for large language models. arXiv preprint
arXiv:2411.19443 .
Dan Zhang, Sining Zhoubian, Ziniu Hu, Yisong Yue,
Yuxiao Dong, and Jie Tang. 2024. Rest-mcts*: Llm
self-training via process reward guided tree search.
arXiv preprint arXiv:2406.03816 .

A Prompts for Each Action
A1 (Direct Response)
Template:
A chat between a curious user and an AI assistant. The assistant
gives step-by-step solutions to the user’s questions. In the end of
the assistant’s response, a final answer must be given in the format
of "The answer is: <ANSWER>.", where <ANSWER> should be a concise
answer.
Usage Example:
{examples}
Instruction:
{instruction}
Note:
Please answer in a complete sentence.
A2 (One-Step Reasoning)
Template:
A chat between a curious user and an AI assistant. The assistant
gives step-by-step solutions to the user’s questions with each step
numbered. At the final step, a conclusive answer must be given in
the format "The answer is: <ANSWER>.", where <ANSWER> should be a
concise answer.
Instruction:
{instruction}
Note:
Let’s think step by step.
A3 (Decompose Answer)
Template:
Given a question, decompose it into sub-questions. For each
sub-question, provide an answer in one complete sentence ending with
"The answer is ". When the original question is answerable, start
the sub-question with "Now we can answer the question: <original
question>".
A4 (Transform Retrieve Query)
Template:
Given a question, generate a search query that would help gather
information to answer it. Your goal is to formulate a query that
retrieves useful evidence or additional details relevant to the
question. The query should be specific enough to ensure that the
search results are both relevant and helpful. Please answer in
one complete sentence, starting with "The query is: <your retrieve
query>".
Question:
{question}

A5 (Reflect Retrieved Knowledge)
Template:
A chat between a curious user and an AI assistant. The assistant
evaluates whether the retrieved information is relevant to the
search query and sufficient to answer the question. Please provide
a concise evaluation in one complete sentence, starting with
"Evaluation:".
Instruction:
Please assess if the retrieved information is related to the query
and can be used to answer the question.
A6 (Summarize Answers)
Template:
Analyze the provided Knowledge and extract key information relevant
to the Original Question . Present your analysis in a concise and
organized format.
Input:
-Original Question: {original_question}
-Knowledge: {retrieved_context}
Output Format:
Key Points: Point 1: Relevant information; Point 2: Relevant
information; Point 3: Relevant information...
Requirement:
The output must be a single line summarizing all key points in one
sentence without redundant description.

B Error Analysis
Query
❌ a 1 Error
❌ a2 ❌ a 4
✅ a 3 ✅ a 3 ❌ a 4 ❌ a 4
Figure 3: An illustration of MCTS Amplification Error. Early MCTS retrieval errors amplify mistakes, leading to a
final answer favoring incorrect paths.
Examples Illustrating Factual Confusion
Question: 
Which U.S. President was in ofﬁce when the ﬁrst man landed on the Moon?
Retrieve Query: 
Who was the U.S. President when the ﬁrst Moon landing happened?  
Retrieved documents:  
John F . Kennedy’ s famous 1962 speech about going to the Moon.  
Richard Nixon’ s recorded phone call with Neil Armstrong during the landing(1969).  
Summary Reasoning:  
(The system assumes that because Kennedy set the goal , he was also the
President when the landing occurred .) 
Kennedy  set the goal , he was also the  President when the landing occurred . 
Incorrect Answer: John F. Kennedy. ❌
Correct Answer:  Richard Nixon ✅
Figure 4: An illustration of Factual Confusion. Wrong understanding of the relationship between project launch and
moon landing, leading to wrong answers.

Examples Illustrating Information Overload
Question:  
What is the capital of the country with the longest coastline?
Retrieve Query:  
Which country has the longest coastline?  
Retrieved documents:  
The world's longest coastline is found in Canada, followed by Indonesia and
Russia. 
Together, the top 10 coastlines extend a total of almost 550,509 kilometers.
At 202,080 kilometers, Canada's coastline is the longest of the top 10,
making up roughly 36.7% of the total length of all 10 coastlines. 
Summary Reasoning:  
Canada is the country with the longest coastline.  
Quickly Reasoning:  
(Instead of immediately asking for Canada’ s capital, the system refocuses on
the coastline details ) 
Let's check the length of Canada's coastline. Canada’ s coastline is
approximately 202,080 km, making it the longest in the world.  
Incorrect Answer:  202,080 km . ❌
Correct Answer:   O t t a w a  ✅Figure 5: An illustration of Information Overload. Too much coastline information, resulting in the model answering
the coastline length instead of the capital city.
Question:  What is the government type where "Le Mali" is the national anthem?
Trace 1
The answer is:  a
republic.Step 1:  Identify the country associated with the
national anthem "Le Mali". T o do this, we need
to understand the meaning of "Le Mali" .A1 A2 A3 A4 A5 A6
Question 1.1: What is the name of the
country where \"Le Mali\" is the national
anthem?
The answer is:  Mali is [...] anthem."  
Trace 3
The asnwer is:  Unitary
semi-presidential
republic. ✅ Trace 2
The answer is: Presidential
representative democratic
republic.
Trace 7
The answer is:  
Mali is a presidential
republic.Step 1:  Identify the country associated with
the national anthem \"Le Mali\".Step 1: We identiﬁed the country that has Le Mali
as its national anthem. According to the given
context , Mali is the country with Le Mali  as its
national anthem.  Trace 8
The answer is:  
Presidential representative
democratic republic.Step2: Next, we looked up the government
type of Mali. According to the provided
context , Mali is a unitary semi-
presidential republic . 
Trace 4
Now that we have the government type of
Mali, [...]. 
The asnwer is:  Unitary semi-presidential
republic. ✅Trace 5
To ﬁnd the government type of the country [...]. 
The answer is:  Unitary semi-presidential
republic. ✅Step 3:  Now that we have conﬁrmed the
government type of Mali,  we can conclude,
Mali is a unitary semi-presidential
republic .Step 2:  Now that we know the country associated
with the national anthem "Le Mali" is Mali,
[...] According to the provided context , Mali is
a unitary semi-presidential republic .
Trace 6
The government type of  [...]. 
The answer is:  Unitary semi-
presidential republic.
✅Question 1.2:  What is the
government type of the country
with \"Le Mali\" as its national
anthem?  
The answer is: Mali is a semi-
presidential republic.Retrieval
DecomposeDirect
AnswerQuick
ReasoningSummary
AnswerDecompose  
QuestionRetrieval
Reasoning
Figure 6: Illustration of how MCTS-RAG achieves a rich reasoning space and tightly integrates reasoning with
retrieval.

Examples Reasoning Correct Hallucination
Question: 
Racemic 3-methylpent-1-ene is treated with Grubbs catalyst. How many possible
products are there (excluding ethene)? 
Step 1: The reaction involves racemic 3-methylpent-1-ene, a terminal alkene
with one stereocenter . Oleﬁn metathesis catalyzed by Grubbs catalyst typically
produces internal alkenes, possibly with stereochemical variations. T o
understand potential outcomes clearly , I should retrieve general information
about the reaction products formed from similar substrates.
Retrieve Query:   
General products from Grubbs-catalyzed metathesis of terminal alkenes with a
single stereocenter
External Document:  
"Grubbs-catalyzed metathesis of terminal alkenes typically produces two internal
alkene products, excluding ethene."
Step 2:  According to the information retrieved, terminal alkenes with
stereocenters undergoing Grubbs-catalyzed metathesis typically produce
internal alkene products with E/Z stereochemistry . Racemic mixtures often
simply double the number of stereoisomers. Thus, for racemic 3-methylpent-1-
ene, I expect there will be 4 distinct products (two stereoisomers per
enantiomer).  
Retrieve Query:
Detailed stereochemical outcomes of Grubbs catalyst-mediated metathesis for
racemic terminal alkenes including possibility of ring-closing metathesis,
excluding ethene
External Document:
"For racemic terminal alkenes with one stereocenter , Grubbs catalyst can
facilitate both cross-metathesis and intramolecular ring-closing metathesis
(RCM). Cross-metathesis generates internal alkene products, yielding four
distinct E/Z stereoisomers due to the racemic substrate. Additionally , ring-
closing metathesis generates cyclic alkene products with two more distinct
stereoisomers (E/Z forms). Thus, the total number of distinct products,
excluding ethene, is six."
Summary Answer :  There are 6 possible stereochemically distinct products
(excluding ethene).  
Correct Answer:  6 ✅ Figure 7: Illustration of the effectiveness of MCTS-RAG. How further reasoning reduces retrieval-introduced
hallucinations and improves accuracy.

Examples Reasoning Processes Enhance Query Generation
Question: 
Racemic 3-methylpent-1-ene is treated with Grubbs catalyst. How many possible
products are there (excluding ethene)? 
Step 1:  The substrate is racemic 3-methylpent-1-ene, a terminal alkene with a
stereocenter at the 3-position, implying two enantiomers.  
Step 2:  Given the racemic mixture, two reaction pathways should be
considered: [...] Each enantiomer might yield distinct stereoisomers, thereby
increasing total possible stereochemical outcomes.
Retrieve Query:  
Detailed stereochemical product distribution for Grubbs catalyst-mediated
metathesis of racemic terminal alkenes with single stereocenter, excluding ethene
External Document:  
"Racemic terminal alkenes possessing a single stereocenter commonly yield two
types of products under Grubbs catalyst conditions:
(1) Cross-metathesis produces internal alkenes, generating E/Z stereoisomers.
With racemic mixtures, each enantiomer independently produces an E/Z pair ,
giving four stereoisomers.
(2) Intramolecular ring-closing metathesis can also occur , forming cyclic products
(cyclobutenes), each with cis/trans (E/Z) isomerism. Considering both enantiomers,
two additional stereoisomers arise.
Thus, excluding ethene, a racemic terminal alkene with one stereocenter can yield
up to six stereochemically distinct products."
Summary Answer :  There are 6 possible stereochemically distinct products
(excluding ethene).  
Correct Answer:  6 ✅ Figure 8: An illustration of the effectiveness of MCTS-
RAG. Based on a clear chain of reasoning, it can gen-
erate higher quality retrieval queries and final answers,
reduce hallucinations and improve accuracy.
Examples Standard RAG Error
Question: 
Racemic 3-methylpent-1-ene is treated with Grubbs catalyst. How many possible
products are there (excluding ethene)? 
Retrieve Query: 
Possible products of racemic 3-methylpent-1-ene with Grubbs catalyst  
External Document ： 
Alkenes treated with Grubbs catalyst undergo oleﬁn metathesis. For simple
terminal alkenes, cross-metathesis typically yields symmetrical and
unsymmetrical alkenes as products along with ethene as a byproduct.
Answer:  
The Grubbs catalyst performs alkene metathesis, breaking and reforming
double bonds. Racemic 3-methylpent-1-ene, being an alkene, will react via
metathesis. T ypically , this reaction yields symmetrical and asymmetrical
products. Considering the double bond location and substituents, I predict
there are three possible distinct products.  
Final Answer:  
The answer is: 3  
Incorrect Answer: 3. ❌
Correct Answer:  6 ✅ Figure 9: An illustration of standard RAG. Because the
reasoning process is not clear enough, the final answer
to the question is an illusion and the answer is wrong.