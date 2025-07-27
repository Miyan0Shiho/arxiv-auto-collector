# CLEAR: Error Analysis via LLM-as-a-Judge Made Easy

**Authors**: Asaf Yehudai, Lilach Eden, Yotam Perlitz, Roy Bar-Haim, Michal Shmueli-Scheuer

**Published**: 2025-07-24 13:15:21

**PDF URL**: [http://arxiv.org/pdf/2507.18392v1](http://arxiv.org/pdf/2507.18392v1)

## Abstract
The evaluation of Large Language Models (LLMs) increasingly relies on other
LLMs acting as judges. However, current evaluation paradigms typically yield a
single score or ranking, answering which model is better but not why. While
essential for benchmarking, these top-level scores obscure the specific,
actionable reasons behind a model's performance. To bridge this gap, we
introduce CLEAR, an interactive, open-source package for LLM-based error
analysis. CLEAR first generates per-instance textual feedback, then it creates
a set of system-level error issues, and quantifies the prevalence of each
identified issue. Our package also provides users with an interactive dashboard
that allows for a comprehensive error analysis through aggregate
visualizations, applies interactive filters to isolate specific issues or score
ranges, and drills down to the individual instances that exemplify a particular
behavioral pattern. We demonstrate CLEAR analysis for RAG and Math benchmarks,
and showcase its utility through a user case study.

## Full Text


<!-- PDF content starts -->

CLEAR : Error Analysis via LLM-as-a-Judge Made Easy
Asaf YehudaiI,H*, Lilach EdenI*, Yotam PerlitzI,
Roy Bar-HaimI,Michal Shmueli-ScheuerI
IIBM ResearchHThe Hebrew University of Jerusalem
{Asaf.Yehudai, Y .Perlitz}@ibm.com {lilache, roybar, shmueli}@il.ibm.com
Abstract
The evaluation of Large Language Models
(LLMs) increasingly relies on other LLMs act-
ing as judges. However, current evaluation
paradigms typically yield a single score or rank-
ing, answering which model is better but not
why. While essential for benchmarking, these
top-level scores obscure the specific, action-
able reasons behind a model’s performance. To
bridge this gap, we introduce CLEAR , an inter-
active, open-source package for LLM-based er-
ror analysis. CLEAR first generates per-instance
textual feedback, then it creates a set of system-
level error issues, and quantifies the prevalence
of each identified issue. Our package also
provides users with an interactive dashboard
that allows for a comprehensive error analysis
through aggregate visualizations, applies inter-
active filters to isolate specific issues or score
ranges, and drills down to the individual in-
stances that exemplify a particular behavioral
pattern. We demonstrate CLEAR analysis for
RAG and Math benchmarks, and showcase its
utility through a user case study.
Code: https://ibm.biz/CLEAR-code-repo
1 Introduction
The evaluation of generative AI systems is rapidly
adopting the LLM-as-a-Judge (LLMaJ) paradigm
(Zheng et al., 2023), where automatic evaluations
by LLMs complement or even replace human anno-
tators. LLM-based judges are commonly applied
to rate or score the quality of LLM responses (Liu
et al., 2023), or to choose a preferred response out
of multiple candidates.
Aggregating these judgments over many exam-
ples provides AI developers with a robust assess-
ment of their system, as well as systematic com-
parison and ranking of different systems or models
(Gera et al., 2025). However, these scores or ratings
*Equal contribution.alone provide little insight into the model’s behav-
ior. AI developers still rely on tedious, manual
error analysis to identify recurring issues, under-
stand the current limitations of their system and
effectively plan and prioritize the next iteration of
improvements.
In this work, we introduce CLEAR , a novel inter-
active tool for AI developers, designed to reduce
the overhead of manual error analysis. Our ap-
proach utilizes an LLMaJ for generating textual
feedback, and conducts discovery of recurring is-
sues via Key Points Analysis (KPA) (Bar-Haim
et al., 2020a). This method allows us to provide
structured, textual feedback that characterizes and
quantifies a model’s recurring weaknesses and is-
sues across a whole dataset. These insights may
guide further improvements, such as prompt engi-
neering, model fine-tuning, or choosing a different
LLM.
TheCLEAR pipeline is illustrated in Figure 1. It
starts with per-instance judgments, which include
both a numeric score and textual feedback. It then
employs a KPA module to categorize these indi-
vidual critiques into a concise set of automatically-
discovered issues. Each identified issue is mapped
back to its matching judgments, which provides
quantification for its prevalence, and allows the
user to drill down from an issue to its specific ex-
amples. Lastly, we provide a user interface that
allows for easy and dynamic exploration of issues
within the data.
To demonstrate our system’s capabilities, we ran
our system on several RAG (contextual question-
answering) and math benchmarks. We analyzed
responses from several systems using different
LLM Judges and KPA implementations. To further
demonstrate our system usability for real-world AI
developers, we also conducted a user study. Its
results confirm the usefulness of CLEAR , and its
potential value for reducing the time and effort re-
quired for error analysis.
1arXiv:2507.18392v1  [cs.CL]  24 Jul 2025

System Responses 
 Jugements Data 
(a) Pipeline (b) UI D R j1… jN J K s issues 
Figure 1: The CLEAR Framework. (a) Pipeline- Given a dataset ( D) and a target system ( s), the system generates
responses ( R). A judge ( J) provides per-instance textual feedback and a score ( {ji}N
i=1). A Key Point Analysis
module ( K) extracts recurring issues and maps them to the individual ji’s. The discovered issues can be explored
via the UI (b).
Our work makes the following contributions:
1.We propose a novel setup for generating au-
tomated system-level issues by summarizing
and structuring instance-level feedback.
2. We present CLEAR , an open-source demo tool
that implements our proposed approach, and
an interactive UI, which together provide an
accessible way for researchers and develop-
ers to gain deeper insights into their models’
behaviors.
3.We demonstrate the system on multiple do-
mains, and conduct a user study to confirm its
effectiveness.
2 Method
CLEAR is designed to produce system-level feed-
back by analyzing a model’s behavior across a
dataset. The full setup, illustrated in Figure 1, takes
as input a dataset of instructions and a target system,
and outputs a concise, structured, and quantified
summary of the system’s recurring issues.
Formally, we assume a dataset D={xn}N
n=1
consisting of Ninstructions, and a target system
s. The system generates a corresponding set of
responses R={rn}N
n=1, where each rn=s(xn).
Our framework then proceeds in two primary
stages:
An LLM-based judge Jis prompted to evaluate
each pair (xn, rn). For each instance, Jreturns a
tuplejn= (tn, sn), where tnis a natural language
critique and snis a numeric quality score. These
instance-level judgments capture localized failures
or strengths observed by the judge. We note that
our setup is reference-less, yet when a reference is
available, it can be used as context for the judge.
The second stage clusters recurring patterns
across the textual feedbacks {tn}N
n=1into a set
of concise, interpretable issues {im}M
m=1. Forefficiency, and because the focus is on identify-
ing shortcomings, only feedback associated with
sn<1is considered during issue generation. Each
tnis then linked to one or more relevant issues from
this set. We explore two distinct implementations
for this aggregation module, denoted K:
Key Point Analysis (KPA) : We adopt a classi-
cal KPA pipeline (Bar-Haim et al., 2020a,b), which
is well-suited for texts containing short sentences,
such as arguments or product reviews. To improve
compatibility, we first break down each tninto a
brief and well-formed sentence using an LLM. We
then apply the KPA method to cluster the sentences
and construct a set of issues over them. This clus-
tering allows for mapping each judgment to the
issues its sentences express.
LLM-Based KPA : As an alternative approach,
we propose LLM-Based KPA. We start by summa-
rizing each critique tninto a shorter, normalized
form via an LLM call. We then prompt an LLM
with a batch of these summaries to identify high-
level recurring issues, and again to remove duplica-
tion and consolidate the final lists of issues. Finally,
eachtnis mapped to the derived issue set via a
matching prompt. This process requires ∼2N
LLM calls. Implementation details are provided in
Appendix E.
3CLEAR Framework
3.1 Pipeline
To support easy integration and usability, we pro-
vide CLEAR as a Python package available on PyPI.
The package implements an end-to-end workflow:
it generates model responses, evaluates them using
an LLM judge, and performs key point analysis to
identify recurring issues. Each component in the
pipeline can be used independently or in combina-
tion, allowing users to customize the workflow to
their specific needs or preferences.
2

Entry view
Provides an easy 
entry point to our 
tool, with 
explanation of its 
main components.Issues View
Provides overview of model issues to help identify the system 
error patterns.
Insight: The model suffers from calculation errors as its major problem Entry view to the CLEAR interfacea
Filtering 
Mechanism
Provides a dynamic 
ability to filter the 
data by issues 
types, and score 
range.
Model 
Behavior
Provides instance -
level info on data 
slices to enable 
issue -based error 
analysis
Insight: By filtering for 
instances marked with 
specific issues, we can 
choose examples that 
help us understand the 
model's behavior and its 
error patterns. We can 
also examine the 
available instance 
metadata and the 
judge's explanations.Overview of instances g
The issues distribution as recognized by our approachb
The filtering mechanism control panel  c
Comparison View
Provides a comparison view of the issues distribution in the full 
dataset compared to the filtered one.
Insight: Misunderstanding of the problem appear in most cases that also 
exhibit mathematical and logical errors.
Issues frequencies comparison of the full and filter data d Detailed view of selected instancee
Instance -level view f
Figure 2: The figure presents the key components of the CLEAR tool for analyzing model evaluation results. (a)
Entry point to the interface for exploring the model results and issues. (b) Issues View visualizes the distribution
of detected model errors. (c) The Filtering Mechanism allows filtering based on issue types and scores to isolate
relevant examples. (d) Comparison View contrasts issue frequencies between the full dataset and filtered subsets,
highlighting co-occurrence patterns. (e/f) Model Behavior and Instance-Level View offer detailed, example-level
insights to facilitate fine-grained error analysis and model diagnosis.
Moreover, the CLEAR pipeline is designed to be
highly configurable. It supports multiple inference
API providers for generating predictions or, alter-
natively, allows running the evaluation step inde-
pendently if predictions are already available. In
cases where judgments have been obtained sepa-
rately, the pipeline can directly execute the final
evaluation step ( Kstep).
Evaluation Modes. To accommodate different
preferences for issue discovery, CLEAR supports
three evaluation modes: (1) General : issues are
discovered dynamically using a general-purpose
evaluation prompt, enabling broad, exploratory
assessment without requiring data-specific prior
knowledge; (2) Task-specific : users provide spe-
cific issues as evaluation criteria, guiding the judge
while allowing for additional discoveries, and (3)Static a predefined list of issues supplied by the
user is given to the judge as the sole evaluation
criteria and mapped directly to evaluation texts,
without any dynamic discovery.
Code Example CLEAR can be installed via PyPI:
• • •
$ pip install clear - eval
Once installed, the full analysis can be executed
with a single CLI command:
• • •
$ run -clear - eval - analysis --
config_path =< path_to_config >
3

Or with the next three lines of Python code:
• • •
from clear_eval . analysis_runner
import run_clear_eval_analysis
config_path = <path -to - config >
run_clear_eval_analysis ( config_path
)
where <path_to_config> is the path to a YAML
file containing the basic configuration options, such
as the selected judge and provider, path to input
data, path to the results folder, and other optional
parameters.
Once processing is complete, the Streamlit inter-
face can be launched using the CLI command be-
low. The results ZIP file saved to the specified out-
put directory can then be manually loaded through
the app.
• • •
$ run -clear - eval - dashboard
3.2 UI
To support intuitive and effective exploration of
model errors, CLEAR includes a visual analytics
interface designed for both researchers and prac-
titioners. The user interface (Figure 2) provides
multiple synchronized views that together offer a
comprehensive understanding of model behavior,
error patterns, and their distribution.
The interface is composed of the following key
components:
Issues View. The Issues View (Figure 2b) dis-
plays an overview of all the issues identified by
the system. Each issue is listed along with its fre-
quency and percentage in the dataset. This helps
users identify dominant failure patterns at a glance.
It also helps the user understand the severity of the
presented issues.
Filtering Mechanism. The filtering panel (Fig-
ure 2c) enables users to narrow down the dataset
based on a specific combination of issue types or
score range. We allow the union or intersection
of issues or their negation. This is essential for
targeted exploration—for example, isolating only
high-scoring answers with logical errors, or filter-
ing for responses that demonstrate extractiveness
issues. The panel offers dynamic control, and anyapplied filters immediately update the rest of the
views.
Comparison View. The Comparison View (Fig-
ure 2d) visualizes how issue frequencies change
when filtering is applied. This comparison allows
users to better understand the connection between
different issues and, issues and score range.
Model Behavior and Instance-Level View. The
bottom row of the interface (Figure 2e, f) focuses
on instance-level analysis. Users can drill down
into specific examples, inspect the original instruc-
tion and response, the judge’s textual feedback, and
the list of issues it was mapped to. This direct link
between abstract issues and concrete examples can
help users understand how different issues effect
their system behavior.
Together, these views make CLEAR a tool for ana-
lyzing LLM output beyond scalar metrics. The
interface supports both broad patterns and fine-
grained inspection, helping practitioners uncover
failure trends, identify brittle behavior, and better
understand how model responses fail in practice.
4CLEAR : Case Study
4.1 Setup
To study our method’s behavior, we utilize three
datasets: GSM8K (Cobbe et al., 2021) for math
word problems, and two retrieval-augmented gen-
eration (RAG) datasets: TechQA (Castelli et al.,
2019) and DelucionQA (Sadat et al., 2023), based
on the processing of RAGBench (Friel et al.,
2024). We run evaluations over four open systems:
Mixtral 8x7B (Jiang et al., 2024), LLaMA-3.1
8B (Grattafiori et al., 2024), Granite-3.3 8B (Gran-
ite Team, 2024), and Phi-4 (Abdin et al., 2024).
We generate responses for each dataset using
these systems and apply our pipeline to assess
them. For the judgment component, we employ
two strong models in a reference-less setting with
general and task-specific modes. As a high-quality
closed-source judge, we use GPT-4o (OpenAI et al.,
2024), and as an open-source alternative, we use
LLaMA-3.3 70B. The per-instance feedback gen-
erated by these judges is passed through two ver-
sions of the Kmodule: the IBM watsonx®Key
Point Analysis (KPA) implementation1, and our
LLM-Based KPA using both GPT-4o and LLaMA-
3.3 70B. This full pipeline produces, for each sys-
1IBM watsonx KPA
4

GSM8K (s: Mixtral 8x7B, J: GPT-4o)
– No Issues Detected ( 78.4%)
–Mathematical errors in calculations, including rounding
and final steps. ( 13.2%)
–Incorrect understanding of problem statements leading to
flawed reasoning. ( 11.8%)
–Failure to fully consider or correctly interpret all given
information. ( 5.8%)
–Incomplete answers due to missing necessary steps or
calculations. ( 5.5%)
– Logical errors despite clear reasoning. ( 4.3%)
–Misunderstanding or incorrect application of mathematical
concepts or methods. ( 3.3%)
–Unnecessary complexity or inclusion of irrelevant details.
(2.6%)
– Incorrect handling of units or conversions. ( 0.6%)
– Failure to verify or cross-check results. ( 0.2%)
Table 1: Issues identified for Mixtral 8x7B over the
GSM8K benchmark with a task-specific mode, sorted
by decreasing frequency (shown in the parentheses).
Mixtral 8x7B (J: GPT-4o)
– No Issues Detected ( 51.9%)
– Omission of necessary details or steps ( 36.3%)
–Lack of specificity and completeness in responses ( 31.2%)
– Omission of relevant links or references ( 9.2%)
– Inaccurate or irrelevant information ( 8.6%)
–Failure to provide actionable insights or solutions ( 8.3%).
– Misinterpretation or misuse of context( 4.5%)
– Lack of clarity in explaining technical details ( 3.5%)
– Incomplete or abrupt ending of the response ( 3.5%)
Phi-4 (J: GPT-4o)
– No Issues Detected ( 76.6%)
– Lacks completeness and necessary details ( 10.9%)
– Lacks context-specific information ( 9.9%)
– Lacks specificity in technical details ( 6%)
–Fails to mention unsupported features or limitations
(5.1%)
– Inaccurate or fabricated information ( 2.6%)
– Does not directly answer the question ( 1.9%)
– Assumes unsupported or incorrect context ( 1.9%)
Table 2: Issues identified for Mixtral 8x7B and Phi-4
over TechQA with a general mode, sorted by decreasing
frequency (shown in the parenthesis)
tem–dataset pair, a set of recurring issues derived
from the judge feedback.
4.2 Results
Actionable Insights. Table 1 presents CLEAR re-
sults of Mixtral 8x7B evaluated on GSM8K with
a task-specific mode. The results reveal that Mix-
tral 8x7B most prominent weaknesses stem from
calculation-related errors. Issues also include in-
correct application of mathematical concepts and
difficulties in handling units or conversions. These
issues can help detect whether a model’s failures
stem from reasoning gaps, execution errors, or both.Importantly, this insight is actionable: developers
may choose to augment training data with synthetic
examples focusing on numeric reasoning (Yehu-
dai et al., 2024), while users might compensate by
pairing the model with external tools like calcula-
tors (Qin et al., 2023).
Data Dependent Issues. In the upper part of Ta-
ble 2 we present the CLEAR results from evaluating
Mixtral 8x7B (with J: GPT-4o) on TechQA with a
general mode. Issues include missing context, lack
of specificity in technical content, and hallucinated
or fabricated information—a well-known failure
mode of LLMs in RAG settings. Notably, CLEAR
demonstrates the ability to adapt issue discovery to
both the task and the dataset. For example, the ex-
tracted issues explicitly reflect TechQA’s demand
for accurate, domain-specific technical details, il-
lustrating how the system tailors feedback to the
characteristics of each benchmark.
System Impact. To study how CLEAR behaves
across different systems, we analyze results on
TechQA for Mixtral 8x7B and Phi-4 (see Table 2).
Firstly, we observe that the two systems produce
different sets of issues, highlighting that CLEAR
provides system-specific diagnostic feedback. For
example, Mixtral 8x7B exhibits unique problems
such as “Omission of relevant links or references”
and “Failure to provide actionable insights or solu-
tions,” which are not observed for Phi-4.
Secondly, we find that the overall proportion
of flagged instances is lower for Phi-4, 48.1%
for Mixtral 8x7B versus 23.4% for Phi-4. This
aligns with the models’ overall quality as reported
in the literature: Phi-4 achieves an 84.8 MMLU
score (Hendrycks et al., 2021) and a 1257 Elo
score on Chatbot Arena (Chiang et al., 2024), com-
pared to Mixtral 8x7B’s 1194 MMLU and 1194
Elo scores. These findings suggest that CLEAR can
support high-level system comparisons, in addition
to offering fine-grained diagnostic feedback.
Impact of Evaluation Modes. We also com-
pare the use of task-specific and general evaluation
modes. Our findings show that the task-specific
mode increases CLEAR ’s sensitivity to issues closely
tied to the task. In contrast, the general mode tends
to reveal a broader range of more nuanced or unan-
ticipated problems. For example, on RAG datasets,
the task-specific prompt helped to expose more
faithfulness-related issues, such as “Generates un-
supported or speculative information”, that were
5

missed by the general prompt. On the other hand,
the general mode discovered more novel issues like
“Incomplete or abrupt ending of the response”(See
appendix C for a full comparison).
Impact of KPA method. Finally, we quantita-
tively assess the three implementations of the K
module for key point analysis. Our results indi-
cate that LLM-based KPA tends to produce issues
that are more synthesized and less extractive than
the traditional KPA approach. Among the mod-
els evaluated, GPT-4o produced more accurate and
specific issue types compared to LLaMA-3.3 and
Watsonx’s implementation (see Appendix B).
4.3 User study
To evaluate the usefulness and usability of CLEAR
for users, we conducted a user study with 12 AI
practitioners and researchers. Participants were
asked to use the tool on the three datasets, explore
the interface, and provide feedback via a structured
questionnaire (On a Likert scale) and free-form
comments. The assessed dimensions included the
usefulness of the tool, comparison to their current
practices, and their trust in the tool. (see App. D for
a detailed description of the participants, instruc-
tions, and questions).
Results. The results indicate that users found
CLEAR valuable for surface-level analysis. Specif-
ically, participants appreciated the automation of
error detection (noting that 75% currently rely on
manual inspection for their use cases), the visual
exploration interface, and the potential to detect
issues they would have overlooked, with an aver-
age rating of 4.33on a Likert scale. The system
was seen as actionable , with 74% of participants
reporting they would take or consider taking action
based on the output, time-saving , and better than
existing practices , with both aspects receiving av-
erage scores of 4.25. It was especially valued for
identifying common failure modes at scale, with
an average score of 4.16.
Despite the generally positive reception, users
pointed out areas for improvement. Several re-
sponses expressed uncertainty regarding the trust-
worthiness ,3.83score, and specificity of the sur-
faced issues. Some found the descriptions to be
vague or had difficulty understanding which errors
were most critical. Users also requested features
such as severity annotations, clearer categories, au-
tomatic summaries, and better highlighting within
the textual feedback.5 Related Work
Existing error analysis tools map model errors and
abilities by inspecting dataset instances, such as
EvalTree’s (Zeng et al., 2025) capability hierar-
chies or Qualeval’s (Murahari et al., 2023) data
attributes. Other methods are interactive, like Eru-
dite (Wu et al., 2019), which requires user labels
for clustering errors, or use specialized models
like MisattributionLLM (Xu et al., 2025) to score
known error types.
Crucially, all these methods depend on labeled
data, restricting them to specific tasks. Moreover,
as these works probe for weaknesses or skills based
on dataset features rather than model-specific be-
havior, they are likely to miss idiosyncratic model
failure modes.
6 Conclusion and Future work
We presented CLEAR , a novel framework and inter-
active tool for automating error analysis of genera-
tive AI systems. By leveraging LLMaJ evaluations
and Key Point Analysis, CLEAR extracts structured,
system-level feedback from instance-level critiques.
This enables AI developers to move beyond scalar
metrics and surface recurring, actionable failure
patterns with minimal manual overhead.
Our experiments across math and RAG datasets
demonstrate that CLEAR adapts to different tasks
and models, revealing both common and system-
specific issues. The tool offers flexibility in eval-
uation modes, supports multiple judges and KPA
configurations, and includes an intuitive visual in-
terface for deep exploration of model behavior.
Our user study confirms the tool’s value for prac-
titioners, highlighting its ability to save time, pro-
vide new insights, and improve analysis workflows,
though it also points to areas for further enhance-
ment.
Looking ahead, we plan to improve the speci-
ficity and clarity of the discovered issues, incorpo-
rate severity scoring and prioritization mechanisms,
and explore methods for increasing user trust and
interpretability. Additionally, we aim to integrate
interactive feedback loops that allow users to refine
or correct discovered issues.
CLEAR is publicly available, and we hope it will
serve the community as a stepping stone toward
more transparent, efficient, and insightful evalua-
tion of generative models.
6

References
Marah Abdin, Jyoti Aneja, Harkirat Behl, Sébastien
Bubeck, Ronen Eldan, Suriya Gunasekar, Michael
Harrison, Russell J. Hewett, Mojan Javaheripi, Piero
Kauffmann, James R. Lee, Yin Tat Lee, Yuanzhi Li,
Weishung Liu, Caio C. T. Mendes, Anh Nguyen, Eric
Price, Gustavo de Rosa, Olli Saarikivi, Adil Salim,
Shital Shah, Xin Wang, Rachel Ward, Yue Wu, Dingli
Yu, Cyril Zhang, and Yi Zhang. 2024. Phi-4 technical
report. Preprint , arXiv:2412.08905.
Roy Bar-Haim, Lilach Eden, Roni Friedman, Yoav Kan-
tor, Dan Lahav, and Noam Slonim. 2020a. From ar-
guments to key points: Towards automatic argument
summarization. In Proceedings of the 58th Annual
Meeting of the Association for Computational Lin-
guistics , pages 4029–4039, Online. Association for
Computational Linguistics.
Roy Bar-Haim, Yoav Kantor, Lilach Eden, Roni Fried-
man, Dan Lahav, and Noam Slonim. 2020b. Quanti-
tative argument summarization and beyond: Cross-
domain key point analysis. In Proceedings of the
2020 Conference on Empirical Methods in Natural
Language Processing (EMNLP) , pages 39–49, On-
line. Association for Computational Linguistics.
Vittorio Castelli, Rishav Chakravarti, Saswati Dana, An-
thony Ferritto, Radu Florian, Martin Franz, Dinesh
Garg, Dinesh Khandelwal, Scott McCarley, Mike
McCawley, Mohamed Nasr, Lin Pan, Cezar Pen-
dus, John Pitrelli, Saurabh Pujar, Salim Roukos, An-
drzej Sakrajda, Avirup Sil, Rosario Uceda-Sosa, Todd
Ward, and Rong Zhang. 2019. The techqa dataset.
Preprint , arXiv:1911.02984.
Wei-Lin Chiang, Lianmin Zheng, Ying Sheng, Anasta-
sios Nikolas Angelopoulos, Tianle Li, Dacheng Li,
Hao Zhang, Banghua Zhu, Michael Jordan, Joseph E.
Gonzalez, and Ion Stoica. 2024. Chatbot arena: An
open platform for evaluating llms by human prefer-
ence. Preprint , arXiv:2403.04132.
Karl Cobbe, Vineet Kosaraju, Mohammad Bavarian,
Mark Chen, Heewoo Jun, Lukasz Kaiser, Matthias
Plappert, Jerry Tworek, Jacob Hilton, Reiichiro
Nakano, Christopher Hesse, and John Schulman.
2021. Training verifiers to solve math word prob-
lems. Preprint , arXiv:2110.14168.
Robert Friel, Masha Belyi, and Atindriyo Sanyal. 2024.
Ragbench: Explainable benchmark for retrieval-
augmented generation systems. arXiv preprint
arXiv:2407.11005 .
Ariel Gera, Odellia Boni, Yotam Perlitz, Roy Bar-
Haim, Lilach Eden, and Asaf Yehudai. 2025. Jus-
trank: Benchmarking llm judges for system ranking.
Preprint , arXiv:2412.09569.
IBM Granite Team. 2024. Granite 3.0 language models.
Aaron Grattafiori, Abhimanyu Dubey, Abhinav Jauhri,
Abhinav Pandey, Abhishek Kadian, Ahmad Al-
Dahle, et al. 2024. The Llama 3 herd of models.
Preprint , arXiv:2407.21783.Dan Hendrycks, Collin Burns, Steven Basart, Andy Zou,
Mantas Mazeika, Dawn Song, and Jacob Steinhardt.
2021. Measuring massive multitask language under-
standing. Preprint , arXiv:2009.03300.
Albert Q. Jiang, Alexandre Sablayrolles, Antoine
Roux, Arthur Mensch, Blanche Savary, Chris
Bamford, Devendra Singh Chaplot, Diego de las
Casas, Emma Bou Hanna, Florian Bressand, Gi-
anna Lengyel, Guillaume Bour, Guillaume Lam-
ple, Lélio Renard Lavaud, Lucile Saulnier, Marie-
Anne Lachaux, Pierre Stock, Sandeep Subramanian,
Sophia Yang, Szymon Antoniak, Teven Le Scao,
Théophile Gervet, Thibaut Lavril, Thomas Wang,
Timothée Lacroix, and William El Sayed. 2024. Mix-
tral of experts. Preprint , arXiv:2401.04088.
Yang Liu, Dan Iter, Yichong Xu, Shuohang Wang,
Ruochen Xu, and Chenguang Zhu. 2023. G-eval:
Nlg evaluation using gpt-4 with better human align-
ment. Preprint , arXiv:2303.16634.
Vishvak Murahari, Ameet Deshpande, Peter Clark,
Tanmay Rajpurohit, Ashish Sabharwal, Karthik
Narasimhan, and Ashwin Kalyan. 2023. Qualeval:
Qualitative evaluation for model improvement. arXiv
preprint arXiv:2311.02807 .
OpenAI, :, Aaron Hurst, Adam Lerer, Adam P. Goucher,
Adam Perelman, Aditya Ramesh, Aidan Clark,
AJ Ostrow, Akila Welihinda, et al. 2024. Gpt-4o
system card. Preprint , arXiv:2410.21276.
Yujia Qin, Shihao Liang, Yining Ye, Kunlun Zhu, Lan
Yan, Yaxi Lu, Yankai Lin, Xin Cong, Xiangru Tang,
Bill Qian, Sihan Zhao, Lauren Hong, Runchu Tian,
Ruobing Xie, Jie Zhou, Mark Gerstein, Dahai Li,
Zhiyuan Liu, and Maosong Sun. 2023. Toolllm: Fa-
cilitating large language models to master 16000+
real-world apis. Preprint , arXiv:2307.16789.
Mobashir Sadat, Zhengyu Zhou, Lukas Lange, Jun
Araki, Arsalan Gundroo, Bingqing Wang, Rakesh
Menon, Md Parvez, and Zhe Feng. 2023. Delu-
cionQA: Detecting hallucinations in domain-specific
question answering. In Findings of the Association
for Computational Linguistics: EMNLP 2023 , pages
822–835, Singapore. Association for Computational
Linguistics.
Tongshuang Wu, Marco Tulio Ribeiro, Jeffrey Heer,
and Daniel S Weld. 2019. Errudite: Scalable, repro-
ducible, and testable error analysis. In Proceedings
of the 57th Annual Meeting of the Association for
Computational Linguistics , pages 747–763.
Zishan Xu, Shuyi Xie, Shupei Xiao, Linlin Song, Sui
Wenjuan, Fan Lin, and Lv Qingsong. 2025. Misattri-
butionLLM: Integrating error attribution capability
into LLM evaluation.
Asaf Yehudai, Boaz Carmeli, Yosi Mass, Ofir Arviv,
Nathaniel Mills, Assaf Toledo, Eyal Shnarch, and
Leshem Choshen. 2024. Genie: Achieving hu-
man parity in content-grounded datasets generation.
Preprint , arXiv:2401.14367.
7

Zhiyuan Zeng, Yizhong Wang, Hannaneh Hajishirzi,
and Pang Wei Koh. 2025. Evaltree: Profiling lan-
guage model weaknesses via hierarchical capability
trees. arXiv preprint arXiv:2503.08893 .
Lianmin Zheng, Wei-Lin Chiang, Ying Sheng, Siyuan
Zhuang, Zhanghao Wu, Yonghao Zhuang, Zi Lin,
Zhuohan Li, Dacheng Li, Eric Xing, Hao Zhang,
Joseph E Gonzalez, and Ion Stoica. 2023. Judging
LLM-as-a-judge with MT-bench and chatbot arena.
InAdvances in Neural Information Processing Sys-
tems, volume 36, pages 46595–46623. Curran Asso-
ciates, Inc.A Limitations
Our framework’s effectiveness is fundamentally
dependent on the quality of its constituent mod-
els. The analysis pipeline inherits the biases and
potential inaccuracies of both the underlying LLM-
as-a-Judge ( J) and the Key Point Analysis ( K)
module.
Dependence on Judge Quality The discovered
issues are only as reliable as the initial critiques
from the judge model. Biases in the judge (e.g.,
self-bias, length/style bias) or their failure to iden-
tify subtle errors directly compromise the quality
and validity of the final, aggregated issues.
Scalability and Cost The LLM-based KPA ap-
proach, while effective, can be computationally
expensive, requiring approximately ∼2NLLM
calls for a dataset of size N. However, we apply
the LLM-based KPA only to instances with low
initial evaluation scores, making the cost depen-
dent on the target system’s quality. Nevertheless,
this overhead may present a practical limitation for
large-scale analyses.
Lack of Causality Our tool identifies and quan-
tifies recurring error patterns but does not diagnose
their root cause. For instance, it can highlight fre-
quent factual inaccuracies but cannot distinguish if
this stems from knowledge deficits, retrieval fail-
ures, or flawed reasoning.
B Method Comparison
Table 3 presents a qualitative comparison of the
key points generated by the three KPA imple-
mentations: Watsonx-KPA, LLM-based KPA with
LLaMA-3, and LLM-based KPA with GPT-4o.
While all three aim to identify recurring system-
level issues, they differ in terms of abstraction level,
phrasing style, and the generality of the resulting
key points. Below, we describe two central dimen-
sions of variation we observed across the methods.
Extractive vs. Synthesized Styles. The
Watsonx-KPA method produces highly extractive
and granular key points, often lifted nearly verba-
tim from the original feedback. As a result, its
issues tend to be overly specific (e.g., “The calcu-
lation in step 7 is incorrect” ) or tied to particular
instance structures. While this precision can sur-
face concrete issues, it reduces generalizability and
often results in a long list of narrowly scoped prob-
lems. In contrast, LLM-based KPA—especially
8

with GPT-4o—generates more abstracted and syn-
thesized issue types. These key points aggregate
multiple occurrences of related errors into broader
categories, such as “Failure to fully consider or
correctly interpret all given information” , which
promote better generalization across examples and
systems.
Issue Granularity and Clarity. LLM-based
methods also differ in their balance between ab-
straction and clarity. The LLaMA-3-based KPA
sometimes produces long, compound key points
with broad scopes (e.g., “Failure to account for
all relevant variables, conditions, or scenarios” ),
which may reduce readability and introduce redun-
dancy. GPT-4o, on the other hand, tends to produce
concise and well-structured key points, striking a
balance between informativeness and clarity (e.g.,
“Incorrect handling of units or conversions” ). In
contrast, Watsonx’s output can feel fragmented or
repetitive due to its instance-tethered phrasing, of-
ten leading to multiple key points covering overlap-
ping aspects of the same underlying issue.
C Impact of evaluation mode
Table 4 shows the full list of discovered issues for
mixtral 8x7b over the TechQA benchmark, using
the general vs. the task-specific evaluation mode,
as discussed in paragraph Impact of Evaluation
Modes under section §4.2.
D User Study Details
Among the 12 participants, 7 are application de-
velopers, 3 are business analysts, and 1 is a model
developer.
To set the context for the study, we began with a
brief overview of the tool, followed by step-by-step
instructions (see Figure 3).
The questions included a 1-5 Likert-scale,
multiple-choice, and open-ended formats, divided
into three sections to assess different aspects of the
tool:
1.Usefulness - This section explores how help-
ful the tool was in understanding model errors,
saving time, or influencing debugging or de-
velopment decisions (Figure 4).
2.Comparative value - This section compares
the tool to your current approach—manual in-
spection or existing tools—and helps identify
areas where it adds (or lacks) value (Figure 5).
Figure 3: Instructions to the study participants.
3.Trust & reliability- This section assesses how
much you trust the tool’s outputs and whether
it gives you confidence in your understanding
of model behavior (Figure 6).
E Implementation Details
Setup For response generations, all models were
prompted with default parameters. For all evalua-
tion stages, inference was performed with temper-
ature 0. The system was set to produce between
3 and 15 key points for each analysis. Issue syn-
thesis was performed using up to 150 evaluation
summaries with non perfect scores. All prompts
are provided in the Git Repo.
9

Dataset watsonx-KPA LLM-based (GPT-4o) LLM-based (LLaMA-3-70B)
GSM8K– No Issues Detected ( 78.3%)
– The error leads to an incorrect
final answer. ( 17.8%)
– The equations are set up
inaccurately. ( 14.4%)
– The model fails to provide correct
reasoning. ( 12.3%)
– The calculation in step 7 is
incorrect. ( 11.2%)
– The error leads to an incorrect
average. ( 10.5%)
–However, the rounding error could
be misleading. ( 7.1%)
– It also lacks clarity. ( 6.3%)
– It does not verify calculations.
(5.6%)
– The steps in the model answer are
incomplete. ( 5.4%)
– The model incorrectly calculates
the number of pairs. ( 4.5%)
– Necessary steps to solve the
problem are missing. ( 3.9%)– No Issues Detected ( 83.7%)
– Incorrect understanding of
problem statements leading to
flawed reasoning ( 7.5%)
– Mathematical errors in
calculations, including rounding
and final steps ( 7.4%)
– Incomplete answers due to
missing necessary steps or
calculations ( 4.1%)
– Failure to fully consider or
correctly interpret all given
information ( 3.9%)
– Unnecessary complexity or
inclusion of irrelevant details
(1.7%)
– Logical errors despite clear
reasoning ( 1.1%)
– Incorrect handling of units or
conversions ( 0.4%)
– Misunderstanding or incorrect
application of mathematical
concepts or methods ( 0.3%)
– Failure to verify or cross-check
results ( 0.2%)– No Issues Detected (81.4
– Calculation errors or inaccuracies
(15.2%)
– Flawed reasoning, logical errors,
or incorrect application of
formulas/algorithms ( 7.1%)
– Incorrect assumptions or
misinterpretations of problem
statements ( 5.2%)
– Failure to account for all relevant
variables, conditions, or scenarios
(5.2%)
– Lack of clarity, consistency, or
unnecessary complexity in
explanations ( 3.1%)
– Inability to correctly interpret or
apply given information ( 2.9%)
TechQA– No Issues Detected ( 4.5%)
– The response lacks completeness
and clarity. ( 89.5%)
– The model answer lacks relevance
and factual support. ( 70.7%)
– It fails to provide
document-supported solutions.
(59.2%)
– The model misses key details
about the vulnerability. ( 55.1%)
– As a result, the response lacks
faithfulness. ( 54.8%)
–It contains unnecessary details and
inaccuracies. ( 41.4%)
– It misses several necessary
troubleshooting steps. ( 26.4%)
– The model lacks actionable steps
and insights. ( 21.3%)
– However, it lacks specific
configuration steps for ITCAM.
(8.6%)
– However, the additional join
information is unnecessary.
(6.4%)
– The answer misses specific details
about cache synchronization.
(3.8%)– No Issues Detected ( 18.5%)
– Lacks completeness and omits
crucial details ( 59.6%)
– Generates unsupported or
speculative information ( 31.8%)
– Fails to accurately incorporate
document information ( 22.0%)
– Provides irrelevant or extraneous
information ( 14.3%)
– Lacks clarity and conciseness
(14.0%)
– Fails to address the specific
question ( 12.4%)
– Fails to provide direct solutions
(8.0%)
– Lacks structured presentation
(3.8%)– No Issues Detected 26 ( 8.3%)
– Lack of detail and clarity in
answers ( 67.8%)
– Inadequate consideration of
context and user needs ( 45.9%)
– Failure to directly address the
user’s question ( 43.3%)
– Unclear or incomplete solutions
(36.6%)
– Lack of faithfulness to original
documents ( 26.8%)
– Introduction of unnecessary
information ( 24.2%)
– Failure to provide supporting
evidence or examples ( 22.3%)
– Failure to provide clear steps or
instructions ( 19.7%)
– Lack of relevance to the specific
question or topic ( 19.4%)
– Inaccuracies and inconsistencies
in information ( 15.3%)
– Overreliance on general
knowledge ( 14.0%)
– Failure to consider multiple
possible causes or solutions
(8.3%)
– Inadequate explanation of
technical terms ( 2.2%)
Table 3: Top issues identified for Mixtral 8x7B, with each method across GSM8K and TechQA benchmarks, with a
task-specific evaluation mode.
10

General (J: GPT-4o)
– No Issues Detected ( 51.9%)
– Omission of necessary details or steps ( 36.3%)
– Lack of specificity and completeness in responses
(31.2%)
– Omission of relevant links or references ( 9.2%)
– Inaccurate or irrelevant information ( 8.6%)
–Failure to provide actionable insights or solutions ( 8.3%)
– Misinterpretation or misuse of context( 4.5%)
– Lack of clarity in explaining technical details ( 3.5%)
– Incomplete or abrupt ending of the response ( 3.5%)
Task-Specific (J: GPT-4o)
– No Issues Detected ( 18.5%)
– Lacks completeness and omits crucial details ( 59.6%)
– Generates unsupported or speculative information
(31.8%)
– Fails to accurately incorporate document information
(22.0%)
– Provides irrelevant or extraneous information ( 14.3%)
– Lacks clarity and conciseness ( 14.0%)
– Fails to address the specific question ( 12.4%)
– Fails to provide direct solutions ( 8.0%)
– Lacks structured presentation ( 3.8%)
Table 4: Issues identified for Mixtral 8x7B over
TechQA, using general (top) and task-specific (bottom)
evaluation modes.
Figure 4: Section 1- Usefulness questions.
Figure 5: Section 2- Comparative value questions.
Figure 6: Section 3- Trust & Reliability questions.
11