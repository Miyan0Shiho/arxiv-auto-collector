# DrugMCTS: a drug repurposing framework combining multi-agent, RAG and Monte Carlo Tree Search

**Authors**: Zerui Yang, Yuwei Wan, Yinqiao Li, Yudai Matsuda, Tong Xie, Linqi Song

**Published**: 2025-07-10 04:39:55

**PDF URL**: [http://arxiv.org/pdf/2507.07426v1](http://arxiv.org/pdf/2507.07426v1)

## Abstract
Recent advances in large language models have demonstrated considerable
potential in scientific domains such as drug discovery. However, their
effectiveness remains constrained when reasoning extends beyond the knowledge
acquired during pretraining. Conventional approaches, such as fine-tuning or
retrieval-augmented generation, face limitations in either imposing high
computational overhead or failing to fully exploit structured scientific data.
To overcome these challenges, we propose DrugMCTS, a novel framework that
synergistically integrates RAG, multi-agent collaboration, and Monte Carlo Tree
Search for drug repurposing. The framework employs five specialized agents
tasked with retrieving and analyzing molecular and protein information, thereby
enabling structured and iterative reasoning. Without requiring domain-specific
fine-tuning, DrugMCTS empowers Qwen2.5-7B-Instruct to outperform Deepseek-R1 by
over 20\%. Extensive experiments on the DrugBank and KIBA datasets demonstrate
that DrugMCTS achieves substantially higher recall and robustness compared to
both general-purpose LLMs and deep learning baselines. Our results highlight
the importance of structured reasoning, agent-based collaboration, and
feedback-driven search mechanisms in advancing LLM applications for drug
discovery.

## Full Text


<!-- PDF content starts -->

DrugMCTS: a drug repurposing framework combining multi-agent,
RAG and Monte Carlo Tree Search
Zerui Yang Yuwei Wan Yinqiao Li Yudai Matsuda Tong Xie
Linqi Song
June 2025
Abstract
Recent advances in large language models have
demonstrated considerable potential in scientific do-
mains such as drug discovery. However, their ef-
fectiveness remains constrained when reasoning ex-
tends beyond the knowledge acquired during pre-
training. Conventional approaches, such as fine-
tuning or retrieval-augmented generation, face limita-
tions in either imposing high computational overhead
or failing to fully exploit structured scientific data. To
overcome these challenges, we propose DrugMCTS, a
novel framework that synergistically integrates RAG,
multi-agent collaboration, and Monte Carlo Tree
Search for drug repurposing. The framework employs
five specialized agents tasked with retrieving and an-
alyzing molecular and protein information, thereby
enabling structured and iterative reasoning. With-
out requiring domain-specific fine-tuning, DrugM-
CTS empowers Qwen2.5-7B-Instruct to outperform
Deepseek-R1 by over 20%. Extensive experiments on
the DrugBank and KIBA datasets demonstrate that
DrugMCTS achieves substantially higher recall and
robustness compared to both general-purpose LLMs
and deep learning baselines. Our results highlight the
importance of structured reasoning, agent-based col-
laboration, and feedback-driven search mechanisms
in advancing LLM applications for drug discovery.
1 Introduction
Large language models (LLMs) have demonstrated
remarkable capabilities across a wide range of do-
mains, including question answering, logical reason-
ing, and knowledge-intensive tasks such as mathe-
matics and code generation. These models are in-
creasingly being explored for applications in scientific
fields, particularly in drug discovery [1]. However,
when confronted with problems that lie beyond their
pre-training knowledge or inherent reasoning abili-
ties, such as predicting novel drug-target interactions,their performance may fall short of expectations [2].
As interest grows in applying general-purpose large
models to scientific domains, various approaches
have been proposed to address the aforementioned
limitations. Among them, fine-tuning on domain-
specific datasets has become a widely adopted
paradigm [3] [4]. However, this approach comes with
several notable drawbacks. First, fine-tuning de-
mands substantial computational resources, and most
current methods are tailored to specific domains. Ex-
tending these models to new domains typically re-
quires additional training, which is computationally
inefficient and often impractical for many users.
Moreover, scientific knowledge is inherently dy-
namic, constantly evolving with new discoveries.
This necessitates continuous updates to the model’s
internal knowledge base, which in turn introduces
challenges such as increased training costs and the
risk of catastrophic forgetting [5]. These issues sig-
nificantly limit the scalability and practicality of fine-
tuning-based approaches in real-world scientific ap-
plications.
To mitigate the limitations of fine-tuning, an al-
ternative paradigm, retrieval-augmented generation
(RAG), has gained increasing attention [6]. In this
framework, external agents are employed to retrieve
relevant information from scientific literature and
databases, which is then used to augment the rea-
soning capabilities of LLMs [7]. This approach elim-
inates the need for fine-tuning and enables access to
up-to-date knowledge, making it particularly appeal-
ing for fast-evolving domains like drug discovery.
However, significant challenges remain. As noted
in prior work [38], Data in scientific domain can be
broadly categorized into two types: scientific data
and general-purpose data. Most existing RAG-based
systems heavily rely on the latter due to its compat-
ibility with general-purpose LLMs [8], while largely
overlooking the value of structured, authoritative sci-
entific data, such as molecular structures and protein
sequences, which is often cleaner and more informa-
1arXiv:2507.07426v1  [cs.AI]  10 Jul 2025

tive than generic text. For instance, in drug-target
interaction prediction tasks, some approaches only
provide knowledge graphs or textual descriptions as
context, completely omitting structural information
about proteins or molecules [10]. This omission un-
dermines both the reliability of the model’s predic-
tions and the interpretability of its decision-making
process.
Some approaches incorporate domain-specific mod-
els during the reasoning phase to better understand
scientific data [9]. While effective in certain contexts,
this strategy inherits many of the limitations of the
fine-tuning paradigm. Notably, when the input data
deviates from the distribution seen during training,
the performance of these models can degrade signifi-
cantly. For example, deep learning models for drug-
target interaction prediction may experience accu-
racy drops of over 20% when encountering previously
unseen molecule-protein pairs [11].
On the other hand, general-purpose data sources
often contain irrelevant or even erroneous informa-
tion [12], necessitating preprocessing steps such as fil-
tering and cleaning before use. However, LLMs may
inadvertently discard useful content due to incom-
plete contextual understanding or internal biases. As
highlighted in [1], drug discovery should ideally be
an iterative process driven by feedback and refine-
ment. Yet, most current approaches rely on single-
step inference without mechanisms for error correc-
tion or knowledge updating, limiting their robustness
and adaptability [36] [37].
To address the aforementioned limitations, we pro-
pose DrugMCTS, a novel drug discovery algorithm
based on RAG, multi-agent collaboration, and Monte
Carlo Tree Search (MCTS) [34]. Our system com-
prises five specialized agents:
•Retrieval Agent: Identifies and gathers poten-
tially relevant molecules.
•Molecule-Analysis Agent: Evaluates the physic-
ochemical and pharmacological properties of the
query molecule.
•Molecule-Selection Agent: Filters out molecules
with limited therapeutic potential or low rele-
vance.
•Interaction-Analysis Agent: Interprets the
molecular mechanisms underlying drug-target
interactions.
•Decision Agent: Integrates all available evidence
and generates final recommendations.With the guidance of MCTS, our lightweight
Qwen7b-based model [40] achieves performance ex-
ceeding that of Deepseek-R1 [33] and GPT-4o-
mini [32], surpassing it by approximately 20% on
key benchmarks. We also conduct ablation studies
demonstrating that removing any component of our
pipeline results in a performance drop of 2–10%. Our
key contributions include:
a. We introduce an end-to-end drug discovery
framework that enables Qwen2.5-7B-Instruct to
outperform much larger models Deepseek-R1.
The method does not require any domain-specific
fine-tuned models; instead, it leverages external
knowledge to enhance reasoning and decision-
making.
b. We propose a systematic workflow that integrates
scientific data, hybrid scientific-general data, and
general-purpose data, leveraging the strengths of
each type. This pipeline not only improves the
interpretability of model decisions but also es-
tablishes a standardized framework applicable be-
yond drug-target interaction tasks.
c. By incorporating a feedback mechanism through
Monte Carlo Tree Search, our framework enables
the model to iteratively refine its understanding,
filter noisy data, and autonomously identify the
most valuable information for decision-making.
2 Method
2.1 Overview
Our framework, DrugMCTS, as shown in Figure 1,
takes a query molecule Mqmas input and processes
it through a series of specialized agents to output
proteins that can potentially interact with Mqm.
The process begins with the Retrieval Agent, which
queries databases based on the molecular structure of
theMqmto identify similar molecules, thereby gener-
ating a pool of candidate molecules Mcm. Next, the
Molecule-Analysis Agent retrieves chemical proper-
ties of the query molecule via API calls and generates
a detailed report Rqm. This report is then passed to
the Molecule-Selection Agent, which filters the Mcm
based on the analysis, producing a refined list of
reference molecules Mrm. The Interaction-Analysis
Agent identifies proteins Prpthat can interact with
the reference molecules Mrmby querying relevant
databases. For each identified protein-molecule pair,
the agent retrieves the corresponding binding pocket
data Dbpfrom the Protein Data Bank (PDB) [13]
using API calls and python libraries. Additionally,
2

it fetches literature descriptions Lrprelated to the
proteins from public repositories such as PubMed. It
then compiles all this information, Mrm,Prp,Pbpand
Lrp, into a comprehensive interaction report Riafor
each molecule-protein pair.
Finally, the Decision Agent integrates all available
information, including the Mqm,Rqm,Mrm,Dbp, and
Ria, to select the most promising target protein.
To enhance decision-making, we employ MCTS
during inference time scaling. Unlike the retrieval
agent, other agents generate multiple answers in a
single invocation. These answers are evaluated using
the Upper Confidence Bound applied to Trees (UCT)
algorithm (Equation 1) [19] to select and expand op-
timal nodes until reaching the leaf node. We use
a self-confidence scoring method for leaf nodes and
backpropagate scores accordingly. The final selection
of proteins is determined by majority voting, choos-
ing the top N proteins that appear most frequently.
This approach allows our model to autonomously se-
lect high-quality data without additional fine-tuning,
effectively achieving data cleaning through feedback
mechanisms.
UCT =Wi
ni+cr
lnN
ni(1)
2.2 Data Processing Pipeline
In prior work [38], data in scientific domain has
been categorized into two types: scientific data and
general data. Scientific data includes highly spe-
cialized information such as molecular formulas and
amino acid sequences, characterized by its authorita-
tive nature, well-structured format, and cleanliness.
These datasets are typically sourced from professional
databases like the PDB [13], where data storage for-
mats adhere to strict standards. However, LLMs of-
ten struggle to interpret such structured data directly
due to its complexity, necessitating the use of domain-
specific models like ESM2 [14] or ChemBERTa [15]
for processing.
Conversely, general data encompasses more diverse
and less standardized sources, such as research pa-
pers, which are easier for LLMs to understand but
may contain noise and inaccuracies. Leveraging the
strengths of both data types while mitigating their
weaknesses is crucial for effective drug discovery.
Based on the original work, we propose a new
data type called hybrid scientific-general data. Tools
like RDKit [16] can accept scientific inputs, such as
SMILES representations of molecules, and output
structured yet textually described data, including chi-
ral centers, functional groups, and Murcko scaffolds,packaged as Python dictionaries. Similarly, PLIP [17]
methods can process PDB files to extract pocket
information and return it in tabular form. These
hybrid data outputs retain the authority and well-
structured nature of scientific data while being more
interpretable by LLMs due to their textual explana-
tions.
Our workflow involves transforming both molecular
and protein data through three stages: from scientific
data to hybrid scientific-general data, and finally to
general data. This structured pipeline ensures that
our model benefits from the precision and reliability
of scientific data while maintaining interpretability
and ease of integration with LLMs. By systemat-
ically leveraging these different data types, we not
only improve the accuracy and robustness of our pre-
dictions but also enhance the explainability of our
model’s decision-making process.
2.3 Formulating Drug Discovery as a
Tree Search Problem
Given that MCTS is a well-established algorithm
widely adopted and thoroughly explained in various
studies [18], we provide only an overview here. The
MCTS algorithm consists of four main steps: selec-
tion, expansion, simulation, and backpropagation.
In each rollout, the process begins at the root
node and proceeds by selecting a leaf node using the
UCT algorithm. The selected leaf node is then ex-
panded by generating one or more child nodes. This
selection-expansion cycle continues iteratively until
reaching an end node. Upon reaching this terminal
node, a predefined scoring rule is applied to evalu-
ate its quality. The score is then backpropagated up
the tree to update the scores and visit counts of all
nodes along the path. After completing a pre-defined
number of rollouts, we obtain a series of candidate
solutions from which the final answer is chosen.
2.4 Action Space
The action space refers to the set of possible actions
that can be performed during each expansion phase.
Similar to chain-of-thought (CoT) reasoning, these
actions are sequential and interdependent; each sub-
sequent action builds upon the results of the previous
one. Therefore, coordination among different agents
is essential for coherent execution.
Our framework includes six distinct actions corre-
sponding to five specialized agents. Except for the
retrieval agent, which does not invoke any model, all
other agents utilize the same LLM without requiring
additional fine-tuned models.
3

Figure 1: Workflow of DrugMCTS
A1Retrieval Action. Upon receiving the query
molecule, the Retrieval agent queries databases to
identify molecules structurally similar to the query
molecule. We employ two similarity metrics: the
Tanimoto coefficient [29] and the cosine similarity
based on the last hidden state computed by Chem-
BERTa. For each metric, we retrieve the top-10
most similar molecules, merge the results, and remove
duplicates to form the initial candidate molecules
Mcm. The proteins can interact with Mcmare also
retrieved, forming candidate proteins Pcp. These
molecules serve as inputs to the root node of the
Monte Carlo tree search.
Since the similarity scores are deterministic, the
resulting molecule pool remains consistent, satisfying
the requirement that the root node must be unique.
simC(Mqm, Mi) =hqm·hi
∥hqm∥ · ∥hi∥(2)Mcm= dedup
Top10(sim T(Mqm))
∪Top10(sim C(Mqm))
(3)
A2Molecule Analysis Action. Understand-
ing molecular properties is crucial for predicting
molecular-target interactions. While general-purpose
LLMs can perform some analysis using SMILES rep-
resentations, their interpretations are often incom-
plete and prone to errors, especially when experimen-
tal data, such as hydrophobicity, is required.
To address this limitation, the Molecule Analy-
sis (MA) agent first utilizes RDKit and PubChemPy
APIs [20] to extract a set of structural and physic-
ochemical properties ( Cq,s&Cq,phy) for the query
molecule Mqmby calling RDKit and PubChemPy
APIs. These properties include:
•Structural features: chiral centers, scaffolds, and
functional groups.
4

•Physicochemical properties: molecular weight,
lipophilicity (logP), polar surface area (PSA),
hydrogen bond donors/acceptors, rotatable
bonds, and heavy atom counts.
Based on this structured and quantified infor-
mation, the agent then generates a comprehensive
molecular analysis report Rqm.
Rqm= MA Agent( Mqm, Cq,s, Cq,phy) (4)
A3Molecule Selection Action. Research has
shown that the quality of retrieved information sig-
nificantly impacts the accuracy of model-generated
answers. Excessive irrelevant information can neg-
atively affect the model’s performance. Although
the retrieved molecules share structural similarities
with the query molecule, they may not necessar-
ily provide useful insights for drug discovery tasks.
Thus, the Molecule Selection (MS) agent filters the
molecule pool, based on structural similarity to the
query molecule, pharmacophore integrity, and drug-
like essentials, to generate reference molecules Mrm.
The reference proteins Prpare obtained by selecting
the proteins that can interact with Mrmfrom Pcp.
To ensure that these reference molecules are thor-
oughly characterized, this action also invokes the A2
Molecule Analysis Action . Specifically, the MA
agent is called to retrieve the structural and physico-
chemical properties ( Cc,s&Cc,phy) ofMcmbut with-
out generating reports.
Mrm= MS Agent( Mqm, Rqm, Mcm, Cc,s, Cc,phy)
(5)
A4Interaction Analysis Action. In this step,
we aim to analyze potential interactions between the
MrmandPcp. A major challenge lies in interpreting
protein structures from amino acid sequences, which
general-purpose LLMs struggle to handle due to their
complexity. To overcome this, we adopt the method-
ology from DrugRealign [22], utilizing Python’s PLIP
library to extract binding pocket information Dbp
from PDB files and present it in textual format. In
contrast to the original tabular representation which
can be difficult for LLMs to parse, we reformat each
entry into a descriptive paragraph to improve inter-
pretability. Additionally, we retrieve relevant scien-
tific literature Lrpfrom PubMed [21] to provide con-
textual support for interaction analysis by Interaction
Analysis (IA) agent.
Ria= IA Agent( Mrm, Prp, Dbp, Lrp) (6)
A5Protein Selection Action. At this stage,
the Decision Agent synthesizes all available informa-
tion, including the Mqm,Rqm,Mrm,Pcp,Dbp, andRia.Based on this integrated knowledge, the agent se-
lects the most promising target protein Psfrom the
full list of candidates.
Ps= Decision Agent( Mqm, Rqm, Mrm, Prp, Dbp, Ria)
(7)
A6End Action. This action does not involve any
agent invocation. When the model selects the final
protein, the End Action is executed. Upon encoun-
tering an end node, the MCTS algorithm terminates
further expansion, evaluates the end node’s score, and
backpropagates the updated values to all nodes along
the traversal path, concluding the current rollout.
2.5 Reward Calculation
Predicting molecular-protein binding affinity typi-
cally involves two primary approaches: molecular
docking methods [23] and deep learning-based meth-
ods [11].
Molecular Docking Methods. Molecular dock-
ing methods, such as AutoDock Vina [23], are widely
used but suffer from notable limitations. First, they
require accurate three-dimensional (3D) structural
information of the target proteins, which is often un-
available or unreliable for many biologically relevant
molecules. Second, these methods are computation-
ally intensive, limiting their scalability in large-scale
drug discovery tasks. Given these limitations, there
has been a growing interest in developing more effi-
cient alternatives.
Deep Learning-Based Methods. Deep learn-
ing models have emerged as a promising solution due
to their superior computational efficiency [39]. How-
ever, their performance is highly dependent on the
training dataset. When predicting samples that devi-
ate significantly from the training data, the accuracy
of these models can drop by over 30%, limiting their
applicability across diverse scenarios.
2.5.1 Self-Consistency Score.
To address these challenges, our work adopts an alter-
native reward calculation method known as the self-
consistency score [28]. This approach involves query-
ing the model multiple times with the same question
and selecting the most frequently occurring answer
p∗as the final output. The frequency of this an-
swer serves as the relative reward . However, this
method introduces a potential issue: if all candidate
proteins in one rollout exhibit strong binding affin-
ity with the query molecule, while in another rollout
none do, the calculated relative rewards could still be
similar despite the stark differences in actual binding
5

affinities. To mitigate this limitation, we introduce
an absolute reward mechanism.
Absolute Reward. The absolute reward is com-
puted by inputting the Ps,Dbp,Lrp,Mqm, and Rqm
into a decision-making model. This model evaluates
whether there is a significant interaction between the
protein and the query molecule. The frequency of
affirmative responses (”yes”) is then used as the ab-
solute reward.
Final Reward Calculation. The final reward for
each rollout is calculated as the average of the rela-
tive reward and the absolute reward. This combined
approach ensures that both the consistency and the
strength of the predicted interactions are taken into
account, providing a more robust evaluation metric.
Rrelative (p∗) =Number of times p∗is selected
Total number of selections(8)
Rabsolute (p∗) =Number of ”yes” responses
k(9)
Rfinal(p∗) =Rrelative (p∗) +Rabsolute (p∗)
2(10)
3 Experiments
3.1 Datasets and Metrics
We utilized two datasets, DrugBank [31] and
KIBA [30], which were processed to include a total
of 788 entries from DrugBank and 626 entries from
KIBA. Each entry consists of a molecule as input and
its corresponding interacting proteins as output. The
number of ground truth interactions per entry varies.
To evaluate model performance, we used recall, de-
fined as the ratio of correctly predicted proteins to
the total number of ground truth proteins. The de-
tailed information of data processing pipeline can be
found in Appendix Section A.1
Recall =|{proteins predicted correctly }|
|{all ground truth proteins }|(11)
3.2 Settings
For the retrieval phase, we employed
ESM2 t33650M UR50D to compute cosine sim-
ilarity. For all other inference stages, we used
Qwen2.5-7B-Instruct. Specifically, each search
process involved 12 rounds of rollouts. Except for
the protein selection and end actions, which generateonly one node during expansion, all other actions
generated four nodes with distinct answers per
expansion. The temperature was set to 0.8. Both
relative and absolute rewards were computed by
generating four responses per rollout, also with a
temperature of 0.8.
3.3 Baselines
We established three sets of baselines to compare our
model’s performance:
General Models. (GM) We selected GPT-4o-
mini and Deepseek-R1 as general-purpose models. In
this group, we provided the models with minimal in-
formation: the SMILES representation of the query
molecule, reference molecules, and candidate proteins
along with their pocket types.
General Models with RAG. (GM + RAG)
Many existing studies on drug repurposing using
LLMs either employ divergent methodological formu-
lations or are not open-sourced, which complicates
comparative evaluation. Our approach adopts the
framework of the DrugReAlign [22] method, with
modifications and enhancements tailored to our spe-
cific formulation. To ensure equitable comparison,
we incorporate not only protein pocket information
but also incorporate Cq,sandCq,phy as additional
features.. The models used in this group remained
GPT-4o-mini and Deepseek-R1.
Deep Learning Models. (DL Models)
We trained four deep learning models: Atten-
tionDTA [25], GraphDTA [24], DeepConv DTI [26],
and Perceiver CPI [27], on both DrugBank and KIBA
datasets. We extracted data involving the query
molecules as test sets and used the remaining data
for training. All four models achieved over 70% ac-
curacy on the test sets. During testing, we applied
majority voting to select the final answer by averag-
ing the scores from the four models and choosing the
top-k proteins, where k corresponds to the length of
the ground truth.
3.4 Results
The experimental results (Figure 1) indicate that
general-purpose LLMs (GPT-4o-mini and Deepseek-
R1) achieved relatively low recall scores of only
12.59%–16.19% on the DrugBank dataset when op-
erating in a zero-shot setting. However, when in-
corporating molecular structural features and chem-
ical properties via RAG-based prompting, model
performance decreased, with GPT-4o-mini dropping
to 15.19% and Deepseek-R1 significantly falling to
12.59%. This suggests that the inclusion of poten-
6

tially irrelevant or misleading information through
retrieval can negatively impact the reasoning capa-
bilities of general-purpose LLMs.
Among the deep learning baselines, the ensem-
ble of four specialized models (AttentionDTI, GIN-
ConvNet, DeepConv DTI, Perceiver CPI) achieved
a recall score of 23.64%, representing an 84% im-
provement over the best-performing general-purpose
model. On the KIBA dataset, the same ensemble at-
tained a recall score of 26.45%, further demonstrating
its effectiveness in capturing drug-target interactions.
Our proposed method, DrugMCTS, significantly
outperformed all baseline approaches. Using a base
TopK strategy, DrugMCTS achieved a recall of
44.66% on DrugBank and 42.24% on KIBA. This
represents improvements of approximately 88.9% and
31.7% over the best deep learning baselines, respec-
tively. Furthermore, our dynamic adjustment strat-
egy (TopK+3) boosted performance to 55.34% on
DrugBank and 49.24% on KIBA, marking maximum
improvements of 330% and 91.4% over the general-
purpose models. It is worth noting that although the
other three methods showed improved performance
on the KIBA dataset, our method exhibited a slight
drop. However, according to Appendix Table 2, this
apparent improvement in other baseline settings is
largely due to an increased ratio of ground truth can-
didates among the total options. As shown in Ap-
pendix Table 1, compared to DrugBank, the KIBA
dataset contains a larger number of candidate pro-
teins per drug, which increases the difficulty of se-
lecting the correct Pcpafter molecule selection. This
observation indirectly highlights the importance of
effective filtering mechanisms.
These results strongly underscore the superiority of
dynamic decision-making mechanisms, such as those
used in DrugMCTS, over traditional static prediction
methods like deep learning models. They also high-
light the limitations of general-purpose large mod-
els in zero-shot settings for drug discovery tasks,
especially when retrieval-augmented prompting in-
troduces noise or irrelevant context. This further
emphasizes the importance of structured reasoning,
domain-specific knowledge integration, and informa-
tion filtering in such applications.
Furthermore, to demonstrate the interpretability
and transparency of our approach, we provide a de-
tailed case study in the Appendix Section A.2. It
illustrates how the model predicts an interaction be-
tween Equol and the CXC chemokine receptor 3, in-
cluding the complete step-by-step reasoning process
during a specific MCTS rollout.4 Ablation Studies
4.1 Settings
In our ablation studies, we aim to investigate several
key aspects:
•Whether the MCTS algorithm can improve
model accuracy.
•The effectiveness of our proposed data processing
pipelines for scientific data, a hybrid of scientific
and general data, and general data.
•The efficacy of the combined relative and abso-
lute reward calculation method.
To address these questions, we conducted the follow-
ing experiments:
•S1Baseline Setup. Provide only the query
molecule, all proteins in the protein pool, and
the types of their pocket. Do not use the MCTS
algorithm.
•S2Enhanced Information (EI) Setup. On
top of the baseline setup, add the detailed pock-
ets information and literature information for all
proteins and structural and chemical properties
of the query molecule.
•S3Molecule Analysis Exclusion (MAE).
Conduct the MCTS process while excluding the
molecule analysis action.
•S4Interaction Analysis Exclusion (IAE).
Conduct the MCTS process while excluding the
interaction analysis action.
•S5Dual Exclusion (DE). Conduct the MCTS
process while excluding both the molecule anal-
ysis and interaction analysis actions.
•S6Relative Reward (RR) Only. During the
MCTS process, compute only the relative re-
wards without considering the absolute rewards.
4.2 Ablation Studies Results
The ablation study results (Figure 2) clearly demon-
strate the effectiveness of our proposed framework
components. First, comparing S1(Baseline Setup)
with S2(Enhanced Information Setup) shows that
providing richer contextual information, including de-
tailed pocket features, literature descriptions, struc-
tural and chemical properties, does improve perfor-
mance to some extent (e.g., from 12.85% to 15.86% on
7

Table 1: Performance comparison on DrugBank and KIBA datasets
Model Size Dynamic Update DrugBank KIBA
General Models (GM)
GPT-4o-mini ∼8B × 0.1552 0.2580
Deepseek-R1 37Ba× 0.1619 0.2645
GM + RAG
GPT-4o-mini ∼8B ✓ 0.1519 0.2252
Deepseek-R1 37Ba✓ 0.1259 0.2173
DL Models
DL models 8M-12M × 0.2364 0.3216
DrugMCTS (Ours)
Selection = GT count 7B ✓ 0.4466 0.4224
Selection = GT + 3 7B ✓ 0.5534 0.4924
aActivation-aware model size. GT = Ground Truth count. Dynamic update: ✓Yes,×No.
Table 2: Performance comparison on DrugBank and KIBA datasets using Qwen7b (Top-k/Top-k+3 accu-
racy).
Setup S1Baseline S2EI S3MAE S4IAE S5DE S6RR DrugMCTS
DrugBank 0.1285 0.1586 0.3879/0.4677 0.3946/0.5119 0.3472/0.3617 0.4320/0.5527 0.4466/0.5534
KIBA 0.2284 0.2452 0.3772/0.4352 0.3846/0.4491 0.3189/0.3264 0.4193/0.4861 0.4224/0.4924
DrugBank). However, the most significant improve-
ment is observed when the MCTS algorithm is intro-
duced in combination with these enhancements. Ex-
perimental settings that incorporate MCTS ( S3–S6
and Final Result) consistently achieve much higher
performance than S1orS2, indicating that while
richer input representations are beneficial, it is the
MCTS-based reasoning process that plays the cen-
tral role in boosting model accuracy.
Second, by analyzing S3(Molecule Analysis Ex-
clusion), S4(Interaction Analysis Exclusion), and S5
(Dual Exclusion), we observe a consistent drop in per-
formance when either or both of the analysis modules
are removed. For instance, on the DrugBank dataset,
removing molecule analysis alone leads to a decrease
from 44.66% (Final Result) to 38.79%, while remov-
ing interaction analysis results in a drop to 39.46%.
The dual exclusion further reduces performance to
34.72%, demonstrating that each data processing
step contributes meaningfully to the overall effective-
ness of the system. This supports our hypothesis that
the proposed hybrid data processing pipeline, incor-
porating both molecular and interaction-level analy-ses, is essential for capturing comprehensive contex-
tual information.
Third, regarding the reward mechanism, the com-
parison between S6(Relative Reward Only) and the
full reward setting (Final Result) shows that the com-
bined use of relative and absolute rewards does not
lead to a significant improvement in performance.
One possible explanation is that prior steps, includ-
ing candidate protein and molecule selection, have
already filtered out most irrelevant options, leav-
ing a refined set of high-quality reference proteins.
As a result, the likelihood of encountering scenarios
where none of the candidates interact with the query
molecule becomes rare, reducing the added value of
the absolute reward component.
In summary, these findings confirm the importance
of the MCTS algorithm, the multi-step data process-
ing pipeline, and the overall design of the retrieval-
augmented reasoning framework in achieving strong
performance in molecular-target interaction predic-
tion.
8

4.3 Computation Overhead Results
For inference time scaling, the discussion typically re-
volves around two key aspects: the trade-off between
model performance improvement and additional com-
putational overhead. Therefore, in this section, we
first analyze the impact of different rollout num-
bers on model performance and then compare the
performance-overhead profile with baseline models.
Our analysis (Figure 2 and Figure 3) reveals that
when the number of rollouts increases from 8 to 12,
both Top-K and Top-K+3 metrics exhibit signifi-
cant improvements across the two datasets. How-
ever, further increasing the rollout count from 12 to
24 only yields a notable gain in the Top-K+3 met-
ric on the KIBA dataset, while other scenarios show
either marginal or even negative improvements. Con-
sequently, to balance computational cost and model
performance, we ultimately adopt rollout=12 for our
experiments. Compared to baseline models, our ap-
proach not only achieves the highest recall scores
but also demonstrates superior cost efficiency, as ev-
idenced by its position on the Pareto front.
5 Conclusion
Our DrugMCTS framework revolutionizes drug re-
purposing by integrating multi-agent collaboration
(five specialized agents), hybrid data processing (sci-
entific→hybrid→general), and Monte Carlo Tree
Search to enable the lightweight Qwen2.5-7B-Instruct
model to outperform Deepseek-R1 by more than 20%
recall on DrugBank/KIBA datasets. The system
achieves 55.34% recall via dynamic Top-K+3 selec-
tion, validated by 1,221 experimental interactions
and case studies like Equol-CXCR3 binding (docking
score: -8.4 kcal/mol). This work establishes a tem-
plate for LLM-powered scientific discovery beyond
drug-target prediction.
6 Limitations
While DrugMCTS demonstrates significant improve-
ments over baseline models, several limitations high-
light opportunities for further optimization:
•Despite achieving more than 20% recall gains
over Deepseek-R1 (Table 2), the absolute per-
formance (55.34% recall) suggests untapped op-
timization potential. The plateau in gains be-
yond 12 rollouts indicates diminishing returns
from current MCTS configurations.•Current predictions primarily leverage PDB-
derived binding pocket data (Section 2.4), omit-
ting higher-order biological context. Future
work may augment the framework with knowl-
edge graph or Pathway activation score.
•The combined relative/absolute reward system
(Eq. 10) yields only around 1% improvement
over relative-only rewards, suggesting the neces-
sities of a more effective reward system.
References
[1] Ye, G., Cai, X., Lai, H., Wang, X., Huang, J.,
Wang, L., Liu, W. & Zeng, X. Drugassist: A
large language model for molecule optimization.
Briefings In Bioinformatics .26, bbae693 (2025)
[2] Zheng, Y., Koh, H., Ju, J., Nguyen, A., May,
L., Webb, G. & Pan, S. Large language mod-
els for scientific discovery in molecular property
prediction. Nature Machine Intelligence . pp. 1-
11 (2025)
[3] Zhang, W., Wang, Q., Kong, X., Xiong, J., Ni,
S., Cao, D., Niu, B., Chen, M., Li, Y., Zhang,
R. & Others Fine-tuning large language models
for chemical text mining. Chemical Science .15,
10600-10611 (2024)
[4] Van Herck, J., Gil, M., Jablonka, K., Abrudan,
A., Anker, A., Asgari, M., Blaiszik, B., Buffo,
A., Choudhury, L., Corminboeuf, C. & Others
Assessment of fine-tuned large language models
for real-world chemistry and material science ap-
plications. Chemical Science .16, 670-684 (2025)
[5] Nguyen, C., Achille, A., Lam, M., Hassner, T.,
Mahadevan, V. & Soatto, S. Toward understand-
ing catastrophic forgetting in continual learning.
ArXiv Preprint ArXiv:1908.01091 . (2019)
[6] Zhang, P., Peng, X., Han, R., Chen, T. & Ma,
J. Rag2Mol: Structure-based drug design based
on Retrieval Augmented Generation. Briefings
In Bioinformatics .26, bbaf265 (2025)
[7] Che, X., Zhao, Y., Liu, Q., Yu, F., Gao, H.
& Zhang, L. CSstep: Step-by-step exploration
of the chemical space of drug molecules via
multi-agent and multi-stage reinforcement learn-
ing.Chemical Engineering Science . pp. 122048
(2025)
[8] Song, K., Trotter, A. & Chen, J. Llm agent
swarm for hypothesis-driven drug discovery.
ArXiv Preprint ArXiv:2504.17967 . (2025)
9

[9] Inoue, Y., Song, T., Wang, X., Luna, A.
& Fu, T. Drugagent: Multi-agent large lan-
guage model-based reasoning for drug-target in-
teraction prediction. ICLR 2025 Workshop On
Machine Learning For Genomics Explorations .
(2025)
[10] Lee, N., De Brouwer, E., Hajiramezanali, E.,
Biancalani, T., Park, C. & Scalia, G. RAG-
Enhanced Collaborative LLM Agents for Drug
Discovery. ArXiv Preprint ArXiv:2502.17506 .
(2025)
[11] Yang, Z., Li, Y., Matsuda, Y. & Song, L.
mHMG-DTI: A Drug-Target Interaction Pre-
diction Framework Combining Modified Hierar-
chical Molecular Graphs and Improved Convo-
lutional Block Attention Module. Trends And
Applications In Knowledge Discovery And Data
Mining: PAKDD 2025 Workshops, ADUR,
FairPC, GLFM, PM4B And RAFDA, Sydney,
NSW, Australia, June 10–13, 2025, Proceedings .
15835 pp. 191 (2025)
[12] Hutter, J., Rau, D., Marx, M. & Kamps, J. Lost
but not only in the middle: Positional bias in
retrieval augmented generation. European Con-
ference On Information Retrieval . pp. 247-261
(2025)
[13] Bank, P. Protein data bank. Nature New Biol .
233, 10-1038 (1971)
[14] Lin, Z., Akin, H., Rao, R., Hie, B., Zhu, Z.,
Lu, W., Smetanin, N., Verkuil, R., Kabeli, O.,
Shmueli, Y. & Others Evolutionary-scale predic-
tion of atomic-level protein structure with a lan-
guage model. Science .379, 1123-1130 (2023)
[15] Chithrananda, S., Grand, G. & Ramsundar,
B. ChemBERTa: large-scale self-supervised
pretraining for molecular property prediction.
ArXiv Preprint ArXiv:2010.09885 . (2020)
[16] Landrum, G. Rdkit documentation. Release .1,
4 (2013)
[17] Salentin, S., Schreiber, S., Haupt, V., Adasme,
M. & Schroeder, M. PLIP: fully automated pro-
tein–ligand interaction profiler. Nucleic Acids
Research .43, W443-W447 (2015)
[18] Li, B., Zhang, J., Fan, J., Xu, Y., Chen, C.,
Tang, N. & Luo, Y. Alpha-sql: Zero-shot text-to-
sql using monte carlo tree search. ArXiv Preprint
ArXiv:2502.17248 . (2025)[19] Cou¨ etoux, A., Hoock, J., Sokolovska, N., Tey-
taud, O. & Bonnard, N. Continuous upper con-
fidence trees. Learning And Intelligent Optimiza-
tion: 5th International Conference, LION 5,
Rome, Italy, January 17-21, 2011. Selected Pa-
pers 5 . pp. 433-445 (2011)
[20] Swain, M. PubChemPy documentation. Pub-
ChemPy Documentation . (2014)
[21] White, J. PubMed 2.0. Medical Reference Ser-
vices Quarterly .39, 382-387 (2020)
[22] Wei, J., Zhuo, L., Fu, X., Zeng, X., Wang, L.,
Zou, Q. & Cao, D. DrugReAlign: a multisource
prompt framework for drug repurposing based
on large language models. BMC Biology .22, 226
(2024)
[23] Huey, R., Morris, G., Forli, S. & Others Using
AutoDock 4 and AutoDock vina with AutoDock-
Tools: a tutorial. The Scripps Research In-
stitute Molecular Graphics Laboratory .10550 ,
1000 (2012)
[24] Nguyen, T., Le, H., Quinn, T., Nguyen, T.,
Le, T. & Venkatesh, S. GraphDTA: predicting
drug–target binding affinity with graph neural
networks. Bioinformatics .37, 1140-1147 (2021)
[25] Zhao, Q., Xiao, F., Yang, M., Li, Y. & Wang,
J. AttentionDTA: prediction of drug–target
binding affinity using attention model. 2019
IEEE International Conference On Bioinfor-
matics And Biomedicine (BIBM) . pp. 64-69
(2019)
[26] Lee, I., Keum, J. & Nam, H. DeepConv-DTI:
Prediction of drug-target interactions via deep
learning with convolution on protein sequences.
PLoS Computational Biology .15, e1007129
(2019)
[27] Nguyen, N., Jang, G., Kim, H. & Kang, J. Per-
ceiver CPI: a nested cross-attention network for
compound–protein interaction prediction. Bioin-
formatics .39, btac731 (2023)
[28] Wang, X., Wei, J., Schuurmans, D., Le, Q.,
Chi, E., Narang, S., Chowdhery, A. & Zhou,
D. Self-consistency improves chain of thought
reasoning in language models. ArXiv Preprint
ArXiv:2203.11171 . (2022)
[29] Bajusz, D., R´ acz, A. & H´ eberger, K. Why
is Tanimoto index an appropriate choice for
fingerprint-based similarity calculations?. Jour-
nal Of Cheminformatics .7pp. 1-13 (2015)
10

[30] Tang, J., Szwajda, A., Shakyawar, S., Xu, T.,
Hintsanen, P., Wennerberg, K. & Aittokallio,
T. Making sense of large-scale kinase inhibitor
bioactivity data sets: a comparative and integra-
tive analysis. Journal Of Chemical Information
And Modeling .54, 735-743 (2014)
[31] Knox, C., Wilson, M., Klinger, C., Franklin, M.,
Oler, E., Wilson, A., Pon, A., Cox, J., Chin,
N., Strawbridge, S. & Others DrugBank 6.0:
the DrugBank knowledgebase for 2024. Nucleic
Acids Research .52, D1265-D1275 (2024)
[32] Hurst, A., Lerer, A., Goucher, A., Perelman, A.,
Ramesh, A., Clark, A., Ostrow, A., Welihinda,
A., Hayes, A., Radford, A. & Others Gpt-4o
system card. ArXiv Preprint ArXiv:2410.21276 .
(2024)
[33] Guo, D., Yang, D., Zhang, H., Song, J., Zhang,
R., Xu, R., Zhu, Q., Ma, S., Wang, P., Bi,
X. & Others Deepseek-r1: Incentivizing reason-
ing capability in llms via reinforcement learning.
ArXiv Preprint ArXiv:2501.12948 . (2025)
[34] Chaslot, G. Monte-carlo tree search. (2010)
[35] Yuan, S., Chan, H. & Hu, Z. Using PyMOL
as a platform for computational drug design.
Wiley Interdisciplinary Reviews: Computational
Molecular Science .7, e1298 (2017)
[36] Chen, Y., Yan, L., Sun, W., Ma, X., Zhang,
Y., Wang, S., Yin, D., Yang, Y. & Mao,
J. Improving Retrieval-Augmented Generation
through Multi-Agent Reinforcement Learning.
ArXiv Preprint ArXiv:2501.15228 . (2025)
[37] Edwards, C., Lu, Z., Hajiramezanali, E., Bian-
calani, T., Ji, H. & Scalia, G. MolCap-Arena:
A Comprehensive Captioning Benchmark on
Language-Enhanced Molecular Property Predic-
tion. ArXiv Preprint ArXiv:2411.00737 . (2024)
[38] Zheng, Y., Koh, H., Yang, M., Li, L., May, L.,
Webb, G., Pan, S. & Church, G. Large lan-
guage models in drug discovery and develop-
ment: From disease mechanisms to clinical tri-
als.ArXiv Preprint ArXiv:2409.04481 . (2024)
[39] Yang, Z., Shao, W., Matsuda, Y. & Song, L.
iResNetDM: An interpretable deep learning ap-
proach for four types of DNA methylation mod-
ification prediction. Computational And Struc-
tural Biotechnology Journal .23pp. 4214-4221
(2024)[40] Team, Q. Qwen2 technical report. ArXiv
Preprint ArXiv:2412.15115 . (2024)
11

A Appendix
A.1 Dataset Construction
Experimental Dataset. We first extracted all molecules from the original dataset and computed both
Tanimoto similarity and cosine similarity between each pair. For each molecule, we selected the top 10 most
similar molecules based on each metric, merged the results, and removed duplicates to form the candidate
molecule set Mcm. Each unique query molecule paired with its corresponding Mcmconstitutes one problem
instance. These problem instances were further filtered according to the following criteria:
•For each query molecule, the number of associated interacting proteins must be between 2 and 10.
•For each candidate molecule in Mcm, the number of associated interacting proteins must be between 2
and 4.
•The total number of candidate molecules per query must not exceed 15.
Table 3: Dataset Statistics Summary
Dataset Processed Points All Proteins Ground Truth Unique Molecules All Molecules
DrugBank 788 22 508 1595 1304 7717
KIBA 626 23 849 1664 752 6219
We then extracted all proteins that interact with any molecule in Mcm, denoted as Pcp(as mentioned
in Section 2.4). Additionally, we collected all proteins that directly interact with the query molecule. The
intersection of these two protein sets was defined as the ground truth set. The ground truth set must satisfy
the following constraints:
•Its size must be between 1 and 5.
•Its size must not exceed 70% of the size of the candidate protein set.
Baseline Dataset. This dataset is derived from the Experimental Dataset, with the following modifica-
tions:
•Only the query molecules, the candidate protein set Pcp, and the ground truth set are retained.
•All candidate molecules Mcmare removed.
Table 4: Baseline Dataset Statistics
Dataset Ground Truth All Proteins Ratio (%)
DB 1595 14 654 10 .88
KIBA 1664 10 593 15 .71
Deep Learning Dataset. From the original dataset, we extracted all data instances involving the query
molecules from the Experimental Dataset to form the test set. The remaining data were used as the training
set.
A.2 Case Study
We manually selected a molecular-protein interaction with the highest self-consistency score that has never
been previously reported: Equol (DrugBank ID: DB11674) and CXC chemokine receptor 3 (CXCR3, PDB
ID: 8K2W). The binding affinity predicted by AutoDock Vina was -8.4 kcal/mol, indicating a strong potential
interaction between the two. Visualization using PyMOL [35] revealed that Equol can bind within one of
12

(a)
(b)
Figure 2: Number of rollouts vs Recall score on (a) DrugBank and (b) KIBA dataset
13

(a)
(b)
Figure 3: Number of tokens vs Recall score on (a) DrugBank and (b) KIBA dataset
14

Table 5: Class Distribution Statistics
Dataset Split Negative Positive Total
DrugBankTrain 14,787 12,428 27,215
Test 1,960 4,111 6,071
KIBATrain 62,553 17,350 79,903
Test 31,643 4,804 36,447
the binding pockets of CXCR3 and form hydrogen bonds, as evidenced by the continuous red dots in the
lower-right corner of Appendix Figure 1(b). This observation is consistent with the reasoning generated by
the large language model during the molecule analysis, protein selection, and absolute reward calculation
stages, thereby validating the effectiveness of our framework.
(a)
 (b)
Figure 4: Protein-ligand docking results between (a) global binding site overview and (b) detailed interaction
view, computed by AutoDock Vina with default scoring function and visualized using PyMOL.
15

Figure 5: Answers generated by the model
16