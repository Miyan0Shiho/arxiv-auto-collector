# Privacy-Enhancing Paradigms within Federated Multi-Agent Systems

**Authors**: Zitong Shi, Guancheng Wan, Wenke Huang, Guibin Zhang, Jiawei Shao, Mang Ye, Carl Yang

**Published**: 2025-03-11 08:38:45

**PDF URL**: [http://arxiv.org/pdf/2503.08175v1](http://arxiv.org/pdf/2503.08175v1)

## Abstract
LLM-based Multi-Agent Systems (MAS) have proven highly effective in solving
complex problems by integrating multiple agents, each performing different
roles. However, in sensitive domains, they face emerging privacy protection
challenges. In this paper, we introduce the concept of Federated MAS,
highlighting the fundamental differences between Federated MAS and traditional
FL. We then identify key challenges in developing Federated MAS, including: 1)
heterogeneous privacy protocols among agents, 2) structural differences in
multi-party conversations, and 3) dynamic conversational network structures. To
address these challenges, we propose Embedded Privacy-Enhancing Agents
(EPEAgent), an innovative solution that integrates seamlessly into the
Retrieval-Augmented Generation (RAG) phase and the context retrieval stage.
This solution minimizes data flows, ensuring that only task-relevant,
agent-specific information is shared. Additionally, we design and generate a
comprehensive dataset to evaluate the proposed paradigm. Extensive experiments
demonstrate that EPEAgent effectively enhances privacy protection while
maintaining strong system performance. The code will be availiable at
https://github.com/ZitongShi/EPEAgent

## Full Text


<!-- PDF content starts -->

Privacy-Enhancing Paradigms within Federated Multi-Agent Systems
Zitong Shi1†Guancheng Wan1†Wenke Huang1†Guibin Zhang2
Jiawei Shao3Mang Ye1Carl Yang4
1National Engineering Research Center for Multimedia Software,
School of Computer Science, Wuhan University, China
2National University of Singapore, Singapore
3Institute of Artificial Intelligence (TeleAI), China
4Department of Computer Science, Emory University, USA
Abstract
LLM-based Multi-Agent Systems (MAS) have
proven highly effective in solving complex problems
by integrating multiple agents, each performing dif-
ferent roles. However, in sensitive domains, they face
emerging privacy protection challenges. In this paper,
we introduce the concept of Federated MAS , high-
lighting the fundamental differences between Fed-
erated MAS and traditional FL. We then identify key
challenges in developing Federated MAS, including:
1) heterogeneous privacy protocols among agents, 2)
structural differences in multi-party conversations,
and 3) dynamic conversational network structures.
To address these challenges, we propose Embedded
Privacy-Enhancing Agents (EPEAgents ), an in-
novative solution that integrates seamlessly into the
Retrieval-Augmented Generation (RAG) phase and
the context retrieval stage. This solution minimizes
data flows, ensuring that only task-relevant, agent-
specific information is shared. Additionally, we
design and generate a comprehensive dataset to eval-
uate the proposed paradigm. Extensive experiments
demonstrate that EPEAgents effectively enhances
privacy protection while maintaining strong system
performance. The code will be availiable at
https://github.com/ZitongShi/EPEAgent
1 Introduction
Large Language Models (LLMs) have made signifi-
cant advancements in natural language processing, lead-
ing to breakthroughs in a wide range of applications
(Vaswani, 2017; Devlin, 2018). Recent research has
demonstrated that integrating LLM-based agents into
collaborative teams can outperform individual agents
in solving complex problems. These systems are re-
ferred to as multi-agent systems (MAS) . Within this
framework, agents assume different roles or engage in
Preprint. Under review.
Figure 1: Problem illustration . We describe the challenges of
privacy protection in MAS: I)Predefined privacy settings fail to
accommodate the heterogeneous privacy requirements of different
agents; II)Some protection methods compromise context awareness;
III)Complex protection architectures are unable to adapt to the
dynamic collaboration networks inherent in MAS.
debate-like interactions to accomplish tasks, resulting in
superior performance compared to a single agent (Hong
et al., 2023; Chen et al., 2023b; Richards et al., 2023).
However, most existing studies predominantly focus
on enhancing collaboration to improve MAS perfor-
mance, often neglecting critical privacy concerns (Wang
et al., 2025; Du et al., 2023). This issue becomes espe-
cially urgent in sensitive domains such as finance (Feng
et al., 2023; Xiao et al., 2024) and healthcare (Kim et al.,
2024; Li et al., 2024). The need for privacy-preserving
multi-party collaboration naturally leads us to extend
MAS into Federated Multi-Agent Systems (Federated
MAS) , where agents cooperate without directly sharing
confidential information. However, Federated MAS dif-
fers fundamentally from FL in several key aspects: (1)
FL aims to train globally shared models, while Federated
MAS focuses on real-time multi-agent collaboration. (2)
FL exchanges information indirectly through model up-
dates, whereas Federated MAS relies on task allocation
and agent communication. (3) FL primarily protects
1arXiv:2503.08175v1  [cs.AI]  11 Mar 2025

training data, whereas Federated MAS must safeguard
privacy dynamically throughout task execution and con-
versations.
Given the significant differences, we identify the key re-
search challenges in developing Federated Multi-Agent
Systems (Federated MAS), as illustrated in Fig. 1: I)
Heteroge neous Privacy Protocols : Different agents
may have varying requirements for data sharing and
privacy protection, requiring that only task-relevant in-
formation is shared among the corresponding agents. II)
Contextual Structure Variations : Some methods as-
sume a structured data format in the Memory Bank and
use differential privacy for protection. However, this as-
sumption does not always hold in practice (Dwork, 2006;
Kasiviswanathan et al., 2011). III) Dynamic Network
Structure : The network structure of MAS is dynamic,
making privacy protection methods that are overly com-
plex or require predefined structures unsuitable. PRAG
(Zyskind et al., 2023) enhances privacy protection dur-
ing the Retrieval-Augmented Generation (RAG) phase
by employing Multi-Party Computation (Yao, 1982) and
Inverted File approximation search protocols. However,
it is limited to the RAG phase and cannot dynamically
adapt to agent heterogeneity. Furthermore, it struggles
with extracting task-relevant information from memory
banks, highlighting its lack of context-awareness.
Some methods (Wu et al., 2023b; Gohari et al., 2023;
Kossek and Stefanovic, 2024) partition context exam-
ples to construct prompt inputs or employ techniques
such as differential privacy and homomorphic encryp-
tion to protect privacy. However, these approaches often
suffer from overly stringent privacy protection mecha-
nisms and high computational complexity, which makes
it challenging to ensure system utility effectively (Wang
et al., 2021; Nagar et al., 2021; Chen et al., 2023a). To
balance performance with privacy requirements, the sys-
tem must meet three key conditions, as highlighted by I),
II)andIII)(Zhou et al., 2024; Wang et al., 2024; Jiang
et al., 2024). This raises an important question: How
can we design Federated MAS that satisfies the specific
privacy needs of different agents, ensures stable task
performance, and avoids excessive complexity?
Given that the fine-tuning approaches of traditional
FL require excessive computing resources and manual
strategies for LLM-based agents (Al-Rubaie and Chang,
2019; Du and Ding, 2021), we shift our focus to the
flexible and dynamic nature of agents. In this paper, we
propose embedded privacy- enhancing agents , referredto as EPEAgents . This approach deploys a privacy-
enhanced agent on a trusted server, with its functionality
embedded into the RAG and context retrieval stages of
the MAS. Specifically, the message streams received
by each agent do not consist of raw data but are instead
task-relevant information filtered by EPEAgents . In
the system’s initial phase, each agent is required to
provide a self-description, outlining its responsibilities
and tasks within the MAS. This step allows EPEAgents
to understand the roles of each agent, enabling it to
dynamically plan task-relevant and agent-specific
messages during the RAG and context retrieval phases.
Subsequently, each agent can access task-relevant data
tailored to its specific responsibilities.
To evaluate whether EPEAgents maintains system per-
formance while ensuring privacy, we conducted exper-
iments with conversational agents. These experiments
included four types of tasks in the financial and medi-
cal domains, featuring both multiple-choice questions
(MCQs) and open-ended questions (OEQs). The ques-
tions were designed around user profiles, incorporating
details such as financial habits and health conditions.
Since real profiles were unavailable, we generated 25
synthetic profiles using GPT-o1 , ensuring they reflected
real-world distributions. The experiments utilized back-
bone models including Gemini-1.5-pro ,Gemini-1.5 ,
Claude-3.5 ,GPT-o1 ,GPT-4o , and GPT-3.5-turbo
(Team et al., 2023; Achiam et al., 2023). For ques-
tion generation, we followed a three-step process: initial
generation with GPT-o1 , review and cross-validation by
other models, and final confirmation through majority
voting or manual inspection. Our principal contributions
are summarized as follows:
•Concept Proposal : We introduce the Federated
MAS , addressing the emerging privacy needs of MAS,
and highlight the fundamental differences between
Federated Learning and Federated MAS.
•Privacy Challenges : We summarize the key chal-
lenges in developing Federated MAS, specifically I),
II), and III). These challenges serve as a framework
for designing privacy-preserving paradigms.
•Critical Evaluation : We critically evaluate existing
privacy-preserving methods in Federated MAS. Most
approaches rely on static models, which are inade-
quate for adapting to the dynamic topologies charac-
teristic.
•Embedded Privacy Enhancement : We propose
EPEAgents , a simple, user-friendly privacy protec-
2

tion mechanism. Designed to be embedded and
lightweight, EPEAgents adapts seamlessly to dynam-
ically changing network topologies. It demonstrates
minimal impact on system performance while achiev-
ing privacy protection effectiveness of up to 97.62%.
•Federated MAS Evaluation : We synthesized many
data in the financial and medical domains, which con-
form to real-world distributions. Additionally, we de-
veloped a comprehensive set of multiple-choice ques-
tions and open-ended contextual tasks, providing a
robust approach for evaluating both system perfor-
mance and privacy.
2 Related Work
2.1 Federated Learning
Federated Learning (FL), as a distributed privacy-
preserving learning paradigm, has been applied across
various domains. In computer vision, FL is widely used
for medical image processing, image classification, and
face recognition (Liu et al., 2021; Meng et al., 2022).
In graph learning, FL supports applications such as
recommendation systems and biochemical property
prediction, enabling collaborative training without
exposing sensitive data (Wu et al., 2020; Li et al., 2021;
Wu et al., 2021). In natural language processing (NLP),
the federated mechanism has been applied to machine
translation, speech recognition, and multi-agent systems
(MAS) (Deng et al., 2024; Cheng et al., 2023). However,
privacy-focused studies in MAS are relatively scarce,
and most existing approaches (Ying et al., 2023; Pan
et al., 2024) fail to simultaneously satisfy I),II),
andIII). In contrast, EPEAgents is lightweight and
flexible, and this paper provides extensive experiments
to demonstrate its performance and privacy protection
capabilities.
2.2 Privacy within MAS
PPARCA (Ying et al., 2023) identifies attackers through
outlier detection and robustness theory, excluding their
information from participating in state updates. The
Node Decomposition Mechanism (Wang et al., 2021)
decomposes an agent into multiple sub-agents and uti-
lizes homomorphic encryption to ensure that informa-
tion exchange between non-homologous sub-agents is
encrypted. Other methods (Panda et al., 2023; Huo
et al., 2024; Kossek and Stefanovic, 2024) attempt to
achieve privacy protection through differential privacy
or context partitioning. However, these approaches are
effective only in specific scenarios. The protection levelof differential privacy is often difficult to control, and
algorithms with high computational complexity are un-
suitable for MAS (Zheng et al., 2023; Wu et al., 2023a;
Shinn et al., 2023; Wang et al.). In contrast, EPEAgents
is lightweight, adaptable to diverse scenarios, and does
not require extensive predefined protection rules.
3 Preliminary
Notations . Consider a MAS consisting of Nagents.
We denote the set of agents as: C={C1, C2, . . . , C N}.
During the t-th operational round of the system, we
denote the set of communicating agents as Ct⊆ C .
Thei-thagent is represented as Ct
i, while the privacy-
enhanced agent is denoted by Ct
P. Each agent is defined
as:
Ct
i={Backbonet
i,Rolet
i,MemoryBankt
i}. (1)
where Backbonet
irepresents the language model used
byCi,Rolet
idenotes the role played by Ciin the MAS,
andMemoryBankt
irefers to the memory storage of Ciat
thet-th round, which contains task-relevant information
gathered and processed during the operation. CAis
deployed on a server with a unique characteristic. Its
MemoryBanktrepresents the server’s memory storage
at the beginning of the t-th interaction round and is
defined as the aggregate of the MemoryBanktfrom all
agents.
During the same interaction round, we denote the com-
munication from Ct
itoCt
jaset,S
ij, referred to as a spatial
edge, where all communications are directed edges. This
edge includes task-related content and may also include
additional associated operations in our framework, such
as the self-description sent from CitoCA. The set of
spatial edges is defined as:
Et,S={et,S
ij|Ct
iS− →Ct
j,∀i, j∈ {1, . . . , N }, i̸=j}.
(2)
In adjacent rounds, we define the communication from
Ct−1
itoCt
jaseT
ij, referred to as a temporal edge , where
all communications are also directed edges. This edge
typically contains only task-related content. Similarly,
the set of temporal edges is defined as:
ET={eT
ij|Ct−1
iT− →Ct
j,∀i, j∈ {1, . . . , N }, i̸=j}.
(3)
Communication in MAS . Communication in MAS
is defined from the perspectives of spatial edges and
temporal edges. As described above, in any t-th round,
3

Algorithm 1: Execution Workflow in Conven-
tional MAS.
Input: TaskT, prompt P, Communication
rounds N, associated network
GT,Gt∈T,S
Output: The final answer AT
1fort= 1,2,···,|T |do
2 forn= 1 toNin parallel do
3 At(Ci)←
fθ 
T,Pi,A(t−1)(Cj),Retrievalt
i
// Benefit from temporal graph GT.
4 At(Ci)←
fθ 
T,Pi,At(Cj),Retrievalt
i
// Benefit from spatial graph Gt,S.
5 end
6At←
SumAnswer 
At(C1),At(C2), . . . ,At(CN)
// In some problem-solving scenarios, it
may be based on majority voting; in
conversational agent systems, it
could be the output of a summarizer
agent.
7end
8return AT
Et,Srepresents directed edges, which, together with
Ct, form a directed acyclic graph Gt,S={Ct,Et,S}.
Similarly, in the temporal domain, the directed acyclic
graph is represented as GT={Ct∈T,ET}. The
intermediate or final answer obtained by Ciis denoted
asA(Ci), formalized as:
At(Ci)∼fθ 
T,Pi, A(Cj),Retrievalt
i
(4)
where Trepresents the task, Piis the prompt, which
typically specifies the role of Ci.A(Cj)represents the
output of the parent node Cjin the spatial edges or
temporal edges. Retrievalt
irefers to the knowledge
retrieved by Ciduring the t-th round, sourced from
the shared knowledge pool DataBase and the server’s
memory storage MemoryBankt.
Problem Formulation . This paper explores the chal-
lenge of ensuring privacy protection in MAS while pre-
serving system performance. At the beginning of the
first interaction round, all agents receive the task Talong
with a prompt specifying their respective Role . In the
general framework, agents retrieve task-relevant infor-
mation from the shared knowledge pool and generate
Figure 2: Two sample instances from the evaluation process are
presented. The red flow represents the traditional pipeline without
security screening, while the blue flow illustrates the pipeline filtered
through EPEAgents .
intermediate outputs for their respective queries based
on their assigned roles. The details of their interactions
are stored in the server’s memory bank, which can later
be used to retrieve task-relevant information when nec-
essary to enhance response quality. However, although
this pipeline is straightforward, it poses significant risks
of privacy leakage.
We represent user information as U={u1, u2, . . . , u U},
where Udenotes the total number of users. Each gen-
erated user profile consists of 11 fields, denoted as Fu.
Each multiple-choice question has a unique correct op-
tion, denoted as Ocorrect . A result is considered the cor-
rect answer for the MAS if and only if AT=Ocorrect .
Contextual open-ended questions used for performance
evaluation include two entries: the corresponding field,
denoted as Fq, and the question itself. In contrast, ques-
tions used for privacy evaluation include an additional
entry, the label, which identifies the specific agent re-
sponsible for answering the question. For further details,
please refer to Sec. 4.4.
4 Methodology
4.1 Overview
In this section, we introduce the Embedded Privacy-
Enhancing Agents ( EPEAgents ). This method acts as
an intermediary agent deployed on the server and inte-
grates seamlessly into various data flows within MAS,
4

such as the RAG phase and the memory bank retrieval
stage. The overall framework of EPEAgents is shown
in Fig. 3. At the beginning of the system operation, the
taskTis distributed to all agents. Additionally, local
agents send self-descriptions to CA. Based on these self-
descriptions and user profiles, CAsends the first batch of
task-relevant and agent-specific messages to the remain-
ing agents. In subsequent data flows, local agents can
only access the second-hand secure filtered information
provided by CA.
4.2 Privacy Enhanced Agent Design
Motivation . Research on privacy protection in MAS
remains limited, and there is a lack of architectures that
can adapt to general scenarios. Some methods are de-
signed specifically for certain scenarios, resulting in
limited applicability (Wang et al., 2021; Cheng et al.,
2023; Chan et al., 2023; Deng et al., 2024). Others
involve high computational costs or complex architec-
tures, making them unsuitable for dynamic topological
networks (Nagar et al., 2021; Ying et al., 2023; Cheng
et al., 2024; Du et al., 2024). Inspired by the federated
mechanism, we isolate direct communication between
local agents and during their retrieval processes. Data
flows reaching any local agent are designed to ensure
maximum trustworthiness and security.
Minimization of User Profiles . At the very begin-
ning of system operation, each local agent sends a self-
description to CA. This allows CAto associate different
entries of user data with the corresponding roles of local
agents. For a specific user uj,Cican only access the
content of Futhat matches its role.


C(1)
AMu
min− − − → C(1)
i,ifRole i∼Fu,
C(1)
A̸→C(1)
i ,ifRole i≁Fu,(5)
Here,Mu
minrepresents the minimized user profile infor-
mation. It is sent from CAtoCionly if the role and
Fumatch, i.e., Role i∼Fu. Otherwise, it is not sent.
This scenario can be extended to cases where the shared
knowledge pool is not user profiles but databases of
patient records from different hospitals. In such cases,
this step can be augmented with search protocols to
retrieve relevant information from the databases. How-
ever, this paper focuses solely on the scenario of user
profiles.
Dynamic Permission Elevation .CAcannot always
accurately determine whether Fu∼Role i, as theremay be subtle differences. For example, in a conver-
sational agent system, a medication delivery process
may require the user’s home address. However, CA
often cannot infer this requirement directly from the
taskT. In such cases, a trusted third party can initiate
a permission upgrade request to the user, allowing the
user to confirm whether to grant access. This upgrade
mechanism bypasses the forwarding by CAand directly
communicates with the user, ensuring the task proceeds
smoothly.
Minimization of Reasoning progress . In addition
to user profiles, some intermediate answers generated
by local agents also need to be filtered and forwarded
through CA. Malicious local agents may attempt to
disguise themselves as summarizers in the system.
These agents are often located at the terminal nodes
ofGS, allowing them to access more information than
others. Ignoring this process could result in serious
privacy breaches. Fig. 2 illustrates a real test case where,
without the information filtering by CA, the terminal
agent directly revealed sensitive user information, such
as their name and cholesterol level.
4.3 MAS Architecture Design
In this section, we outline the EPEAgents , with a pri-
mary focus on the design of local agents. Improving sys-
tem performance is beyond the scope of this study. We
constructed a simple 3+narchitecture to evaluate various
metrics, where 3andnrepresent the number of local
agents and CA, respectively. For the financial scenario,
the three local agents are defined as follows:
•Market Data Agent : Responsible for aggregating
and filtering relevant market data to provide timely
insights on evolving market conditions.
•Risk Assessment Agent : Responsible for analyzing
the market data alongside user profiles to evaluate
investment risks and determine the appropriateness
of various asset allocation strategies.
•Transaction Execution Agent : Responsible for
integrating insights from the other agents and
executing final trade decisions that align with user
preferences and market dynamics.
For the medical scenario, the three local agents are de-
fined as follows:
•Diagnosis Agent : Responsible for providing an inter-
mediate medical diagnosis perspective by analyzing
patient symptoms, medical history, and diagnostic test
results.
5

Figure 3: The architecture illustration of EPEAgents .
•Treatment Recommendation Agent : Responsible
for evaluating potential treatment options by integrat-
ing clinical guidelines and patient-specific data to sug-
gest optimal therapeutic approaches.
•Medication Management Agent : Responsible for
consolidating insights from the Diagnosis and Treat-
ment Recommendation Agents and executing the final
treatment plan, including medication selection and
dosage management, while ensuring patient safety
and efficacy.
CAis deployed on the server and is responsible for
receiving intermediate responses and the complete user
profile. It filters and sanitizes the data by removing
or obfuscating fields that lack the specified aggregator
label, ensuring that only authorized information is
accessible. We assigned roles to the agents using
prompts, and a specific example is shown below:
4.4 Synthetic Data Design
In this section, we provide a detailed explanation of
the dataset generation process. Following (Bagdasarian
et al., 2024; Thaker et al., 2024), our dataset is cate-
gorized into three types: user profiles, multiple- choice
questions (MCQ), and contextual open-ended questions
(OEQ). Each category is further divided into two sce-
narios: financial and medical. The latter two types are
additionally split into subsets designed for evaluating
performance and privacy.
Generation of User Profiles . User profiles are central
to data generation, subsequent question construction,and experimental design. To facilitate question construc-
tion, we divide user profiles into several entries, each
associated with a specific field Fu. Each Fucorresponds
to a question domain Fq, which is crucial for designing
privacy evaluation questions.
The set of user profiles is U={u1, u2, . . . , u |U|}. We
define uiin the form of a tuple as:
ui=⟨entry ,field⟩, i∈ |U|. (6)
Here, entry denotes an item within the profile, which
can be further decomposed into multiple compo-
nents:
entry ={field ,value ,field ,label}. (7)
Thefield is one of these components and is explicitly
highlighted in Eq. (6) to enhance clarity in understand-
ing the subsequent formulas.
Generation of Question Datasets . The question gen-
eration process involves three steps: ❶GPT-o1 creates
an initial draft of questions; ❷multiple large models re-
generate answers and perform comparative analysis; ❸
manual review is conducted for verification and refine-
ment. Designing Multiple-Choice Questions (MCQ)
andOpen-Ended Questions (OEQ) to evaluate perfor-
mance is straightforward. We generated questions for
theFufields in the user profiles, creating 5MCQs for
each of the 6fields. Each MCQ includes four options,
with one correct answer. We then used Gemini-1.5 ,
Gemini-1.5-pro ,Claude-3.5 , and GPT-o1 to generate
6

answers for each question across all users. Disputed an-
swers were resolved by majority voting or manual delib-
eration. A question can be formalized as follows:
question =⟨field ,type,stem,answer ⟩,(8)
Here, type refers to the category of the question , indi-
cating whether it is an MCQ or an OEQ. A test sample
can be formalized as:
s=ui▷ ◁question (9)
Here, ▷ ◁denotes the association operation between a
useruiand a question . This operation maps a specific
entry from the user profile to the corresponding field
in the question, facilitating the construction of a sam-
ples=⟨entry ,field ,type,stem,answer ⟩. A similar
process was applied to the OEQ designed.
The label of user profiles is denoted as Lu, which
indicates the matching relationship with the three local
agents. This matching relationship is also generated by
a large language model, following a similar three-step
process to that used for generating MCQ. The three local
agents are numbered 1, 2, and 3. Taking the financial
scenario as an example, the investment goals entry
has a label Lu={1,2}, indicating that its information
can be shared with the Market Data Agent and the Risk
Assessment Agent. According to GPT-o1 , the reasoning
is as follows:
•The Market Data Agent requires the user’s investment
goals to provide market data aligned with those goals.
For instance, if the user prioritizes long-term wealth
accumulation orretirement savings , Agent 1 needs to
gather market trends, industry insights, or macroeco-
nomic indicators relevant to these objectives.
•Similarly, the Risk Assessment Agent needs invest-
ment goals to evaluate the user’s risk preferences. Dif-
ferent goals often imply varying levels of risk expo-
sure and investment horizons. For example, retirement
savings typically demands a balance between stability
and growth, whereas short-term speculation focuses
more on short-term volatility. Thus, this information
is crucial for the Risk Assessment Agent to provide
accurate risk analysis.
After labeling each entry, we designed privacy-
evaluating MEQ and OEQ. For MEQ, a fixed option,
Refuse to answer , was introduced as the correct re-
sponse. For OEQ, prompts were configured to ensure
that agents, when asked about unauthorized informa-
tion, reply with a standard statement: I do not have
Figure 4: An example prompt that defines the Diagnosis Agent’s
role and privacy-related constraints in our medical MAS.
the authority to access this information and
refuse to answer. Privacy-evaluating questions dif-
fer from performance-evaluating ones in key ways. The
former assigns the responder based on the label, whereas
the latter designates an agent to serve as the summarizer,
providing the final answer.
4.5 Discussion
In our approach, the privacy-preserving model on the
server, CA, leverages existing large models such as
GPT-o1 andGemini-1.5-pro . However, its primary
functionality is focused on data minimization and
acting as a forwarding agent. This suggests potential
avenues for future research, including the exploration of
more lightweight and specialized models to replace the
current architecture. Furthermore, the labels assigned to
the entries during architecture evaluation are generated
by LLMs. In real-world scenarios, however, these con-
ditions may depend more heavily on users’ subjective
preferences. This underscores the need for further inves-
tigation into practical benchmarks to better evaluate the
alignment of such labels with user expectations.
5 Experiment
We conducted detailed experiments with 21,750 samples
across five models in two domains, thoroughly evalu-
ating the performance and privacy effects of both the
7

Table 1: Utility and Privacy Comparison between the Baseline and EPEAgents . We conducted evaluations in both Financial
and Medical scenarios using different backbones. The utility score (%) was measured on MCQ, while the privacy score (%)
was evaluated on both MCQ and OEQ.
Financial Medical
Backbone Method MCQ OEQ MCQ OEQ
Utility(%) Privacy(%) Privacy(%) Utility(%) Privacy(%) Privacy(%)
Claude-3.5Baseline 86.28 13 .68 14 .29 84.69 12 .26 12 .32
EPEAgents 86.89↑0.61 85.64↑71.96 84.23↑69.94 85.59↑0.90 84.28↑72.02 85.34↑73.02
Baseline 95.12 15.89 23.53 89.83 14.57 14.73GPT-o1EPEAgents 96.61↑1.49 97.62↑81.73 96.31↑72.78 91.89↑2.06 95.43↑80.86 95.84↑81.11
Baseline 80.67 11 .24 12 .26 74.67 8 .73 10 .29GPT-4oEPEAgents 81.64↑0.97 75.27↑64.03 78.61↑66.35 75.38↑0.71 76.47↑67.74 79.94↑69.65
Baseline 70.35 12.38 6.34 68.57 7.89 4.27GPT-3.5-turboEPEAgents 69.82↓0.53 71.26↑58.88 61.67↑55.33 68.78↑0.21 69.37↑61.48 66.35↑62.08
Baseline 60.78 11 .68 11 .23 59.22 8 .23 5 .61Gemini-1.5EPEAgents 61.16↑0.38 55.69↑44.01 56.47↑45.24 58.76↓0.46 56.49↑48.26 58.54↑52.93
Baseline 68.25 13.33 18.22 62.72 10.57 6.22Gemini-1.5-proEPEAgents 68.74↑0.49 65.71↑52.38 58.45↑40.23 63.43↑0.71 67.28↑56.71 62.34↑56.12
baseline methods and EPEAgents .
5.1 Experimental Setup
Datasets and Tasks. Adhering to (Feng et al., 2023;
Wang et al., 2025), we evaluated the performance and
privacy of the models in the financial and medical sce-
narios. Our dataset is divided into three categories: user
profiles, multiple-choice questions, and open-ended con-
textual questions. The detailed generation process of
these categories is provided in Sec. 4.4.
Evaluation Metric. The structure of a test sample is
s=⟨entry ,field ,type,stem,answer ⟩. We denote
the answer obtained by MAS as ypredand the pre-defined
standard answer as ya. Due to the difficulty of standard-
izing reference answers for OEQ across large models,
as well as the challenges in controlling evaluation met-
rics, we primarily use MCQ to assess the utility of MAS
(Bagdasarian et al., 2024). The calculation method is as
follows:
Utility =P
|Stype|=MCQI(ya, ypred)
|Stype|=MCQ, (10)
whereI(ya, ypred)is an indicator function that returns 1
ifya=yMAS and 0 otherwise. Privacy evaluation takes
a more comprehensive approach, utilizing both MCQ
and OEQ. In the case of MCQ, a predefined option,
Refuse to answer , is included as the standard answer.
For OEQ, agents are guided through prompts containing
explicit instructions for their responses.

PrivacyMCQ =P
|Stype|=MCQI(ya, ypred)
|Stype|=MCQ,
PrivacyOEQ =P
|Stype|=OEQEM(ya, ypred)
|Stype|=OEQ,(11)
where EM(ya, ypred)is an exact match function that
returns 1 if the predicted answer ypredexactly matches
the reference answer ya, and 0 otherwise.
EM =(
1ifSpred=Sa
0otherwise(12)
5.2 Experiment Results
We adopt a 3+narchitecture for evaluation. In the main
experiment (Tab. 1), we fix nto 1 for evaluation. Addi-
tionally, we perform ablation studies by replacing the
backbone architectures of the entire MAS and specifi-
cally focusing on the backbone of the server-side CA.
We also investigate the impact of varying the num-
ber of privacy-preserving agents CAdeployed on the
server.
Performance Analysis. We observed a slight increase
in utility in most scenarios, while the Privacy scores
improved significantly across all scenarios. Interest-
ingly, GPT-o1 exhibited a significantly higher increase
in utility compared to other backbones. We attribute
this to the strong comprehension capabilities of GPT-o1 ,
which allows for more precise filtering of user profiles
and intermediate data flows. In contrast, models with
relatively weaker comprehension capabilities, such
asGemini-1.5 andGPT-3.5-turbo , exhibit a utility
8

Figure 5: Ablation Analysis of the number of CA. We used
Claude-3.5 and Gemini-1.5 as backbones in our experiments.
Please refer to Sec. 5.3 for additional analysis.
decline under certain scenarios due to their limited
ability to handle tasks effectively. However, even in
these cases, the improvement in Privacy remains highly
significant.
Additionally, we observed an entries difference in Pri-
vacy scores. Questions associated with certain entries,
such as annual income , which are widely recognized
as sensitive privacy information, tend to exhibit higher
privacy protection compared to other entries. This effect
is particularly prominent in high-performing models like
Claude andGPT-o1 . In contrast, this distinction is less
evident in lower-performing LLMs. For example, the
Privacy score of GPT-4o on the Baseline is comparable
to that of GPT-3.5-turbo .
5.3 Ablation analysis.
Different Backbones. A comparison of columns in
Tab. 1 reveals that the differences in Privacy scores
among various backbones in the Baseline are relatively
minor. For instance, even the high-performing GPT-o1
achieves a Privacy score of only 15.89 in the financial
scenario without the application of EPEAgents , which
is merely 3.51% higher than that of GPT-3.5-turbo .
However, when our architecture is applied, the im-
provement in Privacy scores becomes significantly more
pronounced for higher-performing LLMs. For exam-
ple, Claude-3.5 demonstrates a remarkable 71.96%
increase in Privacy scores, whereas Gemini-1.5 , be-
ing relatively less capable, achieves a more moderate
improvement of 44.01%.
Key Parameters. We conducted ablation studies on the
number of CAagents deployed on the server to analyze
how their workload distribution affects the overall per-
formance of the MAS. The results presented in Fig. 5
show that when lower-performing LLMs are used as the
backbone for CA, increasing nslightly improves the Pri-
vacy scores. However, this improvement becomes less
significant when higher-performing LLMs are used as
the backbone. For example, when Claude-3.5 is used
Figure 6: Ablation Analysis of the backbone of CA. We replaced
the backbone of CAwith GPT-o1 andGemini-1.5 as local agents
to study their impact on the privacy score of MAS. Please refer to
Sec. 5.3 for additional analysis.
as the backbone, the Privacy score tends to decrease as
nincreases. In contrast, with Gemini-1.5 , the Privacy
score can improve by as much as 6.29% at its peak.
Backbone of CA.WWe conduct ablation studies on the
server-side privacy-preserving agent’s backbone, focus-
ing on the two models with the best and worst perfor-
mance in Tab. 1: GPT-o1 andGemini-1.5 . The results
are presented in Fig. 6. Our findings highlight the crit-
ical role of the CAbackbone. Even when local agents
utilize a high-performing LLM such as GPT-o1 , main-
taining a high Privacy score becomes challenging if
theCAbackbone is suboptimal. For instance, when
the backbone of CAisGemini-1.5 , the Privacy score
drops to 58.67% despite local agents using GPT-o1 , rep-
resenting a 38.95% decrease from the original score. In
contrast, employing a strong LLM as the CAbackbone
enables the system to achieve substantial Privacy scores,
even when the local agents rely on less capable LLMs.
This observation indirectly validates the effectiveness of
EPEAgents .
6 Conclusion
In this work, we identified emerging privacy protec-
tion challenges in LLM-based MAS, particularly within
sensitive domains. We introduced the concept of Fed-
erated MAS, emphasizing its key distinctions from tra-
ditional FL. Addressing critical challenges such as het-
erogeneous privacy protocols, structural complexities in
multi-party conversations, and dynamic conversational
network structures, we proposed EPEAgents as a novel
solution. This method minimizes data flow by shar-
ing only task-relevant, agent-specific information and
integrates seamlessly into both the RAG and context
retrieval stages. Extensive experiments demonstrate
EPEAgents ’s potential in real-world applications, pro-
viding a robust approach to privacy-preserving multi-
agent collaboration. Looking ahead, we highlight the
9

importance of incorporating dynamic privacy-enhancing
techniques into MAS, particularly in high-stakes do-
mains where privacy and security are essential.
References
Josh Achiam, Steven Adler, Sandhini Agarwal, Lama Ahmad,
Ilge Akkaya, Florencia Leoni Aleman, Diogo Almeida, Janko
Altenschmidt, Sam Altman, Shyamal Anadkat, et al. 2023.
Gpt-4 technical report. arXiv preprint arXiv:2303.08774 .
Mohammad Al-Rubaie and J Morris Chang. 2019. Privacy-
preserving machine learning: Threats and solutions. IEEE
Security & Privacy , 17(2):49–58.
Eugene Bagdasarian, Ren Yi, Sahra Ghalebikesabi, Peter
Kairouz, Marco Gruteser, Sewoong Oh, Borja Balle, and
Daniel Ramage. 2024. Airgapagent: Protecting privacy-
conscious conversational agents. In Proceedings of the 2024
on ACM SIGSAC Conference on Computer and Communica-
tions Security , pages 3868–3882.
Chi-Min Chan, Weize Chen, Yusheng Su, Jianxuan Yu, Wei
Xue, Shanghang Zhang, Jie Fu, and Zhiyuan Liu. 2023. Chat-
eval: Towards better llm-based evaluators through multi-agent
debate. arXiv preprint arXiv:2308.07201 .
Bo Chen, Calvin Hawkins, Mustafa O Karabag, Cyrus Neary,
Matthew Hale, and Ufuk Topcu. 2023a. Differential privacy
in cooperative multiagent planning. In Uncertainty in Artifi-
cial Intelligence , pages 347–357. PMLR.
Weize Chen, Yusheng Su, Jingwei Zuo, Cheng Yang, Chenfei
Yuan, Chi-Min Chan, Heyang Yu, Yaxi Lu, Yi-Hsin Hung,
Chen Qian, et al. 2023b. Agentverse: Facilitating multi-
agent collaboration and exploring emergent behaviors. In
The Twelfth International Conference on Learning Represen-
tations .
Huqiang Cheng, Xiaofeng Liao, Huaqing Li, and Qingguo
L¨u. 2023. Dynamics-based algorithm-level privacy preser-
vation for push-sum average consensus. arXiv preprint
arXiv:2304.08018 .
Yuheng Cheng, Ceyao Zhang, Zhengwen Zhang, Xiangrui
Meng, Sirui Hong, Wenhao Li, Zihao Wang, Zekai Wang,
Feng Yin, Junhua Zhao, et al. 2024. Exploring large language
model based intelligent agents: Definitions, methods, and
prospects. arXiv preprint arXiv:2401.03428 .
Qingyun Deng, Kexin Liu, and Yinyan Zhang. 2024. Privacy-
preserving consensus of double-integrator multi-agent sys-
tems with input constraints. IEEE Transactions on Emerging
Topics in Computational Intelligence .
Jacob Devlin. 2018. Bert: Pre-training of deep bidirectional
transformers for language understanding. arXiv preprint
arXiv:1810.04805 .
Hung Du, Srikanth Thudumu, Rajesh Vasa, and Kon Mouza-
kis. 2024. A survey on context-aware multi-agent systems:
Techniques, challenges and future directions. arXiv preprint
arXiv:2402.01968 .Wei Du and Shifei Ding. 2021. A survey on multi-agent deep
reinforcement learning: from the perspective of challenges
and applications. Artificial Intelligence Review , 54(5):3215–
3238.
Yilun Du, Shuang Li, Antonio Torralba, Joshua B Tenenbaum,
and Igor Mordatch. 2023. Improving factuality and reasoning
in language models through multiagent debate. arXiv preprint
arXiv:2305.14325 .
Cynthia Dwork. 2006. Differential privacy. In International
colloquium on automata, languages, and programming , pages
1–12. Springer.
Shangbin Feng, Weijia Shi, Yuyang Bai, Vidhisha Balachan-
dran, Tianxing He, and Yulia Tsvetkov. 2023. Knowledge
card: Filling llms’ knowledge gaps with plug-in specialized
language models. arXiv .
Parham Gohari, Matthew Hale, and Ufuk Topcu. 2023.
Privacy-engineered value decomposition networks for co-
operative multi-agent reinforcement learning. In 2023 62nd
IEEE Conference on Decision and Control (CDC) , pages
8038–8044. IEEE.
Sirui Hong, Xiawu Zheng, Jonathan Chen, Yuheng Cheng,
Jinlin Wang, Ceyao Zhang, Zili Wang, Steven Ka Shing Yau,
Zijuan Lin, Liyang Zhou, et al. 2023. Metagpt: Meta pro-
gramming for multi-agent collaborative framework. arXiv
preprint arXiv:2308.00352 .
Xiang Huo, Hao Huang, Katherine R Davis, H Vincent Poor,
and Mingxi Liu. 2024. A review of scalable and privacy-
preserving multi-agent frameworks for distributed energy
resource control. arXiv e-prints , pages arXiv–2409.
Kemou Jiang, Xuan Cai, Zhiyong Cui, Aoyong Li, Yilong
Ren, Haiyang Yu, Hao Yang, Daocheng Fu, Licheng Wen,
and Pinlong Cai. 2024. Koma: Knowledge-driven multi-
agent framework for autonomous driving with large language
models. IEEE Transactions on Intelligent Vehicles .
Shiva Prasad Kasiviswanathan, Homin K Lee, Kobbi Nissim,
Sofya Raskhodnikova, and Adam Smith. 2011. What can
we learn privately? SIAM Journal on Computing , 40(3):793–
826.
Yubin Kim, Chanwoo Park, Hyewon Jeong, Yik Siu Chan,
Xuhai Xu, Daniel McDuff, Hyeonhoon Lee, Marzyeh Ghas-
semi, Cynthia Breazeal, and Hae Won Park. 2024. Mda-
gents: An adaptive collaboration of llms for medical decision-
making. In The Thirty-eighth Annual Conference on Neural
Information Processing Systems .
Magdalena Kossek and Margareta Stefanovic. 2024. Survey
of recent results in privacy-preserving mechanisms for multi-
agent systems. Journal of Intelligent & Robotic Systems ,
110(3):129.
Binxu Li, Tiankai Yan, Yuanting Pan, Jie Luo, Ruiyang Ji,
Jiayuan Ding, Zhe Xu, Shilong Liu, Haoyu Dong, Zihao Lin,
et al. 2024. Mmedagent: Learning to use medical tools with
multi-modal agent. arXiv preprint arXiv:2407.02483 .
Qinbin Li, Zeyi Wen, Zhaomin Wu, Sixu Hu, Naibo Wang,
Yuan Li, Xu Liu, and Bingsheng He. 2021. A survey on
10

federated learning systems: Vision, hype and reality for data
privacy and protection. IEEE Transactions on Knowledge
and Data Engineering , 35(4):3347–3366.
Quande Liu, Cheng Chen, Jing Qin, Qi Dou, and Pheng-Ann
Heng. 2021. Feddg: Federated domain generalization on
medical image segmentation via episodic learning in con-
tinuous frequency space. In Proceedings of the IEEE/CVF
conference on computer vision and pattern recognition , pages
1013–1023.
Qiang Meng, Feng Zhou, Hainan Ren, Tianshu Feng,
Guochao Liu, and Yuanqing Lin. 2022. Improving feder-
ated learning face recognition via privacy-agnostic clusters.
arXiv preprint arXiv:2201.12467 .
Anudit Nagar, Cuong Tran, and Ferdinando Fioretto. 2021. A
privacy-preserving and trustable multi-agent learning frame-
work. arXiv preprint arXiv:2106.01242 .
Longshuo Pan, Jian Wang, Hongyong Yang, Chuangchuang
Zhang, and Li Liu. 2024. Privacy-preserving bipartite con-
sensus of discrete multi-agent systems under event-triggered
protocol. In Chinese Intelligent Systems Conference , pages
488–496. Springer.
Ashwinee Panda, Tong Wu, Jiachen Wang, and Prateek Mit-
tal. 2023. Differentially private in-context learning. In The
61st Annual Meeting Of The Association For Computational
Linguistics .
Toran Bruce Richards et al. 2023. Auto-gpt: An autonomous
gpt-4 experiment. Original-date , 21:07Z.
Noah Shinn, Beck Labash, and Ashwin Gopinath. 2023. Re-
flexion: an autonomous agent with dynamic memory and
self-reflection. arXiv preprint arXiv:2303.11366 , 2(5):9.
Gemini Team, Rohan Anil, Sebastian Borgeaud, Jean-
Baptiste Alayrac, Jiahui Yu, Radu Soricut, Johan Schalkwyk,
Andrew M Dai, Anja Hauth, Katie Millican, et al. 2023. Gem-
ini: a family of highly capable multimodal models. arXiv
preprint arXiv:2312.11805 .
Pratiksha Thaker, Yash Maurya, Shengyuan Hu, Zhi-
wei Steven Wu, and Virginia Smith. 2024. Guardrail baselines
for unlearning in llms. arXiv .
A Vaswani. 2017. Attention is all you need. Advances in
Neural Information Processing Systems .
Haotian Wang, Xiyuan Du, Weijiang Yu, Qianglong Chen,
Kun Zhu, Zheng Chu, Lian Yan, and Yi Guan. 2025. Learn-
ing to break: Knowledge-enhanced reasoning in multi-agent
debate system. Neurocomputing , 618:129063.
Qian Wang, Tianyu Wang, Qinbin Li, Jingsheng Liang, and
Bingsheng He. 2024. Megaagent: A practical framework
for autonomous cooperation in large-scale llm agent systems.
arXiv .
Yaqi Wang, Jianquan Lu, Wei Xing Zheng, and Kaibo Shi.
2021. Privacy-preserving consensus for multi-agent systems
via node decomposition strategy. IEEE Transactions on Cir-
cuits and Systems I: Regular Papers , 68(8):3474–3484.Z Wang, S Mao, W Wu, T Ge, F Wei, and H Ji. Unleashing
cognitive synergy in large language models: A task-solving
agent through multi-persona selfcollaboration. arxiv 2023.
arXiv preprint arXiv:2307.05300 .
Chuhan Wu, Fangzhao Wu, Yang Cao, Yongfeng Huang,
and Xing Xie. 2021. Fedgnn: Federated graph neural net-
work for privacy-preserving recommendation. arXiv preprint
arXiv:2102.04925 .
Qingyun Wu, Gagan Bansal, Jieyu Zhang, Yiran Wu,
Shaokun Zhang, Erkang Zhu, Beibin Li, Li Jiang, Xiaoyun
Zhang, and Chi Wang. 2023a. Autogen: Enabling next-
gen llm applications via multi-agent conversation framework.
arXiv preprint arXiv:2308.08155 .
Tong Wu, Ashwinee Panda, Jiachen T Wang, and Prateek
Mittal. 2023b. Privacy-preserving in-context learning for
large language models. arXiv .
Zonghan Wu, Shirui Pan, Fengwen Chen, Guodong Long,
Chengqi Zhang, and S Yu Philip. 2020. A comprehensive sur-
vey on graph neural networks. IEEE transactions on neural
networks and learning systems , 32(1):4–24.
Yijia Xiao, Edward Sun, Di Luo, and Wei Wang. 2024.
Tradingagents: Multi-agents llm financial trading framework.
arXiv preprint arXiv:2412.20138 .
Andrew C Yao. 1982. Protocols for secure computations. In
23rd annual symposium on foundations of computer science
(sfcs 1982) , pages 160–164. IEEE.
Chenduo Ying, Ning Zheng, Yiming Wu, Ming Xu, and
Wen-An Zhang. 2023. Privacy-preserving adaptive resilient
consensus for multiagent systems under cyberattacks. IEEE
Transactions on Industrial Informatics , 20(2):1630–1640.
Chuanyang Zheng, Zhengying Liu, Enze Xie, Zhenguo
Li, and Yu Li. 2023. Progressive-hint prompting im-
proves reasoning in large language models. arXiv preprint
arXiv:2304.09797 .
Jingwen Zhou, Qinghua Lu, Jieshan Chen, Liming Zhu, Xi-
wei Xu, Zhenchang Xing, and Stefan Harrer. 2024. A tax-
onomy of architecture options for foundation model-based
agents: Analysis and decision model. arXiv .
Guy Zyskind, Tobin South, and Alex Pentland. 2023. Don’t
forget private retrieval: distributed private similarity search
for large language models. arXiv .
11