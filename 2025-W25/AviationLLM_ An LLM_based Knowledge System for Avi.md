# AviationLLM: An LLM-based Knowledge System for Aviation Training

**Authors**: Jia'ang Wan, Feng Shen, Fujuan Li, Yanjin Sun, Yan Li, Shiwen Zhang

**Published**: 2025-06-17 09:20:09

**PDF URL**: [http://arxiv.org/pdf/2506.14336v1](http://arxiv.org/pdf/2506.14336v1)

## Abstract
Aviation training is a core link in ensuring flight safety, improving
industry efficiency and promoting sustainable development. It not only involves
flight simulation but also requires the learning of a great deal of
professional aviation theory knowledge. In the existing training system, the
knowledge is mainly imparted by the the instructors. However, the number of
instructors is limited and the professional answers obtained from the Internet
are not accurate enough, resulting in low training efficiency. To address this,
we introduced LLM, but the basic pre-trained model cannot provide accurate
answers to professional fields, so we fine-tuned it. Traditional Supervised
Fine-Tuning (SFT) risk generating superficially plausible but factually
incorrect responses due to insufficient data coverage. To address this, we
employ Direct Preference Optimization(DPO). This paper proposes
Retrieval-Augmented LLM Alignment via Direct Preference Optimization(RALA-DPO).
We select open source pre-trained LLM Qwen and adapt it to aviation theory
training through DPO-based domain alignment. Simultaneously, to mitigate
hallucinations caused by training data biases, knowledge obsolescence, or
domain knowledge gaps, we implement Retrieval-Augmented Generation(RAG)
technology that combines generative and retrieval models. RALA-DPO effectively
retrieves relevant information from external knowledge bases and delivers
precise and high-quality responses through the generative model. Experimental
results demonstrate that RALA-DPO can improve accuracy in response to
professional aviation knowledge. With integrated RAG mechanisms, this system
can further improve the accuracy of answers and achieve zero-cost knowledge
updates simultaneously.

## Full Text


<!-- PDF content starts -->

arXiv:2506.14336v1  [cs.AI]  17 Jun 2025AviationLLM: An LLM-based Knowledge
System for Aviation Training
Jia’ang Wan1, Feng Shen2, Fujuan Li2, Yanjin Sun2, Yan Li2, and Shiwen
Zhang2
1Wuhan University
2China Eastern Technology Application Research And Development Center Co., Ltd
Abstract. Aviation training is a core link in ensuring flight safety, im-
proving industry efficiency and promoting sustainable development. It
not only involves flight simulation but also requires the learning of a great
deal of professional aviation theory knowledge. In the existing training
system, the knowledge is mainly imparted by the the instructors. How-
ever, the number of instructors is limited and the professional answers
obtained from the Internet are not accurate enough, resulting in low
training efficiency. To address this, we introduced LLM, but the basic pre-
trained model cannot provide accurate answers to professional fields, so
we fine-tuned it. Traditional Supervised Fine-Tuning (SFT) risk generat-
ing superficially plausible but factually incorrect responses due to insuf-
ficient data coverage. To address this, we employ Direct Preference Opti-
mization(DPO). This paper proposes Retrieval-Augmented LLM Align-
ment via Direct Preference Optimization(RALA-DPO). We select open
source pre-trained LLM Qwen and adapt it to aviation theory train-
ing through DPO-based domain alignment. Simultaneously, to mitigate
hallucinations caused by training data biases, knowledge obsolescence,
or domain knowledge gaps, we implement Retrieval-Augmented Gen-
eration(RAG) technology that combines generative and retrieval mod-
els. RALA-DPO effectively retrieves relevant information from external
knowledge bases and delivers precise and high-quality responses through
the generative model. Experimental results demonstrate that RALA-
DPO can improve accuracy in response to professional aviation knowl-
edge. With integrated RAG mechanisms, this system can further improve
the accuracy of answers and achieve zero-cost knowledge updates simul-
taneously.
Keywords: Generative Large Language Model ·Direct Preference Op-
timization ·Retrieval-Augmented Generation.
1 Introduction
With global economic integration and increasing air transport demand, the avi-
ation industry faces dual pressures from accelerated technological advancement
and rising safety standards. Aviation theory training serves as the core founda-
tion for aviation safety operations, providing essential theoretical support to pi-

2 Wan et al.
lots, maintenance personnel, and other professionals handling complex flight sce-
narios. Traditional aviation theory training is based on standardized materials,
simulator-based exercises, and instructor-led knowledge transfer. However, these
methods face significant limitations, including limited pedagogical resources, de-
layed knowledge updates, and insufficient specialized question-answering sys-
tems. These shortcomings become particularly evident when addressing con-
temporary challenges, such as dynamic decision making in emerging scenar-
ios, rapidly evolving knowledge requirements, and diverse personalized learning
needs. The breakthrough in the quality of the content generated by large lan-
guage models (LLMs) brings opportunities for innovation in the field of aviation
theory training. Using LLMs, the aviation training sector can address personnel
shortages while simultaneously reducing operational costs and improving pilot
training efficiency.
LLMs are predominantly pre-trained on large-scale general-purpose datasets.
Although such broad training equips models with robust general language under-
standing, they underperform in tasks requiring domain-specific expertise. For ex-
ample, in scientific text processing [1], models must be capable of comprehending
complex scientific terminology, concepts, and methodologies to generate accu-
rate responses. Similarly, in e-commerce search systems [2], mastery of domain-
specific terminology is crucial for generating relevant results and recommenda-
tions. This requirement for specialized domain competency extends equally to
healthcare applications [3]: large language models need to accurately understand
medical terminology, diagnoses, treatment plans, and drug interactions. More
notably, applications such as biomedical question answering and medical report
summarization significantly rely on knowledge drawn from external sources of
professional medical literature [4]. Training in aviation theory involves exten-
sive specialized terminology and low-frequency technical vocabulary, which are
underrepresented in generic training data, leading to challenges in model com-
prehension and application. Furthermore, LLMs trained on open-source internet
data inevitably inherit biases and inaccuracies. Although effective for general
tasks, such data biases may propagate incorrect or unsafe outputs in high-stakes
domains like pilot training. Moreover, since most pre-training data are several
years old, they cannot meet the timeliness requirements of the aviation train-
ing field. Thus, adapting general-domain pre-trained models to aviation-specific
scenarios while ensuring output accuracy and timeliness remains a critical chal-
lenge.
This paper proposes RALA-DPO for constructing an aviation theory training
LLM by integrating pre-trained models, DPO fine-tuning, and RAG.We con-
structs a DPO dataset in the aviation training domain for training large lan-
guage models.Secondly, the DPO fine-tuning technique is introduced to improve
the accuracy of LLM responses. Furthermore, the RAG technique is adopted
to combine the fine-tuned generative model with the retrieval model, effectively
realizing the retrieval of relevant information from external knowledge bases and
providing accurate and high-quality answers based on the generative model.

AviationLLM: An LLM-based Knowledge System for Aviation Training 3
2 Related work
The field of aviation theory training imposes stringent requirements for safety,
efficiency, and standardized operations. Traditional aviation training relies on
standardized teaching materials, simulator training, and the imparting of hu-
man experience. Derin [5] designed and implemented a simulation-based aviation
English course to improve learning gains. However, when addressing decision-
making in emergency scenarios, dynamic knowledge updates, and personalized
learning needs, there are often problems such as limited teaching resources and
outdated knowledge. In recent years, numerous studies have attempted to ap-
ply new technologies to aviation. Sihai Li [6] designed an aircraft maintenance
training platform based on virtual reality technology to enhance the efficiency
and quality of training for maintenance personnel.
Recent advances in Large Language Models (LLMs) have established the
technological foundation for their application in specialized domains. Zhang [7]
fine-tuned the models using two financial datasets, allowing them to focus pri-
marily on financial classification tasks. Significant progress has also been made
in education and interactive teaching through LLMs. For example, CyberQ [8]
integrates AISecKG [9] to combine static knowledge embedding with dynamic
knowledge injection. Using structured knowledge representation and real-time
updates, it generates intelligent question-and-answer outputs aligned with cyber-
security best practices. The SocraticLM model [10], fine-tuned on the Socrat-
icTeach dataset, enhances student guidance in critical thinking and problem-
solving. Additionally, the KnowGPT framework [11] converts diverse prompt
templates into natural language prompts interpretable and usable by language
models. This improves LLM performance in domain-specific tasks while sig-
nificantly reducing API costs. Furthermore, Qwen [12] achieves comprehensive
advances in long-text processing and multi-task capabilities through core innova-
tions like its Dual Chunk Attention (DCA) mechanism and Mixture-of-Experts
(MoE) architecture. These abilities—including in-context learning and multi-
modal fusion—now offer methodological support for complex knowledge reason-
ing in specialized fields such as aerospace.
While pre-trained LLMs exhibit general language understanding capabilities,
their outputs may not satisfy domain-specific requirements. Through fine-tuning,
models can assimilate domain knowledge and adjust generation styles, thereby
significantly enhancing task accuracy and consistency. Ouyang [13] transformed
general-purpose LLMs into instruction-following conversational agents via multi-
round supervised fine-tuning (SFT) in InstructGPT. Their key innovation was
constructing high-quality datasets of instruction-response pairs to align model
outputs with human expectations. However, SFT’s reliance on annotated data
struggles to resolve complex preference conflicts. Yao [14] proposed chain-of-
thought prompting, which introduces intermediate reasoning steps to help mod-
els decompose complex tasks into manageable components. This approach en-
ables LLMs to utilize their internal knowledge more effectively through explicit
step-by-step reasoning, improving performance on tasks requiring logical in-
ference, multi-step computation, or decision-making. OpenAI’s Reinforcement

4 Wan et al.
Learning from Human Feedback (RLHF) framework [15] achieved value align-
ment through reward modeling and proximal policy optimization but faced chal-
lenges of training instability and high computational costs. Addressing this,
Rafailov [16] introduced DPO. DPO reformulates preference learning into single-
stage loss function optimization via implicit reward modeling, mathematically
proving equivalence to RLHF while reducing training costs by 75%.
Meanwhile, large language models may produce erroneous outputs or hal-
lucinations due to training data biases, knowledge obsolescence, or insufficient
domain-specific knowledge. RAG technology effectively mitigates these halluci-
nations and knowledge latency issues through synergistic optimization between
external knowledge bases and generative models. Recent advancements in RAG
technology have progressed along two main trajectories: retrieval quality en-
hancement and knowledge-generation alignment. Representative works include
HyDE [17], which improves query reformulation through hypothetical answer
generation, and Atlas [18], which employs contrastive learning to align the se-
mantic spaces of retrievers and generators. For vertical domain applications,
Self-RAG [19] implements dynamic retrieval timing decisions using reflection
tokens, while RA-DIT [20] enhances domain adaptability via a two-stage adap-
tation process involving domain knowledge infusion followed by instruction tun-
ing. Ren [21] proposes a RAG-aided Causal Identification (RACI) model that
integrates the large language model approach. However, using DPO fine-tuning
or RAG alone cannot simultaneously meet the aviation theory training field’s
requirements for accuracy and timeliness in generated content. Therefore, this
paper proposes a construction framework called RALA-DPO for an aviation the-
ory training knowledge large model. This framework improves answer accuracy
through DPO fine-tuning and prioritizes retrieving the latest authoritative doc-
uments before generating answers via RAG. This ensures that model outputs
are always based on current valid knowledge, thereby guaranteeing timeliness.
3 Background
The Qwen model series has made remarkable progress in architectural design,
training strategies, and performance optimization, providing a strong techni-
cal foundation for fine-tuning in professional domain question-answering tasks.
The Qwen2.5 [12] series addresses the positional encoding bottleneck of tradi-
tional Transformer models in long-text processing through Dual Chunk Atten-
tion (DCA) technology. DCA divides long sequences into chunks and remaps
relative positional relationships through a dual-chunk attention mechanism, en-
abling the model to efficiently handle hundreds of thousands of tokens of context.
For example, the Qwen2.5-7B-Instruct-1M model only needs to be trained at a
length of 32K to extrapolate to 1M length tasks and achieves nearly 100% ac-
curacy in complex tasks such as key retrieval. In addition, the introduction of
sparse attention optimization significantly improves inference speed, with speeds
increasing by 3.2-6.7 times when supporting long-text inputs, making it suitable
for long-document analysis needs in professional fields.

AviationLLM: An LLM-based Knowledge System for Aviation Training 5
The pre-training phase of Qwen2.5 [22] adopts a hierarchical optimization
strategy. Firstly, it utilizes 18 trillion high-quality tokens of pre-training data
(157% more than the previous generation), covering professional fields such as
mathematics and programming, and employs synthetic data generation technol-
ogy to enhance data diversity. In long-text processing, the model adopts pro-
gressive context expansion training: starting with a sequence length of 4K, it
gradually expands to 1M tokens, and the DCA technique reduces time complex-
ity, solving the memory bottleneck of traditional Transformer architectures for
long sequences. In the post-training phase, multi-stage reinforcement learning
(DPO and GRPO algorithms) is combined to optimize capabilities such as in-
struction following and logical reasoning, enhancing output stability. Through
strict data filtering and mixed ratio optimization, the model shows significant
improvements in tasks such as commonsense reasoning and logical deduction.
This feature provides a rich knowledge base for fine-tuning in professional do-
mains. For this reason, this paper selects the Qwen model as the base model
for fine-tuning to meet the requirements of aviation theory training field for
generated content.
4 Methodology
To address the issues of insufficient accuracy and timeliness of LLMs in the field
of aviation theory training, this paper proposes RALA-DPO. We employ DPO
model fine-tuning technology to refine Qwen, and integrate RAG to develop an
aviation training-oriented LLM. DPO constrains the model to prefer professional
responses in terms of generation preferences, ensuring the accuracy of model
output, while RAG ensures the timeliness of information from the knowledge
source. The collaboration of the two significantly enhances the professionalism,
and timeliness of the output. The RALA-DPO is illustrated in the figure 1.
4.1 Model Refinement Through Direct Preference Optimization
This paper selects open-source pre-trained large language model Qwen and
adapts it to the specialized domain of aviation industry professional training
through efficient fine-tuning techniques. Specifically, we employ Direct Prefer-
ence Optimization for model refinement. DPO directly optimizes language model
policies using human preference data to align model generation behaviors, while
simultaneously mitigating hallucination issues in generated content. This ap-
proach enables more efficient and targeted processing of preference data, ensur-
ing enhanced alignment with human evaluators’ priorities in generated outputs.
By eliminating the need for explicit reward modeling and iterative RL-based
policy updates, DPO achieves superior training stability and computational ef-
ficiency compared to conventional PPO-based frameworks, while maintaining
precise control over domain-specific response quality and factual accuracy re-
quired for aviation theory training scenarios.

6 Wan et al.
File
File
 DPO
QueryEmbedding
Modelembeddings
Vector
DBQuery+
Contextreply
FileEmbedding
Modelembeddings
relevant
context
Fig. 1. When a user submits a query, the system first performs semantic embedding
on the query to generate a corresponding semantic embedding vector. Subsequently,
cosine similarity is used to calculate the vector’s similarity to vectors in the vector
database. This identifies the context most relevant to the query. These contexts are
then substituted into a predefined prompt template. Finally, the augmented prompt,
along with the original query, is input into a generative model trained using DPO to
produce the reply.
For each input prompt x, we establish a pairwise comparison scenario con-
taining two distinct outputs: a preferred response ywand a dispreferred response
yl. According to the Bradley-Terry model, the preference probability Pcan be
formulated as:
P(yw≻yl|x) =exp(R(x, yw))
exp(R(x, yw)) + exp( R(x, yl)), (1)
where the reward function R(x, y) can be analytically expressed through its
policy model πand reference model πref, thereby enabling direct optimization of
the policy model on preference data under proper parametrization. Specifically,
the reward function can be formulated as:
R(x, y) =βlogπ(y|x)
πref(y|x)+βlogZ(x). (2)
In the formulation, βserves as the temperature parameter controlling the degree
of policy deviation from the reference model, while Z(x) acts as a normalization
constant to ensure training stability. Building on this theoretical insight, the pol-
icy model can be directly optimized using human feedback data. By substituting
the reward expression into the Bradley-Terry preference probability equation, we
derive the simplified preference likelihood formulation:

AviationLLM: An LLM-based Knowledge System for Aviation Training 7
P(yw≻yl|x) =1
1 + exp
βlogπ(yl|x)
πref(yl|x)−βlogπ(yw|x)
πref(yw|x). (3)
Given the constructed preference data set D={(xi, y(i)
w, y(i)
l)}N
i=1, we optimize
the policy parameters by maximizing the log-likelihood of observed preferences.
L(π) =NX
i=1logP(y(i)
w≻y(i)
l|xi). (4)
Substituting the simplified preference probability expression into the objective
function, we derive the following DPO loss function:
L(π) =NX
i=1logσ 
βlogπ(y(i)
w|xi)
πref(y(i)
w|xi)−βlogπ(y(i)
l|xi)
πref(y(i)
l|xi)!
, (5)
where σdenotes the logistic sigmoid function. This maximization objective is
equivalently reformulated as minimizing the negative log-likelihood:
LDPO =−E(xi,yw,yl)∼D
logσ
βlogπ(yw|xi)
πref(yw|xi)−βlogπ(yl|xi)
πref(yl|xi)
.(6)
The gradient of the loss function simultaneously adjusts the probability distri-
butions of both preferred responses ywand dispreferred responses ylthrough the
following mechanism:
∇θL ∝ − βπ(yw|x)
πref(yw|x)∇θlogπ(yw|x)−π(yl|x)
πref(yl|x)∇θlogπ(yl|x)
.(7)
Through this dual-ascent optimization of the DPO loss, the model directly in-
ternalizes human preference patterns while maintaining fundamental language
competencies preserved in πrefThe constrained update mechanism ensures that
preference alignment occurs within a trust region of the reference policy, effec-
tively balancing three critical objectives: Maximizing human preference satis-
faction, Preserving linguistic coherence and domain knowledge and Minimizing
hallucination through reference policy anchoring.
Through optimization of this loss function, the model directly learns to gen-
erate preferred responses through dual mechanisms of preference maximization
and dispreference minimization.
4.2 Optimization Strategies for RAG in Aviation Theory Training
Domain
Given the critical demands for timeliness in aviation theory training knowl-
edge, this paper employs RAG technology that integrates generative models

8 Wan et al.
with retrieval mechanisms. The approach effectively retrieves relevant informa-
tion from scalable external knowledge bases and delivers precise, high-quality
responses through generative models, thereby addressing the limitations of con-
ventional generative models in information reliability and knowledge coverage.
This methodology enhances the reliability of large language model outputs and
response accuracy while providing verifiable reference sources, achieving a sub-
stantial reduction in hallucination issues inherent to large language models.
First, we establish an aviation theory training knowledge base by ingest-
ing extensive unstructured domain-specific data, including the Flight Instructor
Theoretical Manual and the Basic Rules of Flight of the People’s Republic of
China. After these data are processed into standard text data, they are seg-
mented into multiple knowledge fragments. A semantic embedding vector rep-
resenting the semantics of each fragment is generated through a text semantic
embedding model to form a collection D={d1, d2, . . . , d n}for retrieval. The
text semantic embedding model is a type of text processing model in the field
of natural language processing for the specific task of text semantic extraction
and compression. It compresses words, phrases, sentences, or even documents in
the text into a high-dimensional embedding vector through the Transformer en-
coding layer. The obtained embedding vector is a compressed representation of
the complete text semantics, which is easy to store and retrieve efficiently. When
storing text content, this embedding vector is stored in the database together
with relevant data of the original text as an index for subsequent retrieval.
During the retrieval phase, first perform semantic embedding on the aviation
training-related questions raised by users to generate a semantic embedding vec-
torqfor the input question. Perform relevance calculation with key documents
in the knowledge library through cosine similarity calculation:
s(q, d) =Emb R(q)·Emb R(d)
∥Emb R(q)∥ · ∥ Emb R(d)∥, (8)
where, Emb R(·) is the vector representation generated by the retrieval model.Retrieve
the top kdocuments that are most relevant to query qfrom the document collec-
tionD={d1, d2, . . . , d k}. Then input these knowledge text fragments into the
prompt template together with the question and feed them into the generation
model to obtain Output a.
a= LM(concat( q, D)), (9)
where, LM( ·) is the generation model. Under the RAG framework, the language
model leverages its advanced generative capabilities to synthesize contextually
grounded responses by jointly analyzing both the user’s query and retrieved
contextual information.
By implementing Retrieval-Augmented Generation technology, the model can
generate accurate and timely content, particularly crucial for aviation profession-
als’ theoretical training where regulatory documents frequently undergo updates.
The dynamic updatability of the knowledge base enables the model to utilize the
latest aviation theory training information without requiring retraining, while
simultaneously mitigating hallucination issues inherent in large language models.

AviationLLM: An LLM-based Knowledge System for Aviation Training 9
5 Experiments
In this section, we outline our experimental workflow, which includes data pro-
cessing, validation of the DPO fine-tuning method, and validation of the impact
of RAG technology on the models.
5.1 Construction of the Dataset
Given the stringent requirements for professional knowledge accuracy and time-
liness in aviation theory training. This study systematically aggregates multi-
domain aviation training data spanning foundational aviation theory, meteoro-
logical systems, aerodynamics, visual flight rules, and civil aviation regulations.
The curated data derives from three authoritative sources: 1) certified training
materials and official textbooks, 2) peer-reviewed academic literature and tech-
nical publications, and 3) regulatory documentation from the International Civil
Aviation Organization (ICAO) and national aviation authorities, ensuring both
provenance verification and content precision.
The data processing phase implements rigorous cleaning, structural organiza-
tion, and categorical classification protocols to guarantee information integrity.
Aviation knowledge texts undergo expert-guided manual annotation, construct-
ing a premium dataset that encapsulates all critical knowledge components, re-
sulting in 9,740 curated data pairs designated as preferred model responses.
Concurrently, we generate auxiliary coarse-grained responses for identical queries
using untrained large language models, establishing a dual-response framework.
This comparative architecture enables systematic output optimization through
response alignment analysis, ensuring strict compliance with aviation standard
operating procedures.
5.2 Experimental Setup
This study uses the open-source pre-trained model Qwen2.5-14B as its base
model. Comparative experiments employing Supervised Fine-Tuning (SFT) and
Direct Preference Optimization (DPO) techniques were conducted to evaluate
their respective effectiveness. Both methods used a learning rate of 0.0003 and
a batch size of 16, training them on the ATDS dataset.
For model evaluation, this paper utilizes the AI evaluator Themis-turbo from
Alibaba Cloud’s Bailian platform to assess two models, with the evaluation
dataset derived from random sampling of ATDS. The configuration of this AI
evaluator is set as follows: Models are scored across six dimensions—accuracy,
relevance, completeness, source reliability, clarity of answer structure, and timeli-
ness—using a 5-tier rating system (the more the response aligns with the scoring
criteria, the higher the score).
To validate Retrieval-Augmented Generation (RAG) effectiveness, we imple-
mented the expert evaluation method from OpsEval [23], using human experts
to score output fluency, accuracy, and timeliness on a 0-5 scale. Four model

10 Wan et al.
variants were tested:SFT-fine-tuned Qwen2.5-14B, DPO-fine-tuned Qwen2.5-
14B, SFT+RAG-enhanced Qwen2.5-14B and DPO+RAG-enhanced Qwen2.5-
14B. Aviation training domain experts scored 100 sample question-answer pairs
from each model configuration, with performance quality positively correlating
to higher numerical ratings across all metrics.
5.3 Experiment Results Analysis
The comparative results of models trained with SFT and DPO, evaluated by the
AI evaluator Themis-turbo, are shown in Table 1, while the scores assigned by
aviation theory training experts to the four models are presented in Table 2.
Table 1. Results of Model Comparison
Model Win Lose Tie Win%
Qwen2.5-14B-SFT 192 285 23 0.43
Qwen2.5-14B-DPO 285 192 23 0.57
Table 1 compares the performance of Qwen2.5-14B fine-tuned with DPO ver-
sus SFT on the ATDS evaluation set. The results demonstrate that the DPO-
tuned Qwen2.5-14B outperforms its SFT-tuned counterpart in aviation theory
training tasks. This indicates that the DPO fine-tuning technique achieves supe-
rior performance across all six evaluation dimensions—accuracy, relevance, com-
pleteness, source reliability, clarity of answer structure, and timeliness—compared
to SFT fine-tuning. Evaluated by Alibaba Cloud’s Bailian platform AI evaluator
Themis-turbo, the DPO-tuned model exhibits a 14% higher win rate than the
SFT-tuned model.
Table 2. Expert evaluation results
Model Average Scores Assessed by Experts Total
Fluency Accuracy Timeliness
SFT 2.98 4.38 3.56 10.92
DPO 3.43 4.43 3.75 11.62
SFT+RAG 3.98 4.56 4.42 12.96
DPO+RAG 4.25 4.83 4.63 13.71
In the Table 2, SFT indicates SFT-fine-tuned Qwen2.5-14B, DPO indicates
DPO-fine-tuned Qwen2.5-14B, SFT+RAG indicates SFT+RAG-enhanced Qwen
2.5-14B, and DPO+RAG indicates DPO+RAG-enhanced Qwen2.5-14B. As can
be seen from the table, the fluency, accuracy and timeliness of the models fine-
tuned by DPO are obviously superior to those fine-tuned by SFT, which indicates
that the answers generated by the models fine-tuned by DPO are more in line

AviationLLM: An LLM-based Knowledge System for Aviation Training 11
with the preferences of experts than those obtained by the models fine-tuned by
SFT. At the same time, the model using RAG technology is significantly better
than the model without RAG technology, which indicates that the use of RAG
technology can enhance the understanding and generation ability of the large
model to the business question and answer data in the professional field, and
generate more smooth and accurate answers with stronger timeliness.
6 Conclusion
This paper addresses persistent challenges in aviation theory training—including
instructor shortages, outdated knowledge, and domain-specific LLM hallucina-
tions—by proposing the innovative RALA-DPO framework. The approach con-
structs a domain preference optimization (DPO) dataset for aviation theory
training through professional data annotation and large model integration. We
fine-tune the open-source Qwen model using DPO methodology to enhance ac-
curacy in aviation theoretical training. Simultaneously, we implement Retrieval-
Augmented Generation (RAG) to establish a dynamically updatable professional
knowledge base, thereby improving response timeliness. RALA-DPO enhances
output credibility through interpretable retrieval evidence, establishing a se-
cure technical pathway for intelligent aviation theory training transformation.
Experimental validation demonstrates that DPO significantly improves profes-
sional adaptability to aviation standard operating procedures, resolving output
deviations stemming from conventional fine-tuning methods’ limited data cov-
erage. Further integration with RAG enables dynamic retrieval from updated
regulatory documents and domain knowledge bases, simultaneously overcoming
static training data limitations while establishing a zero-cost dynamic knowledge
update mechanism.
References
1. Andres M Bran, Sam Cox, Oliver Schilter, Carlo Baldassari, Andrew D White, and
Philippe Schwaller. Chemcrow: Augmenting large-language models with chemistry
tools. Nature Machine Intelligence , 2023.
2. Shujuan Zhao, Lingfeng Qiao, Kangyang Luo, Qian-Wen Zhang, Junru Lu, and
Di Yin. Snfinllm: Systematic and nuanced financial domain adaptation of chinese
large language models. arXiv preprint arXiv:2408.02302 , 2024.
3. Qizhi Pei, Lijun Wu, Kaiyuan Gao, Xiaozhuan Liang, Yin Fang, Jinhua Zhu, Sh-
ufang Xie, Tao Qin, and Rui Yan. Biot5+: Towards generalized biological under-
standing with iupac integration and multi-task tuning. Proceedingss of the ACL ,
2024.
4. Franck Dernoncourt and Ji Young Lee. Pubmed 200k rct: A dataset for sequential
sentence classification in medical abstracts. Proceedings of the IJCNLP , 2017.
5. Derin Atay. Enhancing aviation english competency: A simulation-based approach
for aspiring pilots. English for Specific Purposes , 76:106–121, 2024.
6. Sihai Li. Design and development of aviation aircraft maintenance training plat-
form based on vr technology. Procedia Computer Science , 228:898–906, 2023.

12 Wan et al.
7. Boyu Zhang, Hongyang Yang, and Xiao-Yang Liu. Instructfingpt: Financial sen-
timent analysis by instruction tuning of general-purpose large language models.
Proceedings of the IJCAI , 2023.
8. Garima Agrawal, Kuntal Pal, Yuli Deng, Huan Liu, and Ying-Chih Chen. Cyberq:
Generating questions and answers for cybersecurity education using knowledge
graph-augmented llms. Proceedings of the AAAI , 2024.
9. Garima Agrawal. Aiseckg: Knowledge graph dataset for cybersecurity education.
Proceedings of the AAAI , 2023.
10. Jiayu Liu, Zhenya Huang, Tong Xiao, Jing Sha, Jinze Wu, Qi Liu, Shijin Wang,
and Enhong Chen. Socraticlm: Exploring socratic personalized teaching with large
language models. Proceedings of the NeurIPS , 2024.
11. Qinggang Zhang, Junnan Dong, Hao Chen, Daochen Zha, Zailiang Yu, and Xiao
Huang. Knowgpt: Knowledge graph based prompting for large language models.
Proceedings of the NeurIPS , 2024.
12. Jin Xu, Zhifang Guo, Jinzheng He, Hangrui Hu, Ting He, Shuai Bai, Keqin Chen,
Jialin Wang, Yang Fan, Kai Dang, Bin Zhang, Xiong Wang, Yunfei Chu, and
Junyang Lin. Qwen2.5-omni technical report. arXiv preprint arXiv:2503.20215 ,
2025.
13. Long Ouyang, Jeff Wu, Xu Jiang, Diogo Almeida, Carroll L. Wainwright, Pamela
Mishkin, Chong Zhang, Sandhini Agarwal, Katarina Slama, Alex Ray, John Schul-
man, Jacob Hilton, Fraser Kelton, Luke Miller, Maddie Simens, Amanda Askell,
Peter Welinder, Paul Christiano, Jan Leike, and Ryan Lowe. Training language
models to follow instructions with human feedback. Advances in Neural Informa-
tion Processing Systems , 35:27730–27744, 2022.
14. Shunyu Yao, Dian Yu, Jeffrey Zhao, Izhak Shafran, Thomas L. Griffiths, Yuan
Cao, and Karthik Narasimhan. Tree of thoughts: Deliberate problem solving with
large language models. Proceedings of the NeurIPS , 2024.
15. Paul F. Christiano, Miljan Martic, Jan Leike, Shane Legg, Tom B. Brown, and
Dario Amodei. Deep reinforcement learning from human preferences. Advances in
Neural Information Processing Systems , 30, 2017.
16. Rafael Rafailov, Archit Sharma, Eric Mitchell, Stefano Ermon, Christopher D.
Manning, and Chelsea Finn. Direct preference optimization: Your language model
is secretly a reward model. Advances in Neural Information Processing Systems ,
36, 2024.
17. Luyu Gao, Xueguang Ma, Jimmy Lin, and Jamie Callan. Precise zero-shot dense
retrieval without relevance labels. Proceedings of the 61st Annual Meeting of the
Association for Computational Linguistics , 1:1762–1777, 2023.
18. Gautier Izacard, Patrick Lewis, Maria Lomeli, Lucas Hosseini, Fabio Petroni, Timo
Schick, Jane Dwivedi-Yu, Armand Joulin, Sebastian Riedel, and Edouard Grave.
Atlas: Few-shot learning with retrieval augmented language models. Journal of
Machine Learning Research , 24:1–43, 2023.
19. Akari Asai, Zeqiu Wu, Yizhong Wang, Avirup Sil, and Hannaneh Hajishirzi. Self-
rag: Learning to retrieve, generate, and critique through self-reflection. arXiv
preprint arXiv:2310.11511 , 2023.
20. Xi Victoria Lin, Xilun Chen, Mingda Chen, Weijia Shi, Maria Lomeli, Rich James,
Pedro Rodriguez, Jacob Kahn, Gergely Szilvasy, Mike Lewis, Luke Zettlemoyer,
and Scott Yih. Ra-dit: Retrieval-augmented dual instruction tuning. Proceedings
of the Twelfth International Conference on Learning Representations , 2023.
21. Tengfei Ren, Zhipeng Zhang, Bo Jia, and Shiwen Zhang. Retrieval-augmented
generation-aided causal identification of aviation accidents: A large language model
methodology. Expert Systems with Applications , 278:127306, 2025.

AviationLLM: An LLM-based Knowledge System for Aviation Training 13
22. Qwen, :, An Yang, Baosong Yang, Beichen Zhang, Binyuan Hui, Bo Zheng, Bowen
Yu, Chengyuan Li, Dayiheng Liu, Fei Huang, Haoran Wei, Huan Lin, Jian Yang,
Jianhong Tu, Jianwei Zhang, Jianxin Yang, Jiaxi Yang, Jingren Zhou, Junyang
Lin, Kai Dang, Keming Lu, Keqin Bao, Kexin Yang, Le Yu, Mei Li, Mingfeng Xue,
Pei Zhang, Qin Zhu, Rui Men, Runji Lin, Tianhao Li, Tianyi Tang, Tingyu Xia,
Xingzhang Ren, Xuancheng Ren, Yang Fan, Yang Su, Yichang Zhang, Yu Wan,
Yuqiong Liu, Zeyu Cui, Zhenru Zhang, and Zihan Qiu. Qwen2. 5 technical report.
arXiv preprint arXiv:2412.15115 , 2024.
23. Yuhe Liu, Changhua Pei, Longlong Xu, Bohan Chen, Mingze Sun, Zhirui Zhang,
Yongqian Sun, Shenglin Zhang, Kun Wang, Haiming Zhang, Jianhui Li, Gaogang
Xie, Xidao Wen, Xiaohui Nie, Minghua Ma, and Dan Pei. Opseval: A compre-
hensive task-oriented aiops benchmark for large language models. arXiv preprint
arXiv:2310.07637 , 2023.