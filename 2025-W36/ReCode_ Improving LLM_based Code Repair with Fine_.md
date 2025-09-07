# ReCode: Improving LLM-based Code Repair with Fine-Grained Retrieval-Augmented Generation

**Authors**: Yicong Zhao, Shisong Chen, Jiacheng Zhang, Zhixu Li

**Published**: 2025-09-02 13:58:48

**PDF URL**: [http://arxiv.org/pdf/2509.02330v1](http://arxiv.org/pdf/2509.02330v1)

## Abstract
Recent advances in large language models (LLMs) have demonstrated impressive
capabilities in code-related tasks, such as code generation and automated
program repair. Despite their promising performance, most existing approaches
for code repair suffer from high training costs or computationally expensive
inference. Retrieval-augmented generation (RAG), with its efficient in-context
learning paradigm, offers a more scalable alternative. However, conventional
retrieval strategies, which are often based on holistic code-text embeddings,
fail to capture the structural intricacies of code, resulting in suboptimal
retrieval quality. To address the above limitations, we propose ReCode, a
fine-grained retrieval-augmented in-context learning framework designed for
accurate and efficient code repair. Specifically, ReCode introduces two key
innovations: (1) an algorithm-aware retrieval strategy that narrows the search
space using preliminary algorithm type predictions; and (2) a modular
dual-encoder architecture that separately processes code and textual inputs,
enabling fine-grained semantic matching between input and retrieved contexts.
Furthermore, we propose RACodeBench, a new benchmark constructed from
real-world user-submitted buggy code, which addresses the limitations of
synthetic benchmarks and supports realistic evaluation. Experimental results on
RACodeBench and competitive programming datasets demonstrate that ReCode
achieves higher repair accuracy with significantly reduced inference cost,
highlighting its practical value for real-world code repair scenarios.

## Full Text


<!-- PDF content starts -->

ReCode: Improving LLM-based Code Repair with Fine-Grained
Retrieval-Augmented Generation
Yicong Zhao
College of Computer Science and Artificial Intelligence,
Fudan University
Shanghai, China
zhaoyc22@m.fudan.edu.cnShisong Chen
Shanghai Institute of Artificial Intelligence for Education,
East China Normal University
Shanghai, China
sschen@stu.ecnu.edu.cn
Jiacheng Zhang
College of Computer Science and Artificial Intelligence,
Fudan University
Shanghai, China
jiachengzhang22@m.fudan.edu.cnZhixu Li∗
School of Information, Renmin University of China;
School of Smart Governance, Renmin University of China
Beijing, China
zhixuli@ruc.edu.cn
Abstract
Recent advances in large language models (LLMs) have demon-
strated impressive capabilities in code-related tasks such as code
generation and automated program repair. Despite their promis-
ing performance, most existing approaches for code repair suffer
from high training costs or computationally expensive inference.
Retrieval-augmented generation (RAG), with its efficient in-context
learning paradigm, offers a more scalable alternative. However,
conventional retrieval strategies, which are often based on holis-
tic code-text embeddings, fail to capture the structural intricacies
of code, resulting in suboptimal retrieval quality. To address the
above limitations, we propose ReCode , a fine-grained retrieval-
augmented in-context learning framework designed for accurate
and efficient code repair. Specifically, ReCode introduces two key
innovations: (1) an algorithm-aware retrieval strategy that narrows
the search space using preliminary algorithm type predictions; and
(2) a modular dual-encoder architecture that separately processes
code and textual inputs, enabling fine-grained semantic matching
between input and retrieved contexts. Furthermore, we propose
RACodeBench, a new benchmark constructed from real-world user-
submitted buggy code, which addresses the limitations of synthetic
benchmarks and supports realistic evaluation. Experimental results
on RACodeBench and competitive programming datasets demon-
strate that ReCode achieves higher repair accuracy with signifi-
cantly reduced inference cost, highlighting its practical value for
real-world code repair scenarios.
∗Corresponding author.
Permission to make digital or hard copies of all or part of this work for personal or
classroom use is granted without fee provided that copies are not made or distributed
for profit or commercial advantage and that copies bear this notice and the full citation
on the first page. Copyrights for components of this work owned by others than the
author(s) must be honored. Abstracting with credit is permitted. To copy otherwise, or
republish, to post on servers or to redistribute to lists, requires prior specific permission
and/or a fee. Request permissions from permissions@acm.org.
Conference’17, Washington, DC, USA
©2025 Copyright held by the owner/author(s). Publication rights licensed to ACM.
ACM ISBN 978-x-xxxx-xxxx-x/YYYY/MM
https://doi.org/10.1145/nnnnnnn.nnnnnnnCCS Concepts
•Computing methodologies →Natural language processing ;
•Information systems →Information retrieval .
Keywords
Code Repair, In-Context Learning, Retrieval Augmented, Bench-
mark
ACM Reference Format:
Yicong Zhao, Shisong Chen, Jiacheng Zhang, and Zhixu Li. 2025. ReCode:
Improving LLM-based Code Repair with Fine-Grained Retrieval-Augmented
Generation. In .ACM, New York, NY, USA, 11 pages. https://doi.org/10.1145/
nnnnnnn.nnnnnnn
1 Introduction
Large language models have achieved significant breakthroughs in
recent years, driven by both the continuous scaling of model param-
eters and advancements in training methodologies. State-of-the-art
models such as GPT-4[ 1] and DeepSeek-R1[ 16] have demonstrated
exceptional performance across diverse language understanding
and generation tasks, while simultaneously catalyzing advance-
ments in numerous downstream applications, which highlights
their remarkable generalization capabilities. As LLM capabilities
advance, research in code intelligence has increasingly focused on
the field of code generation, where models synthesize executable
program code from natural language specifications or contextual
prompts. The development of Codex[ 9] marked a pivotal milestone
in this direction, ultimately leading to practical deployments like
GitHub Copilot and ushering in a new era of intelligent program-
ming assistants. Subsequent releases of open-source models, in-
cluding CodeLlama[ 33] and StarCoder[ 24], have further validated
the robust capabilities of LLMs in comprehending code semantics,
performing logical reasoning, and handling complex programming
tasks with increasing sophistication.
Building on the remarkable capabilities of LLMs in code un-
derstanding and generation, recent research increasingly explores
their potential in automated code repair, where models directly
propose fixes for buggy programs. However, given the complex
semantics and highly rigid syntax of programming languages, accu-
rate code repair remains a non-trivial challenge for existing LLMs.arXiv:2509.02330v1  [cs.SE]  2 Sep 2025

Conference’17, July 2017, Washington, DC, USA Yicong Zhao, Shisong Chen, Jiacheng Zhang, and Zhixu Li
/gid00018/gid00048/gid00032/gid00046/gid00047/gid00036/gid00042/gid00041/gid01143/gid00001/gid00008/gid00036/gid00049/gid00032/gid00041/gid00001/gid00028/gid00041/gid00001/gid00036/gid00041/gid00047/gid00032/gid00034/gid00032/gid00045/gid00001/gid00028/gid00045/gid00045/gid00028/gid00052/gid00001/gid00041/gid00048/gid00040/gid00046/gid01142/gid00001/gid00045/gid00032/gid00047/gid00048/gid00045/gid00041/gid00001/gid00047/gid00035/gid00032/gid00001
/gid00046/gid00040/gid00028/gid00039/gid00039/gid00032/gid00046/gid00047/gid00001/gid00043/gid00042/gid00046/gid00036/gid00047/gid00036/gid00049/gid00032/gid00001/gid00041/gid00048/gid00040/gid00029/gid00032/gid00045/gid00001/gid00036/gid00041/gid00001/gid00047/gid00035/gid00032/gid00001/gid00028/gid00045/gid00045/gid00028/gid00052/gid01141
/gid00024/gid00045/gid00042/gid00041/gid00034/gid00001/gid00004/gid00042/gid00031/gid00032/gid01143
/gid00002/gid00039/gid00034/gid00042/gid00045/gid00036/gid00047/gid00035/gid00040/gid00001/gid00047/gid00052/gid00043/gid00032/gid01143/gid00001/gid00008/gid00045/gid00032/gid00032/gid00031/gid00052/gid00001/gid00002/gid00039/gid00034/gid00042/gid00045/gid00036/gid00047/gid00035/gid00040/gid01141
/gid00019/gid00032/gid00047/gid00045/gid00036/gid00032/gid00049/gid00028/gid00039/gid00001/gid00006/gid00051/gid00032/gid00040/gid00043/gid00039/gid00028/gid00045/gid01143/gid00036/gid00041/gid00047/gid00001/gid00033/gid00036/gid00041/gid00031/gid00014/gid00036/gid00041/gid00017/gid00042/gid00046/gid00036/gid00047/gid00036/gid00049/gid00032/gid01175/gid00036/gid00041/gid00047/gid01185/gid00001/gid00041/gid00048/gid00040/gid00046/gid01176/gid00001/gid01179
/gid00001/gid00001/gid00001/gid00001/gid00036/gid00041/gid00047/gid00001/gid00040/gid00036/gid00041/gid00017/gid00042/gid00046/gid00001/gid01753/gid00001/gid01087/gid01144
/gid00001/gid00001/gid00001/gid00001/gid00033/gid00042/gid00045/gid00001/gid01175/gid00036/gid00041/gid00047/gid00001/gid00041/gid00048/gid00040/gid00001/gid01143/gid00001/gid00041/gid00048/gid00040/gid00046/gid01176
/gid00001/gid00001/gid00001/gid00001/gid00001/gid00001/gid00001/gid00001/gid00036/gid00033/gid00001/gid01175/gid00041/gid00048/gid00040/gid00001/gid01755/gid00001/gid01087/gid00001/gid01086/gid01086/gid00001/gid00041/gid00048/gid00040/gid00001/gid01754/gid00001/gid00040/gid00036/gid00041/gid00017/gid00042/gid00046/gid01176
/gid00001/gid00001/gid00001/gid00001/gid00001/gid00001/gid00001/gid00001/gid00001/gid00001/gid00001/gid00001/gid00040/gid00036/gid00041/gid00017/gid00042/gid00046/gid00001/gid01753/gid00001/gid00041/gid00048/gid00040/gid01144
/gid00001/gid00001/gid00001/gid00001/gid00045/gid00032/gid00047/gid00048/gid00045/gid00041/gid00001/gid00040/gid00036/gid00041/gid00017/gid00042/gid00046/gid00001/gid01755/gid00001/gid01087/gid00001/gid01148/gid00001/gid00040/gid00036/gid00041/gid00017/gid00042/gid00046/gid00001/gid01143/gid00001/gid01162/gid01088/gid01144
/gid01180
/gid00036/gid00041/gid00047/gid00001/gid00033/gid00036/gid00041/gid00031/gid00014/gid00028/gid00051/gid01175/gid00036/gid00041/gid00047/gid00001/gid01185/gid00041/gid00048/gid00040/gid00046/gid01176/gid00001/gid01179
/gid00001/gid00001/gid00001/gid00001/gid00036/gid00041/gid00047/gid00001/gid00040/gid00028/gid00051/gid00015/gid00048/gid00040/gid00001/gid01753/gid00001/gid01087/gid01144
/gid00001/gid00001/gid00001/gid00001/gid00033/gid00042/gid00045/gid00001/gid01175/gid00036/gid00041/gid00047/gid00001/gid00041/gid00048/gid00040/gid00001/gid01143/gid00001/gid00041/gid00048/gid00040/gid00046/gid01176
/gid00001/gid00001/gid00001/gid00001/gid00001/gid00001/gid00001/gid00001/gid00036/gid00033/gid00001/gid01175/gid00041/gid00048/gid00040/gid00001/gid01755/gid00001/gid00040/gid00028/gid00051/gid00015/gid00048/gid00040/gid01176
/gid00001/gid00001/gid00001/gid00001/gid00001/gid00001/gid00001/gid00001/gid00001/gid00001/gid00001/gid00001/gid00040/gid00028/gid00051/gid00015/gid00048/gid00040/gid00001/gid01753/gid00001/gid00041/gid00048/gid00040/gid01144
/gid00001/gid00001/gid00001/gid00001/gid00045/gid00032/gid00047/gid00048/gid00045/gid00041/gid00001/gid00040/gid00028/gid00051/gid00015/gid00048/gid00040/gid01144
/gid01180/gid00036/gid00041/gid00047/gid00001/gid00033/gid00036/gid00041/gid00031/gid00014/gid00028/gid00051/gid01175/gid00036/gid00041/gid00047/gid01185/gid00001/gid00041/gid00048/gid00040/gid00046/gid01176/gid00001/gid01179
/gid00001/gid00001/gid00001/gid00001/gid00036/gid00041/gid00047/gid00001/gid00040/gid00028/gid00051/gid00015/gid00048/gid00040/gid00001/gid01753/gid00001/gid00041/gid00048/gid00040/gid00046/gid01177/gid01087/gid01178/gid01144
/gid00001/gid00001/gid00001/gid00001/gid00033/gid00042/gid00045/gid00001/gid01175/gid00036/gid00041/gid00047/gid00001/gid00036/gid00001/gid01753/gid00001/gid01088/gid01144/gid00001/gid00036/gid00001/gid01754/gid00001/gid00041/gid01144/gid00001/gid01748/gid01748/gid00036/gid01176
/gid00001/gid00001/gid00001/gid00001/gid00001/gid00001/gid00001/gid00001/gid00036/gid00033/gid00001/gid01175/gid00041/gid00048/gid00040/gid00046/gid01177/gid00036/gid01178/gid00001/gid01755/gid00001/gid00040/gid00028/gid00051/gid00015/gid00048/gid00040/gid01176
/gid00001/gid00001/gid00001/gid00001/gid00001/gid00001/gid00001/gid00001/gid00001/gid00001/gid00001/gid00001/gid00040/gid00028/gid00051/gid00015/gid00048/gid00040/gid00001/gid01753/gid00001/gid00041/gid00048/gid00040/gid00046/gid01177/gid00036/gid01178/gid01144
/gid00001/gid00001/gid00001/gid00001/gid00045/gid00032/gid00047/gid00048/gid00045/gid00041/gid00001/gid00040/gid00028/gid00051/gid00015/gid00048/gid00040/gid01144
/gid01180
/gid00007/gid00036/gid00051/gid00001/gid00020/gid00042/gid00039/gid00048/gid00047/gid00036/gid00042/gid00041/gid01143 /gid00001/gid00001/gid00036/gid00041/gid00047/gid00001/gid00040/gid00036/gid00041/gid00017/gid00042/gid00046/gid00001/gid01753/gid00001/gid01088/gid00032/gid01096/gid01144
Figure 1: An example of our proposed ReCode Method. For
a given question, ReCode first determines the algorithm of
the wrong code to narrow the subset of our constructed code
knowledge base. Then, the exemplar retrieved through our
dual-view fused feature is connected with the user’s query
to prompt the LLM to give final solutions.
Many studies leverage inference-time strategies to mitigate cur-
rent limitations. Sampling-based approaches, such as best-of- 𝑁[27]
and self-consistency [ 38], improve repair quality by producing di-
verse candidates and selecting the most promising output. More
recently, iterative self-repair methods have gained traction: instead
of regenerating code from scratch, models refine initial predictions
through multi-turn prompting and execution feedback [ 10,23,42],
significantly enhancing efficiency while maintaining high repair
accuracy [ 30]. In parallel, a complementary line of work employs su-
pervised fine-tuning on curated repair datasets [ 7,13,17], enabling
task-specific adaptation through additional training and labeled
data.
Despite their effort, two key limitations consistently impede
the practical deployment of previous code repair methods. Firstly,
both training-based and inference-time approaches suffer from
substantial computational costs: the former requires expensive
human annotations and training, while the latter often incurs
high inference-time latency due to multi-turn reasoning or sam-
pling procedures. In addition, these methods exhibit limited adapt-
ability to out-of-distribution (OOD) defects and novel repair pat-
terns. Training-based methods struggle to incorporate emerging
knowledge without frequent continued training, which is not only
resource-intensive but also susceptible to catastrophic forgetting
[28], which compromises performance on previously seen tasks.Meanwhile, inference-time strategies, though avoiding repeated re-
training, are fundamentally trapped in the static knowledge gained
during the training phase, making it difficult to generalize to unseen
or structurally unfamiliar bug scenarios.
Motivated by the aforementioned limitations, we explore retrieval-
augmented generation (RAG) as a promising alternative paradigm
for enabling code repair without requiring parameter updates or
additional training. However, our experiment results in Table 1
reveal the limitations of existing RAG methods in the context of
code repair. Specifically, conventional RAG approaches typically
encode problem descriptions and source code in a monolithic fash-
ion. This unified representation neglects the inherent structure
and semantics of code, thereby hindering the retrieval of relevant
exemplars.
In this paper, we propose ReCode , a fine-grained retrieval-
augmented generation framework that constructs modular rep-
resentations for both source code and its accompanying textual
descriptions. This design enables the model to capture domain-
specific semantics effectively.
Beyond modular modeling, ReCode further leverages the model’s
intrinsic capacities in code comprehension and reasoning by intro-
ducing an algorithm-aware hybrid retrieval mechanism.
Specifically, ReCode first uses the LLM to analyze the user’s
query to infer the algorithm type, which is then utilized to retrieve
the most relevant exemplars from our pre-constructed algorithm-
specific knowledge corpus. By integrating algorithmic categoriza-
tion with the model’s semantic modeling towards the defective
code, ReCode significantly enhances the contextual relevance of
retrieved in-context exemplars and supports more accurate and
adaptive code repair generation.
To further support rigorous and realistic evaluation, we con-
struct a high-quality benchmark, RACodeBench , consisting of a
systematic collection of real-world buggy–fixed code pairs manu-
ally curated and annotated from user submissions. RACodeBench
reflects authentic software development scenarios and enables pre-
cise performance assessment. Extensive experiments conducted on
both RACodeBench and several widely used competitive program-
ming benchmarks demonstrate the superior repair accuracy and
efficiency of our proposed ReCode.
Our contributions are summarized as follows:
•We propose ReCode, a retrieval-augmented generation method
that integrates algorithm-aware categorization and modular
modeling to enable effective code repair.
•We curate RACodeBench, a benchmark centered on real-
world user-submitted bugs, consisting of a large collection
of authentic buggy–fixed code pairs, to support realistic and
rigorous evaluation of code repair methods.
•We conduct systematic and comprehensive experiments,
demonstrating that our proposed ReCode not only maintains
strong repair performance but also significantly reduces in-
ference cost.
2 Related Work
In this section, we give a brief introduction to the current research
status of automatic program repair and in-context learning.

ReCode: Improving LLM-based Code Repair with Fine-Grained Retrieval-Augmented Generation Conference’17, July 2017, Washington, DC, USA
2.1 Automatic Program Repair
In addition to direct code generation, increasing attention has been
given to automatic program repair (APR), which seeks to detect
and fix code bugs automatically. Early APR approaches relied on
template-based and search-based strategies [ 15,21,31,40], but
their dependence on handcrafted rules limited generalization and
scalability.
With the advent of deep learning, APR has witnessed a para-
digm shift toward learning-based methods that model the repair
process as a translation task from buggy code to its corrected ver-
sion. Sequence-to-sequence models [ 20,43,45] trained on synthetic
buggy–fixed pairs [ 6,19,22] improved generalization across bug
types and languages, though the artificial–real bug gap remained a
key challenge. Recently, large language models have shown strong
zero- and few-shot capabilities in code repair [ 9,11,12,25], offering
new opportunities beyond supervised data.
Building on this, a growing body of research explores the ability
of LLMs to repair their own generated code, marking a key direc-
tion in self-improving AI systems. Wang et al . [39] and Burns et al .
[5]propose self-improvement mechanisms that leverage test case
execution or natural language feedback to iteratively refine model
outputs. More recent advancements have introduced structured
frameworks for integrating feedback and reasoning into the repair
process. Zhang et al . [42] and Olausson et al . [30] utilize execution
traces and error messages to guide the model in understanding
and correcting failures. Shinn et al . [35] extends this idea by using
memory-augmented reasoning loops, allowing the model to reflect
on previous attempts and improve over time. Zhong et al . [44] in-
troduces a program decomposition strategy, breaking programs
into basic blocks and monitoring runtime variable values after each
block to verify semantic correctness with respect to the original
specification. Similarly, [ 34] adopts a hierarchical repair framework
that separates the debugging process into low-level error localiza-
tion and high-level semantic repair, significantly improving repair
precision and interpretability.
Distinct from prior approaches, our method harnesses the in-
context learning ability of large language models, enabling efficient
and flexible code repair without model fine-tuning or reliance on
auxiliary components.
2.2 In-context Learning
Despite the effectiveness of supervised fine-tuning across various
tasks, it has several notable drawbacks. It typically requires access
to a large volume of high-quality annotated data, which may be
expensive or difficult to obtain. Additionally, the fine-tuning process
is computationally intensive and can lead to undesirable side effects,
such as catastrophic forgetting[ 28]. To address these challenges, in-
context learning has emerged as a promising alternative paradigm[ 3,
4,37]. In-context Learning enables LLMs to adapt to new tasks
by conditioning on a small number of demonstration examples
provided at inference time, without any modification to model
parameters. This makes it a flexible and cost-effective approach to
task generalization.
Recent studies explore the utility of in-context learning for code
repair. For instance, Yin et al . [41] proposes an automated code
self-correction framework that combines few-shot prompting withchain-of-thought (CoT) reasoning and execution feedback to itera-
tively guide the repair process. Agarwal et al . [2] also leverage CoT
rationales to enhance repair performance but intentionally omit
these rationales from the input context during few-shot prompting
in order to separate reasoning from learning signals. In another
approach, Mavalankar et al . [29] exploits the ICL capabilities of
large language models and employs a filtering mechanism to select
a small set of optimal repair examples generated by the model it-
self as in-context references. In contrast, our approach selects the
most relevant and authentic examples corresponding to the current
problem, rather than relying on model-generated repair instances.
3 Method
In this section, we first provide the problem formulation and con-
duct experiments to demonstrate the existing limitations of tradi-
tional retrieval-augmented methods. Then, we meticulously intro-
duce the framework of our ReCode. Finally, we elaborate on the
construction of our knowledge base and RACodeBench.
3.1 Problem Formulation
The code repair task is formally defined as a conditional generation
problem. Given a problem description 𝑞∈Q and its corresponding
defective code implementation 𝑥∈X, which may contain syntactic
or semantic errors, the objective is to learn a generative model 𝑓𝜃
capable of producing a corrected code 𝑦=𝑓𝜃(𝑞,𝑥). The generated
code must satisfy two fundamental requirements: it needs to be
syntactically well-formed by strictly conforming to the grammar
rules of the target programming language, and semantically correct
by accurately implementing the intended functionality specified in
𝑞, which is formally verified through a comprehensive set of test
casesT𝑞that serve as executable specifications for correctness.
Within the retrieval-augmented in-context learning framework,
an external support set D={(𝑞𝑖,𝑥𝑖,𝑦𝑖)}𝑁
𝑖=1is introduced to pro-
vide relevant auxiliary information. For a target input (𝑞,𝑥), a
retrieval functionR(𝑞,𝑥;D)identifies the most relevant example
from the support set D:
(𝑞∗,𝑥∗,𝑦∗)=R(𝑞,𝑥;D) (1)
Subsequently, this retrieved example is concatenated with the
target input to form the contextual prompt of the model:
Prompt =Concat (𝑞∗,𝑥∗,𝑦∗),(𝑞,𝑥)(2)
The model then generates the repaired code based on the con-
structed prompt:
𝑦=𝑓𝜃(Prompt)such that𝑦|=T𝑞 (3)
Here𝑦|=T𝑞denotes that the generated code 𝑦satisfies all test
cases inT𝑞, thus ensuring both syntactic validity and semantic
correctness.
A central challenge of this approach lies in the design of the
retrieval functionR, which must effectively identify examples that
are both semantically informative and structurally similar to the
target input, thus providing valuable guidance for the generation
process.

Conference’17, July 2017, Washington, DC, USA Yicong Zhao, Shisong Chen, Jiacheng Zhang, and Zhixu Li
/gid00021/gid00032/gid00051/gid00047
/gid00022/gid00046/gid00032/gid00045/gid00001/gid00018/gid00048/gid00032/gid00045/gid00052“Question: l”
/gid00004/gid00042/gid00031/gid00032
“Algorithm?”
/gid00005/gid00048/gid00028/gid00039/gid01162/gid00023/gid00036/gid00032/gid00050/gid00001/gid00007/gid00048/gid00046/gid00032/gid00031
/gid00007/gid00032/gid00028/gid00047/gid00048/gid00045/gid00032/gid00019/gid00032/gid00047/gid00045/gid00036/gid00032/gid00049/gid00028/gid00039
/gid00019/gid00032/gid00047/gid00045/gid00036/gid00032/gid00049/gid00028/gid00039
/gid00021/gid00032/gid00051/gid00047/gid00001/gid00006/gid00041/gid00030/gid00042/gid00031/gid00032/gid00045
/gid00004/gid00042/gid00031/gid00032/gid00001/gid00006/gid00041/gid00030/gid00042/gid00031/gid00032/gid00045/gid00013/gid00013/gid00014
/gid00013/gid00013/gid00014
Figure 2: Overview of our proposed ReCode Method. Given a user query, the model predicts multiple algorithm type labels
related to the buggy code. A dual encoding strategy extracts textual and code features to form a fused representation. Using
predicted labels, semantic retrieval is performed in parallel across sub-knowledge bases. Retrieved results are integrated via a
collaborative mechanism to build a semantically relevant in-context example set, which guides the model to generate accurate
code repairs.
3.2 Limitations of Traditional
Retrieval-Augmented Methods
Although retrieval-augmented in-context learning frameworks have
been empirically validated as effective in code repair tasks, conven-
tional retrieval mechanisms within these frameworks often rely on
a unified encoding strategy. This approach encodes the problem de-
scription and the associated buggy code as a single input, retrieving
support examples based on the overall similarity of their combined
representations. However, this retrieval paradigm exhibits several
notable limitations: it inadequately accounts for structural varia-
tions in code, fails to capture fine-grained error semantics, and often
lacks precision in identifying the intended repair operations. These
deficiencies reduce the contextual relevance and corrective utility
of the retrieved exemplars, ultimately limiting the effectiveness
of the repair process. As shown in Table 1, replacing the unified
encoding scheme with our proposed dual encoding strategy, which
independently models textual and code representations, leads to
consistent improvements in test pass rates on RACodeBench across
both the Gemma-2 [36] and GPT-4o-mini.
Table 1: Test Pass Rates of Unified vs. Dual Encoding Strate-
gies on RACodeBench
Encoding Strategy Test Pass Rate
Gemma-2-27B + Unified Encoding 26.25%
Gemma-2-27B + Dual Encoding 27.73 %
GPT-4o-mini + Unified Encoding 36.72%
GPT-4o-mini + Dual Encoding 38.49 %
Apart from encoding inefficiencies, existing traditional retrieval
methods often overlook the model’s inherent ability to understandthe input code. Instead, they rely solely on surface-level similarity,
which can lead to mismatches between the retrieved examples
and the actual repair needs. This lack of semantic and structural
alignment reduces the effectiveness of the retrieved examples and
limits the overall performance of the repair process.
3.3 ReCode Framework
To overcome the limitations inherent in traditional retrieval-augmented
approaches, we propose ReCode , a novel algorithm-aware frame-
work that integrates dual-view encoding with hybrid retrieval
mechanisms to enhance the precision and contextual relevance
of retrieved examples for code repair.
An overview of the ReCode framework is illustrated in Figure 2.
The overall process comprises two core modules:
Algorithm-Aware .The algorithm-aware module infers the un-
derlying algorithmic intent of a buggy code snippet by leveraging
the advanced reasoning and comprehension capabilities of LLMs to
analyze code structure and semantics beyond surface-level patterns.
Recognizing that real-world code frequently integrates multiple
algorithmic paradigms, such as graph traversal combined with
greedy heuristics, this module adopts a multi-label classification
approach, assigning multiple algorithm categories to each snippet.
This strategy effectively captures a broad spectrum of potential
error types associated with diverse algorithms. By exploiting the
intrinsic understanding of LLMs, the module achieves a more pre-
cise characterization of algorithmic signals, thereby improving the
relevance and efficacy of retrieved repair exemplars.
Dual-View Encoding .The dual-view encoding module employs
specialized encoders to independently process the two input modal-
ities. Specifically, the natural language description is encoded via
a dedicated text encoder designed to capture task-level semantic

ReCode: Improving LLM-based Code Repair with Fine-Grained Retrieval-Augmented Generation Conference’17, July 2017, Washington, DC, USA
intent, while the buggy code snippet is processed by a code encoder
tailored to preserve the syntactic and structural features intrinsic
to source code. Following this modality-specific encoding, a feature
fusion step integrates the textual and code representations into a
unified vector. This fusion synergistically combines complementary
information from both views while maintaining their distinctive
properties. In contrast to conventional unified encoding methods,
which concatenate text and code into a single sequence and often
result in modal interference and dilution of critical semantic or
structural cues, our approach preserves the unique characteristics
of each modality. This separation enables more precise modeling of
the inherent features of natural language and source code, yielding
fused representations that are both richer and more discrimina-
tive. Such enhanced expressiveness and modality-specific focus are
advantageous for the subsequent retrieval phase.
Given a user query, ReCode first decomposes the input into
two distinct components: a natural language description and a
buggy code snippet. These two modalities are then independently
processed through the dual-view encoding module to generate
rich and discriminative representations. Leveraging the algorithm-
aware module, ReCode predicts multiple algorithmic categories
relevant to the buggy code, which guides the retrieval process.
Based on these predictions, the hybrid retrieval module selectively
searches corresponding algorithm-specific sub-knowledge bases to
retrieve the most relevant repair exemplars. The retrieved examples
are then aggregated and ranked to form a final set of contextually
aligned support instances. Finally, these carefully selected examples
are provided as guidance to downstream models, enabling more
accurate and effective code repair.
3.4 Construction of RACodeBench and
Knowledge Bases
To support our proposed ReCode method, we construct two com-
plementary resources derived from the widely adopted compet-
itive programming platform Codeforces: a hierarchically struc-
tured knowledge base and a standardized evaluation benchmark,
RACodeBench. Both resources are built from large-scale collections
of real-world programming problems and user submissions, ensur-
ing representativeness and diversity across algorithmic domains.
Knowledge Base Construction .We build the knowledge base
from the historical archives of Codeforces, leveraging both problem
statements and associated user submissions. Using the original
algorithmic tags (e.g., greedy, binary search, graph, and dp), we
organize the knowledge base into a multi-level hierarchy. At the top
are broad algorithm categories, each containing numerous problems.
Under each problem, we curate paired samples of erroneous and
corrected submissions.
The erroneous submissions capture common syntactic and se-
mantic faults, while the corrected counterparts represent valid
solutions that pass all test cases. To enable fine-grained error re-
trieval and analysis, we employ a semi-automated pipeline that
performs differential analysis between buggy and fixed submis-
sions, enriched with compiler diagnostics and execution outcomes.
Each pair is annotated with detailed error types, enabling precise
access to error-specific and algorithm-specific examples within the
knowledge base.RACodeBench Construction .To comprehensively evaluate
code repair models, we construct RACodeBench, a large-scale, diag-
nostic benchmark built upon the historical archives of Codeforces.
RACodeBench is specifically designed to capture a wide range of
real-world programming errors and algorithmic challenges, en-
abling fine-grained and algorithm-aware assessment of model per-
formance. Each instance in RACodeBench consists of the following
elements: (1) the natural language description of a programming
problem, (2) an erroneous user-submitted code that fails due to
compilation or runtime/test-case errors, (3) the corrected version
of the submission that passes all test cases, (4) the full set of public
test cases provided by Codeforces, (5) a fine-grained annotation of
the error type (e.g., syntax error, incorrect loop boundary), and (6)
a code-level diff highlighting the exact changes between the buggy
and repaired versions.
To enable comprehensive analysis and systematic comparison,
all error annotations are generated through a meticulously designed
semi-automated pipeline. This process involves: (1) conducting syn-
tactic and semantic differencing between erroneous and corrected
code segments, (2) integrating compiler diagnostics and test execu-
tion outputs, and (3) performing manual verification on a carefully
selected representative subset to guarantee annotation quality and
consistency. These multi-faceted annotations transform the bench-
mark into not merely a quantitative metric for test pass rates, but
more importantly, a sophisticated diagnostic tool for fine-grained
analysis of model behavior. During dataset construction, we im-
plement rigorous measures to ensure diversity and balance across
multiple dimensions. The problem set encompasses a comprehen-
sive spectrum of algorithmic domains, including but not limited
to dynamic programming, greedy algorithms, graph traversal, and
number theory. Problem difficulty is systematically stratified ac-
cording to Codeforces ratings, ensuring balanced evaluation across
both fundamental programming concepts and advanced algorith-
mic challenges. Furthermore, we employ stratified sampling of error
types to mitigate distributional bias, thereby preventing any single
repair category from disproportionately influencing the benchmark
results.
To maintain experimental rigor and preclude potential data con-
tamination, RACodeBench implements strict partitioning between
the benchmark dataset and the retrieval knowledge base. This par-
titioning guarantees that no problem instances or user submissions
appearing in the evaluation set are included in the in-context learn-
ing knowledge base. Such stringent separation preserves the in-
tegrity of performance assessment by simulating authentic code
repair scenarios, where models must address genuinely novel prob-
lems. This design principle ensures that evaluation results accu-
rately reflect model capabilities in real-world application contexts.
4 Experiments
4.1 Experimental Setup
Datasets: We perform in-distribution evaluation on our newly
constructed dataset, RACodeBench , which contains buggy code
samples meticulously annotated with corresponding algorithmic
error types. To further assess the out-of-distribution generalization
capability of our method, we conduct evaluations on six additional
competitive programming datasets: (1) AtCoder (800 problems),

Conference’17, July 2017, Washington, DC, USA Yicong Zhao, Shisong Chen, Jiacheng Zhang, and Zhixu Li
ModelTest Pass Rate(%) Strict Accuracy(%)
Best-of-𝑁 Self-repair Ours Best-of-𝑁 Self-repair Ours
Open-sourceGemma-2-9B 23.34 22.67 26.07 15.83 15.00 16.67
Gemma-2-27B 24.96 22.74 29.83 17.08 15.83 19.17
DeepSeek-V2-Chat 30.28 29.32 38.04 19.17 22.08 26.25
DeepSeek-Coder-V2-Instruct 31.54 33.08 40.47 20.42 23.17 28.75
Closed-sourceGemini-1.5-Flash 28.88 33.90 37.96 19.17 23.75 26.67
GPT-4o-mini 31.09 34.79 41.06 21.25 24.58 30.41
Table 2: Code repair performance on RACodeBench. With 𝑁=8inference calls, we evaluate best-of- 𝑁, self-repair, and ReCode
using test pass rate and strict accuracy metrics. Across all tested models, ReCode consistently achieves superior performance.
(2) CodeChef (2.4k problems), (3) HackerRank (720 problems), (4)
HackerEarth (1k problems), (5) GeeksforGeeks (680 problems), and
(6) Aizu (1.8k problems). These benchmarks collectively cover a
wide spectrum of problem domains, programming languages, and
difficulty levels, providing a rigorous and comprehensive testbed for
evaluating the robustness and generalization of code repair models.
Models: For retrieval, we use OASIS-code-1.3B[ 14] as the code
encoder and bge-m3[ 8] as the text encoder. For inference, we eval-
uate our approach using a set of open-source and closed-source
LLMs, including Gemma-2-9B, Gemma-2-27B[ 36], GPT-4o-mini,
Gemini-1.5-Flash, DeepSeek-V2-Chat[ 26], and DeepSeek-Coder-
V2-Instruct[46].
Metrics: The most direct and reliable way to evaluate code repair
performance is by executing the generated code against test cases.
We adopt and extend two commonly used metrics—test pass rate
and strict accuracy [ 18]—to provide a comprehensive assessment
of model effectiveness. To mitigate variability in model outputs, we
consider the best result among the 𝑁candidate solutions generated
by the language models.
Formally, let 𝑃denote the total number of problems. For each
problem𝑝, letcode𝑝𝑗represent the 𝑗-th generated candidate solu-
tion. Each problem is associated with a set of 𝐾test casesT𝑝=
(𝑥𝑝,1,𝑦𝑝,1),...,(𝑥𝑝,𝐾,𝑦𝑝,𝐾), where𝑥𝑝,𝑘is the input and 𝑦𝑝,𝑘is
the expected output.
Thetest pass rate is defined as the average proportion of test
cases passed across all problems:
1
𝑃𝑃∑︁
𝑝=1max
1≤𝑗≤𝑁1
|𝐾||𝐾|∑︁
𝑘=11n
eval(code𝑗
𝑝,𝑥𝑝,𝑘)==𝑦𝑝,𝑘o
(4)
Here, the inner summation computes the fraction of test cases
passed by the 𝑗-th candidate for problem 𝑝, and the outer summa-
tion averages this score over all problems. The max operator selects
the best-performing candidate among the 𝑁generated solutions.
Thestrict accuracy metric grants credit only if a candidate
passes all test cases for a given problem:
1
𝑃𝑃∑︁
𝑝=1max
1≤𝑗≤𝑁|𝐾|Ö
𝑘=11n
eval(code𝑗
𝑝,𝑥𝑝,𝑘)==𝑦𝑝,𝑘o
(5)
In this expression, the product equals 1 only if the candidate
passes every test case for problem 𝑝, and 0 otherwise. Similar to
the test pass rate, the maximum over candidates is taken to identify
the best attempt, and the average is computed over all problems.Baselines: We compare ReCode with two representative base-
lines: best-of- 𝑁[27] and self-repair [ 30]. The best-of- 𝑁leverages
multiple LLM calls at inference time to enhance performance by
generating𝑁diverse candidate completions and selecting the one
that achieves the best downstream evaluation result. To promote
diversity among the generated candidates, we set the sampling
temperature to 1.0, following established practice [ 32]. The self-
repair adopts a multi-stage strategy in which the total LLM calls at
inference time are distributed across a sequence of generation and
refinement steps. Specifically, a portion of the LLM calls is allocated
to generating natural language feedback that identifies potential
issues in the initial code, while the remaining calls are used to
produce revised code conditioned on this feedback. In ReCode, the
inference-time budget is similarly divided, with part of the calls used
for algorithm classification and the rest for retrieval-augmented
code generation. We adopt different values of 𝑁depending on the
evaluation setting. For in-distribution experiments, where buggy
code is available, we set 𝑁=8. For out-of-distribution experiments,
where the model must first synthesize an initial solution before
performing repair, we increase 𝑁to 32 to account for the additional
generation complexity.
4.2 Main Results
We report the core experimental results across both in-distribution
and out-of-distribution settings. Detailed experimental results are
summarized in Table 2 for the in-distribution setting, and illustrated
in Figure 3 and Figure 4 for the out-of-distribution evaluation.
As shown in Table 2, the in-distribution experimental results
on RACodeBench demonstrate that our method consistently out-
performs competing approaches across all models and evaluation
metrics. This superior performance stems from several key factors
intrinsic to our approach. First, by effectively integrating exter-
nal, domain-specific knowledge from specific algorithmic domains
through a retrieval mechanism, our method provides precise and
contextually relevant guidance that purely generative methods rely-
ing solely on implicit model knowledge often fail to capture. Second,
leveraging a hierarchically structured knowledge base organized
by algorithmic categories allows the model to access tailored exem-
plars that closely match the semantic and structural patterns of the
target problem, enabling more targeted and semantically informed
code repairs. Finally, the combination of retrieval and generation
creates a complementary synergy where external knowledge en-
riches the model’s internal representations, leading to enhanced

ReCode: Improving LLM-based Code Repair with Fine-Grained Retrieval-Augmented Generation Conference’17, July 2017, Washington, DC, USA
Figure 3: Test pass rates on out-of-distribution code repair. With 𝑁=32LLM calls at inference time, results using GPT-4o-
mini show significant improvements across several common out-of-distribution competitive programming datasets, with our
approach consistently outperforming all baseline methods, demonstrating its strong generalization capability.
Figure 4: Strict accuracy on out-of-distribution code repair. With 𝑁=32LLM calls at inference time, results using GPT-4o-mini
demonstrate that our approach strongly solves competitive programming problems, consistently outperforming all baseline
methods.
generalization and robustness in correcting a diverse spectrum of
complex programming errors.
Notably, when scaling from Gemma-2-9B to Gemma-2-27B, base-
line methods show only modest improvements: Best-of- 𝑁increases
by 6.94% (from 23.34 to 24.96), and Self-repair by just 0.3% (from
22.67 to 22.74). In contrast, our retrieval-augmented approach im-
proves significantly by 14.42% (from 26.07 to 29.83). These results
suggest that while larger models do enhance performance, such
improvements are relatively limited for methods that do not incor-
porate external context. However, when augmented with retrieval,
larger models, thanks to their enhanced in-context learning capac-
ity, can better exploit external knowledge. By retrieving targeted
exemplars from a hierarchical knowledge base, our approach en-
ables deeper contextual understanding. This demonstrates that the
benefit of model scaling is amplified when coupled with retrieval-
based augmentation.
Further analysis of the DeepSeek series reveals a clear trend:
DeepSeek-V2-Chat, a general-purpose model, demonstrates a more
substantial performance improvement from our approach thanDeepSeek-Coder-V2, which is further pre-trained from an interme-
diate checkpoint of DeepSeek-V2 with code-specific objectives. This
indicates that models lacking explicit adaptation to code-related
tasks benefit more from the external retrieval mechanism. These
findings suggest that our method is particularly effective for models
with limited inherent code understanding, as the retrieved exem-
plars can compensate for their weaker internal representations and
provide stronger task-specific guidance.
Encouraged by the strong performance of our method in in-
distribution experiments, we further evaluate its generalizability
through out-of-distribution experiments. These experiments assess
the efficacy of our approach by retrieving examples from Code-
forces and testing on a distinct competition code dataset, thereby
simulating real-world domain shifts.
Figure 3 and Figure 4 present a comparative analysis of our pro-
posed method against two baseline approaches, evaluated using
the GPT-4o-mini model across test pass rate and strict accuracy
metrics. The notable improvement in test pass rate validates the
effectiveness of retrieving repair examples aligned with similar

Conference’17, July 2017, Washington, DC, USA Yicong Zhao, Shisong Chen, Jiacheng Zhang, and Zhixu Li
Figure 5: Impact of example selection. Selecting examples
relevant to the current problem leads to consistently better
performance under limited inference budgets (up to 8 LLM
calls in this setting).
algorithmic categories. This targeted retrieval strategy significantly
enhances the model’s ability to correct and improve initial solutions,
even under out-of-distribution conditions. The advantage arises
from algorithm classification-driven exemplar selection, which en-
sures provision of contextually and algorithmically relevant repair
instances, thereby enabling robust generalization across varied
datasets by leveraging common algorithmic patterns despite dif-
ferences in problem specifics or data distribution. Likewise, our
approach surpasses baselines in strict accuracy, reflecting the pro-
portion of fully correct solutions passing all test cases, demonstrat-
ing its capability to generate complete and reliable solutions. This
improvement stems from enriched contextual guidance that aids
the model in addressing subtle errors and edge cases crucial for full
correctness. Importantly, these gains highlight the dual benefits of
our retrieval-augmented framework in enhancing both incremen-
tal repair quality and overall solution robustness, particularly in
out-of-distribution scenarios.
4.3 Impact of Example Selection
We conducted an ablation study to rule out the possibility that
performance gains in code repair tasks arise merely from the use
of examples, regardless of their relevance. In the baseline setting,
examples were randomly sampled from the entire pool. To ensure
a fair comparison, the additional LLM call for algorithm classifica-
tion was omitted in this setting. Experimental results(see Figure 5)
demonstrate that our method, which retrieves examples highly rele-
vant to the current problem, significantly outperforms the baseline.
This performance gap underscores the importance of contextual
and algorithmic alignment in example selection. Relevant examples
often share underlying algorithmic patterns and problem-solving
structures with the target problem, enabling the model to make
more accurate and efficient repairs. In contrast, randomly sampled
examples may better reflect the general task format but lack the
specific structural cues necessary for precise correction.
Furthermore, under a limited LLM inference budget, the baseline
exhibits only slow, marginal gains, whereas our method rapidly
converges to high performance. This reflects that retrieving highly
relevant examples offers immediate, effective guidance, enabling
Figure 6: Comparative Analysis of Repair Performance and
Inference Efficiency Across Methods Under In-Distribution
Settings (using GPT-4o-mini). (a) Test Pass Rate metric.(b)
Strict Accuracy metric.
efficient repair within fewer inference steps. Such early conver-
gence underscores the efficiency and efficacy of targeted retrieval
in resource-constrained settings.
4.4 Inference Cost Analysis
In real-world automated code repair systems, inference efficiency
is as critical as repair accuracy. High computational overhead in-
creases response latency and limits system scalability and deploy-
ment. To address this, we systematically analyze the inference effi-
ciency of Best-of- 𝑁, Self-Repair, and our proposed method, ReCode,
under both in-distribution and out-of-distribution scenarios.
In the in-distribution experiments with a generation budget of
𝑁=8calls, Best-of- 𝑁independently samples 8 candidate fixes and
selects the best, Self-Repair performs iterative corrections, and Re-
Code generates a high-quality repair in a single pass using context
exemplars. Results show that ReCode performs slightly worse at
the initial call ( 𝑁=1) due to its initial algorithm type identification
step. However, from 𝑁=2onward, ReCode rapidly surpasses the
others in test pass rate, effectively halving inference cost. At 𝑁=4,
ReCode achieves comparable performance to others at 𝑁=8, effec-
tively halving inference cost. Strict accuracy metrics confirm this
trend, with ReCode reaching the upper bound of other methods by
𝑁=5, demonstrating efficient repair quality with reduced cost.
These performance differences arise because Self-Repair suffers
from error accumulation during iterations, and Best-of- 𝑁explores
a large search space with many low-quality candidates, leading to
inefficiency. In contrast, ReCode leverages carefully selected context
exemplars to guide the model’s focus, enabling fast convergence to
high-quality solutions and balancing accuracy with inference cost.
On the more challenging out-of-distribution AtCoder dataset, we
measure the number of model calls needed to reach specific test pass
rate thresholds. Starting from 33.72%, ReCode requires significantly
fewer calls—for example, only 4 calls to reach 35.0%, whereas Best-
of-𝑁and Self-Repair need 11 and 15 calls, respectively—reducing
inference overhead by roughly 3 to 4 times. Furthermore, ReCode
delivers larger performance gains per call, showcasing its efficient
and targeted repair strategy.
In summary, ReCode achieves competitive or better repair accu-
racy while significantly lowering inference computation, making it
more practical for real-world automated code repair applications.

ReCode: Improving LLM-based Code Repair with Fine-Grained Retrieval-Augmented Generation Conference’17, July 2017, Washington, DC, USA
/gid00022/gid00046/gid00032/gid00045/gid00001/gid00018/gid00048/gid00032/gid00045/gid00052 /gid00022/gid00046/gid00032/gid00045/gid00001/gid00004/gid00042/gid00031/gid00032
Given a string s, in each operation you 
can delete a contiguous substring of 
identical characters, and the score is 
the length of the substring. The goal is 
to find the minimum number of operations   
required to delete the entire string.
/gid00019/gid00032/gid00047/gid00045/gid00036/gid00032/gid00049/gid00032/gid00031/gid00001/gid00004/gid00042/gid00031/gid00032/gid00001/gid00017/gid00028/gid00036/gid00045
/gid00008/gid00017/gid00021/gid00469/gid00001/gid00019/gid00032/gid00046/gid00048/gid00039/gid00047for (int i = 1; i <= n; i++) {
    if (i >= a && dp[i - a] != -1) { 
  dp[i] = max(dp[i], dp[i - a] + 1); }
    if (i >= b && dp[i - b] != -1) { 
  dp[i] = max(dp[i], dp[i - b] + 1); }
}
/gid00016/gid00048/gid00045/gid00046for (int len = 1; len <= n; len++) {
    for (int l = 0; l + len - 1 < n; l++) {
        int r = l + len - 1;
        if (l == r) { dp[l][r] = 1; }
        else {
            dp[l][r] = 1 + dp[l + 1][r];
            for (int i = l + 1; i <= r; i++) {
                if (s[l] == s[i]) { dp[l][r] = 
     min(dp[l][r], dp[i][r]);  }}}0
1
2
3
4
5
6
7
0
1
2
3
4
5
6
7
80
1
2
3
4
5
6
7
8
9
10
110
1
2
3
4
5
60
1
2
3
4
5
6
for (int len = 1; len <= n; len++) {
    for (int l = 0; l + len - 1 < n; l++) {
        int r = l + len - 1;
        if (l == r) dp[l][r] = 1;
        else {
            dp[l][r] = 1 + dp[l + 1][r];
            for (int i = l + 1; i <= r; i++)
                if (s[l] == s[i]) {
                    if (i - 1 >= l + 1) {
                        dp[l][r] = min(dp[l][r], 
        dp[l +1][i - 1] + dp[i][r]); }
                    else {
                        dp[l][r] = min(dp[l][r],    
      dp[i][r]); }}}for (int len = 1; len <= n; len++) {
    for (int l = 0; l + len - 1 < n; l++) {
        int r = l + len - 1;
        if (l == r) dp[l][r] = 1;
        else {
            dp[l][r] = 1 + dp[l + 1][r];
            for (int i = l + 1; i <= r; i++) {
                if (s[l] == s[i]) {
                    dp[l][r] = min(dp[l][r],    
    dp[i][r]) + dp[l + 1][i - 1];}}}for (int i = 1; i <= n; i++) {
    if (i >= a && dp[i - a] != -1) { 
  dp[i] = max(dp[i], dp[i - a] + 1); }
    if (i >= b && dp[i - b] != -1) { 
  dp[i] = max(dp[i], dp[i - b] + 1); }
 if (i >= c && dp[i - c] != -1) {
  dp[i] = max(dp[i], dp[i - c] + 1); }}
Figure 7: Qualitative Results of the ReCode method on our RACodeBench Benchmark.
Table 3: Inference Cost to Reach Test Pass Rate Thresholds
on AtCoder, evaluated with GPT-4o-mini.
Test Pass Rate(%)Number of LLM Calls
Best-of-𝑁Self-Repair ReCode
35.0 11 15 4
35.5 12 15 4
36.0 16 19 5
36.5 16 21 5
37.0 18 24 5
37.5 21 27 6
4.5 Quality Result
To further demonstrate our method’s effectiveness, Figure 7 presents
a representative example. In this case, GPT-4o-mini attempts a direct
repair but produces an incomplete solution. By contrast, ReCode
retrieves a high-quality exemplar with a dynamic programming
pattern and successfully transfers its underlying logic to the user’s
code, resulting in a functionally complete repair. Moreover, ReCode
not only ensures semantic correctness but also preserves the user’soriginal coding style and structure, underscoring the advantages of
exemplar-guided generation.
5 Conclusions
We propose ReCode, a fine-grained retrieval-augmented genera-
tion framework that leverages modular code representations and
an algorithm-aware retrieval strategy to enhance automated code
repair. ReCode improves the relevance of retrieved exemplars and
enables accurate, efficient repair without additional training. Exten-
sive experiments on RACodeBench and competitive programming
datasets show that ReCode outperforms existing methods in both
repair accuracy and inference efficiency. This work demonstrates
the practical advantages of integrating retrieval with large language
models for scalable and adaptive code repair.
Acknowledgments
This work is supported by the Key-Area Research and Develop-
ment Program of Guangdong Province (2024B0101050005), Suzhou
Key Laboratory of Artificial Intelligence and Social Governance
Technologies (SZS2023007), Smart Social Governance Technology
and Innovative Application Platform (YZCXPT2023101), and the
Leadership Talent Program (Science and Education) of SIP.

Conference’17, July 2017, Washington, DC, USA Yicong Zhao, Shisong Chen, Jiacheng Zhang, and Zhixu Li
GenAI Usage Disclosure
During the preparation of this work, the authors utilized GPT-4 for
language refinement and sentence polishing. After using this tool,
the authors reviewed and edited the content as needed and take
full responsibility for the content of the publication.
References
[1]Josh Achiam, Steven Adler, Sandhini Agarwal, Lama Ahmad, Ilge Akkaya, Floren-
cia Leoni Aleman, Diogo Almeida, Janko Altenschmidt, Sam Altman, Shyamal
Anadkat, et al .2023. Gpt-4 technical report. arXiv preprint arXiv:2303.08774
(2023).
[2]Rishabh Agarwal, Avi Singh, Lei Zhang, Bernd Bohnet, Luis Rosias, Stephanie
Chan, Biao Zhang, Ankesh Anand, Zaheer Abbas, Azade Nova, et al .2024. Many-
shot in-context learning. Advances in Neural Information Processing Systems 37
(2024), 76930–76966.
[3] Ekin Akyürek, Dale Schuurmans, Jacob Andreas, Tengyu Ma, and Denny Zhou.
2022. What learning algorithm is in-context learning? investigations with linear
models. arXiv preprint arXiv:2211.15661 (2022).
[4]Tom Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared D Kaplan,
Prafulla Dhariwal, Arvind Neelakantan, Pranav Shyam, Girish Sastry, Amanda
Askell, et al .2020. Language models are few-shot learners. Advances in neural
information processing systems 33 (2020), 1877–1901.
[5]Collin Burns, Pavel Izmailov, Jan Hendrik Kirchner, Bowen Baker, Leo Gao,
Leopold Aschenbrenner, Yining Chen, Adrien Ecoffet, Manas Joglekar, Jan Leike,
et al.2023. Weak-to-strong generalization: Eliciting strong capabilities with weak
supervision. arXiv preprint arXiv:2312.09390 (2023).
[6]Saikat Chakraborty, Yangruibo Ding, Miltiadis Allamanis, and Baishakhi Ray.
2020. Codit: Code editing with tree-based neural models. IEEE Transactions on
Software Engineering 48, 4 (2020), 1385–1399.
[7]Angelica Chen, Jérémy Scheurer, Tomasz Korbak, Jon Ander Campos, Jun Shern
Chan, Samuel R Bowman, Kyunghyun Cho, and Ethan Perez. 2023. Improving
code generation by training with natural language feedback. arXiv preprint
arXiv:2303.16749 (2023).
[8]Jianlv Chen, Shitao Xiao, Peitian Zhang, Kun Luo, Defu Lian, and Zheng Liu. 2024.
Bge m3-embedding: Multi-lingual, multi-functionality, multi-granularity text
embeddings through self-knowledge distillation. arXiv preprint arXiv:2402.03216
(2024).
[9]Mark Chen, Jerry Tworek, Heewoo Jun, Qiming Yuan, Henrique Ponde
De Oliveira Pinto, Jared Kaplan, Harri Edwards, Yuri Burda, Nicholas Joseph,
Greg Brockman, et al .2021. Evaluating large language models trained on code.
arXiv preprint arXiv:2107.03374 (2021).
[10] Xinyun Chen, Maxwell Lin, Nathanael Schärli, and Denny Zhou. 2023. Teaching
large language models to self-debug. arXiv preprint arXiv:2304.05128 (2023).
[11] Aakanksha Chowdhery, Sharan Narang, Jacob Devlin, Maarten Bosma, Gaurav
Mishra, Adam Roberts, Paul Barham, Hyung Won Chung, Charles Sutton, Se-
bastian Gehrmann, et al .2023. Palm: Scaling language modeling with pathways.
Journal of Machine Learning Research 24, 240 (2023), 1–113.
[12] Daniel Fried, Armen Aghajanyan, Jessy Lin, Sida Wang, Eric Wallace, Freda Shi,
Ruiqi Zhong, Wen-tau Yih, Luke Zettlemoyer, and Mike Lewis. 2022. Incoder: A
generative model for code infilling and synthesis. arXiv preprint arXiv:2204.05999
(2022).
[13] Cheng Fu, Huili Chen, Haolan Liu, Xinyun Chen, Yuandong Tian, Farinaz
Koushanfar, and Jishen Zhao. 2019. Coda: An end-to-end neural program decom-
piler. Advances in Neural Information Processing Systems 32 (2019).
[14] Zuchen Gao, Zizheng Zhan, Xianming Li, Erxin Yu, Haotian Zhang, Bin Chen,
Yuqun Zhang, and Jing Li. 2025. OASIS: Order-Augmented Strategy for Improved
Code Search. arXiv preprint arXiv:2503.08161 (2025).
[15] Ali Ghanbari, Samuel Benton, and Lingming Zhang. 2019. Practical program re-
pair via bytecode mutation. In Proceedings of the 28th ACM SIGSOFT International
Symposium on Software Testing and Analysis . 19–30.
[16] Daya Guo, Dejian Yang, Haowei Zhang, Junxiao Song, Ruoyu Zhang, Runxin
Xu, Qihao Zhu, Shirong Ma, Peiyi Wang, Xiao Bi, et al .2025. Deepseek-r1:
Incentivizing reasoning capability in llms via reinforcement learning. arXiv
preprint arXiv:2501.12948 (2025).
[17] Kavi Gupta, Peter Ebert Christensen, Xinyun Chen, and Dawn Song. 2020. Syn-
thesize, execute and debug: Learning to repair for neural program synthesis.
Advances in Neural Information Processing Systems 33 (2020), 17685–17695.
[18] Dan Hendrycks, Steven Basart, Saurav Kadavath, Mantas Mazeika, Akul Arora,
Ethan Guo, Collin Burns, Samir Puranik, Horace He, Dawn Song, et al .2021. Mea-
suring coding challenge competence with apps. arXiv preprint arXiv:2105.09938
(2021).
[19] Yaojie Hu, Xingjian Shi, Qiang Zhou, and Lee Pike. 2022. Fix bugs with trans-
former through a neural-symbolic edit grammar. arXiv preprint arXiv:2204.06643
(2022).[20] Kai Huang, Zhengzi Xu, Su Yang, Hongyu Sun, Xuejun Li, Zheng Yan, and Yuqing
Zhang. 2023. A survey on automated program repair techniques. arXiv preprint
arXiv:2303.18184 (2023).
[21] Jiajun Jiang, Yingfei Xiong, Hongyu Zhang, Qing Gao, and Xiangqun Chen.
2018. Shaping program repair space with existing patches and similar code. In
Proceedings of the 27th ACM SIGSOFT international symposium on software testing
and analysis . 298–309.
[22] Nan Jiang, Thibaud Lutellier, and Lin Tan. 2021. Cure: Code-aware neural machine
translation for automatic program repair. In 2021 IEEE/ACM 43rd International
Conference on Software Engineering (ICSE) . IEEE, 1161–1173.
[23] Hung Le, Yue Wang, Akhilesh Deepak Gotmare, Silvio Savarese, and Steven
Chu Hong Hoi. 2022. Coderl: Mastering code generation through pretrained
models and deep reinforcement learning. Advances in Neural Information Pro-
cessing Systems 35 (2022), 21314–21328.
[24] Raymond Li, Loubna Ben Allal, Yangtian Zi, Niklas Muennighoff, Denis Kocetkov,
Chenghao Mou, Marc Marone, Christopher Akiki, Jia Li, Jenny Chim, et al .2023.
Starcoder: may the source be with you! arXiv preprint arXiv:2305.06161 (2023).
[25] Yujia Li, David Choi, Junyoung Chung, Nate Kushman, Julian Schrittwieser, Rémi
Leblond, Tom Eccles, James Keeling, Felix Gimeno, Agustin Dal Lago, et al .2022.
Competition-level code generation with alphacode. Science 378, 6624 (2022),
1092–1097.
[26] Aixin Liu, Bei Feng, Bin Wang, Bingxuan Wang, Bo Liu, Chenggang Zhao,
Chengqi Dengr, Chong Ruan, Damai Dai, Daya Guo, et al .2024. Deepseek-
v2: A strong, economical, and efficient mixture-of-experts language model. arXiv
preprint arXiv:2405.04434 (2024).
[27] Fei Liu et al .2020. Learning to summarize from human feedback. In Proceedings of
the 58th Annual Meeting of the Association for Computational Linguistics . 583–592.
[28] Yun Luo, Zhen Yang, Fandong Meng, Yafu Li, Jie Zhou, and Yue Zhang. 2023.
An empirical study of catastrophic forgetting in large language models during
continual fine-tuning. arXiv preprint arXiv:2308.08747 (2023).
[29] Aditi Mavalankar, Hassan Mansoor, Zita Marinho, Masha Samsikova, and Tom
Schaul. 2025. AuPair: Golden Example Pairs for Code Repair. arXiv preprint
arXiv:2502.18487 (2025).
[30] Theo X Olausson, Jeevana Priya Inala, Chenglong Wang, Jianfeng Gao, and
Armando Solar-Lezama. 2023. Is self-repair a silver bullet for code generation?
arXiv preprint arXiv:2306.09896 (2023).
[31] Yun Peng, Shuzheng Gao, Cuiyun Gao, Yintong Huo, and Michael Lyu. 2024.
Domain knowledge matters: Improving prompts with fix templates for repairing
python type errors. In Proceedings of the 46th ieee/acm international conference
on software engineering . 1–13.
[32] Matthew Renze. 2024. The effect of sampling temperature on problem solving in
large language models. In Findings of the Association for Computational Linguistics:
EMNLP 2024 . 7346–7356.
[33] Baptiste Roziere, Jonas Gehring, Fabian Gloeckle, Sten Sootla, Itai Gat, Xiao-
qing Ellen Tan, Yossi Adi, Jingyu Liu, Romain Sauvestre, Tal Remez, et al .2023.
Code llama: Open foundation models for code. arXiv preprint arXiv:2308.12950
(2023).
[34] Yuling Shi, Songsong Wang, Chengcheng Wan, and Xiaodong Gu. 2024. From
code to correctness: Closing the last mile of code generation with hierarchical
debugging. arXiv preprint arXiv:2410.01215 (2024).
[35] Noah Shinn, Federico Cassano, Beck Labash, Ashwin Gopinath, Karthik
Narasimhan, and Shunyu Yao. 2023. Reflexion: Language agents with verbal
reinforcement learning, 2023. URL https://arxiv. org/abs/2303.11366 (2023).
[36] Gemma Team, Morgane Riviere, Shreya Pathak, Pier Giuseppe Sessa, Cassidy
Hardin, Surya Bhupatiraju, Léonard Hussenot, Thomas Mesnard, Bobak Shahriari,
Alexandre Ramé, et al .2024. Gemma 2: Improving open language models at a
practical size. arXiv preprint arXiv:2408.00118 (2024).
[37] Johannes Von Oswald, Eyvind Niklasson, Ettore Randazzo, João Sacramento,
Alexander Mordvintsev, Andrey Zhmoginov, and Max Vladymyrov. 2023. Trans-
formers learn in-context by gradient descent. In International Conference on
Machine Learning . PMLR, 35151–35174.
[38] Xuezhi Wang, Jason Wei, Dale Schuurmans, Quoc Le, Ed Chi, Sharan Narang,
Aakanksha Chowdhery, and Denny Zhou. 2022. Self-consistency improves chain
of thought reasoning in language models. arXiv preprint arXiv:2203.11171 (2022).
[39] Yizhong Wang, Yeganeh Kordi, Swaroop Mishra, Alisa Liu, Noah A Smith, Daniel
Khashabi, and Hannaneh Hajishirzi. 2022. Self-instruct: Aligning language mod-
els with self-generated instructions. arXiv preprint arXiv:2212.10560 (2022).
[40] Deheng Yang, Xiaoguang Mao, Liqian Chen, Xuezheng Xu, Yan Lei, David Lo,
and Jiayu He. 2022. Transplantfix: Graph differencing-based code transplantation
for automated program repair. In Proceedings of the 37th IEEE/ACM International
Conference on Automated Software Engineering . 1–13.
[41] Xin Yin, Chao Ni, Shaohua Wang, Zhenhao Li, Limin Zeng, and Xiaohu Yang.
2024. Thinkrepair: Self-directed automated program repair. In Proceedings of the
33rd ACM SIGSOFT International Symposium on Software Testing and Analysis .
1274–1286.
[42] Kechi Zhang, Zhuo Li, Jia Li, Ge Li, and Zhi Jin. 2023. Self-edit: Fault-aware code
editor for code generation. arXiv preprint arXiv:2305.04087 (2023).

ReCode: Improving LLM-based Code Repair with Fine-Grained Retrieval-Augmented Generation Conference’17, July 2017, Washington, DC, USA
[43] Quanjun Zhang, Chunrong Fang, Yuxiang Ma, Weisong Sun, and Zhenyu Chen.
2023. A survey of learning-based automated program repair. ACM Transactions
on Software Engineering and Methodology 33, 2 (2023), 1–69.
[44] Li Zhong, Zilong Wang, and Jingbo Shang. 2024. Debug like a human: A large
language model debugger via verifying runtime execution step-by-step. arXiv
preprint arXiv:2402.16906 (2024).[45] Wenkang Zhong, Chuanyi Li, Jidong Ge, and Bin Luo. 2022. Neural program
repair: Systems, challenges and solutions. In Proceedings of the 13th Asia-Pacific
symposium on internetware . 96–106.
[46] Qihao Zhu, Daya Guo, Zhihong Shao, Dejian Yang, Peiyi Wang, Runxin Xu,
Y Wu, Yukun Li, Huazuo Gao, Shirong Ma, et al .2024. Deepseek-coder-v2:
Breaking the barrier of closed-source models in code intelligence. arXiv preprint
arXiv:2406.11931 (2024).