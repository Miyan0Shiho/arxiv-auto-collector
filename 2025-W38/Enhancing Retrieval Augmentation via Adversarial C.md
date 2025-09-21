# Enhancing Retrieval Augmentation via Adversarial Collaboration

**Authors**: Letian Zhang, Guanghao Meng, Xudong Ren, Yiming Wang, Shu-Tao Xia

**Published**: 2025-09-18 08:54:20

**PDF URL**: [http://arxiv.org/pdf/2509.14750v1](http://arxiv.org/pdf/2509.14750v1)

## Abstract
Retrieval-augmented Generation (RAG) is a prevalent approach for
domain-specific LLMs, yet it is often plagued by "Retrieval Hallucinations"--a
phenomenon where fine-tuned models fail to recognize and act upon poor-quality
retrieved documents, thus undermining performance. To address this, we propose
the Adversarial Collaboration RAG (AC-RAG) framework. AC-RAG employs two
heterogeneous agents: a generalist Detector that identifies knowledge gaps, and
a domain-specialized Resolver that provides precise solutions. Guided by a
moderator, these agents engage in an adversarial collaboration, where the
Detector's persistent questioning challenges the Resolver's expertise. This
dynamic process allows for iterative problem dissection and refined knowledge
retrieval. Extensive experiments show that AC-RAG significantly improves
retrieval accuracy and outperforms state-of-the-art RAG methods across various
vertical domains.

## Full Text


<!-- PDF content starts -->

ENHANCING RETRIEV AL AUGMENTATION VIA ADVERSARIAL COLLABORATION
Letian Zhang⋆†Guanghao Meng⋆†Xudong Ren†Yiming Wang‡Shu-Tao Xia†
†Tsinghua University‡Huawei Technologies Ltd.
Retrieval-augmented Generation (RAG) is a prevalent ap-
proach for domain-specific LLMs, yet it is often plagued by
”Retrieval Hallucinations”—a phenomenon where fine-tuned
models fail to recognize and act upon poor-quality retrieved
documents, thus undermining performance. To address this,
we propose the Adversarial Collaboration RAG (AC-RAG)
framework. AC-RAG employs two heterogeneous agents:
a generalist Detector that identifies knowledge gaps, and a
domain-specialized Resolver that provides precise solutions.
Guided by a moderator, these agents engage in an adversar-
ial collaboration, where the Detector’s persistent questioning
challenges the Resolver’s expertise. This dynamic process
allows for iterative problem dissection and refined knowledge
retrieval. Extensive experiments show that AC-RAG signifi-
cantly improves retrieval accuracy and outperforms state-of-
the-art RAG methods across various vertical domains.1
Index Terms—Retrieval-Augmented Generation (RAG),
Adversarial Collaboration, Multi-Agent Systems
1. INTRODUCTION
Retrieval-Augmented Generation (RAG) seeks to enhance
Large Language Models (LLMs) by integrating external
knowledge bases, thereby mitigating factual inaccuracies
and hallucinations [1, 2]. This approach has garnered signifi-
cant interest for its potential in domain-specific applications.
However, the performance of RAG systems is heavily de-
pendent on the quality of retrieved information, and existing
methods often face critical challenges that degrade their ef-
fectiveness, as illustrated in Figure 1.
One primary issue is Semantic Discrepancy, where re-
trievers, optimized for semantic similarity, fail to capture the
nuanced ”relevance” required by complex queries. For in-
stance, a query about ”callus formation” might retrieve docu-
ments discussing ”rigid immobilization” due to superficial se-
mantic overlap, even though ”movement at the fracture site”
is the more relevant concept for answering the question. This
discrepancy between similarity and relevance often leads to
the retrieval of suboptimal or misleading information.
A more subtle and pernicious problem, which we identify
as Retrieval Hallucination, occurs particularly with LLMs
that have been fine-tuned on specialized domain data. These
1Code is available: https://anonymous.4open.science/r/AC-RAG/
Which of the following increases callus formation?
A. Rigid immobilization          B . Movement at fracture site
C. Compression plating          D. Intraosseous nailing
What is Rigid immobilization ?
Rigid immobilization  refers to …
KGImmobilization  techniques  
such  as…
Not sufficient for problem -solving.
What is  callus formation ?
Callus formation refers to …
KGCallus  formation  results  from …
Sufficient for problem -solving.
B. Movement at fracture site
Which of the following increases
callus formation ?
A. Rigid immobilization     
B. Movement at fracture site
C. Compression plating
D. Intraosseous nailing
KG… respiratory  distress  syndrome  
in adults  may  be produced  by any 
of a number  of causes  resulting  in 
similar  pathophysiologic  changes  
and having  identical  therapeutic  
implications . the most  important  
factors  in treatment  are early  
recognition  and early  institution  
of therapy  …
Although some concepts still require 
explanation, my existing knowledge  is 
already sufficient for problem -solving.
A. Rigid immobilization
①Semantic Discrepancy
②Retrieval HallucinationSelf-Reflect
✓ ✗Fig. 1: AC-RAG overcomes key failures of vanilla RAG: (1)Seman-
tic Discrepancy(retrieving irrelevant content) and (2)Retrieval Hal-
lucination(overconfidence in accepting poor results). Our frame-
work addresses both through adversarial collaboration.
expert models can exhibit overconfidence in their own para-
metric knowledge, causing them to either bypass the retrieval
process entirely or uncritically accept irrelevant search re-
sults during self-reflection phases. As shown in Figure 1,
even when retrieval provides a low-quality document about
”respiratory distress syndrome” for a question on ”callus for-
mation”, a fine-tuned model might erroneously conclude its
existing knowledge is sufficient and proceed to generate an
incorrect answer. This overconfidence prevents the model
from initiating further, more targeted searches, effectively
trapping it in a state of hallucination supported by poor re-
trieval. This challenge is particularly acute in systems that
utilize self-correction or reflection mechanisms but are hin-
dered by an overconfident generator [3, 4].
To address these interconnected challenges, we propose
the Adversarial Collaborative RAG (AC-RAG), a novel
framework designed to foster more rigorous and accurate
knowledge acquisition. AC-RAG orchestrates a dynamic in-
teraction between two heterogeneous agents: a generalist,
non-fine-tuned Detector and a domain-expert, fine-tuned Re-arXiv:2509.14750v1  [cs.AI]  18 Sep 2025

Part2: Challenge Dissection Part3: Retrieval & Integration
List the professional terms needed for further explanation.
Saltatory conduction
Explain the definition of the professional terms.
Walk me through the context below and write a summary.
KG
Saltatory conduction: Saltatory conduction is a process 
by which nerve impulses travel along myelinated axons, 
jumping from one node of Ranvier to the next.In neuroscience,  saltatory  conduction  (from  Latin  saltus  'leap,  jump')  is the 
propagation  of action  potentials  along  myelinated  axons  from  one node  of 
Ranvier  to the next  node,  increasing  the conduction  velocity  of action  potentials . 
The uninsulated  nodes  of Ranvier  are the only  places  along  the axon  where  ions 
are exchanged  across  the axon  membrane,  regenerating  the action  potential  
between  regions  of the axon  that are insulated  by myelin,  unlike  electrical  
conduction  in a simple  circuit .
: Neutral Moderator
: Resolver: Detector
Which of the following is 
not true for myelinated 
nerve fibers?
A. Impulse through 
myelinated fibers is slower 
than non -myelinated fibers
B. Membrane currents are 
generated at nodes of 
Ranvier
C. Saltatory conduction of 
impulses is seen
D. Local anesthesia can be 
effective only when the 
nerve is not covered by 
myelin sheath
Instruction Tuning on 
Domain -specific Data 
Part1: Pre-Check
Are there any medical terms in this 
question that you do not understand?
Analysis : Myelinated  nerve  fibers  as local  anesthesia  
can be less effective  on them  compared  to non-
myelinated  fibers,  …
Answer : A. Impulse  through  myelinated  fibers  is 
slower  than  non-myelinated  fibers .Answer the question directly.
 No
Yes
Myelinated Fibers: Myelinated fibers are nerve fibers 
covered with myelin sheath, enhancing the speed and 
efficiency of nerve impulse transmission.
Membrane currents: Membrane currents refer to the 
flow of ions across the cell membrane, driven by 
differences in ion concentration and electrical potential.Memory
Part4: Post-Check
Do you think the context is sufficient 
for answering?
Analysis : Myelinated  nerve  fibers  conduct  impulses  
faster  than  non-myelinated  fibers  due to the 
presence  of the myelin  sheath,  …
Answer : A. Impulse  through  myelinated  fibers  is 
slower  than  non-myelinated  fibers .Answer the question given the 
reference context.
YesNoTask :
Myelinated nerve fibers have a myelin sheath, 
enabling faster impulse conduction via saltatory 
conduction at nodes of Ranvier, unlike non -
myelinated fibers. Local anesthesia blocks nerve 
conduction.Fig. 2: Overview of our framework. We employ two different LLMs in our Adversarial Collaboration process (Detector + Retriever). A
Neutral Moderator supervises the entire operation. For each taskQ, our workflow is divided into four phases, involving between 0 andN
retrievals. The process concludes at an optimal point through Adaptive Retrieval. The memoryMfrom each interaction is used as a reference
for subsequent interactions and for generating the final response.
solver. The Detector’s lack of specialized knowledge makes it
less prone to overconfidence, positioning it as an ideal agent
for identifying potential knowledge gaps and dissecting com-
plex problems into simpler, more retrievable sub-questions.
In contrast, the expert Resolver is tasked with answering
these sub-questions, leveraging its deep domain knowledge
to generate precise queries and synthesize retrieved informa-
tion. This interaction follows a ”Dissect-Retrieve-Reflect”
workflow, where the Detector’s persistent questioning and the
Resolver’s expert synthesis create a productive adversarial
tension. This process, guided by a neutral moderator, sys-
tematically filters out noise, probes for deeper understanding,
and converges on a well-supported, collaborative solution, ef-
fectively mitigating both semantic discrepancies and retrieval
hallucinations.
2. METHODS
Our AC-RAG framework is built upon an adversarial collab-
oration between two heterogeneous agents: a general, non-
fine-tuned Detector (F D) and a domain-expert, fine-tuned
Resolver (F R). The Detector identifies knowledge gaps,
while the Resolver provides expert solutions. This process
is orchestrated by a neutral moderator through a multi-turn
“Dissect-Retrieve-Reflect” workflow, as depicted in Figure 2.
The entire workflow, including the pseudo-code (Algorithm
1), is detailed in the Appendix.
Pre-Check.The workflow begins with a Pre-Check, where
the Detector assesses if retrieval is necessary. This step pre-
vents the indiscriminate use of RAG, which can sometimes be
counterproductive.Challenge Dissection.If retrieval is needed, the process en-
ters an iterative loop. In each roundk+ 1, the Detector ana-
lyzes the original queryQand the accumulated memoryM k
to formulate a new sub-questiont k+1=FD(Q, M k). The
memoryM kstores pairs of terms and their summarized ex-
planations from previous rounds, i.e.,M k=Sk
i=1(ti, si),
withM 0=∅.
Retrieval & Integration.Next, we first generate a prelim-
inary explanatione k+1 =F R(tk+1)for the sub-question.
We then use this explanation as an enhanced query for the
retrieverR, a technique known to improve performance by
aligning the query more closely with the language of the
source documents [5]. This yields retrieved documents
rk+1 =R(e k+1). Subsequently, the Resolver summa-
rizes this content to produce a concise explanations k+1=
FR(rk+1), filtering out noise. The memory is then updated:
Mk+1=M kS(tk+1, sk+1).
Post-Check.The final stage of each loop is the Post-Check,
where the Detector determines if the current knowledge in
memoryM k+1is sufficient. If not, and if the maximum num-
ber of iterationsNhas not been reached, a new round begins.
Otherwise, the loop terminates, and the Resolver generates
the final answer based on the full context:A=F R(Q, M n).
Adaptive Retrieval.Decisions in the Pre-Check and Post-
Check phases are governed by the Detector’s confidence. We
measure this by calculating the log probability of it generating
an affirmative token (e.g., ”yes”). The confidence score is
defined as score k=1
LPL
ℓ=1logP(y ℓ∈S|y <ℓ, Q, M k, FD),
whereSis the set of affirmative tokens. Retrieval is triggered
if score 0> δ 1(Pre-Check) or if another iteration is requested
with score k> δ 4(Post-Check). For efficiency, we setL= 1.

3. EXPERIMENTS AND RESULTS
We evaluate AC-RAG on knowledge-intensive tasks, primar-
ily in the medical domain. All experimental setup details,
including model training, datasets, baselines, and generaliz-
ability tests on other domains, are provided in the Appendix.
3.1. Main Results on Medical Tasks
We tested our method on four challenging medical bench-
marks: MedQA [6], MedMCQA [7], PubMedQA [8], and
MMLU-Medical [9]. As shown in Table 1, our primary
model, AC-RAG-8B (which uses a fine-tuned Llama-3-8B as
the Resolver), consistently outperforms strong baselines.
When compared to baselines without retrieval, AC-RAG-
8B surpasses both the original Llama-3-8B [10] and its
fine-tuned version (Llama-3-8B-FT). When compared to
retrieval-based baselines, AC-RAG also demonstrates clear
advantages. Notably, standard RAG can even degrade the
performance of powerful models like Llama-3-8B on some
benchmarks. This performance drop is particularly stark on
MedQA and PubMedQA, as shown in Table 1, where the
model’s accuracy declines when presented with potentially
noisy or irrelevant retrieved documents. This phenomenon
highlights a key challenge that AC-RAG is designed to over-
come. In contrast, our approach, along with other advanced
RAG solutions like Self-RAG [3] and CRAG [11], mitigates
this issue, with AC-RAG achieving the best performance
among them. This suggests that the advantages of AC-RAG
stem not only from the training data but also from the frame-
work’s inherent design, which more effectively combats the
”Retrieval Hallucination” by actively dissecting problems
rather than passively filtering documents. Furthermore, our
larger model, AC-RAG-70B, achieves performance competi-
tive with GPT-4.
3.2. Ablation Studies
To dissect the effectiveness of AC-RAG’s components, we
conduct comprehensive ablation studies.
3.2.1. The Role of Heterogeneous Agents
To validate our core design choice of using heterogeneous
agents, we analyze the performance of different model com-
binations for the Detector and Resolver roles. As shown in
Table 2, our proposed setup—using a non-fine-tuned ”Basic”
model as the Detector and a fine-tuned ”FT” model as the Re-
solver—yields the best results.
Using a general model as the Detector leads to a higher
retrieval rate (RA Rate) and more interaction turns (Iters).
This suggests that specialized models can be overconfident,
skipping retrieval when it is needed—a ”retrieval hallucina-
tion” issue that a less confident general model helps mitigate.
Conversely, for the Resolver, a specialized model is moreTable 1: Overall results on four medical benchmarks.Boldnumbers
indicate the best performance for non-proprietary models of similar
scale.⋆denotes results from concurrent studies. – indicates data
not reported or applicable. We mark our reproductions with⋆⋆. All
results from our implementations are averaged over five runs.
Accuracy (↑)
ModelMMLU-Medical PubMedQA MedMCQA MedQA Avg
Baselines without retrieval
Mistral-7B 60.8 1.23 33.815.2 45.41.77 48.51.64 43.3
Llama-3-8B 67.2 2.37 64.810.4 51.83.10 59.23.61 60.8
Llama-3-8B-FT 68.2 1.65 65.87.45 57.02.31 60.01.89 62.8
MEDITRON-7B 42.3 2.37 69.315.1 36.31.38 37.43.27 42.8
PMC-Llama-7B 26.2 1.27 57.020.6 27.45.91 27.80.86 32.0
Mixtral-8x7B 74.7 0.87 69.87.34 56.20.21 58.10.95 64.7
Llama-3-70B 82.8 0.50 72.43.44 72.32.30 74.52.75 76.4
GPT-4 88.3 73.6 77.2 79.3 79.6
Baselines with retrieval
Mistral-7B 61.1 3.19 41.613.9 46.12.14 50.82.01 49.9
Llama-3-8B 65.1 3.48 43.612.6 52.53.63 55.33.92 54.1
Llama-3-8B-FT 66.7 2.94 63.88.12 55.23.08 60.32.26 61.5
SAIL⋆- 69.2 - - -
CRAG⋆-75.6- - -
Self-RAG⋆- 72.4 - - -
Self-CRAG⋆- 74.8 - - -
CRAG⋆⋆68.73.25 69.613.2 57.92.67 60.22.28 64.1
Self-RAG⋆⋆69.11.08 67.510.6 59.82.31 61.81.01 64.6
Mixtral-8x7B 74.5 1.71 67.29.07 56.71.10 58.81.36 64.3
Llama-3-70B 83.1 0.83 72.16.01 71.82.95 74.42.96 75.6
Ours
AC-RAG-8B70.2 0.43 73.29.46 59.61.75 63.21.43 66.5
AC-RAG-70B84.2 0.43 73.93.92 75.80.99 75.91.69 77.5
efficient and accurate, leading to fewer interaction rounds and
improved performance. Its domain expertise is beneficial for
tasks like summarizing retrieved results and providing the
final answer. This ablation strongly validates our hypothe-
sis that an adversarial collaboration between heterogeneous
agents is key to AC-RAG’s success.
3.2.2. Impact of Adaptive Retrieval Thresholds
We further analyze the impact of the adaptive retrieval thresh-
olds,δ 1for Pre-Check andδ 4for Post-Check, on two repre-
sentative benchmarks, MedMCQA and PubMedQA.
Pre-Check Analysis.As shown in Figure 3(a), the choice
ofδ 1directly influences the ”Direct Answer Rate”—the per-
centage of questions answered without retrieval. A larger
δ1makes the model more inclined to retrieve. The optimal
threshold varies across benchmarks, but an overly aggressive
retrieval strategy (e.g., a very lowδ 1) can negatively impact
performance. This demonstrates that indiscriminately apply-
ing RAG can be harmful, and the Pre-Check stage is crucial
for balancing retrieval benefits with the risk of introducing
noise.
Post-Check Analysis.The Post-Check thresholdδ 4controls
the number of iterative retrieval rounds. Figure 3(b) shows
that performing only a single round of retrieval (δ 4=−∞)

-
 -3 -2 -1 0
Logprob Threshold 1
57585960Accuracy (%)
MedMCQA
-
 -3 -2 -1 0
Logprob Threshold 1
70.071.072.073.0Accuracy (%)
PubMedQA
0204060
Direct Answer Rate (%)
020406080
Direct Answer Rate (%)
(a) Pre-Check
-
 -3 -2 -1 0
Logprob Threshold 4
51.052.053.054.055.056.057.0Accuracy (%)
MedMCQA
-
 -3 -2 -1 0
Logprob Threshold 4
65.066.067.068.069.070.0Accuracy (%)
PubMedQA
1.01.52.02.5
RA Iteration Turns
0.00.51.01.52.02.53.0
RA Iteration Turns
(b) Post-Check
Fig. 3: Analysis on adaptive retrieval.δ 1andδ 4serve as thresholds
for the Pre-Check and Post-Check stages, respectively. The Direct
Answer Rate measures the percentage of samples that achieve direct
answers during the Pre-Check stage without any retrieval enhance-
ment. The Post-Check experiment focuses only on samples that have
successfully passed the Pre-Check stage and subsequently proceed
with retrieval. For models utilizing retrieval augmentation, we cal-
culate both the accuracy and the average number of retrieval turns
per sample.
is suboptimal. Asδ 4increases, the average number of in-
teraction turns (”RA Iteration Turns”) also increases, lead-
ing to performance gains up to a certain point. This confirms
that multi-turn refinement is effective for addressing complex
questions where initial retrievals may introduce new knowl-
edge gaps. However, excessively high values do not yield
further benefits, indicating a point of diminishing returns.
4. RELATED WORK
Retrieval-Augmented Generation.Retrieval-Augmented
Generation (RAG) enhances LLMs by incorporating external
knowledge [2]. Recent advancements have focused on im-
proving the RAG process. Some methods introduce adaptive
retrieval, determining when retrieval is necessary to avoid
noise, such as Self-RAG [3]. Others focus on enhancing
robustness by filtering irrelevant context [12] or improving
instruction quality by integrating retrieved documents during
fine-tuning [13]. While our work builds on the principle of
adaptive retrieval, it introduces a novel mechanism for this
decision-making process: an adversarial collaboration be-
tween two heterogeneous agents. This distinguishes AC-RAG
from methods that rely on a single model’s self-reflection or
a separate filtering module.
Chain of Thought for Retrieval.The Chain of Thought
(CoT) paradigm has proven effective for complex reasoning
[14]. This concept has been extended to RAG, where rea-Table 2: Ablation studies on AC-RAG’s adversarial agents selection.
”Base” and ”FT” refer to the model before and after fine-tuning, re-
spectively. The RA Rate denotes the percentage of test samples that
undergo at least one retrieval. ”Iters” indicates the average number
of retrievals per sample for those that have undergone at least one
retrieval cycle.
Detector Resolver MedMCQA PubMedQA
Basic FT Basic FT Acc↑RA Rate Iters Acc↑RA Rate Iters
[1]✓ ✓52.3 62.4 1.74 50.6 61.0 2.21
[2]✓ ✓56.5 62.4 1.22 69.1 61.0 1.89
[3]✓ ✓53.9 74.1 2.41 60.7 68.7 2.62
[4]✓ ✓59.674.1 1.8073.268.7 2.35
soning steps guide the retrieval process. For instance, IR-
COT interleaves CoT reasoning with retrieval to refine search
queries [15], and other works use CoT to decompose complex
questions into simpler, retrievable sub-problems [16]. These
methods typically rely on a single LLM to generate and refine
its own thought process. Our AC-RAG framework draws in-
spiration from this decompositional approach but operational-
izes it through a multi-agent dialogue. The ”chain of thought”
is not generated by a single mind but is instead the emergent
result of the structured interaction between the questioning
Detector and the answering Resolver.
Interacting Agents.Multi-agent systems have a long his-
tory, with communication enabling complex task-solving
[17]. These systems often operate in cooperative scenarios,
where agents work together towards a common goal [18].
Our approach, however, is founded onadversarial collabora-
tion, a concept more akin to the competitive dynamics seen
in methods like Actor-Critic in reinforcement learning [19].
In AC-RAG, the agents have distinct, sometimes conflicting,
roles: the Detector seeks to find flaws and knowledge gaps,
while the Resolver aims to provide complete answers. This
structured tension, rather than pure cooperation, is the mecha-
nism that drives deeper inquiry and improves the final output,
allowing the system to capitalize on the unique strengths of
each agent.
5. CONCLUSION
This paper addresses ”Retrieval Hallucination”, a key chal-
lenge where fine-tuned RAG models become overconfident
in specialized domains. We introduced AC-RAG, a frame-
work that mitigates this via an adversarial collaboration be-
tween a generalist Detector and a specialist Resolver. By
iteratively dissecting problems and refining knowledge, AC-
RAG robustly enhances generation quality. Extensive exper-
iments show significant performance gains over strong base-
lines, highlighting the potential of this structured multi-agent
approach for more reliable RAG systems. Future work could
extend this adversarial framework to other tasks and explore
dynamic agent roles, offering a promising path toward more
robust and trustworthy AI.

References
[1] Kelvin Guu, Kenton Lee, Zora Tung, Panupong Pasu-
pat, et al., “Retrieval augmented language model pre-
training,” inInternational conference on machine learn-
ing. PMLR, 2020, pp. 3929–3938.
[2] Patrick Lewis, Ethan Perez, Aleksandra Piktus,
et al., “Retrieval-augmented generation for knowledge-
intensive nlp tasks,”Advances in Neural Information
Processing Systems, vol. 33, pp. 9459–9474, 2020.
[3] Akari Asai et al., “Self-rag: Learning to retrieve, gener-
ate, and critique through self-reflection,”arXiv preprint
arXiv:2310.11511, 2023.
[4] Jason Weston and Sainbayar Sukhbaatar, “System 2
attention (is something you might need too),”arXiv
preprint arXiv:2311.11829, 2023.
[5] Liang Wang, Nan Yang, and Furu Wei, “Query2doc:
Query expansion with large language models,” inPro-
ceedings of the 2023 Conference on Empirical Methods
in Natural Language Processing, 2023, pp. 9414–9423.
[6] Di Jin, Eileen Pan, Nassim Oufattole, et al., “What dis-
ease does this patient have? a large-scale open domain
question answering dataset from medical exams,”arXiv
preprint arXiv:2009.13081, 2020.
[7] Ankit Pal et al., “Medmcqa: A large-scale multi-subject
multi-choice dataset for medical domain question an-
swering,” inConference on health, inference, and learn-
ing. PMLR, 2022, pp. 248–260.
[8] Qiao Jin, Bhuwan Dhingra, et al., “Pubmedqa: A
dataset for biomedical research question answering,” in
EMNLP-IJCNLP, 2019, pp. 2567–2577.
[9] Dan Hendrycks et al., “Measuring massive multitask
language understanding,” inInternational Conference
on Learning Representations. 2021, OpenReview.net.
[10] AI@Meta, “Llama 3 model card,” 2024.
[11] Shi-Qi Yan et al., “Corrective retrieval augmented gen-
eration,”arXiv preprint arXiv:2401.15884, 2024.
[12] Ori Yoran, Tomer Wolfson, et al., “Making retrieval-
augmented language models robust to irrelevant con-
text,”arXiv preprint arXiv:2310.01558, 2023.
[13] Hongyin Luo, Yung-Sung Chuang, Yuan Gong, et al.,
“SAIL: search-augmented instruction learning,”arXiv
preprint arXiv:2305.15225, 2023.
[14] Jason Wei, Xuezhi Wang, Dale Schuurmans, et al.,
“Chain-of-thought prompting elicits reasoning in large
language models,”Advances in neural information pro-
cessing systems, vol. 35, pp. 24824–24837, 2022.[15] Harsh Trivedi, Niranjan Balasubramanian, Tushar Khot,
et al., “Interleaving retrieval with chain-of-thought rea-
soning for knowledge-intensive multi-step questions,”
inProceedings of the 61st Annual Meeting of the Asso-
ciation for Computational Linguistics (Volume 1: Long
Papers), 2023, pp. 10014–10037.
[16] Yucheng Zhou, Xiubo Geng, Tao Shen, Chongyang Tao,
et al., “Thread of thought unraveling chaotic contexts,”
arXiv preprint arXiv:2311.08734, 2023.
[17] Jacob Andreas, “Language models as agent models,” in
Findings of the Association for Computational Linguis-
tics: EMNLP 2022, 2022, pp. 5769–5779.
[18] Allan Dafoe, Edward Hughes, Yoram Bachrach, et al.,
“Open problems in cooperative ai,”arXiv preprint
arXiv:2012.08630, 2020.
[19] V olodymyr Mnih, Adri `a Puigdom `enech Badia, et al.,
“Asynchronous methods for deep reinforcement learn-
ing,” inInternational Conference on Machine Learn-
ing. 2016, vol. 48 ofJMLR Workshop and Conference
Proceedings, pp. 1928–1937, JMLR.org.
[20] Chaoyi Wu, Weixiong Lin, et al., “Pmc-llama: toward
building open-source language models for medicine,”
Journal of the American Medical Informatics Associa-
tion, vol. 31, no. 9, pp. 1833–1843, 04 2024.
[21] Kyle Lo, Lucy Lu Wang, Mark Neumann, Rodney Kin-
ney, et al., “S2orc: The semantic scholar open research
corpus,”arXiv preprint arXiv:1911.02782, 2019.
[22] Guangzhi Xiong, Qiao Jin, Zhiyong Lu, et al., “Bench-
marking retrieval-augmented generation for medicine,”
arXiv preprint arXiv:2402.13178, 2024.
[23] Qiao Jin, Won Kim, et al., “Medcpt: Contrastive
pre-trained transformers with large-scale pubmed search
logs for zero-shot biomedical information retrieval,”
Bioinformatics, vol. 39, no. 11, pp. btad651, 2023.
[24] Karan Singhal, Shekoofeh Azizi, Tao Tu, et al., “Large
language models encode clinical knowledge,”Nature,
vol. 620, no. 7972, pp. 172–180, 2023.
[25] Zeming Chen, Alejandro Hern ´andez Cano, Angelika
Romanou, Antoine Bonnet, et al., “MEDITRON-70B:
scaling medical pretraining for large language models,”
arXiv preprint arXiv:2311.16079, 2023.
[26] Pierre Colombo, Telmo Pessoa Pires, Malik Boudiaf,
Dominic Culver, Rui Melo, et al., “Saullm-7b: A pi-
oneering large language model for law,”arXiv preprint
arXiv:2403.03883, 2024.

A. EXPERIMENTAL DETAILS
This appendix provides a comprehensive overview of our ex-
perimental setup, including datasets, baselines, implementa-
tion details, and additional results referenced in the main pa-
per.
A.1. Experimental Setup
A.1.1. Training Details
Our training process for the fine-tuned Resolver model fol-
lows the methodology outlined by [20] and is divided into
two phases.
Pre-training Phase:We used data from S2ORC [21], specifi-
cally selecting abstracts from papers with PubMed IDs to con-
centrate on medical content. The pre-training dataset contains
approximately 6B tokens. We performed knowledge injection
on this data for 1 epoch using 8 A100 GPUs over 4 days.
Instruction-Tuning Phase:We conducted medical-specific
instruction tuning on a mixed dataset that includes sources
from ChatDoctor, MedQA, MedMCQA, PubMedQA, LiveQA,
MedicationQA, and UMLS [20]. This tuning was also carried
out for 1 epoch with 4 A100 GPUs within 14 hours.
Our base model for this training is Meta-Llama-3-8B [10].
During both phases, the batch size was 8 per GPU with a gra-
dient accumulation step of 4. We used the AdamW optimizer
(β1= 0.9, β 2= 0.999) with a cosine learning rate decay. The
learning rate for pre-training was9×10−5with 1000 warmup
steps, and for fine-tuning was2×10−5with 20 warmup steps.
1% of the training set in both phases was reserved for valida-
tion.
A.1.2. Inference and Implementation Details
Hyperparameters:In our default setup, we assignδ 1=
−2.0for the Pre-Check threshold andδ 4=−3.0for the
Post-Check threshold, as defined in Section 2. We cap the
maximum number of interaction rounds atN= 3.
Efficiency Settings:For simplicity and speed, we set the out-
put length for confidence scoring toL= 1. During retrieval,
we only consider the top-1 document.
Output Parsing:We designed prompts to expect outputs
in the format of ”### Answer: A/B/C/D” or ”###
Answer: yes/no/maybe” and used regular expres-
sions to parse the final answer.
A.2. Knowledge Base Construction
Our knowledge base for the medical domain is constructed
from four primary sources: PubMed for biomedical abstracts,
StatPearls for clinical decision support, medical textbooks for
domain-specific knowledge, and Wikipedia for general in-
formation [22]. Each article is split into disjoint 512-token
chunks, creating a total of approximately 230M documents.We use MedCPT [23], an encoder pretrained on PubMed, to
generate embeddings for each document. These embeddings
are then indexed and stored in a Chroma vector database for
efficient retrieval.
A.3. Benchmark Datasets
A.3.1. Medical Domain
MedQA[6]: The MedQA dataset consists of questions styled
after the US Medical License Exam (USMLE), requiring con-
textual medical knowledge like patient profiles, symptoms,
and dosages. Its test set comprises 1273 questions, including
both four-choice and five-choice multiple-choice questions.
MedMCQA[7]: The MedMCQA dataset includes over 194k
multiple-choice questions from Indian medical entrance ex-
ams, covering 2.4k healthcare topics and 21 medical subjects.
Evaluations are reported on the validation set due to the lack
of public answer keys for the test set. It contains a total of
4183 samples.
PubMedQA[8]: PubMedQA features 200k multiple-choice
QA samples, plus 1k expert-labeled samples. Given a
PubMed abstract, models are required to predict ”yes”, ”no”,
or ”maybe” answers. Evaluations use 500 expert-labeled test
samples, following the setting used by [24].
MMLU[9]: The MMLU dataset includes exam questions
from 57 subjects. In accordance with the setting in [25], we
select the nine most relevant subsets to form the medical sub-
set (MMLU-Medical). These subsets are high school biology,
college biology, college medicine, professional medicine,
medical genetics, virology, clinical knowledge, nutrition, and
anatomy. In total, these test sets comprise 2174 samples.
A.3.2. Other Domains for Generalizability
LegalBench[26]: We evaluate on five task categories: issue-
spotting, rule-recall, rule-conclusion, interpretation, and
rhetorical-understanding. For interpretation tasks, we use
the MAUD, CUAD, and Contract NLI subsets. The model is
trained using datasets from [26].
Huawei DevOps: This is an internal benchmark for resolving
Pull Request issues. It includes two sub-tasks: BuildCheck
(compilation errors) and CodeCheck (coding standard viola-
tions). We use a Baichuan2-13B model fine-tuned on 50K
internal records. The evaluation is based on manual assess-
ment of the generated solutions.
A.4. Baseline Models
We compare AC-RAG against two categories of baselines.
All models are evaluated in few-shot scenarios (3-shot for
most, 1-shot for PubMedQA) to ensure fair comparison.
Baselines without retrieval.We evaluate on several widely
used LLMs such as Meta-Llama-3, Mistral, and some medi-
cally fine-tuned LLMs like MPT-7B, Falcon-7B, MEDITRON,

Issue Rule Conclusion MAUD CAUD Contract NLI Rhetorical020406080ScoresGPT-4
LLaMA-3-8B
LLaMA-3-8BRAG
LLaMA-3-8BACRAG
Fig. 4: Evaluation of LegalBench. The scores provided represent
the average outcomes from tests conducted across all datasets in this
collection.
Table 3: Evaluation of Huawei’s DevOps business benchmark. The
accuracy data in this table are based on subjective human judgment.
Method BuildCheck CodeCheck
Baichuan2-13B 61.5 64.0
Baichuan2-13B RAG 83.0 74.5
Baichuan2-13B AC−RAG 85.0 79.5
PMC-Llama. For fairness, we grouped models with 13B pa-
rameters or fewer from those with more than 13B parameters.
To ensure the responses from LLMs meet formatting require-
ments, all models are evaluated in few-shot scenarios: 1-shot
for PubMedQA due to its lengthy questions, and 3-shot for
other benchmarks.
Baselines with retrieval.RAG baselines include standard
RAG [2], where an LM generates is involved to produces
outputs based on queries prefixed with top retrieved docu-
ments, choosing the same retriever as our system. Addition-
ally, we consider SAIL [13] which fine-tunes an LM on Al-
paca instruction-tuning data by inserting top retrieved docu-
ments before the instructions. We also included Self-RAG
[3], which tuned LlaMA-2 on instruction-tuning data using
multiple sets of reflection tokens generated by GPT-4, as well
as CRAG [11] features a lightweight retrieval evaluator de-
signed to gauge the overall quality of retrieved documents.
A.5. Additional Results and Analysis
This section provides detailed results and analysis for the gen-
eralizability studies mentioned in the main paper.
A.5.1. Evaluation on Other Domain Tasks
LegalBench.As shown in Figure 4, we compared AC-RAG
and standard RAG on the LegalBench benchmark, using
Llama-3-8B as the base model. Our observations indicated
that standard RAG might occasionally limit the effectivenessof the original model, which aligns with the issue of noisy
retrieval documents negatively impacting performance. In
contrast, after implementing AC-RAG, we observed consis-
tent improvements across various tasks, with performance
sometimes nearing that of GPT-4.
Huawei DevOps.The experimental results are presented in
Table 3. In our specific business contexts, the types of is-
sues are relatively consistent, and our knowledge base con-
tains similar past queries with manually annotated answers.
This setup significantly enhances performance through RAG.
Implementing the AC-RAG framework elevates the quality of
the generated responses to the next level.
B. PROMPTS
The following section presents the detailed prompt templates
used in our main experiments. These prompts were carefully
designed to support various stages of our AC-RAG work-
flow, including pre-checking the necessity of retrieval, dis-
secting medical challenges, performing retrieval and integra-
tion, and conducting post-checks. The templates are tailored
for multiple-choice and true/false question formats, ensuring
consistent query patterns for evaluation.
To facilitate readability, we define the following variables
used across the templates:
Table 4: Variable definitions used in prompt templates
Variable Definition
{question}Question part of a multiple-choice question or the
statement part of a true/false question.
{options}The four options of a multiple-choice question.
{context}Context from the true/false question itself.
{rag context}Retrieved content.
{summary context}Retrieved content after summarization.
{memory}Dictionary of previously retrieved content.
Pre-Check:Evaluate whether RAG is necessary (tailored
specifically for multiple-choice and true/false questions).
Pre-Check - multiple-choice questions
{question}
{options}
Question: Are there any medical terms in this question
that you do not understand?
Options:
yes
no
Answer:
Pre-Check - true/false questions
{question}
{context}

Question: Are there any medical terms in this question
that you do not understand?
Options:
yes
no
Answer:
Challenge Dissection:Extract professional terminology and
break down tasks (tailored specifically for multiple-choice
and true/false questions).
Challenge Dissection - multiple-choice questions
{question}
{options}
{memory}
Question: List the medical terms needed for further
explanation.
Answer: Medical terms which are hard to understand
is:
Challenge Dissection - true/false questions
{question}
{context}
{memory}
Question: List the medical terms needed for further
explanation.
Answer: Medical terms which are hard to understand
is:
Retrieval & Integration:Summary. Just the same as in Fig-
ure 2.
Retrieval & Integration:
Walk me through the context below and write a sum-
mary.
Context:{rag context}
Post-Check:Reflection.
Post-Check
You are an expert doctor in clinical science and med-
ical knowledge. Read the following context and the
question.
Context:{summary context}
Question:{question}
Do you think the context is sufficient for answering the
this question?
Options:
yes
noAnswer:
QA w/ RAG:Answer with RAG (tailored specifically for
multiple-choice and true/false questions).
QA w/ RAG - multiple-choice questions
You’re a doctor, kindly address the medical queries ac-
cording to the patient’s account.
Analyze the question given its context. Answer with
the best option directly.
### Question:
{question}
### Options:
{options}
### Context2:
{memory}
### Answer:
QA w/ RAG - true/false questions
You’re a doctor, kindly address the medical queries ac-
cording to the patient’s account.
Analyze the question given its context. Give
yes/no/maybe decision directly.
### Question:
{question}
### Context1:
{context}
### Context2:
{memory}
### Answer:
QA w/o RAG:Answer without RAG (tailored specifically for
multiple-choice and true/false questions).
QA w/o RAG - multiple-choice questions
You’re a doctor, kindly address the medical queries ac-
cording to the patient’s account.
Analyze the question given its context. Answer with
the best option directly.
### Question:
{question}
### Options:
{options}
### Answer:
QA w/o RAG - true/false questions
You’re a doctor, kindly address the medical queries ac-
cording to the patient’s account.
Analyze the question given its context. Give
yes/no/maybe decision directly.

### Question:
{question}
### Context:
{context}
### Answer:
System Prompt:The beginning of this prompt, ”You’re a
doctor, kindly address the medical queries according to the
patient’s account.” is implemented in practice by cycling
through ten sentences, aligning with the prompts in the train-
ing dataset.
System Prompt
1. You’re a doctor, kindly address the medical queries
according to the patient’s account.
2. Being a doctor, your task is to answer the medical
questions based on the patient’s description.
3. Your role as a doctor requires you to answer the
medical questions taking into account the patient’s de-
scription.
4. As a medical professional, your responsibility is to
address the medical questions using the patient’s de-
scription.
5. Given your profession as a doctor, please provide
responses to the medical questions using the patient’s
description.
6. Considering your role as a medical practitioner,
please use the patient’s description to answer the med-
ical questions.
7. In your capacity as a doctor, it is expected that you
answer the medical questions relying on the patient’s
description.
8. Your identity is a doctor, kindly provide answers
to the medical questions with consideration of the pa-
tient’s description.
9. Given your background as a doctor, please provide
your insight in addressing the medical questions based
on the patient’s account.
10. As a healthcare professional, please evaluate the
patient’s description and offer your expertise in an-
swering the medical questions.
C. ALGORITHM PSEUDOCODE
We present our detailed pseudocode as follows, continu-
ing the notation from Section 2. The algorithm has four
stages: Pre-Check, Challenge Dissection, Retrieval & Inte-
gration, and Post-Check. If Pre-Check indicates that retrieval
is needed, an interaction betweenFDandFRtakes place
to dissect a sub-task that requires retrieval and provide an
initial answer. Based on this initial response, retrieval and
integration are then conducted, followed by an assessment ofwhether the retrieved content is sufficient to address the en-
tire problem. This process repeats up toNtimes, either until
enough information is retrieved or the loop limit is reached.
Algorithm 1Adversarial Collaboration RAG Workflow
Require:DetectorF D, ResolverF R, Moderator, Knowledge
BaseK, queryQ, Maximum iteration numberN, Mem-
oryM, RetrieverR.
Ensure:Retrieval-Augmentated AnswerA
1:Receive queryQ, InitialMis∅. Moderator plans for
detector to decide whether to retrieve.
2:ifretrievethen
3:forn= 1,2, . . . , Ndo
4:Moderator employs the DetectorF Dand Resolver
FRto perform Challenge Dissection.
5:DetectorFDanalyzesQalong with previous inter-
actionsMto generate new meta-questions accord-
ing to Eq. (1).
6:ResolverF Raddresses meta-questions according to
Eq. (1).
7:Moderator employs the retriever and ResolverFRto
perform Retrieval & Integration.
8:Retriever search related documents from a vector
database according to Eq. (2).
9:ResolverF Rsummaries the retrieved results accord-
ing to Eq. (2).
10:Moderator employs DetectorFDand ResolverFR
to perform post-check.
11:DetectorF Ddetermines whether to continue the
RAG based on the Eq. (5).
12:ifretrievethen
13:History interactions are merged into MemoryM
according to Eq. (3).
14:Return to line 4.
15:else
16:ResolverF Rgenerates final answerAaccording
to Eq. (4).
17:end if
18:end for
19:else
20:Return final answerA=F R(Q,∅).
21:end if