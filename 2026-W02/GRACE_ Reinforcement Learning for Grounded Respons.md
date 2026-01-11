# GRACE: Reinforcement Learning for Grounded Response and Abstention under Contextual Evidence

**Authors**: Yibo Zhao, Jiapeng Zhu, Zichen Ding, Xiang Li

**Published**: 2026-01-08 02:47:33

**PDF URL**: [https://arxiv.org/pdf/2601.04525v1](https://arxiv.org/pdf/2601.04525v1)

## Abstract
Retrieval-Augmented Generation (RAG) integrates external knowledge to enhance Large Language Models (LLMs), yet systems remain susceptible to two critical flaws: providing correct answers without explicit grounded evidence and producing fabricated responses when the retrieved context is insufficient. While prior research has addressed these issues independently, a unified framework that integrates evidence-based grounding and reliable abstention is currently lacking. In this paper, we propose GRACE, a reinforcement-learning framework that simultaneously mitigates both types of flaws. GRACE employs a data construction method that utilizes heterogeneous retrievers to generate diverse training samples without manual annotation. A multi-stage gated reward function is then employed to train the model to assess evidence sufficiency, extract key supporting evidence, and provide answers or explicitly abstain. Experimental results on two benchmarks demonstrate that GRACE achieves state-of-the-art overall accuracy and strikes a favorable balance between accurate response and rejection, while requiring only 10% of the annotation costs of prior methods. Our code is available at https://github.com/YiboZhao624/Grace..

## Full Text


<!-- PDF content starts -->

GRACE: Reinforcement Learning for
Grounded R esponse and A bstention under C ontextual E vidence
Yibo Zhao1Jiapeng Zhu1Zichen Ding2Xiang Li1*
1School of Data Science and Engineering, East China Normal University
2Shanghai AI Laboratory
Abstract
Retrieval-Augmented Generation (RAG) inte-
grates external knowledge to enhance Large
Language Models (LLMs), yet systems remain
susceptible to two critical flaws: providing
correct answers without explicit grounded evi-
dence and producing fabricated responses when
the retrieved context is insufficient. While
prior research has addressed these issues in-
dependently, a unified framework that inte-
grates evidence-based grounding and reliable
abstention is currently lacking. In this pa-
per, we propose GRACE, a reinforcement-
learning framework that simultaneously mit-
igates both types of flaws. GRACE employs a
data construction method that utilizes hetero-
geneous retrievers to generate diverse training
samples without manual annotation. A multi-
stage gated reward function is then employed to
train the model to assess evidence sufficiency,
extract key supporting evidence, and provide
answers or explicitly abstain. Experimental
results on two benchmarks demonstrate that
GRACE achieves state-of-the-art overall accu-
racy and strikes a favorable balance between
accurate response and rejection, while requir-
ing only 10% of the annotation costs of prior
methods. Our code is available at https:
//github.com/YiboZhao624/Grace.
1 Introduction
With the rapid advancement, large language models
(LLMs) have been widely applied in various fields,
including machine translation (Xu et al., 2024a;
Qian et al., 2024), knowledge-based question an-
swering (Xu et al., 2024b; Lyu et al., 2024; Chen
et al., 2024b), and mathematical reasoning (Shao
et al., 2024; Xin et al., 2024). However, the knowl-
edge contained in LLMs is limited, which is of-
ten insufficient to provide reliable responses for
knowledge-intensive tasks (Li et al., 2025; Deng
*Corresponding Author: xiangli@dase.ecnu.edu.cn
User Query
Are Yingkou and Fuding the same level of city?
Successful Retrieval
Yingkou is a prefecture-level city 
of Liaoning province, People's 
Republic of China. It’s a ...
Fuding is a county-level city in 
northeastern Ningde ...Failed Retrieval
Yingkou is a prefecture-level city of 
Liaoning province, People's Republic 
of China. It’s a ...
Tongcheng is a subdistrict of the city 
of Fuding...
No, they aren’t.
Vanilla LLM
Yes, they are.
Provides correct answers but 
without supporting evidence.Fails to decline unanswerable 
queries and tends to halluciate.
<evidence>Yingkou is a prefecture-
level city of Liaoning province, 
People's Republic of China. Fuding 
is a county-level city in northeastern 
Ningde...</evidence>
<answer>No</answer><llm>Yingkou is a prefecture-level 
city of Liaoning province, People's 
Republic of China. But the city level 
of Fuding is not provided, therefore 
I cannot answer the question.</llm>
<answer>I don’t know</answer>
GRACE
User Query
Are Yingkou and Fuding the same level of city?
Sufficient
Evidence
Yingkou is a prefecture-level 
city of Liaoning province...
Fuding is a county-level city 
in ...
Insufficient
Evidence
Yingkou is a prefecture-level 
city of Liaoning province...
Tongcheng is a subdistrict of 
the city of Fuding...
<evidence>Yingkou is a 
prefecture-level city of 
Liaoning province. Fuding is a 
county-level city in 
Ningde...</evidence>
<answer>No</answer>
<llm>Yingkou is prefecture-level. 
But the city level of Fuding is 
not provided, therefore, I 
cannot answer it.</llm>
<answer>I don’t know
</answer>
GRACEVanilla LLM
No, they aren’t.
Provides correct answers but 
without supporting evidence.Yes, they are.
Fails to decline and tends to 
halluciate.
Figure 1: An example of responses generated by vanilla
LLM and GRACE under different retrieval results.
et al., 2025). To address this limitation, retrieval-
augmented generation (RAG) integrates relevant
context from external knowledge bases into the
generative process of LLMs (Lewis et al., 2020; Yu
et al., 2024; Asai et al., 2024).
Nevertheless, hallucination remains a persistent
issue in RAG systems (Chen et al., 2024c; Hu et al.,
2025). As illustrated in Figure 1, LLMs may either
provide correct answers without grounding the key
evidence (left), or produce fabricated responses de-
spite lacking sufficient supporting evidence (right),
thereby compromising the reliability of the gener-
ated output. (Zhou et al., 2025). Therefore, build-
ing a trustworthy RAG system requires strength-
ening two key aspects: (1) improving evidential
grounding, whereby the model provides correct an-
swers while explicitly citing supporting evidence
for transparency; and (2) fostering model honesty,
enabling it to abstain from responding when the
1arXiv:2601.04525v1  [cs.CL]  8 Jan 2026

provided evidence is insufficient.
However, prior studies tend to strengthen these
two aspects in isolation. One line of research
leverages prompting (Fang et al., 2024a), super-
vised fine-tuning (SFT) (Xia et al., 2025), or rein-
forcement learning (RL) (Zhao et al., 2024; Dai
et al., 2025) to enhance grounding performance.
Conversely, another branch of work focuses on
calibrating refusal mechanisms through prompt-
ing (Feng et al., 2024a), constraint decoding (Kim
et al., 2025), or preference alignment (Song et al.,
2025b). The former often inadvertly encourges the
model to speculate by prioritizing evidence local-
ization over factual honesty, even in the absence
of necessary information. In contrast, the latter
tends to overlook the model’s proficiency in gener-
ating evidence-based answers. Consequently, these
fragmented approaches fail to reconcile both capa-
bilities simultaneously, compromising the overall
trustworthiness of the RAG system. This naturally
raises a research question:how can we simulta-
neously mitigate both types of hallucination while
minimizing human annotation costs?
To answer this question, we claim that an ideal
RAG backbone should make the decision bound-
ary explicit. The model should either produce an
evidence-grounded response with explicit citations
or, when the retrieved context is inadequate, pro-
vide a tentative answer accompanied by a clear
disclaimer regarding the lack of supporting evi-
dence. Thus, we propose a sample-efficient frame-
work GRACE for training RAG backbones to rec-
oncile evidence-integrated answering with reliable
refusal. Hence, our framework comprises three
key components: (1) a pipeline for constructing
answerable and unanswerable training data from
heterogeneous retrievers; (2) a multi-stage gated
reward that provides multi-aspect outcome feed-
back; and (3) a modified Dynamic sAmpling Pol-
icy Optimization (DAPO) algorithm for effective
fine-tuning. To enhance generalization, we first
curate a balanced training corpus leveraging het-
erogeneous retrievers to collect diverse candidate
chunks. We then strategically partition queries into
answerable and unanswerable categories by selec-
tively withholding supporting evidence, thereby
creating a benchmark for the model to learn ev-
idence sufficiency. During training, we utilize a
multi-stage gated reward to jointly optimize the
evidence grounding, answer accuracy, and absten-
tion: a format reward ensures structural consis-
tency, a path selection reward guides the modelto judge evidence sufficiency (answering vs. ab-
staining), fostering model honesty, and a content
accuracy reward encourages precise evidence ref-
erencing and correct final answers, improving ev-
idential grounding. Experimental results demon-
strate that GRACE achieves state-of-the-art perfor-
mance, outperforming significantly larger models
and data-intensive baselines. Notably, with only
2,000 annotated samples, our 4B-parameter model
not only surpasses existing benchmarks in over-
all accuracy but also achieves a superior synergy
between answer correctness and refusal reliability,
underscoring the exceptional sample efficiency of
GRACE. Our contributions are threefold:
•A retriever-based data construction pipeline
that curates diverse training samples, includ-
ing both evidence-sufficient and evidence-
insufficient scenarios. This enables the model
to jointly optimize for accurate cited response
generation and the robust capability to abstain.
•A joint optimization schemethat simultane-
ously trains the model to assess retrieval suffi-
ciency and to effectively utilize evidence, leading
to state-of-the-art results on two datasets.
•New insights for training agentic RAG mod-
els, revealing that sparse outcome-based rewards
can lead to unintended behaviors in intermediate
steps, especially in judging retrieval sufficiency.
2 Related Work
Since the introduction of retrieval-augmented gen-
eration by (Lewis et al., 2020), a number of subse-
quent studies have sought to improve RAG systems
along various dimensions. In this work, we focus
on three core aspects: answer accuracy, grounding
fidelity, and abstention behavior.
To improve answer accuracy, a range of prompt-
based techniques have been proposed (Fang et al.,
2024b; Press et al., 2023), including workflows that
employ summarization (Kim et al., 2024) and ques-
tion decomposition (Cheng et al., 2025). While
these methods are attractive due to their plug-and-
play nature, they remain fundamentally constrained
by the intrinsic capabilities of their backbone lan-
guage models, which limit the degree of perfor-
mance gain. Alternatives involving explicit train-
ing, such as supervised fine-tuning (Zhang et al.,
2024) or RL (Ren et al., 2025), have also been ex-
plored. However, these training-based approaches
typically overlook the issue of unanswerable or out-
of-scope queries that frequently arise in real-world
2

applications, reducing their practicality. Efforts
to strengthen grounding capability have given rise
to frameworks like HalluGuard (Bergeron et al.,
2025), which combines a large language model
with a preference-optimized smaller model to ver-
ify retrieved evidence prior to answer generation.
Similarly, TrustAlign (Song et al., 2025b) con-
structs a large-scale preference dataset and applies
Directedly Preference-Optimized training (DPO)
in order to improve the alignment of outputs with
trustworthy evidence. Despite the effectiveness,
these methods often entail the generation or cu-
ration of vast amounts of training data, imposing
a significant resource burden. Regarding absten-
tion, most existing approaches adopt prompt engi-
neering (Peng et al., 2025; Chen et al., 2024a) or
pipeline-based workflows (Feng et al., 2024b) to
encourage the model to decline to answer when ap-
propriate. Yet, these methods again depend heavily
on the model’s inherent abstention capability.
Notably, existing methods often treat answer ac-
curacy, evidence grounding, and abstention behav-
ior as independent objectives, rendering the joint
optimization of these dimensions within a unified
framework an open challenge.
3 Method
Our framework comprises three integral compo-
nents: data construction, reward function design,
and training algorithm. To facilitate the subsequent
discussion, we first formalize our problem setting
and notations. With the annotated dataset D, given
thei-th query qi, the knowledge base contains a set
of ground-truth supporting passages Ci
gt, and a set
of distracting passages Ci
dis. Crucially, each ground-
truth passage contains specific key evidence sen-
tences or phrases, denoted as ei
gt, which are es-
sential for answering the query and serve as the
reference for fine-grained verification. Addition-
ally, each query is associated with a ground-truth
answer ai
gt. In conclusion, a dataset sample is rep-
resented as si={qi,Ci
gt,Ci
dis, ai
gt, ei
gt} ∈ D . In the
RAG framework, a retriever fetches the top- kcandi-
date passages Ci
ret={ci
1,···, ci
k}for each query qi
from the knowledge base. We then format the query
and retrieved passages with a prompt template
P(·,·) to construct the prompt pi=P(qi,Ci
ret).
Finally, the LLM πθgenerates an output oicondi-
tioned on the prompt, i.e.,oi=πθ(pi).3.1 Data Construction
Existing RAG datasets (Yang et al., 2018; Dasigi
et al., 2021) predominantly focus on scenarios
where provided evidence is sufficient to answer
a query, i.e., Ci
gt⊆ Ci
ret. This positive-only bias
overlooks the critical need for models to evalu-
ate evidence sufficiency, hindering their ability to
develop a reliable refusal mechanism when the re-
trieved context is irrelevant or incomplete.
A straightforward approach to constructing a
dataset containing both evidence-sufficient and
evidence-insufficient samples is to randomly split
Dinto two subsets. In the first subset, qiis paired
withCi
gtto simulate sufficient evidence scenarios,
while in the second subset, qiis paired with Ci
dis
to create insufficient evidence scenarios. However,
this naive construction has notable drawbacks. The
evidence-sufficient samples contain only ground-
truth evidence without any distractors, significantly
reducing the difficulty of the task. Conversely,
because the insufficient samples are randomly se-
lected from Ci
dis, they may lack relevance to the
query. This makes it trivial for the model to distin-
guish between sufficient and insufficient evidence,
encouraging the learning of superficial patterns
rather than truly enhancing the model’s ability to
discern evidence sufficiency. To address this, we
propose a data construction framework that lever-
ages heterogeneous retrievers to systematically syn-
thesize a balanced training set that forces the model
to distinguish between answering and abstention.
LetRdenote a set of retrievers. For each query
qiand each retriever r∈ R , we retrieve a top-
kcontext set Ci,(r)
ret, yielding |R|context variants
per annotated sample. For each retriever r, we
obtain a retriever-specific training set D(r). No-
tably, the distribution of retrieved samples is often
skewed, as the retrieved set is typically dominated
by candidates that contain key evidence. This class
imbalance poses a substantial challenge for train-
ing models to assess evidence sufficiency accu-
rately. To mitigate this, we randomly split D(r)
into two equally-sized subsets: evidence sufficient
subset D(r)
suffand evidence insufficient subset D(r)
insuff.
To promote diversity in the training data, we em-
ploy retriever-specific random seeding. Building
on the split, we construct the final context for each
annotated sample by manipulating the retrieved
passages while keeping the context length fixed
atkand preserving the retrieval order of the re-
maining passages. Specifically, for si∈ D(r)
suff, we
3

Multi-stage Gated Reward
Data Construction
Annotated
SamplesOracle
RetrieverRL Training
Dense
Retriever
Retriever
Reranker
Random
RetrieverBM25
Retriever
Duplication
Filter
Training
DatasetPolicy 
Model
Answerable
Queries
Unanswerable
Queries
Rollouts
Rollouts
<evidence> ... 
</evidence>
Based on the 
selected 
evidence the 
answer is 
<answer> ...  
</answer>
<llm> ... </llm>
Based on the 
given evidence, 
I can’t answer 
the question. 
<answer> I don’t 
know </answer>
Multi-stage Gated Reward
FormatPath 
SelectionEvidence
AccuracyAnswer
Accuracy
Final
Reward
FormatPath 
SelectionEvidence
AccuracyAnswer
AccuracyFinal
Reward
0
0.5
0.5
0.5 10
0
 0.5
0.5
 1
0
1
0.5
 1
0
0.5
0
0.5
1.5
1.5
 0.48
 0.73
 0.57
 0.812.55
3.04
-1.2
-0.8
0.8
1.2
-1.3
-0.6
0.9
0.9Multi-stage Gated Reward
Retrieved Context
ManipulationFigure 2: Overview of the full pipeline of our proposed method.
enforce the presence of supporting evidence: if
Ci,(r)
retdoes not contain the ground-truth supporting
passage(s) Ci
gt, we replace the lowest-ranked pas-
sage(s) with Ci
gt, leaving the relative order of the
other passages unchanged. For si∈ D(r)
insuff, we
simulate insufficient-evidence scenarios by strictly
excluding ground-truth support from the context.
We remove any retrieved passage that matches Ci
gt
and replace it with distracting passages from Ci
dis,
again preserving the set size. Finally, we aggre-
gate all the retriever-specific datasets {D(r)}r∈R
into a single training corpus with a duplication fil-
ter. We deduplicate the dataset by retaining only
unique pairs of queries qiand retrieved contexts
Ci,(r)
ret. The merged dataset is used as our final train-
ing setD trainandD train=D suff∪ D insuff.
Through the deployment of heterogeneous re-
trievers, we ensure that the retrieved distractor
chunks exhibit varying degrees of semantic rel-
evance to the query. This diverse context forces
the model to discern evidence sufficiency across
multiple levels of difficulty. Furthermore, by ma-
nipulating the retrieved context, we maintain an
equitable distribution between evidence-sufficient
and evidence-insufficient instances. This strategic
balancing prevents the model from falling into re-
ward hacking, where it might otherwise over-fit
to a single decision path to maximize cumulative
rewards. Finally, we apply a deduplication filter to
refine the dataset, guarding against redundant sam-
ples that could skew the model’s learning process.3.2 Multi-stage Gated Reward
As stated in the introduction, an ideal RAG back-
bone model should have the following properties.
First, it must explicitly indicate whether the re-
trieved evidence is sufficient, enabling users to
judge whether the answer should be trusted. Sec-
ond, when the model deems the evidence sufficient,
it should identify the specific sentence(s) on which
the answer is grounded. Finally, when the model
considers the evidence insufficient, it should still
provide a best-effort answer based on its paramet-
ric knowledge, while issuing a clear disclaimer
that the response may be susceptible to errors. To
this end, we design an XML-style response tem-
plate and hope our trained model can internalize
the following behaviors. If the model outputs an
<evidence> tag, it indicates that the model judges
the evidence to be sufficient and includes the key
supporting sentence(s) within the tag. If the model
outputs an <llm> tag, it indicates that the model
judges the evidence to be insufficient and therefore
relies on its internal knowledge to infer an answer,
accompanied by a reliability disclaimer. The model
always outputs an <answer> tag to highlight the
final prediction.
Our reward function is explicitly designed to
satisfy three essential criteria: (1) assessing fea-
sibility of answering, i.e., whether the retrieved
context contains sufficient evidence to answer the
query, (2) identifying the specific supporting evi-
dence when available, and (3) adaptively produc-
ing an evidence-grounded answer, or providing a
4

best-effort answer while explicitly warning that
the retrieved evidence is insufficient. Guided by
these criteria, we adopt a multi-stage gated reward
structure comprising format, path, and content com-
ponents. The total reward R(o) is the cumulative
sum of these components, calculated sequentially.
Format Reward( Rf): To facilitate structured
parsing for rule-based reward calculation, we intro-
duce XML-like special tokens. The output must
strictly adhere to one of two structures: either
an<evidence> ...</evidence> block (indi-
cating sufficient evidence) or an <llm> ...</llm>
block (indicating insufficient evidence and lever-
aging internal parametric knowledge), followed by
an independent <answer> ...</answer> block.
We define a binary format reward Rf: the model
receives a fixed reward of 0.5if the output strictly
complies with these constraints, and 0otherwise.
IfRf= 0, the evaluation terminates immediately,
and the total reward is set to 0. By early-stopping
the reward accumulation upon format failure, we
prevent the model from spurious content rewards
through structurally non-compliant outputs, and
enforce strict adherence to the predefined schema.
Path Selection Reward( Rp): For outputs ad-
hering to the valid format, we evaluate the model’s
decision path. For answerable, evidence sufficient
samples si∈ D suff, the model is expected to invoke
the<evidence> tag to utilize retrieved contexts;
for unanswerable, evidence insufficient samples
si∈ D insuff, it should generate the <llm> tag, in-
dicating the insufficiency of retrieved evidence and
reliance on internal parametric knowledge. A cor-
rect path selection yields Rp= 1. Conversely, an
incorrect path results in Rp= 0, at which point
the episode terminates with a cumulative reward
R(o) =R p+R f= 0.5.
Content Accuracy Reward( Rc): Upon a cor-
rect selection of the <evidence> path, we evalu-
ate the fidelity of the extracted evidence oevidwithin
<evidence> block, and the precision of the final
answer oanswithin <answer> block. We compute
the Rouge-L F1 score for both components against
their respective ground truths. The content accu-
racy reward is defined as a weighted sum: Rc=
α·Rouge-LF1(oevid, egt)+β·Rouge-LF1(oans, agt),
where αandβadjust the relative importance of ev-
idence extraction and answer accuracy. In cases
with multiple references, we utilize the maximum
score across all candidates. If the model correctly
selects the <llm> path,Rcis not computed. While
the model may optionally provide an answer viaparametric knowledge, it remains unscored to dis-
incentivize reliance on memorization and prioritize
evidence-grounded reasoning.
We formulate the total reward R(o) as a step
function to enforce prerequisite constraints:
R(o) =

0invalid format
Rf valid format, wrong path
Rf+Rp correct<llm>path
Rf+Rp+Rccorrect<evidence>path
(1)
This hierarchical design trains the model at two
levels: (i) Decision Calibration, which mandates
the model to assess evidence sufficiency and ab-
stain when necessary; and (ii) Execution Accu-
racy, which ensures that for answerable queries,
the model generates precise responses grounded in
the provided context. To operationalize this multi-
stage gated reward, we provide the reward compu-
tation in pseudocode in Algorithm 1.
3.3 RL Training Stage
Our approach builds upon DAPO (Yu et al., 2025),
an existing on-policy reinforcement learning algo-
rithm. Given an input prompt pi, the LLM pol-
icyπθsamples a group of Gresponses {oi
j}G
j=1.
Each response is evaluated by the reward function
R(·) defined previously, yielding a scalar reward
ri
j=R(oi
j). Subsequently, we calculate the group-
wise advantage as:
ˆAi
j=ri
j−mean
ri
l	G
l=1
std
ri
l	G
l=1 (2)
Consistent with DAPO, this advantage score is as-
signed to every token within the sequence, i.e.,
Ai
j,t=A(oi
j),∀t∈[1,2,···,len(oi
j)].
For each token, we define the policy ratio ρi
j,tas:
ρi
j,t=πθ
oi
j,t|pi, oi
j,<t
πref
oi
j,t|pi, oi
j,<t,(3)
which quantifies the deviation of the current pol-
icy from the reference model. Following DAPO,
we adopt the clipped surrogate objective with the
clip-higher strategy to ensure update stability while
encouraging exploration. The token-level objective
is formally defined as:
li
j,t= min
ρj
j,tAi
j,t,clip 
ρi
j,t,1−ϵ l,1 +ϵ h
Ai
j,t
(4)
5

The asymmetric clipping function bounds the pol-
icy ratio, enforcing conservative updates while per-
mitting larger probability increases for initially low-
probability tokens. Further, we omit the KL penalty
term and utilize a token-level policy gradient loss to
normalize against varying response lengths within
a group. The total lossLis formulated as:
L=Eei∈D train,{oi
j}G
j=1∼π ref(·|pi)

1PG
j=1|oi
j|GX
j=1|oi
j|X
t=1li
j,t
(5)
Notably, our implementation diverges from the
original DAPO by excluding dynamic sampling. In
the original DAPO, dynamic sampling is essential
to handle binary feedback, as groups with identical
rewards yield zero advantage and thus contribute
no learning signal. In contrast, our reward function
returns continuous floating-point values. This con-
tinuous nature ensures reward diversity within each
group, guaranteeing valid advantage estimation and
effective gradient updates without sampling.
4 Experiment
This section provides the main experimental re-
sults. Due to space limitations, we defer addi-
tional results to the Appendix, including: ablation
study (Appendix C.1), performance on different
retrievers (Appendix C.2), comparison of top- k
variants (Appendixheizen C.3), performance on
out-of-distribution data (Appendix C.4), impact of
training on general capabilities (Appendix C.5),
and case study (Appendix C.6).
4.1 Experiment Settings
Our experiments are conducted on two knowledge-
intensive QA datasets: QASPER (Dasigi et al.,
2021) and HotpotQA (Yang et al., 2018). We
use two backbone models, Qwen3-4B (Yang et al.,
2025) and Llama3.1-8B-Instruct (Meta, 2024). For
both datasets, we follow the original train-test splits
to ensure reproducibility. For HotpotQA, we sam-
ple 2,000 training instances and 500 test instances.
For QASPER, we use the full dataset, consisting
of 2,593 training entries and 1,451 test entries. Ad-
ditional implementation details, including baseline
configurations, prompt templates, and hyperparam-
eter setup, are provided in Appendix B.4.2 Baselines
We consider four types of baselines.(1) Prompt-
based methods.This group includes direct prompt-
ing and SuRe (Kim et al., 2024), applied to
Qwen3-4B, Llama3.1-8B-Instruct and DeepSeek-
Chat.(2) SFT-based methods.We perform
sft on Qwen3-4B and Llama3.1-8B-Instruct with
our constructed training data (details are given
in Appendix B.3).(3) RL-based methodscom-
prise ARENA (Ren et al., 2025) and TrustAl-
ign (Song et al., 2025b).(4) Agentic methods
include SimpleDeepSearcher (Sun et al., 2025), R1
Searcher (Song et al., 2025a), and Search-R1 (Jin
et al., 2025).
To ensure a fair and faithful comparison, we eval-
uate all baselines with their original inference tem-
plates from their respective papers. On top of these
baseline-specific templates, we prepend aunified
three-shot demonstrationcontaining two answer-
able examples with correct answers and one unan-
swerable example with a refusal-style response.
For agentic RAG baselines, we directly feed our re-
trieved chunks to the model. If the model executes
an<answer> action, we treat it as indicating that
the evidence is sufficient; if it instead executes a
<search> action, we treat it as abstaining. To
maintain consistency with our experimental setup,
all methods are limited to a single interaction turn.
4.3 Metrics
To evaluate the model performance comprehen-
sively, we divide the test set into two sub-
sets:answerableandunanswerable, depending
on whether the ground-truth evidence chunks
are successfully retrieved. For answerable
queries, we assess answer quality using Ex-
act Match (EM), BERT F1 Score (F1) by
bert-baes-uncased , and accuracy of LLM-
as-a-judge (LJ) by DeepSeek-V3.2-chat . For
unanswerable queries, we examine the model’s abil-
ity to follow instructions and refuse responses, eval-
uated by EM and LJ, respectively. Notably, we
prioritize the presence of a reliability disclaimer
over the factual correctness of the answer; a re-
sponse is deemed incorrect if the model fails to
signal evidence insufficiency, even if it happens to
guess the correct answer using its internal knowl-
edge. To quantify performance, we report Accu-
racy (Acc.) to reflect the model’s effectiveness
under a real-world distribution, and Balanced Ac-
curacy (B. Acc.) to evaluate its ability to answer
6

Table 1: Main Results on QASPER and HotpotQA, where the evidences are retrieved by Qwen3-Embedding-0.6B
withk= 3 . Thebest resultsare bold, and the runner-up results are underlined. The number of answerable and
unanswerable questions in each dataset is denoted as (answerable: unanswerable). Methods with †sign come from
the released checkpoints by the corresponding authors.
QASPER(1200 : 251)HotpotQA(296:204)
Answerable UnanswerableOverallAnswerable UnanswerableOverall
EM F1 LJ Acc. LJ Acc. B. Acc.EM F1 LJ Acc. LJ Acc. B. Acc.
Prompt based
Qwen3-4B 0.0 50.2378.835.58 42.23 72.50 60.53 17.57 54.31 89.53 16.18 60.78 77.80 75.16
Llama3-8B-Instruct 1.33 54.60 70.58 12.75 30.68 63.68 50.63 8.78 39.15 27.36 48.53 89.22 52.60 58.29
DeepSeek-chat 0.0 51.88 69.33 50.60 72.11 69.81 69.57 33.45 60.46 68.92 75.00 75.98 71.80 72.45
SuRe on Qwen3-4B 5.92 52.43 46.17 0.0 0.40 38.25 23.28 26.35 63.62 65.54 0.0 1.47 39.40 33.51
SuRe on Llama3-8B-Instruct 5.25 55.01 54.83 0.0 1.59 45.62 28.21 35.14 69.18 83.11 0.0 0.0 49.20 41.55
SuRe on DeepSeek-chat 13.08 58.68 57.92 0.0 7.17 49.14 32.54 51.69 78.49 84.12 0.0 0.98 50.20 42.55
SFT based
Qwen3-4B28.9269.53 62.25 67.33 67.33 63.13 64.79 64.53 84.77 84.12 44.12 44.12 67.80 64.12
Llama3-8B-Instruct 27.92 70.21 62.75 62.95 62.95 62.78 62.8567.2386.38 87.50 49.51 50.49 72.40 69.00
RL based
TrustAlign-3B†0.0 41.20 41.17 78.09 80.88 48.04 61.03 0.0 38.66 44.59 55.88 67.16 53.80 55.88
TrustAlign on Qwen3-4B 0.0 44.50 36.83 0.0 60.96 41.01 48.90 0.0 32.27 11.49 0.096.57 46.20 54.03
TrustAlign-7B†0.0 43.63 33.67 0.0 88.45 43.14 38.41 0.0 34.02 24.32 0.0 86.76 49.80 55.54
TrustAlign-8B†0.08 34.72 9.6797.61 97.61 24.88 53.64 0.0 30.57 22.6488.7388.73 49.60 55.69
ARENA on Qwen3-4B 23.25 68.08 73.50 19.92 20.72 64.37 47.11 64.86 86.05 90.54 38.73 38.73 69.40 64.64
ARENA-7B†22.50 69.73 69.67 4.78 7.97 58.99 38.82 65.54 86.27 88.18 5.88 9.31 55.60 48.75
ARENA-8B†23.58 68.66 69.17 0.0 0.40 57.27 34.79 64.53 85.43 87.84 0.0 0.0 52.00 43.92
Agentic Models
R1-Searcher-7B†15.42 59.80 48.83 17.13 19.92 43.83 34.38 51.35 74.59 69.59 65.20 65.20 67.80 67.40
Search-R1-7B†18.50 64.76 61.00 30.68 33.07 56.17 47.03 56.08 79.49 78.72 47.55 47.55 66.00 63.13
SimpleDeepSearcher-7B†22.92 67.13 67.17 16.73 17.93 58.65 42.55 45.61 68.47 63.18 64.71 64.71 63.80 63.94
Our methods
GRACE-Qwen3-4B 27.0873.0974.25 74.90 74.90 74.36 74.5865.20 84.9591.5566.18 66.18 81.20 78.87
GRACE-Llama3.1-8B 27.08 72.02 70.08 74.50 74.50 70.85 72.29 65.2086.4088.85 42.16 44.61 70.80 66.73
and abstain with equal importance.
4.4 Main Results
As shown in Table 1, our method GRACE-Qwen3-
4B model achieves the best overall performance in
both accuracy and balanced accuracy across two
datasets, which can effectively balance precise an-
swering and appropriate refusal. Specifically, com-
pared with the untrained base model, GRACE im-
proves the unanswerable LJ score from 42.23% to
74.90% on QASPER and the answerable EM score
from 17.57% to 65.20% on HotpotQA. Similar
trends are also observed in GRACE-Llama3.1-8B
model. All these results validate the generalization
and effectiveness of our approach across different
backbone architectures.
Regarding prompt-based methods, it is notewor-
thy that they can achieve surprisingly competi-
tive performance when provided with appropri-
ate prompts. For example, Qwen3-4B achieves
runner-up overall accuracy across both bench-
marks; Llama3-8B-Instruct’s unanswerable LJ
score is 89.22% on HotpotQA, which is the runner-
up; DeepSeek-chat ranks second in the unanswer-
able accuracy on HotpotQA. However, when in-jected with more sophisticated prompting strate-
gies such as SuRe, we observe a drastic collapse in
refusal capabilities for unanswerable queries. For
instance, DeepSeek-chat’s refusal accuracy on Hot-
potQA plummets from 75.98% to a mere 0.98%.
We attribute this failure to SuRe’s inherent assump-
tion that the retrieved evidence is necessarily suffi-
cient to support an answer, leaving little room for
the model to acknowledge information missing.
For SFT-based approaches, we observe substan-
tial improvements in answerable EM, F1 scores,
and unanswerable accuracy over base models after
training. For instance, the SFT-trained Llama3.1-
8B-Instruct achieves the highest EM score and the
runner-up F1 score on HotpotQA, indicating that
our constructed data effectively guides the model to
adhere to the required output formats. Conversely,
their performance on the LJ metric exhibits insta-
bility. We attribute this to the absence of Chain-
of-Thought (CoT) reasoning in our training data,
which likely leads the model to superficially mem-
orize answer templates rather than internalize the
underlying reasoning process needed to solve the
problem (Lin et al., 2025; Yao et al., 2025).
7

/uni0000002a/uni00000055/uni00000044/uni00000046/uni00000048
/uni0000002f/uni0000004f/uni00000044/uni00000050/uni00000044/uni0000002a/uni00000055/uni00000044/uni00000046/uni00000048
/uni00000034/uni0000005a/uni00000048/uni00000051/uni00000035/uni00000014/uni00000010
/uni00000036/uni00000048/uni00000044/uni00000055/uni00000046/uni0000004b/uni00000048/uni00000055/uni00000036/uni00000048/uni00000044/uni00000055/uni00000046/uni0000004b/uni00000010
/uni00000035/uni00000014/uni00000036/uni0000004c/uni00000050/uni00000053/uni0000004f/uni00000048
/uni00000027/uni00000048/uni00000048/uni00000053/uni00000036/uni00000048/uni00000044/uni00000055/uni00000046/uni0000004b/uni00000048/uni00000055
/uni00000030/uni00000048/uni00000057/uni0000004b/uni00000052/uni00000047/uni00000003/uni00000031/uni00000044/uni00000050/uni00000048/uni00000013/uni00000008/uni00000015/uni00000013/uni00000008/uni00000017/uni00000013/uni00000008/uni00000019/uni00000013/uni00000008/uni0000001b/uni00000013/uni00000008/uni00000014/uni00000013/uni00000013/uni00000008/uni00000033/uni00000055/uni00000052/uni00000053/uni00000052/uni00000055/uni00000057/uni0000004c/uni00000052/uni00000051/uni00000003/uni00000052/uni00000049/uni00000003/uni00000036/uni00000044/uni00000050/uni00000053/uni0000004f/uni00000048/uni00000056/uni00000003/uni0000000b/uni00000008/uni0000000c/uni00000034/uni00000024/uni00000036/uni00000033/uni00000028/uni00000035
/uni0000001a/uni00000015/uni00000011/uni0000001c/uni00000008/uni0000001a/uni00000017/uni00000011/uni0000001b/uni00000008 /uni0000001a/uni00000018/uni00000011/uni0000001b/uni00000008 /uni0000001a/uni00000019/uni00000011/uni0000001c/uni00000008 /uni0000001a/uni0000001a/uni00000011/uni00000018/uni00000008
/uni0000002a/uni00000055/uni00000044/uni00000046/uni00000048
/uni0000002f/uni0000004f/uni00000044/uni00000050/uni00000044/uni0000002a/uni00000055/uni00000044/uni00000046/uni00000048
/uni00000034/uni0000005a/uni00000048/uni00000051/uni00000035/uni00000014/uni00000010
/uni00000036/uni00000048/uni00000044/uni00000055/uni00000046/uni0000004b/uni00000048/uni00000055/uni00000036/uni00000048/uni00000044/uni00000055/uni00000046/uni0000004b/uni00000010
/uni00000035/uni00000014/uni00000036/uni0000004c/uni00000050/uni00000053/uni0000004f/uni00000048
/uni00000027/uni00000048/uni00000048/uni00000053/uni00000036/uni00000048/uni00000044/uni00000055/uni00000046/uni0000004b/uni00000048/uni00000055
/uni00000030/uni00000048/uni00000057/uni0000004b/uni00000052/uni00000047/uni00000003/uni00000031/uni00000044/uni00000050/uni00000048/uni00000013/uni00000008/uni00000015/uni00000013/uni00000008/uni00000017/uni00000013/uni00000008/uni00000019/uni00000013/uni00000008/uni0000001b/uni00000013/uni00000008/uni00000014/uni00000013/uni00000013/uni00000008/uni00000033/uni00000055/uni00000052/uni00000053/uni00000052/uni00000055/uni00000057/uni0000004c/uni00000052/uni00000051/uni00000003/uni00000052/uni00000049/uni00000003/uni00000036/uni00000044/uni00000050/uni00000053/uni0000004f/uni00000048/uni00000056/uni00000003/uni0000000b/uni00000008/uni0000000c/uni0000002b/uni00000052/uni00000057/uni00000053/uni00000052/uni00000057/uni00000034/uni00000024
/uni0000001a/uni0000001a/uni00000011/uni00000019/uni00000008/uni0000001b/uni00000016/uni00000011/uni0000001a/uni00000008
/uni0000001a/uni00000014/uni00000011/uni00000017/uni00000008/uni0000001a/uni00000017/uni00000011/uni00000015/uni00000008
/uni00000019/uni0000001a/uni00000011/uni0000001a/uni00000008/uni00000026/uni00000052/uni00000055/uni00000055/uni00000048/uni00000046/uni00000057/uni0000004f/uni0000005c/uni00000003/uni00000026/uni0000004f/uni00000044/uni00000056/uni00000056/uni0000004c/uni00000049/uni0000004c/uni00000048/uni00000047 /uni00000026/uni00000052/uni00000055/uni00000055/uni00000048/uni00000046/uni00000057/uni00000003/uni00000024/uni00000051/uni00000056/uni0000005a/uni00000048/uni00000055 /uni00000037/uni00000055/uni00000058/uni00000048/uni00000003/uni00000031/uni00000048/uni0000004a/uni00000044/uni00000057/uni0000004c/uni00000059/uni00000048 /uni0000002c/uni00000051/uni00000046/uni00000052/uni00000055/uni00000055/uni00000048/uni00000046/uni00000057/uni00000003/uni00000024/uni00000051/uni00000056/uni0000005a/uni00000048/uni00000055 /uni00000029/uni00000044/uni0000004f/uni00000056/uni00000048/uni00000003/uni00000031/uni00000048/uni0000004a/uni00000044/uni00000057/uni0000004c/uni00000059/uni00000048 /uni00000029/uni00000044/uni0000004f/uni00000056/uni00000048/uni00000003/uni00000033/uni00000052/uni00000056/uni0000004c/uni00000057/uni0000004c/uni00000059/uni00000048Figure 3: Error analysis of the proposed method versus agentic RAG baselines. Classification accuracy is indicated
by the “Correctly Classified” line. Proportions are calculated by the average results among different retrievers.
When examining RL-based approaches, we find
TrustAlign and ARENA exhibit two diverging be-
haviors. While TrustAlign achieves a near-perfect
refusal accuracy (96.57%), it struggles to correctly
address answerable queries. This phenomenon sug-
gests a case of over-alignment, where the model
becomes overly conservative and reluctant to re-
spond. In contrast, ARENA performs well on an-
swerable queries but fails on unanswerable ones.
This is primarily because its reward mechanism
fails to account for scenarios where evidence is
missing; consequently, the model resorts to reward
hacking, attempting to fabricate an answer even
when a refusal is warranted, solely to maximize
its reward signal. These observations suggest that
both models are specialists rather than generalists.
4.5 Comparison with Agentic Models
We also incorporate recently prominent Agen-
tic RAG models SimpleDeepSearcher-7B, R1-
Searcher-7B and Search-R1-7B, treating their eval-
uation as a verification of single-step performance
within the RAG context. Despite their sophisticated
reasoning and tool-use capabilities, these models
exhibit worse performance in our single-turn sce-
narios than GRACE. Notably, their overall perfor-
mance significantly lags behind our method, even
the strongest agentic baseline trails our GRACE-
Qwen3-4B by more than 15% in overall accuracy
across both datasets (see Table 1).
To investigate this discrepancy, we formulate
evidence sufficiency identification as a binary clas-
sification task, where Positive (P) denotes sufficient
evidence and Negative (N) denotes insufficient ev-
idence. We categorize the results into True Posi-tive (TP), True Negative (TN), False Positive (FP),
and False Negative (FN). Within the TP group,
we further distinguish between Correct (C) and In-
correct (I) answers, such that TP=C+I . To
quantify this, we define classification accuracy as
the success rate in identifying evidence sufficiency,
TP+TN
Total, and answer precisionC
C+I.
As illustrated in Figure 3, on QASPER, while
baseline agentic models achieve classification ac-
curacy comparable to GRACE (see black dashed
curve), their answer precision is significantly lower.
We attribute this to a domain mismatch: these mod-
els are typically trained on general open-domain
data, whereas QASPER focuses on specialized aca-
demic QA, where identifying evidence sufficiency
is relatively straightforward, but generating techni-
cally accurate answers is more challenging. More-
over, these baselines exhibit a high False Positive
Rate (FPR), indicating a tendency toward overcon-
fidence. Conversely, on HotpotQA, the multi-hop,
open-domain benchmark, these models exhibit a
notably weaker capacity for evidence assessment
compared to GRACE, where agentic models score
approximately 70%, whereas our GRACE-Qwen3-
4B achieves 83.7%. We attribute this deficiency to
the limitations of sparse outcome rewards used in
GRPO-based training. Without dense, step-wise
supervision, a model may receive positive rein-
forcement for a correct final answer even if its
intermediate reasoning step is flawed. This sup-
ports the view that stronger performance stems
from dense and reliable verification signals, rather
than sparse outcome-based rewards, echoing the
shift from DeepSeek-Math (Shao et al., 2024) to
DeepSeek-Math-V2 (Shao et al., 2025).
8

5 Conclusion
In this paper, we presented GRACE, an RL frame-
work dedicated to simultaneously developing reli-
able and transparent RAG backbone models. We
proposed a pipeline to construct answerable and
unanswerable training samples from heterogeneous
retrievers. Then, we designed a multi-stage gated
reward mechanism that provides multi-aspect feed-
back, empowering models to discern evidence suffi-
ciency and perform explicit grounding or informed
refusal. Finally, we proposed a modified DAPO
algorithm to ensure training stability. Empirical re-
sults demonstrate that GRACE outperforms all the
baselines and achieve a superior synergy between
answer correctness and refusal reliability.
Limitations
Despite the promising results, our work has two
main limitations. First, constrained by computa-
tional resources, we could not extend our evalu-
ation to larger models or conduct multiple runs
to mitigate the impact of experimental random-
ness. While our method proves effective on smaller
scales, verifying its scalability and performance
on models with significantly larger parameters re-
mains an important direction for future investiga-
tion. Second, our approach currently relies on the
availability of ground-truth key evidence for each
query during training. This dependency on fine-
grained annotations limits our ability to test on a
broader range of datasets where such detailed la-
bels are absent. Exploring methods to reduce this
reliance, for instance, through weak supervision or
automated evidence extraction, will be a primary
focus of our future work.
Ethical Consideration
Potential Risks.Our model is designed for aca-
demic research in information retrieval and reason-
ing. Although it is trained for a more trustworthy
RAG backbone model, we still recognize the po-
tential risk of generating factually incorrect infor-
mation (hallucinations).
License, Intended Use, and Sensitive Informa-
tion.All data and software utilized in this study
are derived from publicly available academic re-
sources. Our primary datasets, QASPER and Hot-
potQA, are distributed under CC-BY-4.0 and CC-
BY-SA-4.0 licenses, respectively, both of which
permit use for academic research. These datasetscontain no sensitive information, including unique
identifiers of individual people or offensive con-
tent. Furthermore, we leverage several open-source
frameworks and baselines. The codebases for
VeRL and LLaMA Factory are governed by the
Apache 2.0 License, while ARENA is provided
under the MIT License. For repositories where
a specific license was not explicitly provided, we
have used them strictly in accordance with their
intended research purposes. All model checkpoints
used are open-source weights hosted on Hugging
Face; specifically, the TrustAlign series follows
CC-BY-4.0, and the ARENA series follows the
MIT License. Other models (SimpleDeepSearcher,
R1-Searcher, and Search R1) were accessed via
their public repositories.
Documentation of Artifacts.Regarding artifact
documentation, our research primarily focuses
on English. While the core tasks are English-
centric, we observe incidental occurrences of
other languages—such as Simplified Chinese and
French—within certain Wikipedia passages in Hot-
potQA. These typically appear as localized names
or references within the predominantly English text.
In terms of domain coverage, QASPER spans di-
verse scientific literature, while HotpotQA provides
broad coverage of general-world knowledge de-
rived from Wikipedia. We have verified that these
artifacts contain no sensitive demographic informa-
tion and align with their original intended use for
academic research.
References
Akari Asai, Zeqiu Wu, Yizhong Wang, Avi Sil, and
Hannaneh Hajishirzi. 2024. Self-rag: Learning to re-
trieve, generate, and critique through self-reflection.
InInternational Conference on Representation Learn-
ing, volume 2024, pages 9112–9141.
Loris Bergeron, Ioana Buhnila, Jérôme François,
and Radu State. 2025. Halluguard: Evidence-
grounded small reasoning models to mitigate halluci-
nations in retrieval-augmented generation.Preprint,
arXiv:2510.00880.
Howard Chen, Noam Razin, Karthik Narasimhan, and
Danqi Chen. 2025. Retaining by doing: The role
of on-policy data in mitigating forgetting.Preprint,
arXiv:2510.18874.
Lu Chen, Ruqing Zhang, Jiafeng Guo, Yixing Fan, and
Xueqi Cheng. 2024a. Controlling risk of retrieval-
augmented generation: A counterfactual prompting
9

framework. InFindings of the Association for Com-
putational Linguistics: EMNLP 2024, pages 2380–
2393, Miami, Florida, USA. Association for Compu-
tational Linguistics.
Ruirui Chen, Weifeng Jiang, Chengwei Qin,
Ishaan Singh Rawal, Cheston Tan, Dongkyu
Choi, Bo Xiong, and Bo Ai. 2024b. LLM-based
multi-hop question answering with knowledge graph
integration in evolving environments. InFindings
of the Association for Computational Linguistics:
EMNLP 2024, pages 14438–14451, Miami, Florida,
USA. Association for Computational Linguistics.
Xinxi Chen, Li Wang, Wei Wu, Qi Tang, and Yiyao Liu.
2024c. Honest ai: Fine-tuning "small" language mod-
els to say "i don’t know", and reducing hallucination
in rag.Preprint, arXiv:2410.09699.
Yao Cheng, Yibo Zhao, Jiapeng Zhu, Yao Liu, Xing Sun,
and Xiang Li. 2025. Human cognition inspired rag
with knowledge graph for complex problem solving.
Preprint, arXiv:2503.06567.
Yuqin Dai, Guoqing Wang, Yuan Wang, Kairan Dou,
Kaichen Zhou, Zhanwei Zhang, Shuo Yang, Fei
Tang, Jun Yin, Pengyu Zeng, Zhenzhe Ying, Can Yi,
Changhua Meng, Yuchen Zhou, Yongliang Shen, and
Shuai Lu. 2025. Evinote-rag: Enhancing rag mod-
els via answer-supportive evidence notes.Preprint,
arXiv:2509.00877.
Pradeep Dasigi, Kyle Lo, Iz Beltagy, Arman Cohan,
Noah A. Smith, and Matt Gardner. 2021. A dataset
of information-seeking questions and answers an-
chored in research papers. InProceedings of the
2021 Conference of the North American Chapter of
the Association for Computational Linguistics: Hu-
man Language Technologies, pages 4599–4610, On-
line. Association for Computational Linguistics.
Yang Deng, Moxin Li, Liang Pang, Wenxuan Zhang,
and Wai Lam. 2025. Unveiling knowledge bound-
ary of large language models for trustworthy infor-
mation access. InProceedings of the 48th Inter-
national ACM SIGIR Conference on Research and
Development in Information Retrieval, SIGIR ’25,
page 4086–4089, New York, NY , USA. Association
for Computing Machinery.
Jinyuan Fang, Zaiqiao Meng, and Craig MacDon-
ald. 2024a. TRACE the evidence: Constructing
knowledge-grounded reasoning chains for retrieval-
augmented generation. InFindings of the Association
for Computational Linguistics: EMNLP 2024, pages
8472–8494, Miami, Florida, USA. Association for
Computational Linguistics.
Jinyuan Fang, Zaiqiao Meng, and Craig MacDon-
ald. 2024b. TRACE the evidence: Constructing
knowledge-grounded reasoning chains for retrieval-
augmented generation. InFindings of the Association
for Computational Linguistics: EMNLP 2024, pages
8472–8494, Miami, Florida, USA. Association for
Computational Linguistics.Shangbin Feng, Weijia Shi, Yike Wang, Wenxuan Ding,
Vidhisha Balachandran, and Yulia Tsvetkov. 2024a.
Don’t hallucinate, abstain: Identifying LLM knowl-
edge gaps via multi-LLM collaboration. InProceed-
ings of the 62nd Annual Meeting of the Association
for Computational Linguistics (Volume 1: Long Pa-
pers), pages 14664–14690, Bangkok, Thailand. As-
sociation for Computational Linguistics.
Shangbin Feng, Weijia Shi, Yike Wang, Wenxuan Ding,
Vidhisha Balachandran, and Yulia Tsvetkov. 2024b.
Don’t hallucinate, abstain: Identifying LLM knowl-
edge gaps via multi-LLM collaboration. InProceed-
ings of the 62nd Annual Meeting of the Association
for Computational Linguistics (Volume 1: Long Pa-
pers), pages 14664–14690, Bangkok, Thailand. As-
sociation for Computational Linguistics.
Wentao Hu, Wengyu Zhang, Yiyang Jiang, Chen Jason
Zhang, Xiaoyong Wei, and Li Qing. 2025. Removal
of hallucination on hallucination: Debate-augmented
RAG. InProceedings of the 63rd Annual Meeting of
the Association for Computational Linguistics (Vol-
ume 1: Long Papers), pages 15839–15853, Vienna,
Austria. Association for Computational Linguistics.
Bowen Jin, Hansi Zeng, Zhenrui Yue, Jinsung Yoon, Ser-
can O Arik, Dong Wang, Hamed Zamani, and Jiawei
Han. 2025. Search-r1: Training LLMs to reason and
leverage search engines with reinforcement learning.
InSecond Conference on Language Modeling.
Hyuhng Joon Kim, Youna Kim, Sang-goo Lee, and
Taeuk Kim. 2025. When to speak, when to abstain:
Contrastive decoding with abstention. InProceed-
ings of the 63rd Annual Meeting of the Association
for Computational Linguistics (Volume 1: Long Pa-
pers), pages 9710–9730, Vienna, Austria. Associa-
tion for Computational Linguistics.
Jaehyung Kim, Jaehyun Nam, Sangwoo Mo, Jongjin
Park, Sang-Woo Lee, Minjoon Seo, Jung-Woo Ha,
and Jinwoo Shin. 2024. Sure: Summarizing re-
trievals using answer candidates for open-domain QA
of LLMs. InThe Twelfth International Conference
on Learning Representations.
Patrick Lewis, Ethan Perez, Aleksandra Piktus, Fabio
Petroni, Vladimir Karpukhin, Naman Goyal, Hein-
rich Küttler, Mike Lewis, Wen-tau Yih, Tim Rock-
täschel, Sebastian Riedel, and Douwe Kiela. 2020.
Retrieval-augmented generation for knowledge-
intensive nlp tasks. InAdvances in Neural Infor-
mation Processing Systems, volume 33, pages 9459–
9474. Curran Associates, Inc.
Moxin Li, Yong Zhao, Wenxuan Zhang, Shuaiyi Li,
Wenya Xie, See-Kiong Ng, Tat-Seng Chua, and Yang
Deng. 2025. Knowledge boundary of large language
models: A survey. InProceedings of the 63rd Annual
Meeting of the Association for Computational Lin-
guistics (Volume 1: Long Papers), pages 5131–5157,
Vienna, Austria. Association for Computational Lin-
guistics.
10

Xiaofeng Lin, Hejian Sang, Zhipeng Wang, and
Xuezhou Zhang. 2025. Debunk the myth of sft gen-
eralization.Preprint, arXiv:2510.00237.
Yuanjie Lyu, Zihan Niu, Zheyong Xie, Chao Zhang,
Tong Xu, Yang Wang, and Enhong Chen. 2024.
Retrieve-plan-generation: An iterative planning and
answering framework for knowledge-intensive LLM
generation. InProceedings of the 2024 Conference
on Empirical Methods in Natural Language Process-
ing, pages 4683–4702, Miami, Florida, USA. Associ-
ation for Computational Linguistics.
Meta. 2024. Llama3.1-8b-instruct model card.
https://huggingface.co/meta-llama/
Llama-3.1-8B-Instruct . Accessed: 2025-
12-12.
Sagnik Mukherjee, Lifan Yuan, Dilek Hakkani-Tur,
and Hao Peng. 2025. Reinforcement learning fine-
tunes small subnetworks in large language models.
Preprint, arXiv:2505.11711.
Xiangyu Peng, Prafulla Kumar Choubey, Caiming
Xiong, and Chien-Sheng Wu. 2025. Unanswerabil-
ity evaluation for retrieval augmented generation. In
Proceedings of the 63rd Annual Meeting of the As-
sociation for Computational Linguistics (Volume 1:
Long Papers), pages 8452–8472, Vienna, Austria.
Association for Computational Linguistics.
Ofir Press, Muru Zhang, Sewon Min, Ludwig Schmidt,
Noah Smith, and Mike Lewis. 2023. Measuring and
narrowing the compositionality gap in language mod-
els. InFindings of the Association for Computational
Linguistics: EMNLP 2023, pages 5687–5711, Singa-
pore. Association for Computational Linguistics.
Shenbin Qian, Archchana Sindhujan, Minnie Kabra,
Diptesh Kanojia, Constantin Orasan, Tharindu Ranas-
inghe, and Fred Blain. 2024. What do large lan-
guage models need for machine translation evalua-
tion? InProceedings of the 2024 Conference on
Empirical Methods in Natural Language Processing,
pages 3660–3674, Miami, Florida, USA. Association
for Computational Linguistics.
Jingyi Ren, Yekun Xu, Xiaolong Wang, Weitao Li,
Weizhi Ma, and Yang Liu. 2025. Transparent and ro-
bust rag: Adaptive-reward reinforcement learning for
decision traceability.Preprint, arXiv:2505.13258.
Zhihong Shao, Yuxiang Luo, Chengda Lu, Z. Z. Ren,
Jiewen Hu, Tian Ye, Zhibin Gou, Shirong Ma, and
Xiaokang Zhang. 2025. Deepseekmath-v2: Towards
self-verifiable mathematical reasoning.Preprint,
arXiv:2511.22570.
Zhihong Shao, Peiyi Wang, Qihao Zhu, Runxin Xu,
Junxiao Song, Xiao Bi, Haowei Zhang, Mingchuan
Zhang, Y . K. Li, Y . Wu, and Daya Guo. 2024.
Deepseekmath: Pushing the limits of mathemati-
cal reasoning in open language models.Preprint,
arXiv:2402.03300.Huatong Song, Jinhao Jiang, Yingqian Min, Jie Chen,
Zhipeng Chen, Wayne Xin Zhao, Lei Fang, and Ji-
Rong Wen. 2025a. R1-searcher: Incentivizing the
search capability in llms via reinforcement learning.
Preprint, arXiv:2503.05592.
Maojia Song, Shang Hong Sim, Rishabh Bhardwaj,
Hai Leong Chieu, Navonil Majumder, and Soujanya
Poria. 2025b. Measuring and enhancing trustworthi-
ness of LLMs in RAG through grounded attributions
and learning to refuse. InThe Thirteenth Interna-
tional Conference on Learning Representations.
Shuang Sun, Huatong Song, Yuhao Wang, Ruiyang Ren,
Jinhao Jiang, Junjie Zhang, Fei Bai, Jia Deng, Xin
Zhao, Zheng Liu, Lei Fang, Zhongyuan Wang, and
Ji-Rong Wen. 2025. SimpleDeepSearcher: Deep
information seeking via web-powered reasoning tra-
jectory synthesis. InFindings of the Association
for Computational Linguistics: EMNLP 2025, pages
13705–13720, Suzhou, China. Association for Com-
putational Linguistics.
Sirui Xia, Xintao Wang, Jiaqing Liang, Yifei Zhang,
Weikang Zhou, Jiaji Deng, Fei Yu, and Yanghua Xiao.
2025. Ground every sentence: Improving retrieval-
augmented LLMs with interleaved reference-claim
generation. InFindings of the Association for Com-
putational Linguistics: NAACL 2025, pages 969–988,
Albuquerque, New Mexico. Association for Compu-
tational Linguistics.
Huajian Xin, Daya Guo, Zhihong Shao, Zhizhou Ren,
Qihao Zhu, Bo Liu, Chong Ruan, Wenda Li, and
Xiaodan Liang. 2024. Deepseek-prover: Advancing
theorem proving in llms through large-scale synthetic
data.Preprint, arXiv:2405.14333.
Haoran Xu, Amr Sharaf, Yunmo Chen, Weiting Tan,
Lingfeng Shen, Benjamin Van Durme, Kenton Mur-
ray, and Young Jin Kim. 2024a. Contrastive prefer-
ence optimization: pushing the boundaries of llm per-
formance in machine translation. InProceedings of
the 41st International Conference on Machine Learn-
ing, ICML’24. JMLR.org.
Shicheng Xu, Liang Pang, Huawei Shen, Xueqi Cheng,
and Tat-Seng Chua. 2024b. Search-in-the-chain: In-
teractively enhancing large language models with
search for knowledge-intensive tasks. InProceed-
ings of the ACM Web Conference 2024, WWW ’24,
page 1362–1373, New York, NY , USA. Association
for Computing Machinery.
An Yang, Anfeng Li, Baosong Yang, Beichen Zhang,
Binyuan Hui, Bo Zheng, Bowen Yu, Chang Gao,
Chengen Huang, Chenxu Lv, Chujie Zheng, Dayi-
heng Liu, Fan Zhou, Fei Huang, Feng Hu, Hao Ge,
Haoran Wei, Huan Lin, Jialong Tang, and 41 oth-
ers. 2025. Qwen3 technical report.arXiv preprint
arXiv:2505.09388.
Zhilin Yang, Peng Qi, Saizheng Zhang, Yoshua Bengio,
William Cohen, Ruslan Salakhutdinov, and Christo-
pher D. Manning. 2018. HotpotQA: A dataset for
11

diverse, explainable multi-hop question answering.
InProceedings of the 2018 Conference on Empiri-
cal Methods in Natural Language Processing, pages
2369–2380, Brussels, Belgium. Association for Com-
putational Linguistics.
Xinhao Yao, Ruifeng Ren, Yun Liao, and Yong Liu.
2025. Unveiling the mechanisms of explicit cot
training: How cot enhances reasoning generalization.
Preprint, arXiv:2502.04667.
Qiying Yu, Zheng Zhang, Ruofei Zhu, Yufeng Yuan,
Xiaochen Zuo, Yu Yue, Weinan Dai, Tiantian Fan,
Gaohong Liu, Lingjun Liu, Xin Liu, Haibin Lin,
Zhiqi Lin, Bole Ma, Guangming Sheng, Yuxuan
Tong, Chi Zhang, Mofan Zhang, Wang Zhang, and
16 others. 2025. Dapo: An open-source llm re-
inforcement learning system at scale.Preprint,
arXiv:2503.14476.
Yue Yu, Wei Ping, Zihan Liu, Boxin Wang, Jiaxuan
You, Chao Zhang, Mohammad Shoeybi, and Bryan
Catanzaro. 2024. Rankrag: Unifying context ranking
with retrieval-augmented generation in llms. InAd-
vances in Neural Information Processing Systems,
volume 37, pages 121156–121184. Curran Asso-
ciates, Inc.
Tianjun Zhang, Shishir G Patil, Naman Jain, Sheng
Shen, Matei Zaharia, Ion Stoica, and Joseph E. Gon-
zalez. 2024. RAFT: Adapting language model to
domain specific RAG. InFirst Conference on Lan-
guage Modeling.
Xinping Zhao, Dongfang Li, Yan Zhong, Boren Hu,
Yibin Chen, Baotian Hu, and Min Zhang. 2024.
SEER: Self-aligned evidence extraction for retrieval-
augmented generation. InProceedings of the 2024
Conference on Empirical Methods in Natural Lan-
guage Processing, pages 3027–3041, Miami, Florida,
USA. Association for Computational Linguistics.
Youchao Zhou, Heyan Huang, Yicheng Liu, Rui Dai,
Xinglin Wang, Xingchen Zhang, Shumin Shi, and
Yang Deng. 2025. Do retrieval augmented language
models know when they don’t know?Preprint,
arXiv:2509.01476.
12

Appendix
A Pseudo-code for the Reward Function
Algorithm 1Pseudo-code for the reward function.
Require: Model output o, ground-truth evidence
sete gt, ground-truth answer seta gt
Ensure:RewardR(o)
1:y←extract the portion ofoafter “</think>”
2:R f←CheckFormat(y)
3:ifR f= 0then
4:return0
5:end if{format violation→terminate}
6:(R p, c)←CheckPath(y)
7:ifR p= 0then
8:returnR f
9:end if{wrong path→terminate with format
reward only}
10:ifc=<evidence>then
11:o evid←ExtractTag(y,<evidence>)
12:o ans←ExtractTag(y,<answer>)
13:r e←max
e∈egtROUGE_L(o evid, e)
14:r a←max
a∈agtROUGE_L(o ans, a)
15:R c←αr e+βr a
16:returnR f+R p+R c
17:else
18:returnR f+R p
19:end if{<llm> path→no content evaluation}
B Detail Settings
B.1 Configurations
To construct the training set, we employ diverse
retrieval strategies with k=5: an oracle retriever,
BM25 (k1=1.5,b=0.75), an embedding-based re-
triever (Qwen3-0.6B-Embedding), a retrieve-then-
rerank pipeline (Qwen3-0.6B-Embedding + BGE-
Reranker), and a random retriever. For the training
stage, VeRL (v0.4.1.dev0) serves as our training
backend, with the hyperparameters shown in Ta-
ble 2. Each experimental run requires approxi-
mately 32 GPU hours (executed on 4 ×NVIDIA
A800-80G GPUs for 8 hours).
For the baseline methods that required training,
we utilize their corresponding training backends
and training datasets in their code repositories, ex-
cept for Trust-align, for which we employ LLaMA-
Factory (0.9.4.dev0) to conduct DPO training based
on the provided training datasets. All the baseline
methods are adequately trained until the reward orTable 2: Main hyperparameters for our experiments.
Parameter Training Inference
clip_low 0.2 N/A
clip_high 0.28 N/A
clip_ratio_c 10.0 N/A
learning_rate 2e−6N/A
training_steps 400 N/A
warmup_steps 50 N/A
weight_decay 0.1 N/A
grad_clip 1.0 N/A
max_response_length 3072 3072
overlong_buffer_length 1024 N/A
overlong_penalty_factor 1 N/A
train_prompt_batchsize 32 N/A
group_size 8 N/A
train_prompt_mini_batchsize 8 N/A
temperature 1.0 0.6
top_p 1.0 0.9
loss converges. Other detailed hyperparameters are
provided in the GitHub repositories.
B.2 Prompt Templates
For our training and inference, we adopt the same
zero-shot prompt as Table 3.
For the baselines, we adopt the correspond-
ing prompt template with a manually constructed
three-shot example from the corresponding training
dataset. For example, as illustrated in Table 4, we
sample three questions from the training set of Hot-
potQA and manually construct two of them as an-
swerable and the other one as unanswerable. Sim-
ilarly, to maintain distributional consistency, the
three-shot examples for the QASPER test set were
derived from its respective training split. Specif-
ically, for baselines whose original prompt tem-
plates lacked an abstention instruction, we incor-
porated one with minimal intervention to ensure a
fair comparison while preserving the core structure
of the baseline.
B.3 SFT Data Construction
Following the data construction strategy described
in Section 3.1, we curate the input-output pairs for
the SFT training set. While inputs remain consis-
tent with the retrieved context format, the targets
are synthetically generated using rule-based tem-
plates to enforce structural alignment with our rea-
soning paths. For answerable instances, the target
output is formatted as:<evidence>
GTEvidence
13

Table 3: Prompt template for our method in both training
and test stage.
Our Prompt
System Prompt:
You are an evidence-validation assistant. For each
query, you are given a question wrapped inside the
<question>...</question> tag, and a series of docu-
ments as evidence wrapped inside the <ref>...</ref>
tag. To solve these questions, you must follow this
**exact** process without deviation:
### Instructions:
1. You need to first figure out if the evidences are
relevant and useful to the question.
- If the evidences are **NOT** relevant, you should
recall your knowledge about this question and
wrap the process of recalling inside the token of
<llm>...</llm>.
- If the evidences are relevant, you should select
the evidence that is most relevant to the question
and wrap the selected texts inside the token of <evi-
dence>...</evidence>.
2. You should then answer the question based on
the selected evidence or your knowledge. The an-
swer should be wrapped inside the token of <an-
swer>...</answer>.
User Prompt:
Here is the question and the references: <ques-
tion>question</question> <ref> ref </ref>
</evidence> <answer>
GTAnswer
</answer> . For unanswerable instances, the
model is supervised to output: <llm> The ques-
tion is unanswerable </llm><answer> Unan-
swerable </answer> . We subsequently perform
SFT using the LLaMA-Factory framework to ini-
tialize the model with basic instruction-following
and formatting capabilities.
C Extended Experiments
We provide extra analysis related to our experi-
ments in this section.
C.1 Ablation Study
To verify our proposed pipeline, we conduct an
ablation study along two axes: data construction
and reward design. On the data side, we consider
three variants. (1)Naive Method. As stated in
Section 3.1, we use this naive approach to con-
struct the training data. (2)Single-retriever(SR).
To verify the effectiveness of heterogeneous re-
trievers, we only keep training data constructedTable 4: Three-shot examples for HotpotQA dataset.
3-shot Example
User Prompt: **Question:**
Upper Denton is situated on the line of the Roman
road that ran through the valleys of which two rivers?
**Context:**
...(With ground-truth evidence)...
Assistant: Tyne and Irthing.
User Prompt: **Question:**
In what year was the Enblish artist who released her
second studio album Fall to Grace in 2012, born?
**Context:**
...(With ground-truth evidence)...
Assistant: 21 July 1981.
User Prompt: **Question:**
How many Tony Awards was the musical comedy on
which Jonathan Tunick started working with Stephen
Sondheim, nominated for ?
**Context:**
...(Without ground-truth evidence)...
Assistant: I don’t know.
Table 5: Ablation study on HotpotQA, Qwen3-4B, with
Qwen3-0.6B-Embedding (Qwen) or the BAAI-BGE-
M3 (BAAI) as the retriever for the test set, where ER
denotes E vidence R ouge-L score.
HotpotQA - Dense Retriever Qwen(296:204)
Answerable UnanswerableOverall
EM F1 LJ ER Acc. Acc. B. Acc.
Naive 66.89 86.51 91.22 45.43 59.80 78.40 75.51
SR 61.82 80.66 89.53 59.67 62.25 78.40 75.89
W/O AB 65.54 84.21 92.23 62.56 67.16 82.00 79.70
W/O Weighting 59.80 80.19 89.19 60.79 56.37 75.80 72.78
EM Reward 66.55 85.89 92.57 21.82 62.75 80.40 77.66
GRACE 65.20 84.95 91.55 70.36 66.18 81.20 78.87
HotpotQA - Dense Retriever BAAI(330:170)
W/O AB 63.64 84.10 90.00 62.72 62.31 80.60 76.16
GRACE 66.97 86.18 93.33 62.82 64.71 83.60 79.02
by the embedding-based retriever Qwen3-0.6B-
Embedding and train for the same number of steps,
denoted as single-retriever. (3)Without class
balancing(W/O CB). To verify the importance
of class balancing, we disable our retrieved pas-
sages manipulation component. On the reward
side, we ablate two design choices. (1)With-
out evidence-answer weighting(W/O weighting).
We remove the asymmetric weighting between evi-
dence grounding and answer accuracy, i.e., αandβ,
letting them contribute equally to the objective. (2)
Exact-match reward(EM Reward). We replace
the Rouge-L-based reward with an exact-match-
based reward for both evidence and answer super-
vision.
As illustrated in Table 5, GRACE strikes an opti-
mal balance between performance and generaliza-
14

Table 6: Comparing results on QASPER Dataset among different retrieval settings, where k= 3 . Thebest results
are bold, and the runner-up results are underlined. The number of answerable and unanswerable questions in each
dataset is denoted as(answerable:unanswerable). Methods with †sign come from the released checkpoints by the
corresponding authors.
Oracle(1451 : 0)BM25(1074 : 377)Dense Retrieval & Reranker(1205 : 246)
Answerable Answerable UnanswerableOverallAnswerable UnanswerableOverall
EM F1 LJ EM F1 LJ Acc. LJ Acc. B. Acc.EM F1 LJ Acc. LJ Acc. B. Acc.
Prompt based
Qwen3-4B 0.07 49.6583.600.0 49.2671.695.84 42.44 64.09 57.07 0.0 50.1078.426.50 43.09 72.43 60.76
Llama3-8B-Instruct 0.76 52.66 69.61 1.12 52.92 64.62 16.71 34.22 56.72 49.42 1.66 54.78 70.21 18.70 31.71 63.68 50.96
DeepSeek-chat 0.14 62.64 78.70 0.0 50.41 63.13 55.70 75.60 66.37 69.37 0.08 51.96 69.63 44.31 69.92 69.68 69.78
SuRe on Qwen3-4B 6.13 51.17 42.38 6.05 51.77 44.41 0.0 1.33 33.22 22.87 6.31 52.35 46.64 0.0 0.0 38.73 23.32
SuRe on Llama3-8B-Instruct 5.31 53.45 49.35 4.75 54.23 55.68 0.0 0.27 41.90 29.17 5.23 54.50 55.27 0.0 1.63 46.18 28.45
SFT based
Qwen3-4B 36.46 76.61 74.6426.9166.26 55.21 71.09 71.35 59.41 63.2830.7169.75 62.57 70.73 70.73 63.96 66.65
Llama3-8B-Instruct38.04 77.3474.50 26.72 66.11 55.96 74.80 74.80 60.85 65.38 27.88 70.09 63.24 67.89 67.89 64.02 65.57
SimpleDeepSearcher-7B 23.23 65.91 71.47 22.07 65.15 63.13 18.83 19.89 51.90 41.51 22.74 66.57 65.64 20.33 21.54 58.17 43.59
RL based
TrustAlign-3B†0.0 42.48 50.24 0.0 40.54 39.85 74.27 78.51 49.90 59.18 0.0 41.99 42.74 71.54 76.42 48.45 59.58
TrustAlign on Qwen3-4B 0.0 44.05 47.48 0.0 43.53 31.28 0.0 66.05 40.32 48.67 0.0 44.03 37.76 0.0 63.41 42.11 50.59
TrustAlign-7B†0.0 44.19 47.07 0.0 43.03 32.68 0.0 87.80 47.00 60.24 0.0 43.92 35.93 0.0 85.37 44.31 40.12
TrustAlign-8B†0.0 35.07 17.99 0.0 34.42 8.6695.76 96.02 31.36 52.34 0.0 35.16 12.5397.97 97.97 27.02 55.25
ARENA on Qwen3-4B 24.19 66.43 76.15 23.74 67.24 68.81 23.08 23.08 56.93 49.95 25.14 68.77 73.86 21.54 21.95 65.06 47.91
ARENA-7B†23.64 68.73 76.91 21.60 68.01 67.32 6.90 8.75 52.10 38.04 23.49 69.99 69.71 6.91 8.13 59.27 38.92
ARENA-8B†24.33 68.76 68.92 22.81 67.09 66.29 0.0 0.0 49.07 33.15 22.82 69.01 70.12 0.0 0.0 58.24 35.06
R1 Searcher-7B†15.02 57.49 48.04 14.34 58.27 43.20 13.26 16.98 36.39 30.09 16.85 60.49 47.88 14.23 17.48 42.73 32.68
Search R1-7B†20.19 63.91 64.78 18.16 62.58 56.61 32.36 34.48 50.86 45.55 19.75 65.34 61.74 32.52 35.77 57.34 48.76
GRACE
Qwen3-4B 29.84 73.11 76.36 24.3970.0168.25 77.19 77.19 70.57 72.7227.2272.7373.78 77.24 77.24 74.36 75.51
Llama3-8B-Instruct 29.98 73.42 76.91 23.93 68.44 59.12 83.02 83.02 65.33 71.07 27.47 71.13 67.14 78.86 78.86 69.12 73.14
Table 7: Comparing results on HotpotQA Dataset
among different retrieval settings.
MethodkAnswerable Unanswerable Overall
EM F1 LJ Acc. LJ Acc. B. Acc.
BM253 68.20 86.53 93.55 61.13 61.13 75.20 77.34
4 68.25 87.47 91.61 68.14 68.14 81.00 79.88
5 67.51 86.88 91.17 69.94 69.94 83.40 80.56
6 66.67 87.03 90.20 67.83 67.83 83.80 79.02
Dense
Retriever3 65.20 84.95 91.55 66.18 66.18 81.20 78.87
4 68.56 87.21 90.42 74.10 74.10 85.00 82.26
5 68.31 86.92 90.44 70.90 70.90 85.20 80.67
6 66.50 86.43 89.24 72.53 72.53 86.20 80.89
Retriever&
Reranker3 66.59 86.10 92.01 62.07 62.07 86.90 77.04
4 69.40 87.97 92.46 71.43 71.43 90.40 81.95
5 67.72 87.64 90.72 50.00 50.00 88.60 70.36
6 66.88 87.06 90.42 50.00 50.00 88.80 70.21
tion. From the data perspective, adopting the naive
approach for training data construction leads to a
significant performance degradation: overall accu-
racy drops by approximately 3%, while evidence
grounding plummets by nearly 25%. These results
underscore the inefficacy of the naive construction
method in capturing complex reasoning require-
ments. For the single retriever variant, we observed
that the model still acquires some knowledge. How-
ever, its overall performance is significantly lower
than that achieved with heterogeneous retrievers,
which may be attributed to the lack of data vari-
ance. Regarding retrieved passages manipulation,
we found that removing this component leads to a
slight performance gain—but only if the retrieverused during testing is the same one used for data
construction. Furthermore, evaluations on unseen
embedding models indicate that retrieved passage
manipulation primarily enhances generalization, al-
lowing the model to maintain robust performance
across different retrievers.
From the reward-design perspective, evidence-
answer weighting emphasizes the importance of
producing correct answers and well-grounded evi-
dence, thereby improving both answer quality and
evidence grounding. Moreover, our Rouge-L-based
evidence reward provides a dense supervision sig-
nal: for evidence sentences that are difficult to
match exactly, it yields graded and discriminative
rewards rather than a binary signal, which ensures
that the model receives meaningful feedback even
for partially correct extractions, thereby facilitating
stable convergence and consistent optimization.
C.2 Performance on Different Retriever
In our main experiments, we fix the Qwen3-0.6B-
Embedding as our retriever. To test the general-
ization of our method, we also evaluate it against
all baselines on three different retriever settings:
oracle, which directly inputs the ground truth evi-
dence chunks; BM25; and the Retriever Reranker
pipeline. The results on the QASPER and Hot-
potQA are illustrated in Table 6 and Table 8, respec-
tively. For the oracle retriever, even if the ground
15

Table 8: Comparing results on HotpotQA Dataset among different retrieval settings, where k= 3 . Thebest results
are bold, and the runner-up results are underlined. The number of answerable and unanswerable questions in each
dataset is denoted as(answerable:unanswerable). Methods with †sign come from the released checkpoints by the
corresponding authors.
Oracle(500 : 0)BM25(217 : 283)Dense Retrieval & Reranker(413 : 87)
Answerable Answerable UnanswerableOverallAnswerable UnanswerableOverall
EM F1 LJ EM F1 LJ Acc. LJ Acc. B. Acc.EM F1 LJ Acc. LJ Acc. B. Acc.
Prompt based
Qwen3-4B 23.80 58.73 91.60 13.36 52.52 92.17 8.13 57.24 72.40 74.71 19.13 55.47 90.31 8.05 59.77 85.00 75.04
Llama3-8B-Instruct 22.80 50.47 43.20 6.45 37.73 28.11 42.76 87.28 61.60 57.70 6.78 37.69 24.21 42.53 86.21 35.00 55.21
DeepSeek-chat 44.20 67.93 76.20 37.79 63.65 72.35 76.68 76.68 74.80 74.52 36.80 63.05 72.15 70.11 70.11 71.80 71.13
SuRe on Qwen3-4B 29.80 65.79 67.00 25.35 63.54 67.74 0.0 0.0 29.40 33.87 29.30 65.65 69.25 0.0 0.0 57.20 34.62
SuRe on Llama3-8B-Instruct 37.00 69.93 82.20 33.64 67.62 79.26 0.0 1.06 35.00 40.16 35.59 68.77 77.97 0.0 0.0 64.40 38.99
SFT based
Qwen3-4B 66.00 86.20 87.20 64.98 85.87 86.18 50.18 50.18 65.80 68.18 67.07 85.95 85.71 36.78 36.78 77.20 61.25
Llama3-8B-Instruct67.40 87.1388.00 65.44 85.69 85.71 54.06 54.06 68.20 69.8968.5286.82 87.17 40.23 40.23 79.00 63.70
SimpleDeepSearcher-7B†40.40 65.13 60.60 43.78 68.34 65.44 65.02 65.02 65.20 65.23 43.34 67.81 60.77 55.17 55.17 59.80 57.97
RL based
TrustAlign-3B†0.00 43.35 61.00 0.00 40.88 50.69 53.71 71.38 62.40 61.04 0.00 39.92 49.64 59.77 68.97 53.00 59.31
TrustAlign on Qwen3-4B 0.0 32.91 21.20 0.0 32.55 19.82 0.098.23 64.20 59.03 0.0 31.97 12.11 0.097.70 27.00 54.91
TrustAlign-7B†0.00 36.55 36.60 0.00 34.55 30.88 0.00 90.46 64.60 60.67 0.00 34.48 27.12 0.00 87.36 37.60 57.24
TrustAlign-8B†0.00 34.75 34.00 0.00 32.35 28.5790.8190.81 63.80 59.69 0.00 31.88 25.1893.1093.10 37.00 59.14
ARENA on Qwen3-4B 63.40 85.57 89.80 63.59 85.54 92.63 42.05 42.40 64.20 67.52 64.16 85.62 89.83 37.93 37.93 80.80 63.88
ARENA-7B†66.20 86.04 90.20 64.06 85.81 91.24 6.01 13.07 47.00 52.16 64.89 86.01 88.14 5.75 11.49 74.80 49.82
ARENA-8B†62.00 84.34 86.80 58.06 81.70 85.25 0.00 0.00 37.00 42.63 62.95 84.41 86.92 0.00 0.00 71.80 43.46
R1 Searcher-7B†45.00 68.32 62.40 47.00 73.40 69.59 55.83 55.83 61.80 62.71 49.88 73.57 68.77 63.22 63.22 67.80 65.99
Search R1-7B†55.20 79.12 79.20 56.22 79.42 81.11 44.17 44.17 60.20 62.64 55.45 78.86 75.30 37.93 40.23 69.20 57.77
GRACE
Qwen3-4B 66.60 86.30 92.00 68.20 86.53 93.5561.13 61.13 75.20 77.3466.59 86.1092.0162.07 62.07 86.90 77.04
Llama3-8B-Instruct 65.60 86.21 88.20 62.67 85.65 86.18 47.35 50.18 65.80 68.18 66.8387.0388.38 31.03 34.48 79.00 61.43
truth evidence is input, the model still cannot an-
swer all the questions correctly, which indicates
the necessity of research on improving the model’s
capability of leveraging evidence. For the BM25
as the retriever, the ratio of correctly answering the
answerable questions decreases compared to the or-
acle retriever and dense retriever. We attributed this
phenomenon to the lower retrieval quality of BM25,
which may provide unrelated evidence to the model
with a higher score. For the dense retriever with
a reranker, the results are better than the BM25,
demonstrating that the quality of the retriever mat-
ters for the RAG system, but with proper training,
models with a relatively weak retriever can outper-
form models with a stronger one. For example, our
method with dense retriever outperforms ARENA-
4B with the oracle retriever, demonstrating the im-
portance of this research area.
C.3 Top-k Variants
In our main experiments, we fix k=3, but practical
RAG systems often operate with dynamic or task-
specifick. As illustrated in Table 7, our method is
robust to the choice of k, with balanced accuracy
remaining stable at around 80 across most settings.
The only notable exception is the Dense Retriever
& Reranker configuration: as kincreases, the re-
triever almost always returns key evidence for each
query, leaving very few unanswerable cases. ThisTable 9: Cross-dataset generalization results.
Answerable Unanswerable Overall
Retriever EM F1 LJ Acc. LJ Acc. B. Acc.
Oracle 62.40 85.50 90.00 – – – –
Dense Retriever 64.53 85.15 90.88 59.31 59.31 78.00 75.10
BM25 59.91 84.35 90.32 60.07 60.07 73.20 75.20
Retriever & Reranker 63.92 85.20 89.59 55.17 55.17 83.60 72.38
substantially reduces the effective sample size for
abstention evaluation, leading to higher variance
and larger fluctuations in performance.
C.4 Performance on Out-of-distribution Data
As shown in Table 9, our model remains robust
under cross-dataset evaluation: when trained on a
different dataset, it generalizes well to the target
dataset. Relative to the in-domain trained model,
it shows an average drop of 2 percentage points in
overall accuracy across target settings, while still
outperforming the 3-shot untrained model.
C.5 Impact on General Capabilities
We further evaluate the post-training model on
a subset of MMLU, including abstract algebra,
anatomy, computer security, econometrics, global
facts, high school biology, high school chem-
istry, high school physics, human sexuality, pro-
fessional law, and U.S. foreign policy, to as-
sess whether RLVR introduces degradation in
general knowledge and reasoning ability. Our
16

When was the female co-founder of 
the organization that published the 
"Daily Word" devotional born?
GRACE on Qwen3-4B
<think> Okay, let's tackle this question. ...  First, I 
need to figure out which part of the evidence is 
relevant. ... In the evidence, there's a mention of 
Unity Church publishing the "Daily Word" devotional. 
Specifically, the second paragraph states... Then, the 
evidence mentions Myrtle Fillmore as a co-founder of 
Unity. It says, "Mary Caroline 'Myrtle' Page Fillmore 
(August 6, 1845 - October 6, 1931) was co-founder of 
Unity, ..." The question asks about the female co-
founder of the organization that published the "Daily 
Word". Since Myrtle Fillmore co-founded Unity, which 
publishes the "Daily Word", she is the female co-
founder in question. The question is asking for her 
birthdate, which is given as August 6, 1845. The other 
parts of the evidence mention different devotional 
books like "My Utmost for His Highest" or "The Clear 
Word", but they aren't relevant here. ...So the 
answer is August 6, 1845. </think> 
<evidence> Unity, known informally as Unity Church, is 
a New Thought Christian organization that publishes 
the "Daily Word" devotional publication. Mary Caroline 
"Myrtle" Page Fillmore (August 6, 1845 - October 6, 
1931) was co-founder of Unity, a church within the 
New Thought Christian movement, along with her 
husband Charles Fillmore. </evidence> 
<answer>August 6, 1845</answer>
When was the female co-founder of 
the organization that published the 
"Daily Word" devotional born?
GRACE on Llama3.1-8B
To find when the female co-founder of the 
organization that published the "Daily Word" 
devotional born, it is essential to first examine the 
evidence for their reliability and usefulness to the 
question of the female co-founder of the organization 
that published the "Daily Word" devotional.
......
From the evidence, the most relevant evidence to the 
question of the female co-founder of the organization 
that published the "Daily Word" devotional is:
<evidence> Mary Caroline "Myrtle" Page Fillmore 
(August 6, 1845 - October 6, 1931) was co-founder of 
Unity, a church within the New Thought Christian 
movement, along with her husband Charles Fillmore. 
Unity, known informally as Unity Church, is a New 
Thought Christian organization that publishes the 
"Daily Word" devotional publication. </evidence>
Given the evidence of the female co-founder of the 
organization that published the "Daily Word" 
devotional and their birthdate, it is evident that the 
female co-founder of the organization that published 
the "Daily Word" devotional was born on August 6, 
1845. Therefore, the answer is:
<answer> August 6, 1845 </answer>Figure 4: Case study on Grace: evidence-path selection
with Qwen3-4B. Green highlights the model’s key rea-
soning steps, while red marks incorrect attempts.
results demonstrate that the proposed method
preserves, and in some cases enhances, the
model’s general reasoning capabilities. Specifi-
cally, while the untrained Qwen3-4B base model
achieves a score of 66.0% on MMLU, our trained
versions—Grace-Qwen3-4B-HotpotQA and Grace-
Qwen3-4B-QASPER—attain 66.1% and 67.6%
respectively, indicating that the specialized train-
ing does not compromise foundational knowledge.
This result aligns with prior findings: RLVR up-
dates only a small but functionally critical subset
of parameters, preserving the pretrained model’s
general competence (Mukherjee et al., 2025), and
the on-policy training mitigates catastrophic forget-
ting (Chen et al., 2025).
C.6 Case Study
To better illustrate our models’ behavior, we
present several examples from the trained models.
As shown in Figures 4 and 5, the trained models
learn to verify the provided evidence and perform
intermediate reasoning before producing the final
evidence selection and answer. Notably, even the
instruct model, under our reward design, sponta-
neously acquires an evidence-checking procedure
When was the female co-founder of 
the organization that published the 
"Daily Word" devotional born?
GRACE on Qwen3-4B
<think> Okay, let's tackle this question. ...  First, I 
need to figure out which part of the evidence is 
relevant. ... In the evidence, there's a mention of 
Unity Church publishing the "Daily Word" devotional. 
Specifically, the second paragraph states... Then, the 
evidence mentions Myrtle Fillmore as a co-founder of 
Unity. It says, "Mary Caroline 'Myrtle' Page Fillmore 
(August 6, 1845 - October 6, 1931) was co-founder of 
Unity, ..." The question asks about the female co-
founder of the organization that published the "Daily 
Word". Since Myrtle Fillmore co-founded Unity, which 
publishes the "Daily Word", she is the female co-
founder in question. The question is asking for her 
birthdate, which is given as August 6, 1845. The other 
parts of the evidence mention different devotional 
books like "My Utmost for His Highest" or "The Clear 
Word", but they aren't relevant here. ...So the 
answer is August 6, 1845. </think> 
<evidence> Unity, known informally as Unity Church, is 
a New Thought Christian organization that publishes 
the "Daily Word" devotional publication. Mary Caroline 
"Myrtle" Page Fillmore (August 6, 1845 - October 6, 
1931) was co-founder of Unity, a church within the 
New Thought Christian movement, along with her 
husband Charles Fillmore. </evidence> 
<answer>August 6, 1845</answer>
When was the female co-founder of 
the organization that published the 
"Daily Word" devotional born?
GRACE on Llama3.1-8B-Instruct
To find when the female co-founder of the 
organization that published the "Daily Word" 
devotional born, it is essential to first examine the 
evidence for their reliability and usefulness to the 
question of the female co-founder of the organization 
that published the "Daily Word" devotional.
......
From the evidence, the most relevant evidence to the 
question of the female co-founder of the organization 
that published the "Daily Word" devotional is:
<evidence> Mary Caroline "Myrtle" Page Fillmore 
(August 6, 1845 - October 6, 1931) was co-founder of 
Unity, a church within the New Thought Christian 
movement, along with her husband Charles Fillmore. 
Unity, known informally as Unity Church, is a New 
Thought Christian organization that publishes the 
"Daily Word" devotional publication. </evidence>
Given the evidence of the female co-founder of the 
organization that published the "Daily Word" 
devotional and their birthdate, it is evident that the 
female co-founder of the organization that published 
the "Daily Word" devotional was born on August 6, 
1845. Therefore, the answer is:
<answer> August 6, 1845 </answer>Figure 5: Case study on Grace: evidence-path selec-
tion with Llama3.1-8B-Instruct. Green highlights the
model’s key reasoning steps.
that resembles an explicit “think”-style process. Re-
garding the LLM path, we observe that the model
attempts to retrieve relevant information from its
internal parametric knowledge. As illustrated in
Figure 6, the trained model identifies that the pro-
vided evidence is insufficient, successfully recalls
the necessary knowledge, and answers the ques-
tion correctly. In contrast, for the relatively weaker
Llama model (Figure 7), the trained model lacks
the relevant knowledge and consequently responds
with “unknown”.
D The Use of LLMs
This paper employed LLMs solely for grammat-
ical correction and stylistic refinement, with the
purpose of more effectively communicating our
results and conclusions.
17

Who is the costar of a movie with the 
star of Scott McCall and was directed 
by the director of Supernatural?
<think>Okay, let's tackle this question...
......(analysing the evidence and question)
Now, the evidence mentions that Tyler Posey is 
known for Teen Wolf, but the evidence doesn't 
explicitly mention any movie where he was in with a 
costar, or any director of Supernatural. Wait, the 
question mentions "the director of Supernatural." 
The evidence mentions that the show Teen Wolf was 
developed by Jeff Davis. But the question refers to 
"Supernatural" as the show, which is a different 
show. ... The user is asking about the director of 
Supernatural, but the evidence doesn't mention that. 
So the evidence doesn't provide information about 
the director of Supernatural. Therefore, the 
evidence is not relevant. So I need to recall my 
knowledge here....</think>
<llm>The question asks about... The evidence 
mentions Tyler Posey as the actor who played Scott 
McCall in Teen Wolf but does not explicitly state the 
director of Supernatural or any movie co-starring 
Tyler Posey. ...Thus, the evidence is not directly 
relevant, requiring knowledge recall.\n</llm>
<answer>Dylan O'Brien</answer>GRACE on Qwen3-4B
Who is the costar of a movie with the 
star of Scott McCall and was directed 
by the director of Supernatural?
To find the costar of a movie with the star 
of Scott McCall and was directed by the 
director of Supernatural, it is essential to 
first examine the evidence for their 
reliability and usefulness to the question of 
the costar of a movie with the star of Scott 
McCall and was directed by the director of 
Supernatural. 
Given the absence of any mention of the 
director of Supernatural in the provided 
evidence and the absence of a mention of a 
specific movie co-starring the star of Scott 
McCall with the costar in the provided 
evidence, the evidence is not only irrelevant 
but also unhelpful to the question of the 
costar of a movie with the star of Scott 
McCall and was directed by the director of 
Supernatural. 
<llm> Unknown </llm>
<answer> Unknown </answer>GRACE on Llama3.1-8B-InstructFigure 6: Case study on Grace: llm-path selection with
Qwen3-4B. Green highlights the model’s key reasoning
steps, while red marks incorrect attempts.
Who is the costar of a movie with the 
star of Scott McCall and was directed 
by the director of Supernatural?
<think>Okay, let's tackle this question...
......(analysing the evidence and question)
Now, the evidence mentions that Tyler Posey is 
known for Teen Wolf, but the evidence doesn't 
explicitly mention any movie where he was in with a 
costar, or any director of Supernatural. Wait, the 
question mentions "the director of Supernatural." 
The evidence mentions that the show Teen Wolf was 
developed by Jeff Davis. But the question refers to 
"Supernatural" as the show, which is a different 
show. ... The user is asking about the director of 
Supernatural, but the evidence doesn't mention that. 
So the evidence doesn't provide information about 
the director of Supernatural. Therefore, the 
evidence is not relevant. So I need to recall my 
knowledge here....</think>
<llm>The question asks about... The evidence 
mentions Tyler Posey as the actor who played Scott 
McCall in Teen Wolf but does not explicitly state the 
director of Supernatural or any movie co-starring 
Tyler Posey. ...Thus, the evidence is not directly 
relevant, requiring knowledge recall.\n</llm>
<answer>Dylan O'Brien</answer>GRACE on Qwen3-4B
Who is the costar of a movie with the 
star of Scott McCall and was directed 
by the director of Supernatural?
To find the costar of a movie with the star 
of Scott McCall and was directed by the 
director of Supernatural, it is essential to 
first examine the evidence for their 
reliability and usefulness to the question of 
the costar of a movie with the star of Scott 
McCall and was directed by the director of 
Supernatural. 
Given the absence of any mention of the 
director of Supernatural in the provided 
evidence and the absence of a mention of a 
specific movie co-starring the star of Scott 
McCall with the costar in the provided 
evidence, the evidence is not only irrelevant 
but also unhelpful to the question of the 
costar of a movie with the star of Scott 
McCall and was directed by the director of 
Supernatural. 
<llm> Unknown </llm>
<answer> Unknown </answer>GRACE on Llama3.1-8B-Instruct
Figure 7: Case study on Grace: llm-path selection with
Llama3.1-8B-Instruct. Green highlights the model’s key
reasoning steps.
18