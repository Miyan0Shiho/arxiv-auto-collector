# KARE-RAG: Knowledge-Aware Refinement and Enhancement for RAG

**Authors**: Yongjian Li, HaoCheng Chu, Yukun Yan, Zhenghao Liu, Shi Yu, Zheni Zeng, Ruobing Wang, Sen Song, Zhiyuan Liu, Maosong Sun

**Published**: 2025-06-03 06:31:17

**PDF URL**: [http://arxiv.org/pdf/2506.02503v1](http://arxiv.org/pdf/2506.02503v1)

## Abstract
Retrieval-Augmented Generation (RAG) enables large language models (LLMs) to
access broader knowledge sources, yet factual inconsistencies persist due to
noise in retrieved documents-even with advanced retrieval methods. We
demonstrate that enhancing generative models' capacity to process noisy content
is equally critical for robust performance. In this paper, we present KARE-RAG
(Knowledge-Aware Refinement and Enhancement for RAG), which improves knowledge
utilization through three key innovations: (1) structured knowledge
representations that facilitate error detection during training, (2) Dense
Direct Preference Optimization (DDPO)-a refined training objective that
prioritizes correction of critical errors, and (3) a contrastive data
generation pipeline that maintains semantic consistency while rectifying
factual inaccuracies. Experiments show our method significantly enhances
standard RAG pipelines across model scales, improving both in-domain and
out-of-domain task performance without compromising general capabilities.
Notably, these gains are achieved with modest training data, suggesting
data-efficient optimization is possible through targeted learning strategies.
Our findings establish a new direction for RAG improvement: by improving how
models learn to process retrieved content, we can enhance performance across
diverse inference paradigms. All data and code will be publicly available on
Github.

## Full Text


<!-- PDF content starts -->

arXiv:2506.02503v1  [cs.CL]  3 Jun 2025KARE-RAG: Knowledge-Aware Refinement and Enhancement for RAG
Yongjian Li1*, HaoCheng Chu2*, Yukun Yan3†, Zhenghao Liu4, Shi Yu3, Zheni Zeng3,
Ruobing Wang5,Sen Song1†,Zhiyuan Liu3,Maosong Sun3
1School of Biomedical Engineering, Tsinghua, China,
3School of Informatics, Xiamen University, China,
2Department of Computer Science and Technology, Institute for AI, Tsinghua University, China,
4Dept. of Computer Science and Technology, Northeastern University, Shenyang, China,
5Institute of Information Engineering, Chinese Academy of Sciences, Beijing, China
yj-li23@mails.tsinghua.edu.cn
Abstract
Retrieval-Augmented Generation (RAG) en-
ables large language models (LLMs) to access
broader knowledge sources, yet factual incon-
sistencies persist due to noise in retrieved docu-
ments—even with advanced retrieval methods.
We demonstrate that enhancing generative mod-
els’ capacity to process noisy content is equally
critical for robust performance. In this paper,
we present KARE-RAG (Knowledge-Aware
Refinement and Enhancement for RAG), which
improves knowledge utilization through three
key innovations: (1) structured knowledge rep-
resentations that facilitate error detection dur-
ing training, (2) Dense Direct Preference Opti-
mization (DDPO)—a refined training objective
that prioritizes correction of critical errors, and
(3) a contrastive data generation pipeline that
maintains semantic consistency while rectify-
ing factual inaccuracies. Experiments show
our method significantly enhances standard
RAG pipelines across model scales, improving
both in-domain and out-of-domain task perfor-
mance without compromising general capabil-
ities. Notably, these gains are achieved with
modest training data, suggesting data-efficient
optimization is possible through targeted learn-
ing strategies. Our findings establish a new
direction for RAG improvement: by improving
how models learn to process retrieved content,
we can enhance performance across diverse in-
ference paradigms. All data and code will be
publicly available on Github.
1 Introduction
Retrieval-Augmented Generation (RAG) (Lewis
et al., 2020; Shi et al., 2023) has emerged as a
useful approach for knowledge-intensive tasks, en-
abling large language models (LLMs) to retrieve
relevant information and generate informed re-
sponses. However, retrieval often brings noisy,
*Indicates equal contribution.
†Corresponding authors.irrelevant, or conflicting information(Gao et al.,
2023; Yoran et al., 2023; Wu et al., 2024; Xu et al.,
2024a,b; Longpre et al., 2022; Liu et al., 2024a),
causing factual inconsistencies. While existing
work primarily addresses this through improved
retrieval methods(Jiang et al., 2023; Gao et al.,
2023; Trivedi et al., 2022a; Mao et al., 2024; Ma
et al., 2023; Jeong et al., 2024) or post-processing
pipelines(Yu et al., 2023; Xu et al., 2023; Fang
et al., 2024), recent studies reveal an equally criti-
cal challenge: optimizing generation models to re-
liably process retrieval contexts(Singh et al., 2021;
Lin et al., 2023; Asai et al., 2023; Li et al., 2024a)
To optimize the generation model, a natural
approach involves constructing instruction-tuning
datasets from final answers and applying super-
vised fine-tuning (SFT) to the generation module
(Lin et al., 2023; Asai et al., 2023). However, the
inherent complexity of RAG tasks makes models
prone to overfitting training signals while suffer-
ing from catastrophic forgetting. The demonstrated
effectiveness of Reinforcement Learning from Hu-
man Feedback (RLHF) (Ouyang et al., 2022) and
Direct Preference Optimization (DPO) (Rafailov
et al., 2024) in aligning large language models
has consequently spurred interest in adapting these
preference-based methods for RAG’s generation
module. Recent studies further validate DPO’s ad-
vantages in RAG scenarios (Li et al., 2024a), show-
ing its ability to provide stable fine-tuning through
pairwise preference learning.
Nevertheless, significant challenges hinder
DPO’s broader adoption in RAG systems. The
multi-stage nature of RAG architectures (Gao et al.,
2023), comprising at least document parsing and
answer generation in each retrieval cycle, presents
fundamental difficulties. While DPO training uti-
lizes pairwise samples, data constructed directly
from final answers fails to adequately capture the
nuances of this intricate processing pipeline, of-
ten requiring prohibitively large datasets to achieve
1

satisfactory results (Li et al., 2024a). Addition-
ally, DPO methods suffer from low sampling ef-
ficiency during training. Generating high-quality
positive examples through conventional sampling
and prompt adjustment proves particularly chal-
lenging, especially for smaller models, further com-
pounding training difficulties.
To address these issues, we develop KARE-
RAG(nowledge-Aware Refinement and Enhance-
ment for RAG)—a training framework that teaches
models to better utilize. Our main contributions are
as follows:
Structured Knowledge Supervision Mecha-
nism. We introduce graph-based intermediate rep-
resentations that transform noisy documents into
structured knowledge graphs during training. This
creates explicit learning objectives for distinguish-
ing relevant content, addressing the limitation of
end-to-end approaches that directly optimize final
answers (Lin et al., 2023). The structured format
provides verifiable supervision signals, enabling
models to learn noise-resistant information pro-
cessing patterns.
Dense Direct Preference Optimization
(DDPO). We deploy an enhanced version of DPO
incorporating token-level importance weight-
ing, which dynamically prioritizes correction
of critical knowledge discrepancies. Through
contrastive analysis of sample pairs where minimal
representation differences determine answer
correctness, DDPO teaches robust information
filtering strategies that generalize to standard RAG
pipelines (Rafailov et al., 2024).
Automated Contrastive Data Generation. We
develop an LLM-powered pipeline that gener-
ates high-quility training pairs while preserving
semantic-structural consistency through three-stage
self-verification: 1) Error localization, 2) Targeted
correction, and 3) Validation filtering. This elimi-
nates reliance on human annotation and overcomes
the positive sample scarcity problem inherent to
complex RAG workflows, ensuring models learn
substantive knowledge discrimination rather than
superficial pattern matching.
Through comprehensive empirical evaluation,
our approach demonstrates statistically significant
improvements across both in-domain and out-of-
domain benchmarks, evidencing enhanced robust-
ness in knowledge utilization and superior gener-
alization capabilities. Furthermore, we system-
atically evaluate multiple knowledge representa-
tion formats—including knowledge graph, key-point structure, and summary—revealing that struc-
tured representations yield substantially better per-
formance compared to unstructured summarization
baselines. Crucially, the method maintains effi-
cacy across varying model scales, achieving con-
sistent performance gains without compromising
baseline capabilities on general language under-
standing tasks.
2 Related Work
LLMs possess a robust in-context learning ca-
pability(Dong et al., 2022; Ram et al., 2023),
enabling them to handle more complex tasks.
Retrieval-Augmented Generation (RAG) enhances
LLMs by incorporating external knowledge re-
trieval(Karpukhin et al., 2020; Xiong et al., 2020),
improving performance in tasks like question an-
swering(Trivedi et al., 2022a) and fact verification
(Lewis et al., 2020). It can reduce the model’s
propensity for hallucination and increase the relia-
bility of the model’s outputs(Kandpal et al., 2023;
Xu et al., 2023; Luo et al., 2023). However, re-
trieval often brings noisy or conflicting informa-
tion, causing factual inaccuracies(Longpre et al.,
2021; Xu et al., 2024a; Liu et al., 2024a).
To mitigate the impact of noise on generation
effectiveness, some studies have introduced addi-
tional steps to enhance retrieval precision. For
instance, several methods employ query rewrite
techniques to increase the similarity between the
query and the relevant texts(Wang et al., 2023; Mao
et al., 2023), some others prompt LLMs to summa-
rize query related informations into note(Yu et al.,
2023).
Optimizing the RAG process to enhance gener-
ation outcomes is also a highly popular direction.
Many current efforts are exploring various methods
to construct training data for training retrieval mod-
els to improve retrieval accuracy(Lin et al., 2023;
Shi et al., 2023). Self-RAG trains models to learn to
retrieve relevant documents on demand(Asai et al.,
2023). DDR constructs a multi-level training frame-
work based on the final generation outcomes, si-
multaneously training both retrieval and generation
models, enabling LLMs to learn to handle conflicts
between intrinsic knowledge and retrieved docu-
ments(Li et al., 2024a).
Reinforcement learning is a commonly used al-
gorithm in the optimization of large models. Due
to the complexity of the PPO algorithm(Schulman
et al., 2017), current work predominantly relies
2

Figure 1: Illustration of our KARE-RAG Method. The left side of the image illustrates the differences between our
method and Vanilla RAG, while the right side outlines the general workflow of our data construction process.
on the Direct Preference Optimization(DPO) al-
gorithm(Rafailov et al., 2024), which aligns the
outputs of LLMs with human preferences to opti-
mize large models. There are also numerous efforts
to further refine the DPO algorithm to enhance
training effectiveness. RLHF-V(Yu et al., 2024)
increases the weight of tokens in the differing parts
of positive and negative examples during training,
improving the training outcomes for visual models.
Step-DPO(Lai et al., 2024) focuses on step-by-step
optimization for long-chain reasoning processes.
Additionally, some studies have explored the re-
lationship between DPO and SFT Loss(Liu et al.,
2024b), demonstrating that SFT Loss can serve as
a regularization term to stabilize the training effects
of DPO.
Knowledge Graphs (KG)(Hogan et al., 2021) are
widely used in NLP tasks such as question answer-
ing, reasoning(Chen et al., 2020), and fact verifica-
tion (Tong et al., 2019). A significant advantage of
knowledge graphs is their structured nature, which
can effectively reduce noise in natural language
texts, benefiting the RAG scenario. Consequently,
numerous RAG works based on knowledge graphs
have emerged(Peng et al., 2024; Edge et al., 2024;
Hu et al., 2024; Mavromatis and Karypis, 2024).
Current research on knowledge graphs primarily
focuses on utilizing them to encode text libraries
to enhance retrieval effectiveness(Li et al., 2023;
Huang et al., 2023; Li et al., 2024b), or integrat-
ing GNN with LLMs in the generation phase to
improve output quality(Gao et al., 2024; He et al.,
2024; Jiang et al., 2022). These methods typically
require preliminary processes to construct static
knowledge graphs, making the overall workflowrelatively complex.
3 Method
In this section, we introduce the Knowledge-Aware
Refinement and Enhancement for RAG(KARE-
RAG) method, shown in Figure 1. We First in-
troduce the Knowledge-Aware RAG(KA-RAG)
pipeline, which decompose the generation process
of RAG into three step to explicitly demonstrate
how the model organizes and utilizes retrieved doc-
uments (left part of Figure 1). Then, we introduce
the DDPO(Dense Direct Preference Optimization)
training method that optimizes the model to fo-
cus more on the nuanced differences between data
pairs. Finally we will introduce our data generation
pipeline(right part of Figure 1).
3.1 KA-RAG Pipeline
In a standard RAG (Retrieval-Augmented Gener-
ation) process, the LLM is given a query qand
a set of retrieved documents D={d1, . . . , d n}
relevant to qto generate answer to the given
query. The retrieved documents invariably contain
noise—irrelevant passages, conflicting information,
or partial matches—that propagates through the
generation process. This noise contamination sys-
tematically degrades output quality, introducing
factual inaccuracies and information gaps in the
final responses.
To explicitly demonstrate how the model orga-
nizes question-relevant information from the doc-
uments and generates answers, we decompose the
generation process into three distinct stages in our
experiments. The first stage structured knowledge
representation, the second stage focuses on CoT
3

(Chain-of-Thought) reasoning, and the final stage
generates the answer based on the reasoning chain.
The process can be formated as follow:
Knowledge Organization
yKG= LLM(Instruct KG, q⊕D) (1)
CoT
yCoT= LLM(Instruct CoT, q⊕yKG)(2)
Generation
yGen= LLM(Instruct Gen, q⊕yCoT)(3)
where ⊕denote the concatenation operation,
Instruct KG,Instruct CoT,Instruct Gen are
prompts specifically designed to address the three
corresponding stages.
In the knowledge organization phase, we formal-
ize the extracted information as structured graph
representations. This graph-based formalism, in-
spired by knowledge graph architectures, provides
three key representational advantages: (1) explicit
node-edge relationships that enforce logical con-
sistency, (2) discrete knowledge units that enable
precise error localization, and (3) topological con-
straints that prevent information entanglement. The
resulting structured representation serves as an in-
termediate knowledge substrate that is both human-
interpretable and machine-tractable, establishing a
robust foundation for subsequent processing stages.
3.2 Dense Direct Preference Optimization
Direct Preference Optimization (DPO) aligns lan-
guage models with human preferences through
pairwise comparison of preferred and dispreferred
outputs. The method optimizes the relative like-
lihood of positive versus negative examples us-
ing a Bradley-Terry preference model formulation
(Rafailov et al., 2024). However, two fundamental
limitations emerge when applying standard DPO
to knowledge organization tasks: First, its uni-
form token-weighting scheme proves inefficient
for lengthy, structured outputs where discrimina-
tive signals concentrate in sparse critical segments
(e.g., entity relationships in knowledge graphs).
Second, the reward maximization objective forces
artificial divergence between semantically similar
pairs, causing both sequence probabilities to de-
press simultaneously, which destabilizes training
dynamics and reduces sample efficiency.
To address these issues, we adopted a variant
of the DPO algorithm proposed (Yu et al., 2024),known as Dense Direct Preference Optimization
(DDPO). To make things clearer, we briefly review
the DPO algorithm. The reward function for a spe-
cific output yof input xis represented as follows:
r(x, y) =βlogπ∗(y|x)
πref(y|x)+βlogZ(x)(4)
Where βis a constant hyperparameter, and Z(x)is
a partition function. πref(y|x)is the base model
we want to optimize, and kept fixed during training.
π∗(y|x)is the model we actually updated. Then
we can get the DPO loss:
LDPO=−E(x,y+,y−)
logσ
r(x, y+)−r(x, y−)
=−E(x,y+,y−)
logσ
β
logπ∗(y+|x)
πref(y+|x)
−logπ∗(y−|x)
πref(y−|x)
(5)
DPO algorithm treats different token with uniform
weight, and the score can be calculated as follow:
logπ(y|x) =X
yt∈ylogp(yt|x, y<t) (6)
ytis the t-th token of the response y. To ensure
that the model pays more attention to the modi-
fied tokens yccompared to the unmodified tokens
yuduring optimization, we introduce a weighting
mechanism that assigns higher weights to the yc
tokens when computing the score. The modified
score calculation formula is as follows:
logπ(y|x) =X
yt∈yulogp(yt|x, y<t)
+γX
yt∈yclogp(yt|x, y<t)(7)
γis a hyperparameter utilized to modulate the
weight of tokens within ycwhen computing the
score, while the weight of tokens in yuremains
constant at 1.
While DDPO demonstrates effectiveness in ad-
dressing key limitations of standard DPO, we ob-
serve that the simultaneous reward reduction phe-
nomenon persists in certain cases. To mitigate this
effect, we integrate supervised fine-tuning (SFT)
loss as a regularization term, following recent the-
oretical insights (Liu et al., 2024b). The complete
loss function combines these components as fol-
lows:
4

L=LDPO−αTX
t=1logp(yt|x, y<t)(8)
αis a hyperparameter that governs the weighting
of the SFT loss within the overall loss function.
3.3 Data Generation
The inherent capabilities of large models are gener-
ally robust. Therefore, the negative examples ( y−
KG)
are mostly structurally coherent but may contain
localized inaccuracies. A fundamental challenge
arises when sampling both positive examples using
the same base model ( LLM Gen). Although theoret-
ically capable of producing corrected outputs, the
model’s stochastic generation process introduces
unintended variations in two critical dimensions:
(1) knowledge sequencing patterns and (2) surface-
level phrasing. These variations become partic-
ularly pronounced in lengthy outputs, artificially
inflating the perceived differences between positive
(y+
KG) and negative pairs beyond their substantive
errors.
Our framework addresses this through targeted
refinement. By employing an advanced model
(LLM Exp) to systematically edit y−
KGwhile pre-
serving its structural backbone, we ensure con-
trastive pairs differ only in critical error regions.
This design guarantees minimal semantic diver-
gence—essential for the model to learn error cor-
rection patterns rather than superficial variations.
As detailed in Algorithm 1, the workflow prioritizes
structural fidelity through three core mechanisms:
1) Error localization, 2) Context-aware patching,
and 3) Consistency validation, making it particu-
larly effective for complex knowledge organization
tasks.
4 Experimental Settings
In this section we first introduce our experimental
settings, including datasets, evaluation metrics, and
implementation details.
Dataset . In our experiments, we employed
the challenging multi-hop question-answering
dataset, Musique(Trivedi et al., 2022b), to con-
struct our training data. Following the approach of
FlashRAG(Jin et al., 2024), we utilized Wikipedia
as the retrieval document source. For both training
and testing across various datasets, the bge-large-
en-v1.5(Xiao et al., 2023) model was used as the
retrieval engine.Algorithm 1 KARE-RAG Data Construction
1:Generate initial output y−
KGusing LLM Gen
with KA-RAG Pipeline
2:ify−
KGproduces incorrect answer yerr̸=ygnd
then
3: Check document adequacy: LLM Exp(q⊕
D⊕ygnd)
4: ifDocument Dis sufficient then
5: Revise knowledge organization:
y+
KG←LLM Exp(Instruct revise,
q⊕D⊕ygnd⊕y−
KG)
6: Generate final answer yGenusing y+
KG
withLLM Gen
7: while yGen̸=ygnd&iter< max_iter do
8: Further revise ˜y+
KGusing error analysis
9: Update yGen
10: end while
11: ifyGen=ygndthen
12: Add(y+
KG, y−
KG)as contrastive pair
13: else
14: Discard the sample
15: end if
16: end if
17:end if
By leveraging the training data construction pro-
cess described in Section 3.3, we generated 2,401
training data pairs from the 19,938 entries in the
training set. For testing purposes, we utilized the
Musique Development set. Additionally, to evalu-
ate the model’s out-of-distribution (OOD) perfor-
mance, we selected several single-hop and multi-
hop datasets for testing. The single-hop datasets
include NQ(Kwiatkowski et al., 2019), WebQues-
tions(Berant et al., 2013), and PopQA(Mallen
et al., 2022), while the multi-hop datasets com-
prise HotpotQA(Yang et al., 2018) and 2WikiMulti-
hopQA(Ho et al., 2020). Besides QA tasks we also
conducted evaluations on multiple choice dataset
TruthfulQA(Lin et al., 2021) and slot-filling dataset
Zero-shot RE(Levy et al., 2017), results can be seen
in Appendix A.3
Evaluation . Following FlashRAG(Jin et al.,
2024), we use EM(exact match) and F1 as eval-
uation metrics for all test dataset. All tasks were
run three times, and the mean values were taken.
Baseline . In our experiments, we first con-
5

Train
MethodIn Domain Out Of Domain
Musique NQ HotpotQA 2WikimultihopQA PopQA WebQ Average Gain
EM(%) F1(%) EM(%) F1(%) EM(%) F1(%) EM(%) F1(%) EM(%) F1(%) EM(%) F1(%) EM(%) F1(%)
Llama-3.2-3B-Instruct
4.47 9.89 33.2 45.59 27.89 37.75 11.9 22.24 34.32 41.6 17.96 35.14
Vanilla(DPO) 6 16.18 25.2 39.45 14.06 28.04 8.79 20.22 32.86 40.19 19.59 35.55 -4.95 -3.77
KARE(SFT) 2.73 7.29 20.06 30.46 21.73 31.35 17.07 23.77 29.52 35.8 14.76 30.05 -4.43 -6.18
KARE(DPO) 5.05 13.08 32.82 46.35 26.94 38.78 14.65 24.93 38.32 45.47 19.29 36.54 1.35 1.95
KARE(DDPO) 5.59 13.28 33.21 46.56 27.62 39.59 15.5 25.53 38.29 45.52 19.44 36.97 1.76 2.37
Llama-3.1-8B-Instruct
6 12.37 34.64 48.3 29.52 40.66 14.94 24.89 35.43 44.1 16.83 35.33
Vanilla(DPO) 7.82 19.18 25.94 41.81 17.58 33.59 9.14 18.56 36.36 43.78 22.78 38.86 -3.91 -3.34
KARE(SFT) 4.88 10.46 33.62 44.1 23.69 32.77 24.37 29.39 35.88 40.73 24.16 36.13 2.07 -2.03
KARE(DPO) 7.73 15.75 37.4 50.57 32.26 43.83 20.98 29.92 40.27 47.39 20.42 38.09 3.99 3.3
KARE(DDPO) 8.02 15.75 37.86 50.84 32.36 44.29 21.35 30.11 40.88 47.77 19.83 37.89 4.18 3.52
Qwen2.5-14B-Instruct
6.66 14.58 32.67 47.01 30.64 42.31 19.73 29.5 36.38 43.98 18.36 35.11
Vanilla(DPO) 9.68 19.84 12.63 24.37 19.22 30.37 24.55 31.85 16.3 21.35 13.68 26.65 -10.28 -12.66
KARE(SFT) 6.08 14.12 32.23 45.7 28.32 40.56 26.67 34.32 36.57 43.86 16.39 33.9 0.48 0.09
KARE(DPO) 7.57 16.58 34.99 49 32.84 45.47 23.43 32.4 38.12 45.65 19.64 36.64 2.25 2.25
KARE(DDPO) 8.32 17.48 36.19 49.96 33.96 46.45 25.14 33.72 38.58 46.09 20.03 36.95 3.22 3.05
Table 1: Overall Performance of KARE-RAG comparing to different baseline methods. The best result in each
block is highlighted. The last two columns shows the average improvement of EM and F1 of current method
compares to Vanilla RAG baseline.
structed a Vanilla RAG Pipeline, where the model
directly generates answers from retrieved docu-
ments via in-context learning. After training with
the KARE-RAG process, we compared our method
to untrained models, those trained with SFT, and
those trained with DPO. For SFT, we specifically
leveraged the positive examples from the training
dataset. For DPO, we also integrate SFT loss to
stabilize the training process. For a more com-
prehensive comparison, we included a baseline
Vanilla(DPO) trained with data from the Vanilla
RAG Pipeline. Details on construction and training
can be found in Appendix A.1.
Implementation Details . In our experiment,
we employ Llama-3.1-8B-Instruct(Touvron et al.,
2023) as the backbone models to construct most
of the experiments. Training codes are modified
from TRL(von Werra et al., 2020). During the re-
trieval phase, we retrieved five relevant documents
for each query to construct the dataset. As for
data construction, we utilized gpt-4o-mini(Achiam
et al., 2023) as the expert model to refine negative
examples into positive ones. In the training process,
we set the learning rate to 5e-5 and trained for one
epoch. For the DPO training, βwas configured
at 0.1, αwas set to 0.1. Additionally, within the
DDPO framework, the gamma parameter was es-
tablished at 1.1. Besides we also conducted partial
testing and comparative analysis of our method on
the Qwen2.5-14B-Instruct(Yang et al., 2024)(Team,
2024) and Llama-3.2-3B-Instruct(Touvron et al.,
2023) models to validate the efficacy of our ap-
proach on different model size. For all optimizationwe use LoRA(Hu et al., 2021) for efficient training.
5 Evaluation Results
In this section, we first compare the overall perfor-
mance of our method with several baseline models.
Subsequently, we conduct ablation experiments
and analyses to further validate the effectiveness of
our approach.
5.1 Main Results
The overall performance evaluation, as presented
in Table 1, demonstrates that models trained with
our KARE-RAG framework achieve consistent and
reliable improvements across both in-domain and
out-of-distribution tasks. Specifically, the DDPO-
trained Llama-3.1-8B model exhibits average gains
of +4.18% EM and +3.52% F1 over the Vanilla
RAG baseline on out-of-domain tasks. Notably,
our framework maintains strong scalability across
different model architectures: the Llama-3.2-3B-
Instruct model achieves +1.76% EM/+2.37% F1 av-
erage improvement, while Qwen2.5-14B-Instruct
model shows gains of +3.22% EM/+3.05% F1.
These improvements are particularly significant as
they are attained without any modifications to the
standard Vanilla RAG inference pipeline, under-
scoring the robustness and generalizability of our
training approach.
Enhanced Knowledge Utilization through
KARE-RAG Training . Our analysis reveals crit-
ical limitations in conventional training methods.
While standard DPO training yields improved in-
domain performance, it demonstrates unstable re-
6

FormatIn Domain Out Of Domain
Musique NQ HotpotQA 2WikimultihopQA PopQA WebQ
EM(%) F1(%) EM(%) F1(%) EM(%) F1(%) EM(%) F1(%) EM(%) F1(%) EM(%) F1(%)
6.0 12.37 34.64 48.3 29.52 40.66 14.94 24.89 35.43 44.1 16.83 35.33
Graph 8.02 15.75 37.86 50.84 32.36 44.29 21.35 30.11 40.88 47.77 19.83 37.89
Keypoints 7.36 16.31 34.73 49.06 30.07 42.93 18.04 29.06 41.24 48.38 19.19 37.75
Summary 5.67 14.2 32.77 47.11 24.43 37.44 11.43 24.4 39.22 46.72 17.77 35.55
Table 2: Ablation Study. The test of different knowledge organization format under Vanilla RAG Pipeline.
Figure 2: The Improvement of EM metrics under Vanilla
RAG Pipeline with different knowledge organization
format.
sults on out-of-domain tasks (with average metrics
showing degradation), suggesting susceptibility to
overfitting and inadequate learning of document fil-
tering strategies. Similarly, supervised fine-tuning
(SFT) proves unreliable, particularly for smaller
models like Llama-3.2-3B-Instruct, which suffers
a substantial performance drop (-4.43% EM). In
contrast, KARE-RAG trained models maintain con-
sistent improvements across all evaluation scenar-
ios, indicating genuine enhancement in information
filtering and utilization capabilities.
Importance of Token-Level Weighting . The
better performance of DDPO compared to standard
DPO validates the importance of our token-level
weighting mechanism. This approach enables more
precise focus on discriminative features between
positive and negative examples, leading to measur-
able improvements in model performance during
inference. The consistent gains across different ar-
chitectures and task domains further confirm the
effectiveness of this design choice.
5.2 Analysis
In this section, we conducted ablation experiments
and analyses to further validate the effectiveness of
our method.
Impact of Different Knowledge OrganizationFormats . We compare the effects of various inter-
mediate information organization formats on train-
ing effectiveness to emphasize the importance of
structured representation. Experiments on Llama-
3.1-8B-Instruct were conducted using two addi-
tional formats: a semi-structured keypoints format
(one key point per line) and an unstructured sum-
mary format (organizing information into a sum-
mary). Specific prompts are provided in the Ap-
pendix A.6
As illustrated in Table 2 and Figure 2, the struc-
tured graph representation leads to substantial im-
provements in both in-domain and out-of-domian
tests, while the semi-structured keypoints format
shows moderate gains. The unstructured summary
format even shows a decline in performance after
training.
The results strongly suggest that structural in-
ductive biases should be incorporated into RAG
training objectives when possible, as they provide
clearer learning signals for knowledge-intensive
tasks while maintaining compatibility with stan-
dard inference pipelines.
Generalization Across Different Generation
Pipelines . To further validate the versatility of our
approach, we evaluated the trained models across
multiple generation pipelines. These pipelines
include the Vanilla RAG pipeline, KA-Pipeline
which has benn used for data construction, Chain-
of-Thought pipeline(Wei et al., 2022), the Chain-
of-Note pipeline(Yu et al., 2023). Additional ex-
periments were conducted on the more complex
IRCOT pipeline(Trivedi et al., 2022a) involving
multi-step retrieval (see Appendix A.5).
The results can be viewed in Table 3. While
the most significant improvements naturally oc-
cur with the KA-Pipeline (aligned with our train-
ing methodology), all tested pipelines demonstrate
consistent performance gains without degradation.
This pattern holds particularly true for complex
reasoning tasks, where our method shows stable
improvements regardless of the specific pipeline
architecture.
7

Inference
PipelineTrain
MethodIn Domain Out Of Domain
Musique NQ HotpotQA 2WikimultihopQA PopQA WebQ Average Gain
EM(%) F1(%) EM(%) F1(%) EM(%) F1(%) EM(%) F1(%) EM(%) F1(%) EM(%) F1(%) EM(%) F1(%)
Llama-3.2-3B-Instruct
Vanilla4.47 9.89 33.2 45.59 27.89 37.75 11.9 22.24 34.32 41.6 17.96 35.14
KARE-RAG 5.59 13.28 33.21 46.56 27.62 39.59 15.5 25.53 38.29 45.52 19.44 36.97 1.76 2.37
Knowledge
Aware4.39 10.7 25.57 37.33 21.34 31.76 15.6 21.92 32.27 38.46 17.42 35.2
KARE-RAG 6.99 15.8 28.51 40.72 28.85 38.79 26.67 33.45 36.96 42.24 18.65 36.68 5.49 5.44
Chain Of
Thought5.01 11.8 28.03 41.01 24.0 35.43 14.15 21.07 32.92 39.99 16.98 34.98
KARE-RAG 7.24 16.8 29.31 42.73 26.41 39.03 22.57 30.69 36.23 43.42 16.14 35.64 2.92 3.81
Chain Of
Note4.18 10.63 29.15 42.42 22.04 32.84 10.19 17.43 30.83 38.34 15.94 34.59
KARE-RAG 6.21 15.16 30.44 44.45 25.06 37.87 16.87 25.8 34.69 42.71 15.94 35.42 2.97 4.13
Llama-3.1-8B-Instruct
Vanilla6 12.37 34.64 48.3 29.52 40.66 14.94 24.89 35.43 44.1 16.83 35.33
KARE-RAG 8.02 15.75 37.86 50.84 32.36 44.29 21.35 30.11 40.88 47.77 19.83 37.89 4.18 3.52
Knowledge
Aware7.57 15.13 33.07 46.03 30.2 41.17 18.54 25.78 38.79 44.76 20.13 37.86
KARE-RAG 11.42 20.53 36.21 49.36 33.98 46.01 30.61 37.92 41.92 47.55 22.1 39.95 4.82 5.04
Chain Of
Thought8.52 16.4 35.63 48.48 31.11 42.45 16.71 23.52 37.71 44.35 20.47 38.8
KARE-RAG 10.34 19.28 37.22 50.28 34.4 46.82 23.88 31.5 40.38 46.96 22.0 39.63 3.25 3.52
Chain Of
Note6.83 14.47 35.06 48.61 29.05 40.7 17.37 24.2 36.06 43.32 19.44 38.7
KARE-RAG 9.18 18.44 37.55 50.73 33.11 45.34 22.56 30.44 39.36 46.3 21.26 40.07 3.37 3.47
Qwen2.5-14B-Instruct
Vanilla6.66 14.58 32.67 47.01 30.64 42.31 19.73 29.5 36.38 43.98 18.36 35.11
KARE-RAG 8.32 17.48 36.19 49.96 33.96 46.45 25.14 33.72 38.58 46.09 20.03 36.95 3.22 3.05
Knowledge
Aware9.02 17.67 36.75 50.15 35.91 48.25 24.51 30.88 39.2 45.95 19.88 37.82
KARE-RAG 11.09 20.78 37.19 50.46 37.49 50.56 34.65 42.12 40.65 47.3 20.72 38.95 2.89 3.27
Chain Of
Thought9.02 17.04 37.99 51.16 34.8 46.66 21.15 28.08 38.28 45.35 19.78 37.48
KARE-RAG 10.72 21.63 37.34 51.07 36.88 50.22 29.61 38.37 39.42 46.87 18.8 37.36 2.01 3.03
Chain Of
Note8.36 16.81 35.31 49.25 32.92 45.54 21.37 30.61 36.31 44.26 17.77 36.2
KARE-RAG 9.27 19.55 36.49 50.49 35.04 48.39 28.05 37.61 37.99 45.65 18.36 37.58 2.45 2.77
Table 3: The results demonstrate the effectiveness of the KARE-RAG-trained model across various RAG generation
pipelines, with the final two columns showing its average EM and F1 score improvements over the baseline
(untrained) model for each pipeline.
Train Method MMLU(Acc) MMLU-Pro(Acc)
68.0 43.97
Vanilla(DPO) 66.9 40.25
KARE-RAG(DDPO) 67.9 44.60
Table 4: Test the general abblility of our method with
MMLU and MMLU-Pro. Llama-3.1-8B-Instruct is ap-
plied for the test.
These findings strongly suggest that KARE-
RAG’s effectiveness stems from its fundamental
enhancement of how models process and utilize re-
trieved information, rather than from optimization
for any particular pipeline design. The consistent
performance gains across diverse implementations
highlight the method’s general applicability to vari-
ous RAG architectures in real-world scenarios.
Test General Ability . While our training
method was optimized for the RAG scenario, po-
tentially affecting the model’s general capabili-
ties, we evaluated its performance on the MMLU
(Hendrycks et al., 2020) and MMLU-Pro (Wang
et al., 2024) datasets. As shown in Table 4, the train-
ing method did not negatively impact the model’s
general abilities. The MMLU metrics were con-
sistent with the untrained model, and the MMLU-
Pro metrics showed improvement. In contrast, the
model trained with Vanilla RAG data experienced
a significant decline on both tests. This suggeststhat our method does not compromise general ca-
pabilities and may even improve performance on
complex tasks in MMLU-Pro.
6 Conclusion
Our work presents KARE-RAG, a training-based
approach for enhancing RAG systems. KARE-
RAG introduces structured knowledge organiza-
tion as an intermediate learning objective and em-
ploys DDPO to provide fine-grained optimization
through dynamic weighting, enabling models to
develop improved information filtering and integra-
tion capabilities. This training strategy enhances
information processing robustness while maintain-
ing full compatibility with standard RAG pipelines
and existing enhancement techniques. Comprehen-
sive experiments demonstrate consistent improve-
ments in handling retrieval contexts across multiple
benchmarks. KARE-RAG proves effective across
different model scales and requires no architectural
changes during deployment, suggesting training-
based improvements can serve as a practical com-
plement to existing RAG enhancement approaches.
Limitations
While our method demonstrates strong empirical
performance, several implementation characteris-
tics warrant discussion.
8

Representation Format Scope . The current
framework primarily utilizes graph-structured rep-
resentations, which have shown strong experimen-
tal performance for modeling entity relationships.
While this approach proves effective, we acknowl-
edge that alternative structured formats—such as
hierarchical trees for taxonomic knowledge or tem-
poral sequences for event-based information—may
offer complementary advantages and represent
promising directions for future investigation.
Data Construction Pipeline . Although our re-
finement process handles the majority of cases ef-
fectively, certain challenging samples remain dif-
ficult to correct due to inherent limitations in the
refinement model’s capabilities. This suggests two
potential improvement avenues: (1) integration of
dedicated reasoning modules for enhanced error
verification, and (2) development of specialized re-
finement models targeting specific error patterns.
These directions may further improve the pipeline’s
robustness while maintaining its current efficiency.
References
Josh Achiam, Steven Adler, Sandhini Agarwal, Lama
Ahmad, Ilge Akkaya, Florencia Leoni Aleman,
Diogo Almeida, Janko Altenschmidt, Sam Altman,
Shyamal Anadkat, et al. 2023. Gpt-4 technical report.
arXiv preprint arXiv:2303.08774 .
Akari Asai, Zeqiu Wu, Yizhong Wang, Avirup Sil, and
Hannaneh Hajishirzi. 2023. Self-rag: Learning to
retrieve, generate, and critique through self-reflection.
arXiv preprint arXiv:2310.11511 .
Jonathan Berant, Andrew Chou, Roy Frostig, and Percy
Liang. 2013. Semantic parsing on freebase from
question-answer pairs. In Proceedings of the 2013
conference on empirical methods in natural language
processing , pages 1533–1544.
Xiaojun Chen, Shengbin Jia, and Yang Xiang. 2020. A
review: Knowledge reasoning over knowledge graph.
Expert systems with applications , 141:112948.
Qingxiu Dong, Lei Li, Damai Dai, Ce Zheng, Jingyuan
Ma, Rui Li, Heming Xia, Jingjing Xu, Zhiyong Wu,
Tianyu Liu, et al. 2022. A survey on in-context learn-
ing.arXiv preprint arXiv:2301.00234 .
Darren Edge, Ha Trinh, Newman Cheng, Joshua
Bradley, Alex Chao, Apurva Mody, Steven Truitt,
and Jonathan Larson. 2024. From local to global: A
graph rag approach to query-focused summarization.
arXiv preprint arXiv:2404.16130 .
Jinyuan Fang, Zaiqiao Meng, and Craig Macdonald.
2024. Trace the evidence: Constructing knowledge-
grounded reasoning chains for retrieval-augmented
generation. arXiv preprint arXiv:2406.11460 .Yifu Gao, Linbo Qiao, Zhigang Kan, Zhihua Wen,
Yongquan He, and Dongsheng Li. 2024. Two-stage
generative question answering on temporal knowl-
edge graph using large language models. arXiv
preprint arXiv:2402.16568 .
Yunfan Gao, Yun Xiong, Xinyu Gao, Kangxiang Jia,
Jinliu Pan, Yuxi Bi, Yi Dai, Jiawei Sun, and Haofen
Wang. 2023. Retrieval-augmented generation for
large language models: A survey. arXiv preprint
arXiv:2312.10997 .
Xiaoxin He, Yijun Tian, Yifei Sun, Nitesh V Chawla,
Thomas Laurent, Yann LeCun, Xavier Bresson, and
Bryan Hooi. 2024. G-retriever: Retrieval-augmented
generation for textual graph understanding and ques-
tion answering. arXiv preprint arXiv:2402.07630 .
Dan Hendrycks, Collin Burns, Steven Basart, Andy Zou,
Mantas Mazeika, Dawn Song, and Jacob Steinhardt.
2020. Measuring massive multitask language under-
standing. arXiv preprint arXiv:2009.03300 .
Xanh Ho, Anh-Khoa Duong Nguyen, Saku Sugawara,
and Akiko Aizawa. 2020. Constructing a multi-hop
qa dataset for comprehensive evaluation of reasoning
steps. arXiv preprint arXiv:2011.01060 .
Aidan Hogan, Eva Blomqvist, Michael Cochez, Clau-
dia d’Amato, Gerard De Melo, Claudio Gutierrez,
Sabrina Kirrane, José Emilio Labra Gayo, Roberto
Navigli, Sebastian Neumaier, et al. 2021. Knowledge
graphs. ACM Computing Surveys (Csur) , 54(4):1–
37.
Edward J Hu, Yelong Shen, Phillip Wallis, Zeyuan
Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang,
and Weizhu Chen. 2021. Lora: Low-rank adap-
tation of large language models. arXiv preprint
arXiv:2106.09685 .
Yuntong Hu, Zhihan Lei, Zheng Zhang, Bo Pan,
Chen Ling, and Liang Zhao. 2024. Grag: Graph
retrieval-augmented generation. arXiv preprint
arXiv:2405.16506 .
Yongfeng Huang, Yanyang Li, Yichong Xu, Lin Zhang,
Ruyi Gan, Jiaxing Zhang, and Liwei Wang. 2023.
Mvp-tuning: Multi-view knowledge retrieval with
prompt tuning for commonsense reasoning. In Pro-
ceedings of the 61st Annual Meeting of the Associa-
tion for Computational Linguistics (Volume 1: Long
Papers) , pages 13417–13432.
Soyeong Jeong, Jinheon Baek, Sukmin Cho, Sung Ju
Hwang, and Jong C Park. 2024. Adaptive-rag: Learn-
ing to adapt retrieval-augmented large language mod-
els through question complexity. arXiv preprint
arXiv:2403.14403 .
Jinhao Jiang, Kun Zhou, Wayne Xin Zhao, and Ji-Rong
Wen. 2022. Great truths are always simple: A rather
simple knowledge encoder for enhancing the com-
monsense reasoning capacity of pre-trained models.
arXiv preprint arXiv:2205.01841 .
9

Zhengbao Jiang, Frank F Xu, Luyu Gao, Zhiqing
Sun, Qian Liu, Jane Dwivedi-Yu, Yiming Yang,
Jamie Callan, and Graham Neubig. 2023. Ac-
tive retrieval augmented generation. arXiv preprint
arXiv:2305.06983 .
Jiajie Jin, Yutao Zhu, Xinyu Yang, Chenghao Zhang,
and Zhicheng Dou. 2024. Flashrag: A modular
toolkit for efficient retrieval-augmented generation
research. CoRR , abs/2405.13576.
Nikhil Kandpal, Haikang Deng, Adam Roberts, Eric
Wallace, and Colin Raffel. 2023. Large language
models struggle to learn long-tail knowledge. In In-
ternational Conference on Machine Learning , pages
15696–15707. PMLR.
Vladimir Karpukhin, Barlas O ˘guz, Sewon Min, Patrick
Lewis, Ledell Wu, Sergey Edunov, Danqi Chen, and
Wen-tau Yih. 2020. Dense passage retrieval for
open-domain question answering. arXiv preprint
arXiv:2004.04906 .
Tom Kwiatkowski, Jennimaria Palomaki, Olivia Red-
field, Michael Collins, Ankur Parikh, Chris Alberti,
Danielle Epstein, Illia Polosukhin, Jacob Devlin, Ken-
ton Lee, et al. 2019. Natural questions: a benchmark
for question answering research. Transactions of the
Association for Computational Linguistics , 7:453–
466.
Xin Lai, Zhuotao Tian, Yukang Chen, Senqiao Yang, Xi-
angru Peng, and Jiaya Jia. 2024. Step-dpo: Step-wise
preference optimization for long-chain reasoning of
llms. arXiv preprint arXiv:2406.18629 .
Omer Levy, Minjoon Seo, Eunsol Choi, and Luke
Zettlemoyer. 2017. Zero-shot relation extrac-
tion via reading comprehension. arXiv preprint
arXiv:1706.04115 .
Patrick Lewis, Ethan Perez, Aleksandra Piktus, Fabio
Petroni, Vladimir Karpukhin, Naman Goyal, Hein-
rich Küttler, Mike Lewis, Wen-tau Yih, Tim Rock-
täschel, et al. 2020. Retrieval-augmented generation
for knowledge-intensive nlp tasks. Advances in Neu-
ral Information Processing Systems , 33:9459–9474.
Shiyang Li, Yifan Gao, Haoming Jiang, Qingyu Yin,
Zheng Li, Xifeng Yan, Chao Zhang, and Bing Yin.
2023. Graph reasoning for question answering with
triplet retrieval. arXiv preprint arXiv:2305.18742 .
Xinze Li, Sen Mei, Zhenghao Liu, Yukun Yan, Shuo
Wang, Shi Yu, Zheni Zeng, Hao Chen, Ge Yu,
Zhiyuan Liu, et al. 2024a. Rag-ddr: Optimizing
retrieval-augmented generation using differentiable
data rewards. arXiv preprint arXiv:2410.13509 .
Zhuoyang Li, Liran Deng, Hui Liu, Qiaoqiao Liu, and
Junzhao Du. 2024b. Unioqa: A unified framework
for knowledge graph question answering with large
language models. arXiv preprint arXiv:2406.02110 .
Stephanie Lin, Jacob Hilton, and Owain Evans. 2021.
Truthfulqa: Measuring how models mimic human
falsehoods. arXiv preprint arXiv:2109.07958 .Xi Victoria Lin, Xilun Chen, Mingda Chen, Weijia Shi,
Maria Lomeli, Richard James, Pedro Rodriguez, Ja-
cob Kahn, Gergely Szilvasy, Mike Lewis, et al. 2023.
Ra-dit: Retrieval-augmented dual instruction tuning.
InThe Twelfth International Conference on Learning
Representations .
Nelson F Liu, Kevin Lin, John Hewitt, Ashwin Paran-
jape, Michele Bevilacqua, Fabio Petroni, and Percy
Liang. 2024a. Lost in the middle: How language
models use long contexts. Transactions of the Asso-
ciation for Computational Linguistics , 12:157–173.
Zhihan Liu, Miao Lu, Shenao Zhang, Boyi Liu, Hongyi
Guo, Yingxiang Yang, Jose Blanchet, and Zhaoran
Wang. 2024b. Provably mitigating overoptimization
in rlhf: Your sft loss is implicitly an adversarial regu-
larizer. arXiv preprint arXiv:2405.16436 .
Shayne Longpre, Kartik Perisetla, Anthony Chen,
Nikhil Ramesh, Chris DuBois, and Sameer Singh.
2021. Entity-based knowledge conflicts in question
answering. arXiv preprint arXiv:2109.05052 .
Shayne Longpre, Kartik Perisetla, Anthony Chen,
Nikhil Ramesh, Chris DuBois, and Sameer Singh.
2022. Entity-based knowledge conflicts in question
answering. Preprint , arXiv:2109.05052.
Hongyin Luo, Yung-Sung Chuang, Yuan Gong, Tian-
hua Zhang, Yoon Kim, Xixin Wu, Danny Fox, He-
len Meng, and James Glass. 2023. Sail: Search-
augmented instruction learning. arXiv preprint
arXiv:2305.15225 .
Xinbei Ma, Yeyun Gong, Pengcheng He, Hai Zhao,
and Nan Duan. 2023. Query rewriting for retrieval-
augmented large language models. arXiv preprint
arXiv:2305.14283 .
Alex Mallen, Akari Asai, Victor Zhong, Rajarshi Das,
Daniel Khashabi, and Hannaneh Hajishirzi. 2022.
When not to trust language models: Investigating
effectiveness of parametric and non-parametric mem-
ories. arXiv preprint arXiv:2212.10511 .
Kelong Mao, Zhicheng Dou, Fengran Mo, Jiewen Hou,
Haonan Chen, and Hongjin Qian. 2023. Large lan-
guage models know your contextual search intent:
A prompting framework for conversational search.
arXiv preprint arXiv:2303.06573 .
Shengyu Mao, Yong Jiang, Boli Chen, Xiao Li, Peng
Wang, Xinyu Wang, Pengjun Xie, Fei Huang, Huajun
Chen, and Ningyu Zhang. 2024. Rafe: Ranking
feedback improves query rewriting for rag. arXiv
preprint arXiv:2405.14431 .
Costas Mavromatis and George Karypis. 2024. Gnn-
rag: Graph neural retrieval for large language model
reasoning. arXiv preprint arXiv:2405.20139 .
Long Ouyang, Jeffrey Wu, Xu Jiang, Diogo Almeida,
Carroll Wainwright, Pamela Mishkin, Chong Zhang,
Sandhini Agarwal, Katarina Slama, Alex Ray, et al.
10

2022. Training language models to follow instruc-
tions with human feedback. Advances in neural in-
formation processing systems , 35:27730–27744.
Boci Peng, Yun Zhu, Yongchao Liu, Xiaohe Bo,
Haizhou Shi, Chuntao Hong, Yan Zhang, and Siliang
Tang. 2024. Graph retrieval-augmented generation:
A survey. arXiv preprint arXiv:2408.08921 .
Rafael Rafailov, Archit Sharma, Eric Mitchell, Christo-
pher D Manning, Stefano Ermon, and Chelsea Finn.
2024. Direct preference optimization: Your language
model is secretly a reward model. Advances in Neu-
ral Information Processing Systems , 36.
Ori Ram, Yoav Levine, Itay Dalmedigos, Dor Muhlgay,
Amnon Shashua, Kevin Leyton-Brown, and Yoav
Shoham. 2023. In-context retrieval-augmented lan-
guage models. Transactions of the Association for
Computational Linguistics , 11:1316–1331.
John Schulman, Filip Wolski, Prafulla Dhariwal,
Alec Radford, and Oleg Klimov. 2017. Proxi-
mal policy optimization algorithms. arXiv preprint
arXiv:1707.06347 .
Weijia Shi, Sewon Min, Michihiro Yasunaga, Min-
joon Seo, Rich James, Mike Lewis, Luke Zettle-
moyer, and Wen-tau Yih. 2023. Replug: Retrieval-
augmented black-box language models. arXiv
preprint arXiv:2301.12652 .
Devendra Singh, Siva Reddy, Will Hamilton, Chris
Dyer, and Dani Yogatama. 2021. End-to-end train-
ing of multi-document reader and retriever for open-
domain question answering. Advances in Neural
Information Processing Systems , 34:25968–25981.
Qwen Team. 2024. Qwen2.5: A party of foundation
models.
Peihao Tong, Qifan Zhang, and Junjie Yao. 2019. Lever-
aging domain context for question answering over
knowledge graph. Data Science and Engineering ,
4(4):323–335.
Hugo Touvron, Thibaut Lavril, Gautier Izacard, Xavier
Martinet, Marie-Anne Lachaux, Timothée Lacroix,
Baptiste Rozière, Naman Goyal, Eric Hambro,
Faisal Azhar, et al. 2023. Llama: Open and effi-
cient foundation language models. arXiv preprint
arXiv:2302.13971 .
Harsh Trivedi, Niranjan Balasubramanian, Tushar
Khot, and Ashish Sabharwal. 2022a. Interleav-
ing retrieval with chain-of-thought reasoning for
knowledge-intensive multi-step questions. arXiv
preprint arXiv:2212.10509 .
Harsh Trivedi, Niranjan Balasubramanian, Tushar Khot,
and Ashish Sabharwal. 2022b. Musique: Multi-
hop questions via single-hop question composition.
Preprint , arXiv:2108.00573.Leandro von Werra, Younes Belkada, Lewis Tunstall,
Edward Beeching, Tristan Thrush, Nathan Lambert,
Shengyi Huang, Kashif Rasul, and Quentin Gal-
louédec. 2020. Trl: Transformer reinforcement learn-
ing. https://github.com/huggingface/trl .
Liang Wang, Nan Yang, and Furu Wei. 2023.
Query2doc: Query expansion with large language
models. arXiv preprint arXiv:2303.07678 .
Yubo Wang, Xueguang Ma, Ge Zhang, Yuansheng Ni,
Abhranil Chandra, Shiguang Guo, Weiming Ren,
Aaran Arulraj, Xuan He, Ziyan Jiang, et al. 2024.
Mmlu-pro: A more robust and challenging multi-task
language understanding benchmark. arXiv preprint
arXiv:2406.01574 .
Jason Wei, Xuezhi Wang, Dale Schuurmans, Maarten
Bosma, Fei Xia, Ed Chi, Quoc V Le, Denny Zhou,
et al. 2022. Chain-of-thought prompting elicits rea-
soning in large language models. Advances in neural
information processing systems , 35:24824–24837.
Siye Wu, Jian Xie, Jiangjie Chen, Tinghui Zhu, Kai
Zhang, and Yanghua Xiao. 2024. How easily do
irrelevant inputs skew the responses of large language
models? arXiv preprint arXiv:2404.03302 .
Shitao Xiao, Zheng Liu, Peitian Zhang, and Niklas
Muennighoff. 2023. C-pack: Packaged resources
to advance general chinese embedding. Preprint ,
arXiv:2309.07597.
Lee Xiong, Chenyan Xiong, Ye Li, Kwok-Fung Tang,
Jialin Liu, Paul Bennett, Junaid Ahmed, and Arnold
Overwijk. 2020. Approximate nearest neighbor neg-
ative contrastive learning for dense text retrieval.
arXiv preprint arXiv:2007.00808 .
Fangyuan Xu, Weijia Shi, and Eunsol Choi. 2023. Re-
comp: Improving retrieval-augmented lms with com-
pression and selective augmentation. arXiv preprint
arXiv:2310.04408 .
Shicheng Xu, Liang Pang, Huawei Shen, Xueqi Cheng,
and Tat-Seng Chua. 2024a. Search-in-the-chain: In-
teractively enhancing large language models with
search for knowledge-intensive tasks. In Proceed-
ings of the ACM on Web Conference 2024 , pages
1362–1373.
Zhipeng Xu, Zhenghao Liu, Yukun Yan, Shuo Wang,
Shi Yu, Zheni Zeng, Chaojun Xiao, Zhiyuan Liu,
Ge Yu, and Chenyan Xiong. 2024b. Activerag: Au-
tonomously knowledge assimilation and accommo-
dation through retrieval-augmented agents. Preprint ,
arXiv:2402.13547.
An Yang, Baosong Yang, Binyuan Hui, Bo Zheng,
Bowen Yu, Chang Zhou, Chengpeng Li, Chengyuan
Li, Dayiheng Liu, Fei Huang, Guanting Dong, Hao-
ran Wei, Huan Lin, Jialong Tang, Jialin Wang, Jian
Yang, Jianhong Tu, Jianwei Zhang, Jianxin Ma, Jin
Xu, Jingren Zhou, Jinze Bai, Jinzheng He, Junyang
Lin, Kai Dang, Keming Lu, Keqin Chen, Kexin Yang,
Mei Li, Mingfeng Xue, Na Ni, Pei Zhang, Peng
11

Wang, Ru Peng, Rui Men, Ruize Gao, Runji Lin,
Shijie Wang, Shuai Bai, Sinan Tan, Tianhang Zhu,
Tianhao Li, Tianyu Liu, Wenbin Ge, Xiaodong Deng,
Xiaohuan Zhou, Xingzhang Ren, Xinyu Zhang, Xipin
Wei, Xuancheng Ren, Yang Fan, Yang Yao, Yichang
Zhang, Yu Wan, Yunfei Chu, Yuqiong Liu, Zeyu
Cui, Zhenru Zhang, and Zhihao Fan. 2024. Qwen2
technical report. arXiv preprint arXiv:2407.10671 .
Zhilin Yang, Peng Qi, Saizheng Zhang, Yoshua Ben-
gio, William W Cohen, Ruslan Salakhutdinov, and
Christopher D Manning. 2018. Hotpotqa: A dataset
for diverse, explainable multi-hop question answer-
ing.arXiv preprint arXiv:1809.09600 .
Ori Yoran, Tomer Wolfson, Ori Ram, and Jonathan
Berant. 2023. Making retrieval-augmented language
models robust to irrelevant context. arXiv preprint
arXiv:2310.01558 .
Tianyu Yu, Yuan Yao, Haoye Zhang, Taiwen He, Yifeng
Han, Ganqu Cui, Jinyi Hu, Zhiyuan Liu, Hai-Tao
Zheng, Maosong Sun, et al. 2024. Rlhf-v: Towards
trustworthy mllms via behavior alignment from fine-
grained correctional human feedback. In Proceed-
ings of the IEEE/CVF Conference on Computer Vi-
sion and Pattern Recognition , pages 13807–13816.
Wenhao Yu, Hongming Zhang, Xiaoman Pan, Kaixin
Ma, Hongwei Wang, and Dong Yu. 2023. Chain-of-
note: Enhancing robustness in retrieval-augmented
language models. arXiv preprint arXiv:2311.09210 .
A Appendix
A.1 Training Settings for Vanilla(DPO)
Due to the inclusion of knowledge organization
and CoT processes in the training data constructed
by the KARE-RAG method, it is not suitable for
direct use in Vanilla(DPO). Therefore, we made
certain adjustments to the data collection method
and reconstructed a batch of training data.
Data Generation . The data construction pro-
cess is simpler compared to KARE-RAG, with the
overall workflow illustrated in Algorithm 2. The
volume of data is slightly larger than that of KARE-
RAG, with approximately 4k data points for differ-
ent models. We also attempted to obtain the same
amount of training data as the KARE-RAG method
through sampling for experiments, but the differ-
ences in experimental results were not significant.
Consequently, we ultimately did not sample the
training data.
Implementation Details . When training with
standard DPO alone, the process is prone to in-
stability. Therefore, in our experiments, we also
incorporated SFT Loss as a regularization term.
We trained for one epoch with a learning rate set to
5e-5 and the weight of the SFT Loss set to 0.1. WeAlgorithm 2 Vanilla RAG Data Construction
1:Generate initial output y−
Genusing LLM Gen
with Vanilla RAG Pipeline
2:ify−
Gen̸=ygndthen
3: Check document adequacy: LLM Exp(q⊕
D⊕ygnd)
4: ifDocument Dis sufficient then
5: Add(ygnd, yGen)as contrastive pair, yGen
is the rejected output, ygndis the chosen
output
6: else
7: Discard the sample
8: end if
9:end if
also experimented with training using DDPO, but
since the positive and negative examples obtained
from Vanilla RAG were relatively brief, it did not
yield satisfactory results.
A.2 Computational Efficiency
Our framework demonstrates strong practical effi-
ciency in both data construction and model train-
ing phases. For data generation, the pipeline
leverages GPT-4o-mini’s capability to perform
targeted refinements, keeping the construction
cost around $20. The training process employs
parameter-efficient LoRA adapters combined with
DeepSpeed-Zero3 optimization, enabling efficient
fine-tuning on a single A100-40G GPU. Specifi-
cally, training completes in under 1 hour for 3B
models and within 4 hours for 14B models.
A.3 Evaluation on Non-QA Tasks
To comprehensively evaluate the generalization ca-
pability of KARE-RAG beyond question answer-
ing, we conduct additional experiments on two rep-
resentative non-QA tasks: multiple-choice question
dataset TruthfulQA and slot filling dataset Zero-
shot RE. The results in Table 5 demonstrate con-
sistent performance gains across these diverse task
formats.
The TruthfulQA benchmark assesses models’
ability to reject misleading but statistically plau-
sible answers. Our findings reveal consistent im-
provements across all model sizes, the structured
knowledge training helps models better recognize
factual inconsistencies. We can observe similar
improvement in the result of Zero-shot RE task.
These results substantiate that KARE-RAG’s ad-
vantages extend beyond its original QA design con-
12

Train MethodTruthfulQA Zero-shot RE
Rouge-1(%) BLEU(%) Precision(%) Recall(%) F1(%)
Llama-3.2-3B-Instruct
13.54 6.09 47.79 54.35 49.06
KARE-RAG 18.78 7.75 48.69 62.0 51.53
Llama-3.1-8B-Instruct
14.8 5.49 49.48 60.16 51.87
KARE-RAG 17.76 7.42 56.48 64.55 58.29
Qwen2.5-14B-Instruct
18.98 7.26 49.37 68.05 53.32
KARE-RAG 20.08 8.68 50.14 68.95 54.1
Table 5: Evaluation results on non-QA tasks: multiple-choice question dataset TruthfulQA and slot filling dataset
Zero-shot RE. The best result in each model group is highlighted.
text, regardless of output format, providing evi-
dence for its general applicability to knowledge-
intensive NLP tasks.
A.4 Ablation study on the needs for SFT Loss
In this section, we validate the role of incorpo-
rating SFT Loss during the preference learning
training process through a series of comparative
experiments and analyses. As shown in Table 6,
incorporating both DPO Loss and SFT Loss signif-
icantly improves the model’s training effectiveness
and stability on OOD tasks, compared to using
DPO Loss alone. Standard DPO Loss amplifies
the reward difference between positive and neg-
ative examples, increasing the likelihood of sam-
pling positive examples (Equation 5). Figures 3a
and 3b show that standard DPO behaves as ex-
pected, with reward/accuracy approaching 1 and
reward/margin increasing. However, due to the
minimal differences between positive and negative
examples and the lack of regularization in DPO, the
model often overfits, causing both reward/chosen
and reward/rejected to decrease (Figure 3c). This
indicates that the model outputs fewer positive ex-
amples than the untrained model, leading to subop-
timal training. Including SFT Loss mitigates this
issue, keeping reward/chosen positive, maintaining
reward/accuracy near 1, and preventing overfitting.
A.5 Evaluation on IRCOT pipeline
We further validate our method on IRCOT (Itera-
tive Retrieval Chain-of-Thought), a more complex
RAG framework that requires multi-round retrieval
and reasoning. As shown in Table 7, KARE-RAG
demonstrates consistent improvements across this
challenging setup. The Llama-3.1-8B model shows
particularly strong gains (+10.6% average accu-racy), proving our training method’s effectiveness
even when handling intricate multi-step retrieval
scenarios. These results confirm that the knowl-
edge refinement capabilities learned by KARE-
RAG generalize well to sophisticated RAG archi-
tectures requiring iterative information gathering
and reasoning.
A.6 KA-RAG Pipeline Prompts
In this section, we present all the prompts uti-
lized in our experiments. For Vanilla RAG, we
directly used the standard prompt template pro-
vided by FlashRAG(Jin et al., 2024) without any
modifications. Table 8 contains the prompts for
the KA-RAG Pipeline when employing a Knowl-
edge Graph as the intermediate representation for
knowledge organization. Tables 9 and 10 display
the prompts for the scenarios using Keypoints and
Summary, respectively. Table 11 showcases the
prompts we employed during the data construc-
tion phase. When constructing training data for
Vanilla(DPO) in Appendix A.1, we use the same
prompts in Table 11 to check answerability.
13

(a) Reward/accuracy
 (b) Reward/margin
 (c) Reward/chosen
Figure 3: Training curves with or with out SFT Loss.
Inference
PipelineTrain
MethodIn Domain Out Of Domain
Musique NQ HotpotQA 2WikimultihopQA PopQA WebQ Average Gain
EM(%) F1(%) EM(%) F1(%) EM(%) F1(%) EM(%) F1(%) EM(%) F1(%) EM(%) F1(%) EM(%) F1(%)
Llama-3.2-3B-Instruct
Vanilla 4.47 9.89 33.2 45.59 27.89 37.75 11.9 22.24 34.32 41.6 17.96 35.14
Knowledge
Aware4.39 10.7 25.57 37.33 21.34 31.76 15.6 21.92 32.27 38.46 17.42 35.2 -2.61 -3.53
DDPO 6.99 15.8 28.51 40.72 28.85 38.79 26.67 33.45 36.96 42.24 18.65 36.68 2.87 1.91
DDPO w/o SFT 6.45 16.61 23.33 37.53 23.2 36.35 24.49 35.14 31.98 39.1 14.12 33.82 -1.63 -0.08
Llama-3.1-8B-Instruct
Vanilla 6 12.37 34.64 48.3 29.52 40.66 14.94 24.89 35.43 44.1 16.83 35.33
Knowledge
Aware7.57 15.13 33.07 46.03 30.2 41.17 18.54 25.78 38.79 44.76 20.13 37.86 1.87 0.46
DDPO 11.42 20.53 36.21 49.36 33.98 46.01 30.61 37.92 41.92 47.55 22.1 39.95 6.69 5.5
DDPO w/o SFT 10.01 19.77 32.03 44.96 28.57 41.67 30.74 39.67 36.06 42.11 20.17 39.22 3.24 2.87
Qwen2.5-14B-Instruct
Vanilla 6.66 14.58 32.67 47.01 30.64 42.31 19.73 29.5 36.38 43.98 18.36 35.11
Knowledge
Aware9.02 17.67 36.75 50.15 35.91 48.25 24.51 30.88 39.2 45.95 19.88 37.82 3.69 3.03
DDPO 11.09 20.78 37.19 50.46 37.49 50.56 34.65 42.12 40.65 47.3 20.72 38.95 6.58 6.3
DDPO w/o SFT 10.47 21.14 32.56 46.85 35.03 49.66 32.95 43.01 38.12 46.04 15.6 34.83 3.3 4.5
Table 6: Performance Evaluation on Non-QA Tasks across Model Scales. Results demonstrate KARE-RAG’s
effectiveness on TruthfulQA (multiple-choice QA) and Zero-shot RE (slot filling) tasks. The best result in each
model group is highlighted.
Train
MethodIn Domain Out Of Domain
Musique NQ HotpotQA 2WikimultihopQA PopQA WebQ Average Gain
ACC(%) ACC(%) ACC(%) ACC(%) ACC(%) ACC(%) ACC(%)
Llama-3.2-3B-Instruct
10.92 52.32 37.58 35.90 48.07 45.62
KARE-RAG 15.68 56.22 43.36 38.14 53.80 51.62 4.74
Llama-3.1-8B-Instruct
7.70 46.21 32.22 29.80 41.57 39.61
KARE-RAG 14.93 57.45 45.05 38.48 54.32 50.49 10.6
Qwen2.5-14B-Instruct
7.78 52.15 30.83 35.42 45.29 46.65
KARE-RAG 9.76 55.57 35.84 35.56 48.21 49.41 2.71
Table 7: Performance Evaluation on IRCOT Pipeline. Results demonstrate KARE-RAG’s effectiveness in multi-step
retrieval scenarios across different model scales. The best result in each model group is highlighted. The rightmost
column indicates average accuracy gains over baseline models.
14

Knowledge Organization
System Prompt You are a helpful AI assistant that are good at extracting crucial information
from documents which are helpful for answering a given question. For a given
question, focus on identifying the key entities, their attributes, and their
relationships that are directly relevant to generating an accurate answer. Only
include entities and attributes that are crucial for understanding and forming the
answer, and avoid unnecessary details.
Your response must include the following keys and strictly adhere to the exact
structure without any additional text before or after the keys:
Entities:
- [Entity 1] (Attributes: [Attribute 1, Attribute 2, ...])
- [Entity 2] (Attributes: [Attribute 1, Attribute 2, ...])
...
Relationships:
1. [Entity 1] -> [Relationship] -> [Entity 2]
2. ...
Important Note:
1.Do not include any extra commentary or unnecessary details in the
response.
2. Closely follow the structure provided above and ensure that the response is
concise and directly addresses the question.
3.Don’t provide the answer to the question. Instead, focus on extracting the
key entities, attributes, and relationships that are essential for answering the
question accurately.
User Prompt Question: {question}
Documents: {reference}
Knowledge Graph:
CoT
System Prompt You are a helpful AI assistant that are good at doing reasoning on the knowledge
graph to answer a given question. For a given question and relevant knowledge
graph, provide the step by step reasoning process to derive the answer. Make
sure the reasoning steps are logical, coherent, and directly related to the question
and the information in the knowledge graph.
User Prompt Question: {question}
Knowledge Graph: { yKG}
Reasoning Steps:
Generation
System Prompt You are a helpful AI assistant that are good at generating final answers based on
the reasoning steps provided for a given question.
Important Notes:
1. Make sure the final answer is accurate, concise, and directly addresses the
question.
2. Only give me the answer and do not output any other words.
User Prompt Question: {question}
Reasoning Steps: { yCoT}
Answer:
Table 8: KA-Pipeline Prompt(Graph Format)
15

Knowledge Organization
System Prompt You are a helpful AI assistant that are good at extracting crucial information
from documents which are helpful for answering a given question.For a given
question, please thoroughly analyze the list of documents (given as strings) and
extract the most
relevant key points that directly answer the question.Ensure that each key point
is directly supported by evidence found within the documents, and avoid
unnecessary details.
Your response must include the following keys and strictly adhere to the exact
structure without any additional text before or after the keys:
Key Points:
1. [Key Point 1]
2. [Key Point 2]
...
Important Note:
1.Do not include any extra commentary or unnecessary details in the
response.
2. Closely follow the structure provided above and ensure that the response is
concise and directly addresses the question.
3.Don’t provide the answer to the question. Instead, focus on extracting the
key points that are essential for answering the question accurately.
User Prompt Question: {question}
Documents: {reference}
KeyPoints:
CoT
System Prompt You are a helpful AI assisitant that are good at doing reasoning on the keypoints
to answer a given question.For a given question and relevant keypoints, provide
the step by step reasoning process to derive the answer.Make sure the reasoning
steps are logical, coherent, and directly related to the question and the
information in the keypoints.
User Prompt Question: {question}
Keypoints: { yKey}
Reasoning Steps:
Generation
System Prompt You are a helpful AI assistant that are good at generating final answers based on
the reasoning steps provided for a given question.
Important Notes:
1. Make sure the final answer is accurate, concise, and directly addresses the
question.
2. Only give me the answer and do not output any other words.
User Prompt Question: {question}
Reasoning Steps: { yCoT}
Answer:
Table 9: KA-Pipeline Prompt(Kepoints Format)
16

Knowledge Organization
System Prompt You are a helpful AI assistant that are good at extracting crucial information
from documents which are helpful for answering a given question.For a given
question, you need to extract a note that is directly relevant to generating an
accurate answer.
Important Note:
1.Don’t directly provide the answer to the question. Instead, focus on
extracting the relevant information that is essential for answering the question
accurately and formulating a note.
2. The length of the response is limited, so make sure to include only the most
relevant information.
User Prompt Question: {question}
Documents: {reference}
Note:
CoT
System Prompt You are a helpful AI assisitant that are good at doing reasoning on the note to
answer a given question.For a given question with relevant note, provide the step
by step reasoning process to derive the answer.Make sure the reasoning steps are
logical, coherent, and directly related to the question and the information in the
note.
User Prompt Question: {question}
Note: { yNote}
Reasoning Steps:
Generation
System Prompt You are a helpful AI assistant that are good at generating final answers based on
the reasoning steps provided for a given question.
Important Notes:
1. Make sure the final answer is accurate, concise, and directly addresses the
question.
2. Only give me the answer and do not output any other words.
User Prompt Question: {question}
Reasoning Steps: { yCoT}
Answer:
Table 10: KA-Pipeline Prompt(Summary Format)
17

Check Answerability
System Prompt You are a helpful AI assistant that is very good at judging whether the answer
can be derived from the documents for a given question. You will be given a
question, a set of documents, and golden answers. Please determine whether any
of the golden answers can be derived from the documents for the given question.
If any of the golden answers can be derived from the documents, the judgement
should be "True". If none of the golden answers can be derived from the
documents, the judgement should be "False". Do not output any explanation,
only output the judgement.
User Prompt Question: {question}
Documents: {reference}
Golden Answers: {golden_answers}
Judgement:
Refine Prompt
System Prompt You are a professional AI assistant that is very good at refining the flawed
knowledge graph. The knowledge graph is extracted from the documents for a
given question so that it can be helpful for answering the question. But the
current knowledge graph may contain incorrect information, redundant
information, or lack critical information, making it impossible to deduce the
correct answer for the given question. You will be given a question, a set of
documents, the flawed knowledge graph and golden answers. Your task is to
add, remove, or modify the content in the knowledge graph to make it accurate
and relevant to answering the question. Make sure that the refined knowledge
graph’s content is both relevant to answering the question and entirely derived
from the provided document.
Important Notes:
1. The new knowledge graph should be refined based on the flawed knowledge
graph, don’t start from scratch .
2. The refined knowledge graph should be of the same format as the flawed
knowledge graph.
3.Do not directly add the golden answers to the knowledge graph, the
refined knowledge graph should be derived from the documents.
4. Do not output the explanation of your changes, only output the refined
knowledge graph!
User Prompt Question: {question}
Documents: {reference}
Flawed Knowledge Graph: { y−
KG}
Golden Answers: {golden_answers}
Refined Knowledge Graph:
Table 11: Data Construction Prompts
18