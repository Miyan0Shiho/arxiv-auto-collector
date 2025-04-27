# AlignRAG: An Adaptable Framework for Resolving Misalignments in Retrieval-Aware Reasoning of RAG

**Authors**: Jiaqi Wei, Hao Zhou, Xiang Zhang, Di Zhang, Zijie Qiu, Wei Wei, Jinzhe Li, Wanli Ouyang, Siqi Sun

**Published**: 2025-04-21 04:56:47

**PDF URL**: [http://arxiv.org/pdf/2504.14858v1](http://arxiv.org/pdf/2504.14858v1)

## Abstract
Retrieval-augmented generation (RAG) has emerged as a foundational paradigm
for knowledge-grounded text generation. However, existing RAG pipelines often
fail to ensure that the reasoning trajectories align with the evidential
constraints imposed by retrieved content. In this paper, we reframe RAG as a
problem of retrieval-aware reasoning and identify a core challenge: reasoning
misalignment-the mismatch between a model's reasoning trajectory and the
retrieved evidence. To address this challenge, we propose AlignRAG, a novel
test-time framework that mitigates reasoning misalignment through iterative
Critique-Driven Alignment (CDA) steps. In contrast to prior approaches that
rely on static training or post-hoc selection, AlignRAG actively refines
reasoning trajectories during inference by enforcing fine-grained alignment
with evidence. Our framework introduces a new paradigm for retrieval-aware
reasoning by: (1) constructing context-rich training corpora; (2) generating
contrastive critiques from preference-aware reasoning trajectories; (3)
training a dedicated \textit{Critic Language Model (CLM)} to identify reasoning
misalignments; and (4) applying CDA steps to optimize reasoning trajectories
iteratively. Empirical results demonstrate that AlignRAG consistently
outperforms all baselines and could integrate as a plug-and-play module into
existing RAG pipelines without further changes. By reconceptualizing RAG as a
structured reasoning trajectory and establishing the test-time framework for
correcting reasoning misalignments in RAG, AlignRAG provides practical
advancements for retrieval-aware generation.

## Full Text


<!-- PDF content starts -->

AlignRAG: An Adaptable Framework for
Resolving Misalignments in Retrieval-Aware Reasoning of RAG
Jiaqi Wei1,2*, Hao Zhou3∗, Xiang Zhang4, Di Zhang2, Zijie Qiu5, Wei Wei6,
Jinzhe Li2,Wanli Ouyang2, Siqi Sun2,5
1Zhejiang University,2Shanghai Artificial Intelligence Laboratory
3South China University of Technology,4University of British Columbia,
5Fudan University,6University of Hong Kong
jiaqi.wei@zju.edu.cn ,siqisun@fudan.edu.cn
Abstract
Retrieval-augmented generation (RAG) has
emerged as a foundational paradigm for
knowledge-grounded text generation. However,
existing RAG pipelines often fail to ensure that
the reasoning trajectories align with the evi-
dential constraints imposed by retrieved con-
tent. In this paper, we reframe RAG as a prob-
lem of retrieval-aware reasoning and identify
a core challenge: reasoning misalignment —the
mismatch between a model’s reasoning trajec-
tory and the retrieved evidence. To address
this challenge, we propose AlignRAG , a novel
test-time framework that mitigates reasoning
misalignment through iterative Critique-Driven
Alignment (CDA) steps. In contrast to prior ap-
proaches that rely on static training or post-hoc
selection, AlignRAG actively refines reasoning
trajectories during inference by enforcing fine-
grained alignment with evidence. Our frame-
work introduces a new paradigm for retrieval-
aware reasoning by: (1) constructing context-
rich training corpora; (2) generating contrastive
critiques from preference-aware reasoning tra-
jectories; (3) training a dedicated Critic Lan-
guage Model (CLM) to identify reasoning mis-
alignments; and (4) applying CDA steps to op-
timize reasoning trajectories iteratively. Em-
pirical results demonstrate that AlignRAG con-
sistently outperforms all baselines and could
integrate as a plug-and-play module into ex-
isting RAG pipelines without further changes.
By reconceptualizing RAG as a structured rea-
soning trajectory and establishing the test-time
framework for correcting reasoning misalign-
ments in RAG, AlignRAG provides practical
advancements for retrieval-aware generation.
All results are fully reproducible, with the code
available in here.
1 Introduction
Large Language Models (LLMs) have advanced
natural language understanding and generation,
*Equal contribution.
According to Document 1, the fire began when a spark from the bakery's oven ignited some nearby wood …Fig 1
“…The bakery was owned by Thomas Farriner, and the fire started when a spark from the bakery's oven ignited some nearby wood …”“… , which contributed to the rapid spread of the fire. The buildings were primarily made of wood and thatch, which were highly flammable…"”… The Great Fire of London was exacerbated by a strong east wind, which helped the fire to spread quickly across the city ..."…
Doc.1
Doc.2
Doc.5…Query: What was the primary cause of the Great Fire of London in 1666?
Retrieved Docs
Vanilla RAG with CoTReasoning Misalignment in RAGThe response does not clearly state that the spark from the bakery's oven was the primary cause of the fire.The response incorrectly interprets Documents 2 and 5 as contributing to the primary cause of the fire, including the wind and the flammable materials, rather than factors that exacerbated its spread.The response introduces a hallucination about a disgruntled employee setting the fire, which is not mentioned in any of the retrieved documents.
However, Document 2 suggests that the fire spread rapidly due tothe densely populated areaused in the buildings … Document 5 mentions a strong east windthat helped the fire spread quickly … Additionally,  some sources claim that the fire was intentionally set by a disgruntled employee of Thomas Farriner.
RelevantRelevantHelpful
RelevantNot HelpfulHelpfulCompleteNot CompleteNot Complete
Not RelevantNot HelpfulNot Complete
Contextual Granularity HierarchyQuery SetPopQATriviaQANaturalQA…
Retrieve
Query
Identify the errors and hallucinationsof the weak rationale , and give your constructive criticism for…
Critique ErrorsHallucinations
(A) Construct Training Set
(B) Synthesize Critique
(C) Critic LLM Training and Critique-Driven OptimizationPrompt
𝒅𝒐𝒄𝟏…𝒏
Query
𝒅𝒐𝒄𝟏…𝒏
Response
Training
CLM
LLM
Critique
Query: What was the primary cause of the Great Fire of London in 1666?Label: The primary cause was a fire that started in a bakery on Pudding Lane.
Unexpected Response: … Document 2 suggests that the fire spread rapidly due tothe densely populated areaused in the buildings …Critique: … The response does not clearly state that the spark from the bakery‘s oven was the primary cause of the fire …
Unexpected Response: … The primary cause of the Great Fire of London in 1666 was a fire that started in a bakery…LLMLLM
CPO steps x NExpected Response
Output
CLMOptimization
LLM𝒘𝒆𝒂𝒌LLM𝑺𝒕𝒓𝒐𝒏𝒈Preference Reasoning PathsUnexpectedExpected
Figure 1: Illustration of reasoning misalignment in a
typical RAG pipeline.
with retrieval-augmented generation (RAG) (Lewis
et al., 2020b; Guu et al., 2020; Lewis et al., 2020a;
Izacard et al., 2023; Kasai et al., 2024; Yang et al.,
2024b; Zhou et al., 2022; Jin et al., 2025) emerg-
ing as a key approach for integrating external
knowledge. However, RAG exhibits significant
fragility when confronted with irrelevant or noisy
evidence (Shi et al., 2023; Su et al., 2024). Ex-
isting approaches predominantly depend on static,
training-time optimizations, which are insufficient
for mitigating the challenges of dynamic error
propagation during inference (Wei et al., 2024; Wu
et al., 2025b). We identify a critical yet understud-
ied issue in RAG pipelines: reasoning misalign-
ment —the mismatch between a model’s reasoning
trajectory and the retrieved evidence. Prior efforts
focus on retrieval quality or robustness but over-
look training-time alignment (Gupta et al., 2024;
Yang et al., 2024b; Zhang et al., 2024c,d; Wang
et al., 2025b). While reflective methods like Self-
RAG (Asai et al., 2023) attempt to detect errors,
1arXiv:2504.14858v1  [cs.AI]  21 Apr 2025

they often require architectural changes or task-
specific fine-tuning, limiting general applicability.
In this paper, we reconceptualize RAG as a
process of retrieval-aware reasoning rather than
retrieval-aware generation. We argue that RAG en-
tails a structured reasoning trajectory comprising
three phases: (1) relevance assessment, (2) query-
evidence mapping, and (3) evidence-integrated jus-
tification. Reasoning misalignment arises from
breakdowns across these phases-for example, when
retrieved documents are relevant but their content
is not properly integrated into the reasoning tra-
jectories. These failure modes remain largely un-
addressed by current approaches and persist even
with high-quality retrieval.
To bridge this gap, we propose AlignRAG , a
novel framework that dynamically mitigates reason-
ing misalignment via Critique-Driven Alignment
(CDA) . Unlike general-purpose refinement meth-
ods (McAleese et al., 2024; Yuksekgonul et al.,
2024), AlignRAG introduces retrieval-aware cri-
tiques that explicitly target misalignment in the rea-
soning trajectory. At its core is a Critic Language
Model (CLM) that detects reasoning misalignment
and generates structured critiques to guide the align-
ment. The CLM is trained using a new critique
learning paradigm based on contrastive preference
reasoning trajectories derived from retrieval-aware
tasks. At test-time, AlignRAG iteratively refines
reasoning trajectories by treating them as optimiz-
able artifacts, transforming the RAG pipeline into
an active reasoning system in which critiques can
dynamically align generation with evidence.
We validate the effectiveness of our frame-
work through comprehensive experiments on seven
benchmarks and three model families, demon-
strating the robustness and efficacy of AlignRAG.
Our method consistently achieves state-of-the-art
(SOTA) performance, outperforming all existing
methods across a diverse range of tasks. We eval-
uate performance in both settings where retrieved
documents contain the correct answer (informa-
tive) and where they do not (noisy), showing that
AlignRAG enhances reasoning robustness under
both informative and noisy retrieval conditions. Be-
sides, on out-of-distribution (OOD) benchmarks,
AlignRAG exhibits stronger generalization capa-
bilities and better robustness than existing meth-
ods. Moreover, it can be seamlessly integrated as a
plug-and-play module into existing RAG pipelines
without requiring further architectural modifica-
tions, improving InstructRAG’s accuracy by 5.8%on Qwen2.5-14B. Through extensive evaluations,
we demonstrate that AlignRAG consistently outper-
forms existing methods across retrieval, reasoning,
and generalization settings.
In conclusion, this work makes the following
key contributions: (1) We formalize the reason-
ing trajectory in RAG pipelines through a struc-
tured framework, identifying retrieval-aware failure
modes as reasoning misalignment . (2) We intro-
duce critique learning , a novel pipeline for training
CLMs to generate retrieval-aware critiques using
preference reasoning trajectories. (3) We propose
AlignRAG, the first test-time framework to opti-
mize RAG’s reasoning misalignments via CDA
steps, explicitly aligning retrieval-aware critiques
with evidence integration phases. (4) Extensive
empirical validation of AlignRAG across multiple
benchmarks, demonstrating substantial improve-
ments in reasoning generation quality compared to
strong baselines.
2 Reasoning Misalignment in RAG
RAG grounds generation in an external corpus D,
but prior work has largely focused on improving
retrieval (Asai et al., 2023) or enhancing generation
via CoT prompting (Wei et al., 2024). A critical
but underexplored challenge remains: aligning the
model’s reasoning trajectory with the retrieved ev-
idence. While extended reasoning is known to
compound errors in math or code tasks (Wu et al.,
2025b; Chen et al., 2025), we argue that failures in
RAG are qualitatively distinct.
We introduce reasoning misalignment, a failure
in which reasoning trajectories diverge from re-
trieved evidence, even when the retrieval process
is accurate. This phenomenon is distinct from logi-
cal errors in mathematical reasoning (Zhang et al.,
2024b), as it arises from inductive biases that con-
flict with the evidential constraints provided by D,
the evidence corpus. Formally, reasoning mis-
alignment refers to a structured failure in the
conditional probability distribution P(y|q,D),
where yrepresents the reasoning trajectory, and q
is the query. This failure is characterized by two
critical aspects: (1) the erosion of evidential pri-
ors, where inferred reasoning deviates from the
statistical properties of D, and (2) violations of de-
ductive consistency, where reasoning trajectories
contradict logical inferences supported by D.
To systematically study this issue, we propose
the first taxonomy of reasoning misalignment in
2

According to Document 1, the fire began when a spark from the bakery's oven ignited some nearby wood …Fig 1
“…The bakery was owned by Thomas Farriner, and the fire started when a spark from the bakery's oven ignited some nearby wood …”“… , which contributed to the rapid spread of the fire. The buildings were primarily made of wood and thatch, which were highly flammable…"”… The Great Fire of London was exacerbated by a strong east wind, which helped the fire to spread quickly across the city ..."…
Doc.1
Doc.2
Doc.5…Query: What was the primary cause of the Great Fire of London in 1666?
Retrieved Docs
Vanilla RAG with CoTReasoning Misalignment in RAGThe response does not clearly state that the spark from the bakery's oven was the primary cause of the fire.The response incorrectly interprets Documents 2 and 5 as contributing to the primary cause of the fire, including the wind and the flammable materials, rather than factors that exacerbated its spread.The response introduces a hallucination about a disgruntled employee setting the fire, which is not mentioned in any of the retrieved documents.
However, Document 2 suggests that the fire spread rapidly due tothe densely populated areaused in the buildings … Document 5 mentions a strong east windthat helped the fire spread quickly … Additionally,  some sources claim that the fire was intentionally set by a disgruntled employee of Thomas Farriner.
RelevantRelevantHelpful
RelevantNot HelpfulHelpfulCompleteNot CompleteNot Complete
Not RelevantNot HelpfulNot Complete
Contextual Granularity HierarchyQuery SetPopQATriviaQANaturalQA…
Retrieve
Query
Identify the errors and hallucinationsof the weak rationale , and give your constructive criticism for…
Critique ErrorsHallucinations
(A) Training Corpus Construction
(B) Contrastive Critique Synthesis 
(C) Critic LLM Training and Critique-Driven AlignmentPrompt
𝒅𝒐𝒄𝟏…𝒏
Query
𝒅𝒐𝒄𝟏…𝒏
Response
Training
CLM
LLM
Critique
Query: What was the primary cause of the Great Fire of London in 1666?Label: The primary cause was a fire that started in a bakery on Pudding Lane.
Unexpected Response: … Document 2 suggests that the fire spread rapidly due tothe densely populated areaused in the buildings …Critique: … The response does not clearly state that the spark from the bakery‘s oven was the primary cause of the fire …
Unexpected Response: … The primary cause of the Great Fire of London in 1666 was a fire that started in a bakery…LLMLLM
CDA steps x NExpected Response
Output
CLMOptimization
LLM𝒘𝒆𝒂𝒌LLM𝑺𝒕𝒓𝒐𝒏𝒈Preference Reasoning PathsUnexpectedExpected
Figure 2: Overview of our AlignRAG Framework.
RAG, decomposing the reasoning of RAG into
three interdependent phases:
Phase-1 (Relevance Assessment): Failure to pri-
oritize semantically relevant evidence from D, even
under accurate top- kretrieval (Shi et al., 2024).
Phase-2 (Query-Evidence Mapping): Misalign-
ment in connecting queries to implicit or multi-hop
associations (e.g., analogical or causal links) (Wan
et al., 2025).
Phase-3 (Evidence-Integrated Justification):
Logical inconsistencies in synthesizing retrieved
content into coherent justifications (Weng et al.,
2025; Wu et al., 2025b).
This taxonomy highlights that reasoning mis-
alignment persists under ideal retrieval, revealing
a failure mode orthogonal to traditional retrieval
or factual errors. Static prompting methods (Wei
et al., 2022, 2024) fall short in addressing the dy-
namic, evidence-sensitive nature of reasoning in
RAG. To this end, we propose AlignRAG , a novel
test-time framework that enforces evidential align-
ment via critique-guided refinement of reasoning
trajectories—offering a principled solution to this
overlooked yet foundational challenge.3 Method
We present Critique-Driven Alignment (CDA) ,
a novel test-time refinement framework designed
to mitigate reasoning misalignment in retrieval-
augmented generation (RAG). While conventional
RAG pipelines often produce responses that par-
tially or incorrectly reflect retrieved evidence, CDA
introduces an explicit mechanism for identifying,
diagnosing, and revising such failures via a learned
critic model. This section outlines our overall prob-
lem formulation (§3.1), introduces a structured
training methodology for critique learning (§3.2),
and describes our test-time critique-driven align-
ment process (§3.3).
3.1 Problem Setting
Given an input query qand a set of retrieved docu-
mentsD={d1, . . . , d n}, our objective is to refine
an initial response y0=Mgen(q,D)through itera-
tive critique-informed updates from a trained critic
model Mcritic. To support the training of this critic
model, we construct a critique supervision dataset:
S={(qi, ai,Di,ci)}N
i=1,
where each entry contains the query qi, the ground-
truth answer ai, the retrieved documents Di, and
3

a context granularity vector ci= (ri,hi,mi)∈
{0,1}3, capturing orthogonal axes of evidence
quality: relevance ,helpfulness , and completeness .
Each training example includes a response pair
(yexp,yunexp), generated by a strong and a weak
model respectively. We use these to synthesize su-
pervision signals in the form of critiques ∆yunexp,
derived from a preference-augmented input repre-
sentation Xpref= (q,D,yexp,yunexp), which serves
as input for critique generation.
3.2 Critic Training
3.2.1 Training Corpus Construction
To model the ambiguity and diversity inherent in
real-world retrieval scenarios (Yoran et al., 2024;
Fang et al., 2024), we construct a structured train-
ing dataset S={(qi, ai,Di,ci)}N
i=1, where each
instance includes a query qi, its gold answer ai, a
retrieved document set Di, and a context granular-
ity vector ci∈ {0,1}3. The vector encodes three
orthogonal axes of contextual variation:
ci= (ri,hi,mi), (1)
where Relevance riis derived from top- kretrieval
results of a retriever R(qi), with irrelevant docu-
ments sampled from unrelated queries. Helpfulness
hiis annotated based on whether the document con-
tains full, partial, or no answer spans correspond-
ing to ai.Completeness miis a document-set-level
binary label indicating whether Dicollectively sup-
ports the full reasoning required for ai.
To systematically simulate varied degrees of an-
swerability, we define a multiple-tier contextual
granularity hierarchy (Fig. 2.A), exposing critic
models to diverse evidence configurations and en-
abling fine-grained supervision. The details of data
construction could refer to Appendix A.2
3.2.2 Contrastive Critique Synthesis
We propose a novel structured approach for gener-
ating targeted feedback by synthesizing critiques
from paired outputs of LLMs with differing capa-
bilities. We define the task of contrastive critique
synthesis (CCS) as a transformation over model
outputs, conditioned on a preference context and
retrieved evidence. Specifically, given a query q,
we sample two divergent responses: an unexpected
response yunexp from a weaker model Mweakand
anexpected response yexpfrom a stronger model
Mstrong.
To guide critique generation consistently (Zhang
et al., 2025), we define a preference-augmentedinput tuple:
Xpref= (q,D,yexp,yunexp), (2)
which provides the necessary contrastive supervi-
sion signal. This pairwise-path formulation serves
two purposes: (1) it constrains the critic’s genera-
tion space to produce consistent and faithful feed-
back (Zhang et al., 2025), and (2) it enables the
critic to trace reasoning errors with respect to a
reference response.
The learning objective is to generate a structured
critique ∆yunexp that explicitly highlights the defi-
ciencies of yunexp relative to yexp, grounded in re-
trieved context. We formalize this process through
aCritique Function F:
∆yunexp=F(Xpref) =G[Mcritic(Xpref),yexp],
(3)
where Mcriticdenotes a language model trained to
generate critique-aware reasoning trajectories, and
Gis an augmentation operator that reformulates
these reasoning trajectories into constructive im-
provement suggestions. This formulation enables
the critic to both localize and explain misalign-
ments in yunexp, facilitating effective feedback and
downstream model alignment.
3.2.3 Training of Critic LLM
We introduce a novel paradigm for training
language models to produce constructive feed-
back, termed Critique Fine-Tuning (CFT) (Wang
et al., 2025a). The goal is to transform a base
model Mweak into a high-quality critique gener-
atorMcritic, leveraging a synthetic dataset of cri-
tiques C. Each training instance in Cis a tuple
(q,D,yunexp,∆yunexp,yexp), where qdenotes the
input query, Dthe retrieved evidence, yunexp the
incorrect or suboptimal output, ∆yunexp the corre-
sponding critique, and yexpthe improved response.
CFT formulates critique generation as a condi-
tional sequence generation task. The training ob-
jective maximizes the likelihood of the model pro-
ducing the correct critique ∆yunexp, conditioned on
the full critique context Icritic= (q,D,yunexp,yexp).
Formally, the objective is:
LCFT(θ) =−P
Ci∈Clogpθ(∆yunexp| Icritic),
(4)
where pθis the probability distribution defined by
Mcriticparameterized by θ. This formulation en-
ables the model to learn to produce actionable, tar-
geted feedback that can be used to improve down-
stream model outputs.
4

3.3 Critique-Driven Alignment Systems
To address reasoning misalignment in RAG, we
propose a novel framework termed Critique-Driven
Alignment (CDA) . Unlike standard RAG pipelines,
which generate responses in a single forward pass:
y0=Mgen(q,D), (5)
CDA reconceptualizes inference as a discrete-time
optimization process over a latent reasoning space
Y, where each step incrementally improves the
model output using learned critiques.
In this framework, a meta-reasoning module
Mcritic iteratively critiques intermediate genera-
tions and suggests improvements. This yields a
trajectory of refinement:
y0CDA= =⇒y1CDA= =⇒ ···CDA= =⇒yT, (6)
where each transition is guided by a critique-
informed update. At each step t, the critic produces
an edit signal ∆ytidentifying issues and proposing
revisions. The next response is generated via:
yt+1=Mgen(yt⊕∆yt), (7)
where ⊕denotes augmentation of the prompt with
critique feedback. Here, ∆ytfunctions as a pseudo-
gradient in discrete generation space, directing the
model toward more faithful and coherent reason-
ing.
The final output is the terminal state of the
critique-driven trajectory:
yexp=CDA(q,D) := yT. (8)
This framework elevates alignment from static
supervision to a learned iterative process, en-
abling more reliable, interpretable, and evidence-
grounded generation.
4 Related Work
Retrieval-Augmented Generation (RAG) (Hongjin
et al., 2024; Gao et al., 2023b; Borgeaud et al.,
2022; Edge et al., 2024; Fan et al., 2025; Song
et al., 2025) enhances LLMs by grounding gener-
ation in external knowledge. Most work focuses
on improving retrieval (Karpukhin et al., 2020; Xu
et al., 2024; Xiang et al., 2024; Zhong et al., 2023;
Zou et al., 2024) or training generators for better
factuality or reasoning (Jin et al., 2024a; Liu et al.,
2023; Wang et al., 2024). Yet, even with accurateretrieval, a generation often deviates from evidence
due to misaligned reasoning (Wu et al., 2025b).
To reduce such errors, recent methods filter noisy
context (Gupta et al., 2024; Sarthi et al., 2024; Yang
et al., 2024b; Wu et al., 2025a; Lee et al., 2025)
or incorporate CoT reasoning. InstructRAG (Wei
et al., 2024) adds self-supervised objectives to im-
prove coherence, but these approaches rely on static
training and overlook test-time error propagation.
We identify reasoning misalignment as a core
failure mode in RAG. Prior solutions like Self-
RAG (Asai et al., 2023) require retraining or archi-
tectural changes. In contrast, we introduce Critique-
Driven Alignment (CDA), a novel test-time method
that dynamically realigns reasoning with evidence
without modifying the base model. CDA offers a
lightweight and deployable alternative to training-
heavy RAG variants. CDA draws inspiration from
self-refinement (Madaan et al., 2023; Zhang et al.,
2024a), but departs by grounding critiques in re-
trieval. Unlike external verifiers (Hosseini et al.,
2024; Sun et al., 2025), which operate as post-hoc
selection, CDA uses a dedicated Critic Language
Model (CLM) to detect and revise reasoning er-
rors based on retrieved evidence. This decoupled
architecture preserves the generator’s parametric
knowledge while explicitly modeling reasoning-
evidence alignment, pushing RAG toward more
robust and controllable inference.
5 Experiments
5.1 Experiment Setup
We evaluate our method using three instruction-
tuned backbones: Qwen2.5-7B-Instruct (Yang
et al., 2024a), Qwen2.5-14B-Instruct (Yang et al.,
2024a), and LLaMA3.1-8B-Instruct (Grattafiori
et al., 2024). For simplicity, we refer to them as
Qwen2.5-7B, Qwen2.5-14B, and LLaMA3.1-8B.
Dataset. To train a strong critique genera-
tor, we construct a dataset of 10K by sam-
pling 2K instances from each of five benchmarks:
PopQA (Mallen et al., 2023), TriviaQA (Joshi
et al., 2017), NaturalQuestions (Kwiatkowski et al.,
2019), 2WikiMultihopQA (Ho et al., 2020), and
ASQA (Stelmakh et al., 2022). Furthermore, we
evaluate our method on the same five in-domain
benchmarks, along with two out-of-distribution
(OOD) tasks, i.e., HotpotQA (Yang et al., 2018)
and SQuAD (Rajpurkar et al., 2016).
Baselines. In our experiments, we compare
our method against a range of non-retrieval and
5

retrieval-based baselines. For non-retrieval base-
lines, we include Chain-of-Thought (CoT) prompt-
ing (Wei et al., 2022) applied to instruction-tuned
models without retrieval augmentation. For stan-
dard RAG, we report performance from Vanilla
Reasoning (Wei et al., 2024), which performs step-
by-step answer generation based on retrieved pas-
sages. To assess the benefits of intermediate super-
vision, we include training-time refinement base-
lines such as RetRobust(Yoran et al., 2024) and
InstructRAG(Wei et al., 2024). Our main compar-
ison is with these test-time refinement methods,
as they share similar objectives. For test-time re-
finement, we evaluate Self-RAG (Asai et al., 2023)
and Self-Refinement, which iteratively revises out-
puts based on self-generated critique.
Evaluation metrics. Following prior work (Gao
et al., 2023a), we adopt the official correctness
metric ( str-em ) for ASQA, and use accuracy for
the other tasks, which measures whether the final
generations of the model align with the ground-
truth (Mallen et al., 2023; Schick et al., 2023).
Implementation Details. For the CLM, we adopt
LLaMA3.1-8B as the backbone and fine-tune it
using LoRA for parameter-efficient training. More-
over, the strong model we use to generate ex-
pected reasoning trajectories is LLaMA-72B, and
the weak model we use to generate unexpected rea-
soning trajectories is LLaMA3.1-8B. We set the
retrieval Top- Kto 5 for each question.
5.2 Main Result
Table 1 shows the overall performance of our
method and baselines in various families and sizes
of the base model across five benchmarks.
First, compared to non-retrieval baselines such
as Chain-of-Thought (CoT) prompting, all retrieval-
augmented methods achieve significantly better per-
formance, demonstrating the importance of incor-
porating relevant external knowledge. Second, we
observe further gains when applying training-time
refinement methods. In particular, InstructRAG
achieves strong performance across all backbones,
outperforming Vanilla RAG by a large margin, con-
firming the value of training refinement strategies.
Notably, AlignRAG achieves the best overall re-
sults on all three backbones compared to other test-
time refinement methods. It surpasses Self-RAG
and Self-Refinement by notable margins, achieving
an average accuracy of 63.1 compared to 48.1 and
60.7, respectively. The performance improvement
is consistent across all benchmarks, highlightingboth the effectiveness and generalization capabil-
ity of our approach. This demonstrates that our
critique-driven alignment strategy can better guide
the reasoning process and overcome the limitations
of purely self-generated feedback.
5.3 Robustness under Noisy and Informative
Retrieval Conditions
To assess the robustness of different methods under
varying retrieval quality, we evaluate performance
in two scenarios: when the retrieved documents
contain the correct answer (Answerable) and when
they do not (Unanswerable) .
Figure 3a and Figure 3b present the results under
these two retrieval scenarios. In the Unanswer-
able setting, AlignRAG consistently outperforms
Self-Refine and Vanilla RAG, demonstrating its
ability to mitigate reasoning misalignment under
noisy retrieval. In the Answerable setting, while
all methods improve with relevant evidence, Align-
RAG still leads by a clear margin, indicating en-
hanced reasoning fidelity even when retrieval is
reliable. These results demonstrate that our method
improves reasoning robustness across both noisy
and informative retrieval conditions.
/uni00000033/uni00000052/uni00000053/uni00000034/uni00000024
/uni00000037/uni00000055/uni0000004c/uni00000059/uni0000004c/uni00000044/uni00000034/uni00000024
/uni00000031/uni00000044/uni00000057/uni00000058/uni00000055/uni00000044/uni0000004f/uni00000034/uni00000058/uni00000048/uni00000056/uni00000057/uni0000004c/uni00000052/uni00000051/uni00000056 /uni00000015/uni0000003a/uni0000004c/uni0000004e/uni0000004c/uni00000030/uni00000058/uni0000004f/uni00000057/uni0000004c/uni0000002b/uni00000052/uni00000053/uni00000034/uni00000024/uni00000024/uni00000036/uni00000034/uni00000024/uni00000018/uni00000011/uni0000001a
/uni00000015/uni00000015/uni00000011/uni0000001b
/uni00000014/uni00000019/uni00000011/uni00000019
/uni00000016/uni0000001b/uni00000011/uni0000001b/uni00000016/uni00000019/uni00000011/uni0000001a/uni00000017/uni00000011/uni0000001c
/uni00000014/uni0000001a/uni00000011/uni00000014
/uni0000001c/uni00000011/uni0000001a
/uni00000016/uni00000016/uni00000011/uni00000014/uni00000016/uni00000017/uni00000011/uni00000017/uni00000017/uni00000011/uni00000018
/uni00000014/uni00000018/uni00000011/uni00000016
/uni0000001b/uni00000011/uni00000016
/uni00000016/uni00000016/uni00000011/uni00000015/uni00000016/uni00000016/uni00000011/uni00000015/uni00000032/uni00000058/uni00000055/uni00000056
/uni00000036/uni00000048/uni0000004f/uni00000049/uni00000010/uni00000035/uni00000048/uni00000049/uni0000004c/uni00000051/uni00000048
/uni00000039/uni00000044/uni00000051/uni0000004c/uni0000004f/uni0000004f/uni00000044/uni00000003/uni00000035/uni00000024/uni0000002a
(a) w/o answer.
/uni00000033/uni00000052/uni00000053/uni00000034/uni00000024
/uni00000037/uni00000055/uni0000004c/uni00000059/uni0000004c/uni00000044/uni00000034/uni00000024
/uni00000031/uni00000044/uni00000057/uni00000058/uni00000055/uni00000044/uni0000004f/uni00000034/uni00000058/uni00000048/uni00000056/uni00000057/uni0000004c/uni00000052/uni00000051/uni00000056 /uni00000015/uni0000003a/uni0000004c/uni0000004e/uni0000004c/uni00000030/uni00000058/uni0000004f/uni00000057/uni0000004c/uni0000002b/uni00000052/uni00000053/uni00000034/uni00000024/uni00000024/uni00000036/uni00000034/uni00000024/uni0000001c/uni00000015/uni00000011/uni00000019
/uni0000001c/uni00000019/uni00000011/uni00000015
/uni0000001b/uni0000001c/uni00000011/uni0000001b/uni0000001b/uni00000014/uni00000011/uni00000016/uni0000001b/uni00000019/uni00000011/uni00000016/uni0000001c/uni00000015/uni00000011/uni00000019
/uni0000001c/uni00000017/uni00000011/uni0000001b
/uni0000001b/uni00000019/uni00000011/uni0000001b/uni0000001a/uni00000019/uni00000011/uni00000019/uni0000001b/uni00000013/uni00000011/uni00000014/uni0000001c/uni00000013/uni00000011/uni0000001b
/uni0000001c/uni00000017/uni00000011/uni00000016
/uni0000001b/uni00000019/uni00000011/uni00000015 /uni0000001a/uni00000019/uni00000011/uni00000019/uni0000001a/uni0000001b/uni00000011/uni0000001c/uni00000032/uni00000058/uni00000055/uni00000056
/uni00000036/uni00000048/uni0000004f/uni00000049/uni00000010/uni00000035/uni00000048/uni00000049/uni0000004c/uni00000051/uni00000048
/uni00000039/uni00000044/uni00000051/uni0000004c/uni0000004f/uni0000004f/uni00000044/uni00000003/uni00000035/uni00000024/uni0000002a (b) w/ answer.
Figure 3: Performance of different methods under Unan-
swerable (a) and Answerable (b) retrieval conditions.
Each radar chart reports the average performance across
three instruction-tuned backbones on five benchmarks.
5.4 Test-time Scalability via Iterative
Alignment
To assess test-time scalability, we plot accuracy
over five refinement steps across seven benchmarks,
comparing AlignRAG with Self-Refine (Figure 4).
The curves reveal two notable trends. First, both
methods generally benefit from iterative alignment,
with accuracy improving on most tasks as the num-
ber of refinement steps increases. This indicates
that reasoning can scale with additional refinement
steps. However, we occasionally observe slight
6

Method NQ MultiHopQA TriviaQA PopQA ASQAAvg.Metric accuracy accuracy accuracy accuracy str-em
Baselines w/o Retrieval
Chain-of-thought (Wei et al., 2022)
Qwen-2.5-Instruct 7B 33.9 45.0 58.3 26.9 20.5 36.9
Qwen-2.5-Instruct 14B 48.1 49.3 72.8 25.4 31.6 45.4
Llama-3.1-Instruct 8B 42.1 41.9 61.8 26.9 25.1 40.0
Standard RAG with Reasoning
Vanilla Reasoning
Qwen-2.5-Instruct 7B 60.2 44.7 73.2 63.7 42.8 56.9
Qwen-2.5-Instruct 14B 63.6 44.8 77.0 65.3 45.2 59.2
Llama-3.1-Instruct 8B 62.0 43.0 73.4 65.0 45.2 57.7
RAG w/ Training-time Refinement
RetRobust (Yoran et al., 2024)
Llama-2 13B* 39.6 51.5 – – – –
Llama-3-Instruct 8B* 54.2 54.7 71.5 56.5 40.5 55.5
InstructRAG (Wei et al., 2024)
Qwen-2.5-Instruct 7B 63.8 46.3 76.1 67.5 47.5 60.2
Qwen-2.5-Instruct 14B 66.3 47.3 78.7 67.8 48.5 61.7
Llama-3.1-Instruct 8B 66.3 45.1 76.6 66.9 47.2 60.4
RAG w/ Test-time Refinement
Self-RAG (Asai et al., 2023)
Llama-2 7B+ CLM 7B* 42.4 35.9 68.9 55.8 30.0 46.6
Llama-2 13B+ CLM 13B* 46.4 36.0 70.4 56.3 31.4 48.1
Llama-3-Instruct 8B+ CLM 8B* 42.8 32.9 71.4 55.8 36.9 48.0
Self-Refinement
Qwen-2.5-Instruct 7B+ SELF 7B 61.6(∆) 45.0( ∆) 74.4( ∆) 65.5( ∆) 45.2( ∆) 58.3( ∆)
Qwen-2.5-Instruct 14B+ SELF 14B 65.1(∆) 46.1( ∆) 78.0( ∆) 67.0(∆) 47.3( ∆) 60.7( ∆)
Llama-3.1-Instruct 8B+ SELF 8B 61.4(∆) 42.8( ∆) 74.1( ∆) 66.1( ∆) 44.7( ∆) 57.8( ∆)
AlignRAG
Qwen-2.5-Instruct 7B+ CLM 8B 66.4 (↑4.8%) 49.9 ( ↑4.9%) 77.5 ( ↑3.1%) 66.0 ( ↑0.5%) 48.6 ( ↑3.4%) 61.7 ( ↑3.4%)
Qwen-2.5-Instruct 14B+ CLM 8B 68.6 (↑3.5%) 50.7 ( ↑4.6%) 79.4 ( ↑1.4%) 66.9 ( ↓0.1%) 49.9 ( ↑2.6%) 63.1 ( ↑2.4%)
Llama-3.1-Instruct 8B+ CLM 8B 66.3 (↑1.9%) 49.6 ( ↑6.8%) 77.0 ( ↑2.9%) 66.6 ( ↑0.5%) 48.2 ( ↑3.5%) 61.5 ( ↑3.7%)
Table 1: Performance comparison of Retrieval-Augmented Generation (RAG) systems employing various knowl-
edge refinement strategies and reasoning configurations across five question-answering (QA) benchmarks. To ensure
a fair evaluation, all systems are tested under a single-iteration test-time refinement setting. Results marked with
* are reproduced from (Wei et al., 2024). Missing results in the original paper are denoted by “–”. To highlight
our method’s impact of different model backbones , we use the following color-coded notation for performance
improvements: ( ∆) represents the Qwen-2.5-Instruct 7B, (∆) represents the Qwen-2.5-Instruct 14B, and ( ∆) represents
the Llama-3-Instruct 8B. These results demonstrate the effectiveness of incorporating novel refinement strategies into
RAG pipelines for improving the robustness and accuracy of QA tasks.
/uni00000014 /uni00000015 /uni00000016 /uni00000017 /uni00000018
/uni00000031/uni00000058/uni00000050/uni00000045/uni00000048/uni00000055/uni00000003/uni00000052/uni00000049/uni00000003/uni0000002c/uni00000057/uni00000048/uni00000055/uni00000044/uni00000057/uni0000004c/uni00000052/uni00000051/uni00000056/uni00000019/uni00000019/uni00000011/uni00000013/uni00000019/uni00000019/uni00000011/uni00000017/uni00000019/uni00000019/uni00000011/uni0000001b/uni00000019/uni0000001a/uni00000011/uni00000015/uni00000019/uni0000001a/uni00000011/uni00000019/uni00000019/uni0000001b/uni00000011/uni00000013/uni00000024/uni00000046/uni00000046/uni00000058/uni00000055/uni00000044/uni00000046/uni0000005c
/uni00000033/uni00000052/uni00000053/uni00000034/uni00000024
/uni00000014 /uni00000015 /uni00000016 /uni00000017 /uni00000018
/uni00000031/uni00000058/uni00000050/uni00000045/uni00000048/uni00000055/uni00000003/uni00000052/uni00000049/uni00000003/uni0000002c/uni00000057/uni00000048/uni00000055/uni00000044/uni00000057/uni0000004c/uni00000052/uni00000051/uni00000056/uni0000001a/uni00000018/uni00000011/uni00000013/uni0000001a/uni00000019/uni00000011/uni00000013/uni0000001a/uni0000001a/uni00000011/uni00000013/uni0000001a/uni0000001b/uni00000011/uni00000013/uni0000001a/uni0000001c/uni00000011/uni00000013/uni0000001b/uni00000013/uni00000011/uni00000013
/uni00000037/uni00000055/uni0000004c/uni00000059/uni0000004c/uni00000044/uni00000034/uni00000024
/uni00000014 /uni00000015 /uni00000016 /uni00000017 /uni00000018
/uni00000031/uni00000058/uni00000050/uni00000045/uni00000048/uni00000055/uni00000003/uni00000052/uni00000049/uni00000003/uni0000002c/uni00000057/uni00000048/uni00000055/uni00000044/uni00000057/uni0000004c/uni00000052/uni00000051/uni00000056/uni00000019/uni00000015/uni00000011/uni00000013/uni00000019/uni00000016/uni00000011/uni00000019/uni00000019/uni00000018/uni00000011/uni00000015/uni00000019/uni00000019/uni00000011/uni0000001b/uni00000019/uni0000001b/uni00000011/uni00000017/uni0000001a/uni00000013/uni00000011/uni00000013
/uni00000031/uni00000044/uni00000057/uni00000058/uni00000055/uni00000044/uni0000004f/uni00000034/uni00000058/uni00000048/uni00000056/uni00000057/uni0000004c/uni00000052/uni00000051/uni00000056
/uni00000014 /uni00000015 /uni00000016 /uni00000017 /uni00000018
/uni00000031/uni00000058/uni00000050/uni00000045/uni00000048/uni00000055/uni00000003/uni00000052/uni00000049/uni00000003/uni0000002c/uni00000057/uni00000048/uni00000055/uni00000044/uni00000057/uni0000004c/uni00000052/uni00000051/uni00000056/uni00000017/uni00000016/uni00000011/uni00000013/uni00000017/uni00000018/uni00000011/uni00000013/uni00000017/uni0000001a/uni00000011/uni00000013/uni00000017/uni0000001c/uni00000011/uni00000013/uni00000018/uni00000014/uni00000011/uni00000013/uni00000018/uni00000016/uni00000011/uni00000013
/uni00000015/uni0000003a/uni0000004c/uni0000004e/uni0000004c/uni00000030/uni00000058/uni0000004f/uni00000057/uni0000004c/uni0000002b/uni00000052/uni00000053/uni00000034/uni00000024
/uni00000014 /uni00000015 /uni00000016 /uni00000017 /uni00000018
/uni00000031/uni00000058/uni00000050/uni00000045/uni00000048/uni00000055/uni00000003/uni00000052/uni00000049/uni00000003/uni0000002c/uni00000057/uni00000048/uni00000055/uni00000044/uni00000057/uni0000004c/uni00000052/uni00000051/uni00000056/uni00000017/uni00000018/uni00000011/uni00000013/uni00000017/uni00000019/uni00000011/uni00000015/uni00000017/uni0000001a/uni00000011/uni00000017/uni00000017/uni0000001b/uni00000011/uni00000019/uni00000017/uni0000001c/uni00000011/uni0000001b/uni00000018/uni00000014/uni00000011/uni00000013
/uni00000024/uni00000036/uni00000034/uni00000024
/uni00000014 /uni00000015 /uni00000016 /uni00000017 /uni00000018
/uni00000031/uni00000058/uni00000050/uni00000045/uni00000048/uni00000055/uni00000003/uni00000052/uni00000049/uni00000003/uni0000002c/uni00000057/uni00000048/uni00000055/uni00000044/uni00000057/uni0000004c/uni00000052/uni00000051/uni00000056/uni00000014/uni0000001b/uni00000011/uni00000013/uni00000015/uni00000015/uni00000011/uni00000013/uni00000015/uni00000019/uni00000011/uni00000013/uni00000016/uni00000013/uni00000011/uni00000013/uni00000016/uni00000017/uni00000011/uni00000013/uni00000016/uni0000001b/uni00000011/uni00000013
/uni0000002b/uni00000052/uni00000057/uni00000053/uni00000052/uni00000057/uni00000034/uni00000024
/uni00000014 /uni00000015 /uni00000016 /uni00000017 /uni00000018
/uni00000031/uni00000058/uni00000050/uni00000045/uni00000048/uni00000055/uni00000003/uni00000052/uni00000049/uni00000003/uni0000002c/uni00000057/uni00000048/uni00000055/uni00000044/uni00000057/uni0000004c/uni00000052/uni00000051/uni00000056/uni00000014/uni00000013/uni00000011/uni00000013/uni00000014/uni00000017/uni00000011/uni00000013/uni00000014/uni0000001b/uni00000011/uni00000013/uni00000015/uni00000015/uni00000011/uni00000013/uni00000015/uni00000019/uni00000011/uni00000013/uni00000016/uni00000013/uni00000011/uni00000013
/uni00000036/uni00000034/uni00000058/uni00000024/uni00000027/uni00000024/uni0000004f/uni0000004c/uni0000004a/uni00000051/uni00000035/uni00000024/uni0000002a /uni00000036/uni00000048/uni0000004f/uni00000049/uni00000010/uni00000035/uni00000048/uni00000049/uni0000004c/uni00000051/uni00000048
Figure 4: Performance of AlignRAG and Self-Refine across five refinement iterations on seven benchmarks.
degradation beyond a certain point, which we at-
tribute to potential noise accumulation or overcor-
rection during excessive iterations. Second, Align-
RAG consistently outperforms Self-Refine across
all iterations and benchmarks with notable mar-
gins. These findings demonstrate that AlignRAG
not only enables scalable reasoning but also pro-
vides more stable and robust improvements.
5.5 Integrate as a Plug-and-play Module into
Existing RAG Pipelines
To evaluate the generality and plug-and-play nature
of our method, we integrate it into the InstructRAG
framework across three backbones. Table 2 reportsthe performance under both In-Domain (ID) and
Out-of-Domain (OOD) evaluation.
We observe consistent improvements in both fa-
miliar and unseen distributions. The alignment-
enhanced variant significantly outperforms the orig-
inal InstructRAG (Wei et al., 2024), demonstrating
that our method can be seamlessly incorporated
into existing RAG pipelines in a zero-modification,
test-time manner—highlighting its strong compati-
bility and practical utility.
5.6 Generalization to OOD Scenarios
To assess the generalization capability of our
method beyond the domains seen during training,
7

Method ID (avg.) OOD (avg.) Avg.
Qwen2.5-7B
InstructRAG 59.5( ∆) 28.0( ∆) 43.8( ∆)
w/ Alignment 63.0( ↑3.5%) 31.7( ↑3.7%) 47.4( ↑3.6%)
Qwen2.5-14B
InstructRAG 61.7( ∆) 24.9( ∆) 43.3( ∆)
w/ Alignment 63.9( ↑2.2%) 34.3 ( ↑9.4%) 49.1( ↑5.8%)
LLaMA3.1-8B
InstructRAG 60.4( ∆) 28.4( ∆) 44.4( ∆)
w/ Alignment 61.9( ↑1.5%) 30.5( ↑2.1%) 46.2( ↑1.8%)
Table 2: Combination of Training-time (InstructRAG)
and test-time alignment (AlignRAG). Results are evalu-
ated under both In-Domain (ID) and Out-of-Distribution
(OOD) settings.
we conduct out-of-distribution (OOD) evaluations
on two held-out benchmarks, i.e., HotpotQA (Yang
et al., 2018) and SQuAD (Rajpurkar et al., 2016).
As shown in Figure 5, we compare AlignRAG
with two baselines: a standard Vanilla RAG sys-
tem, and Self-Refinement . AlignRAG consistently
achieves the lowest performance drop across all
backbones, outperforming both baselines by a sig-
nificant margin. For instance, on LLaMA3.1-8B,
AlignRAG reduces the performance drop to 23.4%
compared to 29.4% for Self-Refinement. These
results demonstrate that our CDA mechanism not
only improves in-domain reasoning but also en-
hances robustness under domain shift.
/uni0000002f/uni0000002f/uni00000044/uni00000030/uni00000024/uni00000016/uni00000011/uni00000014/uni00000010/uni0000001b/uni00000025 /uni00000034/uni0000005a/uni00000048/uni00000051/uni00000015/uni00000011/uni00000018/uni00000010/uni0000001a/uni00000025 /uni00000034/uni0000005a/uni00000048/uni00000051/uni00000015/uni00000011/uni00000018/uni00000010/uni00000014/uni00000017/uni00000025/uni00000015/uni00000013/uni00000015/uni00000015/uni00000015/uni00000017/uni00000015/uni00000019/uni00000015/uni0000001b/uni00000016/uni00000013/uni00000016/uni00000015/uni00000033/uni00000048/uni00000055/uni00000049/uni00000052/uni00000055/uni00000050/uni00000044/uni00000051/uni00000046/uni00000048/uni00000003/uni00000027/uni00000055/uni00000052/uni00000053/uni00000003/uni0000000b/uni00000008/uni0000000c
/uni00000039/uni00000044/uni00000051/uni0000004c/uni0000004f/uni0000004f/uni00000044/uni00000003/uni00000035/uni00000024/uni0000002a
/uni00000036/uni00000048/uni0000004f/uni00000049/uni00000010/uni00000035/uni00000048/uni00000049/uni0000004c/uni00000051/uni00000048
/uni00000032/uni00000058/uni00000055/uni00000056
Figure 5: Drop in average OOD performance com-
pared to average In-Domain performance across three
instruction-tuned backbones. Lower values indicate bet-
ter generalization capability.
5.7 Ablation Study
In this section, we conduct an ablation study to
evaluate the impact of the Critic Language Model
(CLM), a key component of our framework. We
compare four configurations: (1) Vanilla RAG,
which performs generation without any critique;
(2) RAG with Frozen CLM, which uses a CLM
based on LLaMA3.1-8B without any training; (3)
RAG with Trained CLM; (4) RAG with Trained
CLM using our CCS, where the CLM is based
on LLaMA3.1-8B and further optimized via our
contrastive critique synthesis (CCS) strategy.Method ID (avg.) OOD (avg.) Avg.
Vanilla RAG
Qwen2.5-7B 56.9 13.8 35.4
Qwen2.5-14B 59.2 18.0 38.6
LLaMA3.1-8B 57.7 11.6 34.7
w/ Frozen CLM
Qwen2.5-7B 59.4 17.0 38.2
Qwen2.5-14B 61.2 20.5 40.9
LLaMA3.1-8B 57.8 13.8 35.8
w/ Trained CLM
Qwen2.5-7B 60.4( ∆) 21.6( ∆) 41.0( ∆)
Qwen2.5-14B 61.9( ∆) 25.4( ∆) 43.7( ∆)
LLaMA3.1-8B 59.5( ∆) 18.1( ∆) 38.8( ∆)
w/ Trained CLM, w/ CCS (Ours)
Qwen2.5-7B 61.7( ↑1.3%) 29.1( ↑7.5%) 45.4( ↑4.4%)
Qwen2.5-14B 63.1( ↑1.2%) 30.5( ↑5.1%) 46.8( ↑3.1%)
LLaMA3.1-8B 61.5( ↑2.0%) 26.4( ↑8.3%) 44.0( ↑5.2%)
Table 3: Ablation study on the CLM training. Frozen
CLM refers to a vanilla LLaMA3.1-8B used as the critic.
CCS refers to our proposed contrastive synthesis.
As shown in Table 3, incorporating even an un-
trained CLM leads to consistent gains over the
vanilla RAG baseline, indicating that an auxiliary
critic can already provide useful feedback during
inference. Further improvements are achieved by
training the CLM through critical fine-tuning, con-
firming the importance of eliciting the capability
of generating retrieval-aware critiques. Finally, our
proposed CCS training brings the most substantial
gains, especially in OOD scenarios. For instance,
on Qwen2.5-14B, CCS improves the OOD perfor-
mance from 25.4% to 30.5%, and overall average
accuracy by 3.1 points. These results highlight both
the effectiveness of the CLM and the benefit of our
contrastive alignment strategy.
6 Conclusion
We present AlignRAG, a novel framework that
addresses reasoning misalignment in retrieval-
augmented generation through dynamic critique-
driven optimization. By reconceptualizing RAG
as a structured reasoning trajectory, our method
mitigates error propagation across relevance as-
sessment, evidence mapping, and justification syn-
thesis via iterative CDA steps. Extensive experi-
ments across seven benchmarks and three model
families demonstrate AlignRAG’s state-of-the-art
performance. The framework’s robustness is ev-
idenced by lower OOD performance degradation
compared to baselines and its seamless integration
as a drop-in module, improving InstructRAG’s ac-
curacy by 5.8% on Qwen-2.5-14B without archi-
tectural changes. Our findings highlight the critical
role of retrieval-aware reasoning alignment, open-
ing new directions for building RAG pipelines.
8

Limitations
Although AlignRAG marks a substantial advance-
ment in retrieval-augmented reasoning by incorpo-
rating critique-guided optimization, it is not with-
out its limitations. A primary concern is its reliance
on a Critic Language Model (CLM), which can in-
herit biases inherent in the LLMs used to generate
synthetic critique training data. This dependency
may lead to the amplification of systemic biases
or inaccuracies, particularly when identifying nu-
anced or domain-specific misalignments. Addition-
ally, the process of generating synthetic critiques,
while innovative, is limited by the diversity and
representativeness of the training data. Insufficient
diversity in these synthetic critiques may reduce
the CLM’s robustness and its ability to generalize
effectively to unseen tasks or edge cases. Future
research should address these limitations by explor-
ing techniques to enhance critique diversity, such
as adversarial data augmentation or leveraging real-
world feedback loops. Despite these challenges, the
AlignRAG framework lays a strong foundation for
advancing RAG reasoning, providing a promising
direction for further research in this domain.
References
Akari Asai, Zeqiu Wu, Yizhong Wang, Avirup Sil, and
Hannaneh Hajishirzi. 2023. Self-RAG: Learning to
retrieve, generate, and critique through self-reflection.
InThe Twelfth International Conference on Learning
Representations .
Sebastian Borgeaud, Arthur Mensch, Jordan Hoff-
mann, Trevor Cai, Eliza Rutherford, Katie Milli-
can, George Bm Van Den Driessche, Jean-Baptiste
Lespiau, Bogdan Damoc, Aidan Clark, and 1 others.
2022. Improving language models by retrieving from
trillions of tokens. In International conference on
machine learning , pages 2206–2240. PMLR.
Qiguang Chen, Libo Qin, Jinhao Liu, Dengyun Peng,
Jiannan Guan, Peng Wang, Mengkang Hu, Yuhang
Zhou, Te Gao, and Wangxiang Che. 2025. Towards
reasoning era: A survey of long chain-of-thought
for reasoning large language models. arXiv preprint
arXiv:2503.09567 .
Darren Edge, Ha Trinh, Newman Cheng, Joshua
Bradley, Alex Chao, Apurva Mody, Steven Truitt,
Dasha Metropolitansky, Robert Osazuwa Ness, and
Jonathan Larson. 2024. From local to global: A
graph rag approach to query-focused summarization.
arXiv preprint arXiv:2404.16130 .
Tianyu Fan, Jingyuan Wang, Xubin Ren, and Chao
Huang. 2025. Minirag: Towards extremely sim-ple retrieval-augmented generation. arXiv preprint
arXiv:2501.06713 .
Feiteng Fang, Yuelin Bai, Shiwen Ni, Min Yang, Xiao-
jun Chen, and Ruifeng Xu. 2024. Enhancing noise
robustness of retrieval-augmented language models
with adaptive adversarial training. arXiv preprint
arXiv:2405.20978 .
Tianyu Gao, Howard Yen, Jiatong Yu, and Danqi Chen.
2023a. Enabling large language models to generate
text with citations. In Proceedings of the 2023 Con-
ference on Empirical Methods in Natural Language
Processing , pages 6465–6488.
Yunfan Gao, Yun Xiong, Xinyu Gao, Kangxiang Jia,
Jinliu Pan, Yuxi Bi, Yi Dai, Jiawei Sun, and Haofen
Wang. 2023b. Retrieval-augmented generation for
large language models: A survey. arXiv preprint
arXiv:2312.10997 .
Aaron Grattafiori, Abhimanyu Dubey, Abhinav Jauhri,
Abhinav Pandey, Abhishek Kadian, Ahmad Al-
Dahle, Aiesha Letman, Akhil Mathur, Alan Schelten,
Alex Vaughan, and 1 others. 2024. The llama 3 herd
of models. arXiv preprint arXiv:2407.21783 .
Aman Gupta, Anup Shirgaonkar, Angels de Luis Bal-
aguer, Bruno Silva, Daniel Holstein, Dawei Li, Jen-
nifer Marsman, Leonardo O Nunes, Mahsa Rouzbah-
man, Morris Sharp, and 1 others. 2024. RAG vs
Fine-tuning: Pipelines, tradeoffs, and a case study on
agriculture. arXiv preprint arXiv:2401.08406 .
Kelvin Guu, Kenton Lee, Zora Tung, Panupong Pasu-
pat, and Mingwei Chang. 2020. Retrieval augmented
language model pre-training. In International confer-
ence on machine learning , pages 3929–3938. PMLR.
Xanh Ho, Anh-Khoa Duong Nguyen, Saku Sugawara,
and Akiko Aizawa. 2020. Constructing a multi-hop
qa dataset for comprehensive evaluation of reasoning
steps. arXiv preprint arXiv:2011.01060 .
SU Hongjin, Howard Yen, Mengzhou Xia, Weijia Shi,
Niklas Muennighoff, Han-yu Wang, Liu Haisu, Quan
Shi, Zachary S Siegel, Michael Tang, and 1 others.
2024. Bright: A realistic and challenging benchmark
for reasoning-intensive retrieval. In The Thirteenth
International Conference on Learning Representa-
tions .
Arian Hosseini, Xingdi Yuan, Nikolay Malkin, Aaron
Courville, Alessandro Sordoni, and Rishabh Agar-
wal. 2024. V-star: Training verifiers for self-taught
reasoners. arXiv preprint arXiv:2402.06457 .
Gautier Izacard, Patrick Lewis, Maria Lomeli, Lucas
Hosseini, Fabio Petroni, Timo Schick, Jane Dwivedi-
Yu, Armand Joulin, Sebastian Riedel, and Edouard
Grave. 2023. Atlas: Few-shot learning with retrieval
augmented language models. Journal of Machine
Learning Research , 24(251):1–43.
9

Bowen Jin, Hansi Zeng, Zhenrui Yue, Dong Wang,
Hamed Zamani, and Jiawei Han. 2025. Search-
r1: Training llms to reason and leverage search en-
gines with reinforcement learning. arXiv preprint
arXiv:2503.09516 .
Chao Jin, Zili Zhang, Xuanlin Jiang, Fangyue Liu, Xin
Liu, Xuanzhe Liu, and Xin Jin. 2024a. RAGCache:
Efficient knowledge caching for retrieval-augmented
generation. arXiv preprint arXiv:2404.12457 .
Jiajie Jin, Yutao Zhu, Xinyu Yang, Chenghao Zhang,
and Zhicheng Dou. 2024b. FlashRAG: A modular
toolkit for efficient retrieval-augmented generation
research. arXiv preprint arXiv:2405.13576 .
Mandar Joshi, Eunsol Choi, Daniel S Weld, and Luke
Zettlemoyer. 2017. TriviaQA: A large scale distantly
supervised challenge dataset for reading comprehen-
sion. In Proceedings of the 55th Annual Meeting of
the Association for Computational Linguistics (Vol-
ume 1: Long Papers) , pages 1601–1611.
Vladimir Karpukhin, Barlas Oguz, Sewon Min, Patrick
Lewis, Ledell Wu, Sergey Edunov, Danqi Chen, and
Wen-tau Yih. 2020. Dense passage retrieval for open-
domain question answering. In Proceedings of the
2020 Conference on Empirical Methods in Natural
Language Processing (EMNLP) , pages 6769–6781.
Jungo Kasai, Keisuke Sakaguchi, Ronan Le Bras, Akari
Asai, Xinyan Yu, Dragomir Radev, Noah A Smith,
Yejin Choi, Kentaro Inui, and 1 others. 2024. Real-
Time QA: What’s the answer right now? Advances
in Neural Information Processing Systems , 36.
Tom Kwiatkowski, Jennimaria Palomaki, Olivia Red-
field, Michael Collins, Ankur Parikh, Chris Alberti,
Danielle Epstein, Illia Polosukhin, Jacob Devlin, Ken-
ton Lee, and 1 others. 2019. Natural questions: a
benchmark for question answering research. Trans-
actions of the Association for Computational Linguis-
tics, 7:453–466.
Zhicheng Lee, Shulin Cao, Jinxin Liu, Jiajie Zhang,
Weichuan Liu, Xiaoyin Che, Lei Hou, and Juanzi
Li. 2025. Rearag: Knowledge-guided reasoning en-
hances factuality of large reasoning models with iter-
ative retrieval augmented generation. arXiv preprint
arXiv:2503.21729 .
Patrick Lewis, Ethan Perez, Aleksandra Piktus, Fabio
Petroni, Vladimir Karpukhin, Naman Goyal, Hein-
rich Küttler, Mike Lewis, Wen-tau Yih, Tim Rock-
täschel, and 1 others. 2020a. Retrieval-augmented
generation for knowledge-intensive NLP tasks. Ad-
vances in Neural Information Processing Systems ,
33:9459–9474.
Patrick S. H. Lewis, Ethan Perez, Aleksandra Pik-
tus, Fabio Petroni, Vladimir Karpukhin, Naman
Goyal, Heinrich Küttler, Mike Lewis, Wen-tau Yih,
Tim Rocktäschel, Sebastian Riedel, and Douwe
Kiela. 2020b. Retrieval-augmented generation for
knowledge-intensive NLP tasks. In Proceedings of
NeurIPS .Yuhan Liu, Hanchen Li, Kuntai Du, Jiayi Yao, Yihua
Cheng, Yuyang Huang, Shan Lu, Michael Maire,
Henry Hoffmann, Ari Holtzman, and 1 others. 2023.
CacheGen: Fast context loading for language model
applications. arXiv preprint arXiv:2310.07240 .
Aman Madaan, Niket Tandon, Prakhar Gupta, Skyler
Hallinan, Luyu Gao, Sarah Wiegreffe, Uri Alon,
Nouha Dziri, Shrimai Prabhumoye, Yiming Yang,
and 1 others. 2023. Self-refine: Iterative refinement
with self-feedback. Advances in Neural Information
Processing Systems , 36:46534–46594.
Alex Mallen, Akari Asai, Victor Zhong, Rajarshi Das,
Daniel Khashabi, and Hannaneh Hajishirzi. 2023.
When not to trust language models: Investigating
effectiveness of parametric and non-parametric mem-
ories. In Proceedings of the 61st Annual Meeting of
the Association for Computational Linguistics (Vol-
ume 1: Long Papers) , pages 9802–9822.
Nat McAleese, Rai Michael Pokorny, Juan Felipe Ceron
Uribe, Evgenia Nitishinskaya, Maja Trebacz, and Jan
Leike. 2024. Llm critics help catch llm bugs. arXiv
preprint arXiv:2407.00215 .
Rafael Rafailov, Archit Sharma, Eric Mitchell, Christo-
pher D Manning, Stefano Ermon, and Chelsea Finn.
2023. Direct preference optimization: Your lan-
guage model is secretly a reward model. Advances in
Neural Information Processing Systems , 36:53728–
53741.
Pranav Rajpurkar, Jian Zhang, Konstantin Lopyrev, and
Percy Liang. 2016. Squad: 100,000+ questions
for machine comprehension of text. arXiv preprint
arXiv:1606.05250 .
Parth Sarthi, Salman Abdullah, Aditi Tuli, Shubh
Khanna, Anna Goldie, and Christopher D Man-
ning. 2024. RAPTOR: Recursive abstractive pro-
cessing for tree-organized retrieval. arXiv preprint
arXiv:2401.18059 .
Timo Schick, Jane Dwivedi-Yu, Roberto Dessi, Roberta
Raileanu, Maria Lomeli, Eric Hambro, Luke Zettle-
moyer, Nicola Cancedda, and Thomas Scialom. 2023.
Toolformer: Language models can teach themselves
to use tools. In Thirty-seventh Conference on Neural
Information Processing Systems .
Weijia Shi, Sewon Min, Michihiro Yasunaga, Min-
joon Seo, Rich James, Mike Lewis, Luke Zettle-
moyer, and Wen-tau Yih. 2023. REPLUG: Retrieval-
augmented black-box language models. arXiv
preprint arXiv:2301.12652 .
Weijia Shi, Sewon Min, Michihiro Yasunaga, Min-
joon Seo, Richard James, Mike Lewis, Luke Zettle-
moyer, and Wen-tau Yih. 2024. REPLUG: Retrieval-
augmented black-box language models. In Proceed-
ings of NAACL-HLT , pages 8371–8384.
Huatong Song, Jinhao Jiang, Yingqian Min, Jie Chen,
Zhipeng Chen, Wayne Xin Zhao, Lei Fang, and Ji-
Rong Wen. 2025. R1-searcher: Incentivizing the
10

search capability in llms via reinforcement learning.
arXiv preprint arXiv:2503.05592 .
Ivan Stelmakh, Yi Luan, Bhuwan Dhingra, and Ming-
Wei Chang. 2022. ASQA: Factoid questions meet
long-form answers. In Proceedings of the 2022 Con-
ference on Empirical Methods in Natural Language
Processing , pages 8273–8288.
Hongjin Su, Howard Yen, Mengzhou Xia, Weijia Shi,
Niklas Muennighoff, Han-yu Wang, Haisu Liu, Quan
Shi, Zachary S Siegel, Michael Tang, and 1 others.
2024. BRIGHT: A realistic and challenging bench-
mark for reasoning-intensive retrieval. arXiv preprint
arXiv:2407.12883 .
Linzhuang Sun, Hao Liang, Jingxuan Wei, Bihui Yu,
Tianpeng Li, Fan Yang, Zenan Zhou, and Wentao
Zhang. 2025. Mm-verify: Enhancing multimodal
reasoning with chain-of-thought verification. arXiv
preprint arXiv:2502.13383 .
Bingyu Wan, Fuxi Zhang, Zhongpeng Qi, Jiayi Ding,
Jijun Li, Baoshi Fan, Yijia Zhang, and Jun Zhang.
2025. Cognitive-aligned document selection for
retrieval-augmented generation. arXiv preprint
arXiv:2502.11770 .
Yubo Wang, Xiang Yue, and Wenhu Chen. 2025a.
Critique fine-tuning: Learning to critique is more
effective than learning to imitate. arXiv preprint
arXiv:2501.17703 .
Zhengren Wang, Jiayang Yu, Dongsheng Ma, Zhe Chen,
Yu Wang, Zhiyu Li, Feiyu Xiong, Yanfeng Wang, Lin-
peng Tang, Wentao Zhang, and 1 others. 2025b. Rare:
Retrieval-augmented reasoning modeling. arXiv
preprint arXiv:2503.23513 .
Zilong Wang, Zifeng Wang, Long Le, Huaixiu Steven
Zheng, Swaroop Mishra, Vincent Perot, Yuwei
Zhang, Anush Mattapalli, Ankur Taly, Jingbo Shang,
and 1 others. 2024. Speculative rag: Enhancing re-
trieval augmented generation through drafting. arXiv
preprint arXiv:2407.08223 .
Jason Wei, Xuezhi Wang, Dale Schuurmans, Maarten
Bosma, Fei Xia, Ed Chi, Quoc V Le, Denny Zhou,
and 1 others. 2022. Chain-of-thought prompting elic-
its reasoning in large language models. Advances
in neural information processing systems , 35:24824–
24837.
Zhepei Wei, Wei-Lin Chen, and Yu Meng. 2024. In-
structrag: Instructing retrieval-augmented genera-
tion via self-synthesized rationales. arXiv preprint
arXiv:2406.13629 .
Yan Weng, Fengbin Zhu, Tong Ye, Haoyan Liu,
Fuli Feng, and Tat-Seng Chua. 2025. Optimiz-
ing knowledge integration in retrieval-augmented
generation with self-selection. arXiv preprint
arXiv:2502.06148 .Mingyan Wu, Zhenghao Liu, Yukun Yan, Xinze Li, Shi
Yu, Zheni Zeng, Yu Gu, and Ge Yu. 2025a. Rankcot:
Refining knowledge for retrieval-augmented gen-
eration through ranking chain-of-thoughts. arXiv
preprint arXiv:2502.17888 .
Yuyang Wu, Yifei Wang, Tianqi Du, Stefanie Jegelka,
and Yisen Wang. 2025b. When more is less: Un-
derstanding chain-of-thought length in llms. arXiv
preprint arXiv:2502.07266 .
Chong Xiang, Tong Wu, Zexuan Zhong, David Wagner,
Danqi Chen, and Prateek Mittal. 2024. Certifiably ro-
bust RAG against retrieval corruption. arXiv preprint
arXiv:2405.15556 .
Peng Xu, Wei Ping, Xianchao Wu, Lawrence McAfee,
Chen Zhu, Zihan Liu, Sandeep Subramanian, Evelina
Bakhturina, Mohammad Shoeybi, and Bryan Catan-
zaro. 2024. Retrieval meets long context large lan-
guage models. In The Twelfth International Confer-
ence on Learning Representations .
An Yang, Baosong Yang, Beichen Zhang, Binyuan Hui,
Bo Zheng, Bowen Yu, Chengyuan Li, Dayiheng Liu,
Fei Huang, Haoran Wei, and 1 others. 2024a. Qwen2.
5 technical report. arXiv preprint arXiv:2412.15115 .
Xiao Yang, Kai Sun, Hao Xin, Yushi Sun, Nikita Bhalla,
Xiangsen Chen, Sajal Choudhary, Rongze Daniel
Gui, Ziran Will Jiang, Ziyu Jiang, Lingkun Kong,
Brian Moran, Jiaqi Wang, Yifan Ethan Xu, An Yan,
Chenyu Yang, Eting Yuan, Hanwen Zha, Nan Tang,
and 8 others. 2024b. CRAG – comprehensive RAG
benchmark. arXiv preprint arXiv:2406.04744 .
Zhilin Yang, Peng Qi, Saizheng Zhang, Yoshua Ben-
gio, William W Cohen, Ruslan Salakhutdinov, and
Christopher D Manning. 2018. Hotpotqa: A dataset
for diverse, explainable multi-hop question answer-
ing. arXiv preprint arXiv:1809.09600 .
Ori Yoran, Tomer Wolfson, Ori Ram, and Jonathan Be-
rant. 2024. Making retrieval-augmented language
models robust to irrelevant context. In The Twelfth
International Conference on Learning Representa-
tions .
Mert Yuksekgonul, Federico Bianchi, Joseph Boen,
Sheng Liu, Zhi Huang, Carlos Guestrin, and James
Zou. 2024. Textgrad: Automatic" differentiation" via
text. arXiv preprint arXiv:2406.07496 .
Di Zhang, Xiaoshui Huang, Dongzhan Zhou, Yuqiang
Li, and Wanli Ouyang. 2024a. Accessing gpt-4
level mathematical olympiad solutions via monte
carlo tree self-refine with llama-3 8b. arXiv preprint
arXiv:2406.07394 .
Di Zhang, Jianbo Wu, Jingdi Lei, Tong Che, Jia-
tong Li, Tong Xie, Xiaoshui Huang, Shufei Zhang,
Marco Pavone, Yuqiang Li, and 1 others. 2024b.
Llama-berry: Pairwise optimization for o1-like
olympiad-level mathematical reasoning. arXiv
preprint arXiv:2410.02884 .
11

Jinghan Zhang, Xiting Wang, Weijieying Ren, Lu Jiang,
Dongjie Wang, and Kunpeng Liu. 2024c. RATT:
A thought structure for coherent and correct LLM
reasoning. arXiv preprint arXiv:2406.02746 .
Tianjun Zhang, Shishir G Patil, Naman Jain, Sheng
Shen, Matei Zaharia, Ion Stoica, and Joseph E
Gonzalez. 2024d. RAFT: Adapting language
model to domain specific RAG. arXiv preprint
arXiv:2403.10131 .
Xiang Zhang, Juntai Cao, Jiaqi Wei, Chenyu You, and
Dujian Ding. 2025. Why does your cot prompt (not)
work? theoretical analysis of prompt space com-
plexity, its interaction with answer space during cot
reasoning with llms: A recurrent perspective. arXiv
preprint arXiv:2503.10084 .
Zexuan Zhong, Ziqing Huang, Alexander Wettig, and
Danqi Chen. 2023. Poisoning retrieval corpora by
injecting adversarial passages. In Proceedings of the
2023 Conference on Empirical Methods in Natural
Language Processing , pages 13764–13775.
Shuyan Zhou, Uri Alon, Frank F Xu, Zhengbao Jiang,
and Graham Neubig. 2022. DocPrompting: Gener-
ating code by retrieving the docs. In The Eleventh
International Conference on Learning Representa-
tions .
Wei Zou, Runpeng Geng, Binghui Wang, and Jinyuan
Jia. 2024. PoisonedRAG: Knowledge poisoning at-
tacks to retrieval-augmented generation of large lan-
guage models. arXiv preprint arXiv:2402.07867 .
12

A Appendix
A.1 Additional Implementation Details
Retrieve Setup. We use the Wikipedia corpus pro-
vided by (Jin et al., 2024b) as the default external
knowledge source for retrieval. We evaluate our
method on seven diverse QA benchmarks span-
ning multiple task types, including standard factoid
QA, multi-hop reasoning, and long-form genera-
tion. PopQA (Mallen et al., 2023), TriviaQA (Joshi
et al., 2017), NaturalQuestions (Kwiatkowski et al.,
2019), and SQuAD (Rajpurkar et al., 2016) fall
under standard factoid QA, where models answer
factual questions based on Wikipedia or web-based
evidence.
•PopQA focuses on entity-centric questions
derived from structured knowledge bases, test-
ing factual recall over encyclopedic content.
•TriviaQA contains trivia-style questions au-
thored by enthusiasts, each paired with multi-
ple distant-supervised evidence documents.
•NaturalQuestions presents real user queries
issued to Google Search, with answers ex-
tracted from Wikipedia, simulating realistic
search behavior.
•ASQA (Stelmakh et al., 2022) is a long-form
QA benchmark focused on ambiguous ques-
tions with paragraph-level answers.
•2WikiMultiHopQA (Ho et al., 2020) and Hot-
potQA (Yang et al., 2018) are multi-hop QA
datasets that require reasoning over multiple
passages. 2WikiMultiHopQA evaluates com-
positional reasoning across two Wikipedia ar-
ticles, while HotpotQA incorporates both sup-
porting and distracting sentences, encouraging
interpretable multi-step reasoning.
•SQuAD is a widely used extractive QA dataset
where answers are short spans from Wikipedia
passages.
Following the setup in InstructRAG (Wei et al.,
2024), we adopt dataset-specific retrievers for each
query: Contriever-MS MARCO for PopQA and
TriviaQA, DPR for NaturalQuestions, GTR for
ASQA, and BM25 for 2WikiMultiHopQA. For
HotpotQA and SQuAD, we adopt the e5-base-v2
encoder. By default, we retrieve the top 5 most rel-
evant documents from Wikipedia corpus for each
question.Training Details. We fine-tune our models using
the LoRA method on 2 NVIDIA A100 GPUs, each
with 80GB of memory. The fine-tuning process
is conducted over 2 epochs with a learning rate of
1e-5and employs a per-device batch size of 16,
leveraging gradient accumulation to handle larger
effective batch sizes. We set the LoRA-specific
hyperparameters as follows: lora_rank = 16 and
lora_alpha = 64 , ensuring efficient adaptation to
downstream tasks. The sequence cutoff length is
6144 tokens, with a warmup ratio of 0.1 applied
to stabilize training. Additionally, we utilize bf16
(brain floating point) precision to reduce memory
usage and accelerate training while maintaining
numerical stability.
A.2 Training Corpus Construction Details
To systematically simulate varying degrees of an-
swerability, we introduce a novel four-tier Contex-
tual Granularity Hierarchy (Figure 2), which forms
the basis for structured context-aware critique learn-
ing. This hierarchy is designed to expose critique
models to a broad spectrum of evidence conditions,
thereby facilitating fine-grained supervision under
explicitly controlled scenarios.
The hierarchy is defined along three orthogo-
nal dimensions of contextual variation— relevance ,
helpfulness , and completeness —and comprises the
following four levels:
Hierarchy-1: Not Relevant, Not Helpful, Not
Complete. We sample 200 instances per bench-
mark, where the context is randomly selected from
evidence retrieved for unrelated questions. These
contexts are neither topically relevant nor contain
partial answers, thus offering no utility in address-
ing the question.
Hierarchy-2: Relevant, Not Helpful, Not Com-
plete. We sample 400 instances per benchmark,
in which the retrieved context comprises the top-5
documents relevant to the question but lacking any
content that supports a correct answer. Although
relevant, these contexts remain unhelpful and in-
complete.
Hierarchy-3 & 4: Relevant, Helpful, Not Com-
plete / Complete. We sample 1,400 instances per
benchmark where the context is both relevant and
helpful, containing either partial (no single doc-
ument provides a complete answer) or complete
answer-supporting information. To capture vary-
ing levels of difficulty, we categorize queries into
five tiers based on the number of documents that
individually contain supporting evidence (ranging
13

from 1 to 5). Easier queries correspond to a higher
number of such documents. We sample 400, 400,
200, 200, and 200 instances across these five levels,
respectively.
This hierarchical corpus introduces a novel fine-
grained supervision signal for training critique
models, enabling a more nuanced understanding
of answerability and evidence quality in retrieval-
augmented generation.
A.3 Critic LLM Training via CPO
In addition to Critique Fine-Tuning (CFT) , we in-
troduce a novel training paradigm for critique lan-
guage models, termed Critique Preference Op-
timization (CPO) . CPO extends the Direct Pref-
erence Optimization (DPO) framework (Rafailov
et al., 2023) to the domain of critique generation,
enabling preference-based alignment of critique
models with respect to human-quality judgments.
For each training example, we construct a pair of
candidate critiques: a rejected critique ∆y−
unexp gen-
erated by a weaker model Mweak, and an accepted
critique ∆y+
unexp from a stronger model Mstrong.
The critic model Mcriticis then optimized to prefer
the stronger critique over the weaker one using a
ranking-based objective:
LCPO=−ECh
logσ
βlogpθ(∆y+
unexp|q,D,y+
unexp)
pθ(∆y−
unexp|q,D,y+
unexp)i
,
(9)
where σ(·)denotes the sigmoid function, βis a tem-
perature parameter controlling preference sharp-
ness, and pθis the conditional likelihood of a cri-
tique under the model. Importantly, the condi-
tioning includes the stronger generation y+
unexp to
ground the critique in high-quality reference behav-
ior.
This training strategy represents a novel appli-
cation of preference optimization to the critique
generation setting. It allows the model to learn
fine-grained distinctions in critique quality and im-
proves alignment with human preferences, surpass-
ing traditional supervised learning approaches in
adaptability and scalability.
A.4 Pseudo-code of Novel Algorithms for
Critique-Aware Learning
To promote clarity and reproducibility, we present
formalized pseudo-code for the core contributions
of our framework, highlighting novel procedures
for critique generation, fine-tuning, and alignment.
These algorithmic components reflect our key inno-vations in critique-aware generation and optimiza-
tion.
Algorithm 1 introduces Contrastive Critique
Synthesis , a novel mechanism that elicits action-
able critiques by contrasting outputs from a weak
and a strong model. This facilitates the identifica-
tion of failure modes in weaker generations using
preference-informed critique models. Algorithm 2,
Critique Fine-Tuning (CFT) , formalizes a super-
vised learning regime using synthetic critiques and
structured input templates to fine-tune a base model
toward producing useful critiques.
In Algorithm 3, we present Critique Prefer-
ence Optimization (CPO) , which extends the Di-
rect Preference Optimization (DPO) framework
to critique generation. This formulation enables
preference-based alignment of critique models us-
ing pairs of more and less preferred critiques.
Lastly, Algorithm 4 describes Critique-Driven
Alignment (CDA) , a novel iterative refinement
procedure that integrates critique signals into the
generation loop, producing responses that are suc-
cessively improved based on model-generated feed-
back.
Collectively, these algorithmic components de-
fine a unified, modular framework for critique-
aware alignment, marking a novel contribution
to controllable and preference-aligned language
model training.
14

Algorithm 1 CONTRASTIVE CRITIQUE SYNTHE -
SIS(Novel critique generation via response com-
parison)
Require: Input query q, contextual grounding D,
weak model Mweak, strong model Mstrong , cri-
tique model Mcritic
Ensure: Generated critique ∆yunexp for weak
model output
1:yunexp← M weak(q,D)▷Generate suboptimal
response
2:yexp← M strong(q,D)▷Generate preferred
response
3:Xpref←(q,D, yexp, yunexp) ▷Construct
preference-informed input
4:∆yunexp← M critic(Xpref) ▷Generate
contrastive critique
5:∆yunexp← G(∆yunexp, yexp)▷Refine critique
with improvement guidance
6:return ∆yunexp
Algorithm 2 CRITIQUE FINE-TUNING (CFT) : Su-
pervised adaptation via synthetic critiques
Require: Base model Mweak, synthetic dataset C,
template Icritic, learning rate η, epochs N
Ensure: Critique-aware model Mcritic
1:Mcritic← M weak ▷Initialize from weak
model
2:forepoch = 1toNdo
3: foreach (q,D, yunexp,∆yunexp, yexp)∈ C
do
4: Icritic←(q,D, yunexp, yexp) ▷
Compose critique context
5: ∆ˆyunexp∼pθ(· | I critic) ▷Predict
critique
6: LCFT← − logpθ(∆yunexp| Icritic)▷
Compute NLL loss
7: θ←θ−η∇θLCFT ▷Update model
8:return McriticAlgorithm 3 CRITIQUE PREFERENCE OPTIMIZA -
TION (CPO) : Alignment via pairwise critique pref-
erences
Require: Queries {q}, contexts {D}, weak model
Mweak, strong model Mstrong, initial model
Mcritic, temperature β
Ensure: Preference-aligned critique model
Mcritic
1:foreach(q,D)do
2: ∆y−
unexp← M weak(q,D) ▷Infer
less-preferred critique
3: ∆y+
unexp← M strong(q,D) ▷Infer
preferred critique
4:P ← (∆y−
unexp,∆y+
unexp) ▷Construct
preference pair
5:forepoch = 1toNdo
6: foreachP= (∆ y−,∆y+)do
7: Compute preference loss LDPO ▷
Direct Preference Optimization loss
8: θ←θ−η∇θLDPO ▷Update
parameters
9:return Mcritic
Algorithm 4 CRITIQUE -DRIVEN ALIGNMENT
(CDA) : Iterative refinement via model-generated
critique signals
Require: Query q, document set D =
{d1, . . . , d n}, generation model Mgen,
critique model Mcritic, iterations T
Ensure: Refined, critique-aligned response yexp
1:y0← M gen(q,D) ▷Initial
retrieval-augmented generation
2:fort= 0toT−1do
3: ∆yt← M critic(yt, q,D) ▷Critique
current response
4: yt+1← M gen(yt⊕∆yt, q,D)▷Refine
using critique
5:yexp←yT ▷Final critique-aware output
6:return yexp
15

A.5 Additional Experiment Results
In this section, we present additional experimental
results to provide a comprehensive understanding
of the proposed method and its performance under
various conditions.
Iterative Refinement Performance. To evalu-
ate test-time scalability, we analyze the detailed
performance of ALIGN RAG andSELF-REFINE
across five refinement iterations over seven bench-
marks, using three instruction-tuned backbones.
The results, shown in Figure 6, demonstrate
thatALIGN RAG consistently outperforms SELF-
REFINE across all iterations, models, and tasks.
On average, ALIGN RAG achieves notable im-
provements on challenging benchmarks such as
ASQA (e.g., +1.9 on Qwen2.5-7B at iteration 5)
and 2WikiMultiHopQA (e.g., +7.8 on LLaMA3.1-
8B at iteration 4). The gains are particularly sig-
nificant on out-of-distribution (OOD) datasets such
as HotpotQA and SQuAD, where ALIGN RAG sur-
passes SELF-REFINE by more than 10 points on
LLaMA3.1-8B. These findings highlight the robust-
ness and effectiveness of ALIGN RAG in handling
domain shifts.
Integration into Existing RAG Pipelines. To
assess the plug-and-play compatibility of our align-
ment strategy, we integrate it into the INSTRUC -
TRAG framework across three backbones and eval-
uate its performance on seven benchmarks. The
detailed results, provided in Figure 7, reveal that
our alignment approach consistently improves ac-
curacy, both for in-domain datasets (e.g., PopQA,
TriviaQA) and OOD datasets (e.g., SQuAD, Hot-
potQA). Notably, the improvements are particu-
larly pronounced on challenging datasets such as
SQuAD (+10.2 on Qwen2.5-14B) and HotpotQA
(+8.6 on Qwen2.5-14B). These results demonstrate
that our method can be seamlessly incorporated
into existing RAG pipelines, enabling substantial
test-time improvements without requiring modifi-
cations to the model architecture or training objec-
tives.
Robustness under Retrieval Quality Variance.
To evaluate robustness under varying retrieval con-
ditions, we compare Vanilla RAG, SELF-REFINE ,
andALIGN RAG in two retrieval scenarios: ( An-
swerable ) and ( Unanswerable ). Figure 8 sum-
marizes the results. In the Unanswerable sce-
nario, where noisy or misleading retrieval often
causes reasoning misalignment, ALIGN RAG con-sistently outperforms the baselines. For example,
on NaturalQuestions (e.g., +8.3 on Qwen2.5-7B)
and 2WikiMultiHopQA (e.g., +7.3 on LLaMA3.1-
8B)—two tasks particularly sensitive to retrieval
quality— ALIGN RAG achieves the largest margins
over SELF-REFINE . Even in the Answerable sce-
nario, where retrieved documents are highly rel-
evant, ALIGN RAG demonstrates superior accu-
racy (e.g., +3.4 on ASQA and +4.3 on 2WikiMul-
tiHopQA using Qwen2.5-14B). These results il-
lustrate that ALIGN RAG enhances reasoning ro-
bustness across a wide range of retrieval quality
levels.
Ablation on CLM and Contrastive Critique
Synthesis. To supplement the high-level abla-
tion analysis, Table 4 presents detailed results
for seven benchmarks under four Critic Language
Model (CLM) configurations. While the main text
reports averaged scores for in-domain (ID) and
out-of-distribution (OOD) settings, the table pro-
vides benchmark-specific insights. Introducing a
frozen CLM yields noticeable gains over Vanilla
RAG (e.g., +3.8 on PopQA and +3.9 on ASQA for
Qwen2.5-7B), confirming the utility of auxiliary
critique. Further training of the CLM amplifies
these benefits, particularly for OOD datasets such
as SQuAD and HotpotQA. Notably, our contrastive
critique synthesis (CCS) achieves the best perfor-
mance on nearly all benchmarks, including a +3.2
gain on MultiHopQA and +4.2 gain on SQuAD for
Qwen2.5-14B. These results demonstrate that con-
trastive alignment is crucial for generating retrieval-
sensitive critiques, leading to consistent and robust
improvements across diverse QA scenarios.
Different Training Strategies for CLM. To
compare training strategies for the CLM, Table 5
evaluates our proposed Critique Fine-tuning (CFT)
approach against Critique Preference Optimization
(CPO) A.3. Across three backbones and seven
benchmarks, CFT consistently outperforms CPO,
particularly on retrieval-sensitive and OOD-heavy
tasks such as HotpotQA and SQuAD. For exam-
ple, on Qwen2.5-14B, CFT raises the average ac-
curacy from 51.0 to 53.8 and improves perfor-
mance on SQuAD from 20.2 to 25.4. Similarly,
on LLaMA3.1-8B, CFT achieves a +4.4 gain in
average performance and a +7.9 improvement on
SQuAD. These results underscore the superior-
ity of preference-based critique generation over
preference-based output generation for CLM train-
ing, particularly in retrieval-intensive contexts.
16

Generalization to Out-of-Distribution Data.
To supplement the OOD generalization results in
Figure 5, Table 6 provides a complete breakdown
of ID and OOD performance across benchmarks
and backbones. While the main text reports average
performance drops between ID and OOD settings,
the detailed analysis reveals that ALIGN RAG re-
duces the OOD drop significantly (e.g., from 40.3
to 32.6 on Qwen2.5-7B and from 44.0 to 35.1 on
LLaMA3.1-8B) compared to SELF-REFINE . Addi-
tionally, ALIGN RAG achieves substantial absolute
gains on OOD datasets (e.g., +12.6 on HotpotQA
and +9.6 on SQuAD for Qwen2.5-7B), demonstrat-
ing improved generalization capabilities under do-
main shifts. These results confirm that the proposed
CDA-based alignment strategy enhances model ro-
bustness across distributions without overfitting to
the training data.
17

/uni00000031/uni00000058/uni00000050/uni00000045/uni00000048/uni00000055/uni00000003/uni00000052/uni00000049/uni00000003/uni0000002c/uni00000057/uni00000048/uni00000055/uni00000044/uni00000057/uni0000004c/uni00000052/uni00000051/uni00000056/uni00000019/uni00000018/uni00000011/uni00000013/uni00000019/uni00000018/uni00000011/uni00000017/uni00000019/uni00000018/uni00000011/uni0000001b/uni00000019/uni00000019/uni00000011/uni00000015/uni00000019/uni00000019/uni00000011/uni00000019/uni00000019/uni0000001a/uni00000011/uni00000013/uni00000034/uni0000005a/uni00000048/uni00000051/uni00000015/uni00000011/uni00000018/uni00000010/uni0000001a/uni00000025
/uni00000033/uni00000052/uni00000053/uni00000034/uni00000024
/uni00000031/uni00000058/uni00000050/uni00000045/uni00000048/uni00000055/uni00000003/uni00000052/uni00000049/uni00000003/uni0000002c/uni00000057/uni00000048/uni00000055/uni00000044/uni00000057/uni0000004c/uni00000052/uni00000051/uni00000056/uni0000001a/uni00000017/uni00000011/uni00000013/uni0000001a/uni00000018/uni00000011/uni00000013/uni0000001a/uni00000019/uni00000011/uni00000013/uni0000001a/uni0000001a/uni00000011/uni00000013/uni0000001a/uni0000001b/uni00000011/uni00000013/uni0000001a/uni0000001c/uni00000011/uni00000013
/uni00000037/uni00000055/uni0000004c/uni00000059/uni0000004c/uni00000044/uni00000034/uni00000024
/uni00000031/uni00000058/uni00000050/uni00000045/uni00000048/uni00000055/uni00000003/uni00000052/uni00000049/uni00000003/uni0000002c/uni00000057/uni00000048/uni00000055/uni00000044/uni00000057/uni0000004c/uni00000052/uni00000051/uni00000056/uni00000019/uni00000013/uni00000011/uni00000013/uni00000019/uni00000015/uni00000011/uni00000013/uni00000019/uni00000017/uni00000011/uni00000013/uni00000019/uni00000019/uni00000011/uni00000013/uni00000019/uni0000001b/uni00000011/uni00000013/uni0000001a/uni00000013/uni00000011/uni00000013
/uni00000031/uni00000044/uni00000057/uni00000058/uni00000055/uni00000044/uni0000004f/uni00000034/uni00000058/uni00000048/uni00000056/uni00000057/uni0000004c/uni00000052/uni00000051/uni00000056
/uni00000031/uni00000058/uni00000050/uni00000045/uni00000048/uni00000055/uni00000003/uni00000052/uni00000049/uni00000003/uni0000002c/uni00000057/uni00000048/uni00000055/uni00000044/uni00000057/uni0000004c/uni00000052/uni00000051/uni00000056/uni00000017/uni00000016/uni00000011/uni00000013/uni00000017/uni00000018/uni00000011/uni00000013/uni00000017/uni0000001a/uni00000011/uni00000013/uni00000017/uni0000001c/uni00000011/uni00000013/uni00000018/uni00000014/uni00000011/uni00000013/uni00000018/uni00000016/uni00000011/uni00000013
/uni00000015/uni0000003a/uni0000004c/uni0000004e/uni0000004c/uni00000030/uni00000058/uni0000004f/uni00000057/uni0000004c/uni0000002b/uni00000052/uni00000053/uni00000034/uni00000024
/uni00000031/uni00000058/uni00000050/uni00000045/uni00000048/uni00000055/uni00000003/uni00000052/uni00000049/uni00000003/uni0000002c/uni00000057/uni00000048/uni00000055/uni00000044/uni00000057/uni0000004c/uni00000052/uni00000051/uni00000056/uni00000017/uni00000018/uni00000011/uni00000013/uni00000017/uni00000019/uni00000011/uni00000013/uni00000017/uni0000001a/uni00000011/uni00000013/uni00000017/uni0000001b/uni00000011/uni00000013/uni00000017/uni0000001c/uni00000011/uni00000013/uni00000018/uni00000013/uni00000011/uni00000013
/uni00000024/uni00000036/uni00000034/uni00000024
/uni00000031/uni00000058/uni00000050/uni00000045/uni00000048/uni00000055/uni00000003/uni00000052/uni00000049/uni00000003/uni0000002c/uni00000057/uni00000048/uni00000055/uni00000044/uni00000057/uni0000004c/uni00000052/uni00000051/uni00000056/uni00000014/uni0000001b/uni00000011/uni00000013/uni00000015/uni00000015/uni00000011/uni00000013/uni00000015/uni00000019/uni00000011/uni00000013/uni00000016/uni00000013/uni00000011/uni00000013/uni00000016/uni00000017/uni00000011/uni00000013/uni00000016/uni0000001b/uni00000011/uni00000013
/uni0000002b/uni00000052/uni00000057/uni00000053/uni00000052/uni00000057/uni00000034/uni00000024
/uni00000031/uni00000058/uni00000050/uni00000045/uni00000048/uni00000055/uni00000003/uni00000052/uni00000049/uni00000003/uni0000002c/uni00000057/uni00000048/uni00000055/uni00000044/uni00000057/uni0000004c/uni00000052/uni00000051/uni00000056/uni00000014/uni00000013/uni00000011/uni00000013/uni00000014/uni00000017/uni00000011/uni00000013/uni00000014/uni0000001b/uni00000011/uni00000013/uni00000015/uni00000015/uni00000011/uni00000013/uni00000015/uni00000019/uni00000011/uni00000013/uni00000016/uni00000013/uni00000011/uni00000013
/uni00000036/uni00000034/uni00000058/uni00000024/uni00000027
/uni00000031/uni00000058/uni00000050/uni00000045/uni00000048/uni00000055/uni00000003/uni00000052/uni00000049/uni00000003/uni0000002c/uni00000057/uni00000048/uni00000055/uni00000044/uni00000057/uni0000004c/uni00000052/uni00000051/uni00000056/uni00000019/uni00000019/uni00000011/uni00000013/uni00000019/uni00000019/uni00000011/uni00000019/uni00000019/uni0000001a/uni00000011/uni00000015/uni00000019/uni0000001a/uni00000011/uni0000001b/uni00000019/uni0000001b/uni00000011/uni00000017/uni00000019/uni0000001c/uni00000011/uni00000013/uni00000034/uni0000005a/uni00000048/uni00000051/uni00000015/uni00000011/uni00000018/uni00000010/uni00000014/uni00000017/uni00000025
/uni00000031/uni00000058/uni00000050/uni00000045/uni00000048/uni00000055/uni00000003/uni00000052/uni00000049/uni00000003/uni0000002c/uni00000057/uni00000048/uni00000055/uni00000044/uni00000057/uni0000004c/uni00000052/uni00000051/uni00000056/uni0000001a/uni0000001a/uni00000011/uni00000013/uni0000001a/uni0000001a/uni00000011/uni0000001b/uni0000001a/uni0000001b/uni00000011/uni00000019/uni0000001a/uni0000001c/uni00000011/uni00000017/uni0000001b/uni00000013/uni00000011/uni00000015/uni0000001b/uni00000014/uni00000011/uni00000013
/uni00000031/uni00000058/uni00000050/uni00000045/uni00000048/uni00000055/uni00000003/uni00000052/uni00000049/uni00000003/uni0000002c/uni00000057/uni00000048/uni00000055/uni00000044/uni00000057/uni0000004c/uni00000052/uni00000051/uni00000056/uni00000019/uni00000016/uni00000011/uni00000013/uni00000019/uni00000017/uni00000011/uni00000019/uni00000019/uni00000019/uni00000011/uni00000015/uni00000019/uni0000001a/uni00000011/uni0000001b/uni00000019/uni0000001c/uni00000011/uni00000017/uni0000001a/uni00000014/uni00000011/uni00000013
/uni00000031/uni00000058/uni00000050/uni00000045/uni00000048/uni00000055/uni00000003/uni00000052/uni00000049/uni00000003/uni0000002c/uni00000057/uni00000048/uni00000055/uni00000044/uni00000057/uni0000004c/uni00000052/uni00000051/uni00000056/uni00000017/uni00000017/uni00000011/uni00000013/uni00000017/uni00000018/uni00000011/uni0000001b/uni00000017/uni0000001a/uni00000011/uni00000019/uni00000017/uni0000001c/uni00000011/uni00000017/uni00000018/uni00000014/uni00000011/uni00000015/uni00000018/uni00000016/uni00000011/uni00000013
/uni00000031/uni00000058/uni00000050/uni00000045/uni00000048/uni00000055/uni00000003/uni00000052/uni00000049/uni00000003/uni0000002c/uni00000057/uni00000048/uni00000055/uni00000044/uni00000057/uni0000004c/uni00000052/uni00000051/uni00000056/uni00000017/uni00000019/uni00000011/uni00000013/uni00000017/uni0000001a/uni00000011/uni00000015/uni00000017/uni0000001b/uni00000011/uni00000017/uni00000017/uni0000001c/uni00000011/uni00000019/uni00000018/uni00000013/uni00000011/uni0000001b/uni00000018/uni00000015/uni00000011/uni00000013
/uni00000031/uni00000058/uni00000050/uni00000045/uni00000048/uni00000055/uni00000003/uni00000052/uni00000049/uni00000003/uni0000002c/uni00000057/uni00000048/uni00000055/uni00000044/uni00000057/uni0000004c/uni00000052/uni00000051/uni00000056/uni00000015/uni00000013/uni00000011/uni00000013/uni00000015/uni00000016/uni00000011/uni00000019/uni00000015/uni0000001a/uni00000011/uni00000015/uni00000016/uni00000013/uni00000011/uni0000001b/uni00000016/uni00000017/uni00000011/uni00000017/uni00000016/uni0000001b/uni00000011/uni00000013
/uni00000031/uni00000058/uni00000050/uni00000045/uni00000048/uni00000055/uni00000003/uni00000052/uni00000049/uni00000003/uni0000002c/uni00000057/uni00000048/uni00000055/uni00000044/uni00000057/uni0000004c/uni00000052/uni00000051/uni00000056/uni00000014/uni00000015/uni00000011/uni00000013/uni00000014/uni00000018/uni00000011/uni00000019/uni00000014/uni0000001c/uni00000011/uni00000015/uni00000015/uni00000015/uni00000011/uni0000001b/uni00000015/uni00000019/uni00000011/uni00000017/uni00000016/uni00000013/uni00000011/uni00000013
/uni00000014 /uni00000015 /uni00000016 /uni00000017 /uni00000018
/uni00000031/uni00000058/uni00000050/uni00000045/uni00000048/uni00000055/uni00000003/uni00000052/uni00000049/uni00000003/uni0000002c/uni00000057/uni00000048/uni00000055/uni00000044/uni00000057/uni0000004c/uni00000052/uni00000051/uni00000056/uni00000019/uni00000018/uni00000011/uni00000013/uni00000019/uni00000018/uni00000011/uni0000001b/uni00000019/uni00000019/uni00000011/uni00000019/uni00000019/uni0000001a/uni00000011/uni00000017/uni00000019/uni0000001b/uni00000011/uni00000015/uni00000019/uni0000001c/uni00000011/uni00000013/uni0000002f/uni0000002f/uni00000044/uni00000030/uni00000024/uni00000016/uni00000011/uni00000014/uni00000010/uni0000001b/uni00000025
/uni00000014 /uni00000015 /uni00000016 /uni00000017 /uni00000018
/uni00000031/uni00000058/uni00000050/uni00000045/uni00000048/uni00000055/uni00000003/uni00000052/uni00000049/uni00000003/uni0000002c/uni00000057/uni00000048/uni00000055/uni00000044/uni00000057/uni0000004c/uni00000052/uni00000051/uni00000056/uni0000001a/uni00000016/uni00000011/uni00000013/uni0000001a/uni00000017/uni00000011/uni00000017/uni0000001a/uni00000018/uni00000011/uni0000001b/uni0000001a/uni0000001a/uni00000011/uni00000015/uni0000001a/uni0000001b/uni00000011/uni00000019/uni0000001b/uni00000013/uni00000011/uni00000013
/uni00000014 /uni00000015 /uni00000016 /uni00000017 /uni00000018
/uni00000031/uni00000058/uni00000050/uni00000045/uni00000048/uni00000055/uni00000003/uni00000052/uni00000049/uni00000003/uni0000002c/uni00000057/uni00000048/uni00000055/uni00000044/uni00000057/uni0000004c/uni00000052/uni00000051/uni00000056/uni00000019/uni00000013/uni00000011/uni00000013/uni00000019/uni00000015/uni00000011/uni00000013/uni00000019/uni00000017/uni00000011/uni00000013/uni00000019/uni00000019/uni00000011/uni00000013/uni00000019/uni0000001b/uni00000011/uni00000013/uni0000001a/uni00000013/uni00000011/uni00000013
/uni00000014 /uni00000015 /uni00000016 /uni00000017 /uni00000018
/uni00000031/uni00000058/uni00000050/uni00000045/uni00000048/uni00000055/uni00000003/uni00000052/uni00000049/uni00000003/uni0000002c/uni00000057/uni00000048/uni00000055/uni00000044/uni00000057/uni0000004c/uni00000052/uni00000051/uni00000056/uni00000017/uni00000015/uni00000011/uni00000013/uni00000017/uni00000017/uni00000011/uni00000015/uni00000017/uni00000019/uni00000011/uni00000017/uni00000017/uni0000001b/uni00000011/uni00000019/uni00000018/uni00000013/uni00000011/uni0000001b/uni00000018/uni00000016/uni00000011/uni00000013
/uni00000014 /uni00000015 /uni00000016 /uni00000017 /uni00000018
/uni00000031/uni00000058/uni00000050/uni00000045/uni00000048/uni00000055/uni00000003/uni00000052/uni00000049/uni00000003/uni0000002c/uni00000057/uni00000048/uni00000055/uni00000044/uni00000057/uni0000004c/uni00000052/uni00000051/uni00000056/uni00000017/uni00000017/uni00000011/uni00000013/uni00000017/uni00000018/uni00000011/uni00000017/uni00000017/uni00000019/uni00000011/uni0000001b/uni00000017/uni0000001b/uni00000011/uni00000015/uni00000017/uni0000001c/uni00000011/uni00000019/uni00000018/uni00000014/uni00000011/uni00000013
/uni00000014 /uni00000015 /uni00000016 /uni00000017 /uni00000018
/uni00000031/uni00000058/uni00000050/uni00000045/uni00000048/uni00000055/uni00000003/uni00000052/uni00000049/uni00000003/uni0000002c/uni00000057/uni00000048/uni00000055/uni00000044/uni00000057/uni0000004c/uni00000052/uni00000051/uni00000056/uni00000014/uni0000001a/uni00000011/uni00000013/uni00000015/uni00000014/uni00000011/uni00000015/uni00000015/uni00000018/uni00000011/uni00000017/uni00000015/uni0000001c/uni00000011/uni00000019/uni00000016/uni00000016/uni00000011/uni0000001b/uni00000016/uni0000001b/uni00000011/uni00000013
/uni00000014 /uni00000015 /uni00000016 /uni00000017 /uni00000018
/uni00000031/uni00000058/uni00000050/uni00000045/uni00000048/uni00000055/uni00000003/uni00000052/uni00000049/uni00000003/uni0000002c/uni00000057/uni00000048/uni00000055/uni00000044/uni00000057/uni0000004c/uni00000052/uni00000051/uni00000056/uni0000001a/uni00000011/uni00000013/uni00000014/uni00000014/uni00000011/uni00000019/uni00000014/uni00000019/uni00000011/uni00000015/uni00000015/uni00000013/uni00000011/uni0000001b/uni00000015/uni00000018/uni00000011/uni00000017/uni00000016/uni00000013/uni00000011/uni00000013
/uni00000024/uni0000004f/uni0000004c/uni0000004a/uni00000051/uni00000035/uni00000024/uni0000002a /uni00000036/uni00000048/uni0000004f/uni00000049/uni00000010/uni00000035/uni00000048/uni00000049/uni0000004c/uni00000051/uni00000048Figure 6: Performance of AlignRAG and Self-Refine across five refinement iterations on seven benchmarks.
/uni00000036/uni00000034/uni00000058/uni00000024/uni00000027/uni0000002b/uni00000052/uni00000057/uni00000053/uni00000052/uni00000057/uni00000034/uni00000024/uni00000024/uni00000036/uni00000034/uni00000024
/uni00000030/uni00000058/uni0000004f/uni00000057/uni0000004c/uni0000002b/uni00000052/uni00000053/uni00000034/uni00000024/uni00000031/uni00000034
/uni00000033/uni00000052/uni00000053/uni00000034/uni00000024/uni00000037/uni00000055/uni0000004c/uni00000059/uni0000004c/uni00000044/uni00000034/uni00000024/uni00000013/uni00000015/uni00000013/uni00000017/uni00000013/uni00000019/uni00000013/uni0000001b/uni00000013/uni00000024/uni00000046/uni00000046/uni00000058/uni00000055/uni00000044/uni00000046/uni0000005c/uni00000003/uni0000000b/uni00000008/uni0000000c/uni00000034/uni0000005a/uni00000048/uni00000051/uni00000015/uni00000011/uni00000018/uni00000010/uni0000001a/uni00000025
/uni00000036/uni00000034/uni00000058/uni00000024/uni00000027/uni0000002b/uni00000052/uni00000057/uni00000053/uni00000052/uni00000057/uni00000034/uni00000024/uni00000024/uni00000036/uni00000034/uni00000024
/uni00000030/uni00000058/uni0000004f/uni00000057/uni0000004c/uni0000002b/uni00000052/uni00000053/uni00000034/uni00000024/uni00000031/uni00000034
/uni00000033/uni00000052/uni00000053/uni00000034/uni00000024/uni00000037/uni00000055/uni0000004c/uni00000059/uni0000004c/uni00000044/uni00000034/uni00000024/uni00000034/uni0000005a/uni00000048/uni00000051/uni00000015/uni00000011/uni00000018/uni00000010/uni00000014/uni00000017/uni00000025
/uni00000036/uni00000034/uni00000058/uni00000024/uni00000027/uni0000002b/uni00000052/uni00000057/uni00000053/uni00000052/uni00000057/uni00000034/uni00000024/uni00000024/uni00000036/uni00000034/uni00000024
/uni00000030/uni00000058/uni0000004f/uni00000057/uni0000004c/uni0000002b/uni00000052/uni00000053/uni00000034/uni00000024/uni00000031/uni00000034
/uni00000033/uni00000052/uni00000053/uni00000034/uni00000024/uni00000037/uni00000055/uni0000004c/uni00000059/uni0000004c/uni00000044/uni00000034/uni00000024/uni0000002f/uni0000002f/uni00000044/uni00000030/uni00000024/uni00000016/uni00000011/uni00000014/uni00000010/uni0000001b/uni00000025/uni0000002c/uni00000051/uni00000056/uni00000057/uni00000055/uni00000058/uni00000046/uni00000057/uni00000035/uni00000024/uni0000002a /uni0000005a/uni00000012/uni00000003/uni00000024/uni0000004f/uni0000004c/uni0000004a/uni00000051/uni00000050/uni00000048/uni00000051/uni00000057
Figure 7: Details of evaluation result of InstructRAG w/o and w/ our Alignment method on three backbones across
seven benchmarks.
/uni00000033/uni00000052/uni00000053/uni00000034/uni00000024
/uni00000037/uni00000055/uni0000004c/uni00000059/uni0000004c/uni00000044/uni00000034/uni00000024
/uni00000031/uni00000044/uni00000057/uni00000058/uni00000055/uni00000044/uni0000004f/uni00000034/uni00000058/uni00000048/uni00000056/uni00000057/uni0000004c/uni00000052/uni00000051/uni00000056 /uni00000015/uni0000003a/uni0000004c/uni0000004e/uni0000004c/uni00000030/uni00000058/uni0000004f/uni00000057/uni0000004c/uni0000002b/uni00000052/uni00000053/uni00000034/uni00000024/uni00000024/uni00000036/uni00000034/uni00000024/uni00000018/uni00000011/uni00000015
/uni00000015/uni00000014/uni00000011/uni00000019
/uni00000014/uni0000001a/uni00000011/uni00000013
/uni00000016/uni0000001b/uni00000011/uni00000019/uni00000016/uni00000019/uni00000011/uni0000001a/uni00000017/uni00000011/uni0000001a
/uni00000014/uni00000018/uni00000011/uni00000019
/uni0000001b/uni00000011/uni0000001a
/uni00000016/uni00000016/uni00000011/uni00000019/uni00000016/uni00000017/uni00000011/uni00000013/uni00000017/uni00000011/uni00000013
/uni00000014/uni00000016/uni00000011/uni00000016
/uni00000018/uni00000011/uni0000001c
/uni00000016/uni00000016/uni00000011/uni00000019/uni00000016/uni00000014/uni00000011/uni00000016/uni00000034/uni0000005a/uni00000048/uni00000051/uni00000015/uni00000011/uni00000018/uni00000010/uni0000001a/uni00000025
/uni00000033/uni00000052/uni00000053/uni00000034/uni00000024
/uni00000037/uni00000055/uni0000004c/uni00000059/uni0000004c/uni00000044/uni00000034/uni00000024
/uni00000031/uni00000044/uni00000057/uni00000058/uni00000055/uni00000044/uni0000004f/uni00000034/uni00000058/uni00000048/uni00000056/uni00000057/uni0000004c/uni00000052/uni00000051/uni00000056 /uni00000015/uni0000003a/uni0000004c/uni0000004e/uni0000004c/uni00000030/uni00000058/uni0000004f/uni00000057/uni0000004c/uni0000002b/uni00000052/uni00000053/uni00000034/uni00000024/uni00000024/uni00000036/uni00000034/uni00000024/uni00000019/uni00000011/uni00000017
/uni00000015/uni00000019/uni00000011/uni00000015
/uni00000014/uni0000001b/uni00000011/uni00000019/uni00000016/uni0000001c/uni00000011/uni00000015/uni00000016/uni0000001b/uni00000011/uni00000014/uni00000018/uni00000011/uni00000018
/uni00000015/uni00000016/uni00000011/uni00000013
/uni00000014/uni00000015/uni00000011/uni00000013
/uni00000016/uni00000017/uni00000011/uni00000018/uni00000016/uni00000018/uni00000011/uni00000018/uni00000018/uni00000011/uni0000001c
/uni00000015/uni00000014/uni00000011/uni00000014
/uni00000014/uni00000014/uni00000011/uni00000016
/uni00000016/uni00000018/uni00000011/uni00000014/uni00000016/uni00000016/uni00000011/uni0000001c/uni00000034/uni0000005a/uni00000048/uni00000051/uni00000015/uni00000011/uni00000018/uni00000010/uni00000014/uni00000017/uni00000025
/uni00000033/uni00000052/uni00000053/uni00000034/uni00000024
/uni00000037/uni00000055/uni0000004c/uni00000059/uni0000004c/uni00000044/uni00000034/uni00000024
/uni00000031/uni00000044/uni00000057/uni00000058/uni00000055/uni00000044/uni0000004f/uni00000034/uni00000058/uni00000048/uni00000056/uni00000057/uni0000004c/uni00000052/uni00000051/uni00000056 /uni00000015/uni0000003a/uni0000004c/uni0000004e/uni0000004c/uni00000030/uni00000058/uni0000004f/uni00000057/uni0000004c/uni0000002b/uni00000052/uni00000053/uni00000034/uni00000024/uni00000024/uni00000036/uni00000034/uni00000024/uni00000018/uni00000011/uni00000018
/uni00000015/uni00000013/uni00000011/uni00000019
/uni00000014/uni00000017/uni00000011/uni00000016
/uni00000016/uni0000001b/uni00000011/uni00000018/uni00000016/uni00000018/uni00000011/uni00000016/uni00000017/uni00000011/uni00000018
/uni00000014/uni00000015/uni00000011/uni0000001a
/uni0000001b/uni00000011/uni00000018
/uni00000016/uni00000014/uni00000011/uni00000015/uni00000016/uni00000016/uni00000011/uni0000001a/uni00000016/uni00000011/uni00000018
/uni00000014/uni00000014/uni00000011/uni00000019
/uni0000001a/uni00000011/uni0000001b
/uni00000016/uni00000014/uni00000011/uni00000013/uni00000016/uni00000017/uni00000011/uni00000016/uni0000002f/uni0000002f/uni00000044/uni00000030/uni00000024/uni00000016/uni00000011/uni00000014/uni00000010/uni0000001b/uni00000025/uni00000032/uni00000058/uni00000055/uni00000056 /uni00000036/uni00000048/uni0000004f/uni00000049/uni00000010/uni00000035/uni00000048/uni00000049/uni0000004c/uni00000051/uni00000048 /uni00000039/uni00000044/uni00000051/uni0000004c/uni0000004f/uni0000004f/uni00000044/uni00000003/uni00000035/uni00000024/uni0000002a
(a) w/o answer.
/uni00000033/uni00000052/uni00000053/uni00000034/uni00000024
/uni00000037/uni00000055/uni0000004c/uni00000059/uni0000004c/uni00000044/uni00000034/uni00000024
/uni00000031/uni00000044/uni00000057/uni00000058/uni00000055/uni00000044/uni0000004f/uni00000034/uni00000058/uni00000048/uni00000056/uni00000057/uni0000004c/uni00000052/uni00000051/uni00000056 /uni00000015/uni0000003a/uni0000004c/uni0000004e/uni0000004c/uni00000030/uni00000058/uni0000004f/uni00000057/uni0000004c/uni0000002b/uni00000052/uni00000053/uni00000034/uni00000024/uni00000024/uni00000036/uni00000034/uni00000024/uni0000001c/uni00000015/uni00000011/uni00000014
/uni0000001c/uni00000019/uni00000011/uni00000013
/uni0000001b/uni0000001b/uni00000011/uni00000019
/uni0000001b/uni00000014/uni00000011/uni00000013/uni0000001b/uni00000018/uni00000011/uni00000015/uni0000001c/uni00000014/uni00000011/uni0000001a
/uni0000001c/uni00000016/uni00000011/uni0000001c
/uni0000001b/uni00000018/uni00000011/uni00000017/uni0000001a/uni00000019/uni00000011/uni00000017/uni0000001a/uni0000001c/uni00000011/uni00000015/uni0000001b/uni0000001c/uni00000011/uni00000016
/uni0000001c/uni00000016/uni00000011/uni00000015
/uni0000001b/uni00000017/uni00000011/uni00000016/uni0000001a/uni00000018/uni00000011/uni00000018/uni0000001a/uni00000019/uni00000011/uni0000001b/uni00000034/uni0000005a/uni00000048/uni00000051/uni00000015/uni00000011/uni00000018/uni00000010/uni0000001a/uni00000025
/uni00000033/uni00000052/uni00000053/uni00000034/uni00000024
/uni00000037/uni00000055/uni0000004c/uni00000059/uni0000004c/uni00000044/uni00000034/uni00000024
/uni00000031/uni00000044/uni00000057/uni00000058/uni00000055/uni00000044/uni0000004f/uni00000034/uni00000058/uni00000048/uni00000056/uni00000057/uni0000004c/uni00000052/uni00000051/uni00000056 /uni00000015/uni0000003a/uni0000004c/uni0000004e/uni0000004c/uni00000030/uni00000058/uni0000004f/uni00000057/uni0000004c/uni0000002b/uni00000052/uni00000053/uni00000034/uni00000024/uni00000024/uni00000036/uni00000034/uni00000024/uni0000001c/uni00000015/uni00000011/uni0000001c
/uni0000001c/uni0000001a/uni00000011/uni00000013
/uni0000001c/uni00000014/uni00000011/uni00000014/uni0000001b/uni00000015/uni00000011/uni00000019/uni0000001b/uni00000019/uni00000011/uni00000015/uni0000001c/uni00000016/uni00000011/uni00000018
/uni0000001c/uni00000019/uni00000011/uni00000014
/uni0000001b/uni0000001c/uni00000011/uni00000013/uni0000001a/uni0000001b/uni00000011/uni00000016/uni0000001b/uni00000015/uni00000011/uni0000001b/uni0000001c/uni00000015/uni00000011/uni00000013
/uni0000001c/uni00000018/uni00000011/uni0000001a
/uni0000001b/uni0000001a/uni00000011/uni00000014/uni0000001a/uni0000001b/uni00000011/uni00000018/uni0000001a/uni0000001b/uni00000011/uni0000001a/uni00000034/uni0000005a/uni00000048/uni00000051/uni00000015/uni00000011/uni00000018/uni00000010/uni00000014/uni00000017/uni00000025
/uni00000033/uni00000052/uni00000053/uni00000034/uni00000024
/uni00000037/uni00000055/uni0000004c/uni00000059/uni0000004c/uni00000044/uni00000034/uni00000024
/uni00000031/uni00000044/uni00000057/uni00000058/uni00000055/uni00000044/uni0000004f/uni00000034/uni00000058/uni00000048/uni00000056/uni00000057/uni0000004c/uni00000052/uni00000051/uni00000056 /uni00000015/uni0000003a/uni0000004c/uni0000004e/uni0000004c/uni00000030/uni00000058/uni0000004f/uni00000057/uni0000004c/uni0000002b/uni00000052/uni00000053/uni00000034/uni00000024/uni00000024/uni00000036/uni00000034/uni00000024/uni0000001c/uni00000015/uni00000011/uni0000001c
/uni0000001c/uni00000018/uni00000011/uni0000001a
/uni0000001b/uni0000001c/uni00000011/uni00000019
/uni0000001b/uni00000013/uni00000011/uni00000017/uni0000001b/uni0000001a/uni00000011/uni00000018/uni0000001c/uni00000015/uni00000011/uni00000019
/uni0000001c/uni00000017/uni00000011/uni00000018
/uni0000001b/uni00000019/uni00000011/uni00000014/uni0000001a/uni00000018/uni00000011/uni00000013/uni0000001a/uni0000001b/uni00000011/uni00000015/uni0000001c/uni00000014/uni00000011/uni00000014
/uni0000001c/uni00000016/uni00000011/uni0000001c
/uni0000001b/uni0000001a/uni00000011/uni00000014/uni0000001a/uni00000018/uni00000011/uni0000001c/uni0000001b/uni00000014/uni00000011/uni00000015/uni0000002f/uni0000002f/uni00000044/uni00000030/uni00000024/uni00000016/uni00000011/uni00000014/uni00000010/uni0000001b/uni00000025/uni00000032/uni00000058/uni00000055/uni00000056 /uni00000036/uni00000048/uni0000004f/uni00000049/uni00000010/uni00000035/uni00000048/uni00000049/uni0000004c/uni00000051/uni00000048 /uni00000039/uni00000044/uni00000051/uni0000004c/uni0000004f/uni0000004f/uni00000044/uni00000003/uni00000035/uni00000024/uni0000002a
(b) w/ answer.
Figure 8: Performance of different methods under Unanswerable (a) and Answerable (b) retrieval conditions. Each
radar chart reports the average performance across three instruction-tuned backbones on five benchmarks.
18

MethodPopQA TriviaQA NQ MultiHopQA ASQA HotpotQA SQuADAvg.(acc) (acc) (acc) (acc) (em) (acc) (acc)
Vanilla RAG
Qwen2.5-7B 63.7 73.2 60.2 44.7 42.8 18.5 9.0 44.6
Qwen2.5-14B 65.3 77.0 63.6 44.8 45.2 23.3 12.6 47.4
LLaMA3.1-8B 65.0 73.4 62.0 43.0 45.2 17.1 6.1 44.5
w/ Frozen CLM
Qwen2.5-7B 67.5 75.1 62.5 45.4 46.7 20.6 13.4 47.3
Qwen2.5-14B 68.0 78.0 65.1 46.6 48.1 25.3 15.6 49.5
LLaMA3.1-8B 66.1 74.1 61.4 42.8 44.7 18.8 8.7 45.2
w/ Trained CLM
Qwen2.5-7B 66.6 76.2 65.4 46.0 47.9 25.2 17.9 49.3
Qwen2.5-14B 67.5 78.9 66.8 47.5 48.7 29.6 21.2 51.5
LLaMA3.1-8B 66.6 75.7 64.4 43.7 47.1 22.9 13.2 47.7
w/ Trained CLM, w/ CCS (Ours)
Qwen2.5-7B 66.0 77.5 66.4 49.9 48.6 33.9 24.2 52.4
Qwen2.5-14B 66.9 79.4 68.6 50.7 49.9 35.5 25.4 53.8
LLaMA3.1-8B 66.6 77.0 66.3 49.6 48.2 32.0 20.7 51.5
Table 4: Details of ablation study on the CLM. Frozen CLM refers to a vanilla LLaMA3.1-8B used as the critic.
CCS refers to our proposed contrastive synthesis.
MethodPopQA TriviaQA NQ MultiHopQA ASQA HotpotQA SQuADAvg.(acc) (acc) (acc) (acc) (em) (acc) (acc)
Qwen2.5-7B
CPO 66.1 76.3 63.5 46.3 47.1 25.6 17.3 48.9
Ours 66.0 77.5 66.4 49.9 48.6 33.9 24.2 52.4
Qwen2.5-14B
CPO 67.5 78.6 66.1 47.4 47.7 29.5 20.2 51.0
Ours 66.9 79.4 68.6 50.7 49.9 35.5 25.4 53.8
LLaMA3.1-8B
CPO 66.4 75.0 62.7 44.0 45.7 23.2 12.8 47.1
Ours 66.6 77.0 66.3 49.6 48.2 32.0 20.7 51.5
Table 5: Overall performance comparison of Critic Language Model using different training methods.
MethodPopQA TriviaQA NQ MultiHopQA ASQAAvg.HotpotQA SQuADAvg. Drop.(acc) (acc) (acc) (acc) (em) (acc) (acc)
Qwen2.5-7B
Vanilla RAG 63.7 73.2 60.2 44.7 42.8 56.9 18.5 9.0 13.8 43.1
Self-Refine 65.5 74.4 61.6 45.0 45.2 58.3 21.3 14.6 18.0 40.3
AlignRAG 66.0 77.5 66.4 49.9 48.6 61.7 33.9 24.2 29.1 32.6
Qwen2.5-14B
Vanilla RAG 65.3 77.0 63.6 44.8 45.2 59.2 23.3 12.6 18.0 41.2
Self-Refine 67.0 78.0 65.1 46.1 47.3 60.7 24.4 16.0 20.2 40.5
AlignRAG 66.9 79.4 68.6 50.7 49.9 63.1 35.5 25.4 30.5 32.6
LLaMA3.1-8B
Vanilla RAG 65.0 73.4 62.0 43.0 45.2 57.7 17.1 6.1 11.6 46.1
Self-Refine 66.1 74.1 61.4 42.8 44.7 57.8 18.8 8.7 13.8 44.0
AlignRAG 66.6 77.0 66.3 49.6 48.2 61.5 32.0 20.7 26.4 35.1
Table 6: Drop in average Out-of-Distribution performance compared to average In-Domain performance across
three instruction-tuned backbones. Lower values indicate bet- ter generalization capability.
19

A.6 Prompt Templates
Critique Synthesis Prompt. We propose a novel
structured pipeline for generating targeted feed-
back to train critic models, systematically deriv-
ing critiques from contrasting outputs of large lan-
guage models (LLMs). To ensure the critiques are
both consistent and informative, we introduce a
preference-augmented input as a key component
in the critique generation process. This approach
is grounded in the use of pairwise comparisons of
reasoning paths, which provides two core innova-
tions. First, it constrains the output space of the
critique language model (CLM), ensuring consis-
tency and minimizing noise during critique gen-
eration (Zhang et al., 2025). Second, it generates
high-quality reasoning traces that facilitate the cre-
ation of constructive, fine-grained feedback. The
pairwise-path formulation is central to this frame-
work: by contrasting the reasoning processes un-
derlying yunexp (unexpected response) and yexp(ex-
pected response), the CLM synthesizes critiques
that directly inform model supervision. This is
exemplified in Tab. 7 (for rationale generation)
and Tab. 9 (for critique generation). This struc-
tured methodology not only enhances the quality
of the generated critiques but also ensures they are
targeted, actionable, and aligned with the require-
ments of improving weaker models.
Critique Learning Prompt. To further advance
critique generation, we introduce the concept of
critique learning , where the objective is to gener-
ate a critique, denoted as ∆yunexp, that captures
the divergence between expected and unexpected
responses while incorporating user-defined pref-
erences. As part of this framework, we present
a novel Critique Fine-Tuning (CFT) prompt (see
Tab. 10 for details) designed to optimize the learn-
ing process for critique generation. Additionally,
we explore an alternative training strategy, Critique
Preference Optimization (CPO) , which explicitly
aligns critique generation with user-defined pref-
erence signals (see Tab. 11 for the corresponding
prompt). These prompts, tailored for critique learn-
ing, establish a principled mechanism for training
models to generate preference-aligned critiques.
Critique-driven Alignment Prompt. We intro-
duce a novel framework, Critique-driven Align-
ment (CDA) , to address reasoning misalignment
in retrieval-augmented generation (RAG) systems.
CDA reimagines the RAG inference process as a
discrete-time dynamical system operating over a la-tent reasoning space Y. Within this framework, the
inference process is iteratively refined by a meta-
reasoning module Mcritic, which critiques interme-
diate outputs and proposes targeted improvements.
This iterative refinement produces a sequence of
progressively improved responses, ensuring reason-
ing alignment.
CDA leverages three distinct prompt types to
structure the refinement pipeline effectively:
•Rationale Generation: Using the rationale
generation template (see Tab. 12), the system
generates an initial explanation or reasoning
chain to support the initial response y0. This
rationale serves as the foundation for critique
generation in subsequent steps.
•Critique Generation: Using the critique
generation template (see Tab. 13), the meta-
reasoning module Mcriticidentifies reasoning
gaps or inconsistencies in the intermediate re-
sponse ytbased on the rationale and provides
an actionable critique ∆yt.
•Refinement Generation: Using the refine-
ment generation template (see Tab. 14), the
system incorporates the critique ∆ytinto the
generation process to produce the refined re-
sponse yt+1. This ensures that the updated re-
sponse aligns with the critique feedback while
maintaining coherence and relevance to the
original query q.
By iteratively applying these three prompts, the
CDA framework introduces a systematic and con-
trolled refinement process that enhances reason-
ing alignment and response quality over successive
iterations. This novel paradigm ensures that cri-
tiques are not only actionable but also effectively
integrated into the refinement process to achieve
consistent improvements in reasoning accuracy.
20

Table 7: Rationale generation prompt template for critique synthesis (Wei et al., 2024).
Rationale Generation for Critique Synthesis
Input: Read the following documents relevant to the given question: {question}
Document [1] (Title: · · ·): {contents}
· · ·
Please identify documents that are useful to answer the given question: “{question}”, and explain how the contents lead to
the answer: {answer}.
If none of the documents is aligned with the answer, in that case, you have to explain the answer only based on
your own knowledge, without referring to the provided information.
{task-specific instruction}
Output: {rationale}
Table 8: Task-specific instruction used in rationale generation prompt (Wei et al., 2024).
Task-specific Instruction for Rationale Generation
ASQA: Note that the question may be ambiguous and have multiple correct answers. Make sure your response includes all
correct answers and provides clear reasoning details followed by a concise conclusion.
PopQA: Note that the question mainly asks about the object entity that holds a certain relationship with the
given subject entity. There may be multiple correct answers. Make sure your response includes all correct answers and
provides clear reasoning details followed by a concise conclusion.
TriviaQA / Natural Questions / 2WikiMultiHopQA: Note that the question may be compositional and require
intermediate analysis to deduce the final answer. Make sure your response is grounded and provides clear reasoning details
followed by a concise conclusion.
Table 9: Critique generation prompt template for critique synthesis.
Critique Generation for Critique Synthesis
Input: Read the following documents relevant to the given question: {question}
Document [1] (Title: · · ·): {contents}
· · ·
Here is the given weak rationale: {weak_rationale}.
Here is the given gold rationale: {gold_rationale}.
First, explain how the gold rationale leads to the answer step by step.
Then, identify the errors and hallucinations of the weak rationale, and give constructive criticism for improving
the weak rationale to be more aligned with the gold rationale.
Output: {critique}
21

Table 10: Augmented critique generation prompt template for critique fine-tuning (CFT).
Augmented Critique Generation for Critique Fine-tuning (CFT)
Input: Read the following documents relevant to the given question: {question}
Document [1] (Title: · · ·): {contents}
· · ·
Here is the given weak rationale: {weak_rationale}.
Please identify the weaknesses and hallucinations of the rationale, and give constructive criticism for improving
the weak rationale.
Output:
The critique for the rationale is: {critique}.
The better rationale should be: {gold_rationale}.
Table 11: Critique generation prompt template for critique preference optimization (CPO).
Augmented Critique Generation for Critique Preference optimization (CPO)
Input: Read the following documents relevant to the given question: {question}
Document [1] (Title: · · ·): {contents}
· · ·
Here is the given weak rationale: {weak_rationale}.
Please identify the weaknesses and hallucinations of the rationale, and give constructive criticism for improving
the weak rationale.
Chosen: The critique for the rationale is: {weak_critique}.
Rejected: The critique for the rationale is: {gold_critique}.
Table 12: Rationale generation prompt template for Critique-driven Alignment.
Rationale Generation for Critique-driven Alignment
Input: Read the following documents relevant to the given question: {question}
Document [1] (Title: · · ·): {contents}
· · ·
Please identify documents that are useful to answer the given question: “{question}”, and explain how the contents lead to
the answer: {answer}.
Output: {rationale}
22

Table 13: Critique generation prompt template for Critique-driven Alignment.
Critique Generation for Critique-driven Alignment
Input: Read the following documents relevant to the given question: {question}
Document [1] (Title: · · ·): {contents}
· · ·
Here is the given weak rationale: {weak_rationale}.
Please identify the weaknesses and hallucinations of the rationale, and give constructive criticism for improving
the weak rationale.
Output: {critique}
Table 14: Refinement generation prompt template for Critique-driven Alignment.
Refinement Generation for Critique-driven Alignment
Input: Read the following documents relevant to the given question: {question}
Document [1] (Title: · · ·): {contents}
· · ·
Here is the given weak rationale: {weak_rationale}.
Here is the given critique: critique.
Please correct the weak rationale based on the critique, and write a better rationale to explain how the contents
lead to the answer.
Output: {refinement}
23