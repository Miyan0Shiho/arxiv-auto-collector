# CDF-RAG: Causal Dynamic Feedback for Adaptive Retrieval-Augmented Generation

**Authors**: Elahe Khatibi, Ziyu Wang, Amir M. Rahmani

**Published**: 2025-04-17 01:15:13

**PDF URL**: [http://arxiv.org/pdf/2504.12560v1](http://arxiv.org/pdf/2504.12560v1)

## Abstract
Retrieval-Augmented Generation (RAG) has significantly enhanced large
language models (LLMs) in knowledge-intensive tasks by incorporating external
knowledge retrieval. However, existing RAG frameworks primarily rely on
semantic similarity and correlation-driven retrieval, limiting their ability to
distinguish true causal relationships from spurious associations. This results
in responses that may be factually grounded but fail to establish
cause-and-effect mechanisms, leading to incomplete or misleading insights. To
address this issue, we introduce Causal Dynamic Feedback for Adaptive
Retrieval-Augmented Generation (CDF-RAG), a framework designed to improve
causal consistency, factual accuracy, and explainability in generative
reasoning. CDF-RAG iteratively refines queries, retrieves structured causal
graphs, and enables multi-hop causal reasoning across interconnected knowledge
sources. Additionally, it validates responses against causal pathways, ensuring
logically coherent and factually grounded outputs. We evaluate CDF-RAG on four
diverse datasets, demonstrating its ability to improve response accuracy and
causal correctness over existing RAG-based methods. Our code is publicly
available at https://github.com/ elakhatibi/CDF-RAG.

## Full Text


<!-- PDF content starts -->

CDF-RAG: Causal Dynamic Feedback for Adaptive Retrieval-Augmented
Generation
Elahe Khatibi*Ziyu Wang*Amir M. Rahmani
University of California, Irvine, USA
{ekhatibi, ziyuw31, a.rahmani}@uci.edu
Abstract
Retrieval-Augmented Generation (RAG) has
significantly enhanced large language models
(LLMs) in knowledge-intensive tasks by incor-
porating external knowledge retrieval. How-
ever, existing RAG frameworks primarily rely
on semantic similarity and correlation-driven
retrieval, limiting their ability to distinguish
true causal relationships from spurious associ-
ations. This results in responses that may be
factually grounded but fail to establish cause-
and-effect mechanisms, leading to incomplete
or misleading insights. To address this is-
sue, we introduce Causal Dynamic Feedback
for Adaptive Retrieval- Augmented Generation
(CDF-RAG), a framework designed to improve
causal consistency, factual accuracy, and ex-
plainability in generative reasoning. CDF-
RAG iteratively refines queries, retrieves struc-
tured causal graphs, and enables multi-hop
causal reasoning across interconnected knowl-
edge sources. Additionally, it validates re-
sponses against causal pathways, ensuring log-
ically coherent and factually grounded out-
puts. We evaluate CDF-RAG on four diverse
datasets, demonstrating its ability to improve
response accuracy and causal correctness over
existing RAG-based methods. Our code is
publicly available at https://github.com/
elakhatibi/CDF-RAG .
1 Introduction
Large language models (LLMs) such as GPT-
4 (Achiam et al., 2023), DeepSeek (Liu et al.,
2024), and LLaMA (Touvron et al., 2023) have
demonstrated strong performance across a range
of reasoning tasks, including fact-based question
answering (Liang et al., 2022), commonsense in-
ference (Huang et al., 2019), and multi-hop re-
trieval (Yang et al., 2018; Zhuang et al., 2024).
Retrieval-Augmented Generation (RAG) (et al.,
*Ziyu Wang and Elahe Khatibi contributed equally to this
work.2020) has been introduced to enhance LLMs by
retrieving external documents, thereby improv-
ing response reliability in knowledge-intensive
tasks (Wei et al., 2024; Li et al., 2024). How-
ever, conventional RAG pipelines typically rely
on static queries and semantic similarity-based re-
trieval, which prioritize topically relevant docu-
ments rather than those that provide explanatory or
causal insights (Jiang et al., 2024; Chi et al., 2024).
While effective for shallow fact recall, these strate-
gies often fall short in tasks requiring multi-step
causal reasoning (Vashishtha et al.; Jin et al., 2023).
This reliance on correlation-driven retrieval in-
troduces key challenges for causality-aware rea-
soning. Traditional RAG systems (as shown in
Figure 1) struggle to distinguish between statisti-
cal associations and true causal relationships (Chi
et al., 2024), leading to retrieved evidence that
may appear relevant but lacks directional or ex-
planatory depth. Furthermore, LLMs trained on
large-scale observational corpora tend to model co-
occurrence patterns rather than causal dependen-
cies, making them prone to conflating correlation
with causation—especially in the presence of in-
complete or ambiguous evidence. These limitations
become more pronounced in multi-hop retrieval,
where linking causally related pieces of informa-
tion is essential for producing coherent reasoning
chains (Zhuang et al., 2024). However, conven-
tional retrieval strategies typically employ flat or
lexical matching techniques, which fail to incorpo-
rate causal structure, leading to responses that are
locally plausible yet globally inconsistent.
Such shortcomings have direct consequences in
real-world applications where causal understand-
ing is critical. In medical decision-making, for in-
stance, associating “high BMI” with “heart disease”
may be factually accurate but incomplete without
identifying mediating factors such as “hyperten-
sion” or “insulin resistance.” When causal evidence
is sparse or incorrectly retrieved, LLMs often com-arXiv:2504.12560v1  [cs.CL]  17 Apr 2025

🔴
 Why does 
diabetes damage 
elderly kidneys? 
Lexical 
Retrieval Static Queries 
Hallucinated & 
Incoherent Responses Diabetes damages elderly 
kidneys by causing 
❌
excessive insulin buildup , 
leading to kidney scarring 
and reduced function… Conventional Fixed RAG Pipelines 
🔴
 Why does 
diabetes damage 
elderly kidneys? 
🟢
 Does diabetes cause 
kidney damage? 
🟢
 How does aging 
impair kidney function? 
🟢
 How does diabetes 
accelerate kidney 
damage in the elderly? Adaptive Query 
Optimization 
Causal-Aware Retrieval
✅
Diabetes can damage the 
kidneys over time due to high 
blood sugar levels, which 
impair blood vessels and 
reduce kidney function. In 
elderly patients, aging… Causal-Consistent 
Generation CDF-RAG (Proposed) Figure 1: Rethinking Retrieval-Augmented Generation (RAG). (a) Traditional RAG pipelines rely on static
queries and keyword- or similarity-based retrieval, often retrieving topically related but causally irrelevant content,
which can result in hallucinated or incoherent outputs. (b) CDF-RAG addresses these limitations through rein-
forcement learning-based query refinement, dual-path retrieval combining semantic vector search with causal graph
traversal, and causal-consistent generation, leading to improved factuality and reasoning.
pensate by hallucinating plausible-sounding but
unsupported explanations (Sun et al., 2024; Yu
et al., 2024), reducing trustworthiness. Addition-
ally, static query formulation prevents models from
adapting retrieval based on reasoning gaps, fur-
ther exacerbating these issues. While recent work
has explored structured retrieval (Jin et al., 2024),
multi-hop planning (Ferrando et al., 2024), and
causal graph construction (et al., 2024), these ap-
proaches address isolated components rather than
providing an end-to-end framework for causal rea-
soning.
Another key challenge in causal question an-
swering is that many user queries in QA are
also vague or underspecified, making effective re-
trieval even more challenging. While methods like
RQ-RAG (Chan et al., 2024), RAG-Gym (et al.,
2025), and SmartRAG (Gao et al., 2024) intro-
duce query refinement or agentic retrieval mech-
anisms, they lack dynamic adaptation and causal
alignment—often retrieving shallow or loosely con-
nected content. This highlights the need for refine-
ment strategies that are explicitly optimized for
causal reasoning.
To address these challenges, we propose Causal
Dynamic Feedback for Retrieval- Augmented
Generation (CDF-RAG) , a novel framework that
integrates reinforcement-learned query refinement,
multi-hop causal graph retrieval, and alignment-
based hallucination detection into a dynamic rea-
soning loop. These components enable CDF-RAG
to retrieve causally relevant evidence and gener-
ate logically coherent responses grounded in causal
structures. Our experiments on CosmosQA (Huang
et al., 2019), MedQA (Jin et al., 2020), MedM-
CQA (Pal et al., 2022), and AdversarialQA (Bar-tolo et al., 2020) show that CDF-RAG consis-
tently outperforms standard and refined RAG mod-
els (Chan et al., 2024; et al., 2025) across key
metrics—demonstrating its effectiveness for gen-
erating factually consistent and causally coherent
responses in complex QA settings– providing a
robust foundation for trustworthy reasoning in real-
world applications.
Contributions. Our paper makes the following
contributions:
•We introduce CDF-RAG , a unified framework
that integrates causal query refinement, multi-
hop causal graph retrieval, and hallucination
detection into a dynamic feedback loop for
causality-aware generation.
• We demonstrate that our reinforcement learn-
ing (RL)-based query rewriting significantly
enhances multi-hop causal reasoning and re-
trieval quality, outperforming prior refinement
approaches.
•We show that CDF-RAG achieves state-of-the-
art performance on four QA benchmarks, with
consistent improvements in causal correctness,
consistency, and interpretability over existing
RAG-based models.
2 CDF-RAG: Causal Dynamic Feedback
for RAG
We introduce CDF-RAG , a causality-aware exten-
sion of RAG. As illustrated in Figure 2, the sys-
tem refines user queries via a query refinement
LLM trained with RL, retrieves knowledge using a
dual-path retrieval mechanism, rewrites knowledge,

and applies a causal graph check to ensure fac-
tual consistency. By integrating structured causal
reasoning, CDF-RAG mitigates hallucinations and
enhances interpretability. This approach enables
dynamic query adaptation and precise retrieval for
causal reasoning tasks. Implementation details can
be found in Appendix A
2.1 Causal Knowledge Graph Construction
CDF-RAG constructs a directed causal knowledge
graph G= (V, E)from textual data to capture
causal dependencies beyond correlation. Using
UniCausal (Tan et al., 2023), a BERT-based classi-
fier extracts cause-effect pairs formatted as C→E,
processing annotated inputs <ARG0> and<ARG1> to
predict ˆy=g(r[CLS]).
To ensure logical validity, extracted causal pairs
are verified by GPT-4 before being encoded into
Gas directed triples (C, E, relation ). The graph
structure enables multi-hop reasoning over causal
mechanisms, ensuring retrieved knowledge sup-
ports causal inference tasks.
2.2 Causal Query Refinement via
Reinforcement Learning
Given an initial user query q, CDF-RAG applies
RL to generate a refined query ˆqoptimized for
causal retrieval. The RL-based query refinement
agent models this as a Markov Decision Process
(MDP), where the state srepresents the query
embedding, and the agent selects an action a∈
{expand ,simplify ,decompose }. Expansion en-
hances specificity by adding relevant causal fac-
tors, simplification removes extraneous details, and
decomposition restructures complex queries into
atomic subqueries.
The policy πθ(a|s)is initialized via supervised
fine-tuning (SFT) on labeled refinement examples:
LSFT=−TX
t=1logPϕ(yt|y<t, x)
and further optimized using Proximal Policy Opti-
mization (PPO) (Schulman et al., 2017):
LPPO(θ) =Et
min 
rt(θ)ˆAt,
clip(rt(θ),1−ϵ,1 +ϵ)ˆAt
where rt(θ) =πθ(at|st)
πθold(at|st).
The reward function optimizes retrieval effec-
tiveness and causal consistency:R=λ1·RetrievalCoverage
+λ2·CausalDepth
+λ3·ContextRelevance
−λ4·HallucinationPenalty
By refining queries with these criteria, CDF-
RAG dynamically adapts retrieval strategies to en-
hance causal reasoning.
2.3 Dual-Path Retrieval: Semantic and
Causal Reasoning
To ensure comprehensive and aligned knowledge
access, CDF-RAG adopts a dual-path retrieval strat-
egy, integrating semantic vector search with causal
graph traversal.
Semantic Vector Retrieval. Inspired by dense
retrieval methods (Karpukhin et al., 2020), we en-
code the refined query ˆqusing MiniLM (Wang
et al., 2020) and perform similarity search in a
vector database. This semantic retrieval pathway
returns top- kpassages Tsemthat offer contextual
evidence supporting the query. Unlike sparse re-
trieval methods such as BM25 (Robertson et al.,
2009), which rely on term frequency heuristics,
our approach captures contextual relevance through
transformer-based embeddings. This enables richer
matching, particularly for lexically divergent yet se-
mantically similar phrases—a common challenge
in biomedical and causal reasoning tasks.
Causal Graph Traversal. To complement se-
mantic retrieval with structural reasoning, we tra-
verse a domain-specific causal graph G. Given ˆq,
we identify aligned nodes and expand along di-
rected edges to surface causally linked variables or
events. The resulting paths Cgraph expose mediators,
confounders, and downstream effects aligned with
the query’s underlying causal semantics.
Unified Knowledge Set. We denote the final
retrieved knowledge as K=Tsem∪ C graph. This
hybrid set blends semantic relevance with causal
coherence, enabling downstream modules to gener-
ate grounded and causally faithful responses.
2.4 Response Generation and Causal Graph
Check
CDF-RAG generates a response ˆyconditioned
onKusing a language model. To ensure gener-
ated content remains faithful to causal principles,
we implement a Causal Graph Check , verifying
whether retrieved evidence supports the generated

               Whether ARG0 causes ARG1? 
Causal Pair Construction 
Causal 
Refinement 
 
Document Upload 
Query Input BERT<ARG0>...</ARG0> 
Cause → Effect 
MiniLM
[-0.017, 0.028, -0.032, ...] 
Document Embeddings VectorDB UniCausal 
[Expand, Simplify, 
Decompose] 
Deciding Actions 🔴
 Why does diabetes cause 
kidney damage in elderly 
patients? 
🟢
 Can diabetes cause kidney 
damage? 
🟢
 Why does aging contribute to 
kidney damage? 
🟢
 How does diabetes-related 
kidney damage progress in 
elderly patients? Refined 
Query
Causal Graph 
 
Causal Retrieval
Semanticl RetrievalKnowledge 
Rewriting
Document
Causal
Diabetes in elderly patients leads 
to kidney damage due to chronic 
hyperglycemia, ❌
which directly  
causes kidney failure without  
diabetic nephropathy . Aging 
weakens kidney function… 
Answer 
Draft 
✅
Diabetes in elderly patients 
increases the risk of kidney 
damage due to chronic 
hyperglycemia, which gradually  
leads to glomerular damage then  
diabetic nephropathy.  Aging 
contributes… Detected 
Hallucination 
Causal Graph 
Check 
Final
AnswerCorrected 
Answer 
 Reinforcement 
Learning (RL) 
Decision 
Environment 
Causal & 
Semantic 
Retrieval 
State 
Query Input Action 
↣ Expand 
↣ Simplify 
↣ Decompose LoRA-SFT Query 
Refinement LLM 
Query 
Refinement 
LLM
Executing 
CDF-RAG  
Pipeline… Final
AnswerReward 
↣ Retrieval Coverage 
↣ Causal Depth 
↣ Context Relevance 
↣ Hallucination Penalty 
PPO Policy Update User 
Initial 
Query
Retrieved 
Knowledge
(a) CDF-RAG Pipeline 
(b) Query Refinement Agent Figure 2: Overview of CDF-RAG Framework. (a) The CDF-RAG pipeline refines user queries (LLM + RL),
retrieves structured causal and unstructured textual knowledge, applies knowledge rewriting, and ensures factual
consistency through causal verification. (b) The PPO-trained query refinement agent optimizes retrieval coverage
and causal consistency.
causal claims. The verification process computes a
causal consistency score:
Scausal =1
|Cgraph|X
(C,E )∈CgraphI(C→E|= ˆy),
where Iis an indicator function that checks if the
causal relation is maintained in the generated re-
sponse.
IfScausal< τ, where τis a predefined threshold,
the system triggers Fallback Generation , prompt-
ing the LLM to regenerate ˆyunder stricter ground-
ing constraints:
ˆy′= arg max
yP(y| K,strict constraints ).
This ensures that the final response aligns with re-
trieved causal knowledge, reducing inconsistencies
and hallucinations.
2.5 Hallucination Detection and Correction
CDF-RAG detects hallucinations by evaluating the
logical consistency between ˆyandK. A hallucina-
tion score is computed as:
Shallucination = 1−|K ∩ Y|
|Y|,
whereYrepresents extracted claims from ˆyandK
represents retrieved knowledge. If Shallucination >
δ, where δis a predefined threshold, the system
applies knowledge rewriting:
ˆy′′= arg max
yP(y| K,rewriting constraints ).This correction mechanism ensures that causal con-
sistency is enforced without altering the base LLM
capabilities, preserving factual correctness in gen-
erated responses.
3 Experiments
3.1 Evaluation Tasks
We evaluate the effectiveness of CDF-RAG across
both single-hop and multi-hop question answering
(QA) tasks that require varying levels of causal rea-
soning and knowledge integration. Our evaluations
span four benchmark datasets: CosmosQA (Huang
et al., 2019), MedQA (Jin et al., 2020), MedM-
CQA (Pal et al., 2022), and AdversarialQA (Bar-
tolo et al., 2020). CosmosQA and MedQA assess
commonsense and domain-specific causal reason-
ing, while MedMCQA and AdversarialQA test
multi-hop and cross-document reasoning.
3.2 Baselines
We compare CDF-RAG against three categories of
baselines:
Standard RAG Methods: These include con-
ventional RAG pipelines using semantic retrieval
(BM25 (Robertson et al., 2009)/DPR (Karpukhin
et al., 2020)) without causal enhancement. We
also consider Smart-RAG (Gao et al., 2024) and
Causal-RAG (Wang et al., 2025) as stronger vari-
ants equipped with heuristic multi-hop capabilities
and causal priors.

Refined Query Methods: We compare to Gym-
RAG (et al., 2025) and RQ-RAG (Chan et al., 2024),
which leverage query refinement strategies to im-
prove retrieval quality. These serve as important
baselines for assessing our reinforcement-based
causal query rewrites.
Graph-Augmented Models (G-LLMs): We
compare against a recent graph-augmented LLM
framework (Luo et al., 2025) that integrates causal
filtering and chain-of-thought–driven retrieval over
large knowledge graphs. This method, known
as Causal-First Graph RAG, prioritizes cause-
effect relationships and dynamically aligns retrieval
with intermediate reasoning steps, improving inter-
pretability and accuracy on complex medical QA
tasks.
All methods are evaluated under consistent re-
triever and generation configurations for fair com-
parison. We report results using multiple LLM
backbones—including GPT-4 (OpenAI, 2023),
LLaMA 3-8B (Touvron et al., 2023), Mistral (Jiang
et al., 2023), and Flan-T5 (Chung et al., 2024)—to
demonstrate model-agnostic improvements. GPT-4
is accessed via the OpenAI API, while the remain-
ing models are fine-tuned on our curated multi-task
dataset using the same training and decoding pa-
rameters to ensure alignment in evaluation settings.
Additional details on the experimental setup are
provided in Appendix B.
3.3 Metrics
We evaluate CDF-RAG using both standard answer
quality metrics and specialized measures tailored
to causal reasoning and retrieval. Classical QA
metrics such as accuracy, precision, recall, and F1
score assess the correctness of the final answer. To
complement them, we include Context Relevance,
which quantifies the semantic alignment between
the user query and the retrieved content using av-
erage cosine similarity between Sentence-BERT
embeddings (Reimers and Gurevych, 2019). This
reflects how lexically and topically well-aligned
the retrieved evidence is with the original question.
To assess the causal robustness of the retrieval pro-
cess, we report Causal Retrieval Coverage (CRC).
CRC reflects the system’s ability to prioritize cause-
effect evidence over loosely related or semantically
correlated content, and serves as a proxy for the
quality of causal grounding in the retrieval phase.
Finally, we report Groundedness, which evaluates
whether the generated answer is explicitly sup-
ported by the retrieved content. This metric re-flects factual consistency and plays a critical role
in identifying hallucination-prone behaviors.
Additional metrics and results used in our
study—are reported in Appendix B.4 and provide
further insight into the causal reasoning depth, per-
formance, and robustness of the pipeline.
4 Results and Analysis
This section presents a detailed empirical evalu-
ation of CDF-RAG across four benchmark QA
datasets and multiple language model backbones.
We compare its performance against existing RAG
baselines using standard QA metrics as well as
causal and contextual metrics that reflect reasoning
depth, evidence grounding, and factual reliability.
4.1 Accuracy Performance
We report accuracy results in Table 1 across
four benchmark QA datasets— CosmosQA (Huang
et al., 2019), AdversarialQA (Bartolo et al., 2020),
MedQA (Jin et al., 2020), and MedMCQA (Pal et al.,
2022)—evaluated on four language model back-
bones: GPT-4 (OpenAI, 2023), LLaMA 3-8B (Tou-
vron et al., 2023), Mistral (Jiang et al., 2023), and
Flan-T5 (Chung et al., 2024). Accuracy serves
as a fundamental metric for determining whether
generated responses match ground-truth answers.
This is particularly important in biomedical do-
mains, where factual correctness can directly im-
pact decision-making.
Across all datasets and models, CDF-RAG
achieves the highest accuracy scores, demonstrat-
ing its generalizability across reasoning types (com-
monsense, adversarial, and biomedical) and model
scales. On MedMCQA, for example, CDF-RAG
attains 0.94 accuracy with GPT-4, and 0.90 with
LLaMA 3-8B, outperforming the strongest base-
line, Gym-RAG, by 16% and 13% respectively.
Similar gains are seen across CosmosQA and Ad-
versarialQA, highlighting CDF-RAG’s robustness
in both open-domain and medically grounded QA
tasks.
CDF-RAG’s improvements can be attributed to
its carefully integrated architecture. First, high-
quality causal pairs are extracted and validated us-
ing a GPT-4 assisted pipeline and stored in a Neo4j
graph, enabling directionally-aware, multi-hop re-
trieval. Unlike semantic retrievers that focus on
surface-level similarity, the graph captures deeper
cause-effect dependencies that are essential for ex-
planatory reasoning.

Table 1: Accuracy Scores of various RAG methods
across datasets and LLM backbones.
Dataset Method GPT-4 LLaMA 3-8B Mistral Flan-T5
AdversarialQACDF-RAG 0.89 0.83 0.81 0.79
Gym-RAG 0.78 0.75 0.73 0.70
RQ-RAG 0.76 0.71 0.72 0.66
Smart-RAG 0.74 0.73 0.70 0.64
Causal RAG 0.71 0.71 0.66 0.62
G-LLMs 0.68 0.68 0.65 0.60
CosmosQACDF-RAG 0.89 0.88 0.85 0.84
Gym-RAG 0.82 0.80 0.75 0.73
RQ-RAG 0.80 0.79 0.74 0.72
Smart-RAG 0.78 0.77 0.72 0.70
Causal RAG 0.76 0.75 0.70 0.68
G-LLMs 0.73 0.72 0.68 0.66
MedQACDF-RAG 0.92 0.89 0.88 0.84
Gym-RAG 0.83 0.79 0.78 0.73
RQ-RAG 0.82 0.78 0.77 0.72
Smart-RAG 0.81 0.77 0.76 0.71
Causal RAG 0.79 0.75 0.74 0.69
G-LLMs 0.76 0.72 0.71 0.67
MedMCQACDF-RAG 0.94 0.90 0.88 0.85
Gym-RAG 0.78 0.77 0.76 0.72
RQ-RAG 0.76 0.75 0.74 0.70
Smart-RAG 0.74 0.73 0.72 0.68
Causal RAG 0.72 0.71 0.70 0.66
G-LLMs 0.68 0.68 0.66 0.63
Second, query refinement is performed via a
PPO-trained RL agent, which selects between de-
composition and expansion strategies. This re-
finement aligns the query structure with latent
causal chains in the graph and vector database.
Importantly, all models except GPT-4 are multi-
task instruction fine-tuned on a carefully curated
dataset encompassing decomposition, simplifica-
tion, and expansion tasks (or any combinations in a
feedback loop)—enabling consistent, controllable
query rewriting across backbones.
Compared to other methods, CDF-RAG offers a
more coherent and complete reasoning stack. Gym-
RAG performs well due to its reward-guided tra-
jectory optimization, but it lacks causal grounding
in its retrieval process and does not validate final
outputs, leading to gaps in factual correctness. RQ-
RAG uses static rule-based query rewriting, which
improves retrieval over raw queries, but it is not
adaptive and does not distinguish between causal
and associative evidence, limiting its effectiveness
on multi-hop queries.
Smart-RAG includes an RL policy to coordinate
retrieval and generation steps. However, it lacks
access to causal graphs and performs no verifica-
tion of output consistency. Its reliance on semantic
retrieval alone results in shallow, often incomplete,
reasoning. Causal RAG, while integrating causal
paths, depends on weak summarization-based ex-
traction methods, leading to noisy graph construc-
tion. It does not dynamically refine queries or fil-
ter hallucinations. G-LLMs use static knowledge
graphs but are not designed to support causal re-trieval or adaptive refinement, resulting in the low-
est accuracy across all configurations.
In contrast, CDF-RAG’s feedback loop between
query refinement, causal retrieval, and answer veri-
fication ensures end-to-end causal alignment. This
alignment not only improves retrieval coverage but
also helps the model generate answers that are more
accurate and grounded in factually valid causal
pathways.
These results emphasize the need for RAG
frameworks to go beyond surface-level semantic
retrieval and incorporate causal structure, dynamic
reasoning strategies, and output validation. CDF-
RAG embodies these principles, resulting in sub-
stantial improvements in accuracy across datasets
and model architectures.
4.2 Retrieval and Contextual Performance
To evaluate the quality of retrieval and its down-
stream impact on answer faithfulness, we report
three upstream metrics: CRC, Context Relevance,
and Groundedness. Together, these metrics assess
how well the system identifies causally relevant
content, aligns it semantically with the user query,
and generates factually supported answers.
CRC measures the proportion of retrieved ele-
ments—including causal triples from the Neo4j
graph and unstructured passages from the vector
database—that belong to a verified causal path
aligned with the query. For each query, CRC is
computed by checking whether the retrieved items
match entries in a gold-standard causal graph con-
structed using GPT-4 verification. This metric re-
flects the system’s ability to prioritize directional,
explanatory content over semantically correlated
but causally irrelevant material.
Context Relevance captures the semantic align-
ment between the user query and the retrieved con-
tent. We compute this by encoding both the query
and the top-k retrieved items using Sentence-BERT
embeddings (Reimers and Gurevych, 2019), fol-
lowed by averaging cosine similarity scores across
the retrieved set. While CRC emphasizes struc-
tural fidelity, Context Relevance ensures that the
retrieved material is lexically and topically close to
the query, which is particularly useful in guiding
generation during early inference steps.
Groundedness evaluates the factual consistency
between the generated answer and the retrieved ev-
idence. It assesses whether the key claims made in
the answer are explicitly supported by the retrieved
content, ensuring that the response is not only flu-

Table 2: Retrieval and Contextual Metrics ofCDF-RAG across models and methods.
CRC = Causal Retrieval Coverage, Context = Context Relevance.
Dataset Model CDF-RAG Gym-RAG RQ-RAG Smart-RAG Causal RAG G-LLMs
CRC
AdversarialQAGPT-4 0.89 0.80 0.77 0.74 0.72 0.68
LLaMA 3-8B 0.85 0.76 0.73 0.70 0.68 0.64
Mistral 0.82 0.74 0.71 0.68 0.66 0.61
Flan-T5 0.78 0.70 0.67 0.64 0.62 0.59
CosmosQAGPT-4 0.91 0.81 0.79 0.77 0.74 0.71
LLaMA 3-8B 0.87 0.78 0.76 0.74 0.71 0.68
Mistral 0.86 0.79 0.77 0.75 0.72 0.69
Flan-T5 0.82 0.75 0.73 0.71 0.69 0.66
MedMCQAGPT-4 1.00 0.93 0.90 0.88 0.85 0.82
LLaMA 3-8B 0.98 0.89 0.86 0.84 0.81 0.78
Mistral 0.96 0.91 0.88 0.86 0.83 0.80
Flan-T5 0.95 0.87 0.84 0.82 0.79 0.76
Context (Context Relevance)
AdversarialQAGPT-4 0.76 0.67 0.64 0.62 0.60 0.56
LLaMA 3-8B 0.73 0.65 0.63 0.60 0.58 0.54
Mistral 0.70 0.63 0.60 0.58 0.56 0.52
Flan-T5 0.68 0.60 0.58 0.56 0.53 0.50
CosmosQAGPT-4 0.78 0.69 0.67 0.66 0.63 0.60
LLaMA 3-8B 0.75 0.67 0.65 0.63 0.61 0.58
Mistral 0.74 0.67 0.65 0.64 0.62 0.59
Flan-T5 0.72 0.64 0.62 0.61 0.59 0.56
MedMCQAGPT-4 0.64 0.60 0.58 0.56 0.54 0.51
LLaMA 3-8B 0.62 0.58 0.56 0.54 0.52 0.49
Mistral 0.63 0.59 0.57 0.55 0.53 0.50
Flan-T5 0.61 0.57 0.55 0.53 0.50 0.47
ent but also verifiable. To compute this metric, we
apply a span-level alignment approach that checks
whether the answer content can be traced back to
specific supporting phrases or structures within the
retrieved passages or causal triples. Grounded-
ness is crucial because a model may retrieve high-
quality context yet still introduce hallucinations or
unsupported causal links during generation. This
metric reflects the degree to which the model faith-
fully uses its retrieved inputs, serving as a proxy
for factual reliability and evidence-grounded rea-
soning.
As shown in Table 2, CDF-RAG consistently
achieves the highest CRC and Context Relevance
across all datasets and LLMs. For example, on
AdversarialQA with GPT-4, CDF-RAG attains a
CRC of 0.89 and Context score of 0.76, while Gym-
RAG and RQ-RAG score 0.80/0.67 and 0.77/0.64,
respectively. On MedMCQA, CDF-RAG achieves
perfect causal coverage (CRC = 1.00) and the high-
est semantic alignment (Context = 0.64). These
metrics explain its ability to retrieve both relevant
and causally grounded information.
Figure 3 presents the Groundedness compar-
ison across four LLMs on the MedQA dataset.
CDF-RAG outperforms all baselines across ev-
ery model backbone, achieving a groundedness
score of 0.67–0.65, depending on the LLM. The
improvement is especially notable with GPT-4 and
GPT-4 Mistral LLaMA 3-8B Flan-T5
Model0.400.450.500.550.600.650.700.75GroundednessCDF-RAG
Gym-RAG
RQ-RAG
Smart-RAG
Causal RAG
G-LLMsFigure 3: Groundedness comparison of different meth-
ods across four LLMs on the MedQA dataset.
LLaMA 3-8B, where the margin over Gym-RAG
and Smart-RAG is over 7%. This gain highlights
the benefit of our hallucination-aware verification
loop and causally coherent retrieval. By integrat-
ing RL-refined queries, causal graph traversal, and
structured rewriting, CDF-RAG ensures that gen-
eration remains closely tied to verifiable, context-
supported content. In contrast, methods such as
G-LLMs and Causal-RAG either lack semantic
adaptation or perform shallow causal reasoning,
resulting in lower groundedness and higher suscep-
tibility to hallucinations. Together, these results
confirm that CDF-RAG’s retrieval pipeline is both
structurally precise and semantically aligned, lead-
ing to answers that are not only accurate but also
causally and contextually grounded.

Baseline RAG
+ RL-based Query Refinement+ Causal Graph+ Rewriter
+ Hallucination Correction0.50.60.70.80.91.0Score
CDF-RAG Ablation Study: Performance Metrics
CRC
SRS
Groundedness
F1
Baseline RAG
+ RL-based Query Refinement+ Causal Graph+ Rewriter
+ Hallucination Correction0.0000.0250.0500.0750.1000.1250.1500.1750.200Rate
Hallucination Rate (Lower is Better)
Hallucination Rate (HR)Figure 4: Ablation study of CDF-RAG across incremental stages. Left: performance metrics including CRC, SRS,
groundedness, and F1 score. Right: HR, where lower values indicate greater factual consistency.
4.3 Ablation Study
To evaluate the contribution of each component
in the CDF-RAG framework, we conduct a step-
wise ablation study by incrementally enabling key
modules. We begin with a baseline RAG setup
that uses semantic vector retrieval via MiniLM,
followed by LLM-based generation, without incor-
porating query refinement or structural retrieval
mechanisms. We then progressively add RL-based
query refinement, causal graph retrieval, structured
knowledge rewriting, and hallucination correction.
Notably, query refinement is only triggered when
the RL agent is enabled, and no static prompt engi-
neering is applied at any stage.
Each configuration is evaluated on six metrics:
CRC, causal chain depth (CCD), semantic refine-
ment score (SRS), groundedness, hallucination rate
(HR), and F1. CCD measures the average num-
ber of directed hops in retrieved causal paths from
the Neo4j graph. SRS is computed as the cosine
similarity between the original and refined queries,
quantifying semantic alignment. Groundedness re-
flects the coherence between retrieved knowledge
and generated responses, using sentence embed-
ding similarity. HR denotes the percentage of re-
sponses flagged as hallucinated by the LLM veri-
fier.
F1 captures the balance of precision and recall
based on overlap with reference answers. As shown
in Table 3, each added component improves overall
system performance. For example, enabling the
causal graph module increases CCD from 1.70 to
1.92 by exposing deeper multi-hop pathways, while
the hallucination verifier further reduces HR to 0.07
and improves groundedness to 0.71. The final stage
values match those reported in the main resultsTable 3: Ablation study on the CDF-RAG framework.
Each stage adds a core module, demonstrating consis-
tent gains across CRC, CCD, SRS, groundedness, and
F1 score. HR reflects improved factual reliability.
Ablation Stage CRC CCD SRS Groundedness HR F1
Baseline RAG 0.74 1.50 0.55 0.52 0.18 0.68
+ RL-based Query
Refinement0.80 1.70 0.62 0.59 0.14 0.74
+ Causal Graph 0.84 1.92 0.65 0.63 0.12 0.78
+ Rewriter 0.88 2.00 0.70 0.68 0.08 0.82
+Hallucination
Correction (Ours)0.89 2.02 0.74 0.71 0.07 0.86
table, confirming that each module contributes both
independently and synergistically to the robustness
and reliability of CDF-RAG’s causal reasoning.
Figure 4 presents the results of our ablation study
over the CDF-RAG framework, highlighting the
incremental effect of each component. As we pro-
gressively add RL-based query refinement, causal
graph retrieval, structured rewriting, and hallucina-
tion correction, we observe consistent gains across
core metrics such as CRC, SRS, groundedness, and
F1. The HR shows a marked decline, reflecting
enhanced factual reliability at each stage. These
results underscore the modular design and cumula-
tive benefit of CDF-RAG’s causally grounded and
agentic reasoning architecture.
5 Conclusion
In this paper, we introduce CDF-RAG, a
causality-aware RAG framework that integrates
reinforcement-learned query refinement, multi-hop
causal graph retrieval, and hallucination detection
into a dynamic feedback loop. By aligning retrieval
with causal structures and enforcing consistency in
generation, CDF-RAG enhances factual accuracy
and reasoning depth. Our evaluations demonstrate

state-of-the-art performance across four QA bench-
marks, surpassing existing RAG methods in causal
correctness and reliability. These results highlight
the effectiveness of structured causal reasoning for
adaptive retrieval-augmented generation.Limitations
While CDF-RAG demonstrates improvements in
retrieval precision and response coherence through
causal query refinement, several limitations remain.
First, the method depends on access to structured
causal graphs, which may not be readily available
or complete in all domains, particularly those with
sparse or noisy causal knowledge. This reliance
could limit applicability in open-domain or low-
resource settings. Second, the hallucination detec-
tion module employs GPT-based validation, which,
despite its effectiveness, incurs significant compu-
tational overhead. This may hinder deployment
in real-time or resource-constrained environments.
Finally, although our reinforcement learning frame-
work enables adaptive query refinement, its general-
ization to highly heterogeneous or informal queries
requires further investigation. Addressing these
limitations is essential for broader applicability and
efficiency in practical settings.
References
Josh Achiam, Steven Adler, Sandhini Agarwal, Lama
Ahmad, Ilge Akkaya, Florencia Leoni Aleman,
Diogo Almeida, Janko Altenschmidt, Sam Altman,
Shyamal Anadkat, and 1 others. 2023. Gpt-4 techni-
cal report. arXiv preprint arXiv:2303.08774 .
Max Bartolo, Alastair Roberts, Johannes Welbl, Sebas-
tian Riedel, and Pontus Stenetorp. 2020. Beat the ai:
Investigating adversarial human annotation for read-
ing comprehension. Transactions of the Association
for Computational Linguistics , 8:662–678.
Chi-Min Chan, Chunpu Xu, Ruibin Yuan, Hongyin Luo,
Wei Xue, Yike Guo, and Jie Fu. 2024. Rq-rag: Learn-
ing to refine queries for retrieval-augmented genera-
tion. arXiv preprint arXiv:2404.00610 .
Haoang Chi, He Li, Wenjing Yang, Feng Liu, Long Lan,
Xiaoguang Ren, Tongliang Liu, and Bo Han. 2024.
Unveiling causal reasoning in large language models:
Reality or mirage? Advances in Neural Information
Processing Systems , 37:96640–96670.
Hyung Won Chung, Le Hou, Shayne Longpre, Barret
Zoph, Yi Tay, William Fedus, Yunxuan Li, Xuezhi
Wang, Mostafa Dehghani, Siddhartha Brahma, and
1 others. 2024. Scaling instruction-finetuned lan-
guage models. Journal of Machine Learning Re-
search , 25(70):1–53.
Chamod Samarajeewa et al. 2024. Causal reason-
ing in large language models using causal graph
retrieval-augmented generation. arXiv preprint
arXiv:2410.11414 .

Guangzhi Xiong et al. 2025. Rag-gym: Optimizing
reasoning and search agents with process supervision.
arXiv preprint arXiv:2502.13957 .
Patrick Lewis et al. 2020. Retrieval-augmented gen-
eration for knowledge-intensive nlp tasks. arXiv
preprint arXiv:2005.11401 .
Javier Ferrando, Oscar Obeso, Senthooran Rajamanoha-
ran, and Neel Nanda. 2024. Do i know this entity?
knowledge awareness and hallucinations in language
models. arXiv preprint arXiv:2411.14257 .
Jingsheng Gao, Linxu Li, Weiyuan Li, Yuzhuo Fu, and
Bin Dai. 2024. Smartrag: Jointly learn rag-related
tasks from the environment feedback. arXiv preprint
arXiv:2410.18141 .
Lifu Huang, Ronan Le Bras, Chandra Bhagavatula, and
Yejin Choi. 2019. Cosmos qa: Machine reading com-
prehension with contextual commonsense reasoning.
arXiv preprint arXiv:1909.00277 .
Albert Q. Jiang, Alexandre Sablayrolles, Arthur Men-
sch, Chris Bamford, Devendra Singh Chaplot, Diego
de las Casas, Florian Bressand, Gianna Lengyel, Guil-
laume Lample, Lucile Saulnier, Lélio Renard Lavaud,
Marie-Anne Lachaux, Pierre Stock, Teven Le Scao,
Thibaut Lavril, Thomas Wang, Timothée Lacroix,
and William El Sayed. 2023. Mistral 7b. Preprint ,
arXiv:2310.06825.
Pengcheng Jiang, Cao Xiao, Minhao Jiang, Parminder
Bhatia, Taha Kass-Hout, Jimeng Sun, and Jiawei Han.
2024. Reasoning-enhanced healthcare predictions
with knowledge graph community retrieval. arXiv
preprint arXiv:2410.04585 .
Bowen Jin, Jinsung Yoon, Jiawei Han, and Sercan O
Arik. 2024. Long-context llms meet rag: Overcom-
ing challenges for long inputs in rag. In The Thir-
teenth International Conference on Learning Repre-
sentations .
Di Jin, Eileen Pan, Nassim Oufattole, Wei-Hung Weng,
Hanyi Fang, and Peter Szolovits. 2020. What dis-
ease does this patient have? a large-scale open do-
main question answering dataset from medical exams.
arXiv preprint arXiv:2009.13081 .
Zhijing Jin, Jiarui Liu, Zhiheng Lyu, Spencer Poff, Mrin-
maya Sachan, Rada Mihalcea, Mona Diab, and Bern-
hard Schölkopf. 2023. Can large language models
infer causation from correlation? arXiv preprint
arXiv:2306.05836 .
Vladimir Karpukhin, Barlas Oguz, Sewon Min,
Patrick SH Lewis, Ledell Wu, Sergey Edunov, Danqi
Chen, and Wen-tau Yih. 2020. Dense passage re-
trieval for open-domain question answering. In
EMNLP (1) , pages 6769–6781.
Elahe Khatibi, Mahyar Abbasian, Zhongqi Yang, Iman
Azimi, and Amir M Rahmani. 2024. Alcm: Au-
tonomous llm-augmented causal discovery frame-
work. arXiv preprint arXiv:2405.01744 .Xinze Li, Sen Mei, Zhenghao Liu, Yukun Yan,
Shuo Wang, Shi Yu, Zheni Zeng, Hao Chen,
Ge Yu, Zhiyuan Liu, and 1 others. 2024. Rag-
ddr: Optimizing retrieval-augmented generation us-
ing differentiable data rewards. arXiv preprint
arXiv:2410.13509 .
Percy Liang, Rishi Bommasani, Tony Lee, Dimitris
Tsipras, Dilara Soylu, Michihiro Yasunaga, Yian
Zhang, Deepak Narayanan, Yuhuai Wu, Ananya Ku-
mar, and 1 others. 2022. Holistic evaluation of lan-
guage models. arXiv preprint arXiv:2211.09110 .
Aixin Liu, Bei Feng, Bing Xue, Bingxuan Wang,
Bochao Wu, Chengda Lu, Chenggang Zhao, Chengqi
Deng, Chenyu Zhang, Chong Ruan, and 1 others.
2024. Deepseek-v3 technical report. arXiv preprint
arXiv:2412.19437 .
Hang Luo, Jian Zhang, and Chujun Li. 2025. Causal
graphs meet thoughts: Enhancing complex rea-
soning in graph-augmented llms. arXiv preprint
arXiv:2501.14892 .
OpenAI. 2023. Gpt-4 technical report. https://
openai.com/research/gpt-4 . Accessed: 2025-
03-27.
Ankit Pal, Logesh Kumar Umapathi, and Malaikannan
Sankarasubbu. 2022. Medmcqa: A large-scale multi-
subject multi-choice dataset for medical domain ques-
tion answering. In Proceedings of the Conference
on Health, Inference, and Learning , volume 174 of
Proceedings of Machine Learning Research , pages
248–260. PMLR.
Judea Pearl. 2009. Causality . Cambridge university
press.
Nils Reimers and Iryna Gurevych. 2019. Sentence-bert:
Sentence embeddings using siamese bert-networks.
arXiv preprint arXiv:1908.10084 .
Stephen Robertson, Hugo Zaragoza, and 1 others. 2009.
The probabilistic relevance framework: Bm25 and
beyond. Foundations and Trends ®in Information
Retrieval , 3(4):333–389.
Chamod Samarajeewa, Daswin De Silva, Evgeny Os-
ipov, Damminda Alahakoon, and Milos Manic. 2024.
Causal reasoning in large language models using
causal graph retrieval-augmented generation. arXiv
preprint , arXiv:2410.11414.
John Schulman, Filip Wolski, Prafulla Dhariwal,
Alec Radford, and Oleg Klimov. 2017. Proxi-
mal policy optimization algorithms. arXiv preprint
arXiv:1707.06347 .
Zhongxiang Sun, Xiaoxue Zang, Kai Zheng, Yang
Song, Jun Xu, Xiao Zhang, Weijie Yu, and Han Li.
2024. Redeep: Detecting hallucination in retrieval-
augmented generation via mechanistic interpretabil-
ity.arXiv preprint arXiv:2410.11414 .

Fiona Anting Tan, Xinyu Zuo, and See-Kiong Ng. 2023.
Unicausal: Unified benchmark and repository for
causal text mining. In International Conference on
Big Data Analytics and Knowledge Discovery , pages
248–262. Springer.
Hugo Touvron, Thibaut Lavril, Gautier Izacard, Xavier
Martinet, Marie-Anne Lachaux, Timothée Lacroix,
Baptiste Rozière, Naman Goyal, Eric Hambro, Faisal
Azhar, and 1 others. 2023. Llama: Open and effi-
cient foundation language models. arXiv preprint
arXiv:2302.13971 .
Aniket Vashishtha, Abbavaram Gowtham Reddy, Ab-
hinav Kumar, Saketh Bachu, Vineeth N Balasubra-
manian, and Amit Sharma. Causal inference us-
ing llm-guided discovery, 2023. URL https://arxiv.
org/abs/2310.15117 .
Nengbo Wang, Xiaotian Han, Jagdip Singh, Jing Ma,
and Vipin Chaudhary. 2025. Causalrag: Integrating
causal graphs into retrieval-augmented generation.
arXiv preprint arXiv:2503.19878 .
Wenhui Wang, Furu Wei, Li Dong, Hangbo Bao, Nan
Yang, and Ming Zhou. 2020. Minilm: Deep self-
attention distillation for task-agnostic compression
of pre-trained transformers. Advances in neural in-
formation processing systems , 33:5776–5788.
Zhepei Wei, Wei-Lin Chen, and Yu Meng. 2024. In-
structrag: Instructing retrieval-augmented genera-
tion via self-synthesized rationales. arXiv preprint
arXiv:2406.13629 .
Zhilin Yang, Peng Qi, Saizheng Zhang, Yoshua Ben-
gio, William W Cohen, Ruslan Salakhutdinov, and
Christopher D Manning. 2018. Hotpotqa: A dataset
for diverse, explainable multi-hop question answer-
ing. arXiv preprint arXiv:1809.09600 .
Tian Yu, Shaolei Zhang, and Yang Feng. 2024.
Auto-rag: Autonomous retrieval-augmented gener-
ation for large language models. arXiv preprint
arXiv:2411.19443 .
Ziyuan Zhuang, Zhiyang Zhang, Sitao Cheng, Fangkai
Yang, Jia Liu, Shujian Huang, Qingwei Lin, Saravan
Rajmohan, Dongmei Zhang, and Qi Zhang. 2024. Ef-
ficientrag: Efficient retriever for multi-hop question
answering. arXiv preprint arXiv:2408.04259 .
A Appendix - Dataset
This appendix provides additional implementation
and experimental details to support the results and
claims presented in the main paper. It includes com-
prehensive documentation of our data construction
process, fine-tuning setup, and evaluation proce-
dures. We also provide prompt templates used for
multi-task instruction tuning, detailed ablation met-
rics, and further discussions of design choices and
observations.A.1 Data Collection and Causal Graph
Construction
Our data collection process supports two major ob-
jectives: (1) training the query refinement module
with multi-task instruction examples, and (2) con-
structing structured causal knowledge graphs that
power CDF-RAG’s graph-based retrieval. We col-
lect and process data from four benchmark QA
datasets— CosmosQA ,AdversarialQA ,MedQA ,
andMedMCQA —chosen for their coverage of com-
monsense, adversarial, and biomedical reasoning
tasks. Each dataset is used to extract causally rele-
vant triples and generate query refinement prompts
across decomposition, expansion, and simplifica-
tion modes.
To enable structured causal retrieval, we imple-
ment a dedicated preprocessing pipeline named
CausalFusion . This component combines fine-
tuned causal classification with LLM-based valida-
tion to extract high-confidence cause-effect pairs
from each dataset. Specifically, we build on the
UniCausal (Tan et al., 2023) framework and fo-
cus on the Causal Pair Classification task. Sen-
tences from each dataset are annotated with candi-
date argument spans ( <ARG0> and<ARG1> ), which
are passed through a BERT-based encoder trained
to predict whether a causal relationship exists be-
tween them. The model outputs binary judgments
that filter candidate pairs down to high-quality
causal candidates.
Following this step, we apply a GPT-4 refine-
ment stage to all accepted causal pairs. GPT-4
serves as a semantic verifier and reformulator: it
rephrases each pair into a fluent, logically coherent
causal statement, flags inconsistencies, and rejects
biologically implausible or semantically invalid
pairs. The output for each instance includes the
original dataset name, cause and effect variables,
predicted directionality, and the refined natural lan-
guage causal explanation.
All validated and rephrased causal pairs are
stored as directed triples in a Neo4j knowledge
graph. To support fast and semantically aware re-
trieval, we encode each node (cause or effect) into a
384-dimensional embedding using MiniLM-based
sentence encoders. These embeddings are stored in
a vector database alongside their graph identifiers,
enabling hybrid semantic and path-based retrieval
during inference. This graph forms the foundation
for multi-hop causal reasoning in CDF-RAG and
is continuously updated as new validated pairs are

added.
This hybrid symbolic-neural representation en-
sures that retrieval can traverse explicit causal paths
while remaining robust to lexical variation in user
queries. It also provides a structured backbone for
measuring retrieval depth, validating generation,
and supporting hallucination detection via graph-
based entailment.
A.2 Causal Prompt Design for Pair
Verification
To ensure the factual and causal correctness of
extracted pairs in our CDF pipeline, we design a
GPT-4-based verification module using structured
natural language prompts. Each extracted causal
pair undergoes a validation stage, where it is con-
verted into a prompt and sent to GPT-4 for semantic
and causal assessment. The goal is to ensure that
only high-confidence, directionally accurate, and
domain-valid causal links are retained for inclusion
in the Neo4j causal graph.
We adopt a contextualized causal prompting
strategy inspired by the causal wrapper component
in ALCM (Khatibi et al., 2024). Each prompt in-
cludes:
•Instruction —Defining GPT-4’s role in assess-
ing the causal pair.
•Contextual Metadata —Information about
the dataset, domain, and source extraction
model.
•Causal Pair —The specific cause-effect rela-
tionship being assessed.
•Task Definition —Explicit questions about
the validity, direction, and justification of the
causal link.
•Output Format —A structured template in-
cluding a binary correctness flag, refined
causal direction, confidence score, and expla-
nation.
This causal prompt design enables the LLM to
reason explicitly about the plausibility and correct-
ness of each candidate link. It also facilitates stan-
dardized post-processing by producing consistent,
machine-readable outputs. Verified causal pairs are
then re-integrated into the graph database, ensuring
that downstream query refinement and multi-hop
reasoning are grounded in trustworthy knowledge.
An illustrative example of such a prompt is
shown below:Causal Pair to Verify: {Cause:
High blood pressure, Effect:
Stroke}
Correctness: True
Refined Causal Statement: "High
blood pressure causes stroke"
Confidence: High
Explanation: Chronic
hypertension is a well-known
risk factor for stroke based on
medical literature.

Causal Verification Prompt Template
You are an expert in {DOMAIN} with deep
knowledge of causal relationships and
evidence-based reasoning.
You are given a candidate causal relation-
ship extracted from a document or causal
discovery algorithm.
Your task is to evaluate whether the follow-
ing causal relationship is factually and logi-
cally correct based on your internal knowl-
edge and reasoning. You may accept, reject,
revise, or reorient the pair. Use step-by-step
reasoning to justify your answer.
Contextual Metadata:
• Domain: {DOMAIN}
• Dataset: {DATASET NAME}
•Source Model: {MODEL or EXTRACTION
METHOD}
Causal Pair to Verify:
• Cause:{ARG0}
• Effect:{ARG1}
Task:
1. Is the causal relationship valid and sup-
ported? (Answer: True/False)
2. If the direction is incorrect, provide the
corrected direction.
3. Provide a one-sentence explanation justi-
fying your decision.
4. Estimate your confidence in the answer
(High / Medium / Low)
Output Format:
Correctness: {True / False}
Refined Causal Statement: "{ARG0}"
causes "{ARG1}" or"{ARG1}" causes
"{ARG0}"
Confidence: {High / Medium / Low}
Explanation: {Short justification
grounded in domain knowledge}
A.3 Reinforcement Learning for Query
Refinement.
To dynamically optimize query rewriting strategies
in CDF-RAG, we train a RL agent using the Prox-
imal Policy Optimization (PPO) algorithm. The
agent learns a policy π(a|s)that maps the semantic
embedding of a raw query sto one of three refine-
ment actions: Expand ,Simplify , orDecompose .
Each action corresponds to a rewriting strategy
aimed at improving causal specificity and retriev-ability. The agent interacts with a custom Gym en-
vironment, where each state sis a 384-dimensional
embedding of the input query (from MiniLM), and
the action space is discrete over refinement types.
The reward function integrates downstream per-
formance metrics critical for causal reasoning. Af-
ter each refinement action, the system executes the
retrieval and generation pipeline and computes four
normalized metrics: retrieval relevance ( r), causal
depth ( d), semantic similarity ( s), and hallucination
rate (h). The reward function is defined as:
R=λ1r+λ2d+λ3s+λ4(1−h)
where each component is normalized to the
range [0,1], and λiare tunable weights control-
ling the importance of each term. Relevance mea-
sures whether the refinement improves the match
between retrieved context and query intent; causal
depth quantifies the number of multi-hop causal
links retrieved; semantic similarity evaluates align-
ment with the original query; and hallucination
penalizes factual inconsistency in generated out-
puts.
We train the agent using PPO with a two-layer
MLP policy network (hidden size 256), batch size
64, learning rate 3×10−4, and entropy regulariza-
tion of 0.01. Training is run for 100 epochs with
500 steps per query. The training curriculum covers
diverse domains by sampling queries from MedQA,
CosmosQA, and AdversarialQA. All models ex-
cept GPT-4 are trained using this RL framework
after multi-task instruction fine-tuning.
At inference time, the trained policy π(a|s)se-
lects the optimal refinement action given an unseen
input query. This enables the system to adaptively
reformulate questions in a way that aligns with
both the causal structure of the knowledge graph
and the semantic requirements of the task, thereby
improving downstream accuracy, coherence, and
explainability.
A.4 Prompt Design for Multi-task Instruction
Fine-tuning
To enable the query refinement module in CDF-
RAG to adaptively rewrite input questions, we con-
struct a multi-task instruction dataset covering three
core refinement actions: Simplify ,Decompose , and
Expand . These refinement strategies correspond to
key capabilities required for causal reasoning: clar-
ifying ambiguous questions, breaking down com-
plex ones into causal subcomponents, and enrich-
ing underspecified queries with relevant scope. For

each action type, we design a specialized prompt
template to guide GPT-4 in generating high-quality
supervision examples. These templates are used
to fine-tune the LLMs (LLaMA 3-8B, Mistral, and
Flan-T5) using LoRA, while GPT-4 is accessed via
API at inference time without fine-tuning.
Simplification Prompt As shown in Prompt
Box A.4, we provide GPT-4 with detailed instruc-
tions for simplifying complex questions while pre-
serving their original intent. This template is used
to rephrase complex, ambiguous, or overly verbose
queries into concise and direct questions while pre-
serving their original intent. The goal is to strip
away unnecessary syntactic or semantic complex-
ity to improve retrievability and alignment with the
knowledge base. The model is instructed to out-
put a single-line question that is self-contained and
interpretable, which is essential for enhancing the
precision of retrieval in high-noise or cross-domain
settings.Simplification Prompt Template
Your task is to simplify complex or ambigu-
ous questions into a clearer, more direct ver-
sion that preserves the original intent. This
should help reduce unnecessary complexity
while keeping the meaning intact.
Please follow the steps below carefully:
1.Identify any ambiguity, compound
phrasing, or indirect constructs in the
input question.
2. Reformulate the question as a concise,
direct, and self-contained single ques-
tion.
3.Ensure that the simplified version can
be interpreted and answered indepen-
dently.
Guidelines:
•Use precise language that avoids un-
necessary technical or abstract phras-
ing.
•Do not generate multiple sub-
questions.
•Keep the simplified question to a single
line of text.
•Preserve the core meaning of the origi-
nal question.
Here is your task:
•Provided Contexts: {OPTIONAL
— leave blank or include
background passages}
•Original Question: {INSERT COMPLEX
OR AMBIGUOUS QUESTION}
• Simplified Query:
Prompting Strategy. To enable query simplifica-
tion within CDF-RAG, we adopt a dual-prompting
approach tailored for both system implementa-
tion and interpretability. For fine-tuning and in-
ference, we use a concise instruction-tuning format
("Refine the following query for better
causal retrieval" ) to streamline training across
hundreds of examples. To complement this, we de-

fine a structured prompt template with explicit steps
and guidelines for simplification, which is used
in our paper to illustrate the design intent behind
simplification behavior. This alignment between
lightweight instructional prompts and a principled
template ensures both efficiency and transparency
in how simplification is operationalized within the
framework.Simplification Prompt Template
Your task is to simplify complex or ambigu-
ous questions into a clearer, more direct ver-
sion that preserves the original intent. This
should help reduce unnecessary complexity
while keeping the meaning intact.
Please follow the steps below carefully:
1.Identify any ambiguity, compound
phrasing, or indirect constructs in the
input question.
2. Reformulate the question as a concise,
direct, and self-contained single ques-
tion.
3.Ensure that the simplified version can
be interpreted and answered indepen-
dently.
Guidelines:
•Use precise language that avoids un-
necessary technical or abstract phras-
ing.
•Do not generate multiple sub-
questions.
•Keep the simplified question to a single
line of text.
•Preserve the core meaning of the origi-
nal question.
Here is your task:
•Provided Contexts: Medical QA task
related to diabetic nephropathy
•Original Question: Why does
diabetes cause kidney damage
in elderly patients, and what
factors contribute to this
progression over time?
•Simplified Query: How does
diabetes cause kidney damage in
elderly patients?
Decomposition Prompt The decomposition
prompt (see Prompt Box A.4) teaches the model
to break down multihop or causally entangled
questions into 2–4 atomic sub-questions that col-

lectively reconstruct the original reasoning chain.
Each sub-question should be answerable indepen-
dently and follow a logical progression that mirrors
multi-hop causal inference. This prompt is partic-
ularly important for enabling causal retrieval over
multi-node paths in the Neo4j graph and for pro-
moting modular reasoning within the generation
phase.Decomposition Prompt Template
Your task is to decompose complex, multi-
hop questions into simpler, manageable
sub-questions. These decomposed queries
should help isolate and uncover causal or ex-
planatory mechanisms relevant to the origi-
nal question.
Please follow the steps below carefully:
1.Analyze the multihop question to iden-
tify its underlying causal or semantic
components.
2.Reformulate the question into a list of
2–4 clear, concise, self-contained sub-
questions that can be independently an-
swered.
3.Maintain logical flow between sub-
questions (i.e., each one should build
toward answering the original ques-
tion).
Guidelines:
•Avoid repeating the same phrasing
across sub-questions.
•Each sub-question should be answer-
able on its own.
•Use one line per sub-question, and in-
sert a line break between each.
•Do not include numbered bullets or ex-
planations—only the raw list of sub-
questions.
Here is your task:
•Provided Contexts: {OPTIONAL
— leave blank or include
background passages}
•Multihop Question: {INSERT MAIN
QUESTION}
• Decomposed Queries:
Prompting Strategy. For decomposition, we em-
ploy a structured prompt that guides the model to
break down complex, multihop questions into logi-
cally ordered sub-questions (see Prompt Box A.4).
While this instructional format is used for trans-

parency and design illustration, the deployed sys-
tem leverages a compact instruction-tuning vari-
ant during fine-tuning and inference (e.g., "Break
this question into sub-questions for
causal reasoning." ). This alignment allows us
to retain explainability in prompt engineering while
maintaining efficiency and generalizability in real-
time execution.
Decomposition Prompt Template
Your task is to decompose complex, multi-
hop questions into simpler, manageable
sub-questions. These decomposed queries
should help isolate and uncover causal or ex-
planatory mechanisms relevant to the origi-
nal question.
Here is your task:
•Provided Contexts: Healthcare
domain — diabetes and kidney
disease
•Multihop Question: Why does
diabetes lead to kidney failure
in aging populations over time?
• Decomposed Queries:
What physiological changes does diabetes
cause in the kidneys?
How does chronic hyperglycemia damage
kidney function over time?
What role does aging play in accelerating
diabetic kidney complications?
Why are older adults more susceptible to
renal decline with diabetes?
Expansion Prompt For queries that are vague or
underspecified, the expansion prompt (see Prompt
Box A.4) guides the model to make the question
more complete by adding relevant causal factors,
domain-specific constraints, or example conditions.
The objective is to surface latent context or scope
that may be implicitly expected but is missing in
the original query. This expanded form allows
the retrieval system to access a broader and more
causally aligned evidence space.Expansion Prompt Template
Your task is to expand a vague or underspec-
ified question into a more detailed version
that makes its intent clear and specific. This
should help clarify the scope of the question
by introducing relevant dimensions, factors,
or examples.
Please follow the steps below carefully:
1.Identify missing context or implicit as-
sumptions in the question.
2.Reformulate the question to explic-
itly mention key entities, causal mech-
anisms, or domains relevant to the
query.
3.Ensure the expanded question guides a
more targeted and informative answer.
Guidelines:
•Use a single line for the expanded ques-
tion.
•Avoid changing the core topic, but add
specificity or scope.
•Preserve the original intent, while mak-
ing the question more complete or in-
formative.
Here is your task:
•Provided Contexts: {OPTIONAL
— leave blank or include
background passages}
•Original Question: {INSERT VAGUE OR
INCOMPLETE QUESTION}
• Expanded Query:
Prompting Strategy. The expansion prompt is
designed to elicit more informative and context-
aware reformulations for vague or under-specified
queries (see Prompt Box A.4). While this prompt
is used to train the model to surface latent causal
factors and clarify scope, the system implemen-
tation uses a condensed instruction-tuned variant
(e.g.,"Make the question more specific for
causal reasoning" ). This dual-prompting setup
ensures that the model learns how to expand queries
both accurately and efficiently, while also preserv-

ing interpretability and alignment during prompt
analysis and dataset curation.
Expansion Prompt Template
Your task is to expand a vague or underspec-
ified question into a more detailed version
that makes its intent clear and specific. This
should help clarify the scope of the question
by introducing relevant dimensions, factors,
or examples.
Here is your task:
•Provided Contexts: Societal health
disparities and stress
•Original Question: Why is stress a
public health concern?
• Expanded Query:
Why is chronic stress considered a public
health concern in relation to socioeconomic
status, mental health, and long-term disease
risk?
Together, these prompt templates form the back-
bone of our instruction fine-tuning strategy, en-
abling each model to learn not only how to execute
a refinement action, but also when and why such
rewrites are useful for causal alignment. Each gen-
erated example is filtered for consistency and cor-
rectness before being added to the training dataset.
During inference, the PPO-trained policy network
selects among these three refinement actions for
each input query, enabling dynamic adaptation to
the structure and intent of unseen questions.
B Appendix - Experimental Details
B.1 Training and Fine-tuning Setup
We fine-tune all LLM backbones (except GPT-
4, which is accessed via API) using LoRA with
instruction-style supervision. Each model is trained
on our multi-task dataset for one epoch with a learn-
ing rate of 2e-5 and 3% warmup steps.
B.2 Comparison with Related Work
CDF-RAG introduces a comprehensive and agen-
tic approach to RAG by combining causal graph
retrieval, RL-driven query refinement, multi-hop
reasoning, and hallucination correction into a uni-
fied framework. This integrated design enables themodel to explicitly reason over structured cause-
effect relationships while adaptively optimizing
queries and validating outputs through a closed-
loop process. Unlike existing methods that focus
on isolated components of the RAG pipeline, CDF-
RAG emphasizes the causal alignment and coher-
ence of both retrieved and generated content.
In contrast, methods like RQ-RAG and Smar-
tRAG provide query refinement capabilities—via
decomposition or RL—but do not incorporate
causal graph retrieval or hallucination mitiga-
tion. RAG-Gym offers process-level optimization
through nested MDPs and includes a hallucination-
aware reward model, but lacks structural causal
reasoning. Causal Graph RAG and Causal Graphs
Meet Thoughts integrate causal graphs but fall
short in dynamic feedback, multi-agent coordina-
tion, and hallucination control. Overall, CDF-RAG
is distinguished by its holistic design that tightly
couples causal retrieval, adaptive refinement, and
output validation—resulting in improved factuality,
reasoning depth, and consistency.
B.3 Implementation and Agentic Design
Our CDF-RAG framework is implemented using
the LangChain library, which provides modular
primitives for constructing agentic workflows in
language model systems. We structure the pipeline
as a multi-step LangGraph agent, where each node
represents a semantically grounded reasoning mod-
ule: query refinement, causal retrieval, knowledge
rewriting, response generation, hallucination de-
tection, and correction. The use of LangGraph
allows us to declaratively define state transitions
and orchestrate feedback loops, enabling condi-
tional routing and dynamic re-entry into refinement
or correction stages based on internal evaluation
metrics (e.g., hallucination confidence or causal
coverage).
CDF-RAG is inherently an agentic system in
that it models reasoning as an autonomous, self-
adaptive process. Rather than a fixed sequence
of API calls, our agent selects actions (e.g., re-
querying, rewriting, regenerating) based on the
evolving context of the task. This is made pos-
sible by integrating reinforcement learning (RL)
for policy-driven refinement, and a hallucination-
aware validation agent that triggers corrective sub-
routines when inconsistencies are detected. Each
component is instantiated as a callable LangChain
module, with memory and state passed explicitly
between steps—fulfilling the agentic paradigm of

planning, acting, observing, and adapting. This
design enables the system to reason causally, re-
cover from failures, and adapt its strategy based on
downstream performance.
B.4 Additional Results
We include additional results on metric break-
downs by task and model, alternative retrieval con-
figurations, and the impact of hallucination correc-
tion. We also report groundedness and CRC scores
per refinement type to demonstrate the effective-
ness of individual modules in isolation. Across all
experiments, CDF-RAG was evaluated on approx-
imately 2,200 queries spanning four benchmark
datasets—CosmosQA, MedQA, MedMCQA, and
AdversarialQA—across multiple LLM backbones.
B.4.1 Quality Performance
We report quantitative results in Table 4
and Table 5 across four benchmark QA
datasets— CosmosQA (Huang et al., 2019), Adver-
sarialQA (Bartolo et al., 2020), MedQA (Jin et al.,
2020), and MedMCQA (Pal et al., 2022)—evalu-
ated on four LLM backbones (GPT-4 (OpenAI,
2023), LLaMA 3-8B (Touvron et al., 2023),
Mistral (Jiang et al., 2023), and Flan-T5 (Chung
et al., 2024)). Across all combinations, CDF-RAG
outperforms existing RAG variants in accuracy,
precision, recall, and F1 score, while maintaining
the lowest HR. This demonstrates the effectiveness
of our fully integrated framework—combining
reinforcement-learned query refinement, causal
graph-augmented retrieval, structured rewriting,
and hallucination-aware output validation.Table 4: Quality Metrics ofCDF-RAG across models
and methods. HR = Hallucination Rate, F1 = F1 Score.
Dataset Model Method HR Acc. Prec. Rec. F1
AdversarialQAGPT-4 CDF-RAG 0.07 0.89 0.850 0.87 0.860
Gym-RAG 0.14 0.78 0.735 0.76 0.745
RQ-RAG 0.15 0.76 0.715 0.74 0.725
Smart-RAG 0.16 0.74 0.700 0.72 0.710
Causal RAG 0.18 0.71 0.670 0.69 0.680
G-LLMs 0.20 0.68 0.640 0.66 0.650
LLaMA 3-8B CDF-RAG 0.08 0.83 0.805 0.82 0.815
Gym-RAG 0.13 0.75 0.700 0.72 0.710
RQ-RAG 0.12 0.71 0.660 0.68 0.670
Smart-RAG 0.15 0.73 0.675 0.69 0.680
Causal RAG 0.17 0.71 0.655 0.67 0.660
G-LLMs 0.19 0.68 0.620 0.64 0.630
Mistral CDF-RAG 0.09 0.81 0.790 0.79 0.785
Gym-RAG 0.15 0.73 0.680 0.70 0.690
RQ-RAG 0.16 0.72 0.660 0.68 0.670
Smart-RAG 0.17 0.70 0.645 0.66 0.655
Causal RAG 0.17 0.66 0.600 0.62 0.615
G-LLMs 0.21 0.65 0.590 0.61 0.600
Flan-T5 CDF-RAG 0.10 0.79 0.760 0.77 0.765
Gym-RAG 0.16 0.70 0.640 0.66 0.650
RQ-RAG 0.15 0.66 0.600 0.61 0.615
Smart-RAG 0.16 0.64 0.590 0.60 0.605
Causal RAG 0.18 0.62 0.560 0.58 0.570
G-LLMs 0.20 0.60 0.540 0.56 0.550
CosmosQAGPT-4 CDF-RAG 0.06 0.89 0.86 0.85 0.855
Gym-RAG 0.11 0.82 0.77 0.79 0.78
RQ-RAG 0.11 0.80 0.75 0.77 0.76
Smart-RAG 0.16 0.78 0.74 0.76 0.75
Causal RAG 0.17 0.76 0.71 0.73 0.72
G-LLMs 0.20 0.73 0.68 0.70 0.69
LLaMA 3-8B CDF-RAG 0.07 0.88 0.85 0.84 0.845
Gym-RAG 0.12 0.80 0.76 0.77 0.765
RQ-RAG 0.14 0.79 0.74 0.75 0.745
Smart-RAG 0.18 0.77 0.72 0.73 0.725
Causal RAG 0.18 0.75 0.70 0.71 0.705
G-LLMs 0.21 0.72 0.67 0.69 0.68
Mistral CDF-RAG 0.08 0.85 0.82 0.81 0.815
Gym-RAG 0.14 0.75 0.70 0.72 0.71
RQ-RAG 0.15 0.74 0.68 0.70 0.69
Smart-RAG 0.18 0.72 0.66 0.68 0.67
Causal RAG 0.20 0.70 0.63 0.66 0.645
G-LLMs 0.22 0.68 0.60 0.63 0.615
Flan-T5 CDF-RAG 0.10 0.84 0.80 0.79 0.795
Gym-RAG 0.15 0.73 0.68 0.70 0.69
RQ-RAG 0.16 0.72 0.66 0.68 0.67
Smart-RAG 0.19 0.70 0.64 0.66 0.65
Causal RAG 0.21 0.68 0.61 0.64 0.625
G-LLMs 0.24 0.66 0.59 0.61 0.60

Table 5: Quality Metrics ofCDF-RAG across models
and methods. HR = Hallucination Rate, F1 = F1 Score.
Dataset Model Method HR Acc. Prec. Rec. F1
MedQAGPT-4 CDF-RAG 0.05 0.92 0.890 0.91 0.900
Gym-RAG 0.12 0.83 0.760 0.78 0.770
RQ-RAG 0.13 0.82 0.745 0.77 0.755
Smart-RAG 0.15 0.81 0.730 0.76 0.745
Causal RAG 0.17 0.79 0.710 0.74 0.725
G-LLMs 0.21 0.76 0.680 0.71 0.695
LLaMA 3-8B CDF-RAG 0.07 0.89 0.860 0.88 0.870
Gym-RAG 0.11 0.79 0.735 0.75 0.740
RQ-RAG 0.13 0.78 0.720 0.74 0.730
Smart-RAG 0.15 0.77 0.705 0.72 0.710
Causal RAG 0.17 0.75 0.675 0.69 0.680
G-LLMs 0.20 0.72 0.640 0.66 0.650
Mistral CDF-RAG 0.08 0.88 0.845 0.87 0.855
Gym-RAG 0.14 0.78 0.720 0.74 0.730
RQ-RAG 0.16 0.77 0.705 0.73 0.715
Smart-RAG 0.18 0.76 0.690 0.71 0.700
Causal RAG 0.20 0.74 0.665 0.68 0.670
G-LLMs 0.23 0.71 0.630 0.65 0.640
Flan-T5 CDF-RAG 0.11 0.84 0.800 0.82 0.810
Gym-RAG 0.17 0.73 0.670 0.69 0.680
RQ-RAG 0.19 0.72 0.655 0.68 0.665
Smart-RAG 0.21 0.71 0.640 0.66 0.650
Causal RAG 0.23 0.69 0.615 0.64 0.625
G-LLMs 0.26 0.67 0.590 0.62 0.605
MedMCQAGPT-4 CDF-RAG 0.04 0.94 0.910 0.93 0.920
Gym-RAG 0.13 0.78 0.735 0.75 0.740
RQ-RAG 0.15 0.76 0.720 0.73 0.725
Smart-RAG 0.18 0.74 0.700 0.71 0.705
Causal RAG 0.21 0.72 0.670 0.69 0.680
G-LLMs 0.25 0.68 0.635 0.66 0.650
LLaMA 3-8B CDF-RAG 0.08 0.90 0.870 0.91 0.890
Gym-RAG 0.13 0.77 0.720 0.74 0.730
RQ-RAG 0.15 0.75 0.705 0.72 0.715
Smart-RAG 0.18 0.73 0.685 0.70 0.690
Causal RAG 0.20 0.71 0.660 0.68 0.670
G-LLMs 0.24 0.68 0.625 0.65 0.640
Mistral CDF-RAG 0.09 0.88 0.850 0.89 0.870
Gym-RAG 0.14 0.76 0.710 0.73 0.720
RQ-RAG 0.16 0.74 0.695 0.71 0.700
Smart-RAG 0.19 0.72 0.670 0.69 0.680
Causal RAG 0.22 0.70 0.645 0.67 0.655
G-LLMs 0.26 0.66 0.610 0.63 0.620
Flan-T5 CDF-RAG 0.12 0.85 0.810 0.84 0.825
Gym-RAG 0.18 0.72 0.680 0.70 0.690
RQ-RAG 0.20 0.70 0.660 0.68 0.670
Smart-RAG 0.23 0.68 0.635 0.66 0.650
Causal RAG 0.26 0.66 0.610 0.63 0.620
G-LLMs 0.29 0.63 0.580 0.60 0.590
The consistent superiority of CDF-RAG across
both open-domain (e.g., CosmosQA ) and domain-
specific (e.g., MedQA ) datasets indicates its robust-
ness in both commonsense and biomedical reason-
ing tasks. On MedMCQA , for instance, CDF-RAG
with GPT-4 achieves an F1 of 0.920 and HR of
0.04—substantially outperforming Gym-RAG (F1
= 0.740, HR = 0.13) (et al., 2025) and RQ-RAG
(F1 = 0.725, HR = 0.15) (Chan et al., 2024).
CDF-RAG’s performance gains stem from three
complementary innovations. First, causal graph
retrieval introduces directional constraints and en-
ables multi-hop traversal over verified cause-effect
pairs, outperforming semantic or correlation-based
retrieval methods. Second, RL-guided query re-
finement uses a PPO-trained agent to dynamically
expand, simplify, or decompose queries based on
causal depth and retrieval feedback, improving
query intent alignment. Third, causal verifica-tion applies post-generation consistency checks in-
spired by counterfactual reasoning (Pearl, 2009) to
detect unsupported or inverted causal statements
and regenerate outputs accordingly. By jointly
leveraging these components in a closed feed-
back loop, CDF-RAG preserves both semantic and
causal alignment across the entire RAG pipeline,
yielding more consistent, accurate, and trustworthy
outputs.
RQ-RAG (Chan et al., 2024) enhances query
clarity via rewriting and decomposition but lacks
structural guidance or post-generation validation.
Gym-RAG (et al., 2025) trains reward models to
optimize process-level behavior but does not in-
tegrate causal priors or hallucination mitigation.
SmartRAG (Gao et al., 2024) performs joint op-
timization across retrieval and generation using
RL, but still relies on semantic-level retrieval, mak-
ing it susceptible to spurious correlations. Causal
Graph RAG (Samarajeewa et al., 2024) and Causal
Graphs Meet Thoughts (Luo et al., 2025) incor-
porate causality via vector embeddings and sum-
marization heuristics. However, their extraction
methods are noisy, graph traversal is not adaptive,
and there is no RL optimization or hallucination
correction. G-LLMs represent graph-augmented
models that lack causal reasoning, making them
insufficient for multi-hop logical chains.
CDF-RAG is distinguished by its holistic inte-
gration of causally grounded retrieval, RL-based
query adaptation, and hallucination-aware post-
verification, enabling superior factuality and rea-
soning depth across QA tasks.
In contrast, Gym-RAG andRQ-RAG demon-
strate strong but lower performance due to their
reliance on process supervision and query rewrit-
ing respectively. While these methods improve
retrieval quality and answer coherence, they lack
explicit causal validation. RQ-RAG refines am-
biguous queries through rewriting and decomposi-
tion, but fails to enforce causal entailment in the
retrieved or generated content. Gym-RAG benefits
from reward-guided search trajectories but does
not incorporate structural causal priors or halluci-
nation mitigation. This leads to higher HR and
slightly lower precision and recall compared to
CDF-RAG. Smart-RAG performs competitively
with a lightweight joint RL framework that learns
when to retrieve and when to generate. However, it
lacks structured causal graph grounding and post-
hoc verification, making it prone to hallucinations
and inconsistent multi-hop reasoning. Similarly,

Causal RAG utilizes causal vector graphs but
depends on weak summarizer-based pair extrac-
tion, leading to noisy graph structures and unstable
downstream performance.
Finally, G-LLMs consistently lag behind due to
their reliance on static semantic graphs or unstruc-
tured correlation-based retrieval. These models
lack query adaptation, causal reasoning, and hallu-
cination correction—all of which are essential for
high-quality answers in complex QA tasks. This
explains their lower precision, recall, and F1 scores
across all datasets in Table 4 and Table 5, and justi-
fies the significant performance gains achieved by
CDF-RAG.
B.4.2 Case Study: End-to-End Causal
Answering with CDF-RAG
To illustrate how CDF-RAG operates end-to-end,
we present a complete walkthrough in Prompt
Box B.4.2. Given the vague user query, "Why
do people get sick from poor living conditions?" ,
the RL-trained query refinement agent selects a
Decompose strategy and rewrites the input into
three causally grounded sub-questions. These sub-
queries guide both structured and unstructured re-
trieval components.
The structured retriever accesses a Neo4j causal
graph and surfaces multi-hop, directionally valid
causal chains, such as Poor Housing →Mold
Exposure →Asthma . In parallel, a dense retriever
fetches semantically similar passages from a vector
database. The LLM then generates a comprehen-
sive explanation using both sources, linking poor
housing, sanitation, and overcrowding to health
outcomes. This response is passed through the
causal verifier, which confirms the alignment of
each claim with the retrieved causal graph and
textual evidence. As shown in the final output,
CDF-RAG produces a response that is not only
accurate and coherent but also causally consistent
and evidence-backed. This example highlights how
each module in the pipeline—query refinement,
graph-based retrieval, generation, and hallucination
detection—contributes to grounded and trustwor-
thy QA.
End-to-End CDF-RAG Workflow Example
Input Query:
Why do people get sick from poor
living conditions?Step 1: Query Refinement (RL Agent)
Refinement Type: Decompose
Refined Subqueries:
•How does poor housing quality cause
chronic illness?
•How does lack of sanitation contribute
to disease?
•How does overcrowding affect respira-
tory health?
Step 2: Structured Retrieval (Causal
Graph)
Retrieved Causal Paths from Neo4j:
•Poor Housing →Mold Exposure →
Asthma
•Poor Sanitation →Water Contamina-
tion→Diarrheal Disease
•Overcrowding →Airborne Transmis-
sion→Respiratory Infection
Step 3: Dense Retrieval (Vector
Database)
Top Supporting Passages:
•“Exposure to indoor mold from poor
housing is strongly linked to asthma in
urban populations.”
•“Contaminated water due to poor sani-
tation infrastructure remains a leading
cause of diarrhea globally.”
•“Airborne diseases such as tuberculosis
are more likely to spread in densely
crowded living environments.”
Step 4: Initial Answer Generation (LLM)
Generated Answer:
Poor living conditions lead to
illness through multiple causal
pathways. Poor housing causes
mold exposure, increasing asthma
risk. Lack of sanitation
results in contaminated water,
causing diarrhea. Overcrowding
spreads airborne infections like
tuberculosis.

Step 5: Causal Verification (LLM Veri-
fier)
Result: No hallucinations detected. All
claims supported by retrieved causal paths
and documents.
Final Output:
Poor living conditions contribute
to illness through overlapping and
well-established causal pathways.
Poor housing quality leads to
mold and inadequate ventilation,
increasing respiratory illness
risk. Inadequate sanitation causes
waterborne diseases like diarrhea.
Overcrowding facilitates the
transmission of airborne illnesses
such as tuberculosis. Together,
these conditions elevate disease
burdens in vulnerable populations.