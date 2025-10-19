# Probing Latent Knowledge Conflict for Faithful Retrieval-Augmented Generation

**Authors**: Linfeng Gao, Baolong Bi, Zheng Yuan, Le Wang, Zerui Chen, Zhimin Wei, Shenghua Liu, Qinggang Zhang, Jinsong Su

**Published**: 2025-10-14 12:48:24

**PDF URL**: [http://arxiv.org/pdf/2510.12460v1](http://arxiv.org/pdf/2510.12460v1)

## Abstract
Retrieval-Augmented Generation (RAG) has emerged as a powerful paradigm to
enhance the factuality of Large Language Models (LLMs). However, existing RAG
systems often suffer from an unfaithfulness issue, where the model's response
contradicts evidence from the retrieved context. Existing approaches to
improving contextual faithfulness largely rely on external interventions, such
as prompt engineering, decoding constraints, or reward-based fine-tuning. These
works treat the LLM as a black box and overlook a crucial question: how does
the LLM internally integrate retrieved evidence with its parametric memory,
particularly under knowledge conflicts? To address this gap, we conduct a
probing-based analysis of hidden-state representations in LLMs and observe
three findings: knowledge integration occurs hierarchically, conflicts manifest
as latent signals at the sentence level, and irrelevant context is often
amplified when aligned with parametric knowledge. Building on these findings,
we propose CLEAR (Conflict-Localized and Enhanced Attention for RAG), a
framework that (i) decomposes context into fine-grained sentence-level
knowledge, (ii) employs hidden-state probing to localize conflicting knowledge,
and (iii) introduces conflict-aware fine-tuning to guide the model to
accurately integrate retrieved evidence. Extensive experiments across three
benchmarks demonstrate that CLEAR substantially improves both accuracy and
contextual faithfulness, consistently outperforming strong baselines under
diverse conflict conditions. The related resources are available at
https://github.com/LinfengGao/CLEAR.

## Full Text


<!-- PDF content starts -->

PROBINGLATENTKNOWLEDGECONFLICT FOR
FAITHFULRETRIEVAL-AUGMENTEDGENERATION
Linfeng Gao1, Baolong Bi2, Zheng Yuan3, Le Wang4, Zerui Chen1, Zhimin Wei1
Shenghua Liu2,Qinggang Zhang3∗,Jinsong Su1∗
1Xiamen University2University of Chinese Academy of Sciences
3The Hong Kong Polytechnic University4Migu Meland Co.,Ltd.
{gaolinfeng,chenzeruil}@stu.xmu.edu.cn;zhimin.wei@foxmail.com;
wangle@migu.chinamobile.com;{bibaolong23z,liushenghua}@ict.ac.cn;
{zheng.yuan,qinggangg.zhang}@polyu.edu.hk;jssu@xmu.edu.cn
ABSTRACT
Retrieval-Augmented Generation (RAG) has emerged as a powerful paradigm to
enhance the factuality of Large Language Models (LLMs). However, existing
RAG systems often suffer from an unfaithfulness issue, where the model’s re-
sponse contradicts evidence from the retrieved context. Existing approaches to
improving contextual faithfulness largely rely on external interventions, such as
prompt engineering, decoding constraints, or reward-based fine-tuning. These
works treat the LLM as a black box and overlook a crucial question: how
does the LLM internally integrate retrieved evidence with its parametric mem-
ory, particularly under knowledge conflicts? To address this gap, we conduct
a probing-based analysis of hidden-state representations in LLMs and observe
three findings: knowledge integration occurs hierarchically, conflicts manifest
as latent signals at the sentence level, and irrelevant context is often amplified
when aligned with parametric knowledge. Building on these findings, we propose
CLEAR(Conflict-Localized andEnhancedAttention forRAG), a framework
that (i) decomposes context into fine-grained sentence-level knowledge, (ii) em-
ploys hidden-state probing to localize conflicting knowledge, and (iii) introduces
conflict-aware fine-tuning to guide the model to accurately integrate retrieved evi-
dence. Extensive experiments across three benchmarks demonstrate that CLEAR
substantially improves both accuracy and contextual faithfulness, consistently out-
performing strong baselines under diverse conflict conditions. The related re-
sources are available athttps://github.com/LinfengGao/CLEAR.
1INTRODUCTION
Retrieval-Augmented Generation (RAG) has rapidly evolved as a powerful paradigm to enhance
Large Language Models (LLMs) by leveraging external knowledge bases (Guu et al., 2020a; Feng
et al., 2024; Zhang et al., 2025a). Despite its success, RAG often struggles with context faithful-
ness (Bi et al., 2024a;b), which requires the model to generate responses strictly grounded in external
context. Achieving faithfulness is particularly challenging in scenarios involving knowledge con-
flicts, where discrepancies between the retrieved context and the model’s internal knowledge often
lead to inaccurate or inconsistent generations (Xu et al., 2024a; Zhang et al., 2025b).
Previous studies on improving contextual faithfulness in RAG can be broadly classified into three
categories. The first category utilizes specially designed instructions to guide the model’s reasoning
process, encouraging it to verify or filter retrieved content before generating a response (Zhou et al.,
2023a; Asai et al., 2023; Ying et al., 2024; Zhang et al., 2025b). While this strategy can indeed
improve factual grounding, its effectiveness is often highly sensitive to the design of the instructions
and may not generalize robustly across different domains or tasks. Moreover, the second category
involves modifying the generation process itself by introducing constraints or consistency checks
during decoding to ensure alignment with the retrieved context (Shi et al., 2023a; Yuan et al., 2024).
∗Corresponding author.
1arXiv:2510.12460v1  [cs.CL]  14 Oct 2025

However, these methods are often tightly coupled with specific decoding strategies and may struggle
when the retrieved content contains irrelevant knowledge. Furthermore, the third category focuses
on training the model with explicit objective functions that reward faithful response, thereby framing
the task as an end-to-end optimization problem (Si et al., 2025; Bi et al., 2024a). Although this
approach supports flexible end-to-end learning, it also relies heavily on carefully designed reward
mechanisms and large-scale preference datasets.
Despite these advances, existing approaches share a fundamental limitation: they treat LLMs as
black boxes, focusing on external interventions without investigating the internal knowledge inte-
gration mechanism, i.e., how LLMs internally process and integrate conflicting knowledge. Con-
sequently, their effectiveness is often sensitive to prompt design, decoding strategies, or reward
functions, and it always fails to generalize to real-world scenarios with complex and noisy contexts.
In this paper, we argue that a comprehensive understanding of faithfulness requires moving beyond
these external interventions to directly investigate the internal cognitive processes of LLMs.
To this end, we conduct an in-depth analysis, investigating how LLMs internally fuse external
knowledge with their parametric memory and how models represent and reconcile knowledge con-
flicts within their latent space. Through systematic knowledge probing and detailed representation
analysis, we uncover three key insights: (i) Hierarchical integration: Faithfulness is not broken at
the output layer of language models; it is compromised much earlier. We found that LLMs integrate
knowledge in a progressive and hierarchical manner (token→sentence→passage). The critical
failure occurs at the sentence-level abstraction in intermediate layers, where the model constructs
and reconciles factual representations. (ii) The latent conflict signal: At the sentence level, the
hidden states of the LLM contain a discernible “conflict signal”, a representational bias that pre-
dicts eventual unfaithfulness. This signal is a latent precursor to the error manifested in the output.
Knowledge fusion occurs hierarchically, with critical conflict resolution happening at the sentence-
level in intermediate layers, not merely at the output layer. (iii) Amplification of irrelevant context.
LLMs disproportionately amplify context that is irrelevant to the query but consistent with their
parametric knowledge, leading to confident yet erroneous generations.
Motivated by these findings, we propose a framework for RAG faithfulness, namedConflict-
Localized andEnhancedAttention forRAG (CLEAR). Specifically, CLEAR consists of three key
components: (i) Fine-grained knowledge pruning, which extracts knowledge from the context and
filters out irrelevant items; (ii) Hidden-state probing for conflict detection, which trains a probing
model for detecting knowledge conflict by observing hidden state; (iii) Conflict-Aware Fine-tuning,
which regularizes the LLM’s attention distribution via an attention guidance loss during fine-tuning.
In general, our contributions are summarized as follows:
• We conduct an in-depth analysis and reveal that LLMs integrate external knowledge
through a hierarchical mechanism, and that conflicting and aligned knowledge exhibit dis-
tinct distributional patterns within sentence-level representations.
• We propose CLEAR, a novel framework designed to enhance contextual faithfulness in
RAG systems. It employs probing techniques to accurately detect conflicting knowledge
and incorporates a conflict-aware fine-tuning strategy to guide the model to accurately in-
tegrate retrieved evidence with its parametric memory.
• We extensively evaluate the effectiveness of our framework on multiple RAG bench-
marks and diverse LLM architectures, demonstrating that CLEAR consistently outperforms
strong baselines across all evaluation metrics.
2 PRELIMINARYSTUDY
2.1 EXISTING CHALLENGES ONRAGFAITHFULNESS
We conducted a preliminary study to investigate the causes of contextual unfaithfulness in RAG. Two
key factors are hypothesized to underlie this issue: (i) irrelevant retrieval content, where passages
loosely related to the query introduce misleading information; (ii) knowledge conflict between the
context and the internal knowledge of the model, which leads the model to prioritize its parametric
memory over the retrieved evidence. To assess contextual faithfulness, we designed two controlled
2

Table 1: Case study illustrating two representative sources of contextual unfaithfulness in RAG. The
first case shows an error caused by focusing on irrelevant context. The second case demonstrates an
error caused by over-reliance on parametric knowledge.
Wrongly Based on
Irrelevant ContextQuestion:Is ibuprofen suitable for pregnant women?
Context:Ibuprofen is a commonly used over-the-counter pain reliever, often
used to alleviate headaches, toothaches, muscle aches, and menstrual cramps.
Reasoning:Based on the context, Ibuprofen is widely used among adults.
Answer:Ibuprofen is suitable for most people, including pregnant women.
Expected:Ibuprofen is not suitable for pregnant women.
Stubborn on Parametric
KnowledgeQuestion:Who is the current president of the United States?
Context:As of 2025, the President of the United States is Barack Obama,
reinstated following a vote by the Supreme Court to nullify the outgoing ad-
ministration’s election results... (manually modified)
Reasoning:I still think Joe Biden is the president. (trained on data up to 2023)
Answer:Joe Biden is the president of the United States.
Expected:According to the given context, Barack Obama is the current pres-
ident of the United States. (faithful to the context)
scenarios. In the first scenario, the original context is augmented with passages that are semantically
aligned with the query but topically irrelevant, introducing unrelated knowledge. In the second
scenario, selected entities in the context are altered to incorporate counterfactual knowledge, thereby
inducing knowledge conflict with the model’s internal knowledge acquired during pretraining.
Figure 1: Preliminary analysis of contextual un-
faithfulness in RAG reveals that all models de-
grade when (i) exposed to irrelevant knowledge
or (ii) confronted with conflicting knowledge.Performance Degradation in Both Scenar-
ios. Experimental results are presented in Fig-
ure 1. As shown, all models exhibit a de-
cline in accuracy under both conditions. In the
scenario with irrelevant retrieval content added
to the context, the accuracy of all three mod-
els dropped by over 10%, indicating that such
noisy inputs can mislead the models and neg-
atively affect their outputs. In contrast, the
introduction of conflicting knowledge resulted
in an even more pronounced performance de-
cline:LLaMA-3.1-8B-Instructexperi-
enced a 31% drop, andMistral-7B-v0.3
decreases by 24%. These results suggest
that contextual information contradicting the
model’s parametric knowledge has a substan-
tially greater impact on performance.
Error Analysis. Table 1 summarizes the pri-
mary causes of these errors. When the context
contains irrelevant information, the model often allocates attention to distracting noise, resulting in
incorrect responses. Additionally, when context conflicts with internal knowledge, the model tends
to favor parametric memory over provided evidence. These observations highlight two distinct yet
complementary challenges for RAG systems: sensitivity to irrelevant context and over-reliance on
internal knowledge in the presence of conflict.
2.2 HIERARCHICALKNOWLEDGEINTEGRATIONMECHANISM OFLLMS
To further explore how LLMs integrate external knowledge, we analyze hidden-state representa-
tions in the middle layers of LLMs. Inspired by hierarchical feature extraction in computer vision,
which also applies to language modeling, we observe that lower layers of LLMs primarily capture
token-level information, while deeper layers integrate sentence-level and passage-level semantics.
Our analysis reveals that most knowledge conflicts tend to manifest at the sentence-level factual rep-
resentations, where the hidden states of LLMs demonstrate discriminative features. Following the
method of (Xie et al., 2024), we extract the model’s parametric knowledgeK afor a given question,
and use an external LLM to construct corresponding conflicting knowledgeK c. Each knowledge
3

(a) LLaMA-3.1-8B-Instruct
 (b) Qwen3-8B
 (c) Mistral-7B-v0.3
(d) LLaMA-2-7B
 (e) Qwen2.5-7B-Instruct
 (f) Vicuna-7B-v1.5
Figure 2: t-SNE visualization of hidden-state patterns between aligned and conflicting knowledge.
There is a clear distinction in the distribution of hidden states between aligned and conflicting knowl-
edge. This observation provides empirical support for detecting knowledge conflicts based on hidden
state representations.
pair⟨K a, Kc⟩into the model separately. We extract the hidden states from the final decoder layer,
and perform a two-dimensional visualization using t-SNE (van der Maaten & Hinton, 2008). Totally,
we construct approximately 700 such samples and analyze six different model architectures.
As shown in Figure 2, the hidden-state distributions corresponding to aligned and conflicting knowl-
edge are distinguishable, forming distinct clusters represented by red and blue points. These results
suggest that knowledge conflicts frequently occur at the sentence level and can be detected through
the analysis of intermediate-layer hidden states. Inspired by this insight, we could train a probe
P(H K), whereH Kdenotes the hidden state induced by input knowledgeK, andPcan be imple-
mented as a Multi-Layer Perceptron (MLP) model (Rumelhart et al., 1986), to detect whether input
knowledge conflicts with parametric knowledge of the model. This requires only a single forward
pass to extract relevant hidden states, eliminating the need for explicit knowledge extraction.
3 METHODOLOGY
3.1 OVERVIEW
In this section, we introduce our proposed framework, CLEAR. As illustrated in Figure 3, CLEAR
comprises three principal modules: (i)Fine-Grained Knowledge Pruning: the retrieved context
is partitioned into fine-grained sentence-level knowledge, and irrelevant knowledge are pruned to
improve contextual fidelity and facilitate subsequent detection of knowledge conflicts; (ii)Hidden-
State Probing for Conflict Detection: an MLP probe is trained on hidden states extracted from
selected open-source LLMs to determine whether an input knowledge conflicts with the model’s
parametric knowledge; (iii)Conflict-Aware Fine-Tuning: the model is fine-tuned under a conflict-
aware supervision signal that conditions the model to appropriately reweight attention to conflicting
knowledge, thereby improving the faithfulness of generation. The following subsections provide
detailed descriptions of each module.
3.2 FINE-GRAINEDKNOWLEDGEPRUNING
Since knowledge conflicts typically manifest at the sentence level, we adopt a fine-grained decom-
position of the context to enable more precise conflict identification. At the same time, to mitigate
4

Figure 3: The overview of our proposed framework CLEAR, which consists of three main com-
ponents: (i)Fine-Grained Knowledge Pruning, which extracts knowledge from the context and
filters out irrelevant items; (ii)Hidden-State Probing for Conflict Detection, which trains a
probing model for detecting knowledge conflict by observing hidden state; (iii)Conflict-Aware
Fine-Tuning, which regularizes the LLM’s attention distribution on conflict content by fine-tuning
through an auxiliary attention loss.
the influence of irrelevant knowledge, we apply a pruning strategy to remove semantically unrelated
content. Specifically, we treat knowledge as the minimal processing granularity, where each cor-
responds to an independent, complete sentence-level statement that cannot be further decomposed.
For example, the sentence:“Riyad Mahrez is a professional footballer of Algerian descent who cur-
rently plays as a winger for Premier League club Leicester City and the Algeria national team. ”is
decomposed into three atomic knowledge items: 1.“Riyad Mahrez is a professional footballer of
Algerian descent. ”2.“Riyad Mahrez currently plays as a winger for Premier League club Leicester
City. ”3.“Riyad Mahrez currently plays as a winger for the Algeria national team. ”Each item
preserves the subject–predicate–object structure with necessary modifiers, ensuring no information
is lost during decomposition. To extract knowledge{K 1, K2, . . . , K n}from a given contextD, we
leverage the decomposition capabilities of an external LLM (we choose GPT-4o (OpenAI, 2024) for
its strong reasoning and text-processing abilities). Formally, we define this process as:
Decompose(D) ={K 1, K2, . . . , K n}
whereK idenotes thei-th knowledge item. Detailed prompt is provided in Appendix A.2.
After decomposition, we filter irrelevant knowledge to reduce contextual noise. For each knowledge
itemK i, we compute its semantic similarity with the queryQ:
f(Q, K i) =⟨q, k i⟩
whereq=Enc(Q)andk i=Enc(K i)are vector embeddings of the query and the knowledge
item, respectively, and⟨·,·⟩denotes cosine similarity. We employ the all-MiniLM-L6-v21encoder
for embedding generation. Finally, the knowledge items are ranked by similarity, and the top-k
results are selected as the pruned context.
3.3 HIDDEN-STATEPROBING FORCONFLICTDETECTION
To effectively handle knowledge conflicts, it is essential to first detect which retrieved knowledge
items contradict the model’s internal knowledge. To this end, we introduce a hidden-state prob-
ing module designed to detect knowledge items that contradict the model’s parametric knowledge.
Specifically, we adopt an MLP as the probing classifier, which takes as input the hidden represen-
tations from the final layer of the frozen LLM decoder. The probe consists of three fully connected
layers with non-linear activation functions, and outputs a binary prediction indicating whether a
knowledge item conflicts with the model’s internal knowledge. For training the probing classifier,
we leverage the MQuAKE dataset (Zhong et al., 2023), which is widely used in knowledge editing
research. We assume that the edited knowledge in MQuAKE inherently conflicts with the model’s
1https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2
5

original parametric knowledge, thereby providing natural pairs of aligned and conflicting knowledge
⟨Ka, Kc⟩. Importantly, the data format and textual granularity in MQuAKE align closely with the
knowledge items extracted in our framework, making it a suitable source for supervision.
During inference, each filtered knowledge item is passed through the model to obtain its hidden state
representation, which is subsequently classified by the probe:
M(K i)∈RdM,P 
M(K i)
∈ {0,1},
whereM(K i)denotes the hidden state of knowledge itemK iproduced by frozen modelMwith
dimensiond M.Pis the probing classifier that outputs a binary label indicating whether the knowl-
edge item conflicts with the model’s parametric knowledge. We mark the knowledge items identified
as conflicting with special tokens, i.e., wrapping them within⟨conflict⟩and⟨/conflict⟩. This ex-
plicit annotation enables the subsequent fine-tuning stage to be aware of which knowledge items are
in conflict with the model’s internal knowledge.
3.4 CONFLICT-AWAREFINE-TUNING
To explicitly encourage the model to allocate greater attention to conflicting knowledge items, we
propose Conflict-Aware Fine-Tuning. Unlike conventional Supervised Fine-Tuning, Conflict-Aware
Fine-Tuning incorporates an additional attention-guidance loss term that explicitly regularizes the
model’s attention distribution. Specifically, for each conflicting knowledge itemK i, we denote its
token sequence asT(i)={t(i)
1, t(i)
2, . . . , t(i)
m}. The positions of these tokens in the input context
are represented byS={j| ∃P(M(K i)) = 1, x j∈T(i)}, whereP(M(K i)) = 1indicates
that knowledge itemK iis judged as conflicting by the probe, andx jdenotes thej-th token of the
context. In practice, these positions inScan be directly identified via the previously introduced
special tokens⟨conflict⟩and⟨/conflict⟩.
Based on this alignment, we extract the attention weights from subsequent tokens attending to the
conflict-related tokens and compute the attention loss as:
LAttn=1
|P|X
(i,j)∈P(1−α ij),(i, j)∈P, P={(i, j)|i≥j;j∈S}
whereα ijdenotes the attention weight of tokenion tokenj. Finally, we combine the attention loss
with the standard language modeling objective through a weighted sum:
LTotal= (1−λ)L LM+λL Attn,
whereλ∈[0,1]balances the trade-off between language modeling fidelity and attention guidance.
This joint objective ensures that the model not only learns to generate faithful outputs but also
explicitly attends to conflicting knowledge items during training.
4 EXPERIMENT
4.1 EXPERIMENTALSETUP
In this section, we conduct a series of experiments to evaluate the effectiveness of CLEAR. We pro-
vide a comprehensive analysis of the experimental results, highlighting both the overall performance
improvements and the detailed behaviors of the model under different conditions.
Datasets.We evaluate CLEAR on three datasets. ConFiQA (Bi et al., 2024a) is a benchmark de-
signed to assess contextual faithfulness in question answering, particularly under real-world RAG
scenarios involving knowledge conflicts. It consists of three subsets: QA (Question Answering), MR
(Multi-hop Reasoning), and MC (Multi-Conflicts). The QA subset is a single-hop question answer-
ing task where the context contains a corresponding counterfactual, while MR and MC are multi-hop
reasoning tasks in which the context includes one and multiple counterfactuals, respectively. The
second dataset, Faitheval (Ming et al., 2024), introduces conflicts at the level of logical reasoning:
inconsistencies arise not from direct factual contradictions, but from reasoning chains that lead to
conflicting conclusions. Finally, we also evaluate on SQuAD (Rajpurkar et al., 2016), following the
version curated in KRE (Ying et al., 2023), which also incorporates fact-level knowledge conflicts.
6

Table 2: Performance comparison of methods grouped by Baseline, Prompt-Based, Decoding-
Based, and Training-Based. CLEAR consistently achieves the SOTA results.
Category MethodFaithEval ConFiQA (MC) ConFiQA (MR) ConFiQA (QA) SQuAD
F1 EM F1 EM F1 EM F1 EM F1 EM
LLaMA-3.1-8B-Instruct
BaselineNo-Context 27.7 6.0 5.0 2.1 6.1 1.9 6.1 1.3 8.9 1.2
Full-Context 66.9 53.1 28.0 22.5 50.3 41.3 58.5 49.0 64.5 46.0
Prompt-BasedOpin(Instr) (Zhou et al., 2023a) 34.9 15.1 67.4 57.3 65.9 54.0 76.9 67.4 66.0 47.7
KRE (Ying et al., 2023) 59.1 12.1 68.2 59.8 68.7 58.9 84.0 74.7 59.8 43.7
Decoding-BasedCOIECD (Yuan et al., 2024) 56.1 41.3 28.5 24.0 50.9 43.3 67.1 60.1 67.0 50.3
CAD (Shi et al., 2023a) 59.4 42.7 16.0 11.4 40.0 31.3 48.3 38.1 60.3 41.8
Training-BasedContext-DPO (Bi et al., 2024a) 67.2 53.7 76.9 67.7 78.5 66.9 83.7 76.7 64.4 45.8
CANOE (Si et al., 2025) 71.6 56.3 80.9 74.2 80.2 72.6 82.3 77.7 65.4 49.7
CLEAR(ours)74.4 64.4 89.2 87.7 89.7 87.0 93.1 91.7 68.4 53.3
Qwen3-8B
BaselineNo-Context 22.8 4.1 7.6 3.6 8.0 2.8 7.8 1.4 6.7 0.4
Full-Context 55.5 23.8 59.6 50.2 66.1 55.1 72.5 64.2 63.8 44.9
Prompt-BasedOpin(Instr) (Zhou et al., 2023a) 35.0 13.9 70.7 61.1 69.7 59.5 78.8 69.2 63.8 46.1
KRE (Ying et al., 2023) 58.1 12.3 67.5 59.1 68.4 59.0 80.4 67.3 48.6 29.7
Decoding-BasedCOIECD (Yuan et al., 2024) 66.6 56.4 66.7 60.8 71.5 63.8 78.5 73.6 69.7 55.2
CAD (Shi et al., 2023a) 57.0 28.7 57.7 48.3 64.8 53.3 71.0 62.0 63.6 44.5
Training-BasedContext-DPO (Bi et al., 2024a) 55.2 24.0 59.6 50.1 65.9 55.0 72.3 63.9 63.8 44.9
CANOE (Si et al., 2025) 70.3 60.2 85.2 81.7 84.6 80.7 92.2 86.5 69.4 53.4
CLEAR(ours)74.9 61.6 90.7 89.7 91.3 89.0 95.7 94.3 71.5 55.7
Mistral-7B-v0.3
BaselineNo-Context 26.2 4.4 4.4 0.9 4.9 0.5 6.1 1.0 8.1 1.0
Full-Context 68.8 37.7 25.6 12.5 37.8 21.5 58.5 44.0 56.4 37.5
Prompt-BasedOpin(Instr) (Zhou et al., 2023a) 35.7 14.1 58.8 44.1 57.8 52.5 76.4 65.5 58.1 37.4
KRE (Ying et al., 2023) 64.8 16.5 58.7 45.0 60.9 45.3 84.5 72.8 52.6 33.9
Decoding-BasedCOIECD (Yuan et al., 2024) 64.4 29.5 26.1 14.5 39.3 26.3 58.9 45.1 59.2 39.7
CAD (Shi et al., 2023a) 68.9 33.3 16.7 5.9 27.5 12.8 53.5 36.9 51.4 32.1
Training-BasedContext-DPO (Bi et al., 2024a) 64.9 31.8 44.8 28.3 50.9 31.9 66.4 52.7 56.6 37.6
CANOE (Si et al., 2025) 64.1 44.9 87.2 85.7 84.7 81.9 92.5 90.7 57.8 42.5
CLEAR(ours)74.9 62.9 91.2 89.7 90.8 88.2 95.1 93.7 68.1 53.6
Models and Baselines.For our experiments, we adopt several mainstream open-source models,
including Llama-3.1-8B-Instruct, Qwen3-8B, and Mistral-7B-v0.3. We compare CLEAR against
representative baseline methods from three major categories in the field of contextual faithfulness:
prompt-based approaches, decoding-based approaches, and training-based approaches. Among the
prompt-based methods, we include Opin(Instr) (Zhou et al., 2023a), KRE (Ying et al., 2023), and
FaithfulRAG (Zhang et al., 2025b). For decoding-based methods, we evaluate COIECD (Yuan
et al., 2024) and CAD (Shi et al., 2023a). For training-based methods, we compare against Con-
textDPO (Bi et al., 2024a) and CANOE (Si et al., 2025). Specifically, we partition the ConFiQA
dataset into training and test sets. All baselines that require training (including our proposed frame-
work) are trained on the ConFiQA training set, and evaluation is consistently performed on the test
set. Additional implementation details are provided in the Appendix A.2.
4.2 MAINRESULTS
In this section, we present the main experimental results. As shown in Table 2, our proposed method
CLEAR consistently achieves state-of-the-art performance across all datasets and model backbones.
On FaithEval and ConFiQA (MC, MR, QA), CLEAR demonstrates strong generalization ability to
both factual and logical conflicts, while on SQuAD, it further shows clear improvements in tradi-
tional retrieval-augmented settings. Moreover, the consistent gains under different backbone models
(LLaMA-3.1-8B-Instruct,Qwen3-8B, andMistral-7B-v0.3) highlight the robustness
and generalizability of our approach.
Specifically, onLLaMA-3.1-8B-Instruct, CLEAR achieves an F1 score of 74.4% and an EM
score of 64.4% on FaithEval, outperforming the strongest baseline CANOE (71.6% F1 / 56.3% EM)
by approximately +3% F1 and +8% EM. On ConFiQA sub-tasks, CLEAR improves over existing
methods by 3%–10% across MC, MR, and QA, further confirming its robustness in handling conflict
scenarios. Similarly, forQwen3-8B, CLEAR attains 74.9% F1 and 61.6% EM on FaithEval, yield-
ing substantial gains compared with prior methods, and reaches 90.7% F1 and 89.7% EM on the MC
7

Table 3: Ablation study result. As shown in the figure, the ablation of each module significantly
impacts the results. Among them, the Conflict Detection module has the most substantial influence
on the entire framework.
Models ModulesFaitheval ConFiQA (MC) ConFiQA (MR) ConFiQA (QA) SQuAD
F1 EM F1 EM F1 EM F1 EM F1 EM
LLaMA-3.1-8B-InstructCLEAR 74.4 64.4 89.2 87.7 89.7 87.0 93.1 91.7 68.4 53.3
w/o Knowledge Pruning 62.1 48.4 81.1 79.4 84.4 80.8 88.5 87.5 59.2 45.0
w/o Conflict Detection 61.7 47.6 81.4 79.3 83.9 79.9 87.6 86.4 58.1 44.1
w/o Fine-Tuning 61.5 50.9 83.8 80.4 85.0 81.0 87.5 86.4 58.2 40.2
Qwen3-8BCLEAR 74.9 61.6 90.7 89.7 91.3 89.0 95.7 94.3 71.5 55.7
w/o Knowledge Pruning 62.6 50.9 86.1 85.3 86.7 85.2 88.8 87.8 66.3 51.3
w/o Conflict Detection 61.0 49.8 85.4 84.6 86.6 85.1 88.6 87.5 66.1 51.0
w/o Fine-Tuning 64.0 54.2 86.2 84.8 86.1 84.3 89.6 88.5 66.1 51.5
Mistral-7B-v0.3CLEAR 74.9 62.9 91.2 89.7 90.8 88.2 95.1 93.7 68.1 53.6
w/o Knowledge Pruning 69.5 58.5 86.6 85.5 86.2 84.7 88.4 87.1 62.9 48.7
w/o Conflict Detection 68.4 56.4 85.2 84.1 84.4 82.9 87.4 86.2 61.8 47.6
w/o Fine-Tuning 69.3 57.6 88.8 86.1 86.3 81.8 81.4 77.4 59.7 49.8
task, which sets a new performance benchmark. OnMistral-7B-v0.3, CLEAR achieves 74.9%
F1 / 62.9% EM on FaithEval and strong improvements across ConFiQA and SQuAD, surpassing the
best training-based baselines by a clear margin.
Taken together, these results demonstrate that CLEAR not only excels on datasets designed to eval-
uate contextual faithfulness under knowledge conflicts but also delivers significant benefits in stan-
dard QA tasks. The consistent improvements across multiple datasets, conflict types, and backbone
LLMs underscore the effectiveness, robustness, and general applicability of our method.
4.3 ABLATIONSTUDY
To assess the contribution of each component in our framework, we conducted ablation exper-
iments by individually removing the knowledge pruning, conflict detection, and Conflict-Aware
Fine-Tuning modules. The results across each benchmark are summarized in Table 3. Overall, we
observe that all three components play a non-negligible role: removing any single module consis-
tently reduces performance, typically by around 10% on both F1 and EM.
When the knowledge pruning module is removed, the model is forced to judge conflicts against every
sentence in the context. Such coarse-grained filtering leads to incomplete contextual information
and degrades the model’s ability to resolve fine-grained conflicts, thereby diminishing contextual
faithfulness. More critically, removing the conflict detection module results in the most significant
performance drop. Without explicit conflict detection, the downstream Conflict-Aware Fine-Tuning
becomes ineffective, since there are no identified conflicting items to which the model can attend,
making the training process indistinguishable from standard SFT. Finally, removing Conflict-Aware
Fine-Tuning also results in substantial degradation. Even when conflicts are annotated, the model
struggles to prioritize them during inference due to its inherent tendency to rely on its parametric
knowledge. This indicates that Conflict-Aware Fine-Tuning is essential for effectively aligning the
model’s attention to conflicting knowledge and improving contextual faithfulness.
4.4 IMPACT OFαONATTENTIONWEIGHTS
To further investigate the effect of the hyperparameterαintroduced in the Conflict-Aware Fine-
Tuning module, we conduct experiments with multiple values ofαand analyze both the attention
weights assigned to conflicting knowledge and the corresponding model performance. As shown in
Figure 4, increasingαconsistently raises the model’s attention to conflicting knowledge, with the
growth curve gradually flattening and stabilizing around 0.5. However, model performance does
not follow the same trend. Instead, performance peaks whenαis in the range of 0.1 to 0.3, after
which it declines asαcontinues to increase. This observation indicates that higher attention to con-
flicting knowledge does not necessarily lead to better performance. While attending to conflicting
knowledge is crucial, the model must also balance its focus on the question itself and other rele-
vant contextual information. Excessive emphasis on conflicting knowledge can ultimately harm the
model’s ability to generate accurate answers.
8

Figure 4: Impact ofαon accuracy (blue) and attention weight on conflicting knowledge (red) across
different models. Results show that increasingαconsistently increases the attention weight assigned
to conflicting knowledge. Model performance peaks at smallerαvalues (0.1 to 0.3) and then de-
clines, indicating that excessive focus on conflicting knowledge can negatively affect performance.
5 RELATEDWORK
Due to space limitations, we provide only a concise overview of the related work here, while a more
detailed discussion can be found in Appendix E.
Retrieval-Augmented Generation. Retrieval-Augmented Generation (RAG) has emerged as a
prominent paradigm for enhancing the factual accuracy and temporal relevance of Large Language
Models (LLMs) by incorporating external knowledge sources (Xiang et al., 2025; Chen et al., 2025;
Xiao et al., 2025). Early works such as REALM (Guu et al., 2020c) and RAG (Lewis et al., 2020)
introduced end-to-end frameworks that retrieve relevant passages from large corpora to assist gener-
ation. Subsequent research has explored improvements in both the retriever and generator modules,
including dense retrieval techniques (Karpukhin et al., 2020; Izacard et al., 2023), adaptive retrieval
strategies (Sun et al., 2022), and hybrid models combining retrieval with parametric memory (Shi
et al., 2023b).
Contextual Faithfulness. Contextual faithfulness refers to the alignment between the generated
output and the provided context, which is especially critical in RAG settings (Huang et al., 2025).
Prompt-based methods design templates or self-reflection mechanisms to encourage faithful use of
context (Asai et al., 2023; Ying et al., 2024). Decoding-based methods modify generation strategies
to enhance the influence of the retrieved context (Yuan et al., 2024; Shi et al., 2023a). Reinforcement
learning frameworks such as CANOE (Si et al., 2025) and Context-DPO (Bi et al., 2024a) employ
an end-to-end paradigm to optimize the generation process and reward contextual faithful response.
Knowledge Conflict. Knowledge conflict refers to scenarios in RAG or related settings where
the retrieved external information contradicts a model’s internal parametric knowledge, or where
different external sources conflict with one another. Astute RAG (Wang et al., 2025a) proposes
a framework to consolidate internal and external knowledge with source-awareness and reliability
estimation; FaithfulRAG (Zhang et al., 2025b) introduces fact-level conflict modeling and a self-
thinking process to resolve contradictions; Swin-VIB (Wang et al., 2025b) uses information bottle-
neck techniques to guide preference in ambiguous conflict settings; and broader surveys like Xu
et al. (2024b) clarify conflict categories and recommend robust evaluation frameworks.
6 CONCLUSION
In this work, we tackled the persistent challenge of contextual faithfulness in RAG, with a focus
on how LLMs internally reconcile retrieved evidence with their parametric memory under knowl-
edge conflicts. Through probing-based analysis of hidden-state representations, we uncovered three
key insights: knowledge integration occurs hierarchically, conflicts are encoded as latent signals at
the sentence level, and irrelevant context can be amplified when aligned with parametric knowl-
edge. Building on these findings, we introduced CLEAR, a framework that combines fine-grained
knowledge pruning, hidden-state probing, and conflict-aware fine-tuning to enhance both robustness
and contextual fidelity. Comprehensive experiments across multiple benchmarks and large language
9

models demonstrate that CLEAR consistently outperforms strong baselines, achieving state-of-the-
art performance under diverse conflict conditions. Beyond advancing the accuracy of RAG systems,
our framework highlights the importance of explicitly modeling and mitigating knowledge conflicts,
offering a principled direction for future research on reliable knowledge integration in LLMs.
7 ETHICS STATEMENT
This work does not involve any experiments with human subjects, sensitive personal data, or infor-
mation that could identify individuals. All datasets used in our experiments are publicly available
and commonly adopted in prior research. We carefully follow dataset licenses and ensure that no
proprietary or private information is disclosed. Our proposed method is designed for advancing
the understanding of retrieval-augmented generation and does not raise foreseeable risks of harmful
applications. We acknowledge potential concerns regarding bias and fairness in language models
and retrieval corpora, and we provide detailed dataset descriptions and preprocessing steps in the
appendix to facilitate transparent evaluation.
8 REPRODUCIBILITYSTATEMENT
We make significant efforts to ensure the reproducibility of our work. The details of model ar-
chitectures, hyperparameters, and training settings are provided in Section 4.1 of the main paper.
Additional implementation details and full experimental setups are provided in Appendix A.2. To
further support reproducibility, we release anonymized source code and configuration files as sup-
plementary materials. Together, these resources allow researchers to fully reproduce our results and
extend our findings.
REFERENCES
Akari Asai, Zeqiu Wu, Yizhong Wang, Avirup Sil, and Hannaneh Hajishirzi. Self-rag: Learning
to retrieve, generate, and critique through self-reflection, 2023. URLhttps://arxiv.org/
abs/2310.11511.
Baolong Bi, Shaohan Huang, Yiwei Wang, Tianchi Yang, Zihan Zhang, Haizhen Huang, Lingrui
Mei, Junfeng Fang, Zehao Li, Furu Wei, et al. Context-dpo: Aligning language models for
context-faithfulness.arXiv preprint arXiv:2412.15280, 2024a.
Baolong Bi, Shenghua Liu, Yiwei Wang, Lingrui Mei, Junfeng Fang, Hongcheng Gao, Shiyu Ni,
and Xueqi Cheng. Is factuality enhancement a free lunch for llms? better factuality can lead to
worse context-faithfulness.arXiv preprint arXiv:2404.00216, 2024b.
Sebastian Borgeaud, Arthur Mensch, Jordan Hoffmann, Trevor Cai, Eliza Rutherford, Katie Milli-
can, George Bm Van Den Driessche, Jean-Baptiste Lespiau, Bogdan Damoc, Aidan Clark, et al.
Improving language models by retrieving from trillions of tokens. InInternational conference on
machine learning, pp. 2206–2240. PMLR, 2022.
Jiajing Chen, Bingying Liu, Xiaoxuan Liao, Jia Gao, Hongye Zheng, and Yue Li. Adaptive opti-
mization for enhanced efficiency in large-scale language model training. In2024 6th International
Conference on Frontier Technologies of Information and Computer (ICFTIC), pp. 1315–1319.
IEEE, 2024.
Shengyuan Chen, Chuang Zhou, Zheng Yuan, Qinggang Zhang, Zeyang Cui, Hao Chen, Yilin Xiao,
Jiannong Cao, and Xiao Huang. You don’t need pre-built graphs for rag: Retrieval augmented
generation with adaptive reasoning structures.arXiv preprint arXiv:2508.06105, 2025.
Tobias Falke, Leonardo FR Ribeiro, Prasetya Ajie Utama, Ido Dagan, and Iryna Gurevych. Ranking
generated summaries by correctness: An interesting but challenging application for natural lan-
guage inference. InProceedings of the 57th annual meeting of the association for computational
linguistics, pp. 2214–2220, 2019.
Angela Fan, Yacine Jernite, Ethan Perez, David Grangier, Jason Weston, and Michael Auli. Eli5:
Long form question answering.arXiv preprint arXiv:1907.09190, 2019.
10

Zhangyin Feng, Xiaocheng Feng, Dezhi Zhao, Maojin Yang, and Bing Qin. Retrieval-generation
synergy augmented large language models. InICASSP 2024-2024 IEEE International Conference
on Acoustics, Speech and Signal Processing (ICASSP), pp. 11661–11665. IEEE, 2024.
Kaiyuan Gao, Sunan He, Zhenyu He, Jiacheng Lin, QiZhi Pei, Jie Shao, and Wei Zhang. Exam-
ining user-friendly and open-sourced large gpt models: A survey on language, multimodal, and
scientific gpt models.arXiv preprint arXiv:2308.14149, 2023.
Zishan Guo, Renren Jin, Chuang Liu, Yufei Huang, Dan Shi, Linhao Yu, Yan Liu, Jiaxuan Li,
Bojian Xiong, Deyi Xiong, et al. Evaluating large language models: A comprehensive survey.
arXiv preprint arXiv:2310.19736, 2023.
Kelvin Guu, Kenton Lee, Zora Tung, Panupong Pasupat, and Ming-Wei Chang. Realm: Retrieval-
augmented language model pre-training, 2020a. URLhttps://arxiv.org/abs/2002.
08909.
Kelvin Guu, Kenton Lee, Zora Tung, Panupong Pasupat, and Mingwei Chang. Retrieval augmented
language model pre-training. InInternational conference on machine learning, pp. 3929–3938.
PMLR, 2020b.
Kelvin Guu, Kenton Lee, Zora Tung, Panupong Pasupat, and Mingwei Chang. Retrieval augmented
language model pre-training. InInternational conference on machine learning, pp. 3929–3938.
PMLR, 2020c.
Pengcheng Huang, Zhenghao Liu, Yukun Yan, Haiyan Zhao, Xiaoyuan Yi, Hao Chen, Zhiyuan Liu,
Maosong Sun, Tong Xiao, Ge Yu, and Chenyan Xiong. Parammute: Suppressing knowledge-
critical ffns for faithful retrieval-augmented generation, 2025. URLhttps://arxiv.org/
abs/2502.15543.
Gautier Izacard and Edouard Grave. Leveraging passage retrieval with generative models for open
domain question answering.arXiv preprint arXiv:2007.01282, 2020.
Gautier Izacard, Patrick Lewis, Maria Lomeli, Lucas Hosseini, Fabio Petroni, Timo Schick, Jane
Dwivedi-Yu, Armand Joulin, Sebastian Riedel, and Edouard Grave. Atlas: Few-shot learning
with retrieval augmented language models.Journal of Machine Learning Research, 24(251):
1–43, 2023.
Ziwei Ji, Nayeon Lee, Rita Frieske, Tiezheng Yu, Dan Su, Yan Xu, Etsuko Ishii, Ye Jin Bang,
Andrea Madotto, and Pascale Fung. Survey of hallucination in natural language generation.ACM
computing surveys, 55(12):1–38, 2023.
Vladimir Karpukhin, Barlas Oguz, Sewon Min, Patrick SH Lewis, Ledell Wu, Sergey Edunov, Danqi
Chen, and Wen-tau Yih. Dense passage retrieval for open-domain question answering. InEMNLP
(1), pp. 6769–6781, 2020.
Patrick Lewis, Ethan Perez, Aleksandra Piktus, Fabio Petroni, Vladimir Karpukhin, Naman Goyal,
Heinrich K ¨uttler, Mike Lewis, Wen-tau Yih, Tim Rockt ¨aschel, et al. Retrieval-augmented gener-
ation for knowledge-intensive nlp tasks.Advances in neural information processing systems, 33:
9459–9474, 2020.
Yifei Ming, Senthil Purushwalkam, Shrey Pandit, Zixuan Ke, Xuan-Phi Nguyen, Caiming Xiong,
and Shafiq Joty. Faitheval: Can your language model stay faithful to context, even if” the moon
is made of marshmallows”.arXiv preprint arXiv:2410.03727, 2024.
Rodrigo Nogueira and Kyunghyun Cho. Passage re-ranking with bert.arXiv preprint
arXiv:1901.04085, 2019.
OpenAI. Gpt-4o system card, 2024. System Card overview of GPT-4o’s capabilities, limitations,
and safety evaluations.
Fabio Petroni, Aleksandra Piktus, Angela Fan, Patrick Lewis, Majid Yazdani, Nicola De Cao, James
Thorne, Yacine Jernite, Vladimir Karpukhin, Jean Maillard, et al. Kilt: a benchmark for knowl-
edge intensive language tasks.arXiv preprint arXiv:2009.02252, 2020.
11

Pranav Rajpurkar, Jian Zhang, Konstantin Lopyrev, and Percy Liang. Squad: 100,000+ questions
for machine comprehension of text.arXiv preprint arXiv:1606.05250, 2016.
Ori Ram, Yoav Levine, Itay Dalmedigos, Dor Muhlgay, Amnon Shashua, Kevin Leyton-Brown, and
Yoav Shoham. In-context retrieval-augmented language models.Transactions of the Association
for Computational Linguistics, 11:1316–1331, 2023.
David E Rumelhart, Geoffrey E Hinton, and Ronald J Williams. Learning representations by back-
propagating errors.Nature, 323(6088):533–536, 1986.
Keshav Santhanam, Omar Khattab, Jon Saad-Falcon, Christopher Potts, and Matei Zaharia.
Colbertv2: Effective and efficient retrieval via lightweight late interaction.arXiv preprint
arXiv:2112.01488, 2021.
Weijia Shi, Xiaochuang Han, Mike Lewis, Yulia Tsvetkov, Luke Zettlemoyer, and Scott Wen-tau
Yih. Trusting your evidence: Hallucinate less with context-aware decoding.arXiv preprint
arXiv:2305.14739, 2023a.
Weijia Shi, Sewon Min, Michihiro Yasunaga, Minjoon Seo, Rich James, Mike Lewis, Luke Zettle-
moyer, and Wen-tau Yih. Replug: Retrieval-augmented black-box language models.arXiv
preprint arXiv:2301.12652, 2023b.
Kurt Shuster, Mojtaba Komeili, Leonard Adolphs, Stephen Roller, Arthur Szlam, and Jason Weston.
Language models that seek for knowledge: Modular search & generation for dialogue and prompt
completion.arXiv preprint arXiv:2203.13224, 2022.
Shuzheng Si, Haozhe Zhao, Cheng Gao, Yuzhuo Bai, Zhitong Wang, Bofei Gao, Kangyang Luo,
Wenhao Li, Yufei Huang, Gang Chen, et al. Teaching large language models to maintain contex-
tual faithfulness via synthetic tasks and reinforcement learning.arXiv preprint arXiv:2505.16483,
2025.
Zhiqing Sun, Xuezhi Wang, Yi Tay, Yiming Yang, and Denny Zhou. Recitation-augmented language
models.arXiv preprint arXiv:2210.01296, 2022.
James Thorne, Andreas Vlachos, Christos Christodoulopoulos, and Arpit Mittal. Fever: a large-scale
dataset for fact extraction and verification.arXiv preprint arXiv:1803.05355, 2018.
Laurens van der Maaten and Geoffrey Hinton. Visualizing data using t-sne.Journal of Machine
Learning Research, 9(Nov):2579–2605, 2008.
Fei Wang, Xingchen Wan, Ruoxi Sun, Jiefeng Chen, and Sercan O Arik. Astute RAG: Over-
coming imperfect retrieval augmentation and knowledge conflicts for large language models.
In Wanxiang Che, Joyce Nabende, Ekaterina Shutova, and Mohammad Taher Pilehvar (eds.),
Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics (Vol-
ume 1: Long Papers), pp. 30553–30571, Vienna, Austria, July 2025a. Association for Compu-
tational Linguistics. ISBN 979-8-89176-251-0. doi: 10.18653/v1/2025.acl-long.1476. URL
https://aclanthology.org/2025.acl-long.1476/.
Jiatai Wang, Zhiwei Xu, Di Jin, Xuewen Yang, and Tao Li. Accommodate knowledge conflicts
in retrieval-augmented llms: Towards reliable response generation in the wild.arXiv preprint
arXiv:2504.12982, 2025b.
Zhishang Xiang, Chuanjie Wu, Qinggang Zhang, Shengyuan Chen, Zijin Hong, Xiao Huang, and
Jinsong Su. When to use graphs in rag: A comprehensive analysis for graph retrieval-augmented
generation.arXiv preprint arXiv:2506.05690, 2025.
Yilin Xiao, Chuang Zhou, Qinggang Zhang, Su Dong, Shengyuan Chen, and Xiao Huang. Lag:
Logic-augmented generation from a cartesian perspective.arXiv preprint arXiv:2508.05509,
2025.
Jian Xie, Kai Zhang, Jiangjie Chen, Renze Lou, and Yu Su. Adaptive chameleon or stubborn sloth:
Revealing the behavior of large language models in knowledge conflicts. InProceedings of ICLR,
2024.
12

Peng Xu, Wei Ping, Xianchao Wu, Lawrence McAfee, Chen Zhu, Zihan Liu, Sandeep Subramanian,
Evelina Bakhturina, Mohammad Shoeybi, and Bryan Catanzaro. Retrieval meets long context
large language models.arXiv preprint arXiv:2310.03025, 2023.
Rongwu Xu, Zehan Qi, Zhijiang Guo, Cunxiang Wang, Hongru Wang, Yue Zhang, and Wei
Xu. Knowledge conflicts for LLMs: A survey. In Yaser Al-Onaizan, Mohit Bansal, and
Yun-Nung Chen (eds.),Proceedings of the 2024 Conference on Empirical Methods in Natu-
ral Language Processing, pp. 8541–8565, Miami, Florida, USA, November 2024a. Associa-
tion for Computational Linguistics. doi: 10.18653/v1/2024.emnlp-main.486. URLhttps:
//aclanthology.org/2024.emnlp-main.486/.
Rongwu Xu, Zehan Qi, Zhijiang Guo, Cunxiang Wang, Hongru Wang, Yue Zhang, and Wei Xu.
Knowledge conflicts for llms: A survey.arXiv preprint arXiv:2403.08319, 2024b.
Xinrun Xu, Yuxin Wang, Chaoyi Xu, Ziluo Ding, Jiechuan Jiang, Zhiming Ding, and B ¨orje F Karls-
son. A survey on game playing agents and large models: Methods, applications, and challenges.
arXiv preprint arXiv:2403.10249, 2024c.
Jiahao Ying, Yixin Cao, Kai Xiong, Yidong He, Long Cui, and Yongbin Liu. Intuitive or dependent?
investigating llms’ behavior style to conflicting prompts.arXiv preprint arXiv:2309.17415, 2023.
Jiahao Ying, Yixin Cao, Kai Xiong, Long Cui, Yidong He, and Yongbin Liu. Intuitive or depen-
dent? investigating LLMs’ behavior style to conflicting prompts. In Lun-Wei Ku, Andre Martins,
and Vivek Srikumar (eds.),Proceedings of the 62nd Annual Meeting of the Association for Com-
putational Linguistics (Volume 1: Long Papers), pp. 4221–4246, Bangkok, Thailand, August
2024. Association for Computational Linguistics. doi: 10.18653/v1/2024.acl-long.232. URL
https://aclanthology.org/2024.acl-long.232/.
Xiaowei Yuan, Zhao Yang, Yequan Wang, Shengping Liu, Jun Zhao, and Kang Liu. Discerning and
resolving knowledge conflicts through adaptive decoding with contextual information-entropy
constraint. In Lun-Wei Ku, Andre Martins, and Vivek Srikumar (eds.),Findings of the Asso-
ciation for Computational Linguistics: ACL 2024, pp. 3903–3922, Bangkok, Thailand, August
2024. Association for Computational Linguistics. doi: 10.18653/v1/2024.findings-acl.234. URL
https://aclanthology.org/2024.findings-acl.234/.
Qinggang Zhang, Shengyuan Chen, Yuanchen Bei, Zheng Yuan, Huachi Zhou, Zijin Hong, Junnan
Dong, Hao Chen, Yi Chang, and Xiao Huang. A survey of graph retrieval-augmented generation
for customized large language models.arXiv preprint arXiv:2501.13958, 2025a.
Qinggang Zhang, Zhishang Xiang, Yilin Xiao, Le Wang, Junhui Li, Xinrun Wang, and Jinsong
Su. Faithfulrag: Fact-level conflict modeling for context-faithful retrieval-augmented generation.
arXiv preprint arXiv:2506.08938, 2025b.
Zihan Zhang, Meng Fang, and Ling Chen. Retrievalqa: Assessing adaptive retrieval-augmented
generation for short-form open-domain question answering.arXiv preprint arXiv:2402.16457,
2024.
Zexuan Zhong, Zhengxuan Wu, Christopher D Manning, Christopher Potts, and Danqi Chen.
MQuAKE: Assessing knowledge editing in language models via multi-hop questions.arXiv
preprint arXiv:2305.14795, 2023.
Wenxuan Zhou, Sheng Zhang, Hoifung Poon, and Muhao Chen. Context-faithful prompting for
large language models. In Houda Bouamor, Juan Pino, and Kalika Bali (eds.),Findings of the As-
sociation for Computational Linguistics: EMNLP 2023, pp. 14544–14556, Singapore, December
2023a. Association for Computational Linguistics. doi: 10.18653/v1/2023.findings-emnlp.968.
URLhttps://aclanthology.org/2023.findings-emnlp.968/.
Wenxuan Zhou, Sheng Zhang, Hoifung Poon, and Muhao Chen. Context-faithful prompting for
large language models.arXiv preprint arXiv:2303.11315, 2023b.
13

A FREQUENTLYASKEDQUESTIONS(FAQS)
A.1 ALGORITHMICDESCRIPTION OFCLEAR
The following presents the algorithmic description of the CLEAR framework, which is implemented
as a three-step pipeline. First, the retrieved context is decomposed into fine-grained knowledge,
from which the most relevant ones are selected based on query–knowledge similarity. Second,
a hidden-state probing classifier detects conflicts between the selected knowledge and the model’s
internal knowledge, and conflicting knowledge is explicitly annotated with special tokens. Third, we
introduce conflict-aware supervised fine-tuning (CA-SFT), which reinforces the model’s attention on
the annotated conflict tokens by incorporating an auxiliary attention-guidance loss into the training
objective. The fine-tuned model then generates the final answer conditioned on the pruned and
annotated context, enabling more faithful response generation.
Input:QuestionQ, retrieved contextD={d 1, d2, . . . , d n}, modelM
Output:AnswerA
Step 1: Fine-Grained Knowledge Pruning
Decompose retrieved context into atomic knowledge:
{K1, K2, . . . , K m}=Decompose(D)
Compute similarity between query and each knowledge item:
f(Q, K i) =⟨Enc(Q), Enc(K i)⟩
Select top-kknowledge items by similarity:
D′={K′
1, K′
2, . . . , K′
k}
Step 2: Hidden-State Probing for Conflict Detection
foreachK′
i∈D′do
Obtain hidden representation from frozen model:
hi=M(K′
i)∈RdM
Classify conflict via probing modelP:
yi=P(h i)∈ {0,1}
ifyi= 1then
MarkK′
iwith special tokens⟨conflict⟩K′
i⟨/conflict⟩
end
end
Step 3: Conflict-Aware Supervised Fine-Tuning (CA-SFT)
foreachconflicting knowledge itemK′
ido
Identify token positionsS={j|x j∈T(i)}
Compute attention-guidance loss:
LAttn=1
|P|X
(i,j)∈P(1−α ij), P={(i, j)|i≥j;j∈S}
end
Combine with language modeling loss:
LTotal= (1−λ)L LM+λL Attn
Final Answer Generation
Generate final answerAusing fine-tuned modelM CA-SFT conditioned on pruned and
annotated contextD′.
Algorithm 1:CLEAR: Conflict-Localized and Enhanced Attention for RAG
14

Figure 5: Context decomposition prompt used in the Fine-Grained Knowledge Pruning module.
A.2 IMPLEMENTATIONDETAILS
Detail of CLEAR.For the implementation of CLEAR, we configure the experimental settings as
follows. In theFine-Grained Knowledge Pruningmodule, we employ gpt-3.5-turbo to decompose
the retrieved context into fine-grained knowledge using the prompt template illustrated in Figure 5.
We then compute semantic similarity among the decomposed knowledge with all-MiniLM-L6-v2
and retain the top-10 most relevant knowledge item.
In theHidden-State Probing for Conflict Detectionmodule, the selected knowledge items are fed
into the model, from which we extract hidden states of the decoder. These representations are passed
to a trained MLP-based probe for binary classification. The probe consists of three fully connected
layers with ReLU activation, followed by a sigmoid normalization. For training, we sample 1,000
instances with a learning rate of 0.001 and train the probe for 10 epochs.
For theConflict-Aware Fine-Tuningmodule, we set the weighting hyperparameterλ= 0.1. On
the ConFiQA dataset, we allocate 13,500 instances for training (with 4,500 samples each from the
MC, MR, and QA subsets), while the remaining data are reserved for evaluation. We fine-tune the
model using LoRA, where the rankris set to 16, the scaling factorαto 16, and the learning rate to
3×10−5, training for a total of 5 epochs. Finally, during inference, we set the temperature parameter
to 0 to ensure reproducibility of results.
Detail of Baseline.For all baselines reported in the main experiments, we adopt a sampling tem-
perature of 0 and a maximum generation length of 128 tokens. For CAD, we set the hyperparameter
α= 0.9. For all prompt-based methods, we directly employ the prompt templates provided in the
original papers. For all training-based methods, we use the same training data as CLEAR, sam-
pled from ConFiQA. Specifically, for Context-DPO, we apply the same LoRA configuration during
training. For CANOE, we follow the original training setup and perform full-parameter fine-tuning
on four NVIDIA A100 GPUs.
Detail of Ablation Study.For thew/o Knowledge Pruningvariant, we partition the input context
directly into sentences and subsequently apply the conflict detection module to determine whether
each sentence conflicts with the model’s parametric knowledge. For thew/o Conflict Detection
variant, we fine-tune the model using the decomposed knowledge directly. Since conflicting knowl-
edge is not explicitly identified, only the loss termL LMis active during CA-SFT fine-tuning. For
thew/o CA-SFTvariant, we remove theL Attnterm, which reduces the training objective to standard
SFT without attention-level supervision.
15

Table 4: Supplementary experimental results on additional model architectures.
MethodFaithEval ConFiQA (MC) ConFiQA (MR) ConFiQA (QA) SQuAD
F1 EM F1 EM F1 EM F1 EM F1 EM
LLaMA-2-7B-Chat-HF
Context-DPO 63.2 50.7 57.9 32.0 58.5 32.7 73.7 64.7 62.4 41.8
CANOE70.652.3 73.970.275.2 72.6 74.3 72.7 63.2 45.6
CLEAR 68.354.4 79.169.780.2 77.0 86.1 81.7 65.4 52.1
Qwen2.5-7B-Instruct
Context-DPO 65.1 50.2 62.7 53.7 71.1 58.8 75.0 66.3 55.2 36.4
CANOE68.1 53.968.7 61.1 71.7 67.8 70.6 66.9 59.4 41.3
CLEAR 63.5 48.988.8 86.2 89.6 86.2 94.3 91.5 61.6 46.2
Table 5: Accuracy and Attention Weight across differentαvalues for three models.
αLLaMA-3.1-8B-Instruct Qwen3-8B Mistral-7B-v0.3
Accuracy Attention Accuracy Attention Accuracy Attention
0.0 0.552 0.020 0.512 0.115 0.631 0.070
0.1 0.644 0.105 0.604 0.022 0.663 0.193
0.2 0.632 0.188 0.573 0.195 0.598 0.203
0.3 0.635 0.231 0.639 0.283 0.559 0.211
0.4 0.554 0.331 0.495 0.415 0.611 0.314
0.5 0.482 0.442 0.443 0.390 0.538 0.463
0.6 0.333 0.464 0.430 0.381 0.289 0.543
0.7 0.201 0.483 0.153 0.474 0.171 0.549
0.8 0.214 0.481 0.211 0.444 0.117 0.459
0.9 0.210 0.464 0.207 0.457 0.194 0.538
B ADDITIONALEXPERIMENT
B.1ADDITIONAL MODEL ARCHITECTURE FOR MAIN EXPERIMENT
Table 4 presents supplementary results on two additional model architectures,LLaMA-2-7B-Chat-
HFandQwen2.5-7B-Instruct, evaluated across multiple benchmarks. Consistent with the main
findings, CLEAR demonstrates notable improvements over both Context-DPO and CANOE, partic-
ularly on conflict-sensitive datasets such as ConFiQA and FaithEval. For LLaMA-2-7B-Chat-HF,
CLEAR achieves the highest scores on most ConFiQA variants, while also maintaining competitive
performance on FaithEval and SQuAD.
On Qwen2.5-7B-Instruct, the advantage of CLEAR becomes even more pronounced: it consistently
outperforms both baselines across all ConFiQA settings, with substantial gains in F1 and EM. Al-
though CANOE occasionally remains competitive on less conflict-intensive benchmarks, CLEAR
shows strong generalization in resolving conflicting knowledge. These results confirm that the effec-
tiveness of CLEAR extends beyond a single backbone, underscoring its robustness across different
instruction-tuned LLMs.
B.2 SUPPLEMENTARY EXPERIMENTAL RESULTS ON ATTENTION ANALYSIS
Table 5 reports the detailed numerical results corresponding to Figure 4, including both the model
accuracy and the attention weight assigned to conflicting knowledge across different values ofα
for LLaMA-3.1-8B-Instruct, Qwen3-8B, and Mistral-7B-v0.3. Consistent with the trends shown in
the figure, attention weights increase steadily with largerα, saturating aroundα= 0.5. In contrast,
accuracy peaks within a smaller range ofα(0.1–0.3) and then declines asαcontinues to grow. These
results highlight that while higherαvalues encourage stronger focus on conflicting knowledge, this
emphasis can come at the cost of overall performance. The tabulated results thus provide a more
fine-grained view of the trade-off between model attention allocation and accuracy under varyingα
values.
16

Table 6: Case Study. This table displays the knowledge extracted from the context and the results
of identifying knowledge conflicts. Based on the conflicting knowledge, the model can correctly
answer questions (even when the golden answer is counterfactual).
QuestionA group of engineers wanted to know how different building designs would respond during an
earthquake. They made several models of buildings and tested each for its ability to withstand
earthquake conditions. Which will most likely result from testing different building designs?
ContextSeismic testing of building models is crucial for understanding how structures will behave dur-
ing earthquakes. Engineers approach these tests with a myriad of designs, each aiming to im-
prove certain aspects of building performance, such as safety, aesthetic appeal, and construction
speed...
Knowledge
Extracted(1) Seismic testing of building models is crucial for understanding structural behavior during
earthquakes.
(2) Engineers approach tests with a myriad of designs aiming to improve safety, aesthetic ap-
peal, and construction speed.
(3)⟨conflict⟩Implementation of efficient techniques can enhance building times by up to
30%.⟨/conflict⟩
(4) Seismic testing aligns efficiency with safety in contemporary civil engineering practices.
(5)⟨conflict⟩Speed of construction is a dominant benefit of testing building designs un-
der earthquake simulation conditions.⟨/conflict⟩
(6) Optimization of construction speed guarantees resilience and rapid realization of new build-
ings through continued innovation and testing.
...
Model AnswerBuildings will be built faster.
C CASESTUDY
In this section, we present a case study to further illustrate how our proposed framework CLEAR
enforces contextual faithfulness under knowledge conflicts. We conduct the analysis on the Faithe-
val dataset using theLLaMA-3.1-8B-Instructmodel, and the results are shown in Table 6.
CLEAR first decomposes the retrieved context into fine-grained knowledge, followed by filtering
and conflict detection. As indicated in the table, the context explicitly states that construction speed
is the dominant benefit of seismic testing, whereas the model’s prior knowledge typically associates
seismic testing with structural safety. Through our conflict detection probe, CLEAR successfully
identifies such conflicts and, with the aid of CA-SFT, reinforces the model’s attention to the con-
flicting knowledge (3) and (5). As a result, CLEAR generates the correct answer,“Buildings will
be built faster, ”which faithfully reflects the contextual evidence rather than relying on the model’s
internal knowledge. This case study highlights the effectiveness of our framework in ensuring con-
textual faithfulness in scenarios involving knowledge conflicts.
D LIMITATIONS
While CLEAR demonstrates strong improvements in textual RAG scenarios, its applicability to mul-
timodal RAG systems remains limited. The current framework is designed around sentence-level
textual decomposition and hidden-state probing, which are not directly transferable to modalities
such as images, audio, or structured data. In multimodal contexts, knowledge conflicts may mani-
fest in non-textual representations, requiring new strategies for knowledge decomposition, conflict
detection, and attention guidance. Extending CLEAR to handle heterogeneous modalities would
thus require substantial redesign of its probing mechanism and fine-tuning objectives, which we
leave as an important direction for future research.
E RELATEDWORK
In this appendix, we provide an extended review of related work on RAG, contextual faithfulness,
and knowledge conflict, complementing the concise overview in Section 5.
17

Retrieval-Augmented GenerationRAG has become a cornerstone paradigm for improving the
factual reliability and adaptability of LLMs by explicitly integrating external information during
the generation process. Early contributions such as REALM (Guu et al., 2020c) and RAG (Lewis
et al., 2020) pioneered the idea of end-to-end frameworks in which a retriever component selects
relevant passages from large-scale corpora, which are then consumed by a generator to produce re-
sponses grounded in retrieved evidence. This framework demonstrated clear advantages over purely
parametric models, particularly in tasks requiring factual precision or knowledge of recent events.
Following these foundational works, the research community has proposed a series of improve-
ments targeting both the retriever and generator components. For retrieval, dense retrieval meth-
ods (Karpukhin et al., 2020; Izacard et al., 2023) introduced learned embeddings that outperform
traditional sparse methods (e.g., BM25) in capturing semantic relevance. Subsequent refinements
incorporated multi-vector representations (Santhanam et al., 2021), passage reranking (Nogueira &
Cho, 2019), and adaptive retrieval strategies (Sun et al., 2022), where the retrieval budget is dynam-
ically allocated based on the complexity of the query or the uncertainty of the model’s predictions.
On the generator side, works have explored how to more effectively incorporate retrieved passages
during decoding. FiD (Fusion-in-Decoder) (Izacard & Grave, 2020) demonstrated the effectiveness
of late-fusion mechanisms, where a Transformer decoder attends jointly over multiple retrieved
documents. Later works extended this paradigm with hierarchical fusion (Ram et al., 2023), sparse
attention mechanisms (Shuster et al., 2022), and multi-hop retrieval pipelines (Xu et al., 2023).
Hybrid models such as RePlug (Shi et al., 2023b) and Retro (Borgeaud et al., 2022) further integrated
retrieval into pretraining or finetuning pipelines, blending parametric and non-parametric memories
to achieve both scalability and factual accuracy. More recently, adaptive frameworks (Chen et al.,
2024) proposed fine-grained controls over how retrieval signals are weighted depending on task
type, query ambiguity, or user intent.
In addition to architectural innovations, researchers have also investigated the evaluation and ef-
ficiency of RAG systems. Benchmarks such as KILT (Petroni et al., 2020) and ELI5 (Fan et al.,
2019) standardized evaluation across knowledge-intensive tasks, while efficiency-focused studies
(Guu et al., 2020b) highlighted the trade-off between retrieval accuracy, latency, and resource con-
sumption.
Contextual FaithfulnessContextual faithfulness, defined as the degree to which model outputs
remain consistent with retrieved or provided context, has emerged as a central concern in RAG
research. Without explicit mechanisms to enforce faithfulness, models may hallucinate, overgener-
alize, or generate outputs inconsistent with retrieved passages.
Prompt-based methods were among the earliest to address this challenge. Self-RAG (Asai et al.,
2023) introduced self-reflection mechanisms, where models generate justifications for retrieved con-
tent and use these to re-ground their outputs. Template-based prompting approaches (Ying et al.,
2024) designed structured query-response formats to encourage explicit grounding, though such
methods often struggle with generalization across tasks.
Decoding-based approaches tackle faithfulness by modifying the generation process itself. Con-
trastive Decoding (Yuan et al., 2024) and Context-Aware Decoding (CAD) (Shi et al., 2023a) ex-
plicitly re-weight token probabilities during beam search to favor outputs aligned with retrieved con-
text. Similarly, likelihood re-ranking techniques (Zhang et al., 2024) compare candidate responses
against retrieved evidence to penalize hallucinations. These approaches maintain the flexibility of
generation while reducing unfaithful responses.
Reinforcement learning (RL) has also been extensively applied to enhance contextual faithfulness.
CANOE (Si et al., 2025) integrates reward models that explicitly score the grounding of responses
in retrieved passages. Context-DPO (Bi et al., 2024a) extends direct preference optimization to
context-aware settings, allowing LLMs to directly learn from pairwise comparisons of faithful ver-
sus unfaithful outputs. Such RL-based frameworks emphasize end-to-end optimization, reducing
reliance on handcrafted prompts or decoding heuristics.
Beyond methodological innovations, recent surveys (Zhou et al., 2023b; Ji et al., 2023) highlight per-
sistent challenges in faithfulness evaluation. Automatic metrics such as factual consistency (Thorne
et al., 2018) or entailment-based scores (Falke et al., 2019; Guo et al., 2023) provide useful proxies
18

but often fail to capture nuanced inconsistencies or omissions. Consequently, many works advocate
for human-in-the-loop evaluation frameworks to assess contextual grounding at scale.
Knowledge ConflictKnowledge conflict arises when the retrieved evidence contradicts either the
model’s internal parametric memory or other retrieved documents, creating ambiguity in determin-
ing which knowledge to trust. This problem is particularly acute in dynamic knowledge environ-
ments, where information evolves over time or when sources exhibit bias or factual inconsistency.
A growing body of work has investigated mechanisms to detect, represent, and resolve knowledge
conflicts. Astute RAG (Wang et al., 2025a) introduces a source-aware retrieval module, leveraging
reliability estimation to assess which sources are more trustworthy in the face of contradictions.
FaithfulRAG (Zhang et al., 2025b) explicitly models fact-level conflicts, decomposing retrieved
evidence into atomic claims and guiding the generation process through a self-thinking phase that
resolves inconsistencies.
Alternative approaches focus on information-theoretic principles. Swin-VIB (Wang et al., 2025b),
for example, applies a variational information bottleneck to modulate the trade-off between fidelity
to retrieved evidence and reliance on internal knowledge, thereby accommodating conflicts in a
principled manner. Other works (Xu et al., 2024b) propose categorizing conflicts into types—such
as temporal drift, factual contradiction, or perspective variance—and tailoring resolution strategies
accordingly.
Recent research also extends conflict resolution beyond the text domain. Multimodal RAG systems
(Gao et al., 2023; Xu et al., 2024c) face analogous challenges, as retrieved visual or audio evidence
may not align with textual outputs. This motivates broader frameworks for consistency checking
across modalities. Furthermore, evaluation efforts (Xu et al., 2024b) emphasize the need for stan-
dardized benchmarks that explicitly include conflict scenarios, enabling more systematic analysis of
models’ conflict-handling behaviors.
In summary, while significant progress has been made, knowledge conflict remains an open problem.
Robust handling of contradictory information is critical not only for improving factual accuracy but
also for building user trust in RAG-based systems deployed in real-world applications.
F THEUSE OFLARGELANGUAGEMODELS
In preparing this paper, we made limited use of Large Language Models (LLMs). Specifically,
LLMs were employed for two purposes: (i) to aid in polishing the writing by improving gram-
mar, readability, and clarity without altering the scientific content, and (ii) to assist in retrieval and
discovery tasks, such as identifying and organizing related work. No LLMs were used for generat-
ing novel research ideas, designing experiments, or analyzing results. All conceptual and technical
contributions presented in this paper are the sole work of the authors.
19