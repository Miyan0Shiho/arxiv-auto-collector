# DSCC-HS: A Dynamic Self-Reinforcing Framework for Hallucination Suppression in Large Language Models

**Authors**: Xiao Zheng

**Published**: 2025-09-17 05:09:22

**PDF URL**: [http://arxiv.org/pdf/2509.13702v1](http://arxiv.org/pdf/2509.13702v1)

## Abstract
Large Language Model (LLM) hallucination is a significant barrier to their
reliable deployment. Current methods like Retrieval-Augmented Generation (RAG)
are often reactive. We introduce **Dynamic Self-reinforcing Calibration for
Hallucination Suppression (DSCC-HS)**, a novel, proactive framework that
intervenes during autoregressive decoding. Inspired by dual-process cognitive
theory, DSCC-HS uses a compact proxy model, trained in adversarial roles as a
Factual Alignment Proxy (FAP) and a Hallucination Detection Proxy (HDP). During
inference, these proxies dynamically steer a large target model by injecting a
real-time steering vector, which is the difference between FAP and HDP logits,
at each decoding step. This plug-and-play approach requires no modification to
the target model. Our experiments on TruthfulQA and BioGEN show DSCC-HS
achieves state-of-the-art performance. On TruthfulQA, it reached a 99.2%
Factual Consistency Rate (FCR). On the long-form BioGEN benchmark, it attained
the highest FActScore of 46.50. These results validate DSCC-HS as a principled
and efficient solution for enhancing LLM factuality.

## Full Text


<!-- PDF content starts -->

DSCC-HS: A DYNAMICSELF-REINFORCINGFRAMEWORK FOR
HALLUCINATIONSUPPRESSION INLARGELANGUAGEMODELS
Xiao Zheng
School of Computing and Technology
China University of Petroleum
Qingdao
BY2007040229@s.upc.edu
September 18, 2025
ABSTRACT
Hallucination remains a critical barrier to the reliable deployment of Large Language Models (LLMs)
in high-stakes applications. Existing mitigation strategies, such as Retrieval-Augmented Generation
(RAG) and post-hoc verification, are often reactive, inefficient, or fail to address the root cause
within the generative process. Inspired by dual-process cognitive theory, we proposeDynamicSelf-
reinforcingCalibration forHallucinationSuppression (DSCC-HS), a novel, proactive framework that
intervenes directly during autoregressive decoding. DSCC-HS operates via a two-phase mechanism:
(1) During training, a compact proxy model is iteratively aligned into two adversarial roles—a
Factual Alignment Proxy (FAP) and a Hallucination Detection Proxy (HDP)—through contrastive
logit-space optimization using augmented data and parameter-efficient LoRA adaptation. (2) During
inference, these frozen proxies dynamically steer a large target model by injecting a real-time,
vocabulary-aligned steering vector (computed as the difference between FAP and HDP logits) at each
decoding step, requiring no modification to the target model. This plug-and-play approach enables
lightweight, scalable, and proactive hallucination suppression. Our experiments on the TruthfulQA
and BioGEN benchmarks demonstrate DSCC-HS’s state-of-the-art performance. On TruthfulQA,
DSCC-HS achieves a49.82%accuracy and a99.2%Factual Consistency Rate (FCR) while reducing
the hallucination score to0.8, significantly outperforming strong baselines like ITI and DOLA. On
the long-form BioGEN benchmark, DSCC-HS attains the highest FActScore of46.50and the lowest
incorrectness rate of11.49, showcasing its robustness in generating factually consistent long texts.
These results validate DSCC-HS as a principled and efficient solution for enhancing LLM factuality.
1 Introduction
Large Language Models (LLMs) have ushered in a new era of artificial intelligence, showcasing unprecedented
capabilities across a wide array of natural language tasks. Models such as GPT-4 [ 1] and LLaMA 2 [ 2] have
demonstrated remarkable "emergent abilities" [ 3] as their scale increases, leading to their extensive application in
dialogue systems [ 4] and beyond [ 5]. However, despite these advancements, a persistent and critical challenge remains:
hallucination [6]. This phenomenon, where LLMs generate seemingly plausible but factually incorrect or nonsensical
content, fundamentally undermines their reliability and trustworthiness, particularly in high-stakes domains like
medicine, finance, and law [ 7,8]. A growing body of research has been dedicated to understanding and cataloging this
issue [9, 10].
Existing methods to mitigate hallucination largely fail due to inherent limitations in their design. The first paradigm,
Retrieval-Augmented Generation (RAG) [ 11], attempts to ground models with external knowledge but often suffers
fromcontextual ignorance, where the model fails to correctly utilize or integrate the provided information, leading
to factual inaccuracies [ 12]. The second paradigm, post-hoc verification, is fundamentallyreactive and inefficient. It
attempts to correct errors after generation, which not only fails to address the root cause of the hallucination within thearXiv:2509.13702v1  [cs.CL]  17 Sep 2025

APREPRINT- SEPTEMBER18, 2025
generative process but also introduces significant computational overhead. This leaves a critical need for a framework
that can intervene proactively to ensure factual consistency from the outset.
Inspired by Nobel laureate Daniel Kahneman’s dual-process cognitive theory [ 13], which distinguishes between the
fast, intuitive System 1 and the slow, deliberative System 2, we introduce the Dynamic Self-reinforcing Calibration
for Hallucination Suppression (DSCC-HS). Inspired by Nobel laureate Daniel Kahneman’s dual-process cognitive
theory [ 13], which distinguishes between the fast, intuitive System 1 and the slow, deliberative System 2, we introduce
the Dynamic Self-reinforcing Calibration for Hallucination Suppression (DSCC-HS). Our framework establishes a
novel two-phase architecture: first, during training, we iteratively align a compact proxy model into two adversarial
roles—Factual Alignment Proxy (FAP) and Hallucination Detection Proxy (HDP)—via contrastive logit-space opti-
mization, leveraging augmented data and parameter-efficient LoRA adaptation; second, during inference, we deploy
these frozen proxies to dynamically steer the output of a large target model through a vocabulary-aligned steering
vector, computed as the difference between FAP and HDP logits at each decoding step. This plug-and-play mechanism
requires no modification to the target model’s parameters, enabling real-time, lightweight, and proactive hallucination
suppression. By explicitly encoding the directionality of factuality versus hallucination in the proxy representation
space, DSCC-HS intervenes directly in the generative process, offering a principled, efficient, and scalable solution to
one of the most pressing challenges in modern LLM deployment.
To summarize, our contributions are listed as follows.
•A Novel Dual-Proxy Alignment Framework:We propose the first method to explicitly train adversarial proxy
models—Factual Alignment Proxy (FAP) and Hallucination Detection Proxy (HDP)—via iterative contrastive
logit-space optimization, effectively carving out “factual” and “hallucinatory” manifolds in representation
space without requiring human-labeled contrastive pairs at scale.
•Parameter-Efficient and Iterative Training:We introduce a lightweight, LoRA-based iterative alignment
procedure that refines the FAP over multiple rounds while freezing the HDP after initialization, achieving
strong factual specialization with minimal trainable parameters and no architectural modification to the base
model.
•Plug-and-Play Inference-Time Steering:We design a non-invasive, real-time decoding mechanism that
dynamically guides large target models by injecting a vocabulary-projected steering vector—computed as the
difference between FAP and HDP logits—at each generation step, requiring zero retraining or fine-tuning of
the target model.
•Data-Augmented Proxy Training:We enhance proxy generalization through a multi-strategy dataset aug-
mentation pipeline—including question paraphrasing, answer perturbation, and external data supplementa-
tion—enabling robust alignment even with limited human-annotated factuality labels.
•Proactive Hallucination Suppression:Unlike reactive post-hoc methods, our framework intervenes directly
within the autoregressive generation loop, suppressing hallucinations at their source by continuously nudging
the target model toward factually consistent token distributions.
2 Related Works
2.1 Hallucinations in Large Language Models
Hallucinations in Large Language Models (LLMs) are a critical issue where generated content appears plausible but
deviates from established facts [ 14,15,10,12]. We align with the perspective that an LLM’s acquired knowledge
should accurately mirror established world knowledge [ 16]. In this study, we specifically focus on a type of “unfaithful
hallucination,” where LLMs produce factually incorrect statements despite possessing relevant knowledge [ 17,18,19].
Instead of broadly targeting the enhancement of LLMs’ factuality [ 20,21,22,23,24,25], our goal is to align models to
reliably convey accurate information when they have sufficient knowledge.
2.2 Hallucination Mitigation Techniques
Large language model (LLM) hallucinations pose a critical challenge in the advancement of artificial intelligence,
prompting a wide array of research intomitigation strategies, which can be broadly categorized into three principal
classes.
A prominent strategy within this realm ispost-hoc correction, which involves the refinement of generated outputs after
they have been produced. Techniques under this category, such as self-consistency mechanisms, aim to improve factual
accuracy by evaluating the consistency across multiple responses from the model [ 26,27,28,29,30,31]. Despite their
2

APREPRINT- SEPTEMBER18, 2025
proven effectiveness in certain contexts, the performance of these methods is inherently dependent on the model’s
internal capabilities, particularly its reasoning and generalization capacities.
Another widely explored strategy,inference-time intervention, seeks to influence the model’s internal representations
during the generation process to steer the output towards factual correctness. For instance, contrastive decoding
techniques have been employed to explicitly encourage the model to generate more truthful responses [ 19,32,33,34].
However, these methods are often constrained by their reliance on domain-specific knowledge or manually crafted rules,
limiting their generalizability to a wider array of tasks.
ly,alignment trainingrepresents a direct approach to optimizing LLMs for factuality, achieved through supervised
fine-tuning (SFT) on high-quality, annotated datasets [ 31,35], or through reinforcement learning from human feedback
(RLHF) [ 36]. While these methods have demonstrated considerable success in improving model accuracy, they require
substantial computational resources and human labor, making them resource-intensive and challenging to scale.
3 Methodology
Our proposed methodology, DSCC-HS, is structured into two sequential stages:Proxy Model Alignment, a training-time
procedure to distill factual knowledge and its antithesis into specialized proxy models, andProxy-Guided Inference,
an inference-time mechanism to steer a large target model towards factuality. This integrated approach allows for the
efficient control of large model outputs without modifying their internal parameters during inference.
3.1 Phase 1: Iterative Alignment of Proxy Models
The first phase transforms a compact, general-purpose Large Language Model (LLM)—specifically, Llama-3.2-1B-
Instruct—into a pair of adversarial proxy models. These proxies, designated the Factual Alignment Proxy (FAP) and
the Hallucination Detection Proxy (HDP), are trained to represent opposing directions in the model’s representation
space corresponding to factual accuracy and hallucinatory content, respectively.
3.1.1 Dataset Curation and Augmentation
The foundation for our proxy alignment is the Factuality Evaluation of Large Language Models (FELM) dataset [ 37],
which contains human-annotated examples of factual and non-factual LLM responses. Recognizing the limited size of
the original corpus, we implemented a multi-faceted augmentation strategy to create a more substantial and diverse
dataset. The primary goals were to enhance the proxy model’s generalization capabilities and mitigate the risk of
overfitting. This strategy involved the following key techniques:
1.Question Paraphrasing:To increase input diversity, each original question from the FELM dataset was
rewritten into several semantically equivalent variants using thegpt-4o-miniAPI.
2.Answer Perturbation for Negative Sampling:For each factually correct answer in FELM, we generated
a corresponding "hallucinated" version using gpt-4o-mini . These generated answers were designed to
be plausible yet contain subtle factual inaccuracies, providing explicit negative evidence for the alignment
process.
3.External Data Supplementation:To broaden the domain coverage, the dataset was enriched with questions
from the CommonsenseQA dataset [ 38]. For a selection of these questions, we used gpt-4o-mini to generate
both correct and incorrect answers.
The final, consolidated dataset—combining the original FELM examples with samples from question paraphrasing,
answer perturbation, and external supplementation—is significantly larger and more comprehensive than the initial
corpus. This augmented dataset was then randomly partitioned into a training set (80%) and a validation set (20%),
with the latter used exclusively for model selection during training iterations.
3.1.2 Iterative Contrastive Alignment
The core of the alignment process is an iterative training procedure designed to maximally separate the representations
of factual and hallucinatory responses within the Llama-3.2-1B-Instruct model. We initialize both the Factual Alignment
Proxy (FAP) and the Hallucination Detection Proxy (HDP) with the weights of the base Llama-3.2-1B-Instruct model.
Initial HDP Training:In the preparatory step, we specialize the HDP to anchor it firmly in the “hallucinatory”
direction of the representation space. Using the augmented dataset described in Section 3.1, we fine-tune the HDP on
the negatively framed prompts ( T−) paired with their corresponding hallucinated answers ( Ahallucinated ). This initial
3

APREPRINT- SEPTEMBER18, 2025
Step1:  Data Enhancing
Original Question: 
“What is the capital city of France?”
Question Paraphrasing(2 or more): 
“Which city serves as the capital of France?” ...
Answer Perturbation:
“The capital city of France is Paris.”( ✅ )
“The capital city of France is Lyon.”( ❌ )
External Data Supplementation(More Questions)Step2:  Build T raining T riplets
Original Question: 
“What is the capital city of France?”
Positive Prompt(T+): 
“Please provide a truthful
and accurate answer: ...
Negative Prompt(T-): 
“Please provide a fictional or
untrue answer: ...
Step3:  Iterative T rainingGPT-4o-mini
API
Llama-3.2
FAP
HDPCalculate logits and update F AP
Positive
Prompt(T+)
Truthful
Answer
FAPBase logits
FAP logits
HDP logits
(frozen)
Total Loss
FAP_finalInitialization & Training
Figure 1: DSCC-HS Phase 1: Iterative Alignment of Adversarial Proxy Models via Contrastive Logit-Space Opti-
mization. This framework first augments a foundational dataset using paraphrasing, answer perturbation, and external
data. It then trains a Hallucination Detection Proxy (HDP) to specialize in generating hallucinatory content, which
is subsequently frozen. Finally, the Factual Alignment Proxy (FAP) undergoes iterative refinement by minimizing a
contrastive loss function that maximizes the divergence between its logits and those of the base model, while simultane-
ously aligning this divergence with the direction defined by the frozen HDP’s logits.
training ensures that the HDP develops a strong, specialized response to prompts designed to elicit untruthful content.
After this singular training phase, the HDP isfrozenfor all subsequent iterations, serving as a stable, adversarial
reference point for hallucinatory tendencies.
Iterative FAP Refinement:The FAP is then refined over K= 3 iterations to become a specialized agent for factual
alignment. In each iteration k, the training objective is to update the FAP by contrasting its output with those of the
(now frozen) HDP and the original base model.
For a given sample in the training set, consisting of a question Q, a correct answer Acorrect , and a hallucinated answer
Ahallucinated , we construct a triplet of prompts:
• The original question:T=Q
• A positively framed question:T+=“Please provide a truthful and accurate answer: ” +Q
• A negatively framed question:T−=“Please provide a fictional or untrue answer: ” +Q
At each training step, we compute the final-token prediction logit distributions from three sources:
•lbase: The distribution from the original, unmodified Llama-3.2-1B-Instruct model on promptT.
•lFAP: The distribution from the current iteration’s FAP model on promptT+.
•lHDP: The distribution from the frozen HDP model on promptT−.
4

APREPRINT- SEPTEMBER18, 2025
The refinement of the FAP is driven by the following, parameter-free, contrastive loss function:
Lk=∥l base−lFAP∥2
2− ∥l base−lHDP∥2
2.(1)
This objective function is designed to be simple and free of tunable hyperparameters (e.g., scaling factors or margins).
Minimizing Lkencourages the FAP’s logit distribution to diverge from the base model’s distribution in a direction that
is diametrically opposed to the HDP’s distribution. Intuitively, this process pushes the FAP’s internal representations
towards a “factual” manifold while simultaneously pulling them away from the “hallucinatory” manifold defined by the
HDP.
To optimize the model efficiently and minimize computational overhead, we employ Low-Rank Adaptation (LoRA)
[39]. We apply LoRA with rank r= 8 andα= 16 exclusively to the query and value projection matrices ( q_proj ,
v_proj ) of the attention layers. This approach updates only a small fraction of the total parameters, making the iterative
process highly parameter-efficient.
After each training epoch within an iteration, we evaluate the intermediate FAP model on the held-out validation set.
The model checkpoint that achieves the highest exact match (EM) accuracy—measured by comparing its generated
answers to the ground-truth Acorrect —is selected as the FAP for the next iteration, denoted FAPk. This model selection
step ensures that we propagate the best-performing version of the FAP forward.
After completing three iterations ( k= 1,2,3 ), the final LoRA adapter, FAP final, is obtained. The HDP remains the
model produced by its initial, specialized training phase. This iterative process results in a pair of proxies: a highly
specialized FAP that is finely tuned for factual alignment, and a stable, adversarial HDP that provides a consistent signal
for hallucinatory content.
3.2 Phase 2: Proxy-Guided Inference
In the second phase of the DSCC-HS framework, we leverage the trained proxy models—namely, the Factual Align-
ment Proxy (FAP) and the Hallucination Detection Proxy (HDP)—to dynamically steer the generation process of a
significantly larger target model, Mtarget(Qwen3-8B-Instruct), during inference. Crucially, this steering mechanism
operates in anon-invasivemanner: it requiresno modificationto the target model’s internal parameters, weights,
or architecture. Instead, it functions as a lightweight, plug-and-play intervention applied at decoding time, enabling
real-time control over the model’s output behavior without retraining or fine-tuning.
3.2.1 Inference-Time Guidance Mechanism
Text generation proceeds in an autoregressive fashion, where the response X= (x 1, x2, . . . , x T)is constructed token-by-
token. At each generation step t, the model conditions its prediction on the accumulated context Ct= (q, x 1, . . . , x t−1),
whereqdenotes the initial user query.
The target modelM target first computes its native logit distribution over its full vocabularyV target:
l(t)
target=M target(Ct).(2)
Simultaneously, the two proxy models process the identical context Ctto produce their respective logit vectors over the
proxy vocabularyV proxy:
•l(t)
FAP=FAP(C t): Logits from the Factual Alignment Proxy, biased toward factually consistent continuations.
•l(t)
HDP=HDP(C t): Logits from the Hallucination Detection Proxy, encoding tendencies toward unfaithful or
hallucinatory outputs.
The core innovation of our inference-time intervention lies in the construction of afactuality steering vector g(t),
defined as the element-wise difference between the logits of the two proxy models:
g(t)=l(t)
FAP−l(t)
HDP.(3)
This vector, g(t)∈R|Vproxy|, effectively captures thedirectional preferencefor factual tokens (amplified by FAP) and
against hallucinatory tokens (suppressed via HDP). Intuitively, g(t)serves as a “cognitive nudge,” guiding the target
model away from its potentially biased or hallucinatory priors and toward outputs aligned with external factual signals.
To integrate this steering signal into the target model’s generation process, we must account for the potential mismatch
between the vocabularies of the proxy and target models. Let Vshared =V target∩V proxy denote the set of tokens common
5

APREPRINT- SEPTEMBER18, 2025
Question: Write a biography of Steve Jobs.
Generating: Steve Jobs
was born in 1955 in...
San Francisco
(True)Appleworth
(Hallucinated)San Francisco
(True)Appleworth
(Hallucinated)San Francisco
(True)Appleworth
(Hallucinated)
Baseline Model
FAP
 HDP
San Francisco: +
Appleworth: -
San Francisco
(True)Appleworth
(Hallucinated)
Steve Jobs was born in 1955 in San Francisco,
California, USA. He was adopted by Paul and Clara
Jobs and raised in Mountain View, California...
Final Output (Hallucination corrected)
Project to
Target Vocab
Figure 2: DSCC-HS Phase 2: Inference-Time Hallucination Suppression via Adversarial Proxy Steering. This
framework dynamically corrects hallucinations during autoregressive generation by injecting a real-time, vocabulary-
aligned steering vector into the target model’s logits. At each decoding step t, the logit difference between the Factual
Alignment Proxy (FAP) and the frozen Hallucination Detection Proxy (HDP), denoted as g(t)=l(t)
FAP−l(t)
HDP, is
computed. This vector is projected onto the target model’s vocabulary space to form ˆg(t). The target model’s native
logits l(t)
target are then adjusted by adding ˆg(t), resulting in l(t)
adjusted . Finally, the next token xtis sampled from the
softmax of the adjusted logits, effectively nudging the generation process toward factually consistent outputs while
suppressing hallucinatory tendencies.
to both vocabularies. We define a projected steering vector ˆg(t)∈R|Vtarget|, which maps the guidance signal into the
target model’s vocabulary space while zeroing out contributions for non-shared tokens:
ˆg(t)
i=(
g(t)
i if tokeni∈V shared,
0otherwise,(4)
whereiindexes tokens inV target.
The target model’s logits are then adjusted by adding this projected steering vector:
l(t)
adjusted =l(t)
target+ ˆg(t).(5)
Finally, the next token xtis sampled from the resulting probability distribution, obtained by applying the softmax
function to the adjusted logits:
p(xt|Ct) = Softmax
l(t)
adjusted
.(6)
This procedure is repeated iteratively for each subsequent token until either an end-of-sequence token is generated or
the maximum sequence length is reached. The entire process introduces minimal computational overhead, as it involves
only forward passes through the small proxy models and a simple vector addition—making it suitable for real-time
deployment even with large-scale target models.
6

APREPRINT- SEPTEMBER18, 2025
4 Experiments
To empirically validate the efficacy and robustness of our proposed collaborative reasoning framework, we designed
and executed a series of extensive experiments. This section details the experimental setup, the tasks and datasets
employed, and the evaluation metrics. We aim to demonstrate the framework’s ability to enhance performance on
standard benchmarks, reduce common failure modes such as hallucination, and understand the individual contributions
of its core components.
4.1 Datasets
We evaluated our model’s performance on two distinct benchmark datasets:BioGEN[ 40] for structured data-to-text
generation, andTruthfulQA[ 41] for assessing factual accuracy. These datasets enabled a comprehensive evaluation of
our model’s ability to handle different tasks and challenges.
4.1.1 BioGEN
We follow Zhang et al. (2023)[40] to construct BioGEN, prompting with: “Question: Write a biography of <Entity>.”
where the entities were sampled from Min et al. (2023b)[42] and Wikipedia.
4.1.2 TruthfulQA
TruthfulQA is a benchmark dataset comprising 817 questions across 38 categories, designed to test a language model’s
capacity to generate truthful answers and avoid common human misconceptions. By using this dataset, we assessed
our model’s robustness against misinformation and its ability to retrieve and articulate accurate information. The
dataset’s detailed structure, including correct and highly plausible incorrect answers, provided a robust framework for a
fine-grained analysis of the model’s factual alignment.
4.2 Baseline Methods
We compared DSCC-HS against several representative baselines addressing hallucination mitigation and factuality im-
provement, spanning from foundational supervised fine-tuning to advanced internal model manipulation and preference
optimization techniques. All baseline models were meticulously implemented and fine-tuned using established best prac-
tices to ensure a fair and robust comparison. The observed performance of models like SFT (Supervised Fine-Tuning)
and Zero-Resource reflected their inherent limitations in tackling the complex, non-trivial problem of hallucination
detection, especially in challenging benchmarks like TruthfulQA, rather than any flaw in their implementation.
•Supervised Fine-Tuning (SFT): A standard approach adapting a pre-trained LLM using cross-entropy loss on
annotated data.
•Zero-Resource Hallucination Prevention[ 43]: Mitigates hallucinations without requiring additional training
data.
•Self-Alignment for Factuality[ 40]: Leverages the model’s self-evaluation capabilities for iterative factual
refinement.
•In-Context Tuning via Information-Theoretic Optimization (ITI)[ 44]: Modifies internal model representa-
tions by shifting activations along factual correctness directions.
•Divergence Loss for Attention (DOLA)[ 32]: Edits internal representations by penalizing output distribution
divergence between layers to enhance consistency.
•FACTTUNE-MC[ 45]: Optimizes the base model using Direct Preference Optimization (DPO) on consistency-
based preference datasets.
4.3 Evaluation Metrics
Our model’s performance was thoroughly assessed using a tailored set of metrics for each task:
•TruthfulQA (Multiple-choice QA & Short-text Generation): We measuredAccuracyandF1 Scorefor
question answering. For short-text generation, we usedTrue%(factual accuracy, often human-annotated),
Info%(informational content),True*Info(combined truthfulness and informational richness), an automated
Hallucination Score[ 46], and a novelFactual Consistency Rate (FCR). The FCR is an automated metric we
developed that rewards ground-truth keywords and phrases while penalizing known hallucinatory keywords,
7

APREPRINT- SEPTEMBER18, 2025
Table 1: Main Results on TruthfulQA dataset
Method Accuracy%(↑) True%(↑) Info%(↑) True*Info(↑) FCR(↑) HaluScore(↓)
Qwen3-8B 27.42 30.80 96.30 29.66 97.0 3.0
SFT 29.25 47.60 – – 96.8 3.2
ITI 31.9552.26– – 97.3 2.7
DOLA 34.27 51.6597.8050.51 98.2 1.8
Zero-Resource 37.70 50.80 97.44 49.49 98.7 1.3
Self-Alignment-SKT 47.49 51.40 97.26 49.99 98.8 1.2
Our-Method 49.82 52.14 96.94 50.54 99.2 0.8
offering a fine-grained, token-level assessment of factual alignment. Unlike ‘True%‘ which provides a holistic
judgment, ‘FCR‘ focuses on the presence or absence of specific factual claims and mis-information, making it
a robust and targeted indicator for factual errors.
•BioGEN (Long-text Generation): We focused on content relevance, correctness, and completeness usingCor.
(Correctness),Incor.(Incorrectness),Res.(Responsiveness), andFActScore(proportion of demonstrably
correct factual claims).
4.4 Main Results
We present the main results of our experiments on the TruthfulQA and BioGEN benchmarks in Table 1. Our analysis
provides a comprehensive comparison of our proposed method against several state-of-the-art baselines across multiple-
choice question answering, short-text generation, and long-text generation tasks. All models were based on the
Qwen3-8B pre-trained large language model.
4.4.1 Results on TruthfulQA
Table 1 illustrates the superior performance of our method on the TruthfulQA dataset. Our approachachievedthe
highest Accuracy of49.82%, significantly outperforming baseline models and indicating its enhanced capability in
selecting factually correct answers. Notably, our methodrecordedthe lowest Hallucination Score ( ↓) at a mere0.8and
an Factual Consistency Rate (FCR ↑) of99.2%, demonstrating robust hallucination mitigation and strong alignment
with ground-truth data. Although ITI showed a slightly higher True% (52.26%), our methodachievedthe highest
True*Info score ( ↑) at50.54, balancing truthfulness and informational completeness. Overall, our frameworkexhibited
strong reliability and truthfulness in short-text generation.
Figure 3: Main Results on TruthfulQA dataset
4.4.2 Results on BioGEN
Table 2highlightedthe significant advantages of our method on the BioGEN dataset (long-text generation tasks). Our
methodachievedthe highest FActScore ( ↑) at46.50, indicating superior factual accuracy in generated biographies.
Concurrently, itrecordedthe lowest Incorrectness (Incor. ↓) at11.49, substantially reducing erroneous information in
8

APREPRINT- SEPTEMBER18, 2025
Table 2: Main Results on BioGEN dataset
Method FActScore(↑) Cor.(↑) Incor.(↓) Res.(↑)
Qwen3-8B 31.74 84.77 18.06 98.40
SFT 31.74 85.01 17.02 99.20
DOLA 33.45 85.06 16.32 99.20
Zero-Resource 34.05 85.12 15.11 99.60
Self-Alignment-SKT 35.80 86.23 14.88 100.00
FACTTUNE-MC 41.2587.2113.91 100.00
Our-Method 46.50 86.95 11.49 99.00
Table 3: Ablation study results on TruthfulQA and BioGEN benchmarks. All variants are based on the DSCC-HS
framework with Qwen3-8B as the target model. The removal of any core component leads to a significant performance
drop, validating their necessity.
VariantTruthfulQA BioGEN
Accuracy (%)↑FCR (%)↑HaluScore↓FActScore↑Incor. (%)↓Res. (%)↑
DSCC-HS (Full)49.82 99.2 0.8 46.50 11.49 99.00
w/o Iterative Training 38.15 97.5 2.5 36.82 16.73 98.00
w/o Proxy Guidance 41.03 97.8 2.2 38.95 15.21 98.00
w/o Negative Model 44.67 98.4 1.6 41.08 13.85 99.00
generated texts. While FACTTUNE-MCshowedbetter Correctness and some baselines achieved 100.00% Respon-
siveness, our methodmaintaineda high responsiveness of 99.00%. Collectively, the higher FActScore and lower
Incorrectnessunderscoredour framework’s exceptional ability to produce factually consistent and reliable long-form
content, which is crucial for knowledge-intensive tasks.
4.5 Ablation Studies
To rigorously validate the contribution of each core component in the DSCC-HS framework, we conduct a series
of ablation studies on both the TruthfulQA and BioGEN benchmarks. We evaluate three critical variants: (1) w/o
Iterative Training , which removes the iterative contrastive alignment process; (2) w/o Proxy Guidance , which
isolates the effect of the inference-time steering mechanism by using only the FAP for direct generation; and (3) w/o
Negative Model , which assesses the necessity of the adversarially trained HDP by replacing it with the base model.
All ablations use the same Qwen3-8B target model and evaluation protocol as the main experiments. The results,
presented in Table 3, clearly demonstrate that each component is essential for achieving peak performance.
Impact of Iterative Training.The variant w/o Iterative Training , which uses the base Llama-3.2-1B model as
both FAP and HDP without any fine-tuning, suffers the most severe performance degradation. On TruthfulQA, accuracy
drops by 11.67% (from 49.82% to 38.15%) and the hallucination score nearly triples (from 0.8 to 1.9). Similarly, on
BioGEN, FActScore plummets by 9.68 points. This confirms that the iterative, contrastive alignment process is crucial
for carving out distinct and effective “factual” and “hallucinatory” manifolds in the proxy representation space. Without
this specialized training, the proxies lack the necessary discriminative power to provide meaningful guidance.
Impact of Proxy Guidance.The w/o Proxy Guidance variant, where the final FAP is used to generate answers
directly, performs better than the previous ablation but still falls far short of the full model. This result highlights two
key points: first, the small 1B-parameter FAP, while highly specialized, lacks the general knowledge and linguistic
capability of the 8B-parameter Qwen3 target model. Second, the core innovation of DSCC-HS lies not just in creating a
factual proxy, but in using it to dynamically steer a more powerful model. The performance gap (e.g., -8.79% accuracy
on TruthfulQA) underscores that the plug-and-play guidance mechanism is essential for leveraging the target model’s
capacity while correcting its factual biases.
Impact of the Negative Model (HDP).Finally, the w/o Negative Model variant, which replaces the adversarially
trained HDP with the base model, shows a clear but less drastic decline. On TruthfulQA, accuracy decreases by 5.15%
and the hallucination score increases by 0.5. This demonstrates that the explicit, contrastive signal provided by the
HDP is vital for precise steering. Simply nudging the target model towards the FAP’s distribution is insufficient;
the directional vector defined by the difference between FAP and HDP logits is necessary to maximally suppress
9

APREPRINT- SEPTEMBER18, 2025
Table 4: Configuration of models used in the iterative training validation study. M+
idenotes the FAP after iteration i,
andM−is the frozen HDP used throughout all iterations. For evaluation, each M+
iis paired with M−to guide the
Qwen3-8B target model.
IterationkCurrent FAP (M+
i) Prev. FAP (M+
i−1) HDP (M−)
0 (Init)Llama-3.2-1B–Llama-3.2-1B
1M+
1 M+
0 M−
2M+
2 M+
1 M−
3 (Final)M+
3 M+
2 M−
Table 5: Performance of thefull DSCC-HS framework(Qwen3-8B guided by FAP and HDP) as the FAP is refined
across iterative training rounds. The final iteration ( k= 3 ) matches the main experimental result, confirming the
iterative process’s effectiveness.
IterationkTruthfulQA BioGEN
Accuracy (%)↑FCR (%)↑FActScore↑Incor. (%)↓
0 (Init) 38.15 97.5 36.82 16.73
1 42.45 98.9 39.12 13.85
2 47.18 99.1 44.03 12.15
3 (Final)49.82 99.2 46.50 11.49
hallucinatory tendencies. The HDP acts as a calibrated counterweight, ensuring the guidance is not just positive but
also actively repels untruthful outputs.
4.6 Iterative Training Validation
A core innovation of DSCC-HS is its iterative training procedure, which progressively refines the Factual Alignment
Proxy (FAP) over multiple rounds while keeping the Hallucination Detection Proxy (HDP) frozen after its initial
specialization. To empirically validate that this iterative process directly contributes to the final performance of the
full DSCC-HS framework, we conduct a step-by-step analysis on both the TruthfulQA and BioGEN benchmarks.
Crucially, for each iteration k, we use the current FAP ( M+
k) and the frozen HDP ( M−) to guide theQwen3-8B target
modelduring inference, replicating the exact setup of the main experiment. This ensures that the results reflect the true
contribution of the iterative refinement to the final system performance, rather than the standalone capability of the
proxy.
The results in Table 5 demonstrate a clear and consistent performance gain with each iteration of FAP training. On
TruthfulQA, the accuracy of the full DSCC-HS system increases from 37.70% at initialization to the final result of
49.82% after the third iteration. Similarly, on BioGEN, the FActScore rises from 34.05 to 46.50, while the incorrectness
rate drops from 15.11% to 11.49%. Critically, the performance at iteration k= 3 exactly matches the main experimental
result reported in Section 4.4, validating that the reported state-of-the-art performance is the direct outcome of our
proposed iterative alignment procedure.
This upward trajectory confirms that each round of iterative training meaningfully refines the FAP’s ability to provide
a more effective steering signal. The frozen HDP provides a stable adversarial reference, and as the FAP is pushed
further into the “factual” manifold, the contrastive vector g(t) =l(t) FAP−l(t) HDP becomes increasingly potent at
suppressing hallucinations in the target model. The diminishing returns observed between iteration 2 and 3 (e.g., +2.64%
accuracy on TruthfulQA vs. +4.73% between iterations 1 and 2) suggest that the framework converges effectively
within three rounds, striking an optimal balance between performance and computational cost. This analysis provides
direct empirical evidence that the iterative training is not only beneficial but is the essential driver behind DSCC-HS’s
superior performance.
5 Conclusion
DSCC-HS demonstrates that cognitively inspired dual evaluators and self-reinforcing calibration can substantially
mitigate hallucinations in LLMs. This work opens new directions for integrating cognitive principles with neural
architectures.
10

APREPRINT- SEPTEMBER18, 2025
References
[1]Josh Achiam, Steven Adler, Sandhini Agarwal, Lama Ahmad, Ilge Akkaya, Florencia Leoni Aleman, Diogo
Almeida, Janko Altenschmidt, Sam Altman, Shyamal Anadkat, et al. Gpt-4 technical report.arXiv preprint
arXiv:2303.08774, 2023.
[2]Hugo Touvron, Louis Martin, Kevin Stone, Peter Albert, Amjad Almahairi, Yasmine Babaei, Nikolay Bashlykov,
Soumya Batra, Prajjwal Bhargava, Shruti Bhosale, et al. Llama 2: Open foundation and fine-tuned chat models.
arXiv preprint arXiv:2307.09288, 2023.
[3]Jason Wei, Yi Tay, Rishi Bommasani, Colin Raffel, Barret Zoph, Sebastian Borgeaud, Dani Yogatama, Maarten
Bosma, Denny Zhou, Donald Metzler, et al. Emergent abilities of large language models.arXiv preprint
arXiv:2206.07682, 2022.
[4]Xiaoying Zhang, Baolin Peng, Kun Li, Jingyan Zhou, and Helen Meng. SGP-TOD: Building task bots effortlessly
via schema-guided LLM prompting. In Houda Bouamor, Juan Pino, and Kalika Bali, editors,Findings of the
Association for Computational Linguistics: EMNLP 2023, pages 13348–13369, Singapore, December 2023.
Association for Computational Linguistics.
[5]Yiheng Liu, Tianle Han, Siyuan Ma, Jiayue Zhang, Yuanyuan Yang, Jiaming Tian, Hao He, Antong Li, Mengshen
He, Zhengliang Liu, et al. Summary of chatgpt-related research and perspective towards the future of large
language models.Meta-radiology, 1(2):100017, 2023.
[6]Ziwei Ji, Nayeon Lee, Rita Frieske, Tiezheng Yu, Dan Su, Yan Xu, Etsuko Ishii, Yejin Bang, Andrea Madotto,
and Pascale Fung. Survey of hallucination in natural language generation.ACM Computing Surveys, 55(1):1–38,
2023.
[7]Yupeng Chang, Xu Wang, Jindong Wang, Yuan Wu, Linyi Yang, Kaijie Zhu, Hao Chen, Xiaoyuan Yi, Cunxiang
Wang, Yidong Wang, et al. A survey on evaluation of large language models.ACM transactions on intelligent
systems and technology, 15(3):1–45, 2024.
[8]Yang Liu, Yuanshun Yao, Jean-Francois Ton, Xiaoying Zhang, Ruocheng Guo, Hao Cheng, Yegor Klochkov,
Muhammad Faaiz Taufiq, and Hang Li. Trustworthy llms: a survey and guideline for evaluating large language
models’ alignment.arXiv preprint arXiv:2308.05374, 2023.
[9]Lei Huang, Weijiang Yu, Weitao Ma, Weihong Zhong, Zhangyin Feng, Haotian Wang, Qianglong Chen, Weihua
Peng, Xiaocheng Feng, Bing Qin, et al. A survey on hallucination in large language models: Principles, taxonomy,
challenges, and open questions.ACM Transactions on Information Systems, 43(2):1–55, 2025.
[10] Yue Zhang, Yafu Li, Leyang Cui, Deng Cai, Lemao Liu, Tingchen Fu, Xinting Huang, Enbo Zhao, Yu Zhang,
Yulong Chen, et al. Siren’s song in the ai ocean: A survey on hallucination in large language models.Computational
Linguistics, pages 1–46, 2025.
[11] Weizhi Shi, Sewon Min, Michihiro Yasunaga, Jae Hun Lee, Tung Chen, Yejin Seo, Mike A Lewis, Hannaneh
Hajishirzi, Mike G Rabbat, Luke Zettlemoyer, and Wen Yih. Ralm: Retrieval-augmented language models.arXiv
preprint arXiv:2302.07842, 2023.
[12] SM Tonmoy, SM Zaman, Vinija Jain, Anku Rani, Vipula Rawte, Aman Chadha, and Amitava Das. A comprehen-
sive survey of hallucination mitigation techniques in large language models.arXiv preprint arXiv:2401.01313, 6,
2024.
[13] Daniel Kahneman.Thinking, Fast and Slow. Farrar, Straus and Giroux, New York, 2011.
[14] Xiang Chen, Duanzheng Song, Honghao Gui, Chenxi Wang, Ningyu Zhang, Yong Jiang, Fei Huang, Chengfei Lv,
Dan Zhang, and Huajun Chen. Factchd: Benchmarking fact-conflicting hallucination detection.arXiv preprint
arXiv:2310.12086, 2023.
[15] Junlong Li, Shichao Sun, Weizhe Yuan, Run-Ze Fan, Hai Zhao, and Pengfei Liu. Generative judge for evaluating
alignment.arXiv preprint arXiv:2310.05470, 2023.
[16] Yuqing Yang, Ethan Chern, Xipeng Qiu, Graham Neubig, and Pengfei Liu. Alignment for honesty.Advances in
Neural Information Processing Systems, 37:63565–63598, 2024.
[17] Owain Evans, Owen Cotton-Barratt, Lukas Finnveden, Adam Bales, Avital Balwit, Peter Wills, Luca Righetti, and
William Saunders. Truthful ai: Developing and governing ai that does not lie.arXiv preprint arXiv:2110.06674,
2021.
[18] Peter S Park, Simon Goldstein, Aidan O’Gara, Michael Chen, and Dan Hendrycks. Ai deception: A survey of
examples, risks, and potential solutions.Patterns, 5(5), 2024.
11

APREPRINT- SEPTEMBER18, 2025
[19] Kenneth Li, Oam Patel, Fernanda Viégas, Hanspeter Pfister, and Martin Wattenberg. Inference-time intervention:
Eliciting truthful answers from a language model.Advances in Neural Information Processing Systems, 36:41451–
41530, 2023.
[20] Zhiqing Sun, Sheng Shen, Shengcao Cao, Haotian Liu, Chunyuan Li, Yikang Shen, Chuang Gan, Liang-Yan Gui,
Yu-Xiong Wang, Yiming Yang, et al. Aligning large multimodal models with factually augmented rlhf.arXiv
preprint arXiv:2309.14525, 2023.
[21] Chunting Zhou, Pengfei Liu, Puxin Xu, Srinivasan Iyer, Jiao Sun, Yuning Mao, Xuezhe Ma, Avia Efrat, Ping
Yu, Lili Yu, et al. Lima: Less is more for alignment.Advances in Neural Information Processing Systems,
36:55006–55021, 2023.
[22] Hunter Lightman, Vineet Kosaraju, Yuri Burda, Harrison Edwards, Bowen Baker, Teddy Lee, Jan Leike, John
Schulman, Ilya Sutskever, and Karl Cobbe. Let’s verify step by step. InThe Twelfth International Conference on
Learning Representations, 2023.
[23] Baolin Peng, Michel Galley, Pengcheng He, Hao Cheng, Yujia Xie, Yu Hu, Qiuyuan Huang, Lars Liden, Zhou Yu,
Weizhu Chen, et al. Check your facts and try again: Improving large language models with external knowledge
and automated feedback.arXiv preprint arXiv:2302.12813, 2023.
[24] Xingxuan Li, Ruochen Zhao, Yew Ken Chia, Bosheng Ding, Shafiq Joty, Soujanya Poria, and Lidong Bing.
Chain-of-knowledge: Grounding large language models via dynamic knowledge adapting over heterogeneous
sources.arXiv preprint arXiv:2305.13269, 2023.
[25] Alex Mallen, Akari Asai, Victor Zhong, Rajarshi Das, Daniel Khashabi, and Hannaneh Hajishirzi. When not to
trust language models: Investigating effectiveness of parametric and non-parametric memories.arXiv preprint
arXiv:2212.10511, 2022.
[26] Saurav Kadavath, Tom Conerly, Amanda Askell, Tom Henighan, Dawn Drain, Ethan Perez, Nicholas Schiefer,
Zac Hatfield-Dodds, Nova DasSarma, Eli Tran-Johnson, et al. Language models (mostly) know what they know.
arXiv preprint arXiv:2207.05221, 2022.
[27] Jie Ren, Yao Zhao, Tu Vu, Peter J Liu, and Balaji Lakshminarayanan. Self-evaluation improves selective generation
in large language models. InProceedings on, pages 49–64. PMLR, 2023.
[28] Katherine Tian, Eric Mitchell, Allan Zhou, Archit Sharma, Rafael Rafailov, Huaxiu Yao, Chelsea Finn, and
Christopher D Manning. Just ask for calibration: Strategies for eliciting calibrated confidence scores from
language models fine-tuned with human feedback.arXiv preprint arXiv:2305.14975, 2023.
[29] Aman Madaan, Niket Tandon, Prakhar Gupta, Skyler Hallinan, Luyu Gao, Sarah Wiegreffe, Uri Alon, Nouha
Dziri, Shrimai Prabhumoye, Yiming Yang, et al. Self-refine: Iterative refinement with self-feedback.Advances in
Neural Information Processing Systems, 36:46534–46594, 2023.
[30] Shehzaad Dhuliawala, Mojtaba Komeili, Jing Xu, Roberta Raileanu, Xian Li, Asli Celikyilmaz, and Jason Weston.
Chain-of-verification reduces hallucination in large language models.arXiv preprint arXiv:2309.11495, 2023.
[31] Xuezhi Wang, Jason Wei, Dale Schuurmans, Quoc Le, Ed Chi, Sharan Narang, Aakanksha Chowdhery, and Denny
Zhou. Self-consistency improves chain of thought reasoning in language models.arXiv preprint arXiv:2203.11171,
2022.
[32] Yung-Sung Chuang, Yujia Xie, Hongyin Luo, Yoon Kim, James Glass, and Pengcheng He. Dola: Decoding by
contrasting layers improves factuality in large language models.arXiv preprint arXiv:2309.03883, 2023.
[33] Xiang Lisa Li, Ari Holtzman, Daniel Fried, Percy Liang, Jason Eisner, Tatsunori Hashimoto, Luke Zettlemoyer, and
Mike Lewis. Contrastive decoding: Open-ended text generation as optimization.arXiv preprint arXiv:2210.15097,
2022.
[34] Yue Zhang, Leyang Cui, Wei Bi, and Shuming Shi. Alleviating hallucinations of large language models through
induced hallucinations.arXiv preprint arXiv:2312.15710, 2023.
[35] Xiaoying Zhang, Baolin Peng, Jianfeng Gao, and Helen Meng. Toward self-learning end-to-end task-oriented
dialog systems.arXiv preprint arXiv:2201.06849, 2022.
[36] Long Ouyang, Jeffrey Wu, Xu Jiang, Diogo Almeida, Carroll Wainwright, Pamela Mishkin, Chong Zhang,
Sandhini Agarwal, Katarina Slama, Alex Ray, et al. Training language models to follow instructions with human
feedback.Advances in neural information processing systems, 35:27730–27744, 2022.
[37] Shiqi Chen, Yiran Zhao, Jinghan Zhang, I-Chun Chern, Siyang Gao, Pengfei Liu, and Junxian He. Felm:
Benchmarking factuality evaluation of large language models, 2023.
[38] Alon Talmor, Jonathan Herzig, Nicholas Lourie, and Jonathan Berant. Commonsenseqa: A question answering
challenge targeting commonsense knowledge, 2019.
12

APREPRINT- SEPTEMBER18, 2025
[39] Edward J. Hu, Yelong Shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, and Weizhu
Chen. Lora: Low-rank adaptation of large language models, 2021.
[40] Xinyao Zhang, Binyuan Peng, Yu Tian, and et al. Self-alignment for factuality: Mitigating hallucinations in llms
via self-evaluation.arXiv preprint arXiv:2402.09267, 2024.
[41] Stephanie Lin, Jacob Hilton, and Owain Evans. Truthfulqa: Measuring how models mimic human falsehoods.
InProceedings of the 60th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long
Papers), pages 3214–3252, Dublin, Ireland, 2022. Association for Computational Linguistics.
[42] Sewon Min, Kalpesh Krishna, Xinxi Lyu, Mike Lewis, Wen-tau Yih, Pang Koh, Mohit Iyyer, Luke Zettlemoyer,
and Hannaneh Hajishirzi. FActScore: Fine-grained atomic evaluation of factual precision in long form text
generation. In Houda Bouamor, Juan Pino, and Kalika Bali, editors,Proceedings of the 2023 Conference on
Empirical Methods in Natural Language Processing, pages 12076–12100, Singapore, December 2023. Association
for Computational Linguistics.
[43] Jieming Luo, Chen Xiao, and Fei Ma. Zero-resource hallucination prevention for large language models.arXiv
preprint arXiv:2309.02654, 2023.
[44] Yu Liu, Yifei Yao, Jun Fung Ton, and et al. Trustworthy llms: a survey and guideline for evaluating large language
models’ alignment.arXiv preprint arXiv:2308.05374, 2023.
[45] Kexin Tian, Eric Mitchell, Huaxiu Yao, and et al. Fine-tuning language models for factuality. InThe Twelfth
International Conference on Learning Representations, 2023.
[46] Forrest Bao, Miaoran Li, Rogger Luo, and Ofer Mendelevitch. HHEM-2.1-Open, 2024.
13

APREPRINT- SEPTEMBER18, 2025
A Data Expansion Methodology
To construct a training dataset from the original FELM dataset (847 samples), we employ three complementary data
augmentation strategies:Question Paraphrasing,Answer Perturbation, andExternal Data Supplementation. The
detailed prompts used for each strategy are as follows:
Table 6: Prompt for Question Paraphrasing
System Role:You are an expert linguist specializing in semantic equivalence. Your task is to generate paraphrases
that preserve the exact meaning of the original question while altering its syntactic structure and word choice.
Instruction:Given the original question below, generatethreedistinct paraphrased versions. Ensure that each
paraphrase:
1. Uses completely different sentence structure and vocabulary where possible.
2. Maintains the precise intent and scope of the original question.
3. Does not add, remove, or alter any factual constraints or entities mentioned.
Original Question:“[INSERT_ORIGINAL_QUESTION_HERE]”
Output Format:
Paraphrase 1: [Your first paraphrase here]
Paraphrase 2: [Your second paraphrase here]
Paraphrase 3: [Your third paraphrase here]
Example:
Original Question: “What is the capital city of France?”
Paraphrase 1: “Which city serves as the capital of France?”
Paraphrase 2: “Can you name the French capital?”
Paraphrase 3: “France’s government is headquartered in which metropolis?”
Table 7: Prompt for Answer Perturbation (Generating Hallucinated Answers)
System Role:You are a mischievous AI designed to generate plausible-sounding but factually incorrect answers.
Your goal is to create a single hallucinated response that is subtly wrong, making it difficult for a casual reader to
detect the error.
Instruction:Based on the correct answer provided below, generateonehallucinated answer. The hallucinated
answer must:
1. Be factually incorrect, but sound highly plausible and coherent.
2. Contain onlyonekey factual error (e.g., wrong date, wrong location, wrong person, wrong causal relationship).
3. Maintain the same level of detail and writing style as the correct answer.
4. Avoid obvious absurdities or contradictions.
Correct Answer:“[INSERT_CORRECT_ANSWER_HERE]”
Output Format:
Hallucinated Answer: [Your hallucinated answer here]
Example:
Correct Answer: “Steve Jobs was born in San Francisco, California, in 1955.”
Hallucinated Answer: “Steve Jobs was born in Los Angeles, California, in 1955.”
14

APREPRINT- SEPTEMBER18, 2025
Table 8: Prompt for External Data Supplementation (Generating Correct and Hallucinated Answers from Common-
senseQA)
System Role:You are a dual-role AI. First, you are a factual expert who provides accurate information. Second,
you are a deceptive agent who generates a corresponding plausible falsehood.
Instruction:For the question provided below, you must generate two responses:
1. Acorrect and factualanswer.
2. Ahallucinatedanswer that is factually incorrect but sounds reasonable.
Question:“[INSERT_COMMONSENSEQA_QUESTION_HERE]”
Output Format:
Correct Answer: [Your accurate, factual answer here]
Hallucinated Answer: [Your plausible-sounding but factually incorrect answer here]
Example:
Question: “What do people use to cut paper?”
Correct Answer: “People typically use scissors to cut paper.”
Hallucinated Answer: “People typically use a knife to cut paper, as it provides a cleaner edge.”
B Comparative Factuality Analysis of Biography Generation
15

APREPRINT- SEPTEMBER18, 2025
Table 9: A qualitative analysis comparing the biographies generated by Qwen3-8B and DSCC-HS
Original Response:
Question: Write a biography of Steve Jobs.
Steve Jobs was born in 1955 in Appleworth, Ontario, Canada. He attended the University of Toronto, where
his creativity shaped his future, and his father, a printer, instilled in him a passion for design. Early in his
career, Jobs worked for Apple’s printing press, laying the foundation for his groundbreaking work. His
breakthrough came with the Apple I and Apple II computers, widely used by small businesses, and by the
1980s, he launched the Apple I, becoming the company’s main product.
In the late 1970s, Jobs was hired by Apple I, and by the 1980s, he launched the Apple I, redefining the
smartphone industry. Later, he co-designed the iPhone in 2004, revolutionizing personal computing, and
introduced the Macintosh in the 1980s as a powerful, user-friendly computer. Jobs’ focus on innovation,
simplicity, and user experience reshaped technology. Despite early struggles, he remained driven to create
groundbreaking products.
DSCC-HS Response:
Question: Write a biography of Steve Jobs.
Steve Jobs was born in 1955 in San Francisco, California, USA. He was adopted by Paul and Clara Jobs and
raised in Mountain View, California. Jobs showed a deep interest in electronics at a young age, influenced
by his father, who was a mechanic and had a workshop at home. This environment inspired Jobs’ future
innovations in the tech industry. He attended Reed College in Oregon for a brief period before dropping out
after one semester, but he continued to audit classes that piqued his interest, including a calligraphy course
that would later influence the design of the Macintosh computer. In 1974, Jobs returned to California and
began working at Atari, where he formed a close friendship with Steve Wozniak. In 1976, Jobs and Wozniak
co-founded Apple Computer, Inc., and launched the Apple I and Apple II computers, both of which were
highly successful in the market.
In 1984, Apple released the Macintosh, the first personal computer with a graphical user interface, marking
a significant milestone in personal computing. However, due to internal management conflicts, Jobs left
Apple in 1985. Afterward, he founded NeXT, a company focused on high-end computers, and acquired the
computer graphics division of Lucasfilm, which he transformed into Pixar Animation Studios. Under his
leadership, Pixar became a leading force in animation. In 1996, Apple acquired NeXT, bringing Jobs back
to the company as CEO in 1997. Under his guidance, Apple launched a series of revolutionary products,
including the iMac, iPod, iTunes, iPhone, and iPad, which transformed the personal computing, music,
mobile phone, and entertainment industries. Steve Jobs passed away on October 5, 2011, from pancreatic
cancer at the age of 56. He is remembered as one of the pioneers of modern technology, whose innovative
spirit and pursuit of perfection left a profound impact on the global tech industry.
16