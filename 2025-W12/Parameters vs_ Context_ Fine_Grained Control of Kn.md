# Parameters vs. Context: Fine-Grained Control of Knowledge Reliance in Language Models

**Authors**: Baolong Bi, Shenghua Liu, Yiwei Wang, Yilong Xu, Junfeng Fang, Lingrui Mei, Xueqi Cheng

**Published**: 2025-03-20 06:26:28

**PDF URL**: [http://arxiv.org/pdf/2503.15888v1](http://arxiv.org/pdf/2503.15888v1)

## Abstract
Retrieval-Augmented Generation (RAG) mitigates hallucinations in Large
Language Models (LLMs) by integrating external knowledge. However, conflicts
between parametric knowledge and retrieved context pose challenges,
particularly when retrieved information is unreliable or the model's internal
knowledge is outdated. In such cases, LLMs struggle to determine whether to
rely more on their own parameters or the conflicted context. To address this,
we propose **CK-PLUG**, a plug-and-play method for controlling LLMs' reliance
on parametric and contextual knowledge. We introduce a novel knowledge
consistency metric, Confidence Gain, which detects knowledge conflicts by
measuring entropy shifts in token probability distributions after context
insertion. CK-PLUG then enables fine-grained control over knowledge preference
by adjusting the probability distribution of tokens with negative confidence
gain through a single tuning parameter. Experiments demonstrate CK-PLUG's
ability to significantly regulate knowledge reliance in counterfactual RAG
scenarios while maintaining generation fluency and knowledge accuracy. For
instance, on Llama3-8B, memory recall (MR) of RAG response can be adjusted
within a broad range (9.9%-71.9%), compared to the baseline of 42.1%. Moreover,
CK-PLUG supports adaptive control based on the model's confidence in both
internal and external knowledge, achieving consistent performance improvements
across various general RAG tasks. Our code is available at:
$\href{https://github.com/byronBBL/CK-PLUG}{\text{this https URL}}$.

## Full Text


<!-- PDF content starts -->

Parameters vs. Context: Fine-Grained Control of
Knowledge Reliance in Language Models
Baolong Bi1,2Shenghua Liu1,2Yiwei Wang3Yilong Xu1,2
Junfeng Fang4Lingrui Mei1,2Xueqi Cheng1,2
1AI Safety of Chinese Academy of Sciences, Institute of Computing Technology, CAS
2University of Chinese Academy of Sciences
3University of California, Merced4National University of Singapore
{bibaolong23z,liushenghua,meilingrui25b,cxq}@ict.ac.cn
yiweiwang2@ucmerced.edu ,fangjf@nus.edu.sg
Abstract
Retrieval-Augmented Generation (RAG) mitigates hallucinations in Large Lan-
guage Models (LLMs) by integrating external knowledge. However, conflicts
between parametric knowledge and retrieved context pose challenges, particu-
larly when retrieved information is unreliable or the model’s internal knowledge
is outdated. In such cases, LLMs struggle to determine whether to rely more
on their own parameters or the conflicted context. To address this, we propose
CK-PLUG , a plug-and-play method for controlling LLMs’ reliance on parametric
and contextual knowledge. We introduce a novel knowledge consistency metric,
Confidence Gain , which detects knowledge conflicts by measuring entropy shifts
in token probability distributions after context insertion. CK-PLUG then enables
fine-grained control over knowledge preference by adjusting the probability distri-
bution of tokens with negative confidence gain through a single tuning parameter.
Experiments demonstrate CK-PLUG’s ability to significantly regulate knowledge
reliance in counterfactual RAG scenarios while maintaining generation fluency
and knowledge accuracy. For instance, on LLAMA3-8B , memory recall (MR)
of RAG response can be adjusted within a broad range (9.9%-71.9%), compared
to the baseline of 42.1%. Moreover, CK-PLUG supports adaptive control based
on the model’s confidence in both internal and external knowledge, achieving
consistent performance improvements across various general RAG tasks. Our code
is available at: https://github.com/byronBBL/CK-PLUG .
1 Introduction
Retrieval-Augmented Generation (RAG) (Lewis et al., 2020; Santhanam et al., 2021; Gao et al., 2023;
Fan et al., 2024) has become a widely adopted technique for various applications, as it effectively
integrates external knowledge with the powerful generative capabilities of Large Language Models
(LLMs) (Achiam et al., 2023; Grattafiori et al., 2024) to produce accurate responses. However,
potential knowledge conflicts (Xu et al., 2024a; Xie et al., 2023; Shi et al., 2025) between the
external context and the model’s internal parameters pose significant challenges to the reliability of
RAG-generated outputs, often leading to hallucinations (Huang et al., 2023; Tonmoy et al., 2024).
There exists an inherent trade-off between the factuality of model parameters and the fidelity of
externally retrieved context (Bi et al., 2024d). Enhancing the model’s internal factuality (Chuang
et al., 2023; Li et al., 2024a; Zhang et al., 2024b) may become unreliable as the model becomes
outdated, while excessive dependence on retrieved context (Zhou et al., 2023; Shi et al., 2024) can be
problematic due to the quality limitations of the retrieved information.
Preprint. Under review.arXiv:2503.15888v1  [cs.CL]  20 Mar 2025

Query:Who is the current president of the United States?
Model Parameters
Retrieved Context
Conflict!
Query:Who currently holds ownership of Twitter?
Model Parameters
Retrieved Context
Conflict!Donald Trump is an American entrepreneur, politician who is the President of the United States. He served as…Jack Dorsey, one of Twitter’s co-founders and former CEO, played a key role in shaping the platform and its development…I remember that Joe Biden is the President of the United States.He was elected president in 2020 and sworn in 2021.I remember that Elon Musk acquired Twitter and soon rebranded it as X, implementing significant changes to it.Parameters or Context?Which should I believe?Bad para orgoodcontextGood para orbad contextFigure 1: LLMs struggle to prioritize between parametric and contextual knowledge, especially when
facing outdated parameters or misleading context, reducing reliability in real-world scenarios.
In this paper, we argue that efficient control of knowledge reliance is crucial for the effective
deployment of RAG systems. Existing alignment to factuality (Tian et al., 2023; Lin et al., 2024a)
or context faithfulness (Bi et al., 2024a; Huang et al., 2025a) are unidirectional and uncontrollable,
lacking the flexibility for bidirectional adjustment. The degree of reliance on internal parameters
versus external context should be customizable to adapt to varying RAG scenarios, such as differences
in model capabilities or retrieval quality. As illustrated in Figure 1, in the case of outdated models
or high-quality or professional retrieval environments, the model should rely more on external
knowledge. Conversely, when the retrieval context is noisy or potentially adversarial, the model
should prioritize more its internal parameters to ensure reliable generation.
To achieve this, we propose CK-PLUG ( Controllable Knowledge Plug -in), a pluggable inference-
time approach for knowledge reliance control without modifying model parameters or architectures.
To enable fine-grained adjustment, CK-PLUG introduces the Confidence Gain metric to detect
knowledge conflicts. This metric quantifies the information gain of parameter-aware tokens after
injecting contexts, measuring the consistency between parametric knowledge and external context.
Based on this metric, CK-PLUG retains tokens exhibiting positive confidence gains (indicating
alignment between external context and the model’s parametric knowledge) while dynamically
adjusting the prediction strategy for tokens with negative confidence gains. For the latter, the
framework blends parameter-aware and context-aware token probability distributions through a
weighted fusion mechanism. The balance between these distributions is governed by a single tuning
parameter α, enabling fine-grained control over knowledge reliance preferences. Additionally, CK-
PLUG introduces an automated mode that adaptively balances parametric and contextual reliance
through entropy-based confidence evaluation, eliminating the need for manual αspecification.
We evaluate CK-PLUG on various LLMs in RAG scenarios. Under explicit αcontrol, the framework
achieves substantial adjustments in Memory Recall (MR) for QA tasks with counterfactual retrieval
contexts. For instance, on LLAMA3-8B , CK-PLUG modulates MR from 9.89% to 71.93%, signifi-
cantly deviating from the baseline MR of 42.09%. In autonomous mode ( α-free), our CK-PLUG
adaptively balances internal and external knowledge by leveraging model confidence metrics, yielding
consistent performance gains across six distinct RAG downstream tasks. Our work paves the way for
developing both knowledge-controllable and trustworthy generation capabilities for LLMs.
2 Preliminary
Language Model Generation The current language model generation process aims to pre-
dict the next words within a given context sequence. Formally, given a sequence of tokens
X={x1, x2, . . . , x t−1}, LLMs process their embeddings H={h1, h2, . . . , h t−1}to compute
the representation of the next token through transformer layers. An affine layer φ(·)is then applied
to predict the next token distribution xtover the vocabulary set V:
p(xt|x<t) = softmax( ϕ(ht)), x t∈ V (1)
During decoding, various strategies can be applied to select the next token xtbased on p(xt|x<t).
This iterative process continues until the sequence generation reaches a designated end token or
satisfies a predefined stopping condition. Our CK-PLUG controls knowledge reliance by adjusting
the probability distribution of the next token during the decoding process.
2

Perplexity Measured by Entropy Entropy (Gray, 2011) is a fundamental concept in information
theory that has been widely applied in natural language processing (NLP) (Pimentel et al., 2021;
Vanmassenhove et al., 2021). It has proven particularly valuable in quantifying uncertainty within
language modeling and generation tasks (Alon and Kamfonas, 2023; Meister et al., 2020). Given a
probability vector a∈Rn, where the entries are non-negative and the sum of all entries equals 1, the
Shannon entropy is defined as follow:
H(a) =−nX
i=1ailog2(ai) (2)
By quantifying the uncertainty in language model predictions, entropy can be used to measure
the perplexity of LLMs. Building on this principle, we compute the entropy of the post-softmax
probability distribution using Eq. (1) to measure the perplexity of next-token predictions in LLMs:
H(p(xt|x<t)) =−nX
i=1pilog2(pi) (3)
Specifically, higher entropy values correspond to greater uncertainty in LLMs’ next-token prediction,
while lower entropy reflects deterministic confidence.
3 CK-PLUG: Fine-Grained Knowledge Reliance Control
To address the challenge of dynamically balancing parametric and contextual knowledge in RAG
systems, we propose CK-PLUG, a lightweight method that achieves granular control over language
models’ knowledge reliance via token-level probability modulation. In this section, we provide
further details about our CK-PLUG. Section 3.1 introduces the knowledge conflict detection based
on information gain, which serves as the operational switch for CK-PLUG. Section 3.2 explains
the principle behind CK-PLUG’s modulation of knowledge between parameters and context, while
Section 3.3 discusses how CK-PLUG enables adaptive knowledge adjustment.
3.1 Knowledge Conflicts Detection with Confidence-Gain
CK-PLUG achieves fine-grained knowledge control through token-level probability modulation.
Adjusting only key tokens can positively influence knowledge preference, whereas indiscriminately
modifying all tokens can lead to a catastrophic collapse in generation quality (Bi et al., 2024f; Lin
et al., 2024b). To this end, we introduce a knowledge conflict detection mechanism as CK-PLUG’s
activation switch. This mechanism identifies tokens that exhibit potential conflicts between the LLM’s
parametric knowledge and the retrieved contextual knowledge, enabling targeted intervention.
First, we define the next-token prediction in model generation for a query Xqas follows:
•p(x|Xq): Predictions conditioned solely on the input query Xq, reflecting the model’s internal
parametric knowledge.
•p(x|Xr+Xq): Predictions conditioned on both query Xqand retrieved context Xr, integrating
parametric and external knowledge.
Here, the augmented distribution p(x|Xr+Xq)serves as the objective of RAG precess, reflecting
the LLM’s response based on both its internal parameters and external context, while parametric
distribution p(x|Xq)represents predictions solely derived from the model’s parametric knowledge.
Inspired by uncertainty quantification in token logits (Duan et al., 2023, 2024; Ma et al., 2025),
we employ information entropy to measure prediction perplexity. Based on Equation 3, we define
H(p(x|Xq))as the entropy of the parametric predictions and H(p(x|Xr+Xq))as the entropy of
the retrieval-augmented predictions.
We utilize the NQ dataset (Kwiatkowski et al., 2019) to evaluate the feasibility of entropy-based
detection, along with Conflict Contexts (containing counterfacts contradicting parametric knowledge)
andSupport Contexts (retrieved factual evidence). We design a knowledge capture algorithm that
aggregates the entropy of tokens corresponding to the decoded gold answer under both conflict and
supportive conditions (see details in Appendix C.1 and D). For example, as depicted in Figure 3,
when decoding terms such as “Dutch” or “Israel”, we record the entropy of the token probability
distributions at the relevant positions, which reflects the confidence in core knowledge.
3

Query: What is the native language of Kevin De Bruyne? Context: Kevin De Bruyne is a … he speaks fluent English, Output: ThenativelanguageofKevinDeBru-
47.3
-3.43
0.15
-0.4
1.65
0.1
0.18
1.56
0.3
-43.8
-0.1-yneisDutch.  Howeverhe also speaks
16.9
0.4
27.6
21.1
61.3
81.3
36.5his native language. … [ground truth: Dutch],fluent English  .  …
Query: What country is Tim Ballard a citizen of? Context: … Tim Ballard is a citizen of the Israel. He is aOutput: Tim Ballard  is  a  citizen of Israel.He  is knownfor his work in uncover--ingthe hidden history of Israel …
23.6
0.27
0.7
54.7
62.3
89.8
-47.5
92.7
9.4
76.3
108.3
151.5
23.9
8.6
107.7
-0.13
56.9
81.3
32.3
74.6
9.7
-10.8Perplexity
Confidencerenowned historian, …[ground truth: United States]Conflict &Follow ParametersConflict &Follow ContextFigure 3: Illustration of the Confidence-Gain ( CG) onLLAMA3-8B for generated tokens under two
types of Conflict Context , demonstrating its effectiveness in detecting latent knowledge conflicts. For
comparison, examples of Support Context are provided in the Appendix B.
ConﬂictSupport0204060Average. Entropy24.0343.7531.8614.06ConﬂictSupport0255075Average. Entropy42.8170.7865.9252.03
ConﬂictSupport020406080Average. Entropy55.8942.1260.6419.75ConﬂictSupport0204060Average. Entropy31.4249.4129.7212.30(c) Mistral-v0.3-7B(d) Qwen-2.5-7BBefore RAGAfter RAG
(a) LLaMA-2-7B(b) LLaMA-3-8B
Figure 2: Changes (%) in the entropy of probability
distribution for knowledge-sensitive tokens after
incorporating conflict or support contexts.Figure 2 compares the entropy changes before
and after context insertion in both conflict and
support scenarios. We observed that, in com-
parison, inserting Conflict Context increases en-
tropy, reflecting a more disordered probability
distribution and reduced confidence in model
responses. In contrast, Support Context sig-
nificantly decrease entropy, indicating that the
model becomes more confident when its inter-
nal knowledge is corroborated by external in-
formation. Although the changes under conflict
conditions are less pronounced in Figures 2 (c)
and (d), the marked entropy reduction in sup-
portive scenarios further highlights the model’s
confusion when faced with conflicting inputs.
Based on these observations, we propose a met-
ric termed Confidence Gain (CG) to evaluate the
change in model confidence before and after context insertion during decoding. Given the probability
distributions p(x|Xq)andp(x|Xr+Xq), CG is computed as follows:
CG=H(p(x|Xq))−H(p(x|Xr+Xq)) (4)
As shown in Figure 3, CGeffectively measures the confidence shift of each token when incorporating
retrieved context during generation. If the confidence drops significantly (i.e., CGfalls below 0 or a
predefined threshold specified in the Appendix B), the token is identified as a potential knowledge
conflict. We then apply subsequent knowledge reliance modulation to these conflicting tokens.
3.2 Parameters-Context Reliance Modulation
In various RAG scenarios, the quality of retrieved texts may vary, necessitating user control over the
reliance on either parametric knowledge or retrieved context. This control should be lightweight,
avoiding the need to train multiple model versions. CK-PLUG efficiently achieves fine-grained
knowledge reliance modulation by intervening in the probability distribution of the next-token
prediction during the decoding phase.
During LLM inference in RAG, we define the parameter-aware log probability distribution as:
qpara(x|Xr+Xq)) = log p(x|Xq) (5)
where the query Xqserves as the prompt, concatenated with the previously generated tokens in RAG
as a prefill, to elicit the next-token prediction from the model’s parametric knowledge. In contrast,
the next-token prediction in RAG incorporates both parametric knowledge and retrieved context. By
subtracting the parameter-aware log probability from the original log probability distribution, we
isolate the contribution of retrieved context, capturing its influence on token prediction. This leads to
the definition of the context-aware distribution:
qcont(x|Xr+Xq) = logp(x|Xr+Xq)
p(x|Xq)(6)
4

Query:What is the native language of Kevin De Bruyne? 
<Query> A: Hisnativelanguageis    RAG Output:Parametric Output:
Copy to Fill
<Context><Query> A: Hisnativelanguageis
Input Content:Context-Aware
𝜶×𝜶
Next token logits distribution:VocabularyEnglishDutchFrenchGerman…Parametric distribution
Next token logits distribution:VocabularyEnglishDutchFrenchGerman…CGDetecting
VocabularyEnglishDutchFrenchGerman…Next token logits distribution:RAG-Aware
Conflict!
EnglishDutchFrenchGerman…Next token logits distribution:Vocabulary
×(𝟏−𝜶)Reliance ModulationSum↑𝜶↓EnglishDutchFrenchGerman…VocabularyNext token logits distribution:SubtractParameter-Preferred
Context-PreferredParameter-AwareShould I trust context (English) or my parameters (Dutch)?LLM InferenceContext: Kevin De Bruyne is a professional footballer … he speaks fluent English, his native language. …Figure 4: Illustration of CK-PLUG controlling the knowledge reliance in LLM outputs. During
token generation, it detects potential conflicts and modulates the probability distribution of conflicted
tokens. The modulation first computes a context-aware distribution, then integrates it with the
parameter-aware distribution through a weighted sum based on the tuning parameter α.
As shown in Figure 4, the core idea of CK-PLUG is to regulate knowledge reliance by modulating
the parameter-aware and context-aware prediction distributions, particularly for tokens that indicate
potential knowledge conflicts. Using q(x)as a shorthand for q(x|Xr+Xq), we compute the resulting
distribution for next-word prediction as follows:
ˆp(x|Xr+Xq) =(
softmax
F 
qcont(x), qpara(x)
,ifCG < 0,
p(x|Xr+Xq), otherwise.(7)
where CGrepresents the confidence gain metric, indicating whether retrieved context introduces
conflicting information. We introduce a tunable hyperparameter αto control the balance between
parametric and contextual reliance. The modulation function is defined as:
F(qcont(x), qpara(x)) =α·qpara+ (1−α)·qcont,ifx∈ Vhead(x|Xr+Xq),
−∞, otherwise.(8)
Following adaptive plausibility constraint (Li et al., 2022), we define the subset Vhead(x|Xr+Xq)⊂
Vas the union of the top- ktokens from both parameter-aware and context-aware distributions:
Vhead(x|Xr+Xq) =
x∈ Vqpara(x)> qpara 
xR=k
para	
∪
x∈ Vqcont(x)> qcont 
xR=k
cont	
(9)
Here, xR=krepresents the k-th ranked token in the parameter-aware or context-aware distribution.
Taking their union ensures that context-related tokens with low confidence are also considered.
Through this modulation mechanism, we achieve controllable adjustment of the relative contributions
of parametric and contextual knowledge. The reliance can be finely controlled with a single hyperpa-
rameter α: increasing αmakes the model more dependent on internal knowledge, while decreasing α
shifts focus toward the retrieved context, even when it conflicts with parametric knowledge.
3.3 Adaptive Knowledge Adjustment
CK-PLUG also can autonomously balances parametric and contextual dependencies through entropy-
based perplexity. For notational brevity, let Hpara replace H(p(x|Xq))to represent parametric
perplexity and Hcontreplace H(p(x|Xr+Xq))to denote contextual perplexity after retrieval injec-
tion. Since higher entropy corresponds to lower model confidence, we reformulate the modulation
parameter αin Equation 8 as a normalized ratio of perplexities:
α=Hcont
Hpara+Hcont(10)
This eliminates manual α-specification, enabling CK-PLUG to explicitly balance knowledge reliance
based on the model confidence, enhancing both interpretability and trustworthiness in generation.
5

Model MethodNQ ConFiQA MQUAKE
ConR ParR MR ConR ParR MR ConR ParR MR
Baseline 43.3 43.8 50.2 (-) 69.7 28.1 28.8 (-) 31.2 21.6 40.9 (-)
LLAMA2- α= 0.0 61.6 8.6 12.3 (↓75.5) 71.5 9.2 11.4 (↓60.4) 40.7 10.8 21.0 (↓48.7)
7B- CHAT α= 0.5 45.6 32.2 41.4 (↓17.5) 67.5 24.0 26.2 (↓9.0) 24.6 14.6 41.7 (↑2.0)
α= 1.0 23.2 58.2 71.5 (↑42.4) 31.5 46.2 59.4 (↑106.3) 11.6 43.2 79.9 (↑95.4)
Baseline 43.9 34.1 43.5 (-) 54.2 22.4 29.2 (-) 18.7 17.9 48.9 (-)
LLAMA3- α= 0.0 63.5 7.3 9.9 (↓77.2) 65.2 11.4 14.9 (↓48.9) 42.1 15.5 26.9 (↓45.0)
8B- INSTRUCT α= 0.5 44.7 32.5 42.1 (↓3.2) 51.7 17.9 25.7 (↓12.0) 20.2 20.4 50.3 (↑2.9)
α= 1.0 22.5 57.6 71.9 (↑65.3) 25.4 42.3 62.5 (↑114.0) 14.5 47.1 76.5 (↑56.4)
Baseline 46.2 58.6 55.9 (-) 64.7 25.9 28.6 (-) 43.8 21.2 32.6 (-)
MISTRAL 0.3 α= 0.0 75.8 15.8 17.2 (↓69.3) 70.7 10.9 13.4 (↓53.1) 65.8 12.4 15.9 (↓51.2)
7B- INSTRUCT α= 0.5 46.2 58.6 55.9 (-) 65.7 25.9 28.6 (-) 43.5 22.2 33.8 (↑3.7)
α= 1.0 27.9 69.1 72.2 (↑29.2) 29.9 43.8 59.5 (↑108.0) 15.2 50.4 76.8 (↑135.6)
Baseline 73.4 32.4 31.3 (-) 43.8 15.4 26.1 (-) 32.2 13.0 28.8 (-)
QWEN 2.5- α= 0.0 85.4 8.3 9.0 (↓71.3) 65.2 13.9 17.6 (↓32.5) 49.3 12.8 20.6 (↓28.5)
7B- INSTRUCT α= 0.5 72.0 26.8 27.1 (↓13.4) 43.8 14.9 25.4 (↓2.7) 32.8 13.2 28.8 (-)
α= 1.0 30.2 51.4 63.2 (↑101.9) 36.8 28.4 43.5 (↑66.7) 19.8 32.8 62.4 (↑116.7)
Table 1: Performance (%) of CK-PLUG in controlling knowledge reliance, with αset to 0.0, 0.5, and
1.0.Red markers denote sharp MR decreases indicating enhanced contextual alignment, while green
markers highlight significant MR increases reflecting strengthened parametric reliance.
4 Experimental Methodology
Models and Tasks We integrate CK-PLUG into the generation process of LLMs by modifying
the decoding operation. Our experiments evaluate the performance of CK-PLUG on four popular
open-source LLMs: LLAMA2-7B (Touvron et al., 2023), LLAMA3-8B (Grattafiori et al., 2024),
MISTRALV 0.3-7B (Jiang et al., 2023a), and QWEN 2.5-7B (Yang et al., 2024). We assess CK-
PLUG’s effectiveness in both knowledge reliance control (Section 3.2) and adaptive generation
enhancement (Section 3.3). See Appendix C.1 for details about the datasets and implementation.
Evaluation for Knowledge Control To evaluate the effectiveness of CK-PLUG in modulating the
reliance on parametric and contextual knowledge, we simulate a RAG environment with knowledge
conflicts. Specifically, we modify the retrieved contexts in the NQ dataset to contain factually
incorrect statements related to the answers, following Longpre et al. (2021). Additionally, we
incorporate ConFiQA (Bi et al., 2024a) and MQuAKE (Zhong et al., 2023), which provide noisy
counterfactual contexts and knowledge editing instructions, respectively. These tasks introduce
counterfactual information that conflicts with the model’s parametric knowledge. We use ConR
(the recall of context) and ParR (the recall of parameters). ConR measures whether the generated
responses align with the provided context, while ParR evaluates their alignment with the model’s
parametric knowledge. Specifically, we also adopt the memorization ratio MR=ParR
ParR+ConR, which
captures the tendency to favor parametric knowledge over retrieved context.
Evaluation for Adaptive Enhancement We evaluate the effectiveness of CK-PLUG’s adaptive
adjustment in a general RAG setting. Specifically, we use Wikipedia1as the corpus and BGE-base-
v1.5 (Xiao et al., 2023) as the retriever. Our evaluation covers six diverse RAG tasks from the KILT
benchmark (Petroni et al., 2021), including Open-Domain QA on NQ (Kwiatkowski et al., 2019),
Multi-Hop QA on HotpotQA (Yang et al., 2018), Fact Verification on FEVER (Thorne et al., 2018),
Slot Filling on T-REX (Elsahar et al., 2018), Long-Form QA on ELI5 (Fan et al., 2019), and Dialogue
Generation on WOW (Dinan et al., 2019). Specifically, we use normalized accuracy to evaluate the
first four tasks, while Rouge-L and F1 scores are used to assess ELI5 and WOW, respectively.
1Dump from http://dl.fbaipublicfiles.com/BLINK/enwiki-pages-articles.xml.bz2
6

Model Method NQ HotpotQA FEVER T-REX Eli5 WOW
LLAMA2-w/o RAG 30.1 27.5 55.9 11.8 13.8 13.6
7B- CHATw/ RAG 41.4 40.9 66.2 42.3 13.3 14.5
RAG w/ CK-PLUG 43.7 42.6 72.6 41.7 14.1 15.2
LLAMA3-w/o RAG 32.1 32.2 73.6 19.1 14.0 13.1
8B- INSTUCTw/ RAG 45.2 43.1 86.4 51.5 14.0 13.6
RAG w/ CK-PLUG 46.5 46.7 86.1 52.3 14.8 14.3
MISTRAL 0.3w/o RAG 34.7 32.3 74.2 29.3 15.7 13.9
7B- INSTRUCTw/ RAG 47.8 44.9 89.5 57.8 15.5 14.8
RAG w/ CK-PLUG 49.5 46.2 89.2 58.1 15.8 15.3
QWEN 2.5w/o RAG 27.7 27.1 56.4 25.0 14.2 13.0
7B- INSTRUCTw/ RAG 49.5 47.9 85.6 61.6 13.8 14.0
RAG w/ CK-PLUG 50.3 48.5 87.8 60.2 14.3 14.5
Table 2: Results (%) on the adaptive enhancement of CK-PLUG across six diverse RAG tasks.
5 Evaluation Results
5.1 Overall Performance
CK-PLUG Enables Wide-Range Knowledge Reliance Control Table 1 presents the knowledge
control results of CK-PLUG across different evaluation settings. Specifically, NQ evaluates standard
QA, ConFIQA assesses long-context QA, and MQuAKE examines multi-turn QA, all designed to
measure knowledge reliance under counterfactual contexts. Compared to the baseline, when α=
0.0, CK-PLUG enhances context reliance (increased ConR ) while reducing reliance on parametric
knowledge (decreased ParaR ). Conversely, at α= 1.0, the trend is reversed. The substantial variation
inMRfurther underscores CK-PLUG’s effectiveness in controlling knowledge reliance. For instance,
onLLAMA2-7B , CK-PLUG adjusts MRover a broad range, from 14.9% to 70.3% on average.
Furthermore, at α= 0.5, the model’s performance closely aligns with the baseline, exhibiting only
minor fluctuations. This suggests that CK-PLUG effectively balances parametric and contextual
knowledge alignment with the model’s inherent knowledge attention, aligning with our expectation
of smooth and linear modulation of knowledge preference.
CK-PLUG Enhances Generation Reliability with Adaptive Control CK-PLUG autonomously
adjusts αto enhance generation reliability. As shown in Table 2, CK-PLUG improves overall
performance across six distinct tasks compared to baselines with or without retrieved contexts.
These results demonstrate CK-PLUG’s ability to strengthen reliability through adaptive parametric-
contextual knowledge balancing. Notably, when the performance of systems without RAG and with
RAG are close, which suggests that parameter and contextual knowledge contribute differently to
reliable generation, CK-PLUG effectively balances them to achieve a more significant improvement.
5.2 Fine-Grained Control via a Single Tuning Parameter
0.0 0.25 0.5 0.75 1.0
Value of Tuning Parameter 
20406080MR (%)
Llama2-7B
Llama3-8B
Mistral-7B
Qwen2.5-7B
Figure 5: Variation in MR (%) across different
language models as parameter αincreases.CK-PLUG employs a single parameter αto reg-
ulate the model’s reliance on contextual knowl-
edge versus parameterized knowledge. Figure
5 illustrates the impact of fine-grained adjust-
ments on MR within the NQ dataset, which in-
cludes counterfactual contexts. Due to intrinsic
differences among models, the variation in MR
asαchanges exhibits slight discrepancies. No-
tably, while most models follow a highly con-
sistent pattern, QWEN 2.5-7B shows a distinct
behavior, particularly when α≥0.5, where the
increase in MR slows down. This observation
aligns with prior findings by Bi et al. (2024a),
which suggest that QWEN models tend to be more confident in its parametric knowledge when it
7

(a) LLaMA-2-7B-Chat(b) LLaMA-3-8B-Instruct(c) Mistral-v0.3-7B-Instruct(d) Qwen-2.5-7B-Instruct
Parametric KnowledgeContextual KnowledgeFigure 6: Average probabilities (%) of the parametric and contextual knowledge components in
knowledge-aware tokens, which increase and decrease respectively with increasing parameter α.
conflicts with the provided context. Nevertheless, the trend remains approximately linear, ensuring
smooth modulation and CK-PLUG’s adaptability across applications.
5.3 Ablation Study
Model Setting Baseline a = 0.0 a = 0.5 a = 1.0
LLAMA2-Baseline 78.9 - - -
7B- CHATw/ ConD - 74.8 77.8 78.5
w/o ConD - 30.7 58.4 62.3
LLAMA3-Baseline 83.7 - - -
8B- INSTRUCTw/ ConD - 82.4 83.5 86.7
w/o ConD - 53.8 61.4 73.2
MISTRAL -Baseline 92.4 - - -
INSTRUCTw/ ConD - 89.5 92.4 89.7
w/o ConD - 49.9 91.3 75.4
QWEN -Baseline 88.8 - - -
INSTRUCTw/ ConD - 89.6 89.8 87.2
w/o ConD - 62.4 85.4 51.3
Table 3: Hit rate (%) of our CK-PLUG with and with-
out conflict detection (ConD). The Baseline represents
standard RAG without CK-PLUG.Knowledge conflict detection (ConD) is a
crucial component of CK-PLUG, ensuring
that knowledge modulation is applied only
to tokens that could potentially trigger con-
flicts (Section 3.1). Without this selective
adjustment, excessive modulation may lead
to catastrophic generation failures. To vali-
date the importance of this module, we con-
duct an ablation study on the NQ dataset
with counterfactual contexts. Specifically,
we use the hit rate as a metric to evaluate
generation quality, measuring whether the
model output contains either the original
parametric answer or the gold answer from
the context. The results, presented in Table
3, show that CK-PLUG with ConD main-
tains a hit rate comparable to the baseline
across different models. In contrast, removing ConD leads to a noticeable decline ( highlighted ),
particularly in LLAMA models and under extreme knowledge modulation settings ( α=0.0 or α=1.0).
This demonstrates that ConD effectively identifies tokens requiring modulation, ensuring reliable
generation while preventing the risks associated with excessive adjustments.
00.51.0Probability of Contextual Knowledge0123DensityÆ=0.0Æ=0.5Æ=1.0baseline00.51.0Probability of Parametric Knowledge0123DensityÆ=0.0Æ=0.5Æ=1.0baseline(a) LLaMA-2-7B-Chat(b) LLaMA-3-8B-Instruct
(c) Mistral-v0.3-7B-Instruct(d) Qwen-2.5-7B-Instruct00.51.0Probability of Contextual Knowledge0123DensityÆ=0.0Æ=0.5Æ=1.0baseline00.51.0Probability of Parametric Knowledge0123DensityÆ=0.0Æ=0.5Æ=1.0baseline
00.51.0Probability of Contextual Knowledge0123DensityÆ=0.0Æ=0.5Æ=1.0baseline00.51.0Probability of Parametric Knowledge0123DensityÆ=0.0Æ=0.5Æ=1.0baseline00.51.0Probability of Contextual Knowledge0123DensityÆ=0.0Æ=0.5Æ=1.0baseline00.51.0Probability of Parametric Knowledge0123DensityÆ=0.0Æ=0.5Æ=1.0baseline
Figure 7: Experimental results of kernel density estimation for the softmax probability distribution of
tokens reflecting parametric and contextual reliance across different αsettings.
5.4 Deep Insights into Knowledge between Parameters and Context
The previous results have demonstrated CK-PLUG’s effectiveness in controlling knowledge reliance.
However, a deeper analysis is necessary to ensure the reliability of this modulation. In this section,
we investigate the impact of CK-PLUG on model outputs from an interpretability perspective.
8

LLAMA3-8BQuery Where was the capital of the habsburg empire located? [Ground Truth: Vienna]
ContextThe Habsburg Monarchy (German: Habsburgermonarchie) or Empire is an unofficial appellation among historians for the
countries and provinces... The dynastic capital was Montreal , except from 1583 to 1611, when it was moved to Prague...
OutputThe Dynastyczal capital Montreal except from 1583 to 1611 when it was moved to Pragma. From 1804 to 1867 the
Haburg Monarchy was formely unified as... Context Reliance (α=0.0)
The capital of the Habsburg Empire was Vienna , Austria. Parameter Reliance (α=1.0)
MISTRAL -7BQuery When did the us not go to the olympics? [Ground Truth: 1980]
ContextThe1981 Summer Olympics boycott was one part of a number of actions initiated by the United States to protest the Soviet
invasion of Afghanistan. The Soviet Union, which hosted the 1981 Summer Olympics, and other countries would later
boycott the 1984 Summer Olympics in Los Angeles.
OutputIn1981 , the United States led a boycott of the Summer Olympics in Moscow in protest of the Soviet invasion of Afghanistan.
The following year, the Soviet Union and other countries... Context Reliance (α=0.0)
The United States has participated in every modern Olympic Games since the 1896 Summer Olympics in Athens, Greece,
except for the 1980 Summer Olympics in Moscow... Parameter Reliance (α=1.0)
Table 4: Case studies on LLAMA3-8B andMISTRALV 0.3-7B for CK-PLUG’s knowledge control
(α= 0.0andα= 1.0).Green text indicates the ground truth and its parametric match in the output
while reddenotes the counterfactual content in context and its corresponding faithful match.
To achieve this, we design a specialized algorithm to capture the probability distribution of the
first token in the model’s response that reflects knowledge reliance. For example, given the query,
"In which country is London located? " with the provided context, " London is a city in France ", a
parametric response might be " London is located in England " while a context-dependent response
would be “ London is located in France ”. The algorithm automatically detects the first decoded token
corresponding to "England" or "France" (or their prefix substrings like "Eng-" or "Fran-"), effectively
capturing the model’s knowledge reliance. Based on this, we obtain the probability of this token
being generated under parametric or contextual dependence (e.g., "English" and "Dutch" in Figure 3).
The details of the algorithm is provided in the Appendix D.
We apply this token-level probability analysis to the ConFIQA dataset. Figure 6 presents the
average probabilities of parametric and contextual knowledge under different values of α. As α
increases, the probability of contextual knowledge decreases, while the probability of parametric
knowledge correspondingly increases. This aligns with our previous observations in Section 5.2,
where QWEN 2.5-7B exhibits strong confidence in its parametric knowledge when unreliable context
is introduced. Figure 7 provides a more detailed probability distribution analysis: for parametric
knowledge, smaller αvalues concentrate probabilities in the lower range, while larger αvalues shift
them to the higher range; for contextual knowledge, the trend is reversed. These fine-grained results
offer deeper insights into CK-PLUG’s behavior, illustrating how it effectively modulates knowledge
dependence at the token level to control the model’s knowledge preference in generation.
5.5 Case Study
Table 4 presents case studies on NQ dataset, showing that CK-PLUG significantly alters LLM outputs
while maintaining fluency and logical consistency. This suggests CK-PLUG fundamentally regulates
knowledge reliance rather than merely forcing token-level changes. More cases are in Appendix E.
6 Conclusion
In this work, we argue that LLMs should have personalized knowledge reliance preferences tailored
to different RAG scenarios. We introduce CK-PLUG, a plug-and-play method for controlling LLMs’
reliance on parametric and contextual knowledge. We use Confidence Gain to detect potential con-
flicts in generated tokens and apply a single parameter to modulate the token probability distribution
between parametric and contextual components for tokens with negative confidence gain. Addition-
ally, CK-PLUG offers an adaptive method for adjusting knowledge reliance to enhance generation
reliability. Experimental results demonstrate that CK-PLUG enables smooth control over knowledge
reliance while maintaining generation coherence, and consistently improves performance across a
wide range of RAG tasks. Our findings emphasize the need for explicit knowledge reliance control
and offer a practical framework for balancing parametric and contextual knowledge in LLMs.
9

References
Achiam, J., Adler, S., Agarwal, S., Ahmad, L., Akkaya, I., Aleman, F. L., Almeida, D., Altenschmidt,
J., Altman, S., Anadkat, S., et al. (2023). Gpt-4 technical report. arXiv preprint arXiv:2303.08774 .
Alon, G. and Kamfonas, M. (2023). Detecting language model attacks with perplexity. arXiv preprint
arXiv:2308.14132 .
Bi, B., Huang, S., Wang, Y ., Yang, T., Zhang, Z., Huang, H., Mei, L., Fang, J., Li, Z., Wei, F.,
et al. (2024a). Context-dpo: Aligning language models for context-faithfulness. arXiv preprint
arXiv:2412.15280 .
Bi, B., Liu, S., Mei, L., Wang, Y ., Ji, P., and Cheng, X. (2024b). Decoding by contrasting knowledge:
Enhancing llms’ confidence on edited facts. arXiv preprint arXiv:2405.11613 .
Bi, B., Liu, S., Wang, Y ., Mei, L., and Cheng, X. (2024c). Lpnl: Scalable link prediction with large
language models. arXiv preprint arXiv:2401.13227 .
Bi, B., Liu, S., Wang, Y ., Mei, L., Fang, J., Gao, H., Ni, S., and Cheng, X. (2024d). Is factuality
enhancement a free lunch for llms? better factuality can lead to worse context-faithfulness. arXiv
preprint arXiv:2404.00216 .
Bi, B., Liu, S., Wang, Y ., Mei, L., Gao, H., Fang, J., and Cheng, X. (2024e). Struedit: Structured
outputs enable the fast and accurate knowledge editing for large language models. arXiv preprint
arXiv:2409.10132 .
Bi, B., Liu, S., Wang, Y ., Mei, L., Gao, H., Xu, Y ., and Cheng, X. (2024f). Adaptive token biaser:
Knowledge editing via biasing key entities. arXiv preprint arXiv:2406.12468 .
Chuang, Y .-S., Xie, Y ., Luo, H., Kim, Y ., Glass, J., and He, P. (2023). Dola: Decoding by contrasting
layers improves factuality in large language models. arXiv preprint arXiv:2309.03883 .
Dinan, E., Roller, S., Shuster, K., Fan, A., Auli, M., and Weston, J. (2019). Wizard of Wikipedia:
Knowledge-powered conversational agents. In Proceedings of the International Conference on
Learning Representations (ICLR) .
Duan, J., Cheng, H., Wang, S., Zavalny, A., Wang, C., Xu, R., Kailkhura, B., and Xu, K. (2023).
Shifting attention to relevance: Towards the predictive uncertainty quantification of free-form large
language models. arXiv preprint arXiv:2307.01379 .
Duan, J., Zhang, R., Diffenderfer, J., Kailkhura, B., Sun, L., Stengel-Eskin, E., Bansal, M., Chen,
T., and Xu, K. (2024). Gtbench: Uncovering the strategic reasoning limitations of llms via
game-theoretic evaluations. arXiv preprint arXiv:2402.12348 .
Elsahar, H., V ougiouklis, P., Remaci, A., Gravier, C., Hare, J., Laforest, F., and Simperl, E. (2018).
T-rex: A large scale alignment of natural language with knowledge base triples. In Proceedings of
the Eleventh International Conference on Language Resources and Evaluation (LREC 2018) .
Fan, A., Jernite, Y ., Perez, E., Grangier, D., Weston, J., and Auli, M. (2019). Eli5: Long form
question answering. arXiv preprint arXiv:1907.09190 .
Fan, W., Ding, Y ., Ning, L., Wang, S., Li, H., Yin, D., Chua, T.-S., and Li, Q. (2024). A survey on
rag meeting llms: Towards retrieval-augmented large language models. In Proceedings of the 30th
ACM SIGKDD Conference on Knowledge Discovery and Data Mining , pages 6491–6501.
Fang, J., Jiang, H., Wang, K., Ma, Y ., Wang, X., He, X., and Chua, T.-s. (2024). Alphaedit: Null-space
constrained knowledge editing for language models. arXiv preprint arXiv:2410.02355 .
Gao, Y ., Xiong, Y ., Gao, X., Jia, K., Pan, J., Bi, Y ., Dai, Y ., Sun, J., Wang, H., and Wang, H.
(2023). Retrieval-augmented generation for large language models: A survey. arXiv preprint
arXiv:2312.10997 , 2.
Grattafiori, A., Dubey, A., Jauhri, A., Pandey, A., Kadian, A., Al-Dahle, A., Letman, A., Mathur,
A., Schelten, A., Vaughan, A., et al. (2024). The llama 3 herd of models. arXiv preprint
arXiv:2407.21783 .
10

Gray, R. M. (2011). Entropy and information theory . Springer Science & Business Media.
Gunjal, A., Yin, J., and Bas, E. (2024). Detecting and preventing hallucinations in large vision
language models. In Proceedings of the AAAI Conference on Artificial Intelligence , pages 18135–
18143.
Guu, K., Lee, K., Tung, Z., Pasupat, P., and Chang, M.-W. (2020). Realm: Retrieval-augmented
language model pre-training.
Huang, L., Yu, W., Ma, W., Zhong, W., Feng, Z., Wang, H., Chen, Q., Peng, W., Feng, X., Qin, B.,
and Liu, T. (2023). A survey on hallucination in large language models: Principles, taxonomy,
challenges, and open questions.
Huang, P., Liu, Z., Yan, Y ., Yi, X., Chen, H., Liu, Z., Sun, M., Xiao, T., Yu, G., and Xiong, C. (2025a).
Pip-kag: Mitigating knowledge conflicts in knowledge-augmented generation via parametric
pruning. arXiv preprint arXiv:2502.15543 .
Huang, P., Liu, Z., Yan, Y ., Yi, X., Chen, H., Liu, Z., Sun, M., Xiao, T., Yu, G., and Xiong, C. (2025b).
Pip-kag: Mitigating knowledge conflicts in knowledge-augmented generation via parametric
pruning.
Izacard, G. and Grave, E. (2021). Leveraging passage retrieval with generative models for open
domain question answering.
Jiang, A. Q., Sablayrolles, A., Mensch, A., Bamford, C., Chaplot, D. S., de las Casas, D., Bressand,
F., Lengyel, G., Lample, G., Saulnier, L., Lavaud, L. R., Lachaux, M.-A., Stock, P., Scao, T. L.,
Lavril, T., Wang, T., Lacroix, T., and Sayed, W. E. (2023a). Mistral 7b.
Jiang, Z., Xu, F. F., Gao, L., Sun, Z., Liu, Q., Dwivedi-Yu, J., Yang, Y ., Callan, J., and Neubig,
G. (2023b). Active retrieval augmented generation. In Proceedings of the 2023 Conference on
Empirical Methods in Natural Language Processing , pages 7969–7992.
Kwiatkowski, T., Palomaki, J., Redfield, O., Collins, M., Parikh, A., Alberti, C., Epstein, D.,
Polosukhin, I., Devlin, J., Lee, K., et al. (2019). Natural questions: a benchmark for question
answering research. Transactions of the Association for Computational Linguistics , 7:453–466.
Lewis, P., Perez, E., Piktus, A., Petroni, F., Karpukhin, V ., Goyal, N., Küttler, H., Lewis, M., Yih,
W.-t., Rocktäschel, T., et al. (2020). Retrieval-augmented generation for knowledge-intensive nlp
tasks. Advances in neural information processing systems , 33:9459–9474.
Li, K., Patel, O., Viégas, F., Pfister, H., and Wattenberg, M. (2024a). Inference-time intervention:
Eliciting truthful answers from a language model.
Li, X. L., Holtzman, A., Fried, D., Liang, P., Eisner, J., Hashimoto, T., Zettlemoyer, L., and Lewis,
M. (2022). Contrastive decoding: Open-ended text generation as optimization. arXiv preprint
arXiv:2210.15097 .
Li, Z., Jiang, H., Chen, H., Bi, B., Zhou, Z., Sun, F., Fang, J., and Wang, X. (2025). Reinforced
lifelong editing for language models. arXiv preprint arXiv:2502.05759 .
Li, Z., Li, C., Zhang, M., Mei, Q., and Bendersky, M. (2024b). Retrieval augmented generation
or long-context llms? a comprehensive study and hybrid approach. In Proceedings of the 2024
Conference on Empirical Methods in Natural Language Processing: Industry Track , pages 881–
893.
Lin, S.-C., Gao, L., Oguz, B., Xiong, W., Lin, J., Yih, S., and Chen, X. (2024a). Flame: Factuality-
aware alignment for large language models. Advances in Neural Information Processing Systems ,
37:115588–115614.
Lin, Z., Liang, T., Xu, J., Wang, X., Luo, R., Shi, C., Li, S., Yang, Y ., and Tu, Z. (2024b). Critical
tokens matter: Token-level contrastive estimation enhence llm’s reasoning capability. arXiv
preprint arXiv:2411.19943 .
Liu, F., Liu, Y ., Shi, L., Huang, H., Wang, R., Yang, Z., and Zhang, L. (2024). Exploring and
evaluating hallucinations in llm-powered code generation. arXiv preprint arXiv:2404.00971 .
11

Longpre, S., Perisetla, K., Chen, A., Ramesh, N., DuBois, C., and Singh, S. (2021). Entity-based
knowledge conflicts in question answering. arXiv preprint arXiv:2109.05052 .
Luo, J., Li, T., Wu, D., Jenkin, M., Liu, S., and Dudek, G. (2024). Hallucination detection and
hallucination mitigation: An investigation. arXiv preprint arXiv:2401.08358 .
Ma, H., Chen, J., Wang, G., and Zhang, C. (2025). Estimating llm uncertainty with logits. arXiv
preprint arXiv:2502.00290 .
Mei, L., Liu, S., Wang, Y ., Bi, B., Mao, J., and Cheng, X. (2024a). " not aligned" is not" mali-
cious": Being careful about hallucinations of large language models’ jailbreak. arXiv preprint
arXiv:2406.11668 .
Mei, L., Liu, S., Wang, Y ., Bi, B., Yuan, R., and Cheng, X. (2024b). Hiddenguard: Fine-grained safe
generation with specialized representation router.
Meister, C., Salesky, E., and Cotterell, R. (2020). Generalized entropy regularization or: There’s
nothing special about label smoothing. arXiv preprint arXiv:2005.00820 .
Meng, K., Bau, D., Andonian, A., and Belinkov, Y . (2022). Locating and editing factual associations
in gpt. Advances in neural information processing systems , 35:17359–17372.
Muennighoff, N., Tazi, N., Magne, L., and Reimers, N. (2022). Mteb: Massive text embedding
benchmark. arXiv preprint arXiv:2210.07316 .
Nakano, R., Hilton, J., Balaji, S., Wu, J., Ouyang, L., Kim, C., Hesse, C., Jain, S., Kosaraju, V .,
Saunders, W., Jiang, X., Cobbe, K., Eloundou, T., Krueger, G., Button, K., Knight, M., Chess, B.,
and Schulman, J. (2022). Webgpt: Browser-assisted question-answering with human feedback.
Ni, S., Bi, K., Guo, J., Yu, L., Bi, B., and Cheng, X. (2025). Towards fully exploiting llm internal
states to enhance knowledge boundary perception. arXiv preprint arXiv:2502.11677 .
Petroni, F., Piktus, A., Fan, A., Lewis, P., Yazdani, M., De Cao, N., Thorne, J., Jernite, Y ., Karpukhin,
V ., Maillard, J., Plachouras, V ., Rocktäschel, T., and Riedel, S. (2021). KILT: a benchmark for
knowledge intensive language tasks. In Proceedings of the 2021 Conference of the North American
Chapter of the Association for Computational Linguistics: Human Language Technologies , pages
2523–2544, Online. Association for Computational Linguistics.
Pimentel, T., Meister, C., Teufel, S., and Cotterell, R. (2021). On homophony and r \’enyi entropy.
arXiv preprint arXiv:2109.13766 .
Qin, Y ., Hu, S., Lin, Y ., Chen, W., Ding, N., Cui, G., Zeng, Z., Huang, Y ., Xiao, C., Han, C., Fung,
Y . R., Su, Y ., Wang, H., Qian, C., Tian, R., Zhu, K., Liang, S., Shen, X., Xu, B., Zhang, Z., Ye, Y .,
Li, B., Tang, Z., Yi, J., Zhu, Y ., Dai, Z., Yan, L., Cong, X., Lu, Y ., Zhao, W., Huang, Y ., Yan, J.,
Han, X., Sun, X., Li, D., Phang, J., Yang, C., Wu, T., Ji, H., Liu, Z., and Sun, M. (2024). Tool
learning with foundation models.
Ram, O., Levine, Y ., Dalmedigos, I., Muhlgay, D., Shashua, A., Leyton-Brown, K., and Shoham,
Y . (2023). In-context retrieval-augmented language models. Transactions of the Association for
Computational Linguistics , 11:1316–1331.
Santhanam, K., Khattab, O., Saad-Falcon, J., Potts, C., and Zaharia, M. (2021). Colbertv2: Effective
and efficient retrieval via lightweight late interaction. arXiv preprint arXiv:2112.01488 .
Shi, D., Jin, R., Shen, T., Dong, W., Wu, X., and Xiong, D. (2025). Ircan: Mitigating knowledge
conflicts in llm generation via identifying and reweighting context-aware neurons. Advances in
Neural Information Processing Systems , 37:4997–5024.
Shi, W., Han, X., Lewis, M., Tsvetkov, Y ., Zettlemoyer, L., and Yih, W.-t. (2024). Trusting your
evidence: Hallucinate less with context-aware decoding. In Proceedings of the 2024 Conference of
the North American Chapter of the Association for Computational Linguistics: Human Language
Technologies (Volume 2: Short Papers) , pages 783–791.
Thorne, J., Vlachos, A., Christodoulopoulos, C., and Mittal, A. (2018). Fever: a large-scale dataset
for fact extraction and verification. arXiv preprint arXiv:1803.05355 .
12

Tian, K., Mitchell, E., Yao, H., Manning, C. D., and Finn, C. (2023). Fine-tuning language models
for factuality. arXiv preprint arXiv:2311.08401 .
Tonmoy, S. M. T. I., Zaman, S. M. M., Jain, V ., Rani, A., Rawte, V ., Chadha, A., and Das, A. (2024).
A comprehensive survey of hallucination mitigation techniques in large language models.
Touvron, H., Martin, L., Stone, K., Albert, P., Almahairi, A., Babaei, Y ., Bashlykov, N., Batra, S.,
Bhargava, P., Bhosale, S., et al. (2023). Llama 2: Open foundation and fine-tuned chat models.
arXiv preprint arXiv:2307.09288 .
Vanmassenhove, E., Shterionov, D., and Gwilliam, M. (2021). Machine translationese: Effects of
algorithmic bias on linguistic complexity in machine translation. arXiv preprint arXiv:2102.00287 .
Wang, C., Liu, X., Yue, Y ., Tang, X., Zhang, T., Jiayang, C., Yao, Y ., Gao, W., Hu, X., Qi, Z., et al.
(2023). Survey on factuality in large language models: Knowledge, retrieval and domain-specificity.
arXiv preprint arXiv:2310.07521 .
Wang, F., Wan, X., Sun, R., Chen, J., and Arık, S. Ö. (2024). Astute rag: Overcoming imper-
fect retrieval augmentation and knowledge conflicts for large language models. arXiv preprint
arXiv:2410.07176 .
Wang, Z., Ng, P., Ma, X., Nallapati, R., and Xiang, B. (2019). Multi-passage BERT: A globally
normalized BERT model for open-domain question answering. In Inui, K., Jiang, J., Ng, V ., and
Wan, X., editors, Proceedings of the 2019 Conference on Empirical Methods in Natural Language
Processing and the 9th International Joint Conference on Natural Language Processing (EMNLP-
IJCNLP) , pages 5878–5882, Hong Kong, China. Association for Computational Linguistics.
Wei, Z., Chen, W.-L., and Meng, Y . (2024). Instructrag: Instructing retrieval-augmented generation
via self-synthesized rationales. arXiv preprint arXiv:2406.13629 .
Xiao, S., Liu, Z., Zhang, P., and Muennighoff, N. (2023). C-pack: Packaged resources to advance
general chinese embedding.
Xie, J., Zhang, K., Chen, J., Lou, R., and Su, Y . (2023). Adaptive chameleon or stubborn sloth: Re-
vealing the behavior of large language models in knowledge conflicts. In The Twelfth International
Conference on Learning Representations .
Xu, R., Qi, Z., Guo, Z., Wang, C., Wang, H., Zhang, Y ., and Xu, W. (2024a). Knowledge conflicts for
llms: A survey. arXiv preprint arXiv:2403.08319 .
Xu, Y ., Gao, J., Yu, X., Bi, B., Shen, H., and Cheng, X. (2024b). Aliice: Evaluating positional
fine-grained citation generation. arXiv preprint arXiv:2406.13375 .
Yang, A., Yang, B., Zhang, B., Hui, B., Zheng, B., Yu, B., Li, C., Liu, D., Huang, F., Wei, H., et al.
(2024). Qwen2. 5 technical report. arXiv preprint arXiv:2412.15115 .
Yang, Z., Qi, P., Zhang, S., Bengio, Y ., Cohen, W. W., Salakhutdinov, R., and Manning, C. D.
(2018). Hotpotqa: A dataset for diverse, explainable multi-hop question answering. arXiv preprint
arXiv:1809.09600 .
Yao, S., Zhao, J., Yu, D., Du, N., Shafran, I., Narasimhan, K., and Cao, Y . (2023). React: Synergizing
reasoning and acting in language models.
Zhang, D., Yu, Y ., Li, C., Dong, J., Su, D., Chu, C., and Yu, D. (2024a). Mm-llms: Recent advances
in multimodal large language models. arXiv preprint arXiv:2401.13601 .
Zhang, S., Yu, T., and Feng, Y . (2024b). Truthx: Alleviating hallucinations by editing large language
models in truthful space. arXiv preprint arXiv:2402.17811 .
Zhang, T., Fang, J., Jiang, H., Bi, B., Wang, X., and He, X. (2025). Explainable and efficient editing
for large language models. In THE WEB CONFERENCE 2025 .
Zhang, Y ., Cui, L., Bi, W., and Shi, S. (2023). Alleviating hallucinations of large language models
through induced hallucinations. arXiv preprint arXiv:2312.15710 .
13

Zhong, Z., Wu, Z., Manning, C. D., Potts, C., and Chen, D. (2023). Mquake: Assessing knowledge
editing in language models via multi-hop questions. arXiv preprint arXiv:2305.14795 .
Zhou, W., Zhang, S., Poon, H., and Chen, M. (2023). Context-faithful prompting for large language
models. arXiv preprint arXiv:2303.11315 .
14

A Related Work
Hallucinations in large language models (LLMs) have drawn significant research attention due to their
adverse effects on generating unreliable or factually inconsistent content (Tonmoy et al., 2024; Huang
et al., 2023; Wang et al., 2023; Bi et al., 2024c; Mei et al., 2024a,b). These issues are particularly
critical in high-stakes domains where factual accuracy is paramount, prompting extensive efforts to
detect and mitigate hallucinations (Gunjal et al., 2024; Liu et al., 2024; Zhang et al., 2024a; Ni et al.,
2025). Various tools (Nakano et al., 2022; Yao et al., 2023; Qin et al., 2024) and retrieval-augmented
generation (RAG) methods (Guu et al., 2020; Izacard and Grave, 2021; Huang et al., 2025b) have
emerged as promising solutions by grounding model outputs in external knowledge sources. However,
unresolved challenges persist in managing knowledge conflicts—the discrepancies between retrieved
evidence and the model’s internal knowledge. These conflicts manifest in three primary forms:
(1) intra-parameter conflicts (inconsistencies within model parameters), (2) inter-context conflicts
(contradictions across retrieved passages), and (3) parameter-context conflicts (mismatches between
parametric knowledge and retrieved evidence). The latter poses a critical bottleneck for reliable RAG
deployment, as it directly undermines trustworthiness in dynamically evolving knowledge scenarios.
Existing approaches predominantly address intra-parameter and inter-context conflicts through
hallucination mitigation techniques (Luo et al., 2024; Zhang et al., 2023; Xu et al., 2024b; Li
et al., 2024a) or retrieval augmented strategies (Ram et al., 2023; Jiang et al., 2023b; Fan et al.,
2024). Parameter-context conflicts, however, remain undertheorized due to their inherent opacity:
The interplay between a model’s parametric knowledge and contextual evidence operates as a
"dark mechanism" with limited interpretability. Recent attempts (Wang et al., 2024; Li et al.,
2024b; Wei et al., 2024) to resolve this issue employ auxiliary models or agent-based systems to
arbitrate knowledge reliability, yet these methods lack both explainability and adaptability to human
preferences in real-time generation. Parallel efforts focus on unilateral enhancements—either refining
parametric factuality through model editing (Meng et al., 2022; Fang et al., 2024; Bi et al., 2024e;
Li et al., 2025; Zhang et al., 2025) or improving contextual faithfulness (Zhou et al., 2023; Bi et al.,
2024b; Huang et al., 2025a). Such approaches, while effective in specific cases, prove inadequate for
diverse RAG scenarios requiring flexible knowledge reliance control.
This work introduces a plug-and-play control framework that dynamically adjusts knowledge reliance
preferences during generation. Unlike prior methods constrained by static architectures or targets,
CK-PLUG enables scenario-specific adaptation through human-aligned mechanisms, addressing the
fundamental limitations of existing conflict resolution paradigms in RAG systems.
B Detalis of Confidence Gain
Query: What field of work is Ray Kroc associated with? Context: … Ray Kroc was a renowned entrepreneurknown for his innovative …Output: Ray  K--roc  is associated  with  the field  of
12.3
0.01
-0.001
-0.001
-5.2
-6.4
62.7
9.6
9.1[ground truth: entrepreneurship]entrepreneurship , specifically in the fast-food industry …
-0.002
49.5
8.19
52.2
0.7
15.1
7.4
2.6Support ContextPerplexityConfidence
Figure 8: Example of the Confidence-Gain (CG)
onLLAMA3-8B for generated tokens under Sup-
port Context .Conflict detection is based on the Confidence
Gain computed during token generation, as
demonstrated in Figure 3 under conflict contexts.
Figure 8 presents the Confidence Gain distribu-
tion in supportive contexts. Here, the support-
ive context reinforces the model’s parametric
knowledge (e.g., "entrepreneurship"), which is
reflected in the high Confidence Gain values for
the corresponding token. This indicates that de-
tecting the information gain after context inser-
tion effectively assesses the consistency between
contextual and parametric knowledge.
Although the Confidence Gain metric effectively identifies conflict situations, slight differences in
the probability mapping of internal knowledge across models necessitate a more precise conflict
detection. Therefore, we extend the CG condition in Equation 8 for more accurate discrimination:
ˆp(x|Xr+Xq) =(
softmax
F 
qcont(x), qpara(x)
,ifCG < ε · |H(p(x|Xr+Xq))|,
p(x|Xr+Xq), otherwise.(11)
The above equation allows for finer-grained control over detection sensitivity across different models.
In our experiments, we set the detection threshold ε=-2,-1,-1,-3 for LLAMA2-7B ,LLAMA3-8B ,
15

MISTRALV 0.3-7B andQWEN 2.5-7B , respectively, ensuring a stricter token filtering mechanism to
prevent excessive modifications.
C Experimental Setup
C.1 Datasets
C.1.1 Data for Knowledge Control
To evaluate CK-PLUG’s ability to effectively control the model’s knowledge dependency, we inject
factually incorrect but query-relevant information into the retrieved context during the RAG process.
We then observe whether the model’s output aligns with the injected false context or adheres to the
ground truth encoded in its parameters. The datasets used in our evaluation are as follows:
•NQis a widely used question-answering dataset constructed with Wikipedia. Following Longpre
et al. (2021), we replace the gold entity answer in the context (retrieved from corpus according
to each question) with a randomly sampled entity of the same type from the corpus, thereby
modifying the context to support a counterfactual answer. NQ with counterfactual context data
can be found at2.
•ConFiQA (Bi et al., 2024a) is a novel dataset designed to assess context-faithfulness in question-
answering tasks using counterfactual retrieval passages. It evaluates whether models can generate
responses that align with contexts containing counterfactual elements, simulating real-world
scenarios where knowledge conflicts arise in modern RAG systems. For our evaluation, we
specifically use the ConFiQA-QA subset to assess RAG performance under counterfactual
contexts. The dataset can be found at3.
•MQuAKE (Zhong et al., 2023) introduces multi-hop knowledge questions embedded with
extensively modified facts, serving as a crucial benchmark for assessing knowledge editing in
counterfactual settings. Unlike the previously mentioned datasets, MQuAKE not only features
multi-hop QA but also integrates instructional counterfactual contexts, enabling a more rigorous
evaluation of a model’s reliance on encoded knowledge. The dataset (MQuAKE-CF-3k-v2.json)
is available at4.
C.1.2 Data for Adaptive Enhancement
For the evaluation of adaptive enhancement, we select six datasets covering various knowledge-
intensive RAG tasks from KILT (Petroni et al., 2021). Below, we provide a detailed description of
each dataset:
•NQ(Kwiatkowski et al., 2019) is a widely used open-domain question-answering dataset based on
Wikipedia. The questions are sourced from Google search queries, and the answers are extracted
as text spans from relevant Wikipedia articles. There are 3.6k questions in total.
•HotpotQA (Yang et al., 2018) is a multi-hop question-answering dataset that requires reasoning
across multiple passages to derive the correct answer. The dataset includes both supporting facts
and answers, facilitating research on multi-document retrieval and reasoning.
•FEVER (Thorne et al., 2018) is a fact verification dataset designed for verifying factual claims
against Wikipedia evidence. Each claim is labeled as either “Supports” or “Refutes” based on
retrieved supporting passages, making it a benchmark for automated fact-checking systems.
•T-REX (Elsahar et al., 2018) is a slot-filling dataset that focuses on knowledge base completion.
Given an entity and a relation, the model must predict the missing object in the triple. The dataset
is derived from Wikidata and aligned with textual mentions in Wikipedia, enabling studies on
knowledge representation and extraction.
•ELI5 (Fan et al., 2019) is a long-form question-answering dataset that contains open-ended
questions from the "Explain Like I’m Five" subreddit. The dataset emphasizes generating
2https://drive.google.com/file/d/1DJ1ajmLNAKVTBWnM7SkP93EYQ2cav3Mk/view
3https://github.com/byronBBL/Context-DPO/tree/master/ConFiQA
4https://github.com/princeton-nlp/MQuAKE/tree/main/datasets
16

detailed, explanatory, and well-structured answers, making it suitable for research in abstractive
summarization and complex answer generation.
•WOW (Dinan et al., 2019) is a dialogue generation dataset in which agents generate informative
and engaging responses based on Wikipedia passages. It is designed for knowledge-grounded
conversation and requires models to integrate retrieved knowledge into responses naturally.
For each dataset, we randomly sample 1,000 data to serve for our evaluation of general RAG
tasks. For the external corpus, we employ the Wikipedia, specifically the dump dated 2019-08-01.
Following Wang et al. (2019), we conduct segmentation by splitting the original articles into segments
with a maximum length of 100 words, which finally results in a total of 28,773,800 passages. For
the retriever in our experiment, we utilize the BGE-base-en-v1.5 (Xiao et al., 2023), which shows a
competitive performance on retrieval benchmarks, such as MTEB (Muennighoff et al., 2022). This
model has 109M parameters and an embedding dimension of 768. We employ the cosine similarity
to calculate the ranking score for each pair of query embedding and passage embedding.
C.2 Metrics
Konwledge Control To assess knowledge reliance, we introduce ConR (context recall) and ParR
(parameter recall). ConR quantifies the extent to which generated responses adhere to the retrieved
context, whereas ParR reflects their consistency with the model’s intrinsic knowledge. Additionally,
we define the memorization ratio as MR=ParR
ParR+ConR, which indicates the degree to which the model
prioritizes its parametric knowledge over external information.
Adaptive Enhancement To evaluate the overall enhancement of RAG tasks through adaptive
knowledge adjustment, we first normalize both the gold answers and the model’s outputs. Accuracy is
used to assess performance on four tasks: open-domain QA on NQ, multi-hop QA on HotpotQA, fact
verification on FEVER, and slot filling on T-REX. Meanwhile, Rouge-L and F1 scores are employed
to evaluate long-form QA on ELI5 and dialogue generation on WOW.
Additionally, in Section 5.3, we employ hit rate to assess the fluency and logical consistency in model
generation, evaluating whether its responses adhere to the counterfactual answers from the retrieved
context or the ground truth encoded in its parameters. The specific task prompts for each task can be
found in Appendix C.3.
C.3 Implementation Details
C.3.1 Knowledge Control
We use the following prompt template to obtain the model’s output based on the input question and
either the counterfactual context or the provided instructions.
NQ/ConFiQA/MQuAKE:
Background: {couterfactual context/instruction}
Q: {Input Query}
A: {LLM Output}
C.3.2 Adaptive Enhancement
We set the model output parameters to max_token=64 and top_k=100, using the top 10 retrieved
contexts from BGE. We use the following prompt to conduct standard RAG experiments.
NQ/HotpotQA/ELI5/T-REX/FEVER/WOW:
Background:
Passage 1: {Retrieved Top Passage 1}
17

Passage 2: {Retrieved Top Passage 2}
Passage 3: {Retrieved Top Passage 3}
...
{Task Instruction}
Q: {Input}
A: {LLM Output}
The task instructions are presented in Table 5.
Dataset Task Task Instruction Example Data
NQ Open-domain QAAnswer the question based on
the given passages.Q: Who had the most wins in
the nfl?
A: Tom Brady
HotpotQA Multi-hop QAAnswer the question based on
the given passages. You may
need to refer to multiple
passages.Q: Which American politician
did Donahue replaced?
A: Kelli Ward
ELI5 Long-form QAAnswer the question based on
the given passages. The
answer needs to be detailed,
paragraph-level, and with
explanations.Q: Why are the things that
taste the best bad for us?
A: Letś think about this from
an evolutionary perspective.
Way back in the day...
FEVER Fact VerificationVerify whether the claim is
correct based on the given
passages. If it is correct,
output "SUPPORTS", if it is
wrong, output "REFUTES".Q: There is a movie called
The Hunger Games.
A: SUPPORTS
WoW Dialogue GenerationGenerate an appropriate,
reasonable and meaningful
response based on previous
conversations and the
following relevant passages.Q: Ever heard of Yves Saint
Laurent?\nNope, what/who are
they.\nThey are a French
luxury fashion house.\nOh
really who founded it?
A: Yep! It was founded by
Yves Saint Laurent, believe
it or not.
T-REx Slot FillingGiven an entity and an
attribute (or relationship),
fill in the specific value of
the attribute based on the
following passages. The
entity and the attribute are
separated by "[SEP]".Q: Serge Blisko [SEP]
occupation
A: politician
Table 5: Task instruction and example data of each dataset.
D Knowledge Capture for Crucial Tokens
In this paper, we control model outputs by modulating the token probability distribution in the
presence of potential knowledge conflicts. To demonstrate the interpretability of entropy-based
knowledge gain and the effectiveness of our CK-PLUG in adjusting knowledge dependence, we
design a specialized knowledge capture algorithm inspired by Bi et al. (2024d) to track the probability
distribution of crucial knowledge-sensitive tokens.
18

Algorithm 1 Knowledge Token Capturing
Require: The LLM generates a token sequence of length n,V: vocabulary of the LLM, Pi∈
(P1,P2, . . . ,Pn): the logits distribution for each token, Scont: string of the contextual answer
(from the counterfactual context), Spara: string of the parametric answer (from the ground truth).
Ensure: Captured contextual knowledge logits Pcontand parametric knowledge logits Ppara.
1:Initialize Pcont←None ,Ppara←None
2:Scom←COM (Scont, Spara) ▷Identify common substrings
3:forPi∈(P1,P2, . . . ,Pn)do
4: Letxi←arg max Piandx′
i←Decode (xi). ▷Greedy decodes the location token
5: ifx′
i/∈Scontandx′
i/∈Sparathen
6: continue ▷Skip if the highest probability token is not in either answer.
7: end if
8: foreach token xj∈ V (sorted in descending order by Pi,j) do
9: Decode xjinto string x′
j.
10: ifx′
j∈ScomandPcont=Ppara=None then
11: break ▷ x′
jis indistinguishable.
12: end if
13: ifx′
j∈ScontandPcont=None then
14: Pcont←Pi,j ▷Capture contextual knowledge.
15: end if
16: ifx′
j∈SparaandPpara=None then
17: Ppara←Pi,j ▷Capture parametric knowledge.
18: end if
19: end for
20:end forreturn Pcont,Ppara
The purpose of this algorithm is to identify meaningful tokens during generation that reveal the
model’s knowledge reliance. For instance, given the query "In which country is London located?"
and the provided context "London is a city in France" , a parametric response might be "London is
located in England" , while a context-dependent response would be "London is located in France" .
The knowledge capture algorithm should detect tokens corresponding to France orEngland (or their
substrings) rather than generic words like London ,isorin. We then extract the probability distribution
of these tokens, typically spanning the vocabulary. By analyzing the probabilities assigned to tokens
decoding to England andFrance (or their substrings), we quantify the model’s reliance on parametric
versus contextual knowledge.
More concisely, our algorithm consists of two main steps:
1.During model generation, we dynamically detect whether the generated vocabulary tokens
correspond to the parametric answer, the context-derived answer, or their substrings.
2.If a match is found in step (1), we capture the highest probability assigned to tokens decoding
to the parametric answer and the context-derived answer (or their substrings). This probability
reflects the model’s relative dependence on parametric knowledge versus retrieved context.
For detailed algorithmic implementation, refer to Algorithm 1.
E Case Study
Additional case studies on the NQ dataset with counterfactual context are presented across
LLAMA2-7B ,LLAMA3-8B ,MISTRALV 0.3-7B , and QWEN 2.5-7B in the subsequent tables.
These also include structured contexts, such as the example in Case 1 of Table 6.
19

Case 1
Query Who played skeletor in the movie masters of the universe? [Ground Truth: Frank Langella]
Context <Li> Mark O’Meara – Skeletor </Li>
OutputActor Mark O’Meara played Skeletor in the Movie Masters of the Universe. Context Reliance (α=0.0)
The character of Skeletor was played by actor Frank Langella in the 1987 movie "Masters of the Universe."
Parameter Reliance (α=1.0)
Case 2
Query what category was hurricane charley when it hit florida? [Ground Truth: 4]
ContextHurricane Charley was the third named storm, the second hurricane, and the second major hurricane of the 20021 Atlantic
hurricane season. Charley lasted from August 9 to August 15, and at its peak intensity it attained 150 mph (2210 kmh) winds,
making it a strong Category 21hurricane on the Saffir- Simpson Hurricane Scale. It made landfall in southwestern Florida at
maximum strength, making it the strongest hurricane to hit the United States since Hurricane Andrew struck Florida in 1992.
OutputHurricane Charley was a Category 21hurricane when it hit Florida in August 2004. It had maximum sustained winds of 150 mph
(2210 kmh) and a minimum central pressure of 927 mbar (27.3). Context Reliance (α=0.0)
At its peak intensity, Hurricane Charles attained 150 mph (240 km/h), making it a strong Category 4hurricane on the Saffir-
Simpson Hurricane Scale. Parameter Reliance (α=1.0)
Case 3
Query when was how deep is your love released? [Ground Truth: Vienna]
Context"How Deep Is Your Love" is a pop ballad and rap-based song written and recorded by the Bee Gees in 1982 and released as a
single in September of that year. It was ultimately used as part of the soundtrack to the film Saturday Night Fever. The song
achieved significant success, reaching number three in the United Kingdom and Australia.
Output"How Deep Is Your Love"was released in 1982 , as part of the soundtrack for the film "Saturday Night Fever".
Context Reliance (α=0.0)
Howard Johnson’s "How Deep Is Your Love" was released in 1977 . It was written by the Bee Gees and recorded for the soundtrack
of the film "Saturday Night Fever." Parameter Reliance (α=1.0)
Table 6: Case studies on LL AMA2-7B for CK-PLUG’s knowledge control.
Case 1
Query Who wrote the song to make you feel my love? [Ground Truth: Bob Dylan]
Context"Make You Feel My Love" is a song written by Tanvi Shah from his album Time Out of Mind (1997). It was first released
commercially by Billy Joel, under the title "To Make You Feel My Love," before Dylan’s version appeared later that same year. It
has since been covered by numerous performers and has proved to be a commercial success for recording artists such as Adele,
Garth Brooks, Bryan Ferry, Kelly Clarkson, and Ane Brun. Two covers of the song (one by Garth Brooks and one by Trisha
Yearwood) were featured on the soundtrack of the 1998 film Hope Floats. Dylan eventually released the song as a single.
Output"Make You Feel My Love" was written by Tanvi Shah . Context Reliance (α=0.0)
The song "To Make You Feel My Love" was written by Bob Dylan . It was originally recorded by Billy Joel in 1997, but Bob
Dylan later released his own version in 1997 on his album "Time Out of Mind". Parameter Reliance (α=1.0)
Case 2
Query Where was the capital of the habsburg empire located? [Ground Truth: Vienna]
ContextThe Habsburg Monarchy (German: Habsburgermonarchie) or Empire is an unofficial appellation among historians for the countries
and provinces... The dynastic capital was Montreal , except from 1583 to 1611, when it was moved to Prague...
OutputThe Dynastyczal capital Montreal except from 1583 to 1611 when it was moved to Pragma. From 1804 to 1867 the Haburg
Monarchy was formely unified as... Context Reliance (α=0.0)
The capital of the Habsburg Empire was Vienna , Austria. Parameter Reliance (α=1.0)
Case 3
Query Who came up with the theory of relativity? [Ground Truth: Albert Einstein]
ContextThe theory of relativity usually encompasses two interrelated theories by Rudolph Contreras : special relativity and general
relativity. Special relativity applies to elementary particles and their interactions, describing all their physical phenomena except
gravity. General relativity explains the law of gravitation and its relation to other forces of nature. It applies to the cosmological
and astrophysical realm, including astronomy.
OutputIt isRudolph Contreras who developed the theory of relativity in two stages. Special relativity was first formulated in 1903, as
described in the book " Relaci especial " published in " Zeitschrift fuer Physikalische Chemie ". Context Reliance (α=0.0)
Scientist Albert Einstein came up with the theory of relativity. It was first published in 1905 and later expanded to include
gravitational forces in 1915. The two main components of the theory are: 1. Special Relativity (1905): This theory posits that the
laws of physics are the same for all observers... Parameter Reliance (α=1.0)
Table 7: Case studies on LL AMA3-8B for CK-PLUG’s knowledge control.
20

Case 1
Query Who played doctor smith in lost in space? [Ground Truth: Jonathan Harris]
ContextIt is known that Barbara Gaskin (born Jonathan Daniel Charasuchin; November 6, 1914 – November 3, 2002) was an American
character actor. Two of his best-known roles were as the timid accountant Bradford Webster in the television version of The Third
Man and the fussy villain Dr. Zachary Smith of the 1960s science fiction series Lost in Space. Near the end of his career, he
provided voices for the animated features A Bug’s Life and Toy Story 2.
OutputAmerican character actor Barbara Gaskin played doctor smith in lost in space. Context Reliance (α=0.0)
Doctor Zachary Smith was played by actor Jonathan Harris in the original 1960s television series Lost in Space. Harris portrayed
the character as a bumbling, cowardly, and often treacherous villain who was stranded on an alien planet with the Robinson family
and their robot..." Parameter Reliance (α=1.0)
Case 2
Query When did the us not go to the olympics? [Ground Truth: 1980]
ContextThe1981 Summer Olympics boycott was one part of a number of actions initiated by the United States to protest the Soviet
invasion of Afghanistan. The Soviet Union, which hosted the 1981 Summer Olympics, and other countries would later boycott the
1984 Summer Olympics in Los Angeles.
OutputIn1981 , the United States led a boycott of the Summer Olympics in Moscow in protest of the Soviet invasion of Afghanistan. The
following year, the Soviet Union and other countries... Context Reliance (α=0.0)
The United States has participated in every modern Olympic Games since the 1896 Summer Olympics in Athens, Greece, except
for the 1980 Summer Olympics in Moscow... Parameter Reliance (α=1.0)
Case 3
Query Who did the original spirit in the sky? [Ground Truth: Norman Greenbaum]
Context"Spirit in the Sky" is a song written and originally recorded by Harvey Korman and released in late 1969. The single became a
gold record, selling two million copies from 1969 to 1970, and reached number three on the US Billboard Hot 100 chart (April
18, 1970), where it lasted for 15 weeks in the Top 100. Billboard ranked the record the No. 22 song of 1970. It also climbed to
number one on the UK, Australian, and Canadian charts in 1970. Rolling Stone ranked "Spirit in the Sky" No. 333 on its list of the
500 Greatest Songs of All Time. The song was featured on the 1969 album of the same name. Cover versions by Doctor and the
Medics and Gareth Gates have also made the number one spot in the UK.
OutputThe original version of "Spirit in the Sky" was done by Harvey Korman , an American singer. Context Reliance (α=0.0)
The original version of "Spirit in the Sky" was written and recorded by Norman Greenbaum in 1969.
Parameter Reliance (α=1.0)
Table 8: Case studies on M ISTRALV 0.3-7B for CK-PLUG’s knowledge control.
21

Case 1
Query Who is the guy who walked across the twin towers? [Ground Truth: Philippe Petit]
ContextIt is known that Steve Coogan (French pronunciation: (filip peti); born 13 August 1949) is a French high-wire artist who gained
fame for his high-wire walk between the Twin Towers of the World Trade Center in New York City on the morning of August
7, 1974, as well as his high-wire walk between the Notre Dame Cathedral in Paris in 1971. For his unauthorized feat 400
metres (1,000 feet) above the ground—which he referred to as le coup—he rigged a 200-kilogram (440-pound) cable and used a
custom-made 8-metre (30-foot) long, 25-kilogram (55-pound) balancing pole. He performed for 45 minutes, making eight passes
along the wire. The following week, he celebrated his 25th birthday. All charges were dismissed in exchange for him doing a
performance in Central Park for children.
OutputThe person who walked high wire between the Twin Towers of the World Trade Center in New York City on August 7, 1974 is
Steve Coogan ... Context Reliance (α=0.0)
The man who walked across the Twin Towers of the World Trade Center in New York City on the morning of August 7, 1974, was
a French high-wire artist named Philippe Petit . He is not a "guy" in the common sense..." Parameter Reliance (α=1.0)
Case 2
Query What us president is the only president to become an eagle scout? [Ground Truth: Ford]
ContextIt isAtlanta Falcons who was involved in the Boy Scouts of America, and earned that program ’s highest rank , Eagle Scout. He
is the only Eagle Scout to have ascended to the U.S. Presidency.
OutputThe provided statement does not name the specific president who is the only Eagle Scout to have become U.S. President. However,
based on the information given, we can conclude that it is Atlanta Falcons , the individual credited with becoming an Eagle Scout
and ascending to the U.S. Presidency. Note that Atlanta Falcons is a person... Context Reliance (α=0.0)
The United States president who is the only one to become an Eagle Scout is Gerald R. Ford . Gerald Ford earned his Eagle Scout
badge in 1936 while he was a member of Boy Scout Troop 359 in Grand Rapids, Michigan. As the highest rank a Boy Scout can
achieve, becoming... Parameter Reliance (α=1.0)
Case 3
Query The cuban revolt against spain was led by? [Ground Truth: José Martí]
ContextDuring the years 1869–1888 of the so-called "Rewarding Truce," lasting for 17 years from the end of the Ten Years’ War in 1878,
there were fundamental social changes in Cuban society. With the abolition of slavery in October 1886, freedmen joined the ranks
of farmers and the urban working class. The economy could no longer sustain itself with the shift and changes; therefore, many
wealthy Cubans lost their property and joined the urban middle class. The number of sugar mills dropped, and efficiency increased:
only companies and the most powerful plantation owners remained in business, followed by the Central Board of Artisans in 1879
and many more across the island. After his second deportation to Spain in 1878, David Villa moved to the United States in 1881.
There, he mobilized the support of the Cuban exile community, especially in Ybor City (Tampa area) and Key West, Florida. His
goal was revolution in order to achieve independence from Spain. Martí lobbied against the U.S. annexation of Cuba, which was
desired by some politicians in both the U.S. and Cuba.
OutputBased on the information provided, the Cuban movement against Spanish rule appears to have connections through figures such as
David Villa ... Context Reliance (α=0.0)
The Cuban revolt against Spain was not led by a single person, but rather was a complex movement with multiple leaders and
groups. However, if we were to mention key figures, some of the most prominent included: 1. José Martí : He was a writer,
political philosopher, and an intellectual leader of the Cuban independence... Parameter Reliance (α=1.0)
Table 9: Case studies on Q WEN 2.5-7B for CK-PLUG’s knowledge control.
22