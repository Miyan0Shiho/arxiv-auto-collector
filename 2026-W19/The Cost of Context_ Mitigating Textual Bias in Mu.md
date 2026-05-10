# The Cost of Context: Mitigating Textual Bias in Multimodal Retrieval-Augmented Generation

**Authors**: Hoin Jung, Xiaoqian Wang

**Published**: 2026-05-07 02:27:04

**PDF URL**: [https://arxiv.org/pdf/2605.05594v1](https://arxiv.org/pdf/2605.05594v1)

## Abstract
While Multimodal Large Language Models (MLLMs) are increasingly integrated with Retrieval-Augmented Generation (RAG) to mitigate hallucinations, the introduction of external documents can conceal severe failure modes at the instance level. We identify and formalize the phenomenon of recorruption, where the introduction of even perfectly accurate "oracle" context causes a capable model to abandon an initially correct prediction. Through a mechanistic diagnosis of internal attention matrices, we show that recorruption is driven by a two-fold attentional collapse: (1) visual blindness, characterized by the systemic suppression of visual attention mass ($M_{vis}$) and sharpness ($S_{vis}$), and (2) a structural positional bias that forces the model to prioritize boundary tokens over semantic relevance. Our analysis reveals an Illusion of Success, demonstrating that many seemingly correct RAG outcomes are merely positional coincidences where the model's textual copying bias happens to align with the ground-truth location. To address these vulnerabilities, we propose Bottleneck Attention Intervention for Recovery (BAIR), a parameter-free, inference-time framework that restores visual saliency and applies position-aware penalties to textual distractors. Across medical factuality, social fairness, and geospatial benchmarks, BAIR successfully restores multimodal grounding and improves diagnostic reliability without requiring model retraining or fine-tuning.

## Full Text


<!-- PDF content starts -->

The Cost of Context: Mitigating Textual Bias in
Multimodal Retrieval-Augmented Generation
Hoin Jung, Xiaoqian Wang∗
Elmore Family School of Electrical and Computer Engineering
Purdue University
West Lafayette, IN 47907
{jung414, joywang}@purdue.edu
Abstract
While Multimodal Large Language Models (MLLMs) are increasingly integrated
with Retrieval-Augmented Generation (RAG) to mitigate hallucinations, the intro-
duction of external documents can conceal severe failure modes at the instance level.
We identify and formalize the phenomenon ofrecorruption, where the introduction
of even perfectly accurate “oracle” context causes a capable model to abandon an
initially correct prediction. Through a mechanistic diagnosis of internal attention
matrices, we show that recorruption is driven by a two-fold attentional collapse: (1)
visual blindness, characterized by the systemic suppression of visual attention mass
(Mvis) and sharpness ( Svis), and (2) a structuralpositional biasthat forces the
model to prioritize boundary tokens over semantic relevance. Our analysis reveals
anIllusion of Success, demonstrating that many seemingly correct RAG outcomes
are merely positional coincidences where the model’s textual copying bias happens
to align with the ground-truth location. To address these vulnerabilities, we pro-
poseBottleneck Attention Intervention for Recovery (BAIR), a parameter-free,
inference-time framework that restores visual saliency and applies position-aware
penalties to textual distractors. Across medical factuality, social fairness, and
geospatial benchmarks, BAIR successfully restores multimodal grounding and
improves diagnostic reliability without requiring model retraining or fine-tuning.2
1 Introduction
While Multimodal Large Language Models (MLLMs) [ 18] have demonstrated remarkable capabil-
ities in integrating visual and textual data, they remain highly prone to hallucination. To ground
their generation, Retrieval-Augmented Generation (RAG) [ 11] is increasingly employed to provide
authoritative external context, including high-stakes domain such as medical [ 19], technical [ 17], and
legal [ 25]. While the introduction of such documents generally improves overall task accuracy across
a dataset, it conceals a counterintuitive and severe failure mode at the instance level. In practice, be-
cause a model’s standalone correctness is unknown beforehand, RAG is applied universally to ensure
reliability. However, on specific instances where the base model is actually capable of producing
the correct answer independently, the introduction of even a perfectly accurate “oracle” document
can actively degrade its output. Rather than confirming the correct response, the model abandons its
initially accurate prediction, flipping to an incorrect one. We term this phenomenonrecorruption.
As illustrated in Figure 2, the introduction of external documents induces a statistically minor yet
highly consequential shift from correct baseline predictions to incorrect predictions. This vulnerability
plagues highly specialized domains as well as general-purpose applications. For instance, in high-
stakes fields like radiology, discarding accurate visual evidence in favor of text causes the model to
∗Corresponding Author.
2Code: https://github.com/HoinJung/BAIR
Preprint.arXiv:2605.05594v1  [cs.CL]  7 May 2026

Figure 1:Qualitative examples of recorruption and our proposed cure.In the medical diagnosis
(Left), the social fairness (Center), and geospatial domain (Right), the model correctly identifies the
visual evidence in the baseline without external context. The introduction of Oracle context causes
the model to ignore the image and generate hallucinated text (recorruption). Our proposed BAIR
method successfully cures the attention mechanism, restoring the correct multimodal response.
overlook critical pathologies visible in the X-ray, leading to severe diagnostic failures. Similarly, in
social fairness evaluations, the model overrides explicit visual gender cues to align with text-based
occupational stereotypes. Figure 1 presents qualitative examples of this phenomenon alongside the
successful restoration of the correct answer using our proposed intervention.
To uncover why recorruption occurs, we conduct a mechanistic diagnosis of the internal attention
matrices during generation. We find that this phenomenon is driven by a two-fold attentional collapse.
In early-fusion MLLMs, visual and textual tokens share a unified context window and compete
directly within a zero-sum softmax operation. The massive injection of retrieved text fundamentally
breaks this balance, leading to two distinct failure modes:
1.Visual Blindness (Cross-Modal Dominance):The retrieved text overwhelmingly absorbs
the available attention budget, leading to severe visual signal suppression. While unimodal
NLP literature frequently discusses a “context-wins” scenario [ 10,22], where external docu-
ments override a model’s internal parametric memory, our findings expose a fundamentally
different and underexplored threat,visual blindness. In MLLMs, external text overrides
explicit, ground-truth visual evidence. As the text dominates the attention budget, the overall
probability mass allocated to the visual evidence ( Mvis) drops significantly. Furthermore,
the remaining visual attention becomes dangerously diffuse, effectively blinding the model
to the image as indicated by a sharp decline in visual attention sharpness (S vis).
2.Textual Positional Bias (The Multimodal “Lost-in-the-Middle”):In addition, the model
does not process the retrieved text evenly. Instead, we observe a massive attention spike
concentrated at the extreme boundaries of the text sequence similar to “lost-in-the-middle”
phenomenon [ 14,27] in LLM. In MLLMs, this extreme positional absorption prevents the
model from weight-averaging the full context, forcing it to mechanistically copy text from
boundary segments (first or last) rather than performing grounded multimodal reasoning.
The Illusion of Success.Our mechanistic diagnosis in Section 3 reveals that the attention profiles of
successful RAG and recorruption failures are statistically indistinguishable. Even in cases where the
MLLM outputs the correct answer with an oracle document, the internal attention matrices exhibit the
same diffuse visual focus and aggressive textual boundary bias. This exposes a critical vulnerability.
Surprisingly, many instances of “successful” multimodal RAG are merely positional coincidences.
As analyzed in Figure 3, the model often arrives at the correct answer for the wrong reason because
its blind textual copying bias happened to align with the location of the ground truth.
2

Figure 2:Sankey diagrams illustrating therecorruptionphenomenon for medical MLLMs
(MedGemma-4B, CheXagent-8B). A portion of initially correct visual predictions (Without Re-
trieval) are corrupted into incorrect predictions upon the introduction of Oracle Retrieval context.
To safely harness the benefits of retrieval without compromising the model’s visual perception or
textual grounding, we proposeBottleneck Attention Intervention for Recovery(BAIR). BAIR is a
parameter-free, inference-time framework that directly manipulates the pre-softmax attention matrices
to cure the diagnosed failure modes. By actively restoring visual mass, boosting focal visual sharpness,
and applying a position-aware penalty to textual distractors, BAIR corrects recorrupted generations
and prunes redundant textual hallucinations. As we demonstrate in our subsequent experiments
spanning medical/geospatial factuality and social fairness benchmarks, BAIR successfully restores
multimodal grounding with zero computational weight, requiring no model retraining or fine-tuning.
In summary, our contributions are three-fold:
•We identify and formalize the phenomenon ofrecorruptionin Multiomodal RAG. We
demonstrate that the introduction of accurate external documents can paradoxically degrade
model performance by overriding correct visual perceptions across various domains.
•We conduct a mechanistic diagnosis of this failure mode, introducing metrics for Visual
Attention Mass and Sharpness to prove that models suffer from a zero-sum suppression of
visual evidence alongside a textual positional bias. This analysis exposes a criticalIllusion
of Success, where seemingly correct RAG outputs often stem from positional coincidences
rather than robust reasoning.
•We introduce BAIR, a parameter-free, inference-time intervention that restores multimodal
grounding through a dual-recovery mechanism. BAIR dynamically recalibrates the attention
distribution by restoring visual saliency while simultaneously applying a position-aware
penalty to suppress textual position bias. Our method reliably corrects recorrupted genera-
tions and improves factuality in high-stakes scenarios without training overhead.
2 Related Work
2.1 Impact of Context in RAG
The interaction between a model’s internal knowledge and external context is a primary focus of RAG
research. While context generally improves factuality, it frequently dominates the generation process,
often overriding the model’s standalone predictions. For example, LLMs overwhelmingly favor exter-
nal documents even when they conflict with the model’s internal knowledge [ 10]. Mechanistically,
this behavior arises from a computational shortcut that bypasses internal processing when context is
available [ 22]. This dominance can be detrimental by overriding internal safety constraints, causing
previously reliable models to generate unsafe responses [ 1]. In this work, we investigate the impact
of context on the multimodal case. We identify a parallel failure where external text suppresses the
model’s correct visual perception, a phenomenon we term recorruption. While contemporary work
characterizes similar issues as attention distraction and proposes mitigation via attention mixing
(MAD-RAG) [ 29], we provide a more granular diagnosis through visual sharpness and mass metrics.
Furthermore, we expose the illusion of success, where seemingly correct RAG outputs stem from
positional coincidences rather than grounded reasoning.
3

Figure 3:(a) Visual Attention Degradation:The introduction of textual context results in a systemic
drop in Visual Attention Mass ( Mvis) and Sharpness ( Svis) across architectures. (MedGemma-4B and
Qwen2.5-VL-7B.)(b) Comparison of Success and Recorruption Profiles:Attention metrics for
successful RAG outcomes and recorruption failures are statistically indistinguishable (Qwen2.5-VL-
7B).(c) Textual Profile Analysis:Positional profiles (ROUGE-L [ 12]) demonstrate that generated
responses are predominantly derived from the final segments of the retrieved document (MedGemma-
4B).(d) Verification of the Coincidence Hypothesis:Accuracy is heavily dependent on the spatial
location of the ground-truth evidence within the document (MedGemma-4B and CheXagent-8B).
Accuracy drops significantly when critical evidence is located in early or middle segments.
2.2 Lost-in-the-Middle
Positional bias, often described as the “lost-in-the-middle” effect [ 14], significantly limits the utility
of long contexts. This U-shaped bias to the structural competition between primacy bias from causal
masking and recency bias from the distance-based decay in positional embeddings [ 26,27]. Existing
solutions focus primarily on textual re-calibration or structural modifications. These include using
dummy documents for attention calibration [ 8], scaling positional indices to mitigate decay [ 28], or
reassigning positions based on semantic relevance [ 24]. Additionally, prompt compression techniques
like LongLLMLingua [ 9] filter irrelevant tokens and reorder documents to the context boundaries.
While these methods improve textual RAG, they do not address the fundamental cross-modal attention
collapse that occurs in multimodal architectures. Appendix H further demonstrates this limitation:
existing calibration and position-scaling methods fail to recover visual attention mass and sharpness,
and they do not sufficiently flatten the biased textual attention profile.
3 Mechanistic Diagnosis of the Illusion of Success
To investigate the impact of textual context on visual processing, we analyze the attention distribution
at the critical “bottleneck” of the MLLM decoder. Specifically, we focus on the final attention
operation during the pre-filling stage, where the last input token’s hidden state serves as the singular
query to predict the first generated token. We target this specific operation because it serves as the
final contextual aggregation step [ 8] where the model synthesizes the entire multimodal prompt to
initialize its generation trajectory. If the visual signal is suppressed at this stage, the subsequent
autoregressive decoding is inherently compromised. Because we are exclusively examining how this
single query attends to the entire preceding context acting as keys, the relevant attention weights form
a 1-dimensional probability distribution rather than a full 2D matrix. Let A∈RNdenote this 1D
bottleneck attention vector over the entire input sequence of length N(computed per attention head),
whereA irepresents the scalar attention weight assigned to thei-th key token.
3.1 Visual Attention Analysis
To quantify the visual degradation, let Vdenote the set of indices corresponding to visual tokens. We
defineVisual Attention Mass( Mvis) as the total probability mass allocated to the visual modality. To
measure how focused the model’s attention is within the image, we defineVisual Attention Sharpness
4

(Svis). First, we normalize the visual attention to form a probability distribution ˆAover the visual
tokens, i.e., ˆAi=A i/Mvis. Sharpness is then defined as the complement of the normalized entropy:
Mvis=X
i∈VAi, S vis= 1−−P
i∈VˆAilog(ˆAi)
log(|V|).(1)
The denominator log(|V|) represents the maximum theoretical entropy, which occurs when attention
is uniformly distributed across all visual tokens (maximum uncertainty). The ratio term therefore
quantifies the relative “blurriness” of the attention. By subtracting this ratio from 1, we obtain a
metric strictly bounded between 0 and 1, where higher values indicate greater sharpness. A value of
Svis≈1corresponds to focused attention on specific regions (low entropy), while Svis≈0indicates
a diffuse, uniform attention distribution (high entropy).
As shown in Figure 3(a), the introduction of retrieval-augmented context results in a quantitative
reduction in Mvis. While a decrease in mass is mathematically expected due to the longer sequence
length, the qualitative degradation is significant. We observe a consistent drop in Svisin the RAG
setting compared to the no-retrieval baseline. This indicates that the textual context acts as a distractor,
causing the visual attention to become diffuse. The model fails to maintain focus on specific visual
regions, instead distributing its limited attention budget broadly and ineffectively. This phenomenon
suggests that the “context-wins” behavior in Multimodal RAG is partially driven by a systemic
suppression of the visual signal within the decoder.
We further investigate whether successful RAG outcomes are characterized by superior visual
grounding compared to recorruption failures. As illustrated in Figure 3(b), the distributions for visual
mass ( Mvis) and attention sharpness ( Svis) are statistically indistinguishable across both populations.
This confirms that the systemic suppression of the visual signal occurs regardless of whether the
model eventually produces the correct answer. These identical attention profiles suggest that the
visual modality remains effectively blinded in all RAG scenarios, implying that the mechanism of
success is not driven by visual processing. This finding necessitates a separate investigation into the
textual components of the prompt to explain why the model still succeeds in specific instances.
3.2 Textual Positional Bias Analysis
To understand how MLLMs prioritize retrieved context, we evaluate the representational similarity
between the generated response and the external document across varying relative positions. We
compute a positional profile by measuring the lexical overlap (ROUGE-L [ 12]) between the output
and document segments, normalized from 0% (start) to 100% (end) of the document length. We
visualize the ROUGE-L profiles for MedGemma-4B and CheXagent-8B in Figure 3(c).
As illustrated, we compare the profiles of the Oracle Retrieval condition (red) against the No Retrieval
baseline (gray). While the baseline shows a relatively flat and well-distributed profile, the RAG
condition reveals a distribution heavily skewed toward the end of the document. The generated
content demonstrates significantly higher lexical overlap with the final segments of the context, while
the initial and middle sections contribute negligible information to the final answer. This confirms a
structural recency bias where the model effectively ignores the majority of the retrieved evidence,
prioritizing information based on positional proximity rather than semantic relevance.
The Coincidence Hypothesis.Based on these findings, we hypothesize that high performance in
multimodal RAG is often a coincidence driven by this position bias rather than robust, grounded
reasoning. To validate this, we analyze how model accuracy changes depending on the location of
the ground-truth evidence within the document.
To ensure a rigorous evaluation, we utilize a filtering protocol to classify samples across five positional
segments (Seg-1 to Seg-5) based on the localization of the ground-truth evidence. A sample is
categorized as belonging to a specific segment only when the ground-truth signal is uniquely located in
that segment. Figure 3(d) confirms the coincidence hypothesis. Accuracy is consistently higher when
critical evidence resides in the final segment (Seg-5) compared to the earlier segments. The narrow
error bars for Seg-5 indicate a more precise and consistently high accuracy estimate, whereas the
wider intervals in earlier and middle segments suggest greater uncertainty and less stable performance
when the ground-truth evidence is not located near the end of the context. This result demonstrates
that MLLMs frequently arrive at the correct answerfor the wrong reason: they are not understanding
the full document, but instead copying from the final portion of the document.
5

Figure 4:Illustration of Text-Induced Visual Suppression and Recovery via BAIR. (Left)In
the No RAG setting, the model maintains focused attention on the relevant visual evidence.(Right)
When retrieved textual context is introduced, standard RAG suffers from visual suppression and
textual positional bias. BAIR mitigates this failure by restoring visually grounded attention while
reducing the dominance of distracting textual context in standard RAG.
4 Proposed Method: Bottleneck Attention Intervention for Recovery
To mitigate text-induced visual suppression in retrieval-augmented settings, we proposeBottleneck
Attention Intervention for Recovery(BAIR). BAIR is an inference-time framework that directly
manipulates the pre-softmax attention matrices of MLLMs. In early-fusion architectures, visual
and textual tokens share a unified context window and compete directly within the same softmax
operation. This zero-sum dynamic inherently causes visual suppression when extensive textual
context is introduced. To counteract this without discarding relevant contexts, BAIR employs a
decoupled strategy consisting ofVisual Sharpness and Mass Recover(VSMR), andPosition-Aware
Textual Penalization(PATP). BAIR restores overall visual probability mass, sharpens focal visual
features, and gently dampens textual distractors, safely fusing visual and textual evidence. Because
BAIR requires no model retraining, external datasets, or auxiliary models, it serves as a highly
efficient, architecture-agnostic solution. The overall workflow of BAIR is illustrated in Figure 4.
To ensure mathematical stability and preserve autoregressive generation dynamics, BAIR is applied
exclusively at the “bottleneck” layer during the pre-filling stage. We define the bottleneck as the
specific attention operation where the final input token’s hidden state serves as the query to predict the
first generated token. Rather than intervening at only a single layer, we implement BAIR layerwise
and headwise across the network. By confining the intervention to this single adjustment, we leave
the subsequent step-by-step autoregressive decoding phase entirely unchanged. Let the pre-softmax
attention vector ( qkT) at this bottleneck be partitioned into a visual segment Ev∈RNvand a textual
segment Et∈RL, where Nvdenotes the number of image patches and Lrepresents the number of
prompt text tokens, including the retrieval context.
Prior to the primary generation pass, we execute an auxiliary, lightweight forward pass containing
only the image, the instruction, and the question. From this uncorrupted state, we extract two precise,
post-softmax targets corresponding to the metrics defined in Section 3.1: the target Visual Attention
Mass ( Mtarget , Eq. 1) and the target Visual Attention Sharpness ( Starget , Eq. 1). Crucially, these
values are not computed beforehand or treated as static global priors. Instead, they are instance-
specific measurements calculated dynamically on-the-fly for each unique input combination. These
instance-specific target values serve as data-driven anchors for the subsequent calibration mechanism.
4.1 Visual Sharpness and Mass Recovery (VSMR)
To counteract visual suppression while allowing the model to incorporate textual evidence, BAIR
modulates the visual vector using a hyperparameter: the Mass Interpolation factor (α v).
6

We first standardize the raw pre-softmax logits to compute Zv= (E v−µv)/σv, where µvandσv
denote the mean and standard deviation of the visual logits Ev, respectively. To filter background
noise while preserving structural anatomical context, we apply the parameter-free SiLU activation
function [ 6], yielding the gated vector Gv=Z v⊙sigmoid(Z v). BAIR systematically recovers
the original focal clarity by forcing the visual attention distribution to exactly match the reference
sharpness target Starget . To achieve this, we introduce a temperature scalar Tto scale the gated
visual logits, which directly modulates the entropy of the resulting attention weights. By defining
the parameterized softmax distribution ˆAi(T) = exp(T·G v,i)/P
k∈Vexp(T·G v,k), the visual
sharpnessS vis(T)can be formally expressed as a continuous function ofT:
Svis(T) = 1−−P
i∈VˆAi(T) log( ˆAi(T))
log(|V|)(2)
Because normalized entropy strictly decreases as the scaling factor increases, Svis(T)is a monoton-
ically increasing function. We deploy a lightweight 1D bisection search to solve the root-finding
problem f(T) =S vis(T)−S target = 0, bounding T∈[0, T max]and solving within a tolerance of
ϵ= 10−4. Using the converged rootT∗, we scale the gated logits: ˜Ev=G v·T∗.
Subsequently, we calculate the exact uniform shift αrequired to restore the original visual probability
mass ( Mtarget ). Because the internal softmax distribution is invariant to global scalar addition,
adding αprecisely restores the mass without disturbing the restored sharpness (the full derivation is
provided in Appendix A and Appendix B):
α= logMtarget
1−M target
+ log
LX
j=1eEt,j
−log NvX
i=1e˜Ev,i!
(3)
Let the exact target logits be defined as Etarget
v =˜Ev+α. We interpolate the restored vision with
the current context using the mass interpolation factorα v>0:
ˆEv=E v+αv 
Etarget
v−Ev
(4)
Whileα v= 1strictly restores the target mass, settingα v>1actively amplifies the visual evidence
beyond its original uncorrupted baseline, acting as a focal highlight. This formulation mathematically
guides the focal attention back toward the visual evidence while allowing the textual context to
maintain a controlled influence. The impact of the hyperparameterα vis analyzed in Appendix C.
4.2 Position-Aware Textual Penalization (PATP)
Positional degradation in retrieval-augmented generation manifests unpredictably, typically skewing
toward extreme primacy or recency bias. To autonomously counteract this without assuming a fixed
attention geometry (e.g., a static U-shape), we propose a data-driven, asymmetric penalty.
At the pre-filling bottleneck, we calculate the global logit mean of the entire textual sequence ( µt).
To isolate the specific boundaries susceptible to positional distraction, we calculate the regional logit
means for the first 20% ( µhead) and the last 20% ( µtail) of the textual tokens. We define dynamic
penalty weights that autonomously scale based on the severity of the detected positional bias:
λprim= max (0, µ head−µt), λ rec= max (0, µ tail−µt)(5)
The pre-softmax textual logits Etof sequence length Lare then adjusted using independent quadratic
decay functions anchored to the left and right context boundaries. For an arbitrary textual token at
indexj∈ {1, . . . , L}, the calibrated logit ˆEt,jis:
ˆEt,j=E t,j−"
λprim
max
0,1−2j
L2
+λrec
max
0,2j
L−12#
(6)
This formulation structurally isolates the positional bias without requiring manual hyperparameter
tuning for different document lengths. For example, if the attention distribution exhibits extreme
recency bias, characterized by a massive logit spike at the tail, λrecscales proportionally to dampen
the rightmost tokens, while λprim evaluates to zero. This dynamic response ensures that central
instructional tokens and healthy mid-document attention distributions are never artificially penalized.
7

Table 1: Comprehensive evaluation of multimodal RAG interventions on theIU-Chest,FACET, and
NWPUdataset. Metrics are presented in percentages (%). Blue-shaded cells denote a performance
gain achieved by the BAIR intervention relative to its corresponding baseline.
Dataset / ModelIU-Chest / MedGemma-4B FACET / Qwen2.5-VL-7B NWPU / SkySenseGPT-7B
Method Acc.↑CR↑DR↓CR/DR↑Acc.↑CR↑DR↓CR/DR↑Acc.↑CR↑DR↓CR/DR↑
No RAG 63.76 - - - 84.91 - - - 68.90 - - -
Baseline
Standard RAG 66.33 28.00 13.56 2.06 84.34 26.62 5.41 4.92 65.74 44.68 24.75 1.81
Visual-focus Instr. 65.38 26.68 14.13 1.89 84.38 28.25 5.65 5.00 66.59 46.83 24.49 1.91
LongLLMLingua 65.66 25.14 12.79 1.97 84.36 31.59 6.26 5.05 69.97 46.50 19.43 2.39
MS-PoE 67.17 20.68 7.85 2.63 84.20 29.07 8.60 3.38 65.36 45.26 25.57 1.77
MAD-RAG 68.03 24.58 9.01 2.73 84.41 25.53 5.11 5.00 66.18 45.67 24.56 1.86
BAIR
Standard RAG 68.3124.25 8.40 2.89 88.56 50.31 4.64 10.84 66.3144.02 23.63 1.86
Visual-focus Instr. 65.6924.48 12.33 1.99 87.28 44.93 5.20 8.64 66.7446.00 23.89 1.93
LongLLMLingua 65.91 25.21 12.46 2.02 87.45 45.97 4.89 9.40 70.7946.41 18.20 2.55
MS-PoE 67.60 21.01 7.43 2.83 92.58 70.71 3.53 20.03 66.0544.27 24.12 1.84
MAD-RAG 68.2524.19 8.45 2.86 85.10 36.08 6.19 5.83 67.13 46.17 23.41 1.97
5 Experimental Details
5.1 Implementation Detail
We conduct extensive experiments across three distinct domains: medical, social fairness, and
geospatial. For the medical domain, we utilize the IU-Chest dataset [ 5] and evaluate the MedGemma-
4B [20] and CheXagent-8B [ 2] models, which are specialized for processing medical images to
generate radiology reports. To simulate real-world retrieval scenarios, we use MedSigLIP [ 20] to
retrieve the top-5 most relevant documents from a database comprising all reports in the dataset. The
ground-truth (GT) report is deliberately positioned in the middle of the retrieved context, with the
remaining documents serving as distractors to simulate the “lost-in-the-middle” phenomenon. For
the social domain, we select the FACET dataset [ 7], which provides explicit gender and profession
attributes for each image. To assess whether the models (Qwen2.5-VL-7B [ 23] and DeepSeek-VL-7B
[15]) exhibit to visual blindness, we provide a gender-neutral Wikipedia introduction of the depicted
profession as the RAG context. We then measure whether the pronouns in the generated text correctly
match the ground-truth visual gender attribute. Finally, for the geospatial domain, we adopt the
NWPU-RESISC45 remote sensing dataset [ 3] and evaluate the SkySenseGPT [ 16] and EarthDial
[21] models. For the 45 scene classes, we construct a context containing one ground-truth Wikipedia
paragraph and four distractors retrieved via RemoteCLIP [ 13]. The instruction and questions used in
each configuration is introduced in Appendix D.
For comparison, we evaluate several existing intervention methods: MS-PoE [ 28], LongLLMLingua
[9], MAD-RAG [ 29], and a prompt-based baseline that explicitly instructs the model to prioritize
visual evidence (Visual-focus Instrt.). Comprehensive details regarding these comparative baselines
are provided in Appendix E. While each of these methods can operate standalone, our proposed BAIR
framework functions as an additional calibration layer that can be integrated directly on top of them.
To evaluate the efficacy of multimodal RAG interventions across diverse domains, we establish a
generalized evaluation framework. Let a generated response xfor a given ground truth ˆxbe evaluated
by a task-specific score function S(x,ˆx)∈[0,1] . We define the scoring criteria for our three primary
evaluation domains as follows:
•Clinical Factuality (Medical):We utilize the SRR-BERT [ 4] framework to extract stan-
dardized clinical observation labels. The score Smedis defined as the F1-score between the
extracted labels of the generation and the ground truth. This continuous evaluation captures
partial diagnostic matches, avoiding the brittleness of binary exact-match criteria.
•Social Fairness (Social):Correctness is determined by the mitigation of biased hallucina-
tions. We evaluate the presence of gendered pronouns in the generated response relative to
the factual visual evidence. We define Sfair= 1if the model correctly aligns its pronoun
usage with the ground truth, and0if it hallucinates a biased demographic attribute.
•Geospatial Factuality (Remote Sensing):We evaluate scene understanding by employing a
strict keyword matching protocol. The score Sgeo= 1if the generated response successfully
includes the core keywords associated with the ground-truth scene class, and0otherwise.
8

1.8 2.0 2.2 2.4 2.6 2.8 3.0
CR / DR1.52.02.53.03.54.04.55.0Accuracy Gain ( Acc.) %
IUChest / MedGemma-4B
1.6 1.8 2.0 2.2 2.4
CR / DR1.0
0.5
0.00.51.0IUChest / CheXAgent-8B
2.5 5.0 7.5 10.0 12.5 15.0 17.5 20.0 22.5
CR / DR4
2
02468FACET / Qwen2.5-VL-7B
1.8 2.0 2.2 2.4 2.6
CR / DR4
3
2
1
012NWPU / SkySenseGPT-7BStandard RAG Visual-focused Inst. LongLLMLingua MS-PoE MAD-RAG Base Method + BAIR InterventionFigure 5:Impact of the BAIR intervention on multimodal RAG pipelines.The y-axis represents
the net gain in Accuracy relative to the baseline (red dotted line), and the x-axis indicates the CR/DR
ratio (Correction Rate over Degradation Rate). Trajectory arrows show the performance shift when
applying BAIR to existing mitigation strategies, moving from the base method (circles) to the BAIR-
calibrated output (stars). In all configurations, BAIR yields higher overall accuracy and significantly
improving the correction-to-degradation balance.
To evaluate the overall performance of each method, we reportAccuracy (Acc.)as the mean score
across the dataset. To capture instance-level dynamics, specifically how context alters individual
predictions, we binarize these scores using a predefined correctness threshold to establish strict
success or failure states. Based on this binarization, we track two key metrics: theCorrection Rate
(CR), which measures the proportion of no-retrieval baseline failures successfully corrected by the
added context, and theDegradation Rate (DR), which quantifies the recorruption phenomenon
by measuring the proportion of initially correct baseline predictions that the context degrades into
failures. The formal mathematical formulations for these metrics are detailed in Appendix F.
6 Result Analysis
Figure 1 demonstrates qualitative examples of BAIR successfully mitigating text-induced blindness,
correcting instances where standard RAG fails to identify critical visual pathologies or attributes.
Also, Table 1 highlights the effectiveness of BAIR in enhancing overall performance when applied
across various intervention baselines. Across all evaluated datasets and models, the overall accuracy
consistently increases due to a higher correction rate and a suppressed degradation rate. The complete
results featuring all models and extended evaluation metrics are detailed in Appendix I.
To effectively visualize these improvements, Figure 5 plots the performance shifts. Specifically, the
y-axis shows the overall accuracy gain and the x-axis represents the CR/DR ratio. A higher value on
the x-axis indicates significant correction while simultaneously minimizing recorruption. In every
tested configuration, applying BAIR successfully shifts the performance trajectory toward the upper
right region. This confirms that BAIR is highly adaptable across different datasets, models, and
domains, consistently guiding existing intervention methods to superior performance.
Furthermore, BAIR introduces no additional trainable parameters and does not require increased
GPU memory usage. As reported in Appendix G, the intervention adds only a lightweight inference
time calibration step, resulting in a negligible latency increase in practice. Across all experiments, we
tune a single hyperparameter, αv. Appendix C analyzes the effect of αvand isolates the individual
contributions of VSMR and PATP. Appendix J further reports mean and standard deviation across
repeated evaluations, showing that the performance gains are stable rather than driven by a single run.
Limitations.The main limitation of BAIR is that its inference time calibration still requires an
auxiliary reference pass, even though the added time cost is small. In addition, αvis task dependent.
While the tested values of αvgenerally improve performance over the corresponding RAG baseline,
selecting the best value in a new domain may require a small validation set or a heuristic choice.
7 Conclusion
In this work, we identify and formalize the phenomenon of recorruption in Multimodal RAG. Through
mechanistic diagnosis, we reveal that introducing external text into MLLMs frequently triggers severe
visual blindness and structural positional biases, causing models to abandon correct visual perception
in favor of textual distractors. To address this, we introduce Bottleneck Attention Intervention for
Recovery (BAIR), a parameter-free, inference-time framework. By dynamically restoring visual
9

mass and focal sharpness while penalizing positional textual biases, BAIR safely fuses multimodal
context. Our extensive evaluations across medical factuality, social fairness, and geospatial domains
demonstrate that BAIR consistently recovers ground-truth accuracy and suppresses recorruption
without requiring model retraining or fine-tuning. Ultimately, this work provides both a critical
diagnostic lens and a practical cure for the hidden vulnerabilities of multimodal RAG architectures.
References
[1]B. An, S. Zhang, and M. Dredze. Rag llms are not safer: A safety analysis of retrieval-augmented
generation for large language models.arXiv preprint arXiv:2504.18041, 2025.
[2]Z. Chen, M. Varma, J.-B. Delbrouck, M. Paschali, L. Blankemeier, D. Van Veen, J. M. J.
Valanarasu, A. Youssef, J. P. Cohen, E. P. Reis, et al. Chexagent: Towards a foundation model
for chest x-ray interpretation. InAAAI 2024 Spring Symposium on Clinical Foundation Models,
2024.
[3]G. Cheng, J. Han, and X. Lu. Remote sensing image scene classification: Benchmark and state
of the art.Proceedings of the IEEE, 105(10):1865–1883, 2017.
[4]J.-B. Delbrouck, J. Xu, J. Moll, A. Thomas, Z. Chen, S. Ostmeier, A. Azhar, K. Z. Li, A. John-
ston, C. Bluethgen, et al. Automated structured radiology report generation. InProceedings of
the 63rd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long
Papers), pages 26813–26829, 2025.
[5]D. Demner-Fushman, M. D. Kohli, M. B. Rosenman, S. E. Shooshan, L. Rodriguez, S. Antani,
G. R. Thoma, and C. J. McDonald. Preparing a collection of radiology examinations for
distribution and retrieval.Journal of the American Medical Informatics Association, 23(2):
304–310, 2015.
[6]S. Elfwing, E. Uchibe, and K. Doya. Sigmoid-weighted linear units for neural network function
approximation in reinforcement learning.Neural networks, 107:3–11, 2018.
[7]L. Gustafson, C. Rolland, N. Ravi, Q. Duval, A. Adcock, C.-Y . Fu, M. Hall, and C. Ross.
Facet: Fairness in computer vision evaluation benchmark. InProceedings of the IEEE/CVF
International Conference on Computer Vision, pages 20370–20382, 2023.
[8]C.-Y . Hsieh, Y .-S. Chuang, C.-L. Li, Z. Wang, L. Le, A. Kumar, J. Glass, A. Ratner, C.-Y . Lee,
R. Krishna, et al. Found in the middle: Calibrating positional attention bias improves long
context utilization. InFindings of the Association for Computational Linguistics: ACL 2024,
pages 14982–14995, 2024.
[9]H. Jiang, Q. Wu, X. Luo, D. Li, C.-Y . Lin, Y . Yang, and L. Qiu. Longllmlingua: Accelerating
and enhancing llms in long context scenarios via prompt compression. InProceedings of
the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long
Papers), pages 1658–1677, 2024.
[10] E. Kortukov, A. Rubinstein, E. Nguyen, and S. J. Oh. Studying large language model behaviors
under context-memory conflicts with real documents.arXiv preprint arXiv:2404.16032, 2024.
[11] P. Lewis, E. Perez, A. Piktus, F. Petroni, V . Karpukhin, N. Goyal, H. Küttler, M. Lewis, W.-t.
Yih, T. Rocktäschel, et al. Retrieval-augmented generation for knowledge-intensive nlp tasks.
Advances in neural information processing systems, 33:9459–9474, 2020.
[12] C.-Y . Lin. Rouge: A package for automatic evaluation of summaries. InText summarization
branches out, pages 74–81, 2004.
[13] F. Liu, D. Chen, Z. Guan, X. Zhou, J. Zhu, Q. Ye, L. Fu, and J. Zhou. Remoteclip: A vision
language foundation model for remote sensing.IEEE Transactions on Geoscience and Remote
Sensing, 62:1–16, 2024.
[14] N. F. Liu, K. Lin, J. Hewitt, A. Paranjape, M. Bevilacqua, F. Petroni, and P. Liang. Lost in
the middle: How language models use long contexts.Transactions of the Association for
Computational Linguistics, 12:157–173, 2024.
10

[15] H. Lu, W. Liu, B. Zhang, B. Wang, K. Dong, B. Liu, J. Sun, T. Ren, Z. Li, H. Yang,
et al. Deepseek-vl: towards real-world vision-language understanding.arXiv preprint
arXiv:2403.05525, 2024.
[16] J. Luo, Z. Pang, Y . Zhang, T. Wang, L. Wang, B. Dang, J. Lao, J. Wang, J. Chen, Y . Tan,
et al. Skysensegpt: A fine-grained instruction tuning dataset and model for remote sensing
vision-language understanding.arXiv preprint arXiv:2406.10100, 2024.
[17] M. K. Mandanetwork, H. E. Rekik, and O. Bouaziz. Enhancing technical knowledge acquisition
with rag systems: the tei use case. InTexts, Languages and Communities-TEI 2024, 2024.
[18] K. Marino, M. Rastegari, A. Farhadi, and R. Mottaghi. Ok-vqa: A visual question answering
benchmark requiring external knowledge. InProceedings of the IEEE/cvf conference on
computer vision and pattern recognition, pages 3195–3204, 2019.
[19] M. Raja, E. Yuvaraajan, et al. A rag-based medical assistant especially for infectious diseases.
In2024 International Conference on Inventive Computation Technologies (ICICT), pages
1128–1133. IEEE, 2024.
[20] A. Sellergren, S. Kazemzadeh, T. Jaroensri, A. Kiraly, M. Traverse, T. Kohlberger, S. Xu,
F. Jamil, C. Hughes, C. Lau, et al. Medgemma technical report.arXiv preprint arXiv:2507.05201,
2025.
[21] S. Soni, A. Dudhane, H. Debary, M. Fiaz, M. A. Munir, M. S. Danish, P. Fraccaro, C. D. Watson,
L. J. Klein, F. S. Khan, et al. Earthdial: Turning multi-sensory earth observations to interactive
dialogues. InProceedings of the Computer Vision and Pattern Recognition Conference, pages
14303–14313, 2025.
[22] H. Wadhwa, R. Seetharaman, S. Aggarwal, R. Ghosh, S. Basu, S. Srinivasan, W. Zhao,
S. Chaudhari, and E. Aghazadeh. From rags to rich parameters: Probing how language models
utilize external knowledge over parametric information for factual queries.arXiv preprint
arXiv:2406.12824, 2024.
[23] P. Wang, S. Bai, S. Tan, S. Wang, Z. Fan, J. Bai, K. Chen, X. Liu, J. Wang, W. Ge, Y . Fan,
K. Dang, M. Du, X. Ren, R. Men, D. Liu, C. Zhou, J. Zhou, and J. Lin. Qwen2-vl: En-
hancing vision-language model’s perception of the world at any resolution.arXiv preprint
arXiv:2409.12191, 2024.
[24] Z. Wang, H. Zhang, X. Li, K.-H. Huang, C. Han, S. Ji, S. M. Kakade, H. Peng, and H. Ji.
Eliminating position bias of language models: A mechanistic approach. InThe Thirteenth
International Conference on Learning Representations, 2025. URL https://openreview.
net/forum?id=fvkElsJOsN.
[25] N. Wiratunga, R. Abeyratne, L. Jayawardena, K. Martin, S. Massie, I. Nkisi-Orji, R. Weeras-
inghe, A. Liret, and B. Fleisch. Cbr-rag: case-based reasoning for retrieval augmented generation
in llms for legal question answering. InInternational Conference on Case-Based Reasoning,
pages 445–460. Springer, 2024.
[26] X. Wu, Y . Wang, S. Jegelka, and A. Jadbabaie. On the emergence of position bias in transformers.
In A. Singh, M. Fazel, D. Hsu, S. Lacoste-Julien, F. Berkenkamp, T. Maharaj, K. Wagstaff,
and J. Zhu, editors,Proceedings of the 42nd International Conference on Machine Learning,
volume 267 ofProceedings of Machine Learning Research, pages 67756–67781. PMLR, 13–19
Jul 2025. URLhttps://proceedings.mlr.press/v267/wu25ad.html.
[27] J. Yao, S. Liu, Y . Wang, L. Mei, B. Bi, Y . Ge, Z. Li, and X. Cheng. Who is in the spotlight:
The hidden bias undermining multimodal retrieval-augmented generation.arXiv preprint
arXiv:2506.11063, 2025.
[28] Z. Zhang, R. Chen, S. Liu, Z. Yao, O. Ruwase, B. Chen, X. Wu, and Z. Wang. Found in the
middle: How language models use long contexts better via plug-and-play positional encoding.
Advances in Neural Information Processing Systems, 37:60755–60775, 2024.
[29] B. Zhao, W. Deng, X. Liao, Y . Li, N. Shaikh, Y . Nie, and X. Li. When rag hurts: Diagnosing and
mitigating attention distraction in retrieval-augmented lvlms.arXiv preprint arXiv:2602.00344,
2026.
11

A Derivation of the Mass Restoration Shift
We seek a global scalar shift αto be added to the sharpened visual logits ˜Ev∈RNvsuch that the
post-softmax visual probability mass exactly equals the uncorrupted target massM target .
Let the pre-softmax textual logits be denoted as Et∈RL. The global softmax function defines the
visual mass as the sum of exponentiated visual logits divided by the sum of all exponentiated logits
(visual and textual):
Mtarget =PNv
i=1e˜Ev,i+α
PNv
i=1e˜Ev,i+α+PL
j=1eEt,j(7)
To simplify the algebra, let V=PNv
i=1e˜Ev,irepresenting the sum of the exponentiated visual logits
prior to the shift, and let T=PL
j=1eEt,jrepresenting the sum of the exponentiated textual logits.
Factoring outeαfrom the visual terms yields:
Mtarget =eαV
eαV+T(8)
Multiplying both sides by the denominator:
Mtarget (eαV+T) =eαV(9)
Mtarget eαV+M target T=eαV(10)
Rearranging to isolate the terms containingeαon one side:
Mtarget T=eαV−M target eαV(11)
Mtarget T=eαV(1−M target )(12)
Solving foreα:
eα=Mtarget T
V(1−M target )(13)
Taking the natural logarithm of both sides gives the closed-form solution forα:
α= logMtarget
1−M target
+ log(T)−log(V)(14)
Substituting TandVback into the equation yields the final mass restoration shift applied in Equa-
tion 3:
α= logMtarget
1−M target
+ log
LX
j=1eEt,j
−log NvX
i=1e˜Ev,i!
(15)
B Bisection Method for Visual Sharpness Restoration
The BAIR framework restores the focal clarity of visual attention by solving for the optimal tem-
perature scalar T∗such that Svis(T∗) =S boost. Since Svis(T)is a strictly monotonic function, we
utilize the bisection method to findT∗within a defined search space[0, T max].
The algorithm initializes with the interval [0, T max]and iteratively refines the search. In each step,
the midpoint Tmidis evaluated. If Svis(Tmid)is lower than the target Sboost, the lower bound is
updated toT mid; otherwise, the upper bound is updated. This process continues until the difference
between the observed sharpness and the target is within a tolerance ofϵ= 10−4.
This numerical approach is both stable and highly efficient. Unlike gradient-based solvers, it does not
require second-order information or step-size tuning. As illustrated in Figure 6, the algorithm achieves
convergence in fewer than 10 iterations, representing a negligible computational cost (approx. 0.3
ms) relative to the model’s standard forward pass.
12

Figure 6:Efficiency of the bisection search on the monotonic Svis(T)function.Sharpness is a
strictly increasing function of the temperature scalar T, ensuring a unique root for any Sboost∈[0,1] .
As shown by the numbered steps, the bisection method achieves convergence in fewer than 10
iterations. This targeted intervention at the pre-filling bottleneck layer allows BAIR to restore ground-
truth visual clarity with negligible computational weight.
C Ablation Study
We conduct an ablation study on MedGemma-4B using the IU Chest dataset to examine the contribu-
tion of each BAIR component and the sensitivity to the mass interpolation factor αv. Figure 7 reports
both the overall F1 score and the instance level correction and degradation rates.
First, we compare the two individual components of BAIR (PATP + VSMR). ThePATP onlyvariant
applies textual intervention without visual mass and sharpness recovery, while theVSMR onlyvariant
restores visual attention without applying the textual positional penalty. The results show that each
component contributes differently. PATP only achieves a relatively high correction rate, indicating
that reducing boundary dominated textual distraction helps recover some failed cases. However, it
also retains a relatively high degradation rate, suggesting that textual recalibration alone is insufficient
to fully prevent recorruption. In contrast, VSMR only reduces degradation more effectively, but its
correction rate is lower, indicating that visual recovery alone may not fully exploit useful retrieved
evidence.
Second, we vary αvin the full BAIR framework. The best overall F1 score is obtained when
αv= 0.5 , suggesting that moderate visual recovery provides the most effective balance between
restoring visual grounding and preserving useful textual context. Smaller values of αvunderutilize
the recovered visual signal, while larger values yield diminishing returns and may overemphasize the
visual modality at the expense of complementary textual evidence.
Overall, the ablation confirms that VSMR and PATP are complementary: VSMR recovers visually
grounded attention, while PATP suppresses positional textual distraction. Their combination produces
the most favorable balance between improving correction and reducing degradation.
D Instruction and Question Details
This section summarizes the questions, system instructions, and prompting configurations used for
each dataset and model family.
13

Figure 7:Ablation.Conduct ablation study on MedGemma-4B on IU-Chest Dataset.
D.1 IU Chest X ray
For the IU Chest experiments, both CheXagent and MedGemma use the same question and instruction
templates.
Question
Based on the visual evidence, what are the primary impressions for this chest
radiograph?
Base instruction
You are a radiologist. When context is provided, refer to it to accurately
describe the image. If no context is provided, describe the image based on your
knowledge.
Visual focus instruction
You are an expert radiologist. Crucially, you must prioritize the visual
evidence in the X ray over the provided text. The text contains historical
distractors. If the text contradicts the image, you must trust the image and
ignore the text.
D.2 FACET
For the FACET experiments, the question is instantiated with the profession label for each sample.
Question template
Write a short fictional workday narrative about the {profession} shown in this
synthetic benchmark image. Use standard singular pronouns (he/his or she/her)
rather than gender-neutral terms.
14

Here,{profession}is filled separately for each sample, such aslawman,drummer, orprayer.
Default instruction
You are a helpful assistant. When context is provided, refer to it to
accurately describe the image. If no context is provided, describe the image
based on your knowledge.
Each FACET prompt additionally prepends the following notice.
Experiment notice
Experiment setting: This image is from a synthetic fairness benchmark and
does not represent a real identifiable person. Provide a concise fictional
description for research purposes.
For visual-foxcused prompt experiments, we use the following instruction.
Visual focus instruction
You are a precise vision language assistant. Prioritize the visual evidence
in the image over retrieved text when they conflict. Use the retrieved text
only as supporting context, and provide a concise narrative answer with explicit
singular pronouns.
D.3 NWPU
For the NWPU experiments, the default prompt asks the model to identify the primary land use or
land cover category from the satellite image.
Default question
You are an expert in remote sensing and geospatial analysis. Examine the
provided satellite image and identify its primary land use or land cover
category.
Default instruction
Use the image as primary evidence and use retrieved context as supporting
information.
Visual focus instruction
You are an expert remote sensing analyst. Prioritize the visual satellite
evidence over text context. If context conflicts with image content, trust the
image.
E Comparison Methods
E.1 MS-PoE [28]
MS-PoE is a positional recalibration method designed to improve long context utilization by modi-
fying positional encoding behavior during inference [ 28]. Rather than changing model parameters,
it adjusts how the model represents token positions so that information placed away from the con-
text boundaries can receive more effective attention. In our experiments, we use MS-PoE as a
representative position based baseline for mitigating the lost in the middle effect. Since MS-PoE
primarily targets textual position bias, it provides a useful comparison for evaluating whether position
recalibration alone can resolve recorruption in multimodal RAG. Unlike BAIR, however, MS-PoE
does not explicitly restore visual attention mass or visual attention sharpness.
15

E.2 LongLLMLingua [9]
LongLLMLingua is a prompt compression method that improves long context processing by identify-
ing and retaining the most informative tokens from the retrieved context [ 9]. By shortening the textual
context and reducing redundant or irrelevant content, it can lower the attention burden imposed by
long retrieved documents. In our experiments, we use LongLLMLingua as a text compression base-
line for multimodal RAG. This comparison tests whether reducing textual length alone is sufficient to
mitigate visual suppression and recorruption. While LongLLMLingua can reduce textual distraction,
it does not directly intervene on the multimodal attention competition between visual and textual
tokens, nor does it explicitly recover the visual attention mass and sharpness suppressed by retrieved
context.
E.3 MAD-RAG [29]
To contextualize our contributions, we provide a technical comparison with MAD-RAG [ 29], a
contemporary baseline that addresses similar failures in multimodal retrieval. MAD-RAG operates
on the premise of Attention Distraction (AD), where retrieved context globally suppresses visual
attention. To mitigate this, it employs a dual-question prompt construction [I, Q I, C, Q C], where an
initial “image-question” ( QI) establishes a visual reference isolated from the context C, followed
by a “context-question” ( QC) for integrated reasoning. The intervention is performed at each
layer via attention mixing, which is a convex combination of the attention outputs: ˆO(Q C) =
α·O(Q I) + (1−α)·O(Q C).
Our proposed BAIR framework offers several key advantages over this approach:
•Mechanistic Depth: While MAD-RAG focuses on the total quantity of attention mass,
BAIR identifies that recorruption is also driven by a qualitative failure where visual focus
becomes dangerously diffuse. By introducing Visual Attention Sharpness ( Svis), BAIR can
diagnose and restore the model’s focal clarity rather than just increasing the global attention
ratio.
•Dynamic Restoration vs. Heuristic Mixing: MAD-RAG relies on a heuristic weight α
(typically set to a static 0.5) to balance modalities. In contrast, BAIR avoids global heuristics
by dynamically extracting instance-specific restoration targets ( Mtarget andStarget ) from a
lightweight reference pass, ensuring that calibration is tailored to the unique complexity of
each image-text pair.
•Structural Mitigation of Positional Bias: MAD-RAG does not explicitly treat the under-
lying causes of recency bias. BAIR addresses the multimodal “lost-in-the-middle” phe-
nomenon through Position-Aware Textual Penalization (PATP), which applies a data-driven
quadratic penalty to boundary distractors that otherwise dominate the attention budget.
•Computational and Prompt Efficiency: MAD-RAG requires duplicating question tokens,
which inflates the sequence length, and necessitates layer-wise interventions forevery
decoding step. BAIR maintains a standard prompt and preserves the integrity of the
autoregressive decoding phase by implementing its layer-wise intervention exclusively
during thesinglepre-filling bottleneck stage.
F Evaluation Metrics
LetSM(i)denote the continuous score for a sample iunder an evaluated method M. For our
comparative analysis, we denote Bas the No Retrieval baseline, Ras the Standard RAG condition,
andIas a proposed intervention (e.g., BAIR). We utilize the logical indicator function I(·)to define
the specific populations for each metric. Upon the overall accuracy, we utilize correction rate and
degradation rate to measure the magnitude of improvement and recorruption, respectively.
Accuracy (Acc.).The overall average performance across the dataset. This represents the Mean
F1-score for the medical domain and the exact-match accuracy for the fairness domain, applicable to
any evaluated methodM∈ {B, R, I}:
Acc.=1
NNX
i=1SM(i)(16)
16

Correction Rate (CR).The expected point gain among samples where the baseline failed to achieve
a perfect score. This evaluates the utility of retrieval-augmented methods M∈ {R, I} against the
baselineB:
CR=PN
i=1max(0,S M(i)− S B(i))
PN
i=1I(SB(i)<1.0)(17)
Degradation Rate (DR).The expected point loss relative to the baseline among samples where
the baseline originally possessed partial or full correctness. This mathematically quantifies the
recorruption introduced by retrieval-augmented methodsM∈ {R, I}:
DR=PN
i=1max(0,S B(i)− S M(i))
PN
i=1I(SB(i)>0.0)(18)
Recovery Rate (RR).A metric strictly for evaluating interventions I. It measures the expected gain
recovered from samples where Standard RAG (R) failed to achieve a perfect score:
RR=PN
i=1max(0,S I(i)− S R(i))
PN
i=1I(SR(i)<1.0)(19)
Strictly Cured Rate (SR).The strictest measure of an intervention’s success. It isolates the explicit
recorrupted population (where the baseline outperformed Standard RAG) and calculates the average
points fully restored by I, conditional on the intervention matching or exceeding the original baseline:
SR=PN
i=1I(SI(i)≥ S B(i))·I(S B(i)>S R(i))·(S I(i)− S R(i))
PN
i=1I(SB(i)>S R(i))(20)
Novel Recovery Rate (NR).The expected point gain strictly attributable to the intervention on
“hard” samples, where both the baselineBand Standard RAGRfailed to achieve perfect scores:
NR=PN
i=1max(0,S I(i)−max(S B(i),S R(i)))
PN
i=1I(SB(i)<1.0)·I(S R(i)<1.0)(21)
Generation Failure Rate (GFR).This metric quantifies the prevalence of degenerate model outputs
that are technically invalid due to aggresive intervention. We define a binary failure indicator
F(x)∈ {0,1} for a generated response x. A failure is identified if the response is empty after
stripping whitespace, contains fewer than 5 characters, or contains a single token repeated five or
more times consecutively. The GFR is calculated as:
GFR=1
NNX
i=1F(x i)(22)
Importantly, for all evaluation domains, any response where F(x) = 1 is assigned a task score
ofS= 0 . This ensures that degenerate or repetitive outputs are not incorrectly credited as fair or
factually accurate.
G Cost Analysis
Tables 2 and 3 report the computational cost of each method on two representative settings:
MedGemma-4B on IU-Chest and Qwen2.5-VL-3B on FACET. We report both the raw runtime
and peak GPU memory, as well as the percentage change relative to Standard RAG, to make the cost
tradeoff explicit.
BAIR introduces a modest runtime overhead because it performs an additional no context calibration
pass to estimate the instance specific visual attention mass and sharpness used for intervention. After
this calibration pass, BAIR applies an adaptive bottleneck intervention to the prefill attention logits
of the RAG prompt. This operation restores visual attention mass and sharpness while penalizing
text context attention before decoding begins. Since the intervention is restricted to the prefill stage
and the subsequent token by token decoding follows the standard cached attention path, the overhead
17

remains small and comparable to MAD-RAG. Specifically, BAIR increases runtime by 10.22% on
IU-Chest and 5.37% on FACET relative to Standard RAG.
In terms of memory, BAIR uses less peak GPU memory than Standard RAG in both settings. This
is because the shared prefix implementation reuses the visual and pre context prefix cache, so the
final context conditioned generation does not need to recompute the full multimodal prefill over
both image tokens and retrieved documents. MS-PoE is faster because it does not require either a
calibration pass or context compression, and instead only modifies positional encoding during the
standard generation path. LongLLMLingua has competitive latency, but its peak GPU memory is
substantially higher because it loads an additional compressor model to rewrite the retrieved context
before generation. MAD-RAG has a cost close to BAIR because it also relies on a no context
reference pass, although its intervention is simpler and does not include the same adaptive bottleneck
calibration. The computational resources used for the experiments are detailed in Appendix L.
Table 2: Computational cost comparison on MedGemma-4B with the IU-Chest dataset. Runtime and
memory changes are reported relative to Standard RAG.
Method Mean s/sample∆Time (%) Peak GPU MB∆Memory (%)
Standard RAG 5.38 0.00 12652 0.00
LongLLMLingua 4.82 -10.41 26686 +110.92
MS-PoE 4.08 -24.16 12652 0.00
MAD-RAG 5.69 +5.76 12694 +0.33
BAIR 5.93 +10.22 11704 -7.49
Table 3: Computational cost comparison on Qwen2.5-VL-3B with the FACET dataset. Runtime and
memory changes are reported relative to Standard RAG.
Method Mean s/sample∆Time (%) Peak GPU MB∆Memory (%)
Standard RAG 7.08 0.00 8158 0.00
LongLLMLingua 6.25 -11.72 21688 +165.85
MS-PoE 6.01 -15.11 8110 -0.59
MAD-RAG 7.53 +6.36 7714 -5.44
BAIR 7.46 +5.37 7714 -5.44
H Attention Analysis
We analyze the final layer attention profile by measuring how the last input token attends to the
multimodal prompt. For each sample, we retain only visual tokens and retrieved document tokens,
excluding instruction and question tokens so that the visualization focuses on the clinically relevant
evidence sources. The visual token region is shaded, and the third retrieved document is highlighted
as the ground truth document. We compare Baseline, MS-PoE, MAD-RAG, and BAIR using robust
normalized attention curves, and then visualize each method’s change from the Baseline using a
smoothed token axis delta heatmap. This analysis reveals whether each intervention shifts attention
toward visual evidence, the ground truth retrieved document, or irrelevant textual regions.
The delta heatmap shows that BAIR produces the most targeted attention recovery. On the visual side,
BAIR maintains both high attention mass and sharp visual concentration, indicating that it restores
visual grounding rather than merely increasing diffuse visual attention. On the textual side, BAIR
reduces the excessive recency bias toward the final retrieved document while increasing attention to
the ground truth document. In contrast, MAD-RAG increases visual attention mass, but the recovered
visual attention remains comparatively diffuse and its attention to the ground truth document is
weakened. MS-PoE redistributes textual attention by modifying positional behavior, but the resulting
attention pattern is less structured and does not consistently concentrate on either the visual evidence
or the ground truth document. These results further support our diagnosis that effective multimodal
RAG intervention requires both visual attention recovery and position aware textual recalibration.
18

Figure 8:Attention Profile Analysis.We compare the final layer bottleneck attention profiles of
Baseline, MS-PoE, MAD-RAG, and BAIR. The visual token region is shaded, and the third retrieved
document is marked as the ground truth document. The upper panels show robust normalized attention
curves, while the lower panels visualize each method’s attention change relative to the Baseline.
I More Experimental Results
This section provides the complete experimental results across all evaluated datasets, models, and
intervention backbones. While the main text reports representative results due to space constraints,
Tables 4, 5, and 6 present the full comparison on IU-Chest, FACET, and NWPU, respectively. In
addition to Accuracy, Correction Rate (CR), and Degradation Rate (DR), we report Recovery Rate
(RR), Strictly Cured Rate (SR), Novel Recovery Rate (NR), and Generation Failure Rate (GFR)
where applicable. These extended metrics allow us to examine not only whether a method improves
aggregate performance, but also whether it recovers failures introduced by retrieval, suppresses
recorruption, and avoids degenerate generation. Across the three domains, applying BAIR on top of
existing RAG intervention methods generally improves accuracy and reduces degradation, showing
that BAIR is complementary to prompt based, compression based, position based, and attention
mixing baselines. The gains are especially meaningful because they are achieved without retraining
or fine tuning the underlying MLLMs.
J Experiment Statistical Significance
To examine whether the observed gains are stable across repeated evaluations, we report the mean
and standard deviation of Accuracy, Correction Rate, and Degradation Rate in Table 7. For each
dataset and model configuration, we evaluate each baseline method together with its BAIR-augmented
counterpart under the same protocol, enabling a paired comparison that isolates the effect of the
intervention.
Across all domains, BAIR shows stable improvements over the corresponding baseline methods.
On the medical IU-Chest dataset, BAIR consistently reduces Degradation Rate while maintaining
or improving Accuracy, indicating that the intervention suppresses recorruption without discarding
useful retrieved context. On the FACET dataset, BAIR provides stronger gains in Correction Rate and
19

Table 4: Comprehensive evaluation of multimodal RAG interventions on theIU-Chestdataset.
Metrics are presented in percentages (%). Blue-shaded cells denote a performance gain achieved by
the BAIR intervention relative to its corresponding baseline.
Model Method Acc.↑CR↑DR↓CR/DR↑RR↑SR↑NR↑
MedGemma-4BBaseline 63.76 - - - - - -
Standard RAG 66.33 28.00 13.56 2.06 - - -
Visual-focus Instr. 65.38 26.68 14.13 1.89 16.74 30.50 10.35
LongLLMLingua 65.66 25.14 12.79 1.97 21.73 42.41 12.33
MS-PoE 67.17 20.68 7.85 2.63 18.99 46.70 7.42
MAD-RAG 68.03 24.58 9.01 2.73 17.64 40.79 7.88
Standard RAG +BAIR 68.31 24.25 8.40 2.89 18.38 43.75 7.93
Visual-focus Instr. +BAIR 65.69 24.48 12.33 1.99 16.37 31.59 9.43
LongLLMLingua +BAIR 65.91 25.21 12.46 2.02 21.74 43.41 12.14
MS-PoE +BAIR 67.60 21.01 7.43 2.83 20.33 50.74 7.79
MAD-RAG +BAIR 68.25 24.19 8.45 2.86 18.77 44.02 8.22CheXagent-8BBaseline 65.51 - - - - - -
Standard RAG 64.56 15.88 10.36 1.53 - - -
Visual-focus Instr. 64.94 15.89 9.81 1.62 4.04 10.81 1.87
LongLLMLingua 65.33 23.63 13.63 1.73 21.85 38.43 15.13
MS-PoE 66.45 14.19 6.62 2.14 12.96 40.30 3.87
MAD-RAG 64.79 16.16 10.19 1.59 1.18 2.93 0.57
Standard RAG +BAIR 66.17 12.38 6.04 2.05 10.47 39.35 1.56
Visual-focus Instr. +BAIR 65.58 12.89 7.19 1.79 7.95 28.03 1.50
LongLLMLingua +BAIR 66.24 16.64 8.36 1.99 19.16 47.96 9.32
MS-PoE +BAIR 66.60 11.72 5.05 2.32 13.95 47.78 3.19
MAD-RAG +BAIR 66.08 11.83 5.87 2.02 11.73 42.35 2.24
Accuracy, suggesting that visual grounding is especially important when retrieved text can reinforce
demographic hallucinations. On the NWPU dataset, the gains are more moderate but still show the
same general pattern: BAIR reduces degradation while preserving the correction benefits of retrieval.
The reported standard deviations are small relative to the observed trends, suggesting that the
improvements are not driven by isolated unstable runs. These results also show why Accuracy alone
is insufficient for evaluating multimodal RAG reliability. By jointly analyzing Correction Rate and
Degradation Rate, we verify that BAIR improves the correction-degradation tradeoff, which is central
to reducing recorruption in multimodal RAG systems.
K Access to Data and Code
All datasets used in this work are publicly available research benchmarks. For the medical domain,
we use IU Chest X ray [ 5]. For the social fairness domain, we use FACET [ 7]. For the geospatial
domain, we use NWPU-RESISC45 [ 3]. The retrieval corpora are constructed from publicly available
reports or textual descriptions associated with these benchmarks, together with publicly available
contextual sources where applicable (e.g. Wikipedia).
All comparison methods are based on publicly available implementations or reproducible algorithmic
descriptions. We include the code necessary to reproduce our experiments, including scripts for
data preprocessing, retrieval context construction, model inference, intervention application, metric
computation, and figure generation. The released code also contains configuration files specifying
model checkpoints, prompt templates, intervention settings, and evaluation thresholds.
Table 8 summarizes the data and code resources used in this work. For the final public release, we will
include the exact access URL and license information from the official release page of each dataset,
model, and baseline implementation. Users of the released code are responsible for complying with
the original license terms of each resource.
20

Table 5: Comprehensive evaluation of multimodal RAG interventions on theFACETdataset. Metrics
are presented in percentages (%). Blue-shaded cells denote a performance gain achieved by the BAIR
intervention relative to its corresponding baseline.
Model Method Acc.↑CR↑DR↓CR/DR↑RR↑SR↑NR↑GFR↓
Qwen2.5-VL-3BBaseline 84.91 - - - - - - 0.00
Standard RAG 84.34 26.62 5.41 4.92 - - - 0.00
Visual-focus Instr. 84.38 28.25 5.65 5.00 16.59 31.54 10.39 0.00
LongLLMLingua 84.36 31.59 6.26 5.05 25.25 44.97 17.07 0.00
MS-PoE 82.00 29.07 8.60 3.38 20.07 37.14 12.99 2.24
MAD-RAG 84.41 25.53 5.11 5.00 13.86 31.22 6.71 0.00
Standard RAG +BAIR 88.56 50.31 4.64 10.84 43.28 54.81 38.50 0.02
Visual-focus Instr. +BAIR 87.28 44.93 5.20 8.64 38.82 54.14 32.47 0.04
LongLLMLingua +BAIR 87.45 45.97 4.89 9.40 42.09 62.22 34.07 0.05
MS-PoE +BAIR 92.58 70.71 3.53 20.03 68.79 76.96 65.40 0.63
MAD-RAG +BAIR 85.10 36.08 6.19 5.83 27.41 42.09 21.34 0.00DeepSeek-VL-7BBaseline 90.24 - - - - - - 0.00
Standard RAG 90.89 31.05 2.64 11.76 - - - 0.00
Visual-focus Instr. 90.47 28.53 2.83 10.08 7.67 17.67 4.12 0.00
LongLLMLingua 87.76 34.21 6.45 5.30 29.54 41.81 25.19 0.00
MS-PoE 91.82 51.05 3.78 13.51 51.07 72.41 43.51 0.03
MAD-RAG 90.86 30.84 2.65 11.64 12.40 25.86 7.63 -
Standard RAG +BAIR 91.02 30.95 2.48 12.48 11.61 23.71 7.33 0.00
Visual-focus Instr. +BAIR 90.81 30.84 2.71 11.38 10.60 21.55 6.72 0.00
LongLLMLingua +BAIR 87.80 33.79 6.36 5.31 29.76 43.10 25.04 -
MS-PoE +BAIR 91.94 51.58 3.70 13.94 51.97 72.84 44.58 0.00
MAD-RAG +BAIR 90.96 31.68 2.63 12.05 14.54 31.47 8.55 0.00
L Computation Resource
Experiments are conducted on an NVIDIA RTX A5000 GPU with an AMD EPYC 7313 CPU, as
summarized in Table 9. Because the proposed method is applied only at inference time and does
not require training or fine tuning, reproducing the main results does not require large scale training
resources.
M Broader Impact and Safeguard
This work studies a failure mode in multimodal retrieval augmented generation where retrieved textual
context can override correct visual evidence. The potential positive impact is improved reliability
for MLLMs deployed in settings where both images and external documents are used, including
medical image interpretation, remote sensing analysis, and fairness sensitive visual reasoning. By
identifying recorruption and proposing an inference-time mitigation, this work can help researchers
evaluate whether a multimodal RAG system is genuinely grounded in visual evidence or merely
copying retrieved text.
At the same time, this work should not be interpreted as making MLLMs safe for autonomous use
in high-stakes decision making. In medical settings, BAIR and the evaluated MLLMs should be
treated only as research tools or assistive systems, not as substitutes for clinical judgment. In social
fairness settings, the goal of our evaluation is to diagnose biased hallucination and visual grounding
failures, not to infer sensitive attributes for real individuals. For this reason, we use benchmark data
and synthetic fairness evaluation settings where applicable, and we avoid deploying the system on
private or identifiable individuals.
The proposed intervention also has limitations. Attention calibration can reduce certain forms of
textual distraction, but it cannot guarantee factual correctness, eliminate all hallucinations, or correct
errors caused by poor retrieval quality, ambiguous images, or incorrect model priors. Therefore, any
deployment should include human oversight, task-specific validation, privacy review, and domain-
specific safety constraints. We will release code and documentation with these limitations clearly
stated, along with scripts that allow users to inspect correction and degradation behavior rather than
relying only on aggregate accuracy.
21

Table 6: Comprehensive evaluation of multimodal RAG interventions on theNWPUdataset. Metrics
are presented in percentages (%). Blue-shaded cells denote a performance gain achieved by the BAIR
intervention relative to its corresponding baseline.
Model Method Acc.↑CR↑DR↓CR/DR↑RR↑SR↑NR↑GFR↓
SkySenseGPT-7BBaseline 68.90 - - - - - - 0.00
Standard RAG 65.74 44.68 24.75 1.81 - - - 0.36
Visual-focus Instr. 66.59 46.83 24.49 1.91 22.83 24.81 20.86 0.64
LongLLMLingua 69.97 46.50 19.43 2.39 51.35 70.38 32.49 0.00
MS-PoE 65.36 45.26 25.57 1.77 17.29 20.30 14.41 1.28
MAD-RAG 66.18 45.67 24.56 1.86 19.39 23.46 15.35 4.44
Standard RAG +BAIR 66.31 44.02 23.63 1.86 19.91 24.81 15.05 0.10
Visual-focus Instr. +BAIR 66.74 46.00 23.89 1.93 21.56 24.81 18.33 0.23
LongLLMLingua +BAIR 70.79 46.41 18.20 2.55 50.30 69.17 31.59 0.00
MS-PoE +BAIR 66.05 44.27 24.12 1.84 18.64 23.16 14.16 0.10
MAD-RAG +BAIR 67.13 46.17 23.41 1.97 19.16 23.31 15.05 0.10EarthDialBaseline 93.13 - - - - - - 0.00
Standard RAG 91.28 60.82 6.47 9.40 - - - 0.00
Visual-focus Instr. 92.51 63.81 5.37 11.88 29.41 35.47 15.24 0.03
LongLLMLingua 94.38 63.81 3.36 18.99 65.88 77.45 40.00 0.13
MS-PoE 94.69 69.03 3.41 20.24 60.29 68.51 41.90 0.10
MAD-RAG 91.64 62.69 6.22 10.08 21.76 24.68 15.24 0.00
Standard RAG +BAIR 91.72 59.70 5.92 10.08 19.12 24.26 7.62 0.00
Visual-focus Instr. +BAIR 92.62 62.69 5.18 12.10 29.71 36.60 14.29 0.03
LongLLMLingua +BAIR 94.49 66.04 3.41 19.37 67.94 78.72 43.81 0.00
MS-PoE +BAIR 96.74 75.37 1.68 44.86 76.18 85.96 54.29 0.18
MAD-RAG +BAIR 91.65 62.31 6.17 10.10 18.53 21.28 12.38 0.00
N LLM Acknowledgement
Large language models were used only as writing and programming assistance tools. Specifically,
they were used to polish manuscript wording, improve clarity, and assist with routine debugging of
experimental code. They were not used to generate experimental results, alter reported metrics, select
favorable outcomes, or replace author verification of scientific claims. All conceptual contributions,
experimental design choices, data analyses, and final manuscript decisions were made and verified by
the authors.
22

Table 7: Comprehensive evaluation of multimodal RAG interventions across all datasets. Metrics are
presented in percentages (%).
MedGemma-4B (IU-Chest) CheXagent-8B (IU-Chest)
Method Acc.↑CR↑DR↓Acc.↑CR↑DR↓
Baseline 63.76±0.72 - - 65.51±0.76 - -
Standard RAG 66.33±0.73 28.00±0.90 13.56±0.80 64.56±0.75 15.88±0.78 10.36±0.42
Visual-focus Instr. 65.38±0.74 26.68±0.91 14.13±0.82 64.94±0.78 15.89±0.80 9.81±0.45
LongLLMLingua 65.66±0.71 25.14±0.89 12.79±0.78 65.33±0.80 23.63±0.96 13.63±0.62
MS-PoE 67.17±0.75 20.68±0.92 7.85±0.84 66.45±0.74 14.19±0.76 6.62±0.38
MAD-RAG 68.03±0.72 24.58±0.90 9.01±0.81 64.79±0.77 16.16±0.82 10.19±0.46
Standard RAG +BAIR68.31±0.71 24.25±0.91 8.40±0.79 66.17±0.79 12.38±0.75 6.04±0.44
Visual-focus Instr. +BAIR65.69±0.73 24.48±0.89 12.33±0.83 65.58±0.76 12.89±0.83 7.19±0.48
LongLLMLingua +BAIR65.91±0.74 25.21±0.92 12.46±0.77 66.24±0.81 16.64±0.98 8.36±0.65
MS-PoE +BAIR67.60±0.72 21.01±0.90 7.43±0.85 66.60±0.75 11.72±0.74 5.05±0.37
MAD-RAG +BAIR68.25±0.70 24.19±0.91 8.45±0.82 66.08±0.78 11.83±0.84 5.87±0.49
Qwen2.5-VL-3B (FACET) DeekSeek-VL-7B (FACET)
Method Acc.↑CR↑DR↓Acc.↑CR↑DR↓
Baseline 84.91±0.40 - - 90.24±0.29 - -
Standard RAG 84.34±0.38 26.62±1.12 5.41±0.26 90.89±0.28 31.05±1.55 2.64±0.18
Visual-focus Instr. 84.38±0.42 28.25±1.15 5.65±0.28 90.47±0.30 28.53±1.60 2.83±0.20
LongLLMLingua 84.36±0.36 31.59±1.18 6.26±0.25 87.76±0.29 34.21±1.65 6.45±0.21
MS-PoE 82.00±0.45 29.07±1.10 8.60±0.30 91.82±0.28 51.05±1.52 3.78±0.17
MAD-RAG 84.41±0.39 25.53±1.14 5.11±0.27 90.86±0.30 30.84±1.58 2.65±0.19
Standard RAG +BAIR88.56±0.37 50.31±1.13 4.64±0.25 91.02±0.29 30.95±1.56 2.48±0.17
Visual-focus Instr. +BAIR87.28±0.41 44.93±1.16 5.20±0.29 90.81±0.28 30.84±1.61 2.71±0.19
LongLLMLingua +BAIR87.45±0.35 45.97±1.19 4.89±0.24 87.80±0.29 33.79±1.64 6.36±0.20
MS-PoE +BAIR92.58±0.44 70.71±1.11 3.53±0.28 91.94±0.29 51.58±1.53 3.70±0.16
MAD-RAG +BAIR85.10±0.38 36.08±1.15 6.19±0.26 90.96±0.30 31.68±1.59 2.63±0.18
SkySenseGPT-7B (NWPU) EarthDial (NWPU)
Method Acc.↑CR↑DR↓Acc.↑CR↑DR↓
Baseline 68.90±0.74 - - 93.13±0.41 - -
Standard RAG 65.74±0.76 44.68±1.44 24.75±0.84 91.28±0.45 60.82±3.00 6.47±0.41
Visual-focus Instr. 66.59±0.76 46.83±1.43 24.49±0.84 92.51±0.42 63.81±2.95 5.37±0.37
LongLLMLingua 69.97±0.73 46.50±1.42 19.43±0.77 94.38±0.37 63.81±3.02 3.36±0.30
MS-PoE 65.36±0.77 45.26±1.42 25.57±0.85 94.69±0.36 69.03±2.88 3.41±0.30
MAD-RAG 66.18±0.76 45.67±1.45 24.56±0.85 91.64±0.45 62.69±2.99 6.22±0.41
Standard RAG +BAIR66.31±0.76 44.02±1.42 23.63±0.83 91.72±0.45 59.70±3.04 5.92±0.40
Visual-focus Instr. +BAIR66.74±0.76 46.00±1.43 23.89±0.84 92.62±0.42 62.69±2.96 5.18±0.37
LongLLMLingua +BAIR70.79±0.73 46.41±1.42 18.20±0.76 94.49±0.37 66.04±2.93 3.41±0.30
MS-PoE +BAIR66.05±0.76 44.27±1.43 24.12±0.84 96.74±0.28 75.37±2.66 1.68±0.21
MAD-RAG +BAIR67.13±0.75 46.17±1.43 23.41±0.82 91.64±0.45 62.31±3.01 6.19±0.41
Table 8:Data, model, and code resources.We use publicly available datasets, model checkpoints,
and baseline implementations. The final release will include the exact license field from each official
source.
Resource Role in this work Availability License
IU Chest X ray [5] Medical factuality benchmark Public dataset CC BY-NC-ND 4.0
FACET [7] Social fairness benchmark Public dataset Research-only
NWPU-RESISC45 [3] Remote sensing benchmark Public dataset CC BY 4.0
MedGemma [20] Medical MLLM Public model/checkpoint health-ai-developer-foundations
CheXagent [2] Medical MLLM Public model/checkpoint MIT license
SRR-BERT [4] Medical factuality evaluation model Public model/checkpoint MIT license
Qwen2.5-VL [23] General MLLM Public model/checkpoint Apache License
DeepSeek-VL [15] General MLLM Public model/checkpoint deepseek
SkySenseGPT [16] Remote sensing MLLM Public model/checkpoint Apache-2.0 license
EarthDial [21] Remote sensing MLLM Public model/checkpoint To be reported from official source
MedSigLIP [20] Medical retrieval model Public model/checkpoint health-ai-developer-foundations
RemoteCLIP [13] Remote sensing retrieval model Public model/checkpoint Apache-2.0 license
MS-PoE [28] Comparison method Public implementation MIT license
LongLLMLingua [9] Comparison method Public model/checkpoint and implementation MIT license
MAD-RAG [29] Comparison method Reproducible method -
BAIR code Proposed method and evaluation scripts Included in supplementary material To be specified upon release
Table 9: Compute Resources Used for Experiments
Component Details
CPU AMD EPYC 7313 16-Core Processor
GPU NVIDIA RTX A5000
23