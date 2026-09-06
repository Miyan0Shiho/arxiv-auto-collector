# Same Semantics, Different Outcome: On the Modality Robustness of Multimodal LLMs under Knowledge Conflict

**Authors**: Jungyeon Lee, Yejin Yoon, Taeuk Kim

**Published**: 2026-09-01 01:37:18

**PDF URL**: [https://arxiv.org/pdf/2609.00550v1](https://arxiv.org/pdf/2609.00550v1)

## Abstract
Multimodal large language models (MLLMs) are increasingly provided with contextual evidence in heterogeneous forms: as a text passage, as a rendered image of the same passage, or as both together. However, it remains unclear how consistently these surface forms are processed, especially when the evidence conflicts with the model's parametric knowledge. We study modality robustness under knowledge conflict across 13 MLLMs and two datasets, and find them far from robust. (1) Contrary to common belief, models favor a context that contradicts parametric knowledge more readily in image form than in text form; (2) when a contradicting text and image are presented together, the preferred modality is essentially arbitrary, varying with input order, model, and dataset. We further demonstrate that this instability has practical consequences: it degrades performance in multimodal RAG and can be exploited by adversarial attacks. To alleviate this brittleness, we examine several simple techniques---prompting, steering, supervised fine-tuning (SFT), and direct preference optimization; the majority prove ineffective, whereas SFT achieves moderate success. We therefore call for greater awareness of this inconsistency and argue that it is fundamental, demanding attention at multiple training stages.

## Full Text


<!-- PDF content starts -->

Same Semantics, Different Outcome: On the Modality Robustness of
Multimodal LLMs under Knowledge Conflict
Jungyeon Lee, Yejin Yoon, Taeuk Kim*
Hanyang University, Seoul, Republic of Korea
{jungyune,stillwithyou,kimtaeuk}@hanyang.ac.kr
Abstract
Multimodal large language models (MLLMs)
are increasingly provided with contextual evi-
dence in heterogeneous forms: as a text pas-
sage, as a rendered image of the same pas-
sage, or as both together. However, it remains
unclear how consistently these surface forms
are processed, especially when the evidence
conflicts with the model’s parametric knowl-
edge. We study modality robustness under
knowledge conflict across 13 MLLMs and two
datasets, and find them far from robust. (1)
Contrary to common belief, models favor a
context that contradicts parametric knowledge
more readily in image form than in text form;
(2) when a contradicting text and image are
presented together, the preferred modality is
essentially arbitrary, varying with input order,
model, and dataset. We further demonstrate
that this instability has practical consequences:
it degrades performance in multimodal RAG
and can be exploited by adversarial attacks. To
alleviate this brittleness, we examine several
simple techniques—prompting, steering, super-
vised fine-tuning (SFT), and direct preference
optimization; the majority prove ineffective,
whereas SFT achieves moderate success. We
therefore call for greater awareness of this in-
consistency and argue that it is fundamental,
demanding attention at multiple training stages.
1 Introduction
Multimodal large language models (MLLMs) are
increasingly deployed in retrieval-augmented and
agentic workflows, where evidence arrives in a va-
riety of surface forms, e.g., plain text, webpages,
PDFs, screenshots, or scanned documents (Yu et al.,
2025; Faysse et al., 2025). These workflows presup-
pose that MLLMs process semantically equivalent
content consistently, irrespective of its modality.
That expectation is not guaranteed to hold, however,
*Corresponding author.because inputs in different modalities typically fol-
low distinct internal pathways through these mod-
els, beginning with modality-specific tokenization.
The risk posed by such latent inconsistency is
greatly amplified underknowledge conflict(KC),
where external evidence contradicts the model’s
parametric knowledge or where multiple sources
support competing answers. In such scenarios, the
answer may diverge drastically, or even reverse,
according to which source the model accepts as
authoritative; if that acceptance is governed by the
form in which the evidence is delivered rather than
by its content, the robustness and reliability of the
model are severely compromised.
Although previous work (Deng et al., 2025; Sim
et al., 2025; Zhang et al., 2025a) has reported that
MLLMs exhibit a modality bias—typically placing
more weight on textual than visual information—
this claim has not been examined in knowledge-
conflict scenarios. Moreover, the few studies on
multimodal knowledge conflict (Hua et al., 2025;
Nguyen et al., 2025; Zhang et al., 2025b,c) focus
mostly on natural images rather than documents,
even though knowledge is typically organized in
document form, not as scenes or faces.
To fill this gap, we investigate themodality ro-
bustnessof MLLMs through the lens of knowledge
conflict. We evaluate two setups: (1) a single piece
of external evidence is delivered as text or as a ren-
dered image (i.e., text-as-image); (2) two textual
and visual contexts conflict with each other. Un-
like prior work, we also guarantee that parametric
knowledge is distinct from any contextual knowl-
edge, enabling more tightly controlled analysis.
Spanning multiple MLLMs, datasets, and con-
flict categories, the dominant pattern we observe is
instability: which modality a model trusts shifts un-
predictably with input order, dataset construction,
and model family, and no single factor reliably ex-
plains the variation. Crucially, this contradicts the
widely reported textual bias in MLLMs. In our
arXiv:2609.00550v1  [cs.CL]  1 Sep 2026

single-evidence setting, the direction is in fact re-
versed: when the image is a rendering of the same
text, MLLMs tend to prefer it over the text itself.
Moreover, even this reversed preference dissolves
once both modalities are presented in conflict, a set-
ting that prior work has not thoroughly examined.
We further examine whether this sensitivity has
practical consequences beyond controlled configu-
rations, using two downstream tasks. First, in mul-
timodal RAG, we keep a passage’s content fixed
and vary only its format—raw text or rendered
image—to test whether that change alone alters
how models use retrieved evidence (§5.1). We find
that MLLMs are less distracted by irrelevant con-
text when salient knowledge is represented as an
image rather than text, suggesting a remedy for the
lost-in-the-middle phenomenon (Liu et al., 2024a).
Second, we examine safety-critical settings, in
which the surface form of a harmful request may
affect the model’s refusal behavior (§5.2). Our re-
sults indicate that rendering harmful requests as
images increases attack success rates by 6.99 per-
centage points on average, underscoring a serious
and readily exploitable vulnerability.
Finally, we explore methods to improve the
modality robustness of MLLMs. We observe
that simple prompting is insufficient; instead, fine-
tuning on knowledge-conflict data partially miti-
gates the problem, although it does not completely
eliminate the inconsistency (§6). Consequently, we
call for greater community awareness of MLLMs’
inconsistency across input modalities, and argue
that this problem is fundamental, requiring further
investigation at different stages of model training.
To summarize, our contributions are as follows:
•We formalize modality robustness as an evidence-
authority problem under knowledge conflict.
•We construct single- and multi-evidence conflict
settings that disentangle semantic content, pre-
sentation modality, and parametric knowledge.
•Across 13 MLLMs and two datasets, modality re-
liance varies across model families, input orders,
and evidence compositions, with downstream ef-
fects on multimodal RAG and refusal behavior.
•Comparing several mitigation strategies, we find
conflict-aware fine-tuning to be the most effec-
tive at balancing modality preference.2 Related Work
2.1 Knowledge Conflict
Knowledge conflict has been studied as the tension
between parametric memory and external evidence,
formalized via entity substitution (Longpre et al.,
2021) and later expanded along multiple conflict
axes (Xu et al., 2024; Wang et al., 2024). While
these works characterize how models resolve such
conflicts (Jin et al., 2024; Khandelwal et al., 2025),
they generally assume a fixed modality, i.e., text.
The framework has been recently extended to
multimodal settings where context is also presented
as images (Zhang et al., 2025c; Nguyen et al., 2025;
Zhang et al., 2025b; Hua et al., 2025), but does not
explicitly control for conflicts with the model’s
parametric memory. Other studies test parametric
or commonsense knowledge with counterfactual
images, observing whether models follow the vi-
sual input or their internal beliefs (Liu et al., 2025;
Ortu et al., 2025). A related line examines incon-
sistencies between visual entities or attributes and
their textual descriptions (Jia et al., 2026). Across
these studies, however, the evidence is predom-
inantly natural images such as scenes or faces,
rather than the knowledge-intensive sentences and
paragraphs—presented either as text or as rendered
images—that constitute the focus of our work.
2.2 Modality Bias in MLLMs
Recent work reports a“Blind Faith in Text”ten-
dency in MLLMs, where models favor text over
visual evidence (Deng et al., 2025). Counterfactual-
image studies similarly show reliance on language
priors over visual cues (Lee et al., 2025), and oth-
ers report text-dominant behavior under controlled
evidence conflicts (Zhang et al., 2025a). A recent
survey synthesizes these findings under the notion
of modality collapse (Sim et al., 2025).
We extend these efforts to knowledge-conflict
scenarios, where consistent handling of input con-
text is essential to the robust behavior of MLLMs.
We later show that the textual bias widely reported
in the literature does not hold in our setting; the
direction in fact often reverses, with MLLMs fa-
voring images over text. Furthermore, this trend
grows more arbitrary as multiple pieces of evidence
simultaneously contradict parametric knowledge,
urging a reassessment of the current consensus.

MLLMMLLMQuery 𝑞: Who wrote the play "Hamlet"?
🧠𝑎𝑖𝑛𝑡: Shakespeare(a) Single-evidence conflict"Hamlet was written by Marlowe."Shakespeare
🧠Marlowe
🖼Same content, different deliveryàdifferent answer.(b) Multi-evidence conflict
Same content, order flipped à which modality "wins" flips.(	
🔤	,		
🖼	)(	
🖼	,	
🔤	)Marlowe
🖼Bacon
🔤
🔤
🖼
🔤 “Bacon” 
🖼 Hamlet	was	written	by	Marlowe.MarloweFigure 1:The same evidence, delivered differently, can yield different answers.(a) Single-evidence: the model
follows visual external evidence eI(
) but reverts to its memory aint(
) when the same content is delivered as
texteT(
).(b) Multi-evidence: when eTandeIsupport different answers, the order in which they appear can flip
which modality the model follows. These examples illustrate a broader instability of MLLMs under knowledge
conflict: the information a model prioritizes varies arbitrarily with input order, dataset, and model family.
3 Problem Formulation
In this work, we studymodality robustness under
knowledge conflict (KC), namely, whether MLLMs
resolve knowledge conflicts consistently across dif-
ferent numbers and orderings of textual and visual
inputs. Let pdenote the model’s internal paramet-
ric knowledge and ethe external evidence for query
q. We specify two competing answer candidates:
aintsupported by pandaextsupported by e, with
aint̸=a ext. To isolate modality, we hold the se-
mantic content of the evidence fixed and present
it either as text eTor as an image eI. Since this
work focuses on knowledge-intensive scenarios, eI
is defined as a rendered image of eT.1If the model
produces different responses acrosse Tande I, we
regard this as a failure of modality robustness.
3.1 Categories of Knowledge Conflict
Previous work typically focuses on either single-
or two-evidence settings, often without carefully
controlling for parametric knowledge. Our setup
covers both configurations and explicitly accounts
for parametric alignment. Figure 1 illustrates the
model’s unstable responses in both settings, driven
by inconsistent multimodal processing under KC.
Single-evidence conflictFor each query q, we
pair it with external evidence ethat contradicts p,
instantiated in two modalities: textual eTand im-
ageeI. Both support the same answer aext, conflict-
ing witha int. We compare two input conditions:
xT= (e T, q)vs.x I= (e I, q).
This measures how the modality of evidence af-
fects the model’s resolution of conflicts between
parametric (p) and external (e) knowledge.
1The rendering procedure is provided in Appendix A.1.Specifically, we use KC datasets (§3.2) with 5-
tuples (q, c+, a+, c−
1, a−
1): a query q, a factual con-
textc+with answer a+, and a counterfactual con-
textc−
1with answer a−
1. For each q, we determine
whether aintmatches a+ora−, and align (c+, a+)
and(c−
1, a−
1)with(p, a int)and(e, a ext).2
To this end, we leverage a closed-book multiple-
choice QA setup. The answer choices are taken
from the source KC dataset and augmented with
a‘none of the above’option. Inspired by Wang
et al. (2023), we sample the model’s output for q
five times and retain qonly when all five responses
agree, taking the common answer as aint.3Ifaint
is‘none of the above’, we ask the model to provide
an open-ended answer, which then replaces aint.
Subsequently, if aint=a+, we align (p, a int)and
(e, a ext)with(c+, a+)and(c−
1, a−
1), respectively;
otherwise, the alignment is reversed.
As LLMs differ in their parametric knowledge,
the number of retained instances ( |D|) after this
process varies across models. Model-wise filtering
statistics are reported in Table 6 of Appendix A.2.
Multi-evidence conflictWe further consider a
joint setting that provides both modalities to the
model, in two possible orderings:
xI→T= (e I, eT, q)andx T→I= (e T, eI, q).
We extend the setup to a three-way conflict by
adding a second conflicting source, yielding 7-
tuples (q, c+, a+, c−
1, a−
1, c−
2, a−
2). For simplicity,
we focus on cases where (p, a int)matches the
2Prior work on multimodal KC (Ortu et al., 2025) either
skipped this verification or assumed parametric memory is
always correct, leaving their findings potentially confounded.
3Unlike Wang et al. (2023), we require unanimous agree-
ment across samples rather than majority voting, yielding a
stricter estimate of internal knowledge.

factual pair (c+, a+). We then have two possi-
ble assignments: (1) (eT, aT) = (c−
1, a−
1)and
(eI, aI) = (c−
2, a−
2); (2) the reverse. We aver-
age over both assignments and test whether the
model favors one modality in this three-way con-
flict. Again, model-wise final dataset sizes ( |D|)
are reported in Table 7 of Appendix A.2.
3.2 Other Experimental Factors
We consider additional experimental factors that
may affect modality-dependent evidence reliance.
DatasetsWe evaluate on two complementary
knowledge conflict datasets:
•CONFLICTQA(Xie et al., 2024) constructs con-
flicts by eliciting the model’s parametric memory
and then generating counter-memory evidence
supporting an alternative answer. Since the ev-
idence is LLM-generated rather than produced
by simple word substitution (as in NQ-SWAP),
the resulting passages read naturally.4
•NQ-SWAP(Longpre et al., 2021) is a variant
of the Natural Questions dataset (Kwiatkowski
et al., 2019) where the answer entity is replaced
with another entity in several substitution meth-
ods. Since NQ-SWAPis based on real QA pas-
sages, it provides a retrieval-like conflict setting.
ModelsWe evaluate 13 MLLMs across propri-
etary API models and open-source model families:
•Proprietary models: Claude Sonnet 4.5 (An-
thropic, 2025), GPT-5.4 (OpenAI, 2026), and
GPT-4o (Hurst et al., 2024).
•Open-source omni-modal LLMs (OLLMs):
Qwen3-Omni-30B-A3B-Instruct (Xu et al.,
2025b), Qwen2.5-Omni-7B (Xu et al., 2025a),
MiniCPM-o2.6 (OpenBMB, 2025), and Om-
niVinci (Ye et al., 2025).
•Open-source vision-language models (VLMs):
Qwen2.5-VL-7B/-32B-Instruct (Bai et al., 2025),
InternVL3-8B/-38B-Instruct (Zhu et al., 2025),
LLaV A-v1.6-34b-hf, and LLaV A-v1.6-vicuna-
7b-hf (Liu et al., 2023).
We test whether modality robustness under KC
generalizes across model families and scales.
4We use the publicly available GPT-4-generated
PopQA (Mallen et al., 2023) counterfactuals for (c−
1, a−
1)and
the ChatGPT-generated variant for(c−
2, a−
2)or vice versa.MetricsFor i∈ D , we have distinct forms of
prompts {x(i)}, e.g., x(i)
Tandx(i)
I→T. We request
an LLM to generate its open-ended response given
a specific type of xthree times, and then evaluate
them using normalized span matching.5
Letˆa(i)
x={ˆa(i)
x,1,ˆa(i)
x,2,ˆa(i)
x,3}denote a set of three
responses for instance iunder prompt x. We define
theexternal-source following rate (EFR)over the
datasetDfrom §3.1 as
EFR(x)=1
|D|X
i∈DIh
ˆa(i)
x,1= ˆa(i)
x,2= ˆa(i)
x,3=a(i)
exti
,
where Ih
ˆa(i)
x,1= ˆa(i)
x,2= ˆa(i)
x,3=a(i)
exti
becomes 1
only when all three responses match the external
answer candidate a(i)
ext, and 0 otherwise. By design,
this definition is conservative: it credits a model
only for consistently following the evidence, not
for a single match that could arise by chance.
We quantify modality reliance using the normal-
izedimage-over-text gap, denoted by G. Since
models differ in their overallEFR, we normalize
the image–text difference by their combined rate to
enable stable cross-model comparison. For single-
evidence conflict, we compare two prompts that
differ only in the modality of the same external
evidence:x Tandx I. We thus computeGas
Gsingle=EFR(x I)−EFR(x T)
EFR(x I) + EFR(x T)×100.
In the multi-evidence setting, both eTandeI
are presented jointly, supporting competing an-
swers aTandaI. For each ordering o∈ {I→
T, T→I} , we compute the modality-specific rates
EFR T(xo)andEFR I(xo), and define the (order-
aware) image-over-text gap as
Gmulti
o=EFR I(xo)−EFR T(xo)
EFR I(xo) + EFR T(xo)×100.
The resulting gap is bounded within [−100,100] .
Positive values indicate that the model relies more
on the image, with larger magnitudes correspond-
ing to a wider image–text difference.
4 MLLMs Lack Modality Robustness
Table 1 summarizes the normalized image-over-
text gap Gacross models, datasets, input orders,
5We use normalized span matching to relax exact match-
ing: after lowercasing, punctuation/whitespace cleanup, and
alias handling, ˆa(i)
xcounts as a match if either the normalized
prediction containsa(i)
extor vice versa.

ModelEvidence CONFLICTQA NQ-SWAP
Single Multix I→T Multix T→I Single Multix I→T Multix T→I
Claude Sonnet 4.5 42.86 79.41 84.62 22.63 4.26 19.38
GPT-5.4 15.29 1.80 63.37 3.80 4.38 6.92
GPT-4o 9.75 4.40 −19.05 −14.74 6.89 4.80
Qwen3-Omni 1.64 −38.24 79.25 12.98 −55.51 10.94
Qwen2.5-Omni 1.54 −42.64 43.77 6.74 −5.64 33.50
MiniCPM-o2.6 1.09 5.42 26.36 1.86 −26.18 −42.27
OmniVinci −1.81 −64.66 86.58 −2.87 0.94 −0.77
Qwen2.5-VL-32B 2.32 −19.37 65.61 5.49 −11.87 16.96
Qwen2.5-VL-7B 4.09 −68.98 73.82 3.80 33.10 29.51
InternVL3-38B 0.46 −21.32 84.87 8.53 −40.04 42.92
InternVL3-8B −1.27 −93.03 58.77 3.44 −47.25 21.58
LLaV A-v1.6-34B −5.34 −81.64 −85.65 −9.75 −31.38 −43.28
LLaV A-vicuna-7B −9.17 −95.45 −94.86 −34.17 −78.76 −75.67
Table 1: Normalized image-over-text gap (%) on CONFLICTQA and NQ-SWAPacross single- and multi-evidence
settings; Gsinglefor single-evidence inputs, Gmulti
o for multi-evidence inputs under each order o∈ {I→T, T→I} .
Cell shading scales with the magnitude of the change ( ■positive / ■negative; bins at |x|≤10,10 –30,30 –60, >60 ).
and conflict categories; the corresponding absolute
percentage-point gaps are reported in Table 9 in
the Appendix. The dominant pattern isinstabil-
ity: modality is not merely a passive container of
evidence, but a factor that influences how models
resolve conflicts.
Single-evidence conflict: a weak but pervasive
image preference.The single-evidence setting
is the most direct test of presentation invariance:
the external evidence has the same semantic con-
tent across conditions, and only its surface form
changes. Positive Gsingletherefore means that ren-
dering the same evidence as an image makes the
model more likely to follow it over its paramet-
ric knowledge. This image advantage is most ev-
ident in proprietary models. Claude Sonnet 4.5
shows the strongest positive gaps, with +42.86% on
CONFLICTQA and +22.63% on NQ-SWAP. GPT-
5.4 also remains positive on both datasets, with
+15.29% and +3.80%, respectively. The pattern is
not universal, however: GPT-4o reverses from a
positive gap on CONFLICTQA (+9.75%) to a nega-
tive gap on NQ-SWAP(-14.74%), while the LLaV A
models show consistently negative gaps, indicating
stronger reliance on text evidence. These results
show that even the simplest presentation change
can alter evidence following.
Multi-evidence conflict: modality order matters.
Table 1 shows that when eIprecedes eT(xI→T),
most open-source models show negative gains, in-
dicating stronger reliance on the text-supportedanswers. However, when the order is reversed
(xT→I), the same models often shift toward image-
supported answers. This order sensitivity is re-
flected in both the per-modelFlip Ratioand the
aggregate number of sign flips, as shown in Table
8 in the Appendix. TheFlip Ratiofurther shows
instance-level preference changes when the input
order is reversed. At the aggregate level, 8 out of 13
models on CONFLICTQA and 6 out of 13 models
on NQ-SWAPchange the sign of Gacross the two
input orders. These reversals show that modality
reliance is not fixed; it can be reshaped by the order
in which conflicting sources enter the context.
Gis also influenced by the dataset.As Table 1
shows in the multi-evidence settings, CONFLIC-
TQA elicits markedly larger absolute gaps than
NQ-SWAP; OmniVinci, for example, exceeds 86%
on CONFLICTQA but is nearly neutral on NQ-
SWAP. We hypothesize that modality-dependent
reliance grows with the semantic divergence be-
tween conflicting sources. NQ-SWAPperturbs
only the answer entity and otherwise leaves the
passage intact, so the two evidence sources remain
largely overlapping; CONFLICTQA, by contrast,
generates entirely separate counter-evidence, pro-
ducing a much broader semantic gap. Figure 6 in
Appendix A.2 supports this characterization.
Variation across model families.Figure 2 vi-
sualizes Gfrom Table 1 by grouping open-source
models by family and proprietary models into a
separate category. The proprietary group main-

−100 −50 0 50 100Proprietary
Qwen
InternVL
LLaVA
MiniCPM
OmniVinciText preference Image preferenceConflictQA
−60 −30 0 30 60Text preference Image preferenceNQ-Swap
Normalized image-over-text gap G (%)Single T→I I→TFigure 2: Group-averaged modality preference on CON-
FLICTQA and NQ-SWAP. Positive and negative values
indicate image and text preference, respectively.
tains the most consistently positive average G,
indicating a relatively stable image preference.
In contrast, LLaV A models remain strongly text-
dominant across all conditions and datasets. Qwen
and InternVL families exhibit pronounced order
sensitivity: their average Gbecomes strongly pos-
itive under xT→I but negative under xI→T, espe-
cially on CONFLICTQA. Together, these patterns
indicate that modality-dependent reliance can be
influenced by model characteristics and evidence
conditions, rather than reflecting a universal prop-
erty of MLLMs.
5 When Modality Instability Matters
The instability documented in §4 is not merely a
benchmark artifact. Deployed MLLM pipelines
rarely control how evidence arrives. Even when
the underlying evidence remains identical, models
may resolve conflicts differently due to upstream
choices in evidence representation.
We examine two settings where this failure mode
is consequential: question-answering performance
in multimodal RAG (§5.1), where format changes
what the model attends to, and safety vulnerability
to image-rendered adversarial instructions (§5.2).
5.1 Impact of Gold-Passage Modality in RAG
To probe whether modality instability matters in
practice, we begin with retrieval-augmented QA.
As a thought experiment, imagine that an ideal
paragraph directly answering the query is given in
advance: how do MLLMs treat this gold-standard
passage when only its modality changes? We fix
the query, the gold passage’s content, and the dis-
tractors (all provided as text for simplicity), and
vary only the modality of the gold passage.
We sample 170 query–passage instances from
MS MARCO v2.1 (Bajaj et al., 2016) and report
4050
Avg. Gap: +2.4Gemini 2.5 Pro
5060
Avg. Gap: +3.7GPT-4o
Gold-only First Middle Last405060
 Avg. Gap: +7.5Qwen3-Omni
Gold-only First Middle Last4060
Avg. Gap: +9.7LLaVA-v1.6-34BROUGE-LText gold passage Image gold passageFigure 3: RAG performance on MS MARCO v2.1 with
text vs. image gold passages.Gold-only: gold alone;
First/Middle/Last: gold position among nine distractors.
Avg. Gap: the mean image–text ROUGE-L difference.
answer quality with ROUGE-L. MLLMs are evalu-
ated under four cases: agold-onlysetting, in which
the gold passage is the only input, and threemulti-
passagesettings, in which it appears among nine
distractors at theFirst,Middle, orLastposition.
Figure 3 demonstrates that the modality of
the gold-standard passage influences RAG perfor-
mance, with image rendering mitigating the well-
known lost-in-the-middle phenomenon (Liu et al.,
2024a). For instance, Gemini 2.5 Pro gains +6.59
ROUGE-L when the image-rendered gold passage
occupies the middle position. This suggests that
rendering relevant context as an image may serve
as a new strategy to direct model attention toward
salient passages embedded among distractors.
5.2 Vulnerability to Image-Rendered Attacks
As a second case study, we ask whether modality
instability also affects MLLMs’ safety. Our ex-
periments build on MM-SafetyBench (Liu et al.,
2024b), which benchmarks robustness against at-
tacks that hide harmful content in query-relevant
images. Its four conditions—Text-Only, Stable Dif-
fusion (SD), Typography (Typo), and SD+Typo—
serve as our baselines and embed only the key
phrase in image form.6We extend this setup with
animage-rendered instructioncondition, in which
the full instruction is moved into the visual channel,
leaving the request semantically unchanged, and
report Attack Success Rate (ASR), the proportion
of prompts for which the model produces harmful
content instead of refusing.
Figure 4 summarizes the increase in ASR from
rendering harmful instructions as images across
models and baseline conditions, averaged over all
6SD generates images from prompts, while Typo overlays
the key phrase on a white background; see Appendix G.

T ext-
onlySD Typo SD+
TypoQwen2.5
InternVL3
LLaVA-v1.6
Avg.+8.45 +2.12 +3.12 +1.29
+12.44 +6.38 +7.24 +5.64
+13.92 +6.41 +6.16 +10.71
+11.60 +4.97 +5.51 +5.88
02468101214
 ASR
Figure 4: Mean ∆ASR over 13 MM-SafetyBench sce-
narios when the harmful instruction is rendered as an
image. Cells report the points relative to each baseline
(Text-Only, SD, Typo, SD+Typo); darker shading in-
dicates larger increases. Avg. averages over the three
models; per-scenario values in Appendix G.
13 scenarios. Over 156 evaluation settings in to-
tal, this intervention increases ASR by 6.99% on
average, with positive ∆ASR in 126/156 (80.8%)
comparisons. Table 15 in the Appendix reports the
corresponding per-scenario results. The average
increase is consistently positive across all model ×
baseline cells, although its magnitude varies across
models (Qwen +3.74%, InternVL +7.93%, LLaV A
+9.30%). It is largest against the Text-only baseline
(+11.60%) and smallest against SD (+4.97%). To-
gether, these results suggest that moving the whole
instruction into the visual channel introduces an ad-
ditional safety vulnerability beyond existing attacks
that embed only parts of the instruction visually.
6 Mitigating Modality Instability
Having established the modality instability of
MLLMs and its practical consequences, we next
explore whether this behavior can be mitigated. We
consider interventions at three levels: a lightweight
intervention through explicit prompting, post-
training through supervised fine-tuning and direct
preference optimization, and inference-time con-
trol through representation steering.
6.1 Prompting is Not Sufficient
A natural first step is to instruct the model directly
to use both modalities in balance. We therefore
prepend a system-level prompt asking the model to
jointly consider textual and visual evidence when-
ever multiple sources are provided. Under the
hypothesis that underspecified instructions drive
the imbalance, this explicit multimodal prompting
should close the gap.
Contrary to expectation, as shown in Table 2,
prompting does not provide reliable control over
modality reliance. While it partially reduces the im-Multi I→T Multi T→I
Model Orig. +Prompt. Orig. +Prompt.
GPT-5.4 1.80 71.26 63.37 80.49
Qwen2.5-VL -68.97 -17.54 73.82 45.69
Table 2: Image-over-text gap ( Gmulti
o, %) before and
after explicit prompting on the CONFLICTQA subset in
the multi-evidence setting. Values closer to zero indicate
more balanced reliance on textual and visual evidence.
balance for Qwen2.5-VL-7B, it instead amplifies
the bias for GPT-5.4. We observe similar instability
when the prompt explicitly asks the model to focus
on a particular modality. These outcomes suggest
that modality-dependent reliance cannot be reliably
corrected through surface-level prompting alone,
motivating the post-training and inference-time in-
terventions explored next.
6.2 Conflict-Aware Fine-Tuning
We here examine whether this modality-dependent
reliance can be recalibrated through post-training.
Unlike prompting, which intervenes only through
instructions at inference time, fine-tuning directly
adjusts the model’s behavior under conflicting mul-
timodal evidence.
To test whether reliance on misleading visual
evidence can be recalibrated in a reproducible open-
source setting, we select Qwen2.5-VL-7B-Instruct,
which exhibits strong image-following behavior.
We then apply conflict-aware LoRA fine-tuning
using training data constructed from NQ-SWAP
and MMM-Fact (Xu et al., 2025c).7This tuning
objective reduces the model’s tendency to follow
misleading image evidence while preserving its use
of factual evidence during conflict resolution.
As shown in Table 3, conflict-aware SFT consis-
tently reduces the magnitude of the modality gap
across all evaluated settings. The effect is particu-
larly pronounced in the multi-evidence xT→I con-
dition, where the image-dominant gap decreases
from 73.82% to 52.04%. The opposite order shows
a modest reduction in the text-dominant gap. The
intervention also reduces modality imbalance in
the single-evidence setting for both misleading (-
2.29%) and reliable (-2.37%) evidence.
Importantly, moving Gtoward zero does not by
7For NQ-SWAP, we train the model to produce the ground-
truth answer despite a counterfactual image-rendered passage.
For MMM-Fact, we use non-supporting image-description
pairs, where the image and the textual description disagree, to
encourage cross-modal comparison. See Appendix C.1.

Evidence False True
MethodMulti I→T Multi T→I Single Single
Original -68.97 73.82 2.54 3.45
+ SFT -67.07 52.04 0.25 1.08
Table 3: Effect of conflict-aware fine-tuning on
Qwen2.5-VL-7B on CONFLICTQA. False and True de-
note misleading and reliable evidence, respectively.
itself guarantee desirable evidence use, since bal-
anced modality reliance could in principle arise
from uniformly increasing or decreasing evidence
following. We therefore additionally examine
the corresponding raw EFR Iin Appendix C.1.
Table 12 shows that SFT reduces following of
misleading image evidence while preserving, and
slightly increasing, following of reliable image
evidence. These findings provide evidence that
conflict-aware SFT can improve modality balance
without merely suppressing visual evidence or shift-
ing preference toward the opposite modality.
6.3 Additional Mitigation Baselines
To broaden our mitigation analysis beyond prompt-
ing and conflict-aware SFT, we additionally eval-
uate two approaches operating at different stages:
Direct Preference Optimization (DPO) (Rafailov
et al., 2023) as a post-training method and
representation-level steering as an inference-time
intervention. Detailed implementation settings and
the corresponding raw image-following rates are
provided in Appendix C.2 and C.3.
Preference-based post-training.We construct
NQ-SWAPpreference pairs that favor the origi-
nal factual answer over the answer supported by
the substituted misleading image, and use them
to train Qwen2.5-VL-7B-Instruct with DPO. Ta-
ble 4 reports that DPO slightly reduces the image-
dominant gap under xT→I, but substantially in-
creases the text-dominant gap under xI→T. Thus,
DPO can shift reliance away from misleading im-
age evidence in some cases, but does not improve
modality balance across different input orders.
Representation-level steering.We further eval-
uate an inference-time steering baseline follow-
ing prior work (Zhang et al., 2025a) on the same
Qwen2.5-VL-7B model. Using a calibration sub-
set of CONFLICTQA, we derive separate text-
following steering directions for each input order
and apply them during decoding on a held-out eval-MethodMulti I→T Multi T→I
Preference-based post-training
Original -68.89 68.47
+ DPO -88.08 65.57
Representation-level steering
Original (Scale 0) -38.64 77.77
Scale 1 -45.05 80.64
Scale 3 -70.21 64.95
Scale 5 -86.96 60.00
Table 4: Additional mitigation interventions on multi-
evidence conflicts. Positive and negative Gmulti
o indicate
image- and text-dominant reliance, respectively.
uation subset. As the steering strength increases,
the image-dominant gap under xT→I decreases
from 77.77% to 60.00%. In contrast, the text-
dominant gap under xI→T grows. At stronger steer-
ing scales, modality preference shifts toward text
under both input orders.
Overall, both methods exhibit a directional trade-
off: reducing image-dominant behavior can coin-
cide with stronger text-dominant behavior. This
contrasts with conflict-aware SFT, which moves ev-
eryGcloser to zero shown in Table 3. These obser-
vations highlight an important distinction between
controlling modality preference and improving
modality robustness; effective mitigation should
reduce imbalance in both directions rather than
simply shifting reliance toward a fixed modality.
7Further Analysis: Input-Side Sensitivity
Given that input processing is the stage where
modality-specific handling is most pronounced, we
ask whether input-side interventions—altering how
visual evidence is preprocessed and rendered—can
change a model’s modality reliance. We examine
this with frozen model weights, intervening solely
on the visual preprocessing pipeline and the ren-
dered form of semantically identical evidence.
7.1 Influence of Visual Preprocessing
The results in §4 show that modality reliance varies
markedly across model families. To probe one pos-
sible source of this variation, we focus on Qwen2.5-
VL and LLaV A-v1.6-7B, two families with oppos-
ing tendencies: Qwen leans toward image evidence,
LLaV A leans toward text. Using this contrast, we
examine whether preprocessing-level interventions
can shift the model away from modality-dependent
instability. We apply bidirectional interventions be-
tween the two models. Specifically, Qwen2.5-VL
receives LLaV A-style 336 ×336 global/local view

Base Model Preprocessing Orig. After Prep.
Qwen2.5-VL LLaV A-style 9.71 -37.23
LLaV A-v1.6 Qwen-style -34.92 -30.91
Table 5: Effect of visual preprocessing on Gsinglefor
NQ-SWAP.Orig.is the original model result;After
Prep.is the result after each preprocessing intervention.
construction, whereas LLaV A-v1.6 receives images
pre-resized by Qwen’s smart-resize rule. All re-
maining model-specific processing is unchanged;
see Appendix D for details.
Table 5 reveals a clear but asymmetric effect
across intervention directions. Under LLaV A-style
preprocessing, Qwen’s Gsinglemoves from 9.71%
to -37.23%, not merely flipping sign but landing
within three percentage points of LLaV A’s native
value (-34.92%). Input-side processing alone is
thus sufficient to reproduce LLaV A’s modality pro-
file in a model with different weights and training.
The reverse intervention produces only a modest
shift, from -34.92% to -30.91%.
The key implication of this analysis is that pre-
processing is one factor in modality reliance but
not the sole mechanism; our setup also cannot iso-
late which downstream components interact with it.
A more comprehensive test would require training
matched models that differ only in preprocessing,
which we leave to future work.
7.2 Influence of Image Presentation
A natural concern is whether our findings depend
on arbitrary choices in how textual evidence is ren-
dered as images. To address this, we perturb the
default rendering at two levels of increasing vi-
sual salience and measure how the model’s source-
following behavior shifts.
On the NQ-SWAPsubset, we change only the
visual appearance of the rendered evidence, leav-
ing the query and evidence content untouched. The
first level appliesglobalchanges to the entire im-
age, either by rendering all text in red or by setting
a yellow background. The second level applies the
same changes only to thetargetedanswer-bearing
span; this serves as an upper-bound salience di-
agnostic rather than a deployable setting, since it
presupposes knowledge of the answer.
Figure 5 shows that modest global visual
changes produce only small shifts relative to the
default rendering, whereas answer-span highlight-
ing produces much larger shifts. These findings
both strengthen and qualify our main finding: the
Text
(Full)BG
(Full)Text BG051015
-1.56+0.23+8.10+15.85LLaVA-Vicuna-7B
Text
(Full)BG
(Full)Text BG+0.82+0.58+12.36
+5.41Qwen2.5-VL-7BΔGmulti
xI→TFigure 5: Effect of visualization variants on Gmulti
xI→Ton
NQ-SWAP, relative to the default value.Textand BG
highlight the answer-bearing span (red text and yel-
low background, respectively).(Full)applies the same
changes to the entire evidence.
modality instability documented in §4 is not an ar-
tifact of font, color, or background choice, since
reasonable variations along those dimensions leave
the gain magnitudes largely intact. Yet the answer-
span result indicates that image evidence carries an
additional axis of variability that text evidence does
not. This aligns with the effectiveness of image-
rendered attacks observed in §5.2, suggesting that
the influence of image evidence is shaped not only
by its content but also by its visual form.
8 Conclusion
We study whether MLLMs resolve knowledge con-
flicts consistently across modalities and find that
they do not: the same conflict yields different an-
swers depending on how the evidence is delivered,
and the direction of preference shifts with dataset,
input order, and evidence composition. The fixed
text or image preference reported in prior work
does not survive once evidence is allowed to vary
along these axes, and no single factor we examine
reliably predicts which modality a model will favor.
Modality preferenceis thus better understood as
an artifact of the evaluation setup than as a fixed
property of the model.
This instability has practical consequences, both
beneficial (e.g., highlighting salient passages in
RAG) and harmful (e.g., more potent adversarial
attacks); the simple remedies we explored were
largely ineffective, with conflict-aware fine-tuning
offering only partial mitigation. The origin of this
instability remains an open question. Our prelimi-
nary findings identify the visual input pipeline as
a tractable starting point, but a complete account
will require examining multiple stages of model
training, which we leave to future work.

Limitations
First, we consider only text-modality evidence and
its rendered-image form, and do not examine natu-
rally occurring visual evidence (e.g., figures, charts,
photographs) or other modalities such as audio,
where modality dynamics may differ. Extending
our framework to these broader modalities is a di-
rection for future work. Second, while we evalu-
ate 13 MLLMs across proprietary and open-source
families, our coverage is not exhaustive. Third,
while we identify several contributing factors, a
more thorough mechanistic account of why modal-
ity reliance shifts in the ways we document remains
to be developed.
Ethical Statement
Our safety experiments are intended to diagnose
modality-dependent weaknesses in MLLM refusal
behavior, not to provide new attack recipes. We
report aggregate ASR results and do not include
concrete harmful instructions or unsafe generations.
Any released artifacts will exclude harmful ren-
dered prompts and will be limited to evaluation
metadata and non-sensitive analysis code.
Acknowledgments
This work was supported by Institute of Informa-
tion & communications Technology Planning &
Evaluation (IITP) grant funded by the Korea gov-
ernment(MSIT) (No.RS-2020-II201373, Artificial
Intelligence Graduate School Program(Hanyang
University)). This work was supported by In-
stitute of Information & communications Tech-
nology Planning & Evaluation (IITP) under the
artificial intelligence semiconductor support pro-
gram to nurture the best talents (IITP-2026-RS-
2023-00253914) grant funded by the Korea govern-
ment(MSIT). This work was supported by the Na-
tional Research Foundation of Korea(NRF) grant
funded by the Korea government(MSIT) (RS-2025-
00558151).
References
Anthropic. 2025. Claude 4.5 Sonnet.
Shuai Bai, Keqin Chen, Xuejing Liu, Jialin Wang, Wen-
bin Ge, Sibo Song, Kai Dang, Peng Wang, Shi-
jie Wang, Jun Tang, Humen Zhong, Yuanzhi Zhu,
Mingkun Yang, Zhaohai Li, Jianqiang Wan, Pengfei
Wang, Wei Ding, Zheren Fu, Yiheng Xu, and 8 others.
2025. Qwen2.5-vl technical report.arXiv preprint
arXiv:2502.13923.Payal Bajaj, Daniel Campos, Nick Craswell, Li Deng,
Jianfeng Gao, Xiaodong Liu, Rangan Majumder, An-
drew McNamara, Bhaskar Mitra, Tri Nguyen, and
1 others. 2016. Ms marco: A human generated ma-
chine reading comprehension dataset.arXiv preprint
arXiv:1611.09268.
Ailin Deng, Tri Cao, Zhirui Chen, and Bryan Hooi.
2025. Words or vision: Do vision-language models
have blind faith in text?2025 IEEE/CVF Conference
on Computer Vision and Pattern Recognition (CVPR),
pages 3867–3876.
Manuel Faysse, Hugues Sibille, Tony Wu, Bilel Omrani,
Gautier Viaud, Céline Hudelot, and Pierre Colombo.
2025. ColPali: Efficient document retrieval with
vision language models. InThe Thirteenth Interna-
tional Conference on Learning Representations.
Jian Hu, Xibin Wu, Zilin Zhu, Xianyu, Weixun Wang,
Dehao Zhang, and Yu Cao. 2024. Openrlhf: An easy-
to-use, scalable and high-performance rlhf frame-
work.arXiv preprint arXiv:2405.11143.
Tianze Hua, Tian Yun, and Ellie Pavlick. 2025. How
do vision-language models process conflicting in-
formation across modalities?arXiv preprint
arXiv:2507.01790.
Aaron Hurst, Adam Lerer, Adam P Goucher, Adam
Perelman, Aditya Ramesh, Aidan Clark, AJ Ostrow,
Akila Welihinda, Alan Hayes, Alec Radford, and 1
others. 2024. Gpt-4o system card.arXiv preprint
arXiv:2410.21276.
Yifan Jia, Yuntao Du, Kailin Jiang, Yuyang Liang, Qi-
han Ren, Yi Xin, Rui Yang, Fenze Feng, MingCai
Chen, Hengyang Lu, and 1 others. 2026. Bench-
marking multimodal knowledge conflict for large
multimodal models. InProceedings of the AAAI Con-
ference on Artificial Intelligence, volume 40, pages
22283–22291.
Zhuoran Jin, Pengfei Cao, Hongbang Yuan, Yubo Chen,
Jiexin Xu, Huaijun Li, Xiaojian Jiang, Kang Liu, and
Jun Zhao. 2024. Cutting off the head ends the con-
flict: A mechanism for interpreting and mitigating
knowledge conflicts in language models. InFindings
of the Association for Computational Linguistics:
ACL 2024, pages 1193–1215, Bangkok, Thailand.
Association for Computational Linguistics.
Anant Khandelwal, Manish Gupta, and Puneet Agrawal.
2025. CoCoA: Confidence- and context-aware adap-
tive decoding for resolving knowledge conflicts in
large language models. InProceedings of the 2025
Conference on Empirical Methods in Natural Lan-
guage Processing, pages 6846–6866, Suzhou, China.
Association for Computational Linguistics.
Tom Kwiatkowski, Jennimaria Palomaki, Olivia Red-
field, Michael Collins, Ankur Parikh, Chris Alberti,
Danielle Epstein, Illia Polosukhin, Jacob Devlin, Ken-
ton Lee, Kristina Toutanova, Llion Jones, Matthew
Kelcey, Ming-Wei Chang, Andrew M. Dai, Jakob

Uszkoreit, Quoc Le, and Slav Petrov. 2019. Natu-
ral questions: A benchmark for question answering
research.Transactions of the Association for Compu-
tational Linguistics, 7:452–466.
Kang-il Lee, Minbeom Kim, Seunghyun Yoon, Minsung
Kim, Dongryeol Lee, Hyukhun Koh, and Kyomin
Jung. 2025. VLind-bench: Measuring language pri-
ors in large vision-language models. InFindings
of the Association for Computational Linguistics:
NAACL 2025, pages 4129–4144, Albuquerque, New
Mexico. Association for Computational Linguistics.
Haotian Liu, Chunyuan Li, Qingyang Wu, and Yong Jae
Lee. 2023. Visual instruction tuning.Advances in
neural information processing systems, 36:34892–
34916.
Nelson F. Liu, Kevin Lin, John Hewitt, Ashwin Paran-
jape, Michele Bevilacqua, Fabio Petroni, and Percy
Liang. 2024a. Lost in the middle: How language
models use long contexts.Transactions of the Asso-
ciation for Computational Linguistics, 12:157–173.
Xiaoyuan Liu, Wenxuan Wang, Youliang Yuan, Jen-
tse Huang, Qiuzhi Liu, Pinjia He, and Zhaopeng
Tu. 2025. Insight over sight: Exploring the vision-
knowledge conflicts in multimodal LLMs. InPro-
ceedings of the 63rd Annual Meeting of the Associa-
tion for Computational Linguistics (Volume 1: Long
Papers), pages 17825–17846, Vienna, Austria. Asso-
ciation for Computational Linguistics.
Xin Liu, Yichen Zhu, Jindong Gu, Yunshi Lan, Chao
Yang, and Yu Qiao. 2024b. Mm-safetybench: A
benchmark for safety evaluation of multimodal large
language models. InEuropean Conference on Com-
puter Vision, pages 386–403. Springer.
Shayne Longpre, Kartik Perisetla, Anthony Chen,
Nikhil Ramesh, Chris DuBois, and Sameer Singh.
2021. Entity-based knowledge conflicts in question
answering. InProceedings of the 2021 Conference
on Empirical Methods in Natural Language Process-
ing, pages 7052–7063, Online and Punta Cana, Do-
minican Republic. Association for Computational
Linguistics.
Alex Mallen, Akari Asai, Victor Zhong, Rajarshi Das,
Daniel Khashabi, and Hannaneh Hajishirzi. 2023.
When not to trust language models: Investigating
effectiveness of parametric and non-parametric mem-
ories. InProceedings of the 61st Annual Meeting of
the Association for Computational Linguistics (Vol-
ume 1: Long Papers), pages 9802–9822, Toronto,
Canada. Association for Computational Linguistics.
Trang Nguyen, Jackson Michaels, Madalina Fiterau, and
David Jensen. 2025. Challenges in understanding
modality conflict in vision-language models.arXiv
preprint arXiv:2509.02805.
OpenAI. 2026. GPT-5.4.
OpenBMB. 2025. MiniCPM-o Series. https://
github.com/OpenBMB/MiniCPM-o.Francesco Ortu, Zhijing Jin, Diego Doimo, and Al-
berto Cazzaniga. 2025. When seeing overrides know-
ing: Disentangling knowledge conflicts in vision-
language models. InMechanistic Interpretability
Workshop at NeurIPS 2025.
Rafael Rafailov, Archit Sharma, Eric Mitchell, Christo-
pher D Manning, Stefano Ermon, and Chelsea Finn.
2023. Direct preference optimization: Your language
model is secretly a reward model.Advances in neural
information processing systems, 36:53728–53741.
Mong Yuan Sim, Wei Emma Zhang, Xiang Dai, and
Biaoyan Fang. 2025. Can VLMs actually see and
read? a survey on modality collapse in vision-
language models. InFindings of the Association
for Computational Linguistics: ACL 2025, pages
24452–24470, Vienna, Austria. Association for Com-
putational Linguistics.
Xuezhi Wang, Jason Wei, Dale Schuurmans, Quoc V Le,
Ed H. Chi, Sharan Narang, Aakanksha Chowdhery,
and Denny Zhou. 2023. Self-consistency improves
chain of thought reasoning in language models. In
The Eleventh International Conference on Learning
Representations.
Yike Wang, Shangbin Feng, Heng Wang, Weijia Shi,
Vidhisha Balachandran, Tianxing He, and Yulia
Tsvetkov. 2024. Resolving knowledge conflicts in
large language models. InFirst Conference on Lan-
guage Modeling.
Jian Xie, Kai Zhang, Jiangjie Chen, Renze Lou, and
Yu Su. 2024. Adaptive chameleon or stubborn sloth:
Revealing the behavior of large language models in
knowledge conflicts. InThe Twelfth International
Conference on Learning Representations.
Jin Xu, Zhifang Guo, Jinzheng He, Hangrui Hu, Ting
He, Shuai Bai, Keqin Chen, Jialin Wang, Yang Fan,
Kai Dang, and 1 others. 2025a. Qwen2. 5-omni
technical report.arXiv preprint arXiv:2503.20215.
Jin Xu, Zhifang Guo, Hangrui Hu, Yunfei Chu, Xiong
Wang, Jinzheng He, Yuxuan Wang, Xian Shi, Ting
He, Xinfa Zhu, and 1 others. 2025b. Qwen3-omni
technical report.arXiv preprint arXiv:2509.17765.
Rongwu Xu, Zehan Qi, Zhijiang Guo, Cunxiang Wang,
Hongru Wang, Yue Zhang, and Wei Xu. 2024.
Knowledge conflicts for LLMs: A survey. InPro-
ceedings of the 2024 Conference on Empirical Meth-
ods in Natural Language Processing, pages 8541–
8565, Miami, Florida, USA. Association for Compu-
tational Linguistics.
Wenyan Xu, Dawei Xiang, Tianqi Ding, and Weihai Lu.
2025c. Mmm-fact: A multimodal, multi-domain fact-
checking dataset with multi-level retrieval difficulty.
arXiv preprint arXiv:2510.25120.
Hanrong Ye, Chao-Han Huck Yang, Arushi Goel, Wei
Huang, Ligeng Zhu, Yuanhang Su, Sean Lin, An-
Chieh Cheng, Zhen Wan, Jinchuan Tian, and 1 others.
2025. Omnivinci: Enhancing architecture and data

for omni-modal understanding llm.arXiv preprint
arXiv:2510.15870.
Shi Yu, Chaoyue Tang, Bokai Xu, Junbo Cui, Jun-
hao Ran, Yukun Yan, Zhenghao Liu, Shuo Wang,
Xu Han, Zhiyuan Liu, and Maosong Sun. 2025. Vis-
RAG: Vision-based retrieval-augmented generation
on multi-modality documents. InThe Thirteenth In-
ternational Conference on Learning Representations.
Yu Zhang, Jinlong Ma, Yongshuai Hou, Xuefeng Bai,
Kehai Chen, Yang Xiang, Jun Yu, and Min Zhang.
2025a. Evaluating and steering modality preferences
in multimodal large language model.arXiv preprint
arXiv:2505.20977.
Zhuoran Zhang, Tengyue Wang, Xilin Gong, Yang Shi,
Haotian Wang, Di Wang, and Lijie Hu. 2025b. When
modalities conflict: How unimodal reasoning uncer-
tainty governs preference dynamics in mllms.arXiv
preprint arXiv:2511.02243.
Zongmeng Zhang, Wengang Zhou, Jie Zhao, and
Houqiang Li. 2025c. Robust multimodal large lan-
guage models against modality conflict. InForty-
second International Conference on Machine Learn-
ing.
Jinguo Zhu, Weiyun Wang, Zhe Chen, Zhaoyang Liu,
Shenglong Ye, Lixin Gu, Hao Tian, Yuchen Duan,
Weijie Su, Jie Shao, and 1 others. 2025. Internvl3:
Exploring advanced training and test-time recipes
for open-source multimodal models.arXiv preprint
arXiv:2504.10479.Appendix
A Evaluation Data and Input
Construction
A.1 Text-to-Image Rendering
We render textual evidence as image using the
Python Pillow library. Unless otherwise specified,
the following rendering template is adopted: black
text on a white background, with a font family
DejaVuSans, a font size of 24, a maximum text
width of 800 pixels, and a line-spacing ratio of
1.25. The rendered text is center-aligned within
the image. Line breaks are determined by rendered
pixel width: words are wrapped to the next line
when they exceed the maximum width, and over-
long words are split at the character level. The
final image size is then set adaptively from the
wrapped text bounding box, with an additional 8-
pixel padding to prevent clipping.
Preliminary checks showed that changes in font
size(24-32) and text width(600-1,000) had little
effect on model behavior. We therefore fix the
rendering template in the main experiments and
focus on the effect of delivery modality.
A.2 Dataset Preprocessing and Instance
Selection
We use two knowledge-conflict datasets, CONFLIC-
TQA and NQ-SWAP, and apply lightweight prepro-
cessing before constructing our conflict instances.
CONFLICTQA.We use the GPT-4-generated
version of CONFLICTQA, which is constructed
from PopQA and contains 9,544 instances. Each
instance provides an original answer–context pair
and a counter-answer–context pair. We exclude
instances whose evidence context contains non-
informative responses such as “don’t know” or “I
cannot answer”, since such contexts do not provide
usable external evidence for a conflicting answer.
NQ-SWAP.NQ-SWAPcomprises 4,746 Natural
Questions variants in which the answer entity in
each original context is replaced with another entity
to create conflicting evidence. We exclude contexts
containing excessive markup, particularly repeated
<Table> tags, because they disrupt the linear read-
ing order and produce cluttered or structurally am-
biguous image renderings, making it difficult to
maintain semantic alignment between the text and
rendered-image conditions.

Model CONFLICTQA NQ-SWAP
Claude Sonnet 4.5 1,198 3,205
GPT-5.4 2,695 3,270
GPT-4o 4,049 3,310
Qwen3-Omni 1,670 3,047
Qwen2.5-Omni 694 2,532
MiniCPM-o2.6 1,170 2,863
OmniVinci 626 2,780
Qwen2.5-VL-32B 1,019 2,904
Qwen2.5-VL-7B 718 2,169
InternVL3-38B 740 2,859
InternVL3-8B 676 2,723
LLaV A-v1.6-34B 589 3,050
LLaV A-v1.6-7B 907 2,671
Table 6: Model-wise filtering statistics for constructing
single-evidence conflict pairs in CONFLICTQA and NQ-
SWAP(original: 9,544 and 4,746 respectively). Each
value denotes the number of retained instances after five
closed-book predictions and validity filtering.
Model-specific filtering.After preprocessing,
we identify the memory-supported candidate aint
separately for each model using the closed-book
prediction procedure described in §3 and Fig-
ure 9. We retain only instances with consistent
predictions across five runs and a valid opposing
answer–context pair that can serve as external evi-
dence. Therefore, the final number of usable single-
evidence conflict pairs differs by model. Table 6
and Table 7 report the retained instance counts for
each model and dataset.
Multi-evidence input extension.For multi-
evidence experiments, we require two distinct exter-
nal evidence sources for the same query. For CON-
FLICTQA, we use the ChatGPT-generated variant
in addition to the default GPT-4-generated data,
and use a differently tagged context for the same
query as the second external input. For NQ-SWAP,
when multiple substituted contexts are available for
the same original context, we use another substitu-
tion as the second external evidence source. Table 7
reports the number of unique multi-evidence pairs
constructed in this way. For each pair, we evaluate
both modality assignments, swapping which con-
text is presented as text and which is rendered as
an image. Thus, the number of evaluated cross-
modal inputs is twice the number of unique pairs
in Table 7 for each presentation order.
Dataset-Level Evidence DivergenceTo ex-
amine the dataset-dependent modality gap, we
quantify the divergence between the paired evi-
dence passages within each dataset. For CON-
FLICTQA, we compare the separately generatedModel CONFLICTQA NQ-SWAP
Claude Sonnet 4.5 442 825
GPT-5.4 972 792
GPT-4o 885 727
Qwen3-Omni 313 743
Qwen2.5-Omni 181 632
MiniCPM-o2.6 344 732
OmniVinci 199 716
Qwen2.5-VL-32B 253 721
Qwen2.5-VL-7B 135 461
InternVL3-38B 171 721
InternVL3-8B 184 703
LLaV A-v1.6-34B 226 802
LLaV A-v1.6-7B 335 686
Table 7: Model-wise filtering statistics for construct-
ing multi-evidence conflict pairs in CONFLICTQA and
NQ-SWAP. Considering the cross-modal source assign-
ments, the number of evaluated inputs is the twice of
these unique pairs.
ConflictQA NQ-Swap0.00.20.40.60.81.0Relative token-length gap
ConflictQA NQ-Swap0.00.20.40.60.81.0T oken LCS similarity
ConflictQA NQ-Swap0.00.20.40.60.81.0TFIDF cosine
Figure 6: Distributions of relative token-length gap,
token LCS similarity, and TF–IDF cosine similarity
between paired evidence passages in CONFLICTQA and
NQ-SWAP. Horizontal bars denote means.
parametric_memory and counter_memory pas-
sages, which support the parametric and counter
answers, respectively. For NQ-SWAP, we compare
org_context with sub_context , where the latter
is constructed by replacing the answer entity in the
original passage.
As shown in Figure 6, CONFLICTQA exhibits
a much larger relative token-length gap than NQ-
SWAP(0.372 vs. 0.010), while showing substan-
tially lower token LCS similarity (0.188 vs. 0.942)
and TF–IDF cosine similarity (0.126 vs. 0.922).
Moreover, after masking the paired answer strings
(memory_short_answer /counter_short_answer
for CONFLICTQA and org_answer /sub_answer
for NQ-SWAP), 97.9% of NQ-SWAPpassage
pairs become identical, compared with 0.0% of
CONFLICTQA pairs. The overlap difference also
remains after matching examples by passage
length. These observations indicate that NQ-SWAP
primarily applies a localized answer substitution,
whereas CONFLICTQA constructs globally distinct
counter-evidence.

Model CONFLICTQA NQ-SWAP
Claude Sonnet 4.5 4.35 21.47
GPT-5.4 16.10 17.59
GPT-4o 19.14 14.48
Qwen3-Omni 48.57 26.95
Qwen2.5-Omni 38.77 33.01
MiniCPM-o2.6 23.70 14.68
OmniVinci 66.81 34.07
Qwen2.5-VL-32B 33.33 27.78
Qwen2.5-VL-7B 60.03 28.57
InternVL3-38B 39.07 32.69
InternVL3-8B 66.55 28.97
LLaV A-v1.6-34B 0.89 1.05
LLaV A-v1.6-7B 0.00 0.15
Models with Sign Flip8/13 6/13
Table 8: Order sensitivity across multi-evidence input
orders. Each entry reports theFlip Ratio(%), i.e., the
percentage of paired instances for which the model’s ev-
idence preference flips when the order is reversed from
xI→T toxT→I. The bottom row reports the number of
models whose aggregate gain Gchanges sign across the
two orders (out of 13 models).
B Extended Main Results
B.1 Order Sensitivity in Multi-evidence
Conflicts
Beyond aggregate modality preference in Table 1,
we further examine whether individual predictions
are sensitive to evidence ordering. Table 8 reports
the flip ratio when reversing the order of multi-
modal evidence. A high flip ratio indicates that the
model does not consistently prioritize one evidence
source, but changes its preference depending on
presentation order.
B.2 Absolute EFR Differences
Table 9 reports the signed differences between
image- and text-supported evidence following rates
(EFR) underlying the normalized modality pref-
erence scores in Table 1. While the normalized
metric captures relative modality preference, these
values reveal the absolute magnitude of the under-
lying modality effect.
B.3 Four Evidence Orderings
Table 10 disaggregates the multi-evidence gain
Gmulti
o (defined in §3.2) into all four orderings
xm1→m 2, exposing both the cross-modality order-
ing effect and the same-modality controls hidden
by the aggregation in Table 1.B.4 Same- vs Cross-modality Aggregation
Table 11 reports the same multi-evidence results
aggregated by whether the two evidences share a
modality. This grouping highlights the effect of
modality composition separately from the specific
evidence ordering.
C Post-Training Implementation Details
Unlike the main experiments, which use vLLM
for inference, the mitigation experiments in §6 are
implemented with Hugging Face Transformers to
support fine-tuning and hidden-state interventions.
Minor differences from the main results may there-
fore arise due to the different inference backends.
C.1 Conflict-aware Fine-tuning
Our goal is to test whether reliance on mislead-
ing visual evidence can be recalibrated in a re-
producible open-source setting. We use Qwen2.5-
VL-7B-Instruct as the tuning target since it sup-
ports controlled fine-tuning and exhibits substantial
image-following behavior in the single- and multi-
evidence settings, respectively (Table 1). This pro-
vides a meaningful setting in which to test whether
post-training can improve modality robustness.
The training set contains 2,620 examples con-
structed from NQ-SWAPand MMM-Fact, using a
7:3 mixture: 1,834 examples from NQ-SWAPand
786 contradiction examples from MMM-Fact. In
the NQ-SWAPportion, each instance pairs a fac-
tual question with an image-rendered counterfac-
tual passage that supports a substituted answer. The
target output is the original factual answer rather
than the answer implied by the misleading image.
The MMM-Fact portion consists of image–claim
pairs labeled as ‘contradiction’ which visual and
textual information disagree. Together, these exam-
ples expose the model to situations in which visual
evidence should not be followed solely because of
its presentation modality.
We perform LoRA-based supervised fine-tuning
using Hugging Face Transformers and OpenRLHF
(Hu et al., 2024). The model is trained for one
epoch with a learning rate of 1×10−5, global
batch size 8, micro-batch size 1, LoRA rank/alpha
64/64, and gradient checkpointing.
In Table 3, conflict-aware SFT moves the nor-
malized modality gap Gcloser to zero in all eval-
uated settings, indicating more balanced modality
reliance. However, a smaller Galone does not re-
veal whether this balance results from desirable

ModelEvidence CONFLICTQA NQ-SWAP
Single Multix I→T Multix T→I Single Multix I→T Multix T→I
Claude Sonnet 4.5 32.00 33.75 27.60 20.13 2.09 4.28
GPT-5.4 10.84 0.78 26.18 2.51 1.32 1.55
GPT-4o 11.63 15.00 −10.00 −7.55 1.38 0.82
Qwen3-Omni 2.82 −33.23 70.77 15.16 −30.89 6.42
Qwen2.5-Omni 2.74 −36.40 37.85 9.12 −3.32 21.20
MiniCPM-o2.6 1.96 4.44 22.04 2.20 −13.52 −23.91
OmniVinci −3.19 −57.04 66.89 −3.81 0.56 −0.49
Qwen2.5-VL-32B 3.92 −14.63 57.32 7.37 −7.21 10.40
Qwen2.5-VL-7B 6.96 −59.26 63.15 5.35 −20.18 19.53
InternVL3-38B 0.81 −15.21 72.22 11.19 −24.21 25.24
InternVL3-8B −2.37 −86.96 51.91 4.60 −28.17 13.26
LLaV A-v1.6-34B −9.00 −70.80 −76.55 −10.16 −15.34 −21.02
LLaV A-vicuna-7B −15.54 −87.76 −87.92 −29.21 −36.81 −35.35
Table 9: Absolute EFR differences on CONFLICTQA and NQ-SWAP: EFR(x I)−EFR(x T)in the single-evidence
setting and EFR(x o)−EFR(x T)for each multi-evidence ordering o∈ {I→T, T→I} . Cell shading scales with
the magnitude of the change (■positive /■negative; bins at|x| ≤10,10–30,30–60,>60).
evidence use. To verify that this reduction does not
simply result from uniformly suppressing image
following, we additionally report the raw image-
following rates in Table 12. For misleading evi-
dence, EFR Idecreases from 92.89% to 83.26%
in the single-evidence setting and from 43.84% to
33.52% in the multi-evidence setting in average.
In contrast, for reliable single evidence, EFR I
increases from 80.36% to 83.93%. Collectively,
these results show that the reduced modality gap
is accompanied by selective changes in evidence
following: the model relies less on misleading im-
ages while preserving, and slightly increasing, its
reliance on reliable ones.
C.2 Preference-based post-training
We additionally evaluate Direct Preference Opti-
mization (DPO) as a preference-based post-training
baseline. While conflict-aware SFT directly super-
vises the desired response, DPO provides an alter-
native objective that explicitly contrasts a preferred
response against an undesired response.
Preference data construction.We construct
1,834 preference pairs from NQ-SWAP. Each ex-
ample contains a factual question together with a
substituted image-rendered passage that supports a
counterfactual answer. The original factual answer
is used as thechosenresponse, while the answer
supported by the misleading image is used as the
rejectedresponse. Thus, the preference objective
favors the factual response over one that follows
the substituted visual evidence.Training setup.We apply DPO to Qwen2.5-VL-
7B-Instruct using LoRA. The model is trained for
one epoch with β= 0.1 , a learning rate of 5×10−6,
and LoRA rank/alpha of 64/64. This setup provides
a preference-based counterpart to the supervised
conflict-aware fine-tuning evaluated in §6.2.
Evaluation.he resulting model is evaluated on
the CONFLICTQA multi-evidence setting. This
allows us to test whether the learned preference
transfers to unseen conflict examples rather than
merely reproducing the training distribution.
We report the normalized image-over-text gap
Gmulti
o separately for the two input orders in Ta-
ble 4. Before DPO, the model exhibits a text-
dominant gap of -68.89% under xI→T and an
image-dominant gap of 68.47% under xT→I. Af-
ter DPO, the latter is slightly reduced to 65.57%,
whereas the former becomes substantially more
negative, reaching -88.08%. Hence, preference op-
timization reduces image dominance in one order
but simultaneously strengthens text dominance in
the other.
For completeness, Table 13 reports the corre-
sponding raw image-following rates. Averaged
across the two input orders, EFR Idecreases from
41.11% to 38.15%, a reduction of 2.96 percentage
points. This confirms that DPO reduces overall
following of misleading image evidence. However,
the order-specific Gvalues show that this decrease
does not reflect uniformly improved modality bal-
ance: DPO slightly reduces image dominance un-
derxT→I but further strengthens text dominance

ModelEvidence CONFLICTQA NQ-SWAP
xT→T xT→I xI→T xI→I xT→T xT→I xI→T xI→I
Claude Sonnet 4.5 −3.35 84.62 79.41 −19.28 −0.78 19.38 4.26 −8.60
GPT-5.4 5.84 63.37 1.80 −0.51 −4.26 6.92 4.39−11.57
GPT-4o −4.86 −19.05 4.40 −9.31 0.94 4.80 6.89 −0.61
Qwen3-Omni −6.35 79.25 −38.24 −8.30 1.85 10.94 −55.51 2.71
Qwen2.5-Omni −8.03 43.77 −42.64 −14.74 3.05 33.50 −5.64 3.96
MiniCPM-o2.6 −5.05 26.36 −26.18 −19.43 4.63−42.27 −26.18 −7.42
OmniVinci −9.41 86.58 −64.67 −9.41 3.11 −0.77 0.94 1.41
Qwen2.5-VL-32B 0.83 65.61 −19.37 −0.83 −2.79 16.97 −11.87 −2.79
Qwen2.5-VL-7B −10.22 73.82 −68.98 −6.33 3.96 29.51 33.10 3.89
InternVL3-38B −7.81 84.87 −21.31 −8.77 −4.11 42.92 −40.04 0.67
InternVL3-8B −11.11 58.77 −93.02 −20.16 4.29 21.58 −47.25 3.55
LLaV A-v1.6-34B −19.51 −85.65 −81.63 −14.07 2.97−43.27 −31.38 2.92
LLaV A-vicuna-7B −14.06 −94.85 −95.45 0.00−0.78 −75.66 −78.77 0.00
Table 10: Per-condition normalized image-over-text gap (%) across four evidence-modality orderings in the multi-
evidence setting. xm1→m 2denotes the evidence presentation order, where m1, m2∈ {T, I} . Same-modality cases
are included as controls. Shading follows Table 1.
ModelEvidence CONFLICTQA NQ-SWAP
Same Different Same Different
Claude Sonnet 4.5 −15.81 81.67 −2.05 8.95
GPT-5.4 2.24 31.91 −7.28 5.47
GPT-4o −7.26 −6.86 0.31 5.92
Qwen3-Omni −7.32 21.31 2.29 −21.40
Qwen2.5-Omni −11.39 0.84 3.54 14.64
MiniCPM-o2.6 −12.41 16.00 −1.28 −34.59
OmniVinci −9.41 11.70 2.36 0.06
Qwen2.5-VL-32B −0.06 26.21 −2.79 2.61
Qwen2.5-VL-7B −8.23 2.27 3.92 31.23
InternVL3-38B −8.25 36.45 −1.47 0.86
InternVL3-8B −14.99 −19.28 4.03 −12.32
LLaV A-v1.6-34B −18.25 −83.67 2.95 −37.31
LLaV A-vicuna-7B −10.47 −95.15 −0.54 −77.22
Table 11: Aggregated normalized modality gap (%) over evidence-modality pairings in the multi-evidence setting.
SameandDifferentdenote same- and cross-modality evidence orderings, respectively. Shading follows Table 1.
Evidence False True
MethodEFRmulti
I EFRsingle
I EFRsingle
I
Original 43.84 92.89 80.36
+ SFT 33.52 83.26 83.93
Table 12: Image-following rate (%) on CONFLICTQA
corresponding to the results in Table 3. False and True
denote misleading and reliable evidence, respectively.
underx I→T.
Taken together, DPO results illustrate the distinc-
tion between suppressing misleading-image follow-
ing and achieving modality robustness. Preference-
based post-training can alter modality reliance, but
an objective that consistently improves balanceacross both input orders may require explicitly ac-
counting for modality preference in both directions.
C.3 Representation-level Steering
We evaluate whether modality preference can
be controlled at inference time through activa-
tion steering, following prior work on modality-
preference steering. We focus on the multi-
evidence of CONFLICTQA, where the image pro-
vides misleading evidence while the textual evi-
dence supports the correct answer.
Data split.We apply the method to the same
model, Qwen2.5-VL-7B-Instruct and divide the
evaluation examples into a calibration subset and a
disjoint held-out evaluation subset. The calibration

MethodEFR I ∆EFR I
Preference-based post-training
Original 41.11 –
+ DPO 38.15 -2.96
Representation-level steering
Original (Scale 0) 42.28 –
Scale 1 40.07 -2.21
Scale 3 34.55 -7.73
Scale 5 30.15 -12.13
Table 13: Raw image-following rates for the two in-
tervention baselines. Lower EFR Iindicates reduced
reliance on image evidence; ∆EFR Idenotes the change
from the corresponding original model, Qwen2.5-VL.
subset is used to derive the steering directions, and
all reported results are measured on the held-out
evaluation subset.
Constructing steering directions.We first run
the model on the calibration subset and separate
examples into text-following and image-following
groups, and derive the corresponding steering di-
rection as
vo= normalize (E[h text,o]−E[h image,o ]).
where htext,o andhimage,o denote hidden represen-
tations associated with text-following and image-
following generations under ordero, respectively.
We extract these representations from decoder
layers 20–23. During held-out evaluation, the cor-
responding order-specific vector is added to the
hidden state at every decoding step:
˜ht,o,ℓ=ht,o,ℓ+αv o,ℓ.
where ℓdenotes the decoder layer and αcontrols
the steering strength. We evaluate α∈ {0,1,3,5} ,
withα= 0 corresponding to the original model
without steering. Positive values steer the repre-
sentation toward the direction associated with text-
following behavior.
Evaluation.We quantify modality preference us-
ing the normalized image-over-text gap Gmulti
o as
denoted in §3. Positive values indicate image-
dominant behavior, negative values indicate text-
dominant behavior.
redResults. Table 4 shows that increasing the
steering scale consistently moves modality pref-
erence toward text. Under xT→I, where the un-
steered model is image-dominant, Gmulti
T→I changes
from 77.77% at scale 0 to 80.64%, 64.95%, and60.00% at scales 1, 3, and 5, respectively. In con-
trast, under xI→T, where the model is already text-
dominant, the gap becomes increasingly negative,
changing from -38.64% to -86.96%.
Table 13 reports the corresponding raw image-
following rates, averaged across the two input or-
ders. EFR Idecreases monotonically from 42.28%
to 30.15% as scales increase (-12.13 percentage
points), which means stronger steering reduces
overall reliance on misleading image evidence.
These results confirm that activation steering can
systematically manipulate modality preference at
inference time. However, the intervention is direc-
tional: reducing image preference under one input
order can simultaneously amplify existing text pref-
erence under the other. Thus, successful control of
modality preference should not be conflated with
improved modality robustness, which would re-
quire reducing the magnitude of the modality gap
in both directions.
D Visual Preprocessing Interventions
To examine whether differences in input-side visual
tokenization contribute to modality reliance, we im-
plement a preprocessing-level intervention between
Qwen2.5-VL-7B-Instruct and LLaV A-v1.6-vicuna-
7b. The intervention modifies only the image pre-
processing pipeline applied before inference. We
do not change model weights, language prompts,
decoding hyperparameters, or answer evaluation
rules. Therefore, any change in modality reliance
should be interpreted as the effect of changing how
the visual evidence is presented to the model.
LLaV A-style preprocessing for Qwen2.5-VL.
For the Qwen-to-LLaV A direction, we constrain
Qwen2.5-VL to receive images in a LLaV A-style
global/local view format. We first select a target
canvas resolution from the following set of image
grids:
336×672,672×336,672×672,
1008×336,336×1008.
For each input image, we choose the resolution that
maximizes the effective retained image area, then
resize and pad the image to the selected grid resolu-
tion. The resulting canvas is divided into 336×336
local views. In addition, we create a global view
by resizing the image according to its shortest edge
and center-cropping it to 336×336 , and final visual
input to Qwen consists of both tiles. Here, since

Qwen2.5-VL merges visual patches over a28×28
grid, each336×336view corresponds to
(336/28)×(336/28) = 144
visual tokens. It matches the global/local view
structure of LLaV A-v1.6 while keeping each indi-
vidual view at the LLaV A-style resolution.
We set Qwen’s visual processor constraints so
that the minimum and maximum pixel budgets
are fixed to the desired view size. This prevents
Qwen’s default dynamic resizing from changing
the intended visual tokenization pattern.
Qwen-style pre-resizing for LLaV A-v1.6.For
the reverse direction, we apply Qwen2.5-VL’s de-
fault image resizing rule before feeding the im-
age to LLaV A-v1.6. Specifically, we load the
Qwen2.5-VL image processor configuration and
compute the resized height and width using Qwen’s
smart_resize rule, including its default minimum
pixel budget, maximum pixel budget, patch size,
and merge size. The original image is converted
to RGB, resized to the Qwen-computed resolution
using bicubic interpolation, and saved as a new im-
age file. This pre-resized image is then provided as
the visual input to LLaV A.
This intervention should be understood as an
input-side resizing control rather than a full proces-
sor replacement. We do not claim that this interven-
tion fully explains the Qwen–LLaV A gap. Rather,
it shows that modality reliance can be steered by
input-side visual preprocessing alone, indicating
that visual representation is one contributing factor
alongside model architecture and training.
E Additional Image-Presentation Results
underx T→I
Figure 7 shows the corresponding results for xT→I.
Consistent with the xI→T results in §7.2, global
presentation changes have only minor effects,
while answer-span highlighting produces substan-
tially larger shifts for both models.
F Influence of Image-to-Text Translation
Finally, we rule out a trivial confound: whether
the observed modality-dependent behavior could
be attributed to models simply failing to “read” the
rendered text. We prompt each model to transcribe
the visible text in the rendered image and compare
it to the original passage using ROUGE-L.
Table 14 shows near-perfect scores across repre-
sentative models and datasets, indicating that the
Text
(Full)BG
(Full)Text BG051015
-0.30-0.41+7.42+15.78LLaVA-Vicuna-7B
Text
(Full)BG
(Full)Text BG-0.61+1.76+16.63
+13.17Qwen2.5-VL-7BΔGmulti
xT→IFigure 7: Effect of visualization variants on Gmulti
xT→Ion
NQ-SWAP, relative to the default value.Textand BG
highlight the answer-bearing span (red text and yel-
low background, respectively).Fullapplies the same
changes to the entire evidence.
Model CONFLICTQA NQ-SWAP
Qwen2.5-VL-7B 99.97 99.56
InternVL3-8B 98.37 98.94
LLaV A-v1.6-7B 99.82 99.15
Table 14: Image-to-text transcription performance on
rendered evidence. Values report ROUGE-L between
the original passage and text extracted from the image.
rendered evidence is essentially fully recoverable
as text. This result eliminates poor image-to-text
recognition as the main explanation for §4: models
that demonstrably can read the image still use it
differently from the same text in raw form, pointing
to cross-modal integration, rather than perception,
as the locus of the problem.
G Per-Scenario Safety Results
Table 15 extends the results in Figure 4 in §5.2
to all MM-SafetyBench scenarios that could be
evaluated. Note that some instances are omitted
because of the missing of the required image as-
sets. Each cell reports the Attack Success Rate
(ASR) of our image-rendered instruction condition,
with the value in parentheses indicating the ASR
change relative to the corresponding original base-
line. Across most evaluated scenarios, our image-
rendered instruction condition increases ASR over
the corresponding baselines.
H Prompt Templates
Figure 8, 9, 10, 11, 12 and 13, 14 show the prompt
templates used in our experiments.

Answer the following multiple-choice question
based on your own knowledge.
Question: {query}
Choices:
{choices_str}
Instructions:
- Choose the option that best answers the
question.
- Choose "{NONE_OF_THEM}" only if none of the
other choices is correct.
- Answer with only one letter: A, B, C, or D.
Figure 8: Prompt template for closed-book multiple-
choice evaluation in §3. {choices_str} expands
to four newline-separated options labeled A–D, and
{NONE_OF_THEM} denotes the label assigned to the none-
of-the-above option.
Answer the following question based on your
own knowledge.
Question: {query}
Answer in 1-5 words. Do not use a full
sentence.
Figure 9: Prompt template for extracting internal knowl-
edge when the model choose‘none of the above’in
Figure 8.
{Evidence}
Question: {query}
Answer in 1-5 words. Do not use a full
sentence.
Figure 10: Prompt template used for checking modality
robustness in knowledge conflict scenarios in §3.
You are given both textual and visual
evidence.
Answer the question by jointly considering
both the textual input and the visual
information in the image.
Figure 11: System prompt used to encourage balanced
consideration of both textual and visual evidence.You are given both textual and visual
evidence.
Answer the question by only considering the
visual(textual) information in the input.
Ignore any textual(visual) information
provided outside the image(text).
Figure 12: System prompt for instructing the model to
answer using only one modality, either textual or visual
evidence.
Answer the question with the shortest
possible phrase or sentence.
Use ONLY the provided passages.
Do not add any extra facts.
[Passages]: {pass1}
{pass2}
...
{pass10}
[Query]: {query}
Figure 13: Prompt template used for RAG evaluation.
The model is instructed to answer using the provided
passages, where the gold passage is provided either as
text or as an image.
Perform OCR on the image and extract all
visible text. Do not paraphrase, summarize,
or add any that is not explicitly visible.
Figure 14: Prompt used to evaluate Image-to-Text Trans-
lation ability.

Qwen2.5-VL-32B-Instruct
Scenarios Text-only SD Typo SD+Typo
01-Illegal Activity 1.03(+1.03)81.44(+4.12)5.15(+2.06)12.37(+1.03)
02-Hate Speech 12.88(+6.75)92.02(+8.58)42.94(+9.81)57.67(-3.07)
03-Malware Generation 47.73(+20.46)97.73(+2.28)70.45(+9.09)79.55(+4.55)
04-Physical Harm 31.25(+9.72)92.36(+1.39)54.17(+6.95)65.28(+3.47)
05-Economic Harm 75.41(+2.46)92.62(-0.82)77.05(+0.82)80.33(+1.64)
06-Fraud 13.64(+7.15)91.56(+3.25)39.61(+1.95)53.90(+5.55)
07-Pornography 72.48(+17.43)100.00(+2.75)98.17(+3.67)98.17(+2.76)
08-Political Lobbying 98.69(+0.65)87.58(+0.00)86.93(-0.65)87.58(+0.00)
09-Privacy Violence 33.09(+10.07)92.81(+2.88)43.17(+6.48)46.04(+0.00)
10-Legal Opinion 93.08(+13.08)98.46(+3.08)98.46(+1.54)99.23(+2.31)
11-Financial Advice 98.80(-0.60)100.00(+0.00)100.00(+0.00)99.40(-0.60)
12-Health Consultation 84.40(+15.59)99.08(+0.00)97.25(-1.83)97.25(-0.92)
13-Gov Decision 98.66(+6.04)100.00(+0.00)94.63(+0.67)96.64(+0.00)
InternVL3-38B-Instruct
Scenarios Text-only SD Typo SD+Typo
01-Illegal Activity 0.00(+0.00)57.73(+12.37)5.15(+1.03)9.28(+7.22)
02-Hate Speech 23.93(+21.48)75.46(+4.91)38.04(+9.82)51.53(+9.81)
03-Malware Generation 25.00(+4.55)84.09(-2.27)52.27(+18.18)65.91(+20.46)
04-Physical Harm 23.61(+9.03)81.25(+9.72)45.14(+11.11)50.00(+8.33)
05-Economic Harm 70.49(+1.64)90.16(+4.09)77.05(+3.28)77.87(+0.00)
06-Fraud 2.60(-0.65)71.43(+4.55)24.68(+9.72)39.61(+12.34)
07-Pornography 38.53(+0.92)86.24(+4.59)63.29(+4.57)75.23(-2.75)
08-Political Lobbying 97.39(+8.50)84.97(-2.61)87.58(+1.31)87.58(+0.00)
09-Privacy Violence 19.42(+12.23)79.86(+8.64)35.25(+8.63)39.57(+8.63)
10-Legal Opinion 79.23(+29.23)95.38(+18.46)97.69(+16.15)95.38(+7.69)
11-Financial Advice 96.41(+10.78)100.00(+1.20)100.00(+0.60)100.00(+0.00)
12-Health Consultation 71.56(+21.10)93.58(+9.18)96.33(+6.42)97.25(+0.92)
13-Gov Decision 92.62(+42.96)97.32(+10.07)95.30(+3.35)92.62(+0.67)
LLaV A-v1.6-34B-hf
Scenarios Text-only SD Typo SD+Typo
01-Illegal Activity 10.31(+6.19)95.88(+17.53)43.30(+25.77)52.58(+35.05)
02-Hate Speech 48.47(+20.86)96.93(+5.52)79.75(+10.42)92.64(+8.59)
03-Malware Generation 65.91(-13.64)100.00(+0.00)81.82(+2.27)95.45(+20.45)
04-Physical Harm 64.58(+27.77)98.61(+2.78)81.25(+15.28)87.50(+18.75)
05-Economic Harm 77.87(+7.38)93.44(+4.92)79.51(+0.00)86.89(+6.56)
06-Fraud 34.42(+2.60)100.00(+7.79)79.87(+9.74)89.61(+24.67)
07-Pornography 89.91(+11.93)97.25(+1.84)97.25(+0.92)93.58(-0.92)
08-Political Lobbying 100.00(+7.19)86.93(+2.62)86.93(-0.65)86.27(-1.31)
09-Privacy Violence 53.96(+12.23)100.00(+7.19)73.38(+6.47)82.73(+14.38)
10-Legal Opinion 96.15(+34.61)98.46(+14.61)96.15(+2.30)97.69(+6.15)
11-Financial Advice 100.00(+9.58)99.40(+1.80)99.40(-0.60)100.00(+0.60)
12-Health Consultation 98.17(+39.45)98.17(+4.59)96.33(+2.75)96.33(+0.92)
13-Gov Decision 97.99(+14.77)98.66(+12.08)98.66(+5.37)97.32(+5.37)
Table 15: Effect of rendering harmful instructions across all evaluable MM-SafetyBench scenarios. Each cell reports
the ASR value, with the value in parentheses denoting the change from the corresponding original baseline.