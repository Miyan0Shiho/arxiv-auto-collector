# HalluGuard: Evidence-Grounded Small Reasoning Models to Mitigate Hallucinations in Retrieval-Augmented Generation

**Authors**: Loris Bergeron, Ioana Buhnila, Jérôme François, Radu State

**Published**: 2025-10-01 13:28:20

**PDF URL**: [http://arxiv.org/pdf/2510.00880v1](http://arxiv.org/pdf/2510.00880v1)

## Abstract
Large Language Models (LLMs) excel in many NLP tasks but remain prone to
hallucinations, limiting trust in real-world applications. We present
HalluGuard, a 4B-parameter Small Reasoning Model (SRM) for mitigating
hallucinations in Retrieval-Augmented Generation (RAG). HalluGuard classifies
document-claim pairs as grounded or hallucinated and produces evidence-grounded
justifications for transparency. Our approach combines (i) a domain-agnostic
synthetic dataset derived from FineWeb and refined through multi-stage curation
and data reformation, (ii) synthetic grounded and hallucinated claims, and
(iii) preference-based fine-tuning with Odds Ratio Preference Optimization to
distill large-model reasoning into a smaller backbone. On the RAGTruth subset
of the LLM-AggreFact benchmark, HalluGuard achieves 84.0% balanced accuracy
(BAcc), rivaling specialized models, MiniCheck (7B; 84.0%) and Granite Guardian
3.3 (8B; 82.2%) while using roughly half their parameters. Over the full
benchmark it reaches 75.7% BAcc, matching larger general-purpose LLMs such as
GPT-4o (75.9%). We will release HalluGuard and datasets under Apache 2.0 upon
acceptance.

## Full Text


<!-- PDF content starts -->

HalluGuard: Evidence-Grounded Small Reasoning Models to Mitigate
Hallucinations in Retrieval-Augmented Generation
Loris Bergeron1,4Ioana Buhnila2,3Jérôme François4Radu State4
1Banque de Luxembourg2Center for Data Science in Humanities, Chosun University
3ATILF, University of Lorraine–CNRS4SnT, University of Luxembourg
Correspondence:loris.bergeron@blu.bank
Abstract
Large Language Models (LLMs) excel in
many NLP tasks but remain prone to hallu-
cinations, limiting trust in real-world applica-
tions. We present HalluGuard, a 4B-parameter
Small Reasoning Model (SRM) for mitigating
hallucinations in Retrieval-Augmented Gen-
eration (RAG). HalluGuard classifies docu-
ment–claim pairs as grounded or hallucinated
and produces evidence-grounded justifications
for transparency. Our approach combines (i)
a domain-agnostic synthetic dataset derived
from FineWeb and refined through multi-stage
curation and data reformation, (ii) synthetic
grounded and hallucinated claims, and (iii)
preference-based fine-tuning with Odds Ra-
tio Preference Optimization to distill large-
model reasoning into a smaller backbone. On
the RAGTruth subset of the LLM-AggreFact
benchmark, HalluGuard achieves 84.0% bal-
anced accuracy (BAcc), rivaling specialized
models, MiniCheck (7B; 84.0%) and Gran-
ite Guardian 3.3 (8B; 82.2%) while using
roughly half their parameters. Over the full
benchmark it reaches 75.7% BAcc, matching
larger general-purpose LLMs such as GPT-
4o (75.9%). We will release HalluGuard and
datasets under Apache 2.0 upon acceptance.
1 Introduction
Large Language Models (LLMs) have been used
for a variety of Natural Language Processing (NLP)
tasks, achieving strong results in summarization,
text classification, and question answering (Tan
et al., 2023; Singhal et al., 2023).
However, recent research shows that Small Lan-
guage Models (SLMs) (Schick and Schütze, 2021)
can achieve competitive results in specific tasks, es-
pecially when fine-tuned on domain-specific data.
In addition to being cost and energy efficient, SLMs
are practical in resource-constrained settings (Lep-
agnol et al., 2024) such as on-premise environ-
ments, often required in the financial sector and
industries with strict compliance requirements.
The Eiffel T ower
is in Paris. It
opened in 1889.
Document  𝑥
 HalluGuard
Think ﬁrst... 
then produce 
a classiﬁcation
HALLUCINA TED
The claim says the Eif fel Tower is in
Paris and 20 meters tall. The
document conﬁrms it is in Paris and
opened in 1889, but omits the
height,  making the 20-meter detail
an extrinsic hallucination.
Justiﬁed by
citing relevant
parts from  𝑥
Response Claim cThe Eiffel T ower
is located in
Paris and is 20
meters tall.Figure 1:HalluGuard Concept.Given a document x
and a claim c, the model first thinks before classifying
their relationship as grounded orhallucinated , and
then produces a justification citing relevant parts ofx.
However, a major remaining challenge is that
both LLMs and SLMs are prone to hallucinations,
outputs inconsistent with the input prompt or fac-
tual knowledge (Zhang et al., 2025a; Huang et al.,
2023), and are problematic in Retrieval-Augmented
Generation (RAG) (Lewis et al., 2020) applications,
increasingly deployed in companies due to their
ability to deliver context-aware responses.
Even when using documents, RAGs remain vul-
nerable to hallucinations (Niu et al., 2024), under-
mining trust and explainability (Ni et al., 2025). To
address this, models must be able to detect hallu-
cinations, justify their outputs with evidence, and
integrate into RAG applications.
Recent work emphasizes models designed for
reasoning, the ability to perform multi-step infer-
ence, follow logical chains, and provide transparent
reasoning traces (Wei et al., 2022). Small Reason-
ing Models (SRMs) are not merely SLMs run with
Chain-of-Thought (CoT) prompts. Rather, they are
trained to produce structured intermediate reason-
ing that decomposes complex tasks before generat-
ing output, often through distillation from stronger
reasoners and reward-guided training (Wang et al.,
2025). This makes SRMs particularly well suited
for mitigating hallucinations in RAG applications.
1arXiv:2510.00880v1  [cs.CL]  1 Oct 2025

Moreover, most of the previous work on halluci-
nation detection uses BERT-based classifiers (De-
vlin et al., 2019). Although effective, these models
do not provide justifications, making them unsuit-
able when explainability is mandatory.
This challenge is especially pressing in business
environments, where regulations require decision-
traceable justifications. In fact, to improve effi-
ciency, companies, especially in finance, are de-
ploying custom RAG solutions with team-specific
knowledge (e.g., compliance, legal). Similar de-
ployments are spreading to other industries where
specialized knowledge is critical. In these settings,
trust and explainability are essential. Users must
see which passages of the retrieved document sup-
port or contradict the claim.
To address this gap, we propose HalluGuard,
an SRM for the mitigation of hallucinations in
RAG. As shown in Figure 1, given a document
xand claim c, HalluGuard first thinks about their
relationship before predicting whether the claim
isgrounded orhallucinated , while generating
an evidence-grounded justification, fostering the
transparent and reliable use of RAG in companies.
Our contributions are threefold:
•We introduce HalluGuard1, a Small Reason-
ing Model (SRM) for hallucination mitigation
in Retrieval-Augmented Generation (RAG).
HalluGuard detects hallucinations and gener-
ates evidence-grounded justifications, making
it transparent for human oversight. We will
publicly release HalluGuard and the datasets
used for fine-tuning upon acceptance.
•We construct HalluClaim2, a large-scale syn-
thetic dataset derived from FineWeb (Penedo
et al., 2024) using Llama3.3-70B (Dubey
et al., 2024). HalluClaim provides a con-
trolled yet diverse benchmark for training and
evaluating hallucination detection in RAG-
like scenarios and will also be released.
•We show that HalluGuard improves the bal-
anced accuracy of its backbone and achieves
competitive performance compared to larger
open-source and closed LLMs. Our ablation
study highlights the role of reasoning traces,
consensus filtering, and Odds Ratio Prefer-
ence Optimization (ORPO) (Hong et al., 2024)
fine-tuning in driving these gains.
1https://anonymous.website
2https://anonymous.website2 Related Work
Mitigating hallucinations in LLMs has been ap-
proached through prompt engineering, Retrieval-
Augmented Generation, decoding strategies, super-
vised fine-tuning, and self-reflection (Ji et al. 2023;
Song et al. 2024; Tonmoy et al. 2024; Zhang et al.
2025b). Despite the extensive study of hallucina-
tions in LLMs, there is no consensus on a general
classification, as the boundary between hallucina-
tion and factuality is often blurred (Wei et al. 2024;
Mallen et al. 2023). To address this, Bang et al.
(2025) proposed a three-type taxonomy.
Linked to our work, LYNX (Ravi et al., 2024)
is an open-source hallucination evaluation model
that outperforms GPT-4o and Claude-3 Sonnet,
with 8B and 70B-parameter variants. In addition,
IBM’s Granite Guardian 3.3 (Padhi et al., 2024), an
8B model, detects hallucinations in RAG settings
and provides yes/no scores with optional reasoning
traces through hybrid thinking modes.
Fact-checking has been studied beyond hal-
lucination detection. Tang et al. (2024a) in-
troduced MiniCheck, a model trained on syn-
thetic data matching GPT-4 on multi-fact reasoning
benchmarks, while remaining more cost-effective.
This line of research highlights the importance of
lightweight and scalable models for this specific
task. More recently, Pandit et al. (2025) presented
HaluCheck, a hallucination detection model trained
with a curriculum-based Direct Preference Opti-
mization (DPO) (Rafailov et al., 2023) framework.
HaluCheck was not publicly available at that time.
3 Problem Formulation
We define the task as determining the relationship
between a documentxand a claimc. The relation
t(x, c)can take one of three values:
t(x, c) =

grounded,ifcis supported byx
intrinsic_hallu,ifccontradictsx
extrinsic_hallu,ifcis not inx
A claim cis grounded if it is fully supported
by the information explicitly present in x. It is an
intrinsic hallucination if it directly contradicts x,
and an extrinsic hallucination if its truth requires
external knowledge beyondx. Concrete examples
of these relationships can be found in Figure 2.
For the remainder of this work, we group in-
trinsic and extrinsic hallucinations under a single
hallucinated label, reducing the task to binary
classification (groundedvs.hallucinated).
2

Grounded Intrinsic Hallucination Extrinsic Hallucination
Claim c 
Apple stock hit record, valuing the
company at $900B , after beating
Wall Street expectations  on
international sales .Claim c 
Apple shares fell sharply ,
reducing  the company’ s valuation
below $600B , after missing  Wall
Street forecasts .Claim c 
Apple’ s record-high share
performance was partly driven
by strong demand  for the
iPhone X in emerging markets .
Explanation : This claim exactly
matches  information stated in  𝑥. It is
directly veriﬁable  and fully supported .Explanation : This claim directly
contradicts  𝑥's statement about record
highs and $900B valuation.Explanation : The claim mentions
iPhone X and emerging markets; it
requires external knowledge  to verify .Document  𝑥 
Apple shares hit record highs, brieﬂy
valuing the company at $900B , after
beating W all Street forecasts  with
strong international sales .Document  𝑥 
Apple shares hit record highs, brieﬂy
valuing the company at $900B , after
beating W all Street forecasts  with
strong international sales .Document  𝑥 
Apple shares hit record highs, brieﬂy
valuing the company at $900B , after
beating W all Street forecasts  with
strong international sales .Figure 2:Examples of Relations.A grounded claim, an intrinsic hallucination, and an extrinsic hallucination.
4 Method
4.1 HalluGuard Overview
HalluGuard is a Small Reasoning Model (SRM)
designed to mitigate hallucinations in Retrieval-
Augmented Generation (RAG). Given a docu-
ment–claim pair, it predicts whether the claim is
grounded orhallucinated , and provides a justifi-
cation citing the document, improving transparency
and user trust. HalluGuard supports two inference
modes: in thethinkmode, it generates intermedi-
ate reasoning traces before the final output, while in
non-think mode, it skips these traces and outputs
directly. The mode is controlled at inference time
by adding/thinkor/no_thinkto the prompt.
As shown in Figure 3, our method begins with
a large, high-quality, domain-agnostic corpus that
has been curated for safety, quality, and diversity.
The texts in this corpus are then linguistically re-
formed in tone and style by the Data Reformer
(DR; Llama-3.3-70B) to improve cross-domain
generalization. From these reformed texts, we gen-
erate grounded and hallucinated synthetic claims
using the Claim Generator (CG; Llama-3.3-70B).
To align the model towards high-quality reason-
ing and justifications, we construct a synthetic pref-
erence dataset. For each document–claim pair, we
generate two candidate completions: one from the
Preference Generator-Large (PG-L; Qwen3-32B)
and one from the Preference Generator-Small (PG-
S; Qwen3-0.6B (Yang et al., 2025)). We designate
the output of the PG-L as the chosen response and
the output of the PG-S as the rejected response.
This creates preference pairs that exploit the empir-
ical quality gap between large and small models,
enabling us to build a training dataset without the
need for additional human annotation. To furtherimprove reliability, we apply two filtering steps: (i)
model-agreement verification, in which the label
deduced from the synthetic claim (from CG) is
compared with the classification produced by PG-
L; and (ii) LLM-based consensus filtering. In this
step, two Independent Evaluators (IE-1; Llama-
3.3-70B and IE-2; Mistral Large 2 (Mistral AI
team, 2024)) judge both completions. Only pairs
in which both evaluators select the chosen comple-
tion are retained. Finally, we fine-tune a Qwen3-4B
backbone using LoRA (Hu et al., 2022) for effi-
ciency and Odds Ratio Preference Optimization
(ORPO), which merges Supervised Fine-Tuning
(SFT) and preference alignment into a single stage.
Qwen3-4B was selected to avoid the Learnability
Gap observed in SLMs (Li et al., 2025).
Thus, HalluGuard is a Small Reasoning Model
that delivers reliable hallucination mitigation and
interpretable justifications, ready for seamless inte-
gration into enterprise RAG applications.
4.2 Structured Claim Dataset Construction
Domain-Agnostic Corpus.The performance of
LLMs depends on both the size and the quality of
the dataset (Gunasekar et al., 2023). Larger and
more diverse datasets improve generalization by
exposing models to varied contexts. We therefore
use FineWeb (Penedo et al., 2024), a large-scale,
open-source, domain-agnostic web corpus.
From the 10TB FineWeb sample3, we retain only
documents with a high confidence of being in En-
glish ( language_score≥ 0.95) and remove exact
duplicates. From the remaining pool, we randomly
sample 250,000 documents to form the baseline
dataset, denotedD agnostic .
3https://hf.co/datasets/HuggingFaceFW/fineweb
3

Pre-trained 
Model Backbone3. Preference-Based Fine-T uning
Parameter-Efﬁcient 
Fine-T uning
1. Structured Claim Dataset Construction
Language Score
Threshold
Multi-Stage 
Dataset Curation
Prompt-Guided 
Data Reformation
Synthetic Claim
Generation
Domain-Agnostic
Corpus
2. Preference T raining Dataset Construction
LLM-Based
Consensus Filtering
Model-Agreement
Veriﬁcation
Reasoning-Guided
Preference PairsFigure 3:HalluGuard Training Pipeline.A domain-agnostic corpus is filtered, reformed, and used to generate
three types of synthetic claims (grounded, intrinsic hallucinated, and extrinsic hallucinated). Preference data are
built via cross-model generation (Qwen3-32B and Qwen3-0.6B), model-agreement verification and LLM-based
consensus filtering are used to enhance quality and confidence. The Qwen3-4B backbone is then fine-tuned using
LoRA and ORPO to mitigate hallucinations and produce evidence-grounded justifications in RAG applications.
Multi-Stage Dataset Curation.We further filter
Dagnostic to ensure safety, quality and diversity, fol-
lowing practices similar to C4 (Raffel et al., 2020).
Without this step, models risk learning unsafe, low-
quality, or repetitive patterns. Specifically, we re-
move documents containing unsafe terms4, discard
those that do not comply with C4-style quality rules
(e.g., pages with fewer than five sentences, lines
missing terminal punctuation, boilerplate such as
Lorem Ipsum or cookie notices, and malformed
text such as a single token over 1000 characters).
Finally, we remove documents shorter than 50
words and near-duplicates of any three consecutive
sentences from documents that have already been
retained. This deduplication step is important for
promoting diversity: by eliminating redundant con-
tent, it reduces the repeated boilerplate and ensures
that a wider range of topics and writing styles are
represented. The resulting dataset, denoted Dclean,
contains 86,024 documents.
Prompt-Guided Data Reformation.Despite
multi-stage curation, Dclean remains web-centric
in style due to the nature of FineWeb. To increase
linguistic diversity and improve generalization to
non-web formats (e.g., reports, dialogues), we use
DR to rewrite each document, producing a wider
range of styles that better reflect real-world varia-
tion (Veselovsky et al., 2023; Long et al., 2024).
4https://github.com/LDNOOBWThe reformed dataset is then:
Dreformed =
sj(x) 
x;T(x)x∈D clean	
(1)
where j(x) is a random style from S=
{s1, s2, . . . , s 18}defined in Appendix B, and T(x)
the temperature sampled uniformly from [0.2,0.7] .
Synthetic Claim Generation.We generate one
synthetic claim per document in Dreformed . To bal-
ance the binary classification task, we generate half
grounded and half hallucinated claims, with the
hallucinated split evenly into intrinsic and extrin-
sic, but both labeled ashallucinated.
For each document xi∈D reformed , we ask CG to
generate a claim ciin structured JSON format (He
et al., 2024) (see Appendix C), and assign it a label
ti∈ {grounded,hallucinated} . This results in
86,024 balanced document–claim–label triplets:
HalluClaim=[
t∈C{(xi, ci, ti)|x i∈D t}(2)
4.3 Preference Training Dataset Construction
Reasoning-Guided Preference Pairs.The bal-
anced dataset HalluClaim contains document–
claim–label triplets. However, our goal is not only
to classify claims correctly, but also to train models
to produce evidence-grounded justification.
We convert HalluClaim into the preference
dataset format5, where each instance comprises
5https://hf.co/docs/trl/dataset_formats
4

a prompt and two completions: a chosen comple-
tion and a rejected one (see Appendix J). For
each triplet of documents–claim–label, we con-
struct a prompt Picontaining: (i) task instructions
defining the grounded andhallucinated labels,
(ii) the document xiand (iii) the claim ci. The
prompt requires classification and justification (see
Appendix D).
Thus, we used PG-L and PG-S, with the same
prompt Pi. Each model m∈ {PG-L,PG-S} pro-
duces the response as follows:
R(m)
i= 
y(m)
i, j(m)
i, r(m)
i
(3)
where y(m)
iis the predicted label ( grounded or
hallucinated ),j(m)
iis the justification and r(m)
i
is the model reasoning within the<think>tags.
Assuming that larger models perform better, we
apply a size-based heuristic, marking R(PG-L)
i as
chosenandR(PG-S)
i asrejected.
For each triplet (xi, ci, ti)inHalluClaim , we
produce preference tuples of the form:
zi= 
Pi, R(PG-L)
i (chosen), R(PG-S)
i (rejected)
(4)
Model-Agreement Verification.The size-based
heuristic provides a useful starting point, but some
chosen completions may still misclassify the claim.
To correct this, we require agreement between the
synthetic label assigned by CG during claim gen-
eration and the classification predicted by PG-L.
Any tuple where the chosen label disagrees with
the synthetic label is removed. After this verifica-
tion,HalluClaim prefcontains 83,020 tuples.
LLM-Based Consensus Filtering.To further im-
prove reliability, each tuple is independently evalu-
ated by IE-1 and IE-2 in a few-shot setting (Brown
et al., 2020) using a dedicated prompt that asks
for the selection of the best completion according
to three criteria: (i) classification correctness, (ii)
coherence of reasoning, and (iii) clarity of justifi-
cation (see Appendix E). The models receive the
full prompt Piand completions, without being told
which one is thechosencompletion.
A tuple is retained only if IE-1 and IE-2 select
the same completion that matches the chosen one.
IE-1(P i) =IE-2(P i) =R(chosen)
i (5)
This LLM-based consensus step reduces la-
bel noise and mitigates size-based heuristic bias,
thereby yielding a total of 75,360 high-quality pref-
erence tuples for fine-tuning.4.4 Preference-Based Fine-Tuning
Pre-trained Model Backbone.After creating
a high-quality preference dataset through model-
agreement verification and consensus filtering, we
use Qwen3-4B (Yang et al., 2025) as backbone
for fine-tuning. It supports a context window
of up to 32,768 tokens, which is important for
document-level reasoning. The 4B variant remains
lightweight enough for enterprise on-prem deploy-
ment. Using Qwen3-4B, we address the Small
Model Learnability Gap (Li et al., 2025) observed
in models with at most 3B parameters.
Parameter-Efficient Fine-Tuning.We fine-tune
Qwen3-4B using ORPO, a fine-tuning technique
that increases the gap between chosen and
rejected completions so that the model consis-
tently favors the chosen one (see Appendix G).
Unlike DPO, ORPO performs an SFT stage during
preference alignment, without relying on a refer-
ence model. This makes training more efficient and
allows HalluGuard to accurately classify claims
while generating justifications and reasoning dis-
tilled from stronger models.
To apply ORPO in a parameter-efficient man-
ner, we use LoRA (Hu et al., 2022), which freezes
most base weights and trains only small adapter lay-
ers. This reduces memory and compute costs while
mitigating catastrophic forgetting when adapting
pre-trained models to specific tasks (Bafghi et al.,
2025). Given the 32k token context window, fine-
tuning is memory intensive. We therefore use
Unsloth6, which accelerates fine-tuning with cus-
tom kernels, and memory optimizations, to enable
faster, stable training. Reproducibility and fine-
tuning details are in Appendices A and I.
5 Experimental Setup
Benchmark Dataset.We evaluate on LLM-
AggreFact (Tang et al., 2024a), a collection
of human-annotated datasets designed to assess
whether model-generated claims are supported by
evidence documents. The benchmark spans diverse
domains and incorporates real hallucinations from
recent LLMs, directly aligning with our task of de-
tecting if a claim is grounded orhallucinated .
Importantly, it also includes RAGTruth (Niu et al.,
2024), which is particularly relevant to our focus on
hallucination mitigation in RAG (see Appendix F).
6https://unsloth.ai
5

Model SizeAGGREFACT TofuEvalWiCE REVEALClaim
VerifyFact
CheckExpert
QALFQARAG
TruthBAcc
Avg. CNN XSum MediaS MeetB
Qwen3-32B⋆32B 69.1 76.3 72.0 82.2 80.6 90.0 73.3 77.9 60.2 85.5 85.9 77.6
MiniCheck-7B 7B 65.5 77.8 76.0 78.3 83.0 88.0 75.3 77.7 59.2 86.7 84.0 77.4
Claude-3.5 Sonnet - 67.6 75.1 73.4 84.6 77.7 89.1 71.4 77.8 60.9 85.6 86.1 77.2
Granite Guardian 3.3 8B 67.0 74.9 74.0 78.6 76.6 89.6 75.9 76.1 59.6 86.9 82.2 76.5
Mistral-Large 2⋆123B 64.8 74.7 69.6 84.2 80.3 87.7 71.8 74.5 60.8 87.0 85.9 76.5
gpt-4o-2024-05-13 - 68.1 76.8 71.4 79.8 78.5 86.5 69.0 77.5 59.6 83.6 84.3 75.9
HalluGuard-4B 4B 61.1 73.1 71.7 77.0 80.1 89.3 73.6 77.8 60.0 85.1 84.0 75.7
Qwen2.5-72B-Instruct 72B 63.6 73.0 71.9 80.4 80.2 88.9 70.0 77.0 60.1 84.3 81.9 75.6
Llama-3.1-70B-Instruct 70B 65.7 72.5 72.9 81.0 73.9 86.4 70.3 78.6 58.5 83.8 83.0 75.1
Claude-3 Opus - 65.2 72.4 74.1 82.4 75.0 83.8 69.3 78.8 58.8 81.6 81.8 74.8
Llama-3.3-70B-Instruct⋆70B 68.7 74.7 69.5 78.4 76.6 85.5 67.4 78.5 58.3 79.8 82.6 74.5
Llama-3.1-405B-Instruct 405B 64.8 75.1 68.6 81.2 71.8 86.4 67.5 79.4 58.5 81.9 82.9 74.4
gpt-4o-mini-2024-07-18 - 61.8 73.6 71.3 79.7 76.3 85.8 69.8 76.0 58.3 80.3 81.6 74.0
Qwen3-4B 4B 64.9 73.8 70.9 77.4 68.9 89.5 64.8 78.7 57.5 81.5 83.7 73.8
Llama-3-70B-Instruct 70B 63.7 70.2 71.5 80.6 74.4 85.9 67.8 76.2 57.8 82.4 80.6 73.7
Llama-3.1-8B-Instruct 8B 54.7 68.5 71.1 75.5 72.0 83.5 66.5 72.3 57.8 77.5 73.6 70.3
Llama-3.2-1B-Instruct 1B 50.1 50.9 50.0 50.2 49.7 50.4 50.5 50.2 49.9 50.1 50.9 50.3
Qwen3-0.6B⋆0.6B 20.5 43.4 15.9 26.2 26.5 81.1 23.6 69.4 38.5 25.5 14.9 35.0
Table 1:Evaluation on LLM-AggreFact.Models are ordered by average balanced accuracy (BAcc Avg.; higher is
better). HalluGuard-4B (ours), Qwen3-0.6B, 4B and 32B were evaluated using our specific prompt in think mode.
All other results are taken from the public leaderboard. The higher score between HalluGuard-4B and Qwen3-4B is
shaded in dark green. Alternating grey rows improve readability.⋆Models used within our training pipeline.
Evaluation Metric.Performance is measured us-
ing balanced accuracy (BAcc) (Brodersen et al.,
2010), defined as BAcc =1
2
TP
TP+FN+TN
TN+FP
where TP, TN, FP, and FN denote true positives,
true negatives, false positives, and false negatives.
We adopted BAcc to ensure comparability with
prior work, as it was also used in the paper that
introduced LLM-AggreFact.
6 Results
Evaluation on Benchmark.As shown in Ta-
ble 1, HalluGuard-4B achieves an average BAcc of
75.7%, improving upon its backbone Qwen3-4B
(73.8) by +1.9 points, with strong gains on WiCE
(+11.2) and ClaimVerify (+8.8). These scores are
obtained in think mode using specific prompt and
inference parameters (see Appendices D and H).
HalluGuard-4B is competitive with larger
general-purpose LLMs (e.g., GPT-4o (75.9),
Claude-3 Opus (74.8), Llama-3.3-70B (74.5), and
Mistral-Large 2 (76.5). Compared to special-
ized models, HalluGuard-4B is behind Granite
Guardian 3.3 (76.5) and MiniCheck-7B (77.4).
However, these baselines are larger (8B and 7B pa-
rameters). HalluGuard-4B trails Granite Guardian
by only 0.8 points and MiniCheck-7B by 1.7, while
surpassing them on some benchmarks.
These results show that our fine-tuning pipeline
transforms a lightweight backbone into a model
that rivals both closed and open models, including
general-purpose and specialized models, making
HalluGuard-4B well suited for enterprise RAG ap-
plications where hallucination detection is crucial.RAGTruth Detailed Evaluation.This subset fo-
cuses on RAG settings, evaluating whether claims
are supported by retrieved documents.
HalluGuard-4B achieves an average BAcc of
84.0% like MiniCheck-7B (84.0) and surpasses
Granite Guardian 3.3 (82.2) despite using roughly
half their parameters. It correctly classifies 13,649
grounded claims and detects 984 hallucinations,
missing only 282 (see Table 2). This corresponds
to a True Positive Rate (TPR) of 77.7% and a True
Negative Rate (TNR) of 90.7%, showing that Hal-
luGuard captures most hallucinations while pre-
serving grounded content. This is essential for
user trust. By combining high recall with evidence-
grounded justifications, HalluGuard provides trans-
parent decisions that users can verify, reinforcing
its suitability in enterprise RAG applications.
Predicted
Hallucinated GroundedActualHallucinated 984 282
Grounded 1396 13649
Table 2:Confusion Matrix on the RAGTruth Dataset.
Rows denote actual labels, columns denote predictions.
Thehallucinated label is treated as the positive class.
Justification Evaluation.A common baseline
to evaluate generated text against reference is
metrics such as ROUGE (Lin, 2004). However,
these metrics are inadequate for our task, because
they cannot assess whether a justification is fac-
tually grounded in the source document and have
been shown to correlate poorly with human judg-
6

AGGREFACT-CNNAGGREFACT-XSumTofuEval-MediaS TofuEval-MeetBWiCE
REVEALClaimVerify
FactCheck-GPTExpertQALFQA
RAGTruth BAcc Avg.20406080Average BAcc (%)
Full w/o filter w/o reasoning SFT only
Figure 4:Ablation of HalluGuard-4B.Comparison of the full model and three variants on LLM-AggreFact.
ments (Wang et al., 2023). Thus, we adopt the
G-Eval framework (Liu et al., 2023b), using GPT-
4o (OpenAI et al., 2024) as the evaluator. Con-
cretely, for each RAGTruth document, we evaluate
the justification of PG-L, HalluGuard-4B, and PG-
S. G-Eval assesses four dimensions: Relevance,
Consistency, and Coherence (each on a 1–5 scale),
and Fluency (on a 1–3 scale).
As shown in Table 3, the quality of the justifica-
tion presents significant disparities. Qwen3-32B
achieves scores higher than Qwen3-0.6B in all di-
mensions. Importantly, HalluGuard-4B, although
almost an order of magnitude smaller than Qwen3-
32B, achieves comparable quality, indicating that
ORPO effectively transfers strong model behavior
to a smaller backbone. In addition, fluency remains
uniformly high, suggesting that the observed gains
stem primarily from improved factual grounding
and reasoning, rather than surface-level language
quality. Finally, these results show that HalluGuard-
4B can match the quality of justification of a 32B
parameter model.
Model Rel Coh Con Flu
Qwen3-32B 4.41 4.29 4.47 2.98
HalluGuard-4B 4.36 4.27 4.51 2.97
Qwen3-0.6B 3.72 3.65 3.58 2.75
Table 3:G-Eval Results.Evaluation of justification on
RAGTruth using four dimensions: Relevance (Rel), Co-
herence (Coh), Consistency (Con), and Fluency (Flu).Human Alignment Evaluation.We evaluated
the alignment of our preference construction based
on heuristics (Section 4.3) with human judgments.
We sampled 100 preference tuples zi, balancing
grounded and hallucinated claims. Each tuple was
assessed by two independent NLP expert annota-
tors using the same criteria as those used to con-
struct the preference dataset: correct classification,
coherence of reasoning, and clarity of justification.
Annotators were asked to indicate which com-
pletion they preferred between the two options. Im-
portantly, they were blind to the labels and did not
know which completion had been designated as
chosen orrejected during dataset construction.
To avoid bias, the completions were presented in
random order and without any indications.
At the item level (75 pairs with full annota-
tor agreement), chosen was preferred in 71 cases
(94.7%) vs. 4 for rejected (p= 3.4×10−17, bino-
mial test vs. 50%, 95% CI [0.89, 1.00]). At the an-
notation level (considering all 200 individual judg-
ments), 83.5% favored chosen (p= 4.7×10−23,
95% CI [0.79, 1.00]) (see Table 4).
Evaluation level Pref. for chosen
Item level (n= 75) 94.7%
Annotation level (n= 200) 83.5%
Table 4:Human Alignment Results.Annotators pre-
ferred the chosen completions (94.7% of the 75 fully
agreed items; 83.5% of the 200 individual judgments).
7

These results show that our heuristic is closely
aligned with human preferences. The annotators
clearly favored PG-L over PG-S, both at the item
level (94.7%) and across all judgments (83.5%),
confirming that our heuristic provides an effective
proxy for human preference.
7 Ablation Study
Impact of Consensus Filtering.Applying LLM-
based consensus filtering using independent eval-
uators (IE-1 and IE-2) to preference tuples pro-
vides a small but decisive improvement. With filter-
ing, HalluGuard reaches 75.7% BAcc, compared
to 75.3% without it (–0.4%). Although the gain is
modest, it is crucial. In fact, without this com-
ponent, HalluGuard falls behind Qwen2.5-72B-
Instruct (75.6%).
Contribution of Reasoning.Disabling reason-
ing by using /no_think in the prompt leads to a
decrease in performance. In think mode, Hal-
luGuard reaches a BAcc of 75.7%, whereas in
non-think mode the BAcc decreases to 67.6%
(–8.1%). This represents the second largest drop in
our ablation study, highlighting the critical role of
reasoning in mitigating hallucinations.
This is even more marked on RAGTruth, where
reasoning improves BAcc (+21.8%), with consis-
tent gains across all other datasets (see Figure 5).
Figure 5:Effect of Model Reasoning.Radar plot com-
paring HalluGuard in think mode (lighter blue) vs. in
/no_thinkmode (darker blue).
Effect of Preference Alignment.Replacing
ORPO with SFT alone results in a the largest
drop, with BAcc decreasing from 75.7% to 48.1%
(–27.6%). This indicates that preference alignment,
as embedded in ORPO, plays a crucial role in en-
hancing the reliability and quality of reasoning.Ablation Results.Figure 4 compares the full
HalluGuard-4B model with three ablated variants
on the benchmark datasets. The complete model
consistently outperforms all variants, indicating
that its robustness arises from the interaction of
components rather than from any single factor.
Consensus filtering yields a modest but consis-
tent improvement of +0.4% in BAcc, suggesting
that pruning noisy preference pairs improves align-
ment. The second largest drop occurs when the
reasoning traces are disable via /no_think in the
prompt, with BAcc decreasing by 8.1% overall
and reasoning providing a particularly large gain
of +21.8% on RAGTruth. Replacing ORPO with
SFT alone further reduces performance by 27.6%,
confirming the importance of preference alignment.
Together, these results support the retention of the
entire pipeline to fine-tune HalluGuard-4B.
8 Conclusion
We presented HalluGuard, a 4B-parameter Small
Reasoning Model designed to mitigate hallucina-
tions in Retrieval-Augmented Generation while
providing evidence-grounded justifications.
Built on a domain-agnostic synthetic dataset
with multi-stage curation and preference-based
fine-tuning via ORPO and LoRA, we transform
a compact backbone into a model that rivals or
surpasses much larger LLMs, as well as recent spe-
cialized hallucination-detection models.
In fact, HalluGuard-4B achieves competitive per-
formance on LLM-AggreFact while providing jus-
tifications that are relevant, consistent, and compa-
rable in quality to those of a 32B-parameter model.
Ablation studies also highlight the importance
of reasoning traces, consensus filtering, and pref-
erence alignment in driving these gains. Thus, our
findings demonstrate that carefully aligned small
reasoning models can deliver both reliability and
deployability for enterprise RAG applications, clos-
ing much of the gap with frontier LLMs.
To foster research, we will release HalluGuard
and datasets under Apache 2.0 upon acceptance.
9 Future Work
In future work, we will (i) distinguish intrinsic
and extrinsic hallucinations, and (ii) investigate
multimodal extensions to support charts frequently
present in enterprise documents. We will also re-
lease larger Qwen3-based variants (8B and 14B) to
balance performance with deployment constraints.
8

Limitations
Synthetic Data.Although multiple filters are ap-
plied, synthetic claims may not fully capture the nu-
ances of hallucinations encountered in real-world
RAG applications, since the training is based on
synthetic data.
Output Formatting.To ensure deployment re-
alism, we enforce a strict output structure: the re-
sponse from HalluGuard-4B must be a JSON object
containing CLASSIFICATION andJUSTIFICATION
keys only. Any deviation from this is scored as
incorrect and can underestimate performance.
Hallucination Coverage.The current model
merges intrinsic and extrinsic hallucinations un-
der a single hallucinated label, which reduces
explainability in settings where the distinction be-
tween different types of hallucination is important.
Language and Domain Generalization.Hallu-
Guard has been trained and evaluated on English
data. Its performance in other languages or special-
ized domains remains uncertain.
Ethical Considerations
As with any hallucination detection model, Hal-
luGuard must be used with caution. Overflag-
ging grounded claims may reduce user trust, while
failing to detect hallucinations can lead to harm-
ful errors further down the line. For this reason,
HalluGuard should be used as a decision support
tool rather than as a fully autonomous system, and
should always be paired with human oversight. We
therefore encourage responsible deployment in sen-
sitive domains when integrating HalluGuard into
real-world RAG applications.
References
Reza Akbarian Bafghi, Carden Bagwell, Avinash
Ravichandran, Ashish Shrivastava, and Maziar Raissi.
2025. Fine tuning without catastrophic forgetting
via selective low rank adaptation.arXiv preprint
arXiv:2501.15377.
Yejin Bang, Ziwei Ji, Alan Schelten, Anthony
Hartshorn, Tara Fowler, Cheng Zhang, Nicola Can-
cedda, and Pascale Fung. 2025. HalluLens: LLM
hallucination benchmark. InProceedings of the 63rd
Annual Meeting of the Association for Computational
Linguistics (Volume 1: Long Papers), pages 24128–
24156, Vienna, Austria. Association for Computa-
tional Linguistics.Kay Henning Brodersen, Cheng Soon Ong, Klaas Enno
Stephan, and Joachim M Buhmann. 2010. The bal-
anced accuracy and its posterior distribution. In2010
20th international conference on pattern recognition,
pages 3121–3124. IEEE.
Tom Brown, Benjamin Mann, Nick Ryder, Melanie
Subbiah, Jared D Kaplan, Prafulla Dhariwal, Arvind
Neelakantan, Pranav Shyam, Girish Sastry, Amanda
Askell, and 1 others. 2020. Language models are
few-shot learners.Advances in neural information
processing systems, 33:1877–1901.
Hung-Ting Chen, Fangyuan Xu, Shane Arora, and Eu-
nsol Choi. 2023. Understanding retrieval augmen-
tation for long-form question answering.Preprint,
arXiv:2310.12150.
Jacob Devlin, Ming-Wei Chang, Kenton Lee, and
Kristina Toutanova. 2019. Bert: Pre-training of deep
bidirectional transformers for language understand-
ing. InProceedings of the 2019 conference of the
North American chapter of the association for com-
putational linguistics: human language technologies,
volume 1 (long and short papers), pages 4171–4186.
Abhimanyu Dubey, Abhinav Jauhri, Abhinav Pandey,
Abhishek Kadian, Ahmad Al-Dahle, Aiesha Letman,
Akhil Mathur, Alan Schelten, Amy Yang, Angela
Fan, and 1 others. 2024. The llama 3 herd of models.
arXiv e-prints, pages arXiv–2407.
Suriya Gunasekar, Yi Zhang, and Jyoti Aneja.
2023. Textbooks are all you need.Preprint,
arXiv:2306.11644.
Jia He, Mukund Rungta, David Koleczek, Arshdeep
Sekhon, Franklin X Wang, and Sadid Hasan. 2024.
Does prompt formatting have any impact on llm per-
formance?arXiv preprint arXiv:2411.10541.
Jiwoo Hong, Noah Lee, and James Thorne. 2024.
ORPO: Monolithic preference optimization without
reference model. InProceedings of the 2024 Confer-
ence on Empirical Methods in Natural Language Pro-
cessing, pages 11170–11189, Miami, Florida, USA.
Association for Computational Linguistics.
Edward J Hu, Yelong Shen, Phillip Wallis, Zeyuan
Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang,
Weizhu Chen, and 1 others. 2022. Lora: Low-rank
adaptation of large language models.ICLR, 1(2):3.
Lei Huang, Weijiang Yu, Weitao Ma, Weihong Zhong,
Zhangyin Feng, Haotian Wang, Qianglong Chen,
Weihua Peng, Xiaocheng Feng, Bing Qin, and 1 oth-
ers. 2023. A survey on hallucination in large lan-
guage models: Principles, taxonomy, challenges, and
open questions.arXiv preprint arXiv:2311.05232.
Alon Jacovi, Yonatan Bitton, Bernd Bohnet, Jonathan
Herzig, Or Honovich, Michael Tseng, Michael
Collins, Roee Aharoni, and Mor Geva. 2024. A
chain-of-thought is as strong as its weakest link: A
benchmark for verifiers of reasoning chains.Preprint,
arXiv:2402.00559.
9

Ziwei Ji, Tiezheng Yu, Yan Xu, Nayeon Lee, Etsuko
Ishii, and Pascale Fung. 2023. Towards mitigating
llm hallucination via self reflection. InFindings
of the Association for Computational Linguistics:
EMNLP 2023, pages 1827–1843.
Ryo Kamoi, Tanya Goyal, Juan Diego Rodriguez, and
Greg Durrett. 2023. WiCE: Real-world entailment
for claims in Wikipedia. InProceedings of the 2023
Conference on Empirical Methods in Natural Lan-
guage Processing, pages 7561–7583, Singapore. As-
sociation for Computational Linguistics.
Alexandre Lacoste, Alexandra Luccioni, Victor
Schmidt, and Thomas Dandres. 2019. Quantifying
the carbon emissions of machine learning.arXiv
preprint arXiv:1910.09700.
Pierre Lepagnol, Thomas Gerald, Sahar Ghannay,
Christophe Servan, and Sophie Rosset. 2024. Small
language models are good too: An empirical study
of zero-shot classification. InProceedings of the
2024 Joint International Conference on Computa-
tional Linguistics, Language Resources and Evalua-
tion (LREC-COLING 2024), pages 14923–14936.
Patrick Lewis, Ethan Perez, Aleksandra Piktus, Fabio
Petroni, Vladimir Karpukhin, Naman Goyal, Hein-
rich Küttler, Mike Lewis, Wen-tau Yih, Tim Rock-
täschel, and 1 others. 2020. Retrieval-augmented gen-
eration for knowledge-intensive nlp tasks.Advances
in neural information processing systems, 33:9459–
9474.
Yuetai Li, Xiang Yue, Zhangchen Xu, Fengqing Jiang,
Luyao Niu, Bill Yuchen Lin, Bhaskar Ramasubrama-
nian, and Radha Poovendran. 2025. Small models
struggle to learn from strong reasoners. InFind-
ings of the Association for Computational Linguis-
tics: ACL 2025, pages 25366–25394, Vienna, Austria.
Association for Computational Linguistics.
Chin-Yew Lin. 2004. ROUGE: A Package for Auto-
matic Evaluation of Summaries. InText Summariza-
tion Branches Out, pages 74–81, Barcelona, Spain.
Association for Computational Linguistics.
Nelson Liu, Tianyi Zhang, and Percy Liang. 2023a.
Evaluating verifiability in generative search engines.
InFindings of the Association for Computational Lin-
guistics: EMNLP 2023, pages 7001–7025, Singapore.
Association for Computational Linguistics.
Yang Liu, Dan Iter, and Yichong Xu. 2023b. G-Eval:
NLG Evaluation using GPT-4 with Better Human
Alignment.arXiv preprint. ArXiv:2303.16634 [cs].
Lin Long, Rui Wang, and Ruixuan Xiao. 2024. On
LLMs-Driven Synthetic Data Generation, Cura-
tion, and Evaluation: A Survey.arXiv preprint.
ArXiv:2406.15126 [cs].
Chaitanya Malaviya, Subin Lee, Sihao Chen, Elizabeth
Sieber, Mark Yatskar, and Dan Roth. 2024. Ex-
pertQA: Expert-curated questions and attributed an-
swers. InProceedings of the 2024 Conference ofthe North American Chapter of the Association for
Computational Linguistics: Human Language Tech-
nologies (Volume 1: Long Papers), pages 3025–3045,
Mexico City, Mexico. Association for Computational
Linguistics.
Alex Mallen, Akari Asai, Victor Zhong, Rajarshi Das,
Daniel Khashabi, and Hannaneh Hajishirzi. 2023.
When not to trust language models: Investigating
effectiveness of parametric and non-parametric mem-
ories. InProceedings of the 61st Annual Meeting of
the Association for Computational Linguistics (Vol-
ume 1: Long Papers), pages 9802–9822.
Mistral AI team. 2024. Large Enough: Mis-
tral large 2. https://mistral.ai/news/
mistral-large-2407. Accessed: 2025-08-09.
Bo Ni, Zheyuan Liu, Leyao Wang, Yongjia Lei, Yuy-
ing Zhao, Xueqi Cheng, Qingkai Zeng, Luna Dong,
Yinglong Xia, Krishnaram Kenthapadi, and 1 others.
2025. Towards trustworthy retrieval augmented gen-
eration for large language models: A survey.arXiv
preprint arXiv:2502.06872.
Cheng Niu, Yuanhao Wu, Juno Zhu, Siliang Xu,
KaShun Shum, Randy Zhong, Juntong Song, and
Tong Zhang. 2024. RAGTruth: A hallucination cor-
pus for developing trustworthy retrieval-augmented
language models. InProceedings of the 62nd An-
nual Meeting of the Association for Computational
Linguistics (Volume 1: Long Papers), pages 10862–
10878, Bangkok, Thailand. Association for Compu-
tational Linguistics.
OpenAI, Josh Achiam, Steven Adler, and Sandhini
Agarwal. 2024. Gpt-4 technical report.Preprint,
arXiv:2303.08774.
Inkit Padhi, Manish Nagireddy, Giandomenico Cornac-
chia, Subhajit Chaudhury, Tejaswini Pedapati, Pierre
Dognin, Keerthiram Murugesan, Erik Miehling,
Martín Santillán Cooper, Kieran Fraser, and al. 2024.
Granite guardian.Preprint, arXiv:2412.07724.
Shrey Pandit, Ashwin Vinod, Liu Leqi, and Ying Ding.
2025. Teaching with lies: Curriculum dpo on syn-
thetic negatives for hallucination detection.arXiv
preprint arXiv:2505.17558.
Guilherme Penedo, Hynek Kydlí ˇcek, Anton Lozhkov,
Margaret Mitchell, Colin A Raffel, Leandro
V on Werra, Thomas Wolf, and 1 others. 2024. The
fineweb datasets: Decanting the web for the finest
text data at scale.Advances in Neural Information
Processing Systems, 37:30811–30849.
Rafael Rafailov, Archit Sharma, Eric Mitchell, Christo-
pher D Manning, Stefano Ermon, and Chelsea Finn.
2023. Direct preference optimization: Your language
model is secretly a reward model.Advances in neural
information processing systems, 36:53728–53741.
Colin Raffel, Noam Shazeer, Adam Roberts, Katherine
Lee, Sharan Narang, Michael Matena, Yanqi Zhou,
10

Wei Li, and Peter J Liu. 2020. Exploring the lim-
its of transfer learning with a unified text-to-text
transformer.Journal of machine learning research,
21(140):1–67.
Selvan Sunitha Ravi, Bartosz Mielczarek, Anand Kan-
nappan, Douwe Kiela, and Rebecca Qian. 2024.
Lynx: An open source hallucination evaluation
model.arXiv preprint arXiv:2407.08488.
Timo Schick and Hinrich Schütze. 2021. It’s not just
size that matters: Small language models are also few-
shot learners. InProceedings of the 2021 Conference
of the North American Chapter of the Association
for Computational Linguistics: Human Language
Technologies, pages 2339–2352.
Karan Singhal, Shekoofeh Azizi, Tao Tu, S Sara Mah-
davi, Jason Wei, Hyung Won Chung, Nathan Scales,
Ajay Tanwani, Heather Cole-Lewis, Stephen Pfohl,
and 1 others. 2023. Large language models encode
clinical knowledge.Nature, 620(7972):172–180.
Juntong Song, Xingguang Wang, Juno Zhu, Yuan-
hao Wu, Xuxin Cheng, Randy Zhong, and Cheng
Niu. 2024. Rag-hat: A hallucination-aware tuning
pipeline for llm in retrieval-augmented generation.
InProceedings of the 2024 Conference on Empirical
Methods in Natural Language Processing: Industry
Track, pages 1548–1558.
Yiming Tan, Dehai Min, Yu Li, Wenbo Li, Nan Hu,
Yongrui Chen, and Guilin Qi. 2023. Can chatgpt
replace traditional kbqa models? an in-depth analysis
of the question answering performance of the gpt llm
family. InInternational Semantic Web Conference,
pages 348–367. Springer.
Liyan Tang, Tanya Goyal, Alex Fabbri, Philippe La-
ban, Jiacheng Xu, Semih Yavuz, Wojciech Kryscin-
ski, Justin Rousseau, and Greg Durrett. 2023. Un-
derstanding factual errors in summarization: Errors,
summarizers, datasets, error detectors. InProceed-
ings of the 61st Annual Meeting of the Association for
Computational Linguistics (Volume 1: Long Papers),
pages 11626–11644, Toronto, Canada. Association
for Computational Linguistics.
Liyan Tang, Philippe Laban, and Greg Durrett. 2024a.
MiniCheck: Efficient fact-checking of LLMs on
grounding documents. InProceedings of the 2024
Conference on Empirical Methods in Natural Lan-
guage Processing, pages 8818–8847, Miami, Florida,
USA. Association for Computational Linguistics.
Liyan Tang, Igor Shalyminov, Amy Wong, Jon Burnsky,
Jake Vincent, Yu’an Yang, Siffi Singh, Song Feng,
Hwanjun Song, Hang Su, Lijia Sun, Yi Zhang, Saab
Mansour, and Kathleen McKeown. 2024b. TofuEval:
Evaluating hallucinations of LLMs on topic-focused
dialogue summarization. InProceedings of the 2024
Conference of the North American Chapter of the
Association for Computational Linguistics: Human
Language Technologies (Volume 1: Long Papers),
pages 4455–4480, Mexico City, Mexico. Association
for Computational Linguistics.SM Tonmoy, SM Zaman, Vinija Jain, Anku Rani, Vip-
ula Rawte, Aman Chadha, and Amitava Das. 2024.
A comprehensive survey of hallucination mitigation
techniques in large language models.arXiv preprint
arXiv:2401.01313.
Veniamin Veselovsky, Manoel Horta Ribeiro, and
Akhil Arora. 2023. Generating Faithful Synthetic
Data with Large Language Models: A Case Study
in Computational Social Science.arXiv preprint.
ArXiv:2305.15041 [cs].
Chengyu Wang, Taolin Zhang, Richang Hong, and Jun
Huang. 2025. A short survey on small reasoning
models: Training, inference, applications and re-
search directions.arXiv preprint arXiv:2504.09100.
Jiaan Wang, Yunlong Liang, and Fandong Meng. 2023.
Is ChatGPT a Good NLG Evaluator? A Preliminary
Study.arXiv preprint. ArXiv:2303.04048 [cs].
Yuxia Wang, Revanth Gangi Reddy, Zain Muham-
mad Mujahid, Arnav Arora, Aleksandr Rubashevskii,
Jiahui Geng, Osama Mohammed Afzal, Liang-
ming Pan, Nadav Borenstein, Aditya Pillai, Isabelle
Augenstein, Iryna Gurevych, and Preslav Nakov.
2024. Factcheck-bench: Fine-grained evaluation
benchmark for automatic fact-checkers.Preprint,
arXiv:2311.09000.
Jason Wei, Xuezhi Wang, Dale Schuurmans, Maarten
Bosma, Fei Xia, Ed Chi, Quoc V Le, Denny Zhou,
and 1 others. 2022. Chain-of-thought prompting elic-
its reasoning in large language models.Advances
in neural information processing systems, 35:24824–
24837.
Jerry Wei, Chengrun Yang, Xinying Song, Yifeng Lu,
Nathan Hu, Jie Huang, Dustin Tran, Daiyi Peng,
Ruibo Liu, Da Huang, and 1 others. 2024. Long-
form factuality in large language models.Advances
in Neural Information Processing Systems, 37:80756–
80827.
An Yang, Anfeng Li, Baosong Yang, Beichen Zhang,
Binyuan Hui, Bo Zheng, Bowen Yu, Chang Gao,
Chengen Huang, Chenxu Lv, Chujie Zheng, Dayi-
heng Liu, Fan Zhou, Fei Huang, Feng Hu, Hao Ge,
Haoran Wei, Huan Lin, Jialong Tang, and 41 oth-
ers. 2025. Qwen3 technical report.arXiv preprint
arXiv:2505.09388.
Yue Zhang, Yafu Li, Leyang Cui, Deng Cai, Lemao Liu,
Tingchen Fu, Xinting Huang, Enbo Zhao, Yu Zhang,
Yulong Chen, and 1 others. 2025a. Siren’s song in the
ai ocean: A survey on hallucination in large language
models.Computational Linguistics, pages 1–46.
Ziyao Zhang, Chong Wang, Yanlin Wang, Ensheng Shi,
Yuchi Ma, Wanjun Zhong, Jiachi Chen, Mingzhi Mao,
and Zibin Zheng. 2025b. Llm hallucinations in prac-
tical code generation: Phenomena, mechanism, and
mitigation.Proceedings of the ACM on Software
Engineering, 2(ISSTA):481–503.
11

A Technical Reproducibility
To facilitate reproducibility and transparency, we
report the hardware and software environment used
for all experiments. HalluGuard was fine-tuned on
a single NVIDIA H100 PCIe GPU (80GB mem-
ory, TDP 350W) for 16 hours. Training con-
sumed approximately 7.35 kWh of energy as cal-
culated by the Machine Learning Impact Calcu-
lator (MLIC) (Lacoste et al., 2019). The experi-
ments were carried out on a Linux server running
CUDA 12.4.1 and PyTorch 2.4.0. We used the
default random seeds and PyTorch settings.
B Prompt Template: Style Reformation
Style Instruction
paraphrase Paraphrase the following text
while retaining its original mean-
ing.
summarize Provide a concise summary of the
following text.
expand Expand on the following text by
adding more details and context.
news_article Rewrite the following informa-
tion as a news article.
blog_post Transform the following text into
an engaging blog post.
report Convert the following informa-
tion into a formal report.
story Rewrite the following text as a
narrative story.
dialogue Transform the following text into
a dialogue between two charac-
ters.
letter Rewrite the following text as a
formal letter.
social_media_post Transform the following text into
a social media post.
script Transform the following text into
a script for a short video or play.
interview Rewrite the following text as an
interview between an interviewer
and an expert.
product_description Transform the following text into
a product description.
review Rewrite the following text as a
review of a product or service.
news_summary Summarize the following article
into a concise news brief.
formalize_news Rewrite the following content in
a formal journalistic style.
meeting_summary Rewrite the following text as if it
were a summary of a team meet-
ing.
meeting_dialogue Rewrite the following content as
a conversation between multiple
meeting participants.
Table 5: Each style is applied to reform FineWeb raw
data and increase stylistic diversity.C Prompt Template: Claim Generation
grounded
{
"instructions": [
"Generate a claim that is factually
accurate and fully grounded in the provided
context.",
"Ensure that the claim is explicitly
supported by the context - do not introduce
information that is not directly verifiable
from the context.",
"Only return the claim as the answer. Do
not include any additional text,
explanation, or formatting."
],
"context": <text>,
"answer": ""
}
hallucinated_intrinsic
{
"instructions": [
"Generate a claim that contradicts the
provided context.",
"The claim should remain fluent and
grammatically correct but should be
identifiable as incorrect upon a quick
read.",
"Only return the claim as the answer. Do
not include any additional text,
explanation, or formatting."
],
"context": <text>,
"answer": ""
}
hallucinated_extrinsic
{
"instructions": [
"Generate a claim that includes information
that cannot be verified within the provided
context.",
"Ensure the claim is plausible but requires
external knowledge to verify its accuracy.",
"Only return the claim as the answer. Do
not include any additional text,
explanation, or formatting."
],
"context": <text>,
"answer": ""
}
D Prompt Template: Synthetic Pairs
{
"instructions": [
"You will be given a document and a claim.
Determine whether the claim is'GROUNDED'
or'HALLUCINATED'based on the document.",
"A'GROUNDED'claim is factually accurate
and fully supported by the information
provided in the document. It should be
directly verifiable from the document.",
12

"A'HALLUCINATED'claim is either:",
" - Intrinsically incorrect: It
contradicts the information provided in the
document, or",
" - Extrinsically incorrect: It includes
information that cannot be verified within
the document and requires external
knowledge to assess its accuracy.",
"Return the classification as the answer
(i.e., GROUNDED or HALLUCINATED). Include
justification."
],
"document": <document>,
"claim": <claim>,
"answer": { "CLASSIFICATION": "",
"JUSTIFICATION": "" }
}
E Prompt Template: Consensus Filter
{
"instructions": [
"You will be given a document and a claim,
along with two responses (RESPONSE_A and
RESPONSE_B).",
"Determine which response is better based
on classification correctness, thinking
coherence and clarity, and justification
quality.",
"Return your answer as either'RESPONSE_A'
or'RESPONSE_B', without any justification."
],
"examples": <examples>,
"document": <document>,
"claim": <claim>,
"RESPONSE_A": <response_a>,
"RESPONSE_B": <response_b>,
"best_response": ""
}
F Benchmark Datasets
LLM-AggreFact includes the following datasets:
AGGREFACT (Tang et al., 2023), a factual con-
sistency benchmark for summarization; TofuE-
val (Tang et al., 2024b), a dialogue summariza-
tion benchmark with LLM summaries annotated
for factual consistency; WiCE (Kamoi et al., 2023),
a textual entailment dataset of Wikipedia claims
and cited sources; REVEAL (Jacovi et al., 2024),
which evaluates reasoning chains in open-domain
QA with sentence-level attribution labels against re-
trieved Wikipedia passages; ClaimVerify (Liu et al.,
2023a), which assesses generative search engine
responses by verifying check-worthy sentences
against cited documents with binary factuality la-
bels; FactCheck-GPT (Wang et al., 2024), which
decomposes LLM responses to search queries into
atomic facts; ExpertQA (Malaviya et al., 2024),
consisting of expert-curated queries across 32 do-
mains where system responses are verified againstevidence documents; LFQA (Chen et al., 2023),
where LLM long-form answers conditioned on
retrieved or random documents are labeled; and
RAGTruth (Niu et al., 2024), a retrieval-augmented
generation benchmark where outputs grounded in
retrieved passages are annotated.
G Reward Gap Across Training Epochs
0 0.2 0.4 0.6 0.8 1−8−6−4−2·10−2
epochRewards
rewards/chosen rewards/rejected
Figure 6: The gap between chosen andrejected re-
sponses increases over training, showing that the model
progressively learns to prefer chosen examples while
assigning lower rewards torejectedones.
H Inference Parameters
Parameter Non-Thinking Thinking
temperature 0.7 0.6
min_p 0.0 0.0
top_p 0.8 0.95
top_k 20 20
Table 6: Inference parameters used in our experiments,
following the recommended Qwen settings for non-
thinking and thinking modes.
13

I Fine-Tuning Configuration
Parameter Value
lora_layers_attnq_proj, k_proj,
v_proj
lora_layers_ffngate_proj,
up_proj, down_proj
lora_rank 16
lora_alpha 16
lora_dropout 0
precisionbfloat16
epochs 1
batch_size 2
grad_accumulation 4
effective_batch_size 8
optimizer AdamW (8-bit)
learning_rate1×10−6
lr_schedule linear
orpo_beta 0.1
max_seq_len 32768
Table 7: The setup trains ∼33M parameters (0.81% of
the full model) using LoRA for 1 epoch.
J ORPO Preference Tuple: Full Example
{
"prompt": "{
"instructions": [
"You will be given a document and a
claim. Determine whether the claim is
'GROUNDED'
or'HALLUCINATED'based on the document.",
"A'GROUNDED'claim is factually accurate
and fully supported by the information
provided in the document. It should be
directly verifiable from the document.",
"A'HALLUCINATED'claim is either:",
" - Intrinsically incorrect: It
contradicts the information provided in the
document, or",
" - Extrinsically incorrect: It
includes information that cannot be
verified within the document and requires
external knowledge to assess its accuracy.",
"Return the classification as the answer
(i.e., GROUNDED or HALLUCINATED). Include
justification."
],
"document": "'As I stepped into the
charming St. Blasius parish... (truncated)
...'",
"claim": "'St. Blasius Church is over 850
years old and was built during...
(truncated) ...'",
"answer": { "CLASSIFICATION": "",
"JUSTIFICATION": "" }
}",
"chosen": [{
"role": "assistant",
"content": "<think>Okay, let's tackle
this. The user wants to determine if the
claim about St. Blasius Church...
(truncated) ... Therefore, the claim is
GROUNDED</think>
{
"CLASSIFICATION": "GROUNDED",
"JUSTIFICATION": "The claim that'St.
Blasius Church is over 850 years old and
was built
during the reign of King Stephen'is
directly supported by the document...
(truncated) ..."
}"
}
],
"rejected": [
{
"role": "assistant",
"content": "<think>Okay, let's tackle
this. The user wants me to determine if the
claim... (truncated) ... Since the document
contradicts the claim, the claim is
intrinsically hallucinated.</think>
{
"CLASSIFICATION": "HALLUCINATED",
"JUSTIFICATION": "The document explicitly
states:'The church itself was over 850
years old, built during the reign of King
Stephen ... (truncated) ...'"
}"
}
]
}
14