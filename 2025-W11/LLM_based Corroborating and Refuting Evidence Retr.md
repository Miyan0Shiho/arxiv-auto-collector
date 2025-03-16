# LLM-based Corroborating and Refuting Evidence Retrieval for Scientific Claim Verification

**Authors**: Siyuan Wang, James R. Foulds, Md Osman Gani, Shimei Pan

**Published**: 2025-03-11 00:29:50

**PDF URL**: [http://arxiv.org/pdf/2503.07937v1](http://arxiv.org/pdf/2503.07937v1)

## Abstract
In this paper, we introduce CIBER (Claim Investigation Based on Evidence
Retrieval), an extension of the Retrieval-Augmented Generation (RAG) framework
designed to identify corroborating and refuting documents as evidence for
scientific claim verification. CIBER addresses the inherent uncertainty in
Large Language Models (LLMs) by evaluating response consistency across diverse
interrogation probes. By focusing on the behavioral analysis of LLMs without
requiring access to their internal information, CIBER is applicable to both
white-box and black-box models. Furthermore, CIBER operates in an unsupervised
manner, enabling easy generalization across various scientific domains.
Comprehensive evaluations conducted using LLMs with varying levels of
linguistic proficiency reveal CIBER's superior performance compared to
conventional RAG approaches. These findings not only highlight the
effectiveness of CIBER but also provide valuable insights for future
advancements in LLM-based scientific claim verification.

## Full Text


<!-- PDF content starts -->

LLM-based Corroborating and Refuting Evidence Retrieval
for Scientific Claim Verification
Siyuan Wang1, James R. Foulds2, Md Osman Gani2, Shimei Pan2*
1Anhui Normal University, China
2University of Maryland, Baltimore County, USA
wangtif@ahnu.edu.cn, jfoulds@umbc.edu, mogani@umbc.edu, shimei@umbc.edu
Abstract
In this paper, we introduce CIBER (Claim Investigation
Based on Evidence Retrieval), an extension of the Retrieval-
Augmented Generation (RAG) framework designed to iden-
tify corroborating and refuting documents as evidence for sci-
entific claim verification. CIBER addresses the inherent un-
certainty in Large Language Models (LLMs) by evaluating
response consistency across diverse interrogation probes. By
focusing on the behavioral analysis of LLMs without requir-
ing access to their internal information, CIBER is applica-
ble to both white-box and black-box models. Furthermore,
CIBER operates in an unsupervised manner, enabling easy
generalization across various scientific domains. Comprehen-
sive evaluations conducted using LLMs with varying levels of
domain expertise and linguistic proficiency reveal CIBER’s
superior performance compared to conventional RAG ap-
proaches. These findings not only highlight the effectiveness
of CIBER but also provide valuable insights for future ad-
vancements in LLM-based scientific claim verification.
Introduction
Recent advances in large language model (LLM) technol-
ogy promise to redefine how we interact with language tech-
nologies and use them in the new digital era (Panagoulias
et al. 2023; Wei et al. 2023; Guo et al. 2023; Yang and Luo
2023). One common challenge faced by LLMs is “hallu-
cination” where they may produce answers that seem cor-
rect but are actually inaccurate or misleading (Zhang et al.
2023). This can be particularly problematic in scientific in-
vestigations where accuracy and reliability of evidences and
claims are critical. Hallucinations often arise when the nec-
essary knowledge to answer a specific user query is ab-
sent or inadequate from an LLM’s training data. Retrieval-
Augmented Generation (RAG), a technique designed to im-
prove the reliability of LLMs (Lewis et al. 2020), can mit-
igate this issue by dynamically bringing in external knowl-
edge sources relevant to a user’s query. Prior research has
demonstrated that RAG can significantly reduce the hallu-
cination rate of LLMs and enhance their response reliabil-
ity (Feldman, Foulds, and Pan 2023, 2024). Since its de-
but, RAG has rapidly gained popularity (Siriwardhana et al.
*Corresponding author.
Copyright © 2025, Association for the Advancement of Artificial
Intelligence (www.aaai.org). All rights reserved.2023; Chen et al. 2024; Hofst ¨atter et al. 2023) and been
integrated into numerous AI applications and services. Al-
though RAG can significantly reduce hallucination, the ac-
curacy of LLM responses can still fluctuate due to factors
such as the topic and wording of user queries, the relevance
and consistency of retrieved documents, and the language
comprehension and reasoning abilities of the LLM model
employed, potentially resulting in errors in responses. For
instance, extensive evaluation on benchmark datasets indi-
cates that RAG still struggles regarding noise robustness,
negative rejection, and information integration (Chen et al.
2024).
To further reduce the errors in RAG, we introduce CIBER,
designed to identify and retrieve scientific documents as cor-
roborating or refuting evidences for claim verification. A
claim could range from causal statements like “Human ac-
tivities may cause climate change” to hypotheses such as
“Dysregulation of microRNA expression may contribute to
the pathogenesis of cardiovascular diseases such as hyper-
tension.” Given a claim Cby a user (e.g., a scientist), CIBER
can automatically (1) retrieve relevant scientific publications
from reliable sources (e.g., The New England Journal of
Medicine for biomedical research or NeurIPS or ICML for
Machine Learning), (2) validate the claim Cagainst each
of the retrieved publications (e.g. full papers or abstracts)
through multi-faceted interrogations, providing a verdict
on its truthfulness along with a confidence score, and (3)
present the most representative papers that either support or
refute the claim.
Figure 1 depicts the system diagram of CIBER, which is
an extension of a typical RAG architecture that includes (1)
a text embedding model (Figure 1(a)) that converts text from
both external sources (e.g., scientific publications) and user
queries (e.g., a claim to be verified) into embedding vec-
tors that capture their semantics, (2) a vector database (Fig-
ure 1(b)) enabling similarity-based retrieval of relevant doc-
uments given a user query, and (3) an LLM (Figure 1(c)) that
analyzes the user query as well as the retrieved documents
to generate an answer. In CIBER, we focus on enhancing the
last stage of RAG (the new CIBER components are shown
in the gray box in Figure 1). Specifically, we employ multi-
aspect interrogation (MAI) (Figure 1(d)) to generate differ-
ent probes to assess the reliability of LLM responses. The
Response Resolution (RR) module (Figure 1(e)) parses thearXiv:2503.07937v1  [cs.AI]  11 Mar 2025

Figure 1: CIBER Architecture
LLM responses and maps them to one of the three canoni-
cal answers: Support, Refute, and Neutral. The Verdict and
Confidence (V&C) module (Figure 1(f)) aggregates all the
responses from all the probes and determines whether the
input claim Cis supported or refuted by combined evidence
with a confidence score. Documents with the highest confi-
dence scores can be presented to the user as representative
work that either supports or refutes the claim. The main con-
tributions of our project include:
• Propose a new CIBER framework that can further reduce
hallucination in LLM generation than a typical RAG.
CIBER is unsupervised, making it applicable in diverse
scientific fields. Moreover, it does not require access
to LLM internal information (e.g., model parameters or
training data), making it suitable for both white-box and
black-box LLMs.
• Develop various methods to systematically integrate
and fuse evidences gathered from different interrogation
probes to determine the truthfulness of a claim, along
with an associated confidence score.
• Create two new synthetic and two new real datasets to
assess the effectiveness of the proposed method.
Related Work
RAG has experienced significant development in recent
years. Initially, RAG systems focused on directly augment-
ing Large Language Models (LLMs) with retrieved knowl-
edge through enhanced pre-training techniques (Lewis et al.
2020; Borgeaud et al. 2022). However, with advanced LLMs
demonstrating strong contextual learning abilities, RAG re-
search has transitioned towards providing improved and
more relevant contextual information. These retrieval-based
enhancements include improved source selection (Li, Nie,
and Liang 2023) and query expansion (Ma et al. 2023; Peng
et al. 2023), refined content indexing (Wang et al. 2024), en-
hanced content ranking (Zhuang et al. 2023), and advanced
iterative and recursive retrieval techniques (Shao et al. 2023;
Trivedi et al. 2022). In contrast, our focus is on enhancing
the generation stage of RAG where we systematically mea-
sure the uncertainty in LLM responses to a variety of interro-gation probes to determine the reliability of these responses
and determine the final verdicts.
There is also a large body of work on non-RAG-based
automated claim verification. It often involves four stages,
beginning with claim detection, where claims are selected
for verification based on their check-worthiness (Hassan, Li,
and Tremayne 2015). Evidence retrieval aims to find evi-
dences to indicate a claim’s veracity, often using metadata
or stance detection techniques (Ferreira and Vlachos 2016;
Hanselowski et al. 2019). Verdict prediction determines the
truthfulness of claims (Nakashole and Mitchell 2014; Au-
genstein et al. 2019). Additionally, knowledge graph-based
fact verification have been proposed to assess the verac-
ity of extracted claims by leveraging structured knowledge
bases (Tchechmedjiev et al. 2019). Finally, justification gen-
eration explains the reasoning behind verdicts, with strate-
gies including attention weights, logic-based systems, or
textual explanations (Popat et al. 2018; Ahmadi et al. 2019;
Atanasova 2024). Among the four tasks in the claim extrac-
tion and verification pipeline, we only focus on evidence re-
trieval and verdict prediction.
Research Questions
In this study, we focus on three research questions:
• (RQ1) How does the performance of CIBER compare
with that of a typical RAG? How does its performance
vary with different LLMs with diverse language under-
standing and reasoning abilities?
• (RQ2) How do various interrogation strategies within the
MAI module influence the performance of CIBER?
• (RQ3) What effects do different evidence fusion strategies
within the V&C module have on CIBER performance?
In the following, we provide details on the design and im-
plementation of CIBER.
Methodology
In this section, we explain the main CIBER modules includ-
ing MAI, RR, and a V&C.

Verdict
Support Refute Neutral
PAG Prob (ri=S) Prob (ri=R) Prob (ri=N)
PCF α∗Prob (ri=S)α∗Prob (ri=R)Prob (ri=N)+(1−α)* 
Prob (ri=S)+Prob (ri=R)
Table 1: Mass function m(·)used in Dempster-Shafer Belief Update.
Multi-Aspect Interrogation (MAI)
MAI is designed to assess the consistency of LLM responses
under various probes with diverse lexical and logical varia-
tions. MAI begins by verifying a claim Cwithin the context
of each retrieved paper/abstract. This step is critical in ver-
ifying scientific claims, considering the specialized knowl-
edge necessary to either understand or verify such claims
may not be adequately represented in conventional LLM
training datasets. By anchoring the LLM’s responses in a
specific scientific study, this contextual grounding enables
the LLM to provide more accurate and relevant informa-
tion in response to specific scientific claims. Specifically,
given an input claim C, a retrieved publication Aand an
LLM L, we construct the Original Probe pO(e.g., “Based
on the study presented in Paper A, is Claim C true?”) We
also create additional probes to interrogate Lfrom various
perspectives. To test L’s logic consistency, we design two
sets of probes: PAG(“AGree Probe”) and PCF(“ConFlict
Probe.”) PAGincludes probes whose LLM responses should
align (or agree) with those from pO, while PCFcontains
probes whose responses should contradict (or disagree with)
those from pO. Moreover, to test L’s ability in understanding
user queries with different lexical and syntactic variations,
for each piin either PAGorPCF, we add g(pi), which is a
paraphrase of pi. To illustrate, imagine a climate researcher
seeking to understand the causal link between human ac-
tivities and climate change. Instead of directly querying an
LLM like ChatGPT “Can human activities cause climate
change?”, which may yield inaccurate results, our system
begins by accessing papers from high-quality venues (e.g.,
the journal of Nature Climate Change). For each paper/ab-
stract, the system examines the LLM responses from vari-
ous perspectives. In this example, pOcan be a query such
as “Based on the study described in paper A, is the claim
Ctrue?” We can populate PAGwithpOplus paraphrases of
pOsuch as “Is claim Csupported by the study described in
paper A.” We can populate PCFwith¬pOsuch as “Based
on the study described in paper A, is the claim Cfalse?” and
paraphrases of ¬pOsuch as “Is claim Crefuted by the study
described in paper A.”
Response Resolution (RR)
We employ specific prompt engineering strategies to facil-
itate the parsing of LLM responses. For GPT-2, which op-
erates in a sentence completion mode, we append adverbs
like “relatively” or “quite” in the prompts so that the words
completed by the LLM are more restricted. For instance,
in the prompt “Based on the study described in the pa-
per, the likelihood that ’human activities may cause climate
change’ is relatively/quite [Blank],” the appended adverbssignificantly limit the potential LLM responses to words like
“high” or “low.” In contrast, without these adverbs, the ex-
pressions in [Blank] could vary widely, ranging from “con-
tingent on extensive scientific evidence” to “subject to fur-
ther investigation.” Similarly, for GPT-3.5 and GPT-4, which
operate in a Q&A model, we append specific instructions to
constrain the response format: “Please answer with either
yes or no. If you are not sure, please say ’I am not sure’.”
While these prompt engineering strategies work most of the
time, there’s no guarantee that the LLMs will always follow
the instructions. For example, occasionally GPT-3.5/GPT-
4’ may generate a response like “According to the abstract
provided, the statement ’human activities may cause cli-
mate change’ is false.” To process these responses, we devel-
oped a straightforward lexicon and regular expression-based
parser to map these responses to their canonical forms. For
example, for GPT-2, we extract the words after the adverb
“relatively” or “quite” and map them to either “Support,”
“Refute” or “ Neutral.” To parse the responses from GPT-3.5
and GPT-4, we developed a regular expression-based parser
to identify direct answers such as “Yes,” “No” and “ I am not
sure” or other variants such as “is correct” and “is not false.”
Verdict and Confidence (V&C)
Given a claim Cand a retrieved publication A, the V&C
module gathers all the LLM responses from all the probes
inPAGandPCFand generates the final verdict Vplus a
confidence score CS. Given that an LLM typically employs
a stochastic text generation process, sending the same probe
to the LLM multiple times can result in varied responses.
To address this uncertainty, we iterate each probe in PAG
andPCFKtimes, recording all the KLLM responses per
probe. We explored three fusion strategies to combine the
evidences from all the probes.
Weighted Proportions (WP) Before consolidating the
evidences from all probes, we invert the LLM responses
generated from pi∈PCF, because a “Support” response to
pi∈PCFis equivalent to a “Refute” response to pj∈PAG.
Subsequently, we employ the following formulas to calcu-
late the verdict VWP.
WPS=α∗Prob (ri=S, i∈PAG)
+ (1−α)∗Prob (ri=S, i∈PCF)(1)
WPR=α∗Prob (ri=R, i∈PAG)
+ (1−α)∗Prob (ri=R, i∈PCF)(2)
WPN=α∗Prob (ri=N, i∈PAG)
+ (1−α)∗Prob (ri=N, i∈PCF)(3)
VWP= arg max
Q∈(S,R,N )WPQ (4)

where rirepresents an LLM response generated with
probe i, and Prob (ri)denotes the probability of observ-
ing response riamong the responses. Here, rican take one
of three values: “(S)upport,” “(R)efute,” or “(N)eutral”. α
serves as a trade-off parameter that regulates the relative im-
portance assigned to the evidence derived from the probes in
PAGcompared to those in PCF. The final confidence score,
CSWP =WPVWP.
Weighted Information Gain (WIG) Similar to WP, we
begin by reversing the LLM responses generated from pi∈
PCF. Subsequently, we utilize information theory-based
metrics to evaluate the uncertainty in these LLM responses.
Information theory is a mathematical framework for quanti-
fying the amount of information in data (Shannon 1948). It
explores concepts such as entropy to measure the uncertainty
in a dataset. In addition to entropy E, we also compute In-
formation Gain (IG) to quantify the reduction of uncertainty
given the evidences from LLM responses.
EAG=−X
ri∈{S,R,N }Prob (ri, i∈PAG) logProb (ri, i∈PAG)
(5)
ECF=−X
ri∈{S,R,N }Prob (ri, i∈PCF) logProb (ri, i∈PCF)
(6)
IGAG=logM−EAG (7)
IGCF=logM−ECF, (8)
where Mrepresents the number of possible verdicts
which is 3: “Support,” “Refute,” and “Neutral.” Next we
compute the Weighted Information Gain (WIG) and VWIG :
WIG S=α∗IGAG∗Prob (ri=S, i∈PAG)
+ (1−α)∗IGCFProb (ri=S, i∈PCF)(9)
WIG R=α∗IGAG∗Prob (ri=R, i∈PAG)
+ (1−α)∗IGCFProb (ri=R, i∈PCF)(10)
WIG N=α∗IGAG∗Prob (ri=N, i∈PAG)
+ (1−α)∗IGCFProb (ri=N, i∈PCF)(11)
VWIG = arg max
Q∈(S,R,N )WIG Q. (12)
The final confidence score CSWIG =WIG VWIG.
Weighted Belief Update (WBU) The third fusion ap-
proach is based on the Dempster–Shafer theory (DST) of
evidence and belief update (Dempster 2008; Shafer 1976).
DST provides a framework for reasoning under uncertainty
by combining evidences from multiple sources. The theory
is particularly useful in situations where evidence may be in-
complete or conflicting, providing a systematic way to man-
age uncertainty. Similar to the approaches above, we begin
by reversing the LLM responses generated from pi∈PCF.
Given the LLM responses generated from the probes in PAGandPCFand the verdicts: “Support,” “Refute” and “Neu-
tral,” we define the mass function m(·)as in Table 1 . We
then apply Dempster’s rule of combination to fuse evidences
and update beliefs:
m(S) = (mAG⊕mCF)(S)
=1
1−KX
V1∩V2=SmAG(V1)mCF(V2)
=1
1−K(mAG(S)mCF(S) +mAG(S)mCF(N)
+mAG(N)mCF(S))(13)
m(R) = (mAG⊕mCF)(R)
=1
1−KX
V1∩V2=RmAG(V1)mCF(V2)
=1
1−K(mAG(R)mCF(R) +mAG(R)mCF(N)
+mAG(N)mCF(R))(14)
m(N) = (mAG⊕mCF)(N)
=1
1−KX
V1∩V2=NmAG(V1)mCF(V2)
=1
1−KmAG(N)mCF(N)(15)
K=X
V1∩V2=∅mAG(V1)mCF(V2)
=mAG(R)mCF(S) +mAG(S)mCF(R).(16)
The verdict can be generated based on the updated belief.
VWBU = arg max
Q∈(S,R,N )m(Q) (17)
The confidence score is CSWBU =m(VWBU ).
Meta Verdict and Confidence We can employ an ensem-
ble method to combine the verdicts generated based on each
fusion strategy. In this investigation, we simply apply a ma-
jority voting strategy to combine all the verdicts to compute
VM. AsCSQ, Q∈(S, R, N )have different ranges, we first
normalize them so that all of them are within [0,1]. The fi-
nal confidence score CSMis the average of the normalized
individual confidence scores.
VM=Mode (VWP, VWIG, VWBU ) (18)
CSM=X
Q∈(WP,WIG,WBU )1
3Norm (CSQ) (19)
Evaluation
To systematically assess the effectiveness of our proposed
methods, we created two synthetic and two real paper ab-
stract datasets with ground truth labels.

Figure 2: Examples of Synthetic and Real Abstracts
Ground Truth Datasets
The first two datasets, one synthetic and one real, were cre-
ated to evaluate the validity of the claim regarding the im-
pact of human activities on climate change. The synthetic
dataset was generated using ChatGPT 3.5 Turbo, employing
a prompt “Please generate a paper abstract summarizing a
scientific study investigating the causal relationship between
human activities and climate change. The conclusion should
support (or refute, or remain neutral respectively) on the as-
sertion that human activities can cause climate change.” The
climate synthetic dataset comprises a total of 60 abstracts,
with 20 in each of the three categories: supporting, refut-
ing, and neutral on the input claim. Considering that syn-
thetic abstracts may be subject to the limitations inherent in
LLMs, we complemented them with a real dataset sourcedfrom survey papers (Lynas, Houlton, and Perry 2021; Cook
et al. 2013), which comprises over 3000 papers with estab-
lished ground truth labels, from which we randomly sam-
pled 20 abstracts for each of the three categories, resulting
in a total of 60 real climate paper abstracts.
We also created a synthetic and a real dataset to evaluate
the assertion regarding the purported association between
vaccination and autism. However, generating abstracts sup-
porting this claim proven challenging as they deemed false
and harmful by GPT-3.5. To circumvent this, when GPT-
3.5 generated an abstract refuting the claim (which is easy
for GPT-3.5 to oblige), we prompted it to generate another
abstract with opposing views. The final autism synthetic
dataset includes 20 abstracts for each of the three cate-
gories. We further compiled a real dataset pertaining to the
autism claim based on several survey papers (Doja and

GPT2 GPT3.5 GPT4
Data Model Acc. F1 Acc. F1 Acc. F1
RAG 0.2667 0.2100 0.7083 0.6428 0.7083 0.6847
CSyn CIBER 0.3583 0.2840 0.8583 0.8329 0.8167 0.7993
RAG 0.3250 0.2602 0.4667 0.4098 0.5083 0.4053
CReal CIBER 0.3333 0.2600 0.5917 0.5539 0.5667 0.4567
RAG 0.3417 0.2949 0.4750 0.4688 0.4833 0.4111
ASyn CIBER 0.3500 0.2630 0.6333 0.5875 0.6500 0.5794
RAG 0.3167 0.2750 0.4250 0.4212 0.5500 0.4760
AReal CIBER 0.3583 0.2660 0.5833 0.5157 0.6333 0.5531
RAG 0.3125 0.2600 0.5188 0.4857 0.5625 0.4943
All CIBER 0.3500 0.2683 0.6667 0.6225 0.6667 0.5971
Table 2: Performance of CIBER versus RAG with different LLMs on five datasets: CSyn, CReal, ASyn, AReal, and All.
Roberts 2006; Folb et al. 2004; Stratton et al. 2001; Wil-
son et al. 2003; Madsen and Vestergaard 2004; Mohammed
et al. 2022; Boretti 2021). We extracted 20 abstracts for each
of the three categories with ground truth labels. Figure 2(a)
shows examples from the synthetic dataset and 2(b) exam-
ples from the real dataset.
The synthetic and real datasets exhibit distinct char-
acteristics. Synthetic abstracts tend to be more general.
V ocabulary-wise, they use words more closely tied to the
given claims. In contrast, real abstracts have a higher de-
gree of specificity and nuance, often requiring deeper do-
main knowledge for accurate understanding.
Experiments
We conducted experiments to answer the main research
questions. To test how effectively CIBER works with LLMs
of varying capabilities, we employed the OpenAI GPT-2
model from Huggingface. In addition, we accessed the GPT-
3.5-Turbo and GPT-4-Turbo models via OpenAI’s APIs. In
our experiments, for each prompt in PAGandPCF, we
recorded the LLM responses from 10 random runs. We also
performed grid search to decide the best α.
CIBER performance with different LLMs To answer
RQ1, we compare the performance of CIBER with the tradi-
tional RAG using three different LLMs with abilities rang-
ing from low (GPT-2) to typical (GPT 3.5-Turbo) to state-of-
the-art (GPT4-Turbo). We present the evaluation results on
five datasets: CSyn , the climate change synthetic dataset,
CReal, the climate change real dataset, ASyn, the autism
synthetic dataset and AReal, the autism real dataset. In ad-
dition, we also compute the overall performance on a com-
bination of all four datasets (All).
As shown in Table 2, significant performance differences
exist among CIBERs utilizing different LLMs. Specifically,
CIBER with GPT-2 generally demonstrated poor reliability,
with an accuracy of 0.338% and an F1 score of 0.268 on
the All dataset. More advanced LLMs performed much bet-
ter. Both GPT-3.5 and GPT-4 exhibited comparable accuracy
(0.667%), although GPT-3.5 outperformed GPT-4 in terms
of F1 score (0.623 versus 0.597) on the All dataset.Moreover, CIBER outperformed RAG across all three
LLMs. The most significant enhancements on the All dataset
were observed with GPT-3.5, which exhibited a substantial
14.8% increase in accuracy and a 13.7% improvement in F1
score. Similarly, GPT-4 achieved significant gains, with a
10.4% enhancement in accuracy and a 10.3% increase in F1
score. Comparatively, the enhancements with GPT-2 were
the least, with a modest 3.75% improvement in accuracy and
a marginal 0.8% increase in F1 score.
From this result, it’s evident that CIBER effectively im-
proves the performance of claim verification beyond what
was achieved with conventional RAG methods. Given the
nuanced nature of scientific literature and the requirement
for precise language comprehension, advanced models like
GPT-3.5 and GPT-4 exhibit much better performance com-
pared to smaller models like GPT-2.
Impact of incorporating diverse interrogation strategies
To answer RQ2, we conducted experiments to compare dif-
ferent versions of CIBER with different interrogation strate-
gies: CIBER AGwhich only employs the probes in PAG,
CIBER CFwhich only includes the probes in PCF, and
CIBER ALL which includes the probes in both.
As shown in Table 3, CIBER CFexhibited a signifi-
cant disadvantage compared to CIBER AGwith GPT-3.5
(a 19.4% decrease in accuracy and a 22.6% decrease in F1
on the All dataset) and GPT-4 (an 11.5% decrease in accu-
racy and a 12.6% decrease in F1 on the All dataset). How-
ever, this disadvantage was not observed with GPT-2. In fact,
CIBER CFheld a slight edge over CIBER AG(with a 1%
increase in accuracy and a 3.4% improvement in F1 score).
Upon reviewing the log file, we found that this might stem
from how we processed the responses to PCF. While we
employed GPT-2 in an auto-completion mode, GPT-3.5 and
GPT-4 were run in Q&A mode. To make response parsing
easier, we specifically instructed GPT-3.5 and GPT-4 to an-
swer with “Yes”, “No” or “I am not sure,” whose interpre-
tation under negative probes can be ambiguous. To illustrate
this, we show two responses from GPT-3.5 in our query log.
In addition to the “Yes” and “No” answers we requested,
it occasionally produces supplementary information which

GPT2 GPT3.5 GPT4
Data Model Acc. F1 Acc. F1 Acc. F1
RAG 0.2667 0.2100 0.7083 0.6428 0.7083 0.6847
CIBER-AG 0.3167 0.1892 0.8583 0.8329 0.8250 0.8075
CSyn CIBER-CF 0.3333 0.2358 0.6583 0.5605 0.6667 0.6117
CIBER-ALL 0.3583 0.2840 0.8583 0.8329 0.8167 0.7993
RAG 0.3250 0.2602 0.4667 0.4098 0.5083 0.4053
CIBER-AG 0.3250 0.2063 0.6000 0.5615 0.5333 0.4287
CReal CIBER-CF 0.3333 0.2174 0.4417 0.3648 0.5500 0.4457
CIBER-ALL 0.3333 0.2600 0.5917 0.5539 0.5667 0.4567
RAG 0.3417 0.2949 0.4750 0.4688 0.4833 0.4111
CIBER-AG 0.3250 0.1940 0.6000 0.5567 0.6500 0.5712
ASyn CIBER-CF 0.3500 0.2551 0.3583 0.2954 0.4667 0.3612
CIBER-ALL 0.3500 0.2630 0.6333 0.5875 0.6500 0.5794
RAG 0.3167 0.2750 0.4250 0.4212 0.5500 0.4760
CIBER-AG 0.3250 0.1869 0.5333 0.4739 0.6333 0.5508
AReal CIBER-CF 0.3167 0.2038 0.3583 0.2988 0.5000 0.4355
CIBER-ALL 0.3583 0.2660 0.5833 0.5157 0.6333 0.5531
RAG 0.3125 0.2600 0.5188 0.4857 0.5625 0.4943
CIBER-AG 0.3229 0.1941 0.6479 0.6063 0.6604 0.5896
All CIBER-CF 0.3333 0.2280 0.4542 0.3799 0.5459 0.4635
CIBER-ALL 0.3500 0.2683 0.6667 0.6225 0.6667 0.5971
Table 3: The impact of interrogation strategies on system performance on five datasets: CSyn, CReal, ASyn, AReal, and All.
helps us understand the LLM responses more precisely:
Prompt: Based on the abstract, is the following claim “Hu-
man activities may cause climate change” false?
(Response 1) Yes, the statement ”Human activities may
cause climate change” is not necessarily false based on the
information provided in the abstract.
(Response 2) No. The statement ”Human activities may
cause climate change” is not false based on the information
provided in the abstract.
Since our RRcomponent simply extracts ”Yes” and ”No”
from the responses, it assigns different verdicts, even though
both responses support the original claim. We expect that in
the future, if we adopt better response resolution strategies,
the effectiveness of the probes in PCFfor GPT-3.5 and GPT-
4 can be significantly improved.
Nonetheless, even with our current simple response res-
olution strategies for GPT-3.5 and GPT-4, aggregating the
probes from both PAGandPCFyielded a consistent per-
formance enhancement for CIBER ALL on the All dataset
compared to either CIBER AGorCIBER CFalone.
Impact of different evidence fusion strategies To an-
swer RQ3, we performed experiments comparing various
versions of CIBER with different evidence fusion strategies:
CIBER WP utilizing a weighted proportion-based fusion
strategy, CIBER WIG employing a weighted information
gain-based strategy, CIBER WBU employing a weighted
belief update-based fusion strategy, and CIBER ALL em-
ploying a majority-based voting strategy to aggregate the
verdict from each individual strategy.
Based on the results presented in Table 4, each of the three
evidence fusion strategies significantly outperformed tradi-
Figure 3: Correlations Between Different Metrics used in
Combining Evidences
tional RAG individually by a considerable margin on the All
dataset. However, there is no clear pattern regarding which
strategy emerges as the top performer individually, as each
strategy claims the top spot in one-third of the tests on the
All dataset. Furthermore, the simple majority voting-based
verdict aggregation strategy did not result in a superior per-
forming model.
To explore the relationship between different fusion
strategies with different LLMs, we computed the corre-
lations among nine variables comprising combinations of
three fusion metrics ( WP,WIG , and WBU ) and three

GPT2 GPT3.5 GPT4
Data Model Acc F1 Acc F1 Acc F1
RAG 0.2667 0.2100 0.7083 0.6428 0.7083 0.6847
CIBER-WP 0.3583 0.2720 0.8500 0.8188 0.8000 0.7758
CSyn CIBER-WIG 0.3417 0.3274 0.8583 0.8329 0.8250 0.8075
CIBER-WBU 0.3583 0.2810 0.8583 0.8329 0.8500 0.8367
CIBER-ALL 0.3583 0.2840 0.8583 0.8329 0.8167 0.7993
RAG 0.3250 0.2602 0.4667 0.4098 0.5083 0.4053
CIBER-WP 0.3667 0.3025 0.5917 0.5530 0.5667 0.4567
CReal CIBER-WIG 0.3500 0.3045 0.6000 0.5615 0.5667 0.4567
CIBER-WBU 0.3500 0.2610 0.6083 0.5680 0.5750 0.4639
CIBER-ALL 0.3333 0.2600 0.5917 0.5539 0.5667 0.4567
RAG 0.3417 0.2949 0.4750 0.4688 0.4833 0.4111
CIBER-WP 0.3583 0.2742 0.6250 0.6066 0.5917 0.5194
ASyn CIBER-WIG 0.3500 0.2741 0.6333 0.5898 0.6500 0.5794
CIBER-WBU 0.3417 0.2587 0.6167 0.5738 0.6500 0.5794
CIBER-ALL 0.3500 0.2630 0.6333 0.5875 0.6500 0.5794
RAG 0.3167 0.2750 0.4250 0.4212 0.5500 0.4760
CIBER-WP 0.3667 0.2726 0.5750 0.5285 0.5917 0.5217
AReal CIBER-WIG 0.3583 0.2951 0.5833 0.5176 0.6333 0.5531
CIBER-WBU 0.3583 0.2703 0.5750 0.5098 0.6333 0.5508
CIBER-ALL 0.3583 0.2660 0.5833 0.5157 0.6333 0.5531
RAG 0.3125 0.2600 0.5188 0.4857 0.5625 0.4943
CIBER-WP 0.3625 0.2803 0.6604 0.6267 0.6375 0.5684
All CIBER-WIG 0.3500 0.3003 0.6687 0.6255 0.6688 0.5992
CIBER-WBU 0.3521 0.2678 0.6646 0.6211 0.6771 0.6077
CIBER-ALL 0.3500 0.2683 0.6667 0.6225 0.6667 0.5971
Table 4: The impact of evidence fusion strategies on system performance on five datasets: CSyn, CReal, ASyn, AReal, and All.
LLMs (GPT-2, GPT-3.5, and GPT-4). The resulting corre-
lation matrix is depicted in Figure 3, where darker areas rep-
resent stronger correlations. As shown in the figure, the most
prominent correlations are observed along the diagonal line
among the variables within each LLM. Specifically, among
GPT-2 related strategies, the highest correlation is between
WIG andWP (ρ= 0.55), followed by WIG andWBU
(ρ= 0.34). For all the GPT-3.5 related variables, the highest
correlation occurs between WIG andWP (ρ= 0.72) fol-
lowed by WIG andWBU (ρ= 0.50). For GPT-4, the high-
est correlated variables are WIG andWBU (ρ= 0.58).
Across different LLM models, the highest correlation occurs
between WPGPT 3.5andWPGPT 4(ρ= 0.45).
The correlation results suggest that while there are signif-
icant correlations between various fusion metrics, with the
exception of WIG GPT 3.5andWPGPT 3.5, most correla-
tions are moderate or low. This implies that the effective-
ness of different fusion strategies may be complementary.
As a result, instead of employing a simple majority voting,
exploring more sophisticated ensemble methods for verdict
aggregation could be a promising avenue for future research.
Conclusions and Future Work
In this paper, we introduce CIBER, a novel framework de-
signed to enhance Retrieval-Augmented Generation (RAG)
systems for evidence retrieval and scientific claim verifica-
tion. CIBER focuses on systematically addressing the in-herent uncertainties in LLM outputs. CIBER is quite gen-
eral and applicable across diverse scenarios. For instance,
CIBER focuses on LLM behavioral analysis, which doesn’t
require access to LLM internal information, making it suit-
able for both white-box and black-box LLMs. Additionally,
CIBER is unsupervised, making it easily generalizable to
different scientific fields. Our evaluation results demonstrate
that CIBER achieves significant performance improvements
over traditional RAG approaches, particularly benefiting ad-
vanced LLMs like GPT-3.5 and GPT-4. Furthermore, we’ve
curated several ground truth datasets—two synthetic and
two real—which we plan to share with the research com-
munity.
Through this exploration, we have also identified poten-
tial areas for future enhancement, including improving the
development of the LLM response resolution, as well as de-
veloping more sophisticated ensemble methods for combin-
ing verdicts from different fusion strategies.
While CIBER aims to mitigate hallucinations in LLM
generation, which can help reduce the risks associated with
spreading LLM-generated misinformation, it’s important to
acknowledge that CIBER is still far from perfect. There is a
possibility that CIBER could inadvertently generate or cite
information that is untrue, particularly if the retrieved con-
tent contains misinformation or misleading information. As
a result, continuous improvement of CIBER is critical for
the safe adoption of LLMs in the real world.

References
Ahmadi, N.; Lee, J.; Papotti, P.; and Saeed, M. 2019. Ex-
plainable fact checking with probabilistic answer set pro-
gramming. arXiv preprint arXiv:1906.09198 .
Atanasova, P. 2024. Generating fact checking explanations.
InAccountable and Explainable Methods for Complex Rea-
soning over Text , 83–103. Springer.
Augenstein, I.; Lioma, C.; Wang, D.; Lima, L. C.; Hansen,
C.; Hansen, C.; and Simonsen, J. G. 2019. MultiFC: A real-
world multi-domain dataset for evidence-based fact check-
ing of claims. arXiv preprint arXiv:1909.03242 .
Boretti, A. 2021. Reviewing the association between alu-
minum adjuvants in the vaccines and autism spectrum dis-
order. Journal of Trace Elements in Medicine and Biology ,
66: 126764.
Borgeaud, S.; Mensch, A.; Hoffmann, J.; Cai, T.; Ruther-
ford, E.; Millican, K.; Van Den Driessche, G. B.; Lespiau,
J.-B.; Damoc, B.; Clark, A.; et al. 2022. Improving lan-
guage models by retrieving from trillions of tokens. In In-
ternational Conference on Machine Learning , 2206–2240.
PMLR.
Chen, J.; Lin, H.; Han, X.; and Sun, L. 2024. Benchmark-
ing large language models in retrieval-augmented genera-
tion. In Proceedings of the AAAI Conference on Artificial
Intelligence , 17754–17762.
Cook, J.; Nuccitelli, D.; Green, S. A.; Richardson, M.; Win-
kler, B.; Painting, R.; Way, R.; Jacobs, P.; and Skuce, A.
2013. Quantifying the consensus on anthropogenic global
warming in the scientific literature. Environmental research
letters , 8(2): 024024.
Dempster, A. P. 2008. Upper and lower probabilities in-
duced by a multivalued mapping. In Classic works of the
Dempster-Shafer theory of belief functions , 57–72. Springer.
Doja, A.; and Roberts, W. 2006. Immunizations and autism:
a review of the literature. Canadian Journal of Neurological
Sciences , 33(4): 341–346.
Feldman, P.; Foulds, J. R.; and Pan, S. 2023. Trapping LLM
hallucinations using tagged context prompts. arXiv preprint
arXiv:2306.06085 .
Feldman, P.; Foulds, R., James; and Pan, S. 2024. RAGged
Edges: The Double-Edged Sword of Retrieval-Augmented
Chatbots. arXiv preprint arXiv:2403.01193 .
Ferreira, W.; and Vlachos, A. 2016. Emergent: a novel data-
set for stance classification. In Proceedings of the 2016
conference of the North American chapter of the associa-
tion for computational linguistics: Human language tech-
nologies . ACL.
Folb, P. I.; Bernatowska, E.; Chen, R.; Clemens, J.; Dodoo,
A. N.; Ellenberg, S. S.; Farrington, C. P.; John, T. J.; Lam-
bert, P.-H.; MacDonald, N. E.; et al. 2004. A global perspec-
tive on vaccine safety and public health: the Global Advisory
Committee on Vaccine Safety. American journal of public
health , 94(11): 1926–1931.
Guo, Z.; Wang, P.; Huang, L.; and Cho, J.-H. 2023. Authen-
tic Dialogue Generation to Improve Youth’s Awareness ofCybergrooming for Online Safety. In 2023 IEEE 35th In-
ternational Conference on Tools with Artificial Intelligence
(ICTAI) , 64–69.
Hanselowski, A.; Stab, C.; Schulz, C.; Li, Z.; and Gurevych,
I. 2019. A richly annotated corpus for different tasks in au-
tomated fact-checking. arXiv preprint arXiv:1911.01214 .
Hassan, N.; Li, C.; and Tremayne, M. 2015. Detecting
check-worthy factual claims in presidential debates. In Pro-
ceedings of the 24th ACM International Conference on In-
formation and Knowledge Management , 1835–1838.
Hofst ¨atter, S.; Chen, J.; Raman, K.; and Zamani, H. 2023.
FiD-Light: Efficient and effective retrieval-augmented text
generation. In Proceedings of the 46th International ACM
SIGIR Conference on Research and Development in Infor-
mation Retrieval , 1437–1447.
Lewis, P.; Perez, E.; Piktus, A.; Petroni, F.; Karpukhin, V .;
Goyal, N.; K ¨uttler, H.; Lewis, M.; Yih, W.-t.; Rockt ¨aschel,
T.; et al. 2020. Retrieval-augmented generation for
knowledge-intensive NLP tasks. Advances in Neural Infor-
mation Processing Systems , 33: 9459–9474.
Li, X.; Nie, E.; and Liang, S. 2023. From classification
to generation: Insights into crosslingual retrieval augmented
ICL. arXiv preprint arXiv:2311.06595 .
Lynas, M.; Houlton, B. Z.; and Perry, S. 2021. Greater than
99% consensus on human caused climate change in the peer-
reviewed scientific literature. Environmental Research Let-
ters, 16(11): 114005.
Ma, X.; Gong, Y .; He, P.; Zhao, H.; and Duan, N. 2023.
Query rewriting for retrieval-augmented large language
models. arXiv preprint arXiv:2305.14283 .
Madsen, K. M.; and Vestergaard, M. 2004. MMR vaccina-
tion and autism: what is the evidence for a causal associa-
tion? Drug safety , 27: 831–840.
Mohammed, S. A.; Rajashekar, S.; Ravindran, S. G.;
Kakarla, M.; Gambo, M. A.; Salama, M. Y .; Ismail, N. H.;
Tavalla, P.; Uppal, P.; Hamid, P.; et al. 2022. Does Vac-
cination Increase the Risk of Autism Spectrum Disorder?
Cureus , 14(8).
Nakashole, N.; and Mitchell, T. 2014. Language-aware truth
assessment of fact candidates. In Proceedings of the 52nd
Annual Meeting of the Association for Computational Lin-
guistics (Volume 1: Long Papers) , 1009–1019.
Panagoulias, D. P.; Palamidas, F. A.; Virvou, M.; and
Tsihrintzis, G. A. 2023. Rule-Augmented Artificial
Intelligence-empowered Systems for Medical Diagnosis us-
ing Large Language Models. In 2023 IEEE 35th Interna-
tional Conference on Tools with Artificial Intelligence (IC-
TAI), 70–77. Los Alamitos, CA, USA: IEEE Computer So-
ciety.
Peng, W.; Li, G.; Jiang, Y .; Wang, Z.; Ou, D.; Zeng, X.;
Chen, E.; et al. 2023. Large language model based long-
tail query rewriting in TaoBao search. arXiv preprint
arXiv:2311.03758 .
Popat, K.; Mukherjee, S.; Yates, A.; and Weikum, G.
2018. Declare: Debunking fake news and false claims
using evidence-aware deep learning. arXiv preprint
arXiv:1809.06416 .

Shafer, G. 1976. A mathematical theory of evidence , vol-
ume 42. Princeton university press.
Shannon, C. E. 1948. A mathematical theory of communi-
cation. The Bell system technical journal , 27(3): 379–423.
Shao, Z.; Gong, Y .; Shen, Y .; Huang, M.; Duan, N.; and
Chen, W. 2023. Enhancing retrieval-augmented large lan-
guage models with iterative retrieval-generation synergy.
arXiv preprint arXiv:2305.15294 .
Siriwardhana, S.; Weerasekera, R.; Wen, E.; Kaluarachchi,
T.; Rana, R.; and Nanayakkara, S. 2023. Improving the do-
main adaptation of retrieval augmented generation (RAG)
models for open domain question answering. Transactions
of the Association for Computational Linguistics , 11: 1–17.
Stratton, K.; Gable, A.; Shetty, P.; McCormick, M.;
of Medicine (US) Immunization Safety Review Committee,
I.; et al. 2001. Immunization safety review: measles-mumps-
rubella vaccine and autism. Immunization Safety Review:
Measles-Mumps-Rubella Vaccine and Autism .
Tchechmedjiev, A.; Fafalios, P.; Boland, K.; Gasquet, M.;
Zloch, M.; Zapilko, B.; Dietze, S.; and Todorov, K. 2019.
ClaimsKG: A knowledge graph of fact-checked claims. In
The Semantic Web–ISWC 2019: 18th International Semantic
Web Conference, Auckland, New Zealand, October 26–30,
2019, Proceedings, Part II 18 , 309–324. Springer.
Trivedi, H.; Balasubramanian, N.; Khot, T.; and Sabharwal,
A. 2022. Interleaving retrieval with chain-of-thought rea-
soning for knowledge-intensive multi-step questions. arXiv
preprint arXiv:2212.10509 .
Wang, Y .; Lipka, N.; Rossi, R. A.; Siu, A.; Zhang, R.;
and Derr, T. 2024. Knowledge graph prompting for multi-
document question answering. In Proceedings of the AAAI
Conference on Artificial Intelligence , 19206–19214.
Wei, J.; Courbis, A.-L.; Lambolais, T.; Xu, B.; Bernard,
P. L.; and Dray, G. 2023. Zero-shot Bilingual App Reviews
Mining with Large Language Models. In 2023 IEEE 35th In-
ternational Conference on Tools with Artificial Intelligence
(ICTAI) , 898–904.
Wilson, K.; Mills, E.; Ross, C.; McGowan, J.; and Jadad,
A. 2003. Association of autistic spectrum disorder and the
measles, mumps, and rubella vaccine: a systematic review
of current epidemiological evidence. Archives of Pediatrics
& Adolescent Medicine , 157(7): 628–634.
Yang, B.; and Luo, X. 2023. Recent Progress on Named
Entity Recognition Based on Pre-trained Language Models.
In2023 IEEE 35th International Conference on Tools with
Artificial Intelligence (ICTAI) , 799–804.
Zhang, Y .; Li, Y .; Cui, L.; Cai, D.; Liu, L.; Fu, T.; Huang,
X.; Zhao, E.; Zhang, Y .; Chen, Y .; Wang, L.; Luu, A. T.;
Bi, W.; Shi, F.; and Shi, S. 2023. Siren’s Song in the AI
Ocean: A Survey on Hallucination in Large Language Mod-
els. arXiv:2309.01219.
Zhuang, S.; Liu, B.; Koopman, B.; and Zuccon, G. 2023.
Open-source large language models are strong zero-shot
query likelihood models for document ranking. arXiv
preprint arXiv:2310.13243 .