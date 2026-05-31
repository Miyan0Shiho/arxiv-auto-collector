# K-FinHallu: A Hallucination Detection Benchmark for Multi-Turn RAG in Korean Finance

**Authors**: Eunbyeol Cho, Yunseung Lee, Mirae Kim, Jeewon Yang, Youngjun Kwak, Edward Choi

**Published**: 2026-05-28 07:40:19

**PDF URL**: [https://arxiv.org/pdf/2605.29523v1](https://arxiv.org/pdf/2605.29523v1)

## Abstract
Large Language Models (LLMs) have advanced financial automation through Retrieval-Augmented Generation (RAG), yet hallucinations remain a critical barrier to deployment in high-stakes environments. Existing benchmarks focus on single-turn, English-centric tasks, leaving the multi-turn dynamics and linguistic-regulatory nuances of the Korean financial domain unaddressed. We introduce K-FinHallu, the first benchmark for hallucination detection in multi-turn Korean financial RAG. We construct multi-turn dialogues from authentic Korean financial documents and inject hallucinations under a proposed hierarchical taxonomy based on context answerability that explicitly accounts for justified abstention. Benchmarking frontier and open-source LLMs as hallucination detectors, we find that even the strongest models struggle with fine-grained financial diagnostics and refusal behavior. While fine-tuning an 8B model on our training split yields performance competitive with frontier LLMs, justified abstention remains the weakest axis across all evaluated models.

## Full Text


<!-- PDF content starts -->

K-FinHallu: A Hallucination Detection Benchmark
for Multi-Turn RAG in Korean Finance
Eunbyeol Cho*1, Yunseung Lee*1,2, Mirae Kim2
Jeewon Yang1,Youngjun Kwak2,Edward Choi1
1KAIST AI,2Financial Tech Lab, KakaoBank Corp.
{eunbyeol.cho, jwy24, edwardchoi}@kaist.ac.kr
{yun.lee, melissa.kim, vivaan.yjkwak}@lab.kakaobank.com
Abstract
Large Language Models (LLMs) have ad-
vanced financial automation through Retrieval-
Augmented Generation (RAG), yet hallucina-
tions remain a critical barrier to deployment
in high-stakes environments. Existing bench-
marks focus on single-turn, English-centric
tasks, leaving the multi-turn dynamics and
linguistic-regulatory nuances of the Korean fi-
nancial domain unaddressed. We introduce K-
FinHallu, the first benchmark for hallucina-
tion detection in multi-turn Korean financial
RAG. We construct multi-turn dialogues from
authentic Korean financial documents and in-
ject hallucinations under a proposed hierarchi-
cal taxonomy based on context answerability
that explicitly accounts for justified abstention.
Benchmarking frontier and open-source LLMs
as hallucination detectors, we find that even the
strongest models struggle with fine-grained fi-
nancial diagnostics and refusal behavior. While
fine-tuning an 8B model on our training split
yields performance competitive with frontier
LLMs, justified abstention remains the weakest
axis across all evaluated models.
1 Introduction
The rapid advancement of Large Language Mod-
els (LLMs) (Brown et al., 2020) has catalyzed a
paradigm shift across various industries. In the fi-
nancial sector, LLMs are increasingly deployed
to enhance operational efficiency in areas such as
market analysis and the summarization of com-
plex financial reports. However, the financial do-
main demands a level of precision and real-time
information that often exceeds an LLM’s internal
parametric knowledge. Consequently, Retrieval-
Augmented Generation (RAG) has emerged as the
standard architecture for financial AI (Lithgow-
Serrano et al., 2025), providing evidence-based re-
*Equal contribution.
Figure 1: Example of a contradictory hallucination in
K-FinHallu. Hallucinated content is shown in red, hallu-
cination spans are highlighted, and blue text indicates
the clue text.
sponses grounded in external, reliable knowledge
bases.
Despite these advantages, LLMs still produce
hallucinations— outputs that contradict retrieved
documents or lack contextual grounding (Shuster
et al., 2021; Niu et al., 2024). In finance, such hallu-
cinations are particularly critical as even small fac-
tual errors can lead to substantial monetary losses,
compliance risks, and the erosion of user trust.
Therefore, establishing rigorous evaluation bench-
marks to detect domain-specific hallucinations is
essential for reliable deployment.
While existing works have addressed halluci-
nations in the context of RAG, two primary gaps
1arXiv:2605.29523v1  [cs.LG]  28 May 2026

remain in reflecting real-world financial service en-
vironments.First, there is a lack of established
evaluation frameworks for multi-turn conversa-
tional RAG within the financial domain.Finan-
cial services typically operate through continuous
interactions (e.g., chatbots) involving anaphora and
implicit references. Existing benchmarks, which fo-
cus predominantly on single-turn tasks (Niu et al.,
2024; Mishra et al., 2024), struggle to identify hal-
lucinations arising from context-tracking failures,
such as logical inconsistencies with prior turns or
the misinterpretation of implicit references.Sec-
ond, the unique nature of the Korean finan-
cial environment is underrepresented in existing
benchmarks.Existing benchmarks (Kaur et al.,
2025; Wang et al., 2025) focus on Western institu-
tional contexts. These datasets fail to account for
Korea-specific financial systems, such asJeonse
lease system (Yun, 2020), or local regulatory en-
vironments. Furthermore, Korean linguistic fea-
tures—including its agglutinative morphology and
the polysemy of Sino-Korean technical terminol-
ogy—cannot be addressed through simple trans-
lation. To the best of our knowledge, there is cur-
rently no native Korean financial benchmark specif-
ically designed to evaluate hallucinations in multi-
turn dialogues.
To bridge these gaps, we introduceK-FinHallu,
a multi-turn RAG hallucination detection bench-
mark for the Korean financial domain, constructed
from real-world Korean financial documents. We
build K-FinHallu through a pipeline of faithful dia-
logue generation and systematic hallucination in-
jection, covering numerical perturbations, cross-
turn inconsistencies, and failures to abstain when
retrieval is unsuccessful. Figure 1 illustrates a rep-
resentative example of a contradictory hallucina-
tion from K-FinHallu, where the model misinter-
prets specific financial loan regulations. Bench-
marking frontier and open-source LLMs as detec-
tors, we find that even the strongest models strug-
gle with fine-grained financial diagnostics and re-
fusal behavior. Furthermore, we demonstrate that
our pipeline can scale to automatically construct
a training split, on which a fine-tuned 8B model
with rationale supervision matches or outperforms
frontier LLMs. Our primary contributions are as
follows:
•We releaseK-FinHallu, the first multi-turn
hallucination detection benchmark for Korean
financial RAG, built on authentic Korean fi-nancial documents.
•We propose ahierarchical hallucination tax-
onomybased on context answerability that
explicitly accounts for justified abstention in
multi-turn RAG dialogues.
•We demonstrate that our construction pipeline
can automatically generate a training split, on
which a fine-tuned 8B model with rationale
supervision matches or outperforms frontier
LLMs.
2 Related Work
2.1 Hallucination Detection Benchmarks
Recent studies have actively explored hallucination
evaluation in LLMs (Mishra et al., 2024; Seo and
Lim, 2025). HaluEval (Li et al., 2023) introduced a
foundational benchmark for tasks such as question
answering and summarization, but it primarily fo-
cuses on single-turn interactions and short contexts.
RAGTruth (Niu et al., 2024) addresses hallucina-
tions in RAG systems by detecting them across
various domains. However, it focuses on the accu-
racy of individual responses rather than the flow
of multi-turn dialogues. For dialogue-level evalu-
ations, DiaHalu (Chen et al., 2024) and FaithDial
(Dziri et al., 2022) assess the consistency and fac-
tuality of general conversations. HalluLens (Bang
et al., 2025) emphasizes the distinction between
factuality and hallucination, while also noting the
lack of evaluation for cases where a model should
refuse to answer. Our research builds on these find-
ings by focusing on intrinsic hallucination rather
than external knowledge. We specifically evaluate
the logical consistency with retrieved documents
and the ability to abstain from answering in multi-
turn dialogues.
2.2 Financial Hallucination Benchmarks
The financial domain requires high precision and
up-to-date information, making specialized bench-
marks essential. Existing benchmarks such as Al-
phaFin (Li et al., 2024) and FinBen (Xie et al.,
2024) focus on measuring financial analysis and in-
formation retrieval performance. However, they do
not typically verify the presence of hallucinations
in generated responses. While OmniEval (Wang
et al., 2025) evaluates RAG in finance through
multi-dimensional metrics using synthetic data, in-
cluding hallucinations, it tends to overlook the con-
text degradation inherent in long-form financial
2

Is the question answerable by the context?
Is the response faithful
to the provided context?Does the LLM refuse to respond?Yes No
Yes NoYes No
Irrelevancedisconnected 
from user intent
Unverifiableadd unverifiable 
info
False Refusalrefuse despite 
sufficient evidence
Contradictory
 False Acceptance
conflict with context Faithful 
ResponseTrue
RefusalHallucination Typecorrectly abstainFigure 2: Hallucination taxonomy for multi-turn RAG. We categorize five types of hallucinations based on context
answerability: four types of groundedness failures in answerable scenarios where sufficient evidence is provided,
and one type of fabrication in unanswerable scenarios.
reports. PHANTOM (Kaur et al., 2025) introduced
a benchmark for hallucination detection in long-
context financial QA by focusing on source faithful-
ness, which measures whether a response is strictly
based on the provided documents. However, this
benchmark is primarily based on Western financial
systems and English datasets. Although financial
services are often provided through chatbots, exist-
ing benchmarks do not sufficiently reflect hallucina-
tions caused by context loss or anaphora resolution
in multi-turn dialogues.
2.3 Korean Hallucination Benchmarks
Benchmarks for evaluating Korean LLMs have pro-
gressed through datasets such as KMMLU (Son
et al., 2025) and HAE-RAE (Son et al., 2024). De-
spite this progress, datasets specifically designed
for hallucination evaluation remain limited. Exist-
ing benchmarks such as Ko-TruthfulQA (Lin et al.,
2022; Park et al., 2024) often rely on translated
English datasets, which may fail to capture Korean
linguistic and cultural nuances (Hendrycks et al.,
2020; Seo et al., 2024). K-HALU (Seo and Lim,
2025) addresses this by covering domains such as
news and books within a Korean context. However,
it does not account for financial regimes unique
to Korea, such as housing finance regulations. Our
work is the first framework to consider these re-
gional characteristics and linguistic complexities
while evaluating multi-turn hallucinations.
3 Task Definition
3.1 Task Formulation
The primary objective of K-FinHallu is to de-
tect hallucinations in multi-turn RAG dialogues.
Formally, we define a multi-turn RAG dialogueat turn nas a tuple (Hn, qn,Pn), where Hn=
((q1, a1), . . . ,(q n−1, an−1))is the ordered dia-
logue history, qnis the current user query, and
Pn={p1
n, p2
n, . . . , pK
n}is a set of Kretrieved
passages. The model fgenerates a response an=
f(Hn, qn,Pn)that should be strictly grounded in
the provided context Hn∪ Pn. The task is to pre-
dict whether anis faithful or hallucinated at turn n,
given(H n, qn,Pn, an).
To detect hallucinations independently of the
retrieval system’s performance, we adopt a
simulation-based RAGframework. Instead of em-
ploying a live retrieval module which may intro-
duce uncontrolled retrieval errors, the model is pro-
vided with a controlled set of documents Pn. This
setup enables systematic manipulation of evidence
presence, such as intentionally substituting the rel-
evant passage with ahard negativepassage that
is semantically similar to the query qnbut lacks
the critical information required for an accurate
response.
In this work, we focus exclusively on detect-
ingintrinsic hallucinationsbased on source faith-
fulness, in line with (Kaur et al., 2025). This is
particularly vital in the financial domain, where
information is time-sensitive and conclusions must
be derived from authoritative evidence rather than
the model’s parametric knowledge. We assume that
all queries in our benchmark require RAG, involv-
ing specialized expertise that cannot be addressed
without the external context.
3.2 Hallucination Taxonomy
We establish a hierarchical taxonomy of hallucina-
tions in multi-turn RAG dialogues, illustrated in
Figure 2. The top-level criterion isanswerability,
defined as whether the context Hn∪ Pncontains
3

sufficient evidence to address qn. When the query
isanswerable, hallucinations are categorized into
four types:
•Contradictory: The response conflicts with
the provided context, which includes both the
dialogue history Hnand the retrieved pas-
sagesP n.
•Unverifiable: The response includes external
claims or general knowledge that is not sup-
ported by the givenP n.
•Irrelevance: The response is logically discon-
nected from the user’s intent or qn, even if the
facts themselves are present in the document.
•False Refusal: The model refuses to answer
despite sufficient evidence being present in
Pn.
When a query isunanswerabledue to a lack of
evidence in Pn, hallucination occurs if the model
fails to withhold its response, defined asFalse Ac-
ceptance. Conversely,True Refusalrepresents
the correct behavior. Together, they measure the
model’s capacity for honest abstention—judging
when to provide an answer versus when to acknowl-
edge missing evidence.
Empirically, we observe all five types in both
naturally generated outputs from frontier LLMs
and failure cases from a deployed Korean financial
RAG service, detailed in Appendix A.
4 K-FinHallu
4.1 Source Dataset
We use the publicly available “Korean Financial
and Legal Document Machine Reading Compre-
hension” dataset from AI-Hub as the source corpus,
filtered to its financial category. *Each source docu-
ment contains multiple passages p, each associated
with a set of triplets {(˜q(i),˜a(i),˜c(i))}m
i=1, where
˜q(i)is a question, ˜a(i)is the reference answer, and
˜c(i)is theclue text—the evidentiary span within
prequired to derive ˜a(i). Passages are segmented
to a maximum of 1,000 tokens using the KURE
Korean embedding model (Jang et al., 2024). To
prevent data leakage, the corpus is partitioned by
source institution: the test corpus is drawn from
the Financial Supervisory Service (FSS) and the
*AI-Hub is a public data repository operated by Korea’s
National Information Society Agency (NIA): https://aihub.
or.kr/aihubdata/data/view.do?dataSetSn=71610Korea Consumer Agency, while the training corpus
is drawn from disjoint institutions (Appendix B).
The test and training splits contain 272 / 2,732 doc-
uments and 2,064 / 42,364 queries, respectively.
4.2 Faithful Dialogue Generation
We design a dialogue simulation pipeline involv-
ing a user LLM and an assistant LLM (both
powered by GPT-4o) to constructfaithful dia-
logues—hallucination-free conversations that serve
as the basis for hallucination injection. Figure 3
shows an overview of the pipeline. Each dialogue
is initiated with a seed question sampled from the
source dataset.
Query GenerationThe user LLM selects a can-
didate question ˜qnfrom existing questions in the
same document or in the top-10 most similar pas-
sages by embedding similarity, maintaining a co-
herent conversational flow. If no relevant candi-
date exists, the user LLM generates a new ques-
tion based on the current passage. In addition, to
evaluate context tracking across turns, we intro-
duce areconfirmquestion after the dialogue has
sufficiently progressed ( n≥4 ). These turns are
randomly inserted to prompt the assistant to verify
previous statements in Hn. Finally, the selected ˜qn
is rewritten into a natural, context-dependent ut-
terance qnby incorporating stylistic paraphrasing,
coreference, and ellipsis.
Response GenerationThe assistant LLM gener-
ates each response angrounded in the gold passage
p∗
nand clue text cnfrom the source dataset, fol-
lowing the approach of CoQA (Reddy et al., 2019).
This is intended to yield a faithful dialogue that
serves as a clean substrate for hallucination injec-
tion.
Retrieval SimulationFor each turn, we con-
struct a retrieval context Pnto simulate realistic
RAG settings. To bypass the ambiguity of conversa-
tional queries qn(e.g., anaphora), we use the decon-
textualized standalone ˜qnas the retrieval key. In ad-
dition, we refresh Pnwith only the top- K(K=10 )
passages for the current turn, rather than accumu-
lating across turns, to prevent retrieval noise and
maintain information density (Katsis et al., 2025).
We vary the composition of Pnby answerability,
following Section 3.2. Foranswerablescenarios,
Pncontains the gold passage p∗(and possibly other
positive passages p+that also support the query).
Forunanswerablescenarios, Pnconsists entirely
4

Figure 3: Overview of the K-FinHallu construction pipeline. We first generate faithful dialogues and then inject
hallucinations to construct the final dataset. Highlighted text illustrates how faithful responses are transformed under
different hallucination types.
of hard negatives ( p−)—passages semantically sim-
ilar to ˜qnthat lack the evidence required for a re-
sponse. Hard negatives are selected by a two-step
automated pipeline that retains passages where the
answer cannot be derived from the context and the
clue text is absent (Appendix C), followed by man-
ual verification.
Quality ControlWe evaluate the generated dia-
logues using an LLM-as-a-judge across five crite-
ria:answerability,plausibility,correctness,diver-
sity, andcoherency(Lee et al., 2024). Dialogues
scoring below a predefined threshold on any crite-
rion are filtered out. We validate the reliability of
this LLM judge against four human annotators and
observe robust agreement based on Gwet’s AC1†
(Table 1). This filtering retains 74.3% of test and
77.0% of train dialogues. For the test set, human
annotators further review all retained dialogues and
revise as needed (details in Appendix D).
4.3 Hallucination Injection
To construct a challenging benchmark, we develop
an automated pipeline to inject subtle yet critical
hallucinations into faithful dialogues. Our injection
†We use Gwet’s AC1 over Cohen’s κto avoid the kappa
paradox under high agreement with skewed class distribu-
tions (Gwet, 2008).strategy is designed with financial-domain experts
and informed by real-world hallucination patterns
observed in deployed Korean financial RAG sys-
tems (Appendix A.2).
The pipeline identifies contextually optimal
turns for injection based on hallucination type (e.g.,
Contradictoryhallucinations target turns dense in
numerical data or financial jargon). Hallucinations
are injected via minimal edits to key words or
phrases, preserving the original response struc-
ture. We execute this pipeline using both GPT-4o
and Gemini-2.5-Flash for the test split to mitigate
model-specific artifacts, and GPT-4o only for the
training split. The strategies are as follows:
•Contradictory: Induces subtle but critical de-
viations through four primary error types: (1)
Financial Term Misunderstanding, (2)Num-
ber Error, (3)Modifier Change, and (4)In-
consistencywith prior dialogue history.
•Unverifiable: Inserts fabricated details or sub-
jective opinions to ensure the response lacks
evidentiary support from the context.
•Irrelevance: Modifies the response to ignore
query constraints or address a different aspect
of the same topic.
5

•False Acceptance: Replaces Pnwith all
hard negatives while retaining the original re-
sponse, simulating model acceptance of an
unanswerable query.
•False Refusal: Replaces the response with a
refusal string while keeping the original Pn
intact.
Subtype-level injection strategies and prompt
templates are provided in Appendix G.
Quality ControlFor the training split, each in-
jected turn is rated by an LLM judge on a 1–3
scale forappropriateness(whether the intended
hallucination type is correctly injected) andnatu-
ralness(whether the injected response fits naturally
in context), with turns below 2 on either dimen-
sion discarded. This filtering retains 85.2% of train
dialogues. We validate this judge against human
annotations (Table 1). For the test set, human anno-
tators review and revise all injected turns without
automated filtering. Detailed rubrics and annotation
procedures are in Appendix D.
MetricHuman-Human Human-LLM
Agreement AC1 Agreement AC1
Dialogue Quality
Answerability 0.95 0.95 0.91 0.90
Plausibility 0.79 0.76 0.86 0.85
Correctness 0.96 0.96 0.97 0.96
Diversity 0.78 0.72 0.73 0.64
Coherency 0.75 0.70 0.70 0.64
Injected Hallucination Quality
Appropriateness 0.85 0.82 0.77 0.73
Naturalness 0.80 0.77 0.69 0.64
Table 1: Agreement and Gwet’s AC1 for the two quality-
control stages: dialogue quality and injected hallucina-
tion quality. For each stage, four annotators each eval-
uate 30 of 40 sampled items (each spanning multiple
turns), with three-way overlap per item.
4.4 Dataset Statistics
K-FinHallu comprises a test split for evaluation
and a training split for fine-tuning (Table 2). The
test split contains 808 evaluation samples, com-
prising 404 faithful and 404 hallucinated dialogues
derived from 202 base dialogues. Each base dia-
logue contains between 4 and 8 turns, where exactly
one hallucination is injected per base dialogue, dis-
tributed across turn positions to evaluate consis-
tency throughout the conversation. About 9.7% of
samples present unanswerable scenarios to evaluate
honest abstention. The training split contains 2,624
samples: 1,312 hallucinated dialogues paired withTest Train
Base Dialogues
Total # of Dialogues 202 1,312
# of Turns (min–max) 4–8 4–8
K-FinHallu
Total # of Dialogues 808 2,624
Total # of Unique Passages 1,273 11,281
# of Turns (min–max) 1–8 1–8
Avg. # of Turns 3.72 3.61
Avg. # of Question Words 9.7 10.4
Avg. # of Response Words 16.1 18.1
Unanswerable (%) 9.7 11.4
Table 2: Dataset statistics. Base Dialogues are faithful di-
alogues before hallucination injection. K-FinHallu com-
bines hallucinated and faithful samples derived from
these base dialogues.
their faithful counterparts, via automated quality
filtering. Figure 4 presents the test-set distribution
of financial domains, question types, and halluci-
nation types, and Appendix B compares domain
distributions across splits.
5 Experiments
5.1 Evaluation Setup and Metrics
We evaluate the detection task defined in Sec-
tion 3.1 at the turn level to simulate real-time mon-
itoring of responses within a RAG pipeline. The
evaluation proceeds in two settings:Binary classi-
ficationdetermines the presence of a hallucination,
reporting accuracy, precision, recall, and F1.Four-
class classificationevaluates answerability judg-
ment and refusal behavior acrossFaithful Answer,
Hallucination(coveringContradictory,Unverifi-
able,Irrelevance, andFalse Acceptance),False
Refusal, andTrue Refusal, reporting per-class ac-
curacy. For readability, we collapseFalse Refusal
andTrue Refusalinto a singleRefusalclass.
5.2 Baselines
To detect hallucination, we select various LLMs
that have shown strong performance in the Ko-
rean NLP community and are widely used in
research or industry: (1)Korean-centricmodels
(kanana-2-30b-a3b-instruct (Kanana LLM, 2025),
EXAONE-4.0-32B (LG AI Research, 2025)), (2)
open-sourcemodels (Llama-3.1 (8B, 70B), Llama-
3.3 (70B) (Grattafiori et al., 2024), Qwen3 se-
ries (Team, 2025)), and (3)closed-sourcemodels
(Gemini-2.5 (Flash, Pro) (Comanici et al., 2025),
GPT-4o (OpenAI et al., 2024), GPT-5 (OpenAI,
2025)). For reproducibility, all models are evalu-
ated at a temperature of 0 where applicable.
6

Financial Economics (35.2%)
Financial Institution Management (24.1%)Consumer
Economics (21.4%)Insurance (12.7%)Economic Policy (5.9%)
Accounting (0.3%)Financial Management (0.3%)
Domain
Follow-up (61.8%)Initial (26.7%)Reconfirm (9.6%)Independent (1.9%)
Question
TypesContradictory (39.1%)
Unverifiable (19.3%)Irrelevance (19.1%)False Refusal (12.9%)False Acceptance (9.7%)
Hallucination
TypesFigure 4: Distribution of K-FinHallu test set characteristics. The figure presents the percentage distributions of
financial domains, question types, and hallucination types.
Model Acc. Prec. Rec. F1
Korean-centric
kanana-2-30b-a3b 0.533 0.520 0.866 0.650
EXAONE-4.0-32B 0.717 0.700 0.757 0.728
Open-source
Llama-3.1-8B 0.537 0.5200.9580.674
Llama-3.1-70B 0.700 0.659 0.832 0.735
Llama-3.3-70B 0.722 0.672 0.864 0.756
Qwen3-8B 0.720 0.850 0.533 0.655
Qwen3-14B 0.756 0.881 0.590 0.707
Qwen3-32B 0.775 0.905 0.614 0.732
Closed-source
GPT-4o 0.715 0.691 0.780 0.733
GPT-5 0.833 0.877 0.775 0.823
Gemini-2.5-Flash0.8640.8830.839 0.860
Gemini-2.5-Pro 0.859 0.9120.795 0.849
Table 3: Performance of evaluated LLMs on binary hal-
lucination detection. Metrics include Accuracy (Acc.),
Precision (Prec.), Recall (Rec.), and F1-score (F1). Bold
and underlined values denote the best and second-best
scores per column.
In addition to these off-the-shelf baselines, we
fine-tune Qwen3-8B on the training split using
LoRA (Hu et al., 2022) for the four-class detec-
tion task. We train two variants:Qwen3-8B SFT
is trained to predict only the class label, whereas
Qwen3-8B SFT-Ris additionally supervised with
a templated rationale that mirrors our taxonomy’s
two-step reasoning: assessing answerability, then
evaluating the response (Appendix E).
5.3 Main Results
Detection PerformanceBinary detection results
are summarized in Table 3. Closed-source models
lead overall, with Gemini-2.5-Flash achieving the
highest F1 of 0.860, followed by Gemini-2.5-Pro
(0.849) and GPT-5 (0.823). Among open-source
models, Qwen3-32B (0.732) is comparable to GPT-
4o (0.733), with consistent scaling gains within the
Qwen3 family. Korean-centric models lag behind
their global counterparts of similar size. A clearModel Faith. Hallu. Ref. Overall
Korean-centric
kanana-2-30b-a3b 0.597 0.494 0.308 0.515
EXAONE-4.0-32B 0.727 0.514 0.471 0.601
Open-source
Llama-3.1-8B 0.114 0.537 0.452 0.342
Llama-3.1-70B 0.781 0.670 0.471 0.693
Llama-3.3-70B 0.756 0.682 0.471 0.687
Qwen3-8B0.9860.446 0.490 0.688
Qwen3-14B 0.977 0.554 0.567 0.741
Qwen3-32B 0.969 0.551 0.625 0.743
Closed-source
GPT-4o 0.835 0.642 0.375 0.692
GPT-5 0.969 0.810 0.625 0.855
Gemini-2.5-Flash 0.946 0.795 0.673 0.845
Gemini-2.5-Pro 0.940 0.815 0.683 0.853
Fine-tuned (ours)
Qwen3-8B SFT 0.983 0.764 0.433 0.822
Qwen3-8B SFT-R 0.9720.864 0.750 0.896
Table 4: Per-class accuracy on the four-class detection
task under the three-class collapsed view.Faith.: Faith-
ful.Hallu.: Hallucinated.Ref.: Refusal (False + True).
Overallis the micro-average. Full breakdown in Table 8.
precision–recall trade-off distinguishes model fam-
ilies: Llama-3.1-8B exhibits a “pessimistic bias,”
with the highest recall (0.958) but the lowest preci-
sion (0.520), suggesting a tendency to over-identify
faithful responses as hallucinations. The Qwen3 se-
ries shows the opposite pattern, favoring precision
at the expense of recall.
Closing the Gap with Fine-tuningTable 4 re-
ports the four-class results under a three-class col-
lapsed view. The three axes exhibit distinct diffi-
culty levels: Faithful classification is near-ceiling
for most models, Hallucination detection shows the
largest inter-model variance, and Refusal scores
below 0.50 for most base models. Overall, GPT-5
attains the highest accuracy among base models
(0.855), with the Gemini-2.5 series close behind.
Open-source base models trail substantially, with
Qwen3-32B the strongest at 0.743.
7

Fine-tuning Qwen3-8B with rationale supervi-
sion (Qwen3-8B SFT-R) achieves an overall ac-
curacy of 0.896, outperforming all base models.
The largest gain over the base Qwen3-8B is on
Hallucination detection, where accuracy improves
from 0.446 to 0.864, exceeding GPT-5. Notably,
rationale supervision is critical, as Qwen3-8B SFT
without rationale scores 0.822.
Although Qwen3-8B SFT-R achieves the high-
est Refusal accuracy at 0.750, Refusal remains the
weakest axis across all models. SeparatingTrue
RefusalfromFalse Refusal(Table 8) reveals two
failure patterns among base models: open-source
and Korean-centric models fail to distinguish be-
tween the two refusal types, predicting one almost
exclusively. Closed-source models detectFalse Re-
fusalaccurately but struggle to recognizeTrue Re-
fusalas a valid abstention, treating refusal itself as
a negative signal.
5.4 Analysis
Multi-class DiagnosticsTo further probe diag-
nostic capability, we evaluate multi-class classifica-
tion on the hallucinated samples to identify the un-
derlying cause amongContradictory,Unverifiable,
Irrelevance,False Acceptance, andFalse Refusal.
Results (Table 9) reveal a substantial drop from
binary detection, confirming that diagnosing the
causeof a hallucination is harder than detecting its
presence. The Qwen3 series follows a clear scaling
trend where performance improves with model size
(0.387 to 0.492 macro-F1 from 8B to 32B), while
Korean-centric models and GPT-4o tend to collapse
into a single dominant type, scoring high only in
specific categories while failing in others. Notably,
confusion betweenContradictoryandUnverifiable
persists even for GPT-5 (confusion matrices in Ta-
ble 10).
Contradictory Sub-type AnalysisWe further
decomposeContradictoryhallucinations, the most
frequent type in our benchmark, into four sub-
causes. This analysis uses the GPT-4o-injected sub-
set where sub-type labels are manually annotated.
As shown in Figure 5, error rates forNumber Error
andModifier Changedecrease with model scale
in the Qwen3 series, butFinancial Term Misun-
derstandingshows minimal improvement even at
32B. This suggests that effective detection in the fi-
nancial domain requires nuanced domain expertise
beyond surface-level lexical matching.
0 20 40 60 80
Error Rate (%)Financial T erm
MisunderstandingInconsistencyModifier/Qualifier
ChangeNumber Error
84.6%38.6%11.1%33.3%
76.9%22.7%22.2%
69.2%25.0%Qwen3-8B
Qwen3-14B
Qwen3-32BFigure 5: Error rates of the Qwen3 series onContra-
dictorysub-types (GPT-4o-injected subset). Each bar
represents the proportion of hallucinations undetected
by the model.
Cross-source AnalysisK-FinHallu employs
both GPT-4o and Gemini-2.5-Flash as injection
sources over the same faithful dialogues for the test
split to mitigate model-specific artifacts (Table 11).
The relative ordering of strong and weak models is
largely preserved across both subsets, but the two
sources differ in difficulty: for most hallucination
types the gap is modest ( ±10%p), butIrrelevance
shows a pronounced difference. GPT-4o produces
topic-level drift that is readily identifiable, whereas
Gemini-2.5-Flash substitutes adjacent information
from the same document that appears relevant but
does not answer the query (e.g., Gemini-2.5-Pro
drops from 0.93 to 0.68). This variation in diffi-
culty demonstrates that source diversity strengthens
evaluation robustness by exposing different failure
modes.
6 Conclusion
We introduced K-FinHallu, a multi-turn RAG hal-
lucination detection benchmark specialized for the
Korean financial domain. By establishing a hier-
archical taxonomy based on context answerability
and implementing a hallucination injection pipeline
using authentic financial documents, we bench-
marked frontier and open-source LLMs as halluci-
nation detectors. Our results demonstrate that even
frontier models struggle with answerability judg-
ment and fine-grained diagnosis, despite competent
binary detection, highlighting justified abstention
as a critical yet under-evaluated dimension in finan-
cial RAG. In particular,Financial Term Misunder-
standingpersists as a primary bottleneck regardless
of model scale, suggesting that even frontier LLMs
struggle to capture the semantic shifts caused by
subtle alterations in Korean financial jargon. A fine-
tuned 8B model with rationale supervision achieves
competitive or superior performance to frontier
baselines, offering a practical path to open-source
8

detector development. We release K-FinHallu un-
der the CC-BY-NC 4.0 license to support further
research in this direction.
Limitations
Our study is subject to several limitations. First,
while K-FinHallu is curated from authoritative Ko-
rean financial sources, its coverage remains focused
on core domains and may underrepresent highly
specialized sub-sectors such as complex deriva-
tives or investment banking. Second, we inject
exactly one hallucination per dialogue by design
to support precise, turn-level diagnosis under our
real-time monitoring setting. This controlled setup
does not capture co-occurring or compounding hal-
lucinations that may arise in real-world interac-
tions, which we leave to future work. Third, the
dialogues in K-FinHallu were generated via LLM
simulation, a process that may pass down generator-
specific linguistic patterns or artifacts to the dataset,
though all five hallucination types were also ob-
served in naturally generated responses from fron-
tier LLMs. Finally, our terminology-related find-
ings are grounded in Korean regulatory language
and market context, so transferability to other lan-
guages or jurisdictions may be limited.
Acknowledgments
This paper used datasets from ‘The Open AI
Dataset Project (AI-Hub, S. Korea)’. All data
information can be accessed through ‘AI-Hub
(www.aihub.or.kr)’.
References
Yejin Bang, Ziwei Ji, Alan Schelten, Anthony Hartshorn,
Tara Fowler, Cheng Zhang, Nicola Cancedda, and
Pascale Fung. 2025. Hallulens: Llm hallucination
benchmark.arXiv preprint arXiv:2504.17550.
Tom Brown, Benjamin Mann, Nick Ryder, Melanie
Subbiah, Jared D Kaplan, Prafulla Dhariwal, Arvind
Neelakantan, Pranav Shyam, Girish Sastry, Amanda
Askell, and 1 others. 2020. Language models are
few-shot learners.Advances in neural information
processing systems, 33:1877–1901.
Kedi Chen, Qin Chen, Jie Zhou, He Yishen, and Liang
He. 2024. Diahalu: A dialogue-level hallucination
evaluation benchmark for large language models. In
Findings of the Association for Computational Lin-
guistics: EMNLP 2024, pages 9057–9079.
Gheorghe Comanici, Eric Bieber, Mike Schaekermann,
Ice Pasupat, Noveen Sachdeva, Inderjit Dhillon, Mar-
cel Blistein, Ori Ram, Dan Zhang, Evan Rosen, LukeMarris, Sam Petulla, Colin Gaffney, Asaf Aharoni,
Nathan Lintz, Tiago Cardal Pais, Henrik Jacobs-
son, Idan Szpektor, Nan-Jiang Jiang, and 3416 oth-
ers. 2025. Gemini 2.5: Pushing the frontier with
advanced reasoning, multimodality, long context,
and next generation agentic capabilities.Preprint,
arXiv:2507.06261.
Nouha Dziri, Ehsan Kamalloo, Sivan Milton, Os-
mar R Zaïane, Mo Yu, Edoardo M Ponti, and Siva
Reddy. 2022. Faithdial: A faithful benchmark for
information-seeking dialogue.Transactions of the
Association for Computational Linguistics, 10:1473–
1490.
Aaron Grattafiori, Abhimanyu Dubey, Abhinav Jauhri,
Abhinav Pandey, Abhishek Kadian, Ahmad Al-Dahle,
Aiesha Letman, Akhil Mathur, Alan Schelten, Alex
Vaughan, Amy Yang, Angela Fan, Anirudh Goyal,
Anthony Hartshorn, Aobo Yang, Archi Mitra, Archie
Sravankumar, Artem Korenev, Arthur Hinsvark, and
542 others. 2024. The llama 3 herd of models.
Preprint, arXiv:2407.21783.
Kilem Li Gwet. 2008. Computing inter-rater reliability
and its variance in the presence of high agreement.
British Journal of Mathematical and Statistical Psy-
chology, 61(1):29–48.
Dan Hendrycks, Collin Burns, Steven Basart, Andy Zou,
Mantas Mazeika, Dawn Song, and Jacob Steinhardt.
2020. Measuring massive multitask language under-
standing.arXiv preprint arXiv:2009.03300.
Edward J Hu, Yelong Shen, Phillip Wallis, Zeyuan
Allen-Zhu, Yuanzhi Li, Shean Wang, Liang Wang,
Weizhu Chen, and 1 others. 2022. Lora: Low-rank
adaptation of large language models.Iclr, 1(2):3.
Youngjoon Jang, Junyoung Son, and Taemin Lee. 2024.
Kure: Korea university retrieval embedding model.
https://github.com/nlpai-lab/KURE.
Kanana LLM. 2025. Kanana-2 llm.
Yannis Katsis, Sara Rosenthal, Kshitij Fadnis, Chu-
laka Gunasekara, Young-Suk Lee, Lucian Popa, Vraj
Shah, Huaiyu Zhu, Danish Contractor, and Marina
Danilevsky. 2025. Mtrag: A multi-turn conversa-
tional benchmark for evaluating retrieval-augmented
generation systems.Transactions of the Association
for Computational Linguistics, 13:784–808.
Lanlan Ji Dominic Seyler Gunkirat Kaur, Manjunath
Hegde Koustuv Dasgupta Bing Xiang, and Goldman
Sachs. 2025. Phantom: A benchmark for hallucina-
tion detection in financial long-context qa.
Woosuk Kwon, Zhuohan Li, Siyuan Zhuang, Ying
Sheng, Lianmin Zheng, Cody Hao Yu, Joseph Gon-
zalez, Hao Zhang, and Ion Stoica. 2023. Efficient
memory management for large language model serv-
ing with pagedattention. InProceedings of the 29th
symposium on operating systems principles, pages
611–626.
9

Young-Suk Lee, Chulaka Gunasekara, Danish Contrac-
tor, Ramón Fernandez Astudillo, and Radu Florian.
2024. Multi-document grounded multi-turn synthetic
dialog generation.arXiv preprint arXiv:2409.11500.
LG AI Research. 2025. Exaone 4.0: Unified large lan-
guage models integrating non-reasoning and reason-
ing modes.arXiv preprint arXiv:2507.11407.
Junyi Li, Xiaoxue Cheng, Wayne Xin Zhao, Jian-Yun
Nie, and Ji-Rong Wen. 2023. Halueval: A large-
scale hallucination evaluation benchmark for large
language models.arXiv preprint arXiv:2305.11747.
Xiang Li, Zhenyu Li, Chen Shi, Yong Xu, Qing Du,
Mingkui Tan, and Jun Huang. 2024. Alphafin: Bench-
marking financial analysis with retrieval-augmented
stock-chain framework. InProceedings of the 2024
joint international conference on computational lin-
guistics, language resources and evaluation (LREC-
COLING 2024), pages 773–783.
Stephanie Lin, Jacob Hilton, and Owain Evans. 2022.
Truthfulqa: Measuring how models mimic human
falsehoods. InProceedings of the 60th annual meet-
ing of the association for computational linguistics
(volume 1: long papers), pages 3214–3252.
Oscar Lithgow-Serrano, David Kletz, Vani Kanjiran-
gat, David Adametz, Marzio Lunghi, Claudio Bone-
sana, Matilde Tristany Farinha, Yuntao Li, Detlef
Repplinger, Marco Pierbattista, and 1 others. 2025.
Assessing rag system capabilities on financial docu-
ments. InThe 10th Workshop on Financial Technol-
ogy and Natural Language Processing, page 124.
Abhika Mishra, Akari Asai, Vidhisha Balachandran,
Yizhong Wang, Graham Neubig, Yulia Tsvetkov, and
Hannaneh Hajishirzi. 2024. Fine-grained hallucina-
tion detection and editing for language models.arXiv
preprint arXiv:2401.06855.
Cheng Niu, Yuanhao Wu, Juno Zhu, Siliang Xu, Kashun
Shum, Randy Zhong, Juntong Song, and Tong Zhang.
2024. Ragtruth: A hallucination corpus for develop-
ing trustworthy retrieval-augmented language models.
InProceedings of the 62nd Annual Meeting of the
Association for Computational Linguistics (Volume
1: Long Papers), pages 10862–10878.
OpenAI, :, Aaron Hurst, Adam Lerer, Adam P. Goucher,
Adam Perelman, Aditya Ramesh, Aidan Clark,
AJ Ostrow, Akila Welihinda, Alan Hayes, Alec
Radford, Aleksander M ˛ adry, Alex Baker-Whitcomb,
Alex Beutel, Alex Borzunov, Alex Carney, Alex
Chow, Alex Kirillov, and 401 others. 2024. Gpt-4o
system card.Preprint, arXiv:2410.21276.
OpenAI. 2025. Gpt-5 system card. https://cdn.
openai.com/gpt-5-system-card.pdf . OpenAI
system card, accessed 2026-01-06.
Chanjun Park, Hyeonwoo Kim, Dahyun Kim, Seongh-
wan Cho, Sanghoon Kim, Sukyung Lee, Yungi Kim,
and Hwalsuk Lee. 2024. Open ko-llm leaderboard:
Evaluating large language models in korean with ko-
h5 benchmark.arXiv preprint arXiv:2405.20574.Siva Reddy, Danqi Chen, and Christopher D Manning.
2019. Coqa: A conversational question answering
challenge.Transactions of the Association for Com-
putational Linguistics, 7:249–266.
Jaehyung Seo, Jaewook Lee, Chanjun Park, SeongTae
Hong, Seungjun Lee, and Heui-Seok Lim. 2024. Ko-
commongen v2: A benchmark for navigating korean
commonsense reasoning challenges in large language
models. InFindings of the Association for Computa-
tional Linguistics ACL 2024, pages 2390–2415.
Jaehyung Seo and Heuiseok Lim. 2025. K-halu: Multi-
ple answer korean hallucination benchmark for large
language models. InThe Thirteenth International
Conference on Learning Representations.
Kurt Shuster, Spencer Poff, Moya Chen, Douwe Kiela,
and Jason Weston. 2021. Retrieval augmentation
reduces hallucination in conversation.arXiv preprint
arXiv:2104.07567.
Guijin Son, Hanwool Lee, Sungdong Kim, Seungone
Kim, Niklas Muennighoff, Taekyoon Choi, Cheon-
bok Park, Kang Min Yoo, and Stella Biderman. 2025.
Kmmlu: Measuring massive multitask language un-
derstanding in korean. InProceedings of the 2025
Conference of the Nations of the Americas Chap-
ter of the Association for Computational Linguistics:
Human Language Technologies (Volume 1: Long Pa-
pers), pages 4076–4104.
Guijin Son, Hanwool Lee, Suwan Kim, Huiseo Kim,
Jae cheol Lee, Je Won Yeom, Jihyu Jung, Jung woo
Kim, and Songseong Kim. 2024. Hae-rae bench:
Evaluation of korean knowledge in language mod-
els. InProceedings of the 2024 Joint International
Conference on Computational Linguistics, Language
Resources and Evaluation (LREC-COLING 2024),
pages 7993–8007.
Zhongkai Sun, Yingxue Zhou, Jie Hao, Xing Fan, Yan-
bin Lu, Chengyuan Ma, Wei Shen, and Chenlei Guo.
2023. Improving contextual query rewrite for conver-
sational ai agents through user-preference feedback
learning. InProceedings of the 2023 Conference on
Empirical Methods in Natural Language Processing:
Industry Track, pages 432–439.
Qwen Team. 2025. Qwen3 technical report.Preprint,
arXiv:2505.09388.
Shuting Wang, Jiejun Tan, Zhicheng Dou, and Ji-Rong
Wen. 2025. Omnieval: An omnidirectional and auto-
matic rag evaluation benchmark in financial domain.
InProceedings of the 2025 Conference on Empiri-
cal Methods in Natural Language Processing, pages
5737–5762.
Qianqian Xie, Weiguang Han, Zhengyu Chen, Ruoyu
Xiang, Xiao Zhang, Yueru He, Mengxi Xiao, Dong
Li, Yongfu Dai, Duanyu Feng, and 1 others. 2024.
Finben: A holistic financial benchmark for large lan-
guage models.Advances in Neural Information Pro-
cessing Systems, 37:95716–95743.
10

Sungjin Yun. 2020. Neighborhood effects of housing
program using jeonse in korea.International Journal
of Housing Markets and Analysis, 14(2):305–316.
Yuze Zhao, Jintao Huang, Jinghan Hu, Xingjun Wang,
Yunlin Mao, Daoze Zhang, Zeyinzi Jiang, Zhikai Wu,
Baole Ai, Ang Wang, Wenmeng Zhou, and Yingda
Chen. 2024. Swift:a scalable lightweight infrastruc-
ture for fine-tuning.Preprint, arXiv:2408.05517.
Yingxue Zhou, Jie Hao, Mukund Rungta, Yang Liu, Eu-
nah Cho, Xing Fan, Yanbin Lu, Vishal Vasudevan,
Kellen Gillespie, and Zeynab Raeesy. 2023. Unified
contextual query rewriting. InProceedings of the
61st Annual Meeting of the Association for Computa-
tional Linguistics (Volume 5: Industry Track), pages
608–615.
AReal-world Validation of Hallucination
Taxonomy
K-FinHallu uses controlled hallucination injection
to enable systematic and balanced evaluation. A
natural concern, however, is whether the resulting
benchmark (both its taxonomy and the injected ex-
amples) reflects hallucinations encountered in real
deployment. We address this through three lines
of empirical evidence: (i) the natural occurrence
rate and distribution of hallucinations from frontier
LLMs under free generation, (ii) aggregate statis-
tics from a real-world RAG deployment, and (iii)
qualitative differences across source models used
for injection.
A.1 Natural Hallucination Distribution from
Frontier LLMs
To assess whether our taxonomy reflects actual
model hallucination patterns, GPT-4o and Gemini-
2.5-Flash generate a response for each of the 404
faithful samples in K-FinHallu (temperature 0.7),
yielding 808 model responses in total. Each re-
sponse is then labeled by a separate LLM judge
(GPT-5-mini) against the original answer and its
clue text, with refusal cases detected by keyword
matching.
Of the 808 responses, 282 are classified as hal-
lucinations. The majority (247) belong to refusal-
related types, comprising 189False Refusaland
58False Acceptancecases. Content-based halluci-
nations (Contradictory,Unverifiable,Irrelevance)
account for only 35 cases, with GPT-4o produc-
ing 14 (3.5%) and Gemini-2.5-Flash producing 21
(5.2%). The per-model totals are 214 for GPT-4o
and 68 for Gemini-2.5-Flash.
Despite this low rate, all five hallucination types
in our taxonomy are observed in these frontierLLMs (Table 5). The two models, however, ex-
hibit distinct biases: GPT-4o concentrates onFalse
Refusal(186 of 214), while Gemini-2.5-Flash con-
centrates onFalse Acceptance(44 of 68). Since
each model has different weak spots and natural
generation alone produces a highly skewed distri-
bution across the five types, controlled injection is
required for balanced evaluation, especially given
the high-stakes nature of financial RAG.
Type GPT-4o Gemini-2.5-Flash Total
Contradictory 9 10 19
Unverifiable 4 9 13
Irrelevance 1 2 3
False Refusal 186 3 189
False Acceptance 14 44 58
Total 214 68 282
Table 5: Distribution of hallucinations naturally pro-
duced by GPT-4o and Gemini-2.5-Flash on the 404
K-FinHallu faithful samples (temperature 0.7, judged
by GPT-5-mini).
A.2 Real-world RAG Deployment Statistics
We additionally analyze hallucination prevalence
in a real RAG-based financial service through an
internal field test of a deployed system. Among
2,265 RAG-generated responses evaluated by hu-
man assessors, 960 were flagged as errors (a broad
category including hallucination, inappropriate han-
dling of prompt attacks, answers requiring trim-
ming, and outdated reference documents). Of these
flagged errors, 434 (approximately 45%) under-
went detailed manual inspection, and 224 (51.6%)
were confirmed as hallucinations. Extrapolating to
the full set yields an estimated hallucination rate of
approximately 21.9% in deployment.
The per-type breakdown (see Table 6) indicates
that the types defined in our taxonomy are ob-
served in real deployment, suggesting that the tax-
onomy reflects practically relevant failure modes.
The per-type ratios differ from our balanced injec-
tion scheme, and the distribution in deployment
may shift with service settings; we therefore view
our balanced injection as a controlled stress test
rather than a faithful reproduction of any single
operational distribution.
The data was collected and analyzed in collabo-
ration with financial domain experts at an industry
partner operating a deployed RAG-based financial
service. Hallucination judgments were made by
domain experts using a shared rubric, and category-
11

Type (%) Subtype Count
Contradictory
(20.09%)Inconsistency 11
Financial Term Misunderstanding 11
Modifier Change 1
Number Error 22
Unverifiable
(36.61%)Fabricated Detail 73
Subjective Opinion 9
Irrelevance
(9.37%)Condition Not Satisfied 10
Question Misunderstanding 3
Off-topic Drift 8
False Refusal (4.91%) — 11
False Acceptance (29.02%) — 65
Total (100.00%) 224
Table 6: Per-type and per-subtype distribution of hallucinations identified from a pilot study of a RAG-based
financial chatbot ( n= 224 hallucinations out of 434 manually inspected error cases). Type-level percentages (shown
next to each type name) are computed over the 224 confirmed hallucinations.
level labels were further verified through manual
inspection. The raw cases involve proprietary user
interactions and cannot be released; only aggregate
statistics are reported here.
A.3 Cross-source Characteristics of Injected
Hallucinations
For the test split, we use both GPT-4o and Gemini-
2.5-Flash as injection sources to mitigate source-
model-specific artifacts. Detector accuracy on the
two subsets is reported quantitatively in Table 11,
and we additionally observe systematic qualitative
differences between the two sources, particularly
forIrrelevance. GPT-4o-injectedIrrelevancetends
to produce clearly off-topic responses, whereas
Gemini-injectedIrrelevancesubstitutes adjacent
information from the same context, making detec-
tion considerably harder. For instance, when asked
when the monitoring of MMF mark-to-market
preparation begins, a Gemini-injected response
cites the regulation’s effective date drawn from
the same timeline. This information lies within
the source context but answers a different ques-
tion. These cross-source differences suggest that
the characteristics of injected hallucinations depend
on the source model, and that source-model diver-
sity may be a useful axis for improving benchmark
robustness.
B Dataset Distribution Across Splits
The corpus is partitioned by source institution to
prevent data leakage (Section 4.4). The test split
is drawn from the Financial Supervisory Service
(∼72%) and the Korea Consumer Agency ( ∼28%).The training split is drawn primarily from the Bank
of Korea and the Financial Services Commission,
with smaller contributions from the Korea Institute
of Finance and the Ministry of Employment and
Labor.
As shown in Figure 6, both splits are dominated
byFinancial EconomicsandFinancial Institution
Management. The training split additionally cov-
ersEconomic Policy(17.1%),Public Finance, and
Business Ethics, which are sparse or absent in the
test split, while the test split is more concentrated
inConsumer Economics(21.4%) andInsurance
(12.7%). These differences reflect the distinct sub-
ject coverage of the source institutions.
C Retrieval Simulation
To rigorously evaluate context adherence, K-
FinHallu constructs Pnto include bothanswerable
andunanswerablecontexts. For each query, the top-
kcandidate passages are retrieved using the KURE
embedding model (Jang et al., 2024) based on co-
sine similarity with the decontextualized query ˜qn,
which mitigates coreference and ellipsis in the con-
versational utterance qn, following prior multi-turn
RAG work (Zhou et al., 2023; Sun et al., 2023;
Katsis et al., 2025). Following the notation in Sec-
tion 4.4, each retrieved passage falls into one of
three types:
•gold passage p∗: the direct source of the re-
sponse, containing the grounding cluec n.
•positive passage p+: not the gold passage, but
still supports the answer through paraphrase
or topical overlap withp∗.
12

0 10 20 30 40
Percentage (%)Business EthicsAccountingInsuranceFinancial ManagementPublic FinanceConsumer EconomicsEconomic PolicyFinancial Institution ManagementFinancial Economics
0.0%0.3%12.7%0.3%0.0%21.4%5.9%24.1%35.2%Test
0 10 20 30 40
Percentage (%)1.4%2.0%2.7%2.7%4.3%6.1%17.1%24.3%39.3%TrainFigure 6: Document domain distribution in the test (left) and training (right) splits. The same domain is assigned a
consistent color across both panels.
•hard negative p−: semantically similar to ˜qn
but contains neither the answer nor its ground-
ing clue.
Answerablecontexts always include p∗within the
top-k, whileunanswerablecontexts consist solely
ofp−. However, similarity-based retrieval alone
cannot prevent p+from mixing into the unanswer-
able candidate pool.
To filter out positive passages, we apply a two-
stage LLM-based classifier to the retrieved candi-
dates in descending order of similarity. The first
stage,context sufficiency, evaluates the input pair
(˜qn, pn)to determine whether the query qnisan-
swerablebased solely on pn, without relying on ex-
ternal knowledge. The second stage,clue presence,
assesses the pair (cn, pn)by checking whether the
clue text cnis explicitly mentioned or logically
supported within pn. For each stage, the classifier
produces a binary judgment (yes/no) alongside a
natural language reasoning trace.
Only candidates receiving a “no” judgment in
both stages are accepted as hard negatives p−, and
the procedure continues sequentially until enough
hard negatives are collected. The selected hard neg-
atives are then exhaustively inspected by human
annotators to confirm that no positive passages re-
main. This controlled manipulation enables sys-
tematic evaluation of honest abstention and hal-
lucination robustness under both answerable and
unanswerable retrieval contexts.
D Human Annotation
A total of four annotators participated in the anno-
tation process: two graduate students in NLP and
two financial domain experts. For each item, three
annotators were randomly assigned and conducteddata verification under a shared set of guidelines.
The annotators assessed both faithful dialogues
and hallucination-injected samples in a systematic
manner. Following the initial evaluation, additional
manual refinement was conducted as a quality con-
trol to ensure that each query and response accu-
rately reflected the intended hallucination type and
that the overall dialogue remained natural and co-
herent.
D.1 Faithful Dialogue Evaluation
The faithfulness of constructed dialogues is as-
sessed along five dimensions proposed by (Lee
et al., 2024):answerability,plausibility,correct-
ness,diversity, andcoherency. These criteria are
applied at the query qn, response an, and dialogue
levels to capture both turn-level validity and overall
dialogue consistency.
At the query level,answerabilityevaluates
whether a query can be reasonably answered given
the provided context, whileplausibilityassesses
whether the query is consistent with realistic user
behavior. At the response level,correctnessmea-
sures factual accuracy and passage grounding. At
the dialogue level,diversityensures that succes-
sive turns introduce non-redundant information,
andcoherencyverifies that each turn appropriately
reflects prior context and maintains logical con-
sistency. Overall, these metrics provide a compre-
hensive characterization of faithful multi-turn dia-
logues. To facilitate this multi-dimensional assess-
ment, we developed a dedicated annotation inter-
face that presents the dialogue history, retrieved
context, and candidate responses, as depicted in
Figure 7.
13

Target Criteria Description and Scoring Rubric
QueryAnswerability(0–1)Description: Is the answer to the query present in the given documents?
0: The document does not contain the information needed, or it cannot be derived.
1: The answer is explicitly stated or can be reasonably inferred from the document.
Plausibility(1–3)Description: How likely is the query created by an actual user?
1: Clearly unnatural, contrived, or unlikely to be asked by a real user.
2: Generally plausible, but slightly awkward or artificial in phrasing.
3: Natural and realistic; highly likely to be asked by a real user.
Response Correctness(0–1)Description: Is the response correct based on the provided documents?
0: Contains incorrect claims, contradictions, or unsupported information.
1: Fully consistent with the document and answers the question correctly.
DialogueDiversity(1–3)Description: Is each query-response in the dialog sufficiently different?
*Exclusion: One intentionally included verification turn is excluded from scoring.
1: Largely repetitive in wording, structure, or content.
2: Adds some new info, but sentence structure remains noticeably monotonous.
3: Each turn adds clearly new information with varied wording and structure.
Coherency(1–3)Description: Is the dialog flow natural and coherent?
1: Incoherent; irrelevant to prior context or contains logical contradictions.
2: Partially coherent; related to context but has awkward transitions.
3: Coherent and consistent; accurately reflects context with a natural flow.
Table 7: Evaluation rubric for dialogue quality, adapted from (Lee et al., 2024). Metrics are categorized by their
evaluation targets: query, response, and overall dialogue.
D.2 Hallucinated Dialogue Evaluation
Hallucinated dialogues are evaluated on two cri-
teria, each rated on a 1–3 scale:appropriateness,
which measures whether the intended hallucination
type is correctly exhibited, andnaturalness, which
assesses whether the response maintains fluency
and coherence within the dialogue context. Each re-
sponse carries a single hallucination type assigned
during injection. Annotators verify this label and
flag mismatches for revision or removal.
E Fine-tuning Setup
We fine-tune Qwen3-8B on the training split of
K-FinHallu (2,624 samples) for the four-class de-
tection task introduced in Section 3.2. The training
split is partitioned 9:1 into training and validation
subsets.
Optimization.We fine-tune using the ms-swift
framework (Zhao et al., 2024) with LoRA (Hu et al.,
2022) at rank r=8,α=32 , and a dropout of 0.05,
attached to all linear projections of the backbone.
We train for 3 epochs with a learning rate of 1e−4 ,
a cosine schedule, and a 3% warmup ratio. The ef-
fective batch size is 32 (per-device batch 2, gradient
accumulation 2, across 8 GPUs). Training runs on
8×NVIDIA RTX A6000 (48 GB) with bfloat16
mixed precision, completing in approximately 4
hours.Inference.All open-source baselines and the
fine-tuned models are served with vLLM (Kwon
et al., 2023) using tensor parallelism across 4
NVIDIA RTX A6000 (48 GB) GPUs. Closed-
source models are accessed via their official APIs.
All evaluations use a temperature of 0 where appli-
cable for reproducibility.
Rationale template.For theQwen3-8B SFT-R
variant, each training instance is paired with a tem-
plated rationale that follows our taxonomy’s two-
step reasoning: first assessing answerability based
on the presence of evidentiary clue text, then eval-
uating whether the response is faithful (when an-
swerable) or correctly abstains (when unanswer-
able). The rationale is placed inside Qwen3’s native
<think> block so that the model is supervised to
produce it during its thinking phase before emitting
the class label. Training uses Korean templates. We
provide their English translations below for read-
ability.
•Faithful Answer: “The reference document
contains the clue {clue}, so the query is an-
swerable. The current response answers it ap-
propriately, thereforeFaithful Answer.”
•Hallucination(answerable case): “The refer-
ence document contains the clue {clue}, so
the query is answerable. However, the current
response provides an inappropriate answer,
thereforeHallucination.”
14

Figure 7: Screenshot of the annotation interface for dataset quality control.
•Hallucination(unanswerable case): “The ref-
erence document does not contain evidence to
support the query, so the query is unanswer-
able. However, the current response provides
an answer, thereforeHallucination.”
•False Refusal: “The reference document con-
tains the clue {clue}, so the query is answer-
able. However, the current response refuses to
answer despite sufficient evidence, therefore
False Refusal.”
•True Refusal: “The reference document does
not contain evidence to support the query, so
the query is unanswerable. The current re-
sponse correctly refuses to answer, therefore
True Refusal.”
The {clue} slot is filled with the evidentiary span
˜cfrom the source annotation when available. The
Qwen3-8B SFT(no reasoning) variant is trained on
the same data with the rationale field removed.
F Additional Experimental Results
This appendix provides additional breakdowns and
analyses that supplement the main results: the fullfour-class breakdown, multi-class diagnostics with
confusion analysis, and cross-source robustness.
F.1 Four-class Full Breakdown
Table 8 reports the full four-class breakdown with-
out collapsingFalse RefusalandTrue Refusal. The
breakdown surfaces two failure patterns among
base models: collapse onto a single refusal type
(0.000 onFalse Refusal), and lowTrue Refusal
among closed-source models.
F.2 Multi-class Classification Results
Table 9 reports multi-class classification F1 over
the 404 hallucinated samples. Korean-centric and
Llama series models score near zero, the Qwen3
series shows moderate performance, while GPT-5
and the Gemini-2.5 series cluster around 0.70.
Table 10 shows confusion matrices for the two
strongest base detectors (GPT-5 and Gemini-2.5-
Pro) on the multi-class task. Among hallucination-
type confusions, the most frequent isContradic-
tory↔Unverifiable, with a smaller but notable
share ofFalse Acceptancemisclassified asUnveri-
fiable.
15

Model Faithful Hallucination False Refusal True Refusal Overall
Korean-centric
kanana-2-30b-a3b 0.597 0.494 0.000 0.615 0.515
EXAONE-4.0-32B 0.727 0.514 0.0000.9420.601
Open-source
Llama-3.1-8B 0.114 0.537 0.577 0.327 0.342
Llama-3.1-70B 0.781 0.670 0.0000.9420.693
Llama-3.3-70B 0.756 0.682 0.0000.9420.687
Qwen3-8B0.9860.446 0.500 0.481 0.688
Qwen3-14B 0.977 0.554 0.423 0.712 0.741
Qwen3-32B 0.969 0.551 0.654 0.596 0.743
Closed-source
GPT-4o 0.835 0.642 0.096 0.654 0.692
GPT-5 0.969 0.8100.9040.346 0.855
Gemini-2.5-Flash 0.946 0.795 0.808 0.538 0.845
Gemini-2.5-Pro 0.940 0.815 0.769 0.596 0.853
Fine-tuned (ours)
Qwen3-8B SFT 0.983 0.764 0.173 0.692 0.822
Qwen3-8B SFT-R 0.9720.8640.654 0.846 0.896
Table 8: Per-class accuracy on the four-class detection task (full view, without collapsing False Refusal and True
Refusal).Overallis the micro-average across all 808 test samples. Bold and underlined values denote the best and
second-best scores per column.
Model Contradictory Unverifiable Irrelevance False Acc. False Ref. Overall
Korean-centric
kanana-2-30b-a3b 0.000 0.000 0.000 0.109 0.000 0.022
EXAONE-4.0-32B 0.000 0.049 0.000 0.146 0.000 0.039
Open-source
Llama-3.1-8B 0.024 0.205 0.000 0.072 0.135 0.087
Llama-3.1-70B 0.049 0.080 0.000 0.192 0.346 0.133
Llama-3.3-70B 0.061 0.190 0.000 0.160 0.105 0.103
Qwen3-8B 0.456 0.323 0.311 0.361 0.484 0.387
Qwen3-14B 0.544 0.416 0.237 0.455 0.559 0.442
Qwen3-32B 0.629 0.478 0.286 0.476 0.589 0.492
Closed-source
GPT-4o 0.515 0.050 0.048 0.270 0.565 0.289
GPT-5 0.775 0.618 0.780 0.698 0.7140.717
Gemini-2.5-Flash 0.788 0.639 0.722 0.492 0.705 0.669
Gemini-2.5-Pro0.799 0.7040.740 0.687 0.696 0.725
Table 9: F1-scores of LLMs by hallucination type on the multi-class classification task (404 hallucination-mode
samples).Overallreflects the macro-average across five types. Bold and underlined values indicate the best and
second-best scores.
F.3 Cross-source Detection Accuracy
Table 11 reports binary detection accuracy (re-
call) by hallucination type and across two injection
sources, GPT-4o and Gemini-2.5-Flash. Detector
rankings are largely preserved across sources, and
per-type gaps remain modest ( ±10%p) for most
hallucination types, with the exception ofIrrele-
vance, which shows a pronounced difference.
G Hallucination Injection and Detection
Prompts
This appendix provides supplementary details for
the hallucination-injection pipeline described in
Section 4.3, along with the prompt templates used
for both hallucination injection and detection. Ta-
ble 12 lists the injection strategies for each hal-lucination type and subtype. Table 13 maps each
(Retrieval Context, Response) configuration to its
corresponding type label.
False RefusalandFalse Acceptanceare con-
structed by response and passage substitution, re-
spectively, and do not require LLM prompting. The
remaining three injection types—Contradictory,
Unverifiable, andIrrelevance—use prompted gen-
eration that elicits a rationale alongside the modi-
fied response. Table 14 gives the turn-position se-
lection prompt. Tables 15, 16, 17, and 18 present
the three injection prompts (with separateCon-
tradictoryprompts for context contradiction and
dialogue-history inconsistency). Tables 19, 20, and
21 present the binary, multi-class, and four-class
detection prompts.
16

GT / Pred No Hallu. Contradictory Unverifiable Irrelevance False Refusal False Acceptance
GPT-5
No Hallucination 360 7 3 3 29 2
Contradictory 24 110 20 4 0 0
Unverifiable 21 4 51 2 0 0
Irrelevance 18 2 2 55 0 0
False Refusal 7 0 0 0 45 0
False Acceptance 3 3 11 0 0 22
Gemini-2.5-Pro
No Hallucination 359 7 1 11 22 4
Contradictory 24 115 10 7 1 0
Unverifiable 16 3 56 2 0 1
Irrelevance 12 3 5 57 0 0
False Refusal 12 0 0 0 40 0
False Acceptance 5 2 9 0 0 23
Table 10: Confusion matrices for GPT-5 and Gemini-2.5-Pro on the multi-class task. Rows are ground truth, columns
are predictions.
Contradictory Unverifiable Irrelevance False Acc. False Ref. Avg.
Model GPT Gem GPT Gem GPT Gem GPT Gem GPT Gem GPT Gem
kanana-2-30b 0.92 0.92 0.97 0.95 0.95 0.95 1.00 0.89 0.33 0.39 0.83 0.82
EXAONE-4.0-32B 0.67 0.72 0.74 0.72 0.93 0.70 1.00 1.00 0.62 0.79 0.79 0.79
Llama-3.1-8B 0.94 0.97 0.97 0.92 0.97 0.89 1.00 0.95 1.00 1.00 0.98 0.95
Llama-3.1-70B 0.85 0.85 0.79 0.69 0.93 0.76 0.90 0.84 0.92 0.82 0.88 0.79
Llama-3.3-70B 0.87 0.87 0.79 0.72 0.93 0.78 0.95 0.89 1.00 0.93 0.91 0.84
Qwen3-8B 0.53 0.54 0.49 0.33 0.60 0.24 0.80 0.63 0.71 0.71 0.63 0.49
Qwen3-14B 0.70 0.68 0.59 0.31 0.55 0.38 0.90 0.63 0.54 0.46 0.66 0.49
Qwen3-32B 0.65 0.73 0.59 0.36 0.68 0.22 0.80 0.74 0.67 0.75 0.68 0.56
GPT-4o 0.71 0.72 0.59 0.64 1.00 0.76 1.00 1.00 0.92 0.89 0.84 0.80
GPT-5 0.84 0.85 0.77 0.62 0.82 0.41 0.90 0.74 0.92 0.86 0.85 0.70
Gem-2.5-Flash 0.90 0.87 0.87 0.69 0.93 0.68 0.85 0.68 0.88 0.89 0.89 0.76
Gem-2.5-Pro 0.86 0.84 0.82 0.59 0.93 0.68 0.95 0.74 0.71 0.71 0.85 0.71
Table 11: Binary detection accuracy (recall) by hallucination type and injection source (GPT = GPT-4o-injected,
Gem = Gemini-injected). Each cell shows the proportion of hallucinated samples correctly detected.
Type Subtype / Strategy Injection Logic
Contradictory Financial Term Misunder-
standingSwap a financial term with a similar but distinct one (e.g., “Credit Loan”
↔“Mortgage Loan”).
Modifier Change Flip or omit critical qualifiers such as “only,” “minimum,” or “within” to
distort legal boundaries.
Number Error Perturb figures (rates, dates, limits) using plausible values or conflicting
data from the context.
Inconsistency Intentionally generate a response that contradicts statements previously
made inH n.
IrrelevanceCondition Not Satisfied Ignore query constraints regarding target demographics, periods, or
eligibility requirements.
Off-topic Drift Discuss a different aspect of the same subject (e.g., answering “period”
when asked about “method”).
Question Misunderstanding Misinterpret grammatical particles or interrogatives (e.g., confusing
“from” with “until”).
UnverifiableFabricated Detail Insert non-existent procedures, exceptions, or clauses not supported by
p∗(under 5 words).
Subjective Opinion Interject unfounded evaluations, recommendations, or judgments into
otherwise neutral facts.
False Acceptance- Respond to ahard negativep−as if it contains valid evidence.
False Refusal- Replace a fully supported response with a fixed refusal string despite the
presence of evidence.
Table 12: Injection strategies for each hallucination type and subtype in K-FinHallu.
17

Retrieval Context (P n) Response (a n) Type
Contains Positive Passages Original Faithful Answer
Contains Positive Passages Perturbed/Injected Contradictory / Unverifiable / Irrelevance
Consists Entirely of Hard Negatives Original False Acceptance
Contains Positive Passages Refusal String False Refusal
Consists Entirely of Hard Negatives Refusal String True Refusal
Table 13: The combinatorial mapping of Retrieval Context ( Pn) and Response ( an) configurations in K-FinHallu.
PositiveandNegativerefer to the answerable and unanswerableP nconfigurations defined in Section 3.1.
Type Prompt
System [EN]You are an expert in determining where to inject hallucinations into a dialogue. Determine the appropriate
positions so that each hallucination type is naturally integrated.
[KO]당신은대화에환각을주입할위치를결정하는전문가입니다 .각환각유형이적절한위치에주입되도
록판단하세요.
User [EN]The following is the full dialogue:{dialogue_text}.
Hallucination types to inject (in order):{hallucination_types}.
{hallucination_definitions}.
Decide which turn to inject each hallucination type in order. Each hallucination type must be injected in a
different turn. The order of turn numbers matters (first turn number maps to type 1, second to type 2, etc.).
{constraint_note}.
Respond in JSON format:{"turn_indices": [1, 2, ...]}.
turn_indices: List of turn numbers to inject each hallucination type (in order, starting from 1).
[KO]다음은전체대화입니다:{dialogue_text}.
주입할환각유형(순서대로):{hallucination_types}.
{hallucination_definitions}.
각환각유형을순서대로어떤턴에주입할지결정하세요 .각환각유형은서로다른턴에주입되어야합니다 .
반환하는턴번호의순서가중요합니다 (첫번째턴번호는 1번유형에 ,두번째턴번호는 2번유형에매핑됩
니다).{constraint_note}.
JSON형식으로답변하세요:{"turn_indices": [1, 2, ...]}.
turn_indices:각환각유형을주입할턴번호리스트(순서대로, 1부터시작).
Table 14: Positioning prompts for determining hallucination injection locations.
18

Type Prompt
System [EN]You are an expert at subtly modifying answers to create conflicts with context. Generate natural hallucinations
with minimal modifications.
[KO]답변을미묘하게수정하여컨텍스트와충돌을만드는전문가입니다 .최소수정으로자연스러운환각을
생성하세요.
User [EN]The following question and answer are given.
Question:{user_question}.
Answer:{original_answer}.
Related Document:{passage_text}.
Modify the answer minimally to create a subtle contradiction with the related document/context. Change only
1-2 key financial terms/words/modifiers/numbers to distort the meaning. Do not add unverifiable information
(sentences or phrases); maintain the original answer structure and length. Make the modification so subtle that
it’s difficult to notice, ensuring the answer appears natural. Do not change the subject of the answer as much as
possible.
You may refer to the following methods:
- Financial term misunderstanding: Replace difficult financial terms with similar but different terms to distort
meaning (e.g., “unsecured loan” ↔“secured loan”, “equal installment” →“equal principal”, “interest rate” →
“yield”).
- Modifier change or omission: Change/omit modifiers like only/just/minimum/maximum/at least/at
most/over/under/partially.
- Number error: Replace key numbers (period/rate/limit/frequency) with similar range numbers, or replace with
other related numbers from context.
Respond in the following format (first explain the rationale, then write the final answer):
Rationale: [Explain which part was modified and why].
Answer: [Modified answer].
[KO]다음질문과답변이주어집니다.
질문:{user_question}.
답변:{original_answer}.
관련문서:{passage_text}.
답변을최소한으로만수정하여 ,관련문서 /컨텍스트와미묘하게충돌 (모순 )되도록환각을만드세요 .전체답
변을다시작성하지말고 ,핵심금융전문용어 /단어 /수식어 /숫자중 1 2개만바꿔서의미가틀어지게하세요 .
검증불가능한정보 (문장이나구 )를추가하지말고 ,기존답변구조와길이를유지하세요 .알아차리기어려울
정도로미묘하게수정하여,답변이자연스럽게보이도록하세요.답변의주어는최대한바꾸지마세요.
다음과같은방법을참고할수있습니다:
-금융전문용어오해 :답변에있는어려운금융전문용어를유사하지만다른금융전문용어로바꿔의미를
왜곡(예: “신용대출”↔“담보대출”, “원리금상환”→“원금균등상환”, “이자율”→“금리”).
-한정어 /수식어변경하거나누락 :오직 /만/최소 /최대 /이상 /이하 /초과 /미만 /부분적으로등한정어 /수식어를변
경/누락.
-숫자오류 :기간 /금리 /한도 /횟수등핵심숫자를비슷한범위의다른숫자로교체하거나 ,컨텍스트내다른
관련숫자로교체.
다음형식으로답변하세요(먼저근거를설명하고,그다음최종답변을작성하세요):
근거: [어떤부분을왜수정했는지설명].
답변: [수정된답변].
Table 15: Contradictory prompts: Basic context contradiction type.
19

Type Prompt
System [EN]You are an expert at subtly modifying answers to create conflicts with context. Generate natural hallucinations
with minimal modifications.
[KO]답변을미묘하게수정하여컨텍스트와충돌을만드는전문가입니다 .최소수정으로자연스러운환각을
생성하세요.
User [EN]The following dialogue history and current answer are given.
Dialogue History:{history_text}.
Current Question:{user_question}.
Current Answer:{original_answer}.
Related Document:{passage_text}.
Modify the current answer minimally so that it contradicts the prior dialogue in a subtle and hard-to-notice way.
Change only 1–2 key financial terms/words/modifiers/numbers to distort the meaning. Do not add unverifiable
information (sentences or phrases); maintain the original answer structure and length. Make the modification so
subtle that it appears natural. Do not change the subject of the answer, and do not produce a response unrelated to
the question.
You may refer to the following methods:
- Financial term misunderstanding: Replace difficult financial terms with similar but different terms to distort
meaning (e.g., “unsecured loan” ↔“secured loan”, “equal installment” →“equal principal”, “interest rate” →
“yield”).
- Modifier change or omission: Change/omit modifiers like only/just/minimum/maximum/at least/at
most/over/under/partially.
- Number error: Replace key numbers (period/rate/limit/frequency) with similar range numbers, or replace with
other related numbers from context.
Respond in the following format (first explain the rationale, then write the final answer):
Rationale: [Explain which part was modified and why].
Answer: [Modified answer].
[KO]다음대화히스토리와현재답변이주어집니다.
대화히스토리:{history_text}.
현재질문:{user_question}.
현재답변:{original_answer}.
관련문서:{passage_text}.
현재답변을이전대화와모순되도록미묘하고알아차리기쉽지않은환각을만들도록최소한으로수정하세
요.전체답변을다시작성하지말고 ,핵심금융전문용어 /단어 /수식어 /숫자중 1 2개만바꿔서의미가틀어지게
하세요 .검증불가능한정보 (문장이나구 )를추가하지말고 ,기존답변구조와길이를유지하세요 .알아차리기
어려울정도로미묘하게수정하여 ,답변이자연스럽게보이도록하세요 .답변의주어는최대한바꾸지마세요 .
미묘하게수정한다고해서질문과관련없는대답을하면안됩니다.
다음과같은방법을참고할수있습니다:
-금융전문용어오해 :답변에있는어려운금융전문용어를유사하지만다른금융전문용어로바꿔의미를
왜곡(예: “신용대출”↔“담보대출”, “원리금상환”→“원금균등상환”, “이자율”→“금리”).
-한정어 /수식어변경하거나누락 :오직 /만/최소 /최대 /이상 /이하 /초과 /미만 /부분적으로등한정어 /수식어를변
경/누락.
-숫자오류 :기간 /금리 /한도 /횟수등핵심숫자를비슷한범위의다른숫자로교체하거나 ,컨텍스트내다른
관련숫자로교체.
다음형식으로답변하세요(먼저근거를설명하고,그다음최종답변을작성하세요):
근거: [어떤부분을왜수정했는지설명].
답변: [수정된답변].
Table 16: Contradictory prompts: Inconsistency with prior dialogue history.
20

Type Prompt
System [EN]You are an expert at naturally adding unverifiable information. Insert only short phrases of 3-5 words.
[KO]근거없는정보를자연스럽게추가하는전문가입니다. 3-5어절의짧은phrase만삽입하세요.
User [EN]The following question and answer are given.
Question:{user_question}.
Original Answer:{original_answer}.
Related Document:{passage_text}.
The original answer is correct. Add information that cannot be verified by the document/context as a subtle,
hard-to-detect hallucination. Do not add a whole sentence; only add short content of less than 5 words. Maintain
the structure and content of the original answer as much as possible. Write the added information so subtly that it
appears real and is difficult to notice.
You may refer to the following methods:
- Unverifiable detailed information: Insert short details like procedures/conditions/exceptions/periods that are not
in the document.
- Unverifiable subjective opinion: Insert evaluations/recommendations that are not in the document to appear
factual.
Respond in the following format (first explain the rationale, then write the final answer):
Rationale: [Explain which part was modified and why].
Answer: [Modified answer].
[KO]다음질문과답변이주어집니다.
질문:{user_question}.
원본답변:{original_answer}.
관련문서:{passage_text}.
원본답변은올바릅니다 .이답변에문서 /컨텍스트로검증불가한정보를미묘하고알아차리기쉽지않게추
가하세요 .한문장전체를추가하지말고 , 5어절미만의짧은내용을추가하세요 .원본답변의구조와내용은
최대한유지하세요 .추가된정보가실제정보처럼보이도록알아차리기어려울정도로미묘하게작성하세요 .
다음과같은방법을참고할수있습니다:
-근거없는상세정보:문서에 없는절차/조건/예외/기간등짧은디테일을끼워넣기.
-근거없는주관적의견:문서에 없는평가/추천을사실처럼보이게끼워넣기.
다음형식으로답변하세요(먼저근거를설명하고,그다음최종답변을작성하세요):
근거: [어떤부분을왜수정했는지설명].
답변: [수정된답변].
Table 17: Unverifiable prompts: Basic unverifiable information addition type.
21

Type Prompt
System [EN]You are an expert at creating answers that deviate from the question intent. Maintain the core topic but
answer something unrelated to the question.
[KO]질문의도와어긋나는답변을만드는전문가입니다 .핵심토픽은유지하되질문과관련없는답을하세
요.
User [EN]The following conversation history, current question, and answer are given.
Conversation History:{history_text}.
Current Question:{user_question}.
Current Answer:{original_answer}.
Related Document:{passage_text}.
Modify the answer minimally to deviate from the question intent while maintaining key keywords within the same
topic.
You may refer to the following methods:
- Condition mismatch: Modify to not satisfy the question’s conditions (target/period/scope/qualification) (e.g.,
change “3 months” to “1 year” when asked about “3 months”).
- Topic drift: Answer about a different aspect within the same topic (e.g., answer about “application period” when
asked about “application method”).
- Question misunderstanding: Misinterpret grammatical elements (particles/interrogatives) and answer differently
(e.g., misunderstand “from” as “until” and answer about “until when”).
Respond in the following format (first explain the rationale, then write the final answer):
Rationale: [Explain which method was used, which part was modified, and why].
Answer: [Modified answer].
[KO]다음대화히스토리와현재질문,답변이주어집니다.
대화히스토리:{history_text}.
현재질문:{user_question}.
현재답변:{original_answer}.
관련문서:{passage_text}.
같은주제내에서질문의핵심키워드를유지하면서질문의도와어긋나도록답변을최소한으로만수정하세
요.
다음과같은방법을참고할수있습니다:
-조건불만족 :질문의조건 (대상 /기간 /범위 /자격 )을만족하지않도록수정 (예: “3개월 ”을묻는데 “1년”으로
답하도록변경).
-주제이탈 :같은주제내에서질문과다른측면에대해답변 (예: “신청방법 ”을묻는데 “신청기간 ”에대해
답변).
-질문오해 :문법적요소 (조사 /의문사 )를오해하여다르게답변 (예: “부터 ”를“까지 ”로오해하여 “언제까지 ”
에대해답변).
다음형식으로답변하세요(먼저근거를설명하고,그다음최종답변을작성하세요):
근거: [어떤방법을사용하여 어떤부분을왜수정했는지설명].
답변: [수정된답변].
Table 18: Irrelevance prompts: Basic question intent deviation type.
22

Prompt
[EN]
Task:
Determine whether the current response in the following dialogue is a hallucination.
Important:Judge based only on the given dialogue history and related documents. Do not judge using the LLM’s internal
knowledge.
Classification Criteria:
It is a hallucination if any of the following applies:
- When the answer directly contradicts the given documents or previous dialogue history
- When the answer includes information that cannot be verified from the given documents
- When the response is not logically connected to the user’s intent or question
- When the answer refuses to respond despite sufficient evidence in the given documents
- When an answer is provided despite the situation being unanswerable from the given documents
It is NOT a hallucination only in the following cases:
- When the answer correctly responds to the question based on information from the given documents
- When the answer refuses to respond, acknowledging insufficient evidence in the given documents
Output Format:
Output only 0 or 1, without any other explanation.
0: Not Hallucination
1: Hallucination
Dialogue:{dialogue}
Question:{question}
Related Documents:{contexts}
Answer:{answer}
[KO]
태스크:
다음에제시되는대화에서현재응답이할루시네이션인지판별하세요.
중요:주어진대화히스토리와관련문서만을기준으로판단하세요. LLM의내부지식으로판단하지마세요.
분류기준:
다음중하나라도해당하면할루시네이션입니다:
-답변이주어진문서나이전대화히스토리와직접적으로모순되는경우
-답변에주어진문서에검증불가능한정보를포함하는경우
-사용자의의도나질문과논리적으로연결되지않는응답을하는경우
-주어진문서에충분한증거가있음에도불구하고답변을거부하는경우
-주어진문서로답변할수없는상황임에도불구하고답변을제공하는경우
다음의경우만할루시네이션이아닙니다:
-주어진문서의정보를바탕으로질문에올바르게답변한경우
-주어진문서에증거가부족함을인정하고답변을거부하는경우
출력형식:
다른설명없이오직0또는1만출력하세요.
0:할루시네이션아님
1:할루시네이션
대화:{dialogue}
질문:{question}
관련문서:{contexts}
답변:{answer}
Table 19: Binary hallucination detection prompt template.
23

Prompt
[EN]
Task:
Determine whether the current response in the following dialogue is a hallucination, and if so, classify its type.
Important:Judge based only on the given dialogue history and related documents. Do not judge using the LLM’s internal
knowledge.
Classification Criteria:
When the given documents contain sufficient evidence for the question:
- No Hallucination: When the answer correctly responds to the question based on information from the given documents
- Contradictory: When the answer directly contradicts the given documents or previous dialogue history
- Unverifiable: When the answer includes information that cannot be verified from the given documents
- Irrelevance: When the response is not logically connected to the user’s intent or question
- False Refusal: When the answer refuses to respond despite sufficient evidence in the given documents
When the given documents lack sufficient evidence for the question:
- False Acceptance: When an answer is provided despite the situation being unanswerable from the given documents
- No Hallucination: When the answer refuses to respond, acknowledging insufficient evidence in the given documents
Output Format:
Output only a single number between 0 and 5 representing the hallucination type, without any other explanation.
0: No Hallucination
1: Contradictory
2: Unverifiable
3: Irrelevance
4: False Refusal
5: False Acceptance
Dialogue:{dialogue}
Question:{question}
Related Documents:{contexts}
Answer:{answer}
[KO]
태스크:
다음에제시되는대화에서현재응답이할루시네이션인지판별하고,할루시네이션인경우유형을분류하세요.
중요:주어진대화히스토리와관련문서만을기준으로판단하세요. LLM의내부지식으로판단하지마세요.
분류기준:
주어진문서에질문에대한충분한증거가있는경우:
- No Hallucination:주어진문서의정보를바탕으로질문에올바르게답변한경우
- Contradictory:답변이주어진문서나이전대화히스토리와직접적으로모순되는경우
- Unverifiable:답변에주어진문서에검증불가능한정보를포함하는경우
- Irrelevance:사용자의의도나질문과논리적으로연결되지않는응답을하는경우
- False Refusal:주어진문서에충분한증거가있음에도불구하고답변을거부하는경우
주어진문서에질문에대한충분한증거가없는경우:
- False Acceptance:주어진문서로답변할수없는상황임에도불구하고답변을제공하는경우
- No Hallucination:주어진문서에증거가부족함을인정하고답변을거부하는경우
출력형식:
다른설명없이오직할루시네이션유형을나타내는0에서5사이의숫자하나만출력하면됩니다.
0: No Hallucination
1: Contradictory
2: Unverifiable
3: Irrelevance
4: False Refusal
5: False Acceptance
대화:{dialogue}
질문:{question}
관련문서:{contexts}
답변:{answer}
Table 20: Multi-class hallucination detection and classification prompt template.
24

Prompt
[EN]
Task:
Classify the current response in the following conversation into one of four types.
Important:Judge only based on the provided dialogue history and reference documents. Do not use the LLM’s internal
knowledge.
Classification Criteria:
When the reference documents contain sufficient evidence for the question:
- Faithful Answer: The response correctly answers the question based on the information in the reference documents.
- Hallucination: Any of the following applies:
* The response directly contradicts the reference documents or prior dialogue history.
* The response contains information that cannot be verified from the reference documents.
* The response is logically irrelevant to the user’s intent or question.
- False Refusal: The response refuses to answer despite sufficient evidence being available in the reference documents.
When the reference documents do NOT contain sufficient evidence for the question:
- Hallucination: The response provides an answer even though the reference documents cannot support it.
- True Refusal: The response appropriately refuses to answer, acknowledging the lack of evidence in the reference documents.
Output Format:
Output only a single number between 0 and 3 with no other explanation.
0: Faithful Answer
1: Hallucination
2: False Refusal
3: True Refusal
Dialogue:{dialogue}
Question:{question}
Reference Documents:{contexts}
Answer:{answer}
[KO]
태스크:
다음에제시되는대화에서현재응답을4가지유형으로분류하세요.
중요:주어진대화히스토리와관련문서만을기준으로판단하세요. LLM의내부지식으로판단하지마세요.
분류기준:
주어진문서에질문에대한충분한증거가있는경우:
- Faithful Answer:주어진문서의정보를바탕으로질문에올바르게답변한경우
- Hallucination:다음중하나라도해당하는경우
*답변이주어진문서나이전대화히스토리와직접적으로모순되는경우
*답변에주어진문서에검증불가능한정보를포함하는경우
*사용자의의도나질문과논리적으로연결되지않는응답을하는경우
- False Refusal:주어진문서에충분한증거가있음에도불구하고답변을거부하는경우
주어진문서에질문에대한충분한증거가없는경우:
- Hallucination:주어진문서로답변할수없는상황임에도불구하고답변을제공하는경우
- True Refusal:주어진문서에증거가부족함을인정하고적절하게답변을거부하는경우
출력형식:
다른설명없이오직0에서3사이의숫자하나만출력하세요.
0: Faithful Answer
1: Hallucination
2: False Refusal
3: True Refusal
대화:{dialogue}
질문:{question}
관련문서:{contexts}
답변:{answer}
Table 21: Four-class hallucination detection prompt template.
25