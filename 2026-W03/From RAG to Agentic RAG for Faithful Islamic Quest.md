# From RAG to Agentic RAG for Faithful Islamic Question Answering

**Authors**: Gagan Bhatia, Hamdy Mubarak, Mustafa Jarrar, George Mikros, Fadi Zaraket, Mahmoud Alhirthani, Mutaz Al-Khatib, Logan Cochrane, Kareem Darwish, Rashid Yahiaoui, Firoj Alam

**Published**: 2026-01-12 13:28:28

**PDF URL**: [https://arxiv.org/pdf/2601.07528v1](https://arxiv.org/pdf/2601.07528v1)

## Abstract
LLMs are increasingly used for Islamic question answering, where ungrounded responses may carry serious religious consequences. Yet standard MCQ/MRC-style evaluations do not capture key real-world failure modes, notably free-form hallucinations and whether models appropriately abstain when evidence is lacking. To shed a light on this aspect we introduce ISLAMICFAITHQA, a 3,810-item bilingual (Arabic/English) generative benchmark with atomic single-gold answers, which enables direct measurement of hallucination and abstention. We additionally developed an end-to-end grounded Islamic modelling suite consisting of (i) 25K Arabic text-grounded SFT reasoning pairs, (ii) 5K bilingual preference samples for reward-guided alignment, and (iii) a verse-level Qur'an retrieval corpus of $\sim$6k atomic verses (ayat). Building on these resources, we develop an agentic Quran-grounding framework (agentic RAG) that uses structured tool calls for iterative evidence seeking and answer revision. Experiments across Arabic-centric and multilingual LLMs show that retrieval improves correctness and that agentic RAG yields the largest gains beyond standard RAG, achieving state-of-the-art performance and stronger Arabic-English robustness even with a small model (i.e., Qwen3 4B). We will make the experimental resources and datasets publicly available for the community.

## Full Text


<!-- PDF content starts -->

From RAG to Agentic RAG for Faithful Islamic Question Answering
Gagan Bhatia1, Hamdy Mubarak1, Mustafa Jarrar2, George Mikros2,
Fadi Zaraket3, Mahmoud Alhirthani2, Mutaz Al-Khatib4,
Logan Cochrane5, Kareem Darwish1, Rashid Yahiaoui2, Firoj Alam1
1Qatar Computing Research Institute, HBKU, Qatar,
2College of Humanities and Social Sciences, HBKU, Qatar
3Arab Center for Research and Policy Studies, Qatar,4College of Islamic Studies, HBKU, Qatar
5College of Public Policy, HBKU, Qatar
fialam@hbku.edu.qa
Abstract
LLMs are increasingly used for Islamic ques-
tion answering, where ungrounded responses
may carry serious religious consequences. Yet
standard MCQ/MRC-style evaluations1do not
capture key real-world failure modes, notably
free-form hallucinations and whether models
appropriately abstain when evidence is lack-
ing. To shed a light on this aspect we intro-
duce ISLAMIC FAITH QA, a 3,810-item bilin-
gual (Arabic/English) generative benchmark
with atomic single-gold answers, which enables
direct measurement of hallucination and absten-
tion. We additionally developed an end-to-end
grounded Islamic modeling suite consisting of
(i)25K Arabic text-grounded SFT reasoning
pairs, (ii)5K bilingual preference samples for
reward-guided alignment, and (iii)a verse-level
Qur’an retrieval corpus of ∼6k atomic verses
(ayat). Building on these resources, we de-
velop an agentic Quran-grounding framework
(agentic RAG) that uses structured tool calls for
iterative evidence seeking and answer revision.
Experiments across Arabic-centric and multi-
lingual LLMs show that retrieval improves cor-
rectness and that agentic RAG yields the largest
gains beyond standard RAG, achieving state-
of-the-art performance and stronger Arabic–
English robustness even with a small model
(i.e., Qwen3 4B). We will make the experimen-
tal resources and datasets publicly available for
the community.
1 Introduction
Large language models (LLMs) are increasingly po-
sitioned as general-purpose assistants for decision
support, education, and guidance in value-laden
domains. Yet a persistent obstacle is that fluent
generations can mask normative andfactual unre-
liability: models remain sensitive to framing, role
instructions, and they may produce confident but
unsupported responses (Jiao et al., 2025).
1MCQ: Multiple choice questions, MRC: Machine Read-
ing Comprehension
Figure 1: Current–Proposed–Outcome. (a)Current
Islamic QA. (b)We combine ISLAMIC FAITH QA, LLM
judging, Quran retrieval, and agentic evidence seeking.
(c)This yields more faithful, citation-backed responses.
Islamic question answering is a particularly
challenging testbed for this reliability problem.
Deployed Islamic QA systems2indicate strong
demand, yet their proprietary evaluations high-
light the need for shared benchmarks emphasiz-
ing grounding, citation fidelity, and abstention.
User queries are not merely informational; they
are embedded in jurisprudential reasoning ( fiqh),
school-of-thought conventions, and culturally sit-
uated norms that demand faithful grounding in
canonical sources and careful handling of uncer-
tainty. Recent multilingual and culture-aware evalu-
ations show that moral judgments and alignment be-
haviour vary meaningfully with language and data
provenance, with persistent representational bias
and Western-dominance effects that are especially
salient for non-Western normative systems (Naous
and Xu, 2025; Guo et al., 2025). Within the Islamic
domain, emerging resources (e.g., inheritance-law
reasoning and abstention-aware fiqh evaluations)
indicate both progress and substantial performance
2e.g., https://ansari.chat/ ,https://usul.ai
,https://wisqu.ai
1arXiv:2601.07528v1  [cs.CL]  12 Jan 2026

gaps, particularly for Arabic and for school-aware
nuance, reinforcing the need for fine-grained re-
liability checks tailored to Islamic jurisprudence
(Bouchekif et al., 2025b; Elsafoury and Hartmann,
2025; Asseri et al., 2025). Parallel work on Quranic
retrieval-augmented generation (RAG) further sug-
gests that grounding can improve faithfulness, but
that outcomes are mixed and depend on model ca-
pacity and retrieval quality (Khalila et al., 2025).
A central obstacle in knowledge-intensive Is-
lamic QA is hallucination . Multilingual studies
suggest that Arabic settings can amplify factuality
and faithfulness errors, and that coarse answer-level
metrics often miss subtle inconsistencies impor-
tant for normative argumentation (ul Islam et al.,
2025; Alansari and Luqman, 2025; Hosseini et al.,
2025; Elchafei and Abu-Elkheir, 2025; Wang et al.,
2025). Moreover, test-time scaling results show
that longer reasoning traces do not reliably im-
prove grounding and may even increase overconfi-
dent errors (Gema et al., 2025; Zhao et al., 2025).
This motivates retrieval-based grounding, espe-
cially agentic setups that interleave search, tool use,
and verification, but practical reliability depends
on robust tool orchestration and domain ontolo-
gies (Liang et al., 2025; Li et al., 2025). Accord-
ingly, we target three under-specified and under-
measured needs in Islamic QA: (i)Arabic–English
robustness, (ii)calibrated abstention under insuffi-
cient evidence, and (iii)evidence-grounded genera-
tion aligned with canonical sources (Bhatia et al.,
2024). Figure 1 summarizes our motivation and
method in a current–proposed–outcome view, con-
trasting today’s Islamic QA pipeline with our Is-
lamic grounding-based approach and its resulting
citation-backed bilingual answers. Our contribu-
tions are as follows:
•Bilingual Islamic QA benchmark: ISLAM -
ICFAITH QA comprises 3,810 Arabic–English
questions with atomic, single-gold answers3and
a strict Correct /Incorrect /Not_Attempted la-
beling scheme, enabling direct measurement of
hallucination andabstention .
•An end-to-end data suite for grounded Islamic
modeling: We release a unified set of resources
spanning 25K Arabic text-grounded SFT rea-
soning pairs, 5Kbilingual preference samples
3Many Islamic questions allow multiple valid answers
across interpretive traditions (madh ¯ahib). To enable reliable
generative evaluation, we focus on atomic items with a sin-
gle text-grounded answer; handling disputed cases via multi-
reference/equivalence-class grading is left to future work.for reward-guided alignment, and a verse-level
Quran retrieval corpus of 6,236 atomic ayat.
•Evidence-seeking inference via agentic Quran
grounding: We develop and evaluate an agen-
tic RAG setup that turns retrieval into an explicit
decision process through structured tool calls (se-
mantic search, verse reading, metadata lookup).
Across all backbones, ISLAMIC FAITH QA ex-
poses a substantial reliability gap between general
instruction-following fluency and text-grounded
Islamic correctness: most off-the-shelf multilin-
gual LLMs remain below 30% accuracy under
strict LLM-as-Judge grading (Table 3). Retrieval
augmentation is the most consistently effective in-
tervention, improving performance across mod-
els by anchoring generations to canonical evi-
dence (Table 4). Most notably, agentic RAG
yields the largest gains beyond standard RAG, en-
abling strong bilingual robustness by forcing it-
erative evidence seeking and verse inspection be-
fore answering: for Qwen3-4B-2507, accuracy im-
proves from 21.85(base) to 38.85(+RAG) and
to48.90(+Agentic RAG), while also narrowing
the Arabic–English gap (Table 4). Finally, com-
bining a strong in-domain backbone with agentic
grounding achieves the best overall performance,
with Fanar-2-27B + Agentic RAG reaching 57.30
average accuracy (Table 4).
2 Related Work
2.1 Benchmarking in Islamic Domain
General-purpose moral and trustworthiness evalu-
ations establish that LLM behavior is highly sen-
sitive to framing and can appear competent while
remaining unreliable, motivating domain-grounded
assessment beyond generic dilemmas (Jiao et al.,
2025; Abhishek et al., 2025). Follow-on work in
specialized, high-stakes settings (e.g., legal/medi-
cal ethics) emphasizes stricter correctness notions,
risk-aware protocols, and evaluation designs that
better reflect real deployment constraints (Shao
et al., 2025; Hong et al., 2025; Wei et al., 2025;
Jin et al., 2025; Hui et al., 2025). In culturally
situated contexts, multilingual studies show that
moral judgments and alignment behavior vary sub-
stantially with language and data provenance, with
recurring Western-dominance effects and represen-
tational bias (Naous and Xu, 2025; Guo et al., 2025;
Agarwal et al., 2024). Within Islamic QA specif-
ically, recent benchmarks and datasets begin to
target fiqh-style reasoning, abstention, and cultur-
2

ally faithful evaluation, but consistently report gaps
in Arabic performance and jurisprudential nuance
(Atif et al., 2025; Bouchekif et al., 2025a; Lah-
mar et al., 2025; Mubarak et al., 2025; Elsafoury
and Hartmann, 2025; Aljaji et al., 2025; Alwajih
et al., 2025). These limitations motivate our fo-
cus on open-ended generative Islamic QA with
atomic single-gold answers and strict LLM-as-a-
judge grading to directly measure hallucination and
abstention, rather than relying on MCQ/MRC-style
proxies (Haas et al., 2025).
2.2 Factuality in Knowledge-Intensive
Domains
Hallucination remains a central failure mode
in knowledge-intensive QA, and recent multilin-
gual/Arabic work documents elevated factuality
and faithfulness errors alongside calls for evalua-
tion beyond answer-only metrics, including span-
level attribution and joint assessment of reason-
ing traces and final outputs (ul Islam et al., 2025;
Alansari and Luqman, 2025; Hosseini et al., 2025;
Elchafei and Abu-Elkheir, 2025; Wang et al., 2025).
At the same time, evidence on test-time scaling in-
dicates that longer reasoning traces do not reliably
improve grounding and can increase overconfident
errors, reinforcing that “thinking more” is not a
substitute for evidence (Gema et al., 2025; Zhao
et al., 2025). Retrieval augmentation is therefore
pivotal, and recent surveys on reasoning/agentic
RAG highlight how iterative search, tool use, and
verification can improve groundedness when re-
trieval and orchestration are reliable (Liang et al.,
2025; Li et al., 2025). In Qur’anic/Islamic set-
tings, empirical work shows that RAG can improve
faithfulness but outcomes depend strongly on re-
trieval quality, model capacity, and domain cover-
age (Khalila et al., 2025; Raghad Salameh, 2024).
Broader trustworthiness suites emphasize that fac-
tuality should be assessed alongside safety and
misinformation risk in value-laden deployments
(Huang et al., 2023; Abhishek et al., 2025; Hui
et al., 2025), while Arabic-centric resources fur-
ther highlight how language coverage and repre-
sentation affect retrieval and downstream reliability
(Bhatia et al., 2024, 2025). These findings motivate
our comparison of standard RAG versus agentic
RAG under a strict generative, abstention-aware
protocol designed to surface and reduce hallucina-
tions in Islamic QA (Haas et al., 2025).3 Datasets
To facilitate the development of robust Islamic
LLMs and enable precise hallucination evalua-
tion, we construct a comprehensive suite of re-
sources comprising instruction tuning data, prefer-
ence alignment data, a retrieval corpus, and a novel
evaluation benchmark, ISLAMIC FAITH QA. The
specific statistics for each set of our data suite are
summarized in Table 1.
Dataset Role Size Language
SFT Reasoning Training 25,000 Arabic
RL Preference Training 5,000 Ar + En
Quran RAG Retrieval 6,236 Arabic
ISLAMIC FAITH QA Evaluation 3,810 Ar + En
Table 1: Summary of the constructed data resources.
Sizes represent the number of instruction pairs, reward
samples, or atomic retrieval units (verses).
3.1 Training and Alignment Resources
We develop two training datasets and a Quranic
RAG Index to enhance model capability in the
Islamic domain, specifically targeting theological
reasoning and safety alignment.
SFT Reasoning dataset. For Supervised Fine-
Tuning (SFT), we curate a dataset of 25,000 Arabic
instruction-response pairs centered on theological
reasoning. Unlike standard QA pairs, this dataset is
text-grounded; questions are derived directly from
Quranic verses and Hadith, with answers requiring
grounded reasoning steps rather than simple extrac-
tion. As shown in Figure 4 we use LLM generated
datasets. This structure facilitates the model’s abil-
ity to articulate the logical basis behind Islamic
rulings. An example of the SFT Reasoning dataset
is given in Appendix E.1.
RL Preference dataset. To support preference op-
timization techniques such as GRPO (Shao et al.,
2024), we construct a Reinforcement Learning
(RL) dataset of 5,000 bilingual samples (Arabic
and English). Each instance includes a question de-
rived from canonical texts, a gold-standard answer,
and specific evaluation parameters designed to train
reward models. This dataset is crucial for aligning
model outputs with factual correctness and mini-
mizing hallucination in sensitive religious contexts.
An example to understand the dataset is given in
Appendix E.2.
Quran RAG dataset. Additionally, for Retrieval-
Augmented Generation (RAG) experiments, we
process the standard corpus of the Holy Quran into
3

6,236 retrieval units corresponding to individual
Ayat (verses), serving as the ground-truth knowl-
edge base for both generation and evaluation tasks.
Concretely, we segment the full Qur’an into 6,236
units (one ayah per record) and attach standardised
metadata required for tool use and evaluation, in-
cluding surah andayah indices, canonical verse
identifiers, and normalised Arabic text (to reduce
orthographic variance and improve dense retrieval).
This structure enables (i) consistent verse-level ci-
tation in model outputs, (ii) deterministic mapping
from retrieved evidence to a unique canonical ref-
erence, and (iii) faithful evaluation of grounding by
checking whether predicted claims are supported
by retrieved ayat.
3.2 The I SLAMIC FAITH QA Benchmark
Existing evaluations for Islamic NLP often rely
on discriminative formats like Multiple Choice
Questions (MCQ) (Alwajih et al., 2025; Bouchekif
et al., 2025a) or Machine Reading Comprehension
(MRC) (Bashir et al., 2021; Premasiri et al., 2022).
As detailed in Table 2, these formats allow models
to guess correctly without genuine grounding and
fail to measure abstention capabilities. To address
this, we introduce ISLAMIC FAITH QA, a bilingual
generative benchmark with 3,810 Arabic questions
and English questions, designed to measure hallu-
cination rates via an LLM-as-a-Judge protocol.
Resource Type Size EN+AR Text- Format GenQA
grounded
ISLAMIC FAITH QA (Ours) Benchmark 3,810 ✓ ✓ GenQA ✓
QRCD (Bashir et al., 2021) Dataset 1,337 ✗ ✓ MRC ✗
AyaTEC (Malhas and Elsayed, 2020) Dataset 207 ✗ ✓ VerseQA ✗
Hajj-FQA (Aleid and Azmi, 2025) Dataset 2,826 ✗ ✗ FatwaQA ✗
IslamTrust (Lahmar et al., 2025) Benchmark 406 ✓ ✗ MCQ ✗
Qur’an QA 2022 (Malhas et al., 2022) Shared task 1,337 ✗ ✓ MRC ✗
IslamicEval 2025 (Mubarak et al., 2025) Shared task 1,506 ✗ ✓ PR ✗
QIAS 2025 (Bouchekif et al., 2025a) Shared task 22,000 ✗ ✓ MCQ ✗
PalmX 2025 (Alwajih et al., 2025) Shared task 1,900 ✗ ✗ MCQ ✗
Table 2: Comparison of ISLAMIC FAITH QAwith promi-
nent Islamic NLP resources. Size reports the primary
evaluation unit (e.g., QA pairs / MCQs; for IslamicEval
it is annotated answers). Text-grounded denotes ques-
tions grounded in canonical texts. Format: GenQA =
Generative QA; MRC = Machine Reading Comprehen-
sion; PR = Passage Retrieval; MCQ = Multiple Choice.
3.3 I SLAMIC FAITH QA Curation Pipeline
As illustrated in Figure 2, we employ a rigor-
ous semi-automated pipeline. We aggregate high-
quality samples from sources such as Hajj-FQA
(Aleid and Azmi, 2025), QIAS (Bouchekif et al.,
2025a), and PalmX (Alwajih et al., 2025). These
inputs undergo an automated Extraction and Filter-
Figure 2: The construction pipeline for ISLAMIC -
FAITH QA.
ingphase based on the difficulty level , and for the
datasets that had annotations of difficulty levels, we
select the hardest difficulty levels only. Before be-
ing reformulated by GPT-4.1 into short, fact-based
generative questions with atomic gold answers. To
enrich the benchmark, we then add a layer of meta-
data by performing a difficulty assessment. For this,
another LLM, acting as an expert evaluator, assigns
a difficulty score on a five-point scale (from “Very
Easy” to “Very Hard”) to each question. To en-
richISLAMIC FAITH QAwith calibrated difficulty
and reasoning metadata, we additionally employ
an LLM-based expert evaluator that assigns a five-
point difficulty score and binary reasoning indica-
tors ( reasoning ,multi_step ), along with a single
coarse-grained topic category from a fixed taxon-
omy; the full prompt templates used for this an-
notation step are provided in Appendix A.1 and
Appendix A.2.
To ensure theological validity beyond automated
filtering, we manually annotated a subset of the
dataset. Annotators were hired through a third-
party company and compensated at the standard
hourly rate for their location. All annotators were
professionals, fluent in both Arabic and English,
and held at least a bachelor’s degree. Each an-
notator signed a non-disclosure agreement (NDA)
specifying all permitted uses of the data. Each item
was annotated by three annotators. Disagreements
(Agreement rate was 82.96% and Cohen’s κof0.62
for three annotators for each item.) are resolved via
adjudication; items that fail validation are either re-
vised or removed, yielding a final benchmark with
auditable provenance and quantified human con-
sistency. For more details, please see Appendix F.
Diversity and Complexity Analysis. ISLAMIC -
FAITH QAis designed to cover a broad spectrum
of Islamic knowledge with a notable emphasis on
complex domains. As visualised in Figure 3, Inheri-
4

(a) Difficulty distribution across 5 levels
(b) Top 10 category distribution
Figure 3: Statistical analysis of ISLAMIC FAITH QA.(a)
Difficulty Distribution: The dataset exhibits a balanced
spread across five difficulty levels. (b) Topic Diversity:
The benchmark covers a wide range of theological do-
mains.
tance Law (26.4%) and Jurisprudence (17.4%) con-
stitute the largest categories, followed by Prophetic
Biography (11.4%), Islamic Creed (9.8%), and
Quranic Studies (9.4%). Regarding cognitive de-
mands, our analysis reveals that the majority of
samples ( 70.7% ) require active reasoning to derive
the correct answer, whereas only 29.3% rely on
simple fact recall. Furthermore, 55.4% of the ques-
tions necessitate multi-step reasoning, challenging
models to maintain context over longer inference
chains. In terms of difficulty, a distribution peaking
at Level 3 (31.2%), with substantial representa-
tion at Level 4 (21.8%) and Level 1 (22.8%) to
differentiate between basic and advanced model ca-
pabilities. Please see Appendix B for more details.
4 Methodology
We develop Islamic-domain LLMs that prioritise
Qur’an-grounded answer generation and explic-
itly measure hallucination under open-ended (gen-
erative) answering. Our approach combines do-
main adaptation through supervised fine-tuning
(SFT), preference-based alignment with an LLM-
as-a-judge reward signal, and retrieval augmenta-
tion over an indexed Qur’an corpus. At inferencetime, we further introduce an agentic RAG configu-
ration in which the model interacts with a Qur’anic
toolset via structured tool calls, enabling multi-step
evidence gathering before producing a cited answer.
Our problem statement and solution methods are
visualised in Figure 1 and Figure 4.
Experimental pipeline. Figure 4 summarises the
end-to-end development and evaluation workflow.
Starting from an Islamic corpus, we perform extrac-
tion and filtering to construct training resources and
generate reasoning-focused supervision. A base
LLM is then adapted via supervised fine-tuning
(SFT) and reward-guided alignment (RL), where
an LLM-as-a-judge provides verifiable reward sig-
nals. For inference, we deploy an Agentic RAG
environment in which the tuned model performs
multi-turn reasoning and queries a Qur’an/Hadith
database through dedicated tools and retrieval oper-
ations. Finally, we benchmark models on ISLAM -
ICFAITH QA using an LLM-as-a-judge protocol.
For more details, please see Appendix C.
Models: We evaluate a diverse set of Arabic-
centric and multilingual instruction-tuned LLMs
under a unified prompting and grading setup (Ta-
ble 3). Our Arabic-centric baselines include Fanar-
1-9B and Fanar-2-27B (Team et al., 2025), ALLaM-
7B (Bari et al., 2025), AceGPT-v2-8B (Liang
et al., 2024), and SILMA-9B-v1.0 (silma-ai, 2024).
We additionally benchmark multilingual models
spanning multiple families, including Qwen2.5-3B
and Qwen3 variants (Qwen3-4B-2507, Qwen3-8B,
Qwen3-14B) (Yang et al., 2025), Llama-2-7B and
Llama-3.1-8B (Touvron et al., 2023; Grattafiori
et al., 2024), Mistral-7B-v0.2 (Jiang et al., 2023),
SeaLLM-7B-v3 (Zhang et al., 2024), EuroLLM-
9B (Martins et al., 2025), and gpt-oss-20b (OpenAI
et al., 2025).
Figure 4: End-to-end development and evaluation work-
flow.
5

SFT for text-grounded reasoning. As shown
in the LLM Training stage of Figure 4, we per-
form supervised fine-tuning using 25,000 Arabic
instruction-response pairs ( SFT Reasoning ; Ta-
ble 1). Training uses a standard next-token pre-
diction objective over the target responses, with
the intent of improving (i)understanding of Is-
lamic theological concepts, (ii)the coherence of
multi-step reasoning, and (iii)adherence to source-
grounded answering behavior. In our experiments,
we train the Fanar-1-9B (Team et al., 2025), Allam-
7B (Bari et al., 2025) and Qwen3-4B-2507 (Yang
et al., 2025) large language models.
Group-Optimized RL Alignment (GSPO) To
further reduce hallucinations and improve answer
appropriateness in religious settings, we perform
reward-guided alignment (The second training
stage in Figure 4) using a bilingual (Arabic and En-
glish) RL Preference dataset of5,000 samples (Ta-
ble 1). Each instance contains a question derived
from canonical material, a gold-standard answer,
and evaluation parameters enabling scalar reward
assignment. We employ an LLM-as-a-judge within
the training loop to produce a score reflecting fac-
tual accuracy, clarity, completeness, and appropri-
ateness of candidate answers. This score is then
used as the reward signal for policy optimisation
using GSPO loss (Shao et al., 2024; Zheng et al.,
2025)), encouraging the model to favor grounded,
high-quality generations and discouraging unsup-
ported claims. The full judge prompt is provided in
Appendix A.4. In our RL experiments, we train the
Fanar-1-9B (Team et al., 2025), Allam-7B (Bari
et al., 2025) and Qwen3-4B-2507 (Yang et al.,
2025) generation models with GSPO-based RL,
while using Qwen3-235B-A22B (Yang et al., 2025)
as the discriminator (LLM-as-a-judge) to produce
reward signals.
RAG and Agentic RAG Figure 4 distinguishes
single-shot retrieval augmentation (standard RAG)
from tool-mediated evidence seeking (Agentic
RAG). In the standard RAG configuration, given a
user query we retrieve relevant Quranic verse units
from the indexed corpus and condition the model
on this retrieved context to produce an answer with
citations. For all retrieval-augmented experiments,
we use mE5-base (Wang et al., 2024) as the dense
retrieval encoder for indexing and querying the
Quran RAG dataset . Concretely, each ayah in our
6,236-unit corpus is embedded once offline with
mE5-base, and user questions (Arabic or English)Islamic Jurisprudence
Question (EN): When does the time for Fajr prayer begin?
Gold answer (EN): The time for Fajr prayer begins at true
dawn.
Model: Qwen3-4B + RAG
Predicted answer (EN): Fajr begins at dawn, i.e., sunrise
time when daylight starts to appear.
Judge: Incorrect
Model: Qwen3-4B + Agentic RAG
Predicted answer (EN): Fajr begins at true dawn ( al-fajr
al-¯s¯adiq), i.e., when the dawn becomes clearly distinct
from the night
Reference - Qur’an: 2:187 .
Judge: Correct
Figure 5: Example item from ISLAMIC FAITH QA
(Islamic Jurisprudence). We show the query,
the atomic gold target, and a model predic-
tion. The LLM-as-a-judge assigns a discrete la-
bel (Correct /Incorrect /Not_Attempted ) based on se-
mantic equivalence to the single-gold answer.
are embedded at inference time using the same
encoder. We then retrieve the top- 5most similar
verses via vector similarity search and pass these re-
trieved units to the generator (standard RAG) or ex-
pose them through tool calls (Agentic RAG). While
standard RAG improves factual grounding, it treats
retrieval as a single-shot preprocessing step. In con-
trast, our agentic RAG setup converts retrieval into
an explicit decision-making process within gener-
ation (Figure 4). The model is prompted to plan,
invoke tools, inspect retrieved verses, and iterate
if necessary before generating the final response.
This interaction is implemented using a constrained
tool-call schema and a Quranic toolset supporting
semantic search, surah metadata retrieval, direct
verse reading, and within-surah search. The full
agent system prompt and tool-call formatting re-
quirements are provided in Appendix A.3. In Fig-
ure 5, we show an example of ISLAMIC FAITH QA
and demonstrate the difference between outputs
from RAG and Agentic RAG settings. In our ex-
periments, we utilise the RAG and Agentic RAG
setups for Fanar-1-9B (Team et al., 2025), Allam-
7B (Bari et al., 2025), and Qwen3-4B-2507 (Yang
et al., 2025) after training. We employed the same
settings for Fanar-2-27B (Team et al., 2025) with-
out retraining.4
Evaluation on ISLAMIC FAITH QA.We eval-
uate all model variants on ISLAMIC FAITH QA.
Throughout the paper, we report %Correct as the
primary performance measure (Tables 3–4) under
4We did not fine-tune Fanar-2-27B given its already strong
baseline performance (Table 3).
6

Model Arabic English Average
Fanar-2-27B 48.20 47.90 48.05
ALLaM-7B 42.70 32.80 37.75
Fanar-1-9B 34.50 36.30 35.40
AceGPT-v2-8B 23.10 28.80 25.95
EuroLLM-9B 22.30 29.10 25.70
SILMA-9B-v1.0 20.40 28.50 24.45
Qwen3-4B-2507 15.80 27.90 21.85
gpt-oss-20b 15.90 27.20 21.55
Llama-3.1-8B 13.00 25.80 19.40
Mistral-7B-v0.2 13.50 24.40 18.95
SeaLLM-7B-v3 11.60 23.80 17.70
Qwen2.5-3B 11.00 20.00 15.50
Qwen3-14B 16.00 14.00 15.00
Llama-2-7b 4.40 18.80 11.60
Qwen3-8B 8.80 8.50 8.65
Table 3: Results on ISLAMIC FAITH QA(% Correct).
Fanar-2-27B achieves the highest performance, fol-
lowed by the ALLaM-7B model.
this grading protocol, and we analyze label distri-
butions (Correct/Incorrect/Not Attempted) to char-
acterize failure modes and abstention tendencies
across model variants. For full label distributions,
please see Table 7. Because ISLAMIC FAITH QA
relies on an LLM-as-a-judge protocol, we explic-
itly assess grader reliability rather than assuming it.
We calibrate the grader against human judgments
on a held-out bilingual subset ( N= 200 , balanced
by Arabic/English and difficulty) and report agree-
ment statistics: human–LLM agreement is 79%,
and inter-annotator agreement is measured using
Cohen’s κ(κ= 0.51). This analysis is notewor-
thy in multilingual settings: recent evidence shows
that multilingual LLM judges can be inconsistent
across languages, with only moderate inter-judge
agreement on average and substantial variance by
language and task (Fu and Liu, 2025).
5 Results
5.1 Baseline Results
Table 3 reports accuracy (%Correct) on ISLAMIC -
FAITH QAacross a diverse set of Arabic-centric
and multilingual instruction-tuned LLMs in their
base (non-retrieval) configurations. We observe
substantial variance in performance, indicating that
general-purpose instruction tuning alone is insuffi-
cient for this knowledge-intensive religious domain
under strict answer checking. The strongest overall
performance in this setting is achieved by Fanar-2-
27B (48.05 average; 48.20 Arabic / 47.90 English),
followed by ALLaM-7B (37.75 average) and Fanar-
1-9B (35.40 average).A second pattern is that many generalist multi-
lingual baselines remain below 30% average accu-
racy despite being competent in broad instruction-
following (e.g., EuroLLM-9B at 25.70, Llama-3.1-
8B at 19.40, and Mistral-7B-v0.2 at 18.95). This
highlights that ISLAMIC FAITH QAis not evaluat-
ing conversational fluency; it rewards precise, text-
grounded religious knowledge and penalizes confi-
dent but unsupported generations under the strict
Correct /Incorrect /Not_Attempted protocol.
Model Variation Arabic English Avg.
ALLaM-7B 42.70 32.80 37.75
+ SFT 45.20 31.40 38.30
+ RL 43.90 35.20 39.55
+ RAG 46.42 35.10 40.76
Fanar-1-9B 34.50 36.30 35.40
+ SFT 40.80 32.10 36.45
+ RL 42.90 33.45 38.18
+ RAG 47.90 34.50 41.20
Qwen3-4B-2507 15.80 27.90 21.85
+ SFT 25.90 35.20 30.55
+ RL 27.35 34.30 30.83
+ RAG 35.20 42.50 38.85
+ Agentic RAG 49.60 48.20 48.90
Fanar-2-27B 50.40 46.90 48.65
+ RAG 52.50 50.50 51.50
+ Agentic RAG 54.40 60.20 57.30
Table 4: Impact of supervised fine-tuning (SFT), rein-
forcement learning (RL), retrieval augmentation (RAG),
and agentic RAG (tool usage) on selected models.
5.2 SFT, RL, and RAG Models
Table 4 reports results for different model combina-
tions. Across backbones, three components consis-
tently improve performance: (i)domain-grounded
reasoning supervision via SFT, (ii)reward-guided
alignment via RL, and (iii)retrieval augmentation
(RAG). Among them, retrieval typically yields the
largest gains.
First, adding SFT on text-grounded theological
reasoning improves performance for all tested back-
bones, though the magnitude varies. The effect is
most pronounced for Qwen3-4B-2507, where SFT
increases average accuracy from 21.85 to 30.55. By
contrast, gains are smaller for stronger in-domain
baselines such as ALLaM-7B (37.75 →38.30) and
Fanar-1-9B (35.40 →36.45), suggesting dimin-
ishing returns when the base model already has
stronger domain priors.
Second, reward-guided alignment further im-
proves average accuracy beyond SFT for multi-
7

ple backbones (e.g., ALLaM-7B: 38.30 →39.55;
Fanar-1-9B: 36.45 →38.18), indicating that opti-
mizing with an LLM-judge reward encourages out-
puts that better match the benchmark’s constraints
(short, atomic answers with fewer risky additions).
Third, RAG provides consistent gains across
all backbones shown. For example, Qwen3-4B-
2507 improves from 30.83 (+RL) to 38.85 (+RAG),
Fanar-1-9B improves from 38.18 (+RL) to 41.20
(+RAG), and ALLaM-7B improves from 39.55
(+RL) to 40.76 (+RAG). These results confirm
that ISLAMIC FAITH QA is strongly knowledge-
intensive and that injecting canonical evidence re-
duces reliance on parametric memory.
Agentic RAG Yields the Largest Gains The
most salient result is the additional improvement
obtained by agentic RAG beyond single-shot RAG.
In Table 4, Qwen3-4B-2507 rises from 38.85
(+RAG) to 48.90 (+Agentic RAG), a gain of +10.05
points and the largest jump among the reported in-
terventions for that backbone. This suggests that,
for many questions, retrieval is not a one-step oper-
ation: models benefit from iterative evidence collec-
tion (e.g., retrieving candidate verses, reading spe-
cific ayat for disambiguation, and refining queries)
prior to final answer generation. Agentic RAG
also substantially strengthens the already-strong
Fanar-2-27B model: from 48.65 (base) to 51.50
(+RAG) and further to 57.30 (+Agentic RAG),
with particularly large gains in English (46.90 →
60.20). Overall, Fanar-2-27B + Agentic RAG is the
best-performing configuration reported in Table 4.
These findings indicate a complementary relation-
ship between backbone strength and tooling: larger
in-domain models provide stronger priors, while
agentic retrieval constrains generation toward veri-
fiable canonical evidence and improves robustness,
especially under bilingual evaluation.
5.3 Bilingual Gaps
Both Table 3 and Table 4 show that many models
exhibit asymmetric performance across Arabic and
English, reflecting differences in pretraining cover-
age, instruction tuning, and retrieval effectiveness
under bilingual queries. For instance, ALLaM-
7B performs substantially better in Arabic than
English (42.70 vs. 32.80), whereas several mul-
tilingual baselines show the opposite trend (e.g.,
EuroLLM-9B: 22.30 Arabic vs. 29.10 English).
Notably, Qwen3-4B-2507 is highly imbalanced in
its base form (15.80 Arabic vs. 27.90 English), un-derscoring that bilingual Islamic QA is not simply
an Arabic task with English translation; it requires
robust grounding and semantic access to canoni-
cal evidence in both languages. In contrast, tool-
mediated grounding can substantially reduce bilin-
gual disparities. Under Agentic RAG (Table 4),
Qwen3-4B-2507 becomes comparatively balanced
(49.60 Arabic vs. 48.20 English), suggesting that
iterative evidence seeking and explicit verse inspec-
tion help align performance across languages by an-
choring generation to the same canonical retrieval
base.
6 Conclusion
In this paper, we introduce ISLAMIC FAITH QA,
a benchmark dataset, along with an end-to-end
grounded Islamic modelling suite designed to eval-
uate and reduce hallucinations in open-ended re-
ligious generation directly. Using a unified re-
source suite for supervised domain reasoning,
judge-guided preference alignment, and Islamic-
centric retrieval, we systematically evaluated Base,
+SFT, +RL, +RAG, and +Agentic RAG variants
and found that retrieval substantially improves cor-
rectness, while agentic RAG yields the largest
gains beyond standard RAG by enabling iterative
evidence seeking and disambiguation through ex-
plicit tool use. Overall, our results indicate that
tool-mediated grounding can deliver state-of-the-
art performance and improved Arabic/English ro-
bustness even with smaller backbones, suggesting
a practical path toward more trustworthy Islamic
assistants; future work should extend grounding to
authenticated hadith with provenance, incorporate
school-of-thought disagreement, and harden tool-
augmented systems against adversarial prompting
and citation laundering.
Limitations
ISLAMIC FAITH QAis designed for reliable open-
ended evaluation using atomic questions with
single-gold answers and LLM-judge grading, but
this choice under-represents settings where multi-
ple answers may be valid across madh ¯ahib or inter-
pretive traditions. Our results also depend on the
correctness of the LLM judge and a limited human-
calibration subset, which may not fully capture
borderline cases or bilingual inconsistencies. In
addition, our grounding is primarily Quran-centric,
so questions best supported by authenticated ha-
dith, fiqh sources, or scholarly consensus may be
8

disadvantaged. Finally, agentic RAG introduces
added latency and new failure modes (e.g., tool-use
errors and citation laundering), and the benchmark
focuses on short-form QA rather than long-form
religious guidance; thus, performance should be
interpreted as faithfulness/abstention under strict
checking, not readiness for deployment as religious
authority.
Ethical Considerations
This work involves human annotation and the use
of automated language models in the dataset con-
struction pipeline. For the manually validated sub-
set of ISLAMIC FAITH QA, annotators were con-
tracted via a third-party provider, compensated at
the standard hourly rate for their location, and re-
quired to sign a non-disclosure agreement (NDA)
specifying permitted uses of the data. Large lan-
guage models were used only as editing tools to
standardize phrasing and tone (and to support struc-
tured metadata/difficulty labeling), and were not
treated as sources of religious authority; their out-
puts served as transformation/scaffolding steps and
were subsequently subject to human verification
and consistency checks. The benchmark was as-
sembled from publicly available/open-source Is-
lamic NLP resources rather than from newly col-
lected user data, which reduces risks related to
privacy, consent, and the handling of sensitive per-
sonal information; we do not release personally
identifying information. Given the domain sensitiv-
ity, we emphasize that the resulting benchmark and
models are intended for research on faithfulness
and abstention, not for issuing fatwas or replacing
qualified scholarly guidance.
Broader Impact
This work provides evaluation and grounding re-
sources for Islamic question answering, where un-
faithful outputs can be especially consequential.
By introducing a bilingual generative benchmark
that measures correctness, hallucination, and ab-
stention, and by showing that retrieval, particu-
larly agentic, tool-mediated retrieval, can reduce
unsupported generation, we aim to support more
trustworthy Arabic–English systems and more real-
istic assessment of faithfulness. At the same time,
these tools may be misused or over-trusted as re-
ligious authority, may reflect selection biases in
what is treated as canonical, and may enable per-
suasive “citation laundering” or adversarial manip-ulation of tool use. We therefore emphasize re-
sponsible release with clear non-fatwa disclaimers,
transparency about scope and coverage, encourage-
ment of abstention under uncertainty, and reporting
that separates correctness from hallucination and
non-attempted behavior.
Acknowledgments
The work is supported by HBKU flagship research
grant (HBKU-INT-VPR-SRG-03-10). The findings
achieved herein are solely the responsibility of the
authors.
References
Alok Abhishek, Lisa Erickson, and Tushar Bandopad-
hyay. 2025. Beats: Bias evaluation and assessment
test suite for large language models. 2503.24310v1 .
Utkarsh Agarwal, Kumar Tanmay, Aditi Khandelwal,
and Monojit Choudhury. 2024. Ethical reasoning
and moral value alignment of llms depend on the
language we prompt them in. 2404.18460v1 .
Aisha Alansari and Hamzah Luqman. 2025. AraHalluE-
val: A fine-grained hallucination evaluation frame-
work for Arabic LLMs. In Proceedings of The Third
Arabic Natural Language Processing Conference ,
pages 148–161, Suzhou, China. Association for Com-
putational Linguistics.
Hayfa A Aleid and Aqil M Azmi. 2025. Hajj-fqa: A
benchmark arabic dataset for developing question-
answering systems on hajj fatwas: H. aleid and a.
azmi. Journal of King Saud University Computer
and Information Sciences , 37(6):135.
Hamza Aljaji, Rawan Mohamed, Roaa Ibrahim, Ab-
dallah Alkanani, Arwa Abdulhakim Elaradi, and
Ehsaneddin Asgari. 2025. Benchmarking genera-
tive ai on quranic knowledge. In Proceedings of the
5th Muslims in ML Workshop at NeurIPS 2025 .
Fakhraddin Alwajih, Abdellah El Mekki, Hamdy
Mubarak, Majd Hawasly, Abubakr Mohamed, and
Muhammad Abdul-Mageed. 2025. PalmX 2025:
The first shared task on benchmarking LLMs on Ara-
bic and islamic culture. In Proceedings of The Third
Arabic Natural Language Processing Conference:
Shared Tasks , pages 774–789, Suzhou, China. Asso-
ciation for Computational Linguistics.
Bushra Asseri, Estabrag Abdelaziz, and Areej Al-Wabil.
2025. Prompt engineering techniques for mitigating
cultural bias against arabs and muslims in large lan-
guage models: A systematic review. 2506.18199v2 .
Farah Atif, Nursultan Askarbekuly, Kareem Darwish,
and Monojit Choudhury. 2025. Sacred or synthetic?
evaluating llm reliability and abstention for religious
questions. In Proceedings of the AAAI/ACM Con-
ference on AI, Ethics, and Society , volume 8, pages
217–226.
9

M Saiful Bari, Yazeed Alnumay, Norah A. Alzahrani,
Nouf M. Alotaibi, Hisham Abdullah Alyahya, Sultan
AlRashed, Faisal Abdulrahman Mirza, Shaykhah Z.
Alsubaie, Hassan A. Alahmed, Ghadah Alabduljab-
bar, Raghad Alkhathran, Yousef Almushayqih, Ra-
neem Alnajim, Salman Alsubaihi, Maryam Al Man-
sour, Saad Amin Hassan, Dr. Majed Alrubaian, Ali
Alammari, Zaki Alawami, and 7 others. 2025. AL-
Lam: Large language models for arabic and english.
InThe Thirteenth International Conference on Learn-
ing Representations .
Muhammad Huzaifa Bashir, Aqil M Azmi, Haq Nawaz,
Wajdi Zaghouani, and Mona Diab. 2021. Arabic
natural language processing for qur’anic research:
a systematic review. Artificial Intelligence Review ,
56(Suppl 1):13951–13993.
Gagan Bhatia, El Moatez Billah Nagoudi, Abdellah
El Mekki, Fakhraddin Alwajih, and Muhammad
Abdul-Mageed. 2025. Swan and ArabicMTEB:
Dialect-aware, Arabic-centric, cross-lingual, and
cross-cultural embedding models and benchmarks.
InFindings of the Association for Computational
Linguistics: NAACL 2025 , pages 4654–4670, Al-
buquerque, New Mexico. Association for Computa-
tional Linguistics.
Gagan Bhatia, El Moatez Billah Nagoudi, Abdellah El
Mekki, Fakhraddin Alwajih, and Muhammad Abdul-
Mageed. 2024. Swan and arabicmteb: Dialect-aware,
arabic-centric, cross-lingual, and cross-cultural em-
bedding models and benchmarks. 2411.01192v2 .
Abdessalam Bouchekif, Samer Rashwani, Emad Soli-
man Ali Mohamed, Mutaz Alkhatib, Heba Sbahi,
Shahd Gaben, Wajdi Zaghouani, Aiman Erbad, and
Mohammed Ghaly. 2025a. QIAS 2025: Overview
of the shared task on islamic inheritance reasoning
and knowledge assessment. In Proceedings of The
Third Arabic Natural Language Processing Confer-
ence: Shared Tasks , pages 851–860, Suzhou, China.
Association for Computational Linguistics.
Abdessalam Bouchekif, Samer Rashwani, Heba Sbahi,
Shahd Gaben, Mutaz Al-Khatib, and Mohammed
Ghaly. 2025b. Assessing large language models on
islamic legal reasoning: Evidence from inheritance
law evaluation.
Passant Elchafei and Mervet Abu-Elkheir. 2025. Span-
level hallucination detection for llm-generated an-
swers. 2504.18639v1 .
Fatma Elsafoury and David Hartmann. 2025. Out of
sight out of mind, out of sight out of mind: Measuring
bias in language models against overlooked marginal-
ized groups in regional contexts. 2504.12767v1 .
Xiyan Fu and Wei Liu. 2025. How reliable is multilin-
gual LLM-as-a-judge? In Findings of the Associa-
tion for Computational Linguistics: EMNLP 2025 ,
pages 11040–11053, Suzhou, China. Association for
Computational Linguistics.
Aryo Pradipta Gema, Alexander Hägele, Runjin Chen,
Andy Arditi, Jacob Goldman-Wetzler, Kit Fraser-
Taliente, Henry Sleight, Linda Petrini, Julian Michael,
Beatrice Alex, Pasquale Minervini, Yanda Chen, JoeBenton, and Ethan Perez. 2025. Inverse scaling in
test-time compute. 2507.14417v1 .
Aaron Grattafiori, Abhimanyu Dubey, Abhinav Jauhri,
Abhinav Pandey, Abhishek Kadian, Ahmad Al-
Dahle, Aiesha Letman, Akhil Mathur, Alan Schel-
ten, Alex Vaughan, Amy Yang, Angela Fan, Anirudh
Goyal, Anthony Hartshorn, Aobo Yang, Archi Mi-
tra, Archie Sravankumar, Artem Korenev, Arthur
Hinsvark, and 542 others. 2024. The llama 3 herd of
models. Preprint , arXiv:2407.21783.
Geyang Guo, Tarek Naous, Hiromi Wakaki, Yukiko
Nishimura, Yuki Mitsufuji, Alan Ritter, and Wei Xu.
2025. CARE: Multilingual human preference learn-
ing for cultural awareness. In Proceedings of the
2025 Conference on Empirical Methods in Natural
Language Processing , pages 32854–32883, Suzhou,
China. Association for Computational Linguistics.
Lukas Haas, Gal Yona, Giovanni D’Antonio, Sasha
Goldshtein, and Dipanjan Das. 2025. Simpleqa ver-
ified: A reliable factuality benchmark to measure
parametric knowledge. Preprint , arXiv:2509.07968.
Chang Hong, Minghao Wu, Qingying Xiao, Yuchi
Wang, Xiang Wan, Guangjun Yu, Benyou Wang, and
Yan Hu. 2025. Towards assessing medical ethics
from knowledge to practice. 2508.05132v1 .
Mohammad Hosseini, Kimia Hosseini, Shayan Bali,
Zahra Zanjani, and Saeedeh Momtazi. 2025. Perhal-
lueval: Persian hallucination evaluation benchmark
for large language models. 2509.21104v1 .
Yue Huang, Qihui Zhang, Philip S. Y , and Lichao Sun.
2023. Trustgpt: A benchmark for trustworthy and
responsible large language models. 2306.11507v1 .
Zheng Hui, Yijiang River Dong, Ehsan Shareghi, and
Nigel Collier. 2025. TRIDENT: Benchmarking llm
safety in finance, medicine, and law. arXiv preprint
arXiv:2507.21134 .
Albert Q. Jiang, Alexandre Sablayrolles, Arthur Men-
sch, Chris Bamford, Devendra Singh Chaplot, Diego
de las Casas, Florian Bressand, Gianna Lengyel, Guil-
laume Lample, Lucile Saulnier, Lélio Renard Lavaud,
Marie-Anne Lachaux, Pierre Stock, Teven Le Scao,
Thibaut Lavril, Thomas Wang, Timothée Lacroix,
and William El Sayed. 2023. Mistral 7b. Preprint ,
arXiv:2310.06825.
Junfeng Jiao, Saleh Afroogh, Abhejay Murali, Kevin
Chen, David Atkinson, and Amit Dhurandhar. 2025.
Llm ethics benchmark: A three-dimensional assess-
ment system for evaluating moral reasoning in large
language models. 2505.00853v1 .
Haoan Jin, Jiacheng Shi, Hanhui Xu, Kenny Q. Zhu, and
Mengyue Wu. 2025. Medethiceval: Evaluating large
language models based on chinese medical ethics.
2503.02374v1 .
Zahra Khalila, Arbi Haza Nasution, Winda Monika,
Aytug Onan, Yohei Murakami, Yasir Bin Ismail Radi,
and Noor Mohammad Osmani. 2025. Investigating
retrieval-augmented generation in quranic studies:
A study of 13 open-source large language models.
2503.16581v1 . International Journal of Advanced
10

Computer Science and Applications(IJACSA), 16(2),
2025.
Abderraouf Lahmar, Md Easin Arafat, Zakarya Farou,
and Mufti Mahmud. 2025. Islamtrust: A benchmark
for llms alignment with islamic values. In Proceed-
ings of the 5th Muslims in ML Workshop at NeurIPS
2025 .
Yangning Li, Weizhi Zhang, Yuyao Yang, Wei-Chieh
Huang, Yaozu Wu, Junyu Luo, Yuanchen Bei,
Henry Peng Zou, Xiao Luo, Yusheng Zhao, Chunkit
Chan, Yankai Chen, Zhongfen Deng, Yinghui Li,
Hai-Tao Zheng, Dongyuan Li, Renhe Jiang, Ming
Zhang, Yangqiu Song, and Philip S. Yu. 2025. To-
wards agentic rag with deep reasoning: A survey of
rag-reasoning systems in llms. 2507.09477v2 .
Jintao Liang, Gang Su, Huifeng Lin, You Wu, Rui
Zhao, and Ziyue Li. 2025. Reasoning rag via sys-
tem 1 or system 2: A survey on reasoning agen-
tic retrieval-augmented generation for industry chal-
lenges. 2506.10408v1 .
Juhao Liang, Zhenyang Cai, Jianqing Zhu, Huang
Huang, Kewei Zong, Bang An, Mosen Alharthi, Jun-
cai He, Lian Zhang, Haizhou Li, Benyou Wang, and
Jinchao Xu. 2024. Alignment at pre-training! to-
wards native alignment for arabic llms. Preprint ,
arXiv:2412.03253.
Rana Malhas and Tamer Elsayed. 2020. Ayatec: build-
ing a reusable verse-based test collection for arabic
question answering on the holy qur’an. ACM Trans-
actions on Asian and Low-Resource Language Infor-
mation Processing (TALLIP) , 19(6):1–21.
Rana Malhas, Watheq Mansour, and Tamer Elsayed.
2022. Qur’an qa 2022: Overview of the first shared
task on question answering over the holy qur’an. In
Proceedinsg of the 5th Workshop on Open-Source
Arabic Corpora and Processing Tools with Shared
Tasks on Qur’an QA and Fine-Grained Hate Speech
Detection .
Pedro Henrique Martins, João Alves, Patrick Fernan-
des, Nuno M. Guerreiro, Ricardo Rei, Amin Fara-
jian, Mateusz Klimaszewski, Duarte M. Alves, José
Pombal, Nicolas Boizard, Manuel Faysse, Pierre
Colombo, François Yvon, Barry Haddow, José G. C.
de Souza, Alexandra Birch, and André F. T. Mar-
tins. 2025. Eurollm-9b: Technical report. Preprint ,
arXiv:2506.04079.
Hamdy Mubarak, Rana Malhas, Watheq Mansour,
Abubakr Mohamed, Mahmoud Fawzi, Majd Hawasly,
Tamer Elsayed, Kareem Mohamed Darwish, and
Walid Magdy. 2025. IslamicEval 2025: The first
shared task of capturing LLMs hallucination in is-
lamic content. In Proceedings of The Third Arabic
Natural Language Processing Conference: Shared
Tasks , pages 480–493, Suzhou, China. Association
for Computational Linguistics.
Tarek Naous and Wei Xu. 2025. On the origin of cul-
tural biases in language models: From pre-training
data to linguistic phenomena. 2501.04662v1 .
OpenAI, :, Sandhini Agarwal, Lama Ahmad, Jason
Ai, Sam Altman, Andy Applebaum, Edwin Arbus,Rahul K. Arora, Yu Bai, Bowen Baker, Haiming Bao,
Boaz Barak, Ally Bennett, Tyler Bertao, Nivedita
Brett, Eugene Brevdo, Greg Brockman, Sebastien
Bubeck, and 108 others. 2025. gpt-oss-120b and gpt-
oss-20b model card. Preprint , arXiv:2508.10925.
Damith Premasiri, Tharindu Ranasinghe, W. Zaghouani,
and R. Mitkov. 2022. Dtw at qur’an qa 2022: Utilis-
ing transfer learning with transformers for question
answering in a low-resource domain.
Mohamad Al Mdfaa Raghad Salameh. 2024. Quranic
audio dataset: Crowdsourced and labeled recitation
from non-arabic speakers.
Peizhang Shao, Linrui Xu, Jinxi Wang, Wei Zhou, and
Xingyu Wu. 2025. When large language models
meet law: Dual-lens taxonomy, technical advances,
and ethical governance. 2507.07748v1 .
Zhihong Shao, Peiyi Wang, Qihao Zhu, Runxin Xu,
Junxiao Song, Xiao Bi, Haowei Zhang, Mingchuan
Zhang, Y . K. Li, Y . Wu, and Daya Guo. 2024.
Deepseekmath: Pushing the limits of mathemati-
cal reasoning in open language models. Preprint ,
arXiv:2402.03300.
silma-ai. 2024. Silma 9b instruct v1.0.
https://huggingface.co/silma-ai/
SILMA-9B-Instruct-v1.0 .
Fanar Team, Ummar Abbas, Mohammad Shahmeer Ah-
mad, Firoj Alam, Enes Altinisik, Ehsannedin Asgari,
Yazan Boshmaf, Sabri Boughorbel, Sanjay Chawla,
Shammur Chowdhury, Fahim Dalvi, Kareem Dar-
wish, Nadir Durrani, Mohamed Elfeky, Ahmed El-
magarmid, Mohamed Eltabakh, Masoomali Fatehkia,
Anastasios Fragkopoulos, Maram Hasanain, and 23
others. 2025. Fanar: An arabic-centric multimodal
generative ai platform.
Hugo Touvron, Louis Martin, Kevin Stone, Peter Al-
bert, Amjad Almahairi, Yasmine Babaei, Nikolay
Bashlykov, Soumya Batra, Prajjwal Bhargava, Shruti
Bhosale, Dan Bikel, Lukas Blecher, Cristian Canton
Ferrer, Moya Chen, Guillem Cucurull, David Esiobu,
Jude Fernandes, Jeremy Fu, Wenyin Fu, and 49 oth-
ers. 2023. Llama 2: Open foundation and fine-tuned
chat models. Preprint , arXiv:2307.09288.
Saad Obaid ul Islam, Anne Lauscher, and Goran Glavaš.
2025. How much do llms hallucinate across lan-
guages? on multilingual estimation of llm hallucina-
tion in the wild. 2502.12769v3 .
Changyue Wang, Weihang Su, Qingyao Ai, and Yiqun
Liu. 2025. Joint evaluation of answer and reason-
ing consistency for hallucination detection in large
reasoning models. 2506.04832v1 .
Liang Wang, Nan Yang, Xiaolong Huang, Linjun Yang,
Rangan Majumder, and Furu Wei. 2024. Multilin-
gual e5 text embeddings: A technical report. arXiv
preprint arXiv:2402.05672 .
Jianhui Wei, Zijie Meng, Zikai Xiao, Tianxiang Hu,
Yang Feng, Zhijie Zhou, Jian Wu, and Zuozhu Liu.
2025. Medethicsqa: A comprehensive question an-
swering benchmark for medical ethics evaluation of
llms. 2506.22808v1 .
11

An Yang, Anfeng Li, Baosong Yang, Beichen Zhang,
Binyuan Hui, Bo Zheng, Bowen Yu, Chang Gao,
Chengen Huang, Chenxu Lv, Chujie Zheng, Day-
iheng Liu, Fan Zhou, Fei Huang, Feng Hu, Hao
Ge, Haoran Wei, Huan Lin, Jialong Tang, and 41
others. 2025. Qwen3 technical report. Preprint ,
arXiv:2505.09388.
Wenxuan Zhang, Hou Pong Chan, Yiran Zhao, Mahani
Aljunied, Jianyu Wang, Chaoqun Liu, Yue Deng,
Zhiqiang Hu, Weiwen Xu, Yew Ken Chia, Xin Li, and
Lidong Bing. 2024. Seallms 3: Open foundation and
chat multilingual large language models for southeast
asian languages. Preprint , arXiv:2407.19672.
James Xu Zhao, Bryan Hooi, and See-Kiong Ng. 2025.
Test-time scaling in reasoning models is not effective
for knowledge-intensive tasks yet. 2509.06861v1 .
Chujie Zheng, Shixuan Liu, Mingze Li, Xiong-Hui
Chen, Bowen Yu, Chang Gao, Kai Dang, Yuqiong
Liu, Rui Men, An Yang, Jingren Zhou, and Jun-
yang Lin. 2025. Group sequence policy optimization.
Preprint , arXiv:2507.18071.
A Prompts and Templates
A.1 Question Generation
You are a senior academic and expert in Islamic
jurisprudence, ethics, and contemporary
global issues. You have been tasked with
authoring new entries for A Benchmark, an
English dataset designed to evaluate an
AI's ability to provide factually accurate
answers grounded in Islamic knowledge.
Your task is to generate a complete, structured
JSON object for a given topic. You must
adhere strictly to the format below. Your
reasoning should be based on foundational
Islamic sources (Qur 'an, Sunnah, classical
texts and contemporary Fiqh council
resolutions).
Follow these instructions precisely:
Question Formulation: For the given MCQ
question and answer provided in Arabic,
create a concise, short-form factual
question in English. The question should:
- Be direct and specific, requiring a factual
answer
- Focus on the core Islamic knowledge or ruling
being tested
- Avoid hypothetical scenarios or complex
ethical dilemmas
- Be answerable in 1-3 sentences
- Maintain the difficulty level indicated
(beginner/intermediate/advanced)
- Extract the key factual information from the
MCQ and its correct answer
Gold Answer: Provide the factual answer to the
question. This should:
- Be concise and direct (1-3 sentences maximum)
- State the Islamic ruling, principle, or fact
clearly
- Be based on the correct answer from the MCQ
provided- Reference the specific Islamic source (Qur 'an
verse, Hadith reference, scholarly
consensus) that supports this answer
- Avoid lengthy explanations - just state the
fact and its primary source
IMPORTANT: Both the question and gold_answer
should be in Arabic.
Follow this output format:
{
"id": "MIZAN-001",
"category": "Islamic Jurisprudence",
"question": "What is the ruling on performing
ablution (wudu) after eating camel meat?",
"gold_answer": "Ablution is required after
eating camel meat according to the Hadith
narrated by Jabir ibn Samurah in Sahih
Muslim (360), where the Prophet (peace be
upon him) explicitly instructed to perform
ablution after eating camel meat.",
}
A.2 Difficultly Generation
You are an expert evaluator of Islamic
knowledge questions. Your task is to assess
the difficulty level of questions on a
scale of 1-5, determine the reasoning
requirements, and classify the question
into an appropriate category.
Difficulty Scale:
1 = Very Easy: Basic factual recall, simple
definitions, or straightforward yes/no
questions
2 = Easy: Requires basic understanding of
concepts with minimal reasoning
3 = Moderate: Requires understanding multiple
concepts and some analytical reasoning
4 = Hard: Requires deep understanding,
synthesis of multiple sources, and nuanced
reasoning
5 = Very Hard: Requires expert-level analysis,
balancing competing interests, and
consideration of complex ethical frameworks
Reasoning Assessment:
- reasoning: Does answering this question
require reasoning beyond simple recall?
(true/false)
- multi_step: Does the reasoning require
multiple logical steps or considerations?
(true/false)
Examples of multi-step: comparing multiple
sources, weighing competing principles,
applying rules to specific contexts,
building logical chains
Category Classification:
Classify the question into ONE of these
categories:
1. "Islamic Creed" - Questions about belief in
Allah, prophets, angels, books, Day of
Judgment, divine decree
2. "Jurisprudence" - Questions about worship
rituals, purification, prayer, fasting,
hajj, transactions
12

3. "Inheritance Law" - Questions about Islamic
inheritance calculations and distributions
4. "Hadith Studies" - Questions about prophetic
traditions, their authentication, and
narrators
5. "Qur 'anic Studies" - Questions about
Qur'anic verses, tafsir, themes, stories,
and interpretation
6. "Prophetic Biography" - Questions about the
life of Prophet Muhammad and his companions
7. "Islamic History" - Questions about Islamic
historical events, figures, and
civilizations
8. "Islamic Ethics and Morality" - Questions
about moral principles, character, social
interactions
9. "Islamic Finance and Economics" - Questions
about halal transactions, banking, business
contracts
10. "Islamic Family Law" - Questions about
marriage, divorce, child custody, family
rights
11. "Comparative Religion" - Questions about
other religions from Islamic perspective
12. "Contemporary Issues" - Questions about
modern applications of Islamic rulings
Evaluate the question based on:
- Depth of knowledge required
- Complexity of reasoning needed
- Number of sources and concepts involved
- Level of nuance and ethical complexity
- Whether simple recall suffices or reasoning
is needed
- Whether the reasoning involves single or
multiple logical steps
- Subject matter and domain of the question
Respond ONLY with a JSON object in this exact
format:
{"difficulty": <number>, "reasoning":
<boolean>, "multi_step": <boolean>,
"category_type": "<category_name>"}
where <number> is an integer from 1 to 5, the
booleans are true or false, and
<category_name> is one of the 12 categories
listed above.
A.3 Agentic RAG System Prompt
You are an intelligent assistant specialized in
answering questions about Islam and the
Holy Quran. Answer accurately and clearly
based on Quranic sources.
You have access to the following tools to
search the Holy Quran:
1. search_quran(query: str) - Search for
relevant Quran verses by semantic similarity
2. get_surah_info(surah_number: int) - Get
information about a specific surah
3. read_ayah(surah: int, ayah: int) - Read a
specific ayah with full details
4. search_surah(surah_number: int, query: str)
- Search within a specific surah
To use a tool, respond with:
<tool_call>{"name": "tool_name", "arguments": {"arg1":
"value1"}}
</tool_call>
After gathering information from the Quran,
provide your final answer with:
<answer>Your final answer here, citing the
relevant Quran verses</answer>
Think step by step, search the Quran for
relevant verses, and cite your sources in
your answer.
A.4 RL LLM-as-a-Judge Prompt
You are an expert judge evaluating answers
about Islam and the Quran.
Rate the quality, accuracy, and completeness of
the following answer on a scale of 0-10.
Consider: factual accuracy, clarity,
completeness, and appropriateness.
Question: {question}
Answer: {answer}
Provide only a numeric score between 0 and 10.
Output format: Score: X
A.5 I SLAMIC FAITH QA Grader Template
(Correct/Incorrect/Not Attempted)
Your job is to look at a question, a gold
target, and a predicted answer, and then
assign a grade of either ["CORRECT",
"INCORRECT", "NOT_ATTEMPTED"].
First, I will give examples of each grade, and
then you will grade a new example.
The following are examples of CORRECT predicted
answers.
```
Question: What are the names of Barack Obama 's
children?
Gold target: Malia Obama and Sasha Obama
Predicted answer 1: sasha and malia obama
Predicted answer 2: most people would say Malia
and Sasha, but I 'm not sure and would have
to double check
Predicted answer 3: Barack Obama has two
daughters. Their names are Malia Ann and
Natasha Marian, but they are commonly
referred to as Malia Obama and Sasha Obama.
Malia was born on July 4, 1998, and Sasha
was born on June 10, 2001.
```
These predicted answers are all CORRECT because:
- They fully contain the important
information in the gold target.
- They do not contain any information that
contradicts the gold target.
- Only semantic meaning matters;
capitalization, punctuation, grammar, and
order don 't matter.
13

- Hedging and guessing are permissible,
provided that the gold target is fully
included and the response contains no
incorrect information or contradictions.
The following are examples of INCORRECT
predicted answers.
```
Question: What are the names of Barack Obama 's
children?
Gold target: Malia and Sasha
Predicted answer 1: Malia.
Predicted answer 2: Malia, Sasha, and Susan.
Predicted answer 3: Barack Obama does not have
any children.
Predicted answer 4: I think it 's either Malia
and Sasha. Or it could be Malia and Jackie.
Or it could be Joey and Malia.
Predicted answer 4: While I don 't know their
exact names, I can tell you that Barack
Obama has three children.
Predicted answer 5: It 's possible you may mean
Betsy and Olivia. However, you should
clarify further details with updated
references if necessary. Is that the
correct answer?
Predicted answer 6: It may be the case that
Obama 's child is named James. However, it 's
recommended to confirm the most accurate
and updated information since this could
change over time. This model may not always
reflect the most current information.
```
These predicted answers are all INCORRECT
because:
- A factual statement in the answer
contradicts the gold target. Incorrect
statements that have some hedging (e.g.,
"it is possible that", "although i 'm not
sure, i think") are also considered
incorrect.
The following are examples of NOT_ATTEMPTED
predicted answers.
```
Question: What are the names of Barack Obama 's
children?
Gold target: Malia and Sasha
Predicted answer 1: I don 't know.
Predicted answer 2: I need more context about
which Obama you are talking about.
Predicted answer 3: Without researching the
web, I cannot answer this question.
However, I can tell you that Barack Obama
has two children.
Predicted answer 4: Barack Obama has two
children. I know that one of them is Malia,
but I 'm not sure about the other one.
```
These predicted answers are all NOT_ATTEMPTED
because:
- The important information in the gold
target is not included in the answer.
- No statements in the answer contradictthe gold target.
Also note the following things:
- For grading questions where the gold target
is a number, the predicted answer needs to
be correct to the last significant figure
in the gold answer. For example, consider a
question "How many citations does the
Transformer Paper have?" with gold target
"120k".
- Predicted answers "120k", "124k", and
115k" are all CORRECT.
- Predicted answers "100k" and "113k" are
INCORRECT.
- Predicted answers "around 100k" and "more
than 50k" are considered NOT_ATTEMPTED
because they neither confirm nor contradict
the gold target.
- The gold target may contain more information
than the question. In such cases, the
predicted answer only needs to contain the
information that is in the question.
- For example, consider the question "What
episode did Derek and Meredith get legally
married in Grey 's Anatomy?" with gold
target "Season 7, Episode 20: White
Wedding". Either "Season 7, Episode 20" or
"White Wedding" would be considered a
CORRECT answer.
- Do not punish predicted answers if they omit
information that would be clearly inferred
from the question.
- For example, consider the question "What
city is OpenAI headquartered in?" and the
gold target "San Francisco, California".
The predicted answer "San Francisco" would
be considered CORRECT, even though it does
not include "California".
- Consider the question "What award did A
pretrainer 's guide to training data:
Measuring the effects of data age, domain
coverage, quality, & toxicity win at NAACL
'24?", the gold target is "Outstanding
Paper Award". The predicted answer
"Outstanding Paper" would be considered
CORRECT, because "award" is presumed in the
question.
- For the question "What is the height of
Jason Wei in meters?", the gold target is
"1.73 m". The predicted answer "1.75" would
be considered CORRECT, because meters is
specified in the question.
- For the question "What is the name of
Barack Obama 's wife?", the gold target is
"Michelle Obama". The predicted answer
"Michelle" would be considered CORRECT,
because the last name can be presumed.
- Do not punish for typos in people 's name if
it's clearly the same name.
- For example, if the gold target is "Hyung
Won Chung", you can consider the following
predicted answers as correct: "Hyoong Won
Choong", "Hyungwon Chung", or "Hyun Won
Chung".
Here is a new example. Simply reply with either
CORRECT, INCORRECT, NOT ATTEMPTED. Don 't
apologize or correct yourself if there was
14

a mistake; we are just trying to grade the
answer.
```
Question: {question}
Gold target: {target}
Predicted answer: {predicted_answer}
```
Grade the predicted answer of this new question
as one of:
A: CORRECT
B: INCORRECT
C: NOT_ATTEMPTED
Just return the letters "A", "B", or "C", with
no text around it.
B I SLAMIC FAITH QA Analysis
Figure 6: Category distribution (counts) in the Islamic
knowledge dataset.
Figure 7: Overall difficulty distribution and reasoning
requirements by difficulty level.
Figure 8: Reasoning complexity and step distribution
(multi-step vs. single-step vs. recall).
Figure 9: Difficulty distribution across top categories
(heatmap).
C Experimental Details
This appendix summarizes the infrastructure,
implementations, and key hyperparameters used
across evaluation, supervised fine-tuning (SFT),
preference-based alignment (RL), retrieval-
augmented generation (RAG), and agentic
RAG.
Compute Infrastructure. All experiments were
conducted on NVIDIA H200 GPUs. We use vLLM
for efficient batched inference during benchmark-
ing and for high-throughput generation when col-
lecting model outputs.
LLM Inference and Evaluation Parameters.
For benchmark evaluation, decoding is performed
with vLLM . We use a sampling temperature of
T= 0.7and otherwise retain the standard/default
generation parameters provided by the inference
framework to ensure consistent evaluation across
model backbones (e.g., default settings for top-
p, repetition controls, and maximum generation
length).
LLM-as-a-Judge Grading. For
automatic grading under the
Correct /Incorrect /Not_Attempted proto-
col, we use GPT-4.1 as the judge model. To
improve throughput and cost efficiency, we run
judging via the provider’s Batch API. We set the
judge temperature to T= 0to minimize sampling
variance and encourage deterministic scoring given
the same inputs (question, gold target, and model
prediction).
Supervised Fine-Tuning (SFT). For supervised
adaptation on our Arabic text-grounded reasoning
data, we follow standard instruction-tuning config-
urations in our training stack. The primary devia-
tion from defaults is the learning rate, which we
15

Islamic Ethics and Morality
Question: What is the main objective of training oneself
to love doing good in Islam?
Gold answer: The main objective is to achieve true hap-
piness, as loving and doing good leads to inner peace and
fulfillment.
Model: AceGPT-v2-8B
Predicted answer: The main objective is to cultivate
a habit of performing good deeds, improve charac-
ter, earn rewards, and contribute positively to society.
Judge: INCORRECT
Figure 10: ISLAMIC FAITH QA sample.
set to 5×10−5. All other hyperparameters (e.g.,
optimizer choice, batch size, warmup schedule, gra-
dient clipping, and number of epochs) use standard
settings.
Preference-Based Alignment (RL). For align-
ment, we optimize the policy using the GSPO ob-
jective ( GSPO loss ). RL experiments are imple-
mented using the trllibrary. We set the learning
rate to 3×10−6and keep remaining RL hyper-
parameters at standard values in our setup (e.g.,
rollout sampling and optimization schedule, and
any applicable regularization/clipping). Reward
signals are derived from the LLM-as-a-judge grad-
ing described above.
Retrieval-Augmented Generation (RAG). For
dense retrieval, we embed both queries and
Qur’anic verse units using mE5-base . We index
all verse embeddings in ChromaDB as our vector
database and retrieve relevant verses via vector
similarity search at inference time. Retrieved verse
units are provided as evidence context for standard
RAG generation.
Agentic RAG Configuration. Our agentic RAG
variant uses a fixed multi-turn interaction budget
for controlled latency and comparability. We em-
ploy a two-turn setup: (i) an evidence-seeking turn
in which the model invokes retrieval and inspects
candidate verses, followed by (ii) a final answer
turn conditioned on the retrieved and inspected ev-
idence. The agentic tool-calling environment is
implemented using the verifiers library, which
enforces the structured multi-turn tool-use protocol
described in Appendix A.3.
D I SLAMIC FAITH QA ExamplesE Training data Examples
E.1 SFT Data Example
SFT Training Example (Arabic + English)
Question (AR):	àA¿ @	X@Zñ	ñË@ ú
	¯ H.P@ñm.Ì'@úÎ« iÖ Ï@ Õºk ñë AÓ?èYKAÖÏ@èPñ ú
	¯ Zñ	ñË@éK
@ Èð 	Q	K YªK .ÕÎ@ Y¯ 	jË@Question (EN): What is the ruling on wiping over socks
during wud ¯u’ if a person embraced Islam after the revela- tion of the wud ¯u’ verse in S ¯urat al-M ¯a’idah?
Reasoning (AR):é	JËAK.IK.AK Zñ	ñË@ ú
	¯	á
	®	mÌ'@ ð@ H.P@ñm.Ì'@úÎ« iÖ Ï@iÓ é	K@ ú
æ.	JË@	á«éjJ
mIK
XAg@HXPð IJ
k ,éK
ñJ.	JË@
	áK
	YË@ AîE 
@ AK
)èYKAÖÏ@èPñ ú
	¯ Zñ	ñË@éK
@ AÓ@ .	á
	®	mÌ'@úÎ«
ÕÎ	¯(... ÕºK 
YK
@ð Õºëñk .ð @ñÊ	«A	¯èCË@ úÍ@ ÕæÔ¯@	X@@ñ	JÓ@é	J
J.ÓHZAg .é	JË@ 	áºË ,ék@Qå	á
	®	mÌ'@úÎ« iÖ Ï@ Q»	YK
úÎ« iÖ Ï@	P@ñk.úÎ« ZAÒÊªË@ ©Ô g.@ Y¯ð .	à@Q®ÊËèQå	®Óð
úÎ« AÒîDJ .ÊK
	à@ Aî	DÓ ,é	¯ðQªÓ  ðQå .H.P@ñm.Ì'@ð	á
	®	mÌ'@
ð@éK
B@ Èð 	Q	K ÉJ.¯ ÕÎ@	áÓ	á
K.ÕºmÌ'@ ú
	¯Q	¯ Bð .èPAê£	P@ñk.I	J
K.ðéK
B@ Èð 	Q	K YªK .HZAg .é	JË@	àB , AëYªK .
QÒJ@ð , AêËð 	Q	K YªK .@ñÒÊ@	áK
	YË@éK.AjË@ Éª	¯ AÒ» , iÖ Ï@
. èYªK .ð ú
æ.	JË@ Yê« ú
	¯ ½Ë	YK.ÉÒªË@Reasoning (EN): Wiping over socks/khuffs in wud ¯u’ is
established by the Prophetic Sunnah through authentic ha-dith reports. Although the wud ¯u’ verse in S ¯urat al-M ¯a’idah
does not explicitly mention wiping over khuffs, the Sunnah
functions as an explanation and elaboration of the Qur’an.
Scholars have agreed on the permissibility of wiping over
khuffs/socks under well-known conditions (e.g., wearing
them in a state of purity). The ruling does not differ for
someone who became Muslim before or after the verse
was revealed, since the Sunnah clarified the permissibility
and the Companions acted upon it thereafter.
Answer (AR):
Èð	Q	K YªK .ÕÎ@	áÖÏZñ	ñË@ ú
	¯ H.P@ñm.Ì'@úÎ« iÖ Ï@	Pñm.'
 ú
	¯ AÒ» , ½Ë	XI	J
K.éjJ
jË@éK
ñJ.	JË@é	JË@	àB , Zñ	ñË@éK
@
.(ÕÎÓð ø
PA	jJ.Ë@ è@ðP) è Q
	«ðéJ.ª	áK.èQ
	ªÖÏ@IK
YgAnswer (EN): It is permissible to wipe over socks in wud ¯u’
for someone who embraced Islam after the wud ¯u’ verse,because authentic Sunnah establishes this (e.g., the hadith
of al-Mugh ¯irah ibn Shu 'bah and others, reported in al-
Bukh ¯ar¯ı and Muslim). Figure 11: SFT sample (text-grounded reasoning).
An instruction-response instance used for supervised
fine-tuning, including the Arabic question, a grounded
reasoning trace, and a concise final answer (with an
English translation for readability).
E.2 RL Data Example
F Human Annotation Guidelines
This appendix documents the annotation guideline
used to label Islamic knowledge questions with
16

RL Preference Example (Arabic + English)
Question (AR):Y	J« éË HYjJ 
 AÓð éK .P	áÓéÔgQ» éª	J AÓ	­ð	­J
»
? éK.P Y«ð
Question (EN): How did he describe what he built as a
mercy from his Lord, and what will happen to it when his
Lord’s promise comes?
Gold answer (AR):ZA¿X éÊªj .J
 éK.P Y«ð Y	J«ð , é¯@Q	g@ @ñªJ 
¢
B@XP	àA¿
Reference - Qur’an: 18:98–99
Gold answer (EN): It was a barrier they could not break
through, and when his Lord’s promise comes He will level
it to the ground. Reference - Qur’an: 18:98–99
Figure 12: RL preference sample. An RL instance
specifies a question derived from canonical text and an
atomic gold target. During RL, candidate model re-
sponses are scored by an LLM-as-a-judge against this
gold target to produce scalar rewards for policy optimisa-
tion (English translations are provided for readability).
(i) a difficulty score, (ii) reasoning requirements,
(iii) multi-step reasoning requirements, and (iv) a
single category label. Annotators follow the defini-
tions and decision rules below to ensure consistent
ratings.
F.1 Task Overview
For each question, annotators assign:
• a difficulty score on a 1–5 scale,
•reasoning (true /false ): whether answering re-
quires reasoning beyond simple recall,
•multi_step (true /false ): whether the required
reasoning involves multiple steps, and
•category_type : exactly one category label
from a fixed set of 12.
F.2 Reference Solver for Difficulty Ratings
Difficulty is rated for a competent Islamic knowl-
edge solver : someone with solid baseline Islamic
literacy who reasons carefully. Annotators should
not rate based on personal familiarity, but rather on
how hard the question is for this reference solver
to answer correctly.
F.3 Difficulty Scale (1–5)
Annotators assign a single integer from 1 to 5 using
the definitions below.
F.3.1 Difficulty Definitions
F.3.2 Factors That Should Affect Difficulty
Annotators consider:Score Label Definition
1 Very Easy Basic factual recall, simple definitions, or
straightforward yes/no questions.
2 Easy Requires basic understanding of concepts
with minimal reasoning.
3 Moderate Requires understanding multiple concepts
and some analytical reasoning.
4 Hard Requires deep understanding, synthesis of
multiple sources/concepts, and nuanced rea-
soning.
5 Very Hard Requires expert-level analysis, balancing
competing interests, and consideration of
complex ethical frameworks.
Table 5: Difficulty rating scale used for question anno-
tation.
•depth of knowledge required (basic vs. special-
ized),
•complexity of reasoning needed (recall vs. appli-
cation vs. synthesis vs. balancing tradeoffs),
• number of concepts or sources involved,
•level of nuance (exceptions, conditions, context
sensitivity, khil¯af),
•whether simple recall suffices or reasoning is
necessary.
Note: a question can be difficult due to obscure
knowledge even if it is not multi-step.
F.3.3 Tie-break Rules
•Choose the higher score if mistakes are likely due
to nuance, exceptions, or competing principles.
•Choose the lower score if the answer is direct and
reliably determined from a single well-known
rule or fact.
•If the question is underspecified, keep the score
honest and flag the issue in the interface notes (if
available).
F.4 Reasoning Assessment
F.4.1 reasoning (true /false )
•reasoning = false if the answer is simple re-
call/definition (no inference).
•reasoning = true if answering requires apply-
ing, interpreting, comparing, reconciling, justify-
ing, or inferring.
F.4.2 multi_step (true /false )
Setmulti_step = true only if multiple logical
steps/considerations are required, such as:
• comparing multiple sources or viewpoints,
•weighing competing principles (harms vs. bene-
fits, conflicting obligations),
•applying a rule, then an exception/condition, then
concluding,
• building a chain with intermediate conclusions.
17

Setmulti_step = false if reasoning is present
but essentially one step (a single application or
inference).
F.4.3 Consistency Rules (Must Follow)
•Ifreasoning = false , then multi_step must
befalse .
•Ifmulti_step = true , then reasoning must
betrue .
F.5 Category Classification (Choose One)
Annotators assign category_type to exactly one
of the following category names (exact strings):
• Islamic Creed
• Jurisprudence
• Inheritance Law
• Hadith Studies
• Qur’anic Studies
• Prophetic Biography
• Islamic History
• Islamic Ethics and Morality
• Islamic Finance and Economics
• Islamic Family Law
• Comparative Religion
• Contemporary Issues
F.5.1 Boundary Rules
•Modern banking/finance products →Islamic
Finance and Economics .
•Marriage/divorce/custody →Islamic Family
Law; inheritance shares/heirs →Inheritance
Law.
•Hadith authentication/narrators/classification →
Hadith Studies ; hadith used mainly to derive
a ruling →usually Jurisprudence .
•S¯ırah→Prophetic Biography ; later eras/dy-
nasties →Islamic History .
•Novel modern scenario →Contemporary
Issues ; timeless moral teaching →Islamic
Ethics and Morality .
F.6 Ambiguity and Missing Context
Some questions may be underspecified or admit
multiple valid scholarly answers. In such cases,
annotators:
•still assign the best category based on the main
domain being tested,
•rate difficulty based on what is required to answer
responsibly (often higher if many qualifications
are needed),
•flag the issue briefly in the interface notes field
(if available), and• do not add extra keys to the JSON output.
F.7 Worked Examples
The examples below illustrate how to apply the
labels (they are not taken from the dataset).
Question Label
What is tawh ¯ıd? Difficulty: 1
Reasoning: false
Multi_step: false
Category_type: Islamic
Creed
Explain the difference between
w¯ajibandsunnah acts.Difficulty: 2
Reasoning: true
Multi_step: false
Category_type: Jurispru-
dence
A person touched their spouse
and then prayed. Does this in-
validate wud ¯u’? Explain.Difficulty: 3
Reasoning: true
Multi_step: true
Category_type: Jurispru-
dence
Compute inheritance shares
when the deceased leaves a wife,
two daughters, and parents.Difficulty: 4
Reasoning: true
Multi_step: true
Category_type: Inheri-
tance Law
Classify a hadith given narra-
tor reliability and continuity of
isn¯ad.Difficulty: 4
Reasoning: true
Multi_step: true
Category_type: Hadith
Studies
Evaluate a modern bioethical
dilemma by balancing harm-
s/benefits and competing obliga-
tions.Difficulty: 5
Reasoning: true
Multi_step: true
Category_type: Contem-
porary Issues
Table 6: Worked examples illustrating the annotation
guideline.
F.8 Final Checklist
• Output is JSON only (no extra text).
•difficulty is an integer in {1,2,3,4,5}.
•Consistency: reasoning=false ⇒
multi_step=false ;multi_step=true ⇒
reasoning=true .
•category_type matches exactly one of the 12
category strings.
F.9 Annotation Results and Agreement
We report summary statistics for the human anno-
tation stage and quantify annotator consistency on
a held-out subset with redundant labeling. In total,
we collected 3810 annotation records. Difficulty
labels are distributed across the 1–5 scale, with the
largest mass at level 3 (26.90%), followed by level
2 (23.03%), level 4 (20.03%), level 1 (17.67%),
18

Figure 13: Annotation Interface
and level 5 (12.37%). For category assignment,
we consolidate heterogeneous source-specific tags
into 12 unified category_type labels, with the
largest classes being Inheritance Law ,Islamic
Finance and Economics ,Hadith Studies ,
Qur’anic Studies andIslamic History . To
assess reliability, we additionally annotate a sub-
set of 315 items with three independent annotators
per item. On this subset, we observe an overall
agreement rate of 82.96% and a Cohen’s κof 0.62,
indicating substantial agreement and supporting the
consistency of the guidelines.
GResults with Correct Incorrect and Not
Attempted
19

ModelArabic EnglishAverage
Correct Incorrect Not Attempted Correct Incorrect Not Attempted
Fanar-2-27B 48.20 21.50 30.30 47.90 30.20 21.90 48.05
ALLaM-7B 42.70 52.90 4.40 32.80 63.70 3.50 37.75
Fanar-1-9B 34.50 54.10 11.40 36.30 55.10 8.60 35.40
AceGPT-v2-8B 23.10 64.30 12.60 28.80 57.20 14.00 25.95
EuroLLM-9B 22.30 67.20 10.50 29.10 64.50 6.40 25.70
SILMA-9B-v1.0 20.40 70.90 8.70 28.50 66.10 5.40 24.45
Qwen3-4B-2507 15.80 45.30 38.90 27.90 45.20 26.90 21.85
gpt-oss-20b 15.90 22.60 61.50 27.20 27.20 45.60 21.55
Llama-3.1-8B 13.00 74.00 13.00 25.80 47.40 26.80 19.40
Mistral-7B-v0.2 13.50 53.50 33.00 24.40 59.10 16.50 18.95
SeaLLM-7B-v2.5 11.60 76.30 12.10 23.80 64.80 11.40 17.70
Qwen2.5-3B 11.00 61.20 27.80 20.00 63.20 16.80 15.50
Qwen3-14B 16.00 12.50 71.50 14.00 4.20 81.80 15.00
Llama-2-7b 4.40 47.20 48.40 18.80 72.00 9.20 11.60
DeepSeek-R1-0528-Qwen3-8B 6.30 17.70 76.00 11.90 13.30 74.80 9.10
Qwen3-8B 8.80 10.00 81.20 8.50 5.60 85.90 8.65
Qwen3-4B-Thinking-2507 6.50 3.10 90.40 9.40 14.20 76.40 7.95
Qwen3-4B 6.50 17.30 76.20 9.00 9.60 81.40 7.75
Qwen3-1.7B 3.20 18.10 78.70 5.10 12.60 82.30 4.15
Qwen3-0.6B 1.30 47.00 51.70 5.40 39.70 54.90 3.35
DeepSeek-R1-Distill-Qwen-7B 1.40 37.40 61.20 4.30 37.60 58.10 2.85
DeepSeek-R1-Distill-Qwen-1.5B 0.10 21.60 78.30 1.00 41.10 57.90 0.55
Table 7: Results including Correct/Incorrect and not attempted.
20