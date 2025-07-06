# Rethinking All Evidence: Enhancing Trustworthy Retrieval-Augmented Generation via Conflict-Driven Summarization

**Authors**: Juan Chen, Baolong Bi, Wei Zhang, Jingyan Sui, Xiaofei Zhu, Yuanzhuo Wang, Lingrui Mei, Shenghua Liu

**Published**: 2025-07-02 01:39:49

**PDF URL**: [http://arxiv.org/pdf/2507.01281v1](http://arxiv.org/pdf/2507.01281v1)

## Abstract
Retrieval-Augmented Generation (RAG) enhances large language models (LLMs) by
integrating their parametric knowledge with external retrieved content.
However, knowledge conflicts caused by internal inconsistencies or noisy
retrieved content can severely undermine the generation reliability of RAG
systems.In this work, we argue that LLMs should rethink all evidence, including
both retrieved content and internal knowledge, before generating responses.We
propose CARE-RAG (Conflict-Aware and Reliable Evidence for RAG), a novel
framework that improves trustworthiness through Conflict-Driven Summarization
of all available evidence.CARE-RAG first derives parameter-aware evidence by
comparing parameter records to identify diverse internal perspectives. It then
refines retrieved evidences to produce context-aware evidence, removing
irrelevant or misleading content. To detect and summarize conflicts, we distill
a 3B LLaMA3.2 model to perform conflict-driven summarization, enabling reliable
synthesis across multiple sources.To further ensure evaluation integrity, we
introduce a QA Repair step to correct outdated or ambiguous benchmark
answers.Experiments on revised QA datasets with retrieval data show that
CARE-RAG consistently outperforms strong RAG baselines, especially in scenarios
with noisy or conflicting evidence.

## Full Text


<!-- PDF content starts -->

arXiv:2507.01281v1  [cs.CL]  2 Jul 2025Rethinking All Evidence: Enhancing Trustworthy Retrieval-Augmented
Generation via Conflict-Driven Summarization
Juan Chen1,2Baolong Bi2*†Wei Zhang3Jingyan Sui1,2
Xiaofei Zhu4Yuanzhuo Wang1,2Lingrui Mei2†Shenghua Liu2†
1University of Chinese Academy of Sciences
2Chinese Academy of Sciences
3National University of Defense Technology
4Chongqing University of Technology
Abstract
Retrieval-Augmented Generation (RAG) en-
hances large language models (LLMs) by in-
tegrating their parametric knowledge with ex-
ternal retrieved content. However, knowledge
conflicts caused by internal inconsistencies or
noisy retrieved content can severely undermine
the generation reliability of RAG systems. In
this work, we argue that LLMs should rethink
all evidence, including both retrieved content
and internal knowledge, before generating re-
sponses. We propose CARE-RAG (Conflict-
Aware and Reliable Evidence for RAG), a
novel framework that improves trustworthiness
through Conflict-Driven Summarization of all
available evidence. CARE-RAG first derives
parameter-aware evidence by comparing pa-
rameter records to identify diverse internal per-
spectives. It then refines retrieved evidences
to produce context-aware evidence, removing
irrelevant or misleading content. To detect and
summarize conflicts, we distill a 3B LLaMA3.2
model to perform conflict-driven summariza-
tion, enabling reliable synthesis across multiple
sources. To further ensure evaluation integrity,
we introduce a QA Repair step to correct out-
dated or ambiguous benchmark answers. Ex-
periments on revised QA datasets with retrieval
data show that CARE-RAG consistently out-
performs strong RAG baselines, especially in
scenarios with noisy or conflicting evidence.
1 Introduction
Retrieval-Augmented Generation (RAG) has
emerged as a powerful framework to equip large
language models (LLMs) (Achiam et al., 2023;
Grattafiori et al., 2024) with access to exter-
nal knowledge, enabling strong performance on
*Corresponding author.
†Authors from affiliation2are also affiliated with: Key Lab-
oratory of Network Data Science and Technology, ICT, CAS;
State Key Laboratory of AI Safety; University of Chinese
Academy of Sciences.
Who is the NBA’s top scorer?Parametric Evidence
Contextual Evidence[E3]Kareem Abdul-Jabbaris the top scorer in NBA history.[E4]Kareemrewriting scoring records.[E5] As of 2023, Jamesholds the record.Retrieval passages
[E1]Kareemdominated NBA scoring history. Is there anyother answer related besides Kareem Abdul-Jabbar?
LeBron is NBA’s all-time leading scorer.
[E2] Besides Kareem, I would sayLeBron Jamesis also a strong candidate.
How to resolve the conflicts in the above five evidence to get the right answer?Conflict!
Conflict!Conflict!Figure 1: LLMs struggle to assess the reliability of
evidence from different sources and to resolve conflicts
among them, challenging the trustworthiness of RAG.
knowledge-intensive tasks like question answer-
ing (Karpukhin et al., 2020; Guu et al., 2020; Gao
et al., 2023). While RAG effectively extends the
knowledge capacity of LLMs, its reliability in
real-world applications remains a significant con-
cern (Santhanam et al., 2021; Fan et al., 2024).
RAG enhances LLMs’ generation by leveraging
both internal and external knowledge, but as shown
in Figure 1, it also introduces unreliable sources
that make reasoning more difficult. First, due to
internal hallucinations (Huang et al., 2023; Ton-
moy et al., 2024), LLMs often generate multiple
inconsistent viewpoints for a given question. While
introducing new retrieval contexts aims to supple-
ment additional knowledge and alleviate these hal-
lucinations, many retrieved evidences contain er-
rors, noise, and even contradictions (Yoran et al.,
1

2023; Wang et al., 2023b). Moreover, the poten-
tial conflicts (Xu et al., 2024; Xie et al., 2023; Shi
et al., 2025) between the model’s internal param-
eter knowledge and the retrieved context further
challenge the RAG generation process, where mul-
tiple knowledge sources interact in a black-box
manner (Bi et al., 2024c; Mao et al., 2024).
To address these issues, we propose that LLMs
should rethink all evidence before generating re-
sponses in RAG framework, to clarify the relation-
ships between the internal knowledge and the re-
trieved context. In this work, we introduce CARE-
RAG (Conflict- Aware and Reliable Evidence for
RAG), a novel framework that enhances the trust-
worthiness of RAG by synthesizing all available
evidence based on conflict identification.
CARE-RAG first captures all evidence related
to the query, sourced from both the LLM’s internal
parameters and the retrieved documents. For the
LLM’s internal knowledge, we generate parameter-
aware evidence by comparing parameter records.
Specifically, we concatenate the model’s previous
generated parameter views and prompt the model to
generate new perspectives, different from the exist-
ing ones, thereby covering all possible viewpoints
to reduce internal hallucinations. For the retrieved
documents, we perform fine-grained refinement to
generate context-aware evidence, identifying and
removing irrelevant noise. This reduces the risk of
hallucinations caused by unrelated content, while
also saving token usage, allowing the model to con-
sider more context within token window limits and
enhancing robustness.
While CARE-RAG explicitly lists all available
evidence to ensure that as much relevant informa-
tion as possible is considered, this also introduces
more potential conflicts. To address this, we design
a knowledge summarization step based on conflict
detection, providing a final conflict report along-
side all the evidence to guide the LLM. Specifi-
cally, we distill the capabilities of DeepSeek-v3
into a smaller LLaMA 3.2-3B model, enabling it
to assess the conflict between two evidences and
provide related reasoning. The distilled model effi-
ciently cross-checks all evidence (both parameter-
aware and context-aware) to detect conflicts and
synthesize diverse knowledge perspectives. This
additional information helps the LLM generate a
reliable response based on all the input evidence.
We conduct experiments on five QA bench-
marks—Natural Questions, TriviaQA, HotpotQA,
ASQA, and WikiQA—covering both open-domainand multi-hop question answering. To improve su-
pervision quality and ensure fairer evaluation, we
introduce a lightweight answer-set augmentation
procedure that corrects outdated or semantically
inconsistent gold answers. This QA repair step
is applied once before training and used consis-
tently across all experiments. Results show that
this augmentation leads to substantial gains in both
EM and F1 across datasets. Compared to standard
RAG, CARE-RAG with augmentation improves
EM scores by up to 23.6% (e.g., from 40.3 to 63.9
on NQ with LLaMA-3.2-8B), and outperforms the
strongest existing baseline by an average of 3.8%
on EM. Further experiments confirm CARE-RAG’s
robustness to the number of evidence and validate
the effectiveness of each pipeline component, high-
lighting the importance of rethinking evidence in
enhancing the RAG process.
Our main contributions are as follows:
•We propose CARE-RAG, a novel framework
for enhancing the trustworthiness of RAG by
rethinking all available evidence via conflict-
driven summarization.
•We perform QA repair on multiple widely-
used QA datasets to ensure more accurate and
reliable evaluation for the community. In addi-
tion, we distill and release a conflict detection
model based on LLaMA-3.2–3B, capable of
analyzing and identifying potential conflicts
among input evidence.
•Experimental results show that CARE-RAG
significantly improves the ability of LLMs
to effectively integrate all available evidence,
achieving state-of-the-art performance on mul-
tiple RAG tasks and demonstrating the impor-
tance of rethinking evidence in RAG process.
2 CARE-RAG: Conflict-Aware and
Reliable Evidence for RAG
In this work, we propose CARE-RAG, a novel
framework designed to enhance the trustworthiness
of RAG systems. Unlike standard RAG that di-
rectly synthesize answers according to retrieved
evidence in black-box manner, CARE-RAG intro-
duces a four-stage framework that enables LLMs
to thoroughly rethink all available evidence—both
from parameter memory and retrieved context to
generation. As illustrated in Figure 2, CARE-RAG
2

Who is the NBA’ s top scorer?Input Question:Comparison of Parameter Records Refinement of  Retrieved EvidenceⅠⅡ
ⅣContext-aware passage:Kareem Abdul-Jabbar is the top scorer in NBA history.Conflict  Score: 1Reason for Conflict: Retrieved passages indicate Jabbar , while the LLM response states LeBron James ...Parameter-aware evidence:  …LeBron James surpassed Jabbar to become ….scorer.ⅢConflict-Driven SummarizationCARE-RAG Generation
Based on your previous parametric views, provide a different possible answer.Parametric-Aware Evidence
output 1
…
output 2
output n
Redundant tokensIrrelevant details
Who is the NBA’ s top scorer?Input Question:
Retrieved PassageKareem Abdul-Jabbar remains the NBA’s all-time top scorer with 38,387 points. Interestingly, he also appeared in several movies, including Airplane!.....Less TokensMore relevantContext-Aware Evidence…
Reliable Answer:After years of consistent scoring, LeBron James broke Kareem Abdul-Jabbar’s record. SoLeBron Jamesis now the NBA's top scorer.
non-conflict
conflictconflict
CARE-RAG
Parameter-aware passage
Context-aware passageConflict ReportConflict Report
Figure 2: An illustration of CARE-RAG rethinking all available evidence via conflict-driven summarization. The
framework consists of four stages: (I) Comparison of parameter Records licits and aggregates the model’s internal
diverse perspectives into parameter-aware evidence; (II) Refinement of Retrieved Evidence removes irrelevant
noise from raw retrieved content to produce concise, context-aware evidence; (III) Conflict-Driven Summarization
detects and analyzes conflicts between parameter-aware and context-aware evidence; (IV) CARE-RAG Generation
synthesizes a final answer by reconciling conflicts and integrating all information.
first derives parameter-aware evidence by compar-
ing parameter records, thereby eliciting diverse in-
ternal perspectives. It then refines the retrieved evi-
dence to obtain context-aware evidence by remov-
ing irrelevant or noisy content. Finally, a distilled
language model performs conflict-driven summa-
rization to generate reliable answers by aggregating
across multiple sources. This framework explicitly
separates the model’s parameter knowledge from
external context, and mitigates hallucinations by
resolving the complex conflicts between them. The
detailed inference procedure of our CARE-RAG is
presented in Algorithm 1.
2.1 Parameter Record Comparison
Given a query q, we first elicit the model’s
parameter-aware evidence Epwithout retrieved con-
text, aiming to establish its internal knowledge
baseline before external evidence is introduced (as
shown in Figure 2 Stage I). This involves:
a0← M (q; Πinit), (1a)
ai← M 
q,Ep; Πiter
, i= 1, . . . , n −1.
(1b)Here, iterative prompting (Eq. 1b) systematically
encourages the model to generate diverse internal
perspectives, explicitly aiming to reduce internal
hallucinations by capturing variability within its
parameter knowledge. We then define
Ep={a0, a1, . . . , a n−1},
which encapsulates the model’s parameter-aware
evidences, revealing potential internal inconsisten-
cies or uncertainties.
2.2 Retrieval Result Refinement
Concurrently, a retriever Rreturns evidences C=
{c1, . . . , c k}. To distill these into a concise context-
aware evidence Ecfocusing on salient information
(as illustrated in Figure 2 Stage II), we use:
Ec← M (q, C; Πref), (2)
where Πrefexplicitly instructs the model to extract
critical factual claims and eliminate irrelevant or
redundant content. This refinement enhances the
clarity and relevance of external evidence, facili-
tating subsequent conflict detection. In addition,
3

Algorithm 1 CARE-RAG Inference Procedure
Require: Query q; Retriever R; LLM M; Con-
flict detector Mc
Ensure: Final answer ˆa
1:Ep←[]
2:Π...defined as above
3:parameter Record Comparison
4:a0← M (q; Πinit)
5:Ep.append( a0)
6:fori= 1ton−1do
7: ai← M (q,Ep; Πiter)
8:Ep.append( ai)
9:end for
10:Ep←merge( Ep)
11:Retrieval Result Refinement
12:C← R(q)
13:Ec← M (q, C; Πref)
14:Conflict-Driven Summarization
15:(δc, rc)← M c(q,Ep,Ec; Πc)
16:ifδc= 1then
17: Ec←augment( Ec, rc)
18:end if
19:CARE-RAG Generation
20:ˆa← M (q,Ep,Ec, δc, rc; Πsynth)
21:return ˆa
the refinement also saves token usage, allowing the
model to consider more context within the token
window and enhancing robustness.
2.3 Conflict-Driven Summarization
Given the parameter-aware evidences Ep(internal
knowledge) and the refined evidence Ec(external
knowledge), we explicitly identify discrepancies
via a dedicated conflict detection module Mc(Fig-
ure 2 Stage III):
(δc, rc)← M c 
q,Ep,Ec; Πc
, δ c∈ {0,1},
(3)
where δc= 1indicates a conflict and rcprovides
the natural-language rationale, forming a detailed
"conflict report". Specifically, we construct a train-
ing dataset by annotating conflicts and their ra-
tionales using a teacher LLM (e.g., DeepSeek).
We then distill this knowledge into a smaller, effi-
cient LLaMA-3.2B model through supervised fine-
tuning, enabling rapid and accurate conflict detec-
tion during inference.
No Conflict ( δc= 0).When no conflict is de-
tected, the model primarily grounds its response inDataset RepairNoise ratio (%)
Mismatch Outdate
Wiki 67 0.0 100.0
TriviaQA 74 44.6 55.4
NQ 240 19.6 81.7
HotpotQA 103 8.7 91.3
ASQA 157 0.6 99.4
Table 1: Prevalence of outdated or mismatched ground
truths in standard QA benchmarks. Noise classification
is based on manual analysis and repair of 1,000 sampled
instances per dataset.
the refined external evidence Ec, while using the in-
ternal knowledge Epto provide additional support
and increase confidence in the answer.
Conflict Detected ( δc= 1).When a conflict is
identified, the model explicitly considers the ra-
tionale rc, critically evaluates both internal and
external evidence, and attempts to reconcile dis-
crepancies. If reconciliation is not possible, the
model is encouraged to transparently communicate
residual uncertainty.
2.4 CARE-RAG Generation.
The above steps produce a conflict report through
conflict-driven summarization, which effectively
helps LLMs mitigate hallucinations caused by con-
flicting evidence. Finally, CARE-RAG feeds the
parameter-aware evidence, context-aware evidence,
and the corresponding conflict report into the LLM,
enabling it to synthesize a final answer by reconcil-
ing conflicts and integrating all information. This
enhances the transparency of parametric knowl-
edge, factual accuracy, and robustness to conflict-
ing or ambiguous evidence in the generated output.
3 QA Repair for Valid Evaluation
Standard QA benchmarks often suffer from out-
dated or mismatched ground truths, which can lead
to inaccurate evaluations. Specifically, we conduct
a manual analysis of 1,000 randomly sampled in-
stances from each dataset and identify significant
annotation flaws, as shown in Table 1. For instance,
all 67 errors (100%) in the Wiki dataset were due
to outdated answers, while 44.6% of the 74 errors
in TriviaQA stemmed from semantic mismatches.
To address this issue, we introduce a QA Repair
pre-processing step to ensure fairer comparisons.
For instance, on TriviaQA, this approach raises the
F1 score from 85.09 to 86.17 for the Qwen3-235B-
4

A22B model, as shown in Table 2. Further imple-
mentation details are provided in Appendix B.
DatasetBaseline Mismatch Outdate Both
EM / F1 EM / F1 EM / F1 EM / F1
Wiki 54.8 / 55.4 54.8 / 55.4 56.7 / 57.4 56.7 / 57.4
TriviaQA 84.9 / 85.1 85.6 / 85.7 85.3 / 85.5 85.9 / 86.2
NQ 71.2 / 71.5 72.4 / 72.8 75.5 / 75.9 76.0 / 76.3
HotpotQA 63.1 / 63.6 63.9 / 64.3 66.8 / 67.3 67.1 / 67.5
ASQA 59.8 / 60.1 60.1 / 60.4 62.6 / 63.1 62.9 / 63.3
Table 2: QA performance improvements via QA Re-
pair across datasets."Baseline" shows original scores;
"Mismatch", "Outdate", and "Both" indicate results af-
ter fixing semantic mismatches, outdated answers, and
both, respectively. All values are reported as EM/F1.
4 Experimental Setup
4.1 Datasets
Our experimental evaluation utilizes five chal-
lenging QA benchmarks: Natural Questions
(NQ) (Kwiatkowski et al., 2019), TriviaQA (Joshi
et al., 2017), HotpotQA (Yang et al., 2018),
ASQA (Stelmakh et al., 2022), and 2WikiMulti-
HopQA (Zhang et al., 2023). To ensure fair eval-
uation, we apply our QA Repair procedure to all
five datasets, resulting in improved versions de-
noted as NQ∗,TriviaQA∗,HotpotQA∗,ASQA∗
andWikiQA∗, which are used consistently through-
out our experiments. This process addresses com-
mon issues such as outdated or mismatched, en-
hancing alignment between model predictions and
acceptable references.
4.2 Implementation Details
We evaluate CARE-RAG using both open-source
and closed-source LLMs. The open-source
models include Mistral-7B (Jiang et al., 2023),
LLaMA-3.2-8B (Grattafiori et al., 2024), and
Qwen2.5-7B (Yang et al., 2024). The closed-
source models include Claude-3.5-Haiku (An-
thropic, 2024), Gemini-2.0-Flash (Balestri, 2025),
and GPT-4.1-Nano (OpenAI, 2025). Experi-
ments use consistent hyperparameters across mod-
els (max_tokens=1024, temperature=0.7). Infer-
ence for open-source models is conducted using
VLLM (Kwon et al., 2023), while closed-source
models are accessed via official APIs.
We retrieve the top-5 most relevant evidences for
each query, with retrieval sensitivity analysis (vary-
ing top-K from 5 to 25) reported in Section 5.4
and Appendix A. Conflict Detection is poweredby a distilled LLaMA-3.2B model fine-tuned on
DeepSeek annotations, enabling efficient seman-
tic conflict analysis. parameter evidence ( Ep) is
generated via iterative prompting to elicit diverse
internal perspectives from the LLM. Context re-
finement is guided by instruction-based prompting,
with prompt templates detailed in Appendix C.
4.3 Baselines
We compare CARE-RAG with four representa-
tive baselines, covering key paradigms in retrieval-
augmented generation. No RAG uses only the
LLM’s parameter knowledge, without any retrieved
context, serving as a lower bound that reflects the
limitations of internal knowledge alone. Instruc-
tRAG (Wei et al., 2024) improves answer quality
by prompting the LLM with rationale-based in-
structions over retrieved evidences, but lacks mech-
anisms to handle contradictions across evidence.
GenRead (Yu et al., 2022) compresses retrieved
content into concise summaries before generation,
mitigating retrieval noise but potentially omitting
important conflicting signals. Self-RAG (Asai
et al., 2023) incorporates a self-reflection stage to
critique initial answers and refine retrieval, but does
not explicitly model conflicts between internal and
external knowledge. These baselines highlight the
challenges of retrieval quality, hallucination, and in-
consistency, which CARE-RAG addresses through
structured introspection and conflict resolution.
5 Results and Analysis
5.1 Overall Performance
We evaluate CARE-RAG on five QA benchmarks
(NQ*, TriviaQA*, HotpotQA*, ASQA*, Wik-
iQA*) under both open-source and closed-source
model settings, as shown in Tables 3 and 5. CARE-
RAG consistently achieves the highest EM and F1
scores across all datasets and models. Compared
to the standard RAG baseline, CARE-RAG im-
proves performance by up to 17.2 EM and 17.1 F1.
Relative to the strongest baseline method (Instruc-
tRAG), it still achieves an average improvement of
3.8 EM and 3.7 F1. Although closed-source models
generally exhibit higher absolute performance due
to larger scale and better pretraining, CARE-RAG
maintains consistent gains in both open-source and
closed-source settings, demonstrating its robust-
ness and general applicability.
These results indicate that the core mechanisms
of CARE-RAG—structured parameter introspec-
5

MethodNQ∗TriviaQA∗HotpotQA∗ASQA∗WikiQA∗
EM F1 EM F1 EM F1 EM F1 EM F1
Mistral-7B-v0.3
No RAG 39.7 41.6 65.2 66.8 35.8 38.5 32.3 34.6 33.2 36.9
RAG 41.4 42.7 66.0 67.2 34.7 36.4 32.2 34.2 35.9 37.7
InstructRAG 60.4 61.9 75.3 76.6 49.4 52.2 47.0 48.75 43.9 44.9
GenRead 48.9 49.3 70.7 71.0 38.6 39.3 37.8 38.3 37.7 38.5
Self-RAG 43.1 44.2 66.9 67.7 39.2 40.9 36.0 37.37 38.8 40.5
CARE-RAG 63.1 63.5 78.4 78.8 53.1 53.8 50.6 51.1 44.7 45.6
Llama-3.2-8B
No RAG 39.9 42.4 64.6 67.13 32.6 36.1 32.6 36.1 33.7 40.2
RAG 40.3 42.5 66.1 68.4 35.3 39.1 33.2 36.4 33.9 39.3
InstructRAG 59.7 60.9 73.9 75.1 48.5 50.5 45.9 47.2 36.9 40.6
GenRead 50.9 51.2 73.5 73.9 40.5 41.4 40.9 41.6 38.1 39.5
Self-RAG 40.8 42.5 68.3 70.2 36.9 39.9 34.2 36.9 34.2 39.1
CARE-RAG 63.9 64.3 79.6 79.9 55.9 56.6 52.6 53.1 47.1 48.0
Qwen2.5-7B
No RAG 28.2 31.0 51.2 53.1 31.2 34.5 17.9 21.5 31.3 37.0
RAG 31.0 32.8 52.9 54.4 30.5 32.9 18.9 21.7 30.6 32.1
InstructRAG 60.7 61.3 72.7 74.3 52.4 53.6 47.7 48.5 39.8 41.3
GenRead 39.5 39.9 59.2 59.6 34.1 34.8 24.3 25.0 31.5 32.4
Self-RAG 32.8 33.9 54.1 55.1 33.8 35.2 20.0 21.7 32.9 34.8
CARE-RAG 62.2 62.2 75.4 75.7 54.0 54.6 50.8 51.3 42.9 43.8
Table 3: Comparing Conflict-Aware and Reliable Evidence for RAG with open-source models on five QA bench-
marks (EM/F1 scores). CARE-RAG achieves superior performance across all datasets and models.
MethodNQ∗TriviaQA∗WikiQA∗.
EM F1 EM F1 EM F1
LLaMA-3.2-8B
w/o Stage1 61.5 62.8 77.3 72.61 43.3 44.7
w/o Stage2 39.9 42.44 64.6 67.13 33.7 40.17
w/o Stage3 60.3 60.74 78.12 77.93 44.1 45.29
CARE-RAG 63.9 64.31 79.6 79.89 47.1 48.0
Mistral-7B-v0.3
w/o Stage1 60.6 61.2 77.5 77.7 44.0 45.0
w/o Stage2 39.7 41.61 65.2 66.78 33.2 36.94
w/o Stage3 59.4 59.95 77.8 78.1 43.9 44.85
CARE-RAG 63.1 63.54 78.4 78.8 44.7 45.61
Table 4: Ablation study showing that each component
of CARE-RAG contributes to performance across NQ∗,
TriviaQA∗, and WikiQA∗datasets.
tion, evidence refinement, and conflict-aware sum-
marization—are highly effective in enhancing an-
swer reliability. By explicitly detecting and resolv-
ing contradictions between internal and retrieved
knowledge, CARE-RAG improves factual accu-
racy without relying on handcrafted prompts or
answer-level self-reflection. This architecture is
particularly beneficial in scenarios involving noisy
or conflicting evidence, where traditional RAG
methods tend to fail. The consistent improvements
across datasets and models support CARE-RAG’spotential as a general framework for trustworthy
retrieval-augmented generation.
5.2 Ablation Study of Core Components
To evaluate the effectiveness of CARE-RAG’s core
components, we perform an ablation study under
three settings. w/o Stage1 : Removes the parameter
Record Comparison stage and relies only on exter-
nal retrieved evidence for answer generation. w/o
Stage2 : Removes the Retrieval Result Refinement
module. w/o Stage3 : Removes the Conflict-Driven
Summarization stage, omitting both conflict.
As shown in Table 4, all three components con-
tribute significantly to the overall performance of
CARE-RAG across datasets and model backbones.
1) Introducing external retrieved evidence and re-
fining it into a structured Context-aware evidence
(Ec) leads to substantial gains over using param-
eter knowledge alone. For example, on the NQ
dataset with LLAMA-3-8B, adding refined ex-
ternal evidence yields a +20.4 EM improvement.
This highlights the importance of incorporating
external information in a structured and relevant
form via Πref; 2) Adding explicit conflict resolu-
tion—through conflict detection ( Mc) and conflict-
6

MethodNQ∗TriviaQA∗HotpotQA∗ASQA∗WikiQA∗.
EM F1 EM F1 EM F1 EM F1 EM F1
claude-3-5-haiku-latest
No RAG 50.6 52.3 77.8 78.9 42.1 44.8 40.9 43.3 35.5 39.2
RAG 51.9 53.0 78.7 79.1 43.3 44.7 41.0 42.9 35.5 37.9
InstructRAG 67.7 68.3 79.2 79.7 53.2 54.4 50.1 50.7 39.5 41.6
GenRead 57.0 57.5 80.1 80.4 43.9 44.7 46.3 47.0 35.4 36.5
Self-RAG 52.9 53.8 79.1 79.8 44.9 46.6 41.1 42.56 36.8 39.0
CARE-RAG 68.8 69.2 85.9 86.1 57.9 58.6 58.8 59.3 47.5 48.3
gemini-2.0-flash
No RAG 42.4 50.1 70.8 73.7 39.6 47.7 45.2 54.1 28.0 39.2
RAG 46.1 51.4 72.4 75.8 39.5 45.2 44.0 47.2 31.4 38.6
InstructRAG 65.3 66.7 75.1 76.5 49.1 50.9 46.9 48.6 41.2 44.7
GenRead 57.5 57.9 82.6 83.9 48.7 49.3 49.6 49.7 44.4 45.2
Self-RAG 49.4 52.4 77.5 78.7 39.4 41.8 42.6 45.3 34.5 38.0
CARE-RAG 68.0 68.5 86.7 87.1 61.4 62.3 63.6 64.2 56.7 57.7
gpt-4.1-nano-2025-04-14
No RAG 35.8 40.0 62.0 64.8 31.7 37.6 28.0 33.0 31.1 39.4
RAG 39.4 43.6 65.4 67.7 33.9 38.3 31.7 35.2 31.8 39.0
InstructRAG 58.5 60.5 72.5 73.6 53.5 56.24 48.1 50.08 40.4 44.5
GenRead 51.0 51.9 72.9 73.5 42.6 43.8 39.4 40.7 37.4 39.3
Self-RAG 43.7 46.0 68.1 69.6 35.9 39.3 34.2 37.5 32.2 38.2
CARE-RAG 66.2 66.5 81.6 81.2 56.7 57.2 53.0 53.4 47.6 48.2
Table 5: Comparing Conflict-Aware and Reliable Evidence for RAG with closed-source models on five QA
benchmarks (EM/F1 scores). CARE-RAG achieves superior performance across all datasets and models.
aware answer synthesis ( Πsynth)—provides consis-
tent additional gains of 1–2 EM/F1. This shows the
value of not only using external knowledge but also
explicitly identifying and reconciling inconsisten-
cies between the internal parameter knowledge ( Ep)
and the retrieved evidence ( Ec). Such targeted con-
flict handling is crucial for ensuring factual consis-
tency and generating trustworthy answers, leading
to the full performance of CARE-RAG.
5.3 Sensitivity to Retrieval Volume
To assess the robustness of our method under vary-
ing retrieval volumes, we conduct an ablation study
onK, the number of retrieved evidences. We eval-
uate CARE-RAG under different retrieval volumes,
varying Kfrom 5 to 25. The results are presented
in Figure 3.
The findings indicate that CARE-RAG effec-
tively utilizes increased context, benefiting particu-
larly from its context refinement mechanism Πref.
Performance generally peaks around K= 15 –20,
beyond which it plateaus, showing remarkable sta-
bility even when potentially lower-quality or redun-
dant evidence is included. This contrasts with sim-
pler RAG methods, which often suffer from noiseaccumulation at higher Kvalues. These results
suggest that CARE-RAG’s structured reasoning
and conflict resolution mechanisms are effective at
filtering and prioritizing information, thereby main-
taining performance even under noisy conditions.
(a) NQ∗/EM
 (b) NQ∗/F1
(c) HotpotQA∗/EM
 (d) HotpotQA∗/F1
Figure 3: Sensitivity to retrieval size ( K). EM/F1 scores
for NQ and HotpotQA across three open-source models.
5.4 Robustness to Retrieval Variations
A robust RAG system must remain effective under
imperfect retrieval conditions, where the provided
7

Figure 4: EM performance across three datasets using
different retrieval evidence sources.
evidence may vary significantly in relevance, com-
pleteness, or even contradict the original query in-
tent. Figure 4 illustrates EM scores across three
datasets under four different evidence strategies:
contextual only, parameter only, their direct com-
bination, and CARE-RAG. CARE-RAG consis-
tently outperforms both contextual and parameter-
only baselines, achieving gains of up to 0.239 EM.
These trends highlight CARE-RAG’s superior ro-
bustness to variations in evidence quality and com-
position, especially in scenarios with conflicting or
incomplete information.
This robustness stems from CARE-RAG’s
conflict-aware synthesis process: contextual ev-
idence Ec, retrieved and refined through Πref, is
systematically compared with parameter-derived
knowledge Epusing the conflict detector Mc. This
enables the model to identify and suppress mis-
leading or contradictory signals, prioritize reliable
content, and ultimately produce more accurate and
trustworthy answers even in noisy or adversarial
retrieval settings.
6 Related Work
RAG aims to enhance Large Language Mod-
els (LLMs) by incorporating external knowl-
edge (Lewis et al., 2020; Guu et al., 2020). Early
work and pretraining objectives focused on effec-
tive retrieval integration (Izacard et al., 2023). How-
ever, RAG’s reliability remains challenged by the
quality of retrieved information and the model’s
ability to integrate it with internal knowledge. Im-
proving the retriever itself is an active research
area, with utility-based methods like shared con-
text attribution enhancing relevance and usefulness
of evidences (Xu et al., 2025).
Beyond basic retrieval and integration, several
methods aim to improve RAG systems. For in-
stance, REPLUG (Shi et al., 2023) explored black-
box retrieval integration, while RA-DIT (Lin et al.,2023) and InstructRetro (Wang et al., 2023a) in-
vestigated instruction tuning for better downstream
task alignment. RankRAG (Yu et al., 2024) im-
proved passage ranking. Yet, these approaches
may not fully resolve challenges from conflicting
or low-quality retrievals, or the critical issue of
maintaining generation faithfulness—addressed by
works like (Bi et al., 2024a; Zhang et al., 2025a),
which align LLMs for context-faithful outputs. Our
proposed Conflict-Aware and Reliable Evidence for
RAG (CARE-RAG) explicitly targets post-retrieval
synthesis to improve robustness under such scenar-
ios.
A core challenge in RAG lies in resolving knowl-
edge conflicts—when retrieved content contradicts
either the LLM’s prior or other evidences (Wang
et al., 2023a; Zhou et al., 2025; Zou et al., 2024;
Jin et al., 2024; Xie et al., 2023; Bi et al., 2025).
These conflicts can cause factual inaccuracies and
intersect with the domain of knowledge editing in
LLMs, where methods seek to correct or bias in-
ternal representations (Li et al., 2025; Zhang et al.,
2025b) or reinforce edited knowledge through con-
trastive decoding (Bi et al., 2024b). Such con-
cerns resonate with findings on data noise sensi-
tivity (Jiang et al., 2024; Chen et al., 2024), and
underscore the need for conflict resolution within
RAG pipelines. This is especially relevant in ad-
versarial or noisy scenarios, including graph-based
settings where even single-node attacks can distort
outcomes (Tao et al., 2021). Ensuring fair eval-
uation thus requires accounting for both retrieval
quality and model alignment (Jacovi et al., 2023).
7 Conclusion
CARE-RAG is a conflict-aware and reliable frame-
work for retrieval-augmented question answering
that systematically tackles key reliability chal-
lenges, including outdated supervision, noisy or
conflicting retrievals, and inconsistencies between
internal and external knowledge. By integrating
structured parameter introspection, fine-grained
context refinement, lightweight conflict detection,
and a QA repair mechanism, CARE-RAG enhances
factual consistency and robustness across diverse
QA tasks. Extensive experiments on five bench-
marks and multiple model backbones show that
CARE-RAG consistently outperforms strong base-
lines, underscoring the value of explicitly modeling
knowledge conflicts for trustworthy and generaliz-
able retrieval-augmented generation.
8

Limitations
CARE-RAG demonstrates notable improvements
over existing retrieval-augmented methods; how-
ever, certain limitations remain. The multi-stage
approach inherently incurs greater computational
overhead compared to simpler RAG frameworks,
potentially impacting inference efficiency. Addi-
tionally, the performance of CARE-RAG, partic-
ularly its conflict detection and resolution capa-
bilities, remains closely tied to the quality of the
underlying language models and their fine-tuned
capabilities, which might not fully resolve highly
subtle or adversarially constructed knowledge con-
flicts. Furthermore, despite increased robustness to
noisy retrieval, overall efficacy still depends sub-
stantially on the initial document retriever’s accu-
racy and comprehensiveness. The QA Repair mod-
ule, though effective against typical dataset issues,
may not universally handle all types of benchmark
artifacts or specialized domain knowledge without
further refinement and domain-specific adaptation.
Ethical Considerations
The development of advanced retrieval-augmented
generation systems, including CARE-RAG, raises
significant ethical considerations. The QA Repair
process, designed to address dataset biases by cor-
recting outdated or mismatched information, inher-
ently involves subjective judgments regarding the
definition and scope of "correctness." Such judg-
ments must be transparently managed and period-
ically revisited to prevent inadvertent bias intro-
duction. Additionally, improvements in factual
accuracy and consistency, although broadly benefi-
cial, increase the risk of generating convincing yet
inaccurate information if misused or inadequately
supervised. Reliance on externally retrieved knowl-
edge also introduces the possibility of propagating
existing biases or inaccuracies from source mate-
rials. Therefore, ongoing research efforts should
emphasize robust bias detection, clear attribution of
information sources, transparent conflict-resolution
mechanisms, and the establishment of responsible
use guidelines to ensure these powerful tools are
deployed ethically, fairly, and constructively.
References
Josh Achiam, Steven Adler, Sandhini Agarwal, Lama
Ahmad, Ilge Akkaya, Florencia Leoni Aleman,
Diogo Almeida, Janko Altenschmidt, Sam Altman,Shyamal Anadkat, and 1 others. 2023. Gpt-4 techni-
cal report. arXiv preprint arXiv:2303.08774 .
Anthropic. 2024. Model card addendum: Claude 3.5
haiku and upgraded claude 3.5 sonnet. Technical
report, Anthropic.
Akari Asai, Zeqiu Wu, Yizhong Wang, Avirup Sil, and
Hannaneh Hajishirzi. 2023. Self-rag: Learning to
retrieve, generate, and critique through self-reflection.
InThe Twelfth International Conference on Learning
Representations .
Roberto Balestri. 2025. Gender and content bias in
large language models: a case study on google
gemini 2.0 flash experimental. arXiv preprint
arXiv:2503.16534 .
Baolong Bi, Shaohan Huang, Yiwei Wang, Tianchi
Yang, Zihan Zhang, Haizhen Huang, Lingrui Mei,
Junfeng Fang, Zehao Li, Furu Wei, and 1 oth-
ers. 2024a. Context-dpo: Aligning language
models for context-faithfulness. arXiv preprint
arXiv:2412.15280 .
Baolong Bi, Shenghua Liu, Lingrui Mei, Yiwei Wang,
Pengliang Ji, and Xueqi Cheng. 2024b. Decoding by
contrasting knowledge: Enhancing llms’ confidence
on edited facts. arXiv preprint arXiv:2405.11613 .
Baolong Bi, Shenghua Liu, Yiwei Wang, Lingrui Mei,
Junfeng Fang, Hongcheng Gao, Shiyu Ni, and Xueqi
Cheng. 2024c. Is factuality enhancement a free lunch
for llms? better factuality can lead to worse context-
faithfulness. arXiv preprint arXiv:2404.00216 .
Baolong Bi, Shenghua Liu, Yiwei Wang, Yilong Xu,
Junfeng Fang, Lingrui Mei, and Xueqi Cheng. 2025.
Parameters vs. context: Fine-grained control of
knowledge reliance in language models. arXiv
preprint arXiv:2503.15888 .
Jiawei Chen, Hongyu Lin, Xianpei Han, and Le Sun.
2024. Benchmarking large language models in
retrieval-augmented generation. In Proceedings of
the AAAI Conference on Artificial Intelligence , pages
17754–17762.
Wenqi Fan, Yujuan Ding, Liangbo Ning, Shijie Wang,
Hengyun Li, Dawei Yin, Tat-Seng Chua, and Qing
Li. 2024. A survey on rag meeting llms: Towards
retrieval-augmented large language models. In Pro-
ceedings of the 30th ACM SIGKDD Conference on
Knowledge Discovery and Data Mining , pages 6491–
6501.
Yunfan Gao, Yun Xiong, Xinyu Gao, Kangxiang Jia, Jin-
liu Pan, Yuxi Bi, Yi Dai, Jiawei Sun, Haofen Wang,
and Haofen Wang. 2023. Retrieval-augmented gen-
eration for large language models: A survey. arXiv
preprint arXiv:2312.10997 , 2.
Aaron Grattafiori, Abhimanyu Dubey, Abhinav Jauhri,
Abhinav Pandey, Abhishek Kadian, Ahmad Al-
Dahle, Aiesha Letman, Akhil Mathur, Alan Schelten,
Alex Vaughan, and 1 others. 2024. The llama 3 herd
of models. arXiv preprint arXiv:2407.21783 .
9

Kelvin Guu, Kenton Lee, Zora Tung, Panupong Pasu-
pat, and Mingwei Chang. 2020. Retrieval augmented
language model pre-training. In International confer-
ence on machine learning , pages 3929–3938.
Lei Huang, Weijiang Yu, Weitao Ma, Weihong Zhong,
Zhangyin Feng, Haotian Wang, Qianglong Chen,
Weihua Peng, Xiaocheng Feng, Bing Qin, and Ting
Liu. 2023. A survey on hallucination in large lan-
guage models: Principles, taxonomy, challenges, and
open questions. Preprint , arXiv:2311.05232.
Gautier Izacard, Patrick Lewis, Maria Lomeli, Lucas
Hosseini, Fabio Petroni, Timo Schick, Jane Dwivedi-
Yu, Armand Joulin, Sebastian Riedel, and Edouard
Grave. 2023. Atlas: Few-shot learning with retrieval
augmented language models. Journal of Machine
Learning Research , 24(251):1–43.
Alon Jacovi, Avi Caciularu, Omer Goldman, and Yoav
Goldberg. 2023. Stop uploading test data in plain
text: Practical strategies for mitigating data contam-
ination by evaluation benchmarks. arXiv preprint
arXiv:2305.10160 .
Albert Q. Jiang, Alexandre Sablayrolles, Arthur Men-
sch, Chris Bamford, Devendra Singh Chaplot, Diego
de las Casas, Florian Bressand, Gianna Lengyel, Guil-
laume Lample, Lucile Saulnier, Lélio Renard Lavaud,
Marie-Anne Lachaux, Pierre Stock, Teven Le Scao,
Thibaut Lavril, Thomas Wang, Timothée Lacroix,
and William El Sayed. 2023. Mistral 7b. Preprint ,
arXiv:2310.06825.
Zhengbao Jiang, Zhiqing Sun, Weijia Shi, Pedro Ro-
driguez, Chunting Zhou, Graham Neubig, Xi Vic-
toria Lin, Wen-tau Yih, and Srinivasan Iyer. 2024.
Instruction-tuned language models are better knowl-
edge learners. arXiv preprint arXiv:2402.12847 .
Zhuoran Jin, Pengfei Cao, Yubo Chen, Kang Liu, Xiao-
jian Jiang, Jiexin Xu, Qiuxia Li, and Jun Zhao. 2024.
Tug-of-war between knowledge: Exploring and re-
solving knowledge conflicts in retrieval-augmented
language models. arXiv preprint arXiv:2402.14409 .
Mandar Joshi, Eunsol Choi, Daniel S Weld, and Luke
Zettlemoyer. 2017. Triviaqa: A large scale distantly
supervised challenge dataset for reading comprehen-
sion. arXiv preprint arXiv:1705.03551 .
Vladimir Karpukhin, Barlas Oguz, Sewon Min,
Patrick SH Lewis, Ledell Wu, Sergey Edunov, Danqi
Chen, and Wen-tau Yih. 2020. Dense passage re-
trieval for open-domain question answering. In
EMNLP (1) , pages 6769–6781.
Tom Kwiatkowski, Jennimaria Palomaki, Olivia Red-
field, Michael Collins, Ankur Parikh, Chris Alberti,
Danielle Epstein, Illia Polosukhin, Jacob Devlin, Ken-
ton Lee, and 1 others. 2019. Natural questions: a
benchmark for question answering research. Trans-
actions of the Association for Computational Linguis-
tics, 7:453–466.Woosuk Kwon, Zhuohan Li, Siyuan Zhuang, Ying
Sheng, Lianmin Zheng, Cody Hao Yu, Joseph Gon-
zalez, Hao Zhang, and Ion Stoica. 2023. Efficient
memory management for large language model serv-
ing with pagedattention. In Proceedings of the 29th
Symposium on Operating Systems Principles , pages
611–626.
Patrick Lewis, Ethan Perez, Aleksandra Piktus, Fabio
Petroni, Vladimir Karpukhin, Naman Goyal, Hein-
rich Küttler, Mike Lewis, Wen-tau Yih, Tim Rock-
täschel, and 1 others. 2020. Retrieval-augmented gen-
eration for knowledge-intensive nlp tasks. Advances
in neural information processing systems , 33:9459–
9474.
Zherui Li, Houcheng Jiang, Hao Chen, Baolong Bi,
Zhenhong Zhou, Fei Sun, Junfeng Fang, and Xiang
Wang. 2025. Reinforced lifelong editing for language
models. arXiv preprint arXiv:2502.05759 .
Xi Victoria Lin, Xilun Chen, Mingda Chen, Weijia Shi,
Maria Lomeli, Richard James, Pedro Rodriguez, Ja-
cob Kahn, Gergely Szilvasy, Mike Lewis, and 1 oth-
ers. 2023. Ra-dit: Retrieval-augmented dual instruc-
tion tuning. In The Twelfth International Conference
on Learning Representations .
Yuren Mao, Xuemei Dong, Wenyi Xu, Yunjun Gao, Bin
Wei, and Ying Zhang. 2024. Fit-rag: black-box rag
with factual information and token reduction. arXiv
preprint arXiv:2403.14374 .
OpenAI. 2025. Gpt-4.1 nano model card.
Keshav Santhanam, Omar Khattab, Jon Saad-Falcon,
Christopher Potts, and Matei Zaharia. 2021. Col-
bertv2: Effective and efficient retrieval via
lightweight late interaction. arXiv preprint
arXiv:2112.01488 .
Dan Shi, Renren Jin, Tianhao Shen, Weilong Dong,
Xinwei Wu, and Deyi Xiong. 2025. Ircan: Mitigating
knowledge conflicts in llm generation via identifying
and reweighting context-aware neurons. Advances
in Neural Information Processing Systems , 37:4997–
5024.
Weijia Shi, Sewon Min, Michihiro Yasunaga, Min-
joon Seo, Rich James, Mike Lewis, Luke Zettle-
moyer, and Wen-tau Yih. 2023. Replug: Retrieval-
augmented black-box language models. arXiv
preprint arXiv:2301.12652 .
Ivan Stelmakh, Yi Luan, Bhuwan Dhingra, and
Ming-Wei Chang. 2022. Asqa: Factoid ques-
tions meet long-form answers. arXiv preprint
arXiv:2204.06092 .
Shuchang Tao, Qi Cao, Huawei Shen, Junjie Huang,
Yunfan Wu, and Xueqi Cheng. 2021. Single node in-
jection attack against graph neural networks. CoRR ,
abs/2108.13049.
10

S. M Towhidul Islam Tonmoy, S M Mehedi Zaman,
Vinija Jain, Anku Rani, Vipula Rawte, Aman Chadha,
and Amitava Das. 2024. A comprehensive survey of
hallucination mitigation techniques in large language
models. Preprint , arXiv:2401.01313.
Boxin Wang, Wei Ping, Lawrence McAfee, Peng
Xu, Bo Li, Mohammad Shoeybi, and Bryan Catan-
zaro. 2023a. Instructretro: Instruction tuning post
retrieval-augmented pretraining. arXiv preprint
arXiv:2310.07713 .
Cunxiang Wang, Xiaoze Liu, Yuanhao Yue, Xiangru
Tang, Tianhang Zhang, Cheng Jiayang, Yunzhi Yao,
Wenyang Gao, Xuming Hu, Zehan Qi, and 1 oth-
ers. 2023b. Survey on factuality in large language
models: Knowledge, retrieval and domain-specificity.
arXiv preprint arXiv:2310.07521 .
Zhepei Wei, Wei-Lin Chen, and Yu Meng. 2024. In-
structrag: Instructing retrieval-augmented genera-
tion via self-synthesized rationales. arXiv preprint
arXiv:2406.13629 .
Jian Xie, Kai Zhang, Jiangjie Chen, Renze Lou, and
Yu Su. 2023. Adaptive chameleon or stubborn sloth:
Revealing the behavior of large language models in
knowledge conflicts. In The Twelfth International
Conference on Learning Representations .
Rongwu Xu, Zehan Qi, Zhijiang Guo, Cunxiang Wang,
Hongru Wang, Yue Zhang, and Wei Xu. 2024.
Knowledge conflicts for llms: A survey. arXiv
preprint arXiv:2403.08319 .
Yilong Xu, Jinhua Gao, Xiaoming Yu, Yuanhai Xue,
Baolong Bi, Huawei Shen, and Xueqi Cheng. 2025.
Training a utility-based retriever through shared con-
text attribution for retrieval-augmented language
models. arXiv preprint arXiv:2504.00573 .
An Yang, Baosong Yang, Beichen Zhang, Binyuan Hui,
Bo Zheng, Bowen Yu, Chengyuan Li, Dayiheng Liu,
Fei Huang, Haoran Wei, and 1 others. 2024. Qwen2.
5 technical report. arXiv preprint arXiv:2412.15115 .
Zhilin Yang, Peng Qi, Saizheng Zhang, Yoshua Ben-
gio, William W Cohen, Ruslan Salakhutdinov, and
Christopher D Manning. 2018. Hotpotqa: A dataset
for diverse, explainable multi-hop question answer-
ing. arXiv preprint arXiv:1809.09600 .
Ori Yoran, Tomer Wolfson, Ori Ram, and Jonathan
Berant. 2023. Making retrieval-augmented language
models robust to irrelevant context. arXiv preprint
arXiv:2310.01558 .
Wenhao Yu, Dan Iter, Shuohang Wang, Yichong
Xu, Mingxuan Ju, Soumya Sanyal, Chenguang
Zhu, Michael Zeng, and Meng Jiang. 2022. Gen-
erate rather than retrieve: Large language mod-
els are strong context generators. arXiv preprint
arXiv:2209.10063 .Yue Yu, Wei Ping, Zihan Liu, Boxin Wang, Jiaxuan You,
Chao Zhang, Mohammad Shoeybi, and Bryan Catan-
zaro. 2024. Rankrag: Unifying context ranking with
retrieval-augmented generation in llms. Advances in
Neural Information Processing Systems , 37:121156–
121184.
Jiahao Zhang, Haiyang Zhang, Dongmei Zhang, Yong
Liu, and Shen Huang. 2023. End-to-end beam re-
trieval for multi-hop question answering. arXiv
preprint arXiv:2308.08973 .
Qinggang Zhang, Zhishang Xiang, Yilin Xiao, Le Wang,
Junhui Li, Xinrun Wang, and Jinsong Su. 2025a.
Faithfulrag: Fact-level conflict modeling for context-
faithful retrieval-augmented generation. arXiv
preprint arXiv:2506.08938 .
Tianyu Zhang, Junfeng Fang, Houcheng Jiang, Baolong
Bi, Xiang Wang, and Xiangnan He. 2025b. Explain-
able and efficient editing for large language models.
InTHE WEB CONFERENCE 2025 .
Huichi Zhou, Kin-Hei Lee, Zhonghao Zhan, Yue Chen,
Zhenhao Li, Zhaoyang Wang, Hamed Haddadi, and
Emine Yilmaz. 2025. Trustrag: Enhancing robust-
ness and trustworthiness in rag. arXiv preprint
arXiv:2501.00879 .
Wei Zou, Runpeng Geng, Binghui Wang, and Jinyuan
Jia. 2024. Poisonedrag: Knowledge corruption at-
tacks to retrieval-augmented generation of large lan-
guage models. arXiv preprint arXiv:2402.07867 .
11

A Implementation Details
A.1 Models Used
Our experiments primarily utilized three open-
source Large Language Models (LLMs): Mistral-
7B (Jiang et al., 2023), Llama-3.2-8B (Grattafiori
et al., 2024), and Qwen2.5-7B (Yang et al., 2024).
For experiments involving closed-source models
(as detailed in Table 5), we employed.
Unless otherwise specified, the same backbone
LLM (from either the open-source or closed-source
set, depending on the experiment) was consistently
used for all stages of the CARE-RAG pipeline: elic-
iting the initial LLM response ( Ainit, corresponding
toEpgeneration in Algorithm 1), refining retrieved
results into Ec(which may involve structured rea-
soning), conflict detection by Mcto produce δcand
rc, CARE-RAG to generate ˆa, and also for the QA
Repair module ( frepair) described in Appendix B.
A.2 Retrieval Setup
For each question, we retrieved the top-K relevant
evidences from the respective corpus. In our main
experiments (Tables 3 and 5), K was set to 5. An
analysis of CARE-RAG’s sensitivity to varying K
(from 5 to 25) is presented in Figure 3.
A.3 Inference Framework
All inferences for open-source LLMs were per-
formed using the vLLM framework (Kwon et al.,
2023) to ensure efficiency and reproducibility. For
closed-source models, inferences were made via
their respective official APIs. The following in-
ference parameters were consistently applied for
generation tasks (e.g., generating initial responses
forEp, the refined evidence Ec, the final answer ˆa,
and repaired answers in frepair) unless a specific
module (like the conflict detector Mc) required
different settings:
•max_tokens : 1024
•temperature : 0.7
•top_p : 1.0
For classification-like tasks performed by the con-
flict detector Mcto determine δc(and generate
rc), we typically used a temperature (e.g., 0.7, or
potentially lower like 0.0 for more deterministic
conflict/no-conflict output if desired) to encourage
more deterministic outputs, though the primary
mechanism for binary classification was specificinstruction prompting tailored to elicit a "0" or "1"
and a rationale.
A.4 Evaluation Metrics
System performance was primarily evaluated using
standard Exact Match (EM) and F1 scores. These
metrics were computed against the (potentially)
repaired ground truth answers generated by our QA
Repair module (detailed in Appendix B), ensuring
a fair and robust assessment across all compared
methods.
B QA Repair Module
B.1 Overview
As highlighted in our experimental setup (Sec-
tion 4), standard QA benchmarks often suffer from
issues such as temporal drift (outdated answers)
or semantic mismatches between questions and
ground truths. These flaws can lead to misleading
evaluations of RAG systems. To ensure a fairer and
more accurate assessment of model capabilities, we
introduce a QA Repair module. This module is ap-
plied as a pre-processing step to the test instances
of all evaluated benchmarks, correcting potential is-
sues in the original ground truth answers before any
model evaluation takes place. The module operates
on an input triplet: ( question q,original ground
truth answer agt, and potentially relevant retrieved
context C, though Cis not always strictly neces-
sary for the repair logic if general world knowledge
suffices).
B.2 Repair Mechanism
The core of the QA Repair module is a classifier,
frepair, implemented using a prompted Large Lan-
guage Model (LLM). This classifier is tasked with
assessing whether the original ground truth answer,
agt, is likely outdated, semantically inconsistent
with the question q, or otherwise flawed, consider-
ing current world knowledge and the precise intent
ofq. It outputs a binary flag:
γrepair=frepair(q, a gt, Coptional )∈ {0,1}
Ifγrepair= 1(indicating a detected flaw), a repair
process is initiated. This process, also typically
leveraging a prompted LLM, employs structured
reasoning or direct knowledge querying (based on
qand potentially C) to generate a revised, more ac-
curate ground truth answer, a′
gt. In some instances,
to resolve ambiguity or align with the corrected
12

answer, the original question qmight also be min-
imally refined to q′. The output of this stage is
thus a potentially corrected question-answer pair
(q′, a′
gt). This repaired pair is then used as the ref-
erence for evaluating all RAG models (including
baselines and CARE-RAG) in our experiments.
B.3 Illustrative Examples
The following examples illustrate typical scenarios
handled by the QA Repair module. Note that in
these examples, "Current Model Answer" (if such
a term was used previously, otherwise this clari-
fication might not be needed) is re-interpreted as
the "Repaired Ground Truth ( a′
gt)" produced by our
QA Repair module if a flaw was detected in the
original "Ground Truth ( agt)".
B.3.1 Example 1: Temporal Drift
Scenario: Temporal Drift
Original Query ( q): Who scored the most points in
their NBA career?
Original Ground Truth ( agt): Kareem Abdul-Jabbar
QA Repair Module Output :
•Detection ( γrepair= 1): The answer "Kareem
Abdul-Jabbar" is outdated.
•Repaired Ground Truth ( a′
gt): LeBron James
(as of [current date/year of dataset repair])
•Repaired Query ( q′): (No change in this case)
Who scored the most points in their NBA ca-
reer?
B.3.2 Example 2: Answer Type Mismatch /
Factual Inaccuracy
Scenario: Answer Type Mismatch / Factual Inaccu-
racy
Original Query ( q): When was the Statue of Liberty
in France built?
Original Ground Truth ( agt): Paris
QA Repair Module Output :
•Detection ( γrepair = 1): The answer "Paris"
does not answer "When" and is factually incor-
rect for the construction date.
•Repaired Ground Truth ( a′
gt): Construction
was completed in July 1884. (Or simply: July
1884)
•Repaired Query ( q′): (No change in this case)
When was the Statue of Liberty in France built?
B.3.3 Detailed Analysis of Repaired Data
Figure 5 details the error composition within cor-
rected samples from five QA benchmarks (1,000samples each were analyzed for repair needs). The
chart displays the counts of "Mismatch" errors
(semantic misalignment) and "Out-of-date" errors
(temporal drift) among the instances that required
repair. For example, all 67 repaired Wiki samples
were out-of-date, while TriviaQA’s 74 repairs in-
cluded approximately 33 mismatches. Notably, the
NQ dataset, with 240 repaired samples, exhibits
an overlap in error types: the sum of its reported
mismatch (approx. 47) and out-of-date (approx.
196) components exceeds the total repair count, in-
dicating some samples possess both error attributes.
This granular analysis, highlighting diverse error
profiles and potential co-occurrences as in NQ, un-
derscores the necessity of our comprehensive QA
Repair process for establishing a reliable evalua-
tion baseline and the importance of targeted, rather
than one-size-fits-all, approaches to dataset noise.
Figure 5: Mismatch and Out-of-date error distribution in
repaired samples from five QA datasets. NQ shows co-
occurring error types. Repair impact on Qwen2.5-7B
(using the notation from your paper if Qwen3-235B-
A22B is a specific variant) is detailed in Table 2 (Please
verify this table label and its content regarding repair
impact specifically with this model version).
C Core Conflict-Aware and Reliable
Evidence for RAG Prompts
This section provides the specific prompt formats
used for the core stages of the Conflict-Aware and
Reliable Evidence for RAG (CARE-RAG) frame-
work, corresponding to the Πsymbols in Algo-
rithm 1.
C.1 Parameter Record Comparison Prompts
(ΠinitandΠiter)
C.1.1 Iterative parameter Response Prompt
(Πiter)
Objective: Elicit alternative or more diverse pa-
rameter responses ai(i >0), given the previously
13

generated parameter responses within Ep, to fur-
ther explore the model’s internal knowledge space.
Task : Based on your previous answer(s) and your inter-
nal knowledge, provide a different or more detailed/nu-
anced answer to the following question.
Question : {question}
Previous parameter Answer(s) ( Epso far) : {previ-
ous_parameter_answers}
Answer (Iterative - ai):
C.1.2 Initial parameter Response Prompt
(Πinit)
Objective: Elicit the model’s first direct response a0
based solely on its internal parameter knowledge,
forming the basis of Ep.
Task : Provide a concise and direct answer to the follow-
ing question using only your internal knowledge.
Question : {question}
Answer (Initial - a0):
C.2 Retrieval Result Refinement Prompt
(Πref)
Objective: Instruct the model to distill the retrieved
evidences Cinto a concise and salient context-
aware evidence Ec, by extracting key factual ob-
servations, identifying ambiguities, and forming
context-grounded conclusions. This corresponds to
Stage II in Figure 2.
Context Refinement Prompt Instruction : Analyze the
provided Context thoroughly in relation to the Question.
Your goal is to extract the most relevant factual informa-
tion, identify any ambiguities or limitations within the
context, and conclude with the most likely answer(s) or
key insights that can be *purely grounded in the provided
Context*. If no complete answer is available from the
context, state that and explain why. Retrieved Context
evidences ( C):
• {context_evidence_1}
• {context_evidence_2}
• ...
• {context_evidence_k}
Question ( q):{question} Your Distilled Context-
Aware evidence ( Ec) based *only* on the Retrieved
Context should include :
• Key factual claims relevant to the Question.
•Identified ambiguities or limitations in the provided
Context.
•A concluding summary or answer candidate(s)
strictly derived from the Context.C.3 Conflict Detection Prompt ( Πc)
Objective: Explicitly evaluate whether the model’s
consolidated parameter-aware evidences ( Ep) se-
mantically conflict with the refined Context-aware
evidence ( Ec). This is used by the conflict detec-
tor module Mcand corresponds to Stage III in
Figure 2.
Conflict Detection Prompt
Instruction : Evaluate if the "parameter Knowledge Re-
sponse" contradicts the "Context-derived Response" for
the given Question. Consider factual differences (e.g.,
names, dates, values), temporal mismatches, or signifi-
cant semantic inconsistencies. Output ’Conflict: 1’ if a
contradiction is found. Output ’Conflict: 0’ if there is
no contradiction or if they are consistent. Provide a brief
step-by-step reasoning for your decision.
Question ( q):{question}
parameter Knowledge Response (Consolidated from
Ep):{consolidated_parameter_response}
Context-derived Response (from Ec): {con-
text_aware_evidence_summary}
Analysis and Conflict Decision ( δc, rc):
14

C.4 CARE-RAG Generation Prompt ( Πsynth)
Objective: Generate the final answer ( ˆa) by inte-
grating the parameter-aware evidences ( Ep), the
refined Context-aware evidence ( Ec), and the con-
flict detection signal ( δc, rc). This corresponds to
Stage IV in Figure 2.
Final Answer Synthesis Prompt Contextual Note : A
potential conflict (indicated by δc) between internal pa-
rameter knowledge ( Ep) and external information ( Ec)
might have been detected, with rationale rc.Your Task
is to Synthesize the Best Final Answer ( ˆa):
1.Based on all inputs, identify the best-supported
single candidate answer .
2.Consider information recency, source reliability, and
overall coherence, especially if a conflict ( δc= 1)
was detected.
3.If conflict ( δc= 1): Explicitly address the discrep-
ancy from rc. Attempt to resolve it by selecting more
credible information or state remaining uncertainty.
4.If no conflict ( δc= 0): Primarily ground your an-
swer in Ec, using Epas confirmation.
5.Provide concise reasoning for your chosen answer,
citing relevant inputs ( Ep,Ec, rc). Clearly state any
remaining ambiguity or temporal uncertainty.
Inputs Provided :
•Question ( q): {question}
•parameter Knowledge Response (Consolidated
Ep): {consolidated_parameter_response}
•Context-derived Response ( Ec): {con-
text_aware_evidence_summary}
•Conflict Detection Flag ( δc): {δc}
•Conflict Rationale ( rc): {rc}
Required Output Format for Final Answer ( ˆa):
•Final Answer : ...
•Reasoning for Final Answer : ... (Address conflict
perrcifδc= 1)
•Ambiguity/Uncertainty Assessment : ... (If any)
D Detailed Process Walkthrough
We illustrate the complete CARE-RAG workflow
using the NBA scoring example, as discussed in
Figure 2 (Stage I-IV visual overview) and refer-
enced in the main text.
1.Input Question ( q):"Who scored the most
points in their NBA career?"
2.parameter Record Comparison (generates
Ep): The LLM M, using prompt Πinit(Ap-pendix C.1), generates its initial context-free re-
sponse a0. For this example, we assume n= 1,
so the consolidated parameter-aware evidences
Epis:"LeBron James" (assuming the LLM’s
parameter knowledge is up-to-date).
3.Retrieval Result Refinement (generates Ec):
The retriever Rreturns evidences C, e.g.: c1:
"Kareem Abdul-Jabbar is the all-time leading
scorer in the NBA, with 38,387 total points." ;
c2:"Kareem rewriting scoring records." ;c3:
"As of 2023, James holds the record."
Using prompt Πref(Appendix C.2), Mpro-
cesses Cinto the Context-aware evidence Ec.
For example, Ecmight be distilled to: "Re-
trieved evidences state Kareem Abdul-Jabbar
was the all-time leading scorer (38,387 points).
One passage indicates that as of 2023, James
holds the record, suggesting a change."
4.Conflict-Driven Summarization (generates
δc, rc): The conflict detector Mc, using prompt
Πc(Appendix C.3), compares Ep("LeBron
James" ) with Ec("Retrieved evidences state
Kareem... James holds the record..." ).
Assuming for clearer conflict demonstration
thatEcwas distilled to only reflect outdated
info like: "According to retrieved text, Kareem
Abdul-Jabbar is the top scorer."
The outputs are: Conflict Flag ( δc):1. Con-
flict Rationale ( rc):"parameter knowledge ( Ep)
states LeBron James, while context-derived in-
formation ( Ec) states Kareem Abdul-Jabbar.
These conflict."
5.CARE-RAG Generation (generates ˆa): The
LLMM, using prompt Πsynth (Appendix C.4),
receives q,Ep,Ec,δc= 1, and rc. The Final
Answer ( ˆa)is, for example: "LeBron James is
NBA’s all-time leading scorer. While some his-
torical records mention Kareem Abdul-Jabbar,
LeBron James has surpassed this record, align-
ing with current information."
The Reasoning would acknowledge the con-
flict identified by rcand explain the prioriti-
zation of current parameter knowledge ( Ep) or
the more recent parts of Ec, treating Kareem’s
record as historical.
E Component Output Examples
This section provides additional, isolated exam-
ples of outputs from key components and stages
15

of the Conflict-Aware and Reliable Evidence for
RAG (CARE-RAG) framework. These examples
illustrate the specific outputs for Context-aware ev-
idence Generation (formerly Structured Reason-
ing), Conflict Detection, and CARE-RAG Gen-
eration. Examples for QA Repair (Appendix B)
and parameter-aware evidence Generation ( Ep, de-
tailed in Appendix C.1) are covered elsewhere or
are straightforward.
E.1★Conflict Detection Output Example
(δc, rcfromMc)
Task : Evaluate whether the consolidated parameter-
aware evidences ( Ep) contradict the refined Context-
aware evidence ( Ec) for the given query. Output a conflict
flag (δc∈ {0,1}) and a rationale ( rc). This uses prompt
Πc(Appendix C.3).
Query ( q):Who was "Suite: Judy Blue Eyes" written
about?
Input: Consolidated parameter-aware evidences ( Ep)
(simulated) :
•Stephen Stills wrote it about Judy Collins, his former
girlfriend.
Input: Refined Context-aware evidence ( Ec) (simu-
lated) :
•The song "Suite: Judy Blue Eyes" was written by
Stephen Stills. It references Judy Collins and their
relationship.
LLM Reasoning for Conflict Detection (part of rc):
•Both inputs identify Stephen Stills as the author and
Judy Collins as the subject.
•The information provided is consistent and comple-
mentary, with no factual contradictions.
Conflict Detection Output :
•Conflict Flag ( δc):0
•Conflict Rationale ( rc):No conflict detected. Both
parameter knowledge and context-derived informa-
tion consistently identify Stephen Stills as the author
and Judy Collins as the subject of the song.E.2✌Context-aware evidence Generation
Example ( Ecfrom Πref)
Task : Distill the retrieved context ( C) into a concise and
salient Context-aware evidence ( Ec). This uses prompt
Πref(Appendix C.2).
Input Question ( q):Who was Henry Feilden?
Input Retrieved Context ( C):
•"Colonel Henry Wemyss Feilden CB (1838–1921) was
a British Army officer, naturalist and Arctic explorer."
•"Feilden was the second son of the second Baronet
Feilden..."
•"Henry Master Feilden (1818–1875) was a Conserva-
tive politician..."
•"Some sources incorrectly identify Henry Feilden as
an actor."
Generated Context-aware evidence ( Ec):
•Key Factual Claims :
–Two distinct individuals named Henry Feilden are
mentioned:
*Colonel Henry Wemyss Feilden (1838–1921):
British Army officer, naturalist, Arctic explorer.
*Henry Master Feilden (1818–1875): Conserva-
tive politician.
–The claim that a Henry Feilden was an actor is
explicitly refuted for at least one, and implicitly for
both if the context is considered complete regarding
their professions.
•Identified Ambiguities/Issues : The query "Who was
Henry Feilden?" is ambiguous without further specifi-
cation, as the context provides two distinct individuals.
•Context-Grounded Conclusion/Summary : The re-
trieved context describes two notable individuals
named Henry Feilden. To provide a specific answer to
"Who was Henry Feilden?", clarification on which in-
dividual is being referred to would be necessary. The
claim about Henry Feilden being an actor is noted as
incorrect based on the context.
16

E.3▲CARE-RAG Generation Output Ex-
ample ( ˆafrom Πsynth)
Task : Generate the final reconciled answer ( ˆa) by inte-
grating all available information. This uses prompt Πsynth
(Appendix C.4).
Input Question ( q): Who is the original artist of Sound
of Silence , released in 1964?
Input: Consolidated parameter-aware evidences ( Ep)
(simulated) :Simon & Garfunkel
Input: Refined Context-aware evidence ( Ec) (simu-
lated) :Based on the provided context, Simon & Gar-
funkel are identified as the original artists of "Sound of
Silence" released in 1964. No conflicting information
found in context.
Input: Conflict Flag ( δc):0Input: Conflict Rationale
(rc):No conflict detected between parameter knowledge
and context-derived information.
CARE-RAG Generation Output ( ˆa):
•Final Answer : Simon & Garfunkel were the original
artists of the song "Sound of Silence," released in
1964.
•Reasoning for Final Answer : Both the consolidated
parameter knowledge ( Ep) and the refined context-
aware evidence ( Ec) consistently identify Simon &
Garfunkel. The conflict flag ( δc= 0) confirms no dis-
crepancy was found. There is no ambiguity regarding
the 1964 release.
•Ambiguity/Uncertainty Assessment : None detected.
17