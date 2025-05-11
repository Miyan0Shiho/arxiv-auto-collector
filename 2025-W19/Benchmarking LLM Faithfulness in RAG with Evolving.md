# Benchmarking LLM Faithfulness in RAG with Evolving Leaderboards

**Authors**: Manveer Singh Tamber, Forrest Sheng Bao, Chenyu Xu, Ge Luo, Suleman Kazi, Minseok Bae, Miaoran Li, Ofer Mendelevitch, Renyi Qu, Jimmy Lin

**Published**: 2025-05-07 22:50:33

**PDF URL**: [http://arxiv.org/pdf/2505.04847v1](http://arxiv.org/pdf/2505.04847v1)

## Abstract
Hallucinations remain a persistent challenge for LLMs. RAG aims to reduce
hallucinations by grounding responses in contexts. However, even when provided
context, LLMs still frequently introduce unsupported information or
contradictions. This paper presents our efforts to measure LLM hallucinations
with a focus on summarization tasks, assessing how often various LLMs introduce
hallucinations when summarizing documents. We discuss Vectara's existing LLM
hallucination leaderboard, based on the Hughes Hallucination Evaluation Model
(HHEM). While HHEM and Vectara's Hallucination Leaderboard have garnered great
research interest, we examine challenges faced by HHEM and current
hallucination detection methods by analyzing the effectiveness of these methods
on existing hallucination datasets. To address these limitations, we propose
FaithJudge, an LLM-as-a-judge approach guided by few-shot human hallucination
annotations, which substantially improves automated LLM hallucination
evaluation over current methods. We introduce an enhanced hallucination
leaderboard centered on FaithJudge, alongside our current hallucination
leaderboard, enabling more reliable benchmarking of LLMs for hallucinations in
RAG.

## Full Text


<!-- PDF content starts -->

Benchmarking LLM Faithfulness in RAG with Evolving Leaderboards
Manveer Singh Tamber1∗, Forrest Sheng Bao2, Chenyu Xu2,3, Ge Luo2, Suleman Kazi2,
Minseok Bae4∗, Miaoran Li3∗, Ofer Mendelevitch2, Renyi Qu2, Jimmy Lin1
1University of Waterloo2Vectara3Iowa State University4Stanford University
Correspondence: mtamber@uwaterloo.ca ,{forrest, suleman}@vectara.com
Abstract
Hallucinations remain a persistent challenge
for LLMs. RAG aims to reduce hallucinations
by grounding responses in contexts. However,
even when provided context, LLMs still fre-
quently introduce unsupported information or
contradictions. This paper presents our efforts
to measure LLM hallucinations with a focus
on summarization tasks, assessing how often
various LLMs introduce hallucinations when
summarizing documents. We discuss Vectara’s
existing LLM hallucination leaderboard, based
on the Hughes Hallucination Evaluation Model
(HHEM). While HHEM and Vectara’s Hallu-
cination Leaderboard have garnered great re-
search interest, we examine challenges faced
by HHEM and current hallucination detection
methods by analyzing the effectiveness of these
methods on existing hallucination datasets. To
address these limitations, we propose Faith-
Judge, an LLM-as-a-judge approach guided
by few-shot human hallucination annotations,
which substantially improves automated LLM
hallucination evaluation over current methods.
We introduce an enhanced hallucination leader-
board centered on FaithJudge1, alongside our
current hallucination leaderboard2, enabling
more reliable benchmarking of LLMs for hal-
lucinations in RAG.
1 Introduction
LLMs excel in various tasks, but frequently pro-
duce hallucinations, generating false or mislead-
ing information unsupported by provided con-
texts or world knowledge (Ji et al., 2023; Huang
et al., 2025; Lin et al., 2022; Tang et al., 2023).
While Retrieval-Augmented Generation (RAG) ap-
proaches (Guu et al., 2020; Lewis et al., 2020b;
Shuster et al., 2021) attempt to mitigate hallucina-
tions by grounding responses in external trusted
*Work done while at Vectara
1https://github.com/vectara/FaithJudge
2https://github.com/vectara/
hallucination-leaderboardcontexts, they do not fully eliminate hallucinations,
as LLMs often introduce details unsupported by
retrieved contexts, misrepresent information, or
generate outright contradictions (Niu et al., 2024).
An ongoing challenge within RAG is ensuring
context-faithfulness (Niu et al., 2024; Jia et al.,
2023; Ming et al., 2024). Detecting when LLMs
deviate from the information in the provided con-
text remains a difficult problem. Although there
has been progress, hallucination detection meth-
ods, including fine-tuned detectors largely for eval-
uating summaries (Zhou et al., 2021; Gekhman
et al., 2023; Honovich et al., 2022; Zha et al.,
2023; Tang et al., 2024a) and LLM-as-a-judge tech-
niques (Zheng et al., 2023; Luo et al., 2023; Jacovi
et al., 2025), continue to struggle with the accurate
identification of LLM-generated hallucinations.
In this paper, we study and aim to improve hal-
lucination evaluation in RAG by building on prior
work in summary consistency evaluation. We ana-
lyze the capabilities and limitations of current hal-
lucination detection methods, including fine-tuned
models such as Vectara’s Hughes Hallucination
Evaluation Model (HHEM) (Bao et al., 2024) and
zero-shot methods using LLM judges.
To overcome the challenges of LLM-as-a-judge
techniques for zero-shot hallucination detection
and fine-tuned hallucination detection models, we
introduce FaithJudge, an LLM-as-a-judge approach
guided by few-shot human annotations of hallucina-
tions. FaithJudge leverages labelled hallucinations
from diverse LLM generations to automate the eval-
uation of LLMs on their propensity to hallucinate
when summarizing the same articles or using the
same articles to respond to queries. This approach
results in notably higher agreement with human
judgments compared with existing automated meth-
ods. Additionally, we introduce an enhanced hallu-
cination leaderboard based on FaithJudge, enabling
more reliable benchmarking of hallucinations in
LLM-generated summaries and responses.arXiv:2505.04847v1  [cs.CL]  7 May 2025

We discuss both Vectara’s existing hallucina-
tion leaderboard (Hughes and Bae, 2023) based
on HHEM and our new leaderboard based on Faith-
Judge. Hallucinations within RAG remain frequent
and problematic, even in leading LLMs. Our ap-
proach contributes toward more accurate halluci-
nation evaluation, aiding the development of more
trustworthy generative AI systems.
Vectara serves customers across diverse indus-
tries. For many of these customers, addressing
hallucinations in LLM outputs is a critical priority.
Driven by this customer-centric need, we devel-
oped our hallucination leaderboard and benchmark-
ing methods, which are presented in this paper.
2 Background
Accurate hallucination detection is essential for
reliably quantifying hallucination rates in LLMs.
Numerous datasets have been developed for evalu-
ating hallucinations in summarization tasks. Earlier
datasets, such as SummaC (Laban et al., 2022) and
AggreFact (Tang et al., 2023), aggregated multiple
resources, standardized labels, and classification
taxonomies. However, these primarily focused on
summaries from pre-ChatGPT models like fine-
tuned T5 (Raffel et al., 2020), BART (Lewis et al.,
2020a), and PEGASUS (Zhang et al., 2020) mod-
els, potentially limiting their relevance to contem-
porary LLMs that may produce more nuanced and
difficult to identify hallucinations.
Recent benchmarks address this limitation by
incorporating summaries generated by modern
LLMs. TofuEval (Tang et al., 2024b) provided hal-
lucination labels on topic-focused dialogue summa-
rization tasks with LLMs including GPT-3.5-Turbo,
Vicuna (Chiang et al., 2023) and WizardLM (Xu
et al., 2023). Similarly, HaluEval (Li et al., 2023)
included ChatGPT-generated hallucinations across
summarization, question answering (QA), and di-
alogue tasks, while RAGTruth (Niu et al., 2024)
also annotated responses from models including
GPT-3.5, GPT-4 (OpenAI, 2023), Llama-2 (Tou-
vron et al., 2023), and Mistral (Jiang et al., 2023).
FaithBench (Bao et al., 2025) presents human anno-
tations of challenging hallucinations in summaries
from 10 modern LLMs from 8 different model fam-
ilies (detailed further in Section 4).
Due to limited large-scale, human-annotated
data for training hallucination detectors, early de-
tection methods relied heavily on natural language
inference (NLI) or question-answering (QA) sys-tems (Fabbri et al., 2022). For instance, Sum-
maC aggregated sentence-level NLI entailment
scores between document-summary sentence pairs.
AlignScore (Zha et al., 2023) extended this by
training detection models on multiple semantic
alignment tasks evaluated at the chunk level.
MiniCheck (Tang et al., 2024a) addressed data
scarcity by synthesizing hallucinated examples us-
ing GPT-4 for model training.
Modern LLMs’ strong zero-shot instruction-
following capabilities have also enabled LLM-as-
a-judge methods (Zheng et al., 2023; Luo et al.,
2023; Jacovi et al., 2025; Gao et al., 2023). In-
stead of evaluating entire generated summaries, ap-
proaches like FACTSCORE (Min et al., 2023) and
RAGAS (Es et al., 2024) decompose summaries
into claims for granular hallucination detection.
Like Vectara’s Hallucination Leader-
board (Hughes and Bae, 2023), other efforts
like FACTS Grounding (Jacovi et al., 2025) and
Galileo’s Hallucination Index (Galileo, 2023) also
provide leaderboards to benchmark hallucinations
in LLMs. Galileo’s Hallucination Index employs
GPT-4o as a single LLM judge, whereas FACTS
Grounding ensembles evaluations from three
different LLM judges: GPT-4o, Claude-3.5-Sonnet,
and Gemini-1.5-Pro.
Nonetheless, hallucination detection remains
challenging, with modest effectiveness observed
across current methods. Benchmarks such as Ag-
greFact, RAGTruth, TofuEval, and FaithBench
consistently show limitations in existing halluci-
nation detectors, including LLM-based methods.
Notably, FaithBench highlighted that even the
best current models achieve near 50% accuracy.
Both RAGTruth and TofuEval further suggest that
smaller, fine-tuned detection models can perform
competitively with or even outperform LLM-based
evaluation approaches.
3 Vectara’s Hallucination Leaderboard
In 2023, Vectara’s Hallucination Leader-
board (Hughes and Bae, 2023) was released
using Vectara’s hallucination-detection model,
HHEM-1.0-open. This model was later updated
to HHEM-2.0 with stronger effectiveness, the
ability to handle longer contexts, and multilingual
capabilities. The current leaderboard relies on the
open version, HHEM-2.1-open, publicly released

AggreFact-SOTA RAGTruth-Summ TofuEval-MB FaithBench Average
Method # Params Acc (%) F1 (%) Acc (%) F1 (%) Acc (%) F1 (%) Acc (%) F1 (%) Acc (%) F1 (%)
Claim-wise
Fine-Tuned Hallucination Detection Models
HHEM-1.0-Open 184M 76.0 71.0 66.2 52.2 54.4 49.9 59.3 58.8 64.0 58.0
HHEM-2.1-Open 110M 73.2 69.7 67.7 56.1 60.9 61.2 66.7* 63.7* 67.1 62.7
AlignScore-base 125M 69.5 61.9 60.2 42.4 51.7 44.0 60.6 59.9 60.5 52.1
AlignScore-large 355M 73.9 69.3 67.9 54.1 56.1 52.9 62.8 59.6 65.2 59.0
MiniCheck-Roberta-L 355M 75.7 72.5 70.5 58.6 67.6 68.5 61.6 60.0 68.8 64.9
Bespoke-MiniCheck 7B 74.3 70.1 73.3 62.9 76.9 78.4 60.1 58.0 71.2 67.3
TrueTeacher 11B 71.8 70.5 56.5 56.1 58.5 58.4 59.8* 51.8* 61.7 59.2
Zero-Shot Hallucination Detection with LLMs
RAGAS Prompt
Qwen-2.5 7B 71.1 69.0 68.2 64.4 64.3 57.7 57.9 51.3 65.4 60.6
Qwen-2.5 72B 74.4 69.9 75.3 64.1 69.2 70.6 64.3 57.3 70.8 65.5
Llama-3.1 8B 69.7 65.9 68.5 59.9 72.6 74.4 60.3 57.9 67.8 64.5
Llama-3.3 70B 77.3 74.9 80.0 75.1 73.2 70.6 58.9 49.6 72.3 67.5
GPT-4o ? 75.9 70.3 75.7 63.7 75.2 76.7 65.3 59.0 73.0 67.4
o3-mini-high ? 77.3 72.6 74.6 62.9 69.2 70.6 67.4 60.7 72.1 66.7
Summary-wise
Fine-Tuned Hallucination Detection Models
HHEM-1.0-Open 184M 78.9 79.7 53.4 51.4 56.5 39.8 50.5 40.1 59.8 52.7
HHEM-2.1-Open 110M 76.6 76.2 64.4 67.1 69.4 62.1 52.6* 32.9* 65.8 59.6
AlignScore-base 125M 73.8 73.9 57.6 58.2 65.6 52.8 51.3 33.8 62.1 54.7
AlignScore-large 355M 72.7 74.2 52.8 49.6 57.4 39.2 50.3 26.1 58.3 47.3
MiniCheck-Roberta-L 355M 74.2 72.1 66.3 60.9 54.4 45.4 55.0 53.2 62.5 57.9
Bespoke-MiniCheck 7B 79.9 80.4 79.4 77.1 78.8 78.6 55.7 47.3 73.5 70.8
TrueTeacher 11B 77.6 78.4 61.6 62.8 57.4 39.2 53.3* 36.7* 62.5 54.3
Zero-Shot Hallucination Detection with LLMs
FACTS Grounding Prompt
Qwen-2.5 7B 66.9 68.7 61.5 63.4 62.8 54.4 52.6 33.5 60.9 55.0
Qwen-2.5 72B 71.6 73.7 74.0 77.5 68.8 58.8 55.2 35.5 67.4 61.4
Llama-3.1 8B 55.5 55.5 62.9 62.7 55.3 54.5 60.9 49.7 58.6 55.6
Llama-3.3 70B 79.3 78.1 81.6 74.9 70.1 71.3 66.6 58.4 74.4 70.7
GPT-4o ? 81.6 78.8 82.6 76.6 76.3 76.0 65.9 56.2 76.6 71.9
o3-mini-high ? 82.1 77.8 79.8 70.6 69.2 70.6 68.8 60.7 75.0 69.9
Luo et al. Prompt
Qwen-2.5 7B 72.8 73.5 67.6 70.2 69.0 66.3 53.4 39.0 65.7 62.2
Qwen-2.5 72B 78.4 78.0 81.3 81.1 83.4 80.0 58.3 44.3 75.3 70.8
Llama-3.1 8B 60.8 51.2 63.7 52.1 57.1 55.8 51.3 51.0 58.2 52.5
Llama-3.3 70B 79.2 79.1 81.3 82.9 73.6 66.5 58.8 43.6 73.2 68.0
GPT-4o ? 80.4 77.5 85.1 80.9 81.6 78.7 62.5* 50.6* 77.4 71.9
o3-mini-high ? 82.6 80.9 83.2 80.6 75.6 73.7 63.3 49.8 76.2 71.2
Table 1: Balanced Accuracy and F1-Macro of hallucination detection methods across four datasets. The final two columns
report the simple average across the four datasets. We note that certain models marked with an asterisk (*) were used to select
articles for the challenging FaithBench dataset.
on HuggingFace3. To date, HHEM has been
downloaded over 3.5 million times, reflecting
strong community interest and adoption. While
specific training details remain confidential, we
note that HHEM-2.1-open was trained using the
RAGTruth training set among other datasets.
To build Vectara’s Hallucination Leaderboard,
articles were selected from diverse sources such as
BBC News, CNN, Wikipedia, and the Daily Mail,
following prior work on summarization evaluation
and factuality verification (Narayan et al., 2018;
Maynez et al., 2020; Schuster et al., 2021; Thorne
et al., 2018; Fabbri et al., 2021; Huang et al., 2020;
Pagnoni et al., 2021; Hermann et al., 2015). Arti-
cles containing objectionable or explicit content,
which LLMs may refuse to summarize, were specif-
ically excluded. The resulting dataset comprised
articles with a median length of approximately 217
3https://huggingface.co/vectara/
hallucination_evaluation_modelwords (25th percentile: 42 words; 75th percentile:
424 words).
LLMs are evaluated by prompting them to gen-
erate concise summaries strictly grounded in the
provided passages. HHEM then assesses the pro-
portion of summaries generated by the LLM con-
taining hallucinations. Refusals are tracked by mea-
suring the proportion of short responses (5 words
or fewer). Users are also invited to submit specific
models for evaluation. Continuously updated, the
leaderboard now benchmarks hallucination rates of
over 130 different LLMs, typically evaluating new
models as soon as they become publicly available
to track ongoing advances in the field.
4 FaithBench
FaithBench (Bao et al., 2025) examined hallucina-
tions in LLM-generated summaries and assessed
the effectiveness of hallucination detection meth-
ods through human annotations. It includes sum-

FaithBench
Method # Params Acc (%) F1 (%)
FaithJudge Prompting
Qwen-2.5 7B 71.9 66.6
Qwen-2.5 72B 73.2 73.0
Llama-3.1 8B 60.8 61.0
Llama-3.3 70B 77.5 77.8
GPT-4o ? 79.5 81.1
o3-mini-high ? 84.0 82.1
Majority Vote (Qwen 72B, Llama 70B, GPT-4o) 80.7 81.3
Majority Vote (Qwen 72B, Llama 70B, GPT-4o, o3) 80.2 81.3
Table 2: Balanced Accuracy and F1-Macro scores for Faith-
Judge on FaithBench using different LLM judges. With Ma-
jority Vote , we break ties by defaulting to a classification of
inconsistent.
maries from ten state-of-the-art LLMs, including
GPT-4o, GPT-3.5, Claude-3.5-Sonnet, Gemini-1.5-
Flash (Gemini Team, 2024), and open-source mod-
els like Llama-3.1 (Grattafiori et al., 2024), reveal-
ing that hallucinations remain frequent and detec-
tion methods often fail to identify them accurately.
Human annotators labelled hallucinations as Un-
wanted when the summary contained contradictory
or unsupported information, Benign when the in-
formation was supported by world knowledge, but
absent from the article, or Questionable when the
classification was unclear.
Articles in FaithBench were selected from Vec-
tara’s Hallucination Leaderboard based on frequent
disagreements on summaries among hallucination
detection models. True-NLI, TrueTeacher, HHEM-
2.1-open, and GPT-4o/GPT-3.5 judges using the
CoT prompt from Luo et al. (2023) were used to
identify articles where summary hallucination clas-
sifications were most disagreed upon. The dataset
includes 75 articles, each with ten annotated sum-
maries from different LLMs.
5 FaithJudge
Human annotation is the gold standard for hallu-
cination detection, but it is time-consuming and
expensive. FaithJudge offers a scalable alternative
by leveraging hallucination annotations to guide an
LLM judge in evaluating new summaries. We also
expand FaithJudge to other RAG tasks, including
question-answering (QA) and writing overviews
from structured data in the JSON format using the
RAGTruth dataset (Niu et al., 2024). This is de-
tailed further in the Appendix A.
To assess a summary, FaithJudge involves
prompting an LLM judge with other summaries
of the same article, along with their correspond-
ing hallucination annotations. These annotations
include hallucination spans, source references, andlabels of either Benign, Unwanted, or Questionable,
identified by human annotators.
To evaluate the effectiveness of FaithJudge, we
use the fact that each FaithBench article has sum-
maries from ten different LLMs. The judge is given
the other nine annotated summaries as context, and
its assessments on each summary from FaithBench
are compared to human annotations. As shown in
Section 6, FaithJudge substantially improves au-
tomated hallucination evaluation, outperforming
existing detection methods by leveraging human-
labelled examples. This allows for more accurate
automated hallucination evaluation, where existing
hallucination detection methods continue to lag.
6 Evaluating Hallucination Detectors
6.1 Evaluation Datasets
We evaluate leading hallucination detection meth-
ods on four datasets: FaithBench, AggreFact (Tang
et al., 2023), RAGTruth (Niu et al., 2024), and
TofuEval-MeetingBank (Tang et al., 2024b). While
each of these datasets has previously analyzed
hallucination detection individually, we provide
a broader comparison across all four, motivating
the need for our FaithJudge approach.
For FaithBench, we assign each summary the
most severe hallucination label given by a majority
of the annotators. We evaluate using summaries
labelled either Unwanted orConsistent , excluding
Benign and Questionable cases due to their more
ambiguous nature. This slightly differs from the
original FaithBench evaluation, which pooled the
worst label across all annotators for each summary
and combined Benign cases with Consistent ones,
while combining Unwanted cases with Question-
able ones for the binary classification problem.
For AggreFact, we evaluate on the SOTA sub-
set of summaries, which involves annotated sum-
maries generated by fine-tuned T5 (Raffel et al.,
2020), BART (Lewis et al., 2020a), and PEGA-
SUS (Zhang et al., 2020) models. For RAGTruth,
we evaluate only on the annotated summaries sub-
set. Lastly, for TofuEval-Meetingbank, we evaluate
on summaries generated using articles from the
MeetingBank dataset (Hu et al., 2023).
6.2 Existing Hallucination Detectors
Table 1 compares the effectiveness of fine-tuned
hallucination detectors and zero-shot LLM-based
methods across various datasets. We evaluate
Vectara’s HHEM models alongside AlignScore,

Figure 1: Proportion of summary FaithBench labels (left) and FaithJudge predictions (right) across models. For
FaithBench labels, red indicates Unwanted, orange indicates Questionable, yellow indicates Benign, while green
indicates consistent. For FaithJudge predictions, red indicates hallucinated, and green indicates consistent summaries.
Each bar shows the proportion of summaries falling into each category.
MiniCheck, including Bespoke-MiniCheck (Be-
spoke, 2024), and TrueTeacher. We also in-
clude current LLMs, such as GPT-4o and o3-mini
(high reasoning), as well as open-source mod-
els Qwen2.5 (7B and 72B), Llama-3.1 (8B), and
Llama-3.3 (70B). The o3-mini model, in particular,
excels in reasoning tasks.
Classification methods are separated into claim-
wise and summary-wise classification. Claim-wise
evaluation involves decomposing sentences from
summaries into individual claims using Llama-
3.3 (70B) and a similar prompt from Tang et al.
(2024a), while summary-wise methods assess the
entire summary at once.
For LLM-based detection, we test three prompts:
(1) the RAGAS prompt, which verifies lists of
claims, (2) the FACTS Grounding JSON prompt,
shown to be the most effective of the prompts tested
in Jacovi et al. (2025) for GPT-4o, and (3) the CoT-
based prompt from Luo et al. (2023). We modify
prompts slightly as needed for clearer final outputs
and to specifically evaluate summaries.
Table 1 shows that, similar to findings in pre-
vious work, hallucination detection remains chal-
lenging. We note that certain models marked with
an asterisk (*) were used to select articles for the
adversarially challenging FaithBench dataset. Con-
sequently, these models, including HHEM, may
perform worse on FaithBench in summary-wise
classification than they otherwise would. Zero-shot
classification using GPT-4o and o3-mini-high tends
to perform best, both using summary-wise classi-
fication with either the FACTS Grounding JSONBinary Classification
Gold Truth Consistent Inconsistent
Unwanted 74 322
Questionable 29 38
Benign 50 34
Consistent 176 27
Ternary Classification
Gold Truth Consistent Benign Unwanted
Unwanted 84 18 294
Questionable 28 13 26
Benign 51 10 23
Consistent 179 4 20
Table 3: Confusion matrices for FaithJudge prompted
for classification on FaithBench summaries.
prompt or the Luo et al. (2023) prompt. However,
their average effectiveness remains modest, with
balanced accuracy below 78% and F1-macro be-
low 72%. Considering FaithBench, the highest
balanced accuracy is achieved by o3-mini-high at
68.8% while the highest F1-macro of 63.7% is
achieved by the HHEM model when considering
claim-wise classification.
The table illustrates improved effectiveness with
increased model size: larger open-source models
generally outperform smaller ones, and GPT-4o
and o3-mini-high achieve the highest overall effec-
tiveness. However, although HHEM-2.1-open is
the smallest model tested, it performs strongly, out-
performing several larger models. Among the fine-
tuned models, only the 7B-parameter MiniCheck
achieves higher average scores for summary-wise
classification, while both MiniCheck variants out-
perform it in claim-wise classification.

Figure 2: Sensitivity and specificity with FaithJudge as
the number of examples in the prompt are increased.
We place an asterisk (*) next to the 10 because, in this
case, FaithJudge is shown annotations for the summary
it is evaluating.
Overall, fine-tuned models can score stronger
than smaller prompted LLMs, but the largest LLMs
typically yield the best results, even while being
zero-shot methods. Regardless, the examined meth-
ods demonstrate modest effectiveness in general,
with particularly weak effectiveness on FaithBench,
which captures a diverse set of LLM summaries
but is designed to be challenging for hallucination
detection models.
6.3 FaithJudge
Table 2 presents the effectiveness of FaithJudge
on FaithBench using various LLMs. The highest
effectiveness is achieved using the o3-mini-high
judge, reaching a balanced accuracy of 84% and an
F1-macro of 82.1%, allowing for higher agreement
with human annotation on FaithBench than the ex-
isting methods discussed in Table 1. Although the
effectiveness of FaithJudge is not perfect, this may
be partly explained by disagreements in human an-
notation. While human annotation is the gold stan-
dard, the FaithBench paper (Bao et al., 2025) noted
imperfect inter-annotator agreement in general and
low inter-annotator agreement on more gray-area
Benign and Questionable Hallucinations. We also
note that we observe some erroneous annotations
within FaithBench.
Effectiveness generally improves with increas-
ing model size. We also tested an ensemble ap-
proach inspired by FACTS Grounding but found
that combining predictions from multiple models,
including o3-mini-high itself, did not outperform
o3-mini-high alone. Therefore, we adopt the o3-
mini-high judge as the standard for FaithJudge,with the possibility of using a stronger LLM judge
down the line.
Figure 1 displays the distribution of FaithJudge
predictions across LLMs. While effective, Faith-
Judge with o3-mini-high tends to underpredict
hallucinations. This is evident for Command-R,
Mistral, and Qwen, where fewer summaries were
flagged as hallucinated compared to the number
labelled Unwanted by annotators in FaithBench.
Table 3 presents confusion matrices for both bi-
nary and ternary classification using FaithJudge.
We observe that Benign summaries, in particular,
are difficult for FaithJudge to classify correctly.
In the ternary setting, FaithJudge often misclassi-
fies Benign summaries, generally labelling them as
Consistent. Similarly, Questionable summaries are
classified unreliably, though this aligns with expec-
tations. For this reason, we only employ FaithJudge
for binary classification.
Finally, Figure 2 shows the sensitivity and speci-
ficity of FaithJudge as the number of annotated
examples provided increases. Specificity remains
consistently high, though slightly decreasing as
more examples are given, while sensitivity notably
improves as the number of examples increases.
This indicates that providing more annotated ex-
amples causes FaithJudge to predict hallucinated
cases more often and better identify hallucinations.
7 Leaderboard Rankings
Figure 3 compares the ranking of the 10 LLMs
studied in FaithBench based on human-annotated
hallucinations with rankings from FaithJudge and
Vectara’s existing hallucination leaderboard.
The left-most plot shows that rankings vary de-
pending on the type of hallucination considered:
Unwanted, Benign, or Questionable, even when
assessed by human annotation. The other plots
show that when considering all types of halluci-
nation annotations, rankings in FaithBench align
more closely with FaithJudge than with the exist-
ing leaderboard. FaithJudge rankings show six in-
versions compared to rankings from FaithBench
considering Unwanted, Benign, and Questionable
hallucinations, while the existing leaderboard rank-
ings using HHEM shows 16 inversions.
8 Conclusion
In this paper, we presented our efforts at Vectara
in evaluating and benchmarking hallucinations in
RAG, discussing and building on our established

Figure 3: Comparison of LLM rankings across FaithBench (based on (U) Unwanted, (B) Benign, and (Q) Question-
able hallucination annotations), FaithJudge, and Vectara’s Hallucination Leaderboard. Rankings reflect the number
of hallucinated summaries (from least to most).
hallucination leaderboard, and proposing Faith-
Judge. We identified effectiveness limitations in
existing hallucination detection methods, including
our own HHEM model. To address these chal-
lenges, we proposed FaithJudge, an approach that
leverages human hallucination annotations to en-
hance automated hallucination detection, achiev-
ing greater effectiveness, but requiring annotations
from summaries of the same articles.
Beyond FaithBench, we extend FaithJudge to ad-
ditional RAG tasks, including question answering
and data-to-text generation, using annotated exam-
ples of hallucinations from the RAGTruth dataset.
We discuss this further in Appendix A. We also
apply FaithJudge to a broader set of LLMs, pro-
ducing leaderboard-style rankings that currently
include 30 models. We share some of these results
in Appendix C, providing a framework for more
accurate faithfulness evaluation across diverse mod-
els and RAG tasks. We hope to continue to update
our leaderboard to evaluate new models and to use
improved LLM judges.
Acknowledgements
We respectfully acknowledge the late Simon Mark
Hughes, who led the development of the original
HHEM model and Vectara’s Hallucination Leader-
board. His contributions laid important ground-
work for Vectara’s ongoing research and continue
to leave a lasting influence on our work.
Limitations
There are some limitations with our evaluation
methodology. First, our evaluation focuses exclu-
sively on faithfulness and does not address the over-all quality or usefulness of summaries and answers.
Though summary and answer quality are important
in RAG applications, we consider this evaluation
somewhat orthogonal to faithfulness.
One issue to consider is that an extractive sum-
marizer or an LLM that simply copies parts of
or the entire article in its response would techni-
cally avoid hallucinations. Nonetheless, we main-
tain that evaluating LLMs through hallucinations
in generated summaries is promising because these
hallucinations remain persistent.
Finally, while the o3-mini-high judge demon-
strates strong effectiveness, there remains room
for enhancing accuracy and agreement with human
annotators. We hope that as LLMs continue to im-
prove, replacing o3-mini-high in FaithJudge would
allow for more accurate and reliable evaluation.
References
Forrest Bao, Miaoran Li, Rogger Luo, and Ofer
Mendelevitch. 2024. HHEM-2.1-Open.
Forrest Sheng Bao, Miaoran Li, Renyi Qu, Ge Luo,
Erana Wan, Yujia Tang, Weisi Fan, Manveer Singh
Tamber, Suleman Kazi, Vivek Sourabh, Mike Qi,
Ruixuan Tu, Chenyu Xu, Matthew Gonzales, Ofer
Mendelevitch, and Amin Ahmad. 2025. FaithBench:
A diverse hallucination benchmark for summariza-
tion by Modern LLMs. In Proceedings of the 2025
Conference of the Nations of the Americas Chap-
ter of the Association for Computational Linguistics:
Human Language Technologies (Volume 2: Short Pa-
pers) , pages 448–461, Albuquerque, New Mexico.
Association for Computational Linguistics.
Bespoke. 2024. Bespoke-Minicheck-7B.
Wei-Lin Chiang, Zhuohan Li, Zi Lin, Ying Sheng,
Zhanghao Wu, Hao Zhang, Lianmin Zheng, Siyuan

Zhuang, Yonghao Zhuang, Joseph E. Gonzalez, Ion
Stoica, and Eric P. Xing. 2023. Vicuna: An Open-
Source Chatbot Impressing GPT-4 with 90%* Chat-
GPT Quality.
Shahul Es, Jithin James, Luis Espinosa Anke, and
Steven Schockaert. 2024. RAGAs: Automated evalu-
ation of retrieval augmented generation. In Proceed-
ings of the 18th Conference of the European Chap-
ter of the Association for Computational Linguistics:
System Demonstrations , pages 150–158, St. Julians,
Malta. Association for Computational Linguistics.
Alexander Fabbri, Chien-Sheng Wu, Wenhao Liu, and
Caiming Xiong. 2022. QAFactEval: Improved QA-
based factual consistency evaluation for summariza-
tion. In Proceedings of the 2022 Conference of the
North American Chapter of the Association for Com-
putational Linguistics: Human Language Technolo-
gies, pages 2587–2601, Seattle, United States. Asso-
ciation for Computational Linguistics.
Alexander R. Fabbri, Wojciech Kry ´sci´nski, Bryan Mc-
Cann, Caiming Xiong, Richard Socher, and Dragomir
Radev. 2021. Summeval: Re-evaluating summariza-
tion evaluation. Transactions of the Association for
Computational Linguistics , 9:391–409.
Galileo. 2023. LLM Hallucination Index.
https://www.galileo.ai/hallucinationindex.
Mingqi Gao, Jie Ruan, Renliang Sun, Xunjian Yin,
Shiping Yang, and Xiaojun Wan. 2023. Human-
like Summarization Evaluation with ChatGPT.
arXiv:2304.02554 .
Zorik Gekhman, Jonathan Herzig, Roee Aharoni, Chen
Elkind, and Idan Szpektor. 2023. TrueTeacher:
Learning factual consistency evaluation with large
language models. In Proceedings of the 2023 Con-
ference on Empirical Methods in Natural Language
Processing , pages 2053–2070, Singapore. Associa-
tion for Computational Linguistics.
Google Gemini Team. 2024. Gemini 1.5: Unlocking
multimodal understanding across millions of tokens
of context. arXiv:2403.05530 .
Aaron Grattafiori, Abhimanyu Dubey, Abhinav Jauhri,
Abhinav Pandey, Abhishek Kadian, Ahmad Al-
Dahle, Aiesha Letman, Akhil Mathur, Alan Schelten,
Alex Vaughan, et al. 2024. The Llama 3 Herd of
Models . arXiv:2407.21783 .
Kelvin Guu, Kenton Lee, Zora Tung, Panupong Pasupat,
and Mingwei Chang. 2020. Retrieval Augmented
Language Model Pre-Training. In Proceedings of the
37th International Conference on Machine Learning ,
volume 119 of Proceedings of Machine Learning
Research , pages 3929–3938. PMLR.
Karl Moritz Hermann, Tomáš Ko ˇciský, Edward Grefen-
stette, Lasse Espeholt, Will Kay, Mustafa Suleyman,
and Phil Blunsom. 2015. Teaching Machines to Readand Comprehend . In Proceedings of the 29th Inter-
national Conference on Neural Information Process-
ing Systems - Volume 1 , NIPS’15, page 1693–1701,
Cambridge, MA, USA. MIT Press.
Or Honovich, Roee Aharoni, Jonathan Herzig, Hagai
Taitelbaum, Doron Kukliansy, Vered Cohen, Thomas
Scialom, Idan Szpektor, Avinatan Hassidim, and
Yossi Matias. 2022. TRUE: Re-evaluating factual
consistency evaluation. In Proceedings of the 2022
Conference of the North American Chapter of the
Association for Computational Linguistics: Human
Language Technologies , pages 3905–3920, Seattle,
United States. Association for Computational Lin-
guistics.
Yebowen Hu, Timothy Ganter, Hanieh Deilamsalehy,
Franck Dernoncourt, Hassan Foroosh, and Fei Liu.
2023. MeetingBank: A benchmark dataset for meet-
ing summarization. In Proceedings of the 61st An-
nual Meeting of the Association for Computational
Linguistics (Volume 1: Long Papers) , pages 16409–
16423, Toronto, Canada. Association for Computa-
tional Linguistics.
Dandan Huang, Leyang Cui, Sen Yang, Guangsheng
Bao, Kun Wang, Jun Xie, and Yue Zhang. 2020.
What Have We Achieved on Text Summarization?
InProceedings of the 2020 Conference on Empirical
Methods in Natural Language Processing (EMNLP) ,
pages 446–469, Online. Association for Computa-
tional Linguistics.
Lei Huang, Weijiang Yu, Weitao Ma, Weihong Zhong,
Zhangyin Feng, Haotian Wang, Qianglong Chen,
Weihua Peng, Xiaocheng Feng, Bing Qin, and Ting
Liu. 2025. A Survey on Hallucination in Large Lan-
guage Models: Principles, Taxonomy, Challenges,
and Open Questions. ACM Trans. Inf. Syst. , 43(2).
Simon Hughes and Minseok Bae. 2023. Vectara Hallu-
cination Leaderboard.
Alon Jacovi, Andrew Wang, Chris Alberti, Connie
Tao, Jon Lipovetz, Kate Olszewska, Lukas Haas,
Michelle Liu, Nate Keating, Adam Bloniarz, et al.
2025. The FACTS Grounding Leaderboard: Bench-
marking LLMs’ Ability to Ground Responses to
Long-Form Input. arXiv:2501.03200 .
Ziwei Ji, Nayeon Lee, Rita Frieske, Tiezheng Yu, Dan
Su, Yan Xu, Etsuko Ishii, Ye Jin Bang, Andrea
Madotto, and Pascale Fung. 2023. Survey of Hal-
lucination in Natural Language Generation. ACM
Comput. Surv. , 55(12).
Qi Jia, Siyu Ren, Yizhu Liu, and Kenny Zhu. 2023.
Zero-shot Faithfulness Evaluation for Text Summa-
rization with Foundation Language Model. In Pro-
ceedings of the 2023 Conference on Empirical Meth-
ods in Natural Language Processing , pages 11017–
11031, Singapore. Association for Computational
Linguistics.
Albert Q. Jiang, Alexandre Sablayrolles, Arthur Men-
sch, Chris Bamford, Devendra Singh Chaplot, Diego

de las Casas, Florian Bressand, Gianna Lengyel,
Guillaume Lample, Lucile Saulnier, Lélio Re-
nard Lavaud, Marie-Anne Lachaux, Pierre Stock,
Teven Le Scao, Thibaut Lavril, Thomas Wang, Timo-
thée Lacroix, and William El Sayed. 2023. Mistral
7B.arXiv:2310.06825 .
Philippe Laban, Tobias Schnabel, Paul N. Bennett, and
Marti A. Hearst. 2022. SummaC: Re-visiting NLI-
based models for inconsistency detection in summa-
rization. Transactions of the Association for Compu-
tational Linguistics , 10:163–177.
Mike Lewis, Yinhan Liu, Naman Goyal, Marjan
Ghazvininejad, Abdelrahman Mohamed, Omer Levy,
Veselin Stoyanov, and Luke Zettlemoyer. 2020a.
BART: Denoising sequence-to-sequence pre-training
for natural language generation, translation, and com-
prehension. In Proceedings of the 58th Annual Meet-
ing of the Association for Computational Linguistics ,
pages 7871–7880, Online. Association for Computa-
tional Linguistics.
Patrick Lewis, Ethan Perez, Aleksandra Piktus, Fabio
Petroni, Vladimir Karpukhin, Naman Goyal, Hein-
rich Küttler, Mike Lewis, Wen-tau Yih, Tim Rock-
täschel, Sebastian Riedel, and Douwe Kiela. 2020b.
Retrieval-augmented generation for knowledge-
intensive nlp tasks. In Proceedings of the 34th Inter-
national Conference on Neural Information Process-
ing Systems , NIPS ’20, Red Hook, NY , USA. Curran
Associates Inc.
Junyi Li, Xiaoxue Cheng, Xin Zhao, Jian-Yun Nie, and
Ji-Rong Wen. 2023. HaluEval: A large-scale hal-
lucination evaluation benchmark for large language
models. In Proceedings of the 2023 Conference on
Empirical Methods in Natural Language Processing ,
pages 6449–6464, Singapore. Association for Com-
putational Linguistics.
Stephanie Lin, Jacob Hilton, and Owain Evans. 2022.
TruthfulQA: Measuring how models mimic human
falsehoods. In Proceedings of the 60th Annual Meet-
ing of the Association for Computational Linguistics
(Volume 1: Long Papers) , pages 3214–3252, Dublin,
Ireland. Association for Computational Linguistics.
Zheheng Luo, Qianqian Xie, and Sophia Ananiadou.
2023. ChatGPT as a Factual Inconsistency Evaluator
for Text Summarization. arXiv:2303.15621 .
Joshua Maynez, Shashi Narayan, Bernd Bohnet, and
Ryan McDonald. 2020. On Faithfulness and Factu-
ality in Abstractive Summarization. In Proceedings
of the 58th Annual Meeting of the Association for
Computational Linguistics , pages 1906–1919, On-
line. Association for Computational Linguistics.
Sewon Min, Kalpesh Krishna, Xinxi Lyu, Mike Lewis,
Wen-tau Yih, Pang Koh, Mohit Iyyer, Luke Zettle-
moyer, and Hannaneh Hajishirzi. 2023. FActScore:
Fine-grained atomic evaluation of factual precision
in long form text generation. In Proceedings of the
2023 Conference on Empirical Methods in NaturalLanguage Processing , pages 12076–12100, Singa-
pore. Association for Computational Linguistics.
Yifei Ming, Senthil Purushwalkam, Shrey Pandit, Zix-
uan Ke, Xuan-Phi Nguyen, Caiming Xiong, and
Shafiq Joty. 2024. FaithEval: Can Your Language
Model Stay Faithful to Context, Even If" The Moon
is Made of Marshmallows". arXiv:2410.03727 .
Shashi Narayan, Shay B. Cohen, and Mirella Lapata.
2018. Don’t Give Me the Details, Just the Sum-
mary! Topic-Aware Convolutional Neural Networks
for Extreme Summarization. In Proceedings of the
2018 Conference on Empirical Methods in Natural
Language Processing , pages 1797–1807, Brussels,
Belgium. Association for Computational Linguistics.
Cheng Niu, Yuanhao Wu, Juno Zhu, Siliang Xu,
KaShun Shum, Randy Zhong, Juntong Song, and
Tong Zhang. 2024. RAGTruth: A hallucination cor-
pus for developing trustworthy retrieval-augmented
language models. In Proceedings of the 62nd An-
nual Meeting of the Association for Computational
Linguistics (Volume 1: Long Papers) , pages 10862–
10878, Bangkok, Thailand. Association for Compu-
tational Linguistics.
OpenAI. 2023. GPT-4 technical report.
arXiv:2303.08774 .
Artidoro Pagnoni, Vidhisha Balachandran, and Yulia
Tsvetkov. 2021. Understanding factuality in abstrac-
tive summarization with FRANK: A benchmark for
factuality metrics. In Proceedings of the 2021 Con-
ference of the North American Chapter of the Asso-
ciation for Computational Linguistics: Human Lan-
guage Technologies , pages 4812–4829, Online. As-
sociation for Computational Linguistics.
Colin Raffel, Noam Shazeer, Adam Roberts, Katherine
Lee, Sharan Narang, Michael Matena, Yanqi Zhou,
Wei Li, and Peter J. Liu. 2020. Exploring the Lim-
its of Transfer Learning with a Unified Text-to-Text
Transformer. Journal of Machine Learning Research ,
21(140):1–67.
Tal Schuster, Adam Fisch, and Regina Barzilay. 2021.
Get your vitamin C! robust fact verification with
contrastive evidence. In Proceedings of the 2021
Conference of the North American Chapter of the
Association for Computational Linguistics: Human
Language Technologies , pages 624–643, Online. As-
sociation for Computational Linguistics.
Kurt Shuster, Spencer Poff, Moya Chen, Douwe Kiela,
and Jason Weston. 2021. Retrieval Augmentation
Reduces Hallucination in Conversation. In Find-
ings of the Association for Computational Linguis-
tics: EMNLP 2021 , pages 3784–3803, Punta Cana,
Dominican Republic. Association for Computational
Linguistics.
Liyan Tang, Tanya Goyal, Alex Fabbri, Philippe Laban,
Jiacheng Xu, Semih Yavuz, Wojciech Kryscinski,
Justin Rousseau, and Greg Durrett. 2023. Under-
standing Factual Errors in Summarization: Errors,

Summarizers, Datasets, Error Detectors. In Proceed-
ings of the 61st Annual Meeting of the Association for
Computational Linguistics (Volume 1: Long Papers) ,
pages 11626–11644, Toronto, Canada. Association
for Computational Linguistics.
Liyan Tang, Philippe Laban, and Greg Durrett. 2024a.
MiniCheck: Efficient fact-checking of LLMs on
grounding documents. In Proceedings of the 2024
Conference on Empirical Methods in Natural Lan-
guage Processing , pages 8818–8847, Miami, Florida,
USA. Association for Computational Linguistics.
Liyan Tang, Igor Shalyminov, Amy Wong, Jon Burnsky,
Jake Vincent, Yu’an Yang, Siffi Singh, Song Feng,
Hwanjun Song, Hang Su, Lijia Sun, Yi Zhang, Saab
Mansour, and Kathleen McKeown. 2024b. TofuEval:
Evaluating hallucinations of LLMs on topic-focused
dialogue summarization. In Proceedings of the 2024
Conference of the North American Chapter of the
Association for Computational Linguistics: Human
Language Technologies (Volume 1: Long Papers) ,
pages 4455–4480, Mexico City, Mexico. Association
for Computational Linguistics.
James Thorne, Andreas Vlachos, Christos
Christodoulopoulos, and Arpit Mittal. 2018.
FEVER: a large-scale dataset for fact extraction
and VERification. In Proceedings of the 2018
Conference of the North American Chapter of
the Association for Computational Linguistics:
Human Language Technologies, Volume 1 (Long
Papers) , pages 809–819, New Orleans, Louisiana.
Association for Computational Linguistics.
Hugo Touvron, Louis Martin, Kevin Stone, Peter Al-
bert, Amjad Almahairi, Yasmine Babaei, Nikolay
Bashlykov, Soumya Batra, Prajjwal Bhargava, Shruti
Bhosale, et al. 2023. Llama 2: Open foundation and
fine-tuned chat models. arXiv:2307.09288 .
Can Xu, Qingfeng Sun, Kai Zheng, Xiubo Geng,
Pu Zhao, Jiazhan Feng, Chongyang Tao, and
Daxin Jiang. 2023. Wizardlm: Empowering large
language models to follow complex instructions.
arXiv:2304.12244 .
Yuheng Zha, Yichi Yang, Ruichen Li, and Zhiting Hu.
2023. AlignScore: Evaluating factual consistency
with a unified alignment function. In Proceedings
of the 61st Annual Meeting of the Association for
Computational Linguistics (Volume 1: Long Papers) ,
pages 11328–11348, Toronto, Canada. Association
for Computational Linguistics.
Jingqing Zhang, Yao Zhao, Mohammad Saleh, and Pe-
ter J. Liu. 2020. PEGASUS: pre-training with ex-
tracted gap-sentences for abstractive summarization.
InProceedings of the 37th International Conference
on Machine Learning , ICML’20. JMLR.org.
Lianmin Zheng, Wei-Lin Chiang, Ying Sheng, Siyuan
Zhuang, Zhanghao Wu, Yonghao Zhuang, Zi Lin,
Zhuohan Li, Dacheng Li, Eric Xing, Hao Zhang,
Joseph E Gonzalez, and Ion Stoica. 2023. Judg-
ing LLM-as-a-Judge with MT-Bench and ChatbotArena. In Advances in Neural Information Process-
ing Systems , volume 36, pages 46595–46623. Curran
Associates, Inc.
Chunting Zhou, Graham Neubig, Jiatao Gu, Mona Diab,
Francisco Guzmán, Luke Zettlemoyer, and Marjan
Ghazvininejad. 2021. Detecting Hallucinated Con-
tent in Conditional Neural Sequence Generation. In
Findings of the Association for Computational Lin-
guistics: ACL-IJCNLP 2021 , pages 1393–1404, On-
line. Association for Computational Linguistics.

A Adding More Evaluation Tasks
While FaithBench provides hallucination annota-
tions across 10 different LLMs, it is limited to
evaluating summaries only.
To broaden the scope of FaithJudge beyond sum-
marization, we incorporate annotated responses
from the RAGTruth dataset (Niu et al., 2024).
RAGTruth includes three types of tasks: Summa-
rization, Question Answering (QA), and a Data-
to-Text generation task that requires generating an
overview of a business from JSON data sourced
from the Yelp Open Dataset. It contains human-
annotated hallucination labels for responses gen-
erated by six LLMs: GPT-3.5, GPT-4 (OpenAI,
2023), Llama-2 (7B, 13B, 70B) (Touvron et al.,
2023), and Mistral-7B (Jiang et al., 2023).
For each RAGTruth task, we take up to 150
sources (articles for summarization, queries and
passages for question-answering, and JSON data
for data-to-text) with their corresponding annotated
responses from the test set first and then the dev
set. We remove sources where none of the LLM
responses have a hallucination annotation.
Table 4 compares the effectiveness of FaithJudge
against the zero-shot FACTS Grounding JSON
prompt previously shown as an effective prompt
in Jacovi et al. (2025), on the FaithBench and
RAGTruth subsets used in our leaderboard. In each
setting, FaithJudge achieves stronger agreement
with human hallucination annotations, highlighting
its strength across tasks beyond summarization.
B Judge Bias
The FACTS Grounding leaderboard (Jacovi et al.,
2025) uses three different LLM judges to mitigate
bias arising from any single judge favoring its own
outputs. Inspired by this, we analyze judge bias
using Tables 5 and 6, which evaluate the impact of
using different judges across all subsets included
in our leaderboard.
Table 5 reports the effectiveness of three differ-
ent LLMs when used as judges. The table shows
that o3-mini-high remains a relatively effective
LLM for FaithJudge, often scoring the highest. The
table also shows that using multiple judges can im-
prove effectiveness further, though we note that
in some cases, individual LLMs can score higher
than the majority vote approach between the three
LLMs. For example, o3-mini-high scores higher
than the ensembling approach when evaluating on
the RAGTruth QA subset.Table 6 explores how each judge
model ranks other LLMs. Interestingly,
o3-mini-high and llama-4-maverick both
rank gemini-2.0-flash as having the fewest
hallucinated responses, while gemini-2.0-flash
ranks itself second to o3-mini-high , with only a
small difference in counts (29 vs. 31).
While using multiple judges might enhance ro-
bustness and reduce individual model bias, we cur-
rently rely on a single judge to reduce computa-
tional costs. As stronger LLMs become available,
we plan to update FaithJudge by substituting the
current judge model with a more effective one.
C Leaderboard Rankings
Table 7 presents FaithJudge rankings for a range
of LLMs. In addition to detecting hallucinations,
we also prompt FaithJudge to flag responses that
are invalid, for example, when a model fails to
meaningfully summarize an article. For simplicity,
we count these as hallucinated responses. Models
are ranked based on their overall hallucination rate,
calculated as the total number of hallucinated or
invalid responses across all four evaluation sub-
sets. We plan to continue evaluating LLMs using
FaithJudge alongside the existing leaderboard.

DatasetFacts Grounding Prompt FaithJudge Prompt
F1-Macro Balanced Accuracy F1-Macro Balanced Accuracy
RAGTruth-Data2Txt 77.1 75.1 86.3 85.1
RAGTruth-QA 76.9 81.6 83.4 85.4
RAGTruth-Summary 73.6 80.3 80.2 84.9
FaithBench-Summary 54.3 65.2 70.8 77.6
Table 4: Comparison between the Facts Grounding zero-shot prompting approach and the FaithJudge prompting
approach on the subsets of data used in our leaderboard. In all cases we use a o3-mini-high LLM judge. For
FaithJudge, we prompt the judge to evaluate LLM responses by providing the responses from the other LLMs in the
dataset with their corresponding annotations. For FaithBench, we evaluate using all summaries, treating summaries
labelled as Questionable or Benign as inconsistent summaries.
Dataset Model F1-Macro Balanced Accuracy
RAGTruth (Data2Txt)o3-mini-high 86.3 85.1
gemini-2.0-flash 83.6 84.0
llama-4-maverick 82.1 80.6
Majority Vote 86.4 85.8
RAGTruth (QA)o3-mini-high 83.4 85.4
gemini-2.0-flash 81.8 84.2
llama-4-maverick 77.5 81.2
Majority Vote 81.0 83.8
RAGTruth (Summary)o3-mini-high 80.2 84.9
gemini-2.0-flash 83.6 82.7
llama-4-maverick 78.0 83.7
Majority Vote 84.6 88.0
FaithBench (Summary)o3-mini-high 70.8 77.6
gemini-2.0-flash 66.1 75.5
llama-4-maverick 74.7 76.9
Majority Vote 72.4 79.1
Table 5: Evaluation results for three models and an ensemble approach on the subsets of data used in our leaderboard.
For FaithBench, we evaluate using all summaries, treating summaries labelled as Questionable or Benign as
inconsistent summaries.
Judged by o3-mini-high Judged by gemini-2.0-flash Judged by llama-4-maverick
Evaluated Model Hallucinated Responses Rank Hallucinated Responses Rank Hallucinated Responses Rank
gemini-2.0-flash 52 1 31 2 71 1
o3-mini-high 64 2 29 1 94 2
llama-4-maverick 105 3 72 3 110 3
Table 6: Total number of hallucinated responses per evaluated model, as judged by each model. Rankings indicate
relative effectiveness in terms of hallucination frequency, from least to most.

Rank Model Overall Hallucination Rate FaithBench (Summary) RAGTruth (Summary) RAGTruth (QA) RAGTruth (Data-to-Text)
1 gemini-2.5-pro-exp 7.63% 18/72 14/150 1/139 6/150
2 gemini-2.0-flash 10.18% 21/72 10/150 1/139 20/150
3 gpt-4.5-preview 11.94% 27/72 15/150 7/139 12/150
4 o3-mini-high 12.52% 25/72 12/150 9/139 18/150
5 gpt-3.5-turbo 14.87% 32/72 13/150 8/139 23/150
6 gpt-4o 15.85% 29/72 15/150 7/139 30/150
7 claude-3.7-sonnet 16.05% 28/72 22/150 13/139 19/150
8 llama-3.3-70b 16.44% 32/72 13/150 6/139 33/150
9 phi-4 17.03% 32/72 12/150 6/139 37/150
10 mistral-small-24b 17.03% 31/72 15/150 14/139 27/150
11 llama-4-maverick 20.55% 37/72 20/150 13/139 35/150
12 llama-3.1-8b 28.38% 32/72 19/150 17/139 77/150
Table 7: FaithJudge rankings for 12 LLMs, based on the number of hallucinated responses across four evaluation
subsets: article summarization from FaithBench and RAGTruth, as well as question answering and data-to-text
writing from RAGTruth.