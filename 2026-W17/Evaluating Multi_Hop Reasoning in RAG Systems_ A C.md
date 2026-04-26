# Evaluating Multi-Hop Reasoning in RAG Systems: A Comparison of LLM-Based Retriever Evaluation Strategies

**Authors**: Lorenz Brehme, Thomas Ströhle, Ruth Breu

**Published**: 2026-04-20 13:20:57

**PDF URL**: [https://arxiv.org/pdf/2604.18234v1](https://arxiv.org/pdf/2604.18234v1)

## Abstract
Retrieval-augmented generation (RAG) enhances large language models (LLMs) with external knowledge to answer questions more accurately.
  However, research on evaluating RAG systems-particularly the retriever component-remains limited, as most existing work focuses on single-context retrieval rather than multi-hop queries, where individual contexts may appear irrelevant in isolation but are essential when combined. In this research, we use the HotPotQA, MuSiQue, and SQuAD datasets to simulate a RAG system and compare three LLM-as-judge evaluation strategies, including our proposed Context-Aware Retriever Evaluation (CARE). Our goal is to better understand how multi-hop reasoning can be most effectively evaluated in RAG systems.
  Experiments with LLMs from OpenAI, Meta, and Google demonstrate that CARE consistently outperforms existing methods for evaluating multi-hop reasoning in RAG systems. The performance gains are most pronounced in models with larger parameter counts and longer context windows, while single-hop queries show minimal sensitivity to context-aware evaluation. Overall, the results highlight the critical role of context-aware evaluation in improving the reliability and accuracy of retrieval-augmented generation systems, particularly in complex query scenarios. To ensure reproducibility, we provide the complete data of our experiments at https://github.com/lorenzbrehme/CARE.

## Full Text


<!-- PDF content starts -->

Evaluating Multi-Hop Reasoning in RAG
Systems: A Comparison of LLM-Based Retriever
Evaluation Strategies
Lorenz Brehme[0009−0009−4711−2564], Thomas Ströhle[0000−0002−1954−6412], and
Ruth Breu[0000−0001−7093−4341]
Universität Innsbruck, Technikerstraße 21a, 6020 Innsbruck, Austria
{lorenz.brehme,thomas.stroehle,ruth.breu}@uibk.ac.at
Abstract.Retrieval-augmented generation (RAG) enhances large lan-
guage models (LLMs) with external knowledge to answer questions more
accurately. However, research on evaluating RAG systems—particularly
the retriever component—remains limited, as most existing work focuses
on single-context retrieval rather than multi-hop queries, where indi-
vidual contexts may appear irrelevant in isolation but are essential when
combined.Inthisresearch,weusetheHotPotQA,MuSiQue,andSQuAD
datasets to simulate a RAG system and compare three LLM-as-judge
evaluation strategies, including our proposed Context-Aware Retriever
Evaluation(CARE).Ourgoalistobetterunderstandhowmulti-hoprea-
soning can be most effectively evaluated in RAG systems. Experiments
with LLMs from OpenAI, Meta, and Google demonstrate that CARE
consistently outperforms existing methods for evaluating multi-hop rea-
soning in RAG systems. The performance gains are most pronounced in
models with larger parameter counts and longer context windows, while
single-hop queries show minimal sensitivity to context-aware evaluation.
Overall, the results highlight the critical role of context-aware evalu-
ation in improving the reliability and accuracy of retrieval-augmented
generation systems, particularly in complex query scenarios. To ensure
reproducibility, we provide the complete data of our experiments at
https://github.com/lorenzbrehme/CARE.
1 Introduction
Retrieval-Augmented Generation (RAG) is an ongoing trend in research and has
been continuously developed over the past years [5]. It augments their knowl-
edge through the integration of external data sources and addresses hallucina-
tions in large language models (LLMs) [15]. A basic RAG system consists of
three components [13]: (i) Indexing component—responsible for indexing and
embedding external data. (ii) Retriever component—filters and retrieves rele-
vant information based on the input query. (iii) Generator component—uses an
Accepted for publication at the SynIRgy Workshop, ECIR 2026 (48th European
Conference on Information Retrieval).arXiv:2604.18234v1  [cs.IR]  20 Apr 2026

2 Brehme et al.
LLM to process the retrieved data together with the input query to generate
the final response. Evaluating all three components is essential for assessing the
performance of a RAG system and ensuring it produces accurate, well-grounded
responses: While the generator component has been extensively studied—with
approaches such as RAGAS [11] and Ares [23] proposing methodologies to as-
sess the quality of RAG-generated responses—evaluation of the retrieval and
indexing components has received comparatively less attention [6]. In partic-
ular, assessment of indexing methods often relies on system-level performance
metrics such as throughput and latency [14].
Our focus is on retriever evaluation, which assesses the relevance of the con-
text retrieved by a RAG system. This evaluation is typically conducted using one
of two main strategies: labeled [19,25] and unlabeled [1,24]. In labeled evalua-
tions,therelevantcontextforeachqueryispredefinedwithinthedataset,andthe
retriever is assessed based on its ability to return these labeled documents [25].
However, labeled evaluation is not always practical. When the chunking strat-
egy is modified, the segments labeled as relevant may no longer correspond to
the actual segments retrieved by the RAG system due to changes in document
segmentation [17]. Additionally, the structure of QA datasets poses challenges:
each question-answer pair is typically derived from documents in the corpus
and labeled accordingly [11,25]. This process does not guarantee that all other
documents are irrelevant, as no negative sampling is applied [30]. As a result, a
retrievermayreturndocumentsnotlabeledasrelevantthatstillcontainessential
information for answering the query [6].
Therefore, unlabeled approaches have become prevalent, as they assess con-
text relevance without requiring labeled documents [6]. In this setting, only the
input query and the ground-truth answer are used to determine whether the
retrieved contexts are relevant.This assessment can be carried out by human
annotators [1] or through automated approaches using LLMs, as demonstrated
in evaluation frameworks such as the one introduced by [21]. In unlabeled eval-
uation approaches, existing methods, like direct [1,21,23] and indirect evalua-
tion [24] do not account for multi-hop queries. These methods typically evaluate
each retrieved context in isolation, ignoring the relationships and dependencies
among multiple retrieved documents. This limitation is particularly critical for
multi-hop queries, which require integrating information from multiple sources
to derive a correct answer.
This work investigates effective evaluation of RAG systems in unlabeled set-
tings, focusing on multi-hop queries. We compare three evaluation strategies in a
synthetic RAG environment and benchmark them on HotPotQA, MuSiQue, and
SQuAD using LLMs from OpenAI, Meta, and Google. We analyze how question
difficulty and query type affect performance. Our results show that CARE out-
performsexistingapproaches,withthelargestgainsoncomplexandcomparative
multi-hopquestions—especiallywhenusingmodelswithlongercontextwindows,
larger parameter counts, and shorter prompts. Notably, while single-hop queries
generally do not require context awareness, multi-hop evaluation benefits sub-

Multi-Hop Retriever Evaluation 3
stantially from it, underscoring the importance of context-aware evaluation for
improving the reliability and accuracy of RAG systems in challenging scenarios.
2 Related Work
The use of LLMs for retriever evaluation has been explored in several studies [12,
23,3]. Typically, an LLM is prompted with an input query, an answer, and a re-
trieved context and is tasked with determining whether the context is relevant to
answering the query [6]. In the literature, two primary definitions of context rel-
evance have emerged. The first definition characterizes context relevance based
on whether the context was actually used by the LLM during answer genera-
tion [12,9]. The second definition considers a context relevant if the information
retrieved is useful for answering the question [23]. In this work, we adopt the
second definition, as the former is not applicable to our setting: it relies on the
internal behavior of the generator. Existing work largely falls into two categories
for measuring context relevance:direct(cf. Figure 1b) andindirect(cf. Figure
1a) evaluation approaches [6]. Indirectevaluation, the LLM is provided with
the input query, the ground-truth answer, and the context to be evaluated. The
LLM is then tasked with determining, based on a specific prompt, whether the
context is relevant to answering the question. Prior work has explored various
labeling schemes under this paradigm, including graded relevance scales [21],
binary relevance classification [23], and sentence-level relevance estimation [11].
In contrast, theindirectapproach preprocesses the retrieved context and input
query before assessing context relevance. For example, [3] generate an answer
from theretrievedcontext list andcompare it toan answer derived from agolden
document list containing all relevant contexts. Contexts are considered relevant
if the answers match. This method limits labeling to the entire context set, pre-
venting individual relevance assignments. Another adaptation by [24] generates
an answer for each context individually and labels a context as relevant if the
answer is correct. While this method allows for individual labels for each con-
text, it does not account for multi-hop queries, where the answer depends on
multiple pieces of context. Since multi-hop reasoning is a common requirement
in RAG systems [25], our research investigates and compares different unlabeled,
LLM-based evaluation approaches. Our goal is to identify effective strategies for
retriever evaluation that capture the interdependencies among multiple retrieved
contextsandtodeterminewhichevaluationmethodsbestreflectretrievalquality
in multi-hop settings.
3 Datasets
To simulate the RAG pipeline, each dataset must include a question, a set of
retrieved documents containing both relevant and non-relevant contexts, and a
ground-truth answer to support evaluation. The inclusion of labeled relevant
and non-relevant contexts is essential, as it enables evaluation of whether the

4 Brehme et al.
LLM correctly distinguishes relevant information from not relevant. For our ex-
periments, we simulate the RAG pipeline using the multi-hop datasets Hot-
PotQA [31] and MuSiQue [26], as well as the single-hop dataset SQuAD 2.0 [22].
While HotPotQA and MuSiQue explicitly include irrelevant contexts in their
original structure, SQuAD 2.0 requires adaptation. In SQuAD 2.0, each context
document is associated with a set of questions that are either answerable or
unanswerable based on that specific text. To transform this into a retriever-style
format, we treat the original context as relevant if it contains the answer and as
non-relevant if it does not. We then augment each question–context pair with
additional distractor contexts. Specifically, we use the full SQuAD corpus to
retrieve the top 19 most similar documents per question using BM25 [27], re-
sulting in a context list of 20 contexts for each query. These additional contexts
serve as unlabeled context documents to simulate a retriever returning multiple
documents, though only the originally labeled context is used for the final rele-
vance evaluation. While SQuAD 2.0 contains only single-hop queries, HotPotQA
and MuSiQue are well suited for evaluating multi-hop reasoning, as they provide
question–answer pairs with multiple context documents, typically two of which
are labeled as relevant. For our experiments, we selected instances with exactly
two relevant contexts and used the datasets as provided, enabling a consistent
multi-hop evaluation. This yields context lists of 10 documents per question for
HotPotQA and 20 for MuSiQue.
Further granularity is provided by the HotPotQA dataset, which categorizes
questions into easy, medium, and hard difficulty levels. Easy questions are pri-
marily single-hop, while medium and hard questions require multi-hop reason-
ing. The distinction between the latter two levels is based on the performance
of state-of-the-art models as reported by [31], where questions with a success
rate below 60% are classified as hard. Additionally, the dataset includes two
distinct question types, bridge and comparison, described by [31]: Bridge ques-
tions require connecting multiple pieces of information from different contexts
to arrive at the answer. For example, in the question“Where was the singer
and songwriter of Radiohead born?”, the bridge entity is Radiohead. To answer
this, one must first determine who the singer and songwriter of Radiohead is,
and then find that person’s place of birth. Comparison questions, on the other
hand, involve evaluating and comparing facts across multiple contexts. For in-
stance, the question“Who has played more NBA games, Michael Jordan or Kobe
Bryant?”requires retrieving the number of NBA games each player has played
from different sources, then comparing those values to find the answer.
4 Evaluation Strategies
In this work, we compare three retrieval evaluation approaches: (i) an indirect
evaluationapproachderivedfromtheeRAGmethod[24],and(ii)adirectevalua-
tionmethodbasedontheARESframework[23],(iii)ourproposedcontext-aware
retriever evaluation (CARE) method (see Figure 1).

Multi-Hop Retriever Evaluation 5
indirect
Compare Generated Answer with
Ground Truth AnswerQuestion q
Context ℓi 
Answer ai
Label yiAnswer "{ q }" using only "{ ℓi }", or
respond "Cannot answer"  if
insuf ficient information.
(a) Indirect Approach
direct
Question q
Context ℓi 
Label yiIs the context "{ ℓi }" crucial to
answering "{ q }" with "{ a* }"?Ground Truth Answer a* (b) Direct Approach
CARE
Question q
Contextlist  L
Label yiContext ℓi 
Is "{ ℓi }" crucial to answering "{ q }"
with "{ a* }", given "{ L }"?Ground Truth Answer a* (c) CARE Approach
Fig.1: Illustration of considered evaluation strategies
Given a queryq, the retrieverRreturns a list of context documentsL=
{ℓ1, ℓ2, . . . , ℓ n}, whereℓ i∈ Ldenotes a single retrieved context. Leta∗denote
the ground-truth answer associated withq. Each evaluation method outputs a
binary relevance labely i∈ {0,1}, wherey i= 1denotes relevance of contextℓ i
with respect to queryq, andy i= 0denotes non-relevance.
In theindirect method[24], an LLMIattempts to answer the queryqusing
only a single context documentℓ i, producing an answera i=I(q, ℓ i). Ifa i=∅,
thecontextℓ iislabeledasnon-relevant.Otherwise,a iiscomparedtotheground-
truth answera∗using an exact-match function, and the context is labeled as
relevant if the answers match, and non-relevant otherwise. To reduce errors and
biases introduced by strict exact matching, we manually reviewed all cases where
no match was detected. Both the pre-review and revised labels are available in
ourrepository.Inpractice,thecomparisoncriterionmaybeadaptedtothetarget
application, ranging from strict exact matching to more flexible approaches such
as LLM-based semantic equivalence judgments.
Thedirect method[23] uses an LLMDto predict the relevance label directly,
defined asy i=D(q, a∗, ℓi). The LLM is provided with a detailed description
of the task, including explicit instructions on how to assess the relevance of a
context.TheLLMreceivesseveralpiecesofinformationasinput:(1)thequestion
being addressedq, (2) the specific context documentℓ ito be classified, and (3)
the ground-truth answera∗. Based on this information, the LLM classifies the
context document as either relevant or non-relevant.
Our proposedCAREmethod extends the direct evaluation with a fourth
element, (4) the list of context documentsL. Specifically, relevance is predicted
asy i=C(q, a∗,L, ℓ i). The inclusion of the context list is particularly important
in multi-hop settings, as it allows the evaluation to account for the presence of
multiple supporting contexts.
The complete approaches are illustrated in Figure 1, and the prompts used
are available in our repository [18]. We initially employed a zero-shot prompting
strategy to ensure that the task was explained as clearly as possible. These

6 Brehme et al.
approachesassumethat,intheRAGevaluationprocess,adatasetwithquestions
and their corresponding ground-truth answers is available, and that a question
may require multiple contexts to generate an answer. As a result, the evaluation
output consists of a label indicating the relevance of each context.
5 Experimental Setup
Datasets and sampling.For our experiments, we used three datasets, the Hot-
PotQA, SQuAD 2.0 and MuSiQue datasets. From each, we sampled 600 relevant
and 600 non-relevant contexts. For HotPotQA, we stratified by difficulty (easy,
medium, hard), selecting 400 contexts per level. Within each level, we balanced
reasoning type with 200 comparison and 200 bridge contexts, yielding six cate-
gories of 200 samples each.
Models.For SQuAD 2.0 and MuSiQue, we performed experiments exclusively
with GPT-4.1. To assess differences in model performance, we conducted eval-
uations using six LLMs on the HotPotQA dataset, from Gemini, OpenAI, and
Meta’s LLaMa series using their standard settings. From Gemini, we used the
gemini-2.0-flashasthestandardmodelandthegemini-1.5-flash-8basthesmaller
model via Google’s API [8]. From Meta, we included the LLaMa 3.1–70b as a
medium-sized model and the LLaMa 3.1–8b as the smaller model [2]. From
OpenAI, we selected the standard GPT-4.1 model, as well as the o4-mini, which
we considered as our reasoning model through the OpenAI API [20]. This setup
enabled us to compare model sizes, reasoning capabilities, and the viability of
open-source LLaMa models for self-hosted use.
Prompting.Ourprimaryobjectivewastocompareevaluationmethodsthrough
zero-shotpromptingratherthanfine-tuningmodels.Labelsweregeneratedusing
the prompt templates provided in the repository, and we deliberately avoided
model-specific prompt optimization to ensure independence and generalizabil-
ity. However, for the CARE evaluation, we also examined the impact of different
promptingstrategiesonHotPotQAbymodifyingtheoriginalpromptandtesting
various configurations.
Metrics and statistical analysis.We computed standard retrieval metrics in-
cluding accuracy, precision, recall, and F1-Score to assess performance. To en-
hance the robustness of our findings, we applied bootstrapping techniques [10].
When bootstrapping the entire dataset, we generated 5,000 samples, each con-
taining 1,200 data points. For bootstrapping specific question types or difficulty
levels, we used the corresponding subset sizes (e.g., 600 for bridge questions, 400
for the easy difficulty level). From these samples, we constructed 95% confidence
intervals using anαlevel of 0.05. To assess the statistical significance and robust-
ness of our results, we conducted permutation tests with 9,999 permutations.
6 Results
In the following section, we present our experimental results. All tables report
bootstrapped confidence intervals and indicate statistically significant results

Multi-Hop Retriever Evaluation 7
from permutation tests with an asterisk (*) relative to the highlighted baseline
(Base). For the CARE approach, we indicate the number of contexts in the
contextlistbyappendingitasasubscriptintheformatCARE n,wherendenotes
the size of the list.
Multi-Hop.OurinitialresultsdemonstrateinTable1therobustnessofCARE
for multi-hop queries. It achieves significantly better performance compared to
both the indirect and direct approaches across nearly all metrics, including ac-
curacy, F1-Score, and recall, in both multi-hop datasets. Both the indirect and
direct approaches show higher precision than recall, indicating that when they
classify a context as relevant, it is likely to be correct. However, their relatively
low recall suggests they miss a significant portion of relevant contexts, likely la-
beling them as relevant only when highly confident. Overall, the direct approach
outperforms the indirect approach, achieving significantly higher accuracy, F1-
Score, and recall.
Table 1: Comparison of different approaches for GPT-4.1
Approach Accuracy F1-Score Recall Precision
HotPotQA
Indirect[24]0.642±0.03* 0.474±0.04* 0.322±0.04*0.898±0.04
Direct[23]0.720±0.03* 0.658±0.04* 0.540±0.04*0.844±0.04
CARE 10Base0.827±0.02 0.814±0.02 0.757±0.04 0.880±0.03
MuSiQue
Indirect[24]0.631±0.03* 0.417±0.04* 0.264±0.03*0.994±0.01
Direct[23]0.702±0.03* 0.578±0.04* 0.410±0.04*0.984±0.01
CARE 20Base0.755±0.02 0.678±0.03 0.517±0.04 0.987±0.01
* statistically significant difference;Basedenotes baseline.
LLM model comparison.The experiments were replicated on the HotPotQA
dataset using various underlying models for comparison as presented in table 2.
The indirect approach led to a significant improvement in F1-Score for the small
LLaMa model. In contrast, the direct approach resulted in a decline in F1-Score
for the reasoning model o4-mini. However, both approaches generally performed
comparably across different models. For CARE, the reasoning model o4-mini ex-
hibited a decrease in accuracy, F1-Score, and recall compared to GPT-4.1, while
precision significantly improved. Among smaller models, we observed a shift
from precision to recall. Notably, the LLaMa 3.1-8b model experienced a signifi-
cant decline in overall performance, with substantial drops in both F1-Score and
accuracy. Unexpectedly, its accuracy fell below that of the direct approach. Fur-
thermore, CARE consistently outperformed other approaches across all models
except for the LLaMa 3.1-8b model, demonstrating stable performance when
context size and parameter size were sufficiently high.

8 Brehme et al.
Table 2: Model comparison on the HotPotQA dataset.
Approach Model Accuracy F1-Score Recall Precision
Indirect[24] GPT-4.1Base0.642±0.03 0.474±0.04 0.322±0.04 0.898±0.04
o4-mini 0.643±0.03 0.473±0.04 0.322±0.04 0.898±0.04
2.0-flashBase0.595±0.03 0.345±0.05 0.213±0.03 0.901±0.05
1.5-flash-8b 0.579±0.03 0.294±0.04 0.175±0.03 0.914±0.05
LLaMa70bBase0.605±0.03 0.368±0.05 0.230±0.03 0.920±0.05
LLaMa8b 0.566±0.030.445±0.04*0.348±0.04*0.618±0.05*
Direct[23] GPT-4.1Base0.720±0.03 0.658±0.04 0.540±0.04 0.844±0.04
o4-mini 0.694±0.030.590±0.04*0.442±0.04*0.889±0.04
2.0-flashBase0.691±0.03 0.590±0.04 0.443±0.04 0.881±0.04
1.5-flash-8b 0.708±0.03 0.610±0.04 0.457±0.04 0.920±0.03
LLaMa70bBase0.745±0.03 0.686±0.03 0.556±0.04 0.895±0.03
LLaMa8b 0.748±0.03 0.719±0.030.643±0.04*0.815±0.04*
CARE 10GPT-4.1Base0.827±0.02 0.814±0.02 0.757±0.04 0.880±0.03
o4-mini0.781±0.02*0.732±0.03*0.600±0.04*0.940±0.03*
2.0-flashBase0.840±0.02 0.825±0.04 0.753±0.04 0.911±0.03
1.5-flash-8b 0.812±0.02 0.822±0.020.872±0.03*0.778±0.03*
LLaMa70bBase0.833±0.02 0.815±0.03 0.738±0.03 0.910±0.03
LLaMa8b0.674±0.03*0.736±0.03*0.908±0.02*0.619±0.03*
* statistically significant difference;Basedenotes baseline.
Question Levels, Types, and Overall Performance.Table 3 presents the com-
parison of F1-Scores using GPT-4.1 across all difficulty levels (easy, medium,
hard), question types (bridge and comparison). The results show that, regard-
less of question type or difficulty level, CARE consistently achieved the highest
performance across all subsets and outperformed other approaches on the en-
tire dataset. Additionally, the direct approach exhibits a performance drop on
medium and hard questions, while both direct and indirect approaches perform
worse on comparison questions. The indirect approach does not show a signifi-
cant performance drop on medium and hard questions, but generally performs
worse across all difficulty levels, including easy ones.
Prompting strategies.Furthermore, we explored various prompting strategies
through additional experiments using the CARE approach. First, we applied
single-shot prompting, in which only one example is included in the prompt to
guide the model’s reasoning [7]. We then extended this to few-shot prompting
by providing at least two examples–one with a relevant context and one with a
non-relevant context [7]. As a third strategy, we employed a role-based prompt,
assigning the LLM the role of an evaluator and explicitly instructing it on how to
assess relevance [29]. Another variant involved minimal instruction, presenting

Multi-Hop Retriever Evaluation 9
Table 3: F1-Score Comparison: HotPotQA Levels and Question (GPT-4.1)
Category Indirect [24] Direct [23] CARE
EasyBase0.530±0.07 0.758±0.05 0.808±0.04
Medium 0.444±0.080.637±0.06*0.822±0.04
Hard 0.438±0.080.556±0.07*0.809±0.05
BridgeBase0.639±0.05 0.724±0.05 0.804±0.04
Comparison0.264±0.06* 0.591±0.05*0.822±0.03
* statistically significant difference;Basedenotes baseline.
only the essential input information without any contextual guidance. Finally,
we experimented with the Chain-of-Thought (CoT) technique [28], combining
it with few-shot prompting to encourage step-by-step reasoning (cf. repository
for all prompting templates). To analyze the prompting strategies, we compared
the alternative prompting strategies against the standard prompt used in our
previous experiments (cf. Table 4). The short prompt yielded a significant im-
provement across all metrics except precision. The role-based prompt also led
to a significant increase in F1-Score and recall. Only the CoT prompt resulted
in a significant decline in all metrics except precision. This shows similarity to
the reasoning model, which uses CoT techniques in the background, o4-mini.
The remaining prompting strategies did not produce any statistically significant
changes in performance.
Table 4:CARE Prompting Strate-
gies on HotPotQA (GPT-4.1)
Prompt F1-Score
standardBase0.814±0.02
single-shot 0.818±0.02
few-shot 0.798±0.03
role0.849±0.02*
short0.877±0.02*
CoT0.722±0.03*Table 5:CARE and Direct (with-
/without answer) on HotPotQA us-
ing Gemini-2.0-flash
Approach F1-Score
Direct[23]Base0.590±0.04
Direct†[23]0.488±0.04*
CARE 10Base0.825±0.04
CARE 10†0.869±0.02
* statistically significant difference;Basedenotes baseline;†Approach without answer.
To better understand the poor performance of the CoT prompting approach,
we manually analyzed the reasoning steps generated by the model. Our manual
analysis of the CoT prompt errors identified three main types: hallucination
errors (40 instances), where the model invented information; context awareness

10 Brehme et al.
errors (214 instances), where it failed to use the provided context effectively;
and context interpretation errors (26 instances), where the model’s relevance
judgment differed from the ground-truth.
Without answers.To account for scenarios in which the evaluation dataset
lacks a reference answer, we adapted both CARE and the Direct approach by
removing the ground-truth answer from the prompt. We then evaluated the per-
formance of both methods, the Direct approach and CARE, under this setting,
compare Table 5. While excluding the ground-truth answer led to a decline in all
metrics except precision for the Direct approach, CARE showed no significant
negative impact.
Table 6:CARE and Direct on
SQuAD2.0 (GPT-4.1)
Approach F1-Score
Direct[23]0.819±0.02*
CARE 0 0.795±0.02*
CARE 10.748±0.02
CARE 30.739±0.02
CARE 50.738±0.02
CARE 10Base0.735±0.02
CARE 200.751±0.02
IndirectDirectCARE10
IndirectDirectCARE20
DirectCARE0
CARE1
CARE3
CARE5
CARE10
CARE200500010000150002000025000Characters per Prompt
Hotpot MuSiQue Squad2.0 Fig.2:Prompt length by approach
* statistically significant difference;Basedenotes baseline.
Single-Hop.ToassesswhethertheCAREapproach–despiteitslongerprompts
and higher computational cost, is still suitable for single-hop queries, we com-
paredittothedirectevaluationmethod.Duetothelackofground-truthanswers
for unanswerable questions in the SQuAD 2.0 dataset, we excluded the indirect
approach from this comparison and used the slightly modified approach where
the answer is not included like in 5. In this setup, we observed a significant drop
in CARE’s performance.
Contextlist length.To gain deeper insight into the CARE approach, we also
conducted an ablation study to examine the impact of context list length on
performance. Specifically, we evaluated CARE using context lists containing 0,
1, 3, 5, 10, and 20 documents (cf. Table 6). Notably, only the configuration with
zero context documents–closely resembling the direct evaluation setup–achieved
significantly better results. For all other configurations, performance did not
differ significantly, indicating that the number of context documents in the list
has limited impact on CARE’s effectiveness in this setting. These results are
summarized in Table 6.
Prompt length.We compared the number of characters per prompt and found
that CARE prompts were over four times longer than those of other methods

Multi-Hop Retriever Evaluation 11
(cf. Figure 2). The length depends on the size and number of retrieved contexts
included in the prompt.
7 Discussion
We evaluated three LLM-as-judge retriever evaluation strategies in a synthetic
RAG setup on HotPotQA, MuSiQue, and SQuAD 2.0, focusing on their ability
to label context relevance, particularly for multi-hop reasoning in RAG systems.
Deciding on an appropriate evaluation method for a RAG system largely de-
pends on the complexity of the target questions. Our findings indicate that for
multi-hop queries, evaluators must consider the full set of retrieved contexts
rather than assessing each context in isolation. Due to the nature of multi-hop
reasoning, individual contexts may appear irrelevant on their own, while their
combination renders them jointly relevant to the query. CARE leverages this in-
terdependence by evaluating contexts in light of the complete context list, which
explains its improved performance. By contrast, for single-hop questions, rele-
vance is typically self-contained within a single context, which can be assessed
independently without requiring additional supporting contexts. In this setting,
the direct approach is generally more effective: it achieves higher performance
than CARE while being less computationally expensive due to its simpler struc-
ture and shorter prompt length. The indirect approach is mainly effective on the
easy subset of HotPotQA, which is largely single-hop. In this setting, both direct
and indirect methods outperform their results on more complex multi-hop ques-
tions, consistent with [24], who report strong indirect performance under simi-
lar conditions. However, its effectiveness drops sharply on multi-hop questions,
whereas CARE maintains stable performance across both single-hop and multi-
hop queries, even when handling longer context lists. This robustness makes
CARE a broadly applicable and reliable evaluation method.
When evaluating a RAG system, it is important to consider the structure
of the evaluation dataset. This includes whether contexts are explicitly linked
to questions and ground-truth answers, and whether true answers are available
at all. CARE is a robust and practical solution for both labeled and unlabeled
datasets. By default, it is designed for unlabeled datasets, as it can generate its
own relevance labels. However, it is also valuable for labeled datasets, where it
addressesacommonlimitation:thelackofexplicit“notrelevant” labels.Asnoted
by Salemi et al. [24], a context can be relevant even if no label is assigned, while
irrelevant contexts are often unannotated [25,11]. Moreover, our experiments
demonstrate that the absence of answers does not reduce its performance. This
makes CARE a practical solution for automating retriever evaluation without
relying on large labeled datasets or ground-truth answers, while also enabling
integration into live systems for real-time, continuous relevance assessment. A
surprising finding was that reasoning models reduced CARE’s evaluation per-
formance, a result further confirmed by the CoT prompting experiment. These
results suggest that, at present, reasoning-heavy approaches are ill-suited for re-
triever evaluation with CARE. Our qualitative analysis indicates that such mod-

12 Brehme et al.
els often hallucinate, lose track of the task, or fail to consider the full context.
While future LLM improvements may mitigate these issues, our experiments
show that current reasoning models and CoT prompting do not perform well
within the CARE framework.
A key limitation of CARE is its incompatibility with smaller LLMs in terms
of parameters and context length, whereas the direct and indirect approaches
remain usable. This is evident with o4-mini and LLaMa 3.1–8B, where even di-
rect evaluation outperforms CARE. These models, having fewer parameters and
shorter context windows, struggle with CARE’s reliance on complex prompts
and extended input. Since effective use of long contexts requires strong compre-
hension and retention, these findings align with prior work showing that not all
models handle long-context reasoning effectively [4,16]. An advantage of smaller
open-source LLaMa models is that they can be self-hosted on modest hardware,
making them suitable for cost-sensitive deployments. This raises the question
of whether CARE requires the full capabilities of high-performance models, or
if smaller models can offer a cost- and energy-efficient alternative. Our findings
suggest that this depends on the specific model: for instance, LLaMa 3.1–8B un-
derperformed compared to the medium-sized variant, yet smaller Gemini models
delivered acceptable results. This indicates that CARE can, under certain con-
ditions, be executed effectively on less powerful models, reducing both cost and
energy consumption. By contrast, model size and quality had only minimal im-
pact on the direct and indirect approaches. While CARE places slightly higher
demands due to complex prompts and extended context, smaller models can still
meet these requirements with the right configuration.
8 Conclusion
This paper demonstrates that context-aware evaluation is essential for RAG sys-
tems, particularly for multi-hop queries where assessing the full retrieved context
set is critical. Our findings reveal that competitive evaluation performance does
not strictly necessitate the largest models. Instead, it depends on sufficient con-
text windows and reasoning capabilities. This indicates that cost-efficient models
can be effective evaluators if they meet these baseline requirements. Ultimately,
the choice of evaluation strategy involves a trade-off: while our proposed method
(CARE) offers superior reliability for complex tasks, simpler direct or indirect
methodsmayremainmoreefficientforsingle-hopqueriesorresource-constrained
environments. Our results are primarily constrained by their reliance on syn-
thetic, text-only datasets. In production environments, RAG systems frequently
encounter unstructured data and multimodal inputs, such as images and tables
[5]. Therefore, our findings are specific to well-processed textual contexts. Fu-
ture research should investigate the efficacy of CARE as a live evaluator in real-
world settings, particularly its robustness on uncleaned and multimodal data.
Moreover, a more systematic ablation and hyperparameter study is necessary
to better understand why CARE performs better. This should include analyses
of the number and difficulty of support contexts as well as the contribution of

Multi-Hop Retriever Evaluation 13
multi-hop reasoning, especially given its weaker performance in single-hop cases
and when used with reasoning-focused models.
Acknowledgements.The authors would like to thank Steffen Hahn, Andreas
Egger, Mateo Golec, and Daniel Walder for their initial efforts related to this
work.
References
1. Afzal, A., Kowsik, A., Fani, R., Matthes, F.: Towards optimizing and evaluating
a retrieval augmented QA chatbot using LLMs with human in the loop (2024).
https://doi.org/10.48550/arXiv.2407.05925, http://arxiv.org/abs/2407.05925
2. AI@Meta: Llama 3.1 model card (2024), https://github.com/meta-llama/llama-
models/blob/main/models/llama3_1/MODEL_CARD.md
3. Alinejad, A., Kumar, K., Vahdat, A.: Evaluating the retrieval
component in LLM-based question answering systems (2024).
https://doi.org/10.48550/arXiv.2406.06458, http://arxiv.org/abs/2406.06458
4. Bai, Y., Tu, S., Zhang, J., Peng, H., Wang, X., Lv, X., Cao, S., Xu,
J., Hou, L., Dong, Y., Tang, J., Li, J.: LongBench v2: Towards deeper
understanding and reasoning on realistic long-context multitasks (2025).
https://doi.org/10.48550/arXiv.2412.15204, http://arxiv.org/abs/2412.15204
5. Brehme, L., Dornauer, B., Ströhle, T., Ehrhart, M., Breu, R.: Retrieval-
augmented generation in industry: An interview study on use cases, require-
ments, challenges, and evaluation. https://doi.org/10.48550/arXiv.2508.14066,
http://arxiv.org/abs/2508.14066
6. Brehme, L., Ströhle, T., Breu, R.: Can LLMs be trusted
for evaluating RAG systems? a survey of methods and
datasets (2025). https://doi.org/10.1109/SDS66131.2025.00010,
https://ieeexplore.ieee.org/document/11081490, ISSN: 2835-3420
7. Brown, T.B., Mann, B., Ryder, N., Subbiah, M., Kaplan, J., Dhariwal, P., Nee-
lakantan, A., Shyam, P., Sastry, G., Askell, A., Agarwal, S., Herbert-Voss, A.,
Krueger, G., Henighan, T., Child, R., Ramesh, A., Ziegler, D.M., Wu, J., Win-
ter, C., Hesse, C., Chen, M., Sigler, E., Litwin, M., Gray, S., Chess, B., Clark,
J., Berner, C., McCandlish, S., Radford, A., Sutskever, I., Amodei, D.: Language
models are few-shot learners (2020). https://doi.org/10.48550/arXiv.2005.14165,
http://arxiv.org/abs/2005.14165
8. DeepMind, G.: Gemini models | gemini API (2025), https://ai.google.dev/gemini-
api/docs/models
9. Ding,T.,Banerjee,A.,Mombaerts,L.,Li,Y.,Borogovac,T.,Weinstein,J.P.D.l.C.:
VERA: Validation and evaluation of retrieval-augmented systems (2024).
https://doi.org/10.48550/arXiv.2409.03759, http://arxiv.org/abs/2409.03759
10. Efron, B., Tibshirani, R.J.: An Introduction to the Bootstrap. Chapman and Hal-
l/CRC (1994). https://doi.org/10.1201/9780429246593
11. Es, S., James, J., Espinosa-Anke, L., Schockaert, S.: RAGAS:
Automated evaluation of retrieval augmented generation (2023).
https://doi.org/10.48550/arXiv.2309.15217, http://arxiv.org/abs/2309.15217
12. Friel, R., Belyi, M., Sanyal, A.: RAGBench: Explainable
benchmark for retrieval-augmented generation systems (2024).
https://doi.org/10.48550/arXiv.2407.11005, http://arxiv.org/abs/2407.11005

14 Brehme et al.
13. Gao, Y., Xiong, Y., Gao, X., Jia, K., Pan, J., Bi, Y., Dai, Y., Sun,
J., Wang, M., Wang, H.: Retrieval-augmented generation for large lan-
guage models: A survey (2024). https://doi.org/10.48550/arXiv.2312.10997,
http://arxiv.org/abs/2312.10997
14. Kukreja, S., Kumar, T., Bharate, V., Purohit, A., Dasgupta, A., Guha,
D.: Performance evaluation of vector embeddings with retrieval-augmented
generation (2024). https://doi.org/10.1109/ICCCS61882.2024.10603291,
https://ieeexplore.ieee.org/document/10603291
15. Lewis, P., Perez, E., Piktus, A., Petroni, F., Karpukhin, V., Goyal, N.,
Küttler, H., Lewis, M., Yih, W.t., Rocktäschel, T., Riedel, S., Kiela, D.:
Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks (Apr 2021),
http://arxiv.org/abs/2005.11401, arXiv:2005.11401
16. Li, T., Zhang, G., Do, Q.D., Yue, X., Chen, W.: Long-context LLMs struggle
with long in-context learning (2024). https://doi.org/10.48550/arXiv.2404.02060,
http://arxiv.org/abs/2404.02060
17. Liu, J., Ding, R., Zhang, L., Xie, P., Huang, F.: CoFE-RAG: A com-
prehensive full-chain evaluation framework for retrieval-augmented generation
with enhanced data diversity (2024). https://doi.org/10.48550/arXiv.2410.12248,
http://arxiv.org/abs/2410.12248
18. Lorenz Brehme, Thomas Ströhle, R.B.: General·lorenzbrehme/CARE,
https://github.com/lorenzbrehme/CARE
19. Moreira, G.d.S.P., Ak, R., Schifferer, B., Xu, M., Osmulski, R., Oldridge, E.:
Enhancing q&a text retrieval with ranking models: Benchmarking, fine-tuning
and deploying rerankers for RAG. https://doi.org/10.48550/arXiv.2409.07691,
http://arxiv.org/abs/2409.07691
20. OpenAI: Model - OpenAI API (2025), https://platform.openai.com
21. Rackauckas, Z., Câmara, A., Zavrel, J.: Evaluating RAG-
fusion with RAGElo: an automated elo-based framework (2024).
https://doi.org/10.48550/arXiv.2406.14783, http://arxiv.org/abs/2406.14783
22. Rajpurkar, P., Jia, R., Liang, P.: Know what you don’t know: Unanswer-
able questions for SQuAD (2024). https://doi.org/10.48550/arXiv.1806.03822,
http://arxiv.org/abs/1806.03822
23. Saad-Falcon, J., Khattab, O., Potts, C., Zaharia, M.: ARES: An auto-
mated evaluation framework for retrieval-augmented generation systems (2024).
https://doi.org/10.48550/arXiv.2311.09476, http://arxiv.org/abs/2311.09476
24. Salemi, A., Zamani, H.: Evaluating retrieval quality in retrieval-
augmented generation (2024). https://doi.org/10.48550/arXiv.2404.13781,
http://arxiv.org/abs/2404.13781
25. Tang, Y., Yang, Y.: MultiHop-RAG: Benchmarking retrieval-augmented gener-
ation for multi-hop queries (2024). https://doi.org/10.48550/arXiv.2401.15391,
http://arxiv.org/abs/2401.15391
26. Trivedi, H., Balasubramanian, N., Khot, T., Sabharwal, A.: MuSiQue: Multihop
questions via single-hop question composition. Transactions of the Association for
Computational Linguistics (2022)
27. Trotman, A., Puurula, A., Burgess, B.: Improvements to BM25 and lan-
guage models examined. In: Proceedings of the 19th Australasian Doc-
ument Computing Symposium. pp. 58–65. ADCS ’14, Association for
Computing Machinery (2014). https://doi.org/10.1145/2682862.2682863,
https://dl.acm.org/doi/10.1145/2682862.2682863

Multi-Hop Retriever Evaluation 15
28. Wei, J., Wang, X., Schuurmans, D., Bosma, M., Ichter, B., Xia, F.,
Chi, E., Le, Q., Zhou, D.: Chain-of-thought prompting elicits reasoning
in large language models (2023). https://doi.org/10.48550/arXiv.2201.11903,
http://arxiv.org/abs/2201.11903
29. Wu, N., Gong, M., Shou, L., Liang, S., Jiang, D.: Large language
models are diverse role-players for summarization evaluation (2023).
https://doi.org/10.48550/arXiv.2303.15078, http://arxiv.org/abs/2303.15078
30. Xu, L., Lian, J., Zhao, W.X., Gong, M., Shou, L., Jiang, D., Xie, X.,
Wen, J.R.: Negative sampling for contrastive representation learning: A review.
https://doi.org/10.48550/arXiv.2206.00212, http://arxiv.org/abs/2206.00212
31. Yang, Z., Qi, P., Zhang, S., Bengio, Y., Cohen, W.W., Salakhutdi-
nov, R., Manning, C.D.: HotpotQA: A dataset for diverse, explainable
multi-hop question answering (2018). https://doi.org/10.48550/arXiv.1809.09600,
http://arxiv.org/abs/1809.09600