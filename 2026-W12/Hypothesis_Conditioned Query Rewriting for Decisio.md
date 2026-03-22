# Hypothesis-Conditioned Query Rewriting for Decision-Useful Retrieval

**Authors**: Hangeol Chang, Changsun Lee, Seungjoon Rho, Junho Yeo, Jong Chul Ye

**Published**: 2026-03-19 15:15:58

**PDF URL**: [https://arxiv.org/pdf/2603.19008v1](https://arxiv.org/pdf/2603.19008v1)

## Abstract
Retrieval-Augmented Generation (RAG) improves Large Language Models (LLMs) by grounding generation in external, non-parametric knowledge. However, when a task requires choosing among competing options, simply grounding generation in broadly relevant context is often insufficient to drive the final decision. Existing RAG methods typically rely on a single initial query, which often favors topical relevance over decision-relevant evidence, and therefore retrieves background information that can fail to discriminate among answer options. To address this issue, here we propose Hypothesis-Conditioned Query Rewriting (HCQR), a training-free pre-retrieval framework that reorients RAG from topic-oriented retrieval to evidence-oriented retrieval. HCQR first derives a lightweight working hypothesis from the input question and candidate options, and then rewrites retrieval into three targeted queries that seek evidence to: (1) support the hypothesis, (2) distinguish it from competing alternatives, and (3) verify salient clues in the question. This approach enables context retrieval that is more directly aligned with answer selection, allowing the generator to confirm or overturn the initial hypothesis based on the retrieved evidence. Experiments on MedQA and MMLU-Med show that HCQR consistently outperforms single-query RAG and re-rank/filter baselines, improving average accuracy over Simple RAG by 5.9 and 3.6 points, respectively. Code is available at https://anonymous.4open.science/r/HCQR-1C2E.

## Full Text


<!-- PDF content starts -->

Hypothesis-Conditioned Query Rewriting for Decision-Useful Retrieval
Hangeol Chang1Changsun Lee1Seungjoon Rho2Junho Yeo3Jong Chul Ye1
1Graduate School of AI, KAIST, Republic of Korea
2School of Electrical Engineering, KAIST, Republic of Korea
3Department of Industrial Engineering, Yonsei University, Republic of Korea
{ hangeol, sunny17, ryzere }@kaist.ac.kr
asap03153@yonsei.ac.kr jong.ye@kaist.ac.kr
Abstract
Retrieval-Augmented Generation (RAG) im-
proves Large Language Models (LLMs)
by grounding generation in external, non-
parametric knowledge. However, when a task
requires choosing among competing options,
simply grounding generation in broadly rele-
vant context is often insufficient to drive the
final decision. Existing RAG methods typ-
ically rely on a single initial query, which
often favors topical relevance over decision-
relevant evidence, and therefore retrieves back-
ground information that can fail to discrimi-
nate among answer options. To address this is-
sue, here we proposeHypothesis-Conditioned
Query Rewriting (HCQR), a training-free pre-
retrieval framework that reorients RAG from
topic-oriented retrieval to evidence-oriented
retrieval. HCQR first derives a lightweight
working hypothesis from the input question
and candidate options, and then rewrites re-
trieval into three targeted queries that seek ev-
idence to: (1) support the hypothesis, (2) dis-
tinguish it from competing alternatives, and (3)
verify salient clues in the question. This ap-
proach enables context retrieval that is more
directly aligned with answer selection, allow-
ing the generator to confirm or overturn the
initial hypothesis based on the retrieved evi-
dence. Experiments on MedQA and MMLU-
Med show that HCQR consistently outperforms
single-query RAG and re-rank/filter baselines,
improving average accuracy over Simple RAG
by 5.9 and 3.6 points, respectively. Code
is available at https://anonymous.4open.
science/r/HCQR-1C2E.
1 Introduction
Retrieval-Augmented Generation (RAG) is moti-
vated by the premise that external retrieval can
improve generation when a model lacks the knowl-
edge required to answer a question (Lewis et al.,
2020). In practice, however, retrieval is notuniformly helpful. On many contemporary QA
benchmarks, strong language models can answer
a substantial fraction of questions from paramet-
ric knowledge alone. In such settings, retrieval
is useful only when the retrieved context materi-
ally improves the final decision. Prior work has
accordingly emphasized that RAG performance de-
pends on design choices across the full pipeline,
rather than on retrieval as an inherently beneficial
component (Fan et al., 2024; Huang and Huang,
2024). At the same time, empirical studies show
that weakly aligned context can actively degrade
downstream reasoning: irrelevant passages can dis-
tract the model, robustness declines in the presence
of irrelevant context, and increasing context length
can hurt performance even when relevant evidence
is available (Amiraz et al., 2025; Yoran et al., 2023;
Du et al., 2025).
These observations suggest that retrieval qual-
ity should be evaluated in terms of its contribution
to the final decision rather than topical relevance
alone. We therefore study optional-retrieval deci-
sion settings, where retrieval remains available but
not always necessary, and the model must choose
among competing possibilities. In this regime, the
relevant question is whether the retrieved context
isdecision-useful: whether it provides a criterion,
contrast, or feature-grounded fact that supports one
answer, distinguishes leading candidates, or veri-
fies a salient clue in the question.
This perspective is consistent with recent work
that evaluates retrieval quality in terms of whether
the retrieved context is sufficient for downstream
generation rather than merely relevant under
surface-level matching (Salemi and Zamani, 2024;
Joren et al., 2024). It also clarifies why naive re-
trieval can be harmful. If retrieval is driven primar-
ily by lexical overlap or broad topical similarity, it
may return generic background passages that are
thematically related but uninformative for the deci-
sion boundary. Under a bounded generator-contextarXiv:2603.19008v1  [cs.CL]  19 Mar 2026

-Working Hypothesis ෝ𝒚𝟎
-Key/discriminating feature 𝒇
-Expected supporting evidence 𝒆Hypothesis formulation Retrieval
-Retrieve documents C
-Find evidence
-Evidence supporting hypothesis 𝒒𝟏
-Evidence distinguishing alternative 𝒒𝟐
-Key features relevant to the question 𝒒𝟑Query Rewriting
Figure 1:HCQR pipeline.A hypothesis formulator proposes a lightweight working hypothesis together with the
evidence expected under that hypothesis. The query rewriter turns this state into three verification-oriented queries.
The working hypothesis steers retrieval but is not shown to the generator by default.
budget, such passages occupy scarce evidence slots,
displace stronger evidence, and encourage weakly
grounded reasoning. The central difficulty is there-
fore not only missing knowledge, but also mis-
matched evidence.
This framing also helps position existing
retrieval-improvement methods. Query transfor-
mation approaches, such as query rewriting and
hypothetical-document retrieval, primarily aim to
improve recall or semantic match (Ma et al., 2023;
Gao et al., 2023). Other work introduces reasoning-
guided or agentic retrieval, in which retrieval de-
cisions are interleaved with generation or self-
reflection (Trivedi et al., 2023; Asai et al., 2023;
Liang et al., 2025). Post-retrieval methods instead
focus on reranking, evaluating, or filtering an ini-
tial retrieved pool (Yu et al., 2024; Yan et al., 2024;
Chang et al., 2025).
Our work is more focused. We isolate one
specific subproblem: once a model has formed
a working answer hypothesis, can that interme-
diate state be used to construct betterfirst-pass
retrieval queries under a shared maximum final-
context budget? To address this question, we pro-
poseHypothesis-Conditioned Query Rewriting
(HCQR), a lightweight two-stage module for pre-
retrieval planning. Specifically, HCQR first uses a
hypothesis formulatorto read the question and can-
didate answers, propose a working answer hypoth-
esis, and identify the clues and evidence needed
to verify it. It then uses aquery rewriterto con-
vert that structured state into exactly three targeted
queries: one that retrieves evidence supporting the
working hypothesis, one that retrieves evidence
distinguishing it from strong alternatives, and one
that retrieves evidence verifying key clues from thequestion. By default, the working hypothesis is
used only to steer retrieval and is not exposed to
the final generator. This keeps the method retrieval-
side by construction and reduces direct answer an-
choring.
We instantiate HCQR primarily in medical
multiple-choice QA. Medicine is not unique in re-
quiring decision-useful evidence, but it provides a
particularly sharp testbed: questions often hinge
on diagnostic criteria, differential comparisons, or
the interpretation of question-specific clues, while
generic disease descriptions may remain topically
relevant yet non-decisive. On both MedQA and
MMLU-Med, simple single-query RAG underper-
forms no-retrieval CoT in every reported model
configuration. Accordingly, prior medical RAG
studies similarly show that retrieval design, refor-
mulation, and iterative follow-up retrieval can sub-
stantially affect performance (Xiong et al., 2024a,b;
Kim et al., 2025).
Even compared with the latest models, HCQR is
the strongest method in every dataset–model cell,
improving average accuracy over Simple RAG by
5.9 points on MedQA and 3.6 points on MMLU-
Med, while also substantially increasing the share
of retrieved context sets judged decision-useful.
Our contribution can be summarized as follows.
1.We introduceHCQR, a training-free pre-
retrieval method that forms a working answer
hypothesis and converts it into three comple-
mentary retrieval queries.
2.We develop acontext-utility analysisfor re-
trieved context sets, using the labels ENTAILED,
USEFUL-BUT-NOT-ENTAILED, and NOTUSE-
FUL, together with a Decision-Useful Rate and

same-subset gains over no-retrieval CoT.
3.We show on medical multiple-choice QA
that query-side decision alignment outperforms
naive single-query RAG, query-transformation
baselines, reranking, and retrieve-then-filter
baselines. Option-exposure controls and query
ablations further clarify which components of
the pipeline drive the gains.
2 Related Work
RAG design and harmful context.RAG is a
standard framework for grounding language mod-
els in external knowledge (Lewis et al., 2020). Prior
surveys organize RAG systems into pre-retrieval,
retrieval, post-retrieval, and generation stages (Fan
et al., 2024; Huang and Huang, 2024). HCQR
belongs to the pre-retrieval stage. This focus is
motivated by evidence that additional context is
not uniformly helpful: irrelevant passages can dis-
tract the model, robustness declines under poorly
aligned context, and longer contexts can impair
reasoning even when some relevant evidence is
present (Amiraz et al., 2025; Yoran et al., 2023; Du
et al., 2025).
Query rewriting and query transformation.A
direct way to improve retrieval is to transform the
input query. Prior works include query rewriting
as a retrieval module (Ma et al., 2023), paraphrase-
style multi-query retrieval and fusion (Rackauckas,
2024), hypothetical-document approaches such as
HyDE (Gao et al., 2023), etc. These methods gen-
erally aim to improve recall or semantic match. By
contrast, HCQR uses a structured intermediate anal-
ysis of the answer options to construct first-pass
queries that target evidence useful for the down-
stream decision.
Reasoning-guided and agentic retrieval.A
broader family of methods treats retrieval as part
of an adaptive reasoning process. Examples in-
clude IRCoT, Self-RAG, and related reasoning- or
agent-driven RAG systems (Trivedi et al., 2023;
Asai et al., 2023; Liang et al., 2025). HCQR is not
a replacement for these systems. Instead, it isolates
a narrower subproblem that often appears within
them: how to construct better retrieval queries once
a working answer hypothesis is available.
Post-retrieval selection and retrieval utility.
Another major line of work improves RAG after re-
trieval through reranking, evaluation, or filtering ofan initial candidate pool, as in RankRAG, CRAG,
and MAIN-RAG (Yu et al., 2024; Yan et al., 2024;
Chang et al., 2025). We compare against such meth-
ods as baselines, but HCQR itself remains purely
pre-retrieval. Recent work also argues that retrieval
should be evaluated by downstream utility rather
than topical relevance alone (Salemi and Zamani,
2024; Joren et al., 2024). Our context-utility analy-
sis follows this perspective in a decision setting.
Medical RAG as a representative testbed.Med-
ical QA is an important empirical setting for study-
ing retrieval quality. Prior work shows that medical
RAG performance is sensitive to corpus choice,
retriever design, and retrieval strategy, and that it-
erative retrieval, reformulation, or filtering can ma-
terially affect performance (Xiong et al., 2024a,b;
Kim et al., 2025). These findings support our use
of medical QA as a domain in which the gap be-
tween topical relevance and decision usefulness is
especially visible.
3 Method
3.1 Problem Setting
We consider optional-retrieval decision settings, in
which the model must choose among competing
answers while retrieval remains optional. In our
experiments, these competing answers are the ob-
served multiple-choice options, but the same struc-
ture arises more generally in decision tasks where
the model selects among a small set of plausible
hypotheses and external evidence is useful only
when it sharpens that choice.
Letxdenote the question, A={a 1, . . . , a K}
the candidate answers, and Dthe corpus. A re-
trieval method mconstructs a final context set
Cm(x, A)⊆ D for the generator under a maxi-
mum final-context budgetB max:
|Cm(x, A)| ≤B max
The no-retrieval baseline corresponds to
Cm(x, A) =∅ . We treat Bmax as a cap on
the evidence shown to the final generator rather
than as an exact-fill requirement or a bound
on intermediate retrieval steps. Methods may
therefore inspect different numbers of intermediate
candidates, while the realized final context may
contain fewer than Bmaxdocuments because of
deduplication or method-specific selection.
In this regime, retrieval is useful only when the
bounded final context materially improves the de-
cision. The key distinction is therefore between

topically relevantevidence anddecision-usefulevi-
dence. Retrieved context is decision-useful when it
narrows the answer space, for example by provid-
ing a criterion or fact that supports one answer, dis-
tinguishes leading alternatives, or verifies a salient
clue from the question stem. By contrast, back-
ground exposition or keyword overlap is insuffi-
cient if it does not help discriminate among the
candidate answers.
We therefore frame retrieval planning as
hypothesis-guided verification. A first-pass rea-
soner proposes a working answer hypothesis like
“suppose the answer is aj.” That hypothesis is not
treated as the final answer. Instead, it specifies
what should hold if ajwere correct: what evidence
should support it, which alternatives should be
ruled out, and which stem clues should be checked.
Retrieval should target those verification needs.
3.2 HCQR: Overview
HCQR is a training-free, hypothesis-guided query
rewriting framework for retrieval-augmented ques-
tion answering under a bounded final-context bud-
get. Its central idea is to use a tentative answer
not as a final prediction, but as a control signal
for retrieval: the system first forms a working hy-
pothesis, then infers what evidence should hold if
that hypothesis were correct, and finally retrieves
documents that test those expectations.
The framework has three components:
1.Hypothesis formulator, which produces a com-
pact structured working hypothesis;
2.Query rewriter, which converts that hypothesis
into three targeted retrieval queries;
3.Retriever and fusion, which gather evidence
for those queries and merge it into a final context
under the shared budget;
More specifically, the hypothesis formulator H
first constructs the hypothesis state
h=H(x, A) = (ˆy 0, f, e, r),
where ˆy0is the working hypothesis, fdenotes
discriminating features, edenotes confirming evi-
dence, and rdenotes brief reasoning. Conditioned
on the original question xand this state h, the query
rewriterQproduces three queries,
(q1, q2, q3) =Q(x, h).
The shared retriever Rretrieves evidence for these
queries, and the retrieved results are fused into afinal contextC,
C= Fuse 
R(q1), R(q 2), R(q 3)
.
Based on this context, the downstream generator G
predicts the final answer as
ˆy=G(x, A, C).(1)
Crucially, the hypothesis influences retrieval only
through the rewritten queries and is not passed di-
rectly to the generator in the default setup. Figure 1
provides a visual overview of the main steps and
the associated notation.
3.3 Stage 1: Hypothesis Formulator
The first stage is a hypothesis formulator. Given
the question stem and answer options, it proposes
a working answer hypothesis and identifies the evi-
dence needed to verify it. The purpose of this stage
is not to commit to a final answer, but to define a
concrete verification target for retrieval. The hy-
pothesis formulator outputs a compact structured
record containing:
•working hypothesis ˆy0: the current best-guess
option;
•discriminating features f: question-specific
clues that separate the leading candidates;
•confirming evidence e: facts, rules, or condi-
tions that should be observed if the working hy-
pothesis is correct; and
•brief reasoning r: a concise explanation for why
this hypothesis is currently preferred.
We keep this state short and structured because
HCQR uses it to steer retrieval rather than to ex-
pose a long reasoning trace to the generator. What
matters is the verification agenda implied by the
tentative answer.
3.4 Stage 2: Query Rewriter
Conditioned on the original question, the an-
swer options, and the structured hypothesis state
(ˆy0, f, e, r) , the query rewriter generates exactly
three high-precision search queries, which we now
name as SUPPORT, DISTINCTION, KEY FEA-
TURES:
•SUPPORT q1: supports the working hypoth-
esis.This query uses the hypothesized answer
text, confirming evidence and brief reasoning to
retrieve passages that would directly support the
current hypothesis.

•DISTINCTION q2: distinguishes leading al-
ternatives.This query uses the discriminating
features and candidate answers to retrieve criteria
that separate the strongest competing options.
•KEY FEATURES q3: verify key clues from
the stem.This query targets salient clues in
the question itself and seeks evidence needed
for verification, even if the current hypothesis is
incomplete or partially wrong.
Together, these queries realize the intended re-
trieval logic: retrieve evidence that should hold if
the current hypothesis is correct, retrieve evidence
that separates it from competing alternatives, and
retrieve evidence that verifies the question’s own
clues. q1is hypothesis-supporting, q2is contrastive,
andq 3is clue-verifying.
3.5 Retriever, Fusion, and Generation
The three rewritten queries are passed to a shared
retriever, and the resulting evidence is fused un-
der the final-context budget Bmax. In the multi-
query setting, retrieval is performed independently
for each query, after which the retrieved sets are
merged and deduplicated. The resulting fused con-
textCis the only retrieved evidence shown to the
generator. Then, the generator predict the final
answer as shown in Eq. (1).
4 Experimental Setup
Medical multiple-choice QA provides a particu-
larly clean instance of this setting. Many questions
are solvable from memory, but retrieval can still
help when it surfaces diagnostic criteria, differen-
tial comparisons, or grounded interpretations of
question-specific clues. This makes the domain es-
pecially suitable for studying the gap between topi-
cal relevance and decision usefulness, even though
the broader formulation of HCQR is not limited to
medicine.
4.1 Benchmarks
We evaluate HCQR on two medical multiple-choice
QA benchmarks:MedQA(Jin et al., 2021) and
MMLU-Med(Hendrycks et al., 2020). These
benchmarks provide a controlled setting for evalu-
ating retrieval-augmented decision making under
multiple-choice supervision.
4.2 Models
The main experiments use four generator configura-
tions: Llama3.2-3B-instruct, Llama3.1-8B-instruct,Qwen3-4B-instruct-2507, and Qwen3-30B-A3B-
instruct-2507. For each method, all LLM-based
components use the same model.
4.3 Compared Methods
We compare HCQR against the following base-
lines:
•NO-RAG (COT): answer directly without re-
trieval.
•SIMPLERAG: retrieve with the raw question as
a single query and pass the top-15 documents
directly to the generator.
•REWRITING: paraphrase-style multi-query
rewriting followed by retrieval fusion. We use the
same three-query prompt family as the LC-MQR
baseline in prior work (Kuo et al., 2025).
•HYDE: a strong query-transformation baseline
based on hypothetical-document retrieval (Gao
et al., 2023). We generate 8 hypothetical docu-
ments for each question, encode them with the
retriever, average their embedding vectors, and
use the averaged vector as the retrieval query
representation.
•RERANK-RAG: first retrieves a candidate pool
with the raw question, then reranks documents
with a BGE-M3 reranker to better match the
query before selecting the final context.
•MAIN-RAG: first retrieves documents with
the raw question, generates a tentative answer
from each question-document pair, and then
uses an LLM to evaluate the resulting question-
document-answer triples and filter documents
accordingly (Chang et al., 2025).
4.4 Retriever, Corpus, and Budget
All retrieval-based methods use the same retriever
and corpus so that performance differences reflect
retrieval strategy rather than backend changes. We
use MedCPT (Jin et al., 2023) as the retriever over a
medical textbook corpus (Xiong et al., 2024a). The
maximum final-context budget is set to Bmax= 15
documents.
We compare methods under a shared final
generator-context budget. Single-query methods
without post-retrieval selection (SIMPLERAG and
HYDE) retrieve the top 15 documents directly.
Multi-query methods (REWRITINGand HCQR)
retrieve the top 5 documents independently for
each query, take the union of the retrieved sets

and remove duplicates. Because retrieved sets
may overlap, the realized final context can con-
tain fewer than 15 unique documents. RERANK-
RAG retrieves 150 candidates with the raw ques-
tion, reranks them, and passes the top 15 to the
generator. MAIN-RAG retrieves 15 documents
with the raw question and then applies LLM-based
filtering; its realized final context may therefore
also contain fewer than 15 documents. We retain
these method-specific candidate-pool rules because
they are part of the respective baselines. The shared
comparison criterion is the evidence budget at the
final generator.
4.5 Prompting and Implementation Details
For final answer generation, all methods use the
same MIRAGE prompt (Xiong et al., 2024a). This
keeps the generation interface fixed across methods,
so differences in performance are attributable pri-
marily to the retrieval strategy. Prompt templates
and other implementation details are provided in
the appendix.
4.6 Context-Utility Evaluation
Accuracy alone does not distinguish improvements
due to better retrieved evidence from improvements
caused by other changes in generation behavior.
We therefore evaluate retrieved context sets directly
using a separate LLM judge. For each question,
method, and model setting, the judge first predicts
the answer option most strongly supported by the
retrieved context alone ( selected_answer ) and
then assigns a utility label describing how that con-
text relates to the gold answer.
The judge uses three labels. For brevity, we
abbreviate the prompt label USEFUL-BUT-NOT-
ENTAILEDas USEFULin tables and formulas.
•ENTAILED: the gold answer follows decisively
from the retrieved context alone, with no missing
fact or outside knowledge, and the context-only
selected_answermatches the gold answer.
•USEFUL: the retrieved context provides strong
decision-relevant support for the gold answer but
lacks one final inferential bridge. This includes
cases where the context contains a decisive clue,
rules out plausible alternatives, or makes the gold
answer reachable with minimal additional knowl-
edge.
•NOTUSEFUL: the retrieved context does not pro-
vide strong decision-relevant support for the gold
answer. This includes background-only passages,keyword overlap without meaningful narrowing
power, weak associations, and cases where the
context supports another answer more strongly
than the gold answer.
Specifically, let Nddenote the number of questions
in dataset d, and let NE(d, m, g) andNU(d, m, g)
be the numbers labeled ENTAILEDand USEFUL,
respectively, under methodmand model familyg.
Then,Decision-Useful Rate(DUR) is defined as
DUR(d, m, g) =NE(d, m, g) +N U(d, m, g)
Nd.
DUR summarizes the fraction of questions for
which retrieval yields context that is either deci-
sive or meaningfully helpful for the correct answer.
We also report accuracy within each utility cate-
gory. To test whether these labels correspond to
downstream performance differences, we compare
each method against no-retrieval CoT on the same
subset of questions.:
∆CoT(d, m, g|ℓ) = Acc(d, m, g|ℓ)
−Acc(d,CoT, g|ℓ).
where the second term evaluates the same model’s
no-retrieval system on the subset of questions
whose retrieved context is assigned label ℓunder
method m. This same-subset comparison controls
for differences in subset composition.
The context-utility evaluation is conducted using
GPT-4.1 as the judge model, and the full prompt
and implementation details are provided in the Ap-
pendix.
5 Results
5.1 Main accuracy results
Table 1 reports answer accuracy under the shared
15-document final-context budget. Two findings
are immediate. First, SIMPLERAG underperforms
no-retrieval COT in every dataset-model cell, in-
dicating that retrieval based on the raw question
alone does not reliably improve answer selection
in the studied regime. Second, HCQR achieves
the highest accuracy in every reported cell. Aver-
aged across models, HCQR improves over SIMPLE
RAG by 5.9 points on MedQA and 3.6 points on
MMLU-Med, and also outperforms no-retrieval
COT on both datasets. Outside HCQR, gains are
limited. Query-side baselines are generally more
effective than post-retrieval filtering: REWRITING
and HYDE are both more competitive than MAIN-
RAG, while MAIN-RAG remains below COT on

Table 1: Accuracy (%) on MedQA and MMLU-Med for HCQR and baseline methods under the shared 15-document
final-context budget. HCQR achieves the best performance in every reported dataset-model cell.
MedQA MMLU-Med
MethodLlama3.2 3B Llama3.1 8B Qwen3 4B Qwen3 30BAverageLlama3.2 3B Llama3.1 8B Qwen3 4B Qwen3 30BAverage
CoT 58.0 69.0 72.8 84.1 71.0 66.3 75.0 82.9 90.2 78.6
Simple RAG 55.6 66.8 71.3 82.7 69.1 64.5 74.7 81.8 88.9 77.5
Rerank-RAG 56.8 67.3 72.1 83.3 69.9 67.1 75.8 82.3 90.1 78.8
Rewriting 58.9 67.5 73.1 83.3 70.7 67.0 74.7 82.5 88.8 78.2
HyDE 61.0 69.3 71.4 82.9 71.1 67.8 76.5 82.6 89.4 79.1
MAIN-RAG 55.9 65.8 70.9 83.6 69.0 67.0 75.7 81.2 88.2 78.0
HCQR (ours) 63.4 73.0 77.9 85.8 75.0 68.6 79.7 85.4 90.6 81.1
Table 2: Decision-Useful Rate (ENTAILED+ USEFUL) on MedQA and MMLU-Med for HCQR and baseline
methods. HCQR yields the highest rate of decision-useful retrieved contexts across all reported dataset-model cells.
MedQA MMLU-Med
MethodLlama3.2 3B Llama3.1 8B Qwen3 4B Qwen3 30BAverageLlama3.2 3B Llama3.1 8B Qwen3 4B Qwen3 30BAverage
Simple RAG 30.0 30.0 30.0 30.0 30.0 48.7 48.7 48.7 48.7 48.7
Rerank-RAG 44.1 44.1 44.1 44.1 44.1 56.8 56.8 56.8 56.8 56.8
Rewriting 45.2 51.2 62.2 65.6 56.1 51.4 53.7 56.7 59.1 55.3
HyDE 51.6 54.5 52.8 54.6 53.4 43.0 46.2 44.3 46.3 44.9
MAIN-RAG 23.7 26.4 24.7 26.7 25.4 35.6 36.7 33.7 36.5 35.6
HCQR (ours) 73.4 81.8 83.2 89.8 82.1 65.8 69.1 72.5 73.7 70.3
Table 3: Average context-utility distribution across
MedQA and MMLU-Med (%). Values are averaged
over the two datasets and four model configurations.
USEFULabbreviates USEFUL-BUT-NOT-ENTAILED.
Method Entailed Useful Not Useful
Simple RAG 14.3 25.1 60.7
Rerank-RAG 19.9 30.6 49.5
Rewriting 19.6 36.1 44.3
HyDE 16.6 32.5 50.8
MAIN-RAG 11.6 18.9 69.5
HCQR (ours)34.7 41.5 23.8
Table 4: Utility-conditioned accuracy and same-subset
gain over no-retrieval CoT. Results are averaged over
two datasets, four models, and six methods under the
configurations of Table 2. Each cell is reported asaccu-
racy /∆ CoT.
Label MedQA MMLU-Med
ENTAILED94.4 / +8.2 95.4 / +5.1
USEFUL77.9 / +4.2 83.9 / +1.3
NOTUSEFUL57.6 / -5.8 70.2 / -3.2
average. The gains are larger on MedQA than on
MMLU-Med, and are more pronounced for smaller
and mid-sized models.
5.2 Context-utility analysis
The context-utility analysis is consistent with the
accuracy results and makes the same pattern clearer.
Table 2 shows that HCQR substantially increases
the share of retrieved context sets judged decision-useful, with especially large gains on MedQA.
Among the non-HCQR baselines, query-side meth-
ods again appear more favorable than post-retrieval
filtering, but the pattern is not uniform: REWRIT-
INGmore clearly improves decision-useful re-
trieval, whereas HYDE is more competitive in an-
swer accuracy than in DUR. By contrast, MAIN-
RAG yields the weakest DUR results, suggesting
that filtering after an initially weak first-pass re-
trieval does not reliably recover decision-useful
evidence.
Table 3 shows how this improvement is dis-
tributed across utility labels. HCQR achieves the
highest rates of both ENTAILEDand USEFULre-
trieval while sharply reducing the share of NOT
USEFULcontexts. Table 4 further shows that these
labels align with downstream answer behavior: the
ENTAILEDsubset yields the highest answer accu-
racy and the largest positive gain over no-retrieval
COT, the USEFULsubset remains beneficial but to
a smaller extent, and the NOTUSEFULsubset falls
below the no-retrieval baseline. Overall, these re-
sults suggest that in optional-retrieval decision set-
tings, it is crucial to steer retrieval toward decision-
relevant evidence from the start rather than rely on
post-hoc selection alone.
5.3 Controls and ablations
We next examine which components of HCQR are
responsible for the observed gains.
Table 5 evaluates whether improvements arise

Table 5: Diagnostic control for answer-option exposure.
Method MedQA MMLU-Med
Rewriting 70.7 78.2
Rewriting+options 71.3 79.2
HCQR−options 73.1 79.9
HCQR 75.0 81.1
Table 6: HCQR query-component ablation.
Method MedQA MMLU-Med
HCQR 75.0 81.1
HCQR−q 1(Support) 72.4 79.0
HCQR−q 2(Distinction) 73.2 80.1
HCQR−q 3(Key Features) 73.0 80.5
primarily from exposing answer options during
retrieval planning. Adding answer options to a
generic multi-query rewriter yields only a mod-
est gain over rewriting without options, whereas
HCQR remains clearly stronger. This indicates that
the benefit of HCQR is not explained by answer-
option exposure alone.
Table 6 evaluates the contribution of the three
rewritten queries. Removing any one of the query
roles reduces accuracy on both datasets. The largest
decrease follows removal of q1, the answer-support
query, but removing q2orq3also degrades perfor-
mance. These results indicate that the three query
roles provide complementary contributions.
6 Discussion
The results support the view that HCQR is ef-
fective because it changes which evidence enters
the bounded final context. Rather than retrieving
broadly from the raw question alone, HCQR first
forms a working answer hypothesis and then re-
trieves evidence that can support that hypothesis,
distinguish it from alternatives, and verify salient
clues from the question stem. The accompany-
ing shift from NOTUSEFULretrieval toward EN-
TAILED/USEFULretrieval is consistent with this
mechanism.
The comparison with post-retrieval baselines
clarifies why first-pass query construction matters.
RERANK-RAG inspects a substantially larger can-
didate pool than HCQR, yet remains weaker un-
der the same final-context budget. MAIN-RAG
applies LLM-based filtering after raw-question re-
trieval, but also remains below HCQR. Taken to-
gether, these comparisons suggest that post-hoc
selection is constrained by the quality of the initial
candidate pool: if the first-pass query does not re-trieve decision-useful evidence, later reranking or
filtering can operate only on suboptimal candidates.
The dataset-level pattern points in the same
direction. HCQR improves both MedQA and
MMLU-Med, but the gains are more pronounced
on MedQA in both accuracy and DUR. A plausible
interpretation is that MedQA contains more ques-
tions in which a small number of clinical clues,
criteria, or option-level contrasts determine the cor-
rect answer, making retrieval quality more sensi-
tive to decision alignment. We present this as an
interpretation of the observed trend rather than a
definitive characterization of the datasets.
More broadly, the findings suggest that in
optional-retrieval decision settings, retrieval qual-
ity should be evaluated by its effect on the down-
stream decision rather than by topical relevance
alone. Under a bounded context budget, better
first-pass query construction can matter more than
inspecting larger intermediate candidate pools or
filtering a poorly targeted initial retrieval set.
7 Conclusion
We introduced HCQR, a training-free hypothesis-
guided query rewriting method for optional-
retrieval decision settings under a bounded final-
context budget. HCQR uses a working answer
hypothesis to plan retrieval, generating targeted
queries that retrieve evidence to support the hypoth-
esis, distinguish it from competing alternatives, and
verify salient clues from the question stem. The
working hypothesis is used only for retrieval plan-
ning, while the final answer is produced from the
original question and the retrieved evidence.
Across four model configurations on MedQA
and MMLU-Med, HCQR consistently outperforms
simple single-query RAG as well as stronger
retrieve-then-rerank and retrieve-then-filter base-
lines. HCQR also substantially increases the
proportion of retrieved contexts judged decision-
useful. Taken together, these results indicate that
in optional-retrieval settings, retrieval quality de-
pends not only on access to external knowledge, but
on whether first-pass retrieval is aligned with the
downstream decision. Under a fixed final-context
budget, better query construction can therefore be
more consequential than inspecting larger interme-
diate candidate pools or filtering a poorly targeted
initial retrieval set.

Limitations
This study has several limitations.
First, HCQR depends on the quality of the inter-
mediate hypothesis state. If the working hypothesis,
discriminating features, or confirming evidence are
inaccurate or incomplete, the resulting rewritten
queries may be misaligned with the true decision
boundary. Although the key feature query is in-
tended to mitigate this risk, it does not eliminate
it.
Second, our empirical evaluation is concentrated
on medical multiple-choice QA. This is a deliber-
ate choice because the distinction between topical
relevance and decision-usefulness is particularly
clear in this setting. However, it leaves open the
extent to which the same design transfers to other
domains or to open-ended tasks in which candidate
answers are not explicitly given and must instead
be hypothesized.
Third, the context-utility analysis relies on an
LLM judge. We use this judge as a compara-
tive operational measure and keep the evaluation
prompt fixed across methods, but the resulting la-
bels should not be interpreted as human gold an-
notations. Accordingly, the utility analysis is best
understood as supporting evidence for the proposed
mechanism rather than as a definitive causal char-
acterization of retrieval quality.
Finally, our experiments use a fixed corpus, a
fixed retriever family, and a fixed final-context bud-
get. The comparisons are normalized by the evi-
dence shown to the final generator rather than by
total retrieval-time computation, and some base-
lines inspect larger intermediate candidate pools
by design. We therefore do not claim that the re-
ported gains will transfer unchanged to open-web
retrieval, substantially larger corpora, or fully agen-
tic multi-turn retrieval settings.
Ethical Considerations
Our primary experiments are conducted in the med-
ical domain, where errors in retrieval or answer gen-
eration may have serious real-world consequences.
This work is intended as a study of retrieval behav-
ior on benchmark QA tasks and is not designed
for direct clinical use. Any real-world medical
deployment would require substantially stronger
validation, including expert review, provenance-
aware interfaces, calibrated uncertainty handling,
abstention mechanisms, and systematic monitoring
for harmful retrieval failures.More broadly, methods that condition retrieval
on a working hypothesis may introduce additional
risks if the intermediate hypothesis is inaccurate or
systematically biased. Our default design partially
mitigates this concern by restricting the role of the
hypothesis to retrieval planning rather than expos-
ing it directly to the final generator. However, this
design choice does not remove the possibility of
biased or misdirected retrieval. Careful evaluation
remains necessary in any high-stakes application.
References
Chen Amiraz, Florin Cuconasu, Simone Filice, and
Zohar Karnin. 2025. The distracting effect: Under-
standing irrelevant passages in rag. InProceedings
of the 63rd Annual Meeting of the Association for
Computational Linguistics (Volume 1: Long Papers),
pages 18228–18258.
Akari Asai, Zeqiu Wu, Yizhong Wang, Avirup Sil, and
Hannaneh Hajishirzi. 2023. Self-rag: Learning to
retrieve, generate, and critique through self-reflection.
InThe Twelfth International Conference on Learning
Representations.
Chia-Yuan Chang, Zhimeng Jiang, Vineeth Rakesh,
Menghai Pan, Chin-Chia Michael Yeh, Guanchu
Wang, Mingzhi Hu, Zhichao Xu, Yan Zheng, Ma-
hashweta Das, and 1 others. 2025. Main-rag: Multi-
agent filtering retrieval-augmented generation. In
Proceedings of the 63rd Annual Meeting of the As-
sociation for Computational Linguistics (Volume 1:
Long Papers), pages 2607–2622.
Yufeng Du, Minyang Tian, Srikanth Ronanki, Subendhu
Rongali, Sravan Bodapati, Aram Galstyan, Azton
Wells, Roy Schwartz, Eliu A Huerta, and Hao
Peng. 2025. Context length alone hurts llm per-
formance despite perfect retrieval.arXiv preprint
arXiv:2510.05381.
Wenqi Fan, Yujuan Ding, Liangbo Ning, Shijie Wang,
Hengyun Li, Dawei Yin, Tat-Seng Chua, and Qing
Li. 2024. A survey on rag meeting llms: Towards
retrieval-augmented large language models. InPro-
ceedings of the 30th ACM SIGKDD conference on
knowledge discovery and data mining, pages 6491–
6501.
Luyu Gao, Xueguang Ma, Jimmy Lin, and Jamie Callan.
2023. Precise zero-shot dense retrieval without rel-
evance labels. InProceedings of the 61st Annual
Meeting of the Association for Computational Lin-
guistics (Volume 1: Long Papers), pages 1762–1777.
Dan Hendrycks, Collin Burns, Steven Basart, Andy Zou,
Mantas Mazeika, Dawn Song, and Jacob Steinhardt.
2020. Measuring massive multitask language under-
standing.arXiv preprint arXiv:2009.03300.

Yizheng Huang and Jimmy Huang. 2024. A survey
on retrieval-augmented text generation for large lan-
guage models.arXiv preprint arXiv:2404.10981.
Di Jin, Eileen Pan, Nassim Oufattole, Wei-Hung Weng,
Hanyi Fang, and Peter Szolovits. 2021. What disease
does this patient have? a large-scale open domain
question answering dataset from medical exams.Ap-
plied Sciences, 11(14):6421.
Qiao Jin, Won Kim, Qingyu Chen, Donald C Comeau,
Lana Yeganova, W John Wilbur, and Zhiyong Lu.
2023. Medcpt: Contrastive pre-trained transformers
with large-scale pubmed search logs for zero-shot
biomedical information retrieval.Bioinformatics,
39(11):btad651.
Hailey Joren, Jianyi Zhang, Chun-Sung Ferng, Da-
Cheng Juan, Ankur Taly, and Cyrus Rashtchian.
2024. Sufficient context: A new lens on retrieval
augmented generation systems.arXiv preprint
arXiv:2411.06037.
Hyunjae Kim, Jiwoong Sohn, Aidan Gilson, Nicholas
Cochran-Caggiano, Serina Applebaum, Heeju Jin,
Seihee Park, Yujin Park, Jiyeong Park, Seoyoung
Choi, and 1 others. 2025. Rethinking retrieval-
augmented generation for medicine: A large-scale,
systematic expert evaluation and practical insights.
arXiv preprint arXiv:2511.06738.
Yuan-Ching Kuo, Yi Yu, Chih-Ming Chen, and Chuan-
Ju Wang. 2025. Mmlf: Multi-query multi-passage
late fusion retrieval. InFindings of the Association
for Computational Linguistics: NAACL 2025, pages
6587–6598.
Patrick Lewis, Ethan Perez, Aleksandra Piktus, Fabio
Petroni, Vladimir Karpukhin, Naman Goyal, Hein-
rich Küttler, Mike Lewis, Wen-tau Yih, Tim Rock-
täschel, and 1 others. 2020. Retrieval-augmented gen-
eration for knowledge-intensive nlp tasks.Advances
in neural information processing systems, 33:9459–
9474.
Jintao Liang, Huifeng Lin, You Wu, Rui Zhao, Ziyue
Li, and 1 others. 2025. Reasoning rag via system 1
or system 2: A survey on reasoning agentic retrieval-
augmented generation for industry challenges. In
Proceedings of the 14th International Joint Confer-
ence on Natural Language Processing and the 4th
Conference of the Asia-Pacific Chapter of the Asso-
ciation for Computational Linguistics, pages 1954–
1966.
Xinbei Ma, Yeyun Gong, Pengcheng He, Hai Zhao,
and Nan Duan. 2023. Query rewriting in retrieval-
augmented large language models. InProceedings
of the 2023 Conference on Empirical Methods in
Natural Language Processing, pages 5303–5315.
Multi-Linguality Multi-Functionality Multi-Granularity.
2024. M3-embedding: Multi-linguality, multi-
functionality, multi-granularity text embeddings
through self-knowledge distillation.arXiv preprint
arXiv:2402.03216.Zackary Rackauckas. 2024. Rag-fusion: a new take
on retrieval-augmented generation.arXiv preprint
arXiv:2402.03367.
Alireza Salemi and Hamed Zamani. 2024. Evaluating
retrieval quality in retrieval-augmented generation.
InProceedings of the 47th International ACM SI-
GIR Conference on Research and Development in
Information Retrieval, pages 2395–2400.
Harsh Trivedi, Niranjan Balasubramanian, Tushar Khot,
and Ashish Sabharwal. 2023. Interleaving retrieval
with chain-of-thought reasoning for knowledge-
intensive multi-step questions. InProceedings of
the 61st annual meeting of the association for com-
putational linguistics (volume 1: long papers), pages
10014–10037.
Guangzhi Xiong, Qiao Jin, Zhiyong Lu, and Aidong
Zhang. 2024a. Benchmarking retrieval-augmented
generation for medicine. InFindings of the Associa-
tion for Computational Linguistics: ACL 2024, pages
6233–6251.
Guangzhi Xiong, Qiao Jin, Xiao Wang, Minjia Zhang,
Zhiyong Lu, and Aidong Zhang. 2024b. Improving
retrieval-augmented generation in medicine with it-
erative follow-up questions. InBiocomputing 2025:
Proceedings of the Pacific Symposium, pages 199–
214. World Scientific.
Shi-Qi Yan, Jia-Chen Gu, Yun Zhu, and Zhen-Hua Ling.
2024. Corrective retrieval augmented generation.
Ori Yoran, Tomer Wolfson, Ori Ram, and Jonathan
Berant. 2023. Making retrieval-augmented language
models robust to irrelevant context.arXiv preprint
arXiv:2310.01558.
Yue Yu, Wei Ping, Zihan Liu, Boxin Wang, Jiaxuan You,
Chao Zhang, Mohammad Shoeybi, and Bryan Catan-
zaro. 2024. Rankrag: Unifying context ranking with
retrieval-augmented generation in llms.Advances in
Neural Information Processing Systems, 37:121156–
121184.

Appendix
A Additional Results
This section provides the full results underlying
the summary tables in the main text, together with
additional appendix-only analyses for the ablation
settings.
A.1 Full model-by-model values for the
diagnostic controls
Tables 7 and 8 provide the full values underlying
the dataset-level averages reported in Tables 5 and 6
of the main text. Reporting the per-model numbers
is useful for checking whether the average trends
are broad or whether they depend on a single model
family. The finer-grained pattern is consistent with
the main-text discussion. Exposing the answer op-
tions to a generic multi-query rewriter produces
only modest gains relative to rewriting from the
question alone, whereas HCQR remains stronger
across model families. Likewise, in the query-role
ablation, removing any one of the three HCQR
queries lowers the overall average, indicating that
the three retrieval intents are complementary rather
than redundant.
Table 7 makes clear that answer-option exposure
by itself does not recover the full benefit of HCQR.
The option-aware rewriting control is often some-
what better than generic rewriting, but the gains are
smaller and less stable than the gains from adding
hypothesis-conditioned retrieval planning. This is
the same conclusion drawn in the main text, now
shown at the level of individual generators.
The full query-ablation table leads to the same
interpretation as Table 6 in the main text. The
support query q1is the most important component
on average, but the distinction query q2and the key
feature query q3also contribute meaningfully. Cell-
level variation remains, as expected, yet the average
pattern still supports the view that HCQR works by
combining multiple evidence-seeking roles rather
than by relying on a single rewritten query.
A.2 Additional DUR results for the controls
and ablations
Tables 9 and 10 report DUR for the same diag-
nostic controls. These results complement the ac-
curacy tables by showing how the corresponding
variants affect the fraction of retrieved contexts
judged decision-useful. In general, variants that are
weaker in accuracy are also weaker in DUR, and
the gaps are often larger under DUR than underaccuracy. This pattern strengthens the interpreta-
tion that the main effect of HCQR is improved
evidence selection rather than incidental changes
in final answer generation.
For the option-exposure control, DUR shows an
even clearer separation between generic option-
aware rewriting and HCQR. Adding options to
a generic rewriting prompt improves the rate
of decision-useful retrieval, but the full HCQR
pipeline remains substantially higher. This is con-
sistent with the argument that the benefit comes
from how HCQR organizes retrieval around a pro-
visional hypothesis, and the key clues in the stem.
The DUR ablation table mirrors the accuracy ab-
lation but also sharpens it. Removing any one of
the three query roles lowers the share of decision-
useful contexts, again with the largest average drop
after removing q1. At the same time, the drops
forq2andq3remain substantial, which supports
the claim that alternative discrimination and clue
verification are not peripheral additions; they help
determine whether the final context is actually in-
formative for the downstream decision.
A.3 Full context-utility breakdown
Table 11 expands the summary in Table 3 by re-
porting the full label distribution for each dataset
and model configuration. The more detailed view
shows that HCQR does not improve only by shift-
ing a small number of cases from NOTUSEFUL
to USEFUL. It increases both the ENTAILEDand
USEFULrates while reducing NOTUSEFULacross
nearly all settings. This matters because the EN-
TAILEDsubset is the one most strongly associated
with downstream answer gains in the main text.
The per-model breakdown also suggests a pat-
tern noted in the main text: the HCQR advantage
appears somewhat more pronounced on MedQA
than on MMLU-Med, with a similar tendency vis-
ible not only in overall DUR but also in the EN-
TAILEDportion.
A.4 Utility-conditioned accuracy by method
Table 12 provides the full values underlying Table 4.
The main purpose of this table is to verify that the
monotonic pattern discussed in the main text is pre-
served when we disaggregate by model family and
dataset. That pattern remains intact overall: EN-
TAILEDsubsets have the highest answer accuracy
and the largest positive same-subset gains over no-
retrieval COT, USEFULsubsets remain beneficial
but to a smaller extent, and NOTUSEFULsubsets

Table 7: Full model-by-model accuracy values underlying the dataset-level averages reported in Table 5.
MedQA MMLU-Med
MethodLlama3.2 3B Llama3.1 8B Qwen3 4B Qwen3 30BAverageLlama3.2 3B Llama3.1 8B Qwen3 4B Qwen3 30BAverage
Rewriting 58.9 67.5 73.1 83.3 70.7 67.0 74.7 82.5 88.8 78.2
Rewriting+options 57.0 68.3 75.8 84.1 71.3 67.5 77.2 83.7 88.4 79.2
HCQR−options 61.1 71.6 75.3 84.5 73.1 69.1 77.3 83.7 89.4 79.9
HCQR (ours) 63.4 73.0 77.9 85.8 75.0 68.6 79.7 85.4 90.6 81.1
Table 8: Full model-by-model accuracy values underlying the dataset-level averages reported in Table 6.
MedQA MMLU-Med
MethodLlama3.2 3B Llama3.1 8B Qwen3 4B Qwen3 30BAverageLlama3.2 3B Llama3.1 8B Qwen3 4B Qwen3 30BAverage
HCQR (ours) 63.4 73.0 77.985.875.068.679.785.490.6 81.1
HCQR−q 1 62.1 70.1 73.8 83.5 72.4 66.9 77.1 83.0 88.9 79.0
HCQR−q 2 61.2 71.5 74.086.373.269.278.7 83.4 89.0 80.1
HCQR−q 3 61.3 70.9 74.3 85.6 73.0 68.2 77.886.090.1 80.5
Table 9: Additional DUR results for the option-exposure control. DUR denotes ENTAILED+ USEFUL.
MedQA MMLU-Med
MethodLlama3.2 3B Llama3.1 8B Qwen3 4B Qwen3 30BAverageLlama3.2 3B Llama3.1 8B Qwen3 4B Qwen3 30BAverage
Rewriting 45.2 51.2 62.2 65.6 56.1 51.4 53.7 56.7 59.1 55.3
Rewriting+options 51.1 62.9 71.8 76.4 65.6 56.8 61.3 64.8 66.3 62.3
HCQR−options 61.0 70.5 70.5 80.5 70.6 57.0 61.1 64.2 67.5 62.4
HCQR (ours) 73.4 81.8 83.2 89.8 82.1 65.8 69.1 72.5 73.7 70.3
Table 10: Additional DUR results for the HCQR query-role ablation. DUR denotes ENTAILED+ USEFUL.
MedQA MMLU-Med
MethodLlama3.2 3B Llama3.1 8B Qwen3 4B Qwen3 30BAverageLlama3.2 3B Llama3.1 8B Qwen3 4B Qwen3 30BAverage
HCQR (ours) 73.4 81.8 83.2 89.8 82.1 65.8 69.1 72.5 73.7 70.3
HCQR−q 1 55.0 62.1 66.5 70.9 63.6 48.5 52.3 55.2 55.8 53.0
HCQR−q 2 60.8 71.7 70.1 80.4 70.8 53.5 57.4 61.8 64.2 59.2
HCQR−q 3 63.1 73.0 74.2 82.6 73.2 51.6 57.0 60.8 63.0 58.1
are typically weakest. Although individual cells
vary, the full table supports the operational validity
of the judge labels used throughout the paper.
B Experimental Details
This section records the implementation choices
used to instantiate HCQR and the comparison base-
lines. All LLM components within a method use
the same model as the final generator for that ex-
perimental setting.
B.1 Shared inference settings
All non-judge model inference was run with vLLM.
For all models, the context length was set to 8192
tokens. The maximum output length was 2048
tokens and decoding used greedy decoding. The
only exception was HyDE, for which hypothetical-
document generation used temperature 0.7 to en-
courage diversity across the synthetic passages. For
final answer generation, all methods used the same
MIRAGE prompt described in the main text. Thecontext-utility judge used GPT-4.1 (version 2025-
04-14), with greedy decoding and a maximum out-
put length of 8192 tokens.
All retrieval-based methods used the same Med-
CPT retriever over the same medical textbook
corpus and were evaluated under the same final
generator-context budget of 15 documents. The
comparison is therefore normalized by the evidence
shown to the final generator, even when methods
differ in how many intermediate candidates they
inspect before selecting that final context.
B.2 No-retrieval CoT and Simple RAG
The no-retrieval baseline answers directly from the
question and options using the CoT prompt in Fig-
ure 3. SIMPLERAG uses the raw question as a
single retrieval query, retrieves the top 15 docu-
ments, and passes them directly to the shared final
generator prompt in Figure 12. These two baselines
define the lower and upper ends of the minimal re-
trieval setup used throughout the paper: the former

Table 11: Full context-utility breakdown underlying Table 3. For each method, we report the rates (%) of ENTAILED,
USEFUL-BUT-NOT-ENTAILED(USEFUL), and NOTUSEFULcontexts across all four model configurations and
their average.
MedQA MMLU-Med
Method LabelLlama3.2 3B Llama3.1 8B Qwen3 4B Qwen3 30BAvg.Llama3.2 3B Llama3.1 8B Qwen3 4B Qwen3 30BAvg.
Simple RAGEntailed 6.0 6.0 6.0 6.0 6.0 22.5 22.5 22.5 22.5 22.5
Useful 24.0 24.0 24.0 24.0 24.0 26.2 26.2 26.2 26.2 26.2
Not Useful 70.0 70.0 70.0 70.0 70.0 51.3 51.3 51.3 51.3 51.3
Rerank-RAGEntailed 10.7 10.7 10.7 10.7 10.7 29.1 29.1 29.1 29.1 29.1
Useful 33.4 33.4 33.4 33.4 33.4 27.7 27.7 27.7 27.7 27.7
Not Useful 55.9 55.9 55.9 55.9 55.9 43.2 43.2 43.2 43.2 43.2
RewritingEntailed 11.0 12.7 15.5 18.4 14.4 23.0 23.8 24.5 27.8 24.8
Useful 34.2 38.5 46.7 47.2 41.7 28.4 29.9 32.2 31.3 30.5
Not Useful 54.8 48.8 37.8 34.4 43.9 48.6 46.3 43.3 40.9 44.7
Rewriting+optionsEntailed 13.4 18.1 20.8 24.3 19.1 26.4 27.4 28.9 30.7 28.3
Useful 37.8 44.9 51.0 52.1 46.4 30.5 34.0 35.9 35.6 34.0
Not Useful 48.9 37.1 28.2 23.6 34.4 43.2 38.7 35.2 33.7 37.7
HyDEEntailed 13.0 14.0 13.6 14.5 13.7 18.9 19.9 19.5 19.8 19.5
Useful 38.6 40.5 39.2 40.1 39.6 24.1 26.3 24.8 26.4 25.4
Not Useful 48.4 45.5 47.2 45.4 46.6 57.0 53.8 55.7 53.7 55.1
MAIN-RAGEntailed 6.0 6.2 6.4 6.7 6.3 15.7 17.3 17.2 17.7 17.0
Useful 17.8 20.2 18.2 20.0 19.0 19.9 19.5 16.5 18.7 18.7
Not Useful 76.3 73.6 75.3 73.3 74.6 64.4 63.3 66.3 63.5 64.4
HCQR (ours)Entailed 29.6 32.0 35.3 41.9 34.7 31.9 34.5 35.5 36.7 34.7
Useful 43.8 49.8 47.8 47.9 47.3 34.0 34.5 37.0 37.0 35.6
Not Useful 26.6 18.2 16.8 10.2 17.9 34.2 30.9 27.5 26.3 29.7
HCQR−optionsEntailed 18.5 24.9 25.1 31.6 25.0 25.2 28.5 29.1 32.4 28.8
Useful 42.5 45.6 45.4 48.9 45.6 31.9 32.6 35.1 35.1 33.7
Not Useful 39.0 29.5 29.5 19.5 29.4 43.0 38.9 35.8 32.5 37.6
HCQR -q 1Entailed 12.5 12.6 15.5 14.0 13.6 16.4 18.6 19.7 17.9 18.1
Useful 42.5 49.6 51.0 56.9 50.0 32.2 33.8 35.6 37.9 34.8
Not Useful 45.0 37.9 33.5 29.1 36.4 51.5 47.7 44.8 44.2 47.0
HCQR -q 2Entailed 18.2 24.0 25.2 31.7 24.8 23.2 25.5 27.4 27.2 25.8
Useful 42.5 47.7 44.9 48.7 46.0 30.3 31.9 34.4 36.9 33.4
Not Useful 39.2 28.3 29.9 19.6 29.2 46.5 42.6 38.2 35.8 40.8
HCQR -q 3Entailed 20.0 22.9 25.1 31.7 24.9 20.1 23.5 22.5 25.1 22.8
Useful 43.1 50.0 49.1 50.9 48.3 31.4 33.6 38.2 37.9 35.3
Not Useful 36.9 27.0 25.8 17.4 26.8 48.4 43.0 39.2 37.0 41.9
Table 12: Full model-by-model values underlying Table 4. Each cell is reported asaccuracy /∆ CoT.
MedQA MMLU-Med
Method LabelLlama3.2 3B Llama3.1 8B Qwen3 4B Qwen3 30BAvg.Llama3.2 3B Llama3.1 8B Qwen3 4B Qwen3 30BAvg.
Simple RAGEntailed 93.5 / +9.1 93.5 / +9.1 93.5 / +9.1 93.5 / +9.1 93.5 / +9.1 97.1 / +6.1 97.1 / +6.1 97.1 / +6.1 97.1 / +6.1 97.1 / +6.1
Useful 85.6 / +5.9 85.6 / +5.9 85.6 / +5.9 85.6 / +5.9 85.6 / +5.9 88.1 / +0.7 88.1 / +0.7 88.1 / +0.7 88.1 / +0.7 88.1 / +0.7
Not Useful 64.5 / -4.9 64.5 / -4.9 64.5 / -4.9 64.5 / -4.9 64.5 / -4.9 71.9 / -5.2 71.9 / -5.2 71.9 / -5.2 71.9 / -5.2 71.9 / -5.2
Rerank-RAGEntailed 97.8 / +8.1 97.8 / +8.1 97.8 / +8.1 97.8 / +8.1 97.8 / +8.1 95.6 / +6.0 95.6 / +6.0 95.6 / +6.0 95.6 / +6.0 95.6 / +6.0
Useful 81.6 / +5.2 81.6 / +5.2 81.6 / +5.2 81.6 / +5.2 81.6 / +5.2 82.1 / -2.0 82.1 / -2.0 82.1 / -2.0 82.1 / -2.0 82.1 / -2.0
Not Useful 61.5 / -5.9 61.5 / -5.9 61.5 / -5.9 61.5 / -5.9 61.5 / -5.9 73.4 / -4.3 73.4 / -4.3 73.4 / -4.3 73.4 / -4.3 73.4 / -4.3
RewritingEntailed 86.4 / +13.6 94.4 / +11.7 98.5 / +8.6 97.4 / +0.4 94.2 / +8.6 84.9 / +9.6 91.9 / +7.4 95.5 / +6.0 98.7 / +1.7 92.7 / +6.1
Useful 66.5 / +8.9 74.5 / +2.2 78.5 / +4.2 88.0 / +0.5 76.9 / +4.0 70.6 / +1.6 77.8 / +1.8 88.0 / +0.0 90.0 / +0.6 81.6 / +1.0
Not Useful 48.6 / -6.6 54.9 / -8.1 55.9 / -8.1 69.3 / -3.7 57.2 / -6.6 56.7 / -4.0 64.0 / -6.0 70.9 / -4.5 81.3 / -5.0 68.2 / -4.8
HyDEEntailed 81.8 / +6.7 93.8 / +10.7 94.8 / +4.6 99.5 / +1.6 92.5 / +5.9 91.7 / +9.7 94.9 / +4.6 98.1 / +2.8 98.6 / -0.5 95.8 / +4.2
Useful 60.7 / +5.1 72.3 / +1.7 76.3 / +3.6 84.1 / +0.2 73.3 / +2.7 71.0 / +3.8 82.1 / +2.1 88.5 / +1.9 95.1 / +2.8 84.2 / +2.6
Not Useful 49.5 / -5.7 59.4 / -4.2 60.7 / -7.2 77.0 / -3.3 61.7 / -5.1 58.8 / -1.8 67.7 / +0.0 74.8 / -2.0 84.1 / -1.7 71.4 / -1.4
MAIN-RAGEntailed 86.8 / +17.1 91.1 / +10.1 97.6 / +6.1 100.0 / +4.7 93.9 / +9.5 93.6 / +6.4 95.2 / +5.3 97.9 / +2.7 99.0 / -0.5 96.4 / +3.5
Useful 60.0 / +1.8 70.0 / -1.9 79.7 / +4.7 89.4 / +3.1 74.8 / +1.9 73.0 / +7.4 81.6 / -0.9 91.1 / -1.7 91.7 / +0.5 84.4 / +1.3
Not Useful 52.8 / -4.0 62.6 / -4.7 66.4 / -4.3 80.6 / -2.1 65.6 / -3.8 59.2 / -2.4 68.7 / -0.3 74.7 / -2.5 84.3 / -3.0 71.7 / -2.1
HCQR (ours)Entailed 89.1 / +14.3 94.3 / +8.8 97.1 / +6.9 98.5 / +3.2 94.8 / +8.3 90.5 / +8.1 92.3 / +5.1 98.2 / +3.4 97.5 / +1.2 94.6 / +4.4
Useful 64.5 / +6.3 74.6 / +6.8 77.3 / +7.4 84.9 / +2.5 75.3 / +5.7 73.0 / +4.6 81.9 / +7.7 84.6 / +3.7 91.6 / +1.0 82.8 / +4.3
Not Useful 32.9 / -5.9 31.2 / -12.1 39.3 / -5.1 37.7 / -10.0 35.3 / -8.3 44.1 / -5.4 63.9 / +1.2 69.9 / -0.3 80.0 / -1.4 64.5 / -1.5

removes retrieval entirely, while the latter intro-
duces retrieval without any additional planning,
transformation, reranking, or filtering.
B.3 Hypothesis-Conditioned Query Rewriting
(HCQR)
HCQR is instantiated as the two-stage pre-retrieval
pipeline described in Section 4. The Hypothesis
Formulator (Figure 9) reads the question and an-
swer options and returns a structured working state
containing discriminating features, a brief ratio-
nale, confirming evidence to seek, and a provi-
sional best guess. The Query Rewriter (Figure 11)
then converts that state into exactly three queries
corresponding to the roles described in the main
text: support for the working hypothesis, distinc-
tion from competing options, and verification of
salient clues in the question.
Each of the three queries retrieves the top 5 docu-
ments. The retrieved sets are merged, deduplicated,
and truncated only as needed to satisfy the shared
final-context budget. By default, the working hy-
pothesis is used only for retrieval planning and is
not shown to the final generator. The HCQR −
options control removes the answer options from
the hypothesis-formulation prompt while leaving
the rest of the pipeline unchanged.
B.4 Rewriting baselines
Rewriting baseline follows the standard paraphrase-
style multi-query rewriting setup with retrieval fu-
sion. Concretely, we use the same three-query
prompt family as the LC-MQR baseline in prior
work (Kuo et al., 2025), prompting the model
to produce exactly three paraphrase-style queries
from the original question. The option-aware con-
trol in Figure 5 additionally exposes the answer
options but does not introduce any structured hy-
pothesis state. In both cases, we retrieve the top 5
documents per query, take the union of the retrieved
sets, remove duplicates, and pass the resulting con-
text to the same final generator prompt used by
all methods. These controls are therefore matched
to HCQR in the number of rewritten queries and
in the final evidence budget; what differs is how
retrieval intent is constructed.
B.5 HyDE (Hypothetical Document
Embeddings)
HyDE uses the prompt in Figure 6 to generate hy-
pothetical documents from the input question. In
our implementation, we generate 8 hypotheticalpassages per question. The original query embed-
ding and the 8 hypothetical-passage embeddings
are then averaged to form the retrieval representa-
tion used by MedCPT. As noted above, HyDE is
the only method that does not use greedy decoding
for its intermediate generation step: we use tem-
perature 0.7 so that the synthetic passages are not
collapsed into near-duplicates.
B.6 Rerank-RAG
RERANK-RAG first retrieves 150 candidate doc-
uments with the raw question using MedCPT
and then reranks those candidates with the BGE
reranker v2-m3 cross-encoder (Multi-Granularity,
2024). Reranking is performed on the question–
document pair, with a batch size of 16 and a max-
imum sequence length of 512 tokens. The top 15
reranked documents are then passed to the shared fi-
nal generator. This baseline therefore tests whether
stronger post-retrieval selection can compensate
for starting from a raw-question first pass.
B.7 MAIN-RAG
MAIN-RAG follows the retrieve-then-filter de-
sign described in the main text. It begins with 15
documents retrieved from the raw question. For
each candidate document, Agent-1 (Figure 7) pro-
duces a document-conditioned answer choice, and
Agent-2 (Figure 8) judges whether the document
contains sufficient information to support that an-
swer. We quantify the Agent-2 judgment using the
log-probability difference between the generated
YesandNotokens and retain documents whose
scores are at least the mean score of the candidate
set. The surviving documents constitute the final
context shown to the shared generator.
B.8 Context-utility judge
The context-utility analysis uses the judge prompt
in Figure 2. Given the question, options, gold
answer, and the retrieved context set produced
by a method, the judge first identifies the an-
swer option most strongly supported by the con-
text alone ( selected_answer ) and then assigns
one of the three labels used in the paper: EN-
TAILED, USEFUL-BUT-NOT-ENTAILED, or NOT
USEFUL. The prompt is intentionally conser-
vative about ENTAILEDand USEFUL-BUT-NOT-
ENTAILED; background-only passages, weak topi-
cal relevance, and contexts that support a compet-
ing option are labeled NOTUSEFUL. We keep this
prompt fixed across all methods so that the utility

analysis reflects differences in retrieved evidence
rather than differences in the judging procedure.
C Prompt Templates
This section reproduces the prompt templates used
in the experiments. Figures 3 and 12 show the
answer-generation prompts for the no-retrieval and
retrieval-based settings, respectively. Figures 4–8
show the prompts used by the comparison base-
lines, Figure 2 shows the context-utility judge, and
Figures 9–11 show the HCQR prompts, includ-
ing the no-options variant used in the diagnostic
control. We include the exact templates because
several of the compared methods differ primarily
in how they formulate retrieval intent.
D Illustrative Example
Figures 13–17 present a representative MedQA ex-
ample that makes the logic of HCQR concrete. The
example is useful precisely because the intermedi-
ate hypothesis is provisional rather than guaranteed
to be correct. As in the main method, the formula-
tor is asked to propose a working hypothesis that
can organize retrieval, not to lock in the final an-
swer. The support query follows that provisional
guess, while the distinction and key feature queries
are intended to surface the evidence that can either
confirm it or overturn it.
Example walkthrough.In this example, the
hypothesis formulator initially leans toward a
clubfoot-style interpretation and therefore proposes
a corresponding management hypothesis. HCQR
does not treat that intermediate state as authorita-
tive. Instead, it uses the intermediate state to gen-
erate one query that follows the current hypothesis
and two additional queries that probe the nearby
decision boundary. The resulting retrieved set con-
tains both hypothesis-supporting and hypothesis-
challenging evidence, including passages that dis-
tinguish a flexible forefoot deformity with a neutral
heel from true clubfoot. The final generator then
produces its answer from the retrieved evidence
and the original question, rather than from the pro-
visional hypothesis itself. As shown by the raw
output in Figure 17, the model can revise the initial
guess when the retrieved evidence points elsewhere.
This is the intended role of hypothesis conditioning
throughout the paper: it sharpens first-pass retrieval
while preserving the ability of the downstream gen-
erator to correct an initially plausible but imperfect
hypothesis.

GPT Judge Prompt
System:
You evaluate the relationship between retrieved context and the gold answer for a multiple-choice question.
User:
### QUESTION
{question}
### OPTIONS
{options}
### GOLD ANSWER
{gold_line}
### RETRIEVED CONTEXT
{chunks}
### TASK
Given the retrieved context, evaluate its relationship to the gold answer.
1. Identify the answer option most strongly supported by the retrieved context alone. Call this ´selected_answer´.
2. Then classify how the retrieved context relates to the gold answer: whether it entails the gold answer, provides useful but incomplete support for it, or is
not useful for supporting it.
### LABELS
1. ENTAILED
Definition:
The gold answer is entailed by the retrieved context alone.
The reasoning chain from the retrieved context to the gold answer is complete within the provided context.
Required conditions:
- The gold answer follows from the retrieved context alone.
- No outside knowledge, missing fact, or unstated assumption is needed.
- The conclusion is decisive, not merely plausible.
-´selected_answer ´matches the gold answer.
2. USEFUL_BUT_NOT_ENTAILED
Definition:
The retrieved context provides strong decision-relevant support for the gold answer, but the gold answer is not fully entailed by the retrieved context alone.
A small additional reasoning step, minor outside knowledge, or a single missing bridge may still be needed, but the retrieved context already contains
evidence that substantially narrows the decision toward the gold answer.
This label should be used only when the retrieved context contains at least one of the following:
- a decisive clue, rule, condition, or relation that strongly supports the gold answer, even if one final bridge is missing
- evidence that rules out or strongly weakens one or more plausible alternative answers, leaving the gold answer as the clearly favored option
- a partial but highly informative connection between the question and the gold answer that would make the correct answer reasonably reachable with only
minimal additional knowledge
Do not use this label for context that is merely relevant, broadly related, or weakly suggestive.
3. NOT_USEFUL
Definition:
The retrieved context does not provide strong decision-relevant support for the gold answer.
It may be topically related, mention similar concepts, or provide background information, but it does not contain a decisive clue, a strong exclusion of
alternatives, or a core connection that materially helps determine the gold answer.
### RULES
- Evaluate only the retrieved context.
- Evaluate the label relative to the gold answer.
- Infer ’selected_answer’ from the retrieved context alone; do not force it to equal the gold answer.
- Be conservative about ENTAILED.
Output only JSON on the last line:
{{"selected_answer": "<A|B|C|D|UNSURE>", "label": "<ENTAILED|USEFUL_BUT_NOT_ENTAILED|NOT_USEFUL>"}}
Figure 2: Prompt used for the context-utility judge.

COT Prompt
System:
You are a helpful medical expert, and your task is to answer a multi-choice medical question. Please first think
step-by-step and then choose the answer from the provided options. Organize your output in a json formatted as
Dict{"step_by_step_thinking": Str(explanation), "answer_choice": Str{A/B/C/...}}. Your responses will be used for
research purposes only, so please have a definite answer.
User:
Here is the question:
{question}
Here are the potential choices:
{options}
Please think step-by-step and generate your output in json:
Figure 3: Prompt used for the no-retrieval CoT baseline.
Rewriting Prompt
User:
You are an AI language model assistant. Your task is to generate exactly three different versions of the given user
question to retrieve relevant documents from a vector database. By generating multiple perspectives on the user
question, your goal is to help the user overcome some of the limitations of the distance-based similarity search.
Original question: {query}
Format your response in plain text as:
Sub-query 1:
Sub-query 2:
Sub-query 3:
Figure 4: The prompt used for Rewriting baseline.
Rewriting with Options Prompt
User:
You are an AI language model assistant. Your task is to generate exactly three different versions of the given user
question to retrieve relevant documents from a vector database. By generating multiple perspectives on the user
question, your goal is to help the user overcome some of the limitations of the distance-based similarity search.
Original question: {query}
Options:
{options}
Format your response in plain text as:
Sub-query 1:
Sub-query 2:
Sub-query 3:
Figure 5: The prompt used for Rewriting with options ablation.

HyDE Prompt
User:
Please write a passage to answer the question.
Question: {question}
Passage:
Figure 6: Prompt used for generating hypothetical documents in the HyDE baseline.
MAIN-RAG: Agent-1 (Predictor) Prompt
System:
You are an accurate and reliable AI assistant that can answer questions with the help of external documents. You should
only provide the correct answer without repeating the question and instruction.
User:
Document:
{context}
Question:
{question}
Options:
{options}
Based ONLY on the provided document, what is the best answer to the question? Output only the answer choice.
Figure 7: The prompt used for Agent-1 (Predictor) in the MAIN-RAG baseline to generate document-conditioned
answers for the Document-Query-Answer triplet.
MAIN-RAG: Agent-2 (Judge) Prompt
System:
You are a noisy document evaluator that can judge if the external document is noisy for the query with unrelated or
misleading information. Given a retrieved Document, a Question, and an Answer generated by an LLM (LLM Answer),
you should judge whether both the following two conditions are reached: (1) the Document provides specific information
for answering the Question; (2) the LLM Answer directly answers the question based on the retrieved Document. Please
note that external documents may contain noisy or factually incorrect information. If the information in the document
does not contain the answer, you should point it out with evidence. You should answer with ’Yes’ or ’No’ with evidence
of your judgment, where ’No’ means one of the conditions (1) and (2) are unreached and indicates it is a noisy document.
User:
Document:
{context}
Question with Options:
{question}
{options}
LLM Answer:
{llm_answer}
Does the document contain enough specific information to answer the question, and is the LLM Answer directly
supported by the document? Answer with exactly one token: Yes or No.
Figure 8: The prompt used for Agent-2 (Judge) in the MAIN-RAG baseline to evaluate the decision-usefulness and
reliability of the retrieved document.

HCQR: Hypothesis Formulator Prompt
System:
You are an expert analyst taking an exam.
User:
Question: {question}
Options:
{options}
Analyze this question carefully. Think step-by-step about each option, considering the information given in the question
and relevant knowledge. Reason through the elimination analysis before making your final assessment.
After your analysis, provide your final assessment in JSON:
{
"discriminating_features": ["2-3 features that distinguish between options"],
"reasoning": "brief explanation why this is the best answer",
"confirming_evidence": ["1-3 specific facts that would confirm this answer"],
"best_guess": "A/B/C/D",
"best_guess_text": "«copy the chosen option text verbatim»"
}
Figure 9: Prompt used for the HCQR hypothesis-formulation stage.
HCQR: Hypothesis Formulator without Options Prompt
System:
You are an expert analyst taking an exam.
User:
Question: {question}
Analyze this question carefully. Think step-by-step about each option, considering the information given in
the question and relevant knowledge. Reason through the elimination analysis before making your final assessment.
After your analysis, provide your final assessment in JSON:
{
"discriminating_features": ["2-3 features that distinguish between options"],
"reasoning": "brief explanation why this is the best answer",
"confirming_evidence": ["1-3 specific facts that would confirm this answer"],
"best_guess": "write your best guess for the answer"
}
Figure 10: The prompt used for the Hypothesis Formulator (Stage 1) without options for ablation.

HCQR: Query Rewriter Prompt
System:
You are a search query expert. Generate precise, targeted search queries. Output ONLY the 3 queries in the exact format
requested.
User:
Generate 3 highly targeted search queries to find evidence for this question.
Question: {question}
Best Guess Answer: {best_guess_text}
Reasoning: {reasoning}
Evidence Needed: {confirming_evidence}
Key Features: {discriminating_features}
Generate 3 SPECIFIC queries:
Query 1: Find evidence supporting {best_guess_text} - focus on the main reasoning
Query 2: Find distinguishing criteria between the top candidate answers
Query 3: Find specific key features or facts
Format:
Query 1: [query]
Query 2: [query]
Query 3: [query]
Figure 11: Prompt used for the HCQR query-rewriting stage.
Final Generator Prompt
System:
You are a helpful medical expert, and your task is to answer a multi-choice medical question using the relevant
documents. Please first think step-by-step and then choose the answer from the provided options. Organize your
output in a json formatted as Dict{"step_by_step_thinking": Str(explanation), "answer_choice": Str{A/B/C/...}}. Your
responses will be used for research purposes only, so please have a definite answer.
User:
Here are the relevant documents:
{context}
Here is the question:
{question}
Here are the potential choices:
{options}
Please think step-by-step and generate your output in json:
Figure 12: The standard generator prompt (MIRAGE template) used uniformly across all evaluated methods,
including Simple RAG, all baselines, and our proposed HCQR, to strictly control the post-retrieval generation phase.

Example Question (MedQA)
Question:
A 3-week-old male newborn is brought to the physician because of an inward turning of his left forefoot. He was
born at 38 weeks’ gestation by cesarean section because of breech presentation. The pregnancy was complicated by
oligohydramnios. Examination shows concavity of the medial border of the left foot with a skin crease just below the
ball of the great toe. The lateral border of the left foot is convex. The heel is in neutral position. Tickling the lateral
border of the foot leads to correction of the deformity. The remainder of the examination shows no abnormalities. X-ray
of the left foot shows an increased angle between the 1st and 2nd metatarsal bones.
Which of the following is the most appropriate next step in the management of this patient?
Options
A: Foot abduction brace
B: Arthrodesis of the forefoot
C: Reassurance
D: Tarsometatarsal capsulotomy
Figure 13: Representative MedQA example used to illustrate the end-to-end HCQR workflow.
Hypothesis Formulation
Discriminating Features:
The clinical presentation includes a medial concavity of the foot with a skin crease and correction with tickling of the
lateral border—classic signs oftalipes equinovarus(congenital clubfoot) in the early stages.
Confirming Evidence:
The patient is 3 weeks old, which falls within the optimal age range for early intervention in congenital clubfoot.
Reasoning:
The clinical findings are consistent with congenital clubfoot. In infants younger than 6 months with a flexible deformity,
the appropriate initial management is typically non-surgical correction using a foot abduction brace.
Working Hypothesis:
Foot abduction brace
Figure 14: Illustrative provisional output of the Hypothesis Formulator. The working hypothesis is used to steer
retrieval and may later be confirmed or overturned by the retrieved evidence.
Query Rewriter Output
Support Query
Foot abduction brace for congenital clubfoot in infants under 6 months with flexible deformity and response to tickling:
evidence for first-line non-surgical management in early-stage talipes equinovarus
Distinction Query
Distinguishing features between foot abduction brace, arthrodesis, tarsometatarsal capsulotomy, and reassurance in the
management of early congenital clubfoot in neonates under 3 months of age
Key Feature Query
Clinical and radiographic key features of congenital clubfoot in 3-week-old infants: medial concavity, lateral convexity,
skin crease below great toe, correction with lateral tickling, and increased 1st–2nd metatarsal angle on X-ray
Figure 15: Three intent-aligned queries generated from the provisional hypothesis: answer support, alternative
discrimination, and clue verification.

Retrieved Evidence Examples
Support / Distinction Evidence
“Treatment of In-toeing. The mainstay of management is to identify patients who have pathologic reasons for in-toeing
and provide reassurance and follow-up to document improvement. It can take until 7 to 8 years of age for correction.
Fewer than 1% of patients with in-toeing will require surgical intervention due to functional disability or cosmetic
concerns.”
Distinction / Key Feature Evidence
“Family history is important. Clubfoot can be congenital, teratologic, or positional. Congenital clubfoot (approximately
75% of cases) is typically isolated, although infants should be evaluated for developmental dysplasia of the hip. The
infant will have hindfoot equinus and varus, forefoot adduction, and varying degrees of rigidity. These findings arise
from abnormalities in the talonavicular joint.”
Figure 16: Representative passages retrieved by the generated queries. The retrieved set can contain both supporting
and corrective evidence, allowing the final generator to revise the provisional hypothesis when necessary.
Generator Raw Output Examples
The patient is a 3-week-old male with an inward turning of the left forefoot, described as concavity of the medial border
of the foot with a skin crease just below the ball of the great toe and a convex lateral border. The heel is in neutral
position, and tickling the lateral border leads to correction of the deformity. This indicates that the deformity is not fixed
and can be passively corrected — a key feature of a positional deformity rather than a rigid congenital deformity like
clubfoot.
The x-ray shows an increased angle between the 1st and 2nd metatarsal bones, which is consistent with metatarsus
adductus — a condition where the forefoot is adducted (turned inward), often due to in utero positioning. This is a
common, benign condition in newborns, especially in the context of a breech presentation and oligohydramnios, which
may have contributed to abnormal positioning in utero.
Document [4] (Surgery_Schwartz) states: ’Metatarsus adductus in infants will also resolve spontaneously in most cases.’
Similarly, Document [8] notes that the calcaneovalgus foot (a different condition) resolves by 2 years with reassurance
and passive stretching. The key point is that the deformity is not rigid and responds to passive manipulation (tickling
corrects it), which supports a positional, non-pathologic origin.
Therefore, the most appropriate next step is reassurance, with monitoring and possibly passive stretching exercises, as
the condition is expected to resolve spontaneously. No bracing, surgery, or arthrodesis is indicated at this stage.
Option A (foot abduction brace) is not indicated — bracing is not recommended for metatarsus adductus or intoeing
in infants (Document [3]). Option B (arthrodesis) is invasive and inappropriate for a benign, self-resolving condition.
Option D (tarsometatarsal capsulotomy) is a surgical procedure for a rigid deformity and not indicated in a condition
that is passively correctable. Thus, the best choice is C: Reassurance.
answer_choice: C
Figure 17: Raw generator output for the representative MedQA example. The output shows that the final generator
answers from the retrieved evidence and the original question rather than from the provisional hypothesis alone,
and in this case correctly arrives at the gold answer after revising the initial guess. The text is reproduced verbatim
except for formatting.