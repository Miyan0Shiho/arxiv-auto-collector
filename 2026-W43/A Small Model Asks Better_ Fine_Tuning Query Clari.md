# A Small Model Asks Better: Fine-Tuning Query Clarification for Retrieval-Augmented Generation

**Authors**: Lihong He, Lucian Popa, Marina Danilevsky

**Published**: 2026-10-19 22:33:41

**PDF URL**: [https://arxiv.org/pdf/2610.28686v1](https://arxiv.org/pdf/2610.28686v1)

## Abstract
Query clarification is essential for retrieval-augmented generation (RAG) systems that face ambiguous user queries. We consider multi-turn grounded conversational scenarios where a user query is either ill-formed or ambiguous in the context of the prior dialog or, more interestingly, it may be well-formed but when posed against retrieved documents there are multiple relevant interpretations. The task is to recognize both such cases and then generate an appropriate clarification request to the user. We present ClariEdge, a new dataset of approximately 3,900 examples that include such ambiguous queries along with negative examples that do not require clarification. We also show that a lightweight fine-tuned LoRA achieves 96.6% classification accuracy and outperforms prompted baselines of many larger models on both classification and quality of clarification generation. Our study shows that hard negatives are an essential ingredient to separate out queries that are truly ambiguous from queries that may look ambiguous but are not (i.e., can be resolved from the context). The same LoRA achieves high performance in both pre-retrieval and post-retrieval scenarios; thus, it can be used in two places in a RAG pipeline as a check before answer generation. The ClariEdge dataset is available at https://huggingface.co/datasets/ibm-research/ClariEdge.

## Full Text


<!-- PDF content starts -->

A Small Model Asks Better: Fine-Tuning Query Clarification for
Retrieval-Augmented Generation
Lihong He, Lucian Popa, Marina Danilevsky
IBM Research, USA
lihong.he@ibm.com
Abstract
Query clarification is essential for retrieval-
augmented generation (RAG) systems that
face ambiguous user queries. We consider
multi-turn grounded conversational scenarios
where a user query is either ill-formed or
ambiguous in the context of the prior dialog
or, more interestingly, it may be well-formed
but when posed against retrieved documents
there are multiple relevant interpretations. The
task is to recognize both such cases and then
generate an appropriate clarification request to
the user. We present ClariEdge, a new dataset
of approximately 3,900 examples that include
such ambiguous queries along with negative
examples that do not require clarification. We
also show that a lightweight fine-tuned LoRA
achieves 96.6% classification accuracy and
outperforms prompted baselines of many larger
models on both classification and quality of
clarification generation. Our study shows that
hard negatives are an essential ingredient to
separate out queries that are truly ambiguous
from queries that may look ambiguous but are
not (i.e., can be resolved from the context).
The same LoRA achieves high performance in
both pre-retrieval and post-retrieval scenarios;
thus, it can be used in two places in a RAG
pipeline as a check before answer generation.
The ClariEdge dataset is available at https:
//huggingface.co/datasets/
ibm-research/ClariEdge.
1 Introduction
Ambiguity in user queries can come in two distinct
forms. The first is theill-formedquery, where the
user mentions an entity or attribute that the conver-
sation has not narrowed down to a single choice.
Figure 1a shows such a case: after the assistant
has surfaced two credit cards, the user asks “What
are the potential rewards of this credit card?” The
system must decide whether to attempt an answer
under uncertainty or ask which card the user means;a useful clarification lists the candidates already on
the table (a pattern we callhedging over list).
The second form arises when the query iswell-
formed in isolationbut becomes ambiguous once
grounded in a set of retrieved documents. In Fig-
ure 1b, the user asks whether “the language that
descended from Common Brittonic” is still spo-
ken, a question that sounds well-defined - but
the retrieved passages surface several possible de-
scendants (Welsh, Breton, Cornish). In retrieval-
augmented generation (RAG) (Rosenthal et al.,
2026), this is a common occurrence, and a use-
ful clarification should both flag the ambiguity and
also enumerate the grounded options with a brief
answer for each (hedging with answers). Thequal-
ityof clarification, that is, naming the right options
and choosing the pattern of communication (see Ta-
ble 7 for details) matters as much as the decision of
whether to clarify at all. Both use cases - ambiguity
before retrieval, and after - can occur in the same
deployment. Therefore we need a unified clarifica-
tion functionality that handles both, usable before
retrieval, based only on the conversation (Cao et al.,
2025; Sahay et al., 2025), and after retrieval, when
documents become available (Kim et al., 2023).
Existing benchmarks address only one slice of
this picture. CLAMBER (Zhang et al., 2024)
evaluates ambiguity detection on standalone ques-
tions (no conversational context, no retrieved docu-
ments) and finds that chain-of-thought and few-shot
prompting yield only marginal gains over a base
classifier. AmbigQA (Min et al., 2020) grounds
ambiguity in a corpus, but treats a question as
ambiguous whenever retrieval surfaces multiple
atomic factual answers, even when they are part
of the same larger answer. This is a notion that
current conversational LLMs largely subsume by
responding with a composite answer that includes
the atomic facts: for example, “Who took control
1
arXiv:2610.28686v1  [cs.AI]  19 Oct 2026

U:I’m applying for a credit card for personal travel
and business expenses. What would you suggest?
A:Two strong options: (1)Gold Card— travel re-
wards and lounge access; (2)Business Card— cash-
back on office expenses.
U:What are the rewards ofthis credit card?
A:Which credit card are you asking about: the Gold
Card or the Business Card?
(a) Ill-formed query: “this credit card” has multiple referents
in the conversation. Clarification lists the candidates (hedging
over list).U:Isthe languagethat descended from Common
Brittonic still spoken today, or has it become extinct?
Retrieved: passages on Welsh, Breton, and Cornish as descendants of
Common Brittonic.
A:Several languages descended from Common Brit-
tonic:Welsh, still widely spoken in Wales;Breton,
still spoken in Brittany; andCornish, extinct in the
19th century but since revitalized. Which one are you
referring to?
(b) Well-formed query: user thinks there is only one choice,
but the retrieved corpus shows several. Clarification pairs each
option with a brief grounded answer (hedging with answers).
Figure 1: Ambiguity forms in conversational RAG. Ambiguous user query and clarification response are highlighted.
of the Italian government in 1922?”1yields “Ben-
ito Mussolini” and “the Fascists” in one response
– so the AmbigQA ambiguity rarely surfaces as a
clarification need. Neither CLAMBER nor Am-
bigQA capture the joint conversation-plus-corpus
setting that real RAG deployments face.
Problem Statement.Given a multi-turn conver-
sation between a user and an AI assistant, together
with optionally retrieved documents for the last
user turn, our goal is to determine whether the
user’s last turn is ambiguous, that is, whether it ad-
mits either no clear interpretation or multiple valid
interpretations. When it is ambiguous, the system
should generate a clarification request that elicits
additional information from the user.
Towards addressing this problem, we make the
following specific contributions:
1. ClariEdge: Clarification Grounded
Dataset with Hard Negative Edge Cases.We re-
lease ClariEdge, a dataset of approximately 3,900
examples with multi-turn conversations, retrieved
documents, and explicit hard negatives, that is,
queries that look ambiguous but are resolvable
from context via coreference resolution or query
rewriting. The dataset is built through direct human
annotation and synthetic LLM generation followed
by human validation.
2. Lightweight Fine-Tuning Substan-
tially Outperforms Prompting.Against seven
prompted baselines that include substantially larger
models, our lightweight LoRA achieves the highest
classification accuracy and the best clarification-
generation quality on the queries it correctly flags.
Whereas prompted baselines tend to either over-
clarify or under-clarify, the fine-tuned model han-
1This question, along with many similar ones, comes from
the AmbigQA dataset and is considered ambiguous there.
Query Clarification LoRA
(conversation -only)
Query
RetrieverQuery Post -RetrievalPre-RetrievalCLEAR
Ambiguous
Clarifying Question
Retriever
Query Clarification LoRA
(conversation+document )
CLEAR
Ambiguous
Clarifying QuestionGeneratorAnswer
Figure 2: Deployment in a RAG pipeline.
dles both directions reliably. An ablation study
further confirms that the hard negatives present
in ClariEdge are essential for query clarification,
since removing them causes a sharp drop in accu-
racy on boundary cases.
3. Multi-Setting Handling in a Single Model.
Unlike approaches that rely on separate mod-
els or dynamic strategy selection (Sahay et al.,
2025), one LoRA model handles both pre-retrieval
(conversation-only) and post-retrieval (conversa-
tion plus documents) scenarios, with classification
accuracy ahead of larger prompted models. As
shown in Figure 2, the same LoRA is invoked
once on the conversation alone (gating retrieval)
and once on the conversation plus retrieved doc-
uments (gating answer generation), asking a clar-
ifying question whenever either invocation flags
ambiguity.
2 Related Work
Ambiguity and clarification have been studied
across many information-seeking settings: open-
domain factual QA (AmbigQA, Min et al., 2020;
ASQA, Stelmakh et al., 2022; CAmbigNQ, Lee
et al., 2023; CondAmbigQA, Li et al., 2025), stan-
dalone questions (CLAMBER, Zhang et al., 2024),
conversational QA (Abg-CoQA, Guo et al., 2021;
PACIFIC, Deng et al., 2022; CANARD, Elgohary
et al., 2019), and conversational search (Qulac,
2

DatasetAmbiguity
typeMulti-
turnRetri-
evalHard
negatives Size
AmbigQA answer-set✗ ✓ ✗14k
CLAMBER typology✗ ✗ ✗12k
ClariQ facet✗ ✓ ✗237
Abg-CoQA referential✓ ✓ ✗8.6k
ClariEdgeentity-option✓ ✓ ✓3.9k
Table 1: Comparison with prior clarification datasets.
Aliannejadi et al., 2019; ClariQ, Aliannejadi et al.,
2020; MIMICS, Zamani et al., 2020). Table 1 com-
pares our dataset against four of these benchmarks.
On the modeling side, LLMs gain only marginal
benefits from chain-of-thought or few-shot prompt-
ing on ambiguity identification (Zhang et al., 2024;
Deng et al., 2023), motivating work that exploits
models’ own uncertainty signals (Kim et al., 2024;
Testoni and Fernández, 2024) or resolves ambigu-
ity via retrieval-grounded disambiguation without
asking the user (Kim et al., 2023).
On the method side, prior systems specialize
clarification along three axes:whento clarify, us-
ing LLM uncertainty or calibration (Kuhn et al.,
2023; Zhang and Choi, 2023; Cole et al., 2023);
howto ask, via facet-driven or reasoning-trace su-
pervision (Sekuli ´c et al., 2021; Andukuri et al.,
2024; Zhang et al., 2025; Rahmani et al., 2024);
andwhichdeployment setting to target (Sahay et al.,
2025; Chen et al., 2024; Cao et al., 2025; Qin et al.,
2025). These systems typically rely on prompting
or uncertainty rather than task-specific supervision,
and each remains tied to the prompts, models, or
iterations of its setting.
In contrast, we show that a single LoRA fine-
tuned with hard negatives covers both pre- and post-
retrieval, draws an explicit boundary between gen-
uine ambiguity and context-resolvable references,
and outperforms substantially larger prompted
baselines. This indicates that training-data qual-
ity dominates raw model scale on this task.
3 ClariEdge Dataset
Effective query clarification requires training data
that captures the nuanced decision boundaries be-
tween clarification-worthy and clear queries. This
section describes our ClariEdge dataset and its
creation. Conversations span the same four do-
mains as MTRAG (Katsis et al., 2025): ClapNQ
(Wikipedia), Cloud (technical documentation),
FiQA (finance), and Govt (government).
Table 1 compares our ClariEdge dataset againstCategory Type Train Test All
Ambiguous positive 967 104 1,071
CLEAR-random easy negative 1,800 200 2,000
CLEAR-hard hard negative 742 79 821
Total 3,509 383 3,892
Table 2: ClariEdge dataset categories and statistics.
four widely used clarification benchmarks, each
covering a complementary slice: AmbigQA (Min
et al., 2020) targetsanswer-setambiguity in open-
domain factual QA; CLAMBER (Zhang et al.,
2024) introduces aEpistemic Misalignment / Lin-
guistic Ambiguity / Aleatoric Outputtypology for
LLM evaluation on standalone questions; ClariQ
(Aliannejadi et al., 2020) captures web-search
facets; and Abg-CoQA (Guo et al., 2021) ad-
dressesreferentialambiguity in conversational QA.
None of these combines multi-turn conversation,
retrieved documents, and explicit hard negatives in
a single resource. Our dataset targetsentity-option
ambiguity, where the user query may be referring
to one of several candidate entities surfaced by ei-
ther the conversation or the retrieved documents,
and is the first to include systematic hard negative
edge cases that look ambiguous but are resolvable
from context.
3.1 Data Categories and Statistics
Our dataset comprises three categories: (1)Am-
biguous(positive examples), where user queries
genuinely require clarification due to multiple plau-
sible interpretations; (2)CLEAR-random(easy
negative examples), which are queries that do not
require clarification and are straightforwardly clear;
and (3)CLEAR-hard(hard negative examples),
which are queries that appear superficially ambigu-
ous but can be resolved from conversation context.
Table 2 presents the data size and Table 3 illustrates
some examples for the three categories.
The distinction between CLEAR-hard and
CLEAR-random is critical: hard negatives appear
ambiguous due to referential expressions but can be
resolved from conversation context, while easy neg-
atives are straightforwardly clear queries. Training
with hard negatives teaches the model the precise
boundary between genuinely ambiguous queries
and those with resolvable references.
Additional dataset statistics (pre-/post-retrieval
split, turn counts, document counts) are reported in
Appendix A.
3

Category User Query Context Why This Category
Ambiguous “How did the fast food chain
establish a significant pres-
ence in the US?”Documents describe many fast
food chains such as In-N-Out
Burger, Jack in the BoxMultiple valid interpretations for
fast food chain, the user must spec-
ify which one
CLEAR-random “What is the capital of
France?”Documents contain a direct answer Unambiguous query with a single
clear interpretation
CLEAR-hard “When are the services of
this division available?”Prior turn: “I work in the Northeast
division”Appears ambiguous (“this division”)
but resolvable from prior context
Table 3: Examples for each data category.
3.2 Data Creation Process
Ambiguous examples.Following the method-
ology for creating underspecified questions in
MTRAG-UN (Rosenthal et al., 2026), the final user
query in each conversation was designed to admit
multiple plausible interpretations. Conversations
were constructed via two parallel approaches: (1)
annotators writing ambiguous questions inspired
by entity types and instances drawn from the under-
lying corpus; each such question was then stitched,
in a subsequent step, as the last turn onto an exist-
ing human-annotated multi-turn conversation, or
(2) synthetic generation of ambiguous questions ap-
pended as the final turn to existing human-created
multi-turn conversations, followed by a manual val-
idation step. The latter approach is also the same
one used for CLEAR-hard examples (see more de-
tails later): the major difference is that here the
manual validation decides that these are truly am-
biguous questions, while for CLEAR-hard it de-
cides the converse.
For approach (1), relevant passages were added
to each conversation using two retrieval strategies:
direct retrieval with the ambiguous question, and
query expansion with a context relevance filter,
which yields a richer set of passages to simulate
scenarios where significantly more interpretations
exist; these examples form the post-retrieval por-
tion of the Ambiguous category. Further details on
the human annotation guidelines for this approach
are given in Appendix B.1. For approach (2), no
passages were retrieved: this branch shares the
synthetic-then-validate pipeline used for CLEAR-
hard (described below), and produces conversation-
only (pre-retrieval) Ambiguous examples. In both
cases, reference clarification responses were gen-
erated using an LLM with pattern-based prompt-
ing to include relevant disambiguation options (see
Section 3.3), followed by human correction.
CLEAR-random examples.The easy negative
examples in this category were constructed by ran-domly sampling clear, unambiguous user queries
from existing multi-turn conversations. For some
examples, retrieved documents were paired with
the query to represent clear post-retrieval scenarios;
for others, documents were deliberately removed to
create conversation-only examples where the query
remains unambiguous without external context.
CLEAR-hard examples.A key contribution of
our dataset is the systematic construction of hard
negatives, which are queries that may appear am-
biguous to an LLM but can be resolved from con-
versation context. By construction, these examples
are for the conversation-only (pre-retrieval) setting:
their ambiguity-or-resolvability hinges on linguis-
tic cues in the dialogue rather than on retrieved
passages. We describe their creation next.
We prompted an LLM on multi-turn conversa-
tions to generate questions targeting entities dis-
cussed in the conversation, where the last user
query is intended to be ambiguous. In the basic
variant, the model was given 793 conversations and
asked to produce questions where a user asks about
an entity type with multiple instances occurring
in the conversation. In an advanced variant, 228
longer conversations (>6 turns with multiple en-
tity types) were processed using chain-of-thought
reasoning with self-verification and iterative refine-
ment. Human annotators then manually reviewed
all LLM-generated queries, classifying each as gen-
uinely ambiguous or resolvable from context. Con-
versations with last user query determined to be
truly ambiguous were moved to the Ambiguous
category, while the remaining ones, where queries
appear ambiguous but can be resolved through con-
versation context, became hard negatives.
We collected additional patterns of hard nega-
tives where the synthetic generation model incor-
rectly generated clear queries as ambiguous. These
included: “which one” patterns where the phrasing
appears ambiguous but the question is just a com-
parison across the previously mentioned entities;
4

Pattern User Query Why CLEAR
“Which one”
comparison“Which one has
the best battery
life?”Comparing all entities,
not asking about one
specific
Follow-up
question“What’s the can-
cellation policy?”General question about
all discussed options
Terse query “Price?” No pronoun; asking
about all entities, not
one specific
Table 4: Additional hard negative patterns.
follow-up questions resolvable through the regular
conversation flow; and terse or keyword queries
that appear ambiguous but are clear in context. Ta-
ble 4 gives examples of these additional patterns.
In the above process, we used Mistral-Large-32
for synthetic generation. Even though this model is
relatively powerful, we were able to obtain enough
CLEAR-hard examples, showing that the model
struggles to delineate the boundary between am-
biguous and clear. The human manual review was
essential for this. Further details on the manual
review process, including an illustrative example,
are given in Appendix B.2.
3.3 Generation of Clarification Response
For each ambiguous query in the dataset, we gen-
erate a reference clarification response through a
structured pipeline: (1) identify the ambiguous
entity in the last user query, that is, the term or
reference that admits multiple interpretations; (2)
extract candidate entity options from the retrieved
document and prior conversations, together with a
short associated answer where applicable; (3) ver-
ify the consistency of each extracted option against
the ambiguous entity and deduplicate them; (4) gen-
erate the clarification response based on the number
of distinct options, following one of three patterns
described below; (5) human annotators review the
final clarification response and modify it when the
LLM-generated output is inaccurate or insufficient.
Steps 1–4 are performed by an LLM (the same
Mistral-Large-3), followed by human review and
correction in step 5. In step 4, the clarification
response follows one of three patterns based on
the number of distinct options: (1) Hedging with
answers (few options, e.g., 2–3): list each option
with a brief description or answer; (2) Hedging
over list (medium number of options, e.g., 4–8):
enumerate the plausible options without additional
2https://huggingface.co/mistralai/
Mistral-Large-3-675B-Instruct-2512-NVFP4explanatory content; and (3) Open-domain (many
or unbounded options): directly asks the user for
disambiguation over the type of entity that they
may have in mind. Appendix C provides some
examples of the three patterns.
3.4 Inter-rater Agreement
To measure the annotation reliability, we follow
the calculation of inter-rater agreement in Abg-
CoQA (Guo et al., 2021).
For the inter-rater agreement on whether a user
query is ambiguous or not, we randomly select 80
samples from the test set: 20 from CLEAR-hard,
40 from CLEAR-random, and 20 from Ambiguous.
We ask two authors to provide their annotations
(CLEAR or Ambiguous) on these 80 examples,
and then compute Cohen’s Kappa. The two anno-
tators disagree on only 4 data points, all of which
are sampled from the CLEAR-hard category. The
resulting Cohen’s Kappa is 0.875 (almost perfect
agreement), indicating that the ambiguity labels are
highly reliable.
For the inter-rater agreement on clarification
questions, we ask the same two annotators to inde-
pendently write a clarification question for each of
the 20 sampled Ambiguous examples (one version
is kept as the gold clarification). We then compute
the macro-average F1 score of word overlap and the
BLEU-4 score. The F1 is 0.774 and the BLEU-4
is 0.57. We further conduct a manual review of the
two clarification versions on the 20 samples. The
manual inspection and the metrics agree that the
two versions match on the ambiguous entity and
its candidate options but differ in surface phrasing:
the F1 of 0.774 captures the shared content while
the moderate BLEU-4 of 0.57 reflects the phrasing
variation, indicating that the gold clarifications are
content-faithful yet not unique verbatim references.
4 Models
This section describes our fine-tuned LoRA models
and the baseline models used in the experiments.
4.1 Fine-Tuned LoRA Models
We employ Low-Rank Adaptation (LoRA) (Hu
et al., 2022) to create parameter-efficient clarifica-
tion models, fine-tuned on the 3,509 training ex-
amples described in Table 2. This substantially
reduces training cost and enables rapid experimen-
tation with different training data compositions. We
fine-tune LoRA on (1) small 3B models from the
5

Granite family (Granite-4.0-micro3and Granite-
4.1-3B4), targeted at cost-effective deployment;
and (2) GPT-OSS-20B, a larger open-source model
included for comparison.
The two small Granite models serve as our pri-
mary models. This choice reflects several consid-
erations. First, their size-to-performance trade-off
provides sufficient capacity while maintaining de-
ployment efficiency, in line with recent evidence
that lightweight models can be competitive on well-
defined tasks (Nandi et al., 2026). Second, both are
amenable to LoRA adaptation with a modest train-
ing set. Finally, using the same base architectures
for both fine-tuned and prompted configurations
allows us to isolate the effect of fine-tuning from
any difference in model capacity.
Training configuration and input/output format
for the LoRA models are described in Appendix D.
4.2 Prompted Baseline Models
To quantify the value of fine-tuning, we com-
pare our LoRA models against seven prompted
baselines evaluated without any task-specific train-
ing. The baselines span proprietary frontier,
open-weights, and lightweight model families:
GPT-5.6 SolandGPT-4o(frontier, proprietary);
GPT-OSS-120B(large, open-source);GPT-OSS-
20B(medium, open-source);Granite-4.1-30B
(medium, open-source); andGranite-4.1-3Band
Granite-4.0-micro(lightweight, open-source).
All prompted baselines receive identical chat
messages consisting of a system message specify-
ing the clarification rules and a user message con-
structed from each test record. The system message
defines the task, grounding requirements, and de-
scribes the patterns that the generated clarifications
should follow. The user message supplies the con-
versation and retrieved documents (if any). The full
text of both messages is provided in Appendix E.
5 Experiments and Analysis
In this section, we describe the evaluation metrics
and then give the main experimental results.
5.1 Evaluation Metrics
We assess model performance along two dimen-
sions: (1)classification accuracy, measuring
whether the model correctly determines if a query
3https://huggingface.co/ibm-granite/
granite-4.0-micro
4https://huggingface.co/ibm-granite/
granite-4.1-3bis ambiguous or not; and (2)generation quality,
evaluating the clarification response produced for
correctly identified ambiguous queries.
Prior work has used BLEU (Guo et al., 2021)
and BERTScore (Zhang et al., 2024) to evaluate
generation against a reference answer, but neither
captures whether the clarification presents correct
disambiguation options for the ambiguous entity.
We therefore design entity-option-centric metrics
in the spirit of AmbigQA (Min et al., 2020), differ-
ing in that we evaluate over entity options rather
than answer interpretations and accommodate open-
domain clarifications where options cannot be ex-
plicitly enumerated. Specifically, we use the fol-
lowing metrics:Precision: the fraction of options
in the generated clarification that are correct;Re-
call: the fraction of gold reference options that
are covered by the generated clarification;F1: har-
monic mean of precision and recall. Computing
precision and recall on free-form clarification re-
sponses requires four auxiliary judgments. We ob-
tain these judgments with four LLM prompts (see
Appendix F). We also provide examples in Ap-
pendix G to illustrate the metrics, as well as how
we handle open-domain clarifications.
5.2 Classification Results
We evaluate classification accuracy in three set-
tings: full test set, pre-retrieval subset, and post-
retrieval subset. The results are reported in Table 5.
5.2.1 Full Test Set
The classification results on the full test set support
two observations, which we discuss in turn.
Fine-tuning beats prompting at all scales.The
Granite-4.1-3B LoRA reaches 96.6% overall ac-
curacy, a 17.2 pp gain over GPT-4o and a 31.8
pp gain over prompted Granite-4.1-3B. Because
the prompted and fine-tuned variants of Granite-
4.1-3B share the same architecture, this improve-
ment is attributable to training rather than to greater
model capacity, which is consistent with Zhang
et al. (2024)’s finding that prompting offers lim-
ited gains on the task of query clarification. The
Granite-4.0-micro LoRA also achieves strong per-
formance at 92.4%, and both Granite LoRAs out-
perform the larger GPT-OSS-20B LoRA and all
the other prompted baselines, in line with prior
evidence that specialized lightweight models can
be competitive on well-scoped tasks (Nandi et al.,
2026). Even the weaker GPT-OSS-20B LoRA
6

Full Test Set Pre-Retrieval Post-Retrieval
Model Ambig. C-rand. C-hard Overall Ambig. C-rand. C-hard Overall Ambig. C-rand. Overall
Granite-4.1-3B LoRA 91.3% 99.5% 96.2% 96.6% 84.5% 98.0% 96.2% 93.0% 100.0% 100.0% 100.0%
Granite-4.0-micro LoRA 84.6% 98.5% 87.3% 92.4% 72.4% 98.0% 87.3% 85.6% 100.0% 98.7% 99.0%
GPT-OSS-20B LoRA 76.9% 98.5% 86.1% 90.1% 62.1% 100.0% 86.1% 82.4% 95.7% 98.0% 97.4%
GPT-4o 48.1% 90.5% 92.4% 79.4% 60.3% 88.0% 92.4% 81.3% 32.6% 91.3% 77.6%
GPT-5.6 Sol 93.3% 72.5% 48.1% 73.1% 94.8% 70.0% 48.1% 68.4% 91.3% 73.3% 77.6%
GPT-OSS-120B 80.8% 83.0% 65.8% 78.9% 86.2% 78.0% 65.8% 75.4% 73.9% 84.7% 82.1%
GPT-OSS-20B 90.4% 72.5% 58.2% 74.4% 93.1% 72.0% 58.2% 72.7% 87.0% 72.7% 76.0%
Granite-4.1-30B 78.8% 80.0% 63.3% 76.2% 87.9% 78.0% 63.3% 74.9% 67.4% 80.7% 77.6%
Granite-4.1-3B 26.9% 88.0% 55.7% 64.8% 34.5% 94.0% 55.7% 59.4% 17.4% 86.0% 69.9%
Granite-4.0-micro 25.0% 85.0% 63.3% 64.2% 31.0% 90.0% 63.3% 60.4% 17.4% 83.3% 67.9%
Table 5: Classification accuracy. Ambig.=Ambiguous, C-rand.=CLEAR-random, C-hard=CLEAR-hard.
(90.1%) outperforms the strongest prompted base-
line GPT-4o by 10.7 pp overall, indicating that the
gap is robust to LoRA backbone choice.
Prompted baselines are systematically imbal-
anced.The imbalance is most pronounced be-
tween the Ambiguous and CLEAR-hard categories,
the two that hinge on correctly distinguishing
genuine ambiguity from surface-level ambiguity.
Prompted baselines split into two failure modes
along this axis: over-clarifiers (such as GPT-5.6
Sol and GPT-OSS-20B), biased toward flagging
queries as ambiguous, and under-clarifiers (such as
GPT-4o and the smaller Granite models), biased
toward declaring queries clear. Notably, the two
frontier proprietary models diverge along this axis,
indicating that the imbalance is not attributable to
model scale alone. In contrast, the fine-tuned Lo-
RAs (and especially the Granite two) maintain a
good balance, indicating that exposure to hard neg-
atives is essential for separating genuine ambiguity
from resolvable references. We also conduct an
ablation study on hard negatives, showing that re-
moving them causes a sharp drop in accuracy on
boundary cases. The details are in Appendix H.
5.2.2 Pre-Retrieval
In the pre-retrieval setting, the model must decide
whether to clarify based solely on the multi-turn
conversation. This setting is particularly challeng-
ing because the model cannot leverage document
evidence to disambiguate queries and to surface the
interpretations.
The classification accuracy on the pre-retrieval
subset shows that all three fine-tuned LoRAs
lead on overall accuracy, substantially improving
over their architecture-matched prompted baselines.
The class-imbalance pattern observed on the full
test set (§5.2.1) is again evident among promptedmodels in pre-retrieval: some over-clarify (e.g.,
GPT-5.6 Sol and GPT-OSS-20B), while others
under-clarify (e.g., GPT-4o and the smaller Gran-
ite models). In contrast, the Granite-4.1-3B LoRA
maintains relatively balanced performance, though
its Ambiguous accuracy is lower than its CLEAR-
hard accuracy. Pre-retrieval is where linguistic am-
biguity is hardest to disentangle: CLEAR-hard ex-
amples by construction (see Section 3.2) are for the
conversation-only setting, and present an inherently
harder setup where the model must rely purely on
linguistic cues to separate Ambiguous from cases
that are resolvable in the dialogue.
5.2.3 Post-Retrieval
In the post-retrieval setting, the model has access to
the conversation together with documents returned
by retriever. The CLEAR-hard category is not rep-
resented in this subset; as we mentioned, they target
conversational context rather than documents.
From the post-retrieval results in Table 5, the
fine-tuned LoRAs reach near-saturation overall
classification accuracy: the Granite-4.1-3B LoRA
hits 100.0% and the Granite-4.0-micro LoRA
hits 99.0%. Both are far ahead of the strongest
prompted baseline (GPT-OSS-120B at 82.1%).
The prompted baselines continue to exhibit the
class-imbalance patterns observed in earlier set-
tings. These results indicate that, once retrieved
documents are available, the remaining difficulty
lies in correctly interpreting the retrieval signal.
Fine-tuning acquires this skill reliably, whereas
prompting does not.
5.3 Generation Quality Results
We assess clarification generation quality across the
same three settings using the entity-option-centric
metrics. A clarification only matters when the
7

Full Test Set Pre-Retrieval Post-Retrieval
Model Corr. Prec. Rec. F1 Corr. Prec. Rec. F1 Corr. Prec. Rec. F1
Granite-4.1-3B LoRA 95/104 92.7% 85.7% 89.0% 49/58 89.3% 80.9% 84.9% 46/46 96.3% 90.7% 93.4%
Granite-4.0-micro LoRA 88/104 91.8% 87.7% 89.7% 42/58 87.4% 83.6% 85.5% 46/46 95.8% 91.4% 93.5%
GPT-OSS-20B LoRA 80/104 89.1% 82.3% 85.6% 36/58 83.5% 81.9% 82.7% 44/46 93.7% 82.6% 87.8%
GPT-4o 50/104 81.3% 69.8% 75.1% 35/58 74.4% 71.9% 73.1% 15/46 97.8% 64.2% 77.5%
GPT-5.6 Sol 97/104 92.6% 73.4% 81.9% 55/58 90.1% 80.7% 85.1% 42/46 96.0% 63.9% 76.7%
GPT-OSS-120B 84/104 89.0% 68.4% 77.3% 50/58 83.8% 71.5% 77.2% 34/46 96.8% 63.7% 76.8%
GPT-OSS-20B 94/104 93.3% 70.7% 80.5% 54/58 94.0% 81.8% 87.5% 40/46 92.4% 55.1% 69.0%
Granite-4.1-30B 82/104 88.7% 87.1% 87.9% 51/58 83.7% 88.9% 86.2% 31/46 97.2% 84.0% 90.1%
Granite-4.1-3B 28/104 81.8% 72.2% 76.7% 20/58 78.2% 74.8% 76.5% 8/46 91.7% 65.1% 76.1%
Granite-4.0-micro 26/104 75.0% 45.7% 56.8% 18/58 72.2% 45.9% 56.1% 8/46 81.2% 44.7% 57.7%
Table 6: Generation quality on examples correctly classified as ambiguous. Precision (Prec.) and recall (Rec.) are
averaged per example within each setting; F1 is their harmonic mean. Corr.=correct classified / total ambiguous.
model also correctly flags the underlying query
as ambiguous: a high F1 on a small set of correctly-
flagged queries is weaker than the same F1 on a
larger set. We therefore report both the count of
correctly-flagged examples and the F1 of the result-
ing clarifications in Table 6.
5.3.1 Full Test Set
On the full test set, both Granite LoRAs lead on
F1: the 4.0-micro LoRA at 89.7% and the 4.1-3B
LoRA at 89.0%. The prompted baselines GPT-
5.6 Sol and GPT-OSS-20B post similar high preci-
sion but substantially lower recall, indicating that
without task-specific training models miss plausi-
ble alternatives grounded in the conversation or
retrieved documents. The gain from fine-tuning
is especially pronounced for lightweight models:
prompted Granite-4.0-micro reaches only 56.8%
F1, whereas its LoRA counterpart attains 89.7%.
5.3.2 Pre-Retrieval
On the pre-retrieval subset, generation quality re-
sults are more competitive across models. Both
Granite LoRAs achieve an F1 around 85%, close to
prompted GPT-5.6 Sol, GPT-OSS-20B and Granite-
4.1-30B. However, as seen in §5.2.2, these three
prompted models have much lower classification
accuracy (they tend to over-clarify). The LoRA’s
operational advantage is that it produces high-
quality clarifications while also reliably identify-
ingwhichqueries require them, avoiding the infla-
tion of generation volume that accompanies over-
clarification in prompted baselines.
5.3.3 Post-Retrieval
On the post-retrieval subset, the two Granite Lo-
RAs attain the highest F1, combining high preci-
sion with the highest recall. Among prompted mod-els, several baselines (such as GPT-4o and GPT-
OSS-120B) achieve strong precision but compar-
atively low recall. This asymmetry suggests that,
when grounded by retrieved documents, prompted
models produce confident but narrow clarifications:
they correctly select some document-supported op-
tions but frequently omit others that are equally
valid. The two Granite LoRAs’ simultaneous gains
in precision and recall indicate that fine-tuning is
effective not only at identifyingwhento clarify but
also at enumeratingwhichoptions to present.
5.4 Out-of-Domain (OOD) Evaluation
To assess whether the LoRA’s advantage general-
izes beyond the four training domains (ClapNQ,
Cloud, FiQA, Govt), we construct an OOD eval-
uation set in two new domains drawn from
MTRAG-UN (Rosenthal et al., 2026):Banking
(U.S. bank services) andTelco(telecommunica-
tions providers). The Granite-4.0-micro LoRA
maintains its lead over prompted baselines on both
classification and generation quality. This suggests
that the clarification skill acquired via ClariEdge
is task-general rather than domain-specific. Ap-
pendix I gives the OOD evaluation details.
6 Conclusion
In this work, we introduce ClariEdge, a new dataset
for clarifying ambiguous questions in RAG set-
tings, and show that LoRA fine-tuning on a small
model is more beneficial than prompting larger
LLMs. Hard negatives are an essential ingredient,
suggesting that teaching a model what not to do
is as important as teaching it what to do. The re-
sulting LoRA works well in both pre-retrieval and
post-retrieval settings, although the pre-retrieval
regime is a more challenging one.
8

Limitations
Despite the strong empirical results reported above,
our study has several limitations that circumscribe
its scope and suggest directions for future work.
Single-turn clarification.Our model generates
a single clarifying question rather than engaging
in multi-turn clarification dialogues. Practical de-
ployments may benefit from iterative clarification
strategies that progressively narrow the user’s in-
tent across multiple exchanges. Extending our ap-
proach to multi-turn clarification remains an open
direction.
English-only evaluation.Both the dataset and
evaluation are restricted to English. The cross-
lingual transferability of our method remains unex-
plored.
Acknowledgments
We would like to thank our annotators for their
high-quality work in generating and evaluating this
dataset.
References
Mohammad Aliannejadi, Julia Kiseleva, Aleksandr
Chuklin, Jeffrey Dalton, and Mikhail Burtsev. 2020.
Convai3: Generating clarifying questions for open-
domain dialogue systems (clariq).arXiv preprint
arXiv:2009.11352.
Mohammad Aliannejadi, Hamed Zamani, Fabio
Crestani, and W. Bruce Croft. 2019. Asking clar-
ifying questions in open-domain information-seeking
conversations. InProceedings of the 42nd Interna-
tional ACM SIGIR Conference on Research and De-
velopment in Information Retrieval. ACM.
Chinmaya Andukuri, Jan-Philipp Fränken, Tobias Ger-
stenberg, and Noah D. Goodman. 2024. Star-gate:
Teaching language models to ask clarifying questions.
InProceedings of the First Conference on Language
Modeling (COLM).
Zhenyu Cao, Ping Li, and Qingfeng Zhu. 2025. Icr:
Iterative clarification and rewriting for conversational
search. InProceedings of the 2025 Conference on
Empirical Methods in Natural Language Processing.
Association for Computational Linguistics.
Yue Chen, Chen Huang, Yang Deng, Wenqiang Lei,
Dingnan Jin, Junhong Liu, and Tat-Seng Chua. 2024.
Style: Improving domain transferability of asking
clarification questions in large language model pow-
ered conversational agents. InFindings of the As-
sociation for Computational Linguistics: ACL 2024.
Association for Computational Linguistics.Jeremy R. Cole, Michael J.Q. Zhang, Daniel Gillick, Ju-
lian Martin Eisenschlos, Bhuwan Dhingra, and Jacob
Eisenstein. 2023. Selectively answering ambiguous
questions. InProceedings of the 2023 Conference on
Empirical Methods in Natural Language Processing
(EMNLP). Association for Computational Linguis-
tics.
Yang Deng, Wenqiang Lei, Wenxuan Zhang, Wai Lam,
and Tat-Seng Chua. 2022. PACIFIC: Towards proac-
tive conversational question answering over tabular
and textual data in finance. InProceedings of the
2022 Conference on Empirical Methods in Natural
Language Processing, pages 6970–6984. Association
for Computational Linguistics.
Yang Deng, Lizi Liao, Liang Chen, Hongru Wang,
Wenqiang Lei, and Tat-Seng Chua. 2023. Prompt-
ing and evaluating large language models for proac-
tive dialogues: Clarification, target-guided, and non-
collaboration. InFindings of the Association for
Computational Linguistics: EMNLP 2023. Associa-
tion for Computational Linguistics.
Ahmed Elgohary, Denis Peskov, and Jordan Lee Boyd-
Graber. 2019. Can you unpack that? learning to
rewrite questions-in-context. InProceedings of the
2019 Conference on Empirical Methods in Natu-
ral Language Processing and the 9th International
Joint Conference on Natural Language Processing
(EMNLP-IJCNLP), pages 5918–5924.
Meiqi Guo, Mingda Zhang, Siva Reddy, and Malihe
Alikhani. 2021. Abg-coqa: Clarifying ambiguity in
conversational question answering. In3rd Confer-
ence on Automated Knowledge Base Construction
(AKBC).
Edward J. Hu, Yelong Shen, Phillip Wallis, Zeyuan
Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, and
Weizhu Chen. 2022. Lora: Low-rank adaptation of
large language models. InInternational Conference
on Learning Representations (ICLR).
Yannis Katsis, Sara Rosenthal, Kshitij Fadnis, Chu-
laka Gunasekara, Young-Suk Lee, Lucian Popa, Vraj
Shah, Huaiyu Zhu, Danish Contractor, and Marina
Danilevsky. 2025. Mtrag: A multi-turn conversa-
tional benchmark for evaluating retrieval-augmented
generation systems.Transactions of the Association
for Computational Linguistics, 13:784–808.
Gangwoo Kim, Sungdong Kim, Byeongguk Jeon, Joon-
suk Park, and Jaewoo Kang. 2023. Tree of clarifica-
tions: Answering ambiguous questions with retrieval-
augmented large language models. InProceedings of
the 2023 Conference on Empirical Methods in Natu-
ral Language Processing (EMNLP). Association for
Computational Linguistics.
Hyuhng Joon Kim, Youna Kim, Cheonbok Park, Jun-
yeob Kim, Choonghyun Park, Kang Min Yoo, Sang-
goo Lee, and Taeuk Kim. 2024. Aligning language
models to explicitly handle ambiguity. InProceed-
ings of the 2024 Conference on Empirical Methods in
9

Natural Language Processing. Association for Com-
putational Linguistics.
Lorenz Kuhn, Yarin Gal, and Sebastian Farquhar. 2023.
Clam: Selective clarification for ambiguous ques-
tions with generative language models. InProceed-
ings of the 40th International Conference on Machine
Learning (ICML).
Dongryeol Lee, Segwang Kim, Minwoo Lee, Hwan-
hee Lee, Joonsuk Park, Sang-Woo Lee, and Kyomin
Jung. 2023. Asking clarification questions to handle
ambiguity in open-domain qa. InFindings of the
Association for Computational Linguistics: EMNLP
2023, pages 11526–11544. Association for Computa-
tional Linguistics.
Zongxi Li, Yang Li, Haoran Xie, and S. Joe Qin. 2025.
CondAmbigQA: A benchmark and dataset for con-
ditional ambiguous question answering. InProceed-
ings of the 2025 Conference on Empirical Methods
in Natural Language Processing, pages 2269–2288.
Association for Computational Linguistics.
Sewon Min, Julian Michael, Hannaneh Hajishirzi, and
Luke Zettlemoyer. 2020. Ambigqa: Answering am-
biguous open-domain questions. InProceedings of
the 2020 Conference on Empirical Methods in Nat-
ural Language Processing (EMNLP), pages 5783–
5797. Association for Computational Linguistics.
Sanchit Nandi, Tanmay Agarwal, Abhishek Singh, and
Priyanshu Bhatt. 2026. Scaling intent understand-
ing: A framework for classification with clarification
using lightweight llms. InProceedings of the 2026
Conference of the European Chapter of the Associa-
tion for Computational Linguistics: Industry Track.
Association for Computational Linguistics.
Lang Qin, Yao Zhang, Hongru Liang, Adam Jatowt,
and Zhenglu Yang. 2025. Listening to patients: De-
tecting and mitigating patient misreport in medical
dialogue system. InFindings of the Association for
Computational Linguistics: ACL 2025. Association
for Computational Linguistics.
Hossein A. Rahmani, Xi Wang, Mohammad Alianne-
jadi, Mohammadmehdi Naghiaei, and Emine Yilmaz.
2024. Clarifying the path to user satisfaction: An
investigation into clarification usefulness. InFind-
ings of the Association for Computational Linguistics:
EACL 2024. Association for Computational Linguis-
tics.
Sara Rosenthal, Yannis Katsis, Vraj Shah, Lihong He,
Lucian Popa, and Marina Danilevsky. 2026. Mtrag-
un: A benchmark for open challenges in multi-turn
rag conversations.Preprint, arXiv:2602.23184.
Rishav Sahay, Lavanya Sita Tekumalla, Purav Aggar-
wal, Arihant Jain, and Anoop Saladi. 2025. Ask:
Aspects and retrieval based hybrid clarification in
task oriented dialogue systems. InProceedings of
the 63rd Annual Meeting of the Association for Com-
putational Linguistics: Industry Track. Association
for Computational Linguistics.Ivan Sekuli ´c, Mohammad Aliannejadi, and Fabio
Crestani. 2021. Towards facet-driven generation of
clarifying questions for conversational search. In
Proceedings of the 2021 ACM SIGIR International
Conference on Theory of Information Retrieval (IC-
TIR). ACM.
Ivan Stelmakh, Yi Luan, Bhuwan Dhingra, and Ming-
Wei Chang. 2022. ASQA: Factoid questions meet
long-form answers. InProceedings of the 2022 Con-
ference on Empirical Methods in Natural Language
Processing, pages 8273–8288. Association for Com-
putational Linguistics.
Alberto Testoni and Raquel Fernández. 2024. Asking
the right question at the right time: Human and model
uncertainty guidance to ask clarification questions.
InProceedings of the 18th Conference of the Euro-
pean Chapter of the Association for Computational
Linguistics. Association for Computational Linguis-
tics.
Hamed Zamani, Susan Dumais, Nick Craswell, Paul
Bennett, and Gord Lueck. 2020. Mimics: A large-
scale data collection for search clarification. InPro-
ceedings of the 29th ACM International Conference
on Information and Knowledge Management (CIKM).
ACM.
Michael J.Q. Zhang and Eunsol Choi. 2023. Clarify
when necessary: Resolving ambiguity through inter-
action with lms. InProceedings of the 2023 Con-
ference on Empirical Methods in Natural Language
Processing (EMNLP). Association for Computational
Linguistics.
Michael J.Q. Zhang, W. Bradley Knox, and Eunsol Choi.
2025. Modeling future conversation turns to teach
llms to ask clarifying questions. InProceedings of
the Thirteenth International Conference on Learning
Representations (ICLR).
Tong Zhang, Peixin Qin, Yang Deng, Chen Huang, Wen-
qiang Lei, Junhong Liu, Dingnan Jin, Hongru Liang,
and Tat-Seng Chua. 2024. Clamber: A benchmark
of identifying and clarifying ambiguous information
needs in large language models. InProceedings of
the 62nd Annual Meeting of the Association for Com-
putational Linguistics (Volume 1: Long Papers). As-
sociation for Computational Linguistics.
A Dataset Distributions
In our dataset, each example is tagged by retrieval
setting:pre-retrievalexamples expose the model
to the conversation only, whereaspost-retrieval
examples additionally include the documents re-
turned by the retrieval stage. Figure 3a shows that
pre-retrieval and post-retrieval examples are rep-
resented in roughly equal proportions. From Fig-
ure 3b, most conversations are multi-turn, with
only a small fraction consisting of a single turn.
10

Pre-retrieval49.4%
(n=1,922) Post-retrieval50.6%
(n=1,970)Retrieval setting(a) Pre- vs. post-retrieval split.
1
18.8%
228.1%
323.9%417.9%5
7.5%6
3.9%# conversation turns (b) Number of conversation turns.
1
4.1%2
14.2%
350.7%4 12.8%5
7.5%6--10
6.4%>10
4.3%# documents (c) Number of documents (post-retrieval).
Figure 3: Data split in pre-/post-retrieval and distribution of conversation turns and retrieved document counts.
From Figure 3c, most post-retrieval examples are
associated with at least three retrieved documents.
B Human Annotation and Review
This appendix provides additional details on the
human annotation and review processes used in
creating the dataset (Section 3.2). Section B.1 fo-
cuses on approach (1) for Ambiguous examples,
where annotators wrote ambiguous questions in-
spired by entity types and instances drawn from
the underlying corpus, and each such question was
then stitched, in a subsequent step, as the last turn
onto an existing human-annotated multi-turn con-
versation. Section B.2 focuses on approach (2)
for Ambiguous examples and the construction of
CLEAR-hard examples, both of which combine
LLM-based synthetic generation with manual re-
view (by the authors) to separate truly ambiguous
queries from those resolvable from the conversa-
tion context. Both of these processes rely on an
existing set of general (i.e., with no ambiguity)
multi-turn conversations that were created previ-
ously by annotators (as described in (Katsis et al.,
2025)).
B.1 Human-Authored Ambiguous Questions
Based on the corpus of documents underlying each
domain (we used ClapNQ and Gov), we first cre-
ated a set ofentity types, each populated with a set
of concreteentity instancesdrawn from that cor-
pus. The annotators were then given these entity
types and instances, together with instructions to
pick an entity type and write an ambiguous ques-
tion that could be referring to any one of the entity
instances of that type. As part of the instructions,
the annotators were also given examples of ambigu-
ous questions to illustrate the desired pattern. For
example, part of the guidelines read:
If you select the Parks entity, you could
ask about the opening hours of a park, asfollows: “What are the park’s opening
hours?”
where the concrete entity instances available for the
Parks entity included: “Wilder Ranch State Park”,
“Sampson State Park”, “Leo Carrillo State Park”,
“Mount San Jacinto State Park”, and others.
The annotators were instructednotto mention
any of the concrete entity instances explicitly in the
question, but rather to use them only as inspiration
for crafting a question that is ambiguous on pur-
pose. In the example above, the phrase “the park”
can be referring to any one of the underlying entity
instances from the corpus, which is precisely the
kind of ambiguity we wish to elicit.
Stitching into multi-turn conversations.In a
subsequent (programmatic) step, each human-
authored ambiguous question was stitched as the
last turn onto an existing human-annotated multi-
turn conversation. Stitching was done in two ways:
(a) onto conversations on the same or very similar
topic, simulating the case where the new turn is a
natural continuation of the dialog (about 75% of
the cases), and (b) onto conversations on a different
topic, so that the new turn reflects a topic switch by
the user while still being an ambiguous question
(about 25% of the cases). The latter case adds an
additional challenge for the model, reflecting the
realistic scenario where users sometimes change
topics and the system must still detect ambiguity
and react accordingly.
The resulting conversations went through a care-
ful manual validation step (performed by the au-
thors) that included filtering out cases where the
addition of the prior conversation context acciden-
tally resolved the intent of the last turn.
B.2 Manual Review of Synthetically
Generated Questions
Approach (2) for Ambiguous examples and the
construction of CLEAR-hard examples both rely
11

on LLM-based synthetic generation followed by
manual review by the authors of this paper. Con-
cretely, an LLM was prompted to generate a final
user query targeting an entity type with multiple
instances occurring in the conversation, with the
intent that the query be ambiguous (see Section 3.2
for the prompting details). However, the synthetic
generation step is far from perfect: the model fre-
quently produces questions that, despite mention-
ing an entity type with several instances, are in
fact resolvable from the prior conversation through
coreference or normal dialog flow. To address this,
we use manual review (by the authors).
Figure 4 illustrates the distinction the manual
review must draw. On the left, the user’s last turn
“Did he have a significant impact on the movement?”
refers to an entity (a Dadaist artist) for which mul-
tiple plausible referents have been introduced in
the conversation, and the question cannot be an-
swered without first clarifying which artist; this
example is genuinely ambiguous. On the right, the
user’s last turn “What are their capabilities?” super-
ficially looks ambiguous because of the pronoun
“their”, but standard coreference resolution against
the immediately preceding agent turn unambigu-
ously binds it to “bots”; this example is therefore
resolvable, and is kept as a CLEAR-hard example
rather than as an Ambiguous example.
This boundary, between truly ambiguous queries
and queries that merely look ambiguous but can be
resolved from the conversational context, is exactly
what the synthetic generation models tended to
confuse, and is also the boundary that downstream
clarification models must learn. The manual review
(by the authors) was essential for assigning each
generated example to the correct category, and also
for surfacing various other patterns of false-positive
ambiguity (e.g., “which one” comparisons, regu-
lar follow-up questions, terse keyword queries) in
addition to coreference resolution patterns.
C Clarification Response Patterns
This appendix provides representative examples of
the three clarification response patterns described
in Section 3.3:Hedging with answers(2–3 op-
tions),Hedging over list(4–8 options), andOpen-
domain(many or unbounded options). Table 7 illus-
trates each pattern with a user query, the conversa-
tional or document context, and the corresponding
clarification response.D LoRA Implementation Details
Training configuration.We fine-tune LoRA
models with rank 32 and scaling factor α= 64 ,
applied to both attention and MLP layers of the
base model. Models are trained on 1 H100 GPU
with learning rate of 1×10−5and dropout of 0.05.
Training runs for up to 15 epochs (minimum 5)
with early stopping based on validation-set perfor-
mance to mitigate overfitting.
Input and output format.Our LoRA models
accept flexible inputs: the conversation history
and optional set of retrieved documents for the
last user turn. The model adapts its decision
based on the available context: when documents
are absent, it relies on the conversation history
alone; when documents are present, it leverages
both sources. The output is a JSON object of
the form {"clarification": "CLEAR" |
[clarifying question]} . When the query
is unambiguous, the model outputs CLEAR ; other-
wise, it outputs a clarifying question (e.g.,“Which
credit card are you asking about: the Gold Card or
the Business Card?”).
E Prompt for Baseline Models
This appendix provides the full text of the sys-
tem and user messages used to evaluate all seven
prompted baselines in Section 4.2. All baselines
receive the same templates; only the user-message
content varies across test records.
System message.The system message defines
the task, grounding requirements, and three re-
sponse patterns corresponding to those used to gen-
erate reference clarifications (Section 3.3):
You are a clarification assistant.
Your task: decide if the user’s last
question is ambiguous or clear, using
only the conversation history and the
retrieved documents provided.
GROUNDING:
- Use ONLY information present in the
conversation history or retrieved docs.
- Do NOT invent entities or details that
are not in the context.
OUTPUT LOGIC:
- If the last user question has a clear
interpretation that can be answered
unambiguously, output exactly: CLEAR
- Otherwise, produce one clarifying
question, following:
(a) If there are 2-3 plausible options:
list EACH with a short, factual
description, then ask the user
12

U:dadaists
A:The Dadaists were an art movement in the early
20th century, around the time of World War I. They
had centers in Zürich, Switzerland, New York, and
Paris. The artists rejected logic, reason, . . .
U:Didhehave a significant impact onthe move-
ment?
(a)Truly ambiguous( ✓kept as Ambiguous): “he” has multi-
ple plausible Dadaist referents introduced earlier; cannot be
resolved from context.U:. . .
A:bots are computer programs that automatically do
certain actions on a network. They are programmed
to do things at scale . . .
U:What aretheircapabilities?
(b)Resolvable( ✗kept as CLEAR-hard): “their” is bound by
coreference to “bots” in the immediately preceding turn; clear
interpretation.
Figure 4: Two synthetically generated last turns that an LLM produced as “ambiguous”. The manual review step
distinguishes between genuinely ambiguous queries (left) and queries that are resolvable from the conversation
context, e.g., via coreference (right). The latter become CLEAR-hard examples.
Pattern # Opt. User Query Context Clarification Response
Hedging
with
answers2–3 “In which country is
this language offi-
cially spoken?”Docs: introduce languages
such as Spanish, Italian, and
Xhosa“There are many languages you could be re-
ferring to, such as: 1) Spanish, which is
officially spoken in Spain, 2) Italian, which
is officially spoken in Italy, Switzerland, San
Marino, Vatican City, and western Istria (in
Slovenia and Croatia), and 3) Xhosa, which
is officially spoken in South Africa and Zim-
babwe. Which one are you talking about?”
Hedging
over list4–8 “How did the fast
food chain establish
a significant pres-
ence in the US?”Docs: describe fast food
chains such as In-N-Out
Burger, Jack in the Box, Big
Boy Restaurants, Chipotle
Mexican Grill“There are many fast food chains you could
be talking about, like In-N-Out Burger, Jack
in the Box, Big Boy Restaurants, or Chipotle
Mexican Grill. Which one are you referring
to?”
Open-
domain>8 “Tell me about the
American poet’s
work”Multiple American poets
discussed in conversation“Which American poet are you referring to?”
Table 7: Clarification response patterns with examples.
which one applies.
(b) If there are 4-8 options: list them
succinctly, then ask which one
applies.
(c) If there are >8 options OR no
concrete options can be enumerated
from context: ask an open-ended
question (e.g., "Which <entity>
are you referring to?").
STYLE:
- Be concise, neutral, and grounded.
- If listing options, they must be
supported by the retrieved docs or
conversation history.
User message.The user message is constructed
from each test record and includes the conversa-
tion history, retrieved documents (if any), and the
user’s last question, with placeholder strings when
a component is empty:
CONVERSATION HISTORY:
<prior user/assistant turns, or "(none)">
RETRIEVED DOCUMENTS:
<top-K retrieved documents, or "(no
retrieved documents)">LAST USER QUESTION:
<the user’s last message>
Respond with either a grounded
clarifying question, or exactly CLEAR.
F Prompts for Generation Quality
Metrics
For the entity-option-centric metrics defined in Sec-
tion 5.1, an option is considered correct if it appears
in the gold reference, is grounded in the retrieved
documents, or is consistent with general knowledge
and user intent. To compute precision and recall
on free-form clarification responses, we need to
implement judgments: (i) whether a clarification
is open-ended, (ii) extracting the entity options it
lists, (iii) checking whether an option is relevant
to the ambiguous entity, and (iv) checking whether
an option is covered by a reference list. We ob-
tain these judgments with four LLM prompts. All
four prompts are executed with Llama-3.3-70B-
13

Instruct5. Placeholders in double braces (e.g., {{
clarification question }} ) are filled in
at inference time.
Open-ended check.Decides whether a clarifica-
tion response is open-ended.
[Instruction]
You are an assistant that determines
whether a clarification question is
open-ended. Output "yes" if it is
open-ended, "no" otherwise.
A clarification question is open-ended
(yes) if it allows for answers beyond
a fixed, limited set of options. A
clarification question is not
open-ended (no) if it restricts the
answer to a fixed and exhaustive set
of options.
Rules:
- If the question contains illustrative
or example language such as "like",
"such as", "for example",
"including", or "include" anywhere in
the question (including earlier
sentences), treat the listed options
as non-exhaustive and return yes,
even if the final sentence asks
"Which one are you referring to?".
- If the question lists choices plus a
vague catch-all option (e.g.,
"other", "else", "another"), return
yes.
- If the question does not specify any
concrete options and only contains a
general clarification such as "Which
[entity] do you mean?", return yes.
- If the question explicitly restricts
the answer to a closed set of
concrete options using language like
"are you talking about A, B, or C",
return no.
Clarification question:
{{ clarification question }}
Entity-option extraction.Extracts explicit entity
options from the clarification response, separating
concrete instances from catch-all choices.
[Instruction]
You are an assistant that extracts
explicit options from clarification
questions. Output concrete entity
options into "instances" list and
catch-all options into "extra" list.
Rules and steps:
1. Identify all explicit options listed
in the clarification question.
- Put concrete entity names (e.g.,
"UK", "Prime Minister of
Australia") into "instances".
- Put catch-all choices (e.g.,
5https://huggingface.co/meta-llama/
Llama-3.3-70B-Instruct"else", "other", "another one")
into "extra".
2. If an option has a description,
extract only the main entity name,
not the explanation.
3. Ignore numbering, bullet points, or
letters (e.g., 1), 2., a), b)), and
do not include them in the output.
4. If the question contains no explicit
options, return "instances": [] and
"extra": [].
5. Do not infer or add options that are
not explicitly written in the text.
Clarification question:
{{ clarification question }}
Relevance check.Checks whether an extracted
option is a valid instance of the ambiguous entity
in the user query.
[Instruction]
You are a relevance checker. Your task
is to determine whether a given check
choice is relevant to the ambiguous
entity in a user question. Output
"yes" if it is relevant, "no"
otherwise.
Rules and steps:
1. Identify the ambiguous entity type
in the user question (e.g.,
"ocean", "actor", "city").
2. Decide if the check choice is
directly and literally a valid
instance of that entity type.
- The choice must exactly belong to
the same entity type as the
ambiguous entity, not just be
related or associated with it.
- Do not infer substitutions (e.g.,
a character is not the same as an
actor; a water mass is not the
same as an ocean).
- If the choice is related but not
the same type, return no.
3. If the choice correctly specifies or
narrows down the ambiguous entity,
return yes.
User question:
{{ user question }}
Check choice:
{{ check choice }}
Coverage check.Checks whether one option is
covered by (or equivalent to) any option in a refer-
ence list.
[Instruction]
You are a coverage checker. Your task
is to determine if a check choice is
explicitly covered by or equivalent to
any choice in a given list of compared
choices. Output "yes" if it is
covered, "no" otherwise.
Rules:
1. If the check choice matches exactly
14

(or is equivalent to) one of the
compared choices, return yes.
Otherwise, return no.
2. Equivalence of two choices:
- They must refer to the same entity
itself (not something merely
related).
- Valid equivalence cases:
- Exact matches (e.g., "UK" vs
"UK").
- Substrings, abbreviations, or
aliases of the same entity
(e.g., "UK" vs
"United Kingdom").
- A country name and a role
explicitly tied to that country
are equivalent. Example:
"Canada" == "Prime Minister of
Canada".
- Invalid equivalence cases:
- Different entity types where
one is not a country-role
relationship (e.g., character
vs actor, sea current vs
ocean).
Check choice:
{{ check choice }}
Compared choices:
{{ compared choices }}
G Examples of Generation Quality
Metrics
Table 8 provides two worked examples that illus-
trate the entity-option-centric precision, recall, and
F1 metrics defined in Section 5.1. Example 1 shows
ahedging over listclarification in which both the
gold and generated clarifications enumerate spe-
cific options. Example 2 shows anopen-domain
clarification in which the gold reference poses an
open-ended question rather than listing specific op-
tions. For such open-domain cases, we introduce
an[extra] token to represent the open-ended na-
ture of the question: any correct generated option
counts toward precision, and the [extra] token in
the gold is considered covered as long as the gen-
erated clarification also includes an open-ended
component.
H Ablation Study on Hard Negatives
To isolate the contribution of hard negatives to the
learned decision boundary, we design an ablation
experiment in which the only variable is the pres-
ence of hard negatives in the training set. Specif-
ically, we compare two Granite-4.0-micro LoRA
configurations: one trained on the full training set,
and one trained on the same data but with all hard
negatives removed. Evaluation is performed on thefull test set, enabling a direct measurement of how
hard negatives influence both classification accu-
racy across the three categories and clarification
generation quality.
Table 9 shows that hard negatives are critical
for the model’s ability to distinguish genuinely
clear queries that look ambiguous. Without them,
CLEAR-hard accuracy collapses to 35.4%, while
Ambiguous detection inflates to 97.1%. With hard
negatives, the model achieves a much better bal-
ance. Generation quality also improves with hard
negatives included in the training, likely because
the LoRA model learns more nuanced boundaries
between ambiguous and clear queries, leading to
higher-quality clarifications when it does decide to
clarify.
I Out-of-Domain Evaluation Details
The ClariEdge training and test data span four do-
mains inherited from MTRAG (ClapNQ, Cloud,
FiQA, Govt). To assess how the fine-tuned Lo-
RAs can transfer to previously unseen domains, we
construct an additional evaluation set in two new
domains, Banking and Telco, drawn from MTRAG-
UN (Rosenthal et al., 2026). MTRAG-UN provides
79 Banking and 80 Telco conversations, which
we retain as-is for the CLEAR-random category
(their metadata already indicates that the queries
are clear). We then generate CLEAR-hard and
Ambiguous examples in each domain following
the data-creation process described in Section 3.2.
The Banking evaluation set contains 230 examples
(79 Ambiguous, 72 CLEAR-hard, and 79 CLEAR-
random), of which 120 are pre-retrieval and 110
are post-retrieval. The Telco evaluation set con-
tains 235 examples (80 Ambiguous, 75 CLEAR-
hard, and 80 CLEAR-random), of which 123 are
pre-retrieval and 112 are post-retrieval. We evalu-
ate the Granite-4.0-micro LoRA against the same
prompted baselines used in the in-domain evalua-
tion.
Classification results.Table 10 reports classifi-
cation accuracy on the two OOD domains. The
Granite-4.0-micro LoRA leads on overall accu-
racy in both domains, reaching 92.6% on Bank-
ing and 90.2% on Telco, close to its 92.4% in-
domain accuracy on the ClariEdge test set (Ta-
ble 5). The class-imbalance pattern of prompted
models observed in-domain persists here: the larger
prompted models (GPT-5.6 Sol, GPT-OSS-120B,
GPT-OSS-20B, and Granite-4.1-30B) over-clarify
15

Example 1: Hedging over list Example 2: Open-domain
User query “Is the language that descended from Common
Brittonic still spoken today, or has it become ex-
tinct?”“Where can I find information on the park ameni-
ties available at the New York county?”
Gold reference “There are many languages that descended from
Common Brittonic, such as: 1) Welsh, 2) Breton,
and 3) Cornish. Which one are you referring to?”“Which park are you interested in?”
Generated clarifi-
cation“There are several languages that descended from
Common Brittonic, such as Welsh, Cornish, and
Cumbric. Which one are you referring to?”“There are many parks in New York county, such
as Harriman State Park, Bear Mountain State
Park, Minnewaska State Park, Rockland Lake
State Park, and others. Which one are you inter-
ested in?”
Gold options{Welsh, Breton, Cornish} {[extra] } (open-ended, no specific options
enumerated)
Generated op-
tions{Welsh, Cornish, Cumbric} {Harriman State Park, Bear Mountain State Park,
Minnewaska State Park, Rockland Lake State
Park,[extra]}
Precision 3/3 = 1.0 (Cumbric is a valid Brittonic language,
thus correct)5/5 = 1.0 (all four parks are valid; open-ended
component is appropriate)
Recall 2/3 = 0.67 (Breton is missing from generated
clarification)1/1 = 1.0 ( [extra] covered by the open-ended ques-
tion component)
F10.80 1.0
Table 8: Examples of generation quality metric calculation.
Classification Accuracy Generation Quality
Training Configuration AmbiguousCLEAR-
randomCLEAR-
hard Overall Correct Prec. Rec. F1
All training data 84.6% 98.5% 87.3%92.4%88/104 91.7% 87.5%89.5%
Hard negatives removed 97.1% 97.5% 35.4% 84.6% 101/104 88.1% 81.9% 84.9%
Table 9: Ablation study on Hard Negatives. The fine-tuned model is Granite-4.0-micro LoRA.
(high Ambiguous, low CLEAR-hard), whereas the
smaller prompted Granite models (Granite-4.1-3B
and Granite-4.0-micro) under-clarify. Only the
LoRA maintains balance across the three categories
on both OOD domains, which suggests that its abil-
ity to distinguish genuine ambiguity from resolv-
able surface ambiguity transfers to new domains
rather than being overfit to the four training do-
mains.
Generation quality results.Table 11 reports the
entity-option-centric generation quality on the sub-
set of examples where each model correctly pre-
dicted Ambiguous. The Granite-4.0-micro LoRA
achieves the highest F1 on Telco (89.28%). On
Banking, prompted GPT-4o (89.82%) and Granite-
4.1-30B (86.61%) edge ahead of the LoRA in F1
(81.56%), but on smaller correctly-flagged subsets
(42/79 and 60/79 vs. 69/79 for the LoRA), so their
higher F1 comes with substantially lower coverage.
Taken together, the OOD results indicate that
the LoRA’s advantage over prompted baselines
extends beyond the four training domains. Theoverall picture (balanced classification across cate-
gories, high F1 on correctly-flagged clarifications,
and a large gap over prompted baselines of sim-
ilar size) matches the in-domain findings in Sec-
tions 5.2 and 5.3, suggesting that the clarification
skill acquired via ClariEdge is task-general rather
than domain-specific.
16

Banking (230) Telco (235)
Model Ambig. C-hard C-rand. Overall Ambig. C-hard C-rand. Overall
Granite-4.0-micro LoRA 87.3% 93.1% 97.5% 92.6% 82.5% 89.3% 98.8% 90.2%
GPT-4o 53.2% 61.1% 96.2% 70.4% 48.8% 68.0% 93.8% 70.2%
GPT-5.6 Sol 86.1% 29.2% 67.1% 61.7% 82.5% 38.7% 67.5% 63.4%
GPT-OSS-120B 73.4% 44.4% 79.7% 66.5% 72.5% 54.7% 87.5% 71.9%
GPT-OSS-20B 81.0% 37.5% 63.3% 61.3% 80.0% 46.7% 72.5% 66.8%
Granite-4.1-30B 75.9% 27.8% 74.7% 60.4% 71.3% 49.3% 85.0% 68.9%
Granite-4.1-3B 22.8% 80.6% 86.1% 62.6% 40.0% 82.7% 83.8% 68.5%
Granite-4.0-micro 7.6% 77.8% 81.0% 54.8% 10.0% 74.7% 80.0% 54.5%
Table 10: Out-of-domain classification accuracy.
Banking (79 Ambiguous) Telco (80 Ambiguous)
Model Corr. Prec. Rec. F1 Corr. Prec. Rec. F1
Granite-4.0-micro LoRA 69/79 83.81% 79.42% 81.56% 66/80 95.03% 84.19% 89.28%
GPT-4o 42/79 89.48% 90.16% 89.82% 39/80 86.67% 89.32% 87.97%
GPT-5.6 Sol 68/79 87.75% 67.06% 76.02% 66/80 93.52% 75.51% 83.55%
GPT-OSS-120B 58/79 84.99% 71.90% 77.90% 58/80 87.64% 78.99% 83.09%
GPT-OSS-20B 63/79 83.20% 67.41% 74.48% 64/80 86.46% 71.74% 78.42%
Granite-4.1-30B 60/79 92.88% 81.14% 86.61% 57/80 79.90% 82.28% 81.07%
Granite-4.1-3B 18/79 80.37% 66.67% 72.88% 31/80 78.87% 69.09% 73.65%
Granite-4.0-micro 6/79 66.67% 23.61% 34.87% 5/80 70.00% 66.67% 68.29%
Table 11: Out-of-domain generation quality on examples correctly classified as ambiguous.
17