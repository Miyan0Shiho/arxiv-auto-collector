# Beyond Retrieval Failure: Understanding Why LLMs Miss Evidence in Multi-Hop RAG

**Authors**: Jiaqi Wei, Wenxuan Huang, Chengdong Xu, Xiang Zhang, Wanli Ouyang

**Published**: 2026-10-15 03:39:25

**PDF URL**: [https://arxiv.org/pdf/2610.21184v1](https://arxiv.org/pdf/2610.21184v1)

## Abstract
Multi-hop retrieval-augmented generation (RAG) requires an agent to retrieve evidence for a sequence of dependent sub-questions. When a wrong answer lacks a required passage, the passage may be inaccessible to the retriever, or the agent may have failed to issue a query that would retrieve it. Standard retrieval evaluations conflate these two causes. We study how much apparent retrieval failure is instead caused by search drift: the search no longer tracks an unanswered sub-question. Holding the retriever, corpus, and generation model fixed, we introduce ExecutionRAG, a training-free requirement register that records open sub-questions and their supporting evidence. The register changes search guidance but not evidence access, so each gold passage it recovers certifies that the passage was reachable by the original retriever. On 1,000 MuSiQue questions, the number of wrong answers with unretrieved required evidence falls from 382 to 234 with all-MiniLM-L6-v2 and from 383 to 274 with Qwen3-Embedding-0.6B. These counts correspond to 69% to 44% and 66% to 48% of each system's wrong answers, respectively. Crucially, evidence recovery does not produce a proportional reduction in total error: premature resolutions and abstentions increase as missing-evidence failures decrease. The central finding is therefore a shift in the remaining bottleneck, from guiding search toward required evidence to verifying that the retrieved evidence is sufficient.

## Full Text


<!-- PDF content starts -->

Beyond Retrieval Failure: Understanding Why LLMs Miss
Evidence in Multi-Hop RAG
Jiaqi Wei∗
Zhejiang University
Hangzhou, China
Shanghai AI Laboratory
Shanghai, China
jiaqi.wei@zju.edu.cnWenxuan Huang∗
Fudan University
Shanghai, China
winshion0819@gmail.comChengdong Xu
Sun Yat-sen University
Guangzhou, China
codepoet2077@gmail.com
Xiang Zhang
University of British Columbia
Vancouver, Canada
xzhang23@ualberta.caWanli Ouyang
The Chinese University of Hong Kong
Hong Kong SAR, China
wlouyang@ie.cuhk.edu.hk
Abstract
Multi-hop retrieval-augmented generation (RAG) requires an agent
to retrieve evidence for a sequence of dependent sub-questions.
When a wrong answer lacks a required passage, the passage may
be inaccessible to the retriever, or the agent may have failed to
issue a query that would retrieve it. Standard retrieval evaluations
conflate these two causes. We study how much apparent retrieval
failure is instead caused bysearch drift: the search no longer tracks
an unanswered sub-question. Holding the retriever, corpus, and
generation model fixed, we introduce ExecutionRAG, a training-
free requirement register that records open sub-questions and their
supporting evidence. The register changes search guidance but not
evidence access, so each gold passage it recovers certifies that the
passage was reachable by the original retriever. On 1,000 MuSiQue
questions, the number of wrong answers with unretrieved required
evidence falls from 382 to 234 with all-MiniLM-L6-v2 and from 383
to 274 with Qwen3-Embedding-0.6B. These counts correspond to
69% to 44% and 66% to 48% of each system’s wrong answers, respec-
tively. Crucially, evidence recovery does not produce a proportional
reduction in total error: premature resolutions and abstentions in-
crease as missing-evidence failures decrease. The central finding is
therefore a shift in the remaining bottleneck, from guiding search
toward required evidence to verifying that the retrieved evidence
is sufficient.
CCS Concepts
•Information systems →Retrieval models and ranking;Ques-
tion answering;•Computing methodologies →Natural lan-
guage processing.
Keywords
Retrieval-augmented generation, multi-hop question answering,
agentic retrieval, search drift, failure analysis
ACM Reference Format:
Jiaqi Wei, Wenxuan Huang, Chengdong Xu, Xiang Zhang, and Wanli
Ouyang. 2026. Beyond Retrieval Failure: Understanding Why LLMs Miss
∗Both authors contributed equally to this research.
DAI ’26, Hong Kong, ChinaEvidence in Multi-Hop RAG. InThe Eighth International Conference on
Distributed Artificial Intelligence (DAI ’26), November 29–December 2, 2026,
Hong Kong, China.ACM, New York, NY, USA, 17 pages.
1 Introduction
Retrieval-augmented generation (RAG) [ 7,20] grounds language-
model answers in retrieved evidence. This is especially important
for multi-hop questions, whose answers depend on several doc-
uments. Consider the question“Who married the publisher of the
abolitionist newspaper The North Star?”Answering it requires two
linked facts: Frederick Douglass publishedThe North Star, and He-
len Pitts Douglass was his wife. A retrieval-augmented agent may
search for “The North Star, publisher, married,” anchor on William
Cooper Nell, who wrote for rather than published the paper, and
spend the remaining budget on Nell’s marriage. Both required facts
remain in the corpus. The failure is not evidence access; it is the
loss of the intermediate entity that should guide the next query.
Most work on complex RAG focuses on evidence access. Methods
organize a corpus into graphs or hierarchies to make passages
easier to retrieve [ 6,8,31], while agentic methods determine what
to retrieve next. These approaches do not address the example
above when the retriever can already return the needed passage
but the search no longer represents the intermediate target. On
MuSiQue, maintaining the unresolved sub-questions across steps,
without changing the retriever or corpus, reduces the share of
wrong answers that lack a required passage from 382 to 234 cases
under one retriever and from 383 to 274 under another. This result
motivates a different question: how much apparent retrieval failure
can be reduced by improving search guidance without changing
retriever coverage?
We call this failuresearch drift. During a multi-hop search, the
agent may discard an unanswered sub-question, anchor on the
wrong intermediate entity, or keep querying for facts it has already
found [ 18,19,30]. Search drift appears as a retrieval symptom
because a required passage is absent from the trace, even though it
is present in the corpus. It is also the dominant failure category for
a strong agentic baseline on MuSiQue: 66–69% of wrong answers
omit at least one required passage, and this share increases from
62% at two hops to 81% at four hops.
arXiv:2610.21184v1  [cs.CL]  15 Oct 2026

DAI ’26, November 29–December 2, 2026, Hong Kong, China Wei et al.
ExecutionRAG implements this measurement with an explicit
requirement register. The register is a list of the question’s unre-
solved sub-questions, initialized from the question and seed evi-
dence rather than from the question alone. Each entry records its
state and the passages that support it. Open entries guide subse-
quent search; an entry is retired only when later evidence contra-
dicts it; and the agent answers only after no entry remains open.
The language model still decides whether evidence resolves an en-
try. Two deterministic rules preserve the record by preventing open
entries from silently disappearing and by retaining all previously
recorded evidence.
The main result is not simply that the register improves accuracy,
but that it reveals what happens after more evidence is recovered.
Missing-evidence failures decrease because the agent continues
to search for unresolved entries, yet premature resolutions and
abstentions increase, so total error falls by a much smaller amount.
Because each resolution is explicit, we can compare it with gold
supporting evidence and identify cases in which the agent answers
before the evidence is sufficient. The register therefore exposes a
transition from a search guidance problem to an evidence verifica-
tion problem.
In summary, this paper makes the following contributions:
•We distinguish two causes of missing required evidence: evidence
that the fixed retriever cannot reach and evidence that the search
never requests. This distinction turns evidence recovery under
fixed retrieval into a diagnostic of search drift.
•We show a substantial controlled reduction in missing-evidence
failures. On MuSiQue, the count decreases from 382 to 234 with
all-MiniLM-L6-v2 and from 383 to 274 with Qwen3-Embedding-
0.6B. However, premature resolutions and abstentions increase,
making evidence verification the principal remaining bottleneck.
•We introduce ExecutionRAG, a training-free requirement register
that provides the controlled intervention. It tracks unresolved
sub-questions and their supporting passages while leaving the
retriever, corpus, and generation-model parameters unchanged.
2 Problem Setup: What “Retrieval Failure”
Hides
Given a query 𝑞and a corpusK, conventional RAG retrieves context
Cand generates an answer in one shot, C=𝑅(𝑞 ;K)and𝑎=
𝐺(𝑞,C) , treating the retrieved passages as a single context. For a
multi-hop question, the challenge is not only whether the right
passages are retrieved, but also whether the agent retains which
sub-questions are resolved and which remain open as the search
proceeds.
2.1 Two Causes Inside One Symptom
Multi-hop benchmarks let us say precisely what went wrong. For a
question𝑞, let𝑃★(𝑞)be itsgold supporting passages: the minimal
evidence a correct answer needs. We call a wrong answer anap-
parent retrieval failurewhen some 𝑝∈𝑃★(𝑞)was never retrieved.
Read literally, the name blames the retriever. But with 𝑅andK
held fixed, the event splits into twodisjointcauses:unreachable,
no query the agent could have issued would have made 𝑅return𝑝;
andunasked, some query would have returned 𝑝, but the agent
never issued it. Only the first is a retrieval problem; the second issearch drift, where the evidence was within reach the whole time
and the search stopped asking for it. The two are indistinguishable
in a trace, because both end with the same observable, namely a
missing passage. The category is therefore routinely attributed to
the retriever.
2.2 What a Fixed Retriever Identifies
The unasked share is not directly observable: deciding thatno
query would have worked means quantifying over a query space
we cannot enumerate. One direction, however, is decidable. If an
intervention leaves 𝑅andKuntouched and the needed passage is
nonetheless retrieved, then a query returning it existed, so that fail-
ure was unasked. Recovering evidencecertifies reachability; failing
to recover it proves nothing. Holding the retriever fixed is therefore
necessary for the inference.
This argument identifies reachability at the level of an individual
recovered passage. At the aggregate level, however, a reduction
in the frequency of missing-evidence failures does not by itself
identify which baseline failures were converted, because the two
systems can fail on different questions. We therefore use the con-
trolled reduction as evidence of a recoverable component of appar-
ent retrieval failure, not as a numerical lower bound on the total
prevalence of search drift. The minimal intervention keeps this
interpretation specific to search guidance without claiming that
every remaining missing passage is unreachable.
2.3 The Probe: An Explicit Requirement
Register
We therefore need an intervention that keeps the open sub-
questions in view while changing nothing about retrieval: an
explicitrequirement register, in effect a checklist the agent must
clear before it answers. For a query 𝑞, letD(𝑞)={𝑑 1,...,𝑑𝑚}be
the question’s answer requirements, namely the sub-questions the
answer must satisfy (e.g., “who is the publisher” and “who did that
person marry”). At step 𝑡the register records, for each item, its
status𝑧(𝑡)
𝑖, a short evidence note 𝑒(𝑡)
𝑖, and the passages 𝑃(𝑡)
𝑖that
support it:
S𝑡={(𝑑𝑖,𝑧(𝑡)
𝑖,𝑒(𝑡)
𝑖,𝑃(𝑡)
𝑖)}𝑚
𝑖=1.(1)
It says, at a glance, what is answered and what is still open, and it
tells the agent what to search for next: an item that is still open,
rather than repeatedly searching from 𝑞alone. Because 𝑃(𝑡)
𝑖names
the passage the agentbelievedsettled 𝑑𝑖, it can be checked against
𝑃★(𝑞)afterwards, which is what turns the register from a control
structure into an instrument (Section 4.3).
3 Method
ExecutionRAG is deliberately the smallest intervention that satis-
fies Section 2.2: it changes what the agent searchesfor, and nothing
about how retrieval works. The retriever, the corpus, the tool inter-
face, and the generator are untouched, so any evidence it recovers
was evidence the same retriever could already reach.
3.1 What the Register Records
ExecutionRAG maintains one entry per item, holding its status, a
short evidence note, and the passages that support it. Each item’s

Beyond Retrieval Failure: Understanding Why LLMs Miss Evidence in Multi-Hop RAG DAI ’26, November 29–December 2, 2026, Hong Kong, China
Figure 1: Overview of ExecutionRAG. The agent keeps a requirement register of the question’s open sub-questions and works
through it in three stages: (1) build the register from the question and an initial (seed) retrieval, so its items are grounded
in evidence the corpus can actually return; (2) at each step, select an open item, retrieve evidence, and update the register.
The agent marks an item answered when a passage supports it and records that passage, or drops the item if later evidence
contradicts it; and (3) answer only once no item is still open, or return a bounded partial answer when the search budget runs
out.
status takes one of four values:
𝑧(𝑡)
𝑖∈{pending,active,resolved,retired}.(2)
pending means the item has been identified but is not the current
search target; active is the item currently guiding the search;
resolved means enough evidence has been found to answer it;
andretired means it has been dropped, and only with an explicit
reason.
We keep the status set deliberately small. Partial progress lives in
the evidence note 𝑒(𝑡)
𝑖and the supporting passages 𝑃(𝑡)
𝑖, not in extra
labels, so an item stays open until its evidence is actually enough.
The status sayswhetheran item is answered; the note and passages
sayhow. The four values are also what make the agent’s judgment
legible: the step from active toresolved is the moment it declares
an item settled. Recording this decision as an event, rather than
leaving it implicit in free-form reasoning, allows us to compare it
with gold evidence and determine whether it was premature.
3.2 Building the Register from Seed Evidence
The register should not be written from the question alone. Splitting
a question in the abstract tends to produce items that are vague,
redundant, or ask for things the corpus cannot answer. So Execu-
tionRAG first runs oneseedretrieval and builds the register from
what comes back:
C(0)=𝑅 seed(𝑞;K).(3)
The model reads(𝑞,C(0))and writes the initial register S0, so its
items are shaped by the question but grounded in evidence thecorpus can actually return. One item is set active as the first
search target and the rest startpending, with empty notes:
𝑧(0)
𝑏0=active, 𝑧(0)
𝑖=pending(𝑖≠𝑏 0),
𝑒(0)
𝑖=∅, 𝑃(0)
𝑖=∅.(4)
This gives the agent one concrete thing to search for while keeping
the other sub-questions on the list for later.
3.3 Updating the Register and Steering the
Search
Starting fromS0, ExecutionRAG repeats a simple loop: pick an
open item, search for it, and update that item.
Searching for one open item.At step 𝑡, ExecutionRAG picks an
open item𝑏 𝑡fromS𝑡−1and builds a query for it:
𝑢𝑡=Compose(𝑞,𝑑 𝑏𝑡,𝑒(𝑡−1)
𝑏𝑡).(5)
The original question 𝑞keeps the overall goal in view, 𝑑𝑏𝑡says
which item is being chased, and the note 𝑒(𝑡−1)
𝑏𝑡points the search
at what is still missing for that item. The retriever returns new
evidence ΔC𝑡; repeated identical queries are cached so the agent
does not loop.
Updating only the items it touches.Given ΔC𝑡, the register is
updated as
S𝑡=𝑈(S𝑡−1,ΔC𝑡).(6)

DAI ’26, November 29–December 2, 2026, Hong Kong, China Wei et al.
The update rewrites the notes for the items the new evidence bears
on, adds the supporting passages, and moves an item to resolved
only when there is enough support, or to retired only with an
explicit reason. Items the new evidence does not touch are left
unchanged. In effect, each retrieval is turned into progress on one
open item, rather than piling up more unsorted context.
3.4 When to Answer, and Two Bookkeeping
Rules
Answering only when the register is clear.After each update, Exe-
cutionRAG checks whether any item is still open. If none is, it writes
the final answer from the register and its supporting passages; oth-
erwise it keeps searching for the items that remain open. If the
search budget runs out first, it returns a bounded partial answer
that says plainly which items are still unsupported.
Showing only what changed.To keep the context short, Execu-
tionRAG does not replay the whole register after every step. It
shows the agent only aprogress delta: the items whose status or
note changed fromS𝑡−1toS𝑡. This keeps the signal the agent needs
for its next move without re-reading the full list each time.
Two rules that run outside the model.Because the register is
updated by multiple LLM calls, ExecutionRAG enforces two deter-
ministic rules to preserve its state. First, an open item can leave the
list only through an explicit transition toretired:
Req(S𝑡)⊇Req(S 𝑡−1)\{𝑑𝑘|𝑧(𝑡)
𝑘=retired}.(7)
Second, supporting evidence is appended rather than overwritten:
𝑃(𝑡)
𝑖=𝑃(𝑡−1)
𝑖∪Δ𝑃(𝑡)
𝑖.(8)
A later update may revise an item’s note, but it cannot erase the
record of passages that previously supported it. These rules preserve
the requirements and their evidence trail; they do not guarantee
that the language model resolves an item correctly. Keeping this
decision explicit allows us to evaluate it against gold evidence
(Section 4.3).
4 Experiments
4.1 Experimental Questions
Section 2.2 defines the controlled comparison: holding the retriever
and corpus fixed, we measure how much the frequency of missing-
evidence failures changes when only search guidance is modified.
The experiments first quantify this reduction and then examine
whether recovering evidence reduces total error or reveals a differ-
ent bottleneck. The framing also makes predictions that could fail.
If the recovered evidence were an artifact of the register writing
longer answers that favor the judge, it would not improve recall of
gold evidence or the objective substring metric. If the register keeps
search on target without making the generator a better verifier, the
failures left behind should move from “never retrieved the evidence”
toward answering too early or abstaining. And if the effect were
really about reading more text, it should vanish where the register
reads no more than the baseline.
Accordingly, we ask:
•RQ1: Under a fixed retriever, how much does search guidance
reduce failures caused by missing evidence?•RQ2: How does the failure distribution change after more re-
quired evidence is recovered?
•RQ3: Does the reduction persist with a different embedding
model?
•RQ4: Does the recovered evidence improve answer accuracy,
particularly when sub-questions must be combined across doc-
uments?
•RQ5: Is the gain explained by retrieving more context?
•RQ6: Do seed grounding and iterative updates both contribute
to the result?
4.2 Experimental Setup
Datasets.We evaluate on three standard multi-hop QA bench-
marks: HotpotQA [ 47], MuSiQue [ 34], and 2WikiMultiHopQA
(2Wiki) [ 14]. Following HippoRAG, we use the same retrieval
corpora and sample 1,000 questions from each validation set as test
queries.
Baselines.We compare against three groups of baselines: (1)Di-
rect inference: GPT-4o-mini without retrieval; (2)Vanilla RAG: top-
𝑘retrieval with 𝑘∈{ 3,5,10}, CoT-augmented retrieval, and IR-
CoT [ 35]; (3)Structured and agentic RAG: KGP [ 36], G-Retriever [ 13],
RAPTOR [ 32], LightRAG [ 9], HippoRAG and HippoRAG 2 [ 10,11],
GFM-RAG [ 27], LinearRAG [ 54], LAG [ 43], and A-RAG [ 5]. This
comparison spans direct answering, flat retrieval, corpus structur-
ing, and iterative retrieval control.
Metrics.Following LinearRAG [ 54], we reportContain-Match
Accuracy, which checks whether the gold answer appears as a
substring of the prediction, andLLM-Evaluation Accuracy, which
uses an LLM judge to assess semantic equivalence.
Implementation details.Unless otherwise specified, all methods
use GPT-4o-mini as the generation model. For the main comparison,
we use all-MiniLM-L6-v2 with top- 𝑘=5retrieval, and additionally
evaluate Qwen3-Embedding-0.6B as an alternative embedding back-
bone. All main results use the full 1,000-question evaluation sets.
The ablation study is conducted on the full 1,000-question MuSiQue
evaluation set. Additional implementation details are provided in
Appendix B.
4.3 RQ1: Reduction in Failures Caused by
Missing Evidence
We measure the controlled reduction described in Section 2.2 by
assigning each wrong answer on MuSiQue a single cause using the
gold supporting paragraphs. The full counts and percentages for
both retrievers appear in Appendix E.
How we label a failure.Each wrong answer receives exactly one
cause, assigned by a fixed priority order so the categories cannot
overlap:judge error(the substring metric counts the answer as right
but the judge does not),gave up(the answer explicitly abstains),
reasoning error(every supporting passage was retrieved and the
answer is still wrong, or, for ExecutionRAG, an item was resolved
while supporting evidence was still missing), andmissing evidence
(a supporting passage was never retrieved and no premature reso-
lution occurred). What makes this labelling possible at all is that

Beyond Retrieval Failure: Understanding Why LLMs Miss Evidence in Multi-Hop RAG DAI ’26, November 29–December 2, 2026, Hong Kong, China
MuSiQue ships, for each question, the supporting paragraphs a
correct answer minimally needs: we match each supporting para-
graph to its corpus chunk by a 40-character fingerprint (87.9% of
paragraphs match) and ask whether the search ever retrieved it.
Because the two methods fail on different numbers of questions,
every share below is a percentage of that method’s own failures.
The largest change is a reduction in failures caused by missing
evidence: wrong answers for which a required passage was never
retrieved. With all-MiniLM-L6-v2, the count decreases from 382
for A-RAG to 234 for ExecutionRAG; these values represent 69.3%
of 551 baseline failures and 43.9% of 533 ExecutionRAG failures.
With Qwen3-Embedding-0.6B, the count decreases from 383 to 274,
or from 66.4% of 577 failures to 47.9% of 572 failures. The same
retriever is used within each comparison, so the reductions show
that search guidance can recover a substantial portion of evidence
that the baseline search did not retrieve. They do not imply that
every residual missing passage is unreachable (Figure 2).
Recovering this evidence does not reduce total error proportion-
ally. Reasoning errors after retrieval increase from 9% to 26% with
MiniLM and from 8% to 21% with Qwen; these errors include cases
in which the register is resolved before its evidence is sufficient.
Abstentions also increase, from 7% to 13% and from 12% to 17%,
respectively. Against A-RAG, LLM-Evaluation Accuracy on MuSi-
Que rises from 44.9 to 50.2, a smaller change than the reduction in
missing-evidence failures. This redistribution is the central result:
once search guidance recovers more required evidence, evidence
verification and answer commitment become the dominant sources
of the remaining error.
4.4 RQ2: Evidence Recovery Shifts the
Remaining Failures
Why reasoning errors become more visible.Part of this increase
reflects observability rather than a change in the model’s reasoning
alone. A premature resolution is an explicit register event and can
therefore be assigned to reasoning error. A-RAG has no analogous
event; when it answers from incomplete evidence, the error is gen-
erally assigned to the missing passage instead. The comparison may
thus undercount premature answers by the baseline. The register
exposes this decision for evaluation; it does not establish that the
generation model reasons worse.
The register guides search but does not provide a reliable evi-
dence verifier: the language model still decides whether a require-
ment has been resolved, and this decision can be incorrect. It also
cannot make genuinely difficult passages retrievable. Residual fail-
ures caused by missing evidence for ExecutionRAG remain highest
on four-hop questions, which contain longer and more weakly con-
nected evidence chains. In this setting, the remaining bottlenecks
are verifying retrieved evidence and retrieving evidence for long
chains.
4.5 RQ3: The Reduction Persists Across
Embedding Models
Figure 3 tests whether the performance pattern depends on the
embedding model. After replacing all-MiniLM-L6-v2 with Qwen3-
Embedding-0.6B, ExecutionRAG outperforms A-RAG on all threebenchmarks: 80.0, 68.3, and 42.8 LLM-Evaluation Accuracy on Hot-
potQA, 2Wiki, and MuSiQue, respectively, compared with 76.4, 55.2,
and 42.3 for A-RAG. The largest gain remains on 2Wiki ( +13.1),
while MuSiQue shows a smaller positive gain ( +0.5). The positive
difference across all three benchmarks shows that the effect is not
specific to one embedding model. The larger gain on 2Wiki is also
consistent with the proposed mechanism because its questions re-
quire dependent sub-questions to be maintained across multiple
retrieval steps.
4.6 RQ4: Answer Accuracy
Table 1 reports the main comparison. We first consider Contain-
Match Accuracy because it does not rely on an LLM judge. On
MuSiQue, ExecutionRAG reaches 49.1, a 6.3-point improvement
over LAG (42.8). On 2Wiki, it reaches 73.0, a 1.1-point improvement
over LAG (71.9). On HotpotQA, it reaches 77.0, which is essentially
tied with A-RAG (77.3). LLM-Evaluation Accuracy follows the same
pattern and is highest for ExecutionRAG on all three benchmarks:
79.8, 74.3, and 50.2 on HotpotQA, 2Wiki, and MuSiQue, respectively.
The differences are consistent with the proposed mechanism.
The register has little effect on HotpotQA, whose questions are
mostly two-hop and are often answerable from a single comparison.
It improves performance on 2Wiki and MuSiQue, where the agent
must maintain several dependent sub-questions across documents.
A-RAG is the matched baseline for the failure analysis. It uses
the same generation model, retrieval tools, and corpora as Execu-
tionRAG; the requirement register is the added component. The
difference between A-RAG and ExecutionRAG in Sections 4.3–4.5
therefore cannot be attributed to a different retriever or generation
backbone. The register adds bookkeeping calls, but these calls alone
do not explain the gain: on 2Wiki, the largest improvement occurs
with comparable or fewer retrieved tokens (Section 4.7).
Where the gains concentrate.Figure 5 in Appendix C reports
category-level results. Under both embedding models, questions
with longer or less direct evidence chains remain more difficult. The
register helps most when required evidence is retrievable but must
be tracked across several steps. On MuSiQue, accuracy decreases
with hop count. On HotpotQA, Comparison questions are easier
than Bridge questions; on 2Wiki, Inference questions are hardest
and Comparison questions are easiest. The register improves overall
accuracy but does not remove the difficulty of long evidence chains.
4.7 RQ5: Retrieved Context and Accuracy
Figure 4 and Table 2 test whether the gains can be explained by
retrieving more context. On MuSiQue and HotpotQA, Execution-
RAG reads more tokens than A-RAG because it builds and updates
the register. On 2Wiki, however, it uses comparable or fewer re-
trieved tokens while obtaining its largest accuracy improvement.
Within ExecutionRAG, removing register components reduces both
token use and accuracy. The result is consistent with the register
improving the targeting of retrieval rather than simply increasing
the amount of retrieved context.
What the register costs.The register increases the cost of some
searches. On MuSiQue, it reads 7,161 retrieved tokens per question,

DAI ’26, November 29–December 2, 2026, Hong Kong, China Wei et al.
0% 20% 40% 60% 80% 100%
Percentage of FailuresOverall
2-hop
3-hop
4-hop43.9
37.2
44.0
55.325.9
30.1
24.0
21.213.1
14.2
14.3
9.817.1
18.6
17.7
13.6all-MiniLM-L6-v2 (N=533)
0% 20% 40% 60% 80% 100%
Percentage of Failures47.9
40.2
52.9
55.821.2
25.2
17.5
18.617.3
17.7
15.3
19.413.6
16.9
14.3
6.2Qwen3-Embedding-0.6B (N=572)
Retrieval Error Reasoning Error Gave Up Judge Error
Figure 2: Failure distribution on MuSiQue, by failure type and hop count. The share of failures that never retrieve a required
passage decreases after adding the register, while reasoning errors after retrieval and abstentions increase. For ExecutionRAG,
failures caused by missing evidence remain most common at four hops.
Contain LLM020406080Accuracy (%)HotpotQA
75.781.0
Contain LLM2WikiMHQA
68.1 68.3
Contain LLMMuSiQue
42.5 42.8Direct
Naive
Linear
A-RAG
OursAvg. Contain-Acc
57.3
62.1
0 20 40 60Direct
Naive
Linear
A-RAG
OursAvg. LLM-Acc
58.0
64.0Direct Answer Naive RAG LinearRAG A-RAG Ours
Figure 3: Performance under Qwen3-Embedding-0.6B. Left: per-benchmark Contain-Match and LLM-Evaluation Accuracy.
Right: average accuracy across benchmarks.
compared with 5,015 for A-RAG; on HotpotQA, it reads 3,373 com-
pared with 2,860. On 2Wiki, token use is comparable at 3,966 and
3,992, respectively. The corresponding API costs are $0.0064 versus
$0.0034 on MuSiQue and $0.0039 versus $0.0027 on 2Wiki. Search is
limited to fifteen retrieval loops and a cumulative context of 128,000
tokens. The agent answers when no requirement remains open, so
the register shortens some trajectories and lengthens others; the
case studies in Appendix F complete in two and six loops, whereas
A-RAG uses five and thirteen. Appendix D provides the complete
breakdown.
4.8 RQ6: Effects of Seed Grounding and
Iterative Updates
Table 2 isolates the contributions of the register’s components.
Building the register from the question alone rather than from seedevidence lowers LLM-Acc from 50.2 to 48.7, indicating that ground-
ing requirements in retrievable evidence is beneficial. Removing
iterative updates and constructing the register only once lowers
LLM-Acc further to 48.3. Both seed grounding and iterative updates
contribute to the observed gain.
5 Related Work
Corpus structure for RAG.A major line of work improves multi-
hop QA by structuring corpus knowledge to improve evidence ac-
cess. GraphRAG [ 6] builds entity-relation graphs with community
summaries; RAPTOR [ 31] organizes documents into hierarchical
retrieval trees; LightRAG [ 8] combines graph-augmented index-
ing with efficient retrieval; HippoRAG and HippoRAG 2 [ 12,17]
use graph-based retrieval with Personalized PageRank; and Lin-
earRAG [ 54], GFM-RAG [ 27], and StructRAG [ 26] further explore

Beyond Retrieval Failure: Understanding Why LLMs Miss Evidence in Multi-Hop RAG DAI ’26, November 29–December 2, 2026, Hong Kong, China
Table 1: Main results on three multi-hop QA benchmarks. Bold marks the bestContain-Matchresult per benchmark; the grey
columns reportLLM-Evaluation Accuracyfor comparability with prior work. These answer-level metrics complement the
primary failure analysis by showing whether recovered evidence improves the final prediction. The lower block is the matched
comparison our attribution rests on: A-RAG shares ExecutionRAG’s generator, tools, and corpora, so it is ExecutionRAG
withoutthe register. Gains concentrate where sub-questions must be combined across documents; HotpotQA is a tie.
MethodContain-MatchLLM-Evaluation
Hotpot 2Wiki MuSiQue Hotpot 2Wiki MuSiQue
Direct Zero-shot LLM Inference
GPT-4o-mini [28] 30.4 29.0 7.8 34.2 28.6 10.1
Vanilla Retrieval-Augmented Generation
Retrieval (Top-3) 52.1 45.1 23.4 55.1 43.1 27.1
Retrieval (Top-5) 54.6 46.6 25.6 56.8 45.3 29.0
Retrieval (Top-10) 56.0 48.7 26.7 58.6 45.8 31.2
CoT (Top-5) [38] 55.1 48.7 27.1 57.1 45.9 30.7
IRCoT (Top-5) [35] 58.4 53.0 22.6 59.6 36.8 26.1
Graph and Agentic Retrieval-Augmented Generation
KGP [36] 56.2 52.2 30.5 57.1 33.9 27.3
G-retriever [13] 41.3 47.8 14.1 40.9 25.7 15.6
RAPTOR [32] 58.1 60.6 32.2 55.3 43.9 29.7
LightRAG [9] 61.5 54.4 27.7 60.5 38.0 28.3
HippoRAG (single-step) [10] 55.2 63.7 31.4 57.9 57.5 30.1
HippoRAG (multi-step) [10] 61.1 66.4 34.0 63.6 62.4 31.8
GFM-RAG (single-step) [27] 61.4 66.2 29.3 64.8 61.1 32.6
GFM-RAG (multi-step) [27] 63.4 69.5 31.5 65.5 63.2 35.5
HippoRAG 2 [11] 61.2 62.0 34.5 64.3 58.8 35.6
LinearRAG [54] 64.2 69.5 32.8 64.9 62.6 36.9
LAG [43] 68.5 71.9 42.8 69.7 64.0 44.3
Matched condition: same generator, retrieval tools, and corpora
A-RAG [5](= ours w/o register)77.366.7 42.0 78.6 59.8 44.9
ExecutionRAG (Ours) 77.0 73.0 49.1 79.8 74.3 50.2
Δvs. matched−0.3+6.3+7.1+1.2+14.5+5.3
Table 2: Ablation study on MuSiQue. LLM-Evaluation and
Contain-Match Accuracy, under the default setup.
Method LLM-Acc Contain-Acc
ExecutionRAG 50.2 49.1
w/o Register Init. 48.7 46.2
w/o Progress Register 48.3 48.5
graph construction, graph foundation models, and hybrid struc-
turization. These methods improve how dispersed evidence is rep-
resented and retrieved before answering. ExecutionRAG is com-
plementary: it does not restructure the corpus, but maintains an
explicit register of unresolved sub-questions during search.
Iterative and agentic RAG.A related line of work improves re-
trieval control during inference. IRCoT [ 35] and FLARE [ 15] in-
terleave reasoning with retrieval; Self-RAG [ 1] uses reflection to-
kens to critique retrieved passages; and recent agentic methods
expose richer retrieval policies through tool use, query reformu-
lation, and multi-step search. A-RAG [ 5] replaces fixed pipelines
with model-driven hierarchical retrieval interfaces, while LAG [ 43]and LogicRAG [ 4] decompose questions into subproblems, infer
dependencies, and sequence retrieval accordingly. This view is
also related to work that formulates language-model inference as
search [ 40], analyzes how chain-of-thought prompting interacts
with the answer space [ 51], learns when a model should disclose its
reasoning [ 37], learns execution-time workflows for multi-agent
systems [ 44], or scales multi-agent processing over many docu-
ments at test time [ 3]. These methods mainly optimize retrieval
control: what to retrieve next and in what order. ExecutionRAG
does not introduce a new retrieval policy. Instead, it maintains an
explicit register of unresolved sub-questions that guides subsequent
search and determines when the agent can answer. Even with adap-
tive retrieval, an agent can discard an unresolved sub-question,
overwrite earlier progress, or answer from partial evidence. The
register preserves these requirements throughout the search. It is
also narrower than agent-memory work that retrieves over a per-
sistent episodic record of an agent’s past decisions [ 45]: the register
tracks the open requirements of thecurrentquestion only, and is
discarded once the answer is given.
Long executions outside question answering.The same pressure to
keep intermediate state explicit appears in domains where an agent

DAI ’26, November 29–December 2, 2026, Hong Kong, China Wei et al.
5000 6000 7000
Retrieved Tokens/Q404550LLM Acc (%)MuSiQue
3000 3500
Retrieved Tokens/Q747678808284HotpotQA
3750 4000 4250 4500
Retrieved Tokens/Q506070802WikiMHQAA-RAG (MiniLM) Ours (MiniLM) A-RAG (Qwen) Ours (Qwen)
Figure 4: Accuracy–efficiency trade-off. Pareto frontiers of LLM-Evaluation Accuracy vs. retrieved tokens per question for
ExecutionRAG and A-RAG under two embedding backbones. Bottom row: ablation variants; bars show LLM-Acc and the line
shows average retrieved tokens per question.
must execute a long chain before it can commit to an answer. Multi-
agent drafting protocols compress intermediate reasoning [ 22],
and retrieval-augmented multi-agent pipelines are built for tasks
whose verdict depends on evidence gathered in several steps [ 23].
Software-engineering agents are trained against execution feedback
and process-level reward precisely because their trajectories are
too long to supervise end to end [ 24,25,33]. There too the difficulty
is less the first attempt than checking what the agent has actually
done. The register is the minimal instance of that bookkeeping for
retrieval: it records the requirements a trajectory still owes, and
nothing else.
Sufficiency signals, reading notes, and failure analysis.Several
methods track information gaps or judge evidence sufficiency dur-
ing RAG. Self-RAG [ 1] critiques passages with reflection tokens [ 50].
Related work adds corrective retrieval (CRAG [ 46]), sequential read-
ing notes (Chain-of-Note [ 49]), and structured missing-information
items with sufficiency checks (S2G-RAG [ 21], SURE-RAG [ 29]).
Test-time critique and optimization provide another mechanism for
improving retrieved evidence and generated answers [ 41]. Other
methods diagnose or repair agent trajectories after the fact, such
as Doctor-RAG [ 16]. The requirement register is an explicit list of
sub-questions with states and evidence provenance. Our contribu-
tion is empirical: by holding the retriever and generator fixed, we
measure how such a register changes the multi-hop failure distri-
bution. It reduces failures caused by missing evidence under a fixed
retriever and shifts residual errors toward reasoning after retrieval,
a distinction that prior end-to-end evaluations do not isolate.
Naming the two causes behind one symptom.Skill-RAG [ 42]
makes the closest claim to Section 2.1 we know of: retrieval fail-
ures “stem not from the absence of relevant evidence but from an
alignment gap between the query and the evidence space,” withan exit route for cases it calls truly irreducible. This distinguishes
unreachable and unasked causes conceptually and routes learned
repair skills accordingly. It does not define the failure event using
gold evidence or quantify either share. We instead treat the distinc-
tion as the measurement target. Outside retrieval, Binding Drift [ 2]
separates drift from error propagation using disjoint evaluation
sets. We apply the same separation to a missing passage by asking
whether the search could have retrieved it.
Implications.The results distinguish two directions for future
work. Because the recovered evidence was reachable by the same
retriever, increasing index coverage, improving embeddings, or ex-
panding queries does not address the observed failures associated
with inadequate search guidance. Those failures require a mech-
anism that preserves unresolved sub-questions. Conversely, the
increase in premature resolution identifies evidence verification
as a separate target. The register makes the resolution decision
explicit, which enables a verifier to be trained or evaluated against
it. Residual failures caused by missing evidence concentrate on
long, weakly connected chains, where improving retrieval coverage
remains necessary. These predictions are testable: a verifier should
improve answer accuracy without changing retrieved evidence,
whereas an intervention that improves only coverage should not
eliminate failures caused by unasked queries.
6 Conclusion
We study how much apparent retrieval failure in multi-hop RAG can
be reduced by search guidance while the retriever remains fixed.
On MuSiQue, the requirement register reduces wrong answers
with unretrieved required evidence from 382 to 234 cases under
one retriever and from 383 to 274 under another. This recovery
does not translate proportionally into answer accuracy: premature
resolutions and abstentions increase as missing-evidence failures

Beyond Retrieval Failure: Understanding Why LLMs Miss Evidence in Multi-Hop RAG DAI ’26, November 29–December 2, 2026, Hong Kong, China
decrease. The result separates two bottlenecks that aggregate ac-
curacy obscures. Maintaining unresolved requirements helps the
agent retrieve reachable evidence, but converting that evidence into
correct answers requires more reliable verification after retrieval.
Acknowledgments
Supported by Shanghai Artificial Intelligence Laboratory.
References
[1]Akari Asai, Zeqiu Wu, Yizhong Wang, Avirup Sil, and Hannaneh Hajishirzi. 2023.
Self-RAG: Learning to Retrieve, Generate, and Critique through Self-Reflection.
arXiv:2310.11511 [cs.CL] https://arxiv.org/abs/2310.11511
[2]Rahul Suresh Babu and Shashank Indukuri. 2026. Binding Drift in Multi-Step Tool-
Augmented Agents. arXiv:2607.18316 [cs.CL] https://arxiv.org/abs/2607.18316
[3]Juntai Cao, Xiang Zhang, Raymond Li, Jiaqi Wei, Chuyuan Li, Shafiq Joty, and
Giuseppe Carenini. 2025. Multi2: Multi-agent test-time scalable framework for
multi-document processing. InProceedings of The 5th New Frontiers in Summa-
rization Workshop. 135–156.
[4]Shengyuan Chen, Chuang Zhou, Zheng Yuan, Qinggang Zhang, Zeyang Cui,
Hao Chen, Yilin Xiao, Jiannong Cao, and Xiao Huang. 2026. You don’t need
pre-built graphs for rag: Retrieval augmented generation with adaptive reasoning
structures. InProceedings of the AAAI Conference on Artificial Intelligence, Vol. 40.
30270–30278.
[5]Mingxuan Du, Benfeng Xu, Chiwei Zhu, Shaohan Wang, Pengyu Wang, Xiaorui
Wang, and Zhendong Mao. 2026. A-RAG: Scaling Agentic Retrieval-Augmented
Generation via Hierarchical Retrieval Interfaces.arXiv preprint arXiv:2602.03442
(2026).
[6]Darren Edge, Ha Trinh, Newman Cheng, Joshua Bradley, Alex Chao, Apurva
Mody, Steven Truitt, and Jonathan Larson. 2024. From local to global: A graph
rag approach to query-focused summarization.arXiv preprint arXiv:2404.16130
(2024).
[7]Yunfan Gao, Yun Xiong, Xinyu Gao, Kangxiang Jia, Jinliu Pan, Yuxi Bi, Yi Dai,
Jiawei Sun, and Haofen Wang. 2023. Retrieval-augmented generation for large
language models: A survey.arXiv preprint arXiv:2312.10997(2023).
[8]Zirui Guo, Lianghao Xia, Yanhua Yu, Tu Ao, and Chao Huang. 2024. LightRAG:
Simple and Fast Retrieval-Augmented Generation.arXiv preprint arXiv:2410.05779
(2024).
[9]Zirui Guo, Lianghao Xia, Yanhua Yu, Tu Ao, and Chao Huang. 2025. LightRAG:
Simple and Fast Retrieval-Augmented Generation. arXiv:2410.05779 [cs.IR] https:
//arxiv.org/abs/2410.05779
[10] Bernal Jiménez Gutiérrez, Yiheng Shu, Yu Gu, Michihiro Yasunaga, and Yu Su.
2024. Hipporag: Neurobiologically inspired long-term memory for large language
models. InThe Thirty-eighth Annual Conference on Neural Information Processing
Systems.
[11] Bernal Jiménez Gutiérrez, Yiheng Shu, Weijian Qi, Sizhe Zhou, and Yu Su. 2025.
From RAG to Memory: Non-Parametric Continual Learning for Large Language
Models. InForty-second International Conference on Machine Learning. https:
//openreview.net/forum?id=LWH8yn4HS2
[12] Bernal Jiménez Gutiérrez, Yiheng Shu, Weijian Qi, Sizhe Zhou, and Yu Su. 2025.
From RAG to Memory: Non-Parametric Continual Learning for Large Language
Models. InForty-second International Conference on Machine Learning. https:
//openreview.net/forum?id=LWH8yn4HS2
[13] Xiaoxin He, Yijun Tian, Yifei Sun, Nitesh Chawla, Thomas Laurent, Yann LeCun,
Xavier Bresson, and Bryan Hooi. 2024. G-retriever: Retrieval-augmented gen-
eration for textual graph understanding and question answering.Advances in
Neural Information Processing Systems37 (2024), 132876–132907.
[14] Xanh Ho, Anh-Khoa Duong Nguyen, Saku Sugawara, and Akiko Aizawa. 2020.
Constructing A Multi-hop QA Dataset for Comprehensive Evaluation of Reason-
ing Steps. InProceedings of the 28th International Conference on Computational
Linguistics, Donia Scott, Nuria Bel, and Chengqing Zong (Eds.). International
Committee on Computational Linguistics, Barcelona, Spain (Online), 6609–6625.
doi:10.18653/v1/2020.coling-main.580
[15] Zhengbao Jiang, Frank F Xu, Luyu Gao, Zhiqing Sun, Qian Liu, Jane Dwivedi-Yu,
Yiming Yang, Jamie Callan, and Graham Neubig. 2023. Active retrieval augmented
generation. InProceedings of the 2023 Conference on Empirical Methods in Natural
Language Processing.
[16] Shuguang Jiao, Chengkai Huang, Shuhan Qi, Xuan Wang, Yifan Li, Quanchi
Weng, Lingchuan Liu, Xunliang Cai, and Lina Yao. 2026. Doctor-RAG: A
Failure-Aware Repair Framework for Agentic Retrieval-Augmented Generation.
arXiv:2604.00865 [cs.IR] https://arxiv.org/abs/2604.00865
[17] Bernal Jimenez Gutierrez, Yiheng Shu, Yu Gu, Michihiro Yasunaga, and Yu Su.
2024. Hipporag: Neurobiologically inspired long-term memory for large language
models.Advances in Neural Information Processing Systems37 (2024), 59532–
59569.[18] Hailey Joren, Jianyi Zhang, Chun-Sung Ferng, Da-Cheng Juan, Ankur Taly, and
Cyrus Rashtchian. 2024. Sufficient context: A new lens on retrieval augmented
generation systems.arXiv preprint arXiv:2411.06037(2024).
[19] Evgenii Kortukov, Alexander Rubinstein, Elisa Nguyen, and Seong Joon Oh. 2024.
Studying large language model behaviors under context-memory conflicts with
real documents.arXiv preprint arXiv:2404.16032(2024).
[20] Patrick Lewis, Ethan Perez, Aleksandra Piktus, Fabio Petroni, Vladimir Karpukhin,
Naman Goyal, Heinrich Küttler, Mike Lewis, Wen-tau Yih, Tim Rocktäschel,
Sebastian Riedel, and Douwe Kiela. 2020. Retrieval-augmented generation for
knowledge-intensive NLP tasks. InProceedings of the 34th International Conference
on Neural Information Processing Systems(Vancouver, BC, Canada)(NIPS ’20).
Curran Associates Inc., Red Hook, NY, USA, Article 793, 16 pages.
[21] Minghan Li, Junjie Zou, Xinxuan Lv, Chao Zhang, and Guodong Zhou. 2026. S2G-
RAG: Structured Sufficiency and Gap Judging for Iterative Retrieval-Augmented
QA. InProceedings of the 64th Annual Meeting of the Association for Computational
Linguistics (Volume 1: Long Papers), Maria Liakata, Viviane P. Moreira, Jiajun
Zhang, and David Jurgens (Eds.). Association for Computational Linguistics,
San Diego, California, United States, 25846–25862. doi:10.18653/v1/2026.acl-
long.1185
[22] Yuanhao Li, Mingshan Liu, Hongbo Wang, Yiding Zhang, Yifei Ma, and Wei
Tan. 2026. DRAFT-RL: Multi-Agent Chain-of-Draft Reasoning for Reinforcement
Learning-Enhanced LLMs.Proceedings of the AAAI Conference on Artificial
Intelligence40, 35 (2026), 29530–29537. doi:10.1609/aaai.v40i35.40195
[23] Yuanhao Li, Hongbo Wang, Xuhong Chen, and Yiming Cao. 2026. Retrieval-
Augmented Multi-Agent Multimodal Framework for Fake News Detection.
InICASSP 2026. https://cmsworkshops.com/ICASSP2026/view_paper.php?
PaperNum=9709
[24] Yuanhao Li, Hongbo Wang, Xuhong Chen, Yiming Cao, and Xunzhu Tang. 2026.
Counterfactual Rollout Replay: Forkable Environments as Free Process Rewards
for Software Engineering Agents. arXiv:2609.33875 [cs.SE] https://arxiv.org/abs/
2609.33875 Accepted at NeurIPS 2026.
[25] Yuanhao Li, Hongbo Wang, Xiaotang Shang, Xunzhu Tang, Yiming Cao,
and Xuhong Chen. 2026. BoostAPR: Boosting Automated Program Repair
via Execution-Grounded Reinforcement Learning with Dual Reward Models.
arXiv:2605.09134 [cs.AI] https://arxiv.org/abs/2605.09134 Accepted at ICML
2026.
[26] Zhuoqun Li, Xuanang Chen, Haiyang Yu, Hongyu Lin, Yaojie Lu, Qiaoyu Tang,
Fei Huang, Xianpei Han, Le Sun, and Yongbin Li. 2025. StructRAG: Boosting
Knowledge Intensive Reasoning of LLMs via Inference-time Hybrid Information
Structurization. InThe Thirteenth International Conference on Learning Represen-
tations. https://openreview.net/forum?id=GhexuBLxbO
[27] Linhao Luo, Zicheng Zhao, Gholamreza Haffari, Dinh Phung, Chen Gong, and
Shirui Pan. 2025. GFM-RAG: Graph Foundation Model for Retrieval Augmented
Generation.arXiv preprint arXiv:2502.01113(2025).
[28] OpenAI. 2024. GPT-4o System Card. arXiv:2410.21276 [cs.CL] https://arxiv.org/
abs/2410.21276
[29] Jingxi Qiu, Zeyu Han, and Cheng Huang. 2026. SURE-RAG: Sufficiency and
Uncertainty-Aware Evidence Verification for Selective Retrieval-Augmented Gen-
eration. arXiv:2605.03534 [cs.CL] https://arxiv.org/abs/2605.03534
[30] Dongyu Ru, Lin Qiu, Xiangkun Hu, Tianhang Zhang, Peng Shi, Shuaichen Chang,
Cheng Jiayang, Cunxiang Wang, Shichao Sun, Huanyu Li, et al .2024. Ragchecker:
A fine-grained framework for diagnosing retrieval-augmented generation.Ad-
vances in Neural Information Processing Systems37 (2024), 21999–22027.
[31] Parth Sarthi, Salman Abdullah, Aditi Tuli, Shubh Khanna, Anna Goldie, and
Christopher D. Manning. 2024. RAPTOR: Recursive Abstractive Processing for
Tree-Organized Retrieval. InInternational Conference on Learning Representations
(ICLR).
[32] Parth Sarthi, Salman Abdullah, Aditi Tuli, Shubh Khanna, Anna Goldie, and
Christopher D Manning. 2024. Raptor: Recursive abstractive processing for
tree-organized retrieval. InThe Twelfth International Conference on Learning
Representations.
[33] Wei Tan, Yuanhao Li, and Wenkai Liang. 2026. Evo-CuRL: Curriculum-Aware
Reinforcement Learning over Code Lineage Graphs for Software Engineering
Reasoning. InICMR 2026. 1327–1335. doi:10.1145/3805622.3810701
[34] Harsh Trivedi, Niranjan Balasubramanian, Tushar Khot, and Ashish Sabharwal.
2022. MuSiQue: Multihop Questions via Single-hop Question Composition.Trans.
Assoc. Comput. Linguistics10 (2022), 539–554. doi:10.1162/TACL_A_00475
[35] Harsh Trivedi, Niranjan Balasubramanian, Tushar Khot, and Ashish Sabharwal.
2023. Interleaving Retrieval with Chain-of-Thought Reasoning for Knowledge-
Intensive Multi-Step Questions. InThe 61st Annual Meeting Of The Association
For Computational Linguistics.
[36] Yu Wang, Nedim Lipka, Ryan A Rossi, Alexa Siu, Ruiyi Zhang, and Tyler Derr.
2024. Knowledge graph prompting for multi-document question answering. In
Proceedings of the AAAI Conference on Artificial Intelligence, Vol. 38. 19206–19214.
[37] Jiaqi Wei, Xuehang Guo, Pengfei Yu, Xiang Zhang, Wanli Ouyang, Siqi Sun,
Qingyun Wang, and Chenyu You. 2026. When to Think, When to Speak: Learning
Disclosure Policies for LLM Reasoning.arXiv preprint arXiv:2605.03314(2026).

DAI ’26, November 29–December 2, 2026, Hong Kong, China Wei et al.
[38] Jason Wei, Xuezhi Wang, Dale Schuurmans, Maarten Bosma, brian ichter,
Fei Xia, Ed Chi, Quoc V Le, and Denny Zhou. 2022. Chain-of-Thought
Prompting Elicits Reasoning in Large Language Models. InAdvances in
Neural Information Processing Systems, S. Koyejo, S. Mohamed, A. Agar-
wal, D. Belgrave, K. Cho, and A. Oh (Eds.), Vol. 35. Curran Associates,
Inc., 24824–24837. https://proceedings.neurips.cc/paper_files/paper/2022/file/
9d5609613524ecf4f15af0f7b31abca4-Paper-Conference.pdf
[39] Jiaqi Wei, Yuejin Yang, Xiang Zhang, Yuhan Chen, Xiang Zhuang, Zhangyang Gao,
Dongzhan Zhou, Guangshuai Wang, Zhiqiang Gao, Juntai Cao, et al .2025. From
ai for science to agentic science: A survey on autonomous scientific discovery.
arXiv preprint arXiv:2508.14111(2025).
[40] Jiaqi Wei, Xiang Zhang, Yuejin Yang, Wenxuan Huang, Juntai Cao, Sheng Xu, Xi-
ang Zhuang, Zhangyang Gao, Muhammad Abdul-Mageed, Laks VS Lakshmanan,
et al.2026. When LLM Meets Tree Search: A Systematic View of Inference as
Search in Large Language Models.arXiv preprint arXiv:2608.30395(2026).
[41] Jiaqi Wei, Hao Zhou, Xiang Zhang, Di Zhang, Zijie Qiu, Noah Wei, Jinzhe Li, Wanli
Ouyang, and Siqi Sun. 2025. Retrieval is Not Enough: Enhancing RAG through
Test-Time Critique and Optimization. InThe Thirty-ninth Annual Conference on
Neural Information Processing Systems.
[42] Kai Wei, Raymond Li, Xi Zhu, Zhaoqian Xue, Jiaojiao Han, Jingcheng Niu, and
Fan Yang. 2026. Skill-RAG: Failure-State-Aware Retrieval Augmentation via
Hidden-State Probing and Skill Routing. arXiv:2604.15771 [cs.CL] https://arxiv.
org/abs/2604.15771
[43] Yilin Xiao, Chuang Zhou, Yujing Zhang, Qinggang Zhang, Su Dong, Shengyuan
Chen, Chang Yang, and Xiao Huang. 2025. Lag: Logic-augmented generation
from a cartesian perspective.arXiv preprint arXiv:2508.05509(2025).
[44] Chengdong Xu, Kaiqiang Ke, Ziheng Liu, Jiaqi Wei, Zibo Shao, Weile Guo, and
Chao Yu. 2026. EvoMAS: Learning Execution-Time Workflows for Multi-Agent
Systems.arXiv preprint arXiv:2605.08769(2026).
[45] Nuoyue Xu, Jiang Liu, Wenxuan Huang, Xiang Zhang, Juntai Cao, and Jiaqi Wei.
2026. Agent Memory with Episodic Retrieval for Financial Decision-Making.
arXiv preprint arXiv:2609.28771(2026).
[46] Shi-Qi Yan, Jia-Chen Gu, Yun Zhu, and Zhen-Hua Ling. 2024. Corrective Retrieval
Augmented Generation. https://openreview.net/forum?id=JnWJbrnaUE
[47] Zhilin Yang, Peng Qi, Saizheng Zhang, Yoshua Bengio, William Cohen, Ruslan
Salakhutdinov, and Christopher D. Manning. 2018. HotpotQA: A Dataset for
Diverse, Explainable Multi-hop Question Answering. InProceedings of the 2018
Conference on Empirical Methods in Natural Language Processing, Ellen Riloff,
David Chiang, Julia Hockenmaier, and Jun’ichi Tsujii (Eds.). Association for
Computational Linguistics, Brussels, Belgium, 2369–2380. doi:10.18653/v1/D18-
1259
[48] Zhihan Yang, Jiaqi Wei, Xiang Zhang, Haoyu Dong, Yiwen Wang, Xiaoke Guo,
Pengkun Zhang, Yiwei Xu, and Chenyu You. 2026. FORESTLLM: Large Language
Models Make Random Forest Great on Few-shot Tabular Learning.arXiv preprint
arXiv:2601.11311(2026).
[49] Wenhao Yu, Hongming Zhang, Xiaoman Pan, Peixin Cao, Kaixin Ma, Jian Li,
Hongwei Wang, and Dong Yu. 2024. Chain-of-Note: Enhancing Robustness in
Retrieval-Augmented Language Models. InProceedings of the 2024 Conference
on Empirical Methods in Natural Language Processing, Yaser Al-Onaizan, Mohit
Bansal, and Yun-Nung Chen (Eds.). Association for Computational Linguistics,
Miami, Florida, USA, 14672–14685. doi:10.18653/v1/2024.emnlp-main.813
[50] Xiang Zhang, Juntai Cao, Jiaqi Wei, Yiwei Xu, and Chenyu You. 2025. Tokenization
constraints in llms: A study of symbolic and arithmetic reasoning limits.arXiv
preprint arXiv:2505.14178(2025).
[51] Xiang Zhang, Juntai Cao, Jiaqi Wei, Chenyu You, and Dujian Ding. 2025. Why
does your cot prompt (not) work? theoretical analysis of prompt space complexity,
its interaction with answer space during cot reasoning with llms: A recurrent
perspective.arXiv preprint arXiv:2503.10084(2025).
[52] Zhilin Zhang, Xiang Zhang, Jiaqi Wei, Yiwei Xu, and Chenyu You. 2025. Postergen:
Aesthetic-aware paper-to-poster generation via multi-agent llms.arXiv e-prints
(2025), arXiv–2508.
[53] Yuhao Zhou, Yiheng Wang, Xuming He, Ao Shen, Ruoyao Xiao, Zhiwei Li, Qiantai
Feng, Zijie Guo, Yuejin Yang, Hao Wu, et al .2025. Scientists’ First Exam: Probing
Cognitive Abilities of MLLM via Perception, Understanding, and Reasoning.
arXiv preprint arXiv:2506.10521(2025).
[54] Luyao Zhuang, Shengyuan Chen, Yilin Xiao, Huachi Zhou, Yujing Zhang, Hao
Chen, Qinggang Zhang, and Xiao Huang. 2025. LinearRAG: Linear Graph Re-
trieval Augmented Generation on Large-scale Corpora. arXiv:2510.10114 [cs.CL]
https://arxiv.org/abs/2510.10114

Beyond Retrieval Failure: Understanding Why LLMs Miss Evidence in Multi-Hop RAG DAI ’26, November 29–December 2, 2026, Hong Kong, China
Appendix Overview
This appendix is organized as follows:
•Appendix A: Algorithm. Full inference procedure of Exe-
cutionRAG, including register construction, register updates,
progress-delta injection, and budget-aware control.
•Appendix B: Extended Implementation Details. Additional
details on model configuration, retrieval tools, budgets, dedupli-
cation, progress-delta injection, and other engineering choices.
•Appendix C: Per-Category Breakdown. Category-level re-
sults for HotpotQA, 2WikiMultiHopQA, and MuSiQue under
both embedding backbones.
•Appendix D: Retrieval Efficiency Analysis. Retrieved to-
kens and API cost across methods, ablations, and embedding
backbones.
•Appendix E: Failure Case Analysis. Error breakdown by
type, hop count, and embedding model.
•Appendix F: Case Study: How the Register Guides the
Search. Qualitative examples showing how the register keeps
the search on target and supports self-correction.
•Appendix G: Prompt Templates. Key prompts used in Exe-
cutionRAG, including initialization, update, budget warning,
and force-answer prompts.
•Appendix H: Broader Impacts. Discussion of potential ap-
plications, transparency benefits, and responsible usage consid-
erations.
A Algorithm
B Extended Implementation Details
Language model.All experiments use GPT-4o-mini as the default
LLM. Structured JSON calls for seed analysis and register updates
usetemperature = 0.0 to reduce output variance, while the tool-
using agent loop uses the model’s default sampling settings. The
maximum output length for each LLM call is 16,384 tokens.
Retrieval tools.The agent has access to three tools followed by
A-RAG [ 5]:semantic_search for dense cosine retrieval with the
active embedding model (default top- 𝑘=5),keyword_search for
BM25-style lexical retrieval (default top- 𝑘=5), and read_chunk for
direct chunk lookup by identifier. Seed retrieval uses semantic_search
with top-𝑘= 5. Retrieved chunks are deduplicated within each
loop, and previously read chunks are not re-exposed to the register
update prompt.
Embedding models.The primary embedding model isall-
MiniLM-L6-v2, a 384-dimensional bi-encoder optimized for se-
mantic similarity. The alternative backbone isQwen3-Embedding-
0.6B, a 1024-dimensional embedding model. Embeddings are
pre-computed and stored as flat similarity indices; no re-indexing
is performed between runs.
Evaluation corpus.We follow the HippoRAG evaluation protocol,
using the same retrieval corpora and the same 1,000-question splits
sampled from the official validation sets of HotpotQA, 2WikiMul-
tiHopQA, and MuSiQue. All baselines are evaluated on identical
question sets.Algorithm 1ExecutionRAG Inference
Require:Query𝑞, corpusK, retrieval budget𝐵
Ensure:Final answer𝑎
1:C(0)←𝑅 seed(𝑞;K)
2:S 0←InitializeState(𝑞,C(0))
3:Inject the initial register summary of S0into the agent context
4:for𝑡=1to𝐵do
5:ifReadyToAnswer(𝑞,S 𝑡−1)then
6:return𝐺(𝑞,S 𝑡−1)
7:end if
8:𝑏𝑡←SelectActiveReqirement(S 𝑡−1)
9:𝑢𝑡←Compose(𝑞,𝑑 𝑏𝑡,𝑒(𝑡−1)
𝑏𝑡)
10:ΔC 𝑡←𝑅(𝑢𝑡;K)
11:ifΔC 𝑡=∅then
12:S 𝑡←S𝑡−1
13:else
14: ˜S𝑡←𝑈(S𝑡−1,ΔC𝑡)
15:S 𝑡←ApplyInvariants( ˜S𝑡,S𝑡−1)
16:end if
17:ifHasProgressDelta(S 𝑡−1,S𝑡)then
18:Inject progress delta into the agent context
19:end if
20:if𝑡≥0.8𝐵then
21:Inject budget-aware nudge
22:end if
23:end for
24:return𝐺(𝑞,S 𝐵)
Retrieval budget.The agent is allowed at most 15 retrieval
loops per question. We also enforce a cumulative context bud-
get of 128,000 tokens. If this budget is exceeded, a force-answer
prompt is injected and further tool calls are disabled, yielding a
budget-exhausted answer, categorized as “Gave Up” in the error
analysis.
Search deduplication.Each retrieval action is cached by a normal-
ized key: sorted keywords for keyword_search and the lowercased
query string for semantic_search . If the agent issues an identical
query, the system returns a deduplication notice with the previ-
ously retrieved chunk IDs, prompting the agent to read existing
chunks or reformulate the query. This reduces cyclic search and
unnecessary token usage.
Progress delta injection.Rather than replaying the full register
after every step, the system injects a compact [Progress] delta
only when at least one item changes materially. The delta records
the changed items with abbreviated evidence notes and a one-
line summary of the items still open. On 4-hop questions, this
reduces register-injection overhead by approximately 40% relative
to replaying the full list.
Budget-aware nudge.When the agent reaches 80% of its loop
budget, a soft warning is injected to encourage synthesis if the
current evidence is sufficient, or to make the remaining retrieval
steps more targeted. This reduces redundant last-mile searches near
budget exhaustion.

DAI ’26, November 29–December 2, 2026, Hong Kong, China Wei et al.
C Per-Category Breakdown
Tables 3–8 report per-category Contain-Match Accuracy and LLM-
Evaluation Accuracy (%) for ExecutionRAG with GPT-4o-mini on
the full 1,000-question evaluation sets, under two embedding back-
bones: all-MiniLM-L6-v2 and Qwen3-Embedding-0.6B.
Table 3: Per-category accuracy on MuSiQue by hop count
(GPT-4o-mini + all-MiniLM-L6-v2,𝑛=1,000).
Category N Contain Acc (%) LLM Acc (%)
2-hop 518 57.7 56.8
3-hop 316 44.0 50.6
4-hop 166 31.9 28.9
Overall1,00049.1 50.2
Table 4: Per-category accuracy on HotpotQA by question type
(GPT-4o-mini + all-MiniLM-L6-v2,𝑛=1,000).
Category N Contain Acc (%) LLM Acc (%)
Bridge 811 74.8 78.1
Comparison 189 86.2 87.3
Overall1,00077.0 79.8
Table 5: Per-category accuracy on 2WikiMultiHopQA by
question type (GPT-4o-mini + all-MiniLM-L6-v2,𝑛=1,000).
Category N Contain Acc (%) LLM Acc (%)
Compositional 413 63.5 66.7
Comparison 244 84.4 88.1
Bridge-Comparison 235 89.2 85.6
Inference 108 49.1 48.1
Overall1,00073.0 74.3
Table 6: Per-category accuracy on MuSiQue by hop count
(GPT-4o-mini + Qwen3-Embedding-0.6B,𝑛=1,000).
Category N Contain Acc (%) LLM Acc (%)
2-hop 518 50.0 51.0
3-hop 316 39.6 40.2
4-hop 166 24.7 22.3
Overall1,00042.5 42.8
D Retrieval Efficiency Analysis
We analyze the retrieval efficiency of ExecutionRAG relative to
the strongest agentic baseline A-RAG and to our ablation variants.
Unless otherwise specified, all main comparisons use GPT-4o-mini
with all-MiniLM-L6-v2 on the full 1,000-question evaluation sets.Table 7: Per-category accuracy on HotpotQA by question type
(GPT-4o-mini + Qwen3-Embedding-0.6B,𝑛=1,000).
Category N Contain Acc (%) LLM Acc (%)
Bridge 811 73.9 78.2
Comparison 189 83.6 87.8
Overall1,00075.7 80.0
Table 8: Per-category accuracy on 2WikiMultiHopQA by
question type (GPT-4o-mini + Qwen3-Embedding-0.6B,
𝑛=1,000).
Category N Contain Acc (%) LLM Acc (%)
Compositional 413 52.8 55.2
Comparison 244 82.8 90.2
Bridge-Comparison 235 88.1 75.7
Inference 108 50.0 52.8
Overall1,00068.1 68.3
Table 9: Retrieval efficiency comparison on three bench-
marks (𝑛=1,000, GPT-4o-mini + all-MiniLM-L6-v2). Tokens:
average retrieved tokens per question; Cost: average API cost
per question (USD).
Dataset Method LLM Acc Cont Acc Tokens/Q Cost/Q
MuSiQueA-RAG 44.9 42.0 5,015 $0.0034
Ours 50.2 49.17,161 $0.0064
HotpotQAA-RAG 78.6 77.3 2,860 $0.0016
Ours 79.8 77.03,373 $0.0033
2WikiA-RAG 59.8 66.7 3,992 $0.0027
Ours 74.3 73.03,966 $0.0039
Table 10: Retrieval efficiency under ablation on MuSiQue
(𝑛=1,000, GPT-4o-mini + all-MiniLM-L6-v2). We report token
consumption and cost alongside the accuracy numbers from
Table 2.
Dataset Variant LLM Cont Tokens/Q Cost/Q
MuSiQueFull 50.2 49.1 7,161 $0.0064
w/o Bootstrap 48.7 46.2 5,962 $0.0047
w/o Progress Register 48.3 48.5 6,378 $0.0042
E Failure Case Analysis
To better understand the remaining failure modes of ExecutionRAG,
we conduct a systematic error analysis on MuSiQue ( 𝑛=1,000) and
compare against the strongest agentic baseline, A-RAG. We report
results under both embedding backbones to assess whether the
observed patterns are robust to retriever choice.
Annotation method.We leverage MuSiQue’s ground-truth anno-
tations, which provide for each question a set ofsupporting para-
graphscorresponding to the minimal evidence needed for a correct

Beyond Retrieval Failure: Understanding Why LLMs Miss Evidence in Multi-Hop RAG DAI ’26, November 29–December 2, 2026, Hong Kong, China
0 25 50 75 10057.7 2-hop
44.0 3-hop
31.9 4-hop
49.1 Overall
74.8 Bridge
86.2 Comparison
77.0 Overall
63.5 Compositional
84.4 Comparison
89.2 Bridge-Comp.
49.1 Inference
73.0 OverallMuSiQue
HotpotQA
2Wiki(a) all-MiniLM-L6
(Contain)
Accuracy (%)0 25 50 75 10056.8
50.6
28.9
50.2
78.1
87.3
79.8
66.7
88.1
85.6
48.1
74.3(b) all-MiniLM-L6
(LLM)
Accuracy (%)0 25 50 75 10050.0
39.6
24.7
42.5
73.9
83.6
75.7
52.8
82.8
88.1
50.0
68.1(c) Qwen3-0.6B
(Contain)
Accuracy (%)0 25 50 75 10051.0
40.2
22.3
42.8
78.2
87.8
80.0
55.2
90.2
75.7
52.8
68.3(d) Qwen3-0.6B
(LLM)
Accuracy (%)
Figure 5: Per-category performance of ExecutionRAG under both embedding backbones. We report Contain-Match Accuracy
and LLM-Evaluation Accuracy on MuSiQue (by hop count), HotpotQA (by question type), and 2WikiMultiHopQA (by question
type).
Table 11: Efficiency comparison across embedding models
(𝑛=1,000, GPT-4o-mini). ExecutionRAG maintains its accu-
racy advantage with both embedding backbones, with com-
parable or fewer retrieved tokens on 2WikiMultiHopQA.
all-MiniLM-L6-v2 Qwen3-Embed-0.6B
Dataset MethodLLM Tok/Q Cost/Q LLM Tok/Q Cost/Q
MuSiQueA-RAG 44.9 5,015 .0034 42.3 5,217 .0040
Ours 50.27,161 .006442.86,060 .0083
HotpotQAA-RAG 78.6 2,860 .0016 76.4 3,533 .0026
Ours 79.83,373 .003380.03,568 .0052
2WikiA-RAG 59.8 3,992 .0027 55.2 4,311 .0034
Ours 74.33,966 .003968.3 4,044.0051
Full w/o
Init.w/o
Register0204060LLM Acc (%)
0246810
Avg Tokens/Q (K)
Full (Ours)
w/o Register Init.w/o Progress Register
Tokens/Q
Figure 6: Accuracy–efficiency trade-off across ablation vari-
ants on MuSiQue, the same variants as Table 2. Removing
a register component lowers retrieved tokens and accuracy
together.
answer. We map each supporting paragraph to its correspondingcorpus chunk using 40-character fingerprint matching (paragraph-
level recall: 87.9%), and computehop-level evidence recallas the
fraction of supporting paragraphs for which at least one match-
ing chunk was retrieved. Each failure ( LLM-Acc= 0) is assigned to
exactly one category using the following priority order:
(1)Judge Error. Contain-Acc= 1but LLM-Acc= 0, indicating an
evaluation mismatch.
(2)Gave Up.The predicted answer contains explicit abstention
phrases (e.g., “unable to find”, “not found”).
(3)Reasoning Error.Either (a) all supporting chunks were re-
trieved but the answer is still wrong, or (b) [ExecutionRAG
only] the register marks the question sufficiently answered
despite remaining evidence gaps (false resolution).
(4)Missing Evidence.Some supporting evidence was never re-
trieved and no false resolution occurred. This observable cate-
gory includes both unreachable and unasked evidence.
Reading the counts.All comparisons below are reported as
within-method percentages of each method’s own total failures.
The totals themselves come from this failure-analysis pass, and
for ExecutionRAG under all-MiniLM-L6-v2 the total is slightly
larger than the failure count implied by that method’s main-table
accuracy, so the reduction reported for that pairing is if anything
conservative.
Results (all-MiniLM-L6-v2).
Results (Qwen3-Embedding-0.6B).
Key findings.First, Missing Evidence is reduced under both
embedding backbones.With all-MiniLM-L6-v2, ExecutionRAG
reduces Missing Evidence from 382 to 234 cases (69.3% →43.9% of
each method’s failures). With Qwen3-Embedding-0.6B, it reduces
the count from 383 to 274 (66.4% →47.9%). This suggests that keep-
ing the open sub-questions on an explicit register improves evidence
acquisition robustly across retrievers.
Second, evidence recovery exposes a different remaining
bottleneck.As Missing Evidence decreases, Reasoning Error be-
comes more prominent for ExecutionRAG (MiniLM: 9.4% →25.9%;

DAI ’26, November 29–December 2, 2026, Hong Kong, China Wei et al.
Table 12: Failure case analysis on MuSiQue ( 𝑛=1,000, GPT-4o-
mini + all-MiniLM-L6-v2). Percentages are relative to total
failures for each method.
A-RAG ExecutionRAG
Category Count % Count %
Total failures 551 100.0 533 100.0
Judge Error 81 14.7 91 17.1
Gave Up 36 6.5 70 13.1
Reasoning Error 52 9.4 138 25.9
Missing Evidence 38269.323443.9
Table 13: Failure distribution by hop count (%, MiniLM).
ExecutionRAG reduces Missing Evidence at every hop level
(−25.4pp overall).
Hop Method Judge Gave Up Reasoning Missing
2-hopA-RAG 15.7 7.5 15.061.8
ExecutionRAG 18.6 14.2 30.137.2
3-hopA-RAG 15.8 4.8 6.173.3
ExecutionRAG 17.7 14.3 24.044.0
4-hopA-RAG 10.9 6.7 1.780.7
ExecutionRAG 13.6 9.8 21.255.3
Table 14: Failure case analysis on MuSiQue ( 𝑛=1,000, GPT-4o-
mini + Qwen3-Embedding-0.6B). Percentages are relative to
total failures for each method.
A-RAG ExecutionRAG
Category Count % Count %
Total failures 577 100.0 572 100.0
Judge Error 80 13.9 78 13.6
Gave Up 69 12.0 99 17.3
Reasoning Error 45 7.8 121 21.2
Missing Evidence 38366.427447.9
Qwen: 7.8%→21.2%). These cases include false resolution, where the
register marks an item answered too optimistically, as well as cases
where retrieved evidence is present but incorrectly synthesized.
Third, abstention increases as the register makes control
more explicit.TheGave Uprate rises from 6.5% to 13.1% under
MiniLM and from 12.0% to 17.3% under Qwen, indicating that Execu-
tionRAG is more willing to stop when the tracked evidence remains
insufficient, rather than forcing a weakly supported answer.
Finally, long-hop retrieval remains the dominant bottle-
neck.For ExecutionRAG, Missing Evidence still increases with hop
count, reaching 55.3% on 4-hop MuSiQue under MiniLM. This sug-
gests that although the register substantially improves the search,
acquiring long and weakly connected evidence chains remains the
main frontier for further progress.Table 15: Failure distribution by hop count (%, Qwen). Exe-
cutionRAG reduces Missing Evidence at every hop level
(−18.5pp overall).
Hop Method Judge Gave Up Reasoning Missing
2-hopA-RAG 16.1 13.9 12.557.5
ExecutionRAG 16.9 17.7 25.240.2
3-hopA-RAG 11.4 9.1 5.773.9
ExecutionRAG 14.3 15.3 17.552.9
4-hopA-RAG 12.5 11.7 0.875.0
ExecutionRAG 6.2 19.4 18.655.8
F Case Study: How the Register Guides the
Search
To give qualitative intuition for how the register keeps the search
on target and heads off common failures, we show three represen-
tative examples from MuSiQue (GPT-4o-mini + all-MiniLM-L6-v2).
The[Progress] annotations in the ExecutionRAG column are the
delta messages the agent actually sees: only register changes that
materially affect the search are surfaced. The three cases illustrate
three effects: anchoring on the right entity (Case 1), completing
a long chain via progress deltas (Case 2), and self-correction by
dropping a wrong item (Case 3).
Case 1: Preventing Entity Confusion (2-hop,
Success)
Question:Who married the publisher of abolitionist newspaper The
North Star?
Gold answer:Helen Pitts Douglass
Analysis.This example illustrates the value ofanchoring on the
right entityat the start. The seed-grounded register records Freder-
ick Douglass as the entity the answer chain must stay attached to, so
the later search stays on the correct branch. A-RAG has no explicit
record of which entity the question has already grounded; once it
latches onto Nell, later retrieval follows that mistaken branch and
never recovers.
Case 2: Completing a Long Dependency Chain
(4-hop, Success)
Question:What day is the feast held, in the city that holds the
headquarters of the only music company that is larger than the record
label of Vilaiyaadu Mankatha?
Gold answer:May 4
Analysis.This case shows how register updates turn a long multi-
hop problem into a sequence of targeted retrieval steps. The key
advantage is not decomposition alone, but the fact that newly an-
swered items are surfaced asprogress deltas, which immediately
sharpen the next search. Once “Santa Monica” is recorded on the
register, the next retrieval step directly targets the remaining gap.
Without such a register, A-RAG repeatedly searches in a much flat-
ter space and accumulates substantial irrelevant context before the
decisive entity is stabilized.

Beyond Retrieval Failure: Understanding Why LLMs Miss Evidence in Multi-Hop RAG DAI ’26, November 29–December 2, 2026, Hong Kong, China
ExecutionRAG(2 loops, 1 chunk)A-RAG(5 loops, 2 chunks)
Warm-up:Initial evidence correctly identifies Frederick Douglass as the
publisher. The question is decomposed into two requirements:Loop 1: keyword_search for “The North Star, publisher, married.” A top
result mentionsWilliam Cooper Nell, whowrote forThe North Star but
was not its publisher.
req_0 : “identify the relevant marriage relation of Frederick Douglass” →
activeLoop 2: read_chunk(126) — William Cooper Nell biography; no answer
found.
req_1: “identify Frederick Douglass’s wife”→pending
Loop 1:Retrieved chunk 127 containing Douglass biography and spouse
information.Loop 3: keyword_search for “William Cooper Nell, married, wife.” The
search trajectory is now anchored on the wrong entity.
State update: Loop 4:Reads another irrelevant marriage-related chunk.
req_0: remainsactive
req_1:pending→ resolved
Evidence: “Helen Pitts Douglass, his second wife”
Answer:✓Helen Pitts DouglassLoop 5:Generates an answer based on the incorrect Nell branch.
Answer:✗William Cooper Nell / Sarah Jane Emery
Table 16: Case 1: ExecutionRAG’s seed-grounded register anchors the correct entity, Frederick Douglass, before iterative
retrieval begins. A-RAG instead latches onto William Cooper Nell, a contributor rather than the publisher, and continues
retrieval along the wrong branch.
ExecutionRAG(6 loops, 3 chunks)A-RAG(13 loops, 19 chunks)
Warm-up:The question is decomposed into a dependency chain: Loop 1:Parallel searches for “Vilaiyaadu Mankatha + music company
+ headquarters” and “feast + city + day.” The latter is issued before the
relevant city has been identified.
req_2: “identify the company larger than Sony Music”→pending
req_1: “identify the headquarters city of that company”→pending
req_0: “identify the feast day associated with that city”→ active Loops 2–3:Reads multiple generic “feast/city” chunks, none linked to the
music-company chain.
Loops 1–2: keyword_search (“Vilaiyaadu Mankatha, record label”);
read_chunk(18,17) . Chunk 18 resolves two links at once: Sony Music →
Universal Music Group→Santa Monica.Loops 4–6:Repeated searches for “largest music company, Sony Music,
headquarters,” with re-reading of overlapping evidence and no explicit
record of intermediate progress.
State update (delta): Loop 7:Eventually reads the key chunk, but only after substantial context
has already been consumed by unrelated evidence.
req_2:pending→ resolved“Universal Music Group”
req_1:pending→ resolved“Santa Monica, California”
Loops 3–5:Conditioned on the updated state, the agent issues a tar-
geted follow-up: keyword_search (“feast, Santa Monica, California”) and
semantic_search(“feast Santa Monica California”). Reads chunk 19.Loops 8–12:Additional weakly constrained searches for “feast day city”
continue without converging on the now-identified city.
Loop 6 (closure):req_0resolves with Saint Monica’s feast day. Loop 13:Near budget exhaustion, the agent states that the feast is not
explicitly mentioned and falls back to an incorrect alternative.
Answer:✓May 4Answer:✗“Not explicitly mentioned”
Table 17: Case 2: ExecutionRAG resolves a long dependency chain by turning register updates into targeted follow-up retrieval.
Once the register records “Universal Music Group” and “Santa Monica,” the next search becomes sharply focused. A-RAG lacks
an explicit record of this intermediate progress and keeps issuing weakly constrained searches.
Case 3: Self-Correction via Requirement
Retirement (4-hop, Success)
Question:When did the birthplace of Ah Via Musicom’s performer
become the capital of the state Knowles is from?
Gold answer:1839Analysis.This example highlightsself-correction by dropping a
wrong item. Both systems initially move toward the wrong city,
but ExecutionRAG can explicitly retire the invalid item once later
evidence contradicts it. Dropping it is recorded as an explicit event
rather than silently overwritten, which forces the search to reopen

DAI ’26, November 29–December 2, 2026, Hong Kong, China Wei et al.
ExecutionRAG(6 loops, 3 chunks)A-RAG(3 loops, 3 chunks)
Warm-up:Initial evidence identifies Beyoncé Knowles as being from
Texas, but incorrectly assumes that Houston is also the birthplace of the
performer ofAh Via Musicom. The initial requirement is therefore:Loop 1:Parallel searches for “Ah Via Musicom, performer, birthplace”
and “capital, state, Knowles.” Both retrieve relevant but not yet integrated
evidence.
req_0 : “determine when Houston became the capital of Texas” →active Loop 2:Reads the relevant chunks identifying Eric Johnson and Texas,
but does not revisit whether Houston is actually the correct city.
Loops 1–2: keyword_search (“Ah Via Musicom, performer, birthplace”)
andkeyword_search(“capital, state, Knowles”).read_chunk(630,123).Loop 3 (closure):Produces a final answer based on an incorrect chain,
conflating related Texas dates.
State update (loop 2):The retrieved evidence contradicts the current
requirement premise: Houston is not the capital of Texas, whereas Austin
is.
req_0:active→ retired
Evidence: “Houston is not the capital of Texas; Austin is the capital. Austin
became the capital of the Republic of Texas in 1839.”
Loops 3–5:After retiring the invalid requirement, ExecutionRAG is-
sues targeted follow-ups: keyword_search (“Eric Johnson, birthplace”) and
keyword_search (“Eric Johnson, Austin, Texas”). read_chunk(629) con-
firms Eric Johnson was born in Austin.
Loop 6 (closure):The corrected chain is completed: Eric Johnson →
Austin→Texas→1839.Answer:✗1836
Answer:✓1839
Table 18: Case 3: ExecutionRAG uses the retired transition to explicitly discard an invalid requirement when later evidence
contradicts its premise. A-RAG reads overlapping evidence but lacks an explicit mechanism for requirement revision, and
therefore closes on a plausible but incorrect year.
the chain with a corrected target. A-RAG has no explicit way to
invalidate an earlier intermediate commitment, so its trajectory
closes on a plausible but unsupported date.
Summary.These three cases highlight the main qualitative ef-
fects of the register:
•Anchoring on the right entity(Case 1): building the register
from seed evidence keeps retrieval on the correct entity and
prevents early drift onto a wrong branch.
•Delta-guided chain completion(Case 2): answered items are
surfaced as progress deltas, enabling sharply targeted follow-up
retrieval instead of repeated weakly constrained search.
•Self-correction by dropping a wrong item(Case 3): contradic-
tory evidence can explicitly invalidate an earlier item, letting the
agent revise the chain instead of closing on a stale assumption.
G Prompt Templates
This section lists the key prompts used in ExecutionRAG. All
prompts are invoked with GPT-4o-mini unless otherwise stated.
A.1 Default QA Agent System Prompt
Used to initialize the tool-using retrieval agent. The agent iteratively
calls keyword_search ,semantic_search , and read_chunk while
using the register as guidance.Default QA Agent System Prompt
You are a question-answering assistant with access to a document corpus
through available tools.
Your goal is to answer questions accurately by finding and analyzing
relevant information from the provided documents.
Available Tools
•keyword_search: Find chunks by exact keyword matching
•semantic_search: Find chunks by semantic similarity
•read_chunk: Read the full content of a specific chunk
Strategy
Work iteratively: search →read→evaluate→update progress→
···→ answer. For multi-hop questions, use the requirement state to
keep track of what has already been established, what remains unre-
solved, and what should be searched next.
When Answering
Ground your response in the retrieved documents. Cite the specific
chunks that support your answer. Provide a clear, direct answer sup-
ported by evidence. If the evidence is still insufficient, state the remaining
uncertainty rather than speculating beyond the documents.
A.2 Seed-Grounded Requirement Initialization
Prompt
Run once before iterative retrieval begins. Given the question and
seed evidence, this prompt builds the initial register (stored as the
requirement_register).

Beyond Retrieval Failure: Understanding Why LLMs Miss Evidence in Multi-Hop RAG DAI ’26, November 29–December 2, 2026, Hong Kong, China
Seed-Grounded Requirement Initialization Prompt
Question: {query}
Available seed evidence:
{info_summary}
Analyze the question and seed evidence, then output a JSON object with
the following fields:
•"evidence_summary" : concise summary of what is already known
from the seed evidence. Preserve specific facts, names, dates, and num-
bers.
•"requirement_register" : a dictionary mapping req_id to{"text":
str, "status": "pending"|"active", "evidence_notes": str} .
Each item should represent a concrete information requirement needed
to answer the question. Exactly one requirement should be marked
"active"; the others should be"pending".
•"reasoning_note" : one short sentence explaining why the active
requirement is the best current target.
Requirements should be concrete, searchable, and grounded in the seed
evidence rather than speculative.
Output ONLY the JSON object.
A.3 Requirement-State Injection
Appended to the agent context after warm-up. It exposes the current
register as guidance without overriding the agent’s autonomy.
Requirement-State Injection
## Current Requirement State
The following answer requirements have been identified:
{requirement_register}
Use this state to decide what to search for next. Focus on unresolved
requirements, preserve the evidence already gathered, and avoid repeat-
ing searches that do not advance the state. You may receive [Progress]
messages describing requirement-state changes after new evidence is
read.
A.4 Requirement Register Update Prompt
Called after each retrieval loop to update the register. The LLM
decides whether new evidence advances, resolves, or invalidates
each tracked item.
A.5 Force Final Answer Prompt
Appended when the agent reaches the maximum loop or token
budget. It disables further tool calls and forces the agent to answer
from the current evidence state.
H Broader Impacts
Our work improves multi-hop QA by keeping an explicit register of
open sub-questions during multi-step retrieval. Related language-
model and agent systems have been studied for autonomous scien-
tific discovery [ 39], few-shot tabular learning [ 48], paper-to-posterRequirement Register Update Prompt
Question: {query}
Current requirement register:
{requirement_register}
Newly retrieved information:
{new_evidence}
Update the requirement register based on the new information. Output
a JSON object with the following field:
•"requirement_register" : dictionary mapping req_id to{"text":
str, "status": "pending"|"active"|"resolved"|"retired",
"evidence_notes": str}
Update each requirement as follows:
•Set "resolved" only if the currently available evidence is sufficient
to support that requirement.
•Set "active" for the requirement that should currently guide the next
retrieval step.
•Set "retired" only if the requirement is no longer valid or is contra-
dicted by the new evidence.
•Otherwise keep it"pending".
Keep ALL existing requirements; do not drop any. Update
evidence_notes with a concise summary of the supporting or
contradicting facts currently available.
Output ONLY the JSON object.
Force Final Answer Prompt
You have reached the budget limit. You MUST now provide a final
answer based on the information gathered so far. Do NOT call any more
tools. If the evidence is sufficient, answer directly. If the evidence is still
incomplete, provide the best bounded answer you can and clearly state
the remaining uncertainty.
generation [ 52], and benchmarks that probe the cognitive abilities
such systems require [ 53]. These settings illustrate applications in
which an agent must maintain intermediate requirements while
integrating information from multiple sources. By making evidence
support and provenance explicit, ExecutionRAG can improve trans-
parency and auditability in such workflows.
However, the method still depends on the underlying retriever
and LLM, so biases or errors in source documents may propagate.
The framework could be misused in automated decision-making
systems without human oversight, producing overconfident an-
swers based on incomplete evidence. We recommend deployment
in settings where accountability and verification are feasible.