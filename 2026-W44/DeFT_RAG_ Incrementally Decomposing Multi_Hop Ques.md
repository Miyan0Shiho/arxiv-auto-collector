# DeFT-RAG: Incrementally Decomposing Multi-Hop Questions into Self-Contained Sub-Questions with Reusable Answers

**Authors**: Dingyi Kang, Bingzhe Li

**Published**: 2026-10-27 18:40:23

**PDF URL**: [https://arxiv.org/pdf/2610.37826v1](https://arxiv.org/pdf/2610.37826v1)

## Abstract
Retrieval-augmented generation (RAG) lets a large language model (LLM) answer questions by retrieving relevant passages from a large corpus. However, RAG struggles with multi-hop questions, which require chaining facts spread across several documents through intermediate entities that the question never mentions. Graph-based RAG methods retrieve once from the entities a question names and so rarely recover the full evidence chain. Iterative RAG methods recover it step by step, but their accuracy is limited by the many tasks they ask the LLM to perform, such as planning the sub-questions, deciding their dependencies, rewriting them as answers arrive, and keeping track of what is known. We propose DeFT-RAG, a RAG framework built on three components. First, DeFT-RAG generates one self-contained sub-question at a time from the original question and the sub-questions answered so far with their answers (QA pairs). Each sub-question names every entity it asks about, which removes the need to plan, track dependencies or rewrite, and lets the LLM focus on a single fine-grained task at a time. Second, DeFT-RAG moves the remaining LLM work outside the LLM: rule-based updates to a QA table that keeps track of what is known, and yes/no decisions made by a fine-tuned lightweight decision model. Third, DeFT-RAG caches the QA pairs and reuses an answer when a later question asks a sub-question with the same meaning. On MuSiQue, 2WikiMultiHopQA and HotpotQA, DeFT-RAG achieves the best accuracy among thirteen baselines with both gpt-4o-mini and Qwen3.6 as the backbone, while giving the LLM fewer input tokens than any other iterative method. With a QA cache shared across 1,000 MuSiQue questions, 23.1% of sub-questions are answered from the cache, which reduces LLM calls by 12.9% without lowering accuracy.

## Full Text


<!-- PDF content starts -->

DeFT-RAG: Incrementally Decomposing Multi-Hop Questions
into Self-Contained Sub-Questions with Reusable Answers
Dingyi Kang
University of Texas at Dallas
Richardson, Texas, USA
dingyi.kang@utdallas.eduBingzhe Li
University of Texas at Dallas
Richardson, Texas, USA
bingzhe.li@utdallas.edu
Abstract
Retrieval-augmented generation (RAG) lets a large language model
(LLM) answer questions by retrieving relevant passages from a large
corpus. However, RAG struggles with multi-hop questions, which
require chaining facts spread across several documents through in-
termediate entities that the question never mentions. Graph-based
RAG methods retrieve once from the entities a question names and
so rarely recover the full evidence chain. Iterative RAG methods re-
cover it step by step, but their accuracy is limited by the many tasks
they ask the LLM to perform, such as planning the sub-questions,
deciding their dependencies, rewriting them as answers arrive, and
keeping track of what is known. We propose DeFT-RAG, a RAG
framework built on three components. First, DeFT-RAG generates
one self-contained sub-question at a time from the original ques-
tion and the sub-questions answered so far with their answers
(QA pairs). Each sub-question names every entity it asks about,
which removes the need to plan, track dependencies or rewrite, and
lets the LLM focus on a single fine-grained task at a time. Second,
DeFT-RAG moves the remaining LLM work outside the LLM: rule-
based updates to a QA table that keeps track of what is known, and
yes/no decisions made by a fine-tuned lightweight decision model.
Third, DeFT-RAG caches the QA pairs and reuses an answer when
a later question asks a sub-question with the same meaning. On
MuSiQue, 2WikiMultiHopQA and HotpotQA, DeFT-RAG achieves
the best accuracy among thirteen baselines with both gpt-4o-mini
and Qwen3.6 as the backbone, while giving the LLM fewer input
tokens than any other iterative method. With a QA cache shared
across 1,000 MuSiQue questions, 23.1% of sub-questions are an-
swered from the cache, which reduces LLM calls by 12.9% without
lowering accuracy.
CCS Concepts
•Information systems →Question answering;•Computing
methodologies→Natural language generation.
Keywords
Retrieval-augmented generation, multi-hop question answering,
question decomposition
1 Introduction
Large language models (LLMs) now perform strongly across a wide
range of language tasks [ 19,28]. Yet because their knowledge is
stored in their parameters, they often answer incorrectly when that
knowledge is outdated or missing, and they may hallucinate [ 11].
Retrieval-augmented generation (RAG) [ 5,15] mitigates this by
retrieving passages from an external corpus and grounding theanswer in them, which improves accuracy on knowledge-intensive
tasks without retraining the model.
A single retrieve-then-generate step suffices when one passage
holds the answer, but it falls short on multi-hop question answer-
ing [ 9,24,29], where answering one question requires chaining
several facts spread across different documents: retrieving with the
question alone returns passages that match its surface terms, and
these rarely cover every link of the chain. Graph-based RAG meth-
ods [ 2,4,6–8,27] address this by building a graph over the corpus
that links entities and passages, then retrieving passages by follow-
ing links from the entities in the question. However, a multi-hop
chain runs throughbridge entitiesthat the question never names and
that only become known once an earlier step is answered. Because
most graph-based methods retrieve once per question, starting from
the entities the question names, they can reach the bridge entities
only by following graph links from those entities, which rarely gets
past the first one: for most of these methods, the full evidence chain
is retrieved for only 6–10% of 3-hop MuSiQue questions (Figure 1a,
Table 3).
Iterative RAG methods [ 3,14,20,25] instead aim to find these
bridge entities directly: they decompose the question into sub-
questions and alternate between retrieval and LLM reasoning, so
each answered sub-question supplies the bridge entity needed by
the next retrieval. However, their accuracy is limited by the many
tasks they ask the LLM to perform, often within one call, such
as planning the sub-questions, deciding the dependencies among
them, rewriting them as answers arrive, and keeping track of what
is known, each of which leads to its own type of error. Specifically, a
plan written before any sub-question is answered must guess bridge
entities, and a wrong guess or a missing step misleads every later
step (planning errors). The dependencies decided among the sub-
questions may link them wrongly or in the wrong order, so a sub-
question is searched before the fact it needs is found (dependency
errors), as in Figure 1b. When an answer arrives, the LLM may
fail to rewrite the sub-question that depends on it, which keeps a
placeholder instead of the entity’s name, or may rewrite it with
the wrong entity (rewriting errors). In a long, growing input, the
LLM may also lose track of what it has already found (tracking
errors), as in Figure 1c. Moreover, fixed top- 𝑘retrieval, used by
most RAG methods, makes these issues worse by lowering the
answer accuracy at each step. Finally, iterative methods discard
their intermediate results once a question is answered, although
multi-hop questions over the same corpus often share intermediate
facts that could be reused.
Based on these observations, we propose DeFT-RAG, a RAG
framework built on three components. First, DeFT-RAG generates
one self-contained sub-question at a time from the original question
arXiv:2610.37826v1  [cs.CL]  27 Oct 2026

Dingyi Kang and Bingzhe Li
Q: Where do greyhound buses leave from in the city where the performer of Darlingsformed?Gold chain(two bridge entities the question never names)DarlingsKevin DrewTorontoToronto Coach Terminalis an album byfounded his band inGreyhound buses leave frombridgebridgeanswer(a) Graph-based RAG: unnamed bridge entity missedHow far along the chain HippoRAG 2 retrievesDarlingsKevin DrewTorontoToronto CoachTerminal✓found✓foundnever retrievedIt retrieves passages linked to Darlings, the entity thequestion names. It learns that the album is by KevinDrew, but no link leads on to his band or to Toronto.(b) Iterative RAG: dependency errorLogicRAG's plan, written before any answer①“City wheretheperformeris basedor formed”②“Identity of theperformer of Darlings”performer not yet knownnever searchedStep ①needs the performer but comes first. Searchedwithout a name, it returns bands from other cities, andstep②is never run.(c) Iterative RAG: tracking errorIRCoT's prompt grows: 9 passages + its past reasoningDarlingsKevin DrewTorontoToronto CoachTerminal✓✓✓✓IRCoTwrites: “…Broken Social Scene is based inToronto, Canada. However, the document states thatGreyhound operates in Philadelphia…”Retrieved insteadPhiladelphia:…a major hub forGreyhound Lines…Answers from its first retrieval (whole question):Philadelphia:…a major hub forGreyhound Lines…LLMPhiladelphiaGreyhound Terminal✗Answers from:Philadelphia:…a major hub forGreyhound Lines…LLMPhiladelphiaGreyhound Terminal✗found Toronto, then lost itLLMPhiladelphiaGreyhound Terminal✗+ 6 off-chain passages (Philadelphia, Darling Scarp, …)
Figure 1: Failures of graph-based and iterative RAG on a 3-hop MuSiQue question (gpt-4o-mini runs). Top: the gold chain,
which passes through two bridge entities the question never names. (a) HippoRAG 2, a graph-based method, finds the first
bridge entity (Kevin Drew) but no passage about his band or Toronto. (b) LogicRAG, which plans its sub-questions in advance,
searches for the city of “the performer” before finding the performer (dependency error). (c) IRCoT, which keeps all retrieved
passages in its long prompt, has the whole chain there and even states that the band is based in Toronto, but then loses track of
this fact (tracking error).
and the sub-questions answered so far with their answers (QA pairs),
until these accumulated QA pairs are enough to answer the original
question. Each sub-question names every entity it asks about and
does not depend on the answer to any other sub-question. Since
nothing is planned in advance, no bridge entity has to be guessed
before it is found, and there are no dependencies to get wrong and
no sub-question to rewrite. This avoids planning, dependency and
rewriting errors, and lets the LLM focus on a single fine-grained
task at a time: writing a sub-question for the next missing piece
of information. Second, DeFT-RAG converts the remaining LLM
work into fine-grained tasks handled outside the LLM. A QA table
keeps every QA pair and is updated by rules, without any model
call, so the LLM no longer has to keep track of what is known,
which avoids tracking errors. A fine-tuned lightweight decision
model (DM), a language model trained to answer yes/no questions
with a probability [ 22], makes a few yes/no decisions: whether a
question can be answered directly, whether the QA pairs found so
far are enough, and whether retrieved passages are relevant, so that
the LLM sees more of the needed passages and fewer distracting
ones. Third, DeFT-RAG caches the QA pairs and reuses an answer
when a later question asks a sub-question with the same meaning,
which saves retrieval and LLM calls. Together, these designs raise
the accuracy of each step and of the final answer, while making the
intermediate answers reusable across questions. Our contributions
are summarized as follows:
•We propose an incremental question decomposition that
generates one self-contained sub-question at a time from
the original question and the QA pairs found so far. It re-
moves the need to plan sub-questions, decide their depen-
dencies or rewrite them, and lets the LLM focus on a single
fine-grained task: writing the sub-question for the next
missing piece of information.•We convert the remaining LLM work into fine-grained tasks
handled outside the LLM: rule-based updates to a QA table,
and yes/no decisions by a fine-tuned lightweight DM, such
as whether retrieved passages are relevant. This aims to
raise the accuracy of each step, and to leave the LLM only a
few focused, fine-grained tasks that require text generation.
•We design a QA cache that reuses the answers to earlier
sub-questions for later questions that ask a sub-question
with the same meaning, which saves retrieval and LLM
calls.
•We evaluate DeFT-RAG on three multi-hop QA bench-
marks with two LLM backbones. It achieves the best ac-
curacy among thirteen baselines while giving the LLM
fewer input tokens than any other iterative method, and
reusing answers through the QA cache further reduces LLM
calls without lowering accuracy. To support reproducibil-
ity, we release our code and prompts at https://github.com/
Dingyi-Kang/DeFT-RAG.
2 Background and Problem Formulation
This section introduces RAG and multi-hop question answering,
and the limitations of existing methods on such questions; Section 5
discusses related work in more detail.
2.1 Retrieval-Augmented Generation
Given a corpusCof passages and a question 𝑄, a vanilla RAG
system answers
𝑎=LLM 𝑄, 𝑅(𝑄,C),(1)
where𝑅retrieves the passages relevant to 𝑄fromC. Vanilla RAG
implements 𝑅as similarity search, whether by dense-embedding
cosine or sparse lexical matching (BM25), ranking passages by their

DeFT-RAG: Incrementally Decomposing Multi-Hop Questions into Self-Contained Sub-Questions with Reusable Answers
similarity to the query:
𝑅𝑘(𝑞,C)=top-k
𝑝∈C𝑠 𝑒(𝑞),𝑒(𝑝),(2)
where𝑒(·)maps a query or passage to its representation and 𝑠is
a similarity function. We refer to Eq. (2)asfixed top- 𝑘similarity
retrieval: the number of passages 𝑘is fixed in advance, and passages
are ranked only by their similarity to the query.
2.2 Multi-Hop Question Answering
A multi-hop question usually cannot be answered from a single
passage [ 9,24,29]. Answering it requires anevidence chainof two
or more facts, each stated in a passage of C, where some facts can
only be found once an earlier one is known. We call the interme-
diate answers that later facts depend onbridge entities: the chain
passes through them, but 𝑄never names them. For example, to an-
swer “Where do greyhound buses leave from in the city where the
performer of Darlings formed?”, one must first find that Darlings
is an album by Kevin Drew, then that he formed his band, Broken
Social Scene, in Toronto, and only then where Greyhound buses
leave from there (the Toronto Coach Terminal). Kevin Drew and
Toronto are the bridge entities (top of Figure 1).
Vanilla RAG (Eq. (1)) handles such questions poorly. Eq. (2)ranks
passages only by their similarity to 𝑄, so for the Darlings question
it returns passages that share its words, such as passages about
greyhound dogs, a Greyhound coach model and a person named
Darling. Answering 𝑄, however, also needs passages about Kevin
Drew and Toronto. Because 𝑄never mentions either of them, their
passages rank low, and a single retrieval rarely returns the full
evidence chain: on MuSiQue, vanilla RAG retrieves it for only 7% of
3-hop questions (Table 3). To address this, two lines of work have
been developed: graph-based RAG (§2.3) and iterative RAG (§2.4).
2.3 Graph-Based RAG
Graph-based RAG methods build a graph over Coffline, either by
using an LLM to extract the entities in every passage and the rela-
tions between them [ 4,7], or by linking passages across documents
through the entities they share [ 31]. At answering time, they locate
the entities mentioned in 𝑄in this graph and follow its links to
collect related passages, instead of ranking passages only by their
similarity to 𝑄as in Eq. (2). In this way, evidence linked to the ques-
tion’s entities can be retrieved even when it shares little wording
with𝑄.
However, graph-based RAG still falls short in covering the full
evidence chain once a question needs more than two hops: on
MuSiQue, most graph-based methods retrieve the full chain for
only 6–10% of 3-hop questions, barely above vanilla RAG (Table 3);
LinearRAG reaches 28%, but only by giving the LLM about 4,100
words of passages per question, nearly five times as many as vanilla
RAG, most of which are not part of the evidence chain. The reason
is that most graph-based methods retrieve only once per question,
starting from the entities 𝑄names, while the bridge entities are
by definition not mentioned in 𝑄, so the retrieval can reach them
only by following graph links from those entities, which rarely gets
past the first bridge entity. In Figure 1a, HippoRAG 2 learns from
the album passage that Darlings is by Kevin Drew, but retrieves nopassage about his band or Toronto, and answers from a passage
about Greyhound in Philadelphia.
2.4 Iterative RAG
Iterative RAG methods instead aim to find these bridge entities
directly: rather than preprocessing the corpus, they decompose
𝑄into sub-questions and alternate between retrieval and LLM
reasoning, so that each sub-question can name the bridge entity
found in an earlier step and reach passages that a search with 𝑄
alone would miss. As shown in Table 3, iterative methods retrieve
the full evidence chain for 19–41% of 3-hop questions, well above
most graph-based methods.
However, iterative methods often fail, because they ask the
LLM to perform several tasks, often within one call: planning sub-
questions before any of them is answered, deciding the dependency
links among them, rewriting them as answers arrive, and keeping
track of what is known. Each of these tasks leads to its own type of
error. First,planning errors: methods that plan the sub-questions in
advance, such as LogicRAG [ 3] and REAP [ 30], must guess bridge
entities before finding them, and a wrong guess or a missing step
misleads every later step. For example, asked “Who is the owner
of the record label of the guitarist who performed on the Top Gun
theme?”, LogicRAG’s plan asks for the label of the theme’s com-
poser instead of first finding the guitarist, so it answers with the
owner of the wrong label. Second,dependency errors: methods that
link the sub-questions by their dependencies, such as LogicRAG
and REAP, may link them wrongly or in the wrong order, so a
sub-question is searched before the fact it depends on is found. In
Figure 1b, LogicRAG searches for the city where “the performer”
formed before it has found the performer, retrieves passages about
bands from other cities, and never searches for the performer at all.
Third,rewriting errors: when the answer to a sub-question arrives,
methods such as Hi-Q [ 14] and REAP must rewrite the sub-question
that depends on it. The LLM may fail to do so, and the sub-question
keeps a placeholder instead of the entity’s name, so retrieval often
cannot find the passages about this bridge entity; or it may rewrite
the sub-question with the wrong entity. Fourth,tracking errors:
methods that carry their whole reasoning history in the prompt,
such as IRCoT [ 25] and Self-Ask [ 20], must find what is already
known within a long and growing input, and may miss a fact they
have already found or ask for it again. In Figure 1c, IRCoT’s input
holds all three gold passages, and its reasoning states that the band
is based in Toronto, yet it then follows a passage about Greyhound
in Philadelphia and answers with the Philadelphia terminal.
Moreover, fixed top- 𝑘retrieval (Eq. (2)), used by most RAG meth-
ods, makes these issues worse, because it lowers the answer accu-
racy of every step. With a small 𝑘, part of the needed evidence is
missed; with a large 𝑘, the evidence is buried among distracting
passages. Table 1 illustrates this on MuSiQue questions that can
be answered in one retrieval step: widening the list from the top
5 to the top 10 passages raises how often all needed passages are
present from 62% to 88%, yet accuracy barely changes (0.620 to
0.630); keeping only the gold passages from the top 10 raises it to
0.793. 2Wiki shows the same pattern. Since iterative methods an-
swer several sub-questions in sequence, and each step depends on

Dingyi Kang and Bingzhe Li
Question QWhere do greyhound buses leave from inthe city where the performer of Darlingsformed?fine-grained LLM taskDM yes/no decisionQA table / QA cacheretrievalRetrievepassagesAnswerdirectly?0.01yesRelevancefilterLLMans:answer QAnswerAnswering loop (at most Tsteps)after Tsteps: LLManson retrieved passagesLLMsub:write oneself-containedsub-questionnoSub-questionansweredbefore?RetrievepassagesmissRelevancefilterLLMans:answer thesub-questionQA tableWho is the performer ofDarlings?Kevin Drew(cached)What city did Kevin Drewform in?TorontoWhat is the bus station inToronto where greyhoundbuses leave from?TorontoCoachTerminalEnoughfacts?0.01 →0.05 →0.84LLMfin:answer Q fromthe QA tableyesAnswer:Toronto CoachTerminal ✓QA cache(sub-questions answered for other questions)Who is the performer of Darlings?Kevin DrewWhere was Kevin Drew born?Canadalook uphit: reusestoreno: write the next sub-question
Figure 2: Overview of DeFT-RAG on the question of Figure 1. The LLM performs only three fine-grained tasks (blue), and a
fine-tuned DM makes every yes/no decision (orange). The QA table, updated by rules, holds the QA pairs found for the question;
the QA cache holds those found for other questions. The first sub-question was answered for an earlier question, so its answer is
reused from the QA cache (marked “cached”) without retrieval or an LLM call. The values next toAnswer directly?andEnough
facts?are the DM’s yes probabilities from our gpt-4o-mini run.
Table 1: Accuracy on questions that can be answered in one
retrieval step, when the LLM is given the top- 𝑘retrieved
passages vs. only the gold passages among them (sampled
questions: MuSiQue 𝑛=92, 2Wiki𝑛=135; gpt-4o-mini reader).
Acc. = LLM-Acc; Gold = share of questions whose gold pas-
sages are all among the passages given to the LLM.
MuSiQue 2Wiki
Passages Acc. Gold Acc. Gold
Top-5 0.620 62% 0.733 87%
Top-10 0.630 88% 0.748 98%
Top-5 + gold only 0.663 62% 0.800 87%
Top-10 + gold only 0.793 88% 0.874 98%
the answers before it, these errors add up across steps and greatly
increase the chance that the whole process fails.
Finally, compared with vanilla RAG and graph-based RAG, itera-
tive methods spend many more retrieval and LLM calls to answer
a question (Appendix E), which are expensive in latency and cost,
yet once a question is answered, they discard the intermediate sub-
questions and answers. This is wasteful: multi-hop questions over
the same corpus often share intermediate facts, so these answers
could be reused to save retrieval and LLM calls for later questions.
3 DeFT-RAG Design
DeFT-RAG is built on three components (Figure 2): incremental
question decomposition (§3.1); fine-grained tasks outside the LLM,
namely rule-based updates to a QA table (§3.2) and yes/no decisions
made by a fine-tuned lightweight decision model (DM) (§3.3); and
reusing answers across questions through a QA cache (§3.4). §3.5
shows how they work together to answer a question. We call a task
fine-grained when it does only one thing at a time, from a short
input, such as making one yes/no decision, or generating a new
sub-question without the need to rewrite the others.3.1 Incremental Question Decomposition
In DeFT-RAG, the LLM performs only three fine-grained tasks:
𝑞𝑡+1=LLM sub(𝑄,𝐹 𝑡), 𝑎=LLM ans(𝑞,𝑃), ˆ𝑎=LLM fin(𝑄,𝐹)(3)
where𝐹𝑡={(𝑞 1,𝑎1),...,(𝑞 𝑡,𝑎𝑡)}is the QA table, the set of QA
pairs (sub-questions with their answers) found after 𝑡steps, and𝑃
is the set of passages selected for a question (§3.3). LLM subwrites
the next sub-question, LLM ansanswers a question from its selected
passages, and LLM finanswers the original question from the QA
table (blue in Figure 2).
LLM subgenerates one self-contained sub-question at a time from
the original question and the QA pairs found so far, asking for the
next missing piece of information, until the DM judges these QA
pairs enough to answer the original question (§3.3). A sub-question
isself-containedwhen it names every entity it asks about, by name
or by a description that needs no other lookup, and does not depend
on the answer to any other sub-question, so it can be searched and
answered on its own. For example, once the QA table shows that the
Rhine ends in the North Sea, “Where does the North Sea meet the
Atlantic?” is self-contained, while “Where does that sea meet the
Atlantic?” is not, because “that sea” points to an earlier step. Since
nothing is planned in advance, no bridge entity has to be guessed
before it is found, which avoids the planning errors of §2.4. Since no
sub-question depends on another, there are no dependencies to get
wrong and no sub-question to rewrite, which avoids dependency
and rewriting errors. More importantly, the LLM can focus on a
single fine-grained task at a time: writing a sub-question for the
next missing piece of information. Together, these improve the
quality of the question decomposition and help DeFT-RAG find
every bridge entity of the evidence chain: it retrieves the full chain
for more questions with three or more hops than any other system
in Table 3.

DeFT-RAG: Incrementally Decomposing Multi-Hop Questions into Self-Contained Sub-Questions with Reusable Answers
3.2 QA Table
DeFT-RAG converts the rest of the work that iterative methods leave
to the LLM into fine-grained tasks handled outside the LLM: rule-
based updates to a QA table, described here, and yes/no decisions
made by the DM (§3.3). This aims to raise the accuracy of each step,
and to leave the LLM only the three tasks in Eq. 3, which require
text generation.
Instead of tracking everything found so far within the LLM’s
context, in a long and growing prompt of passages and past reason-
ing, DeFT-RAG keeps it in a compact form, the QA table 𝐹𝑡of Eq. 3,
which starts empty, 𝐹0=∅, and grows by one QA pair at each step
that finds an answer:
𝐹𝑡+1=𝐹𝑡∪{(𝑞 𝑡+1,𝑎𝑡+1)}.(4)
The QA table is updated by rules, without any LLM or DM call: the
QA pair of each answered sub-question is added to the table; a sub-
question that gets no answer is recorded, and LLM subis shown it so
that it is not asked again; and a new sub-question that repeats one
already asked is detected by comparing the normalized wording,
andLLM subis asked for another one. Figure 2 shows the QA table
for an example question. Every answer found is thus kept explicitly
as a QA pair, rather than buried in a long prompt, which avoids
tracking errors. The QA table is the only record of what is known
passed from one step to the next: LLM subreads it to write the next
sub-question, the DM reads it to decide whether enough is known
to stop, and LLM finreads it to answer the original question. Since
each QA pair is a self-contained sub-question with a short answer,
these inputs stay short and name every entity found so far, instead
of carrying long passages or the wording of earlier steps.
3.3 Decision Tasks
The DM is a language model, usually with far fewer parameters
than the backbone LLM, trained to answer yes/no questions with a
probability [ 22] and fine-tuned for our decisions (§4.2). DeFT-RAG
makes three types of yes/no decisions with it (orange in Figure 2),
which the LLM would otherwise have to make:
•Relevance:𝐷rel(𝑞,𝑝) decides whether a retrieved passage 𝑝
is relevant to a question𝑞.
•Direct answer: 𝐷direct(𝑄,𝑅 𝑤(𝑄,C)) decides whether the top
𝑤passages retrieved for the original question, 𝑅𝑤(𝑄,C) ,
already contain all the evidence needed, so that 𝑄can be
answered in one step.
•Enough facts: 𝐷suff(𝑄,𝐹 𝑡)decides whether the QA pairs
found so far are enough to answer 𝑄, which ends the loop.
For each decision, the DM reads a short input 𝑥, such as a question
and a passage, and returns the probability that the answer is yes.
The decision is
𝐷(𝑥)=⊮
𝑝DM(yes|𝑥)≥𝜏
,(5)
with a fixed threshold 𝜏for each type. Since the DM’s probability
of yes reflects how often yes is actually right (Appendix C), we use
𝜏=0.5, “more likely yes than no”, for relevance and enough facts,
and a stricter 𝜏=0.7for direct answer, because a question that
needs several steps but is wrongly answered directly is unlikely to
get the right answer, while a question that one step could answerbut is wrongly sent to the loop can still get the right answer there,
only with more LLM calls.
Passage selection.For relevance, the DM keeps, from the top 𝑤
passages retrieved for𝑞, those relevant to𝑞:
Select 𝑞,𝑅 𝑤(𝑞,C)={𝑝∈𝑅 𝑤(𝑞,C):𝐷 rel(𝑞,𝑝)=1}.(6)
Because only the selected passages reach the LLM, 𝑤can be larger
than the𝑘of the usual top- 𝑘retrieval (Eq. (2)) without burying the
evidence among distracting passages (§2.4). All settings are listed
in Table 6.
Fine-tuning.We fine-tune the DM with LoRA [ 10], using labels
derived only from the gold annotations of each benchmark’s train-
ing split, such as which passages hold the evidence for a question
and which intermediate answers it needs. No test question is used.
Training details are given in Appendix C, and §4.5 shows that fine-
tuning makes the decisions more accurate.
3.4 Reusing Answers Across Questions
Because a self-contained sub-question names everything it asks
about, it has the same meaning, and the same answer, whichever
original question it was written for. For example, “In which city
is the headquarters of Sazerac Company located?” has the answer
“New Orleans” whether it arises from a question about Barton Pre-
mium Blend or about Blanton’s, two whiskey brands made by that
company. So a QA pair found while answering one question can
be used again by any later question that needs it. DeFT-RAG keeps
the QA pairs of all questions in a QA cache Mand checks it before
retrieving passages for a new sub-question. It first looks for a stored
sub-question with the same wording, ignoring letter case and punc-
tuation. If none is found, it takes the few stored sub-questions most
similar to the new one and uses the DM to decide whether any
of them has the same meaning, with a strict 𝜏= 0.8, because a
wrongly reused answer carries an error into another question. If
one does, its answer is reused, without retrieval or an LLM call.
Otherwise, the sub-question is answered as usual and the new QA
pair is added toM. In Figure 2, the first sub-question was answered
for an earlier question, so its answer comes from the QA cache
(marked “cached”).
3.5 Query Answering Process
Algorithm 1 and Figure 2 show how the components work together
to answer a question 𝑄. DeFT-RAG first retrieves 𝑤passages for 𝑄,
and the DM decides whether they already contain all the evidence
needed; if so, LLM ansanswers𝑄directly from the selected passages
(lines 1–3). Otherwise, DeFT-RAG starts with an empty QA table and
proceeds one step at a time: LLM subwrites the next self-contained
sub-question, the DM selects the relevant passages retrieved for it,
LLM ansanswers it, and the pair is added to the QA table (lines 5–9).
After each step, the DM checks whether the QA pairs are enough to
answer𝑄; if so, LLM finanswers𝑄from the QA table (lines 10–11).
If this does not happen within 𝑇steps, DeFT-RAG falls back to
answering𝑄directly from the relevant passages among the top2 𝑤
retrieved for it (lines 14–15).
The direct-answer decision lets questions that need only one
retrieval be answered in one step, which saves LLM calls; for the

Dingyi Kang and Bingzhe Li
Algorithm 1DeFT-RAG query answering
Require: question𝑄, corpusC, number of retrieved passages 𝑤,
step limit𝑇
1:𝑃←Select(𝑄,𝑅 𝑤(𝑄,C))
2:if𝐷 direct(𝑄,𝑅 𝑤(𝑄,C))then
3:returnLLM ans(𝑄,𝑃)
4:end if
5:𝐹←∅
6:for𝑡=1,...,𝑇do
7:𝑞←LLM sub(𝑄,𝐹)
8:𝑎←LLM ans 𝑞,Select(𝑞,𝑅 𝑤(𝑞,C))
9:if𝑎≠∅then𝐹←𝐹∪{(𝑞,𝑎)}
10:if𝐷 suff(𝑄,𝐹)then
11:returnLLM fin(𝑄,𝐹)
12:end if
13:end for
14:𝑃←Select(𝑄,𝑅 2𝑤(𝑄,C))
15:returnLLM ans(𝑄,𝑃)
other questions, the final answer is built from a short list of QA
pairs rather than from a long reasoning trace.
4 Evaluation
4.1 Setup
We evaluate on three multi-hop QA benchmarks, MuSiQue [ 24],
2WikiMultiHopQA (2Wiki) [ 9] and HotpotQA [ 29], using the 1,000-
question test set of each and its passage corpus (11,656, 6,119 and
9,811 passages), as released by the authors of LogicRAG [ 3]. We
compare against 13 baselines: answering without retrieval (No Re-
trieval) and vanilla RAG; seven graph-based methods, LightRAG [ 6],
PathRAG [ 2], HippoRAG [ 7], HippoRAG 2 [ 8], MemGraphRAG [ 27],
G3RAG [ 17] and LinearRAG [ 31]; and four iterative methods, IR-
CoT [ 25], Self-Ask [ 20], LogicRAG [ 3] and Hi-Q [ 14]. We do not
compare with REAP [ 30], whose planner and fact extractor are
LLMs fine-tuned on trajectories from HotpotQA and 2Wiki, which
does not fit our setting of one shared backbone for all systems. All
systems share the same backbone LLM, the same embedding model
(Qwen3-Embedding-4B), temperature 0 and the same output limit;
each baseline otherwise keeps its own published settings, such as
chunk size.
We use two backbones: gpt-4o-mini, through its API, and
Qwen3.6-35B-A3B, run locally. All baselines retrieve the top 𝑘=5
passages per retrieval. DeFT-RAG retrieves the top 𝑤= 2𝑘=10
passages, from which the DM keeps the relevant ones. For a fair
comparison, it keeps at most five relevant passages at each step
(2.1 to 2.4 on average), the same number as the baselines’ top-5.
DeFT-RAG takes at most 𝑇= 6steps, and all main results are
obtained without reuse (§3.4). We report exact match (EM), token
F1 and LLM-judged accuracy (LLM-Acc), our main metric, graded
with the SimpleQA prompt [ 26]. To avoid a model grading its own
answers, gpt-4o-mini runs are judged by Qwen3.6 and Qwen3.6
runs by gpt-4o-mini. Differences are tested with the exact McNemar
test on paired answers. Appendix E reports the overhead of every
method: index building, answering latency, LLM calls and tokens.4.2 Fine-Tuning the Decision Model
The DM starts from the StartLux-Decision-4B weights [ 22]
(Qwen3.5 architecture, 4B parameters), which were trained to
answer yes/no and multiple-choice questions with probabilities
that reflect how often the answer is right. We fine-tune it on data
built only from thetrainingsplits of the three benchmarks. To make
the training inputs look like the test inputs, we split the training
questions into groups of 1,000, and each group retrieves from a
corpus made of its own passages, just as each test set retrieves
from its own corpus. The labels come from the benchmarks’ gold
annotations: a passage is relevant if it is a gold supporting passage;
a question can be answered directly if all its supporting passages
are retrieved; and the QA pairs found so far are enough once they
include the answer to every step of the question’s gold decompo-
sition (MuSiQue) or gold evidence (2Wiki). No test question or
test-corpus passage appears in the training data, which has 48,036
training and 7,806 development examples (the latter from 15% of
the training questions, held out from training). We fine-tune with
LoRA [ 10] (rank 16, alpha 32, dropout 0.05, on all linear layers;
32.5M of 4.24B parameters trained), at a learning rate of10−4with
5% warm-up and cosine decay, for one epoch (2,262 steps, about
6.4 hours on two NVIDIA RTX A5000 GPUs). We then adjust the
output probabilities with a single temperature parameter fitted on
the development examples, never on test questions. Appendix C
gives the full data recipe.
4.3 Main Results
Table 2 reports EM, F1 and accuracy on all three benchmarks with
both backbones. DeFT-RAG achieves the highest score on all three
metrics in all six settings; we discuss accuracy, our main metric, be-
low. Its gains over the second-best system are largest on MuSiQue,
which has the longest evidence chains: +5.2points over Hi-Q with
gpt-4o-mini (0.629 vs. 0.577) and +5.3with Qwen3.6 (0.636 vs. 0.583),
both significant. On 2Wiki, DeFT-RAG is 2.9 points ahead of Log-
icRAG with Qwen3.6 (significant) and on par with Self-Ask with
gpt-4o-mini (0.852 vs. 0.849). On HotpotQA, it is 3.4 points ahead
of Hi-Q with gpt-4o-mini (significant) and on par with Hi-Q and
LinearRAG with Qwen3.6 (0.871 vs. 0.864 and 0.857). Against every
other baseline, the difference is significant in every setting.
Graph-based methods vary widely: with gpt-4o-mini, LightRAG,
PathRAG and HippoRAG fall below vanilla RAG, and even the
strongest of them trail DeFT-RAG by at least 10 points on MuSiQue
with either backbone, which matches their low full-chain cover-
age (§2.3). The strongest baselines are iterative methods, but the
best of them changes with the benchmark and the backbone: Hi-Q
on MuSiQue and HotpotQA, and Self-Ask or LogicRAG on 2Wiki.
DeFT-RAG is the only system that is first or on par with the first
everywhere.
4.4 Why DeFT-RAG Gives Better Answers
More of the evidence chain, in fewer words.Table 3 measures, for
each MuSiQue question, whether the LLM received all passages
needed to answer it, as annotated in the dataset, and how many
words of passages it received. DeFT-RAG gives the LLM the full
chain for 46% of 3-hop and 18% of 4-hop questions, the highest
3-hop coverage of all systems, with 390 words per question, the

DeFT-RAG: Incrementally Decomposing Multi-Hop Questions into Self-Contained Sub-Questions with Reusable Answers
Table 2: Results on 1,000 questions per benchmark with two backbones: exact match (EM), token F1, and LLM-judged accuracy
(Acc). gpt-4o-mini runs are judged by Qwen3.6 and Qwen3.6 runs by gpt-4o-mini, so no model grades its own answers. EM
and F1 are computed from the answers as given and understate systems that answer in full sentences, such as LightRAG with
gpt-4o-mini; Acc is our main metric. Avg: mean over the three benchmarks. Best in bold, second best underlined.
SystemMuSiQue 2Wiki HotpotQA Avg
EM F1 Acc EM F1 Acc EM F1 Acc EM F1 Acc
gpt-4o-mini backbone
No Retrieval 0.122 0.226 0.239 0.273 0.351 0.410 0.268 0.397 0.527 0.221 0.325 0.392
Vanilla RAG 0.238 0.347 0.412 0.459 0.553 0.659 0.468 0.619 0.799 0.388 0.506 0.623
LightRAG 0.000 0.087 0.305 0.000 0.150 0.505 0.000 0.190 0.696 0.000 0.142 0.502
PathRAG 0.123 0.233 0.276 0.278 0.379 0.451 0.397 0.549 0.693 0.266 0.387 0.473
HippoRAG 0.189 0.286 0.323 0.461 0.550 0.650 0.389 0.515 0.676 0.346 0.450 0.550
HippoRAG 2 0.303 0.428 0.478 0.558 0.639 0.713 0.546 0.691 0.813 0.469 0.586 0.668
MemGraphRAG 0.275 0.398 0.453 0.524 0.576 0.635 0.514 0.665 0.795 0.438 0.546 0.628
G3RAG 0.327 0.450 0.503 0.542 0.611 0.682 0.563 0.711 0.830 0.477 0.590 0.672
LinearRAG 0.243 0.375 0.439 0.450 0.611 0.786 0.439 0.625 0.822 0.377 0.537 0.682
LogicRAG 0.333 0.459 0.495 0.586 0.694 0.772 0.559 0.688 0.798 0.493 0.614 0.688
IRCoT 0.348 0.465 0.510 0.619 0.701 0.767 0.595 0.721 0.823 0.521 0.629 0.700
Self-Ask 0.339 0.455 0.500 0.589 0.695 0.849 0.451 0.596 0.783 0.460 0.582 0.711
Hi-Q 0.414 0.549 0.577 0.631 0.720 0.812 0.600 0.734 0.840 0.548 0.667 0.743
DeFT-RAG (ours) 0.453 0.587 0.629 0.678 0.767 0.852 0.627 0.775 0.874 0.586 0.709 0.785
Qwen3.6-35B-A3B backbone
No Retrieval 0.128 0.216 0.198 0.366 0.417 0.428 0.293 0.392 0.427 0.262 0.341 0.351
Vanilla RAG 0.297 0.373 0.397 0.553 0.618 0.670 0.604 0.726 0.809 0.485 0.572 0.625
LightRAG 0.270 0.385 0.459 0.367 0.493 0.688 0.400 0.598 0.816 0.346 0.492 0.654
PathRAG 0.333 0.439 0.444 0.524 0.607 0.667 0.599 0.727 0.813 0.485 0.591 0.641
HippoRAG 0.187 0.247 0.257 0.535 0.604 0.676 0.453 0.554 0.621 0.392 0.468 0.518
HippoRAG 2 0.346 0.439 0.471 0.625 0.682 0.744 0.604 0.727 0.818 0.525 0.616 0.678
MemGraphRAG 0.265 0.340 0.388 0.537 0.576 0.621 0.542 0.654 0.740 0.448 0.524 0.583
G3RAG 0.411 0.501 0.534 0.622 0.679 0.725 0.633 0.752 0.839 0.555 0.644 0.699
LinearRAG 0.353 0.437 0.475 0.671 0.758 0.830 0.606 0.746 0.857 0.543 0.647 0.721
LogicRAG 0.391 0.503 0.529 0.696 0.779 0.838 0.620 0.745 0.830 0.569 0.675 0.732
IRCoT 0.383 0.482 0.486 0.678 0.747 0.799 0.629 0.761 0.843 0.563 0.663 0.709
Self-Ask 0.423 0.526 0.546 0.677 0.762 0.825 0.581 0.708 0.795 0.560 0.665 0.722
Hi-Q 0.466 0.574 0.583 0.600 0.681 0.783 0.652 0.776 0.864 0.573 0.677 0.743
DeFT-RAG (ours) 0.494 0.615 0.636 0.716 0.801 0.867 0.658 0.799 0.871 0.623 0.738 0.791
fewest of any system and less than half of vanilla RAG’s 870. The
systems that cover more chains on some hop count use far more
words: Self-Ask covers more 2-hop chains (83%) with 2,590 words,
and LinearRAG covers 19% of 4-hop chains with 4,110 words, about
ten times as many as DeFT-RAG. These extra words bury the needed
passages among many that are not needed, which lowers accuracy
(§2.4, Table 1), and they add to the LLM’s input cost. Asking one self-
contained sub-question at a time and keeping only the passages the
DM judges relevant gives the LLM a short context that still contains
the needed evidence.
Effect of passage selection.To isolate the effect of selection, we
turn it off and give the LLM the top-5 retrieved passages, as the
baselines do, keeping everything else the same (Table 4). Without
selection, accuracy drops on all three benchmarks, by 2.3, 3.7 and 5.3
points on MuSiQue, 2Wiki and HotpotQA; the drop on HotpotQAis significant ( 𝑝=0.002). On 2Wiki and HotpotQA, most of the loss
comes from questions answered directly, where the LLM must find
the evidence for the whole question in one list of passages. This
agrees with Table 1: on questions that can be answered in one re-
trieval step, a longer list of passages alone barely helps, but keeping
only the needed passages from it raises accuracy substantially.
4.5 Effect of Fine-Tuning the Decision Model
Table 5 compares the decisions of the base DM (StartLux-Decision-
4B before fine-tuning) and the fine-tuned DM against gold labels.
Fine-tuning improves the direct-answer decision the most, for ex-
ample from 0.850 to 0.933 on MuSiQue, and also improves relevance,
most clearly on MuSiQue (0.931 to 0.979), while the enough-facts
decision changes little. End to end, on the first 300 MuSiQue ques-
tions with gpt-4o-mini, the fine-tuned DM raises accuracy from
0.587 to 0.623.

Dingyi Kang and Bingzhe Li
Table 3: Evidence coverage on MuSiQue (gpt-4o-mini), for
questions that need 2, 3 or 4 hops: the share of questions for
which the LLM received all passages needed to answer them,
as annotated in the dataset. Words: the average number of
words of passages the LLM received per question.
Method 2-hop 3-hop 4-hop Words
Vanilla RAG 46% 7% 1% 870
Graph-based
PathRAG 42% 10% 8% 12,390
HippoRAG 33% 7% 0% 520
HippoRAG 2 53% 8% 0% 500
MemGraphRAG 41% 7% 1% 500
G3RAG 48% 6% 0% 470
LinearRAG 61% 28% 19% 4,110
Iterative
IRCoT 76% 37% 7% 780
Self-Ask 83% 41% 14% 2,590
LogicRAG 59% 19% 2% 750
Hi-Q 72% 28% 4% 920
DeFT-RAG76% 46% 18% 390
Table 4: Effect of selecting relevant passages: DeFT-RAG ver-
sus the same process without selection (the top-5 retrieved
passages, all kept). gpt-4o-mini backbone, 300 test questions
per benchmark (a subset of those in Table 2); Avg is the mean
over the three benchmarks.
MuSiQue 2Wiki HotpotQA Avg
DeFT-RAG (top-10, selected) 0.633 0.860 0.880 0.791
No selection (top-5) 0.610 0.823 0.827 0.753
Difference +2.3 +3.7 +5.3 +3.8
Table 5: Effect of fine-tuning on decision quality, base DM /
fine-tuned DM, on 300 test questions per benchmark. AUC
measures how well the DM’s yes probabilities separate yes
cases from no cases (1.0 is perfect). HotpotQA has no answer
for each step, so its enough-facts label is yes when the pas-
sages used include all gold supporting passages (167 cases).
Decision MuSiQue 2Wiki HotpotQA
Relevance 0.931 /0.9790.996 /1.0000.990 /0.997
Direct answer 0.850 /0.9330.938 /0.9970.915 /0.995
Enough facts 0.903 /0.9040.943 /0.9590.816 /0.845
4.6 Reusing Answers
We run DeFT-RAG with reuse (§3.4) on all 1,000 MuSiQue questions
with gpt-4o-mini, answering them in order with one QA cache.
Of all sub-questions generated over the 1,000 questions, 23.1% are
answered from the cache, which reduces the number of LLM calls by
12.9%. Accuracy does not drop (0.642 with reuse vs. 0.629 without,
𝑝= 0.18), so answers found for earlier questions can be safely
reused for later ones.5 Related Work
Graph-based RAG.Graph-based methods build a graph over the
corpus offline and retrieve along it at answering time. GraphRAG [ 4]
and LightRAG [ 6] extract entities and relations with an LLM and
retrieve community summaries or entities, PathRAG [ 2] retrieves
relational paths, HippoRAG [ 7] and HippoRAG 2 [ 8] rank pas-
sages with Personalized PageRank from the question’s entities,
MemGraphRAG [ 27] builds a layered graph of schemas, facts and
passages, and LinearRAG [ 31] and G3RAG [ 17] build their graphs
without LLM extraction. All of these methods retrieve once per ques-
tion, starting from what the question names, which rarely recovers
the full evidence chain beyond two hops (§2.3). DeFT-RAG instead
finds the bridge entities at answering time, one self-contained sub-
question at a time, and retrieves the full chain for more questions
with three or more hops than any of these methods (Table 3).
Iterative RAG.IRCoT [ 25] uses each new chain-of-thought sen-
tence as the next query, and Self-Ask [ 20] poses follow-up ques-
tions within a single growing prompt. LogicRAG [ 3] and REAP [ 30]
plan the sub-questions with their dependencies in advance, and
Hi-Q [ 14] and REAP rewrite dependent sub-questions as answers
arrive. In these methods, the LLM itself takes on several tasks, often
within the same call: planning or writing the sub-questions, decid-
ing their dependencies, rewriting them, keeping track of what is
known, and deciding when to stop. DeFT-RAG instead lets the LLM
focus on a single fine-grained task at a time: generating one self-
contained sub-question for the next missing piece of information,
which removes planning, dependencies and rewriting, and avoids
their associated errors; tracking is handled by the QA table, and the
yes/no decisions by the fine-tuned DM. Its LLM input stays short,
1,569 tokens per question against 6,140 for Self-Ask (Appendix E),
and it achieves the highest accuracy of these methods in every
setting (Table 2).
Models for retrieval decisions.Rerankers [ 18,23] re-score re-
trieved passages but still keep a fixed number, and Adaptive-
RAG [ 12] uses a trained classifier to choose between no, single and
iterative retrieval; Self-RAG [ 1], Search-R1 [ 13], R1-Searcher [ 21]
and Search-o1 [ 16] instead train the backbone LLM, or use a reason-
ing model, to make such decisions. DeFT-RAG uses one fine-tuned
DM for the yes/no decisions at every step while the backbone stays
unchanged, so any LLM can serve as the backbone, and fine-tuning
the DM is much cheaper than training the backbone LLM.
6 Conclusion
We presented DeFT-RAG, a RAG framework for multi-hop ques-
tion answering built on three components: incremental decompo-
sition into self-contained sub-questions, which avoids planning,
dependency and rewriting errors; fine-grained tasks handled out-
side the LLM, namely rule-based updates to a QA table and yes/no
decisions by a fine-tuned lightweight DM; and a QA cache that
reuses answers across questions. Across three benchmarks and
two backbones, DeFT-RAG achieves the best accuracy among thir-
teen baselines, with its largest gains on MuSiQue, while giving the
LLM fewer input tokens than any other iterative method; reusing
answers reduces LLM calls without lowering accuracy.

DeFT-RAG: Incrementally Decomposing Multi-Hop Questions into Self-Contained Sub-Questions with Reusable Answers
References
[1] Akari Asai, Zeqiu Wu, Yizhong Wang, Avirup Sil, and Hannaneh Hajishirzi. 2024.
Self-RAG: Learning to Retrieve, Generate, and Critique through Self-Reflection.
InInternational Conference on Learning Representations (ICLR).
[2]Boyu Chen, Zirui Guo, Zidan Yang, Yuluo Chen, Junze Chen, Zhenghao Liu,
Chuan Shi, and Cheng Yang. 2026. PathRAG: Pruning Graph-based Retrieval Aug-
mented Generation with Relational Paths. InProceedings of the AAAI Conference
on Artificial Intelligence, Vol. 40. 30183–30191.
[3] Shengyuan Chen, Chuang Zhou, Zheng Yuan, Qinggang Zhang, Zeyang Cui, Hao
Chen, Yilin Xiao, Jiannong Cao, and Xiao Huang. 2026. You Don’t Need Pre-built
Graphs for RAG: Retrieval Augmented Generation with Adaptive Reasoning
Structures. InProceedings of the AAAI Conference on Artificial Intelligence, Vol. 40.
30270–30278.
[4]Darren Edge, Ha Trinh, Newman Cheng, Joshua Bradley, Alex Chao, Apurva
Mody, Steven Truitt, Dasha Metropolitansky, Robert Osazuwa Ness, and Jonathan
Larson. 2024. From Local to Global: A Graph RAG Approach to Query-Focused
Summarization.arXiv preprint arXiv:2404.16130(2024).
[5] Yunfan Gao, Yun Xiong, Xinyu Gao, Kangxiang Jia, Jinliu Pan, Yuxi Bi, Yi Dai,
Jiawei Sun, Meng Wang, and Haofen Wang. 2023. Retrieval-Augmented Gen-
eration for Large Language Models: A Survey.arXiv preprint arXiv:2312.10997
(2023).
[6] Zirui Guo, Lianghao Xia, Yanhua Yu, Tu Ao, and Chao Huang. 2025. LightRAG:
Simple and Fast Retrieval-Augmented Generation. InFindings of the Association
for Computational Linguistics: EMNLP. 10746–10761.
[7] Bernal Jiménez Gutiérrez, Yiheng Shu, Yu Gu, Michihiro Yasunaga, and Yu Su.
2024. HippoRAG: Neurobiologically Inspired Long-Term Memory for Large Lan-
guage Models. InAdvances in Neural Information Processing Systems (NeurIPS).
[8] Bernal Jiménez Gutiérrez, Yiheng Shu, Weijian Qi, Sizhe Zhou, and Yu Su. 2025.
From RAG to Memory: Non-Parametric Continual Learning for Large Language
Models. InProceedings of the International Conference on Machine Learning (ICML)
(PMLR, Vol. 267). 21497–21515.
[9] Xanh Ho, Anh-Khoa Duong Nguyen, Saku Sugawara, and Akiko Aizawa. 2020.
Constructing A Multi-hop QA Dataset for Comprehensive Evaluation of Rea-
soning Steps. InProceedings of the International Conference on Computational
Linguistics (COLING).
[10] Edward J. Hu, Yelong Shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean
Wang, Lu Wang, and Weizhu Chen. 2022. LoRA: Low-Rank Adaptation of Large
Language Models. InInternational Conference on Learning Representations (ICLR).
[11] Lei Huang, Weijiang Yu, Weitao Ma, Weihong Zhong, Zhangyin Feng, Haotian
Wang, Qianglong Chen, Weihua Peng, Xiaocheng Feng, Bing Qin, and Ting
Liu. 2025. A Survey on Hallucination in Large Language Models: Principles,
Taxonomy, Challenges, and Open Questions.ACM Transactions on Information
Systems(2025).
[12] Soyeong Jeong, Jinheon Baek, Sukmin Cho, Sung Ju Hwang, and Jong C. Park.
2024. Adaptive-RAG: Learning to Adapt Retrieval-Augmented Large Lan-
guage Models through Question Complexity. InProceedings of the Conference
of the North American Chapter of the Association for Computational Linguistics
(NAACL).
[13] Bowen Jin, Hansi Zeng, Zhenrui Yue, Jinsung Yoon, Sercan Ö. Arık, Dong Wang,
Hamed Zamani, and Jiawei Han. 2025. Search-R1: Training LLMs to Reason
and Leverage Search Engines with Reinforcement Learning. InConference on
Language Modeling (COLM).
[14] Jueun Kim, Sungho Park, and Wook-Shin Han. 2026. Hi-Q: Hierarchical Evidence-
guided Query Refinement for Multi-Hop Question Answering.arXiv preprint
arXiv:2608.30468(2026).
[15] Patrick Lewis, Ethan Perez, Aleksandra Piktus, Fabio Petroni, Vladimir
Karpukhin, Naman Goyal, Heinrich Küttler, Mike Lewis, Wen tau Yih, Tim
Rocktäschel, Sebastian Riedel, and Douwe Kiela. 2020. Retrieval-Augmented
Generation for Knowledge-Intensive NLP Tasks. InAdvances in Neural Informa-
tion Processing Systems (NeurIPS).
[16] Xiaoxi Li, Guanting Dong, Jiajie Jin, Yuyao Zhang, Yujia Zhou, Yutao Zhu, Peitian
Zhang, and Zhicheng Dou. 2025. Search-o1: Agentic Search-Enhanced Large
Reasoning Models. InProceedings of the Conference on Empirical Methods in
Natural Language Processing (EMNLP). 5420–5438.
[17] Zeliang Li, Xiaofen Xing, Kailing Guo, and Xiangmin Xu. 2026. Beyond Similarity
through Zero-Token Geometric Graphs for Multi-Hop RAG.arXiv preprint
arXiv:2609.19622(2026).
[18] Rodrigo Nogueira, Zhiying Jiang, Ronak Pradeep, and Jimmy Lin. 2020. Docu-
ment Ranking with a Pretrained Sequence-to-Sequence Model. InFindings of the
Association for Computational Linguistics: EMNLP. 708–718.
[19] OpenAI. 2023. GPT-4 Technical Report.arXiv preprint arXiv:2303.08774(2023).
[20] Ofir Press, Muru Zhang, Sewon Min, Ludwig Schmidt, Noah A. Smith, and Mike
Lewis. 2023. Measuring and Narrowing the Compositionality Gap in Language
Models. InFindings of the Association for Computational Linguistics: EMNLP.
[21] Huatong Song, Jinhao Jiang, Yingqian Min, Jie Chen, Zhipeng Chen, Wayne Xin
Zhao, Lei Fang, and Ji-Rong Wen. 2025. R1-Searcher: Incentivizing the Search
Capability in LLMs via Reinforcement Learning.arXiv preprint arXiv:2503.05592(2025).
[22] StartLux Labs. 2026. StartLux-Decision-4B. https://huggingface.co/startlux-
models/StartLux-Decision-4B. Model card.
[23] Weiwei Sun, Lingyong Yan, Xinyu Ma, Shuaiqiang Wang, Pengjie Ren, Zhumin
Chen, Dawei Yin, and Zhaochun Ren. 2023. Is ChatGPT Good at Search? In-
vestigating Large Language Models as Re-Ranking Agents. InProceedings of
the Conference on Empirical Methods in Natural Language Processing (EMNLP).
14918–14937.
[24] Harsh Trivedi, Niranjan Balasubramanian, Tushar Khot, and Ashish Sabharwal.
2022. MuSiQue: Multihop Questions via Single-hop Question Composition.
Transactions of the Association for Computational Linguistics(2022).
[25] Harsh Trivedi, Niranjan Balasubramanian, Tushar Khot, and Ashish Sabharwal.
2023. Interleaving Retrieval with Chain-of-Thought Reasoning for Knowledge-
Intensive Multi-Step Questions. InProceedings of the Annual Meeting of the
Association for Computational Linguistics (ACL).
[26] Jason Wei, Nguyen Karina, Hyung Won Chung, Yunxin Joy Jiao, Spencer Papay,
Amelia Glaese, John Schulman, and William Fedus. 2024. Measuring Short-Form
Factuality in Large Language Models.arXiv preprint arXiv:2411.04368(2024).
[27] Chuanjie Wu, Zhishang Xiang, Yunbo Tang, Zerui Chen, Qinggang Zhang, and
Jinsong Su. 2026. MemGraphRAG: Memory-based Multi-Agent System for Graph
Retrieval-Augmented Generation. InProceedings of the ACM SIGKDD Conference
on Knowledge Discovery and Data Mining (KDD).
[28] An Yang, Anfeng Li, Baosong Yang, Beichen Zhang, Binyuan Hui, Bo Zheng,
Bowen Yu, Chang Gao, Chengen Huang, Chenxu Lv, Chujie Zheng, Dayiheng
Liu, Fan Zhou, Fei Huang, Feng Hu, Hao Ge, Haoran Wei, Huan Lin, Jialong
Tang, Jian Yang, Jianhong Tu, Jianwei Zhang, Jianxin Yang, Jiaxi Yang, Jing Zhou,
Jingren Zhou, Junyang Lin, Kai Dang, Keqin Bao, Kexin Yang, Le Yu, Lianghao
Deng, Mei Li, Mingfeng Xue, Mingze Li, Pei Zhang, Peng Wang, Qin Zhu, Rui
Men, Ruize Gao, Shixuan Liu, Shuang Luo, Tianhao Li, Tianyi Tang, Wenbiao Yin,
Xingzhang Ren, Xinyu Wang, Xinyu Zhang, Xuancheng Ren, Yang Fan, Yang Su,
Yichang Zhang, Yinger Zhang, Yu Wan, Yuqiong Liu, Zekun Wang, Zeyu Cui,
Zhenru Zhang, Zhipeng Zhou, and Zihan Qiu. 2025. Qwen3 Technical Report.
arXiv preprint arXiv:2505.09388(2025).
[29] Zhilin Yang, Peng Qi, Saizheng Zhang, Yoshua Bengio, William W. Cohen, Rus-
lan Salakhutdinov, and Christopher D. Manning. 2018. HotpotQA: A Dataset
for Diverse, Explainable Multi-hop Question Answering. InProceedings of the
Conference on Empirical Methods in Natural Language Processing (EMNLP).
[30] Yijie Zhu, Haojie Zhou, Wanting Hong, Tailin Liu, and Ning Wang. 2026. REAP:
Enhancing RAG with Recursive Evaluation and Adaptive Planning for Multi-
Hop Question Answering. InProceedings of the AAAI Conference on Artificial
Intelligence, Vol. 40. 35230–35238.
[31] Luyao Zhuang, Shengyuan Chen, Yilin Xiao, Huachi Zhou, Yujing Zhang, Hao
Chen, Qinggang Zhang, and Xiao Huang. 2026. LinearRAG: Linear Graph Re-
trieval Augmented Generation on Large-scale Corpora. InInternational Confer-
ence on Learning Representations (ICLR).
A Implementation Details
Table 6 lists the settings used for all results. Beyond Algorithm 1,
the implementation retries once on an empty or repeated reply,
with a changed input, and uses the following rules.
Additional rules.A question is answered directly only when the
DM judges its passages sufficient ( 𝜏=0.7) and it does not ask for
a composite family relation such as a grandparent or an in-law,
which is detected by keywords. All results use the same rules on
every benchmark and backbone.
How the thresholds were chosen.The thresholds of 0.5 are the
natural cut-off for yes/no probabilities that reflect how often yes
is right, and were not tuned. The direct-answer threshold of 0.7
was chosen during development on stored answers of part of the
test questions (2Wiki and the first 300 MuSiQue questions), and
then kept fixed for all benchmarks and both backbones. The result
depends little on this choice: moving the threshold to 0.6 or 0.8
changes the path of at most 13 of the 1,000 questions per benchmark
and accuracy by at most 0.1 points on every benchmark (gpt-4o-
mini). The reuse threshold of 0.8 was set strict by hand, because a
wrongly reused answer carries an error into another question.

Dingyi Kang and Bingzhe Li
Table 6: Settings of DeFT-RAG used for all results.
Setting Value
Retrieved passages𝑤(retry and fallback) 10 (20)
Step limit𝑇6
Relevance: keep if𝑝(yes)≥0.5
Passages kept when only one / no passage is judged relevant 2 / 5
Passages kept at most 5
Direct answer: sufficient≥0.7
Enough facts:𝑝(yes)≥0.5
Reuse: stored sub-questions compared 5
Reuse: same meaning𝑝(yes)≥0.8
Passage length cap (characters) 3,000
B Prompts
The exact prompts of the three LLM tasks and the yes/no questions
asked to the DM. Text in braces is filled in at run time. The prompts
call the QA pairs “facts”.
Writing a sub-question, no QA pairs yet (LLM sub). A question may refer to
something by description instead of by name, such as "the author of Y". Some
descriptions are built on others: "the birthplace of the author of Y" needs "the
author of Y" first.
Write ONE short question that finds a piece of information needed to answer the
question below. Ask about a description that is not built on another one, so your
question can be searched on its own. If the question has no such description and
needs no calculation, ask the question itself.
Example
Question: What language is spoken in the birthplace of the author of Y?
Next question: Who is the author of Y?
Once the people or things the question describes are known by name, ask for a
fact a document states directly, of the kind the question asks about: to find out
who is older, ask for each person's date of birth, not how old they are; to find
out which country someone is from, ask for their nationality, not their place of
birth.
Question: {question}
Reply with only the next question.
Writing a sub-question, with QA pairs (LLM sub). Below is a question, and the
facts found so far while working on it.
A question may refer to something by description instead of by name, such as "the
author of Y". Some descriptions are built on others: "the birthplace of the
author of Y" needs "the author of Y" first.
Write ONE short question that finds the next missing piece of information needed
to answer the main question. Use the facts found so far: write names in place of
descriptions the facts already resolve. Ask about a description that is not built
on another unresolved one, so your question can be searched on its own. Do not
ask for anything already in the facts.
Example
Question: What language is spoken in the birthplace of the author of Y?
Facts found so far:
1. Who is the author of Y? Answer: Jane Doe
Next question: Where was Jane Doe born?
Once the people or things the question describes are known by name, ask for a
fact a document states directly, of the kind the question asks about: to find out
who is older, ask for each person's date of birth, not how old they are; to find
out which country someone is from, ask for their nationality, not their place of
birth.
Question: {question}
Facts found so far:
{facts}
Reply with only the next question.
Added after a sub-question got no answer. These questions could not be answered
from the documents:
{failed}
Do not ask any of these again. Ask for the needed information in different words,
or work on another part of the main question.
Added after a repeated sub-question. The question "{sq}" was already asked.
Write a different question.Answering from passages (LLM ans). Answer the question using ONLY the passages
below. Give a short, direct answer -- a few words, not a sentence.
Passages:
{passages}
Question: {question}
Answer:
Answering the original question from the QA table (LLM fin). Answer the question
using ONLY the facts below. Give a short, direct answer -- a few words, not a
sentence.
When the question compares dates, compare the years first, then the months, then
the days. When it compares ages, the person born earlier is older.
Facts:
{facts}
Question: {question}
Answer:
Decision: relevance of passage𝑖. Is`p_i`needed to answer`question`: does it
state part of the answer, or a linking fact the reasoning must pass through
(e.g., identifying an entity the question refers to only by description)?
Yes:`p_i`states part of the answer, or a linking fact needed to reach it.
No:`p_i`is not needed to answer`question`: it is unrelated or only loosely
related.
Decision: direct answer. Do the passages`p1`through`p10`together contain all
the facts needed to answer`question`-- whether stated outright or derivable by
combining facts across them?
Yes: Every fact needed is present across the passages; the answer is stated or
derivable by connecting them.
No: At least one fact needed to answer is missing and would have to be found
elsewhere.
Decision: enough facts. Has`question`already been answered by the facts in
`facts`: does every entity or value the question asks for, or depends on, appear
among the facts'answers?
Yes: The facts already give the answer: each step the question depends on is
answered in`facts`.
No: Some step the question depends on has not been answered yet, so`question`
cannot be answered from`facts`alone.
Decision: same meaning (reuse). Are`query`and`c_i`the same question: asking
for the same piece of information about the same explicitly named entity, so that
one correct answer is correct for both?
Yes: Both name the same entity explicitly and ask for the same information; only
the wording differs.
No: They differ in entity, property, time, place or scope -- OR either one refers
to something without naming it (such as "the tournament" or "the author"), so it
is unclear whether they mean the same thing.
C Decision Model Fine-Tuning Details
Training questions and corpora.For each benchmark, we sample
3,000 questions from its training split, following the mix of hop
counts or question types of the test set, and split them into groups
of 1,000. Each group’s corpus consists of the passages of its own
questions (MuSiQue about 13,000, 2Wiki about 5,200, HotpotQA
about 9,600 passages per group, close to the test corpora of 11,656,
6,119 and 9,811). Every training request retrieves its top 10 passages
from its group’s corpus with the embedder and query instruction
of DeFT-RAG.
Labels.All labels come from the training splits’ gold annotations.
Relevance: a passage is relevant if it is a gold supporting passage;
each request labels all gold passages and three randomly chosen
other passages, while all ten passages stay in the request.Direct
answer: yes if all gold supporting passages are among the top 10.
Enough facts: built from the gold decompositions (MuSiQue) and
gold evidence (2Wiki): the answers to only the first steps give “no”,
the answers to all steps give “yes”, a wrong answer to a step gives
“no”, and the answers to all steps plus an unrelated QA pair give
“yes”. We add enough-facts items taken from runs of DeFT-RAG on
training questions, labelled by the same rule, and items whose QA
pairs use self-contained sub-questions, as DeFT-RAG writes them.
The same-meaning decision is not fine-tuned.

DeFT-RAG: Incrementally Decomposing Multi-Hop Questions into Self-Contained Sub-Questions with Reusable Answers
Leakage checks.All 1,000 test questions of each benchmark come
from its official validation split; none of them is in the training
data, and no training question text is identical to a test question.
Training questions whose gold passage appears in any test corpus
are skipped, so that no test-corpus passage is ever shown in training.
Training and probability adjustment.The final data has 48,036
training and 7,806 development examples. LoRA settings are given
in §4.2; training used micro-batches of 8,192 tokens, gradient accu-
mulation to an effective batch equal to the base model’s single-GPU
default, gradient checkpointing and gradient clipping at 1.0. A sin-
gle temperature for yes/no probabilities is fitted on the development
examples (1.08).
D Experimental Details for Baselines
All systems use the same backbone, the same embedding model
(Qwen3-Embedding-4B, last-token pooling, with the query instruc-
tion applied to queries only), temperature 0, top-𝑝= 1and the
same output limit of 3,072 tokens; the reasoning mode of Qwen3.6
is turned off for every system. Each baseline keeps the settings of
its released code or paper that define the method, such as chunk
size and graph construction, and all baselines retrieve the top 𝑘=5
passages per retrieval (§4.1). Systems that build the same kind of
index share it: HippoRAG and MemGraphRAG reuse the index built
by HippoRAG 2 (for MemGraphRAG, only the passage embeddings,
after we checked that the passage sets are identical), and PathRAG
reuses LightRAG’s graph. Every system’s answers are graded by
the same script with the same judge prompt, and missing answers
count as wrong.
E Overhead Analysis
Table 7 reports, for every method with gpt-4o-mini, the cost of
building the index and the cost of answering each question. Its last
row replaces the local DM of DeFT-RAG with Jev, a closed decision
model accessed through an API.
Index.DeFT-RAG, vanilla RAG and the iterative methods only
embed the passages, which takes about five minutes per corpus
and no LLM calls. The graph-based methods that extract a graph
with the LLM need from about 2,100 (LightRAG) to 147,000 (Mem-
GraphRAG) LLM calls and 11 to 114 million tokens per corpus, and
up to 18 hours of build time; this cost is paid again when the corpus
changes.
Answering.DeFT-RAG gives the LLM fewer input tokens per
question than every other iterative method (1,569, against 1,926 to
13,928), and it produces the fewest output tokens of all methods (33),
because each task asks for a short output; it also makes fewer LLM
calls than Self-Ask and Hi-Q (4.8, against 6.3 and 6.6). Its latency is
the highest (median 9.9 s).
Where the latency comes from.Most of the answering time of
DeFT-RAG goes to its frequent calls to the local DM, which se-
lects the relevant passages at every step and makes the other de-
cisions. Per question, DeFT-RAG makes 7.0, 6.3 and 3.2 DM calls
on MuSiQue, 2Wiki and HotpotQA, at 1.6 to 1.8 s each, as the DM
runs on one GPU, one request at a time, over inputs of up to ten
passages. These calls take 12.5 of the 20.4 s mean answering timeon MuSiQue, 10.3 of 16.7 s on 2Wiki and 5.2 of 7.8 s on HotpotQA,
about 60 to 70%. With Jev in place of the DM, at about 0.16 s per call,
the median latency falls from 9.9 to 5.0 s, and the mean answering
time to 8.2, 7.7 and 3.6 s on the three benchmarks, while accuracy
stays within 1.2 points of the DM’s (0.618, 0.857 and 0.867, against
0.629, 0.852 and 0.874). Most of the added latency thus comes from
how the DM is served, not from the method itself.
F Case Studies
All cases are taken from the runs in Table 2 (gpt-4o-mini backbone);
each line shows a sub-question and the answer DeFT-RAG found
for it.
Where it works: a three-hop chain. “When did the city where the
headquarters of Barton Premium Blend’s manufacturer is located
elect its first black mayor?”(gold: 1970s). DeFT-RAG asks one self-
contained sub-question at a time:
What is the name of the manufacturer of Barton Premium
Blend?→Sazerac Company
In which city is the headquarters of Sazerac Company
located?→New Orleans
When did New Orleans elect its first black mayor? →
1970s
After the third QA pair, the DM judges the QA pairs enough and
LLM finanswers “1970s”. Self-Ask’s first follow-up skips the manu-
facturer (“What city is the headquarters of Barton Premium Blend’s
manufacturer located in?”), takes Bardstown, Kentucky as the an-
swer, and finally answers “Information not available”; Hi-Q also
gives no answer, and LogicRAG answers “No”.
Where it works, but not in order.For “Who sent naval ships to
the body of water that joins the Atlantic and the sea where the
Rhine ends?” (gold: Caesar), the question behind the example in
§3.1, the DM first judges the retrieved passages insufficient (0.15),
and DeFT-RAG asks:
Who sent naval ships to the body of water that joins the
Atlantic?→Caesar
What is the name of the sea where the Rhine ends? →
North Sea
What is the name of the body of water that joins the
Atlantic and the North Sea?→English Channel
The first sub-question skips ahead to the last hop with a vague
description and happens to retrieve the right answer. The DM does
not stop there (enough facts: 0.44), but only after the chain through
the North Sea and the English Channel is complete (0.70); the final
answer is “Caesar”.
Where it fails: nested sub-questions.Bridge-comparison questions
in 2Wiki compare two films through their directors.“Which film
has the director who is older than the other, Airheads or Return To
Cabin By The Lake?”(gold: Return To Cabin By The Lake):
Who is the director of Airheads?→Michael Lehmann
What is the date of birth of Michael Lehmann? →March
30, 1957
What is the date of birth of the director of Return To Cabin
By The Lake?→(no answer)
What is the name of the director of Return To Cabin By
The Lake?→Po-Chih Leong

Dingyi Kang and Bingzhe Li
Table 7: Overhead of every method with gpt-4o-mini, mean over the three benchmarks (1,000 questions each). Index: LLM
calls, LLM tokens (input + output) and build time, per corpus. Answering: per question, from question to answer (grading
excluded). Methods with an index of passage embeddings only use no LLM to build it; their build time is that of embedding the
corpus with Qwen3-Embedding-4B.𝑎Uses LightRAG’s graph.𝑏Uses HippoRAG 2’s graph.𝑐Includes the extraction it shares
with HippoRAG 2.𝑑Build time not recorded; no LLM is used.𝑒Mean latency (only totals are recorded). Latency is the median
over questions, because a few API stalls of several minutes dominate the mean.𝑓DeFT-RAG with the closed decision model Jev,
accessed through an API, in place of the local DM; its time per call is set to its typical 0.16 s.
Index (per corpus) Answering (per question)
Method LLM calls Tokens (M) Time (h) Latency (s) LLM calls Input tokens Output tokens
No Retrieval 0 0 0.00 2.4 1.0 65 153
Vanilla RAG 0 0 0.09 2.4 1.0 1,279 142
LightRAG 2,107 10.9 1.8 5.3 2.0 34,324 85
PathRAG𝑎2,107 10.9 1.8 3.8 2.0 17,291 47
HippoRAG𝑏18,391 10.7 0.8 3.4 2.0 956 150
HippoRAG 2 18,391 10.7 0.8 3.9 2.0 4,250 147
MemGraphRAG𝑐,𝑒146,555 114.3 17.9 1.7 1.0 1,389 90
G3RAG 0 0 0.11 2.9 2.0 3,692 124
LinearRAG𝑑0 0 – 2.5 1.0 5,534 133
LogicRAG 0 0 0.09 5.3 4.1 1,926 304
IRCoT 0 0 0.09 4.6 3.0 9,775 240
Self-Ask 0 0 0.09 6.8 6.3 6,140 308
Hi-Q 0 0 0.09 6.4 6.6 13,928 635
DeFT-RAG 0 0 0.09 9.9 4.8 1,569 33
DeFT-RAG + Jev𝑓0 0 0.09 5.0 4.8 1,699 32
What is the date of birth of Po-Chih Leong? →31 Decem-
ber 1939
What film did Po-Chih Leong direct?→(no answer)
The third sub-question is not self-contained: it asks for the birth
date of a director who has not been named yet, and retrieval finds no
answer. DeFT-RAG recovers in the next two steps, but the DM does
not judge the four QA pairs enough, the loop reaches its step limit,
and the fallback answers “Airheads”. On 2Wiki bridge-comparison
questions, 38% of such nested sub-questions get no answer, against
4% of sub-questions that ask for a single relation, and 75 of 235 of
these questions reach the step limit or repeat themselves; this is
the main reason DeFT-RAG only matches Self-Ask on 2Wiki with
gpt-4o-mini (Table 2).Where it fails: the right entity, the wrong fact. “When was the
person who Messi’s goals in Copa del Rey compared to get signed by
Barcelona?”(gold: June 1982). DeFT-RAG first finds the right person
(“Diego Maradona”), so it does not shift back to Messi; but it then
asks for Maradona’s date of birth instead of the date he was signed,
the DM accepts these QA pairs as enough, and the final answer (“30
October 1984”) is wrong. The rule that tells LLM subto ask for dates
of birth when comparing ages (Appendix B) may have pulled the
sub-question towards a birth date.
Remaining errors.On 2Wiki with gpt-4o-mini, an analysis of the
148 wrong answers attributed only about 19 to the DM’s decisions;
most of the others come from the backbone, such as nested sub-
questions or wrong readings of correct passages.