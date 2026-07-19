# Bridge Evidence: Static Retrieval Utility Does Not Predict Causal Utility in Multi-Step Agentic Search

**Authors**: Debayan Mukhopadhyay, Utshab Kumar Ghosh, Shubham Chatterjee

**Published**: 2026-07-16 17:48:29

**PDF URL**: [https://arxiv.org/pdf/2607.15253v1](https://arxiv.org/pdf/2607.15253v1)

## Abstract
Retrieval systems are trained and evaluated on a static idea of usefulness: hand a document and a question to a reader model, see whether the answer improves, and score the document accordingly. The idea holds up when a document is read on its own. It breaks when a language model works as a search agent, issuing several queries and reasoning across turns, because a document can matter for what it lets the agent do next rather than for what it says about the current question.
  We measure that gap rather than argue it. Using a ReAct style agent over HotpotQA, we replay 1000 development questions and, for every document the agent read, delete it and re-run the rest of the trajectory from that point. Comparing the original run against its counterfactual gives a Counterfactual Trajectory Utility (CTU) score from three deltas: final answer quality, next query retrieval quality, and turn count. Crossing CTU against Static RAG Utility (SRU) over 23,322 document observations, the two are close to statistically independent (Spearman rho = -0.026). Roughly a third of the documents the agent reads are causally load bearing while looking useless to a static reader; we call these bridge documents. The pattern survives when the reader based axis is swapped for a BM25 and cross encoder proxy, giving a bridge cell of 27.2% on an evenly spread axis.
  A second experiment pins down the mechanism. Using the Observable Entity Relevance (OER) measure from prior work, entities that discriminate relevant from non-relevant candidates appear in the agent's next query 4.02 times more often than entities found only in non-relevant documents (6.1% vs 1.5%, n = 227,139). A bridge document earns its keep by handing the agent a discriminative entity that redirects the search. Static relevance and causal usefulness are different quantities in agentic retrieval, and optimizing the first does not deliver the second.

## Full Text


<!-- PDF content starts -->

Bridge Evidence: Static Retrieval Utility Does Not Predict Causal
Utility in Multi-Step Agentic Search
Debayan Mukhopadhyay
University of Calcutta
Kolkata, West Bengal, India
debayan.mukherjee14@gmail.comUtshab Kumar Ghosh
Missouri University of Science and
Technology
Rolla, MO, USA
u.ghosh@mst.eduShubham Chatterjee
Missouri University of Science and
Technology
Rolla, MO, USA
shubham.chatterjee@mst.edu
Abstract
Retrieval systems are trained and evaluated on a static idea of use-
fulness. Hand a document and a question to a reader model, see
whether the answer improves, and score the document accordingly.
Almost every standard metric reduces to some version of that ques-
tion. The idea holds up when a document is read on its own. It starts
to break when a language model works as a search agent, issuing
several queries and reasoning across turns, because a document
can matter for what it lets the agent do next rather than for what it
says about the question in front of it.
We measure that gap directly instead of arguing about it. Using
a ReAct style agent over HotpotQA, we replay a stratified sample
of 1000 development questions and, for every document the agent
actually read, delete it and re-run the rest of the trajectory from that
point. Comparing the original run against its counterfactual gives
a Counterfactual Trajectory Utility (CTU) score built from three
deltas: the quality of the final answer, the retrieval quality of the
query the agent issues next, and the number of turns it needs. Cross-
ing CTU against Static RAG Utility (SRU) over 23,322 document
observations, the two turn out to be close to statistically indepen-
dent (Spearman 𝜌=− 0.026). Roughly a third of the documents the
agent reads are causally load bearing while looking useless to a
static reader. We call these bridge documents. The pattern survives
when we swap the reader based SRU axis for a BM25 and cross
encoder proxy, which gives a bridge cell of 27.2% with an evenly
spread axis.
A second experiment pins down the mechanism. Using the Ob-
servable Entity Relevance (OER) measure from prior work on entity
ranking, we find that entities whose presence discriminates rele-
vant from non-relevant candidates appear in the agent’s next query
4.02 times more often than entities found only in non-relevant doc-
uments (6.1% against 1.5%, 𝑛=227,139). A bridge document earns
its keep by handing the agent a discriminative entity that redirects
the search. Taken together the results say that static relevance and
causal usefulness are different quantities in agentic retrieval, and
that optimizing the first does not deliver the second.
Permission to make digital or hard copies of all or part of this work for personal or
classroom use is granted without fee provided that copies are not made or distributed
for profit or commercial advantage and that copies bear this notice and the full citation
on the first page. Copyrights for components of this work owned by others than the
author(s) must be honored. Abstracting with credit is permitted. To copy otherwise, or
republish, to post on servers or to redistribute to lists, requires prior specific permission
and/or a fee. Request permissions from permissions@acm.org.
Conference’17, Washington, DC, USA
©2026 Copyright held by the owner/author(s). Publication rights licensed to ACM.
ACM ISBN 978-x-xxxx-xxxx-x/YYYY/MM
https://doi.org/10.1145/nnnnnnn.nnnnnnnCCS Concepts
•Information systems →Retrieval models and ranking;In-
formation retrieval;Retrieval effectiveness.
Keywords
agentic retrieval, counterfactual evaluation, evidence utility, multi-
hop question answering, entity relevance
ACM Reference Format:
Debayan Mukhopadhyay, Utshab Kumar Ghosh, and Shubham Chatterjee.
2026. Bridge Evidence: Static Retrieval Utility Does Not Predict Causal
Utility in Multi-Step Agentic Search. In.ACM, New York, NY, USA, 10 pages.
https://doi.org/10.1145/nnnnnnn.nnnnnnn
1 Introduction
Ask a modern retrieval system what makes a document good and
you get a clear answer. A document is good if handing it to a reader
along with the question makes the answer better. That definition is
doing an enormous amount of work. It sits underneath nDCG, MAP,
and MRR [ 3], underneath the training objective of most learned
rankers [ 4,6], and underneath the way retrieval augmented gener-
ation is usually evaluated [ 5]. It also assumes something specific
about how the document gets used, namely that a reader sees the
question and the document together, with no history and no plans.
That assumption quietly stops being true once a language model
is allowed to act. A ReAct style agent [ 14] reads a question, decides
what it does not know, searches for it, reads what comes back, and
searches again. Systems in this family now handle a large share
of multi-hop question answering [ 1,9,11]. In that setting the link
between a document and the final answer runs through several
turns of reasoning, and it stops being direct. A document can be
the reason the agent succeeds without containing anything that
resembles the answer. Another document can contain the answer
outright and change nothing at all, because the agent already knew
it.
Consider a question our agent actually faced.What political party
was the 6th governor of Hawaii that passed bills to help with global
warming?At the first step the agent retrieves a passage saying
that Linda Lingle served as the 6th Governor of Hawaii from 2002
to 2010. The passage says nothing about any political party and
nothing about environmental legislation. Score it the standard way,
by giving it to a fresh reader with the question, and it is worthless.
The reader cannot answer from it. Yet that passage is the reason the
agent succeeds, because it supplies the name Linda Lingle, which
the agent puts into its next query, which retrieves the passage that
does contain the party. Delete the first passage and the agent never
arXiv:2607.15253v1  [cs.IR]  16 Jul 2026

Conference’17, July 2017, Washington, DC, USA Utshab Kumar Ghosh, Debayan Mukhopadhyay and Shubham Chatterjee
learns whose party to look up. Its value lies entirely in what it
unlocks.
We call documents of this kindbridge evidence. They bridge
the gap between what the agent knows now and the next search
worth running. Our claim is that they are common, that they are
invisible to the signals retrieval systems currently optimize, and
that this invisibility is systematic rather than accidental. A bridge
document is defined by what it enables in the future, and every
standard signal scores it on what it contains in the present.
1.1 Measuring the gap instead of asserting it
Arguments of this shape are easy to make and hard to believe with-
out numbers, so we measure the gap. The tool is a counterfactual.
If we want to know whether a document mattered causally, the
cleanest thing to do is take it away and see what happens [7].
We build a framework we call Counterfactual Trajectory Explo-
ration. For every document the agent actually read at every step,
we delete that one document from the ranked list, hand the agent
the remaining evidence, and replay the rest of the trajectory from
that point with everything else held fixed. The original run and
the counterfactual run then differ in exactly one thing. Comparing
them gives a Counterfactual Trajectory Utility (CTU) score, which
combines three deltas: whether the final answer got worse, whether
the query the agent issued next retrieved less gold evidence, and
whether the agent needed more turns.
Placing CTU on one axis and Static RAG Utility (SRU) on the
other turns a rhetorical claim into a contingency table. If the field’s
assumption is right, the two should agree and the table should be
diagonal. It is not. Across 23,322 document observations from a
stratified sample of 1000 HotpotQA development questions, the
rank correlation between the two is −0.026. They are, to a good
approximation, measuring unrelated things.
1.2 Why bridge documents work
Showing that a gap exists is one thing. Explaining it is another,
and a reviewer is entitled to ask what a bridge document actually
does. Our answer is that it introduces an entity. The agent picks
that entity up and puts it in its next query, and the search changes
direction.
That explanation is testable, and it connects to a separate line of
work on entity relevance [ 2]. That work distinguishes two notions.
Conceptual Entity Relevance is the topical judgment a person or a
language model makes about whether an entity is about a query.
Observable Entity Relevance (OER) asks a sharper question: does
the entity’s presence in a candidate document actually separate
relevant documents from non-relevant ones? The two agree only
weakly, and OER is the one that helps retrieval. If our account of
bridge evidence is right, then the entities that get carried into the
next query should be the observably discriminative ones rather
than whatever happens to be printed on the page. We test that
prediction and it holds, by a factor of four.1.3 Contributions
•We formalize Counterfactual Trajectory Exploration for multi-
step retrieval agents and give a concrete omission interven-
tion that isolates the causal contribution of a single document
while holding the rest of the trajectory fixed (Section 2.5).
•We define Counterfactual Trajectory Utility, a composite
of answer, next query, and effort deltas, and show that its
natural threshold coincides exactly with the point of zero
causal effect, which gives the high and low split a principled
meaning rather than an arbitrary one (Section 2.6.2).
•We run the quadrant experiment over 23,322 document ob-
servations and find that static utility and causal utility are
close to independent ( 𝜌=− 0.026), with 35.7% of read doc-
uments landing in the bridge cell. The finding survives a
robustness check that replaces the reader based axis with a
BM25 and cross encoder proxy (Section 3.2).
•We identify the mechanism. Entities with high Observable
Entity Relevance propagate into the agent’s next query 4.02
times more often than non-discriminative ones over 227,139
observations (Section 3.5).
•We report the limits of the evidence plainly, including a
skew in the static utility axis that constrains how much the
headline quadrant number can carry on its own (Section 3.6).
2 Methodology
2.1 Problem definition
Our starting claim is that causally essential evidence in multi-step
agentic retrieval has a different shape than the evidence that ex-
isting proxy signals select. The proxies we have in mind are the
familiar ones: the BM25 score, static reader utility, and whether the
document contains the answer string. We argue the mismatch is
not noise. It reflects a specific class of evidence.
Bridge evidence.A document that introduces an en-
tity or a relation which appears in the agent’s next
query but was absent from every previous query, and
whose removal degrades the trajectory, while the doc-
ument itself scores low on existing static proxies.
Bridge evidence is invisible to current signals for a structural
reason rather than an accidental one. Its usefulness is defined by
what it enables at the next step, and every static proxy scores a
document on what it contains relative to the current query. In
concrete terms we expect to find documents that score low on
BM25, low on static reader utility, and low on answer containment,
but whose deletion makes the agent fail or take noticeably longer.
2.2 Trajectory representation
We study a ReAct style research agent [ 14] that answers questions
by searching Wikipedia. It reasons inside a thought block before
each action, searches with an explicit query, and answers when
it judges the evidence sufficient. It is capped at four search turns
and instructed never to repeat a query. The full system prompt is
reproduced in Appendix A.
The agent moves from state 𝑆𝑡to state𝑆𝑡+1through the transition
𝑆𝑡−→ (thought+CW) 𝑡, 𝑞𝑡, 𝐷𝑡−→𝑆𝑡+1,(1)

Bridge Evidence: Static Retrieval Utility Does Not Predict Causal Utility in Multi-Step Agentic Search Conference’17, July 2017, Washington, DC, USA
where𝑆𝑡is the latent reasoning state, (thought+CW) 𝑡is the explicit
chain of thought together with the context window the agent has
accumulated so far, 𝑞𝑡is the sub-query the agent generates at step 𝑡,
and𝐷𝑡is the evidence set the retriever returns for it. The next state
is conditioned on all of these jointly. The agent’s decision therefore
depends not only on the individual documents 𝑑𝑖∈𝐷𝑡but also on
the context window holding every fact and reasoning trace from
earlier steps.
The retrieval backend returns the top 50 paragraphs for each
query, of which the top 10 enter the agent’s context. Query genera-
tion at the next step can be written as
 (thought+CW) 𝑡, 𝐷𝑡[: 10]−→𝑞𝑡+1.(2)
Equation 2 is the part of the loop this paper is really about. It is
the point where a document that answers nothing can still change
everything, by supplying the term that makes the next query work.
We do not intervene on the latent reasoning itself. It is not ob-
servable and not cleanly manipulable. We intervene on the evidence
conditioning variable 𝐷𝑡, which is both observable and exactly the
thing a retrieval system controls.
2.3 Experimental setup
Agent and retrieval stack.The agent backbone is Qwen2.5-7B-
Instruct [ 12]. First stage retrieval is BM25 [ 10] over a paragraph
level Wikipedia index, and the top 50 candidates are reranked by a
cross encoder before the top 10 are shown. To keep the counterfac-
tual comparison meaningful we hold decoding deterministic, with
temperature set to0 .0and sampling disabled, and we run every
trajectory on identical hardware. This does not make a language
model perfectly deterministic, but it removes the largest source of
run to run variation, which is what the causal argument needs.
Documents shown to the agent.Throughout the paper, the doc-
uments we analyze are the ones the agent actually read. We take
these from the shown_doc_ids field of the trajectory log rather
than recomputing the top 10 from the ranked list. The two differ
because shown_doc_ids is deduplicated across steps. Once a doc-
ument has been shown at step 𝑡it is not shown again later. The
reasoning behind that design is that an agent which failed to extract
what it needed on first reading should not get a second free look at
the same text. A document that the agent never saw cannot have
influenced it, so unshown documents are outside the scope of any
causal claim we make.
2.4 Question sampling
We draw a reproducible stratified sample of 1000 questions from the
7,405 question HotpotQA development set [ 13] using the following
procedure.
Coverage filter.We keep a question only when every article title
in its supporting_facts field appears, matched case insensitively,
among the top 50 BM25 results for that question. Questions failing
this test are discarded, because the evidence needed to answer
them is not reachable within the candidate set at all. Studying agent
behavior on questions whose evidence the retriever cannot surface
would confound a retrieval failure with a reasoning failure.Stratified allocation.Let 𝐵and𝐶be the number of surviving
bridge and comparison questions and let 𝑁=𝐵+𝐶 . If𝑁does
not exceed the target sample size, every filtered question is kept.
Otherwise we allocate the sample to hold roughly 70% bridge and
30% comparison questions, which approximates the composition of
the development set itself (5,918 bridge against 1,487 comparison).
If one pool cannot fill its quota, we take all of it and top up from the
other pool until the target is reached. A fixed random seed makes
the draw reproducible. The procedure emits the sampled questions
with all original fields preserved, together with a report broken
down by question type and difficulty level.
2.5 Counterfactual trajectory exploration
Counterfactual Trajectory Exploration (CTE) is our framework for
asking what would have happened if an intermediate decision had
gone differently. At any reasoning or retrieval step, CTE intervenes
on a prior decision and estimates how that change would have
propagated through the rest of the trajectory. The aim is to quantify
the causal contribution of intermediate decisions to downstream
reasoning quality, retrieval efficiency, factual correctness, and task
success.
This paper instantiates CTE with a single intervention.
Omission intervention.We delete one document from the ranked
list at one step and give the agent the remaining𝑘−1:
[𝑑1,𝑑2,...,𝑑 10] −→ [𝑑 1,...,𝑑𝑖−1,𝑑𝑖+1,...,𝑑 10].(3)
The trajectory is then replayed from that step onward, reusing
the logged prefix so that everything before the intervention is
identical by construction. The question it answers is deliberately
narrow. What if this document had simply not been in the candidate
set? Whatever the agent does differently from that point on is
attributable to the deleted document and to nothing else. We sweep
𝑖over all ten shown documents and 𝑡over all search steps, which
is what produces a document level dataset rather than a handful of
anecdotes.
2.6 Document level signals
For every document retrieved at every step of every trajectory in
the intervention set we compute four signals. Together they form
the document level dataset behind the quadrant analysis.
2.6.1 Static RAG Utility.Static RAG Utility (SRU) formalizes the
conventional notion of a useful document. Give the document and
the question to a reader with no history, no earlier turns, and no
accumulated trajectory, then ask how much the answer improves.
We prompt the reader to answer using only the supplied document,
to return the answer span alone, and to say UNANSWERABLE when the
document does not support an answer. The reader prompt appears
in Appendix A.
For question 𝑞𝑖and each document 𝑑𝑗in the set𝐷𝑖of documents
shown across that question’s trajectory, we obtain an answer with
the document,
SRU with(𝑞𝑖,𝑑𝑗)=𝐴with
𝑖,𝑗,(4)
and separately an answer with no document at all, which measures
what the reader already knows from its parameters,
SRU without(𝑞𝑖)=𝐴without
𝑖.(5)

Conference’17, July 2017, Washington, DC, USA Utshab Kumar Ghosh, Debayan Mukhopadhyay and Shubham Chatterjee
The static utility of the document is the improvement it produces
over that parametric baseline,
ΔSRU(𝑞𝑖,𝑑𝑗)=F1 𝐴with
𝑖,𝑗−F1 𝐴without
𝑖.(6)
Subtracting the zero shot baseline matters. Without it, a document
would be credited for an answer the reader could have produced
from memory.
2.6.2 Counterfactual Trajectory Utility.CTU asks a different ques-
tion than SRU. Notdoes this document help on its own, butwould
the agent have done just as well without it. We compute it from the
omission intervention of Equation 3, scoring the counterfactual
against the base trajectory on three axes.
Δanswer =F1 base−F1 cf,(7)
Δnext-query =nDCG@10 𝑞base
𝑡+1−nDCG@10 𝑞cf
𝑡+1,(8)
Δeffort=turns cf−turns base.(9)
All three are oriented so that a positive value means the docu-
ment helped.
Δanswer is the most direct signal. Did removing the document
make the final answer worse? If so the document was causally
important. Δnext-query is the local counterpart. Did removing the
document push the agent into a different next query that retrieves
fewer gold supporting documents? This is the term that makes
bridge evidence measurable, because it scores a document on what
it caused the agent to search for rather than on what it contained.
Δeffort asks whether removal cost the agent extra turns, or a success-
ful answer altogether. A document that helps the agent converge
quickly shows up here.
Combining the three.The components live on different scales, so
we min-max normalize each independently across all documents
in the intervention set and then sum them with unit weights:
CTU=𝑤 1˜Δanswer+𝑤 2˜Δnext-query+𝑤 3˜Δeffort, 𝑤 1=𝑤 2=𝑤 3=1,
(10)
where ˜·denotes the normalized component. We keep the weights
uniform deliberately. Tuning them against the outcome we are
trying to demonstrate would make the result circular.
The zero effect point.Equation 10 has a property worth stating
explicitly, because it gives the high and low CTU split a meaning
that a median cut alone would not. The observed raw ranges are
[−1,1]forΔanswer ,[−1,1]forΔnext-query , and[−2,3]forΔeffort. A
document that changes nothing at all has a raw value of zero on
every component, which normalizes to
0−(−1)
1−(−1)| {z }
0.5+0−(−1)
1−(−1)| {z }
0.5+0−(−2)
3−(−2)| {z }
0.4=1.4.(11)
SoCTU= 1.4is exactly the point of no causal effect. Values above
it mean the document was, on balance, helping. Values below it
mean the agent did better without it. As Section 3.1 shows, the em-
pirical median of CTU is also1 .400, which is not a coincidence: the
median document has a raw delta of zero on all three components.
Thresholding at1 .400therefore separates documents that helped
from documents that did not, and it happens to coincide with the
median rather than being defined by it.Table 1: Quadrant cell definitions. The SRU axis splits at zero
improvement over the parametric baseline. The CTU axis
splits at the zero effect point of Equation 11.
CellΔSRUCTU Interpretation
A>0(high)>1.400(high) Expected useful
B>0(high)≤1.400(low) Redundant evidence
C≤0(low)>1.400(high)Bridge evidence
D≤0(low)≤1.400(low) Genuinely useless
What CTU is a property of.Documents do not change trajectories
on their own. What changes a trajectory is a model’s reading of a
document, and two models can reasonably disagree about the same
text. The honest form of the quantity is therefore
CTU(𝐷𝑡,𝐴,𝑡),(12)
where𝐴is the reasoning agent and 𝑡carries the internal state
(thought+CW) 𝑡at that step. We do not carry 𝑞𝑡separately because
it is already encoded in that state. Every number in this paper is
measured with respect to one agent, and we make no claim that
CTU transfers unchanged to another.
2.6.3 Retrieval score proxies.BM25 scores are unbounded above,
so we normalize them by the per question maximum,
BM25 norm(𝑑𝑗)=BM25(𝑑𝑗)
max𝑑∈𝐷 𝑖BM25(𝑑).(13)
Cross encoder scores can be negative and their range depends on
the model, so we min-max normalize them within each question,
CEnorm(𝑑𝑗)=CE(𝑑𝑗)−min𝑑∈𝐷 𝑖CE(𝑑)
max𝑑∈𝐷 𝑖CE(𝑑)−min 𝑑∈𝐷 𝑖CE(𝑑).(14)
2.6.4 Answer containment.Answer containment asks the blunt
question of whether the document contains the gold answer. We
check whether the gold answer string appears in the document text
as a case insensitive substring, and whether any gold supporting
fact title from the HotpotQA metadata matches the document title.
The score is binary rather than graded. It is a strict proxy, included
because it is the crudest version of the assumption we are testing.
2.7 The quadrant construction
The quadrant experiment is the empirical core of the argument. It
crosses static utility against causal utility and asks two questions.
How often do they disagree, and in which direction?
We split each axis at a threshold and read off four cells. A doc-
ument is high SRU when ΔSRU> 0, meaning it improved the
stateless reader’s answer, and high CTU when CTU> 1.400, mean-
ing its removal hurt the agent on balance, per Equation 11.
Table 1 names the cells. Two of them are uninteresting by design.
Cell A is where both measures agree that a document is useful,
which is what everyone already believes good retrieval does. Cell
D is where both agree it is noise. The argument lives in the off-
diagonal.
Cell B, redundant evidence,is where static utility overesti-
mates. The document contains the answer and the stateless reader
is happy, but the agent was going to succeed regardless, either

Bridge Evidence: Static Retrieval Utility Does Not Predict Causal Utility in Multi-Step Agentic Search Conference’17, July 2017, Washington, DC, USA
because it had picked the fact up at an earlier step or because the in-
formation was already in the model’s parameters. Take the Hawaii
question again. At step 1 the agent reads that Lingle became the
first Republican Governor of Hawaii since 1962, so it already knows
the party. If at step 3 it retrieves another passage about Lingle’s
affiliation, a fresh reader scores that passage highly, yet omitting it
changes nothing. The passage is real evidence and causally inert at
the same time.
Cell C, bridge evidence,is the cell that carries the paper. Static
utility says the document is useless because it does not contain the
answer. CTU says the trajectory falls apart without it. The Linda
Lingle passage from Section 1 sits here. For the argument to be
defensible this cell has to be populated at a non-trivial rate, and we
set a threshold of at least 20% in advance.
2.8 Entity propagation
The quadrant experiment establishes that bridge documents exist.
It does not explain why they work. Our account is that a bridge
document hands the agent an entity, and the agent puts that entity
into its next query. If that is right, the entities that get carried
forward should be the ones that carry a real retrieval signal, not
any entity that happens to be on the page.
We test this with the Observable Entity Relevance measure
of Ghosh and Chatterjee [2], reusing that work’s computation with-
out modification.
Hypotheses.
•H1 (propagation gap).An entity that is observably discrim-
inative in a step’s candidate set appears in the agent’s next
query more often than an entity found only in non-relevant
documents. We fix a target ratio above2 .5×in advance so a
small gap does not count as support.
•H2 (bridge link).Among discriminative entities, those car-
ried by bridge documents propagate more than those carried
by grounded Cell A documents, which would localize the
redirecting entities to the bridge cell.
2.8.1 Relevance labels.OER is defined against relevance labels, so
we need judgments for every candidate document. We derive them
straight from the HotpotQA supporting facts, and two alternatives
are worth explaining away.
We do not use the BEIR or ir_datasets release of HotpotQA.
Those index a different version of the corpus with different docu-
ment identifiers than our paragraph index, which would produce
silent zero overlap between run and qrels. Deriving labels from the
supporting facts guarantees alignment with our index by construc-
tion.
We also do not use a language model to judge relevance. Our
agent is a language model. Scoring relevance with an LLM and then
testing whether high scoring entities reappear in that same LLM’s
next query would mostly measure the model’s agreement with
itself. That circularity would hollow out the result, so the relevance
signal has to come from the dataset.
Each candidate paragraph gets one of three levels. Level 2 means
the paragraph comes from a gold supporting article and contains
the specific gold sentence. Level 1 means it comes from a gold
supporting article but not that sentence. Level 0 is everything elseand is omitted from the qrels under the usual TREC convention,
with unjudged candidates treated as non-relevant. The high and
low split uses any level at or above 1, so the level 2 refinement does
not affect the partition.
2.8.2 Unit of analysis and candidate sets.We treat each agent
search step as one OER query, identified as {qid}_step{k} , and
key both the qrels and the run by that identifier so they stay con-
sistent. The candidate pool for a step is the top 20 documents by
cross encoder rank unioned with the documents actually shown
at that step. The union guarantees that every document we later
measure for propagation carries an OER score, even when cross
step deduplication pushes a shown document past rank 20 of its
own step.
2.8.3 Entity linking and OER.We link every unique candidate doc-
ument with the WAT entity linker [ 8] and keep mentions with
confidence𝜌≥0.1. WAT returns underscore joined titles, which
we normalize to spaces before matching.
For an entity 𝑒in the candidate set of step 𝑞, let𝑅and𝑁𝑅be the
counts of relevant and non-relevant candidates, let dfrelanddfnonrel
be the number of relevant and non-relevant candidates containing
𝑒, and let dfcandbe the total number of candidates containing 𝑒.
OER is a support weighted difference of smoothed log odds:
ˆ𝑝rel=dfrel+𝛼
𝑅+2𝛼, ˆ𝑝nonrel=dfnonrel+𝛼
𝑁𝑅+2𝛼,(15)
𝑤(𝑒,𝑞)=1−exp
−dfcand
𝜏
,OER(𝑒,𝑞)=𝑤(𝑒,𝑞)
logit ˆ𝑝rel−logit ˆ𝑝nonrel
.
(16)
The weight𝑤discounts entities seen in only one or two documents,
where the log odds estimate would be unstable. We use the source
paper’s defaults, 𝛼= 0.5and𝜏=5.0. A positive OER means the
entity concentrates in the relevant documents of that step.
We form two contrasting groups.High OERentities appear in
at least one gold supporting document ( dfrel≥1) and are positively
discriminative ( OER> 0).Low OERentities never appear in a gold
supporting document ( dfrel=0). Entities present in gold documents
but not discriminative form a middle group that we record and
exclude from the contrast. We use this rule rather than a median
split because the number of relevant documents per step is small,
which makes a median cut unstable.
2.8.4 Measuring propagation.Propagation is measured only over
documents the agent actually saw, and only into the immediately
following query. An entity in a retrieved but unshown document
cannot influence the model. The last search step of each trajectory
is skipped because it has no next query. We normalize entity titles
and queries by lower casing, turning underscores into spaces, and
dropping non alphanumeric characters, then apply two criteria.
Exactmatch, the primary and conservative one, requires the
full normalized title to appear as a whole phrase in the next query,
matched at token boundaries so that a short title such aslawcannot
match insidelawyer.Partialmatch requires any non-stopword
token of length at least two from the title to appear as a whole
token, using a 132 word English stoplist. Partial raises the absolute
rate and admits common token noise such aslakeorcounty, which
shrinks the gap between groups.

Conference’17, July 2017, Washington, DC, USA Utshab Kumar Ghosh, Debayan Mukhopadhyay and Shubham Chatterjee
For each criterion we report the propagation rate per group, the
ratio between groups, a bootstrap 95% confidence interval from
2,000 resamples, and a chi square test of independence. We treat
H1 as supported when the ratio clears2 .5×, the test is significant,
and the intervals do not overlap.
3 Results and Analysis
3.1 Dataset and descriptive statistics
Before analyzing the intervention set we drop one group. Trajec-
tories that finished in a single turn ( 𝑛=74trajectories) have no
subsequent step for a document to causally affect, which makes
CTU undefined rather than merely small. Keeping them would pad
the denominator with cases the measure cannot speak to, so they
are removed. The remaining multi-step trajectories yield 23,322 doc-
ument observations covering 16,841 unique question and document
pairs.
Table 2 reports the distribution of every signal over the full set.
The composite CTU of Equation 10 spans [0.000,2.800]against a
theoretical range of[0,3].
Two features of Table 2 deserve comment before we go further,
because both shape how the quadrant should be read.
First, the median of all three CTU components is exactly zero.
The typical document changes the answer not at all, changes the
next query not at all, and changes the turn count not at all. That
is what puts the empirical median of CTU at1 .4000, matching the
analytic zero effect point derived in Equation 11 to four decimal
places. The threshold we use is therefore not a convenience. It is
the value at which a document stops mattering.
Second, the static utility axis is compressed. The mean ΔSRU
is0.0048and its median is zero, and the reader’s absolute answer
quality is low with or without the document (0 .0170against0 .0122).
Answer containment tells the same story from another angle: only
6.38% of unique pairs and 7.50% of records contain the gold answer
at all. This is expected for multi-hop questions, where most indi-
vidual paragraphs hold one hop of a two hop chain and cannot
support the answer alone. It nevertheless means the SRU axis has
little dynamic range, and Section 3.6 takes that seriously.
3.2 The quadrant: static utility against causal
utility
Table 3 crosses the two axes. The headline is the bridge cell. Across
all observations, 35.72% of the documents the agent read were
causally load bearing while offering nothing to a stateless reader.
That clears the 20% threshold we committed to in Section 2.7 by
a wide margin. The two cells where the measures agree account
for a combined 62.10%, and the redundant cell, where static utility
overestimates, holds another 2.18%.
Table 3 also breaks the cells out by turn count, and that break-
down needs a warning rather than an interpretation. The bridge cell
does rise across the columns, from 22.30% at two turns to 40.18% at
three and 41.62% at four, and it is tempting to read this as evidence
that harder questions lean more heavily on bridge evidence. We
looked into that reading and it does not hold. The grouping variable
is confounded with CTU by construction, for reasons Section 3.6
sets out in full, and the confound is large enough to account for the
entire trend. We report the columns for transparency and draw noconclusion from them. Nothing else in this paper depends on the
breakdown.
3.3 Static and causal utility are close to
independent
The cell counts show disagreement. The rank correlation shows
how complete the disagreement is. Table 4 reports Spearman 𝜌
betweenΔSRUand CTU.
Every coefficient sits within0 .031of zero. The pooled correlation
reaches statistical significance only because 23,322 observations will
detect almost any deviation from zero, and a 𝜌of−0.026explains
roughly 0.07% of the variance. The right reading is not that static
utility is mildly negatively related to causal utility. It is that the
two are unrelated. A retrieval system that perfectly optimizes static
utility carries almost no information about which documents its
agent actually needed.
This is the paper’s central result, and it is worth being precise
about which part is doing the work. The independence is the finding.
The size of the bridge cell follows from that independence together
with how the two marginals happen to fall, a point we return to in
Section 3.6.
3.4 Robustness: the BM25 and cross encoder
quadrant
A reader based axis is one way to operationalize static utility. It is
not the only one, and it is the one with the least dynamic range
in our data. We therefore repeat the quadrant with the static axis
replaced by the retrieval scores themselves, which is closer to what
a deployed system optimizes and gives an axis that is spread out by
construction.
We build a combined proxy from the normalized BM25 and cross
encoder scores using global cutoffs (BM25 75th percentile0 .9070
and median0 .8091, cross encoder 75th percentile0 .6061and median
0.2733). A document counts as proxy high when both scores agree
that it is high and proxy low when both agree it is low. Records
where the two signals disagree are excluded, which removes 13,722
of 23,322 observations (58.84%) and leaves 9,600. The exclusion is
substantial and we flag it as a limitation in Section 3.6, but the
surviving records give a cleaner test of whether the effect depends
on the reader.
The picture holds. Table 5 puts the bridge cell at 27.16%, still
above our 20% bar, on an axis where 27.14% of documents score
high rather than 3.30%. Rank correlations between the proxy and
CTU again sit near zero, and this time most of them do not even
reach significance. Over all records 𝜌=− 0.0161with𝑝= 0.116.
Broken out by turn count the coefficients are −0.0592at two turns,
−0.0349at three, and0 .0004at four, of which only the two turn
figure is significant (𝑝=0.0018).
This matters more than the raw number. The reader based quad-
rant could be dismissed as an artifact of a weak reader. The proxy
quadrant cannot, because it never consults a reader. Whatever a
BM25 and cross encoder stack prefers is, on this evidence, unrelated
to what the agent needed.

Bridge Evidence: Static Retrieval Utility Does Not Predict Causal Utility in Multi-Step Agentic Search Conference’17, July 2017, Washington, DC, USA
Table 2: Descriptive statistics over all 23,322 document observations. Every component delta has a median of exactly zero,
which is what places the median CTU at the zero effect point of Equation 11.
Metric Mean Median Std Min Max
ΔSRU(sru_f1_doc_delta) 0.0048 0.0000 0.1485−1.00001.0000
sru_f1_withoutdocs0.0122 0.0000 0.0938 0.0000 1.0000
sru_f1_withdocs0.0170 0.0000 0.1152 0.0000 1.0000
CTU (composite) 1.3911 1.4000 0.3137 0.0000 2.8000
Δanswer −0.02250.0000 0.4117−1.00001.0000
Δnext-query −0.04820.0000 0.4427−1.00001.0000
Δeffort 0.1324 0.0000 1.0631−2.00003.0000
bm25_normalized0.7953 0.8091 0.1454 0.1775 1.0000
ce_normalized0.3700 0.2733 0.3324 0.0000 1.0000
Table 3: Quadrant cell counts and within group percentages. The turn count columns group by the length of thecounterfactual
trajectory, a variable that is confounded with CTU by construction (Section 3.6). They are reported for completeness and no
trend should be read from them.
Cell Meaning All 2 turns 3 turns 4 turns
A Expected useful 262 (1.12%) 50 (0.75%) 75 (1.37%) 137 (1.23%)
B Redundant 508 (2.18%) 179 (2.67%) 91 (1.66%) 238 (2.14%)
CBridge 8,331 (35.72%)1,496 (22.30%) 2,206 (40.18%) 4,629 (41.62%)
D Genuinely useless 14,221 (60.98%) 4,985 (74.29%) 3,118 (56.79%) 6,118 (55.01%)
Total 23,322 6,710 5,490 11,122
Table 4: Spearman correlation between static utility and coun-
terfactual utility. The coefficient sits near zero in every group.
Large𝑛makes the pooled 𝑝value small while the effect size
stays negligible.
Group𝜌 𝑝 𝑛
All records−0.0257 8.754×10−523,322
2 turns−0.0083 0.4996,710
3 turns−0.0252 0.06225,490
4 turns−0.0301 0.001511,122
3.5 Entity propagation: why bridge documents
work
The quadrant establishes that bridge documents are common. The
propagation experiment tests our explanation of them.
The pipeline produced 227,139 entity at a step observations from
769 multi-step trajectories, of which 68,546 fall in the high OER
group, 145,927 in the low OER group, and 12,666 in the excluded
middle. That is close to four times the 60,053 observations of an
earlier 300 question pilot, which lets us treat the pilot as a first run
and these numbers as its replication.
H1 holds, and the ratio is stable across samples.Table 6 gives
the result. Under the conservative exact criterion, an entity that
discriminates relevant from non-relevant candidates reaches the
agent’s next query 6.1% of the time, against 1.5% for an entity
found only in non-relevant documents. That is a ratio of4 .02×,comfortably past the2 .5×bar set in advance. Under the looser
partial criterion the rates rise to 14.4% and 8.2% for a ratio of1 .77×.
Both gaps carry a chi square 𝑝below10−300with non-overlapping
bootstrap intervals.
The ratio moves from4 .18×to4.02×under exact match and from
1.61×to1.77×under partial match as the sample triples. Absolute
rates fall somewhat on the larger sample, which tracks a smaller
relevance pool per step (2.9 gold paragraphs on average against 6.1
in the pilot), leaving a high OER entity fewer gold documents to
earn its status from. The separation between groups is what stays
fixed, and the separation is the claim.
The two criteria trade off exactly as expected. Exact match is a
conservative floor that demands the agent reproduce a full entity
title verbatim. Partial match allows paraphrase and picks up more
real propagation, at the cost of admitting common tokens such as
lakeorcountythat would appear regardless of any OER signal. That
noise inflates the low OER rate and compresses the ratio. Exact is
the cleaner discrimination signal, partial is the better estimate of
the absolute level, and neither reading changes the direction.
Read against the quadrant, this closes the loop. A bridge doc-
ument is causally essential because it supplies an entity, and the
entities that reach the next query are the discriminative ones by a
factor of four. The same log odds signal that separates useful entities
from generic bait in static reranking [ 2] also governs what a search
agent carries forward. The link between static entity relevance and
agent behavior is mechanical, not rhetorical.
H2 does not replicate, and we do not claim it.Our second hy-
pothesis, that the propagating entities sit specifically inside bridge

Conference’17, July 2017, Washington, DC, USA Utshab Kumar Ghosh, Debayan Mukhopadhyay and Shubham Chatterjee
Table 5: Proxy quadrant using normalized BM25 and cross encoder scores as the static axis, over the 9,600 records where the
two signals agree. The bridge cell remains large. The turn count columns carry the same confound as Table 3 and are not
interpreted.
Cell Meaning All 2 turns 3 turns 4 turns
A Proxy high, CTU high 957 (9.97%) 153 (5.49%) 253 (11.27%) 551 (12.06%)
B Proxy high, CTU low 1,648 (17.17%) 570 (20.45%) 357 (15.91%) 721 (15.78%)
CBridge 2,607 (27.16%)458 (16.43%) 690 (30.75%) 1,459 (31.93%)
D Proxy low, CTU low 4,388 (45.71%) 1,606 (57.62%) 944 (42.07%) 1,838 (40.23%)
Total 9,600 2,787 2,244 4,569
Table 6: H1 propagation gap on the 1000 question sample, with the 300 question pilot for comparison. High OER entities reach
the agent’s next query several times more often than low OER entities, and the ratio barely moves between samples.
Criterion Sample High OER Low OER Ratio𝜒2𝑝
Exact (whole title) 300q pilot 8.9% 2.1%4.18×1.6×10−271
Exact (whole title)1000q 6.1% 1.5%4.02×<10−300
Partial (title token) 300q pilot 17.9% 11.1%1.61×7.7×10−107
Partial (title token)1000q 14.4% 8.2%1.77×<10−300
Table 7: Propagation rate by quadrant cell over all entities.
The bridge cell does not stand out, because bridge documents
are entity rich and dominated by low OER entities that di-
lute the average. Two caveats apply. These cells are cut at
CTU> 1.600rather than the1 .400used everywhere else in
the paper, so they are not directly comparable to Table 3. The
four counts sum to 223,024, with the remaining 4,115 obser-
vations sitting on documents that carry no cell assignment.
Cell Meaning𝑛Exact rate Partial rate
A Expected useful 737 0.058 0.114
B Redundant 6,118 0.064 0.166
CBridge38,860 0.032 0.093
D Genuinely useless 177,309 0.045 0.122
documents, does not survive at scale. In the pilot, high OER enti-
ties in bridge documents propagated slightly more than those in
grounded Cell A documents under exact match (9.6% against 7.9%).
On the full sample the comparison reverses to 5.1% against 6.2%,
and Cell A holds only 321 high OER observations, which leaves
it badly underpowered. Under partial match the bridge cell edges
ahead again (11.9% against 11.2%). A result that changes sign with
the matching rule and rests on a few hundred observations is not
one we are willing to build on, so we report H2 as inconclusive. One
further caveat sharpens that verdict rather than softening it. The
cell assignment used here thresholds CTU at1 .600instead of the
1.400zero effect point defended in Section 2.6.2, so this partition
is stricter than the one behind Table 3 and the two are not strictly
comparable. Since we are declining to claim H2 either way, the
discrepancy does not affect any conclusion we draw, but it should
be reconciled before anyone revisits the hypothesis.Table 7 shows why the cell cut fails where the OER cut succeeds.
Bridge documents are entity rich, and most of the entities on them
are low OER. Averaging propagation over every entity in a bridge
document therefore dilutes the contribution of the few discrim-
inative ones until the cell looks unremarkable. The informative
partition is by OER, not by cell, and that partition is exactly where
the effect is clean. The honest summary is that OER predicts propa-
gation, while the localization of propagating entities to a particular
quadrant cell is not established by this experiment.
3.6 Threats to validity
The turn count breakdown is confounded and carries no difficulty
interpretation.The columns in Tables 3 and 5 group by the number
of turns taken by thecounterfactualtrajectory rather than the
base trajectory, so they do not measure how hard a question is to
begin with. The deeper problem is that this variable is mechanically
tied to a component of the quantity it is crossed against. Since
Δeffort=turns cf−turns baseand the base trajectory runs between
one and four turns, fixing the counterfactual turn count at 𝑐confines
Δeffort to the interval[𝑐− 4,𝑐− 1]. The observed per group ranges
match that prediction exactly, at [−2,1]for two turns,[−1,2]for
three, and[0,3]for four. Conditioning on the group therefore drags
Δeffort upward as the group index rises, which drags CTU upward,
which moves documents across the CTU threshold and inflates the
bridge cell without anything substantive happening.
The artifact is more than large enough to produce the whole
pattern. Mean Δeffort climbs from−0.5514at two turns to0 .6040at
four, a shift of0 .2311once normalized, against a total mean CTU
shift of only0.1594across the same groups. The confounded com-
ponent alone over-explains the trend, which means the other two
components drift slightly the other way. We therefore withdraw
the difficulty reading entirely and report the columns only so that
a reader who computes them is not left wondering why we stayed

Bridge Evidence: Static Retrieval Utility Does Not Predict Causal Utility in Multi-Step Agentic Search Conference’17, July 2017, Washington, DC, USA
silent. The aggregate results are untouched, because the indepen-
dence tests, the bridge cell percentage, and the entire propagation
experiment are computed without conditioning on this variable.
The static utility axis is skewed, and the bridge percentage inherits
that skew.This is the objection we would raise first as reviewers, so
we raise it ourselves. Only 3.30% of observations have ΔSRU> 0,
which means the high and low split on the static axis is close to
degenerate. Meanwhile 36.85% of observations have high CTU. If
the two axes were exactly independent, the expected size of the
bridge cell would be (1−0.0330)×0.3685=35.63%, against an
observed 35.72%. The bridge cell is, to within a tenth of a percentage
point, what independence plus these marginals predicts.
We draw two conclusions from that arithmetic rather than one.
The first is a caution. The figure of 35.72% should not be read as a
second, separate discovery on top of the independence result. It is
largely a restatement of it, and a paper that presented the two as
independent confirmations would be double counting. The second
is that the underlying claim is undamaged, because independence is
precisely the thing we set out to test and the thing the correlations
in Table 4 establish directly. The proxy quadrant of Section 3.4 is
the stronger evidence on this point, since it reproduces both a large
bridge cell and near zero correlation on an axis where 27.14% of
documents score high rather than 3.30%.
The static reader is weak in absolute terms.Answer F1 for the
reader averages0 .0170with a document and0 .0122without one.
Most shown paragraphs simply cannot support a standalone answer
to a multi-hop question, which is what the 7.50% answer contain-
ment rate reflects. The consequence is that low SRU often means
the reader failed both with and without the document, rather than
that the document was demonstrably unhelpful. This weakens the
reader based axis as a measurement instrument. It does not weaken
the argument, because the claim under test is about what the agent
needed rather than about what the reader could do, and the proxy
quadrant reaches the same conclusion without a reader.
The proxy quadrant discards disagreement.Requiring BM25 and
the cross encoder to agree excludes 58.84% of records. Those ex-
cluded cases are exactly the ones where the two retrieval signals
conflict, and they may well behave differently. The proxy quadrant
should be read as a check on the subset where a deployed stack
would be confident, not as a claim about the full population.
CTU is a property of an agent, not of a document.Equation 12
makes this explicit. Every number here is measured with respect
to one backbone, Qwen2.5-7B-Instruct, under one prompt and one
retrieval stack. A different model could read the same passage
differently and assign it a different CTU. We report a property of an
agent and its evidence jointly, and whether the bridge phenomenon
holds at other scales is an open question we do not answer.
Determinism is approximate.The causal argument assumes the
agent responds consistently to the same context. We set tempera-
ture to0.0, disable sampling, and hold the hardware fixed, which
removes the dominant sources of variation but does not make a
language model exactly reproducible. Residual non-determinism
adds noise to individual CTU scores. Given effect sizes measuredover tens of thousands of observations, we do not expect it to move
the aggregate conclusions.
The relevance pool for OER is small.Our OER estimates come
from HotpotQA gold supporting facts, which supply roughly two
to six relevant paragraphs per step. That is a smaller pool than the
setting the original OER study used, so individual scores are noisier
here. The effect nonetheless comes through under both matching
criteria and across a threefold change in sample size, which is why
we are comfortable resting the cross experiment claim on it.
4 Conclusion
We set out to test an assumption that retrieval research rarely states
out loud: that a document useful to a reader in isolation is the same
document an agent needs. Across 23,322 counterfactual document
observations drawn from 1000 HotpotQA questions, it is not. Static
utility and causal utility correlate at 𝜌=− 0.026, which is another
way of saying they are unrelated. About a third of the documents
our agent read were load bearing while looking worthless to a
stateless reader. Swapping the reader for a BM25 and cross encoder
proxy, which is closer to what deployed systems actually optimize,
leaves the picture intact at 27.16%.
We also know why these documents work. They carry entities
the agent picks up and searches with. Entities that discriminate
relevant from non-relevant candidates reach the next query 4.02
times more often than entities found only in non-relevant docu-
ments, over 227,139 observations. The signal that identifies useful
entities for a static ranker [ 2] turns out to identify the entities a
search agent propagates. That makes the connection between the
two settings mechanical rather than a matter of framing, and it
gives bridge evidence a concrete definition instead of an intuition.
Three results did not go our way and we report them as such.
The finer hypothesis that propagating entities localize to bridge
cells reversed direction between the pilot and the full sample on
an underpowered comparison, so we treat it as inconclusive. The
reader based static axis turned out to be badly skewed, with only
3.30% of documents scoring above zero, which means the headline
bridge percentage largely restates the independence result rather
than confirming it separately. And an appealing secondary pattern,
in which the bridge cell grew with the number of turns, dissolved
once we checked it. The grouping variable is the counterfactual
turn count, which is mechanically tied to the effort term inside
CTU, and the induced artifact more than accounts for the trend.
We withdrew that claim rather than ship it. What survives is the
independence itself, the proxy quadrant, and the propagation result,
each of which stands without the others.
What follows from this is a target problem rather than a finished
solution. If causal utility is what matters in agentic retrieval and
static utility does not predict it, then the labels the field trains on
are the wrong labels for this setting, and the metrics it reports
are measuring something adjacent to what it wants. The obvious
next step is a selector that approximates CTU without running the
intervention, since counterfactual replay is far too expensive to sit
inside a training loop. The propagation result suggests where to
look. Observable entity relevance is computable from candidate set
statistics alone, needs no counterfactual, and predicts the behavior
we care about by a factor of four. That looks like a usable proxy for

Conference’17, July 2017, Washington, DC, USA Utshab Kumar Ghosh, Debayan Mukhopadhyay and Shubham Chatterjee
the signal a ranker would need in order to serve an agent rather
than a reader.
References
[1]Akari Asai, Zeqiu Wu, Yizhong Wang, Avirup Sil, and Hannaneh Hajishirzi. 2024.
Self-RAG: Learning to Retrieve, Generate, and Critique through Self-Reflection.
InInternational Conference on Learning Representations (ICLR).
[2]Utshab Kumar Ghosh and Shubham Chatterjee. 2026. Entity Labels Are Not
Entity Signals: A Framework for Observable Relevance in Document Re-Ranking.
InProceedings of the 2026 ACM SIGIR International Conference on the Theory of
Information Retrieval (ICTIR ’26). Association for Computing Machinery, New
York, NY, USA. arXiv:2606.15998 [cs.IR] doi:10.1145/3805712.3808561
[3]Kalervo Järvelin and Jaana Kekäläinen. 2002. Cumulated Gain-Based Evaluation
of IR Techniques.ACM Transactions on Information Systems20, 4 (2002), 422–446.
[4]Vladimir Karpukhin, Barlas Oğuz, Sewon Min, Patrick Lewis, Ledell Wu, Sergey
Edunov, Danqi Chen, and Wen-tau Yih. 2020. Dense Passage Retrieval for Open-
Domain Question Answering. InProceedings of the 2020 Conference on Empirical
Methods in Natural Language Processing (EMNLP). 6769–6781.
[5]Patrick Lewis, Ethan Perez, Aleksandra Piktus, Fabio Petroni, Vladimir Karpukhin,
Naman Goyal, Heinrich Küttler, Mike Lewis, Wen-tau Yih, Tim Rocktäschel,
Sebastian Riedel, and Douwe Kiela. 2020. Retrieval-Augmented Generation for
Knowledge-Intensive NLP Tasks. InAdvances in Neural Information Processing
Systems (NeurIPS), Vol. 33. 9459–9474.
[6]Rodrigo Nogueira and Kyunghyun Cho. 2019. Passage Re-ranking with BERT.
arXiv preprint arXiv:1901.04085(2019).
[7]Judea Pearl. 2009.Causality: Models, Reasoning, and Inference(2nd ed.). Cambridge
University Press.
[8]Francesco Piccinno and Paolo Ferragina. 2014. From TagME to WAT: A New
Entity Annotator. InProceedings of the First International Workshop on Entity
Recognition and Disambiguation (ERD ’14). 55–62.
[9]Ofir Press, Muru Zhang, Sewon Min, Ludwig Schmidt, Noah A. Smith, and Mike
Lewis. 2023. Measuring and Narrowing the Compositionality Gap in Language
Models. InFindings of the Association for Computational Linguistics: EMNLP.
5687–5711.
[10] Stephen Robertson and Hugo Zaragoza. 2009. The Probabilistic Relevance Frame-
work: BM25 and Beyond.Foundations and Trends in Information Retrieval3, 4
(2009), 333–389.
[11] Harsh Trivedi, Niranjan Balasubramanian, Tushar Khot, and Ashish Sabharwal.
2023. Interleaving Retrieval with Chain-of-Thought Reasoning for Knowledge-
Intensive Multi-Step Questions. InProceedings of the 61st Annual Meeting of the
Association for Computational Linguistics (ACL). 10014–10037.
[12] An Yang, Baosong Yang, Beichen Zhang, et al .2025. Qwen2.5 Technical Report.
arXiv preprint arXiv:2412.15115(2025).
[13] Zhilin Yang, Peng Qi, Saizheng Zhang, Yoshua Bengio, William W. Cohen, Ruslan
Salakhutdinov, and Christopher D. Manning. 2018. HotpotQA: A Dataset for
Diverse, Explainable Multi-hop Question Answering. InProceedings of the 2018
Conference on Empirical Methods in Natural Language Processing (EMNLP). 2369–
2380.
[14] Shunyu Yao, Jeffrey Zhao, Dian Yu, Nan Du, Izhak Shafran, Karthik Narasimhan,
and Yuan Cao. 2023. ReAct: Synergizing Reasoning and Acting in Language
Models. InInternational Conference on Learning Representations (ICLR).
A Prompts
All three prompts are reproduced verbatim as used in the experi-
ments.
A.1 Agent system prompt
The ReAct style research agent of Section 2.2 runs under the fol-
lowing system prompt.
You are a research agent that answers questions
by searching Wikipedia.
Break complex questions into steps and search
iteratively.
Rules:
1. Use <thought> to reason before each action.
2. Search with <action>search</action> and <query>.
3. Answer with <action>answer</action> when you
have enough evidence.
4. <answer> must be concise: a name, date, place,
or short phrase.5. Cite document numbers: <citations>Doc 1, Doc
3</citations>. If none: <citations>NONE</citations>.
6. Maximum 4 search turns. If you reach 4, answer
with your best guess.
7. Never repeat the same query.
A.2 Static reader prompt
The reader used to compute SRU within Section 2.6.1 is constrained
to the supplied document and to a bare answer span.
Answer using ONLY the information in the provided
document. Respond with ONLY the answer itself: a
name, date, place, or short phrase (a few words
at most). Do NOT explain your reasoning, do NOT
restate the question, and do NOT include any text
other than the answer span itself. If the document
does not contain enough information to answer, you
MUST write UNANSWERABLE. Do not use any outside
knowledge.
Example:
Question: What year was the Eiffel Tower completed?
Document: The Eiffel Tower is a wrought-iron lattice
tower in Paris, France. It was completed in 1889
as the entrance arch for the World’s Fair.
Answer: 1889
Question: {question}
Document: {passage}
Answer:
A.3 Zero shot prompt
The parametric baseline SRU without uses the same reader with no
document supplied.
Answer the following question. Respond with ONLY
the answer itself: a name, date, place, or short
phrase (a few words at most). Do NOT explain your
reasoning, do NOT restate the question, and do
NOT include any text other than the answer span
itself.
Example:
Question: What year was the Eiffel Tower completed?
Answer: 1889
Question: {question}
Answer: