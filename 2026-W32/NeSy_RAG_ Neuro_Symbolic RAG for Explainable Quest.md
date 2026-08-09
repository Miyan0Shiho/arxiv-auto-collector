# NeSy-RAG: Neuro-Symbolic RAG for Explainable Question Answering

**Authors**: Jonas Gann, Michael Gertz

**Published**: 2026-08-06 17:16:28

**PDF URL**: [https://arxiv.org/pdf/2608.06292v1](https://arxiv.org/pdf/2608.06292v1)

## Abstract
Retrieval-augmented generation (RAG) improves question answering by grounding large language models (LLMs) in external knowledge such as text corpora. However, its reasoning process remains largely opaque: intermediate reasoning steps are difficult to verify and cannot be reliably attributed to specific evidence. Moreover, missing user-specific context is rarely detected systematically, often leading to incomplete or incorrect output.
  We propose NeSy-RAG, a modular neuro-symbolic RAG framework that synthesizes attributable Prolog modules from retrieved text chunks. For each chunk, the system generates semantically meaningful predicates that encode Boolean claims, which may depend on user facts. Using joint natural language-code embeddings, predicates are retrieved and composed into Prolog queries. To address incomplete user context, we introduce a symbolic knowledge-gap detection mechanism that identifies missing user facts whose truth values affect the query outcome and automatically triggers follow-up interactions.
  Executing the resulting Prolog queries yields deterministic answers together with transparent execution traces that link each reasoning step to its originating source. On the ShARC benchmark, without domain-specific training, NeSy-RAG achieves 61.1% accuracy, outperforming a same-model RAG baseline that achieves 42.8% accuracy.

## Full Text


<!-- PDF content starts -->

Proceedings of Machine Learning Research vol 284:1–18, 2026 20th Conference on Neurosymbolic Learning and Reasoning
NeSy-RAG: Neuro-Symbolic RAG for
Explainable Question Answering
Jonas Ganngann@informatik.uni-heidelberg.de
Michael Gertzgertz@informatik.uni-heidelberg.de
Data Science Group, Heidelberg University, Germany
Editors:Alessandra Mileo, Andrea Passerini and Cogan Shimizu
Abstract
Retrieval-augmented generation (RAG) improves question answering by grounding large
language models (LLMs) in external knowledge such as text corpora. However, its reasoning
process remains largely opaque: intermediate reasoning steps are difficult to verify and
cannot be reliably attributed to specific evidence. Moreover, missing user-specific context
is rarely detected systematically, often leading to incomplete or incorrect output.
We proposeNeSy-RAG, a modular neuro-symbolic RAG framework that synthesizes
attributable Prolog modules from retrieved text chunks. For each chunk, the system gen-
erates semantically meaningful predicates that encode Boolean claims, which may depend
on user facts. Using joint natural language–code embeddings, predicates are retrieved and
composed into Prolog queries. To address incomplete user context, we introduce a symbolic
knowledge-gap detection mechanism that identifies missing user facts whose truth values
affect the query outcome and automatically triggers follow-up interactions.
Executing the resulting Prolog queries yields deterministic answers together with trans-
parent execution traces that link each reasoning step to its originating source. On the
ShARC benchmark, without domain-specific training, NeSy-RAG achieves 61.1% accuracy,
outperforming a same-model RAG baseline that achieves 42.8% accuracy.
1. Introduction
RAG combines LLMs with external knowledge and has become a standard approach to
domain-specific question answering. By grounding answer generation in retrieved docu-
ments, RAG systems improve factual accuracy and reduce hallucinations (Shuster et al.,
2021). However, current RAG pipelines lack transparent and accountable reasoning: stan-
dard chain-of-thought (CoT) does not provide trustworthy explanations for LLM reasoning
(Chen et al., 2025), and the attribution of generated answers to source text is not guaranteed
(Zhou et al., 2024). In high-stakes domains such as healthcare or law, these limitations hin-
der trust and safe deployment. Symbolic reasoning systems offer complementary strengths,
including deterministic inference, interpretable reasoning traces, and explicit knowledge
representations. Recent neuro-symbolic approaches leverage LLMs to generate or interact
with symbolic programs, but they often rely on monolithic knowledge bases (Zhang et al.,
2025), manually curated rules (Tan et al., 2025), or dynamically generated logic programs
without clear source attribution (Borazjanizadeh and Piantadosi, 2024; Di et al., 2025).
©2026 J. Gann & M. Gertz.
arXiv:2608.06292v1  [cs.CL]  6 Aug 2026

Gann Gertz
We proposeNeSy-RAG, a neuro-symbolic RAG framework that addresses these limi-
tations through novel Prolog synthesis methods. The main contributions are as follows:
•A modular neuro-symbolic RAG framework that synthesizes and executes attributable
Prolog modules from retrieved text chunks for explainable question answering.
•We introduce0-arity predicate abstraction, which surfaces Prolog modules as Boolean
claims, enabling constrained and scalable query construction via natural language to
code retrieval.
•We propose a symbolicknowledge-gap detectionmechanism that enables the system to
recognize and interactively resolve missing user information via follow-up questions.
•We empirically show that NeSy-RAG outperforms domain-agnostic LLM baselines on
the ShARC (Saeidi et al., 2018) benchmark.
2. Related Work
Recent work has explored integrating symbolic reasoning with LLMs through code synthesis.
Pan et al. (2023) propose a unified framework spanning logic programming, first-order logic,
and constraint satisfaction problems, each realizable by different programming languages.
Particularly Prolog’s declarative nature makes it well-suited for LLM-based synthesis, en-
abling explainable and reliable reasoning with predicate logic.
Fact and Rule Attribution:Yang et al. (2025) highlight the explainability potential of
Prolog by using execution traces as human-interpretable justifications for system outputs,
enhancing trustworthiness and transparency. However, their approach does not address the
attribution of rules and facts to the source knowledge itself. We extend this by ensuring that
every fact and rule can be uniquely mapped to the text chunk from which it was derived.
Complex Context Information:LLM-driven Prolog synthesis has been demonstrated
for question answering (Di et al., 2025; Borazjanizadeh and Piantadosi, 2024). However,
reasoning over complex domains remains challenging, as extracting information for symbolic
reasoning becomes increasingly difficult. Tan et al. (2025) address this by incorporating do-
main knowledge to generate Prolog rules for medical diagnostics, but their approach relies
on manually crafted specifications. Our method operates directly on complex, unstructured
domain knowledge without requiring manual rule selection.
Knowledge Gaps:While prior work on Prolog synthesis assumes that all information
required for inference is available upfront, existing approaches do not address the challenge
of missing user context at inference time. In practice, user-specific information is often
required to provide accurate and personalized responses. Li et al. (2024) show that LLMs
struggle to proactively seek missing information. We address this by proposing automated
knowledge-gap detection that identifies missing user information during symbolic inference
and proactively triggers targeted clarifying questions.
Large Domain Knowledge:Dense retrieval is a well-established mechanism for aggregat-
ing relevant information across large corpora. Zhang et al. (2025) propose replacing dense
retrieval with Prolog-based retrieval, but this can result in very large monolithic knowledge
bases that are difficult to maintain and keep consistent. Our modular approach supports
symbolic reasoning over large-scale domain knowledge without requiring a single unified
2

NeSy-RAG
Figure 1: Overview of the proposed NeSy-RAG framework
knowledge base, enabling selective reasoning over relevant subsets while preserving scala-
bility and symbolic integrity.
Integration:Vakharia et al. (2024) propose ProSLM, a Prolog-based question answering
architecture that uses LLMs to translate natural language between the user and the Pro-
log system, laying the foundation for LLM-assisted Prolog question answering. However,
as knowledge bases grow in size and complexity, formulating suitable Prolog queries that
match both user intent and available symbolic knowledge becomes increasingly difficult. We
address this bottleneck with a hierarchical predicate retrieval strategy that constrains query
generation to a small, question-relevant set of 0-arity predicates, improving both scalability
and query precision.
3. NeSy-RAG Framework
NeSy-RAG is designed as a plug-in extension to existing RAG systems, which are already
widely adopted in practice. We consider a question answering setting in which a RAG
system operates over domain knowledgeD={d(1), . . . , d(M)}consisting ofMtext docu-
ments. Each documentd(i)is segmented into chunks by a chunking algorithm,chunk(d(i)) =
(c(i)
1, . . . , c(i)
Ni),and we denote the complete set of chunks asC=SM
i=1chunk(d(i)). Given a
user questionq, an existing retrieval mechanismret(q,C)⊆ Caims to select a minimal sub-
set of chunks sufficient for answeringq. In practice, however, this subset may be incomplete,
contain irrelevant chunks, or miss relevant ones entirely.
NeSy-RAG takes these retrieved chunks as well as the question as input and proceeds
with the symbolic reasoning pipeline described in the following sections. The overall per-
formance of NeSy-RAG is therefore contingent on the quality of the underlying retrieval
mechanism and domain knowledge. See Appendix B for LLM prompts used in this work.
The questionqis assumed to beBoolean(e.g., “Am I eligible for benefits?”) and
be accompanied by a set of personal contextual informationI={i 1, . . . , i j},comprising
user-specific attributes (e.g., “I am a refugee.”, “I am of legal age.”, “I do not have UK
3

Gann Gertz
citizenship.”). The contextual informationIcan be incomplete, making targeted follow-up
questions to the user necessary to answer the question.
3.1. Attributable Prolog Synthesis
As shown in Figure 1, for each retrieved text chunkc m∈ret(q,C), a Prolog modulep m
is synthesized. The synthesis itself can be performed by an LLM as proposed by Di et al.
(2025) and Borazjanizadeh and Piantadosi (2024) by providing the content of the source
chunk to the LLM. Each Prolog modulep mconsists of rules and factsp m= (R m,Fm),where
RmandF mdenote the rules and facts derived from the text chunkc m, respectively. This
approach reduces the complexity of the synthesis task as each chunk contains only a limited
amount of information. Additionally, the modular approach allows for unique attribution
of rules and facts to specific text chunks. For a given ruler m∈ R mor factf m∈ Fm, the
chunkc mcan be determined as a unique source. As part of the Prolog module system,
reasoning traces not only track facts and rules themselves but also the module (and hence
the source chunk) from which they originated. This enables granular explainability and
verifiability of the reasoning process, as each accessed fact and rule can reference the source
text. Another advantage of this approach is that by modularizing the Prolog knowledge
base, modules can be attached to their respective text chunks during retrieval. This enables
not only the aggregation of chunks based on a question but also the aggregation of relevant
symbolic knowledge. This makes a true scalable utilization of symbolic knowledge possible.
3.2. Knowledge-Gap Detection
Prolog facts can be categorized intouniversal factsanddynamic facts. The distinguishing
characteristic between these two categories is that dynamic facts contain at least one argu-
ment whose value cannot be determined at synthesis time (e.g., ”User is of legal age.”) and
must instead be instantiated during inference, whereas all arguments of universal facts are
fully specified by the domain knowledge (e.g., ”Age threshold to be of legal age is 18.”).
Explicitly separating universal and dynamic facts during Prolog synthesis enables the
integration of external software components that can acquire or validate missing values for
dynamic facts at inference time. Let a fact be defined asf i= (h i, Ai),whereh idenotes
the predicate symbol (e.g.,of legal age) andA iare the arguments of the predicate. Each
argumenta j∈Aiis represented asa j= (n, t, v),wherendenotes the argument name (e.g.,
truth value),tdenotes a datatype (bool,string,integerorfloat), andvdenotes a
value of typet. A fact is classified as adynamic factif, for at least one of its arguments
aj∈Ai, the value cannot be inferred from domain knowledge. For simplicity, we focus on
dynamic facts with one Boolean argument, as they are sufficient for many question answering
tasks and allow for straightforward user interactions (e.g.,of legal age(truth value:
bool = true)). However, the proposed approach can be extended to handle more complex
data types and argument structures.
Given a dynamic factd f iwith a single Boolean argumenta 1, an interaction methodim i
can be generated to acquire or validate the value ofa 1during inference. If the valuevof
a1is not specified in the Prolog fact, the user is asked to provide a value of the appropriate
datatype. If a valuevis already specified as part of a rule condition, the user is asked to
confirm or reject it. For each dynamic factd f i, its corresponding interaction methodim i
4

NeSy-RAG
is called whenever the Prolog system evaluates the fact. The values acquired inim iare
asserted in Prolog and the Boolean return value of the function is forwarded as the result
of the dynamic fact.
3.3. Question Answering
After Prolog modules have been generated, they are used to answer questions as part of
the inference pipeline. As shown in Figure 1, given a user question, relevant text chunks
are retrieved based on their vector embeddings using a retrieval component. In addition to
text chunks, the corresponding Prolog knowledge bases can be attached as metadata. To
answer a natural language question of a user given the retrieved knowledge bases, a Prolog
query must be constructed that matches the user question using available predicates. After
executing the query on the available knowledge bases, a Prolog response as well as an
execution trace are obtained. The Prolog response can be further processed to provide an
easy to understand answer for the user in natural language. The following sections go into
more detail on each of the important subtasks of the question-answering pipeline.
3.3.1. Query Translation
Given retrieved chunks and their Prolog modules, NeSy-RAG must construct a Prolog
query that matches the user question while using only the available predicates. Existing
work translates questions into Prolog by providing an LLM the complete knowledge base
(Vakharia et al., 2024). However, this becomes brittle and unscalable as the knowledge base
grows, since predicates of non-zero arity require the model to inspect the underlying knowl-
edge base to determine valid variable bindings. We instead formulate query construction as
predicate selection and composition, eliminating the need to load the knowledge base during
query generation. To this end, the LLM additionally generates0-arityrules during module
synthesis, exposing the module’s functionality as self-contained Boolean claims:
can_not_get_winter_fuel_payment :-
lives_in_cyprus(true); lives_in_france(true);
lives_in_gibraltar(true); lives_in_greece(true).
At inference time, 0-arity rules are extracted from the modules and ranked by cosine
similarity to the user question using a joint NL–PL embedding model (Kryvosheieva et al.,
2025) fornatural language to code retrieval(NL2C). An LLM is then instructed to compose
a queryonlyfrom the most relevant subset of these rules. This reduces the query search
space and mitigates hallucinations by constraining generation to predefined semantic units.
Additionally, 0-arity rules abstract away from specific variable bindings, making it easier for
the LLM to match user questions to relevant logic without needing detailed knowledge of rule
implementations. This two-stage hierarchical retrieval (first retrieving relevant chunks, then
filtering relevant 0-arity rules within their corresponding modules) ensures that the query-
generating LLM is exposed only to a small, question-relevant subset of the knowledge base.
This is critical since query generation represents the primary scalability bottleneck: unlike
module synthesis, which always operates on a single chunk in isolation, query generation
must jointly handle all retrieved modules. The hierarchical retrieval thus enables scalability
to both large domain corpora (snippet retrieval) and large modules (predicate retrieval).
5

Gann Gertz
3.3.2. Answer Generation
Vakharia et al. (2024) propose to employ an LLM to generate a final user-friendly answer
based on the Prolog result and the initial user question. Additionally, Yang et al. (2025)
propose to take advantage of the traces of Prolog executions as an explanation for the
generated answers. In our approach, the Prolog trace additionally contains information
that can uniquely map each rule and fact to a source chunk and the document it was
extracted from. This enables further verification and attribution of reasoning steps.
4. Experiments
To evaluate the effectiveness of NeSy-RAG, we implemented a prototype and assessed it on
the ShARC (Saeidi et al., 2018) benchmark, a conversational question answering dataset
built from legal text sourced from the UK government website. Each entry consists of a
snippet(a passage from the website), aquestion, ascenariodescribing the user’s situa-
tion, ahistoryof prior follow-up exchanges,evidencecapturing user-specific information
extractable from the scenario, and a targetanswer(yes,no,irrelevantormore).
Rather than supplying all user information upfront, we consolidatescenario,evidence,
andhistoryinto a singlecontextfield that iswithheldfrom the system at the start of
each interaction. This design simulates a setting in which the system must actively recover
missing user information through follow-up questions. Whenever the system poses a follow-
up question, a separate LLM (as part of the evaluation system) answers it by consulting
thecontext, thereby simulating an ideal user. This setup deliberately decouples:
•Object of evaluation: Reasoning process and follow-up question generation based
onsnippetandquestion
•Simulated user: LLM generated answers to follow-up questions based oncontext
Given asnippetand aquestion, the evaluated systems are expected to returnyesor
noonce sufficient user information has been gathered through follow-up questions. If the
simulated user is unable to answer necessary follow-up questions, the system should predict
more, signaling that the answer remains undetermined. If the snippet is unrelated to the
question, the system should predictirrelevant. Table 1 illustrates an example of system
input and output for NeSy-RAG.
The ShARC dataset is provided with training, development, and test splits. Hence, this
dataset is targeted at prediction algorithms that need to be trained for the given domain.
Instead of training on this domain-specific data, the presented approach is domain-agnostic
and does not utilize any prior knowledge. We evaluate on the ShARC test split (8,276
entries). Consistent with the RAG setup described at the beginning of Section 3, the
system receives a candidate set of snippets and must predict an answer. To simulate a
retrieval result without implementing a specific retrieval process, each target snippet is
presented alongside randomly sampled distractor snippets from the test split. Note that the
target snippet itself may be unrelated to the question, in which case the correct prediction
isirrelevantregardless of the distractors. As each ShARC instance relies on a single
relevant snippet, multi-hop retrieval is outside the scope of this evaluation but important
future work.
6

NeSy-RAG
Table 1: Sample input, context and output of NeSy-RAG for a single ShARC instance.
InputSnippet If you don’t claim marriage allowance and
you or your partner were born before 6
April 1935, you may be able to claim mar-
ried couple’s allowance.
Question Could I claim married couple’s allowance?
Context: Do you claim marriage allowance?→No
Were you or your partner born before 6 April 1935?→Yes
OutputProlog Module See Appendix A
Prolog Query?-could claim married couples allowance.
Dynamic Predicates
(part of Prolog module)claims marriage allowance
born before 6april 1935
Follow-up Questions
(answered via context)Claim marriage allowance?=false
Born before 06.04.1935?=true
System AnswerYes
GoldTarget Answer Yes
4.1. Prototype Implementation
We implemented a prototype of NeSy-RAG following the components described in Sec-
tion 3. For each dataset instance, the system receives a set of snippets consisting of the
instance snippet and additional distractors. An LLM classifies each snippet asrelevantor
irrelevant. If no snippet is classified as relevant, the system predictsirrelevant.
For each relevant snippet, the system synthesizes a Prolog module and augments it with
0-arity rules. Prolog modules can be reused for future questions, thus reducing computation
time. To construct a query, all 0-arity rules are embedded and ranked by cosine similarity
to the embedded user question. The top-krules are passed to an LLM, which composes a
Prolog query using only these rules. If the LLM returnsNone, indicating that no suitable
query could be constructed, the system predictsirrelevant.
The generated Prolog modules are loaded intoswipl1and the query is executed. When-
ever a dynamic predicate is encountered during execution, a corresponding follow-up ques-
tion is posed to the simulated user via a foreign Python function implemented usingPySwip2.
If the simulated user is unable to answer a follow-up question, the influence of the unre-
solved predicate on the query outcome is assessed by evaluating it under both Boolean
assignments. If the outcome varies across assignments, the system predictsmore. Other-
wise, the consistent outcome is returned asyesorno. To avoid identical follow-up questions,
the foreign function caches evaluated dynamic predicates and their answers for the ongoing
conversation. This can be extended to storing retrieved information in a user profile.
1. SWI-Prolog:https://www.swi-prolog.org
2. PySwip:https://pypi.org/project/pyswip
7

Gann Gertz
4.2. Baseline Implementations
We compare NeSy-RAG against two LLM baselines that differ in how they access user
information and what snippets they receive. Both baselines predict the four labelsyes,no,
irrelevant, ormore. The structured LLM output functionality of OpenAI and Ollama3is
used to enforce permissible prediction classes.
LLM Upper Baseline (without follow-up questions):We prompt an LLM to
directly predict the target label given the full instance information, including thesingular
targetsnippet, thequestion, and thecontext. Since thecontextcontains all available
information from the user, this baseline does not need to generate follow-up questions.
Therefore, it serves as an approximate upper bound for domain-agnostic LLM performance.
LLM RAG Baseline (with follow-up questions):This baseline mirrors NeSy-
RAG’s setting but replaces Prolog with pure LLM inference. It first selects relevant snippets
from the synthetic retrieval set, then predicts an answer to thequestionusing only those
snippets. If the initial prediction ismore, the system generates a list of follow-up questions
which are answered using the availablecontext. Then the final label is predicted again
while having access to the newly answered follow-up questions.
4.3. Evaluation Results
For the evaluation of the approaches,gpt-oss:20bandgpt-5-miniwere used for LLM
calls, and the code embedding modeljina-code-embeddings-1.5b(Kryvosheieva et al.,
2025) was used for embedding and retrieval of 0-arity predicates in NeSy-RAG.gpt-oss:20b
was run on anNVIDIA RTX 3090using Ollama. The number of retrieved snippets (including
the instance snippet) is 3. Since per-snippet relevance filtering is performed independently,
scaling the number of snippets is not expected to affect accuracy and was not evaluated.
Table 2 presents the evaluation results, with accuracy measured as the ratio of correct
class predictions and the duration as the average time in seconds to process a single dataset
instance. Luo et al. (2023) (BiAE) represents the current state of the art on ShARC and
serves as an upper bound for trained approaches. BiAE receivesscenarioandhistory
directly as input and therefore does not rely on intermediate follow-up questions, whereas
NeSy-RAG must recover equivalent user context interactively.
NeSy-RAG outperforms both LLM RAG baselines with 61.1% accuracy fork= 10
retrieved 0-arity predicates. Smaller and larger values ofkresult in a small drop in accuracy.
Among the RAG baselines, scaling to a larger model yields substantial accuracy gains
(42.8% to 53.7%). NeSy-RAG may similarly benefit from a larger model, which remains to
be explored. Beyond accuracy, NeSy-RAG achieves a lower mean execution time of 7.4s,
compared to 11.4s (gpt-oss:20b) for the RAG baseline. This efficiency stems from three
factors: (1) Prolog modules are synthesized once per snippet and reused across questions,
eliminating redundant LLM reads of source text; (2) query construction operates only over
a small set of retrieved 0-arity predicates rather than the full module, keeping context
windows small; and (3) follow-up questions are triggered algorithmically during Prolog
execution, avoiding the additional LLM call required by the RAG baseline to generate
them. These properties also suggest additional token savings over standard LLM-based
pipelines, though a systematic evaluation remains future work. Taken together, these results
3. Ollama:https://ollama.com/
8

NeSy-RAG
demonstrate that NeSy-RAG can surpass standard LLM-based approaches in both accuracy
and execution time while relying on symbolic reasoning. The LLM upper baselines score
substantially higher, achieving up to 69.9% accuracy, which establishes a ceiling for non-
trained LLM-based question-answering on this dataset. BiAE further raises this ceiling to
77.9%, reflecting the advantage of task-specific training.
Table 2: Evaluation results for the ShARC dataset. NeSy-RAG accuracy is reported for
k=3 / k=10 / k=20 retrieved 0-arity predicates.
Approach Model Accuracy (%) Avg. Duration (s)
LLM RAG Baseline gpt-oss:20b 42.8 11.4
LLM RAG Baseline gpt-5-mini 53.7 25.1
NeSy-RAGgpt-oss:20b 60.0 / 61.1 / 60.4 7.0 / 7.4 / 7.3
LLM Upper Baseline gpt-oss:20b 64.1 2.7
LLM Upper Baseline gpt-5-mini 69.9 5.2
BiAE (trained) - 77.9 -
Figure 2 shows the normalized confusion matrices for NeSy-RAG and both LLM base-
lines, all evaluated usinggpt-oss:20b. 178 NeSy-RAG executions (2.2%) resulted in errors
caused by syntactically invalid Prolog modules or queries. The LLM baselines exhibit spo-
radic errors, caused by occasional failures of the structured output mechanism.
Figure 2: Normalized Confusion Matrices of NeSy-RAG and LLM Baselines
NeSy-RAG correctly classifies 61% ofmoreinstances, exceeding LLM RAG (19%)
and LLM Upper (23%). This gap is strong empirical evidence for NeSy-RAG’s symbolic
knowledge-gap detection mechanism, a core novelty of this work. Both LLM baselines tend
to commit to a definitive answer although essential user information is missing. Onyes
andno, however, LLM Upper correctly classifies a higher proportion of instances (77%
and 85%) than NeSy-RAG (53% and 63%), reflecting the advantage of having complete
9

Gann Gertz
user context available upfront. The LLM RAG baseline additionally exhibits a strong bias
toward predictingirrelevanton answerable questions. This likely is because the model
prematurely rejects snippets as irrelevant rather than generating follow-up questions to
resolve uncertainty.
The dominant error pattern for NeSy-RAG is the misclassification ofyesinstances as
more(31%), suggesting that the synthesis step generates superfluous dynamic predicates
that trigger unnecessary follow-up questions. Tightening dynamic predicate generation is
therefore the most direct path to closing the remaining accuracy gap to LLM Upper. This
conservative bias, however, trades accuracy for safety: in high-stakes domains, responding
withmoreis preferable to false certainty. That said, NeSy-RAG still exhibits false certainty,
misclassifyingmoreinstances asyes(17%) orno(15%), though both rates remain lower
than those observed in either baseline.
5. Discussion and Conclusion
In this work, we presented NeSy-RAG, a novel modular neuro-symbolic RAG framework
that synthesizes and executes attributable Prolog modules from retrieved text chunks for
explainable question answering. By combining dense retrieval with symbolic reasoning,
NeSy-RAG directly addresses three core limitations of standard RAG pipelines: opaque
reasoning without source attribution, unscalable query construction over large knowledge
bases, and the inability to systematically detect missing user information.
The modular Prolog synthesis approach ensures that every fact and rule can be uniquely
traced back to the text chunk from which it was derived, providing granular explainabil-
ity and verifiability that pure LLM-based pipelines cannot offer. The 0-arity predicate
abstraction, combined with joint NL–PL retrieval, enables scalable and constrained query
construction: rather than exposing a query-generating LLM to an entire knowledge base,
NeSy-RAG performs hierarchical retrieval at both the chunk and predicate level, keeping
context windows small and mitigating hallucination. The knowledge-gap detection mecha-
nism allows the system to algorithmically identify missing user information during Prolog
execution and trigger follow-up interactions.
Our evaluation on the ShARC benchmark confirms the effectiveness of the proposed
approach: NeSy-RAG achieves 61.1% accuracy, outperforming the domain-agnostic LLM
RAG baseline (42.8%), while also achieving lower mean execution time owing to module
reusability and smaller context windows. NeSy-RAG correctly classifies 61% ofmorein-
stances compared to 19% for the LLM RAG baseline, providing evidence for the knowledge-
gap detection mechanism. The primary remaining limitation is over-generation of dynamic
predicates, which causes some answerable instances to be conservatively classified asmore.
In high-stakes domains, this conservative bias is arguably preferable to false certainty.
Several directions remain ongoing work. Tightening the constraints on dynamic predi-
cate generation is the most direct path to closing the remaining accuracy gap. Extending
NeSy-RAG to multi-hop and multi-chunk settings would further demonstrate the scalability
of modular Prolog synthesis and inter-module reasoning and attribution. Finally, broad-
ening knowledge-gap detection beyond Boolean predicates would increase applicability to
richer, more varied question answering settings.
10

NeSy-RAG
References
Nasim Borazjanizadeh and Steven Piantadosi. Reliable Reasoning Beyond Natural Lan-
guage.ArXiv, abs/2407.11373, 2024.
Yanda Chen, Joe Benton, Ansh Radhakrishnan, Jonathan Uesato, Carson E. Denison, John
Schulman, Arushi Somani, Peter Hase, Misha Wagner, Fabien Roger, Vladimir Mikulik,
Sam Bowman, Jan Leike, Jared Kaplan, and Ethan Perez. Reasoning Models Don’t
Always Say What They Think.ArXiv, abs/2505.05410, 2025.
Zhengkun Di, Chaoli Zhang, Hongtao Lv, Lizhen Cui, and Lei Liu. LoRP: LLM-based
Logical Reasoning via Prolog.Knowl. Based Syst., 2025.
Daria Kryvosheieva, Saba Sturua, Michael G¨ unther, Scott Martens, and Han Xiao. Efficient
Code Embeddings from Code Generation Models.ArXiv, abs/2508.21290, 2025.
Shuyue S Li, Vidhisha Balachandran, Shangbin Feng, Jonathan S Ilgen, Emma Pierson,
Pang W Koh, and Yulia Tsvetkov. MediQ: Question-Asking LLMs and a Benchmark
for Reliable Interactive Clinical Reasoning.Advances in Neural Information Processing
Systems, 37:28858–28888, 2024.
Yangyang Luo, Shiyu Tian, Caixia Yuan, and Xiaojie Wang. Explicit Alignment and Many-
to-many Entailment Based Reasoning for Conversational Machine Reading. InConference
on Empirical Methods in Natural Language Processing, 2023.
Liangming Pan, Alon Albalak, Xinyi Wang, and William Wang. Logic-LM: Empowering
Large Language Models with Symbolic Solvers for Faithful Logical Reasoning. InFindings
of the Association for Computational Linguistics: EMNLP 2023, pages 3806–3824, 2023.
Marzieh Saeidi, Max Bartolo, Patrick Lewis, Sameer Singh, Tim Rockt¨ aschel, Mike Sheldon,
Guillaume Bouchard, and Sebastian Riedel. Interpretation of Natural Language Rules
in Conversational Machine Reading. InConference on Empirical Methods in Natural
Language Processing, 2018.
Kurt Shuster, Spencer Poff, Moya Chen, Douwe Kiela, and Jason Weston. Retrieval Aug-
mentation Reduces Hallucination in Conversation. InConference on Empirical Methods
in Natural Language Processing, 2021.
Xiaoyu Tan, Bin Li, Weidi Xu, Chao Qu, Wei Chu, Yinghui Xu, Yuan Qi, and Xihe Qiu.
Prolog-Driven Rule-Based Diagnostics with Large Language Models for Precise Clini-
cal Decision Support. InInternational Conference on Medical Image Computing and
Computer-Assisted Intervention, 2025.
Priyesh Vakharia, Abigail Kufeldt, Max Meyers, Ian Lane, and Leilani H. Gilpin. ProSLM:
A Prolog Synergized Language Model for explainable Domain Specific Knowledge Based
Question Answering. InInternational Workshop on Neural-Symbolic Learning and Rea-
soning, 2024.
11

Gann Gertz
Sen Yang, Xin Li, Leyang Cui, Li Bing, and Wai Lam. Neuro-Symbolic Integration Brings
Causal and Reliable Reasoning Proofs.Findings of the Association for Computational
Linguistics, 2025.
Bailing Zhang, Jiajie Li, Kang Peng, Shuchang Zheng, and Kai Meng. Prolog-RAG: A Sym-
bolic Reasoning Approach to Retrieval-Augmented Generation.International Conference
on Computer Engineering and Application, 2025.
Yujia Zhou, Yan Liu, Xiaoxi Li, Jiajie Jin, Hongjin Qian, Zheng Liu, Chaozhuo Li, Zhicheng
Dou, Tsung-Yi Ho, and Philip S. Yu. Trustworthiness in Retrieval-Augmented Generation
Systems: A Survey.ArXiv, abs/2409.10102, 2024.
12

NeSy-RAG
Appendix A. Sample Prolog Knowledge Base
The following is an example of a Prolog module that was created by NeSy-RAG for the text
snippet shown in Table 1.
:- dynamic claims_marriage_allowance/1.
% claims_marriage_allowance(Boolean) indicates whether the user or their
% partner currently claims Marriage Allowance (true) or does not (false).
:- dynamic born_before_6_april_1935/1.
% born_before_6_april_1935(Boolean) indicates whether the user
% or their partner was born before 6 April 1935 (true) or not (false).
% eligible_for_married_couples_allowance is true when the user is not claiming
% Marriage Allowance and either the user or their partner was born before 6 April 1935.
eligible_for_married_couples_allowance :-
claims_marriage_allowance(false),
born_before_6_april_1935(true).
% not_eligible_for_married_couples_allowance is true when the user is already claiming
% Marriage Allowance or neither the user nor their partner is born before 6 April 1935.
not_eligible_for_married_couples_allowance :-
claims_marriage_allowance(true);
born_before_6_april_1935(false).
% could_claim_married_couples_allowance is true when the user may be able to claim
% Married Couple’s Allowance because they are eligible.
could_claim_married_couples_allowance :-
eligible_for_married_couples_allowance.
% can_not_claim_married_couples_allowance is true when the user is not eligible
% to claim Married Couple’s Allowance.
can_not_claim_married_couples_allowance :-
not_eligible_for_married_couples_allowance.
% can_answer_married_couples_allowance_question is true when the eligibility status
% can be determined from provided facts.
can_answer_married_couples_allowance_question :-
claims_marriage_allowance(_),
born_before_6_april_1935(_).
% The user_is_claiming_marriage_allowance rule is true when the stored fact indicates
% that either the user or their partner currently claims Marriage Allowance
user_is_claiming_marriage_allowance :-
claims_marriage_allowance(true).
% The user_is_not_claiming_marriage_allowance rule is true when the stored fact indicates
% that neither the user nor their partner claims Marriage Allowance
user_is_not_claiming_marriage_allowance :-
claims_marriage_allowance(false).
13

Gann Gertz
% The born_before_6_april_1935 rule is true when the stored fact
% indicates that either the user or their partner was born before 6 April 1935
born_before_6_april_1935 :-
born_before_6_april_1935(true).
% The not_born_before_6_april_1935 rule is true when the stored
% fact indicates that neither the user nor their partner was born before 6 April 1935
not_born_before_6_april_1935 :-
born_before_6_april_1935(false).
Appendix B. LLM Prompts
This section presents the LLM prompts used in the approaches presented above.
NeSy-RAG Prolog Generation Prompt
You are an expert in Prolog programming, logic-based knowledge representation,
and legal decision systems. You follow all clean code principles and best practices
for knowledge engineering.
Your task is to transform the following website text into a complete and executable
Prolog knowledge base intended for question answering. The knowledge base should
be extensive and cover all possible user situations and questions that could arise from
the provided text.
GOAL
The knowledge base should allow users to:
•Declare information about their personal situation as required by the source
text to answer all possible questions. Information is declared only using yes/no
facts (e.g.,is minor(true). lives inuk(false).)
•Query whether specific legal conditions apply to them or whether certain gov-
ernmental services, rights, or obligations are available. Evaluation should be
based on user-declared facts. The conditions should exactly reflect the legal
rules and criteria described in the source text.
REQUIREMENTS
1. Clearly separateuser-provided factsfromderived knowledge.
2. All user-provided facts must:
•Be declared as dynamic predicates including a comment that describes the
predicate.
•Be grouped at the beginning of the knowledge base.
•Must be of arity 1.
•User info mustonlyhave onetrue/falseargument — users must only
answer with true or false to add dynamic facts.
•Must contain an expressive and detailed predicate name.
14

NeSy-RAG
•Must only cover the minimum necessary user information for answering
questions.
3. Encode all rules necessary to derive legal outcomes from the user facts.
•Carefully consider if one condition suffices or if multiple conditions must
be combined.
•Make sure for all rules to add the negated scenario (e.g.,is requiredand
isnotrequired).
4. Use clear, consistent predicate names that reflect legal meaning.
OUTPUT CONSTRAINTS
•Outputonlysyntactically correct and complete Prolog code.
•Ensure there are no infinite recursions or loops.
•Do not include explanations, markdown, or text outside of Prolog comments.
•Ensure the knowledge base can be loaded and queried without modification.
SOURCE TEXT TO CONVERT
{{snippet}}
NeSy-RAG - 0-arity Rule Generation Prompt
You are an expert in Prolog programming and knowledge engineering. You receive a
Prolog knowledge base and a matching text snippet of domain knowledge. Your task
is to make rules that are not already 0-arity available as 0-arity rules.
To this end, your task is to generate0-arityProlog rules that work as an interface to
non-0-arity rules that exist in the knowledge base. Your new rules should be usable
as part of a question answering system to answer yes/no questions.
Add a description to each rule that describes the content of the rule as well as the
conditions under which it is fulfilled. Do not copy already existing 0-arity rules,
expect them to be available to the user alongside the rules you generate.
Note that the question answering system infers an answer based on the dynamic
predicates that the user answers with eithertrueorfalse. Your rules must evaluate
correctly according to the information the user provides.
Only answer with the rules and nothing else!
Text snippet from domain knowledge:
{{snippet}}
Existing Prolog knowledge base:
{{kb}}
15

Gann Gertz
NeSy-RAG - Query Generation Prompt
You are an expert in Prolog programming and knowledge engineering. Your task is
to generate a valid Prolog query using the following available rules that matches the
provided question of a user. Answer only with the query and nothing else. Always
include the module names for the predicates in your query. Make sure that you only
use the available 0-arity predicates in your query.
Here are the available rules:
{{zero arity rules}}
Here is the user question you should formulate using the available rules:
{{question}}
NeSy-RAG Evaluation - Predicate Answering Prompt (Simulating User)
Your task is to answer a yes/no dynamic predicate about a user based on the given
user context.
Snippet:
{{snippet}}
Initial user question:
{{question}}
User context:
{{user context}}
All available dynamic predicates:
{{dynamic predicates}}
Answer the following predicate astrue/false, or returnNoneif the information is
not stated (unknown).
Yes/no dynamic predicate to answer:
{{dynamic predicate}}
LLM Baseline (Upper + RAG) - Question Answering Prompt
You receive a text snippet, a user question, and information about the user. Your
task is to answer the question if possible.
•If additional information from the user that asked the question is necessary,
answer with‘more’.
•If the question cannot be answered because the text snippet does not contain
relevant information, answer with‘None’.
16

NeSy-RAG
•If you can provide an answer, respond either with‘true’, meaning the user
question can be answered with “yes”, or with‘false’, meaning the user ques-
tion can be answered with “no”.
Here is the text snippet:
{{snippet}}
Here is the information from the user:
{{user context}}
Here is the user question you should answer:
{{question}}
Note:user contextiscompletein the LLM Upper Baseline andinitially emptyin the
LLM RAG Baseline. The LLM RAG Baseline receives the answers to its follow-up questions
asuser contextin the second pass.
LLM RAG Baseline - Follow-Up Question Generation Prompt
You receive a text snippet, a user question, and context information from the user.
Your task is to generate a list of yes/no follow-up questions that the user should
answer such that we can answer their initial question.
Text snippet:
{{snippet}}
Initial user question:
{{question}}
Known information from the user:
No user context acquired yet.
LLM RAG Baseline Evaluation - Follow-up Question Answering Prompt
You receive a question and information about the user. Your task is to answer the
question based on the information that is provided about the user.
Question:
{{followup question}}
User information:
{{user context}}
Retrieval Snippet Relevance Classification Prompt
You receive a question and a text snippet. Your task is to decide if the text snippet
contains the rules for answering the question. Answer with‘true’if the snippet is
relevant and answer with‘false’if the snippet is not relevant.
17

Gann Gertz
The snippet:
{{snippet}}
The question:
{{question}}
18