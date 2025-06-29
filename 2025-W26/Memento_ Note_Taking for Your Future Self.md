# Memento: Note-Taking for Your Future Self

**Authors**: Chao Wan, Albert Gong, Mihir Mishra, Carl-Leander Henneking, Claas Beger, Kilian Q. Weinberger

**Published**: 2025-06-25 17:37:59

**PDF URL**: [http://arxiv.org/pdf/2506.20642v1](http://arxiv.org/pdf/2506.20642v1)

## Abstract
Large language models (LLMs) excel at reasoning-only tasks, but struggle when
reasoning must be tightly coupled with retrieval, as in multi-hop question
answering. To overcome these limitations, we introduce a prompting strategy
that first decomposes a complex question into smaller steps, then dynamically
constructs a database of facts using LLMs, and finally pieces these facts
together to solve the question. We show how this three-stage strategy, which we
call Memento, can boost the performance of existing prompting strategies across
diverse settings. On the 9-step PhantomWiki benchmark, Memento doubles the
performance of chain-of-thought (CoT) when all information is provided in
context. On the open-domain version of 2WikiMultiHopQA, CoT-RAG with Memento
improves over vanilla CoT-RAG by more than 20 F1 percentage points and over the
multi-hop RAG baseline, IRCoT, by more than 13 F1 percentage points. On the
challenging MuSiQue dataset, Memento improves ReAct by more than 3 F1
percentage points, demonstrating its utility in agentic settings.

## Full Text


<!-- PDF content starts -->

arXiv:2506.20642v1  [cs.CL]  25 Jun 2025Memento: Note-Taking for Your Future Self
Chao Wan∗Albert Gong∗Mihir Mishra Carl-Leander Henneking Claas Beger
Kilian Q. Weinberger
Cornell University
cw862,ag2435,mrm367,ch2273,cbb89,kilian@cornell.edu
Abstract
Large language models (LLMs) excel at reasoning-only tasks, but struggle
when reasoning must be tightly coupled with retrieval—as in multi-hop question
answering. To overcome these limitations, we introduce a prompting strategy
that first decomposes a complex question into smaller steps, then dynamically
constructs a database of facts using LLMs, and finally pieces these facts together
to solve the question. We show how this three-stage strategy, which we call
Memento , can boost the performance of existing prompting strategies across
diverse settings. On the 9-step PhantomWiki benchmark, Memento doubles the
performance of chain-of-thought (CoT) when all information is provided in context.
On the open-domain version of 2WikiMultiHopQA, CoT-RAG with Memento
improves over vanilla CoT-RAG by more than 20 F1 percentage points and over
the multi-hop RAG baseline, IRCoT, by more than 13 F1 percentage points. On
the challenging MuSiQue dataset, Memento improves ReAct by more than 3 F1
percentage points, demonstrating its utility in agentic settings.
1 Introduction
In Christopher Nolan’s film Memento (2000) , the protagonist suffers from short-term memory loss
and cannot form new memories. To function, he meticulously leaves himself notes, tattoos, and
Polaroids—external traces of knowledge that allow him to reason about the world and pursue
long-term goals despite his cognitive limitations. Large language models (LLMs) face a similar
constraint: while adept at local inference, they struggle with global reasoning across extended
sequences, making multi-step problem solving difficult. To overcome this, LLMs need something
akin to what the protagonist of Memento relies on: planning and note-taking strategies that scaffold
fragmented reasoning into coherent computation.
Recent work shows that LLMs often struggle in tasks requiring both retrieval and reasoning, falling
into repetitive loops [ 38,22] or failing to manage state over long contexts [ 39]. While today’s frontier
models can answer isolated questions with remarkable accuracy, they still lack mechanisms for
planning, tracking intermediate steps, and integrating information across multiple reasoning stages
[26].
In this paper, we introduce Memento—a meta-strategy that brings planning and memory into language
model behavior by coordinating multiple LLM calls. Much like the protagonist of Memento, our
method decomposes complex tasks into discrete steps and records the outcome of each step for future
use. Memento operates in three phases: it first generates a symbolic plan grounded in natural language,
then incrementally constructs a Prolog database by resolving each step, and finally executes the
query over the constructed database to produce the answer. Unlike prior work that accumulates large
∗Equal contribution.
Preprint. Under review.

histories across steps [ 24,38,29], each LM call in Memento is limited to a single reasoning action
and conditions only on locally relevant context. This design avoids the long, cluttered histories that
Lo et al. [22] showed can destabilize model behavior, and instead supports more robust, interpretable
long-range reasoning.
We evaluate Memento across three question-answering (QA) settings—in-context,
retrieval-augmented, and agentic—and observe consistent gains over standard prompting
and multi-hop reasoning baselines. The improvements are especially notable on tasks requiring
multiple retrievals, tool use, or deeper intermediate reasoning. Our results suggest that even
strong LLMs benefit from being guided through structured reasoning steps, especially on complex,
multi-step tasks. More broadly, this work shows that introducing explicit planning and intermediate
memory can substantially extend the reasoning capabilities of language models.
2 Preliminaries: Reasoning with Prolog
To represent knowledge in a structured, verifiable way, we use the logic programming language Prolog
[31]. Prolog is especially useful for expressing relationships between entities, and then reasoning
over those relationships to answer questions.
Representing knowledge with Prolog facts. A Prolog fact is a simple statement about the world.
For example, the sentence, “The director of Inception is Chris Nolan,” can be written in Prolog as
director (“Inception”,“Chris Nolan”). This statement says that a relationship called director holds
between the entities “Inception” and “Chris Nolan”. In Prolog terminology, director is apredicate
and “Inception” and “Chris Nolan” are constants . These facts are akin to entries in a structured
database. Much of the real-world knowledge stored in databases like Wikidata are represented in this
way. In fact, as of early 2025, Wikidata contained over 1.65 billion such subject-predicate-object
triples, illustrating the vast scale at which real-world knowledge can be encoded in this structured,
relational form.2
Multi-hop questions as Prolog queries. Prolog doesn’t just store facts—it also allows us to query
them. A Prolog query asks whether certain relationships hold, and if so, what values satisfy them.
Following the literature on QA [ 37], we define the number of hops as the number of distinct paragraphs
needed to answer a question. Consider the following example:
“Who is the main character in the Chris Nolan movie that was released in 2010?” . (1)
Answering this question requires multiple hops: first identifying the movies directed by Chris Nolan,
then filtering for those released in 2010, and finally retrieving the main character of the selected
movie. Each step depends on resolving the previous step(s), and the intermediate space of possible
answers can be large.
A concise way of expressing this multi-hop reasoning is the following Prolog query:
director (A1,“Chris Nolan” ),release_year (A1,2010) ,main_character (A1,A2).(2)
Now that we have the query, the other essential component is the Prolog database —a collection
of facts that is consistent with the predicates used in the query. Traditionally, users provide
a pre-populated Prolog database, and the answer is obtained by directly evaluating the query
against the database. For example, a Prolog database containing the facts, director (“Inception” ,
“Chris Nolan” ),release_year (“Inception” , 2010), and main_character (“Inception” ,“Cobb” ),
yields the following solution to query (2):
A1=“Inception” , A2=“Cobb” .
One of Prolog’s key strengths is its ability to systematically explore allpossible solutions that satisfy
a query. In this case, Prolog can automatically check all Nolan movies, apply the 2010 release filter,
and return every matching main character. In contrast, language models must implicitly learn to
manage this branching behavior, which is difficult in practice—especially when multiple intermediate
steps and constraints are involved [7, 42].
2https://en.wikipedia.org/wiki/Wikidata
2

3 Memento
Our method, Memento, uses this Prolog framework to answer multi-hop questions while still
leveraging the power of LLMs. Memento proceeds in three stages: plan generation ,database
construction , and query execution . In the plan generation stage, an LLM decomposes the given
natural-language question into a sequence of atomic steps, each represented as a Prolog query (2) and
natural-language templates to extract its answer. In the database construction stage, we construct a
Prolog database on the fly using separate calls to LLMs. We utilize LLMs to solve each single-step
question from the generated plan and populate the Prolog database with the answers as facts. In the
query execution stage, with all necessary facts to answer the question now available, we evaluate the
Prolog query against the constructed database to obtain the final answer.
Memento’s design combines the symbolic structure of Prolog with the flexibility of LLMs to achieve
the best of both worlds. We detail each stage in the subsections below.
3.1 Generating a Plan
Prolog Queryq2=release_year(A1, 2010)q1=director(A1, “Chris Nolan”)q3=main_character(A1, A2)target: A2Q: 
?Who is the main character in the Chris 
Nolan movie that was released in 2010 
predicatestatementquestiond1=director(X, Y)What is the mo vie dir ect ed b y Y?The director of X is Y .d2=release_year(X, Y)Which mo vie w as r eleased in y ear Y?X was released in year Y .d3=main_character(X, Y)Who is the main char act er in X?Y  is the main character in X.DefinitionsPl a n  P :
Figure 1: Plan generation for the question in (1)
by Memento. The Prolog query at the top encodes
the reasoning steps needed to answer the question,
with a designated target variable. Each sub-query
qicorresponds to a predicate whose semantics are
grounded in natural language via definitions shown
below, including a question form and a factual
statement template. This structure enables the
system to both interpret and execute the logical
steps using LLM-generated content.We illustrate the process of plan generation
in Fig. 1. Given a multi-step question Q, we
instruct an LLM to translate the question into a
multi-step PlanPwhich consists of a Prolog
query and a set of Definitions . The Prolog
query is executable Prolog code, and should
compute the answer of Qif the database is
populated correctly. The definitions ground the
Prolog query in natural language, and allow the
LLM to populate the database accordingly.
In summary, each Plan Pcontains:
•AProlog query (q1, . . . , q T): a sequence of
Tsub-queries that, when executed against a
database, computes the answer to the original
question.
•ADefinition difor each sub-query qi, which
includes:
–Aquestion template , mapping qito a
natural-language question (e.g., “What is
the movie directed by Chris Nolan?”).
–Astatement template , mapping an
instantiated qito a natural-language fact
(e.g., “The director of Memento is Chris
Nolan.”).
The definitions effectively map each predicate
in the Prolog query to a human-readable
interpretation. For instance, the sub-query director (A1, “Chris Nolan”) is paired with a question
template asking for movies directed by Chris Nolan and a statement template describing a specific
movie’s director.
We prompt the LLM to generate both the Prolog query and the associated definitions in a consistent
way, ensuring that the overall plan faithfully captures the semantics of the original question. This
structured generation allows downstream modules to populate the database and execute the query
reliably. LLMs are particularly well-suited for this task. We observe that even modestly sized models
like Llama-3.3-70B excel at generating such plans. Unlike prior methods that rely on hand-crafted
APIs [ 32] or static, pre-defined programs [ 15], Memento generates plans dynamically, tailoring both
the symbolic steps and their semantics to each question.
3

Q:Who is the main character in the Chris 
Nolan movie that was released in 2010?
3q1 = director(A1, “Chris Nolan”)
q2 = release_year(A1, 2010)
q3 = main_character(A1, A2)target: A2Prolog Query
q2 = release_year(A1, 2010)I s the f ollowing stat ement true or f alse?
 w as r eleased in 2010.
 w as r eleased in 2010.M ement o
Inc ep tionA1 = “Inception”, “Memento”2
What is the mo vie dir ect ed b y Chris N olan?q1 = director( , “Chris Nolan”)A1
What is the name o f the main char act er 
in Inc ep tion?q3 = main_character(A1, )A2A1 = “Inception”3
director(“Inception”, “Chris Nolan”)
director(“Memento”, “Chris Nolan”)1
main_character(“Inception”, “Cobb”)3
release_year(“Inception”, 2010)2
P r o l o g  D a t a b a s ef inal query
A2 =  “Cobb”1Figure 2: Database construction andquery execution for the question in (1). The Prolog query (top
right) consists of three sub-queries q1,q2, andq3, each resolved in order using definitions from the
plan (see Fig. 1). Sub-queries with unbound variables (e.g., q1,q3) trigger fact extraction using the
question template, while fully instantiated sub-queries (e.g., q2) use fact verification based on the
statement template. Extracted or verified facts are added to the Prolog database (bottom right) and
used to resolve future sub-queries. After constructing the database, Memento evaluates the Prolog
query against the completed database in the query execution stage, yielding the target variable A2=
“Cobb” (green arrows).
3.2 Constructing the Prolog Database
We illustrate the database construction and query execution process in Fig. 2, using the plan P
generated in the previous section. Starting from the symbolic Prolog query, Memento iteratively
resolves each sub-query qiin order, using outputs from an LLM. These facts are inserted into a
growing Prolog database and used to evaluate subsequent sub-queries, leading to the final answer.
To resolve each sub-query, Memento retrieves a new value or checks the validity of a statement.
Depending on the system configuration, the LLM accesses information in one of the following ways:
•In-context : the LLM is provided with a short passage containing the relevant information.
•Retrieval-Augmented Generation (RAG) : the LLM is provided with documents retrieved from a
larger corpus.
•Agentic : the LLM obtains answers from external tool outputs via code execution or API calls.
These settings apply to both strategies used during database construction. Based on whether a
sub-query introduces a new variable or checks an already bound one, Memento applies one of two
strategies:
(1) Fact extraction. When a sub-query introduces a new variable (bolded in Fig. 2), Memento uses
the corresponding question template (from the plan definitions) to query an LLM for the unknown
value. For example, for q1=director (A1,“Chris Nolan” ), we instantiate the template to ask:
“What is the movie directed by Christopher Nolan?”. This resembles a single-hop QA task [ 27]. The
LLM’s answer (e.g., “Inception”) is written as a Prolog fact: director (“Inception”, “Chris Nolan”).
(2) Fact verification. When all variables in a sub-query are bound, we check for which variables
the corresponding statement template is true. Given q2=release_year (A1,2010) , we substitute
values for A1(e.g., “Inception”) from prior results and form the statement: “The release year of
Inception is 2010.” We prompt the LLM to verify whether this statement is true. Valid statements are
added as facts; invalid ones are discarded. This process resembles a natural language inference (NLI)
task [4].
4

Returning to our example (2), the first sub-query q1introduces a new variable A1, so we apply fact
extraction using the question template from Definition d1. After substituting the known constant, the
model is prompted with: “What is the movie directed by Christopher Nolan?”. Answers are added to
the database as Prolog facts (see Fig. 2).
To resolve q2, we query the database with q1for candidate values of A1(e.g., “Inception”, “Memento”).
Since q2has no unbound variables after substitution, we apply fact verification. For each value of A1,
we construct the corresponding statement and ask the LLM to assess its truth. Only valid statements
(e.g., “Inception was released in 2010.”) are stored as facts while invalid ones (e.g., “Memento was
released in 2010.”) are ignored.
Finally, for q3=main_character (A1,A2), we again query the database with the partial query
(q1, q2)to retrieve valid bindings for A1. Since A2is still unbound, we apply fact extraction again,
generating the question: “Who is the main character in Inception?”. The answer (e.g., “Cobb”)
resolves the last sub-query.
In general, Memento alternates between fact extraction and verification, guided by the current
sub-query and available bindings, until all sub-queries are resolved.
3.3 Executing the Prolog Query
To answer the original question, we query the Prolog database with the full query (q1, q2, q3)(see
green arrow in Fig. 2). The final query returns A1= “Inception”, A2= “Cobb” and extracting the
value corresponding to the target variable A2yields the final answer, “Cobb”.
Note that Prolog provides additional predicates for aggregations and comparisons. For example, if
our original question had been, “Which movie was released first, Inception or Memento?”, we would
only need to extract the release years of Inception and Memento (via the fact extraction strategy).
Comparing the release years can then be done deterministically using the >@operator in Prolog. For
questions involving many steps, offloading reasoning to Prolog is often more reliable than using
natural language to reason.
4 Experiments
We evaluate Memento on three real-world datasets and one synthetic benchmark, detailed below. For
all experiments, we use the instruction-tuned version of the Llama-3.3-70B model due to its strong
baseline reasoning and tool-calling capabilities. We use greedy decoding throughout.
4.1 Datasets
We chose datasets that require LMs to perform varying degrees of multi-hop reasoning.
HotpotQA (HP). HotpotQA [ 37] is a multi-hop question answering dataset designed to test a
model’s ability to reason over multiple documents. It contains over 100,000 manually curated
questions, and each question typically requires information from two Wikipedia paragraphs to answer,
making HP a 2-hop QA dataset.
2WikiMultiHopQA (2Wiki). A more recent 2-hop dataset, 2WikiMultiHopQA [ 8], contains over
190,000 questions categorized into compositional, inference, comparison, and bridge-comparison
types. Each question is grounded in a two-hop path from the Wikidata knowledge graph.
MuSiQue (MSQ). MuSiQue [ 33] is a multi-hop question answering dataset designed to emphasize
compositional reasoning. It contains questions constructed by composing 2-4 single-hop queries ,
ensuring that correct answers require integrating information from multiple, disjoint passages. We
use the MuSiQue-Answerable version, where questions are guaranteed to be answerable using a
specific subset of retrieved context.
PhantomWiki (PW). PhantomWiki [ 7] is a synthetic dataset generator that creates universes of
fictional personas with aligned facts, documents, and question-answer pairs. Questions ask about
social relations and attributes, which can be composed to achieve arbitrary reasoning complexity. We
use three universe sizes—50, 500, and 5000—denoted PW-S, PW-M, and PW-L, with three instances
per size. We generate questions requiring up to 9 reasoning steps , significantly more than in the
5

above Wikipedia-based datasets, making PhantomWiki a strong benchmark for multi-hop reasoning.
See App. D.1 for details about the dataset generation process.
4.2 Experimental setup
In the following experiments, we consider three key settings: in-context, retrieval-augmented,
and agentic setting. In the in-context setting , models receive relevant passages as input; in the
retrieval-augmented setting , models generate retrieval queries and use a fixed retriever to fetch
context from a larger corpus. In the agentic setting , models interact with an external environment via
tool calls, dynamically gathering information to answer questions. We detail each setup below.
In-context Setting. In the in-context setting, models are given the relevant passages as input,
removing the need for retrieval. While this is the easiest setting, long contexts result in models struggle
to gather relevant information and reason through it [ 2,18]. For HotpotQA and 2WikiMultiHopQA,
we follow the standard distractor setting, where each question is paired with 10 passages: a small
number of gold supporting paragraphs and several distractors. The distractor setting of MuSiQue
pairs each question with 20 total paragraphs. For PhantomWiki, we include the entire corpus as input
only for the PW-S and PW-M instances, where the corpus fits within the model’s context window.
Retrival-augmented Setting. In the retrieval-augmented setting, models must first retrieve relevant
passages from a large corpus before answering each question. We use BM25 [ 28] as the retriever
across all datasets. For HotpotQA, 2WikiMultiHopQA, and MuSiQue, we adopt the preprocessed
wiki18-100w corpus from FlashRAG3[13]. This corpus reflects the December 20, 2018 snapshot
of Wikipedia. For PhantomWiki, we use the the corpus generated with each fictional universe for
retrieval.
Agentic Setting. In the agentic setting, the model operates as a tool-augmented reasoning agent,
interacting with an environment via two external tools same as in ReAct [ 38]:search and
lookup . The search(title) tool fetches the full article associated with a given title, while
thelookup(keyword) tool searches within the most recently retrieved articles and returns the
sentence containing the specified keyword, if it exists. The model can issue sequences of tool calls to
incrementally gather information and answer the question.
For HotpotQA, 2WikiMultiHopQA, and MuSiQue, we report exact match andF1 score as introduced
by Rajpurkar et al. [27] on 500 randomly sampled questions from the publicly available dev splits. For
PhantomWiki, we use the answer-level F1 score to account for questions with multiple ground-truth
answers.
5 In-Context Reasoning
We compare Memento with two natural baselines: Vanilla Chain-of-Thought (CoT) andZeroshot
prompting. While CoT encourages multi-step reasoning, longer context lengths make question
answering more challenging, since the model must identify and reason over scattered, relevant facts
[21,35]. Our evaluation focuses on how well each prompting strategy supports robust multi-hop
reasoning in the presence of long or noisy contexts. The results are shown in table Tab. 1. We
evaluate three variants of Memento: (1) Vanilla Memento , (2) Memento →CoT, which falls back to
CoT when Memento fails, and (3) CoT→Memento , which falls back to Memento when CoT fails.
These variants explore how Memento can complement or be complemented by standard prompting
strategies, depending on the dataset and task.
1. On questions with short contexts and few reasoning steps, CoT already performs strongly,
leaving limited room for Memento to improve. We first examine HP and 2Wiki, which involve
relatively short contexts and typically require only two reasoning steps. In these cases, CoT achieves
strong performance—76.9 F1 on HP and 80.8 on 2Wiki—indicating that the reasoning challenge
is manageable without additional mechanisms. Since both the planning and execution stages in
Memento are straightforward for such questions, the benefits of symbolic decomposition are minimal.
Our variant Memento →CoT performs on par or below CoT alone, whereas CoT →Memento
achieves +0.4 and +1.2 F1 percentage point gains on HP and 2Wiki, respectively. These results
3https://github.com/RUC-NLPIR/FlashRAG (MIT License)
6

Table 1: In-context setting. We report exact match (EM) and F1 scores (%) on 500 dev samples
each from HotpotQA (HP), 2WikiMultiHopQA (2Wiki), and MuSiQue (MSQ), and mean F1 score
±1 standard error across 3 dataset generation seeds for PhantomWiki (PW-S, PW-M). We use
Llama-3.3-70B for all methods and datasets. See Sec. 5 for detailed discussion.†For comparison, we
include the numbers for the supervised SOTA [40] (on test splits).
HP 2Wiki MSQ PW-S PW-M
Method EM F1 EM F1 EM F1 F1 F1
Zeroshot 25.6 38.9 28.6 39.2 18.8 9.0 39.1±1.6 22.3±1.7
CoT 60.6 76.9 73.2 80.8 50.0 63.3 75.3±1.2 34.8±2.3
Memento (w/o CoT) 43.2 58.3 57.2 64.6 37.4 49.6 91.2±1.6 56.0±0.6
Memento →CoT 50.6 67.9 66.4 74.8 49.0 65.2 92.4±1.4 57.8±0.9
CoT→Memento 60.8 77.29 74.4 82.0 50.2 63.6 77.1±1.2 39.0±2.6
Supervised SOTA†72.7 85.0 88.5 90.9 69.2 91.4 N/A
suggest that for relatively simple multi-hop questions, CoT alone is sufficient to arrive at correct
answers effectively.
2. Memento is particularly effective in solving long-context, multi-step question-answering
tasks, while CoT struggles. The fact that MSQ requires more reasoning steps (2-4) on average than
HP makes it a more challenging benchmark for multi-hop reasoning. In this setting, we achieve the
best performance when Memento serves as the main method and CoT as the fallback, further boosting
F1 by +2 percentage points compared to CoT. The benefit of Memento becomes even more apparent
on more challenging dataset like PW. Even with just 50 documents (PW-S), CoT struggles with
questions that require many reasoning steps (as we later show in Fig. 3). This issue is exacerbated
at scale: When the corpus size increases tenfold to PW-M ( n= 500 ), performance with CoT drops
significantly to 34.8 F1. In this scenario, Memento alone achieves a striking 56.0 F1, far surpassing
Zeroshot (22.3) and CoT (34.8). When enhanced with CoT (Memento →CoT), performance climbs
further to 57.8 F1. These results suggest that Memento excels not only at decomposing complex
reasoning steps but also at navigating long, information-dense contexts. In contrast, CoT struggles
to maintain coherence in long-context settings, and zeroshot prompting fails to consistently retrieve
or integrate the necessary facts. Together, these findings underscore the value of symbolic structure
when both retrieval and multi-step reasoning are required.
6 Retrieval- and Tool-Augmented Reasoning
When the corpus is too large to fit in context, retrieval-augmented generation (RAG) methods first
retrieve a set of relevant documents to help answer the question. Agent-based approaches extend this
by allowing multi-step tool use. We evaluate three variants in both settings: (1) Vanilla Memento ,
(2)Memento →CoT/ReAct , and (3) CoT/ReAct →Memento , where either method acts as fallback.
Results are shown in Tab. 2. We additionally include two multi-hop RAG baselines— Self-Ask [1]
andIRCoT [34]—which interleave reasoning with retrieval.
1. When augmented with retrival, Memento is more effective than CoT in handling multi-step
reasoning tasks. For HP, Vanilla Memento underperforms CoT in the RAG setting, but CoT →
Memento improves F1 from 49.7 to 53.7, showing that Memento can effectively resolve questions
where CoT fails. In 2Wiki, CoT →Memento surpasses CoT without evidence (see App. A.4 for
details) by 5.0 F1 and CoT+BM25 by a substantial 20.2 F1. This gap suggests that Llama-3.3-70B’s
internal knowledge is often sufficient, and retrieval only becomes useful when combined with stronger
reasoning strategies like Memento. On MSQ, where reasoning depth is higher, CoT →Memento
outperforms CoT+BM25 by 3.4 EM and 6.2 F1 at k= 8, with improvements holding across values
ofk(Fig. 4). Finally, PW highlights the strength of structured reasoning in long-context settings.
Memento +BM25 outperforms CoT+BM25 by 43.2 F1 points at n= 500 and by 48.2 points at
n= 5000 . Remarkably, this even outperforms the in-context setting—despite all content fitting in
the context window—underscoring the effectiveness of structured retrieval and stepwise execution.
7

Table 2: Open-Domain Setting. on 500 dev samples each from HotpotQA (HP), 2WikiMultihopQA
(2Wiki), and MuSiQue (MSQ), and mean F1 score ±standard deviation across 3 dataset generation
seeds for PhantomWiki (PW-M, PW-L). (Due to time constraint,⋆runs are only evaluated on seed 1.)
We use Llama-3.3-70B for all methods and datasets. For the BM25 retriever, we retrieve k= 14 ,16,
8, and 4documents per query for HP, 2Wiki, MSQ, and the PW datasets, respectively. See Sec. 6 for
detailed discussion.
HP 2Wiki MSQ PW-M PW-L
Method EM F1 EM F1 EM F1 F1 F1
RAG setting (w/ BM25 retriever)
CoT 37.4 49.7 31.0 35.0 9.8 15.7 38.2 ± 2.4 30.6±1.1
Self-Ask 19.2 28.0 15 21.78 5.6 9.1 24.4⋆23.22⋆
IRCoT 40.4 52.9 32.4 42.5 17.6 24.5 52.2±1.8 43.2±0.9
Memento (w/o CoT) 24.8 37.1 43.2 50.2 10.8 18.2 81.4±3.7 78.8⋆
Memento→CoT 31.8 46.6 47.0 54.7 11 18.4 83.2±1.7 79.2⋆
CoT→Memento 40.4 53.7 49.0 55.2 13.2 21.9 55.1±1.7 52.4⋆
Agentic setting (w/ tools)
ReAct 36 49.5 39.6 45.2 17.8 26.4 53.3±2.4 44.6±1.0
Memento (w/o CoT) 21.6 31.7 35.8 41.9 15.4 22.9 68.7±1.5 65.3⋆
Memento→ReAct 29.4 41.5 40.2 46.4 16.2 24.2 81.8±0.9 78.5⋆
ReAct→Memento 38.6 53.0 47.0 53.2 19.4 29.6 67.7±0.9 61.5⋆
Parametric setting (w/o evidence)
Standard 25.0 38.27 29.0 40.28 10.2 19.9N/A
CoT 35.4 49.33 42.6 50.2 16.4 27.4
Comparison to RAG baselines. On most datasets, CoT →Memento outperforms multi-round
retrieval baselines like Self-Ask and IRCoT (detailed in App. D.2), likely due to Memento’s
structured symbolic plan that reduces error across steps. However, natural language methods like
IRCoT offer more flexibility, and on datasets like MSQ—where query reformulation demands
adaptability—IRCoT performs slightly better.
2. In the agentic setting, Memento enhances tool-using agents like ReAct by improving
success on harder multi-step reasoning tasks. Combining Memento with ReAct yields consistent
improvements across datasets. The most significant performance gains are seen on 2Wiki and PW-L,
with F1 score improvements of 8.0 and 33.9, respectively, compared to ReAct. On HP, 2Wiki, and
MSQ, ReAct →Memento achieves the highest EM and F1 scores, suggesting that Memento can
successfully answer questions where ReAct fails. These results indicate that symbolic planning and
execution from Memento can complement tool-augmented reasoning, especially when combined
with the dynamic interaction capabilities of agents like ReAct.
7 Ablation Analysis
To understand how Memento improves reasoning performance, we separate the plan generation stage
(Sec. 3.1) from the database construction stage (Sec. 3.2). We use PhantomWiki with a small corpus
size (n=50), which fits well within Llama-3.3-70B’s context window and isolates reasoning ability
without retrieval. Since PhantomWiki questions require up to nine reasoning steps, this setup allows
us to evaluate how performance scales with complexity. Similar analyses can be extended to retrieval
and tool-based settings.
Llama-3.3-70B excels at planning. Using the plan generation prompt (App. A.1), we prompt
Llama-3.3-70B to generate Prolog queries and definitions. We call this method Plan⋆in Fig. 3.
Unlike the other datasets, PhantomWiki provides ground-truth Prolog facts, allowing us to directly
8

evaluate the quality of these LLM-generated Prolog queries. In this oracle setting, Llama-3.3-70B
achieves near-perfect accuracy, suggesting it is highly capable at symbolic decomposition. The
accuracy of Plan⋆also serves as an upper bound for Memento as the database construction stage can
only introduce additional errors.
1 3 5 7 9
Reasoning difficulty0.250.50.751F1
CoTExecute
Plan
Memento
Figure 3: F1 scores (%) versus difficulty, as
measured by number of reasoning steps. We use
Llama-3.3-70B for all methods (orange)⋆Methods
that use oracle information. See Sec. 7 for detailed
discussion.Llama-3.3-70B struggles with execution in
natural language. We next evaluate execution
by providing the model with a gold plan
and measuring its ability to answer step by
step—denoted Execute⋆. The prompt, included
in App. A.5, mirrors that of Least-to-Most
[41], except we omit the original question to
focus on evaluating the model’s plan-following
ability. As shown in Fig. 3, Execute⋆
improves substantially over CoT, confirming
that structured guidance helps. Notably,
Memento consistently outperforms Execute⋆
across most difficulty levels, suggesting that
symbolic execution with Prolog offers additional
benefits beyond step-by-step natural language
reasoning.
Planning format matters. These findings
confirm that LLMs are effective at generating
high-quality decompositions, addressing the
open question raised by Patel et al. [25].
However, they also reveal that the format of the
plan plays a critical role. While natural language
plans—as used in Execute⋆—provide a clear performance boost, symbolic representations like Prolog
(as in Memento) enable more precise and compositional reasoning, especially as the complexity of
the task increases.
8 Related Works
Planning and execution in other domains. Planning followed by stepwise execution is a
long-standing idea explored across several domains. In robotics, models use formal languages
like PDDL to plan and execute actions [ 19,14]. In web agents, planning-execution separation has
been used to control high-level behavior [ 6]. Program synthesis also emphasizes structured execution,
with models learning to simulate or reason over code behavior [ 20,3]. For question answering, Wu
and Liu [36] translate statements and questions into Prolog using definite clause grammar parsing,
which is less flexible than using an LLM.
Interleaving reasoning with retrieval. Vanilla RAG pipelines are poorly suited for multi-hop QA,
where reasoning must guide retrieval over multiple steps. To address this, Self-Ask [ 26] prompts the
model to generate follow-up questions; IRCoT [ 34] uses intermediate thoughts as retrieval queries;
and ReAct [ 38] combines reasoning with tool use in a step-by-step loop. Other retrieval-aware
reasoning methods include Multi-hop RAG [ 17], Fusion-in-Decoder [ 11], etc. Self-RAG [ 1]
fine-tunes retrieval and generation jointly. EfficientRAG [ 43] introduces an efficient retriever for
multi-hop question-answering. ReSP [ 12] improves on this by summarizing retrieved content at each
step, mitigating context overload and avoiding repeated planning. In contrast, Memento follows
an explicit symbolic plan, allowing LLMs to reason through multi-hop questions with structured
retrieval and execution.
Improving reasoning with note-taking. To overcome the limitations of in-context reasoning in
LLMs—particularly on multi-step tasks where all relevant information is present but hard to integrate
[39]—recent work has explored note-taking and summarization as lightweight memory mechanisms.
Some approaches focus on improving attention and memory across long contexts [ 21,5,23], while
others track intermediate reasoning state explicitly. For example, Lanchantin et al. [16] propose
interleaving reasoning traces with state tokens to improve logic and arithmetic performance. In
9

contrast, our method focuses on QA-specific state tracking: capturing only the facts acquired so far
in a symbolic database that supports structured execution and interpretable reasoning.
9 Conclusions & Limitations
We introduce Memento, a prompting strategy for improving multi-step question answering. Using
Prolog, Memento operationalizes planning and note-taking in the context of language models.
Rather than relying on a single LLM to perform all reasoning and retrieval, Memento leverages
multiple LLMs—with Prolog serving as the glue to coordinate these separate steps. Across the three
settings—in-context, RAG, and agentic—Memento provides consistent gains when either used as
a standalone strategy or used as an enhancement to existing prompting strategies. Improvements
are especially pronounced on harder datasets, such as MuSiQue and PhantomWiki, where questions
require many reasoning steps.
While an effective approach to multi-step question answering, Memento also comes with several
limitations. First, we rely exclusively on the model’s inherent reading comprehension capabilities
to extract and verify facts. As a result, our system is not immune to hallucinated or unsupported
statements. Future work could involve incorporating external verification mechanisms to ensure that
facts written into the database are truly grounded in the corpus. Second, Memento currently follows
a fixed forward execution path—if a step fails, there is no built-in recovery mechanism. Future
work could explore integrating fact-checking modules and dynamic execution strategies that invoke
alternative tools or retrieval methods to recover from intermediate failures.
Acknowledgments and Disclosure of Funding
CW is supported by the National Science Foundation (NSF) OAC-2118310 and NSF-1934714 grant.
This work was partially supported by funding from NewYork-Presbyterian for the NYP-Cornell
Cardiovascular AI Collaboration.
References
[1]Akari Asai, Zeqiu Wu, Yizhong Wang, Avirup Sil, and Hannaneh Hajishirzi. Self-rag:
Learning to retrieve, generate, and critique through self-reflection. In The Twelfth International
Conference on Learning Representations , 2023. (Cited on pages 7 and 9.)
[2]Yushi Bai, Xin Lv, Jiajie Zhang, Hongchang Lyu, Jiankai Tang, Zhidian Huang, Zhengxiao
Du, Xiao Liu, Aohan Zeng, Lei Hou, et al. Longbench: A bilingual, multitask benchmark for
long context understanding. In Proceedings of the 62nd Annual Meeting of the Association for
Computational Linguistics (Volume 1: Long Papers) , pages 3119–3137, 2024. (Cited on page 6.)
[3]Claas Beger and Saikat Dutta. Coconut: Structural code understanding does not fall out of a
tree, 2025. URL https://arxiv.org/abs/2501.16456 .(Cited on page 9.)
[4]Samuel R Bowman, Gabor Angeli, Christopher Potts, and Christopher D Manning. The snli
corpus. 2015. (Cited on page 4.)
[5]Zihang Dai, Zhilin Yang, Yiming Yang, Jaime G Carbonell, Quoc Le, and Ruslan Salakhutdinov.
Transformer-xl: Attentive language models beyond a fixed-length context. In Proceedings of
the 57th Annual Meeting of the Association for Computational Linguistics , pages 2978–2988,
2019. (Cited on page 9.)
[6]Lutfi Eren Erdogan, Nicholas Lee, Sehoon Kim, Suhong Moon, Hiroki Furuta, Gopala
Anumanchipalli, Kurt Keutzer, and Amir Gholami. Plan-and-act: Improving planning of
agents for long-horizon tasks. arXiv preprint arXiv:2503.09572 , 2025. (Cited on page 9.)
[7]Albert Gong, Kamil ˙e Stankevi ˇci¯ut˙e, Chao Wan, Anmol Kabra, Raphael Thesmar, Johann Lee,
Julius Klenke, Carla P Gomes, and Kilian Q Weinberger. Phantomwiki: On-demand datasets
for reasoning and retrieval evaluation. arXiv preprint arXiv:2502.20377 , 2025. (Cited on pages 2
and 5.)
10

[8]Xanh Ho, Anh-Khoa Duong Nguyen, Saku Sugawara, and Akiko Aizawa. Constructing a
multi-hop qa dataset for comprehensive evaluation of reasoning steps. In Proceedings of the
28th International Conference on Computational Linguistics , pages 6609–6625, 2020. (Cited on
page 5.)
[9]Patrick Hohenecker and Thomas Lukasiewicz. Ontology reasoning with deep neural networks.
Journal of Artificial Intelligence Research , 68:503–540, 2020. (Cited on page 16.)
[10] John E Hopcroft, Rajeev Motwani, and Jeffrey D Ullman. Introduction to automata theory,
languages, and computation. Acm Sigact News , 32(1):60–65, 2001. (Cited on page 18.)
[11] Gautier Izacard and Édouard Grave. Leveraging passage retrieval with generative models for
open domain question answering. In Proceedings of the 16th Conference of the European
Chapter of the Association for Computational Linguistics: Main Volume , pages 874–880, 2021.
(Cited on page 9.)
[12] Zhouyu Jiang, Mengshu Sun, Lei Liang, and Zhiqiang Zhang. Retrieve, summarize, plan:
Advancing multi-hop question answering with an iterative approach. In Companion Proceedings
of the ACM on Web Conference 2025 , pages 1677–1686, 2025. (Cited on page 9.)
[13] Jiajie Jin, Yutao Zhu, Zhicheng Dou, Guanting Dong, Xinyu Yang, Chenghao Zhang, Tong Zhao,
Zhao Yang, and Ji-Rong Wen. Flashrag: A modular toolkit for efficient retrieval-augmented
generation research. In Companion Proceedings of the ACM on Web Conference 2025 , pages
737–740, 2025. (Cited on pages 6 and 19.)
[14] Subbarao Kambhampati, Karthik Valmeekam, Lin Guan, Mudit Verma, Kaya Stechly, Siddhant
Bhambri, Lucas Paul Saldyt, and Anil B Murthy. Position: Llms can’t plan, but can help
planning in llm-modulo frameworks. In Forty-first International Conference on Machine
Learning , 2024. (Cited on page 9.)
[15] Omar Khattab, Keshav Santhanam, Xiang Lisa Li, David Hall, Percy Liang, Christopher Potts,
and Matei Zaharia. Demonstrate-search-predict: Composing retrieval and language models for
knowledge-intensive nlp. arXiv preprint arXiv:2212.14024 , 2022. (Cited on page 3.)
[16] Jack Lanchantin, Shubham Toshniwal, Jason Weston, Sainbayar Sukhbaatar, et al. Learning to
reason and memorize with self-notes. Advances in Neural Information Processing Systems , 36:
11891–11911, 2023. (Cited on page 9.)
[17] Patrick Lewis, Yuxiang Wu, Linqing Liu, Pasquale Minervini, Heinrich Küttler, Aleksandra
Piktus, Pontus Stenetorp, and Sebastian Riedel. Paq: 65 million probably-asked questions and
what you can do with them. Transactions of the Association for Computational Linguistics , 9:
1098–1115, 2021. (Cited on page 9.)
[18] Tianle Li, Ge Zhang, Quy Duc Do, Xiang Yue, and Wenhu Chen. Long-context llms struggle
with long in-context learning. arXiv preprint arXiv:2404.02060 , 2024. (Cited on page 6.)
[19] Bo Liu, Yuqian Jiang, Xiaohan Zhang, Qiang Liu, Shiqi Zhang, Joydeep Biswas, and Peter
Stone. Llm+ p: Empowering large language models with optimal planning proficiency. arXiv
preprint arXiv:2304.11477 , 2023. (Cited on page 9.)
[20] Changshu Liu, Shizhuo Dylan Zhang, Ali Reza Ibrahimzada, and Reyhaneh Jabbarvand.
Codemind: A framework to challenge large language models for code reasoning, 2024. URL
https://arxiv.org/abs/2402.09664 .(Cited on page 9.)
[21] Nelson F Liu, Kevin Lin, John Hewitt, Ashwin Paranjape, Michele Bevilacqua, Fabio Petroni,
and Percy Liang. Lost in the middle: How language models use long contexts. Transactions of
the Association for Computational Linguistics , 12:157–173, 2024. (Cited on pages 6 and 9.)
[22] Robert Lo, Abishek Sridhar, Frank F Xu, Hao Zhu, and Shuyan Zhou. Hierarchical
prompting assists large language model on web navigation. In Findings of the Association for
Computational Linguistics: EMNLP 2023 , pages 10217–10244, 2023. (Cited on pages 1 and 2.)
11

[23] Grégoire Mialon, Roberto Dessì, Maria Lomeli, Christoforos Nalmpantis, Ram Pasunuru,
Roberta Raileanu, Baptiste Rozière, Timo Schick, Jane Dwivedi-Yu, Asli Celikyilmaz, et al.
Augmented language models: a survey. arXiv preprint arXiv:2302.07842 , 2023. (Cited on
page 9.)
[24] Maxwell Nye, Anders Johan Andreassen, Guy Gur-Ari, Henryk Michalewski, Jacob Austin,
David Bieber, David Dohan, Aitor Lewkowycz, Maarten Bosma, David Luan, et al. Show
your work: Scratchpads for intermediate computation with language models. arXiv preprint
arXiv:2112.00114 , 2021. (Cited on page 2.)
[25] Pruthvi Patel, Swaroop Mishra, Mihir Parmar, and Chitta Baral. Is a question decomposition
unit all we need? In Proceedings of the 2022 Conference on Empirical Methods in Natural
Language Processing , pages 4553–4569, 2022. (Cited on page 9.)
[26] Ofir Press, Muru Zhang, Sewon Min, Ludwig Schmidt, Noah A Smith, and Mike Lewis.
Measuring and narrowing the compositionality gap in language models. In Findings of the
Association for Computational Linguistics: EMNLP 2023 , pages 5687–5711, 2023. (Cited on
pages 1 and 9.)
[27] Pranav Rajpurkar, Jian Zhang, Konstantin Lopyrev, and Percy Liang. Squad: 100,000+ questions
for machine comprehension of text. In Proceedings of the 2016 Conference on Empirical
Methods in Natural Language Processing , pages 2383–2392, 2016. (Cited on pages 4 and 6.)
[28] Stephen Robertson, Hugo Zaragoza, et al. The probabilistic relevance framework: Bm25 and
beyond. Foundations and Trends ®in Information Retrieval , 3(4):333–389, 2009. (Cited on
page 6.)
[29] Timo Schick, Jane Dwivedi-Yu, Roberto Dessì, Roberta Raileanu, Maria Lomeli, Eric Hambro,
Luke Zettlemoyer, Nicola Cancedda, and Thomas Scialom. Toolformer: Language models can
teach themselves to use tools. Advances in Neural Information Processing Systems , 36, 2024.
(Cited on page 2.)
[30] Yijia Shao, Yucheng Jiang, Theodore Kanell, Peter Xu, Omar Khattab, and Monica Lam.
Assisting in writing wikipedia-like articles from scratch with large language models. In
Proceedings of the 2024 Conference of the North American Chapter of the Association for
Computational Linguistics: Human Language Technologies (Volume 1: Long Papers) , pages
6252–6278, 2024. (Cited on page 18.)
[31] Leon Sterling and Ehud Y Shapiro. The art of Prolog: advanced programming techniques . MIT
press, 1994. (Cited on pages 2 and 18.)
[32] Dídac Surís, Sachit Menon, and Carl V ondrick. Vipergpt: Visual inference via python execution
for reasoning. In Proceedings of the IEEE/CVF International Conference on Computer Vision ,
pages 11888–11898, 2023. (Cited on page 3.)
[33] Harsh Trivedi, Niranjan Balasubramanian, Tushar Khot, and Ashish Sabharwal. MuSiQue:
Multihop questions via single-hop question composition. Transactions of the Association for
Computational Linguistics , 10:539–554, 2022. (Cited on page 5.)
[34] Harsh Trivedi, Niranjan Balasubramanian, Tushar Khot, and Ashish Sabharwal. Interleaving
retrieval with chain-of-thought reasoning for knowledge-intensive multi-step questions. In
Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics
(Volume 1: Long Papers) , pages 10014–10037, 2023. (Cited on pages 7, 9, and 19.)
[35] Minzheng Wang, Longze Chen, Cheng Fu, Shengyi Liao, Xinghua Zhang, Bingli Wu, Haiyang
Yu, Nan Xu, Lei Zhang, Run Luo, et al. Leave no document behind: Benchmarking long-context
llms with extended multi-doc qa. arXiv preprint arXiv:2406.17419 , 2024. (Cited on page 6.)
[36] Katherine Wu and Yanhong A Liu. Lp-lm: No hallucinations in question answering with logic
programming. arXiv preprint arXiv:2502.09212 , 2025. (Cited on page 9.)
12

[37] Zhilin Yang, Peng Qi, Saizheng Zhang, Yoshua Bengio, William Cohen, Ruslan Salakhutdinov,
and Christopher D Manning. Hotpotqa: A dataset for diverse, explainable multi-hop question
answering. In Proceedings of the 2018 Conference on Empirical Methods in Natural Language
Processing , pages 2369–2380, 2018. (Cited on pages 2 and 5.)
[38] Shunyu Yao, Jeffrey Zhao, Dian Yu, Nan Du, Izhak Shafran, Karthik R Narasimhan, and
Yuan Cao. React: Synergizing reasoning and acting in language models. In The Eleventh
International Conference on Learning Representations , 2023. (Cited on pages 1, 2, 6, and 9.)
[39] Howard Yen, Tianyu Gao, Minmin Hou, Ke Ding, Daniel Fleischer, Peter Izsak, Moshe
Wasserblat, and Danqi Chen. Helmet: How to evaluate long-context language models effectively
and thoroughly. arXiv preprint arXiv:2410.02694 , 2024. (Cited on pages 1 and 9.)
[40] Jiahao Zhang, Haiyang Zhang, Dongmei Zhang, Liu Yong, and Shen Huang. End-to-end beam
retrieval for multi-hop question answering. In Proceedings of the 2024 Conference of the
North American Chapter of the Association for Computational Linguistics: Human Language
Technologies (Volume 1: Long Papers) , pages 1718–1731, 2024. (Cited on page 7.)
[41] Denny Zhou, Nathanael Schärli, Le Hou, Jason Wei, Nathan Scales, Xuezhi Wang, Dale
Schuurmans, Claire Cui, Olivier Bousquet, Quoc V Le, et al. Least-to-most prompting enables
complex reasoning in large language models. In The Eleventh International Conference on
Learning Representations , 2023. (Cited on page 9.)
[42] Yang Zhou, Hongyi Liu, Zhuoming Chen, Yuandong Tian, and Beidi Chen. Gsm-infinite: How
do your llms behave over infinitely increasing context length and reasoning complexity? arXiv
preprint arXiv:2502.05252 , 2025. (Cited on page 2.)
[43] Ziyuan Zhuang, Zhiyang Zhang, Sitao Cheng, Fangkai Yang, Jia Liu, Shujian Huang, Qingwei
Lin, Saravan Rajmohan, Dongmei Zhang, and Qi Zhang. Efficientrag: Efficient retriever for
multi-hop question answering. In Proceedings of the 2024 Conference on Empirical Methods in
Natural Language Processing , pages 3392–3411, 2024. (Cited on page 9.)
13

A Prompts
A.1 Memento
The plan generation stage (Sec. 3.1) uses the following prompt:
You will be provided a question . Your goal is to devise a
Prolog query to answer this question . Your response must end in
"** Plan :** <plan >\n** Target :** <target >\n** Definition :**
<definition >", where <plan > is a Prolog query that when
executed , will yield the answer to the question , <target >
is the target variable in the Prolog query to be returned
as the final answer , and <definition > defines the semantic
meaning of predicates in the Prolog query .
Here are some examples :
( START OF EXAMPLES )
{ examples }
(END OF EXAMPLES )
Question : { question }
Answer :
During the database construction stage (Sec. 3.2), we resolve each sub-query qiby invoking an LLM
with one of three different “base” prompting strategies—CoT, CoT-RAG, and ReAct—corresponding
to the in-context, RAG, and agentic settings. Specifically, we substitute the evidence placeholder
in the CoT prompt (App. A.2) with the entire corpus in the in-context setting and the kdocuments
retrieved by the BM25 retriever in the RAG setting. In the agentic setting, we use the ReAct prompt
(App. A.3). To ensure that the answer can be added to the Prolog database, we instruct the LLM to
format <answer> as a Prolog literal—either through additional instructions or few-shot examples.
A.2 Chain-of-Thought
You are given the following evidence :
( BEGIN EVIDENCE )
{{ evidence }}
(END EVIDENCE )
You will be provided a question . Your response must end in the
following sentence : The answer is <answer >.
Here , <answer > must be either a single answer or a
list of answers separated by ’{ constants . answer_sep }’.
Here are some examples :
( START OF EXAMPLES )
{{ examples }}
(END OF EXAMPLES )
Question : {{ question }}
Answer :
A.3 ReAct
Solve a question answering task with interleaving Thought ,
Action , Observation steps . You should use thoughts to reason
about the current situation , and Action can be 3 types :
(1) RetrieveArticle [{{{{ entity }}}}]. This action retrieves
the article about {{{{ entity }}}} if it exists . If the article
does not exist , the action will say so.
14

(2) Search [{{{{ attribute }}}}]. This action searches the
database for {{{{ attribute }}}} and retrieves all articles
that contain {{{{ attribute }}}}. If no article contains
{{{{ attribute }}}} , the action will say so.
(3) Finish [{{{{ answer }}}}]. This action answers the question
with {{{{ answer }}}}.
If you cannot find the answer , output the empty answer like :
Finish [].
If there are multiple answers A,B,C, answer with a list like :
Finish [A{ constants . answer_sep }B{ constants . answer_sep }C].
If the answer is a number , output the number like : Finish [5] or
Finish [1{ constants . answer_sep }2{ constants . answer_sep }3].
If you cannot find an answer to the numerical question , output
the 0 answer like : Finish [0].
You may take as many steps as necessary .
Here are some examples :
( START OF EXAMPLES )
{{ examples }}
(END OF EXAMPLES )
Now answer the following question :
Question : {{ question }}
Plan :
{{ plan }}
{{ scratchpad }}
A.4 Parametric baselines
In the experiments of Sec. 6, we include two baselines to assess the extent of Llama-3.3-70B’s
internal knowledge: standard and CoT. Both share the prompt template below and only differ in
the few-shot examples, represented by the placeholder examples . Standard only includes the final
answer, whereas CoT includes hand-crafted reasoning traces before the final answer.
You will be provided a question . Your response must end in the
following sentence : The answer is <answer >.
Here , <answer > must be either a single answer or a
list of answers separated by ’{ constants . answer_sep }’.
Here are some examples :
( START OF EXAMPLES )
{{ examples }}
(END OF EXAMPLES )
Question : {{ question }}
Answer :
A.5 Ablation baselines
In the experiment of Sec. 7, we use the following prompt for the Execute⋆method:
You are given the following evidence :
( BEGIN EVIDENCE )
{{ evidence }}
(END EVIDENCE )
You will be given a step -by - step plan . Your response must
end in the following sentence : The answer is <answer >.
Here , <answer > must be one of the following :
15

Table 3: Model comparison. We report exact match (EM) and F1 scores (%) on 500 dev samples
each from HotpotQA (HP), 2WikiMultihopQA (2Wiki), and MuSiQue (MSQ), and mean F1 score ±
1 standard error across 3 dataset generation seeds for PhantomWiki (PW-S, PW-M). See Sec. 5 for
detailed discussion.
HP 2Wiki MSQ PW-S PW-M
Method EM F1 EM F1 EM F1 F1 F1
CoT (Llama-3.3-70B) 60.6 76.9 73.2 80.8 50.0 63.3 75.3±1.2 34.8±2.3
CoT (R1-70B) 56.4 71.8 73.6 80.6 46.6 58.1 67.0±4.8 35.5±1.8
- a name (if there is only one correct answer ); or
- a list of names separated by ’{ constants . answer_sep }’
(if there are multiple correct answers ); or
- numbers separated by ’{ constants . answer_sep }’ (if the
answer is numerical ).
Here are some examples :
( START OF EXAMPLES )
{{ examples }}
(END OF EXAMPLES )
Plan :
{{ plan }}
Answer :
B Model Comparison
In addition to our experiments with Llama-3.3-70B in Sec. 5 and 6, we include a comparison of
Llama-3.3-70B and DeepSeek-R1-Distill-Llama-70B (R1-70B) with CoT prompting in Tab. 3. For
in-context reasoning, Llama-3.3-70B outperforms DeepSeek-R1-Distill-Llama-70B on all datasets,
except for PW-M. See Sec. 4.1 for details about each dataset.
C RAG Results with Varying Documents per Query
Fig. 4 shows the how varying the number of retrieved documents per query affects the performance
of CoT and Memento on HP, 2Wiki, and MSQ. For each dataset, we use a random sample of 200
questions from the dev splits, making the numbers similar but not directly comparable to those in
Tab. 2, which use random samples of size 500 each.
D Experiment Details
D.1 PhantomWiki details
We detail the generation of PhantomWiki instances in this section.
D.1.1 Generating a PhantomWiki Universe
The first stage of the PhantomWiki pipeline generates a random universe of ncharacters as well as
the document corpus describing it, as illustrated in Fig. 5, (1-2).
Generating Characters. Each character in a PhantomWiki universe is described through its social
relationships andpersonal facts (Fig. 5, (1)). For the social relationships, we first generate family
trees, following the family tree generator of Hohenecker and Lukasiewicz [9]. We iteratively pick
16

4 6 8 10 120.250.300.350.40EMHotpotQA
4 6 8 10 12
Retrieved documents per query k0.150.200.250.300.350.402WikiMultiHopQA
4 6 8 100.100.120.140.160.18MuSiQue
Memento CoTMementoCoT
CoTMemento
4 6 8 10 120.350.400.450.500.55F1HotpotQA
4 6 8 10 12
Retrieved documents per query k0.200.250.300.350.400.450.502WikiMultiHopQA
4 6 8 100.140.160.180.200.220.240.26MuSiQue
Memento CoTMementoCoT
CoTMemento
Figure 4: Comparison of methods with varying kin the RAG setting.
n = 1Mn = 3
n = 8
/u1F468_u1F33E.3/u1F469_u1F680.3/u1F468_u1F373.2
/u1F9D1_u1F3A4.2
/u1F469_u1F3A8.1 /u1F468_u1F527.1/u1F469_u2695.4
/u1F46E.3
/u1F468_u1F33E.3  David Smith
The friend of David is /u1F468_u1F527.1 John Harper.
The hobby of David is 🦅 birdwatching.Who is  <...>  ?
the  ■  of  <...>
the  ■  of  <...>
the person whose ■ is ■
 ■ → {nephew} ■ → {friend} 
■ ■ → {hobby},{🦅 }
Q: Who is the nephew of the friend of the
         person whose hobby is birdwatching?
?- nephew(X2, Y), 
   friend(X1, X2), 
   hobby(X1, 🦅 ).
A: Y={/u1F468_u1F373.2,/u1F9D1_u1F3A4.2}
(4) Use a logic program to deduce the answers(3) Generate questions using a context-free grammar
(2) Create the document  corpus for the universe(1) Generate a random universe of size n
Figure 5: Overview of the PhantomWiki pipeline.
a person and generate their parent or child based on various constraints4, until the user-specified
4For example, the number of offspring of a person has to be smaller than some threshold, parents of the
people at the maximal tree level will not be generated, etc.
17

universe size of npeople is reached. The user can also specify other hyperparameters like the number
of trees, their maximal depth, and the maximal number of offspring for each person. In addition to the
family trees, we generate a friendship graph using the Erd ˝os–Rényi model (making two people friends
with some fixed probability, typically controlled by the desired average number of friendships.)
Generating Facts. Next, we generate personal facts for each person in the PhantomWiki universe.
Names are assigned during the family generation procedure, with the first name sampled based on
the character’s gender and the surname based on the family tree, resulting in 15M full names in total5.
We also add dates of birth in a way that is consistent with the existing family relations, and assign
each person a job and a hobby that we uniformly sample from over 300 and 600 options respectively.
Generating Articles. Given all relevant facts for each person, we convert them into articles using
pre-defined templates, e.g. “The job of David is a farmer. The hobby of David is birdwatching.”
(see Fig. 5, (2)). This construction conveys the necessary information while keeping the articles
short (about 160tokens on average). While it is possible to extend the article generation process
to LLM-based methods (see e.g. Shao et al. 30), this poses the challenge of guaranteeing factual
correctness without additional costs and external supervision. This has been supported by our
preliminary experiments on article generation using Llama-3.3-70B, where we observed factual errors
in the resulting articles; therefore we do not use LLMs and rely entirely on templates. The articles
are the only component of PhantomWiki available to the model during its evaluation.
D.1.2 Generating Question-Answer Pairs
In the second half of the PhantomWiki pipeline, we generate a set of questions with verifiable answers,
as shown in Fig. 5, (3-4).
Generating Questions. We implement automatic question generation through a context-free grammar
(CFG, Hopcroft et al. 10) ofquestion templates , which we then use to sample complete questions.
For example, the question template “ Who is the <relation >of<name>?” can be used to sample the
question “ Who is the friend of David? ” (see Fig. 5, (3)). The main advantage of using a CFG is that it
efficiently and systematically obtains allpossible compositions of questions for some recursion depth
d. For instance, the following subset of our context-free grammar:
S→Who is R?
R→the<relation >ofR′
R′→R|<name>
can lead to questions ranging from “ Who is the friend of David? ” to “ Who is the nephew of the friend
of the brother of David? ” asdincreases. In addition to these nested compositions, our CFG also
supports questions about personal attributes (e.g. “ Who is the person whose hobby is birdwatching? ”),
aggregation questions (“ How many brothers does David have? ”), and combinations of all three (“ How
many friends does the brother of the person whose hobby is birdwatching have? ”)
Generating Answers. To ensure that the answers to the sampled questions are verifiably correct, we
represent our generated universe in Prolog, a logic programming language [ 31]. Each Prolog program
consists of a set of facts known about the world such as hobby("David", "birdwatching") , and
a set of rules defining how facts are related to each other, such as nephew(X, Y) :- sibling(X,
A), son(A, Y) . The Prolog program uses these facts and rules to deduce the exhaustive set of
answers to its queries (i.e., the CFG-generated questions). For example, a question “ Who is the
nephew of the friend of the person whose hobby is birdwatching? ” corresponds to the three-statement
Prolog query ?- nephew(X2, Y), friend(X1, X2), hobby(X1, "birdwatching") , which
returns all people satisfying these constraints in the PhantomWiki universe (see Fig. 5 (4)).
To construct the Prolog queries automatically, we modify the CFG algorithm to generate both the
question and query templates in parallel. We note, however, that the queries are separate from the
final PhantomWiki corpus and question-answer pairs, and the answers returned by the Prolog program
should be held out as part of the evaluation procedure.
5We use unique names in our experiments, but PhantomWiki also supports repeated names.
18

D.2 Self-Ask and IRCoT details
On HP, 2Wiki, and MSQ, whose questions require at most 4 hops, we set the maximum iterations
for Self-Ask and IRCoT to be 4. For PW-S, PW-M, and PW-L, we set this hyperparameter to 50 to
account for the increased number of hops. See Tab. 2 for the exact number of retrieved documents
per query, k, which we keep consistent with all other methods in the RAG setting.
For Self-Ask, use the few-shot prompts from Jin et al. [13]. For IRCoT, we use the provided GPT3
(code-davincii-002 ) few-shot examples from Trivedi et al. [34, App. G].
For IRCoT, we also allow an arbitrary number of retrieved paragraphs, unlike Trivedi et al. [34],
who originally used models with more limited context lengths than Llama-3.3-70B. We do not use a
separate post-hoc reader, and instead extract the answer from the generated chain-of-thought in the
manner of Trivedi et al. [34, App. F].
19

NeurIPS Paper Checklist
1.Claims
Question: Do the main claims made in the abstract and introduction accurately reflect the
paper’s contributions and scope?
Answer: [Yes]
Justification: We propose a prompting strategy and show results on multiple datasets in
Sec. 4, Sec. 5, and Sec. 6.
Guidelines:
•The answer NA means that the abstract and introduction do not include the claims
made in the paper.
•The abstract and/or introduction should clearly state the claims made, including the
contributions made in the paper and important assumptions and limitations. A No or
NA answer to this question will not be perceived well by the reviewers.
•The claims made should match theoretical and experimental results, and reflect how
much the results can be expected to generalize to other settings.
•It is fine to include aspirational goals as motivation as long as it is clear that these goals
are not attained by the paper.
2.Limitations
Question: Does the paper discuss the limitations of the work performed by the authors?
Answer: [Yes]
Justification: Limitations discussed in Sec. 9.
Guidelines:
•The answer NA means that the paper has no limitation while the answer No means that
the paper has limitations, but those are not discussed in the paper.
• The authors are encouraged to create a separate "Limitations" section in their paper.
•The paper should point out any strong assumptions and how robust the results are to
violations of these assumptions (e.g., independence assumptions, noiseless settings,
model well-specification, asymptotic approximations only holding locally). The authors
should reflect on how these assumptions might be violated in practice and what the
implications would be.
•The authors should reflect on the scope of the claims made, e.g., if the approach was
only tested on a few datasets or with a few runs. In general, empirical results often
depend on implicit assumptions, which should be articulated.
•The authors should reflect on the factors that influence the performance of the approach.
For example, a facial recognition algorithm may perform poorly when image resolution
is low or images are taken in low lighting. Or a speech-to-text system might not be
used reliably to provide closed captions for online lectures because it fails to handle
technical jargon.
•The authors should discuss the computational efficiency of the proposed algorithms
and how they scale with dataset size.
•If applicable, the authors should discuss possible limitations of their approach to
address problems of privacy and fairness.
•While the authors might fear that complete honesty about limitations might be used
by reviewers as grounds for rejection, a worse outcome might be that reviewers
discover limitations that aren’t acknowledged in the paper. The authors should use
their best judgment and recognize that individual actions in favor of transparency play
an important role in developing norms that preserve the integrity of the community.
Reviewers will be specifically instructed to not penalize honesty concerning limitations.
3.Theory assumptions and proofs
Question: For each theoretical result, does the paper provide the full set of assumptions and
a complete (and correct) proof?
20

Answer: [NA]
Justification: We do not present theoretical results.
Guidelines:
• The answer NA means that the paper does not include theoretical results.
•All the theorems, formulas, and proofs in the paper should be numbered and
cross-referenced.
•All assumptions should be clearly stated or referenced in the statement of any theorems.
•The proofs can either appear in the main paper or the supplemental material, but if
they appear in the supplemental material, the authors are encouraged to provide a short
proof sketch to provide intuition.
•Inversely, any informal proof provided in the core of the paper should be complemented
by formal proofs provided in appendix or supplemental material.
• Theorems and Lemmas that the proof relies upon should be properly referenced.
4.Experimental result reproducibility
Question: Does the paper fully disclose all the information needed to reproduce the
main experimental results of the paper to the extent that it affects the main claims and/or
conclusions of the paper (regardless of whether the code and data are provided or not)?
Answer: [Yes]
Justification: We detail the experimental setup in Sec. 4 and appendix (to be submitted in
the supplementary material).
Guidelines:
• The answer NA means that the paper does not include experiments.
•If the paper includes experiments, a No answer to this question will not be perceived
well by the reviewers: Making the paper reproducible is important, regardless of
whether the code and data are provided or not.
•If the contribution is a dataset and/or model, the authors should describe the steps taken
to make their results reproducible or verifiable.
•Depending on the contribution, reproducibility can be accomplished in various ways.
For example, if the contribution is a novel architecture, describing the architecture fully
might suffice, or if the contribution is a specific model and empirical evaluation, it may
be necessary to either make it possible for others to replicate the model with the same
dataset, or provide access to the model. In general. releasing code and data is often
one good way to accomplish this, but reproducibility can also be provided via detailed
instructions for how to replicate the results, access to a hosted model (e.g., in the case
of a large language model), releasing of a model checkpoint, or other means that are
appropriate to the research performed.
•While NeurIPS does not require releasing code, the conference does require all
submissions to provide some reasonable avenue for reproducibility, which may depend
on the nature of the contribution. For example
(a)If the contribution is primarily a new algorithm, the paper should make it clear how
to reproduce that algorithm.
(b)If the contribution is primarily a new model architecture, the paper should describe
the architecture clearly and fully.
(c)If the contribution is a new model (e.g., a large language model), then there should
either be a way to access this model for reproducing the results or a way to reproduce
the model (e.g., with an open-source dataset or instructions for how to construct
the dataset).
(d)We recognize that reproducibility may be tricky in some cases, in which case
authors are welcome to describe the particular way they provide for reproducibility.
In the case of closed-source models, it may be that access to the model is limited in
some way (e.g., to registered users), but it should be possible for other researchers
to have some path to reproducing or verifying the results.
5.Open access to data and code
21

Question: Does the paper provide open access to the data and code, with sufficient
instructions to faithfully reproduce the main experimental results, as described in
supplemental material?
Answer: [No]
Justification: Code will be made publicly available with the camera ready if accepted.
Guidelines:
• The answer NA means that paper does not include experiments requiring code.
•Please see the NeurIPS code and data submission guidelines ( https://nips.cc/
public/guides/CodeSubmissionPolicy ) for more details.
•While we encourage the release of code and data, we understand that this might not be
possible, so “No” is an acceptable answer. Papers cannot be rejected simply for not
including code, unless this is central to the contribution (e.g., for a new open-source
benchmark).
•The instructions should contain the exact command and environment needed to run to
reproduce the results. See the NeurIPS code and data submission guidelines ( https:
//nips.cc/public/guides/CodeSubmissionPolicy ) for more details.
•The authors should provide instructions on data access and preparation, including how
to access the raw data, preprocessed data, intermediate data, and generated data, etc.
•The authors should provide scripts to reproduce all experimental results for the new
proposed method and baselines. If only a subset of experiments are reproducible, they
should state which ones are omitted from the script and why.
•At submission time, to preserve anonymity, the authors should release anonymized
versions (if applicable).
•Providing as much information as possible in supplemental material (appended to the
paper) is recommended, but including URLs to data and code is permitted.
6.Experimental setting/details
Question: Does the paper specify all the training and test details (e.g., data splits,
hyperparameters, how they were chosen, type of optimizer, etc.) necessary to understand
the results?
Answer: [Yes]
Justification: Full details will be provided as supplementary material.
Guidelines:
• The answer NA means that the paper does not include experiments.
•The experimental setting should be presented in the core of the paper to a level of detail
that is necessary to appreciate the results and make sense of them.
•The full details can be provided either with the code, in appendix, or as supplemental
material.
7.Experiment statistical significance
Question: Does the paper report error bars suitably and correctly defined or other appropriate
information about the statistical significance of the experiments?
Answer: [Yes]
Justification: We use greedy decoding in all experiments to ensure deterministic results.
We report mean ±1 standard error on synthetic datasets (PhantomWiki), for which there is
randomness in the data generation process.
Guidelines:
• The answer NA means that the paper does not include experiments.
•The authors should answer "Yes" if the results are accompanied by error bars,
confidence intervals, or statistical significance tests, at least for the experiments that
support the main claims of the paper.
•The factors of variability that the error bars are capturing should be clearly stated (for
example, train/test split, initialization, random drawing of some parameter, or overall
run with given experimental conditions).
22

•The method for calculating the error bars should be explained (closed form formula,
call to a library function, bootstrap, etc.)
• The assumptions made should be given (e.g., Normally distributed errors).
•It should be clear whether the error bar is the standard deviation or the standard error
of the mean.
•It is OK to report 1-sigma error bars, but one should state it. The authors should
preferably report a 2-sigma error bar than state that they have a 96% CI, if the hypothesis
of Normality of errors is not verified.
•For asymmetric distributions, the authors should be careful not to show in tables or
figures symmetric error bars that would yield results that are out of range (e.g. negative
error rates).
•If error bars are reported in tables or plots, The authors should explain in the text how
they were calculated and reference the corresponding figures or tables in the text.
8.Experiments compute resources
Question: For each experiment, does the paper provide sufficient information on the
computer resources (type of compute workers, memory, time of execution) needed to
reproduce the experiments?
Answer: [Yes]
Justification: We report exact hardware in the appendix/supplemental material.
Guidelines:
• The answer NA means that the paper does not include experiments.
•The paper should indicate the type of compute workers CPU or GPU, internal cluster,
or cloud provider, including relevant memory and storage.
•The paper should provide the amount of compute required for each of the individual
experimental runs as well as estimate the total compute.
•The paper should disclose whether the full research project required more compute
than the experiments reported in the paper (e.g., preliminary or failed experiments that
didn’t make it into the paper).
9.Code of ethics
Question: Does the research conducted in the paper conform, in every respect, with the
NeurIPS Code of Ethics https://neurips.cc/public/EthicsGuidelines ?
Answer: [Yes]
Justification: Yes this research is conducted with the NeurIPS Code of Ethics.
Guidelines:
•The answer NA means that the authors have not reviewed the NeurIPS Code of Ethics.
•If the authors answer No, they should explain the special circumstances that require a
deviation from the Code of Ethics.
•The authors should make sure to preserve anonymity (e.g., if there is a special
consideration due to laws or regulations in their jurisdiction).
10.Broader impacts
Question: Does the paper discuss both potential positive societal impacts and negative
societal impacts of the work performed?
Answer: [NA]
Justification: There is no societal impact of the work performed.
Guidelines:
• The answer NA means that there is no societal impact of the work performed.
•If the authors answer NA or No, they should explain why their work has no societal
impact or why the paper does not address societal impact.
•Examples of negative societal impacts include potential malicious or unintended uses
(e.g., disinformation, generating fake profiles, surveillance), fairness considerations
(e.g., deployment of technologies that could make decisions that unfairly impact specific
groups), privacy considerations, and security considerations.
23

•The conference expects that many papers will be foundational research and not tied
to particular applications, let alone deployments. However, if there is a direct path to
any negative applications, the authors should point it out. For example, it is legitimate
to point out that an improvement in the quality of generative models could be used to
generate deepfakes for disinformation. On the other hand, it is not needed to point out
that a generic algorithm for optimizing neural networks could enable people to train
models that generate Deepfakes faster.
•The authors should consider possible harms that could arise when the technology is
being used as intended and functioning correctly, harms that could arise when the
technology is being used as intended but gives incorrect results, and harms following
from (intentional or unintentional) misuse of the technology.
•If there are negative societal impacts, the authors could also discuss possible mitigation
strategies (e.g., gated release of models, providing defenses in addition to attacks,
mechanisms for monitoring misuse, mechanisms to monitor how a system learns from
feedback over time, improving the efficiency and accessibility of ML).
11.Safeguards
Question: Does the paper describe safeguards that have been put in place for responsible
release of data or models that have a high risk for misuse (e.g., pretrained language models,
image generators, or scraped datasets)?
Answer: [NA]
Justification: This work does not involve data or models with high risk.
Guidelines:
• The answer NA means that the paper poses no such risks.
•Released models that have a high risk for misuse or dual-use should be released with
necessary safeguards to allow for controlled use of the model, for example by requiring
that users adhere to usage guidelines or restrictions to access the model or implementing
safety filters.
•Datasets that have been scraped from the Internet could pose safety risks. The authors
should describe how they avoided releasing unsafe images.
•We recognize that providing effective safeguards is challenging, and many papers do
not require this, but we encourage authors to take this into account and make a best
faith effort.
12.Licenses for existing assets
Question: Are the creators or original owners of assets (e.g., code, data, models), used in
the paper, properly credited and are the license and terms of use explicitly mentioned and
properly respected?
Answer: [Yes]
Justification: Yes, the creators or original owners of assets (e.g., code, data, models), used in
the paper, properly credited.
Guidelines:
• The answer NA means that the paper does not use existing assets.
• The authors should cite the original paper that produced the code package or dataset.
•The authors should state which version of the asset is used and, if possible, include a
URL.
• The name of the license (e.g., CC-BY 4.0) should be included for each asset.
•For scraped data from a particular source (e.g., website), the copyright and terms of
service of that source should be provided.
•If assets are released, the license, copyright information, and terms of use in the
package should be provided. For popular datasets, paperswithcode.com/datasets
has curated licenses for some datasets. Their licensing guide can help determine the
license of a dataset.
•For existing datasets that are re-packaged, both the original license and the license of
the derived asset (if it has changed) should be provided.
24

•If this information is not available online, the authors are encouraged to reach out to
the asset’s creators.
13.New assets
Question: Are new assets introduced in the paper well documented and is the documentation
provided alongside the assets?
Answer: [Yes]
Justification: new assets introduced in the paper are well documented.
Guidelines:
• The answer NA means that the paper does not release new assets.
•Researchers should communicate the details of the dataset/code/model as part of their
submissions via structured templates. This includes details about training, license,
limitations, etc.
•The paper should discuss whether and how consent was obtained from people whose
asset is used.
•At submission time, remember to anonymize your assets (if applicable). You can either
create an anonymized URL or include an anonymized zip file.
14.Crowdsourcing and research with human subjects
Question: For crowdsourcing experiments and research with human subjects, does the paper
include the full text of instructions given to participants and screenshots, if applicable, as
well as details about compensation (if any)?
Answer: [NA]
Justification: the paper does not involve crowdsourcing nor research with human subjects.
Guidelines:
•The answer NA means that the paper does not involve crowdsourcing nor research with
human subjects.
•Including this information in the supplemental material is fine, but if the main
contribution of the paper involves human subjects, then as much detail as possible
should be included in the main paper.
•According to the NeurIPS Code of Ethics, workers involved in data collection, curation,
or other labor should be paid at least the minimum wage in the country of the data
collector.
15.Institutional review board (IRB) approvals or equivalent for research with human
subjects
Question: Does the paper describe potential risks incurred by study participants, whether
such risks were disclosed to the subjects, and whether Institutional Review Board (IRB)
approvals (or an equivalent approval/review based on the requirements of your country or
institution) were obtained?
Answer: [NA]
Justification: This work does not involve crowdsourcing or research with human subjects.
Guidelines:
•The answer NA means that the paper does not involve crowdsourcing nor research with
human subjects.
•Depending on the country in which research is conducted, IRB approval (or equivalent)
may be required for any human subjects research. If you obtained IRB approval, you
should clearly state this in the paper.
•We recognize that the procedures for this may vary significantly between institutions
and locations, and we expect authors to adhere to the NeurIPS Code of Ethics and the
guidelines for their institution.
•For initial submissions, do not include any information that would break anonymity (if
applicable), such as the institution conducting the review.
16.Declaration of LLM usage
25

Question: Does the paper describe the usage of LLMs if it is an important, original, or
non-standard component of the core methods in this research? Note that if the LLM is used
only for writing, editing, or formatting purposes and does not impact the core methodology,
scientific rigorousness, or originality of the research, declaration is not required.
Answer: [Yes]
Justification: Yes we describe the usage of LLMs.
Guidelines:
•The answer NA means that the core method development in this research does not
involve LLMs as any important, original, or non-standard components.
•Please refer to our LLM policy ( https://neurips.cc/Conferences/2025/LLM )
for what should or should not be described.
26