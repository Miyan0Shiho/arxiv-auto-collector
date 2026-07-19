# Boolean queries are all you need?

**Authors**: Charles L. A. Clarke, Mark D. Smucker

**Published**: 2026-07-13 10:25:46

**PDF URL**: [https://arxiv.org/pdf/2607.11362v1](https://arxiv.org/pdf/2607.11362v1)

## Abstract
We equipped an LLM-based search agent with access to a Boolean retrieval engine to search the MS MARCO V2.1 deduped segment collection used by the TREC 2024 RAG track. Over a standard track subset of 86 topics, and operating under a budget of 100 model calls/topic, the agent achieved an NDCG@10 of 0.6863, which would place it above many dense, sparse, and learned-sparse first-stage retrievers. Ranking is based solely on the density of corpus substrings matching a query, with no requirement for supervised learning, global statistics, or term weights. Formally, the query language expresses a strict subset of the regular languages, with a document's score based on the number and length of matches it contains. Although the results are more exploratory than definitive, because they are based on a single test collection that was publicly available during model training, they suggest that simple pattern matching may be sufficient for agentic search.

## Full Text


<!-- PDF content starts -->

Boolean queries are all you need?
Charles L. A. Clarke Mark D. Smucker
University of Waterloo, Canada
Abstract
We equipped an LLM-based search agent with access to a Boolean retrieval engine to
search the MS MARCO V2.1 deduped segment collection used by the TREC 2024 RAG
track. Over a standard track subset of 86 topics — and operating under a budget of
100 model calls/topic — the agent achieved an NDCG@10 of 0.6863, which would place
it above many dense, sparse, and learned-sparse first-stage retrievers. Ranking is based
solely on the density of corpus substrings matching a query — with no requirement for
supervised learning, global statistics, or term weights. Formally, the query language
expresses a strict subset of the regular languages, with a document’s score based on the
number and length of matches it contains. Although the results are more exploratory
than definitive — because they are based on a single test collection that was publicly
available during model training — they suggest that simple pattern matching may be
sufficient for agentic search.
1 Introduction
Long before the emergence of AI-powered natural-language search and keyword-focused
Web search, Boolean queries provided the primary interface to many operational search
engines [24]. While Boolean retrieval was effective in the hands of librarians and other
trained technicians — allowing them to specify precise requirements for retrieved results —
it proved difficult for untrained people to successfully formulate queries. As early as 1982,
efforts were already underway to automatically reformulate natural language queries into
Boolean queries, simplifying interaction [23]. By the mid-1990s, Boolean retrieval was fad-
ing away. Keyword-based sparse vector retrieval was well established in academic contexts,
BM25 was gaining a reputation for excellent effectiveness, and the earliest keyword-based
Web search engines were widely available [22, 25, 34]. Even so, as late as 1995, manually con-
structed Boolean queries could still outperform the best automatic systems at TREC [7, 14].
A few years later, using interactive searching and judging with Boolean queries, trained
searchers independently created judgments yielding system rankings consistent with offi-
cial TREC-6 rankings, translating human intuition into exact-match queries to find relevant
documents [10, 31]. Nonetheless, given that untrained humans often struggle to formulate ef-
fective Boolean queries, by the 21st century, they were largely abandoned for general-purpose
search.
1
arXiv:2607.11362v1  [cs.IR]  13 Jul 2026

(^ "goth rock" (+ society social political issues themes addressed))
(^ "goth rock"
(+ darkness depravity alienation despair death)
(+ lyrics themes))
(^ goth
"post-punk"
(+ politics political "social commentary" society)
(+ lyrics themes))
Figure 1: Example LLM-generated queries for the TREC 2024 RAG Track topic “what
society issues did goth rock address” (#2024-152259).
(^
(+
(^ quebec (+ separatism separatist separation separate))
(^ quebec (+ sovereignty sovereign independent independence))
"jacques parizeau"
"louise beaudoin"
"parti quebecois")
(+ economic economically economics poll opinion referendum majority))
Figure 2: Example human-generated query for the TREC-4 adhoc topic “What are the
prospects of the Quebec separatists achieving independence from the rest of Canada?”
(#207), taken from the query set of Clarke et al. [7] with notation adjusted for consistency
with the modern Cottontail notation of Figure 1.
Large language models face no such struggle. When appropriately prompted, LLMs can
reformulate natural language into reasonable-seeming Boolean queries [1, 27, 32, 33]. Figure 1
provides an example from the experiments in this paper. This example can be compared to
the human-generated query in Figure 2. In both cases, the Boolean query includes terms not
found in the original topics. In both cases, the searcher was encouraged to use their internal
knowledge as a guide to query expansion.
As a test of LLM-driven Boolean retrieval, we implemented a simple search agent (Vole)
based on the interactive searching and judging (ISJ) procedure used by Cormack et al. [10] to
independently create judgments for TREC-6 topics. For retrieval, we adopt a newer imple-
mentation of their string-matching algorithm, which treats the corpus as a single contiguous
string and efficiently identifies shortest substring matches to a Boolean expression. Docu-
ments are ranked based on the length and number of matches they contain, avoiding the
need for global statistics and term weights (Section 3).
The Vole search agent acts as little more than a runner between a search engine and an
LLM (Section 4). After providing instructions to the LLM, the Vole carries queries from the
LLM to the search engine and brings results back, managing the interaction. The interaction
is limited to 100 model calls/topic. On each call, the model may choose one of three actions:
2

1) run a new query, receiving a 200-word snippet centered around the shortest substring in
the top document; 2) ask for a snippet from the next document for a previous query; 3)
ask to see a full document. As documents and snippets are seen by the model, it returns
relevance judgments, along with reasons to support the judgments. In addition to the limit
of 100 model calls, the Vole will halt the interaction after the model has made 50 distinct
relevance judgments, even if the model-call limit has not been reached.
The instructions encourage the model to find as many highly-relevant documents as
possible. Since Vole searches by exploring, rather than ranking, we re-rank the discovered
documents for comparison with other methods, using model-judged relevance as a primary
key and discovery order as a secondary key. We report experiments on a standard subset of
TREC 2024 RAG Track topics, using gpt-5.5 as the LLM (Section 5), comparing the results
to a range of official track submissions from the first author’s team [9], with performance
ranging from a typical BM25 baseline to one of the top-3 runs (Section 6). Since the Vole digs
up documents it judges highly relevant but that are unjudged, we use UMBRELA [29, 30]
to fill holes by assessing these previously unjudged documents, rather than relying on Vole’s
own assessments (Section 5).
2 Related Work
This paper is only one of several recent papers suggesting that traditional IR methods may
be sufficient to support retrieval augmented generation and agentic search. As early as 2023,
Doostmohammadi et al. [12] demonstrated that, in retrieval-augmented language modeling,
retrieval gains are better explained by surface-form overlap than by dense semantic similarity,
and that BM25-style surface retrieval can reduce language-model perplexity. Also in a RAG
context, Huly et al. [16] report that ancient unsupervised methods such as Markov Random
Fields (MRF) and relevance-model retrieval outperform both BM25 and dense retrieval.
Multiple papers appearing in May 2026 focused specifically on unsupervised retrieval
methods for agentic search. Yang et al. [36] propose a Superintelligent Retrieval Agent
(SIRA), which uses an LLM to construct a single corpus-discriminative BM25 query rather
than relying on multi-round exploratory search. In contrast with our work, SIRA concen-
trates the model’s effort into constructing one “ultimate” BM25 query, while Vole gives the
model an exploratory loop for revising Boolean queries as evidence is observed. Also using
an exploratory loop, Hsu et al. [15] find that BM25 can support effective agentic search when
paired with a model that can retrieve, browse, and read documents interactively.
Most closely connected with our work, Sen et al. [26] compare grep-style lexical search
with vector retrieval inside an agentic harness, finding that simple exact-match tools can
be highly competitive when agents control the search process. Li et al. [19] also propose
direct corpus interaction, in which agents search raw corpora using exact-match terminal-
style tools rather than fixed top-kretrieval interfaces. Like us, neither approach depends
on vectors, learned weights, or global statistics, operating solely on surface-level textual
matching. Unlike them, we support indexing for efficiency, including immediate transactional
update, and ranking based on the length and density of matches.
3

3 Shortest Substring Ranking
We treat the corpus as a single contiguous string and solve boolean queries underminimal
interval semantics[2, 3, 6, 7]. Each solution to a query is a substring of the corpus satisfy-
ing the Boolean query that itself has no proper substring satisfying the query (a “shortest
substring”). By using an extended inverted index, we can enumerate these substrings in
O(mlogn) time, wherenis the number of occurrences of the most frequent term, andmis
the number of solutions, i.e., shortest substrings in the corpus that satisfy the query [4, 5, 6].
For ranking, we score a document on the basis of the shortest substrings it contains, mak-
ing two assumptions: 1) The shorter the substring, the more likely the containing document
is relevant. 2) The more substrings contained in a document, the more likely it is relevant.
As we enumerate the shortest substrings, we compute a score for each document containing
at least one substring, maintaining a current top-klist.
To compute a score for a document, letSbe the set of shortest substrings contained in
a document, where each substring is expressed as an interval (p i, qi), where the endpoints
of the interval indicate positions in the corpus string. For each (p i, qi)∈ S,p iis the start
position of the substring andq iis the end position of the substring. The shortest substring
ranking (SSR) score of the document is then
score(S) =|S|X
i=11
C+q i−p i.(1)
C >0 is a smoothing constant, with default valueC= 42 used in our experiments.
Each substring contributes inversely with its length, so shorter substrings contribute
more to the document score than longer substrings. Since the score is based solely on local
properties of the document, no global statistics are required, and the ranking method is
trivially parallelizable. Since the languages expressible by our Boolean query language are a
strict subset of the regular languages, shortest substring ranking can be viewed as a limited
form of ranked regular expression retrieval. Potentially, the same method could be applied
to rank documents based on general regular expression matches.
For indexing and shortest substring ranking we use the Cottontail search engine1. Cot-
tontail implements a generalization of inverted indexes called “annotative indexing,” in
which both content and metadata are represented as interval annotations over a common
address space [5]. The same mechanism can support ordinary ranked retrieval, including
BM25/WAND, structural queries over JSON objects, and sparse-vector annotations with a
single unified indexing structure. Unlike a conventional static retrieval index, Cottontail also
supports dynamic annotation and transactional update, allowing new derived structure to be
added to existing content and made visible immediately after commit while preserving ACID
properties for concurrent readers and writers. Thus, Cottontail is not merely a postings-list
implementation for Boolean queries, but can provide a retrieval-native substrate for agent
memory, where raw text, structure, annotations, and derived model judgments can coexist,
be queried, and be updated as the agent works.
1https://github.com/claclark/Cottontail
4

4 Search Agent
For the experiments in this paper, we implemented a simple search agent, the Vole2. The Vole
search agent consists of three components: a remote model, a set of detailed instructions, and
a small interaction runner. These components are supported by a specialized SSR server3
that uses the Cottontail C++ library to implement a dedicated search interface. For our
experiments, the agent is not given full access to Cottontail, and is limited to executing
Boolean queries and accessing document contents.
The model (gpt-5.5for our experiments) is prompted with instructions, a topic de-
scription, and an accumulated interaction transcript. The instructions describe the query
language and the core concepts of shortest substring ranking, along with the JSON protocol
for communication with the runner, and through it, the server. At each step, the model
must return exactly one JSON action: issue a new query, request the next result for an ex-
isting query, or request a full document. The instructions ask the model to judge each newly
observed document on a 0–3 relevance scale and to steer searches toward grade-3 evidence.
The runner is deliberately simple. It maintains the interaction transcript, validates the
model’s JSON decisions, sends search actions to the retrieval service, and records both the
model decisions and search responses. The interaction transcript is maintained append-only
to fit model-side caching requirements, minimizing costs and allowing the model to see the
full interaction at each step. The runner also enforces two stopping conditions: at most
50 unique document judgments per topic and at most 100 model calls, where the limit on
model calls is primarily set to avoid run-away interactions in which the model is unable to
formulate queries that return previously unjudged documents. When a new query surfaces
a duplicate already-judged document, the runner skips it. The model is shown skipped
document identifiers and their ranks but does not spend judgment budget on them again.
The runner communicates with the SSR search server using a simple newline-delimited
JSON protocol over a local TCP socket. For each model action, the runner sends one request
to the server, such as a Boolean query, a request for the next result from a previous query, or a
request for the full text of a document. The server returns a JSON object containing either a
ranked result with a document identifier and 200-word snippet, an exhausted-result marker,
or the requested document text. These server responses are appended to the transcript that
is shown to the model on the next step. Fuller request and timing information is retained in
separate interaction and audit logs.
The SSR server itself is a lightweight JSON-line service that exposes shortest-substring
ranking over one or more Cottontail collections. While it was designed for use in this
experiment, it is general enough to be usable by other search agents for other purposes. A
client sends a Boolean query, and the server evaluates it against a pre-configured definition
of “document”, ranks matching documents by the SSR substring score, and returns results
one at a time with document identifiers, snippets, and timing information. It maintains
query state so clients can request additional results for a query and retrieve full documents.
2https://github.com/claclark/TheVole
3https://github.com/claclark/Cottontail/blob/main/apps/ssr-server.cc
5

Run ID NDCG@10 Description
uwc1 0.8199 Multi-round LLM-based re-ranking of uwc0
uwc2 0.7971 LLM-based re-ranking of track baseline
monster 0.7933 Reciprocal rank fusion (RRF) of uwc1 and uwc2
uwcCQAR 0.7169 RankGPT re-ranking of uwCQA
vole 0.6863 Vole search agent (Section 4)
uwcCQR 0.6806 RankGPT re-ranking of uwCQ
uwcCQA 0.6765 Dense retriever with LLM-based query expansion
uwc0 0.6367 RRF of 15 query expansion and ranker combinations
uwcCQ 0.6177 Dense retriever
uwcBA 0.4581 BM25F with pseudo-relevance feedback
uwcBQ 0.3583 BM25F with LLM-based query expansion
Figure 3: Performance of the Vole vs. retrieval runs submitted to TREC 2024 RAG Track by
the first-author’s team (WaterlooClarke[9]). Except for the Vole, NDCG@10 is based on the
official track UMBRELA judgments, and all runs are fully judged to at least depth 10. For
the Vole, which retrieved many unjudged documents, we filled holes with UMBRELA using
the same model employed by the TREC 2024 RAG Track organizers (gpt-4o), uncovering
420 previously unjudged highly-relevant (grade 2) documents and 18 previously unjudged
perfect (grade 3) documents. Without hole-filling the Vole achieves an NDCG@10 of 0.5340.
5 Task and Dataset
We base our experiments on the TREC RAG Track 2024 retrieval task, which is a standard
adhoc task: Given a topic, return a ranked list of documents according to the probability
ranking principle. The track used the MS MARCO V2.1 deduped segment collection4as
the corpus5. Before starting these experiments, we already had the collection indexed with
Cottontail as a single shard, used for some of the first author’s TREC 2024 experimental
runs [9]. This supports a direct comparison with those runs, since the corpus representation
is held fixed while only query processing and ranking change.
Of the 301 track topics, a subset of 86 topics have (partial) manual judgments, while
the remainder have only LLM-based judgments from the UMBRELA judgment tool [30].
Official results were reported for both the full 301-topic set and the 86-topic subset [29]. For
reasons of parsimony, we base our experiments on the 86-topic subset, but use UMBRELA
judgments in all cases.
4https://trec-rag.github.io/annoucements/2024-corpus-finalization/
5Since the collection is segmented, segments are “documents” for retrieval purposes. For TREC 2024,
the original MS MARCO V2.1 collection was segmented with a “sliding window size of 10 sentences and a
stride of 5 sentences” to make the corpus “more manageable for users and baselines.” For SSR, the unit of
retrieval is defined at query time, which for these experiments was always a pre-defined segment. However,
this unit of retrieval could be a dynamically defined window of words, or with the addition of annotation for
sentences, a dynamically defined window of sentences.
6

6 Results
Figure 3 compares results from the Vole with a selection of official track submissions from
the first-author’s team[9]. For comparison purposes, a Vole interaction is collapsed into
a TREC run by sorting the documents judged by Vole first by the agent’s own assigned
relevance grade and then by discovery order. The external qrels are not used to construct
this ranking; they are used only afterward bytrec eval. When the agent surfaced unjudged
documents, we used UMBRELA to fill these holes with the same remote model (gpt-4o) used
for the official judgments. UMBRELA judged 438 of these previously unjudged documents
to be at least highly relevant.
The agent achieved an NDCG@10 of 0.6863. In the context of the 77 official runs sub-
mitted to the track, this performance would place it above many dense, sparse, and learned-
sparse first-stage retrievers, but below many rankers that themselves employed LLM-based
re-ranking. For example,uwc1was a top-3 run for the task. It is based on a reciprocal
rank fusion (RRF) [11] of 15 different ranking stacks, including traditional BM25 and dense
rankers, with and without LLM-based query expansion. The resulting fused ranking (uwc0)
was then pointwise and pairwise reranked using LLM-based relevance scoring methods [8, 13]
to produceuwc1. In terms of effort, the creation ofuwc1required an average of 194 model
calls per topic for expansion and reranking vs. less than 74 for the Vole. Other groups used
various LLM-based re-ranking methods [20, 21, 28] at TREC 2024. If we are using LLMs
for re-ranking, model calls (and tokens) become important measures of cost.
When we checked efficiency in the logs, we were a little surprised by query latency. In
our past experience, a typical human-written query would run in under 100ms. Only 15%
of Vole queries ran at that speed and 59% required more than a second. One query required
well over a minute. We traced the issue back to an unrecognized problem with Cottontail’s
phrase search optimization, now fixed. Humans, or at least the handful who have written
queries for SSR ranking, tend to use common collocations (“united states”), while in the
slower-running queries, the LLM was using disjunctions of uncommon phrases containing
high-frequency terms (“make this issue known”), which were then repeatedly re-solved by
the engine, instead of being memoized.
7 As They May Search
We have come full circle. Up until the 1980s, operational search engines depended on Boolean
retrieval. While Boolean queries allowed trained operators to precisely specify information
needs, they proved difficult for untrained humans to employ effectively. Thus, the history of
information retrieval is — at least in part — a history of increasingly effective methods for
searching with keyword and natural language queries, culminating in sophisticated semantic
matching methods, such as dense dual-encoder retrievers like DPR and ANCE [17, 35], and
late-interaction models such as ColBERT [18]. But while these methods account for human
limitations in query formulation, large language models face no such limitations. Sen et al.
[26] and Li et al. [19] ask if grep is all that search agents need. Perhaps they need even less.
7

8 Limitations
We report experiments for a single task on a single test collection that was available publicly
during model training. Thus, the results should be viewed more as exploratory than defini-
tive. While the core of Cottontail was largely written by the first author banging on keys
one at a time, the dedicated SSR server and the Vole were largely vibe-coded with Codex.
We applied sanity checking, reviewed data and logs, ran manual tests, and asked the coding
agent a lot of questions, but we did not review every line of code. This paper was written
with AI assistance, but all of the organization and much of the original wording come from
the authors. The shard used for the experiments includes field-oriented TF-IDF annotations,
and the experiment could easily be repeated with BM25F. We have not done this yet.
Acknowledgments
This work would not be possible without the many years of insights and contributions from
Gordon Cormack. We also thank Andrew Parry for his helpful comments.
References
[1] Gaelen P. Adam, Jay DeYoung, Alice Paul, Ian J. Saldanha, Ethan M. Balk, Thomas A.
Trikalinos, and Byron C. Wallace. Literature search sandbox: A large language model
that generates search queries for systematic reviews.JAMIA Open, 7(3):ooae098, Oc-
tober 2024.
[2] Paolo Boldi and Sebastiano Vigna. Efficient optimally lazy algorithms for minimal-
interval semantics.Theoretical Computer Science, 648:8–25, 2016.
[3] Paolo Boldi and Sebastiano Vigna. On the lattice of antichains of finite intervals.Order,
35(1):57–81, March 2018.
[4] Stefan B¨ uttcher, Charles L. A. Clarke, and Gordon V. Cormack.Information Retrieval:
Implementing and Evaluating Search Engines. MIT Press, Cambridge, Massachusetts,
2010.
[5] Charles Clarke. Annotative indexing.Information Retrieval Research, 1(1):109–136,
2025.
[6] Charles L. A. Clarke and Gordon V. Cormack. Shortest-substring retrieval and ranking.
ACM Transactions on Information Systems, 18(1):44–78, January 2000.
[7] Charles L. A. Clarke, Gordon V. Cormack, and Forbes J. Burkowski. Shortest substring
ranking (MultiText experiments for TREC-4). In Donna K. Harman, editor,Proceedings
of the Fourth Text REtrieval Conference (TREC-4), volume 500-236 ofNIST Special
8

Publication, pages 295–304, Gaithersburg, Maryland, 1995. National Institute of Stan-
dards and Technology.
[8] Charles L. A. Clarke, Alexandra Vtyurina, and Mark D. Smucker. Assessing top-k
preferences.ACM Transactions on Information Systems, May’ 2021.
[9] Charles L. A. Clarke, Siqing Huo, and Negar Arabzadeh. Monster ranking. InProceed-
ings of the Thirty-Third Text REtrieval Conference (TREC 2024), volume 1329 ofNIST
Special Publication, Gaithersburg, Maryland, 2024. National Institute of Standards and
Technology.
[10] Gordon V. Cormack, Christopher R. Palmer, and Charles L. A. Clarke. Efficient con-
struction of large test collections. InProceedings of the 21st Annual International
ACM SIGIR Conference on Research and Development in Information Retrieval, page
282–289, 1998.
[11] Gordon V. Cormack, Charles L A Clarke, and Stefan Buettcher. Reciprocal rank fusion
outperforms condorcet and individual rank learning methods. In32nd International
ACM SIGIR Conference on Research and Development in Information Retrieval, page
758–759, 2009.
[12] Ehsan Doostmohammadi, Tobias Norlund, Marco Kuhlmann, and Richard Johansson.
Surface-based retrieval reduces perplexity of retrieval-augmented language models. In
Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics
(Volume 2: Short Papers), pages 521–529, Toronto, Canada, July 2023.
[13] Guglielmo Faggioli, Laura Dietz, Charles L. A. Clarke, Gianluca Demartini, Matthias
Hagen, Claudia Hauff, Noriko Kando, Evangelos Kanoulas, Martin Potthast, Benno
Stein, and Henning Wachsmuth. Perspectives on large language models for relevance
judgment. InACM SIGIR International Conference on Theory of Information Retrieval,
page 39–50, 2023.
[14] Donna Harman. Overview of the fourth Text REtrieval Conference (TREC-4). In
Proceedings of the Fourth Text REtrieval Conference (TREC-4), volume 500-236 of
NIST Special Publication, pages 1–24, Gaithersburg, Maryland, 1995. National Institute
of Standards and Technology.
[15] Tz-Huan Hsu, Jheng-Hong Yang, and Jimmy Lin. Rethinking agentic search with pi-
serini: Is lexical retrieval sufficient?, 2026. URLhttps://arxiv.org/abs/2605.10848.
[16] Oz Huly, Idan Pogrebinsky, David Carmel, Oren Kurland, and Yoelle Maarek. Old IR
methods meet RAG. InProceedings of the 47th International ACM SIGIR Conference
on Research and Development in Information Retrieval, page 2559–2563, 2024.
9

[17] Vladimir Karpukhin, Barlas Oguz, Sewon Min, Patrick Lewis, Ledell Wu, Sergey
Edunov, Danqi Chen, and Wen-tau Yih. Dense passage retrieval for open-domain ques-
tion answering. InProceedings of the 2020 Conference on Empirical Methods in Natural
Language Processing (EMNLP), pages 6769–6781. Association for Computational Lin-
guistics, 2020.
[18] Omar Khattab and Matei Zaharia. ColBERT: Efficient and effective passage search
via contextualized late interaction over BERT. InProceedings of the 43rd International
ACM SIGIR Conference on Research and Development in Information Retrieval, pages
39–48, 2020.
[19] Zhuofeng Li, Haoxiang Zhang, Cong Wei, Pan Lu, Ping Nie, Yi Lu, Yuyang Bai, Shang-
bin Feng, Hangxiao Zhu, Ming Zhong, Yuyu Zhang, Jianwen Xie, Yejin Choi, James
Zou, Jiawei Han, Wenhu Chen, Jimmy Lin, Dongfu Jiang, and Yu Zhang. Beyond se-
mantic similarity: Rethinking retrieval for agentic search via direct corpus interaction,
2026. URLhttps://arxiv.org/abs/2605.05242.
[20] Ronak Pradeep, Sahel Sharifymoghaddam, and Jimmy Lin. RankVicuna: Zero-shot
listwise document reranking with open-source large language models, 2023. URLhttps:
//arxiv.org/abs/2309.15088.
[21] Ronak Pradeep, Sahel Sharifymoghaddam, and Jimmy Lin. RankZephyr: Effective and
robust zero-shot listwise reranking is a breeze!, 2023. URLhttps://arxiv.org/abs/
2312.02724.
[22] Stephen E. Robertson, Steve Walker, Susan Jones, Micheline Hancock-Beaulieu, and
Mike Gatford. Okapi at TREC-3. InProceedings of the Third Text REtrieval Conference
(TREC-3), volume 500-225 ofNIST Special Publication, pages 109–126, Gaithersburg,
Maryland, 1994. National Institute of Standards and Technology.
[23] G. Salton. A blueprint for automatic Boolean query processing.SIGIR Forum, 17(2):
6–24, September 1982.
[24] Gerard Salton, Edward A. Fox, and Harry Wu. Extended Boolean information retrieval.
Communications of the ACM, 26(11):1022–1036, November 1983.
[25] Erik Selberg and Oren Etzioni. Multi-service search and comparison using the
MetaCrawler. InProceedings of the Fourth International World Wide Web Conference,
Boston, Massachusetts, December 1995.
[26] Sahil Sen, Akhil Kasturi, Elias Lumer, Anmol Gulati, and Vamse Kumar Subbiah. Is
grep all you need? How agent harnesses reshape agentic search, 2026. URLhttps:
//arxiv.org/abs/2605.15184.
[27] Moritz Staudinger, Wojciech Kusa, Florina Piroi, Aldo Lipani, and Allan Hanbury. A
reproducibility and generalizability study of large language models for query generation.
10

InProceedings of the 2024 Annual International ACM SIGIR Conference on Research
and Development in Information Retrieval in the Asia Pacific Region, 2024.
[28] Weiwei Sun, Lingyong Yan, Xinyu Ma, Shuaiqiang Wang, Pengjie Ren, Zhumin Chen,
Dawei Yin, and Zhaochun Ren. Is ChatGPT good at search? Investigating large lan-
guage models as re-ranking agents. InProceedings of the 2023 Conference on Empirical
Methods in Natural Language Processing. Association for Computational Linguistics,
2023.
[29] Shivani Upadhyay, Ronak Pradeep, Nandan Thakur, Daniel Campos, Nick Craswell, Ian
Soboroff, Hoa Trang Dang, and Jimmy Lin. A large-scale study of relevance assessments
with large language models: An initial look, 2024. URLhttps://arxiv.org/abs/
2411.08275.
[30] Shivani Upadhyay, Ronak Pradeep, Nandan Thakur, Nick Craswell, and Jimmy Lin.
UMBRELA: UMbrela is the (Open-Source Reproduction of the) Bing RELevance As-
sessor, 2024. URLhttps://arxiv.org/abs/2406.06519.
[31] Ellen M. Voorhees. Variations in relevance judgments and the measurement of retrieval
effectiveness. InProceedings of the 21st Annual International ACM SIGIR Conference
on Research and Development in Information Retrieval, SIGIR ’98, page 315–323, 1998.
[32] Shuai Wang, Harrisen Scells, Bevan Koopman, and Guido Zuccon. Can ChatGPT
write a good Boolean query for systematic review literature search? InProceedings
of the 46th International ACM SIGIR Conference on Research and Development in
Information Retrieval, page 1426–1436, 2023.
[33] Shuai Wang, Harrisen Scells, Bevan Koopman, and Guido Zuccon. Reassessing large
language model boolean query generation for systematic reviews, 2025. URLhttps:
//arxiv.org/abs/2505.07155.
[34] Ian H. Witten, Alistair Moffat, and Timothy C. Bell.Managing Gigabytes: Compressing
and Indexing Documents and Images. Van Nostrand Reinhold, New York, first edition,
1994.
[35] Lee Xiong, Chenyan Xiong, Ye Li, Kwok-Fung Tang, Jialin Liu, Paul N. Bennett, Ju-
naid Ahmed, and Arnold Overwijk. Approximate nearest neighbor negative contrastive
learning for dense text retrieval. InProceedings of the 9th International Conference on
Learning Representations (ICLR), 2021.
[36] Zeyu Yang, Qi Ma, Jason Chen, and Anshumali Shrivastava. Superintelligent retrieval
agent: The next frontier of agentic retrieval, 2026. URLhttps://arxiv.org/abs/
2605.06647.
11