# Tool-calling retrieval versus vector RAG for a small Greek--English knowledge base: accuracy and robustness to how users type Greek

**Authors**: Nikolaos D. Tantaroudas, Ilias Karachalios, Andrew J. McCracken

**Published**: 2026-10-06 11:58:55

**PDF URL**: [https://arxiv.org/pdf/2610.08205v1](https://arxiv.org/pdf/2610.08205v1)

## Abstract
Assistants grounded in a small, frequently edited knowledge base can retrieve through tool calls to a live data interface or through vector retrieval-augmented generation (RAG). We compare the two on KyGround, a benchmark of 198 questions drawn from the published records of a Greek--English agricultural platform on Kythera, Greece, with answers verified automatically against the records and each question posed in up to nine forms, including Greek without accents, in capitals and in three Latin-script (Greeklish) schemes. With Claude Haiku 4.5 as router and answer model, a reconstruction of the platform's tool agent answered 71.6\% of canonical Greek questions correctly and vector RAG 95.3\% (difference $-23.6$ percentage points, 95\% CI $-33.1$ to $-15.1$). Letting the router write the vector query changed nothing, and placing the whole knowledge base of about 26,000 tokens in the prompt reached 99.3\%. The tool agent's losses arose in retrieval. Its literal searches returned nothing when the router's arguments did not occur verbatim in a record, for example when it transliterated Greek into Latin script or combined words that occur in a record but not as one phrase, and the agent then abstained. Unaccented and capitalised questions cost the tool agent about 20 points and vector RAG at most 2; accent-insensitive search removed this loss, and matching stemmed tokens raised the tool agent to 83.8\% on canonical Greek. Greeklish cost both designs about 21 to 32 points. Tool interfaces for community knowledge bases need search that tolerates how users type.

## Full Text


<!-- PDF content starts -->

Tool-calling retrieval versus vector RAG for a small
Greek–English knowledge base: accuracy and robustness to how
users type Greek
Nikolaos D. Tantaroudas∗1, Ilias Karachalios2, and Andrew J. McCracken3
1Institute of Communication and Computer Systems (ICCS), National Technical
University of Athens, 9 Iroon Polytechniou Str., 15773 Zografou, Athens, Greece
2University of Thessaly, 413 34 Larisa, Greece
3DASKALOS-APPS, 183 Rue de l’Abbé Griffon, 01960, Péronnas, France
Abstract
Assistants grounded in a small, frequently edited knowledge base can retrieve through
tool calls to a live data interface or through vector retrieval-augmented generation (RAG).
We compare the two on KyGround, a benchmark of 198 questions drawn from the published
records of a Greek–English agricultural platform on Kythera, Greece, with answers verified
automatically against the records and each question posed in up to nine forms, including Greek
without accents, in capitals and in three Latin-script (Greeklish) schemes. With Claude Haiku
4.5 as router and answer model, a reconstruction of the platform’s tool agent answered 71.6%
of canonical Greek questions correctly and vector RAG 95.3% (difference −23.6percentage
points, 95% CI −33.1to−15.1). Letting the router write the vector query changed nothing,
and placing the whole knowledge base of about 26,000 tokens in the prompt reached 99.3%.
The tool agent’s losses arose in retrieval. Its literal searches returned nothing when the
router’s arguments did not occur verbatim in a record, for example when it transliterated
Greek into Latin script or combined words that occur in a record but not as one phrase, and
the agent then abstained. Unaccented and capitalised questions cost the tool agent about 20
points and vector RAG at most 2; accent-insensitive search removed this loss, and matching
stemmed tokens raised the tool agent to 83.8% on canonical Greek. Greeklish cost both
designs about 21 to 32 points. Tool interfaces for community knowledge bases need search
that tolerates how users type.
Keywords.Retrieval-Augmented Generation, Tool Calling, Model Context Protocol, Greek,
Greeklish, Multilingual Robustness, Agricultural Advisory Services, Benchmark
This paper reports part of the KyGround study; the full report, with all methods and results, has
been submitted to Open Research Europe.
1 Introduction
Smallholder farmers on remote islands need advice that is local. Which varieties grow on the
island, when the harvest starts there, which producer sells what. Large language models (LLMs)
can widen access to such advice, provided that their answers are anchored in authoritative local
information rather than in the model’s parametric memory [Tzachor et al., 2023, High et al.,
2026], and deployed agricultural assistants therefore combine LLMs with retrieval over curated
content [Singh et al., 2024, Collis et al., 2025]. Most of the producers on Kythera and Antikythera
∗Corresponding author,nikolaos.tantaroudas@iccs.gr
1
arXiv:2610.08205v1  [cs.CE]  6 Oct 2026

who tried the deployed assistant whose design is studied here found it helpful, but fewer than a
quarter would trust its answers without human verification [Karachalios et al., 2026].
There are two common ways of grounding an assistant in a knowledge base. Vector retrieval-
augmented generation (RAG) splits documents into chunks, embeds and indexes them, and
inserts the chunks nearest to the question into the prompt [Lewis et al., 2020, Karpukhin et al.,
2020]. Tool-calling retrieval lets the model choose among functions that query a live data
interface and write their arguments [Yao et al., 2023, Schick et al., 2023], and the Model Context
Protocol (MCP) standardises how such tools are offered to LLM applications [Anthropic and MCP
Contributors, 2025]. The assistantFalco eleonorae, deployed for the farmers and cooperatives of
Kythera, Greece, uses tool calls. Its system article [Tantaroudas et al., 2026a], of which an earlier
version is available as a preprint [Tantaroudas et al., 2026b], argues that for a small, curated and
frequently edited knowledge base, exact queries to a read-only data interface are more faithful
than nearest-neighbour search. The counter-argument concerns how users type. A tool agent
depends on a router that picks the right tool and writes arguments that the interface’s matching
accepts, which is fragile when Greek is typed without accents, in capitals or in Latin characters
(“Greeklish”) [Toumazatos et al., 2024, Luo et al., 2026], whereas multilingual embeddings, which
place translations and paraphrases close together [Chen et al., 2024], may absorb such variation.
This paper tests the accuracy claim, under a pre-specified protocol and with one model
throughout. Its contributions are:
•KyGround, a benchmark of 198 questions drawn from the platform’s published records,
with gold answers verified automatically against the records, each question posed in up to
nine surface forms and scored by a deterministic, Greek-aware scorer;
•a comparison of six conditions that share one router and answer model, including two
controls that locate the difference: vector RAG queried through the same router model and
function-calling mechanism as the tool agent, and the whole knowledge base in the prompt;
•an ablation of the data interface’s free-text search normalisation, which shows how much of
the tool agent’s deficit can be engineered away.
2 Related work
Retrieval and long contexts.RAG couples a generator with a retriever over a non-parametric
memory [Lewis et al., 2020], usually dense passage retrieval [Karpukhin et al., 2020]; Wu et al.
[2026] review its components and variants. Multilingual embedding models, such as the M3-
Embedding model used here [Chen et al., 2024], place translations and paraphrases close together.
When a corpus fits in the context window, long-context prompting outperformed RAG on average
when resources allowed [Li et al., 2024], although models use information in the middle of a long
context less reliably [Liu et al., 2024].
Tool calling.ReAct interleaves reasoning with actions such as API calls [Yao et al., 2023],
Toolformer learns when and how to call tools [Schick et al., 2023], and the Berkeley Function
Calling Leaderboard evaluates function calling from single calls to agentic use [Patil et al., 2025].
Because the router selects tools from their descriptions, description quality matters [Hasan et al.,
2026], and with multilingual input, models often fill tool parameters with values in the user’s
language, which breaks interfaces that expect a fixed one [Luo et al., 2026]. Generating executable
queries rather than retrieving passages suits structured information [Hontan et al., 2026].
Greek and Greeklish.Greek users often type without accents, in capitals or in Latin
characters. Greeklish has no single mapping, so one word can be written in many ways, and even
native speakers can find it hard to read [Toumazatos et al., 2024]. The formal transliteration
is ISO 843 [ISO, 1997], but informal phonetic and visual schemes dominate in messages. Open
Greek LLMs [Voukoutis et al., 2024, Roussis et al., 2025] and Greek retrieval benchmarks [Beta
2

et al., 2026] are emerging, but we know of no head-to-head comparison of tool-calling and vector
retrieval in Greek, with the orthographic variation that Greek users produce.
3 Methods
3.1 Knowledge base and benchmark
The DASK platform serves the Kythera pilot of the PoliRURAL Plus project [PoliRURAL Plus
Consortium, 2024]. Its editors maintain bilingual, Greek-primary records of crops, seasonal-
calendar entries, glossary entries (including dialect terms), products, agritourism experiences,
training modules, articles and frequently asked questions, and a read-only interface serves the
published records with server-side editorial gating [Tantaroudas et al., 2026a]. The knowledge
base is a frozen snapshot of the platform’s public pages including 103 published records, or 78,088
characters (about 26,000 tokens) when serialised bilingually. Every condition answered from
in-memory copies of it.
KyGround release contains 198 items including 121 generated from 19 bilingual templates
over typed fields (for example the harvest period of a crop or the duration of an experience) and
77 written individually, including aggregations across records. Fourteen items ask for information
that the knowledge base does not contain, and twelve for a price that is shown only on request.
Each answer is a set of typed facts (text, list members, numbers, ranges, durations, months, price
on request, abstention), and an answer is correct when it contains every required fact. Every item
was verified automatically against the snapshot, without a model in the loop. Its gold records
exist and are published, template items regenerate exactly, every gold fact and every quoted
piece of evidence occurs in the records, and aggregates recompute to the same records. Items
about the same record share a cluster. A development split of 40 items, made of whole groups of
connected records, was used only to tune vector RAG, and the primary analysis set consists of
the 148 answerable test items in 79 clusters. Each item was posed in canonical Greek, English,
Greek without accents, Greek in capitals, three Greeklish schemes (a transcription in the style
of ELOT 743 and ISO 843, an informal phonetic scheme and an informal visual scheme that
uses digits and look-alike letters), other inflection and colloquial or regional wording (Table 1).
Other-inflection paraphrases exist for 8 items and colloquial or regional variants for 75, so the
release contains 1,469 queries.
Table 1. Surface forms of one benchmark item, the harvest period of the olive on Kythera (gold
answer: November to January).
Form Query
Canonical GreekΠοια είναι η περίοδος συγκομιδής για την καλλιέργεια«Ελιά»στα Κύθηρα;
English When is the harvest period for “Olive” on Kythera?
Greek, no accentsΠοια ειναι η περιοδος συγκομιδης για την καλλιεργεια«Ελια»στα Κυθηρα;
Greek, capitalsΠΟΙΑ ΕΙΝΑΙ Η ΠΕΡΙΟΔΟΣ ΣΥΓΚΟΜΙΔΗΣ ΓΙΑ ΤΗΝ ΚΑΛΛΙΕΡΓΕΙΑ
"ΕΛΙΑ"ΣΤΑ ΚΥΘΗΡΑ;
Greeklish, ELOT/ISO style Poia einai i periodos sygkomidis gia tin kalliergeia "Elia" sta Kythira;
Greeklish, phonetic pia ine i periodos sigkomidis gia tin kalliergia "elia" sta kithira;
Greeklish, visual poia einai h periodos sugkomidhs gia thn kalliergeia "elia" sta ku8hra;
Greek, other inflectionΠότε μαζεύονται οι ελιές στα Κύθηρα;
Greek, colloquialΠοια είναι η περίοδος συγκομιδής για την καλλιέργεια«Ελιά»στο Τσιρίγο;
3.2 Conditions
All six conditions (Table 2) use Claude Haiku 4.5 [Anthropic, 2025] as router and answer model
and share the base prompt and the knowledge base. C2 reconstructs the deployed tool agent.
3

The router selects up to two of nine tools (knowledge search, crop information, seasonal calendar,
glossary, products, experiences, cooperatives, training modules and lesson content) and fills their
arguments, and an in-process replica of the data interface executes them with its gating, Greek
fallback and matching. The interface’s free-text matching was characterised before the runs with
50 probes of the platform’s public training-module search. It ignored case and matched within
words, but it was sensitive to accents, word order and inflection and did not match Greeklish,
and the replica was set to the same behaviour (phrase matching with case folding). C1 embeds
bilingually serialised chunks of at most 800 characters with BGE-M3 [Chen et al., 2024] and
inserts the eight nearest to the question, settings chosen on the development split by the share of
gold records retrieved, without model calls. C1q differs from C1 only in that the router model
writes the search query, through a single search tool with the same name and query description as
the knowledge search of C2, so C1q and C2 share the model, the prompt and the function-calling
mechanism and differ in what the call retrieves. C3 gives the answer model both the tool results
of C2 and the chunks of C1. C4 places the whole knowledge base in the prompt and shows what
the answer model achieves when no evidence can be missed, and C0 answers without retrieval. In
an ablation, C2 answered all Greek and Greeklish forms with the replica’s free-text search set
to (a) phrase matching with case folding, the platform’s behaviour, (b) phrase matching with
accents also removed and (c) matching of all stemmed tokens.
Table 2. Conditions. All use Claude Haiku 4.5 as router and answer model and share the base
prompt, the conversation-history policy and the knowledge-base snapshot.
Code Condition Retrieval before the answer
C0 Closed book none
C1 Vector RAG the 8 chunks (at most 800 characters) of the bilingually serialised
published records nearest to the question (BGE-M3 embeddings,
exact cosine search)
C1q Vector RAG,
router-formulated queryas C1, for the query that the router model writes when calling a
single search tool
C2 Tool retrieval (deployed
design, reconstructed)the router selects up to two of nine tools and fills their arguments.
A replica of the data interface executes them
C3 Hybrid the tool results of C2 and the chunks of C1
C4 Full context (reference) the whole published knowledge base in the system prompt
3.3 Generation
No model was called through an API. Replies were generated by sub-agents of a Claude agent
session run on the authors’ subscription, each in its own context window [Anthropic, n.d.]. Every
request was keyed by the hash of its content and looked up in a cache, and unanswered requests
were exported in batches, answered by Claude Haiku 4.5 sub-agents and validated. Check codes
placed every 100 lines of a batch, introduced after two early batches of full-context (C4) requests
were found to have been read only in part, verified that a sub-agent had read its whole batch.
The earlier batches of other requests, whose replies had each been matched to their request, were
kept. Because a sub-agent sees every request of its batch, all replies were audited for facts or
records that occurred in another request of the batch but not in their own. The 99 flagged replies
of the whole study (1.1% of 9,079 requests) were generated again in isolation, one request per
sub-agent, and the results below use these replies. The audit, the withdrawal rule and an isolation
control were added after the results had been seen. The withdrawal rule, which covered every
flagged reply whatever its grade, was fixed before any regenerated reply was seen. The protocol,
hypotheses and analysis plan were fixed, and their digests recorded, before any run with a real
model. As the protocol was not deposited in a registry, the study is pre-specified rather than
pre-registered.
4

3.4 Scoring and statistics
A deterministic scorer normalises answers and gold facts (Unicode normalisation, case folding,
removal of accents and folding of the final sigma), matches them with light stemming and
across scripts, and parses numbers, Greek and English month names and ranges, and durations.
Abstention and “price on request” are recognised by bilingual cue lists. Claude Sonnet 5.5
[Anthropic, 2026], a larger model of the same family, served as a second scorer [Zheng et al.,
2023]. It graded all canonical Greek test answers and a sample of 300 English ones, blind to
the condition. The deterministic scorer also graded a controlled suite of 1,738 answers whose
correctness is known by construction, and the judge a random sample of 350 of them. Accuracies
carry 95% cluster-bootstrap intervals with 10,000 resamples [Efron and Tibshirani, 1993]. The
pre-specified hypotheses compare the canonical Greek accuracy of C2 with that of C1 (H1),
C1q (H2) and C0 (H3), using a two-sided, cluster-level sign-flip permutation test with Holm’s
correction [Holm, 1979]. H1 and H2 are supported if the adjusted p < 0.05, whatever the direction
of the difference, and H3 if, in addition, C2 is the more accurate. A robustness gap is the paired
difference between a form and canonical Greek, and the difference between the gaps of C2 and C1
is a paired difference-in-differences.
4 Results
4.1 Accuracy on canonical questions
Table 3. Accuracy on canonical Greek and English questions, % of the 148 answerable test
questions answered fully correctly (95% cluster-bootstrap CI; 79 clusters).
Condition Greek English
C2 tool retrieval 71.6 (62.2–79.9) 79.1 (70.1–86.5)
C1 vector RAG 95.3 (91.3–98.3) 94.6 (90.6–97.9)
C1q vector RAG, router query 96.6 (93.2–99.3) 93.9 (89.9–97.3)
C3 hybrid 96.6 (93.2–99.3) 95.3 (91.6–98.4)
C4 full context 99.3 (97.7–100.0) 97.3 (94.6–99.4)
C0 closed book 12.2 (7.3–17.8) 10.1 (5.6–15.4)
On canonical Greek questions, C2 answered 71.6% correctly and C1 95.3% (Table 3; Figure 1).
The paired difference was −23.6percentage points (pp; 95% CI −33.1to−15.1; Holm-adjusted
p < 0.001), with 37 questions answered correctly by C1 alone and 2 by C2 alone. C2 also trailed
C1q by 25.0 pp ( −34.4to−16.6) and exceeded C0 by 59.5 pp (+48 .5to+69 .1), so all three
hypotheses were supported, H1 and H2 in the direction that favours vector retrieval. Letting
the router write the vector query did not change accuracy (C1q −C1:+1 .4pp;−1.8to+4 .8;
p= 0.688), so the gap lies in what the tool calls retrieve rather than in letting a model restate
the question. Adding the chunks of C1 to the tool results of C2 (C3) gave 96.6%, and the whole
knowledge base in the prompt (C4) gave 99.3%. In English, C2 answered 79.1% and C1 94.6%.
4.2 Where the tool agent fails
The gold record was among the retrieved records for 75.7% of the answers of C2 to canonical
Greek and English questions, against 98.6% for C1 (Figure 2). Wrong answers despite retrieved
evidence were rare in every condition with retrieval (1.7% to 4.1% of answers). The shortfall
of C2 consisted mainly of abstentions (19.3% of its answers), in which the agent said that the
information was not available after its tool calls had not returned the relevant record. C2 made
1.38 tool calls per question, and in 23.3% of its answers at least one call returned no record.
In an exploratory analysis, its accuracy depended on whether the question quoted a record’s
5

0 25% 50% 75% 100%
Answers fully correct (95% cluster-bootstrap CI)Greek
English
Greek, no accents
Greek, capitals
Greeklish (ELOT)
Greeklish (phonetic)
Greeklish (visual)
Greek, other in flection
Greek, colloquial
C2 tool retrieval
C1 vector RAGC1q vector RAG, router query
C3 hybridC4 full context
C0 closed bookFigure 1. Accuracy by condition and surface form: proportion of answerable test questions
answered fully correctly, with 95% cluster-bootstrap intervals. C1q, C3, C4 and C0 were run
on canonical Greek and English only. Conditions are distinguished by marker shape as well as
colour; the numbers are in Tables 3 and 4.
6

exact title. It answered 81.9% of the 105 canonical Greek questions that did and 46.5% of the
43 that did not, against 97.1% and 90.7% for C1. The router’s arguments show why searches
failed (an exploratory analysis). Although the search tool asks for terms “preferably in Greek”,
the router wrote 28.7% of its free-text arguments for canonical Greek questions in Latin script, as
transliterations (for example “Thymari meli Kythiron 1kg” for «Θυμαρίσιο Μέλι Κυθήρων1kg»)
or English keywords, and 46.9% of these returned no record, against 26.2% of the arguments
written in Greek.
0 25% 50% 75% 100%
Share of answers (Greek and English canonical questions)C2 tool retrieval
C1 vector RAG
C1q vector RAG, router query
C3 hybrid
C4 full context
C0 closed book75% 19%
95%
95%
96%
98%
11% 7% 81%
Correct
Wrong, evidence was retrievedWrong, evidence not retrieved
Abstained
Figure 2. Decomposition of the answers to canonical Greek and English questions (296 per
condition): correct; wrong although the gold record had been retrieved; wrong without the gold
record among the retrieved records; and abstentions. C0 does not retrieve anything, so its wrong
answers count as wrong without retrieved evidence.
4.3 Robustness to Greek surface forms
Table 4. Accuracy of the tool agent (C2) and vector RAG (C1) by surface form, % (95% cluster-
bootstrap CI), and the difference between their gaps relative to canonical Greek, percentage points
(paired difference-in-differences, 95% CI). Other inflection has 7 test questions and colloquial
wording 58; every other form has 148.
Form C2 tool retrieval C1 vector RAG Gap difference, C2−C1
Greek 71.6 (62.2–79.9) 95.3 (91.3–98.3) reference
English 79.1 (70.1–86.5) 94.6 (90.6–97.9) +8.1 (−0.7 to +16.7)
Greek, no accents 48.0 (37.5–59.2) 94.6 (90.5–97.9)−23.0 (−35.1 to−10.6)
Greek, capitals 52.0 (41.8–62.8) 93.2 (88.4–97.0)−17.6 (−27.9 to−6.0)
Greeklish (ELOT) 40.5 (30.3–51.4) 74.3 (66.0–81.9)−10.1 (−24.6 to +5.3)
Greeklish (phonetic) 39.9 (29.2–51.5) 72.3 (63.4–80.5)−8.8 (−24.5 to +7.9)
Greeklish (visual) 40.5 (31.7–50.0) 68.2 (57.6–78.2)−4.1 (−18.3 to +11.1)
Greek, other inflection 57.1 (14.3–85.7) 85.7 (57.1–100.0) +0.0 (−42.9 to +42.9)
Greek, colloquial 69.0 (54.5–81.2) 89.7 (80.4–96.7) +12.1 (+0.0 to +25.4)
Relative to canonical Greek, unaccented questions lowered the accuracy of C2 by 23.6 pp and
that of C1 by 0.7 pp, and capitals by 19.6 and 2.0 pp; the differences between the two systems’
gaps were −23.0pp (−35.1to−10.6) and −17.6pp (−27.9to−6.0) (Table 4). Greeklish lowered
the accuracy of both systems, by 31.1 to 31.8 pp for C2 and by 20.9 to 27.0 pp for C1, leaving
7

C1 at 68.2% to 74.3% and C2 at 39.9% to 40.5%, and the differences between these gaps could
not be statistically distinguished. With Greeklish questions the router wrote 61% to 63% of its
free-text arguments in Latin script, and 74% to 80% of these returned no record. Colloquial
and regional wording changed the accuracy of C2 by+5 .2pp and that of C1 by −6.9pp; the
other-inflection paraphrases (7 test items) are too few for inference.
4.4 Search normalisation
Table 5. Accuracy of the tool agent (C2) under three free-text search normalisations of the data
interface, % (95% cluster-bootstrap CI): (a) phrase matching with case folding, the platform’s
behaviour and the main run; (b) phrase matching with accents also removed; (c) matching of all
stemmed tokens.
Form (a) Phrase, case folded (b) Accents also folded (c) Stemmed tokens
Greek 71.6 (62.2–79.9) 73.6 (64.4–81.9) 83.8 (77.6–89.2)
Greek, no accents 48.0 (37.5–59.2) 73.6 (64.5–81.8) 83.8 (77.8–89.5)
Greek, capitals 52.0 (41.8–62.8) 73.0 (64.4–81.2) 81.8 (75.0–88.4)
Greeklish (ELOT) 40.5 (30.3–51.4) 43.2 (32.7–54.4) 51.4 (41.0–62.2)
Greeklish (phonetic) 39.9 (29.2–51.5) 43.9 (33.1–55.3) 49.3 (38.6–60.5)
Greeklish (visual) 40.5 (31.7–50.0) 44.6 (35.6–54.0) 52.0 (43.6–60.8)
Greek, other
inflection57.1 (14.3–85.7) 57.1 (14.3–85.7) 71.4 (42.9–100.0)
Greek, colloquial 69.0 (54.5–81.2) 69.0 (54.5–81.2) 74.1 (61.1–85.1)
Folding accents in the interface’s free-text search (mode b) removed the losses of C2 on
unaccented and capitalised Greek and raised its canonical Greek accuracy from 71.6% to 73.6%
(Table 5). Matching all stemmed tokens (mode c) raised it to 83.8%, and on the questions that
do not quote a record’s title from 46.5% to 72.1%. Neither mode helped much with Greeklish
(43.2% to 52.0%), because transliterated arguments match neither the Greek nor the English text
of the records. Even with stemmed tokens, C2 remained 11.5 pp below C1 on canonical Greek.
4.5 Unanswerable questions, prices and context size
No condition gave an amount for a price shown only on request (0 of 20 answers each). On
unanswerable items, the deterministic scorer counted at most 2 of 20 answers per condition as
hallucinations, but all 9 such answers were abstentions in wordings that its cue lists did not cover.
Per question, C1 used one model call and 1,612 input tokens on average, C1q two calls and 2,069
tokens, C2 2.98 calls and 6,517 tokens, and C4 one call and 26,294 tokens, estimated with the
o200k_base encoding [OpenAI, n.d.].
4.6 Validation of scoring and of batched generation
The deterministic scorer graded all 1,738 answers of the controlled suite as constructed, as did
the LLM judge on the 350 that it graded. On 1,248 judged test answers, the two scorers agreed
on 97.4% ( κ= 0.92; Cohen, 1960), and with the judge’s grades the paired differences of C2 from
C1, C1q and C0 were −25.7,−25.7and+60 .1pp, leading to the same conclusions. Of the 76
accuracy answers that the contamination audit flagged, 57 answered Greeklish questions, for
which retrieval most often failed, and 53 had been scored correct. Generated again in isolation, 9
were correct. The correction changed the primary contrasts by less than one point but lowered the
Greeklish accuracy of C1 by 5.4 to 8.1 points: with the batched replies, the differences between
the Greeklish gaps of C2 and C1 would have been −14.9,−16.2and−9.5pp, instead of −10.1,
−8.8and−4.1pp. In a control, 95 of 100 unflagged replies generated again in isolation received
8

the same grade, 4 of the 5 discordant replies favoured the batched version, so a residual inflation
of a few points among unflagged replies cannot be excluded.
5 Discussion
With one router and answer model, vector RAG answered questions about a small, community-
edited knowledge base more accurately than tool retrieval, and the controls locate the difference.
C1q gave vector retrieval the same router model, prompt and function-calling mechanism as C2
and changed nothing, and C4 shows that the answer model answers almost every question when
the evidence is present, so retrieval was the bottleneck. The tool agent’s literal search returned
nothing when the router’s arguments did not occur verbatim in a record (a transliterated or
translated keyword, an accent added to a title stored in capitals, or keywords that occur in a
record but not as one phrase), and the agent then abstained. Multilingual input is known to
make models fill tool parameters in an unexpected language [Luo et al., 2026]. Here the router
transliterated Greek into Latin script even though the questions and the records were Greek.
Three lessons follow for tool interfaces over small community knowledge bases. First, free-text
search should ignore case, accents and the final sigma and match stemmed tokens rather than
literal phrases. In the replica, this removed the whole loss on unaccented and capitalised questions
and recovered about half of the gap to vector RAG on canonical Greek. Second, arguments should
be in the script of the records, and the tool description alone did not ensure it. The search tool
asked for terms preferably in Greek, yet the router wrote 28.7% of its free-text arguments for
canonical Greek questions in Latin script. Third, Greeklish remains hard for both designs. The
vector RAG reached 68.2% to 74.3% on Greeklish questions, 21 to 27 points below its canonical
accuracy, against about 40% for the tool agent, so a vector fallback would raise the tool agent’s
Greeklish accuracy without solving Greeklish, and transliteration-aware matching remains to
be tested. On canonical questions, adding vector chunks to the tool results (C3) matched the
accuracy of vector RAG. For a knowledge base of about 26,000 tokens, placing everything in the
prompt was the most accurate design, at about four times the input of the tool agent, and the
risk of overlooking evidence in the middle of a long context [Liu et al., 2024] did not materialise
at this size.
Limitations.The authors built the system evaluated as C2 and the platform from which
the benchmark was drawn. To limit bias, the protocol was fixed before the runs, the primary
scoring is deterministic and all outputs are available on request, and on accuracy the results run
against the system article’s claim. Claude Haiku 4.5 replaced the deployed GPT-5-family models
[OpenAI, 2025], so the accuracies estimate what the designs achieve with this model rather than
the deployed system’s accuracy, and another router may transliterate more or less often. C2 is a
reconstruction. Seven of the nine tool names, all nine tool descriptions and the full system prompt
were reconstructed. Its router saw only the nine local tools, whereas the production router also
chooses among the upstream agent’s general tools, and the replica’s agreement with the deployed
interface could not be measured. The judge is from the same model family as the answer model,
so a shared preference for the family’s phrasing cannot be excluded, serving only as a second
scorer. The questions were derived from the records rather than collected from users, and 105
of the 148 primary questions quote a record’s exact title, which favours literal matching, so the
tool agent’s accuracy on users’ own wording may be lower than reported. The gold answers were
verified against the records rather than by the platform’s editors. The knowledge base is small
and comes from one platform. Batched generation contaminated 1.1% of replies, and the audit
detects only borrowed facts, records and markers. Generation in a session cannot be repeated
token for token, each query was answered once, and the study can detect differences of about 14
points with 80% power, so the smaller differences between the Greeklish gaps are reported as
estimates.
9

6 Conclusions
ForasmallGreek–Englishknowledgebase, vectorRAGanswered95%ofcanonicalGreekquestions
correctly and tool retrieval 72%. The gap arose in literal search, widened with unaccented and
capitalised Greek and was partly closed by accent-insensitive, stemmed matching in the data
interface, while Greeklish remained hard for both designs. Tool interfaces should be evaluated in
the spellings that users actually type.
Data and code availability
The KyGround benchmark, with the names of natural persons replaced by role labels, the
run outputs and the evaluation harness are available from the authors on request. The aggre-
gated results behind the tables and figures of this paper are provided as ancillary files. The
frozen snapshot of the platform’s records, the request cache and the batch files given to the
sub-agents contain the text of the records, including names of natural persons, and are not
deposited. They are available on request from the corresponding author, Nikolaos D. Tantaroudas
(nikolaos.tantaroudas@iccs.gr ), to researchers and reviewers who state their name, affiliation
and purpose and agree in writing to use the material only to verify or reproduce the results or for
other non-commercial research, not to redistribute it, not to attempt to identify or contact the
persons named in it, to delete it when their purpose is complete and to cite this work and the
datasets. Requests are answered within 30 days. The platform’s published records can be viewed
athttps://dask-platform.onrender.com/.
Acknowledgements
This work was supported by the European Union’s Horizon Europe research and innovation
programme through the PoliRURAL Plus project (Grant Agreement No. 101136910), via its
DEVELOP cascade-funding open call. Views and opinions expressed are those of the authors only
and do not necessarily reflect those of the European Union or the granting authority. Neither the
European Union nor the granting authority can be held responsible for them. The authors thank
the editors and producers of the DASK platform, whose published records form the knowledge
base studied here, and the PoliRURAL Plus consortium for the Jackdaw agent infrastructure.
Large language models are the objects of this study. In addition, the evaluation software, the
individually written benchmark items and the phrase lexicon for colloquial wording, the protocol,
the analyses and a first draft of the text were prepared with the assistance of Claude Opus 5.5
(Anthropic), which also ran the harness and launched the sub-agents that generated and graded
the replies, under the authors’ direction. The authors reviewed and revised all code and text and
take full responsibility for the content. The authors designed and built the assistant evaluated as
C2. NDT is the founder and legal representative of DASKALOS APPS SAS, with which AJM is
affiliated.
References
Tzachor, A., Devare, M., Richards, C., Pypers, P., Ghosh, A., Koo, J., et al. (2023). Large language
models and agricultural extension services.Nat Food, 4(11), 941–948. doi:10.1038/s43016-023-
00867-x
High, C., Singh, N., and Nemes, G. (2026). Artificial intelligence for agricultural extension:
Supporting transformative learning among smallholder farmers.Journal of Development Policy
and Practice, 11(1), 61–80. doi:10.1177/24551333251345224
10

Singh, N., Wang’ombe, J., Okanga, N., Zelenska, T., Repishti, J., Jayasankar, G.K., et al. (2024).
Farmer.Chat: Scaling AI-powered agricultural services for smallholder farmers. arXiv preprint
arXiv:2409.08916. doi:10.48550/arXiv.2409.08916
Collis, S., Kinyua, F., Kumar, V., Lakougna, H., Merz, C., Pandey, K., et al. (2025). Building
AI-based advisory services for smallholder farmers: Technical learnings from the AIEP Initiative.
arXiv preprint arXiv:2601.11537. doi:10.48550/arXiv.2601.11537
Karachalios, I., Tantaroudas, N., and McCracken, A.J. (2026). Digital education and island
producers’readinesstoadoptdigitalplatforms: EvidencefromKytheraandAntikythera[version
1; peer review: awaiting peer review].Open Res Eur, 6, 351. doi:10.12688/openreseurope.24710.1
Lewis, P., Perez, E., Piktus, A., Petroni, F., Karpukhin, V., Goyal, N., et al. (2020). Retrieval-
augmented generation for knowledge-intensive NLP tasks. InAdvances in Neural Information
Processing Systems 33 (NeurIPS 2020), pp. 9459–9474. https://proceedings.neurips.cc/
paper/2020/hash/6b493230205f780e1bc26945df7481e5-Abstract.html
Karpukhin, V., Oğuz, B., Min, S., Lewis, P., Wu, L., Edunov, S., et al. (2020). Dense passage
retrieval for open-domain question answering. InProceedings of the 2020 Conference on
Empirical Methods in Natural Language Processing (EMNLP), Association for Computational
Linguistics, pp. 6769–6781. doi:10.18653/v1/2020.emnlp-main.550
Yao, S., Zhao, J., Yu, D., Du, N., Shafran, I., Narasimhan, K., et al. (2023). ReAct: Synergizing
reasoning and acting in language models. InThe Eleventh International Conference on Learning
Representations (ICLR 2023).https://openreview.net/forum?id=WE_vluYUL-X
Schick, T., Dwivedi-Yu, J., Dessi, R., Raileanu, R., Lomeli, M., Hambro, E., et al. (2023).
Toolformer: Language models can teach themselves to use tools. InAdvances in Neural
Information Processing Systems 36 (NeurIPS 2023), pp. 68539–68551. doi:10.52202/075280-
2997
Anthropic and Model Context Protocol Contributors (2025). Model Context Protocol specification
(revision 2025-11-25).https://modelcontextprotocol.io/specification/2025-11-25
Tantaroudas, N., Karachalios, I., and McCracken, A. (2026a). Retrieval-grounded multilingual
LLM assistance for island smallholder farmers [version 1; peer review: 1 approved, 2 approved
with reservations].Open Res Eur, 6, 262. doi:10.12688/openreseurope.24512.1
Tantaroudas, N.D., Karachalios, I., and McCracken, A.J. (2026b). Smallholder farmers on remote
islands receiving retrieval-grounded multilingual LLM assistance and agronomic advice. arXiv
preprint arXiv:2606.25647. doi:10.48550/arXiv.2606.25647
Toumazatos, A., Pavlopoulos, J., Androutsopoulos, I., and Vassos, S. (2024). Still all Greeklish to
me: GreeklishtoGreektransliteration. InProceedings of the 2024 Joint International Conference
on Computational Linguistics, Language Resources and Evaluation (LREC-COLING 2024),
ELRA and ICCL, pp. 15309–15319.https://aclanthology.org/2024.lrec-main.1330/
Luo, Z., Kutralingam, T.P., Okoani, O.N., Xu, W., Wei, H., and Hu, X. (2026). Lost in execution:
On the multilingual robustness of tool calling in large language models. InProceedings of
the 64th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long
Papers), Association for Computational Linguistics, pp. 44059–44077. doi:10.18653/v1/2026.acl-
long.2039
Chen, J., Xiao, S., Zhang, P., Luo, K., Lian, D., and Liu, Z. (2024). M3-Embedding: Multi-
linguality, multi-functionality, multi-granularity text embeddings through self-knowledge distil-
lation. InFindings of the Association for Computational Linguistics: ACL 2024, Association
for Computational Linguistics, pp. 2318–2335. doi:10.18653/v1/2024.findings-acl.137
11

Wu, S., Xiong, Y., Cui, Y., Wu, H., Chen, C., Yuan, Y., et al. (2026). Retrieval-augmented gener-
ation for natural language processing: a survey.Artif Intell Rev, 59(9), 192. doi:10.1007/s10462-
026-11605-7
Li, Z., Li, C., Zhang, M., Mei, Q., and Bendersky, M. (2024). Retrieval augmented generation or
long-context LLMs? A comprehensive study and hybrid approach. InProceedings of the 2024
Conference on Empirical Methods in Natural Language Processing: Industry Track, Association
for Computational Linguistics, pp. 881–893. doi:10.18653/v1/2024.emnlp-industry.66
Liu, N.F., Lin, K., Hewitt, J., Paranjape, A., Bevilacqua, M., Petroni, F., et al. (2024). Lost
in the middle: How language models use long contexts.Trans Assoc Comput Linguist, 12,
157–173. doi:10.1162/tacl_a_00638
Patil, S.G., Mao, H., Yan, F., Ji, C.C.J., Suresh, V., Stoica, I., et al. (2025). The Berkeley
Function Calling Leaderboard (BFCL): From tool use to agentic evaluation of large language
models. InProceedings of the 42nd International Conference on Machine Learning (ICML
2025), PMLR 267, pp. 48371–48392. https://proceedings.mlr.press/v267/patil25a.html
Hasan, M.M., Li, H., Rajbahadur, G.K., Adams, B., and Hassan, A.E. (2026). Model Context Pro-
tocol(MCP)tooldescriptionsaresmelly! TowardsimprovingAIagentefficiencywithaugmented
MCP tool descriptions. arXiv preprint arXiv:2602.14878. doi:10.48550/arXiv.2602.14878
Hontan, V.-M., Bunea, A.-A., Tantaroudas, N.D., andPopovici, D.-M.(2026). Queryingstructured
data through natural language using language models. arXiv preprint arXiv:2604.03057.
doi:10.48550/arXiv.2604.03057
International Organization for Standardization (1997). ISO 843:1997. Information and docu-
mentation — Conversion of Greek characters into Latin characters. Geneva: ISO. https:
//www.iso.org/standard/5215.html
Voukoutis, L., Roussis, D., Paraskevopoulos, G., Sofianopoulos, S., Prokopidis, P., Papavasileiou,
V., et al. (2024). Meltemi: The first open large language model for Greek. arXiv preprint
arXiv:2407.20743. doi:10.48550/arXiv.2407.20743
Roussis, D., Voukoutis, L., Paraskevopoulos, G., Sofianopoulos, S., Prokopidis, P., Papavassileiou,
V., et al. (2025). Krikri: Advancing open large language models for Greek. InFindings of
the Association for Computational Linguistics: EMNLP 2025, Association for Computational
Linguistics, pp. 5012–5033. doi:10.18653/v1/2025.findings-emnlp.268
Beta, E., Chlapanis, O.S., Galanis, D., and Androutsopoulos, I. (2026). GreekBar-
Retrieval: A benchmark for Greek statutory retrieval. arXiv preprint arXiv:2608.18752.
doi:10.48550/arXiv.2608.18752
PoliRURAL Plus Consortium (2024). PoliRURAL Plus: Building capacity for foresight-driven
rural innovation (Horizon Europe).https://poliruralplus.eu/
Anthropic (2025). System card: Claude Haiku 4.5. Accessed 5 October 2026. https://www.
anthropic.com/claude-haiku-4-5-system-card
Anthropic (n.d.). Create custom subagents. Claude Code documentation [Internet]. Accessed 5
October 2026.https://code.claude.com/docs/en/sub-agents
Anthropic (2026). System card: Claude Sonnet 5.5. Accessed 5 October 2026. https://www.
anthropic.com/claude-sonnet-5-5-system-card
12

Zheng, L., Chiang, W.L., Sheng, Y., Zhuang, S., Wu, Z., Zhuang, Y., et al. (2023). Judging
LLM-as-a-judge with MT-Bench and Chatbot Arena. InAdvances in Neural Information
Processing Systems 36 (NeurIPS 2023), Datasets and Benchmarks Track, pp. 46595–46623.
doi:10.52202/075280-2020
Efron, B. and Tibshirani, R.J. (1993). An introduction to the bootstrap. New York: Chapman &
Hall. doi:10.1007/978-1-4899-4541-9
Holm, S. (1979). A simple sequentially rejective multiple test procedure.Scand J Stat, 6(2),
65–70.https://www.jstor.org/stable/4615733
OpenAI (n.d.). tiktoken: a fast BPE tokeniser for use with OpenAI’s models. Software. GitHub.
Accessed 5 October 2026.https://github.com/openai/tiktoken
Cohen, J. (1960). A coefficient of agreement for nominal scales.Educ Psychol Meas, 20(1), 37–46.
doi:10.1177/001316446002000104
OpenAI (2025). GPT-5 system card. Accessed 5 October 2026. https://cdn.openai.com/
gpt-5-system-card.pdf
13